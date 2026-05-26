"""TwoState-LogNormal-Logit implementation of :class:`LaplaceObservationModel`.

Glues :mod:`scribe.laplace._newton_twostate_ln_logit` (per-cell Newton
kernel for the Poisson-Beta compound likelihood with a log-Normal
**additive** latent on the activation log-odds) into the protocol
expected by :func:`scribe.laplace._em.run_laplace_em`.

The structure mirrors :class:`TwoStateLNRateObservationModel` closely
— same per-cell Newton driver, same Woodbury machinery, same outer
EM/Adam loop on globals — with three substantive differences:

1. **Sampled globals**: ``(rate_loc, kappa_loc, eta_anchor_loc, W,
   d_loc)``.  ``rate_loc`` and ``kappa_loc`` are unconstrained
   pre-images of positive gene-level globals; ``eta_anchor_loc`` is
   real-valued (no positive map) and plays the role of the gene-level
   activation log-odds ``θ_g``.

2. **No reparameterization step**: ``(rate, kappa, eta_anchor)`` feed
   the Newton kernel directly.  There is no analogue of TSLN-Rate's
   ``_twostate_reparam(mu, burst_size, k_off) → (α, β, r_hat)``
   floor-application step because the Beta shape parameters are built
   per-cell from ``α_cg = κ φ(θ + x_g)``, ``β_cg = κ(1 − φ(θ + x_g))``
   inside the factor function.

3. **Capture support is restricted** (plan §3.2 Rev 4).  PR-2 supports
   only ``x_only`` (no capture) and ``x_only_offset`` (capture frozen
   at the cascade MAP).  The joint ``(z, η_cap)`` Newton requires a
   cross-Hessian

       H_{z, η_cap} = φ' · λ · κ · Cov_q(log p − log(1−p), p)

   that does NOT reduce to NBLN's ``+a`` shortcut (the symmetry only
   exists in Variant A where ``z`` and ``η_cap`` perturb the same
   scalar log-rate with opposite signs).  Implementing the joint
   Schur with this cross-block is deferred to phase 3.  The
   constructor rejects soft-cascade ``eta`` priors and biology-
   anchored ``capture_anchor`` configurations with a clear
   ``NotImplementedError``; soft-eta will return when phase 3 ships.

Latent prior centering
----------------------
The latent ``z ~ N(μ_z, Σ)`` is centered at **zero** in the default
(non-cascade) configuration — TSLN-Logit's gene baseline lives in
the gene-level globals ``(rate, kappa, θ)``, so the latent's prior
mean stays at the origin.  We expose ``mu_x = zeros((G,))`` to the
Newton kernel.

Result-side note
----------------
``self.mu`` on the packed result is **zero** for TSLN-Logit — the
latent prior centering.  This differs from TSLN-Rate where
``self.mu = log(r_hat)`` (the gene-level on-rate latent prior
centering).  The biological reporting quantity is computed as
``gene_mean = rate · σ(eta_anchor)`` (the mean at z = 0).  Both are
exposed via ``get_map()`` / ``get_distributions()`` in Step 3.

See the plan in ``.claude/plans/`` §4.C.3 and the Newton kernel
module docstring for the full math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

if TYPE_CHECKING:
    from ._w_priors import WPriorStrategy

from ..core.twostate_laplace_data_init import (
    empirical_mean_from_counts,
    default_d_init,
    pca_loadings_init,
)
from ..models.config import ModelConfig
from ._em import (
    FinalSweepResult,
    InitState,
    LaplaceObservationModel,
    LaplaceRunResult,
)
from ._newton_twostate_ln_logit import (
    _A_MIN,
    _DEFAULT_K,
    _twostate_ln_logit_factors,
    laplace_log_det_neg_H_batch_x_only,
    laplace_log_det_neg_H_batch_x_only_decoupled,
    laplace_log_det_neg_H_batch_x_only_offset,
    laplace_log_det_neg_H_batch_x_only_offset_decoupled,
    laplace_newton_batch_x_only,
    laplace_newton_batch_x_only_decoupled,
    laplace_newton_batch_x_only_offset,
    laplace_newton_batch_x_only_offset_decoupled,
    twostate_ln_logit_grad_x_only_norm_batch,
    twostate_ln_logit_grad_x_only_norm_batch_decoupled,
    twostate_ln_logit_grad_x_only_offset_norm_batch,
    twostate_ln_logit_grad_x_only_offset_norm_batch_decoupled,
)


# Vmapped factor helper for live-globals log-prob and final-sweep
# clamp diagnostics.  Pulls just ``log_marginal`` and ``a_raw`` out of
# the kernel's factor dict.
def _factors_log_marginal_and_a_raw_logit(
    x, u, rate, kappa, theta, eta_cap, K
):
    fac = _twostate_ln_logit_factors(x, u, rate, kappa, theta, eta_cap, K)
    return fac["log_marginal"], fac["a_raw"]


_factors_batch_x_only = jax.vmap(
    _factors_log_marginal_and_a_raw_logit,
    in_axes=(0, 0, None, None, None, None, None),  # x, u per-cell; rest shared
)
_factors_batch_x_only_offset = jax.vmap(
    _factors_log_marginal_and_a_raw_logit,
    in_axes=(0, 0, None, None, None, 0, None),  # eta_cap per-cell too
)


# =====================================================================
# Woodbury helpers for the MVN prior on x — identical to TSLN-Rate
# =====================================================================


def _woodbury_quadform(
    W: jnp.ndarray, d: jnp.ndarray, diff: jnp.ndarray
) -> jnp.ndarray:
    """Compute ``diffᵀ Σ⁻¹ diff`` for ``Σ = WWᵀ + diag(d)``, batched.

    Bit-identical to :func:`_obs_twostate_ln_rate._woodbury_quadform`.
    """
    inv_d = 1.0 / d
    inv_d_diff = inv_d[None, :] * diff if diff.ndim == 2 else inv_d * diff
    direct = jnp.sum(diff * inv_d_diff, axis=-1)
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    rhs = (inv_d_diff @ W) if diff.ndim == 2 else (W.T @ (inv_d * diff))
    z = jax.scipy.linalg.cho_solve(
        (L_K, True), rhs.T if diff.ndim == 2 else rhs
    )
    if diff.ndim == 2:
        correction = jnp.sum((W @ z) * inv_d_diff.T, axis=0)
    else:
        correction = jnp.dot(W @ z, inv_d * diff)
    return direct - correction


def _woodbury_logdet_sigma(W: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """``log det Σ`` for ``Σ = WWᵀ + diag(d)``."""
    inv_d = 1.0 / d
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_K)))
    log_det_d = jnp.sum(jnp.log(d))
    return log_det_d + log_det_K


# =====================================================================
# Observation-model class
# =====================================================================


class TwoStateLNLogitObservationModel(LaplaceObservationModel):
    """TwoState-LogNormal-Logit observation channel for the Laplace driver.

    Parameters
    ----------
    capture_anchor : Optional[Tuple[float, float]]
        Biology-anchored capture prior ``(log M_0, σ_M)``.  **NOT
        supported in PR-2** — pass ``None``.  Normally the user-facing
        API layer (``scribe.api.stages.model_flags``) rejects
        ``priors={"capture_efficiency": ...}`` for TSLN-Logit before
        reaching this constructor; the constructor's own
        ``NotImplementedError`` is a second line of defense for
        bypasses (e.g. tests or direct engine instantiation).
        Activating it would require the joint cross-Hessian Newton
        path, deferred to phase 3.
    model_config : ModelConfig, optional
        Used to resolve ``positive_transform`` for ``rate`` and
        ``kappa``.  ``eta_anchor`` is real-valued; no positive map.
    informative_priors : Optional[Dict[str, Dict[str, jnp.ndarray]]]
        Soft-cascade Gaussian priors.  Valid keys:
        ``{"rate", "kappa", "eta_anchor"}``.  ``"eta"`` (per-cell
        capture prior) is rejected here — PR-2 supports fixed-offset
        capture only.
    freeze_values : Optional[Dict[str, Dict[str, jnp.ndarray]]]
        Hard-cascade point estimates for parameters in
        ``freeze_params``.  Each value has at least a ``"loc"`` field.
        ``"eta"`` (per-cell capture offset) may carry both ``"loc"``
        and an ignored ``"scale"`` — when ``"eta"`` is in
        ``freeze_params`` the Newton dispatch uses the
        ``x_only_offset`` path with ``loc`` as the fixed offset.
    freeze_params : Tuple[str, ...]
        Parameters excluded from the optax optimizer.  Subset of
        ``{"rate", "kappa", "eta_anchor", "eta"}``.  Default level-4
        cascade: ``("rate", "kappa", "eta_anchor")`` — all three
        gene-level globals hard-frozen.  Note: this differs from
        TSLN-Rate's default which freezes ``(mu, burst_size, k_off)``
        because the global parameter set is itself different.
    w_prior_strategy : Optional[WPriorStrategy]
        W-shrinkage strategy.  PR-2 supports ``NoneWPrior`` only
        (matches PR-1's scope decision for TSLN-Rate).
    max_step : float, default 5.0
        Newton step-size cap.
    n_quad_nodes : int, default 60
        Gauss-Legendre quadrature node count.

    Raises
    ------
    NotImplementedError
        If ``capture_anchor`` is non-``None``, OR
        ``informative_priors`` contains an ``"eta"`` key, OR
        the configuration would otherwise require the joint
        ``(z, η_cap)`` Newton with the deferred cross-Hessian.
    ValueError
        If keys in ``informative_priors`` / ``freeze_params`` /
        ``freeze_values`` are outside the valid set, or required
        fields are missing.
    """

    def __init__(
        self,
        capture_anchor: Optional[Tuple[float, float]] = None,
        model_config: Optional[ModelConfig] = None,
        informative_priors: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
        freeze_values: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
        freeze_params: Tuple[str, ...] = (),
        w_prior_strategy: Optional["WPriorStrategy"] = None,
        max_step: float = 5.0,
        n_quad_nodes: int = _DEFAULT_K,
        gene_names: Optional[Any] = None,
        has_pooled_other: Optional[bool] = None,
    ):
        self._max_step = float(max_step)
        self._n_quad_nodes = int(n_quad_nodes)

        # Stash gene_names / has_pooled_other for AxisLayout construction
        # in ``init_state``.  Mirrors NBLN / TSLN-Rate scaffolding (Commits
        # 2 / 3 of the harmonic-hare plan).  TSLN-Logit's per-gene
        # parameters split across two axes under decoupled:
        #   - ``rate`` / ``kappa`` / ``eta_anchor`` live on the
        #     **observation-layer** axis (G_obs,) — they're per-gene
        #     baselines, not regulatory.
        #   - ``W`` / ``d`` / per-cell z live on the **latent-covariance**
        #     axis (G_kept,) — these encode the regulatory deviation.
        # See ``scribe.laplace._axis_layout`` for the shape contract.
        self._gene_names = gene_names
        self._has_pooled_other = has_pooled_other
        self._correlate_other_column = bool(
            getattr(model_config, "correlate_other_column", True)
        )
        self._axis_layout = None

        from ._w_priors import NoneWPrior

        self._w_prior = (
            w_prior_strategy if w_prior_strategy is not None else NoneWPrior()
        )
        if w_prior_strategy is not None and not isinstance(
            w_prior_strategy, NoneWPrior
        ):
            raise NotImplementedError(
                "TSLN-Logit W-shrinkage strategies beyond NoneWPrior are "
                "deferred to a follow-up PR (same scope choice as PR-1 "
                "for TSLN-Rate)."
            )

        # --- Capture-mode validation (Rev 4 invariant — STRICT) -----------
        # PR-2 supports only two capture modes:
        #
        #   * ``x_only``         — no capture.
        #   * ``x_only_offset``  — capture frozen at the upstream MAP.
        #
        # Both require the eta offset (when present) to be a frozen
        # constant, not a Newton iterate.  We reject configurations
        # that would trigger the joint ``(z, eta_cap)`` Newton because
        # the proper cross-Hessian implementation is deferred to phase
        # 3 (see _newton_twostate_ln_logit.py module docstring).
        #
        # Soft-cascade eta (a Normal prior on eta with non-zero scale,
        # supplied via ``informative_priors["eta"]``) and the biology-
        # anchored TruncN prior (``capture_anchor=(log_M0, sigma_M)``)
        # both route to the joint path in TSLN-Rate; here they would
        # silently fall into a broken Newton step.  Reject loudly.
        prior_eta = (
            informative_priors.get("eta")
            if informative_priors is not None
            else None
        )
        if capture_anchor is not None:
            raise NotImplementedError(
                "TSLN-Logit with a biology-anchored capture_anchor "
                f"({capture_anchor!r}) would require the joint "
                "(z, eta_cap) Newton path, whose cross-Hessian "
                "H_{z, eta_cap} = φ' · λ · κ · Cov_q(log p − log(1-p), p) "
                "does NOT reduce to NBLN/TSLN-Rate's '+a' shortcut. "
                "Joint capture Newton is deferred to a phase-3 "
                "follow-up. PR-2 supports only:\n"
                "  * no capture (capture_anchor=None, no eta in freeze_params)\n"
                "  * fixed-offset capture (eta in freeze_params with "
                "freeze_values['eta']['loc'] from cascade).\n"
                "Drop capture_anchor or pass eta via freeze_values + "
                "freeze_params=('eta',)."
            )
        if prior_eta is not None:
            raise NotImplementedError(
                "TSLN-Logit with a soft-cascade eta prior "
                "(informative_priors['eta']) would require the joint "
                "(z, eta_cap) Newton path; see the capture_anchor "
                "message above for details. Cascade eta as a frozen "
                "offset (freeze_params=('eta',), "
                "freeze_values['eta']) instead, or drop eta priors "
                "altogether."
            )

        # No capture anchor stored — TSLN-Logit never uses one in PR-2.
        self._capture_anchor = None

        # --- positive-transform resolution --------------------------------
        from ._global_uncertainty import _JAX_POSITIVE_FNS

        def _resolve_for(internal_name: str):
            if model_config is None:
                return _JAX_POSITIVE_FNS["softplus"]
            transform_name = model_config.resolve_positive_transform(
                internal_name
            )
            if transform_name not in _JAX_POSITIVE_FNS:
                raise ValueError(
                    f"Unknown positive_transform for {internal_name!r}: "
                    f"{transform_name!r}.  Expected one of "
                    f"{set(_JAX_POSITIVE_FNS)}."
                )
            return _JAX_POSITIVE_FNS[transform_name]

        # ``rate`` and ``kappa`` are positive globals (apply pos_forward).
        # ``eta_anchor`` is real-valued — identity transform on its loc.
        # ``d`` (residual variance) is positive.
        self._pos_fns = {
            "rate": _resolve_for("rate"),
            "kappa": _resolve_for("kappa"),
            "d": _resolve_for("d"),
        }
        self._rate_fwd, self._rate_inv = self._pos_fns["rate"]
        self._kappa_fwd, self._kappa_inv = self._pos_fns["kappa"]
        self._d_fwd, self._d_inv = self._pos_fns["d"]
        # Back-compat: expose a (forward, inverse) pair under
        # ``_pos_forward / _pos_inverse``; resolves the ``rate``
        # transform (used by compute_global_uncertainty fallback).
        self._pos_forward, self._pos_inverse = self._pos_fns["rate"]

        # --- valid-key validation ----------------------------------------
        valid_global = {"rate", "kappa", "eta_anchor"}
        valid_freeze = valid_global | {"eta"}
        if informative_priors is not None:
            invalid = set(informative_priors) - valid_global
            if invalid:
                raise ValueError(
                    f"informative_priors has unrecognized keys {invalid}; "
                    f"valid keys are {valid_global} (note: 'eta' priors "
                    "are NOT supported in PR-2 — see capture-mode docs)."
                )
        self._prior_rate = (
            informative_priors.get("rate")
            if informative_priors is not None
            else None
        )
        self._prior_kappa = (
            informative_priors.get("kappa")
            if informative_priors is not None
            else None
        )
        self._prior_eta_anchor = (
            informative_priors.get("eta_anchor")
            if informative_priors is not None
            else None
        )
        # _prior_eta is guaranteed None here by the validation above —
        # we retain the slot as None for symmetry with TSLN-Rate.
        self._prior_eta = None

        bad_frozen = set(freeze_params) - valid_freeze
        if bad_frozen:
            raise ValueError(
                f"freeze_params has invalid keys {bad_frozen}; "
                f"valid keys are {valid_freeze}."
            )
        if freeze_params:
            if freeze_values is None:
                raise ValueError(
                    f"freeze_params={list(freeze_params)} non-empty but "
                    "freeze_values is None."
                )
            missing = set(freeze_params) - set(freeze_values.keys())
            if missing:
                raise ValueError(
                    f"freeze_params requests {sorted(missing)} but "
                    "freeze_values does not provide those keys. "
                    f"Available: {sorted(freeze_values.keys())}."
                )
            for k in freeze_params:
                if "loc" not in freeze_values[k]:
                    raise ValueError(
                        f"freeze_values[{k!r}] missing 'loc' field."
                    )
        self._frozen_params = frozenset(freeze_params)
        self._freeze_values = freeze_values if freeze_values is not None else {}

        self._model_config = model_config

    # ---- Identity --------------------------------------------------------

    @property
    def name(self) -> str:
        return "twostate_ln_logit"

    @property
    def uses_capture(self) -> bool:
        # PR-2: capture is on iff eta is hard-frozen (i.e., supplied as
        # a fixed offset via freeze_values).  Soft-cascade and
        # biology-anchored modes are both rejected in __init__.
        return "eta" in self._frozen_params

    @property
    def freezes_eta(self) -> bool:
        return "eta" in self._frozen_params

    @property
    def frozen_params(self) -> frozenset:
        return self._frozen_params

    # ---- Helpers ---------------------------------------------------------

    def _splice_frozen(
        self, params: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Return ``params`` with frozen values spliced in.

        Frozen entries are excluded from the optimizer dict, so each
        method that needs them re-injects from ``self._freeze_values``.
        ``stop_gradient`` keeps them out of autodiff.  ``eta`` is
        per-cell and is handled directly at the loss call site.
        """
        out = dict(params)
        for k in self._frozen_params:
            if k == "eta":
                continue
            out[f"{k}_loc"] = jax.lax.stop_gradient(
                jnp.asarray(self._freeze_values[k]["loc"])
            )
        return out

    def _globals_from_params(
        self, params: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward-transform ``(rate_loc, kappa_loc, eta_anchor_loc)``.

        Returns ``(rate, kappa, eta_anchor)`` — the gene-level globals
        consumed by the Newton kernel.  ``eta_anchor`` is real-valued
        and passes through identity.
        """
        rate = self._rate_fwd(params["rate_loc"])
        kappa = self._kappa_fwd(params["kappa_loc"])
        eta_anchor = params["eta_anchor_loc"]   # identity transform
        return rate, kappa, eta_anchor

    # ---- init_state ------------------------------------------------------

    def init_state(
        self,
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        latent_dim: int,
        seed: int,
    ) -> InitState:
        counts_np = np.asarray(count_data)

        # Build the AxisLayout (harmonic-hare Commit 4 scaffolding).
        # See NBLN / TSLN-Rate for the precedent.  TSLN-Logit's
        # per-gene parameters (rate, kappa, eta_anchor) stay on the
        # OBSERVATION axis (G_obs,) under decoupled; only W, d, and
        # the per-cell z latent shrink to G_kept.
        from ._axis_layout import build_axis_layout
        self._axis_layout = build_axis_layout(
            n_genes=int(n_genes),
            correlate_other_column=self._correlate_other_column,
            gene_names=self._gene_names,
            has_pooled_other=self._has_pooled_other,
        )
        _layout = self._axis_layout

        # Commit 4b: TSLN-Logit decoupled deviation-form math is live.
        # No early guard here — init proceeds for both layouts.  The
        # decoupled branches in ``loss_fn`` / ``final_sweep`` /
        # ``compute_global_uncertainty`` consume ``self._axis_layout``
        # directly and drive the deviation-form Newton kernels in
        # ``_newton_twostate_ln_logit`` (the ``*_decoupled`` family).

        # ---- rate: frozen > prior > data (default 2 · empirical_mean) ----
        # The default corresponds to ``eta_anchor_init = 0`` (i.e.,
        # σ(θ_anchor) = 0.5), so that ``E[u | z=0] = rate · σ(0) =
        # 0.5 · rate`` matches the empirical mean when rate = 2·mean.
        if "rate" in self._frozen_params:
            rate_pos = self._rate_fwd(
                jnp.asarray(self._freeze_values["rate"]["loc"])
            )
        elif self._prior_rate is not None:
            rate_pos = self._rate_fwd(jnp.asarray(self._prior_rate["loc"]))
        else:
            emp_mean = empirical_mean_from_counts(counts_np)
            rate_pos = 2.0 * jnp.asarray(emp_mean)
        rate_loc_init = self._rate_inv(rate_pos)

        # ---- kappa: frozen > prior > data (default 3.0 = SVI median) ----
        if "kappa" in self._frozen_params:
            kappa_pos = self._kappa_fwd(
                jnp.asarray(self._freeze_values["kappa"]["loc"])
            )
        elif self._prior_kappa is not None:
            kappa_pos = self._kappa_fwd(
                jnp.asarray(self._prior_kappa["loc"])
            )
        else:
            kappa_pos = jnp.full(
                (int(n_genes),), 3.0, dtype=jnp.float32
            )
        kappa_loc_init = self._kappa_inv(kappa_pos)

        # ---- eta_anchor: frozen > prior > data (default 0.0) ------------
        # Real-valued; identity transform.
        if "eta_anchor" in self._frozen_params:
            eta_anchor_loc_init = jnp.asarray(
                self._freeze_values["eta_anchor"]["loc"]
            )
        elif self._prior_eta_anchor is not None:
            eta_anchor_loc_init = jnp.asarray(
                self._prior_eta_anchor["loc"]
            )
        else:
            eta_anchor_loc_init = jnp.zeros(
                (int(n_genes),), dtype=jnp.float32
            )

        # ---- W, d, latent ------------------------------------------------
        # ``W`` and ``d`` live on the LATENT-COVARIANCE axis (G_kept,).
        # Under decoupling we slice the count matrix to kept genes
        # before PCA so the resulting loadings are ``(G_kept, K)``
        # directly — slicing post-PCA would let ``_other`` variance
        # leak into kept-gene loadings via the SVD coupling.  See
        # NBLN / TSLN-Rate ``init_state`` for the same pattern.
        if _layout.decoupled:
            counts_for_pca = counts_np[:, _layout.kept_idx]
        else:
            counts_for_pca = counts_np
        W_init = pca_loadings_init(counts_for_pca, latent_dim=int(latent_dim))
        d_pos = default_d_init(int(_layout.G_kept))
        d_loc_init = self._d_inv(d_pos)

        # ---- per-cell η — only the fixed-offset case is supported -------
        if "eta" in self._frozen_params:
            eta_loc = jnp.asarray(self._freeze_values["eta"]["loc"])
        else:
            eta_loc = None  # x_only path

        # ---- per-cell latent warm start ---------------------------------
        # The latent z has prior N(0, Σ).  For a no-capture
        # configuration, starting at z = 0 is a reasonable warm start;
        # for the frozen-offset path, there is no z-shift to anticipate
        # from η (the offset enters via the Poisson log-rate, not via
        # the prior on z), so z = 0 is correct there as well.  This
        # contrasts with TSLN-Rate where ``log_rate = x − η`` requires
        # a ``+η`` warm-start shift on ``x``.
        # Under decoupling (Commit 4b), the latent has shape
        # ``(n_cells, G_kept)``; z = 0 remains the right warm start
        # since the kept-axis prior centre is still zero.
        latent_loc = jnp.zeros(
            (int(n_cells), int(_layout.G_kept)), dtype=jnp.float32,
        )

        # eta_anchor field on InitState is per-cell in the base class;
        # TSLN-Logit's eta_anchor is per-GENE (= θ_g).  We populate
        # the per-cell ``eta_anchor`` slot with zeros so the
        # base-class signature stays satisfied — the gene-level
        # eta_anchor is carried through ``params["eta_anchor_loc"]``
        # instead.  ``eta_anchor`` here is a placeholder used only
        # when the joint-Newton path is active, which it is not in
        # PR-2.
        eta_anchor_per_cell = (
            jnp.zeros((int(n_cells),), dtype=jnp.float32)
            if eta_loc is not None
            else None
        )

        # Params dict — exclude frozen gene-level keys (eta is per-cell)
        params: Dict[str, jnp.ndarray] = {
            "W": W_init,
            "d_loc": d_loc_init,
        }
        if "rate" not in self._frozen_params:
            params["rate_loc"] = rate_loc_init
        if "kappa" not in self._frozen_params:
            params["kappa_loc"] = kappa_loc_init
        if "eta_anchor" not in self._frozen_params:
            params["eta_anchor_loc"] = eta_anchor_loc_init

        aux_data: Dict[str, jnp.ndarray] = {}
        return InitState(
            params=params,
            latent_loc=latent_loc,
            eta_loc=eta_loc,
            eta_anchor=eta_anchor_per_cell,
            aux_data=aux_data,
        )

    # ---- loss_fn ---------------------------------------------------------

    def loss_fn(
        self,
        params: Dict[str, jnp.ndarray],
        latent_init: jnp.ndarray,
        eta_init: jnp.ndarray,
        counts_batch: jnp.ndarray,
        eta_anchor_batch: jnp.ndarray,
        aux_batch: Dict[str, jnp.ndarray],
        data_scale: float,
        n_newton: int,
        damping: float,
    ) -> Tuple[
        jnp.ndarray,
        Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict[str, jnp.ndarray]],
    ]:
        """Negative Laplace ELBO on one mini-batch.

        Two-way capture dispatch (``x_only`` / ``x_only_offset``).
        Soft-cascade and biology-anchored modes are rejected in
        ``__init__`` so this loss_fn never sees them.
        """
        # Commit 4b: decoupled deviation-form math is live.  Each
        # Newton dispatch branch below has a decoupled twin.
        _layout = self._axis_layout
        _is_decoupled = _layout is not None and _layout.decoupled
        if _is_decoupled:
            _kept_idx_j = jnp.asarray(_layout.kept_idx, dtype=jnp.int32)
        else:
            _kept_idx_j = None

        params_full = self._splice_frozen(params)
        rate, kappa, eta_anchor = self._globals_from_params(params_full)
        W = params_full["W"]
        d = self._d_fwd(params_full["d_loc"])
        n_quad_nodes = self._n_quad_nodes
        max_step = self._max_step

        # Stop-gradient on every Newton input — variational-EM convention.
        latent_init_sg = jax.lax.stop_gradient(latent_init)
        rate_sg = jax.lax.stop_gradient(rate)
        kappa_sg = jax.lax.stop_gradient(kappa)
        eta_anchor_sg = jax.lax.stop_gradient(eta_anchor)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)
        # Latent prior is centred at zero (z ~ N(0, Σ)); no per-cell mu_x.
        n_genes = rate.shape[0]
        mu_x_sg = jnp.zeros((n_genes,), dtype=rate.dtype)

        # Commit 4b: each branch has a decoupled twin.  Under
        # decoupling, ``latent_init`` carries ``z_kept`` on G_kept;
        # the decoupled kernels scatter ``z_kept`` onto a full G_obs
        # zero-base and feed the resulting ``x_full`` to
        # ``_twostate_ln_logit_factors`` (which computes the
        # activation log-odds as ``θ + x_full`` — kept genes get
        # ``θ + z_kept``, ``_other`` gets ``θ`` only).
        if not self.uses_capture:
            if _is_decoupled:
                x_new, final_grad, _ldet_dead, _lm_dead, _a_dead = (
                    laplace_newton_batch_x_only_decoupled(
                        latent_init_sg, counts_batch, W_sg, d_sg,
                        rate_sg, kappa_sg, eta_anchor_sg, _kept_idx_j,
                        n_newton, damping, max_step, n_quad_nodes,
                    )
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = None
                log_det = laplace_log_det_neg_H_batch_x_only_decoupled(
                    x_new, counts_batch, W, d, rate, kappa, eta_anchor,
                    _kept_idx_j, n_quad_nodes,
                )
                gn_x = twostate_ln_logit_grad_x_only_norm_batch_decoupled(
                    x_new, counts_batch, W_sg, d_sg, rate_sg,
                    kappa_sg, eta_anchor_sg, _kept_idx_j, n_quad_nodes,
                )
                eta_cap_for_lp = jnp.asarray(0.0, dtype=rate.dtype)
            else:
                x_new, final_grad, _ldet_dead, _lm_dead, _a_dead = (
                    laplace_newton_batch_x_only(
                        latent_init_sg,
                        counts_batch,
                        mu_x_sg,
                        W_sg,
                        d_sg,
                        rate_sg,
                        kappa_sg,
                        eta_anchor_sg,
                        n_newton,
                        damping,
                        max_step,
                        n_quad_nodes,
                    )
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = None
                log_det = laplace_log_det_neg_H_batch_x_only(
                    x_new, counts_batch, rate, kappa, eta_anchor, W, d,
                    n_quad_nodes,
                )
                gn_x = twostate_ln_logit_grad_x_only_norm_batch(
                    x_new, counts_batch, mu_x_sg, W_sg, d_sg,
                    rate_sg, kappa_sg, eta_anchor_sg, n_quad_nodes,
                )
                eta_cap_for_lp = jnp.asarray(0.0, dtype=rate.dtype)
            gn_blocks = {"x": gn_x}

        else:
            # Fixed-offset capture (``x_only_offset``).  eta_init carries
            # the per-cell offset from the cascade; pin it as a
            # stop_gradient constant.
            eta_offset = jax.lax.stop_gradient(eta_init)
            if _is_decoupled:
                x_new, final_grad, _ldet_dead, _lm_dead, _a_dead = (
                    laplace_newton_batch_x_only_offset_decoupled(
                        latent_init_sg, counts_batch, W_sg, d_sg,
                        rate_sg, kappa_sg, eta_anchor_sg, eta_offset,
                        _kept_idx_j, n_newton, damping, max_step,
                        n_quad_nodes,
                    )
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = eta_offset
                log_det = (
                    laplace_log_det_neg_H_batch_x_only_offset_decoupled(
                        x_new, eta_offset, counts_batch, W, d,
                        rate, kappa, eta_anchor, _kept_idx_j,
                        n_quad_nodes,
                    )
                )
                gn_x = (
                    twostate_ln_logit_grad_x_only_offset_norm_batch_decoupled(
                        x_new, counts_batch, W_sg, d_sg, rate_sg,
                        kappa_sg, eta_anchor_sg, eta_offset,
                        _kept_idx_j, n_quad_nodes,
                    )
                )
            else:
                x_new, final_grad, _ldet_dead, _lm_dead, _a_dead = (
                    laplace_newton_batch_x_only_offset(
                        latent_init_sg,
                        counts_batch,
                        mu_x_sg,
                        W_sg,
                        d_sg,
                        rate_sg,
                        kappa_sg,
                        eta_anchor_sg,
                        eta_offset,
                        n_newton,
                        damping,
                        max_step,
                        n_quad_nodes,
                    )
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = eta_offset
                log_det = laplace_log_det_neg_H_batch_x_only_offset(
                    x_new, eta_offset, counts_batch, rate, kappa, eta_anchor,
                    W, d, n_quad_nodes,
                )
                gn_x = twostate_ln_logit_grad_x_only_offset_norm_batch(
                    x_new, counts_batch, mu_x_sg, W_sg, d_sg,
                    rate_sg, kappa_sg, eta_anchor_sg, eta_offset,
                    n_quad_nodes,
                )
            gn_blocks = {"x": gn_x}

        # Live-globals log-marginal at the stop-grad'd MAP.  Under
        # decoupling, ``x_new`` is ``z_kept`` on G_kept; scatter onto
        # G_obs (zeros at ``_other``) before feeding the factors helper,
        # which then forms ``η_act = θ + x_full``.  Under legacy,
        # ``x_new`` is already on G_obs.
        if _is_decoupled:
            G_obs_ts = int(rate.shape[0])
            x_for_lp_base = jnp.zeros(
                (x_new.shape[0], G_obs_ts), dtype=x_new.dtype,
            )
            x_for_lp = x_for_lp_base.at[:, _kept_idx_j].add(x_new)
        else:
            x_for_lp = x_new

        if not self.uses_capture:
            log_marginal_live_per_gene, _ = _factors_batch_x_only(
                x_for_lp, counts_batch, rate, kappa, eta_anchor,
                eta_cap_for_lp, n_quad_nodes,
            )
        else:
            log_marginal_live_per_gene, _ = _factors_batch_x_only_offset(
                x_for_lp, counts_batch, rate, kappa, eta_anchor,
                eta_offset, n_quad_nodes,
            )

        data_lp_per_cell = log_marginal_live_per_gene.sum(axis=-1)  # (C,)

        # MVN prior on z (zero-centred under both layouts) via inner
        # Woodbury — per-cell.  Under decoupling the MVN dimension is
        # G_kept; under legacy it is G_obs.  ``mu_x_sg`` is zeros in
        # either case so ``diff = x_new`` directly.
        diff = x_new
        quad_form = _woodbury_quadform(W, d, diff)
        log_det_sigma = _woodbury_logdet_sigma(W, d)
        if _is_decoupled:
            mvn_n_genes = int(_layout.G_kept)
        else:
            mvn_n_genes = n_genes
        mvn_lp_per_cell = (
            -0.5 * quad_form
            - 0.5 * log_det_sigma
            - 0.5 * mvn_n_genes * jnp.log(2.0 * jnp.pi)
        )

        # No TruncN prior on eta in PR-2 — soft-eta is rejected upstream.
        eta_lp_per_cell = jnp.zeros_like(data_lp_per_cell)
        laplace_corr_per_cell = -0.5 * log_det

        elbo_per_cell = (
            data_lp_per_cell
            + mvn_lp_per_cell
            + eta_lp_per_cell
            + laplace_corr_per_cell
        )
        elbo_per_cell = jnp.where(
            jnp.isfinite(elbo_per_cell),
            elbo_per_cell,
            jnp.zeros_like(elbo_per_cell),
        )

        # Soft-cascade global Normal priors (gene-level).
        global_prior_lp = jnp.zeros(())
        for key, prior in (
            ("rate_loc", self._prior_rate),
            ("kappa_loc", self._prior_kappa),
            ("eta_anchor_loc", self._prior_eta_anchor),
        ):
            if prior is not None and key in params:
                global_prior_lp = global_prior_lp + jnp.sum(
                    dist.Normal(
                        loc=jnp.asarray(prior["loc"]),
                        scale=jnp.asarray(prior["scale"]),
                    ).log_prob(params[key])
                )

        # W-shrinkage prior on the gauge-invariant projection.
        _W_raw = params_full["W"]
        _W_for_prior = _W_raw - jnp.mean(_W_raw, axis=0, keepdims=True)
        w_aux = {
            name: params_full[name]
            for name in getattr(self._w_prior, "aux_param_names", ())
        }
        global_prior_lp = global_prior_lp + self._w_prior.log_prior(
            _W_for_prior, w_aux, n_constraints=1,
        )

        loss = -float(data_scale) * jnp.sum(elbo_per_cell) - global_prior_lp
        return loss, (x_new, eta_new, gn_blocks)

    # ---- final_sweep -----------------------------------------------------

    def final_sweep(
        self,
        params: Dict[str, jnp.ndarray],
        latent_loc: jnp.ndarray,
        eta_loc: Optional[jnp.ndarray],
        eta_anchor: Optional[jnp.ndarray],
        count_data: jnp.ndarray,
        aux_data: Dict[str, jnp.ndarray],
        n_newton: int,
        damping: float,
    ) -> FinalSweepResult:
        """Full-population Newton sweep with ``2 × n_newton`` iterations."""
        params_full = self._splice_frozen(params)
        rate, kappa, eta_anchor_gene = self._globals_from_params(params_full)
        W = params_full["W"]
        d = self._d_fwd(params_full["d_loc"])
        n_iters = int(2 * n_newton)
        n_q = self._n_quad_nodes
        ms = self._max_step
        n_genes = rate.shape[0]
        mu_x = jnp.zeros((n_genes,), dtype=rate.dtype)

        # Commit 4b: decoupled dispatch mirrors loss_fn.
        _layout = self._axis_layout
        _is_decoupled = _layout is not None and _layout.decoupled
        if _is_decoupled:
            _kept_idx_j = jnp.asarray(_layout.kept_idx, dtype=jnp.int32)

        if not self.uses_capture:
            if _is_decoupled:
                x_new, final_grad, _ldet, _lm, _a_raw = (
                    laplace_newton_batch_x_only_decoupled(
                        latent_loc, count_data, W, d, rate, kappa,
                        eta_anchor_gene, _kept_idx_j, n_iters,
                        damping, ms, n_q,
                    )
                )
            else:
                x_new, final_grad, _ldet, _lm, _a_raw = laplace_newton_batch_x_only(
                    latent_loc,
                    count_data,
                    mu_x,
                    W,
                    d,
                    rate,
                    kappa,
                    eta_anchor_gene,
                    n_iters,
                    damping,
                    ms,
                    n_q,
                )
            final_eta_loc = None
            eta_cap_for_diag = jnp.asarray(0.0, dtype=rate.dtype)
        else:
            eta_offset = jnp.asarray(self._freeze_values["eta"]["loc"])
            if _is_decoupled:
                x_new, final_grad, _ldet, _lm, _a_raw = (
                    laplace_newton_batch_x_only_offset_decoupled(
                        latent_loc, count_data, W, d, rate, kappa,
                        eta_anchor_gene, eta_offset, _kept_idx_j,
                        n_iters, damping, ms, n_q,
                    )
                )
            else:
                (
                    x_new,
                    final_grad,
                    _ldet,
                    _lm,
                    _a_raw,
                ) = laplace_newton_batch_x_only_offset(
                    latent_loc,
                    count_data,
                    mu_x,
                    W,
                    d,
                    rate,
                    kappa,
                    eta_anchor_gene,
                    eta_offset,
                    n_iters,
                    damping,
                    ms,
                    n_q,
                )
            final_eta_loc = eta_offset

        # Build x_for_diag of shape ``(C, G_obs)`` for the clamp
        # diagnostic factor evaluation.  Under decoupling, ``x_new``
        # is ``z_kept`` on G_kept; scatter onto G_obs zero-base.
        if _is_decoupled:
            G_obs_diag = int(rate.shape[0])
            _diag_base = jnp.zeros(
                (x_new.shape[0], G_obs_diag), dtype=x_new.dtype,
            )
            x_for_diag = _diag_base.at[:, _kept_idx_j].add(x_new)
        else:
            x_for_diag = x_new

        if not self.uses_capture:
            _, a_raw_per_cell = _factors_batch_x_only(
                x_for_diag, count_data, rate, kappa, eta_anchor_gene,
                eta_cap_for_diag, n_q,
            )
        else:
            _, a_raw_per_cell = _factors_batch_x_only_offset(
                x_for_diag, count_data, rate, kappa, eta_anchor_gene,
                eta_offset, n_q,
            )

        # Clamp diagnostics.
        a_raw_flat = a_raw_per_cell.reshape(-1)
        a_raw_min_val = float(jnp.min(a_raw_per_cell))
        a_raw_neg_frac = float(jnp.mean((a_raw_flat < 0.0).astype(jnp.float32)))
        a_clamp_frac = float(
            jnp.mean((a_raw_flat < _A_MIN).astype(jnp.float32))
        )
        a_clamp_per_gene = jnp.mean(
            (a_raw_per_cell < _A_MIN).astype(jnp.float32), axis=0
        )
        self._final_clamp_stats = {
            "a_raw_min": a_raw_min_val,
            "a_raw_negative_fraction": a_raw_neg_frac,
            "a_clamp_fraction": a_clamp_frac,
            "a_clamp_per_gene": np.asarray(a_clamp_per_gene),
        }
        if a_clamp_frac > 0.05:
            import logging
            logging.getLogger(__name__).warning(
                "TSLN-Logit curvature clamp activated on %.1f%% of "
                "(cell, gene) entries (threshold 5%%). The Laplace "
                "approximation is locally prior-dominated for those "
                "entries; posterior credible intervals should be "
                "interpreted cautiously. See result.a_clamp_per_gene "
                "for the per-gene breakdown.",
                100.0 * a_clamp_frac,
            )

        return FinalSweepResult(
            latent_loc=x_new,
            eta_loc=final_eta_loc,
            final_grad_norms=final_grad,
        )

    # ---- compute_global_uncertainty -----------------------------------------

    def compute_global_uncertainty(
        self,
        params: Dict[str, jnp.ndarray],
        latent_loc: jnp.ndarray,
        eta_loc: Optional[jnp.ndarray],
        eta_anchor: Optional[jnp.ndarray],
        count_data: jnp.ndarray,
        aux_data: Dict[str, jnp.ndarray],
        model_config: ModelConfig,
    ) -> Dict[str, jnp.ndarray]:
        """Diagonal-Hessian Laplace approximation on gene globals.

        Builds a per-cell ELBO closure parametrized by
        ``(rate_loc, kappa_loc, eta_anchor_loc)`` and takes the
        per-gene diagonal Hessian via ``jnp.diag(jax.hessian(...))``
        for each unfrozen global.

        Same simplifications as TSLN-Rate:

        - **Marginal**, not profiled (no Schur correction for the
          per-cell MAP shifts).
        - **Diagonal**, no cross-parameter Hessian blocks.
        - **NaN sentinels for frozen globals** — bridge moment-matches
          from cascade when present.

        Default freeze (Level 4: rate, kappa, eta_anchor all frozen)
        means this method typically returns NaN for every gene-level
        scale field; that is expected.
        """
        # Defensive decoupled-layout guard (see loss_fn).
        # Commit 4b: decoupled compute_global_uncertainty.  The closed-
        # form curvature path below already operates per-gene on
        # G_obs; under decoupling, ``latent_loc`` is ``z_kept`` on
        # G_kept, so we scatter onto G_obs (zeros at ``_other``)
        # before feeding ``global_curvature_logit_summed``.  No other
        # change is needed because:
        #   * ``rate`` and ``kappa`` are z-independent — their
        #     curvatures don't see the latent at all.
        #   * ``eta_anchor`` curvature uses the per-cell ``a_g`` from
        #     ``_twostate_ln_logit_factors`` evaluated at the full
        #     G_obs activation log-odds (kept genes get
        #     ``θ + z_kept``, ``_other`` gets ``θ``) — that's exactly
        #     what scattering produces.
        # No Schur correction for eta_anchor through z (consistent
        # with the existing "diagonal, no cross-parameter Hessian
        # blocks" approximation noted in the function docstring).
        _layout = self._axis_layout
        _is_decoupled = _layout is not None and _layout.decoupled

        from ._global_uncertainty import curvature_to_scale

        params_full = self._splice_frozen(params)
        x_map_sg = jax.lax.stop_gradient(latent_loc)
        eta_map_sg = (
            jax.lax.stop_gradient(eta_loc) if eta_loc is not None else None
        )
        counts = jnp.asarray(count_data)
        n_cells = counts.shape[0]
        n_q = self._n_quad_nodes
        W_sg = jax.lax.stop_gradient(params_full["W"])
        d_sg = jax.lax.stop_gradient(self._d_fwd(params_full["d_loc"]))

        # eta_cap fed into the factor function (scalar for x_only;
        # per-cell for x_only_offset).
        if eta_map_sg is None:
            eta_cap_at_map = jnp.asarray(0.0)
            factor_batch_helper = _factors_batch_x_only
            extra_args = (eta_cap_at_map, n_q)
        else:
            eta_cap_at_map = eta_map_sg
            factor_batch_helper = _factors_batch_x_only_offset
            extra_args = (eta_cap_at_map, n_q)

        def _prior_precision(prior: Optional[Dict[str, jnp.ndarray]], n_g: int):
            if prior is None:
                return jnp.zeros((n_g,), dtype=jnp.float32)
            return 1.0 / jnp.square(jnp.asarray(prior["scale"]))

        def neg_log_post(rate_loc_v, kappa_loc_v, eta_anchor_loc_v):
            rate_v = self._rate_fwd(rate_loc_v)
            kappa_v = self._kappa_fwd(kappa_loc_v)
            eta_anchor_v = eta_anchor_loc_v
            # Data log-prob via K-axis quadrature reduction at the
            # stop-grad'd MAP.
            log_marg_per_gene, _ = factor_batch_helper(
                x_map_sg, counts, rate_v, kappa_v, eta_anchor_v, *extra_args,
            )
            data_lp = jnp.sum(log_marg_per_gene)

            # MVN prior centred at zero (latent z ~ N(0, Σ)).
            diff = x_map_sg
            quad = _woodbury_quadform(W_sg, d_sg, diff)
            log_det_sigma = _woodbury_logdet_sigma(W_sg, d_sg)
            n_genes = rate_v.shape[0]
            mvn_lp = (
                -0.5 * jnp.sum(quad)
                - 0.5 * n_cells * (
                    log_det_sigma + n_genes * jnp.log(2.0 * jnp.pi)
                )
            )
            return -(data_lp + mvn_lp)

        # Resolve per-parameter constrained MAPs.
        rate_loc = params_full["rate_loc"]
        kappa_loc = params_full["kappa_loc"]
        eta_anchor_loc = params_full["eta_anchor_loc"]
        n_g = rate_loc.shape[0]

        out: Dict[str, jnp.ndarray] = {
            "rate_loc": rate_loc,
            "kappa_loc": kappa_loc,
            "eta_anchor_loc": eta_anchor_loc,
        }

        # Closed-form curvature path (auditor's hand-derived route).
        # See ``paper/_two_state_promoter.qmd``
        # §sec-twostate-tsln-logit-global-uncertainty.  Computes the
        # per-cell-per-gene gradient and Hessian-diagonal in the
        # natural (log_rate, κ, θ) spaces directly from the existing
        # Newton-kernel softmax moments + a handful of extra
        # quadrature reductions, then chain-rules to the
        # unconstrained ``*_loc`` space via per-element autograd of
        # the configured ``pos_forward`` transform.
        #
        # Properties:
        #   * Memory bounded — no autodiff hessian intermediate.
        #   * ~10–50× faster than ``hessian_diag_chunked``.
        #   * Numerically agrees with the chunked-autodiff path to
        #     float32 precision (covered by
        #     ``tests/test_twostate_ln_logit_global_curvature.py``).
        from ._newton_twostate_ln_logit import (
            global_curvature_logit_summed,
        )

        # Resolve gene-level natural values from the optimized locs.
        rate_at_map = self._rate_fwd(rate_loc)
        kappa_at_map = self._kappa_fwd(kappa_loc)
        theta_at_map = eta_anchor_loc       # identity transform

        # Per-cell eta_cap array (zeros under x_only).
        if eta_map_sg is None:
            eta_cap_for_curv = jnp.zeros(
                (n_cells,), dtype=rate_at_map.dtype
            )
        else:
            eta_cap_for_curv = eta_map_sg

        # Build the full G_obs ``x_map`` for the data-side quadrature.
        # Under legacy this is ``x_map_sg`` directly (G_obs).  Under
        # decoupling, ``x_map_sg`` is ``z_kept`` on G_kept; scatter
        # onto G_obs (zeros at ``_other`` — matching the math contract
        # that ``η_act_other = θ_other + 0``).
        if _is_decoupled:
            _kept_idx_g = jnp.asarray(
                _layout.kept_idx, dtype=jnp.int32
            )
            _G_obs_g = int(rate_at_map.shape[0])
            _x_base_g = jnp.zeros(
                (n_cells, _G_obs_g), dtype=x_map_sg.dtype,
            )
            x_map_for_curv = _x_base_g.at[:, _kept_idx_g].add(x_map_sg)
        else:
            x_map_for_curv = x_map_sg

        curv = global_curvature_logit_summed(
            x_map=x_map_for_curv,
            counts=counts,
            rate=rate_at_map,
            kappa=kappa_at_map,
            theta=theta_at_map,
            eta_cap=eta_cap_for_curv,
            n_quad_nodes=n_q,
        )

        # Helpers for elementwise chain-rule from natural -> loc space.
        # All transforms are elementwise so first/second derivatives
        # are diagonal; vmap+grad gives O(G) ops total.
        def _chain_to_loc_positive(pos_forward, x_loc):
            """Return (dy/dx_loc, d²y/dx_loc²) for y = pos_forward(x_loc)."""
            d1 = jax.vmap(jax.grad(pos_forward))(x_loc)
            d2 = jax.vmap(jax.grad(jax.grad(pos_forward)))(x_loc)
            return d1, d2

        # For ``rate``: two-step chain.  First convert curvature in
        # log_rate space to curvature in rate space, then to
        # rate_loc space.
        #   y = log(rate);  ∂²f/∂rate² = (1/rate²) ∂²f/∂(log rate)²
        #                                + (-1/rate²) ∂f/∂(log rate)
        # then ∂²f/∂rate_loc² uses pos_forward chain rule.
        if "rate" in self._frozen_params:
            out["rate_scale"] = jnp.full_like(rate_loc, jnp.nan)
        else:
            H_log_r = curv["H_log_rate"]
            g_log_r = curv["g_log_rate"]
            rate_sq = jnp.maximum(rate_at_map ** 2, 1e-30)
            H_rate_natural = (H_log_r - g_log_r) / rate_sq
            g_rate_natural = g_log_r / jnp.maximum(rate_at_map, 1e-30)
            d1, d2 = _chain_to_loc_positive(self._rate_fwd, rate_loc)
            H_rate_loc = (d1 ** 2) * H_rate_natural + d2 * g_rate_natural
            prior_prec = _prior_precision(self._prior_rate, n_g)
            curvature = H_rate_loc + prior_prec
            scale, diagnostics = curvature_to_scale(curvature)
            out["rate_scale"] = scale
            out["rate_hessian_min"] = diagnostics["hessian_min"]
            out["rate_floor_count"] = diagnostics["floor_count"]
            out["rate_curvature_floor"] = diagnostics["curvature_floor"]

        if "kappa" in self._frozen_params:
            out["kappa_scale"] = jnp.full_like(kappa_loc, jnp.nan)
        else:
            d1, d2 = _chain_to_loc_positive(self._kappa_fwd, kappa_loc)
            H_kappa_loc = (
                (d1 ** 2) * curv["H_kappa"] + d2 * curv["g_kappa"]
            )
            prior_prec = _prior_precision(self._prior_kappa, n_g)
            curvature = H_kappa_loc + prior_prec
            scale, diagnostics = curvature_to_scale(curvature)
            out["kappa_scale"] = scale
            out["kappa_hessian_min"] = diagnostics["hessian_min"]
            out["kappa_floor_count"] = diagnostics["floor_count"]
            out["kappa_curvature_floor"] = diagnostics["curvature_floor"]

        if "eta_anchor" in self._frozen_params:
            out["eta_anchor_scale"] = jnp.full_like(eta_anchor_loc, jnp.nan)
        else:
            # Identity transform: H_loc = H_natural, no chain term.
            H_eta_loc = curv["H_eta_anchor"]
            prior_prec = _prior_precision(self._prior_eta_anchor, n_g)
            curvature = H_eta_loc + prior_prec
            scale, diagnostics = curvature_to_scale(curvature)
            out["eta_anchor_scale"] = scale
            out["eta_anchor_hessian_min"] = diagnostics["hessian_min"]
            out["eta_anchor_floor_count"] = diagnostics["floor_count"]
            out["eta_anchor_curvature_floor"] = (
                diagnostics["curvature_floor"]
            )

        return out

    # ---- pack_result -----------------------------------------------------

    def pack_result(
        self,
        params: Dict[str, jnp.ndarray],
        final: FinalSweepResult,
        losses: np.ndarray,
        n_steps_run: int,
        model_config: ModelConfig,
        early_stopped: bool,
        best_loss: float,
        stopped_at_step: int,
        divergence_aborted: bool,
        global_uncertainty: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> LaplaceRunResult:
        """Assemble :class:`LaplaceRunResult` from final state."""
        params_full = self._splice_frozen(params)
        rate = self._rate_fwd(params_full["rate_loc"])
        kappa = self._kappa_fwd(params_full["kappa_loc"])
        eta_anchor = params_full["eta_anchor_loc"]
        d = self._d_fwd(params_full["d_loc"])
        n_genes = rate.shape[0]

        # Derived biological-reporting quantities (at z = 0).
        phi_anchor = jax.nn.sigmoid(eta_anchor)
        alpha = kappa * phi_anchor
        beta = kappa * (1.0 - phi_anchor)
        # ``gene_mean = rate · σ(eta_anchor)`` — the conditional mean
        # at z = 0.  Reporting-only; not consumed by Newton.
        gene_mean = rate * phi_anchor

        # Latent prior centring is zero for TSLN-Logit (z ~ N(0, Σ)).
        # Expose it as ``mu`` for parity with TSLN-Rate / NBLN where
        # ``self.mu`` is the latent log-rate prior centre.
        mu = jnp.zeros((n_genes,), dtype=rate.dtype)

        # Clamp diagnostics (stashed by final_sweep).
        stats = getattr(self, "_final_clamp_stats", None)
        if stats is None:
            a_raw_min = float("nan")
            a_raw_neg_frac = float("nan")
            a_clamp_fraction = float("nan")
            a_clamp_per_gene_arr = jnp.full(
                (n_genes,), jnp.nan, dtype=jnp.float32
            )
        else:
            a_raw_min = stats["a_raw_min"]
            a_raw_neg_frac = stats["a_raw_negative_fraction"]
            a_clamp_fraction = stats["a_clamp_fraction"]
            a_clamp_per_gene_arr = jnp.asarray(stats["a_clamp_per_gene"])

        globals_dict: Dict[str, jnp.ndarray] = {
            "W": params_full["W"],
            "d": d,
            "d_loc": params_full["d_loc"],
            # Latent prior centring (zeros for TSLN-Logit).
            "mu": mu,
            # Gene-level globals — TSLN-Logit's sampled coordinates.
            "rate_loc": params_full["rate_loc"],
            "rate": rate,
            "kappa_loc": params_full["kappa_loc"],
            "kappa": kappa,
            "eta_anchor_loc": eta_anchor,
            "eta_anchor": eta_anchor,
            # Derived for reporting / cross-variant compatibility.
            "alpha": alpha,
            "beta": beta,
            "gene_mean": gene_mean,
            # Clamp diagnostics.
            "a_raw_min": jnp.asarray(a_raw_min, dtype=jnp.float32),
            "a_raw_negative_fraction": jnp.asarray(
                a_raw_neg_frac, dtype=jnp.float32
            ),
            "a_clamp_fraction": jnp.asarray(
                a_clamp_fraction, dtype=jnp.float32
            ),
            "a_clamp_per_gene": a_clamp_per_gene_arr,
        }

        return LaplaceRunResult(
            globals=globals_dict,
            x_loc=final.latent_loc,
            eta_loc=final.eta_loc,
            final_grad_norms=final.final_grad_norms,
            losses=jnp.asarray(losses),
            n_steps_run=int(n_steps_run),
            model_config=model_config,
            early_stopped=bool(early_stopped),
            best_loss=float(best_loss),
            stopped_at_step=int(stopped_at_step),
            divergence_aborted=bool(divergence_aborted),
            global_uncertainty=(
                global_uncertainty if global_uncertainty is not None else {}
            ),
            frozen_params=self._frozen_params,
            w_prior_diagnostics=(
                self._w_prior.diagnostics(
                    params_full["W"], {}, n_constraints=1
                )
                if hasattr(self._w_prior, "diagnostics")
                else None
            ),
            # Persist the axis layout (built in init_state) so the
            # bridge in ``inference/laplace.py`` can attach it to
            # ``ScribeLaplaceResults.axis_layout``.  Trivial layout
            # under legacy / no-_other; non-trivial under decoupled
            # — but the decoupled init_state guard raises before
            # reaching here, so this is future-proofing for when the
            # math lands.
            axis_layout=self._axis_layout,
        )
