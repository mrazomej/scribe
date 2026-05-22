"""TwoState-LogNormal-Rate implementation of :class:`LaplaceObservationModel`.

Glues :mod:`scribe.laplace._newton_twostate_ln_rate` (per-cell Newton
kernel for the Poisson-Beta compound likelihood with a log-Normal
multiplicative latent on the production rate) into the protocol
expected by :func:`scribe.laplace._em.run_laplace_em`.

Mirrors :class:`NBLNObservationModel` in structure; differences are
surgical:

- **Sampled globals**: ``(mu, burst_size, k_off, W, d)`` — all
  positive (with ``_loc`` unconstrained suffixes), plus the low-rank
  ``W`` and diagonal residual ``d_loc``.
- **Derived per outer step**: ``(alpha, beta, log_r_hat)`` via the
  existing ``_twostate_reparam`` helper.  These flow into the Newton
  kernel, with ``log_r_hat`` playing the role of the prior centering
  ``mu_x`` on the latent log-rate ``x``.
- **Data log-prob**: assembled from the Newton kernel's
  ``log_marginal_sum`` field — no double quadrature.
- **W-shrinkage**: ``NoneWPrior`` only in PR-1 (the strategy interface
  transfers cleanly from NBLN but the integration is deferred to keep
  PR-1 scope tight).

For PR-1, three capture modes are supported (matching NBLN):

- **No capture** (``capture_anchor=None``, no eta cascade): ``x_only``
  Newton path.
- **Frozen-eta offset** (``"eta"`` in ``freeze_params``):
  ``x_only_offset`` path — eta is a stop_gradient'd per-cell offset.
- **Joint Newton (soft-eta cascade or biology-anchored)**: the full
  ``joint`` path with TruncN prior on η. This is the path that
  invokes :func:`laplace_newton_batch`.

See the plan in ``.claude/plans/`` §4.B.2 and the Newton kernel module
docstring for the full math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

if TYPE_CHECKING:
    from ._w_priors import WPriorStrategy

from ..core.twostate_laplace_data_init import (
    empirical_mean_from_counts,
    empirical_burst_size_from_counts,
    empirical_k_off_from_counts,
    default_d_init,
    latent_loc_init_from_counts,
    pca_loadings_init,
)
from ..models.components.likelihoods.two_state import _twostate_reparam
from ..models.config import ModelConfig
from ._em import (
    FinalSweepResult,
    InitState,
    LaplaceObservationModel,
    LaplaceRunResult,
)
from ._global_uncertainty import resolve_positive_fns
from ._newton_twostate_ln_rate import (
    _A_MIN,
    _DEFAULT_K,
    _twostate_ln_rate_factors,
    laplace_log_det_neg_H_batch,
    laplace_log_det_neg_H_batch_x_only,
    laplace_log_det_neg_H_batch_x_only_offset,
    laplace_newton_batch,
    laplace_newton_batch_x_only,
    laplace_newton_batch_x_only_offset,
    twostate_ln_rate_grad_split_batch,
    twostate_ln_rate_grad_x_only_norm_batch,
    twostate_ln_rate_grad_x_only_offset_norm_batch,
)


# Vmapped factor helper used for live-gradient log-prob and final-sweep
# clamp diagnostics.  We only need ``log_marginal`` and ``a_raw``, but
# the factor function returns the full dict — extract those two.
def _factors_log_marginal_and_a_raw(log_rate, u, alpha, beta, K):
    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, K)
    return fac["log_marginal"], fac["a_raw"]


_factors_batch = jax.vmap(
    _factors_log_marginal_and_a_raw,
    in_axes=(0, 0, None, None, None),
)


# =====================================================================
# Woodbury helpers for the MVN prior on x
# =====================================================================
#
# These are bit-identical to NBLN's helpers in _obs_nbln.py.  Could be
# lifted to _obs_woodbury.py per the plan; deferred (auditor said the
# lift is optional).


def _woodbury_quadform(
    W: jnp.ndarray, d: jnp.ndarray, diff: jnp.ndarray
) -> jnp.ndarray:
    """Compute ``diffᵀ Σ⁻¹ diff`` for ``Σ = WWᵀ + diag(d)``, batched.

    Parameters
    ----------
    W : shape ``(G, k)``.
    d : shape ``(G,)``, positive.
    diff : shape ``(*, G)`` — leading axes are broadcast.

    Returns
    -------
    jnp.ndarray, shape ``(*,)``.
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


class TwoStateLNRateObservationModel(LaplaceObservationModel):
    """TwoState-LogNormal-Rate observation channel for the Laplace driver.

    Parameters
    ----------
    capture_anchor : Optional[Tuple[float, float]]
        ``(log_M_0, sigma_M)`` for the biology-informed truncated-Normal
        prior on η_capture. ``None`` disables capture (Newton on x only).
    model_config : ModelConfig, optional
        Used to resolve ``positive_transform`` for ``mu``, ``burst_size``,
        ``k_off``. Falls back to ``softplus`` when ``None``.
    informative_priors : Optional[Dict[str, Dict[str, jnp.ndarray]]]
        Soft-cascade Gaussian priors. Valid keys:
        ``{"mu", "burst_size", "k_off", "eta"}``.
    freeze_values : Optional[Dict[str, Dict[str, jnp.ndarray]]]
        Hard-cascade point estimates for parameters in ``freeze_params``.
        Each value has at least a ``"loc"`` field; ``"eta"`` may
        additionally carry a ``"scale"`` field that's ignored under
        hard-freeze.
    freeze_params : Tuple[str, ...]
        Parameters excluded from the optax optimizer. Subset of
        ``{"mu", "burst_size", "k_off", "eta"}``. Default level-4
        cascade: ``("mu", "burst_size", "k_off")``.
    w_prior_strategy : Optional[WPriorStrategy]
        W-shrinkage strategy. PR-1 supports ``NoneWPrior`` only.
    max_step : float, default 5.0
        Newton step-size cap (passed to the kernel).
    n_quad_nodes : int, default 60
        Gauss-Legendre quadrature node count.
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
        # in ``init_state`` (where ``n_genes`` is available).  The layout
        # captures the G_obs vs G_kept split when
        # ``model_config.correlate_other_column=False`` and the data has
        # a trailing ``_other`` pooled column.  Mirrors NBLN's
        # constructor wiring; see ``scribe.laplace._axis_layout`` for
        # the detection-priority contract and the contradictory-signal
        # raise (skipped under legacy ``correlate_other_column=True``).
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
        # Defensive: PR-1 only supports NoneWPrior. Other strategies
        # transfer cleanly mathematically but are not wired in yet.
        if w_prior_strategy is not None and not isinstance(
            w_prior_strategy, NoneWPrior
        ):
            raise NotImplementedError(
                "TSLN-Rate W-shrinkage strategies beyond NoneWPrior are "
                "deferred to a follow-up PR. The strategy interface "
                "transfers from NBLN but is not yet integrated."
            )

        if capture_anchor is None:
            self._capture_anchor = None
            self._sigma_M = 1.0
        else:
            log_M0, sigma_M = capture_anchor
            self._capture_anchor = (float(log_M0), float(sigma_M))
            self._sigma_M = float(sigma_M)

        # Per-parameter ``positive_transform`` resolution.  When the
        # user passes the dict form ``positive_transform={"mean_expression":
        # "exp", ...}``, ``ModelConfig.resolve_positive_transform(name)``
        # returns the right transform name per parameter (falling back
        # to ``softplus`` for unlisted ones).  Resolve separate forward/
        # inverse callables for each of TSLN-Rate's three positive
        # gene-globals and the diagonal residual ``d``.
        from ._global_uncertainty import _JAX_POSITIVE_FNS

        def _resolve_for(internal_name: str):
            """Return the (forward, inverse) pair for one parameter."""
            if model_config is None:
                # Default to softplus across the board.
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

        self._pos_fns = {
            "mu": _resolve_for("mu"),
            "burst_size": _resolve_for("burst_size"),
            "k_off": _resolve_for("k_off"),
            "d": _resolve_for("d"),
        }
        # Back-compat: a single ``(forward, inverse)`` pair is exposed
        # via ``self._pos_forward / self._pos_inverse`` (resolves the
        # ``mu`` transform), so call sites that need any positive map
        # — e.g. ``compute_global_uncertainty`` and ``init_state`` for
        # paths that pre-date per-parameter resolution — keep working.
        self._pos_forward, self._pos_inverse = self._pos_fns["mu"]
        # Convenience per-parameter accessors (forward only — inverse
        # is rarely needed outside ``init_state``).
        self._mu_fwd = self._pos_fns["mu"][0]
        self._bs_fwd = self._pos_fns["burst_size"][0]
        self._ko_fwd = self._pos_fns["k_off"][0]
        self._d_fwd = self._pos_fns["d"][0]

        # Validate keys
        valid = {"mu", "burst_size", "k_off", "eta"}
        if informative_priors is not None:
            invalid = set(informative_priors) - valid
            if invalid:
                raise ValueError(
                    f"informative_priors has unrecognized keys {invalid}; "
                    f"valid keys are {valid}."
                )
        self._prior_mu = (
            informative_priors.get("mu")
            if informative_priors is not None
            else None
        )
        self._prior_burst_size = (
            informative_priors.get("burst_size")
            if informative_priors is not None
            else None
        )
        self._prior_k_off = (
            informative_priors.get("k_off")
            if informative_priors is not None
            else None
        )
        self._prior_eta = (
            informative_priors.get("eta")
            if informative_priors is not None
            else None
        )

        bad_frozen = set(freeze_params) - valid
        if bad_frozen:
            raise ValueError(
                f"freeze_params has invalid keys {bad_frozen}; "
                f"valid keys are {valid}."
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
        return "twostate_ln_rate"

    @property
    def uses_capture(self) -> bool:
        return (
            self._capture_anchor is not None
            or self._prior_eta is not None
            or "eta" in self._frozen_params
        )

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
        """Return ``params`` with frozen values spliced in (stop_gradient-wrapped).

        Frozen entries are excluded from the optimizer dict, so each
        method that needs them re-injects from ``self._freeze_values``.
        ``stop_gradient`` ensures they don't drift through autodiff.
        """
        out = dict(params)
        for k in self._frozen_params:
            if k == "eta":
                continue  # eta is per-cell; spliced at the loss call site
            out[f"{k}_loc"] = jax.lax.stop_gradient(
                jnp.asarray(self._freeze_values[k]["loc"])
            )
        return out

    def _reparam_from_params(
        self, params: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run forward-transform + ``_twostate_reparam``.

        Returns ``(mu_x, alpha, beta, rate)`` where ``mu_x = log(rate)``
        is the prior centering on the latent log-rate ``x`` (the analog
        of NBLN's ``mu``).  Each positive global passes through its own
        configured ``positive_transform`` — supports the per-parameter
        dict form ``positive_transform={"mu": "exp", ...}``.
        """
        mu_fwd, _ = self._pos_fns["mu"]
        bs_fwd, _ = self._pos_fns["burst_size"]
        ko_fwd, _ = self._pos_fns["k_off"]
        mu = mu_fwd(params["mu_loc"])
        bs = bs_fwd(params["burst_size_loc"])
        ko = ko_fwd(params["k_off_loc"])
        alpha, beta, rate, _eff_burst = _twostate_reparam(mu, bs, ko)
        mu_x = jnp.log(rate)
        return mu_x, alpha, beta, rate

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

        # Build the AxisLayout now that we know n_genes.  Same shape
        # contract as NBLN: under the legacy / no-_other path the
        # layout is trivial (G_kept == G_obs); under the decoupled
        # path the trailing `_other` row is excluded from Σ so W and
        # d shrink to G_kept = G_obs - 1.  All downstream code
        # (loss_fn, final_sweep, compute_global_uncertainty,
        # pack_result, compositional sampler) branches on
        # ``self._axis_layout.decoupled`` — when False, every existing
        # path runs unchanged (bit-equal to legacy).  See
        # ``scribe.laplace._axis_layout`` for the detection contract.
        from ._axis_layout import build_axis_layout
        self._axis_layout = build_axis_layout(
            n_genes=int(n_genes),
            correlate_other_column=self._correlate_other_column,
            gene_names=self._gene_names,
            has_pooled_other=self._has_pooled_other,
        )
        _layout = self._axis_layout

        # Early guard for the decoupled path (auditor rev-6 #3): the
        # decoupled-math kernels in ``loss_fn`` etc. raise
        # NotImplementedError, but ``init_state`` itself still
        # allocates W/d/latent_loc at the kept-gene shapes.  On tiny
        # / degenerate data those allocations could fail before the
        # user sees the intended NotImplementedError.  Fail fast here
        # for the clearest possible signal.
        if _layout.decoupled:
            raise NotImplementedError(
                "TSLN-Rate decoupled deviation-parameterisation math "
                "(loss_fn / final_sweep / global_uncertainty under "
                "`correlate_other_column=False` with a pooled '_other') "
                "is not yet implemented — Commit 3 of the harmonic-hare "
                "plan landed the TSLN-Rate scaffolding (AxisLayout, "
                "init shapes, pack_result); the math lands in a "
                "subsequent commit.  Until then, either pass "
                "`correlate_other_column=True` to recover legacy "
                "behaviour (with `_other` in Σ) or fit on data without "
                "a trailing '_other' column (gene_coverage == 1.0 or "
                "no gene_coverage filter)."
            )

        # Per-parameter (forward, inverse).  Supports the dict form
        # ``positive_transform={"mu": "exp", ...}`` so different
        # gene-globals can use different positive maps.
        mu_fwd, mu_inv = self._pos_fns["mu"]
        bs_fwd, bs_inv = self._pos_fns["burst_size"]
        ko_fwd, ko_inv = self._pos_fns["k_off"]
        d_fwd, d_inv = self._pos_fns["d"]

        # mu: frozen > prior > data
        if "mu" in self._frozen_params:
            # freeze_values store the UNCONSTRAINED loc; forward to constrained
            mu_pos = mu_fwd(jnp.asarray(self._freeze_values["mu"]["loc"]))
        elif self._prior_mu is not None:
            mu_pos = mu_fwd(jnp.asarray(self._prior_mu["loc"]))
        else:
            mu_pos = empirical_mean_from_counts(counts_np)
        mu_loc_init = mu_inv(mu_pos)

        # burst_size: frozen > prior > data (default 1.0)
        if "burst_size" in self._frozen_params:
            bs_pos = bs_fwd(
                jnp.asarray(self._freeze_values["burst_size"]["loc"])
            )
        elif self._prior_burst_size is not None:
            bs_pos = bs_fwd(jnp.asarray(self._prior_burst_size["loc"]))
        else:
            bs_pos = empirical_burst_size_from_counts(counts_np)
        burst_size_loc_init = bs_inv(bs_pos)

        # k_off: frozen > prior > data (default 3.0)
        if "k_off" in self._frozen_params:
            ko_pos = ko_fwd(
                jnp.asarray(self._freeze_values["k_off"]["loc"])
            )
        elif self._prior_k_off is not None:
            ko_pos = ko_fwd(jnp.asarray(self._prior_k_off["loc"]))
        else:
            ko_pos = empirical_k_off_from_counts(counts_np)
        k_off_loc_init = ko_inv(ko_pos)

        # W via PCA, d uniform (with its own potentially-distinct transform).
        # W and d live in the LATENT-COVARIANCE axis (G_kept,).  Under
        # the decoupled layout we slice the count matrix to kept genes
        # before PCA so the resulting loadings are (G_kept, K) directly
        # — slicing post-PCA would let `_other` variance leak into
        # kept-gene loadings via the SVD coupling.  See NBLN's
        # ``init_state`` for the same pattern.
        if _layout.decoupled:
            counts_for_pca = counts_np[:, _layout.kept_idx]
        else:
            counts_for_pca = counts_np
        W_init = pca_loadings_init(counts_for_pca, latent_dim=int(latent_dim))
        d_pos = default_d_init(int(_layout.G_kept))
        d_loc_init = d_inv(d_pos)

        # Per-cell η. Convention: eta = -log(nu). Default zero.
        if "eta" in self._frozen_params:
            eta_loc = jnp.asarray(self._freeze_values["eta"]["loc"])
        elif self._prior_eta is not None:
            eta_loc = jnp.asarray(self._prior_eta["loc"])
        elif self._capture_anchor is not None:
            eta_loc = jnp.full(
                (int(n_cells),),
                float(self._capture_anchor[0]),
                dtype=jnp.float32,
            )
        else:
            eta_loc = None  # no capture

        # Per-cell latent warm start.  With ``log_rate_cg = x_cg −
        # η_c``, we want ``log_rate ≈ log(u + 1)`` at the init so
        # Newton starts a few units from the MAP.  Solving for x:
        #
        #     x_init_c = log(u_c + 1) + η_c.
        #
        # When η is absent (no-capture path) η = 0 and this reduces
        # to the plain ``log(counts + 1)`` init.  Without this
        # eta-aware init, low-capture cells (large η ≈ −log p with
        # small p) start far below the MAP — empirically the cause
        # of large-gradient pathological cells in the tail of the
        # capture distribution under the frozen-eta cascade path.
        #
        # Under the **decoupled** layout the per-cell latent represents
        # the deviation ``x_dev = log(u_kept + 1) − μ_kept`` (the
        # absolute-log-rate prior centre is μ in the legacy
        # parameterisation; in the deviation parameterisation it is 0
        # and μ enters the NB log-mean directly).  We initialise at
        # the deviation form so Newton starts near the MAP for both
        # layouts.  See NBLN's ``init_state`` for the same pattern.
        if _layout.decoupled:
            kept_idx_jnp = jnp.asarray(_layout.kept_idx)
            _log_u_kept = jnp.log(jnp.asarray(counts_np[:, kept_idx_jnp]) + 1.0)
            # μ in unconstrained coord; convert to constrained log to subtract.
            _mu_pos_kept = mu_pos[kept_idx_jnp]
            latent_loc = _log_u_kept - jnp.log(jnp.maximum(_mu_pos_kept, 1e-30))[None, :]
        else:
            latent_loc = latent_loc_init_from_counts(counts_np)
        if eta_loc is not None:
            latent_loc = latent_loc + eta_loc[:, None]

        # eta_anchor: per-cell scalar.  For biology-anchored mode it's
        # the log_M0 (broadcast).  For soft-cascade it can be the prior
        # loc.  None when capture is off entirely.
        if eta_loc is not None:
            if self._capture_anchor is not None:
                eta_anchor = jnp.full(
                    (int(n_cells),),
                    float(self._capture_anchor[0]),
                    dtype=jnp.float32,
                )
            elif self._prior_eta is not None:
                eta_anchor = jnp.asarray(self._prior_eta["loc"])
            else:
                # Frozen-only path: no anchor needed (Newton doesn't
                # touch eta).  Use zeros as a placeholder.
                eta_anchor = jnp.zeros((int(n_cells),), dtype=jnp.float32)
        else:
            eta_anchor = None

        # Params dict — exclude frozen keys (excluding eta which is per-cell)
        params: Dict[str, jnp.ndarray] = {
            "W": W_init,
            "d_loc": d_loc_init,
        }
        if "mu" not in self._frozen_params:
            params["mu_loc"] = mu_loc_init
        if "burst_size" not in self._frozen_params:
            params["burst_size_loc"] = burst_size_loc_init
        if "k_off" not in self._frozen_params:
            params["k_off_loc"] = k_off_loc_init

        aux_data: Dict[str, jnp.ndarray] = {}
        return InitState(
            params=params,
            latent_loc=latent_loc,
            eta_loc=eta_loc,
            eta_anchor=eta_anchor,
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

        Mirrors NBLN's loss_fn pattern (see ``_obs_nbln.py``):
        per-cell ELBO is computed as a ``(C,)`` array; the total loss
        is ``-data_scale * sum(elbo_per_cell) - global_prior_lp``
        where ``data_scale = n_cells / batch_size`` scales the
        mini-batch up to full-data semantics.  Global priors (and the
        W-shrinkage prior) are NOT scaled — they're parameter priors,
        not per-cell likelihood contributions.

        All globals passed into the Newton kernel are
        ``stop_gradient``-wrapped per the variational-EM convention;
        gradients on globals flow only through the live ELBO terms
        recomputed at the stop-gradient'd Newton MAP.
        """
        # Decoupled-layout guard.  When the AxisLayout marks `_other`
        # as excluded from Σ (``layout.decoupled == True``), the full
        # deviation-parameterised loss (with the kept-gene MVN prior
        # on ``x_dev``, the deterministic ``_other`` log-rate, and the
        # Newton / global-uncertainty re-derivations) is tracked in
        # the harmonic-hare plan as Commit 3b.  Until that math
        # lands, fail loudly with a clear remediation message.  The
        # legacy (``layout.decoupled is False``) path runs unchanged.
        if (
            self._axis_layout is not None
            and self._axis_layout.decoupled
        ):
            raise NotImplementedError(
                "TSLN-Rate decoupled deviation-parameterisation math "
                "(loss_fn / final_sweep / global_uncertainty under "
                "`correlate_other_column=False` with a pooled '_other') "
                "is not yet implemented — Commit 3 of the harmonic-hare "
                "plan landed the TSLN-Rate scaffolding (AxisLayout, "
                "init shapes, pack_result); the math lands in a "
                "subsequent commit.  Until then, either pass "
                "`correlate_other_column=True` to recover legacy "
                "behaviour (with `_other` in Σ) or fit on data without "
                "a trailing '_other' column (gene_coverage == 1.0 or "
                "no gene_coverage filter)."
            )

        params_full = self._splice_frozen(params)
        mu_x, alpha, beta, rate = self._reparam_from_params(params_full)
        W = params_full["W"]
        d = self._d_fwd(params_full["d_loc"])
        n_quad_nodes = self._n_quad_nodes
        max_step = self._max_step

        # Stop-gradient on EVERY Newton input — globals included.  This
        # is the variational-EM convention (NBLN does the same): treat
        # the inner Newton solve as a fixed function of frozen globals,
        # then evaluate the live-globals ELBO terms at the stop-grad'd
        # Newton MAP.  Without this, autograd backprops through the
        # entire Newton scan, which (a) is the wrong derivative for the
        # marginal ELBO and (b) is much more expensive than the live-
        # term evaluations.
        latent_init_sg = jax.lax.stop_gradient(latent_init)
        eta_init_sg = (
            jax.lax.stop_gradient(eta_init) if eta_init is not None else None
        )
        mu_x_sg = jax.lax.stop_gradient(mu_x)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)
        alpha_sg = jax.lax.stop_gradient(alpha)
        beta_sg = jax.lax.stop_gradient(beta)

        if not self.uses_capture:
            x_new, final_grad, _ldet_dead, _lm_dead, _a_dead = (
                laplace_newton_batch_x_only(
                    latent_init_sg,
                    counts_batch,
                    mu_x_sg,
                    W_sg,
                    d_sg,
                    alpha_sg,
                    beta_sg,
                    n_newton,
                    damping,
                    max_step,
                    n_quad_nodes,
                )
            )
            x_new = jax.lax.stop_gradient(x_new)
            eta_new = None
            # Live-globals log det per cell.
            log_det = laplace_log_det_neg_H_batch_x_only(
                x_new, None, counts_batch, alpha, beta, W, d,
                self._sigma_M, n_quad_nodes,
            )
            # log_rate at the stop-grad'd MAP (no capture: log_rate = x).
            log_rate_for_lp = x_new
            gn_x = twostate_ln_rate_grad_x_only_norm_batch(
                x_new, counts_batch, mu_x_sg, W_sg, d_sg,
                alpha_sg, beta_sg, n_quad_nodes,
            )
            gn_blocks = {"x": gn_x}

        elif self.freezes_eta:
            eta_offset = jax.lax.stop_gradient(eta_init_sg)
            x_new, final_grad, _ldet_dead, _lm_dead, _a_dead = (
                laplace_newton_batch_x_only_offset(
                    latent_init_sg,
                    counts_batch,
                    mu_x_sg,
                    W_sg,
                    d_sg,
                    alpha_sg,
                    beta_sg,
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
                x_new, eta_offset, counts_batch, alpha, beta, W, d,
                n_quad_nodes,
            )
            # eta_offset is per-cell ``(C,)``; broadcast to ``(C, G)``.
            log_rate_for_lp = x_new - eta_offset[:, None]
            gn_x = twostate_ln_rate_grad_x_only_offset_norm_batch(
                x_new, counts_batch, mu_x_sg, W_sg, d_sg,
                alpha_sg, beta_sg, eta_offset, n_quad_nodes,
            )
            gn_blocks = {"x": gn_x}

        else:
            # Joint Newton on (x, eta) — biology-anchored or soft-cascade.
            if self._prior_eta is not None:
                sigma_M_per_cell = jnp.asarray(self._prior_eta["scale"])[
                    : counts_batch.shape[0]
                ]
            else:
                sigma_M_per_cell = jnp.full(
                    (counts_batch.shape[0],),
                    self._sigma_M,
                    dtype=jnp.float32,
                )
            sigma_M_per_cell_sg = jax.lax.stop_gradient(sigma_M_per_cell)
            eta_anchor_sg = jax.lax.stop_gradient(eta_anchor_batch)
            (
                x_new, eta_new, final_grad, _ldet_dead, _lm_dead, _a_dead,
            ) = laplace_newton_batch(
                latent_init_sg,
                eta_init_sg,
                counts_batch,
                mu_x_sg,
                W_sg,
                d_sg,
                alpha_sg,
                beta_sg,
                eta_anchor_sg,
                sigma_M_per_cell_sg,
                n_newton,
                damping,
                max_step,
                n_quad_nodes,
            )
            x_new = jax.lax.stop_gradient(x_new)
            eta_new = jax.lax.stop_gradient(eta_new)
            # Joint log_det_neg_H_batch is vmapped over
            # ``(x_map, eta_map, u, alpha, beta, W, d, sigma_M,
            # n_quad_nodes)`` with ``sigma_M`` axis 7 = per-cell.
            # Pass ``sigma_M_per_cell`` directly (NOT both scalar
            # and per-cell — that was the audit's argument-mismatch
            # finding).
            log_det = laplace_log_det_neg_H_batch(
                x_new, eta_new, counts_batch, alpha, beta, W, d,
                sigma_M_per_cell, n_quad_nodes,
            )
            log_rate_for_lp = x_new - eta_new[:, None]
            gn_x, gn_eta = twostate_ln_rate_grad_split_batch(
                x_new, eta_new, counts_batch, mu_x_sg, W_sg, d_sg,
                alpha_sg, beta_sg, eta_anchor_sg, sigma_M_per_cell_sg,
                n_quad_nodes,
            )
            gn_blocks = {"x": gn_x, "η": gn_eta}

        # Per-cell data log-prob via live-globals quadrature at the
        # stop-grad'd MAP.  Each entry of log_marginal_live is the
        # per-gene Poisson-Beta marginal; sum over genes → per-cell.
        log_marginal_live_per_gene, _ = _factors_batch(
            log_rate_for_lp, counts_batch, alpha, beta, n_quad_nodes,
        )
        data_lp_per_cell = log_marginal_live_per_gene.sum(axis=-1)  # (C,)

        # MVN prior on x via inner Woodbury — per-cell.
        diff = x_new - mu_x[None, :]
        quad_form = _woodbury_quadform(W, d, diff)  # (C,)
        log_det_sigma = _woodbury_logdet_sigma(W, d)
        n_genes = mu_x.shape[0]
        mvn_lp_per_cell = (
            -0.5 * quad_form
            - 0.5 * log_det_sigma
            - 0.5 * n_genes * jnp.log(2.0 * jnp.pi)
        )  # (C,)

        # TruncN log-prob on η when capture is soft/anchored (per-cell).
        if eta_new is not None and not self.freezes_eta:
            # Joint mode: use the live per-cell sigma_M for the prior.
            truncn = dist.TruncatedNormal(
                loc=eta_anchor_batch,
                scale=sigma_M_per_cell,
                low=0.0,
            )
            eta_lp_per_cell = truncn.log_prob(eta_new)  # (C,)
        else:
            eta_lp_per_cell = jnp.zeros_like(data_lp_per_cell)

        # Laplace correction: -½ log det(-H) per cell.
        laplace_corr_per_cell = -0.5 * log_det  # (C,)

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

        # Global-parameter priors (NOT scaled by data_scale).
        global_prior_lp = jnp.zeros(())
        for key, prior in (
            ("mu_loc", self._prior_mu),
            ("burst_size_loc", self._prior_burst_size),
            ("k_off_loc", self._prior_k_off),
        ):
            if prior is not None and key in params:
                global_prior_lp = global_prior_lp + jnp.sum(
                    dist.Normal(
                        loc=jnp.asarray(prior["loc"]),
                        scale=jnp.asarray(prior["scale"]),
                    ).log_prob(params[key])
                )

        # W-shrinkage prior on the gauge-invariant projection (NoneWPrior
        # contributes 0 in PR-1; other strategies deferred).  Same
        # convention as NBLN: prior unscaled by data_scale.
        _W_raw = params_full["W"]
        _W_for_prior = _W_raw - jnp.mean(_W_raw, axis=0, keepdims=True)
        w_aux = {
            name: params_full[name]
            for name in getattr(self._w_prior, "aux_param_names", ())
        }
        global_prior_lp = global_prior_lp + self._w_prior.log_prior(
            _W_for_prior, w_aux, n_constraints=1,
        )

        # Match NBLN's scaling: scale the per-cell ELBO sum up to full-
        # data by data_scale = n_cells / batch_size; global priors are
        # parameter priors (unscaled).
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
        # Decoupled-layout guard (mirrors ``loss_fn``).  The
        # deviation-parameterised final sweep is part of TSLN-Rate's
        # math commit (Commit 3b in the harmonic-hare plan).
        if (
            self._axis_layout is not None
            and self._axis_layout.decoupled
        ):
            raise NotImplementedError(
                "TSLN-Rate decoupled final_sweep is not yet implemented — "
                "see the loss_fn guard for the remediation. Pass "
                "`correlate_other_column=True` or omit the "
                "gene_coverage filter to use the legacy path."
            )

        params_full = self._splice_frozen(params)
        mu_x, alpha, beta, _rate = self._reparam_from_params(params_full)
        W = params_full["W"]
        d = self._d_fwd(params_full["d_loc"])
        n_iters = int(2 * n_newton)
        n_q = self._n_quad_nodes
        ms = self._max_step

        if not self.uses_capture:
            x_new, final_grad, _ldet, _lm, _a_raw = laplace_newton_batch_x_only(
                latent_loc,
                count_data,
                mu_x,
                W,
                d,
                alpha,
                beta,
                n_iters,
                damping,
                ms,
                n_q,
            )
            final_eta_loc = None
            log_rate_for_diag = x_new
        elif self.freezes_eta:
            eta_offset = jnp.asarray(self._freeze_values["eta"]["loc"])
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
                alpha,
                beta,
                eta_offset,
                n_iters,
                damping,
                ms,
                n_q,
            )
            final_eta_loc = eta_offset
            # eta_offset is per-cell ``(C,)``; broadcast to ``(C, G)``.
            log_rate_for_diag = x_new - eta_offset[:, None]
        else:
            if self._prior_eta is not None:
                sigma_M_per_cell = jnp.asarray(self._prior_eta["scale"])
            else:
                sigma_M_per_cell = jnp.full(
                    (count_data.shape[0],),
                    self._sigma_M,
                    dtype=jnp.float32,
                )
            (
                x_new,
                eta_new,
                final_grad,
                _ldet,
                _lm,
                _a_raw,
            ) = laplace_newton_batch(
                latent_loc,
                eta_loc,
                count_data,
                mu_x,
                W,
                d,
                alpha,
                beta,
                eta_anchor,
                sigma_M_per_cell,
                n_iters,
                damping,
                ms,
                n_q,
            )
            final_eta_loc = eta_new
            log_rate_for_diag = x_new - eta_new[:, None]

        # Compute clamp diagnostics at the final MAP (count_data is on
        # hand here; not the case in pack_result). Stashed on self so
        # pack_result can pick them up without reaching for count_data.
        _, a_raw_per_cell = _factors_batch(
            log_rate_for_diag, count_data, alpha, beta, n_q,
        )
        # a_raw_per_cell shape: (C, G)
        a_raw_flat = a_raw_per_cell.reshape(-1)
        a_raw_min_val = float(jnp.min(a_raw_per_cell))
        a_raw_neg_frac = float(jnp.mean((a_raw_flat < 0.0).astype(jnp.float32)))
        a_clamp_frac = float(
            jnp.mean((a_raw_flat < _A_MIN).astype(jnp.float32))
        )
        # Per-gene clamp activation rate.
        a_clamp_per_gene = jnp.mean(
            (a_raw_per_cell < _A_MIN).astype(jnp.float32), axis=0
        )
        self._final_clamp_stats = {
            "a_raw_min": a_raw_min_val,
            "a_raw_negative_fraction": a_raw_neg_frac,
            "a_clamp_fraction": a_clamp_frac,
            "a_clamp_per_gene": np.asarray(a_clamp_per_gene),
        }
        # User-facing warning if clamp activated on >5% of (cell, gene)
        # entries — matches the threshold in the plan §4.A.3.
        if a_clamp_frac > 0.05:
            import logging
            logging.getLogger(__name__).warning(
                "TSLN-Rate curvature clamp activated on %.1f%% of "
                "(cell, gene) entries (threshold 5%%). The Laplace "
                "approximation is locally prior-dominated for those "
                "entries; posterior credible intervals on the affected "
                "genes should not be interpreted at face value. See "
                "result.a_clamp_per_gene for the per-gene breakdown.",
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
        """Diagonal-Hessian Laplace approximation on the gene globals.

        Builds a per-cell ELBO closure parametrized by
        ``(mu_loc, burst_size_loc, k_off_loc)``, with everything else
        (``W``, ``d``, the per-cell Newton MAP) ``stop_gradient``'d at
        the post-final-sweep values.  For each unfrozen gene-global
        parameter, computes the per-gene diagonal of the negative-
        ELBO Hessian via ``jnp.diag(jax.hessian(...))``, adds the
        informative-prior precision (if a soft cascade was supplied),
        and converts the diagonal curvature to a Normal posterior
        scale via :func:`curvature_to_scale`.

        Simplifications relative to NBLN's compute_global_uncertainty
        ---------------------------------------------------------------
        - **Marginal**, not profiled Hessian.  NBLN applies the Schur
          complement ``H_{θθ} − H_{θx} H_{xx}^{-1} H_{xθ}`` to absorb
          how the per-cell MAPs shift with θ; here we ignore that
          correction.  Resulting scales are slightly over-confident.
          A profiled version is a phase-2 follow-up.
        - **Diagonal**, no cross-parameter Hessian blocks.  Per-gene
          ``∂² L / ∂θ_g²`` is computed for each parameter
          independently; cross terms like ``∂² L / ∂mu_g ∂burst_size_g``
          are ignored.  Mirrors NBLN's diagonal-Σ approximation for
          ``r_scale``.
        - **NaN sentinels for frozen globals**, matching NBLN: the
          bridge replaces these with moment-matched values from
          ``cascade_source.get_posterior_samples()`` when a cascade
          source is present.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Keys: ``mu_scale``, ``burst_size_scale``, ``k_off_scale``
            (each shape ``(G,)`` or NaN-filled for frozen) plus
            ``mu_loc``, ``burst_size_loc``, ``k_off_loc`` mirrors of
            the optimized unconstrained values.  Diagnostic fields
            ``*_hessian_min``, ``*_floor_count``, ``*_curvature_floor``
            from ``curvature_to_scale`` are also surfaced for
            inspection.
        """
        # Decoupled-layout guard (mirrors ``loss_fn``).
        if (
            self._axis_layout is not None
            and self._axis_layout.decoupled
        ):
            raise NotImplementedError(
                "TSLN-Rate decoupled compute_global_uncertainty is not "
                "yet implemented — the deviation-parameterised Schur "
                "re-derivation by capture mode lands in TSLN-Rate's "
                "math commit.  Pass `correlate_other_column=True` or "
                "omit the gene_coverage filter to use the legacy path."
            )

        from ._global_uncertainty import curvature_to_scale

        params_full = self._splice_frozen(params)
        x_map_sg = jax.lax.stop_gradient(latent_loc)
        eta_map_sg = (
            jax.lax.stop_gradient(eta_loc) if eta_loc is not None else None
        )
        counts = jnp.asarray(count_data)
        n_cells = counts.shape[0]
        n_quad_nodes = self._n_quad_nodes

        # log_rate at the MAP given the eta path.
        if eta_map_sg is None:
            log_rate_at_map = x_map_sg
        else:
            # eta_map is (C,); broadcast to (C, G).
            log_rate_at_map = x_map_sg - eta_map_sg[:, None]

        W_sg = jax.lax.stop_gradient(params_full["W"])
        d_sg = jax.lax.stop_gradient(self._d_fwd(params_full["d_loc"]))

        # Capture the soft-cascade Normal-prior precision (per gene) for
        # each parameter — added to the data-side Hessian diagonal so
        # the post-fit Laplace scale reflects both data and prior.
        def _prior_precision(prior: Optional[Dict[str, jnp.ndarray]]):
            if prior is None:
                return jnp.zeros((mu_x.shape[0],), dtype=jnp.float32)
            return 1.0 / jnp.square(jnp.asarray(prior["scale"]))

        # Recompute the latent prior centering (log r_hat) inside the
        # closure so its dependence on (mu_loc, burst_size_loc, k_off_loc)
        # is autodiffed.
        def neg_log_post(mu_loc_v, burst_size_loc_v, k_off_loc_v):
            mu_pos = self._mu_fwd(mu_loc_v)
            bs_pos = self._bs_fwd(burst_size_loc_v)
            ko_pos = self._ko_fwd(k_off_loc_v)
            alpha_v, beta_v, rate_v, _eff = _twostate_reparam(
                mu_pos, bs_pos, ko_pos
            )
            mu_x_v = jnp.log(rate_v)

            # Data log-prob via the K-axis quadrature reduction at the
            # stop_gradient'd MAP.
            log_marg_per_gene, _ = _factors_batch(
                log_rate_at_map, counts, alpha_v, beta_v, n_quad_nodes,
            )
            data_lp = jnp.sum(log_marg_per_gene)

            # MVN prior on x at the MAP (prior center = mu_x_v).
            diff = x_map_sg - mu_x_v[None, :]
            quad = _woodbury_quadform(W_sg, d_sg, diff)
            log_det_sigma = _woodbury_logdet_sigma(W_sg, d_sg)
            n_genes = mu_x_v.shape[0]
            mvn_lp = (
                -0.5 * jnp.sum(quad)
                - 0.5 * n_cells * (
                    log_det_sigma + n_genes * jnp.log(2.0 * jnp.pi)
                )
            )

            return -(data_lp + mvn_lp)

        mu_loc = params_full["mu_loc"]
        bs_loc = params_full["burst_size_loc"]
        ko_loc = params_full["k_off_loc"]

        # Resolve mu_x just so the prior-precision helper has a shape
        # reference; not used downstream of the helper itself.
        mu_x = jnp.log(
            _twostate_reparam(
                self._mu_fwd(mu_loc),
                self._bs_fwd(bs_loc),
                self._ko_fwd(ko_loc),
            )[2]
        )

        out: Dict[str, jnp.ndarray] = {
            "mu_loc": mu_loc,
            "burst_size_loc": bs_loc,
            "k_off_loc": ko_loc,
        }

        # ---- Hand-derived global curvature ------------------------------
        # See paper/_two_state_promoter.qmd
        #   §sec-twostate-tsln-rate-global-uncertainty
        # for the full derivation.  Strategy:
        #
        #   (1) Compute the data-side NLP gradient + Hessian in the
        #       (α, β) natural basis via the closed-form softmax-moment
        #       reductions in ``global_curvature_rate_summed`` (no
        #       autodiff, no transient (C, G, G) tensor).
        #   (2) Add the MVN prior contribution to the (log r, log r)
        #       diagonal entry (and the log-r gradient at the MAP).
        #       The MVN block is independent of (α, β) so the 3×3
        #       Hessian in (α, β, log r) is block-diagonal.
        #   (3) Per-gene Faà di Bruno chain through ``_twostate_reparam``
        #       and the configured ``pos_forward`` transforms to
        #       extract the (mu_loc, burst_size_loc, k_off_loc) Hessian
        #       diagonal.  Implemented via the quadratic-substitution
        #       trick: build a per-gene scalar
        #
        #           f̃(loc) = g_φ^T δ + 0.5 δ^T H_φ δ,
        #           δ = φ(loc) - φ(loc_MAP),
        #
        #       and call ``jax.hessian(f̃)(loc_MAP)`` — its diagonal
        #       equals the Faà di Bruno expansion of ∂²f/∂loc² exactly.
        #       ``jax`` handles the ``_twostate_reparam`` clamps via
        #       their subgradients.
        from ._newton_twostate_ln_rate import global_curvature_rate_summed
        from ._global_uncertainty import (
            woodbury_inv_diag, woodbury_apply_inv,
        )

        mu_at_map = self._mu_fwd(mu_loc)
        bs_at_map = self._bs_fwd(bs_loc)
        ko_at_map = self._ko_fwd(ko_loc)
        alpha_at_map, beta_at_map, rate_at_map, _ = _twostate_reparam(
            mu_at_map, bs_at_map, ko_at_map,
        )
        log_rate_at_map = jnp.log(jnp.maximum(rate_at_map, 1e-30))

        # eta_cap for the data-side helper (zeros under no-capture).
        if eta_map_sg is None:
            eta_cap_for_curv = jnp.zeros(
                (n_cells,), dtype=mu_at_map.dtype,
            )
        else:
            eta_cap_for_curv = eta_map_sg

        curv = global_curvature_rate_summed(
            x_map=x_map_sg,
            counts=counts,
            alpha=alpha_at_map,
            beta=beta_at_map,
            eta_cap=eta_cap_for_curv,
            n_quad_nodes=n_quad_nodes,
        )

        # MVN prior contribution in the (log r) axis:
        #   mvn_lp = -1/2 Σ_c (x_c - mu_x)^T Σ^{-1} (x_c - mu_x), mu_x = log r
        #   ∂(-mvn_lp)/∂log r_g  = -Σ_c [Σ^{-1}(x_c - log r)]_g
        #                        = -[Σ^{-1} (Σ_c x_c - C·log r)]_g
        #   ∂²(-mvn_lp)/∂log r_g²= +C (Σ^{-1})_{gg}
        sigma_inv_diag = woodbury_inv_diag(W_sg, d_sg)            # (G,)
        H_logr_logr_mvn = float(n_cells) * sigma_inv_diag         # (G,)
        sum_diff = jnp.sum(x_map_sg, axis=0) - float(n_cells) * log_rate_at_map
        g_logr_mvn = -woodbury_apply_inv(W_sg, d_sg, sum_diff)    # (G,)

        # Stitch per-gene φ-basis gradient and 3×3 Hessian.
        #   φ = (α, β, log r)
        #   data fills (α, β) block; MVN fills (log r) entry only.
        g_phi = jnp.stack(
            [curv["g_alpha"], curv["g_beta"], g_logr_mvn], axis=-1,
        )                                                          # (G, 3)
        # H_phi shape (G, 3, 3) — symmetric.
        zero_g = jnp.zeros_like(curv["g_alpha"])
        H_phi = jnp.stack(
            [
                jnp.stack([curv["H_aa"], curv["H_ab"], zero_g], axis=-1),
                jnp.stack([curv["H_ab"], curv["H_bb"], zero_g], axis=-1),
                jnp.stack([zero_g,       zero_g,       H_logr_logr_mvn],
                          axis=-1),
            ],
            axis=-2,
        )                                                          # (G, 3, 3)
        phi_at_map = jnp.stack(
            [alpha_at_map, beta_at_map, log_rate_at_map], axis=-1,
        )                                                          # (G, 3)

        # Capture the configured pos_forward closures for the per-gene
        # chain.  All three transforms are scalar→scalar; the chain
        # function evaluates them on a single gene's loc triple.
        mu_fwd, bs_fwd, ko_fwd = self._mu_fwd, self._bs_fwd, self._ko_fwd

        def _per_gene_chain_diag(
            loc_arr: jnp.ndarray,                # (3,) = (mu_loc, bs_loc, ko_loc)
            phi_map_g: jnp.ndarray,              # (3,)
            g_phi_g: jnp.ndarray,                # (3,)
            H_phi_g: jnp.ndarray,                # (3, 3)
        ) -> jnp.ndarray:
            """Faà di Bruno via quadratic substitution; returns (3,) diag."""
            def f_local(loc_v):
                mu_v = mu_fwd(loc_v[0])
                bs_v = bs_fwd(loc_v[1])
                ko_v = ko_fwd(loc_v[2])
                a_v, b_v, r_v, _ = _twostate_reparam(mu_v, bs_v, ko_v)
                phi_v = jnp.stack(
                    [a_v, b_v, jnp.log(jnp.maximum(r_v, 1e-30))]
                )
                delta = phi_v - phi_map_g
                return jnp.dot(g_phi_g, delta) + 0.5 * jnp.dot(
                    delta, jnp.dot(H_phi_g, delta),
                )
            return jnp.diag(jax.hessian(f_local)(loc_arr))

        loc_at_map = jnp.stack([mu_loc, bs_loc, ko_loc], axis=-1)  # (G, 3)
        H_loc_diag = jax.vmap(_per_gene_chain_diag)(
            loc_at_map, phi_at_map, g_phi, H_phi,
        )                                                           # (G, 3)
        H_mu_loc = H_loc_diag[:, 0]
        H_bs_loc = H_loc_diag[:, 1]
        H_ko_loc = H_loc_diag[:, 2]

        for name, hess_diag, loc_val, prior in (
            ("mu",         H_mu_loc, mu_loc, self._prior_mu),
            ("burst_size", H_bs_loc, bs_loc, self._prior_burst_size),
            ("k_off",      H_ko_loc, ko_loc, self._prior_k_off),
        ):
            if name in self._frozen_params:
                # Frozen: NaN sentinels (mirrors NBLN convention; the
                # bridge moment-matches from cascade_source when one
                # is present).
                out[f"{name}_scale"] = jnp.full_like(loc_val, jnp.nan)
                continue

            prior_prec = _prior_precision(prior)
            curvature = hess_diag + prior_prec
            scale, diagnostics = curvature_to_scale(curvature)
            out[f"{name}_scale"] = scale
            out[f"{name}_hessian_min"] = diagnostics["hessian_min"]
            out[f"{name}_floor_count"] = diagnostics["floor_count"]
            out[f"{name}_curvature_floor"] = diagnostics["curvature_floor"]

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
        mu = self._mu_fwd(params_full["mu_loc"])
        burst_size = self._bs_fwd(params_full["burst_size_loc"])
        k_off = self._ko_fwd(params_full["k_off_loc"])
        alpha, beta, rate, _eff = _twostate_reparam(mu, burst_size, k_off)
        d = self._d_fwd(params_full["d_loc"])

        # Read clamp diagnostics that final_sweep stashed on self.  If
        # final_sweep was never called (defensive — should not happen in
        # the run_laplace_em flow), fall back to NaN sentinels so
        # downstream consumers can still detect "diagnostics absent".
        stats = getattr(self, "_final_clamp_stats", None)
        if stats is None:
            a_raw_min = float("nan")
            a_raw_neg_frac = float("nan")
            a_clamp_fraction = float("nan")
            a_clamp_per_gene_arr = jnp.full(
                (mu.shape[0],), jnp.nan, dtype=jnp.float32
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
            "mu_loc": params_full["mu_loc"],
            "mu": mu,
            "burst_size_loc": params_full["burst_size_loc"],
            "burst_size": burst_size,
            "k_off_loc": params_full["k_off_loc"],
            "k_off": k_off,
            "alpha": alpha,
            "beta": beta,
            "r_hat": rate,
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
            # for legacy fits; non-trivial under decoupled (but the
            # decoupled math raises before reaching here, so this is
            # mainly future-proofing for Commit 3b).
            axis_layout=self._axis_layout,
        )
