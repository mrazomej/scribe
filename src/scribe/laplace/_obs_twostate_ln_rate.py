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
    ):
        self._max_step = float(max_step)
        self._n_quad_nodes = int(n_quad_nodes)

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

        self._pos_forward, self._pos_inverse = resolve_positive_fns(
            model_config
        )

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
        of NBLN's ``mu``).
        """
        mu = self._pos_forward(params["mu_loc"])
        bs = self._pos_forward(params["burst_size_loc"])
        ko = self._pos_forward(params["k_off_loc"])
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

        # mu: frozen > prior > data
        if "mu" in self._frozen_params:
            mu_pos = jnp.asarray(self._freeze_values["mu"]["loc"])
            # freeze_values store the UNCONSTRAINED loc; forward to constrained
            mu_pos = self._pos_forward(mu_pos)
        elif self._prior_mu is not None:
            mu_pos = self._pos_forward(jnp.asarray(self._prior_mu["loc"]))
        else:
            mu_pos = empirical_mean_from_counts(counts_np)
        mu_loc_init = self._pos_inverse(mu_pos)

        # burst_size: frozen > prior > data (default 1.0)
        if "burst_size" in self._frozen_params:
            bs_pos = self._pos_forward(
                jnp.asarray(self._freeze_values["burst_size"]["loc"])
            )
        elif self._prior_burst_size is not None:
            bs_pos = self._pos_forward(
                jnp.asarray(self._prior_burst_size["loc"])
            )
        else:
            bs_pos = empirical_burst_size_from_counts(counts_np)
        burst_size_loc_init = self._pos_inverse(bs_pos)

        # k_off: frozen > prior > data (default 3.0)
        if "k_off" in self._frozen_params:
            ko_pos = self._pos_forward(
                jnp.asarray(self._freeze_values["k_off"]["loc"])
            )
        elif self._prior_k_off is not None:
            ko_pos = self._pos_forward(jnp.asarray(self._prior_k_off["loc"]))
        else:
            ko_pos = empirical_k_off_from_counts(counts_np)
        k_off_loc_init = self._pos_inverse(ko_pos)

        # W via PCA, d uniform
        W_init = pca_loadings_init(counts_np, latent_dim=int(latent_dim))
        d_pos = default_d_init(int(n_genes))
        d_loc_init = self._pos_inverse(d_pos)

        # Per-cell latent warm start
        latent_loc = latent_loc_init_from_counts(counts_np)

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
        """Negative Laplace ELBO on one mini-batch."""
        params_full = self._splice_frozen(params)
        mu_x, alpha, beta, rate = self._reparam_from_params(params_full)
        W = params_full["W"]
        d = self._pos_forward(params_full["d_loc"])
        n_quad_nodes = self._n_quad_nodes
        max_step = self._max_step

        # Stop-gradient on the Newton inputs so the inner loop is
        # treated as a fixed function of the live globals.
        latent_init_sg = jax.lax.stop_gradient(latent_init)
        eta_init_sg = (
            jax.lax.stop_gradient(eta_init) if eta_init is not None else None
        )

        if not self.uses_capture:
            # x-only path
            (
                x_new,
                final_grad,
                log_det_neg_H,
                log_marginal_sum,
                _a_raw_min,
            ) = laplace_newton_batch_x_only(
                latent_init_sg,
                counts_batch,
                mu_x,
                W,
                d,
                alpha,
                beta,
                n_newton,
                damping,
                max_step,
                n_quad_nodes,
            )
            # Re-evaluate log_marginal at the live globals (the Newton
            # iterates above stop-gradient; we need gradients to flow
            # through the determinant + log-prob from the current
            # globals).
            log_det_live = laplace_log_det_neg_H_batch_x_only(
                x_new,
                None,  # eta_map
                counts_batch,
                alpha,
                beta,
                W,
                d,
                self._sigma_M,
                n_quad_nodes,
            )
            # Data log-prob: recompute at stop_gradient'd Newton MAP with
            # LIVE globals (alpha, beta) so that the ELBO gradient on
            # (mu, burst_size, k_off) flows through this term.  Without
            # this recomputation the kernel's ``log_marginal_sum`` carries
            # only the stop_gradient'd alpha/beta from the Newton call,
            # so soft-cascade and uncascaded fits would have zero data-
            # side gradient on the gene globals.  Mirrors NBLN's pattern
            # of computing the data log-prob via LogMeanNegativeBinomial
            # at the post-Newton MAP with live globals.
            log_rate_for_lp = jax.lax.stop_gradient(x_new)
            log_marginal_live, _ = _factors_batch(
                log_rate_for_lp, counts_batch, alpha, beta, n_quad_nodes,
            )
            data_log_prob = jnp.sum(log_marginal_live)
            eta_new = None
            gn_x = twostate_ln_rate_grad_x_only_norm_batch(
                x_new,
                counts_batch,
                mu_x,
                W,
                d,
                alpha,
                beta,
                n_quad_nodes,
            )
            gn_blocks = {"x": gn_x}

        elif self.freezes_eta:
            # x-only-with-offset path (eta is stop_gradient'd per-cell)
            eta_offset = jax.lax.stop_gradient(eta_init_sg)
            (
                x_new,
                final_grad,
                log_det_neg_H,
                log_marginal_sum,
                _a_raw_min,
            ) = laplace_newton_batch_x_only_offset(
                latent_init_sg,
                counts_batch,
                mu_x,
                W,
                d,
                alpha,
                beta,
                eta_offset,
                n_newton,
                damping,
                max_step,
                n_quad_nodes,
            )
            # Live-globals log det evaluation
            log_det_live_per_cell = laplace_log_det_neg_H_batch_x_only_offset(
                x_new,
                eta_offset,
                counts_batch,
                alpha,
                beta,
                W,
                d,
                n_quad_nodes,
            )
            log_det_live = log_det_live_per_cell
            # Live-gradient data log-prob at stop_gradient'd MAP.
            log_rate_for_lp = (
                jax.lax.stop_gradient(x_new) - eta_offset
            )
            log_marginal_live, _ = _factors_batch(
                log_rate_for_lp, counts_batch, alpha, beta, n_quad_nodes,
            )
            data_log_prob = jnp.sum(log_marginal_live)
            eta_new = eta_offset  # unchanged
            gn_x = twostate_ln_rate_grad_x_only_offset_norm_batch(
                x_new,
                counts_batch,
                mu_x,
                W,
                d,
                alpha,
                beta,
                eta_offset,
                n_quad_nodes,
            )
            gn_blocks = {"x": gn_x}

        else:
            # Joint Newton on (x, eta) — biology-anchored or soft-cascade.
            if self._prior_eta is not None:
                # Per-cell sigma_M from the soft-cascade prior.
                sigma_M_per_cell = jnp.asarray(self._prior_eta["scale"])[
                    : counts_batch.shape[0]
                ]
            else:
                sigma_M_per_cell = jnp.full(
                    (counts_batch.shape[0],), self._sigma_M, dtype=jnp.float32
                )
            (
                x_new,
                eta_new,
                final_grad,
                log_det_neg_H,
                log_marginal_sum,
                _a_raw_min,
            ) = laplace_newton_batch(
                latent_init_sg,
                eta_init_sg,
                counts_batch,
                mu_x,
                W,
                d,
                alpha,
                beta,
                eta_anchor_batch,
                sigma_M_per_cell,
                n_newton,
                damping,
                max_step,
                n_quad_nodes,
            )
            log_det_live = laplace_log_det_neg_H_batch(
                x_new,
                eta_new,
                counts_batch,
                alpha,
                beta,
                W,
                d,
                self._sigma_M,
                sigma_M_per_cell,
                n_quad_nodes,
            )
            # Live-gradient data log-prob at stop_gradient'd MAP.
            log_rate_for_lp = (
                jax.lax.stop_gradient(x_new) - jax.lax.stop_gradient(eta_new)
            )
            log_marginal_live, _ = _factors_batch(
                log_rate_for_lp, counts_batch, alpha, beta, n_quad_nodes,
            )
            data_log_prob = jnp.sum(log_marginal_live)
            gn_x, gn_eta = twostate_ln_rate_grad_split_batch(
                x_new,
                eta_new,
                counts_batch,
                mu_x,
                W,
                d,
                alpha,
                beta,
                eta_anchor_batch,
                sigma_M_per_cell,
                n_quad_nodes,
            )
            gn_blocks = {"x": gn_x, "η": gn_eta}

        # MVN prior on x via inner Woodbury
        diff = x_new - mu_x[None, :]
        quad_form = _woodbury_quadform(W, d, diff)
        log_det_sigma = _woodbury_logdet_sigma(W, d)
        n_genes = mu_x.shape[0]
        mvn_log_prob = -0.5 * jnp.sum(quad_form) - 0.5 * counts_batch.shape[
            0
        ] * (log_det_sigma + n_genes * jnp.log(2.0 * jnp.pi))

        # Optional TruncN log-prob on eta (when capture is soft/anchored)
        if eta_new is not None and not self.freezes_eta:
            truncn = dist.TruncatedNormal(
                loc=eta_anchor_batch,
                scale=self._sigma_M,
                low=0.0,
            )
            eta_log_prob = jnp.sum(truncn.log_prob(eta_new))
        else:
            eta_log_prob = jnp.asarray(0.0)

        # Soft-cascade Normal priors on the unconstrained loc params.
        prior_log_prob = jnp.asarray(0.0)
        for key, prior in (
            ("mu_loc", self._prior_mu),
            ("burst_size_loc", self._prior_burst_size),
            ("k_off_loc", self._prior_k_off),
        ):
            if prior is not None and key in params:
                prior_log_prob = prior_log_prob + jnp.sum(
                    dist.Normal(
                        loc=jnp.asarray(prior["loc"]),
                        scale=jnp.asarray(prior["scale"]),
                    ).log_prob(params[key])
                )

        # ELBO = data + MVN + eta + soft-priors - 0.5 * log det(-H)
        elbo = (
            data_log_prob
            + mvn_log_prob
            + eta_log_prob
            + prior_log_prob
            - 0.5 * jnp.sum(log_det_live)
        )
        loss = -elbo / float(data_scale)
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
        mu_x, alpha, beta, _rate = self._reparam_from_params(params_full)
        W = params_full["W"]
        d = self._pos_forward(params_full["d_loc"])
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
            log_rate_for_diag = x_new - eta_offset
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
        mu = self._pos_forward(params_full["mu_loc"])
        burst_size = self._pos_forward(params_full["burst_size_loc"])
        k_off = self._pos_forward(params_full["k_off_loc"])
        alpha, beta, rate, _eff = _twostate_reparam(mu, burst_size, k_off)
        d = self._pos_forward(params_full["d_loc"])

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
        )
