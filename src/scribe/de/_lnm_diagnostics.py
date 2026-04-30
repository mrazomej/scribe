"""Diagnostics for logistic-normal multinomial (LNM) models.

Moment-matching tools bridge between compositional latent structure and
interpretable per-gene count summaries used in negative-binomial models.
"""

from typing import Dict, Optional

import jax.numpy as jnp
from jax import random

from ..stats.distributions import LowRankLogisticNormal


def effective_per_gene_nb(
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: Optional[jnp.ndarray],
    r_T: float,
    p: float,
    n_mc_samples: int = 2048,
    key: Optional[random.PRNGKey] = None,
    floor_var: float = 1e-12,
    reference_idx: int = -1,
) -> Dict[str, jnp.ndarray]:
    """Moment-match an effective per-gene negative binomial from LNM parameters.

    Under ``lnm`` / ``lnmvcp``, gene counts conditional on library size and
    composition follow a multinomial, and the marginal per-gene distribution is
    not negative-binomial.  This routine constructs a crude *effective* NB
    ``(r_{\\mathrm{eff},g}, p_{\\mathrm{eff},g})`` per gene by matching the
    first two moments of

        u_g = u_T · ρ_g,

    where u_T ~ NB(r_T, p) is the total UMI count and ρ is an independent
    simplex draw from the low-rank logistic-normal induced by (mu, W, d) in
    ALR space.

    Using independence of u_T and ρ (same as in the generative model):

        ⟨u_g⟩ = ⟨u_T⟩ · ⟨ρ_g⟩
        Var(u_g) = ⟨ρ_g⟩² · Var(u_T)
            + ⟨u_T⟩² · Var(ρ_g)
            + Var(u_T) · Var(ρ_g)

    Moments of u_T use the "probs" parameterization of NumPyro’s negative
    binomial:

        ⟨u_T⟩ = r_T · p / (1 - p)
        Var(u_T) = r_T · p / (1 - p)²

    Effective NB parameters invert the mean–variance mapping (only valid when
    Var(u_g) > ⟨u_g⟩):

        p_eff,g = 1 - ⟨u_g⟩ / Var(u_g)
        r_eff,g = ⟨u_g⟩² / (Var(u_g) - ⟨u_g⟩)


    Parameters
    ----------
    mu : jnp.ndarray
        ALR population mean, shape ``(G-1,)``.
    W : jnp.ndarray
        Low-rank factor, shape ``(G-1, k)``.
    d : jnp.ndarray or None
        Diagonal ALR variance, shape ``(G-1,)``.  If ``None``, ``floor_var``
        fills the diagonal for sampling (low-rank singularity path).
    r_T : float
        Total-count NB dispersion / concentration ``total_count``.
    p : float
        Base total-count NB success probability in ``(0, 1)`` (NumPyro
        ``probs``, before VCP modulation when applicable).
    n_mc_samples : int, default=2048
        Monte Carlo sample size for :math:`\\rho` (simplex draws).
    key : jax.random.PRNGKey, optional
        Random key; default ``PRNGKey(0)``.
    floor_var : float, default=1e-12
        Minimum diagonal entry passed to :class:`LowRankLogisticNormal` when
        ``d`` is ``None`` or contains very small values.
    reference_idx : int, default=-1
        ALR reference simplex index for composition sampling (``-1`` = last
        gene), aligned with :attr:`ModelConfig.alr_reference_idx`.

    Returns
    -------
    dict of str to jnp.ndarray
        * **mean** — shape ``(G,)``, :math:`\\mathbb{E}[u_g]`.
        * **var** — shape ``(G,)``, :math:`\\mathrm{Var}(u_g)`.
        * **r_eff** — shape ``(G,)``, effective ``total_count``; ``NaN``
          where ``var <= mean``.
        * **p_eff** — shape ``(G,)``, effective ``probs``; ``NaN`` where
          ``var <= mean``.
        * **well_defined** — shape ``(G,)``, boolean mask
          ``var > mean`` (strict NB moment-matching regime).

    Notes
    -----
    This diagnostic ignores multinomial sampling variability conditional on
    :math:`(u_T, \\rho)`; it approximates :math:`u_g \\approx u_T \\rho_g`
    for second-moment matching, consistent with a product-of-independent-means
    shortcut often used for quick calibration.
    """
    if key is None:
        key = random.PRNGKey(0)

    mu_j = jnp.asarray(mu)
    W_j = jnp.asarray(W)
    g1 = int(mu_j.shape[0])

    if d is None:
        cov_diag = jnp.full((g1,), floor_var, dtype=mu_j.dtype)
    else:
        cov_diag = jnp.maximum(jnp.asarray(d), floor_var)

    # Monte Carlo moments for composition — independent of u_T in the model.
    dist_ln = LowRankLogisticNormal(
        loc=mu_j,
        cov_factor=W_j,
        cov_diag=cov_diag,
        reference_idx=reference_idx,
    )
    rho = dist_ln.sample(key, (int(n_mc_samples),))  # (S, G)

    e_rho = jnp.mean(rho, axis=0)
    v_rho = jnp.var(rho, axis=0, ddof=1)

    p_safe = float(jnp.clip(jnp.asarray(p, dtype=mu_j.dtype), 1e-9, 1.0 - 1e-9))
    r_T_f = float(r_T)

    # NB(probs): E[u_T] and Var[u_T] from totals prior.
    e_u_T = jnp.asarray(r_T_f * p_safe / (1.0 - p_safe), dtype=mu_j.dtype)
    v_u_T = jnp.asarray(
        r_T_f * p_safe / ((1.0 - p_safe) ** 2), dtype=mu_j.dtype
    )

    mean_u = e_u_T * e_rho
    var_u = (
        jnp.square(e_rho) * v_u_T + jnp.square(e_u_T) * v_rho + v_u_T * v_rho
    )

    # Invert NB(probs) moments: var > mean required for overdispersion.
    well_defined = var_u > mean_u
    denom = var_u - mean_u
    p_eff = jnp.where(well_defined, 1.0 - mean_u / var_u, jnp.nan)
    r_eff = jnp.where(
        well_defined,
        jnp.square(mean_u) / jnp.maximum(denom, 1e-30),
        jnp.nan,
    )

    return {
        "mean": mean_u,
        "var": var_u,
        "r_eff": r_eff,
        "p_eff": p_eff,
        "well_defined": well_defined,
    }
