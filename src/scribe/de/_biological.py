"""Biological-level differential expression from posterior NB parameters.

This module computes differential expression metrics in the space of
the **biological** (denoised) Negative Binomial distribution, bypassing
the compositional simplex entirely.  For each gene the model learns
parameters that define the NB distribution of true transcript counts
*before* capture loss and dropout.  Comparing these parameters between
two conditions yields three complementary metrics:

1. **Biological log-fold change (LFC)** — shift in the NB mean ``mu_g``.
2. **Log-variance ratio (LVR)** — shift in the NB variance.
3. **Gamma Jeffreys divergence** — symmetrised KL divergence between
   the latent Gamma rate distributions that generate the NB counts
   via the Poisson-Gamma representation.

All metrics are computed **per gene, per posterior sample** and then
summarised into posterior means, SDs, local false sign rates, and
probabilities of exceeding user-specified practical significance
thresholds.  Because the posterior samples of ``(r, p)`` (or the
native ``(mu, phi)``) are already available from the CLR differencing
pipeline, these computations come at near-zero additional cost.

Unlike CLR-based DE, these metrics are free of compositional closure:
a gene whose transcription does not change will have LFC ≈ 0
regardless of what other genes do.

Parameterization-aware computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function automatically selects the most numerically stable
computation path based on which posterior samples are supplied:

- **``mean_odds``** (``mu`` and ``phi`` available): All quantities are
  derived from native samples.  The Gamma rate is ``beta = 1/phi``
  (no subtraction), ``mu`` is used directly for LFC, and the
  variance is ``mu * (1 + phi)``.  This completely avoids the
  catastrophic ``1 − p`` subtraction when ``p → 1``.

- **``mean_prob``** (``mu`` available, no ``phi``): ``mu`` is used
  directly for LFC.  The Gamma rate is still ``beta = p / (1 − p)``,
  but ``mu`` avoids the ``r * (1 − p) / p`` reconstruction.

- **``canonical``** (fallback, only ``r`` and ``p``): ``mu`` and
  ``beta`` are derived from ``(r, p)`` using the standard formulae.

.. warning::

   For genes with near-zero biological expression the Gamma rate
   ``beta → ∞`` regardless of parameterization (``phi → 0`` in
   ``mean_odds``, ``p → 1`` in ``canonical``/``mean_prob``).  Tiny
   posterior fluctuations in these parameters inflate all biological
   metrics.  **Always filter genes by minimum biological expression**
   (``mu_A`` or ``mu_B``) before interpreting the results.
"""

from typing import Optional, List

import jax.numpy as jnp

from ..stats.divergences import gamma_jeffreys


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------


def biological_differential_expression(
    r_samples_A: jnp.ndarray,
    r_samples_B: jnp.ndarray,
    p_samples_A: jnp.ndarray,
    p_samples_B: jnp.ndarray,
    mu_samples_A: Optional[jnp.ndarray] = None,
    mu_samples_B: Optional[jnp.ndarray] = None,
    phi_samples_A: Optional[jnp.ndarray] = None,
    phi_samples_B: Optional[jnp.ndarray] = None,
    tau_lfc: float = 0.0,
    tau_var: float = 0.0,
    tau_kl: float = 0.0,
    gene_names: Optional[List[str]] = None,
) -> dict:
    """Compute biological-level DE statistics from posterior NB parameters.

    Automatically selects the most numerically stable computation path
    based on which optional native-parameterization samples are provided
    (see module docstring for details).

    Parameters
    ----------
    r_samples_A : jnp.ndarray, shape ``(N, D)``
        Posterior samples of the NB dispersion ``r`` for condition A.
    r_samples_B : jnp.ndarray, shape ``(N, D)``
        Posterior samples of the NB dispersion ``r`` for condition B.
    p_samples_A : jnp.ndarray, shape ``(N, D)`` or ``(N,)``
        Posterior samples of the NB success probability ``p`` for
        condition A.  Shape ``(N,)`` for non-hierarchical models
        where ``p`` is shared across genes (broadcast automatically).
    p_samples_B : jnp.ndarray, shape ``(N, D)`` or ``(N,)``
        Same as above for condition B.
    mu_samples_A : jnp.ndarray, optional, shape ``(N, D)``
        Directly sampled biological mean ``mu`` for condition A.
        Available from ``mean_odds`` and ``mean_prob`` parameterizations.
        When provided, avoids reconstructing ``mu`` from ``r`` and ``p``.
    mu_samples_B : jnp.ndarray, optional, shape ``(N, D)``
        Same as above for condition B.
    phi_samples_A : jnp.ndarray, optional, shape ``(N, D)`` or ``(N,)``
        Directly sampled odds ratio ``phi`` for condition A.
        Available from the ``mean_odds`` parameterization.
        When provided, the Gamma rate is ``1 / phi`` (avoids ``p/(1-p)``).
    phi_samples_B : jnp.ndarray, optional, shape ``(N, D)`` or ``(N,)``
        Same as above for condition B.
    tau_lfc : float, default=0.0
        Practical significance threshold for the biological LFC.
    tau_var : float, default=0.0
        Practical significance threshold for the log-variance ratio.
    tau_kl : float, default=0.0
        Practical significance threshold for the Jeffreys divergence.
    gene_names : list of str, optional
        Gene names.  If ``None``, generic names are generated.

    Returns
    -------
    dict
        Dictionary with the following keys (all arrays shape ``(D,)``
        unless noted):

        **Mean shift (LFC)**

        - ``lfc_mean`` : Posterior mean biological LFC per gene.
        - ``lfc_sd`` : Posterior standard deviation.
        - ``lfc_prob_positive`` : ``P(LFC_g > 0 | data)``.
        - ``lfc_lfsr`` : Local false sign rate for LFC.
        - ``lfc_prob_up`` : ``P(LFC_g > tau_lfc | data)``.
        - ``lfc_prob_down`` : ``P(LFC_g < -tau_lfc | data)``.
        - ``lfc_prob_effect`` : ``P(|LFC_g| > tau_lfc | data)``.
        - ``lfc_lfsr_tau`` : Practical-significance lfsr for LFC.

        **Variance shift (LVR)**

        - ``lvr_mean``, ``lvr_sd``, ``lvr_prob_positive``,
          ``lvr_lfsr``, ``lvr_prob_up``, ``lvr_prob_down``,
          ``lvr_prob_effect``, ``lvr_lfsr_tau`` : Analogous
          statistics for the log-variance ratio.

        **Distributional shift (Jeffreys divergence)**

        - ``kl_mean`` : Posterior mean Jeffreys divergence.
        - ``kl_sd`` : Posterior standard deviation.
        - ``kl_prob_effect`` : ``P(J_g > tau_kl | data)``.

        **Auxiliary**

        - ``mu_A_mean``, ``mu_B_mean`` : Posterior mean biological
          expression per gene in each condition.
        - ``var_A_mean``, ``var_B_mean`` : Posterior mean biological
          variance per gene in each condition.
        - ``max_bio_expr`` : ``max(mu_A_mean, mu_B_mean)`` per gene.
          Useful for filtering out near-zero expression genes whose
          metrics are unreliable (see warning above).
        - ``gene_names`` : list of str.

    Notes
    -----
    The three computation tiers (selected automatically):

    1. **phi path** (``mean_odds``): ``mu`` used directly,
       ``var = mu * (1 + phi)``, ``beta = 1 / phi``.
    2. **mu path** (``mean_prob``): ``mu`` used directly,
       ``var = mu / p``, ``beta = p / (1 - p)``.
    3. **Fallback** (``canonical``): ``mu = r * (1 - p) / p``,
       ``var = mu / p``, ``beta = p / (1 - p)``.
    """
    # Broadcast shared p to (N, 1) so element-wise ops work correctly
    if p_samples_A.ndim == 1:
        p_samples_A = p_samples_A[:, None]
    if p_samples_B.ndim == 1:
        p_samples_B = p_samples_B[:, None]

    # Broadcast shared phi to (N, 1) when present
    if phi_samples_A is not None and phi_samples_A.ndim == 1:
        phi_samples_A = phi_samples_A[:, None]
    if phi_samples_B is not None and phi_samples_B.ndim == 1:
        phi_samples_B = phi_samples_B[:, None]

    # Determine which computation tier to use:
    #   - phi_path: both mu and phi available (mean_odds)
    #   - mu_path:  mu available but no phi (mean_prob)
    #   - fallback: only r and p (canonical)
    _has_mu = mu_samples_A is not None and mu_samples_B is not None
    _has_phi = phi_samples_A is not None and phi_samples_B is not None

    # ------------------------------------------------------------------
    # Biological mean: mu_g
    # ------------------------------------------------------------------
    if _has_mu:
        mu_A = mu_samples_A
        mu_B = mu_samples_B
    else:
        # Fallback: mu = r * (1 - p) / p
        mu_A = r_samples_A * (1.0 - p_samples_A) / p_samples_A
        mu_B = r_samples_B * (1.0 - p_samples_B) / p_samples_B

    # ------------------------------------------------------------------
    # Biological variance: var_g
    #   mean_odds:  var = mu * (1 + phi)   [since 1/p = 1 + phi]
    #   otherwise:  var = mu / p
    # ------------------------------------------------------------------
    if _has_phi:
        var_A = mu_A * (1.0 + phi_samples_A)
        var_B = mu_B * (1.0 + phi_samples_B)
    else:
        var_A = mu_A / p_samples_A
        var_B = mu_B / p_samples_B

    # ------------------------------------------------------------------
    # Biological LFC: log(mu_A / mu_B)
    # ------------------------------------------------------------------
    eps = 1e-30
    lfc_samples = jnp.log(jnp.maximum(mu_A, eps)) - jnp.log(
        jnp.maximum(mu_B, eps)
    )
    lfc_stats = _summarise_signed_metric(lfc_samples, tau_lfc)

    # ------------------------------------------------------------------
    # Log-variance ratio: log(var_A / var_B)
    # ------------------------------------------------------------------
    lvr_samples = jnp.log(jnp.maximum(var_A, eps)) - jnp.log(
        jnp.maximum(var_B, eps)
    )
    lvr_stats = _summarise_signed_metric(lvr_samples, tau_var)

    # ------------------------------------------------------------------
    # Gamma Jeffreys divergence (symmetrised KL on the latent rate)
    #   Gamma shape = r,  Gamma rate beta:
    #     mean_odds:  beta = 1 / phi
    #     otherwise:  beta = p / (1 - p)
    # ------------------------------------------------------------------
    if _has_phi:
        # beta = 1 / phi avoids the catastrophic p / (1 - p) computation
        beta_A = 1.0 / phi_samples_A
        beta_B = 1.0 / phi_samples_B
    else:
        beta_A = p_samples_A / (1.0 - p_samples_A)
        beta_B = p_samples_B / (1.0 - p_samples_B)

    jeffreys_samples = gamma_jeffreys(
        r_samples_A, beta_A, r_samples_B, beta_B
    )
    kl_stats = _summarise_nonneg_metric(jeffreys_samples, tau_kl)

    # ------------------------------------------------------------------
    # Gene names and auxiliary quantities for downstream filtering
    # ------------------------------------------------------------------
    D = r_samples_A.shape[1]
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(D)]

    mu_A_mean = jnp.mean(mu_A, axis=0)
    mu_B_mean = jnp.mean(mu_B, axis=0)
    max_bio_expr = jnp.maximum(mu_A_mean, mu_B_mean)

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    return {
        # LFC
        "lfc_mean": lfc_stats["mean"],
        "lfc_sd": lfc_stats["sd"],
        "lfc_prob_positive": lfc_stats["prob_positive"],
        "lfc_lfsr": lfc_stats["lfsr"],
        "lfc_prob_up": lfc_stats["prob_up"],
        "lfc_prob_down": lfc_stats["prob_down"],
        "lfc_prob_effect": lfc_stats["prob_effect"],
        "lfc_lfsr_tau": lfc_stats["lfsr_tau"],
        # LVR
        "lvr_mean": lvr_stats["mean"],
        "lvr_sd": lvr_stats["sd"],
        "lvr_prob_positive": lvr_stats["prob_positive"],
        "lvr_lfsr": lvr_stats["lfsr"],
        "lvr_prob_up": lvr_stats["prob_up"],
        "lvr_prob_down": lvr_stats["prob_down"],
        "lvr_prob_effect": lvr_stats["prob_effect"],
        "lvr_lfsr_tau": lvr_stats["lfsr_tau"],
        # KL
        "kl_mean": kl_stats["mean"],
        "kl_sd": kl_stats["sd"],
        "kl_prob_effect": kl_stats["prob_effect"],
        # Auxiliary
        "mu_A_mean": mu_A_mean,
        "mu_B_mean": mu_B_mean,
        "var_A_mean": jnp.mean(var_A, axis=0),
        "var_B_mean": jnp.mean(var_B, axis=0),
        "max_bio_expr": max_bio_expr,
        "gene_names": gene_names,
    }


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _summarise_signed_metric(
    samples: jnp.ndarray,
    tau: float,
) -> dict:
    """Summarise a signed per-gene metric (LFC or LVR) across samples.

    Parameters
    ----------
    samples : jnp.ndarray, shape ``(N, D)``
        Per-sample values of the signed metric.
    tau : float
        Practical significance threshold.

    Returns
    -------
    dict
        Summary statistics: mean, sd, prob_positive, lfsr,
        prob_up, prob_down, prob_effect, lfsr_tau.
    """
    mean = jnp.mean(samples, axis=0)
    sd = jnp.std(samples, axis=0, ddof=1)

    prob_positive = jnp.mean(samples > 0, axis=0)
    lfsr = jnp.minimum(prob_positive, 1.0 - prob_positive)

    prob_up = jnp.mean(samples > tau, axis=0)
    prob_down = jnp.mean(samples < -tau, axis=0)
    prob_effect = prob_up + prob_down
    lfsr_tau = 1.0 - jnp.maximum(prob_up, prob_down)

    return {
        "mean": mean,
        "sd": sd,
        "prob_positive": prob_positive,
        "lfsr": lfsr,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "prob_effect": prob_effect,
        "lfsr_tau": lfsr_tau,
    }


def _summarise_nonneg_metric(
    samples: jnp.ndarray,
    tau: float,
) -> dict:
    """Summarise a non-negative per-gene metric (KL) across samples.

    Parameters
    ----------
    samples : jnp.ndarray, shape ``(N, D)``
        Per-sample values of the non-negative metric.
    tau : float
        Practical significance threshold.

    Returns
    -------
    dict
        Summary statistics: mean, sd, prob_effect.
    """
    mean = jnp.mean(samples, axis=0)
    sd = jnp.std(samples, axis=0, ddof=1)
    prob_effect = jnp.mean(samples > tau, axis=0)

    return {
        "mean": mean,
        "sd": sd,
        "prob_effect": prob_effect,
    }
