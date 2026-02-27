"""Biological-level differential expression from posterior NB parameters.

This module computes differential expression metrics in the space of
the **biological** (denoised) Negative Binomial distribution, bypassing
the compositional simplex entirely.  For each gene the model learns
parameters ``(r_g, p_g)`` that define the NB distribution of true
transcript counts *before* capture loss and dropout.  Comparing these
parameters between two conditions yields three complementary metrics:

1. **Biological log-fold change (LFC)** — shift in the NB mean
   ``mu_g = r_g (1 - p_g) / p_g``.
2. **Log-variance ratio (LVR)** — shift in the NB variance
   ``var_g = r_g (1 - p_g) / p_g^2``.
3. **Gamma Jeffreys divergence** — symmetrised KL divergence between
   the latent Gamma rate distributions that generate the NB counts
   via the Poisson-Gamma representation.

All metrics are computed **per gene, per posterior sample** and then
summarised into posterior means, SDs, local false sign rates, and
probabilities of exceeding user-specified practical significance
thresholds.  Because the posterior samples of ``(r, p)`` are already
available from the CLR differencing pipeline, these computations come
at near-zero additional cost.

Unlike CLR-based DE, these metrics are free of compositional closure:
a gene whose transcription does not change will have LFC ≈ 0
regardless of what other genes do.
"""

from typing import Optional, List, Union

import jax.numpy as jnp

from ..stats.divergences import gamma_kl, gamma_jeffreys


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------


def biological_differential_expression(
    r_samples_A: jnp.ndarray,
    r_samples_B: jnp.ndarray,
    p_samples_A: jnp.ndarray,
    p_samples_B: jnp.ndarray,
    tau_lfc: float = 0.0,
    tau_var: float = 0.0,
    tau_kl: float = 0.0,
    gene_names: Optional[List[str]] = None,
) -> dict:
    """Compute biological-level DE statistics from posterior NB parameters.

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
        - ``gene_names`` : list of str.

    Notes
    -----
    The biological NB mean is ``mu_g = r_g (1 - p_g) / p_g`` and the
    variance is ``var_g = r_g (1 - p_g) / p_g^2 = mu_g / p_g``.

    The Gamma Jeffreys divergence uses the Poisson-Gamma representation
    of the NB: the latent rate ``lambda ~ Gamma(r, p/(1-p))`` where
    ``r`` is the shape and ``p/(1-p)`` is the rate.  The Jeffreys
    divergence is the symmetrised KL ``KL(A||B) + KL(B||A)``.
    """
    # Broadcast shared p to (N, 1) so element-wise ops work correctly
    if p_samples_A.ndim == 1:
        p_samples_A = p_samples_A[:, None]
    if p_samples_B.ndim == 1:
        p_samples_B = p_samples_B[:, None]

    # ------------------------------------------------------------------
    # Per-sample biological mean:  mu_g = r_g * (1 - p_g) / p_g
    # ------------------------------------------------------------------
    mu_A = r_samples_A * (1.0 - p_samples_A) / p_samples_A
    mu_B = r_samples_B * (1.0 - p_samples_B) / p_samples_B

    # ------------------------------------------------------------------
    # Per-sample biological variance:  var_g = mu_g / p_g
    # ------------------------------------------------------------------
    var_A = mu_A / p_samples_A
    var_B = mu_B / p_samples_B

    # ------------------------------------------------------------------
    # Biological LFC:  log(mu_A / mu_B)
    # ------------------------------------------------------------------
    eps = 1e-30
    lfc_samples = jnp.log(jnp.maximum(mu_A, eps)) - jnp.log(
        jnp.maximum(mu_B, eps)
    )
    lfc_stats = _summarise_signed_metric(lfc_samples, tau_lfc)

    # ------------------------------------------------------------------
    # Log-variance ratio:  log(var_A / var_B)
    # ------------------------------------------------------------------
    lvr_samples = jnp.log(jnp.maximum(var_A, eps)) - jnp.log(
        jnp.maximum(var_B, eps)
    )
    lvr_stats = _summarise_signed_metric(lvr_samples, tau_var)

    # ------------------------------------------------------------------
    # Gamma Jeffreys divergence (symmetrised KL on the latent rate)
    #   Gamma shape = r,  Gamma rate = p / (1 - p)
    # ------------------------------------------------------------------
    beta_A = p_samples_A / (1.0 - p_samples_A)
    beta_B = p_samples_B / (1.0 - p_samples_B)

    jeffreys_samples = gamma_jeffreys(
        r_samples_A, beta_A, r_samples_B, beta_B
    )
    kl_stats = _summarise_nonneg_metric(jeffreys_samples, tau_kl)

    # ------------------------------------------------------------------
    # Gene names
    # ------------------------------------------------------------------
    D = r_samples_A.shape[1]
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(D)]

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
        "mu_A_mean": jnp.mean(mu_A, axis=0),
        "mu_B_mean": jnp.mean(mu_B, axis=0),
        "var_A_mean": jnp.mean(var_A, axis=0),
        "var_B_mean": jnp.mean(var_B, axis=0),
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
