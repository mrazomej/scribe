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
  ``beta`` are derived from ``(r, p)`` using the runtime canonical
  mapping used by model extraction.

.. warning::

   For genes with near-zero biological expression the Gamma rate
   ``beta → ∞`` regardless of parameterization (``phi → 0`` in
   ``mean_odds``, ``p → 1`` in ``canonical``/``mean_prob``).  Tiny
   posterior fluctuations in these parameters inflate all biological
   metrics.  **Always filter genes by minimum biological expression**
   (``mu_A`` or ``mu_B``) before interpreting the results.
"""

from typing import Optional, List, Iterable, Set, TYPE_CHECKING

import jax.numpy as jnp

from ..stats.divergences import gamma_jeffreys

if TYPE_CHECKING:
    from ..core.axis_layout import AxisLayout


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _needs_gene_broadcast(
    arr: jnp.ndarray,
    layout: Optional["AxisLayout"] = None,
) -> bool:
    """Return ``True`` if ``arr`` has no gene dimension and needs ``[:, None]``.

    When a layout is available the gene axis is checked semantically.
    Otherwise the legacy ``ndim == 1`` heuristic is used: a 1-D array
    is assumed to have only the sample dimension (no gene axis).

    Parameters
    ----------
    arr : jnp.ndarray
        Array to inspect.
    layout : AxisLayout, optional
        Semantic axis descriptor for ``arr``.

    Returns
    -------
    bool
    """
    if layout is not None:
        return layout.gene_axis is None
    return arr.ndim == 1


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
    metric_families: Optional[Iterable[str]] = None,
    p_layout: Optional["AxisLayout"] = None,
    phi_layout: Optional["AxisLayout"] = None,
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
    metric_families : iterable of {'bio_lfc', 'bio_lvr', 'bio_kl', 'bio_aux'}, optional
        Biological families to compute. When ``None``, all biological families
        are computed for backward compatibility.
    p_layout : AxisLayout, optional
        Semantic axis layout for the ``p`` samples.  When provided, the
        gene-axis check uses ``layout.gene_axis is None`` instead of
        the ``ndim == 1`` heuristic for broadcast decisions.
    phi_layout : AxisLayout, optional
        Semantic axis layout for the ``phi`` samples.

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
    3. **Fallback** (``canonical``): ``mu = r * p / (1 - p)``,
       ``var = mu / p``, ``beta = p / (1 - p)``.
    """
    # Resolve requested biological families once so downstream blocks can skip
    # unused expensive computations (for example KL divergence).
    metric_families_set = _resolve_biological_metric_families(metric_families)
    include_lfc = "bio_lfc" in metric_families_set
    include_lvr = "bio_lvr" in metric_families_set
    include_kl = "bio_kl" in metric_families_set
    include_aux = "bio_aux" in metric_families_set

    # Broadcast shared (scalar-across-genes) p to (N, 1) so element-wise
    # ops work correctly.  Use layout.gene_axis when available; otherwise
    # fall back to the ndim == 1 heuristic.
    if _needs_gene_broadcast(p_samples_A, p_layout):
        p_samples_A = p_samples_A[:, None]
    if _needs_gene_broadcast(p_samples_B, p_layout):
        p_samples_B = p_samples_B[:, None]

    # Broadcast shared phi to (N, 1) when present
    if phi_samples_A is not None and _needs_gene_broadcast(
        phi_samples_A, phi_layout
    ):
        phi_samples_A = phi_samples_A[:, None]
    if phi_samples_B is not None and _needs_gene_broadcast(
        phi_samples_B, phi_layout
    ):
        phi_samples_B = phi_samples_B[:, None]

    # Determine which computation tier to use:
    #   - phi_path: both mu and phi available (mean_odds)
    #   - mu_path:  mu available but no phi (mean_prob)
    #   - fallback: only r and p (canonical)
    _has_mu = mu_samples_A is not None and mu_samples_B is not None
    _has_phi = phi_samples_A is not None and phi_samples_B is not None

    # Compute mu only when any requested family depends on it.
    compute_mu = include_lfc or include_lvr or include_aux
    mu_A = None
    mu_B = None
    if compute_mu:
        if _has_mu:
            mu_A = mu_samples_A
            mu_B = mu_samples_B
        else:
            # Fallback: keep mu reconstruction consistent with canonical
            # extraction from (r, p) used by the training/runtime path.
            mu_A = r_samples_A * p_samples_A / (1.0 - p_samples_A)
            mu_B = r_samples_B * p_samples_B / (1.0 - p_samples_B)

    # Compute variance only for requested LVR/aux families.
    compute_var = include_lvr or include_aux
    var_A = None
    var_B = None
    if compute_var:
        if mu_A is None or mu_B is None:
            raise RuntimeError(
                "Internal error: variance requested but mu was not computed."
            )
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
    lfc_stats = None
    if include_lfc:
        if mu_A is None or mu_B is None:
            raise RuntimeError(
                "Internal error: LFC requested but mu was not computed."
            )
        lfc_samples = jnp.log(jnp.maximum(mu_A, eps)) - jnp.log(
            jnp.maximum(mu_B, eps)
        )
        lfc_stats = _summarise_signed_metric(lfc_samples, tau_lfc)

    # ------------------------------------------------------------------
    # Log-variance ratio: log(var_A / var_B)
    # ------------------------------------------------------------------
    lvr_stats = None
    if include_lvr:
        if var_A is None or var_B is None:
            raise RuntimeError(
                "Internal error: LVR requested but variance was not computed."
            )
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
    kl_stats = None
    if include_kl:
        if _has_phi:
            # beta = 1 / phi avoids catastrophic p / (1 - p) cancellation.
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

    mu_A_mean = jnp.mean(mu_A, axis=0) if mu_A is not None else None
    mu_B_mean = jnp.mean(mu_B, axis=0) if mu_B is not None else None
    max_bio_expr = (
        jnp.maximum(mu_A_mean, mu_B_mean)
        if (mu_A_mean is not None and mu_B_mean is not None)
        else None
    )

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    results = {"gene_names": gene_names}
    if include_lfc and lfc_stats is not None:
        results.update(
            {
                "lfc_mean": lfc_stats["mean"],
                "lfc_sd": lfc_stats["sd"],
                "lfc_prob_positive": lfc_stats["prob_positive"],
                "lfc_lfsr": lfc_stats["lfsr"],
                "lfc_prob_up": lfc_stats["prob_up"],
                "lfc_prob_down": lfc_stats["prob_down"],
                "lfc_prob_effect": lfc_stats["prob_effect"],
                "lfc_lfsr_tau": lfc_stats["lfsr_tau"],
            }
        )
    if include_lvr and lvr_stats is not None:
        results.update(
            {
                "lvr_mean": lvr_stats["mean"],
                "lvr_sd": lvr_stats["sd"],
                "lvr_prob_positive": lvr_stats["prob_positive"],
                "lvr_lfsr": lvr_stats["lfsr"],
                "lvr_prob_up": lvr_stats["prob_up"],
                "lvr_prob_down": lvr_stats["prob_down"],
                "lvr_prob_effect": lvr_stats["prob_effect"],
                "lvr_lfsr_tau": lvr_stats["lfsr_tau"],
            }
        )
    if include_kl and kl_stats is not None:
        results.update(
            {
                "kl_mean": kl_stats["mean"],
                "kl_sd": kl_stats["sd"],
                "kl_prob_effect": kl_stats["prob_effect"],
            }
        )
    if include_aux:
        if mu_A_mean is None or mu_B_mean is None:
            raise RuntimeError(
                "Internal error: auxiliary outputs requested but means missing."
            )
        results.update(
            {
                "mu_A_mean": mu_A_mean,
                "mu_B_mean": mu_B_mean,
                "var_A_mean": (
                    jnp.mean(var_A, axis=0) if var_A is not None else jnp.nan
                ),
                "var_B_mean": (
                    jnp.mean(var_B, axis=0) if var_B is not None else jnp.nan
                ),
                "max_bio_expr": max_bio_expr,
            }
        )
    return results


def _resolve_biological_metric_families(
    metric_families: Optional[Iterable[str]],
) -> Set[str]:
    """Normalize requested biological families into a validated set.

    Parameters
    ----------
    metric_families : iterable of str, optional
        Requested biological families. ``None`` requests all families.

    Returns
    -------
    set[str]
        Validated biological family set.
    """
    supported = {"bio_lfc", "bio_lvr", "bio_kl", "bio_aux"}
    if metric_families is None:
        return set(supported)
    normalized = set(metric_families)
    invalid = normalized - supported
    if invalid:
        raise ValueError(
            "Unsupported biological metric families: "
            f"{sorted(invalid)}. Supported: {sorted(supported)}."
        )
    return normalized


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
