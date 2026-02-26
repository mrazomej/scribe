"""Per-gene goodness-of-fit diagnostics via randomized quantile residuals.

This module implements the Randomized Quantile Residual (RQR) framework
of Dunn and Smyth (1996) for assessing how well a fitted model describes
each gene's count distribution.  The key property is *expression-scale
invariance*: under a correctly specified model, the residuals are
standard normal for every gene regardless of expression level, dispersion,
or number of cells.

The workflow is:

1. **Compute residuals** (``compute_quantile_residuals``): for each
   cell-gene pair, transform the observed UMI count through the model's
   predictive CDF and the randomized PIT, yielding a standard-normal
   residual.
2. **Summarize per gene** (``goodness_of_fit_scores``): aggregate
   residuals across cells into interpretable per-gene diagnostics
   (mean, variance, tail excess, KS distance).
3. **Build a boolean mask** (``compute_gof_mask``): a high-level function
   that extracts MAP parameters from a results object, computes
   residuals, and returns a ``(G,)`` boolean mask suitable for the
   ``gene_mask`` argument of the DE pipeline.

See ``paper/_goodness_of_fit.qmd`` for full mathematical derivations.

References
----------
Dunn, P.K. & Smyth, G.K. (1996). "Randomized Quantile Residuals."
    *Journal of Computational and Graphical Statistics*, 5(3), 236--244.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
from jax.scipy.stats import norm


# ---------------------------------------------------------------------------
# Core: compute randomized quantile residuals
# ---------------------------------------------------------------------------


def compute_quantile_residuals(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    rng_key: jnp.ndarray,
    mixing_weights: Optional[jnp.ndarray] = None,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """Compute randomized quantile residuals for NB or NB-mixture models.

    For each cell-gene pair, the observed count is mapped through the
    model's predictive CDF (randomized for discrete data) and then
    through the inverse normal CDF.  Under a correctly specified model
    the residuals are i.i.d. standard normal.

    Parameters
    ----------
    counts : jnp.ndarray, shape ``(C, G)``
        Observed UMI count matrix.  Rows are cells, columns are genes.
    r : jnp.ndarray
        NB dispersion parameter.

        * Single-component: shape ``(G,)`` — one dispersion per gene.
        * Mixture: shape ``(K, G)`` — one dispersion per component per
          gene, where ``K`` is the number of mixture components.
    p : jnp.ndarray
        NB success probability.

        * Scalar or shape ``(1,)``: shared across genes and components.
        * Shape ``(K,)``: one per component (shared across genes).
        * Shape ``(G,)``: one per gene (single-component only).
        * Shape ``(K, G)``: one per component per gene.
    rng_key : jnp.ndarray
        JAX PRNG key for the uniform randomization step.
    mixing_weights : jnp.ndarray or None, optional
        Mixture component weights.

        * ``None`` for single-component models.
        * Shape ``(K,)`` for global weights.
        * Shape ``(C, K)`` for per-cell weights.
    epsilon : float, default=1e-6
        Clipping bound: PIT values are clamped to ``(epsilon, 1-epsilon)``
        before applying the inverse normal CDF, preventing infinite
        residuals from floating-point boundary cases.

    Returns
    -------
    jnp.ndarray, shape ``(C, G)``
        Randomized quantile residuals.  Under the true model, each
        entry is approximately drawn from N(0, 1).
    """
    counts = jnp.asarray(counts)
    r = jnp.asarray(r)
    p = jnp.asarray(p)
    C, G = counts.shape

    if mixing_weights is not None:
        # ---- Mixture model: marginal CDF ----
        cdf_upper = _marginal_nb_cdf(counts, r, p, mixing_weights)
        # CDF at (counts - 1); F(-1) = 0 by convention
        counts_minus_1 = jnp.maximum(counts - 1, 0)
        cdf_lower_raw = _marginal_nb_cdf(counts_minus_1, r, p, mixing_weights)
        cdf_lower = jnp.where(counts == 0, 0.0, cdf_lower_raw)
    else:
        # ---- Single-component model ----
        nb = dist.NegativeBinomialProbs(r, p)
        cdf_upper = nb.cdf(counts)
        counts_minus_1 = jnp.maximum(counts - 1, 0)
        cdf_lower_raw = nb.cdf(counts_minus_1)
        cdf_lower = jnp.where(counts == 0, 0.0, cdf_lower_raw)

    # Randomize: v ~ Uniform(cdf_lower, cdf_upper)
    u = random.uniform(rng_key, shape=(C, G))
    v = cdf_lower + u * (cdf_upper - cdf_lower)

    # Clip to (epsilon, 1 - epsilon) to avoid infinite residuals
    v = jnp.clip(v, epsilon, 1.0 - epsilon)

    # Transform to the normal scale
    q = norm.ppf(v)
    return q


def _marginal_nb_cdf(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    mixing_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the marginal CDF of an NB mixture.

    F(u) = sum_k pi_k * F_NB(u | r_k, p_k)

    Parameters
    ----------
    counts : jnp.ndarray, shape ``(C, G)``
        Count values at which to evaluate the CDF.
    r : jnp.ndarray, shape ``(K, G)``
        Per-component, per-gene dispersion.
    p : jnp.ndarray
        Success probability.  Broadcast-compatible shapes accepted:
        scalar, ``(K,)``, ``(G,)``, or ``(K, G)``.
    mixing_weights : jnp.ndarray
        Shape ``(K,)`` for global weights or ``(C, K)`` for per-cell.

    Returns
    -------
    jnp.ndarray, shape ``(C, G)``
    """
    K = r.shape[0]

    # Normalize p to shape (K, G) for component-wise NB construction
    p_broadcast = jnp.broadcast_to(
        _ensure_component_gene_shape(p, K, counts.shape[1]), (K, counts.shape[1])
    )

    # Normalize mixing weights to (C, K)
    if mixing_weights.ndim == 1:
        # Global weights (K,) -> (1, K) for broadcasting
        weights = mixing_weights[None, :]  # (1, K)
    else:
        weights = mixing_weights  # already (C, K)

    # Compute per-component CDFs and take weighted sum
    # Expand counts to (C, G, 1) for broadcasting against component axis
    counts_3d = counts[:, :, None]  # (C, G, 1)
    r_3d = r.T[None, :, :]  # (1, G, K) — transpose (K, G) -> (G, K)
    p_3d = p_broadcast.T[None, :, :]  # (1, G, K)

    # Per-component NB CDF: shape (C, G, K)
    nb_components = dist.NegativeBinomialProbs(r_3d, p_3d)
    component_cdfs = nb_components.cdf(counts_3d)  # (C, G, K)

    # Weights shape for broadcasting: (C, 1, K)
    weights_3d = weights[:, None, :]  # (C, 1, K)

    # Marginal CDF: sum_k pi_k * F_k  -> (C, G)
    marginal_cdf = jnp.sum(component_cdfs * weights_3d, axis=-1)
    return marginal_cdf


def _ensure_component_gene_shape(
    p: jnp.ndarray, K: int, G: int
) -> jnp.ndarray:
    """Broadcast ``p`` into shape ``(K, G)``.

    Handles scalar, ``(K,)``, ``(G,)``, and ``(K, G)`` inputs.
    """
    p = jnp.atleast_1d(p)
    if p.ndim == 0 or (p.ndim == 1 and p.shape[0] == 1):
        return jnp.broadcast_to(p, (K, G))
    if p.ndim == 1:
        if p.shape[0] == K:
            return p[:, None] * jnp.ones((1, G))
        if p.shape[0] == G:
            return jnp.ones((K, 1)) * p[None, :]
    return p  # already (K, G)


# ---------------------------------------------------------------------------
# Per-gene summary statistics
# ---------------------------------------------------------------------------


def goodness_of_fit_scores(
    residuals: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Compute per-gene goodness-of-fit summary statistics from residuals.

    Under a correctly specified model the residuals are i.i.d. N(0, 1)
    for each gene.  These statistics measure departures from that reference.

    Parameters
    ----------
    residuals : jnp.ndarray, shape ``(C, G)``
        Randomized quantile residual matrix (output of
        ``compute_quantile_residuals``).

    Returns
    -------
    dict
        Dictionary with per-gene arrays of shape ``(G,)``:

        ``mean``
            Sample mean of residuals.  Should be near 0.
        ``variance``
            Sample variance (Bessel-corrected).  Should be near 1.
            Values >> 1 indicate the model underestimates gene variability;
            values << 1 indicate overestimation.
        ``tail_excess``
            Fraction of |residual| > 2 minus the N(0,1) expectation
            (0.0455).  Should be near 0.
        ``ks_distance``
            Kolmogorov--Smirnov distance between the empirical residual
            distribution and the standard normal CDF.  An omnibus measure
            of departure from N(0, 1).

    Notes
    -----
    Computational cost is O(C * G) for mean, variance, and tail excess.
    The KS distance additionally requires sorting each gene's residuals,
    adding an O(C log C * G) term, which is subdominant for typical
    single-cell dataset sizes.
    """
    C = residuals.shape[0]

    # Location miscalibration
    mean = jnp.mean(residuals, axis=0)

    # Scale miscalibration (Bessel-corrected)
    variance = jnp.var(residuals, axis=0, ddof=1)

    # Tail excess: fraction of |q| > 2, centered at the N(0,1) expectation
    tail_frac = jnp.mean(jnp.abs(residuals) > 2.0, axis=0)
    expected_tail = 2.0 * (1.0 - norm.cdf(2.0))  # ~0.0455
    tail_excess = tail_frac - expected_tail

    # KS distance vs N(0,1)
    ks_distance = _ks_distance_normal(residuals)

    return {
        "mean": mean,
        "variance": variance,
        "tail_excess": tail_excess,
        "ks_distance": ks_distance,
    }


def _ks_distance_normal(residuals: jnp.ndarray) -> jnp.ndarray:
    """Per-gene KS distance between empirical residuals and N(0, 1).

    Parameters
    ----------
    residuals : jnp.ndarray, shape ``(C, G)``

    Returns
    -------
    jnp.ndarray, shape ``(G,)``
    """
    C, G = residuals.shape
    # Sort each gene's residuals
    sorted_q = jnp.sort(residuals, axis=0)  # (C, G)

    # Empirical CDF values at sorted points: i/C for i = 1..C
    ecdf = jnp.arange(1, C + 1)[:, None] / C  # (C, 1) broadcast to (C, G)

    # Theoretical CDF at sorted points
    theo_cdf = norm.cdf(sorted_q)  # (C, G)

    # KS = max_i |F_empirical(q_(i)) - Phi(q_(i))|
    # Also check |F_empirical(q_(i)) - 1/C - Phi(q_(i))| for left jumps
    diff_upper = jnp.abs(ecdf - theo_cdf)
    diff_lower = jnp.abs(ecdf - 1.0 / C - theo_cdf)
    ks = jnp.maximum(jnp.max(diff_upper, axis=0), jnp.max(diff_lower, axis=0))
    return ks


# ---------------------------------------------------------------------------
# High-level mask builder
# ---------------------------------------------------------------------------


def compute_gof_mask(
    counts: jnp.ndarray,
    results,
    component: Optional[int] = None,
    rng_key: Optional[jnp.ndarray] = None,
    counts_for_map: Optional[jnp.ndarray] = None,
    min_variance: float = 0.5,
    max_variance: float = 1.5,
    max_ks: Optional[float] = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Build a per-gene goodness-of-fit boolean mask from a fitted model.

    This is the high-level entry point analogous to
    ``scribe.de.compute_expression_mask``.  It extracts MAP parameters
    from the results object, computes randomized quantile residuals, and
    returns a boolean mask indicating which genes are adequately described
    by the model.

    Under a correctly specified model, per-gene residual variance is
    approximately 1.  Variance substantially above 1 indicates the model
    **underestimates** gene variability (e.g., missing zero-inflation or
    overdispersion); variance substantially below 1 indicates the model
    **overestimates** variability (e.g., prior too diffuse).

    Parameters
    ----------
    counts : jnp.ndarray, shape ``(C, G)``
        Observed UMI count matrix used for residual computation.
    results : ScribeSVIResults or ScribeMCMCResults
        Fitted model results object.  Must support ``get_map()`` and
        expose ``n_components``.
    component : int or None, optional
        For mixture models, if specified, slice to a single component
        before computing MAP.  If ``None`` and the model is a mixture,
        the full marginal mixture CDF is used (recommended).
    rng_key : jnp.ndarray or None, optional
        JAX PRNG key for the randomization step.  If ``None``, a default
        key ``PRNGKey(0)`` is used.
    counts_for_map : jnp.ndarray or None, optional
        Count matrix to pass to ``get_map()`` for models with amortized
        capture probability.  If ``None``, ``counts`` is used.
    min_variance : float, default=0.5
        Lower bound on the residual variance per gene.  Genes with
        ``s_g^2 <= min_variance`` are masked out (``False``).  This
        catches genes where the model overestimates variability.
        Set to 0.0 to disable the lower-bound check.
    max_variance : float, default=1.5
        Upper bound on the residual variance per gene.  Genes with
        ``s_g^2 >= max_variance`` are masked out (``False``).  This
        catches genes where the model underestimates variability.
    max_ks : float or None, optional
        If provided, upper bound on the KS distance per gene.  Genes
        exceeding this are also masked out.  If ``None``, only the
        variance criteria are applied.
    epsilon : float, default=1e-6
        Clipping bound for the PIT values (see
        ``compute_quantile_residuals``).

    Returns
    -------
    np.ndarray, shape ``(G,)``
        Boolean mask: ``True`` for genes passing the fit criteria,
        ``False`` for poorly fit genes.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    map_counts = counts_for_map if counts_for_map is not None else counts

    is_mixture = (
        getattr(results, "n_components", None) is not None
        and results.n_components > 1
    )

    if component is not None:
        # Slice to a single component — treated as single-component
        comp_results = results.get_component(component)
        map_est = comp_results.get_map(
            use_mean=True, canonical=True, verbose=False, counts=map_counts
        )
        r, p = _extract_r_p(map_est)
        residuals = compute_quantile_residuals(
            counts, r, p, rng_key, mixing_weights=None, epsilon=epsilon
        )
    elif is_mixture:
        # Full mixture: use marginal CDF
        map_est = results.get_map(
            use_mean=True, canonical=True, verbose=False, counts=map_counts
        )
        r, p, mixing_weights = _extract_mixture_params(map_est)
        residuals = compute_quantile_residuals(
            counts, r, p, rng_key,
            mixing_weights=mixing_weights, epsilon=epsilon,
        )
    else:
        # Single-component model
        map_est = results.get_map(
            use_mean=True, canonical=True, verbose=False, counts=map_counts
        )
        r, p = _extract_r_p(map_est)
        residuals = compute_quantile_residuals(
            counts, r, p, rng_key, mixing_weights=None, epsilon=epsilon
        )

    # Compute summary scores
    scores = goodness_of_fit_scores(residuals)

    # Build mask: variance must be within [min_variance, max_variance]
    mask = np.asarray(scores["variance"] < max_variance)
    if min_variance > 0.0:
        mask = mask & np.asarray(scores["variance"] > min_variance)

    # Optionally add KS criterion
    if max_ks is not None:
        mask = mask & np.asarray(scores["ks_distance"] < max_ks)

    return mask


# ---------------------------------------------------------------------------
# Parameter extraction helpers
# ---------------------------------------------------------------------------


def _extract_r_p(map_estimates: dict):
    """Extract (r, p) from MAP estimates for single-component models.

    Parameters
    ----------
    map_estimates : dict
        MAP parameter dictionary from ``get_map(canonical=True)``.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        ``(r, p)`` arrays suitable for ``NegativeBinomialProbs``.

    Raises
    ------
    ValueError
        If neither ``r`` nor ``p`` can be found.
    """
    if "r" not in map_estimates:
        raise ValueError(
            "MAP estimates must contain 'r' (dispersion). "
            f"Available keys: {sorted(map_estimates.keys())}"
        )
    if "p" not in map_estimates:
        raise ValueError(
            "MAP estimates must contain 'p' (success probability). "
            f"Available keys: {sorted(map_estimates.keys())}"
        )
    r = jnp.squeeze(map_estimates["r"])
    p = jnp.squeeze(map_estimates["p"])
    return r, p


def _extract_mixture_params(map_estimates: dict):
    """Extract (r, p, mixing_weights) from MAP estimates for mixture models.

    Parameters
    ----------
    map_estimates : dict
        MAP parameter dictionary from ``get_map(canonical=True)`` on the
        full (un-sliced) mixture results.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray, jnp.ndarray)
        ``(r, p, mixing_weights)`` where ``r`` has shape ``(K, G)``,
        ``p`` is broadcast-compatible, and ``mixing_weights`` has shape
        ``(K,)``.

    Raises
    ------
    ValueError
        If required keys are missing.
    """
    for key in ("r", "p", "mixing_weights"):
        if key not in map_estimates:
            raise ValueError(
                f"Mixture MAP estimates must contain '{key}'. "
                f"Available keys: {sorted(map_estimates.keys())}"
            )
    r = jnp.squeeze(map_estimates["r"])
    p = jnp.squeeze(map_estimates["p"])
    mixing_weights = jnp.squeeze(map_estimates["mixing_weights"])
    return r, p, mixing_weights
