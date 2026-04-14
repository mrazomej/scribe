"""Non-parametric differential expression from posterior samples.

This module provides the empirical (Monte Carlo) path for Bayesian
differential expression.  Instead of assuming the ALR/CLR marginals are
Gaussian and using ``norm.cdf`` for lfsr, it computes all DE statistics
by **counting** over paired posterior samples — no distributional
assumptions required.

Main functions
--------------
- ``compute_clr_differences`` : Go from Dirichlet concentration samples
  ``r`` to CLR-space differences ``Delta = CLR(rho_A) - CLR(rho_B)``.
  Handles mixture models (component slicing) and both independent and
  within-mixture (paired) comparisons.
- ``empirical_differential_expression`` : Compute gene-level DE
  statistics (lfsr, prob_effect, etc.) from a ``(N, D)`` matrix of CLR
  differences by vectorized counting.

The validity of pairwise differencing rests on posterior independence:

- **Independent models** (``paired=False``): the joint posterior
  factorises as ``pi(rho_A, rho_B | data_A, data_B) = pi(rho_A | data_A)
  * pi(rho_B | data_B)``, so any pairing of samples is valid.  We pair
  by index for convenience.
- **Within-mixture** (``paired=True``): both components come from the
  same posterior draw, so we must preserve the sample-index pairing.
  Dirichlet draws use the same per-sample RNG sub-key for both
  components.

References
----------
- Stephens, M. (2017). "False discovery rates: a new deal."
  *Biostatistics*, 18(2), 275--294.  (lfsr concept)
"""

from typing import Optional, List, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

if TYPE_CHECKING:
    from ..core.axis_layout import AxisLayout


# --------------------------------------------------------------------------
# Gene aggregation for expression filtering
# --------------------------------------------------------------------------


def _aggregate_genes(
    r_A: jnp.ndarray,
    r_B: jnp.ndarray,
    gene_mask: jnp.ndarray,
) -> tuple:
    """Pool filtered genes into a single "other" pseudo-gene.

    Genes marked ``False`` in ``gene_mask`` are summed into a single
    aggregate concentration that is appended as the last column.  This
    preserves the total Dirichlet concentration exactly so that the
    simplex constraint is maintained downstream.

    Parameters
    ----------
    r_A : jnp.ndarray, shape ``(N, D)``
        Dirichlet concentration samples for condition A.
    r_B : jnp.ndarray, shape ``(N, D)``
        Dirichlet concentration samples for condition B.
    gene_mask : jnp.ndarray, shape ``(D,)``
        Boolean mask.  ``True`` = keep the gene, ``False`` = pool into
        "other".

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        ``(r_A_agg, r_B_agg)`` each of shape ``(N, D_kept + 1)``, where
        ``D_kept = gene_mask.sum()``.  The last column is the summed
        concentration of all filtered genes.

    Raises
    ------
    ValueError
        If ``gene_mask`` has the wrong length or keeps no genes.
    """
    gene_mask = jnp.asarray(gene_mask, dtype=bool)

    if gene_mask.ndim != 1 or gene_mask.shape[0] != r_A.shape[1]:
        raise ValueError(
            f"gene_mask must be a 1-D boolean array of length D={r_A.shape[1]}, "
            f"got shape {gene_mask.shape}."
        )

    D_kept = int(gene_mask.sum())
    if D_kept == 0:
        raise ValueError("gene_mask must keep at least one gene (all False).")

    # Split kept vs. filtered for each condition
    r_A_kept = r_A[:, gene_mask]  # (N, D_kept)
    r_A_other = r_A[:, ~gene_mask].sum(axis=1, keepdims=True)  # (N, 1)
    r_A_agg = jnp.concatenate([r_A_kept, r_A_other], axis=1)

    r_B_kept = r_B[:, gene_mask]
    r_B_other = r_B[:, ~gene_mask].sum(axis=1, keepdims=True)
    r_B_agg = jnp.concatenate([r_B_kept, r_B_other], axis=1)

    return r_A_agg, r_B_agg


def _aggregate_simplex(
    simplex: np.ndarray,
    gene_mask: np.ndarray,
) -> np.ndarray:
    """Pool filtered genes into a single "other" column in simplex space.

    This is the post-sampling counterpart to :func:`_aggregate_genes`.
    It is used when gene-specific ``p`` samples are present (Gamma path),
    because there is no principled way to merge ``p`` values before
    sampling.  Instead, the full-gene simplex is sampled first and the
    masked-out gene columns are summed into a single "other" proportion
    afterwards.  The simplex constraint (rows sum to 1) is preserved
    exactly because summing a subset of non-negative values that sum to 1
    still sums to 1.

    Operates on CPU (numpy) arrays.  JAX inputs are converted
    automatically.

    Parameters
    ----------
    simplex : numpy.ndarray, shape ``(N, D)``
        Simplex samples with rows summing to 1.
    gene_mask : numpy.ndarray, shape ``(D,)``
        Boolean mask.  ``True`` = keep the gene as its own column,
        ``False`` = merge into the "other" column.

    Returns
    -------
    numpy.ndarray, shape ``(N, D_kept + 1)``
        Reduced simplex where the last column holds the summed proportion
        of all filtered genes.  ``D_kept = gene_mask.sum()``.

    Raises
    ------
    ValueError
        If ``gene_mask`` has the wrong length or keeps no genes.
    """
    simplex = np.asarray(simplex)
    gene_mask = np.asarray(gene_mask, dtype=bool)

    if gene_mask.ndim != 1 or gene_mask.shape[0] != simplex.shape[1]:
        raise ValueError(
            f"gene_mask must be a 1-D boolean array of length D={simplex.shape[1]}, "
            f"got shape {gene_mask.shape}."
        )

    D_kept = int(gene_mask.sum())
    if D_kept == 0:
        raise ValueError("gene_mask must keep at least one gene (all False).")

    # Kept genes stay as individual columns; filtered genes are summed.
    kept = simplex[:, gene_mask]  # (N, D_kept)
    other = simplex[:, ~gene_mask].sum(axis=1, keepdims=True)  # (N, 1)
    return np.concatenate([kept, other], axis=1)  # (N, D_kept + 1)


# --------------------------------------------------------------------------
# Expression mask helper
# --------------------------------------------------------------------------


def compute_expression_mask(
    results_A,
    results_B,
    component_A: int,
    component_B: int,
    min_mean_expression: float = 1.0,
    counts_A: Optional[jnp.ndarray] = None,
    counts_B: Optional[jnp.ndarray] = None,
) -> np.ndarray:
    """Build a boolean gene mask from MAP mean-expression estimates.

    A gene passes the filter if its MAP mean expression ``mu`` is at
    least ``min_mean_expression`` in **either** condition.  This
    preserves genes that are genuinely condition-specific (highly
    expressed in one condition only).

    The function works with any parameterization: if the MAP estimates
    include ``mu`` directly (``linked`` / ``mean_prob`` / ``mean_odds``
    / ``odds_ratio``), it is used as-is.  For the ``standard``
    parameterization (which provides ``r`` and ``p`` but not ``mu``),
    the mean expression is derived as ``mu = r * p / (1 - p)``.

    Parameters
    ----------
    results_A : ScribeSVIResults
        Fitted model for condition A.
    results_B : ScribeSVIResults
        Fitted model for condition B.
    component_A : int
        Mixture-component index for condition A.
    component_B : int
        Mixture-component index for condition B.
    min_mean_expression : float, default=1.0
        Minimum MAP mean expression (in count space) for a gene to pass
        the filter.  Genes below this threshold in *both* conditions are
        pooled into "other".
    counts_A : jnp.ndarray, optional
        Count matrix for condition A.  Required when the model uses
        amortized capture probability.
    counts_B : jnp.ndarray, optional
        Count matrix for condition B.  Required when the model uses
        amortized capture probability.

    Returns
    -------
    np.ndarray, shape ``(D,)``
        Boolean mask — ``True`` for genes that pass the expression
        filter.
    """
    map_A = results_A.get_component(component_A).get_map(
        use_mean=True, canonical=True, verbose=False, counts=counts_A
    )
    map_B = results_B.get_component(component_B).get_map(
        use_mean=True, canonical=True, verbose=False, counts=counts_B
    )

    mu_A = np.asarray(_extract_mu(map_A))
    mu_B = np.asarray(_extract_mu(map_B))

    return (mu_A >= min_mean_expression) | (mu_B >= min_mean_expression)


# ------------------------------------------------------------------------------


def _extract_mu(map_estimates: dict) -> jnp.ndarray:
    """Extract or derive mean expression ``mu`` from MAP estimates.

    For parameterizations that include ``mu`` directly (``linked``,
    ``mean_prob``, ``mean_odds``, ``odds_ratio``) the value is returned
    as-is.  For the ``standard`` parameterization (``r`` and ``p`` only),
    ``mu`` is computed as ``r * p / (1 - p)`` from the negative-binomial
    mean formula.

    Parameters
    ----------
    map_estimates : dict
        MAP parameter dictionary returned by ``get_map()``.

    Returns
    -------
    jnp.ndarray
        Mean expression vector, shape ``(D,)`` (or ``(K, D)`` for
        mixture models before component slicing).

    Raises
    ------
    ValueError
        If neither ``mu`` nor both ``r`` and ``p`` are present.
    """
    if "mu" in map_estimates:
        return map_estimates["mu"]

    if "r" in map_estimates and "p" in map_estimates:
        r = map_estimates["r"]
        p = map_estimates["p"]
        return r * p / (1.0 - p)

    available = sorted(map_estimates.keys())
    raise ValueError(
        "Cannot determine mean expression: MAP estimates contain neither "
        "'mu' nor both 'r' and 'p'.  "
        f"Available keys: {available}"
    )


# --------------------------------------------------------------------------
# Composition sampling (Stage 1 of CLR pipeline)
# --------------------------------------------------------------------------


def _drop_scalar_p(
    p: Optional[jnp.ndarray],
    post_layout: Optional["AxisLayout"] = None,
) -> Optional[jnp.ndarray]:
    """Return ``None`` if ``p`` has no gene dimension (scalar across genes).

    A scalar ``p`` produces a constant scaling factor in Gamma-based
    composition sampling that cancels after normalization, making
    Gamma equivalent to Dirichlet.  When a post-sliced layout is
    available the gene axis is checked semantically; otherwise the
    ``ndim < 2`` heuristic is used as a fallback.

    Parameters
    ----------
    p : jnp.ndarray or None
        Sliced (post-component) ``p`` samples.
    post_layout : AxisLayout, optional
        Layout for ``p`` **after** component slicing.  This is the
        layout returned by ``_slice_component`` — component axis
        already removed.

    Returns
    -------
    jnp.ndarray or None
        The input unchanged if it has a gene axis, ``None`` otherwise.
    """
    if p is None:
        return None

    if post_layout is not None:
        if post_layout.gene_axis is None:
            return None
        return p

    # Legacy fallback: ndim < 2 means no gene dimension
    if p.ndim < 2:
        return None
    return p


def sample_compositions(
    r_samples_A: jnp.ndarray,
    r_samples_B: jnp.ndarray,
    component_A: Optional[int] = None,
    component_B: Optional[int] = None,
    paired: bool = False,
    n_samples_dirichlet: int = 1,
    rng_key=None,
    batch_size: int = 2048,
    p_samples_A: Optional[jnp.ndarray] = None,
    p_samples_B: Optional[jnp.ndarray] = None,
    param_layouts: Optional[dict] = None,
) -> tuple:
    """Draw full-dimensional simplex samples from posterior parameters.

    This is Stage 1 of the empirical DE pipeline: go from Dirichlet
    concentration parameters ``r`` (and optionally gene-specific ``p``)
    to paired simplex compositions of shape ``(N_total, D)`` each.

    No gene aggregation or CLR transformation is performed here.  The
    returned simplices can be stored and reused with different gene masks
    via :func:`compute_delta_from_simplex`.

    Sampling is performed on GPU in small batches, but each batch is
    transferred to CPU immediately.  The returned arrays are numpy
    (host) arrays, so the full simplex never resides on device memory.

    Parameters
    ----------
    r_samples_A : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Posterior samples of Dirichlet concentration parameters for
        condition A.  If 3D, ``K`` is the number of mixture components
        and ``component_A`` must be specified.
    r_samples_B : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Posterior samples for condition B.
    component_A : int, optional
        Mixture component index to extract from ``r_samples_A``.
        Required if ``r_samples_A`` is 3D.
    component_B : int, optional
        Mixture component index to extract from ``r_samples_B``.
        Required if ``r_samples_B`` is 3D.
    paired : bool, default=False
        If ``True``, use the **same** RNG sub-key per sample index for
        both conditions (required for within-mixture comparisons).
    n_samples_dirichlet : int, default=1
        Number of Dirichlet draws per posterior sample.
    rng_key : jax.random.PRNGKey, optional
        JAX PRNG key.  If ``None``, uses ``jax.random.PRNGKey(0)``.
    batch_size : int, default=2048
        Number of posterior samples per batched sampling call.
    p_samples_A : jnp.ndarray, optional
        Success probabilities for condition A.  When gene-specific
        (shape ``(N, D)`` or ``(N, K, D)``), Gamma-based composition
        sampling is used.  Scalar ``p`` (shape ``(N,)`` or ``(N, K)``)
        is dropped because the constant scaling factor cancels in
        the normalization, making Gamma equivalent to Dirichlet.
    p_samples_B : jnp.ndarray, optional
        Same as above for condition B.
    param_layouts : dict of str to AxisLayout, optional
        Semantic axis layouts keyed by parameter name (e.g. ``"r"``,
        ``"p"``).  When available, component slicing and gene-axis
        checks use layout metadata instead of ``ndim`` heuristics.

    Returns
    -------
    simplex_A : numpy.ndarray, shape ``(N_total, D)``
        Full-dimensional simplex samples for condition A (on CPU).
    simplex_B : numpy.ndarray, shape ``(N_total, D)``
        Full-dimensional simplex samples for condition B (on CPU).

    Raises
    ------
    ValueError
        If 3D input is given without the corresponding ``component_*``
        argument, or if ``paired=True`` and sample counts differ.

    Notes
    -----
    By the Dirichlet aggregation property, sampling the full D-dimensional
    Dirichlet and aggregating afterwards is equivalent to aggregating the
    concentration parameters and sampling the lower-dimensional Dirichlet.
    Storing the full simplex therefore allows exact re-aggregation with any
    gene mask without re-sampling.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    # Resolve per-parameter layouts (None when no layouts available)
    r_layout = param_layouts.get("r") if param_layouts else None
    p_layout = param_layouts.get("p") if param_layouts else None

    # --- Slice mixture components if needed ---
    # _slice_component returns (array, post_layout) — post_layout has the
    # component axis removed and is None when no layout was provided.
    r_A, _ = _slice_component(r_samples_A, component_A, "A", layout=r_layout)
    r_B, _ = _slice_component(r_samples_B, component_B, "B", layout=r_layout)

    # Slice p samples if provided.  Scalar p (no gene dimension) produces
    # a constant scaling factor that cancels in the normalization, so
    # Gamma-based sampling is equivalent to Dirichlet -- skip it.
    p_A, p_post = (
        _slice_component(p_samples_A, component_A, "A", layout=p_layout)
        if p_samples_A is not None
        else (None, None)
    )
    p_B, _ = (
        _slice_component(p_samples_B, component_B, "B", layout=p_layout)
        if p_samples_B is not None
        else (None, None)
    )

    # Drop scalar p (no gene axis): the constant scaling factor cancels
    # in the normalization.  Use the *post-sliced* layout's gene_axis
    # when available, otherwise fall back to ndim < 2.
    p_A = _drop_scalar_p(p_A, p_post)
    p_B = _drop_scalar_p(p_B, p_post)

    # Gamma-based sampling only when p is gene-specific (2D)
    use_gamma = p_A is not None or p_B is not None

    # --- Validate sample counts ---
    N_A, D_A = r_A.shape
    N_B, D_B = r_B.shape

    if D_A != D_B:
        raise ValueError(
            f"Gene dimensions do not match: A has D={D_A}, B has D={D_B}."
        )

    if paired and N_A != N_B:
        raise ValueError(
            f"paired=True requires equal sample counts, "
            f"got N_A={N_A}, N_B={N_B}."
        )

    # Truncate to the shorter if independent (pair by index)
    N = min(N_A, N_B)
    r_A = r_A[:N]
    r_B = r_B[:N]
    if p_A is not None:
        p_A = p_A[:N]
    if p_B is not None:
        p_B = p_B[:N]

    # Move sliced inputs to CPU.  The full posterior-sample arrays (potentially
    # multi-GiB each) are kept on GPU only as long as they're referenced by
    # Python.  Converting the slices to numpy here lets JAX free the original
    # device buffers, so the batched samplers below only put one small chunk on
    # GPU at a time.
    r_A = np.asarray(r_A)
    r_B = np.asarray(r_B)
    if p_A is not None:
        p_A = np.asarray(p_A)
    if p_B is not None:
        p_B = np.asarray(p_B)

    # --- Composition sampling (always full-dimensional, no aggregation) ---
    if use_gamma:
        key_A, key_B = random.split(rng_key)
        simplex_A = _batched_gamma_normalize(
            r_A, p_A, n_samples_dirichlet, key_A, batch_size
        )
        simplex_B = _batched_gamma_normalize(
            r_B, p_B, n_samples_dirichlet, key_B, batch_size
        )
    elif paired:
        simplex_A, simplex_B = _paired_dirichlet_sample(
            r_A, r_B, n_samples_dirichlet, rng_key, batch_size
        )
    else:
        key_A, key_B = random.split(rng_key)
        simplex_A = _batched_dirichlet(
            r_A, n_samples_dirichlet, key_A, batch_size
        )
        simplex_B = _batched_dirichlet(
            r_B, n_samples_dirichlet, key_B, batch_size
        )

    return simplex_A, simplex_B


# --------------------------------------------------------------------------
# CLR aggregation + differencing (Stage 2 of CLR pipeline)
# --------------------------------------------------------------------------


def compute_delta_from_simplex(
    simplex_A: np.ndarray,
    simplex_B: np.ndarray,
    gene_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Aggregate, CLR-transform, and difference paired simplex samples.

    This is Stage 2 of the empirical DE pipeline.  Given full-dimensional
    simplex samples from :func:`sample_compositions`, it optionally
    aggregates filtered genes into an "other" pseudo-gene, applies the
    CLR transform, computes paired differences, and drops the "other"
    column.

    Because the CLR geometric mean defines the reference against which
    effects are measured, the gene mask **must** be applied before the
    CLR transform—not as a post-hoc filter on the differences.
    See :ref:`sec-diffexp-expression-mask` in the paper for details.

    All computation runs on CPU (numpy).  JAX inputs are converted
    automatically via the underlying helpers.

    Parameters
    ----------
    simplex_A : numpy.ndarray, shape ``(N, D)``
        Full-dimensional simplex samples for condition A.
    simplex_B : numpy.ndarray, shape ``(N, D)``
        Full-dimensional simplex samples for condition B.
    gene_mask : numpy.ndarray, shape ``(D,)``, optional
        Boolean mask selecting genes to keep.  Genes marked ``False``
        are aggregated into a single "other" pseudo-gene via
        :func:`_aggregate_simplex` before CLR.  The "other" column is
        dropped from the returned differences.
        If ``None`` (default), all genes are kept and no aggregation
        is performed.

    Returns
    -------
    numpy.ndarray, shape ``(N, D)`` or ``(N, D_kept)``
        CLR-space differences ``CLR(rho_A) - CLR(rho_B)``.
        When ``gene_mask`` is provided, ``D_kept = gene_mask.sum()``.
    """
    s_A, s_B = simplex_A, simplex_B

    # Aggregate filtered genes into "other" column before CLR so the
    # geometric mean reference reflects only the expressed genes.
    if gene_mask is not None:
        s_A = _aggregate_simplex(s_A, gene_mask)
        s_B = _aggregate_simplex(s_B, gene_mask)

    clr_A = _clr_transform(s_A)
    clr_B = _clr_transform(s_B)
    delta = clr_A - clr_B

    # Drop the "other" pseudo-gene column
    if gene_mask is not None:
        delta = delta[:, :-1]

    return delta


# --------------------------------------------------------------------------
# Legacy convenience wrapper
# --------------------------------------------------------------------------


def compute_clr_differences(
    r_samples_A: jnp.ndarray,
    r_samples_B: jnp.ndarray,
    component_A: Optional[int] = None,
    component_B: Optional[int] = None,
    paired: bool = False,
    n_samples_dirichlet: int = 1,
    rng_key=None,
    batch_size: int = 2048,
    gene_mask: Optional[jnp.ndarray] = None,
    p_samples_A: Optional[jnp.ndarray] = None,
    p_samples_B: Optional[jnp.ndarray] = None,
) -> np.ndarray:
    """Compute CLR-space posterior differences from Dirichlet concentration samples.

    Convenience wrapper that calls :func:`sample_compositions` followed
    by :func:`compute_delta_from_simplex`.  Use the two-stage API
    directly when you need to store the intermediate simplex samples for
    interactive mask exploration.

    Parameters
    ----------
    r_samples_A : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Posterior samples of Dirichlet concentration parameters for
        condition A.  If 3D, ``K`` is the number of mixture components
        and ``component_A`` must be specified.
    r_samples_B : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Posterior samples for condition B.
    component_A : int, optional
        Mixture component index to extract from ``r_samples_A``.
    component_B : int, optional
        Mixture component index to extract from ``r_samples_B``.
    paired : bool, default=False
        If ``True``, use the **same** RNG sub-key per sample index for
        both conditions (required for within-mixture comparisons).
    n_samples_dirichlet : int, default=1
        Number of Dirichlet draws per posterior sample.
    rng_key : jax.random.PRNGKey, optional
        JAX PRNG key.  If ``None``, uses ``jax.random.PRNGKey(0)``.
    batch_size : int, default=2048
        Number of posterior samples per batched Dirichlet sampling call.
    gene_mask : jnp.ndarray, shape ``(D,)``, optional
        Boolean mask selecting genes to keep.  Genes marked ``False``
        are aggregated into an "other" pseudo-gene before CLR.
    p_samples_A : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``, optional
        Gene-specific success probabilities for condition A.
    p_samples_B : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``, optional
        Gene-specific success probabilities for condition B.

    Returns
    -------
    numpy.ndarray, shape ``(N_total, D)`` or ``(N_total, D_kept)``
        CLR-space differences on CPU.
        ``N_total = N * n_samples_dirichlet``.
        When ``gene_mask`` is provided, ``D_kept = gene_mask.sum()``.

    Raises
    ------
    ValueError
        If 3D input is given without the corresponding ``component_*``
        argument, or if ``paired=True`` and sample counts differ.
    """
    simplex_A, simplex_B = sample_compositions(
        r_samples_A=r_samples_A,
        r_samples_B=r_samples_B,
        component_A=component_A,
        component_B=component_B,
        paired=paired,
        n_samples_dirichlet=n_samples_dirichlet,
        rng_key=rng_key,
        batch_size=batch_size,
        p_samples_A=p_samples_A,
        p_samples_B=p_samples_B,
    )
    return compute_delta_from_simplex(simplex_A, simplex_B, gene_mask)


# --------------------------------------------------------------------------
# Empirical DE statistics
# --------------------------------------------------------------------------


def empirical_differential_expression(
    delta_samples: jnp.ndarray,
    tau: float = 0.0,
    gene_names: Optional[List[str]] = None,
) -> dict:
    """Compute gene-level DE statistics from CLR difference samples.

    All statistics are computed by vectorized counting over the ``N``
    posterior samples — no distributional assumptions.  The output dict
    has the same keys as the parametric ``differential_expression()``
    so that downstream code (error control, formatting) works
    interchangeably.

    Parameters
    ----------
    delta_samples : jnp.ndarray, shape ``(N, D)``
        Posterior CLR-space differences
        ``Delta_g^(s) = CLR(rho_A)_g^(s) - CLR(rho_B)_g^(s)``.
    tau : float, default=0.0
        Practical significance threshold (log-scale).
    gene_names : list of str, optional
        Gene names.  If ``None``, generic names are generated.

    Returns
    -------
    dict
        Dictionary with the following keys, each of shape ``(D,)``:

        - **delta_mean** : Posterior mean effect per gene.
        - **delta_sd** : Posterior standard deviation per gene.
        - **prob_positive** : ``P(Delta_g > 0 | data)`` estimated as
          the fraction of positive samples.
        - **prob_effect** : ``P(|Delta_g| > tau | data)`` estimated as
          the fraction of samples exceeding the threshold.
        - **lfsr** : Local false sign rate =
          ``min(P(Delta_g > 0), P(Delta_g < 0))``.
        - **lfsr_tau** : Practical-significance lfsr =
          ``1 - max(P(Delta_g > tau), P(Delta_g < -tau))``.
        - **gene_names** : list of str.

    Notes
    -----
    Resolution is limited by the number of samples ``N``.  The smallest
    non-zero lfsr is ``1/N``.  The standard error of the empirical lfsr
    estimate is ``SE = sqrt(lfsr * (1 - lfsr) / N)``, which is ~0.001
    for lfsr=0.01 with N=10,000.

    All operations are fully vectorized JAX and run on GPU.  Cost is
    ``O(N * D)`` with no additional memory beyond the input.
    """
    # Posterior mean and standard deviation per gene
    delta_mean = jnp.mean(delta_samples, axis=0)
    delta_sd = jnp.std(delta_samples, axis=0, ddof=1)

    # Fraction of samples with positive difference
    prob_positive = jnp.mean(delta_samples > 0, axis=0)

    # Local false sign rate: posterior probability of the minority sign
    lfsr = jnp.minimum(prob_positive, 1.0 - prob_positive)

    # Probability of practical effect: fraction of samples where
    # |Delta_g| > tau
    prob_up = jnp.mean(delta_samples > tau, axis=0)
    prob_down = jnp.mean(delta_samples < -tau, axis=0)
    prob_effect = prob_up + prob_down

    # Practical-significance lfsr (paper definition):
    # lfsr_tau = 1 - max(P(Delta > tau), P(Delta < -tau))
    lfsr_tau = 1.0 - jnp.maximum(prob_up, prob_down)

    # Generate gene names if not provided
    if gene_names is None:
        D = delta_samples.shape[1]
        gene_names = [f"gene_{i}" for i in range(D)]

    return {
        "delta_mean": delta_mean,
        "delta_sd": delta_sd,
        "prob_positive": prob_positive,
        "prob_effect": prob_effect,
        "lfsr": lfsr,
        "lfsr_tau": lfsr_tau,
        "gene_names": gene_names,
    }


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _slice_component(
    r_samples: jnp.ndarray,
    component: Optional[int],
    label: str,
    layout: Optional["AxisLayout"] = None,
) -> tuple:
    """Slice a mixture component from posterior samples.

    When an ``AxisLayout`` is provided, the component axis is determined
    semantically via ``layout.component_axis`` and the returned layout
    reflects the post-sliced shape (component axis removed).  Without a
    layout the function falls back to the legacy ``ndim``-based heuristic.

    Parameters
    ----------
    r_samples : jnp.ndarray
        Posterior samples with one of these shapes:

        - ``(N, D)`` -- non-mixture gene-specific
        - ``(N, K, D)`` -- mixture gene-specific
        - ``(N,)`` -- non-mixture scalar (or scalar already sliced)
        - ``(N, K)`` -- mixture scalar
    component : int or None
        Component index to extract from mixture models.
    label : str
        Label for error messages (``"A"`` or ``"B"``).
    layout : AxisLayout, optional
        Semantic axis descriptor for ``r_samples``.  When given, the
        component axis is read from ``layout.component_axis`` and slicing
        uses ``jnp.take`` along that axis.  When ``None``, falls back to
        the legacy ``ndim`` ladder.

    Returns
    -------
    sliced : jnp.ndarray
        Sliced samples: ``(N, D)`` for gene-specific, ``(N,)`` for scalar.
    post_layout : AxisLayout or None
        The layout after removing the component axis (if one was present),
        or ``None`` when no layout was provided.
    """
    # --- Layout-aware path: use semantic axis metadata ---
    if layout is not None:
        comp_ax = layout.component_axis
        if comp_ax is not None:
            if component is None:
                raise ValueError(
                    f"r_samples_{label} has a component axis (layout={layout}) "
                    f"but component_{label} was not specified."
                )
            sliced = jnp.take(r_samples, component, axis=comp_ax)
            # Derive the post-sliced layout (component axis removed)
            post_layout = layout.subset_axis("components")
            return sliced, post_layout
        # No component axis -> nothing to slice, layout unchanged
        return r_samples, layout

    # --- Legacy ndim fallback (no layout provided) ---
    if r_samples.ndim == 3:
        if component is None:
            raise ValueError(
                f"r_samples_{label} is 3D (mixture model) but "
                f"component_{label} was not specified."
            )
        return r_samples[:, component, :], None
    elif r_samples.ndim == 2:
        return r_samples, None
    elif r_samples.ndim == 1:
        return r_samples, None
    else:
        raise ValueError(
            f"r_samples_{label} must be 1-3D, got {r_samples.ndim}D."
        )


# ------------------------------------------------------------------------------


def _clr_transform(simplex_samples: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform on simplex samples.

    Operates on CPU (numpy) arrays.  JAX inputs are converted
    automatically.

    Parameters
    ----------
    simplex_samples : numpy.ndarray, shape ``(N, D)``
        Samples on the D-simplex (rows sum to 1).

    Returns
    -------
    numpy.ndarray, shape ``(N, D)``
        CLR-transformed samples.
    """
    simplex_samples = np.asarray(simplex_samples)
    # Guard against exact zeros from Dirichlet sampling
    log_samples = np.log(np.maximum(simplex_samples, 1e-30))
    # CLR: subtract the geometric mean (= arithmetic mean of logs)
    geometric_mean = np.mean(log_samples, axis=-1, keepdims=True)
    return log_samples - geometric_mean


# ------------------------------------------------------------------------------


def _batched_dirichlet(
    r_samples: jnp.ndarray,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
) -> np.ndarray:
    """Batched Dirichlet sampling from concentration parameters.

    Reuses the batching strategy from
    ``scribe.core.normalization_logistic._batched_dirichlet_sample_raw``
    but implemented locally to avoid circular imports.

    Each batch is sampled on GPU and immediately transferred to CPU so
    the full simplex array never resides on device.

    Parameters
    ----------
    r_samples : jnp.ndarray, shape ``(N, D)``
        Dirichlet concentration parameters.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        Number of posterior samples per batch.

    Returns
    -------
    numpy.ndarray, shape ``(N_total, D)``
        Simplex samples on CPU.  ``N_total = N * n_samples_dirichlet``.
    """
    N, D = r_samples.shape
    chunks: list[np.ndarray] = []

    # When n_samples_dirichlet > 1, each posterior sample fans out to S
    # output rows.  Shrink the batch size so the per-chunk GPU tensor
    # (B * S, D) stays manageable.
    effective_bs = max(1, batch_size // max(1, n_samples_dirichlet))

    for start in range(0, N, effective_bs):
        end = min(start + effective_bs, N)
        r_batch = r_samples[start:end]  # (B, D)

        # Deterministic sub-key for this batch
        key_batch = random.fold_in(rng_key, start)

        # Sample from Dirichlet for each row in the batch
        if n_samples_dirichlet == 1:
            # Shape: (B, D)
            samples = jax.random.dirichlet(key_batch, r_batch)
        else:
            # Draw multiple Dirichlet samples per posterior sample
            # We vmap over the batch dimension
            keys = random.split(key_batch, end - start)

            def _sample_one(key, alpha):
                return jax.random.dirichlet(
                    key, alpha, shape=(n_samples_dirichlet,)
                )

            # (B, S, D)
            samples = jax.vmap(_sample_one)(keys, r_batch)
            # Flatten to (B * S, D)
            samples = samples.reshape(-1, D)

        # Transfer to CPU immediately so the full result never lives on GPU
        chunks.append(np.asarray(samples))

    return np.concatenate(chunks, axis=0)


# ------------------------------------------------------------------------------


def _batched_gamma_normalize(
    r_samples: jnp.ndarray,
    p_samples: jnp.ndarray,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
) -> np.ndarray:
    """Sample compositions via scaled Gamma variates and normalization.

    This generalizes Dirichlet sampling to the case where each gene has
    its own success probability ``p_g``.  The generative process is:

    1. Draw ``lambda_raw_g ~ Gamma(r_g, rate=1)`` independently per gene.
    2. Scale: ``lambda_g = lambda_raw_g * p_g / (1 - p_g)``.
    3. Normalize: ``rho_g = lambda_g / sum_j lambda_j``.

    When all ``p_g`` are equal, the scaling factor ``p / (1 - p)`` is a
    constant that cancels in the normalization, recovering exactly
    ``Dirichlet(r)``.  When ``p_g`` vary across genes, the compositions
    reflect gene-specific rate heterogeneity from the Negative Binomial
    model.

    Each batch is sampled on GPU and immediately transferred to CPU so
    the full simplex array never resides on device.

    Parameters
    ----------
    r_samples : jnp.ndarray, shape ``(N, D)``
        Dirichlet concentration (dispersion) parameters.
    p_samples : jnp.ndarray, shape ``(N, D)``
        Gene-specific success probabilities.  Must be in ``(0, 1)``.
    n_samples_dirichlet : int
        Number of composition draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        Number of posterior samples per batch.

    Returns
    -------
    numpy.ndarray, shape ``(N_total, D)``
        Simplex samples on CPU.  ``N_total = N * n_samples_dirichlet``.

    Notes
    -----
    The Gamma(r_g, 1) variate is the latent rate parameter of the
    Negative Binomial.  Scaling by p_g / (1 - p_g) converts from the
    NB parameterization to expected counts, which are then normalized
    to get compositional proportions.
    """
    N, D = r_samples.shape
    chunks: list[np.ndarray] = []

    # When n_samples_dirichlet > 1, each posterior sample fans out to S
    # output rows.  Shrink the batch size so the per-chunk GPU tensor
    # (B * S, D) stays manageable.
    effective_bs = max(1, batch_size // max(1, n_samples_dirichlet))

    for start in range(0, N, effective_bs):
        end = min(start + effective_bs, N)
        r_batch = r_samples[start:end]  # (B, D)
        p_batch = p_samples[start:end]  # (B, D) or (B, 1)

        key_batch = random.fold_in(rng_key, start)

        # Clamp p away from 0 and 1 to avoid inf/nan in p/(1-p)
        p_batch = jnp.clip(p_batch, 1e-7, 1.0 - 1e-7)

        if n_samples_dirichlet == 1:
            # Draw Gamma(r, 1) and scale by p / (1 - p)
            gamma_raw = jax.random.gamma(key_batch, r_batch)  # (B, D)
            lambda_scaled = gamma_raw * p_batch / (1.0 - p_batch)
            total = jnp.maximum(
                lambda_scaled.sum(axis=-1, keepdims=True), 1e-30
            )
            samples = lambda_scaled / total  # (B, D)
        else:
            keys = random.split(key_batch, end - start)

            def _sample_one(key, alpha, p_gene):
                gamma_raw = jax.random.gamma(
                    key, alpha, shape=(n_samples_dirichlet,) + alpha.shape
                )
                # alpha has shape (D,); gamma_raw has shape (S, D)
                lambda_scaled = gamma_raw * p_gene / (1.0 - p_gene)
                total = jnp.maximum(
                    lambda_scaled.sum(axis=-1, keepdims=True), 1e-30
                )
                return lambda_scaled / total  # (S, D)

            # (B, S, D)
            samples = jax.vmap(_sample_one)(keys, r_batch, p_batch)
            samples = samples.reshape(-1, D)  # (B * S, D)

        # Transfer to CPU immediately so the full result never lives on GPU
        chunks.append(np.asarray(samples))

    return np.concatenate(chunks, axis=0)


# ------------------------------------------------------------------------------


def _paired_dirichlet_sample(
    r_A: jnp.ndarray,
    r_B: jnp.ndarray,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
) -> tuple:
    """Paired Dirichlet sampling for within-mixture comparisons.

    Uses the **same** per-sample RNG sub-key for both conditions,
    ensuring that the joint posterior correlation structure between
    components is preserved.

    Each batch is sampled on GPU and immediately transferred to CPU so
    the full simplex arrays never reside on device.

    Parameters
    ----------
    r_A : jnp.ndarray, shape ``(N, D)``
        Concentration parameters for component A.
    r_B : jnp.ndarray, shape ``(N, D)``
        Concentration parameters for component B.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        Number of posterior samples per batch.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        ``(simplex_A, simplex_B)`` each of shape ``(N_total, D)`` on CPU.
    """
    N, D = r_A.shape
    chunks_A: list[np.ndarray] = []
    chunks_B: list[np.ndarray] = []

    # When n_samples_dirichlet > 1, each posterior sample fans out to S
    # output rows (for BOTH conditions).  Shrink the batch size so the
    # per-chunk GPU tensors stay manageable.
    effective_bs = max(1, batch_size // max(1, n_samples_dirichlet))

    for start in range(0, N, effective_bs):
        end = min(start + effective_bs, N)
        r_batch_A = r_A[start:end]
        r_batch_B = r_B[start:end]
        B = end - start

        # Same base key for this batch — both conditions share it
        key_batch = random.fold_in(rng_key, start)

        if n_samples_dirichlet == 1:
            # Split the shared key into per-sample sub-keys
            keys = random.split(key_batch, B)

            # For each sample, split the per-sample key into two
            # sub-keys (one for A, one for B).  The correlation comes
            # from using the same per-sample seed, NOT from sharing
            # the exact key — the Dirichlet draws themselves are
            # independent given the concentrations.
            def _paired_draw(key, alpha_a, alpha_b):
                k_a, k_b = random.split(key)
                return (
                    jax.random.dirichlet(k_a, alpha_a),
                    jax.random.dirichlet(k_b, alpha_b),
                )

            samples_A, samples_B = jax.vmap(_paired_draw)(
                keys, r_batch_A, r_batch_B
            )
        else:
            keys = random.split(key_batch, B)

            def _paired_draw_multi(key, alpha_a, alpha_b):
                k_a, k_b = random.split(key)
                s_a = jax.random.dirichlet(
                    k_a, alpha_a, shape=(n_samples_dirichlet,)
                )
                s_b = jax.random.dirichlet(
                    k_b, alpha_b, shape=(n_samples_dirichlet,)
                )
                return s_a, s_b

            samples_A, samples_B = jax.vmap(_paired_draw_multi)(
                keys, r_batch_A, r_batch_B
            )
            # (B, S, D) -> (B * S, D)
            samples_A = samples_A.reshape(-1, D)
            samples_B = samples_B.reshape(-1, D)

        # Transfer to CPU immediately so full results never live on GPU
        chunks_A.append(np.asarray(samples_A))
        chunks_B.append(np.asarray(samples_B))

    return np.concatenate(chunks_A, axis=0), np.concatenate(chunks_B, axis=0)
