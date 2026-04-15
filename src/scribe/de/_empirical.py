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

import math
import warnings
from functools import partial
from typing import Optional, List, Callable, Sequence, TYPE_CHECKING, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

if TYPE_CHECKING:
    from ..core.axis_layout import AxisLayout

#: Union type accepted by backend-aware functions.
Array = Union[np.ndarray, jax.Array]


# --------------------------------------------------------------------------
# Mixture-component validation
# --------------------------------------------------------------------------


def _require_mixture_components(
    component_A,
    component_B,
    param_layouts,
    caller: str,
) -> None:
    """Raise early if mixture-component indices are missing.

    Uses the semantic ``AxisLayout`` on the ``"r"`` parameter to decide
    whether the data is from a mixture model.  When no layouts are
    available (raw-array callers), this function is a no-op; the
    existing checks in ``_slice_component`` serve as defense-in-depth.

    Parameters
    ----------
    component_A, component_B : int or None
        Component indices supplied by the caller.
    param_layouts : dict or None
        Layout metadata.  When ``None``, the function returns
        immediately (no heuristic fallback).
    caller : str
        Name of the calling function (used in the error message).
    """
    if param_layouts is None:
        return

    r_layout = param_layouts.get("r")
    if r_layout is None or r_layout.component_axis is None:
        return

    missing = []
    if component_A is None:
        missing.append("component_A")
    if component_B is None:
        missing.append("component_B")
    if missing:
        raise ValueError(
            f"{caller}(): posterior samples have a mixture-component "
            f"axis but {' and '.join(missing)} "
            f"{'was' if len(missing) == 1 else 'were'} not "
            f"specified. For mixture models, provide component_A "
            f"and component_B to select which components to compare."
        )


# --------------------------------------------------------------------------
# Gene aggregation for expression filtering
# --------------------------------------------------------------------------


def _aggregate_genes(
    r_A: Array,
    r_B: Array,
    gene_mask: Array,
) -> tuple:
    """Pool filtered genes into a single "other" pseudo-gene.

    Genes marked ``False`` in ``gene_mask`` are summed into a single
    aggregate concentration that is appended as the last column.  This
    preserves the total Dirichlet concentration exactly so that the
    simplex constraint is maintained downstream.

    Parameters
    ----------
    r_A : numpy.ndarray or jax.Array, shape ``(N, D)``
        Dirichlet concentration samples for condition A.
    r_B : numpy.ndarray or jax.Array, shape ``(N, D)``
        Dirichlet concentration samples for condition B.
    gene_mask : numpy.ndarray or jax.Array, shape ``(D,)``
        Boolean mask.  ``True`` = keep the gene, ``False`` = pool into
        "other".

    Returns
    -------
    tuple of arrays
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
    counts_A: Optional[Array] = None,
    counts_B: Optional[Array] = None,
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
    counts_A : numpy.ndarray or jax.Array, optional
        Count matrix for condition A.  Required when the model uses
        amortized capture probability.
    counts_B : numpy.ndarray or jax.Array, optional
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


def _extract_mu(map_estimates: dict) -> Array:
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
    numpy.ndarray or jax.Array
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
    p: Optional[Array],
    post_layout: Optional["AxisLayout"] = None,
) -> Optional[Array]:
    """Return ``None`` if ``p`` has no gene dimension (scalar across genes).

    A scalar ``p`` produces a constant scaling factor in Gamma-based
    composition sampling that cancels after normalization, making
    Gamma equivalent to Dirichlet.  When a post-sliced layout is
    available the gene axis is checked semantically; otherwise the
    ``ndim < 2`` heuristic is used as a fallback.

    Parameters
    ----------
    p : numpy.ndarray or jax.Array, or None
        Sliced (post-component) ``p`` samples.
    post_layout : AxisLayout, optional
        Layout for ``p`` **after** component slicing.  This is the
        layout returned by ``_slice_component`` — component axis
        already removed.

    Returns
    -------
    numpy.ndarray or jax.Array, or None
        The input unchanged if it has a gene axis, ``None`` otherwise.
    """
    if p is None:
        return None

    if post_layout is not None:
        if post_layout.gene_axis is None:
            return None
        return p

    # Legacy fallback: ndim < 2 means no gene dimension.
    # Deprecated — callers should provide layout metadata.
    warnings.warn(
        "Calling _drop_scalar_p without layout metadata is deprecated. "
        "Pass param_layouts explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    if p.ndim < 2:
        return None
    return p


def sample_compositions(
    r_samples_A: Array,
    r_samples_B: Array,
    component_A: Optional[int] = None,
    component_B: Optional[int] = None,
    paired: bool = False,
    n_samples_dirichlet: int = 1,
    rng_key=None,
    batch_size: int = 2048,
    p_samples_A: Optional[Array] = None,
    p_samples_B: Optional[Array] = None,
    param_layouts: Optional[dict] = None,
) -> tuple:
    """Draw full-dimensional simplex samples from posterior parameters.

    This is Stage 1 of the empirical DE pipeline: go from Dirichlet
    concentration parameters ``r`` (and optionally gene-specific ``p``)
    to paired simplex compositions of shape ``(N_total, D)`` each.

    No gene aggregation or CLR transformation is performed here.  The
    returned simplices can be stored and reused with different gene masks
    via :func:`compute_delta_from_simplex`.

    Sampling is JIT-compiled and dispatched in as few GPU round-trips
    as possible.  The adaptive memory layer queries the device once and
    splits the work only when the full output would exceed available
    memory — on most GPUs the entire array is sampled in a single
    kernel launch.  The returned arrays are always ``numpy.ndarray``
    (host) arrays.

    Parameters
    ----------
    r_samples_A : numpy.ndarray or jax.Array, shape ``(N, D)`` or ``(N, K, D)``
        Posterior samples of Dirichlet concentration parameters for
        condition A.  If 3D, ``K`` is the number of mixture components
        and ``component_A`` must be specified.
    r_samples_B : numpy.ndarray or jax.Array, shape ``(N, D)`` or ``(N, K, D)``
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
        Upper-bound cap on chunk size.  The adaptive layer may use a
        larger chunk when GPU memory allows; this parameter acts as a
        safety net for backward compatibility or explicit memory
        control.
    p_samples_A : numpy.ndarray or jax.Array, optional
        Success probabilities for condition A.  When gene-specific
        (shape ``(N, D)`` or ``(N, K, D)``), Gamma-based composition
        sampling is used.  Scalar ``p`` (shape ``(N,)`` or ``(N, K)``)
        is dropped because the constant scaling factor cancels in
        the normalization, making Gamma equivalent to Dirichlet.
    p_samples_B : numpy.ndarray or jax.Array, optional
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

    # Early guard: require component indices for mixture-model inputs
    _require_mixture_components(
        component_A,
        component_B,
        param_layouts,
        "sample_compositions",
    )

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

    # Query the device memory budget once; the adaptive wrapper uses it
    # to decide how many chunks (ideally just 1) to split the work into.
    from ..core._array_dispatch import _gpu_memory_budget

    budget = _gpu_memory_budget()

    # --- Composition sampling (always full-dimensional, no aggregation) ---
    # Inputs stay on their current device (GPU or CPU) — the JIT kernels
    # handle the transfer implicitly, and np.asarray is called once per
    # chunk on the *output* inside the adaptive wrapper.
    if use_gamma:
        key_A, key_B = random.split(rng_key)
        simplex_A = _batched_gamma_normalize(
            r_A, p_A, n_samples_dirichlet, key_A, batch_size, budget
        )
        simplex_B = _batched_gamma_normalize(
            r_B, p_B, n_samples_dirichlet, key_B, batch_size, budget
        )
    elif paired:
        simplex_A, simplex_B = _paired_dirichlet_sample(
            r_A, r_B, n_samples_dirichlet, rng_key, batch_size, budget
        )
    else:
        key_A, key_B = random.split(rng_key)
        simplex_A = _batched_dirichlet(
            r_A, n_samples_dirichlet, key_A, batch_size, budget
        )
        simplex_B = _batched_dirichlet(
            r_B, n_samples_dirichlet, key_B, batch_size, budget
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
    r_samples_A: Array,
    r_samples_B: Array,
    component_A: Optional[int] = None,
    component_B: Optional[int] = None,
    paired: bool = False,
    n_samples_dirichlet: int = 1,
    rng_key=None,
    batch_size: int = 2048,
    gene_mask: Optional[Array] = None,
    p_samples_A: Optional[Array] = None,
    p_samples_B: Optional[Array] = None,
) -> np.ndarray:
    """Compute CLR-space posterior differences from Dirichlet concentration samples.

    Convenience wrapper that calls :func:`sample_compositions` followed
    by :func:`compute_delta_from_simplex`.  Use the two-stage API
    directly when you need to store the intermediate simplex samples for
    interactive mask exploration.

    Parameters
    ----------
    r_samples_A : numpy.ndarray or jax.Array, shape ``(N, D)`` or ``(N, K, D)``
        Posterior samples of Dirichlet concentration parameters for
        condition A.  If 3D, ``K`` is the number of mixture components
        and ``component_A`` must be specified.
    r_samples_B : numpy.ndarray or jax.Array, shape ``(N, D)`` or ``(N, K, D)``
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
        Upper-bound cap on chunk size.  The adaptive memory layer may
        use larger chunks when GPU memory allows.
    gene_mask : numpy.ndarray or jax.Array, shape ``(D,)``, optional
        Boolean mask selecting genes to keep.  Genes marked ``False``
        are aggregated into an "other" pseudo-gene before CLR.
    p_samples_A : numpy.ndarray or jax.Array, shape ``(N, D)`` or ``(N, K, D)``, optional
        Gene-specific success probabilities for condition A.
    p_samples_B : numpy.ndarray or jax.Array, shape ``(N, D)`` or ``(N, K, D)``, optional
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
    delta_samples: Array,
    tau: float = 0.0,
    gene_names: Optional[List[str]] = None,
) -> dict:
    """Compute gene-level DE statistics from CLR difference samples.

    All statistics are computed by vectorized counting over the ``N``
    posterior samples — no distributional assumptions.  The output dict
    has the same keys as the parametric ``differential_expression()``
    so that downstream code (error control, formatting) works
    interchangeably.

    Backend-aware: when ``delta_samples`` is a ``numpy.ndarray`` the
    function uses the NumPy stack; when it is a ``jax.Array`` it uses
    JAX (and therefore GPU when available).

    Parameters
    ----------
    delta_samples : array-like, shape ``(N, D)``
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

    All operations are fully vectorized.  Cost is ``O(N * D)`` with no
    additional memory beyond the input.
    """
    from ..core._array_dispatch import _array_module

    xp = _array_module(delta_samples)

    # Posterior mean and standard deviation per gene
    delta_mean = xp.mean(delta_samples, axis=0)
    delta_sd = xp.std(delta_samples, axis=0, ddof=1)

    # Fraction of samples with positive difference
    prob_positive = xp.mean(delta_samples > 0, axis=0)

    # Local false sign rate: posterior probability of the minority sign
    lfsr = xp.minimum(prob_positive, 1.0 - prob_positive)

    # Probability of practical effect: fraction of samples where
    # |Delta_g| > tau
    prob_up = xp.mean(delta_samples > tau, axis=0)
    prob_down = xp.mean(delta_samples < -tau, axis=0)
    prob_effect = prob_up + prob_down

    # Practical-significance lfsr (paper definition):
    # lfsr_tau = 1 - max(P(Delta > tau), P(Delta < -tau))
    lfsr_tau = 1.0 - xp.maximum(prob_up, prob_down)

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
    r_samples: Array,
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
    r_samples : numpy.ndarray or jax.Array
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
    sliced : numpy.ndarray or jax.Array
        Sliced samples: ``(N, D)`` for gene-specific, ``(N,)`` for scalar.
    post_layout : AxisLayout or None
        The layout after removing the component axis (if one was present),
        or ``None`` when no layout was provided.
    """
    from ..core._array_dispatch import _array_module

    xp = _array_module(r_samples)

    # --- Layout-aware path: use semantic axis metadata ---
    if layout is not None:
        comp_ax = layout.component_axis
        if comp_ax is not None:
            if component is None:
                raise ValueError(
                    f"r_samples_{label} has a component axis (layout={layout}) "
                    f"but component_{label} was not specified."
                )
            sliced = xp.take(r_samples, component, axis=comp_ax)
            # Derive the post-sliced layout (component axis removed)
            post_layout = layout.subset_axis("components")
            return sliced, post_layout
        # No component axis -> nothing to slice, layout unchanged
        return r_samples, layout

    # --- Legacy ndim fallback (no layout provided) ---
    # Deprecated — callers should provide layout metadata.
    warnings.warn(
        "Calling _slice_component without layout metadata is deprecated. "
        "Pass param_layouts explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
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


# ===========================================================================
# JIT-compiled sampling kernels
# ===========================================================================
# Defined at module level so JAX traces them once per unique shape and
# caches the compiled XLA program across calls.  All kernels accept and
# return device arrays — the host transfer (np.asarray) happens in the
# adaptive wrapper, not inside the kernel.


@jax.jit
def _jit_dirichlet_single(
    key: random.PRNGKey,
    r: jax.Array,
) -> jax.Array:
    """Single Dirichlet draw per row.  ``r`` has shape ``(B, D)``."""
    return jax.random.dirichlet(key, r)


@jax.jit
def _jit_gamma_normalize_single(
    key: random.PRNGKey,
    r: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Single Gamma-normalise draw per row.

    Draws Gamma(r_g, 1) variates, scales by p_g/(1-p_g), normalises.
    """
    p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
    gamma_raw = jax.random.gamma(key, r)
    lam = gamma_raw * p / (1.0 - p)
    total = jnp.maximum(lam.sum(axis=-1, keepdims=True), 1e-30)
    return lam / total


@jax.jit
def _jit_paired_dirichlet_single(
    keys: jax.Array,
    r_A: jax.Array,
    r_B: jax.Array,
) -> tuple:
    """Paired single-draw Dirichlet for within-mixture comparisons.

    Uses the same per-sample seed for both conditions so the joint
    posterior correlation structure is preserved.
    """

    def _draw(key, alpha_a, alpha_b):
        k_a, k_b = random.split(key)
        return jax.random.dirichlet(k_a, alpha_a), jax.random.dirichlet(
            k_b, alpha_b
        )

    return jax.vmap(_draw)(keys, r_A, r_B)


# Multi-draw kernels use ``static_argnums`` for ``n_samples`` so that
# JAX compiles a separate XLA program per distinct S value (typically
# only 1 or 2 unique values per session).


@partial(jax.jit, static_argnums=(2,))
def _jit_dirichlet_multi(
    keys: jax.Array,
    r: jax.Array,
    n_samples: int,
) -> jax.Array:
    """Multiple Dirichlet draws per row, flattened to ``(B*S, D)``.

    ``keys`` has shape ``(B, 2)``; ``r`` has shape ``(B, D)``.
    """

    def _one(key, alpha):
        return jax.random.dirichlet(key, alpha, shape=(n_samples,))

    # (B, S, D) -> (B*S, D)
    return jax.vmap(_one)(keys, r).reshape(-1, r.shape[-1])


@partial(jax.jit, static_argnums=(3,))
def _jit_gamma_normalize_multi(
    keys: jax.Array,
    r: jax.Array,
    p: jax.Array,
    n_samples: int,
) -> jax.Array:
    """Multiple Gamma-normalise draws per row, flattened to ``(B*S, D)``."""
    p = jnp.clip(p, 1e-7, 1.0 - 1e-7)

    def _one(key, alpha, p_gene):
        gamma_raw = jax.random.gamma(
            key, alpha, shape=(n_samples,) + alpha.shape
        )
        lam = gamma_raw * p_gene / (1.0 - p_gene)
        total = jnp.maximum(lam.sum(axis=-1, keepdims=True), 1e-30)
        return lam / total

    return jax.vmap(_one)(keys, r, p).reshape(-1, r.shape[-1])


@partial(jax.jit, static_argnums=(3,))
def _jit_paired_dirichlet_multi(
    keys: jax.Array,
    r_A: jax.Array,
    r_B: jax.Array,
    n_samples: int,
) -> tuple:
    """Paired multi-draw Dirichlet, each flattened to ``(B*S, D)``."""

    def _draw(key, alpha_a, alpha_b):
        k_a, k_b = random.split(key)
        s_a = jax.random.dirichlet(k_a, alpha_a, shape=(n_samples,))
        s_b = jax.random.dirichlet(k_b, alpha_b, shape=(n_samples,))
        return s_a, s_b

    s_A, s_B = jax.vmap(_draw)(keys, r_A, r_B)
    D = r_A.shape[-1]
    return s_A.reshape(-1, D), s_B.reshape(-1, D)


# ===========================================================================
# Adaptive memory-aware wrapper
# ===========================================================================


def _estimate_chunk_size(
    N: int,
    D: int,
    n_samples_dirichlet: int,
    memory_budget: float,
    n_input_arrays: int = 1,
) -> int:
    """Compute the largest chunk size that fits within *memory_budget*.

    Accounts for both the output tensor (the dominant allocation) and
    the input slices transferred to the device.

    Parameters
    ----------
    N : int
        Total number of posterior samples (rows).
    D : int
        Number of genes (columns).
    n_samples_dirichlet : int
        Fan-out factor per posterior sample.
    memory_budget : float
        Usable device bytes (from ``_gpu_memory_budget``).
    n_input_arrays : int
        Number of input arrays transferred per chunk (e.g. 1 for
        Dirichlet, 2 for Gamma-normalise, 2 for paired).

    Returns
    -------
    int
        Chunk size (number of posterior-sample rows per JIT call).
        Always >= 1.  When *memory_budget* is ``math.inf`` (CPU-only),
        returns *N* so the loop executes exactly once.
    """
    if math.isinf(memory_budget):
        return N

    bytes_per_element = 4  # float32

    # Output: (chunk * n_samples_dirichlet, D) float32
    # Inputs: n_input_arrays * (chunk, D) float32
    # Rough 2x headroom for XLA temporaries (key splits, intermediates)
    per_row = (
        n_samples_dirichlet * D * bytes_per_element  # output
        + n_input_arrays * D * bytes_per_element  # inputs on device
    ) * 2  # XLA headroom factor

    chunk = max(1, int(memory_budget // per_row))
    return min(chunk, N)


def _adaptive_sample(
    jit_fn: Callable,
    arrays: Sequence[Array],
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
    memory_budget: float,
    multi: bool,
) -> np.ndarray:
    """Run a JIT sampling kernel in as few chunks as possible.

    Chooses chunk size as ``min(batch_size, memory-derived chunk)``.
    When the entire array fits in one chunk the Python loop executes
    exactly once — the common case on modern GPUs.

    Parameters
    ----------
    jit_fn : callable
        One of the ``_jit_*`` kernels.  For the single-draw path it is
        called as ``jit_fn(key, *arrays)``; for the multi-draw path as
        ``jit_fn(keys, *arrays, n_samples_dirichlet)``.
    arrays : sequence of array-like
        Input arrays, each with leading dimension N (posterior samples).
    n_samples_dirichlet : int
        Fan-out factor; 1 for the single-draw path.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        User-provided upper bound on chunk size (backward-compat cap).
    memory_budget : float
        Usable device bytes.
    multi : bool
        If ``True``, use the multi-draw calling convention.

    Returns
    -------
    numpy.ndarray, shape ``(N_total, D)``
        Concatenated simplex samples on CPU.
    """
    N = arrays[0].shape[0]
    D = arrays[0].shape[1]

    mem_chunk = _estimate_chunk_size(
        N,
        D,
        n_samples_dirichlet,
        memory_budget,
        n_input_arrays=len(arrays),
    )
    # Respect user-provided batch_size as an upper bound, but scale it
    # down for multi-draw the same way the old code did.
    effective_user_bs = max(1, batch_size // max(1, n_samples_dirichlet))
    chunk_size = min(mem_chunk, effective_user_bs)

    chunks: list[np.ndarray] = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sliced = [a[start:end] for a in arrays]

        key_chunk = random.fold_in(rng_key, start)

        if multi:
            keys = random.split(key_chunk, end - start)
            result = jit_fn(keys, *sliced, n_samples_dirichlet)
        else:
            result = jit_fn(key_chunk, *sliced)

        chunks.append(np.asarray(result))

    return np.concatenate(chunks, axis=0)


def _adaptive_sample_paired(
    jit_fn: Callable,
    r_A: Array,
    r_B: Array,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
    memory_budget: float,
    multi: bool,
) -> tuple:
    """Paired variant of ``_adaptive_sample`` returning two arrays.

    Parameters
    ----------
    jit_fn : callable
        ``_jit_paired_dirichlet_single`` or ``_jit_paired_dirichlet_multi``.
    r_A, r_B : array-like, shape ``(N, D)``
        Concentration parameters for conditions A and B.
    n_samples_dirichlet : int
        Fan-out factor.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        User-provided upper bound on chunk size.
    memory_budget : float
        Usable device bytes.
    multi : bool
        If ``True``, use the multi-draw calling convention.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        ``(simplex_A, simplex_B)`` each of shape ``(N_total, D)`` on CPU.
    """
    N = r_A.shape[0]
    D = r_A.shape[1]

    # Two input arrays + two output arrays -> 2 for n_input_arrays
    mem_chunk = _estimate_chunk_size(
        N,
        D,
        n_samples_dirichlet,
        memory_budget,
        n_input_arrays=2,
    )
    effective_user_bs = max(1, batch_size // max(1, n_samples_dirichlet))
    chunk_size = min(mem_chunk, effective_user_bs)

    chunks_A: list[np.ndarray] = []
    chunks_B: list[np.ndarray] = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        rA_chunk = r_A[start:end]
        rB_chunk = r_B[start:end]

        key_chunk = random.fold_in(rng_key, start)
        keys = random.split(key_chunk, end - start)

        if multi:
            sA, sB = jit_fn(keys, rA_chunk, rB_chunk, n_samples_dirichlet)
        else:
            sA, sB = jit_fn(keys, rA_chunk, rB_chunk)

        chunks_A.append(np.asarray(sA))
        chunks_B.append(np.asarray(sB))

    return (
        np.concatenate(chunks_A, axis=0),
        np.concatenate(chunks_B, axis=0),
    )


# ===========================================================================
# Public batched-sampling functions (thin wrappers)
# ===========================================================================


def _batched_dirichlet(
    r_samples: Array,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
    memory_budget: float = math.inf,
) -> np.ndarray:
    """Dirichlet sampling with JIT compilation and adaptive chunking.

    Draws simplex samples from ``Dirichlet(r_samples[i, :])`` for each
    posterior sample *i*.  The work is JIT-compiled and dispatched in as
    few GPU round-trips as possible; the adaptive wrapper only splits
    into multiple chunks when device memory would be exceeded.

    Parameters
    ----------
    r_samples : array-like, shape ``(N, D)``
        Dirichlet concentration parameters.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        Upper-bound cap on chunk size (backward-compat safety net).
    memory_budget : float, default=math.inf
        Usable device bytes.  ``math.inf`` disables chunking.

    Returns
    -------
    numpy.ndarray, shape ``(N * n_samples_dirichlet, D)``
        Simplex samples on CPU.
    """
    if n_samples_dirichlet == 1:
        return _adaptive_sample(
            _jit_dirichlet_single,
            [r_samples],
            n_samples_dirichlet,
            rng_key,
            batch_size,
            memory_budget,
            multi=False,
        )
    return _adaptive_sample(
        _jit_dirichlet_multi,
        [r_samples],
        n_samples_dirichlet,
        rng_key,
        batch_size,
        memory_budget,
        multi=True,
    )


def _batched_gamma_normalize(
    r_samples: Array,
    p_samples: Array,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
    memory_budget: float = math.inf,
) -> np.ndarray:
    """Gamma-normalise sampling with JIT compilation and adaptive chunking.

    Generalises Dirichlet sampling to the case where each gene has its
    own success probability ``p_g``.  The generative process is:

    1. Draw ``lambda_raw_g ~ Gamma(r_g, rate=1)`` independently per gene.
    2. Scale: ``lambda_g = lambda_raw_g * p_g / (1 - p_g)``.
    3. Normalise: ``rho_g = lambda_g / sum_j lambda_j``.

    Parameters
    ----------
    r_samples : array-like, shape ``(N, D)``
        Dirichlet concentration (dispersion) parameters.
    p_samples : array-like, shape ``(N, D)``
        Gene-specific success probabilities in ``(0, 1)``.
    n_samples_dirichlet : int
        Number of composition draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        Upper-bound cap on chunk size.
    memory_budget : float, default=math.inf
        Usable device bytes.

    Returns
    -------
    numpy.ndarray, shape ``(N * n_samples_dirichlet, D)``
        Simplex samples on CPU.
    """
    if n_samples_dirichlet == 1:
        return _adaptive_sample(
            _jit_gamma_normalize_single,
            [r_samples, p_samples],
            n_samples_dirichlet,
            rng_key,
            batch_size,
            memory_budget,
            multi=False,
        )
    return _adaptive_sample(
        _jit_gamma_normalize_multi,
        [r_samples, p_samples],
        n_samples_dirichlet,
        rng_key,
        batch_size,
        memory_budget,
        multi=True,
    )


def _paired_dirichlet_sample(
    r_A: Array,
    r_B: Array,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
    memory_budget: float = math.inf,
) -> tuple:
    """Paired Dirichlet sampling with JIT compilation and adaptive chunking.

    Uses the **same** per-sample RNG sub-key for both conditions so
    the joint posterior correlation structure is preserved.

    Parameters
    ----------
    r_A : array-like, shape ``(N, D)``
        Concentration parameters for condition A.
    r_B : array-like, shape ``(N, D)``
        Concentration parameters for condition B.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        Upper-bound cap on chunk size.
    memory_budget : float, default=math.inf
        Usable device bytes.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        ``(simplex_A, simplex_B)`` each of shape
        ``(N * n_samples_dirichlet, D)`` on CPU.
    """
    if n_samples_dirichlet == 1:
        return _adaptive_sample_paired(
            _jit_paired_dirichlet_single,
            r_A,
            r_B,
            n_samples_dirichlet,
            rng_key,
            batch_size,
            memory_budget,
            multi=False,
        )
    return _adaptive_sample_paired(
        _jit_paired_dirichlet_multi,
        r_A,
        r_B,
        n_samples_dirichlet,
        rng_key,
        batch_size,
        memory_budget,
        multi=True,
    )


# ===========================================================================
# Mixture-weighted composition sampling
# ===========================================================================


def _weight_simplex_by_components(
    component_simplices: List[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Compute a weighted average of per-component simplex samples.

    Given K simplex arrays (one per mixture component) and posterior mixture
    weight samples, form the convex combination on the simplex. The result is
    itself a valid simplex vector for every row because the weights are
    non-negative and sum to 1 along the component axis, and each input row sums
    to 1.

    Parameters
    ----------
    component_simplices : list of numpy.ndarray
        K arrays each of shape ``(N, D)``.  Each row is a point on
        the D-simplex (sums to 1).
    weights : numpy.ndarray, shape ``(N, K)``
        Posterior mixture weight samples.  Each row sums to 1.

    Returns
    -------
    numpy.ndarray, shape ``(N, D)``
        Weighted simplex: ``sum_k weights[:, k] * component_simplices[k]``.
    """
    # Stack to (N, K, D) then use einsum for the weighted sum
    stacked = np.stack(component_simplices, axis=1)  # (N, K, D)
    return np.einsum("nk,nkd->nd", weights, stacked)


def sample_mixture_compositions(
    r_samples_A: Array,
    r_samples_B: Array,
    weights_A: Array,
    weights_B: Array,
    paired: bool = False,
    n_samples_dirichlet: int = 1,
    rng_key=None,
    batch_size: int = 2048,
    p_samples_A: Optional[Array] = None,
    p_samples_B: Optional[Array] = None,
    param_layouts: Optional[dict] = None,
) -> tuple:
    """Draw mixture-weighted simplex samples from all components.

    Instead of slicing a single mixture component, this function samples
    compositions from *every* component and forms a weighted average on the
    simplex using the posterior mixture weights.  The resulting weighted
    compositions feed directly into the standard CLR pipeline via
    :func:`compute_delta_from_simplex`.

    The marginal composition of a cell drawn from a K-component mixture model is
    a *mixture of Dirichlets*:

        p(rho) = sum_k pi_k Dir(rho | r_k)

    Rather than drawing from this mixture (which would introduce discrete
    component-assignment noise), we compute the expected composition conditional
    on each component and form the convex combination.  This produces the
    population-level mean composition for each posterior draw, which is the
    natural target for cell-type- level differential expression.

    Parameters
    ----------
    r_samples_A : numpy.ndarray or jax.Array, shape ``(N, K, D)``
        Posterior samples of Dirichlet concentrations for condition A.
        The second axis indexes mixture components.
    r_samples_B : numpy.ndarray or jax.Array, shape ``(N, K, D)``
        Same for condition B.
    weights_A : numpy.ndarray or jax.Array, shape ``(N, K)``
        Posterior mixture weight samples for condition A.
    weights_B : numpy.ndarray or jax.Array, shape ``(N, K)``
        Posterior mixture weight samples for condition B.
    paired : bool, default=False
        If ``True``, use the same RNG sub-key per sample index for
        both conditions (for within-model comparisons).
    n_samples_dirichlet : int, default=1
        Number of Dirichlet draws per posterior sample.
    rng_key : jax.random.PRNGKey, optional
        JAX PRNG key.  If ``None``, uses ``jax.random.PRNGKey(0)``.
    batch_size : int, default=2048
        Upper-bound cap on chunk size for adaptive memory management.
    p_samples_A : numpy.ndarray or jax.Array, optional
        Gene-specific success probabilities for condition A.
        Shape ``(N, K, D)`` for the Gamma-normalise path.
    p_samples_B : numpy.ndarray or jax.Array, optional
        Same for condition B.
    param_layouts : dict of str to AxisLayout, optional
        Semantic axis layouts (not used for component slicing here,
        but threaded for ``_drop_scalar_p`` checks on the p axis).

    Returns
    -------
    simplex_A : numpy.ndarray, shape ``(N_total, D)``
        Mixture-weighted simplex samples for condition A.
    simplex_B : numpy.ndarray, shape ``(N_total, D)``
        Mixture-weighted simplex samples for condition B.

    Raises
    ------
    ValueError
        If input shapes are inconsistent (e.g., K mismatch between
        r_samples and weights, or D mismatch between conditions).
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    r_A = jnp.asarray(r_samples_A)
    r_B = jnp.asarray(r_samples_B)
    w_A = np.asarray(weights_A)
    w_B = np.asarray(weights_B)

    # --- Validate shapes ---
    if r_A.ndim != 3 or r_B.ndim != 3:
        raise ValueError(
            "sample_mixture_compositions requires 3D r_samples "
            f"(N, K, D), got ndim={r_A.ndim} for A and {r_B.ndim} for B."
        )

    N_A, K_A, D_A = r_A.shape
    N_B, K_B, D_B = r_B.shape

    if D_A != D_B:
        raise ValueError(
            f"Gene dimensions do not match: A has D={D_A}, B has D={D_B}."
        )
    if w_A.shape != (N_A, K_A):
        raise ValueError(
            f"weights_A shape {w_A.shape} does not match "
            f"r_samples_A (N={N_A}, K={K_A})."
        )
    if w_B.shape != (N_B, K_B):
        raise ValueError(
            f"weights_B shape {w_B.shape} does not match "
            f"r_samples_B (N={N_B}, K={K_B})."
        )
    if paired and N_A != N_B:
        raise ValueError(
            f"paired=True requires equal sample counts, "
            f"got N_A={N_A}, N_B={N_B}."
        )

    # Truncate to the shorter chain if independent
    N = min(N_A, N_B)
    r_A = r_A[:N]
    r_B = r_B[:N]
    w_A = w_A[:N]
    w_B = w_B[:N]

    # Slice p samples per component if provided
    p_A_3d = jnp.asarray(p_samples_A[:N]) if p_samples_A is not None else None
    p_B_3d = jnp.asarray(p_samples_B[:N]) if p_samples_B is not None else None

    from ..core._array_dispatch import _gpu_memory_budget

    budget = _gpu_memory_budget()

    # --- Sample compositions for each component and weight ---
    # We loop over components (small K, typically 2-5) and reuse the
    # existing JIT-compiled sampling kernels for each slice.
    K = K_A

    def _sample_all_components(r_3d, p_3d, rng_base):
        """Sample Dirichlet (or Gamma-normalise) per component."""
        per_component = []
        for k in range(K):
            r_k = r_3d[:, k, :]  # (N, D)
            key_k = random.fold_in(rng_base, k)

            # Check if this component has gene-specific p
            p_k = None
            if p_3d is not None:
                if p_3d.ndim == 3:
                    p_k = p_3d[:, k, :]
                elif p_3d.ndim == 2:
                    # Scalar p per component: (N, K) -> (N,)
                    p_k = p_3d[:, k]

            # Drop scalar p (no gene axis => constant scaling cancels)
            if p_k is not None and p_k.ndim < 2:
                p_k = None

            if p_k is not None:
                simplex_k = _batched_gamma_normalize(
                    r_k, p_k, n_samples_dirichlet, key_k, batch_size, budget
                )
            else:
                simplex_k = _batched_dirichlet(
                    r_k, n_samples_dirichlet, key_k, batch_size, budget
                )
            per_component.append(simplex_k)
        return per_component

    # Split keys for the two conditions
    if paired:
        # Paired: same base key, components get different sub-keys
        # but both conditions share the per-component seeds
        key_A = rng_key
        key_B = rng_key
    else:
        key_A, key_B = random.split(rng_key)

    components_A = _sample_all_components(r_A, p_A_3d, key_A)
    components_B = _sample_all_components(r_B, p_B_3d, key_B)

    # Tile weights when n_samples_dirichlet > 1: each posterior row
    # fans out to S Dirichlet draws, so repeat each weight S times.
    if n_samples_dirichlet > 1:
        w_A = np.repeat(w_A, n_samples_dirichlet, axis=0)
        w_B = np.repeat(w_B, n_samples_dirichlet, axis=0)

    simplex_A = _weight_simplex_by_components(components_A, w_A)
    simplex_B = _weight_simplex_by_components(components_B, w_B)

    return simplex_A, simplex_B
