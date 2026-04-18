"""Log-probability helpers for the :mod:`scribe` likelihood components.

This module centralises the JIT-friendly, full-array log-probability
implementations that back :meth:`Likelihood.log_prob` on the four NB-family
subclasses (``NegativeBinomialLikelihood``, ``ZeroInflatedNBLikelihood``,
``NBWithVCPLikelihood``, ``ZINBWithVCPLikelihood``).  The BNB family inherits
these methods unchanged because :func:`_build_ll_count_dist` dispatches on
the presence of ``bnb_concentration`` in the parameter dictionary.

Design notes
------------
The original free-function API in ``models/log_likelihood.py`` batched over
cells with Python ``for`` loops, which unroll at trace time when the
functions are called from inside ``@jit``.  Each delegate here is a single
full-array expression (no Python loops, no ``.at[...].set(...)`` scatter
updates) and therefore compiles to a single XLA kernel.

Shared helpers
~~~~~~~~~~~~~~
``_normalize_counts``, ``_clip_r``, ``_clip_p``, ``_extract_capture``,
``_validate_weights``, ``_check_return_by``, ``_check_weight_type`` perform
trivial pre-processing that every family needs.

``_build_ll_count_dist`` dispatches between NB and BNB.
``_validate_mixture_component_shapes`` guards against component-axis
misalignment after mixture pruning.

``_prepare_p_mixture_layout`` handles the scalar/component/gene-specific
broadcast reshape for ``p``.  ``_mixture_reduce`` performs the final sum
over cells or genes, applies optional gene/cell weights, adds the log
mixing weights, and optionally collapses over components via ``logsumexp``.

Delegate functions
~~~~~~~~~~~~~~~~~~
``nb_log_prob``, ``zinb_log_prob``, ``nbvcp_log_prob``, ``zinbvcp_log_prob``
each implement both the non-mixture and the mixture branch for one of the
four NB-family variants; the BNB family shares them entirely.
"""

# Standard-library imports
from typing import Dict, Optional

# Scientific stack
import jax.numpy as jnp
import jax.scipy as jsp

# NumPyro distributions
import numpyro.distributions as dist

# NOTE: ``build_count_dist`` lives in ``beta_negative_binomial`` which
# imports (at module load) the four NB-family classes from
# ``negative_binomial``, ``zero_inflated``, and ``vcp``.  Those same four
# classes import this module to bind their ``.log_prob`` methods, so a
# top-level import here would produce a circular dependency.  We defer the
# import to first call time; Python caches module imports after that, so
# the lookup cost is effectively free.


# =========================================================================
# Core distribution builder
# =========================================================================


def _build_ll_count_dist(
    r: jnp.ndarray,
    p: jnp.ndarray,
    params: Dict,
) -> dist.Distribution:
    """Build NB or BNB count distribution for log-likelihood evaluation.

    The function delegates to :func:`build_count_dist` from
    ``beta_negative_binomial``.  When ``params`` contains
    ``"bnb_concentration"`` and ``r`` has already been reshaped to the
    mixture broadcast layout ``(1, n_genes, n_components)``, the concentration
    tensor is re-aligned so the resulting :class:`BetaNegativeBinomial` can
    broadcast against ``r`` and ``p``.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion parameter.  In mixture log-likelihood paths this has
        already been reshaped to ``(1, n_genes, n_components)``.
    p : jnp.ndarray
        Success probability, pre-clamped to ``(eps, 1 - eps)``.
    params : dict
        Full posterior parameter dictionary; may carry
        ``"bnb_concentration"`` (per-gene excess dispersion) and, in
        mixture settings, a component-indexed version of it.

    Returns
    -------
    numpyro.distributions.Distribution
        Either :class:`NegativeBinomialProbs` or
        :class:`BetaNegativeBinomial` aligned with ``r`` and ``p`` so that
        a subsequent ``.log_prob(counts)`` call broadcasts correctly.
    """
    # Deferred import breaks the module-load cycle with
    # beta_negative_binomial (see note at top of module).
    from .beta_negative_binomial import build_count_dist

    # Look up the optional BNB concentration parameter.
    bnb_conc = params.get("bnb_concentration")
    # Align bnb_concentration with the mixture broadcast layout of r.
    if bnb_conc is not None and r.ndim == 3 and bnb_conc.ndim != r.ndim:
        # r is (1, G, K) in mixture LL paths.
        if bnb_conc.ndim == 2:
            # Component-specific: (K, G) -> (G, K) -> (1, G, K).
            bnb_conc = jnp.expand_dims(jnp.transpose(bnb_conc), axis=0)
        elif bnb_conc.ndim == 1:
            # Shared per-gene: (G,) -> (1, G, 1) to broadcast across K.
            bnb_conc = bnb_conc[jnp.newaxis, :, jnp.newaxis]
    return build_count_dist(r, p, bnb_conc)


# =========================================================================
# Validation helpers
# =========================================================================


def _validate_mixture_component_shapes(
    r: jnp.ndarray,
    probs: jnp.ndarray,
    n_components: int,
    context: str,
) -> None:
    """Validate component-axis alignment for mixture NB likelihood tensors.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion tensor after mixture reshaping; trailing axis layout is
        expected to be ``(..., n_genes, n_components)``.
    probs : jnp.ndarray
        Success-probability tensor (``p_hat`` after capture adjustment);
        trailing axis layout is expected to be
        ``(..., n_genes or 1, n_components or 1)``.
    n_components : int
        Number of active mixture components implied by ``mixing_weights``.
    context : str
        Human-readable caller context for error messages.

    Raises
    ------
    ValueError
        If the trailing component axis is inconsistent with
        ``n_components``.

    Notes
    -----
    The explicit check surfaces a clear error when component pruning left
    the posterior tensors inconsistent (for example, ``mixing_weights``
    pruned to ``K=3`` while canonical ``p``/``r`` stayed at ``K=4``)
    rather than a low-level JAX broadcasting failure downstream.
    """
    if r.shape[-1] != n_components:
        raise ValueError(
            f"{context}: dispersion tensor has incompatible component axis. "
            f"Expected r.shape[-1] == {n_components}, got r.shape="
            f"{tuple(r.shape)}."
        )
    if probs.shape[-1] not in (1, n_components):
        # Heuristic hint: flag an axis swap when the *second-to-last* axis
        # matches the active component count - a common accidental layout
        # after mixture pruning.
        hint = ""
        if probs.ndim >= 2 and probs.shape[-2] == n_components:
            hint = (
                " Detected component-like size on probs.shape[-2], "
                "suggesting a swapped or stale component axis after "
                "mixture pruning."
            )
        raise ValueError(
            f"{context}: probability tensor has incompatible component "
            f"axis. Expected probs.shape[-1] in {{1, {n_components}}}, "
            f"got probs.shape={tuple(probs.shape)}.{hint}"
        )


def _check_return_by(return_by: str) -> None:
    """Raise ``ValueError`` unless ``return_by`` is ``"cell"`` or ``"gene"``."""
    if return_by not in ("cell", "gene"):
        raise ValueError("return_by must be one of ['cell', 'gene']")


def _check_weight_type(weight_type: Optional[str]) -> None:
    """Raise ``ValueError`` on unsupported ``weight_type`` strings."""
    if weight_type is not None and weight_type not in (
        "multiplicative",
        "additive",
    ):
        raise ValueError(
            "weight_type must be one of ['multiplicative', 'additive']"
        )


def _validate_weights(
    weights: Optional[jnp.ndarray],
    return_by: str,
    n_cells: int,
    n_genes: int,
    dtype: jnp.dtype,
) -> Optional[jnp.ndarray]:
    """Cast ``weights`` to ``dtype`` after verifying its length.

    Parameters
    ----------
    weights : jnp.ndarray or None
        Per-gene (``return_by="cell"``) or per-cell (``return_by="gene"``)
        weighting array.  ``None`` disables weighting.
    return_by : str
        Axis selector; determines the expected length.
    n_cells, n_genes : int
        Observed dataset shape.
    dtype : jnp.dtype
        Output dtype (typically ``jnp.float32``).

    Returns
    -------
    jnp.ndarray or None
        ``weights`` cast to ``dtype`` (or ``None`` if input was ``None``).

    Raises
    ------
    ValueError
        If the shape is incompatible with ``return_by``.
    """
    if weights is None:
        return None
    expected_length = n_genes if return_by == "cell" else n_cells
    if len(weights) != expected_length:
        raise ValueError(
            f"For return_by='{return_by}', weights must be of shape "
            f"({expected_length},)"
        )
    return jnp.asarray(weights, dtype=dtype)


# =========================================================================
# Parameter preparation
# =========================================================================


def _normalize_counts(
    counts: jnp.ndarray,
    cells_axis: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Cast ``counts`` to ``dtype`` and orient as ``(n_cells, n_genes)``.

    Parameters
    ----------
    counts : jnp.ndarray
        Raw count matrix.
    cells_axis : int
        Axis along which cells are arranged.  ``0`` (default) means cells
        are rows; ``1`` means cells are columns and the matrix is
        transposed.
    dtype : jnp.dtype
        Target dtype (typically ``jnp.float32``).

    Returns
    -------
    jnp.ndarray
        Cell-major, ``dtype``-typed counts.
    """
    # Avoid an unnecessary copy when already in the expected layout.
    if not isinstance(counts, jnp.ndarray) or counts.dtype != dtype:
        counts = jnp.asarray(counts, dtype=dtype)
    # Transpose when genes are the leading axis.
    if cells_axis == 1:
        counts = jnp.transpose(counts)
    return counts


def _clip_r(r: jnp.ndarray, r_floor: float) -> jnp.ndarray:
    """Clamp ``r`` to ``[r_floor, +inf)`` to neutralise degenerate samples.

    Posterior draws from a wide variational guide occasionally underflow
    ``r`` below machine epsilon, which cascades into NaN log-likelihoods
    through ``lgamma(r)``.  A tiny floor costs nothing in the likelihood
    but eliminates the NaN sample.
    """
    if r_floor > 0.0:
        return jnp.maximum(r, r_floor)
    return r


def _clip_p(p: jnp.ndarray, p_floor: float) -> jnp.ndarray:
    """Clamp ``p`` to the open interval ``(p_floor, 1 - p_floor)``.

    Prevents ``log(0)`` and ``log(1-1)`` NaNs when gene-specific ``p``
    degenerately lands on an endpoint (hierarchical models make this
    reachable in float32).
    """
    if p_floor > 0.0:
        return jnp.clip(p, p_floor, 1.0 - p_floor)
    return p


def _extract_capture(params: Dict, dtype: jnp.dtype) -> jnp.ndarray:
    """Return ``p_capture`` from ``params`` under either parameterisation.

    SCRIBE supports two capture parameterisations:

    - Native constrained: ``p_capture`` directly (``(n_cells,)``).
    - Odds-ratio:         ``phi_capture``; the capture probability is
      recovered as ``p = 1 / (1 + phi)``.

    Parameters
    ----------
    params : dict
        Parameter dictionary that must contain either ``"p_capture"`` or
        ``"phi_capture"``.
    dtype : jnp.dtype
        Output dtype.

    Returns
    -------
    jnp.ndarray
        Cell-specific capture probability with shape ``(n_cells,)``.
    """
    if "phi_capture" in params:
        phi_capture = jnp.squeeze(params["phi_capture"]).astype(dtype)
        return 1.0 / (1.0 + phi_capture)
    return jnp.squeeze(params["p_capture"]).astype(dtype)


def _prepare_p_mixture_layout(
    p: jnp.ndarray,
    n_components: int,
) -> jnp.ndarray:
    """
    `Reshape ``p`` to the mixture broadcast layout
    ``(1, G, K) or (1, 1, K) or (1,1,1)``.

    The dispersion and count tensors in mixture log-likelihoods live in the
    canonical ``(n_cells, n_genes, n_components)`` layout, so ``p`` must be
    expanded accordingly depending on its original shape:

    * scalar                  -> ``(1, 1, 1)``
    * ``(n_components,)``      -> ``(1, 1, n_components)``
    * ``(n_components, n_genes)`` -> ``(1, n_genes, n_components)``

    Parameters
    ----------
    p : jnp.ndarray
        Raw ``p`` tensor as stored in the posterior.
    n_components : int
        Active component count implied by ``mixing_weights``.

    Returns
    -------
    jnp.ndarray
        ``p`` expanded into the mixture broadcast layout.
    """
    # Gene-specific, component-indexed: (K, G) - distinguishable from the
    # (K,) case via p.shape[1] > 1.
    p_is_gene_specific = (
        p.ndim == 2 and p.shape[0] == n_components and p.shape[1] > 1
    )
    # Component-only: (K,)
    p_is_component_specific = (
        not p_is_gene_specific and p.ndim >= 1 and p.shape[0] == n_components
    )
    if p_is_gene_specific:
        # (K, G) -> (G, K) -> (1, G, K)
        return jnp.expand_dims(jnp.transpose(p), axis=0)
    if p_is_component_specific:
        # (K,) -> (1, 1, K)
        return jnp.expand_dims(p, axis=(0, 1))
    # Shared scalar: () -> (1, 1, 1)
    return jnp.array(p)[None, None, None]


# =========================================================================
# Mixture reduction helper
# =========================================================================


def _mixture_reduce(
    gene_log_probs: jnp.ndarray,
    mixing_weights: jnp.ndarray,
    return_by: str,
    split_components: bool,
    weights: Optional[jnp.ndarray],
    weight_type: Optional[str],
) -> jnp.ndarray:
    """Reduce ``(n_cells, n_genes, n_components)`` log probs to mixture LL.

    Applies optional gene/cell weights, sums over the appropriate
    observation axis, adds log mixing weights, and (optionally) collapses
    over components via ``logsumexp``.

    Parameters
    ----------
    gene_log_probs : jnp.ndarray
        Elementwise log probabilities with shape
        ``(n_cells, n_genes, n_components)``.
    mixing_weights : jnp.ndarray
        Component mixing weights with shape ``(n_components,)``.
    return_by : {"cell", "gene"}
        ``"cell"`` sums over genes (axis 1); ``"gene"`` sums over cells
        (axis 0).
    split_components : bool
        If ``True`` return the per-component matrix; otherwise collapse
        via ``logsumexp`` over the component axis.
    weights : jnp.ndarray or None
        Optional per-gene (``return_by="cell"``) or per-cell
        (``return_by="gene"``) weighting array.
    weight_type : {"multiplicative", "additive"} or None
        How ``weights`` enters: ``"multiplicative"`` scales the log probs;
        ``"additive"`` adds them (after axis alignment matching the
        original free-function semantics).

    Returns
    -------
    jnp.ndarray
        Shape depends on ``return_by`` / ``split_components``:

        - ``cell`` + not split:  ``(n_cells,)``
        - ``cell`` + split:      ``(n_cells, n_components)``
        - ``gene`` + not split:  ``(n_genes,)``
        - ``gene`` + split:      ``(n_genes, n_components)``
    """
    # Apply weights first: behaviour preserves the axis conventions of the
    # legacy free-function implementation exactly.
    if weights is not None and weight_type is not None:
        if return_by == "cell":
            # Cell branch: weights has shape (n_genes,).
            if weight_type == "multiplicative":
                # Legacy behaviour: direct multiplication (rightmost-axis
                # broadcast). Preserved verbatim for numerical parity.
                gene_log_probs = gene_log_probs * weights
            else:  # additive
                # Expand to (1, n_genes, 1) so it aligns with the gene
                # axis of gene_log_probs.
                gene_log_probs = gene_log_probs + jnp.expand_dims(
                    weights, axis=(0, -1)
                )
        else:  # return_by == "gene"
            # Gene branch: weights has shape (n_cells,).
            if weight_type == "multiplicative":
                gene_log_probs = gene_log_probs * weights
            else:  # additive
                # Expand to (1, 1, n_cells) -> broadcasts against (n_cells,
                # n_genes, n_components) via rightmost-axis alignment.
                gene_log_probs = gene_log_probs + jnp.expand_dims(
                    weights, axis=(0, 1)
                )

    # Reduce over the observation axis to obtain per-(cell|gene, component)
    # log likelihoods.  The log mixing weights are added at the end, with
    # the broadcasting shape chosen to match the legacy semantics.
    if return_by == "cell":
        # Shape: (n_cells, n_components)
        log_probs = jnp.sum(gene_log_probs, axis=1) + jnp.log(mixing_weights)
    else:
        # Shape: (n_genes, n_components); .T makes (n_components,) align
        # with axis 1 after broadcasting.
        log_probs = jnp.sum(gene_log_probs, axis=0) + jnp.log(mixing_weights).T

    if split_components:
        return log_probs
    # Marginalise over components via logsumexp for numerical stability.
    return jsp.special.logsumexp(log_probs, axis=1)


# =========================================================================
# NB family delegate functions
# =========================================================================


def nb_log_prob(
    counts: jnp.ndarray,
    params: Dict,
    *,
    return_by: str = "cell",
    cells_axis: int = 0,
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
) -> jnp.ndarray:
    """Full-array log likelihood for NB / NBDM (mixture or non-mixture).

    Dispatches on ``"mixing_weights" in params`` to pick the mixture or
    non-mixture branch.  Both branches reduce to a single full-array
    ``dist.log_prob(counts)`` call followed by axis reduction - no
    Python-level loops, no scatter updates.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed count matrix of shape ``(n_cells, n_genes)`` (or
        transposed when ``cells_axis=1``).
    params : dict
        Posterior parameter dictionary.  Required keys vary by model:

        - Non-mixture: ``"p"``, ``"r"`` (and optionally
          ``"bnb_concentration"`` for BNB).
        - Mixture:     the above plus ``"mixing_weights"``.
    return_by : {"cell", "gene"}, default="cell"
        Output reduction axis.
    cells_axis : int, default=0
        Orientation of ``counts``.
    r_floor, p_floor : float, default=1e-6
        Numerical floors applied to ``r`` and ``p`` respectively.  See
        :func:`_clip_r` and :func:`_clip_p`.
    dtype : jnp.dtype, default=jnp.float32
        Floating-point dtype used throughout.
    split_components : bool, default=False
        Mixture-only: if ``True`` return per-component log probs.
    weights : jnp.ndarray or None, default=None
        Mixture-only: optional weighting array; see :func:`_mixture_reduce`.
    weight_type : {"multiplicative", "additive"} or None, default=None
        Mixture-only: weighting mode.

    Returns
    -------
    jnp.ndarray
        Log-likelihood array whose shape follows the standard SCRIBE
        contract:

        - Non-mixture:   ``(n_cells,)`` or ``(n_genes,)``.
        - Mixture split: ``(n_cells, n_components)`` or
          ``(n_genes, n_components)``.
        - Mixture marginal: ``(n_cells,)`` or ``(n_genes,)``.
    """
    # Input validation mirrors the legacy free functions so existing
    # callers observe identical error messages.
    _check_return_by(return_by)
    _check_weight_type(weight_type)

    # Normalise layout and cast to the requested working dtype.
    counts = _normalize_counts(counts, cells_axis, dtype)
    n_cells, n_genes = counts.shape

    # Extract and sanitise the shared NB parameters.
    p = _clip_p(jnp.squeeze(params["p"]).astype(dtype), p_floor)
    r = _clip_r(jnp.squeeze(params["r"]).astype(dtype), r_floor)

    is_mixture = "mixing_weights" in params
    if not is_mixture:
        # -----------------------------------------------------------------
        # Non-mixture branch: single NB/BNB distribution vectorised across
        # cells and genes.  ``to_event(1)`` sums over the gene axis when
        # computing per-cell log probs; for gene-wise output we skip
        # to_event and sum across cells explicitly.
        # -----------------------------------------------------------------
        base_dist = _build_ll_count_dist(r, p, params)
        if return_by == "cell":
            return base_dist.to_event(1).log_prob(counts)
        return jnp.sum(base_dist.log_prob(counts), axis=0)

    # -----------------------------------------------------------------
    # Mixture branch: broadcast to (n_cells, n_genes, n_components).
    # -----------------------------------------------------------------
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]

    # Validate optional gene/cell weights up front.
    weights = _validate_weights(weights, return_by, n_cells, n_genes, dtype)

    # Broadcast layout: counts (n_cells, G, 1), r (1, G, K), p flexible.
    counts_b = jnp.expand_dims(counts, axis=-1)
    r_b = jnp.expand_dims(jnp.transpose(r), axis=0)
    p_b = _prepare_p_mixture_layout(p, n_components)

    # Single full-array log_prob: shape (n_cells, n_genes, n_components).
    gene_log_probs = _build_ll_count_dist(r_b, p_b, params).log_prob(counts_b)

    return _mixture_reduce(
        gene_log_probs,
        mixing_weights,
        return_by=return_by,
        split_components=split_components,
        weights=weights,
        weight_type=weight_type,
    )


def zinb_log_prob(
    counts: jnp.ndarray,
    params: Dict,
    *,
    return_by: str = "cell",
    cells_axis: int = 0,
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
) -> jnp.ndarray:
    """Full-array log likelihood for ZINB / ZINB-mixture models.

    Identical pipeline to :func:`nb_log_prob` but wraps the base count
    distribution in a :class:`ZeroInflatedDistribution` with gene/component
    -specific ``gate``.

    Parameters
    ----------
    counts, params, return_by, cells_axis, r_floor, p_floor, dtype,
    split_components, weights, weight_type
        See :func:`nb_log_prob`.  ``params`` must additionally contain
        ``"gate"`` (shape ``(n_genes,)`` non-mixture or
        ``(n_components, n_genes)`` mixture).

    Returns
    -------
    jnp.ndarray
        Log-likelihood array; shape contract matches :func:`nb_log_prob`.
    """
    _check_return_by(return_by)
    _check_weight_type(weight_type)

    counts = _normalize_counts(counts, cells_axis, dtype)
    n_cells, n_genes = counts.shape

    p = _clip_p(jnp.squeeze(params["p"]).astype(dtype), p_floor)
    r = _clip_r(jnp.squeeze(params["r"]).astype(dtype), r_floor)
    gate = jnp.asarray(params["gate"]).astype(dtype)

    is_mixture = "mixing_weights" in params
    if not is_mixture:
        # Non-mixture: gate is expected to already be the squeezed
        # per-gene tensor.  Legacy code applied ``jnp.squeeze``; keep it.
        gate_nm = jnp.squeeze(gate)
        base_dist = _build_ll_count_dist(r, p, params)
        zi_dist = dist.ZeroInflatedDistribution(base_dist, gate=gate_nm)
        if return_by == "cell":
            return zi_dist.to_event(1).log_prob(counts)
        return jnp.sum(zi_dist.log_prob(counts), axis=0)

    # Mixture branch.
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]
    weights = _validate_weights(weights, return_by, n_cells, n_genes, dtype)

    # Ensure gate has at least two dims so it can be transposed into the
    # mixture broadcast layout.
    if gate.ndim < 2:
        gate = gate[jnp.newaxis, :]

    counts_b = jnp.expand_dims(counts, axis=-1)
    r_b = jnp.expand_dims(jnp.transpose(r), axis=0)
    # gate: (K, G) -> (G, K) -> (1, G, K)
    gate_b = jnp.expand_dims(jnp.transpose(gate), axis=0)
    p_b = _prepare_p_mixture_layout(p, n_components)

    base_dist = _build_ll_count_dist(r_b, p_b, params)
    zi_dist = dist.ZeroInflatedDistribution(base_dist, gate=gate_b)
    gene_log_probs = zi_dist.log_prob(counts_b)

    return _mixture_reduce(
        gene_log_probs,
        mixing_weights,
        return_by=return_by,
        split_components=split_components,
        weights=weights,
        weight_type=weight_type,
    )


def nbvcp_log_prob(
    counts: jnp.ndarray,
    params: Dict,
    *,
    return_by: str = "cell",
    cells_axis: int = 0,
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
) -> jnp.ndarray:
    """Full-array log likelihood for NBVCP / NBVCP-mixture models.

    Combines the NB pipeline with a cell-specific capture adjustment:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))

    ``p_hat`` is clamped to ``(p_floor, 1 - p_floor)`` to prevent NaN when
    ``p_capture`` underflows to 0 or when ``p`` saturates at 1.

    Parameters
    ----------
    counts, params, return_by, cells_axis, r_floor, p_floor, dtype,
    split_components, weights, weight_type
        See :func:`nb_log_prob`.  ``params`` must additionally contain
        ``"p_capture"`` *or* ``"phi_capture"``.

    Returns
    -------
    jnp.ndarray
        Log-likelihood array; shape contract matches :func:`nb_log_prob`.
    """
    _check_return_by(return_by)
    _check_weight_type(weight_type)

    counts = _normalize_counts(counts, cells_axis, dtype)
    n_cells, n_genes = counts.shape

    p = _clip_p(jnp.squeeze(params["p"]).astype(dtype), p_floor)
    r = _clip_r(jnp.squeeze(params["r"]).astype(dtype), r_floor)
    p_capture = _extract_capture(params, dtype)

    is_mixture = "mixing_weights" in params
    if not is_mixture:
        # Non-mixture: broadcast p_capture over genes. Shape (n_cells, 1).
        p_capture_nm = p_capture[:, None]
        # Effective capture-adjusted success probability.
        p_hat = p * p_capture_nm / (1 - p * (1 - p_capture_nm))
        # Guard against NaN at the endpoints of (0, 1).
        p_hat = _clip_p(p_hat, p_floor)
        base_dist = _build_ll_count_dist(r, p_hat, params)
        if return_by == "cell":
            return base_dist.to_event(1).log_prob(counts)
        return jnp.sum(base_dist.log_prob(counts), axis=0)

    # Mixture branch.
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]
    weights = _validate_weights(weights, return_by, n_cells, n_genes, dtype)

    counts_b = jnp.expand_dims(counts, axis=-1)
    r_b = jnp.expand_dims(jnp.transpose(r), axis=0)
    # p_capture: (n_cells,) -> (n_cells, 1, 1)
    p_capture_b = jnp.expand_dims(p_capture, axis=(-1, -2))
    p_b = _prepare_p_mixture_layout(p, n_components)

    # Effective probability, shape (n_cells, n_genes, n_components) or a
    # broadcast-compatible variant like (n_cells, 1, n_components).
    p_hat = p_b * p_capture_b / (1 - p_b * (1 - p_capture_b))
    _validate_mixture_component_shapes(
        r=r_b,
        probs=p_hat,
        n_components=n_components,
        context="nbvcp_log_prob",
    )
    p_hat = _clip_p(p_hat, p_floor)

    gene_log_probs = _build_ll_count_dist(r_b, p_hat, params).log_prob(counts_b)

    return _mixture_reduce(
        gene_log_probs,
        mixing_weights,
        return_by=return_by,
        split_components=split_components,
        weights=weights,
        weight_type=weight_type,
    )


def zinbvcp_log_prob(
    counts: jnp.ndarray,
    params: Dict,
    *,
    return_by: str = "cell",
    cells_axis: int = 0,
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
) -> jnp.ndarray:
    """Full-array log likelihood for ZINBVCP / ZINBVCP-mixture models.

    Combines the VCP capture adjustment with zero-inflation:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))
        counts ~ ZeroInflated(NB(r, p_hat), gate)

    Parameters
    ----------
    counts, params, return_by, cells_axis, r_floor, p_floor, dtype,
    split_components, weights, weight_type
        See :func:`nb_log_prob`.  ``params`` must additionally contain
        ``"gate"`` and either ``"p_capture"`` or ``"phi_capture"``.

    Returns
    -------
    jnp.ndarray
        Log-likelihood array; shape contract matches :func:`nb_log_prob`.
    """
    _check_return_by(return_by)
    _check_weight_type(weight_type)

    counts = _normalize_counts(counts, cells_axis, dtype)
    n_cells, n_genes = counts.shape

    p = _clip_p(jnp.squeeze(params["p"]).astype(dtype), p_floor)
    r = _clip_r(jnp.squeeze(params["r"]).astype(dtype), r_floor)
    gate = jnp.asarray(params["gate"]).astype(dtype)
    p_capture = _extract_capture(params, dtype)

    is_mixture = "mixing_weights" in params
    if not is_mixture:
        # Non-mixture: legacy code squeezed gate, broadcast p_capture.
        gate_nm = jnp.squeeze(gate)
        p_capture_nm = p_capture[:, None]
        p_hat = p * p_capture_nm / (1 - p * (1 - p_capture_nm))
        p_hat = _clip_p(p_hat, p_floor)
        base_dist = _build_ll_count_dist(r, p_hat, params)
        zi_dist = dist.ZeroInflatedDistribution(base_dist, gate=gate_nm)
        if return_by == "cell":
            return zi_dist.to_event(1).log_prob(counts)
        return jnp.sum(zi_dist.log_prob(counts), axis=0)

    # Mixture branch.
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]
    weights = _validate_weights(weights, return_by, n_cells, n_genes, dtype)

    if gate.ndim < 2:
        gate = gate[jnp.newaxis, :]

    counts_b = jnp.expand_dims(counts, axis=-1)
    r_b = jnp.expand_dims(jnp.transpose(r), axis=0)
    gate_b = jnp.expand_dims(jnp.transpose(gate), axis=0)
    p_capture_b = jnp.expand_dims(p_capture, axis=(-1, -2))
    p_b = _prepare_p_mixture_layout(p, n_components)

    p_hat = p_b * p_capture_b / (1 - p_b * (1 - p_capture_b))
    _validate_mixture_component_shapes(
        r=r_b,
        probs=p_hat,
        n_components=n_components,
        context="zinbvcp_log_prob",
    )
    p_hat = _clip_p(p_hat, p_floor)

    base_dist = _build_ll_count_dist(r_b, p_hat, params)
    zi_dist = dist.ZeroInflatedDistribution(base_dist, gate=gate_b)
    gene_log_probs = zi_dist.log_prob(counts_b)

    return _mixture_reduce(
        gene_log_probs,
        mixing_weights,
        return_by=return_by,
        split_components=split_components,
        weights=weights,
        weight_type=weight_type,
    )
