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

All axis handling is driven by semantic :class:`AxisLayout` metadata
passed in via the ``param_layouts`` kwarg; **no** ``ndim``/``shape``
heuristics are used anywhere in this module.  Layouts are passed by the
caller (SVI / MCMC log-likelihood mixins) via closure so that JAX treats
them as static during JIT compilation.

Shared helpers
~~~~~~~~~~~~~~
``_normalize_counts``, ``_clip_r``, ``_clip_p``, ``_validate_weights``,
``_check_return_by``, ``_check_weight_type`` perform trivial pre-processing
that every family needs.

``_prepare_mixture_tensor`` reshapes a parameter tensor into the canonical
``(1, n_genes, n_components)`` mixture broadcast layout using its
:class:`AxisLayout`.  ``_prepare_capture_mixture`` does the equivalent for
the cell-specific capture tensor.  ``_prepare_capture_non_mixture`` handles
the non-mixture variant ``(n_cells, 1)``.

``_build_ll_count_dist`` is now a thin wrapper over
:func:`~scribe.models.components.likelihoods.beta_negative_binomial.build_count_dist`
- the caller is responsible for supplying a correctly-shaped
``bnb_concentration`` (via :func:`_prepare_mixture_tensor` in the mixture
path).

``_mixture_reduce`` performs the final sum over cells or genes, applies
optional gene/cell weights, adds the log mixing weights, and optionally
collapses over components via ``logsumexp``.

Delegate functions
~~~~~~~~~~~~~~~~~~
``nb_log_prob``, ``zinb_log_prob``, ``nbvcp_log_prob``, ``zinbvcp_log_prob``
each implement both the non-mixture and the mixture branch for one of the
four NB-family variants; the BNB family shares them entirely.
"""

# Standard-library imports
from typing import Dict, Mapping, Optional

# Scientific stack
import jax.numpy as jnp
import jax.scipy as jsp

# NumPyro distributions
import numpyro.distributions as dist

# Semantic axis metadata (static during JIT -- always passed via closure by
# the caller, never as a dynamic argument).
from ....core.axis_layout import AxisLayout

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
    bnb_concentration: Optional[jnp.ndarray],
) -> dist.Distribution:
    """Build NB or BNB count distribution for log-likelihood evaluation.

    Thin wrapper over
    :func:`~scribe.models.components.likelihoods.beta_negative_binomial.build_count_dist`.
    The caller is responsible for shaping ``bnb_concentration`` so that it
    broadcasts against ``r`` and ``p``; in the mixture path this is done
    via :func:`_prepare_mixture_tensor` using the parameter's
    :class:`AxisLayout`.

    Parameters
    ----------
    r : jnp.ndarray
        NB dispersion, already broadcast-ready against ``p``.
    p : jnp.ndarray
        Success probability, already clamped and broadcast-ready.
    bnb_concentration : jnp.ndarray or None
        Pre-shaped BNB concentration (``None`` selects the plain NB path).

    Returns
    -------
    numpyro.distributions.Distribution
        Either :class:`NegativeBinomialProbs` or
        :class:`BetaNegativeBinomial`.
    """
    # Deferred import breaks the module-load cycle with
    # beta_negative_binomial (see note at top of module).
    from .beta_negative_binomial import build_count_dist

    return build_count_dist(r, p, bnb_concentration)


# =========================================================================
# Validation helpers (layout-driven; no shape heuristics)
# =========================================================================


def _validate_component_axis(
    tensor: jnp.ndarray,
    layout: AxisLayout,
    n_components: int,
    *,
    context: str,
    param_name: str,
) -> None:
    """Check that a tensor's component axis has the expected size.

    Parameters
    ----------
    tensor : jnp.ndarray
        Parameter tensor to check.
    layout : AxisLayout
        Semantic layout of *tensor*.
    n_components : int
        Expected number of mixture components (from ``mixing_weights``).
    context : str
        Caller identifier used in the error message (e.g.
        ``"nbvcp_log_prob"``).
    param_name : str
        Parameter name used in the error message (e.g. ``"r"``, ``"p"``).

    Raises
    ------
    ValueError
        If the layout declares a component axis whose size does not
        match *n_components*.
    """
    if layout.component_axis is None:
        return
    axis_size = tensor.shape[layout.component_axis]
    if axis_size != n_components:
        raise ValueError(
            f"{context}: component axis of '{param_name}' is inconsistent "
            f"with the mixture (expected size {n_components}, got "
            f"{axis_size}). This typically indicates that posterior "
            f"tensors became out of sync after mixture pruning - "
            f"mixing_weights has {n_components} active components but "
            f"'{param_name}' still has {axis_size}."
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


def _require_layout(
    param_layouts: Mapping[str, AxisLayout],
    key: str,
    *,
    context: str,
) -> AxisLayout:
    """Look up *key* in *param_layouts* with a clear error on miss.

    Parameters
    ----------
    param_layouts : Mapping[str, AxisLayout]
        Layout dictionary (typically produced by
        :func:`scribe.sampling._helpers._build_canonical_layouts`).
    key : str
        Canonical parameter name (``"p"``, ``"r"``, ``"gate"``,
        ``"p_capture"``, ``"mixing_weights"``, ``"bnb_concentration"``).
    context : str
        Caller identifier used in the error message.

    Returns
    -------
    AxisLayout
        The layout registered under *key*.

    Raises
    ------
    KeyError
        If *key* is not in *param_layouts*.
    """
    if key not in param_layouts:
        raise KeyError(
            f"{context}: param_layouts is missing an entry for '{key}'. "
            "Log-likelihood evaluation requires a semantic AxisLayout for "
            "every consumed parameter. Build layouts via "
            "scribe.sampling._helpers._build_canonical_layouts or pass "
            "ParamSpec-derived layouts directly."
        )
    return param_layouts[key]


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


def _extract_capture(
    params: Dict,
    param_layouts: Mapping[str, AxisLayout],
    dtype: jnp.dtype,
    *,
    context: str,
) -> tuple:
    """Return ``(p_capture, p_capture_layout)`` under either parameterisation.

    SCRIBE supports two capture parameterisations:

    - Native constrained: ``p_capture`` directly.
    - Odds-ratio:         ``phi_capture``; the capture probability is
      recovered as ``p = 1 / (1 + phi)``.

    Both carry the same :class:`AxisLayout` (``("cells",)``) so the layout
    is returned alongside the tensor for downstream broadcasting.

    Parameters
    ----------
    params : dict
        Parameter dictionary; must contain either ``"p_capture"`` or
        ``"phi_capture"``.
    param_layouts : Mapping[str, AxisLayout]
        Layouts for every parameter.
    dtype : jnp.dtype
        Output dtype.
    context : str
        Caller identifier used in error messages.

    Returns
    -------
    tuple of (jnp.ndarray, AxisLayout)
        Cell-specific capture probability and its layout.
    """
    if "phi_capture" in params:
        layout = _require_layout(param_layouts, "phi_capture", context=context)
        phi_capture = jnp.asarray(params["phi_capture"]).astype(dtype)
        return 1.0 / (1.0 + phi_capture), layout
    layout = _require_layout(param_layouts, "p_capture", context=context)
    return jnp.asarray(params["p_capture"]).astype(dtype), layout


def _prepare_non_mixture_tensor(
    tensor: jnp.ndarray,
    layout: AxisLayout,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Place a non-mixture parameter's gene axis at the trailing position.

    The non-mixture count distributions receive counts of shape
    ``(n_cells, n_genes)`` (after :func:`_normalize_counts`) and need
    ``r`` / ``p`` / ``gate`` to broadcast against the gene axis.  This
    helper moves the ``GENES`` axis to the last position and leaves any
    leading singleton (or missing) axes to NumPy/JAX broadcasting.

    Parameters
    ----------
    tensor : jnp.ndarray
        Parameter tensor.
    layout : AxisLayout
        Semantic layout of *tensor*.
    dtype : jnp.dtype
        Target dtype.

    Returns
    -------
    jnp.ndarray
        Dtype-cast tensor, oriented with the gene axis (if any) at index
        ``-1``.
    """
    tensor = jnp.asarray(tensor).astype(dtype)
    g_ax = layout.gene_axis
    if g_ax is not None and g_ax != tensor.ndim - 1:
        tensor = jnp.moveaxis(tensor, g_ax, -1)
    return tensor


def _prepare_mixture_tensor(
    tensor: jnp.ndarray,
    layout: AxisLayout,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Reshape a parameter tensor to the mixture broadcast layout.

    Mixture log-likelihoods evaluate over a canonical
    ``(n_cells, n_genes, n_components)`` grid.  This helper reshapes an
    input parameter tensor into a grid-compatible layout using its
    :class:`AxisLayout`:

    =================================  =====================
    Source layout                       Result shape
    =================================  =====================
    ``()``                              ``(1, 1, 1)``
    ``("components",)``                 ``(1, 1, K)``
    ``("genes",)``                      ``(1, G, 1)``
    ``("components", "genes")``         ``(1, G, K)``
    ``("genes", "components")``         ``(1, G, K)``
    =================================  =====================

    Any additional singleton leading axes present in *tensor* but not
    described by *layout* raise a ``ValueError`` - callers must supply
    layouts whose rank matches the tensor's ``ndim``.

    Parameters
    ----------
    tensor : jnp.ndarray
        Source parameter tensor.
    layout : AxisLayout
        Semantic layout (``has_sample_dim=False``).
    dtype : jnp.dtype
        Target dtype.

    Returns
    -------
    jnp.ndarray
        Tensor in the canonical mixture broadcast layout.
    """
    tensor = jnp.asarray(tensor).astype(dtype)
    if layout.has_sample_dim:
        raise ValueError(
            "_prepare_mixture_tensor expected a per-sample layout (i.e. "
            "has_sample_dim=False); callers must strip the sample axis "
            "before invoking this helper."
        )
    if tensor.ndim != layout.rank:
        raise ValueError(
            f"Tensor ndim={tensor.ndim} does not match layout rank="
            f"{layout.rank} (axes={layout.axes}). Supply a layout whose "
            "rank matches the tensor."
        )

    g_ax = layout.gene_axis
    k_ax = layout.component_axis

    if g_ax is None and k_ax is None:
        # Pure scalar -- treat as broadcast (1, 1, 1).
        return jnp.reshape(tensor, (1, 1, 1))
    if g_ax is None:
        # Only component axis present -> (1, 1, K).  The layout is
        # ``("components",)`` so tensor is 1D and a simple reshape
        # suffices.
        return jnp.reshape(tensor, (1, 1, -1))
    if k_ax is None:
        # Only gene axis present -> (1, G, 1).  Layout is
        # ``("genes",)`` so tensor is 1D.
        return jnp.reshape(tensor, (1, -1, 1))

    # Both axes present: move genes to -2 and components to -1, then
    # prepend a singleton cells axis.
    tensor = jnp.moveaxis(tensor, (g_ax, k_ax), (-2, -1))
    return jnp.expand_dims(tensor, axis=0)


def _prepare_capture_non_mixture(
    p_capture: jnp.ndarray,
    layout: AxisLayout,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Reshape ``p_capture`` for non-mixture broadcasting.

    Target layout is ``(n_cells, 1)`` so that element-wise products with
    ``p`` (shape ``(..., n_genes)``) produce a ``(n_cells, n_genes)``
    effective-probability grid.

    Parameters
    ----------
    p_capture : jnp.ndarray
        Cell-specific capture probability.
    layout : AxisLayout
        Layout of *p_capture* (expected ``("cells",)``).
    dtype : jnp.dtype
        Target dtype.

    Returns
    -------
    jnp.ndarray
        Reshaped tensor with shape ``(n_cells, 1)``.
    """
    p_capture = jnp.asarray(p_capture).astype(dtype)
    c_ax = layout.cell_axis
    if c_ax is None:
        # Scalar capture probability -- broadcast over cells implicitly.
        return jnp.reshape(p_capture, (1, 1))
    if c_ax != 0:
        p_capture = jnp.moveaxis(p_capture, c_ax, 0)
    return p_capture[:, None]


def _prepare_capture_mixture(
    p_capture: jnp.ndarray,
    layout: AxisLayout,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Reshape ``p_capture`` for the mixture broadcast grid.

    Target layout is ``(n_cells, 1, 1)`` so the capture probability
    broadcasts against ``(1, n_genes, n_components)``-shaped parameters.

    Parameters
    ----------
    p_capture : jnp.ndarray
        Cell-specific capture probability.
    layout : AxisLayout
        Layout of *p_capture* (expected ``("cells",)``).
    dtype : jnp.dtype
        Target dtype.

    Returns
    -------
    jnp.ndarray
        Reshaped tensor with shape ``(n_cells, 1, 1)`` (or ``(1, 1, 1)``
        when no cell axis is present).
    """
    p_capture = jnp.asarray(p_capture).astype(dtype)
    c_ax = layout.cell_axis
    if c_ax is None:
        return jnp.reshape(p_capture, (1, 1, 1))
    if c_ax != 0:
        p_capture = jnp.moveaxis(p_capture, c_ax, 0)
    return p_capture[:, None, None]


def _prepare_mixing_weights(
    mixing_weights: jnp.ndarray,
    layout: AxisLayout,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Return ``mixing_weights`` oriented so the component axis is last.

    The canonical layout is ``("components",)``; any deviation (e.g.
    stale transpose after mixture pruning) is corrected here so that
    downstream ``logsumexp`` / reduction logic in :func:`_mixture_reduce`
    is agnostic to the input orientation.

    Parameters
    ----------
    mixing_weights : jnp.ndarray
        Component mixing probabilities.
    layout : AxisLayout
        Semantic layout.
    dtype : jnp.dtype
        Target dtype.

    Returns
    -------
    jnp.ndarray
        Dtype-cast weights with shape ``(n_components,)``.
    """
    mixing_weights = jnp.asarray(mixing_weights).astype(dtype)
    k_ax = layout.component_axis
    if k_ax is None:
        # No component axis -- treat as scalar (degenerate 1-component case).
        return jnp.reshape(mixing_weights, (-1,))
    if k_ax != mixing_weights.ndim - 1:
        mixing_weights = jnp.moveaxis(mixing_weights, k_ax, -1)
    return jnp.reshape(mixing_weights, (mixing_weights.shape[-1],))


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
# BNB concentration helper (shared between NB and ZINB variants)
# =========================================================================


def _maybe_prepare_bnb(
    params: Dict,
    param_layouts: Mapping[str, AxisLayout],
    is_mixture: bool,
    dtype: jnp.dtype,
    *,
    context: str,
) -> Optional[jnp.ndarray]:
    """Extract and reshape ``bnb_concentration`` when present.

    Returns ``None`` when the parameter dictionary does not contain a
    BNB concentration (plain NB/ZINB path).  When present, the tensor is
    reshaped using :func:`_prepare_mixture_tensor` (mixture) or
    :func:`_prepare_non_mixture_tensor` (non-mixture) so it broadcasts
    against ``r`` and ``p``.

    Parameters
    ----------
    params : dict
        Posterior parameter dictionary.
    param_layouts : Mapping[str, AxisLayout]
        Layouts for every parameter.
    is_mixture : bool
        Whether the caller is in the mixture branch.
    dtype : jnp.dtype
        Target dtype.
    context : str
        Caller identifier used in error messages.

    Returns
    -------
    jnp.ndarray or None
        Reshaped concentration tensor, or ``None``.
    """
    if "bnb_concentration" not in params:
        return None
    layout = _require_layout(
        param_layouts, "bnb_concentration", context=context
    )
    if is_mixture:
        return _prepare_mixture_tensor(
            params["bnb_concentration"], layout, dtype
        )
    return _prepare_non_mixture_tensor(
        params["bnb_concentration"], layout, dtype
    )


# =========================================================================
# NB family delegate functions
# =========================================================================


def nb_log_prob(
    counts: jnp.ndarray,
    params: Dict,
    param_layouts: Mapping[str, AxisLayout],
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
    param_layouts : Mapping[str, AxisLayout]
        Semantic :class:`AxisLayout` for every parameter in *params*.
        Must be provided with ``has_sample_dim=False``.
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
    _check_return_by(return_by)
    _check_weight_type(weight_type)

    counts = _normalize_counts(counts, cells_axis, dtype)
    n_cells, n_genes = counts.shape

    # Look up layouts for the consumed parameters.
    p_layout = _require_layout(param_layouts, "p", context="nb_log_prob")
    r_layout = _require_layout(param_layouts, "r", context="nb_log_prob")

    is_mixture = "mixing_weights" in params

    if not is_mixture:
        # -----------------------------------------------------------------
        # Non-mixture branch: single NB/BNB distribution vectorised across
        # cells and genes.  ``to_event(1)`` sums over the gene axis when
        # computing per-cell log probs; for gene-wise output we skip
        # to_event and sum across cells explicitly.
        # -----------------------------------------------------------------
        p = _clip_p(
            _prepare_non_mixture_tensor(params["p"], p_layout, dtype), p_floor
        )
        r = _clip_r(
            _prepare_non_mixture_tensor(params["r"], r_layout, dtype), r_floor
        )
        bnb_conc = _maybe_prepare_bnb(
            params, param_layouts, is_mixture=False, dtype=dtype,
            context="nb_log_prob",
        )
        base_dist = _build_ll_count_dist(r, p, bnb_conc)
        if return_by == "cell":
            return base_dist.to_event(1).log_prob(counts)
        return jnp.sum(base_dist.log_prob(counts), axis=0)

    # ---------------------------------------------------------------------
    # Mixture branch: broadcast to (n_cells, n_genes, n_components).
    # ---------------------------------------------------------------------
    mw_layout = _require_layout(
        param_layouts, "mixing_weights", context="nb_log_prob"
    )
    mixing_weights = _prepare_mixing_weights(
        params["mixing_weights"], mw_layout, dtype
    )
    n_components = int(mixing_weights.shape[0])

    # Defensive axis check before we reshape anything.
    _validate_component_axis(
        jnp.asarray(params["r"]), r_layout, n_components,
        context="nb_log_prob", param_name="r",
    )
    _validate_component_axis(
        jnp.asarray(params["p"]), p_layout, n_components,
        context="nb_log_prob", param_name="p",
    )

    weights = _validate_weights(weights, return_by, n_cells, n_genes, dtype)

    counts_b = jnp.expand_dims(counts, axis=-1)
    r_b = _clip_r(
        _prepare_mixture_tensor(params["r"], r_layout, dtype), r_floor
    )
    p_b = _clip_p(
        _prepare_mixture_tensor(params["p"], p_layout, dtype), p_floor
    )
    bnb_conc = _maybe_prepare_bnb(
        params, param_layouts, is_mixture=True, dtype=dtype,
        context="nb_log_prob",
    )

    # Single full-array log_prob: shape (n_cells, n_genes, n_components).
    gene_log_probs = _build_ll_count_dist(r_b, p_b, bnb_conc).log_prob(counts_b)

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
    param_layouts: Mapping[str, AxisLayout],
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
    counts, params, param_layouts, return_by, cells_axis, r_floor, p_floor,
    dtype, split_components, weights, weight_type
        See :func:`nb_log_prob`.  ``params`` must additionally contain
        ``"gate"`` with a corresponding layout entry.

    Returns
    -------
    jnp.ndarray
        Log-likelihood array; shape contract matches :func:`nb_log_prob`.
    """
    _check_return_by(return_by)
    _check_weight_type(weight_type)

    counts = _normalize_counts(counts, cells_axis, dtype)
    n_cells, n_genes = counts.shape

    p_layout = _require_layout(param_layouts, "p", context="zinb_log_prob")
    r_layout = _require_layout(param_layouts, "r", context="zinb_log_prob")
    gate_layout = _require_layout(
        param_layouts, "gate", context="zinb_log_prob"
    )

    is_mixture = "mixing_weights" in params

    if not is_mixture:
        p = _clip_p(
            _prepare_non_mixture_tensor(params["p"], p_layout, dtype), p_floor
        )
        r = _clip_r(
            _prepare_non_mixture_tensor(params["r"], r_layout, dtype), r_floor
        )
        gate = _prepare_non_mixture_tensor(params["gate"], gate_layout, dtype)
        bnb_conc = _maybe_prepare_bnb(
            params, param_layouts, is_mixture=False, dtype=dtype,
            context="zinb_log_prob",
        )
        base_dist = _build_ll_count_dist(r, p, bnb_conc)
        zi_dist = dist.ZeroInflatedDistribution(base_dist, gate=gate)
        if return_by == "cell":
            return zi_dist.to_event(1).log_prob(counts)
        return jnp.sum(zi_dist.log_prob(counts), axis=0)

    # Mixture branch.
    mw_layout = _require_layout(
        param_layouts, "mixing_weights", context="zinb_log_prob"
    )
    mixing_weights = _prepare_mixing_weights(
        params["mixing_weights"], mw_layout, dtype
    )
    n_components = int(mixing_weights.shape[0])

    _validate_component_axis(
        jnp.asarray(params["r"]), r_layout, n_components,
        context="zinb_log_prob", param_name="r",
    )
    _validate_component_axis(
        jnp.asarray(params["p"]), p_layout, n_components,
        context="zinb_log_prob", param_name="p",
    )
    _validate_component_axis(
        jnp.asarray(params["gate"]), gate_layout, n_components,
        context="zinb_log_prob", param_name="gate",
    )

    weights = _validate_weights(weights, return_by, n_cells, n_genes, dtype)

    counts_b = jnp.expand_dims(counts, axis=-1)
    r_b = _clip_r(
        _prepare_mixture_tensor(params["r"], r_layout, dtype), r_floor
    )
    p_b = _clip_p(
        _prepare_mixture_tensor(params["p"], p_layout, dtype), p_floor
    )
    gate_b = _prepare_mixture_tensor(params["gate"], gate_layout, dtype)
    bnb_conc = _maybe_prepare_bnb(
        params, param_layouts, is_mixture=True, dtype=dtype,
        context="zinb_log_prob",
    )

    base_dist = _build_ll_count_dist(r_b, p_b, bnb_conc)
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
    param_layouts: Mapping[str, AxisLayout],
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
    counts, params, param_layouts, return_by, cells_axis, r_floor, p_floor,
    dtype, split_components, weights, weight_type
        See :func:`nb_log_prob`.  ``params`` must additionally contain
        ``"p_capture"`` *or* ``"phi_capture"`` with a corresponding layout.

    Returns
    -------
    jnp.ndarray
        Log-likelihood array; shape contract matches :func:`nb_log_prob`.
    """
    _check_return_by(return_by)
    _check_weight_type(weight_type)

    counts = _normalize_counts(counts, cells_axis, dtype)
    n_cells, n_genes = counts.shape

    p_layout = _require_layout(param_layouts, "p", context="nbvcp_log_prob")
    r_layout = _require_layout(param_layouts, "r", context="nbvcp_log_prob")
    p_capture, cap_layout = _extract_capture(
        params, param_layouts, dtype, context="nbvcp_log_prob"
    )

    is_mixture = "mixing_weights" in params

    if not is_mixture:
        # Non-mixture: broadcast p_capture over genes -> (n_cells, 1).
        p = _clip_p(
            _prepare_non_mixture_tensor(params["p"], p_layout, dtype), p_floor
        )
        r = _clip_r(
            _prepare_non_mixture_tensor(params["r"], r_layout, dtype), r_floor
        )
        p_capture_nm = _prepare_capture_non_mixture(
            p_capture, cap_layout, dtype
        )
        # Effective capture-adjusted success probability.
        p_hat = p * p_capture_nm / (1 - p * (1 - p_capture_nm))
        p_hat = _clip_p(p_hat, p_floor)
        bnb_conc = _maybe_prepare_bnb(
            params, param_layouts, is_mixture=False, dtype=dtype,
            context="nbvcp_log_prob",
        )
        base_dist = _build_ll_count_dist(r, p_hat, bnb_conc)
        if return_by == "cell":
            return base_dist.to_event(1).log_prob(counts)
        return jnp.sum(base_dist.log_prob(counts), axis=0)

    # Mixture branch.
    mw_layout = _require_layout(
        param_layouts, "mixing_weights", context="nbvcp_log_prob"
    )
    mixing_weights = _prepare_mixing_weights(
        params["mixing_weights"], mw_layout, dtype
    )
    n_components = int(mixing_weights.shape[0])

    _validate_component_axis(
        jnp.asarray(params["r"]), r_layout, n_components,
        context="nbvcp_log_prob", param_name="r",
    )
    _validate_component_axis(
        jnp.asarray(params["p"]), p_layout, n_components,
        context="nbvcp_log_prob", param_name="p",
    )

    weights = _validate_weights(weights, return_by, n_cells, n_genes, dtype)

    counts_b = jnp.expand_dims(counts, axis=-1)
    r_b = _clip_r(
        _prepare_mixture_tensor(params["r"], r_layout, dtype), r_floor
    )
    p_b = _clip_p(
        _prepare_mixture_tensor(params["p"], p_layout, dtype), p_floor
    )
    p_capture_b = _prepare_capture_mixture(p_capture, cap_layout, dtype)

    # Effective probability, shape (n_cells, n_genes, n_components) or a
    # broadcast-compatible variant.
    p_hat = p_b * p_capture_b / (1 - p_b * (1 - p_capture_b))
    p_hat = _clip_p(p_hat, p_floor)

    bnb_conc = _maybe_prepare_bnb(
        params, param_layouts, is_mixture=True, dtype=dtype,
        context="nbvcp_log_prob",
    )
    gene_log_probs = _build_ll_count_dist(r_b, p_hat, bnb_conc).log_prob(
        counts_b
    )

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
    param_layouts: Mapping[str, AxisLayout],
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
    counts, params, param_layouts, return_by, cells_axis, r_floor, p_floor,
    dtype, split_components, weights, weight_type
        See :func:`nb_log_prob`.  ``params`` must additionally contain
        ``"gate"`` and either ``"p_capture"`` or ``"phi_capture"`` with
        matching layout entries.

    Returns
    -------
    jnp.ndarray
        Log-likelihood array; shape contract matches :func:`nb_log_prob`.
    """
    _check_return_by(return_by)
    _check_weight_type(weight_type)

    counts = _normalize_counts(counts, cells_axis, dtype)
    n_cells, n_genes = counts.shape

    p_layout = _require_layout(param_layouts, "p", context="zinbvcp_log_prob")
    r_layout = _require_layout(param_layouts, "r", context="zinbvcp_log_prob")
    gate_layout = _require_layout(
        param_layouts, "gate", context="zinbvcp_log_prob"
    )
    p_capture, cap_layout = _extract_capture(
        params, param_layouts, dtype, context="zinbvcp_log_prob"
    )

    is_mixture = "mixing_weights" in params

    if not is_mixture:
        p = _clip_p(
            _prepare_non_mixture_tensor(params["p"], p_layout, dtype), p_floor
        )
        r = _clip_r(
            _prepare_non_mixture_tensor(params["r"], r_layout, dtype), r_floor
        )
        gate = _prepare_non_mixture_tensor(params["gate"], gate_layout, dtype)
        p_capture_nm = _prepare_capture_non_mixture(
            p_capture, cap_layout, dtype
        )
        p_hat = p * p_capture_nm / (1 - p * (1 - p_capture_nm))
        p_hat = _clip_p(p_hat, p_floor)
        bnb_conc = _maybe_prepare_bnb(
            params, param_layouts, is_mixture=False, dtype=dtype,
            context="zinbvcp_log_prob",
        )
        base_dist = _build_ll_count_dist(r, p_hat, bnb_conc)
        zi_dist = dist.ZeroInflatedDistribution(base_dist, gate=gate)
        if return_by == "cell":
            return zi_dist.to_event(1).log_prob(counts)
        return jnp.sum(zi_dist.log_prob(counts), axis=0)

    # Mixture branch.
    mw_layout = _require_layout(
        param_layouts, "mixing_weights", context="zinbvcp_log_prob"
    )
    mixing_weights = _prepare_mixing_weights(
        params["mixing_weights"], mw_layout, dtype
    )
    n_components = int(mixing_weights.shape[0])

    _validate_component_axis(
        jnp.asarray(params["r"]), r_layout, n_components,
        context="zinbvcp_log_prob", param_name="r",
    )
    _validate_component_axis(
        jnp.asarray(params["p"]), p_layout, n_components,
        context="zinbvcp_log_prob", param_name="p",
    )
    _validate_component_axis(
        jnp.asarray(params["gate"]), gate_layout, n_components,
        context="zinbvcp_log_prob", param_name="gate",
    )

    weights = _validate_weights(weights, return_by, n_cells, n_genes, dtype)

    counts_b = jnp.expand_dims(counts, axis=-1)
    r_b = _clip_r(
        _prepare_mixture_tensor(params["r"], r_layout, dtype), r_floor
    )
    p_b = _clip_p(
        _prepare_mixture_tensor(params["p"], p_layout, dtype), p_floor
    )
    gate_b = _prepare_mixture_tensor(params["gate"], gate_layout, dtype)
    p_capture_b = _prepare_capture_mixture(p_capture, cap_layout, dtype)

    p_hat = p_b * p_capture_b / (1 - p_b * (1 - p_capture_b))
    p_hat = _clip_p(p_hat, p_floor)

    bnb_conc = _maybe_prepare_bnb(
        params, param_layouts, is_mixture=True, dtype=dtype,
        context="zinbvcp_log_prob",
    )
    base_dist = _build_ll_count_dist(r_b, p_hat, bnb_conc)
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
