"""Shared helpers for sample-dimension detection and per-draw slicing."""

from typing import Dict, Optional

import jax.numpy as jnp


def _build_canonical_layouts(
    samples: Dict[str, "jnp.ndarray"],
    model_config,
    *,
    n_genes: int,
    n_cells: Optional[int] = None,
    n_components: Optional[int] = None,
    has_sample_dim: bool = False,
) -> Dict[str, "AxisLayout"]:
    """Build semantic layouts keyed by canonical parameter names.

    The returned dict maps canonical keys (``"r"``, ``"p"``, ``"gate"``,
    ``"mixing_weights"``, ``"p_capture"``, ``"bnb_concentration"``, …)
    to their :class:`AxisLayout` descriptors.

    Internally delegates to :func:`build_sample_layouts` which uses
    ``param_specs`` where available and falls back to shape-based
    inference for derived/canonical keys.

    Parameters
    ----------
    samples : dict of str to jnp.ndarray
        Parameter arrays keyed by *canonical* names.  May be MAP
        estimates or posterior draws — ``has_sample_dim`` controls
        whether a leading sample axis is expected.
    model_config
        Model configuration object (supplies ``param_specs``,
        ``n_components``, ``n_datasets``, ``mixture_params``, etc.).
    n_genes : int
        Number of genes in the data.
    n_cells : int, optional
        Number of cells.
    n_components : int, optional
        Mixture components.  Falls back to ``model_config.n_components``.
    has_sample_dim : bool, default False
        Whether tensors in *samples* carry a leading posterior-draw axis.

    Returns
    -------
    dict of str to AxisLayout
        Layouts keyed by canonical parameter names, with
        ``has_sample_dim`` set appropriately.
    """
    # Lazy import to avoid circular dependency (sampling is
    # imported by svi/mcmc modules that also import from it).
    from ..core.axis_layout import build_sample_layouts, derive_axis_membership

    specs = getattr(model_config, "param_specs", None) or []

    # Unified derivation of mixture_params and dataset_params from
    # model_config.  Replaces ~80 lines of inline logic that duplicated
    # derive_axis_membership's cascade (HierarchicalPriorType flags,
    # concat shape scan, derived-param expansion).
    _mp, ds_params = derive_axis_membership(
        model_config, samples=samples, has_sample_dim=has_sample_dim,
    )

    _nc = (
        n_components
        if n_components is not None
        else getattr(model_config, "n_components", None)
    )

    return build_sample_layouts(
        specs,
        samples,
        n_genes=n_genes,
        n_cells=n_cells,
        n_components=_nc,
        n_datasets=getattr(model_config, "n_datasets", None),
        mixture_params=_mp,
        dataset_params=ds_params,
        has_sample_dim=has_sample_dim,
    )


def _has_sample_dim(
    param_layouts: Dict[str, "AxisLayout"],
) -> bool:
    """Determine whether ``r`` has a leading posterior-sample dimension.

    Reads the answer directly from the ``AxisLayout`` metadata stored
    in ``param_layouts["r"]`` — no shape or ``ndim`` inspection needed.

    Parameters
    ----------
    param_layouts : dict of str to AxisLayout
        Semantic axis layouts keyed by canonical parameter name.
        Must contain an ``"r"`` entry.

    Returns
    -------
    bool
        ``True`` when ``r`` carries a leading sample axis.
    """
    return param_layouts["r"].has_sample_dim


def _slice_draw(
    arr: Optional[jnp.ndarray],
    layout: Optional["AxisLayout"],
    idx: int,
) -> Optional[jnp.ndarray]:
    """Slice the sample dimension of a single parameter at index ``idx``.

    When ``layout.has_sample_dim`` is ``True``, returns ``arr[idx]``.
    Otherwise returns ``arr`` unchanged.  ``None`` arrays pass through.

    Parameters
    ----------
    arr : jnp.ndarray or None
        Parameter array, possibly with a leading sample axis.
    layout : AxisLayout or None
        Semantic layout for this parameter.  ``None`` means no layout
        information is available (should not happen in normal flow).
    idx : int
        Index of the posterior draw to extract.

    Returns
    -------
    jnp.ndarray or None
        The parameter for a single draw.
    """
    if arr is None or layout is None:
        return arr
    return arr[idx] if layout.has_sample_dim else arr


def _slice_posterior_draw(
    idx: int,
    *,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    mixing_weights: Optional[jnp.ndarray],
    param_layouts: Dict[str, "AxisLayout"],
    bnb_concentration: Optional[jnp.ndarray] = None,
) -> dict:
    """Extract parameter values for a single posterior draw.

    Uses ``AxisLayout.has_sample_dim`` per parameter to decide whether
    to index into the leading axis — no ``ndim`` / ``is_mixture``
    heuristics are needed.

    Parameters
    ----------
    idx : int
        Index of the posterior draw to extract.
    r : jnp.ndarray
        NB dispersion array, with a leading sample axis.
    p : jnp.ndarray
        Success probability, possibly with a leading sample axis.
    p_capture : jnp.ndarray or None
        Capture probability, or ``None``.
    gate : jnp.ndarray or None
        Zero-inflation gate, or ``None``.
    mixing_weights : jnp.ndarray or None
        Component mixing weights, or ``None``.
    param_layouts : dict of str to AxisLayout
        Semantic axis layouts at the posterior level.  Used to determine
        which arrays carry a sample dimension that needs slicing.
    bnb_concentration : jnp.ndarray or None
        Optional BNB concentration.

    Returns
    -------
    dict
        Keys: ``r``, ``p``, ``p_capture``, ``gate``, ``mixing_weights``,
        ``bnb_concentration`` — each sliced to remove the sample axis.
    """
    return {
        "r": _slice_draw(r, param_layouts.get("r"), idx),
        "p": _slice_draw(p, param_layouts.get("p"), idx),
        "p_capture": _slice_draw(
            p_capture, param_layouts.get("p_capture"), idx
        ),
        "gate": _slice_draw(gate, param_layouts.get("gate"), idx),
        "mixing_weights": _slice_draw(
            mixing_weights, param_layouts.get("mixing_weights"), idx
        ),
        "bnb_concentration": _slice_draw(
            bnb_concentration, param_layouts.get("bnb_concentration"), idx
        ),
    }


def _slice_gene_axis(
    arr: Optional[jnp.ndarray],
    gene_axis: Optional[int],
    gene_indices: jnp.ndarray,
) -> Optional[jnp.ndarray]:
    """Subset the gene dimension of a tensor using a known axis index.

    If either the array or the gene axis is ``None`` the array is
    returned unchanged.  This is a convenience wrapper that replaces
    the pattern of branching on ``ndim`` / ``shape[-1] == n_genes``
    when the layout already tells us which axis carries genes.

    Parameters
    ----------
    arr : jnp.ndarray or None
        Tensor to subset. ``None`` is passed through.
    gene_axis : int or None
        Axis index carrying the gene dimension, as returned by
        ``AxisLayout.gene_axis``.  ``None`` means the tensor does not
        have a gene axis and should be returned unchanged.
    gene_indices : jnp.ndarray
        Integer indices selecting a subset of genes.

    Returns
    -------
    jnp.ndarray or None
        The input array with the gene dimension subsetted, or the
        original array when subsetting is not applicable.
    """
    if arr is None or gene_axis is None:
        return arr

    # Build an index tuple that slices only the gene axis
    slicer = [slice(None)] * arr.ndim
    slicer[gene_axis] = gene_indices
    return arr[tuple(slicer)]
