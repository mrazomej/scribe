"""
Sampling utilities for SCRIBE.

This module provides functions for posterior sampling, predictive sampling,
and posterior predictive checks (PPCs). It also provides:

* A **biological PPC** utility that strips technical noise parameters
  (capture probability, zero-inflation gate) and samples from the base
  Negative Binomial distribution only, reflecting the underlying biology
  without experimental artifacts.

* A **Bayesian denoising** utility that takes *observed* count matrices and
  posterior parameter estimates to compute the closed-form posterior of the
  true (pre-capture, pre-dropout) transcript counts.  See
  ``paper/_denoising.qmd`` for the full mathematical derivation.

Parameterization Convention
---------------------------
Throughout this module the canonical ``p`` follows the numpyro convention:
it is the ``probs`` argument of ``NegativeBinomialProbs``, i.e. the
probability of each Bernoulli trial producing a count.  The NB mean is
therefore ``r * p / (1 - p)``.  This is the *complement* of the paper's
:math:`p` (which appears as :math:`p^r` in the PMF).
"""

from jax import random, vmap
import jax.numpy as jnp
import numpyro.distributions as dist

from scribe.models.components.likelihoods.beta_negative_binomial import (
    build_count_dist,
)
from scribe.stats.quadrature import gauss_legendre_nodes_weights
from numpyro.infer import Predictive
from typing import Dict, Optional, Tuple, Union, Callable, List
from numpyro.infer import SVI
from numpyro.handlers import block


# ==============================================================================
# Shared helpers for sample-dimension detection and per-draw slicing
# ==============================================================================


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
    # Lazy imports to avoid circular dependency (sampling.py is
    # imported by svi/mcmc modules that also import from it).
    from .core.axis_layout import build_sample_layouts
    from .models.config import HierarchicalPriorType

    specs = getattr(model_config, "param_specs", None) or []

    # Derive dataset_params from hierarchical-prior flags when not
    # explicitly set (mirrors _derive_dataset_params in
    # svi/_parameter_extraction.py but inlined here to avoid
    # importing from the svi subpackage).
    _n_ds = getattr(model_config, "n_datasets", None)
    ds_params = getattr(model_config, "dataset_params", None)
    if ds_params is None:
        _NONE = HierarchicalPriorType.NONE
        ds: list = []
        param = getattr(model_config, "parameterization", "linked")
        if getattr(model_config, "expression_dataset_prior", _NONE) != _NONE:
            ds.append("r" if param in ("canonical", "standard") else "mu")
        if getattr(model_config, "prob_dataset_prior", _NONE) != _NONE:
            ds.append("phi" if param in ("mean_odds", "odds_ratio") else "p")
        if (
            getattr(model_config, "zero_inflation_dataset_prior", _NONE)
            != _NONE
        ):
            ds.append("gate")
        if (
            getattr(model_config, "overdispersion_dataset_prior", _NONE)
            != _NONE
        ):
            ds.append("bnb_concentration")

        # Concatenated multi-dataset results: n_datasets >= 2 but no
        # hierarchical priors were set (original fits were single-dataset).
        # The concat path promotes ALL non-cell canonical params to
        # dataset-specific, so treat every key in *samples* that has a
        # leading dimension equal to n_datasets as dataset-specific.
        if not ds and _n_ds is not None and _n_ds >= 2:
            from .core.axis_layout import _KNOWN_CELL_PARAMS

            _offset = 1 if has_sample_dim else 0
            for key, arr in samples.items():
                if not hasattr(arr, "shape"):
                    continue
                base = key.split("_loc")[0].split("_scale")[0]
                if base in _KNOWN_CELL_PARAMS:
                    continue
                if arr.ndim > _offset and arr.shape[_offset] == _n_ds:
                    ds.append(key)

        ds_params = ds if ds else None

    _nc = (
        n_components
        if n_components is not None
        else getattr(model_config, "n_components", None)
    )

    # Expand mixture_params and dataset_params to include derived
    # canonical counterparts.  When the model uses mean_odds
    # parameterization, mixture_params may be ["phi", "mu"] — but
    # _compute_canonical_parameters derives "r" and "p" from those
    # with identical component structure.  Without expansion,
    # infer_layout would fail to assign a component axis to derived
    # canonical keys like "r" and "p".
    #
    # Instead of hardcoding canonical pairs, we read the dep graph
    # from the parameterization strategy's DerivedParam list: any
    # derived param whose deps overlap with the current member set
    # inherits that axis membership (mirroring merge_layouts).
    _mp = getattr(model_config, "mixture_params", None)
    if _mp is not None or ds_params is not None:
        from .models.parameterizations import PARAMETERIZATIONS
        from .core.axis_layout import expand_membership_from_derived

        _param = getattr(model_config, "parameterization", "canonical")
        _derived = PARAMETERIZATIONS[_param].build_derived_params()
        if _derived:
            if _mp is not None:
                _mp = sorted(expand_membership_from_derived(_mp, _derived))
            if ds_params is not None:
                ds_params = sorted(
                    expand_membership_from_derived(ds_params, _derived)
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


# ------------------------------------------------------------------------------
# Posterior predictive samples
# ------------------------------------------------------------------------------


def sample_variational_posterior(
    guide: Callable,
    params: Dict,
    model: Callable,
    model_args: Dict,
    rng_key: Optional[random.PRNGKey] = None,
    n_samples: int = 100,
    return_sites: Optional[Union[str, List[str]]] = None,
    counts: Optional[jnp.ndarray] = None,
) -> Dict:
    """
    Sample parameters from the variational posterior distribution.

    Parameters
    ----------
    guide : Callable
        Guide function
    params : Dict
        Dictionary containing optimized variational parameters
    model : Callable
        Model function
    model_args : Dict
        Dictionary containing model arguments. For standard models, this is
        just the number of cells and genes. For mixture models, this is the
        number of cells, genes, and components.
    rng_key : random.PRNGKey
        JAX random number generator key
    n_samples : int, optional
        Number of posterior samples to generate (default: 100)
    return_sites : Optional[Union[str, List[str]]], optional
        Sites to return from the model. If None, returns all sites.
    counts : Optional[jnp.ndarray], optional
        Observed count matrix of shape (n_cells, n_genes). Required when using
        amortized capture probability (e.g., with
        amortization.capture.enabled=true). For non-amortized models, this can
        be None. Default: None.

    Returns
    -------
    Dict
        Dictionary containing samples from the variational posterior
    """
    # Create default RNG key if not provided (lazy initialization)
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    # Add counts to model_args if provided (needed for amortized guides)
    if counts is not None:
        model_args = {**model_args, "counts": counts}

    # Create predictive object for posterior parameter samples
    predictive_param = Predictive(guide, params=params, num_samples=n_samples)

    # Sample parameters from the variational posterior
    posterior_samples = predictive_param(rng_key, **model_args)

    # Also run the model to get deterministic sites.
    # We block the 'counts' site to prevent Predictive from sampling it,
    # which avoids a potentially huge memory allocation.
    blocked_model = block(model, hide=["counts"])
    predictive_model = Predictive(
        blocked_model, posterior_samples=posterior_samples
    )
    model_samples = predictive_model(rng_key, **model_args)

    # Combine samples from guide and model
    posterior_samples.update(model_samples)
    return posterior_samples


# ------------------------------------------------------------------------------


def generate_predictive_samples(
    model: Callable,
    posterior_samples: Dict,
    model_args: Dict,
    rng_key: random.PRNGKey,
) -> jnp.ndarray:
    """Generate predictive samples using posterior parameter samples.

    NumPyro's ``Predictive`` vectorises over the posterior-sample
    dimension automatically (controlled by its ``parallel`` flag).
    Cell-level batching is intentionally absent here: posterior samples
    are drawn at full-cell resolution, and the predictive model must
    replay with the same cell dimension.

    Parameters
    ----------
    model : Callable
        Model function.
    posterior_samples : Dict
        Dictionary containing samples from the variational posterior.
    model_args : Dict
        Dictionary containing model arguments (n_cells, n_genes,
        model_config, etc.).  Passed as keyword arguments to the model
        via ``Predictive``.
    rng_key : random.PRNGKey
        JAX random number generator key.

    Returns
    -------
    jnp.ndarray
        Array of predictive count samples.
    """
    # Find the first array value to get num_samples
    # Skip nested dicts from flax_module parameters (e.g., "amortizer$params")
    num_samples = None
    for value in posterior_samples.values():
        if hasattr(value, "shape"):
            num_samples = value.shape[0]
            break

    if num_samples is None:
        raise ValueError(
            "Could not determine num_samples from posterior_samples. "
            "No array values found (all values are nested dicts?)."
        )

    # Create predictive object for generating new data
    predictive = Predictive(
        model,
        posterior_samples,
        num_samples=num_samples,
        exclude_deterministic=False,
    )

    # NumPyro's Predictive.__call__ passes **kwargs directly to the
    # model.  We must NOT add extra kwargs (like batch_size) here --
    # they would leak into the model as cell-plate subsample_size and
    # create a shape mismatch with the full-cell posterior samples.
    predictive_samples = predictive(rng_key, **model_args)

    return predictive_samples["counts"]


# ------------------------------------------------------------------------------


def generate_ppc_samples(
    model: Callable,
    guide: Callable,
    params: Dict,
    model_args: Dict,
    rng_key: random.PRNGKey,
    n_samples: int = 100,
    counts: Optional[jnp.ndarray] = None,
) -> Dict:
    """Generate posterior predictive check samples.

    Parameters
    ----------
    model : Callable
        Model function.
    guide : Callable
        Guide function.
    params : Dict
        Dictionary containing optimized variational parameters.
    model_args : Dict
        Dictionary containing model arguments (n_cells, n_genes,
        model_config, etc.).
    rng_key : random.PRNGKey
        JAX random number generator key.
    n_samples : int, optional
        Number of posterior samples to generate (default: 100).
    counts : Optional[jnp.ndarray], optional
        Observed count matrix of shape (n_cells, n_genes).  Required
        when using amortized capture probability.  Default: None.

    Returns
    -------
    Dict
        Dictionary with keys ``parameter_samples`` and
        ``predictive_samples``.
    """
    # Split RNG key for parameter sampling and predictive sampling
    key_params, key_pred = random.split(rng_key)

    # Sample from variational posterior
    posterior_param_samples = sample_variational_posterior(
        guide, params, model, model_args, key_params, n_samples, counts=counts
    )

    # Generate predictive samples
    predictive_samples = generate_predictive_samples(
        model,
        posterior_param_samples,
        model_args,
        key_pred,
    )

    return {
        "parameter_samples": posterior_param_samples,
        "predictive_samples": predictive_samples,
    }


# ------------------------------------------------------------------------------


def generate_prior_predictive_samples(
    model: Callable,
    model_args: Dict,
    rng_key: random.PRNGKey,
    n_samples: int = 100,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """
    Generate prior predictive samples using the model.

    Parameters
    ----------
    model : Callable
        Model function
    model_args : Dict
        Dictionary containing model arguments. For standard models, this is
        just the number of cells and genes. For mixture models, this is the
        number of cells, genes, and components.
    rng_key : random.PRNGKey
        JAX random number generator key
    n_samples : int, optional
        Number of prior predictive samples to generate (default: 100)
    batch_size : int, optional
        Batch size for generating samples. If None, uses full dataset.

    Returns
    -------
    jnp.ndarray
        Array of prior predictive samples
    """
    # Create predictive object for generating new data from the prior
    predictive = Predictive(model, num_samples=n_samples)

    # Generate prior predictive samples
    prior_predictive_samples = predictive(
        rng_key, **model_args, batch_size=batch_size
    )

    return prior_predictive_samples["counts"]


# ------------------------------------------------------------------------------
# Biological (denoised) PPC sampling
# ------------------------------------------------------------------------------


def sample_biological_nb(
    r: jnp.ndarray,
    p: jnp.ndarray,
    n_cells: int,
    rng_key: random.PRNGKey,
    n_samples: int = 1,
    mixing_weights: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
    bnb_concentration: Optional[jnp.ndarray] = None,
    param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
) -> jnp.ndarray:
    """Sample from the base Negative Binomial, stripping technical noise.

    Generates count samples from the biological NB(r, p) distribution only,
    ignoring technical parameters such as capture probability (``p_capture``
    / ``phi_capture``) and zero-inflation gate. This reflects the "true"
    underlying gene expression as modeled by the Negative Binomial portion
    of the generative process (see the Dirichlet-Multinomial derivation in
    the paper supplement).

    For NBDM models this is equivalent to a standard PPC.  For VCP and ZINB
    variants it yields a *denoised* view of the data.

    The function supports both point estimates (MAP) and full posterior
    samples.  When ``r`` has a leading sample dimension the function uses
    ``jax.vmap`` to vectorise over samples efficiently.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion parameter.

        * Standard model, MAP: shape ``(n_genes,)``.
        * Standard model, posterior: shape ``(n_samples, n_genes)``.
        * Mixture model, MAP: shape ``(n_components, n_genes)``.
        * Mixture model, posterior: shape ``(n_samples, n_components,
          n_genes)``.
    p : jnp.ndarray
        Success probability of the Negative Binomial.

        * MAP: scalar or shape ``(n_components,)`` for component-specific p.
        * Posterior: shape ``(n_samples,)`` or ``(n_samples, n_components)``.
    n_cells : int
        Number of cells to generate counts for.
    rng_key : random.PRNGKey
        JAX PRNG key for reproducible sampling.
    n_samples : int, optional
        Number of posterior samples.  When ``r`` already has a leading sample
        dimension this is inferred automatically and this argument is
        ignored.  Default: 1.
    mixing_weights : jnp.ndarray or None, optional
        Component mixing weights for mixture models.

        * MAP: shape ``(n_components,)``.
        * Posterior: shape ``(n_samples, n_components)``.

        When ``None`` the model is treated as a standard (non-mixture) model.
    cell_batch_size : int or None, optional
        If set, cells are processed in batches of this size to limit peak
        memory usage.  When ``None`` all cells are sampled at once.

    Returns
    -------
    jnp.ndarray
        Sampled counts with shape ``(n_samples, n_cells, n_genes)``.

    Notes
    -----
    The mathematical justification is that the VCP model composes a base
    NB(r, p) with a Binomial capture step:

    .. math::
        \\hat{p} = \\frac{p \\cdot \\nu}{1 - p(1 - \\nu)}

    By sampling from NB(r, p) directly we bypass the capture distortion and
    any zero-inflation, recovering the latent biological distribution.

    Examples
    --------
    >>> # MAP-based biological PPC (standard model)
    >>> samples = sample_biological_nb(
    ...     r=map_estimates["r"],  # (n_genes,)
    ...     p=map_estimates["p"],  # scalar
    ...     n_cells=1000,
    ...     rng_key=jax.random.PRNGKey(0),
    ...     n_samples=5,
    ... )
    >>> samples.shape
    (5, 1000, 5)

    >>> # Full posterior biological PPC (mixture model)
    >>> samples = sample_biological_nb(
    ...     r=posterior["r"],                   # (100, 3, n_genes)
    ...     p=posterior["p"],                   # (100,)
    ...     n_cells=500,
    ...     rng_key=jax.random.PRNGKey(1),
    ...     mixing_weights=posterior["mixing_weights"],  # (100, 3)
    ... )
    >>> samples.shape
    (100, 500, n_genes)
    """
    is_mixture = mixing_weights is not None

    # When called without explicit layouts, infer from tensor shapes.
    # The sample-dim detection replicates the old ndim heuristic:
    # r has a sample dim when its rank exceeds the expected MAP rank.
    if param_layouts is None:
        from .core.axis_layout import infer_layout

        _expected_r_rank = 2 if is_mixture else 1
        _has_sd = r.ndim > _expected_r_rank
        _n_comp = r.shape[-2] if is_mixture and r.ndim >= 2 else None
        _params: Dict[str, jnp.ndarray] = {"r": r, "p": p}
        if mixing_weights is not None:
            _params["mixing_weights"] = mixing_weights
        if bnb_concentration is not None:
            _params["bnb_concentration"] = bnb_concentration
        param_layouts = {
            k: infer_layout(k, v, n_genes=int(r.shape[-1]),
                            n_cells=n_cells, n_components=_n_comp,
                            has_sample_dim=_has_sd)
            for k, v in _params.items()
        }

    # Detect whether r carries a leading posterior-sample dimension
    # purely from AxisLayout metadata — no ndim heuristics.
    has_sample_dim = _has_sample_dim(param_layouts)

    # Pre-compute layout-derived boolean flags at Python (trace) time.
    # These are static and safe to close over inside vmap.
    # Use MAP-level layouts (without sample dim) since vmap / the inner
    # function operates on individual draws.
    _base = {k: v.without_sample_dim() for k, v in param_layouts.items()}
    _p_has_comp = (
        _base["p"].component_axis is not None if "p" in _base else False
    )
    _p_has_genes = (
        _base["p"].gene_axis is not None if "p" in _base else False
    )
    _bnb_comp = (
        _base["bnb_concentration"].component_axis is not None
        if "bnb_concentration" in _base
        else False
    )

    if has_sample_dim:
        # Infer n_samples from the leading dimension of r
        actual_n_samples = r.shape[0]
        # Generate one PRNG key per posterior sample
        keys = random.split(rng_key, actual_n_samples)

        # Flags evaluated at Python (trace) time, safe inside vmap.
        _is_mixture = is_mixture
        _has_bnb = bnb_concentration is not None

        def _sample_one(key_i, r_i, p_i, mw_i, bnb_i):
            return _sample_biological_nb_single(
                r=r_i,
                p=p_i,
                n_cells=n_cells,
                rng_key=key_i,
                mixing_weights=mw_i if _is_mixture else None,
                cell_batch_size=cell_batch_size,
                bnb_concentration=bnb_i if _has_bnb else None,
                p_has_components=_p_has_comp,
                p_has_genes=_p_has_genes,
                bnb_has_components=_bnb_comp,
            )

        # vmap requires concrete arrays for every argument, so we
        # substitute dummy arrays for optional parameters that are None.
        mw_arr = mixing_weights if is_mixture else jnp.zeros(actual_n_samples)
        bnb_arr = (
            bnb_concentration
            if bnb_concentration is not None
            else jnp.zeros(actual_n_samples)
        )

        return vmap(_sample_one)(keys, r, p, mw_arr, bnb_arr)
    else:
        # MAP path: no leading sample dimension, so we loop n_samples times
        keys = random.split(rng_key, n_samples)
        all_samples = []
        for i in range(n_samples):
            sample_i = _sample_biological_nb_single(
                r=r,
                p=p,
                n_cells=n_cells,
                rng_key=keys[i],
                mixing_weights=mixing_weights,
                cell_batch_size=cell_batch_size,
                bnb_concentration=bnb_concentration,
                p_has_components=_p_has_comp,
                p_has_genes=_p_has_genes,
                bnb_has_components=_bnb_comp,
            )
            all_samples.append(sample_i)
        # Stack along a new leading sample axis → (n_samples, n_cells, n_genes)
        return jnp.stack(all_samples, axis=0)


# ------------------------------------------------------------------------------


def _sample_biological_nb_single(
    r: jnp.ndarray,
    p: jnp.ndarray,
    n_cells: int,
    rng_key: random.PRNGKey,
    mixing_weights: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
    bnb_concentration: Optional[jnp.ndarray] = None,
    *,
    p_has_components: bool = False,
    p_has_genes: bool = False,
    bnb_has_components: bool = False,
) -> jnp.ndarray:
    """Sample one realisation of biological NB counts for all cells.

    This is the inner workhorse called once per posterior sample (or once
    per MAP draw).  It handles both standard and mixture models and
    supports optional cell batching to bound memory usage.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion parameter.

        * Standard: shape ``(n_genes,)``.
        * Mixture: shape ``(n_components, n_genes)``.
    p : jnp.ndarray
        Success probability (scalar or ``(n_components,)``).
    n_cells : int
        Number of cells.
    rng_key : random.PRNGKey
        PRNG key.
    mixing_weights : jnp.ndarray or None
        Component weights ``(n_components,)`` for mixture models.
    cell_batch_size : int or None
        Optional cell-level batching.
    bnb_concentration : jnp.ndarray or None
        BNB concentration, ``(n_genes,)`` or ``(n_components, n_genes)``.
    p_has_components : bool
        Whether ``p`` has a component axis (from AxisLayout).
    p_has_genes : bool
        Whether ``p`` has a gene axis (from AxisLayout).
    bnb_has_components : bool
        Whether ``bnb_concentration`` has a component axis (from AxisLayout).

    Returns
    -------
    jnp.ndarray
        Counts array of shape ``(n_cells, n_genes)``.
    """
    is_mixture = mixing_weights is not None

    if cell_batch_size is None:
        cell_batch_size = n_cells

    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size
    batch_results = []

    for batch_idx in range(n_batches):
        start = batch_idx * cell_batch_size
        end = min(start + cell_batch_size, n_cells)
        batch_n = end - start

        rng_key, batch_key = random.split(rng_key)

        if is_mixture:
            comp_key, sample_key = random.split(batch_key)

            components = dist.Categorical(probs=mixing_weights).sample(
                comp_key, (batch_n,)
            )

            # Gather per-cell r values: (batch_n, n_genes)
            r_batch = r[components]

            if p_has_components:
                p_batch = p[components]
                # (K,) gathered → (batch_n,); expand for broadcasting
                if not p_has_genes:
                    p_batch = p_batch[:, None]
            else:
                p_batch = p

            bnb_batch = bnb_concentration
            if bnb_concentration is not None and bnb_has_components:
                bnb_batch = bnb_concentration[components]

            nb = build_count_dist(r_batch, p_batch, bnb_batch)
            batch_counts = nb.sample(sample_key)
        else:
            nb = build_count_dist(r, p, bnb_concentration)
            batch_counts = nb.sample(batch_key, (batch_n,))

        batch_results.append(batch_counts)

    return jnp.concatenate(batch_results, axis=0)


# ------------------------------------------------------------------------------
# Full-model posterior PPC sampling (NB / ZINB / VCP / mixtures)
# ------------------------------------------------------------------------------


def sample_posterior_ppc(
    r: jnp.ndarray,
    p: jnp.ndarray,
    n_cells: int,
    rng_key: random.PRNGKey,
    n_samples: int = 1,
    gate: Optional[jnp.ndarray] = None,
    p_capture: Optional[jnp.ndarray] = None,
    mixing_weights: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
    bnb_concentration: Optional[jnp.ndarray] = None,
    param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
) -> jnp.ndarray:
    """Sample from the full generative model using posterior parameters.

    Generates posterior predictive count samples that include **all** model
    components (NB base, zero-inflation gate, capture probability, mixture
    assignments).  Unlike :func:`sample_biological_nb`, this produces
    replicate data comparable to the *observed* counts and is appropriate
    for PPC-based goodness-of-fit evaluation.

    The function supports both MAP point estimates and full posterior
    parameter arrays.  When ``r`` has a leading sample dimension the
    function uses ``jax.vmap`` to vectorise over posterior draws.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion parameter.

        * Standard model, MAP: shape ``(n_genes,)``.
        * Standard model, posterior: shape ``(n_samples, n_genes)``.
        * Mixture model, MAP: shape ``(n_components, n_genes)``.
        * Mixture model, posterior: shape ``(n_samples, n_components,
          n_genes)``.
    p : jnp.ndarray
        Success probability of the Negative Binomial.

        * MAP: scalar or ``(n_components,)`` for component-specific p.
        * Posterior: ``(n_samples,)`` or ``(n_samples, n_components)``.
    n_cells : int
        Number of cells to generate counts for.
    rng_key : random.PRNGKey
        JAX PRNG key for reproducible sampling.
    n_samples : int, optional
        Number of draws when ``r`` has no leading sample dimension (MAP
        path).  Ignored when the sample dimension is inferred from ``r``.
        Default: 1.
    gate : jnp.ndarray or None, optional
        Zero-inflation gate probability.

        * MAP standard: ``(n_genes,)``.
        * Posterior standard: ``(n_samples, n_genes)``.
        * MAP mixture: ``(n_components, n_genes)``.
        * Posterior mixture: ``(n_samples, n_components, n_genes)``.

        ``None`` for non-ZINB models.
    p_capture : jnp.ndarray or None, optional
        Per-cell capture probability.

        * MAP: ``(n_cells,)``.
        * Posterior: ``(n_samples, n_cells)``.

        ``None`` for non-VCP models.
    mixing_weights : jnp.ndarray or None, optional
        Component mixing weights for mixture models.

        * MAP: ``(n_components,)``.
        * Posterior: ``(n_samples, n_components)``.

        ``None`` for non-mixture models.
    cell_batch_size : int or None, optional
        If set, cells are processed in batches of this size to limit peak
        memory.  Particularly useful for VCP models.  ``None`` processes
        all cells at once.

    Returns
    -------
    jnp.ndarray
        Sampled counts with shape ``(n_samples, n_cells, n_genes)``.

    See Also
    --------
    sample_biological_nb : Biological-only (denoised) PPC sampling.

    Examples
    --------
    >>> # Full posterior PPC for a ZINB-VCP model
    >>> samples = sample_posterior_ppc(
    ...     r=posterior["r"],          # (S, n_genes)
    ...     p=posterior["p"],          # (S,)
    ...     n_cells=5000,
    ...     rng_key=jax.random.PRNGKey(0),
    ...     gate=posterior["gate"],    # (S, n_genes)
    ...     p_capture=posterior["p_capture"],  # (S, n_cells)
    ... )
    >>> samples.shape
    (S, 5000, n_genes)
    """
    is_mixture = mixing_weights is not None

    # When called without explicit layouts, infer from tensor shapes.
    # The sample-dim detection replicates the old ndim heuristic:
    # r has a sample dim when its rank exceeds the expected MAP rank.
    if param_layouts is None:
        from .core.axis_layout import infer_layout

        _expected_r_rank = 2 if is_mixture else 1
        _has_sd = r.ndim > _expected_r_rank
        _n_comp = r.shape[-2] if is_mixture and r.ndim >= 2 else None
        _params: Dict[str, jnp.ndarray] = {"r": r, "p": p}
        if gate is not None:
            _params["gate"] = gate
        if p_capture is not None:
            _params["p_capture"] = p_capture
        if mixing_weights is not None:
            _params["mixing_weights"] = mixing_weights
        if bnb_concentration is not None:
            _params["bnb_concentration"] = bnb_concentration
        param_layouts = {
            k: infer_layout(k, v, n_genes=int(r.shape[-1]),
                            n_cells=n_cells, n_components=_n_comp,
                            has_sample_dim=_has_sd)
            for k, v in _params.items()
        }

    # Detect whether r carries a leading posterior-sample dimension
    # purely from AxisLayout metadata — no ndim heuristics.
    has_sample_dim = _has_sample_dim(param_layouts)

    # Pre-compute layout-derived boolean flags at Python (trace) time.
    # These are static and safe to close over inside vmap.
    # Use MAP-level layouts (without sample dim) since the inner function
    # operates on individual draws.
    _base = {k: v.without_sample_dim() for k, v in param_layouts.items()}
    _p_has_comp = (
        _base["p"].component_axis is not None if "p" in _base else False
    )
    _p_has_genes = (
        _base["p"].gene_axis is not None if "p" in _base else False
    )
    _gate_comp = (
        _base["gate"].component_axis is not None
        if "gate" in _base
        else False
    )
    _bnb_comp = (
        _base["bnb_concentration"].component_axis is not None
        if "bnb_concentration" in _base
        else False
    )

    if has_sample_dim:
        actual_n_samples = r.shape[0]
        keys = random.split(rng_key, actual_n_samples)

        # Build per-sample slices, using dummy arrays for None optionals
        # so vmap sees concrete array inputs.
        gate_arr = gate if gate is not None else jnp.zeros(actual_n_samples)
        p_cap_arr = (
            p_capture if p_capture is not None else jnp.zeros(actual_n_samples)
        )
        mw_arr = (
            mixing_weights
            if mixing_weights is not None
            else jnp.zeros(actual_n_samples)
        )
        bnb_arr = (
            bnb_concentration
            if bnb_concentration is not None
            else jnp.zeros(actual_n_samples)
        )

        # Flags must be static for the vmap-ed function
        _has_gate = gate is not None
        _has_p_capture = p_capture is not None
        _is_mixture = is_mixture
        _has_bnb = bnb_concentration is not None

        def _sample_one(key_i, r_i, p_i, gate_i, p_cap_i, mw_i, bnb_i):
            return _sample_posterior_ppc_single(
                r=r_i,
                p=p_i,
                n_cells=n_cells,
                rng_key=key_i,
                gate=gate_i if _has_gate else None,
                p_capture=p_cap_i if _has_p_capture else None,
                mixing_weights=mw_i if _is_mixture else None,
                cell_batch_size=cell_batch_size,
                bnb_concentration=bnb_i if _has_bnb else None,
                p_has_components=_p_has_comp,
                p_has_genes=_p_has_genes,
                gate_has_components=_gate_comp,
                bnb_has_components=_bnb_comp,
            )

        return vmap(_sample_one)(
            keys, r, p, gate_arr, p_cap_arr, mw_arr, bnb_arr
        )
    else:
        # MAP path: loop n_samples times
        keys = random.split(rng_key, n_samples)
        all_samples = []
        for i in range(n_samples):
            sample_i = _sample_posterior_ppc_single(
                r=r,
                p=p,
                n_cells=n_cells,
                rng_key=keys[i],
                gate=gate,
                p_capture=p_capture,
                mixing_weights=mixing_weights,
                cell_batch_size=cell_batch_size,
                bnb_concentration=bnb_concentration,
                p_has_components=_p_has_comp,
                p_has_genes=_p_has_genes,
                gate_has_components=_gate_comp,
                bnb_has_components=_bnb_comp,
            )
            all_samples.append(sample_i)
        return jnp.stack(all_samples, axis=0)


def _sample_posterior_ppc_single(
    r: jnp.ndarray,
    p: jnp.ndarray,
    n_cells: int,
    rng_key: random.PRNGKey,
    gate: Optional[jnp.ndarray] = None,
    p_capture: Optional[jnp.ndarray] = None,
    mixing_weights: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
    bnb_concentration: Optional[jnp.ndarray] = None,
    *,
    p_has_components: bool = False,
    p_has_genes: bool = False,
    gate_has_components: bool = False,
    bnb_has_components: bool = False,
) -> jnp.ndarray:
    """Sample one PPC realisation from the full generative model.

    Inner workhorse called once per posterior draw (or once per MAP draw).
    Handles standard, ZINB, VCP, and mixture models with optional cell
    batching.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion.  ``(n_genes,)`` for standard, ``(K, n_genes)``
        for mixture.
    p : jnp.ndarray
        Success probability.  Scalar or ``(K,)`` for mixture.
    n_cells : int
        Number of cells.
    rng_key : random.PRNGKey
        PRNG key.
    gate : jnp.ndarray or None
        Zero-inflation gate.  ``(n_genes,)`` or ``(K, n_genes)``.
    p_capture : jnp.ndarray or None
        Per-cell capture probability ``(n_cells,)``.
    mixing_weights : jnp.ndarray or None
        Component weights ``(K,)`` for mixture models.
    cell_batch_size : int or None
        Optional cell-level batching.
    bnb_concentration : jnp.ndarray or None
        BNB concentration, ``(n_genes,)`` or ``(K, n_genes)``.
    p_has_components : bool
        Whether ``p`` has a component axis (from AxisLayout).
    p_has_genes : bool
        Whether ``p`` has a gene axis (from AxisLayout).
    gate_has_components : bool
        Whether ``gate`` has a component axis (from AxisLayout).
    bnb_has_components : bool
        Whether ``bnb_concentration`` has a component axis (from AxisLayout).

    Returns
    -------
    jnp.ndarray
        Counts array of shape ``(n_cells, n_genes)``.
    """
    is_mixture = mixing_weights is not None
    has_vcp = p_capture is not None
    has_gate = gate is not None

    if cell_batch_size is None:
        cell_batch_size = n_cells

    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size
    batch_results = []

    for batch_idx in range(n_batches):
        start = batch_idx * cell_batch_size
        end = min(start + cell_batch_size, n_cells)
        batch_n = end - start

        rng_key, batch_key = random.split(rng_key)

        if is_mixture:
            comp_key, sample_key = random.split(batch_key)

            components = dist.Categorical(probs=mixing_weights).sample(
                comp_key, (batch_n,)
            )

            r_batch = r[components]

            if p_has_components:
                p_batch = p[components]
                if not p_has_genes:
                    p_batch = p_batch[:, None]
            else:
                p_batch = p

            if has_gate:
                gate_batch = gate[components] if gate_has_components else gate
            else:
                gate_batch = None

            if has_vcp:
                p_cap = p_capture[start:end]
                p_cap_exp = p_cap[:, None]
                p_effective = (
                    p_batch * p_cap_exp / (1 - p_batch * (1 - p_cap_exp))
                )
            else:
                p_effective = p_batch

            bnb_batch = bnb_concentration
            if bnb_concentration is not None and bnb_has_components:
                bnb_batch = bnb_concentration[components]

            nb = build_count_dist(r_batch, p_effective, bnb_batch)

            if gate_batch is not None:
                sample_dist = dist.ZeroInflatedDistribution(nb, gate=gate_batch)
            else:
                sample_dist = nb

            batch_counts = sample_dist.sample(sample_key)

        else:
            # -------------------------------------------------------
            # Standard (non-mixture) model
            # -------------------------------------------------------
            # VCP: compute effective p per cell in this batch
            if has_vcp:
                p_cap = p_capture[start:end]  # (batch_n,)
                p_cap_reshaped = p_cap[:, None]  # (batch_n, 1)
                p_effective = (
                    p * p_cap_reshaped / (1 - p * (1 - p_cap_reshaped))
                )
            else:
                p_effective = p

            nb = build_count_dist(r, p_effective, bnb_concentration)

            if has_gate:
                sample_dist = dist.ZeroInflatedDistribution(nb, gate=gate)
            else:
                sample_dist = nb

            # Shape depends on whether VCP gives the distribution a
            # batch dimension.
            if has_vcp:
                batch_counts = sample_dist.sample(batch_key)
            else:
                batch_counts = sample_dist.sample(batch_key, (batch_n,))

        batch_results.append(batch_counts)

    return jnp.concatenate(batch_results, axis=0)


# ------------------------------------------------------------------------------
# Bayesian denoising of observed counts
# ------------------------------------------------------------------------------

# Allowed values for individual method elements
_VALID_DENOISE_METHODS = {"mean", "mode", "sample"}


def _validate_denoise_method(method: Union[str, Tuple[str, str]]) -> None:
    """Validate the ``method`` argument for denoising functions.

    Accepts a single string or a tuple of two strings, each of which
    must be one of ``'mean'``, ``'mode'``, or ``'sample'``.

    Parameters
    ----------
    method : str or tuple of (str, str)
        The method specification to validate.

    Raises
    ------
    ValueError
        If the method is not a valid string or 2-tuple of valid strings.
    """
    if isinstance(method, str):
        if method not in _VALID_DENOISE_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_DENOISE_METHODS} or a "
                f"2-tuple thereof, got '{method}'"
            )
    elif isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError(
                f"method tuple must have exactly 2 elements, "
                f"got {len(method)}"
            )
        for i, m in enumerate(method):
            if m not in _VALID_DENOISE_METHODS:
                raise ValueError(
                    f"method[{i}] must be one of {_VALID_DENOISE_METHODS}, "
                    f"got '{m}'"
                )
    else:
        raise ValueError(
            f"method must be a string or a 2-tuple of strings, "
            f"got {type(method).__name__}"
        )


def _method_needs_rng(method: Union[str, Tuple[str, str]]) -> bool:
    """Return True if any element of ``method`` requires an RNG key."""
    if isinstance(method, str):
        return method == "sample"
    return "sample" in method


def denoise_counts(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray] = None,
    gate: Optional[jnp.ndarray] = None,
    method: Union[str, Tuple[str, str]] = "mean",
    rng_key: Optional[random.PRNGKey] = None,
    return_variance: bool = False,
    mixing_weights: Optional[jnp.ndarray] = None,
    component_assignment: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
    bnb_concentration: Optional[jnp.ndarray] = None,
    param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Denoise observed counts using the Bayesian posterior of true transcripts.

    Given observed UMI counts and posterior parameter estimates, computes
    the posterior distribution of the true (pre-capture, pre-dropout)
    transcript counts for each cell and gene.  The derivation exploits
    Poisson-Gamma conjugacy and the Poisson thinning property; see
    ``paper/_denoising.qmd`` for the full mathematics.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed UMI count matrix of shape ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion (total_count) parameter in canonical form.

        * Standard model, single param set: ``(n_genes,)``.
        * Standard model, multi-sample: ``(n_samples, n_genes)``.
        * Mixture model, single: ``(n_components, n_genes)``.
        * Mixture model, multi-sample: ``(n_samples, n_components, n_genes)``.
    p : jnp.ndarray
        Success probability (numpyro probs convention, *not* the paper's p).

        * Single param set: scalar or ``(n_components,)``.
        * Multi-sample: ``(n_samples,)`` or ``(n_samples, n_components)``.
    p_capture : jnp.ndarray or None, optional
        Per-cell capture probability :math:`\\nu_c`.  Shape ``(n_cells,)``
        for a single param set or ``(n_samples, n_cells)`` for multi-sample.
        ``None`` for models without variable capture probability (nbdm, zinb),
        which is equivalent to :math:`\\nu_c = 1` (perfect capture).
    gate : jnp.ndarray or None, optional
        Zero-inflation gate probability.  Shape ``(n_genes,)`` or
        ``(n_components, n_genes)`` for a single param set; with a leading
        ``n_samples`` dimension for multi-sample.  ``None`` for models
        without zero-inflation.
    method : str or tuple of (str, str), optional
        Summary statistic to return.  Accepts either a single string
        applied uniformly to all positions, or a tuple
        ``(general_method, zi_zero_method)`` for independent control:

        * ``general_method``: used for non-zero positions and for all
          positions in non-ZINB models (no gate).
        * ``zi_zero_method``: used exclusively for zero positions in
          ZINB models (the gate/NB mixture posterior).

        Valid values for each element:

        * ``'mean'``: closed-form posterior mean (shrinkage estimator).
        * ``'mode'``: posterior mode (MAP denoised count).
        * ``'sample'``: one stochastic draw from the denoised posterior.

        A single string ``s`` is equivalent to ``(s, s)``.
        Default: ``'mean'``.
    rng_key : random.PRNGKey or None, optional
        JAX PRNG key.  Required when any element of ``method`` is
        ``'sample'``.
    return_variance : bool, optional
        If ``True``, return a dictionary with keys ``'denoised_counts'``
        and ``'variance'`` instead of a plain array.  Default: ``False``.
    mixing_weights : jnp.ndarray or None, optional
        Component mixing weights for mixture models.  Shape
        ``(n_components,)`` or ``(n_samples, n_components)``.
    component_assignment : jnp.ndarray or None, optional
        Pre-computed per-cell component assignments of shape
        ``(n_cells,)`` (integer indices).  When provided, each cell uses
        its assigned component's parameters instead of marginalising
        over components.  Ignored for non-mixture models.
    cell_batch_size : int or None, optional
        Process cells in batches of this size to limit peak memory.
        ``None`` processes all cells at once.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        If ``return_variance`` is ``False`` (default): denoised count matrix
        with shape ``(n_cells, n_genes)`` (single param set) or
        ``(n_samples, n_cells, n_genes)`` (multi-sample).

        If ``return_variance`` is ``True``: dictionary with keys
        ``'denoised_counts'`` and ``'variance'``, each with the shape above.

    Notes
    -----
    The denoising posterior for uncaptured transcripts is:

    .. math::

        d_g \\mid u_g \\sim \\text{NB}\\!\\left(r_g + u_g,\\;
        \\nu_c + (1-\\nu_c)(1-p)\\right)

    where :math:`p` is the paper's success probability
    (= ``1 - canonical_p``).  The posterior mean simplifies to:

    .. math::

        \\mathbb{E}[m_g \\mid u_g] =
        \\frac{u_g + r_g \\, p_{\\text{can}} (1-\\nu_c)}
        {1 - p_{\\text{can}} (1-\\nu_c)}

    For ZINB models, zero observations use a mixture posterior
    weighted by the probability that the zero came from the gate.

    See Also
    --------
    sample_biological_nb : Biological PPC (samples from NB prior, not
        conditioned on observed counts).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from scribe.sampling import denoise_counts
    >>> counts = jnp.array([[5, 0, 3], [0, 2, 0]])
    >>> r = jnp.array([2.0, 1.5, 3.0])
    >>> p = jnp.float32(0.6)
    >>> nu = jnp.array([0.5, 0.7])
    >>> denoised = denoise_counts(counts, r, p, p_capture=nu)
    >>> denoised.shape
    (2, 3)

    Tuple method for independent control of ZINB zeros:

    >>> denoised = denoise_counts(
    ...     counts, r, p, p_capture=nu,
    ...     gate=jnp.array([0.2, 0.3, 0.1]),
    ...     method=("mean", "sample"),
    ... )
    """
    _validate_denoise_method(method)
    if _method_needs_rng(method) and rng_key is None:
        rng_key = random.PRNGKey(42)

    is_mixture = mixing_weights is not None

    # When called without explicit layouts (e.g. directly from user code
    # with raw tensors), infer layouts from the parameter shapes.
    # The sample-dim detection replicates the old ndim heuristic as a
    # one-time compatibility shim: r has a sample dim when its rank
    # exceeds the expected MAP rank.
    if param_layouts is None:
        _expected_r_rank = 2 if is_mixture else 1
        _has_sd = r.ndim > _expected_r_rank
        _params: Dict[str, jnp.ndarray] = {"r": r, "p": p}
        if p_capture is not None:
            _params["p_capture"] = p_capture
        if gate is not None:
            _params["gate"] = gate
        if mixing_weights is not None:
            _params["mixing_weights"] = mixing_weights
        if bnb_concentration is not None:
            _params["bnb_concentration"] = bnb_concentration
        from .core.axis_layout import infer_layout

        param_layouts = {
            k: infer_layout(
                k, v,
                n_genes=counts.shape[-1],
                n_cells=counts.shape[0],
                n_components=(
                    r.shape[-2] if is_mixture and r.ndim >= 2 else None
                ),
                has_sample_dim=_has_sd,
            )
            for k, v in _params.items()
        }

    # Detect whether r carries a leading posterior-sample dimension
    # purely from AxisLayout metadata — no ndim heuristics.
    has_sample_dim = _has_sample_dim(param_layouts)

    # Derive MAP-level layouts (no sample dim) for the denoising functions,
    # since the sample dimension is handled by the outer loop.
    _base_layouts = {
        k: v.without_sample_dim() for k, v in param_layouts.items()
    }

    if not has_sample_dim:
        return _denoise_single(
            counts=counts,
            r=r,
            p=p,
            p_capture=p_capture,
            gate=gate,
            method=method,
            rng_key=rng_key,
            return_variance=return_variance,
            mixing_weights=mixing_weights,
            component_assignment=component_assignment,
            cell_batch_size=cell_batch_size,
            bnb_concentration=bnb_concentration,
            param_layouts=_base_layouts,
        )

    # Multi-sample path: iterate over posterior draws
    n_samples = r.shape[0]
    keys = (
        random.split(rng_key, n_samples)
        if _method_needs_rng(method)
        else [None] * n_samples
    )

    result_list: List[jnp.ndarray] = []
    var_list: List[jnp.ndarray] = []

    for s in range(n_samples):
        # Extract parameters for this single posterior draw, using
        # layout metadata to decide which arrays carry a sample axis.
        draw = _slice_posterior_draw(
            s,
            r=r,
            p=p,
            p_capture=p_capture,
            gate=gate,
            mixing_weights=mixing_weights,
            param_layouts=param_layouts,
            bnb_concentration=bnb_concentration,
        )

        # After slicing, the draw parameters are MAP-level (no sample dim);
        # pass _base_layouts for layout-driven flag computation.
        out = _denoise_single(
            counts=counts,
            r=draw["r"],
            p=draw["p"],
            p_capture=draw["p_capture"],
            gate=draw["gate"],
            method=method,
            rng_key=keys[s],
            return_variance=return_variance,
            mixing_weights=draw["mixing_weights"],
            component_assignment=component_assignment,
            cell_batch_size=cell_batch_size,
            bnb_concentration=draw["bnb_concentration"],
            param_layouts=_base_layouts,
        )

        if return_variance:
            result_list.append(out["denoised_counts"])
            var_list.append(out["variance"])
        else:
            result_list.append(out)

    stacked = jnp.stack(result_list, axis=0)
    if return_variance:
        return {
            "denoised_counts": stacked,
            "variance": jnp.stack(var_list, axis=0),
        }
    return stacked


# ------------------------------------------------------------------------------


def _denoise_single(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    return_variance: bool,
    mixing_weights: Optional[jnp.ndarray],
    component_assignment: Optional[jnp.ndarray],
    cell_batch_size: Optional[int],
    bnb_concentration: Optional[jnp.ndarray] = None,
    *,
    param_layouts: Dict[str, "AxisLayout"],
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Dispatch denoising for a single set of parameters.

    Handles both standard and mixture models.  For mixture models with
    ``component_assignment`` provided, gathers per-cell parameters and
    delegates to the standard path.  Otherwise marginalises over components.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts, shape ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion.  ``(n_genes,)`` for standard, ``(n_components, n_genes)``
        for mixture.
    p : jnp.ndarray
        Success probability.  Scalar or ``(n_components,)``.
    p_capture : jnp.ndarray or None
        Capture probability per cell, ``(n_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate probability.  ``(n_genes,)`` or ``(n_components, n_genes)`` or
        ``None``.
    method : str or tuple of (str, str)
        Denoising method.
    rng_key : random.PRNGKey or None
        PRNG key.
    return_variance : bool
        Whether to include variance in the output.
    mixing_weights : jnp.ndarray or None
        Component weights ``(n_components,)`` for mixture models.
    component_assignment : jnp.ndarray or None
        Per-cell component indices ``(n_cells,)``.
    cell_batch_size : int or None
        Batch cells to limit memory.
    bnb_concentration : jnp.ndarray or None
        Optional BNB concentration.
    param_layouts : dict of str to AxisLayout
        MAP-level semantic layouts used to derive boolean flags that
        replace ``ndim``/``shape`` heuristics.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts (and optionally variance).
    """
    is_mixture = mixing_weights is not None

    # Layout-derived flags — no shape heuristics.
    _p_has_comp = (
        param_layouts["p"].component_axis is not None
        if "p" in param_layouts else False
    )
    _p_has_genes = (
        param_layouts["p"].gene_axis is not None
        if "p" in param_layouts else False
    )
    _gate_has_comp = (
        param_layouts["gate"].component_axis is not None
        if "gate" in param_layouts else False
    )

    if is_mixture and component_assignment is not None:
        r_cell = r[component_assignment]

        p_cell = p[component_assignment] if _p_has_comp else p
        # (K,) gathered → (n_cells,); expand to (n_cells, 1) so
        # downstream broadcasts correctly with (n_cells, n_genes).
        if _p_has_comp and not _p_has_genes:
            p_cell = p_cell[:, None]

        g_cell = (
            gate[component_assignment]
            if gate is not None and _gate_has_comp
            else gate
        )

        return _denoise_standard(
            counts,
            r_cell,
            p_cell,
            p_capture,
            g_cell,
            method,
            rng_key,
            return_variance,
            cell_batch_size,
            bnb_concentration=bnb_concentration,
            # r and gate were gathered to per-cell via component_assignment;
            # p was gathered + expanded to (n_cells, 1) when per-component.
            r_is_per_cell=True,
            p_is_per_cell=_p_has_comp,
            gate_is_per_cell=_gate_has_comp,
            bnb_is_per_cell=False,
        )

    if is_mixture and component_assignment is None:
        return _denoise_mixture_marginal(
            counts,
            r,
            p,
            p_capture,
            gate,
            method,
            rng_key,
            return_variance,
            mixing_weights,
            cell_batch_size,
            bnb_concentration=bnb_concentration,
            param_layouts=param_layouts,
        )

    # Standard (non-mixture) model — no per-cell gathering needed.
    return _denoise_standard(
        counts,
        r,
        p,
        p_capture,
        gate,
        method,
        rng_key,
        return_variance,
        cell_batch_size,
        bnb_concentration=bnb_concentration,
    )


# ------------------------------------------------------------------------------


def _denoise_standard(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    return_variance: bool,
    cell_batch_size: Optional[int],
    bnb_concentration: Optional[jnp.ndarray] = None,
    *,
    r_is_per_cell: bool = False,
    p_is_per_cell: bool = False,
    gate_is_per_cell: bool = False,
    bnb_is_per_cell: bool = False,
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Denoise counts for a standard (non-mixture) model, single param set.

    Implements the core denoising formulas from ``paper/_denoising.qmd``.

    The key quantity is ``probs_post`` = canonical_p * (1 - nu_c), the
    numpyro-convention success probability of the posterior NB for the
    uncaptured transcripts d_g.  When ``p_capture`` is ``None`` (no VCP),
    ``probs_post = 0`` and denoising reduces to identity (plus any gate
    correction at zeros).

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion ``(n_genes,)`` or ``(n_cells, n_genes)`` when gathered
        from mixture component assignments.
    p : jnp.ndarray
        Success probability (scalar or broadcastable).
    p_capture : jnp.ndarray or None
        Capture probability ``(n_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate probability ``(n_genes,)`` or ``(n_cells, n_genes)`` or
        ``None``.
    method : str or tuple of (str, str)
        Denoising method.  A single string or ``(general, zi_zeros)``
        tuple; see :func:`denoise_counts`.
    rng_key : random.PRNGKey or None
        PRNG key (needed when any element of ``method`` is ``'sample'``).
    return_variance : bool
        Whether to return variance alongside denoised counts.
    cell_batch_size : int or None
        Optional cell batching.
    r_is_per_cell : bool
        ``True`` when ``r`` has been gathered to ``(n_cells, n_genes)``
        via component assignment and must be sliced per batch.
    p_is_per_cell : bool
        ``True`` when ``p`` has been gathered/expanded to
        ``(n_cells, ...)`` and must be sliced per batch.
    gate_is_per_cell : bool
        ``True`` when ``gate`` has been gathered to
        ``(n_cells, n_genes)`` and must be sliced per batch.
    bnb_is_per_cell : bool
        ``True`` when ``bnb_concentration`` has been gathered to
        ``(n_cells, n_genes)`` and must be sliced per batch.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts (and optionally variance).
    """
    n_cells, n_genes = counts.shape

    if cell_batch_size is None:
        cell_batch_size = n_cells

    needs_rng = _method_needs_rng(method)

    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size
    denoised_parts: List[jnp.ndarray] = []
    variance_parts: List[jnp.ndarray] = []

    for b in range(n_batches):
        start = b * cell_batch_size
        end = min(start + cell_batch_size, n_cells)
        counts_b = counts[start:end]

        pc_b = p_capture[start:end] if p_capture is not None else None
        r_b = r[start:end] if r_is_per_cell else r
        gate_b = gate[start:end] if gate is not None and gate_is_per_cell else gate
        bnb_b = (
            bnb_concentration[start:end]
            if bnb_concentration is not None and bnb_is_per_cell
            else bnb_concentration
        )
        p_b = p[start:end] if p_is_per_cell else p

        if needs_rng:
            rng_key, batch_key = random.split(rng_key)
        else:
            batch_key = None

        d, v = _denoise_batch(
            counts_b,
            r_b,
            p_b,
            pc_b,
            gate_b,
            method,
            batch_key,
            bnb_concentration=bnb_b,
        )
        denoised_parts.append(d)
        if return_variance:
            variance_parts.append(v)

    denoised = jnp.concatenate(denoised_parts, axis=0)
    if return_variance:
        variance = jnp.concatenate(variance_parts, axis=0)
        return {"denoised_counts": denoised, "variance": variance}
    return denoised


# ------------------------------------------------------------------------------
# BNB denoising helpers
# ------------------------------------------------------------------------------

# Minimum epsilon to prevent log(0) / division-by-zero in BNB denoising.
_BNB_DENOISE_EPS = 1e-6


def _bnb_omega_to_alpha_kappa(
    r: jnp.ndarray,
    p: jnp.ndarray,
    omega: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert ``omega`` (excess dispersion fraction) to ``(alpha, kappa)``.

    Mirrors the reparameterization in ``build_bnb_dist`` but returns
    the intermediate parameters instead of a distribution object.

    Parameters
    ----------
    r : jnp.ndarray
        NB dispersion (>0).
    p : jnp.ndarray
        NB success probability in numpyro convention (>0, <1).
    omega : jnp.ndarray
        Per-gene excess dispersion fraction (>0).

    Returns
    -------
    alpha : jnp.ndarray
        First Beta shape parameter (mean-preserving).
    kappa : jnp.ndarray
        BNB concentration (second Beta shape parameter, >2).
    """
    omega = jnp.clip(omega, _BNB_DENOISE_EPS, None)
    kappa = 2.0 + (r + 1.0) / omega
    kappa = jnp.clip(kappa, 2.0 + _BNB_DENOISE_EPS, None)
    # Mean-preserving alpha: NB mean = r*p/(1-p) = BNB mean = r*alpha/(kappa-1)
    alpha = p * (kappa - 1.0) / (1.0 - p)
    return alpha, kappa


def _bnb_p_log_posterior_unnorm(
    p_grid: jnp.ndarray,
    r: jnp.ndarray,
    alpha: jnp.ndarray,
    kappa: jnp.ndarray,
    counts: jnp.ndarray,
    nu: jnp.ndarray,
) -> jnp.ndarray:
    r"""Log of the unnormalized posterior of the latent mixing variable.

    Uses the **numpyro convention** where ``p`` is the observation
    probability (NB mean = r*p/(1-p)).  The unnormalized posterior is:

    .. math::

        (u + \alpha - 1)\log p + (r + \kappa - 1)\log(1-p)
        - (r + u)\log[1 - p(1-\nu)]

    This follows from:
    - Likelihood: :math:`\propto (1-p)^r \, p^u / [1-p(1-\nu)]^{r+u}`
    - Prior: :math:`p^{\alpha-1}(1-p)^{\kappa-1}`

    All inputs are expected to broadcast: ``p_grid`` has a leading
    node/grid axis while all other arrays have cell/gene axes.

    Parameters
    ----------
    p_grid : jnp.ndarray
        Grid points in (0, 1), shape ``(N, 1, 1)`` or ``(N,)``.
    r, alpha, kappa : jnp.ndarray
        Gene-level parameters, shape ``(..., G)``.
    counts : jnp.ndarray
        Observed counts, shape ``(C, G)``.
    nu : jnp.ndarray
        Capture probability, shape ``(C, 1)`` or scalar.

    Returns
    -------
    jnp.ndarray
        Log unnormalized posterior, shape ``(N, C, G)``.
    """
    pg = jnp.clip(p_grid, _BNB_DENOISE_EPS, 1.0 - _BNB_DENOISE_EPS)
    log_p = jnp.log(pg)
    log_1mp = jnp.log(1.0 - pg)
    # Denominator: 1 - p*(1-nu) = effective posterior success prob p'
    denom = 1.0 - pg * (1.0 - nu)
    log_denom = jnp.log(jnp.clip(denom, _BNB_DENOISE_EPS, None))

    return (
        (counts + alpha - 1.0) * log_p
        + (r + kappa - 1.0) * log_1mp
        - (r + counts) * log_denom
    )


def _denoise_bnb_quadrature(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    bnb_concentration: jnp.ndarray,
    n_nodes: int = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""BNB MAP denoising via Gauss--Legendre quadrature.

    Computes the posterior mean and variance of the denoised count
    :math:`m_g = u_g + d_g` under the BNB model by numerically
    integrating over the latent Beta mixing variable's posterior.

    See ``paper/_beta_negative_binomial.qmd``, @sec-bnb-denoising for
    the full derivation.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed UMI counts, shape ``(C, G)``.
    r : jnp.ndarray
        NB dispersion, shape ``(G,)`` or ``(C, G)``.
    p : jnp.ndarray
        NB success probability (numpyro convention), scalar or ``(G,)``
        or ``(C, 1)``.
    p_capture : jnp.ndarray or None
        Capture probability, shape ``(C,)`` or ``None``.
        When ``None``, treated as :math:`\nu_c = 1` (identity denoising).
    bnb_concentration : jnp.ndarray
        Excess dispersion :math:`\omega_g`, shape ``(G,)`` or ``(C, G)``.
    n_nodes : int, optional
        Number of Gauss--Legendre quadrature nodes.  Default: 64.

    Returns
    -------
    denoised_mean : jnp.ndarray
        Posterior mean of the denoised count, shape ``(C, G)``.
    denoised_var : jnp.ndarray
        Posterior variance of the denoised count, shape ``(C, G)``.
    """
    alpha, kappa = _bnb_omega_to_alpha_kappa(r, p, bnb_concentration)

    # Capture probability: (C, 1) for broadcasting with (C, G)
    if p_capture is not None:
        nu = p_capture[:, None]
    else:
        nu = jnp.ones(())

    # Gauss-Legendre nodes/weights on [0, 1], shape (N,)
    nodes, weights = gauss_legendre_nodes_weights(n_nodes, 0.0, 1.0)

    # Reshape nodes to (N, 1, 1) for broadcasting with (C, G)
    pg = nodes[:, None, None]
    w_gl = weights[:, None, None]

    # Log of unnormalized posterior at each quadrature node
    log_post = _bnb_p_log_posterior_unnorm(pg, r, alpha, kappa, counts, nu)

    # Subtract maximum for numerical stability (log-sum-exp trick)
    log_post_max = jnp.max(log_post, axis=0, keepdims=True)
    post_unnorm = jnp.exp(log_post - log_post_max)  # (N, C, G)

    # f(p) = p*(1-nu) / [1-p*(1-nu)] — conditional mean factor
    # (numpyro convention: E[d|u,p] = (r+u)*f(p))
    pg_safe = jnp.clip(pg, _BNB_DENOISE_EPS, 1.0 - _BNB_DENOISE_EPS)
    pprime = 1.0 - pg_safe * (1.0 - nu)  # p' = 1 - p*(1-nu)
    f_p = pg_safe * (1.0 - nu) / jnp.clip(pprime, _BNB_DENOISE_EPS, None)

    # Numerator integral: sum_i w_i * post(p_i) * f(p_i)
    Z = jnp.sum(w_gl * post_unnorm, axis=0)  # (C, G) — normalizer
    Z = jnp.clip(Z, _BNB_DENOISE_EPS, None)

    E_f = jnp.sum(w_gl * post_unnorm * f_p, axis=0) / Z  # E[f(p)]

    # Posterior mean: u + (r+u) * E[f(p)]
    denoised_mean = counts + (r + counts) * E_f

    # Variance via law of total variance:
    # Var(d|u) = E_p[Var(d|u,p)] + Var_p[E(d|u,p)]

    # g(p) = p*(1-nu) / [1-p*(1-nu)]^2 — conditional variance factor
    # (numpyro convention: Var(d|u,p) = (r+u)*g(p))
    g_p = pg_safe * (1.0 - nu) / jnp.clip(pprime**2, _BNB_DENOISE_EPS, None)

    # E[g(p)] — average conditional variance factor
    E_g = jnp.sum(w_gl * post_unnorm * g_p, axis=0) / Z

    # E[f(p)^2] for the variance-of-means term
    E_f2 = jnp.sum(w_gl * post_unnorm * f_p**2, axis=0) / Z

    # Total variance: (r+u)*E[g] + (r+u)^2 * (E[f^2] - E[f]^2)
    rpu = r + counts
    denoised_var = rpu * E_g + rpu**2 * (E_f2 - E_f**2)

    return denoised_mean, denoised_var


def _sample_p_posterior_bnb(
    rng_key: random.PRNGKey,
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    bnb_concentration: jnp.ndarray,
    n_grid: int = 1000,
) -> jnp.ndarray:
    r"""Sample the latent Beta mixing variable from its posterior.

    Uses grid-based inverse CDF sampling: evaluate the unnormalized
    posterior on a fine uniform grid over (0, 1), normalize to a PMF,
    compute the CDF, and invert via ``searchsorted``.

    Parameters
    ----------
    rng_key : random.PRNGKey
        PRNG key for the uniform draw.
    counts : jnp.ndarray
        Observed counts, shape ``(C, G)``.
    r : jnp.ndarray
        NB dispersion, shape ``(G,)`` or ``(C, G)``.
    p : jnp.ndarray
        NB success probability (numpyro convention).
    p_capture : jnp.ndarray or None
        Capture probability, shape ``(C,)`` or ``None``.
    bnb_concentration : jnp.ndarray
        Excess dispersion :math:`\omega_g`, shape ``(G,)`` or ``(C, G)``.
    n_grid : int, optional
        Number of grid points for inverse CDF sampling.  Default: 1000.

    Returns
    -------
    p_samples : jnp.ndarray
        Sampled latent mixing variable values, shape ``(C, G)``.
    """
    alpha, kappa = _bnb_omega_to_alpha_kappa(r, p, bnb_concentration)

    if p_capture is not None:
        nu = p_capture[:, None]
    else:
        nu = jnp.ones(())

    # Fine uniform grid on (0, 1), shape (M,)
    grid = jnp.linspace(_BNB_DENOISE_EPS, 1.0 - _BNB_DENOISE_EPS, n_grid)
    pg = grid[:, None, None]  # (M, 1, 1)

    # Log unnormalized posterior on the grid
    log_post = _bnb_p_log_posterior_unnorm(pg, r, alpha, kappa, counts, nu)

    # Stabilize and exponentiate
    log_post_max = jnp.max(log_post, axis=0, keepdims=True)
    post_unnorm = jnp.exp(log_post - log_post_max)  # (M, C, G)

    # Normalize to PMF along grid axis
    pmf = post_unnorm / jnp.clip(
        jnp.sum(post_unnorm, axis=0, keepdims=True), _BNB_DENOISE_EPS, None
    )

    # CDF along grid axis
    cdf = jnp.cumsum(pmf, axis=0)  # (M, C, G)

    # Draw uniform samples and invert CDF
    u_samples = random.uniform(rng_key, shape=counts.shape)  # (C, G)

    # For each (c, g) pair, find the grid index where CDF >= u
    # searchsorted works along the first axis; we need to transpose
    # to work per-(c,g) pair.  Flatten cell/gene dims, searchsorted
    # on each, then reshape.
    C, G = counts.shape
    cdf_flat = cdf.reshape(n_grid, -1)  # (M, C*G)
    u_flat = u_samples.reshape(-1)  # (C*G,)

    # vmap searchsorted over the C*G dimension
    def _search_one(cdf_col, u_val):
        return jnp.searchsorted(cdf_col, u_val, side="right")

    indices = vmap(_search_one, in_axes=(1, 0), out_axes=0)(
        cdf_flat, u_flat
    )  # (C*G,)

    # Clamp indices to valid range and look up grid values
    indices = jnp.clip(indices, 0, n_grid - 1)
    p_samples = grid[indices].reshape(C, G)

    return p_samples


# ------------------------------------------------------------------------------


def _denoise_batch(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    bnb_concentration: Optional[jnp.ndarray] = None,
) -> tuple:
    """Denoise a single batch of cells (no further splitting).

    Returns ``(denoised, variance)`` where ``variance`` is always computed
    (the caller decides whether to keep it).

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts for this batch, ``(batch_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion, ``(n_genes,)`` or ``(batch_cells, n_genes)``.
    p : jnp.ndarray
        Success probability: scalar ``()``, gene-specific ``(n_genes,)``,
        or per-cell ``(batch_cells, 1)`` / ``(batch_cells, n_genes)``.
    p_capture : jnp.ndarray or None
        Capture probability ``(batch_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate probability, ``(n_genes,)`` or ``(batch_cells, n_genes)`` or
        ``None``.
    method : str or tuple of (str, str)
        Denoising method.  A single string applies uniformly; a tuple
        ``(general_method, zi_zero_method)`` allows the ZINB zero
        correction to use a different method from the rest.
    rng_key : random.PRNGKey or None
        PRNG key (needed when any element of ``method`` is ``'sample'``).

    Returns
    -------
    denoised : jnp.ndarray
        Denoised counts ``(batch_cells, n_genes)``.
    variance : jnp.ndarray
        Posterior variance ``(batch_cells, n_genes)``.
    """
    # Normalize method to (general_method, zi_zero_method)
    if isinstance(method, str):
        general_method, zi_zero_method = method, method
    else:
        general_method, zi_zero_method = method

    # Per-cell p arrives as (batch_cells, 1) from the gathering step in
    # _denoise_single / _denoise_mixture_marginal, gene-specific p as
    # (n_genes,), and scalar p as ().  All broadcast correctly with
    # (batch_cells, n_genes) tensors without further reshaping.
    p_eff = p

    # probs_post is the numpyro probs for the posterior NB of uncaptured
    # transcripts d_g.  probs_post = canonical_p * (1 - nu_c).
    # When no VCP (nu_c = 1): probs_post = 0 → d_g = 0 → identity.
    if p_capture is not None:
        nu = p_capture[:, None]  # (batch_cells, 1)
        probs_post = p_eff * (1.0 - nu)
    else:
        probs_post = jnp.zeros(())

    # Complement: 1 - probs_post = p'_paper (the posterior "success" prob
    # in the paper's convention).  Used as denominator in most formulas.
    one_minus_pp = 1.0 - probs_post

    # ------------------------------------------------------------------
    # Denoising: BNB branch (quadrature / augmented sampling) when
    # bnb_concentration is present and VCP is active, NB closed-form
    # otherwise.  When BNB is active, we precompute the quadrature
    # mean/var once so that both the main path and the ZINB zero-
    # correction block can reuse them.
    # ------------------------------------------------------------------
    _use_bnb_denoise = bnb_concentration is not None and p_capture is not None

    # Precompute BNB quadrature results (reused across main + ZINB paths)
    _bnb_mean = _bnb_var = None
    if _use_bnb_denoise:
        _bnb_mean, _bnb_var = _denoise_bnb_quadrature(
            counts, r, p_eff, p_capture, bnb_concentration
        )

    if _use_bnb_denoise and general_method in ("mean", "mode"):
        # BNB quadrature denoising for mean/mode.  The mode is
        # approximated as floor(quadrature_mean) since the BNB
        # denoising posterior has no closed-form mode.
        denoised_nb = _bnb_mean
        var_nb = _bnb_var
        if general_method == "mode":
            denoised_nb = jnp.floor(denoised_nb)

    elif _use_bnb_denoise and general_method == "sample":
        # Augmented BNB sampling: draw p from its 1D posterior,
        # then draw d from the conditional NB denoising posterior.
        key_p, key_nb, rng_key = random.split(rng_key, 3)
        p_sampled = _sample_p_posterior_bnb(
            key_p, counts, r, p_eff, p_capture, bnb_concentration
        )
        # Conditional posterior NB probs: p_sampled * (1 - nu)
        probs_cond = p_sampled * (1.0 - nu)
        alpha_cond = r + counts
        d_sample = dist.NegativeBinomialProbs(
            total_count=alpha_cond, probs=probs_cond
        ).sample(key_nb)
        denoised_nb = counts + d_sample
        # Use quadrature variance as the best estimate
        var_nb = _bnb_var

    else:
        # Standard NB closed-form denoising
        if general_method == "mean":
            denoised_nb = (counts + r * probs_post) / one_minus_pp
        elif general_method == "mode":
            alpha = r + counts
            d_mode = jnp.floor(
                jnp.maximum(alpha - 1.0, 0.0) * probs_post / one_minus_pp
            )
            denoised_nb = counts + d_mode
        else:
            # general_method == "sample"
            alpha = r + counts
            key_nb, rng_key = random.split(rng_key)
            d_sample = build_count_dist(
                alpha, probs_post, bnb_concentration
            ).sample(key_nb)
            denoised_nb = counts + d_sample

        # NB variance: alpha * probs_post / (1-probs_post)^2
        var_nb = (r + counts) * probs_post / one_minus_pp**2

    # ------------------------------------------------------------------
    # ZINB zero correction: when gate is present and u_g = 0, the
    # denoised posterior is a mixture of gate and NB pathways.
    # Uses zi_zero_method for the zero positions.
    # ------------------------------------------------------------------
    if gate is not None:
        is_zero = counts == 0

        # Gate weight w = P(gate fired | u=0)
        w = _compute_gate_weight(gate, r, p_eff, one_minus_pp)

        # Gate pathway: the cell was expressing normally but dropout
        # prevented observation.  Denoised count follows the prior
        # NB(r, p) (or BNB(r, p, omega) when BNB is active).
        if zi_zero_method == "mean":
            gate_val = r * p_eff / (1.0 - p_eff)
        elif zi_zero_method == "mode":
            gate_val = jnp.floor(
                jnp.maximum(r - 1.0, 0.0) * p_eff / (1.0 - p_eff)
            )
        else:
            key_gate, rng_key = random.split(rng_key)
            gate_val = build_count_dist(r, p_eff, bnb_concentration).sample(
                key_gate
            )

        # NB/BNB pathway value at u=0: the posterior for unobserved mRNA
        # given that the NB/BNB component produced the zero.  For VCP
        # models this is positive; without VCP probs_post=0 so it is 0.
        if zi_zero_method == general_method:
            nb_zero_val = denoised_nb
        elif _use_bnb_denoise and zi_zero_method in ("mean", "mode"):
            # Reuse precomputed BNB quadrature mean/mode
            nb_zero_val = _bnb_mean
            if zi_zero_method == "mode":
                nb_zero_val = jnp.floor(nb_zero_val)
        elif _use_bnb_denoise and zi_zero_method == "sample":
            # Augmented BNB sampling for the zero-correction path
            key_p_z, key_nb_z, rng_key = random.split(rng_key, 3)
            p_sampled_z = _sample_p_posterior_bnb(
                key_p_z, counts, r, p_eff, p_capture, bnb_concentration
            )
            probs_cond_z = p_sampled_z * (1.0 - nu)
            alpha_z = r + counts
            d_z = dist.NegativeBinomialProbs(
                total_count=alpha_z, probs=probs_cond_z
            ).sample(key_nb_z)
            nb_zero_val = counts + d_z
        elif zi_zero_method == "mean":
            nb_zero_val = (counts + r * probs_post) / one_minus_pp
        elif zi_zero_method == "mode":
            alpha_z = r + counts
            d_mode_z = jnp.floor(
                jnp.maximum(alpha_z - 1.0, 0.0) * probs_post / one_minus_pp
            )
            nb_zero_val = counts + d_mode_z
        else:
            # Sample from the NB posterior for unobserved transcripts
            # at u=0.  For VCP, d ~ NB(r, probs_post) gives the mRNA
            # lost to capture.  Without VCP probs_post=0 → d=0.
            alpha_z = r + counts
            key_nb_z, rng_key = random.split(rng_key)
            d_sample_z = build_count_dist(
                alpha_z, probs_post, bnb_concentration
            ).sample(key_nb_z)
            nb_zero_val = counts + d_sample_z

        # Combine gate and NB pathways at zero positions
        if zi_zero_method == "mean":
            zinb_zero = w * gate_val + (1.0 - w) * nb_zero_val
        elif zi_zero_method == "mode":
            zinb_zero = jnp.where(w > 0.5, gate_val, nb_zero_val)
        else:
            # Sample: use w to decide whether the zero was from dropout.
            # If gate fired (dropout), sample a replacement from the
            # biological prior NB(r, p).  If genuine NB zero, use the
            # NB posterior (accounts for mRNA lost to capture in VCP;
            # collapses to 0 without VCP since probs_post=0).
            key_bern, rng_key = random.split(rng_key)
            chose_gate = dist.Bernoulli(probs=w).sample(key_bern).astype(bool)
            zinb_zero = jnp.where(chose_gate, gate_val, nb_zero_val)

        denoised = jnp.where(is_zero, zinb_zero, denoised_nb)

        # Variance at zero positions: law of total variance for the mixture
        var_gate = r * p_eff / (1.0 - p_eff) ** 2
        var_nb_zero = var_nb  # already correct at u=0 positions
        mean_gate = r * p_eff / (1.0 - p_eff)
        mean_nb_zero = (r * probs_post) / one_minus_pp
        mixture_var = (
            w * var_gate
            + (1.0 - w) * var_nb_zero
            + w * (1.0 - w) * (mean_gate - mean_nb_zero) ** 2
        )
        variance = jnp.where(is_zero, mixture_var, var_nb)
    else:
        denoised = denoised_nb
        variance = var_nb

    return denoised, variance


# ------------------------------------------------------------------------------


def _compute_gate_weight(
    gate: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    one_minus_probs_post: jnp.ndarray,
) -> jnp.ndarray:
    """Posterior probability that a zero observation came from the gate.

    Implements Bayes' rule for the zero-inflation mixture:

    .. math::

        w = \\frac{g}{g + (1-g)\\,\\hat{p}_{\\text{paper}}^{\\,r_g}}

    where :math:`\\hat{p}_{\\text{paper}} = (1 - p_{\\text{can}}) /
    (1 - p_{\\text{can}}(1-\\nu_c))`.

    In the numpyro probs convention, the NB probability of observing zero
    is :math:`(1 - \\text{probs})^r`.  For the *observation* model the
    relevant probs is ``p_hat_numpyro``, and its complement is exactly
    ``(1 - canonical_p) / one_minus_probs_post``.

    Parameters
    ----------
    gate : jnp.ndarray
        Gate probability, ``(n_genes,)`` or ``(batch_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion, ``(n_genes,)`` or ``(batch_cells, n_genes)``.
    p : jnp.ndarray
        Canonical success probability (scalar or broadcastable).
    one_minus_probs_post : jnp.ndarray
        ``1 - probs_post``, the paper's :math:`p'`.  Shape ``()`` or
        ``(batch_cells, 1)``.

    Returns
    -------
    jnp.ndarray
        Gate weight *w* with the same shape as ``gate`` (broadcast).
    """
    # p_hat_paper = (1 - canonical_p) / one_minus_probs_post
    # P_NB(u=0) = p_hat_paper^r  in the paper convention
    p_hat_paper = (1.0 - p) / one_minus_probs_post
    nb_zero_prob = p_hat_paper**r

    w = gate / (gate + (1.0 - gate) * nb_zero_prob)
    return w


# ------------------------------------------------------------------------------


def _denoise_mixture_marginal(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    return_variance: bool,
    mixing_weights: jnp.ndarray,
    cell_batch_size: Optional[int],
    bnb_concentration: Optional[jnp.ndarray] = None,
    *,
    param_layouts: Dict[str, "AxisLayout"],
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Denoise by marginalising over mixture components.

    For ``method='mean'`` (or general_method ``'mean'``):

    .. math::

        \\mathbb{E}[m_g \\mid u_g] = \\sum_k w_k \\,
        \\mathbb{E}[m_g \\mid u_g, \\text{comp}=k]

    For ``method='sample'`` (or general_method ``'sample'``): sample a
    component per cell from ``mixing_weights``, then sample from that
    component's denoised posterior.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion ``(n_components, n_genes)``.
    p : jnp.ndarray
        Success probability, scalar or ``(n_components,)``.
    p_capture : jnp.ndarray or None
        Capture probability ``(n_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate ``(n_genes,)`` or ``(n_components, n_genes)`` or ``None``.
    method : str or tuple of (str, str)
        Denoising method.
    rng_key : random.PRNGKey or None
        PRNG key.
    return_variance : bool
        Whether to return variance.
    mixing_weights : jnp.ndarray
        Component weights ``(n_components,)``.
    cell_batch_size : int or None
        Cell batching.
    bnb_concentration : jnp.ndarray or None
        Optional BNB concentration.
    param_layouts : dict of str to AxisLayout
        MAP-level semantic layouts for layout-driven flag computation.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts (and optionally variance).
    """
    general_method = method[0] if isinstance(method, tuple) else method

    n_components = r.shape[0]

    # Layout-derived flags — no shape heuristics.
    p_is_comp = (
        param_layouts["p"].component_axis is not None
        if "p" in param_layouts else False
    )
    _p_has_genes = (
        param_layouts["p"].gene_axis is not None
        if "p" in param_layouts else False
    )
    _gate_is_comp = (
        param_layouts["gate"].component_axis is not None
        if "gate" in param_layouts else False
    )

    if general_method == "sample":
        key_comp, key_rest = random.split(rng_key)
        comp = dist.Categorical(probs=mixing_weights).sample(
            key_comp, (counts.shape[0],)
        )
        r_cell = r[comp]
        p_cell = p[comp] if p_is_comp else p
        # (K,) gathered → (n_cells,); expand for broadcasting.
        if p_is_comp and not _p_has_genes:
            p_cell = p_cell[:, None]
        g_cell = gate[comp] if gate is not None and _gate_is_comp else gate
        return _denoise_standard(
            counts,
            r_cell,
            p_cell,
            p_capture,
            g_cell,
            method,
            key_rest,
            return_variance,
            cell_batch_size,
            bnb_concentration=bnb_concentration,
            # r and gate gathered per-cell from component assignments.
            r_is_per_cell=True,
            p_is_per_cell=p_is_comp,
            gate_is_per_cell=_gate_is_comp,
            bnb_is_per_cell=False,
        )

    # Marginalise over components (mean or mode for the general path).
    # An rng_key may still be needed if zi_zero_method is "sample".
    needs_rng = _method_needs_rng(method)
    n_cells, n_genes = counts.shape
    denoised_acc = jnp.zeros((n_cells, n_genes))
    variance_acc = jnp.zeros((n_cells, n_genes))

    for k in range(n_components):
        r_k = r[k]
        p_k = p[k] if p_is_comp else p
        g_k = gate[k] if gate is not None and _gate_is_comp else gate

        # Split rng_key per component if the zi_zero path needs sampling
        if needs_rng:
            rng_key, comp_key = random.split(rng_key)
        else:
            comp_key = None

        # Each component slice is gene-level, not per-cell.
        out_k = _denoise_standard(
            counts,
            r_k,
            p_k,
            p_capture,
            g_k,
            method,
            comp_key,
            True,
            cell_batch_size,
            bnb_concentration=bnb_concentration,
        )

        d_k = out_k["denoised_counts"]
        v_k = out_k["variance"]
        w_k = mixing_weights[k]

        denoised_acc = denoised_acc + w_k * d_k
        # Law of total variance: Var = E[Var_k] + Var[E_k]
        variance_acc = variance_acc + w_k * (v_k + d_k**2)

    variance = variance_acc - denoised_acc**2

    if return_variance:
        return {"denoised_counts": denoised_acc, "variance": variance}
    return denoised_acc
