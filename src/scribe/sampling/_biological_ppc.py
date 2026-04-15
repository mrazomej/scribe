"""Biological (denoised) PPC sampling — base Negative Binomial only."""

import warnings
from typing import Dict, Optional

from jax import random, vmap
import jax.numpy as jnp
import numpyro.distributions as dist

from scribe.models.components.likelihoods.beta_negative_binomial import (
    build_count_dist,
)

from ..core._array_dispatch import _vmap_chunk_size
from ._helpers import _has_sample_dim


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
    bnb_concentration : jnp.ndarray or None, optional
        BNB concentration parameter.  ``None`` for non-BNB models.
    param_layouts : dict of str to AxisLayout, optional
        Semantic axis layouts keyed by canonical parameter name
        (``"r"``, ``"p"``, ``"mixing_weights"``, …).  These are
        provided automatically by results-object methods.  Passing
        ``None`` triggers a deprecated fallback that infers layouts
        from tensor shapes.

        .. deprecated::
            Omitting *param_layouts* is deprecated and will raise an
            error in a future release.

    Returns
    -------
    jnp.ndarray
        Sampled counts with shape ``(n_samples, n_cells, n_genes)``.

    Notes
    -----
    The mathematical justification is that the VCP model composes a base
    NB(r, p) with a Binomial capture step:

        p̂ = p·ν / (1 − p·(1 − ν))

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

    # param_layouts should always be provided by results-object callers.
    # The fallback infers layouts from tensor shapes and will be removed
    # in a future release.
    if param_layouts is None:
        warnings.warn(
            "Calling sample_biological_nb without param_layouts is "
            "deprecated. Pass param_layouts explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..core.axis_layout import infer_layout

        _expected_r_rank = 2 if is_mixture else 1
        _has_sd = r.ndim > _expected_r_rank
        _n_comp = r.shape[-2] if is_mixture and r.ndim >= 2 else None
        _params: Dict[str, jnp.ndarray] = {"r": r, "p": p}
        if mixing_weights is not None:
            _params["mixing_weights"] = mixing_weights
        if bnb_concentration is not None:
            _params["bnb_concentration"] = bnb_concentration
        param_layouts = {
            k: infer_layout(
                k,
                v,
                n_genes=int(r.shape[-1]),
                n_cells=n_cells,
                n_components=_n_comp,
                has_sample_dim=_has_sd,
            )
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
    _p_has_genes = _base["p"].gene_axis is not None if "p" in _base else False
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
        # MAP path: vmap over independent RNG keys with broadcast
        # parameters.  All parameters are identical across draws — only
        # the PRNG key varies.  cell_batch_size is set to None inside
        # the vmap'd function so XLA fuses the full cell dimension;
        # adaptive chunking over n_samples prevents OOM.
        keys = random.split(rng_key, n_samples)

        _is_mixture = is_mixture
        _has_bnb = bnb_concentration is not None

        def _sample_one_map(key_i):
            return _sample_biological_nb_single(
                r=r,
                p=p,
                n_cells=n_cells,
                rng_key=key_i,
                mixing_weights=mixing_weights if _is_mixture else None,
                cell_batch_size=None,
                bnb_concentration=(
                    bnb_concentration if _has_bnb else None
                ),
                p_has_components=_p_has_comp,
                p_has_genes=_p_has_genes,
                bnb_has_components=_bnb_comp,
            )

        # Estimate per-sample memory: output + intermediates
        _n_genes = int(r.shape[-1])
        _per_sample = n_cells * _n_genes * 4 * 2
        _chunk = _vmap_chunk_size(n_samples, _per_sample)

        if _chunk >= n_samples:
            return vmap(_sample_one_map)(keys)
        # Chunk over the n_samples axis to stay within GPU memory
        parts = []
        for _s in range(0, n_samples, _chunk):
            _e = min(_s + _chunk, n_samples)
            parts.append(vmap(_sample_one_map)(keys[_s:_e]))
        return jnp.concatenate(parts, axis=0)


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
