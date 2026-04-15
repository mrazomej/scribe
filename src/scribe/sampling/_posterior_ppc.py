"""Full-model posterior PPC sampling (NB / ZINB / VCP / mixtures)."""

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
    bnb_concentration : jnp.ndarray or None, optional
        BNB concentration parameter.  ``None`` for non-BNB models.
    param_layouts : dict of str to AxisLayout, optional
        Semantic axis layouts keyed by canonical parameter name
        (``"r"``, ``"p"``, ``"gate"``, ``"p_capture"``, …).  These
        are provided automatically by results-object methods.  Passing
        ``None`` triggers a deprecated fallback that infers layouts
        from tensor shapes.

        .. deprecated::
            Omitting *param_layouts* is deprecated and will raise an
            error in a future release.

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

    # param_layouts should always be provided by results-object callers.
    # The fallback infers layouts from tensor shapes and will be removed
    # in a future release.
    if param_layouts is None:
        warnings.warn(
            "Calling sample_posterior_ppc without param_layouts is "
            "deprecated. Pass param_layouts explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..core.axis_layout import infer_layout

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
        # MAP path: vmap over independent RNG keys with broadcast
        # parameters.  All parameters are identical across draws — only
        # the PRNG key varies.  cell_batch_size is set to None inside
        # the vmap'd function so XLA fuses the full cell dimension;
        # adaptive chunking over n_samples prevents OOM.
        keys = random.split(rng_key, n_samples)

        _has_gate = gate is not None
        _has_p_capture = p_capture is not None
        _is_mixture = is_mixture
        _has_bnb = bnb_concentration is not None

        def _sample_one_map(key_i):
            return _sample_posterior_ppc_single(
                r=r,
                p=p,
                n_cells=n_cells,
                rng_key=key_i,
                gate=gate if _has_gate else None,
                p_capture=p_capture if _has_p_capture else None,
                mixing_weights=mixing_weights if _is_mixture else None,
                cell_batch_size=None,
                bnb_concentration=(
                    bnb_concentration if _has_bnb else None
                ),
                p_has_components=_p_has_comp,
                p_has_genes=_p_has_genes,
                gate_has_components=_gate_comp,
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
            # Standard (non-mixture) model
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
