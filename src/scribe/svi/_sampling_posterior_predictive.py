"""
Posterior and constrained predictive sampling mixin for SVI results.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import jax.numpy as jnp
from jax import random
import logging

from ..sampling import generate_predictive_samples, sample_variational_posterior
from ..core.posterior_matrix import posterior_samples_to_matrix

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.axis_layout import AxisLayout


# ---------------------------------------------------------------------------
# Helpers for flow-guide-aware gene subsetting
# ---------------------------------------------------------------------------


def _has_flow_params(params: Dict[str, Any]) -> bool:
    """Return True if the param dict contains normalizing flow weights."""
    return any(
        k.endswith("$params")
        and (k.startswith("flow_") or k.startswith("joint_flow_"))
        for k in params
    )


def _subset_gene_dim_samples(
    samples: Dict[str, Any],
    gene_index: np.ndarray,
    original_n_genes: int,
    layouts: Optional[Dict[str, "AxisLayout"]] = None,
) -> Dict[str, Any]:
    """Slice the gene dimension of posterior/predictive sample arrays.

    When *layouts* are provided the gene axis is determined semantically
    via ``AxisLayout.gene_axis`` — no shape scanning needed.  When
    *layouts* is ``None`` the function falls back to matching
    *original_n_genes* against the value's shape (the legacy heuristic).

    Parameters
    ----------
    samples : dict
        Mapping from parameter name to array.
    gene_index : np.ndarray
        Integer indices selecting a subset of genes.
    original_n_genes : int
        Total number of genes in the full (un-subsetted) array.
        Used only by the fallback shape-scanning path.
    layouts : dict or None, optional
        ``{key: AxisLayout}`` mapping.  When available, gene-axis
        identification is exact rather than heuristic.

    Returns
    -------
    dict
        Copy of *samples* with the gene dimension sliced where present.
    """
    out: Dict[str, Any] = {}
    for key, value in samples.items():
        if not hasattr(value, "shape"):
            out[key] = value
            continue

        # Prefer semantic gene-axis lookup from AxisLayout when available
        if layouts is not None and key in layouts:
            gene_ax = layouts[key].gene_axis
            if gene_ax is not None:
                slicer = [slice(None)] * value.ndim
                slicer[gene_ax] = gene_index
                out[key] = value[tuple(slicer)]
            else:
                out[key] = value
        elif original_n_genes in value.shape:
            # Legacy fallback: match the first axis whose size equals
            # original_n_genes (fragile when multiple dims share that
            # size, but preserved for backward compatibility)
            axis = list(value.shape).index(original_n_genes)
            slicer = [slice(None)] * value.ndim
            slicer[axis] = gene_index
            out[key] = value[tuple(slicer)]
        else:
            out[key] = value
    return out


def _build_cell_specific_sample_keys(param_specs: list) -> set:
    """Collect constrained names of cell-specific parameters.

    Parameters
    ----------
    param_specs : list
        Parameter specifications from ``model_config.param_specs``.

    Returns
    -------
    set
        Constrained parameter names (matching posterior sample keys)
        that are cell-specific.
    """
    if not param_specs:
        return set()
    return {
        spec.name
        for spec in param_specs
        if getattr(spec, "is_cell_specific", False)
    }


def _merge_per_dataset_posterior_samples(
    per_dataset_samples: List[Dict[str, jnp.ndarray]],
    cell_specific_keys: set,
) -> Dict[str, jnp.ndarray]:
    """Re-stack per-dataset posterior samples into a single dict.

    For concatenated independent fits, every non-cell-specific
    parameter is per-dataset (independently estimated).  Three
    categories:

    1. **Cell-specific** (from ``ParamSpec.is_cell_specific``):
       concatenated at axis 1 to reassemble the full cell population.
    2. **Shape-compatible** (same shape across datasets): stacked at
       axis 1 to create the dataset dimension.
    3. **Shape-incompatible** (different shapes, e.g. deterministic
       sites that depend on cell count): concatenated at axis 1.

    Parameters
    ----------
    per_dataset_samples : list of dict
        One posterior-sample dict per dataset, each with arrays of shape
        ``(n_samples, ...)``.
    cell_specific_keys : set
        Constrained parameter names that are cell-specific (e.g.
        ``p_capture``).

    Returns
    -------
    Dict[str, jnp.ndarray]
        Merged posterior samples.  Per-dataset keys with matching shapes
        have ``(n_samples, n_datasets, ...)``.
    """
    reference = per_dataset_samples[0]
    merged: Dict[str, jnp.ndarray] = {}

    for key in reference:
        arrays = [ds_samples[key] for ds_samples in per_dataset_samples]

        if key in cell_specific_keys:
            # Known cell-specific — concatenate along cell axis
            merged[key] = jnp.concatenate(arrays, axis=1)
        else:
            # Check if shapes match (they differ when n_cells varies
            # across datasets, e.g. for deterministic sites).
            shapes_match = all(a.shape == arrays[0].shape for a in arrays)
            if shapes_match:
                merged[key] = jnp.stack(arrays, axis=1)
            else:
                # Shape mismatch implies a cell-dependent quantity —
                # concatenate along axis 1 (cell axis after sample axis)
                merged[key] = jnp.concatenate(arrays, axis=1)

    return merged


def _convert_to_numpy(samples: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all JAX arrays in a posterior-samples dict to NumPy.

    Each value that is a ``jax.Array`` is transferred to host memory and
    returned as a plain ``numpy.ndarray``.  This frees GPU memory
    immediately and lets downstream code (e.g., the DE pipeline) use the
    NumPy/SciPy stack directly — avoiding JAX's XLA CPU backend overhead
    and unnecessary GPU round-trips.  Non-array entries (e.g., metadata)
    are passed through unchanged.

    Parameters
    ----------
    samples : Dict[str, Any]
        Posterior-samples dict as returned by NumPyro ``Predictive``.

    Returns
    -------
    Dict[str, Any]
        Same dict with every JAX array converted to ``numpy.ndarray``.
    """
    import jax

    return {
        k: np.asarray(v) if isinstance(v, jax.Array) else v
        for k, v in samples.items()
    }


def _split_index_ranges(total: int, chunk_size: int) -> List[tuple]:
    """Split ``[0, total)`` into contiguous half-open ``(start, stop)`` ranges.

    Parameters
    ----------
    total : int
        Length of the axis to partition (assumed > 0).
    chunk_size : int
        Maximum width of each range (must be > 0); the final range is shorter
        when ``total`` is not an exact multiple of ``chunk_size``.

    Returns
    -------
    list of tuple
        ``(start, stop)`` ranges that exactly tile ``[0, total)``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    return [
        (start, min(start + chunk_size, total))
        for start in range(0, total, chunk_size)
    ]


def _concat_posterior_chunks(
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Concatenate per-chunk posterior dicts along the sample axis (axis 0).

    Used by the sample-batched posterior draw: each chunk holds a disjoint
    block of posterior samples but identical keys and identical non-sample
    axes, so array values are concatenated along axis 0 to reassemble the
    full ``n_samples`` draw.  Non-array (or zero-dim) entries are taken from
    the first chunk unchanged.  Inputs are expected to already be host
    (NumPy) arrays so the concatenation never re-occupies device memory.

    Parameters
    ----------
    chunks : list of dict
        Per-chunk posterior-samples dicts, in sample order. Must be non-empty.

    Returns
    -------
    Dict[str, Any]
        A single dict whose array values span every chunk's samples.
    """
    if len(chunks) == 1:
        return chunks[0]
    merged: Dict[str, Any] = {}
    for key, value in chunks[0].items():
        if getattr(value, "ndim", 0) >= 1:
            merged[key] = np.concatenate(
                [chunk[key] for chunk in chunks], axis=0
            )
        else:
            merged[key] = value
    return merged


class PosteriorPredictiveSamplingMixin:
    """Mixin providing posterior and constrained predictive sampling methods."""

    def _is_concatenated_multi_dataset(self) -> bool:
        """Check if this results object was produced by concatenating
        independent single-dataset fits.

        Concatenated results have ``_promoted_dataset_keys`` set to a
        non-empty set by ``ScribeSVIResults.concat``.  Jointly
        hierarchical multi-dataset models (fit with
        ``hierarchical_dataset_*`` / ``horseshoe_dataset`` flags) have
        ``_promoted_dataset_keys = None`` — their guide was built for
        the multi-dataset structure natively and ``Predictive`` works.

        Returns
        -------
        bool
            True if per-dataset decomposition is required for posterior
            sampling.
        """
        promoted = getattr(self, "_promoted_dataset_keys", None)
        return promoted is not None and len(promoted) > 0

    def get_posterior_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        convert_to_numpy: bool = False,
        counts: Optional[jnp.ndarray] = None,
        descriptive_names: bool = False,
    ) -> Dict:
        """Sample parameters from the variational posterior distribution.

        For concatenated multi-dataset results (produced by
        ``ScribeSVIResults.concat``), sampling is performed
        independently on each dataset via ``get_dataset(d)`` and the
        results are re-stacked.  For jointly hierarchical models (fit
        with ``n_datasets > 1`` natively), the standard ``Predictive``
        path is used directly.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key (default: PRNGKey(42))
        n_samples : int, optional
            Number of posterior samples to generate (default: 100)
        batch_size : Optional[int], optional
            If set, cap peak device memory by drawing the posterior in chunks
            of ``batch_size`` samples along the *sample* axis, offloading each
            chunk to host and concatenating. Device memory then stays bounded
            by one chunk regardless of ``n_samples``. The cell plate is never
            subsampled, so cell-specific tensors (e.g. ``p_capture``) keep
            their full ``n_cells`` width; a batched draw is returned as NumPy
            arrays (host accumulation makes ``convert_to_numpy`` redundant).
            Default: None (a single draw of all ``n_samples`` at once).
        store_samples : bool, optional
            Whether to store samples in self.posterior_samples (default: True)
        convert_to_numpy : bool, optional
            If True, convert all sampled arrays to plain ``numpy.ndarray``
            before returning (and before optional storing). This frees GPU memory
            immediately after sampling, which is important when downstream
            operations (e.g., differential expression) need GPU headroom.
            The DE pipeline's array-backend dispatch will then use the
            NumPy/SciPy stack directly for summary statistics, avoiding
            JAX's XLA CPU backend overhead and unnecessary GPU
            round-trips.  This flag is independent from ``store_samples``:
            callers can request NumPy return values without mutating
            ``self.posterior_samples`` by setting
            ``store_samples=False, convert_to_numpy=True``. Default: False.
        counts : Optional[jnp.ndarray], optional
            Observed count matrix of shape (n_cells, n_genes). Required when
            using amortized capture probability (e.g., with
            amortization.capture.enabled=true).

            IMPORTANT: When using amortized capture with gene-subset results,
            you must pass the ORIGINAL full-gene count matrix, not a gene-subset.
            The amortizer computes sufficient statistics (e.g., total UMI count)
            by summing across ALL genes, so it requires the full data.

            For non-amortized models, this can be None. Default: None.
        descriptive_names : bool, default=False
            If True, rename dict keys from internal short names (``r``,
            ``p``, ``mu``, ``phi``, ``gate``, ...) to user-friendly
            descriptive names (``dispersion``, ``prob``, ``expression``,
            ``odds``, ``zero_inflation``, ...).

        Returns
        -------
        Dict
            Dictionary containing samples from the variational posterior
        """
        # Validate counts for amortized capture (checks original gene count)
        # This uses methods from ParameterExtractionMixin (inherited by ScribeSVIResults)
        if self._uses_amortized_capture():
            if counts is None:
                raise ValueError(
                    "counts parameter is required when using amortized capture "
                    "probability. Please provide the observed count matrix of shape "
                    "(n_cells, n_genes) that was used during inference."
                )
            self._validate_counts_for_amortizer(
                counts, context="posterior sampling"
            )

        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # A single full-cell-width draw of ``draw_n`` posterior samples.
        # Concatenated multi-dataset results cannot run ``Predictive`` on the
        # stacked params (the guide was built for single-dataset models), so
        # they decompose into per-dataset sampling; jointly hierarchical
        # multi-dataset models use the standard path directly.  ``batch_size``
        # is intentionally NOT forwarded here — it is a *sample-axis* memory
        # control handled below, not a cell-plate subsample.
        def _draw_full_width(draw_key, draw_n):
            if self._is_concatenated_multi_dataset():
                return self._get_posterior_samples_per_dataset(
                    rng_key=draw_key,
                    n_samples=draw_n,
                    batch_size=None,
                    counts=counts,
                )
            return self._get_posterior_samples_standard(
                rng_key=draw_key,
                n_samples=draw_n,
                batch_size=None,
                counts=counts,
            )

        # ``batch_size`` bounds peak device memory by chunking the *sample*
        # axis: each chunk draws up to ``batch_size`` samples at full cell width
        # and is offloaded to host before the next chunk is drawn, so the device
        # only ever holds one chunk's worth of draws regardless of
        # ``n_samples``.  Chunking the sample axis (rather than the cell axis)
        # shrinks *every* posterior tensor uniformly -- including the gene- and
        # dataset-level tensors that have no cell axis and dominate memory -- and
        # never subsamples the cell plate, so cell-specific tensors (e.g.
        # ``p_capture``) are returned at full ``n_cells`` width.  Each chunk uses
        # an independent rng subkey so the pooled draws are distinct samples.
        if batch_size is not None and int(batch_size) < int(n_samples):
            ranges = _split_index_ranges(int(n_samples), int(batch_size))
            subkeys = random.split(rng_key, len(ranges))
            chunks = [
                _convert_to_numpy(
                    _draw_full_width(subkey, int(stop - start))
                )
                for (start, stop), subkey in zip(ranges, subkeys)
            ]
            posterior_samples = _concat_posterior_chunks(chunks)
        else:
            posterior_samples = _draw_full_width(rng_key, int(n_samples))

        # Single-leaf multi-factor view (from get_dataset on a multi-factor
        # fit): the model reconstructs a size-1 dataset axis; squeeze it so the
        # view presents single-dataset (S, G) arrays consistent with its stored
        # (sliced) posterior.
        posterior_samples = self._squeeze_collapsed_leaf_axis(posterior_samples)

        # Convert independently from storage behavior. This enables callers
        # to request NumPy return values while keeping results object state
        # unchanged (store_samples=False).
        if convert_to_numpy:
            posterior_samples = _convert_to_numpy(posterior_samples)

        if store_samples:
            self.posterior_samples = posterior_samples

        from ..models.config.parameter_mapping import rename_dict_keys

        return rename_dict_keys(posterior_samples, descriptive_names)

    def _squeeze_collapsed_leaf_axis(self, samples: Optional[Dict]) -> Optional[Dict]:
        """Squeeze the singleton dataset axis from a single-leaf view.

        ``get_dataset`` on a multi-factor fit restricts the model to one leaf
        (``n_datasets == 1`` with a 1-leaf ``grouping_spec``) so re-sampling
        reconstructs that leaf via the additive hierarchy. The reconstruction
        carries a size-1 dataset axis, which is removed here so the view's
        re-sampled arrays match its stored, leaf-sliced single-dataset ``(S, G)``
        shape. No-op for ordinary results.
        """
        if samples is None:
            return samples
        mc = self.model_config
        gs = getattr(mc, "grouping_spec", None)
        if not (
            getattr(mc, "n_datasets", None) == 1
            and gs is not None
            and getattr(gs, "n_leaves", None) == 1
        ):
            return samples

        from ..core.axis_layout import (
            build_sample_layouts,
            derive_axis_membership,
        )

        _mp, _dp = derive_axis_membership(mc, samples=samples, has_sample_dim=True)
        layouts = build_sample_layouts(
            list(mc.param_specs or []),
            samples,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            n_components=getattr(mc, "n_components", None),
            n_datasets=1,
            mixture_params=_mp,
            dataset_params=_dp,
            has_sample_dim=True,
        )
        out: Dict = {}
        for key, value in samples.items():
            lay = layouts.get(key)
            ax = lay.dataset_axis if lay is not None else None
            if (
                ax is not None
                and hasattr(value, "ndim")
                and value.ndim > ax
                and value.shape[ax] == 1
            ):
                out[key] = jnp.squeeze(value, axis=ax)
            else:
                out[key] = value
        return out

    def get_posterior_matrix(
        self,
        *,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        counts: Optional[jnp.ndarray] = None,
        store_samples: bool = False,
        convert_to_numpy: bool = True,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        exclude_deterministic: bool = True,
        coords: Optional[Dict[str, Any]] = None,
        descriptive_names: bool = False,
    ) -> tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Export posterior draws as a 2D feature matrix.

        This method wraps :meth:`get_posterior_samples` and flattens all
        selected posterior tensors into a single matrix with one row per
        posterior draw and one column per parameter element.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random key forwarded to :meth:`get_posterior_samples`.
        n_samples : int, default=100
            Number of posterior draws to sample.
        batch_size : int or None, optional
            Optional batch size for memory-efficient posterior sampling.
        counts : jnp.ndarray or None, optional
            Observed count matrix required for amortized capture models.
        store_samples : bool, default=False
            Whether sampled posterior tensors should be stored on the
            results object before matrix conversion.
        convert_to_numpy : bool, default=True
            Whether posterior tensors should be converted to NumPy arrays
            before flattening. This is recommended for downstream plotting
            and avoids keeping JAX device memory occupied.
        include : list of str or None, optional
            Optional whitelist of posterior parameter keys to export.
        exclude : list of str or None, optional
            Optional blacklist of posterior parameter keys to skip.
        exclude_deterministic : bool, default=True
            Whether to exclude common deterministic/derived keys for the
            active parameterization (for example ``mu`` in canonical mode).
        coords : dict or None, optional
            Optional coordinate selectors by semantic axis name. Example:
            ``{"genes": ["GeneA", "GeneB"]}``.
        descriptive_names : bool, default=False
            If True, use descriptive parameter names in both sampled keys
            and exported column labels.

        Returns
        -------
        matrix : numpy.ndarray
            Posterior matrix with shape ``(n_draws, n_features)``.
        columns : list of str
            Feature label per matrix column.
        metadata : list of dict
            Per-feature metadata aligned with ``columns``. Each record
            includes parameter key and axis index/value mappings.
        """
        # Reuse the canonical posterior-sampling path so this method inherits
        # all existing validation (e.g., amortized-count checks).
        posterior_samples = self.get_posterior_samples(
            rng_key=rng_key,
            n_samples=n_samples,
            batch_size=batch_size,
            store_samples=store_samples,
            convert_to_numpy=convert_to_numpy,
            counts=counts,
            descriptive_names=descriptive_names,
        )

        # Use semantic layouts to flatten tensors in a robust, axis-aware way.
        return posterior_samples_to_matrix(
            posterior_samples=posterior_samples,
            base_layouts=self.layouts,
            model_config=self.model_config,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            include=include,
            exclude=exclude,
            exclude_deterministic=exclude_deterministic,
            coords=coords,
            var_index=(
                self.var.index
                if getattr(self, "var", None) is not None
                else None
            ),
            descriptive_names=descriptive_names,
        )

    def _get_posterior_samples_standard(
        self,
        rng_key: random.PRNGKey,
        n_samples: int,
        batch_size: Optional[int],
        counts: Optional[jnp.ndarray],
    ) -> Dict:
        """Standard posterior sampling via NumPyro Predictive."""
        model, guide = self._model_and_guide()

        # VAE generative prior-path: with no counts, a VAE samples z from the
        # N(0, I) prior and pushes it through the trained decoder (no
        # encoder/guide) rather than the encoder-based posterior q(z|counts).
        # Detect it here so we neither require a guide nor run one.
        _is_vae_result = (
            hasattr(self, "_encoder")
            and getattr(self, "_encoder", None) is not None
        )
        _use_vae_prior_path = _is_vae_result and counts is None
        if _use_vae_prior_path:
            import warnings

            _latent_spec = getattr(self, "_latent_spec", None)
            _has_flow_prior = (
                _latent_spec is not None
                and getattr(_latent_spec, "flow", None) is not None
            )
            _base_model = getattr(self.model_config, "base_model", None)
            _base_model = (
                _base_model.value
                if hasattr(_base_model, "value")
                else _base_model
            )
            _parameterization = getattr(
                self.model_config, "parameterization", None
            )
            _parameterization = (
                _parameterization.value
                if hasattr(_parameterization, "value")
                else _parameterization
            )
            _is_lnm_model = (
                self.model_type in ("lnm", "lnmvcp")
                or _base_model in ("lnm", "lnmvcp")
                or _parameterization == "logistic_normal"
            )
            # Warn only for plain VAEs, where prior sampling is least
            # meaningful; LNM and flow-prior VAEs have structured priors.
            if not _is_lnm_model and not _has_flow_prior:
                warnings.warn(
                    "No counts provided for VAE posterior sampling. "
                    "Sampling z from the uninformative N(0, I) prior "
                    "instead of the encoder-based posterior q(z|counts). "
                    "Pass counts= for encoder-based inference.",
                    UserWarning,
                    stacklevel=3,
                )

        if guide is None and not _use_vae_prior_path:
            raise ValueError(
                f"Could not find a guide for model '{self.model_type}'."
            )

        # When the results have been gene-subsetted AND contain flow
        # guide params, the flow must run at the original full
        # dimensionality.  We sample at _original_n_genes and then
        # slice the output to the requested gene indices.
        #
        # Crucially, we must use the *original* unsubsetted params dict
        # for the guide call.  Gene subsetting slices array-valued
        # variational params (e.g. nondense loc/alpha in a joint flow
        # guide) to the gene subset, but the flow chain still expects
        # them at full dimension.  ``_original_params`` is stored by
        # ``_create_subset`` when flow params are detected.
        _orig_ng = getattr(self, "_original_n_genes", None)
        _gene_idx = getattr(self, "_subset_gene_index", None)
        _orig_params = getattr(self, "_original_params", None)
        _full_dim = (
            _orig_ng is not None
            and _orig_ng != self.n_genes
            and _gene_idx is not None
            and _has_flow_params(self.params)
        )
        base_n_genes = self._factory_n_genes() or self.n_genes
        n_genes_for_guide = _orig_ng if _full_dim else int(base_n_genes)
        params_for_guide = (
            _orig_params
            if (_full_dim and _orig_params is not None)
            else self.params
        )

        if counts is not None and int(counts.shape[1]) != int(
            n_genes_for_guide
        ):
            # VAE results may legitimately receive counts with one extra
            # trailing column (the pooled "other" gene from gene-coverage
            # filtering) because the decoder was trained on the narrower
            # gene set.  Trim it only when we can confirm this is a VAE
            # result so non-VAE paths never silently drop data.
            _is_vae = (
                hasattr(self, "_encoder")
                and getattr(self, "_encoder", None) is not None
            )
            if (
                _is_vae
                and int(counts.shape[1]) == int(n_genes_for_guide) + 1
            ):
                _log.info(
                    "Posterior sampling received counts with one extra gene "
                    f"column ({int(counts.shape[1])}) relative to the VAE "
                    f"reconstruction width ({int(n_genes_for_guide)}). "
                    "Dropping trailing column (pooled 'other' gene)."
                )
                counts = counts[:, : int(n_genes_for_guide)]
            else:
                raise ValueError(
                    "Posterior sampling counts width does not match model "
                    f"reconstruction width: counts.shape[1]={int(counts.shape[1])}, "
                    f"expected {int(n_genes_for_guide)}."
                )

        # Include dataset_indices so that when the model is replayed to
        # compute deterministic sites, index_dataset_params can convert
        # (K, D, G) parameters to per-cell layout.
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": n_genes_for_guide,
            "model_config": self.model_config,
        }
        ds_idx = getattr(self, "_dataset_indices", None)
        if ds_idx is not None:
            model_args["dataset_indices"] = ds_idx
        _tcm = getattr(self, "_total_count_max", None)
        if _tcm is not None:
            model_args["total_count_max"] = int(_tcm)

        # Add batch_size to model_args if provided for memory-efficient sampling
        if batch_size is not None:
            model_args["batch_size"] = batch_size

        if _use_vae_prior_path:
            from numpyro.handlers import block
            from numpyro.infer import Predictive

            # Hide the observed-counts site so Predictive samples it from the
            # generative model (prior z -> trained decoder) rather than
            # conditioning on data.
            blocked_model = block(model, hide=["counts"])
            predictive = Predictive(
                blocked_model,
                num_samples=n_samples,
                params=self.params,
                exclude_deterministic=False,
            )
            posterior_samples = predictive(rng_key, **model_args)
        else:
            posterior_samples = sample_variational_posterior(
                guide,
                params_for_guide,
                model,
                model_args,
                rng_key=rng_key,
                n_samples=n_samples,
                counts=counts,
            )

        # Slice full-dim flow samples down to the gene subset.
        # Build sample-level layouts (with_sample_dim) so that gene_axis
        # accounts for the leading posterior-draw dimension.
        if _full_dim:
            _sample_layouts = {
                k: v.with_sample_dim() for k, v in self.layouts.items()
            }
            posterior_samples = _subset_gene_dim_samples(
                posterior_samples,
                _gene_idx,
                _orig_ng,
                layouts=_sample_layouts,
            )

        return posterior_samples

    def _get_posterior_samples_per_dataset(
        self,
        rng_key: random.PRNGKey,
        n_samples: int,
        batch_size: Optional[int],
        counts: Optional[jnp.ndarray],
    ) -> Dict:
        """Posterior sampling for concatenated multi-dataset results.

        Iterates over ``get_dataset(d)`` to produce single-dataset
        views where ``Predictive`` works, then re-stacks promoted
        keys along the dataset axis and concatenates cell-specific
        keys along the cell axis.
        """
        n_datasets = self.model_config.n_datasets
        ds_indices = getattr(self, "_dataset_indices", None)
        cell_keys = _build_cell_specific_sample_keys(
            self.model_config.param_specs or []
        )

        per_dataset_samples: List[Dict[str, jnp.ndarray]] = []

        for d in range(n_datasets):
            rng_key, ds_key = random.split(rng_key)

            ds_view = self.get_dataset(d)

            # Subset counts to this dataset's cells if counts provided
            ds_counts = None
            if counts is not None and ds_indices is not None:
                mask = ds_indices == d
                ds_counts = counts[mask]

            ds_samples = ds_view.get_posterior_samples(
                rng_key=ds_key,
                n_samples=n_samples,
                batch_size=batch_size,
                store_samples=False,
                counts=ds_counts,
            )
            per_dataset_samples.append(ds_samples)

        return _merge_per_dataset_posterior_samples(
            per_dataset_samples, cell_keys
        )

    def get_predictive_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        store_samples: bool = True,
        counts: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples.

        Two PPC modes, dispatched on whether ``counts`` is provided:

        * **Generative PPC** (``counts=None``) — every latent in the
          model is sampled fresh from its posterior draw, including
          per-cell totals. For LNM-family models this means
          ``u_T_c ~ NB(r_T, p)`` is re-rolled per posterior sample.
          Answers "for a *new* cell drawn from the same population
          as my data, what counts would I expect?"
        * **Conditional PPC** (``counts`` provided) — for LNM-family
          models only, the per-cell observed totals
          ``u_T_obs = sum(counts, axis=-1)`` are injected via
          ``numpyro.handlers.condition`` so the predictive replay
          fixes ``u_T`` and only ``counts`` is sampled fresh from
          ``Multinomial(u_T_obs, softmax(y_alr))``. Answers "given
          *this specific cell's* observed library size, what gene-
          by-gene counts does the model predict for it?"

        For non-LNM models, the ``counts`` argument is currently
        ignored (no analogous total-count latent to condition on);
        a future extension can plumb conditional PPC through to
        the relevant site for other models.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key (default ``PRNGKey(42)``).
        store_samples : bool, optional
            Whether to store the predictive samples in
            ``self.predictive_samples`` (default ``True``).
        counts : jnp.ndarray, optional
            Observed count matrix of shape ``(n_cells, n_genes)``.
            When provided for an LNM-family model, switches to
            conditional PPC mode (see above). When ``None`` (the
            default) the original generative PPC is produced.

        Returns
        -------
        jnp.ndarray
            Array of predictive count samples; shape
            ``(n_samples, n_cells, n_genes)``.
        """
        from ..models.config import GuideFamilyConfig
        from ..models.model_registry import get_model_and_guide

        # For predictive sampling, we need the *constrained* model, which has
        # the 'counts' sample site. The posterior samples from the unconstrained
        # guide can be used with the constrained model.
        # Use an empty GuideFamilyConfig so the guide is built with default
        # MeanField families — the guide is discarded anyway, and keeping the
        # original families (e.g. JointLowRankGuide) would be incompatible
        # with constrained specs that lack a .transform attribute.
        model, _, model_config_for_pred = get_model_and_guide(
            self.model_config,
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
            n_genes=self._factory_n_genes(),
        )
        n_genes_for_model = int(self._factory_n_genes() or self.n_genes)

        # Use model_config_for_pred (which has param_specs populated) so
        # index_dataset_params in the likelihood can correctly identify
        # mixture+dataset params.  Include dataset_indices so multi-dataset
        # params are indexed to per-cell layout.
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": n_genes_for_model,
            "model_config": model_config_for_pred,
        }
        ds_idx = getattr(self, "_dataset_indices", None)
        if ds_idx is not None:
            model_args["dataset_indices"] = ds_idx
        _tcm = getattr(self, "_total_count_max", None)
        if _tcm is not None:
            model_args["total_count_max"] = int(_tcm)

        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Build the optional condition_data dict for conditional-PPC
        # mode. Only LNM-family models have a ``u_T`` latent that
        # makes sense to fix at observed totals; other models silently
        # ignore the ``counts`` arg here for now.
        condition_data: Optional[Dict[str, jnp.ndarray]] = None
        if counts is not None and self._is_lnm_family_model():
            counts_arr = jnp.asarray(counts)
            # Drop a trailing pooled "_other" column if the user
            # passed full original-gene counts to a coverage-filtered
            # result. The ``u_T`` site lives in model-space (post-
            # filter), so we sum across the model-space columns.
            if int(counts_arr.shape[1]) == n_genes_for_model + 1:
                counts_arr = counts_arr[:, :n_genes_for_model]
            elif int(counts_arr.shape[1]) != n_genes_for_model:
                raise ValueError(
                    "Conditional PPC counts width does not match model "
                    f"reconstruction width: counts.shape[1]="
                    f"{int(counts_arr.shape[1])}, expected "
                    f"{n_genes_for_model} (or {n_genes_for_model + 1} "
                    "for the trailing pooled '_other' column)."
                )
            u_T_obs = counts_arr.sum(axis=-1)
            condition_data = {"u_T": u_T_obs}

        # Generative-mode bookkeeping: if ``counts`` was passed to a
        # prior ``get_posterior_samples`` call, NumPyro's ``Predictive``
        # captured the observed ``u_T = sum(counts, axis=-1)`` into
        # ``self.posterior_samples`` (it is reachable in the model
        # trace as a non-guide site). Re-substituting that fixed
        # value during predictive replay would lock per-cell totals
        # to the observed values even when the caller asked for a
        # generative PPC — exactly defeating the purpose of
        # ``counts=None``. Drop it so the model resamples ``u_T``
        # from its NB. (Only relevant for LNM-family models.)
        posterior_samples_for_predictive = self.posterior_samples
        if (
            condition_data is None
            and self._is_lnm_family_model()
            and "u_T" in self.posterior_samples
        ):
            posterior_samples_for_predictive = {
                k: v
                for k, v in self.posterior_samples.items()
                if k != "u_T"
            }

        # Generate predictive samples
        predictive_samples = generate_predictive_samples(
            model,
            posterior_samples_for_predictive,
            model_args,
            rng_key=rng_key,
            condition_data=condition_data,
        )

        # Store samples if requested
        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

    def _is_lnm_family_model(self) -> bool:
        """Return True when this result is from an LNM-family model.

        LNM-family models (``"lnm"`` and ``"lnmvcp"``) have a
        ``"u_T"`` total-count latent that can be conditioned on
        observed library sizes for conditional PPC. Other models
        either lack such a site or use a different name; conditional
        PPC for those is a future extension.
        """
        bm = getattr(self.model_config, "base_model", None)
        return bm in ("lnm", "lnmvcp")

    def get_ppc_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        counts: Optional[jnp.ndarray] = None,
    ) -> Dict:
        """Generate posterior predictive check samples.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key (default: PRNGKey(42))
        n_samples : int, optional
            Number of posterior samples to generate (default: 100)
        batch_size : Optional[int], optional
            Batch size for generating samples (default: None)
        store_samples : bool, optional
            Whether to store samples in self.posterior_samples and
            self.predictive_samples (default: True)
        counts : Optional[jnp.ndarray], optional
            Observed count matrix of shape (n_cells, n_genes). Required when
            using amortized capture probability (e.g., with
            amortization.capture.enabled=true).

            IMPORTANT: When using amortized capture with gene-subset results,
            you must pass the ORIGINAL full-gene count matrix, not a gene-subset.
            The amortizer computes sufficient statistics (e.g., total UMI count)
            by summing across ALL genes, so it requires the full data.

            For non-amortized models, this can be None. Default: None.

        Returns
        -------
        Dict
            Dictionary containing:
            - 'parameter_samples': Samples from the variational posterior
            - 'predictive_samples': Samples from the predictive distribution
        """
        # Validate counts for amortized capture (checks original gene count)
        if self._uses_amortized_capture():
            if counts is None:
                raise ValueError(
                    "counts parameter is required when using amortized capture "
                    "probability. Please provide the observed count matrix of shape "
                    "(n_cells, n_genes) that was used during inference."
                )
            self._validate_counts_for_amortizer(counts, context="PPC sampling")

        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Check if we need to resample parameters
        need_params = self.posterior_samples is None

        # Generate posterior samples if needed
        if need_params:
            # Sample parameters and generate predictive samples
            self.get_posterior_samples(
                rng_key=rng_key,
                n_samples=n_samples,
                batch_size=batch_size,
                store_samples=store_samples,
                counts=counts,
            )

        # Generate predictive samples using existing parameters
        _, key_pred = random.split(rng_key)

        self.get_predictive_samples(
            rng_key=key_pred,
            store_samples=store_samples,
            counts=counts,
        )

        return {
            "parameter_samples": self.posterior_samples,
            "predictive_samples": self.predictive_samples,
        }
