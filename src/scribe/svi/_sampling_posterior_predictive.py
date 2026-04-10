"""
Posterior and constrained predictive sampling mixin for SVI results.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import jax.numpy as jnp
from jax import random

from ..sampling import generate_predictive_samples, sample_variational_posterior


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
) -> Dict[str, Any]:
    """Slice the gene dimension of posterior/predictive sample arrays.

    For every array in *samples* whose shape contains *original_n_genes*,
    index the first matching axis with *gene_index*.  Non-matching arrays
    are passed through unchanged.
    """
    out: Dict[str, Any] = {}
    for key, value in samples.items():
        if not hasattr(value, "shape"):
            out[key] = value
            continue
        if original_n_genes in value.shape:
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


def _move_to_cpu(samples: Dict[str, Any]) -> Dict[str, Any]:
    """Transfer all JAX arrays in a posterior-samples dict to CPU device.

    Each value that is a ``jax.Array`` is placed on the first CPU device
    via ``jax.device_put``.  The returned arrays are still ``jax.Array``
    instances (not plain numpy), so downstream code using ``jnp``,
    ``vmap``, or NumPyro continues to work transparently.  Non-array
    entries (e.g., metadata) are passed through unchanged.

    Parameters
    ----------
    samples : Dict[str, Any]
        Posterior-samples dict as returned by NumPyro ``Predictive``.

    Returns
    -------
    Dict[str, Any]
        Same dict with every JAX array moved to CPU host memory.
    """
    import jax

    cpu = jax.devices("cpu")[0]
    return {
        k: jax.device_put(v, cpu) if isinstance(v, jax.Array) else v
        for k, v in samples.items()
    }


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
        store_on_cpu: bool = False,
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
            Batch size for memory-efficient sampling (default: None)
        store_samples : bool, optional
            Whether to store samples in self.posterior_samples (default: True)
        store_on_cpu : bool, optional
            If True, transfer all sampled arrays to CPU-resident JAX arrays
            via ``jax.device_put`` before storing and returning.  The arrays
            remain ``jax.Array`` instances (not plain numpy) so downstream
            code that uses ``jnp``, ``vmap``, or NumPyro continues to work
            transparently.  This frees GPU memory immediately after
            sampling, which is important when downstream operations (e.g.,
            differential expression) need GPU headroom.  Implies
            ``store_samples=True``.  Default: False.
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

        # Concatenated multi-dataset results: Predictive cannot run
        # on the stacked params because the guide was built for
        # single-dataset models.  Decompose into per-dataset sampling.
        if self._is_concatenated_multi_dataset():
            posterior_samples = self._get_posterior_samples_per_dataset(
                rng_key=rng_key,
                n_samples=n_samples,
                batch_size=batch_size,
                counts=counts,
            )
        else:
            posterior_samples = self._get_posterior_samples_standard(
                rng_key=rng_key,
                n_samples=n_samples,
                batch_size=batch_size,
                counts=counts,
            )

        # store_on_cpu implies store_samples — if the caller explicitly
        # asks for CPU-resident storage, the intent is always to persist.
        if store_on_cpu:
            store_samples = True
            posterior_samples = _move_to_cpu(posterior_samples)

        if store_samples:
            self.posterior_samples = posterior_samples

        from ..models.config.parameter_mapping import rename_dict_keys

        return rename_dict_keys(posterior_samples, descriptive_names)

    def _get_posterior_samples_standard(
        self,
        rng_key: random.PRNGKey,
        n_samples: int,
        batch_size: Optional[int],
        counts: Optional[jnp.ndarray],
    ) -> Dict:
        """Standard posterior sampling via NumPyro Predictive."""
        model, guide = self._model_and_guide()

        if guide is None:
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
        n_genes_for_guide = _orig_ng if _full_dim else self.n_genes
        params_for_guide = (
            _orig_params if (_full_dim and _orig_params is not None)
            else self.params
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

        # Add batch_size to model_args if provided for memory-efficient sampling
        if batch_size is not None:
            model_args["batch_size"] = batch_size

        posterior_samples = sample_variational_posterior(
            guide,
            params_for_guide,
            model,
            model_args,
            rng_key=rng_key,
            n_samples=n_samples,
            counts=counts,
        )

        # Slice full-dim flow samples down to the gene subset
        if _full_dim:
            posterior_samples = _subset_gene_dim_samples(
                posterior_samples, _gene_idx, _orig_ng,
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
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples."""
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
        )

        # Use model_config_for_pred (which has param_specs populated) so
        # index_dataset_params in the likelihood can correctly identify
        # mixture+dataset params.  Include dataset_indices so multi-dataset
        # params are indexed to per-cell layout.
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "model_config": model_config_for_pred,
        }
        ds_idx = getattr(self, "_dataset_indices", None)
        if ds_idx is not None:
            model_args["dataset_indices"] = ds_idx

        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Generate predictive samples
        predictive_samples = generate_predictive_samples(
            model,
            self.posterior_samples,
            model_args,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        # Store samples if requested
        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

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
            batch_size=batch_size,
            store_samples=store_samples,
        )

        return {
            "parameter_samples": self.posterior_samples,
            "predictive_samples": self.predictive_samples,
        }
