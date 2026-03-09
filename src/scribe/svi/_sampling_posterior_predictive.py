"""Posterior and constrained predictive sampling mixin for SVI results."""

from typing import Any, Dict, List, Optional

import jax.numpy as jnp
from jax import random

from ..sampling import generate_predictive_samples, sample_variational_posterior
from ._dataset import _build_cell_specific_keys


class PosteriorPredictiveSamplingMixin:
    """Mixin providing posterior and constrained predictive sampling methods."""

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        """Compare nested values with explicit array support.

        Parameters
        ----------
        left : Any
            Left value to compare.
        right : Any
            Right value to compare.

        Returns
        -------
        bool
            ``True`` when both values are equal under recursive
            array-aware comparison; otherwise ``False``.
        """
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            if set(left.keys()) != set(right.keys()):
                return False
            return all(
                PosteriorPredictiveSamplingMixin._values_equal(
                    left[k], right[k]
                )
                for k in left
            )
        if hasattr(left, "shape") and hasattr(right, "shape"):
            return bool(jnp.array_equal(jnp.asarray(left), jnp.asarray(right)))
        return left == right

    def _is_promoted_concat_for_sampling(self) -> bool:
        """Check whether this object is a concat-promoted multi-dataset view.

        Returns
        -------
        bool
            ``True`` when the object has synthetic dataset metadata created by
            ``ScribeSVIResults.concat`` promotion and should use per-dataset
            posterior sampling fallback.
        """
        promoted = getattr(self, "_promoted_dataset_keys", None)
        n_datasets = getattr(self.model_config, "n_datasets", None)
        ds_indices = getattr(self, "_dataset_indices", None)
        per_ds = getattr(self, "_n_cells_per_dataset", None)
        return (
            promoted is not None
            and len(promoted) > 0
            and n_datasets is not None
            and ds_indices is not None
            and per_ds is not None
        )

    def _merge_promoted_dataset_posterior_samples(
        self,
        per_dataset_samples: List[Dict[str, Any]],
        *,
        n_datasets: int,
        n_cells_per_dataset: jnp.ndarray,
    ) -> Dict[str, Any]:
        """Merge per-dataset posterior sample dicts into promoted layout.

        Parameters
        ----------
        per_dataset_samples : list of dict
            Posterior sample dictionaries produced by each
            ``get_dataset(i).get_posterior_samples(...)`` call.
        n_datasets : int
            Number of promoted datasets.
        n_cells_per_dataset : jnp.ndarray
            Per-dataset cell counts in promoted order.

        Returns
        -------
        Dict[str, Any]
            Merged posterior samples where:
            - cell-specific keys are concatenated on axis 1,
            - non-cell keys are stacked on synthetic dataset axis 1,
            - non-array metadata must agree across datasets.

        Raises
        ------
        ValueError
            If sample dictionaries have incompatible keys or inconsistent
            sample/cell dimensions that prevent safe merging.
        """
        if len(per_dataset_samples) != n_datasets:
            raise ValueError(
                "Internal error: per-dataset sample list length does not match "
                f"n_datasets ({len(per_dataset_samples)} vs {n_datasets})."
            )
        if not per_dataset_samples:
            return {}

        # Use ParamSpec metadata to identify cell-specific posterior sites.
        cell_specific_keys = _build_cell_specific_keys(
            self.model_config.param_specs or [],
            per_dataset_samples[0],
        )

        keys = set(per_dataset_samples[0].keys())
        for idx, sample_dict in enumerate(per_dataset_samples[1:], start=1):
            if set(sample_dict.keys()) != keys:
                raise ValueError(
                    "Per-dataset posterior sample keys do not match across "
                    f"datasets (mismatch at dataset index {idx})."
                )

        merged: Dict[str, Any] = {}
        stacked_non_cell_keys = set()
        for key in keys:
            values = [sample_dict[key] for sample_dict in per_dataset_samples]

            # Keep nested metadata structures only when identical.
            if not all(hasattr(v, "ndim") for v in values):
                first = values[0]
                if not all(self._values_equal(first, v) for v in values[1:]):
                    raise ValueError(
                        "Cannot merge non-array posterior site "
                        f"'{key}': values differ across datasets."
                    )
                merged[key] = first
                continue

            sample_sizes = [int(v.shape[0]) for v in values]
            if len(set(sample_sizes)) != 1:
                raise ValueError(
                    "Sample-axis mismatch while merging promoted posterior "
                    f"site '{key}': {sample_sizes}"
                )

            if key in cell_specific_keys:
                # Cell-specific posterior tensors are (S, C, ...); concatenate
                # cells so the combined result preserves global cell ordering.
                for idx, val in enumerate(values):
                    if val.ndim < 2:
                        raise ValueError(
                            f"Cell-specific posterior site '{key}' must have "
                            f"ndim >= 2; found ndim={val.ndim} at dataset {idx}."
                        )
                    expected_cells = int(n_cells_per_dataset[idx])
                    observed_cells = int(val.shape[1])
                    if observed_cells != expected_cells:
                        raise ValueError(
                            f"Cell-axis mismatch for site '{key}' at dataset {idx}: "
                            f"expected {expected_cells}, found {observed_cells}."
                        )
                merged[key] = jnp.concatenate(values, axis=1)
                continue

            # Non-cell posterior tensors are promoted to a dataset axis so
            # get_dataset(i) can recover per-dataset posterior values.
            if values[0].ndim >= 1:
                try:
                    merged[key] = jnp.stack(values, axis=1)
                    stacked_non_cell_keys.add(key)
                    continue
                except ValueError as exc:
                    raise ValueError(
                        "Cannot stack promoted posterior site "
                        f"'{key}' across datasets: {exc}"
                    ) from exc

            # Scalar arrays cannot carry a dataset axis. Keep only if equal.
            first = values[0]
            if not all(self._values_equal(first, v) for v in values[1:]):
                raise ValueError(
                    f"Cannot merge scalar posterior site '{key}': values differ."
                )
            merged[key] = first

        # Persist any newly promoted posterior keys so future get_dataset()
        # calls can slice them correctly from combined posterior_samples.
        existing_promoted = getattr(self, "_promoted_dataset_keys", None) or set()
        self._promoted_dataset_keys = set(existing_promoted).union(
            stacked_non_cell_keys
        )
        return merged

    def _get_posterior_samples_promoted_concat(
        self,
        *,
        rng_key: random.PRNGKey,
        n_samples: int,
        batch_size: Optional[int],
        counts: Optional[jnp.ndarray],
    ) -> Dict[str, Any]:
        """Sample posterior per dataset then merge into promoted layout.

        Parameters
        ----------
        rng_key : random.PRNGKey
            Base PRNG key used to derive deterministic per-dataset keys.
        n_samples : int
            Number of posterior draws per dataset.
        batch_size : int, optional
            Optional mini-batch size for variational posterior sampling.
        counts : jnp.ndarray, optional
            Observed count matrix of shape ``(n_cells, n_genes)``. When
            provided, rows are subset using ``_dataset_indices`` and passed
            to each dataset view.

        Returns
        -------
        Dict[str, Any]
            Merged posterior sample dictionary compatible with promoted
            concat objects.
        """
        n_datasets = int(getattr(self.model_config, "n_datasets"))
        n_cells_per_dataset = jnp.asarray(getattr(self, "_n_cells_per_dataset"))
        dataset_indices = jnp.asarray(getattr(self, "_dataset_indices"))

        if counts is not None and int(counts.shape[0]) != int(self.n_cells):
            raise ValueError(
                "counts has incompatible cell axis for promoted concat sampling: "
                f"expected n_cells={self.n_cells}, found {counts.shape[0]}."
            )

        # Split once and map each dataset to a deterministic subkey.
        keys = random.split(rng_key, n_datasets)
        per_dataset_samples: List[Dict[str, Any]] = []
        for dataset_index in range(n_datasets):
            dataset_view = self.get_dataset(dataset_index)
            dataset_counts = None
            if counts is not None:
                dataset_mask = dataset_indices == dataset_index
                dataset_counts = counts[dataset_mask]

            samples_ds = dataset_view.get_posterior_samples(
                rng_key=keys[dataset_index],
                n_samples=n_samples,
                batch_size=batch_size,
                store_samples=False,
                counts=dataset_counts,
            )
            per_dataset_samples.append(samples_ds)

        return self._merge_promoted_dataset_posterior_samples(
            per_dataset_samples=per_dataset_samples,
            n_datasets=n_datasets,
            n_cells_per_dataset=n_cells_per_dataset,
        )

    def get_posterior_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        counts: Optional[jnp.ndarray] = None,
    ) -> Dict:
        """Sample parameters from the variational posterior distribution.

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
            self._validate_counts_for_amortizer(counts, context="posterior sampling")

        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Concatenating independently fit single-dataset results promotes a
        # synthetic dataset axis for parameter recovery. Sampling each dataset
        # view independently avoids guide/model shape mismatches at the combined
        # level and reassembles a posterior dictionary with promoted axes.
        if self._is_promoted_concat_for_sampling():
            posterior_samples = self._get_posterior_samples_promoted_concat(
                rng_key=rng_key,
                n_samples=n_samples,
                batch_size=batch_size,
                counts=counts,
            )
            if store_samples:
                self.posterior_samples = posterior_samples
            return posterior_samples

        # Get the guide function
        model, guide = self._model_and_guide()

        if guide is None:
            raise ValueError(
                f"Could not find a guide for model '{self.model_type}'."
            )

        # Prepare base model arguments
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "model_config": self.model_config,
        }

        # Add batch_size to model_args if provided for memory-efficient sampling
        if batch_size is not None:
            model_args["batch_size"] = batch_size

        # Sample from posterior
        posterior_samples = sample_variational_posterior(
            guide,
            self.params,
            model,
            model_args,
            rng_key=rng_key,
            n_samples=n_samples,
            counts=counts,
        )

        # Store samples if requested
        if store_samples:
            self.posterior_samples = posterior_samples

        return posterior_samples

    def get_predictive_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples."""
        from ..models.model_registry import get_model_and_guide

        # For predictive sampling, we need the *constrained* model, which has
        # the 'counts' sample site. The posterior samples from the unconstrained
        # guide can be used with the constrained model.
        model, _, _ = get_model_and_guide(
            self.model_config,
            unconstrained=False,  # Explicitly get the constrained model
            guide_families=None,  # Not relevant for the model (only guide)
        )

        # Prepare base model arguments
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "model_config": self.model_config,
        }

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
