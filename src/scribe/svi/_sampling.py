"""
Sampling mixin for SVI results.

This mixin provides methods for posterior and predictive sampling, including
posterior predictive checks and MAP-based sampling.  It also exposes:

* **Biological PPC** methods that sample from the base Negative Binomial
  distribution only, stripping technical noise parameters (capture probability,
  zero-inflation gate) so the resulting counts reflect the latent biology.

* **Bayesian denoising** methods that take observed count matrices and
  posterior parameter estimates to compute the closed-form posterior of
  true (pre-capture, pre-dropout) transcript counts.
"""

from typing import Dict, Optional, Union
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist

from ..sampling import (
    sample_variational_posterior,
    generate_predictive_samples,
    sample_biological_nb,
    denoise_counts,
)

# ==============================================================================
# Sampling Mixin
# ==============================================================================


class SamplingMixin:
    """Mixin providing posterior and predictive sampling methods."""

    # --------------------------------------------------------------------------
    # Posterior sampling methods
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------

    def get_map_ppc_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 1,
        cell_batch_size: Optional[int] = None,
        use_mean: bool = True,
        store_samples: bool = True,
        verbose: bool = True,
        counts: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Generate predictive samples using MAP parameter estimates with cell
        batching.

        This method is memory-efficient for models with cell-specific parameters
        (like VCP models) because it:
            1. Uses MAP estimates directly instead of running the full guide
            2. Samples from observation distributions in cell batches
            3. Avoids materializing full (n_cells, n_genes) intermediate arrays

        Parameters
        ----------
        rng_key : random.PRNGKey, default=random.PRNGKey(42)
            JAX random number generator key
        n_samples : int, default=1
            Number of predictive samples to generate
        cell_batch_size : Optional[int], default=None
            Number of cells to process at once. If None, processes all cells
            at once (may cause OOM for VCP models with many cells).
        use_mean : bool, default=True
            If True, replaces undefined MAP values (NaN) with posterior means
        store_samples : bool, default=True
            If True, stores the samples in self.predictive_samples
        verbose : bool, default=True
            If True, prints progress messages
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
        jnp.ndarray
            Predictive samples with shape (n_samples, n_cells, n_genes)

        Notes
        -----
        This method is particularly useful for:
        - UMAP visualizations where only 1 sample is needed
        - Large datasets where full posterior sampling causes OOM
        - VCP models (nbvcp, zinbvcp) with cell-specific capture probabilities

        The method supports all model types:
        - nbdm, zinb: Standard negative binomial models
        - nbvcp, zinbvcp: Models with variable capture probability
        - *_mix variants: Mixture models
        """
        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        if verbose:
            print("Getting MAP estimates...")

        # Get MAP estimates with canonical parameters
        map_estimates = self.get_map(
            use_mean=use_mean, canonical=True, verbose=False, counts=counts
        )

        # Extract common parameters
        r = map_estimates.get("r")
        p = map_estimates.get("p")

        if r is None or p is None:
            raise ValueError(
                "Could not extract r and p from MAP estimates. "
                f"Available keys: {list(map_estimates.keys())}"
            )

        # Determine model characteristics
        is_mixture = self.n_components is not None and self.n_components > 1
        has_gate = "gate" in map_estimates
        has_vcp = "p_capture" in map_estimates

        if verbose:
            print(
                f"Model type: {self.model_type} "
                f"(mixture={is_mixture}, gate={has_gate}, vcp={has_vcp})"
            )

        # Get optional parameters
        gate = map_estimates.get("gate")
        p_capture = map_estimates.get("p_capture")
        mixing_weights = map_estimates.get("mixing_weights")

        # Determine dimensions
        if is_mixture:
            # r has shape (n_components, n_genes)
            n_genes = r.shape[1]
        else:
            # r has shape (n_genes,)
            n_genes = r.shape[0]

        # Use cell_batch_size or process all at once
        if cell_batch_size is None:
            cell_batch_size = self.n_cells

        # Generate samples
        if is_mixture:
            samples = self._sample_mixture_model(
                rng_key=rng_key,
                n_samples=n_samples,
                cell_batch_size=cell_batch_size,
                r=r,
                p=p,
                gate=gate,
                p_capture=p_capture,
                mixing_weights=mixing_weights,
                verbose=verbose,
            )
        else:
            samples = self._sample_standard_model(
                rng_key=rng_key,
                n_samples=n_samples,
                cell_batch_size=cell_batch_size,
                r=r,
                p=p,
                gate=gate,
                p_capture=p_capture,
                verbose=verbose,
            )

        if verbose:
            print(f"Generated predictive samples with shape {samples.shape}")

        # Store samples if requested
        if store_samples:
            self.predictive_samples = samples

        return samples

    # --------------------------------------------------------------------------

    def _sample_standard_model(
        self,
        rng_key: random.PRNGKey,
        n_samples: int,
        cell_batch_size: int,
        r: jnp.ndarray,
        p: jnp.ndarray,
        gate: Optional[jnp.ndarray],
        p_capture: Optional[jnp.ndarray],
        verbose: bool = True,
    ) -> jnp.ndarray:
        """Sample from standard (non-mixture) models with cell batching."""
        n_cells = self.n_cells
        n_genes = r.shape[0]
        has_vcp = p_capture is not None
        has_gate = gate is not None

        # Initialize output array
        all_samples = []

        n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * cell_batch_size
            end = min(start + cell_batch_size, n_cells)
            batch_size = end - start

            if verbose and n_batches > 1 and batch_idx % 10 == 0:
                print(f"  Processing cells {start}-{end} of {n_cells}...")

            # Split key for this batch
            rng_key, batch_key = random.split(rng_key)

            # Compute effective p for this batch
            if has_vcp:
                # Get capture probability for this batch of cells
                p_capture_batch = p_capture[start:end]  # (batch_size,)
                # Reshape for broadcasting: (batch_size, 1)
                p_capture_reshaped = p_capture_batch[:, None]
                # Compute p_hat: (batch_size, 1) broadcasts with p (scalar)
                p_effective = (
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
            else:
                # No VCP: p is the same for all cells
                p_effective = p

            # Create base NB distribution
            # r: (n_genes,), p_effective: scalar or (batch_size, 1)
            nb_dist = dist.NegativeBinomialProbs(r, p_effective)

            # Apply zero-inflation if present
            if has_gate:
                sample_dist = dist.ZeroInflatedDistribution(nb_dist, gate=gate)
            else:
                sample_dist = nb_dist

            # Sample counts for this batch
            # For VCP models, p_effective has shape (batch_size, 1) which gives
            # the distribution a batch dimension, so sample((n_samples,)) works.
            # For non-VCP models, p_effective is scalar, so we need to
            # explicitly include batch_size in the sample shape.
            if has_vcp:
                # Shape: (n_samples, batch_size, n_genes)
                batch_samples = sample_dist.sample(batch_key, (n_samples,))
            else:
                # Shape: (n_samples, batch_size, n_genes)
                batch_samples = sample_dist.sample(
                    batch_key, (n_samples, batch_size)
                )
            all_samples.append(batch_samples)

        # Concatenate all batches along cell dimension (axis=1)
        samples = jnp.concatenate(all_samples, axis=1)

        return samples

    # --------------------------------------------------------------------------

    def _sample_mixture_model(
        self,
        rng_key: random.PRNGKey,
        n_samples: int,
        cell_batch_size: int,
        r: jnp.ndarray,
        p: jnp.ndarray,
        gate: Optional[jnp.ndarray],
        p_capture: Optional[jnp.ndarray],
        mixing_weights: jnp.ndarray,
        verbose: bool = True,
    ) -> jnp.ndarray:
        """Sample from mixture models with cell batching."""
        n_cells = self.n_cells
        n_components = self.n_components
        n_genes = r.shape[1]  # r has shape (n_components, n_genes)
        has_vcp = p_capture is not None
        has_gate = gate is not None

        # Determine if p is component-specific and/or gene-specific.
        # Standard mixture: p is (n_components,) — component-specific only.
        # Hierarchical mixture: p is (n_components, n_genes) — both.
        p_is_component_specific = (
            p.ndim >= 1 and p.shape[0] == n_components
        )
        p_is_gene_specific = p.ndim == 2 and p.shape[1] == n_genes

        # Initialize output array
        all_samples = []

        n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * cell_batch_size
            end = min(start + cell_batch_size, n_cells)
            batch_size = end - start

            if verbose and n_batches > 1 and batch_idx % 10 == 0:
                print(f"  Processing cells {start}-{end} of {n_cells}...")

            # Split keys for component sampling and count sampling
            rng_key, component_key, sample_key = random.split(rng_key, 3)

            # Sample component assignments for each cell in batch
            # Shape: (n_samples, batch_size)
            components = dist.Categorical(probs=mixing_weights).sample(
                component_key, (n_samples, batch_size)
            )

            # Get parameters for assigned components
            # r: (n_components, n_genes) -> (n_samples, batch_size, n_genes)
            r_batch = r[components]

            # Handle p parameter
            if p_is_component_specific:
                # p[components] works for both (n_components,) and
                # (n_components, n_genes):
                #   (n_components,)        → (n_samples, batch_size)
                #   (n_components, n_genes) → (n_samples, batch_size, n_genes)
                p_batch = p[components]
            else:
                p_batch = p

            # Handle gate parameter if present.
            # gate is (n_components, n_genes) when in mixture_params,
            # otherwise (n_genes,) shared across components.
            if has_gate:
                if gate.ndim == 2 and gate.shape[0] == n_components:
                    gate_batch = gate[components]
                else:
                    gate_batch = gate
            else:
                gate_batch = None

            # Handle VCP
            if has_vcp:
                # Get capture probability for this batch
                p_capture_batch = p_capture[start:end]  # (batch_size,)
                # Expand for (n_samples, batch_size, 1)
                p_capture_expanded = p_capture_batch[None, :, None]

                # Reshape p_batch for broadcasting with p_capture_expanded
                if p_is_gene_specific:
                    # Already (n_samples, batch_size, n_genes) — broadcasts
                    # with (1, batch_size, 1) directly.
                    p_expanded = p_batch
                elif p_is_component_specific:
                    # (n_samples, batch_size) → (n_samples, batch_size, 1)
                    p_expanded = p_batch[:, :, None]
                else:
                    p_expanded = p_batch

                # Compute p_hat
                p_effective = (
                    p_expanded
                    * p_capture_expanded
                    / (1 - p_expanded * (1 - p_capture_expanded))
                )
            else:
                if p_is_gene_specific:
                    # Already (n_samples, batch_size, n_genes)
                    p_effective = p_batch
                elif p_is_component_specific:
                    # (n_samples, batch_size) → (n_samples, batch_size, 1)
                    p_effective = p_batch[:, :, None]
                else:
                    p_effective = p_batch

            # Create NB distribution with batch parameters
            nb_dist = dist.NegativeBinomialProbs(r_batch, p_effective)

            # Apply zero-inflation if present
            if gate_batch is not None:
                sample_dist = dist.ZeroInflatedDistribution(
                    nb_dist, gate=gate_batch
                )
            else:
                sample_dist = nb_dist

            # Sample counts
            # We already have the right shape from indexing, just sample
            batch_samples = sample_dist.sample(sample_key)
            all_samples.append(batch_samples)

        # Concatenate all batches along cell dimension (axis=1)
        samples = jnp.concatenate(all_samples, axis=1)

        return samples

    # ==========================================================================
    # Biological (denoised) PPC methods
    # ==========================================================================

    def get_ppc_samples_biological(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        counts: Optional[jnp.ndarray] = None,
    ) -> Dict:
        """Generate biological posterior predictive check samples.

        Samples from the base Negative Binomial distribution NB(r, p) only,
        stripping all technical noise parameters.  For VCP models this
        removes the cell-specific capture probability (``p_capture`` /
        ``phi_capture``).  For ZINB variants this additionally removes the
        zero-inflation gate.  For NBDM the result is identical to
        :meth:`get_ppc_samples`.

        The method reuses existing posterior samples when available,
        extracting only the biological parameters (``r``, ``p``, and
        ``mixing_weights`` for mixture models) before sampling from the
        clean NB distribution via :func:`scribe.sampling.sample_biological_nb`.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key.  Defaults to
            ``random.PRNGKey(42)`` when ``None``.
        n_samples : int, optional
            Number of posterior samples to draw from the variational guide.
            Only used when ``self.posterior_samples`` is ``None``.
            Default: 100.
        batch_size : Optional[int], optional
            Batch size for posterior sampling (passed to
            :meth:`get_posterior_samples`).  This is *not* the cell batch
            size for count generation – use ``cell_batch_size`` in
            :meth:`get_map_ppc_samples_biological` for that purpose.
            Default: ``None``.
        store_samples : bool, optional
            If ``True``, stores the generated predictive samples in
            ``self.predictive_samples_biological``.  Default: ``True``.
        counts : Optional[jnp.ndarray], optional
            Observed count matrix ``(n_cells, n_genes)``.  Required when
            using amortized capture probability so the guide can compute
            sufficient statistics.  For non-amortized models this can be
            ``None``.

        Returns
        -------
        Dict
            Dictionary with keys:

            - ``'parameter_samples'``: Full posterior samples (including
              technical parameters).
            - ``'predictive_samples'``: Biological NB samples with shape
              ``(n_posterior_samples, n_cells, n_genes)``.

        See Also
        --------
        get_ppc_samples : Standard PPC that includes technical noise.
        get_map_ppc_samples_biological : MAP-based biological PPC.
        scribe.sampling.sample_biological_nb : Core sampling utility.

        Notes
        -----
        The biological PPC is motivated by the Dirichlet-Multinomial
        derivation: the composition of NB with a Binomial capture step
        yields another NB with an effective :math:`\\hat{p}`.  By sampling
        from NB(r, p) directly we recover the pre-capture distribution.
        """
        # Create default RNG key if not provided
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Ensure we have posterior samples
        if self.posterior_samples is None:
            key_post, rng_key = random.split(rng_key)
            self.get_posterior_samples(
                rng_key=key_post,
                n_samples=n_samples,
                batch_size=batch_size,
                store_samples=True,
                counts=counts,
            )

        # Extract only the biological parameters from posterior samples
        r = self.posterior_samples["r"]
        p = self.posterior_samples["p"]
        mixing_weights = self.posterior_samples.get("mixing_weights", None)

        # Generate biological (denoised) count samples
        _, key_bio = random.split(rng_key)
        bio_samples = sample_biological_nb(
            r=r,
            p=p,
            n_cells=self.n_cells,
            rng_key=key_bio,
            mixing_weights=mixing_weights,
        )

        if store_samples:
            self.predictive_samples_biological = bio_samples

        return {
            "parameter_samples": self.posterior_samples,
            "predictive_samples": bio_samples,
        }

    # --------------------------------------------------------------------------

    def get_map_ppc_samples_biological(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 1,
        cell_batch_size: Optional[int] = None,
        use_mean: bool = True,
        store_samples: bool = True,
        verbose: bool = True,
        counts: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Generate biological predictive samples from MAP estimates.

        Like :meth:`get_map_ppc_samples` but strips technical noise
        parameters.  Uses MAP (or posterior-mean) point estimates for ``r``
        and ``p`` (and ``mixing_weights`` for mixture models) and samples
        directly from NB(r, p), bypassing capture probability and zero-
        inflation.

        This method is memory-efficient because it processes cells in
        configurable batches and avoids materialising full
        ``(n_cells, n_genes)`` intermediate arrays for technical parameters.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key.  Defaults to ``random.PRNGKey(42)``.
        n_samples : int, optional
            Number of predictive draws to generate.  Default: 1.
        cell_batch_size : int or None, optional
            Cells processed per batch.  ``None`` processes all cells at
            once (may OOM for very large datasets).
        use_mean : bool, optional
            If ``True``, replaces undefined MAP values (NaN) with posterior
            means.  Default: ``True``.
        store_samples : bool, optional
            If ``True``, stores the result in
            ``self.predictive_samples_biological``.  Default: ``True``.
        verbose : bool, optional
            Print progress messages.  Default: ``True``.
        counts : Optional[jnp.ndarray], optional
            Observed count matrix for amortized-capture models.

        Returns
        -------
        jnp.ndarray
            Biological count samples with shape
            ``(n_samples, n_cells, n_genes)``.

        See Also
        --------
        get_map_ppc_samples : MAP PPC including technical noise.
        get_ppc_samples_biological : Full-posterior biological PPC.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        if verbose:
            print("Getting MAP estimates for biological PPC...")

        # Retrieve MAP estimates in canonical (p, r) form
        map_estimates = self.get_map(
            use_mean=use_mean, canonical=True, verbose=False, counts=counts
        )

        r = map_estimates.get("r")
        p = map_estimates.get("p")
        if r is None or p is None:
            raise ValueError(
                "Could not extract r and p from MAP estimates. "
                f"Available keys: {list(map_estimates.keys())}"
            )

        # For mixture models, also grab mixing_weights
        is_mixture = self.n_components is not None and self.n_components > 1
        mixing_weights = map_estimates.get("mixing_weights") if is_mixture else None

        if verbose:
            model_desc = (
                f"mixture ({self.n_components} components)"
                if is_mixture
                else "standard"
            )
            print(
                f"Sampling biological NB for {model_desc} model "
                f"({self.model_type})..."
            )

        # Sample from the base NB(r, p) only – no capture, no gate
        samples = sample_biological_nb(
            r=r,
            p=p,
            n_cells=self.n_cells,
            rng_key=rng_key,
            n_samples=n_samples,
            mixing_weights=mixing_weights,
            cell_batch_size=cell_batch_size,
        )

        if verbose:
            print(
                f"Generated biological predictive samples with shape "
                f"{samples.shape}"
            )

        if store_samples:
            self.predictive_samples_biological = samples

        return samples

    # --------------------------------------------------------------------------
    # Bayesian denoising of observed counts
    # --------------------------------------------------------------------------

    def denoise_counts_map(
        self,
        counts: jnp.ndarray,
        method: str = "mean",
        rng_key: Optional[random.PRNGKey] = None,
        return_variance: bool = False,
        cell_batch_size: Optional[int] = None,
        use_mean: bool = True,
        store_result: bool = True,
        verbose: bool = True,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Denoise observed counts using MAP parameter estimates.

        Computes the posterior of true (pre-capture, pre-dropout)
        transcript counts for each cell and gene, using point estimates
        of the model parameters.  For VCP models this accounts for the
        per-cell capture probability; for ZINB variants it additionally
        corrects zero observations for dropout.  For NBDM the result is
        the identity (``denoised == counts``).

        Parameters
        ----------
        counts : jnp.ndarray
            Observed UMI count matrix of shape ``(n_cells, n_genes)``.
        method : {'mean', 'mode', 'sample'}, optional
            Summary of the denoised posterior to return.

            * ``'mean'``: closed-form posterior mean (shrinkage estimator).
            * ``'mode'``: posterior mode (MAP denoised count).
            * ``'sample'``: one stochastic draw per cell/gene.

            Default: ``'mean'``.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key.  Required when ``method='sample'``.
            Defaults to ``random.PRNGKey(42)`` when ``None``.
        return_variance : bool, optional
            If ``True``, return a dictionary with ``'denoised_counts'``
            and ``'variance'`` keys.  Default: ``False``.
        cell_batch_size : int or None, optional
            Process cells in batches of this size to limit memory.
            ``None`` processes all cells at once.
        use_mean : bool, optional
            If ``True``, replaces undefined MAP values (NaN) with
            posterior means.  Default: ``True``.
        store_result : bool, optional
            If ``True``, stores the denoised counts in
            ``self.denoised_counts``.  Default: ``True``.
        verbose : bool, optional
            Print progress messages.  Default: ``True``.

        Returns
        -------
        jnp.ndarray or Dict[str, jnp.ndarray]
            Denoised count matrix of shape ``(n_cells, n_genes)`` (or
            dict with variance when ``return_variance=True``).

        See Also
        --------
        denoise_counts_posterior : Full-posterior Bayesian denoising.
        get_map_ppc_samples_biological : MAP-based biological PPC.
        scribe.sampling.denoise_counts : Core denoising utility.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        if verbose:
            print("Getting MAP estimates for denoising...")

        map_estimates = self.get_map(
            use_mean=use_mean, canonical=True, verbose=False,
            counts=counts,
        )

        r = map_estimates.get("r")
        p = map_estimates.get("p")
        if r is None or p is None:
            raise ValueError(
                "Could not extract r and p from MAP estimates. "
                f"Available keys: {list(map_estimates.keys())}"
            )

        p_capture = map_estimates.get("p_capture")
        gate = map_estimates.get("gate")
        is_mixture = self.n_components is not None and self.n_components > 1
        mixing_weights = (
            map_estimates.get("mixing_weights") if is_mixture else None
        )

        if verbose:
            model_desc = (
                f"mixture ({self.n_components} components)"
                if is_mixture
                else "standard"
            )
            extras = []
            if p_capture is not None:
                extras.append("VCP")
            if gate is not None:
                extras.append("gate")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            print(
                f"Denoising {model_desc} model "
                f"({self.model_type}){extra_str}, method='{method}'..."
            )

        result = denoise_counts(
            counts=counts,
            r=r,
            p=p,
            p_capture=p_capture,
            gate=gate,
            method=method,
            rng_key=rng_key,
            return_variance=return_variance,
            mixing_weights=mixing_weights,
            cell_batch_size=cell_batch_size,
        )

        if verbose:
            shape = (
                result["denoised_counts"].shape
                if return_variance
                else result.shape
            )
            print(f"Denoised counts shape: {shape}")

        if store_result:
            self.denoised_counts = (
                result["denoised_counts"] if return_variance else result
            )

        return result

    # --------------------------------------------------------------------------

    def denoise_counts_posterior(
        self,
        counts: jnp.ndarray,
        method: str = "mean",
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        return_variance: bool = False,
        cell_batch_size: Optional[int] = None,
        store_result: bool = True,
        verbose: bool = True,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Denoise observed counts using full posterior samples.

        Propagates parameter uncertainty by repeating the denoising
        computation for each draw from the variational posterior.  The
        result has a leading ``n_samples`` dimension that can be
        summarised (e.g. ``.mean(axis=0)`` for a fully Bayesian point
        estimate, or quantiles for credible intervals).

        Parameters
        ----------
        counts : jnp.ndarray
            Observed UMI count matrix ``(n_cells, n_genes)``.
        method : {'mean', 'mode', 'sample'}, optional
            Summary of the per-sample denoised posterior.
            Default: ``'mean'``.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key.  Defaults to ``random.PRNGKey(42)``.
        n_samples : int, optional
            Number of posterior samples to draw from the guide if
            ``self.posterior_samples`` is ``None``.  Default: 100.
        batch_size : int or None, optional
            Batch size for posterior sampling (passed to
            :meth:`get_posterior_samples`).
        return_variance : bool, optional
            If ``True``, return dict with ``'denoised_counts'`` and
            ``'variance'``.  Default: ``False``.
        cell_batch_size : int or None, optional
            Cell batching inside each posterior draw.
        store_result : bool, optional
            Store result in ``self.denoised_counts``.  Default: ``True``.
        verbose : bool, optional
            Print progress messages.  Default: ``True``.

        Returns
        -------
        jnp.ndarray or Dict[str, jnp.ndarray]
            Denoised counts with shape ``(n_samples, n_cells, n_genes)``
            (or dict with variance when ``return_variance=True``).

        See Also
        --------
        denoise_counts_map : MAP-based denoising (single point estimate).
        get_ppc_samples_biological : Full-posterior biological PPC.
        scribe.sampling.denoise_counts : Core denoising utility.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Ensure posterior samples exist
        if self.posterior_samples is None:
            key_post, rng_key = random.split(rng_key)
            if verbose:
                print("Drawing posterior samples...")
            self.get_posterior_samples(
                rng_key=key_post,
                n_samples=n_samples,
                batch_size=batch_size,
                store_samples=True,
                counts=counts,
            )

        r = self.posterior_samples["r"]
        p = self.posterior_samples["p"]
        p_capture = self.posterior_samples.get("p_capture")
        gate = self.posterior_samples.get("gate")
        is_mixture = self.n_components is not None and self.n_components > 1
        mixing_weights = (
            self.posterior_samples.get("mixing_weights")
            if is_mixture
            else None
        )

        if verbose:
            extras = []
            if p_capture is not None:
                extras.append("VCP")
            if gate is not None:
                extras.append("gate")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            n_post = r.shape[0]
            print(
                f"Denoising with {n_post} posterior samples"
                f" ({self.model_type}){extra_str}, method='{method}'..."
            )

        _, key_denoise = random.split(rng_key)
        result = denoise_counts(
            counts=counts,
            r=r,
            p=p,
            p_capture=p_capture,
            gate=gate,
            method=method,
            rng_key=key_denoise,
            return_variance=return_variance,
            mixing_weights=mixing_weights,
            cell_batch_size=cell_batch_size,
        )

        if verbose:
            shape = (
                result["denoised_counts"].shape
                if return_variance
                else result.shape
            )
            print(f"Denoised counts shape: {shape}")

        if store_result:
            self.denoised_counts = (
                result["denoised_counts"] if return_variance else result
            )

        return result
