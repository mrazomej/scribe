"""
MAP and full-model predictive sampling mixin for SVI results.
"""

from typing import Optional

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random

from ..sampling import sample_posterior_ppc


class MapPredictiveSamplingMixin:
    """Mixin providing MAP and full-model predictive sampling methods."""

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

    def get_posterior_ppc_samples(
        self,
        gene_indices: Optional[jnp.ndarray] = None,
        n_samples: int = 500,
        cell_batch_size: int = 500,
        rng_key: Optional[random.PRNGKey] = None,
        counts: Optional[jnp.ndarray] = None,
        store_samples: bool = False,
        verbose: bool = True,
    ) -> jnp.ndarray:
        """Generate full-model posterior predictive samples for GoF evaluation.

        Draws PPC samples that include **all** model components (NB base,
        zero-inflation gate, capture probability, mixture assignments), using
        full posterior parameter draws rather than MAP point estimates.  This
        makes the resulting samples directly comparable to observed counts and
        suitable for PPC-based goodness-of-fit scoring.

        The method implements a *sample-once, predict-per-batch* strategy:
        posterior parameters are drawn (or reused) once, then for each gene
        batch the relevant parameter slices are passed to
        :func:`scribe.sampling.sample_posterior_ppc` for efficient direct
        distribution sampling via ``jax.vmap``.

        Parameters
        ----------
        gene_indices : jnp.ndarray or None, optional
            Integer indices of genes to generate PPC for.  When ``None`` all
            genes are included.  Providing a subset drastically reduces peak
            memory.
        n_samples : int, optional
            Number of posterior draws.  Ignored when posterior samples are
            already cached on ``self``.  Default: 500.
        cell_batch_size : int, optional
            Cells processed per batch inside the sampling helper.  Relevant
            mainly for VCP models where per-cell capture probability creates
            large intermediates.  Default: 500.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key.  Defaults to ``random.PRNGKey(42)``.
        counts : jnp.ndarray or None, optional
            Observed count matrix ``(n_cells, n_genes)`` needed for amortized
            capture-probability models.
        store_samples : bool, optional
            If ``True``, stores the result in ``self.predictive_samples``.
            Default: ``False``.
        verbose : bool, optional
            Print progress messages.  Default: ``True``.

        Returns
        -------
        jnp.ndarray
            PPC count samples with shape ``(S, n_cells, n_genes_batch)``
            where ``S`` is the number of posterior draws and
            ``n_genes_batch`` is ``len(gene_indices)`` (or total genes when
            ``gene_indices`` is ``None``).

        See Also
        --------
        get_ppc_samples_biological : Posterior PPC stripping technical noise.
        get_map_ppc_samples : MAP-based PPC including technical noise.
        scribe.sampling.sample_posterior_ppc : Core sampling helper.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # ---- 1. Draw or reuse posterior parameters ----
        if self.posterior_samples is None:
            key_post, rng_key = random.split(rng_key)
            if verbose:
                print(
                    f"Drawing {n_samples} posterior samples from the "
                    f"variational guide..."
                )
            self.get_posterior_samples(
                rng_key=key_post,
                n_samples=n_samples,
                store_samples=True,
                counts=counts,
            )

        # ---- 2. Extract parameters from posterior_samples ----
        r = self.posterior_samples["r"]   # (S, G) or (S, K, G)
        p = self.posterior_samples["p"]   # (S,) or (S, K)

        is_mixture = self.n_components is not None and self.n_components > 1
        has_gate = "gate" in self.posterior_samples
        has_vcp = "p_capture" in self.posterior_samples

        gate = self.posterior_samples.get("gate") if has_gate else None
        p_capture = (
            self.posterior_samples.get("p_capture") if has_vcp else None
        )
        mixing_weights = (
            self.posterior_samples.get("mixing_weights")
            if is_mixture
            else None
        )

        if verbose:
            model_desc = (
                f"mixture ({self.n_components} components)"
                if is_mixture
                else "standard"
            )
            extras = []
            if has_gate:
                extras.append("ZINB")
            if has_vcp:
                extras.append("VCP")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            print(
                f"Generating posterior PPC for {model_desc} model"
                f"{extra_str}..."
            )

        # ---- 3. Slice gene dimension if requested ----
        # Parameters that may carry a gene axis: r, p (hierarchical),
        # gate.  p_capture is cell-indexed, not gene-indexed.
        if gene_indices is not None:
            n_genes = r.shape[-1]
            if is_mixture:
                # r: (S, K, G) -> (S, K, G_batch)
                r = r[:, :, gene_indices]
                if gate is not None and gate.ndim == 3:
                    gate = gate[:, :, gene_indices]
                # p may be (S, K, G) for hierarchical mixtures
                if p.ndim == 3 and p.shape[-1] == n_genes:
                    p = p[:, :, gene_indices]
            else:
                # r: (S, G) -> (S, G_batch)
                r = r[:, gene_indices]
                if gate is not None and gate.ndim == 2:
                    gate = gate[:, gene_indices]
                # p may be (S, G) for hierarchical (per-gene p) models
                if p.ndim == 2 and p.shape[-1] == n_genes:
                    p = p[:, gene_indices]

        # ---- 4. Sample via the full-model helper ----
        _, key_ppc = random.split(rng_key)
        samples = sample_posterior_ppc(
            r=r,
            p=p,
            n_cells=self.n_cells,
            rng_key=key_ppc,
            gate=gate,
            p_capture=p_capture,
            mixing_weights=mixing_weights,
            cell_batch_size=cell_batch_size,
        )

        if verbose:
            print(
                f"Generated posterior PPC samples with shape {samples.shape}"
            )

        if store_samples:
            self.predictive_samples = samples

        return samples
