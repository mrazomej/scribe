"""
MAP and full-model predictive sampling mixin for SVI results.
"""

from typing import Optional

import jax.numpy as jnp
from jax import random

from ..sampling import (
    sample_posterior_ppc,
    _slice_gene_axis,
    _build_canonical_layouts,
)


def _ensure_map_arrays(
    r: jnp.ndarray,
    p: jnp.ndarray,
    gate: Optional[jnp.ndarray],
    p_capture: Optional[jnp.ndarray],
    mixing_weights: Optional[jnp.ndarray] = None,
) -> tuple:
    """Convert MAP estimate tensors to JAX arrays.

    A minimal helper that replaces the old ``_normalize_map_standard_inputs``,
    ``_normalize_map_mixture_inputs``, and ``_coerce_map_capture_vector``.
    All semantic shape interpretation is now handled by ``AxisLayout`` flags
    propagated into the sampling / denoising functions, so no squeeze / ndim
    coercion is necessary here.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion from MAP estimates.
    p : jnp.ndarray
        Success probability from MAP estimates.
    gate : jnp.ndarray or None
        Optional zero-inflation gate.
    p_capture : jnp.ndarray or None
        Optional per-cell capture probability.
    mixing_weights : jnp.ndarray or None
        Optional component mixing weights.

    Returns
    -------
    tuple
        ``(r, p, gate, p_capture, mixing_weights)`` — each converted
        to a JAX array (or ``None`` left as ``None``).
    """
    r = jnp.asarray(r)
    p = jnp.asarray(p)
    if gate is not None:
        gate = jnp.asarray(gate)
    if p_capture is not None:
        p_capture = jnp.asarray(p_capture)
    if mixing_weights is not None:
        mixing_weights = jnp.asarray(mixing_weights)
    return r, p, gate, p_capture, mixing_weights


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

        # Use cell_batch_size or process all at once
        if cell_batch_size is None:
            cell_batch_size = self.n_cells

        # Ensure all MAP tensors are JAX arrays.  Semantic shape
        # interpretation is now handled by AxisLayout flags inside the
        # sampling functions, so no squeeze/coerce normalization is needed.
        r, p, gate, p_capture, mixing_weights = _ensure_map_arrays(
            r, p, gate, p_capture, mixing_weights
        )

        # Build MAP-level canonical layouts (keyed by "r", "p", etc.)
        # for the canonical parameter dict.
        _map_layouts = _build_canonical_layouts(
            map_estimates,
            self.model_config,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            n_components=self.n_components,
            has_sample_dim=False,
        )

        # Infer n_genes from layout (gene axis is the last semantic axis
        # for r); the layout knows where it is.
        _r_layout = _map_layouts.get("r")
        if _r_layout is not None and _r_layout.gene_axis is not None:
            n_genes = int(r.shape[_r_layout.gene_axis])
        else:
            # Fallback: last axis is genes for both standard (G,) and
            # mixture (K, G) models.
            n_genes = int(r.shape[-1])

        # Generate MAP PPC samples through the full-model helper.
        # MAP estimates have no sample dim; pass canonical layouts.
        samples = sample_posterior_ppc(
            r=r,
            p=p,
            n_cells=self.n_cells,
            rng_key=rng_key,
            n_samples=n_samples,
            gate=gate,
            p_capture=p_capture,
            mixing_weights=mixing_weights if is_mixture else None,
            cell_batch_size=cell_batch_size,
            bnb_concentration=map_estimates.get("bnb_concentration"),
            param_layouts=_map_layouts,
        )

        if verbose:
            print(
                "Generated predictive samples with shape "
                f"{samples.shape} for n_genes={n_genes}"
            )

        # Store samples if requested
        if store_samples:
            self.predictive_samples = samples

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
        r = self.posterior_samples["r"]  # (S, G) or (S, K, G)
        p = self.posterior_samples["p"]  # (S,) or (S, K)

        is_mixture = self.n_components is not None and self.n_components > 1
        has_gate = "gate" in self.posterior_samples
        has_vcp = "p_capture" in self.posterior_samples

        gate = self.posterior_samples.get("gate") if has_gate else None
        p_capture = self.posterior_samples.get("p_capture") if has_vcp else None
        mixing_weights = (
            self.posterior_samples.get("mixing_weights") if is_mixture else None
        )
        bnb_concentration = self.posterior_samples.get("bnb_concentration")

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
        # Use semantic AxisLayout lookups to find each parameter's gene
        # axis, eliminating ndim/shape heuristics.  Build posterior-level
        # canonical layouts so gene_axis already accounts for the leading
        # sample dimension.
        if gene_indices is not None:
            # Posterior-level canonical layouts (with sample dim).
            layouts = _build_canonical_layouts(
                self.posterior_samples,
                self.model_config,
                n_genes=self.n_genes,
                n_cells=self.n_cells,
                n_components=self.n_components,
                has_sample_dim=True,
            )
            _offset = 0  # gene_axis already includes sample dim offset

            r_ga = layouts["r"].gene_axis
            r = _slice_gene_axis(
                r, r_ga + _offset if r_ga is not None else None,
                gene_indices,
            )

            p_ga = layouts.get("p", None)
            p_ga = p_ga.gene_axis if p_ga is not None else None
            p = _slice_gene_axis(
                p, p_ga + _offset if p_ga is not None else None,
                gene_indices,
            )

            gate_ga = layouts.get("gate", None)
            gate_ga = gate_ga.gene_axis if gate_ga is not None else None
            gate = _slice_gene_axis(
                gate, gate_ga + _offset if gate_ga is not None else None,
                gene_indices,
            )

            bnb_ga = layouts.get("bnb_concentration", None)
            bnb_ga = bnb_ga.gene_axis if bnb_ga is not None else None
            bnb_concentration = _slice_gene_axis(
                bnb_concentration,
                bnb_ga + _offset if bnb_ga is not None else None,
                gene_indices,
            )

        # ---- 4. Sample via the full-model helper ----
        # Build posterior-level canonical layouts (keyed by "r", "p", etc.)
        # using actual posterior tensor shapes and model metadata.
        _post_layouts = _build_canonical_layouts(
            self.posterior_samples,
            self.model_config,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            n_components=self.n_components,
            has_sample_dim=True,
        )
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
            bnb_concentration=bnb_concentration,
            param_layouts=_post_layouts,
        )

        if verbose:
            print(f"Generated posterior PPC samples with shape {samples.shape}")

        if store_samples:
            self.predictive_samples = samples

        return samples
