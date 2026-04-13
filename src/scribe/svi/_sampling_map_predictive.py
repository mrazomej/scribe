"""
MAP and full-model predictive sampling mixin for SVI results.
"""

from typing import Optional

import jax.numpy as jnp
from jax import random

from ..sampling import sample_posterior_ppc


def _coerce_map_capture_vector(
    p_capture: jnp.ndarray, n_cells: int
) -> jnp.ndarray:
    """Normalize MAP capture probabilities to a 1-D cell vector.

    Parameters
    ----------
    p_capture : jnp.ndarray
        Capture probability candidate from MAP estimates.
    n_cells : int
        Expected number of cells.

    Returns
    -------
    jnp.ndarray
        Capture probabilities with shape ``(n_cells,)``.

    Raises
    ------
    ValueError
        If the value cannot be interpreted as per-cell capture probabilities.
    """
    # Convert to an array and collapse singleton axes to tolerate shape
    # artifacts from upstream subsetting.
    p_capture = jnp.asarray(p_capture)
    p_capture = jnp.squeeze(p_capture)

    # Scalar capture probability is valid; broadcast to all cells.
    if p_capture.ndim == 0:
        return jnp.full((n_cells,), p_capture)

    # A vector must match the model's cell axis exactly.
    if p_capture.ndim == 1 and p_capture.shape[0] == n_cells:
        return p_capture

    raise ValueError(
        "Invalid MAP p_capture shape. Expected scalar or "
        f"(n_cells,) with n_cells={n_cells}, got shape {p_capture.shape}."
    )


def _normalize_map_standard_inputs(
    r: jnp.ndarray,
    p: jnp.ndarray,
    gate: Optional[jnp.ndarray],
    p_capture: Optional[jnp.ndarray],
    n_cells: int,
) -> tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Normalize MAP inputs for non-mixture models.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion candidate from MAP estimates.
    p : jnp.ndarray
        Success probability candidate from MAP estimates.
    gate : Optional[jnp.ndarray]
        Optional zero-inflation gate candidate.
    p_capture : Optional[jnp.ndarray]
        Optional per-cell capture probability candidate.
    n_cells : int
        Number of cells in the current results view.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]
        Normalized ``(r, p, gate, p_capture)`` for standard-model PPC.

    Raises
    ------
    ValueError
        If any input has incompatible dimensions.
    """
    # Ensure r always exposes a gene axis; singleton genes become length-1
    # vectors instead of rank-0 scalars.
    r = jnp.atleast_1d(jnp.asarray(r))
    if r.ndim != 1:
        r = jnp.squeeze(r)
        r = jnp.atleast_1d(r)
    if r.ndim != 1:
        raise ValueError(
            "Invalid MAP r shape for non-mixture model. Expected "
            f"(n_genes,), got {r.shape}."
        )
    n_genes = int(r.shape[0])

    # p can be scalar or per-gene; convert singleton vectors to scalar.
    p = jnp.asarray(p)
    p = jnp.squeeze(p)
    if p.ndim == 0:
        pass
    elif p.ndim == 1:
        if p.shape[0] == 1:
            p = p[0]
        elif p.shape[0] != n_genes:
            raise ValueError(
                "Invalid MAP p shape for non-mixture model. Expected scalar "
                f"or (n_genes,) with n_genes={n_genes}, got {p.shape}."
            )
    else:
        raise ValueError(
            "Invalid MAP p rank for non-mixture model. Expected scalar or "
            f"vector, got shape {p.shape}."
        )

    # gate can be absent, scalar, or per-gene; normalize to per-gene when set.
    if gate is not None:
        gate = jnp.asarray(gate)
        gate = jnp.squeeze(gate)
        if gate.ndim == 0:
            gate = jnp.full((n_genes,), gate)
        elif gate.ndim == 1 and gate.shape[0] == n_genes:
            pass
        else:
            raise ValueError(
                "Invalid MAP gate shape for non-mixture model. Expected scalar "
                f"or (n_genes,) with n_genes={n_genes}, got {gate.shape}."
            )

    # p_capture can be absent, scalar, or per-cell; normalize to per-cell.
    if p_capture is not None:
        p_capture = _coerce_map_capture_vector(p_capture, n_cells=n_cells)

    return r, p, gate, p_capture


def _normalize_map_mixture_inputs(
    r: jnp.ndarray,
    p: jnp.ndarray,
    gate: Optional[jnp.ndarray],
    p_capture: Optional[jnp.ndarray],
    mixing_weights: Optional[jnp.ndarray],
    n_cells: int,
    n_components: int,
) -> tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], jnp.ndarray]:
    """Normalize MAP inputs for mixture models.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion candidate from MAP estimates.
    p : jnp.ndarray
        Success probability candidate from MAP estimates.
    gate : Optional[jnp.ndarray]
        Optional zero-inflation gate candidate.
    p_capture : Optional[jnp.ndarray]
        Optional per-cell capture probability candidate.
    mixing_weights : Optional[jnp.ndarray]
        Component weights from MAP estimates.
    n_cells : int
        Number of cells in the current results view.
    n_components : int
        Expected number of mixture components.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], jnp.ndarray]
        Normalized ``(r, p, gate, p_capture, mixing_weights)`` for mixture PPC.

    Raises
    ------
    ValueError
        If any input has incompatible dimensions.
    """
    # Mixture r must be (K, G); tolerate squeezed singleton component axes.
    r = jnp.asarray(r)
    r = jnp.squeeze(r)
    if r.ndim == 1 and n_components == 1:
        r = r[None, :]
    if r.ndim != 2 or r.shape[0] != n_components:
        raise ValueError(
            "Invalid MAP r shape for mixture model. Expected (n_components, n_genes) "
            f"with n_components={n_components}, got {r.shape}."
        )
    n_genes = int(r.shape[1])

    # p may be scalar, (K,), (G,), or (K, G). Reject incompatible shapes.
    p = jnp.asarray(p)
    p = jnp.squeeze(p)
    if p.ndim == 0:
        pass
    elif p.ndim == 1:
        if p.shape[0] in (1, n_components, n_genes):
            if p.shape[0] == 1:
                p = p[0]
        else:
            raise ValueError(
                "Invalid MAP p shape for mixture model. Expected scalar, "
                f"(n_components,), (n_genes,), or (n_components, n_genes); got {p.shape}."
            )
    elif p.ndim == 2:
        if p.shape != (n_components, n_genes):
            raise ValueError(
                "Invalid MAP p matrix shape for mixture model. Expected "
                f"(n_components, n_genes)=({n_components}, {n_genes}), got {p.shape}."
            )
    else:
        raise ValueError(
            "Invalid MAP p rank for mixture model. Expected scalar, vector, "
            f"or matrix, got shape {p.shape}."
        )

    # gate may be absent, scalar, shared per-gene (G,), or per-component (K, G).
    if gate is not None:
        gate = jnp.asarray(gate)
        gate = jnp.squeeze(gate)
        if gate.ndim == 0:
            gate = jnp.full((n_genes,), gate)
        elif gate.ndim == 1:
            if gate.shape[0] != n_genes:
                raise ValueError(
                    "Invalid MAP gate vector shape for mixture model. Expected "
                    f"(n_genes,) with n_genes={n_genes}, got {gate.shape}."
                )
        elif gate.ndim == 2:
            if gate.shape != (n_components, n_genes):
                raise ValueError(
                    "Invalid MAP gate matrix shape for mixture model. Expected "
                    f"(n_components, n_genes)=({n_components}, {n_genes}), got {gate.shape}."
                )
        else:
            raise ValueError(
                "Invalid MAP gate rank for mixture model. Expected scalar, "
                f"vector, or matrix, got shape {gate.shape}."
            )

    # p_capture can be absent, scalar, or per-cell; normalize to per-cell.
    if p_capture is not None:
        p_capture = _coerce_map_capture_vector(p_capture, n_cells=n_cells)

    # Mixture weights are required; normalize to a 1-D (K,) vector.
    if mixing_weights is None:
        raise ValueError(
            "MAP mixture sampling requires mixing_weights in MAP estimates."
        )
    mixing_weights = jnp.asarray(mixing_weights)
    mixing_weights = jnp.squeeze(mixing_weights)
    if mixing_weights.ndim == 0 and n_components == 1:
        mixing_weights = jnp.array([mixing_weights])
    if mixing_weights.ndim != 1 or mixing_weights.shape[0] != n_components:
        raise ValueError(
            "Invalid MAP mixing_weights shape. Expected "
            f"(n_components,) with n_components={n_components}, got {mixing_weights.shape}."
        )

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

        # Normalize MAP tensors to stable rank/shape contracts before sampling.
        # This avoids scalar-shape failures after singleton subsetting.
        if is_mixture:
            r, p, gate, p_capture, mixing_weights = _normalize_map_mixture_inputs(
                r=r,
                p=p,
                gate=gate,
                p_capture=p_capture,
                mixing_weights=mixing_weights,
                n_cells=self.n_cells,
                n_components=int(self.n_components),
            )
            n_genes = int(r.shape[1])
        else:
            r, p, gate, p_capture = _normalize_map_standard_inputs(
                r=r,
                p=p,
                gate=gate,
                p_capture=p_capture,
                n_cells=self.n_cells,
            )
            n_genes = int(r.shape[0])

        # Generate MAP PPC samples through the full-model helper so shape
        # handling is consistent with posterior PPC paths.
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
        # Parameters that may carry a gene axis: r, p (hierarchical),
        # gate, bnb_concentration.  p_capture is cell-indexed, not gene-indexed.
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
                if (
                    bnb_concentration is not None
                    and bnb_concentration.shape[-1] == n_genes
                ):
                    bnb_concentration = bnb_concentration[..., gene_indices]
            else:
                # r: (S, G) -> (S, G_batch)
                r = r[:, gene_indices]
                if gate is not None and gate.ndim == 2:
                    gate = gate[:, gene_indices]
                # p may be (S, G) for hierarchical (per-gene p) models
                if p.ndim == 2 and p.shape[-1] == n_genes:
                    p = p[:, gene_indices]
                if (
                    bnb_concentration is not None
                    and bnb_concentration.shape[-1] == n_genes
                ):
                    bnb_concentration = bnb_concentration[..., gene_indices]

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
            bnb_concentration=bnb_concentration,
        )

        if verbose:
            print(f"Generated posterior PPC samples with shape {samples.shape}")

        if store_samples:
            self.predictive_samples = samples

        return samples
