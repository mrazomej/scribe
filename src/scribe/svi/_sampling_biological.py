"""
Biological predictive sampling mixin for SVI results.
"""

from typing import Dict, Optional

import jax.numpy as jnp
from jax import random

from ..sampling import sample_biological_nb, _build_canonical_layouts


class BiologicalSamplingMixin:
    """Mixin providing biological (technical-noise-stripped) sampling methods."""

    def get_ppc_samples_biological(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        cell_batch_size: Optional[int] = None,
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
            :meth:`get_posterior_samples`).  Only used when
            ``self.posterior_samples`` is ``None``.  Default: ``None``.
        cell_batch_size : Optional[int], optional
            Number of cells to process per batch when drawing biological
            count samples via :func:`scribe.sampling.sample_biological_nb`.
            Use this to limit GPU peak memory on large datasets; ``None``
            processes all cells in a single pass.  Default: ``None``.
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
        # TwoState (Poisson-Beta compound) does not expose NB-style
        # (r, p), but the *biological* PPC has a perfectly clean
        # definition for it: drop the per-cell capture factor and
        # sample observations from the pre-capture Poisson-Beta
        # rate.  By closure under binomial thinning, this is the
        # cell-level distribution of the latent mRNA counts before
        # the sequencing pipeline samples them.  Dispatch to the
        # TwoState-specific helper.
        _bm = getattr(self.model_config, "base_model", None)
        if _bm in ("twostate", "twostatevcp"):
            return self._twostate_biological_ppc_samples(
                rng_key=rng_key,
                n_samples=n_samples,
                batch_size=batch_size,
                store_samples=store_samples,
                counts=counts,
            )

        # Create default RNG key if not provided
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # A narrowed (DE) cache lacks the technical sites this biological PPC
        # reads; reject rather than treat them as absent. A None cache is fine —
        # it is drawn full below.
        self._require_full_posterior_cache(method="get_ppc_samples_biological")

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
        bnb_concentration = self.posterior_samples.get("bnb_concentration")

        # Build posterior-level canonical layouts (keyed by "r", "p", etc.)
        # using the actual posterior tensor shapes and model metadata.
        _layouts = _build_canonical_layouts(
            self.posterior_samples,
            self.model_config,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            n_components=self.n_components,
            has_sample_dim=True,
        )

        # Generate biological (denoised) count samples, processing cells in
        # batches when cell_batch_size is set to avoid GPU OOM on large datasets.
        _, key_bio = random.split(rng_key)
        bio_samples = sample_biological_nb(
            r=r,
            p=p,
            n_cells=self.n_cells,
            rng_key=key_bio,
            mixing_weights=mixing_weights,
            bnb_concentration=bnb_concentration,
            cell_batch_size=cell_batch_size,
            param_layouts=_layouts,
        )

        if store_samples:
            self.predictive_samples_biological = bio_samples

        return {
            "parameter_samples": self.posterior_samples,
            "predictive_samples": bio_samples,
        }

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
        # TwoState branch — see ``get_ppc_samples_biological`` above for
        # the same closure-under-thinning rationale.  Dispatches to a
        # MAP-anchored Poisson-Beta sampler with the capture factor
        # dropped from the rate.
        _bm = getattr(self.model_config, "base_model", None)
        if _bm in ("twostate", "twostatevcp"):
            return self._twostate_map_biological_ppc_samples(
                rng_key=rng_key,
                n_samples=n_samples,
                use_mean=use_mean,
                store_samples=store_samples,
                verbose=verbose,
                counts=counts,
            )

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
        mixing_weights = (
            map_estimates.get("mixing_weights") if is_mixture else None
        )
        bnb_concentration = map_estimates.get("bnb_concentration")

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

        # Sample from the base NB(r, p) only - no capture, no gate.
        # MAP estimates have no sample dim; pass canonical layouts.
        samples = sample_biological_nb(
            r=r,
            p=p,
            n_cells=self.n_cells,
            rng_key=rng_key,
            n_samples=n_samples,
            param_layouts=_map_layouts,
            mixing_weights=mixing_weights,
            cell_batch_size=cell_batch_size,
            bnb_concentration=bnb_concentration,
        )

        if verbose:
            print(
                f"Generated biological predictive samples with shape "
                f"{samples.shape}"
            )

        if store_samples:
            self.predictive_samples_biological = samples

        return samples

    # ------------------------------------------------------------------
    # TwoState biological PPC helpers (drop p_capture from the rate)
    # ------------------------------------------------------------------

    def _twostate_biological_ppc_samples(
        self,
        *,
        rng_key,
        n_samples: int,
        batch_size: Optional[int],
        store_samples: bool,
        counts: Optional[jnp.ndarray],
    ):
        """Full-posterior biological PPC for the TwoState family.

        By closure under binomial thinning, the latent pre-capture
        mRNA count distribution is exactly Poisson-Beta with the same
        ``(α, β, r̂)`` triple but with the per-cell capture factor
        ``ν^(c)`` dropped from the rate.  This helper iterates over
        posterior samples of ``(μ, *extras*)``, dispatches via
        :func:`_twostate_dispatch_reparam` to recover ``(α, β, r̂)``
        for the active parameterization, and draws ``n_cells`` count
        replicates from ``PoissonBetaCompound(α, β, r̂)`` per posterior
        sample.

        Returns a dict mirroring the NB-family API:
        ``{"parameter_samples": ..., "predictive_samples": ...}``.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # A narrowed (DE) cache lacks the technical sites this biological PPC
        # reads; reject rather than treat them as absent. A None cache is fine —
        # it is drawn full below.
        self._require_full_posterior_cache(
            method="get_map_ppc_samples_biological"
        )

        # Ensure posterior samples exist.
        if self.posterior_samples is None:
            key_post, rng_key = random.split(rng_key)
            self.get_posterior_samples(
                rng_key=key_post,
                n_samples=n_samples,
                batch_size=batch_size,
                store_samples=True,
                counts=counts,
            )

        from ..models.components.likelihoods.two_state import (
            _twostate_dispatch_reparam,
        )
        from ..stats.distributions import PoissonBetaCompound

        # Strip the capture site from the dispatch input — the
        # dispatcher reads the sampled keys to detect the
        # parameterization; ``p_capture`` is not one of them, but
        # explicit removal makes the intent clear.
        post = {
            k: v
            for k, v in self.posterior_samples.items()
            if k != "p_capture"
        }

        # Run a posterior-sample loop with vmap-like batching: build the
        # compound at the gene-rank rate per sample and draw one
        # ``(n_cells, n_genes)`` count matrix per sample.
        n_post = jnp.shape(post["mu"])[0]

        def _draw_one(sample_index, key):
            params_one = {k: v[sample_index] for k, v in post.items()}
            alpha, beta, rate_gene, _eff, _raw = _twostate_dispatch_reparam(
                params_one
            )
            compound = PoissonBetaCompound(
                alpha=alpha, beta=beta, rate=rate_gene
            )
            # Replicate across cells: sample_shape (n_cells,) gives
            # (n_cells, n_genes) independent per (c, g) — the
            # ancestral-sampling correctness fix from earlier on this
            # branch.
            return compound.sample(key, sample_shape=(self.n_cells,))

        keys = random.split(rng_key, n_post)
        bio_samples = jnp.stack(
            [_draw_one(i, keys[i]) for i in range(n_post)], axis=0
        )

        if store_samples:
            self.predictive_samples_biological = bio_samples

        return {
            "parameter_samples": self.posterior_samples,
            "predictive_samples": bio_samples,
        }

    def _twostate_map_biological_ppc_samples(
        self,
        *,
        rng_key,
        n_samples: int,
        use_mean: bool,
        store_samples: bool,
        verbose: bool,
        counts: Optional[jnp.ndarray],
    ):
        """MAP-anchored biological PPC for the TwoState family.

        Identical structure to :meth:`_twostate_map_ppc_samples` (in
        ``_sampling_map_predictive.py``) except the per-cell rate is
        the gene-rank ``r̂_g`` directly — no multiplication by
        ``p_capture``.  Returns the count tensor at shape
        ``(n_samples, n_cells, n_genes)``.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        if verbose:
            print("Getting MAP estimates for TwoState biological PPC...")

        map_estimates = self.get_map(
            use_mean=use_mean, canonical=True, verbose=False, counts=counts
        )

        from ..models.components.likelihoods.two_state import (
            _twostate_dispatch_reparam,
        )
        from ..stats.distributions import PoissonBetaCompound

        # Drop p_capture from the dispatch input to compute the
        # biological (pre-capture) rate.
        map_no_capture = {
            k: v for k, v in map_estimates.items() if k != "p_capture"
        }
        alpha, beta, rate_gene, _eff, _raw = _twostate_dispatch_reparam(
            map_no_capture
        )

        compound = PoissonBetaCompound(
            alpha=alpha, beta=beta, rate=rate_gene
        )
        # Replicate across cells: sample_shape (n_samples, n_cells) →
        # (n_samples, n_cells, n_genes), independent p_gc per draw.
        samples = compound.sample(
            rng_key, sample_shape=(n_samples, self.n_cells)
        )

        if verbose:
            print(
                f"Generated TwoState biological PPC with shape "
                f"{samples.shape}"
            )
        if store_samples:
            self.predictive_samples_biological = samples
        return samples
