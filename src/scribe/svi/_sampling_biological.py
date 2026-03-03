"""
Biological predictive sampling mixin for SVI results.
"""

from typing import Dict, Optional

import jax.numpy as jnp
from jax import random

from ..sampling import sample_biological_nb


class BiologicalSamplingMixin:
    """Mixin providing biological (technical-noise-stripped) sampling methods."""

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
            size for count generation - use ``cell_batch_size`` in
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

        # Sample from the base NB(r, p) only - no capture, no gate
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
