"""
Latent-space mixin for VAE results.

Provides methods for encoding, decoding, and sampling in the latent space.
All encoder-dependent logic is delegated to the dispatched free functions
in ``_latent_dispatch.py``, making this mixin distribution-agnostic.

Classes
-------
LatentSpaceMixin
    Mixin providing latent-space operations for VAE results.

See Also
--------
scribe.svi._latent_dispatch : Dispatched encoder-dependent operations.
scribe.svi.vae_results : ScribeVAEResults dataclass that uses this mixin.
"""

from typing import Dict, Optional

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random

from ..flows import FlowDistribution
from ._latent_dispatch import (
    get_latent_embedding,
    run_encoder,
    sample_latent_posterior,
)

# ==============================================================================
# Param-store key constants
# ==============================================================================

_ENCODER_KEY = "vae_encoder$params"
_DECODER_KEY = "vae_decoder$params"
_FLOW_KEY = "vae_prior_flow$params"


# ==============================================================================
# Latent Space Mixin
# ==============================================================================


class LatentSpaceMixin:
    """Mixin providing latent-space operations for VAE results.

    This mixin assumes the host class (``ScribeVAEResults``) exposes:

    * ``self.params`` — full NumPyro param dict with keys like
      ``vae_encoder$params``, ``vae_decoder$params``, and optionally
      ``vae_prior_flow$params``.
    * ``self._encoder`` — un-initialized encoder Linen module.
    * ``self._decoder`` — un-initialized decoder Linen module.
    * ``self._latent_spec`` — a ``LatentSpec`` subclass (e.g.
      ``GaussianLatentSpec``) with optional ``.flow``.

    All encoder-dependent logic (output structure, sampling strategy,
    embedding extraction) is delegated to the dispatched free functions
    in ``_latent_dispatch.py``.  This keeps the mixin distribution-agnostic.
    """

    # ------------------------------------------------------------------
    # Internal helpers — param extraction
    # ------------------------------------------------------------------

    def _get_encoder_params(self) -> Dict:
        """Extract encoder params subtree from the full params dict.

        Returns
        -------
        dict
            The trained encoder parameters (value at key
            ``vae_encoder$params``).

        Raises
        ------
        KeyError
            If the encoder param key is missing.
        """
        return self.params[_ENCODER_KEY]

    # --------------------------------------------------------------------------

    def _get_decoder_params(self) -> Dict:
        """Extract decoder params subtree from the full params dict.

        Returns
        -------
        dict
            The trained decoder parameters (value at key
            ``vae_decoder$params``).

        Raises
        ------
        KeyError
            If the decoder param key is missing.
        """
        return self.params[_DECODER_KEY]

    # --------------------------------------------------------------------------

    def _get_flow_params(self) -> Optional[Dict]:
        """Extract flow params subtree, or None if no flow.

        Returns
        -------
        dict or None
            The trained flow parameters, or ``None`` if no flow prior
            is configured.
        """
        return self.params.get(_FLOW_KEY)

    # --------------------------------------------------------------------------
    # Internal helpers — encoder / decoder / prior
    # --------------------------------------------------------------------------

    def _run_encoder(self, counts: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Run encoder on counts via dispatch.

        Delegates to ``run_encoder(spec, encoder, enc_params, counts)``
        which dispatches on ``self._latent_spec`` type.  The returned
        dict is opaque to this mixin — its contents depend on the
        encoder type (e.g. ``{"loc", "log_scale"}`` for Gaussian).

        Parameters
        ----------
        counts : jnp.ndarray, shape (n_cells, n_genes) or (batch, n_genes)
            Input count matrix.

        Returns
        -------
        var_params : Dict[str, jnp.ndarray]
            Variational parameters produced by the encoder.
        """
        enc_params = self._get_encoder_params()
        return run_encoder(self._latent_spec, self._encoder, enc_params, counts)

    # --------------------------------------------------------------------------

    def _run_decoder(self, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Run decoder on latent z to get parameter values.

        Parameters
        ----------
        z : jnp.ndarray, shape (..., latent_dim)
            Latent codes.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names (e.g. ``"r"``, ``"gate"``)
            to their decoded values.
        """
        dec_params = self._get_decoder_params()
        return self._decoder.apply({"params": dec_params}, z)

    # --------------------------------------------------------------------------

    def _build_prior_distribution(self) -> dist.Distribution:
        """Build the prior distribution for z.

        If ``_latent_spec.flow`` is set, wraps the trained flow in a
        ``FlowDistribution``.  Otherwise returns the standard prior
        from ``_latent_spec.make_prior_dist()`` (e.g. Normal(0, I)).

        Returns
        -------
        dist.Distribution
            The prior on z.
        """
        if self._latent_spec.flow is not None:
            flow = self._latent_spec.flow
            flow_params = self._get_flow_params()

            # Build a closure that applies the flow with trained params
            def flow_fn(x, reverse=False):
                return flow.apply({"params": flow_params}, x, reverse=reverse)

            # Base distribution: standard Normal matching latent dim
            base = dist.Normal(
                jnp.zeros(self._latent_spec.latent_dim),
                jnp.ones(self._latent_spec.latent_dim),
            ).to_event(1)
            return FlowDistribution(flow_fn, base)
        else:
            return self._latent_spec.make_prior_dist()

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------

    def get_latent_embeddings(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """Encode counts to latent-space point embeddings (no sampling).

        Runs the encoder via ``_run_encoder`` (dispatched) and extracts a point
        embedding via ``get_latent_embedding`` (dispatched). For Gaussian
        encoders this returns the posterior mean; other encoder types may return
        a different summary statistic.

        Parameters
        ----------
        counts : jnp.ndarray, shape (n_cells, n_genes)
            Count matrix to encode.
        batch_size : int, optional
            If set, process counts in batches to limit memory usage.

        Returns
        -------
        jnp.ndarray, shape (n_cells, latent_dim)
            Point latent embeddings for each cell.

        Raises
        ------
        ValueError
            If ``counts`` is None.
        """
        if counts is None:
            raise ValueError(
                "counts is required for get_latent_embeddings "
                "(the encoder needs input data)."
            )
        if batch_size is None:
            var_params = self._run_encoder(counts)
            return get_latent_embedding(self._latent_spec, var_params)

        # Batch processing to limit memory
        embeddings = []
        for start in range(0, counts.shape[0], batch_size):
            batch = counts[start : start + batch_size]
            var_params = self._run_encoder(batch)
            embeddings.append(
                get_latent_embedding(self._latent_spec, var_params)
            )
        return jnp.concatenate(embeddings, axis=0)

    # ------------------------------------------------------------------

    def get_latent_samples(
        self,
        n_samples: int = 100,
        rng_key=None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Sample z from the prior distribution.

        If a prior flow is set on the latent spec, samples pass through
        the learned flow.  Otherwise samples come from the standard
        prior (e.g. Normal(0, I) for Gaussian).

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        rng_key : PRNGKey, optional
            JAX PRNG key.  Default: ``PRNGKey(42)``.
        store_samples : bool
            If True, caches result in ``self.latent_samples``.

        Returns
        -------
        jnp.ndarray, shape (n_samples, latent_dim)
            Latent samples from the prior.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        prior = self._build_prior_distribution()
        samples = prior.sample(rng_key, sample_shape=(n_samples,))

        if store_samples:
            self.latent_samples = samples
        return samples

    # --------------------------------------------------------------------------

    def get_latent_samples_conditioned_on_data(
        self,
        counts: jnp.ndarray,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        rng_key=None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Encode counts and sample z from the approximate posterior q(z|x).

        Runs the encoder via ``_run_encoder`` (dispatched) then draws
        posterior samples via ``sample_latent_posterior`` (dispatched).
        The mixin never touches distribution-specific details — sampling
        logic lives entirely in the dispatch implementations.

        Parameters
        ----------
        counts : jnp.ndarray, shape (n_cells, n_genes)
            Count matrix.
        n_samples : int
            Number of posterior samples per cell.
        batch_size : int, optional
            Process cells in batches to limit memory.
        rng_key : PRNGKey, optional
            JAX PRNG key.  Default: ``PRNGKey(42)``.
        store_samples : bool
            If True, caches result in ``self.latent_samples``.

        Returns
        -------
        jnp.ndarray, shape (n_samples, n_cells, latent_dim)
            Posterior latent samples.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        def _sample_batch(counts_batch, key):
            # Encode counts -> variational parameters (dispatch-based)
            var_params = self._run_encoder(counts_batch)
            # Sample z from q(z|x) using the dispatched sampler
            return sample_latent_posterior(
                self._latent_spec, var_params, key, n_samples
            )

        if batch_size is None:
            samples = _sample_batch(counts, rng_key)
        else:
            all_samples = []
            for start in range(0, counts.shape[0], batch_size):
                rng_key, subkey = random.split(rng_key)
                batch = counts[start : start + batch_size]
                all_samples.append(_sample_batch(batch, subkey))
            samples = jnp.concatenate(all_samples, axis=1)

        if store_samples:
            self.latent_samples = samples
        return samples

    # ------------------------------------------------------------------

    def get_decoded_params(
        self,
        z: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Run decoder on latent z to get parameter values.

        Parameters
        ----------
        z : jnp.ndarray, shape (..., latent_dim)
            Latent codes (from prior or posterior).

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to decoded arrays.
        """
        return self._run_decoder(z)
