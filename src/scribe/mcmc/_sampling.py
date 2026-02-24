"""
Sampling mixin for MCMC results.

Provides posterior predictive checks (standard and biological), Bayesian
denoising, and prior predictive sampling.
"""

from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import random

from ..models.config import ModelConfig
from ..sampling import (
    generate_predictive_samples,
    generate_prior_predictive_samples,
    sample_biological_nb,
    denoise_counts as _denoise_counts_util,
)


# ==============================================================================
# Sampling Mixin
# ==============================================================================


class SamplingMixin:
    """Mixin providing predictive sampling and denoising methods."""

    # --------------------------------------------------------------------------
    # Posterior predictive checks
    # --------------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate posterior predictive check (PPC) samples.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        batch_size : int or None, optional
            Batch size for sample generation.
        store_samples : bool, default=True
            Store result in ``self.predictive_samples``.

        Returns
        -------
        jnp.ndarray
            Posterior predictive samples.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        predictive_samples = _generate_ppc_samples(
            self.get_posterior_samples(),
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

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
        """Generate MAP-based posterior predictive samples.

        This method builds a synthetic one-point posterior from the MAP estimate
        and feeds it through the same predictive sampler used by
        :meth:`get_ppc_samples`. The MAP point is repeated ``n_samples`` times so
        callers receive an array with shape ``(n_samples, n_cells, n_genes)``
        that mirrors the SVI API.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        n_samples : int, default=1
            Number of predictive draws to return from the MAP point.
        cell_batch_size : int or None, optional
            Batch size for predictive generation over cells.
        use_mean : bool, default=True
            Included for API parity with SVI. MCMC MAP extraction already
            includes a posterior-mean fallback when potential energy is
            unavailable, so this flag is currently informational.
        store_samples : bool, default=True
            Store generated samples in ``self.predictive_samples``.
        verbose : bool, default=True
            Print lightweight progress messages.
        counts : jnp.ndarray or None, optional
            Included for API parity with SVI. Not required for MCMC MAP PPC.

        Returns
        -------
        jnp.ndarray
            Predictive count samples with shape
            ``(n_samples, n_cells, n_genes)``.

        Notes
        -----
        Unlike full posterior PPC, this method does not integrate over all
        posterior draws; it conditions on a single MAP point estimate.
        """
        _ = use_mean
        _ = counts
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1 for MAP PPC sampling.")

        if verbose:
            print("Generating MCMC MAP PPC samples...")

        # Convert MAP parameter values to a pseudo-posterior with sample axis.
        map_estimates = self.get_map()
        map_parameter_samples = {}
        for key, value in map_estimates.items():
            value_array = jnp.asarray(value)
            value_with_sample_axis = jnp.expand_dims(value_array, axis=0)
            map_parameter_samples[key] = jnp.repeat(
                value_with_sample_axis, repeats=n_samples, axis=0
            )

        predictive_samples = _generate_ppc_samples(
            map_parameter_samples,
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            batch_size=cell_batch_size,
        )

        if store_samples:
            self.predictive_samples = predictive_samples
        return predictive_samples

    # --------------------------------------------------------------------------
    # Biological (denoised) PPC
    # --------------------------------------------------------------------------

    def get_ppc_samples_biological(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        cell_batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """Generate biological PPC samples from base NB(r, p).

        Strips technical noise parameters (capture probability,
        zero-inflation gate) to yield counts reflecting the latent
        biology.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        batch_size : int or None, optional
            Kept for API symmetry; use ``cell_batch_size`` instead.
        store_samples : bool, default=True
            Store in ``self.predictive_samples_biological``.
        cell_batch_size : int or None, optional
            Process cells in batches of this size.

        Returns
        -------
        jnp.ndarray
            Biological count samples of shape
            ``(n_posterior_samples, n_cells, n_genes)``.

        See Also
        --------
        get_ppc_samples : Standard PPC including technical noise.
        scribe.sampling.sample_biological_nb : Core sampling utility.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        samples = self.get_posterior_samples()

        bio_samples = sample_biological_nb(
            r=samples["r"],
            p=samples["p"],
            n_cells=self.n_cells,
            rng_key=rng_key,
            mixing_weights=samples.get("mixing_weights"),
            cell_batch_size=cell_batch_size,
        )

        if store_samples:
            self.predictive_samples_biological = bio_samples

        return bio_samples

    # --------------------------------------------------------------------------
    # Bayesian denoising
    # --------------------------------------------------------------------------

    def denoise_counts(
        self,
        counts: jnp.ndarray,
        method: str = "mean",
        rng_key: Optional[random.PRNGKey] = None,
        return_variance: bool = False,
        cell_batch_size: Optional[int] = None,
        store_result: bool = True,
        verbose: bool = True,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Denoise observed counts using posterior samples.

        Propagates parameter uncertainty by computing the denoised
        posterior for each MCMC draw.

        Parameters
        ----------
        counts : jnp.ndarray
            Observed UMI count matrix ``(n_cells, n_genes)``.
        method : {'mean', 'mode', 'sample'}, default='mean'
            Summary of the per-sample denoised posterior.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        return_variance : bool, default=False
            Return dict with ``'denoised_counts'`` and ``'variance'``.
        cell_batch_size : int or None, optional
            Cell batching inside each posterior draw.
        store_result : bool, default=True
            Store result in ``self.denoised_counts``.
        verbose : bool, default=True
            Print progress messages.

        Returns
        -------
        jnp.ndarray or Dict[str, jnp.ndarray]
            Denoised counts ``(n_posterior_samples, n_cells, n_genes)``.

        See Also
        --------
        get_ppc_samples_biological : Biological PPC (unconditional).
        scribe.sampling.denoise_counts : Core denoising utility.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        samples = self.get_posterior_samples()

        r = samples["r"]
        p = samples["p"]
        p_capture = samples.get("p_capture")
        gate = samples.get("gate")
        mixing_weights = samples.get("mixing_weights")

        if verbose:
            extras = []
            if p_capture is not None:
                extras.append("VCP")
            if gate is not None:
                extras.append("gate")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            n_post = r.shape[0]
            print(
                f"Denoising with {n_post} MCMC samples"
                f" ({self.model_type}){extra_str}, method='{method}'..."
            )

        result = _denoise_counts_util(
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
    # Prior predictive sampling
    # --------------------------------------------------------------------------

    def get_prior_predictive_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate samples from the prior predictive distribution.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        n_samples : int, default=100
            Number of prior predictive samples.
        batch_size : int or None, optional
            Batch size for sample generation.
        store_samples : bool, default=True
            Store in ``self.prior_predictive_samples``.

        Returns
        -------
        jnp.ndarray
            Prior predictive samples.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        prior_predictive_samples = _generate_prior_predictive_samples(
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        if store_samples:
            self.prior_predictive_samples = prior_predictive_samples

        return prior_predictive_samples


# ==============================================================================
# Module-level helpers
# ==============================================================================


def _generate_ppc_samples(
    samples: Dict,
    model_type: str,
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    rng_key: Optional[random.PRNGKey] = None,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate predictive samples using posterior parameter samples."""
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    from ._model_helpers import _get_model_fn

    model = _get_model_fn(model_config)
    model_args = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": model_config,
    }

    return generate_predictive_samples(
        model,
        samples,
        model_args,
        rng_key=rng_key,
        batch_size=batch_size,
    )


def _generate_prior_predictive_samples(
    model_type: str,
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    rng_key: Optional[random.PRNGKey] = None,
    n_samples: int = 100,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate prior predictive samples using the model."""
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    from ._model_helpers import _get_model_fn

    model = _get_model_fn(model_config)
    model_args = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": model_config,
    }

    return generate_prior_predictive_samples(
        model,
        model_args,
        rng_key=rng_key,
        n_samples=n_samples,
        batch_size=batch_size,
    )
