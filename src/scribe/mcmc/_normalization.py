"""
Normalization mixin for MCMC results.

Provides Dirichlet-based count normalization using posterior samples of
the dispersion parameter *r*.
"""

from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import random

from ..core.normalization import normalize_counts_from_posterior


# ==============================================================================
# Normalization Mixin
# ==============================================================================


class NormalizationMixin:
    """Mixin providing count normalization methods."""

    def normalize_counts(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples_dirichlet: int = 1000,
        fit_distribution: bool = True,
        store_samples: bool = False,
        sample_axis: int = 0,
        return_concentrations: bool = False,
        backend: str = "numpyro",
        batch_size: int = 2048,
        verbose: bool = True,
    ) -> Dict[str, Union[jnp.ndarray, object]]:
        """Normalize counts using posterior samples of *r*.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        n_samples_dirichlet : int, default=1000
            Number of Dirichlet samples per posterior draw.
        fit_distribution : bool, default=True
            Fit a Dirichlet to the generated samples.
        store_samples : bool, default=False
            Include raw Dirichlet samples in the output.
        sample_axis : int, default=0
            Axis for Dirichlet fitting.
        return_concentrations : bool, default=False
            Return the raw *r* samples alongside.
        backend : {'numpyro', 'scipy'}, default='numpyro'
            Distribution backend.
        batch_size : int, default=2048
            Posterior samples per Dirichlet batch.
        verbose : bool, default=True
            Print progress messages.

        Returns
        -------
        Dict[str, Union[jnp.ndarray, object]]
            Normalized expression results.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        posterior_samples = self.get_posterior_samples()

        return normalize_counts_from_posterior(
            posterior_samples=posterior_samples,
            n_components=self.n_components,
            rng_key=rng_key,
            n_samples_dirichlet=n_samples_dirichlet,
            fit_distribution=fit_distribution,
            store_samples=store_samples,
            sample_axis=sample_axis,
            return_concentrations=return_concentrations,
            backend=backend,
            batch_size=batch_size,
            verbose=verbose,
        )
