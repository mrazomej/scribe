"""
Likelihood mixin for MCMC results.

Provides log-likelihood computation using posterior samples.
"""

from typing import Dict, Optional

import jax.numpy as jnp
from jax import jit, vmap


# ==============================================================================
# Likelihood Mixin
# ==============================================================================


class LikelihoodMixin:
    """Mixin providing log-likelihood computation methods."""

    def log_likelihood(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        sample_chunk_size: Optional[int] = None,
        return_by: str = "cell",
        cells_axis: int = 0,
        ignore_nans: bool = False,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Compute log-likelihood of data under posterior samples.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate.
        batch_size : int or None, default=None
            Cell mini-batch size.
        sample_chunk_size : int or None, default=None
            Posterior-sample chunk size (bounds peak memory).
        return_by : {'cell', 'gene'}, default='cell'
            Aggregation axis.
        cells_axis : int, default=0
            Axis along which cells are arranged.
        ignore_nans : bool, default=False
            Drop samples containing NaNs.
        split_components : bool, default=False
            Return per-component log-likelihoods (mixture models).
        weights : jnp.ndarray or None, default=None
            Gene weights for the log-likelihood.
        weight_type : str or None, default=None
            ``'multiplicative'`` or ``'additive'``.
        dtype : jnp.dtype, default=jnp.float32
            Numerical precision.

        Returns
        -------
        jnp.ndarray
            Log-likelihoods; shape depends on *return_by* and
            *split_components*.
        """
        samples = self.get_posterior_samples()

        return _compute_log_likelihood(
            samples,
            counts,
            self.model_type,
            n_components=self.n_components,
            batch_size=batch_size,
            sample_chunk_size=sample_chunk_size,
            return_by=return_by,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            split_components=split_components,
            weights=weights,
            weight_type=weight_type,
            dtype=dtype,
        )


# ==============================================================================
# Module-level helper
# ==============================================================================


def _compute_log_likelihood(
    samples: Dict,
    counts: jnp.ndarray,
    model_type: str,
    n_components: Optional[int] = None,
    batch_size: Optional[int] = None,
    sample_chunk_size: Optional[int] = None,
    return_by: str = "cell",
    cells_axis: int = 0,
    ignore_nans: bool = False,
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Compute log-likelihood of *counts* under *samples*."""
    from ._model_helpers import _get_log_likelihood_fn

    n_samples = samples[next(iter(samples))].shape[0]
    likelihood_fn = _get_log_likelihood_fn(model_type)
    is_mixture = n_components is not None and n_components > 1

    @jit
    def compute_sample_lik(i):
        params_i = {k: v[i] for k, v in samples.items()}
        if is_mixture:
            return likelihood_fn(
                counts,
                params_i,
                batch_size=batch_size,
                cells_axis=cells_axis,
                return_by=return_by,
                split_components=split_components,
                weights=weights,
                weight_type=weight_type,
                dtype=dtype,
            )
        return likelihood_fn(
            counts,
            params_i,
            batch_size=batch_size,
            cells_axis=cells_axis,
            return_by=return_by,
            dtype=dtype,
        )

    # Optionally chunk over samples to bound peak memory
    if (
        sample_chunk_size is None
        or sample_chunk_size <= 0
        or sample_chunk_size >= n_samples
    ):
        log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))
    else:
        chunks = []
        for start in range(0, n_samples, sample_chunk_size):
            end = min(start + sample_chunk_size, n_samples)
            chunks.append(vmap(compute_sample_lik)(jnp.arange(start, end)))
        log_liks = jnp.concatenate(chunks, axis=0)

    if ignore_nans:
        if is_mixture and split_components:
            valid = ~jnp.any(jnp.any(jnp.isnan(log_liks), axis=-1), axis=-1)
        else:
            valid = ~jnp.any(jnp.isnan(log_liks), axis=-1)
        if jnp.any(~valid):
            print(
                f"    - Fraction of samples removed: "
                f"{1 - jnp.mean(valid)}"
            )
            return log_liks[valid]

    return log_liks
