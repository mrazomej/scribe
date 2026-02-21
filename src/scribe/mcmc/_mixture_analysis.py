"""
Mixture analysis mixin for MCMC results.

Provides probabilistic cell-type assignment for mixture models.
"""

from typing import Dict, Optional

import jax.numpy as jnp


# ==============================================================================
# Mixture Analysis Mixin
# ==============================================================================


class MixtureAnalysisMixin:
    """Mixin providing mixture-model analysis methods."""

    def cell_type_probabilities(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        ignore_nans: bool = False,
        dtype: jnp.dtype = jnp.float32,
        fit_distribution: bool = True,
        temperature: Optional[float] = None,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """Compute probabilistic cell-type assignments.

        For each cell, computes component-specific log-likelihoods using
        posterior samples, converts them to probability distributions,
        and optionally fits a Dirichlet to characterise uncertainty.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate assignments for.
        batch_size : int or None, default=None
            Mini-batch size for likelihood computation.
        cells_axis : int, default=0
            Axis along which cells are arranged.
        ignore_nans : bool, default=False
            Drop NaN-containing samples.
        dtype : jnp.dtype, default=jnp.float32
            Numerical precision.
        fit_distribution : bool, default=True
            Fit a Dirichlet to the assignment probabilities.
        temperature : float or None, default=None
            Temperature scaling for log probabilities.
        weights : jnp.ndarray or None, default=None
            Gene weights for log-likelihood computation.
        weight_type : str or None, default=None
            ``'multiplicative'`` or ``'additive'``.
        verbose : bool, default=True
            Print progress messages.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary with ``'sample_probabilities'`` and, when
            *fit_distribution* is ``True``, ``'concentration'`` and
            ``'mean_probabilities'``.
        """
        from ..core.cell_type_assignment import compute_cell_type_probabilities

        return compute_cell_type_probabilities(
            results=self,
            counts=counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            dtype=dtype,
            fit_distribution=fit_distribution,
            temperature=temperature,
            weights=weights,
            weight_type=weight_type,
            verbose=verbose,
        )
