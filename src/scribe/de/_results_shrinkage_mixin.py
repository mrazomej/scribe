"""Shrinkage-specific DE methods layered on empirical results."""

from __future__ import annotations

import jax.numpy as jnp


class ShrinkageResultsMixin:
    """Empirical Bayes shrinkage operations for DE results."""

    def gene_level(
        self,
        tau: float = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level DE via empirical Bayes shrinkage.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold.
        coordinate : str, default='clr'
            Coordinate system. Only ``'clr'`` is supported.

        Returns
        -------
        dict
            Shrunk per-gene DE statistics and shrinkage diagnostics.
        """
        from ._shrinkage import shrinkage_differential_expression
        from ..core._array_dispatch import _array_module

        xp = _array_module(self.delta_samples)

        raw_mean = xp.mean(self.delta_samples, axis=0)
        raw_sd = xp.std(self.delta_samples, axis=0, ddof=1)

        self._cached_tau = tau
        self._gene_results = shrinkage_differential_expression(
            delta_mean=raw_mean,
            delta_sd=raw_sd,
            tau=tau,
            gene_names=self.gene_names,
            sigma_grid=self.sigma_grid,
            max_iter=self.shrinkage_max_iter,
            tol=self.shrinkage_tol,
        )

        self.null_proportion = self._gene_results.get("null_proportion")
        self.prior_weights = self._gene_results.get("prior_weights")

        return self._gene_results

    def set_gene_mask(self, mask: jnp.ndarray) -> "ScribeShrinkageDEResults":
        """Apply a mask and invalidate fitted shrinkage parameters.

        Parameters
        ----------
        mask : jnp.ndarray
            Boolean mask over full gene space.

        Returns
        -------
        ScribeShrinkageDEResults
            Returns ``self`` for method chaining.
        """
        super().set_gene_mask(mask)
        self.null_proportion = None
        self.prior_weights = None
        return self

    def clear_mask(self) -> "ScribeShrinkageDEResults":
        """Clear mask and invalidate fitted shrinkage parameters.

        Returns
        -------
        ScribeShrinkageDEResults
            Returns ``self`` for method chaining.
        """
        super().clear_mask()
        self.null_proportion = None
        self.prior_weights = None
        return self

    def __repr__(self) -> str:
        """Return a concise representation of shrinkage comparison."""
        null_str = (
            f", null_proportion={self.null_proportion:.2%}"
            if self.null_proportion is not None
            else ""
        )
        return (
            f"ScribeShrinkageDEResults("
            f"D={self.D}, "
            f"n_samples={self.n_samples}"
            f"{null_str}, "
            f"labels='{self.label_A}' vs '{self.label_B}')"
        )
