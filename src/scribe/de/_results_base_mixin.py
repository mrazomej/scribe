"""Shared methods for DE results objects.

This module contains behavior that is common across all DE result
implementations, independent of how gene-level statistics are produced.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ._error_control import compute_pefp, find_lfsr_threshold, format_de_table
from ._gene_level import call_de_genes

if TYPE_CHECKING:
    import jax.numpy as jnp
    import pandas


class BaseResultsMixin:
    """Shared cache-aware helpers for all DE results classes.

    Notes
    -----
    The concrete dataclass is expected to provide:

    - ``_gene_results`` cache storage.
    - ``_cached_tau`` cache key.
    - a ``gene_level(tau, coordinate)`` method.
    """

    def _ensure_gene_results(self, tau: float = 0.0) -> None:
        """Recompute gene-level results when the cache is stale.

        Parameters
        ----------
        tau : float, default=0.0
            Practical-significance threshold. A cache miss is triggered
            when this differs from the threshold used for the current
            cached results.
        """
        if self._gene_results is None or self._cached_tau != tau:
            self.gene_level(tau=tau)

    def call_genes(
        self,
        tau: float = 0.0,
        lfsr_threshold: float = 0.05,
        prob_effect_threshold: float = 0.95,
    ) -> "jnp.ndarray":
        """Call DE genes using Bayesian decision rules.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold.
        lfsr_threshold : float, default=0.05
            Maximum acceptable local false sign rate.
        prob_effect_threshold : float, default=0.95
            Minimum posterior probability of practical effect.

        Returns
        -------
        jnp.ndarray
            Boolean mask of DE genes with shape ``(D,)``.
        """
        self._ensure_gene_results(tau=tau)
        return call_de_genes(
            self._gene_results,
            lfsr_threshold=lfsr_threshold,
            prob_effect_threshold=prob_effect_threshold,
        )

    def compute_pefp(
        self,
        threshold: float = 0.05,
        tau: float = 0.0,
        use_lfsr_tau: bool = False,
    ) -> float:
        """Compute posterior expected false discovery proportion.

        Parameters
        ----------
        threshold : float, default=0.05
            lfsr threshold for calling genes DE.
        tau : float, default=0.0
            Practical significance threshold.
        use_lfsr_tau : bool, default=False
            If ``True``, use ``lfsr_tau`` instead of ``lfsr``.

        Returns
        -------
        float
            Expected false discovery proportion among called genes.
        """
        self._ensure_gene_results(tau=tau)
        lfsr_key = "lfsr_tau" if use_lfsr_tau else "lfsr"
        return compute_pefp(self._gene_results[lfsr_key], threshold=threshold)

    def find_threshold(
        self,
        target_pefp: float = 0.05,
        tau: float = 0.0,
        use_lfsr_tau: bool = False,
    ) -> float:
        """Find an lfsr threshold that controls PEFP.

        Parameters
        ----------
        target_pefp : float, default=0.05
            Target PEFP level.
        tau : float, default=0.0
            Practical significance threshold.
        use_lfsr_tau : bool, default=False
            If ``True``, use ``lfsr_tau`` instead of ``lfsr``.

        Returns
        -------
        float
            Threshold value for the selected lfsr variant.
        """
        self._ensure_gene_results(tau=tau)
        lfsr_key = "lfsr_tau" if use_lfsr_tau else "lfsr"
        return find_lfsr_threshold(
            self._gene_results[lfsr_key], target_pefp=target_pefp
        )

    def summary(
        self,
        tau: float = 0.0,
        sort_by: str = "lfsr",
        top_n: Optional[int] = 20,
    ) -> str:
        """Format a DE summary table.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold.
        sort_by : str, default='lfsr'
            Column to sort by.
        top_n : int, optional
            Number of rows to display.

        Returns
        -------
        str
            Formatted table representation.
        """
        self._ensure_gene_results(tau=tau)
        return format_de_table(self._gene_results, sort_by=sort_by, top_n=top_n)

    def to_dataframe(
        self,
        tau: float = 0.0,
        target_pefp: Optional[float] = None,
        use_lfsr_tau: bool = True,
    ) -> "pandas.DataFrame":
        """Export cached gene-level statistics to a pandas DataFrame.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold used for gene-level metrics.
        target_pefp : float, optional
            If provided, add an ``is_de`` column based on the threshold
            from :meth:`find_threshold`.
        use_lfsr_tau : bool, default=True
            Select whether PEFP control uses ``lfsr_tau`` or ``lfsr``.

        Returns
        -------
        pandas.DataFrame
            One row per gene with posterior summary metrics.
        """
        import numpy as np
        import pandas as pd

        self._ensure_gene_results(tau=tau)
        gs = self._gene_results
        df = pd.DataFrame(
            {
                "gene": gs["gene_names"],
                "delta_mean": np.asarray(gs["delta_mean"]),
                "delta_sd": np.asarray(gs["delta_sd"]),
                "lfsr": np.asarray(gs["lfsr"]),
                "lfsr_tau": np.asarray(gs["lfsr_tau"]),
                "prob_effect": np.asarray(gs["prob_effect"]),
                "prob_positive": np.asarray(gs["prob_positive"]),
            }
        )

        # Use a PEFP-controlled threshold when requested.
        if target_pefp is not None:
            threshold = self.find_threshold(
                target_pefp=target_pefp, tau=tau, use_lfsr_tau=use_lfsr_tau
            )
            lfsr_col = "lfsr_tau" if use_lfsr_tau else "lfsr"
            df["is_de"] = df[lfsr_col].to_numpy() < threshold

        return df
