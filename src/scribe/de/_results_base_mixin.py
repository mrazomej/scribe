"""Shared methods for DE results objects.

This module contains behavior that is common across all DE result
implementations, independent of how gene-level statistics are produced.
"""

from __future__ import annotations

from collections.abc import Iterable
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

    # Supported metric families at the base level (CLR summaries only).
    _DATAFRAME_METRIC_ORDER = ("clr",)
    _SUPPORTED_DATAFRAME_METRICS = frozenset({"clr"})

    def _resolve_dataframe_metrics(
        self,
        metrics: str | Iterable[str] | None,
    ) -> tuple[str, ...]:
        """Normalize and validate metric-family selections for export.

        Parameters
        ----------
        metrics : str or iterable of str, optional
            Requested metric families. ``None`` defaults to ``("clr",)``.
            The ``"all"`` alias expands to all supported families.

        Returns
        -------
        tuple of str
            Ordered tuple of validated metric-family identifiers.

        Raises
        ------
        ValueError
            If the selection is empty or contains unsupported names.
        """
        # Preserve current behavior by defaulting to CLR metrics.
        if metrics is None:
            return ("clr",)

        if isinstance(metrics, str):
            metric_values = [metrics]
        else:
            metric_values = list(metrics)

        if not metric_values:
            raise ValueError("metrics must include at least one metric family.")

        resolved: list[str] = []
        supported = self._SUPPORTED_DATAFRAME_METRICS
        metric_order = tuple(
            metric
            for metric in self._DATAFRAME_METRIC_ORDER
            if metric in supported
        )

        # Expand aliases and preserve order while removing duplicates.
        for metric in metric_values:
            if metric == "all":
                resolved.extend(
                    m for m in metric_order if m not in resolved
                )
                continue

            if metric not in supported:
                raise ValueError(
                    f"Unsupported metrics family '{metric}'. "
                    f"Supported values: {sorted(supported | {'all'})}."
                )
            if metric not in resolved:
                resolved.append(metric)

        return tuple(resolved)

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
        metrics: str | Iterable[str] | None = None,
        column_naming: str = "prefixed",
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
        metrics : {'clr', 'bio_lfc', 'bio_lvr', 'bio_kl', 'bio_aux', 'all'}
            or iterable, optional
            Metric families to export. ``None`` and ``'clr'`` produce the
            current CLR-focused DataFrame. Valid families are:

            - ``'clr'``: compositional CLR gene-level DE summaries
              (``delta_*``, ``lfsr*``, and effect probabilities).
            - ``'bio_lfc'``: biological mean-shift summaries based on
              log-fold-change (``lfc_*`` columns).
            - ``'bio_lvr'``: biological variance-shift summaries based on
              log-variance ratio (``lvr_*`` columns).
            - ``'bio_kl'``: biological distribution-shift summaries from
              Jeffreys divergence (``kl_*`` columns).
            - ``'bio_aux'``: auxiliary biological context columns
              (``mu_*``, ``var_*``, ``max_bio_expr``).
            - ``'all'``: alias that expands to every family supported by the
              concrete results class.

            On this base results class, only ``'clr'`` (and therefore
            ``'all'`` -> ``'clr'``) is supported.
        column_naming : {'prefixed', 'legacy'}, default='prefixed'
            Column naming convention for CLR metrics. ``'prefixed'`` uses
            explicit namespaced columns (for example ``clr_delta_mean`` and
            ``clr_is_de``). ``'legacy'`` preserves historical names
            (for example ``delta_mean`` and ``is_de``).

        Returns
        -------
        pandas.DataFrame
            One row per gene with posterior summary metrics.
        """
        import numpy as np
        import pandas as pd

        metric_families = self._resolve_dataframe_metrics(metrics)
        if column_naming not in {"prefixed", "legacy"}:
            raise ValueError(
                "column_naming must be one of {'prefixed', 'legacy'}."
            )

        # Base results objects only provide CLR gene-level summaries.
        if "clr" not in metric_families:
            raise ValueError(
                "At least one CLR metric family is required for this "
                "results class."
            )

        self._ensure_gene_results(tau=tau)
        gs = self._gene_results
        # Keep CLR column naming explicit while retaining a legacy mode.
        if column_naming == "prefixed":
            clr_columns = {
                "clr_delta_mean": np.asarray(gs["delta_mean"]),
                "clr_delta_sd": np.asarray(gs["delta_sd"]),
                "clr_lfsr": np.asarray(gs["lfsr"]),
                "clr_lfsr_tau": np.asarray(gs["lfsr_tau"]),
                "clr_prob_effect": np.asarray(gs["prob_effect"]),
                "clr_prob_positive": np.asarray(gs["prob_positive"]),
            }
            lfsr_col = "clr_lfsr_tau" if use_lfsr_tau else "clr_lfsr"
            is_de_col = "clr_is_de"
        else:
            clr_columns = {
                "delta_mean": np.asarray(gs["delta_mean"]),
                "delta_sd": np.asarray(gs["delta_sd"]),
                "lfsr": np.asarray(gs["lfsr"]),
                "lfsr_tau": np.asarray(gs["lfsr_tau"]),
                "prob_effect": np.asarray(gs["prob_effect"]),
                "prob_positive": np.asarray(gs["prob_positive"]),
            }
            lfsr_col = "lfsr_tau" if use_lfsr_tau else "lfsr"
            is_de_col = "is_de"

        df = pd.DataFrame(
            {
                "gene": gs["gene_names"],
                **clr_columns,
            }
        )

        # Use a PEFP-controlled threshold when requested.
        if target_pefp is not None:
            threshold = self.find_threshold(
                target_pefp=target_pefp, tau=tau, use_lfsr_tau=use_lfsr_tau
            )
            df[is_de_col] = df[lfsr_col].to_numpy() < threshold

        return df
