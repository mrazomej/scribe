"""Shared methods for DE results objects.

This module contains behavior that is common across all DE result
implementations, independent of how gene-level statistics are produced.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional, TYPE_CHECKING

from ._error_control import compute_pefp, find_lfsr_threshold, format_de_table
from ._gene_level import call_de_genes

if TYPE_CHECKING:
    import jax.numpy as jnp
    import pandas


def _normalize_tau(tau: float | Sequence[float]) -> tuple[float, ...]:
    """Normalize scalar/sequence tau input to a sorted tuple of floats.

    Parameters
    ----------
    tau : float or sequence of float
        Practical-significance threshold(s) requested by the caller.

    Returns
    -------
    tuple of float
        Sorted thresholds used as a stable cache key.

    Raises
    ------
    ValueError
        If the sequence is empty.
    """
    if isinstance(tau, Sequence) and not isinstance(tau, (str, bytes)):
        tau_values = tuple(float(value) for value in tau)
    else:
        tau_values = (float(tau),)

    if not tau_values:
        raise ValueError("tau must contain at least one value.")

    return tuple(sorted(tau_values))


def _format_tau_label(tau: float) -> str:
    """Format tau values into compact, deterministic column labels.

    Parameters
    ----------
    tau : float
        Practical-significance threshold.

    Returns
    -------
    str
        Compact string representation suitable for column naming.
    """
    return format(float(tau), ".6g")


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
                resolved.extend(m for m in metric_order if m not in resolved)
                continue

            if metric not in supported:
                raise ValueError(
                    f"Unsupported metrics family '{metric}'. "
                    f"Supported values: {sorted(supported | {'all'})}."
                )
            if metric not in resolved:
                resolved.append(metric)

        return tuple(resolved)

    def _ensure_gene_results(self, tau: float | Sequence[float] = 0.0) -> None:
        """Recompute gene-level results when the cache is stale.

        Parameters
        ----------
        tau : float or sequence of float, default=0.0
            Practical-significance threshold(s). A cache miss is triggered
            when this differs from the threshold set used for the current
            cached results.
        """
        tau_values = _normalize_tau(tau)
        if self._gene_results is None or self._cached_tau != tau_values:
            # Recompute with normalized tau ordering so cache identity and
            # output column ordering stay deterministic across callers.
            self.gene_level(tau=tau_values)

    def _slice_gene_results_for_tau(self, tau: float) -> dict:
        """Extract a single-threshold view from cached gene-level results.

        Parameters
        ----------
        tau : float
            Practical-significance threshold to extract.

        Returns
        -------
        dict
            Gene-level summary with scalar-threshold fields represented as
            ``(D,)`` vectors.
        """
        gs = self._gene_results
        if gs is None:
            raise RuntimeError("Gene-level results are not initialized.")

        tau_values = tuple(
            float(value) for value in gs.get("tau_values", _normalize_tau(tau))
        )
        if len(tau_values) == 1:
            return gs

        target = float(tau)
        tau_idx = None
        for idx, value in enumerate(tau_values):
            if abs(value - target) <= 1e-12:
                tau_idx = idx
                break
        if tau_idx is None:
            raise ValueError(
                f"Requested tau={target} is not present in cached tau_values "
                f"{tau_values}."
            )

        sliced = dict(gs)
        if getattr(sliced.get("prob_effect"), "ndim", 1) == 2:
            sliced["prob_effect"] = sliced["prob_effect"][:, tau_idx]
        if getattr(sliced.get("lfsr_tau"), "ndim", 1) == 2:
            sliced["lfsr_tau"] = sliced["lfsr_tau"][:, tau_idx]
        sliced["tau_values"] = (tau_values[tau_idx],)
        return sliced

    def _compute_is_de_mask_from_scores(
        self,
        error_scores: "jnp.ndarray",
        target_pefp: float,
    ) -> tuple["jnp.ndarray", float]:
        """Compute PEFP-controlled DE calls from an error-score vector.

        Parameters
        ----------
        error_scores : jnp.ndarray
            Per-gene Bayesian error scores where smaller values are better
            evidence for differential expression.
        target_pefp : float
            Desired posterior expected false discovery proportion.

        Returns
        -------
        tuple of (jnp.ndarray, float)
            Boolean mask of called genes and the selected threshold.
        """
        threshold = find_lfsr_threshold(error_scores, target_pefp=target_pefp)
        is_de = error_scores < threshold
        return is_de, threshold

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
        gene_results = self._slice_gene_results_for_tau(tau=tau)
        return call_de_genes(
            gene_results,
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
        gene_results = self._slice_gene_results_for_tau(tau=tau)
        lfsr_key = "lfsr_tau" if use_lfsr_tau else "lfsr"
        return compute_pefp(gene_results[lfsr_key], threshold=threshold)

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
        gene_results = self._slice_gene_results_for_tau(tau=tau)
        lfsr_key = "lfsr_tau" if use_lfsr_tau else "lfsr"
        return find_lfsr_threshold(
            gene_results[lfsr_key], target_pefp=target_pefp
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
        gene_results = self._slice_gene_results_for_tau(tau=tau)
        return format_de_table(gene_results, sort_by=sort_by, top_n=top_n)

    def to_dataframe(
        self,
        tau: float | Sequence[float] = 0.0,
        target_pefp: Optional[float] = None,
        use_lfsr_tau: bool = True,
        metrics: str | Iterable[str] | None = None,
        column_naming: str = "prefixed",
        tau_format: str = "suffix",
    ) -> "pandas.DataFrame":
        """Export cached gene-level statistics to a pandas DataFrame.

        Parameters
        ----------
        tau : float or sequence of float, default=0.0
            Practical significance threshold(s) used for gene-level metrics.
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
        tau_format : {'suffix', 'multiindex'}, default='suffix'
            Output layout used when multiple tau values are provided.
            ``'suffix'`` appends tau values to flat column names; ``'multiindex'``
            adds a tau level for tau-dependent columns.

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
        if tau_format not in {"suffix", "multiindex"}:
            raise ValueError(
                "tau_format must be one of {'suffix', 'multiindex'}."
            )

        # Base results objects only provide CLR gene-level summaries.
        if "clr" not in metric_families:
            raise ValueError(
                "At least one CLR metric family is required for this "
                "results class."
            )

        self._ensure_gene_results(tau=tau)
        gs = self._gene_results
        tau_values = tuple(
            float(value) for value in gs.get("tau_values", _normalize_tau(tau))
        )
        has_multi_tau = (
            len(tau_values) > 1
            and getattr(gs.get("prob_effect"), "ndim", 1) == 2
            and getattr(gs.get("lfsr_tau"), "ndim", 1) == 2
        )
        # Keep CLR column naming explicit while retaining a legacy mode.
        if not has_multi_tau:
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
                    target_pefp=target_pefp,
                    tau=tau_values[0],
                    use_lfsr_tau=use_lfsr_tau,
                )
                df[is_de_col] = df[lfsr_col].to_numpy() < threshold
            return df

        # The multi-tau export keeps tau-independent quantities flat while
        # expanding practical-significance metrics across the tau axis.
        lfsr_tau_matrix = np.asarray(gs["lfsr_tau"])
        prob_effect_matrix = np.asarray(gs["prob_effect"])

        if tau_format == "suffix":
            if column_naming == "prefixed":
                clr_columns = {
                    "clr_delta_mean": np.asarray(gs["delta_mean"]),
                    "clr_delta_sd": np.asarray(gs["delta_sd"]),
                    "clr_lfsr": np.asarray(gs["lfsr"]),
                    "clr_prob_positive": np.asarray(gs["prob_positive"]),
                }
                lfsr_base = "clr_lfsr_tau_tau"
                prob_effect_base = "clr_prob_effect_tau"
                is_de_base = "clr_is_de_tau"
                lfsr_col = "clr_lfsr"
                is_de_col = "clr_is_de"
            else:
                clr_columns = {
                    "delta_mean": np.asarray(gs["delta_mean"]),
                    "delta_sd": np.asarray(gs["delta_sd"]),
                    "lfsr": np.asarray(gs["lfsr"]),
                    "prob_positive": np.asarray(gs["prob_positive"]),
                }
                lfsr_base = "lfsr_tau_tau"
                prob_effect_base = "prob_effect_tau"
                is_de_base = "is_de_tau"
                lfsr_col = "lfsr"
                is_de_col = "is_de"

            for idx, tau_value in enumerate(tau_values):
                tau_label = _format_tau_label(tau_value)
                clr_columns[f"{lfsr_base}{tau_label}"] = lfsr_tau_matrix[:, idx]
                clr_columns[f"{prob_effect_base}{tau_label}"] = (
                    prob_effect_matrix[:, idx]
                )

            df = pd.DataFrame(
                {
                    "gene": gs["gene_names"],
                    **clr_columns,
                }
            )

            if target_pefp is not None:
                if use_lfsr_tau:
                    for idx, tau_value in enumerate(tau_values):
                        tau_label = _format_tau_label(tau_value)
                        score = lfsr_tau_matrix[:, idx]
                        is_de, _ = self._compute_is_de_mask_from_scores(
                            score, target_pefp=target_pefp
                        )
                        df[f"{is_de_base}{tau_label}"] = np.asarray(
                            is_de, dtype=bool
                        )
                else:
                    is_de, _ = self._compute_is_de_mask_from_scores(
                        np.asarray(gs["lfsr"]),
                        target_pefp=target_pefp,
                    )
                    df[is_de_col] = np.asarray(is_de, dtype=bool)
            return df

        # MultiIndex mode: keep the tau level empty for scalar columns and set
        # it only on practical-significance metrics that vary with tau.
        if column_naming == "prefixed":
            base_cols = {
                ("gene", ""): gs["gene_names"],
                ("clr_delta_mean", ""): np.asarray(gs["delta_mean"]),
                ("clr_delta_sd", ""): np.asarray(gs["delta_sd"]),
                ("clr_lfsr", ""): np.asarray(gs["lfsr"]),
                ("clr_prob_positive", ""): np.asarray(gs["prob_positive"]),
            }
            lfsr_tau_name = "clr_lfsr_tau"
            prob_effect_name = "clr_prob_effect"
            is_de_name = "clr_is_de"
        else:
            base_cols = {
                ("gene", ""): gs["gene_names"],
                ("delta_mean", ""): np.asarray(gs["delta_mean"]),
                ("delta_sd", ""): np.asarray(gs["delta_sd"]),
                ("lfsr", ""): np.asarray(gs["lfsr"]),
                ("prob_positive", ""): np.asarray(gs["prob_positive"]),
            }
            lfsr_tau_name = "lfsr_tau"
            prob_effect_name = "prob_effect"
            is_de_name = "is_de"

        for idx, tau_value in enumerate(tau_values):
            tau_label = _format_tau_label(tau_value)
            base_cols[(lfsr_tau_name, tau_label)] = lfsr_tau_matrix[:, idx]
            base_cols[(prob_effect_name, tau_label)] = prob_effect_matrix[
                :, idx
            ]

        df = pd.DataFrame(base_cols)
        df.columns = pd.MultiIndex.from_tuples(
            df.columns, names=("metric", "tau")
        )

        if target_pefp is not None:
            if use_lfsr_tau:
                for idx, tau_value in enumerate(tau_values):
                    tau_label = _format_tau_label(tau_value)
                    score = lfsr_tau_matrix[:, idx]
                    is_de, _ = self._compute_is_de_mask_from_scores(
                        score, target_pefp=target_pefp
                    )
                    df[(is_de_name, tau_label)] = np.asarray(is_de, dtype=bool)
            else:
                is_de, _ = self._compute_is_de_mask_from_scores(
                    np.asarray(gs["lfsr"]),
                    target_pefp=target_pefp,
                )
                df[(is_de_name, "")] = np.asarray(is_de, dtype=bool)

        return df
