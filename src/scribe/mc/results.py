"""ScribeModelComparisonResults: structured results for Bayesian model comparison.

This module defines:

- ``ScribeModelComparisonResults`` — a dataclass that stores raw posterior
  log-likelihoods and provides lazy-computed WAIC, PSIS-LOO, ranking, and
  gene-level comparison methods.
- ``compare_models()`` — factory function that accepts a list of fitted
  ``ScribeSVIResults`` or ``ScribeMCMCResults`` objects, computes log-likelihoods
  for each model, and returns a ``ScribeModelComparisonResults``.

Design decisions
----------------
- Raw log-likelihood matrices ``(S, n_obs)`` are computed once and stored.
  All downstream quantities (WAIC, PSIS-LOO, stacking weights) are derived
  lazily and cached after the first computation.
- WAIC is computed via JAX JIT-compiled functions (fast for large datasets).
- PSIS-LOO uses NumPy/SciPy (Pareto fitting requires scipy); it is the
  recommended criterion for reliable model comparison.
- Gene-level comparison requires gene-level log-likelihoods computed separately
  with ``return_by="gene"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ._waic import compute_waic_stats, pseudo_bma_weights
from ._psis_loo import compute_psis_loo, psis_loo_summary
from ._gene_level import gene_level_comparison, format_gene_comparison_table
from ._stacking import compute_stacking_weights, stacking_summary


# ---------------------------------------------------------------------------
# Results class
# ---------------------------------------------------------------------------


@dataclass
class ScribeModelComparisonResults:
    """Structured results for Bayesian model comparison across K models.

    Stores raw posterior log-likelihood matrices for each model and provides
    methods for computing WAIC, PSIS-LOO, model ranking, and gene-level
    comparisons.  All expensive computations are performed lazily and cached.

    Parameters
    ----------
    model_names : list of str
        Human-readable names for the K models in the comparison.
    log_liks_cell : list of jnp.ndarray
        List of K arrays, each of shape ``(S, C)``, containing the
        per-cell log-likelihoods under each model.  ``S`` is the number of
        posterior samples, ``C`` is the number of cells.
    log_liks_gene : list of jnp.ndarray, optional
        List of K arrays of shape ``(S, G)``, containing per-gene
        log-likelihoods (summed over cells) for each model.  Required for
        :meth:`gene_level_comparison`.  Can be computed by calling
        :func:`compare_models` with ``compute_gene_liks=True``.
    gene_names : list of str, optional
        Names for the G genes.  Used in gene-level comparison output.
    n_cells : int
        Number of cells (observations).
    n_genes : int
        Number of genes.
    dtype : numpy dtype, default=np.float64
        Precision used for PSIS-LOO computations.

    Attributes
    ----------
    K : int
        Number of models.

    Examples
    --------
    >>> from scribe.mc import compare_models
    >>> mc = compare_models(
    ...     [results_nbdm, results_hierarchical],
    ...     counts=counts,
    ...     model_names=["NBDM", "Hierarchical"],
    ...     gene_names=gene_names,
    ...     compute_gene_liks=True,
    ... )
    >>> mc.rank()
    >>> mc.summary()
    >>> mc.gene_level_comparison("NBDM", "Hierarchical")
    """

    # --- Required fields ---
    model_names: List[str]
    log_liks_cell: List[jnp.ndarray]

    # --- Optional fields ---
    log_liks_gene: Optional[List[jnp.ndarray]] = field(default=None, repr=False)
    gene_names: Optional[List[str]] = None
    n_cells: int = 0
    n_genes: int = 0
    dtype: type = field(default=np.float64, repr=False)

    # --- Private cache (not shown in repr) ---
    _waic_cache: Optional[List[dict]] = field(default=None, repr=False, init=False)
    _psis_loo_cache: Optional[List[dict]] = field(default=None, repr=False, init=False)
    _stacking_weights_cache: Optional[np.ndarray] = field(
        default=None, repr=False, init=False
    )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def K(self) -> int:
        """Number of models being compared."""
        return len(self.model_names)

    # ------------------------------------------------------------------
    # WAIC
    # ------------------------------------------------------------------

    def waic(self, model_idx: Optional[int] = None) -> Union[dict, List[dict]]:
        """Compute WAIC statistics for one or all models.

        Results are cached after the first call; repeated calls are free.

        Parameters
        ----------
        model_idx : int, optional
            If provided, return WAIC statistics only for model ``model_idx``.
            If ``None`` (default), return a list of dicts for all K models.

        Returns
        -------
        dict or list of dict
            Each dict contains keys: ``lppd``, ``p_waic_1``, ``p_waic_2``,
            ``elppd_waic_1``, ``elppd_waic_2``, ``waic_1``, ``waic_2``.

        Examples
        --------
        >>> stats_all = mc.waic()
        >>> stats_first = mc.waic(model_idx=0)
        """
        if self._waic_cache is None:
            self._waic_cache = [
                {
                    k: (float(v) if jnp.ndim(v) == 0 else v)
                    for k, v in compute_waic_stats(ll, aggregate=True).items()
                }
                for ll in self.log_liks_cell
            ]
        if model_idx is not None:
            return self._waic_cache[model_idx]
        return self._waic_cache

    # ------------------------------------------------------------------
    # PSIS-LOO
    # ------------------------------------------------------------------

    def psis_loo(self, model_idx: Optional[int] = None) -> Union[dict, List[dict]]:
        """Compute PSIS-LOO statistics for one or all models.

        PSIS-LOO is computed using NumPy/SciPy (Pareto fitting is not JIT-
        compilable).  Results are cached after the first call.

        Parameters
        ----------
        model_idx : int, optional
            If provided, return PSIS-LOO statistics only for model
            ``model_idx``.  If ``None``, return a list for all K models.

        Returns
        -------
        dict or list of dict
            Each dict contains: ``elpd_loo``, ``p_loo``, ``looic``,
            ``elpd_loo_i`` (per-observation), ``k_hat`` (per-observation),
            ``lppd``, ``n_bad`` (number of observations with k̂ ≥ 0.7).

        Examples
        --------
        >>> loo_all = mc.psis_loo()
        >>> print(loo_all[0]["n_bad"])
        """
        if self._psis_loo_cache is None:
            self._psis_loo_cache = [
                compute_psis_loo(np.asarray(ll), dtype=self.dtype)
                for ll in self.log_liks_cell
            ]
        if model_idx is not None:
            return self._psis_loo_cache[model_idx]
        return self._psis_loo_cache

    # ------------------------------------------------------------------
    # Stacking weights
    # ------------------------------------------------------------------

    def stacking_weights(
        self,
        n_restarts: int = 5,
        seed: int = 42,
    ) -> np.ndarray:
        """Compute optimal stacking weights from PSIS-LOO estimates.

        The stacking weights maximize the LOO log predictive score of the
        model ensemble.  They are computed once and cached.

        Parameters
        ----------
        n_restarts : int, default=5
            Number of random restarts for the convex optimization.
        seed : int, default=42
            Random seed.

        Returns
        -------
        np.ndarray, shape ``(K,)``
            Optimal stacking weights summing to 1.
        """
        if self._stacking_weights_cache is None:
            loo_results = self.psis_loo()
            loo_log_i = [r["elpd_loo_i"] for r in loo_results]
            self._stacking_weights_cache = compute_stacking_weights(
                loo_log_i, n_restarts=n_restarts, seed=seed
            )
        return self._stacking_weights_cache

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(
        self,
        criterion: str = "psis_loo",
        include_stacking: bool = True,
    ) -> pd.DataFrame:
        """Rank models by predictive performance.

        Produces a summary DataFrame analogous to ``arviz.compare()``, with
        columns for elpd, effective parameter count, elpd difference from the
        best model, standard error of the difference, and model weights.

        Parameters
        ----------
        criterion : str, default='psis_loo'
            Criterion to use for ranking.  One of:
            - ``'psis_loo'``: PSIS-LOO elpd (recommended).
            - ``'waic_2'``: WAIC using variance-based penalty.
            - ``'waic_1'``: WAIC using bias-corrected penalty.
        include_stacking : bool, default=True
            If ``True``, include stacking weights in the output.

        Returns
        -------
        pd.DataFrame
            Rows are models, sorted by elpd descending (best first).
            Columns:

            ``model``
                Model name.
            ``elpd``
                Expected log predictive density.
            ``p_eff``
                Effective number of parameters.
            ``elpd_diff``
                Difference in elpd from the best model (0 for the best).
            ``elpd_diff_se``
                Standard error of the elpd difference (from pointwise CLT).
            ``z_score``
                ``elpd_diff / elpd_diff_se``.
            ``weight_pseudo_bma``
                Pseudo-BMA (AIC-style) weight.
            ``weight_stacking``
                Optimal stacking weight (only if ``include_stacking=True``).
            ``n_bad_k``
                Number of observations with k̂ ≥ 0.7 (PSIS-LOO only).

        Examples
        --------
        >>> df = mc.rank()
        >>> print(df[["model", "elpd", "elpd_diff", "weight_stacking"]])
        """
        if criterion == "psis_loo":
            loo_results = self.psis_loo()
            elpd_values = np.array([r["elpd_loo"] for r in loo_results])
            p_eff_values = np.array([r["p_loo"] for r in loo_results])
            elpd_pointwise = [r["elpd_loo_i"] for r in loo_results]
            n_bad = [r["n_bad"] for r in loo_results]
        elif criterion in ("waic_2", "waic_1"):
            waic_results = self.waic()
            elppd_key = "elppd_waic_2" if criterion == "waic_2" else "elppd_waic_1"
            p_key = "p_waic_2" if criterion == "waic_2" else "p_waic_1"
            elpd_values = np.array([r[elppd_key] for r in waic_results])
            p_eff_values = np.array([r[p_key] for r in waic_results])
            # For SE computation: use per-observation lppd differences
            # We need pointwise WAIC contributions, so recompute with aggregate=False
            elpd_pointwise = []
            for ll in self.log_liks_cell:
                pw_stats = compute_waic_stats(ll, aggregate=False)
                elpd_pointwise.append(np.asarray(pw_stats[elppd_key]))
            n_bad = [0] * self.K  # n_bad is only meaningful for PSIS-LOO
        else:
            raise ValueError(
                f"Unknown criterion '{criterion}'. "
                "Use 'psis_loo', 'waic_2', or 'waic_1'."
            )

        # Best model index (highest elpd)
        best_idx = int(np.argmax(elpd_values))
        best_pointwise = elpd_pointwise[best_idx]

        # Pairwise elpd differences and SE (relative to best model)
        elpd_diff = elpd_values - elpd_values[best_idx]  # best model has diff=0

        # SE of the difference via pointwise CLT
        elpd_diff_se = np.zeros(self.K)
        z_scores = np.zeros(self.K)
        for k in range(self.K):
            if k == best_idx:
                continue
            d_i = elpd_pointwise[k] - best_pointwise
            se = float(np.sqrt(np.sum((d_i - d_i.mean()) ** 2)))
            elpd_diff_se[k] = se
            z_scores[k] = elpd_diff[k] / se if se > 0 else 0.0

        # Pseudo-BMA weights (from WAIC or LOO)
        # Use the negative elpd scaled as WAIC: IC = -2 * elpd
        ic_values = -2.0 * elpd_values
        wt_pbma = np.asarray(pseudo_bma_weights(jnp.array(ic_values)))

        # Stacking weights (optional, more expensive)
        wt_stack = None
        if include_stacking:
            try:
                wt_stack = self.stacking_weights()
            except Exception:
                # Stacking can fail if LOO densities are degenerate
                wt_stack = wt_pbma.copy()

        # Assemble DataFrame
        records = []
        for k in range(self.K):
            rec = {
                "model": self.model_names[k],
                "elpd": float(elpd_values[k]),
                "p_eff": float(p_eff_values[k]),
                "elpd_diff": float(elpd_diff[k]),
                "elpd_diff_se": float(elpd_diff_se[k]),
                "z_score": float(z_scores[k]),
                "weight_pseudo_bma": float(wt_pbma[k]),
            }
            if include_stacking and wt_stack is not None:
                rec["weight_stacking"] = float(wt_stack[k])
            if criterion == "psis_loo":
                rec["n_bad_k"] = int(n_bad[k])
            records.append(rec)

        df = pd.DataFrame(records)
        df = df.sort_values("elpd", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Gene-level comparison
    # ------------------------------------------------------------------

    def gene_level_comparison(
        self,
        model_A: Union[int, str],
        model_B: Union[int, str],
        gene_names: Optional[List[str]] = None,
        criterion: str = "waic_2",
    ) -> pd.DataFrame:
        """Compare two models gene by gene.

        Computes per-gene elpd differences, standard errors, and z-scores
        using gene-level log-likelihoods (summed over cells).

        Requires that :func:`compare_models` was called with
        ``compute_gene_liks=True``, otherwise raises ``RuntimeError``.

        Parameters
        ----------
        model_A : int or str
            Index or name of model A in the comparison.
        model_B : int or str
            Index or name of model B.
        gene_names : list of str, optional
            Override the stored gene names.
        criterion : str, default='waic_2'
            WAIC variant to use for per-gene elpd.

        Returns
        -------
        pd.DataFrame
            Per-gene comparison table from :func:`~scribe.mc._gene_level.gene_level_comparison`.

        Raises
        ------
        RuntimeError
            If gene-level log-likelihoods are not available.

        Examples
        --------
        >>> df = mc.gene_level_comparison("NBDM", "Hierarchical")
        >>> print(df.head(10))
        """
        if self.log_liks_gene is None:
            raise RuntimeError(
                "Gene-level log-likelihoods are not available.  "
                "Re-run compare_models() with compute_gene_liks=True."
            )

        # Resolve model indices
        idx_A = _resolve_model_idx(model_A, self.model_names)
        idx_B = _resolve_model_idx(model_B, self.model_names)

        names = gene_names or self.gene_names

        return gene_level_comparison(
            log_liks_A=np.asarray(self.log_liks_gene[idx_A]),
            log_liks_B=np.asarray(self.log_liks_gene[idx_B]),
            gene_names=names,
            label_A=self.model_names[idx_A],
            label_B=self.model_names[idx_B],
            criterion=criterion,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self, model_idx: Optional[int] = None) -> str:
        """Format PSIS-LOO diagnostics (k̂ summary) for one or all models.

        Parameters
        ----------
        model_idx : int, optional
            If provided, show diagnostics only for model ``model_idx``.
            Otherwise show diagnostics for all models.

        Returns
        -------
        str
            Multi-line diagnostic summary.
        """
        loo_results = self.psis_loo()
        if model_idx is not None:
            indices = [model_idx]
        else:
            indices = list(range(self.K))

        parts = []
        for k in indices:
            header = f"\n--- {self.model_names[k]} ---"
            parts.append(header)
            parts.append(psis_loo_summary(loo_results[k]))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(
        self,
        criterion: str = "psis_loo",
        include_stacking: bool = True,
    ) -> str:
        """Format a ranked comparison table as a string.

        Parameters
        ----------
        criterion : str, default='psis_loo'
            Ranking criterion: ``'psis_loo'``, ``'waic_2'``, or ``'waic_1'``.
        include_stacking : bool, default=True
            Whether to include stacking weights.

        Returns
        -------
        str
            Formatted comparison table.

        Examples
        --------
        >>> print(mc.summary())
        """
        df = self.rank(criterion=criterion, include_stacking=include_stacking)
        header = f"Model Comparison ({criterion.upper()})\n" + "=" * 60 + "\n"
        return header + df.to_string(index=False)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise representation of the model comparison."""
        return (
            f"ScribeModelComparisonResults("
            f"K={self.K}, "
            f"n_cells={self.n_cells}, "
            f"n_genes={self.n_genes}, "
            f"models={self.model_names})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_model_idx(model: Union[int, str], model_names: List[str]) -> int:
    """Resolve a model identifier to its index.

    Parameters
    ----------
    model : int or str
        Integer index or string name.
    model_names : list of str
        List of model names.

    Returns
    -------
    int
        Model index.

    Raises
    ------
    ValueError
        If the name is not found or the index is out of range.
    """
    if isinstance(model, int):
        if model < 0 or model >= len(model_names):
            raise ValueError(f"Model index {model} out of range (K={len(model_names)}).")
        return model
    if isinstance(model, str):
        if model not in model_names:
            raise ValueError(
                f"Model '{model}' not found. Available: {model_names}."
            )
        return model_names.index(model)
    raise TypeError(f"model must be int or str, got {type(model)}.")


def _get_log_liks(
    results,
    counts: jnp.ndarray,
    return_by: str,
    batch_size: Optional[int],
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Retrieve log-likelihoods from a fitted results object.

    Handles both ``ScribeSVIResults`` (which needs ``get_posterior_samples``)
    and ``ScribeMCMCResults`` (which already has samples from MCMC).

    Parameters
    ----------
    results : ScribeSVIResults or ScribeMCMCResults
        Fitted model results object.
    counts : jnp.ndarray
        Observed count data, shape ``(C, G)``.
    return_by : str
        ``'cell'`` for shape ``(S, C)`` or ``'gene'`` for shape ``(S, G)``.
    batch_size : int, optional
        Mini-batch size for memory-efficient log-likelihood computation.
    dtype : jnp.dtype
        Floating-point precision.

    Returns
    -------
    jnp.ndarray
        Log-likelihood matrix of shape ``(S, C)`` or ``(S, G)``.
    """
    return results.log_likelihood(
        counts,
        batch_size=batch_size,
        return_by=return_by,
        cells_axis=0,
        ignore_nans=False,
        split_components=False,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def compare_models(
    results_list,
    counts: Union[np.ndarray, jnp.ndarray],
    model_names: Optional[List[str]] = None,
    gene_names: Optional[List[str]] = None,
    n_samples: int = 1000,
    rng_key=None,
    batch_size: Optional[int] = None,
    compute_gene_liks: bool = False,
    dtype_lik: jnp.dtype = jnp.float32,
    dtype_psis: type = np.float64,
) -> ScribeModelComparisonResults:
    """Create a model comparison results object from a list of fitted models.

    For each model, this function:

    1. Ensures posterior samples are available (calls
       ``get_posterior_samples`` if needed for SVI models).
    2. Computes the per-cell log-likelihood matrix of shape ``(S, C)``
       using the model's ``log_likelihood`` method.
    3. Optionally computes per-gene log-likelihood matrices of shape
       ``(S, G)`` when ``compute_gene_liks=True``.
    4. Returns a :class:`ScribeModelComparisonResults` that provides lazy
       WAIC, PSIS-LOO, stacking, and gene-level comparison methods.

    Parameters
    ----------
    results_list : list of ScribeSVIResults or ScribeMCMCResults
        List of K fitted model objects to compare.
    counts : array-like, shape ``(C, G)``
        Observed count matrix (cells × genes).
    model_names : list of str, optional
        Human-readable names for each model.  Defaults to
        ``["model_0", "model_1", ...]``.
    gene_names : list of str, optional
        Gene names for gene-level comparisons.
    n_samples : int, default=1000
        Number of posterior samples to draw for SVI models that do not yet
        have ``posterior_samples`` populated.
    rng_key : jax.random.PRNGKey, optional
        Random key for SVI posterior sampling.  Defaults to
        ``jax.random.PRNGKey(0)`` if ``None``.
    batch_size : int, optional
        Mini-batch size for log-likelihood computation.  ``None`` uses the
        full dataset (fast but memory-intensive).
    compute_gene_liks : bool, default=False
        If ``True``, also compute per-gene log-likelihoods (shape ``(S, G)``)
        for gene-level model comparison.  Doubles the computation time.
    dtype_lik : jnp.dtype, default=jnp.float32
        Precision for log-likelihood computation.
    dtype_psis : numpy dtype, default=np.float64
        Precision for PSIS-LOO computation.  Double precision is recommended
        for reliable Pareto fitting.

    Returns
    -------
    ScribeModelComparisonResults
        Structured comparison results with lazy-computed WAIC, PSIS-LOO,
        and stacking weights.

    Examples
    --------
    >>> from scribe.mc import compare_models
    >>> mc = compare_models(
    ...     [results_nbdm, results_hierarchical],
    ...     counts=counts,
    ...     model_names=["NBDM", "Hierarchical"],
    ...     gene_names=gene_names,
    ...     compute_gene_liks=True,
    ... )
    >>> print(mc.summary())
    >>> print(mc.diagnostics())
    """
    from jax import random as jrandom

    K = len(results_list)

    # Default model names
    if model_names is None:
        model_names = [f"model_{k}" for k in range(K)]
    if len(model_names) != K:
        raise ValueError(
            f"model_names has length {len(model_names)} but results_list has {K} models."
        )

    # Default RNG key for SVI sampling
    if rng_key is None:
        rng_key = jrandom.PRNGKey(0)

    # Ensure counts is a JAX array
    counts = jnp.asarray(counts, dtype=dtype_lik)
    n_cells, n_genes = counts.shape

    # Compute per-cell log-likelihoods for each model
    log_liks_cell = []
    log_liks_gene = [] if compute_gene_liks else None

    # Split RNG keys so each model gets an independent key
    rng_keys = jrandom.split(rng_key, K)

    for k, results in enumerate(results_list):
        name = model_names[k]
        print(f"Computing log-likelihoods for {name}...")

        # Ensure posterior samples are available
        if getattr(results, "posterior_samples", None) is None:
            try:
                # SVI: requires rng_key and n_samples
                results.get_posterior_samples(rng_keys[k], n_samples)
            except TypeError:
                # MCMC: no arguments needed (samples come from MCMC run)
                results.get_posterior_samples()

        # Per-cell log-likelihoods: shape (S, C)
        ll_cell = _get_log_liks(results, counts, "cell", batch_size, dtype_lik)
        log_liks_cell.append(ll_cell)

        # Per-gene log-likelihoods: shape (S, G) — optional
        if compute_gene_liks:
            ll_gene = _get_log_liks(results, counts, "gene", batch_size, dtype_lik)
            log_liks_gene.append(ll_gene)

    return ScribeModelComparisonResults(
        model_names=model_names,
        log_liks_cell=log_liks_cell,
        log_liks_gene=log_liks_gene,
        gene_names=gene_names,
        n_cells=n_cells,
        n_genes=n_genes,
        dtype=dtype_psis,
    )
