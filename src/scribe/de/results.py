"""ScribeDEResults: structured results for differential expression analysis.

This module defines the DE results class hierarchy:

- ``ScribeDEResults`` — abstract base class with shared error-control and
  formatting methods (``call_genes``, ``compute_pefp``, ``find_threshold``,
  ``summary``).  Declares ``gene_level()`` as the method each subclass
  must implement.
- ``ScribeParametricDEResults`` — analytic Gaussian path using low-rank
  ALR parameters (``mu``, ``W``, ``d``).
- ``ScribeEmpiricalDEResults`` — non-parametric path using Monte Carlo
  posterior samples (``delta_samples`` in CLR space).

The companion factory function ``compare()`` constructs the appropriate
subclass, handling parameter extraction, Dirichlet sampling, and CLR
differencing internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import jax.numpy as jnp

from ._extract import extract_alr_params
from ._gene_level import differential_expression, call_de_genes
from ._set_level import test_contrast, test_gene_set
from ._error_control import (
    compute_pefp,
    find_lfsr_threshold,
    format_de_table,
)


# --------------------------------------------------------------------------
# Results-object input extraction
# --------------------------------------------------------------------------


def _is_results_object(obj) -> bool:
    """Check if an object is a scribe results object (SVI, MCMC, or VAE)."""
    return (
        hasattr(obj, "posterior_samples")
        and hasattr(obj, "model_config")
        and hasattr(obj, "var")
    )


def _extract_de_inputs(results, component=None):
    """Extract DE-relevant arrays from a scribe results object.

    Reads the posterior samples from a ``ScribeSVIResults`` or
    ``ScribeMCMCResults`` object and returns the arrays that
    ``compute_clr_differences`` needs, automatically detecting
    hierarchical (gene-specific p) models.

    Parameters
    ----------
    results : ScribeSVIResults or ScribeMCMCResults
        A fitted results object with ``posterior_samples`` populated.
    component : int, optional
        Mixture component index.  Not applied here — passed through
        so that ``compute_clr_differences`` can handle the slicing.

    Returns
    -------
    r_samples : jnp.ndarray
        Posterior samples of the dispersion parameter ``r``.
        Shape ``(N, D)`` or ``(N, K, D)`` for mixture models.
    p_samples : jnp.ndarray or None
        Posterior samples of the gene-specific success probability
        ``p`` if the model is hierarchical; ``None`` otherwise.
    gene_names : list of str or None
        Gene names from ``results.var.index``, or ``None`` if
        ``results.var`` is not available.

    Raises
    ------
    ValueError
        If ``posterior_samples`` is ``None`` (samples not yet drawn).
    """
    if results.posterior_samples is None:
        raise ValueError(
            "Results object has no posterior samples. "
            "Call results.get_posterior_samples() first."
        )

    r_samples = results.posterior_samples["r"]

    # Hierarchical models have gene-specific p stored as a
    # numpyro.deterministic site (derived from phi for mean_odds).
    p_samples = None
    if results.model_config.is_hierarchical:
        p_samples = results.posterior_samples.get("p")

    gene_names = None
    if results.var is not None:
        gene_names = results.var.index.tolist()

    return r_samples, p_samples, gene_names


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------


@dataclass
class ScribeDEResults:
    """Abstract base class for Bayesian differential expression results.

    Holds metadata shared by all DE methods (gene names, condition labels)
    and provides common error-control and formatting methods that delegate
    to the subclass-specific ``gene_level()`` implementation.

    Subclasses must implement ``gene_level(tau, coordinate)`` which returns
    a dict with at least the keys ``delta_mean``, ``delta_sd``,
    ``prob_positive``, ``prob_effect``, ``lfsr``, ``lfsr_tau``,
    ``gene_names``.

    Parameters
    ----------
    gene_names : list of str, optional
        Gene names.  If ``None``, generic names ``gene_0, gene_1, ...``
        are generated.
    label_A : str
        Human-readable label for condition A.
    label_B : str
        Human-readable label for condition B.
    method : str
        Identifier for the DE method (``"parametric"`` or ``"empirical"``).
    """

    # --- Metadata ---
    gene_names: Optional[List[str]] = None
    label_A: str = "A"
    label_B: str = "B"
    method: str = "parametric"

    # --- Cached gene-level results (computed lazily) ---
    _gene_results: Optional[dict] = field(default=None, repr=False, init=False)
    # Track which tau produced the cache so a different tau triggers recompute.
    _cached_tau: Optional[float] = field(default=None, repr=False, init=False)

    # ------------------------------------------------------------------
    # Properties (subclasses may override)
    # ------------------------------------------------------------------

    @property
    def D(self) -> int:
        """Number of genes (CLR dimensionality)."""
        raise NotImplementedError("Subclasses must implement D.")

    @property
    def D_alr(self) -> int:
        """Dimensionality in ALR space (D - 1)."""
        return self.D - 1

    # ------------------------------------------------------------------
    # Abstract method
    # ------------------------------------------------------------------

    def gene_level(
        self,
        tau: float = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level differential expression statistics.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold (log-scale).
        coordinate : str, default='clr'
            Coordinate system for results.

        Returns
        -------
        dict
            Gene-level DE results with keys ``delta_mean``, ``delta_sd``,
            ``prob_positive``, ``prob_effect``, ``lfsr``, ``lfsr_tau``,
            ``gene_names``.
        """
        raise NotImplementedError("Subclasses must implement gene_level().")

    # ------------------------------------------------------------------
    # Cache helper
    # ------------------------------------------------------------------

    def _ensure_gene_results(self, tau: float = 0.0) -> None:
        """Recompute gene-level results if cache is missing or stale.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold.  If this differs from the
            tau used to compute the currently cached results, the cache
            is invalidated and results are recomputed.
        """
        if self._gene_results is None or self._cached_tau != tau:
            self.gene_level(tau=tau)

    # ------------------------------------------------------------------
    # Shared methods (work on any subclass via gene_level())
    # ------------------------------------------------------------------

    def call_genes(
        self,
        tau: float = 0.0,
        lfsr_threshold: float = 0.05,
        prob_effect_threshold: float = 0.95,
    ) -> jnp.ndarray:
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
        ndarray of bool, shape ``(D,)``
            Boolean mask of DE genes.
        """
        self._ensure_gene_results(tau=tau)
        return call_de_genes(
            self._gene_results,
            lfsr_threshold=lfsr_threshold,
            prob_effect_threshold=prob_effect_threshold,
        )

    # ------------------------------------------------------------------

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
            If ``True``, use ``lfsr_tau`` instead of standard ``lfsr``.

        Returns
        -------
        float
            Expected false discovery proportion.
        """
        self._ensure_gene_results(tau=tau)
        lfsr_key = "lfsr_tau" if use_lfsr_tau else "lfsr"
        return compute_pefp(self._gene_results[lfsr_key], threshold=threshold)

    # ------------------------------------------------------------------

    def find_threshold(
        self,
        target_pefp: float = 0.05,
        tau: float = 0.0,
        use_lfsr_tau: bool = False,
    ) -> float:
        """Find lfsr threshold controlling PEFP at target level.

        Parameters
        ----------
        target_pefp : float, default=0.05
            Target PEFP level.
        tau : float, default=0.0
            Practical significance threshold.
        use_lfsr_tau : bool, default=False
            If ``True``, use ``lfsr_tau`` instead of standard ``lfsr``.

        Returns
        -------
        float
            lfsr threshold.
        """
        self._ensure_gene_results(tau=tau)
        lfsr_key = "lfsr_tau" if use_lfsr_tau else "lfsr"
        return find_lfsr_threshold(
            self._gene_results[lfsr_key], target_pefp=target_pefp
        )

    # ------------------------------------------------------------------

    def summary(
        self,
        tau: float = 0.0,
        sort_by: str = "lfsr",
        top_n: Optional[int] = 20,
    ) -> str:
        """Format a summary table of DE results.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold.
        sort_by : str, default='lfsr'
            Column to sort by.
        top_n : int, optional
            Number of top genes to display.

        Returns
        -------
        str
            Formatted table.
        """
        self._ensure_gene_results(tau=tau)
        return format_de_table(self._gene_results, sort_by=sort_by, top_n=top_n)


# --------------------------------------------------------------------------
# Parametric (Gaussian) subclass
# --------------------------------------------------------------------------


@dataclass
class ScribeParametricDEResults(ScribeDEResults):
    """Parametric DE results using analytic Gaussian posteriors.

    Stores low-rank ALR parameters ``(mu, W, d)`` for each condition and
    computes gene-level statistics via exact closed-form Gaussian integrals.

    Parameters
    ----------
    mu_A : jnp.ndarray
        Mean of model A in (D-1)-dimensional ALR space.
    W_A : jnp.ndarray
        Low-rank factor of model A, shape ``(D-1, k_A)``.
    d_A : jnp.ndarray
        Diagonal covariance of model A, shape ``(D-1,)``.
    mu_B : jnp.ndarray
        Mean of model B in (D-1)-dimensional ALR space.
    W_B : jnp.ndarray
        Low-rank factor of model B, shape ``(D-1, k_B)``.
    d_B : jnp.ndarray
        Diagonal covariance of model B, shape ``(D-1,)``.
    gene_names : list of str, optional
        Gene names for output.
    label_A : str
        Human-readable label for condition A.
    label_B : str
        Human-readable label for condition B.

    Examples
    --------
    >>> from scribe.de import compare
    >>> de = compare(model_A, model_B, gene_names=gene_names)
    >>> results = de.gene_level(tau=jnp.log(1.1))
    >>> de.summary()
    """

    # --- Internal ALR parameters (D-1 dimensional) ---
    mu_A: jnp.ndarray = field(default=None)
    W_A: jnp.ndarray = field(default=None)
    d_A: jnp.ndarray = field(default=None)
    mu_B: jnp.ndarray = field(default=None)
    W_B: jnp.ndarray = field(default=None)
    d_B: jnp.ndarray = field(default=None)

    # --- Gene filtering ---
    _drop_last_gene: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Set method identifier."""
        self.method = "parametric"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def D_alr(self) -> int:
        """Dimensionality in ALR space (D-1)."""
        if self._drop_last_gene:
            return self.mu_A.shape[-1] - 1
        return self.mu_A.shape[-1]

    @property
    def D(self) -> int:
        """Number of genes (CLR dimensionality, excluding 'other')."""
        return self.D_alr + 1

    @property
    def D_full(self) -> int:
        """Full CLR dimensionality including 'other' pseudo-gene."""
        return self.mu_A.shape[-1] + 1

    # ------------------------------------------------------------------
    # Gene-level analysis (analytic Gaussian)
    # ------------------------------------------------------------------

    def gene_level(
        self,
        tau: float = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level DE via analytic Gaussian posteriors.

        Wraps ``differential_expression()`` using the stored ALR parameters.
        Results are cached and keyed by ``tau``; calling with a different
        ``tau`` automatically recomputes.

        When a ``gene_mask`` was used during model fitting (i.e. filtered
        genes were pooled into an "other" pseudo-gene), the "other"
        column is automatically dropped from the output.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold (log-scale).
        coordinate : str, default='clr'
            Coordinate system.  Currently only ``'clr'`` is supported.

        Returns
        -------
        dict
            Gene-level DE results (see ``differential_expression``
            for keys).
        """
        # Build internal dict representations for the DE functions
        model_A_dict = {
            "loc": self.mu_A,
            "cov_factor": self.W_A,
            "cov_diag": self.d_A,
        }
        model_B_dict = {
            "loc": self.mu_B,
            "cov_factor": self.W_B,
            "cov_diag": self.d_B,
        }

        # Compute on the full (D_kept + 1) simplex (including "other")
        self._cached_tau = tau
        results = differential_expression(
            model_A_dict,
            model_B_dict,
            tau=tau,
            coordinate=coordinate,
            gene_names=None,  # we manage names ourselves
        )

        # Drop "other" pseudo-gene (last column) when gene_mask was used
        if self._drop_last_gene:
            for key in (
                "delta_mean",
                "delta_sd",
                "prob_positive",
                "prob_effect",
                "lfsr",
                "lfsr_tau",
            ):
                if key in results:
                    results[key] = results[key][:-1]
            # Drop the last auto-generated name
            if "gene_names" in results:
                results["gene_names"] = results["gene_names"][:-1]

        # Overwrite gene_names with the stored names
        results["gene_names"] = list(self.gene_names)
        self._gene_results = results
        return self._gene_results

    # ------------------------------------------------------------------
    # Set-level analysis (analytic, covariance-based)
    # ------------------------------------------------------------------

    def test_gene_set(
        self,
        gene_set_indices: jnp.ndarray,
        tau: float = 0.0,
    ) -> dict:
        """Test enrichment of a gene set using a compositional balance.

        Parameters
        ----------
        gene_set_indices : ndarray of int
            Indices of genes in the set.
        tau : float, default=0.0
            Practical significance threshold.

        Returns
        -------
        dict
            Posterior inference for the gene-set balance.
        """
        model_A_dict = {
            "loc": self.mu_A,
            "cov_factor": self.W_A,
            "cov_diag": self.d_A,
        }
        model_B_dict = {
            "loc": self.mu_B,
            "cov_factor": self.W_B,
            "cov_diag": self.d_B,
        }
        return test_gene_set(model_A_dict, model_B_dict, gene_set_indices, tau)

    # ------------------------------------------------------------------

    def test_contrast(
        self,
        contrast: jnp.ndarray,
        tau: float = 0.0,
    ) -> dict:
        """Test a linear contrast ``c^T Delta`` (analytic Gaussian).

        Parameters
        ----------
        contrast : ndarray, shape ``(D,)``
            Contrast vector in CLR space.
        tau : float, default=0.0
            Practical significance threshold.

        Returns
        -------
        dict
            Posterior inference for the contrast.
        """
        model_A_dict = {
            "loc": self.mu_A,
            "cov_factor": self.W_A,
            "cov_diag": self.d_A,
        }
        model_B_dict = {
            "loc": self.mu_B,
            "cov_factor": self.W_B,
            "cov_diag": self.d_B,
        }
        return test_contrast(model_A_dict, model_B_dict, contrast, tau)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise representation of the parametric DE comparison."""
        return (
            f"ScribeParametricDEResults("
            f"D={self.D}, "
            f"rank_A={self.W_A.shape[-1]}, "
            f"rank_B={self.W_B.shape[-1]}, "
            f"labels='{self.label_A}' vs '{self.label_B}')"
        )


# --------------------------------------------------------------------------
# Empirical (sample-based) subclass
# --------------------------------------------------------------------------


@dataclass
class ScribeEmpiricalDEResults(ScribeDEResults):
    """Non-parametric DE results from posterior samples.

    Stores Monte Carlo CLR differences ``delta_samples`` of shape ``(N, D)``
    and computes gene-level statistics by counting (no Gaussian assumption).

    Parameters
    ----------
    delta_samples : jnp.ndarray, shape ``(N, D)``
        CLR-space posterior differences ``CLR(rho_A) - CLR(rho_B)`` for
        N paired posterior draws and D genes.
    gene_names : list of str, optional
        Gene names for output.
    label_A : str
        Human-readable label for condition A.
    label_B : str
        Human-readable label for condition B.
    n_samples : int
        Number of posterior samples used (informational).

    Attributes
    ----------
    D : int
        Number of genes.
    n_samples : int
        Number of posterior samples.

    Examples
    --------
    >>> from scribe.de import compare
    >>> de = compare(
    ...     r_samples_bleo, r_samples_ctrl,
    ...     method="empirical",
    ...     component_A=0, component_B=0,
    ...     gene_names=gene_names,
    ... )
    >>> results = de.gene_level(tau=jnp.log(1.1))
    >>> de.summary()
    """

    # --- Posterior CLR differences ---
    delta_samples: jnp.ndarray = field(default=None, repr=False)

    # --- Informational ---
    n_samples: int = field(default=0, repr=True)

    def __post_init__(self):
        """Set method identifier and n_samples from delta_samples."""
        self.method = "empirical"
        if self.delta_samples is not None:
            self.n_samples = self.delta_samples.shape[0]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def D(self) -> int:
        """Number of genes (CLR dimensionality)."""
        return self.delta_samples.shape[1]

    # ------------------------------------------------------------------
    # Gene-level analysis (empirical counting)
    # ------------------------------------------------------------------

    def gene_level(
        self,
        tau: float = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level DE via empirical Monte Carlo estimation.

        All statistics are computed by counting over the ``N`` posterior
        CLR difference samples — no distributional assumptions.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold (log-scale).
        coordinate : str, default='clr'
            Coordinate system.  Only ``'clr'`` is supported (samples are
            already in CLR space).

        Returns
        -------
        dict
            Gene-level DE results with the same keys as the parametric
            path: ``delta_mean``, ``delta_sd``, ``prob_positive``,
            ``prob_effect``, ``lfsr``, ``lfsr_tau``, ``gene_names``.
        """
        from ._empirical import empirical_differential_expression

        # Compute, record the tau that was used, and cache the results
        self._cached_tau = tau
        self._gene_results = empirical_differential_expression(
            self.delta_samples,
            tau=tau,
            gene_names=self.gene_names,
        )
        return self._gene_results

    # ------------------------------------------------------------------
    # Set-level analysis (empirical, sample-based)
    # ------------------------------------------------------------------

    def test_contrast(
        self,
        contrast: jnp.ndarray,
        tau: float = 0.0,
    ) -> dict:
        """Test a linear contrast ``c^T Delta`` via posterior samples.

        Projects each of the ``N`` CLR difference vectors onto the contrast
        ``c`` and computes empirical statistics on the resulting scalar
        posterior samples.

        Parameters
        ----------
        contrast : ndarray, shape ``(D,)``
            Contrast vector in CLR space.
        tau : float, default=0.0
            Practical significance threshold.

        Returns
        -------
        dict
            Posterior inference for the contrast with keys:
            ``contrast_mean``, ``contrast_sd``, ``prob_positive``,
            ``prob_effect``, ``lfsr``, ``lfsr_tau``.
        """
        # Project each posterior sample onto the contrast: (N,) = (N, D) @ (D,)
        contrast_samples = self.delta_samples @ contrast

        # Posterior mean and standard deviation
        contrast_mean = float(jnp.mean(contrast_samples))
        contrast_sd = float(jnp.std(contrast_samples, ddof=1))

        # Posterior probabilities by counting
        prob_positive = float(jnp.mean(contrast_samples > 0))
        prob_up = float(jnp.mean(contrast_samples > tau))
        prob_down = float(jnp.mean(contrast_samples < -tau))
        prob_effect = prob_up + prob_down

        # Local false sign rate
        lfsr = min(prob_positive, 1.0 - prob_positive)
        lfsr_tau = 1.0 - max(prob_up, prob_down)

        return {
            "contrast_mean": contrast_mean,
            "contrast_sd": contrast_sd,
            "prob_positive": prob_positive,
            "prob_effect": prob_effect,
            "lfsr": lfsr,
            "lfsr_tau": lfsr_tau,
        }

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise representation of the empirical DE comparison."""
        return (
            f"ScribeEmpiricalDEResults("
            f"D={self.D}, "
            f"n_samples={self.n_samples}, "
            f"labels='{self.label_A}' vs '{self.label_B}')"
        )


# --------------------------------------------------------------------------
# Shrinkage (empirical Bayes) subclass
# --------------------------------------------------------------------------


@dataclass
class ScribeShrinkageDEResults(ScribeEmpiricalDEResults):
    """Empirical Bayes shrinkage DE results.

    Extends ``ScribeEmpiricalDEResults`` by applying a scale-mixture-of-normals
    empirical Bayes layer on top of the raw Monte Carlo CLR differences.  The
    shrinkage uses the genome-wide distribution of effects to improve per-gene
    lfsr estimates by encoding the assumption that most genes are not
    differentially expressed.

    The ``gene_level()`` method returns *shrunk* statistics (posterior mean, sd,
    lfsr, etc.) that account for the global null proportion. All base-class
    methods (``call_genes``, ``compute_pefp``, ``find_threshold``, ``summary``)
    delegate to ``gene_level()`` and therefore automatically use the shrunk
    values.

    Parameters
    ----------
    delta_samples : jnp.ndarray, shape ``(N, D)``
        CLR-space posterior differences (inherited from parent).
    gene_names : list of str, optional
        Gene names for output.
    label_A : str
        Human-readable label for condition A.
    label_B : str
        Human-readable label for condition B.
    sigma_grid : jnp.ndarray, optional
        Prior scale grid.  If ``None``, a default grid is constructed
        from the data.
    shrinkage_max_iter : int
        Maximum number of EM iterations.
    shrinkage_tol : float
        EM convergence tolerance.

    Attributes
    ----------
    null_proportion : float or None
        Estimated fraction of truly null genes (set after first
        ``gene_level()`` call).
    prior_weights : jnp.ndarray or None
        Estimated mixture weights (set after first ``gene_level()``
        call).

    Examples
    --------
    >>> from scribe.de import compare
    >>> de = compare(
    ...     r_samples_bleo, r_samples_ctrl,
    ...     method="shrinkage",
    ...     component_A=0, component_B=0,
    ...     gene_names=gene_names,
    ... )
    >>> results = de.gene_level(tau=jnp.log(1.1))
    >>> print(f"Null proportion: {de.null_proportion:.2%}")
    """

    # --- Shrinkage configuration ---
    sigma_grid: Optional[jnp.ndarray] = field(default=None, repr=False)
    shrinkage_max_iter: int = field(default=200, repr=False)
    shrinkage_tol: float = field(default=1e-8, repr=False)

    # --- Fitted shrinkage results (populated after gene_level()) ---
    null_proportion: Optional[float] = field(
        default=None, repr=True, init=False
    )
    prior_weights: Optional[jnp.ndarray] = field(
        default=None, repr=False, init=False
    )

    def __post_init__(self):
        """Set method identifier."""
        super().__post_init__()
        self.method = "shrinkage"

    # ------------------------------------------------------------------
    # Gene-level analysis (shrinkage)
    # ------------------------------------------------------------------

    def gene_level(
        self,
        tau: float = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level DE via empirical Bayes shrinkage.

        First computes raw empirical summary statistics from the posterior CLR
        difference samples, then applies the scale-mixture shrinkage to produce
        improved estimates.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold (log-scale).
        coordinate : str, default='clr'
            Coordinate system.  Only ``'clr'`` is supported.

        Returns
        -------
        dict
            Gene-level DE results with shrunk estimates.  Contains all
            standard keys (``delta_mean``, ``delta_sd``, ``lfsr``, etc.)
            plus shrinkage extras (``null_proportion``,
            ``prior_weights``, ``sigma_grid``, ``em_converged``,
            ``em_n_iter``, ``em_log_likelihood``).
        """
        from ._shrinkage import shrinkage_differential_expression

        # Raw empirical summary stats from the delta_samples
        raw_mean = jnp.mean(self.delta_samples, axis=0)
        raw_sd = jnp.std(self.delta_samples, axis=0, ddof=1)

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

        # Store fitted shrinkage metadata on the results object
        self.null_proportion = self._gene_results.get("null_proportion")
        self.prior_weights = self._gene_results.get("prior_weights")

        return self._gene_results

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise representation of the shrinkage DE comparison."""
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


# --------------------------------------------------------------------------
# Factory function
# --------------------------------------------------------------------------


def compare(
    model_A,
    model_B,
    gene_names: Optional[List[str]] = None,
    label_A: str = "A",
    label_B: str = "B",
    method: str = "parametric",
    # --- Empirical-only parameters ---
    component_A: Optional[int] = None,
    component_B: Optional[int] = None,
    paired: bool = False,
    n_samples_dirichlet: int = 1,
    rng_key=None,
    batch_size: int = 2048,
    # --- Gene filtering ---
    gene_mask: Optional[jnp.ndarray] = None,
    # --- Hierarchical model: gene-specific p ---
    p_samples_A: Optional[jnp.ndarray] = None,
    p_samples_B: Optional[jnp.ndarray] = None,
    # --- Shrinkage parameters ---
    sigma_grid: Optional[jnp.ndarray] = None,
    shrinkage_max_iter: int = 200,
    shrinkage_tol: float = 1e-8,
) -> ScribeDEResults:
    """Create a DE results object from two fitted models or posterior samples.

    This factory function dispatches to either the parametric (Gaussian)
    or empirical (sample-based) DE path depending on ``method``.

    **Results-object interface (recommended):** For ``method="empirical"``
    or ``method="shrinkage"``, pass ``ScribeSVIResults`` or
    ``ScribeMCMCResults`` objects directly.  The function auto-extracts
    ``r`` and ``p`` posterior samples, detects hierarchical models, and
    infers gene names from ``results.var.index``.

    Parameters
    ----------
    model_A : ScribeSVIResults, ScribeMCMCResults, dict, Distribution, or ndarray
        A scribe results object (recommended for empirical/shrinkage),
        or raw inputs: dict/Distribution for ``method="parametric"``,
        ndarray for ``method="empirical"``/``"shrinkage"``.
    model_B : ScribeSVIResults, ScribeMCMCResults, dict, Distribution, or ndarray
        Same as ``model_A`` for condition B.
    gene_names : list of str, optional
        Gene names.  If ``None``, generic names are generated.
    label_A : str, default='A'
        Human-readable label for condition A.
    label_B : str, default='B'
        Human-readable label for condition B.
    method : str, default='parametric'
        DE method.  ``"parametric"`` uses analytic Gaussian posteriors;
        ``"empirical"`` uses Monte Carlo counting on posterior samples;
        ``"shrinkage"`` runs the empirical pipeline then applies
        empirical Bayes shrinkage via a scale mixture of normals.
    component_A : int, optional
        For ``method="empirical"`` with mixture models: index of the
        component to use from model A.  Required if ``model_A`` is 3D.
    component_B : int, optional
        For ``method="empirical"`` with mixture models: index of the
        component to use from model B.  Required if ``model_B`` is 3D.
    paired : bool, default=False
        For ``method="empirical"``: if ``True``, preserve sample pairing
        (required for within-mixture comparisons where both conditions
        come from the same posterior).  Uses the same RNG sub-key per
        sample index for Dirichlet draws.
    n_samples_dirichlet : int, default=1
        For ``method="empirical"``: number of Dirichlet draws per
        posterior sample.
    rng_key : jax.random.PRNGKey, optional
        For ``method="empirical"``: JAX PRNG key.  If ``None``, uses
        ``jax.random.PRNGKey(0)``.
    batch_size : int, default=2048
        For ``method="empirical"``: batch size for Dirichlet sampling.
    gene_mask : jnp.ndarray, shape ``(D,)``, optional
        Boolean mask selecting genes to keep.  Genes marked ``False``
        are aggregated into a single "other" pseudo-gene before
        Dirichlet sampling and CLR/ALR transformation.  For the
        empirical path, aggregation happens inside
        ``compute_clr_differences``; for the parametric path, the user
        should pass ``gene_mask`` to ``fit_logistic_normal`` before
        calling ``compare``.  Gene names are filtered to match the
        kept genes.  If ``None`` (default), all genes are used.
    p_samples_A : jnp.ndarray, optional
        For ``method="empirical"`` with hierarchical models: posterior
        samples of gene-specific success probabilities for condition A.
        Shape ``(N, D)`` or ``(N, K, D)``.  When provided, Gamma-based
        composition sampling is used instead of Dirichlet.
    p_samples_B : jnp.ndarray, optional
        Same as ``p_samples_A`` for condition B.
    sigma_grid : jnp.ndarray, shape ``(K+1,)``, optional
        For ``method="shrinkage"``: grid of prior standard deviations
        for the scale mixture.  If ``None``, a default geometric grid
        is constructed from the data.
    shrinkage_max_iter : int, default=200
        For ``method="shrinkage"``: maximum EM iterations.
    shrinkage_tol : float, default=1e-8
        For ``method="shrinkage"``: EM convergence tolerance.

    Returns
    -------
    ScribeDEResults
        ``ScribeParametricDEResults``, ``ScribeEmpiricalDEResults``,
        or ``ScribeShrinkageDEResults`` depending on ``method``.

    Examples
    --------
    Parametric (existing behaviour):

    >>> de = compare(fitted_A, fitted_B, gene_names=names)
    >>> de.gene_level(tau=0.1)

    Empirical (independent models):

    >>> de = compare(
    ...     r_samples_A, r_samples_B,
    ...     method="empirical",
    ...     component_A=0, component_B=0,
    ...     gene_names=names,
    ... )

    Empirical (within-mixture, paired):

    >>> de = compare(
    ...     r_samples, r_samples,
    ...     method="empirical",
    ...     component_A=0, component_B=1,
    ...     paired=True,
    ...     gene_names=names,
    ... )

    Empirical (hierarchical model with gene-specific p):

    >>> de = compare(
    ...     r_samples_A, r_samples_B,
    ...     method="empirical",
    ...     component_A=0, component_B=0,
    ...     p_samples_A=p_A, p_samples_B=p_B,
    ...     gene_names=names,
    ... )

    Shrinkage (empirical Bayes):

    >>> de = compare(
    ...     r_samples_A, r_samples_B,
    ...     method="shrinkage",
    ...     component_A=0, component_B=0,
    ...     gene_names=names,
    ... )
    >>> print(f"Null proportion: {de.null_proportion:.2%}")

    Results-object interface (recommended for empirical/shrinkage):

    >>> de = compare(
    ...     results_bleo, results_ctrl,
    ...     method="empirical",
    ...     component_A=0, component_B=0,
    ... )

    The function auto-detects hierarchical models and extracts
    ``r``, ``p``, and gene names from the results objects.
    """
    # ------------------------------------------------------------------
    # Results-object dispatch: when model_A and model_B are scribe
    # results objects (ScribeSVIResults, ScribeMCMCResults, etc.),
    # auto-extract r_samples, p_samples, and gene_names.
    # ------------------------------------------------------------------
    _a_is_results = _is_results_object(model_A)
    _b_is_results = _is_results_object(model_B)

    if _a_is_results or _b_is_results:
        if not (_a_is_results and _b_is_results):
            raise TypeError(
                "Both model_A and model_B must be results objects, "
                "or both must be raw arrays/dicts. "
                f"Got types: {type(model_A).__name__}, "
                f"{type(model_B).__name__}."
            )

        if method == "parametric":
            raise ValueError(
                "method='parametric' requires pre-fitted logistic-normal "
                "models (dicts or distributions), not results objects. "
                "Use method='empirical' or method='shrinkage' with "
                "results objects."
            )

        # Extract arrays from results objects
        r_A, p_A, names_A = _extract_de_inputs(model_A, component_A)
        r_B, p_B, names_B = _extract_de_inputs(model_B, component_B)

        # Use gene_names from the results if not explicitly provided
        if gene_names is None:
            gene_names = names_A

        # Use auto-detected p_samples unless the user explicitly provided
        # their own (explicit overrides take precedence).
        if p_samples_A is None:
            p_samples_A = p_A
        if p_samples_B is None:
            p_samples_B = p_B

        # gene_mask and gene-specific p are mutually exclusive in
        # compute_clr_differences; when p is gene-specific, silently
        # drop gene_mask so the user doesn't have to know about this
        # constraint.
        if gene_mask is not None and p_samples_A is not None:
            import warnings

            warnings.warn(
                "gene_mask is incompatible with gene-specific p_samples "
                "(hierarchical model). Ignoring gene_mask.",
                UserWarning,
                stacklevel=2,
            )
            gene_mask = None

        # Replace model_A/model_B with raw r_samples for downstream
        model_A = r_A
        model_B = r_B

    # ------------------------------------------------------------------
    # Standard dispatch by method
    # ------------------------------------------------------------------
    if method == "parametric":
        return _compare_parametric(
            model_A,
            model_B,
            gene_names,
            label_A,
            label_B,
            gene_mask=gene_mask,
        )
    elif method == "empirical":
        return _compare_empirical(
            model_A,
            model_B,
            gene_names=gene_names,
            label_A=label_A,
            label_B=label_B,
            component_A=component_A,
            component_B=component_B,
            paired=paired,
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=rng_key,
            batch_size=batch_size,
            gene_mask=gene_mask,
            p_samples_A=p_samples_A,
            p_samples_B=p_samples_B,
        )
    elif method == "shrinkage":
        return _compare_shrinkage(
            model_A,
            model_B,
            gene_names=gene_names,
            label_A=label_A,
            label_B=label_B,
            component_A=component_A,
            component_B=component_B,
            paired=paired,
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=rng_key,
            batch_size=batch_size,
            gene_mask=gene_mask,
            p_samples_A=p_samples_A,
            p_samples_B=p_samples_B,
            sigma_grid=sigma_grid,
            shrinkage_max_iter=shrinkage_max_iter,
            shrinkage_tol=shrinkage_tol,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Use 'parametric', 'empirical', or 'shrinkage'."
        )


# --------------------------------------------------------------------------
# Internal dispatchers
# --------------------------------------------------------------------------


def _compare_parametric(
    model_A,
    model_B,
    gene_names: Optional[List[str]],
    label_A: str,
    label_B: str,
    gene_mask: Optional[jnp.ndarray] = None,
) -> ScribeParametricDEResults:
    """Build a parametric DE comparison from fitted ALR models.

    Parameters
    ----------
    model_A : dict or Distribution
        Fitted logistic-normal model for condition A.
    model_B : dict or Distribution
        Fitted logistic-normal model for condition B.
    gene_names : list of str, optional
        Gene names.
    label_A : str
        Label for condition A.
    label_B : str
        Label for condition B.
    gene_mask : jnp.ndarray, shape ``(D_original,)``, optional
        Boolean gene mask.  For the parametric path the user is expected
        to pass ``gene_mask`` to ``fit_logistic_normal`` so that the
        models are already fitted on the reduced simplex.  When provided
        here, ``gene_names`` are filtered to the kept genes.

    Returns
    -------
    ScribeParametricDEResults
    """
    # Extract consistent (D-1)-dimensional ALR parameters
    mu_A, W_A, d_A = extract_alr_params(model_A)
    mu_B, W_B, d_B = extract_alr_params(model_B)

    # Validate that dimensions match
    if mu_A.shape[-1] != mu_B.shape[-1]:
        raise ValueError(
            f"Model dimensions do not match: "
            f"model_A has D-1={mu_A.shape[-1]}, "
            f"model_B has D-1={mu_B.shape[-1]}.  "
            f"Both models must be fitted on the same gene set."
        )

    # D_full = D_alr + 1 (including "other" when gene_mask was used)
    D_full = mu_A.shape[-1] + 1
    drop_last = gene_mask is not None

    # When gene_mask is provided, the model was fitted on the aggregated
    # simplex (D_kept + 1 genes, last is "other").  The user-visible
    # gene count is D_kept = D_full - 1.
    D_user = D_full - 1 if drop_last else D_full

    # If gene_mask is provided, filter gene_names to kept genes
    if gene_mask is not None and gene_names is not None:
        import numpy as _np

        gene_mask_arr = _np.asarray(gene_mask, dtype=bool)
        gene_names = [n for n, m in zip(gene_names, gene_mask_arr) if m]

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(D_user)]
    elif len(gene_names) != D_user:
        raise ValueError(
            f"gene_names has length {len(gene_names)} but models have "
            f"D={D_user} genes (CLR space, excluding 'other')."
        )

    return ScribeParametricDEResults(
        mu_A=mu_A,
        W_A=W_A,
        d_A=d_A,
        mu_B=mu_B,
        W_B=W_B,
        d_B=d_B,
        gene_names=gene_names,
        label_A=label_A,
        label_B=label_B,
        _drop_last_gene=drop_last,
    )


# --------------------------------------------------------------------------
# Empirical (sample-based) subclass
# --------------------------------------------------------------------------


def _compare_empirical(
    r_samples_A: jnp.ndarray,
    r_samples_B: jnp.ndarray,
    gene_names: Optional[List[str]],
    label_A: str,
    label_B: str,
    component_A: Optional[int],
    component_B: Optional[int],
    paired: bool,
    n_samples_dirichlet: int,
    rng_key,
    batch_size: int,
    gene_mask: Optional[jnp.ndarray] = None,
    p_samples_A: Optional[jnp.ndarray] = None,
    p_samples_B: Optional[jnp.ndarray] = None,
) -> ScribeEmpiricalDEResults:
    """Build an empirical DE comparison from posterior r samples.

    Parameters
    ----------
    r_samples_A : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Dirichlet concentration samples for condition A.
    r_samples_B : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Dirichlet concentration samples for condition B.
    gene_names : list of str, optional
        Gene names.
    label_A : str
        Label for condition A.
    label_B : str
        Label for condition B.
    component_A : int, optional
        Mixture component index for condition A.
    component_B : int, optional
        Mixture component index for condition B.
    paired : bool
        Whether to preserve sample pairing (within-mixture).
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : jax.random.PRNGKey or None
        JAX PRNG key.
    batch_size : int
        Batch size for Dirichlet sampling.
    gene_mask : jnp.ndarray, shape ``(D,)``, optional
        Boolean mask selecting genes to keep.  Passed through to
        ``compute_clr_differences`` for compositional aggregation.
    p_samples_A : jnp.ndarray, optional
        Gene-specific success probability samples for condition A.
        When provided, Gamma-based composition sampling is used.
    p_samples_B : jnp.ndarray, optional
        Gene-specific success probability samples for condition B.

    Returns
    -------
    ScribeEmpiricalDEResults
    """
    from ._empirical import compute_clr_differences

    delta_samples = compute_clr_differences(
        r_samples_A=r_samples_A,
        r_samples_B=r_samples_B,
        component_A=component_A,
        component_B=component_B,
        paired=paired,
        n_samples_dirichlet=n_samples_dirichlet,
        rng_key=rng_key,
        batch_size=batch_size,
        gene_mask=gene_mask,
        p_samples_A=p_samples_A,
        p_samples_B=p_samples_B,
    )

    # Filter gene_names to match kept genes when gene_mask is provided
    if gene_mask is not None and gene_names is not None:
        import numpy as _np

        gene_mask_arr = _np.asarray(gene_mask, dtype=bool)
        gene_names = [n for n, m in zip(gene_names, gene_mask_arr) if m]

    # Number of genes is the last dimension (D_kept when mask is used)
    D = delta_samples.shape[1]
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(D)]
    elif len(gene_names) != D:
        raise ValueError(
            f"gene_names has length {len(gene_names)} but samples have "
            f"D={D} genes."
        )

    return ScribeEmpiricalDEResults(
        delta_samples=delta_samples,
        gene_names=gene_names,
        label_A=label_A,
        label_B=label_B,
    )


# --------------------------------------------------------------------------
# Shrinkage (empirical Bayes) dispatcher
# --------------------------------------------------------------------------


def _compare_shrinkage(
    r_samples_A: jnp.ndarray,
    r_samples_B: jnp.ndarray,
    gene_names: Optional[List[str]],
    label_A: str,
    label_B: str,
    component_A: Optional[int],
    component_B: Optional[int],
    paired: bool,
    n_samples_dirichlet: int,
    rng_key,
    batch_size: int,
    gene_mask: Optional[jnp.ndarray] = None,
    p_samples_A: Optional[jnp.ndarray] = None,
    p_samples_B: Optional[jnp.ndarray] = None,
    sigma_grid: Optional[jnp.ndarray] = None,
    shrinkage_max_iter: int = 200,
    shrinkage_tol: float = 1e-8,
) -> ScribeShrinkageDEResults:
    """Build a shrinkage DE comparison from posterior r samples.

    First runs the empirical CLR difference pipeline, then wraps the
    result in a ``ScribeShrinkageDEResults`` which applies empirical
    Bayes shrinkage when ``gene_level()`` is called.

    Parameters
    ----------
    r_samples_A : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Dirichlet concentration samples for condition A.
    r_samples_B : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Dirichlet concentration samples for condition B.
    gene_names : list of str, optional
        Gene names.
    label_A : str
        Label for condition A.
    label_B : str
        Label for condition B.
    component_A : int, optional
        Mixture component index for condition A.
    component_B : int, optional
        Mixture component index for condition B.
    paired : bool
        Whether to preserve sample pairing.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : jax.random.PRNGKey or None
        JAX PRNG key.
    batch_size : int
        Batch size for Dirichlet sampling.
    gene_mask : jnp.ndarray, optional
        Boolean gene mask.
    p_samples_A : jnp.ndarray, optional
        Gene-specific success probability samples for condition A.
    p_samples_B : jnp.ndarray, optional
        Gene-specific success probability samples for condition B.
    sigma_grid : jnp.ndarray, optional
        Prior scale grid for the shrinkage layer.
    shrinkage_max_iter : int
        Maximum EM iterations.
    shrinkage_tol : float
        EM convergence tolerance.

    Returns
    -------
    ScribeShrinkageDEResults
    """
    from ._empirical import compute_clr_differences

    delta_samples = compute_clr_differences(
        r_samples_A=r_samples_A,
        r_samples_B=r_samples_B,
        component_A=component_A,
        component_B=component_B,
        paired=paired,
        n_samples_dirichlet=n_samples_dirichlet,
        rng_key=rng_key,
        batch_size=batch_size,
        gene_mask=gene_mask,
        p_samples_A=p_samples_A,
        p_samples_B=p_samples_B,
    )

    # Filter gene_names when gene_mask is provided
    if gene_mask is not None and gene_names is not None:
        import numpy as _np

        gene_mask_arr = _np.asarray(gene_mask, dtype=bool)
        gene_names = [n for n, m in zip(gene_names, gene_mask_arr) if m]

    D = delta_samples.shape[1]
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(D)]
    elif len(gene_names) != D:
        raise ValueError(
            f"gene_names has length {len(gene_names)} but samples have "
            f"D={D} genes."
        )

    return ScribeShrinkageDEResults(
        delta_samples=delta_samples,
        gene_names=gene_names,
        label_A=label_A,
        label_B=label_B,
        sigma_grid=sigma_grid,
        shrinkage_max_iter=shrinkage_max_iter,
        shrinkage_tol=shrinkage_tol,
    )
