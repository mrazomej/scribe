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
            for key in ("delta_mean", "delta_sd", "prob_positive",
                        "prob_effect", "lfsr", "lfsr_tau"):
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
) -> ScribeDEResults:
    """Create a DE results object from two fitted models or posterior samples.

    This factory function dispatches to either the parametric (Gaussian)
    or empirical (sample-based) DE path depending on ``method``.

    Parameters
    ----------
    model_A : dict, Distribution, or ndarray
        For ``method="parametric"``: fitted logistic-normal model for
        condition A (dict with ``loc``/``cov_factor``/``cov_diag``, or
        a ``LowRankLogisticNormal`` / ``SoftmaxNormal`` distribution).
        For ``method="empirical"``: posterior samples of Dirichlet
        concentration parameters, shape ``(N, D)`` or ``(N, K, D)``
        for mixture models.
    model_B : dict, Distribution, or ndarray
        Same as ``model_A`` for condition B.
    gene_names : list of str, optional
        Gene names.  If ``None``, generic names are generated.
    label_A : str, default='A'
        Human-readable label for condition A.
    label_B : str, default='B'
        Human-readable label for condition B.
    method : str, default='parametric'
        DE method.  ``"parametric"`` uses analytic Gaussian posteriors;
        ``"empirical"`` uses Monte Carlo counting on posterior samples.
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

    Returns
    -------
    ScribeDEResults
        Either ``ScribeParametricDEResults`` or
        ``ScribeEmpiricalDEResults`` depending on ``method``.

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
    """
    if method == "parametric":
        return _compare_parametric(
            model_A, model_B, gene_names, label_A, label_B,
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
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'parametric' or 'empirical'."
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
        gene_names = [
            n for n, m in zip(gene_names, gene_mask_arr) if m
        ]

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

    Returns
    -------
    ScribeEmpiricalDEResults
    """
    from ._empirical import compute_clr_differences

    # Compute CLR differences from posterior r samples
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
