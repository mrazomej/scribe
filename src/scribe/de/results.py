"""ScribeDEResults: structured results for differential expression analysis.

This module defines ``ScribeDEResults``, a dataclass that encapsulates a
pairwise comparison of two fitted logistic-normal models and provides
methods for gene-level analysis, gene-set testing, Bayesian error control,
and result formatting.

The companion factory function ``compare()`` constructs a
``ScribeDEResults`` from two models, handling parameter extraction and
dimensional normalization internally via ``extract_alr_params()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import jax.numpy as jnp

from ._extract import extract_alr_params
from ._gene_level import differential_expression, call_de_genes
from ._set_level import test_contrast, test_gene_set, build_balance_contrast
from ._error_control import (
    compute_lfdr,
    compute_pefp,
    find_lfsr_threshold,
    format_de_table,
)


# --------------------------------------------------------------------------
# ScribeDEResults dataclass
# --------------------------------------------------------------------------


@dataclass
class ScribeDEResults:
    """Structured results for Bayesian differential expression analysis.

    This class encapsulates a pairwise comparison of two fitted
    logistic-normal models (conditions A and B).  It stores the internal
    (D-1)-dimensional ALR parameters and lazily computes gene-level results
    when first requested.

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
        Gene names for output.  If ``None``, generic names are generated.
    label_A : str
        Human-readable label for condition A.
    label_B : str
        Human-readable label for condition B.

    Attributes
    ----------
    D_alr : int
        Dimensionality in ALR space (D-1).
    D : int
        Dimensionality in CLR space (number of genes).

    Examples
    --------
    >>> from scribe.de import compare
    >>> de = compare(model_A, model_B, gene_names=gene_names)
    >>> results = de.gene_level(tau=jnp.log(1.1))
    >>> de.summary()
    """

    # --- Internal ALR parameters (D-1 dimensional) ---
    mu_A: jnp.ndarray
    W_A: jnp.ndarray
    d_A: jnp.ndarray
    mu_B: jnp.ndarray
    W_B: jnp.ndarray
    d_B: jnp.ndarray

    # --- Metadata ---
    gene_names: Optional[List[str]] = None
    label_A: str = "A"
    label_B: str = "B"

    # --- Cached gene-level results (computed lazily) ---
    _gene_results: Optional[dict] = field(
        default=None, repr=False, init=False
    )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def D_alr(self) -> int:
        """Dimensionality in ALR space (D-1)."""
        return self.mu_A.shape[-1]

    @property
    def D(self) -> int:
        """Dimensionality in CLR space (number of genes)."""
        return self.D_alr + 1

    # ------------------------------------------------------------------
    # Core analysis methods
    # ------------------------------------------------------------------

    def gene_level(
        self,
        tau: float = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level differential expression.

        Wraps ``differential_expression()`` using the stored ALR parameters.
        Results are cached so that repeated calls with the same arguments
        are free.

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

        # Compute and cache
        self._gene_results = differential_expression(
            model_A_dict,
            model_B_dict,
            tau=tau,
            coordinate=coordinate,
            gene_names=self.gene_names,
        )
        return self._gene_results

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
            Practical significance threshold (used if gene-level results
            have not yet been computed).
        lfsr_threshold : float, default=0.05
            Maximum acceptable local false sign rate.
        prob_effect_threshold : float, default=0.95
            Minimum posterior probability of practical effect.

        Returns
        -------
        ndarray of bool, shape ``(D,)``
            Boolean mask of DE genes.
        """
        # Ensure gene-level results are computed
        if self._gene_results is None:
            self.gene_level(tau=tau)

        return call_de_genes(
            self._gene_results,
            lfsr_threshold=lfsr_threshold,
            prob_effect_threshold=prob_effect_threshold,
        )

    # ------------------------------------------------------------------
    # Set-level analysis
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
        return test_gene_set(
            model_A_dict, model_B_dict, gene_set_indices, tau
        )

    def test_contrast(
        self,
        contrast: jnp.ndarray,
        tau: float = 0.0,
    ) -> dict:
        """Test a linear contrast ``c^T Delta``.

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
    # Error control
    # ------------------------------------------------------------------

    def compute_pefp(self, threshold: float = 0.05) -> float:
        """Compute posterior expected false discovery proportion.

        Parameters
        ----------
        threshold : float, default=0.05
            lfsr threshold for calling genes DE.

        Returns
        -------
        float
            Expected false discovery proportion.
        """
        if self._gene_results is None:
            self.gene_level()
        return compute_pefp(self._gene_results["lfsr"], threshold=threshold)

    def find_threshold(self, target_pefp: float = 0.05) -> float:
        """Find lfsr threshold controlling PEFP at target level.

        Parameters
        ----------
        target_pefp : float, default=0.05
            Target PEFP level.

        Returns
        -------
        float
            lfsr threshold.
        """
        if self._gene_results is None:
            self.gene_level()
        return find_lfsr_threshold(
            self._gene_results["lfsr"], target_pefp=target_pefp
        )

    # ------------------------------------------------------------------
    # Formatting
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
        if self._gene_results is None:
            self.gene_level(tau=tau)
        return format_de_table(
            self._gene_results, sort_by=sort_by, top_n=top_n
        )

    def __repr__(self) -> str:
        """Concise representation of the DE comparison."""
        return (
            f"ScribeDEResults("
            f"D={self.D}, "
            f"rank_A={self.W_A.shape[-1]}, "
            f"rank_B={self.W_B.shape[-1]}, "
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
) -> ScribeDEResults:
    """Create a ``ScribeDEResults`` from two fitted models.

    This factory function handles parameter extraction from various model
    representations (dicts, ``LowRankLogisticNormal``, ``SoftmaxNormal``),
    resolves the D vs (D-1) dimensional mismatch, and returns a structured
    results object ready for analysis.

    Parameters
    ----------
    model_A : dict or LowRankLogisticNormal or SoftmaxNormal
        Fitted logistic-normal model for condition A.
    model_B : dict or LowRankLogisticNormal or SoftmaxNormal
        Fitted logistic-normal model for condition B.
    gene_names : list of str, optional
        Gene names.  If ``None``, generic names are generated
        (``gene_0, gene_1, ...``).
    label_A : str, default='A'
        Human-readable label for condition A.
    label_B : str, default='B'
        Human-readable label for condition B.

    Returns
    -------
    ScribeDEResults
        Structured DE results ready for analysis.

    Examples
    --------
    >>> from scribe.de import compare
    >>> de = compare(results_A, results_B,
    ...              gene_names=adata.var_names.tolist(),
    ...              label_A="Treatment", label_B="Control")
    >>> gene_results = de.gene_level(tau=jnp.log(1.1))
    >>> is_de = de.call_genes(lfsr_threshold=0.05)
    >>> print(de.summary())
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

    # Generate gene names if not provided (D = D_alr + 1 for CLR space)
    if gene_names is None:
        D = mu_A.shape[-1] + 1
        gene_names = [f"gene_{i}" for i in range(D)]

    return ScribeDEResults(
        mu_A=mu_A,
        W_A=W_A,
        d_A=d_A,
        mu_B=mu_B,
        W_B=W_B,
        d_B=d_B,
        gene_names=gene_names,
        label_A=label_A,
        label_B=label_B,
    )
