"""Structured DE results classes and factory entrypoints.

This module now keeps class declarations thin and composes behavior from
specialized mixins. The public API is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import jax.numpy as jnp

from ._results_base_mixin import BaseResultsMixin
from ._results_empirical_mixin import EmpiricalResultsMixin
from ._results_factory import compare, compare_datasets
from ._results_parametric_mixin import ParametricResultsMixin
from ._results_shrinkage_mixin import ShrinkageResultsMixin


@dataclass
class ScribeDEResults(BaseResultsMixin):
    """Abstract base class for Bayesian differential-expression results.

    Parameters
    ----------
    gene_names : list of str, optional
        Gene names. If omitted, subclasses generate default names.
    label_A : str, default='A'
        Human-readable condition label for group A.
    label_B : str, default='B'
        Human-readable condition label for group B.
    method : str, default='parametric'
        Method identifier updated by concrete subclasses.
    """

    # Core metadata shared across all DE result variants.
    gene_names: Optional[List[str]] = None
    label_A: str = "A"
    label_B: str = "B"
    method: str = "parametric"

    # Caches keyed by tau to avoid stale re-use of gene-level statistics.
    _gene_results: Optional[dict] = field(default=None, repr=False, init=False)
    _cached_tau: Optional[float] = field(default=None, repr=False, init=False)

    @property
    def D(self) -> int:
        """Number of genes (CLR dimensionality)."""
        raise NotImplementedError("Subclasses must implement D.")

    @property
    def D_alr(self) -> int:
        """Dimensionality in ALR coordinates."""
        return self.D - 1

    def gene_level(
        self,
        tau: float = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level DE statistics in the requested coordinate system.

        Parameters
        ----------
        tau : float, default=0.0
            Practical-significance threshold.
        coordinate : str, default='clr'
            Coordinate system for effect summaries.

        Returns
        -------
        dict
            Gene-level differential-expression summary.
        """
        raise NotImplementedError("Subclasses must implement gene_level().")


@dataclass(repr=False)
class ScribeParametricDEResults(ParametricResultsMixin, ScribeDEResults):
    """Parametric DE results using analytic Gaussian posteriors.

    Parameters
    ----------
    mu_A, W_A, d_A : jnp.ndarray
        ALR-space low-rank Gaussian parameters for condition A.
    mu_B, W_B, d_B : jnp.ndarray
        ALR-space low-rank Gaussian parameters for condition B.
    """

    mu_A: jnp.ndarray = field(default=None)
    W_A: jnp.ndarray = field(default=None)
    d_A: jnp.ndarray = field(default=None)
    mu_B: jnp.ndarray = field(default=None)
    W_B: jnp.ndarray = field(default=None)
    d_B: jnp.ndarray = field(default=None)

    # True when masked genes were aggregated into an "other" column.
    _drop_last_gene: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Set method identifier for downstream reporting."""
        self.method = "parametric"


@dataclass(repr=False)
class ScribeEmpiricalDEResults(EmpiricalResultsMixin, ScribeDEResults):
    """Empirical DE results from posterior sample differences.

    Parameters
    ----------
    delta_samples : jnp.ndarray
        Posterior CLR differences with shape ``(N, D)``.
    r_samples_A, r_samples_B, p_samples_A, p_samples_B : jnp.ndarray, optional
        Raw NB posterior samples for biological-level metrics.
    mu_samples_A, mu_samples_B, phi_samples_A, phi_samples_B : jnp.ndarray,
    optional
        Native parameterization samples used for numerically stable
        biological-level computations.
    simplex_A, simplex_B : jnp.ndarray, optional
        Full simplex samples retained for mask recomputation.
    mu_map_A, mu_map_B : jnp.ndarray, optional
        MAP mean-expression vectors used to construct expression masks.
    """

    delta_samples: jnp.ndarray = field(default=None, repr=False)

    r_samples_A: Optional[jnp.ndarray] = field(default=None, repr=False)
    r_samples_B: Optional[jnp.ndarray] = field(default=None, repr=False)
    p_samples_A: Optional[jnp.ndarray] = field(default=None, repr=False)
    p_samples_B: Optional[jnp.ndarray] = field(default=None, repr=False)

    mu_samples_A: Optional[jnp.ndarray] = field(default=None, repr=False)
    mu_samples_B: Optional[jnp.ndarray] = field(default=None, repr=False)
    phi_samples_A: Optional[jnp.ndarray] = field(default=None, repr=False)
    phi_samples_B: Optional[jnp.ndarray] = field(default=None, repr=False)

    simplex_A: Optional[jnp.ndarray] = field(default=None, repr=False)
    simplex_B: Optional[jnp.ndarray] = field(default=None, repr=False)

    mu_map_A: Optional[jnp.ndarray] = field(default=None, repr=False)
    mu_map_B: Optional[jnp.ndarray] = field(default=None, repr=False)

    # Post-component-sliced AxisLayout for p and phi (used by the
    # biological_level() call to avoid ndim heuristics).  None when
    # layouts were not available at construction time.
    p_post_layout: Optional[object] = field(default=None, repr=False)
    phi_post_layout: Optional[object] = field(default=None, repr=False)

    # Internal mask bookkeeping for interactive mask updates.
    _gene_mask: Optional[jnp.ndarray] = field(
        default=None, repr=False, init=False
    )
    _all_gene_names: Optional[List[str]] = field(
        default=None, repr=False, init=False
    )

    # Informational sample count exposed on repr.
    n_samples: int = field(default=0, repr=True)

    def __post_init__(self):
        """Initialize method id, sample count, and biological-level caches."""
        self.method = "empirical"
        if self.delta_samples is not None:
            self.n_samples = self.delta_samples.shape[0]
        self._biological_results = None
        self._cached_bio_taus = None


@dataclass(repr=False)
class ScribeShrinkageDEResults(ShrinkageResultsMixin, ScribeEmpiricalDEResults):
    """Empirical-Bayes shrinkage DE results layered on empirical samples.

    Parameters
    ----------
    empirical : ScribeEmpiricalDEResults, optional
        Source empirical object used as a shortcut initializer.
    sigma_grid : jnp.ndarray, optional
        Scale grid for the scale-mixture prior.
    shrinkage_max_iter : int, default=200
        Maximum EM iterations.
    shrinkage_tol : float, default=1e-8
        EM convergence tolerance.
    """

    empirical: Optional["ScribeEmpiricalDEResults"] = field(
        default=None, repr=False
    )

    sigma_grid: Optional[jnp.ndarray] = field(default=None, repr=False)
    shrinkage_max_iter: int = field(default=200, repr=False)
    shrinkage_tol: float = field(default=1e-8, repr=False)

    null_proportion: Optional[float] = field(
        default=None, repr=True, init=False
    )
    prior_weights: Optional[jnp.ndarray] = field(
        default=None, repr=False, init=False
    )

    def __post_init__(self):
        """Populate from empirical source when provided, then initialize."""
        emp = self.empirical
        if emp is not None:
            self.delta_samples = emp.delta_samples
            self.gene_names = emp.gene_names
            self.label_A = emp.label_A
            self.label_B = emp.label_B
            self.r_samples_A = emp.r_samples_A
            self.r_samples_B = emp.r_samples_B
            self.p_samples_A = emp.p_samples_A
            self.p_samples_B = emp.p_samples_B
            self.mu_samples_A = emp.mu_samples_A
            self.mu_samples_B = emp.mu_samples_B
            self.phi_samples_A = emp.phi_samples_A
            self.phi_samples_B = emp.phi_samples_B
            self.simplex_A = emp.simplex_A
            self.simplex_B = emp.simplex_B
            self.mu_map_A = emp.mu_map_A
            self.mu_map_B = emp.mu_map_B
            self.empirical = None

        super().__post_init__()
        self.method = "shrinkage"

        # Restore mask bookkeeping after parent initializer resets internals.
        if emp is not None:
            self._gene_mask = emp._gene_mask
            self._all_gene_names = emp._all_gene_names
