"""Parametric DE methods for Gaussian results objects."""

from __future__ import annotations

import jax.numpy as jnp

from ._gene_level import differential_expression
from ._set_level import test_contrast, test_gene_set


class ParametricResultsMixin:
    """Analytic Gaussian DE operations for parametric results classes."""

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

    def gene_level(
        self,
        tau: float = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level DE via analytic Gaussian posteriors.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold (log-scale).
        coordinate : str, default='clr'
            Coordinate system. Only ``'clr'`` is supported.

        Returns
        -------
        dict
            Gene-level summary dictionary from
            :func:`~scribe.de._gene_level.differential_expression`.
        """
        # Build compact model dictionaries expected by set/gene-level helpers.
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

        self._cached_tau = tau
        results = differential_expression(
            model_A_dict,
            model_B_dict,
            tau=tau,
            coordinate=coordinate,
            gene_names=None,
        )

        # Drop the synthetic "other" column when masked fitting was used.
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
            if "gene_names" in results:
                results["gene_names"] = results["gene_names"][:-1]

        # Keep outward-facing gene names controlled by the results object.
        results["gene_names"] = list(self.gene_names)
        self._gene_results = results
        return self._gene_results

    def test_gene_set(
        self,
        gene_set_indices: jnp.ndarray,
        tau: float = 0.0,
    ) -> dict:
        """Test enrichment of a gene set using compositional balances.

        Parameters
        ----------
        gene_set_indices : jnp.ndarray
            Gene indices in CLR space.
        tau : float, default=0.0
            Practical significance threshold.

        Returns
        -------
        dict
            Posterior inference for pathway balance shift.
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

    def test_contrast(
        self,
        contrast: jnp.ndarray,
        tau: float = 0.0,
    ) -> dict:
        """Test a linear contrast ``c^T Delta`` under Gaussian posteriors.

        Parameters
        ----------
        contrast : jnp.ndarray
            Contrast vector in CLR space with shape ``(D,)``.
        tau : float, default=0.0
            Practical significance threshold.

        Returns
        -------
        dict
            Posterior statistics for the projected effect.
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

    def __repr__(self) -> str:
        """Return a concise representation of the comparison."""
        return (
            f"ScribeParametricDEResults("
            f"D={self.D}, "
            f"rank_A={self.W_A.shape[-1]}, "
            f"rank_B={self.W_B.shape[-1]}, "
            f"labels='{self.label_A}' vs '{self.label_B}')"
        )
