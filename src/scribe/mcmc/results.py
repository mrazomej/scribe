"""
Results class for SCRIBE MCMC inference.

``ScribeMCMCResults`` is a ``@dataclass`` that wraps a NumPyro MCMC object
and composes analysis functionality from mixins, mirroring the SVI results
architecture.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import jax.numpy as jnp
import pandas as pd

from ..models.config import ModelConfig

# Mixin imports
from ._parameter_extraction import ParameterExtractionMixin
from ._gene_subsetting import GeneSubsettingMixin
from ._component import ComponentMixin
from ._model_helpers import ModelHelpersMixin
from ._sampling import SamplingMixin
from ._likelihood import LikelihoodMixin
from ._normalization import NormalizationMixin
from ._mixture_analysis import MixtureAnalysisMixin


# ==============================================================================
# MCMC Results
# ==============================================================================


@dataclass
class ScribeMCMCResults(
    ParameterExtractionMixin,
    GeneSubsettingMixin,
    ComponentMixin,
    ModelHelpersMixin,
    SamplingMixin,
    LikelihoodMixin,
    NormalizationMixin,
    MixtureAnalysisMixin,
):
    """SCRIBE MCMC results.

    Stores posterior samples and provides analysis methods via mixins.
    The underlying ``numpyro.infer.MCMC`` object is wrapped (composition)
    rather than inherited, so gene/component subsetting always returns
    another ``ScribeMCMCResults`` instance.

    Attributes
    ----------
    samples : Dict
        Raw posterior samples keyed by parameter name.
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes in the dataset.
    model_type : str
        Model identifier (e.g. ``"nbdm"``, ``"zinb_mix"``).
    model_config : ModelConfig
        Configuration used for inference.
    prior_params : Dict[str, Any]
        Prior parameter values used during inference.
    obs : Optional[pd.DataFrame]
        Cell-level metadata from ``adata.obs``.
    var : Optional[pd.DataFrame]
        Gene-level metadata from ``adata.var``.
    uns : Optional[Dict]
        Unstructured metadata from ``adata.uns``.
    n_obs : Optional[int]
        Number of observations (cells).
    n_vars : Optional[int]
        Number of variables (genes).
    predictive_samples : Optional[jnp.ndarray]
        Predictive samples from :meth:`get_ppc_samples`.
    n_components : Optional[int]
        Number of mixture components (``None`` for non-mixture models).
    denoised_counts : Optional[jnp.ndarray]
        Denoised counts from :meth:`denoise_counts`.
    _mcmc : Optional[Any]
        Wrapped ``numpyro.infer.MCMC`` object for diagnostics.
        ``None`` on subsets produced by gene/component indexing.
    """

    # -- core fields ---------------------------------------------------------
    samples: Dict
    n_cells: int
    n_genes: int
    model_type: str
    model_config: ModelConfig
    prior_params: Dict[str, Any]

    # -- AnnData metadata ----------------------------------------------------
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

    # -- optional computed results -------------------------------------------
    predictive_samples: Optional[jnp.ndarray] = None
    n_components: Optional[int] = None
    denoised_counts: Optional[jnp.ndarray] = None

    # -- wrapped MCMC object (None on subsets) -------------------------------
    _mcmc: Optional[Any] = field(default=None, repr=False)

    # -------------------------------------------------------------------------
    # Post-init validation
    # -------------------------------------------------------------------------

    def __post_init__(self):
        """Validate model configuration and set derived attributes."""
        if (
            self.n_components is None
            and self.model_config.n_components is not None
        ):
            self.n_components = self.model_config.n_components

        self._validate_model_config()

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------

    @classmethod
    def from_mcmc(
        cls,
        mcmc,
        n_cells: int,
        n_genes: int,
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        **kwargs,
    ) -> "ScribeMCMCResults":
        """Create results from an existing ``numpyro.infer.MCMC`` instance.

        Extracts samples once and stores the MCMC object for diagnostics.

        Parameters
        ----------
        mcmc : numpyro.infer.MCMC
            Completed MCMC run.
        n_cells : int
            Number of cells.
        n_genes : int
            Number of genes.
        model_type : str
            Model identifier.
        model_config : ModelConfig
            Model configuration.
        prior_params : Dict[str, Any]
            Prior parameter values.
        **kwargs
            Forwarded to the dataclass constructor (e.g. ``obs``, ``var``).

        Returns
        -------
        ScribeMCMCResults
        """
        return cls(
            samples=mcmc.get_samples(group_by_chain=False),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params,
            _mcmc=mcmc,
            **kwargs,
        )

    @classmethod
    def from_anndata(
        cls,
        mcmc,
        adata,
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        **kwargs,
    ) -> "ScribeMCMCResults":
        """Create results from an MCMC instance and AnnData object.

        Parameters
        ----------
        mcmc : numpyro.infer.MCMC
            Completed MCMC run.
        adata : AnnData
            AnnData object with cell/gene metadata.
        model_type : str
            Model identifier.
        model_config : ModelConfig
            Model configuration.
        prior_params : Dict[str, Any]
            Prior parameter values.
        **kwargs
            Forwarded to the dataclass constructor.

        Returns
        -------
        ScribeMCMCResults
        """
        return cls.from_mcmc(
            mcmc=mcmc,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Posterior sample access
    # -------------------------------------------------------------------------

    def get_posterior_samples(self) -> Dict:
        """Return posterior samples.

        MCMC samples already contain canonical parameters (``p``, ``r``,
        ``mixing_weights``, etc.) because derived parameters are
        registered as ``numpyro.deterministic`` sites and unconstrained
        specs sample via ``TransformedDistribution`` in constrained
        space.

        Returns
        -------
        Dict
            Parameter name -> sample array.
        """
        return self.samples

    def get_samples(self, group_by_chain: bool = False) -> Dict:
        """Return samples with optional chain grouping.

        Parameters
        ----------
        group_by_chain : bool, default=False
            Preserve the chain dimension (requires the original MCMC
            object).

        Returns
        -------
        Dict
            Parameter samples.
        """
        if group_by_chain:
            if self._mcmc is None:
                raise RuntimeError(
                    "group_by_chain requires the original MCMC object "
                    "(not available on subsets)."
                )
            return self._mcmc.get_samples(group_by_chain=True)
        return self.samples

    # -------------------------------------------------------------------------
    # MCMC diagnostic delegation
    # -------------------------------------------------------------------------

    def print_summary(self, **kwargs):
        """Print MCMC summary statistics (delegates to the wrapped MCMC).

        Raises
        ------
        RuntimeError
            If the MCMC object is not available (e.g. on subsets).
        """
        if self._mcmc is None:
            raise RuntimeError(
                "print_summary requires the original MCMC object "
                "(not available on subsets)."
            )
        self._mcmc.print_summary(**kwargs)

    def get_extra_fields(self, **kwargs) -> Dict:
        """Return MCMC extra fields (e.g. potential_energy, diverging).

        Returns an empty dict when the MCMC object is not available
        (subsets).
        """
        if self._mcmc is None:
            return {}
        return self._mcmc.get_extra_fields(**kwargs)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_model_config(self):
        """Validate that model_config is consistent with model_type."""
        expected_base = (
            self.model_type[:-4]
            if self.model_type.endswith("_mix")
            else self.model_type
        )
        if self.model_config.base_model != expected_base:
            raise ValueError(
                f"Model type '{self.model_type}' does not match config "
                f"base model '{self.model_config.base_model}'"
            )

        if self.n_components is not None:
            if not self.model_type.endswith("_mix"):
                raise ValueError(
                    f"Model type '{self.model_type}' is not a mixture model "
                    f"but n_components={self.n_components} was specified"
                )
            if self.model_config.n_components != self.n_components:
                raise ValueError(
                    f"n_components mismatch: {self.n_components} vs "
                    f"{self.model_config.n_components} in model_config"
                )

    # -------------------------------------------------------------------------
    # Convenience property
    # -------------------------------------------------------------------------

    @property
    def posterior_samples(self) -> Dict:
        """Posterior samples (read-only property)."""
        return self.get_posterior_samples()
