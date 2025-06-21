"""
Distribution builder for SVI inference.

This module creates the appropriate distributions for SVI models and guides
based on the parameterization and prior specifications.
"""

from typing import Dict, Optional, Any, Type
import numpyro.distributions as dist
from ..core.distribution_builder import DistributionBuilder


class SVIDistributionBuilder:
    """Builds distributions for SVI inference."""

    @staticmethod
    def build_distributions(
        model_type: str,
        parameterization: str,
        priors: Dict[str, Any],
        r_distribution: Optional[Type[dist.Distribution]] = None,
        mu_distribution: Optional[Type[dist.Distribution]] = None,
    ) -> Dict[str, Any]:
        """
        Build all distributions needed for SVI inference.

        Parameters
        ----------
        model_type : str
            Type of model (e.g., "nbdm", "zinb_mix", etc.)
        parameterization : str
            Parameterization type ("standard", "linked", "odds_ratio", "unconstrained")
        priors : Dict[str, Any]
            Dictionary of prior parameters
        r_distribution : Optional[Type[dist.Distribution]]
            NumPyro distribution class for r parameter (e.g., dist.Gamma, dist.LogNormal)
        mu_distribution : Optional[Type[dist.Distribution]]
            NumPyro distribution class for mu parameter (e.g., dist.LogNormal, dist.Gamma)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all model and guide distributions
        """
        # Delegate to unified distribution builder
        return DistributionBuilder.build_distributions(
            model_type=model_type,
            parameterization=parameterization,
            inference_method="svi",
            priors=priors,
            r_distribution=r_distribution,
            mu_distribution=mu_distribution,
        )
