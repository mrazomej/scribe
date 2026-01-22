"""
Parameter collection and mapping utilities for SCRIBE.

This module provides utilities for collecting optional parameters and mapping
them to the correct ModelConfig attribute names based on parameterization type
and constraint settings.
"""

from typing import Dict, Any, Optional, List, Union

# ==============================================================================
# Parameter Collector Class
# ==============================================================================


class ParameterCollector:
    """
    Utility class for collecting and mapping optional parameters.

    This class provides static methods to collect non-None parameters and map
    them to the appropriate ModelConfig attribute names based on the model
    parameterization and constraint settings.

    Examples
    --------
    >>> # Collect only non-None parameters
    >>> params = ParameterCollector.collect_non_none(
    ...     r_prior=(1.0, 1.0),
    ...     p_prior=None,
    ...     gate_prior=(2.0, 0.5)
    ... )
    >>> print(params)
    {'r_prior': (1.0, 1.0), 'gate_prior': (2.0, 0.5)}

    >>> # Collect and map prior parameters for standard parameterization
    >>> prior_config = ParameterCollector.collect_and_map_priors(
    ...     unconstrained=False,
    ...     parameterization="standard",
    ...     r_prior=(1.0, 1.0),
    ...     p_prior=(2.0, 0.5)
    ... )
    >>> print(prior_config)
    {'r_param_prior': (1.0, 1.0), 'p_param_prior': (2.0, 0.5)}

    >>> # Collect VAE parameters
    >>> vae_config = ParameterCollector.collect_vae_params(
    ...     vae_latent_dim=5,
    ...     vae_hidden_dims=[256, 128],
    ...     vae_activation="gelu"
    ... )
    >>> print(vae_config)
    {
        'vae_latent_dim': 5,
        'vae_hidden_dims': [256, 128],
        'vae_activation': 'gelu'
    }
    """

    # --------------------------------------------------------------------------
    # Collect Non-None Parameters
    # --------------------------------------------------------------------------

    @staticmethod
    def collect_non_none(**kwargs) -> Dict[str, Any]:
        """
        Return only non-None keyword arguments.

        This is a simple utility to filter out None values from a dictionary
        of keyword arguments, useful for collecting only the parameters that
        were explicitly provided by the user.

        Parameters
        ----------
        **kwargs
            Keyword arguments to filter

        Returns
        -------
        Dict[str, Any]
            Dictionary containing only non-None values

        Examples
        --------
        >>> params = ParameterCollector.collect_non_none(
        ...     a=1, b=None, c="hello", d=None
        ... )
        >>> print(params)
        {'a': 1, 'c': 'hello'}
        """
        return {k: v for k, v in kwargs.items() if v is not None}

    # --------------------------------------------------------------------------
    # Collect and Map Prior Parameters
    # --------------------------------------------------------------------------

    @staticmethod
    def collect_and_map_priors(
        unconstrained: bool,
        parameterization: str,
        r_prior: Optional[tuple] = None,
        p_prior: Optional[tuple] = None,
        gate_prior: Optional[tuple] = None,
        p_capture_prior: Optional[tuple] = None,
        mixing_prior: Optional[Any] = None,
        mu_prior: Optional[tuple] = None,
        phi_prior: Optional[tuple] = None,
        phi_capture_prior: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Collect and map prior parameters to ModelConfig attribute names.

        This method collects all non-None prior parameters and maps them to
        the correct ModelConfig attribute names based on whether the model
        uses unconstrained parameterization and the specific parameterization
        type (standard, linked, odds_ratio).

        Parameters
        ----------
        unconstrained : bool
            Whether the model uses unconstrained parameterization
        parameterization : str
            Model parameterization type ("standard", "linked", "odds_ratio")
        r_prior : Optional[tuple], default=None
            Prior parameters for dispersion parameter (r)
        p_prior : Optional[tuple], default=None
            Prior parameters for success probability (p)
        gate_prior : Optional[tuple], default=None
            Prior parameters for zero-inflation gate
        p_capture_prior : Optional[tuple], default=None
            Prior parameters for variable capture probability
        mixing_prior : Optional[Any], default=None
            Prior parameters for mixture components
        mu_prior : Optional[tuple], default=None
            Prior parameters for mean parameter (used in linked/odds_ratio)
        phi_prior : Optional[tuple], default=None
            Prior parameters for odds ratio parameter (used in odds_ratio)
        phi_capture_prior : Optional[tuple], default=None
            Prior parameters for variable capture odds ratio

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping ModelConfig attribute names to prior values

        Examples
        --------
        >>> # Standard parameterization (constrained)
        >>> config = ParameterCollector.collect_and_map_priors(
        ...     unconstrained=False,
        ...     parameterization="standard",
        ...     r_prior=(1.0, 1.0),
        ...     p_prior=(2.0, 0.5)
        ... )
        >>> print(config)
        {'r_param_prior': (1.0, 1.0), 'p_param_prior': (2.0, 0.5)}

        >>> # Unconstrained standard parameterization
        >>> config = ParameterCollector.collect_and_map_priors(
        ...     unconstrained=True,
        ...     parameterization="standard",
        ...     r_prior=(0.0, 1.0),
        ...     p_prior=(0.0, 1.0)
        ... )
        >>> print(config)
        {
            'r_unconstrained_prior': (0.0, 1.0),
            'p_unconstrained_prior': (0.0, 1.0)
        }

        >>> # Linked parameterization with mu parameter
        >>> config = ParameterCollector.collect_and_map_priors(
        ...     unconstrained=False,
        ...     parameterization="linked",
        ...     p_prior=(1.0, 1.0),
        ...     mu_prior=(0.0, 1.0)
        ... )
        >>> print(config)
        {'p_param_prior': (1.0, 1.0), 'mu_param_prior': (0.0, 1.0)}
        """
        # Step 1: Collect non-None priors
        user_priors = ParameterCollector.collect_non_none(
            r_prior=r_prior,
            p_prior=p_prior,
            gate_prior=gate_prior,
            p_capture_prior=p_capture_prior,
            mixing_prior=mixing_prior,
            mu_prior=mu_prior,
            phi_prior=phi_prior,
            phi_capture_prior=phi_capture_prior,
        )

        # Step 2: Map to internal names based on unconstrained/parameterization
        if unconstrained:
            # For unconstrained parameterization, use unconstrained prior names
            mapped_priors = {
                "p_unconstrained_prior": user_priors.get("p_prior"),
                "r_unconstrained_prior": user_priors.get("r_prior"),
                "gate_unconstrained_prior": user_priors.get("gate_prior"),
                "p_capture_unconstrained_prior": user_priors.get(
                    "p_capture_prior"
                ),
                "mixing_logits_unconstrained_prior": user_priors.get(
                    "mixing_prior"
                ),
            }

            # Add parameterization-specific unconstrained priors
            if parameterization in ("linked", "mean_prob"):
                mapped_priors.update(
                    {
                        "mu_unconstrained_prior": user_priors.get("mu_prior"),
                    }
                )
            elif parameterization in ("odds_ratio", "mean_odds"):
                mapped_priors.update(
                    {
                        "phi_unconstrained_prior": user_priors.get("phi_prior"),
                        "mu_unconstrained_prior": user_priors.get("mu_prior"),
                        "phi_capture_unconstrained_prior": user_priors.get(
                            "phi_capture_prior"
                        ),
                    }
                )
        else:
            # For constrained parameterization, use standard prior names
            mapped_priors = {
                "p_param_prior": user_priors.get("p_prior"),
                "r_param_prior": user_priors.get("r_prior"),
                "mu_param_prior": user_priors.get("mu_prior"),
                "phi_param_prior": user_priors.get("phi_prior"),
                "gate_param_prior": user_priors.get("gate_prior"),
                "p_capture_param_prior": user_priors.get("p_capture_prior"),
                "phi_capture_param_prior": user_priors.get("phi_capture_prior"),
                "mixing_param_prior": user_priors.get("mixing_prior"),
            }

        # Step 3: Return only non-None mapped values
        return {k: v for k, v in mapped_priors.items() if v is not None}

    # --------------------------------------------------------------------------
    # Collect VAE Parameters
    # --------------------------------------------------------------------------

    @staticmethod
    def collect_vae_params(
        vae_latent_dim: int = 3,
        vae_hidden_dims: Optional[List[int]] = None,
        vae_activation: Optional[str] = None,
        vae_input_transformation: Optional[str] = None,
        vae_vcp_hidden_dims: Optional[List[int]] = None,
        vae_vcp_activation: Optional[str] = None,
        vae_prior_type: str = "standard",
        vae_prior_num_layers: Optional[int] = None,
        vae_prior_hidden_dims: Optional[List[int]] = None,
        vae_prior_activation: Optional[str] = None,
        vae_prior_mask_type: str = "alternating",
        vae_standardize: bool = False,
    ) -> Dict[str, Any]:
        """
        Collect VAE-specific parameters for ModelConfig.

        This method collects all VAE-related parameters and returns them as
        a dictionary ready to be merged into ModelConfig kwargs. Only non-None
        values are included in the returned dictionary.

        Parameters
        ----------
        vae_latent_dim : int, default=3
            Dimension of the VAE latent space
        vae_hidden_dims : Optional[List[int]], default=None
            List of hidden layer dimensions for the VAE encoder/decoder
        vae_activation : Optional[str], default=None
            Activation function name for VAE layers
        vae_input_transformation : Optional[str], default=None
            Input transformation for VAE
        vae_vcp_hidden_dims : Optional[List[int]], default=None
            Hidden layer dimensions for VCP encoder (variable capture models)
        vae_vcp_activation : Optional[str], default=None
            Activation function for VCP encoder
        vae_prior_type : str, default="standard"
            Type of VAE prior ("standard" or "decoupled")
        vae_prior_num_layers : Optional[int], default=None
            Number of coupling layers for decoupled prior
        vae_prior_hidden_dims : Optional[List[int]], default=None
            Hidden layer dimensions for decoupled prior coupling layers
        vae_prior_activation : Optional[str], default=None
            Activation function for decoupled prior coupling layers
        vae_prior_mask_type : str, default="alternating"
            Mask type for decoupled prior coupling layers
        vae_standardize : bool, default=False
            Whether to standardize input data for VAE models

        Returns
        -------
        Dict[str, Any]
            Dictionary of VAE parameters ready for ModelConfig

        Examples
        --------
        >>> # Basic VAE configuration
        >>> vae_config = ParameterCollector.collect_vae_params(
        ...     vae_latent_dim=5,
        ...     vae_hidden_dims=[256, 128],
        ...     vae_activation="gelu"
        ... )
        >>> print(vae_config)
        {'vae_latent_dim': 5, 'vae_hidden_dims': [256, 128], 'vae_activation': 'gelu'}

        >>> # VAE with decoupled prior
        >>> vae_config = ParameterCollector.collect_vae_params(
        ...     vae_latent_dim=3,
        ...     vae_prior_type="decoupled",
        ...     vae_prior_num_layers=3,
        ...     vae_prior_hidden_dims=[64, 64]
        ... )
        >>> print(vae_config)
        {
            'vae_latent_dim': 3,
            'vae_prior_type': 'decoupled',
            'vae_prior_num_layers': 3,
            'vae_prior_hidden_dims': [64, 64]
        }
        """
        # Collect all VAE parameters
        vae_params = {
            "vae_latent_dim": vae_latent_dim,
            "vae_hidden_dims": vae_hidden_dims,
            "vae_activation": vae_activation,
            "vae_input_transformation": vae_input_transformation,
            "vae_vcp_hidden_dims": vae_vcp_hidden_dims,
            "vae_vcp_activation": vae_vcp_activation,
            "vae_prior_type": vae_prior_type,
            "vae_prior_num_layers": vae_prior_num_layers,
            "vae_prior_hidden_dims": vae_prior_hidden_dims,
            "vae_prior_activation": vae_prior_activation,
            "vae_prior_mask_type": vae_prior_mask_type,
            "vae_standardize": vae_standardize,
        }

        # Return only non-None values
        return ParameterCollector.collect_non_none(**vae_params)
