"""
Results classes for SCRIBE inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass, replace
import warnings

import jax.numpy as jnp
import jax.scipy as jsp
from jax.nn import sigmoid, softmax
import pandas as pd
import numpyro.distributions as dist
from jax import random, jit, vmap

import numpy as np
import scipy.stats as stats

from ..sampling import (
    sample_variational_posterior,
    generate_predictive_samples,
)
from ..stats import (
    fit_dirichlet_minka,
    get_distribution_mode,
    hellinger_gamma,
    hellinger_lognormal,
    kl_gamma,
    kl_lognormal,
    jensen_shannon_gamma,
    jensen_shannon_lognormal,
)
from ..models.model_config import ModelConfig
from ..models.standard import get_posterior_distributions
from ..utils import numpyro_to_scipy


from ..core.normalization import normalize_counts_from_posterior

try:
    from anndata import AnnData
except ImportError:
    AnnData = None

# ------------------------------------------------------------------------------
# Base class for inference results
# ------------------------------------------------------------------------------


@dataclass
class ScribeSVIResults:
    """
    Base class for SCRIBE variational inference results.

    This class stores the results from SCRIBE's variational inference procedure,
    including model parameters, loss history, dataset dimensions, and model
    configuration. It can optionally store metadata from an AnnData object and
    posterior/predictive samples.

    Attributes
    ----------
    params : Dict
        Dictionary of inferred model parameters from SCRIBE
    loss_history : jnp.ndarray
        Array containing the ELBO loss values during training
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_type : str
        Type of model used for inference
    model_config : ModelConfig
        Configuration object specifying model architecture and priors
    prior_params : Dict[str, Any]
        Dictionary of prior parameter values used during inference
    obs : Optional[pd.DataFrame]
        Cell-level metadata from adata.obs, if provided
    var : Optional[pd.DataFrame]
        Gene-level metadata from adata.var, if provided
    uns : Optional[Dict]
        Unstructured metadata from adata.uns, if provided
    n_obs : Optional[int]
        Number of observations (cells), if provided
    n_vars : Optional[int]
        Number of variables (genes), if provided
    posterior_samples : Optional[Dict]
        Samples of parameters from the posterior distribution, if generated
    predictive_samples : Optional[Dict]
        Predictive samples generated from the model, if generated
    n_components : Optional[int]
        Number of mixture components, if using a mixture model
    """

    # Core inference results
    params: Dict
    loss_history: jnp.ndarray
    n_cells: int
    n_genes: int
    model_type: str
    model_config: ModelConfig
    prior_params: Dict[str, Any]

    # Standard metadata from AnnData object
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

    # Optional results
    posterior_samples: Optional[Dict] = None
    predictive_samples: Optional[Dict] = None
    n_components: Optional[int] = None

    # --------------------------------------------------------------------------

    def __post_init__(self):
        """Validate model configuration and parameters."""
        # Set n_components from model_config if not explicitly provided
        if (
            self.n_components is None
            and self.model_config.n_components is not None
        ):
            self.n_components = self.model_config.n_components

        self._validate_model_config()

    # --------------------------------------------------------------------------

    def _validate_model_config(self):
        """Validate model configuration matches model type."""
        # Validate base model
        if self.model_config.base_model != self.model_type:
            raise ValueError(
                f"Model type '{self.model_type}' does not match config "
                f"base model '{self.model_config.base_model}'"
            )

        # Validate n_components consistency
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

        # Validate required distributions based on model type and parameterization
        parameterization = self.model_config.parameterization

        # ZINB models require gate priors
        if "zinb" in self.model_type:
            if parameterization == "unconstrained":
                # Unconstrained uses gate_unconstrained_prior
                if self.model_config.gate_unconstrained_prior is None:
                    raise ValueError(
                        "ZINB models with unconstrained parameterization require gate_unconstrained_prior"
                    )
            else:
                # Standard, linked, odds_ratio use gate_param_prior
                if self.model_config.gate_param_prior is None:
                    raise ValueError("ZINB models require gate_param_prior")
        else:
            # Non-ZINB models should not have gate priors
            if parameterization == "unconstrained":
                if self.model_config.gate_unconstrained_prior is not None:
                    raise ValueError(
                        "Non-ZINB models should not have gate_unconstrained_prior"
                    )
            else:
                if self.model_config.gate_param_prior is not None:
                    raise ValueError(
                        "Non-ZINB models should not have gate_param_prior"
                    )

        # VCP models require capture probability priors
        if "vcp" in self.model_type:
            if parameterization == "unconstrained":
                # Unconstrained uses p_capture_unconstrained_prior
                if self.model_config.p_capture_unconstrained_prior is None:
                    raise ValueError(
                        "VCP models with unconstrained parameterization require p_capture_unconstrained_prior"
                    )
            elif parameterization in ["standard", "linked"]:
                # Standard and linked use p_capture_param_prior
                if self.model_config.p_capture_param_prior is None:
                    raise ValueError("VCP models require p_capture_param_prior")
            elif parameterization == "odds_ratio":
                # Odds_ratio uses phi_capture_param_prior
                if self.model_config.phi_capture_param_prior is None:
                    raise ValueError(
                        "VCP models with odds_ratio parameterization require phi_capture_param_prior"
                    )
        else:
            # Non-VCP models should not have capture probability priors
            if parameterization == "unconstrained":
                if self.model_config.p_capture_unconstrained_prior is not None:
                    raise ValueError(
                        "Non-VCP models should not have p_capture_unconstrained_prior"
                    )
            elif parameterization in ["standard", "linked"]:
                if self.model_config.p_capture_param_prior is not None:
                    raise ValueError(
                        "Non-VCP models should not have p_capture_param_prior"
                    )
            elif parameterization == "odds_ratio":
                if self.model_config.phi_capture_param_prior is not None:
                    raise ValueError(
                        "Non-VCP models should not have phi_capture_param_prior"
                    )

    # --------------------------------------------------------------------------
    # Create ScribeSVIResults from AnnData object
    # --------------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        adata: Any,
        params: Dict,
        loss_history: jnp.ndarray,
        model_config: ModelConfig,
        **kwargs,
    ):
        """Create ScribeSVIResults from AnnData object."""
        return cls(
            params=params,
            loss_history=loss_history,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            model_config=model_config,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
            **kwargs,
        )

    # --------------------------------------------------------------------------
    # Get distributions using configs
    # --------------------------------------------------------------------------

    def get_distributions(
        self,
        backend: str = "numpyro",
        split: bool = False,
    ) -> Dict[str, Any]:
        """
        Get the variational distributions for all parameters.

        This method now delegates to the model-specific `get_posterior_distributions`
        function associated with the parameterization.

        Parameters
        ----------
        backend : str, default="numpyro"
            Statistical package to use for distributions. Must be one of:
            - "scipy": Returns scipy.stats distributions
            - "numpyro": Returns numpyro.distributions
        split : bool, default=False
            If True, returns lists of individual distributions for
            multidimensional parameters instead of batch distributions.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to their distributions.

        Raises
        ------
        ValueError
            If backend is not supported.
        """
        if backend not in ["scipy", "numpyro"]:
            raise ValueError(f"Invalid backend: {backend}")

        # Dynamically import the correct posterior distribution function
        if self.model_config.parameterization == "standard":
            from ..models.standard import (
                get_posterior_distributions as get_dist_fn,
            )
        elif self.model_config.parameterization == "linked":
            from ..models.linked import (
                get_posterior_distributions as get_dist_fn,
            )
        elif self.model_config.parameterization == "odds_ratio":
            from ..models.odds_ratio import (
                get_posterior_distributions as get_dist_fn,
            )
        elif self.model_config.parameterization == "unconstrained":
            from ..models.unconstrained import (
                get_posterior_distributions as get_dist_fn,
            )
        else:
            raise NotImplementedError(
                f"get_distributions not implemented for '{self.model_config.parameterization}'."
            )

        distributions = get_dist_fn(self.params, self.model_config, split=split)

        if backend == "scipy":
            # Handle conversion to scipy, accounting for split distributions
            scipy_distributions = {}
            for name, dist_obj in distributions.items():
                if isinstance(dist_obj, list):
                    # Handle split distributions - convert each element
                    if all(isinstance(sublist, list) for sublist in dist_obj):
                        # Handle nested lists (2D case: components × genes)
                        scipy_distributions[name] = [
                            [numpyro_to_scipy(d) for d in sublist]
                            for sublist in dist_obj
                        ]
                    else:
                        # Handle simple lists (1D case: genes or components)
                        scipy_distributions[name] = [
                            numpyro_to_scipy(d) for d in dist_obj
                        ]
                else:
                    # Handle single distribution
                    scipy_distributions[name] = numpyro_to_scipy(dist_obj)
            return scipy_distributions

        return distributions

    # --------------------------------------------------------------------------

    def get_map(
        self,
        use_mean: bool = False,
        canonical: bool = True,
        verbose: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational
        posterior.

        Parameters
        ----------
        use_mean : bool, default=False
            If True, replaces undefined MAP values (NaN) with posterior means
        canonical : bool, default=True
            If True, includes canonical parameters (p, r) computed from other parameters
            for linked, odds_ratio, and unconstrained parameterizations
        verbose : bool, default=True
            If True, prints a warning if NaNs were replaced with means

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary of MAP estimates for each parameter
        """
        # Get distributions with NumPyro backend
        distributions = self.get_distributions(backend="numpyro")
        # Get estimate of map
        map_estimates = {
            param: get_distribution_mode(dist)
            for param, dist in distributions.items()
        }

        # Replace NaN values with means if requested
        if use_mean:
            # Initialize boolean to track if any NaNs were replaced
            replaced_nans = False
            # Check each parameter for NaNs and replace with means
            for param, value in map_estimates.items():
                # Check if any values are NaN
                if jnp.any(jnp.isnan(value)):
                    replaced_nans = True
                    # Get mean value
                    mean_value = distributions[param].mean
                    # Replace NaN values with means
                    map_estimates[param] = jnp.where(
                        jnp.isnan(value), mean_value, value
                    )
            # Print warning if NaNs were replaced
            if replaced_nans and verbose:
                warnings.warn(
                    "NaN values were replaced with means of the distributions",
                    UserWarning,
                )

        # Compute canonical parameters if requested
        if canonical:
            map_estimates = self._compute_canonical_parameters(
                map_estimates, verbose=verbose
            )

        return map_estimates

    # --------------------------------------------------------------------------

    def _compute_canonical_parameters(
        self, map_estimates: Dict, verbose: bool = True
    ) -> Dict:
        """
        Compute canonical parameters (p, r) from other parameters for different parameterizations.

        Parameters
        ----------
        map_estimates : Dict
            Dictionary containing MAP estimates
        verbose : bool, default=True
            If True, prints information about parameter computation

        Returns
        -------
        Dict
            Updated dictionary with canonical parameters computed
        """
        estimates = map_estimates.copy()
        parameterization = self.model_config.parameterization

        # Handle linked parameterization
        if parameterization == "linked":
            if "mu" in estimates and "p" in estimates and "r" not in estimates:
                if verbose:
                    print(
                        "Computing r from mu and p for linked parameterization"
                    )
                # r = mu * (1 - p) / p
                estimates["r"] = (
                    estimates["mu"] * (1 - estimates["p"]) / estimates["p"]
                )

        # Handle odds_ratio parameterization
        elif parameterization == "odds_ratio":
            # Convert phi to p if needed
            if "phi" in estimates and "p" not in estimates:
                if verbose:
                    print(
                        "Computing p from phi for odds_ratio parameterization"
                    )
                estimates["p"] = 1.0 / (1.0 + estimates["phi"])

            # Convert phi and mu to r if needed
            if (
                "phi" in estimates
                and "mu" in estimates
                and "r" not in estimates
            ):
                if verbose:
                    print(
                        "Computing r from phi and mu for odds_ratio parameterization"
                    )
                # Reshape phi to broadcast with mu based on mixture model
                if (
                    self.n_components is not None
                    and self.model_config.component_specific_params
                ):
                    # Mixture model: mu has shape (n_components, n_genes)
                    phi_reshaped = estimates["phi"][:, None]
                else:
                    # Non-mixture model: mu has shape (n_genes,)
                    phi_reshaped = estimates["phi"]
                estimates["r"] = estimates["mu"] * phi_reshaped

            # Handle VCP capture probability conversion
            if "phi_capture" in estimates and "p_capture" not in estimates:
                if verbose:
                    print(
                        "Computing p_capture from phi_capture for odds_ratio parameterization"
                    )
                estimates["p_capture"] = 1.0 / (1.0 + estimates["phi_capture"])

        # Handle unconstrained parameterization
        elif parameterization == "unconstrained":
            # Convert r_unconstrained to r if needed
            if "r_unconstrained" in estimates and "r" not in estimates:
                if verbose:
                    print(
                        "Computing r from r_unconstrained for unconstrained parameterization"
                    )
                estimates["r"] = jnp.exp(estimates["r_unconstrained"])

            # Convert p_unconstrained to p if needed
            if "p_unconstrained" in estimates and "p" not in estimates:
                if verbose:
                    print(
                        "Computing p from p_unconstrained for unconstrained parameterization"
                    )
                estimates["p"] = sigmoid(estimates["p_unconstrained"])

            # Convert gate_unconstrained to gate if needed
            if "gate_unconstrained" in estimates and "gate" not in estimates:
                if verbose:
                    print(
                        "Computing gate from gate_unconstrained for unconstrained parameterization"
                    )
                estimates["gate"] = sigmoid(estimates["gate_unconstrained"])

            # Handle VCP capture probability conversion
            if (
                "p_capture_unconstrained" in estimates
                and "p_capture" not in estimates
            ):
                if verbose:
                    print(
                        "Computing p_capture from p_capture_unconstrained for unconstrained parameterization"
                    )
                estimates["p_capture"] = sigmoid(
                    estimates["p_capture_unconstrained"]
                )
            # Handle mixing weights computation for mixture models
            if (
                "mixing_logits_unconstrained" in estimates
                and "mixing_weights" not in estimates
            ):
                # Compute mixing weights from mixing_logits_unconstrained using
                # softmax
                estimates["mixing_weights"] = softmax(
                    estimates["mixing_logits_unconstrained"], axis=-1
                )

        # Compute p_hat for NBVCP and ZINBVCP models if needed (applies to all parameterizations)
        if (
            "p" in estimates
            and "p_capture" in estimates
            and "p_hat" not in estimates
        ):
            if verbose:
                print("Computing p_hat from p and p_capture")

            # Reshape p_capture for broadcasting
            p_capture_reshaped = estimates["p_capture"][:, None]

            # p_hat = p * p_capture / (1 - p * (1 - p_capture))
            estimates["p_hat"] = (
                estimates["p"]
                * p_capture_reshaped
                / (1 - estimates["p"] * (1 - p_capture_reshaped))
            ).flatten()

        return estimates

    # --------------------------------------------------------------------------
    # Indexing by genes
    # --------------------------------------------------------------------------

    @staticmethod
    def _subset_gene_params(params, param_prefixes, index, n_components=None):
        """
        Utility to subset all gene-specific parameters in params dict.
        param_prefixes: list of parameter name prefixes (e.g., ["r_", "mu_",
        "gate_"]) index: boolean or integer index for genes n_components: if not
        None, keep component dimension
        """
        new_params = dict(params)
        for prefix, arg_constraints in param_prefixes:
            if arg_constraints is None:
                continue
            for param_name in arg_constraints:
                key = f"{prefix}{param_name}"
                if key in params:
                    if n_components is not None:
                        new_params[key] = params[key][..., index]
                    else:
                        new_params[key] = params[key][index]
        return new_params

    # --------------------------------------------------------------------------

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create a new parameter dictionary for the given index.
        """
        new_params = dict(params)
        gene_specific_params = [
            "r_loc",
            "r_scale",
            "gate_alpha",
            "gate_beta",
            "mu_loc",
            "mu_scale",
            "r_unconstrained_loc",
            "r_unconstrained_scale",
            "gate_unconstrained_loc",
            "gate_unconstrained_scale",
        ]

        for param_name in gene_specific_params:
            if param_name in params:
                param = params[param_name]
                if param.ndim == 1:  # (n_genes,)
                    new_params[param_name] = param[index]
                elif param.ndim == 2:  # (n_components, n_genes)
                    new_params[param_name] = param[:, index]

        return new_params

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create a new posterior samples dictionary for the given index.
        """
        if samples is None:
            return None
        # List of gene-specific keys to subset
        gene_keys = []
        # r
        if "r" in samples:
            gene_keys.append("r")
        # mu
        if "mu" in samples:
            gene_keys.append("mu")
        # gate
        if "gate" in samples:
            gene_keys.append("gate")
        # r_unconstrained
        if "r_unconstrained" in samples:
            gene_keys.append("r_unconstrained")
        # gate_unconstrained
        if "gate_unconstrained" in samples:
            gene_keys.append("gate_unconstrained")
        # (extend here for other gene-specific keys if needed)
        new_samples = dict(samples)
        for key in gene_keys:
            if self.n_components is not None:
                new_samples[key] = samples[key][..., index]
            else:
                new_samples[key] = samples[key][..., index]
        return new_samples

    # --------------------------------------------------------------------------

    def _subset_predictive_samples(
        self, samples: jnp.ndarray, index
    ) -> jnp.ndarray:
        """Create a new predictive samples array for the given index."""
        if samples is None:
            return None

        # For predictive samples, subset the gene dimension (last dimension)
        return samples[..., index]

    # --------------------------------------------------------------------------

    def __getitem__(self, index):
        """
        Enable indexing of ScribeSVIResults object.
        """
        # If index is a boolean mask, use it directly
        if isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool:
            bool_index = index
        # Handle integer indexing
        elif isinstance(index, int):
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given index
            bool_index = bool_index.at[index].set(True)
        # Handle slice indexing
        elif isinstance(index, slice):
            # Get indices from slice
            indices = jnp.arange(self.n_genes)[index]
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given indices
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        # Handle list/array indexing (by integer indices)
        elif isinstance(index, (list, np.ndarray, jnp.ndarray)) and not (
            isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool
        ):
            indices = jnp.array(index)
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

        # Create new params dict with subset of parameters
        new_params = self._subset_params(self.params, bool_index)

        # Create new metadata if available
        new_var = self.var.iloc[bool_index] if self.var is not None else None

        # Create new posterior samples if available
        new_posterior_samples = (
            self._subset_posterior_samples(self.posterior_samples, bool_index)
            if self.posterior_samples is not None
            else None
        )

        # Create new predictive samples if available
        new_predictive_samples = (
            self._subset_predictive_samples(self.predictive_samples, bool_index)
            if self.predictive_samples is not None
            else None
        )

        # Create new instance with subset data
        return self._create_subset(
            index=bool_index,
            new_params=new_params,
            new_var=new_var,
            new_posterior_samples=new_posterior_samples,
            new_predictive_samples=new_predictive_samples,
        )

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_var: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict],
        new_predictive_samples: Optional[jnp.ndarray],
    ) -> "ScribeSVIResults":
        """Create a new instance with a subset of genes."""
        return type(self)(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, "sum") else len(index)),
            model_type=self.model_type,
            model_config=self.model_config,
            prior_params=self.prior_params,
            obs=self.obs,
            var=new_var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=new_var.shape[0] if new_var is not None else None,
            posterior_samples=new_posterior_samples,
            predictive_samples=new_predictive_samples,
            n_components=self.n_components,
        )

    # --------------------------------------------------------------------------
    # Indexing by component
    # --------------------------------------------------------------------------

    def get_component(self, component_index):
        """
        Create a view of the results selecting a specific mixture component.

        This method returns a new ScribeSVIResults object that contains parameter
        values for the specified component, allowing for further gene-based
        indexing. Only applicable to mixture models.

        Parameters
        ----------
        component_index : int
            Index of the component to select

        Returns
        -------
        ScribeSVIResults
            A new ScribeSVIResults object with parameters for the selected component

        Raises
        ------
        ValueError
            If the model is not a mixture model
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Component view only applies to mixture models with multiple components"
            )

        # Check if component_index is valid
        if component_index < 0 or component_index >= self.n_components:
            raise ValueError(
                f"Component index {component_index} out of range [0, {self.n_components-1}]"
            )

        # Create new params dict with component subset
        new_params = dict(self.params)

        # Handle all parameters based on their structure
        self._subset_params_by_component(new_params, component_index)

        # Create new posterior samples if available
        new_posterior_samples = None
        if self.posterior_samples is not None:
            new_posterior_samples = self._subset_posterior_samples_by_component(
                self.posterior_samples, component_index
            )

        # Create new predictive samples if available - this is more complex
        # as we would need to condition on the component
        new_predictive_samples = None

        # Create new instance with component subset
        return self._create_component_subset(
            component_index=component_index,
            new_params=new_params,
            new_posterior_samples=new_posterior_samples,
            new_predictive_samples=new_predictive_samples,
        )

    # --------------------------------------------------------------------------

    def _subset_params_by_component(
        self, new_params: Dict, component_index: int
    ):
        """
        Handle subsetting of all parameters based on their structure.

        This method intelligently handles parameters based on their dimensions
        and naming conventions, regardless of parameterization.
        """
        # Define parameter categories based on their structure
        component_gene_specific = [
            "r_loc_comp",
            "r_scale_comp",
            "r_unconstrained_loc",
            "r_unconstrained_scale",
            "gate_alpha_comp",
            "gate_beta_comp",
            "gate_unconstrained_loc",
            "gate_unconstrained_scale",
            "mu_loc",
            "mu_scale",
            "phi_loc",
            "phi_scale",
        ]

        component_specific = [
            "p_alpha_comp",
            "p_beta_comp",
            "p_unconstrained_loc",
            "p_unconstrained_scale",
            "mixing_logits_unconstrained_loc",
            "mixing_logits_unconstrained_scale",
        ]

        cell_specific = [
            "p_capture_alpha",
            "p_capture_beta",
            "p_capture_unconstrained_loc",
            "p_capture_unconstrained_scale",
            "phi_capture_loc",
            "phi_capture_scale",
        ]

        shared_params = [
            "p_alpha",
            "p_beta",
            "gate_alpha",
            "gate_beta",
            "mixing_conc",
        ]

        # Handle component-gene-specific parameters (shape: [n_components, n_genes])
        for param_name in component_gene_specific:
            if param_name in self.params:
                param = self.params[param_name]
                # Check if parameter has component dimension
                if param.ndim > 1:  # Has component dimension
                    new_params[param_name] = param[component_index]
                else:  # Scalar parameter, copy as-is
                    new_params[param_name] = param

        # Handle component-specific parameters (shape: [n_components])
        for param_name in component_specific:
            if param_name in self.params:
                param = self.params[param_name]
                # Check if parameter has component dimension
                if param.ndim > 0:  # Has component dimension
                    new_params[param_name] = param[component_index]
                else:  # Scalar parameter, copy as-is
                    new_params[param_name] = param

        # Handle cell-specific parameters (copy as-is, not component-specific)
        for param_name in cell_specific:
            if param_name in self.params:
                new_params[param_name] = self.params[param_name]

        # Handle shared parameters (copy as-is, used across all components)
        for param_name in shared_params:
            if param_name in self.params:
                new_params[param_name] = self.params[param_name]

    # --------------------------------------------------------------------------

    def _subset_posterior_samples_by_component(
        self, samples: Dict, component_index: int
    ) -> Dict:
        """
        Create a new posterior samples dictionary for the given component index.

        This method handles all parameter types based on their dimensions.
        """
        if samples is None:
            return None

        new_posterior_samples = {}

        # Define parameter categories for posterior samples
        component_gene_specific_samples = [
            "r",
            "r_unconstrained",
            "gate",
            "gate_unconstrained",
            "mu",
            "phi",
            "r_unconstrained_loc",
            "r_unconstrained_scale",
            "gate_unconstrained_loc",
            "gate_unconstrained_scale",
        ]

        component_specific_samples = [
            "p",
            "p_unconstrained",
            "mixing_weights",
            "mixing_logits_unconstrained",
            "p_unconstrained_loc",
            "p_unconstrained_scale",
            "mixing_logits_unconstrained_loc",
            "mixing_logits_unconstrained_scale",
        ]

        cell_specific_samples = [
            "p_capture",
            "p_capture_unconstrained",
            "phi_capture",
            "p_capture_unconstrained_loc",
            "p_capture_unconstrained_scale",
        ]

        # Handle component-gene-specific samples (shape: [n_samples, n_components, n_genes])
        for param_name in component_gene_specific_samples:
            if param_name in samples:
                sample_value = samples[param_name]
                if sample_value.ndim > 2:  # Has component dimension
                    new_posterior_samples[param_name] = sample_value[
                        :, component_index, :
                    ]
                else:  # Scalar parameter, copy as-is
                    new_posterior_samples[param_name] = sample_value

        # Handle component-specific samples (shape: [n_samples, n_components])
        for param_name in component_specific_samples:
            if param_name in samples:
                sample_value = samples[param_name]
                if sample_value.ndim > 1:  # Has component dimension
                    new_posterior_samples[param_name] = sample_value[
                        :, component_index
                    ]
                else:  # Scalar parameter, copy as-is
                    new_posterior_samples[param_name] = sample_value

        # Handle cell-specific samples (copy as-is, not component-specific)
        for param_name in cell_specific_samples:
            if param_name in samples:
                new_posterior_samples[param_name] = samples[param_name]

        return new_posterior_samples

    # --------------------------------------------------------------------------

    def _create_component_subset(
        self,
        component_index,
        new_params: Dict,
        new_posterior_samples: Optional[Dict],
        new_predictive_samples: Optional[jnp.ndarray],
    ) -> "ScribeSVIResults":
        """Create a new instance for a specific component."""
        # Create a non-mixture model type
        base_model = self.model_type.replace("_mix", "")

        # Create a modified model config with n_components=None to indicate
        # this is now a non-mixture result after component selection
        new_model_config = replace(
            self.model_config,
            base_model=base_model,
            n_components=None,
        )

        return type(self)(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=base_model,  # Remove _mix suffix
            model_config=new_model_config,
            prior_params=self.prior_params,
            obs=self.obs,
            var=self.var,
            uns=self.uns,       
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            posterior_samples=new_posterior_samples,
            predictive_samples=new_predictive_samples,
            n_components=None,  # No longer a mixture model
        )

    # --------------------------------------------------------------------------
    # Get model and guide functions
    # --------------------------------------------------------------------------

    def _model_and_guide(self) -> Tuple[Callable, Optional[Callable]]:
        """Get the model and guide functions based on model type."""
        from ..models.model_registry import get_model_and_guide

        parameterization = self.model_config.parameterization or ""
        inference_method = self.model_config.inference_method or ""
        prior_type = self.model_config.vae_prior_type or ""
        return get_model_and_guide(
            self.model_type, parameterization, inference_method, prior_type
        )

    # --------------------------------------------------------------------------
    # Get parameterization
    # --------------------------------------------------------------------------

    def _parameterization(self) -> str:
        """Get the parameterization type."""
        return self.model_config.parameterization or ""

    # --------------------------------------------------------------------------
    # Get log likelihood function
    # --------------------------------------------------------------------------

    def _log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for this model type."""
        from ..models.model_registry import get_log_likelihood_fn

        return get_log_likelihood_fn(self.model_type)

    # --------------------------------------------------------------------------
    # Posterior sampling methods
    # --------------------------------------------------------------------------

    def get_posterior_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        store_samples: bool = True,
        canonical: bool = False,
    ) -> Dict:
        """Sample parameters from the variational posterior distribution."""
        # Get the guide function
        _, guide = self._model_and_guide()

        if guide is None:
            raise ValueError(
                f"Could not find a guide for model '{self.model_type}'."
            )

        # Prepare base model arguments
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "model_config": self.model_config,
        }

        # Sample from posterior
        posterior_samples = sample_variational_posterior(
            guide, self.params, model_args, rng_key=rng_key, n_samples=n_samples
        )

        # Store samples if requested
        if store_samples:
            self.posterior_samples = posterior_samples

        # Convert to canonical form if requested
        if canonical:
            self._convert_to_canonical()

        return posterior_samples

    # --------------------------------------------------------------------------

    def get_predictive_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples."""
        # Get the model and guide functions
        model, _ = self._model_and_guide()

        # Prepare base model arguments
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "model_config": self.model_config,
        }

        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Generate predictive samples
        predictive_samples = generate_predictive_samples(
            model,
            self.posterior_samples,
            model_args,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        # Store samples if requested
        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

    # --------------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        resample_parameters: bool = False,
        canonical: bool = False,
    ) -> Dict:
        """Generate posterior predictive check samples."""
        # Check if we need to resample parameters
        need_params = resample_parameters or self.posterior_samples is None

        # Generate posterior samples if needed
        if need_params:
            # Sample parameters and generate predictive samples
            self.get_posterior_samples(
                rng_key=rng_key,
                n_samples=n_samples,
                store_samples=store_samples,
                canonical=canonical,
            )

        # Generate predictive samples using existing parameters
        _, key_pred = random.split(rng_key)

        self.get_predictive_samples(
            rng_key=key_pred,
            batch_size=batch_size,
            store_samples=store_samples,
        )

        return {
            "parameter_samples": self.posterior_samples,
            "predictive_samples": self.predictive_samples,
        }

    # --------------------------------------------------------------------------
    # Compute log likelihood methods
    # --------------------------------------------------------------------------

    def log_likelihood(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        return_by: str = "cell",
        cells_axis: int = 0,
        ignore_nans: bool = False,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute log likelihood of data under posterior samples.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate likelihood on
        batch_size : Optional[int], default=None
            Size of mini-batches used for likelihood computation
        return_by : str, default='cell'
            Specifies how to return the log probabilities. Must be one of:
                - 'cell': returns log probabilities summed over genes
                - 'gene': returns log probabilities summed over cells
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        ignore_nans : bool, default=False
            If True, removes any samples that contain NaNs.
        split_components : bool, default=False
            If True, returns log likelihoods for each mixture component
            separately. Only applicable for mixture models.
        weights : Optional[jnp.ndarray], default=None
            Array used to weight the log likelihoods (for mixture models).
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        jnp.ndarray
            Array of log likelihoods. Shape depends on model type, return_by and
            split_components parameters. For standard models:
                - 'cell': shape (n_samples, n_cells)
                - 'gene': shape (n_samples, n_genes)
            For mixture models with split_components=False:
                - 'cell': shape (n_samples, n_cells)
                - 'gene': shape (n_samples, n_genes)
            For mixture models with split_components=True:
                - 'cell': shape (n_samples, n_cells, n_components)
                - 'gene': shape (n_samples, n_genes, n_components)

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet
        """
        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Convert posterior samples to canonical form
        self._convert_to_canonical()

        # Get parameter samples
        parameter_samples = self.posterior_samples

        # Get number of samples from first parameter
        n_samples = parameter_samples[next(iter(parameter_samples))].shape[0]

        # Get likelihood function
        likelihood_fn = self._log_likelihood_fn()

        # Determine if this is a mixture model
        is_mixture = self.n_components is not None and self.n_components > 1

        # Define function to compute likelihood for a single sample
        @jit
        def compute_sample_lik(i):
            # Extract parameters for this sample
            params_i = {k: v[i] for k, v in parameter_samples.items()}
            # For mixture models we need to pass split_components and weights
            if is_mixture:
                return likelihood_fn(
                    counts,
                    params_i,
                    batch_size=batch_size,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    split_components=split_components,
                    weights=weights,
                    weight_type=weight_type,
                    dtype=dtype,
                )
            else:
                return likelihood_fn(
                    counts,
                    params_i,
                    batch_size=batch_size,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    dtype=dtype,
                )

        # Use vmap for parallel computation (more memory intensive)
        log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))

        # Handle NaNs if requested
        if ignore_nans:
            # Check for NaNs appropriately based on dimensions
            if is_mixture and split_components:
                # Handle case with component dimension
                valid_samples = ~jnp.any(
                    jnp.any(jnp.isnan(log_liks), axis=-1), axis=-1
                )
            else:
                # Standard case
                valid_samples = ~jnp.any(jnp.isnan(log_liks), axis=-1)

            # Filter out samples with NaNs
            if jnp.any(~valid_samples):
                print(
                    f"    - Fraction of samples removed: {1 - jnp.mean(valid_samples)}"
                )
                return log_liks[valid_samples]

        return log_liks

    # --------------------------------------------------------------------------

    def log_likelihood_map(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        gene_batch_size: Optional[int] = None,
        return_by: str = "cell",
        cells_axis: int = 0,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        use_mean: bool = True,
        verbose: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute log likelihood of data using MAP parameter estimates.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate likelihood on
        batch_size : Optional[int], default=None
            Size of mini-batches used for likelihood computation
        gene_batch_size : Optional[int], default=None
            Size of mini-batches used for likelihood computation by gene
        return_by : str, default='cell'
            Specifies how to return the log probabilities. Must be one of:
                - 'cell': returns log probabilities summed over genes
                - 'gene': returns log probabilities summed over cells
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        split_components : bool, default=False
            If True, returns log likelihoods for each mixture component separately.
            Only applicable for mixture models.
        weights : Optional[jnp.ndarray], default=None
            Array used to weight the log likelihoods (for mixture models).
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        use_mean : bool, default=False
            If True, replaces undefined MAP values (NaN) with posterior means
        verbose : bool, default=True
            If True, prints a warning if NaNs were replaced with means
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        jnp.ndarray
            Array of log likelihoods. Shape depends on model type, return_by and
            split_components parameters.
        """
        # Get the log likelihood function
        likelihood_fn = self._log_likelihood_fn()

        # Determine if this is a mixture model
        is_mixture = self.n_components is not None and self.n_components > 1

        # Get the MAP estimates with canonical parameters included
        map_estimates = self.get_map(
            use_mean=use_mean, canonical=True, verbose=verbose
        )

        # If computing by gene and gene_batch_size is provided, use batched computation
        if return_by == "gene" and gene_batch_size is not None:
            # Determine output shape
            if (
                is_mixture
                and split_components
                and self.n_components is not None
            ):
                result_shape = (self.n_genes, self.n_components)
            else:
                result_shape = (self.n_genes,)

            # Initialize result array
            log_liks = np.zeros(result_shape, dtype=dtype)

            # Process genes in batches
            for i in range(0, self.n_genes, gene_batch_size):
                if verbose and i > 0:
                    print(
                        f"Processing genes {i}-{min(i+gene_batch_size, self.n_genes)} of {self.n_genes}"
                    )

                # Get gene indices for this batch
                end_idx = min(i + gene_batch_size, self.n_genes)
                gene_indices = list(range(i, end_idx))

                # Get subset of results for these genes
                results_subset = self[gene_indices]
                # Get the MAP estimates for this subset (with canonical parameters)
                subset_map_estimates = results_subset.get_map(
                    use_mean=use_mean, canonical=True, verbose=False
                )

                # Get subset of counts for these genes
                if cells_axis == 0:
                    counts_subset = counts[:, gene_indices]
                else:
                    counts_subset = counts[gene_indices, :]

                # Get subset of weights if provided
                weights_subset = None
                if weights is not None:
                    if weights.ndim == 1:  # Shape: (n_genes,)
                        weights_subset = weights[gene_indices]
                    else:
                        weights_subset = weights

                # Compute log likelihood for this gene batch
                if is_mixture:
                    batch_log_liks = likelihood_fn(
                        counts_subset,
                        subset_map_estimates,
                        batch_size=batch_size,
                        cells_axis=cells_axis,
                        return_by=return_by,
                        split_components=split_components,
                        weights=weights_subset,
                        weight_type=weight_type,
                        dtype=dtype,
                    )
                else:
                    batch_log_liks = likelihood_fn(
                        counts_subset,
                        subset_map_estimates,
                        batch_size=batch_size,
                        cells_axis=cells_axis,
                        return_by=return_by,
                        dtype=dtype,
                    )

                # Store results
                log_liks[i:end_idx] = np.array(batch_log_liks)

            # Convert to JAX array for consistency
            return jnp.array(log_liks)

        # Standard computation (no gene batching)
        else:
            # Compute log-likelihood for mixture model
            if is_mixture:
                log_liks = likelihood_fn(
                    counts,
                    map_estimates,
                    batch_size=batch_size,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    split_components=split_components,
                    weights=weights,
                    weight_type=weight_type,
                    dtype=dtype,
                )
            # Compute log-likelihood for non-mixture model
            else:
                log_liks = likelihood_fn(
                    counts,
                    map_estimates,
                    batch_size=batch_size,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    dtype=dtype,
                )

            return log_liks

    # --------------------------------------------------------------------------
    # Compute entropy of component assignments
    # --------------------------------------------------------------------------

    def mixture_component_entropy(
        self,
        counts: jnp.ndarray,
        return_by: str = "gene",
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        ignore_nans: bool = False,
        temperature: Optional[float] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute the entropy of mixture component assignment probabilities.

        This method calculates the Shannon entropy of the posterior component
        assignment probabilities for each observation (cell or gene), providing
        a quantitative measure of assignment uncertainty in mixture models.

        The entropy quantifies how uncertain the model is about which component
        each observation belongs to:
            - Low entropy (≈0): High confidence in component assignment
            - High entropy (≈log(n_components)): High uncertainty in assignment
            - Maximum entropy: Uniform assignment probabilities across all
              components

        The entropy is calculated as:
            H = -∑(p_i * log(p_i))
        where p_i are the posterior probabilities of assignment to component i.

        Parameters
        ----------
        counts : jnp.ndarray
            Input count data to evaluate component assignments for. Shape should
            be (n_cells, n_genes) if cells_axis=0, or (n_genes, n_cells) if
            cells_axis=1.

        return_by : str, default='gene'
            Specifies how to compute and return the entropy. Must be one of:
                - 'cell': Compute entropy of component assignments for each cell
                - 'gene': Compute entropy of component assignments for each gene

        batch_size : Optional[int], default=None
            If provided, processes the data in batches of this size to reduce
            memory usage. Useful for large datasets.

        cells_axis : int, default=0
            Specifies which axis in the input counts contains the cells:
                - 0: cells are rows (shape: n_cells × n_genes)
                - 1: cells are columns (shape: n_genes × n_cells)

        ignore_nans : bool, default=False
            If True, excludes any samples containing NaN values from the entropy
            calculation.

        temperature : Optional[float], default=None
            If provided, applies temperature scaling to the log-likelihoods
            before computing entropy. Temperature scaling modifies the sharpness
            of probability distributions by dividing log probabilities by a
            temperature parameter T.

        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations.

        Returns
        -------
        jnp.ndarray
            Array of entropy values. Shape depends on return_by:
                - If return_by='cell': shape is (n_samples, n_cells)
                - If return_by='gene': shape is (n_samples, n_genes)
            Higher values indicate more uncertainty in component assignments.

        Raises
        ------
        ValueError
            If the model is not a mixture model or if posterior samples haven't
            been generated.

        Notes
        -----
        - This method requires posterior samples to be available. Call
          get_posterior_samples() first if they haven't been generated.
        - The entropy is computed using the full posterior predictive
          distribution, accounting for uncertainty in the model parameters.
        - For a mixture with K components, the maximum possible entropy is
          log(K).
        - Entropy values can be used to identify observations that are difficult
          to classify or that may belong to multiple components.
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Mixture component entropy calculation only applies to mixture "
                "models with multiple components"
            )

        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Convert posterior samples to canonical form
        self._convert_to_canonical()

        print("Computing log-likelihoods...")

        # Compute log-likelihoods for each component
        log_liks = self.log_likelihood(
            counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            return_by=return_by,
            ignore_nans=ignore_nans,
            dtype=dtype,
            split_components=True,  # Ensure we get per-component likelihoods
        )

        # Apply temperature scaling if requested
        if temperature is not None:
            from ..core.cell_type_assignment import temperature_scaling

            log_liks = temperature_scaling(log_liks, temperature, dtype=dtype)

        print("Converting log-likelihoods to probabilities...")

        # Convert log-likelihoods to probabilities
        probs = softmax(log_liks, axis=-1)

        print("Computing entropy...")

        # Compute entropy: -∑(p_i * log(p_i))
        # Add small epsilon to avoid log(0)
        eps = jnp.finfo(dtype).eps
        entropy = -jnp.sum(probs * jnp.log(probs + eps), axis=-1)

        return entropy

    # --------------------------------------------------------------------------

    def assignment_entropy_map(
        self,
        counts: jnp.ndarray,
        return_by: str = "gene",
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        temperature: Optional[float] = None,
        use_mean: bool = True,
        verbose: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute the entropy of component assignments for each cell evaluated at
        the MAP.

        This method calculates the entropy of the posterior component assignment
        probabilities for each cell or gene, providing a measure of assignment
        uncertainty. Higher entropy values indicate more uncertainty in the
        component assignments, while lower values indicate more confident
        assignments.

        The entropy is calculated as:
            H = -∑(p_i * log(p_i))
        where p_i are the normalized probabilities for each component.

        Parameters
        ----------
        counts : jnp.ndarray
            The count matrix with shape (n_cells, n_genes).
        return_by : str, default='gene'
            Whether to return the entropy by cell or gene.
        batch_size : Optional[int], default=None
            Size of mini-batches for likelihood computation
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        temperature : Optional[float], default=None
            If provided, applies temperature scaling to the log-likelihoods
            before computing entropy.
        use_mean : bool, default=True
            If True, uses the mean of the posterior component probabilities
            instead of the MAP.
        verbose : bool, default=True
            If True, prints a warning if NaNs were replaced with means
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        jnp.ndarray
            The component entropy for each cell evaluated at the MAP. Shape:
            (n_cells,).

        Raises
        ------
        ValueError
            - If the model is not a mixture model
            - If posterior samples have not been generated yet
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Component entropy calculation only applies to mixture models "
                "with multiple components"
            )

        # Compute log-likelihood at the MAP
        log_liks = self.log_likelihood_map(
            counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            use_mean=use_mean,
            verbose=verbose,
            dtype=dtype,
            return_by=return_by,
            split_components=True,
        )

        # Apply temperature scaling if requested
        if temperature is not None:
            from ..core.cell_type_assignment import temperature_scaling

            log_liks = temperature_scaling(log_liks, temperature, dtype=dtype)

        # Compute log-sum-exp for normalization
        log_sum_exp = jsp.special.logsumexp(log_liks, axis=-1, keepdims=True)

        # Compute probabilities (avoiding log space for final entropy calculation)
        probs = jnp.exp(log_liks - log_sum_exp)

        # Compute entropy: -∑(p_i * log(p_i))
        # Add small epsilon to avoid log(0)
        eps = jnp.finfo(dtype).eps
        entropy = -jnp.sum(probs * jnp.log(probs + eps), axis=-1)

        return entropy

    # --------------------------------------------------------------------------
    # Hellinger distance for mixture models
    # --------------------------------------------------------------------------

    def hellinger_distance(
        self,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute pairwise Hellinger distances between mixture model components.

        This method calculates the Hellinger distance between each pair of
        components in the mixture model based on their inferred parameter
        distributions. The Hellinger distance is a metric that quantifies the
        similarity between two probability distributions, ranging from 0
        (identical) to 1 (completely different).

        The specific distance calculation depends on the distribution type used
        for the dispersion parameter (r):
            - For LogNormal: Uses location and scale parameters
            - For Gamma: Uses concentration and rate parameters

        Parameters
        ----------
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing pairwise Hellinger distances between
            components. Keys are of the form 'i_j' where i,j are component
            indices. Values are the Hellinger distances between components i and
            j.

        Raises
        ------
        ValueError
            If the model is not a mixture model with multiple components, or if
            the distribution type is not supported (must be LogNormal or Gamma)
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Hellinger distance calculation only applies to mixture models "
                "with multiple components"
            )

        # The 'standard' parameterization uses LogNormal for the r parameter.
        # The 'linked' parameterization uses LogNormal for the mu parameter.
        if self.model_config.parameterization == "standard":
            hellinger_distance_fn = hellinger_lognormal
            param1_key, param2_key = "r_loc_comp", "r_scale_comp"
        elif self.model_config.parameterization == "linked":
            hellinger_distance_fn = hellinger_lognormal
            param1_key, param2_key = "mu_loc_comp", "mu_scale_comp"
        else:
            raise NotImplementedError(
                f"Hellinger distance not implemented for '{self.model_config.parameterization}'."
            )

        # Extract parameters for LogNormal distribution
        r_param1 = self.params[param1_key].astype(dtype)
        r_param2 = self.params[param2_key].astype(dtype)

        # Initialize dictionary to store distances
        hellinger_distances = {}

        # Compute pairwise distances for each component
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                # Compute Hellinger distance between component i and j
                hellinger_distances[f"{i}_{j}"] = hellinger_distance_fn(
                    r_param1[i], r_param2[i], r_param1[j], r_param2[j]
                )

        return hellinger_distances

    # --------------------------------------------------------------------------
    # KL Divergence for mixture models
    # --------------------------------------------------------------------------

    def kl_divergence(
        self,
        dtype: jnp.dtype = jnp.float32,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute pairwise KL divergences between mixture model components.

        This method calculates the Kullback-Leibler (KL) divergence between each
        pair of components in the mixture model based on their inferred
        parameter distributions. The KL divergence is a measure of how one
        probability distribution diverges from a second reference distribution,
        with larger values indicating greater difference.

        Note that KL divergence is asymmetric: KL(P||Q) ≠ KL(Q||P).

        The specific divergence calculation depends on the distribution type
        used for the dispersion parameter (r):
            - For LogNormal: Uses location and scale parameters
            - For Gamma: Uses concentration and rate parameters

        Parameters
        ----------
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing pairwise KL divergences between components.
            Keys are of the form 'i_j' where i,j are component indices. Values
            are the KL divergences from component i to component j.

        Raises
        ------
        ValueError
            If the model is not a mixture model with multiple components, or if
            the distribution type is not supported (must be LogNormal or Gamma)
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "KL divergence calculation only applies to mixture models "
                "with multiple components"
            )

        # The 'standard' parameterization uses LogNormal for the r parameter.
        # The 'linked' parameterization uses LogNormal for the mu parameter.
        if self.model_config.parameterization == "standard":
            kl_divergence_fn = kl_lognormal
            param1_key, param2_key = "r_loc_comp", "r_scale_comp"
        elif self.model_config.parameterization == "linked":
            kl_divergence_fn = kl_lognormal
            param1_key, param2_key = "mu_loc_comp", "mu_scale_comp"
        else:
            raise NotImplementedError(
                f"KL divergence not implemented for '{self.model_config.parameterization}'."
            )

        # Extract parameters from r distribution based on distribution type
        r_param1 = self.params[param1_key].astype(dtype)
        r_param2 = self.params[param2_key].astype(dtype)

        # Initialize dictionary to store divergences
        kl_divergences = {}

        # Compute pairwise divergences for each component
        for i in range(self.n_components):
            for j in range(self.n_components):
                if i != j:  # Skip self-comparisons
                    # Compute KL divergence from component i to j
                    kl_divergences[f"{i}_{j}"] = kl_divergence_fn(
                        r_param1[i], r_param2[i], r_param1[j], r_param2[j]
                    )

        return kl_divergences

    # --------------------------------------------------------------------------
    # Jensen-Shannon divergence for mixture models
    # --------------------------------------------------------------------------

    def jensen_shannon_divergence(
        self,
        dtype: jnp.dtype = jnp.float32,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute pairwise Jensen-Shannon divergences between mixture model
        components.

        This method calculates the Jensen-Shannon (JS) divergence between each
        pair of components in the mixture model based on their inferred
        parameter distributions. The JS divergence is a symmetrized and smoothed
        version of the Kullback-Leibler divergence, defined as:

            JSD(P||Q) = 1/2 × KL(P||M) + 1/2 × KL(Q||M)

        where M = 1/2 × (P + Q) is the average of the two distributions.

        Unlike KL divergence, JS divergence is symmetric and bounded between 0
        and 1 (when using log base 2) or between 0 and ln(2) (when using natural
        logarithm).

        The specific divergence calculation depends on the distribution type
        used for the dispersion parameter (r):
            - For LogNormal: Uses location and scale parameters
            - For Gamma: Uses concentration and rate parameters

        Parameters
        ----------
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing pairwise JS divergences between components.
            Keys are of the form 'i_j' where i,j are component indices. Values
            are the JS divergences between components i and j.

        Raises
        ------
        ValueError
            If the model is not a mixture model with multiple components, or if
            the distribution type is not supported (must be LogNormal or Gamma)
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Jensen-Shannon divergence calculation only applies to mixture models "
                "with multiple components"
            )

        # The 'standard' parameterization uses LogNormal for the r parameter.
        # The 'linked' parameterization uses LogNormal for the mu parameter.
        if self.model_config.parameterization == "standard":
            js_divergence_fn = jensen_shannon_lognormal
            param1_key, param2_key = "r_loc_comp", "r_scale_comp"
        elif self.model_config.parameterization == "linked":
            js_divergence_fn = jensen_shannon_lognormal
            param1_key, param2_key = "mu_loc_comp", "mu_scale_comp"
        else:
            raise NotImplementedError(
                f"Jensen-Shannon divergence not implemented for '{self.model_config.parameterization}'."
            )

        # Extract parameters from r distribution based on distribution type
        r_param1 = self.params[param1_key].astype(dtype)
        r_param2 = self.params[param2_key].astype(dtype)

        # Initialize dictionary to store divergences
        js_divergences = {}

        # Compute pairwise divergences for each component
        for i in range(self.n_components):
            for j in range(
                i + 1, self.n_components
            ):  # Only compute for i < j since JS is symmetric
                # Compute JS divergence between components i and j
                js_divergences[f"{i}_{j}"] = js_divergence_fn(
                    r_param1[i], r_param2[i], r_param1[j], r_param2[j]
                )

        return js_divergences

    # --------------------------------------------------------------------------
    # Cell type assignment method for mixture models
    # --------------------------------------------------------------------------

    def cell_type_probabilities(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        ignore_nans: bool = False,
        dtype: jnp.dtype = jnp.float32,
        fit_distribution: bool = True,
        temperature: Optional[float] = None,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute probabilistic cell type assignments and fit Dirichlet
        distributions to characterize assignment uncertainty.

        For each cell, this method:
            1. Computes component-specific log-likelihoods using posterior
               samples
            2. Converts these to probability distributions over cell types
            3. Fits a Dirichlet distribution to characterize the uncertainty in
               these assignments

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate assignments for
        batch_size : Optional[int], default=None
            Size of mini-batches for likelihood computation
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        ignore_nans : bool, default=False
            If True, removes any samples that contain NaNs.
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations
        fit_distribution : bool, default=True
            If True, fits a Dirichlet distribution to the assignment
            probabilities
        temperature : Optional[float], default=None
            If provided, apply temperature scaling to log probabilities
        weights : Optional[jnp.ndarray], default=None
            Array used to weight genes when computing log likelihoods
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        verbose : bool, default=True
            If True, prints progress messages

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing:
                - 'concentration': Dirichlet concentration parameters for each
                  cell. Shape: (n_cells, n_components). Only returned if
                  fit_distribution is True.
                - 'mean_probabilities': Mean assignment probabilities for each
                  cell. Shape: (n_cells, n_components). Only returned if
                  fit_distribution is True.
                - 'sample_probabilities': Assignment probabilities for each
                  posterior sample. Shape: (n_samples, n_cells, n_components)

        Raises
        ------
        ValueError
            - If the model is not a mixture model
            - If posterior samples have not been generated yet

        Note
        ----
        Most of the log-likelihood value differences between cell types are
        extremely large. Thus, the computation usually returns either 0 or 1.
        This computation is therefore not very useful, but it is included for
        completeness.
        """
        from ..core.cell_type_assignment import compute_cell_type_probabilities

        return compute_cell_type_probabilities(
            results=self,
            counts=counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            dtype=dtype,
            fit_distribution=fit_distribution,
            temperature=temperature,
            weights=weights,
            weight_type=weight_type,
            verbose=verbose,
        )

    # --------------------------------------------------------------------------

    def cell_type_probabilities_map(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        dtype: jnp.dtype = jnp.float32,
        temperature: Optional[float] = None,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        use_mean: bool = False,
        verbose: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute probabilistic cell type assignments using MAP estimates of
        parameters.

        For each cell, this method:
            1. Computes component-specific log-likelihoods using MAP parameter
            estimates
            2. Converts these to probability distributions over cell types

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate assignments for
        batch_size : Optional[int], default=None
            Size of mini-batches for likelihood computation
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations
        temperature : Optional[float], default=None
            If provided, apply temperature scaling to log probabilities
        weights : Optional[jnp.ndarray], default=None
            Array used to weight genes when computing log likelihoods
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        use_mean : bool, default=False
            If True, replaces undefined MAP values (NaN) with posterior means
        verbose : bool, default=True
            If True, prints progress messages

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing:
                - 'probabilities': Assignment probabilities for each cell.
                Shape: (n_cells, n_components)

        Raises
        ------
        ValueError
            If the model is not a mixture model
        """
        from ..core.cell_type_assignment import (
            compute_cell_type_probabilities_map,
        )

        return compute_cell_type_probabilities_map(
            results=self,
            counts=counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            dtype=dtype,
            temperature=temperature,
            weights=weights,
            weight_type=weight_type,
            use_mean=use_mean,
            verbose=verbose,
        )

    # --------------------------------------------------------------------------
    # Count normalization methods
    # --------------------------------------------------------------------------

    def normalize_counts(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples_dirichlet: int = 1,
        fit_distribution: bool = False,
        store_samples: bool = True,
        sample_axis: int = 0,
        return_concentrations: bool = False,
        backend: str = "numpyro",
        verbose: bool = True,
    ) -> Dict[str, Union[jnp.ndarray, object]]:
        """
        Normalize counts using posterior samples of the r parameter.

        This method takes posterior samples of the dispersion parameter (r) and
        uses them as concentration parameters for Dirichlet distributions to generate
        normalized expression profiles. For mixture models, normalization is performed
        per component, resulting in an extra dimension in the output.

        Based on the insights from the Dirichlet-multinomial model derivation, the
        r parameters represent the concentration parameters of a Dirichlet distribution
        that can be used to generate normalized expression profiles.

        The method generates Dirichlet samples using all posterior samples of r, then
        fits a single Dirichlet distribution to all these samples (or one per component
        for mixture models).

        Parameters
        ----------
        rng_key : random.PRNGKey, default=random.PRNGKey(42)
            JAX random number generator key
        n_samples_dirichlet : int, default=1000
            Number of samples to draw from each Dirichlet distribution
        fit_distribution : bool, default=True
            If True, fits a Dirichlet distribution to the generated samples using
            fit_dirichlet_minka from stats.py
        store_samples : bool, default=False
            If True, includes the raw Dirichlet samples in the output
        sample_axis : int, default=0
            Axis containing samples in the Dirichlet fitting (passed to fit_dirichlet_minka)
        return_concentrations : bool, default=False
            If True, returns the original r parameter samples used as concentrations
        backend : str, default="numpyro"
            Statistical package to use for distributions when fit_distribution=True.
            Must be one of:
            - "numpyro": Returns numpyro.distributions.Dirichlet objects
            - "scipy": Returns scipy.stats distributions via numpyro_to_scipy conversion
        verbose : bool, default=True
            If True, prints progress messages

        Returns
        -------
        Dict[str, Union[jnp.ndarray, object]]
            Dictionary containing normalized expression profiles. Keys depend on
            input arguments:
            - 'samples': Raw Dirichlet samples (if store_samples=True)
            - 'concentrations': Fitted concentration parameters (if fit_distribution=True)
            - 'mean_probabilities': Mean probabilities from fitted distribution (if fit_distribution=True)
            - 'distributions': Dirichlet distribution objects (if fit_distribution=True)
            - 'original_concentrations': Original r parameter samples (if return_concentrations=True)

            For non-mixture models:
            - samples: shape (n_posterior_samples, n_genes, n_samples_dirichlet) or
                      (n_posterior_samples, n_genes) if n_samples_dirichlet=1
            - concentrations: shape (n_genes,) - single fitted distribution
            - mean_probabilities: shape (n_genes,) - single fitted distribution
            - distributions: single Dirichlet distribution object

            For mixture models:
            - samples: shape (n_posterior_samples, n_components, n_genes, n_samples_dirichlet) or
                      (n_posterior_samples, n_components, n_genes) if n_samples_dirichlet=1
            - concentrations: shape (n_components, n_genes) - one fitted distribution per component
            - mean_probabilities: shape (n_components, n_genes) - one fitted distribution per component
            - distributions: list of n_components Dirichlet distribution objects

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet, or if 'r' parameter
            is not found in posterior samples

        Examples
        --------
        >>> # For a non-mixture model
        >>> normalized = results.normalize_counts(
        ...     n_samples_dirichlet=100,
        ...     fit_distribution=True
        ... )
        >>> print(normalized['mean_probabilities'].shape)  # (n_genes,)
        >>> print(type(normalized['distributions']))  # Single Dirichlet distribution

        >>> # For a mixture model
        >>> normalized = results.normalize_counts(
        ...     n_samples_dirichlet=100,
        ...     fit_distribution=True
        ... )
        >>> print(normalized['mean_probabilities'].shape)  # (n_components, n_genes)
        >>> print(len(normalized['distributions']))  # n_components
        """
        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Convert to canonical form to ensure r parameter is available
        self._convert_to_canonical()

        # Use the shared normalization function
        return normalize_counts_from_posterior(
            posterior_samples=self.posterior_samples,
            n_components=self.n_components,
            rng_key=rng_key,
            n_samples_dirichlet=n_samples_dirichlet,
            fit_distribution=fit_distribution,
            store_samples=store_samples,
            sample_axis=sample_axis,
            return_concentrations=return_concentrations,
            backend=backend,
            verbose=verbose,
        )

    # --------------------------------------------------------------------------
    # Parameter conversion method
    # --------------------------------------------------------------------------

    def _convert_to_canonical(self):
        """
        Convert posterior samples to canonical (p, r) form.

        Returns
        -------
        self : ScribeSVIResults
            Returns self for method chaining
        """
        # If no posterior samples, return self
        if self.posterior_samples is None:
            return self

        # Get samples
        samples = self.posterior_samples

        # Convert parameters to canonical form based on what's available
        if "phi" in samples and "mu" in samples:
            # Extract phi and mu
            phi = samples["phi"]
            mu = samples["mu"]
            # Compute p from phi
            samples["p"] = 1.0 / (1.0 + phi)

            # Reshape phi to broadcast with mu based on mixture model and
            # component specificity
            if self.n_components is not None:
                # Mixture model: mu has shape (n_samples, n_components, n_genes)
                if self.model_config.component_specific_params:
                    # Component-specific phi: shape (n_samples, n_components)
                    phi_reshaped = phi[:, :, None]
                else:
                    # Shared phi: shape (n_samples,) ->
                    # broadcast to (n_samples, 1, 1)
                    phi_reshaped = phi[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                phi_reshaped = phi[:, None]
            # Compute r from mu and phi
            samples["r"] = mu * phi_reshaped

        # Handle linked parameterization (mu, p -> r)
        elif "mu" in samples and "p" in samples and "r" not in samples:
            # Extract p and mu
            p = samples["p"]
            mu = samples["mu"]
            # Reshape p to broadcast with mu based on mixture model and component specificity
            if self.n_components is not None:
                if self.model_config.component_specific_params:
                    # Component-specific p: shape (n_samples, n_components)
                    p_reshaped = p[:, :, None]
                else:
                    # Shared p: shape (n_samples,) ->
                    # broadcast to (n_samples, 1, 1)
                    p_reshaped = p[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                p_reshaped = p[:, None]
            # Compute r from mu and p
            samples["r"] = mu * (1.0 - p_reshaped) / p_reshaped

        # Handle unconstrained parameterization (r_unconstrained,
        # p_unconstrained, etc.)
        if "r_unconstrained" in samples and "r" not in samples:
            # compute r from r_unconstrained
            samples["r"] = jnp.exp(samples["r_unconstrained"])

        if "p_unconstrained" in samples and "p" not in samples:
            # Compute p from p_unconstrained
            samples["p"] = sigmoid(samples["p_unconstrained"])

        if "gate_unconstrained" in samples and "gate" not in samples:
            # Compute gate from gate_unconstrained
            samples["gate"] = sigmoid(samples["gate_unconstrained"])

        # Handle VCP capture probability conversions
        if "phi_capture" in samples and "p_capture" not in samples:
            # Extract phi_capture
            phi_capture = samples["phi_capture"]
            # Compute p_capture from phi_capture
            samples["p_capture"] = 1.0 / (1.0 + phi_capture)

        if "p_capture_unconstrained" in samples and "p_capture" not in samples:
            # Compute p_capture from p_capture_unconstrained
            samples["p_capture"] = sigmoid(samples["p_capture_unconstrained"])

        # Handle mixing weights computation for mixture models
        if (
            "mixing_logits_unconstrained" in samples
            and "mixing_weights" not in samples
        ):
            # Compute mixing weights from mixing_logits_unconstrained using
            # softmax
            samples["mixing_weights"] = softmax(
                samples["mixing_logits_unconstrained"], axis=-1
            )

        return self
