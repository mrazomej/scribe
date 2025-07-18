"""
Results classes for SCRIBE MCMC inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass, field, replace
import warnings

import jax.numpy as jnp
from jax.nn import sigmoid, softmax
from jax import random, jit, vmap
import pandas as pd
from numpyro.infer import MCMC
import numpy as np

from ..sampling import generate_predictive_samples
from ..models.model_config import ModelConfig
from ..core.normalization import normalize_counts_from_posterior

# ------------------------------------------------------------------------------
# MCMC Subset class (moved to module level)
# ------------------------------------------------------------------------------


@dataclass
class ScribeMCMCSubset:
    """Lightweight container for subset of MCMC results."""

    samples: Dict
    n_cells: int
    n_genes: int
    model_type: str
    model_config: ModelConfig
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None
    n_components: Optional[int] = None
    predictive_samples: Optional[jnp.ndarray] = None

    # Add posterior_samples property for compatibility
    @property
    def posterior_samples(self):
        """Get posterior samples in canonical form."""
        return self.get_posterior_samples(canonical=True)

    # --------------------------------------------------------------------------

    def __getitem__(self, index):
        """Support further indexing of subset by genes."""
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
        # Handle list/array indexing
        elif not isinstance(index, (bool, jnp.bool_)) and not isinstance(
            index[-1], (bool, jnp.bool_)
        ):
            # Get indices from list/array
            indices = jnp.array(index)
            # Initialize boolean index
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        else:
            # Already a boolean index
            bool_index = index

        # Create new metadata if available
        new_var = self.var.iloc[bool_index] if self.var is not None else None

        # Get subset of samples
        new_samples = self._subset_posterior_samples(self.samples, bool_index)

        # Create new instance with subset data
        return ScribeMCMCSubset(
            samples=new_samples,
            n_cells=self.n_cells,
            n_genes=int(
                bool_index.sum()
                if hasattr(bool_index, "sum")
                else len(bool_index)
            ),
            model_type=self.model_type,
            model_config=self.model_config,
            obs=self.obs,
            var=new_var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=new_var.shape[0] if new_var is not None else None,
            n_components=self.n_components,
        )

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

    def get_posterior_samples(self, canonical=True):
        """Get posterior samples in canonical form."""
        if canonical:
            # Convert to canonical form on-the-fly
            return self._convert_to_canonical(self.samples)
        else:
            return self.samples

    # --------------------------------------------------------------------------

    def _convert_to_canonical(self, samples):
        """
        Convert samples to canonical (p, r) form based on present
        parameters.

        Parameters
        ----------
        samples : Dict
            Raw samples dictionary

        Returns
        -------
        Dict
            Canonical samples dictionary
        """
        # If no samples, return empty dict
        if not samples:
            return {}

        # Create a copy of samples to avoid modifying the original
        canonical_samples = samples.copy()

        # Convert based on what parameters are present in the samples

        # Handle odds_ratio parameterization (phi, mu -> p, r)
        if "phi" in canonical_samples and "mu" in canonical_samples:
            # Extract phi and mu
            phi = canonical_samples["phi"]
            mu = canonical_samples["mu"]
            # Compute p from phi
            canonical_samples["p"] = 1.0 / (1.0 + phi)

            # Reshape phi to broadcast with mu based on mixture model
            if self.n_components is not None:
                # Mixture model: mu has shape (n_samples, n_components, n_genes)
                phi_reshaped = phi[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                phi_reshaped = phi[:, None]
            # Compute r from mu and phi
            canonical_samples["r"] = mu * phi_reshaped

        # Handle linked parameterization (mu, p -> r)
        elif (
            "mu" in canonical_samples
            and "p" in canonical_samples
            and "r" not in canonical_samples
        ):
            # Extract p and mu
            p = canonical_samples["p"]
            mu = canonical_samples["mu"]
            # Reshape p to broadcast with mu based on mixture model
            if self.n_components is not None:
                # Mixture model: mu has shape (n_samples, n_components, n_genes)
                p_reshaped = p[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                p_reshaped = p[:, None]
            # Compute r from mu and p
            canonical_samples["r"] = mu * (1.0 - p_reshaped) / p_reshaped

        # Handle unconstrained parameterization (r_unconstrained, p_unconstrained, etc.)
        if (
            "r_unconstrained" in canonical_samples
            and "r" not in canonical_samples
        ):
            # compute r from r_unconstrained
            canonical_samples["r"] = jnp.exp(
                canonical_samples["r_unconstrained"]
            )

        if (
            "p_unconstrained" in canonical_samples
            and "p" not in canonical_samples
        ):
            # Compute p from p_unconstrained
            canonical_samples["p"] = sigmoid(
                canonical_samples["p_unconstrained"]
            )

        if (
            "gate_unconstrained" in canonical_samples
            and "gate" not in canonical_samples
        ):
            # Compute gate from gate_unconstrained
            canonical_samples["gate"] = sigmoid(
                canonical_samples["gate_unconstrained"]
            )

        # Handle VCP capture probability conversions
        if (
            "phi_capture" in canonical_samples
            and "p_capture" not in canonical_samples
        ):
            # Extract phi_capture
            phi_capture = canonical_samples["phi_capture"]
            # Compute p_capture from phi_capture
            canonical_samples["p_capture"] = 1.0 / (1.0 + phi_capture)

        if (
            "p_capture_unconstrained" in canonical_samples
            and "p_capture" not in canonical_samples
        ):
            # Compute p_capture from p_capture_unconstrained
            canonical_samples["p_capture"] = sigmoid(
                canonical_samples["p_capture_unconstrained"]
            )

        # Handle mixing weights computation for mixture models
        if (
            "mixing_logits_unconstrained" in canonical_samples
            and "mixing_weights" not in canonical_samples
        ):
            # Compute mixing weights from mixing_logits_unconstrained using
            # softmax
            canonical_samples["mixing_weights"] = softmax(
                canonical_samples["mixing_logits_unconstrained"],
                axis=-1,
            )

        return canonical_samples

    # --------------------------------------------------------------------------

    def get_posterior_quantiles(self, param, quantiles=(0.025, 0.5, 0.975)):
        """Get quantiles for a specific parameter from samples."""
        return _get_posterior_quantiles(
            self.get_posterior_samples(), param, quantiles
        )

    # --------------------------------------------------------------------------

    def get_map(self):
        """Get MAP estimates for parameters."""
        return _get_map_estimate(self.get_posterior_samples())

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
        """Compute log likelihood of data under posterior samples."""
        # Get canonical samples
        samples = self.get_posterior_samples(canonical=True)

        return _compute_log_likelihood(
            samples,
            counts,
            self.model_type,
            n_components=self.n_components,
            batch_size=batch_size,
            return_by=return_by,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            split_components=split_components,
            weights=weights,
            weight_type=weight_type,
            dtype=dtype,
        )

    # --------------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples."""
        predictive_samples = _generate_ppc_samples(
            self.get_posterior_samples(canonical=True),
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

    # --------------------------------------------------------------------------

    def get_prior_predictive_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate prior predictive samples using the model."""
        # Generate prior predictive samples
        prior_predictive_samples = _generate_prior_predictive_samples(
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        # Store samples if requested
        if store_samples:
            self.prior_predictive_samples = prior_predictive_samples

        return prior_predictive_samples

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
        """Normalize counts using posterior samples of the r parameter."""
        # Get canonical samples
        posterior_samples = self.get_posterior_samples(canonical=True)

        # Use the shared normalization function
        return normalize_counts_from_posterior(
            posterior_samples=posterior_samples,
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
        """Compute probabilistic cell type assignments."""
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


# ------------------------------------------------------------------------------
# MCMC results class
# ------------------------------------------------------------------------------


class ScribeMCMCResults(MCMC):
    """
    SCRIBE MCMC results class that extends numpyro.infer.MCMC.

    This class inherits all functionality from MCMC while adding SCRIBE-specific
    attributes and methods for analyzing single-cell RNA sequencing data.

    Attributes
    ----------
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
    predictive_samples : Optional[Dict]
        Predictive samples generated from the model, if generated
    n_components : Optional[int]
        Number of mixture components, if using a mixture model
    """

    def __init__(
        self,
        mcmc,  # MCMC instance
        n_cells: int,
        n_genes: int,
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[Dict] = None,
        n_obs: Optional[int] = None,
        n_vars: Optional[int] = None,
        predictive_samples: Optional[Dict] = None,
        n_components: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize ScribeMCMCResults from an existing MCMC instance.

        Parameters
        ----------
        mcmc : MCMC
            The MCMC instance containing samples and diagnostics
        n_cells : int
            Number of cells in the dataset
        n_genes : int
            Number of genes in the dataset
        model_type : str
            Type of model used for inference
        model_config : ModelConfig
            Configuration object for model
        prior_params : Dict[str, Any]
            Dictionary of prior parameter values
        obs : Optional[pd.DataFrame]
            Cell metadata
        var : Optional[pd.DataFrame]
            Gene metadata
        uns : Optional[Dict]
            Unstructured metadata
        n_obs : Optional[int]
            Number of observations (cells)
        n_vars : Optional[int]
            Number of variables (genes)
        predictive_samples : Optional[Dict]
            Predictive samples from the model
        n_components : Optional[int]
            Number of mixture components
        """
        # Initialize the MCMC parent class attributes
        # We don't call __init__ because we want to take attributes from the
        # existing mcmc instance Instead, we copy the relevant attributes from
        # the mcmc instance
        self.__dict__.update(mcmc.__dict__)

        # Set SCRIBE-specific attributes
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.model_type = model_type
        self.model_config = model_config
        self.prior_params = prior_params

        # AnnData-related attributes
        self.obs = obs
        self.var = var
        self.uns = uns
        self.n_obs = n_obs
        self.n_vars = n_vars

        # Optional attributes
        self.predictive_samples = predictive_samples
        self.prior_predictive_samples = None
        self.n_components = (
            n_components
            if n_components is not None
            else model_config.n_components
        )

        # Validate configuration
        self._validate_model_config()

    @classmethod
    def from_mcmc(
        cls,
        mcmc: MCMC,
        n_cells: int,
        n_genes: int,
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        **kwargs,
    ):
        """
        Create ScribeMCMCResults from an existing MCMC instance.

        Parameters
        ----------
        mcmc : MCMC
            The MCMC instance to extend
        n_cells : int
            Number of cells in the dataset
        n_genes : int
            Number of genes in the dataset
        model_type : str
            Type of model used for inference
        model_config : ModelConfig
            Configuration object for the model
        prior_params : Dict[str, Any]
            Dictionary of prior parameter values
        **kwargs
            Additional arguments to pass to ScribeMCMCResults constructor

        Returns
        -------
        ScribeMCMCResults
            Extended MCMC instance with SCRIBE functionality
        """
        # Create ScribeMCMCResults instance
        return cls(
            mcmc=mcmc,
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params,
            **kwargs,
        )

    # --------------------------------------------------------------------------
    # From AnnData
    # --------------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        mcmc: MCMC,
        adata: "AnnData",
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        **kwargs,
    ):
        """
        Create ScribeMCMCResults from MCMC instance and AnnData object.

        Parameters
        ----------
        mcmc : MCMC
            The MCMC instance to extend
        adata : AnnData
            AnnData object containing the data
        model_type : str
            Type of model used for inference
        model_config : ModelConfig
            Configuration object for the model
        prior_params : Dict[str, Any]
            Dictionary of prior parameter values
        **kwargs
            Additional arguments to pass to ScribeMCMCResults constructor

        Returns
        -------
        ScribeMCMCResults
            Extended MCMC instance with SCRIBE functionality
        """
        # Create ScribeMCMCResults instance
        return cls(
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

    # --------------------------------------------------------------------------
    # Validation
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

    # --------------------------------------------------------------------------
    # Override get_samples to provide canonical samples by default
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Parameter conversion method
    # --------------------------------------------------------------------------

    def _convert_to_canonical(self, samples):
        """
        Convert samples to canonical (p, r) form based on present parameters.

        Parameters
        ----------
        samples : Dict
            Raw samples dictionary

        Returns
        -------
        Dict
            Canonical samples dictionary
        """
        # If no samples, return empty dict
        if not samples:
            return {}

        # Create a copy of samples to avoid modifying the original
        canonical_samples = samples.copy()

        # Convert based on what parameters are present in the samples

        # Handle odds_ratio parameterization (phi, mu -> p, r)
        if "phi" in canonical_samples and "mu" in canonical_samples:
            # Extract phi and mu
            phi = canonical_samples["phi"]
            mu = canonical_samples["mu"]
            # Compute p from phi
            canonical_samples["p"] = 1.0 / (1.0 + phi)

            # Reshape phi to broadcast with mu based on mixture model
            if self.n_components is not None:
                # Mixture model: mu has shape (n_samples, n_components, n_genes)
                phi_reshaped = phi[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                phi_reshaped = phi[:, None]
            # Compute r from mu and phi
            canonical_samples["r"] = mu * phi_reshaped

        # Handle linked parameterization (mu, p -> r)
        elif (
            "mu" in canonical_samples
            and "p" in canonical_samples
            and "r" not in canonical_samples
        ):
            # Extract p and mu
            p = canonical_samples["p"]
            mu = canonical_samples["mu"]
            # Reshape p to broadcast with mu based on mixture model
            if self.n_components is not None:
                # Mixture model: mu has shape (n_samples, n_components, n_genes)
                p_reshaped = p[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                p_reshaped = p[:, None]
            # Compute r from mu and p
            canonical_samples["r"] = mu * (1.0 - p_reshaped) / p_reshaped

        # Handle unconstrained parameterization (r_unconstrained,
        # p_unconstrained, etc.)
        if (
            "r_unconstrained" in canonical_samples
            and "r" not in canonical_samples
        ):
            # compute r from r_unconstrained
            canonical_samples["r"] = jnp.exp(
                canonical_samples["r_unconstrained"]
            )

        if (
            "p_unconstrained" in canonical_samples
            and "p" not in canonical_samples
        ):
            # Compute p from p_unconstrained
            canonical_samples["p"] = sigmoid(
                canonical_samples["p_unconstrained"]
            )

        if (
            "gate_unconstrained" in canonical_samples
            and "gate" not in canonical_samples
        ):
            # Compute gate from gate_unconstrained
            canonical_samples["gate"] = sigmoid(
                canonical_samples["gate_unconstrained"]
            )

        # Handle VCP capture probability conversions
        if (
            "phi_capture" in canonical_samples
            and "p_capture" not in canonical_samples
        ):
            # Extract phi_capture
            phi_capture = canonical_samples["phi_capture"]
            # Compute p_capture from phi_capture
            canonical_samples["p_capture"] = 1.0 / (1.0 + phi_capture)

        if (
            "p_capture_unconstrained" in canonical_samples
            and "p_capture" not in canonical_samples
        ):
            # Compute p_capture from p_capture_unconstrained
            canonical_samples["p_capture"] = sigmoid(
                canonical_samples["p_capture_unconstrained"]
            )

        # Handle mixing weights computation for mixture models
        if (
            "mixing_logits_unconstrained" in canonical_samples
            and "mixing_weights" not in canonical_samples
        ):
            # Compute mixing weights from mixing_logits_unconstrained using
            # softmax
            canonical_samples["mixing_weights"] = softmax(
                canonical_samples["mixing_logits_unconstrained"], axis=-1
            )

        return canonical_samples

    def get_samples(self, group_by_chain=False, canonical=False):
        """
        Get samples from the MCMC run, with option to return canonical form.

        Parameters
        ----------
        group_by_chain : bool, default=False
            Whether to preserve the chain dimension. If True, all samples will
            have num_chains as the size of their leading dimension.
        canonical : bool, default=True
            Whether to return samples in canonical (p, r) form. If False, returns
            raw samples as they were collected during MCMC.

        Returns
        -------
        Dict
            Dictionary of parameter samples. If canonical=True, parameters are
            converted to canonical form. If canonical=False, returns raw samples.
        """
        # Get raw samples from parent class
        raw_samples = super().get_samples(group_by_chain=group_by_chain)

        if not canonical:
            return raw_samples

        # Convert to canonical form on-the-fly
        return self._convert_to_canonical(raw_samples)

    # --------------------------------------------------------------------------
    # SCRIBE-specific methods
    # --------------------------------------------------------------------------

    def get_posterior_samples(self, canonical=True):
        """
        Get posterior samples from the MCMC run.

        This is a convenience method to match the ScribeResults interface.
        Returns canonical samples by default.

        Returns
        -------
        Dict
            Dictionary of parameter samples in canonical form
        """
        return self.get_samples(canonical=canonical)

    # --------------------------------------------------------------------------

    def get_posterior_quantiles(
        self, param, quantiles=(0.025, 0.5, 0.975), canonical=True
    ):
        """
        Get quantiles for a specific parameter from MCMC samples.

        Parameters
        ----------
        param : str
            Parameter name
        quantiles : tuple, default=(0.025, 0.5, 0.975)
            Quantiles to compute

        Returns
        -------
        dict
            Dictionary mapping quantiles to values
        """
        return _get_posterior_quantiles(
            self.get_samples(canonical=canonical), param, quantiles
        )

    # --------------------------------------------------------------------------

    def get_map(self, canonical=True):
        """
        Get the maximum a posteriori (MAP) estimate from MCMC samples.

        For each parameter, this returns the value with the highest
        posterior density.

        Returns
        -------
        dict
            Dictionary of MAP estimates for each parameter
        """
        samples = self.get_samples(canonical=canonical)

        # Get extra fields to compute joint log density
        try:
            potential_energy = self.get_extra_fields()["potential_energy"]
            return _get_map_estimate(samples, potential_energy)
        except:
            # Fallback: Use general function without potential energy
            return _get_map_estimate(samples)

    # --------------------------------------------------------------------------
    # Get model function
    # --------------------------------------------------------------------------

    def _model(self) -> Callable:
        """Get the model function for this model type."""
        model = _get_model_fn(self.model_type, self.model_config)
        return model

    # --------------------------------------------------------------------------
    # Get log likelihood function
    # --------------------------------------------------------------------------

    def _log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for this model type."""
        return _get_log_likelihood_fn(self.model_type)

    # --------------------------------------------------------------------------
    # Posterior sampling methods
    # --------------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """
        Generate posterior predictive check (PPC) samples using posterior
        parameter samples.

        This method uses the posterior parameter samples to generate new data
        from the model. These samples can be used to assess model fit and
        perform posterior predictive checks.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key (default: PRNGKey(42))
        batch_size : int, optional
            Batch size for generating samples. If None, uses full dataset.
        store_samples : bool, optional
            Whether to store the generated samples in self.predictive_samples
            (default: True)

        Returns
        -------
        jnp.ndarray
            Array of posterior predictive samples
        """
        # Generate predictive samples
        predictive_samples = _generate_ppc_samples(
            self.get_samples(canonical=True),
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        # Store samples if requested
        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

    # --------------------------------------------------------------------------
    # Count normalization methods
    # --------------------------------------------------------------------------

    def normalize_counts(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples_dirichlet: int = 1000,
        fit_distribution: bool = True,
        store_samples: bool = False,
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
        # Get canonical samples using the new method
        posterior_samples = self.get_samples(canonical=True)

        # Use the shared normalization function
        return normalize_counts_from_posterior(
            posterior_samples=posterior_samples,
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
    # Prior predictive sampling methods
    # --------------------------------------------------------------------------

    def get_prior_predictive_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """
        Generate prior predictive samples using the model.

        This method generates samples from the prior predictive distribution by
        first sampling parameters from the prior distributions and then
        generating data from the model using these parameters. These samples can
        be used to assess model behavior before seeing any data.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key (default: PRNGKey(42))
        n_samples : int, optional
            Number of prior predictive samples to generate (default: 100)
        batch_size : int, optional
            Batch size for generating samples. If None, uses full dataset.
        store_samples : bool, optional
            Whether to store the generated samples in
            self.prior_predictive_samples (default: True)

        Returns
        -------
        jnp.ndarray
            Array of prior predictive samples
        """
        # Generate prior predictive samples
        prior_predictive_samples = _generate_prior_predictive_samples(
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        # Store samples if requested
        if store_samples:
            self.prior_predictive_samples = prior_predictive_samples

        return prior_predictive_samples

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
        # Get canonical samples using the new method
        samples = self.get_samples(canonical=True)

        return _compute_log_likelihood(
            samples,
            counts,
            self.model_type,
            n_components=self.n_components,
            batch_size=batch_size,
            return_by=return_by,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            split_components=split_components,
            weights=weights,
            weight_type=weight_type,
            dtype=dtype,
        )

    # --------------------------------------------------------------------------
    # Indexing by genes
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

    def __getitem__(self, index):
        """
        Enable indexing of ScribeMCMCResults object by genes.

        This allows selecting a subset of genes for analysis, e.g.:
        results[0:10]  # Get first 10 genes
        results[gene_indices]  # Get genes by index

        Parameters
        ----------
        index : int, slice, or array-like
            Indices of genes to select

        Returns
        -------
        ScribeMCMCSubset
            A new ScribeMCMCResults object containing only the selected genes
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
        # Handle list/array indexing
        elif not isinstance(index, (bool, jnp.bool_)) and not isinstance(
            index[-1], (bool, jnp.bool_)
        ):
            # Get indices from list/array
            indices = jnp.array(index)
            # Initialize boolean index
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        else:
            # Already a boolean index
            bool_index = index

        # Create new metadata if available
        new_var = self.var.iloc[bool_index] if self.var is not None else None

        # Get subset of samples
        samples = self.get_samples(canonical=False)
        new_samples = self._subset_posterior_samples(samples, bool_index)

        # Create new instance with subset data
        # We can't use inheritance for the subset since we need to detach from
        # the mcmc instance Instead, create a lightweight version that stores
        # the subset data
        from dataclasses import dataclass

        # Create and return the subset using the module-level class
        return ScribeMCMCSubset(
            samples=new_samples,
            n_cells=self.n_cells,
            n_genes=int(
                bool_index.sum()
                if hasattr(bool_index, "sum")
                else len(bool_index)
            ),
            model_type=self.model_type,
            model_config=self.model_config,
            obs=self.obs,
            var=new_var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=new_var.shape[0] if new_var is not None else None,
            n_components=self.n_components,
        )

    # --------------------------------------------------------------------------
    # Indexing by component
    # --------------------------------------------------------------------------

    def get_component(self, component_index):
        """
        Get a specific component from mixture model results.

        This method returns a ScribeMCMCSubset object that contains parameter
        samples for the specified component, allowing for further gene-based
        indexing. Only applicable to mixture models.

        Parameters
        ----------
        component_index : int
            Index of the component to select

        Returns
        -------
        ScribeMCMCSubset
            A ScribeMCMCSubset object with samples for the selected component

        Raises
        ------
        ValueError
            If the model is not a mixture model or if component_index is out of range
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Component selection only applies to mixture models with multiple components"
            )

        # Check if component_index is valid
        if component_index < 0 or component_index >= self.n_components:
            raise ValueError(
                f"Component index {component_index} out of range [0, {self.n_components-1}]"
            )

        # Get samples and subset by component
        samples = self.get_samples(canonical=False)
        component_samples = self._subset_samples_by_component(
            samples, component_index
        )

        # Create modified model config (remove mixture aspects)
        modified_model_config = replace(
            self.model_config,
            base_model=self.model_type.replace("_mix", ""),
            n_components=None,
        )

        # Return ScribeMCMCSubset with component-specific data
        return ScribeMCMCSubset(
            samples=component_samples,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=self.model_type.replace(
                "_mix", ""
            ),  # Remove _mix suffix
            model_config=modified_model_config,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            n_components=None,  # No longer a mixture model
        )

    # --------------------------------------------------------------------------

    def _subset_samples_by_component(
        self, samples: Dict, component_index: int
    ) -> Dict:
        """
        Subset samples for a specific component.

        Parameters
        ----------
        samples : Dict
            Dictionary of parameter samples
        component_index : int
            Index of the component to select

        Returns
        -------
        Dict
            Dictionary of parameter samples for the selected component
        """
        new_samples = {}

        # Define parameter categories based on their typical structure
        component_gene_specific = [
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

        component_specific = [
            "p",
            "p_unconstrained",
            "mixing_weights",
            "mixing_logits_unconstrained",
            "p_unconstrained_loc",
            "p_unconstrained_scale",
            "mixing_logits_unconstrained_loc",
            "mixing_logits_unconstrained_scale",
        ]

        cell_specific = [
            "p_capture",
            "p_capture_unconstrained",
            "phi_capture",
            "p_capture_unconstrained_loc",
            "p_capture_unconstrained_scale",
        ]

        for param_name, values in samples.items():
            if param_name in component_gene_specific:
                # Component-gene specific parameters: (n_samples, n_components, n_genes)
                if values.ndim == 3:
                    new_samples[param_name] = values[:, component_index, :]
                else:  # Already in correct shape or scalar
                    new_samples[param_name] = values
            elif param_name in component_specific:
                # Component-specific parameters: (n_samples, n_components)
                if values.ndim == 2:
                    new_samples[param_name] = values[:, component_index]
                else:  # Scalar parameter
                    new_samples[param_name] = values
            else:
                # Cell-specific or other parameters - copy as-is
                new_samples[param_name] = values

        return new_samples

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


# ------------------------------------------------------------------------------
# Shared helper functions for both ScribeMCMCResults and ScribeMCMCSubset
# ------------------------------------------------------------------------------


def _get_model_fn(model_type: str, model_config) -> Callable:
    """Get the model function for this model type and parameterization."""
    from ..models.model_registry import get_model_and_guide

    parameterization = model_config.parameterization or "standard"
    return get_model_and_guide(model_type, parameterization)[0]


# ------------------------------------------------------------------------------


def _get_log_likelihood_fn(model_type: str) -> Callable:
    """Get the log likelihood function for this model type."""
    from scribe.models.model_registry import get_log_likelihood_fn

    return get_log_likelihood_fn(model_type)


# ------------------------------------------------------------------------------


def _compute_log_likelihood(
    samples: Dict,
    counts: jnp.ndarray,
    model_type: str,
    n_components: Optional[int] = None,
    batch_size: Optional[int] = None,
    return_by: str = "cell",
    cells_axis: int = 0,
    ignore_nans: bool = False,
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Compute log likelihood of data under posterior samples."""
    # Get number of samples from first parameter
    n_samples = samples[next(iter(samples))].shape[0]

    # Get likelihood function
    likelihood_fn = _get_log_likelihood_fn(model_type)

    # Determine if this is a mixture model
    is_mixture = n_components is not None and n_components > 1

    # Define function to compute likelihood for a single sample
    @jit
    def compute_sample_lik(i):
        # Extract parameters for this sample
        params_i = {k: v[i] for k, v in samples.items()}
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


# ------------------------------------------------------------------------------


def _get_posterior_quantiles(
    samples: Dict, param: str, quantiles=(0.025, 0.5, 0.975)
):
    """Get quantiles for a specific parameter from MCMC samples."""
    param_samples = samples[param]
    return {q: jnp.quantile(param_samples, q) for q in quantiles}


# ------------------------------------------------------------------------------


def _get_map_estimate(
    samples: Dict, potential_energy: Optional[jnp.ndarray] = None
):
    """Get the maximum a posteriori (MAP) estimate from samples."""
    if potential_energy is not None:
        # Get index of minimum potential energy (maximum log density)
        map_idx = int(jnp.argmin(potential_energy))
        # Extract parameters at this index
        map_estimate = {
            param: values[map_idx] for param, values in samples.items()
        }
        return map_estimate
    else:
        # Fallback: Return posterior mean as a robust estimator
        map_estimate = {}
        for param, values in samples.items():
            # Using mean as a more robust estimator than mode
            map_estimate[param] = jnp.mean(values, axis=0)
        return map_estimate


# ------------------------------------------------------------------------------


def _generate_ppc_samples(
    samples: Dict,
    model_type: str,
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    rng_key: random.PRNGKey = random.PRNGKey(42),
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate predictive samples using posterior parameter samples."""
    # Get the model function
    model = _get_model_fn(model_type, model_config)

    # Prepare base model arguments
    model_args = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": model_config,
    }

    # Generate predictive samples
    from scribe.sampling import generate_predictive_samples

    return generate_predictive_samples(
        model,
        samples,
        model_args,
        rng_key=rng_key,
        batch_size=batch_size,
    )


# ------------------------------------------------------------------------------


def _generate_prior_predictive_samples(
    model_type: str,
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    rng_key: random.PRNGKey = random.PRNGKey(42),
    n_samples: int = 100,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate prior predictive samples using the model."""
    # Get the model function
    model = _get_model_fn(model_type, model_config)

    # Prepare base model arguments
    model_args = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": model_config,
    }

    # Generate prior predictive samples
    from scribe.sampling import generate_prior_predictive_samples

    return generate_prior_predictive_samples(
        model,
        model_args,
        rng_key=rng_key,
        n_samples=n_samples,
        batch_size=batch_size,
    )
