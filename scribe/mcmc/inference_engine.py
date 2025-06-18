"""
Inference engine for MCMC.

This module handles the execution of MCMC inference using NUTS.
"""

from typing import Union, Optional
import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, NUTS
from ..model_registry import get_model_and_guide
from ..model_config import ConstrainedModelConfig, ModelConfig


class MCMCInferenceEngine:
    """Handles MCMC inference execution."""
    
    @staticmethod
    def run_inference(
        model_config: Union[ConstrainedModelConfig, ModelConfig],
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        n_samples: int = 2_000,
        n_warmup: int = 1_000,
        n_chains: int = 1,
        seed: int = 42,
    ) -> any:
        """
        Execute MCMC inference using NUTS.
        
        Parameters
        ----------
        model_config : Union[ConstrainedModelConfig, ModelConfig]
            Model configuration object
        count_data : jnp.ndarray
            Processed count data (cells as rows)
        n_cells : int
            Number of cells
        n_genes : int
            Number of genes
        n_samples : int, default=2_000
            Number of MCMC samples
        n_warmup : int, default=1_000
            Number of warmup samples
        n_chains : int, default=1
            Number of parallel chains
        seed : int, default=42
            Random seed for reproducibility
            
        Returns
        -------
        numpyro.infer.mcmc.MCMCResults
            Results from MCMC run containing samples and diagnostics
        """
        # Determine if this is an unconstrained model
        is_unconstrained = isinstance(model_config, ModelConfig)
        
        # Get model function (no guide needed for MCMC)
        if is_unconstrained:
            model, _ = get_model_and_guide(
                model_config.base_model, 
                parameterization="unconstrained"
            )
        else:
            model, _ = get_model_and_guide(
                model_config.base_model, 
                parameterization="constrained"
            )
        
        # Create NUTS sampler
        nuts_kernel = NUTS(model)
        
        # Create MCMC instance
        mcmc = MCMC(nuts_kernel, num_samples=n_samples, num_warmup=n_warmup, num_chains=n_chains)
        
        # Create random key
        rng_key = random.PRNGKey(seed)
        
        # Prepare model arguments
        model_args = {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'counts': count_data,
            'model_config': model_config
        }
        
        # Run inference
        mcmc.run(rng_key, **model_args)
        
        return mcmc 