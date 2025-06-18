"""
Inference engine for SVI.

This module handles the execution of SVI inference including setting up the
SVI instance and running the optimization.
"""

from typing import Optional, Dict, Any
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import SVI, TraceMeanField_ELBO
from ..model_registry import get_model_and_guide
from ..model_config import ModelConfig


class SVIInferenceEngine:
    """Handles SVI inference execution."""
    
    @staticmethod
    def run_inference(
        model_config: ModelConfig,
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(step_size=0.001),
        loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
        n_steps: int = 100_000,
        batch_size: Optional[int] = None,
        seed: int = 42,
        stable_update: bool = True,
    ) -> Any:
        """
        Execute SVI inference.
        
        Parameters
        ----------
        model_config : ModelConfig
            Model configuration object
        count_data : jnp.ndarray
            Processed count data (cells as rows)
        n_cells : int
            Number of cells
        n_genes : int
            Number of genes
        optimizer : numpyro.optim.optimizers, default=Adam(step_size=0.001)
            Optimizer for variational inference
        loss : numpyro.infer.elbo, default=TraceMeanField_ELBO()
            Loss function for variational inference
        n_steps : int, default=100_000
            Number of optimization steps
        batch_size : Optional[int], default=None
            Mini-batch size. If None, uses full dataset.
        seed : int, default=42
            Random seed for reproducibility
        stable_update : bool, default=True
            Whether to use numerically stable parameter updates
            
        Returns
        -------
        numpyro.infer.svi.SVIRunResult
            Results from SVI run containing optimized parameters and loss history
        """
        # Get model and guide functions
        model, guide = get_model_and_guide(
            model_config.base_model, 
            parameterization=model_config.parameterization
        )
        
        # Create SVI instance
        svi = SVI(model, guide, optimizer, loss=loss)
        
        # Create random key
        rng_key = random.PRNGKey(seed)
        
        # Prepare model arguments
        model_args = {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'counts': count_data,
            'batch_size': batch_size,
            'model_config': model_config
        }
        
        # Run inference
        svi_results = svi.run(
            rng_key,
            n_steps,
            stable_update=stable_update,
            **model_args
        )
        
        return svi_results 