"""
Inference engine for MCMC.

This module handles the execution of MCMC inference using NUTS.
"""

import warnings
from typing import Any, Dict, Optional, Union

import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, NUTS

from ..models.config import ModelConfig
from ..models.model_registry import get_model_and_guide


class MCMCInferenceEngine:
    """Handles MCMC inference execution."""

    @staticmethod
    def run_inference(
        model_config: ModelConfig,
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        n_samples: int = 2_000,
        n_warmup: int = 1_000,
        n_chains: int = 1,
        seed: int = 42,
        mcmc_kwargs: Optional[dict] = None,
        annotation_prior_logits: Optional[jnp.ndarray] = None,
        init_values: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Any:
        """Execute MCMC inference using NUTS.

        Parameters
        ----------
        model_config : ModelConfig
            Model configuration object.
        count_data : jnp.ndarray
            Processed count data (cells as rows).
        n_cells : int
            Number of cells.
        n_genes : int
            Number of genes.
        n_samples : int, default=2_000
            Number of MCMC samples.
        n_warmup : int, default=1_000
            Number of warmup samples.
        n_chains : int, default=1
            Number of parallel chains.
        seed : int, default=42
            Random seed for reproducibility.
        mcmc_kwargs : Optional[dict], default=None
            Keyword arguments for the NUTS kernel (e.g.,
            ``target_accept_prob``, ``max_tree_depth``).
        annotation_prior_logits : Optional[jnp.ndarray], default=None
            Prior logits for annotation-guided mixture models.
        init_values : Optional[Dict[str, jnp.ndarray]], default=None
            Constrained-space values to initialize MCMC chains via
            ``init_to_value``.  Typically obtained from
            ``compute_init_values(svi_results.get_map(...), model_config)``.
            When provided, an ``init_strategy`` is constructed and merged
            into the NUTS kernel kwargs.  If ``mcmc_kwargs`` already
            contains an ``init_strategy``, a warning is emitted and the
            existing strategy is overridden.

        Returns
        -------
        numpyro.infer.MCMC
            Results from the MCMC run containing samples and diagnostics.
        """
        # Get model function (no guide needed for MCMC)
        model, _, _ = get_model_and_guide(model_config, guide_families=None)

        # Build effective NUTS kwargs, optionally injecting init_to_value
        effective_kwargs: Dict[str, Any] = dict(mcmc_kwargs or {})
        if init_values is not None:
            from numpyro.infer.initialization import init_to_value

            if "init_strategy" in effective_kwargs:
                warnings.warn(
                    "init_values overrides the existing init_strategy "
                    "in mcmc_kwargs.",
                    UserWarning,
                    stacklevel=2,
                )
            effective_kwargs["init_strategy"] = init_to_value(
                values=init_values
            )

        # Create NUTS sampler with the (possibly augmented) kwargs
        nuts_kernel = NUTS(model, **effective_kwargs)

        # Create MCMC instance
        mcmc = MCMC(
            nuts_kernel,
            num_samples=n_samples,
            num_warmup=n_warmup,
            num_chains=n_chains,
        )

        # Create random number generator key
        rng_key = random.PRNGKey(seed)

        # Prepare model arguments
        model_args = {
            "n_cells": n_cells,
            "n_genes": n_genes,
            "counts": count_data,
            "model_config": model_config,
            "annotation_prior_logits": annotation_prior_logits,
        }

        # Run inference
        mcmc.run(rng_key, **model_args)

        return mcmc
