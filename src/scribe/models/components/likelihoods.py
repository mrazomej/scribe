"""Likelihood components for count data models.

This module provides likelihood classes that handle the three plate modes:

- **Prior predictive** (counts=None): Sample synthetic counts from the prior
- **Full sampling** (counts provided, no batching): Condition on all cells
- **Batch sampling** (counts provided, with batch_size): Subsample for SVI

Each likelihood handles cell-specific parameter sampling and observation
sampling within the cell plate.

Classes
-------
Likelihood
    Abstract base class for likelihood components.
NegativeBinomialLikelihood
    Standard Negative Binomial likelihood for count data.
ZeroInflatedNBLikelihood
    Zero-Inflated Negative Binomial likelihood.
NBWithVCPLikelihood
    Negative Binomial with Variable Capture Probability.
ZINBWithVCPLikelihood
    Zero-Inflated NB with Variable Capture Probability.

Examples
--------
>>> from scribe.models.components import NegativeBinomialLikelihood
>>> likelihood = NegativeBinomialLikelihood()
>>> # Use in ModelBuilder
>>> builder.with_likelihood(likelihood)

See Also
--------
scribe.models.builders.model_builder : Uses likelihoods to build models.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

if TYPE_CHECKING:
    from ..builders.parameter_specs import ParamSpec
    from ..config import ModelConfig

# ------------------------------------------------------------------------------
# Likelihood Base Class
# ------------------------------------------------------------------------------


class Likelihood(ABC):
    """
    Abstract base class for likelihood components.

    Subclasses implement the `sample` method which handles:

    1. Cell plate creation (with proper batching mode)
    2. Cell-specific parameter sampling
    3. Observation sampling/conditioning

    All subclasses must handle three plate modes:

    - **Prior predictive**: counts=None → sample counts from prior
    - **Full sampling**: counts provided, batch_size=None → condition on all
    - **Batch sampling**: counts provided, batch_size set → subsample cells

    Examples
    --------
    >>> class MyLikelihood(Likelihood):
    ...     def sample(self, param_values, cell_specs, counts, dims,
    ...                batch_size, model_config):
    ...         # Implementation
    ...         pass
    """

    @abstractmethod
    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
    ) -> None:
        """
        Sample observations given parameters.

        Parameters
        ----------
        param_values : Dict[str, jnp.ndarray]
            Already-sampled parameter values (global and gene-specific).
            Keys are parameter names (e.g., "p", "r", "mu").
        cell_specs : List[ParamSpec]
            Specs for cell-specific parameters to sample inside the cell plate.
            These are sampled within the plate context.
        counts : Optional[jnp.ndarray]
            Observed counts matrix of shape (n_cells, n_genes).
            If None, samples from prior (prior predictive mode).
        dims : Dict[str, int]
            Dimension sizes, e.g., {"n_cells": 10000, "n_genes": 2000}.
        batch_size : Optional[int]
            Mini-batch size for stochastic VI. If None, uses all cells.
        model_config : ModelConfig
            Model configuration with hyperparameters.

        Notes
        -----
        This method should:
            1. Create the appropriate cell plate (with or without subsampling)
            2. Sample any cell-specific parameters from cell_specs
            3. Compute the likelihood distribution
            4. Sample or condition on counts
        """
        pass


# ------------------------------------------------------------------------------
# Negative Binomial Likelihood
# ------------------------------------------------------------------------------


class NegativeBinomialLikelihood(Likelihood):
    """
    Standard Negative Binomial likelihood for count data.

    Expects param_values to contain 'p' and 'r' (or derived equivalents).

    The Negative Binomial distribution is parameterized as:

        counts ~ NegativeBinomialProbs(r, p)

    where:
        - r > 0 is the dispersion parameter (number of failures)
        - p ∈ (0, 1) is the success probability

    Parameters
    ----------
    None

    Examples
    --------
    >>> likelihood = NegativeBinomialLikelihood()
    >>> # In model building:
    >>> builder.with_likelihood(likelihood)
    """

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
    ) -> None:
        """Sample from Negative Binomial likelihood.

        Handles three plate modes:
        - Prior predictive: sample counts from prior
        - Full: condition on all counts
        - Batched: condition on mini-batch with subsampling
        """
        n_cells = dims["n_cells"]
        p = param_values["p"]
        r = param_values["r"]

        # ====================================================================
        # Check if this is a mixture model
        # If r has shape (n_components, n_genes), we're in mixture mode
        # ====================================================================
        is_mixture = r.ndim == 2  # (n_components, n_genes) vs (n_genes,)

        if is_mixture:
            # ================================================================
            # Mixture model: use MixtureSameFamily
            # ================================================================
            mixing_weights = param_values["mixing_weights"]
            mixing_dist = dist.Categorical(probs=mixing_weights)

            # Broadcast p to match r shape if needed
            # p can be scalar (shared) or (n_components,) (component-specific)
            if p.ndim == 0:
                # Shared p: broadcast to (n_components, 1) for broadcasting
                p = p[None, None]
            elif p.ndim == 1:
                # Component-specific p: reshape to (n_components, 1)
                p = p[:, None]
            # p already (n_components, 1) or (n_components, n_genes)

            # Base distribution for each component
            base_dist_component = dist.NegativeBinomialProbs(r, p).to_event(1)
            base_dist = dist.MixtureSameFamily(mixing_dist, base_dist_component)
        else:
            # ================================================================
            # Single-component model: standard distribution
            # ================================================================
            base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)

        # ====================================================================
        # MODE 1: Prior predictive (counts=None)
        # Sample synthetic counts from the prior - no conditioning
        # Used for prior predictive checks and synthetic data generation
        # ====================================================================
        if counts is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific params if any (e.g., p_capture for VCP)
                # Note: For standard NB, cell_specs is typically empty
                for spec in cell_specs:
                    from ..builders.parameter_specs import sample_prior

                    sample_prior(spec, dims, model_config)
                # Sample counts from prior
                numpyro.sample("counts", base_dist)

        # ====================================================================
        # MODE 2: Full sampling (counts provided, no batch_size)
        # Condition on all cells at once - used for MCMC or small datasets
        # ====================================================================
        elif batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific params if any
                for spec in cell_specs:
                    from ..builders.parameter_specs import sample_prior

                    sample_prior(spec, dims, model_config)
                # Condition on observed counts
                numpyro.sample("counts", base_dist, obs=counts)

        # ====================================================================
        # MODE 3: Batch sampling (counts provided, with batch_size)
        # Subsample cells for stochastic VI on large datasets
        # The plate returns indices for the current mini-batch
        # ====================================================================
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample cell-specific params if any
                for spec in cell_specs:
                    from ..builders.parameter_specs import sample_prior

                    sample_prior(spec, dims, model_config)
                # Condition on subsampled counts
                numpyro.sample("counts", base_dist, obs=counts[idx])


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Likelihood
# ------------------------------------------------------------------------------


class ZeroInflatedNBLikelihood(Likelihood):
    """
    Zero-Inflated Negative Binomial likelihood for count data.

    Expects param_values to contain 'p', 'r', and 'gate'.

    The Zero-Inflated Negative Binomial is a mixture model:

        counts ~ ZeroInflatedNegativeBinomial(gate, r, p)

    where:
        - gate ∈ (0, 1) is the zero-inflation probability per gene
        - r > 0 is the dispersion parameter
        - p ∈ (0, 1) is the success probability

    With probability `gate`, the count is zero (structural zero).
    With probability `1 - gate`, the count follows NegativeBinomialProbs(r, p).

    Parameters
    ----------
    None

    Examples
    --------
    >>> likelihood = ZeroInflatedNBLikelihood()
    >>> # In model building:
    >>> builder.with_likelihood(likelihood)
    """

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
    ) -> None:
        """Sample from Zero-Inflated Negative Binomial likelihood."""
        n_cells = dims["n_cells"]
        p = param_values["p"]
        r = param_values["r"]
        gate = param_values["gate"]

        # ====================================================================
        # Check if this is a mixture model
        # ====================================================================
        is_mixture = r.ndim == 2  # (n_components, n_genes) vs (n_genes,)

        if is_mixture:
            # ================================================================
            # Mixture model: use MixtureSameFamily
            # ================================================================
            mixing_weights = param_values["mixing_weights"]
            mixing_dist = dist.Categorical(probs=mixing_weights)

            # Broadcast p to match r shape if needed
            if p.ndim == 0:
                p = p[None, None]
            elif p.ndim == 1:
                p = p[:, None]

            # Broadcast gate to match r shape if needed
            if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
                # gate is (n_genes,) - broadcast to (n_components, n_genes)
                gate = gate[None, :]
            elif gate.ndim == 2:
                # gate is already (n_components, n_genes)
                pass
            else:
                # gate is scalar - broadcast
                gate = gate[None, None]

            # Base distribution for each component
            base_nb = dist.NegativeBinomialProbs(r, p)
            zinb_base = dist.ZeroInflatedDistribution(
                base_nb, gate=gate
            ).to_event(1)
            base_dist = dist.MixtureSameFamily(mixing_dist, zinb_base)
        else:
            # ================================================================
            # Single-component model: standard distribution
            # ================================================================
            base_nb = dist.NegativeBinomialProbs(r, p)
            base_dist = dist.ZeroInflatedDistribution(
                base_nb, gate=gate
            ).to_event(1)

        # ====================================================================
        # MODE 1: Prior predictive
        # ====================================================================
        if counts is None:
            with numpyro.plate("cells", n_cells):
                for spec in cell_specs:
                    from ..builders.parameter_specs import sample_prior

                    sample_prior(spec, dims, model_config)
                numpyro.sample("counts", base_dist)

        # ====================================================================
        # MODE 2: Full sampling
        # ====================================================================
        elif batch_size is None:
            with numpyro.plate("cells", n_cells):
                for spec in cell_specs:
                    from ..builders.parameter_specs import sample_prior

                    sample_prior(spec, dims, model_config)
                numpyro.sample("counts", base_dist, obs=counts)

        # ====================================================================
        # MODE 3: Batch sampling
        # ====================================================================
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                for spec in cell_specs:
                    from ..builders.parameter_specs import sample_prior

                    sample_prior(spec, dims, model_config)
                numpyro.sample("counts", base_dist, obs=counts[idx])


# ------------------------------------------------------------------------------
# Negative Binomial with Variable Capture Probability Likelihood
# ------------------------------------------------------------------------------


class NBWithVCPLikelihood(Likelihood):
    """Negative Binomial with Variable Capture Probability.

    Includes cell-specific p_capture parameter that modulates
    the effective capture probability per cell.

    The effective probability becomes:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))

    where p_capture is sampled per-cell inside the cell plate.

    This models technical variation in capture efficiency across cells,
    which is common in single-cell RNA sequencing.

    Parameters
    ----------
    None

    Examples
    --------
    >>> likelihood = NBWithVCPLikelihood()
    >>> # In model building (p_capture spec should be in cell_specs):
    >>> builder.with_likelihood(likelihood)
    """

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
    ) -> None:
        """Sample from NB likelihood with variable capture probability."""
        n_cells = dims["n_cells"]
        p = param_values["p"]
        r = param_values["r"]

        # ====================================================================
        # Check if this is a mixture model
        # ====================================================================
        is_mixture = r.ndim == 2  # (n_components, n_genes) vs (n_genes,)

        # ====================================================================
        # Get capture parameter spec from cell_specs
        # Check for both p_capture (canonical/mean_prob) and phi_capture (mean_odds)
        # ====================================================================
        capture_spec = None
        for spec in cell_specs:
            if spec.name in ("p_capture", "phi_capture"):
                capture_spec = spec
                break

        # Determine which capture parameter we're using
        use_phi_capture = (
            capture_spec is not None and capture_spec.name == "phi_capture"
        )

        # Get prior params - check both p_capture and phi_capture
        if use_phi_capture:
            prior_attr = "phi_capture"
            default_params = (1.0, 1.0)  # BetaPrime default
        else:
            prior_attr = "p_capture"
            default_params = (1.0, 1.0)  # Beta default

        capture_prior_params = default_params
        if capture_spec is not None:
            # Extract prior from param_specs
            prior_value = None
            param_name = prior_attr.replace("_prior", "").replace(
                "_capture", "_capture"
            )
            for pspec in model_config.param_specs:
                if pspec.name == param_name:
                    prior_value = pspec.prior
                    break
            capture_prior_params = (
                prior_value
                if prior_value is not None
                else capture_spec.default_params
            )

        # ====================================================================
        # MODE 1: Prior predictive (counts=None)
        # ====================================================================
        if counts is None:
            with numpyro.plate("cells", n_cells):
                if use_phi_capture:
                    # Mean-odds parameterization: sample phi_capture, use logits
                    from scribe.stats.distributions import BetaPrime

                    phi_capture = numpyro.sample(
                        "phi_capture", BetaPrime(*capture_prior_params)
                    )
                    phi = param_values["phi"]
                    # Reshape phi_capture for broadcasting: (n_cells,) -> (n_cells, 1, 1)
                    phi_capture_reshaped = phi_capture[:, None, None]

                    if is_mixture:
                        # Mixture model
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)

                        # Broadcast phi to match r shape
                        if phi.ndim == 0:
                            # Shared phi: scalar -> (1, 1)
                            phi = phi[None, None]
                        elif phi.ndim == 1:
                            # Component-specific phi: (n_components,) -> (n_components, 1)
                            phi = phi[:, None]

                        # Compute logits: phi is (n_components, 1) or (1, 1)
                        # phi_capture_reshaped is (n_cells, 1, 1)
                        # Result broadcasts to (n_cells, 1, n_components) which works with r (n_components, n_genes)
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        base_dist = dist.NegativeBinomialLogits(
                            r, logits
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, base_dist
                        )
                        numpyro.sample("counts", mixture_dist)
                    else:
                        # Single-component
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        numpyro.sample(
                            "counts",
                            dist.NegativeBinomialLogits(r, logits).to_event(1),
                        )
                else:
                    # Canonical/mean-prob: sample p_capture, compute p_hat
                    p_capture = numpyro.sample(
                        "p_capture", dist.Beta(*capture_prior_params)
                    )
                    # Reshape p_capture for broadcasting with components
                    # p_capture is (n_cells,) or (batch_size,), reshape to (n_cells, 1, 1)
                    p_capture_reshaped = p_capture[:, None, None]

                    # Broadcast p to match if needed (for component-specific p)
                    if is_mixture:
                        # p can be scalar (shared) or (n_components,) (component-specific)
                        if p.ndim == 0:
                            # Shared p: keep as scalar, will broadcast
                            p_for_hat = p
                        elif p.ndim == 1:
                            # Component-specific p: reshape to (n_components, 1)
                            p_for_hat = p[:, None]
                        else:
                            p_for_hat = p
                    else:
                        p_for_hat = p

                    # Compute p_hat using the derived formula
                    # This broadcasts correctly:
                    # - If p is scalar: p_hat shape is (n_cells, 1, 1)
                    # - If p is (n_components, 1): p_hat shape is (n_cells, 1, n_components)
                    p_hat = (
                        p_for_hat
                        * p_capture_reshaped
                        / (1 - p_for_hat * (1 - p_capture_reshaped))
                    )

                    if is_mixture:
                        # Mixture model: use MixtureSameFamily
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)

                        # r is (n_components, n_genes)
                        # p_hat broadcasts correctly with r when creating the distribution
                        # NumPyro handles the broadcasting automatically
                        base_dist = dist.NegativeBinomialProbs(
                            r, p_hat
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, base_dist
                        )
                        numpyro.sample("counts", mixture_dist)
                    else:
                        numpyro.sample(
                            "counts",
                            dist.NegativeBinomialProbs(r, p_hat).to_event(1),
                        )

        # ====================================================================
        # MODE 2: Full sampling (counts provided, no batch_size)
        # ====================================================================
        elif batch_size is None:
            with numpyro.plate("cells", n_cells):
                if use_phi_capture:
                    # Mean-odds parameterization: sample phi_capture, use logits
                    from scribe.stats.distributions import BetaPrime

                    phi_capture = numpyro.sample(
                        "phi_capture", BetaPrime(*capture_prior_params)
                    )
                    phi = param_values["phi"]
                    phi_capture_reshaped = phi_capture[:, None, None]

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        if phi.ndim == 0:
                            phi = phi[None, None]
                        elif phi.ndim == 1:
                            phi = phi[:, None]
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        base_dist = dist.NegativeBinomialLogits(
                            r, logits
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, base_dist
                        )
                        numpyro.sample("counts", mixture_dist, obs=counts)
                    else:
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        numpyro.sample(
                            "counts",
                            dist.NegativeBinomialLogits(r, logits).to_event(1),
                            obs=counts,
                        )
                else:
                    # Canonical/mean-prob: sample p_capture, compute p_hat
                    p_capture = numpyro.sample(
                        "p_capture", dist.Beta(*capture_prior_params)
                    )
                    # Reshape p_capture for broadcasting with components
                    p_capture_reshaped = p_capture[:, None, None]

                    # Broadcast p to match if needed
                    if is_mixture:
                        if p.ndim == 0:
                            p_for_hat = p
                        elif p.ndim == 1:
                            p_for_hat = p[:, None]
                        else:
                            p_for_hat = p
                    else:
                        p_for_hat = p

                    p_hat = (
                        p_for_hat
                        * p_capture_reshaped
                        / (1 - p_for_hat * (1 - p_capture_reshaped))
                    )

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        base_dist = dist.NegativeBinomialProbs(
                            r, p_hat
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, base_dist
                        )
                        numpyro.sample("counts", mixture_dist, obs=counts)
                    else:
                        numpyro.sample(
                            "counts",
                            dist.NegativeBinomialProbs(r, p_hat).to_event(1),
                            obs=counts,
                        )

        # ====================================================================
        # MODE 3: Batch sampling (counts provided, with batch_size)
        # ====================================================================
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                if use_phi_capture:
                    # Mean-odds parameterization: sample phi_capture, use logits
                    from scribe.stats.distributions import BetaPrime

                    phi_capture = numpyro.sample(
                        "phi_capture", BetaPrime(*capture_prior_params)
                    )
                    phi = param_values["phi"]
                    phi_capture_reshaped = phi_capture[:, None, None]

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        if phi.ndim == 0:
                            phi = phi[None, None]
                        elif phi.ndim == 1:
                            phi = phi[:, None]
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        base_dist = dist.NegativeBinomialLogits(
                            r, logits
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, base_dist
                        )
                        numpyro.sample("counts", mixture_dist, obs=counts[idx])
                    else:
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        numpyro.sample(
                            "counts",
                            dist.NegativeBinomialLogits(r, logits).to_event(1),
                            obs=counts[idx],
                        )
                else:
                    # Canonical/mean-prob: sample p_capture, compute p_hat
                    p_capture = numpyro.sample(
                        "p_capture", dist.Beta(*capture_prior_params)
                    )
                    # Reshape p_capture for broadcasting with components
                    p_capture_reshaped = p_capture[:, None, None]

                    # Broadcast p to match if needed
                    if is_mixture:
                        if p.ndim == 0:
                            p_for_hat = p
                        elif p.ndim == 1:
                            p_for_hat = p[:, None]
                        else:
                            p_for_hat = p
                    else:
                        p_for_hat = p

                    p_hat = (
                        p_for_hat
                        * p_capture_reshaped
                        / (1 - p_for_hat * (1 - p_capture_reshaped))
                    )

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        base_dist = dist.NegativeBinomialProbs(
                            r, p_hat
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, base_dist
                        )
                        numpyro.sample("counts", mixture_dist, obs=counts[idx])
                    else:
                        numpyro.sample(
                            "counts",
                            dist.NegativeBinomialProbs(r, p_hat).to_event(1),
                            obs=counts[idx],
                        )


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial with Variable Capture Probability Likelihood
# ------------------------------------------------------------------------------


class ZINBWithVCPLikelihood(Likelihood):
    """Zero-Inflated Negative Binomial with Variable Capture Probability.

    Combines zero-inflation with cell-specific capture probability.

    The model is:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))
        counts ~ ZeroInflatedNegativeBinomial(gate, r, p_hat)

    Parameters
    ----------
    None

    Examples
    --------
    >>> likelihood = ZINBWithVCPLikelihood()
    >>> # In model building:
    >>> builder.with_likelihood(likelihood)
    """

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
    ) -> None:
        """Sample from ZINB likelihood with variable capture probability."""
        n_cells = dims["n_cells"]
        p = param_values["p"]
        r = param_values["r"]
        gate = param_values["gate"]

        # ====================================================================
        # Check if this is a mixture model
        # ====================================================================
        is_mixture = r.ndim == 2  # (n_components, n_genes) vs (n_genes,)

        # ====================================================================
        # Get capture parameter spec from cell_specs
        # Check for both p_capture (canonical/mean_prob) and phi_capture (mean_odds)
        # ====================================================================
        capture_spec = None
        for spec in cell_specs:
            if spec.name in ("p_capture", "phi_capture"):
                capture_spec = spec
                break

        # Determine which capture parameter we're using
        use_phi_capture = (
            capture_spec is not None and capture_spec.name == "phi_capture"
        )

        # Get prior params - check both p_capture and phi_capture
        if use_phi_capture:
            prior_attr = "phi_capture"
            default_params = (1.0, 1.0)  # BetaPrime default
        else:
            prior_attr = "p_capture"
            default_params = (1.0, 1.0)  # Beta default

        capture_prior_params = default_params
        if capture_spec is not None:
            # Extract prior from param_specs
            prior_value = None
            param_name = prior_attr.replace("_prior", "").replace(
                "_capture", "_capture"
            )
            for pspec in model_config.param_specs:
                if pspec.name == param_name:
                    prior_value = pspec.prior
                    break
            capture_prior_params = (
                prior_value
                if prior_value is not None
                else capture_spec.default_params
            )

        # ====================================================================
        # MODE 1: Prior predictive
        # ====================================================================
        if counts is None:
            with numpyro.plate("cells", n_cells):
                if use_phi_capture:
                    # Mean-odds parameterization: sample phi_capture, use logits
                    from scribe.stats.distributions import BetaPrime

                    phi_capture = numpyro.sample(
                        "phi_capture", BetaPrime(*capture_prior_params)
                    )
                    phi = param_values["phi"]
                    phi_capture_reshaped = phi_capture[:, None, None]

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        if phi.ndim == 0:
                            phi = phi[None, None]
                        elif phi.ndim == 1:
                            phi = phi[:, None]
                        # Broadcast gate if needed
                        if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
                            gate = gate[None, :]
                        elif gate.ndim == 2:
                            pass
                        else:
                            gate = gate[None, None]
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        base_nb = dist.NegativeBinomialLogits(r, logits)
                        zinb_base = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, zinb_base
                        )
                        numpyro.sample("counts", mixture_dist)
                    else:
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        base_nb = dist.NegativeBinomialLogits(r, logits)
                        zinb_dist = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        )
                        numpyro.sample("counts", zinb_dist.to_event(1))
                else:
                    # Canonical/mean-prob: sample p_capture, compute p_hat
                    p_capture = numpyro.sample(
                        "p_capture", dist.Beta(*capture_prior_params)
                    )
                    p_capture_reshaped = p_capture[:, None, None]

                    # Broadcast p and gate if needed
                    if is_mixture:
                        if p.ndim == 0:
                            p_for_hat = p
                        elif p.ndim == 1:
                            p_for_hat = p[:, None]
                        else:
                            p_for_hat = p
                        if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
                            gate = gate[None, :]
                        elif gate.ndim == 2:
                            pass
                        else:
                            gate = gate[None, None]
                    else:
                        p_for_hat = p

                    p_hat = (
                        p_for_hat
                        * p_capture_reshaped
                        / (1 - p_for_hat * (1 - p_capture_reshaped))
                    )

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        base_nb = dist.NegativeBinomialProbs(r, p_hat)
                        zinb_base = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, zinb_base
                        )
                        numpyro.sample("counts", mixture_dist)
                    else:
                        base_nb = dist.NegativeBinomialProbs(r, p_hat)
                        zinb_dist = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        )
                        numpyro.sample("counts", zinb_dist.to_event(1))

        # ====================================================================
        # MODE 2: Full sampling
        # ====================================================================
        elif batch_size is None:
            with numpyro.plate("cells", n_cells):
                if use_phi_capture:
                    # Mean-odds parameterization: sample phi_capture, use logits
                    from scribe.stats.distributions import BetaPrime

                    phi_capture = numpyro.sample(
                        "phi_capture", BetaPrime(*capture_prior_params)
                    )
                    phi = param_values["phi"]
                    phi_capture_reshaped = phi_capture[:, None, None]

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        if phi.ndim == 0:
                            phi = phi[None, None]
                        elif phi.ndim == 1:
                            phi = phi[:, None]
                        if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
                            gate = gate[None, :]
                        elif gate.ndim == 2:
                            pass
                        else:
                            gate = gate[None, None]
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        base_nb = dist.NegativeBinomialLogits(r, logits)
                        zinb_base = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, zinb_base
                        )
                        numpyro.sample("counts", mixture_dist, obs=counts)
                    else:
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        base_nb = dist.NegativeBinomialLogits(r, logits)
                        zinb_dist = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        )
                        numpyro.sample(
                            "counts", zinb_dist.to_event(1), obs=counts
                        )
                else:
                    # Canonical/mean-prob: sample p_capture, compute p_hat
                    p_capture = numpyro.sample(
                        "p_capture", dist.Beta(*capture_prior_params)
                    )
                    p_capture_reshaped = p_capture[:, None, None]

                    if is_mixture:
                        if p.ndim == 0:
                            p_for_hat = p
                        elif p.ndim == 1:
                            p_for_hat = p[:, None]
                        else:
                            p_for_hat = p
                        if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
                            gate = gate[None, :]
                        elif gate.ndim == 2:
                            pass
                        else:
                            gate = gate[None, None]
                    else:
                        p_for_hat = p

                    p_hat = (
                        p_for_hat
                        * p_capture_reshaped
                        / (1 - p_for_hat * (1 - p_capture_reshaped))
                    )

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        base_nb = dist.NegativeBinomialProbs(r, p_hat)
                        zinb_base = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, zinb_base
                        )
                        numpyro.sample("counts", mixture_dist, obs=counts)
                    else:
                        base_nb = dist.NegativeBinomialProbs(r, p_hat)
                        zinb_dist = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        )
                        numpyro.sample(
                            "counts", zinb_dist.to_event(1), obs=counts
                        )

        # ====================================================================
        # MODE 3: Batch sampling
        # ====================================================================
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                if use_phi_capture:
                    # Mean-odds parameterization: sample phi_capture, use logits
                    from scribe.stats.distributions import BetaPrime

                    phi_capture = numpyro.sample(
                        "phi_capture", BetaPrime(*capture_prior_params)
                    )
                    phi = param_values["phi"]
                    phi_capture_reshaped = phi_capture[:, None, None]

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        if phi.ndim == 0:
                            phi = phi[None, None]
                        elif phi.ndim == 1:
                            phi = phi[:, None]
                        if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
                            gate = gate[None, :]
                        elif gate.ndim == 2:
                            pass
                        else:
                            gate = gate[None, None]
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        base_nb = dist.NegativeBinomialLogits(r, logits)
                        zinb_base = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, zinb_base
                        )
                        numpyro.sample("counts", mixture_dist, obs=counts[idx])
                    else:
                        logits = -jnp.log(phi * (1.0 + phi_capture_reshaped))
                        base_nb = dist.NegativeBinomialLogits(r, logits)
                        zinb_dist = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        )
                        numpyro.sample(
                            "counts", zinb_dist.to_event(1), obs=counts[idx]
                        )
                else:
                    # Canonical/mean-prob: sample p_capture, compute p_hat
                    p_capture = numpyro.sample(
                        "p_capture", dist.Beta(*capture_prior_params)
                    )
                    p_capture_reshaped = p_capture[:, None, None]

                    if is_mixture:
                        if p.ndim == 0:
                            p_for_hat = p
                        elif p.ndim == 1:
                            p_for_hat = p[:, None]
                        else:
                            p_for_hat = p
                        if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
                            gate = gate[None, :]
                        elif gate.ndim == 2:
                            pass
                        else:
                            gate = gate[None, None]
                    else:
                        p_for_hat = p

                    p_hat = (
                        p_for_hat
                        * p_capture_reshaped
                        / (1 - p_for_hat * (1 - p_capture_reshaped))
                    )

                    if is_mixture:
                        mixing_weights = param_values["mixing_weights"]
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                        base_nb = dist.NegativeBinomialProbs(r, p_hat)
                        zinb_base = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        ).to_event(1)
                        mixture_dist = dist.MixtureSameFamily(
                            mixing_dist, zinb_base
                        )
                        numpyro.sample("counts", mixture_dist, obs=counts[idx])
                    else:
                        base_nb = dist.NegativeBinomialProbs(r, p_hat)
                        zinb_dist = dist.ZeroInflatedDistribution(
                            base_nb, gate=gate
                        )
                        numpyro.sample(
                            "counts", zinb_dist.to_event(1), obs=counts[idx]
                        )
