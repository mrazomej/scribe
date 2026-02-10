"""Variable Capture Probability (VCP) likelihoods for single-cell data.

This module provides likelihoods that include cell-specific capture probability
parameters, modeling technical variation in capture efficiency across cells.

Classes
-------
NBWithVCPLikelihood
    Negative Binomial with Variable Capture Probability.
ZINBWithVCPLikelihood
    Zero-Inflated NB with Variable Capture Probability.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import (
    Likelihood,
    _sample_phi_capture_constrained,
    _sample_phi_capture_unconstrained,
    _sample_p_capture_constrained,
    _sample_p_capture_unconstrained,
)

if TYPE_CHECKING:
    from ...builders.parameter_specs import ParamSpec
    from ...config import ModelConfig


# ==============================================================================
# Negative Binomial with Variable Capture Probability Likelihood
# ==============================================================================


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
    capture_param_name : str, optional
        Name of the capture parameter ("p_capture" or "phi_capture").
        If provided, this explicitly sets which parameterization to use.
        If None (default), the likelihood will detect it from cell_specs
        for backward compatibility.
    is_unconstrained : bool, default=False
        If True, use TransformedDistribution for capture parameter sampling.
        If False, use native constrained distribution (Beta/BetaPrime).
        This is determined at construction time to avoid runtime isinstance
        checks that can slow down JIT compilation.
    transform : Transform, optional
        The transform to use for unconstrained sampling (required if
        is_unconstrained=True).
    constrained_name : str, optional
        The name for the constrained parameter when using unconstrained
        sampling (e.g., "phi_capture" when the base is "phi_capture_unconstrained").

    Examples
    --------
    >>> # Explicit capture parameter name (recommended)
    >>> likelihood = NBWithVCPLikelihood(capture_param_name="phi_capture")
    >>>
    >>> # Auto-detect from cell_specs (backward compatible)
    >>> likelihood = NBWithVCPLikelihood()
    >>> # In model building (p_capture spec should be in cell_specs):
    >>> builder.with_likelihood(likelihood)
    """

    def __init__(
        self,
        capture_param_name: Optional[str] = None,
        is_unconstrained: bool = False,
        transform: Optional[dist.transforms.Transform] = None,
        constrained_name: Optional[str] = None,
    ):
        """Initialize likelihood with optional capture parameter name.

        Parameters
        ----------
        capture_param_name : str, optional
            "p_capture" for canonical/mean_prob parameterization,
            "phi_capture" for mean_odds parameterization.
            If None, auto-detect from cell_specs.
        is_unconstrained : bool, default=False
            If True, use TransformedDistribution for capture parameter.
        transform : Transform, optional
            Transform for unconstrained sampling.
        constrained_name : str, optional
            Name for constrained parameter in unconstrained mode.
        """
        self.capture_param_name = capture_param_name
        self.is_unconstrained = is_unconstrained
        self.transform = transform
        self.constrained_name = constrained_name or capture_param_name

    # --------------------------------------------------------------------------

    def _sample_capture_param(
        self,
        use_phi_capture: bool,
        capture_prior_params: Tuple[float, float],
    ) -> jnp.ndarray:
        """Sample the capture parameter using pre-configured sampling strategy.

        This method uses the configuration set at construction time to avoid
        runtime isinstance checks that can slow down JIT compilation.
        """
        if use_phi_capture:
            if self.is_unconstrained and self.transform is not None:
                return _sample_phi_capture_unconstrained(
                    capture_prior_params, self.transform, self.constrained_name
                )
            else:
                return _sample_phi_capture_constrained(capture_prior_params)
        else:
            if self.is_unconstrained and self.transform is not None:
                return _sample_p_capture_unconstrained(
                    capture_prior_params, self.transform, self.constrained_name
                )
            else:
                return _sample_p_capture_constrained(capture_prior_params)

    # --------------------------------------------------------------------------

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
        vae_cell_fn: Optional[
            Callable[[Optional[jnp.ndarray]], Dict[str, jnp.ndarray]]
        ] = None,
    ) -> None:
        """Sample from NB likelihood with variable capture probability."""
        n_cells = dims["n_cells"]
        # When vae_cell_fn is set, r (and possibly p) come from the decoder
        # inside the plate; do not read them here.
        if vae_cell_fn is None:
            p = param_values["p"]
            r = param_values["r"]
            is_mixture = "mixing_weights" in param_values

        # Determine which capture parameter we're using
        if self.capture_param_name is not None:
            use_phi_capture = self.capture_param_name == "phi_capture"
        else:
            # Auto-detect from cell_specs (backward compatibility)
            use_phi_capture = any(
                spec.name == "phi_capture" for spec in cell_specs
            )

        # Get prior params
        default_params = (1.0, 1.0)
        capture_prior_params = default_params

        # Try to get from param_specs if available
        target_name = "phi_capture" if use_phi_capture else "p_capture"
        for pspec in model_config.param_specs:
            if pspec.name == target_name and pspec.prior is not None:
                capture_prior_params = pspec.prior
                break

        # Create plate context based on mode
        if counts is None:
            # MODE 1: Prior predictive
            plate_context = numpyro.plate("cells", n_cells)
            obs = None
        elif batch_size is None:
            # MODE 2: Full sampling
            plate_context = numpyro.plate("cells", n_cells)
            obs = counts
        else:
            # MODE 3: Batch sampling - need idx for subsampling
            plate_context = numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            )
            obs = None  # Will be set inside plate

        with plate_context as idx:
            # Get observation for batch mode
            if batch_size is not None and counts is not None:
                obs = counts[idx]

            # VAE path: call decoder inside plate, then read decoder-driven
            # params
            if vae_cell_fn is not None:
                param_values.update(
                    vae_cell_fn(idx if batch_size is not None else None)
                )
                p = param_values["p"]
                r = param_values["r"]
                is_mixture = "mixing_weights" in param_values

            # Check if capture parameter is already in param_values (from
            # posterior_samples)
            # This happens when generating PPC samples with Predictive
            if target_name in param_values:
                # Use value from posterior_samples (for PPC)
                capture_value = param_values[target_name]
                # Handle batch indexing if needed
                if batch_size is not None and idx is not None:
                    capture_value = capture_value[idx]
            else:
                # Sample from prior (for prior predictive checks or when not in
                # param_values)
                capture_value = self._sample_capture_param(
                    use_phi_capture, capture_prior_params
                )

            if use_phi_capture:
                # Mean-odds parameterization
                phi = param_values["phi"]

                # Reshape capture for broadcasting
                if is_mixture:
                    capture_reshaped = capture_value[:, None, None]
                else:
                    capture_reshaped = capture_value[:, None]

                # Broadcast phi if needed
                if is_mixture:
                    if phi.ndim == 0:
                        phi = phi[None, None]
                    elif phi.ndim == 1:
                        phi = phi[:, None]

                # Compute logits
                logits = -jnp.log(phi * (1.0 + capture_reshaped))

                if is_mixture:
                    mixing_weights = param_values["mixing_weights"]
                    mixing_dist = dist.Categorical(probs=mixing_weights)
                    base_dist = dist.NegativeBinomialLogits(r, logits).to_event(
                        1
                    )
                    mixture_dist = dist.MixtureSameFamily(
                        mixing_dist, base_dist
                    )
                    numpyro.sample("counts", mixture_dist, obs=obs)
                else:
                    numpyro.sample(
                        "counts",
                        dist.NegativeBinomialLogits(r, logits).to_event(1),
                        obs=obs,
                    )
            else:
                # Canonical/mean-prob parameterization
                # Reshape capture for broadcasting
                if is_mixture:
                    capture_reshaped = capture_value[:, None, None]
                else:
                    capture_reshaped = capture_value[:, None]

                # Broadcast p if needed
                if is_mixture:
                    if p.ndim == 0:
                        p_for_hat = p
                    elif p.ndim == 1:
                        p_for_hat = p[:, None]
                    else:
                        p_for_hat = p
                else:
                    p_for_hat = p

                # Compute p_hat
                p_hat = (
                    p_for_hat
                    * capture_reshaped
                    / (1 - p_for_hat * (1 - capture_reshaped))
                )

                if is_mixture:
                    mixing_weights = param_values["mixing_weights"]
                    mixing_dist = dist.Categorical(probs=mixing_weights)
                    base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                    mixture_dist = dist.MixtureSameFamily(
                        mixing_dist, base_dist
                    )
                    numpyro.sample("counts", mixture_dist, obs=obs)
                else:
                    numpyro.sample(
                        "counts",
                        dist.NegativeBinomialProbs(r, p_hat).to_event(1),
                        obs=obs,
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
    capture_param_name : str, optional
        Name of the capture parameter ("p_capture" or "phi_capture").
        If provided, this explicitly sets which parameterization to use.
        If None (default), the likelihood will detect it from cell_specs
        for backward compatibility.
    is_unconstrained : bool, default=False
        If True, use TransformedDistribution for capture parameter sampling.
        If False, use native constrained distribution (Beta/BetaPrime).
    transform : Transform, optional
        The transform to use for unconstrained sampling.
    constrained_name : str, optional
        The name for the constrained parameter when using unconstrained sampling.

    Examples
    --------
    >>> # Explicit capture parameter name (recommended)
    >>> likelihood = ZINBWithVCPLikelihood(capture_param_name="phi_capture")
    >>>
    >>> # Auto-detect from cell_specs (backward compatible)
    >>> likelihood = ZINBWithVCPLikelihood()
    >>> # In model building:
    >>> builder.with_likelihood(likelihood)
    """

    def __init__(
        self,
        capture_param_name: Optional[str] = None,
        is_unconstrained: bool = False,
        transform: Optional[dist.transforms.Transform] = None,
        constrained_name: Optional[str] = None,
    ):
        """Initialize likelihood with optional capture parameter name.

        Parameters
        ----------
        capture_param_name : str, optional
            "p_capture" for canonical/mean_prob parameterization,
            "phi_capture" for mean_odds parameterization.
            If None, auto-detect from cell_specs.
        is_unconstrained : bool, default=False
            If True, use TransformedDistribution for capture parameter.
        transform : Transform, optional
            Transform for unconstrained sampling.
        constrained_name : str, optional
            Name for constrained parameter in unconstrained mode.
        """
        self.capture_param_name = capture_param_name
        self.is_unconstrained = is_unconstrained
        self.transform = transform
        self.constrained_name = constrained_name or capture_param_name

    # --------------------------------------------------------------------------

    def _sample_capture_param(
        self,
        use_phi_capture: bool,
        capture_prior_params: Tuple[float, float],
    ) -> jnp.ndarray:
        """Sample the capture parameter using pre-configured sampling strategy."""
        if use_phi_capture:
            if self.is_unconstrained and self.transform is not None:
                return _sample_phi_capture_unconstrained(
                    capture_prior_params, self.transform, self.constrained_name
                )
            else:
                return _sample_phi_capture_constrained(capture_prior_params)
        else:
            if self.is_unconstrained and self.transform is not None:
                return _sample_p_capture_unconstrained(
                    capture_prior_params, self.transform, self.constrained_name
                )
            else:
                return _sample_p_capture_constrained(capture_prior_params)

    # --------------------------------------------------------------------------

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
        vae_cell_fn: Optional[
            Callable[[Optional[jnp.ndarray]], Dict[str, jnp.ndarray]]
        ] = None,
    ) -> None:
        """Sample from ZINB likelihood with variable capture probability."""
        n_cells = dims["n_cells"]
        # When vae_cell_fn is set, r/gate (and possibly p) come from the decoder
        # inside the plate; do not read them here.
        if vae_cell_fn is None:
            p = param_values["p"]
            r = param_values["r"]
            gate = param_values["gate"]
            is_mixture = "mixing_weights" in param_values

        # Determine which capture parameter we're using
        if self.capture_param_name is not None:
            use_phi_capture = self.capture_param_name == "phi_capture"
        else:
            use_phi_capture = any(
                spec.name == "phi_capture" for spec in cell_specs
            )

        # Get prior params
        default_params = (1.0, 1.0)
        capture_prior_params = default_params

        target_name = "phi_capture" if use_phi_capture else "p_capture"
        for pspec in model_config.param_specs:
            if pspec.name == target_name and pspec.prior is not None:
                capture_prior_params = pspec.prior
                break

        # Broadcast gate if needed (non-VAE only; VAE path sets these inside
        # plate)
        if vae_cell_fn is None and is_mixture:
            if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
                gate = gate[None, :]
            elif gate.ndim == 0:
                gate = gate[None, None]

        # Create plate context based on mode
        if counts is None:
            plate_context = numpyro.plate("cells", n_cells)
            obs = None
        elif batch_size is None:
            plate_context = numpyro.plate("cells", n_cells)
            obs = counts
        else:
            plate_context = numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            )
            obs = None

        with plate_context as idx:
            if batch_size is not None and counts is not None:
                obs = counts[idx]

            # VAE path: call decoder inside plate, re-read updated params
            if vae_cell_fn is not None:
                param_values.update(
                    vae_cell_fn(idx if batch_size is not None else None)
                )
                p = param_values["p"]
                r = param_values["r"]
                gate = param_values["gate"]
                is_mixture = "mixing_weights" in param_values

            # Check if capture parameter is already in param_values (from posterior_samples)
            # This happens when generating PPC samples with Predictive
            if target_name in param_values:
                # Use value from posterior_samples (for PPC)
                capture_value = param_values[target_name]
                # Handle batch indexing if needed
                if batch_size is not None and idx is not None:
                    capture_value = capture_value[idx]
            else:
                # Sample from prior (for prior predictive checks or when not in param_values)
                capture_value = self._sample_capture_param(
                    use_phi_capture, capture_prior_params
                )

            if use_phi_capture:
                # Mean-odds parameterization
                phi = param_values["phi"]

                # Reshape capture for broadcasting
                if is_mixture:
                    capture_reshaped = capture_value[:, None, None]
                else:
                    capture_reshaped = capture_value[:, None]

                # Broadcast phi if needed
                if is_mixture:
                    if phi.ndim == 0:
                        phi = phi[None, None]
                    elif phi.ndim == 1:
                        phi = phi[:, None]

                # Compute logits
                logits = -jnp.log(phi * (1.0 + capture_reshaped))

                if is_mixture:
                    mixing_weights = param_values["mixing_weights"]
                    mixing_dist = dist.Categorical(probs=mixing_weights)
                    base_nb = dist.NegativeBinomialLogits(r, logits)
                    zinb_base = dist.ZeroInflatedDistribution(
                        base_nb, gate=gate
                    ).to_event(1)
                    mixture_dist = dist.MixtureSameFamily(
                        mixing_dist, zinb_base
                    )
                    numpyro.sample("counts", mixture_dist, obs=obs)
                else:
                    base_nb = dist.NegativeBinomialLogits(r, logits)
                    zinb_dist = dist.ZeroInflatedDistribution(
                        base_nb, gate=gate
                    )
                    numpyro.sample("counts", zinb_dist.to_event(1), obs=obs)
            else:
                # Canonical/mean-prob parameterization
                if is_mixture:
                    capture_reshaped = capture_value[:, None, None]
                else:
                    capture_reshaped = capture_value[:, None]

                # Broadcast p if needed
                if is_mixture:
                    if p.ndim == 0:
                        p_for_hat = p
                    elif p.ndim == 1:
                        p_for_hat = p[:, None]
                    else:
                        p_for_hat = p
                else:
                    p_for_hat = p

                # Compute p_hat
                p_hat = (
                    p_for_hat
                    * capture_reshaped
                    / (1 - p_for_hat * (1 - capture_reshaped))
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
                    numpyro.sample("counts", mixture_dist, obs=obs)
                else:
                    base_nb = dist.NegativeBinomialProbs(r, p_hat)
                    zinb_dist = dist.ZeroInflatedDistribution(
                        base_nb, gate=gate
                    )
                    numpyro.sample("counts", zinb_dist.to_event(1), obs=obs)
