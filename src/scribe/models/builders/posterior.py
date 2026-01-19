"""
Posterior distribution extraction for the composable builder system.

This module provides functions to extract posterior distributions from
optimized variational parameters. It replaces the parameterization-specific
`get_posterior_distributions` functions in the deprecated model files.

Functions
---------
get_posterior_distributions
    Extract posterior distributions from optimized guide parameters.

Examples
--------
>>> from scribe.models.builders.posterior import get_posterior_distributions
>>> from scribe.models.config import ModelConfig
>>>
>>> # After SVI inference
>>> posteriors = get_posterior_distributions(
...     params=svi_results.params,
...     model_config=model_config,
... )
>>> p_dist = posteriors["p"]  # Beta distribution
>>> r_dist = posteriors["r"]  # LogNormal distribution
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import jax.numpy as jnp
import numpyro.distributions as dist

# Import Parameterization enum directly to avoid circular import
from ..config.enums import Parameterization
from scribe.stats.distributions import BetaPrime

if TYPE_CHECKING:
    from ..config import ModelConfig


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    split: bool = False,
) -> Dict[str, Any]:
    """
    Extract posterior distributions from optimized variational guide parameters.

    This function constructs numpyro distributions from the variational
    parameters learned during SVI inference. It handles all parameterizations
    (canonical/standard, mean_prob/linked, mean_odds/odds_ratio) and both
    constrained and unconstrained variants.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Dictionary of optimized variational parameters from SVI inference.
        Expected keys depend on the parameterization:
        - canonical/standard: p_alpha, p_beta, r_loc, r_scale
        - mean_prob/linked: p_alpha, p_beta, mu_loc, mu_scale
        - mean_odds/odds_ratio: phi_alpha, phi_beta, mu_loc, mu_scale
        For unconstrained: uses _loc, _scale instead of _alpha, _beta
    model_config : ModelConfig
        Model configuration specifying parameterization and model type.
    split : bool, default=False
        If True, return lists of univariate distributions for vector-valued
        parameters (e.g., one distribution per gene). If False, return
        batched distributions.

    Returns
    -------
    Dict[str, Union[dist.Distribution, List[dist.Distribution]]]
        Dictionary mapping parameter names to their posterior distributions.
        For unconstrained parameterizations, returns TransformedDistribution
        objects.

    Examples
    --------
    >>> posteriors = get_posterior_distributions(params, model_config)
    >>> p_posterior = posteriors["p"]
    >>> r_posterior = posteriors["r"]
    >>>
    >>> # Get samples from posterior
    >>> p_samples = p_posterior.sample(random.PRNGKey(0), (1000,))
    """
    distributions = {}
    parameterization = model_config.parameterization
    unconstrained = model_config.unconstrained
    is_mixture = model_config.is_mixture
    is_zero_inflated = model_config.is_zero_inflated
    uses_vcp = model_config.uses_variable_capture

    # Check if this is a low-rank guide by looking for low-rank parameters
    low_rank = any(
        key.startswith("log_") and key.endswith("_W") for key in params.keys()
    )

    # -------------------------------------------------------------------------
    # Build distributions based on parameterization
    # -------------------------------------------------------------------------

    if parameterization in (
        Parameterization.CANONICAL,
        Parameterization.STANDARD,
    ):
        distributions.update(
            _build_canonical_posteriors(
                params, unconstrained, is_mixture, low_rank, split
            )
        )
    elif parameterization in (
        Parameterization.MEAN_PROB,
        Parameterization.LINKED,
    ):
        distributions.update(
            _build_mean_prob_posteriors(
                params, unconstrained, is_mixture, low_rank, split
            )
        )
    elif parameterization in (
        Parameterization.MEAN_ODDS,
        Parameterization.ODDS_RATIO,
    ):
        distributions.update(
            _build_mean_odds_posteriors(
                params, unconstrained, is_mixture, low_rank, split
            )
        )
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    # -------------------------------------------------------------------------
    # Add zero-inflation gate if applicable
    # -------------------------------------------------------------------------
    if is_zero_inflated:
        distributions.update(
            _build_gate_posterior(params, unconstrained, is_mixture, split)
        )

    # -------------------------------------------------------------------------
    # Add capture probability if applicable
    # -------------------------------------------------------------------------
    if uses_vcp:
        # For mean_odds parameterization, use phi_capture instead of p_capture
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
            distributions.update(
                _build_phi_capture_posterior(params, unconstrained, split)
            )
        else:
            distributions.update(
                _build_p_capture_posterior(params, unconstrained, split)
            )

    # -------------------------------------------------------------------------
    # Add mixing weights if mixture model
    # -------------------------------------------------------------------------
    if is_mixture and "mixing_concentrations" in params:
        concentrations = params["mixing_concentrations"]
        distributions["mixing_weights"] = dist.Dirichlet(concentrations)

    return distributions


# =============================================================================
# Helper functions for each parameterization
# =============================================================================


def _build_canonical_posteriors(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
) -> Dict[str, Any]:
    """Build posteriors for canonical (standard) parameterization."""
    distributions = {}

    if unconstrained:
        # Unconstrained: Normal + transform
        distributions["p"] = _build_sigmoid_normal_posterior(
            params, "p", is_scalar=True, split=split
        )
        if low_rank:
            distributions["r"] = _build_low_rank_lognormal_posterior(
                params, "r", is_mixture, split
            )
        else:
            distributions["r"] = _build_exp_normal_posterior(
                params, "r", is_mixture, split
            )
    else:
        # Constrained: Beta and LogNormal
        distributions["p"] = _build_beta_posterior(
            params, "p", is_scalar=True, is_mixture=False, split=split
        )
        if low_rank:
            distributions["r"] = _build_low_rank_lognormal_posterior(
                params, "r", is_mixture, split
            )
        else:
            distributions["r"] = _build_lognormal_posterior(
                params, "r", is_mixture, split
            )

    return distributions


def _build_mean_prob_posteriors(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
) -> Dict[str, Any]:
    """Build posteriors for mean_prob (linked) parameterization."""
    distributions = {}

    if unconstrained:
        distributions["p"] = _build_sigmoid_normal_posterior(
            params, "p", is_scalar=True, split=split
        )
        if low_rank:
            distributions["mu"] = _build_low_rank_lognormal_posterior(
                params, "mu", is_mixture, split
            )
        else:
            distributions["mu"] = _build_exp_normal_posterior(
                params, "mu", is_mixture, split
            )
    else:
        distributions["p"] = _build_beta_posterior(
            params, "p", is_scalar=True, is_mixture=False, split=split
        )
        if low_rank:
            distributions["mu"] = _build_low_rank_lognormal_posterior(
                params, "mu", is_mixture, split
            )
        else:
            distributions["mu"] = _build_lognormal_posterior(
                params, "mu", is_mixture, split
            )

    return distributions


def _build_mean_odds_posteriors(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
) -> Dict[str, Any]:
    """Build posteriors for mean_odds (odds_ratio) parameterization."""
    distributions = {}

    if unconstrained:
        distributions["phi"] = _build_exp_normal_posterior(
            params, "phi", is_mixture=False, split=split, is_scalar=True
        )
        if low_rank:
            distributions["mu"] = _build_low_rank_lognormal_posterior(
                params, "mu", is_mixture, split
            )
        else:
            distributions["mu"] = _build_exp_normal_posterior(
                params, "mu", is_mixture, split
            )
    else:
        distributions["phi"] = _build_betaprime_posterior(
            params, "phi", is_scalar=True, is_mixture=False, split=split
        )
        if low_rank:
            distributions["mu"] = _build_low_rank_lognormal_posterior(
                params, "mu", is_mixture, split
            )
        else:
            distributions["mu"] = _build_lognormal_posterior(
                params, "mu", is_mixture, split
            )

    return distributions


def _build_gate_posterior(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    is_mixture: bool,
    split: bool,
) -> Dict[str, Any]:
    """Build posterior for zero-inflation gate parameter."""
    distributions = {}

    if unconstrained:
        distributions["gate"] = _build_sigmoid_normal_posterior(
            params, "gate", is_scalar=False, split=split
        )
    else:
        distributions["gate"] = _build_beta_posterior(
            params, "gate", is_scalar=False, is_mixture=is_mixture, split=split
        )

    return distributions


def _build_p_capture_posterior(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    split: bool,
) -> Dict[str, Any]:
    """Build posterior for capture probability parameter."""
    distributions = {}

    if unconstrained:
        distributions["p_capture"] = _build_sigmoid_normal_posterior(
            params, "p_capture", is_scalar=False, split=split
        )
    else:
        distributions["p_capture"] = _build_beta_posterior(
            params, "p_capture", is_scalar=False, is_mixture=False, split=split
        )

    return distributions


def _build_phi_capture_posterior(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    split: bool,
) -> Dict[str, Any]:
    """Build posterior for capture odds ratio parameter."""
    distributions = {}

    if unconstrained:
        distributions["phi_capture"] = _build_exp_normal_posterior(
            params,
            "phi_capture",
            is_mixture=False,
            split=split,
            is_scalar=False,
        )
    else:
        distributions["phi_capture"] = _build_betaprime_posterior(
            params,
            "phi_capture",
            is_scalar=False,
            is_mixture=False,
            split=split,
        )

    return distributions


# =============================================================================
# Distribution builders
# =============================================================================


def _build_beta_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    is_scalar: bool,
    is_mixture: bool,
    split: bool,
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """Build Beta posterior from alpha/beta parameters."""
    alpha = params[f"{name}_alpha"]
    beta = params[f"{name}_beta"]

    if is_scalar or (alpha.ndim == 0):
        return dist.Beta(alpha, beta)

    posterior = dist.Beta(alpha, beta)

    if split:
        return _split_distribution(posterior, alpha.shape, is_mixture)
    return posterior


def _build_betaprime_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    is_scalar: bool,
    is_mixture: bool,
    split: bool,
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """Build BetaPrime posterior from alpha/beta parameters."""
    alpha = params[f"{name}_alpha"]
    beta = params[f"{name}_beta"]

    if is_scalar or (alpha.ndim == 0):
        return BetaPrime(alpha, beta)

    posterior = BetaPrime(alpha, beta)

    if split:
        return _split_distribution(posterior, alpha.shape, is_mixture)
    return posterior


def _build_lognormal_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    is_mixture: bool,
    split: bool,
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """Build LogNormal posterior from loc/scale parameters."""
    loc = params[f"{name}_loc"]
    scale = params[f"{name}_scale"]

    posterior = dist.LogNormal(loc, scale)

    if split and loc.ndim > 0:
        return _split_distribution(posterior, loc.shape, is_mixture)
    return posterior


def _build_low_rank_lognormal_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    is_mixture: bool,
    split: bool,
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """Build low-rank LogNormal posterior from MVN parameters."""
    import jax

    loc = params[f"log_{name}_loc"]
    W = params[f"log_{name}_W"]
    raw_diag = params[f"log_{name}_raw_diag"]

    # Ensure diagonal is positive
    D = jax.nn.softplus(raw_diag) + 1e-4

    # Create low-rank MVN in log-space
    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    # Transform to positive via exp
    posterior = dist.TransformedDistribution(
        base, dist.transforms.ExpTransform()
    )

    if split:
        # For low-rank, we can't easily split - return full distribution
        # with a note that individual marginals are not independent
        return posterior

    return posterior


def _build_sigmoid_normal_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    is_scalar: bool,
    split: bool,
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """Build transformed Normal posterior for (0,1) support."""
    loc = params[f"{name}_loc"]
    scale = params[f"{name}_scale"]

    base = dist.Normal(loc, scale)
    posterior = dist.TransformedDistribution(
        base, dist.transforms.SigmoidTransform()
    )

    if is_scalar or (loc.ndim == 0):
        return posterior

    if split:
        return _split_transformed_distribution(posterior, loc.shape)
    return posterior


def _build_exp_normal_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    is_mixture: bool,
    split: bool,
    is_scalar: bool = False,
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """Build transformed Normal posterior for (0,∞) support."""
    loc = params[f"{name}_loc"]
    scale = params[f"{name}_scale"]

    base = dist.Normal(loc, scale)
    posterior = dist.TransformedDistribution(
        base, dist.transforms.ExpTransform()
    )

    if is_scalar or (loc.ndim == 0):
        return posterior

    if split:
        return _split_transformed_distribution(posterior, loc.shape, is_mixture)
    return posterior


# =============================================================================
# Splitting utilities
# =============================================================================


def _split_distribution(
    posterior: dist.Distribution,
    shape: tuple,
    is_mixture: bool,
) -> Union[List[dist.Distribution], List[List[dist.Distribution]]]:
    """Split a batched distribution into individual distributions."""
    if len(shape) == 1:
        # 1D: return list of distributions
        return [
            type(posterior)(*[p[i] for p in _get_dist_params(posterior)])
            for i in range(shape[0])
        ]
    elif len(shape) == 2 and is_mixture:
        # 2D mixture: return nested list [components][genes]
        n_components, n_genes = shape
        return [
            [
                type(posterior)(*[p[c, g] for p in _get_dist_params(posterior)])
                for g in range(n_genes)
            ]
            for c in range(n_components)
        ]
    else:
        # Return as-is for other cases
        return posterior


def _split_transformed_distribution(
    posterior: dist.TransformedDistribution,
    shape: tuple,
    is_mixture: bool = False,
) -> Union[List[dist.Distribution], List[List[dist.Distribution]]]:
    """Split a transformed distribution into individual distributions."""
    base_dist = posterior.base_dist
    transform = posterior.transforms[0] if posterior.transforms else None

    if transform is None:
        return _split_distribution(base_dist, shape, is_mixture)

    if len(shape) == 1:
        return [
            dist.TransformedDistribution(
                dist.Normal(base_dist.loc[i], base_dist.scale[i]),
                transform,
            )
            for i in range(shape[0])
        ]
    elif len(shape) == 2 and is_mixture:
        n_components, n_genes = shape
        return [
            [
                dist.TransformedDistribution(
                    dist.Normal(base_dist.loc[c, g], base_dist.scale[c, g]),
                    transform,
                )
                for g in range(n_genes)
            ]
            for c in range(n_components)
        ]
    else:
        return posterior


def _get_dist_params(d: dist.Distribution) -> List[jnp.ndarray]:
    """Extract parameters from a distribution for splitting."""
    if isinstance(d, dist.Beta):
        return [d.concentration1, d.concentration0]
    elif isinstance(d, dist.LogNormal):
        return [d.loc, d.scale]
    elif isinstance(d, BetaPrime):
        return [d.concentration1, d.concentration0]
    elif isinstance(d, dist.Normal):
        return [d.loc, d.scale]
    else:
        raise ValueError(f"Unsupported distribution type: {type(d)}")
