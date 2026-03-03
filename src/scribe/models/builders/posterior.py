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
    # Constrained: log_r_W, log_r_raw_diag
    # Unconstrained: r_W, r_raw_diag
    low_rank = any(
        key.endswith("_W") and f"{key.replace('_W', '_raw_diag')}" in params
        for key in params.keys()
    )

    hierarchical_p = model_config.hierarchical_p
    hierarchical_gate = model_config.hierarchical_gate

    # -------------------------------------------------------------------------
    # Build distributions based on base parameterization.
    # Horseshoe NCP replaces guide params (e.g. phi_loc -> phi_raw_dataset_loc)
    # so the base builders must skip those parameters to avoid KeyErrors.
    # -------------------------------------------------------------------------
    horseshoe_p = getattr(model_config, "horseshoe_p", False)
    horseshoe_dataset_p = getattr(model_config, "horseshoe_dataset_p", False)
    horseshoe_dataset_mu = getattr(
        model_config, "horseshoe_dataset_mu", False
    )

    skip = set()
    if horseshoe_p or horseshoe_dataset_p:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
            skip.add("phi")
        else:
            skip.add("p")
    if horseshoe_dataset_mu:
        if parameterization in (
            Parameterization.CANONICAL,
            Parameterization.STANDARD,
        ):
            skip.add("r")
        else:
            skip.add("mu")

    if parameterization in (
        Parameterization.CANONICAL,
        Parameterization.STANDARD,
    ):
        distributions.update(
            _build_canonical_posteriors(
                params, unconstrained, is_mixture, low_rank, split, skip
            )
        )
    elif parameterization in (
        Parameterization.MEAN_PROB,
        Parameterization.LINKED,
    ):
        distributions.update(
            _build_mean_prob_posteriors(
                params, unconstrained, is_mixture, low_rank, split, skip
            )
        )
    elif parameterization in (
        Parameterization.MEAN_ODDS,
        Parameterization.ODDS_RATIO,
    ):
        distributions.update(
            _build_mean_odds_posteriors(
                params, unconstrained, is_mixture, low_rank, split, skip
            )
        )
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    # -------------------------------------------------------------------------
    # Override p/phi with hierarchical version (normal or horseshoe)
    # -------------------------------------------------------------------------
    horseshoe_p = getattr(model_config, "horseshoe_p", False)
    if hierarchical_p or horseshoe_p:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
            target_name = "phi"
            loc_name = "log_phi_loc"
            scale_name = "log_phi_scale"
            prefix = "phi"
        else:
            target_name = "p"
            loc_name = "logit_p_loc"
            scale_name = "logit_p_scale"
            prefix = "p"

        if horseshoe_p:
            # Horseshoe: hyper_loc + horseshoe trio + raw z
            distributions.update(
                _build_hyperparameter_posteriors(
                    params, loc_name, loc_name
                )
            )
            distributions.update(
                _build_horseshoe_hyperparameter_posteriors(params, prefix)
            )
            distributions[f"{target_name}_raw"] = _build_normal_posterior(
                params, f"{target_name}_raw", low_rank=low_rank
            )
        else:
            distributions.update(
                _build_hyperparameter_posteriors(
                    params, loc_name, scale_name
                )
            )
            if target_name == "phi":
                distributions[target_name] = _build_exp_normal_posterior(
                    params, target_name, is_mixture, split, is_scalar=False
                )
            else:
                distributions[target_name] = _build_sigmoid_normal_posterior(
                    params, target_name, is_scalar=False, split=split
                )

    # -------------------------------------------------------------------------
    # Dataset-level hierarchical mu/r: hyperparameters + per-dataset override
    # -------------------------------------------------------------------------
    hierarchical_dataset_mu = getattr(
        model_config, "hierarchical_dataset_mu", False
    )
    hierarchical_dataset_p = getattr(
        model_config, "hierarchical_dataset_p", "none"
    )

    horseshoe_dataset_mu = getattr(
        model_config, "horseshoe_dataset_mu", False
    )
    if hierarchical_dataset_mu or horseshoe_dataset_mu:
        if parameterization in (
            Parameterization.CANONICAL,
            Parameterization.STANDARD,
        ):
            hyper_loc = "log_r_dataset_loc"
            hyper_scale = "log_r_dataset_scale"
            target = "r"
            hs_prefix = "r_dataset"
        else:
            hyper_loc = "log_mu_dataset_loc"
            hyper_scale = "log_mu_dataset_scale"
            target = "mu"
            hs_prefix = "mu_dataset"

        if horseshoe_dataset_mu:
            # Horseshoe: hyper_loc posterior + horseshoe trio + raw z
            distributions.update(
                _build_hyperparameter_posteriors(params, hyper_loc, hyper_loc)
            )
            distributions.update(
                _build_horseshoe_hyperparameter_posteriors(params, hs_prefix)
            )
            distributions[f"{target}_raw"] = _build_normal_posterior(
                params, f"{target}_raw", low_rank=low_rank
            )
        else:
            distributions.update(
                _build_hyperparameter_posteriors(
                    params, hyper_loc, hyper_scale
                )
            )
            if low_rank:
                distributions[target] = _build_low_rank_exp_normal_posterior(
                    params, target, is_mixture, split
                )
            else:
                distributions[target] = _build_exp_normal_posterior(
                    params, target, is_mixture, split
                )

    # -------------------------------------------------------------------------
    # Dataset-level hierarchical p/phi: hyperparameters + per-dataset override
    # -------------------------------------------------------------------------
    horseshoe_dataset_p = getattr(model_config, "horseshoe_dataset_p", False)
    if hierarchical_dataset_p != "none" or horseshoe_dataset_p:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
            hyper_loc = "log_phi_dataset_loc"
            hyper_scale = "log_phi_dataset_scale"
            target = "phi"
            hs_prefix = "phi_dataset"
            raw_name = "phi_raw_dataset"
        else:
            hyper_loc = "logit_p_dataset_loc"
            hyper_scale = "logit_p_dataset_scale"
            target = "p"
            hs_prefix = "p_dataset"
            raw_name = "p_raw_dataset"

        if horseshoe_dataset_p:
            distributions.update(
                _build_hyperparameter_posteriors(params, hyper_loc, hyper_loc)
            )
            distributions.update(
                _build_horseshoe_hyperparameter_posteriors(params, hs_prefix)
            )
            distributions[raw_name] = _build_normal_posterior(
                params, raw_name, low_rank=low_rank
            )
        else:
            distributions.update(
                _build_hyperparameter_posteriors(
                    params, hyper_loc, hyper_scale
                )
            )
            if target == "phi":
                distributions[target] = _build_exp_normal_posterior(
                    params, target, is_mixture, split, is_scalar=False
                )
            else:
                distributions[target] = _build_sigmoid_normal_posterior(
                    params, target, is_scalar=False, split=split
                )

    # -------------------------------------------------------------------------
    # Dataset-level hierarchical gate
    # -------------------------------------------------------------------------
    hierarchical_dataset_gate = getattr(
        model_config, "hierarchical_dataset_gate", False
    )
    horseshoe_dataset_gate = getattr(
        model_config, "horseshoe_dataset_gate", False
    )
    if hierarchical_dataset_gate or horseshoe_dataset_gate:
        if horseshoe_dataset_gate:
            distributions.update(
                _build_hyperparameter_posteriors(
                    params,
                    "logit_gate_dataset_loc",
                    "logit_gate_dataset_loc",
                )
            )
            distributions.update(
                _build_horseshoe_hyperparameter_posteriors(
                    params, "gate_dataset"
                )
            )
            distributions["gate_raw_dataset"] = _build_normal_posterior(
                params, "gate_raw_dataset", low_rank=low_rank
            )
        else:
            distributions.update(
                _build_hyperparameter_posteriors(
                    params,
                    "logit_gate_dataset_loc",
                    "logit_gate_dataset_scale",
                )
            )
            distributions["gate"] = _build_sigmoid_normal_posterior(
                params, "gate", is_scalar=False, split=split
            )

    # -------------------------------------------------------------------------
    # Add zero-inflation gate if applicable
    # -------------------------------------------------------------------------
    horseshoe_gate = getattr(model_config, "horseshoe_gate", False)
    if is_zero_inflated and not (
        hierarchical_dataset_gate or horseshoe_dataset_gate
    ):
        if horseshoe_gate:
            # Horseshoe gene-level gate: hyper_loc + horseshoe trio + raw z
            distributions.update(
                _build_hyperparameter_posteriors(
                    params, "logit_gate_loc", "logit_gate_loc"
                )
            )
            distributions.update(
                _build_horseshoe_hyperparameter_posteriors(params, "gate")
            )
            distributions["gate_raw"] = _build_normal_posterior(
                params, "gate_raw", low_rank=low_rank
            )
        elif hierarchical_gate:
            distributions.update(
                _build_hyperparameter_posteriors(
                    params, "logit_gate_loc", "logit_gate_scale"
                )
            )
            distributions.update(
                _build_gate_posterior(
                    params, unconstrained, is_mixture, split
                )
            )
        else:
            distributions.update(
                _build_gate_posterior(
                    params, unconstrained, is_mixture, split
                )
            )

    # -------------------------------------------------------------------------
    # Add capture probability if applicable
    # -------------------------------------------------------------------------
    capture_prior = getattr(model_config, "capture_prior", "default")
    if uses_vcp:
        if capture_prior in ("biology_informed", "data_driven"):
            # Biology-informed capture: posterior is on eta_capture
            distributions.update(
                _build_biology_informed_capture_posterior(
                    params, model_config, split
                )
            )
        elif parameterization in (
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
    skip: Optional[set] = None,
) -> Dict[str, Any]:
    """Build posteriors for canonical (standard) parameterization."""
    distributions = {}
    skip = skip or set()

    if unconstrained:
        if "p" not in skip:
            distributions["p"] = _build_sigmoid_normal_posterior(
                params, "p", is_scalar=True, split=split
            )
        if "r" not in skip:
            if low_rank:
                distributions["r"] = _build_low_rank_exp_normal_posterior(
                    params, "r", is_mixture, split
                )
            else:
                distributions["r"] = _build_exp_normal_posterior(
                    params, "r", is_mixture, split
                )
    else:
        if "p" not in skip:
            distributions["p"] = _build_beta_posterior(
                params, "p", is_scalar=True, is_mixture=False, split=split
            )
        if "r" not in skip:
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
    skip: Optional[set] = None,
) -> Dict[str, Any]:
    """Build posteriors for mean_prob (linked) parameterization."""
    distributions = {}
    skip = skip or set()

    if unconstrained:
        if "p" not in skip:
            distributions["p"] = _build_sigmoid_normal_posterior(
                params, "p", is_scalar=True, split=split
            )
        if "mu" not in skip:
            if low_rank:
                distributions["mu"] = _build_low_rank_exp_normal_posterior(
                    params, "mu", is_mixture, split
                )
            else:
                distributions["mu"] = _build_exp_normal_posterior(
                    params, "mu", is_mixture, split
                )
    else:
        if "p" not in skip:
            distributions["p"] = _build_beta_posterior(
                params, "p", is_scalar=True, is_mixture=False, split=split
            )
        if "mu" not in skip:
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
    skip: Optional[set] = None,
) -> Dict[str, Any]:
    """Build posteriors for mean_odds (odds_ratio) parameterization."""
    distributions = {}
    skip = skip or set()

    if unconstrained:
        if "phi" not in skip:
            distributions["phi"] = _build_exp_normal_posterior(
                params, "phi", is_mixture=False, split=split, is_scalar=True
            )
        if "mu" not in skip:
            if low_rank:
                distributions["mu"] = _build_low_rank_exp_normal_posterior(
                    params, "mu", is_mixture, split
                )
            else:
                distributions["mu"] = _build_exp_normal_posterior(
                    params, "mu", is_mixture, split
                )
    else:
        if "phi" not in skip:
            distributions["phi"] = _build_betaprime_posterior(
                params, "phi", is_scalar=True, is_mixture=False, split=split
            )
        if "mu" not in skip:
            if low_rank:
                distributions["mu"] = _build_low_rank_lognormal_posterior(
                    params, "mu", is_mixture, split
                )
            else:
                distributions["mu"] = _build_lognormal_posterior(
                    params, "mu", is_mixture, split
                )

    return distributions


# =============================================================================
# Hierarchical parameterization posteriors
# =============================================================================


def _build_hyperparameter_posteriors(
    params: Dict[str, jnp.ndarray],
    loc_name: str,
    scale_name: str,
) -> Dict[str, Any]:
    """Build posteriors for hierarchical hyperparameters.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Guide parameters.
    loc_name : str
        Name of the location hyperparameter (e.g. ``"logit_p_loc"``).
    scale_name : str
        Name of the scale hyperparameter (e.g. ``"logit_p_scale"``).
        The posterior is Softplus-transformed to enforce positivity.

    Returns
    -------
    Dict[str, Any]
        Posterior distributions for the two hyperparameters.
    """
    distributions = {}

    # Location hyperparameter: unconstrained Normal
    loc_loc = params[f"{loc_name}_loc"]
    loc_scale = params[f"{loc_name}_scale"]
    distributions[loc_name] = dist.Normal(loc_loc, loc_scale)

    # Scale hyperparameter: Softplus-transformed Normal (positive)
    scale_loc = params[f"{scale_name}_loc"]
    scale_scale = params[f"{scale_name}_scale"]
    distributions[scale_name] = dist.TransformedDistribution(
        dist.Normal(scale_loc, scale_scale),
        dist.transforms.SoftplusTransform(),
    )

    return distributions


def _build_horseshoe_hyperparameter_posteriors(
    params: Dict[str, jnp.ndarray],
    prefix: str,
) -> Dict[str, Any]:
    """Build LogNormal posteriors for horseshoe hyperparameters.

    The horseshoe trio (tau, lambda, c_sq) all have LogNormal variational
    posteriors since they are positive-valued.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Guide parameters.  Must contain ``tau_{prefix}_loc``,
        ``tau_{prefix}_scale``, etc.
    prefix : str
        Naming prefix (e.g. ``"p"``, ``"mu_dataset"``).

    Returns
    -------
    Dict[str, Any]
        Posterior distributions for tau, lambda, and c_sq.
    """
    distributions = {}

    for role in ("tau", "lambda", "c_sq"):
        name = f"{role}_{prefix}"
        loc = params[f"{name}_loc"]
        scale = params[f"{name}_scale"]
        distributions[name] = dist.LogNormal(loc, scale)

    return distributions


def _build_normal_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    low_rank: bool = False,
) -> Union[dist.Distribution, Dict]:
    """Build a Normal posterior for NCP raw z variables.

    Handles both mean-field (Normal with loc/scale) and low-rank
    (LowRankMultivariateNormal with loc/W/raw_diag) guide formats.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Guide parameters.
    name : str
        Parameter name (e.g. ``"p_raw"``, ``"mu_raw"``).
    low_rank : bool
        If True, build a LowRankMultivariateNormal from ``{name}_W``
        and ``{name}_raw_diag`` params.

    Returns
    -------
    Union[dist.Normal, Dict]
        Normal posterior (mean-field) or dict with ``base``/``transform``
        keys (low-rank, no transform applied since z is unconstrained).
    """
    loc = params[f"{name}_loc"]

    if low_rank and f"{name}_W" in params:
        import jax

        W = params[f"{name}_W"]
        raw_diag = params[f"{name}_raw_diag"]
        D = jax.nn.softplus(raw_diag) + 1e-4
        base = dist.LowRankMultivariateNormal(
            loc=loc, cov_factor=W, cov_diag=D
        )
        return {"base": base, "transform": dist.transforms.IdentityTransform()}

    scale = params[f"{name}_scale"]
    return dist.Normal(loc, scale)


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


def _build_biology_informed_capture_posterior(
    params: Dict[str, jnp.ndarray],
    model_config: "ModelConfig",
    split: bool,
) -> Dict[str, Any]:
    """Build posterior for biology-informed capture parameter.

    The variational posterior is on ``eta_capture`` (the unconstrained
    log-ratio log(M_c / L_c)).  The capture parameter is a deterministic
    transformation of eta.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Optimized variational parameters. Expected keys:
        ``eta_capture_loc``, ``eta_capture_scale``, and optionally
        ``mu_eta_loc``, ``mu_eta_scale`` (data-driven mode).
    model_config : ModelConfig
        Model configuration.
    split : bool
        If True, split per-cell distributions.

    Returns
    -------
    Dict[str, Any]
        Posterior distributions for ``eta_capture`` (and ``mu_eta`` if
        data-driven).
    """
    distributions: Dict[str, Any] = {}

    # Shared mu_eta for data-driven mode
    if "mu_eta_loc" in params:
        distributions["mu_eta"] = dist.Normal(
            params["mu_eta_loc"], params["mu_eta_scale"]
        )

    # Per-cell eta_capture posterior
    if "eta_capture_loc" in params:
        loc = params["eta_capture_loc"]
        scale = params["eta_capture_scale"]
        eta_dist = dist.Normal(loc, scale)
        if split:
            distributions["eta_capture"] = [
                dist.Normal(loc[i], scale[i]) for i in range(loc.shape[0])
            ]
        else:
            distributions["eta_capture"] = eta_dist

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


def _build_low_rank_exp_normal_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    is_mixture: bool,
    split: bool,
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """
    Build low-rank TransformedDistribution posterior for unconstrained models.

    For unconstrained models with low-rank guides, the parameters are:
        - {name}_loc: location parameter
        - {name}_W: covariance factor
        - {name}_raw_diag: raw diagonal (will be softplus'd)

    The guide uses TransformedDistribution(LowRankMultivariateNormal,
    transform), so we reconstruct the same structure here.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Dictionary of variational parameters.
    name : str
        Parameter name (e.g., "r", "mu").
    is_mixture : bool
        Whether this is a mixture model.
    split : bool
        Whether to split into per-gene distributions (not supported for
        low-rank).

    Returns
    -------
    Union[dist.Distribution, List[dist.Distribution]]
        Low-rank posterior distribution wrapped with appropriate transform.
    """
    import jax

    loc = params[f"{name}_loc"]
    W = params[f"{name}_W"]
    raw_diag = params[f"{name}_raw_diag"]

    # Ensure diagonal is positive
    D = jax.nn.softplus(raw_diag) + 1e-4

    # Create low-rank MVN in unconstrained space
    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    # Wrap with transform (ExpTransform for positive, SigmoidTransform for (0,1))
    # Determine transform from parameter name or use ExpTransform as default
    if name in ["p", "gate", "p_capture"]:
        transform = dist.transforms.SigmoidTransform()
    else:
        transform = dist.transforms.ExpTransform()

    # Return as dict structure to match get_map expectations
    # get_map expects {"base": base_dist, "transform": transform} for low-rank guides
    if split:
        # For low-rank, we can't easily split - return full distribution
        return {"base": base, "transform": transform}

    return {"base": base, "transform": transform}


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
