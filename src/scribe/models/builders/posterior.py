"""
Posterior distribution extraction for the composable builder system.

Reconstructs numpyro posterior distributions from the optimized variational
parameters produced by SVI inference.  The public entry point is
:func:`get_posterior_distributions`.

Architecture
------------
The module is organised into four layers, top-to-bottom:

1. **Orchestrator** — ``get_posterior_distributions``
   A short, linear pipeline that calls each ``_apply_*`` pass in a fixed
   order.  Each pass may *add* or *override* keys in the shared
   ``distributions`` dict.  The ordering matters: later passes (e.g.
   dataset hierarchy) intentionally replace keys set by earlier ones (e.g.
   base parameterization) when a hierarchical or horseshoe prior is active.

2. **Pipeline passes** — ``_apply_base_parameterization``, …,
   ``_apply_joint_aggregation``
   One function per concern (base params, gene-level hierarchy, three
   dataset-level hierarchies, zero-inflation gate, capture probability,
   mixture weights, joint aggregation).  Each pass is a no-op when its
   config flags are off, so the pipeline is safe for any ``ModelConfig``.

3. **Parameterization builders** — ``_build_canonical_posteriors``,
   ``_build_mean_prob_posteriors``, ``_build_mean_odds_posteriors``
   Mid-level helpers that translate a parameterization family into the
   correct set of distribution constructors.

4. **Leaf builders** — ``_build_beta_posterior``, ``_build_lognormal_posterior``,
   ``_build_low_rank_exp_normal_posterior``, etc.
   Low-level functions that construct a single numpyro distribution object
   from raw variational parameters (``_loc``, ``_scale``, ``_W``,
   ``_raw_diag``, ``_alpha``, ``_beta``).

Joint-prefix pattern
~~~~~~~~~~~~~~~~~~~~
When parameters are modelled jointly via ``JointLowRankGuide``, their
variational params live under a ``joint_{group}_{name}_*`` key namespace
instead of the usual ``{name}_loc`` / ``{name}_scale``.  Every pass that
builds a distribution calls ``_find_joint_prefix(params, name)`` first; if
a joint prefix is found, ``_build_joint_low_rank_posterior`` is used to
reconstruct the marginal from the joint block.

Output contract
~~~~~~~~~~~~~~~
The returned dict maps logical parameter names to either:

- A numpyro ``Distribution`` (constrained guides, e.g. ``dist.Beta``).
- A ``{"base": Distribution, "transform": Transform}`` dict (unconstrained
  / low-rank guides — consumed by ``get_map`` and sampling code).
- A ``"joint:{group}"`` key with the full stacked joint distribution.

Functions
---------
get_posterior_distributions
    Extract posterior distributions from optimized guide parameters.

Examples
--------
>>> from scribe.models.builders.posterior import get_posterior_distributions
>>> from scribe.models.config import ModelConfig
>>>
>>> posteriors = get_posterior_distributions(
...     params=svi_results.params,
...     model_config=model_config,
... )
>>> p_dist = posteriors["p"]
>>> r_dist = posteriors["r"]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import jax.numpy as jnp
import numpyro.distributions as dist

# Import Parameterization and HierarchicalPriorType enums directly to avoid
# circular import
from ..config.enums import HierarchicalPriorType, Parameterization
from scribe.stats.distributions import BetaPrime

if TYPE_CHECKING:
    from ..config import ModelConfig


# =============================================================================
# Joint low-rank helpers
# =============================================================================
#
# When ``joint_params`` is configured, the JointLowRankGuide stores
# variational parameters under ``joint_{group}_{name}_loc/W/raw_diag``
# instead of the usual ``{name}_loc/scale``.  The three helpers below
# resolve these keys and reconstruct either per-parameter marginals or the
# full stacked joint distribution.
# =============================================================================


def _find_joint_prefix(
    params: Dict[str, jnp.ndarray],
    name: str,
) -> Optional[str]:
    """Find the ``joint_{group}_{name}`` key prefix for a parameter.

    Scans ``params`` for a key matching ``joint_*_{name}_loc``.  Returns
    the prefix (everything before ``_loc``) if found, else ``None``.

    Parameters
    ----------
    params : dict
        Variational parameter dictionary.
    name : str
        Logical parameter name (e.g. ``"mu"``, ``"phi"``).

    Returns
    -------
    str or None
        The key prefix (e.g. ``"joint_joint_mu"``) or ``None``.
    """
    suffix = f"_{name}_loc"
    for key in params:
        if key.startswith("joint_") and key.endswith(suffix):
            return key[: -len("_loc")]
    return None


def _build_joint_low_rank_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    prefix: str,
    split: bool,
) -> Dict[str, Any]:
    """Build a low-rank marginal posterior from joint guide params.

    Uses the stored ``{prefix}_loc``, ``{prefix}_W``,
    ``{prefix}_raw_diag`` to reconstruct the per-parameter marginal
    distribution + transform, identical in structure to the output
    of ``_build_low_rank_exp_normal_posterior``.

    For gene-specific parameters (G > 1), returns a
    ``LowRankMultivariateNormal``.  For scalar parameters that were
    expanded to G=1 during joint guide setup, collapses back to a
    ``Normal`` with variance ``sum(W[..., 0, :]**2) + D[..., 0]``.

    Parameters
    ----------
    params : dict
        Variational parameter dictionary.
    name : str
        Logical parameter name (determines transform: Exp vs Sigmoid).
    prefix : str
        Full key prefix (e.g. ``"joint_joint_mu"``).
    split : bool
        Whether to split (not supported for low-rank; ignored).

    Returns
    -------
    dict
        ``{"base": distribution, "transform": transform}``
    """
    import jax

    loc = params[f"{prefix}_loc"]
    W = params[f"{prefix}_W"]
    raw_diag = params[f"{prefix}_raw_diag"]

    D = jax.nn.softplus(raw_diag) + 1e-4

    # Scalar params expanded to G=1: collapse to Normal for correct shape
    if loc.shape[-1] == 1:
        scalar_var = jnp.sum(W[..., 0, :] ** 2, axis=-1) + D[..., 0]
        base = dist.Normal(loc[..., 0], jnp.sqrt(scalar_var))
    else:
        base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    if name in ("p", "gate", "p_capture"):
        transform = dist.transforms.SigmoidTransform()
    else:
        transform = dist.transforms.ExpTransform()

    return {"base": base, "transform": transform}


def _build_joint_full_distribution(
    params: Dict[str, jnp.ndarray],
    joint_params: List[str],
    group: str = "joint",
) -> Optional[Dict[str, Any]]:
    """Build the full joint ``LowRankMVN`` by stacking per-param blocks.

    Concatenates ``loc``, ``W``, ``D`` from each parameter in the group
    into a single distribution.  Supports heterogeneous per-parameter
    dimensions (e.g., a scalar parameter with G=1 alongside gene-specific
    parameters with G=n_genes).  The ``param_sizes`` list records each
    block's trailing dimension so consumers can split the stacked vector.

    Parameters
    ----------
    params : dict
        Variational parameter dictionary.
    joint_params : list of str
        Ordered parameter names in the group.
    group : str
        Group name used in the key prefix.

    Returns
    -------
    dict or None
        ``{"base": LowRankMVN, "param_names": [...], "param_sizes": [...]}``
        or ``None`` if the params are not found.
    """
    import jax

    locs, Ws, Ds = [], [], []
    sizes = []
    for pname in joint_params:
        prefix = f"joint_{group}_{pname}"
        loc_key = f"{prefix}_loc"
        if loc_key not in params:
            return None
        loc_i = params[loc_key]
        W_i = params[f"{prefix}_W"]
        raw_diag_i = params[f"{prefix}_raw_diag"]
        D_i = jax.nn.softplus(raw_diag_i) + 1e-4

        locs.append(loc_i)
        Ws.append(W_i)
        Ds.append(D_i)
        sizes.append(loc_i.shape[-1])

    # Stack into single vectors / matrices
    full_loc = jnp.concatenate(locs, axis=-1)
    full_W = jnp.concatenate(Ws, axis=-2)
    full_D = jnp.concatenate(Ds, axis=-1)

    base = dist.LowRankMultivariateNormal(
        loc=full_loc, cov_factor=full_W, cov_diag=full_D
    )
    return {
        "base": base,
        "param_names": list(joint_params),
        "param_sizes": sizes,
    }


# =============================================================================
# Pipeline passes
# =============================================================================
#
# Each ``_apply_*`` function implements one concern of the posterior
# extraction pipeline.  They share a common pattern:
#
#   1. Check config flags — return immediately when the concern is inactive.
#   2. Resolve param names for the active parameterization family.
#   3. Check for a joint-prefix (``_find_joint_prefix``) before falling back
#      to individual ``_loc/_scale`` params.
#   4. Add or *override* entries in the shared ``distributions`` dict.
#
# Override semantics: later passes intentionally replace keys set by the
# base pass.  For example, when ``hierarchical_dataset_p`` is active,
# ``_apply_dataset_hierarchy_p`` replaces the ``"phi"`` entry originally
# written by ``_apply_base_parameterization``.
# =============================================================================


def _build_base_skip_set(
    model_config: ModelConfig, parameterization: Parameterization
) -> set[str]:
    """Determine which base params to skip because NCP (horseshoe/NEG) replaces them.

    When a horseshoe or NEG NCP prior is active for a parameter, the base guide
    site (e.g. ``phi_loc/phi_scale``) does not exist — the NCP prior creates
    ``phi_raw_loc/phi_raw_scale`` (the z variable) plus its own hyperparameters
    instead.  The base builder must skip these names to avoid ``KeyError``s on
    missing params.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration carrying horseshoe/NEG prior flags.
    parameterization : Parameterization
        Active parameterization enum.

    Returns
    -------
    set of str
        Parameter names to skip in base extraction.
    """
    skip: set[str] = set()
    horseshoe_p = getattr(model_config, "horseshoe_p", False)
    horseshoe_dataset_p = getattr(model_config, "horseshoe_dataset_p", False)
    horseshoe_dataset_mu = getattr(model_config, "horseshoe_dataset_mu", False)
    neg_p = model_config.p_prior == HierarchicalPriorType.NEG
    neg_dataset_p = model_config.p_dataset_prior == HierarchicalPriorType.NEG
    neg_dataset_mu = model_config.mu_dataset_prior == HierarchicalPriorType.NEG

    if horseshoe_p or horseshoe_dataset_p or neg_p or neg_dataset_p:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
            skip.add("phi")
        else:
            skip.add("p")
    if horseshoe_dataset_mu or neg_dataset_mu:
        if parameterization in (
            Parameterization.CANONICAL,
            Parameterization.STANDARD,
        ):
            skip.add("r")
        else:
            skip.add("mu")
    return skip


def _apply_base_parameterization(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    parameterization: Parameterization,
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    skip: set[str],
) -> None:
    """Pass 1: build posteriors for the core model parameters.

    Dispatches to the parameterization-specific builder (canonical,
    mean_prob, or mean_odds) which populates ``distributions`` with the
    primary model parameters (``r``/``p``, ``mu``/``p``, or ``mu``/``phi``).
    Parameters in *skip* are omitted because a horseshoe pass will handle
    them later.
    """
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


def _apply_gene_level_hierarchy(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    parameterization: Parameterization,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
) -> None:
    """Pass 2: override p/phi with a gene-level hierarchical prior.

    Active when ``hierarchical_p=True``, ``horseshoe_p=True``, or
    ``p_prior==NEG``.  Adds hyperparameter posteriors (loc/scale of the
    population prior) and *replaces* the base ``"p"`` or ``"phi"`` entry
    with the gene-level distribution.

    Horseshoe variant: adds the horseshoe trio (tau, lambda, c_sq) and
    the NCP raw z variable instead of the constrained parameter.

    NEG variant: adds psi and zeta LogNormal posteriors and the NCP raw z
    variable instead of the constrained parameter.
    """
    hierarchical_p = model_config.hierarchical_p
    horseshoe_p = getattr(model_config, "horseshoe_p", False)
    neg_p = model_config.p_prior == HierarchicalPriorType.NEG
    if not (hierarchical_p or horseshoe_p or neg_p):
        return

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
        # Horseshoe NCP: hyper_loc + {tau, lambda, c_sq} + raw z variable.
        # The constrained param is reconstructed in _reconstruct_horseshoe_maps.
        distributions.update(
            _build_hyperparameter_posteriors(params, loc_name, loc_name)
        )
        distributions.update(
            _build_horseshoe_hyperparameter_posteriors(params, prefix)
        )
        distributions[f"{target_name}_raw"] = _build_normal_posterior(
            params, f"{target_name}_raw", low_rank=low_rank
        )
        return

    if neg_p:
        # NEG NCP: hyper_loc + {psi, zeta} + raw z variable.
        # The constrained param is reconstructed in _reconstruct_neg_maps.
        distributions.update(
            _build_hyperparameter_posteriors(params, loc_name, loc_name)
        )
        distributions.update(
            _build_neg_hyperparameter_posteriors(params, prefix)
        )
        distributions[f"{target_name}_raw"] = _build_normal_posterior(
            params, f"{target_name}_raw", low_rank=low_rank
        )
        return

    # Non-horseshoe/NEG hierarchical: population hyper-priors + constrained param.
    distributions.update(
        _build_hyperparameter_posteriors(params, loc_name, scale_name)
    )
    # If this param lives inside a JointLowRankGuide, extract its marginal
    # from the joint block instead of building from individual _loc/_scale.
    jp = _find_joint_prefix(params, target_name)
    if jp:
        distributions[target_name] = _build_joint_low_rank_posterior(
            params, target_name, jp, split
        )
    elif target_name == "phi":
        distributions[target_name] = _build_exp_normal_posterior(
            params, target_name, is_mixture, split, is_scalar=False
        )
    else:
        distributions[target_name] = _build_sigmoid_normal_posterior(
            params, target_name, is_scalar=False, split=split
        )


def _apply_dataset_hierarchy_mu(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    parameterization: Parameterization,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
) -> None:
    """Pass 3: override mu/r with a dataset-level hierarchical prior.

    Active when ``hierarchical_dataset_mu=True``,
    ``horseshoe_dataset_mu=True``, or ``mu_dataset_prior==NEG``.  Adds
    dataset-level hyperparameter posteriors and *replaces* the base
    ``"mu"`` (or ``"r"``) entry.  The resulting MAP has shape
    ``(K, D, G)`` instead of ``(K, G)``.
    """
    hierarchical_dataset_mu = getattr(
        model_config, "hierarchical_dataset_mu", False
    )
    horseshoe_dataset_mu = getattr(model_config, "horseshoe_dataset_mu", False)
    neg_dataset_mu = model_config.mu_dataset_prior == HierarchicalPriorType.NEG
    if not (hierarchical_dataset_mu or horseshoe_dataset_mu or neg_dataset_mu):
        return

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
        raw_name = f"{target}_raw"
        # When joint_params includes the target, the horseshoe raw z lives
        # inside the joint block under either "{target}_raw" or "{target}".
        # Try both to find the joint prefix.
        jp = _find_joint_prefix(params, raw_name) or _find_joint_prefix(
            params, target
        )
        if jp:
            # Joint path: hyper-priors may still have individual params
            # (they are not folded into the joint guide).
            if f"{hyper_loc}_loc" in params:
                distributions.update(
                    _build_hyperparameter_posteriors(
                        params, hyper_loc, hyper_loc
                    )
                )
            if f"tau_{hs_prefix}_loc" in params:
                distributions.update(
                    _build_horseshoe_hyperparameter_posteriors(
                        params, hs_prefix
                    )
                )
            distributions[target] = _build_joint_low_rank_posterior(
                params, target, jp, split
            )
        else:
            distributions.update(
                _build_hyperparameter_posteriors(params, hyper_loc, hyper_loc)
            )
            distributions.update(
                _build_horseshoe_hyperparameter_posteriors(params, hs_prefix)
            )
            distributions[raw_name] = _build_normal_posterior(
                params, raw_name, low_rank=low_rank
            )
        return

    if neg_dataset_mu:
        raw_name = f"{target}_raw"
        jp = _find_joint_prefix(params, raw_name) or _find_joint_prefix(
            params, target
        )
        if jp:
            if f"{hyper_loc}_loc" in params:
                distributions.update(
                    _build_hyperparameter_posteriors(
                        params, hyper_loc, hyper_loc
                    )
                )
            if f"psi_{hs_prefix}_loc" in params:
                distributions.update(
                    _build_neg_hyperparameter_posteriors(params, hs_prefix)
                )
            distributions[target] = _build_joint_low_rank_posterior(
                params, target, jp, split
            )
        else:
            distributions.update(
                _build_hyperparameter_posteriors(params, hyper_loc, hyper_loc)
            )
            distributions.update(
                _build_neg_hyperparameter_posteriors(params, hs_prefix)
            )
            distributions[raw_name] = _build_normal_posterior(
                params, raw_name, low_rank=low_rank
            )
        return

    distributions.update(
        _build_hyperparameter_posteriors(params, hyper_loc, hyper_scale)
    )
    jp = _find_joint_prefix(params, target)
    if jp:
        distributions[target] = _build_joint_low_rank_posterior(
            params, target, jp, split
        )
    elif low_rank:
        distributions[target] = _build_low_rank_exp_normal_posterior(
            params, target, is_mixture, split
        )
    else:
        distributions[target] = _build_exp_normal_posterior(
            params, target, is_mixture, split
        )


def _apply_dataset_hierarchy_p(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    parameterization: Parameterization,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
) -> None:
    """Pass 4: override p/phi with a dataset-level hierarchical prior.

    Active when ``hierarchical_dataset_p != "none"``,
    ``horseshoe_dataset_p=True``, or ``p_dataset_prior==NEG``.  Replaces
    the ``"phi"`` (or ``"p"``) entry.  With
    ``hierarchical_dataset_p="gene_specific"`` the MAP gains shape
    ``(K, D, G)``; with ``"scalar"`` it becomes ``(K, D)``.
    """
    hierarchical_dataset_p = getattr(
        model_config, "hierarchical_dataset_p", "none"
    )
    horseshoe_dataset_p = getattr(model_config, "horseshoe_dataset_p", False)
    neg_dataset_p = model_config.p_dataset_prior == HierarchicalPriorType.NEG
    if not (
        hierarchical_dataset_p != "none" or horseshoe_dataset_p or neg_dataset_p
    ):
        return

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
        jp = _find_joint_prefix(params, raw_name) or _find_joint_prefix(
            params, target
        )
        if jp:
            if f"{hyper_loc}_loc" in params:
                distributions.update(
                    _build_hyperparameter_posteriors(
                        params, hyper_loc, hyper_loc
                    )
                )
            if f"tau_{hs_prefix}_loc" in params:
                distributions.update(
                    _build_horseshoe_hyperparameter_posteriors(
                        params, hs_prefix
                    )
                )
            distributions[target] = _build_joint_low_rank_posterior(
                params, target, jp, split
            )
        else:
            distributions.update(
                _build_hyperparameter_posteriors(params, hyper_loc, hyper_loc)
            )
            distributions.update(
                _build_horseshoe_hyperparameter_posteriors(params, hs_prefix)
            )
            distributions[raw_name] = _build_normal_posterior(
                params, raw_name, low_rank=low_rank
            )
        return

    if neg_dataset_p:
        jp = _find_joint_prefix(params, raw_name) or _find_joint_prefix(
            params, target
        )
        if jp:
            if f"{hyper_loc}_loc" in params:
                distributions.update(
                    _build_hyperparameter_posteriors(
                        params, hyper_loc, hyper_loc
                    )
                )
            if f"psi_{hs_prefix}_loc" in params:
                distributions.update(
                    _build_neg_hyperparameter_posteriors(params, hs_prefix)
                )
            distributions[target] = _build_joint_low_rank_posterior(
                params, target, jp, split
            )
        else:
            distributions.update(
                _build_hyperparameter_posteriors(params, hyper_loc, hyper_loc)
            )
            distributions.update(
                _build_neg_hyperparameter_posteriors(params, hs_prefix)
            )
            distributions[raw_name] = _build_normal_posterior(
                params, raw_name, low_rank=low_rank
            )
        return

    distributions.update(
        _build_hyperparameter_posteriors(params, hyper_loc, hyper_scale)
    )
    jp = _find_joint_prefix(params, target)
    if jp:
        distributions[target] = _build_joint_low_rank_posterior(
            params, target, jp, split
        )
    elif target == "phi":
        distributions[target] = _build_exp_normal_posterior(
            params, target, is_mixture, split, is_scalar=False
        )
    else:
        distributions[target] = _build_sigmoid_normal_posterior(
            params, target, is_scalar=False, split=split
        )


def _apply_dataset_hierarchy_gate(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    low_rank: bool,
    split: bool,
) -> None:
    """Pass 5: override gate with a dataset-level hierarchical prior.

    Active when ``hierarchical_dataset_gate=True``,
    ``horseshoe_dataset_gate=True``, or ``gate_dataset_prior==NEG``.
    Replaces the ``"gate"`` entry (or adds ``"gate_raw_dataset"`` for the
    horseshoe/NEG NCP z variable when the parameter is *not* in a joint
    guide group).

    When ``gate`` *is* in ``joint_params``, the horseshoe/NEG raw z lives
    inside the joint block, so we check both ``"gate_raw_dataset"`` and
    ``"gate"`` for a joint prefix.
    """
    hierarchical_dataset_gate = getattr(
        model_config, "hierarchical_dataset_gate", False
    )
    horseshoe_dataset_gate = getattr(
        model_config, "horseshoe_dataset_gate", False
    )
    neg_dataset_gate = (
        model_config.gate_dataset_prior == HierarchicalPriorType.NEG
    )
    if not (
        hierarchical_dataset_gate or horseshoe_dataset_gate or neg_dataset_gate
    ):
        return

    if horseshoe_dataset_gate:
        jp = _find_joint_prefix(
            params, "gate_raw_dataset"
        ) or _find_joint_prefix(params, "gate")
        if jp:
            if "logit_gate_dataset_loc_loc" in params:
                distributions.update(
                    _build_hyperparameter_posteriors(
                        params,
                        "logit_gate_dataset_loc",
                        "logit_gate_dataset_loc",
                    )
                )
            if "tau_gate_dataset_loc" in params:
                distributions.update(
                    _build_horseshoe_hyperparameter_posteriors(
                        params, "gate_dataset"
                    )
                )
            distributions["gate"] = _build_joint_low_rank_posterior(
                params, "gate", jp, split
            )
        else:
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
        return

    if neg_dataset_gate:
        jp = _find_joint_prefix(
            params, "gate_raw_dataset"
        ) or _find_joint_prefix(params, "gate")
        if jp:
            if "logit_gate_dataset_loc_loc" in params:
                distributions.update(
                    _build_hyperparameter_posteriors(
                        params,
                        "logit_gate_dataset_loc",
                        "logit_gate_dataset_loc",
                    )
                )
            if "psi_gate_dataset_loc" in params:
                distributions.update(
                    _build_neg_hyperparameter_posteriors(params, "gate_dataset")
                )
            distributions["gate"] = _build_joint_low_rank_posterior(
                params, "gate", jp, split
            )
        else:
            distributions.update(
                _build_hyperparameter_posteriors(
                    params,
                    "logit_gate_dataset_loc",
                    "logit_gate_dataset_loc",
                )
            )
            distributions.update(
                _build_neg_hyperparameter_posteriors(params, "gate_dataset")
            )
            distributions["gate_raw_dataset"] = _build_normal_posterior(
                params, "gate_raw_dataset", low_rank=low_rank
            )
        return

    distributions.update(
        _build_hyperparameter_posteriors(
            params,
            "logit_gate_dataset_loc",
            "logit_gate_dataset_scale",
        )
    )
    jp = _find_joint_prefix(params, "gate")
    if jp:
        distributions["gate"] = _build_joint_low_rank_posterior(
            params, "gate", jp, split
        )
    else:
        distributions["gate"] = _build_sigmoid_normal_posterior(
            params, "gate", is_scalar=False, split=split
        )


def _apply_zero_inflation_gate(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
) -> None:
    """Pass 6: add the zero-inflation gate posterior.

    Active for zero-inflated models (ZINB, ZINBVCP) when the gate is
    *not* already handled by a dataset-level hierarchy pass (pass 5).
    Handles four sub-cases: horseshoe gene-level gate, NEG gene-level
    gate, hierarchical gene-level gate, or a plain gate.  Each sub-case
    also checks for joint-prefix membership.
    """
    hierarchical_dataset_gate = getattr(
        model_config, "hierarchical_dataset_gate", False
    )
    horseshoe_dataset_gate = getattr(
        model_config, "horseshoe_dataset_gate", False
    )
    neg_dataset_gate = (
        model_config.gate_dataset_prior == HierarchicalPriorType.NEG
    )
    if not (
        model_config.is_zero_inflated
        and not (
            hierarchical_dataset_gate
            or horseshoe_dataset_gate
            or neg_dataset_gate
        )
    ):
        return

    horseshoe_gate = getattr(model_config, "horseshoe_gate", False)
    neg_gate = model_config.gate_prior == HierarchicalPriorType.NEG
    hierarchical_gate = model_config.hierarchical_gate
    joint_gate_prefix = _find_joint_prefix(params, "gate")

    if horseshoe_gate:
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
        return

    if neg_gate:
        distributions.update(
            _build_hyperparameter_posteriors(
                params, "logit_gate_loc", "logit_gate_loc"
            )
        )
        distributions.update(
            _build_neg_hyperparameter_posteriors(params, "gate")
        )
        distributions["gate_raw"] = _build_normal_posterior(
            params, "gate_raw", low_rank=low_rank
        )
        return

    if hierarchical_gate:
        distributions.update(
            _build_hyperparameter_posteriors(
                params, "logit_gate_loc", "logit_gate_scale"
            )
        )
        if joint_gate_prefix:
            distributions["gate"] = _build_joint_low_rank_posterior(
                params, "gate", joint_gate_prefix, split
            )
        else:
            distributions.update(
                _build_gate_posterior(params, unconstrained, is_mixture, split)
            )
        return

    if joint_gate_prefix:
        distributions["gate"] = _build_joint_low_rank_posterior(
            params, "gate", joint_gate_prefix, split
        )
    else:
        distributions.update(
            _build_gate_posterior(params, unconstrained, is_mixture, split)
        )


def _apply_capture(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    parameterization: Parameterization,
    unconstrained: bool,
    split: bool,
) -> None:
    """Pass 7: add the per-cell capture probability posterior.

    Active for VCP models.  Three variants:
    - Biology-informed / shared-scaling: posterior on ``eta_capture``.
    - Mean-odds parameterization: posterior on ``phi_capture``.
    - Other parameterizations: posterior on ``p_capture``.
    """
    if not model_config.uses_variable_capture:
        return

    bio_capture = getattr(model_config, "uses_biology_informed_capture", False)
    shared_scaling = getattr(model_config, "shared_capture_scaling", False)
    if bio_capture or shared_scaling:
        distributions.update(
            _build_biology_informed_capture_posterior(
                params, model_config, split
            )
        )
        return

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


def _apply_mixture_weights(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    is_mixture: bool,
) -> None:
    """Pass 8: add the Dirichlet posterior for mixture weights."""
    if is_mixture and "mixing_concentrations" in params:
        concentrations = params["mixing_concentrations"]
        distributions["mixing_weights"] = dist.Dirichlet(concentrations)


def _apply_joint_aggregation(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
) -> None:
    """Pass 9: build full joint posterior objects keyed ``joint:{group}``.

    When ``joint_params`` is configured, the per-parameter marginals are
    already in ``distributions`` (added by earlier passes via
    ``_build_joint_low_rank_posterior``).  This pass additionally builds
    the full stacked ``LowRankMVN`` that captures cross-parameter
    correlations, keyed as ``"joint:{group}"``.
    """
    config_joint = getattr(model_config, "joint_params", None)
    if not config_joint:
        return

    groups_seen: Dict[str, List[str]] = {}
    for pname in config_joint:
        jp = _find_joint_prefix(params, pname)
        if jp:
            parts = jp.split("_")
            group = parts[1]
            groups_seen.setdefault(group, []).append(pname)

    for group, pnames in groups_seen.items():
        joint_dist = _build_joint_full_distribution(params, pnames, group)
        if joint_dist is not None:
            distributions[f"joint:{group}"] = joint_dist


# =============================================================================
# Public orchestrator
# =============================================================================


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    split: bool = False,
) -> Dict[str, Any]:
    """Extract posterior distributions from optimized guide parameters.

    Runs nine ordered passes over a shared ``distributions`` dict.  Each
    pass is a no-op when its config flags are off, so the pipeline is safe
    for any ``ModelConfig``.

    Pass ordering (do not reorder):

    1. Base parameterization  — core params (r/p, mu/p, mu/phi)
    2. Gene-level hierarchy   — overrides p/phi with per-gene prior
    3. Dataset hierarchy: mu  — overrides mu/r with per-dataset prior
    4. Dataset hierarchy: p   — overrides p/phi with per-dataset prior
    5. Dataset hierarchy: gate — overrides gate with per-dataset prior
    6. Zero-inflation gate    — adds gate (if not handled by pass 5)
    7. Capture probability    — adds per-cell p_capture / phi_capture
    8. Mixture weights        — adds mixing_weights Dirichlet
    9. Joint aggregation      — adds ``joint:{group}`` full distributions

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Optimized variational parameters from SVI inference.
    model_config : ModelConfig
        Model configuration specifying parameterization and model type.
    split : bool, default=False
        If True, return lists of univariate distributions for vector-valued
        parameters.

    Returns
    -------
    Dict[str, Any]
        Maps parameter names to posterior distributions.  Values are
        either numpyro ``Distribution`` objects or ``{"base", "transform"}``
        dicts for unconstrained / low-rank guides.
    """
    distributions: Dict[str, Any] = {}
    parameterization = model_config.parameterization
    unconstrained = model_config.unconstrained
    is_mixture = model_config.is_mixture

    # Detect low-rank guides by checking for the W + raw_diag param pairs
    # that LowRankMVN guides always emit.
    low_rank = any(
        key.endswith("_W") and f"{key.replace('_W', '_raw_diag')}" in params
        for key in params.keys()
    )

    # Horseshoe priors replace base guide sites; skip those in pass 1.
    skip = _build_base_skip_set(model_config, parameterization)

    # --- Execute the pipeline (ordering matters — see docstring) ----------

    _apply_base_parameterization(  # Pass 1
        distributions,
        params,
        parameterization,
        unconstrained,
        is_mixture,
        low_rank,
        split,
        skip,
    )
    _apply_gene_level_hierarchy(  # Pass 2
        distributions,
        params,
        model_config,
        parameterization,
        is_mixture,
        low_rank,
        split,
    )
    _apply_dataset_hierarchy_mu(  # Pass 3
        distributions,
        params,
        model_config,
        parameterization,
        is_mixture,
        low_rank,
        split,
    )
    _apply_dataset_hierarchy_p(  # Pass 4
        distributions,
        params,
        model_config,
        parameterization,
        is_mixture,
        low_rank,
        split,
    )
    _apply_dataset_hierarchy_gate(  # Pass 5
        distributions,
        params,
        model_config,
        low_rank,
        split,
    )
    _apply_zero_inflation_gate(  # Pass 6
        distributions,
        params,
        model_config,
        unconstrained,
        is_mixture,
        low_rank,
        split,
    )
    _apply_capture(  # Pass 7
        distributions,
        params,
        model_config,
        parameterization,
        unconstrained,
        split,
    )
    _apply_mixture_weights(distributions, params, is_mixture)  # Pass 8
    _apply_joint_aggregation(distributions, params, model_config)  # Pass 9

    return distributions


# =============================================================================
# Parameterization-specific builders (layer 3)
# =============================================================================
#
# Each function below populates a dict with the core model parameters for
# one parameterization family.  They are called by _apply_base_param… and
# handle both constrained (Beta/LogNormal/BetaPrime) and unconstrained
# (TransformedNormal) guide families, as well as joint-prefix resolution.
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
            jp = _find_joint_prefix(params, "p")
            if jp:
                distributions["p"] = _build_joint_low_rank_posterior(
                    params, "p", jp, split
                )
            else:
                distributions["p"] = _build_sigmoid_normal_posterior(
                    params, "p", is_scalar=True, split=split
                )
        if "r" not in skip:
            jp = _find_joint_prefix(params, "r")
            if jp:
                distributions["r"] = _build_joint_low_rank_posterior(
                    params, "r", jp, split
                )
            elif low_rank:
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
            jp = _find_joint_prefix(params, "p")
            if jp:
                distributions["p"] = _build_joint_low_rank_posterior(
                    params, "p", jp, split
                )
            else:
                distributions["p"] = _build_sigmoid_normal_posterior(
                    params, "p", is_scalar=True, split=split
                )
        if "mu" not in skip:
            jp = _find_joint_prefix(params, "mu")
            if jp:
                distributions["mu"] = _build_joint_low_rank_posterior(
                    params, "mu", jp, split
                )
            elif low_rank:
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
            jp = _find_joint_prefix(params, "phi")
            if jp:
                distributions["phi"] = _build_joint_low_rank_posterior(
                    params, "phi", jp, split
                )
            else:
                distributions["phi"] = _build_exp_normal_posterior(
                    params, "phi", is_mixture=False, split=split, is_scalar=True
                )
        if "mu" not in skip:
            jp = _find_joint_prefix(params, "mu")
            if jp:
                distributions["mu"] = _build_joint_low_rank_posterior(
                    params, "mu", jp, split
                )
            elif low_rank:
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
# Hierarchical & horseshoe posterior builders (layer 3)
# =============================================================================
#
# Builders for the hyperparameters introduced by hierarchical and horseshoe
# priors: population-level loc/scale, and the horseshoe trio (tau, lambda,
# c_sq).  Also includes the NCP raw-z builder (_build_normal_posterior).
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


def _build_neg_hyperparameter_posteriors(
    params: Dict[str, jnp.ndarray],
    prefix: str,
) -> Dict[str, Any]:
    """Build LogNormal posteriors for NEG hyperparameters.

    The NEG prior uses psi (per-gene variance) and zeta (per-gene rate).
    Both have LogNormal variational posteriors since they are positive-valued,
    mirroring the GammaSpec guide implementation.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Guide parameters.  Must contain ``psi_{prefix}_loc``,
        ``psi_{prefix}_scale``, ``zeta_{prefix}_loc``, ``zeta_{prefix}_scale``.
    prefix : str
        Naming prefix (e.g. ``"p"``, ``"phi"``, ``"gate"``, ``"mu_dataset"``).

    Returns
    -------
    Dict[str, Any]
        Posterior distributions for psi and zeta.
    """
    distributions = {}

    for role in ("psi", "zeta"):
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
        base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
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

    The variational posterior is on ``eta_capture`` (the log-ratio
    log(M_c / L_c), constrained >= 0 via TruncatedNormal).  The
    capture parameter is a deterministic transformation of eta.

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

    # Shared mu_eta for data-driven mode (unconstrained — can be any real)
    if "mu_eta_loc" in params:
        distributions["mu_eta"] = dist.Normal(
            params["mu_eta_loc"], params["mu_eta_scale"]
        )

    # Per-cell eta_capture posterior (truncated at 0 to enforce
    # the physical constraint M_c >= L_c <=> p_capture <= 1)
    if "eta_capture_loc" in params:
        loc = params["eta_capture_loc"]
        scale = params["eta_capture_scale"]
        eta_dist = dist.TruncatedNormal(loc, scale, low=0.0)
        if split:
            distributions["eta_capture"] = [
                dist.TruncatedNormal(loc[i], scale[i], low=0.0)
                for i in range(loc.shape[0])
            ]
        else:
            distributions["eta_capture"] = eta_dist

    return distributions


# =============================================================================
# Leaf distribution builders (layer 4)
# =============================================================================
#
# Low-level constructors that map raw variational parameters (_loc, _scale,
# _alpha, _beta, _W, _raw_diag) to a single numpyro distribution object.
# Each handles both scalar and vector params, and optional splitting.
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
# Splitting utilities (used when split=True)
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
