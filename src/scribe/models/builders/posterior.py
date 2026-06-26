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
   ``_build_low_rank_positive_normal_posterior``, etc.
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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

# Import Parameterization and HierarchicalPriorType enums directly to avoid
# circular import
from ..config.enums import HierarchicalPriorType, Parameterization
from ..config.enums import (
    TWOSTATE_REGIME_COORD,
    TWOSTATE_SIGMOID_REGIME_COORDS,
)


def _twostate_param_names(
    parameterization: Parameterization,
) -> tuple[str, str, str]:
    """Return the (mean, overdispersion, regime) sampled parameter names.

    Deterministic per-parameterization mapping for the TwoState family, used in
    place of variational-key sniffing.  Key sniffing breaks once a coordinate
    carries a horseshoe/NCP or dataset hierarchy (its variational site is
    ``{name}_raw`` / ``{name}_raw_dataset``, not ``{name}_loc``), so the
    detection must come from the parameterization itself.
    """
    if parameterization == Parameterization.TWO_STATE_MOMENT_DELTA:
        return ("mu", "excess_fano", "inv_concentration")
    if parameterization == Parameterization.TWO_STATE_MEAN_FANO:
        return ("mu", "excess_fano", "concentration")
    if parameterization == Parameterization.TWO_STATE_RATIO:
        return ("mu", "burst_size", "switching_ratio")
    # Natural (and any unspecified TwoState default).
    return ("mu", "burst_size", "k_off")

# Set of LNM-family enum values. The compositional path is identical
# across canonical / mean_prob / mean_odds, so any check that needs
# "is this the LNM family?" can membership-test against this set.
_LNM_FAMILY_ENUMS = frozenset(
    {
        Parameterization.LOGISTIC_NORMAL_CANONICAL,
        Parameterization.LOGISTIC_NORMAL_MEAN_PROB,
        Parameterization.LOGISTIC_NORMAL_MEAN_ODDS,
    }
)
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


def _resolve_pos_transform_for(
    pt, name: str
) -> Optional[dist.transforms.Transform]:
    """Return a concrete positive-constraint transform for a named parameter.

    Posterior helpers receive ``pos_transform`` as either:

      * a single :class:`numpyro.distributions.transforms.Transform`
        (legacy / string form of ``ModelConfig.positive_transform``),
      * a callable mapping parameter names to ``Transform`` instances
        (new dict form with mixed values; built in
        :func:`build_posterior_distributions`),
      * ``None`` (rare; preserved for backward compatibility).

    Use this helper at any leaf site that needs a *concrete* ``Transform``
    for a known target parameter; it short-circuits on the legacy form
    and only invokes the callable when needed.
    """
    if pt is None:
        return None
    if callable(pt) and not isinstance(pt, dist.transforms.Transform):
        return pt(name)
    return pt


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
    *,
    transform=None,
) -> Dict[str, Any]:
    """Build a low-rank marginal posterior from joint guide params.

    Uses the stored ``{prefix}_loc``, ``{prefix}_W``,
    ``{prefix}_raw_diag`` to reconstruct the per-parameter marginal
    distribution + transform, identical in structure to the output
    of ``_build_low_rank_positive_normal_posterior``.

    For gene-specific parameters (G > 1), returns a
    ``LowRankMultivariateNormal``.  For scalar parameters that were
    expanded to G=1 during joint guide setup, collapses back to a
    ``Normal`` with variance ``sum(W[..., 0, :]**2) + D[..., 0]``.

    Parameters
    ----------
    params : dict
        Variational parameter dictionary.
    name : str
        Logical parameter name (used for fallback transform selection).
    prefix : str
        Full key prefix (e.g. ``"joint_joint_mu"``).
    split : bool
        Whether to split (not supported for low-rank; ignored).
    transform : numpyro Transform, optional
        Transform mapping unconstrained reals to the parameter support.
        When ``None`` (backward compat), falls back to name-based
        selection: ``SigmoidTransform`` for probability params,
        ``ExpTransform`` for positive params.

    Returns
    -------
    dict
        ``{"base": distribution, "transform": transform}``
    """
    import jax

    loc = params[f"{prefix}_loc"]
    raw_diag = params[f"{prefix}_raw_diag"]
    D = jax.nn.softplus(raw_diag) + 1e-4

    W_key = f"{prefix}_W"
    has_W = W_key in params

    if has_W:
        # Dense (low-rank) param in a joint group
        W = params[W_key]
        if loc.shape[-1] == 1:
            scalar_var = jnp.sum(W[..., 0, :] ** 2, axis=-1) + D[..., 0]
            base = dist.Normal(loc[..., 0], jnp.sqrt(scalar_var))
        else:
            base = dist.LowRankMultivariateNormal(
                loc=loc, cov_factor=W, cov_diag=D
            )
    else:
        # Nondense param in a structured joint group: diagonal Normal.
        # This gives the conditional marginal (at dense params = MAP).
        # Wrap in Independent so the gene dimension is event (matching
        # the LowRankMVN convention used for dense params).
        sigma = jnp.sqrt(D)
        if loc.shape[-1] == 1:
            base = dist.Normal(loc[..., 0], sigma[..., 0])
        else:
            base = dist.Independent(
                dist.Normal(loc, sigma), reinterpreted_batch_ndims=1
            )

    # Use caller-supplied transform, or fall back to name-based default
    if transform is None:
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
    dense_names = []
    batch_shapes: List[Tuple[int, ...]] = []
    for pname in joint_params:
        prefix = f"joint_{group}_{pname}"
        loc_key = f"{prefix}_loc"
        if loc_key not in params:
            return None
        W_key = f"{prefix}_W"
        # Skip nondense params in structured groups (they lack _W)
        if W_key not in params:
            continue
        loc_i = params[loc_key]
        W_i = params[W_key]
        raw_diag_i = params[f"{prefix}_raw_diag"]
        D_i = jax.nn.softplus(raw_diag_i) + 1e-4

        locs.append(loc_i)
        Ws.append(W_i)
        Ds.append(D_i)
        sizes.append(loc_i.shape[-1])
        dense_names.append(pname)
        batch_shapes.append(loc_i.shape[:-1])

    if not locs:
        return None

    # Resolve a shared batch shape for heterogeneous joint groups where
    # some members are shared across datasets and others are dataset-specific.
    ref_batch = max(batch_shapes, key=len)

    # Broadcast helper keeps trailing event/rank dimensions untouched while
    # expanding only leading batch dimensions to the reference shape.
    def _broadcast_batch(
        arr: jnp.ndarray, keep_last_dims: int, name: str
    ) -> jnp.ndarray:
        """Broadcast ``arr`` batch axes to the joint reference batch shape.

        Parameters
        ----------
        arr : jnp.ndarray
            Array to broadcast.
        keep_last_dims : int
            Number of trailing non-batch dimensions that must be preserved.
            Use 1 for ``loc``/``D`` and 2 for ``W``.
        name : str
            Name used in error messages.

        Returns
        -------
        jnp.ndarray
            Broadcasted array with batch shape equal to ``ref_batch``.
        """
        if keep_last_dims <= 0:
            raise ValueError("keep_last_dims must be positive")

        arr_batch = arr.shape[:-keep_last_dims]
        arr_tail = arr.shape[-keep_last_dims:]

        if len(arr_batch) > len(ref_batch):
            raise ValueError(
                f"Joint group '{group}': tensor '{name}' has batch shape "
                f"{arr_batch}, which has higher rank than reference batch "
                f"shape {ref_batch}."
            )

        if arr_batch != ref_batch[: len(arr_batch)]:
            raise ValueError(
                f"Joint group '{group}': tensor '{name}' has incompatible "
                f"batch shape {arr_batch} for reference batch shape "
                f"{ref_batch}. Expected a prefix-compatible batch."
            )

        # Insert singleton axes between existing batch axes and trailing
        # event/rank axes so prefix-compatible tensors can expand to ref_batch.
        pad = (1,) * (len(ref_batch) - len(arr_batch))
        arr = jnp.reshape(arr, arr_batch + pad + arr_tail)
        target_shape = ref_batch + arr_tail
        return jnp.broadcast_to(arr, target_shape)

    locs = [
        _broadcast_batch(loc_i, keep_last_dims=1, name="loc") for loc_i in locs
    ]
    Ds = [_broadcast_batch(D_i, keep_last_dims=1, name="diag") for D_i in Ds]
    Ws = [_broadcast_batch(W_i, keep_last_dims=2, name="factor") for W_i in Ws]

    # Stack into single vectors / matrices
    full_loc = jnp.concatenate(locs, axis=-1)
    full_W = jnp.concatenate(Ws, axis=-2)
    full_D = jnp.concatenate(Ds, axis=-1)

    base = dist.LowRankMultivariateNormal(
        loc=full_loc, cov_factor=full_W, cov_diag=full_D
    )
    return {
        "base": base,
        "param_names": dense_names,
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
    """Determine which base params to skip because a hierarchy pass replaces them.

    When any hierarchical prior (Gaussian, horseshoe, or NEG) is active for
    a parameter, the base builder should skip that parameter.  For NCP priors
    (horseshoe/NEG) the base guide site doesn't exist at all; for Gaussian
    hierarchical priors the guide may use low-rank format (W/raw_diag) that
    the base mean-field builder cannot handle.  In all cases the hierarchy
    pass (Pass 2/2b/3/4) is responsible for building the posterior.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration carrying hierarchical prior flags.
    parameterization : Parameterization
        Active parameterization enum.

    Returns
    -------
    set of str
        Parameter names to skip in base extraction. For logistic-normal
        models this includes ``r_T`` (total-count dispersion) rather than
        canonical ``r`` when expression hierarchies are active.
    """
    skip: set[str] = set()

    # Any gene-level or dataset-level hierarchy on p/phi overrides the base
    any_p_hierarchy = getattr(model_config, "hierarchical_p", False)
    any_dataset_p_hierarchy = (
        model_config.prob_dataset_prior != HierarchicalPriorType.NONE
    )
    if any_p_hierarchy or any_dataset_p_hierarchy:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
            skip.add("phi")
        else:
            skip.add("p")

    # Any gene-level or dataset-level hierarchy on mu/r overrides the base
    any_mu_hierarchy = (
        getattr(model_config, "expression_prior", HierarchicalPriorType.NONE)
        != HierarchicalPriorType.NONE
    )
    any_dataset_mu_hierarchy = (
        model_config.expression_dataset_prior != HierarchicalPriorType.NONE
    )
    if any_mu_hierarchy or any_dataset_mu_hierarchy:
        if parameterization in (
            Parameterization.CANONICAL,
            Parameterization.STANDARD,
        ):
            skip.add("r")
        elif parameterization in _LNM_FAMILY_ENUMS:
            # Logistic-normal models use r_T (total-count dispersion)
            # instead of r (gene-level NB dispersion), so hierarchy
            # overrides must skip r_T in the base pass.
            skip.add("r_T")
        else:
            skip.add("mu")

    # mean_disp dispersion (r) multi-factor hierarchy: r is reconstructed
    # additively (Pass 3b, _apply_dataset_hierarchy_target), so the base
    # mean-field builder must skip the absent ``r_loc``/``r_scale`` site.
    if parameterization == Parameterization.MEAN_DISP:
        _gs = getattr(model_config, "grouping_spec", None)
        if _gs is not None and any(
            f.family("dispersion") != "none"
            for f in getattr(_gs, "factors", ())
        ):
            skip.add("r")

    # TwoState dataset-level regime hierarchy overrides the base regime
    # coordinate.  Under a horseshoe/NEG (NCP) dataset prior the base guide
    # site does not exist (only ``{coord}_raw_dataset`` + hyper sites do), so
    # the base builder must skip it; the dataset-regime pass rebuilds it.
    if (
        getattr(model_config, "regime_dataset_prior", HierarchicalPriorType.NONE)
        != HierarchicalPriorType.NONE
    ):
        _regime_coord = getattr(
            model_config, "regime_dataset_target", None
        ) or TWOSTATE_REGIME_COORD.get(parameterization)
        if _regime_coord is not None:
            skip.add(_regime_coord)
    return skip


def _flow_guided_param_names(
    params: Dict[str, Any],
    model_config: Any = None,
) -> set[str]:
    """Return the set of base parameter names that are flow-guided.

    Scans the variational parameter dict for flow keys and extracts the
    logical parameter name so that earlier passes can skip those
    parameters.  Detects four kinds of flow-managed parameters:

    1. **Per-param flows**: ``flow_{name}$params`` /
       ``flow_{name}_idx{k}$params``.
    2. **Joint flow dense params**: ``joint_flow_{group}_{name}$params``
       / ``joint_flow_{group}_{name}_idx{k}$params``.
    3. **Joint flow nondense params**: Parameters whose variational
       sites live under ``joint_flow_{group}_{name}_loc`` (with a
       matching ``_raw_diag`` key) but have no ``$params`` key of their
       own.  These are the diagonal-regression parameters created by
       ``JointNormalizingFlowGuide`` when ``dense_params`` is set.
    4. **Concatenated joint flows**: ``joint_flow_{group}_concat$params``
       — all parameters in ``model_config.joint_params`` are flow-guided.

    Without (3), nondense params like ``p`` in ``dense_params=r,
    joint_param=r,p`` would be missed by ``flow_skip`` and incorrectly
    processed by hierarchy passes (Pass 2/2b) instead of being deferred
    to the joint-flow nondense fallback in Pass 10.

    Parameters
    ----------
    params : dict
        Variational parameter dictionary.
    model_config : ModelConfig, optional
        Model configuration.  Needed to resolve which parameter names
        belong to a concatenated joint flow (pattern 4).

    Returns
    -------
    set of str
        Base parameter names owned by a normalizing flow guide.
    """
    names: set[str] = set()
    for key in params:
        if not key.endswith("$params"):
            continue

        if key.startswith("flow_"):
            inner = key[len("flow_") : -len("$params")]
            # Independent mixture: flow_{name}_idx{k}$params
            if "_idx" in inner:
                inner = inner.rsplit("_idx", 1)[0]
            names.add(inner)

        elif key.startswith("joint_flow_"):
            inner = key[len("joint_flow_") : -len("$params")]
            # Concatenated flow: joint_flow_{group}_concat$params
            # — handled separately below via model_config.joint_params.
            if inner.endswith("_concat") or inner == "concat":
                continue
            # Independent mixture: joint_flow_{group}_{name}_idx{k}$params
            if "_idx" in inner:
                inner = inner.rsplit("_idx", 1)[0]
            parts = inner.split("_", 1)
            if len(parts) == 2:
                names.add(parts[1])

    # Detect nondense joint-flow params that have _loc / _raw_diag but
    # no $params key (diagonal-regression block in JointNormalizingFlowGuide).
    for key in params:
        if not key.startswith("joint_flow_") or not key.endswith("_loc"):
            continue
        if "$params" in key or "_scalar_" in key:
            continue
        inner = key[len("joint_flow_") : -len("_loc")]
        parts = inner.split("_", 1)
        if len(parts) != 2:
            continue
        _group, name = parts
        # Only add if not already detected as a dense flow param
        if name not in names:
            raw_diag_key = key.replace("_loc", "_raw_diag")
            if raw_diag_key in params:
                names.add(name)

    # Detect concatenated joint flows: joint_flow_{group}_concat$params.
    # Scan guide_families (from model_config) to find all params in the
    # concat group.
    concat_groups: set = set()
    for key in params:
        if key.endswith("_concat$params") and key.startswith("joint_flow_"):
            inner = key[len("joint_flow_") : -len("_concat$params")]
            concat_groups.add(inner)

    if concat_groups and model_config is not None:
        from ..components.guide_families import JointNormalizingFlowGuide

        guide_families = getattr(model_config, "guide_families", None)
        if guide_families is not None:
            families_dict = (
                guide_families.__dict__
                if hasattr(guide_families, "__dict__")
                else {}
            )
            for pname, guide in families_dict.items():
                if (
                    isinstance(guide, JointNormalizingFlowGuide)
                    and guide.group in concat_groups
                ):
                    names.add(pname)

    return names


def _apply_base_parameterization(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    parameterization: Parameterization,
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    skip: set[str],
    *,
    pos_transform=None,
) -> None:
    """Pass 1: build posteriors for the core model parameters.

    Dispatches to the parameterization-specific builder (canonical,
    mean_prob, mean_odds, or logistic_normal) which populates
    ``distributions`` with the primary model parameters. Parameters in
    *skip* are omitted because hierarchy passes will handle them later.
    """
    if parameterization in (
        Parameterization.CANONICAL,
        Parameterization.STANDARD,
    ):
        distributions.update(
            _build_canonical_posteriors(
                params,
                unconstrained,
                is_mixture,
                low_rank,
                split,
                skip,
                pos_transform=pos_transform,
            )
        )
    elif parameterization in (
        Parameterization.MEAN_PROB,
        Parameterization.LINKED,
    ):
        distributions.update(
            _build_mean_prob_posteriors(
                params,
                unconstrained,
                is_mixture,
                low_rank,
                split,
                skip,
                pos_transform=pos_transform,
            )
        )
    elif parameterization in (
        Parameterization.MEAN_ODDS,
        Parameterization.ODDS_RATIO,
    ):
        distributions.update(
            _build_mean_odds_posteriors(
                params,
                unconstrained,
                is_mixture,
                low_rank,
                split,
                skip,
                pos_transform=pos_transform,
            )
        )
    elif parameterization == Parameterization.MEAN_DISP:
        distributions.update(
            _build_mean_disp_posteriors(
                params,
                unconstrained,
                is_mixture,
                low_rank,
                split,
                skip,
                pos_transform=pos_transform,
            )
        )
    elif parameterization in _LNM_FAMILY_ENUMS:
        # The three LNM variants share the compositional path but
        # differ in which scalars of the totals NB are sampled vs
        # derived. The single ``_build_logistic_normal_posteriors``
        # helper dispatches on the variant internally, so the call
        # site stays shape-agnostic.
        distributions.update(
            _build_logistic_normal_posteriors(
                params,
                unconstrained,
                is_mixture,
                low_rank,
                split,
                skip,
                parameterization=parameterization,
                pos_transform=pos_transform,
            )
        )
    elif parameterization == Parameterization.COUNT_LOGNORMAL:
        # PLN has no core NB scalars — log-rates come from the VAE
        # decoder and are added as decoder-derived distributions in
        # ``get_distributions()``. The only guide-level parameter is
        # ``d_pln`` (diagonal noise, learned mode only).
        distributions.update(
            _build_poisson_lognormal_posteriors(params, is_mixture, split)
        )
    elif parameterization in (
        Parameterization.TWO_STATE_NATURAL,
        Parameterization.TWO_STATE_RATIO,
        Parameterization.TWO_STATE_MEAN_FANO,
        Parameterization.TWO_STATE_MOMENT_DELTA,
    ):
        # TwoState (Poisson-Beta compound). The three per-gene
        # parameters (mu, burst_size, k_off-or-switching_ratio) are
        # positive Normals under softplus (or exp). The builder
        # detects which third parameter was sampled by scanning
        # ``params`` keys; no parameterization-specific branch needed
        # here. Phase 1 does not support mixtures.
        distributions.update(
            _build_two_state_posteriors(
                params,
                unconstrained,
                is_mixture,
                low_rank,
                split,
                skip,
                pos_transform=pos_transform,
                parameterization=parameterization,
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
    flow_skip: set[str],
    *,
    pos_transform=None,
) -> None:
    """Pass 2: override p/phi with a gene-level hierarchical prior.

    Active when ``hierarchical_p=True``, ``horseshoe_p=True``, or
    ``prob_prior==NEG``.  Adds hyperparameter posteriors (loc/scale of the
    population prior) and *replaces* the base ``"p"`` or ``"phi"`` entry
    with the gene-level distribution.

    Horseshoe variant: adds the horseshoe trio (tau, lambda, c_sq) and
    the NCP raw z variable instead of the constrained parameter.

    NEG variant: adds psi and zeta LogNormal posteriors and the NCP raw z
    variable instead of the constrained parameter.

    Parameters in *flow_skip* (flow-guided params) are excluded from
    the gene-level posterior build; their hyperparameters are still built
    and Pass 10 handles the gene-level distribution.
    """
    hierarchical_p = model_config.hierarchical_p
    horseshoe_p = getattr(model_config, "horseshoe_p", False)
    neg_p = model_config.prob_prior == HierarchicalPriorType.NEG
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
        # The constrained param is reconstructed in _reconstruct_ncp_maps.
        distributions.update(
            _build_hyperparameter_posteriors(params, loc_name, loc_name)
        )
        if target_name in flow_skip:
            return
        distributions.update(
            _build_horseshoe_hyperparameter_posteriors(params, prefix)
        )
        distributions[f"{target_name}_raw"] = _build_normal_posterior(
            params, f"{target_name}_raw", low_rank=low_rank
        )
        return

    if neg_p:
        # NEG NCP: hyper_loc + {psi, zeta} + raw z variable.
        # The constrained param is reconstructed in _reconstruct_ncp_maps.
        distributions.update(
            _build_hyperparameter_posteriors(params, loc_name, loc_name)
        )
        if target_name in flow_skip:
            return
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
    # Flow-guided: hypers are built above; gene-level param handled by Pass 10.
    if target_name in flow_skip:
        return
    # Resolve transform: phi is positive (use pos_transform), p is (0,1).
    _jp_tf = pos_transform if target_name == "phi" else None
    jp = _find_joint_prefix(params, target_name)
    if jp:
        distributions[target_name] = _build_joint_low_rank_posterior(
            params, target_name, jp, split, transform=_jp_tf
        )
    elif f"{target_name}_W" in params:
        distributions[target_name] = _build_low_rank_positive_normal_posterior(
            params, target_name, is_mixture, split, transform=_jp_tf
        )
    elif target_name == "phi":
        distributions[target_name] = _build_positive_normal_posterior(
            params,
            target_name,
            is_mixture,
            split,
            is_scalar=False,
            transform=pos_transform,
        )
    else:
        distributions[target_name] = _build_sigmoid_normal_posterior(
            params, target_name, is_scalar=False, split=split
        )


def _apply_gene_level_mu_hierarchy(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    parameterization: Parameterization,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    flow_skip: set[str],
    *,
    pos_transform=None,
) -> None:
    """Pass 2b: override mu/r with a gene-level hierarchical prior.

    Active when ``expression_prior != NONE``.  Adds population-level
    hyperparameter posteriors (per-gene loc and scalar scale) and *replaces* the
    base ``"mu"`` (or ``"r"``) entry with the gene-level distribution.

    Horseshoe variant: adds the horseshoe trio (tau, lambda, c_sq) and the NCP
    raw z variable instead of the constrained parameter.

    NEG variant: adds psi and zeta LogNormal posteriors and the NCP raw z
    variable instead of the constrained parameter.

    Parameters in *flow_skip* (flow-guided params) are excluded from the
    gene-level posterior build; their hyperparameters are still built and Pass
    10 handles the gene-level distribution.
    """
    expression_prior = getattr(
        model_config, "expression_prior", HierarchicalPriorType.NONE
    )
    if expression_prior == HierarchicalPriorType.NONE:
        return

    if parameterization in (
        Parameterization.CANONICAL,
        Parameterization.STANDARD,
    ):
        target_name = "r"
        loc_name = "log_r_loc"
        scale_name = "log_r_scale"
        prefix = "r"
    else:
        target_name = "mu"
        loc_name = "log_mu_loc"
        scale_name = "log_mu_scale"
        prefix = "mu"

    if expression_prior == HierarchicalPriorType.HORSESHOE:
        # Horseshoe NCP: hyper_loc + {tau, lambda, c_sq} + raw z variable.
        distributions.update(
            _build_hyperparameter_posteriors(params, loc_name, loc_name)
        )
        if target_name in flow_skip:
            return
        distributions.update(
            _build_horseshoe_hyperparameter_posteriors(params, prefix)
        )
        distributions[f"{target_name}_raw"] = _build_normal_posterior(
            params, f"{target_name}_raw", low_rank=low_rank
        )
        return

    if expression_prior == HierarchicalPriorType.NEG:
        # NEG NCP: hyper_loc + {psi, zeta} + raw z variable.
        distributions.update(
            _build_hyperparameter_posteriors(params, loc_name, loc_name)
        )
        if target_name in flow_skip:
            return
        distributions.update(
            _build_neg_hyperparameter_posteriors(params, prefix)
        )
        distributions[f"{target_name}_raw"] = _build_normal_posterior(
            params, f"{target_name}_raw", low_rank=low_rank
        )
        return

    # Gaussian hierarchical: population hyper-priors + constrained param.
    distributions.update(
        _build_hyperparameter_posteriors(params, loc_name, scale_name)
    )
    # Flow-guided: hypers are built above; gene-level param handled by Pass 10.
    if target_name in flow_skip:
        return
    # mu and r are always positive-valued — use pos_transform.
    jp = _find_joint_prefix(params, target_name)
    if jp:
        distributions[target_name] = _build_joint_low_rank_posterior(
            params, target_name, jp, split, transform=pos_transform
        )
    elif f"{target_name}_W" in params:
        distributions[target_name] = _build_low_rank_positive_normal_posterior(
            params,
            target_name,
            is_mixture,
            split,
            transform=pos_transform,
        )
    else:
        distributions[target_name] = _build_positive_normal_posterior(
            params,
            target_name,
            is_mixture,
            split,
            is_scalar=False,
            transform=pos_transform,
        )


def _resolve_multifactor_factors(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    target: str,
    target_kwarg: str = "expression",
) -> List[Tuple[Any, str]]:
    """Return ``[(factor, site_prefix), ...]`` for the multi-factor mu/r prior.

    The additive multi-factor decomposition (see :func:`_multifactor_hier` in
    the preset factory) applies only when the grouping spec has more than one
    factor and at least one factor carries a non-"none" family for
    ``target_kwarg`` (``"expression"`` for the mean ``mu``/``r``, ``"dispersion"``
    for the ``mean_disp`` dispersion ``r``).  The per-factor NumPyro site prefix
    mirrors ``_multifactor_site_prefix`` exactly
    (``f"{target}_{name.replace(':', '__')}"``).  Returns an empty list when the
    fit is not a multi-factor model for this target, so callers fall back to the
    single-axis dataset-hierarchy path.
    """
    gs = getattr(model_config, "grouping_spec", None)
    factors = getattr(gs, "factors", ()) if gs is not None else ()
    if len(factors) <= 1:
        return []
    resolved: List[Tuple[Any, str]] = []
    for fac in factors:
        if fac.family(target_kwarg) == "none":
            continue
        prefix = f"{target}_{fac.name.replace(':', '__')}"
        if f"{prefix}_raw_loc" in params:
            resolved.append((fac, prefix))
    return resolved


def _build_multifactor_leaf_posterior(
    params: Dict[str, jnp.ndarray],
    mf_factors: List[Tuple[Any, str]],
    hyper_loc: str,
    is_mixture: bool,
    split: bool,
    *,
    transform=None,
    target_kwarg: str = "expression",
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """Reconstruct the leaf-level mu/r posterior for the multi-factor hierarchy.

    Mirrors :meth:`MultiFactorNormalWithTransformSpec.sample_hierarchical` but
    on the variational *parameters* instead of in a trace: the unconstrained
    leaf location is the population intercept plus a sum of per-factor zero-mean
    effects gathered onto the leaf axis::

        acc^(leaf) = loc + sum_f  scale_f * z_f[level_f(leaf)]
        mu^(leaf)  = transform(acc^(leaf))

    Each latent is mean-field Normal in unconstrained space, so the assembled
    leaf location/variance are sums of the component locations/variances (the
    factor scale ``scale_f`` enters as a point estimate: the fixed constant for
    a fixed effect, ``softplus(loc)`` for a gaussian random effect, or the
    regularized-horseshoe combination of the trio's medians).  The result is a
    ``TransformedDistribution(Normal(acc_loc, acc_std), transform)`` of shape
    ``(K?, n_leaves, G)`` — the same shape the single-axis dataset hierarchy
    produces, so downstream MAP / canonical conversion is unchanged.
    """
    n_lead = 1 if is_mixture else 0

    loc_loc = params[f"{hyper_loc}_loc"]  # (K?, G)
    loc_scale = params[f"{hyper_loc}_scale"]  # (K?, G)

    # Insert the leaf axis right after any leading (component) axes so it
    # broadcasts against the per-leaf gathered effects -> (K?, 1, G).
    acc_loc = jnp.expand_dims(loc_loc, axis=n_lead)
    acc_var = jnp.expand_dims(jnp.asarray(loc_scale) ** 2, axis=n_lead)

    for fac, prefix in mf_factors:
        raw_loc = params[f"{prefix}_raw_loc"]  # (K?, L_f, G)
        raw_scale = params[f"{prefix}_raw_scale"]  # (K?, L_f, G)
        family = fac.family(target_kwarg)

        if fac.effect_type == "fixed":
            scale = fac.fixed_scale if fac.fixed_scale is not None else 1.0
        elif family == "gaussian":
            # SoftplusNormal scale site -> median point estimate softplus(loc).
            scale = jnp.logaddexp(0.0, params[f"{prefix}_scale_loc"])
        elif family == "horseshoe":
            # LogNormal hyper-posteriors -> median exp(loc); per-gene lambda.
            tau = jnp.exp(params[f"tau_{prefix}_loc"])
            lam = jnp.exp(params[f"lambda_{prefix}_loc"])
            c_sq = jnp.exp(params[f"c_sq_{prefix}_loc"])
            scale = tau * jnp.sqrt(c_sq) * lam / jnp.sqrt(
                c_sq + tau**2 * lam**2
            )
        else:  # neg or unknown — not emitted by the factory for multi-factor.
            raise NotImplementedError(
                f"multi-factor expression family {family!r} for factor "
                f"{fac.name!r} has no analytic MAP reconstruction; use "
                "get_posterior_samples()-based summaries instead."
            )

        effect_loc = scale * raw_loc
        effect_var = (scale**2) * jnp.asarray(raw_scale) ** 2
        level_idx = jnp.asarray(fac.leaf_to_level)
        acc_loc = acc_loc + jnp.take(effect_loc, level_idx, axis=n_lead)
        acc_var = acc_var + jnp.take(effect_var, level_idx, axis=n_lead)

    if transform is None:
        transform = dist.transforms.ExpTransform()
    base = dist.Normal(acc_loc, jnp.sqrt(acc_var))
    posterior = dist.TransformedDistribution(base, transform)
    if split:
        return _split_transformed_distribution(
            posterior, acc_loc.shape, is_mixture
        )
    return posterior


def _apply_ncp_dataset_posterior(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    *,
    kind: str,
    target: str,
    hyper_loc: str,
    hs_prefix: str,
    raw_name: str,
    target_tf,
    low_rank: bool,
    split: bool,
) -> None:
    """Reconstruct one NCP (horseshoe/NEG) dataset-level target posterior.

    This is the shared body of the horseshoe/NEG branches that every Pass-3/4/4b/5
    builder repeated verbatim (8 near-identical copies): a joint-prefix check on
    the raw-z site (or the target inside a joint block), the population-loc and
    shrinkage hyperparameter posteriors, and either the joint-block target
    reconstruction or the standalone raw-z normal. The per-pass differences are
    only the site names and the ``target_tf`` applied to a joint reconstruction,
    so they are passed in — this helper carries no transform or target logic.

    Parameters
    ----------
    kind : str
        ``"horseshoe"`` (τ/λ/c² hypers) or ``"neg"`` (ζ/ψ Gamma-Gamma hypers).
    target, hyper_loc, hs_prefix, raw_name : str
        The target site, its population-location site, the shrinkage hyper
        prefix, and the non-centered raw-z site name.
    target_tf : Transform or None
        Transform applied when the target is reconstructed from a joint block
        (the caller's per-target choice; ``None`` for the standalone raw-z path).
    """
    if kind == "horseshoe":
        hyper_present = f"tau_{hs_prefix}_loc" in params
        build_shrinkage = _build_horseshoe_hyperparameter_posteriors
    else:  # neg
        hyper_present = f"psi_{hs_prefix}_concentration" in params
        build_shrinkage = _build_neg_hyperparameter_posteriors

    jp = _find_joint_prefix(params, raw_name) or _find_joint_prefix(
        params, target
    )
    if jp:
        # Joint path: the raw z lives in the joint block, but the hyper-priors
        # may still carry individual params (not folded into the joint guide).
        if f"{hyper_loc}_loc" in params:
            distributions.update(
                _build_hyperparameter_posteriors(params, hyper_loc, hyper_loc)
            )
        if hyper_present:
            distributions.update(build_shrinkage(params, hs_prefix))
        distributions[target] = _build_joint_low_rank_posterior(
            params, target, jp, split, transform=target_tf
        )
    else:
        distributions.update(
            _build_hyperparameter_posteriors(params, hyper_loc, hyper_loc)
        )
        distributions.update(build_shrinkage(params, hs_prefix))
        distributions[raw_name] = _build_normal_posterior(
            params, raw_name, low_rank=low_rank
        )


def _apply_dataset_hierarchy_target(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    *,
    target: str,
    hyper_loc: str,
    hyper_scale: str,
    hs_prefix: str,
    target_kwarg: str = "expression",
    horseshoe_dataset: bool = False,
    neg_dataset: bool = False,
    single_axis_active: bool = False,
    pos_transform=None,
) -> None:
    """Pass 3: override a positive target (``mu``/``r``) with its dataset-level
    hierarchical prior.

    Target-parametric — the caller selects the target and its site names, so one
    implementation serves both the expression mean (``mu``, or ``r`` in
    canonical) and the ``mean_disp`` dispersion ``r``
    (``target_kwarg="dispersion"``).

    The **additive multi-factor** hierarchy (crossed grouping factors) is handled
    first and applies to either target whenever the per-factor effect sites are
    present. The **single-axis** dataset hierarchy (legacy ``dataset_key="batch"``
    with a gaussian/horseshoe/NEG family) is expression-only and runs only when
    ``single_axis_active``; it adds dataset-level hyperparameter posteriors and
    replaces the target with a ``(K, D, G)`` entry.
    """
    # Multi-factor additive hierarchy: the leaf parameter is the population
    # intercept plus a sum of per-factor effects (no single ``{hyper_scale}``
    # site), so reconstruct it directly. Applies to mu OR r.
    mf_factors = _resolve_multifactor_factors(
        params, model_config, target, target_kwarg=target_kwarg
    )
    if mf_factors:
        distributions[target] = _build_multifactor_leaf_posterior(
            params,
            mf_factors,
            hyper_loc,
            is_mixture,
            split,
            transform=pos_transform,
            target_kwarg=target_kwarg,
        )
        return

    # Single-axis dataset hierarchy (expression-only).
    if not single_axis_active:
        return

    # The horseshoe/NEG raw z lives inside the joint block under either
    # "{target}_raw" or "{target}"; the shared helper tries both.
    if horseshoe_dataset:
        _apply_ncp_dataset_posterior(
            distributions,
            params,
            kind="horseshoe",
            target=target,
            hyper_loc=hyper_loc,
            hs_prefix=hs_prefix,
            raw_name=f"{target}_raw",
            target_tf=pos_transform,
            low_rank=low_rank,
            split=split,
        )
        return

    if neg_dataset:
        _apply_ncp_dataset_posterior(
            distributions,
            params,
            kind="neg",
            target=target,
            hyper_loc=hyper_loc,
            hs_prefix=hs_prefix,
            raw_name=f"{target}_raw",
            target_tf=pos_transform,
            low_rank=low_rank,
            split=split,
        )
        return

    distributions.update(
        _build_hyperparameter_posteriors(params, hyper_loc, hyper_scale)
    )
    jp = _find_joint_prefix(params, target)
    if jp:
        distributions[target] = _build_joint_low_rank_posterior(
            params, target, jp, split, transform=pos_transform
        )
    elif f"{target}_W" in params:
        distributions[target] = _build_low_rank_positive_normal_posterior(
            params,
            target,
            is_mixture,
            split,
            transform=pos_transform,
        )
    else:
        distributions[target] = _build_positive_normal_posterior(
            params,
            target,
            is_mixture,
            split,
            transform=pos_transform,
        )


def _apply_dataset_hierarchy_p(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    parameterization: Parameterization,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    *,
    pos_transform=None,
) -> None:
    """Pass 4: override p/phi with a dataset-level hierarchical prior.

    Active when ``hierarchical_dataset_p != "none"``,
    ``horseshoe_dataset_p=True``, or ``prob_dataset_prior==NEG``.  Replaces
    the ``"phi"`` (or ``"p"``) entry.  With
    ``hierarchical_dataset_p="gene_specific"`` the MAP gains shape
    ``(K, D, G)``; with ``"scalar"`` it becomes ``(K, D)``.
    """
    hierarchical_dataset_p = getattr(
        model_config, "hierarchical_dataset_p", "none"
    )
    horseshoe_dataset_p = getattr(model_config, "horseshoe_dataset_p", False)
    neg_dataset_p = model_config.prob_dataset_prior == HierarchicalPriorType.NEG
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

    # phi is positive (use pos_transform); p is (0, 1) (no joint transform).
    _jp_tf = pos_transform if target == "phi" else None

    if horseshoe_dataset_p:
        _apply_ncp_dataset_posterior(
            distributions,
            params,
            kind="horseshoe",
            target=target,
            hyper_loc=hyper_loc,
            hs_prefix=hs_prefix,
            raw_name=raw_name,
            target_tf=_jp_tf,
            low_rank=low_rank,
            split=split,
        )
        return

    if neg_dataset_p:
        _apply_ncp_dataset_posterior(
            distributions,
            params,
            kind="neg",
            target=target,
            hyper_loc=hyper_loc,
            hs_prefix=hs_prefix,
            raw_name=raw_name,
            target_tf=_jp_tf,
            low_rank=low_rank,
            split=split,
        )
        return

    distributions.update(
        _build_hyperparameter_posteriors(params, hyper_loc, hyper_scale)
    )
    jp = _find_joint_prefix(params, target)
    if jp:
        distributions[target] = _build_joint_low_rank_posterior(
            params, target, jp, split, transform=_jp_tf
        )
    elif f"{target}_W" in params:
        _tf = pos_transform if target == "phi" else None
        distributions[target] = _build_low_rank_positive_normal_posterior(
            params, target, is_mixture, split, transform=_tf
        )
    elif target == "phi":
        distributions[target] = _build_positive_normal_posterior(
            params,
            target,
            is_mixture,
            split,
            is_scalar=False,
            transform=pos_transform,
        )
    else:
        distributions[target] = _build_sigmoid_normal_posterior(
            params, target, is_scalar=False, split=split
        )


def _apply_dataset_hierarchy_regime(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    parameterization: Parameterization,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    *,
    pos_transform=None,
) -> None:
    """Pass 4b: rebuild the TwoState regime coordinate's dataset-level hierarchy.

    Active when ``regime_dataset_prior != "none"``.  The regime coordinate is
    ``k_off`` / ``switching_ratio`` / ``concentration`` / ``inv_concentration``
    depending on the parameterization (see ``TWOSTATE_REGIME_COORD``).  This
    mirrors :func:`_apply_dataset_hierarchy_p`, but selects sigmoid (for the
    bounded ``inv_concentration``) vs. positive (the others) reconstruction by
    the coordinate's support.  The resulting MAP gains a leading dataset axis.
    """
    regime_prior = getattr(
        model_config, "regime_dataset_prior", HierarchicalPriorType.NONE
    )
    if regime_prior == HierarchicalPriorType.NONE:
        return

    coord = getattr(
        model_config, "regime_dataset_target", None
    ) or TWOSTATE_REGIME_COORD.get(parameterization)
    if coord is None:
        return

    # inv_concentration lives in (0, 1) -> sigmoid hyper names + transform;
    # k_off / switching_ratio / concentration live in (0, inf) -> log/positive.
    is_sigmoid = coord in TWOSTATE_SIGMOID_REGIME_COORDS
    if is_sigmoid:
        hyper_loc = f"logit_{coord}_dataset_loc"
        hyper_scale = f"logit_{coord}_dataset_scale"
        target_tf = dist.transforms.SigmoidTransform()
    else:
        hyper_loc = f"log_{coord}_dataset_loc"
        hyper_scale = f"log_{coord}_dataset_scale"
        target_tf = pos_transform

    target = coord
    hs_prefix = f"{coord}_dataset"
    raw_name = f"{coord}_raw_dataset"

    # --- Horseshoe / NEG (NCP): hyper-loc + shrinkage hypers + raw-z normal -
    if regime_prior == HierarchicalPriorType.HORSESHOE:
        _apply_ncp_dataset_posterior(
            distributions,
            params,
            kind="horseshoe",
            target=target,
            hyper_loc=hyper_loc,
            hs_prefix=hs_prefix,
            raw_name=raw_name,
            target_tf=target_tf,
            low_rank=low_rank,
            split=split,
        )
        return

    if regime_prior == HierarchicalPriorType.NEG:
        _apply_ncp_dataset_posterior(
            distributions,
            params,
            kind="neg",
            target=target,
            hyper_loc=hyper_loc,
            hs_prefix=hs_prefix,
            raw_name=raw_name,
            target_tf=target_tf,
            low_rank=low_rank,
            split=split,
        )
        return

    # --- Gaussian (centered) dataset hierarchy: hyperparams + (D, G) target -
    distributions.update(
        _build_hyperparameter_posteriors(params, hyper_loc, hyper_scale)
    )
    jp = _find_joint_prefix(params, target)
    if jp:
        distributions[target] = _build_joint_low_rank_posterior(
            params, target, jp, split, transform=target_tf
        )
    elif f"{target}_W" in params:
        distributions[target] = _build_low_rank_positive_normal_posterior(
            params, target, is_mixture, split, transform=target_tf
        )
    elif is_sigmoid:
        distributions[target] = _build_sigmoid_normal_posterior(
            params, target, is_scalar=False, split=split
        )
    else:
        distributions[target] = _build_positive_normal_posterior(
            params,
            target,
            is_mixture,
            split,
            is_scalar=False,
            transform=pos_transform,
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
    ``horseshoe_dataset_gate=True``, or ``zero_inflation_dataset_prior==NEG``.
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
        model_config.zero_inflation_dataset_prior == HierarchicalPriorType.NEG
    )
    if not (
        hierarchical_dataset_gate or horseshoe_dataset_gate or neg_dataset_gate
    ):
        return

    if horseshoe_dataset_gate:
        _apply_ncp_dataset_posterior(
            distributions,
            params,
            kind="horseshoe",
            target="gate",
            hyper_loc="logit_gate_dataset_loc",
            hs_prefix="gate_dataset",
            raw_name="gate_raw_dataset",
            target_tf=None,  # gate is sigmoid; no joint transform
            low_rank=low_rank,
            split=split,
        )
        return

    if neg_dataset_gate:
        _apply_ncp_dataset_posterior(
            distributions,
            params,
            kind="neg",
            target="gate",
            hyper_loc="logit_gate_dataset_loc",
            hs_prefix="gate_dataset",
            raw_name="gate_raw_dataset",
            target_tf=None,  # gate is sigmoid; no joint transform
            low_rank=low_rank,
            split=split,
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
        model_config.zero_inflation_dataset_prior == HierarchicalPriorType.NEG
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
    neg_gate = model_config.zero_inflation_prior == HierarchicalPriorType.NEG
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
    *,
    pos_transform=None,
) -> None:
    """Pass 7: add the per-cell capture probability posterior.

    Active for VCP models and PLN with capture anchor.  Three variants:
    - Biology-informed / data-driven mu_eta: posterior on ``eta_capture``.
    - Mean-odds parameterization: posterior on ``phi_capture``.
    - Other parameterizations: posterior on ``p_capture``.
    """
    # PLN and NBLN use capture as an internal flag (no separate
    # 'plnvcp' / 'nblnvcp' models), so ``uses_variable_capture`` is
    # False for them. We still need to extract the ``eta_capture``
    # posterior when the biology-informed anchor is active. The PLN/NBLN
    # branch is gated on ``base_model in ("pln", "nbln")`` — accessed
    # via ``getattr`` because some unit-test fixtures use a
    # ``types.SimpleNamespace`` mock that lacks that attribute. Real
    # ``ModelConfig`` instances always have it.
    _has_capture = model_config.uses_variable_capture or (
        getattr(model_config, "base_model", None) in ("pln", "nbln")
        and getattr(model_config, "uses_biology_informed_capture", False)
    )
    if not _has_capture:
        return

    from ..config.enums import HierarchicalPriorType

    bio_capture = getattr(model_config, "uses_biology_informed_capture", False)
    mu_eta_active = (
        getattr(
            model_config, "capture_scaling_prior", HierarchicalPriorType.NONE
        )
        != HierarchicalPriorType.NONE
    )
    if bio_capture or mu_eta_active:
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
            _build_phi_capture_posterior(
                params,
                unconstrained,
                split,
                transform=pos_transform,
            )
        )
    else:
        distributions.update(
            _build_p_capture_posterior(params, unconstrained, split)
        )


def _apply_bnb_concentration(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    low_rank: bool,
    *,
    pos_transform=None,
) -> None:
    """Pass 7b: add the BNB overdispersion concentration posterior.

    Active when ``model_config.is_bnb`` is True.  Handles three prior
    families:

    * **Horseshoe / NEG** (NCP priors): adds posteriors for the hyper-
      location (``bnb_concentration_loc``), auxiliary sites (tau, lambda,
      c_sq for horseshoe; psi, zeta for NEG), and the NCP raw z variable.
      The constrained ``bnb_concentration`` is later reconstructed in
      ``_reconstruct_ncp_maps``.
    * **Gaussian** (hierarchical Normal): adds the hyper-location
      (``bnb_omega_hyper_loc``), hyper-scale (``bnb_omega_hyper_scale``),
      and the per-gene ``bnb_concentration`` directly as a
      ``TransformedDistribution(Normal, pos_transform)``.

    Parameters
    ----------
    distributions : Dict[str, Any]
        Accumulator dict (mutated in place).
    params : Dict[str, jnp.ndarray]
        Optimized variational parameters.
    model_config : ModelConfig
        Model configuration.
    low_rank : bool
        Whether LowRank guides are used.
    pos_transform : numpyro Transform, optional
        Positive-value transform (SoftplusTransform or ExpTransform).
        Required for Gaussian prior; ignored for NCP priors.
    """
    if not getattr(model_config, "is_bnb", False):
        return

    prefix = "bnb_concentration"

    # ------------------------------------------------------------------
    # Gaussian hierarchical prior
    # ------------------------------------------------------------------
    # Detect by checking for the guide's variational params for the
    # ``bnb_concentration`` sample site (``bnb_concentration_loc``),
    # while the NCP priors would instead have ``bnb_concentration_raw_loc``.
    is_gaussian = (
        f"{prefix}_loc" in params and f"{prefix}_raw_loc" not in params
    )
    if is_gaussian:
        # Hyper-location: bnb_omega_hyper_loc ~ Normal(loc, scale)
        hyper_loc_key = "bnb_omega_hyper_loc"
        if f"{hyper_loc_key}_loc" in params:
            distributions[hyper_loc_key] = dist.Normal(
                params[f"{hyper_loc_key}_loc"],
                params[f"{hyper_loc_key}_scale"],
            )

        # Hyper-scale: bnb_omega_hyper_scale ~ Softplus(Normal(loc, scale))
        hyper_scale_key = "bnb_omega_hyper_scale"
        if f"{hyper_scale_key}_loc" in params:
            _t = pos_transform or dist.transforms.SoftplusTransform()
            distributions[hyper_scale_key] = {
                "base": dist.Normal(
                    params[f"{hyper_scale_key}_loc"],
                    params[f"{hyper_scale_key}_scale"],
                ),
                "transform": _t,
            }

        # Per-gene bnb_concentration: transform(Normal(loc, scale))
        _t = pos_transform or dist.transforms.SoftplusTransform()
        distributions[prefix] = {
            "base": _build_normal_posterior(params, prefix, low_rank=low_rank),
            "transform": _t,
        }
        return

    # ------------------------------------------------------------------
    # Horseshoe / NEG NCP priors (existing logic)
    # ------------------------------------------------------------------

    # Hyper-location posterior (NCP: model site is ``bnb_concentration_loc``)
    loc_key = f"{prefix}_loc"
    if f"{loc_key}_loc" in params:
        distributions[loc_key] = dist.Normal(
            params[f"{loc_key}_loc"], params[f"{loc_key}_scale"]
        )

    # Horseshoe auxiliary sites
    for role in ("tau", "lambda", "c_sq"):
        name = f"{prefix}_{role}"
        if f"{name}_loc" in params:
            distributions[name] = dist.LogNormal(
                params[f"{name}_loc"], params[f"{name}_scale"]
            )

    # NEG auxiliary sites
    for role in ("psi", "zeta"):
        name = f"{prefix}_{role}"
        if f"{name}_concentration" in params:
            distributions[name] = dist.Gamma(
                params[f"{name}_concentration"], params[f"{name}_rate"]
            )

    # NCP raw z variable
    raw_name = f"{prefix}_raw"
    if f"{raw_name}_loc" in params:
        distributions[raw_name] = _build_normal_posterior(
            params, raw_name, low_rank=low_rank
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

    Runs ten ordered passes over a shared ``distributions`` dict.  Each
    pass is a no-op when its config flags are off, so the pipeline is safe
    for any ``ModelConfig``.

    Pass ordering (do not reorder):

    1.  Base parameterization  — core params (r/p, mu/p, mu/phi)
    2.  Gene-level hierarchy   — overrides p/phi with per-gene prior
    3.  Dataset hierarchy: mu  — overrides mu/r with per-dataset prior
    4.  Dataset hierarchy: p   — overrides p/phi with per-dataset prior
    5.  Dataset hierarchy: gate — overrides gate with per-dataset prior
    6.  Zero-inflation gate    — adds gate (if not handled by pass 5)
    7.  Capture probability    — adds per-cell p_capture / phi_capture
    8.  Mixture weights        — adds mixing_weights Dirichlet
    9.  Joint aggregation      — adds ``joint:{group}`` full distributions
    10. Flow posteriors        — reconstructs ``FlowDistribution`` objects
        for flow-guided params missed by earlier passes

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
        dicts for unconstrained / low-rank guides.  Flow-based entries
        have ``{"base": FlowDistribution, "transform": ...}``, and
        conditional joint-flow entries additionally include
        ``"conditional": True``.
    """
    distributions: Dict[str, Any] = {}
    parameterization = model_config.parameterization
    unconstrained = model_config.unconstrained
    is_mixture = model_config.is_mixture

    # Resolve the positive-value transform from config.  Honours both
    # the string form (``"softplus"`` or ``"exp"``) and the dict form
    # (``{"<param>": "softplus" | "exp", ...}``) for per-parameter
    # overrides.  When the dict carries mixed values, we build a
    # callable resolver and let the per-parameter posterior builders
    # query it.  For NB-family parameterizations that share a single
    # transform across all positive parameters, the resolver collapses
    # to that single transform (no behavior change).
    _pt_raw = getattr(model_config, "positive_transform", "exp")

    def _kind_to_transform(_kind: str) -> dist.transforms.Transform:
        return (
            dist.transforms.SoftplusTransform()
            if _kind == "softplus"
            else dist.transforms.ExpTransform()
        )

    if isinstance(_pt_raw, dict):
        # A ``positive_transform`` dict only *names* the transform for the
        # parameters it lists; every unlisted positive parameter falls back to
        # the model default (softplus) — exactly as the model factory and
        # ``ModelConfig.resolve_positive_transform`` resolve them at fit time.
        # We therefore decide uniform-vs-mixed by resolving the transform for
        # this parameterization's *actual* positive parameters, NOT by counting
        # distinct dict values.  The old ``len(set(values)) == 1`` test wrongly
        # collapsed e.g. ``{"mu": "exp"}`` to a global ExpTransform, which then
        # re-applied ``exp`` to the softplus-fit ``phi`` / ``r`` /
        # ``phi_capture`` during reconstruction.  For VCP odds-family models
        # that inflated ``phi_capture`` ~10x (exp(loc) vs softplus(loc)),
        # halving the reconstructed capture and every MAP-based mean — while
        # the sampling-based PPC, which goes through the real guide, stayed
        # correct.
        def _resolver(_name: str) -> dist.transforms.Transform:
            if hasattr(model_config, "resolve_positive_transform"):
                return _kind_to_transform(
                    model_config.resolve_positive_transform(_name)
                )
            # Fallback: direct dict lookup with softplus default.
            return _kind_to_transform(_pt_raw.get(_name, "softplus"))

        # Positive parameters whose reconstruction consumes ``pos_transform``,
        # per parameterization.  For VCP odds-family models, capture is the
        # positive ``phi_capture`` and must be included; sigmoid-based
        # ``p_capture`` (other families) ignores the positive transform.
        _pos_targets = {
            Parameterization.CANONICAL: ["r"],
            Parameterization.STANDARD: ["r"],
            Parameterization.MEAN_PROB: ["mu"],
            Parameterization.LINKED: ["mu"],
            Parameterization.MEAN_ODDS: ["phi", "mu"],
            Parameterization.ODDS_RATIO: ["phi", "mu"],
            # mean_disp samples both mu and r as positive parameters; both
            # consume pos_transform during MAP reconstruction.
            Parameterization.MEAN_DISP: ["mu", "r"],
        }.get(parameterization, [])
        if (
            _pos_targets
            and parameterization
            in (Parameterization.MEAN_ODDS, Parameterization.ODDS_RATIO)
            and getattr(model_config, "uses_variable_capture", False)
        ):
            _pos_targets = _pos_targets + ["phi_capture"]

        if _pos_targets and hasattr(
            model_config, "resolve_positive_transform"
        ):
            _kinds = {
                model_config.resolve_positive_transform(_t)
                for _t in _pos_targets
            }
        else:
            # Unknown target set (e.g. TwoState, LNM/PLN): preserve the legacy
            # value-based collapse so their behavior is unchanged.
            _kinds = set(_pt_raw.values()) or {"softplus"}

        if len(_kinds) == 1:
            # Genuinely uniform across this model's positive parameters.
            pos_transform = _kind_to_transform(next(iter(_kinds)))
        else:
            # Mixed transforms — thread the per-name resolver.  Supported for
            # the TwoState family and the odds-family (mean_odds / odds_ratio),
            # whose Pass-1 builders resolve ``pos_transform`` per parameter.
            if parameterization not in (
                Parameterization.TWO_STATE_NATURAL,
                Parameterization.TWO_STATE_RATIO,
                Parameterization.TWO_STATE_MEAN_FANO,
                Parameterization.TWO_STATE_MOMENT_DELTA,
                Parameterization.MEAN_ODDS,
                Parameterization.ODDS_RATIO,
                # mean_disp's Pass-1 builder resolves pos_transform per
                # parameter (mu and r), so mixed transforms are supported.
                Parameterization.MEAN_DISP,
            ):
                raise NotImplementedError(
                    "Per-parameter positive_transform (dict form with mixed "
                    "effective transforms) is supported for the TwoState, "
                    "odds-family (mean_odds / odds_ratio), and mean_disp "
                    "parameterizations. For other parameterizations, pass a "
                    "single string (\"softplus\" or \"exp\") or a dict whose "
                    "listed values match the softplus default."
                )
            pos_transform = _resolver
    else:
        pos_transform = _kind_to_transform(_pt_raw)

    # Detect low-rank guides by checking for the W + raw_diag param pairs
    # that LowRankMVN guides always emit.
    low_rank = any(
        key.endswith("_W") and f"{key.replace('_W', '_raw_diag')}" in params
        for key in params.keys()
    )

    # Horseshoe priors replace base guide sites; skip those in pass 1.
    skip = _build_base_skip_set(model_config, parameterization)

    # Flow-guided params have no _loc/_scale keys; skip in pass 1 and
    # let pass 10 (_apply_flow_posteriors) reconstruct them.
    # Keep the flow set separate so hierarchy passes only skip flow params,
    # not params they are responsible for overriding.
    flow_skip = _flow_guided_param_names(params, model_config)
    skip |= flow_skip

    # --- Execute the pipeline (ordering matters — see docstring) ----------

    # Pre-resolve per-helper positive transforms when ``pos_transform``
    # is the per-name resolver callable (dict form of
    # ``ModelConfig.positive_transform`` with mixed values).  Each
    # hierarchy / dataset helper has a SINGLE positive target so we
    # collapse to one ``Transform`` at the call site; for legacy
    # single-Transform inputs ``_resolve_pos_transform_for`` is a
    # no-op.  ``_apply_base_parameterization`` keeps the callable
    # because its TwoState branch threads it into the per-name builder.
    _is_mean_odds_family = parameterization in (
        Parameterization.MEAN_ODDS,
        Parameterization.ODDS_RATIO,
    )
    _is_mean_family = parameterization in (
        Parameterization.MEAN_PROB,
        Parameterization.LINKED,
        Parameterization.MEAN_ODDS,
        Parameterization.ODDS_RATIO,
        # mean_disp's expression hierarchy targets the gene mean mu.
        Parameterization.MEAN_DISP,
        Parameterization.TWO_STATE_NATURAL,
        Parameterization.TWO_STATE_RATIO,
        Parameterization.TWO_STATE_MEAN_FANO,
        Parameterization.TWO_STATE_MOMENT_DELTA,
    )
    _p_target = "phi" if _is_mean_odds_family else "p"
    _mu_target = "mu" if _is_mean_family else "r"

    _apply_base_parameterization(  # Pass 1
        distributions,
        params,
        parameterization,
        unconstrained,
        is_mixture,
        low_rank,
        split,
        skip,
        pos_transform=pos_transform,
    )
    _apply_gene_level_hierarchy(  # Pass 2: p/phi
        distributions,
        params,
        model_config,
        parameterization,
        is_mixture,
        low_rank,
        split,
        flow_skip,
        pos_transform=_resolve_pos_transform_for(pos_transform, _p_target),
    )
    _apply_gene_level_mu_hierarchy(  # Pass 2b: mu/r
        distributions,
        params,
        model_config,
        parameterization,
        is_mixture,
        low_rank,
        split,
        flow_skip,
        pos_transform=_resolve_pos_transform_for(pos_transform, _mu_target),
    )
    # Pass 3: dataset hierarchy on the expression mean (mu, or r in canonical).
    _expr_hyper = f"log_{_mu_target}_dataset"
    _apply_dataset_hierarchy_target(
        distributions,
        params,
        model_config,
        is_mixture,
        low_rank,
        split,
        target=_mu_target,
        hyper_loc=f"{_expr_hyper}_loc",
        hyper_scale=f"{_expr_hyper}_scale",
        hs_prefix=f"{_mu_target}_dataset",
        target_kwarg="expression",
        horseshoe_dataset=getattr(model_config, "horseshoe_dataset_mu", False),
        neg_dataset=(
            model_config.expression_dataset_prior == HierarchicalPriorType.NEG
        ),
        single_axis_active=(
            getattr(model_config, "hierarchical_dataset_mu", False)
            or getattr(model_config, "horseshoe_dataset_mu", False)
            or model_config.expression_dataset_prior
            == HierarchicalPriorType.NEG
        ),
        pos_transform=_resolve_pos_transform_for(pos_transform, _mu_target),
    )
    # Pass 3b: dataset hierarchy on the dispersion r (mean_disp only; the
    # additive multi-factor path -- there is no single-axis dispersion family).
    if parameterization == Parameterization.MEAN_DISP:
        _apply_dataset_hierarchy_target(
            distributions,
            params,
            model_config,
            is_mixture,
            low_rank,
            split,
            target="r",
            hyper_loc="log_r_dataset_loc",
            hyper_scale="log_r_dataset_scale",
            hs_prefix="r_dataset",
            target_kwarg="dispersion",
            single_axis_active=False,
            pos_transform=_resolve_pos_transform_for(pos_transform, "r"),
        )
    _apply_dataset_hierarchy_p(  # Pass 4
        distributions,
        params,
        model_config,
        parameterization,
        is_mixture,
        low_rank,
        split,
        pos_transform=_resolve_pos_transform_for(pos_transform, _p_target),
    )
    _regime_coord = getattr(
        model_config, "regime_dataset_target", None
    ) or TWOSTATE_REGIME_COORD.get(parameterization)
    _apply_dataset_hierarchy_regime(  # Pass 4b (TwoState regime coordinate)
        distributions,
        params,
        model_config,
        parameterization,
        is_mixture,
        low_rank,
        split,
        pos_transform=(
            _resolve_pos_transform_for(pos_transform, _regime_coord)
            if _regime_coord is not None
            else None
        ),
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
        # ``_apply_capture`` consumes the positive transform on two branches:
        # the odds-family ``phi_capture`` posterior (mean_odds / odds_ratio)
        # and the biology-informed ``eta_capture`` posterior.  Resolve for the
        # site this model actually builds so a mixed-dict resolver picks the
        # capture parameter's own transform (softplus by default) rather than
        # ``mu``'s.  The ``p_capture`` branch is sigmoid-based and ignores it.
        pos_transform=_resolve_pos_transform_for(
            pos_transform,
            "phi_capture" if _is_mean_odds_family else "eta_capture",
        ),
    )
    _apply_bnb_concentration(  # Pass 7b
        distributions,
        params,
        model_config,
        low_rank,
        pos_transform=_resolve_pos_transform_for(
            pos_transform, "bnb_concentration"
        ),
    )
    _apply_mixture_weights(distributions, params, is_mixture)  # Pass 8
    _apply_joint_aggregation(distributions, params, model_config)  # Pass 9
    _apply_flow_posteriors(  # Pass 10
        distributions, params, model_config, pos_transform=pos_transform
    )

    return distributions


# =============================================================================
# Pass 10: flow-guide posterior reconstruction
# =============================================================================


def _apply_flow_posteriors(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: "ModelConfig",
    *,
    pos_transform=None,
) -> None:
    """Pass 10: reconstruct posterior distributions for flow-guided params.

    After passes 1-9, any parameter whose guide is a normalizing flow
    will be missing from *distributions* because the standard key
    patterns (``_loc``/``_scale``, ``_W``/``_raw_diag``) do not match
    flow param keys (``flow_*$params``, ``joint_flow_*$params``).

    This pass detects those missing entries and reconstructs a
    ``FlowDistribution`` wrapped in the appropriate constraint transform.

    Per-parameter flows produce standalone distributions.  Joint flow
    (dense) parameters produce **conditional** distributions — a flag
    ``"conditional": True`` is set on the returned dict so downstream
    consumers (``get_map``) know to use sampling-based strategies.

    Joint-flow nondense / scalar params use ``_loc``/``_raw_diag`` keys
    that should already be handled by ``_find_joint_prefix`` in earlier
    passes; this function also covers them as a fallback if they were
    missed (e.g. because the ``joint_flow_`` prefix differs from the
    ``joint_`` prefix expected by ``_find_joint_prefix``).

    Parameters
    ----------
    distributions : dict
        Accumulated distributions dict (mutated in-place).
    params : dict
        Optimized variational parameters.
    model_config : ModelConfig
        Model configuration (must retain ``guide_families``).
    pos_transform : Transform, optional
        Positive-value transform (SoftplusTransform or ExpTransform).
    """
    guide_families = getattr(model_config, "guide_families", None)
    if guide_families is None:
        return

    # Lazy import to avoid circular deps
    from ..components.guide_families import (
        NormalizingFlowGuide,
        JointNormalizingFlowGuide,
    )
    from scribe.flows import FlowChain, FlowDistribution

    if pos_transform is None:
        pos_transform = dist.transforms.ExpTransform()
    elif callable(pos_transform) and not isinstance(
        pos_transform, dist.transforms.Transform
    ):
        # Per-name resolver (mixed positive_transform dict).  Flow params
        # reconstruct a single positive base transform; resolve with the
        # common positive target so the resolver collapses to one Transform.
        pos_transform = pos_transform("mu")

    # ------------------------------------------------------------------
    # Per-parameter flows: flow_{name}$params  (non-mixture)
    #                      flow_{name}_idx{k}$params (independent mixture)
    #                      flow_{name}$params + shared ctx (shared mixture)
    # ------------------------------------------------------------------
    _handled_per_param: set = set()

    # Pass A: detect independent-mixture component flows and group them
    _component_groups: Dict[str, List[Tuple[int, str]]] = {}
    for key in params:
        if not key.startswith("flow_") or not key.endswith("$params"):
            continue
        inner = key[len("flow_") : -len("$params")]
        # Independent mixture: flow_{name}_idx{k}$params
        if "_idx" in inner:
            base_name = inner.rsplit("_idx", 1)[0]
            try:
                k_idx = int(inner.rsplit("_idx", 1)[1])
            except ValueError:
                continue
            _component_groups.setdefault(base_name, []).append((k_idx, key))

    for base_name, entries in _component_groups.items():
        if base_name in distributions:
            continue
        guide = guide_families.get(base_name)
        if not isinstance(guide, NormalizingFlowGuide):
            continue
        comp_dist = _reconstruct_component_flow(
            params,
            base_name,
            entries,
            guide,
            pos_transform,
        )
        if comp_dist is not None:
            distributions[base_name] = comp_dist
        _handled_per_param.add(base_name)

    # Pass B: non-mixture / shared-mixture per-parameter flows
    for key in params:
        if not key.startswith("flow_") or not key.endswith("$params"):
            continue
        inner = key[len("flow_") : -len("$params")]
        if "_idx" in inner:
            continue
        name = inner
        if name in distributions or name in _handled_per_param:
            continue

        guide = guide_families.get(name)
        if not isinstance(guide, NormalizingFlowGuide):
            continue

        # Shared mixture: guide has mixture_strategy="shared" and
        # the model_config indicates n_components > 0
        n_comps = _get_n_components(model_config)
        if (
            getattr(guide, "mixture_strategy", "independent") == "shared"
            and n_comps is not None
            and n_comps > 1
        ):
            comp_dist = _reconstruct_shared_component_flow(
                params,
                key,
                name,
                guide,
                n_comps,
                pos_transform,
            )
            if comp_dist is not None:
                distributions[name] = comp_dist
            continue

        flow_dist = _reconstruct_per_param_flow(
            params, key, name, guide, pos_transform
        )
        if flow_dist is not None:
            distributions[name] = flow_dist

    # ------------------------------------------------------------------
    # Joint flow dense params: joint_flow_{group}_{name}$params
    #                         joint_flow_{group}_{name}_idx{k}$params
    # ------------------------------------------------------------------
    _handled_joint: set = set()

    # Pass A: detect independent-mixture joint component flows
    _joint_comp_groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for key in params:
        if not key.startswith("joint_flow_") or not key.endswith("$params"):
            continue
        inner = key[len("joint_flow_") : -len("$params")]
        if "_idx" not in inner:
            continue
        # Split: {group}_{name}_idx{k}
        pre_idx, idx_part = inner.rsplit("_idx", 1)
        try:
            k_idx = int(idx_part)
        except ValueError:
            continue
        parts = pre_idx.split("_", 1)
        if len(parts) != 2:
            continue
        group, name = parts
        _joint_comp_groups.setdefault((group, name), []).append((k_idx, key))

    for (group, name), entries in _joint_comp_groups.items():
        if name in distributions:
            continue
        guide = guide_families.get(name)
        if not isinstance(guide, JointNormalizingFlowGuide):
            continue
        comp_dist = _reconstruct_joint_component_flow(
            params,
            name,
            group,
            entries,
            guide,
            pos_transform,
        )
        if comp_dist is not None:
            distributions[name] = comp_dist
        _handled_joint.add(name)

    # Pass B: non-mixture / shared joint flows
    for key in params:
        if not key.startswith("joint_flow_") or not key.endswith("$params"):
            continue
        inner = key[len("joint_flow_") : -len("$params")]
        if "_idx" in inner:
            continue
        parts = inner.split("_", 1)
        if len(parts) != 2:
            continue
        group, name = parts
        if name in distributions or name in _handled_joint:
            continue

        guide = guide_families.get(name)
        if not isinstance(guide, JointNormalizingFlowGuide):
            continue

        flow_dist = _reconstruct_joint_flow_block(
            params, key, name, group, guide, pos_transform
        )
        if flow_dist is not None:
            distributions[name] = flow_dist

    # ------------------------------------------------------------------
    # Concatenated joint flows: joint_flow_{group}_concat$params
    # ------------------------------------------------------------------
    # All parameters in the group share a single FlowChain +
    # SlicedTransform.  Individual parameter distributions cannot be
    # extracted independently, so we mark each as "conditional" to
    # force sampling-based MAP.
    _apply_concatenated_joint_flow(
        distributions, params, model_config, guide_families, pos_transform
    )

    # ------------------------------------------------------------------
    # Joint flow nondense / scalar fallback
    # ------------------------------------------------------------------
    # Nondense params use joint_flow_{group}_{name}_loc which
    # _find_joint_prefix misses (it expects joint_{group}_{name}_loc).
    # Scalar params use joint_flow_{group}_{name}_scalar_loc.
    # Handle both as simple Normal posteriors.
    _apply_joint_flow_nondense_fallback(
        distributions, params, guide_families, pos_transform
    )


def _apply_concatenated_joint_flow(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    model_config: Any,
    guide_families: Any,
    pos_transform,
) -> None:
    """Handle concatenated joint flows (``dense_params=None``).

    When all parameters in a joint flow group are modeled by a single
    ``FlowChain`` + ``SlicedTransform``, the individual parameter
    distributions are not separable.  We mark each as ``"conditional"``
    so that ``get_map`` uses sampling-based MAP (running the guide to
    produce correlated samples).

    Detection strategy: find ``joint_flow_{group}_concat$params`` keys,
    then scan ``guide_families`` for all parameters whose guide is a
    ``JointNormalizingFlowGuide`` with matching group name.

    Parameters
    ----------
    distributions : dict
        Accumulated distributions dict (mutated in-place).
    params : dict
        Optimized variational parameters.
    model_config : ModelConfig
        Model configuration.
    guide_families : GuideFamilyConfig or None
        Guide family configuration.
    pos_transform : Transform
        Positive-value transform.
    """
    if guide_families is None:
        return

    from ..components.guide_families import JointNormalizingFlowGuide

    # Collect groups that have a concatenated flow key
    concat_groups: set = set()
    for key in params:
        if key.endswith("_concat$params") and key.startswith("joint_flow_"):
            # Extract group: joint_flow_{group}_concat$params
            inner = key[len("joint_flow_") : -len("_concat$params")]
            concat_groups.add(inner)

    if not concat_groups:
        return

    # Scan guide_families for params belonging to these concat groups
    families_dict = (
        guide_families.__dict__
        if hasattr(guide_families, "__dict__")
        else {}
    )
    for pname, guide in families_dict.items():
        if not isinstance(guide, JointNormalizingFlowGuide):
            continue
        if guide.group in concat_groups and pname not in distributions:
            distributions[pname] = {"conditional": True}


def _reconstruct_per_param_flow(
    params: Dict[str, jnp.ndarray],
    key: str,
    name: str,
    guide: Any,
    pos_transform,
) -> Optional[Dict[str, Any]]:
    """Rebuild a standalone ``FlowDistribution`` for a per-parameter flow.

    Parameters
    ----------
    params : dict
        Full SVI params dict.
    key : str
        Key for the Flax module weights (e.g. ``"flow_mu$params"``).
    name : str
        Logical parameter name.
    guide : NormalizingFlowGuide
        Guide marker with architecture hyperparameters.
    pos_transform : Transform
        Positive-value transform.

    Returns
    -------
    dict or None
        ``{"base": FlowDistribution, "transform": transform}`` or None.
    """
    from scribe.flows import FlowChain, FlowDistribution

    flax_params = params[key]

    # Infer feature dim from the first flow layer's weight shape.
    # FlowChain stores layers under 'layer_0', 'layer_1', etc.
    features = _infer_flow_features(flax_params)
    if features is None:
        return None

    chain = FlowChain(
        features=features,
        num_layers=guide.num_layers,
        flow_type=guide.flow_type,
        hidden_dims=list(guide.hidden_dims),
        activation=guide.activation,
        n_bins=guide.n_bins,
    )

    def flow_fn(x, reverse=False):
        return chain.apply({"params": flax_params}, x, reverse=reverse)

    base = dist.Normal(jnp.zeros(features), jnp.ones(features)).to_event(1)
    flow_dist = FlowDistribution(flow_fn, base)

    return {"base": flow_dist, "transform": pos_transform}


def _get_n_components(model_config) -> Optional[int]:
    """Extract n_components from a model_config, returning None if absent."""
    if hasattr(model_config, "n_components"):
        return getattr(model_config, "n_components", None)
    dims = getattr(model_config, "dims", None) or {}
    if isinstance(dims, dict):
        return dims.get("n_components")
    return None


def _reconstruct_component_flow(
    params: Dict[str, jnp.ndarray],
    base_name: str,
    entries: List[Tuple[int, str]],
    guide: Any,
    pos_transform,
) -> Optional[Dict[str, Any]]:
    """Rebuild a ``ComponentFlowDistribution`` for independent mixture flows.

    Each entry is ``(component_index, param_key)`` for one component's
    FlowChain weights.
    """
    from scribe.flows import (
        ComponentFlowDistribution,
        FlowChain,
        FlowDistribution,
    )

    entries_sorted = sorted(entries, key=lambda e: e[0])
    component_dists = []
    for _k, key in entries_sorted:
        flax_params = params[key]
        features = _infer_flow_features(flax_params)
        if features is None:
            return None

        chain = FlowChain(
            features=features,
            num_layers=guide.num_layers,
            flow_type=guide.flow_type,
            hidden_dims=list(guide.hidden_dims),
            activation=guide.activation,
            n_bins=guide.n_bins,
        )

        def _fn(x, reverse=False, _p=flax_params, _c=chain):
            return _c.apply({"params": _p}, x, reverse=reverse)

        base = dist.Normal(jnp.zeros(features), jnp.ones(features)).to_event(1)
        component_dists.append(FlowDistribution(_fn, base))

    comp_dist = ComponentFlowDistribution(
        component_dists, axis_name="component"
    )
    return {"base": comp_dist, "transform": pos_transform}


def _reconstruct_shared_component_flow(
    params: Dict[str, jnp.ndarray],
    key: str,
    name: str,
    guide: Any,
    n_components: int,
    pos_transform,
) -> Optional[Dict[str, Any]]:
    """Rebuild a shared ``ComponentFlowDistribution`` from one FlowChain.

    The single FlowChain is conditioned on a one-hot component index.
    """
    from scribe.flows import (
        ComponentFlowDistribution,
        FlowChain,
        FlowDistribution,
    )

    flax_params = params[key]
    features = _infer_flow_features(flax_params)
    if features is None:
        return None

    chain = FlowChain(
        features=features,
        num_layers=guide.num_layers,
        flow_type=guide.flow_type,
        hidden_dims=list(guide.hidden_dims),
        activation=guide.activation,
        n_bins=guide.n_bins,
        context_dim=n_components,
    )

    component_dists = []
    for k in range(n_components):
        ctx = jax.nn.one_hot(k, n_components)

        def _fn(x, reverse=False, _p=flax_params, _c=chain, _ctx=ctx):
            return _c.apply(
                {"params": _p},
                x,
                reverse=reverse,
                context=_ctx,
            )

        base = dist.Normal(jnp.zeros(features), jnp.ones(features)).to_event(1)
        component_dists.append(FlowDistribution(_fn, base))

    comp_dist = ComponentFlowDistribution(
        component_dists, axis_name="component"
    )
    return {"base": comp_dist, "transform": pos_transform}


def _reconstruct_joint_component_flow(
    params: Dict[str, jnp.ndarray],
    name: str,
    group: str,
    entries: List[Tuple[int, str]],
    guide: Any,
    pos_transform,
) -> Optional[Dict[str, Any]]:
    """Rebuild a ``ComponentFlowDistribution`` for independent joint flows."""
    from scribe.flows import (
        ComponentFlowDistribution,
        FlowChain,
        FlowDistribution,
    )

    entries_sorted = sorted(entries, key=lambda e: e[0])
    component_dists = []
    for _k, key in entries_sorted:
        flax_params = params[key]
        features = _infer_flow_features(flax_params)
        if features is None:
            return None

        chain = FlowChain(
            features=features,
            num_layers=guide.num_layers,
            flow_type=guide.flow_type,
            hidden_dims=list(guide.hidden_dims),
            activation=guide.activation,
            n_bins=guide.n_bins,
        )

        def _fn(x, reverse=False, _p=flax_params, _c=chain):
            return _c.apply({"params": _p}, x, reverse=reverse)

        base = dist.Normal(jnp.zeros(features), jnp.ones(features)).to_event(1)
        component_dists.append(FlowDistribution(_fn, base))

    comp_dist = ComponentFlowDistribution(
        component_dists, axis_name="component"
    )
    return {
        "base": comp_dist,
        "transform": pos_transform,
        "conditional": True,
    }


def _reconstruct_joint_flow_block(
    params: Dict[str, jnp.ndarray],
    key: str,
    name: str,
    group: str,
    guide: Any,
    pos_transform,
) -> Optional[Dict[str, Any]]:
    """Rebuild a conditional ``FlowDistribution`` for a joint-flow block.

    The returned distribution is **conditional** on an unspecified
    context vector from previously-sampled parameters.  Downstream
    MAP helpers must use sampling-based strategies for these entries.

    Parameters
    ----------
    params : dict
        Full SVI params dict.
    key : str
        Flax module weight key.
    name : str
        Logical parameter name.
    group : str
        Joint group identifier.
    guide : JointNormalizingFlowGuide
        Guide marker.
    pos_transform : Transform
        Positive-value transform.

    Returns
    -------
    dict or None
        ``{"base": FlowDistribution, "transform": ..., "conditional": True}``
    """
    from scribe.flows import FlowChain, FlowDistribution

    flax_params = params[key]
    features = _infer_flow_features(flax_params)
    if features is None:
        return None

    # Infer context_dim from the conditioner network input shape.
    # For a flow with context, the first layer's conditioner has
    # input_dim = ceil(features/2) + context_dim.  Without inspecting
    # all preceding blocks we cannot recover the exact context_dim,
    # so we reconstruct the chain with context_dim=0 for density eval.
    # The actual conditioned sampling is handled by guide execution
    # in get_map's flow strategies.
    chain = FlowChain(
        features=features,
        num_layers=guide.num_layers,
        flow_type=guide.flow_type,
        hidden_dims=list(guide.hidden_dims),
        activation=guide.activation,
        n_bins=guide.n_bins,
        context_dim=0,
    )

    def flow_fn(x, reverse=False):
        return chain.apply({"params": flax_params}, x, reverse=reverse)

    base = dist.Normal(jnp.zeros(features), jnp.ones(features)).to_event(1)
    flow_dist = FlowDistribution(flow_fn, base)

    return {
        "base": flow_dist,
        "transform": pos_transform,
        "conditional": True,
    }


def _apply_joint_flow_nondense_fallback(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    guide_families,
    pos_transform,
) -> None:
    """Handle joint-flow nondense and scalar params missed by earlier passes.

    Nondense params have keys like ``joint_flow_{group}_{name}_loc``.
    Scalar params have keys like ``joint_flow_{group}_{name}_scalar_loc``.
    Both are simple Normal posteriors wrapped in the constraint transform.

    Parameters
    ----------
    distributions : dict
        Accumulated distributions dict (mutated in-place).
    params : dict
        Optimized variational parameters.
    guide_families : GuideFamilyConfig
        Guide family configuration.
    pos_transform : Transform
        Positive-value transform.
    """
    # Probability-support parameters use SigmoidTransform; all others
    # use the caller-provided pos_transform (softplus or exp).
    _SIGMOID_PARAMS = frozenset({"p", "gate", "p_capture"})

    # Detect nondense: joint_flow_{group}_{name}_loc (not $params, not scalar)
    seen = set()
    for key in params:
        if not key.startswith("joint_flow_"):
            continue
        if key.endswith("$params"):
            continue

        # Nondense pattern: joint_flow_{group}_{name}_loc
        if key.endswith("_loc") and "_scalar_" not in key:
            inner = key[len("joint_flow_") : -len("_loc")]
            parts = inner.split("_", 1)
            if len(parts) != 2:
                continue
            _group, name = parts
            if name in distributions or name in seen:
                continue
            seen.add(name)

            loc = params[key]
            raw_diag_key = key.replace("_loc", "_raw_diag")
            if raw_diag_key in params:
                import jax

                sigma = jnp.sqrt(jax.nn.softplus(params[raw_diag_key]) + 1e-4)
            else:
                sigma = jnp.ones_like(loc)

            base = dist.Independent(
                dist.Normal(loc, sigma),
                reinterpreted_batch_ndims=1,
            )
            # Mark as conditional: the actual guide loc is adjusted by
            # regression on the dense flow residuals (alpha * r_resid),
            # so get_map must use sampling-based estimation.
            param_tf = (
                dist.transforms.SigmoidTransform()
                if name in _SIGMOID_PARAMS
                else pos_transform
            )
            distributions[name] = {
                "base": base,
                "transform": param_tf,
                "conditional": True,
            }

        # Scalar pattern: joint_flow_{group}_{name}_scalar_loc
        if "_scalar_loc" in key and key.endswith("_scalar_loc"):
            inner = key[len("joint_flow_") : -len("_scalar_loc")]
            parts = inner.split("_", 1)
            if len(parts) != 2:
                continue
            _group, name = parts
            if name in distributions or name in seen:
                continue
            seen.add(name)

            loc = params[key]
            raw_scale_key = key.replace("_scalar_loc", "_scalar_raw_scale")
            if raw_scale_key in params:
                import jax

                scale = jax.nn.softplus(params[raw_scale_key]) + 1e-4
            else:
                scale = jnp.ones_like(loc)

            base = dist.Normal(loc, scale)
            param_tf = (
                dist.transforms.SigmoidTransform()
                if name in _SIGMOID_PARAMS
                else pos_transform
            )
            distributions[name] = {
                "base": base,
                "transform": param_tf,
                "conditional": True,
            }


def _infer_flow_features(flax_params) -> Optional[int]:
    """Infer the feature dimensionality from stored Flax flow params.

    Inspects ``layer_0`` of the flow chain to recover the feature
    count.  Strategy (in priority order):

    1. **Affine coupling**: ``shift`` or ``log_scale`` param has shape
       ``(ceil(F/2),)`` → ``features = ceil(F/2) * 2``.
    2. **Coupling / autoregressive**: the first hidden-layer kernel
       (``hidden_0/kernel``) has shape ``(input_dim, hidden_dim)``
       where ``input_dim = ceil(F/2)`` for coupling flows without
       context → ``features = input_dim * 2``.

    Parameters
    ----------
    flax_params : dict
        Flax ``params`` pytree for a ``FlowChain``.

    Returns
    -------
    int or None
        Inferred feature count, or None if detection fails.
    """
    layer0 = None
    if isinstance(flax_params, dict):
        layer0 = flax_params.get("layer_0")

    if layer0 is None:
        return None

    # Affine coupling stores explicit shift / log_scale leaves
    shift = _deep_get(layer0, "shift")
    if shift is not None and hasattr(shift, "shape") and len(shift.shape) == 1:
        return shift.shape[0] * 2

    log_scale = _deep_get(layer0, "log_scale")
    if (
        log_scale is not None
        and hasattr(log_scale, "shape")
        and len(log_scale.shape) == 1
    ):
        return log_scale.shape[0] * 2

    # For coupling / autoregressive flows the first hidden-layer
    # kernel has shape (ceil(features/2) [+ context_dim], hidden_dim).
    # When context_dim == 0 (per-param flows), ceil(features/2) ==
    # kernel.shape[0].
    hidden0 = _deep_get(layer0, "hidden_0")
    if isinstance(hidden0, dict) and "kernel" in hidden0:
        kernel = hidden0["kernel"]
        if hasattr(kernel, "shape") and len(kernel.shape) == 2:
            half_features = kernel.shape[0]
            return half_features * 2

    return None


def _deep_get(d, key):
    """Recursively search a nested dict for *key* and return the value."""
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    for v in d.values():
        result = _deep_get(v, key)
        if result is not None:
            return result
    return None


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
    *,
    pos_transform=None,
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
                    params, "r", jp, split, transform=pos_transform
                )
            elif "r_W" in params:
                distributions["r"] = _build_low_rank_positive_normal_posterior(
                    params,
                    "r",
                    is_mixture,
                    split,
                    transform=pos_transform,
                )
            else:
                distributions["r"] = _build_positive_normal_posterior(
                    params,
                    "r",
                    is_mixture,
                    split,
                    transform=pos_transform,
                )
    else:
        if "p" not in skip:
            distributions["p"] = _build_beta_posterior(
                params, "p", is_scalar=True, is_mixture=False, split=split
            )
        if "r" not in skip:
            # Constrained low-rank guides store log-space MVN params
            # (e.g., log_r_W), while unconstrained guides use r_W.
            if "r_W" in params or "log_r_W" in params:
                distributions["r"] = _build_low_rank_lognormal_posterior(
                    params, "r", is_mixture, split
                )
            else:
                distributions["r"] = _build_lognormal_posterior(
                    params, "r", is_mixture, split
                )

    return distributions


def _build_logistic_normal_posteriors(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    skip: Optional[set] = None,
    *,
    parameterization: Optional[Parameterization] = None,
    pos_transform=None,
) -> Dict[str, Any]:
    """Build variational posteriors for the LNM totals NB submodel.

    Dispatches on the LNM-family ``parameterization`` to extract the
    posteriors of the *sampled* scalars (the ones the variational
    guide actually fits). Derived scalars — e.g. ``r_T`` and ``p``
    under ``mean_odds`` — are not in ``params`` and are not built
    here; they are computed downstream by the post-extraction logic
    that already handles the DM-family analogues
    (``_compute_mu_from_r_p`` etc.).

    Per-variant sampled scalars
    ---------------------------
    - ``logistic_normal_canonical``: ``(r_T, p)``
    - ``logistic_normal_mean_prob``: ``(mu_T, p)``
    - ``logistic_normal_mean_odds``: ``(mu_T, phi_T)``

    The compositional path (``y_alr``, ``z``, ``d_lnm``) lives outside
    the totals NB submodel and is handled identically across variants
    by the shared trailing block below.

    Parameters
    ----------
    parameterization : Parameterization, optional
        LNM-family enum value selecting the variant. Defaults to
        ``LOGISTIC_NORMAL_CANONICAL`` if not provided, mirroring the
        historical (variant-less) entry-point semantics so any legacy
        caller still gets the correct posteriors.
    """
    distributions = {}
    skip = skip or set()

    # Variant resolution. ``LOGISTIC_NORMAL`` (the legacy alias) and
    # ``LOGISTIC_NORMAL_CANONICAL`` are the same enum member, so the
    # default catches both. We compare via ``.value`` to be robust
    # against future enum refactors.
    if parameterization is None:
        variant = "canonical"
    else:
        variant = parameterization.value.replace("logistic_normal_", "")

    # ------------------------------------------------------------------
    # Variant-specific posterior extraction for the *sampled* scalars.
    # ------------------------------------------------------------------
    if variant == "canonical":
        # Sampled: (r_T, p). r_T uses LogNormal / PositiveNormal guide;
        # p uses Beta / SigmoidNormal.
        _build_lnm_p_and_dispersion_pair(
            distributions,
            params,
            dispersion_name="r_T",
            unconstrained=unconstrained,
            is_mixture=is_mixture,
            split=split,
            skip=skip,
            pos_transform=pos_transform,
        )
    elif variant == "mean_prob":
        # Sampled: (mu_T, p). mu_T uses LogNormal / PositiveNormal;
        # p uses Beta / SigmoidNormal. r_T is derived downstream.
        _build_lnm_p_and_dispersion_pair(
            distributions,
            params,
            dispersion_name="mu_T",
            unconstrained=unconstrained,
            is_mixture=is_mixture,
            split=split,
            skip=skip,
            pos_transform=pos_transform,
        )
    elif variant == "mean_odds":
        # Sampled: (mu_T, phi_T). Both are positive scalars — mu_T uses
        # LogNormal / PositiveNormal as in mean_prob; phi_T uses
        # BetaPrime / PositiveNormal (mirroring DM mean_odds for phi).
        # p and r_T are derived downstream.
        if "mu_T" not in skip:
            distributions["mu_T"] = _build_lnm_positive_scalar_posterior(
                params,
                "mu_T",
                unconstrained=unconstrained,
                is_mixture=is_mixture,
                split=split,
                pos_transform=pos_transform,
            )
        if "phi_T" not in skip:
            distributions["phi_T"] = _build_lnm_positive_scalar_posterior(
                params,
                "phi_T",
                unconstrained=unconstrained,
                is_mixture=is_mixture,
                split=split,
                pos_transform=pos_transform,
                # phi_T's constrained guide is BetaPrime (positive
                # support, supports the odds-ratio interpretation
                # 0 < phi_T < ∞). The unconstrained path uses the
                # standard PositiveNormal/positive-transform pair.
                constrained_is_beta_prime=True,
            )
    else:
        raise ValueError(
            f"Unknown LNM-family variant {variant!r}. Expected one of "
            f"'canonical', 'mean_prob', 'mean_odds'."
        )

    # ------------------------------------------------------------------
    # Compositional-path trailing block (variant-independent).
    # ------------------------------------------------------------------
    # Learned diagonal ALR noise vector d_lnm (positive, LogNormal guide).
    # Present only when d_mode="learned".
    if "d_lnm_loc" in params and "d_lnm_scale" in params:
        distributions["d_lnm"] = _build_lognormal_posterior(
            params, "d_lnm", is_mixture, split
        )

    return distributions


def _build_poisson_lognormal_posteriors(
    params: Dict[str, jnp.ndarray],
    is_mixture: bool,
    split: bool,
) -> Dict[str, Any]:
    """Build variational posteriors for PLN guide-level parameters.

    PLN's log-rates are produced by the VAE decoder and are *not*
    explicit guide sites — they are added as decoder-derived
    distributions in ``get_distributions()``.  The only guide-level
    parameter that this function handles is the learned diagonal noise
    ``d_pln`` (present when ``d_mode="learned"``).

    Parameters
    ----------
    params : dict
        NumPyro variational parameters (``{name}_loc``, ``{name}_scale``).
    is_mixture : bool
        Whether the model uses mixture components.
    split : bool
        Whether to split per-gene posteriors into a list.

    Returns
    -------
    dict
        Posterior distributions keyed by parameter name.
    """
    distributions: Dict[str, Any] = {}

    # Learned diagonal noise vector d_pln (positive, LogNormal guide).
    if "d_pln_loc" in params and "d_pln_scale" in params:
        distributions["d_pln"] = _build_lognormal_posterior(
            params, "d_pln", is_mixture, split
        )

    return distributions


def _build_lnm_p_and_dispersion_pair(
    distributions: Dict[str, Any],
    params: Dict[str, jnp.ndarray],
    *,
    dispersion_name: str,
    unconstrained: bool,
    is_mixture: bool,
    split: bool,
    skip: set,
    pos_transform,
) -> None:
    """LNM canonical / mean_prob posterior pair: ``p`` + a positive scalar.

    Both variants sample a Beta / SigmoidNormal ``p`` plus a positive
    scalar (``r_T`` for canonical, ``mu_T`` for mean_prob) with the
    same guide family pair. Sharing this code keeps the dispatch in
    ``_build_logistic_normal_posteriors`` readable.
    """
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
        if dispersion_name not in skip:
            jp = _find_joint_prefix(params, dispersion_name)
            if jp:
                distributions[dispersion_name] = (
                    _build_joint_low_rank_posterior(
                        params,
                        dispersion_name,
                        jp,
                        split,
                        transform=pos_transform,
                    )
                )
            elif f"{dispersion_name}_W" in params:
                distributions[dispersion_name] = (
                    _build_low_rank_positive_normal_posterior(
                        params,
                        dispersion_name,
                        is_mixture,
                        split,
                        transform=pos_transform,
                    )
                )
            else:
                distributions[dispersion_name] = (
                    _build_positive_normal_posterior(
                        params,
                        dispersion_name,
                        is_mixture,
                        split,
                        transform=pos_transform,
                    )
                )
    else:
        if "p" not in skip:
            distributions["p"] = _build_beta_posterior(
                params, "p", is_scalar=True, is_mixture=False, split=split
            )
        if dispersion_name not in skip:
            # Constrained low-rank guides store log-space MVN params
            # (e.g., log_r_T_W), while unconstrained guides use the
            # un-prefixed variant. Either layout is supported.
            low_rank_keys = (
                f"{dispersion_name}_W",
                f"log_{dispersion_name}_W",
            )
            if any(k in params for k in low_rank_keys):
                distributions[dispersion_name] = (
                    _build_low_rank_lognormal_posterior(
                        params, dispersion_name, is_mixture, split
                    )
                )
            else:
                distributions[dispersion_name] = (
                    _build_lognormal_posterior(
                        params, dispersion_name, is_mixture, split
                    )
                )


def _build_lnm_positive_scalar_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    *,
    unconstrained: bool,
    is_mixture: bool,
    split: bool,
    pos_transform,
    constrained_is_beta_prime: bool = False,
) -> Any:
    """Build a posterior for a positive scalar in the LNM totals submodel.

    Used for ``mu_T`` and ``phi_T`` in the mean_odds variant. Selects
    between LogNormal / BetaPrime (constrained) and
    PositiveNormal-with-transform (unconstrained), with low-rank /
    joint guide variants as for the canonical scalars.

    Parameters
    ----------
    constrained_is_beta_prime : bool, default=False
        When ``True``, the constrained guide is BetaPrime (used for
        ``phi_T``). When ``False`` (default), it's LogNormal (used
        for ``mu_T``). The unconstrained path is identical for both.
    """
    if unconstrained:
        jp = _find_joint_prefix(params, name)
        if jp:
            return _build_joint_low_rank_posterior(
                params, name, jp, split, transform=pos_transform
            )
        if f"{name}_W" in params:
            return _build_low_rank_positive_normal_posterior(
                params, name, is_mixture, split, transform=pos_transform
            )
        return _build_positive_normal_posterior(
            params, name, is_mixture, split, transform=pos_transform
        )

    if constrained_is_beta_prime:
        # phi_T's constrained guide is BetaPrime(alpha, beta). Same
        # storage convention as the DM family's ``phi`` parameter,
        # which uses ``_build_betaprime_posterior``.
        return _build_betaprime_posterior(
            params, name, is_scalar=True, is_mixture=is_mixture, split=split
        )

    # Default constrained path: LogNormal, with optional low-rank
    # variant when log-MVN guide params are present.
    low_rank_keys = (f"{name}_W", f"log_{name}_W")
    if any(k in params for k in low_rank_keys):
        return _build_low_rank_lognormal_posterior(
            params, name, is_mixture, split
        )
    return _build_lognormal_posterior(params, name, is_mixture, split)


def _build_mean_prob_posteriors(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    skip: Optional[set] = None,
    *,
    pos_transform=None,
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
                    params, "mu", jp, split, transform=pos_transform
                )
            elif "mu_W" in params:
                distributions["mu"] = _build_low_rank_positive_normal_posterior(
                    params,
                    "mu",
                    is_mixture,
                    split,
                    transform=pos_transform,
                )
            else:
                distributions["mu"] = _build_positive_normal_posterior(
                    params,
                    "mu",
                    is_mixture,
                    split,
                    transform=pos_transform,
                )
    else:
        if "p" not in skip:
            distributions["p"] = _build_beta_posterior(
                params, "p", is_scalar=True, is_mixture=False, split=split
            )
        if "mu" not in skip:
            # Constrained low-rank guides store log-space MVN params
            # (e.g., log_mu_W), while unconstrained guides use mu_W.
            if "mu_W" in params or "log_mu_W" in params:
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
    *,
    pos_transform=None,
) -> Dict[str, Any]:
    """Build posteriors for mean_odds (odds_ratio) parameterization."""
    distributions = {}
    skip = skip or set()

    # ``phi`` and ``mu`` may use DIFFERENT positive transforms (e.g. the
    # common ``positive_transform={"mean_expression": "exp"}`` leaves ``phi``
    # on the softplus default).  Resolve each separately so a per-name resolver
    # callable (mixed dict) maps to the correct transform; for a single
    # ``Transform`` input ``_resolve_pos_transform_for`` is a no-op.
    _pt_phi = _resolve_pos_transform_for(pos_transform, "phi")
    _pt_mu = _resolve_pos_transform_for(pos_transform, "mu")

    if unconstrained:
        if "phi" not in skip:
            jp = _find_joint_prefix(params, "phi")
            if jp:
                distributions["phi"] = _build_joint_low_rank_posterior(
                    params, "phi", jp, split, transform=_pt_phi
                )
            elif "phi_W" in params:
                distributions["phi"] = (
                    _build_low_rank_positive_normal_posterior(
                        params,
                        "phi",
                        is_mixture=False,
                        split=split,
                        transform=_pt_phi,
                    )
                )
            else:
                distributions["phi"] = _build_positive_normal_posterior(
                    params,
                    "phi",
                    is_mixture=False,
                    split=split,
                    is_scalar=True,
                    transform=_pt_phi,
                )
        if "mu" not in skip:
            jp = _find_joint_prefix(params, "mu")
            if jp:
                distributions["mu"] = _build_joint_low_rank_posterior(
                    params, "mu", jp, split, transform=_pt_mu
                )
            elif "mu_W" in params:
                distributions["mu"] = _build_low_rank_positive_normal_posterior(
                    params,
                    "mu",
                    is_mixture,
                    split,
                    transform=_pt_mu,
                )
            else:
                distributions["mu"] = _build_positive_normal_posterior(
                    params,
                    "mu",
                    is_mixture,
                    split,
                    transform=_pt_mu,
                )
    else:
        if "phi" not in skip:
            distributions["phi"] = _build_betaprime_posterior(
                params, "phi", is_scalar=True, is_mixture=False, split=split
            )
        if "mu" not in skip:
            # Constrained low-rank guides store log-space MVN params
            # (e.g., log_mu_W), while unconstrained guides use mu_W.
            if "mu_W" in params or "log_mu_W" in params:
                distributions["mu"] = _build_low_rank_lognormal_posterior(
                    params, "mu", is_mixture, split
                )
            else:
                distributions["mu"] = _build_lognormal_posterior(
                    params, "mu", is_mixture, split
                )

    return distributions


def _build_mean_disp_posteriors(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    skip: Optional[set] = None,
    *,
    pos_transform=None,
) -> Dict[str, Any]:
    """Build posteriors for mean_disp parameterization (samples mu and r).

    Both ``mu`` and ``r`` are gene-specific positive parameters; ``p`` and
    ``phi`` are derived deterministics computed downstream, so they are NOT
    built here. ``mu`` and ``r`` may carry different positive transforms (e.g.
    ``positive_transform={"mean_expression": "exp"}`` leaves ``r`` on the
    softplus default), so each is resolved separately. Each parameter honors
    ``skip``, joint-group prefixes, and ``_W`` / ``log_*_W`` low-rank guides —
    so ``guide_rank`` / ``joint_params=["mu","r"]`` (with or without a rank)
    are both supported.
    """
    distributions = {}
    skip = skip or set()

    _pt_mu = _resolve_pos_transform_for(pos_transform, "mu")
    _pt_r = _resolve_pos_transform_for(pos_transform, "r")

    if unconstrained:
        for name, _pt in (("mu", _pt_mu), ("r", _pt_r)):
            if name in skip:
                continue
            jp = _find_joint_prefix(params, name)
            if jp:
                distributions[name] = _build_joint_low_rank_posterior(
                    params, name, jp, split, transform=_pt
                )
            elif f"{name}_W" in params:
                distributions[name] = (
                    _build_low_rank_positive_normal_posterior(
                        params,
                        name,
                        is_mixture,
                        split,
                        transform=_pt,
                    )
                )
            else:
                distributions[name] = _build_positive_normal_posterior(
                    params,
                    name,
                    is_mixture,
                    split,
                    transform=_pt,
                )
    else:
        for name in ("mu", "r"):
            if name in skip:
                continue
            # Constrained low-rank guides store log-space MVN params
            # (e.g., log_mu_W / log_r_W), while unconstrained guides use
            # mu_W / r_W.
            if f"{name}_W" in params or f"log_{name}_W" in params:
                distributions[name] = _build_low_rank_lognormal_posterior(
                    params, name, is_mixture, split
                )
            else:
                distributions[name] = _build_lognormal_posterior(
                    params, name, is_mixture, split
                )

    return distributions


def _build_two_state_posteriors(
    params: Dict[str, jnp.ndarray],
    unconstrained: bool,
    is_mixture: bool,
    low_rank: bool,
    split: bool,
    skip: Optional[set] = None,
    *,
    pos_transform=None,
    parameterization: Optional[Parameterization] = None,
) -> Dict[str, Any]:
    """Build posteriors for the TwoState (Poisson-Beta compound) family.

    All three per-gene parameters (``mu``, ``burst_size``, ``k_off``
    or their reparameterized equivalents) are positive-valued in
    unconstrained-Normal-with-softplus/exp form (configurable via
    ``positive_transform``).  The constrained fallback
    (``unconstrained=False``) uses LogNormal.

    Per-parameter dispatch mirrors the NBDM gene-level builder:
      1. ``_find_joint_prefix(params, name)`` hit → joint low-rank
         marginal via ``_build_joint_low_rank_posterior``.
      2. Standalone ``{name}_W`` (or ``log_{name}_W`` in the
         constrained path) present → ``_build_low_rank_*``.
      3. Otherwise → mean-field ``_build_positive_normal_posterior``
         (or ``_build_lognormal_posterior``).

    Mixture support
    ---------------
    When ``is_mixture=True``, parameters that are marked
    mixture-specific in the model config will have a leading component
    axis in their variational parameters.  The ``is_mixture`` flag is
    forwarded to the per-parameter builders so that the split
    utilities correctly produce per-component × per-gene posteriors.
    """
    distributions = {}
    skip = skip or set()
    del low_rank  # dispatch handles low-rank per-name

    # Detect the parameterization by which extras were sampled.
    # The variational guide writes ``{name}_loc`` (mean-field /
    # standalone low-rank) or ``joint_{group}_{name}_loc`` (joint
    # low-rank); checking for any key containing the name suffix
    # handles both.
    def _has(name: str) -> bool:
        suffix = f"_{name}_loc"
        for k in params:
            if k == f"{name}_loc":
                return True
            if k.startswith("joint_") and k.endswith(suffix):
                return True
            if k in (f"{name}_W", f"log_{name}_W"):
                return True
        return False

    # Prefer the parameterization enum when available: variational-key
    # sniffing is unreliable once a coordinate carries a horseshoe/NCP or
    # dataset hierarchy (its site is ``{name}_raw[_dataset]``, not
    # ``{name}_loc``), which would otherwise mis-detect the parameterization
    # (e.g. moment-delta read as mean-Fano -> KeyError on ``concentration``).
    if parameterization is not None:
        param_names = _twostate_param_names(parameterization)
    elif _has("inv_concentration"):
        # Moment-delta: samples (mu, excess_fano, inv_concentration).
        param_names = ("mu", "excess_fano", "inv_concentration")
    elif _has("excess_fano"):
        # Mean-Fano: samples (mu, excess_fano, concentration).
        param_names = ("mu", "excess_fano", "concentration")
    elif _has("switching_ratio"):
        # Ratio: samples (mu, burst_size, switching_ratio).
        param_names = ("mu", "burst_size", "switching_ratio")
    else:
        # Natural: samples (mu, burst_size, k_off).
        param_names = ("mu", "burst_size", "k_off")

    for name in param_names:
        if name in skip:
            continue
        jp = _find_joint_prefix(params, name)

        # ``inv_concentration`` lives in (0, 1) under a sigmoid
        # transform — different from the other TwoState extras
        # which are positive-real under softplus/exp.  Dispatch to
        # the sigmoid-normal builder regardless of ``unconstrained``.
        if name == "inv_concentration":
            if jp is not None:
                distributions[name] = _build_joint_low_rank_posterior(
                    params,
                    name,
                    jp,
                    split,
                    transform=dist.transforms.SigmoidTransform(),
                )
            elif f"{name}_W" in params:
                distributions[name] = (
                    _build_low_rank_positive_normal_posterior(
                        params,
                        name,
                        is_mixture=is_mixture,
                        split=split,
                        transform=dist.transforms.SigmoidTransform(),
                    )
                )
            else:
                distributions[name] = _build_sigmoid_normal_posterior(
                    params,
                    name,
                    is_scalar=False,
                    split=split,
                )
            continue

        # Resolve the positive transform for THIS parameter.  When
        # ``pos_transform`` is the per-name resolver callable (dict
        # form of ``ModelConfig.positive_transform``), call it with
        # the current name; otherwise it is already a single Transform.
        _pt_name = _resolve_pos_transform_for(pos_transform, name)

        if unconstrained:
            if jp is not None:
                distributions[name] = _build_joint_low_rank_posterior(
                    params,
                    name,
                    jp,
                    split,
                    transform=_pt_name,
                )
            elif f"{name}_W" in params:
                distributions[name] = (
                    _build_low_rank_positive_normal_posterior(
                        params,
                        name,
                        is_mixture=is_mixture,
                        split=split,
                        transform=_pt_name,
                    )
                )
            else:
                distributions[name] = _build_positive_normal_posterior(
                    params,
                    name,
                    is_mixture=is_mixture,
                    split=split,
                    transform=_pt_name,
                )
        else:
            if jp is not None:
                distributions[name] = _build_joint_low_rank_posterior(
                    params, name, jp, split, transform=None
                )
            elif f"log_{name}_W" in params:
                distributions[name] = _build_low_rank_lognormal_posterior(
                    params, name, is_mixture=is_mixture, split=split
                )
            else:
                distributions[name] = _build_lognormal_posterior(
                    params, name, is_mixture=is_mixture, split=split
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
    """Build Gamma posteriors for NEG hyperparameters.

    The NEG prior uses psi (per-gene variance) and zeta (per-gene rate).
    Both have Gamma variational posteriors (conjugate match), parameterised
    by concentration and rate.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Guide parameters.  Must contain ``psi_{prefix}_concentration``,
        ``psi_{prefix}_rate``, ``zeta_{prefix}_concentration``,
        ``zeta_{prefix}_rate``.
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
        concentration = params[f"{name}_concentration"]
        rate = params[f"{name}_rate"]
        distributions[name] = dist.Gamma(concentration, rate)

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
    *,
    transform=None,
) -> Dict[str, Any]:
    """Build posterior for capture odds ratio parameter."""
    distributions = {}

    if unconstrained:
        distributions["phi_capture"] = _build_positive_normal_posterior(
            params,
            "phi_capture",
            is_mixture=False,
            split=split,
            is_scalar=False,
            transform=transform,
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

    Handles both:
    - **Hierarchical mu_eta** (``mu_eta_pop_loc`` present): builds
      posteriors for ``mu_eta_pop``, shrinkage auxiliaries, and
      ``mu_eta_raw``.
    - **Single-scalar mu_eta** (``mu_eta_loc`` present): backward-
      compatible path for D <= 1 or old checkpoints.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Optimized variational parameters.
    model_config : ModelConfig
        Model configuration.
    split : bool
        If True, split per-cell distributions.

    Returns
    -------
    Dict[str, Any]
        Posterior distributions for ``eta_capture`` and ``mu_eta``
        (and optional hierarchical auxiliaries).
    """
    distributions: Dict[str, Any] = {}

    # --- Hierarchical per-dataset mu_eta ----------------------------------
    if "mu_eta_pop_loc" in params:
        # Population mean (unconstrained real)
        distributions["mu_eta_pop"] = dist.Normal(
            params["mu_eta_pop_loc"], params["mu_eta_pop_scale"]
        )

        # Gaussian: tau_eta (Softplus-transformed)
        if "tau_eta_loc" in params:
            distributions["tau_eta"] = dist.TransformedDistribution(
                dist.Normal(params["tau_eta_loc"], params["tau_eta_scale"]),
                dist.transforms.SoftplusTransform(),
            )

        # Horseshoe auxiliaries
        if "tau_mu_eta_loc" in params:
            distributions["tau_mu_eta"] = dist.TransformedDistribution(
                dist.Normal(
                    params["tau_mu_eta_loc"],
                    params["tau_mu_eta_scale"],
                ),
                dist.transforms.SoftplusTransform(),
            )
        if "lambda_mu_eta_loc" in params:
            distributions["lambda_mu_eta"] = dist.TransformedDistribution(
                dist.Normal(
                    params["lambda_mu_eta_loc"],
                    params["lambda_mu_eta_scale"],
                ),
                dist.transforms.SoftplusTransform(),
            )
        if "c_sq_mu_eta_loc" in params:
            distributions["c_sq_mu_eta"] = dist.TransformedDistribution(
                dist.Normal(
                    params["c_sq_mu_eta_loc"],
                    params["c_sq_mu_eta_scale"],
                ),
                dist.transforms.SoftplusTransform(),
            )

        # NEG auxiliaries
        if "zeta_mu_eta_loc" in params:
            distributions["zeta_mu_eta"] = dist.TransformedDistribution(
                dist.Normal(
                    params["zeta_mu_eta_loc"],
                    params["zeta_mu_eta_scale"],
                ),
                dist.transforms.SoftplusTransform(),
            )
        if "psi_mu_eta_loc" in params:
            distributions["psi_mu_eta"] = dist.TransformedDistribution(
                dist.Normal(
                    params["psi_mu_eta_loc"],
                    params["psi_mu_eta_scale"],
                ),
                dist.transforms.SoftplusTransform(),
            )

        # NCP deviations (shared across prior types)
        if "mu_eta_raw_loc" in params:
            distributions["mu_eta_raw"] = dist.Normal(
                params["mu_eta_raw_loc"], params["mu_eta_raw_scale"]
            )

    # --- Single-scalar mu_eta (D<=1 fallback or old checkpoint) -----------
    elif "mu_eta_loc" in params:
        distributions["mu_eta"] = dist.Normal(
            params["mu_eta_loc"], params["mu_eta_scale"]
        )

    # --- Per-cell eta_capture ------------------------------------------------
    # Two guide parameterizations: softplus-normal (new, detected by
    # eta_capture_raw_loc) and truncated-normal (legacy, eta_capture_loc).
    if "eta_capture_raw_loc" in params:
        # Softplus-normal guide: eta = softplus(raw), raw ~ Normal
        raw_loc = params["eta_capture_raw_loc"]
        raw_scale = params["eta_capture_raw_scale"]
        if split:
            distributions["eta_capture"] = [
                dist.TransformedDistribution(
                    dist.Normal(raw_loc[i], raw_scale[i]),
                    dist.transforms.SoftplusTransform(),
                )
                for i in range(raw_loc.shape[0])
            ]
        else:
            distributions["eta_capture"] = dist.TransformedDistribution(
                dist.Normal(raw_loc, raw_scale),
                dist.transforms.SoftplusTransform(),
            )
    elif "eta_capture_loc" in params:
        # Legacy truncated-normal guide: eta ~ TruncatedNormal(low=0)
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


def _build_low_rank_positive_normal_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    is_mixture: bool,
    split: bool,
    transform=None,
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """
    Build low-rank TransformedDistribution posterior for positive-valued params.

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
    transform : numpyro Transform, optional
        Transform mapping unconstrained reals to (0, inf).  When ``None``
        (backward compat), falls back to ``ExpTransform`` for positive
        params or ``SigmoidTransform`` for probability params.

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

    # Use caller-supplied transform, or fall back to name-based default
    if transform is None:
        if name in ["p", "gate", "p_capture"]:
            transform = dist.transforms.SigmoidTransform()
        else:
            transform = dist.transforms.ExpTransform()

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


def _build_positive_normal_posterior(
    params: Dict[str, jnp.ndarray],
    name: str,
    is_mixture: bool,
    split: bool,
    is_scalar: bool = False,
    transform=None,
) -> Union[dist.Distribution, List[dist.Distribution]]:
    """Build transformed Normal posterior for (0, inf) support.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Dictionary of variational parameters.
    name : str
        Parameter name (e.g., "r", "mu", "phi").
    is_mixture : bool
        Whether this is a mixture model.
    split : bool
        Whether to split into per-gene distributions.
    is_scalar : bool, optional
        Whether this is a scalar parameter.
    transform : numpyro Transform, optional
        Transform mapping unconstrained reals to (0, inf).  When ``None``
        (backward compat), defaults to ``ExpTransform``.
    """
    loc = params[f"{name}_loc"]
    scale = params[f"{name}_scale"]

    if transform is None:
        transform = dist.transforms.ExpTransform()

    base = dist.Normal(loc, scale)
    posterior = dist.TransformedDistribution(base, transform)

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
