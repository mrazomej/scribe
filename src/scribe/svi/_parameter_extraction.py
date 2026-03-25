"""
Parameter extraction mixin for SVI results.

This mixin provides methods for extracting parameters from variational
distributions, including MAP estimates and canonical parameter conversions.
"""

from typing import Dict, Optional, Any, Union, List, Set
import warnings

import jax.numpy as jnp
from jax.nn import sigmoid, softmax
import numpyro.distributions as dist
from jax import random

from ..utils import numpyro_to_scipy
from ..models.config.enums import HierarchicalPriorType
from ..models.parameterizations import (
    _align_gene_params,
    _broadcast_scalar_for_mixture,
)
from ..flows import ComponentFlowDistribution, FlowDistribution


# ==============================================================================
# Horseshoe NCP MAP Reconstruction
# ==============================================================================


def _horseshoe_eff_scale(
    tau: jnp.ndarray,
    lam: jnp.ndarray,
    c_sq: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the regularized horseshoe effective scale.

    Parameters
    ----------
    tau : jnp.ndarray
        Global shrinkage (scalar).
    lam : jnp.ndarray
        Per-gene local scales.
    c_sq : jnp.ndarray
        Slab parameter (scalar).

    Returns
    -------
    jnp.ndarray
        Effective scale ``tau * c * lambda / sqrt(c^2 + tau^2 * lambda^2)``.
    """
    c = jnp.sqrt(c_sq)
    return tau * c * lam / jnp.sqrt(c_sq + tau**2 * lam**2)


def _neg_eff_scale(psi: jnp.ndarray) -> jnp.ndarray:
    """Compute the NEG effective scale from psi.

    For the NEG (Normal-Exponential-Gamma) prior, the effective scale is
    simply sqrt(psi), where psi is the per-gene variance from the
    Gamma-Gamma hierarchy. This is much simpler than the horseshoe formula.

    Parameters
    ----------
    psi : jnp.ndarray
        Per-gene variance from the NEG hierarchy (positive).

    Returns
    -------
    jnp.ndarray
        Effective scale ``sqrt(psi)``.
    """
    return jnp.sqrt(psi)


def _reconstruct_horseshoe_maps(
    map_estimates: Dict[str, jnp.ndarray],
    model_config,
) -> Dict[str, jnp.ndarray]:
    """Reconstruct constrained MAP estimates from NCP horseshoe components.

    When a horseshoe prior with NCP is used, the MAP contains entries for
    ``{raw_name}`` (z), ``tau_{prefix}``, ``lambda_{prefix}``, and
    ``c_sq_{prefix}`` instead of the constrained parameter.  This function
    computes ``constrained = transform(hyper_loc + eff_scale * z)`` and
    injects it into the MAP dict.

    Parameters
    ----------
    map_estimates : Dict[str, jnp.ndarray]
        MAP estimates including raw z and horseshoe hyperparameters.
    model_config
        Model configuration with horseshoe flags.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Updated MAP with constrained parameters added.
    """
    configs = []

    # Gene-level horseshoe p
    if getattr(model_config, "horseshoe_p", False):
        parameterization = model_config.parameterization
        if parameterization in ("mean_odds", "odds_ratio"):
            configs.append(("phi_raw", "phi", "phi", "log_phi_loc", jnp.exp))
        else:
            configs.append(("p_raw", "p", "p", "logit_p_loc", sigmoid))

    # Gene-level horseshoe gate
    if getattr(model_config, "horseshoe_gate", False):
        configs.append(("gate_raw", "gate", "gate", "logit_gate_loc", sigmoid))

    # Gene-level horseshoe mu (across mixture components)
    if (
        getattr(model_config, "mu_prior", None)
        == HierarchicalPriorType.HORSESHOE
    ):
        parameterization = model_config.parameterization
        if parameterization in ("canonical", "standard"):
            configs.append(("r_raw", "r", "r", "log_r_loc", jnp.exp))
        else:
            configs.append(("mu_raw", "mu", "mu", "log_mu_loc", jnp.exp))

    # Dataset-level horseshoe mu
    if getattr(model_config, "horseshoe_dataset_mu", False):
        parameterization = model_config.parameterization
        if parameterization in ("canonical", "standard"):
            configs.append(
                ("r_raw", "r", "r_dataset", "log_r_dataset_loc", jnp.exp)
            )
        else:
            configs.append(
                ("mu_raw", "mu", "mu_dataset", "log_mu_dataset_loc", jnp.exp)
            )

    # Dataset-level horseshoe p
    if getattr(model_config, "horseshoe_dataset_p", False):
        parameterization = model_config.parameterization
        if parameterization in ("mean_odds", "odds_ratio"):
            configs.append(
                (
                    "phi_raw_dataset",
                    "phi",
                    "phi_dataset",
                    "log_phi_dataset_loc",
                    jnp.exp,
                )
            )
        else:
            configs.append(
                (
                    "p_raw_dataset",
                    "p",
                    "p_dataset",
                    "logit_p_dataset_loc",
                    sigmoid,
                )
            )

    # Dataset-level horseshoe gate
    if getattr(model_config, "horseshoe_dataset_gate", False):
        configs.append(
            (
                "gate_raw_dataset",
                "gate",
                "gate_dataset",
                "logit_gate_dataset_loc",
                sigmoid,
            )
        )

    for raw_name, target_name, hs_prefix, loc_name, transform in configs:
        if raw_name not in map_estimates:
            continue

        z = map_estimates[raw_name]
        tau = map_estimates.get(f"tau_{hs_prefix}")
        lam = map_estimates.get(f"lambda_{hs_prefix}")
        c_sq = map_estimates.get(f"c_sq_{hs_prefix}")
        loc = map_estimates.get(loc_name)

        if any(v is None for v in (tau, lam, c_sq, loc)):
            continue

        eff = _horseshoe_eff_scale(tau, lam, c_sq)
        unconstrained = loc + eff * z
        map_estimates[target_name] = transform(unconstrained)

    # BNB concentration uses a different naming convention
    # (bnb_concentration_tau, not tau_bnb_concentration).
    if (
        getattr(model_config, "is_bnb", False)
        and getattr(model_config, "overdispersion_prior", None)
        == HierarchicalPriorType.HORSESHOE
        and "bnb_concentration_raw" in map_estimates
    ):
        z = map_estimates["bnb_concentration_raw"]
        tau = map_estimates.get("bnb_concentration_tau")
        lam = map_estimates.get("bnb_concentration_lambda")
        c_sq = map_estimates.get("bnb_concentration_c_sq")
        loc = map_estimates.get("bnb_concentration_loc")

        if all(v is not None for v in (tau, lam, c_sq, loc)):
            from numpyro.distributions.transforms import SoftplusTransform

            # Global scalars (loc, tau, c_sq) may acquire a leading
            # dataset dimension (D,) after concat while gene-level
            # arrays (z, lam) have shape (D, G).  Expand the
            # lower-rank tensors so everything broadcasts correctly.
            ref_ndim = z.ndim
            if loc.ndim > 0 and loc.ndim < ref_ndim:
                loc = loc[..., jnp.newaxis]
            if tau.ndim > 0 and tau.ndim < ref_ndim:
                tau = tau[..., jnp.newaxis]
            if c_sq.ndim > 0 and c_sq.ndim < ref_ndim:
                c_sq = c_sq[..., jnp.newaxis]

            eff = _horseshoe_eff_scale(tau, lam, c_sq)
            unconstrained = loc + eff * z
            map_estimates["bnb_concentration"] = SoftplusTransform()(
                unconstrained
            )

    return map_estimates


# ==============================================================================
# NEG NCP MAP Reconstruction
# ==============================================================================


def _reconstruct_neg_maps(
    map_estimates: Dict[str, jnp.ndarray],
    model_config,
) -> Dict[str, jnp.ndarray]:
    """Reconstruct constrained MAP estimates from NCP NEG components.

    When an NEG prior with NCP is used, the MAP contains entries for
    ``{raw_name}`` (z) and ``psi_{prefix}`` instead of the constrained
    parameter.  This function computes ``constrained = transform(loc +
    eff_scale * z)`` where ``eff_scale = sqrt(psi)``, and injects it into
    the MAP dict.

    The NEG effective scale is simpler than the horseshoe: it is just
    sqrt(psi), with no tau/lambda/c_sq combination.

    Parameters
    ----------
    map_estimates : Dict[str, jnp.ndarray]
        MAP estimates including raw z and NEG psi hyperparameters.
        psi values come from a Gamma variational posterior and are already
        in the correct (positive) space — no exp() is needed.
    model_config
        Model configuration with p_prior, gate_prior, mu_dataset_prior,
        p_dataset_prior, gate_dataset_prior enum fields.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Updated MAP with constrained parameters added.
    """
    configs = []

    # Gene-level NEG p
    if model_config.p_prior == HierarchicalPriorType.NEG:
        parameterization = model_config.parameterization
        if parameterization in ("mean_odds", "odds_ratio"):
            configs.append(("phi_raw", "phi", "phi", "log_phi_loc", jnp.exp))
        else:
            configs.append(("p_raw", "p", "p", "logit_p_loc", sigmoid))

    # Gene-level NEG gate
    if model_config.gate_prior == HierarchicalPriorType.NEG:
        configs.append(("gate_raw", "gate", "gate", "logit_gate_loc", sigmoid))

    # Gene-level NEG mu (across mixture components)
    if getattr(model_config, "mu_prior", None) == HierarchicalPriorType.NEG:
        parameterization = model_config.parameterization
        if parameterization in ("canonical", "standard"):
            configs.append(("r_raw", "r", "r", "log_r_loc", jnp.exp))
        else:
            configs.append(("mu_raw", "mu", "mu", "log_mu_loc", jnp.exp))

    # Dataset-level NEG mu
    if model_config.mu_dataset_prior == HierarchicalPriorType.NEG:
        parameterization = model_config.parameterization
        if parameterization in ("canonical", "standard"):
            configs.append(
                ("r_raw", "r", "r_dataset", "log_r_dataset_loc", jnp.exp)
            )
        else:
            configs.append(
                ("mu_raw", "mu", "mu_dataset", "log_mu_dataset_loc", jnp.exp)
            )

    # Dataset-level NEG p
    if model_config.p_dataset_prior == HierarchicalPriorType.NEG:
        parameterization = model_config.parameterization
        if parameterization in ("mean_odds", "odds_ratio"):
            configs.append(
                (
                    "phi_raw_dataset",
                    "phi",
                    "phi_dataset",
                    "log_phi_dataset_loc",
                    jnp.exp,
                )
            )
        else:
            configs.append(
                (
                    "p_raw_dataset",
                    "p",
                    "p_dataset",
                    "logit_p_dataset_loc",
                    sigmoid,
                )
            )

    # Dataset-level NEG gate
    if model_config.gate_dataset_prior == HierarchicalPriorType.NEG:
        configs.append(
            (
                "gate_raw_dataset",
                "gate",
                "gate_dataset",
                "logit_gate_dataset_loc",
                sigmoid,
            )
        )

    for raw_name, target_name, neg_prefix, loc_name, transform in configs:
        if raw_name not in map_estimates:
            continue

        z = map_estimates[raw_name]
        psi = map_estimates.get(f"psi_{neg_prefix}")
        loc = map_estimates.get(loc_name)

        if psi is None or loc is None:
            continue

        # psi from Gamma variational posterior — already positive-valued;
        # MAP estimate is concentration/rate (the Gamma mean).
        eff_scale = _neg_eff_scale(psi)
        unconstrained = loc + eff_scale * z
        map_estimates[target_name] = transform(unconstrained)

    # BNB concentration uses a different naming convention
    # (bnb_concentration_psi, not psi_bnb_concentration).
    if (
        getattr(model_config, "is_bnb", False)
        and getattr(model_config, "overdispersion_prior", None)
        == HierarchicalPriorType.NEG
        and "bnb_concentration_raw" in map_estimates
    ):
        z = map_estimates["bnb_concentration_raw"]
        psi = map_estimates.get("bnb_concentration_psi")
        loc = map_estimates.get("bnb_concentration_loc")

        if psi is not None and loc is not None:
            from numpyro.distributions.transforms import SoftplusTransform

            # loc is a global scalar that may acquire a leading dataset
            # dimension (D,) after concat.  Expand it so it broadcasts
            # with the gene-level z of shape (D, G).
            if loc.ndim > 0 and loc.ndim < z.ndim:
                loc = loc[..., jnp.newaxis]

            eff_scale = _neg_eff_scale(psi)
            unconstrained = loc + eff_scale * z
            map_estimates["bnb_concentration"] = SoftplusTransform()(
                unconstrained
            )

    return map_estimates


# ==============================================================================
# Flow-guide MAP helpers
# ==============================================================================


def _detect_flow_params(distributions: Dict[str, Any]) -> set:
    """Return the set of parameter names backed by a FlowDistribution.

    A distribution entry is considered "flow-based" if:

    - It is a dict with a ``"base"`` whose type name contains
      ``"FlowDistribution"`` (avoids hard import of the flows module).
    - It is a ``TransformedDistribution`` whose ``base_dist`` is a
      ``FlowDistribution`` (or wraps one through ``Independent``).

    Parameters
    ----------
    distributions : dict
        Output of ``get_posterior_distributions`` / ``get_distributions``.

    Returns
    -------
    set of str
        Parameter names that need sampling-based MAP estimation.
    """
    flow_names: set = set()
    for name, obj in distributions.items():
        if name.startswith("joint:"):
            continue
        if _is_flow_dist(obj):
            flow_names.add(name)
    return flow_names


_FLOW_DIST_TYPES = (FlowDistribution, ComponentFlowDistribution)


def _is_flow_dist(obj) -> bool:
    """Check whether a distribution entry wraps a flow distribution.

    Recognises both ``FlowDistribution`` (scalar / non-mixture) and
    ``ComponentFlowDistribution`` (mixture / dataset-aware flows).
    """
    # Dict-style: {"base": FlowDistribution|ComponentFlowDistribution, ...}
    if isinstance(obj, dict) and "base" in obj:
        base = obj["base"]
        if isinstance(base, _FLOW_DIST_TYPES):
            return True
        # Independent wrapping (e.g. .to_event())
        if hasattr(base, "base_dist"):
            if isinstance(base.base_dist, _FLOW_DIST_TYPES):
                return True
    # TransformedDistribution wrapping
    if isinstance(obj, dist.TransformedDistribution):
        bd = obj.base_dist
        if isinstance(bd, _FLOW_DIST_TYPES):
            return True
        if hasattr(bd, "base_dist"):
            if isinstance(bd.base_dist, _FLOW_DIST_TYPES):
                return True
    return False


def _flow_map_estimates(
    model_and_guide_fn,
    params: Dict[str, jnp.ndarray],
    n_cells: int,
    n_genes: int,
    model_config,
    flow_params: set,
    method: str = "mean",
    n_samples: int = 1000,
    optimize_steps: int = 300,
    optimize_lr: float = 1e-3,
    counts=None,
    verbose: bool = True,
    batch_size: Optional[int] = None,
) -> Dict[str, jnp.ndarray]:
    """Compute MAP estimates for flow-guided parameters.

    Reconstructs the optimized guide via ``model_and_guide_fn`` and
    samples from it repeatedly.  The three strategies differ only in
    how the N samples are aggregated into a point estimate.

    Parameters
    ----------
    model_and_guide_fn : callable
        ``() -> (model, guide)`` factory (typically ``self._model_and_guide``).
    params : dict
        Full SVI parameter dict.
    n_cells, n_genes : int
        Data dimensions for the guide call.
    model_config : ModelConfig
        Model configuration.
    flow_params : set of str
        Names of flow-guided parameters to estimate.
    method : str
        ``"mean"``, ``"empirical"``, or ``"optimize"``.
    n_samples : int
        Number of guide samples for mean / empirical strategies.
    optimize_steps : int
        Adam steps for the optimize strategy.
    optimize_lr : float
        Adam learning rate for the optimize strategy.
    counts : array, optional
        Count matrix (forwarded to guide if needed).
    verbose : bool
        Whether to print progress info.
    batch_size : int, optional
        Mini-batch size for cell-level sampling inside the guide.
        ``None`` processes all cells at once.

    Returns
    -------
    Dict[str, jnp.ndarray]
        MAP estimates keyed by parameter name.
    """
    import numpyro.handlers
    import jax

    if method not in ("mean", "empirical", "optimize"):
        raise ValueError(
            f"flow_map_method must be 'mean', 'empirical', or 'optimize', "
            f"got '{method}'"
        )

    _, guide = model_and_guide_fn()
    if guide is None:
        return {}

    # ---- Collect N samples from the guide --------------------------------
    if verbose:
        print(f"Flow MAP ({method}): drawing {n_samples} guide samples...")

    samples_by_site = _flow_sample_guide(
        guide,
        params,
        n_cells,
        n_genes,
        model_config,
        flow_params,
        n_samples,
        counts,
        batch_size=batch_size,
    )

    if not samples_by_site:
        return {}

    # ---- Aggregate according to strategy ---------------------------------
    if method == "mean":
        return {
            name: jnp.mean(stacked, axis=0)
            for name, stacked in samples_by_site.items()
        }

    if method == "empirical":
        return _flow_empirical_mode(
            guide,
            params,
            n_cells,
            n_genes,
            model_config,
            samples_by_site,
            counts,
            batch_size=batch_size,
        )

    # method == "optimize"
    return _flow_optimize_mode(
        guide,
        params,
        n_cells,
        n_genes,
        model_config,
        samples_by_site,
        flow_params,
        optimize_steps,
        optimize_lr,
        counts,
        verbose,
        batch_size=batch_size,
    )


def _flow_sample_guide(
    guide,
    params,
    n_cells,
    n_genes,
    model_config,
    flow_params,
    n_samples,
    counts,
    batch_size=None,
) -> Dict[str, jnp.ndarray]:
    """Run the guide *n_samples* times and stack flow-param samples.

    When the guide includes cell-level parameters (e.g. amortized
    capture), ``batch_size`` controls how many cells are processed
    per guide call.  This avoids OOM for large datasets.

    Parameters
    ----------
    guide : callable
        Guide function.
    params : dict
        Optimized SVI parameters.
    n_cells, n_genes : int
        Data dimensions.
    model_config : ModelConfig
        Model configuration.
    flow_params : set of str
        Names of flow-guided sites to collect.
    n_samples : int
        Number of guide executions.
    counts : array, optional
        Count matrix forwarded to the guide.
    batch_size : int, optional
        Mini-batch size for cell-level sampling inside the guide.
        ``None`` (default) processes all cells at once.

    Returns
    -------
    Dict[str, jnp.ndarray]
        ``{name: array of shape (n_samples, *param_shape)}``.
    """
    import numpyro.handlers

    guide_kwargs = dict(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=model_config,
        counts=counts,
        batch_size=batch_size,
    )

    collected: Dict[str, list] = {name: [] for name in flow_params}

    for i in range(n_samples):
        key = random.PRNGKey(i)
        with numpyro.handlers.seed(rng_seed=key):
            with numpyro.handlers.substitute(data=params):
                with numpyro.handlers.trace() as tr:
                    guide(**guide_kwargs)

        for name in flow_params:
            if name in tr and "value" in tr[name]:
                collected[name].append(tr[name]["value"])

    # Stack collected samples
    result = {}
    for name, vals in collected.items():
        if vals:
            result[name] = jnp.stack(vals, axis=0)
    return result


def _flow_empirical_mode(
    guide,
    params,
    n_cells,
    n_genes,
    model_config,
    samples_by_site,
    counts,
    batch_size=None,
) -> Dict[str, jnp.ndarray]:
    """Pick the sample with highest joint log-density across flow params.

    Re-runs the guide for each sample to compute per-sample joint
    log-probability, then selects the sample index with the highest
    total.

    Parameters
    ----------
    batch_size : int, optional
        Mini-batch size for cell-level sampling inside the guide.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Best sample per flow-guided parameter.
    """
    import numpyro.handlers

    n_samples = next(iter(samples_by_site.values())).shape[0]
    guide_kwargs = dict(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=model_config,
        counts=counts,
        batch_size=batch_size,
    )

    log_probs = jnp.zeros(n_samples)
    for i in range(n_samples):
        key = random.PRNGKey(i)
        # Substitute both learned params and this particular sample
        sample_data = {**params}
        for name, stacked in samples_by_site.items():
            sample_data[name] = stacked[i]

        with numpyro.handlers.seed(rng_seed=key):
            with numpyro.handlers.substitute(data=sample_data):
                with numpyro.handlers.trace() as tr:
                    guide(**guide_kwargs)

        # Sum log_prob of flow-guided sites
        lp = 0.0
        for name in samples_by_site:
            if name in tr and "fn" in tr[name]:
                try:
                    lp = lp + tr[name]["fn"].log_prob(tr[name]["value"])
                except Exception:
                    pass
        log_probs = log_probs.at[i].set(lp)

    best_idx = int(jnp.argmax(log_probs))
    return {
        name: stacked[best_idx] for name, stacked in samples_by_site.items()
    }


def _flow_optimize_mode(
    guide,
    params,
    n_cells,
    n_genes,
    model_config,
    samples_by_site,
    flow_params,
    n_steps,
    lr,
    counts,
    verbose,
    batch_size=None,
) -> Dict[str, jnp.ndarray]:
    """Gradient-ascent mode finding starting from the sample mean.

    Runs the guide under ``trace`` and ``substitute`` to compute the
    joint log-density of the flow-guided sample sites, then optimizes
    the unconstrained values with Adam.

    Parameters
    ----------
    batch_size : int, optional
        Mini-batch size for cell-level sampling inside the guide.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Optimized MAP estimate per flow-guided parameter.
    """
    import numpyro.handlers

    try:
        import optax
    except ImportError:
        warnings.warn(
            "optax is required for flow_map_method='optimize'. "
            "Falling back to 'mean'.",
            UserWarning,
        )
        return {
            name: jnp.mean(stacked, axis=0)
            for name, stacked in samples_by_site.items()
        }

    import jax

    # Initialize from the sample mean
    init_values = {
        name: jnp.mean(stacked, axis=0)
        for name, stacked in samples_by_site.items()
    }

    guide_kwargs = dict(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=model_config,
        counts=counts,
        batch_size=batch_size,
    )

    def neg_log_prob(values):
        """Negative joint log-prob of the flow-guided sites."""
        sample_data = {**params, **values}
        with numpyro.handlers.seed(rng_seed=random.PRNGKey(0)):
            with numpyro.handlers.substitute(data=sample_data):
                with numpyro.handlers.trace() as tr:
                    guide(**guide_kwargs)

        total = 0.0
        for name in values:
            if name in tr and "fn" in tr[name]:
                total = total + tr[name]["fn"].log_prob(tr[name]["value"])
        return -total

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_values)
    current = init_values

    grad_fn = jax.grad(neg_log_prob)

    for step in range(n_steps):
        grads = grad_fn(current)
        updates, opt_state = optimizer.update(grads, opt_state, current)
        current = optax.apply_updates(current, updates)

    if verbose:
        print(f"Flow MAP optimize: completed {n_steps} steps.")

    return current


# ==============================================================================
# Targeted MAP dependency helpers
# ==============================================================================


def _normalize_map_targets(
    targets: Optional[Union[str, List[str], Set[str]]],
) -> Optional[Set[str]]:
    """Normalize requested MAP targets to a validated set.

    Parameters
    ----------
    targets : str or list of str or set of str, optional
        Requested output keys for selective MAP extraction.

    Returns
    -------
    set of str, optional
        Normalized key set, or ``None`` when full MAP output is requested.

    Raises
    ------
    TypeError
        If *targets* has an unsupported type or contains non-string entries.
    ValueError
        If *targets* is empty.
    """
    if targets is None:
        return None
    if isinstance(targets, str):
        normalized = {targets}
    elif isinstance(targets, (list, set, tuple)):
        normalized = set(targets)
    else:
        raise TypeError(
            "targets must be a string, a list/set/tuple of strings, or None."
        )

    if not normalized:
        raise ValueError("targets must contain at least one parameter name.")
    if not all(isinstance(name, str) for name in normalized):
        raise TypeError("All targets entries must be strings.")
    return normalized


def _required_map_keys_for_targets(
    targets: Optional[Set[str]],
) -> Optional[Set[str]]:
    """Expand requested MAP keys with derivation dependencies.

    This closure is intentionally conservative: for derived keys that can be
    computed via more than one pathway (for example ``r`` from ``mu`` + ``p`` or
    ``mu`` + ``phi``), all candidate parents are included. This ensures
    selective mode has enough ingredients while still enabling flow pruning.

    Parameters
    ----------
    targets : set of str, optional
        Requested output keys. ``None`` means full-map mode.

    Returns
    -------
    set of str, optional
        Required keys for selective extraction. ``None`` in full-map mode.
    """
    if targets is None:
        return None

    # Dependency graph for keys derived in get_map /
    # _compute_canonical_parameters. Values list *candidate* parents, not
    # necessarily all required at runtime.
    deps = {
        "r": {"mu", "p", "phi", "r_unconstrained"},
        "p": {"phi", "p_unconstrained"},
        "mu": {"r", "p"},
        "p_capture": {
            "phi_capture",
            "p_capture_unconstrained",
            "eta_capture",
        },
        "phi_capture": {"eta_capture"},
        "p_hat": {"p", "p_capture"},
        "bnb_kappa": {"bnb_concentration", "r", "mu", "phi"},
        # Horseshoe / NEG reconstructions can synthesize constrained params.
        "bnb_concentration": {
            "bnb_concentration_raw",
            "bnb_concentration_tau",
            "bnb_concentration_lambda",
            "bnb_concentration_c_sq",
            "bnb_concentration_loc",
            "bnb_concentration_psi",
        },
    }

    required = set(targets)
    queue = list(targets)
    while queue:
        key = queue.pop()
        for parent in deps.get(key, set()):
            if parent not in required:
                required.add(parent)
                queue.append(parent)
    return required


# ==============================================================================
# Parameter Extraction Mixin
# ==============================================================================


class ParameterExtractionMixin:
    """Mixin providing parameter extraction methods."""

    # --------------------------------------------------------------------------
    # Helper methods for amortizer validation
    # --------------------------------------------------------------------------

    def _uses_amortized_capture(self) -> bool:
        """Check if this model uses amortized capture probability."""
        guide_families = self.model_config.guide_families
        if (
            guide_families is None
            or not self.model_config.uses_variable_capture
        ):
            return False
        amort_config = getattr(guide_families, "capture_amortization", None)
        return amort_config is not None and getattr(
            amort_config, "enabled", False
        )

    def _validate_counts_for_amortizer(
        self, counts, context: str = "sampling"
    ) -> None:
        """Validate that counts have the correct shape for amortized capture.

        When using amortized capture probability, the amortizer computes sufficient
        statistics (e.g., total UMI count) by summing across ALL genes. If genes
        have been subset from the results, the counts must still have the original
        number of genes to ensure the amortizer receives the correct statistics.

        Parameters
        ----------
        counts : jnp.ndarray
            Count matrix to validate.
        context : str
            Description of the operation for error messages.

        Raises
        ------
        ValueError
            If counts shape doesn't match expected dimensions.
        """
        if counts is None:
            return

        # Get the expected gene count for amortizer input
        original_n_genes = (
            getattr(self, "_original_n_genes", None) or self.n_genes
        )

        # Check counts shape
        if counts.ndim != 2:
            raise ValueError(
                f"counts must be a 2D array of shape (n_cells, n_genes), "
                f"got shape {counts.shape}"
            )

        n_cells_counts, n_genes_counts = counts.shape

        # Check cell dimension
        if n_cells_counts != self.n_cells:
            raise ValueError(
                f"counts has {n_cells_counts} cells but model was trained with "
                f"{self.n_cells} cells. The counts matrix must have the same "
                f"number of cells as used during training."
            )

        # Check gene dimension - this is the critical check for amortizers
        if n_genes_counts != original_n_genes:
            # Determine if this is a gene-subset scenario
            is_subset = (
                getattr(self, "_original_n_genes", None) is not None
                and self.n_genes < original_n_genes
            )

            if is_subset:
                raise ValueError(
                    f"counts has {n_genes_counts} genes but the amortizer requires "
                    f"{original_n_genes} genes (the original training data shape). "
                    f"This results object has been subset to {self.n_genes} genes, "
                    f"but the amortizer computes sufficient statistics (e.g., total "
                    f"UMI count) by summing across ALL genes. You must pass the "
                    f"original full-gene count matrix for {context}, not a gene-subset."
                )
            else:
                raise ValueError(
                    f"counts has {n_genes_counts} genes but model was trained with "
                    f"{original_n_genes} genes. The counts matrix must have the same "
                    f"dimensions as used during training for {context}."
                )

    # --------------------------------------------------------------------------
    # Get distributions using configs
    # --------------------------------------------------------------------------

    def get_distributions(
        self,
        backend: str = "numpyro",
        split: bool = False,
        counts: Optional[jnp.ndarray] = None,
        params: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Get the variational distributions for all parameters.

        This method uses the composable builder system to extract posterior
        distributions from the optimized variational parameters. It
        automatically handles all parameterizations, constrained/unconstrained
        variants, and guide families including normalizing-flow guides.

        For flow-guided parameters, the returned entry is a dict
        ``{"base": FlowDistribution, "transform": ...}``.  Joint-flow
        (dense) parameters additionally include ``"conditional": True``
        to flag that they represent conditional distributions from the
        chain-rule decomposition and require guide execution for proper
        joint sampling.

        Parameters
        ----------
        backend : str, default="numpyro"
            Statistical package to use for distributions. Must be one of:
            - "scipy": Returns scipy.stats distributions
            - "numpyro": Returns numpyro.distributions
        split : bool, default=False
            If True, returns lists of individual distributions for
            multidimensional parameters instead of batch distributions.
        counts : Optional[jnp.ndarray], optional
            Observed count matrix of shape (n_cells, n_genes). Required when
            using amortized capture probability. For non-amortized models, this
            can be None. Default: None.
        params : Optional[Dict[str, jnp.ndarray]], optional
            Optional params dictionary to use instead of self.params. Used
            internally when amortized parameters need to be computed. Default:
            None (uses self.params).

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to their distributions.
            Flow-guided parameters are wrapped in
            ``{"base": FlowDistribution, "transform": ...}`` dicts.

        Raises
        ------
        ValueError
            If backend is not supported.
        """
        if backend not in ["scipy", "numpyro"]:
            raise ValueError(f"Invalid backend: {backend}")

        # Use provided params or self.params
        params_to_use = params if params is not None else self.params

        # Use the new composable builder system for posterior extraction
        from ..models.builders import get_posterior_distributions

        distributions = get_posterior_distributions(
            params_to_use, self.model_config, split=split
        )

        if backend == "scipy":
            # Handle conversion to scipy, accounting for split distributions
            scipy_distributions = {}
            for name, dist_obj in distributions.items():
                # Joint distributions are numpyro-only metadata dicts
                if name.startswith("joint:"):
                    continue
                if isinstance(dist_obj, list):
                    # Handle split distributions - convert each element
                    if all(isinstance(sublist, list) for sublist in dist_obj):
                        # Handle nested lists (2D case: components × genes)
                        scipy_distributions[name] = [
                            [numpyro_to_scipy(d) for d in sublist]
                            for sublist in dist_obj
                        ]
                    else:
                        # Handle simple lists (1D case: genes or components)
                        scipy_distributions[name] = [
                            numpyro_to_scipy(d) for d in dist_obj
                        ]
                else:
                    # Handle single distribution
                    scipy_distributions[name] = numpyro_to_scipy(dist_obj)
            return scipy_distributions

        return distributions

    # --------------------------------------------------------------------------

    def get_map(
        self,
        targets: Optional[Union[str, List[str], Set[str]]] = None,
        use_mean: bool = False,
        canonical: bool = True,
        verbose: bool = True,
        counts: Optional[jnp.ndarray] = None,
        flow_map_method: str = "mean",
        flow_n_samples: int = 1000,
        flow_optimize_steps: int = 300,
        flow_optimize_lr: float = 1e-3,
        flow_batch_size: Optional[int] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational
        posterior.

        Parameters
        ----------
        targets : str or list of str, optional
            Optional parameter key(s) to return. When provided, ``get_map``
            computes only what is needed for these targets and returns a
            filtered dictionary with those keys only.

            Examples:
            - ``get_map("p_capture")``
            - ``get_map(["r", "p"])``

            Backward compatibility note: passing a boolean as the first
            positional argument is interpreted as ``use_mean`` from the
            legacy signature.
        use_mean : bool, default=False
            If True, replaces undefined MAP values (NaN) with posterior means
        canonical : bool, default=True
            If True, includes canonical parameters (p, r) computed from other
            parameters for linked, odds_ratio, and unconstrained
            parameterizations
        verbose : bool, default=True
            If True, prints a warning if NaNs were replaced with means
        counts : Optional[jnp.ndarray], optional
            Observed count matrix of shape (n_cells, n_genes). Required when
            using amortized capture probability (e.g., with
            amortization.capture.enabled=true).

            IMPORTANT: When using amortized capture with gene-subset results,
            you must pass the ORIGINAL full-gene count matrix, not a gene-subset.
            The amortizer computes sufficient statistics (e.g., total UMI count)
            by summing across ALL genes, so it requires the full data.

            For non-amortized models, this can be None. Default: None.
        flow_map_method : str, default="mean"
            Strategy for MAP estimation of flow-guided parameters.
            Ignored when no normalizing-flow guides are present.

            - ``"mean"``: Sample ``flow_n_samples`` points from the
              guide and average them.  Fast; gives the posterior mean.
            - ``"empirical"``: Sample ``flow_n_samples`` points, evaluate
              each sample's joint log-probability, and pick the sample
              with highest density.  Medium cost; approximate mode.
            - ``"optimize"``: Initialize from the sample mean, then run
              gradient ascent on the guide's log-density for
              ``flow_optimize_steps`` steps.  Highest quality mode
              estimate.
        flow_n_samples : int, default=1000
            Number of samples drawn from the guide for the ``"mean"``
            and ``"empirical"`` strategies.
        flow_optimize_steps : int, default=300
            Number of Adam steps for the ``"optimize"`` strategy.
        flow_optimize_lr : float, default=1e-3
            Learning rate for the ``"optimize"`` strategy.
        flow_batch_size : int, optional
            Mini-batch size for cell-level sampling inside the guide
            during flow MAP estimation.  When the guide includes
            cell-specific parameters (e.g. amortized capture), setting
            this avoids processing all cells on every guide call.
            ``None`` (default) processes all cells at once, which is
            fine when the guide only has gene-level flow parameters.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary of MAP estimates. In selective mode (``targets`` set),
            only requested keys are returned.
        """
        # Backward compatibility shim for the legacy positional signature:
        # get_map(True, canonical=...) where the first argument was use_mean.
        if isinstance(targets, bool):
            if use_mean is not False:
                raise TypeError(
                    "Ambiguous call: boolean first argument is interpreted as "
                    "legacy use_mean. Please pass either targets=... or "
                    "use_mean=..., not both."
                )
            use_mean = targets
            targets = None

        requested_targets = _normalize_map_targets(targets)
        required_keys = _required_map_keys_for_targets(requested_targets)

        # For amortized capture, we need to compute variational parameters
        # by running the amortizer with counts, then add them to params
        params = self.params

        # Check if amortized capture is enabled
        has_amortized_capture = self._uses_amortized_capture()

        # Error handling: if amortized capture is enabled, counts are required
        if has_amortized_capture and counts is None:
            raise ValueError(
                "counts parameter is required when using amortized capture "
                "probability (amortization.capture.enabled=true). "
                "Please provide the observed count matrix of shape "
                "(n_cells, n_genes) that was used during inference."
            )

        # Validate counts shape for amortizer (checks original gene count)
        if has_amortized_capture and counts is not None:
            self._validate_counts_for_amortizer(
                counts, context="MAP estimation"
            )

        if counts is not None and self.model_config.uses_variable_capture:
            # Check if amortized capture is enabled (already checked above)
            if has_amortized_capture:
                # Amortized capture is enabled - need to compute variational
                # parameters by running the guide with counts and extracting
                # the distribution parameters from the trace
                # We use a custom handler to capture the distribution parameters
                import numpyro.handlers

                # Run guide with trace to extract distribution parameters
                _, guide = self._model_and_guide()
                if guide is not None:
                    dummy_key = random.PRNGKey(0)
                    with numpyro.handlers.seed(rng_seed=dummy_key):
                        with numpyro.handlers.substitute(data=params):
                            with numpyro.handlers.trace() as tr:
                                guide(
                                    n_cells=self.n_cells,
                                    n_genes=self.n_genes,
                                    model_config=self.model_config,
                                    counts=counts,
                                    batch_size=None,
                                )

                    # Extract distribution parameters from trace
                    # The trace contains the distribution objects in the 'fn' field
                    from ..models.parameterizations import PARAMETERIZATIONS

                    param_strategy = PARAMETERIZATIONS[
                        self.model_config.parameterization
                    ]
                    capture_param_name = param_strategy.transform_model_param(
                        "p_capture"
                    )

                    # Look for the capture parameter in the trace
                    if capture_param_name in tr:
                        dist_obj = tr[capture_param_name].get("fn")
                        if dist_obj is not None:
                            # Extract parameters based on distribution type
                            unconstrained = self.model_config.unconstrained
                            if unconstrained:
                                # TransformedDistribution with Normal base
                                if hasattr(dist_obj, "base_dist"):
                                    base = dist_obj.base_dist
                                    if hasattr(base, "loc") and hasattr(
                                        base, "scale"
                                    ):
                                        loc_key = f"{capture_param_name}_loc"
                                        scale_key = (
                                            f"{capture_param_name}_scale"
                                        )
                                        params = {**params}
                                        params[loc_key] = base.loc
                                        params[scale_key] = base.scale
                            else:
                                # BetaPrime distribution
                                if hasattr(
                                    dist_obj, "concentration1"
                                ) and hasattr(dist_obj, "concentration0"):
                                    alpha_key = f"{capture_param_name}_alpha"
                                    beta_key = f"{capture_param_name}_beta"
                                    params = {**params}
                                    params[alpha_key] = dist_obj.concentration1
                                    params[beta_key] = dist_obj.concentration0

        # Get distributions with NumPyro backend
        distributions = self.get_distributions(
            backend="numpyro", counts=counts, params=params
        )

        # Detect which parameters are flow-guided (their distributions
        # are backed by a FlowDistribution which lacks .loc / .mode).
        flow_params = _detect_flow_params(distributions)
        flow_targets = (
            flow_params
            if required_keys is None
            else flow_params.intersection(required_keys)
        )

        # ----------------------------------------------------------
        # Standard MAP extraction for non-flow parameters
        # ----------------------------------------------------------
        map_estimates = {}
        for param, dist_obj in distributions.items():
            # Skip full joint distributions (keyed as "joint:{group}");
            # individual per-parameter marginals are already in the dict.
            if param.startswith("joint:"):
                continue

            # Defer flow-guided params to the sampling-based handler
            if param in flow_params:
                continue

            # Handle transformed distributions (dict with 'base' and
            # 'transform') This is used for low-rank guides with transformations
            if (
                isinstance(dist_obj, dict)
                and "base" in dist_obj
                and "transform" in dist_obj
            ):
                # For transformed distributions, MAP is transform(base.loc)
                base_dist = dist_obj["base"]
                transform = dist_obj["transform"]
                if hasattr(base_dist, "loc"):
                    map_estimates[param] = transform(base_dist.loc)
                else:
                    # Fallback to mean if loc not available
                    map_estimates[param] = transform(base_dist.mean)
            # Handle TransformedDistribution objects (for low-rank guides) Check
            # if it's a TransformedDistribution wrapping
            # LowRankMultivariateNormal
            elif isinstance(dist_obj, dist.TransformedDistribution) and hasattr(
                dist_obj, "base_dist"
            ):
                base_dist = dist_obj.base_dist
                # Get transform from transforms list (usually first element)
                transform = (
                    dist_obj.transforms[0] if dist_obj.transforms else None
                )
                if transform is not None and hasattr(base_dist, "loc"):
                    # For LowRankMultivariateNormal, MAP is transform(loc)
                    map_estimates[param] = transform(base_dist.loc)
                elif hasattr(base_dist, "loc"):
                    # If no transform, just use loc
                    map_estimates[param] = base_dist.loc
                else:
                    # Fallback: try to get mean from base distribution
                    try:
                        base_mean = base_dist.mean
                        if transform is not None:
                            map_estimates[param] = transform(base_mean)
                        else:
                            map_estimates[param] = base_mean
                    except (NotImplementedError, AttributeError):
                        # If mean not available, use the distribution's mean if
                        # it exists
                        try:
                            map_estimates[param] = dist_obj.mean
                        except NotImplementedError:
                            # Last resort: use base.loc if available
                            if hasattr(base_dist, "loc"):
                                if transform is not None:
                                    map_estimates[param] = transform(
                                        base_dist.loc
                                    )
                                else:
                                    map_estimates[param] = base_dist.loc
                            else:
                                raise ValueError(
                                    f"Cannot compute MAP for {param}: "
                                    "distribution does not support mean or loc"
                                )
            # Handle multivariate distributions (like LowRankMultivariateNormal)
            # For multivariate normals, mode = mean = loc
            elif hasattr(dist_obj, "loc") and not hasattr(dist_obj, "mode"):
                map_estimates[param] = dist_obj.loc
            elif hasattr(dist_obj, "mode"):
                map_estimates[param] = dist_obj.mode
            else:
                map_estimates[param] = dist_obj.mean

        # ----------------------------------------------------------
        # Flow MAP estimation via guide sampling
        # ----------------------------------------------------------
        if flow_targets:
            # Gene-subsetted results must run the flow guide at full
            # dimensionality, then slice the output to the subset.
            # Use the original (unsubsetted) params so nondense
            # regression coefficients match the flow output shape.
            _orig_ng = getattr(self, "_original_n_genes", None)
            _gene_idx = getattr(self, "_subset_gene_index", None)
            _orig_params = getattr(self, "_original_params", None)
            _full_dim_flow = (
                _orig_ng is not None
                and _orig_ng != self.n_genes
                and _gene_idx is not None
            )
            n_genes_for_flow = _orig_ng if _full_dim_flow else self.n_genes
            params_for_flow = (
                _orig_params
                if (_full_dim_flow and _orig_params is not None)
                else params
            )

            flow_estimates = _flow_map_estimates(
                model_and_guide_fn=self._model_and_guide,
                params=params_for_flow,
                n_cells=self.n_cells,
                n_genes=n_genes_for_flow,
                model_config=self.model_config,
                flow_params=flow_targets,
                method=flow_map_method,
                n_samples=flow_n_samples,
                optimize_steps=flow_optimize_steps,
                optimize_lr=flow_optimize_lr,
                counts=counts,
                verbose=verbose,
                batch_size=flow_batch_size,
            )

            if _full_dim_flow:
                from ._sampling_posterior_predictive import (
                    _subset_gene_dim_samples,
                )

                flow_estimates = _subset_gene_dim_samples(
                    flow_estimates, _gene_idx, _orig_ng
                )

            map_estimates.update(flow_estimates)

        # Replace NaN values with means if requested
        if use_mean:
            # Initialize boolean to track if any NaNs were replaced
            replaced_nans = False
            # Check each parameter for NaNs and replace with means
            for param, value in map_estimates.items():
                # Check if any values are NaN
                if jnp.any(jnp.isnan(value)):
                    replaced_nans = True
                    # Compute mean, handling dict-structured joint
                    # distributions ({"base": ..., "transform": ...})
                    # that lack a .mean attribute directly.
                    dist_obj = distributions[param]
                    if (
                        isinstance(dist_obj, dict)
                        and "base" in dist_obj
                        and "transform" in dist_obj
                    ):
                        mean_value = dist_obj["transform"](
                            dist_obj["base"].mean
                        )
                    else:
                        mean_value = dist_obj.mean
                    # Replace NaN values with means
                    map_estimates[param] = jnp.where(
                        jnp.isnan(value), mean_value, value
                    )
            # Print warning if NaNs were replaced
            if replaced_nans and verbose:
                warnings.warn(
                    "NaN values were replaced with means of the distributions",
                    UserWarning,
                )

        # Reconstruct constrained parameters from NCP horseshoe if applicable
        map_estimates = _reconstruct_horseshoe_maps(
            map_estimates, self.model_config
        )

        # Reconstruct constrained parameters from NCP NEG if applicable
        map_estimates = _reconstruct_neg_maps(map_estimates, self.model_config)

        # Derive kappa_g from omega_g (bnb_concentration) when available.
        # kappa_g = 2 + (r + 1) / omega_g, matching build_bnb_dist.
        if "bnb_concentration" in map_estimates:
            omega = map_estimates["bnb_concentration"]
            r_for_kappa = map_estimates.get("r")
            if r_for_kappa is None:
                # mean_odds/mean_prob: derive r from mu and phi (or p)
                mu = map_estimates.get("mu")
                phi = map_estimates.get("phi")
                if mu is not None and phi is not None:
                    r_for_kappa = mu * phi
            if r_for_kappa is not None:
                omega_safe = jnp.clip(omega, 1e-6, None)
                # omega is (…, G) but r may have extra axes such as a
                # mixture-component dimension (…, K, G).  Expand omega
                # so that it broadcasts correctly.
                while omega_safe.ndim < r_for_kappa.ndim:
                    omega_safe = jnp.expand_dims(omega_safe, axis=-2)
                map_estimates["bnb_kappa"] = (
                    2.0 + (r_for_kappa + 1.0) / omega_safe
                )

        # Compute canonical parameters if requested
        if canonical:
            map_estimates = self._compute_canonical_parameters(
                map_estimates, verbose=verbose
            )

        # In selective mode, return only the requested keys with a clear error
        # when any requested key is unavailable under the current model setup.
        if requested_targets is not None:
            missing = sorted(
                key for key in requested_targets if key not in map_estimates
            )
            if missing:
                available = ", ".join(sorted(map_estimates.keys()))
                raise ValueError(
                    "Requested MAP parameter(s) are unavailable: "
                    f"{missing}. Available keys: [{available}]. "
                    "If you are requesting derived canonical parameters, "
                    "ensure canonical=True."
                )
            return {key: map_estimates[key] for key in requested_targets}

        return map_estimates

    # --------------------------------------------------------------------------

    def _compute_canonical_parameters(
        self, map_estimates: Dict, verbose: bool = True
    ) -> Dict:
        """
        Compute canonical parameters from other parameters for different
        parameterizations.

        Parameters
        ----------
        map_estimates : Dict
            Dictionary containing MAP estimates
        verbose : bool, default=True
            If True, prints information about parameter computation

        Returns
        -------
        Dict
            Updated dictionary with canonical parameters computed.  For
            canonical/standard parameterization this may also include
            deterministic ``mu`` derived from ``p`` and ``r``.
        """
        estimates = map_estimates.copy()
        parameterization = self.model_config.parameterization
        unconstrained = self.model_config.unconstrained

        # Handle linked / mean_prob parameterization
        if parameterization in (
            "linked",
            "mean_prob",
        ):
            if "mu" in estimates and "p" in estimates and "r" not in estimates:
                if verbose:
                    print(
                        "Computing r from mu and p for "
                        f"{parameterization} parameterization"
                    )
                # r = mu * (1 - p) / p
                p = estimates["p"]
                # For standard (non-hierarchical) mixture models, p is
                # (n_components,) and needs reshaping to broadcast with
                # mu of shape (n_components, n_genes).  For hierarchical
                # models, p is already (n_genes,) or (n_components,
                # n_genes) and broadcasts element-wise with mu.
                # NOTE: mixture_params=None means *all* params are
                # mixture-specific (the default), so we treat None the
                # same as the param being listed explicitly.
                _mp = self.model_config.mixture_params
                _p_is_mixture = _mp is None or "p" in _mp
                if (
                    self.n_components is not None
                    and _p_is_mixture
                    and p.ndim == 1
                    and p.shape[0] == self.n_components
                ):
                    # Mixture model: mu has shape (n_components, n_genes)
                    # p has shape (n_components,). Reshape for broadcasting.
                    p_reshaped = p[:, None]
                else:
                    p_reshaped = p

                # Scalar-per-dataset p has shape (n_datasets,) while mu
                # is (n_datasets, n_genes).  Reshape to (n_datasets, 1)
                # so the trailing dimension broadcasts against n_genes.
                _n_ds = getattr(self.model_config, "n_datasets", None)
                if (
                    _n_ds is not None
                    and p_reshaped.ndim == 1
                    and p_reshaped.shape[0] == _n_ds
                    and estimates["mu"].ndim >= 2
                ):
                    p_reshaped = p_reshaped[:, None]

                # Mixture+dataset scalar p: shape (K, D) while mu is
                # (K, D, G).  Add trailing singleton for broadcasting.
                mu_shape = estimates["mu"]
                if (
                    p_reshaped.ndim >= 1
                    and mu_shape.ndim > p_reshaped.ndim
                    and mu_shape.shape[: p_reshaped.ndim] == p_reshaped.shape
                ):
                    p_reshaped = p_reshaped[..., None]

                # Align intermediate dims when p and mu are both
                # gene-specific but differ in dataset dimension, e.g.
                # p=(K, G) vs mu=(K, D, G).
                p_reshaped, mu_aligned = _align_gene_params(
                    p_reshaped, estimates["mu"]
                )
                estimates["r"] = mu_aligned * (1 - p_reshaped) / p_reshaped

        # Handle odds_ratio / mean_odds parameterization
        elif parameterization in (
            "odds_ratio",
            "mean_odds",
        ):
            # Convert phi to p if needed
            if "phi" in estimates and "p" not in estimates:
                if verbose:
                    print(
                        "Computing p from phi for "
                        f"{parameterization} parameterization"
                    )
                estimates["p"] = 1.0 / (1.0 + estimates["phi"])

            # Convert phi and mu to r if needed
            if (
                "phi" in estimates
                and "mu" in estimates
                and "r" not in estimates
            ):
                if verbose:
                    print(
                        "Computing r from phi and mu for "
                        f"{parameterization} parameterization"
                    )
                # For standard (non-hierarchical) mixture models, phi is
                # (n_components,) and needs reshaping.  For hierarchical
                # models, phi is already (n_genes,) or (n_components,
                # n_genes) and broadcasts element-wise with mu.
                # NOTE: mixture_params=None means *all* params are
                # mixture-specific (see note above for p).
                _mp = self.model_config.mixture_params
                _phi_is_mixture = _mp is None or "phi" in _mp
                if (
                    self.n_components is not None
                    and _phi_is_mixture
                    and estimates["phi"].ndim == 1
                    and estimates["phi"].shape[0] == self.n_components
                ):
                    # Mixture model: mu has shape (n_components, n_genes)
                    phi_reshaped = estimates["phi"][:, None]
                else:
                    phi_reshaped = estimates["phi"]

                # Scalar-per-dataset phi has shape (n_datasets,) while mu
                # is (n_datasets, n_genes).  Reshape to (n_datasets, 1)
                # so the trailing dimension broadcasts against n_genes.
                _n_ds = getattr(self.model_config, "n_datasets", None)
                if (
                    _n_ds is not None
                    and phi_reshaped.ndim == 1
                    and phi_reshaped.shape[0] == _n_ds
                    and estimates["mu"].ndim >= 2
                ):
                    phi_reshaped = phi_reshaped[:, None]

                # Mixture+dataset scalar phi: shape (K, D) while mu is
                # (K, D, G).  Add trailing singleton for broadcasting.
                mu_shape = estimates["mu"]
                if (
                    phi_reshaped.ndim >= 1
                    and mu_shape.ndim > phi_reshaped.ndim
                    and mu_shape.shape[: phi_reshaped.ndim]
                    == phi_reshaped.shape
                ):
                    phi_reshaped = phi_reshaped[..., None]

                # Align intermediate dims when phi and mu are both
                # gene-specific but differ in dataset dimension, e.g.
                # phi=(K, G) vs mu=(K, D, G).
                phi_reshaped, mu_aligned = _align_gene_params(
                    phi_reshaped, estimates["mu"]
                )
                estimates["r"] = mu_aligned * phi_reshaped

            # Handle VCP capture probability conversion
            if "phi_capture" in estimates and "p_capture" not in estimates:
                if verbose:
                    print(
                        "Computing p_capture from phi_capture for "
                        f"{parameterization} parameterization"
                    )
                estimates["p_capture"] = 1.0 / (1.0 + estimates["phi_capture"])

        # Handle unconstrained parameterization
        if unconstrained:
            # Convert r_unconstrained to r if needed
            if "r_unconstrained" in estimates and "r" not in estimates:
                if verbose:
                    print(
                        "Computing r from r_unconstrained for "
                        "unconstrained parameterization"
                    )
                estimates["r"] = jnp.exp(estimates["r_unconstrained"])

            # Convert p_unconstrained to p if needed
            if "p_unconstrained" in estimates and "p" not in estimates:
                if verbose:
                    print(
                        "Computing p from p_unconstrained for "
                        "unconstrained parameterization"
                    )
                estimates["p"] = sigmoid(estimates["p_unconstrained"])

            # Convert gate_unconstrained to gate if needed
            if "gate_unconstrained" in estimates and "gate" not in estimates:
                if verbose:
                    print(
                        "Computing gate from gate_unconstrained for "
                        "unconstrained parameterization"
                    )
                estimates["gate"] = sigmoid(estimates["gate_unconstrained"])

            # Handle VCP capture probability conversion
            if (
                "p_capture_unconstrained" in estimates
                and "p_capture" not in estimates
            ):
                if verbose:
                    print(
                        "Computing p_capture from p_capture_unconstrained for "
                        "unconstrained parameterization"
                    )
                estimates["p_capture"] = sigmoid(
                    estimates["p_capture_unconstrained"]
                )
            # Handle mixing weights computation for mixture models
            if (
                "mixing_logits_unconstrained" in estimates
                and "mixing_weights" not in estimates
            ):
                # Compute mixing weights from mixing_logits_unconstrained using
                # softmax
                estimates["mixing_weights"] = softmax(
                    estimates["mixing_logits_unconstrained"], axis=-1
                )

        # Canonical/standard parameterization may only have p and r in the MAP.
        # Add deterministic mu so downstream code can rely on a mean parameter
        # regardless of the sampling parameterization.
        if (
            parameterization in ("canonical", "standard")
            and "r" in estimates
            and "p" in estimates
            and "mu" not in estimates
        ):
            if verbose:
                print(
                    "Computing mu from r and p for "
                    f"{parameterization} parameterization"
                )
            # Reuse shared broadcasting/alignment helpers to support
            # scalar/per-component/per-dataset and gene-specific layouts.
            p_for_mu = _broadcast_scalar_for_mixture(
                estimates["p"], estimates["r"]
            )
            # Special-case dataset+gene p against mixture+dataset+gene r:
            # p=(D,G), r=(K,D,G) should become p=(1,D,G) so broadcasting
            # happens across components.  Generic gene-axis alignment inserts
            # singleton dims before the gene axis, which would yield (D,1,G).
            if (
                p_for_mu.ndim + 1 == estimates["r"].ndim
                and p_for_mu.shape == estimates["r"].shape[1:]
            ):
                p_for_mu = jnp.expand_dims(p_for_mu, axis=0)
            p_for_mu, r_aligned = _align_gene_params(p_for_mu, estimates["r"])
            # Guard against numerical blow-ups when p is very close to 1.
            one_minus_p = jnp.clip(1.0 - p_for_mu, 1e-8, None)
            estimates["mu"] = r_aligned * p_for_mu / one_minus_p

        # Convert eta_capture to capture parameter (biology-informed prior)
        if "eta_capture" in estimates:
            eta = estimates["eta_capture"]
            if "phi_capture" not in estimates:
                estimates["phi_capture"] = jnp.exp(eta) - 1.0
            if "p_capture" not in estimates:
                estimates["p_capture"] = jnp.exp(-eta)
            if verbose:
                print(
                    "Computing p_capture and phi_capture from eta_capture "
                    "(biology-informed prior)"
                )

        # Compute p_hat for NBVCP and ZINBVCP models if needed (applies to all
        # parameterizations)
        if (
            "p" in estimates
            and "p_capture" in estimates
            and "p_hat" not in estimates
        ):
            p_val = estimates["p"]

            # For hierarchical models p has shape (n_components, n_genes)
            # or (K, D, G) for multi-dataset hierarchical models.
            # Pre-computing p_hat would require a huge per-cell tensor —
            # too large to materialise here.  The cell-batched sampling
            # code handles the broadcast per-batch instead, so we skip.
            # Guard: skip whenever p has >=2 dimensions with all sizes >1,
            # since broadcasting with per-cell p_capture would either fail
            # or produce a massive intermediate tensor.
            p_is_high_dim = p_val.ndim >= 2 and all(s > 1 for s in p_val.shape)

            if p_is_high_dim:
                if verbose:
                    print(
                        "Skipping p_hat precomputation (gene-specific p "
                        f"with shape {p_val.shape}; handled per-batch)"
                    )
            else:
                if verbose:
                    print("Computing p_hat from p and p_capture")

                # p_capture is (n_cells,); reshape to (n_cells, 1) for
                # broadcasting against p which may be scalar, (n_genes,),
                # or (n_components,).
                p_capture_reshaped = estimates["p_capture"][:, None]

                # p_hat = p * p_capture / (1 - p * (1 - p_capture))
                p_hat_raw = (
                    p_val
                    * p_capture_reshaped
                    / (1 - p_val * (1 - p_capture_reshaped))
                )

                # When p is scalar or per-component (not gene-specific),
                # p_hat is per-cell only → flatten to (n_cells,).
                # When p is gene-specific, p_hat is (n_cells, n_genes).
                if p_val.ndim == 0 or (
                    p_val.ndim == 1
                    and self.n_components is not None
                    and p_val.shape[0] == self.n_components
                ):
                    estimates["p_hat"] = p_hat_raw.flatten()
                else:
                    estimates["p_hat"] = p_hat_raw

        return estimates
