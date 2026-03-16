"""
Parameter extraction mixin for SVI results.

This mixin provides methods for extracting parameters from variational
distributions, including MAP estimates and canonical parameter conversions.
"""

from typing import Dict, Optional, Any
import warnings

import jax.numpy as jnp
from jax.nn import sigmoid, softmax
import numpyro.distributions as dist
from jax import random

from ..utils import numpyro_to_scipy
from ..models.config.enums import HierarchicalPriorType
from ..models.parameterizations import _align_gene_params


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

    return map_estimates


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
        variants, and guide families.

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
        use_mean: bool = False,
        canonical: bool = True,
        verbose: bool = True,
        counts: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational
        posterior.

        Parameters
        ----------
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

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary of MAP estimates for each parameter
        """
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
        # Get estimate of map
        map_estimates = {}
        for param, dist_obj in distributions.items():
            # Skip full joint distributions (keyed as "joint:{group}");
            # individual per-parameter marginals are already in the dict.
            if param.startswith("joint:"):
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

        # Compute canonical parameters if requested
        if canonical:
            map_estimates = self._compute_canonical_parameters(
                map_estimates, verbose=verbose
            )

        return map_estimates

    # --------------------------------------------------------------------------

    def _compute_canonical_parameters(
        self, map_estimates: Dict, verbose: bool = True
    ) -> Dict:
        """
        Compute canonical parameters (p, r) from other parameters for different
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
            Updated dictionary with canonical parameters computed
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
