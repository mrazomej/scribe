"""Unified model factory for creating SCRIBE models.

This module provides a single entry point for creating model and guide
functions from a ModelConfig. It replaces the individual preset factories
(create_nbdm, create_zinb, create_nbvcp, create_zinbvcp) with a unified
approach using registries and helper builders.

Functions
---------
create_model
    Create model and guide functions from a ModelConfig.
create_model_from_params
    Create model and guide functions from individual parameters.

Examples
--------
>>> from scribe.models.config import ModelConfig
>>> from scribe.models.presets.factory import create_model
>>>
>>> # Create model from config
>>> model, guide = create_model(model_config)
>>>
>>> # Or use the convenience function with individual params
>>> model, guide = create_model_from_params(
...     model="zinb",
...     parameterization="linked",
...     unconstrained=True,
... )
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpyro

from ..builders import GuideBuilder, ModelBuilder
from ..builders.parameter_specs import (
    ExpNormalSpec,
    GaussianLatentSpec,
    HierarchicalExpNormalSpec,
    HierarchicalSigmoidNormalSpec,
    NormalWithTransformSpec,
    SoftplusNormalSpec,
)
from ..components.guide_families import VAELatentGuide
from ..components.vae_components import (
    DecoderOutputHead,
    GaussianEncoder,
    MultiHeadDecoder,
)
from ..config import GuideFamilyConfig, ModelConfig
from ..config.enums import InferenceMethod
from ..config.enums import Parameterization as ParamEnum
from ..config.groups import VAEConfig
from ..parameterizations import PARAMETERIZATIONS
from scribe.flows import FlowChain
from .registry import (
    LIKELIHOOD_REGISTRY,
    MODEL_EXTRA_PARAMS,
    apply_prior_guide_overrides,
    build_extra_param_spec,
)

# Map VAEConfig flow_type to FlowChain flow_type
_FLOW_TYPE_MAP = {
    "coupling_affine": "affine_coupling",
    "coupling_spline": "spline_coupling",
    "maf": "maf",
    "iaf": "iaf",
}

# ==============================================================================
# Model/Guide Validation
# ==============================================================================


def validate_model_guide_compatibility(
    model: Callable,
    guide: Callable,
    model_config: ModelConfig,
    n_cells: int = 10,
    n_genes: int = 5,
) -> None:
    """Validate that model and guide are compatible by performing a dry run.

    This function runs both the model and guide with small synthetic dimensions
    to verify they work correctly together. It checks that:
    - Both model and guide execute without errors
    - All sample sites in the guide have corresponding sites in the model

    Parameters
    ----------
    model : Callable
        NumPyro model function.
    guide : Callable
        NumPyro guide function.
    model_config : ModelConfig
        Model configuration (passed to model/guide).
    n_cells : int, default=10
        Number of cells for dry run.
    n_genes : int, default=5
        Number of genes for dry run.

    Raises
    ------
    RuntimeError
        If the model or guide fails to execute.
    ValueError
        If guide sample sites don't match model sample sites.

    Examples
    --------
    >>> model, guide = create_model(config, validate=False)
    >>> validate_model_guide_compatibility(model, guide, config)
    """
    # Run model to get sample sites
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.trace() as model_trace:
            try:
                model(
                    n_cells=n_cells,
                    n_genes=n_genes,
                    model_config=model_config,
                    counts=None,  # Prior predictive mode
                )
            except Exception as e:
                raise RuntimeError(
                    f"Model validation failed during dry run: {e}"
                ) from e

    # Extract model sample sites (excluding observed sites and deterministic)
    model_sample_sites = {
        name
        for name, site in model_trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }

    # Run guide to get sample sites
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.trace() as guide_trace:
            try:
                guide(
                    n_cells=n_cells,
                    n_genes=n_genes,
                    model_config=model_config,
                    counts=None,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Guide validation failed during dry run: {e}"
                ) from e

    # Extract guide sample sites
    guide_sample_sites = {
        name for name, site in guide_trace.items() if site["type"] == "sample"
    }

    # Check that guide sites are a subset of model sites
    # (guide doesn't need to cover observed or derived parameters)
    extra_guide_sites = guide_sample_sites - model_sample_sites
    if extra_guide_sites:
        raise ValueError(
            f"Guide has sample sites not found in model: {extra_guide_sites}. "
            f"Model sample sites: {model_sample_sites}"
        )


# ==============================================================================
# VAE Model Factory
# ==============================================================================


def _create_vae_model(
    model_config: ModelConfig,
    priors: Optional[Dict[str, Tuple[float, ...]]] = None,
    guides: Optional[Dict[str, Tuple[float, ...]]] = None,
    validate: bool = True,
    n_genes: int = 0,
) -> Tuple[Callable, Callable, List]:
    """
    Create VAE model and guide from ModelConfig with composable architecture.
    """
    vae = model_config.vae
    base_model = model_config.base_model
    unconstrained = model_config.unconstrained
    param_key = _get_parameterization_key(model_config.parameterization)
    param_strategy = PARAMETERIZATIONS[param_key]
    guide_families = model_config.guide_families or GuideFamilyConfig()

    # 1. Build decoder output heads from parameterization (with optional
    #    overrides)
    output_spec = param_strategy.decoder_output_spec(base_model)
    if vae.decoder_transforms:
        output_spec = [
            (name, vae.decoder_transforms.get(name, transform))
            for name, transform in output_spec
        ]
    output_heads = tuple(
        DecoderOutputHead(name, n_genes, transform)
        for name, transform in output_spec
    )

    # 2. Build encoder
    encoder = GaussianEncoder(
        input_dim=n_genes,
        latent_dim=vae.latent_dim,
        hidden_dims=vae.encoder_hidden_dims,
        activation=vae.activation,
        input_transformation=vae.input_transform,
    )

    # 3. Build decoder
    decoder = MultiHeadDecoder(
        output_dim=0,
        latent_dim=vae.latent_dim,
        hidden_dims=vae.decoder_hidden_dims,
        output_heads=output_heads,
        activation=vae.activation,
    )

    # 4. Build flow (if needed) and latent spec
    flow = None
    if vae.flow_type != "none":
        flow_type = _FLOW_TYPE_MAP.get(vae.flow_type, vae.flow_type)
        flow = FlowChain(
            features=vae.latent_dim,
            num_layers=vae.flow_num_layers,
            flow_type=flow_type,
            hidden_dims=vae.flow_hidden_dims,
        )
    latent_spec = GaussianLatentSpec(
        latent_dim=vae.latent_dim,
        sample_site="z",
        flow=flow,
    )

    # 5. Build VAELatentGuide
    vae_guide = VAELatentGuide(
        encoder=encoder,
        decoder=decoder,
        latent_spec=latent_spec,
    )
    decoder_param_names = set(vae_guide.param_names)

    # 6. Build param specs: latent marker + non-decoder core + non-decoder extra
    core_specs = param_strategy.build_param_specs(
        unconstrained=unconstrained,
        guide_families=guide_families,
    )
    non_decoder_specs = [
        s for s in core_specs if s.name not in decoder_param_names
    ]

    # Latent marker spec (cell-specific, carries VAELatentGuide)
    latent_marker = ExpNormalSpec(
        name="z",
        shape_dims=("n_genes",),
        default_params=(0.0, 1.0),
        is_cell_specific=True,
        guide_family=vae_guide,
    )

    # Extra params (gate, p_capture) that aren't decoder outputs
    extra_param_names = MODEL_EXTRA_PARAMS.get(base_model, [])
    extra_specs = []
    for param_name in extra_param_names:
        transformed = param_strategy.transform_model_param(param_name)
        if transformed not in decoder_param_names:
            specs = build_extra_param_spec(
                param_name=param_name,
                unconstrained=unconstrained,
                guide_families=guide_families,
                param_strategy=param_strategy,
                hierarchical_gate=model_config.hierarchical_gate,
            )
            extra_specs.extend(specs)

    param_specs = [latent_marker] + non_decoder_specs + extra_specs

    # 7. Split derived params into pre-plate and in-plate
    all_derived = param_strategy.build_derived_params()
    pre_plate_derived = []
    in_plate_derived = []
    for d in all_derived:
        if any(dep in decoder_param_names for dep in d.deps):
            in_plate_derived.append(d)
        else:
            pre_plate_derived.append(d)

    # 8. Apply prior/guide overrides
    merged_priors = _extract_priors_from_param_specs(model_config.param_specs)
    if priors:
        merged_priors.update(priors)
    merged_guides = _extract_guides_from_param_specs(model_config.param_specs)
    if guides:
        merged_guides.update(guides)
    if merged_priors or merged_guides:
        param_specs = apply_prior_guide_overrides(
            param_specs,
            priors=merged_priors or None,
            guides=merged_guides or None,
        )

    # 9. Build model
    likelihood_class = LIKELIHOOD_REGISTRY[base_model]
    if base_model in ("nbvcp", "zinbvcp"):
        capture_param_name = param_strategy.transform_model_param("p_capture")
        likelihood_instance = likelihood_class(
            capture_param_name=capture_param_name
        )
    else:
        likelihood_instance = likelihood_class()

    model_builder = ModelBuilder()
    for spec in param_specs:
        model_builder.add_param(spec)
    for d in pre_plate_derived:
        model_builder.add_derived(d.name, d.compute, d.deps)
    model_builder.set_vae_in_plate_derived(in_plate_derived)
    model_builder.with_likelihood(likelihood_instance)
    model = model_builder.build()

    # 10. Build guide
    guide = GuideBuilder().from_specs(param_specs).build()

    # 11. Validate (optional) — VAE guide requires counts; use dummy data
    if validate:
        import jax.numpy as jnp

        dummy_counts = jnp.zeros((10, n_genes))
        try:
            with numpyro.handlers.seed(rng_seed=0):
                model(
                    n_cells=10,
                    n_genes=n_genes,
                    model_config=model_config,
                    counts=dummy_counts,
                )
            with numpyro.handlers.seed(rng_seed=0):
                guide(
                    n_cells=10,
                    n_genes=n_genes,
                    model_config=model_config,
                    counts=dummy_counts,
                )
        except Exception as e:
            raise RuntimeError(
                f"VAE model/guide validation failed during dry run: {e}"
            ) from e

    return model, guide, param_specs


# ==============================================================================
# Unified Model Factory
# ==============================================================================


def create_model(
    model_config: ModelConfig,
    priors: Optional[Dict[str, Tuple[float, ...]]] = None,
    guides: Optional[Dict[str, Tuple[float, ...]]] = None,
    validate: bool = True,
    n_genes: Optional[int] = None,
) -> Tuple[Callable, Callable, List]:
    """Create model and guide functions from a ModelConfig.

    This is the unified factory that replaces create_nbdm, create_zinb,
    create_nbvcp, and create_zinbvcp. It uses registries to determine
    model-specific components and builds the model/guide using the
    composable builder pattern.

    By default, the factory validates that the model and guide are compatible
    by performing a dry run. This catches configuration errors early. Set
    ``validate=False`` to skip validation (e.g., for performance).

    Parameters
    ----------
    model_config : ModelConfig
        Fully configured model configuration object containing:
        - base_model: Model type (nbdm, zinb, nbvcp, zinbvcp)
        - parameterization: Parameterization scheme
        - unconstrained: Whether to use unconstrained distributions
        - n_components: Number of mixture components (optional)
        - mixture_params: Which params are mixture-specific (optional)
        - guide_families: Per-parameter guide family config (optional)
    priors : Dict[str, Tuple[float, ...]], optional
        User-provided prior hyperparameters keyed by parameter name.
        Overrides default priors. Example: {"p": (2.0, 2.0)}
    guides : Dict[str, Tuple[float, ...]], optional
        User-provided guide hyperparameters keyed by parameter name.
    validate : bool, default=True
        If True, perform a dry run to validate model and guide compatibility.
        This catches configuration errors early but adds a small overhead.
        Set to False for performance-critical code or when validation has
        already been done. Note: validation is skipped for MCMC inference
        since MCMC doesn't use a guide.

    Returns
    -------
    model : Callable
        NumPyro model function with signature:
        model(n_cells, n_genes, model_config, counts=None, batch_size=None)
    guide : Callable
        NumPyro guide function with the same signature.
    param_specs : List
        The parameter specifications used to build the model and guide.
        Callers can attach these to model_config (e.g. for results subsetting).

    Raises
    ------
    ValueError
        If model type or parameterization is not recognized.
        If parameter names in priors/guides are not valid for this model.
    RuntimeError
        If validation fails (model or guide doesn't run correctly).

    Examples
    --------
    >>> from scribe.models.config import ModelConfigBuilder
    >>> from scribe.models.presets.factory import create_model
    >>>
    >>> # Build config and create model
    >>> config = (
    ...     ModelConfigBuilder()
    ...     .for_model("zinb")
    ...     .with_parameterization("linked")
    ...     .build()
    ... )
    >>> model, guide, param_specs = create_model(config)
    >>>
    >>> # With custom priors
    >>> model, guide = create_model(
    ...     config,
    ...     priors={"p": (2.0, 2.0), "mu": (1.0, 0.5)},
    ... )

    See Also
    --------
    create_model_from_params : Convenience function with flat parameters.
    """
    # ==========================================================================
    # Step 0: VAE path — use composable factory when inference is VAE
    # ==========================================================================
    if (
        model_config.inference_method == InferenceMethod.VAE
        and model_config.vae is not None
    ):
        if n_genes is None:
            raise ValueError("n_genes is required for VAE inference")
        return _create_vae_model(
            model_config=model_config,
            priors=priors,
            guides=guides,
            validate=validate,
            n_genes=n_genes,
        )

    # ==========================================================================
    # Step 1: Validate and get parameterization strategy
    # ==========================================================================
    param_key = _get_parameterization_key(model_config.parameterization)
    if param_key not in PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown parameterization: {model_config.parameterization}. "
            f"Supported: {list(PARAMETERIZATIONS.keys())}"
        )
    param_strategy = PARAMETERIZATIONS[param_key]

    # Validate model type
    base_model = model_config.base_model
    if base_model not in MODEL_EXTRA_PARAMS:
        raise ValueError(
            f"Unknown model type: {base_model}. "
            f"Supported: {list(MODEL_EXTRA_PARAMS.keys())}"
        )

    # ==========================================================================
    # Step 2: Validate mixture_params if provided
    # ==========================================================================
    if model_config.mixture_params is not None:
        _validate_mixture_params(
            mixture_params=model_config.mixture_params,
            param_strategy=param_strategy,
            base_model=base_model,
        )

    # ==========================================================================
    # Step 3: Resolve guide families
    # ==========================================================================
    guide_families = model_config.guide_families or GuideFamilyConfig()

    # ==========================================================================
    # Step 4: Build core parameter specs from parameterization strategy
    # ==========================================================================
    # When stored param_specs are present (i.e. this is a loaded / already-
    # trained model), derive mixture_params from those specs rather than
    # relying on the current code default.  This preserves the exact
    # is_mixture flags used during training, which determines parameter shapes
    # (scalar vs. per-component), and prevents guide/params mismatches when
    # the default mixture_params list changes between code versions.
    effective_mixture_params = model_config.mixture_params
    if model_config.param_specs and model_config.n_components is not None:
        stored_mixture_params = [
            spec.name
            for spec in model_config.param_specs
            if spec.is_mixture
        ]
        if stored_mixture_params:
            effective_mixture_params = stored_mixture_params

    param_specs = param_strategy.build_param_specs(
        unconstrained=model_config.unconstrained,
        guide_families=guide_families,
        n_components=model_config.n_components,
        mixture_params=effective_mixture_params,
    )

    # ==========================================================================
    # Step 4.5: Apply hierarchical_p flag (replace flat p/phi with triplet)
    # ==========================================================================
    if model_config.hierarchical_p:
        param_specs = _hierarchicalize_p(
            param_specs=param_specs,
            param_key=param_key,
            guide_families=guide_families,
            n_components=model_config.n_components,
            mixture_params=effective_mixture_params,
        )

    # ==========================================================================
    # Step 5: Add model-specific extra parameters
    # ==========================================================================
    extra_param_names = MODEL_EXTRA_PARAMS[base_model]
    for param_name in extra_param_names:
        extra_specs = build_extra_param_spec(
            param_name=param_name,
            unconstrained=model_config.unconstrained,
            guide_families=guide_families,
            param_strategy=param_strategy,
            n_components=model_config.n_components,
            mixture_params=effective_mixture_params,
            hierarchical_gate=model_config.hierarchical_gate,
        )
        param_specs.extend(extra_specs)

    # ==========================================================================
    # Step 6: Apply user-provided prior/guide overrides
    # ==========================================================================
    # Merge priors from model_config.param_specs with explicit priors argument
    merged_priors = _extract_priors_from_param_specs(model_config.param_specs)
    if priors:
        merged_priors.update(priors)

    merged_guides = _extract_guides_from_param_specs(model_config.param_specs)
    if guides:
        merged_guides.update(guides)

    if merged_priors or merged_guides:
        param_specs = apply_prior_guide_overrides(
            param_specs,
            priors=merged_priors or None,
            guides=merged_guides or None,
        )

    # ==========================================================================
    # Step 7: Get derived parameters from parameterization strategy
    # ==========================================================================
    derived_params = param_strategy.build_derived_params()

    # ==========================================================================
    # Step 8: Build model using ModelBuilder
    # ==========================================================================
    model_builder = ModelBuilder()
    for spec in param_specs:
        model_builder.add_param(spec)
    for d_param in derived_params:
        model_builder.add_derived(d_param.name, d_param.compute, d_param.deps)

    # Get likelihood from registry
    # For VCP models, pass the capture parameter name from the parameterization
    likelihood_class = LIKELIHOOD_REGISTRY[base_model]
    if base_model in ("nbvcp", "zinbvcp"):
        # Get the transformed capture param name (p_capture or phi_capture)
        capture_param_name = param_strategy.transform_model_param("p_capture")
        model_builder.with_likelihood(
            likelihood_class(capture_param_name=capture_param_name)
        )
    else:
        model_builder.with_likelihood(likelihood_class())

    model = model_builder.build()

    # ==========================================================================
    # Step 9: Build guide using GuideBuilder
    # ==========================================================================
    guide = GuideBuilder().from_specs(param_specs).build()

    # ==========================================================================
    # Step 10: Validate model/guide compatibility (optional)
    # ==========================================================================
    if validate:
        # Only validate guide for SVI/VAE - MCMC doesn't use a guide
        needs_guide = model_config.inference_method != InferenceMethod.MCMC
        # Skip validation for mixture models - they have dynamic guide structure
        is_mixture = model_config.n_components is not None
        # Skip validation if any guide uses amortization (requires actual data)
        has_amortized = _has_amortized_guide(model_config.guide_families)
        if needs_guide and not is_mixture and not has_amortized:
            validate_model_guide_compatibility(model, guide, model_config)

    return model, guide, param_specs


# ------------------------------------------------------------------------------


def create_model_from_params(
    model: str,
    parameterization: str = "canonical",
    unconstrained: bool = False,
    guide_families: Optional[GuideFamilyConfig] = None,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    priors: Optional[Dict[str, Tuple[float, ...]]] = None,
    guides: Optional[Dict[str, Tuple[float, ...]]] = None,
    validate: bool = True,
) -> Tuple[Callable, Callable]:
    """Create model and guide functions from individual parameters.

    This is a convenience function that creates a ModelConfig internally and
    calls create_model(). It provides the same interface as the old preset
    factories but routes through the unified factory.

    Parameters
    ----------
    model : str
        Model type: "nbdm", "zinb", "nbvcp", or "zinbvcp".
    parameterization : str, default="canonical"
        Parameterization scheme: "canonical", "mean_prob", "mean_odds"
        (or aliases: "standard", "linked", "odds_ratio").
    unconstrained : bool, default=False
        If True, use Normal+transform instead of constrained distributions.
    guide_families : GuideFamilyConfig, optional
        Per-parameter guide family configuration.
    n_components : int, optional
        Number of mixture components for mixture models.
    mixture_params : List[str], optional
        List of parameters to make mixture-specific.
    priors : Dict[str, Tuple[float, ...]], optional
        Prior hyperparameters keyed by parameter name.
    guides : Dict[str, Tuple[float, ...]], optional
        Guide hyperparameters keyed by parameter name.
    validate : bool, default=True
        If True, validate model/guide compatibility with a dry run.

    Returns
    -------
    model : Callable
        NumPyro model function.
    guide : Callable
        NumPyro guide function.

    Examples
    --------
    >>> model, guide = create_model_from_params(
    ...     model="zinb",
    ...     parameterization="linked",
    ...     n_components=3,
    ... )
    """
    from ..config import ModelConfigBuilder

    # Build ModelConfig
    builder = (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization(parameterization)
    )

    if unconstrained:
        builder.unconstrained()

    if guide_families is not None:
        builder.with_guide_families(guide_families)

    if n_components is not None:
        builder.as_mixture(n_components, mixture_params)

    model_config = builder.build()

    model, guide, _ = create_model(
        model_config, priors=priors, guides=guides, validate=validate
    )
    return model, guide


# ==============================================================================
# Helper Functions
# ==============================================================================


def _get_parameterization_key(param: Union[str, ParamEnum]) -> str:
    """Convert parameterization enum or string to registry key."""
    if isinstance(param, ParamEnum):
        enum_to_key = {
            ParamEnum.STANDARD: "canonical",
            ParamEnum.LINKED: "mean_prob",
            ParamEnum.ODDS_RATIO: "mean_odds",
        }
        return enum_to_key.get(param, param.value)
    return param


# ------------------------------------------------------------------------------


def _hierarchicalize_p(
    param_specs: List,
    param_key: str,
    guide_families,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
) -> List:
    """Replace the flat p or phi spec with a hierarchical triplet.

    Given a list of parameter specs, this function finds the p (or phi) spec
    and replaces it with three specs: hyperprior loc, hyperprior scale, and
    a hierarchical gene-specific spec. The p/phi hyper names depend on the
    parameterization.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Current list of parameter specs (from parameterization strategy).
    param_key : str
        Parameterization registry key ("canonical", "mean_prob", "mean_odds").
    guide_families : GuideFamilyConfig
        Per-parameter guide family configuration.
    n_components : int, optional
        Number of mixture components.
    mixture_params : List[str], optional
        Parameters marked as mixture-specific.

    Returns
    -------
    List[ParamSpec]
        Updated parameter specs with the flat p/phi replaced by the
        hierarchical triplet.
    """
    # Determine target parameter and hyperparameter names
    if param_key == "mean_odds":
        target_name = "phi"
        hyper_loc_name = "log_phi_loc"
        hyper_scale_name = "log_phi_scale"
        HierSpec = HierarchicalExpNormalSpec
    else:
        target_name = "p"
        hyper_loc_name = "logit_p_loc"
        hyper_scale_name = "logit_p_scale"
        HierSpec = HierarchicalSigmoidNormalSpec

    # Get guide family for the target parameter
    target_family = guide_families.get(target_name)

    # Determine mixture flag for the target parameter
    is_target_mixture = False
    if n_components is not None:
        if mixture_params is None:
            is_target_mixture = True
        else:
            is_target_mixture = target_name in mixture_params

    # Build the hierarchical triplet
    hyper_loc = NormalWithTransformSpec(
        name=hyper_loc_name,
        shape_dims=(),
        default_params=(0.0, 1.0),
    )
    hyper_scale = SoftplusNormalSpec(
        name=hyper_scale_name,
        shape_dims=(),
        default_params=(0.0, 0.5),
    )
    hier_spec = HierSpec(
        name=target_name,
        shape_dims=("n_genes",),
        default_params=(0.0, 1.0),
        hyper_loc_name=hyper_loc_name,
        hyper_scale_name=hyper_scale_name,
        is_gene_specific=True,
        guide_family=target_family,
        is_mixture=is_target_mixture,
    )

    # Replace the flat spec with the hierarchical triplet
    new_specs = []
    for spec in param_specs:
        if spec.name == target_name:
            new_specs.extend([hyper_loc, hyper_scale, hier_spec])
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------


def _extract_priors_from_param_specs(
    param_specs: List,
) -> Dict[str, Tuple[float, ...]]:
    """Extract prior overrides from a list of ParamSpec objects."""
    priors = {}
    for spec in param_specs:
        if hasattr(spec, "prior") and spec.prior is not None:
            priors[spec.name] = spec.prior
    return priors


# ------------------------------------------------------------------------------


def _extract_guides_from_param_specs(
    param_specs: List,
) -> Dict[str, Tuple[float, ...]]:
    """Extract guide overrides from a list of ParamSpec objects."""
    guides = {}
    for spec in param_specs:
        if hasattr(spec, "guide") and spec.guide is not None:
            guides[spec.name] = spec.guide
    return guides


# ------------------------------------------------------------------------------


def _validate_mixture_params(
    mixture_params: List[str],
    param_strategy,
    base_model: str,
) -> None:
    """Validate that mixture_params contains only valid parameter names.

    Parameters
    ----------
    mixture_params : List[str]
        User-provided list of parameters to make mixture-specific.
    param_strategy : Parameterization
        The parameterization strategy being used.
    base_model : str
        The base model type (nbdm, zinb, nbvcp, zinbvcp).

    Raises
    ------
    ValueError
        If mixture_params contains invalid parameter names.
    """
    # Get valid parameters from parameterization's core parameters
    valid_params = set(param_strategy.core_parameters)

    # Add model-specific extra parameters (transformed for this parameterization)
    extra_params = MODEL_EXTRA_PARAMS.get(base_model, [])
    for param in extra_params:
        # Transform param name (e.g., p_capture -> phi_capture for mean_odds)
        transformed = param_strategy.transform_model_param(param)
        valid_params.add(transformed)
        # Also accept the original name as an alias
        valid_params.add(param)

    # Check for invalid parameters
    invalid_params = set(mixture_params) - valid_params

    if invalid_params:
        # Build helpful error message
        param_name = param_strategy.name
        core_params = param_strategy.core_parameters
        extra_info = ""

        if extra_params:
            transformed_extras = [
                param_strategy.transform_model_param(p) for p in extra_params
            ]
            extra_info = (
                f"\n  Model-specific ({base_model}): {transformed_extras}"
            )

        raise ValueError(
            f"Invalid mixture_params for '{param_name}' "
            f"parameterization: {sorted(invalid_params)}\n"
            f"Valid parameters are:\n"
            f"  Core ({param_name}): {core_params}{extra_info}\n"
            f"\n"
            f"Note: Derived parameters (like 'r' in mean_odds) "
            f"cannot be mixture-specific since they are computed from other "
            f"parameters, not sampled directly."
        )


# ------------------------------------------------------------------------------


def _has_amortized_guide(guide_families: Optional[GuideFamilyConfig]) -> bool:
    """Check if any guide uses amortization.

    Amortized guides require actual data during execution, so validation
    with dummy data will fail.
    """
    if guide_families is None:
        return False

    from ..components.guide_families import AmortizedGuide

    # Check for capture_amortization config (creates AmortizedGuide for capture
    # params)
    if hasattr(guide_families, "capture_amortization"):
        amort_config = getattr(guide_families, "capture_amortization", None)
        if amort_config is not None and getattr(amort_config, "enabled", False):
            return True

    # Check all configured guide families for direct AmortizedGuide instances
    # Access model_fields from the class, not the instance
    for field in GuideFamilyConfig.model_fields:
        value = getattr(guide_families, field, None)
        if isinstance(value, AmortizedGuide):
            return True
    return False


# ==============================================================================
# Export
# ==============================================================================

__all__ = [
    "create_model",
    "create_model_from_params",
    "validate_model_guide_compatibility",
]
