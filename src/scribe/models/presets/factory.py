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

import jax.numpy as jnp
import numpyro

from ..builders import GuideBuilder, ModelBuilder
from ..builders.parameter_specs import (
    DatasetHierarchicalPositiveNormalSpec,
    DatasetHierarchicalSigmoidNormalSpec,
    PositiveNormalSpec,
    GaussianLatentSpec,
    GammaSpec,
    HalfCauchySpec,
    HierarchicalPositiveNormalSpec,
    HierarchicalSigmoidNormalSpec,
    HorseshoeDatasetPositiveNormalSpec,
    HorseshoeDatasetSigmoidNormalSpec,
    HorseshoeHierarchicalPositiveNormalSpec,
    HorseshoeHierarchicalSigmoidNormalSpec,
    InverseGammaSpec,
    NEGDatasetPositiveNormalSpec,
    NEGDatasetSigmoidNormalSpec,
    NEGHierarchicalPositiveNormalSpec,
    NEGHierarchicalSigmoidNormalSpec,
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
from ..config.enums import HierarchicalPriorType, InferenceMethod
from ..config.enums import Parameterization as ParamEnum
from ..config.groups import VAEConfig
from ..parameterizations import PARAMETERIZATIONS
import numpyro.distributions as npdist
from scribe.flows import FlowChain
from .registry import (
    LIKELIHOOD_REGISTRY,
    BNB_LIKELIHOOD_REGISTRY,
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
    # Build dummy dataset_indices for multi-dataset models so that
    # per-dataset parameters get indexed to per-cell values during the
    # dry run (otherwise shapes like (n_datasets, n_genes) won't
    # broadcast with per-cell capture parameters).
    dataset_indices = None
    if model_config.n_datasets is not None:
        dataset_indices = jnp.zeros(n_cells, dtype=jnp.int32)

    # Run model to get sample sites
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.trace() as model_trace:
            try:
                model(
                    n_cells=n_cells,
                    n_genes=n_genes,
                    model_config=model_config,
                    counts=None,  # Prior predictive mode
                    dataset_indices=dataset_indices,
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
                    dataset_indices=dataset_indices,
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
    # Resolve positive transform for unconstrained positive-valued parameters.
    # Fallback to "exp" preserves behavior for legacy configs missing the field.
    _pt = getattr(model_config, "positive_transform", "exp")
    _pos_transform = (
        npdist.transforms.SoftplusTransform()
        if _pt == "softplus"
        else npdist.transforms.ExpTransform()
    )

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
    latent_marker = PositiveNormalSpec(
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
                hierarchical_gate=(
                    model_config.gate_prior != HierarchicalPriorType.NONE
                ),
                model_config=model_config,
                positive_transform=_pos_transform,
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
    # Capture-prior keys (organism, eta_capture, mu_eta) are handled by
    # ModelConfig and the registry; filter them out before applying overrides.
    _CAPTURE_PRIOR_KEYS = {"organism", "eta_capture", "mu_eta"}
    merged_priors = _extract_priors_from_param_specs(model_config.param_specs)
    if priors:
        merged_priors.update(
            {k: v for k, v in priors.items() if k not in _CAPTURE_PRIOR_KEYS}
        )
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
    # Select BNB likelihood class when overdispersion is active
    if model_config.is_bnb:
        likelihood_class = BNB_LIKELIHOOD_REGISTRY[base_model]
    else:
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

    # Resolve the positive-parameter transform from config.
    # "softplus" prevents float32 overflow; "exp" is the classic log-Normal.
    _pos_transform = (
        npdist.transforms.SoftplusTransform()
        if model_config.positive_transform == "softplus"
        else npdist.transforms.ExpTransform()
    )

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
            spec.name for spec in model_config.param_specs if spec.is_mixture
        ]
        if stored_mixture_params:
            effective_mixture_params = stored_mixture_params

    param_specs = param_strategy.build_param_specs(
        unconstrained=model_config.unconstrained,
        guide_families=guide_families,
        n_components=model_config.n_components,
        mixture_params=effective_mixture_params,
    )

    # Override the default ExpTransform on any flat PositiveNormalSpec created
    # by the parameterization strategy (e.g. phi, mu, r before hierarchy).
    for i, spec in enumerate(param_specs):
        if isinstance(spec, PositiveNormalSpec):
            param_specs[i] = spec.model_copy(
                update={"transform": _pos_transform}
            )

    # ==========================================================================
    # Step 4.5: Apply gene-level p/phi hierarchy (Gaussian, horseshoe, or NEG)
    # ==========================================================================
    _NONE = HierarchicalPriorType.NONE
    if model_config.p_prior != _NONE:
        param_specs = _hierarchicalize_p(
            param_specs=param_specs,
            param_key=param_key,
            guide_families=guide_families,
            n_components=model_config.n_components,
            mixture_params=effective_mixture_params,
            positive_transform=_pos_transform,
        )

    # ==========================================================================
    # Step 4.55: Apply gene-level mu hierarchy (across-component shrinkage)
    # ==========================================================================
    if model_config.mu_prior != _NONE:
        param_specs = _hierarchicalize_mu(
            param_specs=param_specs,
            param_key=param_key,
            guide_families=guide_families,
            n_components=model_config.n_components,
            mixture_params=effective_mixture_params,
            positive_transform=_pos_transform,
        )

    # ==========================================================================
    # Step 4.6: Apply dataset-level hierarchy flags
    # ==========================================================================
    if model_config.n_datasets is not None:
        n_ds = model_config.n_datasets
        # shared_component_indices is populated at runtime by fit()
        # when annotation_key + dataset_key are both provided.
        _sci = getattr(model_config, "shared_component_indices", None)

        # Hierarchical mu/r across datasets
        if model_config.mu_dataset_prior != _NONE:
            param_specs = _datasetify_mu(
                param_specs=param_specs,
                param_key=param_key,
                guide_families=guide_families,
                n_datasets=n_ds,
                shared_component_indices=_sci,
                positive_transform=_pos_transform,
            )

        # Dataset-level p/phi — resolve structural mode
        if model_config.p_dataset_prior != _NONE:
            dataset_p_mode = model_config.hierarchical_dataset_p
            if dataset_p_mode in ("scalar", "gene_specific"):
                param_specs = _datasetify_p(
                    param_specs=param_specs,
                    param_key=param_key,
                    guide_families=guide_families,
                    n_datasets=n_ds,
                    mode=dataset_p_mode,
                    shared_component_indices=_sci,
                    positive_transform=_pos_transform,
                )

    # ==========================================================================
    # Step 5: Add model-specific extra parameters
    # ==========================================================================
    effective_hierarchical_gate = model_config.gate_prior != _NONE
    extra_param_names = list(MODEL_EXTRA_PARAMS[base_model])
    # Append BNB concentration when overdispersion is enabled
    if model_config.is_bnb:
        extra_param_names.append("bnb_concentration")
    for param_name in extra_param_names:
        extra_specs = build_extra_param_spec(
            param_name=param_name,
            unconstrained=model_config.unconstrained,
            guide_families=guide_families,
            param_strategy=param_strategy,
            n_components=model_config.n_components,
            mixture_params=effective_mixture_params,
            hierarchical_gate=effective_hierarchical_gate,
            model_config=model_config,
            positive_transform=_pos_transform,
        )
        param_specs.extend(extra_specs)

    # ==========================================================================
    # Step 5.5: Apply dataset-level gate hierarchy (after gate spec exists)
    # ==========================================================================
    if (
        model_config.n_datasets is not None
        and model_config.gate_dataset_prior != _NONE
    ):
        param_specs = _datasetify_gate(
            param_specs=param_specs,
            guide_families=guide_families,
            n_datasets=model_config.n_datasets,
        )

    # ==========================================================================
    # Step 5.7: Apply horseshoe priors (upgrade normal hierarchies in-place)
    # ==========================================================================
    _HS = HierarchicalPriorType.HORSESHOE
    horseshoe_kwargs = _horseshoe_kwargs_from_config(model_config)

    if model_config.mu_prior == _HS:
        param_specs = _horseshoe_mu(param_specs, param_key, **horseshoe_kwargs)

    if model_config.p_prior == _HS:
        param_specs = _horseshoe_p(param_specs, param_key, **horseshoe_kwargs)

    if model_config.gate_prior == _HS:
        param_specs = _horseshoe_gate(param_specs, **horseshoe_kwargs)

    if model_config.mu_dataset_prior == _HS:
        param_specs = _horseshoe_dataset_mu(
            param_specs, param_key, **horseshoe_kwargs
        )

    if model_config.p_dataset_prior == _HS:
        param_specs = _horseshoe_dataset_p(
            param_specs, param_key, **horseshoe_kwargs
        )

    if model_config.gate_dataset_prior == _HS:
        param_specs = _horseshoe_dataset_gate(param_specs, **horseshoe_kwargs)

    # ==========================================================================
    # Step 5.8: Apply NEG priors (upgrade normal hierarchies in-place)
    # ==========================================================================
    _NEG = HierarchicalPriorType.NEG
    neg_kwargs = _neg_kwargs_from_config(model_config)

    if model_config.mu_prior == _NEG:
        param_specs = _neg_mu(param_specs, param_key, **neg_kwargs)

    if model_config.p_prior == _NEG:
        param_specs = _neg_p(param_specs, param_key, **neg_kwargs)

    if model_config.gate_prior == _NEG:
        param_specs = _neg_gate(param_specs, **neg_kwargs)

    if model_config.mu_dataset_prior == _NEG:
        param_specs = _neg_dataset_mu(param_specs, param_key, **neg_kwargs)

    if model_config.p_dataset_prior == _NEG:
        param_specs = _neg_dataset_p(param_specs, param_key, **neg_kwargs)

    if model_config.gate_dataset_prior == _NEG:
        param_specs = _neg_dataset_gate(param_specs, **neg_kwargs)

    # ==========================================================================
    # Step 6: Apply user-provided prior/guide overrides
    # ==========================================================================
    # Capture-prior keys are handled by ModelConfig / registry; filter them.
    _CAPTURE_PRIOR_KEYS = {"organism", "eta_capture", "mu_eta"}
    merged_priors = _extract_priors_from_param_specs(model_config.param_specs)
    if priors:
        merged_priors.update(
            {k: v for k, v in priors.items() if k not in _CAPTURE_PRIOR_KEYS}
        )

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
    # Select BNB likelihood class when overdispersion is active
    if model_config.is_bnb:
        likelihood_class = BNB_LIKELIHOOD_REGISTRY[base_model]
    else:
        likelihood_class = LIKELIHOOD_REGISTRY[base_model]
    if base_model in ("nbvcp", "zinbvcp"):
        # Get the transformed capture param name (p_capture or phi_capture)
        capture_param_name = param_strategy.transform_model_param("p_capture")
        # Detect biology-informed capture spec to pass to likelihood
        from ..builders.parameter_specs import BiologyInformedCaptureSpec

        capture_spec = next(
            (
                s
                for s in param_specs
                if isinstance(s, BiologyInformedCaptureSpec)
            ),
            None,
        )
        model_builder.with_likelihood(
            likelihood_class(
                capture_param_name=capture_param_name,
                biology_informed_spec=capture_spec,
            )
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
    positive_transform=None,
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
        HierSpec = HierarchicalPositiveNormalSpec
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
    # Only pass transform for positive (Exp/Softplus) specs, not Sigmoid
    extra_kwargs = {}
    if (
        positive_transform is not None
        and HierSpec is HierarchicalPositiveNormalSpec
    ):
        extra_kwargs["transform"] = positive_transform
    hier_spec = HierSpec(
        name=target_name,
        shape_dims=("n_genes",),
        default_params=(0.0, 1.0),
        hyper_loc_name=hyper_loc_name,
        hyper_scale_name=hyper_scale_name,
        is_gene_specific=True,
        guide_family=target_family,
        is_mixture=is_target_mixture,
        **extra_kwargs,
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


def _hierarchicalize_mu(
    param_specs: List,
    param_key: str,
    guide_families,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    positive_transform=None,
) -> List:
    """Replace the flat mu (or r) spec with a hierarchical triplet.

    Adds population-level hyperparameters (per-gene loc and scalar scale)
    and replaces the flat mu/r spec with a ``HierarchicalPositiveNormalSpec``
    that draws per-component, per-gene values from the population prior.
    This provides shrinkage across mixture components: most genes share
    similar means across cell types, with only some deviating.

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
        Updated parameter specs with the flat mu/r replaced by a
        hierarchical triplet (hyper_loc, hyper_scale, hier_mu).
    """
    # Determine target parameter and hyperparameter names
    if param_key in ("mean_odds", "mean_prob"):
        target_name = "mu"
        hyper_loc_name = "log_mu_loc"
        hyper_scale_name = "log_mu_scale"
    else:
        target_name = "r"
        hyper_loc_name = "log_r_loc"
        hyper_scale_name = "log_r_scale"

    target_family = guide_families.get(target_name)

    # Population-level hyperparameters: per-gene location, scalar scale
    hyper_loc = NormalWithTransformSpec(
        name=hyper_loc_name,
        shape_dims=("n_genes",),
        default_params=(0.0, 1.0),
        is_gene_specific=True,
    )
    hyper_scale = SoftplusNormalSpec(
        name=hyper_scale_name,
        shape_dims=(),
        default_params=(-2.0, 0.5),
    )

    new_specs = []
    for spec in param_specs:
        if spec.name == target_name:
            # Preserve is_mixture from the original spec
            orig_is_mixture = getattr(spec, "is_mixture", False)
            hier_spec = HierarchicalPositiveNormalSpec(
                name=target_name,
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                hyper_loc_name=hyper_loc_name,
                hyper_scale_name=hyper_scale_name,
                is_gene_specific=True,
                is_mixture=orig_is_mixture,
                guide_family=target_family,
                **(
                    {"transform": positive_transform}
                    if positive_transform is not None
                    else {}
                ),
            )
            new_specs.extend([hyper_loc, hyper_scale, hier_spec])
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Gene-level horseshoe mu
# ------------------------------------------------------------------------------


def _horseshoe_mu(
    param_specs: List,
    param_key: str,
    tau0: float = 1.0,
    slab_df: int = 4,
    slab_scale: float = 2.0,
) -> List:
    """Upgrade gene-level hierarchical mu/r to horseshoe.

    Finds the hierarchical triplet (hyper_loc, hyper_scale, hier-mu)
    produced by ``_hierarchicalize_mu``, replaces hyper_scale
    (SoftplusNormalSpec) with the horseshoe trio (tau, lambda, c_sq),
    and replaces the ``HierarchicalPositiveNormalSpec`` with
    ``HorseshoeHierarchicalPositiveNormalSpec``.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_hierarchicalize_mu`` has run.
    param_key : str
        Parameterization key.
    tau0, slab_df, slab_scale : float
        Horseshoe hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with horseshoe mu.
    """
    if param_key in ("mean_odds", "mean_prob"):
        target_name = "mu"
        scale_name = "log_mu_scale"
        loc_name = "log_mu_loc"
        prefix = "mu"
    else:
        target_name = "r"
        scale_name = "log_r_scale"
        loc_name = "log_r_loc"
        prefix = "r"

    raw_name = f"{target_name}_raw"
    tau_spec, lambda_spec, c_sq_spec = _make_horseshoe_hypers(
        prefix, tau0, slab_df, slab_scale
    )

    new_specs = []
    for spec in param_specs:
        if spec.name == scale_name:
            new_specs.extend([tau_spec, lambda_spec, c_sq_spec])
        elif spec.name == target_name and isinstance(
            spec, HierarchicalPositiveNormalSpec
        ):
            horseshoe_spec = HorseshoeHierarchicalPositiveNormalSpec(
                name=target_name,
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name=loc_name,
                hyper_scale_name=f"tau_{prefix}",
                tau_name=f"tau_{prefix}",
                lambda_name=f"lambda_{prefix}",
                c_sq_name=f"c_sq_{prefix}",
                raw_name=raw_name,
                is_gene_specific=spec.is_gene_specific,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
                transform=spec.transform,
            )
            new_specs.append(horseshoe_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Gene-level NEG mu
# ------------------------------------------------------------------------------


def _neg_mu(
    param_specs: List,
    param_key: str,
    u: float = 1.0,
    a: float = 1.0,
    tau: float = 1.0,
) -> List:
    """Upgrade gene-level hierarchical mu/r to NEG.

    Finds the hierarchical triplet (hyper_loc, hyper_scale, hier-mu)
    produced by ``_hierarchicalize_mu``, replaces hyper_scale
    (SoftplusNormalSpec) with the NEG pair (zeta, psi), and replaces the
    ``HierarchicalPositiveNormalSpec`` with ``NEGHierarchicalPositiveNormalSpec``.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_hierarchicalize_mu`` has run.
    param_key : str
        Parameterization key.
    u, a, tau : float
        NEG hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with NEG mu.
    """
    if param_key in ("mean_odds", "mean_prob"):
        target_name = "mu"
        scale_name = "log_mu_scale"
        loc_name = "log_mu_loc"
        prefix = "mu"
    else:
        target_name = "r"
        scale_name = "log_r_scale"
        loc_name = "log_r_loc"
        prefix = "r"

    raw_name = f"{target_name}_raw"
    zeta_spec, psi_spec = _make_neg_hypers(prefix, u, a, tau)

    new_specs = []
    for spec in param_specs:
        if spec.name == scale_name:
            new_specs.extend([zeta_spec, psi_spec])
        elif spec.name == target_name and isinstance(
            spec, HierarchicalPositiveNormalSpec
        ):
            neg_spec = NEGHierarchicalPositiveNormalSpec(
                name=target_name,
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name=loc_name,
                hyper_scale_name=f"psi_{prefix}",
                psi_name=f"psi_{prefix}",
                zeta_name=f"zeta_{prefix}",
                raw_name=raw_name,
                is_gene_specific=spec.is_gene_specific,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
                transform=spec.transform,
            )
            new_specs.append(neg_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------


def _datasetify_mu(
    param_specs: List,
    param_key: str,
    guide_families,
    n_datasets: int,
    shared_component_indices: Optional[Tuple[int, ...]] = None,
    positive_transform=None,
) -> List:
    """Replace mu (or r) with a dataset-hierarchical triplet.

    Adds population-level hyperparameters (loc, scale) and replaces the flat
    mu/r spec with a ``DatasetHierarchicalPositiveNormalSpec`` that produces
    per-dataset gene-specific values.

    When the original parameter is mixture-aware (``is_mixture=True``),
    the population hyperprior ``hyper_loc`` is also made per-component so
    that each cell type gets its own population expression profile.  The
    ``shared_component_indices`` mask tells ``sample_hierarchical()``
    which components should use the learned cross-dataset scale vs a
    clamped near-zero scale.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Current list of parameter specs.
    param_key : str
        Parameterization registry key ("canonical", "mean_prob", "mean_odds").
    guide_families : GuideFamilyConfig
        Per-parameter guide family configuration.
    n_datasets : int
        Number of datasets.
    shared_component_indices : tuple of int, optional
        Component indices shared across 2+ datasets.  Passed through to
        the hierarchical spec for scale masking.

    Returns
    -------
    List[ParamSpec]
        Updated parameter specs with the flat mu/r replaced by a
        dataset-hierarchical triplet.
    """
    # For mean_odds and mean_prob, the target is mu; for canonical, it's r
    if param_key in ("mean_odds", "mean_prob"):
        target_name = "mu"
        hyper_loc_name = "log_mu_dataset_loc"
        hyper_scale_name = "log_mu_dataset_scale"
    else:
        target_name = "r"
        hyper_loc_name = "log_r_dataset_loc"
        hyper_scale_name = "log_r_dataset_scale"

    target_family = guide_families.get(target_name)

    new_specs = []
    for spec in param_specs:
        if spec.name == target_name:
            # Preserve is_mixture from the original spec so the dataset-
            # hierarchical parameter keeps its component dimension.
            orig_is_mixture = getattr(spec, "is_mixture", False)

            # Per-component hyperprior: when the parameter is mixture-
            # aware, each component gets its own population profile.
            hyper_loc = NormalWithTransformSpec(
                name=hyper_loc_name,
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                is_mixture=orig_is_mixture,
            )
            # Shared scalar shrinkage across all components
            hyper_scale = SoftplusNormalSpec(
                name=hyper_scale_name,
                shape_dims=(),
                default_params=(-2.0, 0.5),
            )

            hier_spec = DatasetHierarchicalPositiveNormalSpec(
                name=target_name,
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                hyper_loc_name=hyper_loc_name,
                hyper_scale_name=hyper_scale_name,
                is_gene_specific=True,
                is_dataset=True,
                is_mixture=orig_is_mixture,
                guide_family=target_family,
                shared_component_indices=shared_component_indices,
                **(
                    {"transform": positive_transform}
                    if positive_transform is not None
                    else {}
                ),
            )
            new_specs.extend([hyper_loc, hyper_scale, hier_spec])
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------


def _datasetify_p(
    param_specs: List,
    param_key: str,
    guide_families,
    n_datasets: int,
    mode: str = "scalar",
    shared_component_indices: Optional[Tuple[int, ...]] = None,
    positive_transform=None,
) -> List:
    """Replace p/phi with a dataset-specific version.

    For mode="scalar": one p per dataset (shared across genes), with a
    population-level hierarchical prior.

    For mode="gene_specific": single-level hierarchy where each (dataset,
    gene) pair draws from a shared population distribution.

    When the original parameter is mixture-aware, per-component
    hyperpriors are created (see ``_datasetify_mu`` for rationale).

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Current list of parameter specs.
    param_key : str
        Parameterization registry key.
    guide_families : GuideFamilyConfig
        Per-parameter guide family configuration.
    n_datasets : int
        Number of datasets.
    mode : str
        "scalar" or "gene_specific".
    shared_component_indices : tuple of int, optional
        Component indices shared across 2+ datasets.

    Returns
    -------
    List[ParamSpec]
        Updated parameter specs.
    """
    if param_key == "mean_odds":
        target_name = "phi"
        hyper_loc_name = "log_phi_dataset_loc"
        hyper_scale_name = "log_phi_dataset_scale"
        HierSpec = DatasetHierarchicalPositiveNormalSpec
    else:
        target_name = "p"
        hyper_loc_name = "logit_p_dataset_loc"
        hyper_scale_name = "logit_p_dataset_scale"
        HierSpec = DatasetHierarchicalSigmoidNormalSpec

    target_family = guide_families.get(target_name)

    new_specs = []
    for spec in param_specs:
        if spec.name == target_name:
            # Preserve is_mixture from the original spec so the dataset-
            # hierarchical parameter keeps its component dimension.
            orig_is_mixture = getattr(spec, "is_mixture", False)

            # Per-component hyperprior when mixture-aware: each component
            # gets its own population-level loc.  For scalar mode the
            # hyper_loc shape_dims stays () (per-component but not per-gene);
            # for gene_specific mode it becomes ("n_genes",).
            hyper_loc_dims: Tuple[str, ...] = ()
            hyper_loc_gene_specific = False
            if mode == "gene_specific":
                hyper_loc_dims = ("n_genes",)
                hyper_loc_gene_specific = True

            hyper_loc = NormalWithTransformSpec(
                name=hyper_loc_name,
                shape_dims=hyper_loc_dims,
                default_params=(0.0, 1.0),
                is_gene_specific=hyper_loc_gene_specific,
                is_mixture=orig_is_mixture,
            )
            hyper_scale = SoftplusNormalSpec(
                name=hyper_scale_name,
                shape_dims=(),
                default_params=(0.0, 0.5),
            )

            extra_kwargs = {}
            if (
                positive_transform is not None
                and HierSpec is DatasetHierarchicalPositiveNormalSpec
            ):
                extra_kwargs["transform"] = positive_transform

            if mode == "scalar":
                hier_spec = HierSpec(
                    name=target_name,
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    hyper_loc_name=hyper_loc_name,
                    hyper_scale_name=hyper_scale_name,
                    is_gene_specific=False,
                    is_dataset=True,
                    is_mixture=orig_is_mixture,
                    guide_family=target_family,
                    shared_component_indices=shared_component_indices,
                    **extra_kwargs,
                )
            else:
                hier_spec = HierSpec(
                    name=target_name,
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    hyper_loc_name=hyper_loc_name,
                    hyper_scale_name=hyper_scale_name,
                    is_gene_specific=True,
                    is_dataset=True,
                    is_mixture=orig_is_mixture,
                    guide_family=target_family,
                    shared_component_indices=shared_component_indices,
                    **extra_kwargs,
                )
            new_specs.extend([hyper_loc, hyper_scale, hier_spec])
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------


def _datasetify_gate(
    param_specs: List,
    guide_families,
    n_datasets: int,
) -> List:
    """Replace gate with per-dataset independent gates shrunk toward zero.

    Each (dataset, gene) pair—and each (component, dataset, gene) triple
    when the gate is mixture-aware—gets its own gate independently pushed
    toward zero.  The structure mirrors the gene-level gate hierarchy but
    with an extra dataset dimension on the NCP ``z`` variable:

    * **Scalar loc** ``N(-5, 1)``—shared across genes, datasets, and
      components.  A single scalar is robust against being overwhelmed
      by the likelihood (unlike a per-gene loc).
    * **Per-gene NEG/horseshoe psi** provides adaptive per-gene shrinkage.
    * **Per-dataset z** gives each dataset its own independent gate draw.

    Unlike ``_datasetify_mu`` / ``_datasetify_p``, this function does
    **not** create a hierarchical structure that pools gates across
    datasets.  The NEG/horseshoe controls how far each gene's gate can
    deviate from the scalar center, not the cross-dataset spread.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Current list of parameter specs (must already contain a gate spec,
        i.e. this should be called after Step 5 adds extra params).
    guide_families : GuideFamilyConfig
        Per-parameter guide family configuration.
    n_datasets : int
        Number of datasets.

    Returns
    -------
    List[ParamSpec]
        Updated parameter specs with the gate replaced by a
        dataset-level triplet (scalar loc, scalar scale, per-dataset gate).
    """
    hyper_loc_name = "logit_gate_dataset_loc"
    hyper_scale_name = "logit_gate_dataset_scale"

    gate_family = guide_families.get("gate")

    new_specs = []
    for spec in param_specs:
        if spec.name == "gate":
            # Preserve is_mixture from the original spec so the dataset-
            # hierarchical parameter keeps its component dimension.
            orig_is_mixture = getattr(spec, "is_mixture", False)

            # Population-level location is a scalar shared across all
            # genes, datasets, and components.  A very tight prior
            # N(-5, 0.01) anchors logit(gate) deep in the off region
            # so the likelihood cannot drag it positive.  Per-gene
            # adaptive shrinkage is handled by the NEG/horseshoe psi,
            # and per-dataset independence comes from the NCP z variable.
            hyper_loc = NormalWithTransformSpec(
                name=hyper_loc_name,
                shape_dims=(),
                default_params=(-5.0, 0.01),
                is_gene_specific=False,
                is_mixture=False,
            )
            hyper_scale = SoftplusNormalSpec(
                name=hyper_scale_name,
                shape_dims=(),
                default_params=(-2.0, 0.5),
            )

            hier_spec = DatasetHierarchicalSigmoidNormalSpec(
                name="gate",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                hyper_loc_name=hyper_loc_name,
                hyper_scale_name=hyper_scale_name,
                is_gene_specific=True,
                is_dataset=True,
                is_mixture=orig_is_mixture,
                guide_family=gate_family,
            )
            new_specs.extend([hyper_loc, hyper_scale, hier_spec])
        else:
            new_specs.append(spec)
    return new_specs


# ==============================================================================
# Horseshoe Factory Helpers
# ==============================================================================


def _horseshoe_kwargs_from_config(model_config: ModelConfig) -> dict:
    """Extract horseshoe hyperparameters from model config.

    These are shared across all horseshoe usages (gene-level and dataset-level).

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    dict
        Keys: ``tau0``, ``slab_df``, ``slab_scale``.
    """
    return {
        "tau0": getattr(model_config, "horseshoe_tau0", 1.0),
        "slab_df": getattr(model_config, "horseshoe_slab_df", 4),
        "slab_scale": getattr(model_config, "horseshoe_slab_scale", 2.0),
    }


def _make_horseshoe_hypers(
    prefix: str,
    tau0: float = 1.0,
    slab_df: int = 4,
    slab_scale: float = 2.0,
) -> Tuple:
    """Create the three horseshoe hyperparameter specs for a given prefix.

    Parameters
    ----------
    prefix : str
        Naming prefix, e.g. ``"p"``, ``"gate"``, ``"mu_dataset"``.
    tau0 : float
        Scale for the global shrinkage Half-Cauchy.
    slab_df : int
        Degrees of freedom for the slab Inverse-Gamma.
    slab_scale : float
        Scale for the slab Inverse-Gamma.

    Returns
    -------
    Tuple[HalfCauchySpec, HalfCauchySpec, InverseGammaSpec]
        (tau_spec, lambda_spec, c_sq_spec)
    """
    tau_name = f"tau_{prefix}"
    lambda_name = f"lambda_{prefix}"
    c_sq_name = f"c_sq_{prefix}"

    tau_spec = HalfCauchySpec(
        name=tau_name,
        shape_dims=(),
        scale=tau0,
    )
    lambda_spec = HalfCauchySpec(
        name=lambda_name,
        shape_dims=("n_genes",),
        scale=1.0,
        is_gene_specific=True,
    )
    c_sq_spec = InverseGammaSpec(
        name=c_sq_name,
        shape_dims=(),
        concentration=slab_df / 2.0,
        rate=slab_df * slab_scale**2 / 2.0,
    )
    return tau_spec, lambda_spec, c_sq_spec


# ------------------------------------------------------------------------------
# Gene-level horseshoe p
# ------------------------------------------------------------------------------


def _horseshoe_p(
    param_specs: List,
    param_key: str,
    tau0: float = 1.0,
    slab_df: int = 4,
    slab_scale: float = 2.0,
) -> List:
    """Upgrade gene-level hierarchical p/phi to horseshoe.

    Finds the hierarchical triplet (hyper_loc, hyper_scale, hier-p) produced
    by ``_hierarchicalize_p``, replaces hyper_scale (SoftplusNormalSpec) with
    the horseshoe trio (tau, lambda, c_sq), and replaces the
    ``HierarchicalSigmoidNormalSpec``/``HierarchicalPositiveNormalSpec`` with
    the corresponding horseshoe spec.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_hierarchicalize_p`` has run.
    param_key : str
        Parameterization key.
    tau0, slab_df, slab_scale : float
        Horseshoe hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with horseshoe p.
    """
    if param_key == "mean_odds":
        target_name = "phi"
        scale_name = "log_phi_scale"
        loc_name = "log_phi_loc"
        prefix = "phi"
    else:
        target_name = "p"
        scale_name = "logit_p_scale"
        loc_name = "logit_p_loc"
        prefix = "p"

    raw_name = f"{target_name}_raw"
    tau_spec, lambda_spec, c_sq_spec = _make_horseshoe_hypers(
        prefix, tau0, slab_df, slab_scale
    )

    new_specs = []
    for spec in param_specs:
        if spec.name == scale_name:
            # Replace the SoftplusNormal hyper_scale with horseshoe trio
            new_specs.extend([tau_spec, lambda_spec, c_sq_spec])
        elif spec.name == target_name and isinstance(
            spec,
            (HierarchicalSigmoidNormalSpec, HierarchicalPositiveNormalSpec),
        ):
            # Select horseshoe spec matching the original transform:
            # ExpNormal (phi, range (0,inf)) vs SigmoidNormal (p, range (0,1))
            if isinstance(spec, HierarchicalPositiveNormalSpec):
                HorseshoeSpec = HorseshoeHierarchicalPositiveNormalSpec
            else:
                HorseshoeSpec = HorseshoeHierarchicalSigmoidNormalSpec
            horseshoe_spec = HorseshoeSpec(
                name=target_name,
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name=loc_name,
                hyper_scale_name=f"tau_{prefix}",
                tau_name=f"tau_{prefix}",
                lambda_name=f"lambda_{prefix}",
                c_sq_name=f"c_sq_{prefix}",
                raw_name=raw_name,
                is_gene_specific=spec.is_gene_specific,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
                transform=spec.transform,
            )
            new_specs.append(horseshoe_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Gene-level horseshoe gate
# ------------------------------------------------------------------------------


def _horseshoe_gate(
    param_specs: List,
    tau0: float = 1.0,
    slab_df: int = 4,
    slab_scale: float = 2.0,
) -> List:
    """Upgrade gene-level hierarchical gate to horseshoe.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``build_gate_spec(hierarchical=True)`` has run.
    tau0, slab_df, slab_scale : float
        Horseshoe hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with horseshoe gate.
    """
    tau_spec, lambda_spec, c_sq_spec = _make_horseshoe_hypers(
        "gate", tau0, slab_df, slab_scale
    )

    new_specs = []
    for spec in param_specs:
        if spec.name == "logit_gate_scale":
            new_specs.extend([tau_spec, lambda_spec, c_sq_spec])
        elif spec.name == "gate" and isinstance(
            spec, HierarchicalSigmoidNormalSpec
        ):
            horseshoe_spec = HorseshoeHierarchicalSigmoidNormalSpec(
                name="gate",
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name="logit_gate_loc",
                hyper_scale_name="tau_gate",
                tau_name="tau_gate",
                lambda_name="lambda_gate",
                c_sq_name="c_sq_gate",
                raw_name="gate_raw",
                is_gene_specific=spec.is_gene_specific,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
            )
            new_specs.append(horseshoe_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Dataset-level horseshoe mu
# ------------------------------------------------------------------------------


def _horseshoe_dataset_mu(
    param_specs: List,
    param_key: str,
    tau0: float = 1.0,
    slab_df: int = 4,
    slab_scale: float = 2.0,
) -> List:
    """Upgrade dataset-level hierarchical mu/r to horseshoe.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_datasetify_mu`` has run.
    param_key : str
        Parameterization key.
    tau0, slab_df, slab_scale : float
        Horseshoe hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with horseshoe dataset mu.
    """
    if param_key in ("mean_odds", "mean_prob"):
        target_name = "mu"
        scale_name = "log_mu_dataset_scale"
        loc_name = "log_mu_dataset_loc"
        prefix = "mu_dataset"
    else:
        target_name = "r"
        scale_name = "log_r_dataset_scale"
        loc_name = "log_r_dataset_loc"
        prefix = "r_dataset"

    raw_name = f"{target_name}_raw"
    tau_spec, lambda_spec, c_sq_spec = _make_horseshoe_hypers(
        prefix, tau0, slab_df, slab_scale
    )

    new_specs = []
    for spec in param_specs:
        if spec.name == scale_name:
            new_specs.extend([tau_spec, lambda_spec, c_sq_spec])
        elif spec.name == target_name and isinstance(
            spec, DatasetHierarchicalPositiveNormalSpec
        ):
            horseshoe_spec = HorseshoeDatasetPositiveNormalSpec(
                name=target_name,
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name=loc_name,
                hyper_scale_name=f"tau_{prefix}",
                tau_name=f"tau_{prefix}",
                lambda_name=f"lambda_{prefix}",
                c_sq_name=f"c_sq_{prefix}",
                raw_name=raw_name,
                is_gene_specific=spec.is_gene_specific,
                is_dataset=spec.is_dataset,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
                transform=spec.transform,
            )
            new_specs.append(horseshoe_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Dataset-level horseshoe p
# ------------------------------------------------------------------------------


def _horseshoe_dataset_p(
    param_specs: List,
    param_key: str,
    tau0: float = 1.0,
    slab_df: int = 4,
    slab_scale: float = 2.0,
) -> List:
    """Upgrade dataset-level hierarchical p/phi to horseshoe.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_datasetify_p`` has run.
    param_key : str
        Parameterization key.
    tau0, slab_df, slab_scale : float
        Horseshoe hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with horseshoe dataset p.
    """
    if param_key == "mean_odds":
        target_name = "phi"
        scale_name = "log_phi_dataset_scale"
        loc_name = "log_phi_dataset_loc"
        prefix = "phi_dataset"
    else:
        target_name = "p"
        scale_name = "logit_p_dataset_scale"
        loc_name = "logit_p_dataset_loc"
        prefix = "p_dataset"

    raw_name = f"{target_name}_raw_dataset"
    tau_spec, lambda_spec, c_sq_spec = _make_horseshoe_hypers(
        prefix, tau0, slab_df, slab_scale
    )

    # Determine the correct horseshoe spec class based on the target transform
    if param_key == "mean_odds":
        TargetHierClass = DatasetHierarchicalPositiveNormalSpec
        HorseshoeClass = HorseshoeDatasetPositiveNormalSpec
    else:
        TargetHierClass = DatasetHierarchicalSigmoidNormalSpec
        HorseshoeClass = HorseshoeDatasetSigmoidNormalSpec

    new_specs = []
    for spec in param_specs:
        if spec.name == scale_name:
            new_specs.extend([tau_spec, lambda_spec, c_sq_spec])
        elif spec.name == target_name and isinstance(spec, TargetHierClass):
            horseshoe_spec = HorseshoeClass(
                name=target_name,
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name=loc_name,
                hyper_scale_name=f"tau_{prefix}",
                tau_name=f"tau_{prefix}",
                lambda_name=f"lambda_{prefix}",
                c_sq_name=f"c_sq_{prefix}",
                raw_name=raw_name,
                is_gene_specific=spec.is_gene_specific,
                is_dataset=spec.is_dataset,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
                transform=spec.transform,
            )
            new_specs.append(horseshoe_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Dataset-level horseshoe gate
# ------------------------------------------------------------------------------


def _horseshoe_dataset_gate(
    param_specs: List,
    tau0: float = 1.0,
    slab_df: int = 4,
    slab_scale: float = 2.0,
) -> List:
    """Upgrade dataset-level hierarchical gate to horseshoe.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_datasetify_gate`` has run.
    tau0, slab_df, slab_scale : float
        Horseshoe hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with horseshoe dataset gate.
    """
    tau_spec, lambda_spec, c_sq_spec = _make_horseshoe_hypers(
        "gate_dataset", tau0, slab_df, slab_scale
    )

    new_specs = []
    for spec in param_specs:
        if spec.name == "logit_gate_dataset_scale":
            new_specs.extend([tau_spec, lambda_spec, c_sq_spec])
        elif spec.name == "gate" and isinstance(
            spec, DatasetHierarchicalSigmoidNormalSpec
        ):
            horseshoe_spec = HorseshoeDatasetSigmoidNormalSpec(
                name="gate",
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name="logit_gate_dataset_loc",
                hyper_scale_name="tau_gate_dataset",
                tau_name="tau_gate_dataset",
                lambda_name="lambda_gate_dataset",
                c_sq_name="c_sq_gate_dataset",
                raw_name="gate_raw_dataset",
                is_gene_specific=spec.is_gene_specific,
                is_dataset=spec.is_dataset,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
            )
            new_specs.append(horseshoe_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ==============================================================================
# NEG (Normal-Exponential-Gamma) prior helpers
# ==============================================================================


def _neg_kwargs_from_config(model_config: ModelConfig) -> dict:
    """Extract NEG hyperparameters from model config.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    dict
        Keys: ``u``, ``a``, ``tau``.
    """
    return {
        "u": getattr(model_config, "neg_u", 1.0),
        "a": getattr(model_config, "neg_a", 1.0),
        "tau": getattr(model_config, "neg_tau", 1.0),
    }


def _make_neg_hypers(
    prefix: str,
    u: float = 1.0,
    a: float = 1.0,
    tau: float = 1.0,
) -> Tuple:
    """Create the two NEG hyperparameter specs (zeta, psi) for a prefix.

    The Gamma-Gamma hierarchy:
        zeta_g ~ Gamma(a, tau)      [per-gene rate]
        psi_g  ~ Gamma(u, zeta_g)   [per-gene variance]

    Parameters
    ----------
    prefix : str
        Naming prefix, e.g. ``"p"``, ``"gate"``, ``"mu_dataset"``.
    u : float
        Shape for the inner Gamma (psi). u=1 is NEG (Exponential).
    a : float
        Shape for the outer Gamma (zeta).
    tau : float
        Rate for the outer Gamma (global shrinkage).

    Returns
    -------
    Tuple[GammaSpec, GammaSpec]
        (zeta_spec, psi_spec) — note zeta is listed first because psi
        depends on zeta at sample time.
    """
    zeta_name = f"zeta_{prefix}"
    psi_name = f"psi_{prefix}"

    # Outer layer: zeta_g ~ Gamma(a, tau) with fixed rate
    zeta_spec = GammaSpec(
        name=zeta_name,
        shape_dims=("n_genes",),
        concentration=a,
        rate=tau,
        is_gene_specific=True,
    )
    # Inner layer: psi_g ~ Gamma(u, zeta_g) with dynamic rate from zeta
    psi_spec = GammaSpec(
        name=psi_name,
        shape_dims=("n_genes",),
        concentration=u,
        rate_name=zeta_name,
        is_gene_specific=True,
    )
    return zeta_spec, psi_spec


# ------------------------------------------------------------------------------
# Gene-level NEG p
# ------------------------------------------------------------------------------


def _neg_p(
    param_specs: List,
    param_key: str,
    u: float = 1.0,
    a: float = 1.0,
    tau: float = 1.0,
) -> List:
    """Upgrade gene-level hierarchical p/phi to NEG.

    Finds the hierarchical triplet (hyper_loc, hyper_scale, hier-p) produced
    by ``_hierarchicalize_p``, replaces hyper_scale (SoftplusNormalSpec) with
    the NEG pair (zeta, psi), and replaces the
    ``HierarchicalSigmoidNormalSpec``/``HierarchicalPositiveNormalSpec`` with the
    corresponding NEG spec.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_hierarchicalize_p`` has run.
    param_key : str
        Parameterization key.
    u, a, tau : float
        NEG hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with NEG p.
    """
    if param_key == "mean_odds":
        target_name = "phi"
        scale_name = "log_phi_scale"
        loc_name = "log_phi_loc"
        prefix = "phi"
    else:
        target_name = "p"
        scale_name = "logit_p_scale"
        loc_name = "logit_p_loc"
        prefix = "p"

    raw_name = f"{target_name}_raw"
    zeta_spec, psi_spec = _make_neg_hypers(prefix, u, a, tau)

    new_specs = []
    for spec in param_specs:
        if spec.name == scale_name:
            # Replace the SoftplusNormal hyper_scale with NEG pair (zeta, psi)
            new_specs.extend([zeta_spec, psi_spec])
        elif spec.name == target_name and isinstance(
            spec,
            (HierarchicalSigmoidNormalSpec, HierarchicalPositiveNormalSpec),
        ):
            # Select NEG spec matching the original transform:
            # ExpNormal (phi, range (0,inf)) vs SigmoidNormal (p, range (0,1))
            if isinstance(spec, HierarchicalPositiveNormalSpec):
                NEGSpec = NEGHierarchicalPositiveNormalSpec
            else:
                NEGSpec = NEGHierarchicalSigmoidNormalSpec
            neg_spec = NEGSpec(
                name=target_name,
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name=loc_name,
                hyper_scale_name=f"psi_{prefix}",
                psi_name=f"psi_{prefix}",
                zeta_name=f"zeta_{prefix}",
                raw_name=raw_name,
                is_gene_specific=spec.is_gene_specific,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
                transform=spec.transform,
            )
            new_specs.append(neg_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Gene-level NEG gate
# ------------------------------------------------------------------------------


def _neg_gate(
    param_specs: List,
    u: float = 1.0,
    a: float = 1.0,
    tau: float = 1.0,
) -> List:
    """Upgrade gene-level hierarchical gate to NEG.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``build_gate_spec(hierarchical=True)`` has run.
    u, a, tau : float
        NEG hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with NEG gate.
    """
    zeta_spec, psi_spec = _make_neg_hypers("gate", u, a, tau)

    new_specs = []
    for spec in param_specs:
        if spec.name == "logit_gate_scale":
            new_specs.extend([zeta_spec, psi_spec])
        elif spec.name == "gate" and isinstance(
            spec, HierarchicalSigmoidNormalSpec
        ):
            neg_spec = NEGHierarchicalSigmoidNormalSpec(
                name="gate",
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name="logit_gate_loc",
                hyper_scale_name="psi_gate",
                psi_name="psi_gate",
                zeta_name="zeta_gate",
                raw_name="gate_raw",
                is_gene_specific=spec.is_gene_specific,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
            )
            new_specs.append(neg_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Dataset-level NEG mu
# ------------------------------------------------------------------------------


def _neg_dataset_mu(
    param_specs: List,
    param_key: str,
    u: float = 1.0,
    a: float = 1.0,
    tau: float = 1.0,
) -> List:
    """Upgrade dataset-level hierarchical mu/r to NEG.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_datasetify_mu`` has run.
    param_key : str
        Parameterization key.
    u, a, tau : float
        NEG hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with NEG dataset mu.
    """
    if param_key in ("mean_odds", "mean_prob"):
        target_name = "mu"
        scale_name = "log_mu_dataset_scale"
        loc_name = "log_mu_dataset_loc"
        prefix = "mu_dataset"
    else:
        target_name = "r"
        scale_name = "log_r_dataset_scale"
        loc_name = "log_r_dataset_loc"
        prefix = "r_dataset"

    raw_name = f"{target_name}_raw"
    zeta_spec, psi_spec = _make_neg_hypers(prefix, u, a, tau)

    new_specs = []
    for spec in param_specs:
        if spec.name == scale_name:
            new_specs.extend([zeta_spec, psi_spec])
        elif spec.name == target_name and isinstance(
            spec, DatasetHierarchicalPositiveNormalSpec
        ):
            neg_spec = NEGDatasetPositiveNormalSpec(
                name=target_name,
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name=loc_name,
                hyper_scale_name=f"psi_{prefix}",
                psi_name=f"psi_{prefix}",
                zeta_name=f"zeta_{prefix}",
                raw_name=raw_name,
                is_gene_specific=spec.is_gene_specific,
                is_dataset=spec.is_dataset,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
                transform=spec.transform,
            )
            new_specs.append(neg_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Dataset-level NEG p
# ------------------------------------------------------------------------------


def _neg_dataset_p(
    param_specs: List,
    param_key: str,
    u: float = 1.0,
    a: float = 1.0,
    tau: float = 1.0,
) -> List:
    """Upgrade dataset-level hierarchical p/phi to NEG.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_datasetify_p`` has run.
    param_key : str
        Parameterization key.
    u, a, tau : float
        NEG hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with NEG dataset p.
    """
    if param_key == "mean_odds":
        target_name = "phi"
        scale_name = "log_phi_dataset_scale"
        loc_name = "log_phi_dataset_loc"
        prefix = "phi_dataset"
    else:
        target_name = "p"
        scale_name = "logit_p_dataset_scale"
        loc_name = "logit_p_dataset_loc"
        prefix = "p_dataset"

    raw_name = f"{target_name}_raw_dataset"
    zeta_spec, psi_spec = _make_neg_hypers(prefix, u, a, tau)

    # Determine the correct NEG spec class based on the target transform
    if param_key == "mean_odds":
        TargetHierClass = DatasetHierarchicalPositiveNormalSpec
        NEGClass = NEGDatasetPositiveNormalSpec
    else:
        TargetHierClass = DatasetHierarchicalSigmoidNormalSpec
        NEGClass = NEGDatasetSigmoidNormalSpec

    new_specs = []
    for spec in param_specs:
        if spec.name == scale_name:
            new_specs.extend([zeta_spec, psi_spec])
        elif spec.name == target_name and isinstance(spec, TargetHierClass):
            neg_spec = NEGClass(
                name=target_name,
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name=loc_name,
                hyper_scale_name=f"psi_{prefix}",
                psi_name=f"psi_{prefix}",
                zeta_name=f"zeta_{prefix}",
                raw_name=raw_name,
                is_gene_specific=spec.is_gene_specific,
                is_dataset=spec.is_dataset,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
                transform=spec.transform,
            )
            new_specs.append(neg_spec)
        else:
            new_specs.append(spec)
    return new_specs


# ------------------------------------------------------------------------------
# Dataset-level NEG gate
# ------------------------------------------------------------------------------


def _neg_dataset_gate(
    param_specs: List,
    u: float = 1.0,
    a: float = 1.0,
    tau: float = 1.0,
) -> List:
    """Upgrade dataset-level hierarchical gate to NEG.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Specs after ``_datasetify_gate`` has run.
    u, a, tau : float
        NEG hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated specs with NEG dataset gate.
    """
    zeta_spec, psi_spec = _make_neg_hypers("gate_dataset", u, a, tau)

    new_specs = []
    for spec in param_specs:
        if spec.name == "logit_gate_dataset_scale":
            new_specs.extend([zeta_spec, psi_spec])
        elif spec.name == "gate" and isinstance(
            spec, DatasetHierarchicalSigmoidNormalSpec
        ):
            neg_spec = NEGDatasetSigmoidNormalSpec(
                name="gate",
                shape_dims=spec.shape_dims,
                default_params=spec.default_params,
                hyper_loc_name="logit_gate_dataset_loc",
                hyper_scale_name="psi_gate_dataset",
                psi_name="psi_gate_dataset",
                zeta_name="zeta_gate_dataset",
                raw_name="gate_raw_dataset",
                is_gene_specific=spec.is_gene_specific,
                is_dataset=spec.is_dataset,
                is_mixture=spec.is_mixture,
                guide_family=getattr(spec, "guide_family", None),
            )
            new_specs.append(neg_spec)
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
