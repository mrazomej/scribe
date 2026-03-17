"""Base model configuration classes using Pydantic."""

import math
from typing import Optional, Set, Dict, Any, List, Tuple
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    computed_field,
    ConfigDict,
)
import warnings

from .enums import (
    ModelType,
    Parameterization,
    InferenceMethod,
    HierarchicalPriorType,
)
from .groups import (
    VAEConfig,
    GuideFamilyConfig,
    PriorOverrides,
)
from .parameter_mapping import get_active_parameters
from ..builders.parameter_specs import ParamSpec

# ==============================================================================
# Unified Model Configuration Class
# ==============================================================================


class ModelConfig(BaseModel):
    """
    Unified model configuration for SCRIBE models.

    This class defines the complete set of configuration options for SCRIBE
    models, supporting both constrained (interpretable) and unconstrained
    (unconstrained) parameterizations. The class enforces correctness,
    immutability, and clarity of model setup.

    This is not constructed directly; users should assemble configurations via
    the ModelConfigBuilder, which ensures validity and best practices.

    The actual model and guide functions are created by the unified factory
    `create_model(model_config)` in `scribe.models.presets.factory`.

    Parameters
    ----------
    base_model : str
        The core model family (e.g., 'nbdm', 'zinb', 'nbvcp', 'zinbvcp').
    parameterization : Parameterization
        How parameters are represented internally (standard, linked, etc.).
    inference_method : InferenceMethod
        Inference engine type (SVI, MCMC, VAE, etc.).
    unconstrained : bool, default=False
        If True, use unconstrained parameterization (Normal distributions).
        If False, use constrained parameterization (Beta, LogNormal, etc.).
    n_components : int, optional
        Number of mixture components, if mixture modeling is enabled.
    mixture_params : List[str], optional
        List of parameter names that should be mixture-specific. If None and
        n_components is set, all sampled core parameters for the selected
        parameterization will be mixture-specific by default.
    guide_families : GuideFamilyConfig, optional
        Per-parameter guide family configuration. Allows specifying different
        variational families (MeanField, LowRank, Amortized) for each parameter.
    param_specs : List[ParamSpec], optional
        Optional list of parameter specifications for user-provided overrides.
        When provided, these can contain custom prior/guide hyperparameters
        that override the defaults. The unified factory uses these to customize
        the model. If empty (default), the factory uses default hyperparameters.
    vae : VAEConfig, optional
        Nested configuration for Variational Autoencoders, if applicable.

    Notes
    -----
    - Configuration objects are immutable and validated automatically on
      creation.
    - Unrecognized parameters are forbidden and will raise validation errors.
    - Intended for safe, reproducible, and fully-specified SCRIBE model setups.
    - The `param_specs` field is primarily for user overrides. The unified
      factory (`create_model`) constructs the complete parameter specifications
      based on the model type and parameterization, then applies any overrides
      from `param_specs`.

    See Also
    --------
    scribe.models.presets.factory.create_model : Creates model/guide from config.
    ModelConfigBuilder : Builder for creating ModelConfig objects.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ------------------------------------------------------------------
    # Pickle backward compatibility: translate old boolean flags
    # (hierarchical_p, horseshoe_p, etc.) into the new
    # HierarchicalPriorType enum fields.  Also backfills any fields
    # that may be absent in very old pickle payloads.
    # ------------------------------------------------------------------

    def __setstate__(self, state: dict) -> None:
        d = state.get("__dict__", {})

        # --- Migrate old boolean prior flags → enum fields ---------------
        # Gene-level p/phi
        if "hierarchical_p" in d or "horseshoe_p" in d:
            hp = d.pop("hierarchical_p", False)
            hs = d.pop("horseshoe_p", False)
            if hs:
                d.setdefault("p_prior", "horseshoe")
            elif hp:
                d.setdefault("p_prior", "gaussian")
            else:
                d.setdefault("p_prior", "none")

        # Gene-level gate
        if "hierarchical_gate" in d or "horseshoe_gate" in d:
            hg = d.pop("hierarchical_gate", False)
            hsg = d.pop("horseshoe_gate", False)
            if hsg:
                d.setdefault("gate_prior", "horseshoe")
            elif hg:
                d.setdefault("gate_prior", "gaussian")
            else:
                d.setdefault("gate_prior", "none")

        # Dataset-level mu
        if "hierarchical_dataset_mu" in d or "horseshoe_dataset_mu" in d:
            hdm = d.pop("hierarchical_dataset_mu", False)
            hsdm = d.pop("horseshoe_dataset_mu", False)
            if hsdm:
                d.setdefault("mu_dataset_prior", "horseshoe")
            elif hdm:
                d.setdefault("mu_dataset_prior", "gaussian")
            else:
                d.setdefault("mu_dataset_prior", "none")

        # Dataset-level p/phi: old field was a string mode
        # ("none"/"scalar"/"gene_specific"/"two_level") + boolean horseshoe
        if "hierarchical_dataset_p" in d or "horseshoe_dataset_p" in d:
            old_mode = d.pop("hierarchical_dataset_p", "none")
            hsdp = d.pop("horseshoe_dataset_p", False)
            if hsdp:
                d.setdefault("p_dataset_prior", "horseshoe")
                d.setdefault("p_dataset_mode", "gene_specific")
            elif old_mode != "none":
                d.setdefault("p_dataset_prior", "gaussian")
                d.setdefault("p_dataset_mode", old_mode)
            else:
                d.setdefault("p_dataset_prior", "none")

        # Dataset-level gate
        if "hierarchical_dataset_gate" in d or "horseshoe_dataset_gate" in d:
            hdg = d.pop("hierarchical_dataset_gate", False)
            hsdg = d.pop("horseshoe_dataset_gate", False)
            if hsdg:
                d.setdefault("gate_dataset_prior", "horseshoe")
            elif hdg:
                d.setdefault("gate_dataset_prior", "gaussian")
            else:
                d.setdefault("gate_dataset_prior", "none")

        # Remove the old hierarchical_datasets shortcut if present
        d.pop("hierarchical_datasets", None)

        # --- Backfill missing fields for very old pickles ----------------
        d.setdefault("n_datasets", None)
        d.setdefault("dataset_params", None)
        # Migrate old hierarchical_mu boolean → mu_prior enum
        if "hierarchical_mu" in d:
            old_hmu = d.pop("hierarchical_mu", False)
            if old_hmu:
                d.setdefault("mu_prior", "gaussian")
            else:
                d.setdefault("mu_prior", "none")
        d.setdefault("mu_prior", "none")
        d.setdefault("shared_capture_scaling", False)
        d.setdefault("joint_params", None)
        # Old pickles predate the softplus default; preserve exp behavior
        d.setdefault("positive_transform", "exp")
        d.setdefault("p_prior", "none")
        d.setdefault("gate_prior", "none")
        d.setdefault("mu_dataset_prior", "none")
        d.setdefault("p_dataset_prior", "none")
        d.setdefault("p_dataset_mode", "gene_specific")
        d.setdefault("gate_dataset_prior", "none")

        # Migrate legacy top-level capture prior fields into priors dict.
        old_capture = d.pop("capture_prior", "default")
        old_organism = d.pop("organism", None)
        old_mrna_mean = d.pop("total_mrna_mean", None)
        old_mrna_sigma = d.pop("total_mrna_log_sigma", None)
        d.pop("total_mrna_mean_bounds", None)

        priors_obj = d.get("priors")
        if priors_obj is None:
            priors_obj = PriorOverrides()
            d["priors"] = priors_obj

        extra = getattr(priors_obj, "__pydantic_extra__", None) or {}
        if old_capture == "biology_informed" or old_mrna_mean is not None:
            if "eta_capture" not in extra:
                log_M0 = math.log(old_mrna_mean) if old_mrna_mean else 12.2
                sigma_M = old_mrna_sigma if old_mrna_sigma else 0.5
                extra["eta_capture"] = (log_M0, sigma_M)
        if old_organism is not None and "organism" not in extra:
            extra["organism"] = old_organism
        if extra:
            d["priors"] = PriorOverrides(**extra)

        super().__setstate__(state)

    # Core configuration
    base_model: str = Field(
        ..., description="Model type (e.g., 'nbdm', 'zinb_mix')"
    )
    parameterization: Parameterization = Field(
        Parameterization.STANDARD, description="Parameterization type"
    )
    inference_method: InferenceMethod = Field(
        InferenceMethod.SVI, description="Inference method"
    )
    unconstrained: bool = Field(
        False, description="Use unconstrained parameterization"
    )

    # Hierarchical prior type for each parameter slot.
    # Each field selects the prior family for one parameter at one level.
    p_prior: HierarchicalPriorType = Field(
        HierarchicalPriorType.NONE,
        description=(
            "Gene-level hierarchical prior for p/phi. "
            "Requires unconstrained=True for non-NONE values."
        ),
    )
    gate_prior: HierarchicalPriorType = Field(
        HierarchicalPriorType.NONE,
        description=(
            "Gene-level hierarchical prior for gate. "
            "Requires a zero-inflated model and unconstrained=True."
        ),
    )
    mu_prior: HierarchicalPriorType = Field(
        HierarchicalPriorType.NONE,
        description=(
            "Gene-level hierarchical prior for mu (or r) across mixture "
            "components. Provides shrinkage so that per-component means "
            "are drawn from a shared gene-level population distribution. "
            "Requires unconstrained=True and a mixture model "
            "(n_components >= 2). Each gene has its own hyperprior because "
            "expression magnitudes vary by orders of magnitude across genes."
        ),
    )

    # Mixture configuration
    n_components: Optional[int] = Field(
        None, gt=1, description="Number of mixture components"
    )
    mixture_params: Optional[List[str]] = Field(
        None,
        description="List of parameter names that should be mixture-specific",
    )

    # Multi-dataset configuration
    n_datasets: Optional[int] = Field(
        None,
        gt=1,
        description=(
            "Number of datasets for joint multi-dataset modeling. "
            "Dataset assignments are fixed (observed), not inferred."
        ),
    )
    dataset_params: Optional[List[str]] = Field(
        None,
        description=(
            "List of parameter names that should be per-dataset. "
            "Analogous to mixture_params but for the dataset axis."
        ),
    )
    # Dataset-level hierarchical prior types
    mu_dataset_prior: HierarchicalPriorType = Field(
        HierarchicalPriorType.NONE,
        description=(
            "Dataset-level hierarchical prior for mu (or r). "
            "Requires n_datasets >= 2 and unconstrained=True."
        ),
    )
    p_dataset_prior: HierarchicalPriorType = Field(
        HierarchicalPriorType.NONE,
        description=(
            "Dataset-level hierarchical prior for p/phi. "
            "Requires n_datasets >= 2 and unconstrained=True."
        ),
    )
    p_dataset_mode: str = Field(
        "gene_specific",
        description=(
            "Structural mode for the dataset-level p/phi hierarchy. "
            "Only used when p_dataset_prior is GAUSSIAN. Options: "
            "'scalar' (one p per dataset, shared across genes), "
            "'gene_specific' (p_g per dataset-gene pair), "
            "'two_level' (two-level hierarchy with dataset hyperparameters). "
            "Horseshoe and NEG priors always use 'gene_specific'."
        ),
    )
    gate_dataset_prior: HierarchicalPriorType = Field(
        HierarchicalPriorType.NONE,
        description=(
            "Dataset-level hierarchical prior for gate. "
            "Requires n_datasets >= 2, unconstrained=True, "
            "and a zero-inflated model."
        ),
    )

    # Horseshoe hyperparameters (shared by all horseshoe priors)
    horseshoe_tau0: float = Field(
        1.0,
        description="Global shrinkage scale for horseshoe Half-Cauchy prior.",
    )
    horseshoe_slab_df: int = Field(
        4,
        description="Degrees of freedom for horseshoe slab Inverse-Gamma.",
    )
    horseshoe_slab_scale: float = Field(
        2.0,
        description="Scale for horseshoe slab Inverse-Gamma.",
    )

    # NEG (Normal-Exponential-Gamma) hyperparameters.
    # The NEG prior is a member of the TPBN family with a Gamma-Gamma
    # hierarchy: psi_g | zeta_g ~ Gamma(u, zeta_g),
    #            zeta_g ~ Gamma(a, tau).
    # u=1 gives NEG (inner layer is Exponential), u=0.5 recovers horseshoe.
    neg_u: float = Field(
        1.0,
        description=(
            "TPBN shape parameter for the inner Gamma layer. "
            "u=1 is NEG (Exponential mixing), u=0.5 is horseshoe-like."
        ),
    )
    neg_a: float = Field(
        1.0,
        description=(
            "TPBN tail parameter for the outer Gamma layer. "
            "Controls polynomial tail weight."
        ),
    )
    neg_tau: float = Field(
        1.0,
        description=(
            "Global shrinkage rate for the outer Gamma (zeta ~ Gamma(a, tau))."
        ),
    )

    # Positive-parameter transform for hierarchical specs.
    # Controls how unconstrained Normal samples are mapped to (0, inf).
    positive_transform: str = Field(
        "softplus",
        description=(
            "Transform for positive-valued hierarchical parameters "
            "(phi, mu, r). 'softplus' (default) prevents float32 overflow "
            "via log(1+exp(z)); 'exp' restores the original log-Normal "
            "behavior via exp(z). Power users may prefer 'exp' for exact "
            "log-Normal priors at the cost of numerical stability."
        ),
    )

    # Component matching for multi-dataset mixtures.
    # When annotation_key and dataset_key are both provided, labels in 2+
    # datasets are automatically treated as "shared" and get dataset-level
    # hierarchy.  This field lets the user override that automatic detection.
    shared_components: Optional[List[str]] = Field(
        None,
        description=(
            "Manual list of component labels that are shared across "
            "datasets. Only used when annotation_key and dataset_key are "
            "both provided. Overrides automatic detection (which considers "
            "labels appearing in 2+ datasets as shared). Labels not in "
            "this list are treated as dataset-specific: their dataset "
            "hierarchy scale is clamped, suppressing inter-dataset "
            "variation for those components."
        ),
    )

    # Runtime-populated: integer indices of shared components, derived
    # from ComponentMapping in fit().  The factory reads this to build
    # per-component scale masking in dataset hierarchical specs.
    shared_component_indices: Optional[Tuple[int, ...]] = Field(
        None,
        description=(
            "Runtime field: component indices shared across 2+ datasets. "
            "Populated by fit() from the ComponentMapping; not set by users."
        ),
    )

    # Biology-informed capture prior configuration.
    # The capture prior is configured via the priors section:
    #   priors.organism    — shortcut to set defaults (e.g. "human")
    #   priors.eta_capture — [log_M0, sigma_M] per-cell prior
    #   priors.mu_eta      — [center, sigma_mu] shared mu_eta prior
    # The biology-informed path activates automatically when any of
    # these keys is present.
    shared_capture_scaling: bool = Field(
        False,
        description=(
            "When True, learn a shared mu_eta parameter across "
            "datasets/components instead of using a fixed M_0. "
            "Requires priors.organism or priors.eta_capture to be set. "
            "priors.mu_eta controls the prior on the shared parameter."
        ),
    )

    # Joint low-rank guide configuration
    joint_params: Optional[List[str]] = Field(
        None,
        description=(
            "List of parameter names to model jointly via "
            "JointLowRankGuide (e.g. ['mu', 'phi', 'gate']). Requires "
            "guide_rank to be set via guide_families. All listed "
            "parameters share a single low-rank covariance structure "
            "capturing cross-parameter correlations. Supports "
            "heterogeneous dimensions: scalar parameters (e.g. phi "
            "when p_prior='none') can be mixed with gene-specific "
            "parameters (e.g. mu, gate)."
        ),
    )

    # Guide configuration
    guide_families: Optional[GuideFamilyConfig] = Field(
        None,
        description="Per-parameter guide family configuration",
    )

    # Parameter specifications (optional user overrides)
    param_specs: List[ParamSpec] = Field(
        default_factory=list,
        description=(
            "Optional list of parameter specifications for user-provided "
            "prior/guide hyperparameter overrides. The unified factory uses "
            "these to customize default parameters."
        ),
    )

    # Prior overrides (populated by ModelConfigBuilder with defaults)
    priors: PriorOverrides = Field(
        default_factory=PriorOverrides,
        description="Prior hyperparameters (e.g., Beta, LogNormal) per parameter.",
    )

    vae: Optional[VAEConfig] = Field(
        None, description="VAE configuration (if using VAE inference)"
    )

    # --------------------------------------------------------------------------
    # Validation Methods
    # --------------------------------------------------------------------------

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, v: str) -> str:
        """Validate base model type."""
        base = v.replace("_mix", "")
        valid_models = {m.value for m in ModelType}
        if base not in valid_models:
            raise ValueError(
                f"Invalid model type: {v}. Must be one of {valid_models}"
            )
        return v

    # --------------------------------------------------------------------------

    @field_validator("vae")
    @classmethod
    def validate_vae_inference(
        cls, v: Optional[VAEConfig], info
    ) -> Optional[VAEConfig]:
        """Validate VAE config is provided for VAE inference."""
        if (
            info.data.get("inference_method") == InferenceMethod.VAE
            and v is None
        ):
            # Provide default VAE config
            return VAEConfig()
        return v

    # --------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_hierarchical_flags(self) -> "ModelConfig":
        """Validate hierarchical prior configuration.

        With the enum-based config, mutual-exclusivity errors are
        impossible by construction (each slot is a single enum value).
        Validation focuses on structural constraints:

        - Sparsity priors (HORSESHOE, NEG) require ``unconstrained=True``.
        - Gate priors require a zero-inflated model.
        - Dataset-level priors require ``n_datasets >= 2``.
        - Gene-level and dataset-level p/gate priors are mutually exclusive.
        - ``positive_transform`` must be ``"softplus"`` or ``"exp"``.
        """
        _NONE = HierarchicalPriorType.NONE

        # --- positive_transform validation ------------------------------------
        valid_transforms = {"softplus", "exp"}
        if self.positive_transform not in valid_transforms:
            raise ValueError(
                f"positive_transform must be one of {valid_transforms}, "
                f"got {self.positive_transform!r}."
            )

        # --- Gene-level p/phi ------------------------------------------------
        if self.p_prior != _NONE and not self.unconstrained:
            raise ValueError(
                f"p_prior={self.p_prior.value!r} requires "
                "unconstrained=True."
            )

        # --- Gene-level gate -------------------------------------------------
        if self.gate_prior != _NONE:
            if not self.is_zero_inflated:
                raise ValueError(
                    f"gate_prior={self.gate_prior.value!r} requires a "
                    "zero-inflated model (zinb or zinbvcp), but "
                    f"base_model={self.base_model!r}."
                )
            if not self.unconstrained:
                raise ValueError(
                    f"gate_prior={self.gate_prior.value!r} requires "
                    "unconstrained=True."
                )

        # --- Gene-level mu (mixture components) --------------------------------
        if self.mu_prior != _NONE:
            if not self.unconstrained:
                raise ValueError(
                    f"mu_prior={self.mu_prior.value!r} requires "
                    "unconstrained=True."
                )
            if not self.is_mixture:
                raise ValueError(
                    f"mu_prior={self.mu_prior.value!r} requires a mixture "
                    "model (n_components >= 2). The hierarchical prior on "
                    "mu provides shrinkage across mixture components."
                )
            if self.mu_dataset_prior != _NONE:
                raise ValueError(
                    f"mu_prior={self.mu_prior.value!r} and "
                    f"mu_dataset_prior={self.mu_dataset_prior.value!r} "
                    "cannot be set simultaneously. mu_prior provides "
                    "shrinkage across mixture components; "
                    "mu_dataset_prior across datasets."
                )

        # --- p_dataset_mode validation ---------------------------------------
        valid_modes = {"scalar", "gene_specific", "two_level"}
        if self.p_dataset_mode not in valid_modes:
            raise ValueError(
                f"p_dataset_mode must be one of {valid_modes}, "
                f"got {self.p_dataset_mode!r}."
            )

        # --- Dataset-level mu ------------------------------------------------
        if self.mu_dataset_prior != _NONE:
            if self.n_datasets is None:
                raise ValueError(
                    f"mu_dataset_prior={self.mu_dataset_prior.value!r} "
                    "requires n_datasets >= 2."
                )
            if not self.unconstrained:
                raise ValueError(
                    f"mu_dataset_prior={self.mu_dataset_prior.value!r} "
                    "requires unconstrained=True."
                )

        # --- Dataset-level p/phi ---------------------------------------------
        if self.p_dataset_prior != _NONE:
            if self.n_datasets is None:
                raise ValueError(
                    f"p_dataset_prior={self.p_dataset_prior.value!r} "
                    "requires n_datasets >= 2."
                )
            if not self.unconstrained:
                raise ValueError(
                    f"p_dataset_prior={self.p_dataset_prior.value!r} "
                    "requires unconstrained=True."
                )

        # --- Dataset-level gate ----------------------------------------------
        if self.gate_dataset_prior != _NONE:
            if self.n_datasets is None:
                raise ValueError(
                    f"gate_dataset_prior={self.gate_dataset_prior.value!r} "
                    "requires n_datasets >= 2."
                )
            if not self.unconstrained:
                raise ValueError(
                    f"gate_dataset_prior={self.gate_dataset_prior.value!r} "
                    "requires unconstrained=True."
                )
            if not self.is_zero_inflated:
                raise ValueError(
                    f"gate_dataset_prior={self.gate_dataset_prior.value!r} "
                    "requires a zero-inflated model (zinb or zinbvcp), "
                    f"but base_model={self.base_model!r}."
                )

        # --- Cross-level conflicts -------------------------------------------
        # Gene-level p + dataset-level p are mutually exclusive
        if self.p_prior != _NONE and self.p_dataset_prior != _NONE:
            raise ValueError(
                f"p_prior={self.p_prior.value!r} and "
                f"p_dataset_prior={self.p_dataset_prior.value!r} "
                "cannot be set simultaneously. The dataset-level "
                "hierarchy subsumes gene-level."
            )
        # Gene-level gate + dataset-level gate are mutually exclusive
        if self.gate_prior != _NONE and self.gate_dataset_prior != _NONE:
            raise ValueError(
                f"gate_prior={self.gate_prior.value!r} and "
                f"gate_dataset_prior={self.gate_dataset_prior.value!r} "
                "cannot be set simultaneously. The dataset-level "
                "hierarchy subsumes gene-level."
            )

        # shared_components validation: requires multi-dataset mixture setup
        if self.shared_components is not None:
            if self.n_datasets is None:
                raise ValueError(
                    "shared_components requires n_datasets >= 2."
                )
        return self

    # --------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_capture_prior(self) -> "ModelConfig":
        """Resolve and validate biology-informed capture prior configuration.

        The capture prior is inferred from keys in the ``priors`` dict:

        * ``priors.organism``    — resolves to default ``eta_capture``
        * ``priors.eta_capture`` — explicit ``[log_M0, sigma_M]``
        * ``priors.mu_eta``      — explicit ``[center, sigma_mu]``

        When any of these is present the biology-informed path activates.
        ``shared_capture_scaling`` additionally learns a shared ``mu_eta``.
        """
        extra = getattr(self.priors, "__pydantic_extra__", None) or {}

        organism: Optional[str] = extra.get("organism")
        eta_capture: Optional[Tuple[float, float]] = extra.get("eta_capture")
        mu_eta: Optional[Tuple[float, float]] = extra.get("mu_eta")

        # Resolve organism → eta_capture defaults when eta_capture absent
        if organism is not None and eta_capture is None:
            from .organism_priors import resolve_organism_priors

            org_priors = resolve_organism_priors(organism)
            log_M0 = math.log(org_priors["total_mrna_mean"])
            sigma_M = org_priors["total_mrna_log_sigma"]
            eta_capture = (log_M0, sigma_M)

        has_capture_priors = eta_capture is not None or mu_eta is not None

        # VCP model required for capture priors or shared scaling
        if has_capture_priors or self.shared_capture_scaling:
            if not self.uses_variable_capture:
                raise ValueError(
                    "Biology-informed capture priors (priors.organism, "
                    "priors.eta_capture, priors.mu_eta) or "
                    "shared_capture_scaling=True requires a VCP model "
                    "(nbvcp or zinbvcp)."
                )

        # Default mu_eta prior when shared scaling is on and we have an
        # anchor from eta_capture.  sigma_mu defaults to 1.0 (anchored)
        # or 5.0 (vague, no anchor).
        if self.shared_capture_scaling and eta_capture is not None:
            if mu_eta is None:
                mu_eta = (eta_capture[0], 1.0)

        # Store resolved values back into priors (frozen model bypass)
        updated = dict(extra)
        if eta_capture is not None:
            updated["eta_capture"] = tuple(eta_capture)
        if mu_eta is not None:
            updated["mu_eta"] = tuple(mu_eta)
        if updated != extra:
            new_priors = PriorOverrides(**updated)
            object.__setattr__(self, "priors", new_priors)

        return self

    # --------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_param_specs_consistency(self) -> "ModelConfig":
        """Validate that param_specs overrides are consistent with config.

        This validates:
        1. Any provided param_specs have consistent unconstrained flag
        2. Any provided param_specs are valid for this model type

        Note: param_specs is optional and may be empty. The unified factory
        constructs complete specs based on model type/parameterization.
        """
        if not self.param_specs:
            # No overrides provided, nothing to validate
            return self

        # Check unconstrained consistency for any provided specs
        for spec in self.param_specs:
            if spec.unconstrained != self.unconstrained:
                raise ValueError(
                    f"Parameter '{spec.name}': "
                    f"unconstrained flag ({spec.unconstrained}) "
                    f"must match ModelConfig.unconstrained "
                    f"({self.unconstrained})"
                )

        # Validate provided params are valid for this model
        is_mixture = self.n_components is not None
        is_zero_inflated = self.is_zero_inflated
        uses_variable_capture = self.uses_variable_capture

        provided_params = {spec.name for spec in self.param_specs}
        active_params = get_active_parameters(
            self.parameterization,
            self.base_model,
            is_mixture,
            is_zero_inflated,
            uses_variable_capture,
            hierarchical_mu=self.mu_prior != HierarchicalPriorType.NONE,
            hierarchical_p=self.hierarchical_p,
            hierarchical_gate=self.hierarchical_gate,
        )

        unexpected_params = provided_params - active_params
        if unexpected_params:
            raise ValueError(
                f"Unexpected parameters for {self.base_model} with "
                f"{self.parameterization.value} parameterization: "
                f"{', '.join(sorted(unexpected_params))}"
            )

        return self

    # --------------------------------------------------------------------------
    # Deprecated backward-compat property accessors.
    # These map old boolean flag names to the new enum fields so that
    # existing call sites, tests, and documentation continue to work.
    # New code should use p_prior, gate_prior, etc. directly.
    # --------------------------------------------------------------------------

    @property
    def hierarchical_mu(self) -> bool:
        """Whether gene-level mu uses any hierarchical prior.

        .. deprecated::
            Use ``mu_prior`` instead.
        """
        return self.mu_prior != HierarchicalPriorType.NONE

    @property
    def hierarchical_p(self) -> bool:
        """Whether gene-level p/phi uses any hierarchical prior.

        .. deprecated::
            Use ``p_prior`` instead.
        """
        return self.p_prior != HierarchicalPriorType.NONE

    @property
    def horseshoe_p(self) -> bool:
        """Whether gene-level p/phi uses the horseshoe prior.

        .. deprecated::
            Use ``p_prior`` instead.
        """
        return self.p_prior == HierarchicalPriorType.HORSESHOE

    @property
    def hierarchical_gate(self) -> bool:
        """Whether gene-level gate uses any hierarchical prior.

        .. deprecated::
            Use ``gate_prior`` instead.
        """
        return self.gate_prior != HierarchicalPriorType.NONE

    @property
    def horseshoe_gate(self) -> bool:
        """Whether gene-level gate uses the horseshoe prior.

        .. deprecated::
            Use ``gate_prior`` instead.
        """
        return self.gate_prior == HierarchicalPriorType.HORSESHOE

    @property
    def hierarchical_dataset_mu(self) -> bool:
        """Whether dataset-level mu uses any hierarchical prior.

        .. deprecated::
            Use ``mu_dataset_prior`` instead.
        """
        return self.mu_dataset_prior != HierarchicalPriorType.NONE

    @property
    def horseshoe_dataset_mu(self) -> bool:
        """Whether dataset-level mu uses the horseshoe prior.

        .. deprecated::
            Use ``mu_dataset_prior`` instead.
        """
        return self.mu_dataset_prior == HierarchicalPriorType.HORSESHOE

    @property
    def hierarchical_dataset_p(self) -> str:
        """Dataset-level p/phi hierarchy mode string.

        Returns ``"none"`` when ``p_dataset_prior`` is ``NONE``.
        For horseshoe/NEG returns ``"gene_specific"``.
        Otherwise returns ``p_dataset_mode``.

        .. deprecated::
            Use ``p_dataset_prior`` and ``p_dataset_mode`` instead.
        """
        if self.p_dataset_prior == HierarchicalPriorType.NONE:
            return "none"
        if self.p_dataset_prior in (
            HierarchicalPriorType.HORSESHOE,
            HierarchicalPriorType.NEG,
        ):
            return "gene_specific"
        return self.p_dataset_mode

    @property
    def horseshoe_dataset_p(self) -> bool:
        """Whether dataset-level p/phi uses the horseshoe prior.

        .. deprecated::
            Use ``p_dataset_prior`` instead.
        """
        return self.p_dataset_prior == HierarchicalPriorType.HORSESHOE

    @property
    def hierarchical_dataset_gate(self) -> bool:
        """Whether dataset-level gate uses any hierarchical prior.

        .. deprecated::
            Use ``gate_dataset_prior`` instead.
        """
        return self.gate_dataset_prior != HierarchicalPriorType.NONE

    @property
    def horseshoe_dataset_gate(self) -> bool:
        """Whether dataset-level gate uses the horseshoe prior.

        .. deprecated::
            Use ``gate_dataset_prior`` instead.
        """
        return self.gate_dataset_prior == HierarchicalPriorType.HORSESHOE

    @property
    def hierarchical_datasets(self) -> bool:
        """Whether any dataset-level hierarchy is active.

        .. deprecated::
            Use ``mu_dataset_prior``, ``p_dataset_prior``, or
            ``gate_dataset_prior`` instead.
        """
        _NONE = HierarchicalPriorType.NONE
        return any(
            getattr(self, f)
            != _NONE
            for f in (
                "mu_dataset_prior",
                "p_dataset_prior",
                "gate_dataset_prior",
            )
        )

    # --------------------------------------------------------------------------
    # Computed Fields
    # --------------------------------------------------------------------------

    @computed_field
    @property
    def is_mixture(self) -> bool:
        """Check if this is a mixture model."""
        return self.n_components is not None and self.n_components > 1

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def is_zero_inflated(self) -> bool:
        """Check if this is a zero-inflated model."""
        return "zinb" in self.base_model

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def uses_variable_capture(self) -> bool:
        """Check if this model uses variable capture."""
        return "vcp" in self.base_model

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def uses_biology_informed_capture(self) -> bool:
        """Whether the biology-informed capture prior path is active.

        True when any of priors.organism, priors.eta_capture, or
        priors.mu_eta is set, signalling that the model should use
        the eta_c parameterization instead of a flat prior.
        """
        extra = getattr(self.priors, "__pydantic_extra__", None) or {}
        return any(
            extra.get(k) is not None
            for k in ("organism", "eta_capture", "mu_eta")
        )

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def is_hierarchical(self) -> bool:
        """
        Check if this model uses any hierarchical prior (mu, p/phi, or gate).
        """
        _NONE = HierarchicalPriorType.NONE
        return (
            self.mu_prior != _NONE
            or self.p_prior != _NONE
            or self.gate_prior != _NONE
        )

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def is_multi_dataset(self) -> bool:
        """Check if this is a multi-dataset joint model."""
        return self.n_datasets is not None and self.n_datasets > 1

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def active_parameters(self) -> Set[str]:
        """Get the set of active parameters for this configuration."""
        return get_active_parameters(
            parameterization=self.parameterization,
            model_type=self.base_model,
            is_mixture=self.is_mixture,
            is_zero_inflated=self.is_zero_inflated,
            uses_variable_capture=self.uses_variable_capture,
            hierarchical_mu=self.mu_prior != HierarchicalPriorType.NONE,
            hierarchical_p=self.hierarchical_p,
            hierarchical_gate=self.hierarchical_gate,
        )

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def get_prior_overrides(self) -> Dict[str, Any]:
        """Extract prior overrides from priors field or param_specs.

        Returns a dictionary mapping parameter names to their prior
        hyperparameters. Prefers the priors field when it has content;
        otherwise falls back to param_specs.

        Returns
        -------
        Dict[str, Any]
            Dictionary of prior overrides, e.g., {"p": (2.0, 2.0)}.
        """
        # Prefer priors from the priors field when it has content
        extra = getattr(self.priors, "__pydantic_extra__", None)
        if extra:
            return dict(extra)
        # Fall back to param_specs
        priors = {}
        for spec in self.param_specs:
            if hasattr(spec, "prior") and spec.prior is not None:
                priors[spec.name] = spec.prior
        return priors

    # --------------------------------------------------------------------------

    def get_guide_overrides(self) -> Dict[str, Any]:
        """Extract guide overrides from param_specs.

        Returns a dictionary mapping parameter names to their guide
        hyperparameters. Only includes parameters that have explicit
        guide values set.

        Returns
        -------
        Dict[str, Any]
            Dictionary of guide overrides.
        """
        guides = {}
        for spec in self.param_specs:
            if hasattr(spec, "guide") and spec.guide is not None:
                guides[spec.name] = spec.guide
        return guides

    # --------------------------------------------------------------------------

    def get_active_priors(self) -> Dict[str, Any]:
        """Get active prior parameters as a dictionary.

        .. deprecated::
            Use `get_prior_overrides()` instead.
        """
        priors_dict = {}
        for spec in self.param_specs:
            if spec.prior is not None:
                priors_dict[f"{spec.name}_prior"] = spec.prior
        return priors_dict

    # --------------------------------------------------------------------------

    def with_updated_priors(self, **priors) -> "ModelConfig":
        """Create a new config with updated priors (immutable pattern).

        Parameters
        ----------
        **priors
            Prior parameters keyed by parameter name.
            Example: with_updated_priors(p=(2.0, 2.0), r=(1.0, 0.5))

        Returns
        -------
        ModelConfig
            New config with updated priors.
        """
        # Use priors field (populated by builder)
        current = self.get_prior_overrides()
        updated = {**current, **priors}
        return self.model_copy(update={"priors": PriorOverrides(**updated)})

    # --------------------------------------------------------------------------

    def with_updated_vae(self, **vae_params) -> "ModelConfig":
        """Create a new config with updated VAE parameters.

        Parameters
        ----------
        **vae_params
            VAE parameters to update.

        Returns
        -------
        ModelConfig
            New config with updated VAE settings.

        Raises
        ------
        ValueError
            If VAE config is None.
        """
        if self.vae is None:
            raise ValueError("Cannot update VAE config when VAE is None")
        return self.model_copy(
            update={"vae": self.vae.model_copy(update=vae_params)}
        )

    # --------------------------------------------------------------------------
    # Serialization Methods
    # --------------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize config to YAML string.

        Returns
        -------
        str
            YAML representation of the config.

        Examples
        --------
        >>> config = ModelConfigBuilder().for_model("nbdm").build()
        >>> yaml_str = config.to_yaml()
        >>> print(yaml_str)
        """
        import yaml

        # Use model_dump with mode="json" to get serializable dict
        # Exclude computed properties that shouldn't be serialized
        data = self.model_dump(
            mode="json",
            exclude={
                "is_mixture",
                "is_zero_inflated",
                "uses_variable_capture",
                "uses_biology_informed_capture",
                "is_hierarchical",
                "is_multi_dataset",
                "active_parameters",
            },
        )
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ModelConfig":
        """Deserialize config from YAML string.

        Parameters
        ----------
        yaml_str : str
            YAML string representation of the config.

        Returns
        -------
        ModelConfig
            Deserialized config object.

        Examples
        --------
        >>> yaml_str = '''
        ... base_model: nbdm
        ... parameterization: standard
        ... '''
        >>> config = ModelConfig.from_yaml(yaml_str)
        """
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)

    # --------------------------------------------------------------------------

    def to_yaml_file(self, path: str) -> None:
        """Save config to YAML file.

        Parameters
        ----------
        path : str
            Path to save the YAML file.

        Examples
        --------
        >>> config.to_yaml_file("model_config.yaml")
        """
        with open(path, "w") as f:
            f.write(self.to_yaml())

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml_file(cls, path: str) -> "ModelConfig":
        """Load config from YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        ModelConfig
            Loaded config object.

        Examples
        --------
        >>> config = ModelConfig.from_yaml_file("model_config.yaml")
        """
        with open(path) as f:
            return cls.from_yaml(f.read())
