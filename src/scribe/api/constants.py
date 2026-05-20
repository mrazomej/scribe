"""
Validation sets and deprecated alias mappings for the SCRIBE simplified API.

These constants define the accepted values for model, parameterization, and
inference method selectors in :func:`scribe.api.fit`.

**Single source of truth**: ``VALID_MODELS`` is derived from
:class:`scribe.models.config.enums.ModelType` so that the simplified
API automatically picks up new ``base_model`` entries — adding a new
model means adding ONE enum entry, not touching three separate
allowlists.

``LAPLACE_SUPPORTED_BASE_MODELS`` is the per-method capability
allowlist consulted by both
:func:`scribe.inference.laplace._run_laplace_inference` (the bridge
guard) and :func:`scribe.inference.preset_builder.build_config_from_preset`
(the preset-builder pre-flight check).  Adding a new Laplace-supported
model means adding ONE entry here.
"""

from ..models.config.enums import ModelType
from ..models.parameterizations import PARAMETERIZATIONS

# Valid model types accepted by ``fit(model=...)`` — derived from the
# ``ModelType`` enum so the two never drift out of sync (the audit
# round-7 weakness).
VALID_MODELS: set[str] = {m.value for m in ModelType}

# Deprecated model name aliases mapped to their canonical replacements.
_DEPRECATED_MODEL_ALIASES = {"nbdm_lnm": "lnm"}

# Derive valid parameterizations from the single source of truth in the
# parameterization registry.
VALID_PARAMETERIZATIONS = set(PARAMETERIZATIONS.keys())

# Valid inference methods accepted by ``fit(inference_method=...)``.
VALID_INFERENCE_METHODS = {"svi", "mcmc", "vae", "laplace"}

# Per-method capability allowlist: which ``base_model`` strings support
# ``inference_method="laplace"``.  Single source of truth consulted by
# both ``scribe.inference.laplace._run_laplace_inference`` and
# ``scribe.inference.preset_builder.build_config_from_preset``.
LAPLACE_SUPPORTED_BASE_MODELS: set[str] = {
    "pln",
    "nbln",
    "lnm",
    "lnmvcp",
    "twostate_ln_rate",
    # ``twostate_ln_logit`` is reserved for PR-2; engine dispatch
    # raises ``NotImplementedError`` until then but the API gate
    # accepts the string so the cascade kwargs flow through.
    "twostate_ln_logit",
}

# Which base models accept the ``informative_priors_from=`` (SVI-to-
# Laplace) cascade.  Consulted by ``api.stages.run_inference``.  These
# are the targets that have a dedicated cascade adapter in
# ``scribe.laplace.priors``.
CASCADE_FROM_SVI_SUPPORTED_BASE_MODELS: set[str] = {
    "nbln",
    "twostate_ln_rate",
    "twostate_ln_logit",
}
