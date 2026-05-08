"""
Validation sets and deprecated alias mappings for the SCRIBE simplified API.

These constants define the accepted values for model, parameterization, and
inference method selectors in :func:`scribe.api.fit`.
"""

from ..models.parameterizations import PARAMETERIZATIONS

# Valid model types accepted by ``fit(model=...)``.
VALID_MODELS = {
    "nbdm",
    "zinb",
    "nbvcp",
    "zinbvcp",
    "lnm",
    "lnmvcp",
    "pln",
    "nbln",
}

# Deprecated model name aliases mapped to their canonical replacements.
_DEPRECATED_MODEL_ALIASES = {"nbdm_lnm": "lnm"}

# Derive valid parameterizations from the single source of truth in the
# parameterization registry.
VALID_PARAMETERIZATIONS = set(PARAMETERIZATIONS.keys())

# Valid inference methods accepted by ``fit(inference_method=...)``.
VALID_INFERENCE_METHODS = {"svi", "mcmc", "vae", "laplace"}
