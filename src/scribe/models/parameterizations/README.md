# Parameterizations Module

This module implements the **Strategy Pattern** for handling different
parameterization schemes in probabilistic models. Each parameterization defines
how model parameters are sampled and derived, eliminating nested conditionals in
preset factories.

## Overview

A parameterization scheme determines:
- **Core parameters**: Which parameters are sampled directly from priors
- **Derived parameters**: Which parameters are computed deterministically from
  core parameters
- **Parameter transformations**: How model-specific parameters (like
  `p_capture`) are transformed

## Parameterization Classes

### `CanonicalParameterization` (formerly "standard")

Directly samples both the success probability `p` and dispersion `r`:

- **Core parameters**: `p`, `r`
- **Derived parameters**: `mu = r * p / (1 - p)` — declared so that axis
  membership (dataset, mixture) propagates correctly from `r`/`p` to `mu`
  via `expand_membership_from_derived`.  Without this, `get_dataset()` would
  not slice `mu` along the dataset axis, producing identical `mu_map_A` and
  `mu_map_B` in empirical DE.
- **Gene parameter**: `r` (gene-specific)

**Example**:
```python
from scribe.models.parameterizations import PARAMETERIZATIONS

param_strategy = PARAMETERIZATIONS["canonical"]
param_specs = param_strategy.build_param_specs(
    unconstrained=False,
    guide_families=GuideFamilyConfig(),
)
# Returns: [BetaSpec("p"), LogNormalSpec("r")]
derived_params = param_strategy.build_derived_params()
# Returns: [DerivedParam("mu", _compute_mu_from_r_p, ["r", "p"])]
```

### `MeanProbParameterization` (formerly "linked")

Samples `p` and mean expression `mu`, then derives `r`:

- **Core parameters**: `p`, `mu`
- **Derived parameters**: `r = mu * (1 - p) / p`
- **Gene parameter**: `mu` (gene-specific)

This links dispersion to mean expression, which can help capture correlations in
the variational posterior.

**Example**:
```python
param_strategy = PARAMETERIZATIONS["mean_prob"]
param_specs = param_strategy.build_param_specs(...)
derived_params = param_strategy.build_derived_params()
# Returns: [DerivedParam("r", lambda p, mu: mu * (1 - p) / p, ["p", "mu"])]
```

### `MeanOddsParameterization` (formerly "odds_ratio")

Samples odds ratio `phi` and mean expression `mu`, then derives both `p` and
`r`:

- **Core parameters**: `phi`, `mu`
- **Derived parameters**: 
  - `p = 1 / (1 + phi)`
  - `r = mu * phi`
- **Gene parameter**: `mu` (gene-specific)
- **Parameter transformations**: `p_capture` → `phi_capture`

This parameterization is numerically more stable than `mean_prob` when `p` is
close to 0 or 1, as it avoids division by `p` in the computation of `r`.

**Example**:
```python
param_strategy = PARAMETERIZATIONS["mean_odds"]
# Transform model-specific parameter
capture_param = param_strategy.transform_model_param("p_capture")
# Returns: "phi_capture"
```

## Hierarchical Priors (Boolean Flags)

Hierarchical behavior is controlled by **boolean flags on ModelConfig**, not by
separate parameterization types. The flags are independent and composable:

- **`hierarchical_p`** (default False): Gene-specific `p`/`phi` hierarchical
  prior. Each gene draws its own `p_g` (or `phi_g`) from a learned population
  distribution defined by a Normal hyperprior in unconstrained (logit/log)
  space. **Requires `unconstrained=True`**.
- **`hierarchical_gate`** (default False): Gene-specific gate hierarchical
  prior for zero-inflation. **ZI models only; requires `unconstrained=True`**.

The hierarchy lives **only in the model prior**; the guide treats gene-level
parameters as ordinary variational parameters. The KL divergence in the ELBO
couples them to the hyperprior, providing adaptive shrinkage.

Use any base parameterization (`canonical`, `mean_prob`, `mean_odds`) together
with these flags, e.g. `parameterization: canonical` + `hierarchical_p: true`.

### Posterior Diagnostics

After fitting with `hierarchical_p=True`, `posterior_samples["logit_p_scale"]`
(or `"log_phi_scale"` for mean_odds) provides a diagnostic of the shared-p
assumption: small values validate shared-p; large values indicate gene-specific
heterogeneity.

### Data-Informed Mean Anchoring

When `ModelConfig.expression_anchor=True`, the factory computes per-gene prior
centers from observed sample means and replaces the flat `log_mu_loc` (or
`log_mu_dataset_loc`) hyperprior with an `AnchoredNormalSpec`. This resolves
the mu-phi likelihood ridge that makes the mean_odds and mean_prob
parameterizations fragile with hierarchical phi priors. The anchor works with
all three parameterizations (canonical targets `log_r_loc`; mean_prob and
mean_odds target `log_mu_loc`). See `paper/_mean_anchoring_prior.qmd`.

## Usage in Unified Factory

The unified factory uses parameterizations to eliminate nested conditionals:

```python
from ..parameterizations import PARAMETERIZATIONS

def create_model(model_config):
    # Get parameterization strategy
    param_strategy = PARAMETERIZATIONS[model_config.parameterization]
    
    # Build parameter specs
    param_specs = param_strategy.build_param_specs(
        unconstrained, guide_families
    )
    
    # Get derived parameters
    derived_params = param_strategy.build_derived_params()
    
    # Transform model-specific parameters if needed
    capture_param_name = param_strategy.transform_model_param("p_capture")
    # For mean_odds: returns "phi_capture"
    # For others: returns "p_capture"
```

### `LogisticNormalParameterization`

Used by the `lnm` model type. Samples scalar total-count NB parameters
and delegates gene compositions to a linear-decoder VAE in ALR space:

- **Core parameters**: `r_T` (total-count dispersion), `p` (total-count
  success probability)
- **Derived parameters**: none (compositions come from the decoder)
- **Gene parameter**: `y_alr` (ALR coordinates, produced by the VAE decoder)
- **`requires_vae`**: `True` — inference always uses `VAELatentGuide`

The decoder output spec is `[("y_alr", "identity")]`, enforcing a linear
(no hidden layers) decoder so the kernel acts as the low-rank factor `W` and
the bias as the ALR mean `mu`.

### `PoissonLogNormalParameterization`

Used by both the `pln` and `nbln` model types. Has **no sampled core
parameters** — the per-cell log-rate latent is produced by the linear-decoder
VAE, not sampled directly:

- **Core parameters**: none
- **Derived parameters**: none
- **Gene parameter**: `y_log_rate` (per-cell log-rates from the VAE decoder)
- **`requires_vae`**: `True` — same VAE pattern as the LNM family
- **Decoder output spec**: `[("y_log_rate", "identity")]` — linear decoder,
  exponentiated inside the likelihood (PLN) or fed as the NB log-mean (NBLN).

NBLN reuses this parameterization unchanged and adds gene dispersion `r_g`
through the `MODEL_EXTRA_PARAMS["nbln"] = ["r"]` registration mechanism in
`scribe.models.presets.registry`. The factory's
`build_extra_param_spec("r", ...)` arm dispatches to a dedicated
`build_r_spec` builder that returns a `LogNormalSpec` (constrained) or
`PositiveNormalSpec` (unconstrained) for the gene-specific `r_g`.

## Registry

The `PARAMETERIZATIONS` dictionary maps names to parameterization instances:

```python
PARAMETERIZATIONS = {
    # Core parameterizations
    "canonical": CanonicalParameterization(),
    "mean_prob": MeanProbParameterization(),
    "mean_odds": MeanOddsParameterization(),
    "logistic_normal_canonical": LogisticNormalParameterization(...),
    "logistic_normal_mean_prob": LogisticNormalParameterization(...),
    "logistic_normal_mean_odds": LogisticNormalParameterization(...),
    "poisson_lognormal": PoissonLogNormalParameterization(),
    # Backward-compatible aliases
    "standard": CanonicalParameterization(),
    "linked": MeanProbParameterization(),
    "odds_ratio": MeanOddsParameterization(),
}
```

Both new and old names are supported for backward compatibility. Hierarchical
behavior is controlled by `ModelConfig.hierarchical_p` and
`ModelConfig.hierarchical_gate`, not by parameterization name.

## Benefits

1. **Eliminates nested conditionals**: No more `if parameterization ==
   "standard": if unconstrained: ...`
2. **Reduces duplication**: Parameterization logic is centralized
3. **Clearer semantics**: Names like "mean_prob" are more descriptive than
   "linked"
4. **Easy to extend**: Add new parameterizations by creating a new class
5. **Handles transformations**: Model-specific parameter transformations are
   encapsulated

## Extending

To add a new parameterization:

1. Create a new class inheriting from `Parameterization`
2. Implement all abstract methods:
   - `name`: Human-readable name
   - `core_parameters`: List of core parameter names
   - `gene_param_name`: Name of gene-specific parameter
   - `build_param_specs()`: Build parameter specs
   - `build_derived_params()`: Build derived parameter computations
   - `transform_model_param()`: Transform model-specific parameters (optional)
3. Add to `PARAMETERIZATIONS` registry

**Example**:
```python
class MyParameterization(Parameterization):
    @property
    def name(self) -> str:
        return "my_param"
    
    @property
    def core_parameters(self) -> List[str]:
        return ["alpha", "beta"]
    
    # ... implement other methods

PARAMETERIZATIONS["my_param"] = MyParameterization()
```

## See Also

- `scribe.models.presets`: Preset factories that use parameterizations
- `scribe.models.builders.parameter_specs`: Parameter specification classes,
  including `HierarchicalSigmoidNormalSpec` and `HierarchicalPositiveNormalSpec`
  for hierarchical gene-specific parameters
- `scribe.models.components.likelihoods`: Likelihoods that handle parameter
  transformations and gene-specific `p` broadcasting
