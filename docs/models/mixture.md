# Mixture Models

This page explains SCRIBE's mixture model extensions for handling heterogeneous
cell populations. Each base model ([NBDM](nbdm.md), [ZINB](zinb.md),
[NBVCP](nbvcp.md), [ZINBVCP](zinbvcp.md)) can be extended to a mixture model
by introducing multiple components and component-specific parameters.

## General Structure

All mixture models in SCRIBE share a common hierarchical structure:

1. **Global Parameters** (shared across all components):
    - Base success probability
      \(p \sim \text{Beta}(\alpha_p, \beta_p)\)
    - Mixing weights
      \(\pi \sim \text{Dirichlet}(\alpha_{\text{mixing}})\)

2. **Component-Specific Parameters**:
    - Gene dispersion parameters
      \(r_{k,g} \sim \text{Gamma}(\alpha_r, \beta_r)\)
    - One per gene \(g\) per component \(k\)
    - Additional parameters depending on base model

3. **Cell-Specific Parameters** (when applicable):
    - Capture probabilities \(\nu^{(c)}\) (for [NBVCP](nbvcp.md) and
      [ZINBVCP](zinbvcp.md) variants)
    - Independent of components

4. **Gene-Specific Parameters** (when applicable):
    - Dropout probabilities (for [ZINB](zinb.md) and [ZINBVCP](zinbvcp.md)
      variants)
    - Component-specific versions in the mixture setting

## Parameter Dependencies by Model

### NBDM Mixture

- **Component-dependent**: Gene dispersion parameters \(r_{k,g}\)
- **Component-independent**: Base success probability \(p\)
- No cell-specific parameters

### ZINB Mixture

- **Component-dependent**: Gene dispersion parameters \(r_{k,g}\), dropout
  probabilities \(\pi_{k,g}\)
- **Component-independent**: Base success probability \(p\)
- No cell-specific parameters

### NBVCP Mixture

- **Component-dependent**: Gene dispersion parameters \(r_{k,g}\)
- **Component-independent**: Base success probability \(p\), cell capture
  probabilities \(\nu^{(c)}\)
- **Cell-specific**: Capture probabilities

### ZINBVCP Mixture

- **Component-dependent**: Gene dispersion parameters \(r_{k,g}\), dropout
  probabilities \(\pi_{k,g}\)
- **Component-independent**: Base success probability \(p\), cell capture
  probabilities \(\nu^{(c)}\)
- **Cell-specific**: Capture probabilities

## Learning Process

For all mixture models:

1. **Component Assignment Phase**:
    - Each cell's data influences the posterior over component assignments
    - Mixing weights are learned globally
    - Component-specific parameters adapt to their assigned cells

2. **Parameter Updates**:
    - Global parameters: Updated using data from all cells
    - Component parameters: Updated primarily using data from cells assigned to
      that component
    - Cell-specific parameters: Updated using that cell's data across all
      components

## Usage Guidelines

When to use mixture models:

1. Clear biological heterogeneity (multiple cell types)
2. Multimodal expression patterns
3. Complex technical variation that varies by cell type

Model selection considerations:

- **NBDM Mixture**: Baseline mixture model, good for initial exploration
- **ZINB Mixture**: When dropout patterns vary by cell type
- **NBVCP Mixture**: When capture efficiency varies significantly
- **ZINBVCP Mixture**: Most complex, but handles both dropout and capture
  variation

## Implementation Details

All mixture models use:

1. Shared parameters across cells within each component
2. Soft assignments of cells to components
3. Variational inference for parameter estimation
4. Mini-batch processing for scalability

## Inference and Results

The mixture model variants return specialized results objects that provide:

1. Component-specific parameter estimates
2. Cell assignment probabilities
3. Model-specific normalizations
4. Uncertainty quantification for all parameters

## Key Differences from Base Models

### Parameter Interpretation

- Parameters now represent component-specific patterns
- Cell assignments provide clustering information
- Mixing weights quantify population proportions

### Computational Considerations

- Higher computational cost
- More parameters to estimate
- Requires more data for reliable inference

### Biological Interpretation

- Captures subpopulation structure
- Allows different technical characteristics by component
- Provides a natural clustering framework

## See Also

- [NBDM model](nbdm.md) — the foundational model
- [Results class](../guide/results.md) — working with mixture model results
- [Model selection guide](index.md#model-selection-guide)
