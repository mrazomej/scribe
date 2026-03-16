# Future Refactor: Dimension-Aware Broadcasting via Layout Dict

**Status**: Planned (not urgent — current heuristic fixes cover active configurations)
**Trigger**: Revisit when the next broadcasting bug surfaces or when touching the model builder pipeline

## Problem

Parameters in scribe can independently have or lack three semantic axes:

| Axis  | Source                          | Example dim |
|-------|---------------------------------|-------------|
| **K** | `is_mixture` (in mixture_params)| n_components|
| **D** | `is_dataset` (dataset hierarchy)| n_datasets  |
| **G** | `is_gene_specific`              | n_genes     |

The canonical ordering is `(K, D, G)`, but any subset may be present. Inside the
cell plate, **D** is replaced by **batch** after `index_dataset_params`.

Once `sample_hierarchical` returns a bare `jnp.ndarray`, dimension semantics are
**erased**. Every downstream function that combines two parameters must rediscover
what each axis means by inspecting `.ndim` and `.shape` — leading to heuristics
that break when two different semantic layouts produce the same numeric shape
(e.g., `(K, G)` vs `(batch, G)` when K happens to equal batch).

### Bugs from this pattern (as of March 2026)

1. **Derived params**: `r = mu * phi` fails when `mu=(K,D,G)` and `phi=(K,G)` —
   JAX rank-promotes `(K,G)` to `(1,K,G)` which conflicts with `(K,D,G)`.
   Fixed with `_align_gene_params()` in `parameterizations/__init__.py`.

2. **Likelihood broadcasting**: `broadcast_param_for_mixture` couldn't distinguish
   `(batch, G)` from `(K, G)` when `r` is `(batch, K, G)`.
   Fixed with `shape[0] == r.shape[0]` heuristic in `likelihoods/base.py`.

3. **NCP dataset specs**: 4 subclasses (Horseshoe Exp/Sigmoid, NEG Exp/Sigmoid)
   did `loc + eff_scale * z` without the singleton-D insertion the Gaussian parent
   class had. `loc=(K,G)` + `z=(K,D,G)` fails.
   Fixed by adding `expand_dims(loc, axis=1)` when `is_mixture and is_dataset`.

4. **Gate broadcasting in VCP**: gate `(batch, G)` not expanded for mixture dim
   `(batch, K, G)` in `ZeroInflatedDistribution`.

5. **MAP canonical param reconstruction**: `_compute_canonical_parameters` in
   `svi/_parameter_extraction.py` does `r = mu * phi` on MAP estimates without
   aligning `phi=(K,G)` vs `mu=(K,D,G)`. This is a *separate code path* from
   the model builder — the layout dict as designed wouldn't cover it directly.

All five share the same root cause: 
dimension semantics lost → heuristic guess → edge case.

## Proposed Solution: `param_layouts` Dict

Thread a `dict[str, tuple[str, ...]]` alongside `param_values` that records the
axis names for each parameter:

```python
param_layouts = {
    "mu":   ("K", "D", "G"),
    "phi":  ("K", "G"),
    "gate": ("D", "G"),
    "mixing_weights": ("K",),
    "capture": (),
}
```

### Where it's populated

In `model_builder.model()`, right after sampling each spec. The spec already has
`is_mixture`, `is_dataset`, `is_gene_specific` — the layout is deterministic:

```python
layout = []
if spec.is_mixture:
    layout.append("K")
if spec.is_dataset:
    layout.append("D")
if spec.is_gene_specific:
    layout.append("G")
param_layouts[spec.name] = tuple(layout)
```

### Where it's consumed

One central function replaces all current broadcasting helpers:

```python
def align_params(
    a: jnp.ndarray, b: jnp.ndarray,
    layout_a: tuple[str, ...], layout_b: tuple[str, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Insert singleton dims so a and b broadcast by axis name."""
    all_axes = _merge_ordered(layout_a, layout_b)  # e.g., ("K", "D", "G")
    a = _expand_to(a, layout_a, all_axes)
    b = _expand_to(b, layout_b, all_axes)
    return a, b
```

Call sites that would use this:

| Location | Current fix | With layout dict |
|----------|-------------|------------------|
| `_compute_r_from_mu_phi` | `_align_gene_params` | `align_params(phi, mu, layouts["phi"], layouts["mu"])` |
| `_compute_r_from_mu_p` | `_align_gene_params` | same pattern |
| `broadcast_param_for_mixture` | `shape[0] == r.shape[0]` heuristic | `align_params(p, r, layouts["p"], layouts["r"])` |
| NCP `sample_hierarchical` | manual `expand_dims(loc, 1)` | `align_params(loc, z, loc_layout, target_layout)` |
| `index_dataset_params` | transform layouts: replace `"D"` → `"batch"` | same, plus update `param_layouts` |

### What NOT to do

- **Don't wrap arrays in a TaggedArray class.** NumPyro/JAX expect raw arrays
  everywhere. A wrapper would require constant wrapping/unwrapping and would break
  at every library boundary.
- **Don't try to centralize ALL shape logic.** Some shapes (e.g., cell-specific
  VAE outputs) are handled by completely separate code paths. Focus on the
  `param_values` dict flowing through `model_builder → derived_params → likelihood`.

## Migration Strategy

1. Add `param_layouts` population in `model_builder.model()` (one loop, ~5 lines)
2. Write `align_params` + `_expand_to` + `_merge_ordered` (~30 lines)
3. Thread `param_layouts` into derived param computation (update `DerivedParam`)
4. Thread `param_layouts` into the likelihood via `model_args`
5. Replace heuristic functions one at a time, keeping them as fallbacks
6. Update `index_dataset_params` to transform the layout dict

Each step is independently testable and backward-compatible.
