# Differential Expression

SCRIBE provides a fully Bayesian framework for differential expression (DE)
analysis that works in compositional space, accounts for gene-gene
correlations, and provides true posterior probabilities instead of frequentist
p-values.

## Overview

The DE framework is:

- **Compositional** -- works in CLR (Centered Log-Ratio) or ILR (Isometric
  Log-Ratio) space, ensuring reference-invariant results
- **Correlation-aware** -- leverages low-rank covariance structure to account
  for gene-gene correlations
- **Fully Bayesian** -- provides exact posterior probabilities under the
  Gaussian assumption (parametric) or via Monte Carlo counting (empirical)
- **Computationally efficient** -- all operations are \(O(kG)\) or
  \(O(k^2 G)\) where \(k \ll G\), avoiding \(O(G^3)\) dense matrix
  operations

---

## Quick Start

```python
from scribe.de import compare

# Fit models for two conditions
results_A = scribe.fit(adata_treatment, model="nbdm")
results_B = scribe.fit(adata_control, model="nbdm")

# Create comparison (empirical method, recommended)
de = compare(
    results_A, results_B,
    method="empirical",
    component_A=0, component_B=0,
)

# Gene-level analysis with practical significance threshold
results = de.gene_level(tau=jnp.log(1.1))

# Call DE genes (Bayesian decision)
is_de = de.call_genes(lfsr_threshold=0.05, prob_effect_threshold=0.95)
print(f"Found {is_de.sum()} DE genes")

# Summary table
print(de.summary(sort_by="lfsr", top_n=20))
```

---

## Three DE Methods

The `compare()` factory returns different result types depending on `method=`:

| Method | When to use | How it works |
|--------|-------------|--------------|
| **Parametric** | Gaussian assumption holds | Analytic posteriors from fitted logistic-normal models |
| **Empirical** | General use (recommended) | Monte Carlo counting on Dirichlet-sampled compositions |
| **Shrinkage** | Most genes are not DE | Empirical Bayes shrinkage on top of empirical results |

### Parametric

Requires pre-fitted logistic-normal models. All statistics are computed
analytically:

```python
# Fit logistic-normal models first
model_A = results_A.fit_logistic_normal(rank=16)
model_B = results_B.fit_logistic_normal(rank=16)

de = compare(model_A, model_B, gene_names=adata.var_names.tolist())
```

### Empirical (recommended)

Works directly with posterior samples via Monte Carlo counting -- no
distributional assumptions required:

```python
de = compare(
    results_A, results_B,
    method="empirical",
    component_A=0, component_B=0,
)
```

When results objects are passed, `compare()` automatically:

- Extracts `r` samples from `posterior_samples["r"]`
- Detects hierarchical models (gene-specific \(p\)) and extracts \(p\) samples
- Infers gene names from `results.var.index`

### Shrinkage

Improves on empirical DE by learning a genome-wide effect-size distribution
and adaptively shrinking noisy estimates toward zero:

```python
de = compare(
    results_A, results_B,
    method="shrinkage",
    component_A=0, component_B=0,
)

results = de.gene_level(tau=jnp.log(1.1))
print(f"Estimated null proportion: {de.null_proportion:.2%}")
```

!!! tip "Zero-copy upgrade"
    You can upgrade an existing empirical result to shrinkage without
    recomputing the expensive Dirichlet sampling:

    ```python
    de_emp = compare(results_A, results_B, method="empirical", ...)
    de_shrink = de_emp.shrink()
    ```

---

## Gene-Level Analysis

All DE result types share the same analysis API:

| Method | Description |
|--------|-------------|
| `de.gene_level(tau)` | Per-gene posterior summaries (lfsr and lfsr_tau) |
| `de.call_genes(tau, lfsr_threshold, prob_effect_threshold)` | Bayesian gene calling |
| `de.summary(tau, sort_by, top_n)` | Formatted results table |
| `de.to_dataframe(tau, target_pefp)` | Export to pandas DataFrame |

### Exporting results

```python
# Basic export
df = de.to_dataframe(tau=0.5)

# With automatic PEFP-controlled DE calls
df = de.to_dataframe(tau=0.5, target_pefp=0.05)
de_genes = df[df["clr_is_de"]]

# Include biological metrics (empirical/shrinkage only)
df = de.to_dataframe(metrics="all", tau=0.5)
```

---

## Bayesian Error Control

Unlike frequentist methods that use FDR, SCRIBE uses true posterior
probabilities:

| Metric | Description |
|--------|-------------|
| **lfsr** | Local false sign rate -- posterior probability of having the wrong sign |
| **lfsr_tau** | Modified lfsr incorporating practical significance threshold \(\tau\) |
| **PEFP** | Posterior expected false discovery proportion -- Bayesian analogue of FDR |

### Controlling the false discovery rate

```python
import jax.numpy as jnp

# Gene-level analysis with practical significance
results = de.gene_level(tau=jnp.log(1.1))

# Find the lfsr threshold that controls PEFP at 5%
threshold = de.find_threshold(target_pefp=0.05)
print(f"lfsr threshold for 5% PEFP: {threshold:.3f}")

# Call genes at that threshold
is_de = de.call_genes(lfsr_threshold=threshold)

# Verify the Bayesian FDR
pefp = de.compute_pefp(threshold=0.05)
print(f"Expected false discovery proportion: {pefp:.3f}")
```

---

## Biological-Level DE

While CLR-based metrics operate in the compositional simplex, biological-level
DE computes metrics directly on the denoised Negative Binomial distribution.
This is especially valuable for lowly expressed genes where compositional
artifacts dominate.

| Metric | What it captures |
|--------|------------------|
| **Biological LFC** | Mean expression shift: \(\log(\mu_A / \mu_B)\) |
| **Log-variance ratio** | Dispersion shift: \(\log(\text{var}_A / \text{var}_B)\) |
| **Gamma Jeffreys divergence** | Full distributional shift via symmetrized KL |

```python
# Biological-level DE (empirical/shrinkage only)
bio = de.biological_level(
    tau_lfc=jnp.log(1.5),
    tau_var=jnp.log(2.0),
    tau_kl=0.5,
)

bio["lfc_mean"]     # posterior mean biological LFC per gene
bio["lfc_lfsr"]     # local false sign rate for LFC
bio["lvr_mean"]     # posterior mean log-variance ratio
bio["kl_mean"]      # posterior mean Jeffreys divergence
```

!!! info "Recommended workflow"
    1. **Screen with CLR**: use CLR-based lfsr as the primary filter
    2. **Validate with biological LFC**: filter out compositional artifacts
    3. **Detect variance changes**: inspect log-variance ratio for genes with
       small LFC but high distributional shift
    4. **Flag distributional shifts**: use Jeffreys divergence as a catch-all

---

## Mixture-Weighted DE

When a cell type is modeled as a multi-component mixture (e.g., to capture
distinct cellular states), you can perform **population-level** DE that
marginalises over the mixture instead of comparing individual components.

The pipeline samples compositions from all K components and averages them on the
simplex using the posterior mixture weights, then feeds the result into the
standard CLR machinery.

```python
# Auto-extract mixing_weights from results objects
de = compare(
    results_A, results_B,
    method="empirical",
    mixture_weighted=True,
)

results = de.gene_level(tau=jnp.log(1.1))
```

For raw arrays, provide weights explicitly:

```python
de = compare(
    r_samples_A,  # (N, K, D)
    r_samples_B,  # (N, K, D)
    method="empirical",
    mixture_weighted=True,
    mixture_weights_A=weights_A,  # (N, K)
    mixture_weights_B=weights_B,  # (N, K)
)
```

| Scenario                                               | Approach                                            |
| ------------------------------------------------------ | --------------------------------------------------- |
| Compare biologically distinct states                   | `component_A=`, `component_B=`                      |
| Population-level change of a multi-component cell type | `mixture_weighted=True`                             |
| Single-component model                                 | Standard `compare()` (mixture weighting is a no-op) |

!!! note
    `mixture_weighted=True` is mutually exclusive with `component_A` /
    `component_B`. The parametric method is not supported because the CLR
    of a mixture of Dirichlets is not Gaussian.

Shrinkage works on top of the mixture-weighted empirical result:

```python
de_shrink = compare(
    results_A, results_B,
    method="shrinkage",
    mixture_weighted=True,
)
```

Biological-level metrics (LFC, LVR, Jeffreys divergence) are computed
from mixture-weighted NB parameters when `compute_biological=True`.

---

## Population Differential Expression Across Grouping Factors

When you fit a **crossed hierarchical** model (e.g. donors crossed with a
treatment — see [the fit guide](fit.md#crossed-multi-factor-designs)), the
question is usually the **population** treatment effect, *paired within donor*.
`scribe.compare_groups` computes it directly: for every donor present in both
arms it forms the within-donor CLR difference, then averages those over donors
(the **paired main effect**). The donor effect cancels inside each within-donor
difference, so the result is the average treatment effect with donor
heterogeneity differenced out — see
[Theory: population DE](../theory/differential-expression.md#part-iii-population-differential-expression-across-grouping-factors).

```python
import scribe

# results_joint was fit with hierarchy=[GroupLevel("perturbation", ...),
#                                       GroupLevel("sample")]
de = scribe.compare_groups(
    results_joint,
    "perturbation",        # the contrast (a base grouping factor)
    "control", "panobinostat",   # the two levels: delta = CLR(level_A) - CLR(level_B)
)

# Everything downstream is identical to compare():
genes = de.gene_level(tau=np.log(1.1))
df = de.to_dataframe(tau=np.log(2.0), target_pefp=0.05, metrics="all")
```

The returned object is a standard `ScribeEmpiricalDEResults`, so `gene_level`,
`to_dataframe`, lfsr/PEFP, the biological metrics, and the
`scribe.viz.plot_de_*` plots all work unchanged (including `mode="bio"`).

Useful keyword arguments:

| Argument | Default | Purpose |
|----------|---------|---------|
| `pairing_factor` | inferred | The factor held across the contrast (the donor). Inferred when exactly one non-contrast factor exists; required otherwise |
| `pair_weighting` | `"uniform"` | How to weight donors when averaging: `"uniform"`, `"min_cells"`, `"harmonic"`, `"total_cells"` |
| `incomplete_pairs` | `"drop"` | Donors missing one arm: `"drop"` (with a warning) or `"error"` |
| `min_complete_pairs` | `2` | Error if fewer complete donor pairs remain |
| `gene_mask` | `None` | Expression filter, applied **inside each pair before CLR** (see below) |
| `n_samples` | cached / 100 | Posterior draws to sample before slicing leaves — sets `N` in `delta_samples`. Raise it (e.g. `5000`) for smoother lfsr/PEFP |
| `batch_size` | `None` | Chunk the draw and each pair's composition sampling, to cap peak memory |
| `convert_to_numpy` | auto | Offload the posterior draw to host RAM (default `True` when `n_samples > 500`) so a large draw does not exhaust GPU memory |
| `reference` | `"clr"` | Log-ratio reference frame: `"clr"`, `"iqlr"` (driver-robust; resolved per pair), or a gene-name list / boolean mask (curated). See [Choosing the reference frame](#choosing-the-reference-frame) |

!!! tip "Smoother estimates without running out of memory"
    The default draw is only 100 samples. Increase it with `n_samples`, and for
    a large draw over many leaves × genes add `batch_size` (and let
    `convert_to_numpy` offload to host RAM, which is automatic above 500) so the
    GPU is not asked to hold every sample at once:

    ```python
    de = scribe.compare_groups(
        results_joint, "perturbation", "control", "panobinostat",
        n_samples=5_000, batch_size=500,   # offloads to host automatically
    )
    ```

!!! note "Sign convention"
    As elsewhere in SCRIBE, `delta = CLR(level_A) − CLR(level_B)`, so a gene
    **up in `level_B`** has a **negative** delta. With an interaction present,
    this is the *observed-donor-average* treatment effect (including the
    average interaction deviation), which is not identical to the pure
    main-effect parameter — both are legitimate; the distinction is documented
    in the theory page.

To inspect the fitted effects directly (the treatment contrast, the donor
deviations, the learned heterogeneity scale) rather than run DE, see
[`get_factor_effect`](results.md#multi-dataset-and-multi-factor-accessors).

---

## Gene Expression Filter

Low-expression genes can appear spuriously DE due to compositional artifacts.
The `gene_mask` parameter aggregates filtered genes into a single "other"
pseudo-gene before Dirichlet sampling:

```python
from scribe.de import compare, compute_expression_mask

# Build a mask from MAP mean expression
mask = compute_expression_mask(
    results_A, results_B,
    component_A=0, component_B=0,
    min_mean_expression=1.0,
)

# Pass to compare()
de = compare(
    results_A, results_B,
    method="empirical",
    component_A=0, component_B=0,
    gene_mask=mask,
)
```

A scale-free alternative to a fixed UMI floor is **compositional coverage** —
keep the smallest set of genes covering a target fraction of the expressed
mass:

```python
# Keep genes making up 95% of the composition in either condition.
de.set_composition_coverage(coverage=0.95)
```

### Interactive mask exploration

After the initial comparison, you can change the expression mask without
re-running the expensive Dirichlet sampling:

```python
# Explore a different threshold
de.set_expression_threshold(min_expression=3.0)
df2 = de.to_dataframe(tau=0.5)

# Apply a custom mask
de.set_gene_mask(my_custom_mask)

# Restore all genes
de.clear_mask()
```

!!! warning "Filtering a `compare_groups` result"
    The in-place `set_expression_threshold` / `set_composition_coverage` /
    `set_gene_mask` recompute deltas from the stored simplex samples, which
    `compare_groups` does **not** retain (averaging simplices would not equal
    the paired CLR average). On a `compare_groups` result they raise. Instead,
    **build** the mask and pass it up front so the pooling happens inside each
    donor pair, before CLR:

    ```python
    de_raw = scribe.compare_groups(results_joint, "perturbation", "control", "panobinostat")
    gene_mask = de_raw.composition_coverage_mask(0.95)   # or .expression_mask(2.0)
    de = scribe.compare_groups(
        results_joint, "perturbation", "control", "panobinostat",
        gene_mask=gene_mask,
    )
    ```

    `expression_mask(min_expression)` and `composition_coverage_mask(coverage)`
    read only the MAP mean expression, so they work without stored simplex
    samples.

### Excluding known confounders by name

When a few high-mass genes (mitochondrial, ribosomal, hemoglobin) dominate the
composition, the principled way to remove them is **by identity** — decided from
the gene name, independently of the contrast. (Thresholding on the CLR effect
size itself and re-running is *circular*: it selects genes by the very quantity
under test.) `exclude_gene_name_mask` builds a keep-mask from name prefixes
and/or regular expressions; it reads only gene names, so it works on
`compare_groups` results too. Combine it with a coverage mask via boolean `&`
and pass the result up front as `gene_mask=`:

```python
de_raw = scribe.compare_groups(results_joint, "perturbation", "control", "panobinostat")
nuisance_keep = de_raw.exclude_gene_name_mask(
    prefixes=("MT-", "RPL", "RPS", "MRPL", "MRPS"),
    patterns=(r"^MT(RNR|CO\d|ND\d|ATP\d|CYB)", r"^HB[ABDEGMQZ]\d?$"),
)
gene_mask = de_raw.composition_coverage_mask(0.90) & nuisance_keep
de = scribe.compare_groups(
    results_joint, "perturbation", "control", "panobinostat", gene_mask=gene_mask,
)
```

The "other" pseudo-gene is always pooled, matching the other mask builders.

---

## Choosing the reference frame

A compositional contrast is defined only **up to a reference**: each gene's CLR
coordinate is its log-proportion minus the geometric mean of a reference set.
The default reference is **all** genes (standard CLR). Under a broad perturbation
this all-gene reference itself drifts, and a few high-variance "driver" genes
(mitochondrial, hemoglobin) can dominate the contrast. The `reference` argument
of `compare` / `compare_datasets` / `compare_groups` lets you choose the
reference set:

| `reference` | Meaning |
|-------------|---------|
| `"clr"` (default) | Geometric mean over all kept genes (+ the pooled "other"). Bit-identical to the legacy behaviour. |
| `"iqlr"` | **Inter-quartile log-ratio** (ALDEx2-style, deterministic variant): the reference is the genes whose CLR variance lies in the inter-quartile range, excluding the high-variance drivers. Resolved per pair in `compare_groups`. |
| gene-name list / boolean mask | A **curated** reference (e.g. housekeeping genes). Names must be kept, non-"other" genes; a boolean mask must be full-gene or aggregated length. |

```python
de = scribe.compare_groups(
    results_joint, "perturbation", "control", "panobinostat",
    gene_mask=gene_mask, reference="iqlr",
)
```

On a stored leaf-vs-leaf `compare` result you can switch the frame in place and
inspect which genes anchor the IQLR reference:

```python
de.set_reference("iqlr")          # recompute deltas under a new reference
ref_genes = de.iqlr_reference_mask()  # boolean over the kept genes
```

!!! note "What a reference change can and cannot do"
    The reference enters every gene's contrast as a **single shared shift**, so
    changing it slides all contrasts by a common amount — it cannot re-rank
    genes or shrink a gene whose own log-ratio change is large. IQLR therefore
    helps when a drifting reference biases signs *uniformly* (few features, a
    dominant driver); for a broad, high-gene-count perturbation it is close to a
    no-op. Note also that only the all-gene CLR puts the *full* coordinate
    vector in the sum-to-zero subspace, so pathway / balance tests
    (`test_gene_set`, `test_pathway_perturbation`) require `reference="clr"` and
    raise on a subset reference. `set_reference` needs stored simplex samples,
    so it works on `compare` results but raises on `compare_groups` — pass
    `reference=` up front there.

---

## Pathway Analysis

The empirical DE path supports pathway-level analysis via ILR balances:

### Single pathway test

```python
import jax.numpy as jnp

# Test whether a pathway shifts as a whole
pathway = jnp.array([10, 25, 42, 101, 200])
result = de.test_gene_set(pathway, tau=jnp.log(1.1))
print(f"Balance: {result['balance_mean']:.3f} +/- {result['balance_sd']:.3f}")
print(f"lfsr: {result['lfsr']:.4f}")
```

### Within-pathway perturbation test

Detects coordinated rearrangement within a pathway even when the average
balance is near zero:

```python
perturb = de.test_pathway_perturbation(pathway, n_permutations=999)
print(f"Perturbation T: {perturb['t_obs']:.4f}, p={perturb['p_value']:.4f}")
```

### Batch testing with PEFP control

```python
gene_sets = [
    jnp.array([10, 25, 42, 101, 200]),
    jnp.array([5, 8, 15, 33]),
    jnp.array([60, 70, 80, 90, 100]),
]
batch = de.test_multiple_gene_sets(gene_sets, target_pefp=0.05)
```

---

## Multi-Dataset Comparisons

For multi-dataset models, `compare_datasets()` is a convenience wrapper that
slices per-dataset views and preserves within-posterior correlation:

```python
from scribe.de import compare_datasets

de = compare_datasets(results, dataset_A=0, dataset_B=1)
de = compare_datasets(results, 0, 1, component=0, method="shrinkage")
```

### Label-based component matching

When mixture models are fit with `annotation_key`, you can look up components
by label (e.g., cell type name) instead of tracking indices manually:

```python
from scribe.de import compare, match_components_by_label, get_shared_labels

# Find component indices for "Fibroblast" in both results
idx_A, idx_B = match_components_by_label(results_A, results_B, "Fibroblast")

de = compare(
    results_A, results_B,
    method="empirical",
    component_A=idx_A, component_B=idx_B,
)

# Discover labels shared across results
labels = get_shared_labels(results_A, results_B)
```

---

## Class Hierarchy

```
ScribeDEResults (base)
├── ScribeParametricDEResults   -- analytic Gaussian
├── ScribeEmpiricalDEResults    -- Monte Carlo counting
└── ScribeShrinkageDEResults    -- empirical Bayes shrinkage (extends Empirical)
```

The `compare()` factory returns the appropriate subclass based on `method=`.
All shared methods (`call_genes`, `compute_pefp`, `find_threshold`, `summary`)
work identically on all three.

For the full API, see the [API Reference](../reference/scribe/de/).
