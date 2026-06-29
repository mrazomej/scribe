# Defining Priors

SCRIBE takes **all** prior information through a single keyword: `priors`. One
dict expresses three different kinds of prior, and which kind you get is decided
by the *shape* of the value you provide --- there are no separate
`*_prior` / `*_dataset_prior` keyword arguments to remember.

```python
results = scribe.fit(
    adata,
    parameterization="mean_disp",
    priors={
        "mean_expression": (0.0, 1.0),                       # base hyperparameters
        "probability":     "horseshoe",                      # gene-level shrinkage
        "dispersion":      {"perturbation": "gaussian"},     # condition hierarchy
    },
)
```

!!! tip "One key, three roles"
    The value you attach to a parameter name selects the role:

    | Value shape | Role | Example |
    |---|---|---|
    | a **tuple** `(loc, scale)` | **base** prior hyperparameters | `{"dispersion": (0.0, 1.0)}` |
    | a **family string** (or `{"type": ...}`) | **gene-level** adaptive shrinkage | `{"probability": "horseshoe"}` |
    | a **`{level: family}` dict** | **dataset / condition hierarchy** | `{"mean_expression": {"sample": "horseshoe"}}` |

---

## Parameter names

Prior keys are **canonical, parameterization-independent names** that map
one-to-one to a model parameter. You never need to know the internal Greek
symbol.

| Name | Symbol | Meaning |
|---|---|---|
| `mean_expression` | \(\mu\) | per-gene mean expression |
| `dispersion` | \(r\) | NB size / dispersion (lower = burstier) |
| `probability` | \(p\) | NB success probability |
| `odds_ratio` | \(\phi\) | NB odds ratio (`mean_odds`) |
| `zero_inflation` | gate | dropout / zero-inflation gate (ZINB) |
| `overdispersion` | \(\kappa\) | BNB concentration (`overdispersion="bnb"`) |
| `regime` | --- | two-state bursting-regime hierarchy (`twostate`) |
| `capture_probability` | \(p_{\text{cap}}\) | per-cell capture probability (VCP) |
| `capture_efficiency` | \(\eta\) | capture-efficiency anchor `(log_M0, sigma_M)` (VCP/NBLN) |
| `capture_scaling` | \(\mu_\eta\) | per-dataset capture-scaling hierarchy (VCP) |
| `loadings` | \(W\) | low-rank loadings (PLN/NBLN) |

A prior key is only valid if the chosen parameterization actually **samples**
that parameter. For example `dispersion` is a free coordinate under
`mean_disp` and `canonical`, but it is *derived* under `mean_prob` / `mean_odds`
--- asking for a prior on it there raises a clear error.

---

## 1. Base (non-hierarchical) priors

Pass a **tuple** to set the hyperparameters of a parameter's prior distribution.
These are the location/scale of the (transformed) Normal in unconstrained mode,
or the natural parameters of the constrained distribution otherwise.

```python
scribe.fit(
    adata,
    parameterization="mean_disp",
    priors={
        "mean_expression": (0.0, 1.0),   # LogNormal(0, 1) on mu
        "dispersion":      (1.0, 1.0),   # prior on r
    },
)
```

Capture and low-rank parameters take the same tuple form
(`{"capture_efficiency": (10.0, 1e5)}`), and the low-rank loadings accept a
strategy spec (`{"loadings": {"type": "horseshoe_columnwise", ...}}`). See
[Anchoring Priors](../theory/anchoring-priors.md) and
[Loadings Shrinkage](../theory/loadings-shrinkage.md).

---

## 2. Gene-level hierarchical priors

Pass a **family string** to put an *adaptive shrinkage* prior on a parameter
across genes (and, for the mean, across mixture components). This is the
classic SCRIBE hierarchical prior --- the shrinkage scale is learned from the
data.

```python
scribe.fit(
    adata,
    parameterization="mean_prob",
    unconstrained=True,                  # hierarchical priors require this
    priors={
        "probability":     "horseshoe",  # sparse gene-specific p
        "zero_inflation":  "gaussian",   # pooled gate (ZINB)
    },
)
```

Families: `"gaussian"` (pooled, learned scale), `"horseshoe"` (sparse,
heavy-tailed), `"neg"` (normal--exponential--gamma). To override the family's
hyperparameters, pass a **family spec** instead of a bare string:

```python
priors={"probability": {"type": "horseshoe", "tau0": 1.0, "slab_df": 4}}
```

!!! note "Unconstrained mode"
    Hierarchical priors decompose parameters in unconstrained space, so they
    require `unconstrained=True`. SCRIBE raises a clear error otherwise.

**Theory:** [Hierarchical Priors](../theory/hierarchical-priors.md).

---

## 3. Dataset / condition hierarchical priors

Pass a **`{level: family}` dict** to share a parameter *partially* across the
levels of a grouping factor (donors, conditions, batches). Declare the grouping
with `dataset_key` (or `hierarchy=`); the dict keys are the grouping factor
names.

```python
scribe.fit(
    adata,
    parameterization="mean_disp",
    unconstrained=True,
    dataset_key=["perturbation", "sample"],     # donor x condition
    priors={
        "mean_expression": {
            "perturbation": "gaussian",          # fixed condition contrast
            "sample":       "horseshoe",         # shrink across donors
        },
    },
)
```

Each factor named in the dict gets its own zero-mean effect with a learned (or
fixed) scale; factors you omit carry no effect. A single declared factor
(`dataset_key="sample"`) collapses to the familiar single-axis dataset
hierarchy. Add a `"base"` key to set the gene-level prior for the same
parameter at the same time:

```python
priors={"mean_expression": {"base": "gaussian", "sample": "horseshoe"}}
```

!!! note "Log link is enforced"
    An additive hierarchy on the mean (or dispersion) is log-additive: SCRIBE
    forces the `exp` link on the affected target so the per-factor effects are
    interpretable **log-fold-changes** (`bio_lfc` equals the effect contrast).

**Theory:** [Hierarchical Priors --- crossed and nested designs](../theory/hierarchical-priors.md#crossed-and-nested-designs-multiple-grouping-factors).

---

## 4. Condition-specific dispersion

Single-cell exists to measure more than the mean. With `mean_disp`, the
Fisher-orthogonal `(mu, r)` parameterization lets you put a hierarchy on the
**dispersion** `r` *independently* of the mean --- so you can ask whether a
mechanism changes the **noise** at fixed expression.

```python
scribe.fit(
    adata,
    parameterization="mean_disp",
    unconstrained=True,
    variable_capture=True,                  # joint inference resolves capture/mean
    dataset_key=["perturbation", "sample"],
    priors={
        "mean_expression": {"perturbation": "gaussian", "sample": "horseshoe"},
        "dispersion":      {"perturbation": "gaussian"},   # r varies by condition
    },
)
```

Here `r` carries the condition (`perturbation`) effect **only** --- one `r` per
condition, shared across donors --- while `mu` keeps its full donor x condition
hierarchy. Downstream, the per-condition effect is exactly \(\Delta \log r\),
so a gene with a small mean shift but a large dispersion shift becomes
detectable.

The `dispersion` hierarchy reuses the **same** additive machinery as the mean,
so you can give `r` the *full* crossed decomposition too --- just list the same
factors. The example below mirrors the mean's donor x condition hierarchy on
`r` as well:

```python
priors={
    "mean_expression": {"perturbation": "gaussian", "sample": "horseshoe"},
    "dispersion":      {"perturbation": "gaussian", "sample": "horseshoe"},
}
```

!!! warning "Scope"
    A dispersion hierarchy requires `mean_disp` (the only parameterization that
    samples `r` as a free orthogonal coordinate), one or more **base** grouping
    factors (interaction factors are not yet supported on this path), and a
    `"gaussian"` or `"horseshoe"` family (`"neg"` is not yet supported). It must
    be a **joint** fit: with `variable_capture=True`, capture efficiency is
    degenerate with the mean, so the conditions have to share that structure in
    one inference. `get_dataset(i)` correctly re-samples a single leaf of an
    additive multi-factor fit (it restricts the grouping to that leaf), so
    per-dataset posterior sampling and differential expression both work.

**Theory:** [Differential Expression](differential-expression.md) and
[Hierarchical Priors](../theory/hierarchical-priors.md).

---

## 5. Two-state regime hierarchy

Two-state (`twostate` / `twostatevcp`) models carry a *bursting regime*
coordinate whose concrete name depends on the parameterization (`k_off`,
`switching_ratio`, `concentration`, `inv_concentration`). The single canonical
key `regime` lets you put a dataset/condition hierarchy on it without knowing
which coordinate the active parameterization uses:

```python
scribe.fit(
    adata,
    model="twostate",
    parameterization="two_state_natural",
    unconstrained=True,
    dataset_key="condition",
    priors={
        "mean_expression": {"condition": "horseshoe"},
        "regime":          {"condition": "horseshoe"},   # regime varies by condition
    },
)
```

The `regime` key is **hierarchy-only**: it has no gene-level (bare-string) or
base (tuple) form. To pin the regime hierarchy to a *specific* coordinate (the
regime coordinate vs. the overdispersion coordinate), pass the structural
`regime_dataset_target` kwarg --- it is the only escape hatch and defaults to
the parameterization's regime coordinate.

---

## Capture scaling (per-dataset \(\mu_\eta\))

For variable-capture (VCP) models with the biology-informed capture prior, the
per-dataset capture-scaling parameter \(\mu_\eta\) can itself carry a shrinkage
hierarchy across datasets, shrunk toward a shared population mean. Declare it on
the `capture_scaling` key (an alias of `mu_eta`); its **value shape** picks the
behavior:

| Value shape | Meaning |
|---|---|
| `(center, sigma_mu)` tuple | fixed population hyperparameters, **no** shrinkage |
| family string `"gaussian"`/`"horseshoe"`/`"neg"` | shrinkage with default hyperparameters |
| `{"type": family, "center": ..., "sigma_mu": ...}` | shrinkage **and** custom hyperparameters |

```python
scribe.fit(
    adata,
    variable_capture=True,
    unconstrained=True,
    priors={
        "organism": "human",            # resolves the per-cell capture anchor
        "capture_scaling": "horseshoe",  # per-dataset mu_eta hierarchy
    },
)
```

A non-`"none"` family additionally needs a capture anchor — `capture_efficiency`
/ `eta_capture` `(log_M0, sigma_M)` or `organism`. (This replaces the former
`capture_scaling_prior=` kwarg, which no longer exists.)

---

## Putting it together

```python
results = scribe.fit(
    adata,
    parameterization="mean_disp",
    unconstrained=True,
    variable_capture=True,
    dataset_key=["perturbation", "sample"],
    priors={
        "mean_expression": {"base": "gaussian",
                             "perturbation": "gaussian",
                             "sample": "horseshoe"},
        "dispersion":      {"perturbation": "gaussian"},
        "capture_efficiency": (10.0, 1e5),
    },
)
```

This single `priors` dict simultaneously sets a gene-level prior on the mean, a
donor x condition hierarchy on the mean, a condition hierarchy on the
dispersion, and a base prior on capture efficiency --- with the role of each
entry determined entirely by its value shape.

---

## See also

- [The `scribe.fit()` Interface](fit.md)
- [Parameter Reference](parameters.md) --- every parameter name and symbol
- [Theory: Hierarchical Priors](../theory/hierarchical-priors.md)
