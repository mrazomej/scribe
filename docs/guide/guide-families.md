# Variational Guide Families

In variational inference, the **guide** (also called the variational
distribution) is the family of distributions used to approximate the true
posterior. The choice of guide family controls the trade-off between
computational cost, posterior accuracy, and the ability to capture correlations
between parameters.

SCRIBE supports five guide families, configurable **per parameter** ---meaning
different parameters in the same model can use different guide families. For
example, gene-specific dispersion might use a low-rank guide while a scalar
success probability uses mean-field.

---

## At a glance

| Guide Family | Correlations | Memory | Speed | Best For |
|-------------|-------------|--------|-------|----------|
| [Mean-Field](#mean-field) | None | \(O(G)\) | Fastest | Default, most analyses |
| [Low-Rank](#low-rank) | Within a parameter group | \(O(Gk)\) | Fast | Gene-gene correlations |
| [Joint Low-Rank](#joint-low-rank) | Across parameter groups | \(O(Gk)\) | Moderate | Cross-parameter correlations (e.g., \(\mu\) and \(p\)) |
| [Amortized](#amortized) | Data-driven | Network size | Moderate | Cell-specific parameters (capture probability) |
| [VAE Latent](#vae-latent) | Learned latent space | Network size | Slowest | Representation learning, embeddings |

---

## Mean-Field

The simplest and fastest guide family. Each parameter has an independent
variational distribution---no correlations are captured between parameters:

\[
q(\theta_1, \theta_2, \ldots) = q(\theta_1)\,q(\theta_2)\,\cdots
\]

This is the **default** for all parameters when no `guide_rank` is specified.

**Advantages:**

- Fast convergence, low memory
- Works well when parameters are approximately independent
- Good baseline for most analyses

**Limitations:**

- Ignores correlations between genes or parameters
- Can underestimate posterior uncertainty

```python
# Mean-field is the default --- no special arguments needed
results = scribe.fit(adata, model="nbdm")
```

!!! tip "When mean-field is sufficient"
    For many scRNA-seq analyses, mean-field provides excellent results.
    Upgrade to low-rank only when downstream tasks (DE, denoising)
    benefit from capturing gene correlations.

---

## Low-Rank

Captures correlations within a parameter group (e.g., between genes for the
dispersion parameter \(r_g\)) using a low-rank multivariate normal
approximation:

\[
\underline{\underline{\Sigma}} = \underline{\underline{W}}\,\underline{\underline{W}}^\top + \text{diag}(\underline{d}),
\]

where \(\underline{\underline{W}}\) is \((G, k)\) and \(\underline{d}\) is the
diagonal. The rank \(k\) controls how many correlation modes are captured, with
memory scaling as \(O(Gk)\) instead of \(O(G^2)\) for a full covariance.

**Advantages:**

- Captures the top-\(k\) correlations between genes
- Memory-efficient compared to full covariance
- Important for accurate DE and denoising (cross-gene uncertainty)

**Limitations:**

- More parameters to optimize than mean-field
- May be slower to converge

```python
# Low-rank guide with rank 8
results = scribe.fit(adata, model="nbdm", guide_rank=8)
```

| `guide_rank` | Use case |
|:---:|----------|
| 5--10 | Standard analysis, moderate gene correlations |
| 10--20 | DE analysis where cross-gene uncertainty matters |
| 20--50 | Large datasets with complex correlation structures |

---

## Joint Low-Rank

Extends the low-rank guide to capture correlations **across** parameter groups.
For example, the gene-specific mean \(\mu_g\) and the success probability may be
correlated in the posterior---a joint guide captures this structure.

Internally, the joint guide uses a chain-rule decomposition via the Woodbury
identity:

\[
q(\theta_1, \theta_2) = q(\theta_1)\,q(\theta_2 \mid \theta_1),
\]

where both the marginal and the conditional are low-rank MVN of the same rank.
This extends naturally to three or more parameter groups (e.g., \(\mu\), \(p\),
and gate in a ZINB model).

**Advantages:**

- Captures cross-parameter correlations (e.g., \(\mu_g\) and \(\phi\))
- Supports heterogeneous dimensions (scalar + gene-specific in one group)
- Each conditional is itself a low-rank MVN (efficient computation)

**Limitations:**

- At rank \(k\), within-group expressivity is reduced vs. separate
  rank-\(k\) guides
- More complex optimization landscape

```python
# Joint low-rank for mu and phi
results = scribe.fit(
    adata,
    model="nbdm",
    parameterization="mean_odds",  # alias: "odds_ratio"
    unconstrained=True,
    guide_rank=10,
    joint_params=["mu", "phi"],
)
```

### Dense vs. structured params

For models with many parameter groups, you can designate which parameters
get full cross-gene low-rank coupling (`dense_params`) while others only
couple locally:

```python
# mu gets cross-gene correlations; phi and gate only couple to mu per gene
results = scribe.fit(
    adata,
    model="zinb",
    unconstrained=True,
    guide_rank=10,
    joint_params=["mu", "phi", "gate"],
    dense_params=["mu"],
)
```

---

## Amortized

Instead of learning separate variational parameters for each data point, an
amortized guide uses a neural network to predict variational parameters from
data features (sufficient statistics like total UMI count). This is particularly
useful for cell-specific parameters where the number of variational parameters
would otherwise scale with the number of cells.

**Advantages:**

- Scales to arbitrarily many cells without per-cell parameters
- Shares statistical strength across similar cells
- Fewer total variational parameters

**Limitations:**

- Requires choosing a network architecture
- May not be as flexible as per-cell optimization
- Training can be more sensitive to hyperparameters

The primary use case in SCRIBE is **amortized capture probability** for
VCP models:

```python
# Amortized inference for capture probability
results = scribe.fit(
    adata,
    model="nbvcp",
    amortize_capture=True,
    capture_hidden_dims=[128, 64],
    capture_activation="leaky_relu",
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `amortize_capture` | `False` | Enable amortized capture inference |
| `capture_hidden_dims` | `[64, 32]` | MLP hidden layer sizes |
| `capture_activation` | `"leaky_relu"` | Activation function |
| `capture_output_transform` | `"softplus"` | Output transform for positive params |

---

## VAE Latent

The VAE guide uses an encoder-decoder neural network architecture. The encoder
maps each cell's counts to a low-dimensional latent representation, and the
decoder maps back to model parameters. This provides both a variational
approximation and a learned cell embedding.

**Advantages:**

- Produces low-dimensional cell embeddings for visualization/clustering
- Captures complex nonlinear relationships
- Can be enhanced with normalizing flow priors for richer latent
  distributions

**Limitations:**

- Most computationally expensive guide family
- Requires tuning network architecture
- Posterior quality depends on encoder/decoder capacity

```python
# Standard VAE
results = scribe.fit(
    adata,
    model="nbdm",
    inference_method="vae",
    vae_latent_dim=10,
    n_steps=100_000,
    batch_size=256,
)

# Cell embeddings
embeddings = results.get_latent_embeddings(data=adata.X, n_samples=100)
```

### Normalizing flow priors

For more expressive latent distributions, attach a normalizing flow:

```python
results = scribe.fit(
    adata,
    model="nbdm",
    inference_method="vae",
    vae_latent_dim=10,
    vae_flow_type="spline_coupling",
    vae_flow_num_layers=4,
    vae_flow_hidden_dims=[64, 64],
)
```

| Flow type | Description |
|-----------|-------------|
| `"affine_coupling"` | Fast baseline |
| `"spline_coupling"` | Expressive, recommended for production |
| `"maf"` | Fast density evaluation |
| `"iaf"` | Fast sampling |

---

## How to choose

```mermaid
graph TD
    Start["Start"] --> Q1{"Need cell<br/>embeddings?"}
    Q1 -->|Yes| VAE["VAE Latent"]
    Q1 -->|No| Q2{"Cell-specific<br/>parameters?"}
    Q2 -->|Yes| Amort["Amortized"]
    Q2 -->|No| Q3{"Need cross-parameter<br/>correlations?"}
    Q3 -->|Yes| Joint["Joint Low-Rank"]
    Q3 -->|No| Q4{"Need gene-gene<br/>correlations?"}
    Q4 -->|Yes| LR["Low-Rank"]
    Q4 -->|No| MF["Mean-Field"]
```

**Rules of thumb:**

1. **Start with mean-field.** It is fast and works well for most
   analyses.
2. **Add low-rank when doing DE or denoising**, where cross-gene
   uncertainty propagation matters. Rank 8--15 is usually sufficient.
3. **Use joint low-rank for unconstrained models** with hierarchical
   priors, where \(\mu\) and \(\phi\) (or \(p\)) are expected to
   correlate.
4. **Use amortized for VCP models** with many cells, to avoid a
   per-cell variational parameter.
5. **Use VAE when you also need cell embeddings** for visualization or
   clustering.

---

## Combining guide families in one model

SCRIBE's guide families are per-parameter, so a single model can use
multiple families:

```python
# In a custom model, different parameters can have different guides:
# - r_g: low-rank (captures gene correlations)
# - p: mean-field (scalar, no correlations needed)
# - p_capture: amortized (cell-specific, scales with cells)
```

For the built-in models via `scribe.fit()`, the `guide_rank` and
`joint_params` arguments configure the gene-specific parameters while
`amortize_capture` configures cell-specific parameters.

---

For more on how guide families fit into the broader inference workflow,
see the [Inference Methods](inference.md) page.
