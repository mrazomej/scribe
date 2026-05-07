# Logistic-Normal Multinomial Model

The Logistic-Normal Multinomial (LNM) model replaces the Dirichlet prior on gene
compositions with a logistic-normal distribution, introducing an explicit,
parameterizable covariance structure in log-ratio space. This enables the model
to represent arbitrary cross-gene correlations---a capability the
[Dirichlet-Multinomial](dirichlet-multinomial.md) fundamentally lacks. Beyond
the statistical motivation, the logistic-normal has a direct
[biophysical justification](grn-biophysics.md): it is the steady-state
distribution of interacting genes under the linear noise approximation.

---

## Motivation: limitations of the Dirichlet

The Dirichlet distribution \(\text{Dir}(\underline{r})\) on the simplex has a
structural limitation for correlation modeling:

1. **All pairwise covariances are negative.** The simplex constraint forces
   \(\text{Cov}(\rho_g, \rho_{g'}) = -r_g r_{g'} / [r_T^2(r_T + 1)] < 0\).
   There is no way to make two components positively correlated.

2. **The correlation matrix is fully determined by \(\underline{r}\).** Once
   the mean is fixed, zero free parameters remain for the correlation structure.

3. **Low-rank guides capture only likelihood-induced coupling.** Even with a
   low-rank variational guide that learns posterior correlations, the
   generative model cannot express true cross-gene dependencies.

The logistic-normal fills this gap by placing a Gaussian in log-ratio space with
a free covariance matrix.

---

## The logistic-normal distribution

A random vector \(\underline{\rho}\) on the \(G\)-simplex follows a
logistic-normal distribution if its **additive log-ratio** (ALR) transform is
multivariate normal:

\[
\underline{z} \sim \mathcal{N}(\underline{\mu},\,
\underline{\underline{\Sigma}}),
\quad \underline{z} \in \mathbb{R}^{G-1},
\]

\[
\underline{\rho} = \text{ALR}^{-1}(\underline{z}) =
\text{softmax}(\underline{z}, 0),
\quad \underline{\rho} \in \Delta^G.
\]

The ALR maps the simplex to \(\mathbb{R}^{G-1}\) via
\(z_g = \log(\rho_g / \rho_G)\), removing the one redundant degree of freedom
inherent to compositional data.

### Parameterization choices

Three equivalent parameterizations exist:

| Parameterization   | Dimensions | Covariance rank          | Constraints                        |
| ------------------ | ---------- | ------------------------ | ---------------------------------- |
| **ALR**            | \(G-1\)    | Full                     | None                               |
| **CLR**            | \(G\)      | \(G-1\) (rank-deficient) | \(\Sigma \mathbf{1} = 0\)          |
| **Softmax-normal** | \(G\)      | Full                     | Gauge freedom along \(\mathbf{1}\) |

SCRIBE adopts **ALR** for fitting (full-rank covariance, no constraints,
integrates directly with low-rank Gaussian machinery). After fitting, results
can be translated to **CLR** coordinates for symmetric biological
interpretation.

!!! info "Gauge fixing"
    The softmax map is invariant to adding a constant to all logits
    (\(\text{softmax}(\underline{x} + c\mathbf{1}) = \text{softmax}(\underline{x})\)).
    The ALR removes this one-dimensional null direction by construction. For
    Laplace inference, this gauge fix is essential---without it, the Hessian of
    the multinomial likelihood is singular along \(\mathbf{1}\).

---

## Contrast: Dirichlet vs logistic-normal

| Property                | Dirichlet                              | Logistic-Normal                                |
| ----------------------- | -------------------------------------- | ---------------------------------------------- |
| Parameters              | \(G\) concentrations                   | \((G-1)\) location + \((G-1)(k+1)\) covariance |
| Pairwise correlations   | Always negative                        | Arbitrary sign in log-ratio space              |
| Correlation freedom     | 0 (fixed by concentrations)            | \(O(Gk)\) (low-rank)                           |
| Closed-form DM integral | Yes                                    | No                                             |
| Biophysical origin      | Independent genes (Gamma steady state) | Interacting genes (LNA / Lyapunov)             |

---

## The LNM generative model

For a single cell \(c\):

**Step 1: Total count.** Draw from a negative binomial:

\[
u_T^{(c)} \sim \text{NB}(r_T, \hat{p}).
\]

**Step 2: Composition latent.** Draw in ALR space from the population:

\[
\underline{z}^{(c)} \sim \mathcal{N}(\underline{\mu},\,
\underline{\underline{\Sigma}}),
\quad \Sigma = WW^\top + \text{diag}(d).
\]

**Step 3: Count allocation.** Map to simplex and draw counts:

\[
\underline{\rho}^{(c)} = \text{ALR}^{-1}(\underline{z}^{(c)}),
\quad
\underline{u}^{(c)} \mid u_T^{(c)} \sim
\text{Multinomial}(u_T^{(c)},\, \underline{\rho}^{(c)}).
\]

The low-rank covariance \(\Sigma = WW^\top + \text{diag}(d)\) with
\(W \in \mathbb{R}^{(G-1) \times k}\) captures \(k\) dominant regulatory
programs; \(d\) captures gene-intrinsic noise.

!!! note "Total-composition independence"
    The factorization treats \(u_T\) and \(\underline{z}\) as independent given
    model parameters. This is a modeling choice, not a biophysical
    consequence---under the GRN, total and composition are generally coupled.
    The [PLN model](poisson-lognormal.md) avoids this assumption.

---

## LNMVCP: variable capture extension

The **LNMVCP** model (`model="lnmvcp"`) adds a per-cell capture latent
\(\eta^{(c)}\) that modifies the total count distribution while leaving the
composition block unchanged:

\[
u_T^{(c)} \mid \eta^{(c)} \sim \text{NB}(r_T,\, p(\eta^{(c)})),
\quad
\eta^{(c)} \sim \text{TruncN}(\eta_\text{anchor}, \sigma_M^2, \text{low}=0).
\]

The composition and capture are modeled with a **block-diagonal Hessian**: the
multinomial likelihood conditions on the observed \(u_T\), and the NB on totals
conditions only on \(\eta\). This block-diagonal structure means the two latents
decouple cleanly during Newton iteration:

- **Composition block:** Newton over \(\underline{z}\) (or \(\underline{y}_\text{ALR}\))
- **Capture block:** Scalar Newton on \(\eta\) (strictly log-concave 1D problem)

The scalar \(\eta\)-block converges to float precision in 1--2 Newton
iterations from any sensible warm start.

---

## Inference

The LNM supports two primary inference paths:

### Laplace approximation (recommended for LNMVCP)

```python
import scribe

results = scribe.fit(
    adata,
    model="lnmvcp",
    inference_method="laplace",
    n_steps=50_000,
)
```

The Laplace path is particularly well-suited to LNMVCP because it avoids
**encoder collapse**---a failure mode where the VAE encoder cannot track the
per-cell capture latent. Two Newton variants are available depending on
`d_mode`:

| `d_mode`              | Latent                                            | Newton cost              | Notes                            |
| --------------------- | ------------------------------------------------- | ------------------------ | -------------------------------- |
| `'low_rank'`          | \(\underline{z} \in \mathbb{R}^k\)                | \(O(k^3)\) per cell      | No Woodbury needed; small system |
| `'learned'` (default) | \(\underline{y}_\text{ALR} \in \mathbb{R}^{G-1}\) | \(O(Gk + k^3)\) per cell | Woodbury + Sherman--Morrison     |

The `d_mode='learned'` path uses the same Woodbury structure as PLN's Newton
solver, plus an additional **Sherman--Morrison** correction for the rank-1
outer product \(-u_T \rho\rho^\top\) in the multinomial Fisher information
matrix.

**Full details:** [Inference Methods > Laplace](../guide/inference.md#laplace-approximation)

### Variational Autoencoder (VAE)

```python
results = scribe.fit(
    adata,
    model="lnm",
    inference_method="vae",
    vae_latent_dim=10,
    n_steps=100_000,
    batch_size=256,
)
```

The VAE path uses a linear decoder (equivalent to the low-rank prior structure)
with an encoder that takes `log1p(proportions)` as input. This path is preferred
for representation learning and when amortized scoring of new cells is needed.

---

## Posterior geometry

The LNM posterior in \(\underline{z}\)-space is **log-concave** (the multinomial
log-likelihood is concave in the logits, and the Gaussian prior is log-concave).
However, it has a structural subtlety compared to the PLN:

- The multinomial is invariant to adding a constant to all logits
  (\(\text{softmax}\) gauge symmetry). Under the full \(G\)-logit
  parameterization, this creates a **singular Hessian** along \(\mathbf{1}\).
- The ALR gauge fix removes this null direction, but the Hessian can still have
  a **near-flat ridge** when \(W\mathbf{1}_k\) has a large component along
  \(\mathbf{1}\).
- The Gaussian prior breaks the exact degeneracy, ensuring the posterior is
  strictly log-concave, but Newton convergence may be slower along the
  near-flat direction.

This is why the PLN's Poisson likelihood (which has no softmax invariance and
stronger identification in every direction) often gives faster Newton
convergence on the same data.

---

## Posterior predictive checks

Two PPC modes are available:

| Mode                    | Description                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| **MAP-only**            | Fix \(\hat{z}\) at the MAP, draw Multinomial\((u_T, \text{softmax}(\hat{z}))\)           |
| **Laplace-uncertainty** | Sample \(\underline{z}\) from \(\mathcal{N}(\hat{z}, (-H)^{-1})\), then draw Multinomial |

For LNMVCP, the capture PPC samples \(\eta\) from its scalar Gaussian
approximation, draws \(u_T \sim \text{NB}\), then allocates via the composition
sampler.

---

## When to use LNM vs PLN

| Use LNM/LNMVCP when... | Use PLN when... |
|-------------------------|-----------------|
| Compositional analysis is the primary goal | Gene-level denoising with full correlation |
| You want explicit normalization built into the model | Total-count distribution matters (heavy tails) |
| Downstream analysis is in CLR/ALR space | Capture enters most naturally as log-offset |
| Total-composition independence is acceptable | You need the strongest posterior guarantees |

!!! tip "Practical recommendation"
    For most compositional analyses, start with `model="lnmvcp",
    inference_method="laplace"`. The variable capture extension handles
    library-size heterogeneity, and the Laplace path avoids the encoder
    collapse that can affect VAE inference on capture latents. See
    [Model Selection](../guide/model-selection.md) for the full decision guide.
