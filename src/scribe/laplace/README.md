# SCRIBE Laplace inference

This directory contains scribe's Laplace-approximation inference path. Unlike
the SVI / VAE path, Laplace inference has no encoder: each cell's per-cell
posterior is found locally via Newton iteration on the log-density, and the
Gaussian approximation is centered at the resulting MAP with covariance
$-H^{-1}$ from the Hessian curvature. The outer loop optimizes global
parameters (decoder $\mu$, $W$, $d$, NB-on-totals globals when present) by
SVI on the Laplace-approximated ELBO.

This module deliberately bypasses NumPyro's SVI machinery — it is a parallel
inference mode with its own enum (`InferenceMethod.LAPLACE`), engine, results
class, and orbax checkpointer. The only cross-submodule dependency is
`scribe.svi._progress_backend` (shared progress-bar infrastructure).

## When to use Laplace vs SVI/VAE

| Use Laplace when… | Use SVI/VAE when… |
|---|---|
| You need to score the *training* cells precisely | You need amortised inference for held-out cells |
| You suspect the encoder is collapsing on a per-cell latent | The encoder is well-calibrated for your data |
| You want the math-grade variational-EM guarantees (no aggregate-posterior drift, posterior uncertainty from the Hessian) | You need fast inference at serving time |
| Your data has high cell-to-cell variability that the encoder can't track | Your dataset is small enough that encoder collapse isn't a concern |

## Supported models

| Model | `inference_method="laplace"` | Notes |
|---|---|---|
| `pln` | ✅ | Per-cell `(x, η)` joint Newton with Schur back-substitution + Sherman–Morrison. |
| `nbln` | ✅ | NB-LogNormal: per-cell `(x, η)` Newton with the NB-Hessian diagonal `a_g = (u_g + r_g) p_g (1 - p_g)`. Adds gene dispersion `r` as a global; everything else mirrors PLN. |
| `lnm` | ✅ | Per-cell composition Newton; `d_mode='learned'` → Newton over `y_alr` (G−1 dim) with Woodbury; `d_mode='low_rank'` → Newton over `z` (k dim, no Woodbury). |
| `lnmvcp` | ✅ | LNM composition Newton + scalar Newton on per-cell `eta_capture`. The `(z, η)` Hessian is **block-diagonal** (multinomial conditions on observed `u_T`, NB conditions on η only) so the two blocks decouple cleanly. |
| `twostate_ln_rate` | ✅ | TwoState-LogNormal-Rate (PR-1 of the cross-gene TwoState extension): per-cell `(x, η)` Newton on a Poisson-Beta compound likelihood with closed-form factors `a_g = λ E_q[p] − λ² Var_q(p)`, `g_data,g = u − λ E_q[p]` (via fixed Gauss-Legendre quadrature). Cascades from a TwoState SVI fit. **W-shrinkage strategies beyond `NoneWPrior` are not yet wired** (transferrable from NBLN; deferred to a follow-up PR). |
| `twostate_ln_logit` | 🟨 | TwoState-LogNormal-Logit (PR-2 of the cross-gene TwoState extension): planned. Latent z enters on the activation log-odds rather than on the production rate; saturating mean response, exact gauge. Currently raises `NotImplementedError`. |
| `nbdm`, `nbvcp`, `zinb`, `zinbvcp` | ❌ | DM-family Laplace would require its own Newton kernel — no current implementation. |

## Architecture

All four supported models share the same generic Laplace-EM driver
(`_em.run_laplace_em`).  The driver owns every piece of cross-model
scaffolding: outer Adam, mini-batching, three-mode divergence detection,
best-snapshot recording, smoothed-loss patience early stopping, restore-best,
orbax checkpoint save/resume, progress reporting, final convergence check.

Per-model code lives in a thin observation-model adapter (`_obs_pln.py`,
`_obs_nbln.py`, `_obs_lnm.py`) that implements four hooks on the
`LaplaceObservationModel` ABC:

- `init_state` — build globals (`params`), per-cell latent (`latent_loc`),
  optional `eta_loc`, and any aux data.
- `loss_fn` — call the model's Newton kernel(s) under `stop_gradient`,
  compute `log det(-H)` at live globals, sum the per-cell joint log-density.
- `final_sweep` — full-data Newton pass for diagnostics.
- `pack_result` — assemble the canonical `LaplaceRunResult`.

Adding a new Laplace-supported observation channel reduces to writing one
~250-line adapter and adding one `elif` arm in
`LaplaceInferenceEngine.run_inference` — there is no engine-level
scaffolding to duplicate.

## File layout

```text
laplace/
  __init__.py             # Public API re-exports
  engine.py               # Thin LaplaceInferenceEngine dispatcher (~150 lines)
  _em.py                  # Generic Laplace-EM driver + LaplaceObservationModel ABC
  _obs_pln.py             # PLN observation-model adapter
  _obs_nbln.py            # NBLN observation-model adapter
  _obs_lnm.py             # LNM/LNMVCP observation-model adapter
  results.py              # ScribeLaplaceResults dataclass + mixin composition
  _core.py                # Model-agnostic parameter and correlation accessors
  _dispatch.py            # base_model-dispatching accessors (map/distributions/embeddings)
  _sampling.py            # Public PPC/predictive entry points
  _likelihood.py          # Public MAP log-likelihood entry point
  _gene_subsetting.py     # Gene-axis slicing behavior (PLN and ALR-safe LNM)
  _serialization.py       # Pickle hooks + plotting sample-cache compatibility
  _results_shared.py      # Shared constants and utility helpers
  _global_uncertainty.py   # Post-fit Hessian utilities for global-parameter posteriors
  _results_sampling_helpers.py   # Module-private PLN/LNM PPC backends
  _results_likelihood_helpers.py # Module-private PLN/LNM likelihood backends
  checkpoint.py           # Orbax checkpoint helpers
  _newton_pln.py          # PLN Newton kernels (joint x, η)
  _newton_nbln.py         # NBLN Newton kernels (joint x, η; NB Hessian)
  _newton_lnm.py          # LNM Newton kernels (z, y_alr, scalar η)
```

### Results mixin architecture

`ScribeLaplaceResults` now follows the same compositional style as
`ScribeSVIResults`: a single dataclass state object plus focused mixins.
This keeps the public API unchanged while making responsibilities easier to
find and maintain.

- `CoreResultsMixin`: shared accessors (`get_mu`, `get_W`, `get_d`,
  correlation diagnostics, and `get_p_capture`)
- `DispatchResultsMixin`: `base_model`-aware methods (`get_map`,
  `get_distributions`, `get_latent_embeddings`)
- `SamplingResultsMixin`: public predictive/PPC methods that route to
  private PLN/LNM sampling helpers
- `LikelihoodResultsMixin`: MAP log-likelihood API routed to private
  PLN/LNM likelihood helpers
- `GeneSubsettingResultsMixin`: `__getitem__` and PLN/LNM subsetting logic,
  including ALR reference safety checks
- `SerializationResultsMixin`: pickle-safe state handling and compatibility
  sample-cache properties (`predictive_samples`, `posterior_samples`)

### Interactive result display

`ScribeLaplaceResults` implements compact `__repr__` and `_repr_html_`
representations so notebook frontends render a small summary table instead of
expanding every array field in the dataclass. The summary includes:

- `model` (`base_model` from `model_config`)
- `n_cells`, `n_genes`, and `n_steps` (from `losses`)
- `latent` slots currently populated (for example `x,eta` or `y_alr`)
- `uncertainty` blocks currently populated (for example `r`, `mu`, `totals`)

## Configuration via `LaplaceConfig`

The full set of inner-loop and outer-loop knobs lives in
[`scribe.models.config.LaplaceConfig`](../models/config/groups.py). Every
field can be overridden either by passing a `LaplaceConfig` instance or a
plain dict to `scribe.fit(..., laplace_config=...)`:

```python
result = scribe.fit(
    adata,
    model="lnmvcp",
    inference_method="laplace",
    n_steps=50_000,
    laplace_config={
        "n_newton_steps": 15,           # default 5; bump for hard cells
        "damping": 1e-3,                # default 1e-2; tighten for tight MAP
        "newton_tolerance": 1e-3,       # default 1e-4; relax for production fits
        "convergence_action": "warn",   # warn / raise / ignore on non-convergence
    },
)
```

Top-level `scribe.fit` kwargs (`n_steps`, `batch_size`, `optimizer_config`,
`early_stopping`, `restore_best`, `log_progress_lines`) populate the
`LaplaceConfig` defaults; anything in `laplace_config` overrides those defaults.

## Training-time diagnostics

The progress bar emits a periodic loss-info line that combines four signals:

```text
LNM Laplace (learned + capture):  21%|██  |
  init loss: -8.857e+07,
  avg. loss [10001-10500]: -8.896e+07,
  comp max/p99/med 1.38e+01/3.42e+00/2.51e-03;
  η    max/p99/med 1.79e-06/1.61e-06/4.92e-07
```

Each piece tells you something distinct.

### `init loss` and `avg. loss [a-b]`

* `init loss` — the loss at the very first step. Kept as a scale reference so
  later "avg. loss" values are immediately comparable.
* `avg. loss [a-b]` — mean loss over the most recent display window
  (typically the last `n_steps // 100` steps).

The loss is the **negative Laplace ELBO**:

$$
\mathcal{L} =
- \sum_c \log p(u_c, \xi_c^*)
+ \tfrac{1}{2} \sum_c \log\det(-H_c).
$$

`mathcal{L}` may be **negative** (yes — really, that's normal) when the
Gaussian-prior log-density $\log p(\xi^*)$ has a large positive
contribution from $-\tfrac{1}{2} \log \det \Sigma$ for tight prior
covariances (small `d`). What you should watch is the **trend**:

* Healthy → monotonically decreasing, eventually plateauing (exponential
  approach to the local minimum).
* Stalled / diverging → loss flat or increasing → kill the run, something
  numerical is wrong.

### `comp max/p99/med` and `η max/p99/med`

Per-cell Newton gradient-norm summary, split by block. The three numbers per
block are:

| Statistic | Meaning |
|---|---|
| `max` | Strict L∞ over cells. The single worst-converged cell. Used by `convergence_action` for the end-of-training warning. |
| `p99` | 99th percentile across cells. Filters out the single worst outlier; reflects the broader tail. |
| `median` | Typical cell. On a healthy fit this should be well below `newton_tolerance`. |

Reading the three together:

* **Healthy** — `median` ≪ tolerance, `p99` close to median, `max` larger
  but trending down. Most cells are converged; a few outliers (often
  low-count cells with rank-deficient multinomial Fisher) are still
  finishing.
* **A few problem cells** — `median` and `p99` small, `max` large and
  bouncing. Newton converges most cells fine but is fighting a small
  number of pathological cells. Usually fine for the global fit.
* **Genuine convergence problem** — `median` plateaus *and* `max` doesn't
  trend down. Indicates a real issue (Hessian conditioning, step-size cap,
  numerical precision). Bump `n_newton_steps`, tighten `damping`, or check
  for outlier cells.

### Per-block split: composition vs η

For **LNMVCP** (`comp` and `η` blocks):

* The **composition block** is Newton over $z$ (low_rank) or $y_\text{alr}$
  (learned). It can be slow on cells with very few active genes because the
  multinomial Fisher matrix
  $M_\text{alr} = u_T (\mathrm{diag}(\rho) - \rho\rho^\top)$ has rank
  $\le u_T$ — much smaller than $G-1$ on typical scRNA-seq. Newton's
  quadratic convergence rate degrades to *linear* along the rank-deficient
  directions, where the only curvature is the prior's $\Sigma^{-1}$.
* The **η block** is a scalar Newton on a strictly log-concave 1D
  problem. It converges to float-precision (~`1e-6`) in 1–2 Newton
  iterations from any sensible warm start.

For **PLN with capture anchor** (`x` and `η` blocks):

* The **`x` block** is Newton over the latent log-rate
  $x_c \in \mathbb{R}^G$. The Hessian is full-rank for any cell with
  positive counts (every diagonal entry of $-H_{xx}$ is
  $\exp(x_g - \eta) + \Sigma^{-1}_{gg} > 0$), so per-cell convergence is
  typically faster than LNMVCP's composition block.
* The **η block** is the per-cell capture-offset latent. Unlike LNMVCP,
  PLN's $(x, \eta)$ Hessian is *not* block-diagonal — $x$ and $\eta$
  are coupled through the Poisson rate $\exp(x_g - \eta)$, and Newton
  uses a Schur-complement back-substitution to solve the joint system.
  The split here is computed from $\nabla f$ at the post-Newton MAP,
  not from a separate Newton solve.

In both cases:

* If `η` is at `~1e-6` but the composition / `x` block is much larger,
  the engine plumbing is correct and the bottleneck is the composition
  geometry (multinomial Fisher rank for LNMVCP) or some specific cell
  (low counts for PLN).
* If `η` were also high, that would point to a numerical bug.

### Plain LNM, PLN without capture, low_rank d_mode

* **Plain LNM** has no per-cell capture latent, so the η column is
  suppressed: only `Newton grad max/p99/med <numbers>` is shown.
* **PLN without capture anchor** (no `priors={"capture_efficiency": ...}`)
  also has no η latent; Newton runs over $x$ alone and the display is
  `Newton grad max/p99/med <numbers>` (single block).
* **low_rank d_mode** (LNM only): no diagonal residual is fit, so the
  composition latent is just the k-dim $z$. Newton is k×k, much smaller
  than the y_alr branch's (G-1)×(G-1).

## End-of-training convergence handling

The engine runs one final Newton sweep over all cells (longer than the
inner-loop iterations during training) and reports the worst per-cell
gradient norm. If it exceeds `newton_tolerance`, behavior is controlled
by `LaplaceConfig.convergence_action`:

| Value | Effect |
|---|---|
| `"warn"` (default) | Emit a `logging.warning` listing the number of non-converged cells. |
| `"raise"` | Raise `RuntimeError` and abort. |
| `"ignore"` | Silent. |

The full per-cell final-grad array is stored on the result as
`result.final_grad_norms` regardless of `convergence_action`, so you can
inspect the distribution post-hoc:

```python
import numpy as np
gn = np.asarray(result.final_grad_norms)
print(f"Cells: {gn.size}")
print(f"max: {gn.max():.3e}, p99: {np.percentile(gn, 99):.3e}, "
      f"median: {np.median(gn):.3e}")
print(f"Cells above 1e-3: {(gn > 1e-3).sum()}")
```

## Single-cell explosive divergence

Occasionally an LNM(VCP) fit will run smoothly for thousands of steps and
then catastrophically diverge: loss jumps by several orders of magnitude,
`comp max` spikes to `1e+3` or higher while `comp p99` and `comp median`
remain small. This signature is **one or a handful of cells whose per-cell
Newton has wandered into an unstable regime** — most commonly cells whose
softmax probability concentrates near a single gene (e.g., a low-count
cell where one gene happens to have all the counts), driving the
Sherman–Morrison denominator `1 - N · ρᵀ A⁻¹ ρ` close to zero. Float32
cancellation in that subtraction produces wildly wrong Newton steps, and
the divergence cascades through the loss into the gradient on globals.

The engine has three layered defenses:

1. **Sherman–Morrison denominator floor** in
   `scribe.laplace._newton_lnm.newton_step_y_alr`. The denominator is
   floored at 1% of `1/N` (a relative floor that scales with cell total
   count), well above the float32 catastrophic-cancellation regime.
2. **Per-cell NaN/Inf mask** inside each observation-model `loss_fn`
   (`_obs_pln.py`, `_obs_nbln.py`, `_obs_lnm.py`).  Per-cell ELBO
   contributions that go non-finite are replaced with zeros before the
   batch sum, masking the divergent cells from the current step's
   gradient on globals.
3. **Outer-loop divergence detector** in `_em.run_laplace_em`.  Three
   conditions trigger a clean abort that restores the best snapshot
   seen so far:
   - Loss becomes NaN or Inf.
   - Loss climbs more than `0.5 × |init_loss|` above its running min
     (after a 50-step warmup).
   - `|loss|` grows by more than 1000× from `|init_loss|` (last-line
     backstop).

When a divergence guard fires, the driver **restores the best snapshot
seen so far** and continues through the final-convergence sweep + result
packaging.  The returned `LaplaceRunResult` carries
`divergence_aborted=True` and `early_stopped=True`, plus the best-loss
globals and per-cell MAPs at the running-minimum loss — typically a
meaningful improvement over the data-driven init even when the run
aborted early.  Look for the warning emitted to the
`scribe.laplace._em` logger to know which guard fired.

Remediation, in order of escalation:

- Bump `laplace_config['n_newton_steps']` to 20–30 to give Newton more
  iterations to escape the unstable regime before damage propagates.
- Tighten `laplace_config['damping']` to 1e-3 or below so the
  Tikhonov regularization helps the Newton solve stay well-conditioned.
- Pre-filter outlier cells by total count or compositional skew (e.g.,
  drop cells with `u_T < 50` or with one gene comprising > 80% of
  counts) before fitting.

After an abort, identify the offending cells by inspecting
`result.final_grad_norms` from a shorter completed run — cells with
`final_grad_norms[c] >> newton_tolerance` are the ones to investigate.

## Troubleshooting flowchart

```text
worst comp grad doesn't converge below tolerance
│
├─ median grad < tolerance?
│   ├─ YES → only outlier cells; fit is fine for the bulk.
│   │        Optional: relax newton_tolerance, or bump
│   │        n_newton_steps if the outlier count grows.
│   │
│   └─ NO  → bulk is non-converged; bump n_newton_steps to
│            10–30. If still not enough, tighten damping
│            (1e-3 → 1e-4) and check logs for NaNs / overflow.
│
├─ loss is decreasing?
│   ├─ YES → globals are improving. Worst-grad lag is a
│   │        symptom of the rank-deficient Hessian; it will
│   │        shrink as globals stabilize.
│   │
│   └─ NO  → numerical issue. Check init: empirical mu
│            (from log-counts) should not be NaN, W should
│            be a proper PCA loading matrix.
│
└─ η-block converged to ~1e-6?
    ├─ YES → engine plumbing is correct; bottleneck is
    │        genuinely the composition geometry.
    │
    └─ NO  → engine bug. The η-block has no rank issues
             and should always converge to float precision.
             Check for NaNs in r_T, mu_T, eta_anchor.
```

## Global parameter posterior uncertainty

After EM convergence and the final Newton sweep, the Laplace path computes an
approximate posterior for selected global parameters using the **profiled
observed-information Hessian**. This extends the Laplace-EM framework to
provide uncertainty quantification for globals that were previously represented
only by MAP point estimates.

### How it works

At convergence, the global negative-ELBO objective $\mathcal{L}(\theta)$
implicitly depends on all per-cell latent MAPs $\xi_c^*(\theta)$. The
**profiled Hessian** is the Schur complement:

$$
H_\text{profile} = H_{\theta\theta}
  - H_{\theta z}\, H_{zz}^{-1}\, H_{z\theta}
$$

This correction accounts for the fact that per-cell latent MAPs shift when
globals $\theta$ change, preventing the posterior scale from being
underestimated.

The posterior covariance in **unconstrained** space is then:

$$
\Sigma_\theta \approx H_\text{profile}^{-1}
$$

Constrained positive parameters ($r$, $\mu_T$, $r_T$) are obtained by
applying `model_config.positive_transform` (default `softplus`) to the
unconstrained `*_loc` values.

### Per-model behavior

| Model      | Global parameters                | Uncertainty structure                                                                                          |
| ---------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| PLN        | None                             | Empty (PLN's globals are the latent-Gaussian parameters, already covered by the population-level distribution) |
| NBLN       | $r_g$ (gene-specific dispersion) | Diagonal covariance: independent $\text{Normal}(r\_loc_g, r\_scale_g)$ per gene                                |
| LNM/LNMVCP | $\mu_T, r_T$ (NB-on-totals)      | Full $2 \times 2$ covariance: $\text{MVN}([mu\_T\_loc, r\_T\_loc], \Sigma_{2 \times 2})$                       |

### Accessing uncertainty

```python
result = scribe.fit(adata, model="nbln", inference_method="laplace", ...)

# Unconstrained posterior parameters
m = result.get_map()
m["r_loc"]    # unconstrained location (shape G)
m["r_scale"]  # unconstrained scale (shape G)
m["r"]        # constrained MAP: positive_transform(r_loc)

# NumPyro distributions
dists = result.get_distributions()
dists["r_unconstrained"]  # Normal(r_loc, r_scale).to_event(1)
dists["r"]                # TransformedDistribution (SoftplusTransform)

# LNM totals
dists["totals_unconstrained"]  # MultivariateNormal (2D)
dists["mu_T"]                  # TransformedDistribution (marginal)
dists["r_T"]                   # TransformedDistribution (marginal)
```

### PPC behavior with global uncertainty

Non-MAP posterior predictive checks (`get_ppc_samples`,
`get_per_cell_predictive_samples`) **always include global parameter
uncertainty** by default. Each predictive sample draws fresh global
parameters from their posterior before generating counts:

- **NBLN (non-frozen)**: each sample draws $r_g$ from
  $\text{Normal}(r\_loc, r\_scale)$ and maps through `positive_transform`.
  `mu` is held at its point estimate (the Laplace `mu_scale` carries
  gauge-slop contamination and is deliberately *not* propagated through
  PPC; honest `mu` uncertainty for cascade fits is available via the
  frozen-cascade routing below).
- **NBLN (cascade-frozen, Phase-2)**: for each parameter listed in
  `result.frozen_params`, samples are drawn from
  `result.cascade_source` (the embedded SVI guide) and transformed
  into the NBLN target coordinate per sample. Full SVI guide fidelity
  is preserved (the routing does not moment-match to a Normal). See
  "Cascade-aware PPC routing" below for the full sampler/coordinate map.
- **LNM/LNMVCP**: each sample draws $(\mu_T, r_T)$ jointly from
  $\text{MVN}(\text{loc}, \Sigma_{2 \times 2})$ and maps through
  `positive_transform`.

`get_map_ppc_samples` remains MAP-only and does not incorporate global
uncertainty.

This can widen predictive tails and credible bands compared to previous
MAP-only PPCs, especially for NBLN where dispersion uncertainty enters
multiplicatively into the predictive variance.

#### Conditioning levels (`ppc_level`)

`get_ppc_samples` (and `scribe.viz.plot_ppc`) accept a `level` /
`ppc_level` kwarg that selects how much observed-data information
enters each predictive draw:

| Level | Latent `x` | `η` | Per-cell totals `N_c` | Use for |
|---|---|---|---|---|
| `marginal` | fresh from $\mathcal{N}(\mu, WW^{\top} + \mathrm{diag}(d))$ | from cascade / Laplace posterior | drawn from the model (NB sampling against `exp(x − η)`) | Honest "does the entire generative story match the data?" test — **no** observed-data conditioning |
| `library_anchored` (default) | fresh from $\mathcal{N}(\mu, WW^{\top} + \mathrm{diag}(d))$ | (cancels via softmax) | **observed** (`Multinomial(N_c^{obs}, p_s)`) | Compositional fit test, isolated from the totals/capture submodel |
| `per_cell` | per-cell MAP `x_loc` + Laplace noise | per-cell MAP `eta_loc` | **observed** | Per-cell predictive — most conditioned |

`plot_ppc` defaults to `library_anchored` because the histogram aggregation
is most interpretable under fixed totals. For honest population-level
predictive tests, pass `ppc_level="marginal"`.

### Important caveats

- **Diagonal NBLN approximation**: the diagonal covariance for NBLN $r_g$
  ignores cross-gene covariance. Per-gene credible intervals are the intended
  use case; joint gene-set hypotheses may need a richer covariance
  approximation.
- **Gene subsetting semantics**: sliced uncertainty fields (`r_loc[genes]`,
  `r_scale[genes]`) are marginals from the model fitted on the full gene
  panel, not the posterior one would obtain by refitting on only the selected
  genes.
- **Unconstrained space**: all `*_loc` and `*_scale` fields live in the
  unconstrained pre-transform space. The `*_scale` fields are NOT standard
  deviations of the constrained positive parameter.
- **Curvature diagnostics**: non-positive profiled curvature entries (locally
  unidentified parameters) are floored and logged as warnings. Inspect
  `result.metadata` for Hessian diagnostic fields if needed.

### NBLN `mu` posterior

In addition to `r_loc, r_scale`, NBLN-Laplace exposes `mu_loc, mu_scale`
— a per-gene Gaussian approximation of the latent-prior mean `mu` in
log-rate coordinates. Like `r`, `mu` uses a **diagonal-Σ
approximation**: each gene's posterior is computed as if the cross-gene
coupling through `Σ^{-1}` were zero. The diagonal value
`(Σ^{-1})_{gg}` is exact (from `woodbury_inv_diag`); only the off-
diagonals are dropped. Unlike `r`, `mu` is *not* a positive parameter
— it is real-valued in NBLN's coordinate system (the prior mean of the
latent log-rate). No `positive_transform` is applied; the constrained
distribution returned by `get_distributions()` is `Normal(mu_loc,
mu_scale).to_event(1)` directly.

## SVI-to-Laplace informative-prior cascade (TwoState → TSLN-Rate)

Mirrors the NBLN cascade pattern below, but the SVI source is a
`TwoState` (or `TwoStateVCP`) fit and the target is
`twostate_ln_rate`. The adapter lives in `scribe.laplace.priors`
under `priors_from_twostate_results` and
`freeze_values_from_twostate_results`.

### Coordinate map

| TSLN-Rate target | SVI source (TwoState) | Transform |
|---|---|---|
| `mu_loc` | `mu` (positive gene mean) | `pos_inverse` (`inv_softplus` for `softplus`, `log` for `exp`) |
| `burst_size_loc` | `burst_size` (positive) | `pos_inverse` |
| `k_off_loc` | `k_off` (positive) | `pos_inverse` |
| `eta` | `eta_capture` (when present; biology-informed capture) | identity |

All three positive globals map through `pos_inverse` to the
unconstrained location space — the cascade samples a posterior
draw, applies `pos_inverse`, and moment-matches a Gaussian. The
``logit`` variant (PR-2) is gated with `NotImplementedError`.

### Recommended freeze level

The default cascade for TSLN-Rate is
`freeze_params=("mu", "burst_size", "k_off")` — all three gene-level
positive globals hard-frozen at the SVI MAP. This is the "Level 4"
freeze from the cross-gene plan: it structurally pins the rigid-
translation gauge between `log r_hat_g` and the per-cell latent
`z_c`, so the Laplace fit's Newton iteration cannot drift along
the gauge null direction.

```python
svi_results = scribe.fit(
    adata, model="twostate", inference_method="svi", n_steps=20_000,
)

laplace_results = scribe.fit(
    adata, model="twostate_ln_rate", inference_method="laplace",
    informative_priors_from=svi_results,
    informative_priors_freeze=("mu", "burst_size", "k_off"),
    n_steps=50_000,
)
```

### Result-dataclass semantics

`ScribeLaplaceResults.mu` for `twostate_ln_rate` carries the
**latent log-rate prior center** (`log(r_hat)`) to match the
PLN/NBLN convention where `LowRankMultivariateNormal(loc=self.mu, …)`
is the latent distribution. The **positive TwoState gene-mean
parameter** lives on `result.gene_mean` (and the unconstrained
version on `gene_mean_loc`). Plotting code that reads `result.mu`
treats it as the log-rate; biology-side reporting should read
`result.gene_mean` instead.

### Curvature-clamp diagnostics

The Poisson–Beta marginal is not uniformly log-concave (the Beta
density is U-shaped when `α` or `β` < 1), so the Hessian-diagonal
factor `a_g = λ E_q[p] − λ² Var_q(p)` is defensively floored at
`_A_MIN = 1e-3` before the Woodbury solve. Four diagnostic fields
on `ScribeLaplaceResults` surface how often the clamp activated:

- `a_raw_min` — scalar minimum of the un-clamped `a_raw` across
  all `(cell, gene)` entries.
- `a_raw_negative_fraction` — fraction with `a_raw < 0` (genuine
  log-concavity violations).
- `a_clamp_fraction` — fraction where the `_A_MIN` floor activated.
- `a_clamp_per_gene` — per-gene clamp-activation rate, shape `(G,)`.

A `logging.warning` is emitted at end-of-training if
`a_clamp_fraction > 0.05`, pointing the user at the per-gene
breakdown.

## SVI-to-Laplace informative-prior cascade (NBLN only)

For NBLN-Laplace fits, you can pass a previously-fit `ScribeSVIResults`
object (typically NBVCP-SVI on the same dataset) and have it derive
empirical Gaussian priors on the NBLN globals `r`, `mu`, and (when
the SVI source has `eta_capture`) per-cell `eta`. The priors enter the
Laplace loss as proper `Normal(loc, scale).log_prob(...)` terms — so
their **uncertainty** (not just their location) shapes both training
dynamics and post-fit global Hessian.

```python
svi_results = scribe.fit(adata, model="nbvcp",
                         parameterization="mean_odds",
                         priors={"capture_efficiency": (np.log(100_000), 0.5)},
                         inference_method="svi", n_steps=250_000)

laplace_results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    informative_priors_from=svi_results,
    informative_priors_tau=1.0,          # trust SVI exactly; raise to
                                          # 2-3 in noisy / sparse regimes
    informative_priors_n_samples=1000,
    n_steps=50_000,
)
```

### Coordinate handling

The adapter (`scribe.laplace.priors.priors_from_results`) moves SVI
posterior samples from their **constrained** space into the target
NBLN-Laplace **unconstrained** coordinate space:

| NBLN target | SVI source | Transform applied |
|---|---|---|
| `r_loc` (per gene) | `r` (positive) | inverse of target `positive_transform` |
| `mu` (per gene) | `mu` (positive NB mean) | plain `jnp.log` (NBLN `mu` is real-valued log-rate; **never** uses `positive_transform`) |
| `eta_anchor`, per-cell `sigma_eta` | `eta_capture` (constrained ≥ 0) | identity (matches the target's TruncatedNormal prior) |

Posterior samples (not extracted variational parameters) are the
intermediate representation because they are parameterization-agnostic:
SVI may have used `mean_odds`, `mean_prob`, or `canonical`
parameterizations, but the samples are always in the natural
constrained coordinate.

### Capture-mode trichotomy

Detection from the SVI samples dict:

| Source state | Mode | Behavior |
|---|---|---|
| `eta_capture` present | `"eta"` | Per-cell prior supersedes the scalar `capture_anchor`. Activates capture on the target even without an explicit `capture_efficiency` prior. |
| only `phi_capture` / `p_capture` | `"phi_only"` | Warn; apply `r` and `mu` priors only; leave target capture configuration intact. |
| no capture keys | `"none"` | Warn; apply `r` and `mu` priors only; leave target capture configuration intact. |

The two "leave intact" branches are deliberate: SVI provides
*information about the source*, not *permission to override the
target*. A user who passes both `capture_efficiency=...` and a
non-eta SVI source still gets their explicit capture anchor.

### Safeguards

- **Strict gene identity** check, via priority `var_names > mask > count`.
  A var-names mismatch raises `ValueError` even when counts agree.
- **Amortized-capture sources** require either `results._original_counts`
  to be stored *or* strict var-name identity verified plus `source_counts`
  passed. Mask-only identity is **not** sufficient (the encoder's input
  dimension is fixed at SVI fit time; an identity-by-mask check does
  not guarantee positional equivalence in the encoder's input space).
- **Scope validation** at the API layer rejects `informative_priors_from`
  for non-NBLN or non-Laplace fits with a clear error.
- **`d_loc`** is never overridden; NBVCP has no `d` counterpart and the
  prior would be ill-posed.

### Practical recommendations

- Default `informative_priors_tau=1.0` trusts SVI exactly. Raise to
  `2-3` if the SVI posterior is over-confident on noisy or sparse
  datasets (small N or low coverage).
- The cascade is the recommended fix for NBLN-Laplace divergence on
  low-count low-`r` data, where the per-cell Newton's curvature
  `(u+r) p (1-p)` collapses if `r` drifts far from its data-supported
  value. The SVI prior pins `r` and unlocks stable optimization of
  `(W, d)`.

### Subset-aware cascade across heterogeneous `gene_coverage` settings

When the upstream SVI fit and the downstream Laplace fit use
**different** `gene_coverage` thresholds, the cascade must reconcile
the two gene panels. Higher `gene_coverage` keeps more genes, so a
typical broad-to-narrow workflow has

```text
SVI gene_coverage  >=  Laplace gene_coverage
```

and the Laplace panel is a strict subset of the SVI panel. The
cascade adapter auto-detects this and pools the SVI's per-gene
posteriors on the SVI-kept-but-Laplace-dropped genes (plus the SVI's
own `_other` posterior if present) into the Laplace target's pooled
`_other` column via per-sample NB **moment matching**:

```text
μ_other^(s) = μ_svi_other^(s) + Σ_{g ∈ dropped} μ_g^(s)
r_other^(s) = (μ_other^(s))² / [(μ_svi_other^(s))²/r_svi_other^(s)
                                + Σ_{g ∈ dropped} μ_g^(s)²/r_g^(s)]
```

aggregated per posterior sample, then moment-matched to a Gaussian
prior in the NBLN target coordinate exactly as for the other per-gene
slots. See [`paper/_nb_lognormal.qmd`](../../../paper/_nb_lognormal.qmd)
section `sec-nbln-cascade-aggregation` for the full derivation.

**Behavior matrix.**

| Relationship between SVI and Laplace gene panels | Behavior |
|---|---|
| SVI panel == Laplace panel | Pass-through (bit-equal to legacy). |
| SVI panel ⊃ Laplace panel | Auto-on aggregation; emits explicit `INFO` log line. |
| SVI panel ⊉ Laplace panel | `ValueError` listing first 10 missing genes. |
| SVI is amortized-capture **and** panels differ | `NotImplementedError`. |
| TSLN-Logit cascade with panels differ | `NotImplementedError`. |

**Important caveats.**

- The pooled `_other` prior is exact in the **first two moments**
  under the upstream's conditional-independence assumption. It is
  *not* distributionally exact: a sum of independent NBs with
  gene-specific `p_g` is not itself NB except in the shared-`p` NBDM
  limit. The pooled prior is the closest-NB approximation of the
  aggregate's mean and variance, sufficient for anchor-prior use.
- TSLN-Rate uses an analogous but **approximate** aggregator: μ is
  additive (exact), while `burst_size` and `k_off` inherit the SVI's
  `_other` column when present, falling back to per-sample medians.
  The two-state telegraph parameters do not close under summation.
- TSLN-Logit subset cascades raise `NotImplementedError`; the
  `(rate, kappa, eta_anchor)` reparameterization is even less
  tractable. Use TSLN-Rate for broad-to-narrow cascades.
- Post-fit PPC on the `_other` column draws from the
  moment-matched aggregate (re-aggregated at sample time in
  `_resolve_nbln_ppc_arrays`), not from the SVI guide's own `_other`
  column directly — the two represent different aggregates.

### Decorrelating the `_other` aggregate from Σ (`correlate_other_column`)

The trailing `_other` column emitted by the gene-coverage stage is a
pooled-counts aggregate, not a real gene. Its row in the latent
low-rank covariance `Σ = W Wᵀ + diag(d)` has no biophysical meaning —
including it wastes capacity on spurious cross-gene correlations and
biases the `W` loadings used for regulatory-program identification
and gauge-invariant diagnostics (see
[`paper/_diffexp_nbln_robustness.qmd`](../../../paper/_diffexp_nbln_robustness.qmd)
Theorem 2). The `correlate_other_column: bool` flag on `ModelConfig`
controls whether the trailing `_other` row participates in Σ.

**Current default in this release: `True`** (legacy — `_other`
participates in Σ, identical to pre-flag behaviour). The default is
deliberately held at `True` because the decoupled-math path (`False`)
is implemented only as scaffolding in this commit; the actual
deviation-parameterisation math (loss / Newton / global-uncertainty)
lands in Commit 2b. When 2b ships, the default flips to `False`
(the biologically cleaner setting) and `True` becomes the explicit
legacy opt-in. Under `False`, the layout has:

```text
W shape: (G_kept, K)        # latent-covariance axis (no _other row)
d shape: (G_kept,)
mu shape: (G_obs,)           # observation-layer axis (with _other)
r shape: (G_obs,)            # NB dispersion, observation-layer
per-cell x_dev: (G_kept,)    # deviation from μ_kept, prior N(0, Σ_kept)
```

Effective per-gene log-rate fed to the NB likelihood:

- For kept gene `g` at kept-position `k`: `μ[g] + x_dev[k] − η_c`
- For `_other`: `μ[other_idx] − η_c` — no z-modulation

**Current landing status (Commit 2 of the harmonic-hare plan).** This
release lands the **scaffolding** for the decorrelated layout — the
`AxisLayout` abstraction, the signal threading from the
`gene_coverage` stage through the engine to the obs model,
`ScribeLaplaceResults.axis_layout` + `G_obs` / `G_kept` properties,
the layout-aware `init_state` (W and d sized to `G_kept`), and the
`pack_result` plumbing. The **deviation-parameterised math** in the
loss / Newton / `compute_global_uncertainty` paths is tracked in
Commit 2b on the `feature/decorrelate_other_column` branch (the
plan's per-commit ladder gives the auditor a focused review for
the scaffolding before the math lands).

**Default is held at `True` (legacy) for Commit 2** so existing
`gene_coverage < 1.0` fits do not break by routing through the
not-yet-implemented decoupled math. When Commit 2b ships the math,
the default flips to `False` (the new biologically cleaner default)
and `True` becomes the explicit legacy opt-in.

Behaviour matrix for this release (Commit 2, default=True):

| Configuration | Behaviour |
|---|---|
| No `gene_coverage` filter (no `_other` column) | Trivial layout (`G_kept == G_obs`); bit-equal to today regardless of flag. |
| `gene_coverage < 1.0` AND `correlate_other_column=True` (current default) | Legacy: `_other` participates in Σ; trivial layout; bit-equal to today. **Recommended path for Commit 2 until Commit 2b lands.** |
| `gene_coverage < 1.0` AND `correlate_other_column=False` (explicit opt-in) — NBLN | Decoupled layout detected; `loss_fn` raises `NotImplementedError` with a clear message pointing at Commit 2b. |
| `gene_coverage < 1.0` AND `correlate_other_column=False` (explicit opt-in) — TSLN-Rate | Scaffolded as of Commit 3: AxisLayout + init-shape slicing + `pack_result` plumbing are in; the obs model's `init_state` raises `NotImplementedError` pointing at TSLN-Rate's math commit (3b). |
| `gene_coverage < 1.0` AND `correlate_other_column=False` (explicit opt-in) — TSLN-Logit | Scaffolded as of Commit 4: AxisLayout + `pack_result` plumbing are in; the obs model's `init_state` raises `NotImplementedError`. Per-gene `rate` / `kappa` / `eta_anchor` stay on G_obs under decoupled — only `W` / `d` / per-cell `z` shrink to `G_kept`. |
| `gene_coverage < 1.0` AND `correlate_other_column=False` (explicit opt-in) — PLN | Scaffolded as of Commit 5: AxisLayout + `pack_result` plumbing are in; the obs model's `init_state` raises `NotImplementedError`. As of Commit 5 the engine early-fail block has been **retired entirely** — every affected model owns its own decoupled detection via its obs-model `init_state`. |
| Array-input fit (no AnnData) with pooled `_other` | Detected via `ctx._has_pooled_other` primary signal — array fits do NOT silently fall back to legacy when the user explicitly sets `correlate_other_column=False`. |
| AnnData fit with no gene_coverage but `var_names[-1] == "_other"` (manually-pre-filtered AnnData) | Detected via the AnnData var_names fallback (rev-4 #3); the layout factory honours the literal `_other` sentinel even without the gene_coverage stage running. |
| `gene_coverage < 1.0` AND `correlate_other_column=False` — LNM / LNMVCP | The ALR reference is auto-pinned to the `_other` position by `apply_gene_coverage_and_alr` (the min-variance auto-selection is skipped under this flag).  LNM realises the decoupling through ALR construction; no deviation reparameterisation needed. |
| `gene_coverage < 1.0` AND `correlate_other_column=False` AND explicit `alr_reference_idx` points to a retained gene — LNM / LNMVCP | `apply_gene_coverage_and_alr` raises `ValueError`: the explicit reference contradicts the flag's intent.  The user must either drop the override (auto-pin to `_other`) or pass `correlate_other_column=True`. |
| `gene_coverage < 1.0` AND `correlate_other_column=True` (legacy) AND any explicit `alr_reference_idx` — LNM / LNMVCP | Pass-through: any retained-gene reference accepted silently; references to `_other` raise (preserves today's contract). |
| Direct `ModelConfig` construction bypassing `apply_gene_coverage_and_alr` with inconsistent `correlate_other_column` / `alr_reference_idx` — LNM | `LNMObservationModel.init_state` raises `ValueError` at init time, naming both flags.  Defensive guard for users who skip the standard engine path. |

**Disagreement is loud.** When the data context provides BOTH a
`has_pooled_other` flag AND a `gene_names` tail and the two disagree
(`has_pooled_other=True` but `gene_names[-1] != "_other"`, or
vice-versa), `build_axis_layout` raises `ValueError` rather than
silently choosing one — silent disagreement here can corrupt the
axis split downstream.

**LNM real wiring (Commit 6, landed).** LNM excludes the ALR
reference gene from Σ by construction.  Under
`correlate_other_column=False` AND a pooled `_other` column, the
gene-coverage stage now pins the ALR reference to `_other`'s
position automatically: the min-variance auto-selection is skipped
because the variance-winner is typically a low-expression
individual gene, not the pooled aggregate.  Explicit overrides to
a non-`_other` reference under this flag raise `ValueError` with
a message pointing at both flags.  Under legacy
`correlate_other_column=True`, explicit references to any retained
gene continue to be accepted as today (bit-equal contract).

A defensive consistency check at `LNMObservationModel.init_state`
catches the case where a user constructs `ModelConfig` directly and
bypasses the gene-coverage stage; it raises `ValueError` if the
detected pooled `_other` and the configured ALR reference disagree.
Under legacy, the check is a no-op (the trivial `AxisLayout` short-
circuits before the position check).  See
`paper/_nb_lognormal.qmd` §sec-nbln-decorrelate-lnm for the
biophysical rationale and `paper/_logistic_normal_multinomial.qmd`
§sec-lnm-alr-pooled-other for the LNM-side cross-reference.

**PLN underdispersion caveat.** Under `correlate_other_column=False`,
PLN's `_other` column has `log_rate = μ_other − η_c` —
deterministic up to the capture offset.  The marginal on the
pooled count collapses to `Poisson(exp(μ_other − η_c))` with
variance equal to mean.  When the pooled tail exhibits substantial
overdispersion (common in practice, since it aggregates many
low-expression genes), PLN's predictive intervals on this column
will be unrealistically narrow.  NBLN / TSLN-Rate / TSLN-Logit do
not have this issue — their per-gene NB dispersion (or burst
kernel) fits the `_other` column freely on the observation axis.
For PLN fits where the pooled tail's overdispersion is biologically
meaningful, pass `correlate_other_column=True` to retain `_other`
in Σ (the legacy layout couples `μ_other` to the latent through
the diagonal `d_other` term).  See
`paper/_nb_lognormal.qmd` §sec-nbln-decorrelate-pln-caveat for the
mathematical treatment.

**Current default-flip schedule.** Commit 2 ships with the default
held at `True` (legacy) so existing `gene_coverage < 1.0` fits do
not break by routing through the not-yet-implemented decoupled
math. When Commit 2b ships the math, the default flips to `False`
and `True` becomes the explicit legacy opt-in. See
`ModelConfig.correlate_other_column` docstring for the schedule.

### Phase-2: Cascade-parameter freeze

The soft cascade above injects Gaussian priors with finite scale; on
real data this is sometimes insufficient to break the **per-cell
rigid-translation gauge** between `x_c` and `eta_c` (see
[`paper/_diffexp_nbln_robustness.qmd`](../../../paper/_diffexp_nbln_robustness.qmd)).
The Phase-2 **freeze mechanism** fixes selected globals exactly at the
SVI MAP, structurally pinning the gauge:

```python
laplace_results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    informative_priors_from=svi_results,
    informative_priors_freeze=("r", "eta"),   # default — Level 3
    n_steps=50_000,
)
```

Four levels of aggressiveness via the tuple kwarg:

- `()`        — Level 1: no freeze (soft cascade only, finite τ).
- `("r",)`    — Level 2: freeze dispersion; keep μ, η refined.
- `("r", "eta")` — **default**. Pins the per-cell gauge structurally.
  Empirically reduces `gauge_contamination_ratio` < 0.05.
- `("r", "mu", "eta")` — Level 4: NBLN learns only W and d on top of NBVCP.

Frozen parameters are **excluded from the optax optimizer's params
dict** — they cannot drift, regardless of optimizer internals. The full
`ScribeSVIResults` is embedded on `result.cascade_source` for downstream
PPC/distribution access; counts cached on `result.cascade_source_counts`
for amortized sources.

### Cascade-aware PPC routing

When `result.frozen_params` is non-empty, posterior predictive samplers
route frozen parameters through `result.cascade_source` instead of
through Laplace post-fit moments. This preserves full SVI guide fidelity
(non-Gaussian guides, low-rank, flows — whatever the SVI fit was) and
sidesteps the NaN sentinels that `compute_global_uncertainty` writes
into `r_scale` / `mu_scale` for frozen entries.

Implementation lives in
[`_resolve_nbln_ppc_arrays`](_sampling.py) (the resolver) and the
`r_samples` / `mu_samples` / `eta_samples` kwargs threaded through
`_ppc_nbln_marginal` / `_ppc_nbln_per_cell_laplace` /
`_ppc_pln_library_anchored`. Per-parameter routing:

| Param | Frozen + cascade | Non-frozen | Coordinate transform applied |
|---|---|---|---|
| `r` | SVI samples `(S, G)` from `cascade_source` | `Normal(r_loc, r_scale)` per draw via `pos_forward` | identity (SVI `r` is already in positive space) |
| `mu` | SVI samples `(S, G)` from `cascade_source` | point `result.mu` (Laplace `mu_scale` is gauge-contaminated and deliberately *not* propagated) | `log` (SVI `mu` is the NB mean; NBLN `mu` is log-rate) |
| `eta` | SVI samples `(S, N)` from `cascade_source.eta_capture` | legacy uniform-pick from `eta_loc[idx]` (marginal) or point `eta_loc` (per-cell) | identity (both already in `[0, ∞)`) |

Two operational details worth knowing:

- **Gene-subsetted results**. `ScribeLaplaceResults.__getitem__` slices
  `mu`/`W`/`d`/`r_loc`/`mu_loc` to the selected gene panel but leaves
  `cascade_source` and `cascade_source_counts` unchanged (the amortizer
  encoder needs the full-gene count matrix). The resolver reads
  `result._subset_gene_index` and slices the gene axis of the cascade
  samples before returning them — so the gene-subsetted PPC works
  out of the box for `scribe.viz.plot_ppc(..., n_genes=K)`.
- **Cascade-pool cap**. `plot_ppc(level="marginal")` inflates
  `n_samples` to `n_eff × n_cells_obs` (frequently 1e6+). Drawing that
  many SVI samples is wasteful (the guide is a fixed posterior
  approximation) and would OOM the GPU for amortized cascades. The
  resolver caps the SVI pool at `_CASCADE_POOL_MAX = 2048` and
  resamples-with-replacement to reach the requested predictive count.
  Statistically equivalent; memory bounded.

For pure-MAP PPCs (no global-uncertainty propagation), use
`get_map_ppc_samples` — it does not consult `cascade_source` at all.

### Gauge-invariant accessors (all Laplace models)

For any model where W lives in absolute log-rate space (PLN, NBLN),
the gauge-invariant cross-gene correlation structure is the
gene-centered projection W_perp = (I − 1·1^T/G) W:

```python
# Returns gene-centered W for PLN/NBLN; W unchanged for LNM/LNMVCP
# (which is already in ALR compositional coordinates).
W_perp = laplace_results.get_W_compositional()

# Quantify how much rank-1 contamination W carries.  For a clean
# Phase-2-frozen fit, gauge_contamination_ratio should be < 0.05.
diag = laplace_results.get_gauge_diagnostics()
# {"W_compositional_norm": ..., "W_all_ones_component_norm": ...,
#  "gauge_contamination_ratio": ...}
```

For biological interpretation of cross-gene correlations (gene-set
analysis, regulatory program identification, co-expression
clustering), **use `get_W_compositional()` rather than the raw `W`**
— it is gauge-invariant by construction. Full theory and
recommendations in
[`paper/_diffexp_nbln_robustness.qmd`](../../../paper/_diffexp_nbln_robustness.qmd).

`get_compositional_samples()` consumes `W_⟂` internally for PLN/NBLN.
Mathematically this is a no-op (softmax kills the rigid-translation
gauge regardless of which `W` enters), but it makes the gauge
invariance manifest in the code and is friendlier to floating-point
precision when the gauge contamination ratio is non-negligible. The
companion plotting entry points `scribe.viz.plot_compositional_ppc`
and `scribe.viz.plot_compositional_corner_ppc` render compositional
PPCs against per-cell empirical and dataset-level pseudobulk
comparators — see [`src/scribe/viz/README.md`](../viz/README.md).

## Phase 3: shrinkage priors on the loadings matrix W

At generous `latent_dim` (e.g. 32), the gauge-invariant singular
value spectrum of `W_⟂` often shows a flat shelf — the model uses
every available latent dimension to fit noise, producing spurious
cross-gene correlations visible in `plot_compositional_corner_ppc`. A
**shrinkage prior on W** lets users keep `latent_dim` generous
and have the prior pick the effective rank adaptively, replacing manual
`latent_dim` capping with a soft selection.

```python
results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    informative_priors_from=svi_results,        # Phase-2 cascade (optional)
    informative_priors_freeze=("r", "eta"),     # Phase-2 freeze (default)
    priors={
        "loadings": {"type": "horseshoe_columnwise", "tau_scale": 1.0},
    },
    latent_dim=16,
    n_steps=20_000,
)
print(results.w_prior_diagnostics["effective_rank"])     # e.g. 3
print(results.w_prior_diagnostics["sigma_k"])            # per-factor scales

# Companion diagnostic plot (singular-value-style elbow).
scribe.viz.plot_w_shrinkage_spectrum(results)
```

The W-prior strategy spec lives inside the canonical ``priors`` dict
under the descriptive key ``"loadings"`` (the factor-analysis term for
``W``).  The internal alias ``"W"`` is also accepted.  The legacy
top-level ``w_prior=`` kwarg still works for backward compatibility
but emits a ``DeprecationWarning``; new code should use the priors
dict form so the API stays uniform across cascade priors, capture
priors, and parameter overrides.

The math (softplus-floor parameterization, (G−1)-dim subspace
correction, std-vs-variance conventions), the calibration workflow,
and the extension roadmap for future row-wise / element-wise /
LNM-family strategies are documented canonically in the paper
section ``paper/_loadings_shrinkage_priors.qmd`` (anchor
``sec-loadings-shrinkage``).

### Strategies registered in v1

All v1 strategies are **column-wise** (per-factor scales). Future
row-wise / element-wise families plug in with one new class and one
registry entry — see `src/scribe/laplace/_w_priors.py`.

| `type` string | Hierarchy | When to use |
|---|---|---|
| `none` (default) | — | No shrinkage. Byte-identical to plain Laplace fit. |
| `gaussian` | `W[:, k] ~ N(0, scale)` | Simple ridge baseline. Tells you if the symptom is "any shrinkage helps" vs "specifically sparsity helps". |
| `horseshoe_columnwise` | `λ_k ~ HalfCauchy(τ)`, `τ ~ HalfCauchy(tau_scale)`, `W[:, k] ~ N(0, λ_k)` | Recommended default for cascade-frozen fits. Kills unused factors cleanly while preserving strong ones. |
| `neg_columnwise` | `ψ_k ~ Exponential(γ)`, `γ ~ Gamma(α, β)`, `W[:, k] ~ N(0, √ψ_k)` | More aggressive near-zero shrinkage than horseshoe. Useful when horseshoe is insufficient to kill noise factors. |

### Design choices baked into the implementation

- **Targets `W_⟂`, not raw `W`.** The obs model gauge-cleans `W` before
  passing to the strategy (`W_for_prior = W − mean(W, axis=0)`) and
  passes `n_constraints=1` so the strategy uses `d_eff = G - 1` in the
  centered-column Gaussian normalizer. Result: the prior targets only
  the biologically meaningful signal, never the all-ones gauge
  component.
- **Softplus-floor reparameterization** on all positive aux scales
  (`λ_k = lambda_min + softplus(raw_λ_k)`). Blocks the
  scale-collapse-to-zero MAP singularity that an unconstrained
  `exp(log_*)` parameterization would have when `W → 0` and aux
  scale `→ 0` simultaneously.
- **Headline rank is `||W_⟂[:, k]||`**, not the aux MAP `σ_k`. The
  column norm directly enters `W_⟂ W_⟂^⊤` and hence the compositional
  covariance; aux scales can be weakly identified under heavy-tailed
  priors.

### Calibration workflow

The W-prior log-density enters `loss_fn` **unscaled** while the
likelihood is `O(N_cells)`. The prior's effective strength therefore
scales inversely with dataset size — re-tune `tau_scale` /
`alpha,beta` when transferring between substantially different
datasets. Rule of thumb: multiply `tau_scale` by `sqrt(N_old / N_new)`
for cross-dataset transfer.

1. Fit with the default (`tau_scale=1.0`) and a generous `latent_dim`.
2. Inspect `results.w_prior_diagnostics["column_norm_effective_rank"]`
   (alias `["effective_rank"]`):
   - `== latent_dim`: no shrinkage — tighten by reducing `tau_scale` 10×.
   - `== 1`: over-shrunk — loosen by 10×.
   - In a reasonable range (`2` – `latent_dim/2`): keep.
3. Inspect `plot_w_shrinkage_spectrum(results)` — a clean fit shows
   a sharp elbow; factors beyond should collapse to near-zero.
4. Sanity-check `results.w_prior_diagnostics["sigma_k"].min()` is
   well above `lambda_min` (default `1e-3`) for active factors. A
   handful touching the floor is expected for dead factors.
5. Re-run the compositional corner PPC — spurious diagonal contours
   should collapse to data-consistent ones.

### Compatibility

- **Combines cleanly with Phase-2 cascade freeze.** The W-prior
  strategy is orthogonal to `informative_priors_freeze` — freeze pins
  `r`/`η`; shrinkage regularizes `W`. The two can be combined freely.
- **LNM-family (LNM / LNMVCP) is not supported in v1.** ALR-space W
  has different shrinkage semantics that need a separate design pass.
  Engine raises `NotImplementedError` for non-`none` configs.
- **`w_prior={"type": "none"}`** is normalized to `None` *before*
  scope validation, so the explicit no-op config is universally
  accepted (useful for parameterized testing).

## Jacobian-corrected MAP (`map_method`)

The Laplace approximation stores the posterior as a Gaussian in
*unconstrained* space (``*_loc``, ``*_scale`` per parameter), then maps
through ``positive_transform`` (Exp or Softplus) at result-construction
time to produce the constrained stored fields (``self.r``,
``self.gene_mean``, ``self.mu_T``, ``self.r_T``, etc.).

By default, those stored fields are ``transform(loc)`` — the **median**
of the constrained-space posterior for monotone transforms, NOT the
mode (MAP). To get the true constrained-space mode, you need the
**Jacobian correction**: for ``X ~ N(\\mu, \\sigma^2)``, ``Y = e^X`` has
mode ``e^{\\mu - \\sigma^2}``, not ``e^{\\mu}``.

### `get_map(map_method=...)`

```python
result = scribe.fit(counts, ...)

# Default: Jacobian-corrected MAP. New in v1; produces ~exp(sigma^2)-
# shifted values compared to legacy.
m = result.get_map()

# Legacy: f(loc) (=median); reproduces pre-correction byte-for-byte.
m_legacy = result.get_map(map_method="transform")

# Strict: raise on unsupported (transform, base) pairs.
m_strict = result.get_map(map_method="jacobian")
```

The method controls how the **returned dict** computes constrained
values from the persisted ``(_loc, _scale)`` pairs. It does NOT mutate
the stored fields — ``result.r`` still has the median value unless you
call ``with_jacobian_map()`` (see below).

### `with_jacobian_map()` — opt-in persistent correction

If downstream code reads ``self.r`` / ``self.gene_mean`` directly (e.g.,
PPC samplers, cascade adapters), use ``with_jacobian_map()`` to
materialize the corrected view into the stored fields:

```python
new_result = result.with_jacobian_map()
# new_result.r is now the corrected MAP.
# For TSLN-Rate, new_result.alpha / .beta / .r_hat are also re-derived
# from the corrected (gene_mean, burst_size, k_off) via _twostate_reparam.
# For LNM, the derived p is recomputed from the corrected (mu_T, r_T)
# inside get_map().
```

The ``_loc`` / ``_scale`` pairs are NOT modified, so
``new_result.get_map(map_method="transform")`` still derives from
``_loc`` and returns the uncorrected median. The chosen semantics:
``"transform"`` always means "raw transform-of-loc", regardless of
what's in the stored view fields.

### Cascade reproducibility (`cascade_map_method`)

When ``scribe.fit(..., informative_priors_from=svi_results)`` is used,
the cascade extractors (``freeze_values_from_results``,
``freeze_values_from_twostate_results``) call
``svi_results.get_map()`` internally to extract MAP values from the SVI
source. The SVI default flipped to ``map_method="auto"`` (Jacobian-
corrected MAP) in v1, which means:

* Pre-v1 cascade fits used ``transform(loc)`` median values.
* Post-v1 cascade fits use ``exp(\\mu - \\sigma^2)``-shifted values by
  default.

For reproducibility of pre-v1 cascade results, pass
``cascade_map_method="transform"`` to ``scribe.fit``:

```python
# Reproduce pre-correction cascade behavior.
result = scribe.fit(
    counts,
    informative_priors_from=svi_results,
    cascade_map_method="transform",   # pins legacy uncorrected median
    ...
)

# Default (post-v1): cascade uses Jacobian-corrected MAP.
result = scribe.fit(
    counts,
    informative_priors_from=svi_results,
    ...
)
```

### Limitations (v1)

* **`p_capture` cannot be corrected**: requires persisting
  ``eta_scale``, which is not yet wired into ``pack_result``. Calling
  ``get_map(map_method="auto")`` on a result with ``eta_loc`` emits a
  one-per-call-site warning; ``map_method="jacobian"`` raises.
* **`LowRankMVN + Sigmoid/Softplus`** requires a coupled multi-start
  optimizer not in v1. Falls back to ``transform(loc)`` with a warning
  under ``"auto"``; raises under ``"jacobian"``. In practice this only
  matters for low-rank guides on probability or positive parameters —
  most production fits use independent Normal bases for these.
* **Heuristic sigma ceiling**: the grid+Newton refinement for
  Sigmoid/Softplus is reliable for ``sigma < SIGMA_CEILING_WARN``
  (default 10). Beyond that, the adaptive grid may not cover
  asymptotic modes; the public wrapper warns under ``"auto"`` and
  raises under ``"jacobian"``.

### See also

* [`scribe/stats/jacobian_map.py`](../stats/jacobian_map.py) for the
  math and dispatch table.
* [`scribe/laplace/_derived.py`](_derived.py) — derivation helpers
  used by ``with_jacobian_map`` to recompute TSLN ``(\\alpha, \\beta,
  r_{\\hat{}})`` and LNM ``p``.

## See also

* [`paper/_poisson_lognormal.qmd`](../../../paper/_poisson_lognormal.qmd)
  — full PLN Laplace derivation including the Woodbury construction.
* [`paper/_logistic_normal_multinomial.qmd`](../../../paper/_logistic_normal_multinomial.qmd)
  — LNM(VCP) Laplace derivation, gauge-fix argument, block-diagonal
  Hessian.
* [`scribe.models.config.LaplaceConfig`](../models/config/groups.py) —
  full list of configurable knobs.
