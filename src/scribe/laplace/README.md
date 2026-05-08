# SCRIBE Laplace inference

This directory contains scribe's Laplace-approximation inference path. Unlike
the SVI / VAE path, Laplace inference has no encoder: each cell's per-cell
posterior is found locally via Newton iteration on the log-density, and the
Gaussian approximation is centred at the resulting MAP with covariance
$-H^{-1}$ from the Hessian curvature. The outer loop optimises global
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
gradient norm. If it exceeds `newton_tolerance`, behaviour is controlled
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
│   │        shrink as globals stabilise.
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

## See also

* [`paper/_poisson_lognormal.qmd`](../../../paper/_poisson_lognormal.qmd)
  — full PLN Laplace derivation including the Woodbury construction.
* [`paper/_logistic_normal_multinomial.qmd`](../../../paper/_logistic_normal_multinomial.qmd)
  — LNM(VCP) Laplace derivation, gauge-fix argument, block-diagonal
  Hessian.
* [`scribe.models.config.LaplaceConfig`](../models/config/groups.py) —
  full list of configurable knobs.
