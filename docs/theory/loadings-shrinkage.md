# Loadings-Matrix Shrinkage Priors

The low-rank covariance parameterization
\(\underline{\underline{\Sigma}} = \underline{\underline{W}}\,
\underline{\underline{W}}^\top + \text{diag}(\underline{d})\)
underlies all the [GRN-based observation
models](grn-biophysics.md) — PLN, LNM/LNMVCP, and
[NBLN](nb-lognormal.md). The latent dimension \(k\) controls how many
regulatory programs the loadings matrix can represent. Too small
loses real signal; too large lets the model fit noise. **Loadings
shrinkage priors** are an adaptive alternative: they let users keep
\(k\) generous and let the data select the effective rank.

---

## Motivation

At generous \(k\) (say \(k=32\)) with no shrinkage on
\(\underline{\underline{W}}\), the gauge-invariant singular-value
spectrum of \(\underline{\underline{W}}_\perp\) typically shows a
**flat shelf**: the dominant 2--3 factors carry real biological
signal, but the remaining \(k-3\) factors sit at \(\sim\)20% of the
dominant factor's energy each. This excess capacity paints **spurious
cross-block diagonal contours** in compositional corner PPCs — model
density extends in directions the empirical scatter does not support.

Manually capping \(k\) fixes the symptom but is not portable across
datasets: different cell-type panels and assays support different
effective ranks. A shrinkage prior on the columns of
\(\underline{\underline{W}}\) is the principled fix — it pays no
penalty for keeping strong factors at their natural scale and
penalizes unused factors heavily, producing an approximately
rank-sparse MAP.

---

## Where the prior acts: \(\underline{\underline{W}}_\perp\), not raw \(\underline{\underline{W}}\)

For PLN and NBLN, raw \(\underline{\underline{W}}\) carries a rank-1
**all-ones-direction gauge contamination** that reflects cell-scaling
slop rather than biology
(see [Theorem 2 in the NBLN robustness section](nb-lognormal.md)).
Shrinking raw \(\underline{\underline{W}}\) would dissipate prior
mass on this gauge component as well as the biological one.

The obs-model integration layer therefore projects
\(\underline{\underline{W}} \to \underline{\underline{W}}_\perp =
\underline{\underline{W}} - \overline{\underline{\underline{W}}}\) at
the strategy boundary. The strategy itself stays model-agnostic — it
accepts whatever loadings matrix is handed in and uses an
`n_constraints` flag to scale the Gaussian normalizer correctly. For
LNM-family models (where \(\underline{\underline{W}}\) already lives
in ALR-quotient coordinates with no all-ones-gauge), the obs model
would pass raw \(\underline{\underline{W}}\) — same strategy code,
different boundary treatment.

---

## The three v1 strategies

All v1 strategies are **column-wise** — per-factor scales. Each
strategy registers with a `type_name` and accepts a small set of
hyperparameters.

### Gaussian (ridge) — `"gaussian"`

Single shared scale:
\(\underline{\underline{W}}_\perp[:, k] \sim
\mathcal{N}(\underline{0}, \sigma^2 \underline{\underline{I}}_G)\)
for all \(k\). No auxiliary parameters. Use as a sanity-check
baseline — if Gaussian gives the right answer, you don't need the
heavier machinery.

### Column-wise horseshoe — `"horseshoe_columnwise"` (recommended default)

Local-global hierarchy with standard-deviation local scales:

\[
\underline{\underline{W}}_\perp[:, k] \mid \lambda_k \sim
\mathcal{N}(\underline{0}, \lambda_k \underline{\underline{I}}_G),
\quad
\lambda_k \mid \tau \sim \mathrm{HalfCauchy}(\tau),
\quad
\tau \sim \mathrm{HalfCauchy}(\tau_0).
\]

The heavy-tailed local scales \(\lambda_k\) let strong factors
through; the global scale \(\tau\) pulls unused factors toward zero.
Recommended default for cascade-frozen [NBLN](nb-lognormal.md) fits.

### Column-wise NEG — `"neg_columnwise"`

Normal-Exponential-Gamma hierarchy with variance local scales:

\[
\underline{\underline{W}}_\perp[:, k] \mid \psi_k \sim
\mathcal{N}(\underline{0}, \sqrt{\psi_k}\, \underline{\underline{I}}_G),
\quad
\psi_k \mid \gamma \sim \mathrm{Exponential}(\gamma),
\quad
\gamma \sim \mathrm{Gamma}(\alpha, \beta).
\]

More aggressive near-zero shrinkage than horseshoe at default
hyperparameters. Use when horseshoe is insufficient to kill noise
factors on a particular dataset.

---

## Three implementation details that matter

These three points have to be right or the prior runs but silently
produces a wrong fit.

### Standard deviation vs variance

The NumPyro `Normal(loc, scale)` parameterization uses **scale =
standard deviation**. Horseshoe writes
\(\mathcal{N}(\underline{0}, \lambda_k)\) where \(\lambda_k\) is
explicitly std; NEG writes \(\mathcal{N}(\underline{0},
\sqrt{\psi_k})\) where \(\psi_k\) is explicitly variance. Mixing
these up produces a fit that runs but converges to the wrong scale
by a factor of two in log-units. Unit tests guard the distinction.

### Softplus-floor reparameterization

A naive log-space parameterization
\(\lambda_k = \exp(\log \lambda_k)\) combined with the Gaussian
likelihood produces an **unbounded above** log-prior as
\(\lambda_k \to 0\). The negative log-likelihood loss diverges to
\(-\infty\) along the ridge
\(\underline{\underline{W}}_\perp \to \underline{0},\, \lambda_k \to 0\)
— the optimizer collapses to that degenerate MAP in a few steps.

The fix is the **softplus-floor reparameterization**:
\(\lambda_k = \lambda_{\min} + \mathrm{softplus}(\mathrm{raw}_{\lambda_k})\)
with default \(\lambda_{\min} = 10^{-3}\). The log-Jacobian
\(\log \sigma(\mathrm{raw}_{\lambda_k})\) is bounded above by 0 and
tends to \(-\infty\) as \(\mathrm{raw}_{\lambda_k} \to -\infty\),
actively preventing the optimizer from collapsing the scale.

### Subspace correction

When the obs model passes \(\underline{\underline{W}}_\perp\) to the
strategy (as it does for PLN/NBLN), each column satisfies
\(\sum_g W_{\perp,gk} = 0\) — each column lives in a
\((G-1)\)-dimensional subspace, not all of \(\mathbb{R}^G\). The
centered-column Gaussian normalizer must use
\(d_{\mathrm{eff}} = G - 1\), not \(G\). Naively calling
`dist.Normal(0, λ_k).log_prob(W_perp).sum()` over-counts the
normalizer by \(-\log \lambda_k\) per column and biases the optimal
scale by \(\sqrt{G/(G-1)}\). Writing the centered-column density
manually with the correct \(d_{\mathrm{eff}}\) keeps the math right
at any data size.

---

## Diagnostics: column norms, not aux scales

The strategy's `diagnostics` method reports both a primary
**column-norm-based** rank and a secondary scale-based rank. The
headline anchors on the column norm because **only the column norm
directly enters \(\underline{\underline{W}}_\perp\,
\underline{\underline{W}}_\perp^\top\)** and hence the compositional
covariance that downstream correlation analyses visualize.

| Key | What it is |
|---|---|
| `column_frobenius_compositional` | \(\|\underline{\underline{W}}_\perp[:, k]\|\) per factor (data-supported) |
| `column_norm_effective_rank` | # factors with column norm > 5% of max — **headline** |
| `effective_rank` | alias for `column_norm_effective_rank` |
| `sigma_k` | per-column aux MAP scales (\(\lambda_k\) for horseshoe, \(\sqrt{\psi_k}\) for NEG) |
| `scale_effective_rank` | same threshold applied to `sigma_k` (secondary) |

Aux scales can be weakly identified under heavy-tailed priors — a
horseshoe fit can drive \(\lambda_k\) to its floor while the data
still supports a small non-zero column norm. Reporting both
diagnostics surfaces this disagreement when it happens.

The companion plot
[`plot_w_shrinkage_spectrum`](../guide/results.md) renders
\(\|\underline{\underline{W}}_\perp[:, k]\|\) on a log-scale primary
axis with `sigma_k` as an optional dashed-secondary overlay; the
5% threshold for `column_norm_effective_rank` is drawn as a
horizontal line.

---

## API: priors dict with the `"loadings"` key

The W-prior strategy spec lives inside the canonical `priors` dict
under the descriptive key `"loadings"` (the factor-analysis term for
\(\underline{\underline{W}}\)) — alongside other prior overrides:

```python
import scribe, numpy as np

results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    informative_priors_from=svi_results,
    informative_priors_freeze=("r", "eta"),
    priors={
        "capture_efficiency": (np.log(100_000), 0.5),
        "loadings": {
            "type": "horseshoe_columnwise",
            "tau_scale": 1.0,
        },
    },
    latent_dim=16,
    n_steps=20_000,
)
```

Available types: `"none"` (no-op default), `"gaussian"`,
`"horseshoe_columnwise"`, `"neg_columnwise"`. The loadings shrinkage is
configured exclusively through `priors={"loadings": ...}` — there is no
top-level `w_prior=` kwarg.

---

## Calibration workflow

The W-prior log-density enters the loss **unscaled** while the
likelihood scales as \(O(N_{\text{cells}})\). The prior's effective
strength therefore scales inversely with dataset size — the same
\(\tau_0\) that produces a clean elbow on a 10k-cell dataset may
produce no visible shrinkage on a 100k-cell dataset. Rule of thumb
for cross-dataset transfer: multiply \(\tau_0\) by
\(\sqrt{N_{\text{old}} / N_{\text{new}}}\).

The 5-step recipe:

1. **Fit with the default** (`tau_scale=1.0`) and a generous
   `latent_dim` (e.g. 16).
2. **Inspect `column_norm_effective_rank`** (alias `effective_rank`):
    - Equals `latent_dim`: no shrinkage — tighten by reducing
      `tau_scale` 10x.
    - Equals 1: over-shrunk — loosen 10x.
    - In `[2, latent_dim/2]`: keep.
3. **Inspect the spectrum plot** via
   `plot_w_shrinkage_spectrum`. A clean fit shows a sharp elbow at
   the effective rank.
4. **Sanity-check the aux-scale floor.** Confirm
   `sigma_k.min()` is well above `lambda_min` for active factors. A
   handful of dead factors touching the floor is expected.
5. **Re-run the compositional corner PPC.** Spurious diagonal
   contours should collapse to data-consistent ones.

---

## Gauge-contamination diagnostic in the shrinkage regime

The `get_gauge_diagnostics()` method on `ScribeLaplaceResults` returns
three numbers — `W_compositional_norm` (\(\|W_\perp\|\)),
`W_all_ones_component_norm` (\(\|W_\parallel\|\)), and their ratio.
For **unshrunk** fits the ratio is the headline diagnostic with
clear thresholds (< 0.05 clean, > 0.2 trouble). For **shrunk** fits
the ratio means something qualitatively different and needs a
different reading.

The shrinkage prior targets `W_⟂` aggressively and leaves
`W_∥` unconstrained — the cascade freeze on \(\eta\) is the
gauge-pinning mechanism, not a ridge on the loadings gauge
component. With the prior in place, `W_⟂` shrinks rapidly to
match the data-supported rank, while `W_∥` only shrinks via the
implicit constraint from the likelihood + frozen \(\eta\). The
ratio therefore climbs **as a consequence of `W_⟂` shrinking**,
not because `W_∥` is growing.

On real cascade-frozen NBLN fits with horseshoe or NEG at default
hyperparameters, ratios of 0.5–0.8 on clean fits are routine. The
diagnostic to inspect in this regime is the **absolute norms**:

| Pattern | Reading |
|---|---|
| Both norms modest; ratio 0.5–0.8 differs across shrinkage prior families | **Healthy.** The shrinkage has done its job on `W_⟂`. The ratio differing across NEG vs horseshoe is the expected signature of the unidentified all-ones direction — different priors put it in different places. |
| Both norms modest; ratio similar across prior families | Healthy and the gauge component is well-determined by the cascade freeze. |
| Ratio ≫ 1 *and* both norms large in absolute terms | Concerning — the original failure mode (gauge component carrying real signal). Rare when cascade freeze is active. |

The rank convergence between NEG and horseshoe — both selecting the
same effective rank on the same data despite landing at different
`W_∥` magnitudes — is itself the strongest evidence that the
all-ones direction is data-unidentified and that the biological
signal lives entirely in `W_⟂`.

**Bottom line for downstream analyses:** for compositional PPCs,
cross-gene correlations via `get_W_compositional()`, and any
quantity that's gauge-invariant by Theorem 1 or 2, the ratio is
irrelevant. The only analyses sensitive to the all-ones component
are those that interpret raw `W` or per-cell `x_loc` as absolute
log-rates, and those weren't meaningful under the per-cell gauge
to begin with.

---

## Compatibility and scope (v1)

- **PLN and NBLN Laplace fits** are supported. The engine raises
  `NotImplementedError` for `model="lnm"` / `"lnmvcp"` with a
  non-`"none"` `loadings` config — ALR-space \(\underline{\underline{W}}\)
  has different shrinkage semantics that need a separate design pass.
- **Orthogonal to Phase-2 cascade freeze.** The shrinkage strategy
  is independent of `informative_priors_freeze` — freeze pins
  \(r\) and \(\eta\) at cascade values; shrinkage regularizes
  \(\underline{\underline{W}}\). Both mechanisms run together by
  default for cascade-frozen NBLN fits.
- **Gauge component left unregularized by the prior.** Shrinkage
  targets only \(\underline{\underline{W}}_\perp\); the all-ones
  component of raw \(\underline{\underline{W}}\) is pinned
  structurally by the cascade freeze on \(\eta\), not by a ridge in
  the W-prior. Monitor drift via `get_gauge_diagnostics()`.

---

## When to use loadings shrinkage

| Use loadings shrinkage when…                                                 | Skip it when…                                                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `latent_dim` is generous and you want adaptive rank selection                | You've already manually tuned `latent_dim` to a small value (e.g. 3-5)         |
| The compositional corner PPC shows spurious cross-block diagonals            | The compositional corner PPC already looks clean                               |
| You want a portable workflow that doesn't need per-dataset `latent_dim` tuning | The dataset is well-characterized and `latent_dim` transfers across replicates |
| The singular-value spectrum of \(\underline{\underline{W}}_\perp\) has no elbow | The spectrum already has a sharp elbow without shrinkage                       |

!!! tip "Recommended default for NBLN cascade fits"
    Use `priors={"loadings": {"type": "horseshoe_columnwise",
    "tau_scale": 1.0}}` with `latent_dim=16` or `32`. Horseshoe's
    combination of a sharp peak at zero and heavy tails kills unused
    factors cleanly while preserving strong ones. If horseshoe is
    insufficient, switch to `neg_columnwise` for more aggressive
    near-zero shrinkage.
