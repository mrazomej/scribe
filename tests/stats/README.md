# tests/stats

**Purpose.** Probability-distribution objects, divergences, numerical
quadrature, and the Jacobian-corrected MAP math.

**Source under test.** `src/scribe/stats` (`distributions`, `divergences`,
`quadrature`, `_jacobi_quad`, `jacobian_map`).

**What lives here.**
- `test_stats` — general statistics helpers.
- `test_poisson_beta_compound` — the `PoissonBetaCompound` distribution.
- `test_jacobi_quad` — Gauss-Legendre / Jacobi quadrature nodes & weights.
- `test_jacobian_map` — `jacobian_corrected_map` math contract (SIGMA ceiling, sliced transforms).
- `test_de_divergences` — low-rank Gaussian divergences. **Named `de_` for historical reasons but imports only `scribe.stats.divergences` / `scribe.stats.distributions`** (no `scribe.de`); rename candidate `test_divergences` (deferred).

**What does NOT live here.**
- Differential-expression estimators that *use* these divergences → `tests/de/`.
- The MAP method exercised end-to-end through `fit()` → `tests/integration/` (`test_map_method_integration`).

**Key fixtures.** Root `tests/conftest.py`. No folder-local conftest.
