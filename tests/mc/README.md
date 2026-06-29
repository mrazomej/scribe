# tests/mc

**Purpose.** Model-comparison machinery: WAIC, PSIS-LOO, stacking, and
goodness-of-fit diagnostics.

**Source under test.** `src/scribe/mc` (`_waic`, `_psis_loo`, `_stacking`,
`_goodness_of_fit`, `_gene_level`, `results`).

**What lives here.**
- `test_mc_waic` — lppd, p_waic_1/2, WAIC against references.
- `test_mc_psis_loo` — Pareto-tail fitting, single-observation smoothing.
- `test_mc_goodness_of_fit` — randomized quantile-residual calibration.
- `test_mc_results` — the `ScribeModelComparisonResults` class (waic/psis_loo/rank/summary, stacking, gene-level tables).

**What does NOT live here.**
- Serialization round-trips of the results class → `tests/integration/` (`test_results_serialization`).
- The per-draw log-likelihood computation it builds on → `tests/inference/` (svi/mcmc likelihood mixins).

**Key fixtures.** Root `tests/conftest.py`. No folder-local conftest.
