# tests/laplace

**Purpose.** The Laplace-approximation inference path: the Newton kernels,
observation models, global-uncertainty / curvature math, frozen-prior cascades,
and the per-model Laplace specializations (PLN, LNM, NBLN, two-state).

**Source under test.** `src/scribe/laplace` (`_newton_*`, `_obs_*`,
`_global_uncertainty`, `_em`, `_dispatch`, `_derived`, `_sampling`,
`_axis_layout`, `_w_priors`, `priors`, `results`) and `inference.laplace`
(`_run_laplace_inference`).

**What lives here.**
- Generic kernels/priors: `test_laplace_newton`, `test_laplace_newton_lnm`, `test_laplace_priors`, `test_laplace_priors_subset_cascade`, `test_laplace_rescue`, `test_w_priors`, `test_decorrelate_other_column_scaffolding`.
- Per-model Laplace: `test_pln_laplace`, `test_lnm_laplace`, `test_nbln_laplace`.
- Two-state rate gauge: `test_twostate_ln_rate_{cascade,gauge,global_curvature,newton,public_api,recovery}`.
- Two-state logit gauge: `test_twostate_ln_logit_{cascade,gauge,global_curvature,newton,obs_model,public_api,recovery}`.
- `bench_twostate_ln_rate.py` — a **benchmark script**, not a test (the `bench_` prefix means pytest does not collect it).

**What does NOT live here.**
- Two-state / LNM / PLN model construction & likelihood → `tests/models/`.
- Two-state data-initialization → `tests/core/`.
- The MAP method exercised end-to-end through `fit()` → `tests/integration/` (`test_map_method_integration`).
- Jacobi quadrature / Jacobian-MAP math primitives → `tests/stats/`.

**Key fixtures.** Root `tests/conftest.py`. The shared synthetic NBLN result
factory is in `tests/_synthetic_results.py` (imported as `from _synthetic_results
import _nbln_result`). No folder-local conftest.
