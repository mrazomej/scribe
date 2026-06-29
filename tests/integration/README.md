# tests/integration

**Purpose.** Cross-cutting, end-to-end tests that span several subsystems and
have no single "home" module. If a test exercises one subsystem it belongs in
that subsystem's folder; this folder is for the genuinely system-wide ones.

**Source under test.** Multiple subpackages in combination (model build →
inference → results → DE/viz), reached through the public API.

**What lives here.**
- `test_multi_dataset` — multi-dataset modeling end-to-end (core axes + de + mcmc + builders + svi).
- `test_hierarchical` — hierarchical-model fitting across the stack.
- `test_multifactor_recover` — synthesize a crossed design → fit → recover a known effect.
- `test_vae_integration` — the full VAE pipeline (flows + components + sampling + svi).
- `test_map_method_integration` — the MAP/Laplace cascade through `fit()`.
- `test_x64_precision` — x64-precision propagation through CLI + `fit()` + inference (uses the `project_root` fixture for `conf/` lookups).
- `test_results_serialization` — round-trip serialization of *all* results classes together.
- `test_optional_dependency_boundaries` — the core import path stays usable when optional deps are absent.

**What does NOT live here.**
- Subsystem-local end-to-end tests (e.g. `test_de_integration`) → their own subsystem folder (`tests/de/`).

**Key fixtures.** Root `tests/conftest.py` (`data_dir`, `project_root`, device/method options). No folder-local conftest.
