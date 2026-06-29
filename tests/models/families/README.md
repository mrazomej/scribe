# tests/models/families

**Purpose.** Per-model behavior — the end-to-end "does this model fit and recover
its parameters" suites, plus the two-state model's non-Laplace behavior. These
are named after a *model* rather than a single source module, so they have no
1:1 src counterpart.

**Source under test.** The full model stack for each family (likelihood + guide +
inference path), reached via `run_scribe` / `create_model` /
`build_config_from_preset`.

**What lives here.**
- Count families: `test_nbdm`(+`_mix`), `test_zinb`(+`_mix`), `test_nbvcp`(+`_mix`), `test_zinbvcp`(+`_mix`), `test_bnb`.
- Log-normal families: `test_nbln_factory`, `test_lnm_factory`, `test_pln_factory`.
- Two-state (non-Laplace): `test_twostate_mean_fano`, `test_twostate_moment_delta`, `test_twostate_ratio`, `test_twostate_n_quad_nodes`, `test_twostate_public_api`, `test_twostate_mixture`, `test_twostate_regime_prior_override`.

**What does NOT live here.**
- Likelihood `log_prob` unit tests → `../likelihoods/`.
- Two-state **Laplace** inference (`ln_rate` / `ln_logit`) → `tests/laplace/`.
- Generic inference-engine behavior (early stopping, KL annealing, SVI↔MCMC init) → `tests/inference/`.

**Key fixtures.** Root `tests/conftest.py` — note the `--method` / `--parameterization` / `--unconstrained` / `--guide-rank` options heavily parametrize these suites. No folder-local conftest.
