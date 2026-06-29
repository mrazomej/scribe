# tests/inference

**Purpose.** The inference engines and their plumbing — SVI and MCMC run paths,
preset/config building, checkpointing and early stopping, KL annealing,
SVI→MCMC initialization, and the variational/MCMC/VAE results classes.

**Source under test.** `src/scribe/inference`, `src/scribe/svi`, and
`src/scribe/mcmc` (run_scribe, `preset_builder`, `inference_config`,
`optimizer_factory`, `svi.inference_engine`, `svi.checkpoint`, `svi.kl_annealing`,
`svi._component` / `_latent_dispatch` / `_sampling` mixins, `svi.results`,
`svi.vae_results`, `mcmc.inference_engine`, `mcmc._init_from_svi`, `mcmc.results`).

**What lives here.**
- `test_run_scribe`, `test_inference_config`, `test_preset_builder`, `test_utils` — the run entry point + config/preset construction (formerly `tests/test_inference/`).
- `test_early_stopping` — SVI checkpointing + early-stopping behavior.
- `test_kl_annealing` — KL-annealing schedules.
- `test_svi_mcmc_init` — SVI→MCMC initialization (`compute_init_values`, clamping).
- `test_mcmc_results`, `test_vae_results` — the MCMC and VAE results classes.
- `test_sampling_mixin_compat`, `test_latent_dispatch`, `test_component_pruning_subset` — SVI mixin composition, latent dispatch, component pruning.

**What does NOT live here.**
- Per-model end-to-end recovery fits (`test_nbdm`, `test_zinb`, …) → `tests/models/families/`.
- The `fit()` orchestration stages that call these engines → `tests/api/`.
- VAE *components* → `tests/models/components/`; full VAE pipeline → `tests/integration/`.

**Key fixtures.** Root `tests/conftest.py` **plus a folder-local `conftest.py`**
(carried over from the old `tests/test_inference/`) providing inference-specific
fixtures (`sample_counts`, `nbdm_model_config`, `zinb_model_config`, …) and its
own `pytest_addoption`/`pytest_configure`/`pytest_collection_modifyitems`.
