# tests/cli

**Purpose.** The command-line interface and surrounding runtime concerns: the
`infer` / `visualize` / `initialize` commands, Hydra integration, SLURM job
generation, data ingestion, and logging setup.

**Source under test.** `src/scribe/cli` (`infer`, `infer_runner`, `dispatch`,
`initialize`, `visualize`, `slurm*`, `hydra_callbacks`, `output_layout`,
`split_orchestrator`) plus the top-level `scribe.catalog`, `scribe.data_loader`,
`scribe._logging`, and the repo-root `slurm_infer.py` script.

**What lives here.**
- Infer CLI: `test_scribe_infer_cli`, `test_scribe_infer_initialize`, `test_infer_cli_downgrade`, `test_infer_min_cells_guard`, `test_infer_override_dirname`, `test_infer_split`, `test_cli_legacy_prior_fold`.
- Visualize CLI: `test_scribe_visualize_cli`.
- Hydra / output layout: `test_hydra_output_prefix`, `test_output_layout`.
- SLURM: `test_slurm_infer` (imports the repo-root `slurm_infer.py`).
- Data ingestion: `test_catalog` (`scribe.catalog`), `test_data_loader` (`scribe.data_loader`).
- Logging: `test_logging` (`scribe._logging.setup_logging`).

**What does NOT live here.**
- The inference *engines* the CLI ultimately invokes → `tests/inference/`.
- The `fit()` orchestration stages → `tests/api/`.

**Key fixtures.** Root `tests/conftest.py` (`scribe_caplog` is used by `test_logging`;
`project_root` anchors repo-relative `conf/` lookups). No folder-local conftest.
