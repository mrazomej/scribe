# tests/api

**Purpose.** The `fit()` orchestration layer — the pipeline stages that turn a
user request into a configured model and dispatched inference, plus
model-string / feature-flag resolution.

**Source under test.** `src/scribe/api` (`fit`, `context.FitContext`,
`stages/*` — `data_processing`, `model_config_build`, `model_flags`, ...).

**What lives here.**
- `test_fit_feature_flags` — model-string resolution behavior (unit-level, no fit).
- `test_grouping_pipeline` — the `process_data_and_datasets` data-processing stage (leaf grouping, no inference).
- `test_hierarchy_exp_transform` — the `model_config_build` stage's expression-hierarchy transform.

**What does NOT live here.**
- The inference *engines* the orchestration dispatches to → `tests/inference/`.
- Per-model end-to-end recovery fits → `tests/models/families/`.
- Raw model/guide construction → `tests/models/builders/`.

**Key fixtures.** Root `tests/conftest.py`. No folder-local conftest.
