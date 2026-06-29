# tests/models/config

**Purpose.** Model configuration: `ModelConfig` / `ModelConfigBuilder`, the model
registry, grouping config, the unified-priors contract, and parameter mapping.

**Source under test.** `src/scribe/models/config` (`base`, `builder`, `enums`,
`groups`, `grouping`, `parameter_mapping`) plus `models.model_registry` and
`scribe.utils.parameter_collector`.

**What lives here.**
- `test_model_config` — `ModelConfig` validation and construction.
- `test_model_registry` — `get_model_and_guide` retrieval across model types / parameterizations / methods.
- `test_grouping` — `GroupLevel` / grouping normalization in `models.config.grouping`.
- `test_unified_priors` — `normalize_unified_priors` value forms and rejections.
- `test_parameter_mapping`, `test_parameter_collector` — parameterization↔parameter mapping and the `ParameterCollector` utility.
- `test_positive_transform_dict` — the `ModelConfig.positive_transform` field forms.

**What does NOT live here.**
- Grouping *views* (`core.grouping_view`, `core.factor_effect_view`) → `tests/core/`.
- The data-processing pipeline stage that consumes grouping → `tests/api/` (`test_grouping_pipeline`).
- Parameterization *classes* and derived params → `../parameterizations/`.

**Key fixtures.** Root `tests/conftest.py`. No folder-local conftest.
