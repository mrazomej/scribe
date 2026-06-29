# tests/core

**Purpose.** The foundational metadata and data-preparation layer that the rest
of the library builds on: the `AxisLayout` system, factor/group views,
gene-coverage/subsetting, vectorized normalization, and data-driven parameter
initialization.

**Source under test.** `src/scribe/core` (`axis_layout`, `factor_effect_view`,
`grouping_view`, `gene_coverage`, `normalization`, `normalization_logistic`,
`lnm_data_init`, `pln_data_init`, `twostate_data_init`, `annotation_prior`).

**What lives here.**
- `test_axis_layout`, `test_factor_layout` — the `AxisLayout` metadata system and factor-axis membership.
- `test_factor_effect_view`, `test_grouping_view` — per-factor effect and group-leaf views.
- `test_gene_coverage`, `test_gene_subsetting` — gene-coverage/ALR gating and metadata-based gene subsetting.
- `test_normalization_vectorized` — batched Dirichlet / logistic-normal normalization.
- `test_lnm_stability`, `test_pln_stability`, `test_twostate_data_init` — data-driven initialization for LNM/PLN/two-state.
- `test_annotation_prior` — annotation-prior resolution and component mapping.

**What does NOT live here.**
- Model-construction *uses* of `AxisLayout` (building parameter layouts) → `tests/models/`.
- Grouping *configuration* (`models.config.grouping`) → `tests/models/config/`.
- The inference that *consumes* the init values → `tests/inference/`.

**Key fixtures.** Root `tests/conftest.py` (`rng_key`, `small_dataset`, device options). No folder-local conftest.
