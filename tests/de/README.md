# tests/de

**Purpose.** Differential-expression tests: the empirical, shrinkage, and
biological DE estimators, compositional transforms, gene-set / pathway-level
contrasts, and the DE results class.

**Source under test.** `src/scribe/de` (`_empirical`, `_shrinkage`, `_biological`,
`_set_level`, `_transforms`, `_factors`, `_results_factory`, `_extract`,
`_lnm_diagnostics`, `_component_matching`, `results`, and the public `compare` /
`compare_datasets` / `extract_alr_params` / `build_balance_contrast` entry points).

**What lives here.**
- `test_de_empirical`, `test_de_reference` — empirical (sample-based) DE + CLR reference handling.
- `test_de_shrinkage` — shrinkage DE estimator.
- `test_de_biological` — biological (denoised) LFC / dispersion-shift DE.
- `test_de_set_level`, `test_de_transformations` — gene-set/pathway contrasts and ILR/ALR/CLR transforms.
- `test_de_factors`, `test_de_leaf_addressing` — pair resolution, group comparisons, leaf indexing.
- `test_de_gene_level`, `test_de_extract`, `test_de_utils`, `test_de_nbln_tsln_compare` — gene-level DE + ALR-param extraction utilities.
- `test_de_results` — the `ScribeEmpiricalDEResults` / `ScribeShrinkageDEResults` classes.
- `test_de_integration` — end-to-end DE pipeline (model creation → compare).
- `test_lnm_sampling`, `test_lnm_diagnostics` — LNM sampling/diagnostic helpers that live in `de._empirical` / `de._lnm_diagnostics`.

**What does NOT live here.**
- Low-rank Gaussian *divergences* (`stats.divergences`) → `tests/stats/` (`test_de_divergences`, despite the name, imports only `scribe.stats`).
- LNM model construction / likelihood / parameterization → `tests/models/`.
- LNM **Laplace** inference → `tests/laplace/`.
- Multi-dataset DE comparisons exercised end-to-end → `tests/integration/` (`test_multi_dataset`).

**Key fixtures.** Inherits the root `tests/conftest.py` fixtures (`rng_key`,
`small_dataset`, and the `--device` / `--method` / `--parameterization` options).
No folder-local conftest.
