# tests/models/parameterizations

**Purpose.** Parameterization classes and their derived-parameter contracts
(e.g. canonical / mean_prob / mean_disp variants and the maps between them).

**Source under test.** `src/scribe/models/parameterizations` (and the
`PARAMETERIZATIONS` registry).

**What lives here.**
- `test_parameterizations` — the parameterization classes and derived-parameter resolution.
- `test_lnm_parameterization`, `test_pln_parameterization` — the LNM and PLN family variants.
- `test_mean_disp` — the `mean_disp` parameterization (direct `mu`/`r` sampling) across the init/sampling path.

**What does NOT live here.**
- Parameter *specs* used by builders → `../builders/`.
- `ModelConfig`/registry wiring → `../config/`.

**Key fixtures.** Root `tests/conftest.py`. No folder-local conftest.
