# SCRIBE test suite

The tests are organized **subsystem-first**: a test lives in the folder for the
`src/scribe/` subsystem whose code it primarily exercises. The guiding rule is

> **changed `src/scribe/<X>` → run `pytest tests/<X>`**

so coverage maps deterministically onto the source tree. Filenames keep their
descriptive prefixes (`test_twostate_*`, `test_lnm_*`, `test_nbdm*`, …) for
navigation *within* a folder.

## Folder index

| folder | scope | source under test |
| --- | --- | --- |
| [`core/`](core/) | AxisLayout, factor/group views, gene coverage, normalization, data-init | `core` |
| [`models/`](models/) | model defs, builders, config, parameterizations, likelihoods, families *(subdivided)* | `models`, `flows` |
| [`inference/`](inference/) | SVI/MCMC engines, run_scribe, presets, checkpoint, results | `inference`, `svi`, `mcmc` |
| [`api/`](api/) | `fit()` orchestration stages | `api` |
| [`laplace/`](laplace/) | Laplace approximation, incl. all per-model Newton/obs paths | `laplace` |
| [`de/`](de/) | differential expression | `de` |
| [`mc/`](mc/) | model comparison (WAIC / PSIS-LOO / stacking) | `mc` |
| [`sampling/`](sampling/) | posterior-predictive + denoising *sampling* | `sampling` |
| [`viz/`](viz/) | visualization + PPC *plotting* | `viz` |
| [`cli/`](cli/) | CLI, hydra, slurm, data ingestion, logging | `cli`, `catalog`, `data_loader`, `_logging` |
| [`stats/`](stats/) | distributions, divergences, quadrature, jacobian map | `stats` |
| [`integration/`](integration/) | cross-cutting end-to-end (multi-dataset, hierarchical, VAE, parity, x64, serialization) | *(many)* |

`models/` is the only subdivided folder (it holds ~⅓ of the suite); see
[`models/README.md`](models/README.md) for its `builders/ config/ likelihoods/
parameterizations/ components/ families/` layout.

## Running tests

```bash
pytest                       # whole suite
pytest tests/de              # one subsystem
pytest tests/models          # a subsystem + all its subfolders
pytest tests/models/families # one subfolder
pytest tests/laplace -k twostate_ln_rate   # filter within a folder
```

Custom options (defined in the root `conftest.py`):

| option | values | meaning |
| --- | --- | --- |
| `--device` | `cpu` (default) / `gpu` | JAX platform |
| `--method` | `svi` / `mcmc` / `all` | inference method(s) to parametrize |
| `--parameterization` | `standard` / `linked` / `odds_ratio` / `all` | parameterization(s) |
| `--unconstrained` | `false` / `true` / `all` | unconstrained variants |
| `--guide-rank` | `none` / int / `all` | mean-field vs low-rank guide |

## Import mode & shared helpers

The suite uses **`--import-mode=importlib`** (set in `pyproject.toml`), so:
- there are **no `__init__.py`** files under `tests/` — it is not a package;
- test files are imported under unique, path-derived names, so **basenames need
  not be globally unique** across folders;
- **do not import one test module from another.** Shared, non-collected helpers
  live in top-level modules under `tests/` (importable because `pythonpath =
  ["tests", "."]`). Example: the synthetic NBLN result factory in
  [`_synthetic_results.py`](_synthetic_results.py), used as
  `from _synthetic_results import _nbln_result`.

## Fixtures

Global fixtures are in the root [`conftest.py`](conftest.py):
- `rng_key`, `small_dataset` — common inputs.
- `data_dir` — `tests/data/` (golden artifacts); use instead of `__file__`-relative paths.
- `project_root` — repo root; use for repo-relative lookups (e.g. `conf/`).
- `scribe_caplog` — captures the `scribe` logger hierarchy.

`tests/inference/` additionally has a **folder-local `conftest.py`** with
inference-specific fixtures (`sample_counts`, `nbdm_model_config`, …).

## Where do I add a new test?

1. Find the `src/scribe/` module it primarily exercises → put it in that folder
   (the table above maps subsystems to folders).
2. If it spans **many** subsystems with no single primary one and runs
   end-to-end → `integration/`.
3. If it is named after a **model family** and validates that model via a fit →
   `models/families/`.

Because the layout is subsystem-first, a few features are intentionally **split
across folders** (each piece sits with the code it tests). The most common:

| feature | where its pieces live |
| --- | --- |
| two-state | likelihood → `models/likelihoods/`; behavior → `models/families/`; **Laplace → `laplace/`**; data-init → `core/`; recovery → `integration/` |
| VAE | components → `models/components/`; results → `inference/`; pipeline → `integration/` |
| multifactor | spec/guide/posterior/factory → `models/builders/`; recovery → `integration/` |
| PPC | sampling → `sampling/`; plotting → `viz/` |

Each folder's `README.md` has a **"What does NOT live here"** section that points
to these siblings.
