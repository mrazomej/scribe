# tests/models

**Purpose.** The model layer — how models and guides are *defined, configured,
and built*, plus per-model end-to-end behavior. This is the largest subsystem,
so (unlike the other folders) it is subdivided one level deep.

**Source under test.** `src/scribe/models` (+ `scribe.flows`,
`scribe.utils.parameter_collector`).

## Subfolders

| folder | scope | maps to |
| --- | --- | --- |
| [`builders/`](builders/) | model+guide construction, parameter specs, posterior reconstruction, hierarchical/multifactor structure | `models/builders` |
| [`config/`](config/) | `ModelConfig`/builder/registry, grouping, unified priors, parameter mapping | `models/config` (+ `utils.parameter_collector`) |
| [`likelihoods/`](likelihoods/) | likelihood functions + numerical correctness | `models/components/likelihoods` |
| [`parameterizations/`](parameterizations/) | parameterization classes + derived parameters | `models/parameterizations` |
| [`components/`](components/) | VAE encoders/decoders, covariate embeddings, normalizing flows | `models/components` (+ `flows`) |
| [`families/`](families/) | per-model end-to-end fit/recovery + two-state model behavior | (no single src module) |

## Where cross-cutting model pieces live (subsystem-first splits)

A feature that spans subsystems is split by *what each test exercises*:
- **two-state**: likelihood → `likelihoods/`; model behavior/recovery-via-fit → `families/`; **Laplace inference → `tests/laplace/`**; data-init → `tests/core/`; full recovery end-to-end → `tests/integration/`.
- **VAE**: encoders/decoders/embeddings → `components/`; **results class → `tests/inference/`**; full pipeline → `tests/integration/`.
- **multifactor**: spec/guide/posterior/factory → `builders/`; **recovery → `tests/integration/`**.

**What does NOT live here.** Inference *engines* → `tests/inference/`; `fit()`
orchestration stages → `tests/api/`; differential expression → `tests/de/`;
visualization → `tests/viz/`.

**Key fixtures.** Root `tests/conftest.py` (`rng_key`, `small_dataset`, device options). No folder-local conftest.
