# tests/models/builders

**Purpose.** Construction of models and guides: parameter specs, the model/guide
builders, posterior reconstruction, and hierarchical / multi-factor structure.

**Source under test.** `src/scribe/models/builders` (`parameter_specs`,
`guide_builder`, `posterior`, `hier_descriptors`, `_guide_*_mixin`) and the
`models.presets.factory` paths that assemble them.

**What lives here.**
- `test_builders` — the core builder/spec/posterior surface (largest file; spec splitting is a deferred follow-up).
- `test_unified_factory` — the unified `create_model` factory across model/guide/parameterization combinations.
- `test_hier_descriptors` — hierarchical-prior descriptor site names and target resolution.
- `test_latent_spec` — `GaussianLatentSpec` / `LatentSpec`.
- `test_capture_prior` — `BiologyInformedCaptureSpec` and organism-prior resolution.
- `test_multifactor_spec`, `test_multifactor_guide`, `test_multifactor_posterior`, `test_multifactor_factory` — multi-factor hierarchy *structure* (specs, guide registration, posterior, config build).

**What does NOT live here.**
- Multi-factor end-to-end *recovery* (synthesize → fit → recover) → `tests/integration/` (`test_multifactor_recover`).
- `ModelConfig` field validation and the registry → `../config/`.
- Likelihood `log_prob` correctness → `../likelihoods/`.

**Key fixtures.** Root `tests/conftest.py`. No folder-local conftest.
