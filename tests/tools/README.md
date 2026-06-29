# tests/tools

Developer regression utilities — **not** pytest test cases. Files here have no
`test_` prefix, so a normal `pytest tests/` run does not collect them; invoke
them directly.

## `spec_harness.py` — byte-identical param-spec diff

A correctness gate for refactors of the hierarchical-prior builders
(`_gaussianize`, `_horseshoe_ncp`, `_neg_ncp` in
`src/scribe/models/presets/factory.py`, driven by the `HierParam` descriptors in
`src/scribe/models/builders/hier_descriptors.py`).

Those builders must be **byte-identical at the spec level**: a refactor may move
or merge code, but it must not change `param_specs` class names, fields,
distribution parameters, or serialization. The harness drives
`create_model(config, validate=False)` over a matrix of
`(model, parameterization, prior family, gene-level / dataset / two-state regime)`
configurations and serializes the resulting specs into a canonical text
snapshot (numpyro `Transform`s reduced to class+params, with volatile object
addresses stripped, so two snapshots compare exactly).

### Usage

```bash
# Pin GPU off for speed/determinism:
export PREFIX='CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu'

# 1. Capture the GOLDEN snapshot on the unchanged tree
#    (git stash your refactor, or check out the base commit first):
$PREFIX python tests/tools/spec_harness.py snapshot /tmp/golden.txt

# 2. Apply / unstash the refactor, then capture the BRANCH snapshot:
$PREFIX python tests/tools/spec_harness.py snapshot /tmp/branch.txt

# 3. Assert identical (exit 0 = identical; exit 1 = differs + prints a diff):
$PREFIX python tests/tools/spec_harness.py diff /tmp/golden.txt /tmp/branch.txt
```

A faithful refactor prints `IDENTICAL: N lines match`. Any diff is a real
behavior change (or an intentional one you must then re-baseline).

### Coverage and limits

The matrix uses **single-grouping, non-mixture** configs. That exhaustively
covers the single-axis gene/dataset/regime builder paths, but it does **not**
cover:

- the mixture (`n_components` / `mixture_params`) path, or
- the crossed additive multi-factor path.

Those are covered by the behavioral suites (`tests/test_multi_dataset.py`,
`tests/test_multifactor_factory.py`, `tests/test_multifactor_posterior.py`). If
a change touches them, extend `_iter_configs` (or lean on those suites).

The harness also only snapshots the **param-spec** surface. It does **not**
exercise the fitted-param posterior reconstruction
(`get_posterior_distributions`), which operates on guide outputs rather than
specs — verify that path with the posterior behavioral tests.
