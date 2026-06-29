"""Byte-identical param-spec diff harness for hierarchical-prior refactors.

This is a developer regression tool, NOT a pytest test (no ``test_`` prefix, so
pytest does not collect it). It drives ``create_model(config, validate=False)``
over a matrix of `(model, parameterization, prior family, gene-level / dataset /
two-state regime)` configurations that exercises every hierarchical-prior
builder path, and emits a canonical, deterministic text snapshot of the
resulting ``param_specs``.

It exists because the factory's hierarchical-prior builders (`_gaussianize`,
`_horseshoe_ncp`, `_neg_ncp` in ``models/presets/factory.py``, driven by the
``HierParam`` descriptors in ``models/builders/hier_descriptors.py``) are
expected to be **byte-identical** at the spec level: any refactor of them must
not change spec class names, fields, distribution params, or serialization. The
snapshot serializes every ``ParamSpec``'s concrete class plus all pydantic
fields, with numpyro ``Transform`` objects reduced to a class+params repr
(volatile object addresses stripped), so two snapshots compare exactly.

Workflow for a hierarchical-prior change (the correctness gate used to land the
descriptor refactor):

    # 1. On the unchanged tree (or `git stash` your changes), capture the golden:
    python tests/tools/spec_harness.py snapshot /tmp/golden.txt

    # 2. Apply / unstash your refactor, then capture the branch:
    python tests/tools/spec_harness.py snapshot /tmp/branch.txt

    # 3. Assert identical (exit 0 = identical, 1 = differs, prints a diff):
    python tests/tools/spec_harness.py diff /tmp/golden.txt /tmp/branch.txt

Run on CPU to avoid GPU init overhead:
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python tests/tools/spec_harness.py ...

NOTE ON COVERAGE: the matrix below uses single-grouping, non-mixture configs.
That covers the single-axis gene/dataset/regime builder paths exhaustively, but
NOT the mixture (`n_components`) or crossed multi-factor paths — those are
covered by the behavioral suites (``test_multi_dataset.py``,
``test_multifactor_*``). Extend ``_iter_configs`` if a change touches those.
"""

import re
import sys

from numpyro.distributions.transforms import Transform

from scribe.models.config import ModelConfigBuilder
from scribe.models.config.enums import HierarchicalPriorType
from scribe.models.presets.factory import create_model

# Object memory addresses ("object at 0x7f...") vary run-to-run and are noise.
_ADDR_RE = re.compile(r" object at 0x[0-9a-fA-F]+")


# ------------------------------------------------------------------------------
# Canonical serialization
# ------------------------------------------------------------------------------


def _canon(value):
    """Stable, deterministic representation of a spec field value."""
    if isinstance(value, Transform):
        # Transforms compare by identity, not value; the class name plus a repr
        # with the volatile object address stripped captures the class and any
        # bound params (e.g. AffineTransform loc/scale).
        return f"<Transform {type(value).__name__} {_ADDR_RE.sub('', repr(value))}>"
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, dict):
        return {k: _canon(value[k]) for k in sorted(value, key=str)}
    if hasattr(value, "tolist"):  # numpy / jax arrays
        try:
            return ("array", value.shape, _canon(value.tolist()))
        except Exception:
            return repr(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def _serialize_spec(spec):
    """(class_name, {field: canon_value}) for one ParamSpec."""
    fields = {}
    # ParamSpec is a pydantic BaseModel; model_fields lists declared fields.
    field_names = getattr(type(spec), "model_fields", None) or vars(spec)
    for fname in sorted(field_names):
        try:
            fields[fname] = _canon(getattr(spec, fname))
        except Exception as exc:  # pragma: no cover - diagnostic
            fields[fname] = f"<ERR {exc!r}>"
    return type(spec).__name__, fields


def _serialize_specs(specs):
    # Preserve list order (order is part of the contract for some passes) and
    # print index + name + class + fields for a line-by-line diff.
    lines = []
    for i, spec in enumerate(specs):
        cls, fields = _serialize_spec(spec)
        name = getattr(spec, "name", f"<spec{i}>")
        lines.append(f"    [{i}] name={name!r} class={cls}")
        for fk in sorted(fields):
            lines.append(f"        {fk} = {fields[fk]!r}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# Config matrix
# ------------------------------------------------------------------------------

_FAMILIES = {
    "gaussian": HierarchicalPriorType.GAUSSIAN,
    "horseshoe": HierarchicalPriorType.HORSESHOE,
    "neg": HierarchicalPriorType.NEG,
}

# Standard NB-family parameterizations (p/phi + mu/r single-axis hierarchies).
_NB_PARAMS = ["canonical", "mean_prob", "mean_odds", "mean_disp"]
# Two-state parameterizations (regime coordinate hierarchies).
_TWOSTATE_PARAMS = [
    "two_state_natural",
    "two_state_ratio",
    "two_state_mean_fano",
    "two_state_moment_delta",
]


def _base_config(model, param):
    return (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization(param)
        .unconstrained()
        .build()
    )


def _iter_configs():
    """Yield (label, ModelConfig) for the full builder matrix."""
    # --- Gene-level p/phi + mu/r + gate hierarchies (each family) ---
    for param in _NB_PARAMS:
        for fam_name, fam in _FAMILIES.items():
            cfg = _base_config("nbdm", param)
            yield (
                f"gene-mu/{param}/{fam_name}",
                cfg.model_copy(update={"expression_prior": fam}),
            )
            if param != "mean_disp":  # mean_disp has no scalar p site
                cfg = _base_config("nbdm", param)
                yield (
                    f"gene-p/{param}/{fam_name}",
                    cfg.model_copy(update={"prob_prior": fam}),
                )
            cfg = _base_config("zinb", param)
            yield (
                f"gene-gate/{param}/{fam_name}",
                cfg.model_copy(update={"zero_inflation_prior": fam}),
            )

    # --- Dataset-level hierarchies (n_datasets >= 2) ---
    for param in _NB_PARAMS:
        for fam_name, fam in _FAMILIES.items():
            cfg = _base_config("nbdm", param)
            yield (
                f"ds-mu/{param}/{fam_name}",
                cfg.model_copy(
                    update={"n_datasets": 3, "expression_dataset_prior": fam}
                ),
            )
            if param != "mean_disp":
                for mode in ("scalar", "gene_specific"):
                    cfg = _base_config("nbdm", param)
                    yield (
                        f"ds-p/{param}/{fam_name}/{mode}",
                        cfg.model_copy(
                            update={
                                "n_datasets": 3,
                                "prob_dataset_prior": fam,
                                "hierarchical_dataset_p": mode,
                            }
                        ),
                    )
            cfg = _base_config("zinb", param)
            yield (
                f"ds-gate/{param}/{fam_name}",
                cfg.model_copy(
                    update={
                        "n_datasets": 3,
                        "zero_inflation_dataset_prior": fam,
                    }
                ),
            )

    # --- Two-state regime dataset hierarchies (each family) ---
    for param in _TWOSTATE_PARAMS:
        for fam_name, fam in _FAMILIES.items():
            cfg = _base_config("twostate", param)
            yield (
                f"ds-regime/{param}/{fam_name}",
                cfg.model_copy(
                    update={"n_datasets": 3, "regime_dataset_prior": fam}
                ),
            )


# ------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------


def snapshot(out_path):
    blocks = []
    ok = failed = 0
    for label, cfg in _iter_configs():
        try:
            _, _, param_specs = create_model(cfg, validate=False)
            blocks.append(f"### {label}\n{_serialize_specs(param_specs)}")
            ok += 1
        except Exception as exc:
            blocks.append(
                f"### {label}\n    <BUILD-ERROR {type(exc).__name__}: {exc}>"
            )
            failed += 1
    text = "\n\n".join(blocks) + "\n"
    with open(out_path, "w") as fh:
        fh.write(text)
    print(
        f"wrote {out_path}: {ok} ok, {failed} build-error blocks, "
        f"{len(text)} bytes"
    )


def diff(golden_path, branch_path):
    with open(golden_path) as fh:
        golden = fh.read().splitlines()
    with open(branch_path) as fh:
        branch = fh.read().splitlines()
    if golden == branch:
        print(f"IDENTICAL: {len(golden)} lines match")
        return 0
    import difflib

    diff_lines = list(
        difflib.unified_diff(
            golden, branch, golden_path, branch_path, lineterm=""
        )
    )
    print(f"DIFFERENT: {len(diff_lines)} diff lines")
    for ln in diff_lines[:200]:
        print(ln)
    return 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "snapshot":
        snapshot(sys.argv[2])
    elif cmd == "diff":
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(
            "usage: spec_harness.py snapshot <out.txt>\n"
            "       spec_harness.py diff <golden.txt> <branch.txt>"
        )
