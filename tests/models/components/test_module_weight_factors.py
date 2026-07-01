"""Unit tests for the additive module-weight operator builder + assembly.

Covers the pure (non-fit) machinery in
``scribe.models.components.module_weights``:

- the shared additive assembly ``module_weights_leaf_from_factors`` and its
  three-constraint gauge (global leaf anchor + leaf-count-weighted per-factor
  sum-to-zero + interaction zero-margin);
- the single-factor fast path reduces *exactly* to the flat kernel
  ``module_weights_from_raw`` (bit-identity with the pre-refold model);
- the build-time gaussian-only reject;
- the build-time identifiability rank guard (raises on a rank-deficient design,
  passes on a complete crossed design).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from scribe.models.config.grouping import (
    Factor,
    GroupingSpec,
    PriorFamilySpec,
)
import scribe.models.components.module_weights as mw


def _factor(name, kind, levels, leaf_to_level, *, family="gaussian",
            effect_type="random", fixed_scale=None):
    priors = (
        {}
        if family is None
        else {"module_weight": PriorFamilySpec(type=family)}
    )
    return Factor(
        name=name,
        kind=kind,
        nested_in=None,
        effect_type=effect_type,
        fixed_scale=fixed_scale,
        levels=tuple(levels),
        leaf_to_level=tuple(leaf_to_level),
        priors=priors,
    )


def _crossed_2x7_spec():
    """A complete 2 (perturbation) × 7 (sample) crossed + interaction spec."""
    donors = [f"D{i}" for i in range(7)]
    pert = _factor(
        "perturbation", "base", ["ctrl", "pano"], [0] * 7 + [1] * 7,
        effect_type="fixed",
    )
    samp = _factor("sample", "base", donors, list(range(7)) * 2)
    inter = _factor(
        "perturbation:sample", "interaction",
        [f"c{i}" for i in range(14)], list(range(14)),
    )
    return GroupingSpec(
        factors=(pert, samp, inter),
        leaf_labels=tuple(f"L{i}" for i in range(14)),
        n_leaves=14,
    )


def _raw_tau(ops, K, seed=0):
    rng = np.random.default_rng(seed)
    raw = {
        f["name"]: jnp.asarray(rng.standard_normal((f["n_levels"], K)))
        for f in ops["factors"]
    }
    tau = {
        f["name"]: jnp.asarray(0.3)
        for f in ops["factors"]
        if f["is_random"]
    }
    return raw, tau


def test_crossed_ranks_and_rank_guard_passes():
    """Complete 2×7 crossed+interaction: internal ranks 1/6/6, guard passes."""
    ops = mw.build_module_weight_operators(_crossed_2x7_spec())
    assert ops is not None
    assert not ops["fast_path"]
    ranks = {
        f["name"]: int(
            np.linalg.matrix_rank(np.asarray(f["P"]).astype(np.float64), tol=1e-6)
        )
        for f in ops["factors"]
    }
    assert ranks == {
        "perturbation": 1,
        "sample": 6,
        "perturbation:sample": 6,
    }  # total free = 13 = n_leaves - 1


def test_global_anchor_and_per_factor_gauges():
    """Assembled log s is leaf-anchored; each factor effect is weighted-zero."""
    ops = mw.build_module_weight_operators(_crossed_2x7_spec())
    K = 4
    raw, tau = _raw_tau(ops, K)
    s = np.asarray(mw.module_weights_leaf_from_factors(ops, raw, tau))
    assert s.shape == (14, K)
    # Global leaf anchor: Σ_leaf log s = 0 per module.
    np.testing.assert_allclose(np.log(s).mean(axis=0), 0.0, atol=1e-5)

    eff = mw.module_weight_effects_from_raw(ops, raw, tau)
    for f in ops["factors"]:
        l2l = np.asarray(f["leaf_to_level"])
        counts = np.bincount(l2l, minlength=f["n_levels"]).astype(float)
        a = np.asarray(eff[f["name"]])
        # Leaf-count-weighted sum-to-zero over the factor's levels.
        np.testing.assert_allclose(
            (counts[:, None] * a).sum(axis=0), 0.0, atol=1e-5
        )


def test_interaction_zero_per_parent_margins():
    """Interaction effect has zero margins along BOTH parents (present cells)."""
    ops = mw.build_module_weight_operators(_crossed_2x7_spec())
    K = 3
    raw, tau = _raw_tau(ops, K, seed=1)
    eff = mw.module_weight_effects_from_raw(ops, raw, tau)
    g = np.asarray(eff["perturbation:sample"])  # (14, K); level i = (i//7, i%7)
    pert_of = np.array([i // 7 for i in range(14)])
    don_of = np.array([i % 7 for i in range(14)])
    m_pert = np.array([g[pert_of == p].sum(0) for p in range(2)])
    m_don = np.array([g[don_of == d].sum(0) for d in range(7)])
    np.testing.assert_allclose(m_pert, 0.0, atol=1e-5)
    np.testing.assert_allclose(m_don, 0.0, atol=1e-5)


def test_single_factor_fast_path_matches_flat_kernel():
    """One random base factor with identity gather == flat module_weights_from_raw."""
    donors = [f"D{i}" for i in range(7)]
    samp = _factor("sample", "base", donors, list(range(7)))
    spec = GroupingSpec(
        factors=(samp,), leaf_labels=tuple(donors), n_leaves=7
    )
    ops = mw.build_module_weight_operators(spec)
    assert ops["fast_path"] is True
    K = 4
    rng = np.random.default_rng(2)
    raw = jnp.asarray(rng.standard_normal((7, K)))
    tau_raw = jnp.asarray(-0.4)
    s_fast = mw.module_weights_leaf_from_factors(
        ops, {"sample": raw}, {"sample": tau_raw}
    )
    s_flat = mw.module_weights_from_raw(raw, tau_raw)
    # Bit-identity (the fast path delegates to the flat kernel).
    np.testing.assert_array_equal(np.asarray(s_fast), np.asarray(s_flat))


def test_fixed_factor_uses_default_fixed_scale():
    """A fixed factor with fixed_scale=None resolves to the module default."""
    donors = [f"D{i}" for i in range(5)]
    samp = _factor(
        "sample", "base", donors, list(range(5)),
        effect_type="fixed", fixed_scale=None,
    )
    spec = GroupingSpec(
        factors=(samp,), leaf_labels=tuple(donors), n_leaves=5
    )
    ops = mw.build_module_weight_operators(spec)
    (fo,) = ops["factors"]
    assert fo["is_random"] is False
    assert fo["fixed_scale"] == mw._DEFAULT_MODULE_WEIGHT_FIXED_SCALE
    # Fixed factor: no tau needed; assembly runs from raw alone.
    raw = {"sample": jnp.asarray(np.random.default_rng(0).standard_normal((5, 3)))}
    s = np.asarray(mw.module_weights_leaf_from_factors(ops, raw, {}))
    assert s.shape == (5, 3)
    np.testing.assert_allclose(np.log(s).mean(axis=0), 0.0, atol=1e-5)


def test_gaussian_only_reject():
    """A non-gaussian module_weight family raises at operator-build time."""
    donors = [f"D{i}" for i in range(7)]
    bad = _factor("sample", "base", donors, list(range(7)), family="horseshoe")
    spec = GroupingSpec(
        factors=(bad,), leaf_labels=tuple(donors), n_leaves=7
    )
    with pytest.raises(ValueError, match="gaussian"):
        mw.build_module_weight_operators(spec)


def test_rank_guard_raises_on_confounded_factors():
    """Two perfectly-confounded factors are non-identifiable -> guard raises."""
    a = _factor("donor", "base", ["x", "y"], [0, 0, 1, 1])
    b = _factor("batch", "base", ["p", "q"], [0, 0, 1, 1])  # identical gather
    spec = GroupingSpec(
        factors=(a, b), leaf_labels=tuple(f"L{i}" for i in range(4)), n_leaves=4
    )
    with pytest.raises(ValueError, match="not identifiable"):
        mw.build_module_weight_operators(spec)


def test_unused_category_level_is_compacted():
    """A declared-but-unused level (phantom) is dropped, not left contaminating.

    A pandas categorical retains unused categories after subsetting, so
    Factor.levels can exceed the number of present leaves. The operator builder
    must compact to present levels: the effect is then properly centered on the
    present levels, the fast path is restored, and no phantom NCP row survives.
    """
    K = 3
    # 3 present donors, but 4 declared levels (D3 unused -> phantom).
    phantom = _factor("sample", "base", ["D0", "D1", "D2", "D3"], [0, 1, 2])
    spec = GroupingSpec(
        factors=(phantom,), leaf_labels=("D0", "D1", "D2"), n_leaves=3
    )
    ops = mw.build_module_weight_operators(spec)
    (fo,) = ops["factors"]
    # Compacted to the 3 present levels; the single-factor identity fast path
    # is restored (it was disabled when n_levels != n_leaves).
    assert fo["n_levels"] == 3
    assert ops["fast_path"] is True
    # Raw is over present levels only (no phantom row).
    rng = np.random.default_rng(0)
    raw = {"sample": jnp.asarray(rng.standard_normal((3, K)))}
    s = np.asarray(mw.module_weights_leaf_from_factors(ops, raw, {"sample": jnp.asarray(0.5)}))
    assert s.shape == (3, K)
    np.testing.assert_allclose(np.log(s).mean(axis=0), 0.0, atol=1e-5)
    # Effect present-level weighted sum is ~0 (properly centered, not leaked).
    eff = np.asarray(
        mw.module_weight_effects_from_raw(ops, raw, {"sample": jnp.asarray(0.5)})["sample"]
    )
    assert eff.shape == (3, K)
    np.testing.assert_allclose(eff.sum(axis=0), 0.0, atol=1e-5)


def test_inactive_when_no_participating_factor():
    """No module_weight family declared -> operators are None (inactive)."""
    donors = [f"D{i}" for i in range(4)]
    samp = _factor("sample", "base", donors, list(range(4)), family=None)
    spec = GroupingSpec(
        factors=(samp,), leaf_labels=tuple(donors), n_leaves=4
    )
    assert mw.build_module_weight_operators(spec) is None
