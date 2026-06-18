"""Tests for the additive multi-factor hierarchical param specs (M2).

Exercises ``MultiFactor{Positive,Sigmoid}NormalSpec.sample_hierarchical``
directly via a NumPyro trace: leaf/effect shapes, the gather+sum math, the
single-factor identity reduction, fixed vs random (gaussian/horseshoe) scales,
the sigmoid variant, and mixtures.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
import pytest

from scribe.models.builders.parameter_specs import (
    GroupingFactorSpec,
    MultiFactorPositiveNormalSpec,
    MultiFactorSigmoidNormalSpec,
)


def _trace(spec, dims, param_values, seed=0):
    def _model():
        spec.sample_hierarchical(dims, param_values)

    seeded = handlers.seed(_model, jax.random.PRNGKey(seed))
    return handlers.trace(seeded).get_trace()


# ------------------------------------------------------------------------------
# Two crossed gaussian factors on gene-specific mu
# ------------------------------------------------------------------------------


def test_two_factor_gaussian_shapes_and_math():
    G = 3
    # 5 present leaves (donor x condition, missing one combo).
    t = GroupingFactorSpec(
        name="treatment", n_levels=2, leaf_to_level=(0, 0, 0, 1, 1),
        prior="gaussian", scale_name="s_t",
        raw_name="mu_raw_t", effect_name="mu_eff_t",
    )
    s = GroupingFactorSpec(
        name="sample", n_levels=3, leaf_to_level=(0, 1, 2, 0, 1),
        prior="gaussian", scale_name="s_s",
        raw_name="mu_raw_s", effect_name="mu_eff_s",
    )
    spec = MultiFactorPositiveNormalSpec(
        name="mu", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        hyper_loc_name="log_mu_loc", is_gene_specific=True, is_dataset=True,
        factors=(t, s),
    )
    loc = jnp.array([0.1, -0.2, 0.3])
    pv = {"log_mu_loc": loc, "s_t": jnp.array(0.5), "s_s": jnp.array(0.3)}
    dims = {"n_genes": G}
    tr = _trace(spec, dims, pv)

    assert tr["mu_raw_t"]["value"].shape == (2, G)
    assert tr["mu_raw_s"]["value"].shape == (3, G)
    assert tr["mu_eff_t"]["value"].shape == (2, G)
    assert tr["mu_eff_s"]["value"].shape == (3, G)
    assert tr["mu"]["value"].shape == (5, G)

    # Reconstruct the expected leaf parameter from the sampled z's.
    z_t = np.asarray(tr["mu_raw_t"]["value"])
    z_s = np.asarray(tr["mu_raw_s"]["value"])
    eff_t = 0.5 * z_t
    eff_s = 0.3 * z_s
    l2l_t = np.array([0, 0, 0, 1, 1])
    l2l_s = np.array([0, 1, 2, 0, 1])
    expected = np.exp(np.asarray(loc)[None, :] + eff_t[l2l_t] + eff_s[l2l_s])
    np.testing.assert_allclose(np.asarray(tr["mu"]["value"]), expected, rtol=1e-5)


# ------------------------------------------------------------------------------
# Single-factor identity reduces to loc + scale * z
# ------------------------------------------------------------------------------


def test_single_factor_identity_reduction():
    G = 4
    D = 3
    f = GroupingFactorSpec(
        name="sample", n_levels=D, leaf_to_level=tuple(range(D)),
        prior="gaussian", scale_name="s",
        raw_name="mu_raw", effect_name="mu_eff",
    )
    spec = MultiFactorPositiveNormalSpec(
        name="mu", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        hyper_loc_name="loc", is_gene_specific=True, is_dataset=True,
        factors=(f,),
    )
    loc = jnp.zeros(G)
    pv = {"loc": loc, "s": jnp.array(0.7)}
    tr = _trace(spec, {"n_genes": G}, pv)
    z = np.asarray(tr["mu_raw"]["value"])  # (D, G)
    expected = np.exp(0.7 * z)  # loc = 0
    np.testing.assert_allclose(np.asarray(tr["mu"]["value"]), expected, rtol=1e-5)


# ------------------------------------------------------------------------------
# Fixed effect: uses fixed_scale, no hyperparameter site needed
# ------------------------------------------------------------------------------


def test_fixed_effect_uses_fixed_scale():
    G = 2
    f = GroupingFactorSpec(
        name="treatment", n_levels=2, leaf_to_level=(0, 1, 0, 1),
        effect_type="fixed", fixed_scale=0.9,
        raw_name="mu_raw_t", effect_name="mu_eff_t",
    )
    spec = MultiFactorPositiveNormalSpec(
        name="mu", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        hyper_loc_name="loc", is_gene_specific=True, is_dataset=True,
        factors=(f,),
    )
    pv = {"loc": jnp.zeros(G)}  # NOTE: no scale site for a fixed factor
    tr = _trace(spec, {"n_genes": G}, pv)
    z = np.asarray(tr["mu_raw_t"]["value"])
    np.testing.assert_allclose(
        np.asarray(tr["mu_eff_t"]["value"]), 0.9 * z, rtol=1e-6
    )


# ------------------------------------------------------------------------------
# Horseshoe scale computation
# ------------------------------------------------------------------------------


def test_horseshoe_scale():
    G = 3
    f = GroupingFactorSpec(
        name="sample", n_levels=2, leaf_to_level=(0, 1),
        prior="horseshoe", tau_name="tau", lambda_name="lam", c_sq_name="c_sq",
        raw_name="mu_raw", effect_name="mu_eff",
    )
    spec = MultiFactorPositiveNormalSpec(
        name="mu", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        hyper_loc_name="loc", is_gene_specific=True, is_dataset=True,
        factors=(f,),
    )
    tau = jnp.array(0.4)
    lam = jnp.array([0.1, 2.0, 0.5])
    c_sq = jnp.array(4.0)
    pv = {"loc": jnp.zeros(G), "tau": tau, "lam": lam, "c_sq": c_sq}
    tr = _trace(spec, {"n_genes": G}, pv)
    z = np.asarray(tr["mu_eff"]["value"]) / np.asarray(
        # back out z is awkward; recompute scale and compare to effect/z
        1.0
    )
    # Recompute scale and compare effect == scale * z_raw.
    z_raw = np.asarray(tr["mu_raw"]["value"])
    c = np.sqrt(np.asarray(c_sq))
    scale = np.asarray(tau) * c * np.asarray(lam) / np.sqrt(
        np.asarray(c_sq) + np.asarray(tau) ** 2 * np.asarray(lam) ** 2
    )
    np.testing.assert_allclose(
        np.asarray(tr["mu_eff"]["value"]), scale[None, :] * z_raw, rtol=1e-5
    )


# ------------------------------------------------------------------------------
# Sigmoid variant on a scalar (per-leaf) p
# ------------------------------------------------------------------------------


def test_sigmoid_scalar_p():
    f = GroupingFactorSpec(
        name="treatment", n_levels=2, leaf_to_level=(0, 1, 0, 1),
        prior="gaussian", scale_name="s",
        raw_name="p_raw", effect_name="p_eff",
    )
    spec = MultiFactorSigmoidNormalSpec(
        name="p", shape_dims=(), default_params=(0.0, 1.0),
        hyper_loc_name="logit_p_loc", is_gene_specific=False, is_dataset=True,
        factors=(f,),
    )
    pv = {"logit_p_loc": jnp.array(0.0), "s": jnp.array(0.5)}
    tr = _trace(spec, {}, pv)
    assert tr["p"]["value"].shape == (4,)
    vals = np.asarray(tr["p"]["value"])
    assert np.all((vals > 0) & (vals < 1))
    z = np.asarray(tr["p_raw"]["value"])  # (2,)
    expected = 1.0 / (1.0 + np.exp(-(0.5 * z[np.array([0, 1, 0, 1])])))
    np.testing.assert_allclose(vals, expected, rtol=1e-5)


# ------------------------------------------------------------------------------
# Mixture (component axis present)
# ------------------------------------------------------------------------------


def test_mixture_shapes():
    G, K, D = 2, 2, 3
    f = GroupingFactorSpec(
        name="sample", n_levels=D, leaf_to_level=tuple(range(D)),
        prior="gaussian", scale_name="s",
        raw_name="mu_raw", effect_name="mu_eff",
    )
    spec = MultiFactorPositiveNormalSpec(
        name="mu", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        hyper_loc_name="loc", is_gene_specific=True, is_dataset=True,
        is_mixture=True, factors=(f,),
    )
    pv = {"loc": jnp.zeros((K, G)), "s": jnp.array(0.5)}
    tr = _trace(spec, {"n_genes": G, "n_components": K}, pv)
    assert tr["mu_raw"]["value"].shape == (K, D, G)
    assert tr["mu_eff"]["value"].shape == (K, D, G)
    assert tr["mu"]["value"].shape == (K, D, G)


def test_companion_specs():
    f = GroupingFactorSpec(
        name="sample", n_levels=2, leaf_to_level=(0, 1),
        prior="gaussian", scale_name="s",
        raw_name="mu_raw", effect_name="mu_eff",
    )
    spec = MultiFactorPositiveNormalSpec(
        name="mu", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        hyper_loc_name="loc", is_gene_specific=True, is_dataset=True,
        factors=(f,),
    )
    comps = {c.name: c for c in spec.companion_specs}
    assert set(comps) == {"mu_raw", "mu_eff"}
    assert all(c.is_factor and not c.is_dataset for c in comps.values())
