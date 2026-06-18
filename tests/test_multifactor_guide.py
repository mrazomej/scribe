"""Tests for the multi-factor mean-field guide dispatch (M2).

Verifies that the guide registers per-factor NCP ``z`` variational params at
the right shapes, samples each factor's raw site (incl. fixed-effect factors),
does NOT sample the deterministic leaf, and that the guide's sample-site names
exactly match the model's ``sample_hierarchical`` sites.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
import pytest

from scribe.models.builders._guide_meanfield_mixin import setup_guide
from scribe.models.builders.parameter_specs import (
    GroupingFactorSpec,
    MultiFactorPositiveNormalSpec,
)
from scribe.models.components.guide_families import MeanFieldGuide


def _spec():
    # One fixed factor (no scale hyper) + one gaussian random factor.
    t = GroupingFactorSpec(
        name="treatment", n_levels=2, leaf_to_level=(0, 0, 1, 1),
        effect_type="fixed", fixed_scale=0.8,
        raw_name="mu_raw_t", effect_name="mu_eff_t",
    )
    s = GroupingFactorSpec(
        name="sample", n_levels=2, leaf_to_level=(0, 1, 0, 1),
        prior="gaussian", scale_name="s_s",
        raw_name="mu_raw_s", effect_name="mu_eff_s",
    )
    return MultiFactorPositiveNormalSpec(
        name="mu", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        hyper_loc_name="log_mu_loc", is_gene_specific=True, is_dataset=True,
        factors=(t, s),
    )


def _guide_trace(spec, dims, seed=0):
    def _guide():
        setup_guide(spec, MeanFieldGuide(), dims, None)

    seeded = handlers.seed(_guide, jax.random.PRNGKey(seed))
    return handlers.trace(seeded).get_trace()


def test_guide_registers_per_factor_params():
    G = 3
    spec = _spec()
    tr = _guide_trace(spec, {"n_genes": G})

    # Variational params for both factors' z, shape (L_f, G).
    for raw in ("mu_raw_t", "mu_raw_s"):
        assert tr[f"{raw}_loc"]["type"] == "param"
        assert tr[f"{raw}_loc"]["value"].shape == (2, G)
        assert tr[f"{raw}_scale"]["value"].shape == (2, G)
        # Scale is positively constrained.
        assert np.all(np.asarray(tr[f"{raw}_scale"]["value"]) > 0)

    # Both z sites are sampled (fixed factor included).
    assert tr["mu_raw_t"]["type"] == "sample"
    assert tr["mu_raw_s"]["type"] == "sample"
    # The deterministic leaf is NOT a guide latent.
    assert "mu" not in tr


def test_guide_sites_match_model_sites():
    G = 3
    spec = _spec()
    dims = {"n_genes": G}

    # Model latent sample sites (z's) from sample_hierarchical.
    pv = {"log_mu_loc": jnp.zeros(G), "s_s": jnp.array(0.4)}

    def _model():
        spec.sample_hierarchical(dims, pv)

    model_tr = handlers.trace(
        handlers.seed(_model, jax.random.PRNGKey(1))
    ).get_trace()
    model_sample_sites = {
        k for k, v in model_tr.items() if v["type"] == "sample"
    }

    guide_tr = _guide_trace(spec, dims)
    guide_sample_sites = {
        k for k, v in guide_tr.items() if v["type"] == "sample"
    }

    assert model_sample_sites == guide_sample_sites == {"mu_raw_t", "mu_raw_s"}


def test_guide_mixture_shapes():
    G, K = 2, 3
    s = GroupingFactorSpec(
        name="sample", n_levels=4, leaf_to_level=(0, 1, 2, 3),
        prior="gaussian", scale_name="s_s",
        raw_name="mu_raw_s", effect_name="mu_eff_s",
    )
    spec = MultiFactorPositiveNormalSpec(
        name="mu", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        hyper_loc_name="loc", is_gene_specific=True, is_dataset=True,
        is_mixture=True, factors=(s,),
    )
    tr = _guide_trace(spec, {"n_genes": G, "n_components": K})
    assert tr["mu_raw_s_loc"]["value"].shape == (K, 4, G)
    assert tr["mu_raw_s"]["value"].shape == (K, 4, G)
