"""Factory integration tests for the additive multi-factor hierarchy (M2).

Builds a ModelConfig from a 2-factor grouping spec (treatment = fixed gaussian,
sample = random horseshoe) and checks that ``create_model`` wires the
``MultiFactorPositiveNormalSpec`` for ``mu`` with the right per-factor effect
specs + hyperparameters, and that the resulting model traces to leaves of the
correct shape with per-factor effect sites present.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
import pytest

from scribe.models.config import ModelConfigBuilder
from scribe.models.config.grouping import normalize_grouping, GroupLevel
from scribe.models.presets.factory import create_model
from scribe.models.builders.parameter_specs import (
    MultiFactorPositiveNormalSpec,
)


def _zhao_like_spec():
    # Complete crossed donor x condition (6 leaves), treatment fixed.
    obs = pd.DataFrame(
        {
            "sample": ["D1", "D2", "D3", "D1", "D2", "D3"],
            "treatment": ["control", "control", "control", "drug", "drug", "drug"],
        }
    )
    spec, _ = normalize_grouping(
        dataset_key=None,
        hierarchy=[
            GroupLevel(name="treatment", effect_type="fixed", fixed_scale=1.0),
            GroupLevel(name="sample"),
        ],
        interactions=None,
        obs=obs,
        dataset_priors={
            "expression": {"treatment": "gaussian", "sample": "horseshoe"},
            "prob": "gaussian",
            "zero_inflation": "none",
            "overdispersion": "none",
            "regime": "none",
        },
    )
    return spec


def _build_config(spec):
    b = (
        ModelConfigBuilder()
        .for_model("nbvcp")
        .with_parameterization("mean_odds")
        .unconstrained()
    )
    b._n_datasets = spec.n_leaves
    b._grouping_spec = spec
    b._expression_dataset_prior = "gaussian"  # reduced leaf-axis family
    b._prob_dataset_prior = "gaussian"
    b._prob_dataset_mode = "gene_specific"
    return b.build()


def test_factory_builds_multifactor_mu_spec():
    spec = _zhao_like_spec()
    config = _build_config(spec)
    model, guide, specs = create_model(config)
    assert callable(model) and callable(guide)

    by_name = {s.name: s for s in specs}
    mu = by_name["mu"]
    assert isinstance(mu, MultiFactorPositiveNormalSpec)
    assert len(mu.factors) == 2

    facs = {f.name: f for f in mu.factors}
    assert facs["treatment"].effect_type == "fixed"
    assert facs["treatment"].fixed_scale == 1.0
    assert facs["sample"].prior == "horseshoe"

    # Population loc present; horseshoe hypers for the random factor present;
    # NO learned scale for the fixed treatment factor.
    assert "log_mu_dataset_loc" in by_name
    assert facs["sample"].tau_name in by_name
    assert facs["sample"].lambda_name in by_name
    assert facs["sample"].c_sq_name in by_name
    assert facs["treatment"].scale_name is None


def test_factory_model_traces_to_leaf_shape():
    spec = _zhao_like_spec()
    config = _build_config(spec)
    model, guide, specs = create_model(config)

    n_cells, n_genes, n_leaves = 12, 4, spec.n_leaves
    rng = np.random.default_rng(0)
    dataset_indices = jnp.asarray(
        rng.integers(0, n_leaves, size=n_cells), dtype=jnp.int32
    )

    def _run():
        model(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=None,
            dataset_indices=dataset_indices,
        )

    tr = numpyro.handlers.trace(
        numpyro.handlers.seed(_run, jax.random.PRNGKey(0))
    ).get_trace()

    # Leaf mu and per-factor effect sites present with the right shapes.
    facs = {f.name: f for f in by_factors(specs)}
    assert tr["mu"]["value"].shape == (n_leaves, n_genes)
    assert tr[facs["treatment"].effect_name]["value"].shape == (2, n_genes)
    assert tr[facs["sample"].effect_name]["value"].shape == (3, n_genes)
    # The per-factor NCP z sites are sampled.
    assert tr[facs["treatment"].raw_name]["type"] == "sample"
    assert tr[facs["sample"].raw_name]["type"] == "sample"


def by_factors(specs):
    for s in specs:
        if isinstance(s, MultiFactorPositiveNormalSpec):
            return s.factors
    return ()


def test_guide_traces_param_shapes():
    spec = _zhao_like_spec()
    config = _build_config(spec)
    model, guide, specs = create_model(config)
    n_cells, n_genes = 10, 4
    dataset_indices = jnp.zeros(n_cells, dtype=jnp.int32)

    def _run_guide():
        guide(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=None,
            dataset_indices=dataset_indices,
        )

    tr = numpyro.handlers.trace(
        numpyro.handlers.seed(_run_guide, jax.random.PRNGKey(1))
    ).get_trace()
    facs = {f.name: f for f in by_factors(specs)}
    # Guide registers per-factor NCP z params at (L_f, n_genes).
    assert tr[f"{facs['sample'].raw_name}_loc"]["value"].shape == (3, n_genes)
    assert tr[f"{facs['treatment'].raw_name}_loc"]["value"].shape == (2, n_genes)
