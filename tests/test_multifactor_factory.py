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


# ------------------------------------------------------------------------------
# Condition-specific dispersion r (mean_disp) — v2
# ------------------------------------------------------------------------------


def _dispersion_spec():
    # donor x condition; dispersion hierarchy on the condition factor only.
    obs = pd.DataFrame(
        {
            "sample": ["D1", "D2", "D3", "D1", "D2", "D3"],
            "treatment": ["control", "control", "control", "drug", "drug", "drug"],
        }
    )
    spec, _ = normalize_grouping(
        dataset_key=None,
        hierarchy=[GroupLevel(name="treatment"), GroupLevel(name="sample")],
        interactions=None,
        obs=obs,
        dataset_priors={
            "expression": {"sample": "horseshoe"},   # mean varies by donor
            "dispersion": {"treatment": "gaussian"},  # r varies by condition
            "prob": "none",
            "zero_inflation": "none",
            "overdispersion": "none",
            "regime": "none",
        },
    )
    return spec


def _build_mean_disp_config(spec, parameterization="mean_disp"):
    b = (
        ModelConfigBuilder()
        .for_model("nbvcp")
        .with_parameterization(parameterization)
        .unconstrained()
    )
    b._n_datasets = spec.n_leaves
    b._grouping_spec = spec
    b._expression_dataset_prior = "horseshoe"  # reduced leaf-axis family for mu
    return b.build()


def test_factory_builds_condition_specific_dispersion_r():
    spec = _dispersion_spec()
    config = _build_mean_disp_config(spec)
    model, guide, specs = create_model(config)
    by_name = {s.name: s for s in specs}

    # r becomes an additive multi-factor parameter carrying ONLY the condition
    # (treatment) effect -> one r per condition, shared across donors.
    r = by_name["r"]
    assert isinstance(r, MultiFactorPositiveNormalSpec)
    assert {f.name for f in r.factors} == {"treatment"}
    assert "log_r_dataset_loc" in by_name

    # mu independently carries its own donor (sample) hierarchy.
    mu = by_name["mu"]
    assert isinstance(mu, MultiFactorPositiveNormalSpec)
    assert {f.name for f in mu.factors} == {"sample"}


def test_factory_builds_crossed_dispersion_r():
    """The dispersion hierarchy mirrors the mean: the SAME crossed (treatment x
    sample) decomposition on BOTH mu and r (not restricted to one factor)."""
    obs = pd.DataFrame(
        {
            "sample": ["D1", "D2", "D3", "D1", "D2", "D3"],
            "treatment": ["control", "control", "control", "drug", "drug", "drug"],
        }
    )
    spec, _ = normalize_grouping(
        dataset_key=None,
        hierarchy=[GroupLevel(name="treatment"), GroupLevel(name="sample")],
        interactions=None,
        obs=obs,
        dataset_priors={
            "expression": {"treatment": "gaussian", "sample": "horseshoe"},
            "dispersion": {"treatment": "gaussian", "sample": "horseshoe"},
            "prob": "none",
            "zero_inflation": "none",
            "overdispersion": "none",
            "regime": "none",
        },
    )
    config = _build_mean_disp_config(spec)
    _model, _guide, specs = create_model(config)
    by_name = {s.name: s for s in specs}

    for name in ("mu", "r"):
        p = by_name[name]
        assert isinstance(p, MultiFactorPositiveNormalSpec)
        assert {f.name for f in p.factors} == {"treatment", "sample"}


def test_dispersion_hierarchy_rejected_for_non_mean_disp():
    spec = _dispersion_spec()
    config = _build_mean_disp_config(spec, parameterization="mean_odds")
    with pytest.raises(ValueError, match="mean_disp"):
        create_model(config)


def test_dispersion_hierarchy_forces_exp_link():
    from scribe.api.stages.model_config_build import (
        _force_exp_for_expression_hierarchy,
    )

    spec = _dispersion_spec()
    config = _build_mean_disp_config(spec)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config2 = _force_exp_for_expression_hierarchy(config)
    # Δ log r interpretability requires the exp link on r.
    assert config2.resolve_positive_transform("r") == "exp"


def test_dispersion_horseshoe_honors_per_spec_tau0():
    # A dict family-spec on the dispersion hierarchy overrides the global
    # horseshoe tau0 default (family-as-spec hyperparameter threading).
    obs = pd.DataFrame(
        {
            "sample": ["D1", "D2", "D3", "D1", "D2", "D3"],
            "treatment": ["control", "control", "control", "drug", "drug", "drug"],
        }
    )
    spec, _ = normalize_grouping(
        dataset_key=None,
        hierarchy=[GroupLevel(name="treatment"), GroupLevel(name="sample")],
        interactions=None,
        obs=obs,
        dataset_priors={
            "expression": "none",
            "dispersion": {"treatment": {"type": "horseshoe", "tau0": 5.0}},
            "prob": "none",
            "zero_inflation": "none",
            "overdispersion": "none",
            "regime": "none",
        },
    )
    b = (
        ModelConfigBuilder()
        .for_model("nbvcp")
        .with_parameterization("mean_disp")
        .unconstrained()
    )
    b._n_datasets = spec.n_leaves
    b._grouping_spec = spec
    config = b.build()
    _, _, specs = create_model(config)
    by_name = {s.name: s for s in specs}
    # Global default tau0 is 1.0; the per-spec value must win.
    assert float(by_name["tau_r_treatment"].scale) == 5.0
