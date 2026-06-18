"""Tests for the fitted-effect accessor: get_factor_effect / FactorEffectView.

The multi-factor expression hierarchy stores each factor's additive effect as a
deterministic site (``mu_<factor>_effect``, shape ``(N, [K,] L_f, G)``) and the
learned shrinkage scale (``mu_<factor>_scale``). These tests use fake posterior
samples to verify the per-level indexing, the contrast, and the scale handling
(learned vs fixed), plus one real fit to cover the site-name wiring.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from scribe.core.factor_effect_view import FactorEffectView, get_factor_effect


def _fac(name, levels, effect_type, fixed_scale, family="gaussian"):
    return SimpleNamespace(
        name=name,
        levels=tuple(levels),
        n_levels=len(levels),
        effect_type=effect_type,
        fixed_scale=fixed_scale,
        priors={"expression": family},
    )


def _results(samples, factors, gene_names=None):
    spec = SimpleNamespace(base_factors=tuple(factors), factors=tuple(factors))
    return SimpleNamespace(
        model_config=SimpleNamespace(grouping_spec=spec),
        posterior_samples=samples,
        gene_names=gene_names,
    )


# ------------------------------------------------------------------------------
# Fixed contrast factor (condition)
# ------------------------------------------------------------------------------


def test_get_factor_effect_fixed_contrast():
    N, G = 5, 4
    effect = np.zeros((N, 2, G))
    effect[:, 0, :] = 1.0  # control
    effect[:, 1, :] = 3.0  # drug
    cond = _fac("condition", ["control", "drug"], "fixed", 3.0)
    res = _results({"mu_condition_effect": effect}, [cond])

    fx = get_factor_effect(res, "condition")
    assert isinstance(fx, FactorEffectView)
    assert fx.levels == ["control", "drug"]
    assert fx.effect_type == "fixed"
    # Identified quantity: the contrast.
    assert fx.contrast("drug", "control").shape == (N, G)
    np.testing.assert_allclose(fx.contrast("drug", "control"), 2.0)
    np.testing.assert_allclose(fx.map_contrast("drug", "control"), 2.0)
    # Fixed effect -> scale is the constant, not a learned sample.
    assert fx.scale == 3.0
    assert fx.effects().shape == (2, G)


# ------------------------------------------------------------------------------
# Random factor (donor): per-level deviations + learned scale
# ------------------------------------------------------------------------------


def test_get_factor_effect_random_donor():
    N, G = 6, 4
    rng = np.random.default_rng(0)
    effect = rng.normal(size=(N, 3, G))
    scale = rng.gamma(2.0, size=N)
    donor = _fac("donor", ["D1", "D2", "D3"], "random", None, "gaussian")
    res = _results(
        {"mu_donor_effect": effect, "mu_donor_scale": scale}, [donor]
    )

    fx = get_factor_effect(res, "donor")
    assert fx.levels == ["D1", "D2", "D3"]
    # Per-level deviations are individually meaningful for a random factor.
    assert fx["D2"].shape == (N, G)
    np.testing.assert_array_equal(fx["D2"], effect[:, 1, :])
    np.testing.assert_allclose(fx.map_effect("D3"), effect[:, 2, :].mean(axis=0))
    assert fx.effects().shape == (3, G)
    # Learned heterogeneity scale exposed as samples.
    np.testing.assert_array_equal(np.asarray(fx.scale), scale)
    assert len(fx) == 3
    assert list(fx) == ["D1", "D2", "D3"]


# ------------------------------------------------------------------------------
# Canonical parameterization uses the r_<factor>_effect site
# ------------------------------------------------------------------------------


def test_get_factor_effect_canonical_r_site():
    N, G = 4, 3
    effect = np.ones((N, 2, G))
    cond = _fac("condition", ["a", "b"], "fixed", 1.0)
    res = _results({"r_condition_effect": effect}, [cond])
    fx = get_factor_effect(res, "condition")
    assert fx.samples("a").shape == (N, G)


# ------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------


def test_get_factor_effect_unknown_factor():
    cond = _fac("condition", ["a", "b"], "fixed", 1.0)
    res = _results({"mu_condition_effect": np.ones((2, 2, 3))}, [cond])
    with pytest.raises(ValueError, match="unknown factor"):
        get_factor_effect(res, "donor")


def test_get_factor_effect_no_effect_site():
    # Factor present but its expression prior is "none" -> no effect site.
    cond = _fac("condition", ["a", "b"], "random", None, family="none")
    res = _results({}, [cond])
    with pytest.raises(ValueError, match="no fitted effect"):
        get_factor_effect(res, "condition")


def test_factor_effect_view_unknown_level():
    cond = _fac("condition", ["a", "b"], "fixed", 1.0)
    res = _results({"mu_condition_effect": np.ones((2, 2, 3))}, [cond])
    fx = get_factor_effect(res, "condition")
    with pytest.raises(KeyError, match="unknown level"):
        fx["zzz"]


# ------------------------------------------------------------------------------
# End-to-end: site-name wiring on a real crossed fit
# ------------------------------------------------------------------------------


def test_get_factor_effect_real_fit():
    import anndata as ad
    import pandas as pd

    import scribe

    rng = np.random.default_rng(0)
    G, donors, conds, cpl = 8, ["D1", "D2", "D3"], ["control", "drug"], 20
    base = rng.normal(2.0, 0.3, G)
    deff = rng.normal(0.0, 0.15, (len(donors), G))
    rows, od, oc = [], [], []
    for di, d in enumerate(donors):
        for c in conds:
            lm = base + deff[di]
            if c == "drug":
                lm = lm.copy()
                lm[:2] += 2.0  # up in drug
            mu = np.exp(lm)
            for _ in range(cpl):
                rows.append(rng.poisson(mu * rng.uniform(0.8, 1.2) * 8.0))
                od.append(d)
                oc.append(c)
    adata = ad.AnnData(
        X=np.asarray(rows, dtype=float),
        obs=pd.DataFrame({"donor": od, "condition": oc}),
    )
    results = scribe.fit(
        adata,
        parameterization="mean_odds",
        variable_capture=True,
        unconstrained=True,
        positive_transform={"mean_expression": "exp"},
        hierarchy=[
            scribe.GroupLevel("condition", effect_type="fixed", fixed_scale=3.0),
            scribe.GroupLevel("donor"),
        ],
        expression_dataset_prior={"condition": "gaussian", "donor": "gaussian"},
        n_steps=1500,
        batch_size=128,
    )

    cond_fx = results.get_factor_effect("condition")
    assert cond_fx.effect_type == "fixed"
    assert cond_fx.scale == 3.0
    contrast = cond_fx.map_contrast("drug", "control")  # (G,) log-mu effect
    assert contrast.shape == (G,)
    # Genes 0,1 were up-regulated in drug -> the drug-vs-control log-mu effect
    # is clearly larger for them than for the untouched genes. (The recovered
    # magnitude is compressed vs the injected +2.0: in the odds/capture model
    # the population intercept and p absorb part of the shift, so this checks
    # the recovered *direction*, not the literal effect size.)
    assert float(np.mean(contrast[:2])) > float(np.mean(contrast[2:])) + 0.1

    donor_fx = results.get_factor_effect("donor")
    assert donor_fx.effect_type == "random"
    assert np.asarray(donor_fx.scale).ndim == 1  # learned scale samples
    assert donor_fx.effects().shape == (len(donors), G)
