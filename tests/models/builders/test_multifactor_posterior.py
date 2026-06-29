"""Posterior reconstruction (get_map / get_distributions) for multi-factor fits.

The additive multi-factor expression hierarchy fits a population intercept plus
per-factor zero-mean effects, with no single ``log_mu_dataset_scale`` site. The
analytic posterior path (``get_distributions`` -> ``get_map``, used by every MAP
plot such as ``plot_mean_calibration``) must reconstruct the leaf-level mu/r from
those per-factor sites rather than the single-axis dataset-hierarchy sites.

Regression guard for the ``KeyError: 'log_mu_dataset_scale_loc'`` raised when
the multi-factor branch was missing.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from scribe.models.builders.posterior import (
    _build_multifactor_leaf_posterior,
    _resolve_multifactor_factors,
)


def _factor(name, *, family, effect_type, fixed_scale, leaf_to_level):
    _priors = {"expression": family}
    ns = SimpleNamespace(
        name=name,
        priors=_priors,
        effect_type=effect_type,
        fixed_scale=fixed_scale,
        leaf_to_level=leaf_to_level,
    )
    # Mirror Factor.family: family type string, "none" when absent.
    ns.family = lambda target, _p=_priors: _p.get(target, "none")
    return ns


# ------------------------------------------------------------------------------
# _resolve_multifactor_factors
# ------------------------------------------------------------------------------


def test_resolve_multifactor_factors_selects_present():
    """Only >1-factor specs with present, non-'none' expression sites resolve."""
    cond = _factor(
        "condition", family="gaussian", effect_type="fixed",
        fixed_scale=3.0, leaf_to_level=(0, 1),
    )
    donor = _factor(
        "donor", family="none", effect_type="random",
        fixed_scale=1.0, leaf_to_level=(0, 1),
    )
    gs = SimpleNamespace(factors=(cond, donor))
    mc = SimpleNamespace(grouping_spec=gs)
    params = {"mu_condition_raw_loc": jnp.zeros((2, 3))}

    out = _resolve_multifactor_factors(params, mc, "mu")
    assert [prefix for _, prefix in out] == ["mu_condition"]


def test_resolve_multifactor_factors_empty_for_legacy():
    """No spec or a single factor -> empty (single-axis path used instead)."""
    cond = _factor(
        "condition", family="gaussian", effect_type="random",
        fixed_scale=1.0, leaf_to_level=(0, 1),
    )
    assert _resolve_multifactor_factors(
        {}, SimpleNamespace(grouping_spec=None), "mu"
    ) == []
    assert _resolve_multifactor_factors(
        {}, SimpleNamespace(grouping_spec=SimpleNamespace(factors=(cond,))), "mu"
    ) == []


# ------------------------------------------------------------------------------
# _build_multifactor_leaf_posterior
# ------------------------------------------------------------------------------


def test_build_multifactor_leaf_posterior_assembles_leaf_mu():
    """Leaf location = population loc + sum_f scale_f * z_f[level_f(leaf)]."""
    G = 4
    # Leaves are the crossed (condition, donor) combinations:
    #   leaf:        0      1      2      3      4      5
    #   condition:   c0     c0     c0     c1     c1     c1
    #   donor:       d0     d1     d2     d0     d1     d2
    cond_l2l = (0, 0, 0, 1, 1, 1)
    donor_l2l = (0, 1, 2, 0, 1, 2)
    cond = _factor(
        "condition", family="gaussian", effect_type="fixed",
        fixed_scale=3.0, leaf_to_level=cond_l2l,
    )
    donor = _factor(
        "donor", family="gaussian", effect_type="random",
        fixed_scale=1.0, leaf_to_level=donor_l2l,
    )

    rng = np.random.default_rng(0)
    loc = rng.normal(0.0, 1.0, G)
    cond_raw = rng.normal(0.0, 1.0, (2, G))
    donor_raw = rng.normal(0.0, 1.0, (3, G))
    donor_scale_loc = 0.5  # gaussian scale -> softplus(loc)

    params = {
        "log_mu_dataset_loc_loc": jnp.asarray(loc),
        "log_mu_dataset_loc_scale": jnp.asarray(0.1 * np.ones(G)),
        "mu_condition_raw_loc": jnp.asarray(cond_raw),
        "mu_condition_raw_scale": jnp.asarray(0.1 * np.ones((2, G))),
        "mu_donor_raw_loc": jnp.asarray(donor_raw),
        "mu_donor_raw_scale": jnp.asarray(0.1 * np.ones((3, G))),
        "mu_donor_scale_loc": jnp.asarray(donor_scale_loc),
    }
    mf = [(cond, "mu_condition"), (donor, "mu_donor")]

    post = _build_multifactor_leaf_posterior(
        params, mf, "log_mu_dataset_loc",
        is_mixture=False, split=False,
        transform=dist.transforms.ExpTransform(),
    )

    # Reconstruct the expected unconstrained leaf locations.
    d_scale = float(np.logaddexp(0.0, donor_scale_loc))
    expected = np.empty((6, G))
    for leaf in range(6):
        expected[leaf] = (
            loc
            + 3.0 * cond_raw[cond_l2l[leaf]]
            + d_scale * donor_raw[donor_l2l[leaf]]
        )

    assert isinstance(post, dist.TransformedDistribution)
    got = np.asarray(post.base_dist.loc)
    assert got.shape == (6, G)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    # Positive support after the exp transform; finite variance.
    assert np.all(np.asarray(post.base_dist.scale) > 0)
    assert np.all(np.exp(got) > 0)


def test_build_multifactor_leaf_posterior_neg_unsupported():
    """An unsupported (neg) family raises a clear NotImplementedError."""
    bad = _factor(
        "donor", family="neg", effect_type="random",
        fixed_scale=1.0, leaf_to_level=(0, 1),
    )
    params = {
        "log_mu_dataset_loc_loc": jnp.zeros(3),
        "log_mu_dataset_loc_scale": jnp.ones(3),
        "mu_donor_raw_loc": jnp.zeros((2, 3)),
        "mu_donor_raw_scale": jnp.ones((2, 3)),
    }
    with pytest.raises(NotImplementedError, match="no analytic MAP"):
        _build_multifactor_leaf_posterior(
            params, [(bad, "mu_donor")], "log_mu_dataset_loc",
            is_mixture=False, split=False,
        )


# ------------------------------------------------------------------------------
# End-to-end: a real multi-factor fit must support get_map (the MAP-plot path)
# ------------------------------------------------------------------------------


def test_multifactor_get_map_reconstructs_leaf_expression():
    """A real crossed fit reconstructs leaf-level r via get_map without error."""
    import anndata as ad
    import pandas as pd

    import scribe

    rng = np.random.default_rng(0)
    G, donors, conds, cpl = 8, ["D1", "D2"], ["control", "drug"], 20
    base = rng.normal(2.0, 0.3, G)
    deff = rng.normal(0.0, 0.15, (len(donors), G))
    rows, od, oc = [], [], []
    for di, d in enumerate(donors):
        for c in conds:
            lm = base + deff[di]
            if c == "drug":
                lm = lm.copy()
                lm[:2] += 2.0
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
        priors={
            "mean_expression": {"condition": "gaussian", "donor": "gaussian"}
        },
        n_steps=150,
        batch_size=128,
    )

    n_leaves = results.model_config.n_datasets
    assert n_leaves == 4  # 2 conditions x 2 donors

    map_est = results.get_map(
        targets=["r", "p"], canonical=True, verbose=False, counts=adata
    )
    r = np.asarray(map_est["r"])
    # Leaf axis present and sized to the number of present combinations.
    assert r.shape[-2:] == (n_leaves, G)
    assert np.all(np.isfinite(r))
