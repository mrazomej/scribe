"""End-to-end Laplace fits of the per-leaf module-weight hierarchy (Rung 1.5).

Exercises ``scribe.fit(model="nbln", inference_method="laplace",
priors={"module_weight": {...}}, ...)`` on small synthetic multi-donor data:

- the single-factor form (``dataset_key="donor"`` +
  ``priors={"module_weight": {"donor": "gaussian"}}``) reproduces the flat
  per-leaf behaviour: the fit runs and surfaces a gauge-respecting ``s`` via
  ``get_module_weights()`` + ``get_module_weight_tau()``;
- the hierarchy is correctly inert (``get_module_weights() is None``) for a
  plain NBLN-Laplace fit, so the default path is unchanged;
- the **decoupled** layout (``correlate_other_column=False`` with a pooled
  ``_other`` column) also supports the hierarchy;
- a **crossed + interaction** grouping (``perturbation × sample`` +
  ``perturbation:sample``) fits the additive decomposition and surfaces
  per-factor effects via ``get_module_weight_effects()`` with the correct
  shapes and gauges.

The per-cell-W Newton kernels themselves are unit-tested in
``test_nbln_percellW_newton.py``; the additive assembly / gauge / rank guard in
``tests/models/components/test_module_weight_factors.py``; here we test the full
Laplace integration through the new ``priors`` API.
"""

from __future__ import annotations

import numpy as np
import anndata as ad
import pytest

import scribe
from scribe import GroupLevel


def _multi_donor_adata(seed: int = 0, N: int = 240, G: int = 14, D: int = 4):
    """Synthetic multi-donor NBLN-ish counts with shared-W module structure."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(G, 3)) * 0.5
    s_true = np.exp(rng.normal(size=(D, 3)) * 0.8)  # distinct donor activity
    donor = rng.integers(0, D, size=N)
    z = np.stack([rng.normal(size=3) * s_true[donor[c]] for c in range(N)])
    counts = rng.poisson(np.exp(1.2 + z @ W.T)).astype(np.float32)
    adata = ad.AnnData(counts)
    adata.obs["donor"] = donor.astype(str)
    return adata


def _crossed_adata(seed: int = 0, N: int = 320, G: int = 14, D: int = 4):
    """Multi-donor counts with a crossed perturbation × donor design."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(G, 3)) * 0.5
    donor = rng.integers(0, D, size=N)
    pert = rng.integers(0, 2, size=N)
    # Distinct per-(pert,donor) activity so the additive effects are non-trivial.
    s_pert = np.exp(rng.normal(size=(2, 3)) * 0.5)
    s_don = np.exp(rng.normal(size=(D, 3)) * 0.7)
    z = np.stack(
        [rng.normal(size=3) * s_pert[pert[c]] * s_don[donor[c]] for c in range(N)]
    )
    counts = rng.poisson(np.exp(1.2 + z @ W.T)).astype(np.float32)
    adata = ad.AnnData(counts)
    adata.obs["donor"] = donor.astype(str)
    adata.obs["pert"] = pert.astype(str)
    return adata


def test_single_factor_surfaces_module_weights():
    """Single-factor form runs end-to-end and exposes a gauge-respecting s."""
    adata = _multi_donor_adata()
    D = adata.obs["donor"].nunique()
    K = 3
    res = scribe.fit(
        adata,
        model="nbln",
        inference_method="laplace",
        dataset_key="donor",
        priors={"module_weight": {"donor": "gaussian"}},
        correlate_other_column=True,  # legacy layout
        latent_dim=K,
        n_steps=200,
        seed=0,
    )

    s = res.get_module_weights()
    assert s is not None, "module_weights should be populated"
    s = np.asarray(s)
    assert s.shape == (D, K)
    assert np.all(s > 0.0)
    # Between-level scale learned + reported (scalar for the single-factor case).
    tau = res.get_module_weight_tau()
    assert tau is not None and float(tau) > 0.0
    # Global leaf-anchor gauge holds on the fitted values (per module).
    np.testing.assert_allclose(np.log(s).mean(axis=0), 0.0, atol=1e-4)
    # The fit actually moved s off the all-ones init (some spread).
    assert np.log(s).std() > 1e-4
    # Single-factor effects dict has exactly the one factor.
    eff = res.get_module_weight_effects()
    assert set(eff.keys()) == {"donor"}
    assert np.asarray(eff["donor"]).shape == (D, K)


def test_plain_nbln_laplace_has_no_module_weights():
    """A plain NBLN-Laplace fit (no hierarchy) leaves module_weights None."""
    adata = _multi_donor_adata(seed=1)
    res = scribe.fit(
        adata,
        model="nbln",
        inference_method="laplace",
        correlate_other_column=True,
        latent_dim=3,
        n_steps=50,
        seed=0,
    )
    assert res.get_module_weights() is None
    assert res.get_module_weight_tau() is None
    assert res.get_module_weight_effects() is None


def test_decoupled_layout_hierarchy_runs():
    """module_weight hierarchy on the decoupled layout runs end-to-end."""
    adata = _multi_donor_adata(seed=2)
    D = adata.obs["donor"].nunique()
    K = 3
    res = scribe.fit(
        adata,
        model="nbln",
        inference_method="laplace",
        dataset_key="donor",
        priors={"module_weight": {"donor": "gaussian"}},
        correlate_other_column=False,  # decoupled when _other present
        gene_coverage=0.5,  # pools low-coverage genes into `_other`
        latent_dim=K,
        n_steps=150,
        seed=0,
    )
    s = res.get_module_weights()
    assert s is not None
    s = np.asarray(s)
    assert s.shape == (D, K)
    assert np.all(s > 0.0)
    assert float(res.get_module_weight_tau()) > 0.0
    np.testing.assert_allclose(np.log(s).mean(axis=0), 0.0, atol=1e-4)


def test_crossed_interaction_surfaces_per_factor_effects():
    """Crossed + interaction fit surfaces gauge-respecting per-factor effects."""
    adata = _crossed_adata()
    D = adata.obs["donor"].nunique()
    K = 3
    res = scribe.fit(
        adata,
        model="nbln",
        inference_method="laplace",
        hierarchy=[
            GroupLevel("pert", effect_type="fixed"),
            GroupLevel("donor"),
        ],
        interactions=[("pert", "donor")],
        priors={
            "module_weight": {
                "pert": "gaussian",
                "donor": "gaussian",
                "pert:donor": "gaussian",
            }
        },
        correlate_other_column=True,
        latent_dim=K,
        n_steps=200,
        seed=0,
    )

    eff = res.get_module_weight_effects()
    assert set(eff.keys()) == {"pert", "donor", "pert:donor"}
    assert np.asarray(eff["pert"]).shape == (2, K)
    assert np.asarray(eff["donor"]).shape == (D, K)
    # Present (pert, donor) combinations = the leaves.
    n_leaves = adata.obs.groupby(["pert", "donor"]).ngroups
    assert np.asarray(eff["pert:donor"]).shape == (n_leaves, K)

    # Realized per-leaf weights: global leaf-anchor gauge holds.
    s = np.asarray(res.get_module_weights())
    assert s.shape == (n_leaves, K)
    np.testing.assert_allclose(np.log(s).mean(axis=0), 0.0, atol=1e-4)

    # τ reported per random factor; the fixed `pert` factor has no τ.
    tau_by = res.get_module_weight_tau()
    assert isinstance(tau_by, dict)
    assert set(tau_by.keys()) == {"donor", "pert:donor"}
    assert all(v > 0.0 for v in tau_by.values())


def test_non_gaussian_module_weight_family_rejected():
    """A non-gaussian family on module_weight raises (v1 gaussian-only)."""
    adata = _multi_donor_adata(seed=3)
    with pytest.raises((ValueError, Exception)):
        scribe.fit(
            adata,
            model="nbln",
            inference_method="laplace",
            dataset_key="donor",
            priors={"module_weight": {"donor": {"type": "horseshoe"}}},
            correlate_other_column=True,
            latent_dim=3,
            n_steps=20,
            seed=0,
        )
