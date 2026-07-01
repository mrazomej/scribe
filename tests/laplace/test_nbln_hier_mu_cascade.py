"""Per-donor marginal cascade for NB-LogNormal-Laplace (step 4b).

Two layers are exercised:

1. **Extractor** ``freeze_values_hier_from_results`` — reconstructs the
   leaf-level mean matrix ``mu^(d)`` of shape ``(D, G)`` from a *hierarchical*
   independent-gene SVI source, ALIGNED to the target's leaf ordering by
   label, and log-transformed to the NBLN log-rate coordinate.  Tested with a
   light stub source (so the alignment / shape / error paths are exact and
   fast).
2. **End-to-end** ``scribe.fit`` — a hierarchical NBVCP-SVI source feeds a
   grouped NBLN-Laplace fit with a per-leaf module-weight hierarchy
   (``priors={"module_weight": {"donor": "gaussian"}}``) and
   ``informative_priors_freeze=("r", "mu")``.  The per-donor freeze flows
   through the per-cell-mu Newton path; the result surfaces the unpooled
   per-donor table via ``get_gene_mean_per_dataset()`` while the standard
   ``get_mu()`` stays pooled ``(G,)``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import anndata as ad
import pytest

import scribe
from scribe.laplace.priors import freeze_values_hier_from_results


# ---------------------------------------------------------------------------
# Stub hierarchical SVI source for the extractor unit tests
# ---------------------------------------------------------------------------


def _stub_hier_source(leaf_labels, mu_dg, r_g=None, var_names=None):
    """Minimal object satisfying the extractor's source contract.

    Provides ``n_genes``, ``var_names``, ``model_config.grouping_spec.
    leaf_labels``, and a ``get_map()`` returning a per-donor ``(D, G)`` mu
    (and optional pooled ``(G,)`` r).  No ``_uses_amortized_capture`` →
    treated as a non-amortized source.
    """
    mu_dg = np.asarray(mu_dg, dtype=np.float32)
    G = mu_dg.shape[1]
    map_dict = {"mu": mu_dg}
    if r_g is not None:
        map_dict["r"] = np.asarray(r_g, dtype=np.float32)
    gs = SimpleNamespace(leaf_labels=tuple(leaf_labels))
    mc = SimpleNamespace(grouping_spec=gs)

    def get_map(verbose=False, **_):
        return map_dict

    return SimpleNamespace(
        n_genes=G,
        var_names=(np.asarray(var_names) if var_names is not None else None),
        model_config=mc,
        get_map=get_map,
    )


def test_extractor_aligns_per_donor_mu_to_target_order():
    """(D, G) mu is reordered so row d matches target_leaf_labels[d]."""
    G = 5
    # Distinct per-donor NB means (positive); rows keyed by source label.
    src_labels = ["donorA", "donorB", "donorC"]
    mu_src = np.array(
        [
            np.full(G, 2.0),   # donorA
            np.full(G, 5.0),   # donorB
            np.full(G, 9.0),   # donorC
        ],
        dtype=np.float32,
    )
    var = np.array([str(i) for i in range(G)])
    src = _stub_hier_source(src_labels, mu_src, r_g=np.ones(G), var_names=var)

    # Target requests a DIFFERENT leaf order.
    target_order = ["donorC", "donorA", "donorB"]
    fv = freeze_values_hier_from_results(
        src,
        target_positive_transform="softplus",
        target_n_genes=G,
        target_n_cells=30,
        target_leaf_labels=target_order,
        target_n_datasets=3,
        target_gene_names=var,
        freeze_params=("mu",),
        verbose=False,
    )
    mu_loc = np.asarray(fv["mu"]["loc"])
    assert mu_loc.shape == (3, G)
    # Row d == log(NB mean of target_order[d]).
    expected = np.log(
        np.stack([mu_src[src_labels.index(lbl)] for lbl in target_order])
    )
    np.testing.assert_allclose(mu_loc, expected, atol=1e-5)


def test_extractor_missing_target_leaf_raises():
    """A target leaf with no matching source label is a clear error."""
    G = 4
    var = np.array([str(i) for i in range(G)])
    src = _stub_hier_source(
        ["d0", "d1"], np.ones((2, G)), var_names=var
    )
    with pytest.raises(ValueError, match="absent from the hierarchical"):
        freeze_values_hier_from_results(
            src,
            target_positive_transform="softplus",
            target_n_genes=G,
            target_n_cells=10,
            target_leaf_labels=["d0", "d_missing"],
            target_n_datasets=2,
            target_gene_names=var,
            freeze_params=("mu",),
            verbose=False,
        )


def test_extractor_rejects_1d_source_mu():
    """A pooled (1-D) source mu signals a non-hierarchical fit → error."""
    G = 4
    var = np.array([str(i) for i in range(G)])
    src = _stub_hier_source(["d0", "d1"], np.ones((2, G)), var_names=var)
    # Override get_map to return a 1-D mu (pooled).
    src.get_map = lambda verbose=False, **_: {"mu": np.ones(G, np.float32)}
    with pytest.raises(ValueError, match="2-D"):
        freeze_values_hier_from_results(
            src,
            target_positive_transform="softplus",
            target_n_genes=G,
            target_n_cells=10,
            target_leaf_labels=["d0", "d1"],
            target_n_datasets=2,
            target_gene_names=var,
            freeze_params=("mu",),
            verbose=False,
        )


# ---------------------------------------------------------------------------
# End-to-end: hierarchical SVI source -> per-donor mu cascade -> NBLN-Laplace
# ---------------------------------------------------------------------------


def _multi_donor_adata(seed=0, N=240, G=12, D=3):
    """Multi-donor counts with a per-donor expression shift + shared programs."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(G, 2)) * 0.4
    donor = rng.integers(0, D, size=N)
    # Per-donor baseline log-mean shift (the marginal the cascade freezes).
    donor_base = rng.normal(size=(D, G)) * 0.5 + 1.2
    z = rng.normal(size=(N, 2))
    log_rate = donor_base[donor] + z @ W.T
    counts = rng.poisson(np.exp(log_rate)).astype(np.float32)
    adata = ad.AnnData(counts)
    adata.obs["donor"] = donor.astype(str)
    return adata


def test_e2e_per_donor_mu_cascade():
    """Hierarchical source + freeze ('r','mu') routes through the per-donor path."""
    adata = _multi_donor_adata()
    D = adata.obs["donor"].nunique()
    G = adata.shape[1]

    # Hierarchical independent-gene SVI source (per-donor mean).
    svi = scribe.fit(
        adata,
        model="nbvcp",
        parameterization="standard",
        unconstrained=True,
        dataset_key="donor",
        priors={"mean_expression": {"donor": "gaussian"}},
        inference_method="svi",
        n_steps=300,
        seed=0,
    )
    # The hierarchical source exposes a per-donor (D, G) mu.
    mu_src = np.asarray(svi.get_map()["mu"])
    assert mu_src.ndim == 2 and mu_src.shape[0] == D

    # NBLN-Laplace with the per-donor marginal cascade + module-weight hierarchy.
    res = scribe.fit(
        adata,
        model="nbln",
        inference_method="laplace",
        priors={"module_weight": {"donor": "gaussian"}},
        correlate_other_column=True,
        dataset_key="donor",
        informative_priors_from=svi,
        informative_priors_freeze=("r", "mu"),
        latent_dim=2,
        n_steps=150,
        seed=0,
    )

    # Per-donor mean table surfaced; standard mu pooled to (G,).
    per_donor = res.get_gene_mean_per_dataset()
    assert per_donor is not None
    per_donor = np.asarray(per_donor)
    assert per_donor.shape == (D, G)
    mu_pooled = np.asarray(res.get_mu())
    assert mu_pooled.shape == (G,)
    # The pooled mean is the per-donor average (per-gene) of the frozen table.
    np.testing.assert_allclose(
        mu_pooled, per_donor.mean(axis=0), atol=1e-4
    )
    # The per-donor rows are genuinely distinct (the cascade is doing work).
    assert per_donor.std(axis=0).max() > 1e-3
    # The module-weight hierarchy is also active alongside the per-donor mu.
    s = res.get_module_weights()
    assert s is not None and np.asarray(s).shape == (D, 2)


def _sparse_multi_donor_adata(seed=0, N=300, G=14, D=3, n_sparse=3):
    """Multi-donor counts with a few low-coverage genes (pooled by gene_coverage).

    The last ``n_sparse`` genes are made sparse (~5% of cells expressing) so a
    ``gene_coverage < 1`` filter pools them into the trailing ``_other`` column
    — the condition that triggers the decoupled layout.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(G, 2)) * 0.4
    donor = rng.integers(0, D, size=N)
    donor_base = rng.normal(size=(D, G)) * 0.5 + 1.2
    z = rng.normal(size=(N, 2))
    counts = rng.poisson(np.exp(donor_base[donor] + z @ W.T)).astype(np.float32)
    counts[:, -n_sparse:] = counts[:, -n_sparse:] * (
        rng.random((N, n_sparse)) < 0.05
    )
    adata = ad.AnnData(counts)
    adata.obs["donor"] = donor.astype(str)
    return adata


def test_e2e_per_donor_mu_cascade_decoupled_equal_panels():
    """Per-donor mu cascade on the DECOUPLED layout with matching gene panels.

    The decoupled layout (``correlate_other_column=False``) keeps the pooled
    ``_other`` pseudo-gene OUT of the low-rank covariance ``W`` — the
    biologically-correct choice (``W`` should not waste capacity correlating an
    aggregate).  The per-donor ``mu^(d)`` cascade needs the SVI source and the
    Laplace target to share a gene panel; using the SAME ``gene_coverage`` on
    both makes the panels equal (same kept genes + same ``_other``), so the
    cascade succeeds while the covariance stays decoupled.  This is the
    equal-panel companion to ``test_per_donor_mu_panel_mismatch_unsupported``.
    """
    adata = _sparse_multi_donor_adata(seed=5)
    D = adata.obs["donor"].nunique()
    GC = 0.8  # pools the sparse genes into `_other`
    svi = scribe.fit(
        adata,
        model="nbvcp",
        parameterization="standard",
        unconstrained=True,
        dataset_key="donor",
        priors={"mean_expression": {"donor": "gaussian"}},
        inference_method="svi",
        n_steps=300,
        seed=0,
        gene_coverage=GC,
    )
    res = scribe.fit(
        adata,
        model="nbln",
        inference_method="laplace",
        priors={"module_weight": {"donor": "gaussian"}},
        correlate_other_column=False,   # DECOUPLED: `_other` excluded from W
        gene_coverage=GC,               # MATCH the source -> equal panels
        dataset_key="donor",
        informative_priors_from=svi,
        informative_priors_freeze=("r", "mu"),
        latent_dim=2,
        n_steps=150,
        seed=0,
    )
    # Decoupled layout is active.
    assert res.axis_layout.decoupled
    # Per-donor mu^(d) on the full observation axis (kept genes + `_other`).
    per_donor = np.asarray(res.get_gene_mean_per_dataset())
    g_obs = per_donor.shape[1]
    assert per_donor.shape[0] == D
    # `_other` is EXCLUDED from W: W lives on the kept-gene axis (G_kept < G_obs).
    g_kept = np.asarray(res.get_W()).shape[0]
    assert g_kept < g_obs, (g_kept, g_obs)
    # The module-weight hierarchy still fits per-leaf module weights.
    s = res.get_module_weights()
    assert s is not None and np.asarray(s).shape == (D, 2)


def test_per_donor_mu_panel_mismatch_unsupported():
    """Per-donor mu freeze fails clearly when source/target panels differ.

    The per-donor ``mu^(d)`` extractor aligns source leaf means to target genes
    and requires EQUAL gene panels.  Here the SVI source uses the full panel
    while the Laplace target pools genes with ``gene_coverage=0.5`` (a strict
    subset + ``_other``), so there is no per-gene correspondence — the cascade
    must raise a clear error rather than mis-align.  This is a PANEL-MISMATCH
    constraint, NOT a layout one: with matching ``gene_coverage`` the same
    decoupled fit succeeds (see
    ``test_e2e_per_donor_mu_cascade_decoupled_equal_panels``).
    """
    adata = _multi_donor_adata(seed=5)
    svi = scribe.fit(
        adata,
        model="nbvcp",
        parameterization="standard",
        unconstrained=True,
        dataset_key="donor",
        priors={"mean_expression": {"donor": "gaussian"}},
        inference_method="svi",
        n_steps=150,
        seed=0,
    )
    with pytest.raises((NotImplementedError, Exception), match="equal"):
        scribe.fit(
            adata,
            model="nbln",
            inference_method="laplace",
            priors={"module_weight": {"donor": "gaussian"}},
            correlate_other_column=False,
            gene_coverage=0.5,              # subset panel != full-panel source
            dataset_key="donor",
            informative_priors_from=svi,
            informative_priors_freeze=("r", "mu"),
            latent_dim=2,
            n_steps=50,
            seed=0,
        )


def test_e2e_per_donor_mu_requires_hierarchy():
    """Freezing per-donor mu without the module-weight hierarchy fails fast."""
    adata = _multi_donor_adata(seed=3)
    svi = scribe.fit(
        adata,
        model="nbvcp",
        parameterization="standard",
        unconstrained=True,
        dataset_key="donor",
        priors={"mean_expression": {"donor": "gaussian"}},
        inference_method="svi",
        n_steps=150,
        seed=0,
    )
    # No module-weight hierarchy => the per-donor (D, G) freeze has no
    # per-cell-W path; the obs model must reject it rather than mis-broadcast.
    with pytest.raises((NotImplementedError, Exception)):
        scribe.fit(
            adata,
            model="nbln",
            inference_method="laplace",
            correlate_other_column=True,
            dataset_key="donor",
            informative_priors_from=svi,
            informative_priors_freeze=("mu",),
            latent_dim=2,
            n_steps=50,
            seed=0,
        )
