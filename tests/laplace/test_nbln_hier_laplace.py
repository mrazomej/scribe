"""End-to-end Laplace fits of the per-donor correlation hierarchy (Rung 1).

Exercises ``scribe.fit(model="nbln", inference_method="laplace",
correlation_hierarchy="program_scales", ...)`` on small synthetic multi-donor
data:

- the fit runs end-to-end and surfaces the learned relative per-donor program
  activity ``s_d`` via ``get_program_activity()`` + ``program_scale_tau``,
  with the sum-to-zero gauge holding on the *fitted* values;
- the hierarchy is correctly inert (``program_activity is None``) for a plain
  NBLN-Laplace fit, so the default path is unchanged;
- the **decoupled** layout (``correlate_other_column=False`` with a pooled
  ``_other`` column) also supports the hierarchy (step 6): the per-cell-W
  ``*_decoupled_percellW`` Newton twins run end-to-end and surface ``s_d``.

The per-cell-W Newton kernels themselves are unit-tested in
``test_nbln_percellW_newton.py``; here we test the full Laplace integration.
"""

from __future__ import annotations

import numpy as np
import anndata as ad
import pytest

import scribe


def _multi_donor_adata(seed: int = 0, N: int = 240, G: int = 14, D: int = 4):
    """Synthetic multi-donor NBLN-ish counts with shared-W program structure."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(G, 3)) * 0.5
    s_true = np.exp(rng.normal(size=(D, 3)) * 0.8)  # distinct donor activity
    donor = rng.integers(0, D, size=N)
    z = np.stack([rng.normal(size=3) * s_true[donor[c]] for c in range(N)])
    counts = rng.poisson(np.exp(1.2 + z @ W.T)).astype(np.float32)
    adata = ad.AnnData(counts)
    adata.obs["donor"] = donor.astype(str)
    return adata


def test_hier_laplace_fit_surfaces_program_activity():
    """Fit runs end-to-end and exposes a learned, gauge-respecting s_d."""
    adata = _multi_donor_adata()
    D = adata.obs["donor"].nunique()
    K = 3
    res = scribe.fit(
        adata,
        model="nbln",
        inference_method="laplace",
        correlation_hierarchy="program_scales",
        correlate_other_column=True,  # legacy layout (decoupled unsupported)
        dataset_key="donor",
        latent_dim=K,
        n_steps=200,
        seed=0,
    )

    s = res.get_program_activity()
    assert s is not None, "program_activity should be populated"
    s = np.asarray(s)
    assert s.shape == (D, K)
    assert np.all(s > 0.0)
    # Between-donor scale learned + reported.
    assert res.program_scale_tau is not None and res.program_scale_tau > 0.0
    # Sum-to-zero scale gauge holds on the fitted values (per program).
    np.testing.assert_allclose(np.log(s).mean(axis=0), 0.0, atol=1e-4)
    # The fit actually moved s off the all-ones init (some donor spread).
    assert np.log(s).std() > 1e-4


def test_plain_nbln_laplace_has_no_program_activity():
    """A plain NBLN-Laplace fit (no hierarchy) leaves program_activity None."""
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
    assert res.get_program_activity() is None
    assert res.program_scale_tau is None


def test_decoupled_layout_hierarchy_runs():
    """correlation_hierarchy on the decoupled layout runs end-to-end (step 6)."""
    adata = _multi_donor_adata(seed=2)
    D = adata.obs["donor"].nunique()
    K = 3
    res = scribe.fit(
        adata,
        model="nbln",
        inference_method="laplace",
        correlation_hierarchy="program_scales",
        correlate_other_column=False,  # decoupled when _other present
        gene_coverage=0.5,  # pools low-coverage genes into `_other`
        dataset_key="donor",
        latent_dim=K,
        n_steps=150,
        seed=0,
    )
    # The decoupled per-cell-W twins ran and surfaced a gauge-respecting s_d.
    s = res.get_program_activity()
    assert s is not None
    s = np.asarray(s)
    assert s.shape == (D, K)
    assert np.all(s > 0.0)
    assert res.program_scale_tau is not None and res.program_scale_tau > 0.0
    np.testing.assert_allclose(np.log(s).mean(axis=0), 0.0, atol=1e-4)
