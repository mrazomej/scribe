"""Unit tests for empirical DE on NBLN and TSLN-Rate Laplace fits.

Closes the gap surfaced by the post-implementation audit: before the
``has_compositional_marginal`` predicate was added, ``compare()``'s
empirical short-circuit (``_compare_empirical_from_marginal``) was
gated on ``is_lnm_or_pln_results``, which excluded NBLN and the TSLN
variants. Calling ``scribe.de.compare(nbln_A, nbln_B,
method='empirical')`` would therefore fall through to the
posterior-samples path that NBLN-Laplace does not populate, and raise
``ValueError: Results object has no posterior samples`` despite the
fact that NBLN exposes a perfectly good generative-marginal sampler.

This module exercises the fix end-to-end:

* :func:`scribe.de._extract.has_compositional_marginal` returns True
  for NBLN / TSLN-Rate / TSLN-Logit (in addition to LNM / PLN) and
  False for other model types.
* ``compare(nbln_A, nbln_B, method='empirical')`` routes through the
  marginal-driven pipeline and returns a valid ``ScribeEmpiricalDEResults``.
* When both results were fit with ``gene_coverage < 1.0`` and have a
  trailing ``_other`` pooled column, the marginal-driven path auto-
  detects it via ``_gene_coverage_mask`` and drops the column from
  the CLR output — mirroring the legacy posterior-samples branch.
* The subset-aware cascade end-to-end (broad NBVCP-SVI ⊃ narrow
  NBLN-Laplace) produces DE output of the correct target shape.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from types import SimpleNamespace

from scribe.de._extract import (
    has_compositional_marginal,
    is_lnm_or_pln_results,
)


# =====================================================================
# Predicate unit tests
# =====================================================================


class TestHasCompositionalMarginal:
    """``has_compositional_marginal`` is broader than ``is_lnm_or_pln_results``."""

    @pytest.mark.parametrize(
        "base_model",
        ["lnm", "lnmvcp", "pln", "nbln", "twostate_ln_rate", "twostate_ln_logit"],
    )
    def test_returns_true_for_compositional_marginal_models(self, base_model):
        # Minimal stub with ``model_config.base_model`` and a
        # ``get_compositional_samples`` method (presence-only check).
        stub = SimpleNamespace(
            model_config=SimpleNamespace(base_model=base_model),
            get_compositional_samples=lambda *a, **kw: None,
        )
        assert has_compositional_marginal(stub) is True

    @pytest.mark.parametrize(
        "base_model", ["nbdm", "nbvcp", "twostate", "zinb"],
    )
    def test_returns_false_for_non_compositional_marginal_models(self, base_model):
        stub = SimpleNamespace(
            model_config=SimpleNamespace(base_model=base_model),
            get_compositional_samples=lambda *a, **kw: None,
        )
        assert has_compositional_marginal(stub) is False

    def test_returns_false_without_get_compositional_samples(self):
        # Even if the base model is in the allow-list, the predicate
        # also requires the sampler method.
        stub = SimpleNamespace(
            model_config=SimpleNamespace(base_model="nbln"),
        )
        assert has_compositional_marginal(stub) is False

    def test_returns_false_without_model_config(self):
        stub = SimpleNamespace(
            get_compositional_samples=lambda *a, **kw: None,
        )
        assert has_compositional_marginal(stub) is False

    def test_strictly_broader_than_is_lnm_or_pln_results(self):
        # For NBLN: marginal predicate is True, parametric predicate
        # is False (because NBLN lacks a PLN-style ALR converter).
        stub_nbln = SimpleNamespace(
            model_config=SimpleNamespace(base_model="nbln"),
            get_compositional_samples=lambda *a, **kw: None,
        )
        assert has_compositional_marginal(stub_nbln) is True
        assert is_lnm_or_pln_results(stub_nbln) is False

        # For LNM/PLN: both predicates agree.
        for bm in ("lnm", "pln"):
            stub = SimpleNamespace(
                model_config=SimpleNamespace(base_model=bm),
                get_compositional_samples=lambda *a, **kw: None,
            )
            assert has_compositional_marginal(stub) is True
            assert is_lnm_or_pln_results(stub) is True


# =====================================================================
# End-to-end DE tests on NBLN-Laplace fits
# =====================================================================


def _make_random_adata(n_cells: int, n_genes: int, seed: int):
    rng = np.random.default_rng(seed)
    counts = rng.negative_binomial(5, 0.5, size=(n_cells, n_genes)).astype(
        np.float32
    )
    return ad.AnnData(counts)


def test_compare_nbln_laplace_empirical_routes_through_marginal():
    """Regression: ``compare(nbln_A, nbln_B, method='empirical')`` works.

    Before the fix this raised ``ValueError: Results object has no
    posterior samples``.  After the fix, the empirical short-circuit
    fires (because both inputs satisfy ``has_compositional_marginal``)
    and the marginal-driven pipeline runs end-to-end.
    """
    import scribe

    adata = _make_random_adata(40, 8, seed=0)
    res_A = scribe.fit(
        adata, model="nbln", inference_method="laplace", latent_dim=2,
        n_steps=15, seed=0,
    )
    res_B = scribe.fit(
        adata, model="nbln", inference_method="laplace", latent_dim=2,
        n_steps=15, seed=1,
    )

    de = scribe.de.compare(
        res_A, res_B, method="empirical", n_samples_dirichlet=32,
    )
    assert type(de).__name__ == "ScribeEmpiricalDEResults"
    # No gene_coverage filter applied, so all genes survive in CLR.
    assert de.delta_samples.shape == (32, 8)


def test_compare_nbln_laplace_with_gene_coverage_drops_other_column():
    """DE on coverage-filtered NBLN-Laplace drops the trailing `_other`.

    The auto-mask logic in ``compare()`` detects both results carrying
    ``_gene_coverage_mask`` metadata and constructs a default
    ``gene_mask = [True]*(D-1) + [False]`` to drop the trailing
    aggregated column from the CLR output.
    """
    import scribe

    adata = _make_random_adata(40, 12, seed=2)
    # `gene_coverage < 1.0` triggers `_other` aggregation in the data
    # processing stage (see `core/gene_coverage.aggregate_counts_by_mask`).
    # The current default `correlate_other_column=True` keeps `_other`
    # in Σ (legacy path) — this matches today's NBLN behaviour and
    # works end-to-end with DE.  When Commit 2b lands the decoupled
    # math, the default flips to False and this test will need an
    # explicit `correlate_other_column=True` to keep testing the
    # legacy behaviour.
    res_A = scribe.fit(
        adata, model="nbln", inference_method="laplace", latent_dim=2,
        n_steps=15, seed=0, gene_coverage=0.85,
    )
    res_B = scribe.fit(
        adata, model="nbln", inference_method="laplace", latent_dim=2,
        n_steps=15, seed=1, gene_coverage=0.85,
    )
    assert res_A._gene_coverage_mask is not None
    assert res_B._gene_coverage_mask is not None
    # n_genes includes the trailing `_other`; CLR should drop it.
    n_kept_plus_other = int(res_A.n_genes)
    expected_de_cols = n_kept_plus_other - 1

    de = scribe.de.compare(
        res_A, res_B, method="empirical", n_samples_dirichlet=32,
    )
    assert de.delta_samples.shape == (32, expected_de_cols)
    assert len(de.gene_names) == expected_de_cols


def test_compare_nbln_subset_cascade_de_end_to_end():
    """End-to-end DE on the broad-SVI ⊃ narrow-NBLN cascade workflow.

    Stage 1: NBVCP-SVI on the full panel (gene_coverage=1.0).
    Stage 2: NBLN-Laplace on a narrower panel (gene_coverage < 1.0)
             with the SVI as informative-prior source.  The cascade
             auto-aggregates the SVI's per-gene posteriors on the
             dropped genes into the Laplace target's `_other` column.
    Stage 3: ``compare(nbln_A, nbln_B)`` runs end-to-end and returns
             CLR differences on the kept gene set (with `_other`
             dropped by the auto-mask logic).
    """
    import scribe

    adata = _make_random_adata(60, 16, seed=3)
    # Stage 1: broad NBVCP-SVI sources for each condition.
    svi_A = scribe.fit(
        adata, model="nbvcp", inference_method="svi", n_steps=20, seed=0,
        gene_coverage=1.0,
    )
    svi_B = scribe.fit(
        adata, model="nbvcp", inference_method="svi", n_steps=20, seed=1,
        gene_coverage=1.0,
    )
    # Stage 2: narrow NBLN-Laplace with cascade.  Current default
    # `correlate_other_column=True` keeps `_other` in Σ (legacy
    # path).  When Commit 2b ships the math, this test will need an
    # explicit `correlate_other_column=True` to keep testing the
    # legacy cascade behaviour.
    nbln_A = scribe.fit(
        adata, model="nbln", inference_method="laplace", latent_dim=2,
        n_steps=15, seed=0, gene_coverage=0.9,
        informative_priors_from=svi_A,
    )
    nbln_B = scribe.fit(
        adata, model="nbln", inference_method="laplace", latent_dim=2,
        n_steps=15, seed=1, gene_coverage=0.9,
        informative_priors_from=svi_B,
    )
    # When the SVI uses gene_coverage=1.0 (no `_other`) and the
    # Laplace uses gene_coverage<1.0 (yields `_other`), the cascade
    # auto-detects the subset and aggregates the dropped SVI genes
    # into the Laplace target's `_other` column.  Verify subset
    # metadata is populated.
    si_A = nbln_A._cascade_subset_info
    assert si_A is not None
    assert si_A.is_subset and not si_A.is_equal
    assert si_A.target_has_other is True
    # SVI used cov=1.0 so no source `_other`.
    assert si_A.source_has_other is False
    assert si_A.source_other_index_in_source is None
    # All dropped source genes pool into the target's trailing slot.
    assert si_A.dropped_idx_in_source.size >= 1

    # Stage 3: DE.  Auto-mask drops the trailing `_other` column.
    de = scribe.de.compare(
        nbln_A, nbln_B, method="empirical", n_samples_dirichlet=32,
    )
    n_kept_plus_other = int(nbln_A.n_genes)
    assert de.delta_samples.shape == (32, n_kept_plus_other - 1)
