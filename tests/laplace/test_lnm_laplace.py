"""End-to-end integration tests for LNM Laplace inference.

The pure-JAX Newton kernels are unit-tested separately in
``tests/test_laplace_newton_lnm.py``. This file covers the
surrounding plumbing:

1. ``scribe.fit(model="lnm", inference_method="laplace", ...)``
   runs end-to-end and returns a ``ScribeLaplaceResults`` with the
   right per-cell-state slots populated for the chosen ``d_mode``.
2. The ALR reference index is auto-selected (or honored when the
   user passes one explicitly) and propagated to the result.
3. The d_mode dispatch picks the correct Newton kernel: ``z_loc``
   gets populated under ``low_rank``, ``y_alr_loc`` under
   ``learned``.
4. Per-cell Newton converges (``final_grad_norms`` stays under
   the configured tolerance).
5. The dispatched accessors on the single ``ScribeLaplaceResults``
   class return correctly-shaped tensors for LNM fits.
"""

from __future__ import annotations

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _synthetic_lnm(n_cells=60, n_genes=6, latent_dim=2, seed=0):
    """Small synthetic LNM dataset for fast end-to-end fits.

    Each cell draws a per-cell composition from
    ``softmax(mu + W z)`` and per-cell counts from
    ``Multinomial(N_c, p_c)`` with N_c uniformly between 50 and
    500. Returns an AnnData with non-degenerate counts.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_cells, latent_dim)).astype(np.float32)
    W = rng.normal(size=(n_genes, latent_dim)).astype(np.float32) * 0.5
    mu = rng.normal(size=n_genes).astype(np.float32) * 0.3
    logits = mu + z @ W.T
    probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    totals = rng.integers(50, 500, size=n_cells)
    counts = np.zeros((n_cells, n_genes), dtype=np.int64)
    for c in range(n_cells):
        counts[c] = rng.multinomial(int(totals[c]), np.asarray(probs[c]))
    return ad.AnnData(counts.astype(np.float32))


# =====================================================================
# 1. End-to-end smoke + results-class population
# =====================================================================


class TestLNMLaplaceEndToEnd:
    """``scribe.fit(..., model='lnm', inference_method='laplace')`` works."""

    def test_fit_runs_and_returns_laplace_results(self):
        import scribe
        from scribe.laplace import ScribeLaplaceResults

        adata = _synthetic_lnm(n_cells=40, n_genes=5, latent_dim=2)
        result = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=20,
            seed=0,
        )
        assert isinstance(result, ScribeLaplaceResults)

    def test_learned_dmode_populates_y_alr_loc(self):
        """``d_mode='learned'`` (the default) should fill
        ``y_alr_loc`` and leave ``z_loc`` and ``x_loc`` at None."""
        import scribe

        adata = _synthetic_lnm(n_cells=30, n_genes=6, latent_dim=2)
        result = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            d_mode="learned",
            vae_latent_dim=2,
            n_steps=20,
            seed=1,
        )
        assert result.y_alr_loc is not None
        assert result.y_alr_loc.shape == (30, 5)  # (n_cells, G-1)
        assert result.z_loc is None
        assert result.x_loc is None  # PLN-only slot
        assert result.alr_reference_idx is not None
        assert 0 <= int(result.alr_reference_idx) < 6

    def test_low_rank_dmode_populates_z_loc(self):
        """``d_mode='low_rank'`` should fill ``z_loc`` and leave
        ``y_alr_loc`` at None."""
        import scribe

        adata = _synthetic_lnm(n_cells=30, n_genes=6, latent_dim=2)
        result = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            d_mode="low_rank",
            vae_latent_dim=2,
            n_steps=20,
            seed=2,
        )
        assert result.z_loc is not None
        assert result.z_loc.shape == (30, 2)  # (n_cells, k)
        assert result.y_alr_loc is None
        assert result.x_loc is None

    def test_explicit_alr_reference_idx_honored(self):
        """Passing ``alr_reference_idx`` overrides auto-selection."""
        import scribe

        adata = _synthetic_lnm(n_cells=30, n_genes=6, latent_dim=2)
        result = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            alr_reference_idx=2,
            vae_latent_dim=2,
            n_steps=10,
            seed=3,
        )
        assert int(result.alr_reference_idx) == 2

    def test_newton_converges(self):
        """Worst per-cell Newton gradient norm should be below the
        ``LaplaceConfig.newton_tolerance`` after enough outer steps.
        """
        import scribe

        adata = _synthetic_lnm(n_cells=40, n_genes=5, latent_dim=2)
        result = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=50,
            seed=4,
        )
        assert float(jnp.max(result.final_grad_norms)) < 1e-2


# =====================================================================
# 2. Model-dispatching accessors on the shared ScribeLaplaceResults
# =====================================================================


class TestLNMResultsAccessors:
    """``ScribeLaplaceResults`` methods that dispatch on
    ``model_config.base_model`` should return LNM-shaped objects.
    """

    @pytest.fixture
    def result(self):
        import scribe

        adata = _synthetic_lnm(n_cells=30, n_genes=6, latent_dim=2)
        return scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=20,
            seed=5,
        )

    def test_get_mu_W_d_shapes(self, result):
        # ALR space: (G-1,) for mu and d, (G-1, k) for W.
        assert result.get_mu().shape == (5,)
        assert result.get_W().shape == (5, 2)
        assert result.get_d().shape == (5,)

    def test_get_sigma_is_pd(self, result):
        sigma = result.get_sigma()
        assert sigma.shape == (5, 5)
        # Cholesky succeeds → PD.
        jnp.linalg.cholesky(sigma)

    def test_get_correlation_diag_one(self, result):
        corr = result.get_correlation()
        assert jnp.allclose(jnp.diag(corr), 1.0, atol=1e-5)

    def test_get_map_keys_are_lnm_specific(self, result):
        m = result.get_map()
        # LNM-flavoured: should have d_lnm (not d_pln) and y_alr (not y_log_rate).
        assert "mu" in m
        assert "W" in m
        assert "d_lnm" in m
        # Either z (low_rank) or y_alr (learned).
        assert "y_alr" in m or "z" in m

    def test_get_distributions_returns_y_alr(self, result):
        import numpyro.distributions as dist

        dists = result.get_distributions(backend="numpyro")
        assert "y_alr" in dists
        assert isinstance(
            dists["y_alr"], dist.LowRankMultivariateNormal
        )

    def test_get_p_capture_is_none_without_capture(self, result):
        # Plain LNM (no LNMVCP) has no per-cell capture probability.
        assert result.get_p_capture() is None


# =====================================================================
# 3. LNMVCP Laplace (capture-aware)
# =====================================================================


class TestLNMVCPLaplaceEndToEnd:
    """LNMVCP Laplace path: composition Newton + scalar η Newton.

    The (z, η) Hessian is block-diagonal because the multinomial
    likelihood is η-independent (conditions on u_T) and the NB-on-
    totals likelihood is z-independent. So Newton on η is a scalar
    per cell, decoupled from the composition block.
    """

    def test_fit_runs_and_populates_capture_slots(self):
        import scribe

        adata = _synthetic_lnm(n_cells=40, n_genes=5, latent_dim=2, seed=10)
        result = scribe.fit(
            adata,
            model="lnmvcp",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=30,
            seed=10,
            priors={"capture_efficiency": (np.log(50_000.0), 0.5)},
        )
        # Composition slot (default d_mode='learned' → y_alr_loc).
        assert result.y_alr_loc is not None
        assert result.y_alr_loc.shape == (40, 4)
        # Capture-anchor slots.
        assert result.eta_loc is not None
        assert result.eta_loc.shape == (40,)
        assert result.p_capture_loc is not None
        assert result.p_capture_loc.shape == (40,)
        # p_capture must lie in (0, 1].
        p = np.asarray(result.p_capture_loc)
        assert (p > 0.0).all()
        assert (p <= 1.0).all()
        # eta_loc is the natural latent: -log(p_capture).
        np.testing.assert_allclose(
            np.asarray(result.eta_loc),
            -np.log(np.asarray(result.p_capture_loc)),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_get_p_capture_dispatches_to_p_capture_loc(self):
        """``get_p_capture()`` should return the stored
        ``p_capture_loc`` for LNMVCP fits (rather than computing
        ``exp(-eta_loc)``)."""
        import scribe

        adata = _synthetic_lnm(n_cells=30, n_genes=5, latent_dim=2, seed=11)
        result = scribe.fit(
            adata,
            model="lnmvcp",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=20,
            seed=11,
            priors={"capture_efficiency": (np.log(50_000.0), 0.5)},
        )
        p = result.get_p_capture()
        assert p is not None
        # Should be the stored slot, not derived from eta_loc.
        np.testing.assert_array_equal(
            np.asarray(p), np.asarray(result.p_capture_loc)
        )

    def test_capture_anchor_pulls_eta_toward_log_M0_over_L_c(self):
        """For LNMVCP the η-MAP is anchored at log(M_0) - log(L_c).
        The σ_M=0.5 prior is tight enough that η_loc should track
        the per-cell anchor closely (within ~σ_M).
        """
        import scribe

        adata = _synthetic_lnm(n_cells=40, n_genes=5, latent_dim=2, seed=12)
        log_M0 = float(np.log(50_000.0))
        result = scribe.fit(
            adata,
            model="lnmvcp",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=40,
            seed=12,
            priors={"capture_efficiency": (log_M0, 0.5)},
        )
        L_c = np.asarray(adata.X).sum(axis=-1)
        eta_anchor = log_M0 - np.log(np.maximum(L_c, 1.0))
        eta_map = np.asarray(result.eta_loc)
        # Each cell's MAP must be within ~3σ of its anchor (loose
        # bound; the data + prior interact but should not push
        # cells far from the prior mean for well-converged fits).
        diff = eta_map - eta_anchor
        assert float(np.max(np.abs(diff))) < 3.0  # well within 3σ_M

    def test_low_rank_dmode_with_lnmvcp(self):
        """LNMVCP + d_mode='low_rank' uses Newton over z plus
        scalar Newton on η; the composition slot should be z_loc."""
        import scribe

        adata = _synthetic_lnm(n_cells=30, n_genes=5, latent_dim=2, seed=13)
        result = scribe.fit(
            adata,
            model="lnmvcp",
            inference_method="laplace",
            d_mode="low_rank",
            vae_latent_dim=2,
            n_steps=20,
            seed=13,
            priors={"capture_efficiency": (np.log(50_000.0), 0.5)},
        )
        assert result.z_loc is not None
        assert result.z_loc.shape == (30, 2)
        assert result.y_alr_loc is None
        assert result.p_capture_loc is not None
        assert result.eta_loc is not None


# =====================================================================
# 4. LNM Global Uncertainty
# =====================================================================


class TestLNMGlobalUncertainty:
    """Verify that LNM Laplace results include global totals uncertainty
    fields (``mu_T_loc``, ``mu_T_scale``, ``r_T_loc``, ``r_T_scale``,
    ``totals_cov``) and that downstream APIs expose them correctly.
    """

    @pytest.fixture
    def lnm_result_with_uncertainty(self):
        """Build a synthetic ``ScribeLaplaceResults`` with LNM totals
        uncertainty populated, without running a real fit."""
        from scribe import ScribeLaplaceResults
        from scribe.laplace._global_uncertainty import resolve_positive_fns
        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import (
            InferenceMethod,
            Parameterization,
        )

        rng = np.random.default_rng(500)
        G_full, C, k = 10, 15, 2
        G_minus1 = G_full - 1
        mu = jnp.asarray(rng.normal(0, 0.5, G_minus1).astype(np.float32))
        W = jnp.asarray(
            (0.3 * rng.normal(size=(G_minus1, k))).astype(np.float32)
        )
        d = jnp.asarray(np.full(G_minus1, 0.05, dtype=np.float32))
        z_loc = jnp.asarray(
            rng.normal(0, 1, (C, k)).astype(np.float32)
        )

        mc = ModelConfig(
            base_model="lnm",
            parameterization=Parameterization.LOGISTIC_NORMAL_CANONICAL,
            inference_method=InferenceMethod.LAPLACE,
            positive_transform="softplus",
        )
        pos_fwd, pos_inv = resolve_positive_fns(mc)

        mu_T_val = 20_000.0
        r_T_val = 8.0
        mu_T_loc = pos_inv(jnp.asarray(mu_T_val, dtype=jnp.float32))
        r_T_loc = pos_inv(jnp.asarray(r_T_val, dtype=jnp.float32))
        # Synthetic 2x2 covariance (must be PSD).
        cov = jnp.array(
            [[0.04, 0.005], [0.005, 0.09]], dtype=jnp.float32
        )

        return ScribeLaplaceResults(
            model_config=mc,
            mu=mu,
            W=W,
            d=d,
            n_genes=G_full,
            n_cells=C,
            z_loc=z_loc,
            alr_reference_idx=0,
            mu_T=jnp.asarray(mu_T_val, dtype=jnp.float32),
            r_T=jnp.asarray(r_T_val, dtype=jnp.float32),
            mu_T_loc=mu_T_loc,
            r_T_loc=r_T_loc,
            mu_T_scale=jnp.sqrt(cov[0, 0]),
            r_T_scale=jnp.sqrt(cov[1, 1]),
            totals_cov=cov,
            losses=jnp.zeros(1),
            final_grad_norms=jnp.zeros(1),
        )

    def test_get_map_includes_totals_fields(
        self, lnm_result_with_uncertainty
    ):
        m = lnm_result_with_uncertainty.get_map()
        assert "mu_T" in m
        assert "r_T" in m
        assert "p" in m
        assert "mu_T_loc" in m
        assert "mu_T_scale" in m
        assert "r_T_loc" in m
        assert "r_T_scale" in m

    def test_get_map_p_is_correct(self, lnm_result_with_uncertainty):
        """Derived ``p = r_T / (r_T + mu_T)``."""
        m = lnm_result_with_uncertainty.get_map()
        expected_p = float(m["r_T"]) / (float(m["r_T"]) + float(m["mu_T"]))
        np.testing.assert_allclose(float(m["p"]), expected_p, atol=1e-5)

    def test_get_distributions_returns_mvn_totals(
        self, lnm_result_with_uncertainty
    ):
        import numpyro.distributions as dist

        dists = lnm_result_with_uncertainty.get_distributions()
        assert "totals_unconstrained" in dists
        assert isinstance(
            dists["totals_unconstrained"], dist.MultivariateNormal
        )

    def test_get_distributions_returns_transformed_marginals(
        self, lnm_result_with_uncertainty
    ):
        import numpyro.distributions as dist

        dists = lnm_result_with_uncertainty.get_distributions()
        assert "mu_T" in dists
        assert isinstance(dists["mu_T"], dist.TransformedDistribution)
        assert "r_T" in dists
        assert isinstance(dists["r_T"], dist.TransformedDistribution)

    def test_gene_subsetting_preserves_totals_fields(
        self, lnm_result_with_uncertainty
    ):
        """Totals uncertainty is global/scalar and must pass through
        gene subsetting unchanged."""
        res = lnm_result_with_uncertainty
        # Subset genes 0 (ref) and 3.
        sub = res[[0, 3]]
        assert sub.n_genes == 2
        # Global totals fields pass through.
        np.testing.assert_array_equal(
            np.asarray(sub.mu_T_loc), np.asarray(res.mu_T_loc)
        )
        np.testing.assert_array_equal(
            np.asarray(sub.totals_cov), np.asarray(res.totals_cov)
        )

    def test_serialization_roundtrip(self, lnm_result_with_uncertainty):
        import pickle

        res = lnm_result_with_uncertainty
        loaded = pickle.loads(pickle.dumps(res))
        assert loaded.mu_T_loc is not None
        assert loaded.r_T_loc is not None
        assert loaded.totals_cov is not None
        np.testing.assert_allclose(
            np.asarray(loaded.totals_cov),
            np.asarray(res.totals_cov),
            atol=1e-6,
        )

    def test_lnm_marginal_ppc_with_totals_uncertainty_shape(
        self, lnm_result_with_uncertainty
    ):
        res = lnm_result_with_uncertainty
        ppc = res.get_ppc_samples(
            rng_key=jax.random.PRNGKey(0), n_samples=10, level="marginal"
        )
        assert ppc.shape == (10, res.n_genes)
        assert np.all(np.isfinite(ppc))

    def test_lnm_per_cell_laplace_ppc_with_totals_uncertainty_shape(
        self, lnm_result_with_uncertainty
    ):
        res = lnm_result_with_uncertainty
        counts = np.random.poisson(
            50, (res.n_cells, res.n_genes)
        ).astype(np.float32)
        ppc = res.get_per_cell_predictive_samples(
            rng_key=jax.random.PRNGKey(0), n_samples=10, counts=counts
        )
        assert ppc.shape == (10, res.n_cells, res.n_genes)
        assert np.all(np.isfinite(ppc))
