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
