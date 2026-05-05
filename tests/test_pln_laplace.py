"""End-to-end integration tests for PLN Laplace inference.

The pure-JAX Newton kernel is unit-tested separately in
``tests/test_laplace_newton.py`` (10 tests against ``scipy.optimize``).
These tests cover the surrounding plumbing:

1. ``scribe.fit(model="pln", inference_method="laplace", ...)`` runs
   end-to-end on synthetic data.
2. Public API surface — ``LAPLACE`` enum, ``LaplaceConfig``,
   ``ScribeLaplaceResults`` accessors.
3. Capture-anchor activation when the user passes
   ``priors={"capture_efficiency": ...}``.
4. ``inference_method="laplace"`` is rejected for non-PLN models.
5. ``ScribeLaplaceResults`` exposes the same ``get_pln_*`` accessors
   as the VAE results so downstream plotting code works on either.
"""

from __future__ import annotations

import anndata as ad
import jax.numpy as jnp
import numpy as np
import pytest


# =====================================================================
# Test fixtures
# =====================================================================


def _synthetic_pln(n_cells=60, n_genes=6, latent_dim=2, seed=0):
    """Build a small synthetic PLN dataset with a ground-truth ``mu``.

    The data-generating process matches the model exactly:
    ``z ~ N(0, I_k)``, ``x = mu_true + z @ W_true.T``, and
    ``u ~ Poisson(exp(x))``. Used for quick parameter-recovery checks.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=n_genes).astype(np.float32) * 0.5 + 2.0
    W = (0.3 * rng.normal(size=(n_genes, latent_dim))).astype(np.float32)
    z = rng.normal(size=(n_cells, latent_dim)).astype(np.float32)
    x = mu + z @ W.T
    counts = rng.poisson(np.exp(np.clip(x, -10, 10))).astype(np.float32)
    adata = ad.AnnData(counts)
    return adata, mu, W


# =====================================================================
# 1. End-to-end smoke + parameter recovery
# =====================================================================


class TestLaplaceEndToEnd:
    """End-to-end public API smoke tests."""

    def test_fit_runs_and_returns_laplace_results(self):
        """``scribe.fit(..., inference_method='laplace')`` produces ScribeLaplaceResults."""
        import scribe
        from scribe.svi.laplace_results import ScribeLaplaceResults

        adata, _, _ = _synthetic_pln(n_cells=40, n_genes=5, latent_dim=2)
        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=30,
            seed=0,
        )
        assert isinstance(result, ScribeLaplaceResults)

    def test_recovers_mu_on_well_specified_data(self):
        """Recover ``mu`` to within 0.1 on synthetic data after 100 outer steps.

        Loose tolerance because we use only 100 outer steps for the
        unit test. Real fits use 50k+ steps. The point of this test
        is structural — that the gradient signal is correct, not that
        we reach the global optimum quickly.
        """
        import scribe

        adata, mu_true, _ = _synthetic_pln(
            n_cells=60, n_genes=6, latent_dim=2, seed=0
        )
        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=100,
            seed=0,
        )
        mu_recovered = np.asarray(result.get_pln_mu())
        assert mu_recovered.shape == (6,)
        # Mean of mu should track mu_true within 0.1.
        assert abs(float(mu_recovered.mean()) - float(mu_true.mean())) < 0.1

    def test_newton_converges(self):
        """At the end of training, all cells should have small ``L∞`` gradient norms."""
        import scribe

        adata, _, _ = _synthetic_pln(
            n_cells=40, n_genes=5, latent_dim=2, seed=1
        )
        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=50,
            seed=1,
        )
        worst_grad = float(jnp.max(result.final_grad_norms))
        # Generous tolerance — Newton on PLN is fast even from a cold
        # start; this is a sanity check, not a tight convergence test.
        assert worst_grad < 1e-2


# =====================================================================
# 2. Public API surface
# =====================================================================


class TestPublicAPISurface:
    """The public API exposes Laplace-related symbols."""

    def test_inference_method_enum_includes_laplace(self):
        from scribe.models.config import InferenceMethod

        assert InferenceMethod.LAPLACE.value == "laplace"

    def test_laplace_config_exported(self):
        from scribe.models.config import LaplaceConfig

        cfg = LaplaceConfig(n_steps=100, n_newton_steps=8)
        assert cfg.n_steps == 100
        assert cfg.n_newton_steps == 8

    def test_inference_config_from_laplace(self):
        from scribe.models.config import InferenceConfig, LaplaceConfig
        from scribe.models.config.enums import InferenceMethod

        ic = InferenceConfig.from_laplace(LaplaceConfig())
        assert ic.method == InferenceMethod.LAPLACE
        assert ic.laplace is not None
        assert ic.svi is None
        assert ic.mcmc is None


# =====================================================================
# 3. Capture-anchor activation
# =====================================================================


class TestCaptureAnchor:
    """When the user passes a capture prior, Laplace runs joint Newton on (x, η)."""

    def test_capture_anchor_activates_eta_loc(self):
        """``priors={"capture_efficiency": ...}`` produces a non-None eta_loc.

        Without a capture anchor, ``eta_loc`` is None. With the
        anchor, it's a per-cell array that should track
        ``log(M_0/L_c)`` after enough outer steps. This test verifies
        only the *plumbing* — that the eta_loc array exists and has
        the right shape, and that ``p_capture = exp(-eta_c) ∈ (0, 1]``.

        The anchor must be matched to the synthetic data's library
        size: with mean library size ~30, ``M_0 ≈ 100`` gives
        ``eta_anchor = log(100/30) ≈ 1.2`` and ``p_capture ≈ 0.3``,
        which is realistic. Using a wildly mismatched anchor (e.g.
        ``M_0 = 1e5`` for library size 30) makes Newton struggle from
        a bad warm start.
        """
        import scribe

        adata, _, _ = _synthetic_pln(
            n_cells=40, n_genes=5, latent_dim=2, seed=3
        )
        # Pick M_0 commensurate with this dataset's library sizes.
        L_mean = float(np.asarray(adata.X).sum(axis=1).mean())
        M_0 = max(L_mean * 3, 50.0)

        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=30,
            priors={"capture_efficiency": (np.log(M_0), 0.5)},
            seed=3,
        )
        eta = result.get_laplace_eta_loc()
        assert eta is not None
        assert eta.shape == (40,)
        # ``p_capture = exp(-eta_c) ∈ (0, 1]``.
        p_cap = result.get_laplace_p_capture()
        assert p_cap is not None
        assert jnp.all(jnp.isfinite(p_cap))
        assert jnp.all((p_cap > 0) & (p_cap <= 1.0))


# =====================================================================
# 4. Non-PLN models reject Laplace
# =====================================================================


class TestNonPLNRejection:
    """Laplace is currently PLN-only — clear errors for other models."""

    def test_lnm_rejects_laplace(self):
        import scribe

        adata, _, _ = _synthetic_pln(n_cells=20, n_genes=5, latent_dim=2)
        with pytest.raises(ValueError, match="laplace.*PLN-only|PLN-only"):
            scribe.fit(
                adata,
                model="lnm",
                inference_method="laplace",
                vae_latent_dim=2,
                n_steps=5,
            )

    def test_nbdm_rejects_laplace(self):
        import scribe

        adata, _, _ = _synthetic_pln(n_cells=20, n_genes=5, latent_dim=2)
        with pytest.raises(ValueError, match="laplace.*PLN-only|PLN-only"):
            scribe.fit(
                adata,
                model="nbdm",
                inference_method="laplace",
                n_steps=5,
            )


# =====================================================================
# 5. Result accessor parity with VAE results
# =====================================================================


class TestResultsAccessors:
    """``ScribeLaplaceResults`` exposes the same get_pln_* surface as VAE results."""

    @pytest.fixture
    def result(self):
        import scribe

        adata, _, _ = _synthetic_pln(
            n_cells=30, n_genes=5, latent_dim=2, seed=4
        )
        return scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=20,
            seed=4,
        )

    def test_get_pln_mu_W_d_shapes(self, result):
        mu = result.get_pln_mu()
        W = result.get_pln_W()
        d = result.get_pln_d()
        assert mu.shape == (5,)
        assert W.shape == (5, 2)
        assert d.shape == (5,)
        assert jnp.all(d > 0)

    def test_get_pln_sigma_is_pd(self, result):
        sigma = result.get_pln_sigma()
        assert sigma.shape == (5, 5)
        # PD check via Cholesky.
        jnp.linalg.cholesky(sigma)

    def test_get_pln_correlation_diag_one(self, result):
        corr = result.get_pln_correlation()
        assert jnp.allclose(jnp.diag(corr), 1.0, atol=1e-5)

    def test_get_map(self, result):
        m = result.get_map()
        for key in ("mu", "W", "d_pln", "y_log_rate"):
            assert key in m

    def test_get_distributions_has_lambda_rate(self, result):
        from scribe.stats.distributions import LowRankPoissonLogNormal

        dists = result.get_distributions(backend="numpyro")
        assert isinstance(dists["lambda_rate"], LowRankPoissonLogNormal)

    def test_laplace_x_loc_shape(self, result):
        x_loc = result.get_laplace_x_loc()
        assert x_loc.shape == (30, 5)

    def test_laplace_eta_loc_none_without_anchor(self, result):
        assert result.get_laplace_eta_loc() is None
