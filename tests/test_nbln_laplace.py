"""End-to-end and unit tests for NBLN Laplace inference with global uncertainty.

This module covers:

1. End-to-end smoke tests for ``scribe.fit(model="nbln",
   inference_method="laplace")`` returning ``ScribeLaplaceResults`` with
   ``r_loc``, ``r_scale`` populated.
2. ``get_map`` / ``get_distributions`` accessor shapes and types with
   global uncertainty fields.
3. Gene subsetting preserving ``r_loc``, ``r_scale`` marginals.
4. PPC behavior: non-MAP PPCs include ``r`` uncertainty by default;
   MAP PPC stays fixed.
5. Utility-level tests for ``curvature_to_scale``, ``woodbury_inv_diag``,
   ``resolve_positive_fns``.
6. A small curvature sanity check: profiled curvature < conditional
   curvature (larger posterior scale).
7. Transform consistency: constrained ``r`` is derived via the configured
   ``positive_transform``.
8. Serialization round-trip with global uncertainty fields.
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scribe import ScribeLaplaceResults
from scribe.laplace._global_uncertainty import (
    curvature_to_scale,
    resolve_positive_fns,
    woodbury_inv_diag,
)
from scribe.models.config import ModelConfig
from scribe.models.config.enums import InferenceMethod, Parameterization

# ``_nbln_result`` lives in tests/_synthetic_results.py so it can be shared with
# other suites (e.g. the viz compositional-PPC tests) without a cross-test
# relative import.  Imported as a top-level module via ``pythonpath = ["tests"]``.
from _synthetic_results import _nbln_result


# =====================================================================
# 1. Utility-level tests
# =====================================================================


class TestResolvePositiveFns:
    """``resolve_positive_fns`` returns correct forward/inverse pairs."""

    def test_softplus_roundtrip(self):
        mc = ModelConfig(
            base_model="nbln",
            parameterization=Parameterization.COUNT_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
            positive_transform="softplus",
        )
        fwd, inv = resolve_positive_fns(mc)
        x = jnp.linspace(-3.0, 5.0, 50)
        np.testing.assert_allclose(
            inv(fwd(x)), x, atol=1e-4, rtol=1e-4
        )

    def test_exp_roundtrip(self):
        mc = ModelConfig(
            base_model="nbln",
            parameterization=Parameterization.COUNT_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
            positive_transform="exp",
        )
        fwd, inv = resolve_positive_fns(mc)
        x = jnp.linspace(-3.0, 5.0, 50)
        np.testing.assert_allclose(
            inv(fwd(x)), x, atol=1e-5, rtol=1e-5
        )


class TestCurvatureToScale:
    """``curvature_to_scale`` handles positive, zero, and negative entries."""

    def test_all_positive_curvature(self):
        hess = jnp.array([4.0, 9.0, 16.0])
        scale, diag = curvature_to_scale(hess)
        np.testing.assert_allclose(scale, 1.0 / jnp.sqrt(hess), atol=1e-6)
        assert int(diag["floor_count"]) == 0

    def test_non_positive_curvature_is_floored(self):
        hess = jnp.array([4.0, -1.0, 9.0])
        scale, diag = curvature_to_scale(hess)
        assert jnp.all(jnp.isfinite(scale))
        assert jnp.all(scale > 0)
        assert int(diag["floor_count"]) >= 1

    def test_all_non_positive_curvature(self):
        hess = jnp.array([-1.0, -2.0, 0.0])
        scale, diag = curvature_to_scale(hess)
        assert jnp.all(jnp.isfinite(scale))
        assert jnp.all(scale > 0)


class TestWoodburyInvDiag:
    """``woodbury_inv_diag`` matches dense inverse diagonal."""

    def test_matches_dense_inverse(self):
        rng = np.random.default_rng(42)
        G, k = 10, 3
        W = jnp.asarray(rng.normal(size=(G, k)).astype(np.float32))
        d = jnp.asarray(rng.uniform(0.1, 1.0, G).astype(np.float32))
        # Dense reference.
        Sigma = W @ W.T + jnp.diag(d)
        ref_diag = jnp.diag(jnp.linalg.inv(Sigma))
        wb_diag = woodbury_inv_diag(W, d)
        np.testing.assert_allclose(wb_diag, ref_diag, atol=1e-4, rtol=1e-4)


# =====================================================================
# 2. Results accessor tests (get_map, get_distributions)
# =====================================================================


class TestNBLNGetMap:
    """``get_map`` for NBLN returns uncertainty fields."""

    def test_contains_r_loc_and_r_scale(self):
        res = _nbln_result()
        m = res.get_map()
        assert "r" in m
        assert "r_loc" in m
        assert "r_scale" in m
        assert m["r_loc"].shape == (res.n_genes,)
        assert m["r_scale"].shape == (res.n_genes,)

    def test_without_uncertainty_omits_scale(self):
        res = _nbln_result(with_uncertainty=False)
        m = res.get_map()
        assert "r" in m
        assert "r_loc" not in m
        assert "r_scale" not in m


class TestNBLNGetDistributions:
    """``get_distributions`` for NBLN returns proper uncertainty distributions."""

    def test_r_unconstrained_is_normal(self):
        import numpyro.distributions as dist

        res = _nbln_result()
        dists = res.get_distributions()
        assert "r_unconstrained" in dists
        assert isinstance(dists["r_unconstrained"], dist.Independent)

    def test_r_is_transformed_distribution(self):
        import numpyro.distributions as dist

        res = _nbln_result()
        dists = res.get_distributions()
        assert "r" in dists
        assert isinstance(dists["r"], dist.TransformedDistribution)

    def test_softplus_transform_used(self):
        import numpyro.distributions as dist

        res = _nbln_result(positive_transform="softplus")
        dists = res.get_distributions()
        r_dist = dists["r"]
        assert isinstance(
            r_dist.transforms[-1], dist.transforms.SoftplusTransform
        )

    def test_exp_transform_used(self):
        import numpyro.distributions as dist

        res = _nbln_result(positive_transform="exp")
        dists = res.get_distributions()
        r_dist = dists["r"]
        assert isinstance(
            r_dist.transforms[-1], dist.transforms.ExpTransform
        )

    def test_without_uncertainty_falls_back_to_delta(self):
        import numpyro.distributions as dist

        res = _nbln_result(with_uncertainty=False)
        dists = res.get_distributions()
        assert "r_unconstrained" not in dists
        assert isinstance(dists["r"], dist.Delta)


# =====================================================================
# 3. Gene subsetting
# =====================================================================


class TestNBLNGeneSubsetting:
    """Gene subsetting slices ``r_loc`` and ``r_scale``."""

    def test_subset_slices_uncertainty_fields(self):
        res = _nbln_result(G=20, C=10)
        idx = np.array([0, 5, 10, 15])
        sub = res[idx]
        assert sub.n_genes == 4
        assert sub.r.shape == (4,)
        assert sub.r_loc.shape == (4,)
        assert sub.r_scale.shape == (4,)
        np.testing.assert_array_equal(
            np.asarray(sub.r_loc), np.asarray(res.r_loc[idx])
        )

    def test_subset_without_uncertainty_is_none(self):
        res = _nbln_result(G=20, C=10, with_uncertainty=False)
        sub = res[[0, 1, 2]]
        assert sub.r_loc is None
        assert sub.r_scale is None


# =====================================================================
# 4. PPC behavior
# =====================================================================


class TestNBLNPPCBehavior:
    """Non-MAP PPCs include ``r`` uncertainty; MAP PPC stays fixed."""

    def test_map_ppc_shape(self):
        res = _nbln_result(G=10, C=8)
        ppc = res.get_map_ppc_samples(
            rng_key=jax.random.PRNGKey(0), n_samples=4
        )
        assert ppc.shape == (4, 8, 10)

    def test_marginal_ppc_shape(self):
        res = _nbln_result(G=10, C=8)
        ppc = res.get_ppc_samples(
            rng_key=jax.random.PRNGKey(0), n_samples=4, level="marginal"
        )
        assert ppc.shape == (4, 10)

    def test_per_cell_laplace_ppc_shape(self):
        res = _nbln_result(G=10, C=8)
        ppc = res.get_per_cell_predictive_samples(
            rng_key=jax.random.PRNGKey(0), n_samples=4
        )
        assert ppc.shape == (4, 8, 10)

    def test_uncertainty_widens_marginal_ppc_variance(self):
        """When ``r_scale > 0``, marginal PPC should have wider variance
        than when ``r`` is fixed (uncertainty=False)."""
        key = jax.random.PRNGKey(42)
        n_samples = 500

        # With uncertainty.
        res_unc = _nbln_result(G=10, C=8, with_uncertainty=True, seed=77)
        ppc_unc = np.asarray(
            res_unc.get_ppc_samples(
                rng_key=key, n_samples=n_samples, level="marginal"
            )
        )
        var_unc = float(ppc_unc.var(axis=0).mean())

        # Without uncertainty: build a version with r_loc/r_scale = None.
        res_fix = _nbln_result(G=10, C=8, with_uncertainty=False, seed=77)
        ppc_fix = np.asarray(
            res_fix.get_ppc_samples(
                rng_key=key, n_samples=n_samples, level="marginal"
            )
        )
        var_fix = float(ppc_fix.var(axis=0).mean())

        assert var_unc > var_fix, (
            f"Marginal PPC variance with uncertainty ({var_unc:.1f}) "
            f"should exceed fixed-r variance ({var_fix:.1f})"
        )

    def test_ppc_samples_are_finite(self):
        res = _nbln_result(G=10, C=8)
        for level in ("marginal", "per_cell"):
            ppc = res.get_ppc_samples(
                rng_key=jax.random.PRNGKey(1), n_samples=10, level=level
            )
            assert np.all(np.isfinite(ppc)), f"Non-finite PPC at level={level}"


# =====================================================================
# 5. Transform consistency
# =====================================================================


class TestTransformConsistency:
    """Constrained ``r`` is derived via the configured ``positive_transform``."""

    def test_softplus_derives_r_from_r_loc(self):
        res = _nbln_result(positive_transform="softplus")
        pos_fwd, _ = resolve_positive_fns(res.model_config)
        r_derived = pos_fwd(res.r_loc)
        np.testing.assert_allclose(
            np.asarray(r_derived), np.asarray(res.r), atol=1e-4
        )

    def test_exp_derives_r_from_r_loc(self):
        res = _nbln_result(positive_transform="exp")
        pos_fwd, _ = resolve_positive_fns(res.model_config)
        r_derived = pos_fwd(res.r_loc)
        np.testing.assert_allclose(
            np.asarray(r_derived), np.asarray(res.r), atol=1e-4
        )


# =====================================================================
# 6. Serialization round-trip
# =====================================================================


class TestNBLNSerialization:
    """Pickle round-trip preserves global uncertainty fields."""

    def test_pickle_roundtrip_preserves_uncertainty(self):
        res = _nbln_result()
        data = pickle.dumps(res)
        loaded = pickle.loads(data)
        assert loaded.r_loc is not None
        assert loaded.r_scale is not None
        np.testing.assert_array_equal(
            np.asarray(loaded.r_loc), np.asarray(res.r_loc)
        )
        np.testing.assert_array_equal(
            np.asarray(loaded.r_scale), np.asarray(res.r_scale)
        )

    def test_pickle_roundtrip_without_uncertainty(self):
        res = _nbln_result(with_uncertainty=False)
        data = pickle.dumps(res)
        loaded = pickle.loads(data)
        assert loaded.r_loc is None
        assert loaded.r_scale is None
        assert loaded.r is not None


# =====================================================================
# 7. Curvature sanity: profiled < conditional
# =====================================================================


class TestProfiledCurvatureSanity:
    """Profiled Hessian (including latent correction) must yield lower
    curvature (larger posterior scale) than the conditional Hessian that
    ignores latent uncertainty.

    This is a direct consequence of the Schur complement: the profiled
    curvature subtracts a positive correction from the conditional
    curvature, so ``H_profiled <= H_conditional`` element-wise.
    """

    def test_woodbury_inv_diag_is_positive(self):
        """The diagonal of ``(W W^T + diag(d))^{-1}`` must be positive."""
        rng = np.random.default_rng(99)
        G, k = 15, 3
        W = jnp.asarray(rng.normal(size=(G, k)).astype(np.float32))
        d = jnp.asarray(rng.uniform(0.1, 1.0, G).astype(np.float32))
        inv_diag = woodbury_inv_diag(W, d)
        assert jnp.all(inv_diag > 0), "Inverse diagonal must be positive"

    def test_profiled_curvature_lower_than_conditional(self):
        """Construct a toy case where the Schur correction is nonzero
        and verify profiled curvature < conditional curvature."""
        rng = np.random.default_rng(123)
        G = 8
        # Conditional curvature (positive).
        H_cond_diag = jnp.asarray(
            rng.uniform(5.0, 20.0, G).astype(np.float32)
        )
        # Cross-derivative (nonzero for all genes).
        H_cross = jnp.asarray(
            rng.normal(1.0, 0.5, G).astype(np.float32)
        )
        # Inverse diagonal from latent Hessian block.
        inv_lat_diag = jnp.asarray(
            rng.uniform(0.01, 0.1, G).astype(np.float32)
        )
        # Schur correction: H_cross^2 * inv_lat_diag (always >= 0).
        schur = H_cross ** 2 * inv_lat_diag
        H_profiled = H_cond_diag - schur
        # Profiled curvature should be strictly less than conditional.
        assert jnp.all(H_profiled < H_cond_diag)
        # For identified genes, profiled curvature should still be positive.
        scale_prof, _ = curvature_to_scale(H_profiled)
        scale_cond, _ = curvature_to_scale(H_cond_diag)
        # Profiled scale >= conditional scale (element-wise).
        assert jnp.all(scale_prof >= scale_cond - 1e-6)


# =====================================================================
# NBLN informative-prior cascade (SVI-results → empirical Gaussian)
# =====================================================================


class TestInformativePriorsIntegration:
    """Tests for the SVI-results → empirical-Gaussian-prior pathway
    inside ``NBLNObservationModel`` and downstream result handling."""

    def _model_config(self, transform: str = "softplus") -> ModelConfig:
        return ModelConfig(
            base_model="nbln",
            parameterization=Parameterization.COUNT_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
            positive_transform=transform,
        )

    def test_constructor_validates_keys(self):
        """Unrecognized keys raise ValueError."""
        from scribe.laplace._obs_nbln import NBLNObservationModel

        with pytest.raises(ValueError, match="unrecognized keys"):
            NBLNObservationModel(
                capture_anchor=None,
                model_config=self._model_config(),
                informative_priors={
                    "bad_key": {"loc": jnp.zeros(3), "scale": jnp.ones(3)},
                },
            )

    def test_uses_capture_activated_by_prior_eta_alone(self):
        """Round-2 Finding A: prior_eta must activate capture even
        without a scalar capture_anchor."""
        from scribe.laplace._obs_nbln import NBLNObservationModel

        N = 5
        mc = self._model_config()
        obs = NBLNObservationModel(
            capture_anchor=None,
            model_config=mc,
            informative_priors={
                "eta": {
                    "loc": jnp.full((N,), 0.5),
                    "scale": jnp.full((N,), 0.1),
                }
            },
        )
        assert obs.uses_capture is True
        # And without any eta prior, uses_capture remains False.
        obs2 = NBLNObservationModel(
            capture_anchor=None, model_config=mc, informative_priors=None
        )
        assert obs2.uses_capture is False

    def test_init_state_three_way_branching(self):
        """Round-3 Finding A: init_state must not crash on
        prior_eta + capture_anchor=None (the broken tuple unpack)."""
        from scribe.laplace._obs_nbln import NBLNObservationModel

        N, G = 4, 6
        rng = np.random.default_rng(0)
        counts = jnp.asarray(rng.integers(0, 10, (N, G)), dtype=jnp.float32)

        mc = self._model_config()
        # Case 1: prior_eta only, no capture_anchor.
        eta_loc_prior = jnp.full((N,), 0.4)
        eta_scale_prior = jnp.full((N,), 0.05)
        obs = NBLNObservationModel(
            capture_anchor=None,
            model_config=mc,
            informative_priors={
                "eta": {"loc": eta_loc_prior, "scale": eta_scale_prior},
            },
        )
        init = obs.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=2, seed=0
        )
        assert init.eta_anchor is not None
        assert "eta_scale" in init.aux_data
        np.testing.assert_allclose(
            init.aux_data["eta_scale"], eta_scale_prior
        )

        # Case 2: capture_anchor only.
        obs2 = NBLNObservationModel(
            capture_anchor=(float(jnp.log(1000.0)), 0.5),
            model_config=mc,
            informative_priors=None,
        )
        init2 = obs2.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=2, seed=0
        )
        assert init2.eta_anchor is not None
        assert "eta_scale" in init2.aux_data
        np.testing.assert_allclose(
            init2.aux_data["eta_scale"], jnp.full((N,), 0.5)
        )

        # Case 3: neither — no capture at all.
        obs3 = NBLNObservationModel(
            capture_anchor=None, model_config=mc, informative_priors=None
        )
        init3 = obs3.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=2, seed=0
        )
        assert init3.eta_anchor is None
        assert init3.eta_loc is None
        assert "eta_scale" not in init3.aux_data

    def test_init_state_r_and_mu_overrides(self):
        """Prior loc overrides the data-driven init for r and mu."""
        from scribe.laplace._obs_nbln import NBLNObservationModel

        N, G = 4, 6
        rng = np.random.default_rng(0)
        counts = jnp.asarray(rng.integers(0, 10, (N, G)), dtype=jnp.float32)
        mc = self._model_config()
        r_loc_prior = jnp.full((G,), -2.5)
        mu_prior_loc = jnp.full((G,), 1.7)
        obs = NBLNObservationModel(
            capture_anchor=None,
            model_config=mc,
            informative_priors={
                "r": {"loc": r_loc_prior, "scale": jnp.full((G,), 0.2)},
                "mu": {"loc": mu_prior_loc, "scale": jnp.full((G,), 0.3)},
            },
        )
        init = obs.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=2, seed=0
        )
        np.testing.assert_allclose(init.params["r_loc"], r_loc_prior)
        np.testing.assert_allclose(init.params["mu"], mu_prior_loc)

    def test_compute_global_uncertainty_returns_mu_fields(self):
        """``compute_global_uncertainty`` should now return mu_loc/mu_scale."""
        from scribe.laplace._obs_nbln import NBLNObservationModel

        N, G, k = 6, 5, 2
        rng = np.random.default_rng(7)
        counts = jnp.asarray(rng.integers(0, 10, (N, G)), dtype=jnp.float32)
        mc = ModelConfig(
            base_model="nbln",
            parameterization=Parameterization.COUNT_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
            positive_transform="softplus",
        )
        obs = NBLNObservationModel(capture_anchor=None, model_config=mc)
        init = obs.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=k, seed=0
        )
        gu = obs.compute_global_uncertainty(
            params=init.params,
            latent_loc=init.latent_loc,
            eta_loc=init.eta_loc,
            eta_anchor=init.eta_anchor,
            count_data=counts,
            aux_data=init.aux_data,
            model_config=mc,
        )
        assert "mu_loc" in gu and "mu_scale" in gu
        assert gu["mu_loc"].shape == (G,)
        assert gu["mu_scale"].shape == (G,)
        assert jnp.all(jnp.isfinite(gu["mu_scale"]))
        assert jnp.all(gu["mu_scale"] > 0)

    def test_compute_global_uncertainty_prior_precision_tightens_r_scale(self):
        """Adding a tight r prior should TIGHTEN the posterior r_scale."""
        from scribe.laplace._obs_nbln import NBLNObservationModel

        N, G, k = 6, 5, 2
        rng = np.random.default_rng(11)
        counts = jnp.asarray(rng.integers(0, 10, (N, G)), dtype=jnp.float32)
        mc = ModelConfig(
            base_model="nbln",
            parameterization=Parameterization.COUNT_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
            positive_transform="softplus",
        )

        obs_nopr = NBLNObservationModel(
            capture_anchor=None, model_config=mc, informative_priors=None
        )
        init_nopr = obs_nopr.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=k, seed=0
        )
        gu_nopr = obs_nopr.compute_global_uncertainty(
            params=init_nopr.params, latent_loc=init_nopr.latent_loc,
            eta_loc=None, eta_anchor=None, count_data=counts,
            aux_data=init_nopr.aux_data, model_config=mc,
        )

        obs_pr = NBLNObservationModel(
            capture_anchor=None, model_config=mc,
            informative_priors={
                "r": {
                    "loc": init_nopr.params["r_loc"],
                    "scale": jnp.full((G,), 0.05),
                }
            },
        )
        init_pr = obs_pr.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=k, seed=0
        )
        gu_pr = obs_pr.compute_global_uncertainty(
            params=init_pr.params, latent_loc=init_pr.latent_loc,
            eta_loc=None, eta_anchor=None, count_data=counts,
            aux_data=init_pr.aux_data, model_config=mc,
        )

        # Prior precision adds curvature → r_scale must decrease.
        assert jnp.all(gu_pr["r_scale"] <= gu_nopr["r_scale"] + 1e-6)
        # And the prior scale (0.05) bounds r_scale from above.
        assert jnp.all(gu_pr["r_scale"] <= 0.05 + 1e-6)

    def test_capture_aware_r_uncertainty_runs(self):
        """For capture-on, compute_global_uncertainty must succeed
        end-to-end with the joint (x, η) inverse path (Round-3 Finding B)."""
        from scribe.laplace._obs_nbln import NBLNObservationModel

        N, G, k = 8, 6, 2
        rng = np.random.default_rng(17)
        counts = jnp.asarray(
            rng.integers(1, 12, (N, G)), dtype=jnp.float32
        )
        mc = ModelConfig(
            base_model="nbln",
            parameterization=Parameterization.COUNT_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
            positive_transform="softplus",
        )

        obs = NBLNObservationModel(
            capture_anchor=(float(jnp.log(1000.0)), 0.5), model_config=mc
        )
        init = obs.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=k, seed=0
        )
        gu = obs.compute_global_uncertainty(
            params=init.params,
            latent_loc=init.latent_loc,
            eta_loc=init.eta_loc,
            eta_anchor=init.eta_anchor,
            count_data=counts,
            aux_data=init.aux_data,
            model_config=mc,
        )
        assert gu["r_scale"].shape == (G,)
        assert gu["mu_scale"].shape == (G,)
        assert jnp.all(jnp.isfinite(gu["r_scale"]))
        assert jnp.all(jnp.isfinite(gu["mu_scale"]))

    def test_global_uncertainty_uses_per_cell_eta_scale(self):
        """Regression: ``compute_global_uncertainty`` must consume the
        per-cell ``aux_data["eta_scale"]`` (soft-cascade scales from
        SVI samples) rather than a placeholder scalar.  Earlier code
        used ``self._sigma_M = 1.0`` here, which silently inflated
        ``r_scale`` and ``mu_scale`` on Phase-1 cascade fits because
        the joint (x, η) Schur correction was computed against a
        too-loose η precision.  The fix is to (a) read
        ``aux_data["eta_scale"]`` directly and (b) collapse to the
        x-only path when η is frozen.
        """
        from scribe.laplace._obs_nbln import NBLNObservationModel

        N, G, k = 8, 6, 2
        rng = np.random.default_rng(23)
        counts = jnp.asarray(rng.integers(1, 12, (N, G)), dtype=jnp.float32)
        mc = ModelConfig(
            base_model="nbln",
            parameterization=Parameterization.COUNT_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
            positive_transform="softplus",
        )
        eta_loc = jnp.asarray(rng.uniform(0.5, 1.5, (N,)), dtype=jnp.float32)

        # --- Tight σ_η: every cell uses 0.01.
        priors_tight = {
            "eta": {
                "loc": eta_loc,
                "scale": jnp.full((N,), 0.01, dtype=jnp.float32),
            }
        }
        obs_tight = NBLNObservationModel(
            capture_anchor=None, model_config=mc,
            informative_priors=priors_tight,
        )
        init_tight = obs_tight.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=k, seed=0
        )
        # Sanity check: aux_data carries the per-cell σ_η we passed in.
        assert "eta_scale" in init_tight.aux_data
        assert jnp.allclose(
            init_tight.aux_data["eta_scale"],
            jnp.full((N,), 0.01, dtype=jnp.float32),
        )
        gu_tight = obs_tight.compute_global_uncertainty(
            params=init_tight.params, latent_loc=init_tight.latent_loc,
            eta_loc=init_tight.eta_loc, eta_anchor=init_tight.eta_anchor,
            count_data=counts, aux_data=init_tight.aux_data, model_config=mc,
        )

        # --- Loose σ_η: every cell uses 1.0 (the old placeholder).
        priors_loose = {
            "eta": {
                "loc": eta_loc,
                "scale": jnp.full((N,), 1.0, dtype=jnp.float32),
            }
        }
        obs_loose = NBLNObservationModel(
            capture_anchor=None, model_config=mc,
            informative_priors=priors_loose,
        )
        init_loose = obs_loose.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=k, seed=0
        )
        gu_loose = obs_loose.compute_global_uncertainty(
            params=init_loose.params, latent_loc=init_loose.latent_loc,
            eta_loc=init_loose.eta_loc, eta_anchor=init_loose.eta_anchor,
            count_data=counts, aux_data=init_loose.aux_data, model_config=mc,
        )

        # Both finite + positive.
        assert jnp.all(jnp.isfinite(gu_tight["r_scale"]))
        assert jnp.all(jnp.isfinite(gu_loose["r_scale"]))
        assert jnp.all(gu_tight["r_scale"] > 0)
        assert jnp.all(gu_loose["r_scale"] > 0)

        # Tight σ_η means η has higher precision in the joint inverse →
        # less η-uncertainty leaks into the r profile → smaller r_scale.
        # If the bug were present, both calls would silently use σ_η=1.0
        # and r_scale would be IDENTICAL between the two configs.  With
        # the fix in place we expect a meaningful gap.
        assert not jnp.allclose(gu_tight["r_scale"], gu_loose["r_scale"]), (
            "r_scale identical across σ_η configs — compute_global_uncertainty "
            "is ignoring aux_data['eta_scale'] (bug regressed)."
        )
        assert jnp.all(gu_tight["r_scale"] <= gu_loose["r_scale"] + 1e-6)

    def test_results_propagates_mu_fields(self):
        """``ScribeLaplaceResults`` round-trip preserves mu_loc/mu_scale."""
        G, C = 6, 4
        result = _nbln_result(G=G, C=C, k=2, with_uncertainty=True)
        # Patch on mu_loc / mu_scale manually.
        from dataclasses import replace
        rng = np.random.default_rng(0)
        mu_loc = jnp.asarray(rng.normal(0, 0.3, G).astype(np.float32))
        mu_scale = jnp.asarray(
            rng.uniform(0.05, 0.4, G).astype(np.float32)
        )
        result2 = replace(result, mu_loc=mu_loc, mu_scale=mu_scale)
        # get_map exposes them.
        m = result2.get_map()
        assert "mu_loc" in m and "mu_scale" in m
        np.testing.assert_allclose(m["mu_loc"], mu_loc)
        np.testing.assert_allclose(m["mu_scale"], mu_scale)
        # Pickle round-trip preserves fields.
        round_tripped = pickle.loads(pickle.dumps(result2))
        np.testing.assert_allclose(round_tripped.mu_loc, mu_loc)
        np.testing.assert_allclose(round_tripped.mu_scale, mu_scale)

    def test_gene_subsetting_slices_mu_fields(self):
        """Subsetting must propagate mu_loc/mu_scale alongside r."""
        from dataclasses import replace
        G, C = 8, 5
        result = _nbln_result(G=G, C=C, k=2, with_uncertainty=True)
        rng = np.random.default_rng(0)
        result = replace(
            result,
            mu_loc=jnp.asarray(rng.normal(0, 0.3, G).astype(np.float32)),
            mu_scale=jnp.asarray(
                rng.uniform(0.05, 0.4, G).astype(np.float32)
            ),
        )
        idx = np.array([0, 2, 5])
        subset = result[idx]
        assert subset.mu_loc is not None and subset.mu_loc.shape == (3,)
        assert subset.mu_scale is not None and subset.mu_scale.shape == (3,)
        np.testing.assert_allclose(
            subset.mu_loc, np.asarray(result.mu_loc)[idx]
        )

    def test_scope_validation_rejects_non_nbln(self):
        """Round-4 Finding 1: scope validation runs on the resolved
        config, not the raw kwarg."""
        from types import SimpleNamespace
        from scribe.api.stages.run_inference import dispatch_inference

        ctx = SimpleNamespace()
        ctx.kwargs = {"informative_priors_from": object()}
        ctx.model_config = SimpleNamespace(base_model="lnm")
        ctx.inference_config = SimpleNamespace(
            method=SimpleNamespace(value="laplace")
        )
        with pytest.raises(ValueError, match="only supported"):
            dispatch_inference(ctx)


# =====================================================================
# Phase-2 freeze (cascade-parameter freeze + gauge-invariant accessors)
# =====================================================================


class TestFrozenCascade:
    """Cascade-parameter-freeze unit tests (NBLN-only).

    Tests cover:
    - Constructor validation (invalid keys, missing freeze_values entries).
    - ``uses_capture`` activation by frozen eta.
    - ``init_state`` excludes frozen keys from optimizer params and
      stashes them on ``self._frozen_runtime``.
    - ``aux_data["eta_frozen"]`` is populated when eta is frozen.
    - ``compute_global_uncertainty`` emits NaN sentinels for frozen
      scales.
    - ``pack_result`` splices frozen values back so ``run_result.globals``
      carries the full params dict.
    - ``get_W_compositional`` / ``get_gauge_diagnostics`` model-aware
      dispatch (PLN/NBLN project; LNM no-op).
    - Default-freeze tuple normalizes to ``()`` when no cascade source is
      supplied (so the default-on kwarg never disrupts plain Laplace fits).
    """

    def _mc(self, transform: str = "softplus") -> ModelConfig:
        return ModelConfig(
            base_model="nbln",
            parameterization=Parameterization.COUNT_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
            positive_transform=transform,
        )

    # ---- Constructor validation ----

    def test_freeze_rejects_invalid_keys(self):
        from scribe.laplace._obs_nbln import NBLNObservationModel
        with pytest.raises(ValueError, match="invalid keys"):
            NBLNObservationModel(
                capture_anchor=None,
                model_config=self._mc(),
                freeze_params=("bogus",),
                freeze_values={"bogus": {"loc": jnp.zeros(3)}},
            )

    def test_freeze_requires_corresponding_freeze_values(self):
        from scribe.laplace._obs_nbln import NBLNObservationModel
        with pytest.raises(ValueError, match="freeze_values"):
            NBLNObservationModel(
                capture_anchor=None,
                model_config=self._mc(),
                freeze_params=("r",),
                freeze_values=None,
            )
        with pytest.raises(ValueError, match="missing 'loc'"):
            NBLNObservationModel(
                capture_anchor=None,
                model_config=self._mc(),
                freeze_params=("r",),
                freeze_values={"r": {}},  # missing 'loc'
            )

    # ---- Default-freeze normalization (no cascade source) ----

    def test_default_freeze_normalizes_to_empty_without_cascade(self):
        """Without ``informative_priors_from``, the default-on
        ``informative_priors_freeze=("r", "eta")`` kwarg must normalize
        to ``()`` so plain Laplace fits are unaffected.  Regression for
        the bridge-stage gating in
        ``api/stages/run_inference.py``: without this guard, the freeze
        machinery would attempt to populate ``freeze_values`` from a
        non-existent cascade source.
        """
        import scribe
        from scribe.laplace.results import ScribeLaplaceResults

        N, G = 6, 4
        rng = np.random.default_rng(31)
        counts = jnp.asarray(rng.integers(0, 8, (N, G)), dtype=jnp.float32)
        # No informative_priors_from kwarg → freeze should normalize to ().
        result = scribe.fit(
            counts,
            model="nbln",
            inference_method="laplace",
            latent_dim=2,
            n_steps=2,
            seed=0,
        )
        assert isinstance(result, ScribeLaplaceResults)
        assert result.frozen_params == frozenset(), (
            "Default informative_priors_freeze=('r','eta') was not "
            "normalized to () in the absence of a cascade source. "
            "Plain Laplace fits must be unaffected by the default-on kwarg."
        )
        # Cascade-source fields should also be cleared.
        assert result.cascade_source is None
        assert result.cascade_source_counts is None

    # ---- uses_capture activation ----

    def test_frozen_eta_activates_uses_capture(self):
        """uses_capture must include 'eta' in frozen_params."""
        from scribe.laplace._obs_nbln import NBLNObservationModel
        N = 4
        obs = NBLNObservationModel(
            capture_anchor=None,
            model_config=self._mc(),
            freeze_params=("eta",),
            freeze_values={"eta": {"loc": jnp.full((N,), 0.5)}},
        )
        assert obs.uses_capture is True
        assert obs.freezes_eta is True

    # ---- init_state: frozen keys excluded from optimizer ----

    def test_init_state_excludes_frozen_r_from_params(self):
        from scribe.laplace._obs_nbln import NBLNObservationModel
        N, G = 4, 6
        rng = np.random.default_rng(0)
        counts = jnp.asarray(rng.integers(0, 10, (N, G)), dtype=jnp.float32)
        r_loc_frozen = jnp.full((G,), -2.0)
        obs = NBLNObservationModel(
            capture_anchor=None,
            model_config=self._mc(),
            freeze_params=("r",),
            freeze_values={"r": {"loc": r_loc_frozen}},
        )
        init = obs.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=2, seed=0
        )
        # r_loc must NOT be in the optimizer's params dict (excluded).
        assert "r_loc" not in init.params
        # mu, W, d_loc remain in params (not frozen).
        assert "mu" in init.params and "W" in init.params and "d_loc" in init.params
        # Frozen value is stashed on the obs model for splicing.
        assert "r_loc" in obs._frozen_runtime
        np.testing.assert_allclose(obs._frozen_runtime["r_loc"], r_loc_frozen)

    def test_init_state_eta_frozen_populates_aux_data(self):
        from scribe.laplace._obs_nbln import NBLNObservationModel
        N, G = 4, 6
        rng = np.random.default_rng(0)
        counts = jnp.asarray(rng.integers(0, 10, (N, G)), dtype=jnp.float32)
        eta_loc_frozen = jnp.full((N,), 0.5)
        obs = NBLNObservationModel(
            capture_anchor=None,
            model_config=self._mc(),
            freeze_params=("eta",),
            freeze_values={"eta": {"loc": eta_loc_frozen}},
        )
        init = obs.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=2, seed=0
        )
        # aux_data carries the per-cell eta offset for x-only Newton.
        assert "eta_frozen" in init.aux_data
        np.testing.assert_allclose(init.aux_data["eta_frozen"], eta_loc_frozen)
        # eta_loc on the result also populated (PPC reads from it).
        np.testing.assert_allclose(init.eta_loc, eta_loc_frozen)

    # ---- splice helper ----

    def test_splice_frozen_reconstructs_full_params(self):
        from scribe.laplace._obs_nbln import NBLNObservationModel
        N, G = 4, 6
        rng = np.random.default_rng(0)
        counts = jnp.asarray(rng.integers(0, 10, (N, G)), dtype=jnp.float32)
        obs = NBLNObservationModel(
            capture_anchor=None,
            model_config=self._mc(),
            freeze_params=("r",),
            freeze_values={"r": {"loc": jnp.full((G,), -1.5)}},
        )
        init = obs.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=2, seed=0
        )
        params_full = obs._splice_frozen(init.params)
        # Full dict includes r_loc (spliced) plus everything else.
        assert {"mu", "W", "d_loc", "r_loc"} == set(params_full.keys())
        np.testing.assert_allclose(params_full["r_loc"], jnp.full((G,), -1.5))

    # ---- compute_global_uncertainty: NaN sentinels for frozen ----

    def test_compute_global_uncertainty_emits_nan_for_frozen_r(self):
        from scribe.laplace._obs_nbln import NBLNObservationModel
        N, G, k = 6, 5, 2
        rng = np.random.default_rng(7)
        counts = jnp.asarray(rng.integers(0, 10, (N, G)), dtype=jnp.float32)
        mc = self._mc()
        obs = NBLNObservationModel(
            capture_anchor=None, model_config=mc,
            freeze_params=("r",),
            freeze_values={"r": {"loc": jnp.full((G,), -1.0)}},
        )
        init = obs.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=k, seed=0
        )
        gu = obs.compute_global_uncertainty(
            params=init.params, latent_loc=init.latent_loc,
            eta_loc=None, eta_anchor=None, count_data=counts,
            aux_data=init.aux_data, model_config=mc,
        )
        # r_scale is NaN sentinel — authoritative posterior lives at
        # cascade_source, accessed at get_distributions / PPC time.
        assert jnp.all(jnp.isnan(gu["r_scale"]))
        # mu path unaffected — finite scale.
        assert jnp.all(jnp.isfinite(gu["mu_scale"]))

    # ---- pack_result: splices frozen values into globals ----

    def test_pack_result_includes_frozen_in_globals(self):
        from scribe.laplace._obs_nbln import NBLNObservationModel
        from scribe.laplace._em import FinalSweepResult
        N, G = 4, 6
        rng = np.random.default_rng(0)
        counts = jnp.asarray(rng.integers(0, 10, (N, G)), dtype=jnp.float32)
        r_frozen = jnp.full((G,), -2.0)
        obs = NBLNObservationModel(
            capture_anchor=None,
            model_config=self._mc(),
            freeze_params=("r",),
            freeze_values={"r": {"loc": r_frozen}},
        )
        init = obs.init_state(
            count_data=counts, n_cells=N, n_genes=G, latent_dim=2, seed=0
        )
        # Fake final-sweep result.
        x_loc = jnp.zeros((N, G))
        final = FinalSweepResult(
            latent_loc=x_loc, eta_loc=None,
            final_grad_norms=jnp.zeros(N),
        )
        run_result = obs.pack_result(
            params=init.params, final=final,
            losses=np.zeros(10), n_steps_run=10,
            model_config=self._mc(), early_stopped=False,
            best_loss=0.0, stopped_at_step=10, divergence_aborted=False,
        )
        # Frozen r_loc must be present in run_result.globals so
        # _format_laplace_results (which reads g["r_loc"]) doesn't KeyError.
        assert "r_loc" in run_result.globals
        np.testing.assert_allclose(run_result.globals["r_loc"], r_frozen)
        # frozen_params is propagated.
        assert run_result.frozen_params == frozenset({"r"})

    # ---- gauge-invariant accessors ----

    def test_get_W_compositional_gene_centers_for_NBLN(self):
        result = _nbln_result(G=8, C=4, k=2, with_uncertainty=False)
        W_perp = result.get_W_compositional()
        # Each column of W_perp is gene-centered.
        col_means = jnp.mean(W_perp, axis=0)
        np.testing.assert_allclose(
            np.asarray(col_means), np.zeros_like(np.asarray(col_means)),
            atol=1e-5,
        )

    def test_get_gauge_diagnostics_keys_and_values(self):
        result = _nbln_result(G=8, C=4, k=2, with_uncertainty=False)
        diag = result.get_gauge_diagnostics()
        assert set(diag.keys()) == {
            "W_compositional_norm",
            "W_all_ones_component_norm",
            "gauge_contamination_ratio",
        }
        assert all(np.isfinite(v) for v in diag.values())
        # All values non-negative.
        assert all(v >= 0 for v in diag.values())


class _StubCascadeSource:
    """Minimal ``ScribeSVIResults``-shaped stub for PPC cascade-routing tests.

    Exposes only ``get_posterior_samples`` and ``_uses_amortized_capture``.
    Returns the configured per-sample arrays (truncated/recycled to match
    the requested ``n_samples``).
    """

    def __init__(
        self,
        r_samples: jnp.ndarray,
        mu_samples: jnp.ndarray,
        eta_samples: jnp.ndarray,
        amortized: bool = False,
    ):
        self._r = r_samples
        self._mu = mu_samples
        self._eta = eta_samples
        self._amortized = amortized

    def _uses_amortized_capture(self) -> bool:
        return self._amortized

    def get_posterior_samples(
        self,
        rng_key=None,
        n_samples: int = 100,
        counts=None,
        store_samples: bool = False,
        **_kwargs,
    ):
        def _take(arr):
            S = int(arr.shape[0])
            if n_samples <= S:
                return arr[:n_samples]
            # Tile-then-trim so the test can request more than the cache.
            reps = (n_samples + S - 1) // S
            tiled = jnp.tile(arr, (reps,) + (1,) * (arr.ndim - 1))
            return tiled[:n_samples]
        return {
            "r": _take(self._r),
            "mu": _take(self._mu),
            "eta_capture": _take(self._eta),
        }


def _frozen_nbln_result_with_cascade(
    *,
    G: int = 8,
    C: int = 5,
    k: int = 2,
    n_cascade_samples: int = 50,
    frozen: frozenset = frozenset({"r", "eta"}),
    seed: int = 0,
) -> ScribeLaplaceResults:
    """Build a frozen-cascade NBLN result with a stub ``cascade_source``.

    Mirrors the production result shape for default freeze:

    - ``frozen_params = {"r", "eta"}``
    - ``r_scale`` is the NaN sentinel (no Laplace Hessian for frozen r).
    - ``mu_loc`` / ``mu_scale`` are finite (non-frozen mu).
    - ``cascade_source`` returns plausible positive ``r``, ``mu``, and
      ``eta_capture`` samples.
    """
    rng = np.random.default_rng(seed)
    mu = jnp.asarray(rng.normal(0, 0.5, G).astype(np.float32))
    W = jnp.asarray((0.3 * rng.normal(size=(G, k))).astype(np.float32))
    d = jnp.asarray(np.full(G, 0.05, dtype=np.float32))
    x_loc = jnp.asarray(rng.normal(0, 1, (C, G)).astype(np.float32))
    r = jnp.asarray(rng.uniform(0.5, 5.0, G).astype(np.float32))
    eta_loc = jnp.asarray(rng.uniform(0.2, 1.5, C).astype(np.float32))

    mc = ModelConfig(
        base_model="nbln",
        parameterization=Parameterization.COUNT_LOGNORMAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform="softplus",
    )

    # NaN sentinel for frozen r_scale (matches production
    # compute_global_uncertainty when "r" is frozen).
    r_loc = resolve_positive_fns(mc)[1](r)
    r_scale_nan = (
        jnp.full((G,), jnp.nan)
        if "r" in frozen
        else jnp.asarray(rng.uniform(0.01, 0.2, G).astype(np.float32))
    )
    # mu_loc / mu_scale: finite if not frozen, NaN if frozen.
    mu_loc = mu
    mu_scale = (
        jnp.full((G,), jnp.nan)
        if "mu" in frozen
        else jnp.asarray(rng.uniform(0.01, 0.2, G).astype(np.float32))
    )

    # Build cascade samples.  SVI r and mu live in positive space; eta in [0,∞).
    r_svi = jnp.asarray(
        rng.lognormal(mean=0.5, sigma=0.3, size=(n_cascade_samples, G))
        .astype(np.float32)
    )
    mu_svi = jnp.asarray(
        rng.lognormal(mean=1.0, sigma=0.2, size=(n_cascade_samples, G))
        .astype(np.float32)
    )
    eta_svi = jnp.asarray(
        np.abs(rng.normal(0.6, 0.2, size=(n_cascade_samples, C)))
        .astype(np.float32)
    )
    cascade = _StubCascadeSource(r_svi, mu_svi, eta_svi)

    return ScribeLaplaceResults(
        model_config=mc,
        mu=mu,
        W=W,
        d=d,
        n_genes=G,
        n_cells=C,
        x_loc=x_loc,
        r=r,
        eta_loc=eta_loc,
        r_loc=r_loc,
        r_scale=r_scale_nan,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
        losses=jnp.zeros(1),
        final_grad_norms=jnp.zeros(1),
        frozen_params=frozen,
        cascade_source=cascade,
    )


class TestCascadeAwarePPC:
    """Phase-2 R5-2: PPC routing for cascade-frozen NBLN results.

    Production-critical regression: with default freeze ``("r", "eta")``,
    ``r_scale`` is NaN.  The legacy PPC path drew ``Normal(r_loc, NaN)``,
    producing all-NaN samples.  These tests guard the cascade-routing
    fix that draws ``r`` and ``eta`` from ``cascade_source`` instead.
    """

    def test_default_freeze_marginal_ppc_is_finite(self):
        """Marginal PPC with default freeze must NOT produce NaN counts."""
        res = _frozen_nbln_result_with_cascade()
        samples = res.get_ppc_samples(
            rng_key=jax.random.PRNGKey(1),
            n_samples=24,
            level="marginal",
        )
        assert samples.shape == (24, 8)
        assert jnp.all(jnp.isfinite(samples)), (
            "Marginal PPC produced NaN/Inf — cascade routing for frozen r/eta "
            "is not active or is broken."
        )

    def test_default_freeze_per_cell_laplace_ppc_is_finite(self):
        """Per-cell Laplace PPC with default freeze must NOT produce NaN."""
        res = _frozen_nbln_result_with_cascade()
        samples = res.get_per_cell_predictive_samples(
            rng_key=jax.random.PRNGKey(2),
            n_samples=16,
        )
        assert samples.shape == (16, 5, 8)
        assert jnp.all(jnp.isfinite(samples)), (
            "Per-cell Laplace PPC produced NaN/Inf — cascade routing for "
            "frozen r/eta is not active or is broken."
        )

    def test_marginal_ppc_eta_distribution_reflects_cascade(self):
        """Per-draw eta should span the cascade's eta posterior range.

        Indirect check: predictive counts vary across draws (rules out
        the case where eta resolution silently degenerated to a single
        value).  Combined with finiteness, this catches all-NaN and
        all-zero PPC failure modes.
        """
        res = _frozen_nbln_result_with_cascade(n_cascade_samples=200)
        samples = res.get_ppc_samples(
            rng_key=jax.random.PRNGKey(3),
            n_samples=64,
            level="marginal",
        )
        # Per-gene variance across the 64 draws should be > 0 for at
        # least one gene (sanity that the sampler is exercising
        # uncertainty rather than emitting a constant).
        gene_var = jnp.var(samples.astype(jnp.float32), axis=0)
        assert float(jnp.max(gene_var)) > 0.0

    def test_no_freeze_no_nan_when_laplace_scales_finite(self):
        """Non-frozen Laplace fit: existing path still works post-fix.

        Cascade routing is skipped when ``frozen_params`` is empty.  The
        non-frozen ``r`` sampling continues to use ``Normal(r_loc, r_scale)``
        (legacy correct behavior).  Non-frozen ``mu`` stays as a point in
        PPC — the Laplace ``mu_scale`` is gauge-slop-contaminated for NBLN
        and was never the right uncertainty source for predictive draws.
        """
        # frozen=∅, r_scale and mu_scale are finite arrays.
        res = _frozen_nbln_result_with_cascade(frozen=frozenset())
        samples = res.get_ppc_samples(
            rng_key=jax.random.PRNGKey(4),
            n_samples=12,
            level="marginal",
        )
        assert jnp.all(jnp.isfinite(samples))

    def test_non_frozen_mu_not_sampled_via_laplace_normal(self):
        """``mu`` must stay a point for non-frozen NBLN fits.

        Regression guard against re-introducing the gauge-slop blow-up:
        with non-frozen mu and a wide ``mu_scale``, the resolver must
        NOT inject per-draw mu noise into the PPC composition.
        """
        from scribe.laplace._sampling import _resolve_nbln_ppc_arrays
        res = _frozen_nbln_result_with_cascade(frozen=frozenset())
        arrays = _resolve_nbln_ppc_arrays(
            res, jax.random.PRNGKey(0), n_samples=8, per_cell=False,
        )
        # Cascade routing skipped (frozen empty), Laplace-Normal-mu
        # path also skipped → mu_samples is None, helper uses point mu.
        assert arrays["mu_samples"] is None

    def test_resolver_returns_none_for_non_frozen_without_scales(self):
        """Helper bails gracefully when neither cascade nor Laplace scale exists."""
        from scribe.laplace._sampling import _resolve_nbln_ppc_arrays
        # Build a result with no uncertainty fields at all.
        res = _nbln_result(G=6, C=4, k=2, with_uncertainty=False)
        arrays = _resolve_nbln_ppc_arrays(
            res, jax.random.PRNGKey(0), n_samples=8, per_cell=False,
        )
        assert arrays["r_samples"] is None
        assert arrays["mu_samples"] is None
        assert arrays["eta_samples"] is None

    def test_resolver_slices_cascade_samples_to_gene_subset(self):
        """Gene-subsetted result must slice cascade samples to match.

        Regression guard against the broadcast-shapes bug encountered
        when ``plot_ppc(n_genes=K)`` calls into a Laplace result that
        carries ``_subset_gene_index``: the cascade lives in the full
        gene panel; the resolver must subset it before returning.
        """
        from dataclasses import replace
        from scribe.laplace._sampling import _resolve_nbln_ppc_arrays
        # Build a full-G result with frozen r+eta and a stub cascade.
        full = _frozen_nbln_result_with_cascade(G=8, C=5, k=2)
        # Slice to a 3-gene subset, mimicking what ``__getitem__`` does
        # (the production helper additionally re-slices ``r``/``mu``/etc.;
        # for resolver-isolation we only need ``_subset_gene_index`` set).
        subset_idx = np.array([0, 3, 5])
        subsetted = replace(
            full,
            mu=full.mu[subset_idx],
            W=full.W[subset_idx, :],
            d=full.d[subset_idx],
            r=full.r[subset_idx],
            r_loc=full.r_loc[subset_idx],
            n_genes=int(subset_idx.size),
            _subset_gene_index=subset_idx,
        )
        arrays = _resolve_nbln_ppc_arrays(
            subsetted, jax.random.PRNGKey(0), n_samples=12, per_cell=False,
        )
        # Cascade r samples should match the subsetted gene count, not 8.
        assert arrays["r_samples"] is not None
        assert arrays["r_samples"].shape == (12, 3)
        # eta is per-cell, no gene slicing — shape (n_samples,) for marginal.
        assert arrays["eta_samples"] is not None
        assert arrays["eta_samples"].shape == (12,)

    def test_resolver_caps_cascade_pool_for_large_n_samples(self):
        """Large ``n_samples`` must not draw an unbounded cascade pool.

        Regression guard against the OOM blow-up when
        ``plot_ppc(level="marginal")`` inflates ``n_samples`` to
        ``n_eff × n_cells_obs``.  The resolver caps the SVI pool at
        ``_CASCADE_POOL_MAX`` and resamples-with-replacement to reach
        the requested predictive count.
        """
        from scribe.laplace._sampling import (
            _resolve_nbln_ppc_arrays,
            _CASCADE_POOL_MAX,
        )
        # Cascade source built with just ``_CASCADE_POOL_MAX`` samples:
        # if the resolver respects the cap, this is enough.  If it
        # asks for more than that, the stub recycles via tile-then-trim
        # (so the shape is still (n_samples, G)), but we verify
        # capping by asserting that the cascade was *called* with the
        # bounded count, not the full ``n_samples``.
        res = _frozen_nbln_result_with_cascade(
            G=8, C=5, k=2, n_cascade_samples=_CASCADE_POOL_MAX,
        )
        # Track the n_samples value passed into ``get_posterior_samples``.
        recorded: dict = {}
        orig = res.cascade_source.get_posterior_samples

        def _spy(rng_key=None, n_samples=100, **kw):
            recorded["n_samples"] = int(n_samples)
            return orig(rng_key=rng_key, n_samples=n_samples, **kw)

        res.cascade_source.get_posterior_samples = _spy  # type: ignore

        # Request 50_000 predictive samples — far above the pool cap.
        arrays = _resolve_nbln_ppc_arrays(
            res, jax.random.PRNGKey(0), n_samples=50_000, per_cell=False,
        )
        # The cascade should have been asked for *at most* the pool cap.
        assert recorded["n_samples"] <= _CASCADE_POOL_MAX, (
            f"Cascade pool exceeded cap: requested "
            f"{recorded['n_samples']}, cap is {_CASCADE_POOL_MAX}."
        )
        # The returned arrays still have the requested S dimension via
        # resample-with-replacement.
        assert arrays["r_samples"].shape == (50_000, 8)
        assert arrays["eta_samples"].shape == (50_000,)
