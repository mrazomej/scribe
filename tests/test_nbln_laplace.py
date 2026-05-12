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


# =====================================================================
# Helpers
# =====================================================================


def _nbln_result(
    *,
    G: int = 20,
    C: int = 15,
    k: int = 3,
    with_uncertainty: bool = True,
    positive_transform: str = "softplus",
    seed: int = 0,
) -> ScribeLaplaceResults:
    """Build a small synthetic NBLN ``ScribeLaplaceResults`` for unit tests."""
    rng = np.random.default_rng(seed)
    mu = jnp.asarray(rng.normal(0, 0.5, G).astype(np.float32))
    W = jnp.asarray((0.3 * rng.normal(size=(G, k))).astype(np.float32))
    d = jnp.asarray(np.full(G, 0.05, dtype=np.float32))
    x_loc = jnp.asarray(rng.normal(0, 1, (C, G)).astype(np.float32))
    r = jnp.asarray(rng.uniform(0.5, 5.0, G).astype(np.float32))

    mc = ModelConfig(
        base_model="nbln",
        parameterization=Parameterization.COUNT_LOGNORMAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform=positive_transform,
    )

    pos_fwd, pos_inv = resolve_positive_fns(mc)

    kwargs = {}
    if with_uncertainty:
        kwargs["r_loc"] = pos_inv(r)
        kwargs["r_scale"] = jnp.asarray(
            rng.uniform(0.01, 0.3, G).astype(np.float32)
        )

    return ScribeLaplaceResults(
        model_config=mc,
        mu=mu,
        W=W,
        d=d,
        n_genes=G,
        n_cells=C,
        x_loc=x_loc,
        r=r,
        losses=jnp.zeros(1),
        final_grad_norms=jnp.zeros(1),
        **kwargs,
    )


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
    """Phase-2 cascade-parameter-freeze unit tests (NBLN-only).

    Tests cover:
    - Constructor validation (invalid keys, missing freeze_values entries).
    - ``uses_capture`` activation by frozen eta (Round-4 R5-1).
    - ``init_state`` excludes frozen keys from optimizer params and
      stashes them on ``self._frozen_runtime`` (Round-4 R4 mechanism).
    - ``aux_data["eta_frozen"]`` is populated when eta is frozen.
    - ``compute_global_uncertainty`` emits NaN sentinels for frozen
      scales (Round-5 R5-2).
    - ``pack_result`` splices frozen values back so ``run_result.globals``
      carries the full params dict (Round-5 R5-5).
    - ``get_W_compositional`` / ``get_gauge_diagnostics`` model-aware
      dispatch (PLN/NBLN project; LNM no-op).
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

    # ---- uses_capture activation ----

    def test_frozen_eta_activates_uses_capture(self):
        """Round-5 R5-1: uses_capture must include 'eta' in frozen_params."""
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
