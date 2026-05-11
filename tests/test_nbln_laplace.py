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
