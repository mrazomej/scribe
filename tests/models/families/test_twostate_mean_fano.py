"""Tests for the ``two_state_mean_fano`` parameterization.

Validates:

- The reparam algebra: mean- AND Fano-preserving by construction
  for all positive (mu, excess_fano, concentration).
- Enum, alias resolution ('mean_fano', 'fano') and resolver path.
- Builder + factory wire the two extras (excess_fano, concentration)
  in place of (burst_size, k_off).
- End-to-end SVI on twostate / twostatevcp returns the expected
  sampled and derived sites.
- Phase-1 validator accepts ``two_state_mean_fano`` and still
  rejects DM-family parameterizations on twostate.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scribe.models import ModelConfigBuilder, get_model_and_guide
from scribe.models.components.likelihoods.two_state import (
    _twostate_moments_reparam,
)
from scribe.models.config.enums import Parameterization


N_CELLS = 8
N_GENES = 4


# ==============================================================================
# Reparam algebra
# ==============================================================================


class TestMeanFanoReparam:
    """Mean and Fano are preserved by ``_twostate_moments_reparam``."""

    def test_mean_preserved(self):
        mu = jnp.asarray([1.0, 5.0, 50.0, 174.0])
        F = jnp.asarray([0.5, 2.0, 0.1, 5.0])
        kappa = jnp.asarray([0.5, 5.0, 50.0, 500.0])
        alpha, beta, rate, _ = _twostate_moments_reparam(mu, F, kappa)
        E = rate * alpha / (alpha + beta)
        np.testing.assert_allclose(np.asarray(E), np.asarray(mu), rtol=1e-4)

    def test_fano_preserved(self):
        mu = jnp.asarray([1.0, 5.0, 50.0, 174.0])
        F = jnp.asarray([0.5, 2.0, 0.1, 5.0])
        kappa = jnp.asarray([0.5, 5.0, 50.0, 500.0])
        alpha, beta, rate, _ = _twostate_moments_reparam(mu, F, kappa)
        E = rate * alpha / (alpha + beta)
        ab = alpha + beta
        Vp = (alpha * beta) / (ab**2 * (ab + 1.0))
        Var = E + rate**2 * Vp
        excess = Var / E - 1.0
        np.testing.assert_allclose(np.asarray(excess), np.asarray(F), rtol=1e-4)

    def test_nb_limit_alpha_to_mu_over_F(self):
        """``concentration → large`` → alpha ≈ mu / excess_fano."""
        mu = jnp.asarray([10.0])
        F = jnp.asarray([2.0])  # NB-limit burst_size
        kappa = jnp.asarray([1.0e6])  # very large concentration
        alpha, _, _, _ = _twostate_moments_reparam(mu, F, kappa)
        # NB-limit: alpha → mu / F = 5
        assert abs(float(alpha[0]) - 5.0) / 5.0 < 1e-3


# ==============================================================================
# Enum and resolver
# ==============================================================================


class TestMeanFanoEnumAndResolver:
    def test_enum_value(self):
        assert (
            Parameterization.TWO_STATE_MEAN_FANO.value
            == "two_state_mean_fano"
        )

    @pytest.mark.parametrize("alias", ["mean_fano", "fano"])
    def test_alias_resolves(self, alias):
        assert (
            Parameterization(alias) == Parameterization.TWO_STATE_MEAN_FANO
        )

    @pytest.mark.parametrize(
        "alias", ["two_state_mean_fano", "mean_fano", "fano"]
    )
    def test_resolver_accepts(self, alias):
        from scribe.models.parameterizations import (
            resolve_user_parameterization_for_model,
        )

        assert (
            resolve_user_parameterization_for_model("twostate", alias)
            == "two_state_mean_fano"
        )


# ==============================================================================
# Builder + factory + active parameters
# ==============================================================================


def _build_mean_fano_cfg(model="twostate"):
    builder = (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization("two_state_mean_fano")
        .with_inference("svi")
        .unconstrained()
    )
    if model == "twostatevcp":
        builder = builder.with_priors(p_capture=(1.0, 1.0))
    return builder.build()


class TestMeanFanoBuilderAndFactory:
    def test_builder_extras(self):
        cfg = _build_mean_fano_cfg("twostate")
        active = cfg.active_parameters
        assert "mu" in active
        assert "excess_fano" in active
        assert "concentration" in active
        # burst_size and k_off are DERIVED in mean_fano mode but are
        # in the mapping's optional set, so they remain in
        # active_parameters (as deterministics).

    def test_builder_twostatevcp(self):
        cfg = _build_mean_fano_cfg("twostatevcp")
        active = cfg.active_parameters
        assert "p_capture" in active
        assert "excess_fano" in active
        assert "concentration" in active

    def test_factory_builds_runnable_model(self):
        cfg = _build_mean_fano_cfg("twostate")
        model, guide, cfg_full = get_model_and_guide(cfg)
        from numpyro.handlers import seed, trace

        counts = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.int32)
        tr = trace(seed(model, jax.random.PRNGKey(0))).get_trace(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=cfg_full,
            counts=counts,
        )
        # Sampled sites
        assert tr["mu"].get("type") == "sample"
        assert tr["excess_fano"].get("type") == "sample"
        assert tr["concentration"].get("type") == "sample"
        # Derived sites
        for det_name in (
            "alpha",
            "beta",
            "k_on",
            "k_off",
            "r_hat",
            "eta_act",
            "burst_size",
            "effective_burst_size",
        ):
            assert tr[det_name].get("type") == "deterministic", det_name


# ==============================================================================
# End-to-end SVI
# ==============================================================================


class TestMeanFanoEndToEndSVI:
    def _make_counts(self, seed=0, n_cells=40):
        rng = np.random.default_rng(seed)
        per_gene = np.array([2.0, 5.0, 10.0, 50.0, 174.0])
        counts = np.stack(
            [rng.poisson(m, n_cells) for m in per_gene], axis=1
        )
        return jnp.asarray(counts, dtype=jnp.int32)

    def test_svi_runs_and_returns_distributions(self):
        import scribe

        counts = self._make_counts()
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="mean_fano",
            inference_method="svi",
            n_steps=5,
            unconstrained=True,
        )
        assert jnp.isfinite(res.loss_history[-1])
        dists = res.get_distributions()
        assert "excess_fano" in dists
        assert "concentration" in dists
        assert "mu" in dists
        assert "burst_size" not in dists  # derived, not a guide site
        assert "k_off" not in dists  # derived, not a guide site

    def test_posterior_samples_include_derived(self):
        import scribe

        counts = self._make_counts()
        res = scribe.fit(
            counts,
            model="twostate",
            parameterization="mean_fano",
            inference_method="svi",
            n_steps=5,
            unconstrained=True,
        )
        samples = res.get_posterior_samples(
            rng_key=jax.random.PRNGKey(0), n_samples=4, counts=counts
        )
        # Sampled
        assert "excess_fano" in samples
        assert "concentration" in samples
        # Derived deterministics
        assert "burst_size" in samples
        assert "k_off" in samples
        assert "r_hat" in samples


# ==============================================================================
# Phase-1 validator still accepts and still rejects DM family
# ==============================================================================


class TestMeanFanoPhase1Validator:
    def test_accepted(self):
        cfg = _build_mean_fano_cfg("twostate")
        assert (
            cfg.parameterization == Parameterization.TWO_STATE_MEAN_FANO
        )

    def test_canonical_still_rejected(self):
        from scribe.models.config import ModelConfig

        with pytest.raises(ValueError, match="two_state"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.CANONICAL,
            )
