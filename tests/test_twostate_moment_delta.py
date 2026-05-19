"""Tests for the ``two_state_moment_delta`` parameterization.

Validates:

- Reparam algebra: mean AND Fano preserved by construction; NB
  limit ``delta -> 0`` recovers ``alpha = mu/excess_fano``.
- Enum, alias resolution (``'moment_delta'``, ``'delta'``), and
  the user-facing resolver.
- Builder + factory wire the two extras (excess_fano,
  inv_concentration) and the SigmoidNormal spec for delta.
- End-to-end SVI returns the expected sampled and derived sites.
- Phase-1 validator accepts the new parameterization.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scribe.models import ModelConfigBuilder, get_model_and_guide
from scribe.models.components.likelihoods.two_state import (
    _twostate_moment_delta_reparam,
)
from scribe.models.config.enums import Parameterization


N_CELLS = 8
N_GENES = 4


# ==============================================================================
# Reparam algebra
# ==============================================================================


class TestMomentDeltaReparam:
    def test_mean_preserved(self):
        mu = jnp.asarray([1.0, 5.0, 50.0, 174.0])
        F = jnp.asarray([0.5, 2.0, 0.1, 5.0])
        delta = jnp.asarray([0.02, 0.5, 0.001, 0.1])
        alpha, beta, rate, _ = _twostate_moment_delta_reparam(mu, F, delta)
        E = rate * alpha / (alpha + beta)
        np.testing.assert_allclose(np.asarray(E), np.asarray(mu), rtol=1e-4)

    def test_fano_preserved(self):
        mu = jnp.asarray([1.0, 5.0, 50.0, 174.0])
        F = jnp.asarray([0.5, 2.0, 0.1, 5.0])
        delta = jnp.asarray([0.02, 0.5, 0.001, 0.1])
        alpha, beta, rate, _ = _twostate_moment_delta_reparam(mu, F, delta)
        E = rate * alpha / (alpha + beta)
        ab = alpha + beta
        Vp = (alpha * beta) / (ab**2 * (ab + 1.0))
        Var = E + rate**2 * Vp
        excess = Var / E - 1.0
        np.testing.assert_allclose(np.asarray(excess), np.asarray(F), rtol=1e-4)

    def test_nb_limit_alpha_to_mu_over_F(self):
        """``delta -> 0`` -> alpha ~ mu / excess_fano."""
        mu = jnp.asarray([10.0])
        F = jnp.asarray([2.0])
        delta = jnp.asarray([1.0e-6])
        alpha, _, _, _ = _twostate_moment_delta_reparam(mu, F, delta)
        # NB-limit: alpha -> mu / F = 5
        assert abs(float(alpha[0]) - 5.0) / 5.0 < 1e-3

    def test_kappa_matches_inverse_delta(self):
        """The implied concentration kappa = (1-delta)/delta matches
        the original mean-Fano parameterization at the same
        (mu, excess_fano)."""
        from scribe.models.components.likelihoods.two_state import (
            _twostate_moments_reparam,
        )

        mu = jnp.asarray([10.0, 50.0])
        F = jnp.asarray([1.5, 0.5])
        delta = jnp.asarray([0.02, 0.1])
        kappa = (1.0 - delta) / delta

        a_d, b_d, r_d, _ = _twostate_moment_delta_reparam(mu, F, delta)
        a_f, b_f, r_f, _ = _twostate_moments_reparam(mu, F, kappa)

        np.testing.assert_allclose(np.asarray(a_d), np.asarray(a_f), rtol=1e-4)
        np.testing.assert_allclose(np.asarray(b_d), np.asarray(b_f), rtol=1e-4)
        np.testing.assert_allclose(np.asarray(r_d), np.asarray(r_f), rtol=1e-4)


# ==============================================================================
# Enum and resolver
# ==============================================================================


class TestMomentDeltaEnumAndResolver:
    def test_enum_value(self):
        assert (
            Parameterization.TWO_STATE_MOMENT_DELTA.value
            == "two_state_moment_delta"
        )

    @pytest.mark.parametrize("alias", ["moment_delta", "delta"])
    def test_alias_resolves(self, alias):
        assert (
            Parameterization(alias)
            == Parameterization.TWO_STATE_MOMENT_DELTA
        )

    @pytest.mark.parametrize(
        "alias", ["two_state_moment_delta", "moment_delta", "delta"]
    )
    def test_resolver_accepts(self, alias):
        from scribe.models.parameterizations import (
            resolve_user_parameterization_for_model,
        )

        assert (
            resolve_user_parameterization_for_model("twostate", alias)
            == "two_state_moment_delta"
        )


# ==============================================================================
# Builder + factory
# ==============================================================================


def _build_moment_delta_cfg(model="twostate"):
    builder = (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization("two_state_moment_delta")
        .with_inference("svi")
        .unconstrained()
    )
    if model == "twostatevcp":
        builder = builder.with_priors(p_capture=(1.0, 1.0))
    return builder.build()


class TestMomentDeltaBuilderAndFactory:
    def test_builder_extras(self):
        cfg = _build_moment_delta_cfg("twostate")
        active = cfg.active_parameters
        assert "mu" in active
        assert "excess_fano" in active
        assert "inv_concentration" in active

    def test_builder_twostatevcp(self):
        cfg = _build_moment_delta_cfg("twostatevcp")
        active = cfg.active_parameters
        assert "p_capture" in active
        assert "excess_fano" in active
        assert "inv_concentration" in active

    def test_factory_builds_runnable_model(self):
        cfg = _build_moment_delta_cfg("twostate")
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
        assert tr["inv_concentration"].get("type") == "sample"
        # Derived sites — both burst_size, k_off, and concentration
        # are deterministics in moment_delta mode.
        for det_name in (
            "alpha",
            "beta",
            "k_on",
            "k_off",
            "r_hat",
            "eta_act",
            "burst_size",
            "concentration",
            "effective_burst_size",
        ):
            assert tr[det_name].get("type") == "deterministic", det_name


# ==============================================================================
# End-to-end SVI
# ==============================================================================


class TestMomentDeltaEndToEndSVI:
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
            parameterization="moment_delta",
            inference_method="svi",
            n_steps=5,
            unconstrained=True,
        )
        assert jnp.isfinite(res.loss_history[-1])
        dists = res.get_distributions()
        assert "excess_fano" in dists
        assert "inv_concentration" in dists
        assert "mu" in dists
        # The derived sites are not guide-fit posteriors.
        assert "burst_size" not in dists
        assert "k_off" not in dists
        assert "concentration" not in dists

    def test_posterior_samples_include_derived(self):
        import scribe

        counts = self._make_counts()
        res = scribe.fit(
            counts,
            model="twostate",
            parameterization="moment_delta",
            inference_method="svi",
            n_steps=5,
            unconstrained=True,
        )
        samples = res.get_posterior_samples(
            rng_key=jax.random.PRNGKey(0), n_samples=4, counts=counts
        )
        # Sampled
        assert "excess_fano" in samples
        assert "inv_concentration" in samples
        # Derived deterministics
        assert "burst_size" in samples
        assert "k_off" in samples
        assert "concentration" in samples
        assert "r_hat" in samples


# ==============================================================================
# Phase-1 validator
# ==============================================================================


class TestMomentDeltaPhase1Validator:
    def test_accepted(self):
        cfg = _build_moment_delta_cfg("twostate")
        assert (
            cfg.parameterization
            == Parameterization.TWO_STATE_MOMENT_DELTA
        )

    def test_canonical_still_rejected(self):
        from scribe.models.config import ModelConfig

        with pytest.raises(ValueError, match="two_state"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.CANONICAL,
            )
