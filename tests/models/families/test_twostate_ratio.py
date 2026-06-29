"""Tests for the ``two_state_ratio`` parameterization.

Validates:

- The reparam helper algebra: mean-preserving, NB-limit reduces to
  the natural parameterization's NB limit, clamp behavior matches
  ``_twostate_reparam``.
- The new ``Parameterization.TWO_STATE_RATIO`` enum + ``'ratio'``
  alias resolve correctly.
- ``ModelConfigBuilder`` accepts the ratio parameterization for both
  twostate and twostatevcp.
- The factory builds the right param specs (``switching_ratio``
  instead of ``k_off``).
- A 5-step SVI run on synthetic data starts at a finite loss and
  ``get_distributions`` returns ``switching_ratio`` (not ``k_off``).
- The likelihood produces the same observation distribution as the
  natural parameterization when ``switching_ratio = k_off / k_on``
  is computed from a known ``(mu, b, k_off)`` triple.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scribe.models import ModelConfigBuilder, get_model_and_guide
from scribe.models.config.enums import Parameterization
from scribe.models.components.likelihoods.two_state import (
    _twostate_ratio_reparam,
    _twostate_reparam,
)


N_CELLS = 8
N_GENES = 4


# ==============================================================================
# Reparam algebra
# ==============================================================================


class TestTwoStateRatioReparam:
    """Algebraic checks on the (mu, b, s) → (alpha, beta, rate) map."""

    def test_mean_preserved_at_natural_values(self):
        """``rate · alpha / (alpha + beta) == mu`` for representative
        values that do NOT hit the safety clamps."""
        mu = jnp.array([5.0, 50.0, 174.0])
        b = jnp.array([1.5, 2.0, 0.8])
        s = jnp.array([10.0, 50.0, 0.5])
        alpha, beta, rate, _ = _twostate_ratio_reparam(mu, b, s)
        recovered = rate * alpha / (alpha + beta)
        np.testing.assert_allclose(
            np.asarray(recovered), np.asarray(mu), rtol=1e-5
        )

    def test_natural_rate_formula(self):
        """When no clamp activates, ``rate == mu · (1 + s)``."""
        mu = jnp.array([5.0, 50.0])
        b = jnp.array([2.0, 1.5])
        s = jnp.array([10.0, 30.0])
        _, _, rate, _ = _twostate_ratio_reparam(mu, b, s)
        np.testing.assert_allclose(
            np.asarray(rate), np.asarray(mu * (1.0 + s)), rtol=1e-5
        )

    def test_equivalent_to_natural_with_matching_k_off(self):
        """Given (mu, b, k_off), the natural reparam and the ratio
        reparam applied with ``s = k_off / (mu/b)`` should produce
        equal (alpha, beta, rate) up to floating-point round-off
        (assuming no clamp activates)."""
        mu = jnp.array([5.0, 50.0])
        b = jnp.array([2.0, 1.5])
        k_off = jnp.array([8.0, 30.0])
        s = k_off / (mu / b)

        a1, b1, r1, _ = _twostate_reparam(mu, b, k_off)
        a2, b2, r2, _ = _twostate_ratio_reparam(mu, b, s)

        np.testing.assert_allclose(np.asarray(a1), np.asarray(a2), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(b1), np.asarray(b2), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(r1), np.asarray(r2), rtol=1e-5)

    def test_clamps_preserve_mean(self):
        """When a gene's alpha would land below the lower clamp,
        the rescaled rate must still satisfy mean preservation."""
        # Tiny mu → alpha_nat = mu/b is very small and gets floored.
        mu = jnp.array([0.001])
        b = jnp.array([1.0])
        s = jnp.array([5.0])
        alpha, beta, rate, _ = _twostate_ratio_reparam(mu, b, s)
        recovered = rate * alpha / (alpha + beta)
        np.testing.assert_allclose(
            np.asarray(recovered), np.asarray(mu), rtol=1e-3
        )


# ==============================================================================
# Enum and resolver wiring
# ==============================================================================


class TestRatioEnumAndResolver:
    """``two_state_ratio`` + ``ratio`` alias resolve through the enum
    and through the user-facing parameterization resolver."""

    def test_enum_member_exists(self):
        assert Parameterization.TWO_STATE_RATIO.value == "two_state_ratio"

    def test_short_alias_resolves_to_enum(self):
        """``Parameterization("ratio")`` should yield TWO_STATE_RATIO."""
        assert Parameterization("ratio") == Parameterization.TWO_STATE_RATIO

    def test_resolver_accepts_ratio(self):
        from scribe.models.parameterizations import (
            resolve_user_parameterization_for_model,
        )

        assert (
            resolve_user_parameterization_for_model("twostate", "ratio")
            == "two_state_ratio"
        )
        assert (
            resolve_user_parameterization_for_model(
                "twostatevcp", "two_state_ratio"
            )
            == "two_state_ratio"
        )

    def test_resolver_still_accepts_natural(self):
        """The new branch must not break the existing natural path."""
        from scribe.models.parameterizations import (
            resolve_user_parameterization_for_model,
        )

        assert (
            resolve_user_parameterization_for_model("twostate", "natural")
            == "two_state_natural"
        )

    def test_resolver_rejects_dm_strings_still(self):
        from scribe.models.parameterizations import (
            resolve_user_parameterization_for_model,
        )

        with pytest.raises(ValueError, match="two_state_natural"):
            resolve_user_parameterization_for_model("twostate", "canonical")


# ==============================================================================
# Builder + factory wiring
# ==============================================================================


def _build_ratio_cfg(model="twostate"):
    builder = (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization("two_state_ratio")
        .with_inference("svi")
        .unconstrained()
    )
    if model == "twostatevcp":
        builder = builder.with_priors(p_capture=(1.0, 1.0))
    return builder.build()


class TestRatioBuilderAndFactory:
    """Builder accepts the parameterization; factory builds the right specs."""

    def test_builder_twostate(self):
        cfg = _build_ratio_cfg("twostate")
        assert cfg.base_model == "twostate"
        assert cfg.parameterization == Parameterization.TWO_STATE_RATIO
        active = cfg.active_parameters
        assert "mu" in active
        assert "burst_size" in active
        assert "switching_ratio" in active
        # ``k_off`` is in the optional set as a derived deterministic
        # under the ratio parameterization (it's still exposed for
        # posterior inspection), so it stays in ``active_parameters``.

    def test_builder_twostatevcp(self):
        cfg = _build_ratio_cfg("twostatevcp")
        assert cfg.parameterization == Parameterization.TWO_STATE_RATIO
        active = cfg.active_parameters
        assert "p_capture" in active
        assert "switching_ratio" in active

    def test_factory_builds_runnable_model(self):
        cfg = _build_ratio_cfg("twostate")
        model, guide, cfg_full = get_model_and_guide(cfg)
        from numpyro.handlers import seed, trace

        counts = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.int32)
        tr = trace(seed(model, jax.random.PRNGKey(0))).get_trace(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=cfg_full,
            counts=counts,
        )
        # ``switching_ratio`` is a sampled site; ``k_off`` is a
        # deterministic.  Both should exist in the trace.
        assert "switching_ratio" in tr
        assert "k_off" in tr
        # ``k_off`` should not be a sampled site under ratio.
        assert tr["switching_ratio"].get("type") == "sample"
        assert tr["k_off"].get("type") == "deterministic"


# ==============================================================================
# End-to-end SVI smoke
# ==============================================================================


class TestRatioEndToEndSVI:
    """A 5-step SVI run on synthetic data: finite loss, expected sites."""

    def _make_counts(self, seed=0, n_cells=40):
        # 5 genes matches the test_twostate_data_init pattern and
        # avoids the gene_coverage stage's pooled-gene re-shaping.
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
            parameterization="ratio",
            inference_method="svi",
            n_steps=10,
            unconstrained=True,
        )
        assert jnp.isfinite(res.loss_history[-1])
        dists = res.get_distributions()
        assert "switching_ratio" in dists
        assert "k_off" not in dists  # k_off is a deterministic, not a guide site
        assert "mu" in dists
        assert "burst_size" in dists
        assert "p_capture" in dists

    def test_natural_still_works_alongside(self):
        """Regression: adding ratio must not break the natural path."""
        import scribe

        counts = self._make_counts()
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="natural",
            inference_method="svi",
            n_steps=10,
            unconstrained=True,
        )
        assert jnp.isfinite(res.loss_history[-1])
        dists = res.get_distributions()
        assert "k_off" in dists
        assert "switching_ratio" not in dists


# ==============================================================================
# Phase-1 validator
# ==============================================================================


class TestRatioPhase1Validator:
    """Build-time validator should accept ``two_state_ratio`` and
    still reject DM-family parameterizations on twostate."""

    def test_ratio_accepted(self):
        # _build_ratio_cfg succeeds — itself a passing test.
        cfg = _build_ratio_cfg("twostate")
        assert cfg.parameterization == Parameterization.TWO_STATE_RATIO

    def test_canonical_still_rejected(self):
        from scribe.models.config import ModelConfig

        with pytest.raises(ValueError, match="two_state"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.CANONICAL,
            )
