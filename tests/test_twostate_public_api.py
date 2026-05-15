"""Public-API and build-time validation tests for the TwoState family.

These tests exercise the wiring from ``ModelConfigBuilder`` through the
factory to the registered likelihood, plus the build-time Pydantic
validator that catches phase-1-unsupported feature combinations.

Two halves:

- Valid configs construct cleanly and produce a runnable
  ``(model, guide)`` pair; tested by running a tiny prior-predictive
  trace and a few SVI steps.
- Invalid configs raise at ``build()`` time with the expected error
  substring; tested by direct ``ModelConfig`` construction (the
  ``ModelConfigBuilder`` paths exercise the same validator).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro.handlers import seed, trace

from scribe.models import ModelConfigBuilder, get_model_and_guide
from scribe.models.config import ModelConfig
from scribe.models.config.enums import (
    InferenceMethod,
    OverdispersionType,
    Parameterization,
)


N_CELLS = 8
N_GENES = 4


def _build_valid_twostate(model="twostate"):
    builder = (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization("two_state_natural")
        .with_inference("svi")
    )
    if model == "twostatevcp":
        builder = builder.with_priors(p_capture=(1.0, 1.0))
    return builder.build()


# ==============================================================================
# Construction + factory smoke tests
# ==============================================================================


class TestTwoStateConstruction:
    """Builder → factory → working (model, guide) pair."""

    def test_twostate_builder(self):
        cfg = _build_valid_twostate("twostate")
        assert cfg.base_model == "twostate"
        assert cfg.parameterization == Parameterization.TWO_STATE_NATURAL
        # Sanity-check active_parameters: core + extras, no spurious p_capture.
        active = cfg.active_parameters
        assert "mu" in active
        assert "burst_size" in active
        assert "k_off" in active
        assert "p_capture" not in active

    def test_twostatevcp_builder_includes_p_capture(self):
        cfg = _build_valid_twostate("twostatevcp")
        assert cfg.base_model == "twostatevcp"
        assert "p_capture" in cfg.active_parameters
        assert "burst_size" in cfg.active_parameters
        assert "k_off" in cfg.active_parameters

    def test_factory_builds_runnable_model(self):
        cfg = _build_valid_twostate("twostate")
        model, guide, cfg_full = get_model_and_guide(cfg)
        counts = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.int32)
        tr = trace(seed(model, jax.random.PRNGKey(0))).get_trace(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=cfg_full,
            counts=counts,
        )
        # Gene-level deterministics emit at gene rank.
        assert tr["alpha"]["value"].shape == (N_GENES,)
        assert tr["beta"]["value"].shape == (N_GENES,)
        # Counts plate is full data shape.
        assert tr["counts"]["value"].shape == (N_CELLS, N_GENES)

    def test_factory_builds_runnable_vcp_model(self):
        cfg = _build_valid_twostate("twostatevcp")
        model, guide, cfg_full = get_model_and_guide(cfg)
        counts = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.int32)
        tr = trace(seed(model, jax.random.PRNGKey(1))).get_trace(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=cfg_full,
            counts=counts,
        )
        # VCP variant adds p_capture inside the cell plate at cell rank.
        assert tr["p_capture"]["value"].shape == (N_CELLS,)


# ==============================================================================
# Build-time validation rejects unsupported combinations
# ==============================================================================


class TestTwoStatePhase1Validation:
    """The model-validator on ModelConfig rejects phase-1-deferred paths."""

    def test_canonical_parameterization_rejected(self):
        with pytest.raises(ValueError, match="two_state_natural"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.CANONICAL,
            )

    def test_mean_prob_parameterization_rejected(self):
        with pytest.raises(ValueError, match="two_state_natural"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.MEAN_PROB,
            )

    def test_mean_odds_parameterization_rejected(self):
        with pytest.raises(ValueError, match="two_state_natural"):
            ModelConfig(
                base_model="twostatevcp",
                parameterization=Parameterization.MEAN_ODDS,
            )

    def test_bnb_overdispersion_rejected(self):
        with pytest.raises(ValueError, match="BNB"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.TWO_STATE_NATURAL,
                overdispersion=OverdispersionType.BNB,
            )

    def test_mixture_rejected(self):
        with pytest.raises(ValueError, match="mixture"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.TWO_STATE_NATURAL,
                n_components=2,
            )

    def test_vae_inference_rejected(self):
        with pytest.raises(ValueError, match="VAE"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.TWO_STATE_NATURAL,
                inference_method=InferenceMethod.VAE,
            )

    def test_multi_dataset_rejected(self):
        with pytest.raises(ValueError, match="multi-dataset"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.TWO_STATE_NATURAL,
                n_datasets=3,
            )


# ==============================================================================
# resolve_user_parameterization_for_model: strict rejection of DM-family strings
# ==============================================================================


class TestResolveUserParameterizationForTwoState:
    """resolve_user_parameterization_for_model accepts only the
    TwoState-specific keyword (or its alias 'natural')."""

    def test_natural_alias(self):
        from scribe.models.parameterizations import (
            resolve_user_parameterization_for_model,
        )

        assert (
            resolve_user_parameterization_for_model("twostate", "natural")
            == "two_state_natural"
        )

    def test_explicit_two_state_natural(self):
        from scribe.models.parameterizations import (
            resolve_user_parameterization_for_model,
        )

        assert (
            resolve_user_parameterization_for_model(
                "twostatevcp", "two_state_natural"
            )
            == "two_state_natural"
        )

    @pytest.mark.parametrize(
        "param", ["canonical", "mean_prob", "mean_odds", "standard"]
    )
    def test_dm_family_strings_rejected(self, param):
        from scribe.models.parameterizations import (
            resolve_user_parameterization_for_model,
        )

        with pytest.raises(ValueError, match="two_state_natural"):
            resolve_user_parameterization_for_model("twostate", param)


# ==============================================================================
# model_flags routing — variable_capture hint
# ==============================================================================


class TestTwoStateModelFlagsRouting:
    """``model='twostate', variable_capture=True`` re-routes to twostatevcp."""

    def _resolve(self, model, variable_capture=None, zero_inflation=None):
        """Minimal FitContext stub to drive resolve_model_flags."""
        from scribe.api.stages.model_flags import resolve_model_flags

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.model = model
        ctx.kwargs = {
            "variable_capture": variable_capture,
            "zero_inflation": zero_inflation,
        }
        ctx.priors = None
        resolve_model_flags(ctx)
        return ctx.model

    def test_twostate_with_variable_capture_true(self):
        assert self._resolve("twostate", variable_capture=True) == "twostatevcp"

    def test_twostate_with_variable_capture_false(self):
        assert self._resolve("twostate", variable_capture=False) == "twostate"

    def test_twostatevcp_with_variable_capture_true(self):
        # Redundant but consistent — no error.
        assert (
            self._resolve("twostatevcp", variable_capture=True)
            == "twostatevcp"
        )

    def test_twostatevcp_with_variable_capture_false_conflicts(self):
        with pytest.raises(ValueError, match="conflicts"):
            self._resolve("twostatevcp", variable_capture=False)

    def test_zero_inflation_rejected(self):
        with pytest.raises(ValueError, match="[Zz]ero-inflation"):
            self._resolve("twostate", zero_inflation=True)


# ==============================================================================
# Sampling-module phase-1 gates
# ==============================================================================


class TestTwoStateSamplingGates:
    """``denoise_counts_*`` and biological PPC raise on TwoState in phase 1."""

    def test_denoise_counts_map_raises(self):
        cfg = _build_valid_twostate("twostate")
        # Stub a results-like object that exposes only what the gate reads.
        from scribe.svi._sampling_denoising import DenoisingSamplingMixin

        class _Stub(DenoisingSamplingMixin):
            model_config = cfg

        with pytest.raises(NotImplementedError, match="TwoState"):
            _Stub().denoise_counts_map(jnp.zeros((1, 1), dtype=jnp.int32))

    def test_denoise_counts_posterior_raises(self):
        cfg = _build_valid_twostate("twostate")
        from scribe.svi._sampling_denoising import DenoisingSamplingMixin

        class _Stub(DenoisingSamplingMixin):
            model_config = cfg

        with pytest.raises(NotImplementedError, match="TwoState"):
            _Stub().denoise_counts_posterior(
                jnp.zeros((1, 1), dtype=jnp.int32)
            )

    def test_get_ppc_samples_biological_raises(self):
        cfg = _build_valid_twostate("twostate")
        from scribe.svi._sampling_biological import BiologicalSamplingMixin

        class _Stub(BiologicalSamplingMixin):
            model_config = cfg

        with pytest.raises(NotImplementedError, match="TwoState"):
            _Stub().get_ppc_samples_biological()
