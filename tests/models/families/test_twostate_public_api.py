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

    def test_mixture_accepted(self):
        """Mixtures are now fully supported for TwoState models."""
        cfg = ModelConfig(
            base_model="twostate",
            parameterization=Parameterization.TWO_STATE_NATURAL,
            n_components=2,
        )
        assert cfg.n_components == 2

    def test_vae_inference_rejected(self):
        with pytest.raises(ValueError, match="VAE"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.TWO_STATE_NATURAL,
                inference_method=InferenceMethod.VAE,
            )

    def test_multi_dataset_accepted(self):
        """Two-state multi-dataset fitting is supported.

        The config builds without error, and the dataset-level mu and regime
        hierarchies can be enabled (unconstrained + n_datasets >= 2).
        """
        cfg = ModelConfig(
            base_model="twostate",
            parameterization=Parameterization.TWO_STATE_MOMENT_DELTA,
            unconstrained=True,
            n_datasets=3,
            expression_dataset_prior="horseshoe",
            regime_dataset_prior="horseshoe",
        )
        assert cfg.n_datasets == 3
        assert cfg.regime_dataset_prior.value == "horseshoe"
        # Overdispersion defaults to free-per-dataset for multi-dataset fits.
        assert cfg.overdispersion_dataset_independent is True

    def test_mixture_plus_multi_dataset_rejected(self):
        """Mixture (n_components>=2) + multi-dataset is not yet supported.

        The component axis on ``mu`` and the dataset axis on the regime /
        overdispersion coordinates collide in the reparameterization; reject
        the combination at config time instead of crashing inside SVI.
        """
        with pytest.raises(ValueError, match="mixture.*multi-dataset|not yet"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.TWO_STATE_MOMENT_DELTA,
                unconstrained=True,
                n_datasets=2,
                n_components=2,
                regime_dataset_prior="horseshoe",
            )

    def test_invalid_regime_dataset_target_rejected(self):
        """regime_dataset_target must name a coordinate of the parameterization."""
        # k_off is the natural/ratio regime coord, not a mean_fano coordinate.
        with pytest.raises(ValueError, match="regime_dataset_target"):
            ModelConfig(
                base_model="twostate",
                parameterization=Parameterization.TWO_STATE_MEAN_FANO,
                unconstrained=True,
                n_datasets=2,
                regime_dataset_prior="horseshoe",
                regime_dataset_target="k_off",
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
    """``denoise_counts_*`` and biological PPC are now wired for the
    TwoState family in phase 1.

    Closure under binomial thinning (TwoState §2.3 in the paper) makes
    the pre-capture rate the natural denoising target, so both
    ``denoise_counts_map`` and ``denoise_counts_posterior`` dispatch
    into a Poisson–Beta quadrature path rather than raising
    ``NotImplementedError`` as they did during early development.
    """

    def test_denoise_counts_map_wired(self):
        """MAP denoising reaches the TwoState quadrature dispatch."""
        import scribe
        import jax

        rng = np.random.default_rng(0)
        counts = np.stack(
            [rng.poisson(m, 32) for m in [2.0, 5.0, 10.0, 50.0, 100.0]],
            axis=1,
        )
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="natural",
            inference_method="svi",
            n_steps=2,
            unconstrained=True,
        )
        # Should NOT raise.  Output shape is (n_cells, n_genes) for
        # MAP denoising — one denoised count matrix.
        denoised = res.denoise_counts_map(
            counts=counts,
            rng_key=jax.random.PRNGKey(0),
        )
        arr = np.asarray(denoised)
        assert arr.shape == counts.shape, (
            f"MAP denoised shape should match input counts; got {arr.shape}"
        )

    def test_denoise_counts_posterior_wired(self):
        """Full-posterior denoising reaches the TwoState quadrature dispatch."""
        import scribe
        import jax

        rng = np.random.default_rng(0)
        counts = np.stack(
            [rng.poisson(m, 32) for m in [2.0, 5.0, 10.0, 50.0, 100.0]],
            axis=1,
        )
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="natural",
            inference_method="svi",
            n_steps=2,
            unconstrained=True,
        )
        # Should NOT raise.  Posterior denoising returns
        # (n_samples, n_cells, n_genes).
        denoised = res.denoise_counts_posterior(
            counts=counts,
            n_samples=2,
            rng_key=jax.random.PRNGKey(0),
        )
        arr = np.asarray(denoised)
        assert arr.ndim == 3
        assert arr.shape[-2:] == counts.shape

    def test_get_ppc_samples_biological_wired(self):
        """Smoke check that biological PPC now reaches the TwoState
        dispatch branch instead of raising NotImplementedError."""
        import scribe
        import jax

        rng = np.random.default_rng(0)
        counts = np.stack(
            [rng.poisson(m, 32) for m in [2.0, 5.0, 10.0, 50.0, 100.0]],
            axis=1,
        )
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="natural",
            inference_method="svi",
            n_steps=2,
            unconstrained=True,
        )
        out = res.get_ppc_samples_biological(
            rng_key=jax.random.PRNGKey(0),
            n_samples=2,
            counts=counts,
        )
        assert "predictive_samples" in out
        # Biological PPC strips the capture factor; samples shape is
        # (n_posterior_samples, n_cells, n_genes).
        arr = np.asarray(out["predictive_samples"])
        assert arr.ndim == 3
        assert arr.shape[-2:] == counts.shape


# ==============================================================================
# Posterior builder honors low-rank / joint-low-rank guides
# ==============================================================================


class TestTwoStatePosteriorBuilderLowRank:
    """``get_posterior_distributions`` does not crash when the user
    requests joint low-rank guides on ``(burst_size, k_off)`` (which
    also causes the preset builder to auto-add a ``LowRankGuide`` on
    ``mu``).  Regression for KeyError 'mu_scale' bug.
    """

    def _build_lowrank_params(self, n_genes=4, rank=2):
        """Synthesize the param dict shape a fit with ``joint_params=
        ('burst_size', 'k_off'), guide_rank=2`` would produce."""
        rng = np.random.default_rng(0)
        # mu got the auto-added LowRankGuide → stand-alone (mu_loc,
        # mu_W, mu_raw_diag) at gene rank.
        mu_params = {
            "mu_loc": jnp.asarray(rng.normal(size=(n_genes,))),
            "mu_W": jnp.asarray(rng.normal(size=(n_genes, rank))),
            "mu_raw_diag": jnp.asarray(rng.normal(size=(n_genes,))),
        }
        # burst_size and k_off share a JointLowRankGuide(group="joint").
        # That stores: joint_joint_{name}_loc, joint_joint_{name}_W,
        # joint_joint_{name}_raw_diag, each at gene rank.
        joint_params = {}
        for name in ("burst_size", "k_off"):
            joint_params[f"joint_joint_{name}_loc"] = jnp.asarray(
                rng.normal(size=(n_genes,))
            )
            joint_params[f"joint_joint_{name}_W"] = jnp.asarray(
                rng.normal(size=(n_genes, rank))
            )
            joint_params[f"joint_joint_{name}_raw_diag"] = jnp.asarray(
                rng.normal(size=(n_genes,))
            )
        return {**mu_params, **joint_params}

    def _build_unconstrained_cfg(self, model="twostate"):
        """Mirror the user scenario: unconstrained=True at builder time."""
        builder = (
            ModelConfigBuilder()
            .for_model(model)
            .with_parameterization("two_state_natural")
            .with_inference("svi")
            .unconstrained()
        )
        if model == "twostatevcp":
            builder = builder.with_priors(p_capture=(1.0, 1.0))
        return builder.build()

    def test_get_posterior_distributions_with_low_rank_mu(self):
        """LowRankGuide on mu + JointLowRankGuide on (burst_size, k_off)
        must dispatch correctly through _build_two_state_posteriors.

        Matches the real-user scenario: ``scribe.fit(..., unconstrained=
        True, joint_params=('burst_size', 'k_off'), guide_rank=2)``.
        """
        from scribe.models.builders import get_posterior_distributions

        cfg = self._build_unconstrained_cfg("twostate")
        params = self._build_lowrank_params(n_genes=N_GENES, rank=2)

        dists = get_posterior_distributions(params, cfg, split=False)

        # All three sites must be present and constructible.
        assert "mu" in dists
        assert "burst_size" in dists
        assert "k_off" in dists

    def test_summary_string_lists_burst_size_and_k_off(self):
        """The repr's guide summary must enumerate burst_size and
        k_off, not silently drop them.  Regression for the misleading
        ``low_rank(k=2) on mu`` display."""
        from scribe.models.components.guide_families import LowRankGuide
        from scribe.models.config.groups import GuideFamilyConfig
        from scribe.svi.results import ScribeSVIResults

        gfc = GuideFamilyConfig(
            burst_size=LowRankGuide(rank=2),
            k_off=LowRankGuide(rank=2),
        )

        class _CfgStub:
            guide_families = gfc

        stub = ScribeSVIResults.__new__(ScribeSVIResults)
        stub.model_config = _CfgStub()
        summary = stub._summarize_guide_families()
        assert "burst_size" in summary
        assert "k_off" in summary
