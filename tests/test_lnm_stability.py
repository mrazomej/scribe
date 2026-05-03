"""Unit and integration tests for the LNM training-stability pass.

This file covers the LNM-only stability pieces introduced in the
``feature/logistic-normal_multinomial_model`` branch:

- ``empirical_alr_mean_from_counts`` (data-derived decoder bias init)
- ``compute_encoder_standardization`` / ``moments_to_lognormal_r_T``
  (LNM data-init helpers)
- ``LNMGaussianEncoder`` (log-scale clamp)
- ``DecoderOutputHead.bias_init`` plumbing through ``MultiHeadDecoder``
- LNM-only defaults in ``build_config_from_preset`` (the
  ``vae_standardize`` sentinel) and the LNM-only injection block in
  :func:`scribe.api.fit`.

All tests are deliberately small and free of network / data downloads
so they fit comfortably in the existing fast-pytest budget.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from scribe.core.lnm_data_init import (
    BIOLOGY_DEFAULT_R_T_MEDIAN,
    BIOLOGY_DEFAULT_R_T_SIGMA_LOG,
    CAPTURE_ANCHOR_KEYS,
    compute_encoder_standardization,
    empirical_alr_mean_from_counts,
    inject_lnm_vae_data_init,
    moments_to_lognormal_r_T,
    resolve_lnm_priors,
    resolve_r_T_prior,
)
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models.components.vae_components import (
    DecoderOutputHead,
    GaussianEncoder,
    LNMGaussianEncoder,
    MultiHeadDecoder,
    ENCODER_REGISTRY,
)


# ---------------------------------------------------------------------------
# empirical_alr_mean_from_counts
# ---------------------------------------------------------------------------


class TestEmpiricalALRMean:
    """Unit tests for the pooled ALR-mean estimator."""

    def test_shape_matches_n_genes_minus_one(self):
        # The decoder's ``y_alr`` head outputs G-1 coordinates, so the
        # initializer it consumes must match that shape exactly.
        counts = jnp.ones((4, 7))
        alr = empirical_alr_mean_from_counts(counts, reference_idx=-1)
        assert alr.shape == (6,)

    def test_uniform_data_gives_zero_alr(self):
        # When every gene has the same total, all log-ratios are zero
        # and the empirical ALR mean is exactly the zero vector.
        counts = jnp.ones((10, 5)) * 7.0
        alr = empirical_alr_mean_from_counts(counts)
        assert jnp.allclose(alr, 0.0, atol=1e-6)

    def test_pseudocount_keeps_zero_count_genes_finite(self):
        # Without the pseudocount ``log(0) = -inf`` would propagate. We
        # explicitly verify the result is finite for an all-zero gene.
        counts = jnp.array(
            [[10, 0, 5], [12, 0, 6], [11, 0, 4]], dtype=jnp.float32
        )
        alr = empirical_alr_mean_from_counts(counts, reference_idx=-1)
        assert jnp.isfinite(alr).all()

    def test_negative_index_resolves_correctly(self):
        # ``reference_idx=-1`` and ``reference_idx=n_genes-1`` should be
        # equivalent: both refer to the last gene.
        counts = jnp.array(
            [[10.0, 5.0, 2.0], [12.0, 6.0, 3.0]], dtype=jnp.float32
        )
        a1 = empirical_alr_mean_from_counts(counts, reference_idx=-1)
        a2 = empirical_alr_mean_from_counts(counts, reference_idx=2)
        assert jnp.allclose(a1, a2)

    def test_invalid_reference_index_raises(self):
        counts = jnp.ones((3, 4))
        with pytest.raises(ValueError):
            empirical_alr_mean_from_counts(counts, reference_idx=99)

    def test_two_genes_returns_length_one(self):
        # The minimal valid case (G=2) returns a length-1 array.
        counts = jnp.array([[3.0, 7.0], [5.0, 9.0]])
        alr = empirical_alr_mean_from_counts(counts)
        assert alr.shape == (1,)


# ---------------------------------------------------------------------------
# compute_encoder_standardization
# ---------------------------------------------------------------------------


class TestEncoderStandardization:
    """Tests for the per-feature mean/std helper."""

    def test_returns_per_gene_arrays(self):
        counts = jnp.ones((50, 8)) * 5.0
        mean, std = compute_encoder_standardization(counts, "log1p")
        assert mean.shape == (8,)
        assert std.shape == (8,)

    def test_constant_columns_have_floored_std(self):
        # A constant column has zero standard deviation; the helper
        # floors it so downstream divisions remain well-defined.
        counts = jnp.zeros((10, 3))
        mean, std = compute_encoder_standardization(counts, "log1p_prop")
        assert (std >= 1e-3).all()

    def test_unknown_transform_raises(self):
        counts = jnp.ones((4, 4))
        with pytest.raises(ValueError):
            compute_encoder_standardization(
                counts, "definitely_not_a_transform"
            )

    def test_uses_transformed_space(self):
        # Standardization happens in the *transformed* space, so the
        # returned mean/std should differ between two transforms.
        counts = jnp.array(
            np.random.default_rng(0).integers(0, 100, size=(20, 5)),
            dtype=jnp.float32,
        )
        m1, s1 = compute_encoder_standardization(counts, "log1p")
        m2, s2 = compute_encoder_standardization(counts, "log1p_prop")
        # Different transforms produce different stats — bare equality
        # would mean the helper is not actually applying the transform.
        assert not jnp.allclose(m1, m2)


# ---------------------------------------------------------------------------
# moments_to_lognormal_r_T
# ---------------------------------------------------------------------------


class TestMomentsToLogNormalRT:
    """Tests for the method-of-moments NB inversion on ``r_T``."""

    def test_returns_finite_floats(self):
        rng = np.random.default_rng(0)
        counts = rng.negative_binomial(n=50, p=0.005, size=(200, 5))
        mu, sigma = moments_to_lognormal_r_T(counts)
        assert np.isfinite(mu) and np.isfinite(sigma)
        assert sigma > 0.0

    def test_recovers_order_of_magnitude(self):
        # We sample u_T directly from NB(r=50, p=0.005), so the
        # method-of-moments estimate should land near 50 with the
        # generous sigma=1.0 prior.
        rng = np.random.default_rng(1)
        counts = rng.negative_binomial(n=50, p=0.005, size=(2000, 4))
        mu, _ = moments_to_lognormal_r_T(counts)
        r_T_estimate = float(np.exp(mu))
        # One order of magnitude either side is enough for a smoke
        # test; the point is it is not stuck near the prior default of 1.
        assert 5.0 < r_T_estimate < 500.0

    def test_sigma_log_kwarg_propagates(self):
        # The helper exposes ``sigma_log`` as a kwarg; ``api.py`` uses
        # this to widen the prior for LNMVCP (where the moment-of-moments
        # estimator is biased downward by capture-variability noise).
        rng = np.random.default_rng(2)
        counts = rng.negative_binomial(n=50, p=0.005, size=(200, 5))
        _, sigma_a = moments_to_lognormal_r_T(counts, sigma_log=1.0)
        _, sigma_b = moments_to_lognormal_r_T(counts, sigma_log=1.5)
        assert sigma_a == 1.0
        assert sigma_b == 1.5

    def test_under_dispersed_falls_back_to_floor(self):
        # When var(u_T) <= mean(u_T), the moment-of-moments inversion
        # is undefined; the helper must fall back to ``min_r_T``.
        # A perfectly constant total-count matrix is the cleanest such
        # case (var=0).
        counts = jnp.ones((10, 4)) * 3.0
        mu, _ = moments_to_lognormal_r_T(counts, min_r_T=2.5)
        assert jnp.isclose(jnp.exp(mu), 2.5, atol=1e-5)


# ---------------------------------------------------------------------------
# LNMGaussianEncoder
# ---------------------------------------------------------------------------


class TestLNMGaussianEncoder:
    """Tests for the LNM encoder's log-scale clamp."""

    def test_log_scale_is_clamped_in_default_range(self):
        enc = LNMGaussianEncoder(input_dim=12, latent_dim=4, hidden_dims=[8])
        params = enc.init(jax.random.PRNGKey(0), jnp.zeros(12))
        # Replace the log_scale_head bias with values guaranteed to
        # blow past the clamp range, then re-apply and verify the
        # clamp held. This stress-tests the runtime guard, not just
        # whatever happened to come out of LeCun init.
        bad_params = jax.tree.map(lambda x: x, params)
        bad_params = (
            bad_params.unfreeze()
            if hasattr(bad_params, "unfreeze")
            else dict(bad_params)
        )
        bad_params["params"]["log_scale_head"]["bias"] = jnp.array(
            [100.0, -100.0, 50.0, -50.0]
        )
        loc, log_scale = enc.apply(bad_params, jnp.zeros(12))
        assert jnp.all(log_scale >= -7.0)
        assert jnp.all(log_scale <= 2.0)

    def test_loc_is_unconstrained(self):
        # The clamp must apply only to log_scale, never to loc — the
        # location head must remain free to take any real value.
        enc = LNMGaussianEncoder(input_dim=8, latent_dim=3, hidden_dims=[6])
        params = enc.init(jax.random.PRNGKey(0), jnp.zeros(8))
        loc, _ = enc.apply(params, jnp.zeros(8))
        assert loc.shape == (3,)

    def test_registered_under_gaussian_lnm_key(self):
        assert "gaussian_lnm" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY["gaussian_lnm"] is LNMGaussianEncoder

    def test_baseline_encoder_unaffected(self):
        # The baseline ``GaussianEncoder`` should still produce
        # unclamped log-scales — we deliberately do not regress
        # non-LNM models onto the clamp.
        enc = GaussianEncoder(input_dim=8, latent_dim=3, hidden_dims=[6])
        params = enc.init(jax.random.PRNGKey(0), jnp.zeros(8))
        # Inject an extreme log_scale bias.
        bad_params = dict(params)
        bad_params["params"] = dict(bad_params["params"])
        bad_params["params"]["log_scale_head"] = dict(
            bad_params["params"]["log_scale_head"]
        )
        bad_params["params"]["log_scale_head"]["bias"] = jnp.array(
            [100.0, -100.0, 50.0]
        )
        _loc, log_scale = enc.apply(bad_params, jnp.zeros(8))
        # If the baseline accidentally inherited the clamp, the absolute
        # value would be at most 100; without the clamp it stays at 100.
        assert jnp.max(jnp.abs(log_scale)) > 50.0


# ---------------------------------------------------------------------------
# DecoderOutputHead bias_init plumbing
# ---------------------------------------------------------------------------


class TestDecoderOutputHeadBiasInit:
    """Tests that DecoderOutputHead.bias_init is honored end-to-end."""

    def test_bias_init_is_optional_and_default_none(self):
        head = DecoderOutputHead("y_alr", 3, "identity")
        assert head.bias_init is None

    def test_bias_init_used_when_provided(self):
        # With a constant bias_init and zero input, the linear-decoder
        # output must equal the bias exactly. This is the property the
        # LNM relies on: anchoring the bias to the empirical ALR mean.
        bias_arr = jnp.array([1.0, -2.0, 3.5])
        head = DecoderOutputHead(
            "y_alr",
            3,
            "identity",
            bias_init=nn.initializers.constant(bias_arr),
        )
        # ``hidden_dims=()`` here mirrors the LNM linear-decoder
        # configuration set in the factory.
        dec = MultiHeadDecoder(
            output_dim=0,
            latent_dim=2,
            hidden_dims=(),
            output_heads=(head,),
        )
        params = dec.init(jax.random.PRNGKey(0), jnp.zeros(2))
        out = dec.apply(params, jnp.zeros(2))
        assert jnp.allclose(out["y_alr"], bias_arr)

    def test_default_zero_bias_when_not_provided(self):
        # When bias_init is None, flax falls back to its zero bias.
        # Decoder(z=0) → 0 confirms that pre-stability behavior is
        # preserved for every other model that uses ``DecoderOutputHead``.
        head = DecoderOutputHead("r", 4, "identity")
        dec = MultiHeadDecoder(
            output_dim=0,
            latent_dim=2,
            hidden_dims=(),
            output_heads=(head,),
        )
        params = dec.init(jax.random.PRNGKey(0), jnp.zeros(2))
        out = dec.apply(params, jnp.zeros(2))
        assert jnp.allclose(out["r"], jnp.zeros(4), atol=1e-6)


# ---------------------------------------------------------------------------
# preset_builder LNM-only defaults
# ---------------------------------------------------------------------------


class TestPresetBuilderLNMDefaults:
    """Tests for the LNM-only ``vae_standardize`` sentinel resolution."""

    def test_lnm_default_standardize_is_true(self):
        # With no explicit override, LNM resolves to True.
        cfg = build_config_from_preset(model="lnm")
        assert cfg.vae.standardize is True

    def test_nbdm_default_standardize_is_false(self):
        # Outside LNM, the historical default of False is preserved.
        cfg = build_config_from_preset(model="nbdm", inference_method="vae")
        assert cfg.vae.standardize is False

    def test_explicit_false_for_lnm_is_honoured(self):
        # Passing the explicit boolean must override the auto-default.
        cfg = build_config_from_preset(model="lnm", vae_standardize=False)
        assert cfg.vae.standardize is False

    def test_explicit_true_for_nbdm_is_honoured(self):
        cfg = build_config_from_preset(
            model="nbdm",
            inference_method="vae",
            vae_standardize=True,
        )
        assert cfg.vae.standardize is True


# ---------------------------------------------------------------------------
# inject_lnm_vae_data_init: end-to-end ModelConfig wiring
# ---------------------------------------------------------------------------


class TestInjectLNMVAEDataInit:
    """Tests that ``inject_lnm_vae_data_init`` correctly populates ``VAEConfig``.

    These exercise the same transformation that ``scribe.api.fit``
    applies for LNM models (Step 3d), without going through the full
    SVI pipeline. The goal is to verify the *plumbing* — that a
    correctly-shaped bias init and standardization stats land on the
    config — not to test the underlying numerical correctness of the
    helpers (which is covered by the per-helper unit tests above).
    """

    @staticmethod
    def _toy_lnm_config(input_transform: str = "log1p_prop"):
        """Return a tiny LNM ``ModelConfig`` for plumbing tests."""
        return build_config_from_preset(
            model="lnm",
            vae_input_transform=input_transform,
            vae_latent_dim=2,
            vae_encoder_hidden_dims=[8],
        )

    def test_empirical_alr_bias_shape_matches_n_genes_minus_one(self):
        cfg = self._toy_lnm_config()
        rng = np.random.default_rng(0)
        counts = jnp.asarray(
            rng.integers(0, 50, size=(20, 6)), dtype=jnp.float32
        )
        new_cfg = inject_lnm_vae_data_init(cfg, counts, alr_reference_idx=-1)
        assert new_cfg.vae.empirical_alr_bias_init is not None
        assert new_cfg.vae.empirical_alr_bias_init.shape == (5,)

    def test_standardize_stats_shapes_match_n_genes(self):
        cfg = self._toy_lnm_config()
        counts = jnp.asarray(
            np.random.default_rng(1).integers(0, 50, size=(20, 6)),
            dtype=jnp.float32,
        )
        new_cfg = inject_lnm_vae_data_init(cfg, counts)
        assert new_cfg.vae.standardize_mean is not None
        assert new_cfg.vae.standardize_std is not None
        assert new_cfg.vae.standardize_mean.shape == (6,)
        assert new_cfg.vae.standardize_std.shape == (6,)

    def test_does_not_mutate_input_config(self):
        # Pydantic frozen models guarantee immutability, but it is
        # still worth exercising that the inject helper returns a
        # *new* config and leaves the original alone.
        cfg = self._toy_lnm_config()
        counts = jnp.asarray(
            np.random.default_rng(2).integers(0, 50, size=(10, 4)),
            dtype=jnp.float32,
        )
        new_cfg = inject_lnm_vae_data_init(cfg, counts)
        assert cfg.vae.empirical_alr_bias_init is None
        assert new_cfg is not cfg

    def test_factory_creates_lnm_model_with_data_init(self):
        # Regression test for the closure-scoping bug discovered when
        # the y_alr ``bias_init`` branch was first wired in: a redundant
        # ``import jax.numpy as jnp`` inside ``_create_vae_model``
        # promoted ``jnp`` to a function-local variable, breaking the
        # inner ``_build_head`` closure that referenced it before the
        # local import was reached. The fix removed the redundant
        # import and hoisted ``flax.linen`` to module scope. This test
        # exercises the full factory build path with empirical-bias
        # init populated, which is exactly the path that failed.
        from scribe.models.presets.factory import create_model

        cfg = self._toy_lnm_config()
        rng = np.random.default_rng(7)
        counts = jnp.asarray(
            rng.integers(1, 50, size=(20, 6)), dtype=jnp.float32
        )
        new_cfg = inject_lnm_vae_data_init(cfg, counts)
        # ``validate=False`` because the dry-run path requires more
        # plumbing than this minimal smoke test provides; the closure
        # bug we are guarding against fires *before* validation.
        model, guide, _ = create_model(new_cfg, n_genes=6, validate=False)
        assert callable(model) and callable(guide)

    def test_uses_input_transform_from_config(self):
        # If the config's ``input_transform`` changes, the
        # standardization stats should track it (different transform
        # → different per-feature mean).
        #
        # Note: ``build_config_from_preset`` silently rewrites
        # ``log1p`` to ``log1p_prop`` for the LNM family, so we contrast
        # ``log1p_prop`` against ``clr`` (both compositional but with
        # very different scales) to actually exercise the dependence
        # on ``vae.input_transform``.
        cfg_prop = self._toy_lnm_config("log1p_prop")
        cfg_clr = self._toy_lnm_config("clr")
        # Sanity-check that the rewrite did not silently equate them.
        assert cfg_prop.vae.input_transform != cfg_clr.vae.input_transform
        counts = jnp.asarray(
            np.random.default_rng(3).integers(1, 50, size=(20, 6)),
            dtype=jnp.float32,
        )
        new_a = inject_lnm_vae_data_init(cfg_prop, counts)
        new_b = inject_lnm_vae_data_init(cfg_clr, counts)
        assert not jnp.allclose(
            new_a.vae.standardize_mean, new_b.vae.standardize_mean
        )


# ---------------------------------------------------------------------------
# resolve_r_T_prior: branching between MoM and biology default
# ---------------------------------------------------------------------------


class TestResolveRTPrior:
    """Tests for the capture-anchor-aware ``r_T`` prior resolver.

    The resolver decides between (a) a method-of-moments inversion on the
    empirical totals, (b) a biology-informed default when the user has
    activated the capture anchor, or (c) no auto-assignment when the user
    supplied an explicit ``r_T`` override or the model is non-LNM. Each
    of those branches is exercised individually here so a regression in
    any single one is immediately localized.
    """

    @staticmethod
    def _toy_counts(seed: int = 0):
        """Return a ``(200, 5)`` count matrix sampled from NB(50, 0.005).

        The shape is chosen so the moment-of-moments inversion has enough
        data to be meaningful but the test runs in milliseconds.
        """
        rng = np.random.default_rng(seed)
        return rng.negative_binomial(n=50, p=0.005, size=(200, 5))

    # -- Non-LNM short-circuit ---------------------------------------------

    def test_returns_none_for_non_lnm_models(self):
        # Every non-LNM model must short-circuit to ``None`` so the
        # existing prior calibration of NBDM/NBVCP/ZINB/ZINBVCP is
        # bit-identical to pre-stability behaviour.
        for model in ("nbdm", "nbvcp", "zinb", "zinbvcp"):
            assert resolve_r_T_prior(model, self._toy_counts(), None) is None

    # -- User explicit override always wins --------------------------------

    def test_explicit_r_T_short_circuits(self):
        # The caller's intent is to fix the prior — never silently
        # override that. This must hold regardless of model and regardless
        # of whether the capture anchor is active.
        priors = {"r_T": (3.0, 0.5)}
        assert resolve_r_T_prior("lnm", self._toy_counts(), priors) is None
        assert resolve_r_T_prior("lnmvcp", self._toy_counts(), priors) is None

    def test_explicit_r_T_wins_over_capture_anchor(self):
        # Composite case: the user has activated the capture anchor *and*
        # supplied an explicit ``r_T``. Explicit wins.
        priors = {"organism": "human", "r_T": (3.0, 0.5)}
        assert resolve_r_T_prior("lnmvcp", self._toy_counts(), priors) is None

    # -- Branch 1: no capture anchor → MoM ---------------------------------

    def test_lnm_no_anchor_uses_mom_with_sigma_one(self):
        # Plain LNM: the moment-of-moments inversion is exact (no capture
        # variability to bias it), so we tighten ``sigma_log = 1.0``.
        out = resolve_r_T_prior("lnm", self._toy_counts(), priors=None)
        assert out is not None
        _mu, sigma = out
        assert sigma == 1.0

    def test_lnmvcp_no_anchor_uses_mom_with_sigma_one_point_five(self):
        # LNMVCP without the anchor: MoM is biased downward by capture
        # variability; widening ``sigma_log = 1.5`` lets the prior cover
        # the bias. Documented in the qmd.
        out = resolve_r_T_prior("lnmvcp", self._toy_counts(), priors={})
        assert out is not None
        _mu, sigma = out
        assert sigma == 1.5

    # -- Branch 2: capture anchor active → biology default -----------------

    def test_capture_anchor_via_organism_uses_biology_default(self):
        # ``organism`` is one of the canonical capture-anchor activation
        # keys (along with ``eta_capture`` and ``mu_eta``). Any of them
        # should trigger the biology-default branch.
        out = resolve_r_T_prior(
            "lnmvcp", self._toy_counts(), priors={"organism": "human"}
        )
        assert out is not None
        mu, sigma = out
        assert np.isclose(np.exp(mu), BIOLOGY_DEFAULT_R_T_MEDIAN)
        assert sigma == BIOLOGY_DEFAULT_R_T_SIGMA_LOG

    def test_capture_anchor_via_eta_capture_uses_biology_default(self):
        # ``eta_capture`` is the lower-level knob; the API exposes both.
        # We test it explicitly so accidental removal of either key from
        # the activation list is caught.
        out = resolve_r_T_prior(
            "lnmvcp",
            self._toy_counts(),
            priors={"eta_capture": (12.2, 0.05)},
        )
        assert out is not None
        mu, sigma = out
        assert np.isclose(np.exp(mu), BIOLOGY_DEFAULT_R_T_MEDIAN)
        assert sigma == BIOLOGY_DEFAULT_R_T_SIGMA_LOG

    def test_capture_anchor_via_mu_eta_uses_biology_default(self):
        # ``mu_eta`` is the multi-dataset hierarchical-capture key; same
        # contract.
        out = resolve_r_T_prior(
            "lnmvcp",
            self._toy_counts(),
            priors={"mu_eta": (12.2, 0.05)},
        )
        assert out is not None
        mu, sigma = out
        assert np.isclose(np.exp(mu), BIOLOGY_DEFAULT_R_T_MEDIAN)

    def test_capture_anchor_keys_are_exposed(self):
        # Sanity-check that the public constant matches what the resolver
        # actually consumes. If somebody adds a new key to one without
        # updating the other, this catches the drift.
        assert "eta_capture" in CAPTURE_ANCHOR_KEYS
        assert "mu_eta" in CAPTURE_ANCHOR_KEYS
        assert "organism" in CAPTURE_ANCHOR_KEYS

    # -- Branch ordering: MoM and biology give different answers -----------

    def test_branches_are_distinct(self):
        # The two branches must produce different priors (otherwise the
        # branching is pointless). On the same counts, MoM gives whatever
        # ``m^2/(v-m)`` returns; biology gives 50. They should differ
        # except on a measure-zero coincidence we skip past.
        counts = self._toy_counts()
        out_mom = resolve_r_T_prior("lnmvcp", counts, priors={})
        out_anchor = resolve_r_T_prior(
            "lnmvcp", counts, priors={"organism": "human"}
        )
        assert out_mom is not None and out_anchor is not None
        # The medians should differ — sigma_log happens to coincide at 1.5
        # so we compare ``mu`` only.
        assert not np.isclose(out_mom[0], out_anchor[0], atol=1e-2)

    # -- Sigma_log of biology default is the documented value --------------

    def test_biology_default_sigma_log_value(self):
        # Documented in the qmd as 1.5. Tests pin both ends of the
        # contract: the constant exposed in the module *and* the value
        # the resolver returns.
        assert BIOLOGY_DEFAULT_R_T_SIGMA_LOG == 1.5
        out = resolve_r_T_prior(
            "lnmvcp", self._toy_counts(), priors={"organism": "human"}
        )
        assert out is not None
        assert out[1] == 1.5

    # -- Biology default median is the documented value -------------------

    def test_biology_default_median_value(self):
        # Documented in the qmd as 50. The constant and the resolver
        # output must agree.
        assert BIOLOGY_DEFAULT_R_T_MEDIAN == 50.0
        out = resolve_r_T_prior(
            "lnmvcp", self._toy_counts(), priors={"organism": "human"}
        )
        assert out is not None
        assert np.isclose(np.exp(out[0]), 50.0)

    # -- Caller's priors dict is not mutated ------------------------------

    def test_resolver_does_not_mutate_priors(self):
        # The resolver is read-only on its ``priors`` argument; the
        # caller in api.py is responsible for cloning before insertion.
        priors = {"organism": "human"}
        before = dict(priors)
        _ = resolve_r_T_prior("lnmvcp", self._toy_counts(), priors)
        assert priors == before


# ---------------------------------------------------------------------------
# End-to-end: user-supplied r_T prior survives the full preset-builder path
# ---------------------------------------------------------------------------


class TestUserRTPriorEndToEnd:
    """Verify the user-facing override path for the ``r_T`` prior.

    These tests pin the user-visible contract: ``priors={"r_T": ...}``
    or ``priors={"total_dispersion": ...}`` lands on
    ``model_config.priors.r_T`` after ``build_config_from_preset``.
    The downstream propagation to the final ``ParamSpec`` is the
    factory's responsibility and is exercised by the existing
    LNM smoke tests in ``test_lnm_factory.py``; here we focus on the
    public-API entry point.
    """

    @staticmethod
    def _build_lnm_config(priors):
        """Run the builder for an LNM config and return it for inspection."""
        return build_config_from_preset(
            model="lnm",
            vae_latent_dim=2,
            vae_encoder_hidden_dims=[8],
            priors=priors,
        )

    def test_explicit_r_T_lands_on_priors_overrides(self):
        # The canonical user override. After ``build_config_from_preset``
        # processes ``priors={"r_T": (3.5, 0.5)}``, the value must be
        # accessible via ``model_config.priors.r_T`` — the official
        # storage point for user-supplied prior hyperparameters.
        cfg = self._build_lnm_config({"r_T": (3.5, 0.5)})
        assert cfg.priors.r_T == (3.5, 0.5), (
            f"Expected user-supplied r_T prior (3.5, 0.5) to land on "
            f"model_config.priors.r_T; got {cfg.priors.r_T!r}."
        )

    def test_descriptive_alias_resolves_to_r_T(self):
        # The descriptive alias ``total_dispersion`` registered in
        # PRIOR_KEY_ALIASES should be normalized to ``r_T`` during
        # build, so the value lands at the same canonical location
        # as if the user had passed the internal name directly.
        cfg = self._build_lnm_config({"total_dispersion": (2.0, 0.3)})
        assert cfg.priors.r_T == (2.0, 0.3), (
            f"Expected descriptive alias 'total_dispersion' to "
            f"resolve to r_T; got priors.r_T={cfg.priors.r_T!r}."
        )

    def test_descriptive_name_registered(self):
        # Pin the registry contract: r_T must have a descriptive name,
        # symmetric with how r has 'dispersion'. If the registry drops
        # this entry, the alias path silently breaks.
        from scribe.models.config.parameter_mapping import (
            DESCRIPTIVE_NAMES,
            PRIOR_KEY_ALIASES,
        )

        assert DESCRIPTIVE_NAMES["r_T"] == "total_dispersion"
        assert PRIOR_KEY_ALIASES["total_dispersion"] == "r_T"
        # And the DM gene-level r retains its independent descriptive
        # name; the two parameters are distinct in the public API.
        assert DESCRIPTIVE_NAMES["r"] == "dispersion"
        assert PRIOR_KEY_ALIASES["dispersion"] == "r"

    def test_lnmvcp_capture_anchor_e2e(self):
        # Regression test for the user-reported failure
        # ``RuntimeError: Site p_capture must be sampled in trace.``
        # which fired when ``priors={"capture_efficiency": ...}`` was
        # passed to LNMVCP. Without the fix in
        # ``LNMWithVCPLikelihood.sample`` (and the matching factory
        # wiring of ``biology_informed_spec``), the guide sampled the
        # anchored ``eta_capture`` site while the model still sampled
        # the flat ``p_capture``, and SVI's replay handler failed.
        # This test exercises the whole config-build → factory →
        # SVI.init path with the capture anchor active and asserts
        # that initialisation succeeds and a single SVI step produces
        # a finite loss.
        import math

        import numpyro
        from numpyro.infer import SVI, TraceMeanField_ELBO

        from scribe.models.model_registry import get_model_and_guide

        n_cells, n_genes = 32, 20
        rng = np.random.default_rng(7)
        counts = jnp.asarray(
            rng.negative_binomial(
                n=20, p=0.005, size=(n_cells, n_genes)
            ),
            dtype=jnp.float32,
        )
        # ``capture_efficiency`` is the descriptive alias for
        # ``eta_capture``; (log M_0, sigma_M) tuple. We use M_0 = 2e5,
        # the typical mammalian-cell value documented in the
        # capture-prior qmd.
        capture_efficiency = (math.log(2e5), 0.05)

        cfg = build_config_from_preset(
            model="lnmvcp",
            vae_latent_dim=4,
            vae_encoder_hidden_dims=[16],
            priors={"capture_efficiency": capture_efficiency},
        )
        model, guide, _ = get_model_and_guide(cfg, n_genes=n_genes)

        # The classic failure was at SVI init — the replay handler
        # asserted ``"Site p_capture must be sampled in trace."``.
        # With the fix, init must succeed and the model and guide
        # both sample ``eta_capture`` instead.
        svi = SVI(
            model, guide, numpyro.optim.Adam(1e-3),
            loss=TraceMeanField_ELBO(),
        )
        svi_state = svi.init(
            jax.random.PRNGKey(0),
            n_cells=n_cells, n_genes=n_genes,
            model_config=cfg, counts=counts,
        )
        # One update to confirm the gradient path executes end-to-end.
        svi_state, loss = svi.update(
            svi_state, n_cells=n_cells, n_genes=n_genes,
            model_config=cfg, counts=counts,
        )
        assert jnp.isfinite(loss), (
            f"Expected finite SVI loss after one update with the "
            f"capture anchor active; got loss={float(loss)!r}."
        )

    def test_r_and_r_T_are_disjoint_keys(self):
        # Sanity: passing ``priors={"r": (...)}`` (the DM-family
        # gene-level dispersion key) must not silently land on
        # ``r_T``. The two parameters have different shapes and
        # distinct registry entries; the API treats them as disjoint.
        # We check this by verifying that the user's ``r`` value does
        # not *equal* the value that lands on ``r_T`` — ``r_T`` should
        # take its declared default ``(0.0, 1.0)`` since the user did
        # not override it, NOT the user's ``r`` value.
        user_r = (4.2, 0.7)
        cfg = self._build_lnm_config({"r": user_r})
        r_T_value = getattr(cfg.priors, "r_T", None)
        assert r_T_value != user_r, (
            f"Setting priors['r'] = {user_r} should not silently land "
            f"on r_T; got priors.r_T={r_T_value!r}."
        )
        # Optional positive check: r_T should be at the default for
        # the LNM parameterization. This pins the disjointness more
        # strongly: the user's r value left no fingerprint on r_T.
        assert r_T_value == (0.0, 1.0), (
            f"Expected priors.r_T to fall through to its declared "
            f"default (0.0, 1.0); got {r_T_value!r}."
        )


# ---------------------------------------------------------------------------
# resolve_lnm_priors: parameterization-aware auto-defaults
# ---------------------------------------------------------------------------


class TestResolveLNMPriors:
    """Tests for the parameterization-aware LNM prior resolver.

    ``resolve_lnm_priors`` returns the auto-default priors for the
    *sampled* scalars of the chosen LNM-family parameterization:
    canonical → r_T; mean_prob → mu_T; mean_odds → mu_T + phi_T. Each
    test exercises one branch of the dispatch and pins the resulting
    keys.
    """

    @staticmethod
    def _toy_counts(seed: int = 0):
        rng = np.random.default_rng(seed)
        return rng.negative_binomial(n=50, p=0.005, size=(200, 5))

    def test_returns_empty_for_non_lnm_models(self):
        # Non-LNM models always return ``{}`` so DM-family fits stay
        # bit-identical to pre-parameterization-pass behavior.
        for model in ("nbdm", "nbvcp", "zinb", "zinbvcp"):
            assert resolve_lnm_priors(
                model, "canonical", self._toy_counts(), None
            ) == {}

    def test_canonical_returns_r_T_only(self):
        out = resolve_lnm_priors(
            "lnmvcp", "canonical", self._toy_counts(), priors=None
        )
        assert set(out.keys()) == {"r_T"}
        # MoM mode (no anchor) should produce sigma_log=1.5 for LNMVCP.
        _mu, sigma = out["r_T"]
        assert sigma == 1.5

    def test_mean_prob_returns_mu_T_only(self):
        out = resolve_lnm_priors(
            "lnmvcp", "mean_prob", self._toy_counts(), priors=None
        )
        assert set(out.keys()) == {"mu_T"}
        # mu_T's median should be the empirical mean library size.
        mu, sigma = out["mu_T"]
        assert sigma == 1.0
        assert mu > 0.0  # log of a positive estimate

    def test_mean_odds_returns_mu_T_and_phi_T(self):
        out = resolve_lnm_priors(
            "lnmvcp", "mean_odds", self._toy_counts(), priors=None
        )
        # Both scalars get auto-defaults — exactly what we want for
        # mean_odds, which has no aliasing under the capture anchor.
        assert set(out.keys()) == {"mu_T", "phi_T"}

    def test_user_override_skips_internal_name(self):
        # Setting the internal name short-circuits that scalar's
        # auto-default. Other scalars are still set automatically.
        out = resolve_lnm_priors(
            "lnmvcp",
            "mean_odds",
            self._toy_counts(),
            priors={"mu_T": (10.0, 0.5)},
        )
        assert "mu_T" not in out
        assert "phi_T" in out

    def test_user_override_skips_descriptive_alias(self):
        # ``total_mean`` is the descriptive alias for ``mu_T``; the
        # resolver must respect it as a user override even though the
        # internal name is what gets stored.
        out = resolve_lnm_priors(
            "lnmvcp",
            "mean_odds",
            self._toy_counts(),
            priors={"total_mean": (10.0, 0.5)},
        )
        assert "mu_T" not in out

    def test_canonical_capture_anchor_uses_biology_default(self):
        # When the capture anchor is active under the canonical
        # variant, r_T falls back to the biology-informed default —
        # mirrors the existing resolve_r_T_prior behavior.
        out = resolve_lnm_priors(
            "lnmvcp",
            "canonical",
            self._toy_counts(),
            priors={"organism": "human"},
        )
        assert "r_T" in out
        mu, sigma = out["r_T"]
        assert np.isclose(np.exp(mu), BIOLOGY_DEFAULT_R_T_MEDIAN)
        assert sigma == BIOLOGY_DEFAULT_R_T_SIGMA_LOG

    def test_accepts_internal_or_user_facing_param_names(self):
        # The resolver accepts both ``"canonical"`` (user-facing) and
        # ``"logistic_normal_canonical"`` (internal). Same outcome.
        counts = self._toy_counts(seed=2)
        out_user = resolve_lnm_priors(
            "lnmvcp", "canonical", counts, priors=None
        )
        out_internal = resolve_lnm_priors(
            "lnmvcp", "logistic_normal_canonical", counts, priors=None
        )
        assert out_user == out_internal

    def test_explicit_user_priors_for_all_scalars_yield_empty(self):
        # If the user supplied every scalar's prior, the resolver has
        # nothing to do and returns ``{}``.
        out = resolve_lnm_priors(
            "lnmvcp",
            "mean_odds",
            self._toy_counts(),
            priors={"mu_T": (10.0, 0.5), "phi_T": (1.0, 0.5)},
        )
        assert out == {}


# ---------------------------------------------------------------------------
# End-to-end: each parameterization round-trips through scribe.fit's setup
# ---------------------------------------------------------------------------


class TestLNMParameterizationDispatch:
    """End-to-end smoke tests for the three LNM parameterizations.

    These exercise the dispatch from user-facing
    ``parameterization=canonical/mean_prob/mean_odds`` all the way to
    a fully-built ``ModelConfig`` and a runnable model+guide pair.
    Catches regressions in any of the wiring layers (preset_builder,
    factory, parameterization classes, base.py validator).
    """

    @staticmethod
    def _build_lnm_config(parameterization: str):
        return build_config_from_preset(
            model="lnm",
            parameterization=parameterization,
            vae_latent_dim=2,
            vae_encoder_hidden_dims=[8],
        )

    def test_canonical_resolves_to_internal_key(self):
        cfg = self._build_lnm_config("canonical")
        assert cfg.parameterization.value == "logistic_normal_canonical"

    def test_mean_prob_resolves_to_internal_key(self):
        cfg = self._build_lnm_config("mean_prob")
        assert cfg.parameterization.value == "logistic_normal_mean_prob"

    def test_mean_odds_resolves_to_internal_key(self):
        cfg = self._build_lnm_config("mean_odds")
        assert cfg.parameterization.value == "logistic_normal_mean_odds"

    def test_each_variant_builds_model_and_guide(self):
        # The factory must produce a runnable model+guide for every
        # variant. Skipping validate=False would let
        # parameterization-specific bugs slip through; we want the
        # smoke test to exercise the full path.
        from scribe.models.model_registry import get_model_and_guide

        for p in ("canonical", "mean_prob", "mean_odds"):
            cfg = self._build_lnm_config(p)
            model, guide, _ = get_model_and_guide(cfg, n_genes=6)
            assert callable(model) and callable(guide)

    def test_legacy_logistic_normal_string_raises(self):
        # The user-facing API no longer accepts ``"logistic_normal"``;
        # users should pass one of the three explicit variants.
        with pytest.raises(ValueError, match="logistic_normal"):
            build_config_from_preset(
                model="lnm", parameterization="logistic_normal"
            )
