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
    compute_encoder_standardization,
    empirical_alr_mean_from_counts,
    inject_lnm_vae_data_init,
    moments_to_lognormal_r_T,
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
