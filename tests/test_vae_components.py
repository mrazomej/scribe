"""
Tests for VAE encoder and decoder Flax Linen components.

Tests cover:
- Output shapes (single sample and batched)
- Parameter initialization and structure
- Input transformation and standardization
- Encoder-decoder round-trip consistency
- Numerical stability
- MultiHeadDecoder with output heads and transforms
- Integration with NumPyro via flax_module
"""

import pytest
import numpy.testing as npt
import jax
import jax.numpy as jnp

from scribe.models.components.vae_components import (
    GaussianEncoder,
    MultiHeadDecoder,
    DecoderOutputHead,
    AbstractEncoder,
    AbstractDecoder,
    OUTPUT_TRANSFORMS,
    ENCODER_REGISTRY,
    DECODER_REGISTRY,
    _get_input_transform,
    _get_output_transform,
    _get_act,
)
from scribe.models.components.covariate_embedding import CovariateSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def input_dim():
    return 100


@pytest.fixture(scope="session")
def latent_dim():
    return 10


@pytest.fixture(scope="session")
def hidden_dims():
    return [64, 32]


@pytest.fixture(scope="session")
def sample_counts(rng, input_dim):
    """Simulated count data (non-negative)."""
    return jax.random.poisson(rng, lam=5.0, shape=(8, input_dim)).astype(
        jnp.float32
    )


@pytest.fixture(scope="session")
def standardization_stats(sample_counts):
    """Compute standardization stats from sample data."""
    transformed = jnp.log1p(sample_counts)
    mean = jnp.mean(transformed, axis=0)
    std = jnp.std(transformed, axis=0)
    return mean, std


# ---------------------------------------------------------------------------
# Input Transform Tests
# ---------------------------------------------------------------------------


class TestInputTransforms:
    """Tests for input transformation helpers."""

    def test_log1p_transform(self):
        x = jnp.array([0.0, 1.0, 10.0])
        transform = _get_input_transform("log1p")
        npt.assert_allclose(transform(x), jnp.log1p(x))

    def test_sqrt_transform(self):
        x = jnp.array([0.0, 1.0, 4.0, 9.0])
        transform = _get_input_transform("sqrt")
        npt.assert_allclose(transform(x), jnp.sqrt(x))

    def test_identity_transform(self):
        x = jnp.array([1.0, 2.0, 3.0])
        transform = _get_input_transform("identity")
        npt.assert_allclose(transform(x), x)

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown input_transformation"):
            _get_input_transform("nonexistent")

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            _get_act("nonexistent")


# ---------------------------------------------------------------------------
# Output Transform Tests
# ---------------------------------------------------------------------------


class TestOutputTransforms:
    """Tests for output transformation registry."""

    def test_identity(self):
        x = jnp.array([-1.0, 0.0, 1.0])
        npt.assert_allclose(_get_output_transform("identity")(x), x)

    def test_exp(self):
        x = jnp.array([0.0, 1.0, 2.0])
        npt.assert_allclose(_get_output_transform("exp")(x), jnp.exp(x))

    def test_softplus(self):
        x = jnp.array([-1.0, 0.0, 1.0])
        npt.assert_allclose(
            _get_output_transform("softplus")(x), jax.nn.softplus(x)
        )

    def test_sigmoid(self):
        x = jnp.array([-2.0, 0.0, 2.0])
        result = _get_output_transform("sigmoid")(x)
        npt.assert_allclose(result, jax.nn.sigmoid(x))
        assert jnp.all(result > 0) and jnp.all(result < 1)

    def test_clamp_exp(self):
        x = jnp.array([-10.0, 0.0, 10.0])
        result = _get_output_transform("clamp_exp")(x)
        # Values outside [-5, 5] are clamped before exp
        npt.assert_allclose(
            result, jnp.exp(jnp.clip(x, -5.0, 5.0)), rtol=1e-5
        )

    def test_invalid_output_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown output transform"):
            _get_output_transform("nonexistent")

    def test_all_transforms_jit_compatible(self):
        """All registered transforms must work under JAX JIT."""
        x = jnp.array([0.0, 1.0, -1.0])
        for name, fn in OUTPUT_TRANSFORMS.items():
            result = jax.jit(fn)(x)
            assert jnp.all(jnp.isfinite(result)), f"JIT failed for '{name}'"


# ---------------------------------------------------------------------------
# DecoderOutputHead Tests
# ---------------------------------------------------------------------------


class TestDecoderOutputHead:
    """Tests for the DecoderOutputHead frozen dataclass."""

    def test_basic_creation(self):
        head = DecoderOutputHead("r", output_dim=100, transform="exp")
        assert head.param_name == "r"
        assert head.output_dim == 100
        assert head.transform == "exp"

    def test_default_transform_is_identity(self):
        head = DecoderOutputHead("x", output_dim=10)
        assert head.transform == "identity"

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown transform"):
            DecoderOutputHead("r", output_dim=10, transform="bad")

    def test_frozen(self):
        head = DecoderOutputHead("r", output_dim=10)
        with pytest.raises(AttributeError):
            head.param_name = "q"

    def test_hashable(self):
        """Frozen dataclass must be hashable for Flax module attributes."""
        h1 = DecoderOutputHead("r", 100, "exp")
        h2 = DecoderOutputHead("r", 100, "exp")
        assert hash(h1) == hash(h2)
        assert h1 == h2
        # Can be used in a set / as dict key
        assert len({h1, h2}) == 1


# ---------------------------------------------------------------------------
# GaussianEncoder Tests
# ---------------------------------------------------------------------------


class TestGaussianEncoder:
    """Tests for the Gaussian (diagonal posterior) encoder module."""

    @pytest.fixture
    def encoder(self, input_dim, latent_dim, hidden_dims):
        return GaussianEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            input_transformation="log1p",
        )

    @pytest.fixture
    def encoder_params(self, encoder, rng, input_dim):
        return encoder.init(rng, jnp.zeros(input_dim))

    def test_output_shape_single(self, encoder, encoder_params, input_dim, latent_dim):
        x = jnp.ones(input_dim)
        loc, log_scale = encoder.apply(encoder_params, x)
        assert loc.shape == (latent_dim,)
        assert log_scale.shape == (latent_dim,)

    def test_output_shape_batched(
        self, encoder, encoder_params, input_dim, latent_dim
    ):
        x = jnp.ones((5, input_dim))
        loc, log_scale = encoder.apply(encoder_params, x)
        assert loc.shape == (5, latent_dim)
        assert log_scale.shape == (5, latent_dim)

    def test_output_finite(self, encoder, encoder_params, sample_counts):
        loc, log_scale = encoder.apply(encoder_params, sample_counts)
        assert jnp.all(jnp.isfinite(loc))
        assert jnp.all(jnp.isfinite(log_scale))

    def test_zero_input(self, encoder, encoder_params, input_dim):
        """Encoder should handle zero counts gracefully (log1p(0) = 0)."""
        x = jnp.zeros((2, input_dim))
        loc, log_scale = encoder.apply(encoder_params, x)
        assert jnp.all(jnp.isfinite(loc))
        assert jnp.all(jnp.isfinite(log_scale))

    def test_params_structure(self, encoder_params):
        """Check that the parameter tree has expected structure."""
        params = encoder_params["params"]
        assert "hidden_0" in params
        assert "hidden_1" in params
        assert "loc_head" in params
        assert "log_scale_head" in params

    def test_different_activations(self, rng, input_dim, latent_dim, hidden_dims):
        """Different activations should all produce valid outputs."""
        x = jnp.ones((2, input_dim))
        for act_name in ["relu", "gelu", "silu", "tanh", "elu"]:
            enc = GaussianEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=act_name,
            )
            params = enc.init(rng, jnp.zeros(input_dim))
            loc, log_scale = enc.apply(params, x)
            assert jnp.all(jnp.isfinite(loc)), f"Failed for activation={act_name}"
            assert jnp.all(
                jnp.isfinite(log_scale)
            ), f"Failed for activation={act_name}"

    def test_different_input_transforms(
        self, rng, input_dim, latent_dim, hidden_dims
    ):
        """Different input transforms should all produce valid outputs."""
        x = jnp.ones((2, input_dim)) * 5.0  # positive counts
        for transform_name in ["log1p", "sqrt", "identity"]:
            enc = GaussianEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                input_transformation=transform_name,
            )
            params = enc.init(rng, jnp.zeros(input_dim))
            loc, log_scale = enc.apply(params, x)
            assert jnp.all(jnp.isfinite(loc)), (
                f"Failed for transform={transform_name}"
            )

    def test_with_standardization(
        self,
        rng,
        input_dim,
        latent_dim,
        hidden_dims,
        standardization_stats,
        sample_counts,
    ):
        """Encoder with standardization should produce different outputs."""
        mean, std = standardization_stats

        enc_no_std = GaussianEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        enc_with_std = GaussianEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            standardize_mean=mean,
            standardize_std=std,
        )

        params_no = enc_no_std.init(rng, jnp.zeros(input_dim))
        params_with = enc_with_std.init(rng, jnp.zeros(input_dim))

        loc_no, _ = enc_no_std.apply(params_no, sample_counts)
        loc_with, _ = enc_with_std.apply(params_with, sample_counts)

        # Outputs should be finite
        assert jnp.all(jnp.isfinite(loc_with))
        # Outputs should differ (standardization changes input distribution)
        assert not jnp.allclose(loc_no, loc_with, atol=1e-3)

    def test_deterministic(self, encoder, encoder_params, sample_counts):
        """Same input + same params should give identical output."""
        loc1, ls1 = encoder.apply(encoder_params, sample_counts)
        loc2, ls2 = encoder.apply(encoder_params, sample_counts)
        npt.assert_allclose(loc1, loc2)
        npt.assert_allclose(ls1, ls2)


# ---------------------------------------------------------------------------
# MultiHeadDecoder Tests
# ---------------------------------------------------------------------------


class TestMultiHeadDecoder:
    """Tests for the multi-head decoder module."""

    @pytest.fixture
    def single_head(self, input_dim):
        return (DecoderOutputHead("r", output_dim=input_dim, transform="exp"),)

    @pytest.fixture
    def multi_heads(self, input_dim):
        return (
            DecoderOutputHead("r", output_dim=input_dim, transform="exp"),
            DecoderOutputHead("gate", output_dim=input_dim, transform="sigmoid"),
        )

    @pytest.fixture
    def decoder_single(self, latent_dim, hidden_dims, single_head):
        return MultiHeadDecoder(
            output_dim=0,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            output_heads=single_head,
        )

    @pytest.fixture
    def decoder_multi(self, latent_dim, hidden_dims, multi_heads):
        return MultiHeadDecoder(
            output_dim=0,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            output_heads=multi_heads,
        )

    @pytest.fixture
    def single_params(self, decoder_single, rng, latent_dim):
        return decoder_single.init(rng, jnp.zeros(latent_dim))

    @pytest.fixture
    def multi_params(self, decoder_multi, rng, latent_dim):
        return decoder_multi.init(rng, jnp.zeros(latent_dim))

    # -- Single head tests ---

    def test_single_head_output_shape(
        self, decoder_single, single_params, input_dim, latent_dim
    ):
        z = jnp.zeros(latent_dim)
        out = decoder_single.apply(single_params, z)
        assert isinstance(out, dict)
        assert "r" in out
        assert out["r"].shape == (input_dim,)

    def test_single_head_batched(
        self, decoder_single, single_params, input_dim, latent_dim
    ):
        z = jnp.zeros((5, latent_dim))
        out = decoder_single.apply(single_params, z)
        assert out["r"].shape == (5, input_dim)

    def test_single_head_exp_positive(
        self, decoder_single, single_params, rng, latent_dim
    ):
        """exp transform should always produce positive values."""
        z = jax.random.normal(rng, (8, latent_dim))
        out = decoder_single.apply(single_params, z)
        assert jnp.all(out["r"] > 0)

    # -- Multi head tests ---

    def test_multi_head_output_keys(
        self, decoder_multi, multi_params, latent_dim
    ):
        z = jnp.zeros(latent_dim)
        out = decoder_multi.apply(multi_params, z)
        assert set(out.keys()) == {"r", "gate"}

    def test_multi_head_output_shapes(
        self, decoder_multi, multi_params, input_dim, latent_dim
    ):
        z = jnp.zeros((4, latent_dim))
        out = decoder_multi.apply(multi_params, z)
        assert out["r"].shape == (4, input_dim)
        assert out["gate"].shape == (4, input_dim)

    def test_multi_head_constraints(
        self, decoder_multi, multi_params, rng, latent_dim
    ):
        """r should be positive (exp), gate should be in (0,1) (sigmoid)."""
        z = jax.random.normal(rng, (8, latent_dim))
        out = decoder_multi.apply(multi_params, z)
        assert jnp.all(out["r"] > 0)
        assert jnp.all(out["gate"] > 0) and jnp.all(out["gate"] < 1)

    def test_output_finite(self, decoder_multi, multi_params, rng, latent_dim):
        z = jax.random.normal(rng, (8, latent_dim))
        out = decoder_multi.apply(multi_params, z)
        for name, val in out.items():
            assert jnp.all(jnp.isfinite(val)), f"Non-finite in head '{name}'"

    def test_params_structure(self, multi_params):
        """Decoder should have reversed hidden layers plus per-head Dense."""
        params = multi_params["params"]
        assert "hidden_0" in params
        assert "hidden_1" in params
        assert "head_r" in params
        assert "head_gate" in params

    def test_reversed_architecture(self, rng, input_dim, latent_dim):
        """Decoder hidden dims should be in reversed order."""
        heads = (DecoderOutputHead("r", output_dim=input_dim, transform="exp"),)
        decoder = MultiHeadDecoder(
            output_dim=0,
            latent_dim=latent_dim,
            hidden_dims=[256, 128, 64],
            output_heads=heads,
        )
        params = decoder.init(rng, jnp.zeros(latent_dim))
        p = params["params"]
        # Reversed: [64, 128, 256]
        assert p["hidden_0"]["kernel"].shape == (latent_dim, 64)
        assert p["hidden_1"]["kernel"].shape == (64, 128)
        assert p["hidden_2"]["kernel"].shape == (128, 256)
        assert p["head_r"]["kernel"].shape == (256, input_dim)

    def test_deterministic(self, decoder_multi, multi_params, rng, latent_dim):
        z = jax.random.normal(rng, (4, latent_dim))
        out1 = decoder_multi.apply(multi_params, z)
        out2 = decoder_multi.apply(multi_params, z)
        for name in out1:
            npt.assert_allclose(out1[name], out2[name])

    def test_no_heads_returns_empty_dict(self, rng, latent_dim, hidden_dims):
        """Decoder with zero heads should return empty dict."""
        decoder = MultiHeadDecoder(
            output_dim=0,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_heads=(),
        )
        params = decoder.init(rng, jnp.zeros(latent_dim))
        out = decoder.apply(params, jnp.zeros(latent_dim))
        assert out == {}


# ---------------------------------------------------------------------------
# Encoder-Decoder Integration Tests
# ---------------------------------------------------------------------------


class TestEncoderDecoderIntegration:
    """Tests for the encoder-decoder pipeline."""

    @pytest.fixture
    def encoder_decoder(self, rng, input_dim, latent_dim, hidden_dims):
        encoder = GaussianEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        heads = (DecoderOutputHead("r", output_dim=input_dim, transform="exp"),)
        decoder = MultiHeadDecoder(
            output_dim=0,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_heads=heads,
        )
        rng_enc, rng_dec = jax.random.split(rng)
        enc_params = encoder.init(rng_enc, jnp.zeros(input_dim))
        dec_params = decoder.init(rng_dec, jnp.zeros(latent_dim))
        return encoder, enc_params, decoder, dec_params

    def test_round_trip_shapes(
        self, encoder_decoder, sample_counts, latent_dim, input_dim
    ):
        """Full encode -> reparameterize -> decode pipeline."""
        encoder, enc_params, decoder, dec_params = encoder_decoder

        loc, log_scale = encoder.apply(enc_params, sample_counts)
        assert loc.shape == (8, latent_dim)

        # Reparameterization trick
        scale = jnp.exp(log_scale)
        eps = jax.random.normal(jax.random.PRNGKey(99), loc.shape)
        z = loc + scale * eps
        assert z.shape == (8, latent_dim)

        reconstruction = decoder.apply(dec_params, z)
        assert reconstruction["r"].shape == (8, input_dim)

    def test_round_trip_finite(self, encoder_decoder, sample_counts):
        """All intermediate and final values should be finite."""
        encoder, enc_params, decoder, dec_params = encoder_decoder

        loc, log_scale = encoder.apply(enc_params, sample_counts)
        assert jnp.all(jnp.isfinite(loc))
        assert jnp.all(jnp.isfinite(log_scale))

        z = loc  # use mean (no noise)
        reconstruction = decoder.apply(dec_params, z)
        assert jnp.all(jnp.isfinite(reconstruction["r"]))

    def test_gradients_flow(self, encoder_decoder, sample_counts):
        """Gradients should flow through both encoder and decoder."""
        encoder, enc_params, decoder, dec_params = encoder_decoder

        def loss_fn(enc_p, dec_p):
            loc, log_scale = encoder.apply(enc_p, sample_counts)
            reconstruction = decoder.apply(dec_p, loc)
            return jnp.mean((reconstruction["r"] - jnp.log1p(sample_counts)) ** 2)

        grads_enc, grads_dec = jax.grad(loss_fn, argnums=(0, 1))(
            enc_params, dec_params
        )

        # Check that gradients are non-zero (not vanishing)
        enc_leaves = jax.tree.leaves(grads_enc)
        dec_leaves = jax.tree.leaves(grads_dec)
        assert any(jnp.any(g != 0) for g in enc_leaves)
        assert any(jnp.any(g != 0) for g in dec_leaves)

    def test_with_standardization_round_trip(
        self, rng, input_dim, latent_dim, hidden_dims, standardization_stats,
        sample_counts,
    ):
        """Standardized encoder + destandardized decoder pipeline."""
        mean, std = standardization_stats

        encoder = GaussianEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            standardize_mean=mean,
            standardize_std=std,
        )
        heads = (DecoderOutputHead("r", output_dim=input_dim, transform="exp"),)
        decoder = MultiHeadDecoder(
            output_dim=0,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            standardize_mean=mean,
            standardize_std=std,
            output_heads=heads,
        )

        rng_enc, rng_dec = jax.random.split(rng)
        enc_params = encoder.init(rng_enc, jnp.zeros(input_dim))
        dec_params = decoder.init(rng_dec, jnp.zeros(latent_dim))

        loc, log_scale = encoder.apply(enc_params, sample_counts)
        reconstruction = decoder.apply(dec_params, loc)

        assert jnp.all(jnp.isfinite(reconstruction["r"]))
        assert reconstruction["r"].shape == sample_counts.shape

    def test_round_trip_with_covariates(
        self, rng, input_dim, latent_dim, hidden_dims
    ):
        """Encode and decode with same covariate_specs; full pipeline is finite."""
        covariate_specs = [
            CovariateSpec("batch", num_categories=4, embedding_dim=8),
        ]
        encoder = GaussianEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            covariate_specs=covariate_specs,
        )
        heads = (DecoderOutputHead("r", output_dim=input_dim, transform="exp"),)
        decoder = MultiHeadDecoder(
            output_dim=0,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            covariate_specs=covariate_specs,
            output_heads=heads,
        )
        batch_size = 4
        x = jax.random.poisson(
            rng, lam=5.0, shape=(batch_size, input_dim)
        ).astype(jnp.float32)
        covs = {"batch": jnp.array([0, 1, 2, 0])}

        rng_enc, rng_dec = jax.random.split(rng)
        enc_params = encoder.init(rng_enc, x, covariates=covs)
        dec_params = decoder.init(
            rng_dec, jnp.zeros((batch_size, latent_dim)), covariates=covs
        )

        loc, log_scale = encoder.apply(enc_params, x, covariates=covs)
        z = loc  # use mean for deterministic round-trip
        reconstruction = decoder.apply(dec_params, z, covariates=covs)

        assert reconstruction["r"].shape == (batch_size, input_dim)
        assert jnp.all(jnp.isfinite(reconstruction["r"]))


# ---------------------------------------------------------------------------
# Registry and modular API tests
# ---------------------------------------------------------------------------


class TestVAERegistryAndModularAPI:
    """Tests for ENCODER_REGISTRY, DECODER_REGISTRY, and concrete class names."""

    def test_encoder_registry_has_gaussian(self):
        assert "gaussian" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY["gaussian"] is GaussianEncoder

    def test_decoder_registry_has_multi_head(self):
        assert "multi_head" in DECODER_REGISTRY
        assert DECODER_REGISTRY["multi_head"] is MultiHeadDecoder

    def test_gaussian_encoder_is_abstract_encoder_subclass(self):
        assert issubclass(GaussianEncoder, AbstractEncoder)

    def test_multi_head_decoder_is_abstract_decoder_subclass(self):
        assert issubclass(MultiHeadDecoder, AbstractDecoder)

    def test_build_from_registry(self, rng, input_dim, latent_dim, hidden_dims):
        """Can construct encoder and decoder from registry by name."""
        EncCls = ENCODER_REGISTRY["gaussian"]
        DecCls = DECODER_REGISTRY["multi_head"]
        encoder = EncCls(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        heads = (DecoderOutputHead("r", output_dim=input_dim, transform="exp"),)
        decoder = DecCls(
            output_dim=0,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_heads=heads,
        )
        enc_params = encoder.init(rng, jnp.zeros(input_dim))
        dec_params = decoder.init(jax.random.PRNGKey(1), jnp.zeros(latent_dim))
        x = jnp.ones((2, input_dim))
        loc, log_scale = encoder.apply(enc_params, x)
        out = decoder.apply(dec_params, loc)
        assert out["r"].shape == (2, input_dim)


# ---------------------------------------------------------------------------
# Numerical Stability Tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Edge cases and numerical stability."""

    def test_large_counts(self, rng):
        """Encoder should handle very large count values."""
        encoder = GaussianEncoder(
            input_dim=10, latent_dim=3, hidden_dims=[16]
        )
        params = encoder.init(rng, jnp.zeros(10))
        x = jnp.ones((2, 10)) * 1e6  # very large counts
        loc, log_scale = encoder.apply(params, x)
        assert jnp.all(jnp.isfinite(loc))
        assert jnp.all(jnp.isfinite(log_scale))

    def test_single_hidden_layer(self, rng):
        """Minimal architecture: one hidden layer."""
        encoder = GaussianEncoder(
            input_dim=20, latent_dim=3, hidden_dims=[8]
        )
        heads = (DecoderOutputHead("r", output_dim=20, transform="exp"),)
        decoder = MultiHeadDecoder(
            output_dim=0, latent_dim=3, hidden_dims=[8], output_heads=heads,
        )
        enc_params = encoder.init(rng, jnp.zeros(20))
        dec_params = decoder.init(rng, jnp.zeros(3))

        x = jnp.ones((2, 20))
        loc, _ = encoder.apply(enc_params, x)
        recon = decoder.apply(dec_params, loc)
        assert recon["r"].shape == (2, 20)

    def test_deep_architecture(self, rng):
        """Deep architecture: four hidden layers."""
        dims = [128, 64, 32, 16]
        encoder = GaussianEncoder(
            input_dim=50, latent_dim=5, hidden_dims=dims
        )
        heads = (DecoderOutputHead("r", output_dim=50, transform="exp"),)
        decoder = MultiHeadDecoder(
            output_dim=0, latent_dim=5, hidden_dims=dims, output_heads=heads,
        )
        enc_params = encoder.init(rng, jnp.zeros(50))
        dec_params = decoder.init(rng, jnp.zeros(5))

        x = jax.random.poisson(rng, lam=10.0, shape=(4, 50)).astype(jnp.float32)
        loc, log_scale = encoder.apply(enc_params, x)
        recon = decoder.apply(dec_params, loc)

        assert jnp.all(jnp.isfinite(loc))
        assert jnp.all(jnp.isfinite(recon["r"]))

    def test_large_latent_dim(self, rng):
        """Latent dimension larger than input (unusual but valid)."""
        encoder = GaussianEncoder(
            input_dim=10, latent_dim=50, hidden_dims=[32]
        )
        heads = (DecoderOutputHead("r", output_dim=10, transform="exp"),)
        decoder = MultiHeadDecoder(
            output_dim=0, latent_dim=50, hidden_dims=[32], output_heads=heads,
        )
        enc_params = encoder.init(rng, jnp.zeros(10))
        dec_params = decoder.init(rng, jnp.zeros(50))

        x = jnp.ones((3, 10))
        loc, _ = encoder.apply(enc_params, x)
        assert loc.shape == (3, 50)

        recon = decoder.apply(dec_params, loc)
        assert recon["r"].shape == (3, 10)
