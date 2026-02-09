"""Tests for CovariateSpec, CovariateEmbedding, and covariate conditioning.

Covers:
- CovariateSpec validation
- CovariateEmbedding forward pass and shapes
- Encoder/decoder covariate conditioning
- Flow covariate conditioning (coupling + autoregressive + chain)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scribe.models.components.covariate_embedding import (
    CovariateEmbedding,
    CovariateSpec,
)


# ===========================================================================
# CovariateSpec validation
# ===========================================================================


class TestCovariateSpec:
    def test_valid_spec(self):
        spec = CovariateSpec("batch", num_categories=4, embedding_dim=8)
        assert spec.name == "batch"
        assert spec.num_categories == 4
        assert spec.embedding_dim == 8

    def test_frozen(self):
        spec = CovariateSpec("batch", num_categories=4, embedding_dim=8)
        with pytest.raises(AttributeError):
            spec.name = "donor"

    def test_invalid_num_categories(self):
        with pytest.raises(ValueError, match="num_categories must be >= 1"):
            CovariateSpec("batch", num_categories=0, embedding_dim=8)

    def test_invalid_embedding_dim(self):
        with pytest.raises(ValueError, match="embedding_dim must be >= 1"):
            CovariateSpec("batch", num_categories=4, embedding_dim=0)

    def test_negative_num_categories(self):
        with pytest.raises(ValueError, match="num_categories must be >= 1"):
            CovariateSpec("batch", num_categories=-1, embedding_dim=8)


# ===========================================================================
# CovariateEmbedding module
# ===========================================================================


class TestCovariateEmbedding:
    @pytest.fixture
    def single_spec(self):
        return [CovariateSpec("batch", num_categories=4, embedding_dim=8)]

    @pytest.fixture
    def multi_spec(self):
        return [
            CovariateSpec("batch", num_categories=4, embedding_dim=8),
            CovariateSpec("donor", num_categories=10, embedding_dim=6),
        ]

    def test_single_covariate_shape(self, single_spec):
        embedder = CovariateEmbedding(covariate_specs=single_spec)
        covs = {"batch": jnp.array([0, 1, 2, 3])}
        params = embedder.init(jax.random.PRNGKey(0), covs)
        emb = embedder.apply(params, covs)
        assert emb.shape == (4, 8)

    def test_multi_covariate_shape(self, multi_spec):
        embedder = CovariateEmbedding(covariate_specs=multi_spec)
        covs = {
            "batch": jnp.array([0, 1, 2]),
            "donor": jnp.array([3, 5, 7]),
        }
        params = embedder.init(jax.random.PRNGKey(0), covs)
        emb = embedder.apply(params, covs)
        # total_embedding_dim = 8 + 6 = 14
        assert emb.shape == (3, 14)

    def test_total_embedding_dim(self, multi_spec):
        embedder = CovariateEmbedding(covariate_specs=multi_spec)
        assert embedder.total_embedding_dim == 14

    def test_same_ids_same_embeddings(self, single_spec):
        embedder = CovariateEmbedding(covariate_specs=single_spec)
        covs = {"batch": jnp.array([2, 2, 2])}
        params = embedder.init(jax.random.PRNGKey(0), covs)
        emb = embedder.apply(params, covs)
        # All rows should be identical (same category ID)
        np.testing.assert_allclose(emb[0], emb[1], atol=1e-7)
        np.testing.assert_allclose(emb[0], emb[2], atol=1e-7)

    def test_different_ids_different_embeddings(self, single_spec):
        embedder = CovariateEmbedding(covariate_specs=single_spec)
        covs = {"batch": jnp.array([0, 1])}
        params = embedder.init(jax.random.PRNGKey(0), covs)
        emb = embedder.apply(params, covs)
        assert not jnp.allclose(emb[0], emb[1])

    def test_deterministic(self, single_spec):
        embedder = CovariateEmbedding(covariate_specs=single_spec)
        covs = {"batch": jnp.array([0, 1, 3])}
        params = embedder.init(jax.random.PRNGKey(0), covs)
        emb1 = embedder.apply(params, covs)
        emb2 = embedder.apply(params, covs)
        np.testing.assert_array_equal(emb1, emb2)

    def test_batched_input(self, single_spec):
        """Test with 2D batch of IDs."""
        embedder = CovariateEmbedding(covariate_specs=single_spec)
        covs = {"batch": jnp.array([[0, 1], [2, 3]])}
        params = embedder.init(jax.random.PRNGKey(0), covs)
        emb = embedder.apply(params, covs)
        assert emb.shape == (2, 2, 8)


# ===========================================================================
# Encoder covariate conditioning
# ===========================================================================


class TestEncoderCovariateConditioning:
    def test_gaussian_encoder_with_covariates(self):
        from scribe.models.components.vae_components import GaussianEncoder

        specs = [CovariateSpec("batch", num_categories=4, embedding_dim=8)]
        encoder = GaussianEncoder(
            input_dim=20,
            latent_dim=5,
            hidden_dims=[32, 16],
            covariate_specs=specs,
        )
        x = jnp.ones((3, 20))
        covs = {"batch": jnp.array([0, 1, 2])}
        params = encoder.init(jax.random.PRNGKey(0), x, covs)
        loc, log_scale = encoder.apply(params, x, covs)
        assert loc.shape == (3, 5)
        assert log_scale.shape == (3, 5)

    def test_encoder_without_covariates_unchanged(self):
        """Encoder without covariate_specs should work as before."""
        from scribe.models.components.vae_components import GaussianEncoder

        encoder = GaussianEncoder(
            input_dim=20,
            latent_dim=5,
            hidden_dims=[32, 16],
        )
        x = jnp.ones((3, 20))
        params = encoder.init(jax.random.PRNGKey(0), x)
        loc, log_scale = encoder.apply(params, x)
        assert loc.shape == (3, 5)
        assert log_scale.shape == (3, 5)

    def test_covariates_affect_output(self):
        """Different covariate IDs should produce different outputs."""
        from scribe.models.components.vae_components import GaussianEncoder

        specs = [CovariateSpec("batch", num_categories=4, embedding_dim=8)]
        encoder = GaussianEncoder(
            input_dim=20,
            latent_dim=5,
            hidden_dims=[32, 16],
            covariate_specs=specs,
        )
        x = jnp.ones((1, 20))

        # Init with one set of covariates
        covs_0 = {"batch": jnp.array([0])}
        covs_1 = {"batch": jnp.array([1])}
        params = encoder.init(jax.random.PRNGKey(0), x, covs_0)

        loc_0, _ = encoder.apply(params, x, covs_0)
        loc_1, _ = encoder.apply(params, x, covs_1)
        # Same input, different covariate → different output
        assert not jnp.allclose(loc_0, loc_1)


# ===========================================================================
# Decoder covariate conditioning
# ===========================================================================


class TestDecoderCovariateConditioning:
    def test_gaussian_decoder_with_covariates(self):
        from scribe.models.components.vae_components import GaussianDecoder

        specs = [CovariateSpec("batch", num_categories=4, embedding_dim=8)]
        decoder = GaussianDecoder(
            output_dim=20,
            latent_dim=5,
            hidden_dims=[32, 16],
            covariate_specs=specs,
        )
        z = jnp.ones((3, 5))
        covs = {"batch": jnp.array([0, 1, 2])}
        params = decoder.init(jax.random.PRNGKey(0), z, covs)
        out = decoder.apply(params, z, covs)
        assert out.shape == (3, 20)

    def test_decoder_without_covariates_unchanged(self):
        from scribe.models.components.vae_components import GaussianDecoder

        decoder = GaussianDecoder(
            output_dim=20,
            latent_dim=5,
            hidden_dims=[32, 16],
        )
        z = jnp.ones((3, 5))
        params = decoder.init(jax.random.PRNGKey(0), z)
        out = decoder.apply(params, z)
        assert out.shape == (3, 20)

    def test_covariates_affect_output(self):
        from scribe.models.components.vae_components import GaussianDecoder

        specs = [CovariateSpec("batch", num_categories=4, embedding_dim=8)]
        decoder = GaussianDecoder(
            output_dim=20,
            latent_dim=5,
            hidden_dims=[32, 16],
            covariate_specs=specs,
        )
        z = jnp.ones((1, 5))
        covs_0 = {"batch": jnp.array([0])}
        covs_1 = {"batch": jnp.array([1])}
        params = decoder.init(jax.random.PRNGKey(0), z, covs_0)

        out_0 = decoder.apply(params, z, covs_0)
        out_1 = decoder.apply(params, z, covs_1)
        assert not jnp.allclose(out_0, out_1)


# ===========================================================================
# Flow covariate conditioning (coupling layers)
# ===========================================================================


class TestCouplingCovariateConditioning:
    def test_affine_coupling_with_context(self):
        from scribe.flows.coupling import AffineCoupling

        layer = AffineCoupling(
            features=6,
            hidden_dims=[16, 16],
            mask_parity=0,
            context_dim=8,
        )
        x = jnp.ones((3, 6))
        ctx = jnp.ones((3, 8))
        params = layer.init(jax.random.PRNGKey(0), x, context=ctx)
        y, log_det = layer.apply(params, x, context=ctx)
        assert y.shape == (3, 6)
        assert log_det.shape == (3,)

    def test_affine_coupling_context_affects_output(self):
        from scribe.flows.coupling import AffineCoupling

        layer = AffineCoupling(
            features=6,
            hidden_dims=[16, 16],
            mask_parity=0,
            context_dim=8,
        )
        x = jnp.ones((1, 6))
        ctx_a = jnp.zeros((1, 8))
        ctx_b = jnp.ones((1, 8))
        params = layer.init(jax.random.PRNGKey(0), x, context=ctx_a)
        y_a, _ = layer.apply(params, x, context=ctx_a)
        y_b, _ = layer.apply(params, x, context=ctx_b)
        assert not jnp.allclose(y_a, y_b)

    def test_affine_coupling_invertible_with_context(self):
        from scribe.flows.coupling import AffineCoupling

        layer = AffineCoupling(
            features=6,
            hidden_dims=[16, 16],
            mask_parity=0,
            context_dim=8,
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (4, 6))
        ctx = jax.random.normal(jax.random.PRNGKey(2), (4, 8))
        params = layer.init(jax.random.PRNGKey(0), x, context=ctx)
        y, ld_fwd = layer.apply(params, x, context=ctx)
        x_rec, ld_inv = layer.apply(params, y, reverse=True, context=ctx)
        np.testing.assert_allclose(x_rec, x, atol=1e-5)
        np.testing.assert_allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_spline_coupling_with_context(self):
        from scribe.flows.coupling import SplineCoupling

        layer = SplineCoupling(
            features=6,
            hidden_dims=[16, 16],
            mask_parity=0,
            context_dim=8,
            n_bins=4,
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (3, 6)) * 0.5
        ctx = jnp.ones((3, 8))
        params = layer.init(jax.random.PRNGKey(0), x, context=ctx)
        y, log_det = layer.apply(params, x, context=ctx)
        assert y.shape == (3, 6)
        assert log_det.shape == (3,)


# ===========================================================================
# Flow covariate conditioning (autoregressive layers)
# ===========================================================================


class TestAutoregCovariateConditioning:
    def test_made_with_context(self):
        from scribe.flows.autoregressive import MADE

        made = MADE(
            features=4,
            hidden_dims=[16],
            context_dim=8,
        )
        x = jnp.ones((3, 4))
        ctx = jnp.ones((3, 8))
        params = made.init(jax.random.PRNGKey(0), x, ctx)
        shift, log_scale = made.apply(params, x, ctx)
        assert shift.shape == (3, 4)
        assert log_scale.shape == (3, 4)

    def test_made_context_affects_output(self):
        from scribe.flows.autoregressive import MADE

        made = MADE(features=4, hidden_dims=[16], context_dim=8)
        x = jnp.ones((1, 4))
        ctx_a = jnp.zeros((1, 8))
        ctx_b = jnp.ones((1, 8))
        params = made.init(jax.random.PRNGKey(0), x, ctx_a)
        shift_a, _ = made.apply(params, x, ctx_a)
        shift_b, _ = made.apply(params, x, ctx_b)
        assert not jnp.allclose(shift_a, shift_b)

    def test_maf_with_context(self):
        from scribe.flows.autoregressive import MAF

        maf = MAF(features=4, hidden_dims=[16], context_dim=8)
        x = jnp.ones((3, 4))
        ctx = jnp.ones((3, 8))
        params = maf.init(jax.random.PRNGKey(0), x, context=ctx)
        z, log_det = maf.apply(params, x, context=ctx)
        assert z.shape == (3, 4)
        assert log_det.shape == (3,)

    def test_iaf_with_context(self):
        from scribe.flows.autoregressive import IAF

        iaf = IAF(features=4, hidden_dims=[16], context_dim=8)
        x = jnp.ones((3, 4))
        ctx = jnp.ones((3, 8))
        params = iaf.init(jax.random.PRNGKey(0), x, context=ctx)
        z, log_det = iaf.apply(params, x, context=ctx)
        assert z.shape == (3, 4)
        assert log_det.shape == (3,)


# ===========================================================================
# FlowChain covariate conditioning (end-to-end)
# ===========================================================================


class TestFlowChainCovariateConditioning:
    @pytest.fixture
    def specs(self):
        return [
            CovariateSpec("batch", num_categories=4, embedding_dim=8),
            CovariateSpec("donor", num_categories=6, embedding_dim=4),
        ]

    def test_chain_with_covariates_forward(self, specs):
        from scribe.flows.base import FlowChain

        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16, 16],
            covariate_specs=specs,
        )
        x = jnp.ones((3, 6))
        covs = {"batch": jnp.array([0, 1, 2]), "donor": jnp.array([0, 3, 5])}
        params = chain.init(jax.random.PRNGKey(0), x, covariates=covs)
        z, log_det = chain.apply(params, x, covariates=covs)
        assert z.shape == (3, 6)
        assert log_det.shape == (3,)

    def test_chain_with_covariates_invertible(self, specs):
        from scribe.flows.base import FlowChain

        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16, 16],
            covariate_specs=specs,
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (4, 6))
        covs = {"batch": jnp.array([0, 1, 2, 3]), "donor": jnp.array([0, 1, 2, 3])}
        params = chain.init(jax.random.PRNGKey(0), x, covariates=covs)
        z, ld_fwd = chain.apply(params, x, covariates=covs)
        x_rec, ld_inv = chain.apply(params, z, reverse=True, covariates=covs)
        np.testing.assert_allclose(x_rec, x, atol=1e-4)
        np.testing.assert_allclose(ld_fwd + ld_inv, 0.0, atol=1e-4)

    def test_chain_covariates_affect_output(self, specs):
        from scribe.flows.base import FlowChain

        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16, 16],
            covariate_specs=specs,
        )
        x = jnp.ones((1, 6))
        covs_a = {"batch": jnp.array([0]), "donor": jnp.array([0])}
        covs_b = {"batch": jnp.array([1]), "donor": jnp.array([2])}
        params = chain.init(jax.random.PRNGKey(0), x, covariates=covs_a)
        z_a, _ = chain.apply(params, x, covariates=covs_a)
        z_b, _ = chain.apply(params, x, covariates=covs_b)
        assert not jnp.allclose(z_a, z_b)

    def test_chain_without_covariates_backward_compat(self):
        """FlowChain without covariate_specs should work exactly as before."""
        from scribe.flows.base import FlowChain

        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16, 16],
        )
        x = jnp.ones((3, 6))
        params = chain.init(jax.random.PRNGKey(0), x)
        z, log_det = chain.apply(params, x)
        assert z.shape == (3, 6)
        assert log_det.shape == (3,)

    def test_spline_chain_with_covariates(self, specs):
        from scribe.flows.base import FlowChain

        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="spline_coupling",
            hidden_dims=[16, 16],
            n_bins=4,
            covariate_specs=specs,
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (3, 6)) * 0.5
        covs = {"batch": jnp.array([0, 1, 2]), "donor": jnp.array([0, 3, 5])}
        params = chain.init(jax.random.PRNGKey(0), x, covariates=covs)
        z, log_det = chain.apply(params, x, covariates=covs)
        assert z.shape == (3, 6)
        assert log_det.shape == (3,)
