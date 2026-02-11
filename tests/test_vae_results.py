"""Tests for ``ScribeVAEResults`` and ``LatentSpaceMixin``.

Covers:
- Mixin unit tests (encoder, decoder, prior, embeddings, sampling)
- Integration tests (from_training, full pipeline, gene subsetting)
- Validation tests (missing params, None counts)
- Flow prior results tests

Device: use ``pytest --device cpu`` (default) or ``pytest --device gpu``.
"""

import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO

from scribe.flows import FlowChain
from scribe.models.builders import (
    BetaSpec,
    LogNormalSpec,
    ModelBuilder,
    GuideBuilder,
)
from scribe.models.builders.parameter_specs import GaussianLatentSpec
from scribe.models.components import (
    MeanFieldGuide,
    NegativeBinomialLikelihood,
)
from scribe.models.components.guide_families import VAELatentGuide
from scribe.models.components.vae_components import (
    DecoderOutputHead,
    GaussianEncoder,
    MultiHeadDecoder,
)
from scribe.models.config import ModelConfigBuilder
from scribe.models.config.enums import ModelType
from scribe.svi.vae_results import ScribeVAEResults

# ==============================================================================
# Constants
# ==============================================================================

LATENT_DIM = 5
N_GENES = 20
N_CELLS = 30
HIDDEN_DIMS = [16]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def model_config():
    """Create a basic ModelConfig (same pattern as test_vae_integration)."""
    return ModelConfigBuilder().for_model(ModelType.NBDM).build()


@pytest.fixture(scope="session")
def rng_key():
    """Create a PRNG key."""
    return random.PRNGKey(0)


@pytest.fixture(scope="session")
def small_counts(rng_key):
    """Create a small count matrix."""
    return random.poisson(
        rng_key, lam=10.0, shape=(N_CELLS, N_GENES)
    ).astype(jnp.float32)


@pytest.fixture(scope="session")
def encoder():
    """Create a GaussianEncoder."""
    return GaussianEncoder(
        input_dim=N_GENES,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
    )


@pytest.fixture(scope="session")
def decoder():
    """Create a MultiHeadDecoder with a single 'r' head."""
    return MultiHeadDecoder(
        output_dim=0,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_heads=(
            DecoderOutputHead("r", output_dim=N_GENES, transform="exp"),
        ),
    )


@pytest.fixture(scope="session")
def latent_spec():
    """Create a GaussianLatentSpec."""
    return GaussianLatentSpec(latent_dim=LATENT_DIM)


@pytest.fixture(scope="session")
def vae_guide_family(encoder, decoder, latent_spec):
    """Create a VAELatentGuide."""
    return VAELatentGuide(
        encoder=encoder,
        decoder=decoder,
        latent_spec=latent_spec,
    )


def _build_and_train_vae(
    vae_guide_family, model_config, counts, rng_key, n_steps=50
):
    """Build a VAE model+guide and train for a few SVI steps.

    Parameters
    ----------
    vae_guide_family : VAELatentGuide
        The guide family with encoder, decoder, latent_spec.
    model_config : ModelConfig
        Model configuration.
    counts : jnp.ndarray
        Training count matrix.
    rng_key : PRNGKey
        PRNG key.
    n_steps : int
        Number of SVI steps to run.

    Returns
    -------
    params : Dict
        Trained params from ``svi.get_params(svi_state)``.
    losses : list
        Loss values from training.
    """
    # Build model and guide
    z_spec = BetaSpec(
        name="z_marker",
        shape_dims=(),
        default_params=(1.0, 1.0),
        is_cell_specific=True,
        guide_family=vae_guide_family,
    )
    p_spec = BetaSpec(
        name="p",
        shape_dims=(),
        default_params=(1.0, 1.0),
        guide_family=MeanFieldGuide(),
    )

    model = ModelBuilder()
    guide = GuideBuilder()
    for spec in [p_spec, z_spec]:
        model.add_param(spec)
        guide.add_param(spec)
    model.with_likelihood(NegativeBinomialLikelihood())

    model_fn = model.build()
    guide_fn = guide.build()

    # Train
    svi = SVI(model_fn, guide_fn, numpyro.optim.Adam(1e-3), Trace_ELBO())
    svi_state = svi.init(
        rng_key,
        n_cells=N_CELLS,
        n_genes=N_GENES,
        model_config=model_config,
        counts=counts,
    )

    losses = []
    for _ in range(n_steps):
        svi_state, loss = svi.update(
            svi_state,
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=counts,
        )
        losses.append(float(loss))

    params = svi.get_params(svi_state)
    return params, losses


@pytest.fixture(scope="session")
def trained_vae(request, vae_guide_family, model_config, small_counts, rng_key):
    """Train a VAE once per session; return (params, losses, vae_guide_family).

    Device is controlled by ``--device`` (default: cpu). Session scope avoids
    re-running SVI for every test.
    """
    device_type = request.config.getoption("--device", default="cpu")
    if device_type == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        jax.config.update("jax_platform_name", "cpu")
    else:
        if "JAX_PLATFORM_NAME" in os.environ:
            del os.environ["JAX_PLATFORM_NAME"]

    params, losses = _build_and_train_vae(
        vae_guide_family, model_config, small_counts, rng_key
    )
    return params, losses, vae_guide_family


@pytest.fixture
def vae_results(trained_vae, model_config):
    """Create a ScribeVAEResults from trained VAE."""
    params, losses, vgf = trained_vae
    return ScribeVAEResults(
        params=params,
        loss_history=jnp.array(losses),
        n_cells=N_CELLS,
        n_genes=N_GENES,
        model_type="vae_nb",
        model_config=model_config,
        prior_params={},
        _encoder=vgf.encoder,
        _decoder=vgf.decoder,
        _latent_spec=vgf.latent_spec,
    )


# ==============================================================================
# LatentSpaceMixin unit tests
# ==============================================================================


class TestLatentSpaceMixin:
    """Unit tests for LatentSpaceMixin methods."""

    def test_run_encoder_returns_var_params_dict(
        self, vae_results, small_counts
    ):
        """_run_encoder should return a dict with variational parameters.

        For Gaussian encoder this should have keys 'loc' and 'log_scale'.
        """
        var_params = vae_results._run_encoder(small_counts)
        assert isinstance(var_params, dict)
        assert "loc" in var_params
        assert "log_scale" in var_params
        assert var_params["loc"].shape == (N_CELLS, LATENT_DIM)
        assert var_params["log_scale"].shape == (N_CELLS, LATENT_DIM)

    def test_run_decoder_produces_correct_output_keys(self, vae_results):
        """_run_decoder should return dict with keys matching output heads.

        The decoder has a single head 'r', so the output dict should
        have key 'r'.
        """
        z = jnp.ones((N_CELLS, LATENT_DIM))
        decoded = vae_results._run_decoder(z)
        assert isinstance(decoded, dict)
        assert "r" in decoded
        assert decoded["r"].shape == (N_CELLS, N_GENES)

    def test_build_prior_distribution_without_flow(self, vae_results):
        """Without flow, prior should be Independent(Normal, 1).

        Samples from the prior should have shape (latent_dim,) per sample.
        """
        prior = vae_results._build_prior_distribution()
        assert prior.event_shape == (LATENT_DIM,)
        sample = prior.sample(random.PRNGKey(0))
        assert sample.shape == (LATENT_DIM,)
        assert jnp.all(jnp.isfinite(sample))

    def test_get_latent_embeddings_shape_and_finiteness(
        self, vae_results, small_counts
    ):
        """get_latent_embeddings should return (n_cells, latent_dim).

        All values should be finite.
        """
        embeddings = vae_results.get_latent_embeddings(small_counts)
        assert embeddings.shape == (N_CELLS, LATENT_DIM)
        assert jnp.all(jnp.isfinite(embeddings))

    def test_get_latent_embeddings_batched(self, vae_results, small_counts):
        """Batched embedding should match unbatched result.

        The batch_size parameter is for memory efficiency — the result
        should be identical.
        """
        full = vae_results.get_latent_embeddings(small_counts)
        batched = vae_results.get_latent_embeddings(
            small_counts, batch_size=10
        )
        np.testing.assert_allclose(
            np.array(full), np.array(batched), atol=1e-5
        )

    def test_get_latent_embeddings_none_counts_raises(self, vae_results):
        """Passing None counts should raise ValueError.

        The encoder needs input data to produce embeddings.
        """
        with pytest.raises(ValueError, match="counts is required"):
            vae_results.get_latent_embeddings(None)

    def test_get_latent_samples_shape(self, vae_results):
        """get_latent_samples should return (n_samples, latent_dim).

        Samples from the standard Normal prior.
        """
        n_samples = 50
        samples = vae_results.get_latent_samples(n_samples=n_samples)
        assert samples.shape == (n_samples, LATENT_DIM)
        assert jnp.all(jnp.isfinite(samples))

    def test_get_latent_samples_stores_when_requested(self, vae_results):
        """store_samples=True should cache result in latent_samples.

        This avoids re-sampling when the same samples are needed
        multiple times.
        """
        assert vae_results.latent_samples is None
        vae_results.get_latent_samples(n_samples=10, store_samples=True)
        assert vae_results.latent_samples is not None
        assert vae_results.latent_samples.shape == (10, LATENT_DIM)

    def test_get_latent_samples_conditioned_on_data(
        self, vae_results, small_counts
    ):
        """Posterior samples should have shape (n_samples, n_cells, latent_dim).

        These are reparameterized samples from q(z|x).
        """
        n_samples = 7
        samples = vae_results.get_latent_samples_conditioned_on_data(
            small_counts, n_samples=n_samples
        )
        assert samples.shape == (n_samples, N_CELLS, LATENT_DIM)
        assert jnp.all(jnp.isfinite(samples))

    def test_get_latent_samples_conditioned_batched(
        self, vae_results, small_counts
    ):
        """Batched conditioned sampling should match unbatched shape.

        With batch_size set, the result should still have the correct
        shape and be finite.
        """
        n_samples = 5
        samples = vae_results.get_latent_samples_conditioned_on_data(
            small_counts, n_samples=n_samples, batch_size=10
        )
        assert samples.shape == (n_samples, N_CELLS, LATENT_DIM)
        assert jnp.all(jnp.isfinite(samples))

    def test_get_decoded_params(self, vae_results):
        """get_decoded_params should return dict matching decoder heads.

        Given a latent z, the decoder should produce parameters for
        each output head.
        """
        z = jnp.ones((N_CELLS, LATENT_DIM))
        decoded = vae_results.get_decoded_params(z)
        assert "r" in decoded
        assert decoded["r"].shape == (N_CELLS, N_GENES)
        # 'r' uses exp transform, so all values should be positive
        assert jnp.all(decoded["r"] > 0)


# ==============================================================================
# ScribeVAEResults integration tests
# ==============================================================================


class TestScribeVAEResultsIntegration:
    """Integration tests for ScribeVAEResults."""

    def test_from_training_creates_valid_results(
        self, trained_vae, model_config
    ):
        """from_training classmethod should create a valid results object.

        All VAE-specific fields should be properly set from the
        VAELatentGuide.
        """
        params, losses, vgf = trained_vae
        results = ScribeVAEResults.from_training(
            params=params,
            loss_history=jnp.array(losses),
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            vae_guide_family=vgf,
        )
        assert results._encoder is vgf.encoder
        assert results._decoder is vgf.decoder
        assert results._latent_spec is vgf.latent_spec
        assert results.n_cells == N_CELLS
        assert results.n_genes == N_GENES

    def test_full_encode_sample_decode_pipeline(
        self, vae_results, small_counts
    ):
        """Full pipeline: encode -> sample z -> decode -> param dict.

        This tests the end-to-end latent-space inference workflow.
        """
        # Encode
        var_params = vae_results._run_encoder(small_counts)
        assert "loc" in var_params

        # Sample
        from scribe.svi._latent_dispatch import sample_latent_posterior

        z = sample_latent_posterior(
            vae_results._latent_spec,
            var_params,
            random.PRNGKey(0),
            3,
        )
        assert z.shape == (3, N_CELLS, LATENT_DIM)

        # Decode (take one sample)
        decoded = vae_results.get_decoded_params(z[0])
        assert "r" in decoded
        assert decoded["r"].shape == (N_CELLS, N_GENES)

    def test_gene_subsetting(self, vae_results):
        """Gene subsetting should produce correct shapes.

        results[0:5] should return a new results with n_genes=5 and
        decoder params subsetted along the gene dimension.
        """
        subset = vae_results[0:5]
        assert subset.n_genes == 5
        assert subset._decoder is not vae_results._decoder
        # Check decoder output heads have updated output_dim
        for head in subset._decoder.output_heads:
            assert head.output_dim == 5

    def test_gene_subsetting_preserves_encoding(
        self, vae_results, small_counts
    ):
        """Subsetted results should still encode successfully.

        The encoder operates on the full gene space, but the decoder
        heads are subsetted.  Encoding + decoding should still produce
        valid output with the subsetted decoder.
        """
        subset = vae_results[0:5]

        # Encoding uses the original encoder (same n_genes)
        embeddings = subset.get_latent_embeddings(small_counts)
        assert embeddings.shape == (N_CELLS, LATENT_DIM)

        # Decoding with subsetted decoder produces fewer genes
        z = jnp.ones((N_CELLS, LATENT_DIM))
        decoded = subset.get_decoded_params(z)
        assert decoded["r"].shape == (N_CELLS, 5)

    def test_gene_subsetting_with_boolean_mask(self, vae_results):
        """Boolean mask gene subsetting should work correctly.

        Tests the alternative indexing path (boolean mask instead of
        slice).
        """
        mask = jnp.zeros(N_GENES, dtype=bool)
        mask = mask.at[:7].set(True)
        subset = vae_results[mask]
        assert subset.n_genes == 7


# ==============================================================================
# Validation tests
# ==============================================================================


class TestScribeVAEResultsValidation:
    """Tests for ScribeVAEResults validation logic."""

    def test_missing_encoder_params_raises(
        self, encoder, decoder, latent_spec, model_config
    ):
        """Params missing 'vae_encoder$params' should raise ValueError.

        The encoder params are required to run inference-time encoding.
        """
        with pytest.raises(ValueError, match="vae_encoder"):
            ScribeVAEResults(
                params={"vae_decoder$params": {}},
                loss_history=jnp.array([1.0]),
                n_cells=N_CELLS,
                n_genes=N_GENES,
                model_type="vae",
                model_config=model_config,
                prior_params={},
                _encoder=encoder,
                _decoder=decoder,
                _latent_spec=latent_spec,
            )

    def test_missing_decoder_params_raises(
        self, encoder, decoder, latent_spec, model_config
    ):
        """Params missing 'vae_decoder$params' should raise ValueError.

        The decoder params are required to run inference-time decoding.
        """
        with pytest.raises(ValueError, match="vae_decoder"):
            ScribeVAEResults(
                params={"vae_encoder$params": {}},
                loss_history=jnp.array([1.0]),
                n_cells=N_CELLS,
                n_genes=N_GENES,
                model_type="vae",
                model_config=model_config,
                prior_params={},
                _encoder=encoder,
                _decoder=decoder,
                _latent_spec=latent_spec,
            )

    def test_missing_encoder_module_raises(
        self, decoder, latent_spec, model_config
    ):
        """Missing _encoder module should raise ValueError.

        The encoder Linen module is required for all encoding operations.
        """
        with pytest.raises(ValueError, match="_encoder"):
            ScribeVAEResults(
                params={
                    "vae_encoder$params": {},
                    "vae_decoder$params": {},
                },
                loss_history=jnp.array([1.0]),
                n_cells=N_CELLS,
                n_genes=N_GENES,
                model_type="vae",
                model_config=model_config,
                prior_params={},
                _encoder=None,
                _decoder=decoder,
                _latent_spec=latent_spec,
            )

    def test_flow_without_flow_params_raises(
        self, encoder, decoder, model_config
    ):
        """Flow set but params missing 'vae_prior_flow$params' should raise.

        When the latent spec has a flow prior, the corresponding trained
        parameters must be present in the params dict.
        """
        flow = FlowChain(
            features=LATENT_DIM,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16],
        )
        latent_spec_with_flow = GaussianLatentSpec(
            latent_dim=LATENT_DIM, flow=flow
        )
        with pytest.raises(ValueError, match="vae_prior_flow"):
            ScribeVAEResults(
                params={
                    "vae_encoder$params": {},
                    "vae_decoder$params": {},
                },
                loss_history=jnp.array([1.0]),
                n_cells=N_CELLS,
                n_genes=N_GENES,
                model_type="vae",
                model_config=model_config,
                prior_params={},
                _encoder=encoder,
                _decoder=decoder,
                _latent_spec=latent_spec_with_flow,
            )


# ==============================================================================
# Flow prior results tests
# ==============================================================================


class TestFlowPriorResults:
    """Tests for ScribeVAEResults with flow-based priors."""

    @pytest.fixture
    def flow_vae_results(self, model_config, small_counts, rng_key):
        """Train a VAE with flow prior and return results.

        Uses a 2-layer affine coupling flow as the prior on z.
        """
        prior_flow = FlowChain(
            features=LATENT_DIM,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16],
        )
        encoder = GaussianEncoder(
            input_dim=N_GENES,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        decoder = MultiHeadDecoder(
            output_dim=0,
            latent_dim=LATENT_DIM,
            hidden_dims=HIDDEN_DIMS,
            output_heads=(
                DecoderOutputHead(
                    "r", output_dim=N_GENES, transform="exp"
                ),
            ),
        )
        latent_spec = GaussianLatentSpec(
            latent_dim=LATENT_DIM, flow=prior_flow
        )
        vgf = VAELatentGuide(
            encoder=encoder, decoder=decoder, latent_spec=latent_spec
        )

        params, losses = _build_and_train_vae(
            vgf, model_config, small_counts, rng_key
        )

        return ScribeVAEResults.from_training(
            params=params,
            loss_history=jnp.array(losses),
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            vae_guide_family=vgf,
        )

    def test_build_prior_with_flow(self, flow_vae_results):
        """With flow, prior should be a FlowDistribution.

        Samples should be finite and have correct shape.
        """
        from scribe.flows import FlowDistribution

        prior = flow_vae_results._build_prior_distribution()
        assert isinstance(prior, FlowDistribution)
        sample = prior.sample(random.PRNGKey(0))
        assert sample.shape == (LATENT_DIM,)
        assert jnp.all(jnp.isfinite(sample))

    def test_flow_prior_get_latent_samples(self, flow_vae_results):
        """Samples from flow prior should be finite.

        The learned flow should produce valid latent samples.
        """
        samples = flow_vae_results.get_latent_samples(n_samples=20)
        assert samples.shape == (20, LATENT_DIM)
        assert jnp.all(jnp.isfinite(samples))
