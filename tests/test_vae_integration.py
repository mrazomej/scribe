"""Integration tests for the VAE path through model/guide builders.

Tests that the full VAE pipeline works:
1. Model builder detects VAELatentGuide and constructs vae_cell_fn
2. Guide builder runs encoder and samples z
3. SVI training with ELBO converges (loss decreases)
4. Prior predictive sampling produces valid counts
5. Batched SVI training works
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive

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
    ZeroInflatedNBLikelihood,
    NBWithVCPLikelihood,
    GaussianEncoder,
    MultiHeadDecoder,
    DecoderOutputHead,
)
from scribe.models.components.guide_families import VAELatentGuide
from scribe.models.config import ModelConfigBuilder


# ==============================================================================
# Constants
# ==============================================================================

N_CELLS = 50
N_GENES = 20
LATENT_DIM = 5
HIDDEN_DIMS = [32]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def rng_key():
    return random.PRNGKey(42)


@pytest.fixture
def small_counts():
    key = random.PRNGKey(0)
    return random.poisson(key, lam=10.0, shape=(N_CELLS, N_GENES)).astype(
        jnp.float32
    )


@pytest.fixture
def model_config():
    from scribe.models.config.enums import ModelType

    return ModelConfigBuilder().for_model(ModelType.NBDM).build()


@pytest.fixture
def vae_guide_family():
    """Create a VAELatentGuide with encoder, decoder, and latent_spec."""
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
            DecoderOutputHead("r", output_dim=N_GENES, transform="exp"),
        ),
    )
    latent_spec = GaussianLatentSpec(latent_dim=LATENT_DIM)
    return VAELatentGuide(
        encoder=encoder,
        decoder=decoder,
        latent_spec=latent_spec,
    )


def _build_nb_vae_model_and_guide(vae_guide_family):
    """Build a simple NB VAE model and guide.

    Model: p ~ Beta(1,1) globally, r from decoder, counts ~ NB(r, p)
    Guide: encoder maps counts → z; p has MeanField guide
    """
    # The VAE marker spec: a cell-specific spec with VAELatentGuide.
    # This tells both builders about the VAE. The spec type doesn't matter
    # much (it's skipped in both guide and model loops).
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

    specs = [p_spec, z_spec]

    model = ModelBuilder()
    guide = GuideBuilder()
    for spec in specs:
        model.add_param(spec)
        guide.add_param(spec)
    model.with_likelihood(NegativeBinomialLikelihood())

    return model.build(), guide.build()


# ==============================================================================
# Tests
# ==============================================================================


class TestVAEModelBuilderDetection:
    """Test that the model builder correctly detects VAE components."""

    def test_vae_cell_fn_constructed(
        self, vae_guide_family, model_config, small_counts, rng_key
    ):
        """Model should run without error when VAELatentGuide is present."""
        model_fn, guide_fn = _build_nb_vae_model_and_guide(vae_guide_family)

        # The model should trace successfully (no duplicate sample sites,
        # no missing params)
        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(model_fn, rng_key)
        ).get_trace(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
        )

        # z should be in the trace (sampled by vae_cell_fn)
        assert "z" in trace
        assert trace["z"]["value"].shape == (N_CELLS, LATENT_DIM)

        # r should be a deterministic site (from decoder)
        assert "r" in trace
        assert trace["r"]["value"].shape == (N_CELLS, N_GENES)

        # p should be sampled from the prior
        assert "p" in trace

        # counts should be observed
        assert "counts" in trace

    def test_no_vae_when_no_vae_guide(
        self, model_config, small_counts, rng_key
    ):
        """Without VAELatentGuide, model should behave as standard NB."""
        p_spec = BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0))
        r_spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
        )

        model = ModelBuilder()
        model.add_param(p_spec).add_param(r_spec)
        model.with_likelihood(NegativeBinomialLikelihood())
        model_fn = model.build()

        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(model_fn, rng_key)
        ).get_trace(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
        )

        # z should NOT be in the trace
        assert "z" not in trace
        # r should be a sample site (from prior), not deterministic
        assert "r" in trace
        assert trace["r"]["type"] == "sample"


class TestVAEPriorPredictive:
    """Test prior predictive sampling through the VAE path."""

    def test_prior_predictive_produces_counts(
        self, vae_guide_family, model_config, rng_key
    ):
        """Prior predictive (counts=None) should produce synthetic counts."""
        model_fn, _ = _build_nb_vae_model_and_guide(vae_guide_family)

        predictive = Predictive(model_fn, num_samples=3)
        samples = predictive(
            rng_key,
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=None,
        )

        assert "counts" in samples
        assert samples["counts"].shape == (3, N_CELLS, N_GENES)
        # Counts should be non-negative integers
        assert jnp.all(samples["counts"] >= 0)

    def test_prior_predictive_z_shape(
        self, vae_guide_family, model_config, rng_key
    ):
        """z samples from prior should have correct shape."""
        model_fn, _ = _build_nb_vae_model_and_guide(vae_guide_family)

        predictive = Predictive(model_fn, num_samples=2)
        samples = predictive(
            rng_key,
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=None,
        )

        assert "z" in samples
        assert samples["z"].shape == (2, N_CELLS, LATENT_DIM)


class TestVAESVITraining:
    """Test SVI training with the VAE model/guide pair."""

    def test_svi_init_and_step(
        self, vae_guide_family, model_config, small_counts, rng_key
    ):
        """SVI should initialize and run a step without error."""
        model_fn, guide_fn = _build_nb_vae_model_and_guide(vae_guide_family)

        svi = SVI(model_fn, guide_fn, numpyro.optim.Adam(1e-3), Trace_ELBO())
        svi_state = svi.init(
            rng_key,
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
        )

        svi_state, loss = svi.update(
            svi_state,
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
        )

        assert jnp.isfinite(loss)

    def test_svi_loss_decreases(
        self, vae_guide_family, model_config, small_counts, rng_key
    ):
        """SVI loss should decrease over training steps."""
        model_fn, guide_fn = _build_nb_vae_model_and_guide(vae_guide_family)

        svi = SVI(model_fn, guide_fn, numpyro.optim.Adam(5e-3), Trace_ELBO())
        svi_state = svi.init(
            rng_key,
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
        )

        losses = []
        for _ in range(200):
            svi_state, loss = svi.update(
                svi_state,
                n_cells=N_CELLS,
                n_genes=N_GENES,
                model_config=model_config,
                counts=small_counts,
            )
            losses.append(float(loss))

        # All losses should be finite
        assert all(np.isfinite(l) for l in losses)
        # Average of last 20 should be lower than average of first 20
        early_avg = np.mean(losses[:20])
        late_avg = np.mean(losses[-20:])
        assert (
            late_avg < early_avg
        ), f"Loss did not decrease: early={early_avg:.1f}, late={late_avg:.1f}"

    def test_svi_with_batching(
        self, vae_guide_family, model_config, small_counts, rng_key
    ):
        """SVI with mini-batching should work."""
        model_fn, guide_fn = _build_nb_vae_model_and_guide(vae_guide_family)
        batch_size = 16

        svi = SVI(model_fn, guide_fn, numpyro.optim.Adam(1e-3), Trace_ELBO())
        svi_state = svi.init(
            rng_key,
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
            batch_size=batch_size,
        )

        svi_state, loss = svi.update(
            svi_state,
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
            batch_size=batch_size,
        )

        assert jnp.isfinite(loss)


class TestVAEMultiHead:
    """Test VAE with multi-head decoder (multiple output parameters)."""

    def test_multi_head_decoder_in_model(
        self, model_config, small_counts, rng_key
    ):
        """Decoder with two heads should produce both params in trace."""
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
                DecoderOutputHead("r", output_dim=N_GENES, transform="exp"),
                DecoderOutputHead(
                    "gate", output_dim=N_GENES, transform="sigmoid"
                ),
            ),
        )
        latent_spec = GaussianLatentSpec(latent_dim=LATENT_DIM)

        vae_guide_family = VAELatentGuide(
            encoder=encoder, decoder=decoder, latent_spec=latent_spec
        )

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
        model.add_param(p_spec).add_param(z_spec)
        model.with_likelihood(ZeroInflatedNBLikelihood())
        model_fn = model.build()

        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(model_fn, rng_key)
        ).get_trace(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
        )

        # Both decoder heads should appear
        assert "r" in trace
        assert "gate" in trace
        assert trace["r"]["value"].shape == (N_CELLS, N_GENES)
        assert trace["gate"]["value"].shape == (N_CELLS, N_GENES)

        # gate should be in (0, 1) due to sigmoid transform
        gate_vals = trace["gate"]["value"]
        assert jnp.all(gate_vals > 0) and jnp.all(gate_vals < 1)

        # r should be positive due to exp transform
        assert jnp.all(trace["r"]["value"] > 0)

    def test_multi_head_svi_training(self, model_config, small_counts, rng_key):
        """Multi-head VAE should train with SVI."""
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
                DecoderOutputHead("r", output_dim=N_GENES, transform="exp"),
                DecoderOutputHead(
                    "gate", output_dim=N_GENES, transform="sigmoid"
                ),
            ),
        )
        latent_spec = GaussianLatentSpec(latent_dim=LATENT_DIM)

        vae_gf = VAELatentGuide(
            encoder=encoder, decoder=decoder, latent_spec=latent_spec
        )

        z_spec = BetaSpec(
            name="z_marker",
            shape_dims=(),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            guide_family=vae_gf,
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
        model.with_likelihood(ZeroInflatedNBLikelihood())

        model_fn = model.build()
        guide_fn = guide.build()

        svi = SVI(model_fn, guide_fn, numpyro.optim.Adam(1e-3), Trace_ELBO())
        svi_state = svi.init(
            rng_key,
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
        )

        for _ in range(5):
            svi_state, loss = svi.update(
                svi_state,
                n_cells=N_CELLS,
                n_genes=N_GENES,
                model_config=model_config,
                counts=small_counts,
            )

        assert jnp.isfinite(loss)


class TestVAEWithVCPLikelihood:
    """VAE path works with Variable Capture Probability likelihoods."""

    def test_vae_nbvcp_trace_has_z_r_p_capture(
        self, vae_guide_family, model_config, small_counts, rng_key
    ):
        """Model with VAE + NBWithVCPLikelihood should have z, r (deterministic), p, p_capture."""
        p_spec = BetaSpec(
            name="p",
            shape_dims=(),
            default_params=(1.0, 1.0),
            guide_family=MeanFieldGuide(),
        )
        p_capture_spec = BetaSpec(
            name="p_capture",
            shape_dims=(),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            guide_family=MeanFieldGuide(),
        )
        z_spec = BetaSpec(
            name="z_marker",
            shape_dims=(),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            guide_family=vae_guide_family,
        )

        model = ModelBuilder()
        for spec in [p_spec, p_capture_spec, z_spec]:
            model.add_param(spec)
        model.with_likelihood(
            NBWithVCPLikelihood(capture_param_name="p_capture")
        )
        model_fn = model.build()

        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(model_fn, rng_key)
        ).get_trace(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            model_config=model_config,
            counts=small_counts,
        )

        assert "z" in trace
        assert trace["z"]["value"].shape == (N_CELLS, LATENT_DIM)
        assert "r" in trace
        assert trace["r"]["value"].shape == (N_CELLS, N_GENES)
        assert "p" in trace
        assert "p_capture" in trace
        assert "counts" in trace


class TestVAEMixtureExclusion:
    """VAE and mixture models are mutually exclusive."""

    def test_vae_with_mixture_raises(self, vae_guide_family):
        """Building a model with both VAE and mixture specs should fail."""
        z_spec = BetaSpec(
            name="z_marker",
            shape_dims=(),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            guide_family=vae_guide_family,
        )
        # A mixture-specific parameter
        r_spec = LogNormalSpec(
            name="r",
            shape_dims=("n_components", "n_genes"),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            is_mixture=True,
        )

        model = ModelBuilder()
        model.add_param(z_spec).add_param(r_spec)
        model.with_likelihood(NegativeBinomialLikelihood())

        with pytest.raises(ValueError, match="VAE and mixture models cannot"):
            model.build()


class TestVAELatentGuideParamNames:
    """Test that VAELatentGuide.param_names derives from decoder heads."""

    def test_param_names_from_single_head(self):
        decoder = MultiHeadDecoder(
            output_dim=0,
            latent_dim=5,
            hidden_dims=[32],
            output_heads=(
                DecoderOutputHead("r", output_dim=20, transform="exp"),
            ),
        )
        vg = VAELatentGuide(decoder=decoder)
        assert vg.param_names == ["r"]

    def test_param_names_from_multi_head(self):
        decoder = MultiHeadDecoder(
            output_dim=0,
            latent_dim=5,
            hidden_dims=[32],
            output_heads=(
                DecoderOutputHead("r", output_dim=20, transform="exp"),
                DecoderOutputHead("gate", output_dim=20, transform="sigmoid"),
            ),
        )
        vg = VAELatentGuide(decoder=decoder)
        assert vg.param_names == ["r", "gate"]

    def test_param_names_empty_without_decoder(self):
        vg = VAELatentGuide()
        assert vg.param_names == []
