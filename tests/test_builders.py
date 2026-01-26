"""Tests for the composable model builder system.

This module tests:
1. Parameter specifications (ParamSpec subclasses)
2. Model building via ModelBuilder
3. Guide building via GuideBuilder
4. Preset factory functions
5. GuideFamilyConfig for per-parameter guide families
6. Equivalence with legacy models (where applicable)
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from scribe.models.config import ModelConfigBuilder, GuideFamilyConfig

# Import builder components
from scribe.models.builders import (
    BetaSpec,
    LogNormalSpec,
    BetaPrimeSpec,
    SigmoidNormalSpec,
    ExpNormalSpec,
    ModelBuilder,
    GuideBuilder,
    resolve_shape,
)

# Import guide families
from scribe.models.components import (
    MeanFieldGuide,
    LowRankGuide,
    NegativeBinomialLikelihood,
    ZeroInflatedNBLikelihood,
    NBWithVCPLikelihood,
    Amortizer,
    AmortizedGuide,
    TOTAL_COUNT,
)

# Import unified factory
from scribe.models.presets import create_model_from_params

# Import main API
from scribe.models import get_model_and_guide
from scribe.inference.preset_builder import build_config_from_preset

# Import parameterizations
from scribe.models.parameterizations import (
    PARAMETERIZATIONS,
    CanonicalParameterization,
    MeanProbParameterization,
    MeanOddsParameterization,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def rng_key():
    """Provide a fixed random key for reproducibility."""
    return random.PRNGKey(42)


@pytest.fixture
def small_counts():
    """Small count matrix for testing."""
    key = random.PRNGKey(0)
    # Generate some realistic-ish counts
    return random.poisson(key, lam=10.0, shape=(50, 20)).astype(jnp.float32)


@pytest.fixture
def model_config():
    """Basic model config for testing."""
    from scribe.models.config import ModelType

    return ModelConfigBuilder().for_model(ModelType.NBDM).build()


# ==============================================================================
# Test Parameter Specifications
# ==============================================================================


class TestParameterSpecs:
    """Test parameter specification classes."""

    def test_resolve_shape_scalar(self):
        """Test resolving empty shape (scalar)."""
        dims = {"n_cells": 100, "n_genes": 50}
        assert resolve_shape((), dims) == ()

    def test_resolve_shape_gene_specific(self):
        """Test resolving gene-specific shape."""
        dims = {"n_cells": 100, "n_genes": 50}
        assert resolve_shape(("n_genes",), dims) == (50,)

    def test_resolve_shape_cell_specific(self):
        """Test resolving cell-specific shape."""
        dims = {"n_cells": 100, "n_genes": 50}
        assert resolve_shape(("n_cells",), dims) == (100,)

    def test_beta_spec_support(self):
        """Test BetaSpec has correct support."""
        spec = BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0))
        assert spec.support.is_discrete == False  # unit_interval is continuous

    def test_beta_spec_arg_constraints(self):
        """Test BetaSpec has correct arg_constraints."""
        spec = BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0))
        assert "concentration1" in spec.arg_constraints
        assert "concentration0" in spec.arg_constraints

    def test_lognormal_spec_support(self):
        """Test LogNormalSpec has correct support."""
        spec = LogNormalSpec(
            name="r", shape_dims=("n_genes",), default_params=(0.0, 1.0)
        )
        # positive support
        assert hasattr(spec, "support")
        assert hasattr(spec, "arg_constraints")

    def test_sigmoid_normal_spec_support(self):
        """Test SigmoidNormalSpec has correct support via transform."""
        spec = SigmoidNormalSpec(
            name="p", shape_dims=(), default_params=(0.0, 1.0)
        )
        # Support derived from transform codomain
        assert hasattr(spec, "support")
        assert hasattr(spec, "arg_constraints")

    def test_exp_normal_spec_support(self):
        """Test ExpNormalSpec has correct support via transform."""
        spec = ExpNormalSpec(
            name="r", shape_dims=("n_genes",), default_params=(0.0, 1.0)
        )
        # Support derived from transform codomain
        assert hasattr(spec, "support")
        assert hasattr(spec, "arg_constraints")


# ==============================================================================
# Test Model Builder
# ==============================================================================


class TestModelBuilder:
    """Test ModelBuilder class."""

    def test_build_simple_model(self, model_config, small_counts):
        """Test building a simple NBDM-like model."""
        model = (
            ModelBuilder()
            .add_param(
                BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0))
            )
            .add_param(
                LogNormalSpec(
                    name="r",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                )
            )
            .with_likelihood(NegativeBinomialLikelihood())
            .build()
        )

        # Model should be callable
        assert callable(model)

        # Test prior predictive (no counts)
        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                model(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=None,
                )

        # Check expected sample sites exist
        assert "p" in tr
        assert "r" in tr
        assert "counts" in tr

    def test_build_with_derived_params(self, model_config, small_counts):
        """Test building model with derived parameters (linked parameterization)."""
        model = (
            ModelBuilder()
            .add_param(
                BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0))
            )
            .add_param(
                LogNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                )
            )
            .add_derived("r", lambda p, mu: mu * (1 - p) / p, ["p", "mu"])
            .with_likelihood(NegativeBinomialLikelihood())
            .build()
        )

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                model(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=None,
                )

        # Check derived param exists
        assert "r" in tr
        # r should be deterministic
        assert tr["r"]["type"] == "deterministic"

    def test_build_requires_likelihood(self):
        """Test that build() fails without likelihood."""
        builder = ModelBuilder().add_param(
            BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0))
        )

        with pytest.raises(ValueError, match="Likelihood must be set"):
            builder.build()


# ==============================================================================
# Test Guide Builder
# ==============================================================================


class TestGuideBuilder:
    """Test GuideBuilder class."""

    def test_build_mean_field_guide(self, model_config, small_counts):
        """Test building a mean-field guide."""
        specs = [
            BetaSpec(
                name="p",
                shape_dims=(),
                default_params=(1.0, 1.0),
                guide_family=MeanFieldGuide(),
            ),
            LogNormalSpec(
                name="r",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=MeanFieldGuide(),
            ),
        ]

        guide = GuideBuilder().from_specs(specs).build()

        # Guide should be callable
        assert callable(guide)

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                guide(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=small_counts,
                )

        # Check expected sample sites
        assert "p" in tr
        assert "r" in tr

    def test_build_low_rank_guide(self, model_config, small_counts):
        """Test building a guide with low-rank component."""
        specs = [
            BetaSpec(
                name="p",
                shape_dims=(),
                default_params=(1.0, 1.0),
                guide_family=MeanFieldGuide(),
            ),
            LogNormalSpec(
                name="r",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=LowRankGuide(rank=5),
            ),
        ]

        guide = GuideBuilder().from_specs(specs).build()

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                guide(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=small_counts,
                )

        assert "p" in tr
        assert "r" in tr


# ==============================================================================
# Test Presets
# ==============================================================================


class TestPresets:
    """Test preset factory functions."""

    @pytest.mark.parametrize(
        "parameterization", ["standard", "linked", "odds_ratio"]
    )
    def test_create_nbdm(self, parameterization, model_config, small_counts):
        """Test create_model_from_params with different parameterizations."""
        model, guide = create_model_from_params(
            model="nbdm", parameterization=parameterization
        )

        # Both should be callable
        assert callable(model)
        assert callable(guide)

        # Test they run without error
        with numpyro.handlers.seed(rng_seed=0):
            model(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )
            guide(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    @pytest.mark.parametrize("unconstrained", [False, True])
    def test_create_nbdm_unconstrained(
        self, unconstrained, model_config, small_counts
    ):
        """Test create_model_from_params with unconstrained option."""
        model, guide = create_model_from_params(
            model="nbdm", unconstrained=unconstrained
        )

        with numpyro.handlers.seed(rng_seed=0):
            model(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    def test_create_nbdm_low_rank(self, model_config, small_counts):
        """Test create_model_from_params with low-rank guide using GuideFamilyConfig."""
        model, guide = create_model_from_params(
            model="nbdm",
            guide_families=GuideFamilyConfig(r=LowRankGuide(rank=5)),
        )

        with numpyro.handlers.seed(rng_seed=0):
            guide(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    def test_create_nbdm_linked_low_rank(self, model_config, small_counts):
        """Test create_model_from_params linked parameterization with low-rank guide for mu."""
        model, guide = create_model_from_params(
            model="nbdm",
            parameterization="linked",
            guide_families=GuideFamilyConfig(mu=LowRankGuide(rank=5)),
        )

        with numpyro.handlers.seed(rng_seed=0):
            guide(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    def test_create_zinb(self, model_config, small_counts):
        """Test create_model_from_params for ZINB."""
        model, guide = create_model_from_params(model="zinb")

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                model(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=small_counts,
                )

        # ZINB should have gate parameter
        assert "gate" in tr

    def test_create_nbvcp(self, model_config, small_counts):
        """Test create_model_from_params for NBVCP."""
        model, guide = create_model_from_params(model="nbvcp")

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                model(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=small_counts,
                )

        # NBVCP should have p_capture parameter
        assert "p_capture" in tr

    def test_create_nbvcp_amortized(self, model_config, small_counts):
        """Test create_model_from_params with amortized p_capture using GuideFamilyConfig."""
        # Create amortizer for p_capture
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["log_alpha", "log_beta"],
        )
        model, guide = create_model_from_params(
            model="nbvcp",
            guide_families=GuideFamilyConfig(
                p_capture=AmortizedGuide(amortizer=amortizer)
            ),
        )

        with numpyro.handlers.seed(rng_seed=0):
            guide(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    def test_create_nbvcp_mixed_guides(self, model_config, small_counts):
        """Test create_model_from_params with mixed guide families."""
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["log_alpha", "log_beta"],
        )
        model, guide = create_model_from_params(
            model="nbvcp",
            parameterization="linked",
            guide_families=GuideFamilyConfig(
                p=MeanFieldGuide(),
                mu=LowRankGuide(rank=5),
                p_capture=AmortizedGuide(amortizer=amortizer),
            ),
        )

        with numpyro.handlers.seed(rng_seed=0):
            guide(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    def test_create_zinbvcp(self, model_config, small_counts):
        """Test create_model_from_params for ZINBVCP."""
        model, guide = create_model_from_params(model="zinbvcp")

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                model(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=small_counts,
                )

        # Should have both gate and p_capture
        assert "gate" in tr
        assert "p_capture" in tr


# ==============================================================================
# Test Main API
# ==============================================================================


class TestMainAPI:
    """Test get_model_and_guide function."""

    @pytest.mark.parametrize("model_type", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_all_model_types(self, model_type, model_config, small_counts):
        """Test v2 API supports all model types."""
        config = build_config_from_preset(model=model_type)
        model, guide = get_model_and_guide(config)

        with numpyro.handlers.seed(rng_seed=0):
            model(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )
            guide(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    def test_with_guide_families(self, model_config, small_counts):
        """Test get_model_and_guide with GuideFamilyConfig options."""
        config = build_config_from_preset(
            model="nbdm",
            parameterization="linked",
            guide_rank=5,  # This creates LowRankGuide for mu
        )
        model, guide = get_model_and_guide(config)

        with numpyro.handlers.seed(rng_seed=0):
            guide(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    def test_with_amortized_guide(self, model_config, small_counts):
        """Test get_model_and_guide with amortized guide for p_capture."""
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["log_alpha", "log_beta"],
        )
        # Build config with amortized guide using ModelConfigBuilder
        from scribe.models.config import ModelConfigBuilder

        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("canonical")
            .with_inference("svi")
            .with_guide_families(
                GuideFamilyConfig(p_capture=AmortizedGuide(amortizer=amortizer))
            )
            .build()
        )
        model, guide = get_model_and_guide(config)

        with numpyro.handlers.seed(rng_seed=0):
            guide(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    def test_unknown_model_type(self):
        """Test that ModelConfigBuilder raises error for unknown model type."""
        from scribe.models.config import ModelConfigBuilder
        from pydantic import ValidationError

        # ModelConfigBuilder validates model type at build time
        with pytest.raises(
            (ValueError, ValidationError), match="Invalid model type"
        ):
            config = (
                ModelConfigBuilder()
                .for_model("unknown_model")
                .with_parameterization("canonical")
                .with_inference("svi")
                .build()
            )


# ==============================================================================
# Test SVI Training (Integration)
# ==============================================================================


class TestSVIIntegration:
    """Integration tests with SVI training."""

    def test_svi_training_nbdm(self, model_config, small_counts, rng_key):
        """Test that built NBDM model/guide can be trained with SVI."""
        model, guide = create_model_from_params(model="nbdm")

        optimizer = Adam(1e-2)
        svi = SVI(model, guide, optimizer, Trace_ELBO())

        # Run a few steps
        svi_state = svi.init(
            rng_key,
            n_cells=small_counts.shape[0],
            n_genes=small_counts.shape[1],
            model_config=model_config,
            counts=small_counts,
        )

        for _ in range(3):
            svi_state, loss = svi.update(
                svi_state,
                n_cells=small_counts.shape[0],
                n_genes=small_counts.shape[1],
                model_config=model_config,
                counts=small_counts,
            )

        # Loss should be finite
        assert jnp.isfinite(loss)

    def test_svi_training_with_batching(
        self, model_config, small_counts, rng_key
    ):
        """Test SVI training with mini-batching."""
        model, guide = create_model_from_params(model="nbdm")

        optimizer = Adam(1e-2)
        svi = SVI(model, guide, optimizer, Trace_ELBO())

        # Run with batch_size
        svi_state = svi.init(
            rng_key,
            n_cells=small_counts.shape[0],
            n_genes=small_counts.shape[1],
            model_config=model_config,
            counts=small_counts,
            batch_size=10,
        )

        svi_state, loss = svi.update(
            svi_state,
            n_cells=small_counts.shape[0],
            n_genes=small_counts.shape[1],
            model_config=model_config,
            counts=small_counts,
            batch_size=10,
        )

        assert jnp.isfinite(loss)

    def test_amortizer_parameters_registered(
        self, model_config, small_counts, rng_key
    ):
        """Test that amortizer parameters are registered with NumPyro and optimized."""
        from scribe.models.components import (
            Amortizer,
            AmortizedGuide,
            TOTAL_COUNT,
        )
        from scribe.models.config import GuideFamilyConfig
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.optim import Adam

        # Create amortizer for p_capture
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["log_alpha", "log_beta"],
        )
        model, guide = create_model_from_params(
            model="nbvcp",
            guide_families=GuideFamilyConfig(
                p_capture=AmortizedGuide(amortizer=amortizer)
            ),
        )

        optimizer = Adam(1e-2)
        svi = SVI(model, guide, optimizer, Trace_ELBO())

        # Initialize SVI
        svi_state = svi.init(
            rng_key,
            n_cells=small_counts.shape[0],
            n_genes=small_counts.shape[1],
            model_config=model_config,
            counts=small_counts,
        )

        # Get initial parameters
        params = svi.get_params(svi_state)

        # Check that amortizer parameters are registered
        # The module name should be "p_capture_amortizer" based on our implementation
        # NumPyro's flax_module stores parameters as "module_name$params"
        amortizer_param_key = "p_capture_amortizer$params"
        
        assert (
            amortizer_param_key in params
        ), f"Amortizer parameters should be registered with NumPyro. Found params: {list(params.keys())[:10]}"

        # Verify the parameters have the expected structure
        # The amortizer has Linear layers, so the params should be a nested structure
        amortizer_params = params[amortizer_param_key]
        
        # The params should be a dict/pytree containing the amortizer's parameters
        # We can check that it's not empty and has some structure
        import jax.tree_util as jtu
        param_leaves = jtu.tree_leaves(amortizer_params)
        assert len(param_leaves) > 0, "Amortizer params should contain parameter arrays"

        # Run a few optimization steps
        # Store initial amortizer params for comparison
        import jax.tree_util as jtu
        initial_amortizer_params = jtu.tree_map(lambda x: x.copy(), amortizer_params)

        for _ in range(5):
            svi_state, loss = svi.update(
                svi_state,
                n_cells=small_counts.shape[0],
                n_genes=small_counts.shape[1],
                model_config=model_config,
                counts=small_counts,
            )

        # Get updated parameters
        updated_params = svi.get_params(svi_state)
        updated_amortizer_params = updated_params[amortizer_param_key]

        # Verify parameters have changed (they were optimized)
        # Compare the pytrees
        def params_differ(x, y):
            return not jnp.allclose(x, y)
        
        differences = jtu.tree_map(params_differ, initial_amortizer_params, updated_amortizer_params)
        params_changed = any(jtu.tree_leaves(differences))
        
        assert (
            params_changed
        ), "Amortizer parameters should change during optimization"


# ==============================================================================
# Test Mixture Models
# ==============================================================================


class TestMixtureModels:
    """Test mixture model functionality."""

    def test_is_mixture_validation(self):
        """Test that is_mixture and is_cell_specific cannot both be True."""
        from scribe.models.builders import BetaSpec

        # This should work
        spec1 = BetaSpec(
            name="p", shape_dims=(), default_params=(1.0, 1.0), is_mixture=True
        )
        assert spec1.is_mixture is True

        # This should fail
        with pytest.raises(ValueError, match="is_mixture and is_cell_specific"):
            BetaSpec(
                name="p_capture",
                shape_dims=("n_cells",),
                default_params=(1.0, 1.0),
                is_cell_specific=True,
                is_mixture=True,
            )

    def test_resolve_shape_with_mixture(self):
        """Test shape resolution with mixture parameters."""
        from scribe.models.builders import resolve_shape

        dims = {"n_genes": 50, "n_components": 3}

        # Non-mixture gene-specific
        shape = resolve_shape(("n_genes",), dims, is_mixture=False)
        assert shape == (50,)

        # Mixture gene-specific
        shape = resolve_shape(("n_genes",), dims, is_mixture=True)
        assert shape == (3, 50)

        # Non-mixture scalar
        shape = resolve_shape((), dims, is_mixture=False)
        assert shape == ()

        # Mixture scalar
        shape = resolve_shape((), dims, is_mixture=True)
        assert shape == (3,)

    def test_model_builder_is_mixture_property(self):
        """Test ModelBuilder.is_mixture property."""
        from scribe.models.builders import BetaSpec, LogNormalSpec, ModelBuilder

        builder = ModelBuilder()
        assert builder.is_mixture is False

        builder.add_param(
            BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0))
        )
        assert builder.is_mixture is False

        builder.add_param(
            LogNormalSpec(
                name="r",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_mixture=True,
            )
        )
        assert builder.is_mixture is True

    def test_create_nbdm_mixture(self, small_counts):
        """Test creating NBDM mixture model."""
        from scribe.models.config import ModelConfigBuilder, ModelType

        # Create model config with n_components
        model_config = (
            ModelConfigBuilder()
            .for_model(ModelType.NBDM)
            .as_mixture(n_components=3)
            .build()
        )

        model, guide = create_model_from_params(model="nbdm", n_components=3)

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                model(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=small_counts,
                )

        # Should have mixing_weights
        assert "mixing_weights" in tr
        # r should have shape (n_components, n_genes)
        assert tr["r"]["value"].shape == (3, 20)

    def test_create_nbdm_mixture_specific_params(self, small_counts):
        """Test creating NBDM mixture with specific mixture params."""
        from scribe.models.config import ModelConfigBuilder, ModelType

        model_config = (
            ModelConfigBuilder()
            .for_model(ModelType.NBDM)
            .as_mixture(n_components=3)
            .build()
        )

        # Only r is mixture-specific, p is shared
        model, guide = create_model_from_params(
            model="nbdm", n_components=3, mixture_params=["r"]
        )

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                model(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=small_counts,
                )

        # r should be (3, 20) - mixture-specific
        assert tr["r"]["value"].shape == (3, 20)
        # p should be scalar - shared
        assert tr["p"]["value"].shape == ()

    def test_create_zinb_mixture(self, small_counts):
        """Test creating ZINB mixture model."""
        from scribe.models.config import ModelConfigBuilder, ModelType

        model_config = (
            ModelConfigBuilder()
            .for_model(ModelType.ZINB)
            .as_mixture(n_components=3)
            .build()
        )

        model, guide = create_model_from_params(model="zinb", n_components=3)

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                model(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=small_counts,
                )

        # Should have mixing_weights
        assert "mixing_weights" in tr
        # mu should have shape (n_components, n_genes) if mixture
        # gate should also be mixture if default behavior
        assert "gate" in tr


# ==============================================================================
# Test Parameterizations
# ==============================================================================


class TestParameterizations:
    """Test Parameterization strategy classes."""

    def test_canonical_parameterization(self, model_config):
        """Test CanonicalParameterization."""
        param_strategy = CanonicalParameterization()

        assert param_strategy.name == "canonical"
        assert param_strategy.core_parameters == ["p", "r"]
        assert param_strategy.gene_param_name == "r"
        assert param_strategy.transform_model_param("p_capture") == "p_capture"

        # Test building specs
        guide_families = GuideFamilyConfig()
        param_specs = param_strategy.build_param_specs(
            unconstrained=False, guide_families=guide_families
        )
        assert len(param_specs) == 2
        assert param_specs[0].name == "p"
        assert param_specs[1].name == "r"

        derived_params = param_strategy.build_derived_params()
        assert len(derived_params) == 0

    def test_mean_prob_parameterization(self, model_config):
        """Test MeanProbParameterization."""
        param_strategy = MeanProbParameterization()

        assert param_strategy.name == "mean_prob"
        assert param_strategy.core_parameters == ["p", "mu"]
        assert param_strategy.gene_param_name == "mu"
        assert param_strategy.transform_model_param("p_capture") == "p_capture"

        # Test building specs
        guide_families = GuideFamilyConfig()
        param_specs = param_strategy.build_param_specs(
            unconstrained=False, guide_families=guide_families
        )
        assert len(param_specs) == 2
        assert param_specs[0].name == "p"
        assert param_specs[1].name == "mu"

        derived_params = param_strategy.build_derived_params()
        assert len(derived_params) == 1
        assert derived_params[0].name == "r"
        assert derived_params[0].deps == ["p", "mu"]

    def test_mean_odds_parameterization(self, model_config):
        """Test MeanOddsParameterization."""
        param_strategy = MeanOddsParameterization()

        assert param_strategy.name == "mean_odds"
        assert param_strategy.core_parameters == ["phi", "mu"]
        assert param_strategy.gene_param_name == "mu"
        assert (
            param_strategy.transform_model_param("p_capture") == "phi_capture"
        )
        assert param_strategy.transform_model_param("other") == "other"

        # Test building specs
        guide_families = GuideFamilyConfig()
        param_specs = param_strategy.build_param_specs(
            unconstrained=False, guide_families=guide_families
        )
        assert len(param_specs) == 2
        assert param_specs[0].name == "phi"
        assert param_specs[1].name == "mu"

        derived_params = param_strategy.build_derived_params()
        assert len(derived_params) == 2
        assert derived_params[0].name == "r"
        assert derived_params[1].name == "p"

    def test_parameterization_registry(self):
        """Test that all parameterizations are in the registry."""
        # New names
        assert "canonical" in PARAMETERIZATIONS
        assert "mean_prob" in PARAMETERIZATIONS
        assert "mean_odds" in PARAMETERIZATIONS
        # Backward compatibility
        assert "standard" in PARAMETERIZATIONS
        assert "linked" in PARAMETERIZATIONS
        assert "odds_ratio" in PARAMETERIZATIONS

        # Check they map to correct classes
        assert isinstance(
            PARAMETERIZATIONS["canonical"], CanonicalParameterization
        )
        assert isinstance(
            PARAMETERIZATIONS["standard"], CanonicalParameterization
        )
        assert isinstance(
            PARAMETERIZATIONS["mean_prob"], MeanProbParameterization
        )
        assert isinstance(PARAMETERIZATIONS["linked"], MeanProbParameterization)
        assert isinstance(
            PARAMETERIZATIONS["mean_odds"], MeanOddsParameterization
        )
        assert isinstance(
            PARAMETERIZATIONS["odds_ratio"], MeanOddsParameterization
        )

    def test_parameterization_with_new_names(self, model_config, small_counts):
        """Test factory works with new parameterization names."""
        # Test canonical
        model, guide = create_model_from_params(
            model="nbdm", parameterization="canonical"
        )
        with numpyro.handlers.seed(rng_seed=0):
            model(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

        # Test mean_prob
        model, guide = create_model_from_params(
            model="nbdm", parameterization="mean_prob"
        )
        with numpyro.handlers.seed(rng_seed=0):
            model(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

        # Test mean_odds
        model, guide = create_model_from_params(
            model="nbdm", parameterization="mean_odds"
        )
        with numpyro.handlers.seed(rng_seed=0):
            model(
                n_cells=50,
                n_genes=20,
                model_config=model_config,
                counts=small_counts,
            )

    def test_nbvcp_with_mean_odds_phi_capture(self, model_config, small_counts):
        """Test NBVCP with mean_odds uses phi_capture instead of p_capture."""
        model, guide = create_model_from_params(
            model="nbvcp", parameterization="mean_odds"
        )

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                model(
                    n_cells=50,
                    n_genes=20,
                    model_config=model_config,
                    counts=small_counts,
                )

        # Should have phi_capture, not p_capture
        assert "phi_capture" in tr
        assert "p_capture" not in tr
        # Should have derived p and r
        assert "p" in tr
        assert "r" in tr


# ==============================================================================
# Test GuideFamilyConfig
# ==============================================================================


class TestGuideFamilyConfig:
    """Test GuideFamilyConfig class."""

    def test_default_creates_empty_config(self):
        """Test that default GuideFamilyConfig has all None values."""
        config = GuideFamilyConfig()
        assert config.p is None
        assert config.r is None
        assert config.mu is None
        assert config.p_capture is None

    def test_get_returns_mean_field_for_none(self):
        """Test that get() returns MeanFieldGuide for unspecified params."""
        config = GuideFamilyConfig()
        family = config.get("r")
        assert isinstance(family, MeanFieldGuide)

    def test_get_returns_specified_family(self):
        """Test that get() returns the specified family."""
        config = GuideFamilyConfig(r=LowRankGuide(rank=10))
        family = config.get("r")
        assert isinstance(family, LowRankGuide)
        assert family.rank == 10

    def test_get_unknown_param_returns_mean_field(self):
        """Test that get() returns MeanFieldGuide for unknown params."""
        config = GuideFamilyConfig()
        family = config.get("unknown_param")
        assert isinstance(family, MeanFieldGuide)

    def test_config_is_immutable(self):
        """Test that GuideFamilyConfig is immutable (frozen)."""
        config = GuideFamilyConfig(r=LowRankGuide(rank=10))
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            config.r = MeanFieldGuide()

    def test_config_rejects_extra_fields(self):
        """Test that GuideFamilyConfig rejects unknown fields."""
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            GuideFamilyConfig(unknown_field=MeanFieldGuide())

    def test_mixed_guide_families(self):
        """Test creating config with mixed guide families."""
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32],
            output_params=["log_alpha", "log_beta"],
        )
        config = GuideFamilyConfig(
            p=MeanFieldGuide(),
            r=LowRankGuide(rank=5),
            p_capture=AmortizedGuide(amortizer=amortizer),
        )

        assert isinstance(config.get("p"), MeanFieldGuide)
        assert isinstance(config.get("r"), LowRankGuide)
        assert isinstance(config.get("p_capture"), AmortizedGuide)
        # Unspecified still returns MeanField
        assert isinstance(config.get("gate"), MeanFieldGuide)


# ==============================================================================
# Test Amortizer
# ==============================================================================


class TestAmortizer:
    """Test Amortizer component."""

    def test_amortizer_creation(self):
        """Test creating an amortizer."""
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["log_alpha", "log_beta"],
        )

        # Test forward pass (Linen modules need initialization)
        counts = jnp.ones((10, 50))
        # Initialize with dummy input to get params
        rng_key = random.PRNGKey(0)
        params = amortizer.init(rng_key, counts)
        outputs = amortizer.apply(params, counts)

        assert "log_alpha" in outputs
        assert "log_beta" in outputs
        assert outputs["log_alpha"].shape == (10,)
        assert outputs["log_beta"].shape == (10,)

    def test_total_count_statistic(self):
        """Test TOTAL_COUNT sufficient statistic."""
        counts = jnp.array([[10, 20, 30], [5, 5, 5]])
        result = TOTAL_COUNT.compute(counts)

        # log1p(sum) for each row
        expected = jnp.log1p(jnp.array([[60], [15]]))
        assert jnp.allclose(result, expected)

    def test_activation_function_mapping(self):
        """Test _get_activation_fn maps activation names to JAX functions."""
        from scribe.models.components.amortizers import _get_activation_fn

        # Test all supported activations
        activations = [
            "relu",
            "gelu",
            "silu",
            "swish",  # alias for silu
            "tanh",
            "sigmoid",
            "elu",
            "leaky_relu",
            "softplus",
            "celu",
            "selu",
        ]

        for act in activations:
            fn = _get_activation_fn(act)
            assert callable(fn)
            # Test that it works on a simple array
            x = jnp.array([-1.0, 0.0, 1.0])
            result = fn(x)
            assert result.shape == x.shape

        # Test that swish maps to silu
        assert _get_activation_fn("swish") == _get_activation_fn("silu")

        # Test invalid activation raises error
        with pytest.raises(ValueError, match="Unknown activation"):
            _get_activation_fn("invalid_activation")

    def test_amortizer_with_custom_activation(self):
        """Test Amortizer accepts and uses custom activation function."""
        # Test with different activations
        activations = ["relu", "gelu", "leaky_relu", "silu", "tanh"]

        for activation in activations:
            amortizer = Amortizer(
                sufficient_statistic=TOTAL_COUNT,
                hidden_dims=[32, 16],
                output_params=["log_alpha", "log_beta"],
                activation=activation,
            )

            assert amortizer.activation == activation
            # In Linen, activation_fn is computed in __call__, so we test it works
            from scribe.models.components.amortizers import _get_activation_fn
            assert callable(_get_activation_fn(activation))

            # Test forward pass works (Linen modules need initialization)
            counts = jnp.ones((10, 50))
            rng_key = random.PRNGKey(0)
            params = amortizer.init(rng_key, counts)
            outputs = amortizer.apply(params, counts)

            assert "log_alpha" in outputs
            assert "log_beta" in outputs
            assert outputs["log_alpha"].shape == (10,)
            assert outputs["log_beta"].shape == (10,)

    def test_amortizer_default_activation(self):
        """Test Amortizer defaults to relu if activation not specified."""
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["log_alpha", "log_beta"],
        )

        assert amortizer.activation == "relu"
        # In Linen, activation_fn is computed in __call__, so we test it works
        from scribe.models.components.amortizers import _get_activation_fn
        assert _get_activation_fn(amortizer.activation) == jax.nn.relu

    def test_amortizer_output_order(self):
        """Test that amortizer outputs maintain consistent order."""
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["log_alpha", "log_beta"],
        )

        counts = jnp.ones((10, 50))
        # Linen modules need initialization
        rng_key = random.PRNGKey(0)
        params = amortizer.init(rng_key, counts)
        outputs = amortizer.apply(params, counts)

        # Verify output keys match output_params order
        assert list(outputs.keys()) == amortizer.output_params
        assert "log_alpha" in outputs
        assert "log_beta" in outputs
        assert outputs["log_alpha"].shape == (10,)
        assert outputs["log_beta"].shape == (10,)

    def test_amortizer_jit_compilation(self):
        """Test that amortizer can be JIT compiled."""
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["log_alpha", "log_beta"],
        )

        # Initialize to get params
        counts = jnp.ones((10, 50))
        rng_key = random.PRNGKey(0)
        params = amortizer.init(rng_key, counts)

        @jax.jit
        def jitted_forward(params, data):
            return amortizer.apply(params, data)

        # This should compile without errors
        outputs = jitted_forward(params, counts)

        assert "log_alpha" in outputs
        assert "log_beta" in outputs
        assert outputs["log_alpha"].shape == (10,)
        assert outputs["log_beta"].shape == (10,)
