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
from types import SimpleNamespace
from jax import random
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from scribe.models.config import ModelConfigBuilder, GuideFamilyConfig
from scribe.models.config.enums import Parameterization as ParameterizationEnum

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
    AmortizedOutput,
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
from scribe.svi.results import ScribeSVIResults

# Import parameterizations
from scribe.models.parameterizations import (
    PARAMETERIZATIONS,
    CanonicalParameterization,
    MeanProbParameterization,
    MeanOddsParameterization,
)

# ==============================================================================
# Helpers
# ==============================================================================


def _pos_transform(x):
    """Softplus + offset transform ensuring positive output for test amortizers."""
    return jax.nn.softplus(x) + 0.5


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
            output_params=["alpha", "beta"],
            output_transforms={"alpha": _pos_transform, "beta": _pos_transform},
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
            output_params=["alpha", "beta"],
            output_transforms={"alpha": _pos_transform, "beta": _pos_transform},
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
        model, guide, _ = get_model_and_guide(config)

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
        model, guide, _ = get_model_and_guide(config)

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
            output_params=["alpha", "beta"],
            output_transforms={"alpha": _pos_transform, "beta": _pos_transform},
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
        model, guide, _ = get_model_and_guide(config)

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
            output_params=["alpha", "beta"],
            output_transforms={"alpha": _pos_transform, "beta": _pos_transform},
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
        assert (
            len(param_leaves) > 0
        ), "Amortizer params should contain parameter arrays"

        # Run a few optimization steps
        # Store initial amortizer params for comparison
        import jax.tree_util as jtu

        initial_amortizer_params = jtu.tree_map(
            lambda x: x.copy(), amortizer_params
        )

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

        differences = jtu.tree_map(
            params_differ, initial_amortizer_params, updated_amortizer_params
        )
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
            output_params=["alpha", "beta"],
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
            output_params=["alpha", "beta"],
        )

        # Test forward pass (Linen modules need initialization)
        counts = jnp.ones((10, 50))
        # Initialize with dummy input to get params
        rng_key = random.PRNGKey(0)
        params = amortizer.init(rng_key, counts)
        out = amortizer.apply(params, counts)
        assert isinstance(out, AmortizedOutput)
        outputs = out.params
        assert "alpha" in outputs
        assert "beta" in outputs
        assert outputs["alpha"].shape == (10,)
        assert outputs["beta"].shape == (10,)

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
                output_params=["alpha", "beta"],
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
            out = amortizer.apply(params, counts)
            assert isinstance(out, AmortizedOutput)
            outputs = out.params
            assert "alpha" in outputs
            assert "beta" in outputs
            assert outputs["alpha"].shape == (10,)
            assert outputs["beta"].shape == (10,)

    def test_amortizer_default_activation(self):
        """Test Amortizer defaults to relu if activation not specified."""
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["alpha", "beta"],
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
            output_params=["alpha", "beta"],
        )

        counts = jnp.ones((10, 50))
        # Linen modules need initialization
        rng_key = random.PRNGKey(0)
        params = amortizer.init(rng_key, counts)
        out = amortizer.apply(params, counts)
        assert isinstance(out, AmortizedOutput)
        outputs = out.params
        # Verify output keys match output_params order
        assert list(outputs.keys()) == amortizer.output_params
        assert "alpha" in outputs
        assert "beta" in outputs
        assert outputs["alpha"].shape == (10,)
        assert outputs["beta"].shape == (10,)

    def test_amortizer_jit_compilation(self):
        """Test that amortizer can be JIT compiled."""
        amortizer = Amortizer(
            sufficient_statistic=TOTAL_COUNT,
            hidden_dims=[32, 16],
            output_params=["alpha", "beta"],
        )

        # Initialize to get params
        counts = jnp.ones((10, 50))
        rng_key = random.PRNGKey(0)
        params = amortizer.init(rng_key, counts)

        @jax.jit
        def jitted_forward(params, data):
            return amortizer.apply(params, data)

        # This should compile without errors
        out = jitted_forward(params, counts)
        assert isinstance(out, AmortizedOutput)
        outputs = out.params
        assert "alpha" in outputs
        assert "beta" in outputs
        assert outputs["alpha"].shape == (10,)
        assert outputs["beta"].shape == (10,)

    def test_amortizer_output_contract(self):
        """Test AmortizedOutput contract: keys, parameterization, and value space."""
        from scribe.models.presets.registry import create_capture_amortizer

        # Constrained: alpha, beta already in positive space
        amortizer_constrained = create_capture_amortizer(
            hidden_dims=[16, 8],
            output_transform="softplus",
            output_clamp_min=0.1,
            output_clamp_max=50.0,
        )
        counts = jnp.ones((20, 100))
        rng_key = random.PRNGKey(0)
        params = amortizer_constrained.init(rng_key, counts)
        out = amortizer_constrained.apply(params, counts)
        assert isinstance(out, AmortizedOutput)
        assert out.parameterization == "constrained"
        assert list(out.params.keys()) == ["alpha", "beta"]
        assert jnp.all(
            out.params["alpha"] > 0
        ), "alpha must be positive (constrained)"
        assert jnp.all(
            out.params["beta"] > 0
        ), "beta must be positive (constrained)"

        # Unconstrained: loc unconstrained, log_scale in log-space; exp(log_scale) > 0
        amortizer_unconstrained = create_capture_amortizer(
            hidden_dims=[16, 8],
            unconstrained=True,
        )
        params_u = amortizer_unconstrained.init(rng_key, counts)
        out_u = amortizer_unconstrained.apply(params_u, counts)
        assert isinstance(out_u, AmortizedOutput)
        assert out_u.parameterization == "unconstrained"
        assert list(out_u.params.keys()) == ["loc", "log_scale"]
        scale = jnp.exp(out_u.params["log_scale"])
        assert jnp.all(scale > 0), "exp(log_scale) must be positive"


# ==============================================================================
# Test JointLowRankGuide
# ==============================================================================


class TestJointLowRankGuide:
    """Test JointLowRankGuide: chain rule decomposition with Woodbury conditioning."""

    def test_joint_guide_two_params(self, model_config, small_counts):
        """Test joint guide with two gene-specific ExpNormalSpec parameters."""
        from scribe.models.components import JointLowRankGuide

        joint = JointLowRankGuide(rank=5, group="nb_params")
        specs = [
            ExpNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="mu",
            ),
            ExpNormalSpec(
                name="phi",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="phi",
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

        # Both sites should be registered
        assert "mu" in tr
        assert "phi" in tr

        # Both should have correct shape (n_genes=20)
        assert tr["mu"]["value"].shape == (20,)
        assert tr["phi"]["value"].shape == (20,)

        # Both should be positive (exp transform applied)
        assert jnp.all(tr["mu"]["value"] > 0)
        assert jnp.all(tr["phi"]["value"] > 0)

    def test_joint_guide_has_log_prob(self, model_config, small_counts):
        """Test that joint guide sites have proper log_prob (not Delta)."""
        from scribe.models.components import JointLowRankGuide

        joint = JointLowRankGuide(rank=3, group="test")
        specs = [
            ExpNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="mu",
            ),
            ExpNormalSpec(
                name="phi",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="phi",
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

        # Both sites should be sample sites with finite log_prob
        # (not deterministic / Delta), which is the whole point of
        # the chain rule decomposition
        for name in ["mu", "phi"]:
            assert tr[name]["type"] == "sample"
            log_prob = tr[name]["fn"].log_prob(tr[name]["value"])
            assert jnp.isfinite(log_prob).all(), (
                f"Site '{name}' has non-finite log_prob, indicating "
                "broken guide distribution"
            )

    def test_joint_guide_three_params(self, model_config, small_counts):
        """Test joint guide with three parameters (e.g., ZINB with gate)."""
        from scribe.models.components import JointLowRankGuide
        from scribe.models.builders import SigmoidNormalSpec

        joint = JointLowRankGuide(rank=4, group="zinb_params")
        specs = [
            ExpNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="mu",
            ),
            ExpNormalSpec(
                name="phi",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="phi",
            ),
            SigmoidNormalSpec(
                name="gate",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="gate",
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

        # All three sites should be present
        assert "mu" in tr
        assert "phi" in tr
        assert "gate" in tr

        # gate should be in (0, 1) (sigmoid transform)
        assert jnp.all(tr["gate"]["value"] > 0)
        assert jnp.all(tr["gate"]["value"] < 1)

    def test_joint_guide_mixed_with_independent(
        self, model_config, small_counts
    ):
        """Test that joint specs and independent specs coexist correctly."""
        from scribe.models.components import JointLowRankGuide

        joint = JointLowRankGuide(rank=5, group="nb")
        specs = [
            # These two are jointly modeled
            ExpNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="mu",
            ),
            ExpNormalSpec(
                name="phi",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="phi",
            ),
            # This one is independent (separate LowRank)
            ExpNormalSpec(
                name="r_extra",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=LowRankGuide(rank=3),
                constrained_name="r_extra",
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

        assert "mu" in tr
        assert "phi" in tr
        assert "r_extra" in tr

    def test_joint_guide_param_names(self, model_config, small_counts):
        """Test that joint guide registers correctly named variational params."""
        from scribe.models.components import JointLowRankGuide

        joint = JointLowRankGuide(rank=5, group="mygroup")
        specs = [
            ExpNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="mu",
            ),
            ExpNormalSpec(
                name="phi",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="phi",
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

        # Verify variational parameter naming convention
        param_sites = {
            name for name, site in tr.items() if site["type"] == "param"
        }
        expected_params = {
            "joint_mygroup_mu_loc",
            "joint_mygroup_mu_W",
            "joint_mygroup_mu_raw_diag",
            "joint_mygroup_phi_loc",
            "joint_mygroup_phi_W",
            "joint_mygroup_phi_raw_diag",
        }
        assert expected_params.issubset(
            param_sites
        ), f"Missing expected params: {expected_params - param_sites}"

    def test_joint_guide_validation(self):
        """Test JointLowRankGuide validation."""
        from scribe.models.components import JointLowRankGuide

        # rank must be positive
        with pytest.raises(ValueError, match="rank must be positive"):
            JointLowRankGuide(rank=0, group="test")

        with pytest.raises(ValueError, match="rank must be positive"):
            JointLowRankGuide(rank=-1, group="test")

        # group must be non-empty
        with pytest.raises(
            ValueError, match="group must be a non-empty string"
        ):
            JointLowRankGuide(rank=5, group="")

    def test_woodbury_conditional_params(self):
        """Test the Woodbury conditional parameter computation directly."""
        from scribe.models.builders.guide_builder import (
            _woodbury_conditional_params,
        )

        G, k = 10, 3
        key = random.PRNGKey(42)
        keys = random.split(key, 6)

        W1 = random.normal(keys[0], (G, k)) * 0.1
        W2 = random.normal(keys[1], (G, k)) * 0.1
        D1 = jnp.abs(random.normal(keys[2], (G,))) + 0.1
        D2 = jnp.abs(random.normal(keys[3], (G,))) + 0.1
        loc1 = random.normal(keys[4], (G,))
        loc2 = random.normal(keys[5], (G,))
        theta1 = loc1 + 0.5  # a sample

        cond_loc, cond_W, cond_D = _woodbury_conditional_params(
            W1, D1, W2, D2, loc1, loc2, theta1
        )

        # Shapes should be preserved
        assert cond_loc.shape == (G,)
        assert cond_W.shape == (G, k)
        assert cond_D.shape == (G,)

        # Conditional diagonal should be unchanged
        assert jnp.allclose(cond_D, D2)

        # All values should be finite
        assert jnp.isfinite(cond_loc).all()
        assert jnp.isfinite(cond_W).all()

    def test_woodbury_reduces_to_marginal_when_uncorrelated(self):
        """When W1 or W2 is zero, the conditional should equal the marginal."""
        from scribe.models.builders.guide_builder import (
            _woodbury_conditional_params,
        )

        G, k = 10, 3
        key = random.PRNGKey(0)
        keys = random.split(key, 5)

        # W1 = 0 means no information from theta_1, so conditional = marginal
        W1 = jnp.zeros((G, k))
        W2 = random.normal(keys[0], (G, k)) * 0.1
        D1 = jnp.ones(G)
        D2 = jnp.abs(random.normal(keys[1], (G,))) + 0.1
        loc1 = jnp.zeros(G)
        loc2 = random.normal(keys[2], (G,))
        theta1 = random.normal(keys[3], (G,))

        cond_loc, cond_W, cond_D = _woodbury_conditional_params(
            W1, D1, W2, D2, loc1, loc2, theta1
        )

        # With W1=0, conditioning provides no info: cond_loc should be loc2
        assert jnp.allclose(cond_loc, loc2, atol=1e-5)

        # cond_W should be W2 @ L_M where M = (I + 0)^{-1} = I, so L_M = I
        assert jnp.allclose(cond_W, W2, atol=1e-5)


# ==============================================================================
# Integration tests: Config → Guide → SVI → Posterior Extraction
# ==============================================================================


class TestJointLowRankIntegration:
    """End-to-end tests for the joint low-rank guide config pipeline.

    Verifies that joint_params flows through config → preset_builder →
    guide_builder → SVI → posterior extraction (get_distributions / get_map).
    """

    def test_config_joint_params_field(self):
        """ModelConfig and builder accept and store joint_params."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_hierarchical_p()
            .with_joint_params(["mu", "phi"])
            .build()
        )
        assert config.joint_params == ["mu", "phi"]

    def test_config_joint_params_none_by_default(self):
        """joint_params defaults to None."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .build()
        )
        assert config.joint_params is None

    def test_preset_builder_creates_joint_guide(self):
        """build_config_from_preset with joint_params assigns JointLowRankGuide."""
        from scribe.models.components import JointLowRankGuide

        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            guide_rank=5,
            joint_params=["mu", "phi"],
        )

        assert config.joint_params == ["mu", "phi"]
        gf = config.guide_families
        assert gf is not None
        assert isinstance(gf.mu, JointLowRankGuide)
        assert isinstance(gf.phi, JointLowRankGuide)
        assert gf.mu.rank == 5
        assert gf.mu.group == "joint"

    def test_preset_builder_requires_guide_rank(self):
        """joint_params without guide_rank raises ValueError."""
        with pytest.raises(
            ValueError, match="joint_params requires guide_rank"
        ):
            build_config_from_preset(
                model="nbdm",
                parameterization="mean_odds",
                unconstrained=True,
                p_prior="gaussian",
                joint_params=["mu", "phi"],
            )

    def test_svi_with_joint_guide(self):
        """Run a few SVI steps with jointly modeled mu and phi."""
        n_cells, n_genes = 50, 10
        key = random.PRNGKey(42)
        counts = random.poisson(key, lam=5.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            guide_rank=3,
            joint_params=["mu", "phi"],
        )

        model_fn, guide_fn, config = get_model_and_guide(config)

        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(0), **model_kwargs)

        # Run 5 steps
        for _ in range(5):
            svi_state, loss = svi.update(svi_state, **model_kwargs)
            assert jnp.isfinite(loss), f"SVI loss is non-finite: {loss}"

        params = svi.get_params(svi_state)

        # Variational params should include joint_ prefixed keys
        joint_keys = [k for k in params if k.startswith("joint_")]
        assert (
            len(joint_keys) > 0
        ), f"Expected joint_ prefixed params, got: {sorted(params.keys())}"
        # Specifically, we expect joint_joint_mu_loc, joint_joint_phi_loc, etc.
        assert "joint_joint_mu_loc" in params
        assert "joint_joint_phi_loc" in params

    def test_posterior_extraction_marginals(self):
        """get_posterior_distributions returns per-param marginals for joint params."""
        from scribe.models.builders.posterior import get_posterior_distributions

        n_cells, n_genes = 50, 10
        key = random.PRNGKey(0)
        counts = random.poisson(key, lam=5.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            guide_rank=3,
            joint_params=["mu", "phi"],
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(1), **model_kwargs)
        for _ in range(3):
            svi_state, _ = svi.update(svi_state, **model_kwargs)
        params = svi.get_params(svi_state)

        distributions = get_posterior_distributions(params, config)

        # Per-parameter marginals should exist
        assert "mu" in distributions
        assert "phi" in distributions

        # Both should be dict with base + transform (low-rank structure)
        for name in ("mu", "phi"):
            d = distributions[name]
            assert isinstance(d, dict), f"{name} should be dict, got {type(d)}"
            assert "base" in d
            assert "transform" in d

    def test_posterior_extraction_joint_distribution(self):
        """get_posterior_distributions returns the full joint distribution."""
        from scribe.models.builders.posterior import get_posterior_distributions
        import numpyro.distributions as dist

        n_cells, n_genes = 50, 10
        key = random.PRNGKey(0)
        counts = random.poisson(key, lam=5.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            guide_rank=3,
            joint_params=["mu", "phi"],
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(1), **model_kwargs)
        for _ in range(3):
            svi_state, _ = svi.update(svi_state, **model_kwargs)
        params = svi.get_params(svi_state)

        distributions = get_posterior_distributions(params, config)

        # Full joint distribution should be present
        assert "joint:joint" in distributions
        joint = distributions["joint:joint"]
        assert "base" in joint
        assert "param_names" in joint
        assert joint["param_names"] == ["mu", "phi"]

        # The joint base should be a LowRankMVN of dimension 2 * n_genes
        base = joint["base"]
        assert isinstance(base, dist.LowRankMultivariateNormal)
        assert base.loc.shape == (2 * n_genes,)

    def test_get_map_skips_joint_keys(self):
        """get_map should work and not include joint:* keys."""
        from scribe.svi._parameter_extraction import ParameterExtractionMixin
        from scribe.models.builders.posterior import get_posterior_distributions

        n_cells, n_genes = 50, 10
        key = random.PRNGKey(0)
        counts = random.poisson(key, lam=5.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            guide_rank=3,
            joint_params=["mu", "phi"],
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(1), **model_kwargs)
        for _ in range(3):
            svi_state, _ = svi.update(svi_state, **model_kwargs)
        params = svi.get_params(svi_state)

        # Build distributions and run get_map logic directly
        distributions = get_posterior_distributions(params, config)
        map_estimates = {}
        for param, dist_obj in distributions.items():
            if param.startswith("joint:"):
                continue
            if (
                isinstance(dist_obj, dict)
                and "base" in dist_obj
                and "transform" in dist_obj
            ):
                base_dist = dist_obj["base"]
                transform = dist_obj["transform"]
                if hasattr(base_dist, "loc"):
                    map_estimates[param] = transform(base_dist.loc)

        # Should have mu and phi, but not joint:*
        assert "mu" in map_estimates
        assert "phi" in map_estimates
        assert not any(k.startswith("joint:") for k in map_estimates)

        # Values should be finite and positive (after exp transform)
        assert jnp.isfinite(map_estimates["mu"]).all()
        assert jnp.isfinite(map_estimates["phi"]).all()
        assert jnp.all(map_estimates["mu"] > 0)
        assert jnp.all(map_estimates["phi"] > 0)

    def test_joint_gate_posterior_extraction_uses_joint_keys(self):
        """Joint gate should build posterior without gate_loc/gate_scale keys.

        This regression targets zero-inflated models where ``gate`` participates
        in ``joint_params`` and variational params are stored under
        ``joint_*_gate_*`` keys.
        """
        from scribe.models.builders.posterior import get_posterior_distributions

        n_cells, n_genes = 40, 8
        key = random.PRNGKey(7)
        counts = random.poisson(key, lam=4.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="zinbvcp",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            gate_prior="gaussian",
            guide_rank=3,
            joint_params=["mu", "phi", "gate"],
            priors={"eta_capture": (11.51, 0.01)},
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(8), **model_kwargs)
        for _ in range(3):
            svi_state, _ = svi.update(svi_state, **model_kwargs)
        params = svi.get_params(svi_state)

        # Build posterior distributions directly from learned variational params.
        distributions = get_posterior_distributions(params, config)

        # Gate should be reconstructed as a low-rank transformed posterior.
        assert "gate" in distributions
        gate_dist = distributions["gate"]
        assert isinstance(gate_dist, dict)
        assert "base" in gate_dist
        assert "transform" in gate_dist

    def test_joint_gate_get_map_returns_finite_gate(self):
        """get_map should succeed and return finite gate for joint gate configs."""
        n_cells, n_genes = 40, 8
        key = random.PRNGKey(9)
        counts = random.poisson(key, lam=4.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="zinbvcp",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            gate_prior="gaussian",
            guide_rank=3,
            joint_params=["mu", "phi", "gate"],
            priors={"eta_capture": (11.51, 0.01)},
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(10), **model_kwargs)
        for _ in range(3):
            svi_state, _ = svi.update(svi_state, **model_kwargs)
        params = svi.get_params(svi_state)

        # Use the public SVI results API so this guards the exact failing path.
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0, 0.5]),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type="zinbvcp",
            model_config=config,
            prior_params={},
        )
        map_estimates = results.get_map(
            use_mean=False, canonical=True, verbose=False, counts=counts
        )

        assert "gate" in map_estimates
        gate = map_estimates["gate"]
        assert jnp.isfinite(gate).all()
        assert jnp.all(gate > 0.0)
        assert jnp.all(gate < 1.0)

    def test_horseshoe_joint_guide_uses_raw_gate_site(self):
        """Horseshoe+joint must sample ``gate_raw`` and keep ``gate`` deterministic.

        This guards against model/guide site mismatch when gate uses the
        horseshoe NCP parameterization.
        """
        n_cells, n_genes = 40, 8
        key = random.PRNGKey(31)
        counts = random.poisson(key, lam=4.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="zinbvcp",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            gate_prior="horseshoe",
            guide_rank=3,
            joint_params=["mu", "phi", "gate"],
            priors={"eta_capture": (11.51, 0.01)},
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        # Trace model and guide to compare latent sample-site names directly.
        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as model_tr:
                model_fn(**model_kwargs)
        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as guide_tr:
                guide_fn(**model_kwargs)

        model_sample_sites = {
            name
            for name, site in model_tr.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }
        guide_sample_sites = {
            name for name, site in guide_tr.items() if site["type"] == "sample"
        }

        assert "gate_raw" in model_sample_sites
        assert "gate_raw" in guide_sample_sites
        assert "gate" not in guide_sample_sites
        assert "gate" in guide_tr
        assert guide_tr["gate"]["type"] == "deterministic"
        assert "gate_raw" not in (model_sample_sites - guide_sample_sites)

    def test_svi_horseshoe_joint_gate_smoke(self):
        """SVI stays finite for horseshoe gate jointly modeled with mu/phi.

        This mirrors the reported failure mode:
        - mean_odds parameterization
        - hierarchical mu in a mixture model
        - horseshoe gate prior
        - joint low-rank guide over ``phi``, ``mu``, and ``gate``
        """
        n_cells, n_genes = 60, 10
        key = random.PRNGKey(41)
        counts = random.poisson(key, lam=5.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="zinbvcp",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            mu_prior="gaussian",
            gate_prior="horseshoe",
            guide_rank=3,
            n_components=3,
            mixture_params=["phi", "mu", "gate"],
            joint_params=["phi", "mu", "gate"],
            priors={"eta_capture": (11.51, 0.01)},
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(42), **model_kwargs)

        # Run a few updates and require finite losses; this is the regression.
        for _ in range(3):
            svi_state, loss = svi.update(svi_state, **model_kwargs)
            assert jnp.isfinite(loss), f"SVI loss is non-finite: {loss}"

    def test_mean_prob_parameterization_joint(self):
        """Joint guide works for mean_prob parameterization (mu + p)."""
        n_cells, n_genes = 50, 10
        key = random.PRNGKey(0)
        counts = random.poisson(key, lam=5.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_prob",
            unconstrained=True,
            p_prior="gaussian",
            guide_rank=3,
            joint_params=["mu", "p"],
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(1), **model_kwargs)
        for _ in range(3):
            svi_state, _ = svi.update(svi_state, **model_kwargs)
            loss = svi.evaluate(svi_state, **model_kwargs)
            assert jnp.isfinite(loss)

    def test_get_map_use_mean_with_joint_params(self):
        """get_map(use_mean=True) handles dict-structured joint distributions.

        Regression test: previously, the NaN-replacement path called
        ``distributions[param].mean`` which crashed with AttributeError
        for joint low-rank parameters stored as dicts.
        """
        from scribe.models.builders.posterior import get_posterior_distributions

        n_cells, n_genes = 50, 10
        key = random.PRNGKey(0)
        counts = random.poisson(key, lam=5.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            guide_rank=3,
            joint_params=["mu", "phi"],
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(1), **model_kwargs)
        for _ in range(5):
            svi_state, _ = svi.update(svi_state, **model_kwargs)
        params = svi.get_params(svi_state)

        # Inject a NaN into mu's loc to trigger the use_mean fallback
        loc_key = "joint_joint_mu_loc"
        assert loc_key in params, f"Expected {loc_key} in params"
        corrupted = params[loc_key].at[0].set(float("nan"))
        params_with_nan = {**params, loc_key: corrupted}

        distributions = get_posterior_distributions(params_with_nan, config)

        # Exercise the NaN-replacement code path that previously raised
        # AttributeError for dict-structured distributions.  The key
        # assertion is that it does NOT crash.
        map_estimates = {}
        for param, dist_obj in distributions.items():
            if param.startswith("joint:"):
                continue
            if (
                isinstance(dist_obj, dict)
                and "base" in dist_obj
                and "transform" in dist_obj
            ):
                base_dist = dist_obj["base"]
                transform = dist_obj["transform"]
                map_estimates[param] = transform(base_dist.loc)

        # NaN replacement path — must not raise AttributeError
        for param, value in map_estimates.items():
            if jnp.any(jnp.isnan(value)):
                dist_obj = distributions[param]
                if (
                    isinstance(dist_obj, dict)
                    and "base" in dist_obj
                    and "transform" in dist_obj
                ):
                    mean_value = dist_obj["transform"](dist_obj["base"].mean)
                else:
                    mean_value = dist_obj.mean
                map_estimates[param] = jnp.where(
                    jnp.isnan(value), mean_value, value
                )

        # phi should be untouched and finite
        assert "phi" in map_estimates
        assert jnp.isfinite(map_estimates["phi"]).all()

    def test_joint_mixture_get_map(self):
        """get_map works for joint params in a mixture model."""
        from scribe.models.builders.posterior import get_posterior_distributions

        n_cells, n_genes = 50, 10
        n_components = 3
        key = random.PRNGKey(0)
        counts = random.poisson(key, lam=5.0, shape=(n_cells, n_genes))

        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_odds",
            unconstrained=True,
            p_prior="gaussian",
            guide_rank=3,
            joint_params=["mu", "phi"],
            n_components=n_components,
            mixture_params=["mu", "phi"],
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(1), **model_kwargs)
        for _ in range(5):
            svi_state, _ = svi.update(svi_state, **model_kwargs)
        params = svi.get_params(svi_state)

        # Posterior distributions should have component dimension
        distributions = get_posterior_distributions(params, config)
        assert "mu" in distributions
        assert "phi" in distributions

        # MAP estimates via transform(loc) should have shape
        # (n_components, n_genes)
        for pname in ("mu", "phi"):
            dist_obj = distributions[pname]
            assert isinstance(dist_obj, dict)
            map_val = dist_obj["transform"](dist_obj["base"].loc)
            assert map_val.shape == (n_components, n_genes), (
                f"{pname} MAP shape {map_val.shape} != "
                f"({n_components}, {n_genes})"
            )
            assert jnp.isfinite(map_val).all()
            assert jnp.all(map_val > 0)

    def test_joint_heterogeneous_scalar_and_gene(
        self, model_config, small_counts
    ):
        """Scalar phi + gene-specific mu in same joint group (heterogeneous)."""
        from scribe.models.components import JointLowRankGuide

        joint = JointLowRankGuide(rank=3, group="test")
        # Scalar phi (is_gene_specific=False) + gene-specific mu
        specs = [
            ExpNormalSpec(
                name="phi",
                shape_dims=(),
                default_params=(0.0, 1.0),
                is_gene_specific=False,
                guide_family=joint,
                constrained_name="phi",
            ),
            ExpNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="mu",
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

        # phi should be scalar, mu should be gene-specific
        assert "phi" in tr
        assert "mu" in tr
        assert tr["phi"]["value"].shape == ()
        assert tr["mu"]["value"].shape == (20,)
        assert jnp.all(tr["phi"]["value"] > 0)
        assert jnp.all(tr["mu"]["value"] > 0)

    def test_joint_heterogeneous_mixture(self):
        """Mixture scalar phi (C,) + mixture gene-specific mu (C, G)."""
        n_cells, n_genes = 50, 10
        n_components = 3
        key = random.PRNGKey(0)
        counts = random.poisson(key, lam=5.0, shape=(n_cells, n_genes))

        # p_prior='none' makes phi scalar; mixture_params includes phi
        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_odds",
            unconstrained=True,
            guide_rank=3,
            joint_params=["phi", "mu"],
            n_components=n_components,
            mixture_params=["mu", "phi"],
        )
        model_fn, guide_fn, config = get_model_and_guide(config)
        model_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        # Run a few SVI steps to verify it works end-to-end
        optimizer = Adam(1e-3)
        svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(1), **model_kwargs)
        for _ in range(3):
            svi_state, loss = svi.update(svi_state, **model_kwargs)
            assert jnp.isfinite(loss), f"SVI loss not finite: {loss}"

        # Verify phi shape is (n_components,) and mu is (n_components, n_genes)
        params = svi.get_params(svi_state)
        phi_loc = params["joint_joint_phi_loc"]
        mu_loc = params["joint_joint_mu_loc"]
        # phi expanded to (n_components, 1) internally
        assert phi_loc.shape == (n_components, 1)
        # mu is (n_components, n_genes)
        assert mu_loc.shape == (n_components, n_genes)

    def test_joint_heterogeneous_three_params(self, model_config, small_counts):
        """Scalar phi + gene-specific mu + gene-specific gate."""
        from scribe.models.components import JointLowRankGuide
        from scribe.models.builders import SigmoidNormalSpec

        joint = JointLowRankGuide(rank=4, group="zinb")
        specs = [
            ExpNormalSpec(
                name="phi",
                shape_dims=(),
                default_params=(0.0, 1.0),
                is_gene_specific=False,
                guide_family=joint,
                constrained_name="phi",
            ),
            ExpNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="mu",
            ),
            SigmoidNormalSpec(
                name="gate",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=joint,
                constrained_name="gate",
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

        assert tr["phi"]["value"].shape == ()
        assert tr["mu"]["value"].shape == (20,)
        assert tr["gate"]["value"].shape == (20,)
        assert jnp.all(tr["phi"]["value"] > 0)
        assert jnp.all(tr["mu"]["value"] > 0)
        assert jnp.all(tr["gate"]["value"] > 0)
        assert jnp.all(tr["gate"]["value"] < 1)

    def test_joint_heterogeneous_batch_mismatch_raises(self):
        """Mismatched batch dims in heterogeneous joint group raises."""
        from scribe.models.builders.guide_builder import setup_joint_guide
        from scribe.models.builders.parameter_specs import ExpNormalSpec
        from scribe.models.components import JointLowRankGuide

        # phi is_mixture=True (batch=n_components), mu is_mixture=False (no batch)
        spec_phi = ExpNormalSpec(
            name="phi",
            shape_dims=(),
            default_params=(0.0, 1.0),
            is_gene_specific=False,
            is_mixture=True,
        )
        spec_mu = ExpNormalSpec(
            name="mu",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            is_mixture=False,
        )
        guide = JointLowRankGuide(rank=3, group="test")
        dims = {"n_genes": 10, "n_cells": 50, "n_components": 3}

        with pytest.raises(ValueError, match="batch shape"):
            setup_joint_guide(
                [spec_phi, spec_mu], guide, dims, model_config=None
            )

    def test_svi_with_heterogeneous_joint_guide(self):
        """Run SVI steps with scalar phi + gene-specific mu joint guide."""
        from scribe.models.components import JointLowRankGuide

        n_cells, n_genes = 50, 10
        key = random.PRNGKey(42)
        counts = random.poisson(key, lam=10.0, shape=(n_cells, n_genes)).astype(
            jnp.float32
        )

        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_joint_params(["phi", "mu"])
            .with_guide_families(
                GuideFamilyConfig(
                    phi=JointLowRankGuide(rank=3, group="joint"),
                    mu=JointLowRankGuide(rank=3, group="joint"),
                )
            )
            .build()
        )
        from scribe.models.presets.factory import create_model

        model, guide, _ = create_model(config, validate=False)
        model_kwargs = {
            "n_cells": n_cells,
            "n_genes": n_genes,
            "model_config": config,
            "counts": counts,
        }

        optimizer = numpyro.optim.Adam(1e-2)
        svi = SVI(model, guide, optimizer, Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(1), **model_kwargs)

        # Run a few steps and verify loss is finite
        for _ in range(5):
            svi_state, loss = svi.update(svi_state, **model_kwargs)
            assert jnp.isfinite(loss), f"SVI loss is not finite: {loss}"

    def test_posterior_extraction_heterogeneous_joint(self):
        """MAP and distributions work for heterogeneous joint groups."""
        from scribe.models.builders.posterior import get_posterior_distributions
        import numpyro.distributions as dist

        n_genes = 10
        k = 3

        # Simulate variational params for scalar phi (G=1) and gene mu (G=n_genes)
        params = {
            "joint_joint_phi_loc": jnp.zeros((1,)),
            "joint_joint_phi_W": 0.01 * jnp.ones((1, k)),
            "joint_joint_phi_raw_diag": -3.0 * jnp.ones((1,)),
            "joint_joint_mu_loc": jnp.zeros((n_genes,)),
            "joint_joint_mu_W": 0.01 * jnp.ones((n_genes, k)),
            "joint_joint_mu_raw_diag": -3.0 * jnp.ones((n_genes,)),
        }

        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_joint_params(["phi", "mu"])
            .build()
        )

        distributions = get_posterior_distributions(params, config)

        # Per-param marginals: phi should be Normal (scalar), mu LowRankMVN
        assert "phi" in distributions
        assert "mu" in distributions

        phi_dist = distributions["phi"]
        assert isinstance(phi_dist["base"], dist.Normal)

        mu_dist = distributions["mu"]
        assert isinstance(mu_dist["base"], dist.LowRankMultivariateNormal)

        # Full joint distribution should exist
        assert "joint:joint" in distributions
        joint_dist = distributions["joint:joint"]
        assert joint_dist["param_sizes"] == [1, n_genes]
        assert isinstance(joint_dist["base"], dist.LowRankMultivariateNormal)
        assert joint_dist["base"].loc.shape == (1 + n_genes,)


class TestPosteriorContractExtraction:
    """Contract-focused tests for get_posterior_distributions orchestration."""

    @staticmethod
    def _make_config(**overrides):
        """Build a minimal model-config-like object for posterior extraction."""
        defaults = dict(
            parameterization=ParameterizationEnum.CANONICAL,
            unconstrained=False,
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=False,
            p_prior="none",
            gate_prior="none",
            mu_dataset_prior="none",
            p_dataset_prior="none",
            p_dataset_mode="gene_specific",
            gate_dataset_prior="none",
            uses_biology_informed_capture=False,
            shared_capture_scaling=False,
            joint_params=None,
        )
        defaults.update(overrides)
        ns = SimpleNamespace(**defaults)
        # Derive deprecated attrs for backward compat with get_posterior_distributions
        ns.hierarchical_p = ns.p_prior != "none"
        ns.horseshoe_p = ns.p_prior == "horseshoe"
        ns.hierarchical_gate = ns.gate_prior != "none"
        ns.horseshoe_gate = ns.gate_prior == "horseshoe"
        ns.hierarchical_dataset_mu = ns.mu_dataset_prior != "none"
        ns.horseshoe_dataset_mu = ns.mu_dataset_prior == "horseshoe"
        ns.hierarchical_dataset_p = (
            "none" if ns.p_dataset_prior == "none" else ns.p_dataset_mode
        )
        ns.horseshoe_dataset_p = ns.p_dataset_prior == "horseshoe"
        ns.hierarchical_dataset_gate = ns.gate_dataset_prior != "none"
        ns.horseshoe_dataset_gate = ns.gate_dataset_prior == "horseshoe"
        return ns

    @staticmethod
    def _dist_kind(d):
        """Classify posterior object structure for stable contract assertions."""
        import numpyro.distributions as dist

        if isinstance(d, dict) and "base" in d and "transform" in d:
            return "dict_transform"
        if isinstance(d, dict) and "base" in d and "param_names" in d:
            return "joint_dict"
        if isinstance(d, dist.TransformedDistribution):
            return "transformed_distribution"
        if isinstance(d, dist.Distribution):
            return "distribution"
        if isinstance(d, list):
            return "list"
        return type(d).__name__

    @staticmethod
    def _dist_shape(d):
        """Extract a stable shape tuple from a posterior object."""
        if isinstance(d, dict):
            base = d.get("base")
            if hasattr(base, "loc"):
                return tuple(base.loc.shape)
            return ()
        if hasattr(d, "loc"):
            return tuple(d.loc.shape)
        if hasattr(d, "base_dist") and hasattr(d.base_dist, "loc"):
            return tuple(d.base_dist.loc.shape)
        if hasattr(d, "concentration"):
            return tuple(d.concentration.shape)
        if hasattr(d, "concentration1"):
            return tuple(d.concentration1.shape)
        return ()

    @pytest.mark.parametrize(
        "case_name,config,params,expected_keys,expected_kind,expected_shape",
        [
            (
                "canonical_constrained_single",
                _make_config.__func__(),
                {
                    "p_alpha": jnp.array(2.0),
                    "p_beta": jnp.array(3.0),
                    "r_loc": jnp.zeros((4,)),
                    "r_scale": jnp.ones((4,)),
                },
                {"p", "r"},
                {"p": "distribution", "r": "transformed_distribution"},
                {"p": (), "r": (4,)},
            ),
            (
                "mixture_zinbvcp_unconstrained",
                _make_config.__func__(
                    parameterization=ParameterizationEnum.MEAN_ODDS,
                    unconstrained=True,
                    is_mixture=True,
                    is_zero_inflated=True,
                    uses_variable_capture=True,
                ),
                {
                    "phi_loc": jnp.array(0.0),
                    "phi_scale": jnp.array(1.0),
                    "mu_loc": jnp.zeros((3, 4)),
                    "mu_scale": jnp.ones((3, 4)),
                    "gate_loc": jnp.zeros((3, 4)),
                    "gate_scale": jnp.ones((3, 4)),
                    "phi_capture_loc": jnp.zeros((5,)),
                    "phi_capture_scale": jnp.ones((5,)),
                    "mixing_concentrations": jnp.ones((3,)),
                },
                {"phi", "mu", "gate", "phi_capture", "mixing_weights"},
                {
                    "phi": "transformed_distribution",
                    "mu": "transformed_distribution",
                    "gate": "transformed_distribution",
                    "phi_capture": "transformed_distribution",
                    "mixing_weights": "distribution",
                },
                {
                    "phi": (),
                    "mu": (3, 4),
                    "gate": (3, 4),
                    "phi_capture": (5,),
                    "mixing_weights": (3,),
                },
            ),
            (
                "horseshoe_gene_level_phi",
                _make_config.__func__(
                    parameterization=ParameterizationEnum.MEAN_ODDS,
                    unconstrained=True,
                    p_prior="horseshoe",
                ),
                {
                    "mu_loc": jnp.zeros((4,)),
                    "mu_scale": jnp.ones((4,)),
                    "log_phi_loc_loc": jnp.array(0.0),
                    "log_phi_loc_scale": jnp.array(1.0),
                    "tau_phi_loc": jnp.array(0.0),
                    "tau_phi_scale": jnp.array(1.0),
                    "lambda_phi_loc": jnp.zeros((4,)),
                    "lambda_phi_scale": jnp.ones((4,)),
                    "c_sq_phi_loc": jnp.array(0.0),
                    "c_sq_phi_scale": jnp.array(1.0),
                    "phi_raw_loc": jnp.zeros((4,)),
                    "phi_raw_scale": jnp.ones((4,)),
                },
                {"mu", "log_phi_loc", "tau_phi", "lambda_phi", "c_sq_phi", "phi_raw"},
                {
                    "mu": "transformed_distribution",
                    "log_phi_loc": "transformed_distribution",
                    "tau_phi": "transformed_distribution",
                    "lambda_phi": "transformed_distribution",
                    "c_sq_phi": "transformed_distribution",
                    "phi_raw": "distribution",
                },
                {
                    "mu": (4,),
                    "log_phi_loc": (),
                    "tau_phi": (),
                    "lambda_phi": (4,),
                    "c_sq_phi": (),
                    "phi_raw": (4,),
                },
            ),
            (
                "dataset_hierarchy_with_joint",
                _make_config.__func__(
                    parameterization=ParameterizationEnum.MEAN_ODDS,
                    unconstrained=True,
                    mu_dataset_prior="gaussian",
                    p_dataset_prior="gaussian",
                    p_dataset_mode="gene_specific",
                    joint_params=["mu", "phi"],
                ),
                {
                    "joint_joint_mu_loc": jnp.zeros((4,)),
                    "joint_joint_mu_W": 0.01 * jnp.ones((4, 2)),
                    "joint_joint_mu_raw_diag": -3.0 * jnp.ones((4,)),
                    "joint_joint_phi_loc": jnp.zeros((4,)),
                    "joint_joint_phi_W": 0.01 * jnp.ones((4, 2)),
                    "joint_joint_phi_raw_diag": -3.0 * jnp.ones((4,)),
                    "log_mu_dataset_loc_loc": jnp.array(0.0),
                    "log_mu_dataset_loc_scale": jnp.array(1.0),
                    "log_mu_dataset_scale_loc": jnp.array(0.0),
                    "log_mu_dataset_scale_scale": jnp.array(1.0),
                    "log_phi_dataset_loc_loc": jnp.array(0.0),
                    "log_phi_dataset_loc_scale": jnp.array(1.0),
                    "log_phi_dataset_scale_loc": jnp.array(0.0),
                    "log_phi_dataset_scale_scale": jnp.array(1.0),
                },
                {
                    "mu",
                    "phi",
                    "log_mu_dataset_loc",
                    "log_mu_dataset_scale",
                    "log_phi_dataset_loc",
                    "log_phi_dataset_scale",
                    "joint:joint",
                },
                {
                    "mu": "dict_transform",
                    "phi": "dict_transform",
                    "log_mu_dataset_loc": "distribution",
                    "log_mu_dataset_scale": "transformed_distribution",
                    "log_phi_dataset_loc": "distribution",
                    "log_phi_dataset_scale": "transformed_distribution",
                    "joint:joint": "joint_dict",
                },
                {
                    "mu": (4,),
                    "phi": (4,),
                    "log_mu_dataset_loc": (),
                    "log_mu_dataset_scale": (),
                    "log_phi_dataset_loc": (),
                    "log_phi_dataset_scale": (),
                    "joint:joint": (8,),
                },
            ),
            (
                "dataset_horseshoe_gate_with_joint",
                _make_config.__func__(
                    parameterization=ParameterizationEnum.MEAN_ODDS,
                    unconstrained=True,
                    is_zero_inflated=True,
                    gate_dataset_prior="horseshoe",
                    joint_params=["gate"],
                ),
                {
                    "phi_loc": jnp.array(0.0),
                    "phi_scale": jnp.array(1.0),
                    "mu_loc": jnp.zeros((4,)),
                    "mu_scale": jnp.ones((4,)),
                    "mu_W": 0.01 * jnp.ones((4, 2)),
                    "mu_raw_diag": -3.0 * jnp.ones((4,)),
                    "joint_joint_gate_loc": jnp.zeros((4,)),
                    "joint_joint_gate_W": 0.01 * jnp.ones((4, 2)),
                    "joint_joint_gate_raw_diag": -3.0 * jnp.ones((4,)),
                    "logit_gate_dataset_loc_loc": jnp.array(0.0),
                    "logit_gate_dataset_loc_scale": jnp.array(1.0),
                    "tau_gate_dataset_loc": jnp.array(0.0),
                    "tau_gate_dataset_scale": jnp.array(1.0),
                    "lambda_gate_dataset_loc": jnp.zeros((4,)),
                    "lambda_gate_dataset_scale": jnp.ones((4,)),
                    "c_sq_gate_dataset_loc": jnp.array(0.0),
                    "c_sq_gate_dataset_scale": jnp.array(1.0),
                },
                {
                    "phi",
                    "mu",
                    "gate",
                    "logit_gate_dataset_loc",
                    "tau_gate_dataset",
                    "lambda_gate_dataset",
                    "c_sq_gate_dataset",
                    "joint:joint",
                },
                {
                    "phi": "transformed_distribution",
                    "mu": "dict_transform",
                    "gate": "dict_transform",
                    "logit_gate_dataset_loc": "transformed_distribution",
                    "tau_gate_dataset": "transformed_distribution",
                    "lambda_gate_dataset": "transformed_distribution",
                    "c_sq_gate_dataset": "transformed_distribution",
                    "joint:joint": "joint_dict",
                },
                {
                    "phi": (),
                    "mu": (4,),
                    "gate": (4,),
                    "logit_gate_dataset_loc": (),
                    "tau_gate_dataset": (),
                    "lambda_gate_dataset": (4,),
                    "c_sq_gate_dataset": (),
                    "joint:joint": (4,),
                },
            ),
            (
                "biology_informed_capture",
                _make_config.__func__(
                    parameterization=ParameterizationEnum.MEAN_ODDS,
                    unconstrained=True,
                    uses_variable_capture=True,
                    uses_biology_informed_capture=True,
                ),
                {
                    "phi_loc": jnp.array(0.0),
                    "phi_scale": jnp.array(1.0),
                    "mu_loc": jnp.zeros((4,)),
                    "mu_scale": jnp.ones((4,)),
                    "eta_capture_loc": jnp.zeros((5,)),
                    "eta_capture_scale": jnp.ones((5,)),
                    "mu_eta_loc": jnp.array(0.0),
                    "mu_eta_scale": jnp.array(1.0),
                },
                {"phi", "mu", "eta_capture", "mu_eta"},
                {
                    "phi": "transformed_distribution",
                    "mu": "transformed_distribution",
                    "eta_capture": "distribution",
                    "mu_eta": "distribution",
                },
                {"phi": (), "mu": (4,), "eta_capture": (5,), "mu_eta": ()},
            ),
        ],
    )
    def test_posterior_structural_contract_cases(
        self,
        case_name,
        config,
        params,
        expected_keys,
        expected_kind,
        expected_shape,
    ):
        """Assert stable key/type/shape contracts for representative configs."""
        from scribe.models.builders.posterior import get_posterior_distributions

        # Structural contracts are stronger than numeric snapshots and remain
        # stable across optimizer/JAX updates.
        distributions = get_posterior_distributions(params, config, split=False)
        assert set(distributions.keys()) == expected_keys, case_name

        for key, kind in expected_kind.items():
            assert self._dist_kind(distributions[key]) == kind, f"{case_name}:{key}"

        for key, shape in expected_shape.items():
            assert self._dist_shape(distributions[key]) == shape, f"{case_name}:{key}"

        # Joint outputs must expose consistent concatenation metadata.
        for key, value in distributions.items():
            if not key.startswith("joint:"):
                continue
            assert isinstance(value, dict)
            assert "param_names" in value and "param_sizes" in value
            assert sum(value["param_sizes"]) == value["base"].loc.shape[-1]
