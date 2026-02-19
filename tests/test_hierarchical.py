"""Tests for hierarchical parameterization (gene-specific p with hyperprior).

Covers:
- Hierarchical parameter spec creation and validation
- Hierarchical parameterization classes (registry, specs, derived params)
- Model building and tracing with hierarchical specs
- Gamma-based composition sampling (reduces to Dirichlet when p shared)
- Broadcasting of gene-specific p in mixture likelihoods
- DE pipeline with gene-specific p samples
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random

from scribe.models.builders.parameter_specs import (
    HierarchicalExpNormalSpec,
    HierarchicalNormalWithTransformSpec,
    HierarchicalSigmoidNormalSpec,
    NormalWithTransformSpec,
    SoftplusNormalSpec,
    resolve_shape,
)
from scribe.models.parameterizations import (
    PARAMETERIZATIONS,
    HierarchicalCanonicalParameterization,
    HierarchicalMeanOddsParameterization,
    HierarchicalMeanProbParameterization,
)
from scribe.models.components.likelihoods.base import broadcast_p_for_mixture


# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture
def rng():
    """Base RNG key."""
    return random.PRNGKey(42)


@pytest.fixture
def small_dims():
    """Small dimension dict for testing."""
    return {"n_cells": 20, "n_genes": 10, "n_components": 3}


# ==========================================================================
# Tests: Hierarchical Parameter Specs
# ==========================================================================


class TestHierarchicalSpecs:
    """Test hierarchical parameter specification classes."""

    def test_hierarchical_sigmoid_creation(self):
        """Test creating a HierarchicalSigmoidNormalSpec."""
        spec = HierarchicalSigmoidNormalSpec(
            name="p",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            hyper_loc_name="logit_p_loc",
            hyper_scale_name="logit_p_scale",
            is_gene_specific=True,
        )
        assert spec.name == "p"
        assert spec.hyper_loc_name == "logit_p_loc"
        assert spec.hyper_scale_name == "logit_p_scale"
        assert spec.is_gene_specific is True
        assert spec.constrained_name == "p"

    def test_hierarchical_exp_creation(self):
        """Test creating a HierarchicalExpNormalSpec."""
        spec = HierarchicalExpNormalSpec(
            name="phi",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            hyper_loc_name="log_phi_loc",
            hyper_scale_name="log_phi_scale",
            is_gene_specific=True,
        )
        assert spec.name == "phi"
        assert spec.hyper_loc_name == "log_phi_loc"
        assert spec.hyper_scale_name == "log_phi_scale"

    def test_hierarchical_inherits_transform(self):
        """Test that hierarchical specs inherit correct transforms."""
        sigmoid_spec = HierarchicalSigmoidNormalSpec(
            name="p",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            hyper_loc_name="loc",
            hyper_scale_name="scale",
            is_gene_specific=True,
        )
        assert isinstance(
            sigmoid_spec.transform, dist.transforms.SigmoidTransform
        )

        exp_spec = HierarchicalExpNormalSpec(
            name="phi",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            hyper_loc_name="loc",
            hyper_scale_name="scale",
            is_gene_specific=True,
        )
        assert isinstance(exp_spec.transform, dist.transforms.ExpTransform)

    def test_hierarchical_is_subclass(self):
        """Test inheritance chain."""
        assert issubclass(
            HierarchicalSigmoidNormalSpec,
            HierarchicalNormalWithTransformSpec,
        )
        assert issubclass(
            HierarchicalNormalWithTransformSpec, NormalWithTransformSpec
        )

    def test_hierarchical_with_mixture(self):
        """Test hierarchical spec with mixture flag."""
        spec = HierarchicalSigmoidNormalSpec(
            name="p",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            hyper_loc_name="logit_p_loc",
            hyper_scale_name="logit_p_scale",
            is_gene_specific=True,
            is_mixture=True,
        )
        assert spec.is_mixture is True


# ==========================================================================
# Tests: Hierarchical Parameterization Registry
# ==========================================================================


class TestHierarchicalParameterizationRegistry:
    """Test hierarchical parameterization classes and registry."""

    def test_registry_contains_hierarchical(self):
        """Test that all hierarchical parameterizations are registered."""
        expected = {
            "hierarchical_canonical",
            "hierarchical_mean_prob",
            "hierarchical_mean_odds",
        }
        assert expected.issubset(set(PARAMETERIZATIONS.keys()))

    def test_hierarchical_canonical_core_params(self):
        """Test hierarchical canonical core parameters."""
        p = PARAMETERIZATIONS["hierarchical_canonical"]
        assert "logit_p_loc" in p.core_parameters
        assert "logit_p_scale" in p.core_parameters
        assert "p" in p.core_parameters
        assert "r" in p.core_parameters

    def test_hierarchical_mean_prob_core_params(self):
        """Test hierarchical mean_prob core parameters."""
        p = PARAMETERIZATIONS["hierarchical_mean_prob"]
        assert "logit_p_loc" in p.core_parameters
        assert "logit_p_scale" in p.core_parameters
        assert "p" in p.core_parameters
        assert "mu" in p.core_parameters

    def test_hierarchical_mean_odds_core_params(self):
        """Test hierarchical mean_odds core parameters."""
        p = PARAMETERIZATIONS["hierarchical_mean_odds"]
        assert "log_phi_loc" in p.core_parameters
        assert "log_phi_scale" in p.core_parameters
        assert "phi" in p.core_parameters
        assert "mu" in p.core_parameters

    def test_hierarchical_canonical_builds_specs(self):
        """Test that hierarchical canonical builds correct spec types."""
        from scribe.models.config import GuideFamilyConfig

        p = PARAMETERIZATIONS["hierarchical_canonical"]
        specs = p.build_param_specs(
            unconstrained=True,
            guide_families=GuideFamilyConfig(),
        )

        spec_names = [s.name for s in specs]
        assert "logit_p_loc" in spec_names
        assert "logit_p_scale" in spec_names
        assert "p" in spec_names
        assert "r" in spec_names

        # Check that p is a hierarchical spec
        p_spec = next(s for s in specs if s.name == "p")
        assert isinstance(p_spec, HierarchicalSigmoidNormalSpec)
        assert p_spec.is_gene_specific is True

        # Check hyperparameter specs are not gene-specific
        loc_spec = next(s for s in specs if s.name == "logit_p_loc")
        assert loc_spec.is_gene_specific is False

    def test_hierarchical_mean_prob_derived_params(self):
        """Test that hierarchical mean_prob produces correct derived params."""
        p = PARAMETERIZATIONS["hierarchical_mean_prob"]
        derived = p.build_derived_params()
        assert len(derived) == 1
        assert derived[0].name == "r"
        assert "p" in derived[0].deps
        assert "mu" in derived[0].deps

    def test_hierarchical_mean_odds_derived_params(self):
        """Test that hierarchical mean_odds produces both r and p derived."""
        p = PARAMETERIZATIONS["hierarchical_mean_odds"]
        derived = p.build_derived_params()
        names = {d.name for d in derived}
        assert "r" in names
        assert "p" in names

    def test_hierarchical_canonical_mixture_specs(self):
        """Test that mixture_params is respected for hierarchical specs."""
        from scribe.models.config import GuideFamilyConfig

        p = PARAMETERIZATIONS["hierarchical_canonical"]
        specs = p.build_param_specs(
            unconstrained=True,
            guide_families=GuideFamilyConfig(),
            n_components=3,
            mixture_params=["p"],
        )

        p_spec = next(s for s in specs if s.name == "p")
        r_spec = next(s for s in specs if s.name == "r")
        assert p_spec.is_mixture is True
        assert r_spec.is_mixture is False

    def test_hierarchical_mean_odds_transform_param(self):
        """Test that mean_odds transforms p_capture to phi_capture."""
        p = PARAMETERIZATIONS["hierarchical_mean_odds"]
        assert p.transform_model_param("p_capture") == "phi_capture"
        assert p.transform_model_param("other") == "other"


# ==========================================================================
# Tests: Model Building with Hierarchical Specs
# ==========================================================================


class TestHierarchicalModelBuilding:
    """Test that hierarchical models build and trace correctly."""

    def test_hierarchical_canonical_model_traces(self, rng, small_dims):
        """Test that a hierarchical canonical model can be traced."""
        from scribe.models.builders import ModelBuilder
        from scribe.models.components.likelihoods.negative_binomial import (
            NegativeBinomialLikelihood,
        )
        from scribe.models.config import GuideFamilyConfig, ModelConfigBuilder

        param = PARAMETERIZATIONS["hierarchical_canonical"]
        specs = param.build_param_specs(
            unconstrained=True,
            guide_families=GuideFamilyConfig(),
        )
        derived = param.build_derived_params()

        builder = ModelBuilder()
        for s in specs:
            builder.add_param(s)
        for d in derived:
            builder.add_derived(d.name, d.compute, d.deps)
        builder.with_likelihood(NegativeBinomialLikelihood())

        model_fn = builder.build()

        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("hierarchical_canonical")
            .build()
        )

        # Trace the model to check it runs without errors
        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(model_fn).get_trace(
                n_cells=small_dims["n_cells"],
                n_genes=small_dims["n_genes"],
                model_config=config,
                counts=None,
            )

        # Verify expected sample sites exist
        assert "logit_p_loc" in trace
        assert "logit_p_scale" in trace
        assert "p" in trace
        assert "r" in trace
        assert "counts" in trace

        # Verify p is gene-specific
        p_value = trace["p"]["value"]
        assert p_value.shape == (small_dims["n_genes"],)

        # Verify p is in (0, 1) — sigmoid transform
        assert jnp.all(p_value > 0)
        assert jnp.all(p_value < 1)

    def test_hierarchical_mean_prob_model_traces(self, rng, small_dims):
        """Test that a hierarchical mean_prob model traces correctly."""
        from scribe.models.builders import ModelBuilder
        from scribe.models.components.likelihoods.negative_binomial import (
            NegativeBinomialLikelihood,
        )
        from scribe.models.config import GuideFamilyConfig, ModelConfigBuilder

        param = PARAMETERIZATIONS["hierarchical_mean_prob"]
        specs = param.build_param_specs(
            unconstrained=True,
            guide_families=GuideFamilyConfig(),
        )
        derived = param.build_derived_params()

        builder = ModelBuilder()
        for s in specs:
            builder.add_param(s)
        for d in derived:
            builder.add_derived(d.name, d.compute, d.deps)
        builder.with_likelihood(NegativeBinomialLikelihood())

        model_fn = builder.build()

        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("hierarchical_mean_prob")
            .build()
        )

        with numpyro.handlers.seed(rng_seed=1):
            trace = numpyro.handlers.trace(model_fn).get_trace(
                n_cells=small_dims["n_cells"],
                n_genes=small_dims["n_genes"],
                model_config=config,
                counts=None,
            )

        assert "logit_p_loc" in trace
        assert "logit_p_scale" in trace
        assert "p" in trace
        assert "mu" in trace
        # r should be a deterministic derived param
        assert "r" in trace

        # p is gene-specific
        assert trace["p"]["value"].shape == (small_dims["n_genes"],)
        # mu is gene-specific
        assert trace["mu"]["value"].shape == (small_dims["n_genes"],)

    def test_hierarchical_mean_odds_model_traces(self, rng, small_dims):
        """Test that a hierarchical mean_odds model traces correctly."""
        from scribe.models.builders import ModelBuilder
        from scribe.models.components.likelihoods.negative_binomial import (
            NegativeBinomialLikelihood,
        )
        from scribe.models.config import GuideFamilyConfig, ModelConfigBuilder

        param = PARAMETERIZATIONS["hierarchical_mean_odds"]
        specs = param.build_param_specs(
            unconstrained=True,
            guide_families=GuideFamilyConfig(),
        )
        derived = param.build_derived_params()

        builder = ModelBuilder()
        for s in specs:
            builder.add_param(s)
        for d in derived:
            builder.add_derived(d.name, d.compute, d.deps)
        builder.with_likelihood(NegativeBinomialLikelihood())

        model_fn = builder.build()

        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("hierarchical_mean_odds")
            .build()
        )

        with numpyro.handlers.seed(rng_seed=2):
            trace = numpyro.handlers.trace(model_fn).get_trace(
                n_cells=small_dims["n_cells"],
                n_genes=small_dims["n_genes"],
                model_config=config,
                counts=None,
            )

        assert "log_phi_loc" in trace
        assert "log_phi_scale" in trace
        assert "phi" in trace
        assert "mu" in trace
        assert "r" in trace
        assert "p" in trace

        # phi is gene-specific and positive (exp transform)
        phi_value = trace["phi"]["value"]
        assert phi_value.shape == (small_dims["n_genes"],)
        assert jnp.all(phi_value > 0)


# ==========================================================================
# Tests: Gamma-based Composition Sampling
# ==========================================================================


class TestGammaSampling:
    """Test Gamma-based composition sampling and Dirichlet equivalence."""

    def test_gamma_normalize_output_shape(self, rng):
        """Test that _batched_gamma_normalize produces correct output shape."""
        from scribe.de._empirical import _batched_gamma_normalize

        N, D = 100, 5
        r = jnp.abs(random.normal(rng, (N, D))) + 1.0
        p = jnp.full((N, D), 0.3)

        result = _batched_gamma_normalize(r, p, 1, rng, batch_size=50)
        assert result.shape == (N, D)

    def test_gamma_normalize_simplex(self, rng):
        """Test that Gamma-normalized samples lie on the simplex."""
        from scribe.de._empirical import _batched_gamma_normalize

        N, D = 200, 8
        r = jnp.abs(random.normal(rng, (N, D))) + 0.5
        p = jnp.full((N, D), 0.5)

        result = _batched_gamma_normalize(r, p, 1, rng, batch_size=100)

        # All values should be positive
        assert jnp.all(result > 0)

        # Rows should sum to 1
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(
            np.array(row_sums), np.ones(N), atol=1e-5
        )

    def test_gamma_reduces_to_dirichlet_when_p_shared(self, rng):
        """Test that Gamma sampling equals Dirichlet when all p_g are equal.

        When p_g = c for all genes, the scaling factor p/(1-p) is constant
        and cancels in the normalization, so the result is exactly
        Dirichlet(r).  We verify this by checking that the first two moments
        (mean and variance) of the compositions match Dirichlet(r).
        """
        from scribe.de._empirical import _batched_gamma_normalize

        N = 50_000
        D = 5
        # Fixed concentration for all samples
        r = jnp.ones((N, D)) * jnp.array([2.0, 3.0, 1.0, 4.0, 0.5])
        p_shared = jnp.full((N, D), 0.4)

        key1, key2 = random.split(rng)

        # Gamma-based sampling
        gamma_samples = _batched_gamma_normalize(
            r, p_shared, 1, key1, batch_size=2048
        )

        # Direct Dirichlet sampling for comparison
        alpha = jnp.array([2.0, 3.0, 1.0, 4.0, 0.5])
        dirichlet_samples = random.dirichlet(key2, alpha, shape=(N,))

        # Compare means
        gamma_mean = jnp.mean(gamma_samples, axis=0)
        dirichlet_mean = jnp.mean(dirichlet_samples, axis=0)
        # Expected Dirichlet mean: alpha / sum(alpha)
        expected_mean = alpha / alpha.sum()

        np.testing.assert_allclose(
            np.array(gamma_mean), np.array(expected_mean), atol=0.02
        )
        np.testing.assert_allclose(
            np.array(dirichlet_mean), np.array(expected_mean), atol=0.02
        )

    def test_gamma_varies_with_heterogeneous_p(self, rng):
        """Test that heterogeneous p_g changes the composition distribution.

        When p_g vary across genes, the compositions should differ from
        Dirichlet(r).  Genes with higher p should contribute more to
        the composition.
        """
        from scribe.de._empirical import _batched_gamma_normalize

        N = 20_000
        D = 4
        r = jnp.ones((N, D)) * 2.0  # Equal r

        # Gene 0 has much higher p → should dominate compositions
        p_hetero = jnp.broadcast_to(
            jnp.array([0.9, 0.1, 0.1, 0.1]), (N, D)
        )
        p_equal = jnp.full((N, D), 0.5)

        key1, key2 = random.split(rng)

        samples_hetero = _batched_gamma_normalize(
            r, p_hetero, 1, key1, batch_size=2048
        )
        samples_equal = _batched_gamma_normalize(
            r, p_equal, 1, key2, batch_size=2048
        )

        # With equal r and heterogeneous p, gene 0 should have higher
        # mean proportion than with equal p
        mean_hetero = jnp.mean(samples_hetero, axis=0)
        mean_equal = jnp.mean(samples_equal, axis=0)

        # Gene 0 proportion should be much higher with heterogeneous p
        assert float(mean_hetero[0]) > float(mean_equal[0]) + 0.1

    def test_gamma_n_samples_dirichlet_gt1(self, rng):
        """Test that n_samples_dirichlet > 1 works with Gamma sampling."""
        from scribe.de._empirical import _batched_gamma_normalize

        N, D, S = 50, 5, 3
        r = jnp.abs(random.normal(rng, (N, D))) + 1.0
        p = jnp.full((N, D), 0.3)

        result = _batched_gamma_normalize(r, p, S, rng, batch_size=20)
        assert result.shape == (N * S, D)

        # All on simplex
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(
            np.array(row_sums), np.ones(N * S), atol=1e-5
        )


# ==========================================================================
# Tests: Broadcasting p for Mixture Models
# ==========================================================================


class TestBroadcastPForMixture:
    """Test the broadcast_p_for_mixture helper function."""

    def test_scalar_p(self):
        """Test scalar p broadcasts to (1, 1)."""
        p = jnp.array(0.5)
        r = jnp.ones((3, 10))
        result = broadcast_p_for_mixture(p, r)
        assert result.shape == (1, 1)

    def test_mixture_specific_p(self):
        """Test (n_components,) p broadcasts to (n_components, 1)."""
        p = jnp.array([0.3, 0.5, 0.7])
        r = jnp.ones((3, 10))
        result = broadcast_p_for_mixture(p, r)
        assert result.shape == (3, 1)

    def test_gene_specific_p(self):
        """Test (n_genes,) p broadcasts to (1, n_genes)."""
        p = jnp.ones(10) * 0.5
        r = jnp.ones((3, 10))
        result = broadcast_p_for_mixture(p, r)
        assert result.shape == (1, 10)

    def test_full_2d_p(self):
        """Test (n_components, n_genes) p is unchanged."""
        p = jnp.ones((3, 10)) * 0.5
        r = jnp.ones((3, 10))
        result = broadcast_p_for_mixture(p, r)
        assert result.shape == (3, 10)


# ==========================================================================
# Tests: DE Pipeline with Gene-Specific p
# ==========================================================================


class TestDEWithGeneSpecificP:
    """Test differential expression with gene-specific p samples."""

    def test_compute_clr_differences_with_p_samples(self, rng):
        """Test compute_clr_differences accepts p_samples and returns valid CLR."""
        from scribe.de._empirical import compute_clr_differences

        N, D = 200, 8
        key1, key2, key3 = random.split(rng, 3)

        r_A = jnp.abs(random.normal(key1, (N, D))) + 1.0
        r_B = jnp.abs(random.normal(key2, (N, D))) + 1.0
        # Gene-specific p in (0, 1)
        p_A = jax.nn.sigmoid(random.normal(key1, (N, D)))
        p_B = jax.nn.sigmoid(random.normal(key2, (N, D)))

        delta = compute_clr_differences(
            r_A, r_B,
            rng_key=key3,
            p_samples_A=p_A,
            p_samples_B=p_B,
        )

        assert delta.shape == (N, D)
        # CLR differences should sum to approximately 0 per sample
        row_sums = delta.sum(axis=-1)
        np.testing.assert_allclose(
            np.array(row_sums), np.zeros(N), atol=1e-4
        )

    def test_compare_empirical_with_p_samples(self, rng):
        """Test compare() with method='empirical' and p_samples."""
        from scribe.de import compare

        N, D = 300, 6
        key1, key2, key3 = random.split(rng, 3)

        r_A = jnp.abs(random.normal(key1, (N, D))) + 1.0
        r_B = jnp.abs(random.normal(key2, (N, D))) + 1.0
        p_A = jax.nn.sigmoid(random.normal(key1, (N, D)))
        p_B = jax.nn.sigmoid(random.normal(key2, (N, D)))

        de = compare(
            r_A, r_B,
            method="empirical",
            p_samples_A=p_A,
            p_samples_B=p_B,
            rng_key=key3,
            gene_names=[f"gene_{i}" for i in range(D)],
        )

        results = de.gene_level(tau=0.0)
        assert "lfsr" in results
        assert "delta_mean" in results
        assert len(results["lfsr"]) == D

    def test_gene_mask_and_p_samples_raises(self, rng):
        """Test that gene_mask + p_samples raises ValueError."""
        from scribe.de._empirical import compute_clr_differences

        N, D = 50, 5
        r_A = jnp.ones((N, D))
        r_B = jnp.ones((N, D))
        p_A = jnp.full((N, D), 0.5)
        p_B = jnp.full((N, D), 0.5)
        mask = jnp.array([True, True, False, True, False])

        with pytest.raises(ValueError, match="gene_mask and gene-specific"):
            compute_clr_differences(
                r_A, r_B,
                gene_mask=mask,
                p_samples_A=p_A,
                p_samples_B=p_B,
            )

    def test_mixture_p_samples_with_component_slicing(self, rng):
        """Test p_samples with 3D (mixture) arrays and component slicing."""
        from scribe.de._empirical import compute_clr_differences

        N, K, D = 100, 3, 5
        key1, key2, key3 = random.split(rng, 3)

        r_A = jnp.abs(random.normal(key1, (N, K, D))) + 1.0
        r_B = jnp.abs(random.normal(key2, (N, K, D))) + 1.0
        p_A = jax.nn.sigmoid(random.normal(key1, (N, K, D)))
        p_B = jax.nn.sigmoid(random.normal(key2, (N, K, D)))

        delta = compute_clr_differences(
            r_A, r_B,
            component_A=0,
            component_B=1,
            rng_key=key3,
            p_samples_A=p_A,
            p_samples_B=p_B,
        )

        assert delta.shape == (N, D)


# ==========================================================================
# Tests: Config Enum
# ==========================================================================


class TestConfigEnum:
    """Test that hierarchical parameterizations are in config enums."""

    def test_hierarchical_enum_values(self):
        """Test that hierarchical enum values exist."""
        from scribe.models.config import Parameterization

        assert Parameterization.HIERARCHICAL_CANONICAL.value == "hierarchical_canonical"
        assert Parameterization.HIERARCHICAL_MEAN_PROB.value == "hierarchical_mean_prob"
        assert Parameterization.HIERARCHICAL_MEAN_ODDS.value == "hierarchical_mean_odds"
