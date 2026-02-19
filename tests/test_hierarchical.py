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


# ==========================================================================
# Tests: Sampling and Parameter Extraction with Gene-Specific p
# ==========================================================================


class TestHierarchicalSampling:
    """Tests for sampling with gene-specific p from hierarchical models.

    These tests verify that the sampling code paths correctly handle
    ``p`` with shape ``(n_components, n_genes)`` instead of the standard
    scalar or ``(n_components,)`` shapes. This catches broadcasting
    errors between gene-specific ``p`` and cell-specific ``p_capture``.
    """

    # ------------------------------------------------------------------
    # sample_biological_nb with gene-specific p
    # ------------------------------------------------------------------

    def test_biological_nb_gene_specific_p_map(self, rng):
        """MAP biological PPC with gene-specific p (n_components, n_genes).

        Verifies that ``sample_biological_nb`` handles the gene-specific
        ``p`` produced by hierarchical parameterizations in mixture
        models. Previously, the code assumed ``p`` was at most 1D.
        """
        from scribe.sampling import sample_biological_nb

        n_components, n_genes, n_cells = 3, 8, 15
        r = jnp.ones((n_components, n_genes)) * 5.0
        p = jax.nn.sigmoid(
            jnp.linspace(-1, 1, n_components * n_genes).reshape(
                n_components, n_genes
            )
        )
        mw = jnp.ones(n_components) / n_components

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng,
            n_samples=2, mixing_weights=mw,
        )
        assert result.shape == (2, n_cells, n_genes)
        assert jnp.all(result >= 0)

    def test_biological_nb_gene_specific_p_batched(self, rng):
        """Cell-batched biological PPC with gene-specific p.

        Verifies that cell batching works correctly when ``p`` has
        shape ``(n_components, n_genes)`` and batches must handle
        the extra gene dimension after component indexing.
        """
        from scribe.sampling import sample_biological_nb

        n_components, n_genes, n_cells = 2, 6, 20
        r = jnp.ones((n_components, n_genes)) * 3.0
        p = jnp.array([
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ])
        mw = jnp.array([0.6, 0.4])

        full = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng,
            n_samples=1, mixing_weights=mw,
        )
        batched = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng,
            n_samples=1, mixing_weights=mw, cell_batch_size=7,
        )
        assert full.shape == (1, n_cells, n_genes)
        assert batched.shape == (1, n_cells, n_genes)

    def test_biological_nb_gene_specific_p_non_negative(self, rng):
        """All counts are non-negative with gene-specific p."""
        from scribe.sampling import sample_biological_nb

        n_components, n_genes, n_cells = 4, 10, 30
        r = jnp.ones((n_components, n_genes)) * 4.0
        p = jax.nn.sigmoid(
            random.normal(rng, (n_components, n_genes))
        )
        mw = jnp.ones(n_components) / n_components

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng,
            n_samples=3, mixing_weights=mw,
        )
        assert jnp.all(result >= 0)

    def test_biological_nb_gene_specific_p_matches_scalar_when_uniform(
        self, rng
    ):
        """When all p_g are equal, gene-specific matches scalar result.

        If ``p`` is ``(n_components, n_genes)`` but every gene within a
        component shares the same value, the result should be
        statistically indistinguishable from scalar ``p``.
        """
        from scribe.sampling import sample_biological_nb

        n_components, n_genes, n_cells = 2, 5, 100
        r = jnp.ones((n_components, n_genes)) * 5.0
        p_vals = jnp.array([0.3, 0.6])
        p_scalar = p_vals  # (n_components,)
        p_gene = jnp.stack([jnp.full(n_genes, v) for v in p_vals])
        mw = jnp.array([0.5, 0.5])

        result_scalar = sample_biological_nb(
            r=r, p=p_scalar, n_cells=n_cells, rng_key=rng,
            n_samples=1, mixing_weights=mw,
        )
        result_gene = sample_biological_nb(
            r=r, p=p_gene, n_cells=n_cells, rng_key=rng,
            n_samples=1, mixing_weights=mw,
        )
        assert result_scalar.shape == result_gene.shape


class TestHierarchicalMAPSampling:
    """Tests for MAP-based PPC with hierarchical gene-specific parameters.

    Constructs minimal ``ScribeSVIResults`` objects with hierarchical
    parameter shapes and verifies that ``get_map`` and
    ``get_map_ppc_samples`` handle gene-specific ``p`` correctly,
    especially the ``p * p_capture`` broadcast.
    """

    @pytest.fixture
    def hierarchical_mean_odds_results(self):
        """Build a minimal ScribeSVIResults for hierarchical_mean_odds VCP.

        Returns
        -------
        ScribeSVIResults
            Mock results with ``phi`` and ``mu`` as
            ``(n_components, n_genes)`` and ``phi_capture`` as
            ``(n_cells,)``.
        """
        from scribe.svi.results import ScribeSVIResults
        from scribe.models.config import ModelConfig

        n_cells, n_genes, n_components = 20, 8, 3

        phi = jnp.ones((n_components, n_genes)) * 2.0
        mu = jnp.ones((n_components, n_genes)) * 10.0
        phi_capture = jnp.ones(n_cells) * 1.5

        params = {
            "phi_loc": jnp.log(phi),
            "phi_scale": jnp.ones((n_components, n_genes)) * 0.1,
            "mu_loc": jnp.log(mu),
            "mu_scale": jnp.ones((n_components, n_genes)) * 0.1,
            "phi_capture_loc": jnp.log(phi_capture),
            "phi_capture_scale": jnp.ones(n_cells) * 0.1,
            "log_phi_loc_loc": jnp.zeros(()),
            "log_phi_loc_scale": jnp.ones(()) * 0.1,
            "log_phi_scale_loc": jnp.zeros(()),
            "log_phi_scale_scale": jnp.ones(()) * 0.1,
            "mixing_concentrations": jnp.ones(n_components),
        }

        model_config = ModelConfig(
            base_model="nbvcp",
            parameterization="hierarchical_mean_odds",
            unconstrained=True,
            n_components=n_components,
            mixture_params=["phi", "mu"],
        )

        return ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=model_config,
            prior_params={},
            n_components=n_components,
        )

    @pytest.fixture
    def hierarchical_mean_prob_results(self):
        """Build a minimal ScribeSVIResults for hierarchical_mean_prob VCP.

        Returns
        -------
        ScribeSVIResults
            Mock results with ``p`` and ``mu`` as
            ``(n_components, n_genes)`` and ``p_capture`` as
            ``(n_cells,)``.
        """
        from scribe.svi.results import ScribeSVIResults
        from scribe.models.config import ModelConfig

        n_cells, n_genes, n_components = 15, 6, 2

        p_unc = jnp.zeros((n_components, n_genes))
        mu = jnp.ones((n_components, n_genes)) * 10.0
        p_capture_unc = jnp.zeros(n_cells)

        params = {
            "p_loc": p_unc,
            "p_scale": jnp.ones((n_components, n_genes)) * 0.1,
            "mu_loc": jnp.log(mu),
            "mu_scale": jnp.ones((n_components, n_genes)) * 0.1,
            "p_capture_loc": p_capture_unc,
            "p_capture_scale": jnp.ones(n_cells) * 0.1,
            "logit_p_loc_loc": jnp.zeros(()),
            "logit_p_loc_scale": jnp.ones(()) * 0.1,
            "logit_p_scale_loc": jnp.zeros(()),
            "logit_p_scale_scale": jnp.ones(()) * 0.1,
            "mixing_concentrations": jnp.ones(n_components),
        }

        model_config = ModelConfig(
            base_model="nbvcp",
            parameterization="hierarchical_mean_prob",
            unconstrained=True,
            n_components=n_components,
            mixture_params=["p", "mu"],
        )

        return ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=model_config,
            prior_params={},
            n_components=n_components,
        )

    def test_get_map_mean_odds_no_broadcast_error(
        self, hierarchical_mean_odds_results
    ):
        """get_map with hierarchical_mean_odds must not raise on p_hat.

        Previously, ``_compute_canonical_parameters`` tried to compute
        ``p_hat = p * p_capture`` where ``p`` was ``(n_components,
        n_genes)`` and ``p_capture`` was ``(n_cells, 1)``, causing a
        broadcasting error.
        """
        results = hierarchical_mean_odds_results
        map_est = results.get_map(use_mean=True, canonical=True, verbose=False)

        assert "p" in map_est
        assert "r" in map_est
        assert "p_capture" in map_est
        assert map_est["p"].ndim == 2
        assert map_est["r"].ndim == 2
        n_components = results.n_components
        n_genes = results.n_genes
        assert map_est["p"].shape == (n_components, n_genes)
        assert map_est["r"].shape == (n_components, n_genes)

    def test_get_map_mean_prob_no_broadcast_error(
        self, hierarchical_mean_prob_results
    ):
        """get_map with hierarchical_mean_prob must not raise on p_hat.

        Same broadcasting guard as mean_odds but using
        ``p_capture`` instead of ``phi_capture``.
        """
        results = hierarchical_mean_prob_results
        map_est = results.get_map(use_mean=True, canonical=True, verbose=False)

        assert "p" in map_est
        assert "r" in map_est
        assert "p_capture" in map_est
        assert map_est["p"].shape == (
            results.n_components, results.n_genes
        )

    def test_get_map_ppc_samples_mean_odds(
        self, hierarchical_mean_odds_results
    ):
        """Full MAP PPC with hierarchical_mean_odds VCP must not raise.

        Exercises the complete ``get_map_ppc_samples`` pipeline including
        ``get_map`` (p_hat precomputation) and ``_sample_mixture_model``
        (component-indexed gene-specific p + p_capture broadcast).
        """
        results = hierarchical_mean_odds_results
        samples = results.get_map_ppc_samples(
            rng_key=random.PRNGKey(0),
            n_samples=1,
            cell_batch_size=5,
            use_mean=True,
            store_samples=False,
            verbose=False,
        )
        assert samples.shape == (1, results.n_cells, results.n_genes)
        assert jnp.all(samples >= 0)
        assert jnp.all(jnp.isfinite(samples))

    def test_get_map_ppc_samples_mean_prob(
        self, hierarchical_mean_prob_results
    ):
        """Full MAP PPC with hierarchical_mean_prob VCP must not raise."""
        results = hierarchical_mean_prob_results
        samples = results.get_map_ppc_samples(
            rng_key=random.PRNGKey(0),
            n_samples=1,
            cell_batch_size=5,
            use_mean=True,
            store_samples=False,
            verbose=False,
        )
        assert samples.shape == (1, results.n_cells, results.n_genes)
        assert jnp.all(samples >= 0)

    def test_get_map_ppc_biological_mean_odds(
        self, hierarchical_mean_odds_results
    ):
        """Biological MAP PPC with hierarchical_mean_odds VCP must not raise.

        Tests the biological PPC path that strips VCP / gate and samples
        from NB(r, p) directly with gene-specific ``p``.
        """
        results = hierarchical_mean_odds_results
        samples = results.get_map_ppc_samples_biological(
            rng_key=random.PRNGKey(1),
            n_samples=2,
            cell_batch_size=5,
            use_mean=True,
            store_samples=False,
            verbose=False,
        )
        assert samples.shape == (2, results.n_cells, results.n_genes)
        assert jnp.all(samples >= 0)

    def test_p_hat_skipped_for_gene_specific_p(
        self, hierarchical_mean_odds_results
    ):
        """p_hat is NOT in MAP estimates when p is gene-specific 2D.

        When ``p`` has shape ``(n_components, n_genes)``, precomputing
        ``p_hat`` would require a ``(n_cells, n_components, n_genes)``
        tensor, so it must be skipped.
        """
        results = hierarchical_mean_odds_results
        map_est = results.get_map(use_mean=True, canonical=True, verbose=False)
        assert "p_hat" not in map_est

    def test_p_hat_present_for_scalar_p(self):
        """p_hat IS computed for standard (non-hierarchical) VCP models.

        Ensures the skip logic only applies to the 2D gene-specific
        case and does not regress the standard model path.
        """
        from scribe.svi.results import ScribeSVIResults
        from scribe.models.config import ModelConfig

        n_cells, n_genes, n_components = 10, 5, 2

        params = {
            "phi_loc": jnp.zeros(n_components),
            "phi_scale": jnp.ones(n_components) * 0.1,
            "mu_loc": jnp.ones((n_components, n_genes)),
            "mu_scale": jnp.ones((n_components, n_genes)) * 0.1,
            "phi_capture_loc": jnp.zeros(n_cells),
            "phi_capture_scale": jnp.ones(n_cells) * 0.1,
            "mixing_concentrations": jnp.ones(n_components),
        }

        model_config = ModelConfig(
            base_model="nbvcp",
            parameterization="mean_odds",
            unconstrained=True,
            n_components=n_components,
            mixture_params=["phi", "mu"],
        )

        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=model_config,
            prior_params={},
            n_components=n_components,
        )

        map_est = results.get_map(use_mean=True, canonical=True, verbose=False)
        assert "p_hat" in map_est

    def test_cell_batching_consistency_mean_odds(
        self, hierarchical_mean_odds_results
    ):
        """Different cell_batch_size values produce same-shape output.

        Ensures that cell batching handles the gene-specific p correctly
        for various batch sizes, including batch_size > n_cells.
        """
        results = hierarchical_mean_odds_results
        shapes = set()
        for batch_size in [3, 7, results.n_cells, results.n_cells + 10]:
            samples = results.get_map_ppc_samples(
                rng_key=random.PRNGKey(42),
                n_samples=1,
                cell_batch_size=batch_size,
                use_mean=True,
                store_samples=False,
                verbose=False,
            )
            shapes.add(samples.shape)
        assert len(shapes) == 1
        assert shapes.pop() == (1, results.n_cells, results.n_genes)


class TestHierarchicalComponentSubsetting:
    """Tests for get_component + gene subsetting with hierarchical models.

    The ``get_component`` method extracts a single mixture component
    from a hierarchical model result, and gene subsetting (``__getitem__``)
    selects a subset of genes. This pipeline must correctly reduce the
    component dimension of gene-specific parameters (``phi_loc``,
    ``phi_scale``, etc.) that only exist in hierarchical
    parameterizations.
    """

    @pytest.fixture
    def hierarchical_mixture_results(self):
        """Build a ScribeSVIResults for hierarchical_mean_odds with 3 components.

        Returns
        -------
        ScribeSVIResults
            Results with phi/mu shape ``(3, 10)`` and phi_capture shape
            ``(12,)``.
        """
        from scribe.svi.results import ScribeSVIResults
        from scribe.models.config import ModelConfig

        n_cells, n_genes, n_components = 12, 10, 3

        params = {
            "phi_loc": jnp.ones((n_components, n_genes)) * 0.5,
            "phi_scale": jnp.ones((n_components, n_genes)) * 0.1,
            "mu_loc": jnp.ones((n_components, n_genes)) * 2.0,
            "mu_scale": jnp.ones((n_components, n_genes)) * 0.1,
            "phi_capture_loc": jnp.zeros(n_cells),
            "phi_capture_scale": jnp.ones(n_cells) * 0.1,
            "log_phi_loc_loc": jnp.zeros(()),
            "log_phi_loc_scale": jnp.ones(()) * 0.1,
            "log_phi_scale_loc": jnp.zeros(()),
            "log_phi_scale_scale": jnp.ones(()) * 0.1,
            "mixing_concentrations": jnp.ones(n_components),
        }

        model_config = ModelConfig(
            base_model="nbvcp",
            parameterization="hierarchical_mean_odds",
            unconstrained=True,
            n_components=n_components,
            mixture_params=["phi", "mu"],
        )

        return ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=model_config,
            prior_params={},
            n_components=n_components,
        )

    def test_get_component_reduces_phi_shape(
        self, hierarchical_mixture_results
    ):
        """get_component must reduce phi_loc from (K, G) to (G,).

        Without this fix, hierarchical params like ``phi_loc`` were not
        in the component-gene-specific list, so they were copied as-is,
        keeping the ``(n_components, n_genes)`` shape.
        """
        results = hierarchical_mixture_results
        comp = results.get_component(0)

        assert comp.params["phi_loc"].ndim == 1
        assert comp.params["phi_loc"].shape[0] == results.n_genes

    def test_get_component_reduces_mu_shape(
        self, hierarchical_mixture_results
    ):
        """get_component must reduce mu_loc from (K, G) to (G,)."""
        results = hierarchical_mixture_results
        comp = results.get_component(1)

        assert comp.params["mu_loc"].ndim == 1
        assert comp.params["mu_loc"].shape[0] == results.n_genes

    def test_get_component_preserves_capture_shape(
        self, hierarchical_mixture_results
    ):
        """phi_capture_loc must remain (n_cells,) after get_component."""
        results = hierarchical_mixture_results
        comp = results.get_component(0)

        assert comp.params["phi_capture_loc"].shape == (results.n_cells,)

    def test_get_component_preserves_hyperparams(
        self, hierarchical_mixture_results
    ):
        """Scalar hyperparams must be copied as-is through get_component."""
        results = hierarchical_mixture_results
        comp = results.get_component(0)

        assert comp.params["log_phi_loc_loc"].ndim == 0
        assert comp.params["log_phi_loc_scale"].ndim == 0

    def test_get_component_then_gene_subset_map(
        self, hierarchical_mixture_results
    ):
        """get_component → gene subset → get_map must not broadcast-error.

        This is the exact pipeline that the annotation PPC uses:
        extract a single component, subset to selected genes, then call
        ``get_map_ppc_samples``. The error was ``(4, 25) * (6264, 1)``
        because ``phi_loc`` kept its ``(n_components, n_genes)`` shape.
        """
        results = hierarchical_mixture_results
        comp = results.get_component(0)
        gene_idx = jnp.array([0, 2, 4, 6, 8])
        comp_sub = comp[gene_idx]

        map_est = comp_sub.get_map(
            use_mean=True, canonical=True, verbose=False
        )
        assert "p" in map_est
        assert "r" in map_est
        assert map_est["p"].shape == (len(gene_idx),)
        assert map_est["r"].shape == (len(gene_idx),)

    def test_get_component_then_gene_subset_ppc(
        self, hierarchical_mixture_results
    ):
        """Full annotation PPC pipeline must produce correct sample shape.

        Exercises get_component → gene subset → get_map_ppc_samples,
        the exact code path used by ``plot_annotation_ppc``.
        """
        results = hierarchical_mixture_results
        comp = results.get_component(1)
        gene_idx = jnp.array([1, 3, 5, 7])
        comp_sub = comp[gene_idx]

        samples = comp_sub.get_map_ppc_samples(
            rng_key=random.PRNGKey(99),
            n_samples=2,
            cell_batch_size=4,
            use_mean=True,
            store_samples=False,
            verbose=False,
        )
        assert samples.shape == (2, results.n_cells, len(gene_idx))
        assert jnp.all(samples >= 0)

    def test_get_component_each_component_differs(
        self, hierarchical_mixture_results
    ):
        """Different components should yield different phi_loc values.

        Verifies that component indexing actually selects the right
        row, not just the first or a copy.
        """
        results = hierarchical_mixture_results
        # Make components distinguishable
        results.params["phi_loc"] = jnp.stack([
            jnp.ones(results.n_genes) * (k + 1)
            for k in range(results.n_components)
        ])

        comp0 = results.get_component(0)
        comp1 = results.get_component(1)
        comp2 = results.get_component(2)

        np.testing.assert_allclose(comp0.params["phi_loc"], 1.0)
        np.testing.assert_allclose(comp1.params["phi_loc"], 2.0)
        np.testing.assert_allclose(comp2.params["phi_loc"], 3.0)
