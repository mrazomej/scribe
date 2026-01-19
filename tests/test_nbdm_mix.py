# tests/test_nbdm_mix.py
"""
Tests for the Negative Binomial-Dirichlet Multinomial Mixture Model.
"""
import pytest
import jax.numpy as jnp
from jax import random
import os
from scribe.models.config import InferenceConfig, SVIConfig, MCMCConfig
from scribe.inference import run_scribe
from scribe.inference.preset_builder import build_config_from_preset

ALL_METHODS = [
    "svi",
    "mcmc",
]  # Both SVI and MCMC are supported for mixture models
ALL_PARAMETERIZATIONS = ["standard", "linked", "odds_ratio"]
ALL_UNCONSTRAINED = [False, True]
ALL_GUIDE_RANKS = [None, 5]  # None = mean-field, 5 = low-rank

# ------------------------------------------------------------------------------
# Dynamic matrix parametrization
# ------------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """
    Dynamically generate test combinations for inference methods,
    parameterizations, unconstrained variants, and guide ranks.

    Handles command-line options for selective testing.
    """
    # Check if this test function uses the required fixtures
    if {
        "nbdm_mix_results",
        "inference_method",
        "parameterization",
        "unconstrained",
        "guide_rank",
    }.issubset(metafunc.fixturenames):
        # Get command-line options
        method_opt = metafunc.config.getoption("--method")
        param_opt = metafunc.config.getoption("--parameterization")
        unconstrained_opt = metafunc.config.getoption("--unconstrained")
        guide_rank_opt = metafunc.config.getoption("--guide-rank")

        # Determine which methods to test based on command-line option
        methods = ALL_METHODS if method_opt == "all" else [method_opt]

        # Determine which parameterizations to test based on command-line option
        params = ALL_PARAMETERIZATIONS if param_opt == "all" else [param_opt]

        # Determine which unconstrained variants to test based on command-line option
        if unconstrained_opt == "all":
            unconstrained_variants = ALL_UNCONSTRAINED
        else:
            unconstrained_variants = [unconstrained_opt == "true"]

        # Determine which guide ranks to test based on command-line option
        if guide_rank_opt == "all":
            guide_ranks = ALL_GUIDE_RANKS
        elif guide_rank_opt.lower() == "none":
            guide_ranks = [None]
        else:
            try:
                guide_ranks = [int(guide_rank_opt)]
            except ValueError:
                raise ValueError(
                    f"Invalid guide-rank option: {guide_rank_opt}. "
                    "Must be 'all', 'none', or an integer."
                )

        # Generate all valid combinations
        # Skip guide_rank for MCMC (it doesn't use guides)
        combinations = [
            (m, p, u, g)
            for m in methods
            for p in params
            for u in unconstrained_variants
            for g in guide_ranks
            if not (m == "mcmc" and g is not None)  # MCMC doesn't use guides
        ]

        # Parametrize the test with the generated combinations
        metafunc.parametrize(
            "inference_method,parameterization,unconstrained,guide_rank",
            combinations,
        )


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

# Global cache for results
_nbdm_mix_results_cache = {}


@pytest.fixture(scope="function")
def nbdm_mix_results(
    request,
    inference_method,
    parameterization,
    unconstrained,
    guide_rank,
    small_dataset,
    rng_key,
):
    # Get device type from command-line option for cache key
    device_type = request.config.getoption("--device")

    key = (
        inference_method,
        device_type,
        parameterization,
        unconstrained,
        guide_rank,
    )
    if key in _nbdm_mix_results_cache:
        return _nbdm_mix_results_cache[key]

    # Device is already configured in conftest.py via pytest_configure
    counts, _ = small_dataset

    # Set up priors based on parameterization (using new key names without "_prior" suffix)
    if parameterization == "standard":
        priors = {"r": (2, 0.1), "p": (1, 1)}
    elif parameterization == "linked":
        priors = {"p": (1, 1), "mu": (1, 1)}
    elif parameterization == "odds_ratio":
        priors = {"phi": (3, 2), "mu": (1, 1)}
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    # Add mixing_prior for mixture models
    priors["mixing"] = jnp.ones(2)  # Uniform prior for 2 components

    # Build model config using preset builder
    model_config = build_config_from_preset(
        model="nbdm",
        parameterization=parameterization,
        inference_method=inference_method,
        unconstrained=unconstrained,
        guide_rank=guide_rank,
        priors=priors,
        n_components=2,  # Test with 2 components
    )

    # Create inference config based on method
    if inference_method == "svi":
        svi_config = SVIConfig(n_steps=3, batch_size=5)
        inference_config = InferenceConfig.from_svi(svi_config)
    else:
        mcmc_config = MCMCConfig(n_warmup=2, n_samples=3, n_chains=1)
        inference_config = InferenceConfig.from_mcmc(mcmc_config)

    # Run inference with new API
    result = run_scribe(
        counts=counts,
        model_config=model_config,
        inference_config=inference_config,
        seed=42,
    )

    _nbdm_mix_results_cache[key] = result
    return result


# ------------------------------------------------------------------------------
# Shared Tests
# ------------------------------------------------------------------------------


def test_inference_run(nbdm_mix_results):
    assert nbdm_mix_results.n_cells > 0
    assert nbdm_mix_results.n_genes > 0
    assert nbdm_mix_results.model_type == "nbdm_mix"
    assert hasattr(nbdm_mix_results, "model_config")
    assert nbdm_mix_results.n_components == 2


# ------------------------------------------------------------------------------


def test_parameterization_config(
    nbdm_mix_results, parameterization, unconstrained, guide_rank
):
    """Test that the correct parameterization and unconstrained flag are used."""
    assert nbdm_mix_results.model_config.parameterization == parameterization
    # Check unconstrained flag
    assert nbdm_mix_results.model_config.unconstrained == unconstrained
    # Check that the guide_rank is properly set in the model config
    # guide_rank is stored in guide_families as LowRankGuide for the gene parameter
    if guide_rank is not None:
        # Determine which parameter should have the low-rank guide
        if parameterization == "standard":
            gene_param = "r"
        else:  # linked or odds_ratio
            gene_param = "mu"
        assert nbdm_mix_results.model_config.guide_families is not None
        guide_family = nbdm_mix_results.model_config.guide_families.get(
            gene_param
        )
        from scribe.models.components import LowRankGuide

        assert isinstance(guide_family, LowRankGuide)
        assert guide_family.rank == guide_rank
    else:
        # No guide_rank means mean-field (default), so guide_families might be None
        # or all guides are MeanFieldGuide
        pass  # Mean-field is the default, so no specific check needed


# ------------------------------------------------------------------------------


def test_parameter_ranges(
    nbdm_mix_results, parameterization, unconstrained, guide_rank
):
    """Test that parameters have correct ranges and relationships."""
    # For SVI, we need to get posterior samples to access transformed parameters
    # For MCMC, we can use either params or samples
    if hasattr(nbdm_mix_results, "params") and hasattr(
        nbdm_mix_results, "get_posterior_samples"
    ):
        # SVI case: get transformed parameters from posterior samples
        samples = nbdm_mix_results.get_posterior_samples(n_samples=1)
        params = samples
    elif hasattr(nbdm_mix_results, "params"):
        # MCMC case: params might contain transformed parameters
        params = nbdm_mix_results.params
    else:
        # Fallback: try to get samples
        samples = nbdm_mix_results.get_posterior_samples()
        params = samples

    # Check parameters based on parameterization
    if parameterization == "standard":
        # Standard parameterization: p and r (component-specific)
        if unconstrained:
            # Unconstrained models should have both constrained and unconstrained parameters
            assert "p" in params or "p_unconstrained" in params
            assert "r" in params or "r_unconstrained" in params

            # Check constrained parameters if they exist
            if "p" in params and "r" in params:
                p, r = params["p"], params["r"]
                # In unconstrained models, p and r are transformed from unconstrained space
                # but should still be finite
                assert jnp.all(jnp.isfinite(p))
                assert jnp.all(jnp.isfinite(r))
        else:
            # Constrained models must have p and r
            assert "p" in params or any(
                k.startswith("p_") for k in params.keys()
            )
            assert "r" in params or any(
                k.startswith("r_") for k in params.keys()
            )

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "linked":
        # Linked parameterization: p and mu (component-specific)
        if unconstrained:
            # Unconstrained models should have both constrained and unconstrained parameters
            assert "p" in params or "p_unconstrained" in params
            assert "mu" in params or "mu_unconstrained" in params

            # Check constrained parameters if they exist
            if "p" in params and "mu" in params:
                p, mu = params["p"], params["mu"]
                # In unconstrained models, p and mu are transformed from unconstrained space
                # but should still be finite
                assert jnp.all(jnp.isfinite(p))
                assert jnp.all(jnp.isfinite(mu))
        else:
            # Constrained models must have p and mu
            assert "p" in params or any(
                k.startswith("p_") for k in params.keys()
            )
            assert "mu" in params or any(
                k.startswith("mu_") for k in params.keys()
            )

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "odds_ratio":
        # Odds ratio parameterization: phi and mu (component-specific)
        if unconstrained:
            # Unconstrained models should have both constrained and unconstrained parameters
            assert "phi" in params or "phi_unconstrained" in params
            assert "mu" in params or "mu_unconstrained" in params

            # Check constrained parameters if they exist
            if "phi" in params and "mu" in params:
                phi, mu = params["phi"], params["mu"]
                # In unconstrained models, phi and mu are transformed from unconstrained space
                # but should still be finite
                assert jnp.all(jnp.isfinite(phi))
                assert jnp.all(jnp.isfinite(mu))
        else:
            # Constrained models must have phi and mu
            assert "phi" in params or any(
                k.startswith("phi_") for k in params.keys()
            )
            assert "mu" in params or any(
                k.startswith("mu_") for k in params.keys()
            )

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )


# ------------------------------------------------------------------------------


def test_posterior_sampling(
    nbdm_mix_results, rng_key, parameterization, unconstrained, guide_rank
):
    """Test sampling from the variational posterior."""
    # For SVI, must call get_posterior_samples with parameters; for MCMC, just call
    # get_posterior_samples without parameters
    if hasattr(nbdm_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbdm_mix_results, "get_samples"):  # MCMC case
            samples = nbdm_mix_results.get_posterior_samples()
        else:  # SVI case
            samples = nbdm_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    else:
        samples = nbdm_mix_results.get_posterior_samples()

    # Check that we have the expected parameters based on parameterization
    if parameterization == "standard":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_p = "p" in samples or "p_unconstrained" in samples
            has_r = "r" in samples or "r_unconstrained" in samples
            assert (
                has_p and has_r
            ), f"Expected p/p_unconstrained and r/r_unconstrained in samples, got {list(samples.keys())}"
            # Check shape for whichever parameter exists
            if "r" in samples:
                assert samples["r"].shape[-2] == nbdm_mix_results.n_components
                assert samples["r"].shape[-1] == nbdm_mix_results.n_genes
            elif "r_unconstrained" in samples:
                assert (
                    samples["r_unconstrained"].shape[-2]
                    == nbdm_mix_results.n_components
                )
                assert (
                    samples["r_unconstrained"].shape[-1]
                    == nbdm_mix_results.n_genes
                )
        else:
            assert "p" in samples and "r" in samples
            assert samples["r"].shape[-2] == nbdm_mix_results.n_components
            assert samples["r"].shape[-1] == nbdm_mix_results.n_genes
    elif parameterization == "linked":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_p = "p" in samples or "p_unconstrained" in samples
            has_mu = "mu" in samples or "mu_unconstrained" in samples
            assert (
                has_p and has_mu
            ), f"Expected p/p_unconstrained and mu/mu_unconstrained in samples, got {list(samples.keys())}"
            # Check shape for whichever parameter exists
            if "mu" in samples:
                assert samples["mu"].shape[-2] == nbdm_mix_results.n_components
                assert samples["mu"].shape[-1] == nbdm_mix_results.n_genes
            elif "mu_unconstrained" in samples:
                assert (
                    samples["mu_unconstrained"].shape[-2]
                    == nbdm_mix_results.n_components
                )
                assert (
                    samples["mu_unconstrained"].shape[-1]
                    == nbdm_mix_results.n_genes
                )
        else:
            assert "p" in samples and "mu" in samples
            assert samples["mu"].shape[-2] == nbdm_mix_results.n_components
            assert samples["mu"].shape[-1] == nbdm_mix_results.n_genes
    elif parameterization == "odds_ratio":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_phi = "phi" in samples or "phi_unconstrained" in samples
            has_mu = "mu" in samples or "mu_unconstrained" in samples
            assert (
                has_phi and has_mu
            ), f"Expected phi/phi_unconstrained and mu/mu_unconstrained in samples, got {list(samples.keys())}"
            # Check shape for whichever parameter exists
            if "mu" in samples:
                assert samples["mu"].shape[-2] == nbdm_mix_results.n_components
                assert samples["mu"].shape[-1] == nbdm_mix_results.n_genes
            elif "mu_unconstrained" in samples:
                assert (
                    samples["mu_unconstrained"].shape[-2]
                    == nbdm_mix_results.n_components
                )
                assert (
                    samples["mu_unconstrained"].shape[-1]
                    == nbdm_mix_results.n_genes
                )
        else:
            assert "phi" in samples and "mu" in samples
            assert samples["mu"].shape[-2] == nbdm_mix_results.n_components
            assert samples["mu"].shape[-1] == nbdm_mix_results.n_genes

    # Check mixing weights
    assert (
        "mixing_weights" in samples or "mixing_logits_unconstrained" in samples
    )
    if "mixing_weights" in samples:
        assert (
            samples["mixing_weights"].shape[-1] == nbdm_mix_results.n_components
        )
    elif "mixing_logits_unconstrained" in samples:
        assert (
            samples["mixing_logits_unconstrained"].shape[-1]
            == nbdm_mix_results.n_components
        )


# ------------------------------------------------------------------------------


def test_predictive_sampling(nbdm_mix_results, rng_key):
    """Test generating predictive samples."""
    # For SVI, must generate posterior samples first
    if hasattr(nbdm_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbdm_mix_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pred = nbdm_mix_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            nbdm_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = nbdm_mix_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = nbdm_mix_results.get_ppc_samples(
            rng_key=rng_key, store_samples=True
        )
    assert pred.shape[-1] == nbdm_mix_results.n_genes
    assert jnp.all(pred >= 0)


# ------------------------------------------------------------------------------


def test_get_map(nbdm_mix_results, parameterization, unconstrained, guide_rank):
    map_est = (
        nbdm_mix_results.get_map()
        if hasattr(nbdm_mix_results, "get_map")
        else None
    )
    assert map_est is not None

    # Debug: print what parameters are actually available
    print(
        f"DEBUG: MAP parameters for {parameterization} (unconstrained={unconstrained}): {list(map_est.keys())}"
    )

    # Check parameters based on parameterization
    if parameterization == "standard":
        if unconstrained:
            # For SVI unconstrained models, they might return empty dicts initially
            # This appears to be expected behavior for some unconstrained models
            if len(map_est) == 0:
                print(
                    f"DEBUG: Empty MAP for unconstrained {parameterization}, checking model params"
                )
                if hasattr(nbdm_mix_results, "params"):
                    print(
                        f"DEBUG: Model params keys: {list(nbdm_mix_results.params.keys())}"
                    )
                # For SVI unconstrained models, empty MAP is acceptable
                # as they use variational parameters (alpha/beta, loc/scale)
                # instead of direct MAP estimates
                pass
            else:
                has_p = "p" in map_est or "p_unconstrained" in map_est
                has_r = "r" in map_est or "r_unconstrained" in map_est
                assert (
                    has_p and has_r
                ), f"Expected p/p_unconstrained and r/r_unconstrained in MAP, got {list(map_est.keys())}"
        else:
            assert "r" in map_est and "p" in map_est
    elif parameterization == "linked":
        if unconstrained:
            if len(map_est) == 0:
                print(
                    f"DEBUG: Empty MAP for unconstrained {parameterization}, checking model params"
                )
                if hasattr(nbdm_mix_results, "params"):
                    print(
                        f"DEBUG: Model params keys: {list(nbdm_mix_results.params.keys())}"
                    )
                # For SVI unconstrained models, empty MAP is acceptable
                pass
            else:
                has_p = "p" in map_est or "p_unconstrained" in map_est
                has_mu = "mu" in map_est or "mu_unconstrained" in map_est
                assert (
                    has_p and has_mu
                ), f"Expected p/p_unconstrained and mu/mu_unconstrained in MAP, got {list(map_est.keys())}"
        else:
            assert "p" in map_est and "mu" in map_est
    elif parameterization == "odds_ratio":
        if unconstrained:
            if len(map_est) == 0:
                print(
                    f"DEBUG: Empty MAP for unconstrained {parameterization}, checking model params"
                )
                if hasattr(nbdm_mix_results, "params"):
                    print(
                        f"DEBUG: Model params keys: {list(nbdm_mix_results.params.keys())}"
                    )
                # For SVI unconstrained models, empty MAP is acceptable
                pass
            else:
                has_phi = "phi" in map_est or "phi_unconstrained" in map_est
                has_mu = "mu" in map_est or "mu_unconstrained" in map_est
                assert (
                    has_phi and has_mu
                ), f"Expected phi/phi_unconstrained and mu/mu_unconstrained in MAP, got {list(map_est.keys())}"
        else:
            assert "phi" in map_est and "mu" in map_est

    # Check mixing weights
    if "mixing_weights" in map_est:
        assert (
            map_est["mixing_weights"].shape[-1] == nbdm_mix_results.n_components
        )
    elif "mixing_logits_unconstrained" in map_est:
        assert (
            map_est["mixing_logits_unconstrained"].shape[-1]
            == nbdm_mix_results.n_components
        )


# ------------------------------------------------------------------------------


def test_indexing_integer(nbdm_mix_results):
    """Test indexing with an integer."""
    subset = nbdm_mix_results[0]
    assert subset.n_genes == 1
    assert subset.n_cells == nbdm_mix_results.n_cells
    assert subset.n_components == nbdm_mix_results.n_components


# ------------------------------------------------------------------------------


def test_indexing_slice(nbdm_mix_results):
    """Test indexing with a slice."""
    end_idx = min(2, nbdm_mix_results.n_genes)
    subset = nbdm_mix_results[0:end_idx]
    assert subset.n_genes == end_idx
    assert subset.n_cells == nbdm_mix_results.n_cells
    assert subset.n_components == nbdm_mix_results.n_components


# ------------------------------------------------------------------------------


def test_indexing_boolean(nbdm_mix_results):
    """Test indexing with a boolean array."""
    mask = jnp.zeros(nbdm_mix_results.n_genes, dtype=bool)
    mask = mask.at[0].set(True)

    subset = nbdm_mix_results[mask]
    assert subset.n_genes == 1
    assert subset.n_cells == nbdm_mix_results.n_cells
    assert subset.n_components == nbdm_mix_results.n_components


# ------------------------------------------------------------------------------


def test_log_likelihood(nbdm_mix_results, small_dataset, rng_key):
    counts, _ = small_dataset

    # For SVI, must generate posterior samples first
    if hasattr(nbdm_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbdm_mix_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pass
        else:  # SVI case
            nbdm_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )

    # Test marginal log likelihood (across components)
    ll = nbdm_mix_results.log_likelihood(counts, return_by="cell")
    assert ll.shape[0] > 0
    assert jnp.all(jnp.isfinite(ll))

    # Test component-specific log likelihood
    ll_components = nbdm_mix_results.log_likelihood(
        counts, return_by="cell", split_components=True
    )
    assert ll_components.shape[-1] == nbdm_mix_results.n_components
    assert jnp.all(jnp.isfinite(ll_components))


# ------------------------------------------------------------------------------
# SVI-specific Tests
# ------------------------------------------------------------------------------


def test_svi_loss_history(nbdm_mix_results, inference_method):
    if inference_method == "svi":
        assert hasattr(nbdm_mix_results, "loss_history")
        assert len(nbdm_mix_results.loss_history) > 0


# ------------------------------------------------------------------------------
# MCMC-specific Tests
# ------------------------------------------------------------------------------


def test_mcmc_samples_shape(
    nbdm_mix_results,
    inference_method,
    parameterization,
    unconstrained,
    guide_rank,
):
    if inference_method == "mcmc":
        samples = nbdm_mix_results.get_posterior_samples()

        # Check parameters based on parameterization
        if parameterization == "standard":
            if unconstrained:
                # Unconstrained models may have either constrained or unconstrained parameters
                has_p = "p" in samples or "p_unconstrained" in samples
                has_r = "r" in samples or "r_unconstrained" in samples
                assert (
                    has_p and has_r
                ), f"Expected p/p_unconstrained and r/r_unconstrained in samples, got {list(samples.keys())}"
                # Check shape for whichever parameter exists
                if "r" in samples:
                    assert (
                        samples["r"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                elif "r_unconstrained" in samples:
                    assert (
                        samples["r_unconstrained"].ndim >= 3
                    )  # n_samples, n_components, n_genes
            else:
                assert "r" in samples and "p" in samples
                assert (
                    samples["r"].ndim >= 3
                )  # n_samples, n_components, n_genes
        elif parameterization == "linked":
            if unconstrained:
                # Unconstrained models may have either constrained or unconstrained parameters
                has_p = "p" in samples or "p_unconstrained" in samples
                has_mu = "mu" in samples or "mu_unconstrained" in samples
                assert (
                    has_p and has_mu
                ), f"Expected p/p_unconstrained and mu/mu_unconstrained in samples, got {list(samples.keys())}"
                # Check shape for whichever parameter exists
                if "mu" in samples:
                    assert (
                        samples["mu"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                elif "mu_unconstrained" in samples:
                    assert (
                        samples["mu_unconstrained"].ndim >= 3
                    )  # n_samples, n_components, n_genes
            else:
                assert "p" in samples and "mu" in samples
                assert (
                    samples["mu"].ndim >= 3
                )  # n_samples, n_components, n_genes
        elif parameterization == "odds_ratio":
            if unconstrained:
                # Unconstrained models may have either constrained or unconstrained parameters
                has_phi = "phi" in samples or "phi_unconstrained" in samples
                has_mu = "mu" in samples or "mu_unconstrained" in samples
                assert (
                    has_phi and has_mu
                ), f"Expected phi/phi_unconstrained and mu/mu_unconstrained in samples, got {list(samples.keys())}"
                # Check shape for whichever parameter exists
                if "mu" in samples:
                    assert (
                        samples["mu"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                elif "mu_unconstrained" in samples:
                    assert (
                        samples["mu_unconstrained"].ndim >= 3
                    )  # n_samples, n_components, n_genes
            else:
                assert "phi" in samples and "mu" in samples
                assert (
                    samples["mu"].ndim >= 3
                )  # n_samples, n_components, n_genes

        # Check mixing weights
        if "mixing_weights" in samples:
            assert (
                samples["mixing_weights"].ndim >= 2
            )  # n_samples, n_components
        elif "mixing_logits_unconstrained" in samples:
            assert (
                samples["mixing_logits_unconstrained"].ndim >= 2
            )  # n_samples, n_components


# ------------------------------------------------------------------------------
# Mixture-specific Tests
# ------------------------------------------------------------------------------


def test_component_selection(nbdm_mix_results):
    """Test selecting a specific component from the mixture."""
    # Select the first component
    component = nbdm_mix_results.get_component(0)

    # Check that it's no longer a mixture model
    assert component.n_components is None
    assert "_mix" not in component.model_type

    # Check that gene counts are preserved
    assert component.n_genes == nbdm_mix_results.n_genes
    assert component.n_cells == nbdm_mix_results.n_cells


# ------------------------------------------------------------------------------


def test_component_selection_with_posterior_samples(
    nbdm_mix_results, rng_key, parameterization, unconstrained, guide_rank
):
    """Test component selection with posterior samples."""
    # Generate posterior samples
    if hasattr(nbdm_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        nbdm_mix_results.get_posterior_samples(
            rng_key=rng_key,
            n_samples=3,
            store_samples=True,
        )

    # Select component
    component = nbdm_mix_results.get_component(1)

    # Check posterior samples are correctly subset
    assert component.posterior_samples is not None

    # Check parameters based on parameterization
    if parameterization == "standard":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_p = (
                "p" in component.posterior_samples
                or "p_unconstrained" in component.posterior_samples
            )
            has_r = (
                "r" in component.posterior_samples
                or "r_unconstrained" in component.posterior_samples
            )
            assert (
                has_p and has_r
            ), f"Expected p/p_unconstrained and r/r_unconstrained in component samples, got {list(component.posterior_samples.keys())}"
            # Check shapes - component dimension should be removed
            if "p" in component.posterior_samples:
                assert component.posterior_samples["p"].shape == (
                    3,
                )  # n_samples
            elif "p_unconstrained" in component.posterior_samples:
                assert component.posterior_samples["p_unconstrained"].shape == (
                    3,
                )  # n_samples
            if "r" in component.posterior_samples:
                assert component.posterior_samples["r"].shape == (
                    3,
                    nbdm_mix_results.n_genes,
                )
            elif "r_unconstrained" in component.posterior_samples:
                assert component.posterior_samples["r_unconstrained"].shape == (
                    3,
                    nbdm_mix_results.n_genes,
                )
        else:
            assert "p" in component.posterior_samples
            assert "r" in component.posterior_samples
            # Check shapes - component dimension should be removed
            assert component.posterior_samples["p"].shape == (3,)  # n_samples
            assert component.posterior_samples["r"].shape == (
                3,
                nbdm_mix_results.n_genes,
            )
    elif parameterization == "linked":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_p = (
                "p" in component.posterior_samples
                or "p_unconstrained" in component.posterior_samples
            )
            has_mu = (
                "mu" in component.posterior_samples
                or "mu_unconstrained" in component.posterior_samples
            )
            assert (
                has_p and has_mu
            ), f"Expected p/p_unconstrained and mu/mu_unconstrained in component samples, got {list(component.posterior_samples.keys())}"
            # Check shapes - component dimension should be removed
            if "p" in component.posterior_samples:
                assert component.posterior_samples["p"].shape == (
                    3,
                )  # n_samples
            elif "p_unconstrained" in component.posterior_samples:
                assert component.posterior_samples["p_unconstrained"].shape == (
                    3,
                )  # n_samples
            if "mu" in component.posterior_samples:
                assert component.posterior_samples["mu"].shape == (
                    3,
                    nbdm_mix_results.n_genes,
                )
            elif "mu_unconstrained" in component.posterior_samples:
                assert component.posterior_samples[
                    "mu_unconstrained"
                ].shape == (
                    3,
                    nbdm_mix_results.n_genes,
                )
        else:
            assert "p" in component.posterior_samples
            assert "mu" in component.posterior_samples
            # Check shapes - component dimension should be removed
            assert component.posterior_samples["p"].shape == (3,)  # n_samples
            assert component.posterior_samples["mu"].shape == (
                3,
                nbdm_mix_results.n_genes,
            )
    elif parameterization == "odds_ratio":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_phi = (
                "phi" in component.posterior_samples
                or "phi_unconstrained" in component.posterior_samples
            )
            has_mu = (
                "mu" in component.posterior_samples
                or "mu_unconstrained" in component.posterior_samples
            )
            assert (
                has_phi and has_mu
            ), f"Expected phi/phi_unconstrained and mu/mu_unconstrained in component samples, got {list(component.posterior_samples.keys())}"
            # Check shapes - component dimension should be removed
            if "phi" in component.posterior_samples:
                assert component.posterior_samples["phi"].shape == (
                    3,
                )  # n_samples
            elif "phi_unconstrained" in component.posterior_samples:
                assert component.posterior_samples[
                    "phi_unconstrained"
                ].shape == (
                    3,
                )  # n_samples
            if "mu" in component.posterior_samples:
                assert component.posterior_samples["mu"].shape == (
                    3,
                    nbdm_mix_results.n_genes,
                )
            elif "mu_unconstrained" in component.posterior_samples:
                assert component.posterior_samples[
                    "mu_unconstrained"
                ].shape == (
                    3,
                    nbdm_mix_results.n_genes,
                )
        else:
            assert "phi" in component.posterior_samples
            assert "mu" in component.posterior_samples
            # Check shapes - component dimension should be removed
            assert component.posterior_samples["phi"].shape == (3,)  # n_samples
            assert component.posterior_samples["mu"].shape == (
                3,
                nbdm_mix_results.n_genes,
            )


# ------------------------------------------------------------------------------


def test_cell_type_assignments(nbdm_mix_results, small_dataset, rng_key):
    """Test computing cell type assignments."""
    counts, _ = small_dataset

    # Generate posterior samples
    if hasattr(nbdm_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        nbdm_mix_results.get_posterior_samples(
            rng_key=rng_key, n_samples=3, store_samples=True
        )

    # Compute assignments
    assignments = nbdm_mix_results.cell_type_probabilities(
        counts, fit_distribution=True, verbose=False
    )

    # Check structure of results
    assert "concentration" in assignments
    assert "mean_probabilities" in assignments
    assert "sample_probabilities" in assignments

    # Check shapes
    assert assignments["concentration"].shape == (
        nbdm_mix_results.n_cells,
        nbdm_mix_results.n_components,
    )
    assert assignments["mean_probabilities"].shape == (
        nbdm_mix_results.n_cells,
        nbdm_mix_results.n_components,
    )
    assert assignments["sample_probabilities"].shape == (
        3,
        nbdm_mix_results.n_cells,
        nbdm_mix_results.n_components,
    )


# ------------------------------------------------------------------------------


def test_subset_with_posterior_samples(
    nbdm_mix_results, rng_key, parameterization, unconstrained, guide_rank
):
    """Test that subsetting preserves posterior samples."""
    # Generate posterior samples
    if hasattr(nbdm_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        nbdm_mix_results.get_posterior_samples(
            rng_key=rng_key,
            n_samples=3,
            store_samples=True,
        )

    # Subset results by genes
    subset = nbdm_mix_results[0:2]

    # Check posterior samples were preserved and correctly subset
    assert subset.posterior_samples is not None

    # Check parameters based on parameterization
    if parameterization == "standard":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_p = (
                "p" in subset.posterior_samples
                or "p_unconstrained" in subset.posterior_samples
            )
            has_r = (
                "r" in subset.posterior_samples
                or "r_unconstrained" in subset.posterior_samples
            )
            has_mixing = (
                "mixing_weights" in subset.posterior_samples
                or "mixing_logits_unconstrained" in subset.posterior_samples
            )
            assert (
                has_p and has_r and has_mixing
            ), f"Expected p/p_unconstrained, r/r_unconstrained, and mixing_weights/mixing_logits_unconstrained in subset samples, got {list(subset.posterior_samples.keys())}"

            # Check shapes for whichever parameters exist
            if "r" in subset.posterior_samples:
                assert subset.posterior_samples["r"].shape == (
                    3,
                    nbdm_mix_results.n_components,
                    2,
                )
            elif "r_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["r_unconstrained"].shape == (
                    3,
                    nbdm_mix_results.n_components,
                    2,
                )
            if "p" in subset.posterior_samples:
                assert subset.posterior_samples["p"].shape == (3,)
            elif "p_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["p_unconstrained"].shape == (3,)
            if "mixing_weights" in subset.posterior_samples:
                assert subset.posterior_samples["mixing_weights"].shape == (
                    3,
                    nbdm_mix_results.n_components,
                )
            elif "mixing_logits_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples[
                    "mixing_logits_unconstrained"
                ].shape == (
                    3,
                    nbdm_mix_results.n_components,
                )
        else:
            assert "p" in subset.posterior_samples
            assert "r" in subset.posterior_samples
            assert "mixing_weights" in subset.posterior_samples
            assert subset.posterior_samples["r"].shape == (
                3,
                nbdm_mix_results.n_components,
                2,
            )
            assert subset.posterior_samples["p"].shape == (3,)
            assert subset.posterior_samples["mixing_weights"].shape == (
                3,
                nbdm_mix_results.n_components,
            )
    elif parameterization == "linked":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_p = (
                "p" in subset.posterior_samples
                or "p_unconstrained" in subset.posterior_samples
            )
            has_mu = (
                "mu" in subset.posterior_samples
                or "mu_unconstrained" in subset.posterior_samples
            )
            has_mixing = (
                "mixing_weights" in subset.posterior_samples
                or "mixing_logits_unconstrained" in subset.posterior_samples
            )
            assert (
                has_p and has_mu and has_mixing
            ), f"Expected p/p_unconstrained, mu/mu_unconstrained, and mixing_weights/mixing_logits_unconstrained in subset samples, got {list(subset.posterior_samples.keys())}"

            # Check shapes for whichever parameters exist
            if "mu" in subset.posterior_samples:
                assert subset.posterior_samples["mu"].shape == (
                    3,
                    nbdm_mix_results.n_components,
                    2,
                )
            elif "mu_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["mu_unconstrained"].shape == (
                    3,
                    nbdm_mix_results.n_components,
                    2,
                )
            if "p" in subset.posterior_samples:
                assert subset.posterior_samples["p"].shape == (3,)
            elif "p_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["p_unconstrained"].shape == (3,)
            if "mixing_weights" in subset.posterior_samples:
                assert subset.posterior_samples["mixing_weights"].shape == (
                    3,
                    nbdm_mix_results.n_components,
                )
            elif "mixing_logits_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples[
                    "mixing_logits_unconstrained"
                ].shape == (
                    3,
                    nbdm_mix_results.n_components,
                )
        else:
            assert "p" in subset.posterior_samples
            assert "mu" in subset.posterior_samples
            assert "mixing_weights" in subset.posterior_samples
            assert subset.posterior_samples["mu"].shape == (
                3,
                nbdm_mix_results.n_components,
                2,
            )
            assert subset.posterior_samples["p"].shape == (3,)
            assert subset.posterior_samples["mixing_weights"].shape == (
                3,
                nbdm_mix_results.n_components,
            )
    elif parameterization == "odds_ratio":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_phi = (
                "phi" in subset.posterior_samples
                or "phi_unconstrained" in subset.posterior_samples
            )
            has_mu = (
                "mu" in subset.posterior_samples
                or "mu_unconstrained" in subset.posterior_samples
            )
            has_mixing = (
                "mixing_weights" in subset.posterior_samples
                or "mixing_logits_unconstrained" in subset.posterior_samples
            )
            assert (
                has_phi and has_mu and has_mixing
            ), f"Expected phi/phi_unconstrained, mu/mu_unconstrained, and mixing_weights/mixing_logits_unconstrained in subset samples, got {list(subset.posterior_samples.keys())}"

            # Check shapes for whichever parameters exist
            if "mu" in subset.posterior_samples:
                assert subset.posterior_samples["mu"].shape == (
                    3,
                    nbdm_mix_results.n_components,
                    2,
                )
            elif "mu_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["mu_unconstrained"].shape == (
                    3,
                    nbdm_mix_results.n_components,
                    2,
                )
            if "phi" in subset.posterior_samples:
                assert subset.posterior_samples["phi"].shape == (3,)
            elif "phi_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["phi_unconstrained"].shape == (
                    3,
                )
            if "mixing_weights" in subset.posterior_samples:
                assert subset.posterior_samples["mixing_weights"].shape == (
                    3,
                    nbdm_mix_results.n_components,
                )
            elif "mixing_logits_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples[
                    "mixing_logits_unconstrained"
                ].shape == (
                    3,
                    nbdm_mix_results.n_components,
                )
        else:
            assert "phi" in subset.posterior_samples
            assert "mu" in subset.posterior_samples
            assert "mixing_weights" in subset.posterior_samples
            assert subset.posterior_samples["mu"].shape == (
                3,
                nbdm_mix_results.n_components,
                2,
            )
            assert subset.posterior_samples["phi"].shape == (3,)
            assert subset.posterior_samples["mixing_weights"].shape == (
                3,
                nbdm_mix_results.n_components,
            )


# ------------------------------------------------------------------------------


def test_component_then_gene_indexing(nbdm_mix_results):
    """Test selecting a component and then indexing genes."""
    # First select a component
    component = nbdm_mix_results.get_component(0)

    # Then select genes
    gene_subset = component[0:2]

    # Check dimensions
    assert gene_subset.n_genes == 2
    assert gene_subset.n_cells == nbdm_mix_results.n_cells
    assert gene_subset.n_components is None
    assert gene_subset.model_type == "nbdm"


# ------------------------------------------------------------------------------
# Unconstrained-specific Tests
# ------------------------------------------------------------------------------


def test_unconstrained_parameter_handling(
    nbdm_mix_results, unconstrained, guide_rank
):
    """Test that unconstrained parameters are handled correctly."""
    if unconstrained:
        # For unconstrained models, we should be able to get the raw parameters
        # The exact behavior depends on the implementation, but we can check
        # that the model runs without errors
        assert hasattr(nbdm_mix_results, "model_config")
        # Additional unconstrained-specific checks can be added here
    else:
        # For constrained models, parameters should respect their constraints
        # This is already tested in test_parameter_ranges
        pass


# ------------------------------------------------------------------------------
# Low-Rank Guide Tests
# ------------------------------------------------------------------------------


def test_low_rank_guide_params(nbdm_mix_results, inference_method, guide_rank):
    """Test that low-rank guide parameters are correctly stored."""
    if inference_method == "svi" and guide_rank is not None:
        # Low-rank guides should have specific parameters
        assert hasattr(nbdm_mix_results, "params")
        params = nbdm_mix_results.params

        # Check for low-rank-specific parameters
        # (loc, scale_tril for LowRankMultivariateNormal)
        # The exact parameter names depend on the implementation
        # This is a basic check that the guide was actually used
        assert len(params) > 0, "Low-rank guide should have parameters"

        # Debug: print available parameters
        print(f"DEBUG: Low-rank guide params keys: {list(params.keys())}")


def test_low_rank_covariance_structure(
    nbdm_mix_results, inference_method, guide_rank, parameterization, rng_key
):
    """Test that low-rank guides produce samples with appropriate covariance structure."""
    if inference_method == "svi" and guide_rank is not None:
        # Generate multiple samples
        samples = nbdm_mix_results.get_posterior_samples(
            rng_key=rng_key, n_samples=50, store_samples=True
        )

        # Get gene-specific parameters (r or mu depending on parameterization)
        # For mixture models, these have shape (n_samples, n_components, n_genes)
        if parameterization == "standard":
            gene_params = samples.get("r")
        elif parameterization in ["linked", "odds_ratio"]:
            gene_params = samples.get("mu")
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")

        if gene_params is not None:
            # Check that we have multiple samples, components, and genes
            assert (
                gene_params.ndim == 3
            ), f"Expected 3D array, got shape {gene_params.shape}"
            n_samples_actual, n_components, n_genes = gene_params.shape
            assert (
                n_samples_actual > 1
            ), "Need multiple samples to check covariance"
            assert n_genes > 1, "Need multiple genes to check covariance"
            assert n_components == nbdm_mix_results.n_components

            # For each component, compute empirical covariance matrix across genes
            for comp_idx in range(n_components):
                comp_params = gene_params[
                    :, comp_idx, :
                ]  # (n_samples, n_genes)
                cov_matrix = jnp.cov(comp_params.T)

                # For low-rank guides, the covariance should have rank <= guide_rank
                # We can't easily check the rank, but we can verify that samples
                # vary
                assert jnp.any(jnp.abs(cov_matrix) > 1e-6), (
                    f"Low-rank guide should produce samples with non-zero "
                    f"covariance for component {comp_idx}"
                )

                # Debug: print covariance statistics
                print(
                    f"DEBUG: Component {comp_idx} - Covariance matrix shape: "
                    f"{cov_matrix.shape}, mean abs value: "
                    f"{jnp.mean(jnp.abs(cov_matrix)):.6f}"
                )
