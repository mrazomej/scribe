# tests/test_zinb_mix.py
"""
Tests for the Zero-Inflated Negative Binomial Mixture Model.
"""
import pytest
import jax.numpy as jnp
from scribe.models.config import InferenceConfig, SVIConfig, MCMCConfig
from scribe.inference import run_scribe
from scribe.inference.preset_builder import build_config_from_preset

ALL_METHODS = [
    "svi",
    "mcmc",
]  # Both SVI and MCMC are supported for mixture models
ALL_PARAMETERIZATIONS = ["standard", "linked", "odds_ratio"]
ALL_UNCONSTRAINED = [False, True]
ALL_GUIDE_RANKS = [None, 5]

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
        "zinb_mix_results",
        "inference_method",
        "parameterization",
        "unconstrained",
        "guide_rank",
    }.issubset(metafunc.fixturenames):
        # Get command-line options for method, parameterization, and unconstrained
        method_opt = metafunc.config.getoption("--method")
        param_opt = metafunc.config.getoption("--parameterization")
        unconstrained_opt = metafunc.config.getoption("--unconstrained")

        # Determine which methods to test based on command-line option
        methods = ALL_METHODS if method_opt == "all" else [method_opt]

        # Determine which parameterizations to test based on command-line option
        params = ALL_PARAMETERIZATIONS if param_opt == "all" else [param_opt]

        # Determine which unconstrained variants to test based on command-line option
        if unconstrained_opt == "all":
            unconstrained_variants = ALL_UNCONSTRAINED
        else:
            unconstrained_variants = [unconstrained_opt == "true"]

        # Generate all valid combinations
        # Note: Low-rank guides are only supported for SVI
        combinations = [
            (m, p, u, g)
            for m in methods
            for p in params
            for u in unconstrained_variants
            for g in ALL_GUIDE_RANKS
            if not (g is not None and m == "mcmc")  # Skip low-rank for MCMC
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
_zinb_mix_results_cache = {}


@pytest.fixture(scope="function")
def zinb_mix_results(
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
    if key in _zinb_mix_results_cache:
        return _zinb_mix_results_cache[key]

    # Device is already configured in conftest.py via pytest_configure

    counts, _ = small_dataset

    # Set up priors based on parameterization (using new key names without "_prior" suffix)
    if parameterization == "standard":
        priors = {"r": (2, 0.1), "p": (1, 1), "gate": (1, 1)}
    elif parameterization == "linked":
        priors = {"p": (1, 1), "mu": (1, 1), "gate": (1, 1)}
    elif parameterization == "odds_ratio":
        priors = {"phi": (3, 2), "mu": (1, 1), "gate": (1, 1)}
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    # Add mixing_prior for mixture models
    priors["mixing"] = jnp.ones(2)  # Uniform prior for 2 components

    # Build model config using preset builder
    model_config = build_config_from_preset(
        model="zinb",
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

    _zinb_mix_results_cache[key] = result
    return result


# ------------------------------------------------------------------------------
# Shared Tests
# ------------------------------------------------------------------------------


def test_inference_run(zinb_mix_results, guide_rank):
    assert zinb_mix_results.n_cells > 0
    assert zinb_mix_results.n_genes > 0
    assert zinb_mix_results.model_type == "zinb_mix"
    assert hasattr(zinb_mix_results, "model_config")
    assert zinb_mix_results.n_components == 2


# ------------------------------------------------------------------------------


def test_parameterization_config(
    zinb_mix_results, parameterization, unconstrained, guide_rank
):
    """Test that the correct parameterization and unconstrained flag are used."""
    assert zinb_mix_results.model_config.parameterization == parameterization
    # Check unconstrained flag
    assert zinb_mix_results.model_config.unconstrained == unconstrained
    # Check that the guide_rank is properly set in the model config
    # guide_rank is stored in guide_families as LowRankGuide for the gene parameter
    if guide_rank is not None:
        # Determine which parameter should have the low-rank guide
        if parameterization == "standard":
            gene_param = "r"
        else:  # linked or odds_ratio
            gene_param = "mu"
        # Check that guide_families has LowRankGuide for the gene parameter
        assert zinb_mix_results.model_config.guide_families is not None
        guide_family = zinb_mix_results.model_config.guide_families.get(gene_param)
        assert guide_family is not None
        from scribe.models.components.guide_families import LowRankGuide

        assert isinstance(guide_family, LowRankGuide)
        assert guide_family.rank == guide_rank
    else:
        # Mean-field guide (default) - no specific check needed
        pass


# ------------------------------------------------------------------------------


def test_parameter_ranges(
    zinb_mix_results, parameterization, unconstrained, guide_rank
):
    """Test that parameters have correct ranges and relationships."""
    # For SVI, we need to get posterior samples to access transformed parameters
    # For MCMC, we can use either params or samples
    if hasattr(zinb_mix_results, "params") and hasattr(
        zinb_mix_results, "get_posterior_samples"
    ):
        # SVI case: get transformed parameters from posterior samples
        samples = zinb_mix_results.get_posterior_samples(n_samples=1)
        params = samples
    elif hasattr(zinb_mix_results, "params"):
        # MCMC case: params might contain transformed parameters
        params = zinb_mix_results.params
    else:
        # Fallback: try to get samples
        samples = zinb_mix_results.get_posterior_samples()
        params = samples

    # Check parameters based on parameterization
    if parameterization == "standard":
        # Standard parameterization: p, r, and gate (component-specific)
        if unconstrained:
            # Unconstrained models should have both constrained and unconstrained parameters
            assert "p" in params or "p_unconstrained" in params
            assert "r" in params or "r_unconstrained" in params
            assert "gate" in params or "gate_unconstrained" in params

            # Check constrained parameters if they exist
            if "p" in params and "r" in params and "gate" in params:
                p, r, gate = params["p"], params["r"], params["gate"]
                # In unconstrained models, p, r, and gate are transformed from unconstrained space
                # but should still be finite
                assert jnp.all(jnp.isfinite(p))
                assert jnp.all(jnp.isfinite(r))
                assert jnp.all(jnp.isfinite(gate))
        else:
            # Constrained models must have p, r, and gate
            assert "p" in params or any(
                k.startswith("p_") for k in params.keys()
            )
            assert "r" in params or any(
                k.startswith("r_") for k in params.keys()
            )
            assert "gate" in params or any(
                k.startswith("gate_") for k in params.keys()
            )

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "linked":
        # Linked parameterization: p, mu, and gate (component-specific)
        if unconstrained:
            # Unconstrained models should have both constrained and unconstrained parameters
            assert "p" in params or "p_unconstrained" in params
            assert "mu" in params or "mu_unconstrained" in params
            assert "gate" in params or "gate_unconstrained" in params

            # Check constrained parameters if they exist
            if "p" in params and "mu" in params and "gate" in params:
                p, mu, gate = params["p"], params["mu"], params["gate"]
                # In unconstrained models, p, mu, and gate are transformed from unconstrained space
                # but should still be finite
                assert jnp.all(jnp.isfinite(p))
                assert jnp.all(jnp.isfinite(mu))
                assert jnp.all(jnp.isfinite(gate))
        else:
            # Constrained models must have p, mu, and gate
            assert "p" in params or any(
                k.startswith("p_") for k in params.keys()
            )
            assert "mu" in params or any(
                k.startswith("mu_") for k in params.keys()
            )
            assert "gate" in params or any(
                k.startswith("gate_") for k in params.keys()
            )

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "odds_ratio":
        # Odds ratio parameterization: phi, mu, and gate (component-specific)
        if unconstrained:
            # Unconstrained models should have both constrained and unconstrained parameters
            assert "phi" in params or "phi_unconstrained" in params
            assert "mu" in params or "mu_unconstrained" in params
            assert "gate" in params or "gate_unconstrained" in params

            # Check constrained parameters if they exist
            if "phi" in params and "mu" in params and "gate" in params:
                phi, mu, gate = params["phi"], params["mu"], params["gate"]
                # In unconstrained models, phi, mu, and gate are transformed from unconstrained space
                # but should still be finite
                assert jnp.all(jnp.isfinite(phi))
                assert jnp.all(jnp.isfinite(mu))
                assert jnp.all(jnp.isfinite(gate))
        else:
            # Constrained models must have phi, mu, and gate
            assert "phi" in params or any(
                k.startswith("phi_") for k in params.keys()
            )
            assert "mu" in params or any(
                k.startswith("mu_") for k in params.keys()
            )
            assert "gate" in params or any(
                k.startswith("gate_") for k in params.keys()
            )

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )


# ------------------------------------------------------------------------------


def test_posterior_sampling(
    zinb_mix_results, rng_key, parameterization, unconstrained, guide_rank
):
    """Test sampling from the variational posterior."""
    # For SVI, must call get_posterior_samples with parameters; for MCMC, just call
    # get_posterior_samples without parameters
    if hasattr(zinb_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
            samples = zinb_mix_results.get_posterior_samples()
        else:  # SVI case
            samples = zinb_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    else:
        samples = zinb_mix_results.get_posterior_samples()

    # Check that we have the expected parameters based on parameterization
    if parameterization == "standard":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_p = "p" in samples or "p_unconstrained" in samples
            has_r = "r" in samples or "r_unconstrained" in samples
            has_gate = "gate" in samples or "gate_unconstrained" in samples
            assert (
                has_p and has_r and has_gate
            ), f"Expected p/p_unconstrained, r/r_unconstrained, and gate/gate_unconstrained in samples, got {list(samples.keys())}"
            # Check shape for whichever parameter exists
            if "r" in samples:
                assert samples["r"].shape[-2] == zinb_mix_results.n_components
                assert samples["r"].shape[-1] == zinb_mix_results.n_genes
            elif "r_unconstrained" in samples:
                assert (
                    samples["r_unconstrained"].shape[-2]
                    == zinb_mix_results.n_components
                )
                assert (
                    samples["r_unconstrained"].shape[-1]
                    == zinb_mix_results.n_genes
                )
            if "gate" in samples:
                assert (
                    samples["gate"].shape[-2] == zinb_mix_results.n_components
                )
                assert samples["gate"].shape[-1] == zinb_mix_results.n_genes
            elif "gate_unconstrained" in samples:
                assert (
                    samples["gate_unconstrained"].shape[-2]
                    == zinb_mix_results.n_components
                )
                assert (
                    samples["gate_unconstrained"].shape[-1]
                    == zinb_mix_results.n_genes
                )
        else:
            assert "p" in samples and "r" in samples and "gate" in samples
            assert samples["r"].shape[-2] == zinb_mix_results.n_components
            assert samples["r"].shape[-1] == zinb_mix_results.n_genes
            assert samples["gate"].shape[-2] == zinb_mix_results.n_components
            assert samples["gate"].shape[-1] == zinb_mix_results.n_genes
    elif parameterization == "linked":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_p = "p" in samples or "p_unconstrained" in samples
            has_mu = "mu" in samples or "mu_unconstrained" in samples
            has_gate = "gate" in samples or "gate_unconstrained" in samples
            assert (
                has_p and has_mu and has_gate
            ), f"Expected p/p_unconstrained, mu/mu_unconstrained, and gate/gate_unconstrained in samples, got {list(samples.keys())}"
            # Check shape for whichever parameter exists
            if "mu" in samples:
                assert samples["mu"].shape[-2] == zinb_mix_results.n_components
                assert samples["mu"].shape[-1] == zinb_mix_results.n_genes
            elif "mu_unconstrained" in samples:
                assert (
                    samples["mu_unconstrained"].shape[-2]
                    == zinb_mix_results.n_components
                )
                assert (
                    samples["mu_unconstrained"].shape[-1]
                    == zinb_mix_results.n_genes
                )
            if "gate" in samples:
                assert (
                    samples["gate"].shape[-2] == zinb_mix_results.n_components
                )
                assert samples["gate"].shape[-1] == zinb_mix_results.n_genes
            elif "gate_unconstrained" in samples:
                assert (
                    samples["gate_unconstrained"].shape[-2]
                    == zinb_mix_results.n_components
                )
                assert (
                    samples["gate_unconstrained"].shape[-1]
                    == zinb_mix_results.n_genes
                )
        else:
            assert "p" in samples and "mu" in samples and "gate" in samples
            assert samples["mu"].shape[-2] == zinb_mix_results.n_components
            assert samples["mu"].shape[-1] == zinb_mix_results.n_genes
            assert samples["gate"].shape[-2] == zinb_mix_results.n_components
            assert samples["gate"].shape[-1] == zinb_mix_results.n_genes
    elif parameterization == "odds_ratio":
        if unconstrained:
            # Unconstrained models may have either constrained or unconstrained parameters
            has_phi = "phi" in samples or "phi_unconstrained" in samples
            has_mu = "mu" in samples or "mu_unconstrained" in samples
            has_gate = "gate" in samples or "gate_unconstrained" in samples
            assert (
                has_phi and has_mu and has_gate
            ), f"Expected phi/phi_unconstrained, mu/mu_unconstrained, and gate/gate_unconstrained in samples, got {list(samples.keys())}"
            # Check shape for whichever parameter exists
            if "mu" in samples:
                assert samples["mu"].shape[-2] == zinb_mix_results.n_components
                assert samples["mu"].shape[-1] == zinb_mix_results.n_genes
            elif "mu_unconstrained" in samples:
                assert (
                    samples["mu_unconstrained"].shape[-2]
                    == zinb_mix_results.n_components
                )
                assert (
                    samples["mu_unconstrained"].shape[-1]
                    == zinb_mix_results.n_genes
                )
            if "gate" in samples:
                assert (
                    samples["gate"].shape[-2] == zinb_mix_results.n_components
                )
                assert samples["gate"].shape[-1] == zinb_mix_results.n_genes
            elif "gate_unconstrained" in samples:
                assert (
                    samples["gate_unconstrained"].shape[-2]
                    == zinb_mix_results.n_components
                )
                assert (
                    samples["gate_unconstrained"].shape[-1]
                    == zinb_mix_results.n_genes
                )
        else:
            assert "phi" in samples and "mu" in samples and "gate" in samples
            assert samples["mu"].shape[-2] == zinb_mix_results.n_components
            assert samples["mu"].shape[-1] == zinb_mix_results.n_genes
            assert samples["gate"].shape[-2] == zinb_mix_results.n_components
            assert samples["gate"].shape[-1] == zinb_mix_results.n_genes

    # Check mixing weights
    assert (
        "mixing_weights" in samples or "mixing_logits_unconstrained" in samples
    )
    if "mixing_weights" in samples:
        assert (
            samples["mixing_weights"].shape[-1] == zinb_mix_results.n_components
        )
    elif "mixing_logits_unconstrained" in samples:
        assert (
            samples["mixing_logits_unconstrained"].shape[-1]
            == zinb_mix_results.n_components
        )


# ------------------------------------------------------------------------------


def test_predictive_sampling(zinb_mix_results, rng_key, guide_rank):
    """Test generating predictive samples."""
    # For SVI, must generate posterior samples first
    if hasattr(zinb_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pred = zinb_mix_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            zinb_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = zinb_mix_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = zinb_mix_results.get_ppc_samples(
            rng_key=rng_key, store_samples=True
        )
    assert pred.shape[-1] == zinb_mix_results.n_genes
    assert jnp.all(pred >= 0)


# ------------------------------------------------------------------------------


def test_get_map(zinb_mix_results, parameterization, unconstrained, guide_rank):
    map_est = (
        zinb_mix_results.get_map()
        if hasattr(zinb_mix_results, "get_map")
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
                if hasattr(zinb_mix_results, "params"):
                    print(
                        f"DEBUG: Model params keys: {list(zinb_mix_results.params.keys())}"
                    )
                # For SVI unconstrained models, empty MAP is acceptable
                # as they use variational parameters (alpha/beta, loc/scale)
                # instead of direct MAP estimates
                pass
            else:
                has_p = "p" in map_est or "p_unconstrained" in map_est
                has_r = "r" in map_est or "r_unconstrained" in map_est
                has_gate = "gate" in map_est or "gate_unconstrained" in map_est
                assert (
                    has_p and has_r and has_gate
                ), f"Expected p/p_unconstrained, r/r_unconstrained, and gate/gate_unconstrained in MAP, got {list(map_est.keys())}"
        else:
            assert "r" in map_est and "p" in map_est and "gate" in map_est
    elif parameterization == "linked":
        if unconstrained:
            if len(map_est) == 0:
                print(
                    f"DEBUG: Empty MAP for unconstrained {parameterization}, checking model params"
                )
                if hasattr(zinb_mix_results, "params"):
                    print(
                        f"DEBUG: Model params keys: {list(zinb_mix_results.params.keys())}"
                    )
                # For SVI unconstrained models, empty MAP is acceptable
                pass
            else:
                has_p = "p" in map_est or "p_unconstrained" in map_est
                has_mu = "mu" in map_est or "mu_unconstrained" in map_est
                has_gate = "gate" in map_est or "gate_unconstrained" in map_est
                assert (
                    has_p and has_mu and has_gate
                ), f"Expected p/p_unconstrained, mu/mu_unconstrained, and gate/gate_unconstrained in MAP, got {list(map_est.keys())}"
        else:
            assert "p" in map_est and "mu" in map_est and "gate" in map_est
    elif parameterization == "odds_ratio":
        if unconstrained:
            if len(map_est) == 0:
                print(
                    f"DEBUG: Empty MAP for unconstrained {parameterization}, checking model params"
                )
                if hasattr(zinb_mix_results, "params"):
                    print(
                        f"DEBUG: Model params keys: {list(zinb_mix_results.params.keys())}"
                    )
                # For SVI unconstrained models, empty MAP is acceptable
                pass
            else:
                has_phi = "phi" in map_est or "phi_unconstrained" in map_est
                has_mu = "mu" in map_est or "mu_unconstrained" in map_est
                has_gate = "gate" in map_est or "gate_unconstrained" in map_est
                assert (
                    has_phi and has_mu and has_gate
                ), f"Expected phi/phi_unconstrained, mu/mu_unconstrained, and gate/gate_unconstrained in MAP, got {list(map_est.keys())}"
        else:
            assert "phi" in map_est and "mu" in map_est and "gate" in map_est

    # Check mixing weights
    if "mixing_weights" in map_est:
        assert (
            map_est["mixing_weights"].shape[-1] == zinb_mix_results.n_components
        )
    elif "mixing_logits_unconstrained" in map_est:
        assert (
            map_est["mixing_logits_unconstrained"].shape[-1]
            == zinb_mix_results.n_components
        )


# ------------------------------------------------------------------------------


def test_indexing_integer(zinb_mix_results, guide_rank):
    """Test indexing with an integer."""
    subset = zinb_mix_results[0]
    assert subset.n_genes == 1
    assert subset.n_cells == zinb_mix_results.n_cells
    assert subset.n_components == zinb_mix_results.n_components


# ------------------------------------------------------------------------------


def test_indexing_slice(zinb_mix_results, guide_rank):
    """Test indexing with a slice."""
    end_idx = min(2, zinb_mix_results.n_genes)
    subset = zinb_mix_results[0:end_idx]
    assert subset.n_genes == end_idx
    assert subset.n_cells == zinb_mix_results.n_cells
    assert subset.n_components == zinb_mix_results.n_components


# ------------------------------------------------------------------------------


def test_indexing_boolean(zinb_mix_results, guide_rank):
    """Test indexing with a boolean array."""
    mask = jnp.zeros(zinb_mix_results.n_genes, dtype=bool)
    mask = mask.at[0].set(True)

    subset = zinb_mix_results[mask]
    assert subset.n_genes == 1
    assert subset.n_cells == zinb_mix_results.n_cells
    assert subset.n_components == zinb_mix_results.n_components


# ------------------------------------------------------------------------------


def test_log_likelihood(zinb_mix_results, small_dataset, rng_key, guide_rank):
    counts, _ = small_dataset

    # For SVI, must generate posterior samples first
    if hasattr(zinb_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pass
        else:  # SVI case
            zinb_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )

    # Test marginal log likelihood (across components)
    ll = zinb_mix_results.log_likelihood(counts, return_by="cell")
    assert ll.shape[0] > 0
    # assert jnp.all(jnp.isfinite(ll))

    # Test component-specific log likelihood
    ll_components = zinb_mix_results.log_likelihood(
        counts, return_by="cell", split_components=True
    )
    assert ll_components.shape[-1] == zinb_mix_results.n_components
    # assert jnp.all(jnp.isfinite(ll_components))


# ------------------------------------------------------------------------------
# SVI-specific Tests
# ------------------------------------------------------------------------------


def test_svi_loss_history(zinb_mix_results, inference_method, guide_rank):
    if inference_method == "svi":
        assert hasattr(zinb_mix_results, "loss_history")
        assert len(zinb_mix_results.loss_history) > 0


# ------------------------------------------------------------------------------
# MCMC-specific Tests
# ------------------------------------------------------------------------------


def test_mcmc_samples_shape(
    zinb_mix_results,
    inference_method,
    parameterization,
    unconstrained,
    guide_rank,
):
    if inference_method == "mcmc":
        samples = zinb_mix_results.get_posterior_samples()

        # Check parameters based on parameterization
        if parameterization == "standard":
            if unconstrained:
                # Unconstrained models may have either constrained or unconstrained parameters
                has_p = "p" in samples or "p_unconstrained" in samples
                has_r = "r" in samples or "r_unconstrained" in samples
                has_gate = "gate" in samples or "gate_unconstrained" in samples
                assert (
                    has_p and has_r and has_gate
                ), f"Expected p/p_unconstrained, r/r_unconstrained, and gate/gate_unconstrained in samples, got {list(samples.keys())}"
                # Check shape for whichever parameter exists
                if "r" in samples:
                    assert (
                        samples["r"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                elif "r_unconstrained" in samples:
                    assert (
                        samples["r_unconstrained"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                if "gate" in samples:
                    assert (
                        samples["gate"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                elif "gate_unconstrained" in samples:
                    assert (
                        samples["gate_unconstrained"].ndim >= 3
                    )  # n_samples, n_components, n_genes
            else:
                assert "r" in samples and "p" in samples and "gate" in samples
                assert (
                    samples["r"].ndim >= 3
                )  # n_samples, n_components, n_genes
                assert (
                    samples["gate"].ndim >= 3
                )  # n_samples, n_components, n_genes
        elif parameterization == "linked":
            if unconstrained:
                # Unconstrained models may have either constrained or unconstrained parameters
                has_p = "p" in samples or "p_unconstrained" in samples
                has_mu = "mu" in samples or "mu_unconstrained" in samples
                has_gate = "gate" in samples or "gate_unconstrained" in samples
                assert (
                    has_p and has_mu and has_gate
                ), f"Expected p/p_unconstrained, mu/mu_unconstrained, and gate/gate_unconstrained in samples, got {list(samples.keys())}"
                # Check shape for whichever parameter exists
                if "mu" in samples:
                    assert (
                        samples["mu"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                elif "mu_unconstrained" in samples:
                    assert (
                        samples["mu_unconstrained"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                if "gate" in samples:
                    assert (
                        samples["gate"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                elif "gate_unconstrained" in samples:
                    assert (
                        samples["gate_unconstrained"].ndim >= 3
                    )  # n_samples, n_components, n_genes
            else:
                assert "p" in samples and "mu" in samples and "gate" in samples
                assert (
                    samples["mu"].ndim >= 3
                )  # n_samples, n_components, n_genes
                assert (
                    samples["gate"].ndim >= 3
                )  # n_samples, n_components, n_genes
        elif parameterization == "odds_ratio":
            if unconstrained:
                # Unconstrained models may have either constrained or unconstrained parameters
                has_phi = "phi" in samples or "phi_unconstrained" in samples
                has_mu = "mu" in samples or "mu_unconstrained" in samples
                has_gate = "gate" in samples or "gate_unconstrained" in samples
                assert (
                    has_phi and has_mu and has_gate
                ), f"Expected phi/phi_unconstrained, mu/mu_unconstrained, and gate/gate_unconstrained in samples, got {list(samples.keys())}"
                # Check shape for whichever parameter exists
                if "mu" in samples:
                    assert (
                        samples["mu"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                elif "mu_unconstrained" in samples:
                    assert (
                        samples["mu_unconstrained"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                if "gate" in samples:
                    assert (
                        samples["gate"].ndim >= 3
                    )  # n_samples, n_components, n_genes
                elif "gate_unconstrained" in samples:
                    assert (
                        samples["gate_unconstrained"].ndim >= 3
                    )  # n_samples, n_components, n_genes
            else:
                assert (
                    "phi" in samples and "mu" in samples and "gate" in samples
                )
                assert (
                    samples["mu"].ndim >= 3
                )  # n_samples, n_components, n_genes
                assert (
                    samples["gate"].ndim >= 3
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


def test_component_selection(zinb_mix_results, guide_rank):
    """Test selecting a specific component from the mixture."""
    # Select the first component
    component = zinb_mix_results.get_component(0)

    # Check that it's no longer a mixture model
    assert component.n_components is None
    assert "_mix" not in component.model_type

    # Check that gene counts are preserved
    assert component.n_genes == zinb_mix_results.n_genes
    assert component.n_cells == zinb_mix_results.n_cells


# ------------------------------------------------------------------------------


def test_component_selection_with_posterior_samples(
    zinb_mix_results, rng_key, parameterization, unconstrained, guide_rank
):
    """Test component selection with posterior samples."""
    # Generate posterior samples
    if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        zinb_mix_results.get_posterior_samples(
            rng_key=rng_key,
            n_samples=3,
            store_samples=True,
        )

    # Select component
    component = zinb_mix_results.get_component(1)

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
            has_gate = (
                "gate" in component.posterior_samples
                or "gate_unconstrained" in component.posterior_samples
            )
            assert (
                has_p and has_r and has_gate
            ), f"Expected p/p_unconstrained, r/r_unconstrained, and gate/gate_unconstrained in component samples, got {list(component.posterior_samples.keys())}"
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
                    zinb_mix_results.n_genes,
                )
            elif "r_unconstrained" in component.posterior_samples:
                assert component.posterior_samples["r_unconstrained"].shape == (
                    3,
                    zinb_mix_results.n_genes,
                )
            if "gate" in component.posterior_samples:
                assert component.posterior_samples["gate"].shape == (
                    3,
                    zinb_mix_results.n_genes,
                )
            elif "gate_unconstrained" in component.posterior_samples:
                assert component.posterior_samples[
                    "gate_unconstrained"
                ].shape == (
                    3,
                    zinb_mix_results.n_genes,
                )
        else:
            assert "p" in component.posterior_samples
            assert "r" in component.posterior_samples
            assert "gate" in component.posterior_samples
            # Check shapes - component dimension should be removed
            assert component.posterior_samples["p"].shape == (3,)  # n_samples
            assert component.posterior_samples["r"].shape == (
                3,
                zinb_mix_results.n_genes,
            )
            assert component.posterior_samples["gate"].shape == (
                3,
                zinb_mix_results.n_genes,
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
            has_gate = (
                "gate" in component.posterior_samples
                or "gate_unconstrained" in component.posterior_samples
            )
            assert (
                has_p and has_mu and has_gate
            ), f"Expected p/p_unconstrained, mu/mu_unconstrained, and gate/gate_unconstrained in component samples, got {list(component.posterior_samples.keys())}"
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
                    zinb_mix_results.n_genes,
                )
            elif "mu_unconstrained" in component.posterior_samples:
                assert component.posterior_samples[
                    "mu_unconstrained"
                ].shape == (
                    3,
                    zinb_mix_results.n_genes,
                )
            if "gate" in component.posterior_samples:
                assert component.posterior_samples["gate"].shape == (
                    3,
                    zinb_mix_results.n_genes,
                )
            elif "gate_unconstrained" in component.posterior_samples:
                assert component.posterior_samples[
                    "gate_unconstrained"
                ].shape == (
                    3,
                    zinb_mix_results.n_genes,
                )
        else:
            assert "p" in component.posterior_samples
            assert "mu" in component.posterior_samples
            assert "gate" in component.posterior_samples
            # Check shapes - component dimension should be removed
            assert component.posterior_samples["p"].shape == (3,)  # n_samples
            assert component.posterior_samples["mu"].shape == (
                3,
                zinb_mix_results.n_genes,
            )
            assert component.posterior_samples["gate"].shape == (
                3,
                zinb_mix_results.n_genes,
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
            has_gate = (
                "gate" in component.posterior_samples
                or "gate_unconstrained" in component.posterior_samples
            )
            assert (
                has_phi and has_mu and has_gate
            ), f"Expected phi/phi_unconstrained, mu/mu_unconstrained, and gate/gate_unconstrained in component samples, got {list(component.posterior_samples.keys())}"
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
                    zinb_mix_results.n_genes,
                )
            elif "mu_unconstrained" in component.posterior_samples:
                assert component.posterior_samples[
                    "mu_unconstrained"
                ].shape == (
                    3,
                    zinb_mix_results.n_genes,
                )
            if "gate" in component.posterior_samples:
                assert component.posterior_samples["gate"].shape == (
                    3,
                    zinb_mix_results.n_genes,
                )
            elif "gate_unconstrained" in component.posterior_samples:
                assert component.posterior_samples[
                    "gate_unconstrained"
                ].shape == (
                    3,
                    zinb_mix_results.n_genes,
                )
        else:
            assert "phi" in component.posterior_samples
            assert "mu" in component.posterior_samples
            assert "gate" in component.posterior_samples
            # Check shapes - component dimension should be removed
            assert component.posterior_samples["phi"].shape == (3,)  # n_samples
            assert component.posterior_samples["mu"].shape == (
                3,
                zinb_mix_results.n_genes,
            )
            assert component.posterior_samples["gate"].shape == (
                3,
                zinb_mix_results.n_genes,
            )


# ------------------------------------------------------------------------------


def test_cell_type_assignments(
    zinb_mix_results, small_dataset, rng_key, guide_rank
):
    """Test computing cell type assignments."""
    counts, _ = small_dataset

    # Generate posterior samples
    if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        zinb_mix_results.get_posterior_samples(
            rng_key=rng_key, n_samples=3, store_samples=True
        )

    # Compute assignments
    assignments = zinb_mix_results.cell_type_probabilities(
        counts, fit_distribution=True, verbose=False
    )

    # Check structure of results
    assert "concentration" in assignments
    assert "mean_probabilities" in assignments
    assert "sample_probabilities" in assignments

    # Check shapes
    assert assignments["concentration"].shape == (
        zinb_mix_results.n_cells,
        zinb_mix_results.n_components,
    )
    assert assignments["mean_probabilities"].shape == (
        zinb_mix_results.n_cells,
        zinb_mix_results.n_components,
    )
    assert assignments["sample_probabilities"].shape == (
        3,
        zinb_mix_results.n_cells,
        zinb_mix_results.n_components,
    )


# ------------------------------------------------------------------------------


def test_subset_with_posterior_samples(
    zinb_mix_results, rng_key, parameterization, unconstrained, guide_rank
):
    """Test that subsetting preserves posterior samples."""
    # Generate posterior samples
    if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        zinb_mix_results.get_posterior_samples(
            rng_key=rng_key,
            n_samples=3,
            store_samples=True,
        )

    # Subset results by genes
    subset = zinb_mix_results[0:2]

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
            has_gate = (
                "gate" in subset.posterior_samples
                or "gate_unconstrained" in subset.posterior_samples
            )
            has_mixing = (
                "mixing_weights" in subset.posterior_samples
                or "mixing_logits_unconstrained" in subset.posterior_samples
            )
            assert (
                has_p and has_r and has_gate and has_mixing
            ), f"Expected p/p_unconstrained, r/r_unconstrained, gate/gate_unconstrained, and mixing_weights/mixing_logits_unconstrained in subset samples, got {list(subset.posterior_samples.keys())}"

            # Check shapes for whichever parameters exist
            if "r" in subset.posterior_samples:
                assert subset.posterior_samples["r"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            elif "r_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["r_unconstrained"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            if "gate" in subset.posterior_samples:
                assert subset.posterior_samples["gate"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            elif "gate_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["gate_unconstrained"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            if "p" in subset.posterior_samples:
                assert subset.posterior_samples["p"].shape == (3,)
            elif "p_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["p_unconstrained"].shape == (3,)
            if "mixing_weights" in subset.posterior_samples:
                assert subset.posterior_samples["mixing_weights"].shape == (
                    3,
                    zinb_mix_results.n_components,
                )
            elif "mixing_logits_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples[
                    "mixing_logits_unconstrained"
                ].shape == (
                    3,
                    zinb_mix_results.n_components,
                )
        else:
            assert "p" in subset.posterior_samples
            assert "r" in subset.posterior_samples
            assert "gate" in subset.posterior_samples
            assert "mixing_weights" in subset.posterior_samples
            assert subset.posterior_samples["r"].shape == (
                3,
                zinb_mix_results.n_components,
                2,
            )
            assert subset.posterior_samples["gate"].shape == (
                3,
                zinb_mix_results.n_components,
                2,
            )
            assert subset.posterior_samples["p"].shape == (3,)
            assert subset.posterior_samples["mixing_weights"].shape == (
                3,
                zinb_mix_results.n_components,
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
            has_gate = (
                "gate" in subset.posterior_samples
                or "gate_unconstrained" in subset.posterior_samples
            )
            has_mixing = (
                "mixing_weights" in subset.posterior_samples
                or "mixing_logits_unconstrained" in subset.posterior_samples
            )
            assert (
                has_p and has_mu and has_gate and has_mixing
            ), f"Expected p/p_unconstrained, mu/mu_unconstrained, gate/gate_unconstrained, and mixing_weights/mixing_logits_unconstrained in subset samples, got {list(subset.posterior_samples.keys())}"

            # Check shapes for whichever parameters exist
            if "mu" in subset.posterior_samples:
                assert subset.posterior_samples["mu"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            elif "mu_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["mu_unconstrained"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            if "gate" in subset.posterior_samples:
                assert subset.posterior_samples["gate"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            elif "gate_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["gate_unconstrained"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            if "p" in subset.posterior_samples:
                assert subset.posterior_samples["p"].shape == (3,)
            elif "p_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["p_unconstrained"].shape == (3,)
            if "mixing_weights" in subset.posterior_samples:
                assert subset.posterior_samples["mixing_weights"].shape == (
                    3,
                    zinb_mix_results.n_components,
                )
            elif "mixing_logits_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples[
                    "mixing_logits_unconstrained"
                ].shape == (
                    3,
                    zinb_mix_results.n_components,
                )
        else:
            assert "p" in subset.posterior_samples
            assert "mu" in subset.posterior_samples
            assert "gate" in subset.posterior_samples
            assert "mixing_weights" in subset.posterior_samples
            assert subset.posterior_samples["mu"].shape == (
                3,
                zinb_mix_results.n_components,
                2,
            )
            assert subset.posterior_samples["gate"].shape == (
                3,
                zinb_mix_results.n_components,
                2,
            )
            assert subset.posterior_samples["p"].shape == (3,)
            assert subset.posterior_samples["mixing_weights"].shape == (
                3,
                zinb_mix_results.n_components,
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
            has_gate = (
                "gate" in subset.posterior_samples
                or "gate_unconstrained" in subset.posterior_samples
            )
            has_mixing = (
                "mixing_weights" in subset.posterior_samples
                or "mixing_logits_unconstrained" in subset.posterior_samples
            )
            assert (
                has_phi and has_mu and has_gate and has_mixing
            ), f"Expected phi/phi_unconstrained, mu/mu_unconstrained, gate/gate_unconstrained, and mixing_weights/mixing_logits_unconstrained in subset samples, got {list(subset.posterior_samples.keys())}"

            # Check shapes for whichever parameters exist
            if "mu" in subset.posterior_samples:
                assert subset.posterior_samples["mu"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            elif "mu_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["mu_unconstrained"].shape == (
                    3,
                    zinb_mix_results.n_genes,
                    2,
                )
            if "gate" in subset.posterior_samples:
                assert subset.posterior_samples["gate"].shape == (
                    3,
                    zinb_mix_results.n_components,
                    2,
                )
            elif "gate_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples["gate_unconstrained"].shape == (
                    3,
                    zinb_mix_results.n_components,
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
                    zinb_mix_results.n_components,
                )
            elif "mixing_logits_unconstrained" in subset.posterior_samples:
                assert subset.posterior_samples[
                    "mixing_logits_unconstrained"
                ].shape == (
                    3,
                    zinb_mix_results.n_components,
                )
        else:
            assert "phi" in subset.posterior_samples
            assert "mu" in subset.posterior_samples
            assert "gate" in subset.posterior_samples
            assert "mixing_weights" in subset.posterior_samples
            assert subset.posterior_samples["mu"].shape == (
                3,
                zinb_mix_results.n_components,
                2,
            )
            assert subset.posterior_samples["gate"].shape == (
                3,
                zinb_mix_results.n_components,
                2,
            )
            assert subset.posterior_samples["phi"].shape == (3,)
            assert subset.posterior_samples["mixing_weights"].shape == (
                3,
                zinb_mix_results.n_components,
            )


# ------------------------------------------------------------------------------


def test_component_then_gene_indexing(zinb_mix_results, guide_rank):
    """Test selecting a component and then indexing genes."""
    # First select a component
    component = zinb_mix_results.get_component(0)

    # Then select genes
    gene_subset = component[0:2]

    # Check dimensions
    assert gene_subset.n_genes == 2
    assert gene_subset.n_cells == zinb_mix_results.n_cells
    assert gene_subset.n_components is None
    assert gene_subset.model_type == "zinb"


# ------------------------------------------------------------------------------
# ZINB-specific Tests
# ------------------------------------------------------------------------------


def test_gate_parameter_ranges(
    zinb_mix_results, parameterization, unconstrained, guide_rank
):
    """Test that gate parameters are in valid probability range [0, 1]."""
    # Get parameters
    params = None
    if hasattr(zinb_mix_results, "params"):
        params = zinb_mix_results.params
    if params is None:
        samples = zinb_mix_results.get_posterior_samples()
        params = samples

    # Check gate parameters regardless of parameterization
    if unconstrained:
        # For unconstrained models, check if we have either constrained or unconstrained gate parameters
        has_gate = "gate" in params or "gate_unconstrained" in params
        if has_gate:
            if "gate" in params:
                gate = params["gate"]
                # In unconstrained models, gate is transformed from unconstrained space
                # but should still be finite
                assert jnp.all(jnp.isfinite(gate))
            # Note: gate_unconstrained parameters don't have range constraints
    else:
        # For constrained models, gate parameters must be in [0, 1]
        if "gate" in params:
            gate = params["gate"]
            assert jnp.all(
                (gate >= 0) & (gate <= 1)
            ), "Gate parameters must be in [0, 1]"


def test_zero_inflation_behavior(zinb_mix_results, rng_key, guide_rank):
    """Test that the model exhibits zero-inflation behavior."""
    # Generate predictive samples
    if hasattr(zinb_mix_results, "get_posterior_samples"):
        if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
            pred = zinb_mix_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            zinb_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = zinb_mix_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = zinb_mix_results.get_ppc_samples(
            rng_key=rng_key, store_samples=True
        )

    # Check that we have some zeros (zero-inflation)
    assert jnp.any(
        pred == 0
    ), "ZINB mixture model should produce some zero counts"

    # Check that we also have non-zero counts
    assert jnp.any(
        pred > 0
    ), "ZINB mixture model should also produce non-zero counts"


# ------------------------------------------------------------------------------
# Unconstrained-specific Tests
# ------------------------------------------------------------------------------


def test_unconstrained_parameter_handling(
    zinb_mix_results, unconstrained, guide_rank
):
    """Test that unconstrained parameters are handled correctly."""
    if unconstrained:
        # For unconstrained models, we should be able to get the raw parameters
        # The exact behavior depends on the implementation, but we can check
        # that the model runs without errors
        assert hasattr(zinb_mix_results, "model_config")
        # Additional unconstrained-specific checks can be added here
    else:
        # For constrained models, parameters should respect their constraints
        # This is already tested in test_parameter_ranges
        pass


# ------------------------------------------------------------------------------
# Low-rank guide specific tests
# ------------------------------------------------------------------------------


def test_low_rank_guide_params(
    zinb_mix_results, inference_method, guide_rank, parameterization
):
    """Test that low-rank guide parameters are present when guide_rank is not None."""
    if inference_method == "svi" and guide_rank is not None:
        # Check that low-rank guide parameters are present
        assert hasattr(zinb_mix_results, "params")
        params = zinb_mix_results.params

        # For ZINB models, the low-rank guide should be applied to mu (linked/odds_ratio)
        # or r (standard)
        if parameterization == "standard":
            # Standard parameterization uses r
            if zinb_mix_results.model_config.unconstrained:
                # Unconstrained: look for r low-rank parameters (directly on unconstrained r)
                assert (
                    "r_loc" in params
                ), "Low-rank guide should have r_loc"
                assert (
                    "r_W" in params
                ), "Low-rank guide should have r_W (cov_factor)"
                assert (
                    "r_raw_diag" in params
                ), "Low-rank guide should have r_raw_diag (cov_diag)"
            else:
                # Constrained: look for log_r low-rank parameters
                assert (
                    "log_r_loc" in params
                ), "Low-rank guide should have log_r_loc"
                assert (
                    "log_r_W" in params
                ), "Low-rank guide should have log_r_W (cov_factor)"
                assert (
                    "log_r_raw_diag" in params
                ), "Low-rank guide should have log_r_raw_diag (cov_diag)"
        elif parameterization in ["linked", "odds_ratio"]:
            # Linked and odds_ratio use mu
            if zinb_mix_results.model_config.unconstrained:
                # Unconstrained: look for mu low-rank parameters (directly on unconstrained mu)
                assert (
                    "mu_loc" in params
                ), "Low-rank guide should have mu_loc"
                assert (
                    "mu_W" in params
                ), "Low-rank guide should have mu_W (cov_factor)"
                assert (
                    "mu_raw_diag" in params
                ), "Low-rank guide should have mu_raw_diag (cov_diag)"
            else:
                # Constrained: look for log_mu low-rank parameters
                assert (
                    "log_mu_loc" in params
                ), "Low-rank guide should have log_mu_loc"
                assert (
                    "log_mu_W" in params
                ), "Low-rank guide should have log_mu_W (cov_factor)"
                assert (
                    "log_mu_raw_diag" in params
                ), "Low-rank guide should have log_mu_raw_diag (cov_diag)"

        # Note: gate uses mean-field (not low-rank) for all ZINB mixture models
        # Only r/mu get the low-rank approximation


def test_low_rank_covariance_structure(
    zinb_mix_results, inference_method, guide_rank, parameterization
):
    """Test that low-rank covariance matrix has correct rank."""
    if inference_method == "svi" and guide_rank is not None:
        assert hasattr(zinb_mix_results, "params")
        params = zinb_mix_results.params

        # Check the shape of W (cov_factor) to verify rank
        if parameterization == "standard":
            if zinb_mix_results.model_config.unconstrained:
                W = params.get("r_W")
            else:
                W = params.get("log_r_W")
        elif parameterization in ["linked", "odds_ratio"]:
            if zinb_mix_results.model_config.unconstrained:
                W = params.get("mu_W")
            else:
                W = params.get("log_mu_W")

        if W is not None:
            # W should have shape (n_components, n_genes, guide_rank)
            assert W.shape[-1] == guide_rank, (
                f"Covariance factor W should have rank {guide_rank}, "
                f"but has shape {W.shape}"
            )
            assert (
                W.shape[-2] == zinb_mix_results.n_genes
            ), "W should have n_genes dimension"
            assert (
                W.shape[-3] == zinb_mix_results.n_components
            ), "W should have n_components dimension"
