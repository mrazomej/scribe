"""Tests for visualization dispatch helpers across inference methods."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from scribe.mcmc.results import ScribeMCMCResults
from scribe.models.config import ModelConfig
from viz_utils import (
    _get_config_values,
    _get_predictive_samples_for_plot,
    _get_training_diagnostic_payload,
)


def _make_mcmc_results_for_viz():
    """Create a tiny MCMC results object suitable for viz unit tests."""
    samples = {
        "p": jnp.array([0.3, 0.4, 0.5], dtype=jnp.float32),
        "r": jnp.array(
            [
                [1.0, 1.5],
                [1.1, 1.6],
                [1.2, 1.7],
            ],
            dtype=jnp.float32,
        ),
    }
    config = ModelConfig(base_model="nbdm", n_components=None)
    return ScribeMCMCResults(
        samples=samples,
        n_cells=3,
        n_genes=2,
        model_type="nbdm",
        model_config=config,
        prior_params={},
    )


def test_get_config_values_uses_mcmc_run_size_token():
    """MCMC config metadata should generate samples/warmup filename tokens."""
    cfg = OmegaConf.create(
        {
            "inference": {
                "method": "mcmc",
                "parameterization": "standard",
                "n_components": 1,
                "n_samples": 1200,
                "n_warmup": 300,
            },
            "model": "nbdm",
        }
    )
    results = _make_mcmc_results_for_viz()
    values = _get_config_values(cfg, results=results)
    assert values["run_size_token"] == "1200samples_300warmup"
    assert values["run_size_label"] == "samples"


def test_predictive_helper_truncates_mcmc_draws():
    """MCMC predictive helper should cap draw count for plotting."""
    results = _make_mcmc_results_for_viz()

    # Stub PPC generation to avoid running model code in this unit test.
    full_predictive = np.arange(30, dtype=np.int32).reshape(5, 3, 2)

    def _fake_get_ppc_samples(rng_key=None, batch_size=None, store_samples=True):
        _ = rng_key
        _ = batch_size
        if store_samples:
            results.predictive_samples = full_predictive
        return full_predictive

    results.get_ppc_samples = _fake_get_ppc_samples

    selected = _get_predictive_samples_for_plot(
        results,
        rng_key=None,
        n_samples=2,
        counts=np.zeros((3, 2), dtype=np.int32),
        batch_size=None,
        store_samples=True,
    )
    assert selected.shape == (2, 3, 2)
    assert np.array_equal(selected, full_predictive[:2])


def test_training_payload_includes_mcmc_diagnostics():
    """MCMC diagnostics payload should include trace and divergence info."""
    results = _make_mcmc_results_for_viz()

    # Attach a mock MCMC backend so grouped-chain traces are available.
    mock_mcmc = MagicMock()
    mock_mcmc.get_extra_fields.return_value = {
        "potential_energy": jnp.array([5.0, 4.0, 3.0]),
        "diverging": jnp.array([0, 1, 0], dtype=jnp.int32),
    }
    mock_mcmc.get_samples.return_value = {
        "r": jnp.ones((2, 3, 2), dtype=jnp.float32)
    }
    results._mcmc = mock_mcmc

    payload = _get_training_diagnostic_payload(results)
    assert payload["plot_kind"] == "mcmc_diagnostics"
    assert payload["potential_energy"].shape == (3,)
    assert payload["diverging"].shape == (3,)
    assert payload["trace_by_chain"].shape == (2, 3)
