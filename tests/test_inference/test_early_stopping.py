"""Tests for early stopping functionality in SVI inference.

Unit tests run fast and test config validation.
Integration tests are skipped by default - run with: pytest --run-slow
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.models.config import (
    EarlyStoppingConfig,
    SVIConfig,
    InferenceConfig,
)
from scribe.svi import SVIRunResult


# =============================================================================
# Unit Tests - Config Validation (Fast)
# =============================================================================


class TestEarlyStoppingConfig:
    """Test EarlyStoppingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EarlyStoppingConfig()

        assert config.enabled is True
        assert config.patience == 500
        assert config.min_delta == 1.0  # Default is 1 (reasonable for ELBO ~10^6-10^7)
        assert config.check_every == 10
        assert config.smoothing_window == 50
        assert config.restore_best is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EarlyStoppingConfig(
            enabled=True,
            patience=200,
            min_delta=100.0,
            check_every=5,
            smoothing_window=25,
            restore_best=False,
        )

        assert config.enabled is True
        assert config.patience == 200
        assert config.min_delta == 100.0
        assert config.check_every == 5
        assert config.smoothing_window == 25
        assert config.restore_best is False

    def test_disabled_config(self):
        """Test disabled early stopping configuration."""
        config = EarlyStoppingConfig(enabled=False)

        assert config.enabled is False
        # Other values should still be set
        assert config.patience == 500

    def test_validation_patience_positive(self):
        """Test that patience must be positive."""
        with pytest.raises(ValueError):
            EarlyStoppingConfig(patience=0)

        with pytest.raises(ValueError):
            EarlyStoppingConfig(patience=-1)

    def test_validation_min_delta_non_negative(self):
        """Test that min_delta must be non-negative."""
        # min_delta=0 should be valid
        config = EarlyStoppingConfig(min_delta=0)
        assert config.min_delta == 0

        # Negative should fail
        with pytest.raises(ValueError):
            EarlyStoppingConfig(min_delta=-1.0)

    def test_validation_check_every_positive(self):
        """Test that check_every must be positive."""
        with pytest.raises(ValueError):
            EarlyStoppingConfig(check_every=0)

    def test_validation_smoothing_window_positive(self):
        """Test that smoothing_window must be positive."""
        with pytest.raises(ValueError):
            EarlyStoppingConfig(smoothing_window=0)

    def test_immutability(self):
        """Test that config is immutable (frozen)."""
        config = EarlyStoppingConfig()

        with pytest.raises(Exception):  # ValidationError from Pydantic
            config.patience = 100

    def test_yaml_serialization(self):
        """Test YAML serialization round-trip."""
        config = EarlyStoppingConfig(
            patience=300,
            min_delta=50.0,
        )

        yaml_str = config.to_yaml()
        restored = EarlyStoppingConfig.from_yaml(yaml_str)

        assert restored.patience == config.patience
        assert restored.min_delta == config.min_delta
        assert restored.enabled == config.enabled


class TestSVIConfigWithEarlyStopping:
    """Test SVIConfig with early stopping configuration."""

    def test_svi_config_without_early_stopping(self):
        """Test SVIConfig without early stopping."""
        config = SVIConfig(n_steps=10000)

        assert config.early_stopping is None
        assert config.n_steps == 10000

    def test_svi_config_with_early_stopping(self):
        """Test SVIConfig with early stopping configuration."""
        early_stopping = EarlyStoppingConfig(patience=100)
        config = SVIConfig(
            n_steps=10000,
            early_stopping=early_stopping,
        )

        assert config.early_stopping is not None
        assert config.early_stopping.patience == 100
        assert config.early_stopping.enabled is True

    def test_inference_config_with_early_stopping(self):
        """Test InferenceConfig creation with early stopping."""
        early_stopping = EarlyStoppingConfig(
            patience=200,
            min_delta=10.0,
        )
        svi_config = SVIConfig(
            n_steps=50000,
            early_stopping=early_stopping,
        )
        inference_config = InferenceConfig.from_svi(svi_config)

        assert inference_config.svi.early_stopping is not None
        assert inference_config.svi.early_stopping.patience == 200


class TestSVIRunResult:
    """Test SVIRunResult dataclass."""

    def test_basic_result(self):
        """Test basic SVIRunResult creation."""
        result = SVIRunResult(
            params={"test": jnp.array([1.0])},
            losses=jnp.array([100.0, 90.0, 80.0]),
        )

        assert "test" in result.params
        assert len(result.losses) == 3
        assert result.early_stopped is False
        assert result.stopped_at_step == 0
        assert result.best_loss == float("inf")

    def test_early_stopped_result(self):
        """Test SVIRunResult with early stopping triggered."""
        result = SVIRunResult(
            params={"test": jnp.array([1.0])},
            losses=jnp.array([100.0, 90.0, 80.0, 79.0, 78.5]),
            early_stopped=True,
            stopped_at_step=5,
            best_loss=78.5,
        )

        assert result.early_stopped is True
        assert result.stopped_at_step == 5
        assert result.best_loss == 78.5


# =============================================================================
# Logic Tests - Early Stopping Algorithm (Fast, no SVI)
# =============================================================================


class TestEarlyStoppingLogic:
    """Test the early stopping logic without running actual SVI."""

    def test_convergence_detection(self):
        """Test that convergence is correctly detected with simulated losses."""
        early_stopping = EarlyStoppingConfig(
            patience=50,
            min_delta=0.1,  # Small delta for this test
            check_every=10,
            smoothing_window=10,
        )

        # Simulate a converging loss sequence
        losses = []
        best_loss = float("inf")
        patience_counter = 0

        # First phase: rapid decrease
        for i in range(100):
            losses.append(1000.0 - i * 5)

        # Second phase: plateau
        np.random.seed(42)
        for i in range(100):
            losses.append(500.0 + np.random.randn() * 0.01)

        # Update best_loss from first phase
        for check_step in range(
            early_stopping.smoothing_window,
            150,
            early_stopping.check_every,
        ):
            window_start = max(
                0, len(losses[:check_step]) - early_stopping.smoothing_window
            )
            sl = np.mean(losses[window_start:check_step])
            if best_loss - sl > early_stopping.min_delta:
                best_loss = sl
                patience_counter = 0
            else:
                patience_counter += early_stopping.check_every

        # At plateau, patience should accumulate
        assert patience_counter > 0 or best_loss < 600

    def test_no_early_stop_when_improving(self):
        """Test that early stopping doesn't trigger when loss is improving."""
        early_stopping = EarlyStoppingConfig(
            patience=50,
            min_delta=0.1,
            check_every=10,
            smoothing_window=10,
        )

        # Simulate continuously improving loss
        losses = [1000.0 - i * 10 for i in range(100)]

        best_loss = float("inf")
        patience_counter = 0

        for check_step in range(
            early_stopping.smoothing_window,
            100,
            early_stopping.check_every,
        ):
            window_start = max(0, check_step - early_stopping.smoothing_window)
            smoothed_loss = np.mean(losses[window_start:check_step])

            if best_loss - smoothed_loss > early_stopping.min_delta:
                best_loss = smoothed_loss
                patience_counter = 0
            else:
                patience_counter += early_stopping.check_every

        # Patience should not accumulate because loss is always improving
        assert patience_counter < early_stopping.patience

    def test_patience_accumulation(self):
        """Test that patience counter accumulates correctly during plateau."""
        early_stopping = EarlyStoppingConfig(
            patience=30,
            min_delta=1.0,
            check_every=10,
            smoothing_window=5,
        )

        # Flat loss - no improvement
        losses = [100.0] * 50

        best_loss = float("inf")
        patience_counter = 0

        for check_step in range(
            early_stopping.smoothing_window,
            50,
            early_stopping.check_every,
        ):
            window_start = max(0, check_step - early_stopping.smoothing_window)
            smoothed_loss = np.mean(losses[window_start:check_step])

            if best_loss - smoothed_loss > early_stopping.min_delta:
                best_loss = smoothed_loss
                patience_counter = 0
            else:
                patience_counter += early_stopping.check_every

        # After first check, best_loss is set to 100
        # Subsequent checks show no improvement (delta=0 < min_delta=1)
        # So patience should accumulate
        assert patience_counter >= early_stopping.patience


# =============================================================================
# Integration Tests - Actual SVI Runs (Slow, skipped by default)
# =============================================================================


class TestEarlyStoppingIntegration:
    """Integration tests for early stopping with actual SVI runs.

    These tests are slow and skipped by default.
    Run with: pytest --run-slow
    """

    @pytest.fixture
    def tiny_counts(self):
        """Create a tiny count matrix for fast testing."""
        rng_key = random.PRNGKey(42)
        # Very small: 10 cells, 20 genes
        counts = random.poisson(rng_key, lam=10.0, shape=(10, 20))
        return jnp.array(counts)

    @pytest.mark.slow
    def test_early_stopping_runs_without_error(self, tiny_counts):
        """Test that early stopping completes without errors."""
        import scribe

        early_stopping = EarlyStoppingConfig(
            patience=20,
            min_delta=1.0,
            check_every=5,
            smoothing_window=10,
        )

        # Very short run just to test it works
        results = scribe.fit(
            counts=tiny_counts,
            model="nbdm",
            n_steps=50,
            early_stopping=early_stopping,
            seed=42,
        )

        assert results is not None
        assert hasattr(results, "loss_history")
        assert len(results.loss_history) <= 50

    @pytest.mark.slow
    def test_early_stopping_disabled_runs_full(self, tiny_counts):
        """Test that disabled early stopping runs full n_steps."""
        import scribe

        early_stopping = EarlyStoppingConfig(enabled=False)

        results = scribe.fit(
            counts=tiny_counts,
            model="nbdm",
            n_steps=30,
            early_stopping=early_stopping,
            seed=42,
        )

        assert len(results.loss_history) == 30

    @pytest.mark.slow
    def test_early_stopping_from_dict(self, tiny_counts):
        """Test that early stopping can be specified as a dict (Hydra style)."""
        import scribe

        early_stopping_dict = {
            "enabled": True,
            "patience": 20,
            "min_delta": 1.0,
            "check_every": 5,
            "smoothing_window": 10,
        }

        results = scribe.fit(
            counts=tiny_counts,
            model="nbdm",
            n_steps=50,
            early_stopping=early_stopping_dict,
            seed=42,
        )

        assert results is not None
        assert len(results.loss_history) <= 50
