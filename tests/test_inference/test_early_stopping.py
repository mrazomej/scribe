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
        assert (
            config.min_delta == 1.0
        )  # Default is 1 (reasonable for ELBO ~10^6-10^7)
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

    def test_fit_api_exposes_log_progress_lines(self):
        """The public fit API accepts the SLURM progress-lines flag."""
        import inspect
        import scribe

        assert "log_progress_lines" in inspect.signature(
            scribe.fit
        ).parameters


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

    def test_warmup_excludes_best_loss_tracking(self):
        """Test that warmup does not update best_loss or patience."""
        early_stopping = EarlyStoppingConfig(
            enabled=True,
            patience=30,
            min_delta=1.0,
            check_every=5,
            warmup=20,
            smoothing_window=5,
        )

        losses = [100.0 - i for i in range(40)]
        best_loss = float("inf")
        patience_counter = 0

        for step in range(len(losses)):
            should_check = (
                step % early_stopping.check_every == 0
                and (step + 1) >= early_stopping.smoothing_window
            )
            if not should_check:
                continue

            window_start = max(0, (step + 1) - early_stopping.smoothing_window)
            smoothed_loss = np.mean(losses[window_start : step + 1])

            if early_stopping.enabled and step >= early_stopping.warmup:
                if not np.isfinite(best_loss):
                    best_loss = smoothed_loss
                    patience_counter = 0
                else:
                    improvement = best_loss - smoothed_loss
                    if improvement > early_stopping.min_delta:
                        best_loss = smoothed_loss
                        patience_counter = 0
                    else:
                        patience_counter += early_stopping.check_every
            else:
                assert np.isinf(best_loss)
                assert patience_counter == 0

        assert np.isfinite(best_loss)
        assert best_loss <= 100.0

    def test_min_delta_pct_first_post_warmup_check_sets_baseline(self):
        """Test that percent-mode baseline initializes on first eligible check."""
        early_stopping = EarlyStoppingConfig(
            enabled=True,
            patience=20,
            min_delta_pct=0.01,
            check_every=5,
            warmup=10,
            smoothing_window=5,
        )

        losses = [100.0] * 30
        best_loss = float("inf")
        patience_counter = 0

        for step in range(len(losses)):
            should_check = (
                step % early_stopping.check_every == 0
                and (step + 1) >= early_stopping.smoothing_window
            )
            if not should_check:
                continue

            if not (early_stopping.enabled and step >= early_stopping.warmup):
                continue

            window_start = max(0, (step + 1) - early_stopping.smoothing_window)
            smoothed_loss = np.mean(losses[window_start : step + 1])

            if not np.isfinite(best_loss):
                best_loss = smoothed_loss
                patience_counter = 0
            else:
                improvement = best_loss - smoothed_loss
                improvement_pct = 100.0 * improvement / max(abs(best_loss), 1e-8)
                if improvement_pct > early_stopping.min_delta_pct:
                    best_loss = smoothed_loss
                    patience_counter = 0
                else:
                    patience_counter += early_stopping.check_every

        assert best_loss == 100.0
        assert patience_counter > 0


class TestSVIProgressLoggingInterval:
    """Test periodic SVI progress logging interval behavior."""

    def test_progress_interval_uses_twentieths(self):
        """The interval is ``max(1, n_steps // 20)``."""
        from scribe.svi.inference_engine import _progress_display_interval

        assert _progress_display_interval(50_000) == 2_500
        assert _progress_display_interval(2_001) == 100
        assert _progress_display_interval(19) == 1

    def test_progress_update_emits_about_twenty_times(self):
        """Periodic updates are emitted ~20 times over large runs."""
        from scribe.svi.inference_engine import _should_emit_progress_update

        n_steps = 50_000
        updates = sum(
            1
            for step in range(n_steps)
            if _should_emit_progress_update(step, n_steps)
        )
        assert updates == 20


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


# =============================================================================
# Checkpoint Tests (Fast unit tests)
# =============================================================================


class TestCheckpointConfig:
    """Test checkpoint configuration in EarlyStoppingConfig."""

    def test_default_checkpoint_values(self):
        """Test default checkpoint configuration values."""
        config = EarlyStoppingConfig()

        assert config.checkpoint_dir is None  # No checkpointing by default
        assert config.resume is True  # Auto-resume enabled by default

    def test_checkpoint_dir_can_be_set(self):
        """Test that checkpoint_dir can be configured."""
        config = EarlyStoppingConfig(
            checkpoint_dir="/tmp/test_checkpoints",
            resume=True,
        )

        assert config.checkpoint_dir == "/tmp/test_checkpoints"
        assert config.resume is True

    def test_resume_false_disables_auto_resume(self):
        """Test that resume=False prevents auto-resume."""
        config = EarlyStoppingConfig(
            checkpoint_dir="/tmp/test_checkpoints",
            resume=False,
        )

        assert config.checkpoint_dir == "/tmp/test_checkpoints"
        assert config.resume is False


class TestCheckpointUtilities:
    """Test checkpoint save/load utilities."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create a temporary checkpoint directory."""
        return str(tmp_path / "checkpoints")

    @pytest.fixture
    def sample_params(self):
        """Create sample variational parameters."""
        return {
            "p_loc": jnp.array([0.5, 0.6, 0.7]),
            "p_scale": jnp.array([0.1, 0.1, 0.1]),
            "r_loc": jnp.array([1.0, 2.0, 3.0]),
            "r_scale": jnp.array([0.5, 0.5, 0.5]),
        }

    def test_checkpoint_exists_false_for_empty_dir(self, temp_checkpoint_dir):
        """Test that checkpoint_exists returns False for non-existent dir."""
        from scribe.svi import checkpoint_exists

        assert checkpoint_exists(temp_checkpoint_dir) is False

    def test_checkpoint_exists_false_for_none(self):
        """Test that checkpoint_exists returns False for None."""
        from scribe.svi import checkpoint_exists

        assert checkpoint_exists(None) is False

    def test_save_and_load_checkpoint(self, temp_checkpoint_dir, sample_params):
        """Test saving and loading a checkpoint."""
        from scribe.svi import (
            save_svi_checkpoint,
            load_svi_checkpoint,
            checkpoint_exists,
        )

        # Save checkpoint - now uses optim_state parameter
        # For testing, we pass sample_params as a stand-in for optim_state
        losses = [100.0, 90.0, 80.0, 75.0, 70.0]
        save_svi_checkpoint(
            checkpoint_dir=temp_checkpoint_dir,
            optim_state=sample_params,  # Using params dict as stand-in
            step=100,
            best_loss=70.0,
            losses=losses,
            patience_counter=20,
        )

        # Verify checkpoint exists
        assert checkpoint_exists(temp_checkpoint_dir) is True

        # Load checkpoint with target structure for proper restoration
        result = load_svi_checkpoint(
            temp_checkpoint_dir, target_optim_state=sample_params
        )
        assert result is not None

        restored_optim_state, metadata, restored_losses = result

        # Verify metadata
        assert metadata.step == 100
        assert metadata.best_loss == 70.0
        assert metadata.n_losses == 5
        assert metadata.patience_counter == 20

        # Verify losses
        assert restored_losses == losses

        # Verify optim_state structure (keys should match)
        assert set(restored_optim_state.keys()) == set(sample_params.keys())

    def test_save_overwrites_existing_checkpoint(
        self, temp_checkpoint_dir, sample_params
    ):
        """Test that saving overwrites existing checkpoint."""
        from scribe.svi import save_svi_checkpoint, load_svi_checkpoint

        # Save first checkpoint
        save_svi_checkpoint(
            checkpoint_dir=temp_checkpoint_dir,
            optim_state=sample_params,
            step=50,
            best_loss=100.0,
            losses=[100.0],
            patience_counter=0,
        )

        # Save second checkpoint (should overwrite)
        save_svi_checkpoint(
            checkpoint_dir=temp_checkpoint_dir,
            optim_state=sample_params,
            step=100,
            best_loss=70.0,
            losses=[100.0, 80.0, 70.0],
            patience_counter=10,
        )

        # Load and verify it's the second checkpoint
        result = load_svi_checkpoint(
            temp_checkpoint_dir, target_optim_state=sample_params
        )
        assert result is not None

        _, metadata, losses = result
        assert metadata.step == 100
        assert metadata.best_loss == 70.0
        assert len(losses) == 3

    def test_remove_checkpoint(self, temp_checkpoint_dir, sample_params):
        """Test removing a checkpoint."""
        from scribe.svi import (
            save_svi_checkpoint,
            checkpoint_exists,
            remove_checkpoint,
        )

        # Save checkpoint
        save_svi_checkpoint(
            checkpoint_dir=temp_checkpoint_dir,
            optim_state=sample_params,
            step=100,
            best_loss=70.0,
            losses=[70.0],
            patience_counter=0,
        )
        assert checkpoint_exists(temp_checkpoint_dir) is True

        # Remove checkpoint
        remove_checkpoint(temp_checkpoint_dir)
        assert checkpoint_exists(temp_checkpoint_dir) is False

    def test_load_nonexistent_checkpoint_returns_none(
        self, temp_checkpoint_dir
    ):
        """Test that loading non-existent checkpoint returns None."""
        from scribe.svi import load_svi_checkpoint

        result = load_svi_checkpoint(temp_checkpoint_dir)
        assert result is None


class TestRealSVICheckpointWorkflow:
    """Integration tests for checkpoint save/restore with real SVI objects.

    These tests verify the actual workflow of saving and restoring optimizer
    state, ensuring that training can properly resume after checkpoint restore.
    This is critical because the optimizer state contains complex nested
    structures (tuples, namedtuples, OptimizerState) that must be preserved.
    """

    @pytest.fixture
    def simple_model_and_guide(self):
        """Create a simple model and guide for testing."""
        import numpyro
        import numpyro.distributions as dist

        def model(data=None):
            mu = numpyro.sample("mu", dist.Normal(0, 10))
            sigma = numpyro.sample("sigma", dist.HalfNormal(5))
            with numpyro.plate("data", len(data) if data is not None else 10):
                numpyro.sample("obs", dist.Normal(mu, sigma), obs=data)

        def guide(data=None):
            mu_loc = numpyro.param("mu_loc", 0.0)
            mu_scale = numpyro.param(
                "mu_scale", 1.0, constraint=dist.constraints.positive
            )
            sigma_loc = numpyro.param("sigma_loc", 1.0)
            sigma_scale = numpyro.param(
                "sigma_scale", 0.5, constraint=dist.constraints.positive
            )
            numpyro.sample("mu", dist.Normal(mu_loc, mu_scale))
            numpyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))

        return model, guide

    @pytest.fixture
    def sample_data(self):
        """Create sample data for the model."""
        return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_save_and_restore_real_optim_state(
        self, tmp_path, simple_model_and_guide, sample_data
    ):
        """Test that real SVI optimizer state can be saved and restored."""
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.svi import SVIState
        from numpyro.optim import Adam
        from jax import random
        from scribe.svi import save_svi_checkpoint, load_svi_checkpoint

        model, guide = simple_model_and_guide
        checkpoint_dir = str(tmp_path / "checkpoints")

        # Create SVI and run a few steps
        optimizer = Adam(0.01)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        rng_key = random.PRNGKey(42)
        svi_state = svi.init(rng_key, data=sample_data)

        # Run 10 steps
        for _ in range(10):
            svi_state, loss = svi.update(svi_state, data=sample_data)

        # Get params before saving
        params_before = svi.get_params(svi_state)

        # Save checkpoint
        save_svi_checkpoint(
            checkpoint_dir=checkpoint_dir,
            optim_state=svi_state.optim_state,
            step=10,
            best_loss=float(loss),
            losses=[float(loss)],
            patience_counter=0,
        )

        # Create fresh SVI state for target structure
        fresh_state = svi.init(random.PRNGKey(0), data=sample_data)

        # Load checkpoint
        result = load_svi_checkpoint(
            checkpoint_dir, target_optim_state=fresh_state.optim_state
        )
        assert result is not None

        restored_optim_state, metadata, losses = result
        assert metadata.step == 10

        # Create restored SVIState
        restored_state = SVIState(
            restored_optim_state, fresh_state.mutable_state, random.PRNGKey(99)
        )

        # Verify params match
        params_after = svi.get_params(restored_state)
        for key in params_before:
            assert jnp.allclose(
                params_before[key], params_after[key]
            ), f"Mismatch in {key}"

    def test_training_can_continue_after_restore(
        self, tmp_path, simple_model_and_guide, sample_data
    ):
        """Test that SVI training can continue after checkpoint restore."""
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.svi import SVIState
        from numpyro.optim import Adam
        from jax import random
        from scribe.svi import save_svi_checkpoint, load_svi_checkpoint

        model, guide = simple_model_and_guide
        checkpoint_dir = str(tmp_path / "checkpoints")

        # Create SVI and run a few steps
        optimizer = Adam(0.01)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        rng_key = random.PRNGKey(42)
        svi_state = svi.init(rng_key, data=sample_data)

        # Run 5 steps
        losses_phase1 = []
        for _ in range(5):
            svi_state, loss = svi.update(svi_state, data=sample_data)
            losses_phase1.append(float(loss))

        # Save checkpoint
        save_svi_checkpoint(
            checkpoint_dir=checkpoint_dir,
            optim_state=svi_state.optim_state,
            step=5,
            best_loss=min(losses_phase1),
            losses=losses_phase1,
            patience_counter=0,
        )

        # Create fresh SVI for restoration
        fresh_state = svi.init(random.PRNGKey(0), data=sample_data)

        # Load checkpoint
        result = load_svi_checkpoint(
            checkpoint_dir, target_optim_state=fresh_state.optim_state
        )
        restored_optim_state, metadata, _ = result

        # Create restored SVIState
        restored_state = SVIState(
            restored_optim_state, fresh_state.mutable_state, random.PRNGKey(99)
        )

        # Continue training for 5 more steps - this should NOT raise an error
        losses_phase2 = []
        for _ in range(5):
            restored_state, loss = svi.update(restored_state, data=sample_data)
            losses_phase2.append(float(loss))

        # Verify training continued (losses should be finite and reasonable)
        assert all(
            jnp.isfinite(l) for l in losses_phase2
        ), "Training produced non-finite losses after restore"

        # Verify we can still get params
        final_params = svi.get_params(restored_state)
        assert "mu_loc" in final_params
        assert "sigma_loc" in final_params

    def test_stable_update_works_after_restore(
        self, tmp_path, simple_model_and_guide, sample_data
    ):
        """Test that stable_update (used in early stopping) works after restore."""
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.svi import SVIState
        from numpyro.optim import Adam
        from jax import random
        from scribe.svi import save_svi_checkpoint, load_svi_checkpoint

        model, guide = simple_model_and_guide
        checkpoint_dir = str(tmp_path / "checkpoints")

        # Create SVI
        optimizer = Adam(0.01)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        rng_key = random.PRNGKey(42)
        svi_state = svi.init(rng_key, data=sample_data)

        # Run a few steps with stable_update
        for _ in range(5):
            svi_state, loss = svi.stable_update(svi_state, data=sample_data)

        # Save checkpoint
        save_svi_checkpoint(
            checkpoint_dir=checkpoint_dir,
            optim_state=svi_state.optim_state,
            step=5,
            best_loss=float(loss),
            losses=[float(loss)],
            patience_counter=0,
        )

        # Load and restore
        fresh_state = svi.init(random.PRNGKey(0), data=sample_data)
        result = load_svi_checkpoint(
            checkpoint_dir, target_optim_state=fresh_state.optim_state
        )
        restored_optim_state, _, _ = result
        restored_state = SVIState(
            restored_optim_state, fresh_state.mutable_state, random.PRNGKey(99)
        )

        # This is the critical test - stable_update must work after restore
        # This is what failed before the fix
        try:
            restored_state, loss = svi.stable_update(
                restored_state, data=sample_data
            )
            assert jnp.isfinite(loss), "Loss is not finite after stable_update"
        except ValueError as e:
            pytest.fail(
                f"stable_update failed after checkpoint restore: {e}. "
                "This indicates the optimizer state structure was not "
                "properly preserved during save/load."
            )


class TestCheckpointPathSanitization:
    """Test checkpoint path sanitization for special characters."""

    def test_sanitize_removes_brackets(self):
        """Test that brackets are removed from checkpoint paths."""
        from scribe.svi.checkpoint import _sanitize_checkpoint_path

        # Brackets should be removed (not replaced with underscores)
        path = "/path/to/mixture_params=[mu,phi]/checkpoints"
        sanitized = _sanitize_checkpoint_path(path)

        assert "[" not in sanitized
        assert "]" not in sanitized
        assert "mixture_params=mu,phi" in sanitized

    def test_sanitize_preserves_normal_paths(self):
        """Test that paths without brackets are unchanged."""
        from scribe.svi.checkpoint import _sanitize_checkpoint_path

        path = "/path/to/normal/checkpoints"
        sanitized = _sanitize_checkpoint_path(path)

        assert sanitized == path

    def test_save_and_load_with_brackets_in_path(self, tmp_path):
        """Test that checkpointing works with brackets in the path."""
        from scribe.svi import (
            save_svi_checkpoint,
            load_svi_checkpoint,
            checkpoint_exists,
        )

        # Path with brackets (like Hydra generates for mixture_params=[mu,phi])
        checkpoint_dir = str(tmp_path / "mixture_params=[mu,phi]" / "checkpoints")

        sample_optim_state = {
            "p_loc": jnp.array([0.5, 0.6, 0.7]),
            "r_loc": jnp.array([1.0, 2.0, 3.0]),
        }

        # Save should work
        save_svi_checkpoint(
            checkpoint_dir=checkpoint_dir,
            optim_state=sample_optim_state,
            step=100,
            best_loss=70.0,
            losses=[100.0, 80.0, 70.0],
            patience_counter=10,
        )

        # Checkpoint should exist (using same path with brackets)
        assert checkpoint_exists(checkpoint_dir) is True

        # Load should work
        result = load_svi_checkpoint(
            checkpoint_dir, target_optim_state=sample_optim_state
        )
        assert result is not None

        restored_optim_state, metadata, losses = result
        assert metadata.step == 100
        assert metadata.best_loss == 70.0
        assert set(restored_optim_state.keys()) == set(sample_optim_state.keys())


class TestCheckpointMetadata:
    """Test CheckpointMetadata dataclass."""

    def test_metadata_to_dict(self):
        """Test metadata serialization to dict."""
        from scribe.svi import CheckpointMetadata

        metadata = CheckpointMetadata(
            step=100,
            best_loss=70.0,
            n_losses=50,
            patience_counter=20,
        )

        data = metadata.to_dict()
        assert data["step"] == 100
        assert data["best_loss"] == 70.0
        assert data["n_losses"] == 50
        assert data["patience_counter"] == 20

    def test_metadata_from_dict(self):
        """Test metadata deserialization from dict."""
        from scribe.svi import CheckpointMetadata

        data = {
            "step": 100,
            "best_loss": 70.0,
            "n_losses": 50,
            "patience_counter": 20,
        }

        metadata = CheckpointMetadata.from_dict(data)
        assert metadata.step == 100
        assert metadata.best_loss == 70.0
        assert metadata.n_losses == 50
        assert metadata.patience_counter == 20
