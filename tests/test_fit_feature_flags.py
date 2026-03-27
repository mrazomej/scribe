"""Tests for the variable_capture / zero_inflation feature flags in scribe.fit().

These are unit-level tests that verify model string resolution without
running actual inference. The default model is "nbvcp" (variable capture on).
"""

import pytest
from unittest.mock import patch, MagicMock

import scribe.api as api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_model_kwarg(**kwargs):
    """Call scribe.fit() with mocked internals, return the resolved model str.

    Patches build_config_from_preset so no inference actually runs. Returns
    the ``model`` value that would be passed to the preset builder.
    """
    with patch.object(api, "build_config_from_preset") as mock_preset, \
         patch.object(api, "process_counts_data") as mock_data, \
         patch.object(api, "run_scribe") as mock_run:
        mock_data.return_value = (MagicMock(), None, 10, 5)
        mock_preset.return_value = MagicMock()
        mock_run.return_value = MagicMock()

        import jax.numpy as jnp
        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        try:
            api.fit(dummy, n_steps=1, **kwargs)
        except Exception:
            pass

        if mock_preset.called:
            return mock_preset.call_args.kwargs.get(
                "model", mock_preset.call_args[1].get("model")
            )
    return None


# ---------------------------------------------------------------------------
# Resolution tests
# ---------------------------------------------------------------------------

class TestFeatureFlagResolution:
    """Verify that variable_capture and zero_inflation resolve correctly."""

    def test_default_is_nbvcp(self):
        """No flags, no model= -> default 'nbvcp' (variable capture on)."""
        assert _capture_model_kwarg() == "nbvcp"

    def test_variable_capture_true(self):
        """variable_capture=True alone -> 'nbvcp'."""
        assert _capture_model_kwarg(variable_capture=True) == "nbvcp"

    def test_variable_capture_false(self):
        """variable_capture=False alone -> 'nbdm'."""
        assert _capture_model_kwarg(variable_capture=False) == "nbdm"

    def test_zero_inflation_true(self):
        """zero_inflation=True alone -> 'zinbvcp' (vc defaults to True)."""
        assert _capture_model_kwarg(zero_inflation=True) == "zinbvcp"

    def test_zero_inflation_true_vc_false(self):
        """zero_inflation=True, variable_capture=False -> 'zinb'."""
        assert _capture_model_kwarg(
            zero_inflation=True, variable_capture=False
        ) == "zinb"

    def test_both_true(self):
        """Both flags True -> 'zinbvcp'."""
        assert _capture_model_kwarg(
            variable_capture=True, zero_inflation=True
        ) == "zinbvcp"

    def test_both_false(self):
        """Both flags explicitly False -> 'nbdm'."""
        assert _capture_model_kwarg(
            variable_capture=False, zero_inflation=False
        ) == "nbdm"

    def test_explicit_model_nbdm(self):
        """model='nbdm' explicitly -> 'nbdm' (overrides default)."""
        assert _capture_model_kwarg(model="nbdm") == "nbdm"

    def test_explicit_model_zinb(self):
        """model='zinb' explicitly -> 'zinb'."""
        assert _capture_model_kwarg(model="zinb") == "zinb"

    def test_vc_only_with_none_zi(self):
        """variable_capture=True, zero_inflation=None -> 'nbvcp'."""
        assert _capture_model_kwarg(
            variable_capture=True, zero_inflation=None
        ) == "nbvcp"


class TestFeatureFlagConflict:
    """Verify that conflicting model= and flags raise ValueError."""

    def test_conflict_model_zinb_vc_true(self):
        """model='zinb' + variable_capture=True -> ValueError."""
        import jax.numpy as jnp
        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        with pytest.raises(ValueError, match="conflicts with the feature flags"):
            api.fit(dummy, model="zinb", variable_capture=True, n_steps=1)

    def test_conflict_model_nbdm_zi_true(self):
        """model='nbdm' + zero_inflation=True -> ValueError."""
        import jax.numpy as jnp
        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        with pytest.raises(ValueError, match="conflicts with the feature flags"):
            api.fit(dummy, model="nbdm", zero_inflation=True, n_steps=1)

    def test_no_conflict_when_model_matches_flags(self):
        """model='nbvcp' + variable_capture=True is fine (consistent)."""
        result = _capture_model_kwarg(
            model="nbvcp", variable_capture=True
        )
        assert result == "nbvcp"

    def test_no_conflict_default_model_with_vc_false(self):
        """Default model + variable_capture=False -> flags win, 'nbdm'."""
        result = _capture_model_kwarg(variable_capture=False)
        assert result == "nbdm"
