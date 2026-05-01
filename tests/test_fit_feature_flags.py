"""Tests for variable_capture / zero_inflation feature flag resolution in ``fit``.

These tests are unit-level and verify model-string resolution behavior without
running real inference.
"""

from unittest.mock import MagicMock, patch

import pytest

import scribe.api as api


def _capture_model_kwarg(**kwargs):
    """Call ``fit`` with mocked internals and return the resolved ``model`` value.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments forwarded to ``scribe.api.fit``.

    Returns
    -------
    str or None
        The resolved ``model`` string passed to ``build_config_from_preset``, or
        ``None`` when that builder is not called.
    """
    # Patch internals to isolate resolution logic and avoid real inference.
    with (
        patch.object(api, "build_config_from_preset") as mock_preset,
        patch.object(api, "process_counts_data") as mock_data,
        patch.object(api, "_run_inference") as mock_run,
    ):
        mock_data.return_value = (MagicMock(), None, 10, 5)
        mock_preset.return_value = MagicMock()
        mock_run.return_value = MagicMock()

        import jax.numpy as jnp

        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        try:
            api.fit(dummy, n_steps=1, **kwargs)
        except Exception:
            # Resolution tests only need the preset builder call arguments.
            pass

        if mock_preset.called:
            return mock_preset.call_args.kwargs.get(
                "model", mock_preset.call_args[1].get("model")
            )
    return None


class TestFeatureFlagResolution:
    """Verify that variable_capture and zero_inflation resolve correctly."""

    def test_default_is_nbvcp(self):
        """No flags and no explicit model resolve to ``nbvcp``."""
        assert _capture_model_kwarg() == "nbvcp"

    def test_variable_capture_true(self):
        """``variable_capture=True`` alone resolves to ``nbvcp``."""
        assert _capture_model_kwarg(variable_capture=True) == "nbvcp"

    def test_variable_capture_false(self):
        """``variable_capture=False`` alone resolves to ``nbdm``."""
        assert _capture_model_kwarg(variable_capture=False) == "nbdm"

    def test_zero_inflation_true(self):
        """``zero_inflation=True`` alone resolves to ``zinbvcp``."""
        assert _capture_model_kwarg(zero_inflation=True) == "zinbvcp"

    def test_zero_inflation_true_vc_false(self):
        """``zero_inflation=True`` with ``variable_capture=False`` resolves to ``zinb``."""
        assert (
            _capture_model_kwarg(zero_inflation=True, variable_capture=False)
            == "zinb"
        )

    def test_both_true(self):
        """Both flags true resolve to ``zinbvcp``."""
        assert (
            _capture_model_kwarg(variable_capture=True, zero_inflation=True)
            == "zinbvcp"
        )

    def test_both_false(self):
        """Both flags false resolve to ``nbdm``."""
        assert (
            _capture_model_kwarg(variable_capture=False, zero_inflation=False)
            == "nbdm"
        )

    def test_explicit_model_nbdm(self):
        """An explicit ``model='nbdm'`` remains ``nbdm``."""
        assert _capture_model_kwarg(model="nbdm") == "nbdm"

    def test_explicit_model_zinb(self):
        """An explicit ``model='zinb'`` remains ``zinb``."""
        assert _capture_model_kwarg(model="zinb") == "zinb"

    def test_vc_only_with_none_zi(self):
        """``variable_capture=True`` and ``zero_inflation=None`` resolve to ``nbvcp``."""
        assert (
            _capture_model_kwarg(variable_capture=True, zero_inflation=None)
            == "nbvcp"
        )


class TestFeatureFlagConflict:
    """Verify that conflicting model strings and flags raise ``ValueError``."""

    def test_conflict_model_zinb_vc_true(self):
        """``model='zinb'`` with ``variable_capture=True`` raises ``ValueError``."""
        import jax.numpy as jnp

        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        with pytest.raises(
            ValueError, match="conflicts with the feature flags"
        ):
            api.fit(dummy, model="zinb", variable_capture=True, n_steps=1)

    def test_conflict_model_nbdm_zi_true(self):
        """``model='nbdm'`` with ``zero_inflation=True`` raises ``ValueError``."""
        import jax.numpy as jnp

        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        with pytest.raises(
            ValueError, match="conflicts with the feature flags"
        ):
            api.fit(dummy, model="nbdm", zero_inflation=True, n_steps=1)

    def test_no_conflict_when_model_matches_flags(self):
        """Consistent explicit model and flags do not raise conflicts."""
        result = _capture_model_kwarg(model="nbvcp", variable_capture=True)
        assert result == "nbvcp"

    def test_no_conflict_default_model_with_vc_false(self):
        """Default model with ``variable_capture=False`` resolves to ``nbdm``."""
        result = _capture_model_kwarg(variable_capture=False)
        assert result == "nbdm"
