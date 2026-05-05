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
        # Return a concrete numeric matrix so fit() preprocessing paths that
        # compute count-derived metadata can execute without type errors.
        import jax.numpy as jnp

        mock_data.return_value = (
            jnp.zeros((10, 5), dtype=jnp.int32),
            None,
            10,
            5,
        )
        mock_preset.return_value = MagicMock()
        mock_run.return_value = MagicMock()

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


class TestPLNFeatureFlags:
    """PLN-specific feature-flag resolution.

    PLN does not have a separate 'plnvcp' model string. Capture is
    activated internally by supplying capture priors. The model string
    always stays ``"pln"``.
    """

    def test_pln_stays_pln_with_vc_true(self):
        """``model='pln', variable_capture=True`` resolves to ``pln``."""
        assert _capture_model_kwarg(model="pln", variable_capture=True) == "pln"

    def test_pln_stays_pln_with_vc_false(self):
        """``model='pln', variable_capture=False`` resolves to ``pln``."""
        assert _capture_model_kwarg(model="pln", variable_capture=False) == "pln"

    def test_pln_stays_pln_with_vc_true_and_capture_prior(self):
        """``model='pln', variable_capture=True`` with capture prior stays ``pln``."""
        import numpy as np

        result = _capture_model_kwarg(
            model="pln",
            variable_capture=True,
            priors={"capture_efficiency": (float(np.log(1e5)), 0.5)},
        )
        assert result == "pln"

    def test_pln_vc_true_no_priors_warns(self):
        """``model='pln', variable_capture=True`` without capture priors warns."""
        import jax.numpy as jnp

        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        with pytest.warns(UserWarning, match="variable_capture=True with model='pln'"):
            try:
                api.fit(dummy, model="pln", variable_capture=True, n_steps=1)
            except Exception:
                pass

    def test_pln_zi_true_raises(self):
        """``model='pln', zero_inflation=True`` raises ``ValueError``."""
        import jax.numpy as jnp

        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        with pytest.raises(
            ValueError, match="Zero-inflation is not supported for the PLN"
        ):
            api.fit(dummy, model="pln", zero_inflation=True, n_steps=1)

    def test_pln_no_flags_stays_pln(self):
        """``model='pln'`` without any flags stays ``pln``."""
        assert _capture_model_kwarg(model="pln") == "pln"


class TestLNMFeatureFlags:
    """LNM-family feature-flag resolution.

    LNM has two member names: ``"lnm"`` (no capture) and ``"lnmvcp"``
    (variable capture probability). The base name auto-promotes via
    ``variable_capture``; an explicit composite name must agree with
    the flag if both are passed.

    Regression tests for the bug where the family-default comparison
    used ``_default_model = "nbvcp"`` (a DM-family name), which made
    ``model="lnm" + variable_capture=True`` raise instead of
    promoting to ``"lnmvcp"``.
    """

    def test_lnm_promotes_to_lnmvcp_with_vc_true(self):
        """``model='lnm', variable_capture=True`` promotes to ``"lnmvcp"``."""
        assert (
            _capture_model_kwarg(model="lnm", variable_capture=True)
            == "lnmvcp"
        )

    def test_lnm_stays_lnm_with_vc_false(self):
        """``model='lnm', variable_capture=False`` stays ``"lnm"``."""
        assert (
            _capture_model_kwarg(model="lnm", variable_capture=False)
            == "lnm"
        )

    def test_lnmvcp_with_matching_vc_true(self):
        """``model='lnmvcp', variable_capture=True`` stays ``"lnmvcp"``."""
        assert (
            _capture_model_kwarg(model="lnmvcp", variable_capture=True)
            == "lnmvcp"
        )

    def test_lnmvcp_with_conflicting_vc_false_raises(self):
        """``model='lnmvcp', variable_capture=False`` is a conflict."""
        import jax.numpy as jnp

        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        with pytest.raises(
            ValueError, match="conflicts with the feature flags"
        ):
            api.fit(
                dummy, model="lnmvcp", variable_capture=False, n_steps=1
            )

    def test_lnm_no_flag_stays_lnm(self):
        """``model='lnm'`` with no flags stays ``"lnm"`` (no auto-promotion)."""
        assert _capture_model_kwarg(model="lnm") == "lnm"

    def test_lnmvcp_no_flag_stays_lnmvcp(self):
        """``model='lnmvcp'`` with no flags stays ``"lnmvcp"``."""
        assert _capture_model_kwarg(model="lnmvcp") == "lnmvcp"

    def test_lnm_zi_true_raises(self):
        """``model='lnm', zero_inflation=True`` raises (LNM family doesn't support ZI)."""
        import jax.numpy as jnp

        dummy = jnp.zeros((10, 5), dtype=jnp.int32)
        with pytest.raises(
            ValueError,
            match="Zero-inflation is not supported for the LNM",
        ):
            api.fit(dummy, model="lnm", zero_inflation=True, n_steps=1)
