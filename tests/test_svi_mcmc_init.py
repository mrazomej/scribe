"""Tests for SVI-to-MCMC chain initialization.

Covers:
    - ``compute_init_values`` cross-parameterization conversion utility
    - ``MCMCInferenceEngine.run_inference`` ``init_values`` parameter
    - ``fit()`` ``svi_init`` parameter validation and injection
"""

import warnings
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from scribe.mcmc._init_from_svi import clamp_init_values, compute_init_values
from scribe.models.config import ModelConfig
from scribe.models.config.enums import Parameterization


# ==========================================================================
# Helpers
# ==========================================================================


def _make_model_config(
    base_model="nbdm",
    parameterization="canonical",
    inference_method="mcmc",
    **kwargs,
):
    """Build a minimal ``ModelConfig`` for testing."""
    return ModelConfig(
        base_model=base_model,
        parameterization=parameterization,
        inference_method=inference_method,
        **kwargs,
    )


def _canonical_map(p_val=0.4, r_val=3.0, n_genes=5):
    """Return a canonical SVI MAP dict with scalar p and gene-specific r."""
    return {
        "p": jnp.full((), p_val),
        "r": jnp.full((n_genes,), r_val),
    }


# ==========================================================================
# Tests for clamp_init_values — boundary clamping
# ==========================================================================


class TestClampInitValues:
    """Tests for the ``clamp_init_values`` boundary clamping utility."""

    def test_clamps_probability_params_to_open_unit_interval(self):
        """p and p_capture should be clamped to (eps, 1-eps)."""
        init = {
            "p": jnp.array([0.0, 0.5, 1.0]),
            "p_capture": jnp.array([0.0, 1.0]),
        }
        result = clamp_init_values(init)

        assert jnp.all(result["p"] > 0)
        assert jnp.all(result["p"] < 1)
        assert jnp.all(result["p_capture"] > 0)
        assert jnp.all(result["p_capture"] < 1)

    def test_clamps_positive_params_away_from_zero(self):
        """phi, phi_capture, mu, r should be clamped to (eps, inf)."""
        init = {
            "phi": jnp.array([0.0, 1.0]),
            "phi_capture": jnp.array([0.0, 0.5]),
            "mu": jnp.array([0.0, 2.0]),
            "r": jnp.array([0.0, 3.0]),
        }
        result = clamp_init_values(init)

        for name in ("phi", "phi_capture", "mu", "r"):
            assert jnp.all(result[name] > 0), f"{name} should be > 0"

    def test_leaves_unknown_params_untouched(self):
        """Parameters not in the support map should pass through."""
        init = {
            "p": jnp.array([0.5]),
            "custom_param": jnp.array([-1.0, 0.0, 1.0]),
        }
        result = clamp_init_values(init)

        np.testing.assert_array_equal(
            result["custom_param"], init["custom_param"]
        )

    def test_does_not_mutate_input(self):
        """Should return a new dict, not modify the original."""
        init = {"phi_capture": jnp.array([0.0])}
        result = clamp_init_values(init)

        assert result is not init
        assert float(init["phi_capture"][0]) == 0.0
        assert float(result["phi_capture"][0]) > 0.0


# ==========================================================================
# Tests for compute_init_values — cross-parameterization conversion
# ==========================================================================


class TestComputeInitValues:
    """Tests for the ``compute_init_values`` conversion utility."""

    def test_same_parameterization_passthrough(self):
        """Canonical SVI → canonical MCMC: no new keys added."""
        svi_map = _canonical_map()
        cfg = _make_model_config(parameterization="canonical")

        result = compute_init_values(svi_map, cfg)

        assert "p" in result
        assert "r" in result
        # mu and phi should NOT be computed for canonical target
        assert "mu" not in result
        assert "phi" not in result

    # ------------------------------------------------------------------

    def test_canonical_to_linked_derives_mu(self):
        """Canonical (p, r) → linked (mean_prob) adds mu = r*p/(1-p)."""
        p_val, r_val = 0.4, 3.0
        svi_map = _canonical_map(p_val=p_val, r_val=r_val)
        cfg = _make_model_config(parameterization="mean_prob")

        result = compute_init_values(svi_map, cfg)

        expected_mu = r_val * p_val / (1.0 - p_val)
        np.testing.assert_allclose(
            result["mu"], jnp.full((5,), expected_mu), atol=1e-5
        )
        assert "phi" not in result

    # ------------------------------------------------------------------

    def test_canonical_to_odds_ratio_derives_phi_and_mu(self):
        """Canonical (p, r) → odds_ratio (mean_odds) adds phi and mu."""
        p_val, r_val = 0.4, 3.0
        svi_map = _canonical_map(p_val=p_val, r_val=r_val)
        cfg = _make_model_config(parameterization="mean_odds")

        result = compute_init_values(svi_map, cfg)

        expected_phi = (1.0 - p_val) / p_val
        expected_mu = r_val * p_val / (1.0 - p_val)
        np.testing.assert_allclose(result["phi"], expected_phi, atol=1e-5)
        np.testing.assert_allclose(
            result["mu"], jnp.full((5,), expected_mu), atol=1e-5
        )

    # ------------------------------------------------------------------

    def test_linked_to_odds_ratio_derives_phi(self):
        """Linked (p, mu, r) → odds_ratio adds phi from existing p."""
        p_val, r_val = 0.3, 2.0
        mu_val = r_val * p_val / (1.0 - p_val)
        svi_map = {
            "p": jnp.full((), p_val),
            "mu": jnp.full((5,), mu_val),
            "r": jnp.full((5,), r_val),
        }
        cfg = _make_model_config(parameterization="mean_odds")

        result = compute_init_values(svi_map, cfg)

        expected_phi = (1.0 - p_val) / p_val
        np.testing.assert_allclose(result["phi"], expected_phi, atol=1e-5)
        # mu was already in the map, should be preserved
        np.testing.assert_allclose(
            result["mu"], jnp.full((5,), mu_val), atol=1e-5
        )

    # ------------------------------------------------------------------

    def test_odds_ratio_to_canonical_already_has_p_r(self):
        """Odds-ratio MAP (phi, mu, p, r) → canonical: p and r present."""
        p_val, phi_val, mu_val = 0.4, 1.5, 2.5
        r_val = mu_val * phi_val
        svi_map = {
            "phi": jnp.full((), phi_val),
            "mu": jnp.full((5,), mu_val),
            "p": jnp.full((), p_val),
            "r": jnp.full((5,), r_val),
        }
        cfg = _make_model_config(parameterization="canonical")

        result = compute_init_values(svi_map, cfg)

        np.testing.assert_allclose(result["p"], p_val, atol=1e-5)
        np.testing.assert_allclose(
            result["r"], jnp.full((5,), r_val), atol=1e-5
        )

    # ------------------------------------------------------------------

    def test_backward_compat_alias_linked(self):
        """The 'linked' alias maps to mean_prob parameterization."""
        svi_map = _canonical_map(p_val=0.5, r_val=2.0)
        cfg = _make_model_config(parameterization="linked")

        result = compute_init_values(svi_map, cfg)

        expected_mu = 2.0 * 0.5 / 0.5  # r * p / (1-p)
        np.testing.assert_allclose(
            result["mu"], jnp.full((5,), expected_mu), atol=1e-5
        )

    # ------------------------------------------------------------------

    def test_backward_compat_alias_odds_ratio(self):
        """The 'odds_ratio' alias maps to mean_odds parameterization."""
        svi_map = _canonical_map(p_val=0.25, r_val=4.0)
        cfg = _make_model_config(parameterization="odds_ratio")

        result = compute_init_values(svi_map, cfg)

        expected_phi = 0.75 / 0.25
        np.testing.assert_allclose(result["phi"], expected_phi, atol=1e-5)
        assert "mu" in result

    # ------------------------------------------------------------------

    def test_missing_canonical_raises(self):
        """If p and r are both absent and can't be derived, raise."""
        svi_map = {"foo": jnp.array(1.0)}
        cfg = _make_model_config(parameterization="canonical")

        with pytest.raises(ValueError, match="canonical parameters"):
            compute_init_values(svi_map, cfg)

    # ------------------------------------------------------------------

    def test_derives_p_from_phi_when_missing(self):
        """If p is absent but phi is present, p = 1/(1+phi)."""
        phi_val = 1.5
        mu_val = 2.0
        svi_map = {
            "phi": jnp.full((), phi_val),
            "mu": jnp.full((5,), mu_val),
        }
        cfg = _make_model_config(parameterization="canonical")

        result = compute_init_values(svi_map, cfg)

        expected_p = 1.0 / (1.0 + phi_val)
        expected_r = mu_val * phi_val
        np.testing.assert_allclose(result["p"], expected_p, atol=1e-5)
        np.testing.assert_allclose(
            result["r"], jnp.full((5,), expected_r), atol=1e-5
        )


# ==========================================================================
# Tests for VCP capture parameter conversion
# ==========================================================================


class TestVCPCaptureConversion:
    """Tests for p_capture ↔ phi_capture conversion in VCP models."""

    def test_p_capture_to_phi_capture(self):
        """Canonical VCP (p_capture) → mean_odds VCP (phi_capture)."""
        p_cap = 0.6
        svi_map = {
            "p": jnp.full((), 0.4),
            "r": jnp.full((5,), 3.0),
            "p_capture": jnp.full((10,), p_cap),
        }
        cfg = _make_model_config(
            base_model="nbvcp", parameterization="mean_odds"
        )

        result = compute_init_values(svi_map, cfg)

        expected = (1.0 - p_cap) / p_cap
        np.testing.assert_allclose(
            result["phi_capture"], jnp.full((10,), expected), atol=1e-5
        )

    # ------------------------------------------------------------------

    def test_phi_capture_to_p_capture(self):
        """Mean-odds VCP (phi_capture) → canonical VCP (p_capture)."""
        phi_cap = 0.5
        svi_map = {
            "p": jnp.full((), 0.4),
            "r": jnp.full((5,), 3.0),
            "phi_capture": jnp.full((10,), phi_cap),
        }
        cfg = _make_model_config(
            base_model="nbvcp", parameterization="canonical"
        )

        result = compute_init_values(svi_map, cfg)

        expected = 1.0 / (1.0 + phi_cap)
        np.testing.assert_allclose(
            result["p_capture"], jnp.full((10,), expected), atol=1e-5
        )

    # ------------------------------------------------------------------

    def test_non_vcp_model_ignores_capture(self):
        """Non-VCP model should not add capture parameters."""
        svi_map = _canonical_map()
        cfg = _make_model_config(
            base_model="nbdm", parameterization="mean_odds"
        )

        result = compute_init_values(svi_map, cfg)

        assert "p_capture" not in result
        assert "phi_capture" not in result

    # ------------------------------------------------------------------

    def test_boundary_p_capture_clamped(self):
        """p_capture=1.0 should produce phi_capture > 0 (not exactly 0)."""
        svi_map = {
            "p": jnp.full((), 0.4),
            "r": jnp.full((5,), 3.0),
            "p_capture": jnp.ones((10,)),  # boundary: p_capture = 1.0
        }
        cfg = _make_model_config(
            base_model="nbvcp", parameterization="mean_odds"
        )

        result = compute_init_values(svi_map, cfg)

        # phi_capture must be strictly positive (BetaPrime support)
        assert jnp.all(result["phi_capture"] > 0)

    # ------------------------------------------------------------------

    def test_boundary_p_values_clamped(self):
        """p=1.0 should produce phi > 0 (not exactly 0)."""
        svi_map = {
            "p": jnp.full((), 1.0),  # boundary
            "r": jnp.full((5,), 3.0),
        }
        cfg = _make_model_config(parameterization="mean_odds")

        result = compute_init_values(svi_map, cfg)

        assert jnp.all(result["phi"] > 0)
        assert jnp.all(jnp.isfinite(result["mu"]))


# ==========================================================================
# Tests for MCMCInferenceEngine.run_inference — init_values parameter
# ==========================================================================


class TestEngineInitValues:
    """Tests for the ``init_values`` parameter on the inference engine."""

    @patch("scribe.mcmc.inference_engine.MCMC")
    @patch("scribe.mcmc.inference_engine.NUTS")
    @patch("scribe.mcmc.inference_engine.get_model_and_guide")
    def test_init_to_value_injected_into_nuts(
        self, mock_get_model, mock_nuts, mock_mcmc
    ):
        """When init_values is provided, NUTS receives init_strategy."""
        mock_get_model.return_value = (MagicMock(), None, None)
        mock_mcmc_instance = MagicMock()
        mock_mcmc.return_value = mock_mcmc_instance

        cfg = _make_model_config()
        init_vals = {"p": jnp.array(0.5), "r": jnp.array([1.0, 2.0])}

        from scribe.mcmc.inference_engine import MCMCInferenceEngine

        MCMCInferenceEngine.run_inference(
            model_config=cfg,
            count_data=jnp.zeros((10, 2)),
            n_cells=10,
            n_genes=2,
            init_values=init_vals,
        )

        # NUTS should have been called with an init_strategy kwarg
        nuts_call_kwargs = mock_nuts.call_args[1]
        assert "init_strategy" in nuts_call_kwargs

    # ------------------------------------------------------------------

    @patch("scribe.mcmc.inference_engine.MCMC")
    @patch("scribe.mcmc.inference_engine.NUTS")
    @patch("scribe.mcmc.inference_engine.get_model_and_guide")
    def test_init_values_overrides_existing_strategy_warns(
        self, mock_get_model, mock_nuts, mock_mcmc
    ):
        """Warning emitted when init_values overrides mcmc_kwargs init_strategy."""
        mock_get_model.return_value = (MagicMock(), None, None)
        mock_mcmc.return_value = MagicMock()

        cfg = _make_model_config()
        init_vals = {"p": jnp.array(0.5)}

        from scribe.mcmc.inference_engine import MCMCInferenceEngine

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MCMCInferenceEngine.run_inference(
                model_config=cfg,
                count_data=jnp.zeros((10, 2)),
                n_cells=10,
                n_genes=2,
                mcmc_kwargs={"init_strategy": "existing"},
                init_values=init_vals,
            )

        override_warnings = [
            x for x in w if "init_values overrides" in str(x.message)
        ]
        assert len(override_warnings) == 1

    # ------------------------------------------------------------------

    @patch("scribe.mcmc.inference_engine.MCMC")
    @patch("scribe.mcmc.inference_engine.NUTS")
    @patch("scribe.mcmc.inference_engine.get_model_and_guide")
    def test_no_init_values_passes_mcmc_kwargs_only(
        self, mock_get_model, mock_nuts, mock_mcmc
    ):
        """Without init_values, mcmc_kwargs are forwarded unchanged."""
        mock_get_model.return_value = (MagicMock(), None, None)
        mock_mcmc.return_value = MagicMock()

        cfg = _make_model_config()

        from scribe.mcmc.inference_engine import MCMCInferenceEngine

        MCMCInferenceEngine.run_inference(
            model_config=cfg,
            count_data=jnp.zeros((10, 2)),
            n_cells=10,
            n_genes=2,
            mcmc_kwargs={"max_tree_depth": 5},
        )

        nuts_call_kwargs = mock_nuts.call_args[1]
        assert nuts_call_kwargs["max_tree_depth"] == 5
        assert "init_strategy" not in nuts_call_kwargs


# ==========================================================================
# Tests for fit() — svi_init parameter
# ==========================================================================


class TestFitSVIInit:
    """Tests for the ``svi_init`` parameter on ``scribe.fit()``."""

    def test_svi_init_wrong_inference_method_raises(self):
        """svi_init with inference_method='svi' raises ValueError."""
        from scribe.api import fit

        mock_svi = MagicMock()

        with pytest.raises(ValueError, match="svi_init is only supported"):
            fit(
                counts=jnp.zeros((10, 5)),
                model="nbdm",
                inference_method="svi",
                svi_init=mock_svi,
            )

    # ------------------------------------------------------------------

    def test_svi_init_wrong_inference_method_vae_raises(self):
        """svi_init with inference_method='vae' raises ValueError."""
        from scribe.api import fit

        mock_svi = MagicMock()

        with pytest.raises(ValueError, match="svi_init is only supported"):
            fit(
                counts=jnp.zeros((10, 5)),
                model="nbdm",
                inference_method="vae",
                svi_init=mock_svi,
            )

    # ------------------------------------------------------------------

    def test_svi_init_model_mismatch_warns(self):
        """Warning when SVI base_model differs from target model."""
        from scribe.api import fit

        mock_svi = MagicMock()
        mock_svi.model_config = _make_model_config(
            base_model="zinb", inference_method="svi"
        )
        mock_svi.get_map.return_value = _canonical_map()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                fit(
                    counts=jnp.zeros((10, 5)),
                    model="nbdm",
                    inference_method="mcmc",
                    svi_init=mock_svi,
                    n_samples=1,
                    n_warmup=1,
                )
            except Exception:
                # MCMC run will fail since this is a mock;
                # we only care about the warning
                pass

        mismatch_warnings = [
            x for x in w if "differs from MCMC target" in str(x.message)
        ]
        assert len(mismatch_warnings) >= 1

    # ------------------------------------------------------------------

    def test_svi_init_extracts_map_and_injects_strategy(self):
        """Same parameterization: get_map(canonical=False) is used directly
        (skips the lossy canonical round-trip)."""
        from scribe.api import fit
        from scribe.models.config import MCMCConfig

        mock_svi = MagicMock()
        mock_svi.model_config = _make_model_config(inference_method="svi")
        mock_svi.get_map.return_value = _canonical_map()

        with patch("scribe.api._run_inference") as mock_run:
            mock_run.return_value = MagicMock()
            fit(
                counts=jnp.zeros((10, 5)),
                model="nbdm",
                inference_method="mcmc",
                svi_init=mock_svi,
            )

        # Same parameterization → canonical=False to skip round-trip
        mock_svi.get_map.assert_called_once_with(
            use_mean=True, canonical=False
        )

        call_kwargs = mock_run.call_args[1]
        inf_config = call_kwargs["inference_config"]
        mcmc_kwargs = inf_config.mcmc.mcmc_kwargs
        assert mcmc_kwargs is not None
        assert "init_strategy" in mcmc_kwargs

    # ------------------------------------------------------------------

    def test_svi_init_cross_parameterization(self):
        """SVI linked → MCMC odds_ratio: phi and mu should be in init."""
        from scribe.api import fit

        p_val, r_val = 0.4, 3.0
        mu_val = r_val * p_val / (1.0 - p_val)
        mock_svi = MagicMock()
        mock_svi.model_config = _make_model_config(
            parameterization="linked", inference_method="svi"
        )
        mock_svi.get_map.return_value = {
            "p": jnp.full((), p_val),
            "mu": jnp.full((5,), mu_val),
            "r": jnp.full((5,), r_val),
        }

        with patch("scribe.api._run_inference") as mock_run:
            mock_run.return_value = MagicMock()
            fit(
                counts=jnp.zeros((10, 5)),
                model="nbdm",
                parameterization="odds_ratio",
                inference_method="mcmc",
                svi_init=mock_svi,
            )

        # Extract the init_strategy callable and inspect its values
        call_kwargs = mock_run.call_args[1]
        mcmc_kwargs = call_kwargs["inference_config"].mcmc.mcmc_kwargs
        assert "init_strategy" in mcmc_kwargs

        # Verify the values passed to init_to_value contain phi
        # init_to_value wraps its values in a closure; we test the
        # conversion utility directly for numerical correctness
        svi_map = mock_svi.get_map.return_value
        init_vals = compute_init_values(
            svi_map,
            _make_model_config(parameterization="odds_ratio"),
        )
        expected_phi = (1.0 - p_val) / p_val
        np.testing.assert_allclose(init_vals["phi"], expected_phi, atol=1e-5)
        np.testing.assert_allclose(
            init_vals["mu"], jnp.full((5,), mu_val), atol=1e-5
        )
