"""Tests for float64 (x64) precision support in inference.

Covers:
    - ``_run_inference()`` enable_x64 parameter: context manager wrapping,
      count_data casting, and state restoration.
    - ``fit()`` enable_x64 parameter: method-dependent defaults and explicit
      overrides.
    - Hydra integration: ``enable_x64`` config flow and ``svi_init`` path
      loading via ``_load_svi_init``.
"""

import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from scribe.inference.dispatcher import _run_inference
from scribe.models.config import (
    DataConfig,
    InferenceConfig,
    MCMCConfig,
    ModelConfig,
    SVIConfig,
)
from scribe.models.config.enums import InferenceMethod


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


# ==========================================================================
# Tests for _run_inference — enable_x64 parameter
# ==========================================================================


class TestRunInferenceX64:
    """Tests for the ``enable_x64`` parameter on ``_run_inference``."""

    def test_x64_casts_count_data_to_float64(self):
        """When enable_x64=True, the handler receives float64 count_data."""
        captured = {}

        def _fake_handler(**kwargs):
            captured["count_data_dtype"] = kwargs["count_data"].dtype
            return MagicMock()

        mcmc_config = MCMCConfig(n_samples=1, n_warmup=1, n_chains=1)
        inf_config = InferenceConfig.from_mcmc(mcmc_config)

        with patch(
            "scribe.inference.dispatcher._INFERENCE_HANDLERS",
            {InferenceMethod.MCMC: _fake_handler},
        ):
            _run_inference(
                inference_method=InferenceMethod.MCMC,
                model_config=_make_model_config(),
                count_data=jnp.zeros((5, 3), dtype=jnp.float32),
                inference_config=inf_config,
                adata=None,
                n_cells=5,
                n_genes=3,
                data_config=DataConfig(),
                seed=0,
                enable_x64=True,
            )

        assert captured["count_data_dtype"] == jnp.float64

    # ------------------------------------------------------------------

    def test_no_x64_keeps_count_data_float32(self):
        """When enable_x64=False, count_data stays float32."""
        captured = {}

        def _fake_handler(**kwargs):
            captured["count_data_dtype"] = kwargs["count_data"].dtype
            return MagicMock()

        mcmc_config = MCMCConfig(n_samples=1, n_warmup=1, n_chains=1)
        inf_config = InferenceConfig.from_mcmc(mcmc_config)

        with patch(
            "scribe.inference.dispatcher._INFERENCE_HANDLERS",
            {InferenceMethod.MCMC: _fake_handler},
        ):
            _run_inference(
                inference_method=InferenceMethod.MCMC,
                model_config=_make_model_config(),
                count_data=jnp.zeros((5, 3), dtype=jnp.float32),
                inference_config=inf_config,
                adata=None,
                n_cells=5,
                n_genes=3,
                data_config=DataConfig(),
                seed=0,
                enable_x64=False,
            )

        assert captured["count_data_dtype"] == jnp.float32

    # ------------------------------------------------------------------

    def test_x64_context_manager_restores_state(self):
        """x64 should not remain permanently enabled after the call."""
        # Ensure x64 is OFF before the call
        prior_state = jax.config.jax_enable_x64

        def _fake_handler(**kwargs):
            return MagicMock()

        mcmc_config = MCMCConfig(n_samples=1, n_warmup=1, n_chains=1)
        inf_config = InferenceConfig.from_mcmc(mcmc_config)

        with patch(
            "scribe.inference.dispatcher._INFERENCE_HANDLERS",
            {InferenceMethod.MCMC: _fake_handler},
        ):
            _run_inference(
                inference_method=InferenceMethod.MCMC,
                model_config=_make_model_config(),
                count_data=jnp.zeros((5, 3), dtype=jnp.float32),
                inference_config=inf_config,
                adata=None,
                n_cells=5,
                n_genes=3,
                data_config=DataConfig(),
                seed=0,
                enable_x64=True,
            )

        # After the call, x64 setting should be restored to its prior state
        assert jax.config.jax_enable_x64 == prior_state

    # ------------------------------------------------------------------

    def test_x64_is_active_inside_handler(self):
        """Inside the handler, x64 must be active so JAX ops use float64."""
        captured = {}

        def _fake_handler(**kwargs):
            captured["x64_active"] = jax.config.jax_enable_x64
            return MagicMock()

        mcmc_config = MCMCConfig(n_samples=1, n_warmup=1, n_chains=1)
        inf_config = InferenceConfig.from_mcmc(mcmc_config)

        with patch(
            "scribe.inference.dispatcher._INFERENCE_HANDLERS",
            {InferenceMethod.MCMC: _fake_handler},
        ):
            _run_inference(
                inference_method=InferenceMethod.MCMC,
                model_config=_make_model_config(),
                count_data=jnp.zeros((5, 3), dtype=jnp.float32),
                inference_config=inf_config,
                adata=None,
                n_cells=5,
                n_genes=3,
                data_config=DataConfig(),
                seed=0,
                enable_x64=True,
            )

        assert captured["x64_active"] is True


# ==========================================================================
# Tests for fit() — enable_x64 defaults and overrides
# ==========================================================================


class TestFitX64:
    """Tests for the ``enable_x64`` parameter on ``scribe.fit()``."""

    def test_mcmc_defaults_to_x64(self):
        """fit(inference_method='mcmc') should pass enable_x64=True."""
        from scribe.api import fit

        with patch("scribe.api._run_inference") as mock_run:
            mock_run.return_value = MagicMock()
            fit(
                counts=jnp.zeros((10, 5)),
                model="nbdm",
                inference_method="mcmc",
            )

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["enable_x64"] is True

    # ------------------------------------------------------------------

    def test_svi_defaults_to_no_x64(self):
        """fit(inference_method='svi') should pass enable_x64=False."""
        from scribe.api import fit

        with patch("scribe.api._run_inference") as mock_run:
            mock_run.return_value = MagicMock()
            fit(
                counts=jnp.zeros((10, 5)),
                model="nbdm",
                inference_method="svi",
            )

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["enable_x64"] is False

    # ------------------------------------------------------------------

    def test_vae_defaults_to_no_x64(self):
        """fit(inference_method='vae') should pass enable_x64=False."""
        from scribe.api import fit

        with patch("scribe.api._run_inference") as mock_run:
            mock_run.return_value = MagicMock()
            fit(
                counts=jnp.zeros((10, 5)),
                model="nbdm",
                inference_method="vae",
            )

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["enable_x64"] is False

    # ------------------------------------------------------------------

    def test_explicit_x64_true_for_svi(self):
        """fit(inference_method='svi', enable_x64=True) passes True."""
        from scribe.api import fit

        with patch("scribe.api._run_inference") as mock_run:
            mock_run.return_value = MagicMock()
            fit(
                counts=jnp.zeros((10, 5)),
                model="nbdm",
                inference_method="svi",
                enable_x64=True,
            )

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["enable_x64"] is True

    # ------------------------------------------------------------------

    def test_explicit_x64_false_for_mcmc(self):
        """fit(inference_method='mcmc', enable_x64=False) passes False."""
        from scribe.api import fit

        with patch("scribe.api._run_inference") as mock_run:
            mock_run.return_value = MagicMock()
            fit(
                counts=jnp.zeros((10, 5)),
                model="nbdm",
                inference_method="mcmc",
                enable_x64=False,
            )

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["enable_x64"] is False


# ==========================================================================
# Tests for Hydra integration — enable_x64 config flow
# ==========================================================================


class TestHydraEnableX64Config:
    """Verify that ``enable_x64`` from inference YAML flows to ``fit()``."""

    def test_mcmc_yaml_has_enable_x64_true(self):
        """conf/inference/mcmc.yaml should set enable_x64: true."""
        import yaml

        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "conf",
            "inference",
            "mcmc.yaml",
        )
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["enable_x64"] is True

    # ------------------------------------------------------------------

    def test_svi_yaml_has_enable_x64_null(self):
        """conf/inference/svi.yaml should set enable_x64: null."""
        import yaml

        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "conf",
            "inference",
            "svi.yaml",
        )
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["enable_x64"] is None

    # ------------------------------------------------------------------

    def test_enable_x64_flows_through_inference_cfg(self):
        """Simulates the kwargs.update(inference_cfg) pattern from infer.py.

        When ``enable_x64`` is in ``inference_cfg``, it should appear in
        the kwargs passed to ``scribe.fit()``.
        """
        from scribe.api import fit

        with patch("scribe.api._run_inference") as mock_run:
            mock_run.return_value = MagicMock()
            # Simulate what infer.py does: inference_cfg dict with enable_x64
            kwargs = {
                "counts": jnp.zeros((10, 5)),
                "model": "nbdm",
                "inference_method": "mcmc",
                "enable_x64": True,
                "n_samples": 100,
                "n_warmup": 50,
                "n_chains": 1,
            }
            fit(**kwargs)

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["enable_x64"] is True


# ==========================================================================
# Tests for _load_svi_init helper in infer.py
# ==========================================================================


class TestLoadSVIInit:
    """Tests for the ``_load_svi_init`` helper that loads SVI results."""

    def test_none_path_returns_none(self):
        """When path is None, return None without any file I/O."""
        from infer import _load_svi_init

        assert _load_svi_init(None) is None

    # ------------------------------------------------------------------

    def test_nonexistent_path_raises(self):
        """Non-existent file path raises FileNotFoundError."""
        from infer import _load_svi_init

        with patch(
            "hydra.utils.to_absolute_path", return_value="/no/such/file.pkl"
        ):
            with pytest.raises(
                FileNotFoundError, match="svi_init file not found"
            ):
                _load_svi_init("/no/such/file.pkl")

    # ------------------------------------------------------------------

    def test_wrong_type_raises(self):
        """A pickle that is not ScribeSVIResults raises TypeError."""
        from infer import _load_svi_init

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "svi_results"}, f)
            tmp_path = f.name

        try:
            with patch(
                "hydra.utils.to_absolute_path", return_value=tmp_path
            ):
                with pytest.raises(TypeError, match="ScribeSVIResults"):
                    _load_svi_init(tmp_path)
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------

    def test_valid_svi_results_loaded(self):
        """A pickled ScribeSVIResults object is loaded successfully."""
        from infer import _load_svi_init

        # Pickle a plain dict and patch ScribeSVIResults to be `dict`
        # so the isinstance check passes (MagicMock(spec=...) is not
        # picklable).
        sentinel = {"marker": "svi_results"}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(sentinel, f)
            tmp_path = f.name

        try:
            with (
                patch(
                    "hydra.utils.to_absolute_path", return_value=tmp_path
                ),
                patch("scribe.svi.results.ScribeSVIResults", dict),
            ):
                loaded = _load_svi_init(tmp_path)

            assert loaded is not None
            assert loaded["marker"] == "svi_results"
        finally:
            os.unlink(tmp_path)
