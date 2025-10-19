"""
Tests for the ParameterCollector utility class.

This module tests the parameter collection and mapping functionality
provided by the ParameterCollector class.
"""

import pytest
from src.scribe.utils.parameter_collector import ParameterCollector


class TestParameterCollector:
    """Test cases for ParameterCollector class."""

    def test_collect_non_none(self):
        """Test filtering of None values from keyword arguments."""
        result = ParameterCollector.collect_non_none(
            a=1, b=None, c="hello", d=None, e=42
        )
        expected = {"a": 1, "c": "hello", "e": 42}
        assert result == expected

    def test_collect_non_none_empty(self):
        """Test that all None values return empty dict."""
        result = ParameterCollector.collect_non_none(a=None, b=None, c=None)
        assert result == {}

    def test_collect_non_none_all_values(self):
        """Test that no None values returns all values."""
        result = ParameterCollector.collect_non_none(a=1, b="hello", c=42.0)
        expected = {"a": 1, "b": "hello", "c": 42.0}
        assert result == expected

    def test_collect_and_map_priors_constrained_standard(self):
        """Test prior mapping for constrained standard parameterization."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=False,
            parameterization="standard",
            r_prior=(1.0, 1.0),
            p_prior=(2.0, 0.5),
            gate_prior=(1.5, 0.8),
        )
        expected = {
            "r_param_prior": (1.0, 1.0),
            "p_param_prior": (2.0, 0.5),
            "gate_param_prior": (1.5, 0.8),
        }
        assert result == expected

    def test_collect_and_map_priors_constrained_linked(self):
        """Test prior mapping for constrained linked parameterization."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=False,
            parameterization="linked",
            p_prior=(1.0, 1.0),
            mu_prior=(0.0, 1.0),
            gate_prior=(1.5, 0.8),
        )
        expected = {
            "p_param_prior": (1.0, 1.0),
            "mu_param_prior": (0.0, 1.0),
            "gate_param_prior": (1.5, 0.8),
        }
        assert result == expected

    def test_collect_and_map_priors_constrained_odds_ratio(self):
        """Test prior mapping for constrained odds_ratio parameterization."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=False,
            parameterization="odds_ratio",
            phi_prior=(1.0, 1.0),
            mu_prior=(0.0, 1.0),
            phi_capture_prior=(0.5, 0.5),
        )
        expected = {
            "phi_param_prior": (1.0, 1.0),
            "mu_param_prior": (0.0, 1.0),
            "phi_capture_param_prior": (0.5, 0.5),
        }
        assert result == expected

    def test_collect_and_map_priors_unconstrained_standard(self):
        """Test prior mapping for unconstrained standard parameterization."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=True,
            parameterization="standard",
            r_prior=(0.0, 1.0),
            p_prior=(0.0, 1.0),
            gate_prior=(0.0, 1.0),
        )
        expected = {
            "r_unconstrained_prior": (0.0, 1.0),
            "p_unconstrained_prior": (0.0, 1.0),
            "gate_unconstrained_prior": (0.0, 1.0),
        }
        assert result == expected

    def test_collect_and_map_priors_unconstrained_linked(self):
        """Test prior mapping for unconstrained linked parameterization."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=True,
            parameterization="linked",
            p_prior=(0.0, 1.0),
            mu_prior=(0.0, 1.0),
            gate_prior=(0.0, 1.0),
        )
        expected = {
            "p_unconstrained_prior": (0.0, 1.0),
            "gate_unconstrained_prior": (0.0, 1.0),
            "mu_unconstrained_prior": (0.0, 1.0),
        }
        assert result == expected

    def test_collect_and_map_priors_unconstrained_odds_ratio(self):
        """Test prior mapping for unconstrained odds_ratio parameterization."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=True,
            parameterization="odds_ratio",
            phi_prior=(0.0, 1.0),
            mu_prior=(0.0, 1.0),
            phi_capture_prior=(0.0, 1.0),
        )
        expected = {
            "phi_unconstrained_prior": (0.0, 1.0),
            "mu_unconstrained_prior": (0.0, 1.0),
            "phi_capture_unconstrained_prior": (0.0, 1.0),
        }
        assert result == expected

    def test_collect_and_map_priors_with_nones(self):
        """Test that None values are filtered out in mapping."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=False,
            parameterization="standard",
            r_prior=(1.0, 1.0),
            p_prior=None,  # Should be filtered out
            gate_prior=(1.5, 0.8),
        )
        expected = {
            "r_param_prior": (1.0, 1.0),
            "gate_param_prior": (1.5, 0.8),
        }
        assert result == expected

    def test_collect_and_map_priors_all_nones(self):
        """Test that all None values return empty dict."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=False,
            parameterization="standard",
            r_prior=None,
            p_prior=None,
            gate_prior=None,
        )
        assert result == {}

    def test_collect_vae_params(self):
        """Test VAE parameter collection."""
        result = ParameterCollector.collect_vae_params(
            vae_latent_dim=5,
            vae_hidden_dims=[256, 128],
            vae_activation="gelu",
            vae_prior_type="decoupled",
            vae_prior_num_layers=3,
        )
        expected = {
            "vae_latent_dim": 5,
            "vae_hidden_dims": [256, 128],
            "vae_activation": "gelu",
            "vae_prior_type": "decoupled",
            "vae_prior_num_layers": 3,
            "vae_prior_mask_type": "alternating",  # Default value
            "vae_standardize": False,  # Default value
        }
        assert result == expected

    def test_collect_vae_params_with_nones(self):
        """Test VAE parameter collection with some None values."""
        result = ParameterCollector.collect_vae_params(
            vae_latent_dim=3,
            vae_hidden_dims=None,  # Should be filtered out
            vae_activation="relu",
            vae_prior_type="standard",
            vae_prior_num_layers=None,  # Should be filtered out
        )
        expected = {
            "vae_latent_dim": 3,
            "vae_activation": "relu",
            "vae_prior_type": "standard",
            "vae_prior_mask_type": "alternating",  # Default value
            "vae_standardize": False,  # Default value
        }
        assert result == expected

    def test_collect_vae_params_all_nones(self):
        """Test VAE parameter collection with all None values except defaults."""
        result = ParameterCollector.collect_vae_params(
            vae_latent_dim=3,  # Default value
            vae_hidden_dims=None,
            vae_activation=None,
            vae_prior_type="standard",  # Default value
            vae_prior_mask_type="alternating",  # Default value
            vae_standardize=False,  # Default value
        )
        expected = {
            "vae_latent_dim": 3,
            "vae_prior_type": "standard",
            "vae_prior_mask_type": "alternating",
            "vae_standardize": False,
        }
        assert result == expected

    def test_collect_vae_params_empty(self):
        """Test VAE parameter collection with only None values."""
        result = ParameterCollector.collect_vae_params(
            vae_latent_dim=3,  # Only default values
            vae_hidden_dims=None,
            vae_activation=None,
            vae_input_transformation=None,
            vae_vcp_hidden_dims=None,
            vae_vcp_activation=None,
            vae_prior_type="standard",
            vae_prior_num_layers=None,
            vae_prior_hidden_dims=None,
            vae_prior_activation=None,
            vae_prior_mask_type="alternating",
            vae_standardize=False,
        )
        expected = {
            "vae_latent_dim": 3,
            "vae_prior_type": "standard",
            "vae_prior_mask_type": "alternating",
            "vae_standardize": False,
        }
        assert result == expected

    def test_parameterization_specific_priors_linked(self):
        """Test that linked parameterization only includes mu_prior."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=True,
            parameterization="linked",
            p_prior=(0.0, 1.0),
            mu_prior=(0.0, 1.0),
            phi_prior=(0.0, 1.0),  # Should not be included for linked
        )
        expected = {
            "p_unconstrained_prior": (0.0, 1.0),
            "mu_unconstrained_prior": (0.0, 1.0),
        }
        assert result == expected

    def test_parameterization_specific_priors_odds_ratio(self):
        """Test that odds_ratio parameterization includes phi and mu priors."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=True,
            parameterization="odds_ratio",
            phi_prior=(0.0, 1.0),
            mu_prior=(0.0, 1.0),
            phi_capture_prior=(0.0, 1.0),
        )
        expected = {
            "phi_unconstrained_prior": (0.0, 1.0),
            "mu_unconstrained_prior": (0.0, 1.0),
            "phi_capture_unconstrained_prior": (0.0, 1.0),
        }
        assert result == expected

    def test_parameterization_specific_priors_standard(self):
        """Test that standard parameterization doesn't include mu or phi priors."""
        result = ParameterCollector.collect_and_map_priors(
            unconstrained=True,
            parameterization="standard",
            p_prior=(0.0, 1.0),
            r_prior=(0.0, 1.0),
            mu_prior=(0.0, 1.0),  # Should not be included for standard
            phi_prior=(0.0, 1.0),  # Should not be included for standard
        )
        expected = {
            "p_unconstrained_prior": (0.0, 1.0),
            "r_unconstrained_prior": (0.0, 1.0),
        }
        assert result == expected

    def test_mixing_prior_mapping(self):
        """Test that mixing_prior is mapped correctly for constrained vs unconstrained."""
        # Constrained
        result_constrained = ParameterCollector.collect_and_map_priors(
            unconstrained=False,
            parameterization="standard",
            mixing_prior=[1.0, 2.0, 3.0],
        )
        expected_constrained = {"mixing_param_prior": [1.0, 2.0, 3.0]}
        assert result_constrained == expected_constrained

        # Unconstrained
        result_unconstrained = ParameterCollector.collect_and_map_priors(
            unconstrained=True,
            parameterization="standard",
            mixing_prior=[1.0, 2.0, 3.0],
        )
        expected_unconstrained = {
            "mixing_logits_unconstrained_prior": [1.0, 2.0, 3.0]
        }
        assert result_unconstrained == expected_unconstrained
