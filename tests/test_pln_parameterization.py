"""Tests for the PoissonLogNormalParameterization.

Unlike the LNM family which has three variants (canonical / mean_prob /
mean_odds), the PLN has a single natural parameterization.  There is no
NB submodel to reparameterize, so no variant dispatch or derived params.
"""

from __future__ import annotations

import pytest

from scribe.models.config import GuideFamilyConfig
from scribe.models.parameterizations import (
    PARAMETERIZATIONS,
    PoissonLogNormalParameterization,
    is_poisson_lognormal_family,
    resolve_user_parameterization_for_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pln_param():
    return PoissonLogNormalParameterization()


# ---------------------------------------------------------------------------
# Identity / introspection
# ---------------------------------------------------------------------------


def test_name(pln_param):
    """The registry key is ``poisson_lognormal``."""
    assert pln_param.name == "poisson_lognormal"


def test_gene_param_name(pln_param):
    """PLN gene parameter is ``y_log_rate`` (G-dimensional)."""
    assert pln_param.gene_param_name == "y_log_rate"


def test_requires_vae(pln_param):
    """PLN requires VAE-based inference."""
    assert pln_param.requires_vae is True


def test_decoder_output_spec(pln_param):
    """Single identity head for ``y_log_rate``."""
    expected = [("y_log_rate", "identity")]
    assert pln_param.decoder_output_spec("pln") == expected


# ---------------------------------------------------------------------------
# Core parameters
# ---------------------------------------------------------------------------


def test_core_parameters_empty(pln_param):
    """PLN has no scalar core parameters (no NB totals submodel)."""
    assert pln_param.core_parameters == []


# ---------------------------------------------------------------------------
# Param specs
# ---------------------------------------------------------------------------


def test_build_param_specs_empty_for_single_component(pln_param):
    """With ``n_components=1`` (default), no scalar specs are needed."""
    specs = pln_param.build_param_specs(
        unconstrained=False, guide_families=GuideFamilyConfig()
    )
    assert specs == []


def test_build_param_specs_rejects_mixtures(pln_param):
    """PLN v1 defers mixture support -- must raise for ``n_components > 1``."""
    with pytest.raises(NotImplementedError, match="mixture"):
        pln_param.build_param_specs(
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
            n_components=2,
        )


# ---------------------------------------------------------------------------
# Derived parameters
# ---------------------------------------------------------------------------


def test_build_derived_params_empty(pln_param):
    """No derived parameters (everything is direct)."""
    assert pln_param.build_derived_params() == []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_poisson_lognormal():
    """The global registry includes the PLN singleton."""
    assert "poisson_lognormal" in PARAMETERIZATIONS
    assert isinstance(
        PARAMETERIZATIONS["poisson_lognormal"],
        PoissonLogNormalParameterization,
    )


# ---------------------------------------------------------------------------
# Family helper
# ---------------------------------------------------------------------------


def test_is_poisson_lognormal_family_positive():
    """``is_poisson_lognormal_family`` recognizes the PLN key."""
    assert is_poisson_lognormal_family("poisson_lognormal")


def test_is_poisson_lognormal_family_negative():
    """Non-PLN keys are rejected."""
    assert not is_poisson_lognormal_family("logistic_normal_canonical")
    assert not is_poisson_lognormal_family("canonical")
    assert not is_poisson_lognormal_family("")
    assert not is_poisson_lognormal_family("pln")


# ---------------------------------------------------------------------------
# User-facing dispatch
# ---------------------------------------------------------------------------


def test_resolve_user_parameterization_for_pln():
    """``pln`` model resolves to ``poisson_lognormal``."""
    assert (
        resolve_user_parameterization_for_model("pln", "poisson_lognormal")
        == "poisson_lognormal"
    )


def test_resolve_user_parameterization_pln_default():
    """When user passes None / default, the PLN branch returns
    ``poisson_lognormal``."""
    result = resolve_user_parameterization_for_model("pln", None)
    assert result == "poisson_lognormal"


def test_resolve_does_not_affect_dm():
    """DM-family dispatch is not altered by PLN additions."""
    assert (
        resolve_user_parameterization_for_model("nbdm", "canonical")
        == "canonical"
    )
