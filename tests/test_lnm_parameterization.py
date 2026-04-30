"""Tests for LogisticNormalParameterization."""

from __future__ import annotations

import pytest

from scribe.models.builders.parameter_specs import (
    BetaSpec,
    LogNormalSpec,
    PositiveNormalSpec,
    SigmoidNormalSpec,
)
from scribe.models.config import GuideFamilyConfig
from scribe.models.parameterizations import (
    PARAMETERIZATIONS,
    LogisticNormalParameterization,
)


@pytest.fixture
def logistic_normal_param():
    """Fresh parameterization instance for introspection tests."""
    return LogisticNormalParameterization()


def test_name(logistic_normal_param):
    """Registry name is ``logistic_normal``."""
    assert logistic_normal_param.name == "logistic_normal"


def test_core_parameters(logistic_normal_param):
    """Core globals are NB total parameters only."""
    assert logistic_normal_param.core_parameters == ["r_T", "p"]


def test_gene_param_name(logistic_normal_param):
    """Decoder gene head is ALR coordinates ``y_alr``."""
    assert logistic_normal_param.gene_param_name == "y_alr"


def test_requires_vae(logistic_normal_param):
    """LNM preset always uses a VAE decoder."""
    assert logistic_normal_param.requires_vae is True


def test_build_param_specs_constrained(logistic_normal_param):
    """Constrained mode uses LogNormal + Beta for population totals."""
    specs = logistic_normal_param.build_param_specs(
        unconstrained=False,
        guide_families=GuideFamilyConfig(),
    )
    assert len(specs) == 2
    assert isinstance(specs[0], LogNormalSpec) and specs[0].name == "r_T"
    assert isinstance(specs[1], BetaSpec) and specs[1].name == "p"


def test_build_param_specs_unconstrained(logistic_normal_param):
    """Unconstrained mode uses Normal reparameterizations on the reals."""
    specs = logistic_normal_param.build_param_specs(
        unconstrained=True,
        guide_families=GuideFamilyConfig(),
    )
    assert len(specs) == 2
    assert isinstance(specs[0], PositiveNormalSpec) and specs[0].name == "r_T"
    assert isinstance(specs[1], SigmoidNormalSpec) and specs[1].name == "p"


def test_decoder_output_spec(logistic_normal_param):
    """Decoder exposes a single linear ALR head."""
    assert logistic_normal_param.decoder_output_spec("lnm") == [
        ("y_alr", "identity")
    ]


def test_build_derived_params_empty(logistic_normal_param):
    """No closed-form derived parameters are registered."""
    assert logistic_normal_param.build_derived_params() == []


def test_registered_in_parameterizations():
    """Singleton registry exposes ``logistic_normal``."""
    assert "logistic_normal" in PARAMETERIZATIONS
    assert isinstance(
        PARAMETERIZATIONS["logistic_normal"], LogisticNormalParameterization
    )
