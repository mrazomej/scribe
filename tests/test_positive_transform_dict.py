"""Tests for the per-parameter ``positive_transform`` mechanism.

The ``ModelConfig.positive_transform`` field accepts two forms:

1. A string (``"softplus"`` or ``"exp"``) — global default, the same
   transform is used for every positive parameter.
2. A ``Dict[str, str]`` — per-parameter override.  Listed parameters use
   the specified transform; every other positive parameter falls back
   to ``"softplus"``.

Dict keys may be either *internal* names (``"mu"``, ``"burst_size"``,
``"k_off"``, ``"p_capture"``, ...) or the *descriptive aliases*
registered in ``parameter_mapping.py`` (``"mean_expression"``,
``"capture_prob"``, ``"capture_efficiency"``, ...).  The validator
normalizes descriptive aliases to internal names so factory code that
queries ``resolve_positive_transform("mu")`` finds entries originally
keyed as ``"mean_expression"``.

These tests pin down the contract of the validator + resolver, which is
the foundation that the factory and posterior reconstruction depend on.
"""

from __future__ import annotations

import pytest

from scribe.models.config import ModelConfig


# ----------------------------------------------------------------------
# String form (legacy / backwards compat)
# ----------------------------------------------------------------------


class TestStringForm:
    """The original single-string behavior must remain unchanged."""

    def test_softplus_string(self):
        cfg = ModelConfig(
            base_model="nbdm", positive_transform="softplus"
        )
        assert cfg.positive_transform == "softplus"
        # Every parameter resolves to the same transform.
        assert cfg.resolve_positive_transform("mu") == "softplus"
        assert cfg.resolve_positive_transform("r") == "softplus"
        assert cfg.resolve_positive_transform("burst_size") == "softplus"

    def test_exp_string(self):
        cfg = ModelConfig(base_model="nbdm", positive_transform="exp")
        assert cfg.positive_transform == "exp"
        assert cfg.resolve_positive_transform("mu") == "exp"
        assert cfg.resolve_positive_transform("anything") == "exp"

    def test_invalid_string_rejected(self):
        with pytest.raises(ValueError, match="positive_transform must be"):
            ModelConfig(base_model="nbdm", positive_transform="sigmoid")


# ----------------------------------------------------------------------
# Dict form — internal names
# ----------------------------------------------------------------------


class TestDictFormInternalNames:
    """Internal parameter names are accepted verbatim."""

    def test_single_override_internal(self):
        cfg = ModelConfig(
            base_model="twostatevcp",
            parameterization="two_state_natural",
            positive_transform={"mu": "exp"},
        )
        # Listed parameter uses the override.
        assert cfg.resolve_positive_transform("mu") == "exp"
        # Unlisted parameters fall back to softplus.
        assert cfg.resolve_positive_transform("burst_size") == "softplus"
        assert cfg.resolve_positive_transform("k_off") == "softplus"
        assert cfg.resolve_positive_transform("p_capture") == "softplus"

    def test_multiple_overrides(self):
        cfg = ModelConfig(
            base_model="twostatevcp",
            parameterization="two_state_natural",
            positive_transform={
                "mu": "exp",
                "burst_size": "exp",
                "k_off": "softplus",
            },
        )
        assert cfg.resolve_positive_transform("mu") == "exp"
        assert cfg.resolve_positive_transform("burst_size") == "exp"
        assert cfg.resolve_positive_transform("k_off") == "softplus"
        # Unlisted parameter still falls back.
        assert cfg.resolve_positive_transform("p_capture") == "softplus"

    def test_empty_dict_falls_back_to_softplus_default(self):
        cfg = ModelConfig(
            base_model="nbdm",
            positive_transform={},
        )
        # Every parameter falls back to the implicit "softplus" default.
        assert cfg.resolve_positive_transform("mu") == "softplus"
        assert cfg.resolve_positive_transform("r") == "softplus"


# ----------------------------------------------------------------------
# Dict form — descriptive aliases
# ----------------------------------------------------------------------


class TestDictFormDescriptiveAliases:
    """Descriptive aliases must be normalized to internal names."""

    def test_mean_expression_alias_normalizes_to_mu(self):
        cfg = ModelConfig(
            base_model="twostatevcp",
            parameterization="two_state_natural",
            positive_transform={"mean_expression": "exp"},
        )
        # The validator normalizes "mean_expression" -> "mu" so that
        # downstream factory code that queries the internal name "mu"
        # finds the override.
        assert cfg.resolve_positive_transform("mu") == "exp"
        # The original alias key should NOT survive on the dict.
        assert "mean_expression" not in cfg.positive_transform
        assert "mu" in cfg.positive_transform

    def test_capture_prob_alias_normalizes_to_p_capture(self):
        cfg = ModelConfig(
            base_model="nbvcp",
            positive_transform={"capture_prob": "exp"},
        )
        assert cfg.resolve_positive_transform("p_capture") == "exp"
        assert "p_capture" in cfg.positive_transform
        assert "capture_prob" not in cfg.positive_transform

    def test_dispersion_alias_normalizes_to_r(self):
        cfg = ModelConfig(
            base_model="nbdm",
            positive_transform={"dispersion": "exp"},
        )
        assert cfg.resolve_positive_transform("r") == "exp"

    def test_descriptive_and_internal_can_mix(self):
        cfg = ModelConfig(
            base_model="twostatevcp",
            parameterization="two_state_natural",
            positive_transform={
                "mean_expression": "exp",  # alias -> mu
                "k_off": "exp",            # already internal
            },
        )
        assert cfg.resolve_positive_transform("mu") == "exp"
        assert cfg.resolve_positive_transform("k_off") == "exp"
        assert cfg.resolve_positive_transform("burst_size") == "softplus"


# ----------------------------------------------------------------------
# Validation errors
# ----------------------------------------------------------------------


class TestValidation:
    """Invalid inputs raise clear errors at config-build time."""

    def test_invalid_value_in_dict_rejected(self):
        with pytest.raises(ValueError, match="positive_transform"):
            ModelConfig(
                base_model="nbdm",
                positive_transform={"mu": "sigmoid"},
            )

    def test_non_string_non_dict_rejected(self):
        # Pydantic schema rejects values that match neither variant of
        # the Union.  The specific exception class is Pydantic's
        # ValidationError; both forms (the model_validator and the
        # type-coercion guard) raise something that subclasses Exception
        # with a useful message.
        with pytest.raises(Exception):
            ModelConfig(base_model="nbdm", positive_transform=42)


# ----------------------------------------------------------------------
# End-to-end: factory honours per-parameter overrides
# ----------------------------------------------------------------------


class TestFactoryWiring:
    """The factory should honour the dict form when building specs."""

    def test_twostate_natural_mu_exp_burst_softplus(self):
        """TwoState with ``{"mu": "exp"}`` instantiates the right specs.

        After ``create_model`` runs, the ``mu`` param spec should use
        ``ExpTransform`` while ``burst_size`` and ``k_off`` keep
        ``SoftplusTransform`` (the default).  This pins down the
        factory's ``_pos_transform_for_name`` per-spec resolution.
        """
        import numpyro.distributions as npd

        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import Parameterization
        from scribe.models.presets.factory import create_model

        cfg = ModelConfig(
            base_model="twostate",
            parameterization=Parameterization.TWO_STATE_NATURAL,
            unconstrained=True,
            positive_transform={"mu": "exp"},
        )
        # ``create_model`` writes the resolved param_specs back onto
        # the config; introspect them for the per-name transform.
        _, _, param_specs = create_model(cfg)
        specs_by_name = {s.name: s for s in param_specs}

        def _transform_of(spec):
            return getattr(spec, "transform", None)

        mu_tf = _transform_of(specs_by_name["mu"])
        burst_tf = _transform_of(specs_by_name["burst_size"])
        k_off_tf = _transform_of(specs_by_name["k_off"])

        assert isinstance(mu_tf, npd.transforms.ExpTransform), (
            f"mu should use ExpTransform under positive_transform="
            f"{{'mu': 'exp'}}, got {type(mu_tf).__name__}"
        )
        assert isinstance(burst_tf, npd.transforms.SoftplusTransform), (
            "burst_size should fall back to SoftplusTransform"
        )
        assert isinstance(k_off_tf, npd.transforms.SoftplusTransform), (
            "k_off should fall back to SoftplusTransform"
        )

    def test_twostate_global_exp_string(self):
        """The legacy global-string form should still propagate to every spec."""
        import numpyro.distributions as npd

        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import Parameterization
        from scribe.models.presets.factory import create_model

        cfg = ModelConfig(
            base_model="twostate",
            parameterization=Parameterization.TWO_STATE_NATURAL,
            unconstrained=True,
            positive_transform="exp",
        )
        _, _, param_specs = create_model(cfg)
        specs_by_name = {s.name: s for s in param_specs}
        for name in ("mu", "burst_size", "k_off"):
            tf = getattr(specs_by_name[name], "transform", None)
            assert isinstance(tf, npd.transforms.ExpTransform), (
                f"{name} should use ExpTransform under global "
                f"positive_transform='exp', got {type(tf).__name__}"
            )
