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


# ----------------------------------------------------------------------
# Posterior reconstruction honours per-parameter transforms
# ----------------------------------------------------------------------


def _leaf_transform(dist_obj):
    """Return the constraint transform of a reconstructed posterior entry.

    Handles both the ``TransformedDistribution`` form (mean-field guides)
    and the ``{"base", "transform"}`` dict form (low-rank guides).
    """
    if isinstance(dist_obj, dict) and "transform" in dist_obj:
        return dist_obj["transform"]
    transforms = getattr(dist_obj, "transforms", None)
    if transforms:
        return transforms[0]
    return None


class TestPosteriorReconstructionTransforms:
    """Reconstruction must mirror the transforms used at fit time.

    Regression for a silent mis-reconstruction: a ``positive_transform``
    dict like ``{"mu": "exp"}`` was collapsed to a *global* ``ExpTransform``
    inside ``get_posterior_distributions`` because the dict had a single
    distinct value.  That re-applied ``exp`` to the softplus-fit ``phi`` and
    ``phi_capture`` when rebuilding the posterior for ``get_map`` — for VCP
    odds-family models ``exp(loc)`` vs ``softplus(loc)`` inflated
    ``phi_capture`` ~10x, halving the reconstructed capture and every
    MAP-based mean (mean-calibration), while the sampling-based PPC, which
    threads the real guide, stayed correct.
    """

    def _nbvcp_mean_odds_params(self, n_genes=6, n_cells=20):
        import numpy as np

        rng = np.random.default_rng(0)
        # Guide locs in the regime where exp(loc) >> softplus(loc) (the
        # bug only bites for loc well above ~1).
        return {
            "phi_loc": rng.normal(3.5, 0.1, size=n_genes).astype("float32"),
            "phi_scale": np.full(n_genes, 0.1, dtype="float32"),
            "mu_loc": rng.normal(0.0, 0.1, size=n_genes).astype("float32"),
            "mu_scale": np.full(n_genes, 0.1, dtype="float32"),
            "phi_capture_loc": rng.normal(
                3.6, 0.1, size=n_cells
            ).astype("float32"),
            "phi_capture_scale": np.full(n_cells, 0.1, dtype="float32"),
        }

    def test_mean_odds_vcp_capture_uses_softplus_not_exp(self):
        """phi/phi_capture stay softplus while mu honours its exp override."""
        import numpyro.distributions as npd

        from scribe.models.config import ModelConfig
        from scribe.models.builders.posterior import (
            get_posterior_distributions,
        )

        cfg = ModelConfig(
            base_model="nbvcp",
            parameterization="mean_odds",
            unconstrained=True,
            positive_transform={"mu": "exp"},
        )
        assert cfg.uses_variable_capture  # sanity: capture pass is active

        dists = get_posterior_distributions(
            self._nbvcp_mean_odds_params(), cfg
        )

        # ``mu`` honours its explicit exp override.
        assert isinstance(
            _leaf_transform(dists["mu"]), npd.transforms.ExpTransform
        )
        # ``phi`` and ``phi_capture`` fall back to the softplus default —
        # NOT the global exp the old collapse produced.
        assert isinstance(
            _leaf_transform(dists["phi"]), npd.transforms.SoftplusTransform
        ), "phi must use the softplus default, not exp"
        assert isinstance(
            _leaf_transform(dists["phi_capture"]),
            npd.transforms.SoftplusTransform,
        ), "phi_capture must use the softplus default, not exp"

    def test_global_exp_string_still_reaches_capture(self):
        """A global ``'exp'`` string must apply exp everywhere (no regression)."""
        import numpyro.distributions as npd

        from scribe.models.config import ModelConfig
        from scribe.models.builders.posterior import (
            get_posterior_distributions,
        )

        cfg = ModelConfig(
            base_model="nbvcp",
            parameterization="mean_odds",
            unconstrained=True,
            positive_transform="exp",
        )
        dists = get_posterior_distributions(
            self._nbvcp_mean_odds_params(), cfg
        )
        for name in ("mu", "phi", "phi_capture"):
            assert isinstance(
                _leaf_transform(dists[name]), npd.transforms.ExpTransform
            ), f"{name} should use exp under global positive_transform='exp'"
