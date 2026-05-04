"""Tests for the three LNM-family parameterization variants.

The LNM family mirrors the DM-family ``canonical`` / ``mean_prob`` /
``mean_odds`` trio: all three variants share the compositional path
(``y_alr``, multinomial, decoder/encoder) but differ in which scalars of
the totals NB submodel are sampled directly versus derived. This test
file verifies the per-variant param specs, derived parameters, and
registry layout.
"""

from __future__ import annotations

import pytest

from scribe.models.builders.parameter_specs import (
    BetaPrimeSpec,
    BetaSpec,
    LogNormalSpec,
    PositiveNormalSpec,
    SigmoidNormalSpec,
)
from scribe.models.config import GuideFamilyConfig
from scribe.models.parameterizations import (
    PARAMETERIZATIONS,
    LogisticNormalParameterization,
    is_logistic_normal_family,
    resolve_user_parameterization_for_model,
)


# ---------------------------------------------------------------------------
# Fixtures: one per variant.
# ---------------------------------------------------------------------------


@pytest.fixture
def lnm_canonical():
    return LogisticNormalParameterization(variant="canonical")


@pytest.fixture
def lnm_mean_prob():
    return LogisticNormalParameterization(variant="mean_prob")


@pytest.fixture
def lnm_mean_odds():
    return LogisticNormalParameterization(variant="mean_odds")


# ---------------------------------------------------------------------------
# Identity / introspection
# ---------------------------------------------------------------------------


def test_name_per_variant(lnm_canonical, lnm_mean_prob, lnm_mean_odds):
    # The ``name`` property feeds the registry; each variant must
    # expose a distinct, deterministic key.
    assert lnm_canonical.name == "logistic_normal_canonical"
    assert lnm_mean_prob.name == "logistic_normal_mean_prob"
    assert lnm_mean_odds.name == "logistic_normal_mean_odds"


def test_variant_property(lnm_canonical, lnm_mean_prob, lnm_mean_odds):
    # ``variant`` is the public-facing accessor a downstream consumer
    # uses to introspect the parameterization choice.
    assert lnm_canonical.variant == "canonical"
    assert lnm_mean_prob.variant == "mean_prob"
    assert lnm_mean_odds.variant == "mean_odds"


def test_invalid_variant_raises():
    # The constructor must reject anything outside the trio.
    with pytest.raises(ValueError):
        LogisticNormalParameterization(variant="bogus")


def test_default_variant_is_canonical():
    # The unconstructed default mirrors the historical LNM behavior so
    # any internal code that instantiates without arguments gets the
    # original semantics.
    p = LogisticNormalParameterization()
    assert p.variant == "canonical"


def test_gene_param_name_unchanged_across_variants(
    lnm_canonical, lnm_mean_prob, lnm_mean_odds
):
    # The compositional path is identical across variants — the gene
    # parameter is always ``y_alr`` regardless of how the totals are
    # parameterized.
    for p in (lnm_canonical, lnm_mean_prob, lnm_mean_odds):
        assert p.gene_param_name == "y_alr"


def test_requires_vae_true_for_all_variants(
    lnm_canonical, lnm_mean_prob, lnm_mean_odds
):
    for p in (lnm_canonical, lnm_mean_prob, lnm_mean_odds):
        assert p.requires_vae is True


def test_decoder_output_spec_unchanged_across_variants(
    lnm_canonical, lnm_mean_prob, lnm_mean_odds
):
    # The decoder spec is part of the compositional path and must be
    # identical across the three variants.
    expected = [("y_alr", "identity")]
    for p in (lnm_canonical, lnm_mean_prob, lnm_mean_odds):
        assert p.decoder_output_spec("lnm") == expected


# ---------------------------------------------------------------------------
# Core parameters per variant
# ---------------------------------------------------------------------------


def test_core_parameters_canonical(lnm_canonical):
    # Historical LNM totals: sample (r_T, p) directly.
    assert lnm_canonical.core_parameters == ["r_T", "p"]


def test_core_parameters_mean_prob(lnm_mean_prob):
    # mean-prob totals: sample (mu_T, p); derive r_T.
    assert lnm_mean_prob.core_parameters == ["mu_T", "p"]


def test_core_parameters_mean_odds(lnm_mean_odds):
    # mean-odds totals: sample (mu_T, phi_T); derive r_T and p.
    assert lnm_mean_odds.core_parameters == ["mu_T", "phi_T"]


# ---------------------------------------------------------------------------
# Param specs (constrained mode)
# ---------------------------------------------------------------------------


def test_build_param_specs_canonical_constrained(lnm_canonical):
    specs = lnm_canonical.build_param_specs(
        unconstrained=False,
        guide_families=GuideFamilyConfig(),
    )
    assert len(specs) == 2
    # r_T uses PositiveNormalSpec so that the factory can route the
    # transform through ``model_config.positive_transform`` (softplus
    # by default), avoiding the LogNormal mode-vs-median trap.
    assert isinstance(specs[0], PositiveNormalSpec) and specs[0].name == "r_T"
    assert isinstance(specs[1], BetaSpec) and specs[1].name == "p"


def test_build_param_specs_mean_prob_constrained(lnm_mean_prob):
    specs = lnm_mean_prob.build_param_specs(
        unconstrained=False,
        guide_families=GuideFamilyConfig(),
    )
    assert len(specs) == 2
    # mu_T uses PositiveNormalSpec — the factory rewrites the transform
    # from ``model_config.positive_transform`` (softplus default).
    assert isinstance(specs[0], PositiveNormalSpec) and specs[0].name == "mu_T"
    # p uses Beta — same as canonical.
    assert isinstance(specs[1], BetaSpec) and specs[1].name == "p"


def test_build_param_specs_mean_odds_constrained(lnm_mean_odds):
    specs = lnm_mean_odds.build_param_specs(
        unconstrained=False,
        guide_families=GuideFamilyConfig(),
    )
    assert len(specs) == 2
    # mu_T: PositiveNormalSpec (positive scalar; transform configured).
    assert isinstance(specs[0], PositiveNormalSpec) and specs[0].name == "mu_T"
    # phi_T: BetaPrime (positive odds-ratio scalar — natural prior with
    # density vanishing at zero, blocking the boundary collapse the
    # LogNormal-with-large-σ prior was prone to).
    assert isinstance(specs[1], BetaPrimeSpec) and specs[1].name == "phi_T"


# ---------------------------------------------------------------------------
# Param specs (unconstrained mode)
# ---------------------------------------------------------------------------


def test_build_param_specs_canonical_unconstrained(lnm_canonical):
    specs = lnm_canonical.build_param_specs(
        unconstrained=True,
        guide_families=GuideFamilyConfig(),
    )
    assert len(specs) == 2
    assert isinstance(specs[0], PositiveNormalSpec) and specs[0].name == "r_T"
    assert isinstance(specs[1], SigmoidNormalSpec) and specs[1].name == "p"


def test_build_param_specs_mean_prob_unconstrained(lnm_mean_prob):
    specs = lnm_mean_prob.build_param_specs(
        unconstrained=True,
        guide_families=GuideFamilyConfig(),
    )
    assert len(specs) == 2
    assert isinstance(specs[0], PositiveNormalSpec) and specs[0].name == "mu_T"
    assert isinstance(specs[1], SigmoidNormalSpec) and specs[1].name == "p"


def test_build_param_specs_mean_odds_unconstrained(lnm_mean_odds):
    specs = lnm_mean_odds.build_param_specs(
        unconstrained=True,
        guide_families=GuideFamilyConfig(),
    )
    assert len(specs) == 2
    assert isinstance(specs[0], PositiveNormalSpec) and specs[0].name == "mu_T"
    assert isinstance(specs[1], PositiveNormalSpec) and specs[1].name == "phi_T"


# ---------------------------------------------------------------------------
# Derived parameters
# ---------------------------------------------------------------------------


def test_derived_params_canonical_empty(lnm_canonical):
    # Both r_T and p are sampled directly; nothing to derive.
    assert lnm_canonical.build_derived_params() == []


def test_derived_params_mean_prob_includes_r_T(lnm_mean_prob):
    derived = lnm_mean_prob.build_derived_params()
    names = {d.name for d in derived}
    assert names == {"r_T"}


def test_derived_params_mean_odds_includes_p_and_r_T(lnm_mean_odds):
    derived = lnm_mean_odds.build_derived_params()
    names = {d.name for d in derived}
    assert names == {"p", "r_T"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_exposes_three_variants():
    # All three internal keys must be present and point at the right
    # singleton; downstream code looks them up by name.
    for key in (
        "logistic_normal_canonical",
        "logistic_normal_mean_prob",
        "logistic_normal_mean_odds",
    ):
        assert key in PARAMETERIZATIONS
        assert isinstance(
            PARAMETERIZATIONS[key], LogisticNormalParameterization
        )


def test_registry_singletons_have_correct_variant():
    # Sanity: the singleton at each key has the matching ``variant``
    # attribute. Catches accidental swap if the registry is ever
    # rewritten.
    assert PARAMETERIZATIONS["logistic_normal_canonical"].variant == "canonical"
    assert PARAMETERIZATIONS["logistic_normal_mean_prob"].variant == "mean_prob"
    assert PARAMETERIZATIONS["logistic_normal_mean_odds"].variant == "mean_odds"


# ---------------------------------------------------------------------------
# Family helper + user-facing dispatch
# ---------------------------------------------------------------------------


def test_is_logistic_normal_family_recognizes_all_variants():
    assert is_logistic_normal_family("logistic_normal_canonical")
    assert is_logistic_normal_family("logistic_normal_mean_prob")
    assert is_logistic_normal_family("logistic_normal_mean_odds")
    # And rejects DM-family / unrelated keys.
    assert not is_logistic_normal_family("canonical")
    assert not is_logistic_normal_family("mean_prob")
    assert not is_logistic_normal_family("nbdm")
    assert not is_logistic_normal_family("")


def test_resolve_user_parameterization_for_lnm():
    # User-facing strings map to the LNM-family internal keys when
    # the model is in the LNM family.
    assert (
        resolve_user_parameterization_for_model("lnm", "canonical")
        == "logistic_normal_canonical"
    )
    assert (
        resolve_user_parameterization_for_model("lnm", "mean_prob")
        == "logistic_normal_mean_prob"
    )
    assert (
        resolve_user_parameterization_for_model("lnm", "mean_odds")
        == "logistic_normal_mean_odds"
    )
    # ``lnmvcp`` follows the same mapping.
    assert (
        resolve_user_parameterization_for_model("lnmvcp", "mean_odds")
        == "logistic_normal_mean_odds"
    )
    # Legacy DM aliases also resolve correctly under LNM.
    assert (
        resolve_user_parameterization_for_model("lnm", "standard")
        == "logistic_normal_canonical"
    )
    assert (
        resolve_user_parameterization_for_model("lnm", "odds_ratio")
        == "logistic_normal_mean_odds"
    )


def test_resolve_user_parameterization_for_dm():
    # DM-family models pass through unchanged.
    assert (
        resolve_user_parameterization_for_model("nbdm", "canonical")
        == "canonical"
    )
    assert (
        resolve_user_parameterization_for_model("nbvcp", "mean_prob")
        == "mean_prob"
    )


def test_resolve_user_parameterization_legacy_logistic_normal_errors():
    # Passing the legacy ``"logistic_normal"`` string for an LNM model
    # must raise — with a clear migration message — rather than
    # silently picking one of the three variants.
    with pytest.raises(ValueError, match="logistic_normal"):
        resolve_user_parameterization_for_model("lnm", "logistic_normal")


def test_resolve_user_parameterization_invalid_raises():
    # Random strings must error rather than silently fall through.
    with pytest.raises(ValueError):
        resolve_user_parameterization_for_model("lnm", "definitely_not_real")
    with pytest.raises(ValueError):
        resolve_user_parameterization_for_model("nbdm", "definitely_not_real")


# ---------------------------------------------------------------------------
# Regression: natural-prior defaults for LNM constrained mode.
#
# These three tests lock in the invariant that the ``unconstrained=False``
# default for LNM uses *natural* priors for bounded parameters and a
# config-driven Normal+positive-transform for positive scalars — never the
# bare ``LogNormalSpec`` that previously sat in this branch.
#
# Background: ``LogNormal(μ, σ)`` has mode at ``exp(μ - σ²)``. For wide
# user-supplied priors (e.g. ``(5.0, 5.0)``) the mode underflows to
# essentially zero in float32, so MAP-based optimization collapses to the
# boundary regardless of the median the user typed. ``BetaPrime`` /
# ``Beta`` have density vanishing at the boundary and don't suffer from
# this trap; ``PositiveNormalSpec`` lets the factory pick an appropriate
# transform (``softplus`` by default, smoother than ``exp``).
# ---------------------------------------------------------------------------


def test_constrained_lnm_uses_betaprime_not_lognormal_for_phi_T(lnm_mean_odds):
    """Mean-odds constrained must use BetaPrime for phi_T, not LogNormal.

    With user prior ``total_odds_ratio=(5.0, 5.0)`` (a natural pick when
    a user thinks in BetaPrime ``α/β``-space), this regression keeps the
    resulting spec a ``BetaPrimeSpec`` so the prior tuple has the
    semantics the user expects: mode at ``(α-1)/(β+1) = 0.67``, density
    ``∝ x^{α-1}`` near zero — actively repelling the boundary collapse
    that bit users under the old ``LogNormalSpec`` branch.
    """
    specs = lnm_mean_odds.build_param_specs(
        unconstrained=False,
        guide_families=GuideFamilyConfig(),
    )
    spec_by_name = {s.name: s for s in specs}
    assert isinstance(spec_by_name["phi_T"], BetaPrimeSpec)
    # And explicitly NOT a LogNormalSpec — the regression we are guarding.
    assert not isinstance(spec_by_name["phi_T"], LogNormalSpec)
    # ``mu_T`` should be PositiveNormalSpec (configurable transform), not
    # LogNormalSpec (which silently fixes the transform to exp).
    assert isinstance(spec_by_name["mu_T"], PositiveNormalSpec)
    assert not isinstance(spec_by_name["mu_T"], LogNormalSpec)


def test_unconstrained_lnm_unchanged_uses_positive_normal_throughout(
    lnm_canonical, lnm_mean_prob, lnm_mean_odds
):
    """Opt-in unconstrained mode must preserve the all-Normal-with-transform
    reparameterization. Joint guides and other downstream consumers depend
    on this — they require vector-space parameter specs because a
    multivariate-Normal guide cannot be defined on a bounded manifold.

    This test guards against accidentally letting the new constrained-mode
    natural-prior change leak across the boundary.
    """
    for strat in (lnm_canonical, lnm_mean_prob, lnm_mean_odds):
        specs = strat.build_param_specs(
            unconstrained=True,
            guide_families=GuideFamilyConfig(),
        )
        # Every positive-valued sampled scalar must remain a
        # PositiveNormalSpec (Normal-in-log-space → transform). No
        # natural-prior leakage.
        positive_names = {s.name for s in specs} & {
            "r_T",
            "mu_T",
            "phi_T",
        }
        for s in specs:
            if s.name in positive_names:
                assert isinstance(s, PositiveNormalSpec), (
                    f"{strat.variant}/{s.name} must be PositiveNormalSpec "
                    f"in unconstrained mode, got {type(s).__name__}"
                )
                # Not BetaPrime — that's reserved for constrained mode.
                assert not isinstance(s, BetaPrimeSpec)


def test_dm_canonical_constrained_still_uses_lognormal_for_r():
    """DM-family scoped invariant: the LNM-only spec change must NOT have
    perturbed the DM family. The DM ``canonical`` parameterization still
    uses ``LogNormalSpec`` for the per-gene dispersion ``r`` because
    DM-family training is well-validated under those defaults and we
    don't want behavioral drift on an unrelated codepath.
    """
    canonical = PARAMETERIZATIONS["canonical"]
    specs = canonical.build_param_specs(
        unconstrained=False,
        guide_families=GuideFamilyConfig(),
    )
    spec_by_name = {s.name: s for s in specs}
    # DM canonical samples (p, r). ``r`` is the per-gene dispersion.
    assert isinstance(spec_by_name["r"], LogNormalSpec), (
        "DM canonical r must remain LogNormalSpec; spec change should be "
        "scoped to the LNM family only."
    )
    assert isinstance(spec_by_name["p"], BetaSpec)
