"""Tests for the ``mean_disp`` parameterization.

``mean_disp`` samples the gene mean ``mu`` and NB dispersion ``r`` directly --
the Fisher-orthogonal coordinate (see ``paper/_guide_reparam.qmd``) -- and
derives ``p`` and ``phi``. Both ``mu`` and ``r`` are gene-specific.

Covers: registry/identity, spec construction, derived-param math, shorthand
resolution, config + active-parameter wiring, the prob/VAE guards, the joint
(linear-only and low-rank) guide path, an end-to-end SVI smoke (posterior +
MAP), the SVI->MCMC init branch, and the native ``mu/r`` compositional path
(equivalence to the derived-``p`` path + ``mu/sum(mu)`` mean composition).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from scribe.models.parameterizations import (
    PARAMETERIZATIONS,
    MeanDispParameterization,
    _compute_p_from_mu_r,
    _compute_phi_from_mu_r,
)
from scribe.models.config.enums import Parameterization
from scribe.inference.preset_builder import build_config_from_preset


# ==============================================================================
# Registry / identity
# ==============================================================================


class TestMeanDispRegistry:
    def test_registry_contains_mean_disp(self):
        assert "mean_disp" in PARAMETERIZATIONS
        assert isinstance(
            PARAMETERIZATIONS["mean_disp"], MeanDispParameterization
        )

    def test_core_parameters_and_gene_param(self):
        md = PARAMETERIZATIONS["mean_disp"]
        assert md.name == "mean_disp"
        assert md.core_parameters == ["mu", "r"]
        assert md.gene_param_name == "mu"

    def test_enum_resolves(self):
        assert Parameterization("mean_disp") is Parameterization.MEAN_DISP


# ==============================================================================
# Spec construction
# ==============================================================================


class TestMeanDispSpecs:
    def _families(self):
        from scribe.models.config import GuideFamilyConfig

        return GuideFamilyConfig()

    def test_unconstrained_specs_both_gene_specific(self):
        from scribe.models.builders import PositiveNormalSpec

        specs = PARAMETERIZATIONS["mean_disp"].build_param_specs(
            unconstrained=True, guide_families=self._families()
        )
        by_name = {s.name: s for s in specs}
        assert set(by_name) == {"mu", "r"}
        for name in ("mu", "r"):
            assert isinstance(by_name[name], PositiveNormalSpec)
            assert by_name[name].is_gene_specific
            assert by_name[name].shape_dims == ("n_genes",)

    def test_constrained_specs_lognormal(self):
        from scribe.models.builders import LogNormalSpec

        specs = PARAMETERIZATIONS["mean_disp"].build_param_specs(
            unconstrained=False, guide_families=self._families()
        )
        by_name = {s.name: s for s in specs}
        assert set(by_name) == {"mu", "r"}
        assert all(isinstance(s, LogNormalSpec) for s in specs)

    def test_mixture_marks_both(self):
        specs = PARAMETERIZATIONS["mean_disp"].build_param_specs(
            unconstrained=True,
            guide_families=self._families(),
            n_components=3,
        )
        by_name = {s.name: s for s in specs}
        assert by_name["mu"].is_mixture
        assert by_name["r"].is_mixture


# ==============================================================================
# Derived parameters
# ==============================================================================


class TestMeanDispDerived:
    def test_derived_defs(self):
        derived = PARAMETERIZATIONS["mean_disp"].build_derived_params()
        by_name = {d.name: d for d in derived}
        assert set(by_name) == {"phi", "p"}
        assert by_name["phi"].deps == ["mu", "r"]
        assert by_name["p"].deps == ["mu", "r"]

    def test_phi_and_p_math(self):
        mu = jnp.array([2.0, 8.0, 50.0])
        r = jnp.array([4.0, 4.0, 10.0])
        np.testing.assert_allclose(
            np.asarray(_compute_phi_from_mu_r(mu, r)), np.asarray(r / mu)
        )
        np.testing.assert_allclose(
            np.asarray(_compute_p_from_mu_r(mu, r)),
            np.asarray(mu / (mu + r)),
        )

    def test_round_trip_with_mean_odds(self):
        # With phi = r/mu, mean_disp's p = mu/(mu+r) must equal mean_odds'
        # p = 1/(1+phi).
        mu = jnp.array([1.0, 5.0, 20.0])
        r = jnp.array([3.0, 2.0, 40.0])
        phi = _compute_phi_from_mu_r(mu, r)
        np.testing.assert_allclose(
            np.asarray(_compute_p_from_mu_r(mu, r)),
            np.asarray(1.0 / (1.0 + phi)),
            rtol=1e-6,
        )


# ==============================================================================
# Shorthand resolution
# ==============================================================================


class TestMeanDispShorthand:
    def _resolve(self, value, model="nbdm"):
        from scribe.models.config.parameter_mapping import (
            resolve_param_shorthand,
        )

        return resolve_param_shorthand(
            value, PARAMETERIZATIONS["mean_disp"], model
        )

    def test_mean_and_biological(self):
        assert self._resolve("mean") == ["mu"]
        assert set(self._resolve("biological")) == {"mu", "r"}

    def test_prob_is_r_documented_wart(self):
        # "prob" resolves to the non-gene-param core member, which is "r"
        # here -- a documented wart for mean_disp.
        assert self._resolve("prob") == ["r"]

    def test_all_includes_gate_for_zinb(self):
        assert set(self._resolve("all", model="zinb")) == {"mu", "r", "gate"}


# ==============================================================================
# Config + active parameters
# ==============================================================================


class TestMeanDispConfig:
    @pytest.mark.parametrize("model", ["nbdm", "zinb", "nbvcp"])
    def test_config_builds(self, model):
        cfg = build_config_from_preset(
            model=model,
            parameterization="mean_disp",
            inference_method="svi",
        )
        assert cfg.parameterization == Parameterization.MEAN_DISP

    def test_active_parameters(self):
        from scribe.models.config.parameter_mapping import (
            get_active_parameters,
        )

        active = get_active_parameters(Parameterization.MEAN_DISP, "nbdm")
        assert {"mu", "r"}.issubset(active)

    def test_expression_prior_adds_log_mu_hypers(self):
        from scribe.models.config.parameter_mapping import (
            get_active_parameters,
        )

        active = get_active_parameters(
            Parameterization.MEAN_DISP, "nbdm", hierarchical_mu=True
        )
        assert {"log_mu_loc", "log_mu_scale"}.issubset(active)


# ==============================================================================
# Guards
# ==============================================================================


class TestMeanDispGuards:
    def test_prob_prior_rejected(self):
        with pytest.raises(ValueError, match="mean_disp"):
            build_config_from_preset(
                model="nbvcp",
                parameterization="mean_disp",
                prob_prior="gaussian",
            )

    def test_prob_dataset_prior_rejected(self):
        with pytest.raises(ValueError, match="mean_disp"):
            build_config_from_preset(
                model="nbvcp",
                parameterization="mean_disp",
                n_datasets=2,
                prob_dataset_prior="gaussian",
            )

    def test_vae_rejected_preset_path(self):
        with pytest.raises(ValueError, match="svi and mcmc"):
            build_config_from_preset(
                model="nbdm",
                parameterization="mean_disp",
                inference_method="vae",
            )

    def test_vae_rejected_config_validator(self):
        # Direct builder path must also reject VAE.
        from scribe.models.config import ModelConfigBuilder, ModelType
        from scribe.models.config.enums import InferenceMethod

        builder = (
            ModelConfigBuilder()
            .for_model(ModelType.NBDM)
            .with_parameterization("mean_disp")
        )
        builder._inference_method = InferenceMethod.VAE
        with pytest.raises(ValueError, match="svi and mcmc"):
            builder.build()


# ==============================================================================
# Joint guide (linear-only and low-rank) on (mu, r)
# ==============================================================================


class TestMeanDispJointGuide:
    def test_linear_only_joint_no_rank(self):
        cfg = build_config_from_preset(
            model="nbdm",
            parameterization="mean_disp",
            joint_params=["mu", "r"],
        )
        from scribe.models.components import JointLowRankGuide

        g = cfg.guide_families.get("mu")
        assert isinstance(g, JointLowRankGuide)
        assert g.rank == 0
        assert g.dense_params == []
        # Both share the same marker object.
        assert cfg.guide_families.get("r") is g

    def test_low_rank_joint_with_rank(self):
        cfg = build_config_from_preset(
            model="nbdm",
            parameterization="mean_disp",
            guide_rank=6,
            joint_params=["mu", "r"],
        )
        from scribe.models.components import JointLowRankGuide

        g = cfg.guide_families.get("mu")
        assert isinstance(g, JointLowRankGuide)
        assert g.rank == 6


# ==============================================================================
# End-to-end SVI smoke
# ==============================================================================


class TestMeanDispSVISmoke:
    @pytest.fixture(scope="class")
    def fitted(self):
        import scribe

        rng = np.random.default_rng(0)
        counts = rng.poisson(4.0, size=(60, 8)).astype(np.int32)
        return scribe.fit(
            counts,
            variable_capture=True,
            parameterization="mean_disp",
            inference_method="svi",
            n_steps=30,
        )

    def test_posterior_exposes_all_four(self, fitted):
        s = fitted.get_posterior_samples(n_samples=16, store_samples=False)
        for k in ("mu", "r", "p", "phi"):
            assert k in s
        assert np.asarray(s["r"]).shape[-1] == 8  # gene axis

    def test_get_map_derives_p(self, fitted):
        m = fitted.get_map()
        for k in ("mu", "r", "p", "phi"):
            assert k in m
        mu, r, p = (np.asarray(m[k]) for k in ("mu", "r", "p"))
        np.testing.assert_allclose(p, mu / (mu + r), atol=1e-4)

    def test_get_map_targeted_p(self, fitted):
        # Selective extraction of p must pull mu, r as parents.
        m = fitted.get_map(targets=["p"])
        assert "p" in m


# ==============================================================================
# SVI -> MCMC init branch
# ==============================================================================


class TestMeanDispMCMCInit:
    def test_init_seeds_mu_and_r(self):
        from scribe.mcmc._init_from_svi import compute_init_values

        # Fake SVI canonical MAP (p, r); target = mean_disp config.
        svi_map = {
            "p": jnp.array([0.3, 0.5, 0.7]),
            "r": jnp.array([2.0, 4.0, 6.0]),
        }
        cfg = build_config_from_preset(
            model="nbdm",
            parameterization="mean_disp",
            inference_method="mcmc",
        )
        init = compute_init_values(svi_map, cfg)
        assert "mu" in init and "r" in init
        # mu = r * p / (1 - p)
        p = np.asarray(svi_map["p"])
        r = np.asarray(svi_map["r"])
        np.testing.assert_allclose(
            np.asarray(init["mu"]), r * p / (1.0 - p), rtol=1e-5
        )


# ==============================================================================
# Native (mu, r) compositional path
# ==============================================================================


class TestMeanDispCompositionNative:
    def test_native_matches_p_path_moderate(self):
        from scribe.de._empirical import sample_composition

        N, D = 400, 6
        r = np.full((N, D), 5.0, np.float32)
        mu = (
            np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], np.float32)[None, :]
            .repeat(N, 0)
        )
        p = (mu / (mu + r)).astype(np.float32)
        scale = (mu / r).astype(np.float32)
        key = jax.random.PRNGKey(0)
        s_p = sample_composition(r, p_samples=p, rng_key=key)
        s_sc = sample_composition(r, scale_samples=scale, rng_key=key)
        np.testing.assert_allclose(s_p.mean(0), s_sc.mean(0), atol=3e-3)

    def test_mean_composition_is_mu_over_sum_mu(self):
        # The composition mean tracks mu/sum(mu), NOT (mu/r)/sum(mu/r).
        from scribe.de._empirical import sample_composition

        N, D = 600, 6
        r = np.full((N, D), 5.0, np.float32)
        mu_vec = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], np.float32)
        mu = mu_vec[None, :].repeat(N, 0)
        scale = (mu / r).astype(np.float32)
        s = sample_composition(
            r, scale_samples=scale, rng_key=jax.random.PRNGKey(1)
        )
        np.testing.assert_allclose(
            s.mean(0), mu_vec / mu_vec.sum(), atol=2e-2
        )

    def test_de_compare_uses_native_path(self):
        import scribe

        rng = np.random.default_rng(2)
        counts = rng.poisson(5.0, size=(80, 10)).astype(np.int32)
        res = scribe.fit(
            counts,
            variable_capture=True,
            parameterization="mean_disp",
            n_components=2,
            inference_method="svi",
            n_steps=30,
        )
        res.get_posterior_samples(n_samples=200, convert_to_numpy=True)
        de = scribe.de.compare(
            model_A=res.get_component(0), model_B=res.get_component(1)
        )
        df = de.set_expression_threshold(1.0).to_dataframe(
            tau=float(np.log(2)), target_pefp=0.05, metrics="clr"
        )
        assert "clr_delta_mean" in df.columns
        assert len(df) == 10


# ==============================================================================
# Native (mu, r) logits VCP likelihood path
# ==============================================================================


class TestMeanDispNativeLogitsPath:
    """mean_disp routes the variable-capture likelihood through the cheap
    NegativeBinomialLogits kernel (logits = log mu - log r + log p_capture),
    never building the rational p_hat. This avoids the ~6x slowdown of the
    NegativeBinomialProbs path that canonical/mean_prob use under capture.
    """

    def _counts_base_dist(self, parameterization, **fit_kwargs):
        import jax
        from numpyro import handlers
        from scribe.models import get_model_and_guide

        cfg = build_config_from_preset(
            model="nbvcp",
            parameterization=parameterization,
            inference_method="svi",
            **fit_kwargs,
        )
        model, _guide, cfg2 = get_model_and_guide(cfg)
        counts = jnp.asarray(
            np.random.default_rng(0).poisson(3.0, (40, 8)).astype("float32")
        )
        seeded = handlers.seed(model, jax.random.PRNGKey(0))
        tr = handlers.trace(seeded).get_trace(
            n_cells=40, n_genes=8, model_config=cfg2, counts=counts
        )
        fn = tr["counts"]["fn"]
        return getattr(fn, "base_dist", fn)

    def test_mean_disp_vcp_uses_logits(self):
        base = self._counts_base_dist("mean_disp")
        assert "Logits" in type(base).__name__

    def test_canonical_vcp_uses_probs(self):
        # Contrast: the canonical p-path keeps NegativeBinomialProbs (the
        # rational p_hat path). Confirms the native branch is mean_disp-only.
        base = self._counts_base_dist("canonical")
        assert "Probs" in type(base).__name__
