"""Tests for TwoState mixture model support.

Covers:
1. Config validation: n_components >= 2 is allowed for TwoState.
2. Parameterization: ``build_param_specs`` sets ``is_mixture`` on mu.
3. Factory: ``_validate_mixture_params`` accepts TwoState extras after
   name rewriting, and ``_resolve_twostate_extra_names`` correctly
   rewrites for all four parameterizations.
4. Likelihood: ``TwoStateLikelihood`` and ``TwoStateVCPLikelihood``
   produce correct trace shapes in the mixture path.
5. Log-prob: ``twostate_log_prob`` handles the mixture branch.
6. Posterior: ``_build_two_state_posteriors`` forwards ``is_mixture``.
7. Denoising: ``_denoise_twostate_mixture_marginal`` marginalizes
   correctly over components.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

from scribe.models.components.likelihoods import (
    TwoStateLikelihood,
    TwoStateVCPLikelihood,
)
from scribe.models.components.likelihoods._log_prob import twostate_log_prob
from scribe.models.components.likelihoods.two_state import (
    _twostate_reparam,
)
from scribe.models.parameterizations import (
    TwoStateMeanFanoParameterization,
    TwoStateMomentDeltaParameterization,
    TwoStateParameterization,
    TwoStateRatioParameterization,
)
from scribe.models.presets.factory import _resolve_twostate_extra_names

# Reusable constants
N_CELLS = 12
N_GENES = 4
N_COMPONENTS = 3
RNG = np.random.default_rng(42)
COUNTS = jnp.asarray(
    RNG.integers(0, 10, size=(N_CELLS, N_GENES)), dtype=jnp.int32
)


# ==============================================================================
# Config validation — n_components >= 2 is allowed
# ==============================================================================


class TestConfigAllowsMixtures:
    """Phase-1 mixture rejection was removed; configs should validate."""

    def test_twostate_mixture_config_validates(self):
        """ModelConfig with base_model='twostate' and n_components=3
        should not raise a ValueError."""
        from scribe.models.config.builder import ModelConfigBuilder

        cfg = (
            ModelConfigBuilder()
            .for_model("twostate")
            .with_parameterization("two_state_natural")
            .as_mixture(n_components=3)
            .build()
        )
        assert cfg.n_components == 3

    def test_twostatevcp_mixture_config_validates(self):
        """Same for the VCP variant."""
        from scribe.models.config.builder import ModelConfigBuilder

        cfg = (
            ModelConfigBuilder()
            .for_model("twostatevcp")
            .with_parameterization("two_state_ratio")
            .as_mixture(n_components=2)
            .build()
        )
        assert cfg.n_components == 2


# ==============================================================================
# Parameterization — build_param_specs honours n_components/mixture_params
# ==============================================================================


class TestParamSpecsMixture:
    """All four TwoState parameterizations set is_mixture on mu."""

    @pytest.mark.parametrize(
        "cls",
        [
            TwoStateParameterization,
            TwoStateRatioParameterization,
            TwoStateMeanFanoParameterization,
            TwoStateMomentDeltaParameterization,
        ],
    )
    def test_is_mixture_true_when_n_components_set(self, cls):
        """With n_components > 1 and no mixture_params, mu is mixture."""
        strategy = cls()
        specs = strategy.build_param_specs(
            unconstrained=True,
            guide_families={},
            n_components=3,
            mixture_params=None,
        )
        assert len(specs) == 1
        assert specs[0].name == "mu"
        assert specs[0].is_mixture is True

    @pytest.mark.parametrize(
        "cls",
        [
            TwoStateParameterization,
            TwoStateRatioParameterization,
            TwoStateMeanFanoParameterization,
            TwoStateMomentDeltaParameterization,
        ],
    )
    def test_is_mixture_false_when_excluded(self, cls):
        """With mixture_params=[] (explicitly empty), mu is NOT mixture."""
        strategy = cls()
        specs = strategy.build_param_specs(
            unconstrained=True,
            guide_families={},
            n_components=3,
            mixture_params=[],
        )
        assert specs[0].is_mixture is False

    @pytest.mark.parametrize(
        "cls",
        [
            TwoStateParameterization,
            TwoStateRatioParameterization,
            TwoStateMeanFanoParameterization,
            TwoStateMomentDeltaParameterization,
        ],
    )
    def test_is_mixture_false_when_single_component(self, cls):
        """With n_components=1, mu is NOT mixture."""
        strategy = cls()
        specs = strategy.build_param_specs(
            unconstrained=False,
            guide_families={},
            n_components=1,
        )
        assert specs[0].is_mixture is False


# ==============================================================================
# Factory helper — _resolve_twostate_extra_names
# ==============================================================================


class TestResolveTwoStateExtraNames:
    """The shared helper rewrites extras for alternate parameterizations."""

    def test_natural_passthrough(self):
        """Natural parameterization keeps (burst_size, k_off)."""
        from scribe.models.config.enums import Parameterization as P

        result = _resolve_twostate_extra_names(
            "twostate", P.TWO_STATE_NATURAL, ["burst_size", "k_off"]
        )
        assert result == ["burst_size", "k_off"]

    def test_ratio_rewrite(self):
        """Ratio replaces k_off with switching_ratio."""
        from scribe.models.config.enums import Parameterization as P

        result = _resolve_twostate_extra_names(
            "twostate", P.TWO_STATE_RATIO, ["burst_size", "k_off"]
        )
        assert result == ["burst_size", "switching_ratio"]

    def test_mean_fano_rewrite(self):
        """Mean-Fano replaces (burst_size, k_off) with
        (excess_fano, concentration)."""
        from scribe.models.config.enums import Parameterization as P

        result = _resolve_twostate_extra_names(
            "twostate",
            P.TWO_STATE_MEAN_FANO,
            ["burst_size", "k_off"],
        )
        assert result == ["excess_fano", "concentration"]

    def test_moment_delta_rewrite(self):
        """Moment-delta replaces (burst_size, k_off) with
        (excess_fano, inv_concentration)."""
        from scribe.models.config.enums import Parameterization as P

        result = _resolve_twostate_extra_names(
            "twostate",
            P.TWO_STATE_MOMENT_DELTA,
            ["burst_size", "k_off"],
        )
        assert result == ["excess_fano", "inv_concentration"]

    def test_vcp_preserves_p_capture(self):
        """VCP extras include p_capture which must be preserved."""
        from scribe.models.config.enums import Parameterization as P

        result = _resolve_twostate_extra_names(
            "twostatevcp",
            P.TWO_STATE_RATIO,
            ["burst_size", "k_off", "p_capture"],
        )
        assert result == ["burst_size", "switching_ratio", "p_capture"]

    def test_non_twostate_passthrough(self):
        """Non-TwoState models pass through unchanged."""
        from scribe.models.config.enums import Parameterization as P

        result = _resolve_twostate_extra_names(
            "nbdm", P.CANONICAL, ["p_capture"]
        )
        assert result == ["p_capture"]


# ==============================================================================
# Likelihood — mixture trace shapes
# ==============================================================================


class TestTwoStateMixtureLikelihood:
    """TwoStateLikelihood produces correct shapes in the mixture path."""

    def _make_model(self, n_components, batch_size, counts):
        """Build a minimal numpyro model with K-component TwoState."""
        n_cells, n_genes = N_CELLS, N_GENES
        like = TwoStateLikelihood()

        def model(_counts=counts):
            # Per-component, per-gene parameters
            mu = numpyro.sample(
                "mu",
                dist.LogNormal(
                    jnp.zeros((n_components, n_genes)), 1.0
                ).to_event(2),
            )
            burst_size = numpyro.sample(
                "burst_size",
                dist.LogNormal(
                    jnp.zeros((n_components, n_genes)), 1.0
                ).to_event(2),
            )
            k_off = numpyro.sample(
                "k_off",
                dist.LogNormal(
                    jnp.ones((n_components, n_genes)), 1.0
                ).to_event(2),
            )
            # Uniform mixing weights
            mixing_weights = jnp.ones(n_components) / n_components

            like.sample(
                {
                    "mu": mu,
                    "burst_size": burst_size,
                    "k_off": k_off,
                    "mixing_weights": mixing_weights,
                },
                cell_specs=[],
                counts=_counts,
                dims={"n_cells": n_cells, "n_genes": n_genes},
                batch_size=batch_size,
                model_config=None,
            )

        return model

    def test_prior_predictive_shape(self):
        """Prior predictive produces (n_cells, n_genes)."""
        model = self._make_model(N_COMPONENTS, None, None)
        tr = trace(seed(model, jax.random.PRNGKey(0))).get_trace()
        assert tr["counts"]["value"].shape == (N_CELLS, N_GENES)

    def test_full_conditioning_shape(self):
        """Full conditioning returns observed counts."""
        model = self._make_model(N_COMPONENTS, None, COUNTS)
        tr = trace(seed(model, jax.random.PRNGKey(1))).get_trace()
        np.testing.assert_array_equal(
            np.asarray(tr["counts"]["value"]), np.asarray(COUNTS)
        )

    def test_deterministics_component_shape(self):
        """Gene-level deterministics have (K, G) shape for mixtures."""
        model = self._make_model(N_COMPONENTS, None, None)
        tr = trace(seed(model, jax.random.PRNGKey(2))).get_trace()
        assert tr["alpha"]["value"].shape == (N_COMPONENTS, N_GENES)
        assert tr["beta"]["value"].shape == (N_COMPONENTS, N_GENES)
        assert tr["r_hat"]["value"].shape == (N_COMPONENTS, N_GENES)


class TestTwoStateVCPMixtureLikelihood:
    """TwoStateVCPLikelihood with mixture support."""

    def _make_model(self, n_components, batch_size, counts):
        n_cells, n_genes = N_CELLS, N_GENES
        like = TwoStateVCPLikelihood()

        class _Config:
            param_specs = []

        def model(_counts=counts):
            mu = numpyro.sample(
                "mu",
                dist.LogNormal(
                    jnp.zeros((n_components, n_genes)), 1.0
                ).to_event(2),
            )
            burst_size = numpyro.sample(
                "burst_size",
                dist.LogNormal(
                    jnp.zeros((n_components, n_genes)), 1.0
                ).to_event(2),
            )
            k_off = numpyro.sample(
                "k_off",
                dist.LogNormal(
                    jnp.ones((n_components, n_genes)), 1.0
                ).to_event(2),
            )
            mixing_weights = jnp.ones(n_components) / n_components

            like.sample(
                {
                    "mu": mu,
                    "burst_size": burst_size,
                    "k_off": k_off,
                    "mixing_weights": mixing_weights,
                },
                cell_specs=[],
                counts=_counts,
                dims={"n_cells": n_cells, "n_genes": n_genes},
                batch_size=batch_size,
                model_config=_Config(),
            )

        return model

    def test_prior_predictive_shape(self):
        """VCP mixture prior predictive has correct shape."""
        model = self._make_model(N_COMPONENTS, None, None)
        tr = trace(seed(model, jax.random.PRNGKey(0))).get_trace()
        assert tr["counts"]["value"].shape == (N_CELLS, N_GENES)


# ==============================================================================
# Log-prob — mixture branch
# ==============================================================================


class TestTwoStateMixtureLogProb:
    """twostate_log_prob handles the mixture path correctly."""

    def _make_params(self, n_components, n_genes, vcp=False):
        """Build synthetic per-component params for log-prob testing."""
        rng = np.random.default_rng(123)
        params = {
            "mu": jnp.asarray(
                rng.uniform(1.0, 5.0, (n_components, n_genes)),
                dtype=jnp.float32,
            ),
            "burst_size": jnp.asarray(
                rng.uniform(0.5, 2.0, (n_components, n_genes)),
                dtype=jnp.float32,
            ),
            "k_off": jnp.asarray(
                rng.uniform(0.5, 5.0, (n_components, n_genes)),
                dtype=jnp.float32,
            ),
            "mixing_weights": jnp.ones(n_components) / n_components,
        }
        if vcp:
            params["p_capture"] = jnp.asarray(
                rng.uniform(0.3, 0.9, (N_CELLS,)), dtype=jnp.float32
            )
        return params

    def _make_layouts(self, n_components, vcp=False):
        """Build AxisLayouts for mixture TwoState params."""
        from scribe.core.axis_layout import AxisLayout

        layouts = {
            "mu": AxisLayout(("components", "genes")),
            "burst_size": AxisLayout(("components", "genes")),
            "k_off": AxisLayout(("components", "genes")),
            "mixing_weights": AxisLayout(("components",)),
        }
        if vcp:
            layouts["p_capture"] = AxisLayout(())
        return layouts

    def test_mixture_log_prob_cell_shape(self):
        """Mixture log-prob returns (n_cells,) for return_by='cell'."""
        params = self._make_params(N_COMPONENTS, N_GENES)
        layouts = self._make_layouts(N_COMPONENTS)
        lp = twostate_log_prob(
            COUNTS, params, param_layouts=layouts, return_by="cell"
        )
        assert lp.shape == (N_CELLS,)
        # All values should be finite negative
        assert jnp.all(jnp.isfinite(lp))

    def test_mixture_log_prob_gene_shape(self):
        """Mixture log-prob returns (n_genes,) for return_by='gene'."""
        params = self._make_params(N_COMPONENTS, N_GENES)
        layouts = self._make_layouts(N_COMPONENTS)
        lp = twostate_log_prob(
            COUNTS, params, param_layouts=layouts, return_by="gene"
        )
        assert lp.shape == (N_GENES,)
        assert jnp.all(jnp.isfinite(lp))

    def test_mixture_split_components(self):
        """split_components=True returns (n_cells, K)."""
        params = self._make_params(N_COMPONENTS, N_GENES)
        layouts = self._make_layouts(N_COMPONENTS)
        lp = twostate_log_prob(
            COUNTS,
            params,
            param_layouts=layouts,
            return_by="cell",
            split_components=True,
        )
        assert lp.shape == (N_CELLS, N_COMPONENTS)
        assert jnp.all(jnp.isfinite(lp))

    def test_mixture_vcp_log_prob(self):
        """VCP mixture log-prob returns correct shape."""
        params = self._make_params(N_COMPONENTS, N_GENES, vcp=True)
        layouts = self._make_layouts(N_COMPONENTS, vcp=True)
        lp = twostate_log_prob(
            COUNTS, params, param_layouts=layouts, return_by="cell"
        )
        assert lp.shape == (N_CELLS,)
        assert jnp.all(jnp.isfinite(lp))

    def test_non_mixture_unchanged(self):
        """Non-mixture log-prob still works correctly."""
        rng = np.random.default_rng(99)
        params = {
            "mu": jnp.asarray(
                rng.uniform(1.0, 5.0, (N_GENES,)), dtype=jnp.float32
            ),
            "burst_size": jnp.asarray(
                rng.uniform(0.5, 2.0, (N_GENES,)), dtype=jnp.float32
            ),
            "k_off": jnp.asarray(
                rng.uniform(0.5, 5.0, (N_GENES,)), dtype=jnp.float32
            ),
        }
        lp = twostate_log_prob(COUNTS, params, return_by="cell")
        assert lp.shape == (N_CELLS,)
        assert jnp.all(jnp.isfinite(lp))


# ==============================================================================
# Posterior builder — is_mixture forwarded
# ==============================================================================


class TestTwoStatePosteriorMixture:
    """_build_two_state_posteriors honours is_mixture flag."""

    def test_is_mixture_produces_2d_loc(self):
        """With is_mixture=True, the posterior should handle (K, G) params."""
        from scribe.models.builders.posterior import (
            _build_two_state_posteriors,
        )

        n_components, n_genes = 3, 5
        params = {
            "mu_loc": jnp.zeros((n_components, n_genes)),
            "mu_scale": jnp.ones((n_components, n_genes)),
            "burst_size_loc": jnp.zeros((n_components, n_genes)),
            "burst_size_scale": jnp.ones((n_components, n_genes)),
            "k_off_loc": jnp.ones((n_components, n_genes)),
            "k_off_scale": jnp.ones((n_components, n_genes)),
        }
        dists = _build_two_state_posteriors(
            params,
            unconstrained=True,
            is_mixture=True,
            low_rank=False,
            split=False,
        )
        assert "mu" in dists
        assert "burst_size" in dists
        assert "k_off" in dists

    def test_non_mixture_still_works(self):
        """With is_mixture=False, the posterior still works."""
        from scribe.models.builders.posterior import (
            _build_two_state_posteriors,
        )

        n_genes = 5
        params = {
            "mu_loc": jnp.zeros(n_genes),
            "mu_scale": jnp.ones(n_genes),
            "burst_size_loc": jnp.zeros(n_genes),
            "burst_size_scale": jnp.ones(n_genes),
            "k_off_loc": jnp.ones(n_genes),
            "k_off_scale": jnp.ones(n_genes),
        }
        dists = _build_two_state_posteriors(
            params,
            unconstrained=True,
            is_mixture=False,
            low_rank=False,
            split=False,
        )
        assert "mu" in dists
        assert "burst_size" in dists
        assert "k_off" in dists


# ==============================================================================
# Denoising — mixture marginal
# ==============================================================================


class TestTwoStateMixtureDenoising:
    """_denoise_twostate_mixture_marginal produces correct outputs."""

    def _make_twostate_params(self, n_components, n_genes):
        """Build synthetic (alpha, beta, rate) for K components."""
        rng = np.random.default_rng(77)
        mu = rng.uniform(1.0, 5.0, (n_components, n_genes))
        burst = rng.uniform(0.5, 2.0, (n_components, n_genes))
        k_off = rng.uniform(1.0, 5.0, (n_components, n_genes))
        alphas, betas, rates = [], [], []
        for k in range(n_components):
            a, b, r, _ = _twostate_reparam(
                jnp.asarray(mu[k]),
                jnp.asarray(burst[k]),
                jnp.asarray(k_off[k]),
            )
            alphas.append(a)
            betas.append(b)
            rates.append(r)
        return (
            jnp.stack(alphas),
            jnp.stack(betas),
            jnp.stack(rates),
        )

    def test_marginal_shape(self):
        """Marginal denoised output has shape (n_cells, n_genes)."""
        from scribe.sampling._denoising import (
            _denoise_twostate_mixture_marginal,
        )

        alpha, beta, rate = self._make_twostate_params(
            N_COMPONENTS, N_GENES
        )
        mixing_weights = jnp.ones(N_COMPONENTS) / N_COMPONENTS
        result = _denoise_twostate_mixture_marginal(
            counts=COUNTS,
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
            p_capture=None,
            mixing_weights=mixing_weights,
            component_assignment=None,
            method="mean",
            rng_key=None,
            return_variance=False,
            cell_batch_size=None,
        )
        assert result.shape == (N_CELLS, N_GENES)
        assert jnp.all(jnp.isfinite(result))

    def test_marginal_with_variance(self):
        """return_variance=True returns dict with both keys."""
        from scribe.sampling._denoising import (
            _denoise_twostate_mixture_marginal,
        )

        alpha, beta, rate = self._make_twostate_params(
            N_COMPONENTS, N_GENES
        )
        mixing_weights = jnp.ones(N_COMPONENTS) / N_COMPONENTS
        result = _denoise_twostate_mixture_marginal(
            counts=COUNTS,
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
            p_capture=None,
            mixing_weights=mixing_weights,
            component_assignment=None,
            method="mean",
            rng_key=None,
            return_variance=True,
            cell_batch_size=None,
        )
        assert isinstance(result, dict)
        assert "denoised_counts" in result
        assert "variance" in result
        assert result["denoised_counts"].shape == (N_CELLS, N_GENES)
        assert result["variance"].shape == (N_CELLS, N_GENES)

    def test_hard_assignment(self):
        """With component_assignment, gathers per-cell params."""
        from scribe.sampling._denoising import (
            _denoise_twostate_mixture_marginal,
        )

        alpha, beta, rate = self._make_twostate_params(
            N_COMPONENTS, N_GENES
        )
        mixing_weights = jnp.ones(N_COMPONENTS) / N_COMPONENTS
        # Assign each cell to a random component
        assignments = jnp.array(
            RNG.integers(0, N_COMPONENTS, N_CELLS)
        )
        result = _denoise_twostate_mixture_marginal(
            counts=COUNTS,
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
            p_capture=None,
            mixing_weights=mixing_weights,
            component_assignment=assignments,
            method="mean",
            rng_key=None,
            return_variance=False,
            cell_batch_size=None,
        )
        assert result.shape == (N_CELLS, N_GENES)
        assert jnp.all(jnp.isfinite(result))

    def test_single_component_matches_standard(self):
        """With K=1, mixture marginal should match the non-mixture path."""
        from scribe.sampling._denoising import (
            _denoise_twostate_mixture_marginal,
        )
        from scribe.sampling._denoising import _denoise_standard

        alpha_1, beta_1, rate_1 = self._make_twostate_params(1, N_GENES)
        mixing_weights = jnp.ones(1)

        # Mixture marginal path
        result_mix = _denoise_twostate_mixture_marginal(
            counts=COUNTS,
            ts_alpha=alpha_1,
            ts_beta=beta_1,
            ts_rate=rate_1,
            p_capture=None,
            mixing_weights=mixing_weights,
            component_assignment=None,
            method="mean",
            rng_key=None,
            return_variance=False,
            cell_batch_size=None,
        )

        # Standard non-mixture path
        result_std = _denoise_standard(
            counts=COUNTS,
            r=None,
            p=None,
            p_capture=None,
            gate=None,
            method="mean",
            rng_key=None,
            return_variance=False,
            cell_batch_size=None,
            ts_alpha=alpha_1[0],
            ts_beta=beta_1[0],
            ts_rate=rate_1[0],
        )

        np.testing.assert_allclose(
            np.asarray(result_mix),
            np.asarray(result_std),
            rtol=1e-5,
        )
