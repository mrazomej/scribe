"""Tests for the TwoStateLikelihood and TwoStateVCPLikelihood classes.

Phase-1 surface — focused on:

- (µ, b, k_off) → (α, β, rate) reparameterisation: mean preservation
  with and without the α / k_off floors,
- plate shapes under all three modes (prior predictive, full, batched),
- phase-1 guards raise ``NotImplementedError`` for the deferred paths,
- ``twostate_log_prob`` handles both no-capture and VCP rate shapes.
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
    _ALPHA_MIN,
    _K_OFF_MIN,
    _twostate_reparam,
)


# ==============================================================================
# (µ, b, k_off) → (α, β, rate) reparameterisation
# ==============================================================================


class TestReparam:
    """The forward map is mean-preserving even when floors activate."""

    def test_natural_mean_preservation(self):
        """With no floor activations, E[count] = µ exactly."""
        mu = jnp.array([2.0, 5.0, 10.0])
        b = jnp.array([1.0, 2.0, 3.0])
        k_off = jnp.array([20.0, 5.0, 0.5])
        alpha, beta, rate, eff_burst = _twostate_reparam(mu, b, k_off)
        implied_mean = rate * alpha / (alpha + beta)
        np.testing.assert_allclose(
            np.asarray(implied_mean), np.asarray(mu), rtol=1e-5
        )

    def test_alpha_floor_preserves_mean(self):
        """Even when the α floor activates (µ/b < _ALPHA_MIN), E[count] = µ."""
        # Low-expression gene that would have α_nat = 0.01, below _ALPHA_MIN=0.05.
        mu = jnp.array([0.01])
        b = jnp.array([1.0])
        k_off = jnp.array([5.0])
        alpha, beta, rate, eff_burst = _twostate_reparam(mu, b, k_off)
        assert float(alpha[0]) == pytest.approx(_ALPHA_MIN)
        # Mean preservation still holds by construction.
        implied_mean = rate * alpha / (alpha + beta)
        np.testing.assert_allclose(
            np.asarray(implied_mean), np.asarray(mu), rtol=1e-5
        )
        # eff_burst_size now reflects the clamp.
        np.testing.assert_allclose(
            float(eff_burst[0]), 0.01 / _ALPHA_MIN, rtol=1e-5
        )

    def test_k_off_floor_preserves_mean(self):
        """When k_off floor activates, E[count] = µ still holds."""
        mu = jnp.array([2.0])
        b = jnp.array([0.5])
        k_off = jnp.array([0.001])  # below _K_OFF_MIN
        alpha, beta, rate, eff_burst = _twostate_reparam(mu, b, k_off)
        assert float(beta[0]) == pytest.approx(_K_OFF_MIN)
        implied_mean = rate * alpha / (alpha + beta)
        np.testing.assert_allclose(
            np.asarray(implied_mean), np.asarray(mu), rtol=1e-5
        )


# ==============================================================================
# Plate-mode shapes
# ==============================================================================


def _no_capture_model(likelihood, n_cells, n_genes, batch_size, counts):
    """Helper: build a minimal numpyro model around the likelihood."""

    def model(_counts=counts):
        mu = numpyro.sample(
            "mu",
            dist.LogNormal(jnp.array([0.0] * n_genes), 1.0).to_event(1),
        )
        b = numpyro.sample(
            "burst_size",
            dist.LogNormal(jnp.array([0.0] * n_genes), 1.0).to_event(1),
        )
        k_off = numpyro.sample(
            "k_off",
            dist.LogNormal(jnp.array([1.0] * n_genes), 1.0).to_event(1),
        )
        likelihood.sample(
            {"mu": mu, "burst_size": b, "k_off": k_off},
            cell_specs=[],
            counts=_counts,
            dims={"n_cells": n_cells, "n_genes": n_genes},
            batch_size=batch_size,
            model_config=None,
        )

    return model


class TestTwoStateLikelihoodPlateModes:
    """sample() must produce the right trace shapes in all three modes."""

    def test_prior_predictive(self):
        like = TwoStateLikelihood()
        n_cells, n_genes = 12, 3
        model = _no_capture_model(like, n_cells, n_genes, None, None)
        tr = trace(seed(model, jax.random.PRNGKey(0))).get_trace()
        assert "counts" in tr
        assert tr["counts"]["value"].shape == (n_cells, n_genes)
        # Gene-level deterministics emit at gene rank, not cell rank.
        assert tr["alpha"]["value"].shape == (n_genes,)
        assert tr["beta"]["value"].shape == (n_genes,)
        assert tr["effective_burst_size"]["value"].shape == (n_genes,)

    def test_full_conditioning(self):
        like = TwoStateLikelihood()
        n_cells, n_genes = 12, 3
        rng = np.random.default_rng(0)
        counts = jnp.asarray(
            rng.integers(0, 5, size=(n_cells, n_genes)), dtype=jnp.int32
        )
        model = _no_capture_model(like, n_cells, n_genes, None, counts)
        tr = trace(seed(model, jax.random.PRNGKey(1))).get_trace()
        # Observed counts site has the observed value.
        np.testing.assert_array_equal(
            np.asarray(tr["counts"]["value"]), np.asarray(counts)
        )

    def test_batched(self):
        like = TwoStateLikelihood()
        n_cells, n_genes, batch = 12, 3, 4
        counts = jnp.zeros((n_cells, n_genes), dtype=jnp.int32)
        model = _no_capture_model(like, n_cells, n_genes, batch, counts)
        tr = trace(seed(model, jax.random.PRNGKey(2))).get_trace()
        # Subsampled obs has batch-sized leading axis.
        assert tr["counts"]["value"].shape == (batch, n_genes)


class TestTwoStateVCPLikelihoodPlateModes:
    """VCP variant adds p_capture inside the cell plate."""

    def _make_model(self, likelihood, n_cells, n_genes, batch_size, counts):
        class _Config:
            param_specs = []

        def model(_counts=counts):
            mu = numpyro.sample(
                "mu",
                dist.LogNormal(jnp.array([0.0] * n_genes), 1.0).to_event(1),
            )
            b = numpyro.sample(
                "burst_size",
                dist.LogNormal(jnp.array([0.0] * n_genes), 1.0).to_event(1),
            )
            k_off = numpyro.sample(
                "k_off",
                dist.LogNormal(jnp.array([1.0] * n_genes), 1.0).to_event(1),
            )
            likelihood.sample(
                {"mu": mu, "burst_size": b, "k_off": k_off},
                cell_specs=[],
                counts=_counts,
                dims={"n_cells": n_cells, "n_genes": n_genes},
                batch_size=batch_size,
                model_config=_Config(),
            )

        return model

    def test_p_capture_is_per_cell(self):
        like = TwoStateVCPLikelihood()
        n_cells, n_genes = 8, 3
        model = self._make_model(like, n_cells, n_genes, None, None)
        tr = trace(seed(model, jax.random.PRNGKey(3))).get_trace()
        # p_capture is cell-rank inside the cell plate.
        assert tr["p_capture"]["value"].shape == (n_cells,)
        # Counts at (C, G).
        assert tr["counts"]["value"].shape == (n_cells, n_genes)
        # Gene-level deterministics still at (G,).
        assert tr["alpha"]["value"].shape == (n_genes,)


# ==============================================================================
# Phase-1 guards
# ==============================================================================


class TestPhase1Guards:
    """Each deferred path raises NotImplementedError with a clear message."""

    def test_mixture_supported(self):
        """Mixtures are now fully supported — no NotImplementedError."""
        like = TwoStateLikelihood()
        n_genes, n_components = 2, 2

        def model():
            mu = numpyro.sample(
                "mu",
                dist.LogNormal(
                    jnp.zeros((n_components, n_genes)), 1.0
                ).to_event(2),
            )
            b = numpyro.sample(
                "burst_size",
                dist.LogNormal(
                    jnp.zeros((n_components, n_genes)), 1.0
                ).to_event(2),
            )
            k = numpyro.sample(
                "k_off",
                dist.LogNormal(
                    jnp.ones((n_components, n_genes)), 1.0
                ).to_event(2),
            )
            like.sample(
                {
                    "mu": mu,
                    "burst_size": b,
                    "k_off": k,
                    "mixing_weights": jnp.array([0.5, 0.5]),
                },
                cell_specs=[],
                counts=None,
                dims={"n_cells": 4, "n_genes": n_genes},
                batch_size=None,
                model_config=None,
            )

        tr = trace(seed(model, jax.random.PRNGKey(0))).get_trace()
        assert tr["counts"]["value"].shape == (4, n_genes)

    def test_vae_raises(self):
        like = TwoStateLikelihood()
        with pytest.raises(NotImplementedError, match="VAE"):
            like.sample(
                {
                    "mu": jnp.array([1.0]),
                    "burst_size": jnp.array([1.0]),
                    "k_off": jnp.array([5.0]),
                },
                cell_specs=[],
                counts=None,
                dims={"n_cells": 1, "n_genes": 1},
                batch_size=None,
                model_config=None,
                vae_cell_fn=lambda idx: {},
            )

    def test_annotation_priors_raise(self):
        like = TwoStateLikelihood()
        with pytest.raises(NotImplementedError, match="annotation"):
            like.sample(
                {
                    "mu": jnp.array([1.0]),
                    "burst_size": jnp.array([1.0]),
                    "k_off": jnp.array([5.0]),
                },
                cell_specs=[],
                counts=None,
                dims={"n_cells": 1, "n_genes": 1},
                batch_size=None,
                model_config=None,
                annotation_prior_logits=jnp.zeros((1, 2)),
            )

    def test_dataset_indices_without_n_datasets_raises(self):
        """Passing dataset_indices without configuring n_datasets is an error.

        Multi-dataset IS supported, but only when ``model_config.n_datasets``
        is set; supplying per-cell dataset assignments without it is a setup
        mistake and is rejected with a clear ValueError.
        """
        like = TwoStateLikelihood()
        with pytest.raises(ValueError, match="n_datasets"):
            like.sample(
                {
                    "mu": jnp.array([1.0]),
                    "burst_size": jnp.array([1.0]),
                    "k_off": jnp.array([5.0]),
                },
                cell_specs=[],
                counts=None,
                dims={"n_cells": 1, "n_genes": 1},
                batch_size=None,
                model_config=None,
                dataset_indices=jnp.array([0]),
            )

    def test_phi_capture_raises(self):
        """VCP rejects phi_capture in phase 1."""
        like = TwoStateVCPLikelihood(capture_param_name="phi_capture")

        class _Config:
            param_specs = []

        with pytest.raises(NotImplementedError, match="phi_capture"):
            like.sample(
                {
                    "mu": jnp.array([1.0]),
                    "burst_size": jnp.array([1.0]),
                    "k_off": jnp.array([5.0]),
                },
                cell_specs=[],
                counts=None,
                dims={"n_cells": 1, "n_genes": 1},
                batch_size=None,
                model_config=_Config(),
            )

    def test_biology_informed_capture_wired(self):
        """VCP accepts biology-informed capture priors end to end.

        Closure under binomial thinning (paper/_capture_prior.qmd) makes
        the capture factor multiplicative, identical to its role in the
        NB family, so the eta_capture-based prior applies unchanged.
        """
        import scribe
        import jax
        import numpy as _np

        rng = _np.random.default_rng(0)
        counts = _np.stack(
            [rng.poisson(m, 32) for m in [2.0, 5.0, 50.0, 100.0]], axis=1
        )
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="natural",
            inference_method="svi",
            n_steps=2,
            unconstrained=True,
            priors={"capture_efficiency": (_np.log(50_000), 0.5)},
        )
        # eta_capture is sampled when bio-informed prior is active;
        # p_capture remains as a derived deterministic.
        samples = res.get_posterior_samples(
            rng_key=jax.random.PRNGKey(0), n_samples=2, counts=counts
        )
        assert "eta_capture" in samples
        assert "p_capture" in samples

    def test_biology_informed_phi_capture_still_rejected(self):
        """``phi_capture`` (mean-odds NB compat) remains unsupported
        for TwoState even under the biology-informed prior."""

        class _BioSpec:
            use_phi_capture = True
            mu_eta_prior = None
            log_M0 = 10.0
            sigma_M = 0.5

        like = TwoStateVCPLikelihood(
            capture_param_name="p_capture",
            biology_informed_spec=_BioSpec(),
        )

        class _Config:
            param_specs = []

        with pytest.raises(NotImplementedError, match="phi_capture"):
            like.sample(
                {
                    "mu": jnp.array([1.0]),
                    "burst_size": jnp.array([1.0]),
                    "k_off": jnp.array([5.0]),
                },
                cell_specs=[],
                counts=None,
                dims={"n_cells": 1, "n_genes": 1},
                batch_size=None,
                model_config=_Config(),
            )


# ==============================================================================
# twostate_log_prob — both rate shapes
# ==============================================================================


class TestTwoStateLogProb:
    """The log-prob helper handles gene-rank and (C, G) rate shapes."""

    def test_no_capture_shape_by_cell(self):
        n_cells, n_genes = 5, 3
        counts = jnp.zeros((n_cells, n_genes), dtype=jnp.int32)
        params = {
            "mu": jnp.array([1.0, 2.0, 5.0]),
            "burst_size": jnp.array([1.0, 2.0, 1.0]),
            "k_off": jnp.array([20.0, 5.0, 0.5]),
        }
        lp = twostate_log_prob(counts, params, return_by="cell")
        assert lp.shape == (n_cells,)
        assert np.all(np.isfinite(np.asarray(lp)))

    def test_no_capture_shape_by_gene(self):
        n_cells, n_genes = 5, 3
        counts = jnp.zeros((n_cells, n_genes), dtype=jnp.int32)
        params = {
            "mu": jnp.array([1.0, 2.0, 5.0]),
            "burst_size": jnp.array([1.0, 2.0, 1.0]),
            "k_off": jnp.array([20.0, 5.0, 0.5]),
        }
        lp = twostate_log_prob(counts, params, return_by="gene")
        assert lp.shape == (n_genes,)
        assert np.all(np.isfinite(np.asarray(lp)))

    def test_vcp_shape(self):
        """With p_capture in params, log_prob composes (C, G) rate."""
        n_cells, n_genes = 5, 3
        counts = jnp.zeros((n_cells, n_genes), dtype=jnp.int32)
        params = {
            "mu": jnp.array([1.0, 2.0, 5.0]),
            "burst_size": jnp.array([1.0, 2.0, 1.0]),
            "k_off": jnp.array([20.0, 5.0, 0.5]),
            "p_capture": jnp.array([0.3, 0.5, 0.7, 0.4, 0.6]),
        }
        lp_cell = twostate_log_prob(counts, params, return_by="cell")
        lp_gene = twostate_log_prob(counts, params, return_by="gene")
        assert lp_cell.shape == (n_cells,)
        assert lp_gene.shape == (n_genes,)


# ==============================================================================
# Multi-dataset indexing (dataset_indices threaded through sample)
# ==============================================================================


class TestMultiDatasetIndexing:
    """The likelihood gathers per-dataset params inside the cell plate.

    These exercise the supported multi-dataset path directly at the likelihood
    level: per-dataset gene params have a leading ``(n_datasets, n_genes)``
    axis, and inside the cell plate each cell is mapped to its dataset's
    parameters via ``index_dataset_params``.  The key regression guard is that
    the observed-count site stays ``(n_cells, n_genes)`` rather than ballooning
    to ``(n_cells, n_cells, n_genes)``.
    """

    @staticmethod
    def _moment_delta_cfg(n_datasets=2):
        from scribe.models.config import ModelConfig
        from scribe.models.presets.factory import create_model

        cfg = ModelConfig(
            base_model="twostate",
            parameterization="two_state_moment_delta",
            unconstrained=True,
            n_datasets=n_datasets,
            expression_dataset_prior="horseshoe",
            regime_dataset_prior="horseshoe",
        )
        # create_model attaches the resolved param_specs to the config, which
        # index_dataset_params consults to find the dataset axis.
        create_model(cfg)
        return cfg

    def test_no_capture_counts_shape(self):
        cfg = self._moment_delta_cfg()
        like = TwoStateLikelihood()
        D, G, N = 2, 4, 20
        # Per-dataset gene parameters carry a leading dataset axis.
        param_values = {
            "mu": jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (D, G))) + 0.5,
            "excess_fano": jnp.abs(
                jax.random.normal(jax.random.PRNGKey(1), (D, G))
            )
            + 0.1,
            "inv_concentration": jnp.full((D, G), 0.05),
        }
        counts = jax.random.randint(
            jax.random.PRNGKey(2), (N, G), 0, 12
        ).astype(jnp.float32)
        di = jnp.array([0] * (N // 2) + [1] * (N - N // 2))

        with seed(rng_seed=0):
            with trace() as tr:
                like.sample(
                    param_values,
                    cell_specs=[],
                    counts=counts,
                    dims={"n_cells": N, "n_genes": G},
                    batch_size=None,
                    model_config=cfg,
                    dataset_indices=di,
                )
        # Observation is per-cell, NOT (N, N, G).
        assert tr["counts"]["value"].shape == (N, G)
        # Gene/dataset-rank deterministics emitted outside the plate.
        assert tr["k_off"]["value"].shape == (D, G)

    def test_no_capture_batched_counts_shape(self):
        cfg = self._moment_delta_cfg()
        like = TwoStateLikelihood()
        D, G, N, B = 2, 4, 24, 8
        param_values = {
            "mu": jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (D, G))) + 0.5,
            "excess_fano": jnp.abs(
                jax.random.normal(jax.random.PRNGKey(1), (D, G))
            )
            + 0.1,
            "inv_concentration": jnp.full((D, G), 0.05),
        }
        counts = jax.random.randint(
            jax.random.PRNGKey(2), (N, G), 0, 12
        ).astype(jnp.float32)
        di = jnp.array([0] * (N // 2) + [1] * (N - N // 2))

        with seed(rng_seed=0):
            with trace() as tr:
                like.sample(
                    param_values,
                    cell_specs=[],
                    counts=counts,
                    dims={"n_cells": N, "n_genes": G},
                    batch_size=B,
                    model_config=cfg,
                    dataset_indices=di,
                )
        assert tr["counts"]["value"].shape == (B, G)
