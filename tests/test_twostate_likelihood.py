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

    def test_mixture_raises(self):
        like = TwoStateLikelihood()
        with pytest.raises(NotImplementedError, match="mixture"):
            like.sample(
                {
                    "mu": jnp.array([1.0]),
                    "burst_size": jnp.array([1.0]),
                    "k_off": jnp.array([5.0]),
                    "mixing_weights": jnp.array([0.5, 0.5]),
                },
                cell_specs=[],
                counts=None,
                dims={"n_cells": 1, "n_genes": 1},
                batch_size=None,
                model_config=None,
            )

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

    def test_multi_dataset_raises(self):
        like = TwoStateLikelihood()
        with pytest.raises(NotImplementedError, match="multi-dataset"):
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

    def test_biology_informed_capture_raises(self):
        """VCP rejects biology-informed capture priors in phase 1."""
        like = TwoStateVCPLikelihood(biology_informed_spec=object())

        class _Config:
            param_specs = []

        with pytest.raises(NotImplementedError, match="biology-informed"):
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
