"""Tests for the TwoState data-driven mu initialization.

Validates:
- ``empirical_mu_anchor_from_counts`` produces a finite per-gene
  array under both ``softplus`` and ``exp`` transforms.
- The softplus inverse stays numerically stable for ribosomal-scale
  gene means (~100–500) that would overflow ``log(expm1(x))`` in
  float32.
- ``inject_twostate_data_init`` stashes the per-gene array on the
  priors extras and the factory threads it into the ``mu`` spec's
  ``default_params``.
- A 5-step SVI run on synthetic data with one high-expression gene
  starts at a finite loss with the variational ``mu_loc``
  initialized at the empirical mean — without the data-init the
  same gene would start at ``mu ≈ 1``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.optim as optim
import pytest
from numpyro.infer import SVI, Trace_ELBO

from scribe.core.twostate_data_init import (
    empirical_mu_anchor_from_counts,
    inject_twostate_data_init,
)
from scribe.models import ModelConfigBuilder, get_model_and_guide


# ==============================================================================
# empirical_mu_anchor_from_counts
# ==============================================================================


class TestEmpiricalMuAnchor:
    """Per-gene anchor in unconstrained space, both transforms."""

    @pytest.mark.parametrize("transform", ["softplus", "exp"])
    def test_shape_matches_gene_axis(self, transform):
        counts = jnp.zeros((10, 5), dtype=jnp.int32)
        anchor = empirical_mu_anchor_from_counts(counts, transform=transform)
        assert anchor.shape == (5,)

    def test_softplus_anchor_is_finite_at_ribosomal_magnitudes(self):
        """``log(expm1(174))`` overflows in float32; the stable
        branch must keep the anchor finite for genes of any
        magnitude that fits in float32 storage."""
        rng = np.random.default_rng(0)
        n_cells = 100
        # Span 5 decades of expression: 0.5 -> 500
        per_gene_mean = np.array([0.5, 5.0, 50.0, 174.0, 500.0])
        counts = np.stack(
            [rng.poisson(m, n_cells) for m in per_gene_mean], axis=1
        )
        anchor = empirical_mu_anchor_from_counts(
            jnp.asarray(counts), transform="softplus"
        )
        assert np.all(np.isfinite(np.asarray(anchor)))
        # Softplus is asymptotically the identity for large x, so for
        # the ribosomal-scale gene the anchor should equal the
        # empirical mean to within ~0.5%.
        emp_mean = counts[:, -1].mean()
        assert abs(float(anchor[-1]) - emp_mean) / emp_mean < 0.01

    def test_softplus_anchor_low_expression_uses_log_branch(self):
        """For empirical mean << 1 the anchor should be ~``log(mean)``,
        the natural mapping through softplus_inv in that regime."""
        counts = jnp.full((100, 1), 0.0)  # all-zero gene
        counts = counts.at[0, 0].set(1)  # one count → mean=0.01
        anchor = empirical_mu_anchor_from_counts(counts, transform="softplus")
        # mean = 0.01, softplus_inv(0.01) = log(expm1(0.01)) ≈ log(0.01)
        # = -4.6
        assert float(anchor[0]) < -3.0

    def test_exp_anchor_matches_log_mean(self):
        counts = jnp.full((100, 3), 10.0)
        anchor = empirical_mu_anchor_from_counts(counts, transform="exp")
        # log(10) ≈ 2.3026
        np.testing.assert_allclose(
            np.asarray(anchor), np.full(3, np.log(10.0)), atol=1e-4
        )

    def test_unknown_transform_raises(self):
        counts = jnp.zeros((4, 2))
        with pytest.raises(ValueError, match="Unknown positive transform"):
            empirical_mu_anchor_from_counts(counts, transform="tanh")


# ==============================================================================
# inject_twostate_data_init + factory threading
# ==============================================================================


class TestInjectAndThread:
    """End-to-end: priors extras → factory → mu spec default_params."""

    def _make_counts(self, seed=0):
        """Synthetic counts including a high-expression gene that
        would otherwise stall the optimizer if the global anchor
        were used."""
        rng = np.random.default_rng(seed)
        per_gene = np.array([2.0, 5.0, 10.0, 50.0, 174.0])
        n_cells = 64
        counts = np.stack(
            [rng.poisson(m, n_cells) for m in per_gene], axis=1
        )
        return jnp.asarray(counts, dtype=jnp.int32)

    def test_priors_extras_carry_array(self):
        counts = self._make_counts()
        cfg = (
            ModelConfigBuilder()
            .for_model("twostate")
            .with_parameterization("two_state_natural")
            .with_inference("svi")
            .unconstrained()
            .build()
        )
        cfg = inject_twostate_data_init(cfg, counts)
        anchor = cfg.priors.__pydantic_extra__["mu_prior_loc"]
        assert jnp.asarray(anchor).shape == (counts.shape[1],)
        assert np.all(np.isfinite(np.asarray(anchor)))

    def test_factory_threads_array_into_mu_spec(self):
        """The factory's ``model_copy`` of the mu spec must preserve
        the array shape, not collapse it to a single ``float()``."""
        counts = self._make_counts()
        cfg = (
            ModelConfigBuilder()
            .for_model("twostate")
            .with_parameterization("two_state_natural")
            .with_inference("svi")
            .unconstrained()
            .build()
        )
        cfg = inject_twostate_data_init(cfg, counts)

        # get_model_and_guide builds param_specs via the factory; we
        # can't pull them out without instantiating the model, so we
        # check the threading by running SVI init and inspecting the
        # variational mu_loc.
        model, guide, cfg_full = get_model_and_guide(cfg)
        svi = SVI(model, guide, optim.Adam(1e-3), Trace_ELBO())
        state = svi.init(
            jax.random.PRNGKey(0),
            n_cells=counts.shape[0],
            n_genes=counts.shape[1],
            model_config=cfg_full,
            counts=counts,
        )
        params = svi.get_params(state)
        # mu_loc should be a (n_genes,) array, each entry = the
        # per-gene anchor we computed.  Verify shape and the
        # high-expression entry tracks the empirical mean.
        mu_loc = np.asarray(params["mu_loc"])
        assert mu_loc.shape == (counts.shape[1],)
        rps2_mean = np.asarray(counts[:, -1]).mean()
        # For ribosomal-scale means, softplus_inv ≈ identity.
        assert abs(mu_loc[-1] - rps2_mean) / rps2_mean < 0.01

    def test_validation_threads_real_n_genes(self):
        """Regression: the factory's dry-run validation must use the
        real ``n_genes`` from the data, not the default ``n_genes=5``.
        Without this, a per-gene ``mu_prior_loc`` array of shape
        ``(real_n_genes,)`` cannot broadcast against the dry-run's
        ``(5,)`` and SVI init raises a ``ValueError``.
        """
        import scribe

        # Use n_genes != 5 to expose the broadcast bug.  10 is enough.
        rng = np.random.default_rng(0)
        n_cells, n_genes = 32, 10
        per_gene = rng.uniform(0.5, 50.0, n_genes)
        counts = np.stack(
            [rng.poisson(m, n_cells) for m in per_gene], axis=1
        )
        # Bare-bones fit, no joint guide; just the per-gene mu init
        # path threading through.
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="natural",
            inference_method="svi",
            n_steps=2,
            unconstrained=True,
        )
        assert jnp.isfinite(res.loss_history[-1])

    def test_svi_initial_loss_finite_with_high_expression_gene(self):
        """Without the per-gene anchor, the high-expression gene's
        log-PMF would be -∞ at step 0 (softplus(scalar median) ≈ 1
        while data mean ≈ 174).  This test guards the regression."""
        counts = self._make_counts()
        cfg = (
            ModelConfigBuilder()
            .for_model("twostate")
            .with_parameterization("two_state_natural")
            .with_inference("svi")
            .unconstrained()
            .build()
        )
        cfg = inject_twostate_data_init(cfg, counts)
        model, guide, cfg_full = get_model_and_guide(cfg)
        svi = SVI(model, guide, optim.Adam(1e-3), Trace_ELBO())
        state = svi.init(
            jax.random.PRNGKey(1),
            n_cells=counts.shape[0],
            n_genes=counts.shape[1],
            model_config=cfg_full,
            counts=counts,
        )
        loss0 = svi.evaluate(
            state,
            n_cells=counts.shape[0],
            n_genes=counts.shape[1],
            model_config=cfg_full,
            counts=counts,
        )
        assert jnp.isfinite(loss0)
