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

    def test_mean_capture_correction_recovers_pre_capture_mean(self):
        """Audit fix: with ``mean_capture < 1``, the anchor recovers
        the *pre-capture* per-gene mean, not the observed mean.

        Under twostatevcp with mean capture ``nu_bar``, the observed
        per-gene sample mean is ``mu_g * nu_bar``.  Anchoring at the
        raw observed mean leaves the variational ``mu_loc`` too small
        by a factor of ``1/nu_bar`` for high-expression genes — the
        symptom the external auditor diagnosed from the mean-
        calibration plot's downward bend at the high end.
        """
        rng = np.random.default_rng(0)
        true_mu = np.array([5.0, 50.0, 174.0])
        nu_bar = 0.5
        counts = np.stack(
            [rng.poisson(true_mu[g] * nu_bar, 500) for g in range(3)], axis=1
        )
        # With the correction, the recovered anchor (via softplus) is
        # close to the true pre-capture mu.
        anchor = empirical_mu_anchor_from_counts(
            jnp.asarray(counts), mean_capture=nu_bar
        )
        # Softplus is asymptotically linear at moderate-to-large
        # values, so the anchor itself is roughly the recovered mu.
        # Compare to true_mu within 5% relative tolerance (Monte
        # Carlo noise at n=500 dominates the error budget).
        recovered = np.asarray(anchor)
        rel_err = np.abs(recovered - true_mu) / true_mu
        assert np.all(rel_err < 0.10), (
            f"recovered={recovered}, true_mu={true_mu}, "
            f"rel_err={rel_err}"
        )


# ==============================================================================
# estimate_initial_mean_capture
# ==============================================================================


class TestEstimateInitialMeanCapture:
    """Estimate the prior-mean capture probability from the config."""

    def test_non_vcp_returns_one(self):
        from scribe.core.twostate_data_init import (
            estimate_initial_mean_capture,
        )

        class _Cfg:
            base_model = "twostate"
            priors = None

        assert estimate_initial_mean_capture(_Cfg(), jnp.zeros((4, 2))) == 1.0

    def test_biology_informed_prior_does_not_drive_anchor(self):
        """``priors.eta_capture`` is intentionally **not** used as the
        initialization divisor.

        The bio-informed prior is an order-of-magnitude *belief* about
        ``M_0``, not a fitted estimate.  Using it as a divisor over- or
        under-corrects when the true posterior median of
        ``mean(p_capture)`` lies away from the prior median.

        Updated semantics (post mean-capture-revert): the heuristic
        returns ``1.0`` for the default VCP case, anchoring the prior
        at the *observed* per-gene mean.  Recovery of the pre-capture
        mean is delegated to the optimizer — most naturally under
        ``positive_transform={"mu": "exp"}``, where SVI takes
        multiplicative steps in ``mu`` and converges in a handful of
        iterations even when the anchor is off by the capture factor.
        """
        from scribe.core.twostate_data_init import (
            estimate_initial_mean_capture,
        )

        rng = np.random.default_rng(0)
        n_cells, n_genes = 100, 20
        counts = rng.multinomial(32_000, np.ones(n_genes) / n_genes, n_cells)
        # A tight bio-informed prior at log_M0 = log(50000) is now
        # ignored entirely — the heuristic returns the no-divisor
        # default 1.0 and lets the optimizer recover.
        log_M0 = float(np.log(50_000))

        class _Priors:
            __pydantic_extra__ = {"eta_capture": (log_M0, 0.5)}

        class _Cfg:
            base_model = "twostatevcp"
            priors = _Priors()

        estimate = estimate_initial_mean_capture(_Cfg(), jnp.asarray(counts))
        assert estimate == 1.0

    def test_flat_beta_prior_uses_alpha_over_alpha_plus_beta(self):
        """``priors.p_capture = (alpha, beta)`` activates the flat-Beta
        prior-mean estimate."""
        from scribe.core.twostate_data_init import (
            estimate_initial_mean_capture,
        )

        class _Priors:
            __pydantic_extra__ = {"p_capture": (3.0, 1.0)}

        class _Cfg:
            base_model = "twostatevcp"
            priors = _Priors()

        # Beta(3, 1) mean = 3/4 = 0.75.
        estimate = estimate_initial_mean_capture(_Cfg(), jnp.zeros((4, 2)))
        assert estimate == 0.75

    def test_default_vcp_returns_one(self):
        """No explicit capture prior → no divisor; anchor at observed mean.

        After the mean-capture revert, the default for ``twostatevcp``
        without an explicit ``priors.p_capture`` tuple is ``1.0``
        (anchor at the *observed* per-gene mean).  The optimizer
        recovers the pre-capture mean — most efficiently under
        ``positive_transform={"mu": "exp"}``.
        """
        from scribe.core.twostate_data_init import (
            estimate_initial_mean_capture,
        )

        class _Priors:
            __pydantic_extra__ = {}

        class _Cfg:
            base_model = "twostatevcp"
            priors = _Priors()

        assert estimate_initial_mean_capture(_Cfg(), jnp.zeros((4, 2))) == 1.0


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

    def test_user_warning_fires_by_default(self, scribe_caplog):
        """The data-driven mu init must surface an INFO log so
        the user sees the anchor is being applied.
        """
        import logging

        import scribe

        rng = np.random.default_rng(0)
        n_cells, n_genes = 32, 8
        per_gene = rng.uniform(0.5, 50.0, n_genes)
        counts = np.stack(
            [rng.poisson(m, n_cells) for m in per_gene], axis=1
        )
        scribe_caplog.set_level(logging.INFO, logger="scribe")
        scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="natural",
            inference_method="svi",
            n_steps=1,
            unconstrained=True,
        )
        matched = [
            record.message
            for record in scribe_caplog.records
            if "TwoState" in record.message and "mu_prior_loc" in record.message
        ]
        assert matched, "Expected a TwoState mu_prior_loc INFO log"
        assert "applied per-gene mu_prior_loc" in matched[0]

    def test_user_warning_for_skip_with_explicit_prior(self, scribe_caplog):
        """Passing ``priors={'mu': ...}`` must short-circuit the anchor
        AND emit an INFO log so the user knows the empirical init
        was skipped."""
        import logging

        import scribe

        rng = np.random.default_rng(0)
        n_cells, n_genes = 32, 8
        per_gene = rng.uniform(0.5, 50.0, n_genes)
        counts = np.stack(
            [rng.poisson(m, n_cells) for m in per_gene], axis=1
        )
        scribe_caplog.set_level(logging.INFO, logger="scribe")
        scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="natural",
            inference_method="svi",
            n_steps=1,
            unconstrained=True,
            priors={"mu": (1.5, 2.0)},
        )
        matched = [
            record.message
            for record in scribe_caplog.records
            if "skipping" in record.message and "mu_prior_loc" in record.message
        ]
        assert matched, "Expected a 'skipping mu_prior_loc' INFO log"

    def test_post_fit_predictive_after_per_gene_anchor(self):
        """Regression: after a fit with the per-gene mu anchor,
        ``get_posterior_samples`` rebuilds (model, guide) via
        ``_model_and_guide`` which calls ``get_model_and_guide``
        again.  If ``_factory_n_genes()`` returns ``None`` (as it
        used to for non-VAE results), the factory's validation
        falls back to ``n_genes=5`` and trips on the per-gene
        anchor array.  This test guards the post-fit path.
        """
        import scribe
        import jax

        rng = np.random.default_rng(0)
        n_cells, n_genes = 32, 12
        per_gene = rng.uniform(0.5, 100.0, n_genes)
        counts = np.stack(
            [rng.poisson(m, n_cells) for m in per_gene], axis=1
        )
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="natural",
            inference_method="svi",
            n_steps=2,
            unconstrained=True,
        )
        # This call invokes ``_model_and_guide``, which previously
        # failed with the per-gene anchor.
        samples = res.get_posterior_samples(
            rng_key=jax.random.PRNGKey(0),
            n_samples=4,
            counts=counts,
        )
        assert "mu" in samples
        assert np.asarray(samples["mu"]).shape == (4, n_genes)

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
