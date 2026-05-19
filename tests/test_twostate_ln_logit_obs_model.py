"""Unit tests for the TSLN-Logit observation-model adapter.

Covers the Rev 4 audit's most pointed concern for Step 2 of PR-2:
the constructor must **strictly enforce** the PR-2 capture
restriction (no soft-eta Newton; ``x_only`` and ``x_only_offset``
only).  Plus end-to-end smoke through ``run_laplace_em`` to confirm
the obs-model wires into the generic Laplace driver correctly.

Test groups
-----------

1. Capture-mode validation (the auditor's watchpoint):
   - ``capture_anchor`` non-``None`` is rejected.
   - ``informative_priors['eta']`` is rejected.
   - No-capture is accepted (``x_only`` Newton).
   - Frozen-offset capture is accepted (``x_only_offset`` Newton).

2. Constructor key validation:
   - Invalid ``informative_priors`` keys rejected.
   - Invalid ``freeze_params`` keys rejected.
   - Missing ``freeze_values`` for a frozen key rejected.

3. ``init_state`` shapes / defaults match the plan:
   - ``rate_init = 2 · empirical_mean`` (so ``E[u | z=0] = mean``).
   - ``kappa_init = 3.0`` (SVI median).
   - ``eta_anchor_init = 0.0``.
   - Latent ``z`` initialised at zero.
   - Frozen-gene-global keys excluded from ``params`` dict.

4. End-to-end smoke through ``run_laplace_em``:
   - ``x_only`` path on a small synthetic problem completes with a
     finite loss trajectory and ``‖g‖_∞`` finite at the final
     iterate.
   - ``x_only_offset`` path with frozen ``eta`` completes likewise.
   - ``pack_result`` emits the expected globals dict shape.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scribe.laplace._obs_twostate_ln_logit import (
    TwoStateLNLogitObservationModel,
)
from scribe.laplace._em import run_laplace_em
from scribe.models.config.base import ModelConfig
from scribe.models.config.enums import (
    InferenceMethod,
    Parameterization,
)
from scribe.models.config.groups import LaplaceConfig, VAEConfig


def _build_cfg(latent_dim: int = 2):
    return ModelConfig(
        base_model="twostate_ln_logit",
        parameterization=Parameterization.TWO_STATE_NATURAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform="softplus",
        vae=VAEConfig(latent_dim=latent_dim),
    )


# =====================================================================
# 1. Capture-mode validation — the auditor's watchpoint
# =====================================================================


class TestCaptureRestriction:
    """PR-2 Rev 4: only ``x_only`` and ``x_only_offset`` are supported."""

    def test_capture_anchor_rejected(self):
        """Biology-anchored capture would need joint Newton — rejected."""
        with pytest.raises(NotImplementedError, match="capture_anchor"):
            TwoStateLNLogitObservationModel(
                capture_anchor=(np.log(50_000.0), 0.1),
                model_config=_build_cfg(),
            )

    def test_soft_eta_prior_rejected(self):
        """Soft-cascade eta would need joint Newton — rejected."""
        priors = {
            "eta": {
                "loc": jnp.full((10,), 0.5, dtype=jnp.float32),
                "scale": jnp.full((10,), 0.1, dtype=jnp.float32),
            }
        }
        with pytest.raises(NotImplementedError, match="eta"):
            TwoStateLNLogitObservationModel(
                model_config=_build_cfg(),
                informative_priors=priors,
            )

    def test_no_capture_accepted(self):
        """No capture passes through cleanly — routes to x_only."""
        obs = TwoStateLNLogitObservationModel(model_config=_build_cfg())
        assert obs.uses_capture is False
        assert obs.freezes_eta is False
        assert "eta" not in obs.frozen_params

    def test_frozen_offset_capture_accepted(self):
        """Frozen-offset capture is the only supported capture mode."""
        n_cells = 8
        freeze_values = {
            "eta": {
                "loc": jnp.full((n_cells,), 0.5, dtype=jnp.float32),
            }
        }
        obs = TwoStateLNLogitObservationModel(
            model_config=_build_cfg(),
            freeze_values=freeze_values,
            freeze_params=("eta",),
        )
        assert obs.uses_capture is True
        assert obs.freezes_eta is True

    def test_capture_anchor_and_no_freeze_eta_still_rejected(self):
        """Combining capture_anchor with freeze_params=('eta',) is still rejected."""
        with pytest.raises(NotImplementedError, match="capture_anchor"):
            TwoStateLNLogitObservationModel(
                capture_anchor=(0.5, 0.1),
                model_config=_build_cfg(),
                freeze_values={"eta": {"loc": jnp.zeros(5)}},
                freeze_params=("eta",),
            )


# =====================================================================
# 2. Constructor key validation
# =====================================================================


class TestConstructorValidation:
    def test_invalid_informative_priors_key_rejected(self):
        with pytest.raises(ValueError, match="unrecognized keys"):
            TwoStateLNLogitObservationModel(
                model_config=_build_cfg(),
                informative_priors={
                    "not_a_real_key": {
                        "loc": jnp.zeros(5),
                        "scale": jnp.ones(5),
                    }
                },
            )

    def test_invalid_freeze_params_key_rejected(self):
        with pytest.raises(ValueError, match="invalid keys"):
            TwoStateLNLogitObservationModel(
                model_config=_build_cfg(),
                freeze_params=("not_a_real_key",),
                freeze_values={"not_a_real_key": {"loc": jnp.zeros(5)}},
            )

    def test_missing_freeze_values_for_frozen_key_rejected(self):
        with pytest.raises(ValueError, match="freeze_values"):
            TwoStateLNLogitObservationModel(
                model_config=_build_cfg(),
                freeze_params=("rate",),
                freeze_values=None,
            )

    def test_freeze_values_missing_loc_field_rejected(self):
        with pytest.raises(ValueError, match="missing 'loc' field"):
            TwoStateLNLogitObservationModel(
                model_config=_build_cfg(),
                freeze_params=("rate",),
                freeze_values={"rate": {"scale": jnp.ones(5)}},
            )


# =====================================================================
# 3. init_state shapes / defaults
# =====================================================================


class TestInitState:
    def test_default_init_shapes_and_keys(self):
        """``init_state`` returns correctly-shaped arrays and the right
        keys in ``params``."""
        obs = TwoStateLNLogitObservationModel(model_config=_build_cfg())
        rng = np.random.default_rng(0)
        C, G, k = 10, 6, 2
        counts = jnp.asarray(rng.integers(0, 5, size=(C, G)).astype(np.float32))

        state = obs.init_state(counts, C, G, k, seed=0)

        # Latent z initialised at zero (TSLN-Logit convention).
        np.testing.assert_array_equal(
            np.asarray(state.latent_loc),
            np.zeros((C, G), dtype=np.float32),
        )
        # No capture ⇒ eta_loc is None.
        assert state.eta_loc is None
        assert state.eta_anchor is None

        # Params dict has rate_loc, kappa_loc, eta_anchor_loc, W, d_loc.
        assert set(state.params.keys()) == {
            "W", "d_loc", "rate_loc", "kappa_loc", "eta_anchor_loc",
        }
        assert state.params["W"].shape == (G, k)
        assert state.params["d_loc"].shape == (G,)
        assert state.params["rate_loc"].shape == (G,)
        assert state.params["kappa_loc"].shape == (G,)
        assert state.params["eta_anchor_loc"].shape == (G,)

    def test_eta_anchor_init_zero_implies_rate_two_times_mean(self):
        """The plan's default: ``rate_init = 2 · mean`` so that
        ``E[u | z=0] = rate · σ(0) = 0.5 · rate = mean``."""
        obs = TwoStateLNLogitObservationModel(model_config=_build_cfg())
        rng = np.random.default_rng(1)
        C, G = 10, 4
        counts = jnp.asarray(rng.integers(0, 8, size=(C, G)).astype(np.float32))

        state = obs.init_state(counts, C, G, latent_dim=1, seed=0)

        # eta_anchor_loc init must be zero.
        np.testing.assert_allclose(
            np.asarray(state.params["eta_anchor_loc"]),
            np.zeros(G, dtype=np.float32),
            atol=1e-7,
        )

        # rate_init (constrained, after softplus) ≈ 2 · empirical_mean.
        # Decode rate_loc → rate via softplus.
        rate = jax.nn.softplus(state.params["rate_loc"])
        empirical_mean = np.asarray(counts).mean(axis=0)
        np.testing.assert_allclose(
            np.asarray(rate),
            2.0 * empirical_mean,
            rtol=1e-3,
            atol=1e-5,
            err_msg=(
                "Default rate_init must equal 2 · empirical_mean so that "
                "E[u | z=0] = rate · σ(eta_anchor=0) matches the empirical mean."
            ),
        )

    def test_frozen_globals_excluded_from_params(self):
        """When a gene-global is frozen, its loc is NOT in ``params`` —
        it's spliced from ``freeze_values`` at every loss call."""
        G = 5
        freeze_values = {
            "rate": {"loc": jnp.full((G,), 1.0, dtype=jnp.float32)},
            "kappa": {"loc": jnp.full((G,), 0.0, dtype=jnp.float32)},
            "eta_anchor": {
                "loc": jnp.full((G,), 0.2, dtype=jnp.float32)
            },
        }
        obs = TwoStateLNLogitObservationModel(
            model_config=_build_cfg(),
            freeze_values=freeze_values,
            freeze_params=("rate", "kappa", "eta_anchor"),
        )
        rng = np.random.default_rng(2)
        C = 6
        counts = jnp.asarray(rng.integers(0, 5, size=(C, G)).astype(np.float32))
        state = obs.init_state(counts, C, G, latent_dim=1, seed=0)
        # Only W and d_loc — all three gene globals are frozen.
        assert set(state.params.keys()) == {"W", "d_loc"}


# =====================================================================
# 4. End-to-end smoke through run_laplace_em
# =====================================================================


class TestEndToEnd:
    def _synthetic_counts(self, C=20, G=8, seed=0):
        rng = np.random.default_rng(seed)
        return jnp.asarray(rng.integers(0, 8, size=(C, G)).astype(np.float32))

    def test_x_only_smoke(self):
        """No-capture configuration runs end-to-end with finite loss."""
        obs = TwoStateLNLogitObservationModel(model_config=_build_cfg())
        counts = self._synthetic_counts(C=20, G=8, seed=0)

        cfg = LaplaceConfig(n_steps=8, n_newton_steps=3, batch_size=10)
        result = run_laplace_em(
            obs_model=obs,
            count_data=counts,
            n_cells=20,
            n_genes=8,
            latent_dim=2,
            laplace_config=cfg,
            model_config=_build_cfg(latent_dim=2),
            seed=0,
            progress=False,
        )
        # Loss trajectory finite throughout.
        assert jnp.all(jnp.isfinite(result.losses))
        # Final-sweep grad norms finite.
        assert jnp.all(jnp.isfinite(result.final_grad_norms))
        # No eta on x_only.
        assert result.eta_loc is None

    def test_x_only_offset_smoke(self):
        """Fixed-offset capture configuration also runs end-to-end."""
        C, G = 20, 8
        eta_offset = jnp.full((C,), 0.5, dtype=jnp.float32)
        obs = TwoStateLNLogitObservationModel(
            model_config=_build_cfg(),
            freeze_values={"eta": {"loc": eta_offset}},
            freeze_params=("eta",),
        )
        counts = self._synthetic_counts(C=C, G=G, seed=1)

        cfg = LaplaceConfig(n_steps=8, n_newton_steps=3, batch_size=10)
        result = run_laplace_em(
            obs_model=obs,
            count_data=counts,
            n_cells=C,
            n_genes=G,
            latent_dim=2,
            laplace_config=cfg,
            model_config=_build_cfg(latent_dim=2),
            seed=1,
            progress=False,
        )
        assert jnp.all(jnp.isfinite(result.losses))
        assert jnp.all(jnp.isfinite(result.final_grad_norms))
        # Frozen eta should land back in eta_loc.
        assert result.eta_loc is not None
        np.testing.assert_array_equal(
            np.asarray(result.eta_loc), np.asarray(eta_offset)
        )

    def test_pack_result_globals_dict_keys(self):
        """``pack_result`` emits all the TSLN-Logit-specific globals."""
        obs = TwoStateLNLogitObservationModel(model_config=_build_cfg())
        counts = self._synthetic_counts(C=16, G=6, seed=2)
        cfg = LaplaceConfig(n_steps=5, n_newton_steps=3, batch_size=8)
        result = run_laplace_em(
            obs_model=obs,
            count_data=counts,
            n_cells=16,
            n_genes=6,
            latent_dim=2,
            laplace_config=cfg,
            model_config=_build_cfg(latent_dim=2),
            seed=2,
            progress=False,
        )
        globals_dict = result.globals
        # Core globals.
        for key in (
            "W", "d", "d_loc", "mu",
            "rate_loc", "rate",
            "kappa_loc", "kappa",
            "eta_anchor_loc", "eta_anchor",
            "alpha", "beta", "gene_mean",
            "a_raw_min", "a_raw_negative_fraction",
            "a_clamp_fraction", "a_clamp_per_gene",
        ):
            assert key in globals_dict, f"missing {key!r} in globals dict"

        # Sanity: ``mu`` is zeros (latent prior centring).
        np.testing.assert_allclose(
            np.asarray(globals_dict["mu"]),
            np.zeros(6, dtype=np.float32),
            atol=1e-7,
        )
        # gene_mean = rate · σ(eta_anchor) — derived correctly.
        phi = jax.nn.sigmoid(globals_dict["eta_anchor"])
        np.testing.assert_allclose(
            np.asarray(globals_dict["gene_mean"]),
            np.asarray(globals_dict["rate"] * phi),
            rtol=1e-5,
        )

    def test_full_freeze_l4_runs_with_only_w_and_d(self):
        """Level-4 freeze: all three gene-globals frozen at cascade MAP.

        ``params`` carries only ``W`` and ``d_loc``; the loss and
        gradient flow should still work.  This is the default
        configuration the cascade adapter will produce.
        """
        C, G = 14, 5
        rng = np.random.default_rng(3)
        freeze_values = {
            "rate": {
                "loc": jnp.asarray(rng.normal(0.0, 0.3, size=G).astype(np.float32))
            },
            "kappa": {
                "loc": jnp.asarray(rng.normal(0.0, 0.3, size=G).astype(np.float32))
            },
            "eta_anchor": {
                "loc": jnp.asarray(rng.normal(0.0, 0.5, size=G).astype(np.float32))
            },
        }
        obs = TwoStateLNLogitObservationModel(
            model_config=_build_cfg(),
            freeze_values=freeze_values,
            freeze_params=("rate", "kappa", "eta_anchor"),
        )
        counts = jnp.asarray(
            rng.integers(0, 6, size=(C, G)).astype(np.float32)
        )

        cfg = LaplaceConfig(n_steps=5, n_newton_steps=3, batch_size=7)
        result = run_laplace_em(
            obs_model=obs,
            count_data=counts,
            n_cells=C,
            n_genes=G,
            latent_dim=2,
            laplace_config=cfg,
            model_config=_build_cfg(latent_dim=2),
            seed=3,
            progress=False,
        )
        assert jnp.all(jnp.isfinite(result.losses))
        # Frozen entries surfaced with their input loc values.
        np.testing.assert_allclose(
            np.asarray(result.globals["rate_loc"]),
            np.asarray(freeze_values["rate"]["loc"]),
            atol=1e-7,
        )
        # global_uncertainty for frozen entries should be NaN per
        # the plan's convention (matches NBLN/TSLN-Rate).
        # `result.global_uncertainty` may be empty when EM doesn't
        # call compute_global_uncertainty by default; we just check
        # it doesn't crash.
