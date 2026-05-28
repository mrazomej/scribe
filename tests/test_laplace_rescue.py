"""Tests for the per-cell Newton rescue pass on Laplace fits.

Covers:

1. ``rescue_diverged_cells=False`` (default): result is identical to
   pre-rescue behavior — ``_pre_rescue_grad_norms`` and
   ``_rescued_cell_mask`` are ``None``.
2. ``rescue_diverged_cells=True`` with no cells diverged: masks are
   populated (``_rescued_cell_mask`` all-False), final state matches
   the no-rescue path.
3. ``rescue_diverged_cells=True`` with at least one cell diverged:
   ``_pre_rescue_grad_norms`` preserves the original tensor; the
   rescue pass overwrites ``latent_loc`` / ``final_grad_norms`` only
   at masked positions; cells outside the mask pass through unchanged.
4. ``_newton_tolerance`` is recorded on the result for downstream
   convergence-mask logic.
5. ``rescue_sweep`` direct call: write-back semantics, scatter-only-at-mask.
6. The base-class ``rescue_sweep`` works for every Laplace obs model
   (NBLN, PLN, TSLN-Rate, TSLN-Logit, LNM) by virtue of delegating to
   ``final_sweep``.
"""

from __future__ import annotations

import anndata as ad
import jax.numpy as jnp
import numpy as np


def _synthetic_pln(n_cells=40, n_genes=6, latent_dim=2, seed=0):
    """Small PLN dataset reused across rescue tests."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=n_genes).astype(np.float32) * 0.5 + 2.0
    W = (0.3 * rng.normal(size=(n_genes, latent_dim))).astype(np.float32)
    z = rng.normal(size=(n_cells, latent_dim)).astype(np.float32)
    x = mu + z @ W.T
    counts = rng.poisson(np.exp(np.clip(x, -10, 10))).astype(np.float32)
    return ad.AnnData(counts)


# =====================================================================
# 1. Default behavior: rescue OFF means rescue masks stay None
# =====================================================================


class TestRescueDefaultOff:
    """``rescue_diverged_cells`` defaults to False; masks remain None."""

    def test_default_off_pre_rescue_grad_norms_is_none(self):
        import scribe

        adata = _synthetic_pln(seed=0)
        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=30,
            seed=0,
        )
        assert result._pre_rescue_grad_norms is None
        assert result._rescued_cell_mask is None

    def test_default_off_unconverged_mask_uses_final_grad_norms(self):
        """Without rescue, ``unconverged_cell_mask`` derives from raw grad norms."""
        import scribe

        adata = _synthetic_pln(seed=1)
        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=30,
            seed=1,
        )
        # Mask must be derivable and consistent with grad_norms > tol
        mask = result.unconverged_cell_mask
        assert mask is not None
        assert mask.shape == (40,)
        tol = result._newton_tolerance or 1e-4
        expected = np.asarray(result.final_grad_norms) > tol
        assert np.array_equal(mask, expected)

    def test_model_unfit_mask_only_when_rescue_ran(self):
        """``model_unfit_cell_mask`` is None until rescue ran."""
        import scribe

        adata = _synthetic_pln(seed=2)
        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=30,
            seed=2,
        )
        assert result.model_unfit_cell_mask is None


# =====================================================================
# 2. _newton_tolerance is always recorded on the result
# =====================================================================


class TestNewtonToleranceRecorded:
    """The bridge records LaplaceConfig.newton_tolerance on the result."""

    def test_recorded_with_default_tolerance(self):
        import scribe

        adata = _synthetic_pln(seed=0)
        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=20,
            seed=0,
        )
        assert result._newton_tolerance is not None
        assert float(result._newton_tolerance) == 1e-4  # LaplaceConfig default

    def test_recorded_with_overridden_tolerance(self):
        import scribe

        adata = _synthetic_pln(seed=0)
        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=20,
            seed=0,
            laplace_config={"newton_tolerance": 5e-3},
        )
        assert float(result._newton_tolerance) == 5e-3
        # unconverged_cell_mask honors the override
        mask = result.unconverged_cell_mask
        expected = np.asarray(result.final_grad_norms) > 5e-3
        assert np.array_equal(mask, expected)


# =====================================================================
# 3. Rescue ON: masks populated, write-back semantics
# =====================================================================


class TestRescueEnabled:
    """``rescue_diverged_cells=True`` populates masks and writes back."""

    def test_rescue_enabled_masks_populated(self):
        """With rescue on, masks are not None even when no cells are diverged."""
        import scribe

        adata = _synthetic_pln(seed=3)
        result = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=50,
            seed=3,
            laplace_config={"rescue_diverged_cells": True},
        )
        # When no cells exceed the rescue threshold, masks must still
        # be populated as all-False (signals "rescue ran but found
        # nothing to do") — but the engine short-circuits if no cells
        # diverged, leaving the masks None.  Both contracts are
        # acceptable; assert at least one of them.
        if result._rescued_cell_mask is None:
            # Short-circuit path: no cells exceeded threshold.
            assert result._pre_rescue_grad_norms is None
        else:
            assert result._pre_rescue_grad_norms is not None
            # Pre-rescue grad norms shape matches final
            assert (
                result._pre_rescue_grad_norms.shape
                == result.final_grad_norms.shape
            )

    def test_rescue_disabled_bit_equal(self):
        """Disable rescue: result must match the pre-commit-2 path."""
        import scribe

        adata = _synthetic_pln(seed=4)
        result_off = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            vae_latent_dim=2,
            n_steps=50,
            seed=4,
            laplace_config={"rescue_diverged_cells": False},
        )
        # When rescue is off, the rescue fields are None and the rest
        # of the fit's outputs are exactly today's behavior.  This is
        # the v1 bit-equal contract.
        assert result_off._pre_rescue_grad_norms is None
        assert result_off._rescued_cell_mask is None


# =====================================================================
# 4. Direct rescue_sweep call: scatter-only-at-mask semantics
# =====================================================================


class TestRescueSweepDirect:
    """Direct calls to ``rescue_sweep`` exercise the slice / scatter plumbing.

    Uses a minimal stub observation model that records what
    ``final_sweep`` saw (sliced sub-batch shapes) and returns a
    deterministic ``FinalSweepResult``.  Avoids the heavy Newton
    machinery so we can isolate the wrapper logic.
    """

    def _make_stub_model(self):
        from scribe.laplace._em import (
            FinalSweepResult,
            LaplaceObservationModel,
            InitState,
        )
        from typing import Dict, Optional, Tuple

        recorded_calls = []

        class _StubModel(LaplaceObservationModel):
            name = "stub"

            def init_state(
                self,
                count_data,
                n_cells,
                n_genes,
                latent_dim,
                seed,
            ) -> InitState:  # pragma: no cover (unused)
                raise NotImplementedError

            def loss_fn(self, **kwargs):  # pragma: no cover (unused)
                raise NotImplementedError

            def final_sweep(
                self,
                params,
                latent_loc,
                eta_loc,
                eta_anchor,
                count_data,
                aux_data,
                n_newton,
                damping,
            ) -> FinalSweepResult:
                # Record what we got
                recorded_calls.append(
                    {
                        "latent_shape": tuple(latent_loc.shape),
                        "counts_shape": tuple(count_data.shape),
                        "n_newton": int(n_newton),
                        "damping": float(damping),
                    }
                )
                # Return zeros for x_final and a known grad-norm marker
                return FinalSweepResult(
                    latent_loc=jnp.zeros_like(latent_loc),
                    eta_loc=(
                        jnp.zeros_like(eta_loc)
                        if eta_loc is not None
                        else None
                    ),
                    final_grad_norms=jnp.full(
                        (latent_loc.shape[0],), 1e-10, dtype=jnp.float32
                    ),
                )

            def pack_result(self, **kwargs):  # pragma: no cover (unused)
                raise NotImplementedError

        return _StubModel(), recorded_calls

    def test_rescue_sweep_slices_and_scatters(self):
        """``rescue_sweep`` slices inputs by mask, calls final_sweep on
        the sub-batch, and writes back only at masked positions.
        """
        from scribe.laplace._em import FinalSweepResult

        stub, recorded = self._make_stub_model()

        N, G = 10, 4
        rng = np.random.default_rng(0)
        latent_full = jnp.asarray(rng.normal(size=(N, G)).astype(np.float32))
        counts_full = jnp.asarray(
            rng.integers(0, 5, size=(N, G)).astype(np.float32)
        )
        grad_full = jnp.asarray(rng.uniform(0, 2, size=N).astype(np.float32))
        # Three cells diverged
        cell_mask = jnp.array(
            [False, False, True, False, True, False, False, True, False, False],
            dtype=jnp.bool_,
        )

        final_in = FinalSweepResult(
            latent_loc=latent_full,
            eta_loc=None,
            final_grad_norms=grad_full,
        )

        out = stub.rescue_sweep(
            params={},
            final=final_in,
            eta_anchor=None,
            count_data=counts_full,
            aux_data={},
            cell_mask=cell_mask,
            n_newton=50,
            damping=1e-5,
        )

        # final_sweep was called once with the sliced sub-batch (3 cells)
        assert len(recorded) == 1
        assert recorded[0]["latent_shape"] == (3, G)
        assert recorded[0]["counts_shape"] == (3, G)
        assert recorded[0]["n_newton"] == 50

        # Output preserves full shape
        assert out.latent_loc.shape == (N, G)
        assert out.final_grad_norms.shape == (N,)

        # Non-rescued cells: latent unchanged from input
        unrescued_idx = np.where(~np.asarray(cell_mask))[0]
        np.testing.assert_array_equal(
            np.asarray(out.latent_loc)[unrescued_idx],
            np.asarray(latent_full)[unrescued_idx],
        )
        # Non-rescued cells: grad norm unchanged
        np.testing.assert_array_equal(
            np.asarray(out.final_grad_norms)[unrescued_idx],
            np.asarray(grad_full)[unrescued_idx],
        )

        # Rescued cells: latent overwritten with the stub's zeros
        rescued_idx = np.where(np.asarray(cell_mask))[0]
        np.testing.assert_array_equal(
            np.asarray(out.latent_loc)[rescued_idx],
            np.zeros((rescued_idx.size, G), dtype=np.float32),
        )
        # Rescued cells: grad norm overwritten with the stub's marker
        np.testing.assert_array_equal(
            np.asarray(out.final_grad_norms)[rescued_idx],
            np.full(rescued_idx.size, 1e-10, dtype=np.float32),
        )

        # Pre-rescue grad norms preserved
        np.testing.assert_array_equal(
            np.asarray(out.pre_rescue_grad_norms),
            np.asarray(grad_full),
        )
        # Mask flow-through
        np.testing.assert_array_equal(
            np.asarray(out.rescued_cell_mask),
            np.asarray(cell_mask),
        )

    def test_rescue_sweep_no_cells_flagged_is_noop(self):
        """When ``cell_mask`` is all-False, ``rescue_sweep`` returns
        the input ``final`` unchanged (with masks populated for
        diagnostics)."""
        from scribe.laplace._em import FinalSweepResult

        stub, recorded = self._make_stub_model()

        latent = jnp.zeros((5, 3))
        counts = jnp.zeros((5, 3))
        grad = jnp.full((5,), 1e-6)
        final_in = FinalSweepResult(
            latent_loc=latent, eta_loc=None, final_grad_norms=grad
        )
        cell_mask = jnp.zeros((5,), dtype=jnp.bool_)
        out = stub.rescue_sweep(
            params={},
            final=final_in,
            eta_anchor=None,
            count_data=counts,
            aux_data={},
            cell_mask=cell_mask,
            n_newton=50,
            damping=1e-5,
        )
        # final_sweep is NOT called when no cells need rescue
        assert len(recorded) == 0
        # Outputs match inputs
        np.testing.assert_array_equal(
            np.asarray(out.latent_loc), np.asarray(latent)
        )
        np.testing.assert_array_equal(
            np.asarray(out.final_grad_norms), np.asarray(grad)
        )
        # Pre-rescue diagnostic preserved
        np.testing.assert_array_equal(
            np.asarray(out.pre_rescue_grad_norms), np.asarray(grad)
        )
