"""Tests for the r_floor and p_floor parameters in log-likelihood functions.

``r_floor`` clamps the NB dispersion parameter to a small positive value,
preventing NaN log-likelihoods when posterior samples produce r ≈ 0 due to
float32 underflow in wide variational guides.

``p_floor`` clips the success probability (or effective p_hat in VCP models)
away from 0 and 1.  Two degenerate cases it guards against:

1. **Hierarchical models** — ``phi_g → 0`` (float32 underflow) causes
   ``p_g = 1/(1+0) = 1.0``.  Then ``r * log(1 - p) = r * log(0)``:
   NaN without ``r_floor``, ``−∞`` with it.  ``p_floor`` caps p at
   ``1 - p_floor`` and fully prevents this.
2. **VCP models** — ``phi_capture → ∞`` (float32 overflow) causes
   ``p_capture = 0``, ``p_hat = 0``.  Then for zero-count genes,
   ``NB(r, 0).log_prob(0) = 0 * log(0) = NaN``, which r_floor cannot fix.

Each test verifies one of the following properties:

1. When ``r`` is near-zero and ``r_floor`` is active (default 1e-6), no NaN
   values appear in the output.
2. When ``r_floor=0.0`` the floor is disabled and NaN values do appear for
   degenerate ``r``.
3. For well-behaved ``r`` values the floor has no effect on the result.
4. All 8 log-likelihood functions accept and forward ``r_floor`` correctly.
5. When ``p`` hits 1.0 exactly (simulating hierarchical phi→0 underflow),
   ``p_floor`` prevents NaN (and also prevents the residual −∞ left by
   r_floor alone for the degenerate p=1 case).
6. When ``p_capture`` is 0 (VCP degenerate), ``p_floor`` prevents NaN.
7. ``p_floor=0.0`` disables the guard, allowing NaN for degenerate p.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from scribe.models.log_likelihood import (
    nbdm_log_likelihood,
    nbvcp_log_likelihood,
    zinb_log_likelihood,
    zinbvcp_log_likelihood,
    nbdm_mixture_log_likelihood,
    zinb_mixture_log_likelihood,
    nbvcp_mixture_log_likelihood,
    zinbvcp_mixture_log_likelihood,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_CELLS = 20
N_GENES = 10
N_COMP = 3
RNG = np.random.default_rng(0)

# Small but valid counts
COUNTS = jnp.array(RNG.poisson(5.0, (N_CELLS, N_GENES)), dtype=jnp.float32)


def _base_params(r_val: float) -> dict:
    """Return minimal NBDM params with the given r value for all genes."""
    return {
        "p": jnp.array(0.3),
        "r": jnp.full((N_GENES,), r_val, dtype=jnp.float32),
    }


def _nbvcp_params(r_val: float) -> dict:
    """Return minimal NBVCP params (includes p_capture per cell)."""
    return {
        "p": jnp.array(0.3),
        "r": jnp.full((N_GENES,), r_val, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
    }


def _zinb_params(r_val: float) -> dict:
    """Return minimal ZINB params."""
    return {
        "p": jnp.array(0.3),
        "r": jnp.full((N_GENES,), r_val, dtype=jnp.float32),
        "gate": jnp.full((N_GENES,), 0.1, dtype=jnp.float32),
    }


def _zinbvcp_params(r_val: float) -> dict:
    """Return minimal ZINBVCP params."""
    return {
        "p": jnp.array(0.3),
        "r": jnp.full((N_GENES,), r_val, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
        "gate": jnp.full((N_GENES,), 0.1, dtype=jnp.float32),
    }


def _mix_params(r_val: float) -> dict:
    """Return minimal NBDM mixture params with n_components=3."""
    return {
        "mixing_weights": jnp.full((N_COMP,), 1.0 / N_COMP, dtype=jnp.float32),
        "p": jnp.array(0.3),
        "r": jnp.full((N_COMP, N_GENES), r_val, dtype=jnp.float32),
    }


def _zinb_mix_params(r_val: float) -> dict:
    """Return minimal ZINB mixture params."""
    return {
        "mixing_weights": jnp.full((N_COMP,), 1.0 / N_COMP, dtype=jnp.float32),
        "p": jnp.array(0.3),
        "r": jnp.full((N_COMP, N_GENES), r_val, dtype=jnp.float32),
        "gate": jnp.full((N_COMP, N_GENES), 0.1, dtype=jnp.float32),
    }


def _nbvcp_mix_params(r_val: float) -> dict:
    """Return minimal NBVCP mixture params."""
    return {
        "mixing_weights": jnp.full((N_COMP,), 1.0 / N_COMP, dtype=jnp.float32),
        "p": jnp.array(0.3),
        "r": jnp.full((N_COMP, N_GENES), r_val, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
    }


def _zinbvcp_mix_params(r_val: float) -> dict:
    """Return minimal ZINBVCP mixture params."""
    return {
        "mixing_weights": jnp.full((N_COMP,), 1.0 / N_COMP, dtype=jnp.float32),
        "p": jnp.array(0.3),
        "r": jnp.full((N_COMP, N_GENES), r_val, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
        "gate": jnp.full((N_COMP, N_GENES), 0.1, dtype=jnp.float32),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_nan(arr) -> bool:
    """Return True if any element is NaN."""
    return bool(jnp.any(jnp.isnan(arr)))


# ---------------------------------------------------------------------------
# NBDM
# ---------------------------------------------------------------------------


def test_nbdm_r_floor_removes_nan():
    """r_floor should prevent NaN when r is near-zero (float32 underflow)."""
    # r = 0 triggers lgamma(0) = NaN
    ll = nbdm_log_likelihood(COUNTS, _base_params(0.0), r_floor=1e-6)
    assert not _has_nan(ll), "r_floor should have eliminated NaN"


def test_nbdm_r_floor_zero_allows_nan():
    """r_floor=0.0 disables the floor; near-zero r should produce NaN."""
    ll = nbdm_log_likelihood(COUNTS, _base_params(0.0), r_floor=0.0)
    assert _has_nan(ll), "Disabling r_floor should allow NaN for r=0"


def test_nbdm_r_floor_no_effect_for_valid_r():
    """r_floor must not change results for well-behaved r values."""
    params = _base_params(1.0)
    ll_no_floor = nbdm_log_likelihood(COUNTS, params, r_floor=0.0)
    ll_with_floor = nbdm_log_likelihood(COUNTS, params, r_floor=1e-6)
    np.testing.assert_allclose(
        np.asarray(ll_no_floor), np.asarray(ll_with_floor), rtol=1e-5
    )


# ---------------------------------------------------------------------------
# ZINB
# ---------------------------------------------------------------------------


def test_zinb_r_floor_removes_nan():
    """r_floor should prevent NaN in ZINB log-likelihood for r=0."""
    ll = zinb_log_likelihood(COUNTS, _zinb_params(0.0), r_floor=1e-6)
    assert not _has_nan(ll)


def test_zinb_r_floor_zero_allows_nan():
    """Disabling r_floor in ZINB should allow NaN for r=0."""
    ll = zinb_log_likelihood(COUNTS, _zinb_params(0.0), r_floor=0.0)
    assert _has_nan(ll)


# ---------------------------------------------------------------------------
# NBVCP
# ---------------------------------------------------------------------------


def test_nbvcp_r_floor_removes_nan():
    """r_floor should prevent NaN in NBVCP log-likelihood for r=0."""
    ll = nbvcp_log_likelihood(COUNTS, _nbvcp_params(0.0), r_floor=1e-6)
    assert not _has_nan(ll)


def test_nbvcp_r_floor_zero_allows_nan():
    """Disabling r_floor in NBVCP should allow NaN for r=0."""
    ll = nbvcp_log_likelihood(COUNTS, _nbvcp_params(0.0), r_floor=0.0)
    assert _has_nan(ll)


# ---------------------------------------------------------------------------
# ZINBVCP
# ---------------------------------------------------------------------------


def test_zinbvcp_r_floor_removes_nan():
    """r_floor should prevent NaN in ZINBVCP log-likelihood for r=0."""
    ll = zinbvcp_log_likelihood(COUNTS, _zinbvcp_params(0.0), r_floor=1e-6)
    assert not _has_nan(ll)


def test_zinbvcp_r_floor_zero_allows_nan():
    """Disabling r_floor in ZINBVCP should allow NaN for r=0."""
    ll = zinbvcp_log_likelihood(COUNTS, _zinbvcp_params(0.0), r_floor=0.0)
    assert _has_nan(ll)


# ---------------------------------------------------------------------------
# Mixture models
# ---------------------------------------------------------------------------


def test_nbdm_mixture_r_floor_removes_nan():
    """r_floor should prevent NaN in NBDM mixture log-likelihood for r=0."""
    ll = nbdm_mixture_log_likelihood(COUNTS, _mix_params(0.0), r_floor=1e-6)
    assert not _has_nan(ll)


def test_nbdm_mixture_r_floor_zero_allows_nan():
    """Disabling r_floor in NBDM mixture should allow NaN for r=0."""
    ll = nbdm_mixture_log_likelihood(COUNTS, _mix_params(0.0), r_floor=0.0)
    assert _has_nan(ll)


def test_zinb_mixture_r_floor_removes_nan():
    """r_floor should prevent NaN in ZINB mixture log-likelihood for r=0."""
    ll = zinb_mixture_log_likelihood(COUNTS, _zinb_mix_params(0.0), r_floor=1e-6)
    assert not _has_nan(ll)


def test_zinb_mixture_r_floor_zero_allows_nan():
    """Disabling r_floor in ZINB mixture should allow NaN for r=0."""
    ll = zinb_mixture_log_likelihood(COUNTS, _zinb_mix_params(0.0), r_floor=0.0)
    assert _has_nan(ll)


def test_nbvcp_mixture_r_floor_removes_nan():
    """r_floor should prevent NaN in NBVCP mixture log-likelihood for r=0."""
    ll = nbvcp_mixture_log_likelihood(COUNTS, _nbvcp_mix_params(0.0), r_floor=1e-6)
    assert not _has_nan(ll)


def test_nbvcp_mixture_r_floor_zero_allows_nan():
    """Disabling r_floor in NBVCP mixture should allow NaN for r=0."""
    ll = nbvcp_mixture_log_likelihood(COUNTS, _nbvcp_mix_params(0.0), r_floor=0.0)
    assert _has_nan(ll)


def test_zinbvcp_mixture_r_floor_removes_nan():
    """r_floor should prevent NaN in ZINBVCP mixture log-likelihood for r=0."""
    ll = zinbvcp_mixture_log_likelihood(COUNTS, _zinbvcp_mix_params(0.0), r_floor=1e-6)
    assert not _has_nan(ll)


def test_zinbvcp_mixture_r_floor_zero_allows_nan():
    """Disabling r_floor in ZINBVCP mixture should allow NaN for r=0."""
    ll = zinbvcp_mixture_log_likelihood(COUNTS, _zinbvcp_mix_params(0.0), r_floor=0.0)
    assert _has_nan(ll)


# ---------------------------------------------------------------------------
# Return-by-gene consistency
# ---------------------------------------------------------------------------


def test_r_floor_works_with_return_by_gene():
    """r_floor should also eliminate NaN when return_by='gene'."""
    ll = nbdm_log_likelihood(COUNTS, _base_params(0.0), r_floor=1e-6, return_by="gene")
    assert not _has_nan(ll)


def test_r_floor_default_is_active():
    """Calling without explicit r_floor should use the 1e-6 default."""
    # No r_floor keyword → default 1e-6 applies
    ll = nbdm_log_likelihood(COUNTS, _base_params(0.0))
    assert not _has_nan(ll), "Default r_floor=1e-6 should prevent NaN"


# ---------------------------------------------------------------------------
# p_floor — hierarchical p=1.0 case (phi_g → 0 in float32)
# ---------------------------------------------------------------------------


def _hierarchical_p1_params() -> dict:
    """Params mimicking a hierarchical posterior sample where phi_g underflowed
    to zero → p_g = 1/(1+0) = 1.0 for all genes.  r is set to a safe value so
    only the p=1 issue is tested in isolation.
    """
    return {
        "p": jnp.ones(N_GENES, dtype=jnp.float32),
        "r": jnp.full((N_GENES,), 1.0, dtype=jnp.float32),
    }


def _hierarchical_p1_vcp_params() -> dict:
    """Same as above but for VCP models."""
    return {
        "p": jnp.ones(N_GENES, dtype=jnp.float32),
        "r": jnp.full((N_GENES,), 1.0, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
    }


def _hierarchical_p1_zinb_params() -> dict:
    """ZINB with gene-specific p=1.0."""
    return {
        "p": jnp.ones(N_GENES, dtype=jnp.float32),
        "r": jnp.full((N_GENES,), 1.0, dtype=jnp.float32),
        "gate": jnp.full((N_GENES,), 0.1, dtype=jnp.float32),
    }


def _hierarchical_p1_zinbvcp_params() -> dict:
    """ZINBVCP with gene-specific p=1.0."""
    return {
        "p": jnp.ones(N_GENES, dtype=jnp.float32),
        "r": jnp.full((N_GENES,), 1.0, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
        "gate": jnp.full((N_GENES,), 0.1, dtype=jnp.float32),
    }


def test_nbdm_p_floor_removes_nan_for_p1():
    """p_floor should prevent NaN/−∞ when gene-specific p=1.0 (hierarchical)."""
    ll = nbdm_log_likelihood(COUNTS, _hierarchical_p1_params(), p_floor=1e-6)
    assert not _has_nan(ll), "p_floor should prevent NaN for p=1.0"
    assert not bool(jnp.any(jnp.isinf(ll))), "p_floor should prevent -inf for p=1.0"


def test_nbdm_p_floor_zero_allows_inf_for_p1():
    """p_floor=0.0 disables the guard; p=1.0 should give −∞ or NaN."""
    ll = nbdm_log_likelihood(COUNTS, _hierarchical_p1_params(), p_floor=0.0)
    has_bad = bool(jnp.any(jnp.isnan(ll))) or bool(jnp.any(jnp.isinf(ll)))
    assert has_bad, "p_floor=0 should leave degenerate values for p=1.0"


def test_zinb_p_floor_removes_nan_for_p1():
    """p_floor should prevent NaN for ZINB when gene-specific p=1.0."""
    ll = zinb_log_likelihood(COUNTS, _hierarchical_p1_zinb_params(), p_floor=1e-6)
    assert not _has_nan(ll)
    assert not bool(jnp.any(jnp.isinf(ll)))


# ---------------------------------------------------------------------------
# p_floor — VCP degenerate p_capture=0 case (phi_capture → ∞ in float32)
# ---------------------------------------------------------------------------


def _vcp_zero_capture_params() -> dict:
    """Params with p_capture=0 for all cells (phi_capture overflowed to ∞)."""
    return {
        "p": jnp.full((N_GENES,), 0.3, dtype=jnp.float32),
        "r": jnp.full((N_GENES,), 1.0, dtype=jnp.float32),
        "p_capture": jnp.zeros(N_CELLS, dtype=jnp.float32),
    }


def _zinbvcp_zero_capture_params() -> dict:
    return {
        "p": jnp.full((N_GENES,), 0.3, dtype=jnp.float32),
        "r": jnp.full((N_GENES,), 1.0, dtype=jnp.float32),
        "p_capture": jnp.zeros(N_CELLS, dtype=jnp.float32),
        "gate": jnp.full((N_GENES,), 0.1, dtype=jnp.float32),
    }


def test_nbvcp_p_floor_prevents_nan_for_zero_capture():
    """p_floor should prevent NaN when p_capture=0 (degenerate VCP sample)."""
    # Counts with zeros to trigger 0 * log(0) = NaN pathway without floor
    counts_with_zeros = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.float32)
    ll = nbvcp_log_likelihood(
        counts_with_zeros, _vcp_zero_capture_params(), p_floor=1e-6
    )
    assert not _has_nan(ll), "p_floor should prevent NaN for p_capture=0"


def test_nbvcp_p_floor_zero_allows_nan_for_zero_capture():
    """p_floor=0 allows NaN when p_capture=0 and counts have zeros."""
    counts_with_zeros = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.float32)
    ll = nbvcp_log_likelihood(
        counts_with_zeros, _vcp_zero_capture_params(), p_floor=0.0
    )
    assert _has_nan(ll), "p_floor=0 should allow NaN for p_capture=0"


def test_zinbvcp_p_floor_prevents_nan_for_zero_capture():
    """p_floor should prevent NaN in ZINBVCP when p_capture=0."""
    counts_with_zeros = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.float32)
    ll = zinbvcp_log_likelihood(
        counts_with_zeros, _zinbvcp_zero_capture_params(), p_floor=1e-6
    )
    assert not _has_nan(ll)


# ---------------------------------------------------------------------------
# p_floor — mixture model p=1.0 (hierarchical mixture)
# ---------------------------------------------------------------------------


def _mix_hierarchical_p1_params() -> dict:
    """Mixture NBDM params where p is per-component, per-gene = 1.0."""
    return {
        "mixing_weights": jnp.full((N_COMP,), 1.0 / N_COMP, dtype=jnp.float32),
        "p": jnp.ones((N_COMP, N_GENES), dtype=jnp.float32),
        "r": jnp.full((N_COMP, N_GENES), 1.0, dtype=jnp.float32),
    }


def test_nbdm_mixture_p_floor_removes_nan_for_p1():
    """p_floor should prevent NaN/−∞ in NBDM mixture when p=1.0 per gene."""
    ll = nbdm_mixture_log_likelihood(
        COUNTS, _mix_hierarchical_p1_params(), p_floor=1e-6
    )
    assert not _has_nan(ll)
    assert not bool(jnp.any(jnp.isinf(ll)))


def test_p_floor_no_effect_for_valid_p():
    """p_floor must not change results when p is well inside (0, 1)."""
    params = _base_params(1.0)
    ll_no_floor = nbdm_log_likelihood(COUNTS, params, p_floor=0.0)
    ll_with_floor = nbdm_log_likelihood(COUNTS, params, p_floor=1e-6)
    np.testing.assert_allclose(
        np.asarray(ll_no_floor), np.asarray(ll_with_floor), rtol=1e-5
    )


def test_p_floor_default_is_active():
    """Calling without explicit p_floor should use the 1e-6 default."""
    ll = nbdm_log_likelihood(COUNTS, _hierarchical_p1_params())
    assert not _has_nan(ll), "Default p_floor=1e-6 should prevent NaN for p=1.0"
