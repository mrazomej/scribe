"""Tests for numerical floor/clamp guards that prevent NaN in log-likelihoods.

Two layers of protection exist:

**Post-hoc evaluation** (``log_likelihood.py``):
    ``r_floor`` and ``p_floor`` are explicit keyword arguments that clamp ``r``
    and ``p`` (or ``p_hat``) before computing NB log-probabilities in
    already-fitted posterior samples.

**Training-time likelihoods** (``negative_binomial.py``, ``vcp.py``):
    A hard ``_P_EPS`` clamp is applied to ``p`` (or ``phi`` / ``p_hat``)
    inside the likelihood components that NumPyro evaluates during SVI.
    This prevents NaN in the ELBO when hierarchical priors produce extreme
    samples (e.g. ``phi_g -> 0`` causing ``p_g -> 1.0``).

Degenerate cases guarded against:

1. ``r ~ 0`` — ``lgamma(0) = NaN``; fixed by ``r_floor``.
2. ``phi_g -> 0`` (hierarchical underflow) — ``p_g = 1/(1+0) = 1.0``,
   ``r * log(1 - p) = r * log(0)`` — NaN/−inf; fixed by ``p_floor``
   (post-hoc) and ``_P_EPS`` clamp (training).
3. ``phi_capture -> inf`` (VCP overflow) — ``p_capture = 0``,
   ``p_hat = 0``, ``NB(r, 0).log_prob(0) = 0 * log(0) = NaN``; fixed
   by ``p_floor`` / ``_P_EPS`` clamp on ``p_hat``.
4. ``phi -> 0`` in VCP logits path — ``log(phi * ...) = log(0) = -inf``;
   fixed by ``jnp.maximum(phi, _P_EPS)`` before the log.
"""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from scribe.models.components.likelihoods import (
    NegativeBinomialLikelihood,
    ZeroInflatedNBLikelihood,
    NBWithVCPLikelihood,
    ZINBWithVCPLikelihood,
)
from scribe.models.components.likelihoods.negative_binomial import (
    _P_EPS as NB_P_EPS,
)
from scribe.models.components.likelihoods.vcp import (
    _P_EPS as VCP_P_EPS,
)
from scribe.core.axis_layout import AxisLayout


# Post-hoc log-likelihood evaluation now lives on ``Likelihood.log_prob``
# rather than as free functions.  Module-level singleton instances keep the
# tests terse - ``Likelihood`` subclasses are effectively stateless so
# reusing one instance is safe.
_NBDM_LIK = NegativeBinomialLikelihood()
_ZINB_LIK = ZeroInflatedNBLikelihood()
_NBVCP_LIK = NBWithVCPLikelihood()
_ZINBVCP_LIK = ZINBWithVCPLikelihood()


# Shape-derived layout helper used by the shim functions.
#
# The ``Likelihood.log_prob`` contract now requires an :class:`AxisLayout`
# per parameter key.  These tests construct many small param dicts with
# well-understood shapes (see ``_base_params`` / ``_mix_params`` et al.),
# so we infer the layout directly from ``ndim`` and the shape of each
# array relative to ``(N_CELLS, N_GENES, N_COMP)``.  This mirrors what
# production code gets from ``_build_canonical_layouts`` but without
# requiring a ``ModelConfig`` object.
def _infer_layout(
    key: str, arr: jnp.ndarray, *, n_components: int
) -> AxisLayout:
    """Return the :class:`AxisLayout` for *key* given its tensor shape.

    Parameters
    ----------
    key : str
        Canonical parameter name (e.g. ``"r"``, ``"gate"``, ``"p"``).
    arr : jnp.ndarray
        The parameter tensor.
    n_components : int
        Number of mixture components (``N_COMP``).  Only used to
        disambiguate mixture vs non-mixture layouts.

    Returns
    -------
    AxisLayout
        Layout with ``has_sample_dim=False``.
    """
    shape = tuple(arr.shape)
    ndim = len(shape)

    # Scalar -> empty layout.
    if ndim == 0:
        return AxisLayout(())

    # Cell-indexed parameters (``p_capture``, ``phi_capture``).
    if key in ("p_capture", "phi_capture", "eta_capture"):
        return AxisLayout(("cells",))

    # Mixing weights are (n_components,).
    if key in ("mixing_weights", "mixing_logits"):
        return AxisLayout(("components",))

    # 1-D tensors either are (n_components,) for mixture components-only
    # params or (n_genes,) for gene-indexed ones.  The test suite uses
    # gene-indexed 1-D tensors only for ``r``/``gate``/``p``; a pure
    # ``(n_components,)`` ``p`` shows up in mixture cases.
    if ndim == 1:
        if key == "p" and shape[0] == n_components:
            return AxisLayout(("components",))
        return AxisLayout(("genes",))

    # 2-D tensors are the mixture layouts ``(n_components, n_genes)``.
    if ndim == 2:
        return AxisLayout(("components", "genes"))

    raise ValueError(
        f"Cannot infer AxisLayout for key={key!r} with shape={shape}"
    )


def _layouts_for(params: dict) -> dict:
    """Build a ``{key: AxisLayout}`` dict for all entries in *params*."""
    return {
        k: _infer_layout(k, v, n_components=N_COMP) for k, v in params.items()
    }


def nbdm_log_likelihood(counts, params, **kwargs):
    """Shim forwarding to :meth:`NegativeBinomialLikelihood.log_prob`.

    The BNB dispatch inside ``_build_ll_count_dist`` picks the right base
    distribution based on the presence of ``bnb_concentration`` in
    ``params``; for the NBDM case (no such key) this is identical to the
    legacy ``nbdm_log_likelihood`` free function.
    """
    return _NBDM_LIK.log_prob(counts, params, _layouts_for(params), **kwargs)


def nbvcp_log_likelihood(counts, params, **kwargs):
    """Shim forwarding to :meth:`NBWithVCPLikelihood.log_prob`."""
    return _NBVCP_LIK.log_prob(counts, params, _layouts_for(params), **kwargs)


def zinb_log_likelihood(counts, params, **kwargs):
    """Shim forwarding to :meth:`ZeroInflatedNBLikelihood.log_prob`."""
    return _ZINB_LIK.log_prob(counts, params, _layouts_for(params), **kwargs)


def zinbvcp_log_likelihood(counts, params, **kwargs):
    """Shim forwarding to :meth:`ZINBWithVCPLikelihood.log_prob`."""
    return _ZINBVCP_LIK.log_prob(
        counts, params, _layouts_for(params), **kwargs
    )


# Mixture variants dispatch on the presence of ``"mixing_weights"`` in the
# params dict, so we reuse the non-mixture classes here as well.
nbdm_mixture_log_likelihood = nbdm_log_likelihood
zinb_mixture_log_likelihood = zinb_log_likelihood
nbvcp_mixture_log_likelihood = nbvcp_log_likelihood
zinbvcp_mixture_log_likelihood = zinbvcp_log_likelihood


# ===========================================================================
# Shared fixtures
# ===========================================================================

N_CELLS = 20
N_GENES = 10
N_COMP = 3
RNG = np.random.default_rng(0)

COUNTS = jnp.array(RNG.poisson(5.0, (N_CELLS, N_GENES)), dtype=jnp.float32)


def _has_nan(arr) -> bool:
    """Return True if any element is NaN."""
    return bool(jnp.any(jnp.isnan(arr)))


# ===========================================================================
# Parameter helpers — post-hoc evaluation
# ===========================================================================


def _base_params(r_val: float) -> dict:
    """Minimal NBDM params with the given r value for all genes."""
    return {
        "p": jnp.array(0.3),
        "r": jnp.full((N_GENES,), r_val, dtype=jnp.float32),
    }


def _nbvcp_params(r_val: float) -> dict:
    """Minimal NBVCP params (includes p_capture per cell)."""
    return {
        "p": jnp.array(0.3),
        "r": jnp.full((N_GENES,), r_val, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
    }


def _zinb_params(r_val: float) -> dict:
    """Minimal ZINB params."""
    return {
        "p": jnp.array(0.3),
        "r": jnp.full((N_GENES,), r_val, dtype=jnp.float32),
        "gate": jnp.full((N_GENES,), 0.1, dtype=jnp.float32),
    }


def _zinbvcp_params(r_val: float) -> dict:
    """Minimal ZINBVCP params."""
    return {
        "p": jnp.array(0.3),
        "r": jnp.full((N_GENES,), r_val, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
        "gate": jnp.full((N_GENES,), 0.1, dtype=jnp.float32),
    }


def _mix_params(r_val: float) -> dict:
    """Minimal NBDM mixture params with n_components=3."""
    return {
        "mixing_weights": jnp.full(
            (N_COMP,), 1.0 / N_COMP, dtype=jnp.float32
        ),
        "p": jnp.array(0.3),
        "r": jnp.full((N_COMP, N_GENES), r_val, dtype=jnp.float32),
    }


def _zinb_mix_params(r_val: float) -> dict:
    """Minimal ZINB mixture params."""
    return {
        "mixing_weights": jnp.full(
            (N_COMP,), 1.0 / N_COMP, dtype=jnp.float32
        ),
        "p": jnp.array(0.3),
        "r": jnp.full((N_COMP, N_GENES), r_val, dtype=jnp.float32),
        "gate": jnp.full((N_COMP, N_GENES), 0.1, dtype=jnp.float32),
    }


def _nbvcp_mix_params(r_val: float) -> dict:
    """Minimal NBVCP mixture params."""
    return {
        "mixing_weights": jnp.full(
            (N_COMP,), 1.0 / N_COMP, dtype=jnp.float32
        ),
        "p": jnp.array(0.3),
        "r": jnp.full((N_COMP, N_GENES), r_val, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
    }


def _zinbvcp_mix_params(r_val: float) -> dict:
    """Minimal ZINBVCP mixture params."""
    return {
        "mixing_weights": jnp.full(
            (N_COMP,), 1.0 / N_COMP, dtype=jnp.float32
        ),
        "p": jnp.array(0.3),
        "r": jnp.full((N_COMP, N_GENES), r_val, dtype=jnp.float32),
        "p_capture": jnp.full((N_CELLS,), 0.8, dtype=jnp.float32),
        "gate": jnp.full((N_COMP, N_GENES), 0.1, dtype=jnp.float32),
    }


# ===========================================================================
# Parameter helpers — hierarchical p=1.0 / p=0.0 edge cases
# ===========================================================================


def _hierarchical_p1_params() -> dict:
    """phi_g underflowed to 0 -> p_g = 1/(1+0) = 1.0 for all genes."""
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


def _vcp_zero_capture_params() -> dict:
    """p_capture=0 for all cells (phi_capture overflowed to inf)."""
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


def _mix_hierarchical_p1_params() -> dict:
    """Mixture NBDM params where p is per-component, per-gene = 1.0."""
    return {
        "mixing_weights": jnp.full(
            (N_COMP,), 1.0 / N_COMP, dtype=jnp.float32
        ),
        "p": jnp.ones((N_COMP, N_GENES), dtype=jnp.float32),
        "r": jnp.full((N_COMP, N_GENES), 1.0, dtype=jnp.float32),
    }


# ###########################################################################
# POST-HOC EVALUATION: r_floor tests (log_likelihood.py)
# ###########################################################################


class TestRFloorPostHoc:
    """r_floor in post-hoc log-likelihood evaluation."""

    def test_nbdm_r_floor_removes_nan(self):
        """r_floor prevents NaN when r is near-zero (float32 underflow)."""
        ll = nbdm_log_likelihood(COUNTS, _base_params(0.0), r_floor=1e-6)
        assert not _has_nan(ll), "r_floor should have eliminated NaN"

    def test_nbdm_r_floor_zero_allows_nan(self):
        """r_floor=0.0 disables the floor; near-zero r produces NaN."""
        ll = nbdm_log_likelihood(COUNTS, _base_params(0.0), r_floor=0.0)
        assert _has_nan(ll), "Disabling r_floor should allow NaN for r=0"

    def test_nbdm_r_floor_no_effect_for_valid_r(self):
        """r_floor does not change results for well-behaved r values."""
        params = _base_params(1.0)
        ll_no = nbdm_log_likelihood(COUNTS, params, r_floor=0.0)
        ll_yes = nbdm_log_likelihood(COUNTS, params, r_floor=1e-6)
        np.testing.assert_allclose(
            np.asarray(ll_no), np.asarray(ll_yes), rtol=1e-5
        )

    def test_zinb_r_floor_removes_nan(self):
        ll = zinb_log_likelihood(COUNTS, _zinb_params(0.0), r_floor=1e-6)
        assert not _has_nan(ll)

    def test_zinb_r_floor_zero_allows_nan(self):
        ll = zinb_log_likelihood(COUNTS, _zinb_params(0.0), r_floor=0.0)
        assert _has_nan(ll)

    def test_nbvcp_r_floor_removes_nan(self):
        ll = nbvcp_log_likelihood(COUNTS, _nbvcp_params(0.0), r_floor=1e-6)
        assert not _has_nan(ll)

    def test_nbvcp_r_floor_zero_allows_nan(self):
        ll = nbvcp_log_likelihood(COUNTS, _nbvcp_params(0.0), r_floor=0.0)
        assert _has_nan(ll)

    def test_zinbvcp_r_floor_removes_nan(self):
        ll = zinbvcp_log_likelihood(COUNTS, _zinbvcp_params(0.0), r_floor=1e-6)
        assert not _has_nan(ll)

    def test_zinbvcp_r_floor_zero_allows_nan(self):
        ll = zinbvcp_log_likelihood(COUNTS, _zinbvcp_params(0.0), r_floor=0.0)
        assert _has_nan(ll)

    # --- Mixtures ---

    def test_nbdm_mixture_r_floor_removes_nan(self):
        ll = nbdm_mixture_log_likelihood(
            COUNTS, _mix_params(0.0), r_floor=1e-6
        )
        assert not _has_nan(ll)

    def test_nbdm_mixture_r_floor_zero_allows_nan(self):
        ll = nbdm_mixture_log_likelihood(
            COUNTS, _mix_params(0.0), r_floor=0.0
        )
        assert _has_nan(ll)

    def test_zinb_mixture_r_floor_removes_nan(self):
        ll = zinb_mixture_log_likelihood(
            COUNTS, _zinb_mix_params(0.0), r_floor=1e-6
        )
        assert not _has_nan(ll)

    def test_zinb_mixture_r_floor_zero_allows_nan(self):
        ll = zinb_mixture_log_likelihood(
            COUNTS, _zinb_mix_params(0.0), r_floor=0.0
        )
        assert _has_nan(ll)

    def test_nbvcp_mixture_r_floor_removes_nan(self):
        ll = nbvcp_mixture_log_likelihood(
            COUNTS, _nbvcp_mix_params(0.0), r_floor=1e-6
        )
        assert not _has_nan(ll)

    def test_nbvcp_mixture_r_floor_zero_allows_nan(self):
        ll = nbvcp_mixture_log_likelihood(
            COUNTS, _nbvcp_mix_params(0.0), r_floor=0.0
        )
        assert _has_nan(ll)

    def test_zinbvcp_mixture_r_floor_removes_nan(self):
        ll = zinbvcp_mixture_log_likelihood(
            COUNTS, _zinbvcp_mix_params(0.0), r_floor=1e-6
        )
        assert not _has_nan(ll)

    def test_zinbvcp_mixture_r_floor_zero_allows_nan(self):
        ll = zinbvcp_mixture_log_likelihood(
            COUNTS, _zinbvcp_mix_params(0.0), r_floor=0.0
        )
        assert _has_nan(ll)

    # --- Misc ---

    def test_r_floor_works_with_return_by_gene(self):
        """r_floor eliminates NaN when return_by='gene'."""
        ll = nbdm_log_likelihood(
            COUNTS, _base_params(0.0), r_floor=1e-6, return_by="gene"
        )
        assert not _has_nan(ll)

    def test_r_floor_default_is_active(self):
        """Default r_floor=1e-6 prevents NaN without explicit keyword."""
        ll = nbdm_log_likelihood(COUNTS, _base_params(0.0))
        assert not _has_nan(ll), "Default r_floor=1e-6 should prevent NaN"


# ###########################################################################
# POST-HOC EVALUATION: p_floor tests (log_likelihood.py)
# ###########################################################################


class TestPFloorPostHoc:
    """p_floor in post-hoc log-likelihood evaluation."""

    def test_nbdm_p_floor_removes_nan_for_p1(self):
        """p_floor prevents NaN/−inf when gene-specific p=1.0."""
        ll = nbdm_log_likelihood(
            COUNTS, _hierarchical_p1_params(), p_floor=1e-6
        )
        assert not _has_nan(ll), "p_floor should prevent NaN for p=1.0"
        assert not bool(jnp.any(jnp.isinf(ll)))

    def test_nbdm_p_floor_zero_allows_inf_for_p1(self):
        """p_floor=0.0 disables the guard; p=1.0 gives −inf or NaN."""
        ll = nbdm_log_likelihood(
            COUNTS, _hierarchical_p1_params(), p_floor=0.0
        )
        has_bad = bool(jnp.any(jnp.isnan(ll))) or bool(
            jnp.any(jnp.isinf(ll))
        )
        assert has_bad, "p_floor=0 should leave degenerate values for p=1.0"

    def test_zinb_p_floor_removes_nan_for_p1(self):
        ll = zinb_log_likelihood(
            COUNTS, _hierarchical_p1_zinb_params(), p_floor=1e-6
        )
        assert not _has_nan(ll)
        assert not bool(jnp.any(jnp.isinf(ll)))

    def test_nbvcp_p_floor_prevents_nan_for_zero_capture(self):
        """p_floor prevents NaN when p_capture=0 (degenerate VCP sample)."""
        zeros = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.float32)
        ll = nbvcp_log_likelihood(
            zeros, _vcp_zero_capture_params(), p_floor=1e-6
        )
        assert not _has_nan(ll)

    def test_nbvcp_p_floor_zero_allows_nan_for_zero_capture(self):
        zeros = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.float32)
        ll = nbvcp_log_likelihood(
            zeros, _vcp_zero_capture_params(), p_floor=0.0
        )
        assert _has_nan(ll)

    def test_zinbvcp_p_floor_prevents_nan_for_zero_capture(self):
        zeros = jnp.zeros((N_CELLS, N_GENES), dtype=jnp.float32)
        ll = zinbvcp_log_likelihood(
            zeros, _zinbvcp_zero_capture_params(), p_floor=1e-6
        )
        assert not _has_nan(ll)

    def test_nbdm_mixture_p_floor_removes_nan_for_p1(self):
        ll = nbdm_mixture_log_likelihood(
            COUNTS, _mix_hierarchical_p1_params(), p_floor=1e-6
        )
        assert not _has_nan(ll)
        assert not bool(jnp.any(jnp.isinf(ll)))

    def test_p_floor_no_effect_for_valid_p(self):
        """p_floor does not change results when p is well inside (0, 1)."""
        params = _base_params(1.0)
        ll_no = nbdm_log_likelihood(COUNTS, params, p_floor=0.0)
        ll_yes = nbdm_log_likelihood(COUNTS, params, p_floor=1e-6)
        np.testing.assert_allclose(
            np.asarray(ll_no), np.asarray(ll_yes), rtol=1e-5
        )

    def test_p_floor_default_is_active(self):
        """Default p_floor=1e-6 prevents NaN without explicit keyword."""
        ll = nbdm_log_likelihood(COUNTS, _hierarchical_p1_params())
        assert not _has_nan(ll)


# ###########################################################################
# TRAINING-TIME: _P_EPS clamp in NegativeBinomialLikelihood._build_dist
# ###########################################################################


class TestNBTrainingClamp:
    """_P_EPS clamp inside NegativeBinomialLikelihood._build_dist.

    These tests call ``_build_dist`` directly (bypassing the full NumPyro
    model context) and evaluate ``dist.log_prob`` on the returned
    distribution object.
    """

    def test_p_equals_one_produces_finite_log_prob(self):
        """p=1.0 (phi_g -> 0 underflow) should be clamped to 1-eps."""
        likelihood = NegativeBinomialLikelihood()
        pv = {
            "p": jnp.ones(N_GENES, dtype=jnp.float32),
            "r": jnp.full((N_GENES,), 1.0, dtype=jnp.float32),
        }
        d = likelihood._build_dist(pv)
        lp = d.log_prob(COUNTS)
        assert not _has_nan(lp), "_P_EPS clamp should prevent NaN for p=1.0"
        assert not bool(jnp.any(jnp.isinf(lp)))

    def test_p_equals_zero_produces_finite_log_prob(self):
        """p=0.0 (phi_g -> inf overflow) should be clamped to eps."""
        likelihood = NegativeBinomialLikelihood()
        pv = {
            "p": jnp.zeros(N_GENES, dtype=jnp.float32),
            "r": jnp.full((N_GENES,), 1.0, dtype=jnp.float32),
        }
        d = likelihood._build_dist(pv)
        lp = d.log_prob(COUNTS)
        assert not _has_nan(lp), "_P_EPS clamp should prevent NaN for p=0.0"

    def test_valid_p_unaffected_by_clamp(self):
        """The clamp should not change results for well-behaved p."""
        p_val = jnp.full((N_GENES,), 0.3, dtype=jnp.float32)
        r_val = jnp.full((N_GENES,), 2.0, dtype=jnp.float32)

        # Manually compute expected log_prob (p=0.3 is far from boundaries)
        expected = dist.NegativeBinomialProbs(r_val, p_val).to_event(1)
        lp_expected = expected.log_prob(COUNTS)

        likelihood = NegativeBinomialLikelihood()
        d = likelihood._build_dist({"p": p_val, "r": r_val})
        lp_actual = d.log_prob(COUNTS)

        np.testing.assert_allclose(
            np.asarray(lp_expected), np.asarray(lp_actual), rtol=1e-5
        )

    def test_mixture_p_equals_one_produces_finite_log_prob(self):
        """Mixture NB with p=1.0 per-component should be clamped."""
        likelihood = NegativeBinomialLikelihood()
        pv = {
            "mixing_weights": jnp.full(
                (N_COMP,), 1.0 / N_COMP, dtype=jnp.float32
            ),
            "p": jnp.ones((N_COMP, N_GENES), dtype=jnp.float32),
            "r": jnp.full((N_COMP, N_GENES), 1.0, dtype=jnp.float32),
        }
        d = likelihood._build_dist(pv)
        lp = d.log_prob(COUNTS)
        assert not _has_nan(lp)


# ###########################################################################
# TRAINING-TIME: _P_EPS clamp in VCP likelihoods (phi and p_hat paths)
# ###########################################################################


class TestVCPTrainingClamp:
    """_P_EPS clamp for phi and p_hat inside VCP likelihood components.

    The clamp is applied inside ``sample()`` which requires a full NumPyro
    context.  Here we test the underlying JAX computations directly to
    verify the clamp produces finite intermediate values.
    """

    def test_phi_clamp_prevents_log_zero(self):
        """phi=0 clamped to _P_EPS should give finite logits."""
        phi = jnp.zeros(N_GENES, dtype=jnp.float32)
        capture = jnp.full((N_CELLS, 1), 0.8, dtype=jnp.float32)

        # Without clamp: log(0 * ...) = -inf
        raw_logits = -jnp.log(phi[None, :] * (1.0 + capture))
        assert bool(jnp.any(jnp.isinf(raw_logits))), (
            "Unclamped phi=0 should produce inf logits"
        )

        # With clamp: finite
        phi_clamped = jnp.maximum(phi, VCP_P_EPS)
        safe_logits = -jnp.log(phi_clamped[None, :] * (1.0 + capture))
        assert not _has_nan(safe_logits)
        assert not bool(jnp.any(jnp.isinf(safe_logits)))

    def test_phi_extreme_large_gives_finite_logits(self):
        """phi=1e10 (extreme but valid) should give finite negative logits."""
        phi = jnp.full(N_GENES, 1e10, dtype=jnp.float32)
        capture = jnp.full((N_CELLS, 1), 0.8, dtype=jnp.float32)

        phi_clamped = jnp.maximum(phi, VCP_P_EPS)
        logits = -jnp.log(phi_clamped[None, :] * (1.0 + capture))
        assert not _has_nan(logits)

    def test_p_hat_clamp_prevents_nan_at_boundaries(self):
        """p_hat at 0 or 1 clamped to (eps, 1-eps) gives finite NB log_prob."""
        r = jnp.full(N_GENES, 2.0, dtype=jnp.float32)

        for p_hat_val in [0.0, 1.0]:
            p_hat = jnp.full(N_GENES, p_hat_val, dtype=jnp.float32)
            p_hat_clamped = jnp.clip(p_hat, VCP_P_EPS, 1.0 - VCP_P_EPS)
            d = dist.NegativeBinomialProbs(r, p_hat_clamped).to_event(1)
            lp = d.log_prob(COUNTS)
            assert not _has_nan(lp), (
                f"p_hat={p_hat_val} clamped should give finite log_prob"
            )

    def test_p_hat_clamp_no_effect_for_valid_values(self):
        """Clamping does not change p_hat when it is well inside (0, 1)."""
        p_hat = jnp.full(N_GENES, 0.4, dtype=jnp.float32)
        p_hat_clamped = jnp.clip(p_hat, VCP_P_EPS, 1.0 - VCP_P_EPS)
        np.testing.assert_array_equal(np.asarray(p_hat), np.asarray(p_hat_clamped))

    def test_eps_constant_is_positive(self):
        """Sanity check that the module-level constants are set correctly."""
        assert NB_P_EPS > 0
        assert VCP_P_EPS > 0
        assert NB_P_EPS == VCP_P_EPS
