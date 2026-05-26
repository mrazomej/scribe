"""Tests for :mod:`scribe.stats.jacobian_map`.

Validates the math contract for the Jacobian-corrected MAP machinery:

* Closed-form correctness for ``ExpTransform`` (Normal and LowRankMVN).
* Grid + Newton refinement against brute-force grid search for
  Sigmoid/Softplus across both convex (small ``σ``) and bimodal
  (large ``σ``) regimes.
* Edge cases: identity transform, ``σ → 0`` limit, SlicedTransform
  per-slice recursion, LowRankMVN + Sigmoid/Softplus raises.
* Method-flag semantics: ``"transform"``, ``"jacobian"``, ``"closed_form"``,
  ``"newton"``, ``"autodiff"`` behave per docstring.
* jit/vmap compatibility.

These tests gate the math contract for the entire ``map_method`` feature.
If they pass, the SVI/Laplace integration layers can safely route through
``jacobian_corrected_map`` without re-deriving the math.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.distributions.transforms import (
    ExpTransform,
    IdentityTransform,
    SigmoidTransform,
    SoftplusTransform,
)

from scribe.stats import SIGMA_CEILING_WARN, jacobian_corrected_map


# ==============================================================================
# Closed-form correctness: ExpTransform
# ==============================================================================


class TestExpClosedForm:
    """ExpTransform: ``x* = loc - scale^2`` for ``Normal``; full
    ``Sigma * 1`` correction for ``LowRankMultivariateNormal``."""

    def test_exp_normal_lognormal_mode_known_value(self):
        """LogNormal(2.0, 0.5) has analytic mode exp(2.0 - 0.25)."""
        base = dist.Normal(2.0, 0.5)
        y_star = float(jacobian_corrected_map(ExpTransform(), base))
        expected = float(jnp.exp(2.0 - 0.25))
        np.testing.assert_allclose(y_star, expected, atol=1e-6)

    def test_exp_normal_recovers_transform_in_zero_sigma_limit(self):
        """As ``σ → 0``, corrected MAP approaches transform(loc)."""
        loc = 3.0
        y_corrected = float(
            jacobian_corrected_map(ExpTransform(), dist.Normal(loc, 1e-6))
        )
        y_transform = float(jnp.exp(loc))
        np.testing.assert_allclose(y_corrected, y_transform, rtol=1e-4)

    def test_exp_normal_vector_loc_scale(self):
        """Vector loc and scale produce vector output."""
        loc = jnp.array([1.0, 2.0, 3.0])
        scale = jnp.array([0.3, 0.5, 0.7])
        y_star = jacobian_corrected_map(ExpTransform(), dist.Normal(loc, scale))
        expected = jnp.exp(loc - scale**2)
        np.testing.assert_allclose(np.asarray(y_star), np.asarray(expected), atol=1e-6)

    def test_exp_independent_normal_unwraps_correctly(self):
        """Independent(Normal) base — recursion finds the inner Normal."""
        inner = dist.Normal(jnp.array([1.0, 2.0]), jnp.array([0.5, 0.5]))
        base = dist.Independent(inner, 1)
        y_star = jacobian_corrected_map(ExpTransform(), base)
        expected = jnp.exp(jnp.array([1.0, 2.0]) - 0.25)
        np.testing.assert_allclose(np.asarray(y_star), np.asarray(expected), atol=1e-6)

    def test_exp_lowrank_uses_full_covariance_not_diagonal(self):
        r"""Closed form for LowRankMVN + Exp uses ``Σ · 1``,
        which depends on the full covariance — diagonal-only marginals
        would be wrong for correlated cases. This test builds a base
        with explicitly correlated off-diagonals and verifies the
        einsum result matches a brute-force ``Σ @ 1`` calculation
        AND differs from a (wrong) diagonal-only result.
        """
        key = jr.PRNGKey(0)
        n, k = 4, 2
        # Pick W such that off-diagonals are nontrivial.
        W = jr.normal(key, (n, k)) * 0.5
        D = jnp.ones(n) * 0.1
        loc = jnp.array([1.0, 2.0, 3.0, 4.0])
        base = dist.LowRankMultivariateNormal(loc, W, D)

        y_ours = jacobian_corrected_map(ExpTransform(), base)

        # Brute-force: materialize Sigma, compute mu - Sigma @ 1
        Sigma = W @ W.T + jnp.diag(D)
        x_star_brute = loc - Sigma @ jnp.ones(n)
        y_brute = jnp.exp(x_star_brute)

        # The diagonal-only answer would be exp(loc - diag(Sigma))
        diag_only = jnp.exp(loc - jnp.diag(Sigma))

        np.testing.assert_allclose(np.asarray(y_ours), np.asarray(y_brute), atol=1e-5)
        # Sanity: diagonal-only differs (otherwise the test is vacuous).
        assert not np.allclose(np.asarray(y_brute), np.asarray(diag_only), atol=1e-3), (
            "Test setup degenerate: off-diagonals are zero, diagonal-only "
            "happens to be correct. Pick a larger W."
        )


# ==============================================================================
# Closed-form correctness: IdentityTransform
# ==============================================================================


class TestIdentity:
    """Identity preserves the mode: ``x* = loc``."""

    def test_identity_normal(self):
        y = float(jacobian_corrected_map(IdentityTransform(), dist.Normal(3.0, 0.5)))
        np.testing.assert_allclose(y, 3.0)

    def test_identity_lowrank(self):
        key = jr.PRNGKey(1)
        W = jr.normal(key, (3, 2)) * 0.2
        D = jnp.ones(3) * 0.1
        loc = jnp.array([1.0, 2.0, 3.0])
        base = dist.LowRankMultivariateNormal(loc, W, D)
        y = jacobian_corrected_map(IdentityTransform(), base)
        np.testing.assert_allclose(np.asarray(y), np.asarray(loc))

    def test_identity_no_dispatch_ambiguity(self):
        """Should not emit an AmbiguityWarning for Identity+Normal."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # promote warnings to errors
            y = jacobian_corrected_map(IdentityTransform(), dist.Normal(0.0, 1.0))
            assert float(y) == 0.0


# ==============================================================================
# Grid + Newton: Sigmoid against brute-force global search
# ==============================================================================


def _brute_force_sigmoid_mode(mu: float, sigma: float, n_grid: int = 50_000) -> float:
    """Brute-force minimizer of ``J_sigmoid`` on a wide dense grid.

    Uses a 50000-point grid spanning
    ``[mu - 10*max(sigma, sigma**2), mu + 10*max(sigma, sigma**2)]``,
    which is comfortably wider than the production grid+Newton's
    3*max(sigma, sigma**2) half-width. The high resolution ensures the
    brute-force result is itself accurate to ~1e-4 in y-space, so
    test tolerances of 1e-3 are meaningful.

    Returns the constrained-space mode ``sigmoid(x_argmin)``.
    """
    width = 10.0 * max(sigma, sigma**2)
    grid = np.linspace(mu - width, mu + width, n_grid)
    log_sig_x = -np.logaddexp(0.0, -grid)
    log_sig_neg_x = -np.logaddexp(0.0, grid)
    J = 0.5 * ((grid - mu) / sigma) ** 2 + log_sig_x + log_sig_neg_x
    x_best = grid[np.argmin(J)]
    return float(1.0 / (1.0 + np.exp(-x_best)))


def _brute_force_softplus_mode(mu: float, sigma: float, n_grid: int = 50_000) -> float:
    """Brute-force minimizer of ``J_softplus``. See
    :func:`_brute_force_sigmoid_mode`.
    """
    width = 10.0 * max(sigma, sigma**2)
    grid = np.linspace(mu - width, mu + width, n_grid)
    log_sig_x = -np.logaddexp(0.0, -grid)
    J = 0.5 * ((grid - mu) / sigma) ** 2 + log_sig_x
    x_best = grid[np.argmin(J)]
    return float(np.log1p(np.exp(x_best)))


class TestSigmoidGridNewton:
    """Sigmoid + Normal: grid + Newton refinement vs brute-force."""

    @pytest.mark.parametrize(
        "mu, sigma",
        [
            (0.0, 0.3),
            (0.0, 1.0),
            (1.0, 0.5),
            (-1.0, 0.5),
            # Convex regime (sigma^2 < 2)
            (2.0, 1.2),
            (-2.0, 1.2),
            # Bimodal regime (sigma^2 > 2) but symmetric — by symmetry
            # both modes are equal global maps, so we compare via J values.
            (1.0, 2.0),
            (-1.0, 2.0),
            # Bimodal with strong asymmetry — one mode clearly dominates.
            (3.0, 2.0),
            (-3.0, 2.0),
        ],
    )
    def test_matches_brute_force_or_symmetric_mode(self, mu, sigma):
        """For convex / asymmetric-bimodal cases the answer matches
        brute force exactly. For symmetric-bimodal cases we verify that
        our answer corresponds to one of the two global minima by
        comparing ``J`` values at our solution and at the brute solution
        — both should equal the minimum to high precision."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_ours = float(
                jacobian_corrected_map(
                    SigmoidTransform(), dist.Normal(mu, sigma)
                )
            )
        y_brute = _brute_force_sigmoid_mode(mu, sigma)

        # Recover x for both and compare J at both points — they should
        # agree to floating-point noise even when ``y`` values are at
        # symmetric mirror modes.
        x_ours = float(np.log(y_ours / (1.0 - y_ours)))
        x_brute = float(np.log(y_brute / (1.0 - y_brute)))

        def J(x):
            ls_x = -np.logaddexp(0.0, -x)
            ls_negx = -np.logaddexp(0.0, x)
            return 0.5 * ((x - mu) / sigma) ** 2 + ls_x + ls_negx

        J_ours = J(x_ours)
        J_brute = J(x_brute)
        # Both should be at the global minimum to within tolerance.
        np.testing.assert_allclose(J_ours, J_brute, atol=1e-2)

    def test_sigma_small_recovers_transform_loc(self):
        """In the small-sigma limit, corrected MAP equals sigmoid(loc)."""
        loc = 1.0
        y_corrected = float(
            jacobian_corrected_map(SigmoidTransform(), dist.Normal(loc, 1e-5))
        )
        y_transform = float(1.0 / (1.0 + np.exp(-loc)))
        np.testing.assert_allclose(y_corrected, y_transform, atol=1e-3)

    def test_bimodal_asymmetric_picks_dominant_mode(self):
        """For ``mu=3, sigma=3`` the positive mode strongly dominates
        and must be selected. Tests that bimodality handling isn't
        biased toward the central saddle."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = float(
                jacobian_corrected_map(SigmoidTransform(), dist.Normal(3.0, 3.0))
            )
        # Positive mode lives in y > 0.5; brute-force confirms.
        assert y > 0.5, f"Should pick positive mode, got y={y}"


class TestSoftplusGridNewton:
    """Softplus + Normal: grid + Newton refinement vs brute-force."""

    @pytest.mark.parametrize(
        "mu, sigma",
        [
            (0.0, 0.3),
            (1.0, 0.5),
            (2.0, 0.5),
            (-1.0, 0.5),
            (3.0, 1.0),
            (0.0, 2.0),
            (2.0, 1.5),
        ],
    )
    def test_matches_brute_force(self, mu, sigma):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_ours = float(
                jacobian_corrected_map(
                    SoftplusTransform(), dist.Normal(mu, sigma)
                )
            )
        y_brute = _brute_force_softplus_mode(mu, sigma)
        # Tolerance accounts for the brute grid's finite resolution
        # (50k points over a width of 10 * max(sigma, sigma^2)) translated
        # through softplus. For sigma~1.5 the y-space residual from grid
        # discretisation is ~1e-3.
        np.testing.assert_allclose(y_ours, y_brute, atol=5e-3, rtol=5e-3)

    def test_sigma_small_recovers_transform_loc(self):
        loc = 2.0
        y_corrected = float(
            jacobian_corrected_map(SoftplusTransform(), dist.Normal(loc, 1e-5))
        )
        y_transform = float(np.log1p(np.exp(loc)))
        np.testing.assert_allclose(y_corrected, y_transform, atol=1e-3)

    def test_correction_shifts_mode_below_transform_loc(self):
        """For Softplus, ``log|f'(x)| = log σ(x)`` is maximized at
        ``x → +∞``, so the J-objective ``(x − μ)²/(2σ²) + log σ(x)``
        is minimized at some ``x < μ`` — pulling the constrained mode
        below ``softplus(μ)``."""
        mu, sigma = 2.0, 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_corrected = float(
                jacobian_corrected_map(
                    SoftplusTransform(), dist.Normal(mu, sigma)
                )
            )
        y_transform = float(np.log1p(np.exp(mu)))
        assert y_corrected < y_transform, (
            f"Softplus correction should reduce mode: "
            f"got {y_corrected} vs transform(loc)={y_transform}"
        )


# ==============================================================================
# Out-of-scope: LowRankMVN + Sigmoid/Softplus
# ==============================================================================


class TestLowRankMVNNonExpRaises:
    """LowRankMVN + Sigmoid/Softplus is out of scope for v1 (needs a
    coupled multi-start solver). Should raise under ``"jacobian"`` and
    warn+fall-back under ``"auto"``."""

    def _make_lowrank_base(self):
        loc = jnp.array([0.0, 0.0])
        W = jnp.array([[0.1], [0.1]])
        D = jnp.array([0.05, 0.05])
        return dist.LowRankMultivariateNormal(loc, W, D)

    def test_sigmoid_lowrank_raises_under_jacobian(self):
        base = self._make_lowrank_base()
        with pytest.raises(NotImplementedError, match="Sigmoid"):
            jacobian_corrected_map(SigmoidTransform(), base, method="jacobian")

    def test_softplus_lowrank_raises_under_jacobian(self):
        base = self._make_lowrank_base()
        with pytest.raises(NotImplementedError, match="Softplus"):
            jacobian_corrected_map(SoftplusTransform(), base, method="jacobian")

    def test_sigmoid_lowrank_auto_warns_and_falls_back(self):
        base = self._make_lowrank_base()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = jacobian_corrected_map(SigmoidTransform(), base, method="auto")
            # The fallback returns sigmoid(loc) elementwise.
            assert any("falling back" in str(msg.message).lower() for msg in w)
        expected_fallback = jax.nn.sigmoid(base.loc)
        np.testing.assert_allclose(np.asarray(y), np.asarray(expected_fallback))


# ==============================================================================
# Sigma ceiling warning
# ==============================================================================


class TestSigmaCeiling:
    """Beyond ``SIGMA_CEILING_WARN`` the adaptive grid may miss
    asymptotic modes. Warn under ``auto``, raise under ``jacobian``."""

    def test_warning_emitted_at_high_sigma_auto(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = jacobian_corrected_map(
                SigmoidTransform(),
                dist.Normal(0.0, SIGMA_CEILING_WARN + 5.0),
                method="auto",
            )
            assert any(
                "SIGMA_CEILING_WARN" in str(msg.message) for msg in w
            ), "Expected sigma-ceiling warning"

    def test_raise_emitted_at_high_sigma_jacobian(self):
        with pytest.raises(NotImplementedError, match="SIGMA_CEILING_WARN"):
            jacobian_corrected_map(
                SigmoidTransform(),
                dist.Normal(0.0, SIGMA_CEILING_WARN + 5.0),
                method="jacobian",
            )

    def test_no_warning_below_ceiling(self):
        """Sigma below the ceiling should not trigger the warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = jacobian_corrected_map(
                SigmoidTransform(),
                dist.Normal(0.0, SIGMA_CEILING_WARN - 1.0),
                method="auto",
            )
            assert not any(
                "SIGMA_CEILING_WARN" in str(msg.message) for msg in w
            ), "Did not expect sigma-ceiling warning below threshold"


# ==============================================================================
# Method-flag semantics
# ==============================================================================


class TestMethodFlags:
    """Each ``method`` value behaves per docstring."""

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown method"):
            jacobian_corrected_map(
                ExpTransform(), dist.Normal(0.0, 1.0), method="bogus"
            )

    def test_method_transform_returns_f_of_loc(self):
        """method='transform' bypasses correction — backward-compat."""
        y = float(
            jacobian_corrected_map(
                ExpTransform(), dist.Normal(2.0, 0.5), method="transform"
            )
        )
        expected = float(jnp.exp(2.0))
        np.testing.assert_allclose(y, expected)

    def test_method_jacobian_supported_pair(self):
        """method='jacobian' on supported pair returns the corrected MAP."""
        y = float(
            jacobian_corrected_map(
                ExpTransform(), dist.Normal(2.0, 0.5), method="jacobian"
            )
        )
        expected = float(jnp.exp(2.0 - 0.25))
        np.testing.assert_allclose(y, expected, atol=1e-6)

    def test_method_closed_form_accepts_exp_and_identity(self):
        """closed_form: only Exp and Identity allowed."""
        y_exp = float(
            jacobian_corrected_map(
                ExpTransform(), dist.Normal(1.0, 0.5), method="closed_form"
            )
        )
        np.testing.assert_allclose(y_exp, float(jnp.exp(0.75)), atol=1e-6)

        y_id = float(
            jacobian_corrected_map(
                IdentityTransform(), dist.Normal(2.0, 0.5), method="closed_form"
            )
        )
        np.testing.assert_allclose(y_id, 2.0)

    def test_method_closed_form_rejects_sigmoid(self):
        with pytest.raises(NotImplementedError, match="closed_form"):
            jacobian_corrected_map(
                SigmoidTransform(), dist.Normal(0.0, 0.5), method="closed_form"
            )

    def test_method_newton_accepts_hand_derived(self):
        """newton: closed-form OR grid+Newton allowed, autodiff rejected."""
        # Exp (closed-form): allowed
        _ = jacobian_corrected_map(
            ExpTransform(), dist.Normal(1.0, 0.5), method="newton"
        )
        # Sigmoid (grid+Newton): allowed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = jacobian_corrected_map(
                SigmoidTransform(), dist.Normal(0.0, 0.5), method="newton"
            )

    def test_method_autodiff_actually_routes_through_autodiff(self, monkeypatch):
        """method='autodiff' on Sigmoid+Normal must NOT short-circuit
        to the hand-derived handler via the dispatch table.

        Hardened via monkeypatch: we replace the autodiff kernel with a
        sentinel that returns a known marker value. If the wrapper
        bypasses dispatch under method='autodiff' (as it should), our
        sentinel runs and we see the marker. If a future refactor
        re-routes through dispatch (incorrectly), the hand-derived
        Sigmoid handler would run instead and we'd see a real sigmoid
        mode value, failing the assertion.
        """
        import scribe.stats.jacobian_map as jm

        sentinel_called = {"flag": False}

        def _sentinel_autodiff(t, loc, scale, **kwargs):
            sentinel_called["flag"] = True
            # Return a recognizable marker shape so the calling
            # `transform(x_star)` produces a known constrained value.
            # We return `loc` unchanged here; the wrapper will apply
            # `transform(x_star)` which gives `sigmoid(loc)` for
            # SigmoidTransform — different from the true corrected MAP,
            # which lets us distinguish the autodiff path from the
            # dispatch path empirically too.
            return loc

        monkeypatch.setattr(jm, "_grid_plus_newton_autodiff", _sentinel_autodiff)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_auto = float(
                jacobian_corrected_map(
                    SigmoidTransform(),
                    dist.Normal(1.0, 0.5),
                    method="autodiff",
                )
            )
        assert sentinel_called["flag"], (
            "method='autodiff' must call _grid_plus_newton_autodiff; "
            "if dispatch is short-circuiting back to the hand-derived "
            "kernel, this test will fail."
        )
        # The sentinel returned loc unchanged, so y = sigmoid(loc):
        np.testing.assert_allclose(
            y_auto, float(jax.nn.sigmoid(1.0)), atol=1e-6
        )

    def test_method_autodiff_matches_handcoded_sigmoid_convex_regime(self):
        """The autodiff path SHOULD produce the same answer as the
        hand-derived Newton path in the convex regime (sigma**2 < 2).
        This is the test that previously was a no-op because dispatch
        always selected the hand-derived handler; now that the wrapper
        bypasses dispatch under method='autodiff', the comparison is
        meaningful and validates that the autodiff fallback is correct."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for mu_val, sigma_val in [(0.0, 0.3), (1.0, 0.5), (-1.0, 0.8)]:
                y_hand = float(
                    jacobian_corrected_map(
                        SigmoidTransform(),
                        dist.Normal(mu_val, sigma_val),
                        method="newton",
                    )
                )
                y_auto = float(
                    jacobian_corrected_map(
                        SigmoidTransform(),
                        dist.Normal(mu_val, sigma_val),
                        method="autodiff",
                    )
                )
                np.testing.assert_allclose(
                    y_hand, y_auto, atol=1e-3,
                    err_msg=f"hand vs autodiff disagree at mu={mu_val}, sigma={sigma_val}",
                )


# ==============================================================================
# jit/vmap compatibility
# ==============================================================================


class TestJitVmapCompatibility:
    """The corrected-MAP routines must compose with jit and vmap. This
    is critical for downstream use inside Newton kernels and per-cell
    posterior reductions where the call is inside a jitted hot path."""

    def test_jit_compat_exp_normal(self):
        @jax.jit
        def f(loc, scale):
            return jacobian_corrected_map(ExpTransform(), dist.Normal(loc, scale))

        y = float(f(jnp.array(2.0), jnp.array(0.5)))
        np.testing.assert_allclose(y, float(jnp.exp(1.75)), atol=1e-6)

    def test_jit_compat_sigmoid_normal(self):
        """Under jit, the wrapper's sigma-ceiling check (which uses
        Python ``float()`` on a JAX value) must be skipped because the
        input is a Tracer. The ``_is_concrete`` guard handles this
        explicitly rather than relying on the
        ``ConcretizationTypeError -> TypeError`` fall-through pattern.
        """

        @jax.jit
        def f(loc, scale):
            return jacobian_corrected_map(
                SigmoidTransform(), dist.Normal(loc, scale)
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = float(f(jnp.array(1.0), jnp.array(0.5)))
        # Compare to brute force
        y_brute = _brute_force_sigmoid_mode(1.0, 0.5)
        np.testing.assert_allclose(y, y_brute, atol=1e-3)

    def test_jit_compat_sigmoid_normal_high_sigma_no_concretization_error(self):
        """Even with sigma above SIGMA_CEILING_WARN (which would
        normally trigger the Python-level warning), jit must not raise
        a ConcretizationTypeError. The tracer guard skips the warning
        entirely under jit — the user sees only the jitted compute, no
        warning, no error. (Outside jit, the warning would still fire;
        see TestSigmaCeiling.)"""

        @jax.jit
        def f(loc, scale):
            return jacobian_corrected_map(
                SigmoidTransform(), dist.Normal(loc, scale)
            )

        # sigma=15 is above SIGMA_CEILING_WARN=10. Under jit, this must
        # not error or warn — the guard kicks in.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = float(f(jnp.array(0.0), jnp.array(15.0)))
            assert not any(
                "SIGMA_CEILING_WARN" in str(msg.message) for msg in w
            ), "No SIGMA_CEILING_WARN warning expected under jit"
        # The result is at least finite (heuristic, but no NaNs).
        assert jnp.isfinite(y)

    def test_jit_compat_vmap_exp_normal(self):
        """vmap-compat for the closed-form Exp path. Verifies the
        wrapper composes with vmap (which traces with batched tracers)."""

        def f(loc, scale):
            return jacobian_corrected_map(ExpTransform(), dist.Normal(loc, scale))

        locs = jnp.array([1.0, 2.0, 3.0])
        scales = jnp.array([0.3, 0.5, 0.7])
        y_vmap = jax.vmap(f)(locs, scales)
        # Expected: exp(loc - scale**2) elementwise.
        expected = jnp.exp(locs - scales**2)
        np.testing.assert_allclose(np.asarray(y_vmap), np.asarray(expected), atol=1e-6)


# ==============================================================================
# SlicedTransform per-slice recursion
# ==============================================================================


class TestSlicedTransform:
    """SlicedTransform applies different per-slice transforms; the
    per-slice MAP recursion must dispatch correctly to each leaf."""

    def test_sliced_normal_per_slice(self):
        # Importing here exercises the lazy registration path in
        # scribe.flows.__init__.
        from scribe.flows.distributions import SlicedTransform

        loc = jnp.array([0.0, 1.0, 2.0, 3.0])
        scale = jnp.array([0.5, 0.5, 0.5, 0.5])
        base = dist.Normal(loc, scale)

        # First element: SigmoidTransform; remaining three: ExpTransform.
        t = SlicedTransform(
            transforms=[SigmoidTransform(), ExpTransform()],
            sizes=[1, 3],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = jacobian_corrected_map(t, base)

        # Head: sigmoid mode at mu=0, sigma=0.5
        y_head_brute = _brute_force_sigmoid_mode(0.0, 0.5)
        np.testing.assert_allclose(float(y[0]), y_head_brute, atol=1e-3)

        # Tail: exp(loc - sigma^2)
        y_tail_expected = jnp.exp(loc[1:] - scale[1:] ** 2)
        np.testing.assert_allclose(
            np.asarray(y[1:]), np.asarray(y_tail_expected), atol=1e-6
        )

    def test_sliced_over_lowrank_raises(self):
        """LowRankMVN base under SlicedTransform raises (rank-k structure
        does not survive slicing cleanly)."""
        from scribe.flows.distributions import SlicedTransform

        loc = jnp.array([0.0, 1.0, 2.0])
        W = jnp.array([[0.1], [0.1], [0.1]])
        D = jnp.array([0.05, 0.05, 0.05])
        base = dist.LowRankMultivariateNormal(loc, W, D)
        t = SlicedTransform(
            transforms=[ExpTransform(), ExpTransform()],
            sizes=[1, 2],
        )
        with pytest.raises(NotImplementedError, match="LowRankMultivariateNormal"):
            jacobian_corrected_map(t, base, method="jacobian")
