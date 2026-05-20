"""Jacobian-corrected MAP estimation for transformed variational guides.

Overview
--------
When a variational guide for parameter `y` is built as
`y = f(x)` with `x ∼ pₓ` (typically Normal) and `f` a smooth
invertible transform, the *median* of `y` is `f(locₓ)` for monotone
`f`, but the *mode* (MAP) of `y` is **not** `f(locₓ)`. The correct
constrained-space MAP requires a Jacobian (change-of-variables)
correction.

For `X ∼ pₓ` on `ℝⁿ` and elementwise `f`, the density of
`Y = f(X)` is

    log pᵧ(y) = log pₓ(x) − log|J_f(x)|,    x = f⁻¹(y),

so the constrained-space MAP solves

    ∇ log pₓ(x*) − ∇ log|J_f(x*)| = 0,
    y* = f(x*).

For `X ∼ N(μ, Σ)` this becomes the coupled equation

    Σ⁻¹(x* − μ) + ∇ log|J_f(x*)| = 0.

Diagonal-marginal reasoning is exact only for *independent* bases
(`Normal`, `Independent(Normal)`). For correlated bases
(`LowRankMultivariateNormal`) the closed-form correction with
`ExpTransform` is `x* = μ − Σ · 𝟙`, where
`Σ · 𝟙` is computed without ever materializing the
full `n × n` covariance using
`W (Wᵀ 𝟙) + cov_diag`.

Supported (transform, base) pairs in v1
---------------------------------------
+-----------------------+------------------------------+-----------------+
| Base                  | Transform                    | Strategy        |
+=======================+==============================+=================+
| Normal / Independent  | ExpTransform                 | Closed form     |
+-----------------------+------------------------------+-----------------+
| Normal / Independent  | IdentityTransform            | Identity        |
+-----------------------+------------------------------+-----------------+
| Normal / Independent  | SoftplusTransform            | Grid + Newton   |
+-----------------------+------------------------------+-----------------+
| Normal / Independent  | SigmoidTransform             | Grid + Newton   |
+-----------------------+------------------------------+-----------------+
| LowRankMVN            | ExpTransform                 | Closed form     |
+-----------------------+------------------------------+-----------------+
| LowRankMVN            | IdentityTransform            | Identity        |
+-----------------------+------------------------------+-----------------+
| LowRankMVN            | Sigmoid / Softplus           | NotImplemented  |
+-----------------------+------------------------------+-----------------+
| Any of the above      | SlicedTransform              | Per-slice       |
+-----------------------+------------------------------+-----------------+
| Normal / Independent  | Other elementwise Transform  | Autodiff Newton |
+-----------------------+------------------------------+-----------------+

Heuristic caveat
----------------
For Sigmoid/Softplus the mode-finding is grid + Newton refinement on a
window of half-width ``K * max(σ, σ²)`` centered at ``μ``.
For Sigmoid the unconstrained density is bimodal when ``σ² > 2``
with modes asymptotically at ``μ ± σ²``  the adaptive grid
covers both regimes for ``σ`` up to ``SIGMA_CEILING_WARN`` (default
10), beyond which the public wrapper emits a warning (under ``"auto"``)
or raises (under ``"jacobian"``).

The shape-based ``_assert_elementwise`` check is a safeguard, not a
mathematical proof of elementwise behavior  transforms that mix
components elementwise while still returning input-shape Jacobians would
slip past. All transforms currently registered in scribe satisfy the
contract by construction.

Why dispatch on ``(transform, base)`` jointly
---------------------------------------------
The naive design ``f(loc, scale)`` cannot represent the off-diagonal
coupling that matters for correlated bases. By passing the full base
distribution and dispatching on ``(type(transform), type(base))``, the
``LowRankMVN + Exp`` handler can compute ``Σ · 𝟙``
via rank-``k`` einsum, while ``Normal + Exp`` collapses to a scalar
operation. ``multipledispatch`` is genuinely necessary here, not just
stylistic — it's the dispatch shape the math demands.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from multipledispatch import dispatch
from numpyro.distributions import (
    Distribution,
    Independent,
    LowRankMultivariateNormal,
    Normal,
)
from numpyro.distributions.transforms import (
    ExpTransform,
    IdentityTransform,
    SigmoidTransform,
    SoftplusTransform,
    Transform,
)

# SlicedTransform lives in ``scribe.stats.transforms`` (the canonical
# home for scribe's custom NumPyro Transform subclasses, mirroring
# ``scribe.stats.distributions`` for custom Distributions). Importing
# from a leaf module here means we can register its dispatch handler
# eagerly below — no lazy-registration dance required.
from scribe.stats.transforms import SlicedTransform

# ==============================================================================
# Constants
# ==============================================================================

#: Below this ``σ`` floor, the Newton kernels would divide by huge
#: ``1/σ²`` terms and produce NaNs  we clip ``σ`` here. Picked
#: small enough that the resulting correction is numerically
#: indistinguishable from ``transform(loc)`` in the limit.
_SIGMA_FLOOR = 1e-8

#: When ``σ`` exceeds this ceiling, the adaptive-width grid may not
#: cover the asymptotic mode locations for Sigmoid/Softplus and the result
#: from the heuristic should not be trusted. The public wrapper emits a
#: warning under ``method="auto"`` and raises under ``method="jacobian"``.
SIGMA_CEILING_WARN = 10.0

#: Default number of grid points used by the adaptive-width grid search.
#: 21 points across a half-width of ``K * max(σ, σ²)`` gives
#: a coarse argmin that the Newton refinement step polishes.
_DEFAULT_N_GRID = 21

#: Default multiplier for the grid half-width. Half-width is
#: ``_DEFAULT_GRID_WIDTH * max(σ, σ²)``. The choice trades
#: coverage (larger = more robust to far-asymptotic modes) against grid
#: density (larger = coarser argmin).
_DEFAULT_GRID_WIDTH = 3.0

#: Default Newton refinement iterations. Convergence on Sigmoid/Softplus
#: in the supported ``σ`` regime is achieved in 5-10 iterations  30
#: is a comfortable margin.
_DEFAULT_NEWTON_ITERS = 30


# ==============================================================================
# Base-location helper (used by the ``method="transform"`` path)
# ==============================================================================


@dispatch(Normal)
def _base_loc(d):  # noqa: F811
    """Per-element location of an independent ``Normal`` base."""
    return d.loc


@dispatch(Independent)
def _base_loc(d):  # noqa: F811
    """Unwrap an ``Independent`` and recurse on the wrapped base."""
    return _base_loc(d.base_dist)


@dispatch(LowRankMultivariateNormal)
def _base_loc(d):  # noqa: F811
    """Per-element location of a ``LowRankMultivariateNormal`` base."""
    return d.loc


@dispatch(Distribution)
def _base_loc(d):  # noqa: F811
    """Generic fallback for distributions that expose ``loc`` or ``mean``.

    Used by the ``method="transform"`` path when the base type is not one
    of the explicitly-dispatched cases. Raises ``TypeError`` if neither
    ``loc`` nor ``mean`` is available — this signals an unsupported guide
    shape rather than a quietly-wrong default.
    """
    if hasattr(d, "loc"):
        return d.loc
    if hasattr(d, "mean"):
        return d.mean
    raise TypeError(
        f"No _base_loc handler for {type(d).__name__}  "
        "distribution exposes neither .loc nor .mean."
    )


# ==============================================================================
# Internal: ``_map_unconstrained`` — returns the unconstrained mode x*
# ==============================================================================
#
# Every handler returns the unconstrained mode ``x*``. The public
# ``jacobian_corrected_map`` applies ``transform(x*)`` once at the
# top-level. This keeps the SlicedTransform recursion clean — pieces are
# concatenated in unconstrained space and the constraining transforms are
# only applied at the leaves by the public wrapper.


# --- Closed-form: ExpTransform ------------------------------------------------


@dispatch(ExpTransform, Normal)
def _map_unconstrained(t, base):  # noqa: F811
    """LogNormal mode in unconstrained space.

    For ``X ~ N(μ, σ²)`` and ``Y = eˣ``, the density of ``Y`` is
    LogNormal with mode ``e^(μ−σ²)``. In unconstrained space:

    .. math::

        x* = μ - σ².

    Derivation: ``J'(x) = (x - μ)/σ² + 1`` (since
    ``log|f'(x)| = x`` for Exp)  setting ``J'(x*) = 0`` gives the
    closed form. No iteration needed.
    """
    return base.loc - base.scale**2


@dispatch(ExpTransform, Independent)
def _map_unconstrained(t, base):  # noqa: F811
    """ExpTransform over Independent base — unwrap and recurse."""
    return _map_unconstrained(t, base.base_dist)


@dispatch(ExpTransform, LowRankMultivariateNormal)
def _map_unconstrained(t, base):  # noqa: F811
    """LogNormal-with-correlated-base mode in unconstrained space.

    For ``X ~ N(μ, Σ)`` and ``Y = eˣ`` (elementwise), the
    coupled stationarity condition is

    .. math::

        Σ⁻¹(x* - μ) + 𝟙 = 0
         ⟹ 
        x* = μ - Σ · 𝟙.

    For ``Σ = W Wᵀ + diag(cov_diag)``, the
    quantity ``Σ · 𝟙`` is computed cheaply via

    .. math::

        Σ · 𝟙 = W (Wᵀ 𝟙) +
            cov_diag

    in ``O(n k)`` flops where ``k`` is the rank — never materializing the
    ``n × n`` covariance.
    """
    W = base.cov_factor  # (..., n, k)
    # Wᵀ @ 1_n: column sums of W, giving a k-vector per leading batch.
    w_sum = W.sum(axis=-2)  # (..., k)
    # W @ (Wᵀ @ 1_n): rank-k contribution to ``Sigma @ 1_n``.
    Sigma_dot_1 = (
        jnp.einsum("...nk,...k->...n", W, w_sum) + base.cov_diag
    )  # (..., n)
    return base.loc - Sigma_dot_1


# --- IdentityTransform --------------------------------------------------------


# Identity handlers are dispatched per concrete base type to disambiguate
# against the generic ``(Transform, Normal)`` autodiff fallback. Otherwise
# multipledispatch raises an AmbiguityWarning because
# ``(IdentityTransform, Normal)`` matches both ``(IdentityTransform, Distribution)``
# and ``(Transform, Normal)``.


@dispatch(IdentityTransform, Normal)
def _map_unconstrained(t, base):  # noqa: F811
    """Identity on independent ``Normal``: mode = loc."""
    return base.loc


@dispatch(IdentityTransform, Independent)
def _map_unconstrained(t, base):  # noqa: F811
    """Identity on Independent base — unwrap and recurse."""
    return _map_unconstrained(t, base.base_dist)


@dispatch(IdentityTransform, LowRankMultivariateNormal)
def _map_unconstrained(t, base):  # noqa: F811
    """Identity on low-rank multivariate Normal: mode = loc."""
    return base.loc


@dispatch(IdentityTransform, Distribution)
def _map_unconstrained(t, base):  # noqa: F811
    """Identity on generic base — fall back to ``_base_loc`` helper."""
    return _base_loc(base)


# --- Grid + Newton: SoftplusTransform on independent Normal -------------------


@dispatch(SoftplusTransform, Normal)
def _map_unconstrained(t, base):  # noqa: F811
    """Softplus-constrained mode via adaptive grid + Newton refinement.

    For ``X ~ N(μ, σ²)`` and ``Y = log(1 + eˣ)``, the objective to
    minimize (negative log-density of ``Y``, ignoring constants) is

        J(x) = (x − μ)² / (2σ²)  +  log σ(x),

    where ``σ(x)`` is the sigmoid function (the Jacobian of softplus).
    The sign on ``log σ(x)`` is ``+`` (not ``−``): the negative-log-
    target is ``J = −log p_X + log|f'|``, mirroring the
    change-of-variables identity. No closed form exists  we use
    adaptive grid + Newton refinement (see :func:`_grid_plus_newton`).
    """
    return _grid_plus_newton(
        J_fn=_J_softplus,
        step_fn=_newton_step_softplus,
        mu=base.loc,
        sigma=base.scale,
    )


@dispatch(SoftplusTransform, Independent)
def _map_unconstrained(t, base):  # noqa: F811
    """SoftplusTransform over Independent base — unwrap and recurse."""
    return _map_unconstrained(t, base.base_dist)


# --- Grid + Newton: SigmoidTransform on independent Normal --------------------


@dispatch(SigmoidTransform, Normal)
def _map_unconstrained(t, base):  # noqa: F811
    """Sigmoid-constrained (logistic-normal) mode via grid + Newton.

    For ``X ~ N(μ, σ²)`` and ``Y = σ(X)``, the objective is

        J(x) = (x − μ)² / (2σ²)  +  log σ(x)  +  log σ(−x).

    The two log-sigmoid terms together implement ``+ log|f'(x)|`` =
    ``log(σ(x)(1 − σ(x)))`` in a numerically stable form. The sign on
    each is ``+`` (not ``−``) per the negative-log-target convention
    ``J = −log p_X + log|f'|``.

    The constrained density is **bimodal** when ``σ² > 2``, with modes
    asymptotically at ``x ≈ μ ± σ²`` for large ``σ``. A single-start
    Newton at ``μ`` would land on a saddle in that regime  the adaptive
    grid (half-width
    ``K · max(σ, σ²)``) covers both modes and the Newton refinement
    polishes the argmin.
    """
    return _grid_plus_newton(
        J_fn=_J_sigmoid,
        step_fn=_newton_step_sigmoid,
        mu=base.loc,
        sigma=base.scale,
    )


@dispatch(SigmoidTransform, Independent)
def _map_unconstrained(t, base):  # noqa: F811
    """SigmoidTransform over Independent base — unwrap and recurse."""
    return _map_unconstrained(t, base.base_dist)


# --- Out of scope: LowRankMVN + Sigmoid/Softplus ------------------------------


@dispatch(SoftplusTransform, LowRankMultivariateNormal)
def _map_unconstrained(t, base):  # noqa: F811
    """Coupled multi-start optimization not implemented for v1.

    The stationarity condition is
    ``Σ⁻¹(x* - μ) + (1 - σ(x*)) = 0``, which is a
    coupled n-dimensional system with a single mode but requires a
    proper non-convex optimizer (the diagonal-Newton/grid scheme would
    ignore off-diagonal coupling). Deferred until a coupled solver is
    added  falls back to ``transform(loc)`` under ``method="auto"``.
    """
    raise NotImplementedError(
        "Softplus + LowRankMultivariateNormal requires a coupled "
        "multi-start optimizer (out of scope for v1). Use "
        "map_method='transform' to opt out of the correction, or "
        "restructure the guide with an Independent(Normal) base."
    )


@dispatch(SigmoidTransform, LowRankMultivariateNormal)
def _map_unconstrained(t, base):  # noqa: F811
    """Coupled bimodal optimization not implemented for v1.

    The Sigmoid + correlated-base density can be multimodal (similar to
    the independent Sigmoid case but with cross-component coupling).
    Deferred  falls back to ``transform(loc)`` under ``method="auto"``.
    """
    raise NotImplementedError(
        "Sigmoid + LowRankMultivariateNormal requires a coupled "
        "multi-start optimizer (out of scope for v1  logistic-normal "
        "densities can be bimodal). Use map_method='transform' to opt "
        "out, or restructure the guide with an Independent(Normal) base."
    )


# --- SlicedTransform recursion ------------------------------------------------


@dispatch(SlicedTransform, Distribution)
def _map_unconstrained(t, base):  # noqa: F811
    """Per-slice MAP recursion for ``SlicedTransform``.

    The Jacobian of a SlicedTransform is block-diagonal across slices
    (each slice's transform contributes independently), so the per-slice
    MAP corrections decouple. We slice ``loc`` and ``scale`` (for
    ``Normal`` / ``Independent(Normal)`` bases) via :func:`_slice_base`,
    recursively compute the unconstrained mode of each piece, then
    concatenate the pieces along the trailing axis.

    Raises ``NotImplementedError`` if the base is
    ``LowRankMultivariateNormal``: slicing breaks the rank-k
    factorization that the closed-form Exp handler relies on, so the
    correction would silently fall back to a less-accurate path.
    """
    if isinstance(base, LowRankMultivariateNormal):
        raise NotImplementedError(
            "SlicedTransform over LowRankMultivariateNormal is not "
            "supported in v1  slicing a low-rank covariance breaks "
            "the rank-k factorization. Use map_method='transform' "
            "instead."
        )
    pieces = []
    for sub_t, start, size in zip(t._transforms, t._offsets, t._sizes):
        end = start + size
        sub_base = _slice_base(base, start, end)
        pieces.append(_map_unconstrained(sub_t, sub_base))
    return jnp.concatenate(pieces, axis=-1)


# --- Generic autodiff fallback ------------------------------------------------


@dispatch(Transform, Normal)
def _map_unconstrained(t, base):  # noqa: F811
    """Generic elementwise Transform on a Normal base — autodiff Newton.

    Used when no specialized handler is registered for the transform
    type. Validates that the transform is elementwise (by output-shape
    probe) before invoking the autodiff grid + Newton routine.

    Raises ``NotImplementedError`` if the transform's
    ``log_abs_det_jacobian`` does not return an input-shape array (e.g.,
    sums the per-element terms into a scalar — convention used by
    multivariate transforms).
    """
    _assert_elementwise(t, base.loc)
    return _grid_plus_newton_autodiff(t, base.loc, base.scale)


@dispatch(Transform, Independent)
def _map_unconstrained(t, base):  # noqa: F811
    """Generic elementwise Transform on Independent base — unwrap."""
    return _map_unconstrained(t, base.base_dist)


# ==============================================================================
# Helpers used by the dispatched handlers
# ==============================================================================


def _slice_base(base, start, end):
    """Return a slice of a base distribution along the trailing axis.

    Only ``Normal`` and ``Independent(Normal)`` are supported.
    ``LowRankMultivariateNormal`` raises because slicing destroys the
    rank-``k`` factorisation that the closed-form Exp handler relies on.
    """
    if isinstance(base, Normal):
        return Normal(
            loc=base.loc[..., start:end],
            scale=base.scale[..., start:end],
        )
    if isinstance(base, Independent):
        # Unwrap and re-wrap  preserve reinterpreted_batch_ndims=1 for
        # the typical "vector" convention.
        inner = _slice_base(base.base_dist, start, end)
        return Independent(inner, base.reinterpreted_batch_ndims)
    raise NotImplementedError(
        f"Cannot slice base distribution of type {type(base).__name__}."
    )


def _assert_elementwise(t, probe_loc):
    """Shape-based safeguard for the autodiff fallback path.

    Calls ``t.log_abs_det_jacobian`` on a small probe vector and verifies
    the output shape matches the input. This catches transforms that sum
    per-element contributions into a scalar (the convention used by
    multivariate transforms like ``StickBreakingTransform``).

    This is a safeguard, NOT a mathematical proof of elementwise
    behavior — a transform that mixes components elementwise but still
    returns an input-shape Jacobian would slip past. Acceptable for v1
    because every transform we route through autodiff in scribe
    (``ExpTransform``, ``SoftplusTransform``, ``SigmoidTransform``,
    ``IdentityTransform`` and their composites) satisfies the contract
    by construction.
    """
    # Build a probe with a small known shape, ensuring trailing axis
    # exists. Use a finite value (zero) to avoid edge cases at +/-inf.
    probe = jnp.zeros_like(probe_loc)
    try:
        y = t(probe)
        ladj = t.log_abs_det_jacobian(probe, y)
    except Exception as exc:  # noqa: BLE001
        raise NotImplementedError(
            f"Transform {type(t).__name__} could not be probed for "
            f"elementwise compatibility: {exc!r}"
        ) from exc
    if jnp.asarray(ladj).shape != probe.shape:
        raise NotImplementedError(
            f"Transform {type(t).__name__} appears non-elementwise: "
            f"log_abs_det_jacobian returned shape {jnp.asarray(ladj).shape} "
            f"for input of shape {probe.shape}. Only elementwise "
            "transforms are supported by the autodiff fallback."
        )


# ==============================================================================
# Newton step kernels (hand-derived gradients and Hessians)
# ==============================================================================
#
# Each (J_*, _newton_step_*) pair is dedicated to one transform. Hessians
# are floored from below at ``0.25 / σ²`` to prevent Newton blowup
# in non-convex regimes (the bimodality fix lives in the grid+Newton
# scaffold  this floor is just inner-loop safety).


def _safe_sigma(sigma):
    """Clip ``sigma`` to a small positive floor to prevent ``1/σ²``
    blowup in degenerate posteriors. The floor is small enough that the
    correction is numerically indistinguishable from ``transform(loc)``
    in the ``σ → 0`` limit (where median equals mode anyway).
    """
    return jnp.maximum(sigma, _SIGMA_FLOOR)


# --- Softplus -----------------------------------------------------------------


def _J_softplus(x, mu, sigma):
    r"""Negative log-density of ``Y = softplus(X)`` (ignoring constants).

    The change-of-variables identity ``log p_Y(y) = log p_X(x) - log|f'(x)|``
    gives the negative log:

    .. math::

        J(x) = -log p_Y(y) = -log p_X(x) + log|f'(x)|
             = (x − μ)² / (2σ²) + log σ(x).

    Note: ``log|f'(x)|`` for Softplus equals ``log σ(x)``. The
    sign is ``+``, not ``-``: a small Jacobian (``f'(x) → 0`` as
    ``x → -∞``) means ``Y`` covers a small chunk of the
    constrained space per unit ``X``, so ``Y``'s density there is
    LARGER. This is why the Softplus mode shifts *negative* from
    ``μ`` (toward small-Jacobian region).

    Uses ``jax.nn.log_sigmoid`` for numerical stability.
    """
    sigma_safe = _safe_sigma(sigma)
    return 0.5 * ((x - mu) / sigma_safe) ** 2 + jax.nn.log_sigmoid(x)


def _newton_step_softplus(x, mu, sigma):
    r"""One Newton step on the negative-log-target ``J`` for Softplus.

    Gradient: ``J'(x) = (x - μ)/σ² + (1 - σ(x))``.
    Hessian:  ``J''(x) = 1/σ² - σ(x)(1 - σ(x))``.

    Since ``σ(1 - σ) ≤ 1/4``, ``J''(x) > 0`` whenever
    ``σ² < 4``. Outside that regime the Hessian can dip slightly
    negative  the floor at ``0.25 / σ²`` keeps the Newton step
    bounded but does not by itself escape a saddle — that's the
    grid+Newton scaffold's job.
    """
    sigma_safe = _safe_sigma(sigma)
    s = jax.nn.sigmoid(x)
    grad = (x - mu) / sigma_safe**2 + (1.0 - s)
    hess = 1.0 / sigma_safe**2 - s * (1.0 - s)
    hess_safe = jnp.maximum(hess, 0.25 / sigma_safe**2)
    return x - grad / hess_safe


# --- Sigmoid ------------------------------------------------------------------


def _J_sigmoid(x, mu, sigma):
    r"""Negative log-density of ``Y = σ(X)`` (ignoring constants).

    .. math::

        J(x) = (x − μ)² / (2σ²)
             + log σ(x) + log σ(-x).

    The two log-sigmoid terms together implement
    ``log|f'(x)| = log(σ(x) (1 - σ(x)))`` in a way that
    is numerically stable across the full real line — ``log_sigmoid``
    avoids overflow in ``exp`` and underflow in ``log``.

    Sign: ``+ log|f'|`` (not ``-``), per the change-of-variables identity
    ``log p_Y(y) = log p_X(x) - log|f'(x)|``  the negative-log-target
    flips that sign.

    Bimodality: ``J''(x) = 1/σ² - 2 σ(x)(1 - σ(x))``
    can be negative (``J`` non-convex) when ``σ² > 2``  the
    central stationary point at ``μ`` then becomes a local MAX of
    ``J`` (= local MIN of density), with two true modes at
    ``x ≈ μ ± σ²``. The grid+Newton scaffold finds
    the global minimum by initializing from multiple candidates.
    """
    sigma_safe = _safe_sigma(sigma)
    return (
        0.5 * ((x - mu) / sigma_safe) ** 2
        + jax.nn.log_sigmoid(x)
        + jax.nn.log_sigmoid(-x)
    )


def _newton_step_sigmoid(x, mu, sigma):
    r"""One Newton step on the negative-log-target ``J`` for Sigmoid.

    Gradient: ``J'(x) = (x - μ)/σ² + (1 - 2 σ(x))``.
    Hessian:  ``J''(x) = 1/σ² - 2 σ(x)(1 - σ(x))``.

    In the convex regime (``σ² < 2``), ``J''(x) > 0`` everywhere
    and Newton from ``μ`` converges to the unique mode. In the
    bimodal regime, ``J''`` can be negative near ``μ``  the Hessian
    floor at ``0.25 / σ²`` keeps the step bounded. Global mode
    selection is delegated to the grid+Newton scaffold (which
    initializes Newton from multiple candidates and selects the
    per-element argmin of ``J``).
    """
    sigma_safe = _safe_sigma(sigma)
    s = jax.nn.sigmoid(x)
    grad = (x - mu) / sigma_safe**2 + (1.0 - 2.0 * s)
    hess = 1.0 / sigma_safe**2 - 2.0 * s * (1.0 - s)
    hess_safe = jnp.maximum(hess, 0.25 / sigma_safe**2)
    return x - grad / hess_safe


# ==============================================================================
# Adaptive grid + Newton refinement (handles bimodal / shifted-mode regimes)
# ==============================================================================


def _grid_plus_newton(
    *,
    J_fn: Callable,
    step_fn: Callable,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    n_grid: int = _DEFAULT_N_GRID,
    grid_width: float = _DEFAULT_GRID_WIDTH,
    newton_iters: int = _DEFAULT_NEWTON_ITERS,
) -> jnp.ndarray:
    r"""Adaptive coarse grid + local Newton refinement.

    Algorithm
    ---------
    1. Compute an adaptive grid half-width per element:
       ``W(σ) = K · max(σ, σ²)``. This covers
       both the Gaussian-dominated regime (``σ < 1``, modes near
       ``μ ± K σ``) and the Jacobian-dominated regime
       (``σ > 1``, modes near ``μ ± K σ²`` for
       Sigmoid).
    2. Sample ``n_grid`` candidate points uniformly across
       ``[μ - W, μ + W]`` per element.
    3. Evaluate ``J`` at each candidate. Per-element argmin selects the
       best initialization.
    4. Run ``newton_iters`` Newton steps from that initialization.

    All shapes are fixed, so the routine is jit- and vmap-compatible.

    Heuristic caveat
    ----------------
    The half-width ``K · max(σ, σ²)`` is wide enough
    for ``σ`` up to ``SIGMA_CEILING_WARN``  beyond that, asymptotic
    modes can escape the grid. The public wrapper checks ``σ``
    against that ceiling and warns / raises as documented.

    Parameters
    ----------
    J_fn : Callable[[x, mu, sigma], jnp.ndarray]
        Per-element negative-log-target objective.
    step_fn : Callable[[x, mu, sigma], jnp.ndarray]
        Per-element Newton step kernel.
    mu, sigma : jnp.ndarray
        Per-element posterior location and stddev.
    n_grid : int
        Number of grid candidates per element.
    grid_width : float
        Half-width multiplier (``K`` in the math).
    newton_iters : int
        Number of Newton refinement iterations.

    Returns
    -------
    jnp.ndarray
        Per-element unconstrained mode ``x*``.
    """
    # Normalize inputs to jax arrays — accepts Python scalars too, so a
    # caller passing ``Normal(0.0, 0.5)`` (literal floats) doesn't crash.
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    sigma_safe = _safe_sigma(sigma)
    # Half-width adapts to whichever regime dominates: σ for tight
    # posteriors, σ² for wide ones where the Jacobian shifts the
    # mode far from μ.
    half_width = grid_width * jnp.maximum(sigma_safe, sigma_safe**2)
    # Offsets in [-1, 1] of shape (n_grid,)
    offsets = jnp.linspace(-1.0, 1.0, n_grid)
    # Reshape offsets to broadcast over mu's trailing axes  resulting
    # x_grid has shape (n_grid, *mu.shape). mu.ndim==0 means scalar
    # input — the reshape still works (offsets is 1-D), and broadcasting
    # via mu[None, ...] correctly produces a 1-D x_grid.
    offsets_b = offsets.reshape((-1,) + (1,) * mu.ndim)
    x_grid = mu[None, ...] + half_width[None, ...] * offsets_b

    # Vectorised J evaluation across the n_grid axis.
    J_grid = jax.vmap(lambda x: J_fn(x, mu, sigma_safe))(x_grid)  # (n_grid, *)
    # Per-element argmin over the n_grid axis -> shape (*mu.shape,).
    idx = jnp.argmin(J_grid, axis=0)
    # Gather the best candidate per element.
    x_init = jnp.take_along_axis(x_grid, idx[None, ...], axis=0).squeeze(0)

    # Newton refinement with fixed iteration count (jit-clean).
    def body(_, x):
        return step_fn(x, mu, sigma_safe)

    x_final = jax.lax.fori_loop(0, newton_iters, body, x_init)
    return x_final


def _grid_plus_newton_autodiff(
    t: Transform,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    n_grid: int = _DEFAULT_N_GRID,
    grid_width: float = _DEFAULT_GRID_WIDTH,
    newton_iters: int = _DEFAULT_NEWTON_ITERS,
) -> jnp.ndarray:
    """Same algorithm as :func:`_grid_plus_newton`, but ``J``, gradient,
    and Hessian are derived via ``jax.grad`` / ``jax.hessian`` on the
    transform's ``log_abs_det_jacobian``.

    Used as the fallback path for unknown elementwise transforms.
    Slower than the hand-derived kernels (more autodiff overhead) but
    handles any transform that satisfies the shape contract verified by
    :func:`_assert_elementwise`.
    """
    sigma_safe = _safe_sigma(sigma)

    # Define scalar J on a single element (mu_i, sigma_i, x_i).
    # J(x) = -log p_Y(y) = -log p_X(x) + log|f'(x)|. The change-of-variables
    # sign is +log|f'(x)|, NOT -log|f'(x)|  the latter is the formula for
    # log p_Y, while J is the negative-log-target we minimize.
    def J_scalar(x_i, mu_i, sigma_i):
        # f(x_i) at one element. For elementwise transforms the global
        # output is the per-element transform  we don't need to vmap the
        # transform itself, just apply it.
        y_i = t(x_i)
        ladj = t.log_abs_det_jacobian(x_i, y_i)
        return 0.5 * ((x_i - mu_i) / sigma_i) ** 2 + ladj

    grad_J = jax.grad(J_scalar)
    hess_J = jax.hessian(J_scalar)

    def J_fn(x, mu_arr, sigma_arr):
        # Vmap scalar J over the flattened arrays.
        return jax.vmap(J_scalar)(
            x.ravel(), mu_arr.ravel(), sigma_arr.ravel()
        ).reshape(x.shape)

    def step_fn(x, mu_arr, sigma_arr):
        # Vmap (grad, hessian) pair  apply Hessian floor for safety.
        flat_x = x.ravel()
        flat_mu = mu_arr.ravel()
        flat_sigma = sigma_arr.ravel()
        g = jax.vmap(grad_J)(flat_x, flat_mu, flat_sigma)
        h = jax.vmap(hess_J)(flat_x, flat_mu, flat_sigma)
        h_safe = jnp.maximum(h, 0.25 / flat_sigma**2)
        flat_x_new = flat_x - g / h_safe
        return flat_x_new.reshape(x.shape)

    return _grid_plus_newton(
        J_fn=J_fn,
        step_fn=step_fn,
        mu=mu,
        sigma=sigma_safe,
        n_grid=n_grid,
        grid_width=grid_width,
        newton_iters=newton_iters,
    )


# ==============================================================================
# Public API
# ==============================================================================


def jacobian_corrected_map(
    transform: Transform,
    base: Distribution,
    *,
    method: str = "auto",
    n_grid: int = _DEFAULT_N_GRID,
    grid_width: float = _DEFAULT_GRID_WIDTH,
    newton_iters: int = _DEFAULT_NEWTON_ITERS,
) -> jnp.ndarray:
    """Return ``f(x*)`` where ``x*`` solves the constrained-space MAP equation.

    For a parameter modelled as ``Y = transform(X)`` with ``X ~ base``,
    this function returns the **mode of Y in constrained space**, which
    accounts for the Jacobian of the transform. This differs from
    ``transform(base.loc)`` (the *median* of Y for monotone transforms)
    by a Jacobian correction.

    Parameters
    ----------
    transform : numpyro.distributions.transforms.Transform
        The transform mapping unconstrained ``X`` to constrained ``Y``.
        Supported: ``ExpTransform``, ``SoftplusTransform``,
        ``SigmoidTransform``, ``IdentityTransform``, ``SlicedTransform``,
        and any other elementwise ``Transform`` (via autodiff fallback).
    base : numpyro.distributions.Distribution
        The unconstrained base distribution. Supported: ``Normal``,
        ``Independent(Normal, ...)``, ``LowRankMultivariateNormal``.
        For correlated bases, ``ExpTransform`` and ``IdentityTransform``
        are supported with closed-form corrections 
        ``SoftplusTransform``/``SigmoidTransform`` raise.
    method : str, default ``"auto"``
        Strategy for the correction. Values:

        - ``"auto"``: closed-form > grid+Newton > autodiff fallback,
          dispatched on ``(type(transform), type(base))``. If no handler
          applies, falls back to ``transform(loc)`` with a warning.
        - ``"transform"``: returns ``transform(loc)`` without correction.
          The backward-compat path  equals the median of ``Y`` for
          monotone ``transform``. Never raises.
        - ``"jacobian"``: same dispatch as ``"auto"`` but raises rather
          than warning on unsupported pairs. Useful for strict
          reproducibility.
        - ``"closed_form"``: only Exp/Identity  raises for others.
        - ``"newton"``: allows closed-form or grid+Newton  raises only
          on autodiff-fallback territory. Useful for certifying that no
          autodiff was used in the computation.
        - ``"autodiff"``: forces the generic autodiff path  raises if
          the transform is not shape-elementwise.
    n_grid : int, default 21
        Number of grid candidates for the grid+Newton refinement.
    grid_width : float, default 3.0
        Adaptive grid half-width multiplier: half-width is
        ``grid_width * max(sigma, sigma**2)``.
    newton_iters : int, default 30
        Number of Newton refinement steps.

    Returns
    -------
    jnp.ndarray
        Constrained-space MAP value ``y* = transform(x*)``, same shape
        as ``base.loc``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the recognized values.
    NotImplementedError
        If ``method != "auto"`` and the requested ``(transform, base)``
        pair is not supported.

    Warnings
    --------
    Emits ``UserWarning`` under ``method="auto"`` when:
        - The requested ``(transform, base)`` pair is not supported and
          the function falls back to ``transform(loc)``.
        - ``max(sigma)`` exceeds ``SIGMA_CEILING_WARN`` (default 10),
          beyond which the adaptive grid may not cover the asymptotic
          modes for Sigmoid/Softplus.

    Examples
    --------
    LogNormal mode (closed form):

    >>> import numpyro.distributions as dist
    >>> from numpyro.distributions.transforms import ExpTransform
    >>> base = dist.Normal(2.0, 0.5)
    >>> jacobian_corrected_map(ExpTransform(), base)  # exp(2.0 - 0.25)
    Array(5.7546..., dtype=...)

    Logistic-normal mode (grid + Newton):

    >>> from numpyro.distributions.transforms import SigmoidTransform
    >>> base = dist.Normal(0.0, 0.5)
    >>> jacobian_corrected_map(SigmoidTransform(), base)  # near 0.5
    Array(0.5, dtype=...)
    """
    # --- Method validation --------------------------------------------
    if method not in (
        "auto",
        "transform",
        "jacobian",
        "closed_form",
        "newton",
        "autodiff",
    ):
        raise ValueError(
            f"Unknown method={method!r}. Expected one of "
            "{'auto', 'transform', 'jacobian', 'closed_form', "
            "'newton', 'autodiff'}."
        )

    # --- "transform" path: no correction, backward-compat -----------
    if method == "transform":
        return transform(_base_loc(base))

    # --- Pre-dispatch method gates ----------------------------------
    if method == "closed_form" and not isinstance(
        transform, (ExpTransform, IdentityTransform)
    ):
        raise NotImplementedError(
            f"method='closed_form' requires Exp or Identity transform  "
            f"got {type(transform).__name__}."
        )
    if method == "newton":
        # Allowed: closed-form (Exp, Identity), grid+Newton (Softplus,
        # Sigmoid), or SlicedTransform (each slice independently
        # newton-compatible  the recursion enforces the leaf-level
        # contract).
        is_newton_eligible = isinstance(
            transform,
            (
                SoftplusTransform,
                SigmoidTransform,
                ExpTransform,
                IdentityTransform,
                SlicedTransform,
            ),
        )
        if not is_newton_eligible:
            raise NotImplementedError(
                f"method='newton' refuses autodiff fallback  "
                f"transform {type(transform).__name__} is not in the "
                "hand-derived set (Exp, Identity, Softplus, Sigmoid, "
                "Sliced)."
            )
    if method == "autodiff":
        # Force autodiff path: probe elementwise compatibility now to
        # give a clear error before dispatch tries to find a specialized
        # handler.
        try:
            _assert_elementwise(transform, _base_loc(base))
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise NotImplementedError(
                f"method='autodiff' could not probe {type(transform).__name__} "
                f"for elementwise compatibility: {exc!r}"
            ) from exc
        # CRITICAL: bypass the dispatch table entirely and call the
        # autodiff kernel directly. If we fell through to
        # ``_map_unconstrained(transform, base)``, dispatch on
        # (SigmoidTransform, Normal) would match the hand-derived
        # handler and we'd never actually exercise autodiff. Bypassing
        # gives the user a way to certify autodiff was used (e.g., for
        # comparison against the hand-derived implementation in tests).
        base_loc = _base_loc(base)
        # Extract scale from supported independent-Normal bases. The
        # autodiff fallback is only sound for elementwise transforms
        # AND independent bases — we already rejected non-elementwise
        # transforms via _assert_elementwise.
        if isinstance(base, Normal):
            base_scale = base.scale
        elif isinstance(base, Independent):
            # Independent(Normal) — recurse until we find the leaf.
            inner = base.base_dist
            while isinstance(inner, Independent):
                inner = inner.base_dist
            if not isinstance(inner, Normal):
                raise NotImplementedError(
                    f"method='autodiff' requires a Normal or "
                    f"Independent(Normal) base  got "
                    f"Independent({type(inner).__name__})."
                )
            base_scale = inner.scale
        else:
            raise NotImplementedError(
                f"method='autodiff' requires a Normal or "
                f"Independent(Normal) base  got {type(base).__name__}."
            )
        x_star = _grid_plus_newton_autodiff(
            transform,
            base_loc,
            base_scale,
            n_grid=n_grid,
            grid_width=grid_width,
            newton_iters=newton_iters,
        )
        return transform(x_star)

    # --- Sigma-ceiling warning (Python-level, before jit) -----------
    # Only meaningful for Sigmoid/Softplus where the heuristic applies.
    # We must NOT call float() on a traced JAX array (jit) — that raises
    # ConcretizationTypeError. The explicit tracer guard makes this
    # contract obvious  without it the previous code relied on
    # ConcretizationTypeError being a subclass of TypeError and falling
    # through the except clause silently, which works but is fragile.
    if isinstance(transform, (SoftplusTransform, SigmoidTransform)):
        scale_for_check = None
        try:
            scale_for_check = _extract_scale_for_warning(base)
        except (TypeError, AttributeError):
            scale_for_check = None
        # Skip the check entirely under jit (tracer inputs).
        # ``jax.core.is_concrete`` returns True for non-traced arrays
        # and Python scalars  under jit, all JAX inputs are tracers.
        if scale_for_check is not None and _is_concrete(scale_for_check):
            max_sigma = float(jnp.max(scale_for_check))
            if max_sigma > SIGMA_CEILING_WARN:
                msg = (
                    f"jacobian_corrected_map: max(sigma)={max_sigma:.2f} "
                    f"exceeds SIGMA_CEILING_WARN={SIGMA_CEILING_WARN}. "
                    f"The adaptive grid (half-width = "
                    f"{grid_width} * max(sigma, sigma^2)) may not "
                    f"cover the asymptotic modes  result is a "
                    f"heuristic. Pin map_method='transform' for "
                    f"deterministic legacy behavior, or increase "
                    f"grid_width."
                )
                if method == "jacobian":
                    raise NotImplementedError(msg)
                warnings.warn(msg, stacklevel=2)

    # --- Dispatch ----------------------------------------------------
    try:
        x_star = _map_unconstrained(transform, base)
    except NotImplementedError as exc:
        if method == "auto":
            warnings.warn(
                f"jacobian_corrected_map: falling back to transform(loc) "
                f"for ({type(transform).__name__}, "
                f"{type(base).__name__}). Reason: {exc}",
                stacklevel=2,
            )
            return transform(_base_loc(base))
        raise

    return transform(x_star)


def _is_concrete(x) -> bool:
    """Return True if ``x`` is a concrete (non-traced) JAX/Python value.

    Used to gate Python-level branches that need to materialize a JAX
    value to a Python scalar (e.g., for emitting a warning). Inside
    ``jax.jit`` the inputs are ``Tracer`` instances  calling ``float()``
    on them raises ``ConcretizationTypeError``. The guard makes the
    contract explicit and lets the public wrapper compose cleanly with
    jit / vmap.

    Uses ``isinstance(x, jax.core.Tracer)`` rather than a duck-typed
    check because Tracer objects expose all the array methods, so a
    hasattr-based test would give false negatives.
    """
    import jax.core

    return not isinstance(x, jax.core.Tracer)


def _extract_scale_for_warning(base) -> Optional[jnp.ndarray]:
    """Best-effort scale extraction for the sigma-ceiling warning.

    Returns ``None`` if the base type doesn't expose a clear marginal
    scale  the warning is skipped in that case. This is intentionally
    permissive — the warning is advisory, not a correctness gate.
    """
    if isinstance(base, Normal):
        return base.scale
    if isinstance(base, Independent):
        return _extract_scale_for_warning(base.base_dist)
    if isinstance(base, LowRankMultivariateNormal):
        # Marginal stddevs: sqrt(diag(WWᵀ) + cov_diag)
        W = base.cov_factor
        marginal_var = jnp.sum(W**2, axis=-1) + base.cov_diag
        return jnp.sqrt(marginal_var)
    return None
