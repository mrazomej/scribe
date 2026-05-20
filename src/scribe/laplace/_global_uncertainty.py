"""Shared utilities for post-fit Laplace uncertainty on global parameters.

This module provides:

- Helpers to resolve ``model_config.positive_transform`` into JAX
  forward/inverse callables so that the Laplace training path and
  post-fit Hessian calculation use the same positivity map.
- Small-matrix Hessian inversion with symmetrisation and jitter.
- Diagonal curvature-to-scale conversion with relative flooring and
  diagnostics for non-positive (unidentified) curvature.
- Closed-form profiled-curvature helpers that apply the Schur-complement
  correction ``H_{theta,z} H_{zz}^{-1} H_{z,theta}`` without
  differentiating through the inner Newton iterations.

All uncertainty quantities live in the *unconstrained* pre-transform
space. Constrained positive values are obtained by applying the
configured ``positive_transform`` to the ``*_loc`` arrays.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


# =====================================================================
# Positive-transform resolution
# =====================================================================


def _inverse_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable inverse of ``softplus``.

    For large ``x``, ``softplus_inv(x) ≈ x``.  For small ``x``,
    ``softplus_inv(x) = log(expm1(x))`` with a floor to avoid
    ``log(0)``.
    """
    # jnp.log(jnp.expm1(x)) is exact but overflows for large x;
    # for x > 20 the result is indistinguishable from x itself.
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(jnp.maximum(x, 1e-10))))


# Registry mapping ``positive_transform`` config strings to JAX
# ``(forward, inverse)`` callable pairs.  Kept here as the single
# source of truth so every site in the Laplace module (training,
# Hessian, PPC, distribution accessors) resolves the same functions.
_JAX_POSITIVE_FNS: Dict[str, Tuple[Callable, Callable]] = {
    "softplus": (jax.nn.softplus, _inverse_softplus),
    "exp": (jnp.exp, jnp.log),
}


def _get_positive_transform_name(model_config: Any) -> str:
    """Extract the ``positive_transform`` string from a config object.

    Raises ``ValueError`` when the attribute is missing or holds an
    unknown value — callers should never silently fall back to a
    default because that hides coordinate-system mismatches.
    """
    name = getattr(model_config, "positive_transform", None)
    if name is None:
        raise ValueError(
            "model_config does not define 'positive_transform'.  "
            "This is required for all Laplace inference paths."
        )
    if name not in _JAX_POSITIVE_FNS:
        raise ValueError(
            f"Unknown positive_transform={name!r}; "
            f"expected one of {set(_JAX_POSITIVE_FNS)}."
        )
    return name


def resolve_positive_fns(
    model_config: Any,
) -> Tuple[Callable, Callable]:
    """
    Return ``(forward, inverse)`` callables for the configured positivity map.

    Parameters
    ----------
    model_config
        Object with a ``positive_transform`` attribute (``"softplus"``
        or ``"exp"``).

    Returns
    -------
    forward : callable
        Maps unconstrained reals to positive reals.
    inverse : callable
        Maps positive reals back to unconstrained space.

    Raises
    ------
    ValueError
        If ``model_config`` lacks ``positive_transform`` or its value is
        not a recognised transform name.

    Notes
    -----
    The same ``(forward, inverse)`` pair must be used during EM training,
    the final Newton sweep, the Hessian calculation, and downstream PPC
    sampling so that all computations share a single coordinate system.
    """
    return _JAX_POSITIVE_FNS[_get_positive_transform_name(model_config)]


def resolve_numpyro_transform(model_config: Any, param_name: Optional[str] = None):
    """Return the NumPyro ``Transform`` for the configured positivity map.

    Parameters
    ----------
    model_config
        Object with a ``positive_transform`` attribute.  This may be a
        single string (``"softplus"`` / ``"exp"``) or — for models like
        TSLN-Rate / TSLN-Logit — a dict mapping internal parameter
        name to transform name.
    param_name : str, optional
        When the config holds a per-parameter dict, pass the internal
        parameter name (e.g. ``"rate"``, ``"kappa"``, ``"mu"``,
        ``"burst_size"``, ``"k_off"``) to resolve the transform for
        that specific parameter.  When ``None`` (the default), the
        function falls back to ``model_config.positive_transform``
        directly — which works only when that field is itself a
        string.  Models with a dict-form ``positive_transform`` MUST
        pass ``param_name`` to avoid an ambiguity (and to dodge the
        ``"unhashable dict"`` error that the legacy lookup raises).

    Returns
    -------
    numpyro.distributions.transforms.Transform
        ``SoftplusTransform`` or ``ExpTransform``.

    Raises
    ------
    ValueError
        If the resolved transform name is unknown.
    """
    import numpyro.distributions as dist

    # Registry mapping config strings to NumPyro distribution transforms.
    _NUMPYRO_TRANSFORMS: Dict[str, dist.transforms.Transform] = {
        "softplus": dist.transforms.SoftplusTransform(),
        "exp": dist.transforms.ExpTransform(),
    }
    # When a parameter name is supplied AND the model config exposes
    # ``resolve_positive_transform`` (per-parameter resolver), route
    # through it — this is the path TSLN-Rate / TSLN-Logit and any
    # other model with a dict-form ``positive_transform`` must use.
    if param_name is not None and hasattr(
        model_config, "resolve_positive_transform"
    ):
        name = model_config.resolve_positive_transform(param_name)
        if name not in _NUMPYRO_TRANSFORMS:
            raise ValueError(
                f"Unknown positive_transform={name!r} for "
                f"param={param_name!r}; expected one of "
                f"{set(_NUMPYRO_TRANSFORMS)}."
            )
        return _NUMPYRO_TRANSFORMS[name]
    # Legacy path: single-string positive_transform.
    return _NUMPYRO_TRANSFORMS[_get_positive_transform_name(model_config)]


# =====================================================================
# Small Hessian inversion (for LNM 2x2 totals block)
# =====================================================================


def invert_hessian_with_jitter(
    H: jnp.ndarray,
    max_jitter_iters: int = 6,
    jitter_init: float = 1e-6,
) -> Tuple[jnp.ndarray, float]:
    """Symmetrise, add diagonal jitter if needed, and invert a small PSD matrix.

    Parameters
    ----------
    H : jnp.ndarray, shape ``(d, d)``
        Hessian of the *negative* objective (should be positive-definite
        at a local minimum).
    max_jitter_iters : int
        Maximum number of jitter-doubling attempts before giving up.
    jitter_init : float
        Starting diagonal jitter.

    Returns
    -------
    cov : jnp.ndarray, shape ``(d, d)``
        Inverse of ``H`` (posterior covariance).
    jitter_used : float
        Total jitter added (0.0 if none was needed).
    """
    H_sym = 0.5 * (H + H.T)
    d = H_sym.shape[0]

    def _try_chol(H_j):
        L = jnp.linalg.cholesky(H_j)
        return jnp.all(jnp.isfinite(L)), L

    ok, L = _try_chol(H_sym)
    jitter = jitter_init
    total_jitter = 0.0

    for _ in range(max_jitter_iters):
        if ok:
            break
        H_sym = H_sym + jitter * jnp.eye(d)
        total_jitter += jitter
        ok, L = _try_chol(H_sym)
        jitter *= 2.0

    # Fall back to pseudo-inverse when Cholesky still fails.
    cov = jnp.where(
        ok,
        jax.scipy.linalg.cho_solve((L, True), jnp.eye(d)),
        jnp.linalg.pinv(H_sym),
    )
    return cov, total_jitter


# =====================================================================
# Diagonal curvature → scale with relative flooring
# =====================================================================


def curvature_to_scale(
    hessian_diag: jnp.ndarray,
    curvature_floor_factor: float = 1e-6,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Convert diagonal Hessian entries to posterior scales with flooring.

    The posterior standard deviation for each coordinate is
    ``1 / sqrt(H_ii)`` where ``H_ii > 0``.  Non-positive entries
    indicate locally unidentified parameters and are floored to a
    small fraction of the median positive curvature.

    Parameters
    ----------
    hessian_diag : jnp.ndarray, shape ``(G,)``
        Diagonal of the profiled negative-objective Hessian (should be
        positive for identified parameters).
    curvature_floor_factor : float
        Floor = ``factor * median(positive entries)``.

    Returns
    -------
    scale : jnp.ndarray, shape ``(G,)``
        Posterior standard deviations in unconstrained space.
    diagnostics : dict
        ``hessian_min`` — smallest diagonal entry.
        ``floor_count`` — number of entries that were floored.
        ``curvature_floor`` — the floor value used.
    """
    positive_mask = hessian_diag > 0.0
    n_positive = jnp.sum(positive_mask)

    # Median of positive entries; fall back to 1.0 if none are positive
    # (pathological — entire parameter vector is unidentified).
    sorted_pos = jnp.sort(jnp.where(positive_mask, hessian_diag, jnp.inf))
    median_pos = jnp.where(
        n_positive > 0,
        sorted_pos[n_positive.astype(int) // 2],
        1.0,
    )
    curvature_floor = curvature_floor_factor * median_pos

    # Floor non-positive or tiny entries.
    floored = jnp.maximum(hessian_diag, curvature_floor)
    floor_count = int(jnp.sum(hessian_diag < curvature_floor))
    hessian_min = float(jnp.min(hessian_diag))

    n_nonpositive = int(jnp.sum(~positive_mask))
    if n_nonpositive > 0:
        logger.warning(
            "Global Laplace: %d / %d curvature entries are non-positive "
            "(locally unidentified); floored to %.2e.  "
            "Min curvature = %.2e.",
            n_nonpositive,
            hessian_diag.shape[0],
            float(curvature_floor),
            hessian_min,
        )

    scale = 1.0 / jnp.sqrt(floored)
    diagnostics = {
        "hessian_min": jnp.asarray(hessian_min, dtype=jnp.float32),
        "floor_count": jnp.asarray(floor_count, dtype=jnp.int32),
        "curvature_floor": jnp.asarray(
            float(curvature_floor), dtype=jnp.float32
        ),
    }
    return scale, diagnostics


# =====================================================================
# Woodbury diagonal helpers for profiled-Hessian corrections
# =====================================================================


def woodbury_inv_diag(
    W: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    r"""Diagonal of ``(W W^T + diag(d))^{-1}`` via the Woodbury identity.

    Uses the factored form:

    .. math::
        \mathrm{diag}(\Sigma^{-1})
        = \mathrm{diag}(\tilde M^{-1})
        - \text{rowsumsq}\!\bigl(\tilde M^{-1}\, V\, L_S^{-T}\bigr)

    where :math:`\tilde M = \mathrm{diag}(d)`,
    :math:`V = W`, and :math:`S = I + W^T \tilde M^{-1} W`.

    Parameters
    ----------
    W : jnp.ndarray, shape ``(G, k)``
        Factor loading matrix.
    d : jnp.ndarray, shape ``(G,)``
        Diagonal variance vector (must be positive).

    Returns
    -------
    inv_diag : jnp.ndarray, shape ``(G,)``
        Diagonal entries of the inverse covariance.
    """
    inv_d = 1.0 / d
    k = W.shape[1]
    # S = I_k + W^T diag(1/d) W
    S = jnp.eye(k) + W.T @ (inv_d[:, None] * W)
    L_S = jnp.linalg.cholesky(S)
    # T = diag(1/d) W  L_S^{-T}  — shape (G, k)
    # We need L_S^{-T} W^T diag(1/d) but arranged for row-sum-of-squares.
    # Solve L_S Z = (diag(1/d) W)^T  => Z = L_S^{-1} W^T diag(1/d)
    # Then T = (diag(1/d) W) L_S^{-T} = Z^T  is (G, k).
    rhs = (inv_d[:, None] * W).T  # (k, G)
    Z = jax.scipy.linalg.solve_triangular(L_S, rhs, lower=True)  # (k, G)
    T = Z.T  # (G, k)
    # diag(Sigma^{-1}) = diag(M_tilde^{-1}) - rowsumsq(T)
    return inv_d - jnp.sum(T * T, axis=-1)


def woodbury_inv_diag_and_col(
    W: jnp.ndarray,
    d: jnp.ndarray,
    d_extra: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""Diagonal of the x-block inverse and the x-eta cross column.

    For the joint ``(x, eta)`` Hessian block with structure:

    .. math::
        H = \begin{pmatrix} A & b \\ b^T & c \end{pmatrix}

    where ``A = W W^T + diag(d)`` is the x-block Hessian,
    ``b`` is a column vector (x-eta cross), and ``c`` is the eta
    diagonal entry (scalar), this function returns
    ``diag(A^{-1})`` and the column ``A^{-1} e`` where ``e`` is
    a specific unit direction (used for the Schur complement).

    For the NBLN capture-anchor case the x-eta cross-derivative
    ``H_{x_g, eta}`` is diagonal across genes, so we only need
    the diagonal of ``A^{-1}`` and one specific column pattern.

    Parameters
    ----------
    W : jnp.ndarray, shape ``(G, k)``
    d : jnp.ndarray, shape ``(G,)``
        Positive diagonal.
    d_extra : float
        Not used in this overload; kept for API symmetry.

    Returns
    -------
    inv_diag : jnp.ndarray, shape ``(G,)``
        Diagonal of ``A^{-1}`` (same as :func:`woodbury_inv_diag`).
    L_S_factor : jnp.ndarray, shape ``(k, k)``
        Cholesky of ``S = I + W^T diag(1/d) W``; callers that need
        ``A^{-1} v`` for a specific ``v`` can reuse this.
    """
    inv_d = 1.0 / d
    k = W.shape[1]
    S = jnp.eye(k) + W.T @ (inv_d[:, None] * W)
    L_S = jnp.linalg.cholesky(S)
    rhs = (inv_d[:, None] * W).T
    Z = jax.scipy.linalg.solve_triangular(L_S, rhs, lower=True)
    T = Z.T
    inv_diag = inv_d - jnp.sum(T * T, axis=-1)
    return inv_diag, L_S
