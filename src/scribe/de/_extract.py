"""Dimension-aware parameter extraction for differential expression.

This module provides ``extract_alr_params``, a function that intelligently
extracts (D-1)-dimensional ALR (Additive Log-Ratio) parameters from various
model representations.  It resolves the critical dimensional mismatch between
``fit_logistic_normal_from_posterior`` (which returns D-dimensional *embedded*
ALR parameters -- with a zero appended for the reference component) and the
DE transformation functions (which expect raw (D-1)-dimensional ALR
parameters).

The function inspects the input and:

1. For **dicts** from ``fit_logistic_normal_from_posterior``:
   Detects the trailing-zero embedding pattern and strips the reference
   component to produce consistent (D-1)-dimensional ALR.

2. For **LowRankLogisticNormal** distributions:
   Parameters are already (D-1)-dimensional ALR -- returns them as-is.

3. For **SoftmaxNormal** distributions:
   Parameters are D-dimensional (embedded ALR).  The trailing reference
   component is stripped to produce (D-1)-dimensional ALR.
"""

import jax.numpy as jnp

from ..stats.distributions import LowRankLogisticNormal, SoftmaxNormal


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def extract_alr_params(model):
    """Extract consistent (D-1)-dimensional ALR parameters from a model.

    This function resolves a critical dimensional mismatch: the output of
    ``fit_logistic_normal_from_posterior`` stores D-dimensional *embedded* ALR
    parameters (where the reference component is appended as zero), but the
    ALR-to-CLR transformation functions expect (D-1)-dimensional raw ALR.

    The detection heuristic checks whether the last element of ``cov_diag``
    is approximately zero (|d[-1]| < 1e-10) **and** the last row of
    ``cov_factor`` is all zeros.  If both hold, the parameters are
    D-dimensional embedded ALR and the reference slice is stripped.

    Parameters
    ----------
    model : dict or LowRankLogisticNormal or SoftmaxNormal
        A fitted logistic-normal model.  Accepted forms:

        * **dict** with keys ``'loc'``, ``'cov_factor'``, ``'cov_diag'``.
          The dict may come from ``fit_logistic_normal_from_posterior``
          (D-dimensional embedded ALR) or may already be (D-1)-dimensional.
        * **LowRankLogisticNormal** distribution object -- parameters are
          always (D-1)-dimensional ALR.
        * **SoftmaxNormal** distribution object -- parameters are always
          D-dimensional (embedded ALR).

    Returns
    -------
    mu : jnp.ndarray, shape (D-1,)
        Mean in raw ALR space.
    W : jnp.ndarray, shape (D-1, k)
        Low-rank factor in raw ALR space.
    d : jnp.ndarray, shape (D-1,)
        Diagonal of the covariance in raw ALR space.

    Raises
    ------
    TypeError
        If ``model`` is not a dict, ``LowRankLogisticNormal``, or
        ``SoftmaxNormal``.

    Notes
    -----
    This function is intentionally conservative: it only strips the last
    dimension when there is strong evidence of the embedding pattern.
    If parameters are already (D-1)-dimensional, they are returned
    unchanged.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> # Embedded ALR dict (D=5, D-1=4)
    >>> model = {
    ...     'loc': jnp.array([0.1, 0.2, 0.3, 0.4, 0.0]),
    ...     'cov_factor': jnp.concatenate([
    ...         jnp.ones((4, 2)) * 0.1,
    ...         jnp.zeros((1, 2))
    ...     ], axis=0),
    ...     'cov_diag': jnp.array([0.5, 0.5, 0.5, 0.5, 0.0]),
    ... }
    >>> mu, W, d = extract_alr_params(model)
    >>> assert mu.shape == (4,)
    """
    # ----- LowRankLogisticNormal: already (D-1)-dimensional ALR -----
    if isinstance(model, LowRankLogisticNormal):
        # The distribution stores its params in the (D-1)-dimensional ALR
        # space directly, so no stripping is needed.
        return model.loc, model.cov_factor, model.cov_diag

    # ----- SoftmaxNormal: always D-dimensional (embedded ALR) -----
    if isinstance(model, SoftmaxNormal):
        # SoftmaxNormal uses D-dimensional parameters (embedded ALR with
        # a zero appended for the reference component).  Strip it.
        return (
            model.loc[..., :-1],
            model.cov_factor[:-1, :],
            model.cov_diag[..., :-1],
        )

    # ----- dict: may be D-dim embedded or (D-1)-dim raw ALR -----
    if isinstance(model, dict):
        # Extract the raw arrays from the dict
        mu = jnp.asarray(model["loc"])
        W = jnp.asarray(model["cov_factor"])
        d = jnp.asarray(model["cov_diag"])

        # Detect the embedding pattern: last element of d is ~0 and last
        # row of W is all zeros.  This is the signature of the embedding
        # step in fit_logistic_normal_from_posterior (line 465-469).
        last_diag_is_zero = jnp.abs(d[-1]) < 1e-10
        last_row_W_is_zero = jnp.allclose(W[-1, :], 0.0, atol=1e-10)

        if last_diag_is_zero and last_row_W_is_zero:
            # D-dimensional embedded ALR -- strip the reference component
            return mu[:-1], W[:-1, :], d[:-1]
        else:
            # Already (D-1)-dimensional raw ALR -- return as-is
            return mu, W, d

    # ----- Unsupported type -----
    raise TypeError(
        f"Unsupported model type: {type(model)}.  Expected dict with keys "
        f"['loc', 'cov_factor', 'cov_diag'], LowRankLogisticNormal, or "
        f"SoftmaxNormal."
    )
