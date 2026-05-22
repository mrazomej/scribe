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

from typing import Dict

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


def is_lnm_or_pln_results(obj) -> bool:
    """Whether ``obj`` is a fitted LNM or PLN result object (Laplace or VAE).

    The parametric DE path consumes the *fitted globals* (μ, 𝑊, 𝑑)
    rather than posterior samples, and these can be extracted
    directly from any LNM- or PLN-family result regardless of
    inference path.  This predicate gates the auto-conversion in
    :func:`scribe.de.compare` for the **parametric** branch only.
    See :func:`has_compositional_marginal` for the broader predicate
    used by the empirical marginal-driven path.
    """
    cfg = getattr(obj, "model_config", None)
    if cfg is None:
        return False
    bm = getattr(cfg, "base_model", None)
    bm_str = str(getattr(bm, "value", bm) or "").lower()
    return bm_str in ("lnm", "lnmvcp", "pln")


def has_compositional_marginal(obj) -> bool:
    """Whether ``obj`` exposes a generative-marginal compositional sampler.

    True for any result object whose fitted globals (μ, 𝑊, 𝑑) define
    a marginal compositional distribution sampled via
    ``softmax_full(𝒩(μ, 𝑊𝑊ᵗ + diag(𝑑)))``.  Currently includes:

    * **LNM** / **LNMVCP** — latent in (G−1)-dim ALR coordinates.
    * **PLN** — latent in G-dim log-rate space.
    * **NBLN** — same generative marginal as PLN with an added per-gene
      Negative Binomial dispersion at the observation step.  The
      composition distribution is identical to PLN's (softmax of
      log-rates) because the NB layer is observation noise that
      averages out under softmax.
    * **TSLN-Rate** / **TSLN-Logit** — log-rate marginal identical to
      NBLN under the two-state observation noise.

    The empirical DE path in :func:`scribe.de.compare` uses this
    predicate (rather than :func:`is_lnm_or_pln_results`) to short-
    circuit to :func:`_compare_empirical_from_marginal`, which calls
    ``results.get_compositional_samples()`` and feeds the resulting
    simplex draws through the standard CLR-difference machinery.

    The presence of an ``_other`` pooled column from
    ``gene_coverage < 1.0`` is handled by ``compare()``'s auto-mask
    logic: when both results have ``_gene_coverage_mask`` set, the
    trailing column is treated as the ``other`` pseudo-gene and
    dropped from the returned CLR differences.

    Parameters
    ----------
    obj
        Candidate results object.

    Returns
    -------
    bool
    """
    if not hasattr(obj, "get_compositional_samples"):
        return False
    cfg = getattr(obj, "model_config", None)
    if cfg is None:
        return False
    bm = getattr(cfg, "base_model", None)
    bm_str = str(getattr(bm, "value", bm) or "").lower()
    return bm_str in (
        "lnm",
        "lnmvcp",
        "pln",
        "nbln",
        "twostate_ln_rate",
        "twostate_ln_logit",
    )


def lnm_or_pln_results_to_parametric_dict(
    results,
    *,
    reference_idx_for_pln: int = -1,
) -> Dict[str, jnp.ndarray]:
    """Extract ALR-form parametric inputs from an LNM or PLN result object.

    Both LNM and PLN low-rank-Gaussian fits parameterise their
    per-cell latent prior as 𝒩(μ, 𝑊𝑊ᵗ + diag(𝑑)).  The parametric
    DE pipeline expects (G−1)-dimensional ALR-form inputs.  This
    adapter normalises across:

    * **LNM / LNMVCP**: latent already in (G−1)-dim ALR coordinates;
      pull (μ_alr, 𝑊_alr, 𝑑_alr) from the result and wrap.
    * **PLN**: latent in G-dim log-rate space; convert to the
      equivalent ALR form via :func:`pln_to_alr_form`.  The latent
      rank gains one extra column (k → k+1) to absorb the rank-1
      cross-coupling that arises from the ALR transform of
      diag(𝑑_pln).

    Works on both ``ScribeLaplaceResults`` (which exposes
    ``mu`` / ``W`` / ``d`` directly as attributes) and
    ``ScribeVAEResults`` (which exposes them via
    ``get_lnm_*`` / ``get_pln_*`` methods).

    Parameters
    ----------
    results : object
        ``ScribeLaplaceResults`` or ``ScribeVAEResults`` with
        PLN/LNM ``base_model``.
    reference_idx_for_pln : int, default -1
        Reference gene index for the PLN ALR transform.  Ignored
        for LNM fits (which already have a fitted
        ``alr_reference_idx``).

    Returns
    -------
    dict
        ``{"loc": μ_alr, "cov_factor": W_alr, "cov_diag": d_alr}``
        in (G−1) dims.  Consumed by
        :func:`scribe.de._results_factory._compare_parametric` the
        same way any other fitted-logistic-normal dict is.

    Raises
    ------
    NotImplementedError
        If the LNM result has a non-default ``alr_reference_idx``
        that doesn't match the convention assumed by the
        downstream parametric pipeline (reference at the last
        gene).  Real-world fits use the default.
    """
    cfg = results.model_config
    bm = getattr(cfg, "base_model", None)
    bm_str = str(getattr(bm, "value", bm) or "").lower()

    def _pull(name_pln: str, name_lnm: str, name_attr: str):
        """VAE-style accessor first, then bare attribute (Laplace)."""
        if bm_str == "pln" and hasattr(results, name_pln):
            return getattr(results, name_pln)()
        if bm_str in ("lnm", "lnmvcp") and hasattr(results, name_lnm):
            return getattr(results, name_lnm)()
        return getattr(results, name_attr)

    if bm_str in ("lnm", "lnmvcp"):
        mu = _pull("get_lnm_mu", "get_lnm_mu", "mu")
        W = _pull("get_lnm_W", "get_lnm_W", "W")
        d = _pull("get_lnm_d", "get_lnm_d", "d")
        # Normalize d=None (low_rank d_mode) to zero vector so
        # the dict has a uniform shape downstream.
        if d is None:
            d = jnp.zeros(W.shape[0], dtype=W.dtype)
        # Parametric downstream defaults to ref=-1; LNM fits with
        # other references would need plumbing we don't have yet.
        ref = int(getattr(cfg, "alr_reference_idx", -1))
        n_genes = int(W.shape[0]) + 1
        if ref not in (-1, n_genes - 1):
            raise NotImplementedError(
                f"LNM parametric DE currently assumes "
                f"alr_reference_idx == -1 (last gene); got {ref}. "
                "Re-fit with the default reference, or open an "
                "issue for explicit reference-index plumbing."
            )
        return {
            "loc": jnp.asarray(mu),
            "cov_factor": jnp.asarray(W),
            "cov_diag": jnp.asarray(d),
        }

    if bm_str == "pln":
        from ._transforms import pln_to_alr_form

        mu = _pull("get_pln_mu", "get_pln_mu", "mu")
        W = _pull("get_pln_W", "get_pln_W", "W")
        d = _pull("get_pln_d", "get_pln_d", "d")
        if d is None:
            d = jnp.zeros(W.shape[0], dtype=W.dtype)
        mu_alr, W_alr_ext, d_alr = pln_to_alr_form(
            jnp.asarray(mu), jnp.asarray(W), jnp.asarray(d),
            reference_idx=reference_idx_for_pln,
        )
        return {
            "loc": mu_alr,
            "cov_factor": W_alr_ext,
            "cov_diag": d_alr,
        }

    raise ValueError(
        f"DE parametric path supports LNM, LNMVCP, and PLN models; "
        f"got base_model={bm_str!r}"
    )
