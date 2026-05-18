"""Data-derived initialization for the TwoState (Poisson-Beta) family.

For each gene ``g`` we anchor the prior loc of ``mu_g`` at the
unconstrained-space value whose transformed image equals the
**pre-capture** empirical mean — that is, ``mu_g`` is the *biological*
mean before the per-cell capture factor ``ν_c`` thins it, so the
anchor must divide the observed sample mean by an estimate of the
mean capture probability.  Concretely:

- ``mu_init_g = softplus_inv(observed_mean_g / mean_capture)``
  (under softplus; ``log(observed_mean_g / mean_capture)`` under
  exp).
- ``mean_capture`` is estimated from the model config:
  - ``twostate`` (no VCP): ``1.0``.
  - ``twostatevcp`` with biology-informed prior
    (``priors={"capture_efficiency": (log_M_0, sigma_M)}``):
    ``mean(L_c) / M_0``, derived from
    ``p_c = exp(-eta_c)`` with
    ``eta_c ~ Normal(log_M_0 - log(L_c), sigma_M^2)``.
  - ``twostatevcp`` with a flat ``priors.p_capture = (alpha, beta)``:
    ``alpha / (alpha + beta)``.
  - ``twostatevcp`` with no explicit capture prior: the Beta(1, 1)
    mean, ``0.5``.

Without the capture correction, the anchor sits at the *observed*
mean, which leaves the variational ``mu_loc`` for high-expression
genes systematically too small by exactly the mean-capture factor.
Under softplus's asymptotically-linear regime, that bias is sticky
and produces a downward bend in mean-calibration plots at the high
end (the symptom the auditor flagged).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


__all__ = [
    "empirical_log_mean_from_counts",
    "empirical_mu_anchor_from_counts",
    "estimate_initial_mean_capture",
    "inject_twostate_data_init",
]


def empirical_log_mean_from_counts(
    counts,
    pseudocount: float = 1.0,
) -> jnp.ndarray:
    """Compute per-gene empirical log-mean of the count matrix.

    Parameters
    ----------
    counts : array_like, shape ``(n_cells, n_genes)``
        Raw count matrix (after any gene-coverage filtering).
    pseudocount : float, default 1.0
        Additive pseudocount applied before the log to keep
        zero-expression genes finite.

    Returns
    -------
    jnp.ndarray, shape ``(n_genes,)``
        ``log(mean(counts[:, g]) + pseudocount)`` per gene.
    """
    counts_arr = jnp.asarray(counts, dtype=jnp.float32)
    per_gene_mean = counts_arr.mean(axis=0)
    return jnp.log(per_gene_mean + pseudocount)


def estimate_initial_mean_capture(model_config: Any, counts) -> float:
    """Estimate a robust initial-mean per-cell capture probability.

    Used to anchor ``mu`` at the pre-capture mean for TwoStateVCP
    fits (see module docstring for the closure-under-thinning
    rationale).

    **This is an initialization heuristic, not a posterior estimate.**
    We intentionally do **not** read the biology-informed prior
    (``priors.eta_capture = (log_M_0, sigma_M)``) here.  That prior
    encodes an order-of-magnitude *belief* about ``M_0``; when the
    true posterior median of ``mean(p_capture)`` is far from the
    prior median (the typical case for a loose prior),
    using the prior median as a divisor for ``mu_init`` over-shoots
    by exactly the ratio, leaving the anchor too high and the
    posterior-predictive bands above the data.  Empirically a
    ``M_0 = 100_000`` prior on a dataset whose fitted ``ν̄ ≈ 0.6``
    yields a prior-implied ``mean(p_c) ≈ 0.16`` — an over-correction
    of ~4× when used as an initialization divisor.

    The robust alternative is a fixed conservative default
    (Beta(1, 1) prior mean) that overshoots by at most ~2× for any
    realistic ``ν̄``, which softplus-Normal can recover within a
    modest number of SVI steps.  Users who want to override the
    default can pass an explicit ``priors={"p_capture": (alpha,
    beta)}`` with a tighter ``alpha / (alpha + beta)``.

    Parameters
    ----------
    model_config : ModelConfig
        Freshly built model config.  The ``base_model`` field plus
        an explicit ``priors.p_capture`` tuple control the estimate.
    counts : array_like, shape ``(n_cells, n_genes)``
        Observed count matrix.  Not used in the current heuristic
        but accepted for forward compatibility.

    Returns
    -------
    float
        Initial estimate of ``mean(p_capture)`` over cells.  Returns
        ``1.0`` for non-VCP models and for ``twostatevcp`` without an
        explicit Beta tuple; ``alpha / (alpha + beta)`` when
        ``priors.p_capture = (alpha, beta)``.

    Note
    ----
    The default for ``twostatevcp`` was previously ``0.5`` — a
    conservative divisor intended to nudge the anchor up to account
    for capture loss.  Empirically this still left the anchor away
    from the true pre-capture mean by a non-trivial factor (the
    posterior median of ``mean(p_capture)`` is rarely exactly 0.5),
    and the resulting under/over-shoot in the unconstrained
    ``mu_loc`` was slow to recover under softplus.

    The proper fix is on the *transform* side: when ``mu`` uses an
    exp transform (LogNormal on the constrained scale), the
    optimizer takes multiplicative steps in ``mu`` and recovers the
    correct pre-capture mean in a few SVI iterations even when the
    anchor is off by the capture factor.  Returning ``1.0`` here
    keeps the anchor at the *observed* per-gene mean and defers
    the capture-loss correction to the optimizer, which is the
    cleaner separation.  Users who still want a pre-correction can
    supply ``priors.p_capture = (alpha, beta)`` explicitly.
    """
    del counts  # not used by the current heuristic
    base_model = getattr(model_config, "base_model", None)
    if base_model != "twostatevcp":
        return 1.0

    extras = (
        getattr(model_config.priors, "__pydantic_extra__", None) or {}
    )

    # Explicit flat Beta prior: priors.p_capture = (alpha, beta).
    # Honour the user's intended prior mean as the initialization
    # divisor — this is the one place the user can control the
    # anchor without overriding the entire ``priors.mu`` block.
    p_cap_prior = extras.get("p_capture")
    if (
        p_cap_prior is not None
        and isinstance(p_cap_prior, (tuple, list))
        and len(p_cap_prior) == 2
    ):
        a, b = float(p_cap_prior[0]), float(p_cap_prior[1])
        if a > 0 and b > 0:
            return a / (a + b)

    # Default for twostatevcp: no pre-correction; anchor at the
    # observed per-gene mean and let the optimizer (especially under
    # ``positive_transform={"mu": "exp"}``) recover the capture loss.
    return 1.0


def empirical_mu_anchor_from_counts(
    counts,
    transform: str = "softplus",
    mean_capture: float = 1.0,
    floor: float = 1e-3,
    ceil: float = 1e6,
) -> jnp.ndarray:
    """Per-gene unconstrained-space anchor for ``mu``.

    Computes the empirical per-gene mean divided by the estimated
    ``mean_capture``, clamps the result to ``[floor, ceil]``, then
    maps through the inverse of the configured positive transform.

    Parameters
    ----------
    counts : array_like, shape ``(n_cells, n_genes)``
        Raw count matrix.
    transform : {"softplus", "exp"}, default "softplus"
        The positive transform used in the model.  The inverse used
        here is ``log(expm1(x))`` for softplus and ``log(x)`` for exp.
    mean_capture : float, default 1.0
        Estimated ``mean(p_capture)`` under the prior.  The observed
        per-gene mean is divided by this to recover an estimate of
        the *pre-capture* mean ``mu_g``.  Use ``1.0`` for non-VCP
        models.  See :func:`estimate_initial_mean_capture` for the
        VCP routing.
    floor : float, default 1e-3
        Lower clamp on ``observed_mean / mean_capture`` before
        inversion (keeps zero-expression genes finite under both
        ``log`` and ``log·expm1``).
    ceil : float, default 1e6
        Upper clamp; protects ``expm1`` from overflowing in float32.

    Returns
    -------
    jnp.ndarray, shape ``(n_genes,)``
        Per-gene unconstrained-space loc for the ``mu`` prior.
    """
    counts_arr = jnp.asarray(counts, dtype=jnp.float32)
    per_gene_mean = counts_arr.mean(axis=0)
    # Pre-capture mean estimate: divide the observed sample mean by
    # the prior mean of the per-cell capture probability.
    safe_capture = float(max(float(mean_capture), 1e-4))
    per_gene_mean = per_gene_mean / safe_capture
    per_gene_mean = jnp.clip(per_gene_mean, min=floor, max=ceil)

    if transform == "softplus":
        # Stable softplus_inv: log(expm1(x)) overflows in float32
        # for x ≳ 88, so we split on a threshold.
        # For x <= 20: log(expm1(x))   — the naive form, stable here.
        # For x  > 20: x + log1p(-exp(-x))
        #              ≡ log(exp(x) - 1) without computing exp(x).
        # The two branches agree to high precision around x=20.
        return jnp.where(
            per_gene_mean > 20.0,
            per_gene_mean + jnp.log1p(-jnp.exp(-per_gene_mean)),
            jnp.log(jnp.expm1(per_gene_mean)),
        )
    if transform == "exp":
        return jnp.log(per_gene_mean)
    raise ValueError(
        f"Unknown positive transform {transform!r}; expected "
        "'softplus' or 'exp'."
    )


def inject_twostate_data_init(model_config, counts):
    """Stash a per-gene ``mu_prior_loc`` array on the priors extras.

    The factory reads ``priors.__pydantic_extra__["mu_prior_loc"]`` and
    replaces the default scalar prior loc on ``mu`` with this array.
    Each gene's variational ``mu_loc`` then initializes at its own
    *pre-capture* empirical mean (in unconstrained space), so the
    optimizer doesn't need to climb decades on the first SVI step.

    For ``twostatevcp`` fits the observed per-gene mean is divided
    by an estimate of ``mean(p_capture)`` derived from the configured
    capture prior — see :func:`estimate_initial_mean_capture`.  This
    is the fix for the systematic shrinkage of high-expression genes
    in mean-calibration plots: anchoring at the observed mean leaves
    the variational ``mu_loc`` too small by exactly the capture
    factor in the asymptotically-linear softplus regime.

    Parameters
    ----------
    model_config : ModelConfig
        Freshly built configuration.  The ``positive_transform`` field
        (default "softplus") controls the inverse-transform used to
        compute the per-gene anchor.
    counts : array_like, shape ``(n_cells, n_genes)``
        Count matrix after gene-coverage filtering.

    Returns
    -------
    ModelConfig
        New ``ModelConfig`` with the priors extras augmented.  The
        original is not mutated.
    """
    # The anchor is for ``mu`` specifically, so resolve the per-parameter
    # positive transform (handles both string and Dict[str, str] forms
    # of ``ModelConfig.positive_transform``).  Falls back to a direct
    # field read for legacy configs missing the resolver.
    if hasattr(model_config, "resolve_positive_transform"):
        transform = model_config.resolve_positive_transform("mu")
    else:
        transform = getattr(model_config, "positive_transform", "softplus")
    mean_capture = estimate_initial_mean_capture(model_config, counts)
    mu_prior_loc = empirical_mu_anchor_from_counts(
        counts, transform=transform, mean_capture=mean_capture
    )

    from ..models.config.groups import PriorOverrides

    existing_extras = (
        getattr(model_config.priors, "__pydantic_extra__", None) or {}
    )
    # PriorOverrides is a Pydantic model with model_config["extra"] =
    # "allow", so the extra fields round-trip via the constructor.
    updated = dict(existing_extras)
    updated["mu_prior_loc"] = mu_prior_loc
    # Stash the mean-capture estimate too, so the API-stage logger
    # can surface it to the user (see ``_inject_twostate_data_init``
    # in api/stages/model_config_build.py).
    updated["_mu_init_mean_capture"] = mean_capture
    # Carry forward any structured priors fields from the original
    # PriorOverrides (Beta/LogNormal tuples for p_capture, etc.).
    if model_config.priors is not None:
        for name in model_config.priors.model_fields_set:
            if name not in updated:
                updated[name] = getattr(model_config.priors, name)

    return model_config.model_copy(
        update={"priors": PriorOverrides(**updated)}
    )
