"""
Per-donor regulatory-program activity scales for hierarchical gene-gene
correlation (NB-LogNormal "Rung 1").

This module implements the *shared primitive* behind the hierarchical
correlation model described in ``paper/_nb_lognormal.qmd``
(``sec-nbln-hierarchical-correlation``). Both inference paths consume it:

- **Phase A (SVI / VAE sanity check)** scales the *K-dimensional latent*
  prior per donor — ``z_c ~ Normal(0, diag(s_{sigma(c)}^2))`` — so the shared
  linear decoder ``W`` *induces* the donor-specific log-rate covariance
  ``W diag(s_d^2) W^T + diag(d)``.
- **Phase B (Laplace production)** folds the same ``s_d`` into *effective
  loadings* ``W_eff,d = W . diag(s_d)`` so that
  ``Sigma_d = W_eff,d W_eff,d^T + diag(d)`` keeps the low-rank-plus-diagonal
  form the Woodbury / Newton kernel already consumes.

The model
---------
For ``D`` donors and ``K`` latent regulatory programs, each donor ``d`` carries
a *relative* program-activity vector ``s_d in R_+^K``. The hierarchical,
non-centered (NCP) prior is

.. math::

    \\log s_{d,k} = \\tau_s \\, \\tilde\\varepsilon_{d,k}, \\qquad
    \\tilde\\varepsilon_{\\cdot,k}
        = \\varepsilon_{\\cdot,k} - \\tfrac{1}{D}\\sum_{d'} \\varepsilon_{d',k},
    \\qquad \\varepsilon_{d,k} \\sim \\mathcal N(0, 1),

with a single shared scale ``tau_s ~ Softplus(Normal(tau_loc, tau_scale))``.

The **sum-to-zero constraint** ``mean_d log s_{d,k} = 0`` is what makes the
parameterization identifiable: it removes the per-column *scale* gauge
``W[:, k] -> a_k W[:, k]``, ``s_{d,k} -> s_{d,k} / a_k`` that otherwise leaves
every ``Sigma_d`` unchanged. With the constraint the absolute magnitude of
program ``k`` lives entirely in ``||W[:, k]||`` and ``s_d`` is purely *relative*
donor activity (geometric mean 1 across donors per program). The *rotation*
gauge ``W -> W R`` is broken only generically (when donor profiles are
sufficiently distinct); column **sign** and **permutation** gauges always
remain and are handled at interpretation time, exactly as for the
single-dataset ``W``.

Implementation note (centering vs. contrasts)
--------------------------------------------
We enforce the constraint by *centering* the raw NCP draws across donors. This
leaves the raw ``(D, K)`` latent with one redundant degree of freedom per
column (the column mean maps to a flat, prior-killed direction), which is
harmless for SVI and Laplace alike. A strictly ``(D-1)``-dof Helmert /
sum-to-zero contrast basis would remove the redundancy but complicates the
gather; centering is the pragmatic, standard choice.
"""

from __future__ import annotations

import math
from typing import Tuple

import jax.numpy as jnp
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist

# Site-name suffixes for the program-scale block. Centralized here so the
# model-side sampler and the guide-side block agree on every site name (a
# mismatch would silently break the mean-field ELBO pairing).
_TAU_RAW_SUFFIX = "_tau_raw"  # scalar Normal raw for the shared scale tau_s
_RAW_SUFFIX = "_raw"  # (D, K) Normal(0, 1) NCP raw effects
_LOG_SUFFIX = "_log"  # (D, K) deterministic log s_{d,k} (post sum-to-zero)
# The bare ``site_prefix`` (no suffix) names the constrained ``s`` deterministic.

# Default hyperprior on the shared between-donor scale tau_s. Mirrors the
# ``tau_mu`` treatment in ``paper/_hierarchical_datasets.qmd`` (softplus of a
# Normal placing prior mass on small values): softplus(-2) ~ 0.127, i.e. a
# prior median around 13% multiplicative donor-to-donor program variation.
_DEFAULT_TAU_LOC = -2.0
_DEFAULT_TAU_SCALE = 0.5


def program_scales_from_raw(
    raw: jnp.ndarray, tau_raw: jnp.ndarray
) -> jnp.ndarray:
    """Map NCP raw effects + unconstrained ``tau`` to constrained scales ``s``.

    This is the **single source of truth** for the raw→``s`` transform, shared
    by the SVI sampler (:func:`sample_program_scales`) and the Laplace obs
    model (which optimizes ``raw``/``tau_raw`` as plain params and rebuilds
    ``s`` with this function). Keeping the gauge in one place means the two
    inference backends cannot drift.

    Applies, in order: ``tau = softplus(tau_raw)``; the sum-to-zero centering
    of ``raw`` across donors (per program); then ``s = exp(tau · centered)``.

    Parameters
    ----------
    raw : jnp.ndarray, shape ``(n_datasets, latent_dim)``
        Standard-Normal NCP raw effects (pre-centering).
    tau_raw : jnp.ndarray, scalar
        Unconstrained shared between-donor scale; ``softplus`` maps it to the
        positive ``tau_s``.

    Returns
    -------
    s : jnp.ndarray, shape ``(n_datasets, latent_dim)``
        Constrained relative per-donor program activity, strictly positive,
        with ``mean_d log s_{d,k} == 0`` per program.
    """
    tau_s = jnn.softplus(tau_raw)
    centered = raw - jnp.mean(raw, axis=0, keepdims=True)
    return jnp.exp(tau_s * centered)


def sample_program_scales(
    n_datasets: int,
    latent_dim: int,
    *,
    tau_loc: float = _DEFAULT_TAU_LOC,
    tau_scale: float = _DEFAULT_TAU_SCALE,
    site_prefix: str = "program_scale",
) -> jnp.ndarray:
    """Sample per-donor relative program-activity scales ``s_d`` (model side).

    Draws the hierarchical, non-centered, sum-to-zero block described in the
    module docstring and returns the constrained scales
    ``s in R_+^{D x K}``. This is the **model-side** sampler; the matching
    **guide-side** variational block is :func:`guide_program_scales`, which
    must register guide sites under the *same* ``site_prefix`` for the
    mean-field ELBO to pair correctly.

    Sampled NumPyro sites (all prefixed by ``site_prefix``)
    -------------------------------------------------------
    ``{prefix}_tau_raw`` : scalar
        Unconstrained Normal draw; ``tau_s = softplus(tau_raw)`` is the shared
        between-donor scale.
    ``{prefix}_raw`` : shape ``(n_datasets, latent_dim)``
        Standard-Normal NCP raw effects (``to_event(2)``).

    Deterministic NumPyro sites
    ---------------------------
    ``{prefix}_log`` : shape ``(n_datasets, latent_dim)``
        ``log s_{d,k}`` after the sum-to-zero centering and ``tau_s`` scaling.
    ``{prefix}`` : shape ``(n_datasets, latent_dim)``
        The constrained scales ``s_{d,k} = exp(log s_{d,k})``.

    Parameters
    ----------
    n_datasets : int
        Number of donors / datasets ``D`` (the leaf count of the grouping).
        Must be ``>= 1``; with ``D == 1`` the sum-to-zero centering forces
        ``s == 1`` identically (no hierarchy), which is the correct
        degenerate behavior.
    latent_dim : int
        Number of latent regulatory programs ``K`` (the rank of ``W``).
    tau_loc : float, optional
        Location of the Normal whose softplus is the shared between-donor
        scale ``tau_s``. Default ``-2.0`` (prior median ``tau_s ~ 0.127``).
    tau_scale : float, optional
        Scale (std) of that Normal. Default ``0.5``.
    site_prefix : str, optional
        Prefix for every sampled/deterministic site name. Default
        ``"program_scale"``. Must match the value passed to
        :func:`guide_program_scales`.

    Returns
    -------
    s : jnp.ndarray, shape ``(n_datasets, latent_dim)``
        Per-donor relative program-activity scales, strictly positive, with
        ``mean_d log s_{d, k} == 0`` for every program ``k`` (up to float
        round-off).

    Notes
    -----
    The returned ``s`` satisfies the scale gauge by construction. Callers that
    need *effective loadings* ``W_eff,d = W . diag(s_d)`` should pass ``s`` to
    :func:`effective_loadings` rather than re-deriving the broadcast.
    """
    # --- Shared between-donor scale tau_s (one scalar for all K programs) ----
    # Sample the unconstrained raw; ``program_scales_from_raw`` applies the
    # softplus. We keep the raw as the latent site (rather than a
    # TransformedDistribution) so the guide can place a simple mean-field
    # Normal on it -- the cleanest pairing for the hand-written guide block
    # used by the NBLN VAE path.
    tau_raw = numpyro.sample(
        f"{site_prefix}{_TAU_RAW_SUFFIX}",
        dist.Normal(tau_loc, tau_scale),
    )

    # --- Non-centered raw effects, shape (D, K) ------------------------------
    # ``to_event(2)`` marks both the donor and program axes as event dims so
    # this is a single global latent (no implicit plate broadcasting).
    eps = numpyro.sample(
        f"{site_prefix}{_RAW_SUFFIX}",
        dist.Normal(0.0, 1.0).expand([n_datasets, latent_dim]).to_event(2),
    )

    # --- Assemble constrained scales via the shared raw->s transform ---------
    # ``program_scales_from_raw`` owns the sum-to-zero gauge + softplus + exp,
    # so SVI here and the Laplace obs model use the identical mapping. We pass
    # ``tau_raw`` (pre-softplus) to keep the gauge in one place; recompute
    # ``log s`` only for the deterministic display site.
    s = program_scales_from_raw(eps, tau_raw)
    numpyro.deterministic(f"{site_prefix}{_LOG_SUFFIX}", jnp.log(s))
    numpyro.deterministic(site_prefix, s)
    return s


def guide_program_scales(
    n_datasets: int,
    latent_dim: int,
    *,
    tau_loc: float = _DEFAULT_TAU_LOC,
    tau_scale: float = _DEFAULT_TAU_SCALE,
    site_prefix: str = "program_scale",
) -> None:
    """Mean-field variational block for :func:`sample_program_scales`.

    Registers learnable ``numpyro.param`` location/scale pairs and the matching
    ``numpyro.sample`` sites for the two *latent* sites emitted by the model
    sampler (``{prefix}_tau_raw`` and ``{prefix}_raw``). The deterministic
    sites (``{prefix}_log`` and ``{prefix}``) are functions of these and so
    require no guide entry.

    Parameters
    ----------
    n_datasets : int
        Number of donors / datasets ``D`` (must match the model sampler).
    latent_dim : int
        Number of latent regulatory programs ``K`` (must match the sampler).
    tau_loc : float, optional
        Initial location for the ``tau_raw`` variational Normal. Default
        ``-2.0`` (matches the model prior center, a sensible warm start).
    tau_scale : float, optional
        Initial scale for the ``tau_raw`` variational Normal. Default ``0.5``.
    site_prefix : str, optional
        Prefix for every guide site name. Must equal the value passed to
        :func:`sample_program_scales`. Default ``"program_scale"``.

    Returns
    -------
    None
        Side-effecting: registers guide sites in the active NumPyro trace.

    Notes
    -----
    Variational scales are parameterized through softplus of an unconstrained
    ``numpyro.param`` so they stay strictly positive without a constraint
    object, matching the convention used elsewhere in the guide builder.
    """
    # --- tau_raw: scalar mean-field Normal -----------------------------------
    tau_loc_q = numpyro.param(f"{site_prefix}{_TAU_RAW_SUFFIX}_loc", tau_loc)
    # Unconstrained scale param -> softplus to keep it positive. Initialize the
    # unconstrained value so softplus(.) == tau_scale at step 0.
    tau_scale_raw_q = numpyro.param(
        f"{site_prefix}{_TAU_RAW_SUFFIX}_scale_raw",
        _inv_softplus(tau_scale),
    )
    numpyro.sample(
        f"{site_prefix}{_TAU_RAW_SUFFIX}",
        dist.Normal(tau_loc_q, jnn.softplus(tau_scale_raw_q)),
    )

    # --- raw effects: (D, K) mean-field Normal -------------------------------
    raw_loc_q = numpyro.param(
        f"{site_prefix}{_RAW_SUFFIX}_loc",
        jnp.zeros((n_datasets, latent_dim)),
    )
    raw_scale_raw_q = numpyro.param(
        f"{site_prefix}{_RAW_SUFFIX}_scale_raw",
        jnp.full((n_datasets, latent_dim), _inv_softplus(1.0)),
    )
    numpyro.sample(
        f"{site_prefix}{_RAW_SUFFIX}",
        dist.Normal(raw_loc_q, jnn.softplus(raw_scale_raw_q)).to_event(2),
    )


def program_scales_active(model_config, dataset_indices) -> bool:
    """Whether the per-donor program-scale hierarchy is active for this fit.

    Both the model builder and the guide builder **must** gate the
    program-scale block on this identical condition. If they disagree, the
    mean-field ELBO sees a latent site present in one trace but not the other
    and raises. Centralizing the predicate here makes drift impossible.

    The check is *necessary but not sufficient* on its own: each builder must
    additionally confirm that a VAE latent spec is present (the ``s_d`` block
    only makes sense for the low-rank correlation models, which use the VAE
    encoder/decoder). Because the model's decoder spec and the guide's encoder
    spec are introduced together, that extra condition is symmetric across the
    two builders.

    Parameters
    ----------
    model_config : ModelConfig
        The resolved model configuration. Read for ``correlation_hierarchy``
        (must equal ``"program_scales"``) and ``n_datasets`` (must be ``>= 2``).
    dataset_indices : jnp.ndarray or None
        Per-cell donor/leaf index array. Must be non-``None`` (a grouped fit).

    Returns
    -------
    bool
        ``True`` iff the per-donor program-scale hierarchy should be wired in.
    """
    if getattr(model_config, "correlation_hierarchy", None) != "program_scales":
        return False
    n_datasets = getattr(model_config, "n_datasets", None) or 0
    if n_datasets < 2:
        # With fewer than two donors there is no between-donor structure to
        # share; the sum-to-zero gauge would force s == 1 identically anyway.
        return False
    if dataset_indices is None:
        return False
    return True


def effective_loadings(W: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Fold per-donor program scales into effective low-rank loadings.

    Computes ``W_eff,d = W . diag(s_d)`` -- i.e. scales column ``k`` of the
    shared loadings ``W`` by donor ``d``'s program activity ``s_{d,k}``. This
    is the algebraic identity that keeps the hierarchical covariance in
    low-rank-plus-diagonal form:

    .. math::

        \\Sigma_d = W \\, \\mathrm{diag}(s_d^2) \\, W^T + \\mathrm{diag}(d)
                  = W_{\\mathrm{eff},d} W_{\\mathrm{eff},d}^T + \\mathrm{diag}(d).

    so the existing Woodbury / Newton machinery is reused unchanged with
    ``W_eff,d`` in place of ``W``.

    Parameters
    ----------
    W : jnp.ndarray, shape ``(G, K)``
        Shared low-rank loadings (``G`` genes, ``K`` programs).
    s : jnp.ndarray, shape ``(K,)`` or ``(D, K)``
        Program-activity scales. A 1-D array is treated as a single donor and
        yields a ``(G, K)`` result; a 2-D ``(D, K)`` array yields the stacked
        ``(D, G, K)`` effective loadings, one ``(G, K)`` slice per donor.

    Returns
    -------
    W_eff : jnp.ndarray
        Effective loadings. Shape ``(G, K)`` if ``s`` is 1-D, else
        ``(D, G, K)``.

    Raises
    ------
    ValueError
        If ``s`` is neither 1-D nor 2-D, or its trailing dimension does not
        match ``W``'s program axis ``K``.
    """
    if W.ndim != 2:
        raise ValueError(
            f"W must be 2-D (G, K); got shape {tuple(W.shape)}."
        )
    K = W.shape[1]
    if s.ndim == 1:
        # Single-donor case: (G, K) * (1, K) -> (G, K).
        if s.shape[0] != K:
            raise ValueError(
                f"s has length {s.shape[0]} but W has K={K} programs."
            )
        return W * s[None, :]
    if s.ndim == 2:
        # Multi-donor case: (1, G, K) * (D, 1, K) -> (D, G, K).
        if s.shape[1] != K:
            raise ValueError(
                f"s has trailing dim {s.shape[1]} but W has K={K} programs."
            )
        return W[None, :, :] * s[:, None, :]
    raise ValueError(
        f"s must be 1-D (K,) or 2-D (D, K); got ndim={s.ndim}."
    )


def _inv_softplus(y: float) -> float:
    """Inverse of ``softplus`` for initializing unconstrained scale params.

    ``softplus(x) = log(1 + exp(x))``; its inverse is
    ``x = log(exp(y) - 1)``. Used so a guide scale param initialized at
    ``_inv_softplus(sigma)`` yields ``softplus(.) == sigma`` at step 0.

    Implemented with the pure-Python :mod:`math` module (not ``jnp``) on
    purpose: this runs at ``numpyro.param`` init time *inside the JIT-traced
    guide*, where ``jnp`` ops on a Python constant produce a tracer and
    ``float(...)`` would raise ``ConcretizationTypeError``. The input is always
    a concrete Python float (a default scale), so plain ``math`` is exact and
    trace-safe.

    Parameters
    ----------
    y : float
        Target positive value (a standard deviation). Must be ``> 0``.

    Returns
    -------
    float
        The unconstrained pre-softplus value ``x`` with ``softplus(x) == y``.
    """
    # ``expm1`` is the numerically stable ``exp(y) - 1`` for small ``y``.
    return math.log(math.expm1(y))
