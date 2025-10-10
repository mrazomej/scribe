"""Divergence and distance functions for probability distributions."""

import jax.numpy as jnp
from jax import scipy as jsp

from numpyro.distributions import Beta, Normal, LogNormal
from numpyro.distributions.kl import kl_divergence
from multipledispatch import dispatch

from .distributions import BetaPrime

# ==============================================================================
# KL Divergence Implementations
# ==============================================================================


@kl_divergence.register(BetaPrime, BetaPrime)
def _kl_betaprime(p, q):
    """
    Compute the KL divergence between two BetaPrime distributions by leveraging
    the relationship between the BetaPrime and Beta distributions.

    Mathematically, the BetaPrime(α, β) distribution is the distribution of the
    random variable φ = X / (1 - X), where X ~ Beta(β, α). The KL divergence
    between two BetaPrime distributions, KL(P || Q), is equal to the KL
    divergence between the corresponding Beta distributions with swapped
    parameters:

    KL(BetaPrime(α₁, β₁) || BetaPrime(α₂, β₂)) = KL(Beta(β₁, α₁) || Beta(β₂, α₂))

    This is because the transformation from Beta to BetaPrime is invertible and
    monotonic, and the KL divergence is invariant under such transformations.
    Therefore, we can compute the KL divergence between BetaPrime distributions
    by calling the KL divergence for Beta distributions with the appropriate
    parameters.

    Parameters
    ----------
    p : BetaPrime
        The first BetaPrime distribution.
    q : BetaPrime
        The second BetaPrime distribution.

    Returns
    -------
    float
        The KL divergence KL(p || q).
    """
    a1, b1 = p.concentration1, p.concentration0  # user-facing α, β
    a2, b2 = q.concentration1, q.concentration0
    return kl_divergence(Beta(a1, b1), Beta(a2, b2))


# ------------------------------------------------------------------------------


@kl_divergence.register(LogNormal, LogNormal)
def _kl_lognormal(p, q):
    """
    Compute the KL divergence between two LogNormal distributions by leveraging
    the invariance of KL divergence under invertible, differentiable
    transformations.

    The LogNormal(μ, σ) distribution is the distribution of exp(X), where X ~
    Normal(μ, σ). Since the transformation x ↦ exp(x) is invertible and
    differentiable, the KL divergence between two LogNormal distributions is
    equal to the KL divergence between their underlying Normal distributions:

        KL(LogNormal(μ₁, σ₁) || LogNormal(μ₂, σ₂)) = KL(Normal(μ₁, σ₁) ||
        Normal(μ₂, σ₂))

    This property allows us to compute the KL divergence for LogNormal
    distributions by simply calling the KL divergence for the corresponding
    Normal distributions.

    Parameters
    ----------
    p : LogNormal
        The first LogNormal distribution.
    q : LogNormal
        The second LogNormal distribution.

    Returns
    -------
    float
        The KL divergence KL(p || q).
    """
    loc1, scale1 = p.loc, p.scale
    loc2, scale2 = q.loc, q.scale
    return kl_divergence(Normal(loc1, scale1), Normal(loc2, scale2))


# ==============================================================================
# Jensen-Shannon Divergence
# ==============================================================================


@dispatch(Beta, Beta)
def jensen_shannon(p, q):
    # Define distributions
    p = Beta(p.concentration1, p.concentration0)
    q = Beta(q.concentration1, q.concentration0)
    # Compute KL divergences
    kl_p_q = kl_divergence(p, q)
    kl_q_p = kl_divergence(q, p)
    # Compute Jensen-Shannon divergence
    return 0.5 * (kl_p_q + kl_q_p)


# ------------------------------------------------------------------------------


@dispatch(BetaPrime, BetaPrime)
def jensen_shannon(p, q):
    # Define distributions
    p = BetaPrime(p.concentration1, p.concentration0)
    q = BetaPrime(q.concentration1, q.concentration0)
    # Compute KL divergences
    kl_p_q = kl_divergence(p, q)
    kl_q_p = kl_divergence(q, p)
    # Compute Jensen-Shannon divergence
    return 0.5 * (kl_p_q + kl_q_p)


# ------------------------------------------------------------------------------


@dispatch(Normal, Normal)
def jensen_shannon(p, q):
    # Define distributions
    p = Normal(p.loc, p.scale)
    q = Normal(q.loc, q.scale)
    # Compute KL divergences
    kl_p_q = kl_divergence(p, q)
    kl_q_p = kl_divergence(q, p)
    # Compute Jensen-Shannon divergence
    return 0.5 * (kl_p_q + kl_q_p)


# ------------------------------------------------------------------------------


@dispatch(LogNormal, LogNormal)
def jensen_shannon(p, q):
    # Define distributions
    p = LogNormal(p.loc, p.scale)
    q = LogNormal(q.loc, q.scale)
    # Compute KL divergences
    kl_p_q = kl_divergence(p, q)
    kl_q_p = kl_divergence(q, p)
    # Compute Jensen-Shannon divergence
    return 0.5 * (kl_p_q + kl_q_p)


# ==============================================================================
# Hellinger Distance
# ==============================================================================


@dispatch(Beta, Beta)
def sq_hellinger(p, q):
    """
    Compute the squared Hellinger distance between two Beta distributions.

    H²(P,Q) = 1 - B((α₁+α₂)/2, (β₁+β₂)/2) / sqrt(B(α₁,β₁) * B(α₂,β₂))
    where B(x,y) is the beta function.
    """
    return 1 - (
        jsp.special.beta(
            (p.concentration1 + q.concentration1) / 2,
            (p.concentration0 + q.concentration0) / 2,
        )
        / jnp.sqrt(
            jsp.special.beta(p.concentration1, p.concentration0)
            * jsp.special.beta(q.concentration1, q.concentration0)
        )
    )


# ------------------------------------------------------------------------------


@dispatch(Beta, Beta)
def hellinger(p, q):
    """Compute the Hellinger distance between two Beta distributions."""
    return jnp.sqrt(sq_hellinger(p, q))


# ------------------------------------------------------------------------------


@dispatch(BetaPrime, BetaPrime)
def sq_hellinger(p, q):
    """
    Compute the squared Hellinger distance between two BetaPrime distributions.

    For BetaPrime distributions, we can use the relationship with Beta
    distributions and the fact that BetaPrime(α, β) corresponds to Beta(β, α) in
    standard form.
    """
    # Convert to Beta parameters for computation
    # BetaPrime(α, β) corresponds to Beta(β, α) in standard form
    return 1 - (
        jsp.special.beta(
            (p.concentration0 + q.concentration0) / 2,
            (p.concentration1 + q.concentration1) / 2,
        )
        / jnp.sqrt(
            jsp.special.beta(p.concentration0, p.concentration1)
            * jsp.special.beta(q.concentration0, q.concentration1)
        )
    )


# ------------------------------------------------------------------------------


@dispatch(BetaPrime, BetaPrime)
def hellinger(p, q):
    """Compute the Hellinger distance between two BetaPrime distributions."""
    return jnp.sqrt(sq_hellinger(p, q))


# ------------------------------------------------------------------------------


@dispatch(Normal, Normal)
def sq_hellinger(p, q):
    """
    Compute the squared Hellinger distance between two Normal distributions.

    H²(P,Q) = 1 - sqrt(2*σ₁*σ₂/(σ₁² + σ₂²)) * exp(-(μ₁-μ₂)²/(4*(σ₁² + σ₂²)))
    """
    mu1, sigma1 = p.loc, p.scale
    mu2, sigma2 = q.loc, q.scale

    # The prefactor under the square root:
    prefactor = jnp.sqrt((2 * sigma1 * sigma2) / (sigma1**2 + sigma2**2))
    # The exponent factor:
    exponent = jnp.exp(-((mu1 - mu2) ** 2) / (4.0 * (sigma1**2 + sigma2**2)))
    return 1.0 - (prefactor * exponent)


# ------------------------------------------------------------------------------


@dispatch(Normal, Normal)
def hellinger(p, q):
    """Compute the Hellinger distance between two Normal distributions."""
    return jnp.sqrt(sq_hellinger(p, q))


# ------------------------------------------------------------------------------


@dispatch(LogNormal, LogNormal)
def sq_hellinger(p, q):
    """
    Compute the squared Hellinger distance between two LogNormal distributions.

    H²(P,Q) = 1 - sqrt(σ₁*σ₂/(σ₁² + σ₂²)) * exp(-(μ₁-μ₂)²/(4*(σ₁² + σ₂²)))
    """
    mu1, sigma1 = p.loc, p.scale
    mu2, sigma2 = q.loc, q.scale

    # The prefactor under the square root:
    prefactor = jnp.sqrt((sigma1 * sigma2) / (sigma1**2 + sigma2**2))
    # The exponent factor:
    exponent = jnp.exp(-((mu1 - mu2) ** 2) / (4.0 * (sigma1**2 + sigma2**2)))
    return 1.0 - (prefactor * exponent)


# ------------------------------------------------------------------------------


@dispatch(LogNormal, LogNormal)
def hellinger(p, q):
    """Compute the Hellinger distance between two LogNormal distributions."""
    return jnp.sqrt(sq_hellinger(p, q))
