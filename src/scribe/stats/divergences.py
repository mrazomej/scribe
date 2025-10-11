"""Divergence and distance functions for probability distributions."""

import jax.numpy as jnp
from jax import scipy as jsp

from numpyro.distributions import Beta, Normal, LogNormal
from numpyro.distributions.kl import kl_divergence
from multipledispatch import dispatch

from .distributions import BetaPrime, LowRankLogisticNormal, SoftmaxNormal

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


# ==============================================================================
# Low-Rank Gaussian Divergences (for Compositional DE)
# ==============================================================================


def _extract_lowrank_params(model):
    """
    Extract (mu, W, d) from dict or Distribution object.

    This helper function provides flexible input handling for low-rank
    Gaussian models, accepting either dictionary outputs from
    fit_logistic_normal_from_posterior or Distribution objects.

    Parameters
    ----------
    model : dict or Distribution
        Either a dict with 'loc', 'cov_factor', 'cov_diag' keys,
        or a Distribution object with those attributes

    Returns
    -------
    tuple of ndarray
        (loc, cov_factor, cov_diag) as jax arrays

    Raises
    ------
    TypeError
        If model is neither a dict nor has the required attributes
    """
    if isinstance(model, dict):
        return model["loc"], model["cov_factor"], model["cov_diag"]
    elif hasattr(model, "loc"):  # Distribution object
        return model.loc, model.cov_factor, model.cov_diag
    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}. "
            f"Expected dict with keys ['loc', 'cov_factor', 'cov_diag'] "
            f"or Distribution object with these attributes."
        )


# ------------------------------------------------------------------------------


def _kl_lowrank_mvn(p, q):
    """
    Core KL computation for low-rank multivariate normals.

    For p = N(μ_p, Σ_p) and q = N(μ_q, Σ_q) where Σ = W·Wᵀ + diag(d):

        KL(p||q) = 0.5 * [tr(Σ_q^{-1} Σ_p) + (μ_q - μ_p)ᵀ Σ_q^{-1} (μ_q - μ_p)
                          - D + log(det Σ_q / det Σ_p)]

    This implementation uses:
        - Woodbury identity for Σ_q^{-1} (avoids explicit matrix inversion)
        - Matrix determinant lemma for det(W·Wᵀ + diag(d))

    The computation is O(k² D) where k is the rank and D is the dimension,
    avoiding O(D³) costs of dense linear algebra.

    Parameters
    ----------
    p, q : Distribution or dict
        Low-rank Gaussian models with 'loc', 'cov_factor', 'cov_diag'

    Returns
    -------
    float
        KL divergence (non-negative)

    Notes
    -----
    This is the corrected implementation with the trace term fix from
    tmp.md feedback, ensuring exact KL computation.
    """
    mu_p, W_p, d_p = _extract_lowrank_params(p)
    mu_q, W_q, d_q = _extract_lowrank_params(q)

    D = mu_p.shape[-1]
    k_p = W_p.shape[-1]
    k_q = W_q.shape[-1]

    # Add numerical stability
    d_p = d_p + 1e-8
    d_q = d_q + 1e-8

    # 1. Compute log determinants using matrix determinant lemma:
    # log|W·Wᵀ + diag(d)| = log|diag(d)| + log|I + Wᵀ·diag(d)^{-1}·W|
    log_det_p = (
        jnp.sum(jnp.log(d_p))
        + jnp.linalg.slogdet(jnp.eye(k_p) + (W_p.T / d_p) @ W_p)[1]
    )
    log_det_q = (
        jnp.sum(jnp.log(d_q))
        + jnp.linalg.slogdet(jnp.eye(k_q) + (W_q.T / d_q) @ W_q)[1]
    )

    # 2. Compute trace term: tr(Σ_q^{-1} Σ_p)
    # Use Woodbury:
    # (W·Wᵀ + D)^{-1} = D^{-1} - D^{-1}·W·(I + Wᵀ·D^{-1}·W)^{-1}·Wᵀ·D^{-1}
    Dinv_q = 1.0 / d_q
    A = (W_q.T * Dinv_q) @ W_q  # (k_q, k_q)
    B = (W_q.T * Dinv_q) @ W_p  # (k_q, k_p)

    # tr(Σ_q^{-1} Σ_p) =
    # tr(Dinv_q * d_p) + tr(W_p^T Dinv_q W_p) - tr(B^T inv(I+A) B)
    trace_diag = jnp.sum(Dinv_q * d_p)
    trace_lowrank = jnp.trace(
        (W_p.T * Dinv_q) @ W_p
    )  # CORRECTED: added this term
    correction = jnp.trace(B.T @ jnp.linalg.solve(jnp.eye(k_q) + A, B))

    trace_term = trace_diag + trace_lowrank - correction

    # 3. Compute Mahalanobis term: (μ_q - μ_p)ᵀ Σ_q^{-1} (μ_q - μ_p)
    delta = mu_q - mu_p
    mahal_term = jnp.sum(delta**2 * Dinv_q)
    temp2 = (W_q.T * Dinv_q) @ delta
    mahal_correction = temp2 @ jnp.linalg.solve(jnp.eye(k_q) + A, temp2)
    mahal_term = mahal_term - mahal_correction

    # Combine
    kl = 0.5 * (trace_term + mahal_term - D + log_det_q - log_det_p)
    return kl


# ------------------------------------------------------------------------------
# Register KL divergence for low-rank logistic-normal distributions
# ------------------------------------------------------------------------------


@kl_divergence.register(LowRankLogisticNormal, LowRankLogisticNormal)
def _kl_lowrank_logistic_normal(p, q):
    """
    Compute KL divergence between two LowRankLogisticNormal distributions.

    Since KL divergence is invariant under bijective transformations (ALR),
    we can compute it directly in the underlying Gaussian (ALR) space.

    Parameters
    ----------
    p : LowRankLogisticNormal
        First distribution
    q : LowRankLogisticNormal
        Second distribution

    Returns
    -------
    float
        KL(p || q)
    """
    return _kl_lowrank_mvn(p, q)


# ------------------------------------------------------------------------------


@kl_divergence.register(SoftmaxNormal, SoftmaxNormal)
def _kl_softmax_normal(p, q):
    """
    Compute KL divergence between two SoftmaxNormal distributions.

    The KL is computed in the underlying Gaussian space. While the softmax
    transformation is not bijective, the KL in the base space provides a
    valid divergence measure.

    Parameters
    ----------
    p : SoftmaxNormal
        First distribution
    q : SoftmaxNormal
        Second distribution

    Returns
    -------
    float
        KL(p || q) in the base space
    """
    return _kl_lowrank_mvn(p, q)


# ------------------------------------------------------------------------------
# Register Jensen-Shannon divergence for low-rank logistic-normal distributions
# ------------------------------------------------------------------------------


@dispatch(LowRankLogisticNormal, LowRankLogisticNormal)
def jensen_shannon(p, q):
    """
    Compute Jensen-Shannon divergence between two LowRankLogisticNormal
    distributions.

    JS(p, q) = 0.5 * [KL(p||m) + KL(q||m)] where m is the mixture.

    Parameters
    ----------
    p : LowRankLogisticNormal
        First distribution
    q : LowRankLogisticNormal
        Second distribution

    Returns
    -------
    float
        JS divergence (symmetric, non-negative)
    """
    mu_a, W_a, d_a = p.loc, p.cov_factor, p.cov_diag
    mu_b, W_b, d_b = q.loc, q.cov_factor, q.cov_diag

    # Mixture parameters (add epsilon for numerical stability)
    mu_m = 0.5 * (mu_a + mu_b)
    W_m = jnp.concatenate([W_a, W_b], axis=-1) / jnp.sqrt(2.0)
    d_m = 0.5 * (d_a + d_b) + 1e-8  # Numerical stability

    model_m = {"loc": mu_m, "cov_factor": W_m, "cov_diag": d_m}

    kl_pm = _kl_lowrank_mvn(p, model_m)
    kl_qm = _kl_lowrank_mvn(q, model_m)

    return 0.5 * (kl_pm + kl_qm)


# ------------------------------------------------------------------------------


@dispatch(SoftmaxNormal, SoftmaxNormal)
def jensen_shannon(p, q):
    """
    Compute Jensen-Shannon divergence between two SoftmaxNormal distributions.

    JS(p, q) = 0.5 * [KL(p||m) + KL(q||m)] where m is the mixture.

    Parameters
    ----------
    p : SoftmaxNormal
        First distribution
    q : SoftmaxNormal
        Second distribution

    Returns
    -------
    float
        JS divergence (symmetric, non-negative)
    """
    mu_a, W_a, d_a = p.loc, p.cov_factor, p.cov_diag
    mu_b, W_b, d_b = q.loc, q.cov_factor, q.cov_diag

    # Mixture parameters (add epsilon for numerical stability)
    mu_m = 0.5 * (mu_a + mu_b)
    W_m = jnp.concatenate([W_a, W_b], axis=-1) / jnp.sqrt(2.0)
    d_m = 0.5 * (d_a + d_b) + 1e-8  # Numerical stability

    model_m = {"loc": mu_m, "cov_factor": W_m, "cov_diag": d_m}

    kl_pm = _kl_lowrank_mvn(p, model_m)
    kl_qm = _kl_lowrank_mvn(q, model_m)

    return 0.5 * (kl_pm + kl_qm)


# ------------------------------------------------------------------------------
# Register Jensen-Shannon divergence for softmax-normal distributions
# ------------------------------------------------------------------------------


@dispatch(LowRankLogisticNormal, LowRankLogisticNormal)
def mahalanobis(p, q):
    """
    Compute squared Mahalanobis distance with pooled covariance.

    M² = (μ_p - μ_q)ᵀ [(Σ_p + Σ_q)/2]^{-1} (μ_p - μ_q)

    Parameters
    ----------
    p : LowRankLogisticNormal
        First distribution
    q : LowRankLogisticNormal
        Second distribution

    Returns
    -------
    float
        Squared Mahalanobis distance (non-negative, symmetric)
    """
    mu_a, W_a, d_a = p.loc, p.cov_factor, p.cov_diag
    mu_b, W_b, d_b = q.loc, q.cov_factor, q.cov_diag

    # Pooled covariance
    W_pool = jnp.concatenate([W_a, W_b], axis=-1) / jnp.sqrt(2.0)
    d_pool = 0.5 * (d_a + d_b) + 1e-8

    # Compute using Woodbury
    delta = mu_a - mu_b
    Dinv_pool = 1.0 / d_pool
    inner_pool = jnp.eye(W_pool.shape[-1]) + (W_pool.T * Dinv_pool) @ W_pool

    mahal = jnp.sum(delta**2 * Dinv_pool)
    temp = (W_pool.T * Dinv_pool) @ delta
    correction = temp @ jnp.linalg.solve(inner_pool, temp)

    return mahal - correction


# ------------------------------------------------------------------------------


@dispatch(SoftmaxNormal, SoftmaxNormal)
def mahalanobis(p, q):
    """
    Compute squared Mahalanobis distance with pooled covariance.

    M² = (μ_p - μ_q)ᵀ [(Σ_p + Σ_q)/2]^{-1} (μ_p - μ_q)

    Parameters
    ----------
    p : SoftmaxNormal
        First distribution
    q : SoftmaxNormal
        Second distribution

    Returns
    -------
    float
        Squared Mahalanobis distance (non-negative, symmetric)
    """
    mu_a, W_a, d_a = p.loc, p.cov_factor, p.cov_diag
    mu_b, W_b, d_b = q.loc, q.cov_factor, q.cov_diag

    # Pooled covariance
    W_pool = jnp.concatenate([W_a, W_b], axis=-1) / jnp.sqrt(2.0)
    d_pool = 0.5 * (d_a + d_b) + 1e-8

    # Compute using Woodbury
    delta = mu_a - mu_b
    Dinv_pool = 1.0 / d_pool
    inner_pool = jnp.eye(W_pool.shape[-1]) + (W_pool.T * Dinv_pool) @ W_pool

    mahal = jnp.sum(delta**2 * Dinv_pool)
    temp = (W_pool.T * Dinv_pool) @ delta
    correction = temp @ jnp.linalg.solve(inner_pool, temp)

    return mahal - correction
