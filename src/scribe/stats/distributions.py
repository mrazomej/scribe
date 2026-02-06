"""Custom probability distributions for SCRIBE."""

import jax.numpy as jnp
import jax.random as random
from jax import scipy as jsp

from numpyro.distributions import (
    Distribution,
    constraints,
    Gamma,
    LowRankMultivariateNormal,
)
from numpyro.distributions.util import promote_shapes, validate_sample

# ==============================================================================
# Beta Prime Distribution
# ==============================================================================


class BetaPrime(Distribution):
    """
    Beta Prime distribution (odds-of-Beta convention).

    Convention
    ----------
    If p ~ Beta(α, β) and φ = (1 - p) / p (odds of "success" 1 - p), then
    φ ~ BetaPrime(α, β) in THIS CLASS.

    Implementation detail
    ---------------------
    Mathematically, φ has the *standard* Beta-prime with swapped parameters:
        φ ~ BetaPrime_std(β, α).
    This class accepts (α, β) at the call site and internally uses (β, α),
    so that your models can pass (α, β) unchanged. This is necessary because the
    NumPyro NegativeBinomial distribution expects the `probs` parameter to be
    the *failure probability* p, so that the odds ratio φ = (1 - p) / p is
    consistent with the parameterization of the BetaPrime distribution.

    Density (with user parameters α, β)
    -----------------------------------
        f(φ; α, β) = φ^(β - 1) * (1 + φ)^(-(α + β)) / B(β, α),    φ > 0

    Note the Beta function arguments B(β, α).

    Parameters
    ----------
    concentration1 : jnp.ndarray
        α (matches the Beta prior's first shape)
    concentration0 : jnp.ndarray
        β (matches the Beta prior's second shape)
    """

    arg_constraints = {
        "concentration1": constraints.positive,  # α
        "concentration0": constraints.positive,  # β
    }
    support = constraints.positive
    has_rsample = False

    def __init__(self, concentration1, concentration0, validate_args=None):
        alpha = jnp.asarray(concentration1, dtype=jnp.float32)  # α
        beta = jnp.asarray(concentration0, dtype=jnp.float32)  # β
        # store user-facing α, β
        self.alpha, self.beta = promote_shapes(alpha, beta)
        # internal standard Beta-prime uses (a_std, b_std) = (β, α)
        self._a_std = self.beta
        self._b_std = self.alpha
        super().__init__(
            batch_shape=self.alpha.shape,
            event_shape=(),
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        key1, key2 = random.split(key)
        # φ = X / Y with X ~ Gamma(β, 1), Y ~ Gamma(α, 1)
        x = Gamma(self._a_std, 1.0).sample(key1, sample_shape)
        y = Gamma(self._b_std, 1.0).sample(key2, sample_shape)
        # Clamp to prevent 0/0 = NaN when shape params are very small
        # (Gamma with tiny shape can underflow to exactly 0.0 in float32)
        x = jnp.clip(x, 1e-30)
        y = jnp.clip(y, 1e-30)
        return x / y

    @validate_sample
    def log_prob(self, value):
        # log f(φ; α, β) = (β-1) log φ - (α+β) log(1+φ) - log B(β, α)
        # Clamp value to prevent log(0) = -inf when samples are near zero
        safe_value = jnp.clip(value, 1e-30)
        log_num = (self.beta - 1) * jnp.log(safe_value) - (
            self.alpha + self.beta
        ) * jnp.log1p(value)
        log_den = (
            jsp.special.gammaln(self._a_std)
            + jsp.special.gammaln(self._b_std)
            - jsp.special.gammaln(self._a_std + self._b_std)
        )
        return log_num - log_den

    @property
    def mean(self):
        # E[φ] = β / (α - 1), defined for α > 1
        return jnp.where(self.alpha > 1, self.beta / (self.alpha - 1), jnp.inf)

    @property
    def variance(self):
        # Var[φ] = β(β + α - 1) / [(α - 1)^2 (α - 2)], defined for α > 2
        return jnp.where(
            self.alpha > 2,
            self.beta
            * (self.beta + self.alpha - 1)
            / ((self.alpha - 1) ** 2 * (self.alpha - 2)),
            jnp.inf,
        )

    @property
    def mode(self):
        # mode = (β - 1) / (α + 1) for β >= 1; else 0
        return jnp.where(
            self.beta >= 1, (self.beta - 1) / (self.alpha + 1), 0.0
        )

    @property
    def concentration1(self):
        """Access to concentration1 parameter (α) for NumPyro compatibility."""
        return self.alpha

    @property
    def concentration0(self):
        """Access to concentration0 parameter (β) for NumPyro compatibility."""
        return self.beta


# ==============================================================================
# Low-Rank Compositional Distributions
# ==============================================================================


class LowRankLogisticNormal(Distribution):
    """
    Low-rank Logistic-Normal distribution for compositional data.

    This distribution models D-dimensional probability vectors (on the simplex)
    using a (D-1)-dimensional low-rank multivariate normal distribution in
    log-ratio space. It uses the Additive Log-Ratio (ALR) transformation.

    Mathematical Definition
    -----------------------
    Let y ∈ ℝ^(D-1) ~ MVN(μ, Σ) where Σ = WW^T + diag(D) is low-rank.
    The ALR transformation maps y to the simplex Δ^D:

        xᵢ = exp(yᵢ) / (1 + Σⱼ exp(yⱼ))   for i = 1, ..., D-1
        x_D = 1 / (1 + Σⱼ exp(yⱼ))

    The inverse transformation (simplex to log-ratio space) is:

        yᵢ = log(xᵢ / x_D)   for i = 1, ..., D-1

    The Jacobian of the transformation has log-determinant:

        log|det(J)| = -Σᵢ₌₁^D log(xᵢ)

    Low-Rank Covariance Structure
    ------------------------------
    The covariance matrix has the form:

        Σ = WW^T + diag(D)

    where:
    - W is a (D-1) × rank factor matrix
    - D is a (D-1) diagonal vector
    - Memory: O((D-1) × rank) vs O((D-1)²) for full covariance

    This is critical for large D (e.g., 30K+ genes) where storing full
    covariance is prohibitive.

    Asymmetry and Reference Component
    ----------------------------------
    The ALR transformation treats the last component (x_D) as a reference.
    This means the distribution is NOT symmetric under permutation of
    components. If you need symmetry, use SoftmaxNormal instead (but note
    that SoftmaxNormal cannot compute log_prob).

    Parameters
    ----------
    loc : jnp.ndarray
        Location parameter μ ∈ ℝ^(D-1) (mean in log-ratio space)
    cov_factor : jnp.ndarray
        Low-rank factor matrix W of shape (D-1, rank)
    cov_diag : jnp.ndarray
        Diagonal component D of shape (D-1,)
    validate_args : bool, optional
        Whether to validate input arguments

    Examples
    --------
    >>> from jax import random
    >>> import jax.numpy as jnp
    >>> # Create a low-rank logistic-normal for 5-dimensional simplex
    >>> D = 5
    >>> rank = 2
    >>> loc = jnp.zeros(D - 1)
    >>> cov_factor = jnp.ones((D - 1, rank)) * 0.1
    >>> cov_diag = jnp.ones(D - 1) * 0.5
    >>> dist = LowRankLogisticNormal(loc, cov_factor, cov_diag)
    >>> # Sample from the distribution (returns D-dimensional simplex points)
    >>> samples = dist.sample(random.PRNGKey(0), (100,))
    >>> samples.shape
    (100, 5)
    >>> # Samples sum to 1
    >>> jnp.allclose(samples.sum(axis=-1), 1.0)
    True
    >>> # Evaluate log probability
    >>> log_p = dist.log_prob(samples[0])

    References
    ----------
    Aitchison, J., & Shen, S. M. (1980). Logistic-normal distributions: Some
    properties and uses. Biometrika, 67(2), 261-272.

    Aitchison, J. (1986). The Statistical Analysis of Compositional Data.
    Chapman & Hall.

    See Also
    --------
    SoftmaxNormal : Symmetric alternative using softmax (no log_prob available)
    """

    # Define constraints
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.independent(constraints.real, 2),
        "cov_diag": constraints.positive,
    }
    support = constraints.simplex
    has_rsample = False

    def __init__(self, loc, cov_factor, cov_diag, validate_args=None):
        loc = jnp.asarray(loc)
        cov_factor = jnp.asarray(cov_factor)
        cov_diag = jnp.asarray(cov_diag)

        # Store parameters
        self.loc = loc
        self.cov_factor = cov_factor
        self.cov_diag = cov_diag

        # Infer dimensions
        self.n_dims_unconstrained = loc.shape[-1]  # D - 1
        self.n_dims_simplex = self.n_dims_unconstrained + 1  # D

        # Create internal low-rank multivariate normal
        self.base_dist = LowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
        )

        super().__init__(
            batch_shape=self.base_dist.batch_shape,
            event_shape=(self.n_dims_simplex,),
            validate_args=validate_args,
        )

    # --------------------------------------------------------------------------

    def sample(self, key, sample_shape=()):
        """
        Sample from the distribution.

        Returns samples on the D-dimensional simplex.

        Parameters
        ----------
        key : random.PRNGKey
            JAX random key
        sample_shape : tuple, optional
            Shape of samples to draw

        Returns
        -------
        jnp.ndarray
            Samples of shape (*sample_shape, *batch_shape, D) on the simplex
        """
        # Sample from base MVN in log-ratio space
        y = self.base_dist.sample(key, sample_shape)  # (..., D-1)

        # Apply ALR transformation
        exp_y = jnp.exp(y)
        sum_exp = 1.0 + jnp.sum(exp_y, axis=-1, keepdims=True)

        # Compute simplex coordinates
        x_first = exp_y / sum_exp  # (..., D-1)
        x_last = 1.0 / sum_exp  # (..., 1)

        return jnp.concatenate([x_first, x_last], axis=-1)  # (..., D)

    # --------------------------------------------------------------------------

    @validate_sample
    def log_prob(self, value):
        """
        Evaluate log probability density.

        Parameters
        ----------
        value : jnp.ndarray
            Points on the simplex of shape (..., D)

        Returns
        -------
        jnp.ndarray
            Log probability density of shape (...)
        """
        # Inverse ALR: x -> y
        # yᵢ = log(xᵢ / x_D)
        x_first = value[..., :-1]  # (..., D-1)
        x_last = value[..., -1:]  # (..., 1)

        # Compute log-ratio coordinates (avoid log(0))
        y = jnp.log(x_first + 1e-20) - jnp.log(x_last + 1e-20)  # (..., D-1)

        # Log probability in unconstrained space
        log_prob_unconstrained = self.base_dist.log_prob(y)  # (...)

        # Jacobian correction: log|det(J)| = -Σᵢ log(xᵢ)
        log_jacob = -jnp.sum(jnp.log(value + 1e-20), axis=-1)  # (...)

        return log_prob_unconstrained + log_jacob

    # --------------------------------------------------------------------------

    @property
    def mean(self):
        """
        Mean of the distribution on the simplex.

        Note: This is an approximation computed via Monte Carlo sampling,
        as the true mean does not have a closed form.

        Returns
        -------
        jnp.ndarray
            Approximate mean of shape (*batch_shape, D)
        """
        # Approximate mean via sampling
        key = random.PRNGKey(0)
        samples = self.sample(key, (10000,))
        return jnp.mean(samples, axis=0)


# ==============================================================================
# Softmax-Normal Distribution
# ==============================================================================


class SoftmaxNormal(Distribution):
    """
    Softmax-Normal distribution for compositional data (symmetric).

    This distribution models D-dimensional probability vectors (on the simplex)
    using a D-dimensional low-rank multivariate normal distribution with a
    softmax transformation. Unlike LowRankLogisticNormal (which uses ALR),
    this treats all components symmetrically.

    Mathematical Definition
    -----------------------
    Let y ∈ ℝ^D ~ MVN(μ, Σ) where Σ = WW^T + diag(D) is low-rank.
    The softmax transformation maps y to the simplex Δ^D:

        xᵢ = exp(yᵢ) / Σⱼ exp(yⱼ)   for i = 1, ..., D

    Symmetry and Invariance
    -----------------------
    The softmax transformation is:
    - Symmetric: All components treated equally (no reference component)
    - Translation-invariant: softmax(y + c·1) = softmax(y) for any constant c

    This translation invariance means the transformation is SINGULAR - you
    cannot uniquely invert it. Therefore, log_prob() is not available.

    Low-Rank Covariance Structure
    ------------------------------
    The covariance matrix has the form:

        Σ = WW^T + diag(D)

    where:
        - W is a D × rank factor matrix
        - D is a D-dimensional diagonal vector
        - Memory: O(D × rank) vs O(D²) for full covariance

    When to Use
    -----------
    Use SoftmaxNormal when:
        - You want symmetric treatment of all components
        - You only need sampling (not log_prob evaluation)
        - You're summarizing/visualizing posterior distributions

    Use LowRankLogisticNormal when:
        - You need to evaluate log_prob() for observed data
        - You're using the distribution as a likelihood in Bayesian inference
        - Asymmetry (reference component) is acceptable

    Parameters
    ----------
    loc : jnp.ndarray
        Location parameter μ ∈ ℝ^D (mean in log-space)
    cov_factor : jnp.ndarray
        Low-rank factor matrix W of shape (D, rank)
    cov_diag : jnp.ndarray
        Diagonal component D of shape (D,)
    validate_args : bool, optional
        Whether to validate input arguments

    Examples
    --------
    >>> from jax import random
    >>> import jax.numpy as jnp
    >>> # Create a softmax-normal for 5-dimensional simplex
    >>> D = 5
    >>> rank = 2
    >>> loc = jnp.zeros(D)
    >>> cov_factor = jnp.ones((D, rank)) * 0.1
    >>> cov_diag = jnp.ones(D) * 0.5
    >>> dist = SoftmaxNormal(loc, cov_factor, cov_diag)
    >>> # Sample from the distribution
    >>> samples = dist.sample(random.PRNGKey(0), (100,))
    >>> samples.shape
    (100, 5)
    >>> # Samples sum to 1
    >>> jnp.allclose(samples.sum(axis=-1), 1.0)
    True
    >>> # Access underlying log-space distribution
    >>> log_samples = dist.base_dist.sample(random.PRNGKey(1), (100,))
    >>> # Apply softmax manually
    >>> manual_samples = jax.nn.softmax(log_samples, axis=-1)

    See Also
    --------
    LowRankLogisticNormal : ALR-based alternative with log_prob() available
    """

    # Define constraints
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.independent(constraints.real, 2),
        "cov_diag": constraints.positive,
    }
    support = constraints.simplex
    has_rsample = False

    def __init__(self, loc, cov_factor, cov_diag, validate_args=None):
        loc = jnp.asarray(loc)
        cov_factor = jnp.asarray(cov_factor)
        cov_diag = jnp.asarray(cov_diag)

        # Store parameters
        self.loc = loc
        self.cov_factor = cov_factor
        self.cov_diag = cov_diag

        # Infer dimension
        self.n_dims = loc.shape[-1]  # D

        # Create internal low-rank multivariate normal
        self.base_dist = LowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
        )

        super().__init__(
            batch_shape=self.base_dist.batch_shape,
            event_shape=(self.n_dims,),
            validate_args=validate_args,
        )

    # --------------------------------------------------------------------------

    def sample(self, key, sample_shape=()):
        """
        Sample from the distribution.

        Returns samples on the D-dimensional simplex.

        Parameters
        ----------
        key : random.PRNGKey
            JAX random key
        sample_shape : tuple, optional
            Shape of samples to draw

        Returns
        -------
        jnp.ndarray
            Samples of shape (*sample_shape, *batch_shape, D) on the simplex
        """
        # Sample from base MVN in log-space
        y = self.base_dist.sample(key, sample_shape)  # (..., D)

        # Apply softmax transformation
        exp_y = jnp.exp(y)
        x = exp_y / jnp.sum(exp_y, axis=-1, keepdims=True)

        return x

    # --------------------------------------------------------------------------

    def log_prob(self, value):
        """
        Evaluate log probability density.

        NOT IMPLEMENTED: The softmax transformation is singular (adding a
        constant to all log-space coordinates doesn't change the output),
        so the Jacobian determinant is zero and log_prob is undefined.

        Parameters
        ----------
        value : jnp.ndarray
            Points on the simplex

        Raises
        ------
        NotImplementedError
            Always raised. Use LowRankLogisticNormal if you need log_prob(),
            or access base_dist.log_prob() for log-space density.
        """
        raise NotImplementedError(
            "log_prob() is not available for SoftmaxNormal because the softmax "
            "transformation is singular (adding a constant c to all components "
            "of y doesn't change softmax(y)). "
            "\n\nOptions:"
            "\n  1. Use LowRankLogisticNormal if you need log_prob() on the "
            "simplex (note: it uses ALR so the last component is a reference)"
            "\n  2. Use self.base_dist.log_prob() to evaluate density in "
            "log-space (but this doesn't account for the softmax transform)"
        )

    # --------------------------------------------------------------------------

    @property
    def mean(self):
        """
        Mean of the distribution on the simplex.

        Note: This is an approximation computed via Monte Carlo sampling,
        as the true mean does not have a closed form.

        Returns
        -------
        jnp.ndarray
            Approximate mean of shape (*batch_shape, D)
        """
        # Approximate mean via sampling
        key = random.PRNGKey(0)
        samples = self.sample(key, (10000,))
        return jnp.mean(samples, axis=0)
