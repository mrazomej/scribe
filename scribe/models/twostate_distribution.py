"""
Beta-Poisson distribution implementation for NumPyro.

This module implements the Beta-Poisson compound distribution, which is
mathematically equivalent to the two-state promoter model.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsps
from jax.scipy.special import logsumexp
from jax import random
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.distributions.util import (
    promote_shapes,
    validate_sample,
    is_prng_key,
)

# ------------------------------------------------------------------------------
# Beta-Poisson distribution
# ------------------------------------------------------------------------------


class BetaPoisson(dist.Distribution):
    """
    Compound distribution comprising of a Beta-Poisson pair.

    The rate parameter λ = d*p for the Poisson distribution is constructed by:
        1. Sampling p from Beta(α, β)
        2. Scaling by dose parameter d
        3. Using λ = d*p as the Poisson rate

    This is mathematically equivalent to the steady-state two-state promoter
    model with:
        - α = k_on / g_m
        - β = k_off / g_m
        - d = r_m / g_m

    Parameters
    ----------
    alpha : float or array_like
        Shape parameter α > 0 of the Beta distribution
    beta : float or array_like
        Shape parameter β > 0 of the Beta distribution
    dose : float or array_like
        Dose parameter d > 0 that scales the Beta-distributed rate
    validate_args : bool, optional
        Whether to validate arguments
    """

    # Define constraints for the distribution parameters
    arg_constraints = {
        "alpha": constraints.positive,
        "beta": constraints.positive,
        "dose": constraints.positive,
    }
    # Define the support of the distribution
    support = constraints.nonnegative_integer
    pytree_data_fields = ("alpha", "beta", "dose", "_beta_dist")

    # Initialize the distribution
    def __init__(self, alpha, beta, dose, *, validate_args=None):
        # Store the parameters
        self.alpha, self.beta, self.dose = promote_shapes(alpha, beta, dose)
        # Create a Beta distribution object
        self._beta_dist = dist.Beta(self.alpha, self.beta)
        # Initialize the distribution
        super(BetaPoisson, self).__init__(
            self._beta_dist.batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        """Sample from Beta-Poisson using two-step process."""
        assert is_prng_key(key)
        key_beta, key_poisson = random.split(key)

        # Sample p from Beta(α, β)
        p = self._beta_dist.sample(key_beta, sample_shape)

        # Sample from Poisson(d*p)
        rate = self.dose * p
        return dist.Poisson(rate).sample(key_poisson)

    @validate_sample
    def log_prob(self, value):
        """
        Compute log probability using the exact Beta-Poisson PMF.

        Uses the derived formula:
        P(X=x) = [Γ(α+x)/(x!Γ(α))] × [Γ(α+β)/Γ(α+β+x)] × d^x × ₁F₁(α+x, α+β+x, -d)
        """
        # Convert value to ensure it's an array
        value = jnp.asarray(value)

        # Compute log probability components
        log_prob = (
            # Γ(α + x) / (x! Γ(α))
            jsps.gammaln(self.alpha + value)
            - jsps.gammaln(value + 1)
            - jsps.gammaln(self.alpha)
            # Γ(α + β) / Γ(α + β + x)
            + jsps.gammaln(self.alpha + self.beta)
            - jsps.gammaln(self.alpha + self.beta + value)
            # d^x
            + value * jnp.log(self.dose)
        )

        # Add hypergeometric function term: ₁F₁(α+x, α+β+x, -d)
        hyp_args = (
            self.alpha + value,  # a
            self.alpha + self.beta + value,  # b
            self.dose,  # z
        )

        # Use standard JAX hypergeometric function with current precision
        hyp_value = jsps.hyp1f1(*hyp_args)

        # Handle infinite values using JAX-compatible operations
        log_hyp = jnp.where(
            jnp.isinf(hyp_value), -jnp.inf, jnp.log(hyp_value) - self.dose
        )

        return log_prob + log_hyp

    @property
    def mean(self):
        """Expected value of Beta-Poisson distribution."""
        # E[X] = E[E[X|p]] = E[d*p] = d*E[p] = d*α/(α+β)
        return self.dose * self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        """Variance of Beta-Poisson distribution."""
        # Var[X] = E[Var[X|p]] + Var[E[X|p]]
        # = E[d*p] + Var[d*p]
        # = d*α/(α+β) + d²*Var[p]
        # = d*α/(α+β) + d²*αβ/[(α+β)²(α+β+1)]
        beta_mean = self.alpha / (self.alpha + self.beta)
        beta_var = (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )
        return self.dose * beta_mean + (self.dose**2) * beta_var


# ------------------------------------------------------------------------------
# Two-State Promoter Steady-State distribution
# ------------------------------------------------------------------------------


class TwoStatePromoter(BetaPoisson):
    """
    Two-State Promoter distribution implemented as a Beta-Poisson.

    This provides the same distribution as the original two-state promoter model
    but with more efficient sampling and a direct connection to the Beta-Poisson
    literature.

    Parameters
    ----------
    k_on : float or array_like
        Rate of promoter switching from OFF to ON state
    k_off : float or array_like
        Rate of promoter switching from ON to OFF state
    r_m : float or array_like
        Production rate of mRNA when promoter is ON
    validate_args : bool, optional
        Whether to validate arguments
    """

    arg_constraints = {
        "k_on": constraints.positive,
        "k_off": constraints.positive,
        "r_m": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(self, k_on, k_off, r_m, *, validate_args=None):
        # Store biological parameters
        self.k_on = k_on
        self.k_off = k_off
        self.r_m = r_m

        super().__init__(k_on, k_off, r_m, validate_args=validate_args)

    @property
    def mean(self):
        """Expected mRNA count in terms of biological parameters."""
        # E[mRNA] = r_m * k_on / (k_on + k_off)
        return self.r_m * self.k_on / (self.k_on + self.k_off)

    @property
    def variance(self):
        """Variance of mRNA count in terms of biological parameters."""
        mean_val = self.mean
        # Additional variance due to promoter switching
        burst_factor = 1 + self.r_m / (self.k_off + self.k_on)
        return mean_val * burst_factor


# ------------------------------------------------------------------------------
# Numerical quadrature for the Beta-Poisson distribution
# ------------------------------------------------------------------------------


def gauss_legendre_quadrature(n_points):
    """
    Compute Gauss-Legendre quadrature nodes and weights for interval [-1, 1].
    These are fixed and don't depend on parameters, making differentiation
    easier.
    """
    # Manual implementation of Gauss-Legendre quadrature
    # For small n_points, we can use pre-computed values
    # For larger n_points, we would need to implement the full algorithm

    if n_points == 10:
        # Pre-computed nodes and weights for n=10
        nodes = jnp.array(
            [
                -0.9739065285171717,
                -0.8650633666889845,
                -0.6794095682990244,
                -0.4333953941292472,
                -0.1488743389816312,
                0.1488743389816312,
                0.4333953941292472,
                0.6794095682990244,
                0.8650633666889845,
                0.9739065285171717,
            ]
        )
        weights = jnp.array(
            [
                0.0666713443086881,
                0.1494513491505806,
                0.2190863625159820,
                0.2692667193099963,
                0.2955242247147529,
                0.2955242247147529,
                0.2692667193099963,
                0.2190863625159820,
                0.1494513491505806,
                0.0666713443086881,
            ]
        )
    elif n_points == 20:
        # Pre-computed nodes and weights for n=20
        nodes = jnp.array(
            [
                -0.9931285991850949,
                -0.9639719272779138,
                -0.9122344282513259,
                -0.8391169718222188,
                -0.7463319064601508,
                -0.6360536807265150,
                -0.5108670019508271,
                -0.3737060887154196,
                -0.2277858511416451,
                -0.0765265211334973,
                0.0765265211334973,
                0.2277858511416451,
                0.3737060887154196,
                0.5108670019508271,
                0.6360536807265150,
                0.7463319064601508,
                0.8391169718222188,
                0.9122344282513259,
                0.9639719272779138,
                0.9931285991850949,
            ]
        )
        weights = jnp.array(
            [
                0.0176140071391521,
                0.0406014298003869,
                0.0626720483341091,
                0.0832767415767047,
                0.1019301198172404,
                0.1181945319615184,
                0.1316886384491766,
                0.1420961093183821,
                0.1491729864726037,
                0.1527533871307258,
                0.1527533871307258,
                0.1491729864726037,
                0.1420961093183821,
                0.1316886384491766,
                0.1181945319615184,
                0.1019301198172404,
                0.0832767415767047,
                0.0626720483341091,
                0.0406014298003869,
                0.0176140071391521,
            ]
        )
    else:
        # For other values, use a simple approximation
        # This is not optimal but will work for testing
        nodes = jnp.linspace(-1.0, 1.0, n_points)
        weights = jnp.ones(n_points) * (2.0 / n_points)

    return nodes, weights


# ------------------------------------------------------------------------------


def beta_poisson_log_prob_quadrature(x, k_on, k_off, r_m, n_quad=20):
    """
    Compute log P(x | k_on, k_off, r_m) using Gauss-Legendre quadrature.

    This approximates:
    ∫₀¹ Poisson(x | r_m*p) * Beta(p | k_on, k_off) dp

    Parameters
    ----------
    x : int or array
        Observed counts
    k_on, k_off, r_m : float or array
        Two-state promoter parameters
    n_quad : int
        Number of quadrature points
    """
    # Get quadrature nodes and weights for [-1, 1]
    nodes_std, weights_std = gauss_legendre_quadrature(n_quad)

    # Transform nodes from [-1, 1] to [0, 1]
    nodes = (nodes_std + 1) / 2
    weights = weights_std / 2  # Jacobian of transformation

    # Ensure all inputs are arrays
    x = jnp.asarray(x)
    k_on = jnp.asarray(k_on)
    k_off = jnp.asarray(k_off)
    r_m = jnp.asarray(r_m)

    # Get the broadcast shape of all parameters
    param_shapes = [
        jnp.shape(x),
        jnp.shape(k_on),
        jnp.shape(k_off),
        jnp.shape(r_m),
    ]
    broadcast_shape = jax.lax.broadcast_shapes(*param_shapes)

    # Flatten all inputs to 1D for processing
    x_flat = x.ravel()
    k_on_flat = k_on.ravel()
    k_off_flat = k_off.ravel()
    r_m_flat = r_m.ravel()

    # Reshape nodes and weights for broadcasting with flattened parameters
    nodes_expanded = nodes[:, None]  # Shape: (n_quad, 1)
    weights_expanded = weights[:, None]  # Shape: (n_quad, 1)

    # Poisson log likelihood: log P(x | r_m * p)
    poisson_rates = r_m_flat * nodes_expanded  # Shape: (n_quad, n_params)
    poisson_log_probs = (
        x_flat * jnp.log(poisson_rates)
        - poisson_rates
        - jsps.gammaln(x_flat + 1)
    )

    # Beta log likelihood: log Beta(p | k_on, k_off)
    beta_log_probs = (
        (k_on_flat - 1) * jnp.log(nodes_expanded)
        + (k_off_flat - 1) * jnp.log(1 - nodes_expanded)
        - jsps.gammaln(k_on_flat)
        - jsps.gammaln(k_off_flat)
        + jsps.gammaln(k_on_flat + k_off_flat)
    )

    # Combined log integrand
    log_integrand = poisson_log_probs + beta_log_probs

    # Numerical integration in log space
    # log(∫ f(p) dp) ≈ log(Σᵢ wᵢ f(pᵢ)) = logsumexp(log(wᵢ) + log(f(pᵢ)))
    log_weights = jnp.log(weights_expanded)
    log_integral_flat = logsumexp(log_weights + log_integrand, axis=0)

    # Reshape back to original broadcast shape
    log_integral = log_integral_flat.reshape(broadcast_shape)

    return log_integral


# ------------------------------------------------------------------------------


def beta_poisson_log_prob_adaptive(
    x, k_on, k_off, r_m, n_quad_base=10, max_points=50
):
    """
    Compute the log-probability of observed counts under the Beta-Poisson model
    using adaptive Gauss-Legendre quadrature.

    This function adaptively increases the number of quadrature points until the
    result converges or a maximum number of points is reached. This can improve
    accuracy for challenging parameter regimes.

    Parameters
    ----------
    x : int or array-like
        Observed count(s) for which to compute the log-probability.
    k_on : float or array-like
        Promoter ON rate parameter(s) of the Beta distribution.
    k_off : float or array-like
        Promoter OFF rate parameter(s) of the Beta distribution.
    r_m : float or array-like
        mRNA production rate(s) when the promoter is ON.
    n_quad_base : int, optional
        Initial number of quadrature points (default: 10).
    max_points : int, optional
        Maximum number of quadrature points to use (default: 50).

    Returns
    -------
    log_prob : float or ndarray
        The log-probability of the observed count(s) under the model.
    """

    # Define a helper function to compute the integral for a given number of points
    def compute_integral(n_points):
        # Compute the log-probability using quadrature with n_points
        return beta_poisson_log_prob_quadrature(x, k_on, k_off, r_m, n_points)

    # Start with the base number of quadrature points
    prev_result = compute_integral(n_quad_base)
    # Double the number of points for the next iteration
    n_points = n_quad_base * 2
    # Continue increasing points until convergence or max_points is reached
    while n_points <= max_points:
        # Compute the result with the current number of points
        curr_result = compute_integral(n_points)
        # Compute the relative error between current and previous result
        rel_error = jnp.abs(curr_result - prev_result) / (
            jnp.abs(prev_result) + 1e-8
        )
        # Check if all elements have converged below the threshold
        converged = jnp.all(rel_error < 1e-6)
        # If converged, return the current result
        if converged:
            return curr_result
        # Otherwise, update previous result and double the points
        prev_result = curr_result
        n_points *= 2

    # If max_points reached, return the last computed result
    return prev_result


# ------------------------------------------------------------------------------


class TwoStatePromoterQuadrature(dist.Distribution):
    """
    Distribution representing the two-state promoter (Beta-Poisson) model using
    numerical quadrature.

    This class implements the compound distribution where the mRNA count is
    modeled as:
        - p ~ Beta(k_on, k_off)
        - x ~ Poisson(r_m * p)

    The marginal likelihood is computed via Gauss-Legendre quadrature to avoid
    numerical instabilities associated with special functions.

    Parameters
    ----------
    k_on : float or array-like
        Promoter ON rate parameter(s) of the Beta distribution.
    k_off : float or array-like
        Promoter OFF rate parameter(s) of the Beta distribution.
    r_m : float or array-like
        mRNA production rate(s) when the promoter is ON.
    n_quad : int, optional
        Number of quadrature points to use for numerical integration (default:
        20).
    validate_args : bool, optional
        Whether to validate the input arguments (default: None).
    """

    # Define argument constraints for the distribution parameters
    arg_constraints = {
        "k_on": constraints.positive,
        "k_off": constraints.positive,
        "r_m": constraints.positive,
    }
    # Define the support of the distribution (nonnegative integers)
    support = constraints.nonnegative_integer

    def __init__(self, k_on, k_off, r_m, n_quad=20, validate_args=None):
        """
        Initialize the TwoStatePromoterQuadrature distribution.

        Parameters
        ----------
        k_on : float or array-like
            Promoter ON rate parameter(s) of the Beta distribution.
        k_off : float or array-like
            Promoter OFF rate parameter(s) of the Beta distribution.
        r_m : float or array-like
            mRNA production rate(s) when the promoter is ON.
        n_quad : int, optional
            Number of quadrature points to use for numerical integration
            (default: 20).
        validate_args : bool, optional
            Whether to validate the input arguments (default: None).
        """
        # Store the parameters as instance variables
        self.k_on = k_on
        self.k_off = k_off
        self.r_m = r_m
        self.n_quad = n_quad

        # Compute the batch shape by broadcasting parameter shapes
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(k_on), jnp.shape(k_off), jnp.shape(r_m)
        )

        # Call the parent class constructor with batch and event shapes
        super().__init__(
            batch_shape=batch_shape, event_shape=(), validate_args=validate_args
        )

    def log_prob(self, value):
        """
        Compute the log-probability of a given value under the two-state
        promoter model using Gauss-Legendre quadrature.

        Parameters
        ----------
        value : int or array-like
            Observed count(s) for which to compute the log-probability.

        Returns
        -------
        log_prob : float or ndarray
            The log-probability of the observed count(s) under the model.
        """
        # Call the quadrature log-probability function with stored parameters
        return beta_poisson_log_prob_quadrature(
            value, self.k_on, self.k_off, self.r_m, self.n_quad
        )

    def sample(self, key, sample_shape=()):
        """
        Generate samples from the two-state promoter (Beta-Poisson)
        distribution.

        Sampling is performed by first drawing p ~ Beta(k_on, k_off), then
        drawing x ~ Poisson(r_m * p). This method does not use quadrature and is
        exact for the generative process.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        sample_shape : tuple, optional
            Shape of the samples to generate (default: ()).

        Returns
        -------
        x_samples : ndarray
            Samples from the two-state promoter distribution.
        """
        # Compute the total sample shape by combining sample_shape and
        # batch_shape
        total_shape = sample_shape + self.batch_shape

        # Create a Beta distribution with the current parameters
        beta_dist = dist.Beta(self.k_on, self.k_off)
        # Sample p from the Beta distribution
        p_samples = beta_dist.sample(key, sample_shape)

        # Split the random key for Poisson sampling
        key, subkey = random.split(key)
        # Compute Poisson rates as r_m * p_samples
        poisson_rates = self.r_m * p_samples
        # Create a Poisson distribution with the computed rates
        poisson_dist = dist.Poisson(poisson_rates)
        # Sample x from the Poisson distribution
        x_samples = poisson_dist.sample(subkey, ())

        # Return the sampled mRNA counts
        return x_samples


# ------------------------------------------------------------------------------
# Monte Carlo version of the two-state promoter distribution
# ------------------------------------------------------------------------------


# Alternative implementation using Monte Carlo integration
def beta_poisson_log_prob_mc(x, k_on, k_off, r_m, n_samples=1000, key=None):
    """
    Monte Carlo approximation of the Beta-Poisson log probability.

    This can be more accurate for complex parameter ranges and is
    naturally differentiable via the reparameterization trick.
    """
    if key is None:
        key = random.PRNGKey(0)

    # Ensure all inputs are arrays
    x = jnp.asarray(x)
    k_on = jnp.asarray(k_on)
    k_off = jnp.asarray(k_off)
    r_m = jnp.asarray(r_m)

    # Get the broadcast shape of all parameters
    param_shapes = [
        jnp.shape(x),
        jnp.shape(k_on),
        jnp.shape(k_off),
        jnp.shape(r_m),
    ]
    broadcast_shape = jax.lax.broadcast_shapes(*param_shapes)

    # Flatten all inputs to 1D for processing
    x_flat = x.ravel()
    k_on_flat = k_on.ravel()
    k_off_flat = k_off.ravel()
    r_m_flat = r_m.ravel()

    # Sample p from Beta(k_on, k_off) for each parameter set
    beta_dist = dist.Beta(k_on_flat, k_off_flat)
    p_samples = beta_dist.sample(
        key, (n_samples, len(x_flat))
    )  # Shape: (n_samples, n_params)

    # Compute Poisson log probs for each sample
    poisson_rates = r_m_flat * p_samples  # Shape: (n_samples, n_params)
    poisson_log_probs = (
        x_flat * jnp.log(poisson_rates)
        - poisson_rates
        - jsps.gammaln(x_flat + 1)
    )

    # Monte Carlo estimate: log(E[likelihood]) ≈ logsumexp(log_probs) - log(n_samples)
    mc_log_prob_flat = logsumexp(poisson_log_probs, axis=0) - jnp.log(n_samples)

    # Reshape back to original broadcast shape
    mc_log_prob = mc_log_prob_flat.reshape(broadcast_shape)

    return mc_log_prob


# ------------------------------------------------------------------------------


class TwoStatePromoterMC(TwoStatePromoterQuadrature):
    """Monte Carlo version of the two-state promoter distribution."""

    def __init__(self, k_on, k_off, r_m, n_samples=1000, validate_args=None):
        self.n_samples = n_samples
        super().__init__(k_on, k_off, r_m, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value):
        """Compute log probability using Monte Carlo integration."""
        # Use a fixed key for reproducibility (could be made configurable)
        key = random.PRNGKey(42)
        return beta_poisson_log_prob_mc(
            value, self.k_on, self.k_off, self.r_m, self.n_samples, key
        )


# Usage example and testing
if __name__ == "__main__":
    # Test parameters
    k_on = 2.0
    k_off = 1.5
    r_m = 3.0
    x = 5

    # Compare quadrature vs Monte Carlo
    quad_result = beta_poisson_log_prob_quadrature(
        x, k_on, k_off, r_m, n_quad=20
    )
    mc_result = beta_poisson_log_prob_mc(x, k_on, k_off, r_m, n_samples=10000)

    print(f"Quadrature result: {quad_result}")
    print(f"Monte Carlo result: {mc_result}")
    print(f"Difference: {abs(quad_result - mc_result)}")

    # Test gradients
    grad_fn = jax.grad(
        lambda params: beta_poisson_log_prob_quadrature(
            x, params[0], params[1], params[2], n_quad=20
        )
    )

    gradients = grad_fn(jnp.array([k_on, k_off, r_m]))
    print(f"Gradients: {gradients}")

    # Test with distribution class
    dist_quad = TwoStatePromoterQuadrature(k_on, k_off, r_m, n_quad=20)
    log_prob = dist_quad.log_prob(x)
    print(f"Distribution log_prob: {log_prob}")

    # Sample from distribution
    key = random.PRNGKey(123)
    samples = dist_quad.sample(key, (10,))
    print(f"Samples: {samples}")
