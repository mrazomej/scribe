"""Dirichlet distribution fitting and sampling functions."""

from typing import Union
from functools import partial

import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax import scipy as jsp
import jax

from numpyro.distributions import Dirichlet

# ==============================================================================
# Dirichlet distribution functions
# ==============================================================================


def sample_dirichlet_from_parameters(
    parameter_samples: Union[np.ndarray, jnp.ndarray],
    n_samples_dirichlet: int = 1,
    rng_key: random.PRNGKey = random.PRNGKey(42),
) -> jnp.ndarray:
    """
    Samples from a Dirichlet distribution given an array of parameter samples.

    Parameters
    ----------
    parameter_samples : array-like
        Array of shape (n_samples, n_variables) containing parameter samples
        to use as concentration parameters for Dirichlet distributions
    n_samples_dirichlet : int, optional
        Number of samples to draw from each Dirichlet distribution (default: 1)
    rng_key : random.PRNGKey
        JAX random number generator key. Defaults to random.PRNGKey(42)

    Returns
    -------
    jnp.ndarray
        If n_samples_dirichlet=1:
            Array of shape (n_samples, n_variables)
        If n_samples_dirichlet>1:
            Array of shape (n_samples, n_variables, n_samples_dirichlet)
    """
    # Get dimensions
    n_samples, n_variables = parameter_samples.shape

    # Create Dirichlet distribution
    dirichlet_dist = Dirichlet(parameter_samples)

    # Sample from the distribution
    samples = dirichlet_dist.sample(
        rng_key, sample_shape=(n_samples_dirichlet,)
    )

    if n_samples_dirichlet == 1:
        # Return 2D array if only one sample per distribution
        return jnp.transpose(samples, (1, 2, 0)).squeeze(axis=-1)
    else:
        # Return 3D array if multiple samples per distribution
        return jnp.transpose(samples, (1, 2, 0))


# ==============================================================================
# Dirichlet distribution fitting functions
# ==============================================================================


def fit_dirichlet_mle(
    samples: Union[np.ndarray, jnp.ndarray],
    max_iter: int = 1000,
    tol: float = 1e-7,
    sample_axis: int = 0,
) -> jnp.ndarray:
    """
    Fit a Dirichlet distribution to samples using Maximum Likelihood Estimation.

    This implementation uses Newton's method to find the concentration
    parameters that maximize the likelihood of the observed samples. The
    algorithm iteratively updates the concentration parameters using gradient
    and Hessian information until convergence.

    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_variables) by default, or (n_variables,
        n_samples) if sample_axis=1, containing Dirichlet samples. Each
        row/column should sum to 1.
    max_iter : int, optional
        Maximum number of iterations for optimization (default: 1000)
    tol : float, optional
        Tolerance for convergence in parameter updates (default: 1e-7)
    sample_axis : int, optional
        Axis containing samples (default: 0)

    Returns
    -------
    jnp.ndarray
        Array of concentration parameters for the fitted Dirichlet distribution.
        Shape is (n_variables,).
    """
    # Convert input samples to JAX array and transpose if needed
    x = jnp.asarray(samples)
    if sample_axis == 1:
        x = x.T

    # Extract dimensions of the input data
    n_samples, n_variables = x.shape

    # Initialize alpha parameters using method of moments estimator
    # This uses the mean and variance of the samples to get a starting point
    alpha = (
        jnp.mean(x, axis=0)
        * (jnp.mean(x * (1 - x), axis=0) / jnp.var(x, axis=0)).mean()
    )

    # Compute mean of log samples - this is a sufficient statistic for the MLE
    mean_log_x = jnp.mean(jnp.log(x), axis=0)

    def iteration_step(alpha):
        """Single iteration of Newton's method"""
        # Compute sum of current alpha values
        alpha_sum = jnp.sum(alpha)

        # Compute gradient using digamma functions
        grad = mean_log_x - (
            jsp.special.digamma(alpha) - jsp.special.digamma(alpha_sum)
        )

        # Compute diagonal of Hessian matrix
        q = -jsp.special.polygamma(1, alpha)

        # Compute sum term for Hessian
        z = 1.0 / jsp.special.polygamma(1, alpha_sum)

        # Compute update step using gradient and Hessian information
        b = jnp.sum(grad / q) / (1.0 / z + jnp.sum(1.0 / q))

        # Return updated alpha parameters
        return alpha - (grad - b) / q

    # Iterate Newton's method until convergence or max iterations reached
    for _ in range(max_iter):
        # Compute new alpha values
        alpha_new = iteration_step(alpha)

        # Check for convergence
        if jnp.max(jnp.abs(alpha_new - alpha)) < tol:
            break

        # Update alpha for next iteration
        alpha = alpha_new

    # Return final concentration parameters
    return alpha


# ==============================================================================
# Digamma function functions
# ==============================================================================


def digamma_inv(y, num_iters=5):
    """
    Approximate the inverse of the digamma function using Newton iterations.

    The digamma function ψ(x) is the derivative of the log of the gamma function.
    This function computes its inverse ψ⁻¹(y) using Newton's method:
        x_{n+1} = x_n - (ψ(x_n) - y) / ψ'(x_n)
    where ψ' is the trigamma function (polygamma of order 1).

    Parameters
    ----------
    y : array-like
        The input value(s) (can be scalar or vector) representing ψ(x) values
        for which we want to find x.
    num_iters : int, optional
        Number of Newton iterations (default: 5). More iterations increase
        accuracy but take longer to compute.

    Returns
    -------
    x : array-like
        The approximate inverse digamma of y, such that ψ(x) ≈ y.
    """
    # Choose initial guess based on input value:
    # For y >= -2.22, use x₀ = exp(y) + 0.5 (good for larger values)
    # For y < -2.22, use x₀ = -1/y (better for negative values)
    x = jnp.where(y >= -2.22, jnp.exp(y) + 0.5, -1.0 / y)

    # Perform Newton iterations to improve the approximation
    for _ in range(num_iters):
        # Newton update: x = x - f(x)/f'(x) where f(x) = ψ(x) - y
        # Here f'(x) = ψ'(x) = polygamma(1, x)
        x = x - (jsp.special.digamma(x) - y) / jsp.special.polygamma(1, x)

    # Return the final approximation
    return x


# ==============================================================================
# Dirichlet distribution fitting functions
# Minka's fixed-point iteration
# ==============================================================================


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
        3,
    ),
)
def fit_dirichlet_minka(samples, max_iter=1000, tol=1e-7, sample_axis=0):
    """
    Fit a Dirichlet distribution to data using Minka's fixed-point iteration.

    This function uses the relation:
        ψ(α_j) - ψ(α₀) = ⟨ln x_j⟩    (with α₀ = ∑ₖ αₖ)
    so that the fixed point update is:
        α_j ← ψ⁻¹( ψ(α₀) + ⟨ln x_j⟩ )

    This method is generally more stable and faster than moment matching or
    maximum likelihood estimation via gradient descent.

    Parameters
    ----------
    samples : array-like
        Data array with shape (n_samples, n_variables) by default (or transposed
        if sample_axis=1). Each row should sum to 1 (i.e., be a probability
        vector).
    max_iter : int, optional
        Maximum number of iterations for the fixed-point algorithm.
    tol : float, optional
        Tolerance for convergence - algorithm stops when max change in α is
        below this.
    sample_axis : int, optional
        Axis containing samples (default: 0). Use 1 if data is (n_variables,
        n_samples).

    Returns
    -------
    jnp.ndarray
        Estimated concentration parameters (α) of shape (n_variables,).
    """
    # Convert input to JAX array and transpose if needed to get (n_samples,
    # n_variables)
    x = jnp.asarray(samples)
    if sample_axis == 1:
        x = x.T
    n_samples, n_variables = x.shape

    # Compute mean of log(x) across samples. Add small constant to avoid log(0)
    # This estimates ⟨ln x_j⟩ for each variable j
    mean_log_x = jnp.mean(jnp.log(x + 1e-10), axis=0)

    # Initialize concentration parameters to ones
    # This is a common choice that works well in practice
    alpha = jnp.ones(n_variables)

    # Define condition function for while loop
    # Continues while iteration count is below max and change in alpha is above
    # tolerance
    def cond_fun(val):
        i, alpha, diff = val
        return jnp.logical_and(i < max_iter, diff > tol)

    # Define update function for while loop
    def body_fun(val):
        i, alpha, _ = val
        # Compute sum of all alphas (α₀)
        alpha0 = jnp.sum(alpha)
        # Implement Minka's fixed-point update:
        #   α_j = ψ⁻¹(ψ(α₀) + ⟨ln x_j⟩)
        alpha_new = digamma_inv(jsp.special.digamma(alpha0) + mean_log_x)
        # Compute maximum absolute change in alpha for convergence check
        diff = jnp.max(jnp.abs(alpha_new - alpha))
        return i + 1, alpha_new, diff

    # Run fixed-point iteration until convergence or max iterations reached
    _, alpha, _ = jax.lax.while_loop(cond_fun, body_fun, (0, alpha, jnp.inf))
    return alpha
