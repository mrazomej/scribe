"""
Statistics functions
"""

# Import numpy for array manipulation
import numpy as np

# Import typing
from typing import Union
from functools import partial

# Import JAX-related libraries
import jax.numpy as jnp
import jax.random as random
from jax import scipy as jsp
import jax

# Import numpyro distributions
from numpyro.distributions import (
    constraints,
    Distribution,
    Beta,
    Normal,
    LogNormal,
    Dirichlet,
    Gamma,
)

from numpyro.distributions.util import promote_shapes, validate_sample

# Import numpyro KL divergence
from numpyro.distributions.kl import kl_divergence

# Import multipledispatch
from multipledispatch import dispatch

# ------------------------------------------------------------------------------
# Histogram functions
# ------------------------------------------------------------------------------


def compute_histogram_percentiles(
    samples, percentiles=[5, 25, 50, 75, 95], normalize=True, sample_axis=0
):
    """
    Compute percentiles of histogram frequencies across multiple samples.

    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points, n_samples)
        if sample_axis=1
    percentiles : list-like, optional
        List of percentiles to compute (default: [5, 25, 50, 75, 95])
    normalize : bool, optional
        Whether to normalize histograms (default: True)
    sample_axis : int, optional
        Axis containing samples (default: 0)

    Returns
    -------
    bin_edges : array
        Array of bin edges (integers from min to max value + 1)
    hist_percentiles : array
        Array of shape (len(percentiles), len(bin_edges)-1) containing
        the percentiles of histogram frequencies for each bin
    """
    # Find global min and max across all samples
    global_min = int(samples.min())
    global_max = int(samples.max())

    # Create bin edges (integers from min to max + 1)
    bin_edges = np.arange(
        global_min, global_max + 2
    )  # +2 because we want right edge

    # Initialize array to store histograms
    n_samples = samples.shape[sample_axis]
    n_bins = len(bin_edges) - 1
    all_hists = np.zeros((n_samples, n_bins))

    # Compute histogram for each sample
    for i in range(n_samples):
        sample = samples[i] if sample_axis == 0 else samples[:, i]
        hist, _ = np.histogram(sample, bins=bin_edges)
        if normalize:
            hist = hist / hist.sum()
        all_hists[i] = hist

    # Compute percentiles across samples for each bin
    hist_percentiles = np.percentile(all_hists, percentiles, axis=0)

    return bin_edges, hist_percentiles


# ------------------------------------------------------------------------------
# Credible regions functions
# ------------------------------------------------------------------------------


def compute_histogram_credible_regions(
    samples,
    credible_regions=[95, 68, 50],
    normalize=True,
    sample_axis=0,
    batch_size=1000,
    max_bin=None,
):
    """
    Compute credible regions of histogram frequencies across multiple samples.

    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points,
        n_samples) if sample_axis=1
    credible_regions : list-like, optional
        List of credible region percentages to compute (default: [95, 68, 50])
        For example, 95 will compute the 2.5 and 97.5 percentiles
    normalize : bool, optional
        Whether to normalize histograms (default: True)
    sample_axis : int, optional
        Axis containing samples (default: 0)
    batch_size : int, optional
        Number of samples to process in each batch (default: 100)
    max_bin : int, optional
        Maximum number of bins to process (default: None)

    Returns
    -------
    dict
        Dictionary containing: - 'bin_edges': array of bin edges - 'regions':
        nested dictionary where each key is the credible region percentage
          and values are dictionaries containing:
            - 'lower': lower bound of the credible region
            - 'upper': upper bound of the credible region
            - 'median': median (50th percentile)
    """
    # Convert to JAX array if not already
    samples = jnp.asarray(samples, dtype=jnp.float32)
    samples_2d = samples if sample_axis == 0 else samples.T

    # Find global min and max across all samples
    global_min = int(samples.min())

    # Define global max if max_bin is not None, else use global max
    global_max = (
        min(int(samples.max()), max_bin)
        if max_bin is not None
        else int(samples.max())
    )

    # Create bin edges (integers from min to max + 1)
    bin_edges = jnp.arange(global_min, global_max + 2)

    # Function to compute histogram for a single sample
    def compute_single_hist(sample):
        return jnp.histogram(sample, bins=bin_edges)[0]

    # Use lax.map to compute histograms (with automatic batching)
    all_counts = jax.lax.map(
        compute_single_hist, samples_2d, batch_size=batch_size
    )

    # Normalize if requested
    if normalize:
        all_hists = all_counts / all_counts.sum(axis=1)[:, None]
    else:
        all_hists = all_counts

    # Convert to numpy for percentile calculations
    all_hists = np.array(all_hists)

    # Compute credible regions
    results = {"bin_edges": np.array(bin_edges), "regions": {}}

    median = np.percentile(all_hists, 50, axis=0)

    for cr in credible_regions:
        lower_percentile = (100 - cr) / 2
        upper_percentile = 100 - lower_percentile

        results["regions"][cr] = {
            "lower": np.percentile(all_hists, lower_percentile, axis=0),
            "upper": np.percentile(all_hists, upper_percentile, axis=0),
            "median": median,
        }

    return results


# ------------------------------------------------------------------------------
# ECDF functions
# ------------------------------------------------------------------------------


def compute_ecdf_percentiles(
    samples, percentiles=[5, 25, 50, 75, 95], sample_axis=0
):
    """
    Compute percentiles of ECDF values across multiple samples of integers.

    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points,
        n_samples) if sample_axis=1, containing raw data samples of positive
        integers
    percentiles : list-like, optional
        List of percentiles to compute (default: [5, 25, 50, 75, 95])
    sample_axis : int, optional
        Axis containing samples (default: 0)

    Returns
    -------
    bin_edges : array
        Array of integer points at which ECDFs were evaluated (from min to max)
    ecdf_percentiles : array
        Array of shape (len(percentiles), len(bin_edges)) containing the
        percentiles of ECDF values at each integer point
    """
    # Convert to JAX arrays
    samples = jnp.asarray(samples)

    # Ensure samples are in the right orientation (n_samples, n_points)
    if sample_axis == 1:
        samples = samples.T

    # Find global min and max across all samples
    global_min = int(samples.min())
    global_max = int(samples.max())

    # Create evaluation points for each integer from min to max
    bin_edges = jnp.arange(global_min, global_max + 1)

    # Define function to compute ECDF for a single sample
    def compute_ecdf(sample):
        # For each integer value, count proportion of sample values <= that value
        return jax.vmap(lambda x: jnp.mean(sample <= x))(bin_edges)

    # Use vmap to apply compute_ecdf across all samples
    all_ecdfs = jax.vmap(compute_ecdf)(samples)

    # Convert to numpy for percentile calculations
    all_ecdfs_np = np.array(all_ecdfs)
    bin_edges_np = np.array(bin_edges)

    # Compute percentiles across samples for each evaluation point
    ecdf_percentiles = np.percentile(all_ecdfs_np, percentiles, axis=0)

    return bin_edges_np, ecdf_percentiles


# ------------------------------------------------------------------------------


def compute_ecdf_credible_regions(
    samples,
    credible_regions=[95, 68, 50],
    sample_axis=0,
    batch_size=1000,
    max_bin=None,
):
    """
    Compute credible regions of ECDF values across multiple samples.

    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points,
        n_samples) if sample_axis=1, containing raw data samples
    credible_regions : list-like, optional
        List of credible region percentages to compute (default: [95, 68, 50])
        For example, 95 will compute the 2.5 and 97.5 percentiles
    sample_axis : int, optional
        Axis containing samples (default: 0)
    batch_size : int, optional
        Number of samples to process in each batch (default: 1000)
    max_bin : int, optional
        Maximum value to include in ECDF evaluation (default: None)

    Returns
    -------
    dict
        Dictionary containing:
            - 'bin_edges': array of points at which ECDFs were evaluated
            - 'regions': nested dictionary where each key is the credible region
              percentage
            and values are dictionaries containing:
                - 'lower': lower bound of the credible region
                - 'upper': upper bound of the credible region
                - 'median': median (50th percentile)
    """
    # Convert to JAX array if not already
    samples = jnp.asarray(samples, dtype=jnp.float32)
    samples_2d = samples if sample_axis == 0 else samples.T

    # Find global min and max across all samples
    global_min = int(samples.min())

    # Define global max if max_bin is not None, else use global max
    global_max = (
        min(int(samples.max()), max_bin)
        if max_bin is not None
        else int(samples.max())
    )

    # Create evaluation points for each integer from min to max
    bin_edges = jnp.arange(global_min, global_max + 1)

    # Define function to compute ECDF for a single sample
    def compute_single_ecdf(sample):
        # For each evaluation point, count proportion of sample values <= that point
        return jax.vmap(lambda x: jnp.mean(sample <= x))(bin_edges)

    # Use vmap to compute ECDFs for all samples
    # Note: We could use jax.lax.map with batching here instead, but vmap is
    # already vectorized and efficient for this operation
    all_ecdfs = jax.vmap(compute_single_ecdf)(samples_2d)

    # Convert to numpy for percentile calculations
    all_ecdfs_np = np.array(all_ecdfs)
    bin_edges_np = np.array(bin_edges)

    # Compute credible regions
    results = {"bin_edges": bin_edges_np, "regions": {}}

    # Calculate median (50th percentile)
    median = np.percentile(all_ecdfs_np, 50, axis=0)

    # Calculate credible regions
    for cr in credible_regions:
        lower_percentile = (100 - cr) / 2
        upper_percentile = 100 - lower_percentile

        results["regions"][cr] = {
            "lower": np.percentile(all_ecdfs_np, lower_percentile, axis=0),
            "upper": np.percentile(all_ecdfs_np, upper_percentile, axis=0),
            "median": median,
        }

    return results


# ------------------------------------------------------------------------------
# Fraction of transcriptome functions
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# Beta Prime Distribution
# ------------------------------------------------------------------------------


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
        return x / y

    @validate_sample
    def log_prob(self, value):
        # log f(φ; α, β) = (β-1) log φ - (α+β) log(1+φ) - log B(β, α)
        log_num = (self.beta - 1) * jnp.log(value) - (
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


# ------------------------------------------------------------------------------


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
# Distribution Mode Monkey Patches
# ==============================================================================

# ------------------------------------------------------------------------------
# Mode functions
# ------------------------------------------------------------------------------


def _beta_mode(self):
    """Monkey patch mode property for Beta distribution."""
    a = self.concentration1
    b = self.concentration0
    interior = (a > 1) & (b > 1)
    left_bd = (a <= 1) & (b > 1)  # mode at 0
    right_bd = (a > 1) & (b <= 1)  # mode at 1
    both_bd = (a <= 1) & (b <= 1)  # two boundary modes: 0 and 1

    interior_val = (a - 1) / (a + b - 2)  # safe because interior ⇒ denom > 0

    # Return NaN where the mode is non-unique (both boundaries), else
    # 0/1/interior
    return jnp.where(
        interior,
        interior_val,
        jnp.where(left_bd, 0.0, jnp.where(right_bd, 1.0, jnp.nan)),
    )


def _lognormal_mode(self):
    """Monkey patch mode property for LogNormal distribution."""
    # mode = exp(μ - σ²)
    return jnp.exp(self.loc - self.scale**2)


def _normal_mode(self):
    """Monkey patch mode property for Normal distribution."""
    # mode = μ (mean)
    return self.loc


def apply_distribution_mode_patches():
    """Apply mode property patches to NumPyro distributions."""
    from numpyro.distributions.continuous import Beta, LogNormal, Normal

    # Only add if not already present
    if not hasattr(Beta, "mode"):
        Beta.mode = property(_beta_mode)

    if not hasattr(LogNormal, "mode"):
        LogNormal.mode = property(_lognormal_mode)

    if not hasattr(Normal, "mode"):
        Normal.mode = property(_normal_mode)


# ------------------------------------------------------------------------------
# Multipledispatch Jensen-Shannon divergence
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# Multipledispatch Hellinger distance functions
# ------------------------------------------------------------------------------


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


@dispatch(Beta, Beta)
def hellinger(p, q):
    """Compute the Hellinger distance between two Beta distributions."""
    return jnp.sqrt(sq_hellinger(p, q))


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


@dispatch(BetaPrime, BetaPrime)
def hellinger(p, q):
    """Compute the Hellinger distance between two BetaPrime distributions."""
    return jnp.sqrt(sq_hellinger(p, q))


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


@dispatch(Normal, Normal)
def hellinger(p, q):
    """Compute the Hellinger distance between two Normal distributions."""
    return jnp.sqrt(sq_hellinger(p, q))


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


@dispatch(LogNormal, LogNormal)
def hellinger(p, q):
    """Compute the Hellinger distance between two LogNormal distributions."""
    return jnp.sqrt(sq_hellinger(p, q))


# ------------------------------------------------------------------------------
# Module exports
# ------------------------------------------------------------------------------


__all__ = [
    # Histogram functions
    "compute_histogram_percentiles",
    "compute_histogram_credible_regions",
    # ECDF functions
    "compute_ecdf_percentiles",
    "compute_ecdf_credible_regions",
    # Fraction of transcriptome functions
    "sample_dirichlet_from_parameters",
    "fit_dirichlet_mle",
    "fit_dirichlet_minka",
    # Hellinger distance functions
    "sq_hellinger",
    "hellinger",
    # Multipledispatch functions
    "jensen_shannon",
    # Distribution classes
    "BetaPrime",
]
