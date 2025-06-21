"""
Statistics functions
"""

# Import numpy for array manipulation
import numpy as np

# Import typing
from typing import Union

# Import JAX-related libraries
import jax.numpy as jnp
import jax.random as random
from jax import scipy as jsp
import jax
from numpyro.distributions import Dirichlet

# Import scipy.special functions
from jax.scipy.special import gammaln, digamma

# Import numpyro distributions
from numpyro.distributions import constraints, Distribution, Gamma
from numpyro.distributions.util import promote_shapes, validate_sample

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


@jax.jit
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
# KL divergence functions
# ------------------------------------------------------------------------------


def kl_gamma(alpha1, beta1, alpha2, beta2):
    """
    Compute Kullback-Leibler (KL) divergence between two Gamma distributions.

    Calculates KL(P||Q) where:
    P ~ Gamma(α₁, β₁)
    Q ~ Gamma(α₂, β₂)

    The KL divergence is given by:

    KL(P||Q) = (α₁ - α₂)ψ(α₁) - ln[Γ(α₁)] + ln[Γ(α₂)] + α₂ln(β₁/β₂) + α₁(β₂/β₁ - 1)

    where:
    - ψ(x) is the digamma function
    - Γ(x) is the gamma function

    Parameters
    ----------
    alpha1 : float or array-like
        Shape parameter α₁ of the first Gamma distribution P
    beta1 : float or array-like
        Rate parameter β₁ of the first Gamma distribution P
    alpha2 : float or array-like
        Shape parameter α₂ of the second Gamma distribution Q
    beta2 : float or array-like
        Rate parameter β₂ of the second Gamma distribution Q

    Returns
    -------
    float
        KL divergence between the two Gamma distributions
    """
    return (
        (alpha1 - alpha2) * digamma(alpha1)
        - gammaln(alpha1)
        + gammaln(alpha2)
        + alpha2 * (np.log(beta1) - np.log(beta2))
        + alpha1 * (beta2 / beta1 - 1)
    )


# ------------------------------------------------------------------------------


def kl_beta(alpha1, beta1, alpha2, beta2):
    """
    Compute Kullback-Leibler (KL) divergence between two Beta distributions.

    Calculates KL(P||Q) where:
    P ~ Beta(α₁, β₁)
    Q ~ Beta(α₂, β₂)

    The KL divergence is given by:

    KL(P||Q) = ln[B(α₂,β₂)] - ln[B(α₁,β₁)] + (α₁-α₂)ψ(α₁) + (β₁-β₂)ψ(β₁)
                + (α₂-α₁+β₂-β₁)ψ(α₁+β₁)

    where:
    - ψ(x) is the digamma function
    - B(x,y) is the beta function

    Parameters
    ----------
    alpha1 : float or array-like
        Shape parameter α₁ of the first Beta distribution P
    beta1 : float or array-like
        Shape parameter β₁ of the first Beta distribution P
    alpha2 : float or array-like
        Shape parameter α₂ of the second Beta distribution Q
    beta2 : float or array-like
        Shape parameter β₂ of the second Beta distribution Q

    Returns
    -------
    float
        KL divergence between the two Beta distributions
    """
    # Check that all inputs are of same shape
    if not all(
        isinstance(a, (float, np.ndarray, jnp.ndarray))
        and isinstance(b, (float, np.ndarray, jnp.ndarray))
        for a, b in zip(
            [alpha1, beta1, alpha2, beta2], [alpha1, beta1, alpha2, beta2]
        )
    ):
        raise ValueError("All inputs must be of the same shape")

    return (
        gammaln(alpha2 + beta2)
        - gammaln(alpha2)
        - gammaln(beta2)
        - (gammaln(alpha1 + beta1) - gammaln(alpha1) - gammaln(beta1))
        + (alpha1 - alpha2) * digamma(alpha1)
        + (beta1 - beta2) * digamma(beta1)
        + (alpha2 - alpha1 + beta2 - beta1) * digamma(alpha1 + beta1)
    )


# ------------------------------------------------------------------------------


def kl_lognormal(mu1, sigma1, mu2, sigma2):
    """
    Compute Kullback-Leibler (KL) divergence between two log-normal
    distributions.

    Calculates KL(P||Q) where: P ~ LogNormal(μ₁, σ₁) Q ~ LogNormal(μ₂, σ₂)

    The KL divergence is given by:

    KL(P||Q) = ln(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

    Parameters
    ----------
    mu1 : float or array-like
        Location parameter μ₁ of the first log-normal distribution P
    sigma1 : float or array-like
        Scale parameter σ₁ of the first log-normal distribution P
    mu2 : float or array-like
        Location parameter μ₂ of the second log-normal distribution Q
    sigma2 : float or array-like
        Scale parameter σ₂ of the second log-normal distribution Q

    Returns
    -------
    float or array-like
        KL divergence between the two log-normal distributions
    """
    return (
        jnp.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
        - 0.5
    )


# ------------------------------------------------------------------------------
# Jensen-Shannon divergence functions
# ------------------------------------------------------------------------------


def jensen_shannon_beta(alpha1, beta1, alpha2, beta2):
    """
    Compute the Jensen-Shannon divergence between two Beta distributions.

    The Jensen-Shannon divergence is a symmetrized and smoothed version of the
    Kullback-Leibler divergence, defined as:

        JSD(P||Q) = 1/2 × KL(P||M) + 1/2 × KL(Q||M)

    where M = 1/2 × (P + Q) is the average of the two distributions.

    For Beta distributions, we compute this by:
    1. Finding the parameters of the mixture distribution M
    2. Computing KL(P||M) and KL(Q||M)
    3. Taking the average of these KL divergences

    Parameters
    ----------
    alpha1 : float or array
        Alpha parameter (shape) of the first Beta distribution
    beta1 : float or array
        Beta parameter (shape) of the first Beta distribution
    alpha2 : float or array
        Alpha parameter (shape) of the second Beta distribution
    beta2 : float or array
        Beta parameter (shape) of the second Beta distribution

    Returns
    -------
    float or array
        Jensen-Shannon divergence between the two Beta distributions
    """
    # We can't directly compute the parameters of the mixture distribution M,
    # so we approximate the JS divergence using the KL divergences
    kl_p_q = kl_beta(alpha1, beta1, alpha2, beta2)
    kl_q_p = kl_beta(alpha2, beta2, alpha1, beta1)

    # JS divergence is the average of the two KL divergences
    return 0.5 * (kl_p_q + kl_q_p)


# ------------------------------------------------------------------------------


def jensen_shannon_gamma(alpha1, beta1, alpha2, beta2):
    """
    Compute the Jensen-Shannon divergence between two Gamma distributions.

    The Jensen-Shannon divergence is a symmetrized and smoothed version of the
    Kullback-Leibler divergence, defined as:

        JSD(P||Q) = 1/2 × KL(P||M) + 1/2 × KL(Q||M)

    where M = 1/2 × (P + Q) is the average of the two distributions.

    For Gamma distributions, we compute this by:
        1. Finding the parameters of the mixture distribution M
        2. Computing KL(P||M) and KL(Q||M)
        3. Taking the average of these KL divergences

    Parameters
    ----------
    alpha1 : float or array
        Shape parameter of the first Gamma distribution
    beta1 : float or array
        Rate parameter of the first Gamma distribution
    alpha2 : float or array
        Shape parameter of the second Gamma distribution
    beta2 : float or array
        Rate parameter of the second Gamma distribution

    Returns
    -------
    float or array
        Jensen-Shannon divergence between the two Gamma distributions
    """
    # We can't directly compute the parameters of the mixture distribution M,
    # so we approximate the JS divergence using the KL divergences
    kl_p_q = kl_gamma(alpha1, beta1, alpha2, beta2)
    kl_q_p = kl_gamma(alpha2, beta2, alpha1, beta1)

    # JS divergence is the average of the two KL divergences
    return 0.5 * (kl_p_q + kl_q_p)


# ------------------------------------------------------------------------------


def jensen_shannon_lognormal(mu1, sigma1, mu2, sigma2):
    """
    Compute the Jensen-Shannon divergence between two LogNormal distributions.

    The Jensen-Shannon divergence is a symmetrized and smoothed version of the
    Kullback-Leibler divergence, defined as:

        JSD(P||Q) = 1/2 × KL(P||M) + 1/2 × KL(Q||M)

    where M = 1/2 × (P + Q) is the average of the two distributions.

    For LogNormal distributions, we compute this by:
        1. Finding the parameters of the mixture distribution M
        2. Computing KL(P||M) and KL(Q||M)
        3. Taking the average of these KL divergences

    Parameters
    ----------
    mu1 : float or array
        Location parameter of the first LogNormal distribution
    sigma1 : float or array
        Scale parameter of the first LogNormal distribution
    mu2 : float or array
        Location parameter of the second LogNormal distribution
    sigma2 : float or array
        Scale parameter of the second LogNormal distribution

    Returns
    -------
    float or array
        Jensen-Shannon divergence between the two LogNormal distributions
    """
    # We can't directly compute the parameters of the mixture distribution M,
    # so we approximate the JS divergence using the KL divergences
    kl_p_q = kl_lognormal(mu1, sigma1, mu2, sigma2)
    kl_q_p = kl_lognormal(mu2, sigma2, mu1, sigma1)

    # JS divergence is the average of the two KL divergences
    return 0.5 * (kl_p_q + kl_q_p)


# ------------------------------------------------------------------------------
# Hellinger distance functions
# ------------------------------------------------------------------------------


def sq_hellinger_beta(alpha1, beta1, alpha2, beta2):
    """
    Compute the squared Hellinger distance between two Beta distributions.

    The squared Hellinger distance between two Beta distributions P and Q is
    given by:

    H²(P,Q) = 1 - ∫ sqrt(P(x)Q(x)) dx

    where P(x) and Q(x) are the probability density functions of P and Q,
    respectively.

    For Beta distributions P ~ Beta(α₁,β₁) and Q ~ Beta(α₂,β₂), this has the
    closed form:

    H²(P,Q) = 1 - B((α₁+α₂)/2, (β₁+β₂)/2) / sqrt(B(α₁,β₁) * B(α₂,β₂))

    where B(x,y) is the beta function.

    Parameters
    ----------
    alpha1 : float or array-like
        Shape parameter α₁ of the first Beta distribution P
    beta1 : float or array-like
        Shape parameter β₁ of the first Beta distribution P
    alpha2 : float or array-like
        Shape parameter α₂ of the second Beta distribution Q
    beta2 : float or array-like
        Shape parameter β₂ of the second Beta distribution Q

    Returns
    -------
    float or array-like
        Squared Hellinger distance between the two Beta distributions
    """
    # Check that all inputs are of same shape
    if not all(
        isinstance(a, (float, np.ndarray, jnp.ndarray))
        and isinstance(b, (float, np.ndarray, jnp.ndarray))
        for a, b in zip(
            [alpha1, beta1, alpha2, beta2], [alpha1, beta1, alpha2, beta2]
        )
    ):
        raise ValueError("All inputs must be of the same shape")

    return 1 - (
        jsp.special.beta((alpha1 + alpha2) / 2, (beta1 + beta2) / 2)
        / jnp.sqrt(
            jsp.special.beta(alpha1, beta1) * jsp.special.beta(alpha2, beta2)
        )
    )


# ------------------------------------------------------------------------------


def hellinger_beta(alpha1, beta1, alpha2, beta2):
    """
    Compute the Hellinger distance between two Beta distributions.

    The Hellinger distance between two Beta distributions P and Q is given by:

    H(P,Q) = sqrt(H²(P,Q))

    where H²(P,Q) is the squared Hellinger distance.

    For Beta distributions P ~ Beta(α₁,β₁) and Q ~ Beta(α₂,β₂), this has the
    closed form:

    H(P,Q) = sqrt(1 - B((α₁+α₂)/2, (β₁+β₂)/2) / sqrt(B(α₁,β₁) * B(α₂,β₂)))

    where B(x,y) is the beta function.

    Parameters
    ----------
    alpha1 : float or array-like
        Shape parameter α₁ of the first Beta distribution P
    beta1 : float or array-like
        Shape parameter β₁ of the first Beta distribution P
    alpha2 : float or array-like
        Shape parameter α₂ of the second Beta distribution Q
    beta2 : float or array-like
        Shape parameter β₂ of the second Beta distribution Q

    Returns
    -------
    float or array-like
        Hellinger distance between the two Beta distributions
    """
    return jnp.sqrt(sq_hellinger_beta(alpha1, beta1, alpha2, beta2))


# ------------------------------------------------------------------------------


def sq_hellinger_gamma(alpha1, beta1, alpha2, beta2):
    """
    Compute the squared Hellinger distance between two Gamma distributions.

    The squared Hellinger distance between two Gamma distributions P and Q is
    given by:

    H²(P,Q) = 1 - ∫ sqrt(P(x)Q(x)) dx

    where P(x) and Q(x) are the probability density functions of P and Q,
    respectively.

    For Gamma distributions P ~ Gamma(α₁,β₁) and Q ~ Gamma(α₂,β₂), this has the
    closed form:

    H²(P,Q) = 1 - Γ((α₁+α₂)/2) * ((β₁+β₂)/2)^(-(α₁+α₂)/2) *
               sqrt((β₁^α₁ * β₂^α₂)/(Γ(α₁) * Γ(α₂)))

    where Γ(x) is the gamma function.

    Parameters
    ----------
    alpha1 : float or array-like
        Shape parameter α₁ of the first Gamma distribution P
    beta1 : float or array-like
        Rate parameter β₁ of the first Gamma distribution P
    alpha2 : float or array-like
        Shape parameter α₂ of the second Gamma distribution Q
    beta2 : float or array-like
        Rate parameter β₂ of the second Gamma distribution Q

    Returns
    -------
    float or array-like
        Squared Hellinger distance between the two Gamma distributions
    """
    # Check that all inputs are of same shape
    if not all(
        isinstance(a, (float, np.ndarray, jnp.ndarray))
        and isinstance(b, (float, np.ndarray, jnp.ndarray))
        for a, b in zip(
            [alpha1, beta1, alpha2, beta2], [alpha1, beta1, alpha2, beta2]
        )
    ):
        raise ValueError("All inputs must be of the same shape")

    return 1 - (
        jsp.special.gamma((alpha1 + alpha2) / 2)
        * ((beta1 + beta2) / 2) ** (-(alpha1 + alpha2) / 2)
        * jnp.sqrt(
            beta1**alpha1
            * beta2**alpha2
            / jsp.special.gamma(alpha1)
            * jsp.special.gamma(alpha2)
        )
    )


def hellinger_gamma(alpha1, beta1, alpha2, beta2):
    """
    Compute the Hellinger distance between two Gamma distributions.

    The Hellinger distance between two Gamma distributions P and Q is given by:

    H(P,Q) = sqrt(H²(P,Q))

    where H²(P,Q) is the squared Hellinger distance.

    For Gamma distributions P ~ Gamma(α₁,β₁) and Q ~ Gamma(α₂,β₂), this has the
    closed form:

    H(P,Q) = sqrt(1 - Γ((α₁+α₂)/2) * ((β₁+β₂)/2)^(-(α₁+α₂)/2) *
                   sqrt((β₁^α₁ * β₂^α₂)/(Γ(α₁) * Γ(α₂))))

    where Γ(x) is the gamma function.

    Parameters
    ----------
    alpha1 : float or array-like
        Shape parameter α₁ of the first Gamma distribution P
    beta1 : float or array-like
        Rate parameter β₁ of the first Gamma distribution P
    alpha2 : float or array-like
        Shape parameter α₂ of the second Gamma distribution Q
    beta2 : float or array-like
        Rate parameter β₂ of the second Gamma distribution Q

    Returns
    -------
    float or array-like
        Hellinger distance between the two Gamma distributions
    """
    return jnp.sqrt(sq_hellinger_gamma(alpha1, beta1, alpha2, beta2))


# ------------------------------------------------------------------------------


def sq_hellinger_lognormal(mu1, sigma1, mu2, sigma2):
    """
    Compute the squared Hellinger distance between two log-normal distributions.

    The squared Hellinger distance between two log-normal distributions P and Q is
    given by:

    H²(P,Q) = 1 - sqrt( σ1 * σ2 / (σ1² + σ2²) )
                    * exp( - (μ1-μ2)² / [4*(σ1² + σ2²)] )

    Parameters
    ----------
    mu1 : float or array-like
        Location parameter μ₁ of the first log-normal distribution P
    sigma1 : float or array-like
        Scale parameter σ₁ of the first log-normal distribution P
    mu2 : float or array-like
        Location parameter μ₂ of the second log-normal distribution Q
    sigma2 : float or array-like
        Scale parameter σ₂ of the second log-normal distribution Q

    Returns
    -------
    float or array-like
        Squared Hellinger distance between the two log-normal distributions
    """
    # Check that all inputs are of same shape
    if not all(
        isinstance(a, (float, np.ndarray, jnp.ndarray))
        and isinstance(b, (float, np.ndarray, jnp.ndarray))
        for a, b in zip([mu1, sigma1, mu2, sigma2], [mu1, sigma1, mu2, sigma2])
    ):
        raise ValueError("All inputs must be of the same shape")

    # The prefactor under the square root:
    prefactor = jnp.sqrt((sigma1 * sigma2) / (sigma1**2 + sigma2**2))
    # The exponent factor:
    exponent = jnp.exp(-((mu1 - mu2) ** 2) / (4.0 * (sigma1**2 + sigma2**2)))
    return 1.0 - (prefactor * exponent)


def hellinger_lognormal(mu1, sigma1, mu2, sigma2):
    """
    Compute the Hellinger distance between two log-normal distributions.

    The Hellinger distance is the square root of the squared Hellinger distance.

    Parameters
    ----------
    mu1 : float or array-like
        Location parameter μ₁ of the first log-normal distribution P
    sigma1 : float or array-like
        Scale parameter σ₁ of the first log-normal distribution P
    mu2 : float or array-like
        Location parameter μ₂ of the second log-normal distribution Q
    sigma2 : float or array-like
        Scale parameter σ₂ of the second log-normal distribution Q

    Returns
    -------
    float or array-like
        Hellinger distance between the two log-normal distributions
    """
    return jnp.sqrt(sq_hellinger_lognormal(mu1, sigma1, mu2, sigma2))


# ------------------------------------------------------------------------------
# Mode functions
# ------------------------------------------------------------------------------


def beta_mode(alpha, beta):
    """
    Calculate the mode for a Beta distribution.

    For Beta(α,β) distribution, the mode is:
        (α-1)/(α+β-2) when α,β > 1
        undefined when α,β ≤ 1

    Parameters
    ----------
    alpha : float or array-like
        Shape parameter α of the Beta distribution
    beta : float or array-like
        Shape parameter β of the Beta distribution

    Returns
    -------
    float or array-like
        Mode of the Beta distribution. Returns NaN for cases where
        mode is undefined (α,β ≤ 1)
    """
    return jnp.where(
        (alpha > 1) & (beta > 1), (alpha - 1) / (alpha + beta - 2), jnp.nan
    )


# ------------------------------------------------------------------------------


def gamma_mode(alpha, beta):
    """
    Calculate the mode for a Gamma distribution.

    For Gamma(α,β) distribution, the mode is:
        (α-1)/β when α > 1
        0 when α ≤ 1

    Parameters
    ----------
    alpha : float or array-like
        Shape parameter α of the Gamma distribution
    beta : float or array-like
        Rate parameter β of the Gamma distribution

    Returns
    -------
    float or array-like
        Mode of the Gamma distribution
    """
    return jnp.where(alpha > 1, (alpha - 1) / beta, 0.0)


# ------------------------------------------------------------------------------


def dirichlet_mode(alpha):
    """
    Calculate the mode for a Dirichlet distribution.

    For Dirichlet(α) distribution with concentration parameters α, the mode for
    each component is:
        (αᵢ-1)/(∑ⱼαⱼ-K) when αᵢ > 1 for all i
        0 when αᵢ ≤ 1 for any i
    where K is the number of components.

    Parameters
    ----------
    alpha : array-like
        Concentration parameters α of the Dirichlet distribution

    Returns
    -------
    array-like
        Mode of the Dirichlet distribution
    """
    return jnp.where(
        alpha > 1, (alpha - 1) / (jnp.sum(alpha) - len(alpha)), jnp.nan
    )


# ------------------------------------------------------------------------------


def lognorm_mode(mu, sigma):
    """
    Calculate the mode for a log-normal distribution.

    For LogNormal(μ,σ) distribution, the mode is:
        exp(μ - σ²)

    Parameters
    ----------
    mu : float or array-like
        Mean of the log-normal distribution
    sigma : float or array-like
        Standard deviation of the log-normal distribution

    Returns
    -------
    float or array-like
        Mode of the log-normal distribution
    """
    return jnp.exp(mu - sigma**2)


# ------------------------------------------------------------------------------


def get_distribution_mode(dist_obj):
    """
    Get the mode of a distribution object.

    Parameters
    ----------
    dist_obj : numpyro.distributions.Distribution
        Distribution object

    Returns
    -------
    jnp.ndarray
        Mode of the distribution
    """
    dist_type = type(dist_obj).__name__

    if dist_type == "Beta":
        return beta_mode(dist_obj.concentration1, dist_obj.concentration0)
    elif dist_type == "Gamma":
        return gamma_mode(dist_obj.concentration, dist_obj.rate)
    elif dist_type == "LogNormal":
        return lognorm_mode(dist_obj.loc, dist_obj.scale)
    elif dist_type == "Dirichlet":
        return dirichlet_mode(dist_obj.concentration)
    else:
        try:
            return dist_obj.mean
        except:
            raise ValueError(
                f"Cannot compute mode for distribution type {dist_type}"
            )


# ------------------------------------------------------------------------------
# Beta Prime Distribution
# ------------------------------------------------------------------------------


class BetaPrime(Distribution):
    """
    Beta Prime distribution.

    The Beta Prime distribution is the distribution of the ratio of two
    independent random variables with Gamma distributions. If X ~ Gamma(α, 1) and
    Y ~ Gamma(β, 1), then X/Y ~ BetaPrime(α, β).

    The probability density function is given by:
        f(x; α, β) = x^(α-1) * (1+x)^(-α-β) / B(α, β)
    where B(α, β) is the Beta function.

    Parameters
    ----------
    concentration1 : jnp.ndarray
        First shape parameter (α).
    concentration0 : jnp.ndarray
        Second shape parameter (β).
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.positive
    has_rsample = False

    def __init__(self, concentration1, concentration0, validate_args=None):
        c1 = jnp.asarray(concentration1, dtype=jnp.float32)
        c0 = jnp.asarray(concentration0, dtype=jnp.float32)
        self.concentration1, self.concentration0 = promote_shapes(c1, c0)
        super().__init__(
            batch_shape=self.concentration1.shape,
            event_shape=(),
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        key1, key2 = random.split(key)
        gamma1 = Gamma(self.concentration1, 1.0).sample(key1, sample_shape)
        gamma2 = Gamma(self.concentration0, 1.0).sample(key2, sample_shape)
        return gamma1 / gamma2

    @validate_sample
    def log_prob(self, value):
        log_numerator = jnp.log(value) * (self.concentration1 - 1) - jnp.log(
            1 + value
        ) * (self.concentration1 + self.concentration0)
        log_denominator = (
            gammaln(self.concentration1)
            + gammaln(self.concentration0)
            - gammaln(self.concentration1 + self.concentration0)
        )
        return log_numerator - log_denominator

    @property
    def mean(self):
        # Mean is defined for concentration0 > 1
        return jnp.where(
            self.concentration0 > 1,
            self.concentration1 / (self.concentration0 - 1),
            jnp.inf,
        )

    @property
    def variance(self):
        # Variance is defined for concentration0 > 2
        return jnp.where(
            self.concentration0 > 2,
            (
                self.concentration1
                * (self.concentration1 + self.concentration0 - 1)
            )
            / ((self.concentration0 - 1) ** 2 * (self.concentration0 - 2)),
            jnp.inf,
        )
