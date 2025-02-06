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

# ------------------------------------------------------------------------------
# Histogram functions
# ------------------------------------------------------------------------------

def compute_histogram_percentiles(
    samples,
    percentiles=[5, 25, 50, 75, 95],
    normalize=True,
    sample_axis=0
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
    bin_edges = np.arange(global_min, global_max + 2)  # +2 because we want right edge
    
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
    max_bin=None
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
    global_max = (min(int(samples.max()), max_bin) 
                 if max_bin is not None 
                 else int(samples.max()))
    
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
    results = {
        'bin_edges': np.array(bin_edges),
        'regions': {}
    }
    
    median = np.percentile(all_hists, 50, axis=0)
    
    for cr in credible_regions:
        lower_percentile = (100 - cr) / 2
        upper_percentile = 100 - lower_percentile
        
        results['regions'][cr] = {
            'lower': np.percentile(all_hists, lower_percentile, axis=0),
            'upper': np.percentile(all_hists, upper_percentile, axis=0),
            'median': median
        }
    
    return results

# ------------------------------------------------------------------------------
# Fraction of transcriptome functions
# ------------------------------------------------------------------------------

def sample_dirichlet_from_parameters(
    parameter_samples: Union[np.ndarray, jnp.ndarray],
    n_samples_dirichlet: int = 1,
    rng_key: random.PRNGKey = random.PRNGKey(42)
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
        rng_key,
        sample_shape=(n_samples_dirichlet,)
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
    sample_axis: int = 0
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
    alpha = jnp.mean(x, axis=0) * (
        jnp.mean(x * (1 - x), axis=0) / jnp.var(x, axis=0)
    ).mean()
    
    # Compute mean of log samples - this is a sufficient statistic for the MLE
    mean_log_x = jnp.mean(jnp.log(x), axis=0)
    
    def iteration_step(alpha):
        """Single iteration of Newton's method"""
        # Compute sum of current alpha values
        alpha_sum = jnp.sum(alpha)
        
        # Compute gradient using digamma functions
        grad = mean_log_x - (jsp.special.digamma(alpha) - jsp.special.digamma(alpha_sum))
        
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
        + alpha1 * (beta2/beta1 - 1)
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
    if not all(isinstance(a, (float, np.ndarray, jnp.ndarray)) and 
               isinstance(b, (float, np.ndarray, jnp.ndarray)) 
               for a, b in zip([alpha1, beta1, alpha2, beta2], [alpha1, beta1, alpha2, beta2])
            ):
        raise ValueError("All inputs must be of the same shape")
    
    return (
        gammaln(alpha2 + beta2) - gammaln(alpha2) - gammaln(beta2)
        - (gammaln(alpha1 + beta1) - gammaln(alpha1) - gammaln(beta1))
        + (alpha1 - alpha2) * digamma(alpha1)
        + (beta1 - beta2) * digamma(beta1)
        + (alpha2 - alpha1 + beta2 - beta1) * digamma(alpha1 + beta1)
    )


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
    if not all(isinstance(a, (float, np.ndarray, jnp.ndarray)) and 
               isinstance(b, (float, np.ndarray, jnp.ndarray)) 
               for a, b in zip([alpha1, beta1, alpha2, beta2], [alpha1, beta1, alpha2, beta2])
            ):
        raise ValueError("All inputs must be of the same shape")
    
    return 1 - (
        jsp.special.beta((alpha1 + alpha2) / 2, (beta1 + beta2) / 2) /
        jnp.sqrt(
            jsp.special.beta(alpha1, beta1) * jsp.special.beta(alpha2, beta2)
        )
    )

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
    return 1 - (
        jsp.special.gamma((alpha1 + alpha2) / 2) *
        ((beta1 + beta2) / 2) ** (- (alpha1 + alpha2) / 2) *
        jnp.sqrt(
            beta1 ** alpha1 * beta2 ** alpha2 /
            jsp.special.gamma(alpha1) * jsp.special.gamma(alpha2)
        )
    )