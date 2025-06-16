import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
SEED = 42

def generate_data(rng_key, true_rs, true_p, n_samples=500):
    """
    Generate multidimensional synthetic data where each dimension follows
    a negative binomial distribution with different r but shared p parameter.
    
    Args:
        rng_key: JAX random key
        true_rs: Array of r parameters (total_count) for each dimension
        true_p: Shared probability parameter across all dimensions
        n_samples: Number of multivariate samples to generate
        
    Returns:
        data: Array of shape (n_samples, num_dimensions) containing the observations
    """
    num_dimensions = len(true_rs)
    
    # Use vectorized sampling for all dimensions at once
    true_rs_array = jnp.array(true_rs)
    
    # Create expanded distribution with a dimension for each r
    neg_binom = dist.NegativeBinomialProbs(total_count=true_rs_array, probs=true_p).to_event(1)
    
    # Generate samples
    data = neg_binom.sample(rng_key, sample_shape=(n_samples,))
    data_np = np.array(data)  # Convert to numpy for printing and analysis
    
    # Print true parameters
    print("True parameters:")
    for i, r in enumerate(true_rs):
        print(f"Dimension {i+1}: r = {r}")
    print(f"Shared p = {true_p}")
    
    # Compute and print empirical statistics
    print("\nEmpirical statistics:")
    for i in range(num_dimensions):
        print(f"Dimension {i+1}:")
        print(f"  Mean: {np.mean(data_np[:, i]):.2f}")
        print(f"  Variance: {np.var(data_np[:, i]):.2f}")
        
        # Theoretical mean and variance of negative binomial
        theo_mean = true_rs[i] * (1 - true_p) / true_p
        theo_var = true_rs[i] * (1 - true_p) / (true_p ** 2)
        print(f"  Theoretical mean: {theo_mean:.2f}, variance: {theo_var:.2f}")
    
    return data_np

def model(obs=None, num_dimensions=3):
    """
    Vectorized model for multivariate data where each dimension follows a negative binomial
    with different r parameters but a shared p parameter.
    
    Args:
        obs: Observed data of shape (n_samples, num_dimensions)
        num_dimensions: Number of dimensions in the multivariate distribution
    """
    # Shared p parameter with a Beta prior
    p = numpyro.sample("p", dist.Beta(2.0, 2.0))
    
    # Sample r parameters for each dimension from Gamma priors using a plate
    r = numpyro.sample("r", dist.Gamma(2.0, 0.5).expand([num_dimensions]))
    
    # Vectorized likelihood using expand and to_event
    with numpyro.plate("observations", obs.shape[0] if obs is not None else 1):
        # Create a multivariate negative binomial distribution
        nb_dist = dist.NegativeBinomialProbs(total_count=r, probs=p).to_event(1)
        
        # Sample observations from the multivariate distribution
        numpyro.sample("obs", nb_dist, obs=obs)

def run_inference(data, num_warmup=1000, num_samples=2000):
    """
    Run HMC inference to recover the parameters.
    
    Args:
        data: Array of observed data with shape (n_samples, num_dimensions)
        num_warmup: Number of warmup samples for MCMC
        num_samples: Number of posterior samples to collect
        
    Returns:
        dict of posterior samples
    """
    # Initialize random number generator
    rng_key = random.PRNGKey(SEED)
    
    # Get the number of dimensions from the data
    num_dimensions = data.shape[1]
    
    # Set up the NUTS sampler
    nuts_kernel = NUTS(model, target_accept_prob=0.8)
    
    # Run MCMC
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, obs=jnp.array(data), num_dimensions=num_dimensions)
    
    # Print summary of posterior samples
    print("\nInference Results:")
    print("=================")
    mcmc.print_summary()
    
    return mcmc.get_samples()

def visualize_results(samples, true_rs, true_p):
    """
    Visualize the posterior distributions of parameters.
    
    Args:
        samples: Dict of posterior samples
        true_rs: List of true r parameters for each dimension
        true_p: True shared p parameter
    """
    num_dimensions = len(true_rs)
    
    # Extract the r samples (this will be a matrix of shape [num_samples, num_dimensions])
    r_samples = samples["r"]
    
    # Extract p samples
    p_samples = samples["p"]
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Determine a good grid layout based on number of dimensions
    total_plots = num_dimensions + 1  # +1 for the p parameter
    grid_cols = min(3, total_plots)
    grid_rows = (total_plots + grid_cols - 1) // grid_cols
    
    # Plot posterior for p
    plt.subplot(grid_rows, grid_cols, 1)
    plt.hist(p_samples, bins=30, alpha=0.7, density=True)
    plt.axvline(true_p, color='red', linestyle='--', label=f'True p = {true_p}')
    plt.title("Posterior Distribution for Shared p")
    plt.xlabel("p (probability)")
    plt.ylabel("Density")
    plt.legend()
    
    # Plot posteriors for each r
    for i in range(num_dimensions):
        plt.subplot(grid_rows, grid_cols, i + 2)
        plt.hist(r_samples[:, i], bins=30, alpha=0.7, density=True)
        plt.axvline(true_rs[i], color='red', linestyle='--', label=f'True r = {true_rs[i]}')
        plt.title(f"Posterior Distribution for r_{i}")
        plt.xlabel(f"r_{i} (total count)")
        plt.ylabel("Density")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("shared_p_negative_binomial_posterior.png")
    print("Posterior plot saved as 'shared_p_negative_binomial_posterior.png'")
    
    # Print posterior statistics and check coverage
    print("\nPosterior Statistics:")
    
    # Check shared p parameter
    p_mean = np.mean(p_samples)
    p_5, p_95 = np.percentile(p_samples, [5, 95])
    p_within_ci = p_5 < true_p < p_95
    
    print(f"Shared p:")
    print(f"  True value: {true_p}")
    print(f"  Posterior mean: {p_mean:.4f}")
    print(f"  90% CI: [{p_5:.4f}, {p_95:.4f}]")
    print(f"  In 90% CI: {p_within_ci}")
    
    # Check r parameters for each dimension
    all_within_ci = p_within_ci
    for i in range(num_dimensions):
        r_dim_samples = r_samples[:, i]
        r_mean = np.mean(r_dim_samples)
        r_5, r_95 = np.percentile(r_dim_samples, [5, 95])
        r_within_ci = r_5 < true_rs[i] < r_95
        all_within_ci = all_within_ci and r_within_ci
        
        print(f"r_{i}:")
        print(f"  True value: {true_rs[i]}")
        print(f"  Posterior mean: {r_mean:.4f}")
        print(f"  90% CI: [{r_5:.4f}, {r_95:.4f}]")
        print(f"  In 90% CI: {r_within_ci}")
    
    if all_within_ci:
        print("\nSuccess! All true parameters are within their 90% credible intervals.")
        print("HMC sampling is working correctly!")
    else:
        print("\nWarning: One or more true parameters fall outside their 90% credible intervals.")
        print("This might be due to random variation or could indicate an issue with the sampling.")

def main(num_dimensions=3, n_samples=500):
    """
    Main function to run the entire example.
    
    Args:
        num_dimensions: Number of dimensions (default: 3)
        n_samples: Number of samples to generate (default: 500)
    """
    # Set the random seed
    rng_key = random.PRNGKey(SEED)
    
    # Define true parameters (can be easily expanded to more dimensions)
    true_rs = jnp.array([3.0, 5.0, 8.0, 12.0, 15.0][:num_dimensions])  # Different r for each dimension
    true_p = 0.7  # Shared p parameter
    
    print(f"Running example with {num_dimensions} dimensions and {n_samples} samples")
    
    # Generate synthetic data
    data = generate_data(rng_key, true_rs, true_p, n_samples=n_samples)
    
    # Run inference
    samples = run_inference(data, num_warmup=1000, num_samples=2000)
    
    # Visualize results
    visualize_results(samples, true_rs, true_p)

if __name__ == "__main__":
    # Can adjust number of dimensions and samples here
    main(num_dimensions=5, n_samples=500)