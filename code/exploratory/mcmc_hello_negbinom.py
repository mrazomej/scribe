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

# Generate synthetic data from a negative binomial distribution with known parameters
def generate_data(rng_key, true_r=5.0, true_p=0.7, n_samples=500):
    # Convert to numpyro's parameterization
    # NB: NumPyro uses concentration (r) and rate (prob) parameterization
    neg_binom = dist.NegativeBinomialProbs(total_count=true_r, probs=true_p)
    
    # Generate samples
    data = neg_binom.sample(rng_key, sample_shape=(n_samples,))
    
    print(f"True parameters: r = {true_r}, p = {true_p}")
    print(f"Generated {n_samples} samples from Negative Binomial")
    print(f"Sample mean: {np.mean(np.array(data))}, variance: {np.var(np.array(data))}")
    
    # Theoretical mean and variance of negative binomial
    theo_mean = true_r * (1 - true_p) / true_p
    theo_var = true_r * (1 - true_p) / (true_p ** 2)
    print(f"Theoretical mean: {theo_mean}, variance: {theo_var}")
    
    return data

# Define the model for inference
def model(obs=None):
    # Priors for r and p
    # We use a gamma prior for r (concentration parameter)
    r = numpyro.sample("r", dist.Gamma(2.0, 0.5))
    
    # We use a beta prior for p (probability parameter)
    p = numpyro.sample("p", dist.Beta(2.0, 2.0))
    
    # Likelihood function
    with numpyro.plate("data", len(obs) if obs is not None else 1):
        numpyro.sample("obs", dist.NegativeBinomialProbs(total_count=r, probs=p), obs=obs)

# Main function to run the inference
def infer_negative_binomial_params():
    # Initialize random number generator
    rng_key = random.PRNGKey(SEED)
    rng_key, data_key = random.split(rng_key)
    
    # True parameters we want to recover
    true_r = 5.0
    true_p = 0.7
    
    # Generate synthetic data
    data = generate_data(data_key, true_r=true_r, true_p=true_p, n_samples=1000)
    
    # Set up the NUTS sampler
    nuts_kernel = NUTS(model, target_accept_prob=0.8)
    
    # Run MCMC
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)
    mcmc.run(rng_key, obs=data)
    
    # Print summary of posterior samples
    print("\nHello, NumPyro Negative Binomial World!")
    print("======================================")
    mcmc.print_summary()
    
    # Extract posterior samples
    samples = mcmc.get_samples()
    
    # Plot posterior distributions
    plt.figure(figsize=(12, 6))
    
    # Plot posterior for r
    plt.subplot(1, 2, 1)
    plt.hist(samples["r"], bins=30, alpha=0.7, density=True)
    plt.axvline(true_r, color='red', linestyle='--', label=f'True r = {true_r}')
    plt.title("Posterior Distribution for r")
    plt.xlabel("r (concentration)")
    plt.ylabel("Density")
    plt.legend()
    
    # Plot posterior for p
    plt.subplot(1, 2, 2)
    plt.hist(samples["p"], bins=30, alpha=0.7, density=True)
    plt.axvline(true_p, color='red', linestyle='--', label=f'True p = {true_p}')
    plt.title("Posterior Distribution for p")
    plt.xlabel("p (probability)")
    plt.ylabel("Density")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("negative_binomial_posterior.png")
    print("Posterior plot saved as 'negative_binomial_posterior.png'")
    
    # Print posterior statistics
    r_mean = np.mean(samples["r"])
    p_mean = np.mean(samples["p"])
    
    print(f"\nPosterior mean of r: {r_mean:.4f} (true: {true_r})")
    print(f"Posterior mean of p: {p_mean:.4f} (true: {true_p})")
    
    # Check if HMC recovered the true parameters
    r_within_ci = (np.percentile(samples["r"], 5) < true_r < np.percentile(samples["r"], 95))
    p_within_ci = (np.percentile(samples["p"], 5) < true_p < np.percentile(samples["p"], 95))
    
    if r_within_ci and p_within_ci:
        print("\nSuccess! Both true parameters are within the 90% credible intervals.")
        print("HMC sampling is working correctly!")
    else:
        print("\nWarning: One or more true parameters fall outside the 90% credible intervals.")
        print("This might be due to random variation or could indicate an issue with the sampling.")

if __name__ == "__main__":
    infer_negative_binomial_params()