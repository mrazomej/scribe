import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random

# Set a random seed for reproducibility
SEED = 42

# Define a simple model: Normal distribution with unknown mean and known variance
def model():
    # Prior for the mean: standard normal
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    # Sample observations from the normal distribution
    numpyro.sample("obs", dist.Normal(mu, 1.0), obs=np.array([0.0, 1.0, 0.5, 0.1, -0.1]))

# Initialize random number generator
rng_key = random.PRNGKey(SEED)

# Set up the No-U-Turn Sampler (NUTS), which is an efficient variant of HMC
nuts_kernel = NUTS(model)

# Set up and run MCMC
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
mcmc.run(rng_key)

# Print summary of posterior samples
print("\nHello, NumPyro HMC World!")
print("=========================")
mcmc.print_summary()

# Extract and display the posterior samples
samples = mcmc.get_samples()
print("\nPosterior mean of 'mu':", np.mean(samples["mu"]))
print("Posterior std of 'mu':", np.std(samples["mu"]))

# If everything worked, you should see samples from the posterior distribution of mu
# which should be centered around the mean of the observed data (which is 0.3)
print("\nIf you see posterior samples and statistics above, HMC sampling is working correctly!")