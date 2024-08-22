# %% ---------------------------------------------------------------------------
import utils
import numpy as np
import math
import scipy
import mpmath
import matplotlib.pyplot as plt
import seaborn as sns
import bayesflow as bf
import tensorflow as tf
# Set TensorFlow to use only the CPU
tf.config.set_visible_devices([], 'GPU')

# Set random seed
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# %% ---------------------------------------------------------------------------

print("Defining the prior function...")


def prior_fn():
    # Prior on log_k_on rate
    log_k_on = rng.uniform(np.log(1E-4), np.log(100))
    # Prior on log_r
    log_r = rng.uniform(np.log(1E-4), np.log(10_000))
    # Prior on log(r / k_off)
    log_r_k_off = rng.uniform(np.log(1E-4), np.log(10))

    return np.float32(np.array([log_k_on, log_r, log_r_k_off]))


# %% ---------------------------------------------------------------------------

# Define parameter names
param_names = ['log_k_on', 'log_r', 'log_r_k_off']

# Define prior simulator
prior = bf.simulation.Prior(
    prior_fun=prior_fn,
    param_names=param_names,
)

# %% ---------------------------------------------------------------------------

print("Defining the likelihood function...")


def likelihood_fn(params, n_obs=10_000, m_range=range(0, 3_000)):
    # Unpack parameters
    log_k_on, log_r, log_r_k_off = params
    # Exponentiate parameters
    k_on = np.exp(log_k_on)
    r = np.exp(log_r)
    k_off = np.exp(log_r - log_r_k_off)
    # Check if k_off is at least 20 times larger than k_on if so use
    # negative binomial distribution
    if k_off >= 20 * k_on:
        logP = utils.two_state_neg_binom_log_probability(
            m_range, k_on, k_off, r
        )
        # If k_off is not at least 20 times larger than k_on use the two-state
        # log probability
    else:
        logP = utils.two_state_log_probability(m_range, k_on, k_off, r)
    # Convert log probabilities to probabilities
    P = np.exp(logP)
    # Normalize the probabilities to use as weights. This is necessary because
    # of numerical precision issues.
    P /= P.sum()
    # Generate random samples using these weights
    u = np.random.choice(m_range, size=n_obs, p=P)
    # Add a 3rd dimension to the array to make output 3D tensor
    u = np.expand_dims(u, axis=1)
    # Return the samples as float32
    return np.float32(u)

# %% ---------------------------------------------------------------------------


print("Defining the generative model...")

# Define Likelihood simulator function for BayesFlow
simulator = bf.simulation.Simulator(simulator_fun=likelihood_fn)

# Build generative model
model = bf.simulation.GenerativeModel(prior, simulator)

# Define summary network as a Deepset
summary_net = bf.networks.DeepSet(summary_dim=32)

# Define the conditional invertible network with affine coupling layers
inference_net = bf.inference_networks.InvertibleNetwork(
    num_params=prior(1)["prior_draws"].shape[-1],
)

# %% ---------------------------------------------------------------------------

print("Training the model...")

# Define the number of epochs
n_epoch = 10
# Define the number of iterations per epoch
n_iter = 128
# Define the batch size
batch_size = 128
# Define number of validation simulations
n_val = 256

# Assemble the amoratizer that combines the summary network and inference
# network
amortizer = bf.amortizers.AmortizedPosterior(
    inference_net, summary_net,
)

# Assemble the trainer with the amortizer and generative model
trainer = bf.trainers.Trainer(
    amortizer=amortizer,
    generative_model=model,
    checkpoint_path="./two_state_log_ratio_bayesflow"
)

# Train the model
history = trainer.train_online(
    epochs=n_epoch,
    iterations_per_epoch=n_iter,
    batch_size=batch_size,
    validation_sims=n_val,
)
