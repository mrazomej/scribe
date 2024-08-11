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
    # Prior on k_on rate
    k_on = np.abs(rng.normal(1E-4, 10))
    # Prior on k_off rate
    k_off = np.abs(rng.normal(1E-4, 10))
    # Prior on rate of transcription r
    r = 1E-3 + np.abs(10 * rng.standard_cauchy())

    return np.float32(np.log(np.array([k_on, k_off, r])))


# %% ---------------------------------------------------------------------------

# Define parameter names
param_names = ['log_k_on', 'log_k_off', 'log_r']

# Define prior simulator
prior = bf.simulation.Prior(
    prior_fun=prior_fn,
    param_names=param_names,
)

# %% ---------------------------------------------------------------------------

print("Defining the likelihood function...")


def likelihood_fn(params, n_obs=10_000, m_range=range(0, 3_000)):
    # Unpack parameters
    k_on, k_off, r = np.exp(params)
    # Compute the log probability over m_range
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
    checkpoint_path="./two_state_log_bayesflow"
)

# Train the model
history = trainer.train_online(
    epochs=n_epoch,
    iterations_per_epoch=n_iter,
    batch_size=batch_size,
    validation_sims=n_val,
)
