# %% ---------------------------------------------------------------------------
from utils import matplotlib_style
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from functools import partial
import bayesflow as bf
import tensorflow as tf
# Set TensorFlow to use only the CPU
tf.config.set_visible_devices([], 'GPU')

cor, pal = matplotlib_style()

# Define number of observations
n_obs = 100
# %% ---------------------------------------------------------------------------

# Set the random number generator to be used
RNG = np.random.default_rng(42)
# %% ---------------------------------------------------------------------------


def prior_fn():
    # Prior on p_hat for negative binomial parameter
    p_hat = RNG.beta(1, 10)
    # Prior on r parameters
    r = RNG.gamma(3, 1/5)

    # Return the prior
    return np.float32(np.array([p_hat, r]))

# %% ---------------------------------------------------------------------------


# Define Prior simulator function for BayesFlow
param_names = [r"$\hat{p}$", r"$r$"]

prior = bf.simulation.Prior(
    prior_fun=prior_fn,
    param_names=param_names,
)

# %% ---------------------------------------------------------------------------


def likelihood_fn(params, n_obs=n_obs):
    # Unpack parameters
    p_hat = params[0]
    r = params[1]

    # Sample from the negative binomial distribution
    u = RNG.negative_binomial(r, p_hat, size=n_obs)

    # Add an extra dimension to make the output a 3D tensor
    u = np.expand_dims(u, axis=1)

    return np.float32(u)

# %% ---------------------------------------------------------------------------


# Define Likelihood simulator function for BayesFlow
simulator = bf.simulation.Simulator(simulator_fun=likelihood_fn)

# Build generative model
model = bf.simulation.GenerativeModel(prior, simulator)

# Draw samples from the generative model
model_draws = model(500)

# %% ---------------------------------------------------------------------------

# Plot the distribution of prior draws for p_hat and a few r parameters

# Define number of rows and columns
rows = 1
cols = 2

# Initialize figure
fig, ax = plt.subplots(rows, cols, figsize=(4, 2))

# Plot prior draws for p_hat on first plot
ax[0].hist(model_draws["prior_draws"][:, 0], bins=20, color=pal[0])
# Set x-axis label
ax[0].set_xlabel(param_names[0])

# Plot histogram of prior draws for r parameters
ax[1].hist(model_draws["prior_draws"][:, 1], bins=20, color=pal[0])
# Set x-axis label
ax[1].set_xlabel(param_names[1])

fig.tight_layout()

# %% ---------------------------------------------------------------------------

# Set up summary network

# Define summary network as a Deepset
summary_net = bf.networks.DeepSet(summary_dim=128)

# Simulate a pass through the summary network
summary_pass = summary_net(model_draws["sim_data"])

summary_pass.shape

# %% ---------------------------------------------------------------------------

# Define the conditional invertible network with affine coupling layers
inference_net = bf.inference_networks.InvertibleNetwork(
    num_params=prior(1)["prior_draws"].shape[-1],
)

# Perform a forward pass through the network given the summary network embedding
z, log_det_J = inference_net(model_draws['prior_draws'], summary_pass)

print(f"Latent variables: {z.numpy()}")
print(f"Log Jacobian Determinant: {log_det_J.numpy()}")

# %% ---------------------------------------------------------------------------

# Assemble the amoratizer that combines the summary network and inference
# network
amortizer = bf.amortizers.AmortizedPosterior(
    inference_net, summary_net,
)

# Assemble the trainer with the amortizer and generative model
trainer = bf.trainers.Trainer(
    amortizer=amortizer,
    generative_model=model,
    checkpoint_path=f"./negbinom_01dim_{n_obs}obs_bayesflow"
)

# %% ---------------------------------------------------------------------------

# Train the model
history = trainer.train_online(
    epochs=100,
    iterations_per_epoch=32,
    batch_size=128,
    validation_sims=128,
)
# %% ---------------------------------------------------------------------------

# Plot the training history
fig, ax = plt.subplots(1, 1, figsize=(3, 2))
ax.plot(history["train_losses"].Loss, 'k-')
# ax.plot(history["train_losses"]["Default.Loss"], 'k-')
# Seet x-axis label
ax.set_xlabel('training step', fontsize=6)
# Set y-axis label
ax.set_ylabel('training loss', fontsize=6)
# Set y-axis scale to log
# ax.set_yscale('log')
# %% ---------------------------------------------------------------------------

# Plot the training and validation losses
fig = bf.diagnostics.plot_losses(
    history["train_losses"],
    history["val_losses"],
    moving_average=True
)
# Get the axes
ax = fig.get_axes()
# ax[0].set_yscale('log')
ax[1].set_yscale('log')
fig.set_figwidth(6)
fig.set_figheight(4)

# %% ---------------------------------------------------------------------------

# Generate samples not seen during training
test_sims = trainer.configurator(model(500))

# %% ---------------------------------------------------------------------------

# Generate samples from latent variables by running data through model
z_samples, _ = amortizer(test_sims)
# Plot corner plot of latent variables
f = bf.diagnostics.plot_latent_space_2d(z_samples)

# %% ---------------------------------------------------------------------------

# Obtain posterior samples for each simulated data set in test_sims
posterior_samples = amortizer.sample(test_sims, n_samples=1000)

# Show simulation-based calibration histograms
f = bf.diagnostics.plot_sbc_histograms(
    posterior_samples, test_sims["parameters"], num_bins=10
)

# %% ---------------------------------------------------------------------------

# Plot fractional rank ECDFs
f = bf.diagnostics.plot_sbc_ecdf(
    posterior_samples, test_sims["parameters"], difference=True
)

# %% ---------------------------------------------------------------------------

# Plot parameter recovery
f = bf.diagnostics.plot_recovery(posterior_samples, test_sims["parameters"])

# %% ---------------------------------------------------------------------------

# Plot z-score contraction
f = bf.diagnostics.plot_z_score_contraction(
    posterior_samples, test_sims["parameters"]
)

# %% ---------------------------------------------------------------------------
