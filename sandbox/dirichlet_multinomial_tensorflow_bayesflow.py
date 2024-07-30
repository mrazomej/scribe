# %% ---------------------------------------------------------------------------
from utils import matplotlib_style
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import bayesflow as bf
import tensorflow as tf
import tensorflow_probability as tfp

cor, pal = matplotlib_style()

# %% ---------------------------------------------------------------------------

# Define number of genes
n_genes = 5_000

# %% ---------------------------------------------------------------------------

# Set the random number generator to be used
RNG = np.random.default_rng(42)

def prior_fn():
    """
    Generates prior parameter samples for the scRNA-seq model

    This function generates prior distributions for the parameters `p_hat` and
    `r_vec`. The `p_hat` parameter is drawn from a Beta(1, 1) distribution,
    representing the probability of success in a Bernoulli trial. The `r_vec`
    parameter is a vector of shape parameters for each gene, drawn from a
    log-normal distribution with mean -1 and standard deviation 1, and then
    sorted in ascending order.

    Returns:
        np.ndarray: A numpy array containing `p_hat` and `r_vec`. The first
        element is `p_hat`, a float representing the probability of success. The
        second element is `r_vec`, a sorted numpy array of shape parameters for
        each gene.

    Note:
        Ensure that `RNG` is an instance of a random number generator (e.g.,
        `numpy.random.default_rng()`) and `n_genes` is defined in the scope
        where this function is called.
    """
    # Prior on p_hat for negative binomial parameter
    p_hat = RNG.beta(1, 10)
    # Prior on r parameters for each gene
    r_vec = RNG.gamma(3, 1/5, n_genes)
    # Sort r_vec
    r_vec = np.sort(r_vec)[::-1]

    # Return the prior
    return np.float32(np.concatenate(([p_hat], r_vec)))

# %% ---------------------------------------------------------------------------


# Define Prior simulator function for BayesFlow
param_names = [r"$\hat{p}$"] + [f"$r_{{{i+1}}}$" for i in range(n_genes)]

prior = bf.simulation.Prior(
    prior_fun=prior_fn,
    param_names=param_names,
)

# Draw samples from the prior
prior(10)

# %% ---------------------------------------------------------------------------


def likelihood_fn(params, n_obs=10_000):
    """
    Computes the likelihood of the parameters given the Dirichlet-Multinomial
    model using TensorFlow Probability for GPU acceleration.
    
    Args:
        params (np.ndarray): Array containing the parameters. The first element
        is `p_hat`, which is the parameter for the Negative Binomial
        distribution. The remaining elements are `r_vec`, which are the
        parameters for the Dirichlet distribution.
        n_obs (int, optional): The number of observations to generate. Defaults to 10,000.
    
    Returns:
        np.ndarray: A sample from the Dirichlet-Multinomial distribution. The
        shape of the returned array is (n_obs, len(r_vec)).
    """
    # Ensure params is a TensorFlow tensor
    params = tf.convert_to_tensor(params, dtype=tf.float32)
    
    # Unpack parameters
    p_hat = params[0]
    r_vec = params[1:]
    
    # Sum of r parameters
    r_o = tf.reduce_sum(r_vec)
    
    # Sample total number of UMIs from Negative Binomial
    neg_binom = tfp.distributions.NegativeBinomial(total_count=r_o, probs=p_hat)
    U = neg_binom.sample(n_obs)
    
    # Sample individual UMI counts from Dirichlet-Multinomial
    dirichlet_multinomial = tfp.distributions.DirichletMultinomial(
        total_count=U, concentration=r_vec
    )
    u_vec = dirichlet_multinomial.sample()
    
    # Convert to numpy array and return
    return u_vec.numpy()

# %% ---------------------------------------------------------------------------


# Define Likelihood simulator function for BayesFlow
simulator = bf.simulation.Simulator(simulator_fun=likelihood_fn)

# Build generative model
model = bf.simulation.GenerativeModel(prior, simulator)
# %% ---------------------------------------------------------------------------

# Draw samples from the generative model
model_draws = model(500)
# %% ---------------------------------------------------------------------------

# Plot the distribution of prior draws for p_hat and a few r parameters

# Define number of rows and columns
rows = 3
cols = 3

# Initialize figure
fig, ax = plt.subplots(rows, cols, figsize=(4, 4))

# Plot prior draws for p_hat on first plot
ax[0, 0].hist(model_draws["prior_draws"][:, 0], bins=20, color=pal[0])
# Set x-axis label
ax[0, 0].set_xlabel(param_names[0])

# Select random indexes for r parameters
r_idx = np.sort(
    RNG.choice(np.arange(1, n_genes), size=(rows * cols) - 1, replace=False)
)

# Loop through the rest of the parameters
for i in range(rows * cols):
    # Calculate row and column index
    row = i // cols
    col = i % cols
    print(row, col)
    # Skip first plot
    if i == 0:
        continue

    # Plot histogram of prior draws for r parameters
    ax[row, col].hist(
        model_draws["prior_draws"][:, r_idx[i-1]], bins=20, color=pal[0]
    )
    # Set x-axis label
    ax[row, col].set_xlabel(param_names[r_idx[i-1]])

fig.tight_layout()
# %% ---------------------------------------------------------------------------

# Set up summary network

# Define summary network as a Deepset
summary_net = bf.networks.DeepSet(
    summary_dim=np.floor(n_genes * 1.5).astype(int)
)

# Define summary network as a SetTransformer
# summary_net = bf.networks.SetTransformer(
#     input_dim=n_genes,
#     summary_dim=np.floor(n_genes * 1.25).astype(int),
#     dense_settings={"units": 128, "activation": "relu"},
# )

# Simulate a pass through the summary network
summary_pass = summary_net(model_draws["sim_data"])

summary_pass.shape

# %% ---------------------------------------------------------------------------

# Define the conditional invertible network with affine coupling layers
inference_net = bf.inference_networks.InvertibleNetwork(
    num_params=prior(1)["prior_draws"].shape[-1],
    # num_coupling_layers=8,
    # coupling_design="affine",
    # permutation="learnable",
    # use_act_norm=True,
    # coupling_settings={
    #     "mc_dropout": True,
    #     "dense_args": dict(units=128, activation="elu"),
    # }
)

# Define conditional invertible network with spline coupling layers
# inference_net = bf.inference_networks.InvertibleNetwork(
#     num_params=prior(1)["prior_draws"].shape[-1],
#     coupling_design="spline",
#     coupling_settings={"dropout_prob": 0.2, "bins": 32, }
# )

# Perform a forward pass through the network given the summary network embedding
z, log_det_J = inference_net(model_draws['prior_draws'], summary_pass)

print(f"Latent variables: {z.numpy()}")
print(f"Log Jacobian Determinant: {log_det_J.numpy()}")
# %% ---------------------------------------------------------------------------

# Assemble the amoratizer that combines the summary network and inference
# network
amortizer = bf.amortizers.AmortizedPosterior(
    inference_net, summary_net,
    # set summary loss function to Maximum Mean Discrepancy
    # summary_loss_fun='MMD'
)

# Assemble the trainer with the amortizer and generative model
trainer = bf.trainers.Trainer(amortizer=amortizer, generative_model=model)
# %% ---------------------------------------------------------------------------

# Define number of epochs
n_epoch = 20
# Define number of iterations per epoch
n_iter = 128

# Define initial learning rate
# initial_learning_rate = 1e-6

# Set up the Adam optimizer with a fixed learning rate
# optimizer = tf.keras.optimizers.legacy.Adam(
#     learning_rate=initial_learning_rate
# )

# Define learning rate schedule. This is the same as the default optimizer in
# BayesFlow
# lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=n_epoch * n_iter,
#     alpha=0.0
# )
# Define optimizer
# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)

# Train the model
history = trainer.train_online(
    epochs=n_epoch,
    iterations_per_epoch=n_iter,
    batch_size=128,
    validation_sims=200,
    # optimizer=optimizer,
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

# Obtain 100 posterior samples for each simulated data set in test_sims
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
