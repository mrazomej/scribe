# %% ---------------------------------------------------------------------------
import pymc as pm
import pytensor.tensor as pt
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
# %% ---------------------------------------------------------------------------

# Define number of genes and range of number of cells
n_genes = 10

# Define number of cells
n_cells = 10
# %% ---------------------------------------------------------------------------

# Initialize random number generator
RNG = np.random.default_rng(42)

# Sample ground truth parameters
p_hat_true = RNG.beta(1, 10)
r_vec_true = RNG.lognormal(-1, 1, n_genes)

# Sample observed counts
U_true = RNG.negative_binomial(r_vec_true.sum(), p_hat_true, size=n_cells)

# Sample dirichlet parameters for each cell
dirichlet_probs = RNG.dirichlet(r_vec_true, size=n_cells)

# Sample observed counts for each gene
u_vec_true = np.array(
    [RNG.multinomial(U_true[i], dirichlet_probs[i]) for i in range(n_cells)]
)

# Define observed counts
total_counts = U_true
observed_counts = u_vec_true
# %% ---------------------------------------------------------------------------

scrnaseq_model = pm.Model()

with scrnaseq_model:
    # Prior on p_hat for negative binomial parameter
    p_hat = pm.Beta("p_hat", alpha=1, beta=10)

    # Prior on r parameters for each gene
    r_vec = pm.LogNormal("r_vec", mu=-1, sigma=1, shape=n_genes)

    # Sum of r parameters
    r_o = pm.math.sum(r_vec)

    # Likelihood for Total observed counts
    U = pm.NegativeBinomial("U", mu=r_o, alpha=p_hat, observed=total_counts)

    # Use Dirichlet-Multinomial distribution for observed counts
    u_vec = pm.DirichletMultinomial(
        "counts", n=U, a=r_vec, observed=observed_counts
    )

# %% ---------------------------------------------------------------------------

# Perform ADVI
with scrnaseq_model:
    approx = pm.fit(
        100_000,
        callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)],
        method='advi'
    )

# %% ---------------------------------------------------------------------------

# Extract posterior samples
posterior_samples = approx.sample(1000)

# Plot the distribution of posterior draws for p_hat and a few r parameters
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot posterior draws for p_hat
az.plot_posterior(posterior_samples, var_names=["p_hat"], ax=axes[0, 0])


# Plot posterior draws for r_vec
for i in range(1, 4):
    az.plot_posterior(
        posterior_samples,
        var_names=["r_vec"],
        ax=axes[i//2, i % 2]
    )

plt.tight_layout()
plt.show()

# %% ---------------------------------------------------------------------------

# Print summary of the posterior
az.summary(posterior_samples, var_names=["p_hat", "r_vec"])

# %% ---------------------------------------------------------------------------
