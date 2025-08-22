"""
Standard parameterization low-rank guides for single-cell RNA sequencing data.
"""

# Import JAX-related libraries
import jax.numpy as jnp
import jax.nn

# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Import model config
from .model_config import ModelConfig


# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------


def nbdm_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Negative Binomial-Dirichlet
    Multinomial Model (NBDM) with a low-rank approximation for `r`.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)

    # Register p_alpha as a variational parameter with positivity constraint
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    # Register p_beta as a variational parameter with positivity constraint
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    # Sample p from the Beta distribution parameterized by p_alpha and p_beta
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Low-rank multivariate normal for r
    G = n_genes
    k = model_config.guide_rank

    # Define location
    loc = numpyro.param("log_r_loc", jnp.zeros(G))
    # Define covariance factor
    W = numpyro.param("log_r_W", 0.01 * jnp.ones((G, k)))
    # Define raw covariance diagonal
    raw = numpyro.param("log_r_raw_diag", -3.0 * jnp.ones(G))
    # Define covariance diagonal (strictly positive)
    D = jax.nn.softplus(raw) + 1e-4

    # Define base distribution
    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    # Define transformed distribution
    # Map ℝ^G -> (0,∞)^G to match the model's support for r
    exp_tr = dist.transforms.ExpTransform()
    r_dist = dist.TransformedDistribution(base, exp_tr)

    # Sample r from the transformed distribution
    numpyro.sample("r", r_dist)


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Model
# ------------------------------------------------------------------------------


def zinb_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Zero-Inflated Negative
    Binomial (ZINB) model with a low-rank approximation for `r`.
    """
    # Define guide parameters for p, r, and gate
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Low-rank multivariate normal for r
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("log_r_loc", jnp.zeros(G))
    W = numpyro.param("log_r_W", 0.01 * jnp.ones((G, k)))
    raw = numpyro.param("log_r_raw_diag", -3.0 * jnp.ones(G))
    D = jax.nn.softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    exp_tr = dist.transforms.ExpTransform()
    r_dist = dist.TransformedDistribution(base, exp_tr)
    numpyro.sample("r", r_dist)

    # Register variational parameters for gate (zero-inflation probability)
    gate_alpha = numpyro.param(
        "gate_alpha",
        jnp.full(n_genes, gate_prior_params[0]),
        constraint=constraints.positive,
    )
    gate_beta = numpyro.param(
        "gate_beta",
        jnp.full(n_genes, gate_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta))


# ------------------------------------------------------------------------------
# Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------


def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Negative Binomial with
    Variable Capture Probability (NBVCP) model with a low-rank approximation for `r`.
    """
    # Define guide parameters for p, r, and p_capture
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Low-rank multivariate normal for r
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("log_r_loc", jnp.zeros(G))
    W = numpyro.param("log_r_W", 0.01 * jnp.ones((G, k)))
    raw = numpyro.param("log_r_raw_diag", -3.0 * jnp.ones(G))
    D = jax.nn.softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    exp_tr = dist.transforms.ExpTransform()
    r_dist = dist.TransformedDistribution(base, exp_tr)
    numpyro.sample("r", r_dist)

    # Set up cell-specific capture probability parameters
    p_capture_alpha = numpyro.param(
        "p_capture_alpha",
        jnp.full(n_cells, p_capture_prior_params[0]),
        constraint=constraints.positive,
    )
    p_capture_beta = numpyro.param(
        "p_capture_beta",
        jnp.full(n_cells, p_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture", dist.Beta(p_capture_alpha, p_capture_beta)
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture",
                dist.Beta(p_capture_alpha[idx], p_capture_beta[idx]),
            )


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------


def zinbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Zero-Inflated Negative
    Binomial with Variable Capture Probability (ZINBVCP) model with a low-rank
    approximation for `r`.
    """
    # Define guide parameters for p, r, and gate
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Low-rank multivariate normal for r
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("log_r_loc", jnp.zeros(G))
    W = numpyro.param("log_r_W", 0.01 * jnp.ones((G, k)))
    raw = numpyro.param("log_r_raw_diag", -3.0 * jnp.ones(G))
    D = jax.nn.softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    exp_tr = dist.transforms.ExpTransform()
    r_dist = dist.TransformedDistribution(base, exp_tr)
    numpyro.sample("r", r_dist)

    # Register variational parameters for gate (zero-inflation probability)
    gate_alpha = numpyro.param(
        "gate_alpha",
        jnp.full(n_genes, gate_prior_params[0]),
        constraint=constraints.positive,
    )
    gate_beta = numpyro.param(
        "gate_beta",
        jnp.full(n_genes, gate_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta))

    # Set up cell-specific capture probability parameters
    p_capture_alpha = numpyro.param(
        "p_capture_alpha",
        jnp.full(n_cells, p_capture_prior_params[0]),
        constraint=constraints.positive,
    )
    p_capture_beta = numpyro.param(
        "p_capture_beta",
        jnp.full(n_cells, p_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture", dist.Beta(p_capture_alpha, p_capture_beta)
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture",
                dist.Beta(p_capture_alpha[idx], p_capture_beta[idx]),
            )


# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Mixture Model
# ------------------------------------------------------------------------------


def nbdm_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Negative Binomial-Dirichlet
    Multinomial (NBDM) mixture model with a low-rank approximation for `r`.
    """
    # Get the number of mixture components from the model configuration
    n_components = model_config.n_components

    # Get prior parameters for the mixture weights, p, and r
    if model_config.mixing_param_guide is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_guide)

    # Register variational parameters for the mixture weights
    mixing_conc = numpyro.param(
        "mixing_concentrations",
        mixing_prior_params,
        constraint=constraints.positive,
    )
    numpyro.sample("mixing_weights", dist.Dirichlet(mixing_conc))

    # Get prior parameters for p and r
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)

    # Low-rank multivariate normal for r
    G = n_genes
    k = model_config.guide_rank

    # Define location
    loc = numpyro.param("log_r_loc", jnp.zeros((n_components, G)))
    # Define covariance factor
    W = numpyro.param("log_r_W", 0.01 * jnp.ones((n_components, G, k)))
    # Define raw covariance diagonal
    raw = numpyro.param("log_r_raw_diag", -3.0 * jnp.ones((n_components, G)))
    # Define covariance diagonal (strictly positive)
    D = jax.nn.softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    exp_tr = dist.transforms.ExpTransform()
    r_dist = dist.TransformedDistribution(base, exp_tr)
    numpyro.sample("r", r_dist)

    if model_config.component_specific_params:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            jnp.full(n_components, p_prior_params[0]),
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    else:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            p_prior_params[0],
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------


def zinb_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Zero-Inflated Negative
    Binomial (ZINB) mixture model with a low-rank approximation for `r`.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_guide is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_guide)
    mixing_conc = numpyro.param(
        "mixing_concentrations",
        mixing_prior_params,
        constraint=constraints.positive,
    )
    numpyro.sample("mixing_weights", dist.Dirichlet(mixing_conc))

    # Get prior parameters for p, r, and gate
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)

    # Low-rank multivariate normal for r
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("log_r_loc", jnp.zeros((n_components, G)))
    W = numpyro.param("log_r_W", 0.01 * jnp.ones((n_components, G, k)))
    raw = numpyro.param("log_r_raw_diag", -3.0 * jnp.ones((n_components, G)))
    D = jnp.softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    exp_tr = dist.transforms.ExpTransform()
    r_dist = dist.TransformedDistribution(base, exp_tr).to_event(1)
    numpyro.sample("r", r_dist)

    # Define parameters for gate
    gate_alpha = numpyro.param(
        "gate_alpha",
        jnp.full((n_components, n_genes), gate_prior_params[0]),
        constraint=constraints.positive,
    )
    gate_beta = numpyro.param(
        "gate_beta",
        jnp.full((n_components, n_genes), gate_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific gate from a Beta prior
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta).to_event(1))

    if model_config.component_specific_params:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            jnp.full(n_components, p_prior_params[0]),
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    else:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            p_prior_params[0],
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))


# ------------------------------------------------------------------------------
# Negative Binomial Mixture Model with Variable Capture Probability
# ------------------------------------------------------------------------------


def nbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Negative Binomial with
    Variable Capture Probability (NBVCP) mixture model with a low-rank approximation
    for `r`.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_guide is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_guide)
    # Get prior parameters for p, r, and p_capture
    mixing_conc = numpyro.param(
        "mixing_concentrations",
        mixing_prior_params,
        constraint=constraints.positive,
    )
    # Sample the mixture weights from a Dirichlet prior
    numpyro.sample("mixing_weights", dist.Dirichlet(mixing_conc))

    # Get prior parameters for p, r, and p_capture
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Low-rank multivariate normal for r
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("log_r_loc", jnp.zeros((n_components, G)))
    W = numpyro.param("log_r_W", 0.01 * jnp.ones((n_components, G, k)))
    raw = numpyro.param("log_r_raw_diag", -3.0 * jnp.ones((n_components, G)))
    D = jnp.softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    exp_tr = dist.transforms.ExpTransform()
    r_dist = dist.TransformedDistribution(base, exp_tr).to_event(1)
    numpyro.sample("r", r_dist)

    if model_config.component_specific_params:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            jnp.full(n_components, p_prior_params[0]),
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    else:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            p_prior_params[0],
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Set up cell-specific capture probability parameters
    p_capture_alpha = numpyro.param(
        "p_capture_alpha",
        jnp.full(n_cells, p_capture_prior_params[0]),
        constraint=constraints.positive,
    )
    p_capture_beta = numpyro.param(
        "p_capture_beta",
        jnp.full(n_cells, p_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture", dist.Beta(p_capture_alpha, p_capture_beta)
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture",
                dist.Beta(p_capture_alpha[idx], p_capture_beta[idx]),
            )


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model with Variable Capture Probability
# ------------------------------------------------------------------------------


def zinbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Zero-Inflated Negative
    Binomial with Variable Capture Probability (ZINBVCP) mixture model
    with a low-rank approximation for `r`.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_guide is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_guide)
    # Get prior parameters for p, r, gate, and p_capture
    mixing_conc = numpyro.param(
        "mixing_concentrations",
        mixing_prior_params,
        constraint=constraints.positive,
    )
    numpyro.sample("mixing_weights", dist.Dirichlet(mixing_conc))

    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Low-rank multivariate normal for r
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("log_r_loc", jnp.zeros((n_components, G)))
    W = numpyro.param("log_r_W", 0.01 * jnp.ones((n_components, G, k)))
    raw = numpyro.param("log_r_raw_diag", -3.0 * jnp.ones((n_components, G)))
    D = jnp.softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    exp_tr = dist.transforms.ExpTransform()
    r_dist = dist.TransformedDistribution(base, exp_tr).to_event(1)
    numpyro.sample("r", r_dist)

    # Define parameters for gate
    gate_alpha = numpyro.param(
        "gate_alpha",
        jnp.full((n_components, n_genes), gate_prior_params[0]),
        constraint=constraints.positive,
    )
    gate_beta = numpyro.param(
        "gate_beta",
        jnp.full((n_components, n_genes), gate_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific gate from a Beta prior
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta).to_event(1))

    # Define parameters for p
    if model_config.component_specific_params:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            jnp.full(n_components, p_prior_params[0]),
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    else:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            p_prior_params[0],
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Set up cell-specific capture probability parameters
    p_capture_alpha = numpyro.param(
        "p_capture_alpha",
        jnp.full(n_cells, p_capture_prior_params[0]),
        constraint=constraints.positive,
    )
    p_capture_beta = numpyro.param(
        "p_capture_beta",
        jnp.full(n_cells, p_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture", dist.Beta(p_capture_alpha, p_capture_beta)
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture",
                dist.Beta(p_capture_alpha[idx], p_capture_beta[idx]),
            )
