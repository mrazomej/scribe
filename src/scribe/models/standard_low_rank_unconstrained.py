"""
Standard parameterization low-rank guides for single-cell RNA sequencing data.
"""

# Import JAX-related libraries
import jax.numpy as jnp
from jax.nn import softplus

# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Import typing
from typing import Dict, Optional

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
    Multinomial Model (NBDM) with a low-rank approximation for the unconstrained
    r parameter.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_guide or (0.0, 1.0)

    # Register p_unconstrained as a variational parameter
    p_loc = numpyro.param("p_unconstrained_loc", p_prior_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_prior_params[1],
        constraint=constraints.positive,
    )
    # Sample p_unconstrained from the Normal distribution
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Low-rank multivariate normal for r_unconstrained
    G = n_genes
    k = model_config.guide_rank

    # Define location
    loc = numpyro.param("r_unconstrained_loc", jnp.zeros(G))
    # Define covariance factor
    W = numpyro.param("r_unconstrained_W", 0.01 * jnp.ones((G, k)))
    # Define raw covariance diagonal
    raw = numpyro.param("r_unconstrained_raw_diag", -3.0 * jnp.ones(G))
    # Define covariance diagonal (strictly positive)
    D = softplus(raw) + 1e-4

    # Define base distribution
    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    # Sample r_unconstrained from the low-rank multivariate normal
    numpyro.sample("r_unconstrained", base)


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
    Binomial (ZINB) model with a low-rank approximation for the unconstrained
    r parameter.
    """
    # Define guide parameters for p, mu, and gate
    p_prior_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_guide or (0.0, 1.0)

    # Register variational parameters for p_unconstrained
    p_loc = numpyro.param("p_unconstrained_loc", p_prior_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_prior_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Low-rank multivariate normal for r_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("r_unconstrained_loc", jnp.zeros(G))
    W = numpyro.param("r_unconstrained_W", 0.01 * jnp.ones((G, k)))
    raw = numpyro.param("r_unconstrained_raw_diag", -3.0 * jnp.ones(G))
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("r_unconstrained", base)

    # Register variational parameters for gate_unconstrained
    gate_loc = numpyro.param(
        "gate_unconstrained_loc",
        jnp.full(n_genes, gate_prior_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full(n_genes, gate_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))


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
    Variable Capture Probability (NBVCP) model with a low-rank approximation for
    the unconstrained r parameter.
    """
    # Define guide parameters for p, mu, and p_capture
    p_prior_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register variational parameters for p_unconstrained
    p_loc = numpyro.param("p_unconstrained_loc", p_prior_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_prior_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Low-rank multivariate normal for r_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("r_unconstrained_loc", jnp.zeros(G))
    W = numpyro.param("r_unconstrained_W", 0.01 * jnp.ones((G, k)))
    raw = numpyro.param("r_unconstrained_raw_diag", -3.0 * jnp.ones(G))
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("r_unconstrained", base)

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_prior_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture_unconstrained depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc, p_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc[idx], p_capture_scale[idx]),
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
    approximation for the unconstrained r parameter.
    """
    # Define guide parameters for p, mu, and gate
    p_prior_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_guide or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register variational parameters for p_unconstrained
    p_loc = numpyro.param("p_unconstrained_loc", p_prior_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_prior_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Low-rank multivariate normal for r_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("r_unconstrained_loc", jnp.zeros(G))
    W = numpyro.param("r_unconstrained_W", 0.01 * jnp.ones((G, k)))
    raw = numpyro.param("r_unconstrained_raw_diag", -3.0 * jnp.ones(G))
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("r_unconstrained", base)

    # Register variational parameters for gate_unconstrained
    gate_loc = numpyro.param(
        "gate_unconstrained_loc",
        jnp.full(n_genes, gate_prior_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full(n_genes, gate_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_prior_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture_unconstrained depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc, p_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc[idx], p_capture_scale[idx]),
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
    Multinomial (NBDM) mixture model with a low-rank approximation for the
    unconstrained r parameter.
    """
    # Get the number of mixture components from the model configuration
    n_components = model_config.n_components

    # Get prior parameters for the mixture weights, p, and mu
    if model_config.mixing_logits_unconstrained_guide is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_guide

    # Register variational parameters for the mixture weights
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_prior_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained", dist.Normal(mixing_loc, mixing_scale)
    )

    # Get prior parameters for p and mu
    p_prior_params = model_config.p_unconstrained_guide or (0.0, 1.0)

    # Low-rank multivariate normal for r_unconstrained
    G = n_genes
    k = model_config.guide_rank

    # Define location
    loc = numpyro.param("r_unconstrained_loc", jnp.zeros((n_components, G)))
    # Define covariance factor
    W = numpyro.param(
        "r_unconstrained_W", 0.01 * jnp.ones((n_components, G, k))
    )
    # Define raw covariance diagonal
    raw = numpyro.param(
        "r_unconstrained_raw_diag", -3.0 * jnp.ones((n_components, G))
    )
    # Define covariance diagonal (strictly positive)
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("r_unconstrained", base.to_event(1))

    if model_config.component_specific_params:
        # Define parameters for p_unconstrained
        p_loc = numpyro.param(
            "p_unconstrained_loc",
            jnp.full(n_components, p_prior_params[0]),
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p_unconstrained
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    else:
        # Define parameters for p_unconstrained
        p_loc = numpyro.param(
            "p_unconstrained_loc",
            p_prior_params[0],
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))


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
    Binomial (ZINB) mixture model with a low-rank approximation for the
    unconstrained r parameter.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_logits_unconstrained_guide is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_guide

    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_prior_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained", dist.Normal(mixing_loc, mixing_scale)
    )

    # Get prior parameters for p, mu, and gate
    p_prior_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_guide or (0.0, 1.0)

    # Low-rank multivariate normal for r_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("r_unconstrained_loc", jnp.zeros((n_components, G)))
    W = numpyro.param(
        "r_unconstrained_W", 0.01 * jnp.ones((n_components, G, k))
    )
    raw = numpyro.param(
        "r_unconstrained_raw_diag", -3.0 * jnp.ones((n_components, G))
    )
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("r_unconstrained", base.to_event(1))

    # Define parameters for gate_unconstrained
    gate_loc = numpyro.param(
        "gate_unconstrained_loc",
        jnp.full((n_components, n_genes), gate_prior_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full((n_components, n_genes), gate_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific gate_unconstrained from a Normal prior
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))

    if model_config.component_specific_params:
        # Define parameters for p_unconstrained
        p_loc = numpyro.param(
            "p_unconstrained_loc",
            jnp.full(n_components, p_prior_params[0]),
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p_unconstrained
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    else:
        # Define parameters for p_unconstrained
        p_loc = numpyro.param(
            "p_unconstrained_loc",
            p_prior_params[0],
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))


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
    Variable Capture Probability (NBVCP) mixture model with a low-rank
    approximation for the unconstrained r parameter.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_logits_unconstrained_guide is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_guide

    # Get prior parameters for p, mu, and p_capture
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_prior_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the mixture weights from a Normal prior
    numpyro.sample(
        "mixing_logits_unconstrained", dist.Normal(mixing_loc, mixing_scale)
    )

    # Get prior parameters for p, mu, and p_capture
    p_prior_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Low-rank multivariate normal for r_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("r_unconstrained_loc", jnp.zeros((n_components, G)))
    W = numpyro.param(
        "r_unconstrained_W", 0.01 * jnp.ones((n_components, G, k))
    )
    raw = numpyro.param(
        "r_unconstrained_raw_diag", -3.0 * jnp.ones((n_components, G))
    )
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("r_unconstrained", base.to_event(1))

    if model_config.component_specific_params:
        # Define parameters for p_unconstrained
        p_loc = numpyro.param(
            "p_unconstrained_loc",
            jnp.full(n_components, p_prior_params[0]),
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p_unconstrained
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    else:
        # Define parameters for p_unconstrained
        p_loc = numpyro.param(
            "p_unconstrained_loc",
            p_prior_params[0],
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_prior_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture_unconstrained depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc, p_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc[idx], p_capture_scale[idx]),
            )


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model with Variable Capture
# Probability
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
    with a low-rank approximation for the unconstrained r parameter.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_logits_unconstrained_guide is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_guide

    # Get prior parameters for p, mu, gate, and p_capture
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_prior_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained", dist.Normal(mixing_loc, mixing_scale)
    )

    p_prior_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_guide or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Low-rank multivariate normal for r_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("r_unconstrained_loc", jnp.zeros((n_components, G)))
    W = numpyro.param(
        "r_unconstrained_W", 0.01 * jnp.ones((n_components, G, k))
    )
    raw = numpyro.param(
        "r_unconstrained_raw_diag", -3.0 * jnp.ones((n_components, G))
    )
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("r_unconstrained", base.to_event(1))

    # Define parameters for gate_unconstrained
    gate_loc = numpyro.param(
        "gate_unconstrained_loc",
        jnp.full((n_components, n_genes), gate_prior_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full((n_components, n_genes), gate_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific gate_unconstrained from a Normal prior
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))

    # Define parameters for p_unconstrained
    if model_config.component_specific_params:
        # Define parameters for p_unconstrained
        p_loc = numpyro.param(
            "p_unconstrained_loc",
            jnp.full(n_components, p_prior_params[0]),
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p_unconstrained
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    else:
        # Define parameters for p_unconstrained
        p_loc = numpyro.param(
            "p_unconstrained_loc",
            p_prior_params[0],
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_prior_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture_unconstrained depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc, p_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc[idx], p_capture_scale[idx]),
            )


# ------------------------------------------------------------------------------
# Get posterior distributions for SVI results
# ------------------------------------------------------------------------------


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    split: bool = False,
) -> Dict[str, dist.Distribution]:
    """
    Constructs posterior distributions for model parameters from variational
    guide outputs.

    This function is specific to the 'linked_low_rank_unconstrained'
    parameterization and builds the appropriate `numpyro` distributions based on
    the guide parameters found in the `params` dictionary. It handles both
    single and mixture models.

    The function constructs posterior distributions for the following
    parameters:
        - p_unconstrained: Unconstrained success probability (Normal
          distribution)
        - r_unconstrained: Unconstrained mean parameter
          (LowRankMultivariateNormal distribution)
        - gate_unconstrained: Unconstrained zero-inflation probability (Normal
          distribution)
        - p_capture_unconstrained: Unconstrained cell-specific capture
          probability (Normal distribution)
        - mixing_logits_unconstrained: Unconstrained component mixing logits
          (Normal distribution)

    For vector-valued parameters (e.g., gene- or cell-specific), the function
    can return either a single batched distribution or a list of univariate
    distributions, depending on the `split` parameter.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Dictionary containing estimated variational parameters (loc, W, raw_diag
        for LowRankMultivariateNormal; loc, scale for Normal distributions) for
        each unconstrained latent variable, as produced by the guide. Expected
        keys include:
            - "p_unconstrained_loc", "p_unconstrained_scale"
            - "r_unconstrained_loc", "r_unconstrained_W",
              "r_unconstrained_raw_diag"
            - "gate_unconstrained_loc", "gate_unconstrained_scale"
            - "p_capture_unconstrained_loc", "p_capture_unconstrained_scale"
            - "mixing_logits_unconstrained_loc",
              "mixing_logits_unconstrained_scale"
        Each value is a JAX array of appropriate shape (scalar or vector).
    model_config : ModelConfig
        Model configuration object containing information about component
        specificity and mixture configuration.
    split : bool, optional (default=False)
        If True, for vector-valued parameters (e.g., gene- or cell-specific),
        return a list of univariate distributions (one per element). If False,
        return a single batched distribution.

    Returns
    -------
    Dict[str, Union[dist.Distribution, List[dist.Distribution]]]
        Dictionary mapping parameter names (e.g., "p_unconstrained",
        "r_unconstrained", etc.) to their corresponding posterior
        distributions. For vector-valued parameters, the value is either a
        batched distribution or a list of univariate distributions, depending on
        `split`.
    """
    distributions = {}

    # p_unconstrained parameter (Normal distribution)
    if "p_unconstrained_loc" in params and "p_unconstrained_scale" in params:
        if split and model_config.component_specific_params:
            # Component-specific p_unconstrained parameters
            distributions["p_unconstrained"] = [
                dist.Normal(
                    params["p_unconstrained_loc"][i],
                    params["p_unconstrained_scale"][i],
                )
                for i in range(params["p_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["p_unconstrained"] = dist.Normal(
                params["p_unconstrained_loc"], params["p_unconstrained_scale"]
            )

    # r parameter (LowRankMultivariateNormal distribution)
    if (
        "r_unconstrained_loc" in params
        and "r_unconstrained_W" in params
        and "r_unconstrained_raw_diag" in params
    ):
        # Define covariance diagonal (strictly positive)
        D = softplus(params["r_unconstrained_raw_diag"]) + 1e-4

        base = dist.LowRankMultivariateNormal(
            loc=params["r_unconstrained_loc"],
            cov_factor=params["r_unconstrained_W"],
            cov_diag=D,
        )
        distributions["r_unconstrained"] = base

    # gate_unconstrained parameter (Normal distribution)
    if (
        "gate_unconstrained_loc" in params
        and "gate_unconstrained_scale" in params
    ):
        if split and len(params["gate_unconstrained_loc"].shape) == 1:
            # Gene-specific gate_unconstrained parameters
            distributions["gate_unconstrained"] = [
                dist.Normal(
                    params["gate_unconstrained_loc"][c],
                    params["gate_unconstrained_scale"][c],
                )
                for c in range(params["gate_unconstrained_loc"].shape[0])
            ]
        elif split and len(params["gate_unconstrained_loc"].shape) == 2:
            # Component and gene-specific gate_unconstrained parameters
            distributions["gate_unconstrained"] = [
                [
                    dist.Normal(
                        params["gate_unconstrained_loc"][c, g],
                        params["gate_unconstrained_scale"][c, g],
                    )
                    for g in range(params["gate_unconstrained_loc"].shape[1])
                ]
                for c in range(params["gate_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["gate_unconstrained"] = dist.Normal(
                params["gate_unconstrained_loc"],
                params["gate_unconstrained_scale"],
            )

    # p_capture_unconstrained parameter (Normal distribution)
    if (
        "p_capture_unconstrained_loc" in params
        and "p_capture_unconstrained_scale" in params
    ):
        if split and len(params["p_capture_unconstrained_loc"].shape) == 1:
            # Cell-specific p_capture_unconstrained parameters
            distributions["p_capture_unconstrained"] = [
                dist.Normal(
                    params["p_capture_unconstrained_loc"][c],
                    params["p_capture_unconstrained_scale"][c],
                )
                for c in range(params["p_capture_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["p_capture_unconstrained"] = dist.Normal(
                params["p_capture_unconstrained_loc"],
                params["p_capture_unconstrained_scale"],
            )

    # mixing_logits_unconstrained parameter (Normal distribution)
    if (
        "mixing_logits_unconstrained_loc" in params
        and "mixing_logits_unconstrained_scale" in params
    ):
        mixing_dist = dist.Normal(
            params["mixing_logits_unconstrained_loc"],
            params["mixing_logits_unconstrained_scale"],
        )
        # Normal is typically not split since it represents a single vector
        distributions["mixing_logits_unconstrained"] = mixing_dist

    return distributions
