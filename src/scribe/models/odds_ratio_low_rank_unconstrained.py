"""
Odds Ratio parameterization low-rank guides for single-cell RNA sequencing data
with unconstrained parameters.

This parameterization differs from the standard parameterization in that the
mean parameter (mu) is linked to the odds ratio parameter (phi) through
the relationship:
    r = mu * phi
where r is the dispersion parameter.

Parameters are sampled in unconstrained space using Normal distributions,
with mu using a low-rank multivariate normal approximation.
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
    mu parameter using odds ratio parameterization.
    """
    # Define prior parameters
    phi_prior_params = model_config.phi_unconstrained_guide or (0.0, 1.0)

    # Register phi_unconstrained as a variational parameter
    phi_loc = numpyro.param("phi_unconstrained_loc", phi_prior_params[0])
    phi_scale = numpyro.param(
        "phi_unconstrained_scale",
        phi_prior_params[1],
        constraint=constraints.positive,
    )
    # Sample phi_unconstrained from the Normal distribution
    numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Low-rank multivariate normal for mu_unconstrained
    G = n_genes
    k = model_config.guide_rank

    # Define location
    loc = numpyro.param("mu_unconstrained_loc", jnp.zeros(G))
    # Define covariance factor
    W = numpyro.param("mu_unconstrained_W", 0.01 * jnp.ones((G, k)))
    # Define raw covariance diagonal
    raw = numpyro.param("mu_unconstrained_raw_diag", -3.0 * jnp.ones(G))
    # Define covariance diagonal (strictly positive)
    D = softplus(raw) + 1e-4

    # Define base distribution
    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    # Sample mu_unconstrained from the low-rank multivariate normal
    numpyro.sample("mu_unconstrained", base)


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
    mu parameter using odds ratio parameterization.
    """
    # Define guide parameters for phi, mu, and gate
    phi_prior_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_guide or (0.0, 1.0)

    # Register variational parameters for phi_unconstrained
    phi_loc = numpyro.param("phi_unconstrained_loc", phi_prior_params[0])
    phi_scale = numpyro.param(
        "phi_unconstrained_scale",
        phi_prior_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Low-rank multivariate normal for mu_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("mu_unconstrained_loc", jnp.zeros(G))
    W = numpyro.param("mu_unconstrained_W", 0.01 * jnp.ones((G, k)))
    raw = numpyro.param("mu_unconstrained_raw_diag", -3.0 * jnp.ones(G))
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("mu_unconstrained", base)

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
    the unconstrained mu parameter using odds ratio parameterization.
    """
    # Define guide parameters for phi, mu, and phi_capture
    phi_prior_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    phi_capture_prior_params = model_config.phi_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register variational parameters for phi_unconstrained
    phi_loc = numpyro.param("phi_unconstrained_loc", phi_prior_params[0])
    phi_scale = numpyro.param(
        "phi_unconstrained_scale",
        phi_prior_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Low-rank multivariate normal for mu_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("mu_unconstrained_loc", jnp.zeros(G))
    W = numpyro.param("mu_unconstrained_W", 0.01 * jnp.ones((G, k)))
    raw = numpyro.param("mu_unconstrained_raw_diag", -3.0 * jnp.ones(G))
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("mu_unconstrained", base)

    # Set up cell-specific capture probability parameters
    phi_capture_loc = numpyro.param(
        "phi_capture_unconstrained_loc",
        jnp.full(n_cells, phi_capture_prior_params[0]),
    )
    phi_capture_scale = numpyro.param(
        "phi_capture_unconstrained_scale",
        jnp.full(n_cells, phi_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample phi_capture_unconstrained depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(phi_capture_loc, phi_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(phi_capture_loc[idx], phi_capture_scale[idx]),
            )


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial with variable capture probability
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
    approximation for the unconstrained mu parameter using odds ratio
    parameterization.
    """
    # Define guide parameters for phi, mu, and gate
    phi_prior_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_guide or (0.0, 1.0)
    phi_capture_prior_params = model_config.phi_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register variational parameters for phi_unconstrained
    phi_loc = numpyro.param("phi_unconstrained_loc", phi_prior_params[0])
    phi_scale = numpyro.param(
        "phi_unconstrained_scale",
        phi_prior_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Low-rank multivariate normal for mu_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("mu_unconstrained_loc", jnp.zeros(G))
    W = numpyro.param("mu_unconstrained_W", 0.01 * jnp.ones((G, k)))
    raw = numpyro.param("mu_unconstrained_raw_diag", -3.0 * jnp.ones(G))
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("mu_unconstrained", base)

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
    phi_capture_loc = numpyro.param(
        "phi_capture_unconstrained_loc",
        jnp.full(n_cells, phi_capture_prior_params[0]),
    )
    phi_capture_scale = numpyro.param(
        "phi_capture_unconstrained_scale",
        jnp.full(n_cells, phi_capture_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample phi_capture_unconstrained depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(phi_capture_loc, phi_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(phi_capture_loc[idx], phi_capture_scale[idx]),
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
    unconstrained mu parameter using odds ratio parameterization.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    phi_guide_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide or (0.0, 1.0)

    # Register mixing weights parameters
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_guide_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(mixing_loc, mixing_scale),
    )

    if model_config.component_specific_params:
        phi_loc = numpyro.param(
            "phi_unconstrained_loc", jnp.full(n_components, phi_guide_params[0])
        )
        phi_scale = numpyro.param(
            "phi_unconstrained_scale",
            jnp.full(n_components, phi_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))
    else:
        phi_loc = numpyro.param("phi_unconstrained_loc", phi_guide_params[0])
        phi_scale = numpyro.param(
            "phi_unconstrained_scale",
            phi_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Low-rank multivariate normal for mu_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("mu_unconstrained_loc", jnp.zeros((n_components, G)))
    W = numpyro.param(
        "mu_unconstrained_W", 0.01 * jnp.ones((n_components, G, k))
    )
    raw = numpyro.param(
        "mu_unconstrained_raw_diag", -3.0 * jnp.ones((n_components, G))
    )
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("mu_unconstrained", base)


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
    unconstrained mu parameter using odds ratio parameterization.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    phi_guide_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)

    # Register mixing weights parameters
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_guide_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(mixing_loc, mixing_scale),
    )

    gate_loc = numpyro.param(
        "gate_unconstrained_loc",
        jnp.full((n_components, n_genes), gate_guide_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full((n_components, n_genes), gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))

    if model_config.component_specific_params:
        phi_loc = numpyro.param(
            "phi_unconstrained_loc", jnp.full(n_components, phi_guide_params[0])
        )
        phi_scale = numpyro.param(
            "phi_unconstrained_scale",
            jnp.full(n_components, phi_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))
    else:
        phi_loc = numpyro.param("phi_unconstrained_loc", phi_guide_params[0])
        phi_scale = numpyro.param(
            "phi_unconstrained_scale",
            phi_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Low-rank multivariate normal for mu_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("mu_unconstrained_loc", jnp.zeros((n_components, G)))
    W = numpyro.param(
        "mu_unconstrained_W", 0.01 * jnp.ones((n_components, G, k))
    )
    raw = numpyro.param(
        "mu_unconstrained_raw_diag", -3.0 * jnp.ones((n_components, G))
    )
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("mu_unconstrained", base)


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
    approximation for the unconstrained mu parameter using odds ratio
    parameterization.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    phi_guide_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide or (0.0, 1.0)
    phi_capture_guide_params = model_config.phi_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register global parameters
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_guide_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(mixing_loc, mixing_scale),
    )

    if model_config.component_specific_params:
        phi_loc = numpyro.param(
            "phi_unconstrained_loc", jnp.full(n_components, phi_guide_params[0])
        )
        phi_scale = numpyro.param(
            "phi_unconstrained_scale",
            jnp.full(n_components, phi_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))
    else:
        phi_loc = numpyro.param("phi_unconstrained_loc", phi_guide_params[0])
        phi_scale = numpyro.param(
            "phi_unconstrained_scale",
            phi_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Low-rank multivariate normal for mu_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("mu_unconstrained_loc", jnp.zeros((n_components, G)))
    W = numpyro.param(
        "mu_unconstrained_W", 0.01 * jnp.ones((n_components, G, k))
    )
    raw = numpyro.param(
        "mu_unconstrained_raw_diag", -3.0 * jnp.ones((n_components, G))
    )
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("mu_unconstrained", base)

    # Cell-specific capture probability parameters
    phi_capture_loc = numpyro.param(
        "phi_capture_unconstrained_loc",
        jnp.full(n_cells, phi_capture_guide_params[0]),
    )
    phi_capture_scale = numpyro.param(
        "phi_capture_unconstrained_scale",
        jnp.full(n_cells, phi_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(phi_capture_loc, phi_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(phi_capture_loc[idx], phi_capture_scale[idx]),
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
    with a low-rank approximation for the unconstrained mu parameter using
    odds ratio parameterization.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    phi_guide_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)
    phi_capture_guide_params = model_config.phi_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register global parameters
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_guide_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(mixing_loc, mixing_scale),
    )

    gate_loc = numpyro.param(
        "gate_unconstrained_loc",
        jnp.full((n_components, n_genes), gate_guide_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full((n_components, n_genes), gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))

    if model_config.component_specific_params:
        phi_loc = numpyro.param(
            "phi_unconstrained_loc", jnp.full(n_components, phi_guide_params[0])
        )
        phi_scale = numpyro.param(
            "phi_unconstrained_scale",
            jnp.full(n_components, phi_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))
    else:
        phi_loc = numpyro.param("phi_unconstrained_loc", phi_guide_params[0])
        phi_scale = numpyro.param(
            "phi_unconstrained_scale",
            phi_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Low-rank multivariate normal for mu_unconstrained
    G = n_genes
    k = model_config.guide_rank

    loc = numpyro.param("mu_unconstrained_loc", jnp.zeros((n_components, G)))
    W = numpyro.param(
        "mu_unconstrained_W", 0.01 * jnp.ones((n_components, G, k))
    )
    raw = numpyro.param(
        "mu_unconstrained_raw_diag", -3.0 * jnp.ones((n_components, G))
    )
    D = softplus(raw) + 1e-4

    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    numpyro.sample("mu_unconstrained", base)

    # Cell-specific capture probability parameters
    phi_capture_loc = numpyro.param(
        "phi_capture_unconstrained_loc",
        jnp.full(n_cells, phi_capture_guide_params[0]),
    )
    phi_capture_scale = numpyro.param(
        "phi_capture_unconstrained_scale",
        jnp.full(n_cells, phi_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(phi_capture_loc, phi_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(phi_capture_loc[idx], phi_capture_scale[idx]),
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

    This function is specific to the 'odds_ratio_low_rank_unconstrained'
    parameterization and builds the appropriate `numpyro` distributions based on
    the guide parameters found in the `params` dictionary. It handles both
    single and mixture models.

    The function constructs posterior distributions for the following
    parameters:
        - phi_unconstrained: Unconstrained odds ratio parameter (Normal
          distribution)
        - mu_unconstrained: Unconstrained mean parameter
          (LowRankMultivariateNormal distribution)
        - gate_unconstrained: Unconstrained zero-inflation probability (Normal
          distribution)
        - phi_capture_unconstrained: Unconstrained cell-specific capture
          parameter (Normal distribution)
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
            - "phi_unconstrained_loc", "phi_unconstrained_scale"
            - "mu_unconstrained_loc", "mu_unconstrained_W",
              "mu_unconstrained_raw_diag"
            - "gate_unconstrained_loc", "gate_unconstrained_scale"
            - "phi_capture_unconstrained_loc", "phi_capture_unconstrained_scale"
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
        Dictionary mapping parameter names (e.g., "phi_unconstrained",
        "mu_unconstrained", etc.) to their corresponding posterior
        distributions. For vector-valued parameters, the value is either a
        batched distribution or a list of univariate distributions, depending on
        `split`.
    """
    distributions = {}

    # phi_unconstrained parameter (Normal distribution)
    if (
        "phi_unconstrained_loc" in params
        and "phi_unconstrained_scale" in params
    ):
        if split and model_config.component_specific_params:
            # Component-specific phi_unconstrained parameters
            distributions["phi_unconstrained"] = [
                dist.Normal(
                    params["phi_unconstrained_loc"][i],
                    params["phi_unconstrained_scale"][i],
                )
                for i in range(params["phi_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["phi_unconstrained"] = dist.Normal(
                params["phi_unconstrained_loc"],
                params["phi_unconstrained_scale"],
            )

    # mu parameter (LowRankMultivariateNormal distribution)
    if (
        "mu_unconstrained_loc" in params
        and "mu_unconstrained_W" in params
        and "mu_unconstrained_raw_diag" in params
    ):
        # Define covariance diagonal (strictly positive)
        D = softplus(params["mu_unconstrained_raw_diag"]) + 1e-4

        base = dist.LowRankMultivariateNormal(
            loc=params["mu_unconstrained_loc"],
            cov_factor=params["mu_unconstrained_W"],
            cov_diag=D,
        )
        distributions["mu_unconstrained"] = base

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

    # phi_capture_unconstrained parameter (Normal distribution)
    if (
        "phi_capture_unconstrained_loc" in params
        and "phi_capture_unconstrained_scale" in params
    ):
        if split and len(params["phi_capture_unconstrained_loc"].shape) == 1:
            # Cell-specific phi_capture_unconstrained parameters
            distributions["phi_capture_unconstrained"] = [
                dist.Normal(
                    params["phi_capture_unconstrained_loc"][c],
                    params["phi_capture_unconstrained_scale"][c],
                )
                for c in range(params["phi_capture_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["phi_capture_unconstrained"] = dist.Normal(
                params["phi_capture_unconstrained_loc"],
                params["phi_capture_unconstrained_scale"],
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
