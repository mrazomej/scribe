"""
Standard parameterization models for single-cell RNA sequencing data.
"""

# Import JAX-related libraries
import jax.numpy as jnp

# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Import typing
from typing import Dict, Optional

# Import model config
from .config import ModelConfig

# Import decorator for model registration
from .model_registry import register

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------


@register(model_type="nbdm", parameterization="standard")
def nbdm_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Negative Binomial-Dirichlet Multinomial (NBDM) model using a
    standard parameterization suitable for variational inference in single-cell
    RNA sequencing data.

    This model assumes that the observed gene expression counts for each cell
    are generated from a Negative Binomial distribution, where the success
    probability (p) and dispersion parameter (r) are sampled directly from their
    constrained distributions. Specifically:

        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter for each gene.

    The parameters are sampled from their constrained distributions:
        - p ~ Beta(alpha, beta)
        - r ~ LogNormal(loc, scale) for each gene

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ NegativeBinomialProbs(r, p)

    where NegativeBinomialProbs denotes the Negative Binomial distribution
    parameterized by the number of failures (r) and the probability of success
    (p), and the distribution is applied independently to each gene
    (to_event(1)).

    The model supports optional batching over cells for scalable inference. If
    `counts` is provided, it is used as observed data; otherwise, the model
    samples counts from the generative process.

    Parameters
    ----------
    n_cells: int
        Number of cells in the dataset.
    n_genes: int
        Number of genes (features) per cell.
    model_config: ModelConfig
        ModelConfig object specifying prior parameters for the constrained
        variables.
    counts: Optional[jnp.ndarray], default=None
        Observed count data (not used in the model, but included for interface
        compatibility).
    batch_size: Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the probabilistic model for use with NumPyro.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    r_prior_params = model_config.r_param_prior or (0.0, 1.0)

    # Sample parameters
    p = numpyro.sample("p", dist.Beta(*p_prior_params))
    r = numpyro.sample(
        "r", dist.LogNormal(*r_prior_params).expand([n_genes]).to_event(1)
    )

    # Define base distribution
    base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)

    # Sample counts
    if counts is not None:
        if batch_size is None:
            # Without batching: sample counts for all cells
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            # With batching: sample counts for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", base_dist)


# ------------------------------------------------------------------------------


@register(model_type="nbdm", parameterization="standard")
def nbdm_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Negative Binomial-Dirichlet
    Multinomial Model (NBDM).

    This guide defines a mean-field variational approximation for the standard
    NBDM, which models count data (such as gene expression) using a Negative
    Binomial distribution parameterized by a success probability (p) and a
    dispersion parameter (r). The model uses constrained parameters sampled from
    appropriate distributions.

    The generative model is:
        - p ~ Beta(alpha, beta)
        - r ~ LogNormal(loc, scale) for each gene
        - For each cell:
            - counts ~ NegativeBinomialProbs(r, p)

    The guide defines variational distributions for the constrained parameters:
        - q(p) = Beta(p_alpha, p_beta)
        - q(r) = LogNormal(r_loc, r_scale) for each gene

    The variational parameters (alpha, beta for p and loc, scale for r) are
    registered as learnable parameters in the guide.

    Parameters
    ----------
    n_cells : int
        Number of cells (samples) in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        Configuration object containing prior and guide parameter settings.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the guide, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the variational guide for use with NumPyro's
        inference machinery.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    r_prior_params = model_config.r_param_guide or (0.0, 1.0)

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

    # Register r_shape as a variational parameter with positivity constraint
    r_loc = numpyro.param("r_loc", jnp.full(n_genes, r_prior_params[0]))
    # Register r_rate as a variational parameter with positivity constraint
    r_scale = numpyro.param(
        "r_scale",
        jnp.full(n_genes, r_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample r from the LogNormal distribution parameterized by r_loc and
    # r_scale, with event dimension 1
    numpyro.sample("r", dist.LogNormal(r_loc, r_scale).to_event(1))


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Model
# ------------------------------------------------------------------------------


@register(model_type="zinb", parameterization="standard")
def zinb_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial (ZINB) model using a standard
    parameterization suitable for variational inference in single-cell RNA
    sequencing data.

    This model assumes that the observed gene expression counts for each cell
    are generated from a Zero-Inflated Negative Binomial distribution, where the
    success probability (p), dispersion parameter (r), and zero-inflation
    probability (gate) are sampled directly from their constrained
    distributions. Specifically:

        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter for each gene,
        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    The parameters are sampled from their constrained distributions:
        - p ~ Beta(alpha, beta)
        - r ~ LogNormal(loc, scale) for each gene
        - gate ~ Beta(alpha, beta) for each gene

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ ZeroInflatedNegativeBinomial(r, p, gate)

    where ZeroInflatedNegativeBinomial denotes the zero-inflated Negative
    Binomial distribution parameterized by the number of failures (r), the
    probability of success (p), and the zero-inflation probability (gate), and
    the distribution is applied independently to each gene (to_event(1)).

    The model supports optional batching over cells for scalable inference. If
    `counts` is provided, it is used as observed data; otherwise, the model
    samples counts from the generative process.

    Parameters
    ----------
    n_cells: int
        Number of cells in the dataset.
    n_genes: int
        Number of genes (features) per cell.
    model_config: ModelConfig
        ModelConfig object specifying prior parameters for the constrained
        variables.
    counts: Optional[jnp.ndarray], default=None
        Observed count data (not used in the model, but included for interface
        compatibility).
    batch_size: Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the probabilistic model for use with NumPyro.
    """
    # Get prior parameters for p (success probability), r (dispersion), and gate
    # (zero-inflation) from model_config, or use defaults if not provided
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    r_prior_params = model_config.r_param_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)

    # Sample p from a Beta distribution with the specified prior parameters
    p = numpyro.sample("p", dist.Beta(*p_prior_params))
    # Sample r from a LogNormal distribution with the specified prior
    # parameters, expanded to n_genes
    r = numpyro.sample(
        "r", dist.LogNormal(*r_prior_params).expand([n_genes]).to_event(1)
    )
    # Sample gate from a Beta distribution with the specified prior parameters,
    # expanded to n_genes
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_genes])
    )

    # Construct the base Negative Binomial distribution using r and p
    base_dist = dist.NegativeBinomialProbs(r, p)
    # Construct the zero-inflated distribution using the base NB and gate, and
    # set event dimension to 1
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)

    # If observed counts are provided
    if counts is not None:
        # If no batching, use a plate over all cells
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample observed counts from the zero-inflated NB distribution
                numpyro.sample("counts", zinb, obs=counts)
        else:
            # If batching, use a plate with subsampling and get indices
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample observed counts for the batch indices
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # If no observed counts, just sample from the prior predictive
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", zinb)


# ------------------------------------------------------------------------------


@register(model_type="zinb", parameterization="standard")
def zinb_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Zero-Inflated Negative
    Binomial (ZINB) model.

    This guide defines a mean-field variational approximation for the standard
    ZINB, which models count data (such as gene expression) using a
    Zero-Inflated Negative Binomial distribution parameterized by a success
    probability (p), a dispersion parameter (r), and a zero-inflation
    probability (gate). The model uses constrained parameters sampled from
    appropriate distributions.

    The generative model is:
        - p ~ Beta(alpha, beta)
        - r ~ LogNormal(loc, scale) for each gene
        - gate ~ Beta(alpha, beta) for each gene
        - For each cell:
            - counts ~ ZeroInflatedNegativeBinomial(r, p, gate)

    The guide defines variational distributions for the constrained parameters:
        - q(p) = Beta(p_alpha, p_beta)
        - q(r) = LogNormal(r_loc, r_scale) for each gene
        - q(gate) = Beta(gate_alpha, gate_beta) for each gene

    The variational parameters (alpha, beta for p and gate, and loc, scale for
    r) are registered as learnable parameters in the guide.

    Parameters
    ----------
    n_cells : int
        Number of cells (samples) in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        Configuration object containing prior and guide parameter settings.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the guide, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the variational guide for use with NumPyro's
        inference machinery.
    """
    # Define guide parameters for p, r, and gate
    # Get initial values for p's Beta distribution parameters (alpha, beta)
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    # Get initial values for r's LogNormal distribution parameters (loc, scale)
    r_prior_params = model_config.r_param_guide or (0.0, 1.0)
    # Get initial values for gate's Beta distribution parameters (alpha, beta)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Register variational parameters for r (dispersion)
    r_loc = numpyro.param("r_loc", jnp.full(n_genes, r_prior_params[0]))
    r_scale = numpyro.param(
        "r_scale",
        jnp.full(n_genes, r_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r", dist.LogNormal(r_loc, r_scale).to_event(1))

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


@register(model_type="nbvcp", parameterization="standard")
def nbvcp_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Negative Binomial model with variable mRNA capture
    probability (NBVCP) using a standard parameterization suitable for
    variational inference in single-cell RNA sequencing data.

    This model assumes that the observed gene expression counts for each cell
    are generated from a Negative Binomial distribution, where the success
    probability (p), dispersion parameter (r), and cell-specific capture
    probability (p_capture) are sampled directly from their constrained
    distributions. Specifically:

        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter for each gene,
        - p_capture ∈ (0, 1) is the mRNA capture probability for each cell.

    The model introduces a cell-specific mRNA capture probability p_capture,
    which modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    The parameters are sampled from their constrained distributions:
        - p ~ Beta(alpha, beta)
        - r ~ LogNormal(loc, scale) for each gene
        - p_capture ~ Beta(alpha, beta) for each cell

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ NegativeBinomialProbs(r, p_hat[cell, :])

    where NegativeBinomialProbs denotes the Negative Binomial distribution
    parameterized by the number of failures (r) and the effective probability of
    success (p_hat), and the distribution is applied independently to each gene
    (to_event(1)).

    The model supports optional batching over cells for scalable inference. If
    `counts` is provided, it is used as observed data; otherwise, the model
    samples counts from the generative process.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes (features) per cell.
    model_config : ModelConfig
        ModelConfig object specifying prior parameters for the constrained
        variables.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the model, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the probabilistic model for use with NumPyro.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    r_prior_params = model_config.r_param_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample global success probability p from Beta prior
    p = numpyro.sample("p", dist.Beta(*p_prior_params))
    # Sample gene-specific dispersion r from LogNormal prior (vectorized over
    # genes)
    r = numpyro.sample(
        "r", dist.LogNormal(*r_prior_params).expand([n_genes]).to_event(1)
    )

    # If observed counts are provided, use them as observations
    if counts is not None:
        if batch_size is None:
            # No batching: sample p_capture for all cells, then sample counts
            # for all cells
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probability from Beta prior
                p_capture = numpyro.sample(
                    "p_capture", dist.Beta(*p_capture_prior_params)
                )
                # Reshape p_capture for broadcasting to (n_cells, n_genes)
                p_capture_reshaped = p_capture[:, None]
                # Compute effective success probability p_hat for each cell/gene
                p_hat = (
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                # Sample observed counts from Negative Binomial with p_hat
                numpyro.sample(
                    "counts",
                    dist.NegativeBinomialProbs(r, p_hat).to_event(1),
                    obs=counts,
                )
        else:
            # With batching: sample p_capture and counts for a batch of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample cell-specific capture probability for the batch
                p_capture = numpyro.sample(
                    "p_capture", dist.Beta(*p_capture_prior_params)
                )
                # Reshape p_capture for broadcasting to (batch_size, n_genes)
                p_capture_reshaped = p_capture[:, None]
                # Compute effective success probability p_hat for each cell/gene
                # in the batch
                p_hat = (
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                # Sample observed counts for the batch from Negative Binomial
                # with p_hat
                numpyro.sample(
                    "counts",
                    dist.NegativeBinomialProbs(r, p_hat).to_event(1),
                    obs=counts[idx],
                )
    else:
        # No observed counts: sample latent counts for all cells
        with numpyro.plate("cells", n_cells):
            # Sample cell-specific capture probability from Beta prior
            p_capture = numpyro.sample(
                "p_capture", dist.Beta(*p_capture_prior_params)
            )
            # Reshape p_capture for broadcasting to (n_cells, n_genes)
            p_capture_reshaped = p_capture[:, None]
            # Compute effective success probability p_hat for each cell/gene
            p_hat = p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            # Sample latent counts from Negative Binomial with p_hat
            numpyro.sample(
                "counts",
                dist.NegativeBinomialProbs(r, p_hat).to_event(1),
            )


# ------------------------------------------------------------------------------


@register(model_type="nbvcp", parameterization="standard")
def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Negative Binomial with
    Variable Capture Probability (NBVCP) model.

    This guide defines a mean-field variational approximation for the standard
    NBVCP, which extends the standard Negative Binomial model by introducing a
    cell-specific mRNA capture probability. The model uses constrained
    parameters sampled from appropriate distributions.

    The generative model assumes that the observed gene expression counts for
    each cell are generated from a Negative Binomial distribution, where the
    success probability (p), dispersion parameter (r), and cell-specific capture
    probability (p_capture) are linked through their distributions. The
    effective success probability for each cell and gene is:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))

    The guide defines variational distributions for the constrained parameters:
        - q(p) = Beta(p_alpha, p_beta)
        - q(r) = LogNormal(r_loc, r_scale) for each gene
        - q(p_capture) = Beta(p_capture_alpha, p_capture_beta) for each cell

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable location and scale parameters for each
    constrained variable.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        ModelConfig object specifying prior and guide parameters for the
        constrained variables.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the guide, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the variational guide for use with NumPyro's
        inference machinery.
    """
    # Define guide parameters for p, r, and p_capture
    # Get initial values for p's Beta distribution parameters (alpha, beta)
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    # Get initial values for r's LogNormal distribution parameters (loc, scale)
    r_prior_params = model_config.r_param_guide or (0.0, 1.0)
    # Get initial values for p_capture's Beta distribution parameters (alpha,
    # beta)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Register variational parameters for r (dispersion)
    r_loc = numpyro.param("r_loc", jnp.full(n_genes, r_prior_params[0]))
    r_scale = numpyro.param(
        "r_scale",
        jnp.full(n_genes, r_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r", dist.LogNormal(r_loc, r_scale).to_event(1))

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


@register(model_type="zinbvcp", parameterization="standard")
def zinbvcp_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial model with variable mRNA
    capture probability (ZINBVCP) using a standard parameterization suitable for
    variational inference in single-cell RNA sequencing data.

    This model assumes that the observed gene expression counts for each cell
    are generated from a Zero-Inflated Negative Binomial distribution, where the
    success probability (p), dispersion parameter (r), zero-inflation
    probability (gate), and cell-specific capture probability (p_capture) are
    sampled directly from their constrained distributions. Specifically:

        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter for each gene,
        - gate ∈ (0, 1) is the zero-inflation probability for each gene,
        - p_capture ∈ (0, 1) is the mRNA capture probability for each cell.

    The model introduces a cell-specific mRNA capture probability p_capture,
    which modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    The parameters are sampled from their constrained distributions:
        - p ~ Beta(alpha, beta)
        - r ~ LogNormal(loc, scale) for each gene
        - gate ~ Beta(alpha, beta) for each gene
        - p_capture ~ Beta(alpha, beta) for each cell

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ ZeroInflatedNegativeBinomial(r, p_hat[cell, :], gate)

    where ZeroInflatedNegativeBinomial denotes the zero-inflated Negative
    Binomial distribution parameterized by the number of failures (r), the
    effective probability of success (p_hat), and the zero-inflation probability
    (gate), and the distribution is applied independently to each gene
    (to_event(1)).

    The model supports optional batching over cells for scalable inference. If
    `counts` is provided, it is used as observed data; otherwise, the model
    samples counts from the generative process.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes (features) per cell.
    model_config : ModelConfig
        ModelConfig object specifying prior parameters for the constrained
        variables.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the model, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the probabilistic model for use with NumPyro.
    """
    # Get prior parameters from model_config, or use defaults if not provided
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    r_prior_params = model_config.r_param_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample global success probability p (Beta prior)
    p = numpyro.sample("p", dist.Beta(*p_prior_params))
    # Sample gene-specific dispersion r (Gamma prior)
    r = numpyro.sample(
        "r", dist.LogNormal(*r_prior_params).expand([n_genes]).to_event(1)
    )
    # Sample gene-specific zero-inflation gate (Beta prior)
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_genes])
    )

    if counts is not None:
        # If observed counts are provided
        if batch_size is None:
            # No batching: sample for all cells
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probability (Beta prior)
                p_capture = numpyro.sample(
                    "p_capture", dist.Beta(*p_capture_prior_params)
                )
                # Reshape p_capture for broadcasting to genes
                p_capture_reshaped = p_capture[:, None]
                # Compute effective success probability for each cell/gene
                p_hat = (
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                # Define base Negative Binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                # Define zero-inflated NB distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                # Observe counts
                numpyro.sample("counts", zinb, obs=counts)
        else:
            # With batching: sample for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample cell-specific capture probability (Beta prior)
                p_capture = numpyro.sample(
                    "p_capture", dist.Beta(*p_capture_prior_params)
                )
                # Reshape p_capture for broadcasting to genes
                p_capture_reshaped = p_capture[:, None]
                # Compute effective success probability for each cell/gene
                p_hat = (
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                # Define base Negative Binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                # Define zero-inflated NB distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                # Observe counts for the batch
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # No observed counts: just define the generative process
        with numpyro.plate("cells", n_cells):
            # Sample cell-specific capture probability (Beta prior)
            p_capture = numpyro.sample(
                "p_capture", dist.Beta(*p_capture_prior_params)
            )
            # Reshape p_capture for broadcasting to genes
            p_capture_reshaped = p_capture[:, None]
            # Compute effective success probability for each cell/gene
            p_hat = p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            # Define base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Define zero-inflated NB distribution
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Sample counts (not observed)
            numpyro.sample("counts", zinb)


# ------------------------------------------------------------------------------


@register(model_type="zinbvcp", parameterization="standard")
def zinbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Zero-Inflated Negative
    Binomial with Variable Capture Probability (ZINBVCP) model.

    This guide defines a mean-field variational approximation for the standard
    ZINBVCP, which extends the standard Zero-Inflated Negative Binomial (ZINB)
    model by introducing a cell-specific mRNA capture probability. The model
    uses constrained parameters sampled from appropriate distributions.

    The generative model assumes that the observed gene expression counts for
    each cell are generated from a Zero-Inflated Negative Binomial distribution,
    where the success probability (p), dispersion parameter (r), zero-inflation
    probability (gate), and cell-specific capture probability (p_capture) are
    linked through their distributions. The effective success probability for
    each cell and gene is:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))

    The guide defines variational distributions for the constrained parameters:
        - q(p) = Beta(p_alpha, p_beta)
        - q(r) = LogNormal(r_loc, r_scale) for each gene
        - q(gate) = Beta(gate_alpha, gate_beta) for each gene
        - q(p_capture) = Beta(p_capture_alpha, p_capture_beta) for each cell

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable location and scale parameters for each
    constrained variable.

    Parameters
    ----------
    n_cells : int
        Number of cells (samples) in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        Configuration object containing prior and guide parameter settings.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the guide, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the variational guide for use with NumPyro's
        inference machinery.
    """
    # Define guide parameters for p, r, and gate
    # Get initial values for p's Beta distribution parameters (alpha, beta)
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    # Get initial values for r's LogNormal distribution parameters (loc, scale)
    r_prior_params = model_config.r_param_guide or (0.0, 1.0)
    # Get initial values for gate's Beta distribution parameters (alpha, beta)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)
    # Get initial values for p_capture's Beta distribution parameters (alpha,
    # beta)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Register variational parameters for r (dispersion)
    r_loc = numpyro.param("r_loc", jnp.full(n_genes, r_prior_params[0]))
    r_scale = numpyro.param(
        "r_scale",
        jnp.full(n_genes, r_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r", dist.LogNormal(r_loc, r_scale).to_event(1))

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


@register(model_type="nbdm_mix", parameterization="standard")
def nbdm_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Negative Binomial-Dirichlet Multinomial (NBDM) mixture model
    using a standard parameterization suitable for variational inference in
    single-cell RNA sequencing data.

    This model extends the standard NBDM model by introducing mixture
    components, where each component has its own set of parameters. The model
    assumes that the observed gene expression counts for each cell are generated
    from a mixture of Negative Binomial distributions, where the success
    probability (p) and dispersion parameter (r) can be either
    component-specific or shared across components.

    The model parameters are:
        - mixing_weights: Component mixing probabilities from Dirichlet prior
        - p: Success probability (Beta prior) - can be component-specific or
          shared
        - r: Dispersion parameter (LogNormal prior) for each component and gene

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ Mixture(mixing_weights, NegativeBinomialProbs(r, p))

    where the mixture distribution combines multiple Negative Binomial
    components, each with their own parameters, and the distribution is applied
    independently to each gene (to_event(1)).

    The model supports optional batching over cells for scalable inference. If
    `counts` is provided, it is used as observed data; otherwise, the model
    samples counts from the generative process.

    Parameters
    ----------
    n_cells: int
        Number of cells in the dataset.
    n_genes: int
        Number of genes (features) per cell.
    model_config: ModelConfig
        ModelConfig object specifying prior parameters and mixture
        configuration.
    counts: Optional[jnp.ndarray], default=None
        Observed count data (not used in the model, but included for interface
        compatibility).
    batch_size: Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the probabilistic model for use with NumPyro.
    """
    # Get the number of mixture components from the model configuration
    n_components = model_config.n_components

    # Get prior parameters for the mixture weights
    if model_config.mixing_param_prior is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_prior)

    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    r_prior_params = model_config.r_param_prior or (0.0, 1.0)

    # Sample the mixture weights from a Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights", dist.Dirichlet(mixing_prior_params)
    )
    # Define the categorical distribution for component assignment
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Sample the gene-specific dispersion r from a LogNormal prior
    r = numpyro.sample(
        "r",
        dist.LogNormal(*r_prior_params)
        .expand([n_components, n_genes])
        .to_event(1),
    )

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p = numpyro.sample(
            "p", dist.Beta(*p_prior_params).expand([n_components])
        )
        # Broadcast p to have the right shape
        p = p[:, None]

    else:
        # All components share p, but have their own r
        p = numpyro.sample("p", dist.Beta(*p_prior_params))

    # Define the base distribution for each component (Negative Binomial)
    base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
    # Create the mixture distribution over components
    mixture = dist.MixtureSameFamily(mixing_dist, base_dist)

    # Sample observed or latent counts for each cell
    if counts is not None:
        if batch_size is None:
            # No batching: sample for all cells
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", mixture, obs=counts)
        else:
            # With batching: sample for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", mixture, obs=counts[idx])
    else:
        # No observed counts: sample latent counts for all cells
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", mixture)


# ------------------------------------------------------------------------------


@register(model_type="nbdm_mix", parameterization="standard")
def nbdm_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Negative Binomial-Dirichlet
    Multinomial (NBDM) mixture model.

    This guide defines a mean-field variational approximation for the standard
    NBDM mixture model, which extends the standard NBDM model by introducing
    mixture components. Each component can have its own set of parameters,
    allowing for more flexible modeling of heterogeneous cell populations.

    The generative model is:
        - mixing_weights ~ Dirichlet(concentrations)
        - p ~ Beta(alpha, beta) - can be component-specific or shared
        - r ~ LogNormal(loc, scale) for each component and gene
        - For each cell:
            - counts ~ Mixture(mixing_weights, NegativeBinomialProbs(r, p))

    The guide defines variational distributions for the mixture parameters:
        - q(mixing_weights) = Dirichlet(mixing_concentrations)
        - q(p) = Beta(p_alpha, p_beta) - can be component-specific or shared
        - q(r) = LogNormal(r_loc, r_scale) for each component and gene

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable concentration and shape parameters for each
    mixture component.

    Parameters
    ----------
    n_cells : int
        Number of cells (samples) in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        Configuration object containing prior and guide parameter settings,
        including mixture configuration.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the guide, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the variational guide for use with NumPyro's
        inference machinery.
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
    r_prior_params = model_config.r_param_guide or (0.0, 1.0)

    # Define parameters for r
    r_loc = numpyro.param(
        "r_loc",
        jnp.full((n_components, n_genes), r_prior_params[0]),
    )
    r_scale = numpyro.param(
        "r_scale",
        jnp.full((n_components, n_genes), r_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample the gene-specific dispersion r from a LogNormal prior
    numpyro.sample("r", dist.LogNormal(r_loc, r_scale).to_event(1))

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


@register(model_type="zinb_mix", parameterization="standard")
def zinb_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial (ZINB) mixture model using a
    standard parameterization suitable for variational inference in single-cell
    RNA sequencing data.

    This model extends the standard ZINB model by introducing mixture
    components, where each component has its own set of parameters. The model
    assumes that the observed gene expression counts for each cell are generated
    from a mixture of Zero-Inflated Negative Binomial distributions, where the
    success probability (p), dispersion parameter (r), and zero-inflation
    probability (gate) can be either component-specific or shared across
    components.

    The model parameters are:
        - mixing_weights: Component mixing probabilities from Dirichlet prior
        - p: Success probability (Beta prior) - can be component-specific or
          shared
        - r: Dispersion parameter (LogNormal prior) for each component and gene
        - gate: Zero-inflation probability (Beta prior) for each component and
          gene

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ Mixture(mixing_weights,
        ZeroInflatedNegativeBinomial(r, p, gate))

    where the mixture distribution combines multiple Zero-Inflated Negative
    Binomial components, each with their own parameters, and the distribution is
    applied independently to each gene (to_event(1)).

    The model supports optional batching over cells for scalable inference. If
    `counts` is provided, it is used as observed data; otherwise, the model
    samples counts from the generative process.

    Parameters
    ----------
    n_cells: int
        Number of cells in the dataset.
    n_genes: int
        Number of genes (features) per cell.
    model_config: ModelConfig
        ModelConfig object specifying prior parameters and mixture
        configuration.
    counts: Optional[jnp.ndarray], default=None
        Observed count data (not used in the model, but included for interface
        compatibility).
    batch_size: Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the probabilistic model for use with NumPyro.
    """
    # Get the number of mixture components from the model configuration
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_prior is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_prior)
    # Get prior parameters for p, r, and gate
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    r_prior_params = model_config.r_param_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)

    # Sample the mixture weights from a Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights", dist.Dirichlet(mixing_prior_params)
    )
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Sample the gene-specific dispersion r from a LogNormal prior
    r = numpyro.sample(
        "r",
        dist.LogNormal(*r_prior_params)
        .expand([n_components, n_genes])
        .to_event(1),
    )
    # Sample the gene-specific gate from a Beta prior
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_components, n_genes])
    )

    if model_config.component_specific_params:
        # Each component has its own p
        p = numpyro.sample(
            "p", dist.Beta(*p_prior_params).expand([n_components])
        )
        # Broadcast p to have the right shape
        p = p[:, None]

    else:
        p = numpyro.sample("p", dist.Beta(*p_prior_params))

    # Define the base distribution for each component (Negative Binomial)
    base_dist = dist.NegativeBinomialProbs(r, p)
    # Create the zero-inflated distribution over components
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)
    # Create the mixture distribution over components
    mixture = dist.MixtureSameFamily(mixing_dist, zinb)

    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", mixture, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", mixture, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", mixture)


# ------------------------------------------------------------------------------


@register(model_type="zinb_mix", parameterization="standard")
def zinb_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Zero-Inflated Negative
    Binomial (ZINB) mixture model.

    This guide defines a mean-field variational approximation for the standard
    ZINB mixture model, which extends the standard ZINB model by introducing
    mixture components. Each component can have its own set of parameters,
    allowing for more flexible modeling of heterogeneous cell populations with
    different zero-inflation patterns.

    The generative model is:
        - mixing_weights ~ Dirichlet(concentrations)
        - p ~ Beta(alpha, beta) - can be component-specific or shared
        - r ~ LogNormal(loc, scale) for each component and gene
        - gate ~ Beta(alpha, beta) for each component and gene
        - For each cell:
            - counts ~ Mixture(mixing_weights, ZeroInflatedNegativeBinomial(r,
              p, gate))

    The guide defines variational distributions for the mixture parameters:
        - q(mixing_weights) = Dirichlet(mixing_concentrations)
        - q(p) = Beta(p_alpha, p_beta) - can be component-specific or shared
        - q(r) = LogNormal(r_loc, r_scale) for each component and gene
        - q(gate) = Beta(gate_alpha, gate_beta) for each component and gene

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable concentration and shape parameters for each
    mixture component.

    Parameters
    ----------
    n_cells : int
        Number of cells (samples) in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        Configuration object containing prior and guide parameter settings,
        including mixture configuration.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the guide, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the variational guide for use with NumPyro's
        inference machinery.
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
    r_prior_params = model_config.r_param_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)

    # Define parameters for r
    r_loc = numpyro.param(
        "r_loc",
        jnp.full((n_components, n_genes), r_prior_params[0]),
    )
    r_scale = numpyro.param(
        "r_scale",
        jnp.full((n_components, n_genes), r_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample the gene-specific dispersion r from a LogNormal prior
    numpyro.sample("r", dist.LogNormal(r_loc, r_scale).to_event(1))

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
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta))

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


@register(model_type="nbvcp_mix", parameterization="standard")
def nbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Negative Binomial with Variable Capture Probability (NBVCP)
    mixture model using a standard parameterization suitable for variational
    inference in single-cell RNA sequencing data.

    This model extends the standard NBVCP model by introducing mixture
    components, where each component has its own set of parameters. The model
    assumes that the observed gene expression counts for each cell are generated
    from a mixture of Negative Binomial distributions with variable capture
    probability, where the success probability (p), dispersion parameter (r),
    and cell-specific capture probability (p_capture) can be either
    component-specific or shared across components.

    The model parameters are:
        - mixing_weights: Component mixing probabilities from Dirichlet prior
        - p: Success probability (Beta prior) - can be component-specific or
          shared
        - r: Dispersion parameter (LogNormal prior) for each component and gene
        - p_capture: Cell-specific capture probability (Beta prior)

    The model introduces a cell-specific mRNA capture probability p_capture,
    which modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ Mixture(mixing_weights, NegativeBinomialProbs(r,
        p_hat))

    where the mixture distribution combines multiple Negative Binomial
    components with variable capture probability, and the distribution is
    applied independently to each gene (to_event(1)).

    The model supports optional batching over cells for scalable inference. If
    `counts` is provided, it is used as observed data; otherwise, the model
    samples counts from the generative process.

    Parameters
    ----------
    n_cells: int
        Number of cells in the dataset.
    n_genes: int
        Number of genes (features) per cell.
    model_config: ModelConfig
        ModelConfig object specifying prior parameters and mixture
        configuration.
    counts: Optional[jnp.ndarray], default=None
        Observed count data (not used in the model, but included for interface
        compatibility).
    batch_size: Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the probabilistic model for use with NumPyro.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_prior is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_prior)
    # Get prior parameters for p, r, and p_capture
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    r_prior_params = model_config.r_param_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample the mixture weights from a Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights", dist.Dirichlet(mixing_prior_params)
    )
    # Define the categorical distribution for component assignment
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Sample the gene-specific dispersion r from a LogNormal prior
    r = numpyro.sample(
        "r",
        dist.LogNormal(*r_prior_params)
        .expand([n_components, n_genes])
        .to_event(1),
    )

    if model_config.component_specific_params:
        # Each component has its own p
        p = numpyro.sample(
            "p", dist.Beta(*p_prior_params).expand([n_components])
        )
        # Broadcast p to have the right shape
        p = p[:, None]
    else:
        p = numpyro.sample("p", dist.Beta(*p_prior_params))

    # Define plate context for sampling
    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )
    with plate_context as idx:
        # Sample cell-specific capture probability
        p_capture = numpyro.sample(
            "p_capture", dist.Beta(*p_capture_prior_params)
        )
        # Reshape p_capture for broadcasting with components
        p_capture_reshaped = p_capture[:, None, None]
        # Compute p_hat using the derived formula
        p_hat = numpyro.deterministic(
            "p_hat", p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
        )

        # Define the base distribution for each component (Negative Binomial)
        base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
        # Create the mixture distribution over components
        mixture = dist.MixtureSameFamily(mixing_dist, base_dist)
        # Define observation context for sampling
        obs = counts[idx] if counts is not None else None
        # Sample the counts from the mixture distribution
        numpyro.sample("counts", mixture, obs=obs)


# ------------------------------------------------------------------------------


@register(model_type="nbvcp_mix", parameterization="standard")
def nbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Negative Binomial with
    Variable Capture Probability (NBVCP) mixture model.

    This guide defines a mean-field variational approximation for the standard
    NBVCP mixture model, which extends the standard NBVCP model by introducing
    mixture components. Each component can have its own set of parameters,
    allowing for more flexible modeling of heterogeneous cell populations with
    different capture probability patterns.

    The generative model is:
        - mixing_weights ~ Dirichlet(concentrations)
        - p ~ Beta(alpha, beta) - can be component-specific or shared
        - r ~ LogNormal(loc, scale) for each component and gene
        - p_capture ~ Beta(alpha, beta) for each cell
        - p_hat = p * p_capture / (1 - p * (1 - p_capture))
        - For each cell:
            - counts ~ Mixture(mixing_weights, NegativeBinomialProbs(r, p_hat))

    The guide defines variational distributions for the mixture parameters:
        - q(mixing_weights) = Dirichlet(mixing_concentrations)
        - q(p) = Beta(p_alpha, p_beta) - can be component-specific or shared
        - q(r) = LogNormal(r_loc, r_scale) for each component and gene
        - q(p_capture) = Beta(p_capture_alpha, p_capture_beta) for each cell

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable concentration and shape parameters for each
    mixture component.

    Parameters
    ----------
    n_cells : int
        Number of cells (samples) in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        Configuration object containing prior and guide parameter settings,
        including mixture configuration.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the guide, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the variational guide for use with NumPyro's
        inference machinery.
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
    r_prior_params = model_config.r_param_guide or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Define parameters for r
    r_loc = numpyro.param(
        "r_loc",
        jnp.full((n_components, n_genes), r_prior_params[0]),
        constraint=constraints.positive,
    )
    r_scale = numpyro.param(
        "r_scale",
        jnp.full((n_components, n_genes), r_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific dispersion r from a LogNormal prior
    numpyro.sample(
        "r",
        dist.LogNormal(r_loc, r_scale).to_event(1),
    )

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


@register(model_type="zinbvcp_mix", parameterization="standard")
def zinbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial with Variable Capture
    Probability (ZINBVCP) mixture model using a standard parameterization
    suitable for variational inference in single-cell RNA sequencing data.

    This model extends the standard ZINBVCP model by introducing mixture
    components, where each component has its own set of parameters. The model
    assumes that the observed gene expression counts for each cell are generated
    from a mixture of Zero-Inflated Negative Binomial distributions with
    variable capture probability, where the success probability (p), dispersion
    parameter (r), zero-inflation probability (gate), and cell-specific capture
    probability (p_capture) can be either component-specific or shared across
    components.

    The model parameters are:
        - mixing_weights: Component mixing probabilities from Dirichlet prior
        - p: Success probability (Beta prior) - can be component-specific or
          shared
        - r: Dispersion parameter (LogNormal prior) for each component and gene
        - gate: Zero-inflation probability (Beta prior) for each component and
          gene
        - p_capture: Cell-specific capture probability (Beta prior)

    The model introduces a cell-specific mRNA capture probability p_capture,
    which modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ Mixture(mixing_weights,
        ZeroInflatedNegativeBinomial(r, p_hat, gate))

    where the mixture distribution combines multiple Zero-Inflated Negative
    Binomial components with variable capture probability, and the distribution
    is applied independently to each gene (to_event(1)).

    The model supports optional batching over cells for scalable inference. If
    `counts` is provided, it is used as observed data; otherwise, the model
    samples counts from the generative process.

    Parameters
    ----------
    n_cells: int
        Number of cells in the dataset.
    n_genes: int
        Number of genes (features) per cell.
    model_config: ModelConfig
        ModelConfig object specifying prior parameters and mixture
        configuration.
    counts: Optional[jnp.ndarray], default=None
        Observed count data (not used in the model, but included for interface
        compatibility).
    batch_size: Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the probabilistic model for use with NumPyro.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_prior is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_prior)
    # Get prior parameters for p, r, gate, and p_capture
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    r_prior_params = model_config.r_param_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample the mixture weights from a Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights", dist.Dirichlet(mixing_prior_params)
    )
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Sample the gene-specific dispersion r from a LogNormal prior
    r = numpyro.sample(
        "r",
        dist.LogNormal(*r_prior_params)
        .expand([n_components, n_genes])
        .to_event(1),
    )
    # Sample the gene-specific gate from a Beta prior
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_components, n_genes])
    )

    if model_config.component_specific_params:
        # Each component has its own p
        p = numpyro.sample(
            "p", dist.Beta(*p_prior_params).expand([n_components])
        )
        # Broadcast p to have the right shape
        p = p[:, None]
    else:
        p = numpyro.sample("p", dist.Beta(*p_prior_params))

    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )
    with plate_context as idx:
        p_capture = numpyro.sample(
            "p_capture", dist.Beta(*p_capture_prior_params)
        )
        p_capture_reshaped = p_capture[:, None, None]
        # Compute p_hat using the derived formula
        p_hat = numpyro.deterministic(
            "p_hat", p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
        )

        # Define the base distribution for each component (Negative Binomial)
        base_dist = dist.NegativeBinomialProbs(r, p_hat)
        # Create the zero-inflated distribution over components
        zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)
        # Create the mixture distribution over components
        mixture = dist.MixtureSameFamily(mixing_dist, zinb)
        # Define observation context for sampling
        obs = counts[idx] if counts is not None else None
        # Sample the counts from the mixture distribution
        numpyro.sample("counts", mixture, obs=obs)


# ------------------------------------------------------------------------------


@register(model_type="zinbvcp_mix", parameterization="standard")
def zinbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the standard Zero-Inflated Negative
    Binomial with Variable Capture Probability (ZINBVCP) mixture model.

    This guide defines a mean-field variational approximation for the standard
    ZINBVCP mixture model, which extends the standard ZINBVCP model by
    introducing mixture components. Each component can have its own set of
    parameters, allowing for more flexible modeling of heterogeneous cell
    populations with different zero-inflation and capture probability patterns.

    The generative model is:
        - mixing_weights ~ Dirichlet(concentrations)
        - p ~ Beta(alpha, beta) - can be component-specific or shared
        - r ~ LogNormal(loc, scale) for each component and gene
        - gate ~ Beta(alpha, beta) for each component and gene
        - p_capture ~ Beta(alpha, beta) for each cell
        - p_hat = p * p_capture / (1 - p * (1 - p_capture))
        - For each cell:
            - counts ~ Mixture(mixing_weights, ZeroInflatedNegativeBinomial(r,
              p_hat, gate))

    The guide defines variational distributions for the mixture parameters:
        - q(mixing_weights) = Dirichlet(mixing_concentrations)
        - q(p) = Beta(p_alpha, p_beta) - can be component-specific or shared
        - q(r) = LogNormal(r_loc, r_scale) for each component and gene
        - q(gate) = Beta(gate_alpha, gate_beta) for each component and gene
        - q(p_capture) = Beta(p_capture_alpha, p_capture_beta) for each cell

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable concentration and shape parameters for each
    mixture component.

    Parameters
    ----------
    n_cells : int
        Number of cells (samples) in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        Configuration object containing prior and guide parameter settings,
        including mixture configuration.
    counts : Optional[jnp.ndarray], default=None
        Observed count data (not used in the guide, but included for interface
        compatibility).
    batch_size : Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the variational guide for use with NumPyro's
        inference machinery.
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
    r_prior_params = model_config.r_param_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Define parameters for r
    r_loc = numpyro.param(
        "r_loc", jnp.full((n_components, n_genes), r_prior_params[0])
    )
    r_scale = numpyro.param(
        "r_scale",
        jnp.full((n_components, n_genes), r_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific dispersion r from a LogNormal prior
    numpyro.sample("r", dist.LogNormal(r_loc, r_scale).to_event(1))

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
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta))

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

    This function is specific to the 'standard' parameterization and builds the
    appropriate `numpyro` distributions based on the guide parameters found in
    the `params` dictionary. It handles both single and mixture models.

    The function constructs posterior distributions for the following
    parameters:
        - p: Success probability (Beta distribution)
        - r: Dispersion parameter (LogNormal distribution)
        - gate: Zero-inflation probability (Beta distribution)
        - p_capture: Cell-specific capture probability (Beta distribution)
        - mixing_weights: Component mixing probabilities (Dirichlet
          distribution)

    For vector-valued parameters (e.g., gene- or cell-specific), the function
    can return either a single batched distribution or a list of univariate
    distributions, depending on the `split` parameter.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Dictionary containing estimated variational parameters (alpha, beta for
        Beta distributions; loc, scale for LogNormal distributions) for each
        constrained latent variable, as produced by the guide. Expected keys
        include:
            - "p_alpha", "p_beta"
            - "r_loc", "r_scale"
            - "gate_alpha", "gate_beta"
            - "p_capture_alpha", "p_capture_beta"
            - "mixing_concentrations"
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
        Dictionary mapping parameter names (e.g., "p", "r", "gate", etc.) to
        their corresponding posterior distributions. For vector-valued
        parameters, the value is either a batched distribution or a list of
        univariate distributions, depending on `split`.
    """
    distributions = {}

    # p parameter (Beta distribution)
    if "p_alpha" in params and "p_beta" in params:
        if split and model_config.component_specific_params:
            # Component-specific p parameters
            distributions["p"] = [
                dist.Beta(params["p_alpha"][i], params["p_beta"][i])
                for i in range(params["p_alpha"].shape[0])
            ]
        else:
            distributions["p"] = dist.Beta(params["p_alpha"], params["p_beta"])

    # r parameter (LogNormal distribution)
    if "r_loc" in params and "r_scale" in params:
        if split and len(params["r_loc"].shape) == 1:
            # Gene-specific r parameters
            distributions["r"] = [
                dist.LogNormal(params["r_loc"][c], params["r_scale"][c])
                for c in range(params["r_loc"].shape[0])
            ]
        elif split and len(params["r_loc"].shape) == 2:
            # Component and gene-specific r parameters
            distributions["r"] = [
                [
                    dist.LogNormal(
                        params["r_loc"][c, g], params["r_scale"][c, g]
                    )
                    for g in range(params["r_loc"].shape[1])
                ]
                for c in range(params["r_loc"].shape[0])
            ]
        else:
            distributions["r"] = dist.LogNormal(
                params["r_loc"], params["r_scale"]
            )

    # gate parameter (Beta distribution)
    if "gate_alpha" in params and "gate_beta" in params:
        if split and len(params["gate_alpha"].shape) == 1:
            # Gene-specific gate parameters
            distributions["gate"] = [
                dist.Beta(params["gate_alpha"][c], params["gate_beta"][c])
                for c in range(params["gate_alpha"].shape[0])
            ]
        elif split and len(params["gate_alpha"].shape) == 2:
            # Component and gene-specific gate parameters
            distributions["gate"] = [
                [
                    dist.Beta(
                        params["gate_alpha"][c, g], params["gate_beta"][c, g]
                    )
                    for g in range(params["gate_alpha"].shape[1])
                ]
                for c in range(params["gate_alpha"].shape[0])
            ]
        else:
            distributions["gate"] = dist.Beta(
                params["gate_alpha"], params["gate_beta"]
            )

    # p_capture parameter (Beta distribution)
    if "p_capture_alpha" in params and "p_capture_beta" in params:
        if split and len(params["p_capture_alpha"].shape) == 1:
            # Cell-specific p_capture parameters
            distributions["p_capture"] = [
                dist.Beta(
                    params["p_capture_alpha"][c], params["p_capture_beta"][c]
                )
                for c in range(params["p_capture_alpha"].shape[0])
            ]
        else:
            distributions["p_capture"] = dist.Beta(
                params["p_capture_alpha"], params["p_capture_beta"]
            )

    # mixing_weights parameter (Dirichlet distribution)
    if "mixing_concentrations" in params:
        mixing_dist = dist.Dirichlet(params["mixing_concentrations"])
        # Dirichlet is typically not split since it represents a single
        # probability vector
        distributions["mixing_weights"] = mixing_dist

    return distributions
