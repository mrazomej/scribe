"""
Standard unconstrained parameterization models for single-cell RNA sequencing
data. Uses Normal distributions on transformed parameters for MCMC inference.

This parameterization uses the standard structure with p and r parameters, but
samples them in unconstrained space using Normal distributions.
"""

# Import JAX-related libraries
import jax.numpy as jnp
import jax.scipy as jsp
from jax.nn import softmax

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


def nbdm_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Negative Binomial-Dirichlet Multinomial (NBDM) model using an
    unconstrained parameterization suitable for variational inference in
    single-cell RNA sequencing data.

    This model assumes that the observed gene expression counts for each cell
    are generated from a Negative Binomial distribution, where the success
    probability (p) and dispersion parameter (r) are sampled in unconstrained
    space using Normal distributions and then transformed to their constrained
    spaces. Specifically:

        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter for each gene.

    The parameters are sampled in unconstrained space:
        - p_unconstrained ~ Normal(loc, scale)
        - r_unconstrained ~ Normal(loc, scale) for each gene

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - r = exp(r_unconstrained) > 0

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
        ModelConfig object specifying prior parameters for the unconstrained
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
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params).expand([n_genes]).to_event(1),
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

    # Define base distribution
    base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)

    # Sample counts
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", base_dist)


# ------------------------------------------------------------------------------


def nbdm_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained Negative Binomial
    Dirichlet-Multinomial Model (NBDM).

    This guide defines a mean-field variational approximation for the
    unconstrained NBDM, which models count data (such as gene expression) using
    a Negative Binomial distribution parameterized by a success probability (p)
    and a dispersion parameter (r). The model uses unconstrained real-valued
    parameters for both p and r, which are transformed to their constrained
    spaces (0 < p < 1, r > 0) via sigmoid (expit) and exponential functions,
    respectively.

    The generative model is:
        - p_unconstrained ~ Normal(loc, scale)
        - r_unconstrained ~ Normal(loc, scale) for each gene
        - p = sigmoid(p_unconstrained)
        - r = exp(r_unconstrained)
        - For each cell:
            - counts ~ NegativeBinomialProbs(r, p)

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(p_unconstrained) = Normal(p_loc, p_scale)
        - q(r_unconstrained) = Normal(r_loc, r_scale) for each gene

    The variational parameters (loc and scale for each latent variable) are
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
    # Define guide parameters with proper defaults
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register unconstrained r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc", jnp.full(n_genes, r_guide_params[0])
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full(n_genes, r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale).to_event(1))


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Model
# ------------------------------------------------------------------------------


def zinb_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial (ZINB) model using an
    unconstrained parameterization suitable for variational inference in
    single-cell RNA sequencing data.

    This model assumes that the observed gene expression counts for each cell
    are generated from a Zero-Inflated Negative Binomial distribution, where the
    success probability (p), dispersion parameter (r), and zero-inflation
    probability (gate) are sampled in unconstrained space using Normal
    distributions and then transformed to their constrained spaces.
    Specifically:

        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter for each gene,
        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    The parameters are sampled in unconstrained space:
        - p_unconstrained ~ Normal(loc, scale)
        - r_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - r = exp(r_unconstrained) > 0
        - gate = sigmoid(gate_unconstrained) ∈ (0, 1)

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
        ModelConfig object specifying prior parameters for the unconstrained
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
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params).expand([n_genes]).to_event(1),
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", dist.Normal(*gate_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Define distributions
    base_dist = dist.NegativeBinomialProbs(r, p)
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)

    # Sample counts
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", zinb, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", zinb)


# ------------------------------------------------------------------------------


def zinb_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained Zero-Inflated Negative
    Binomial (ZINB) model.

    This guide defines a mean-field variational approximation for the
    unconstrained ZINB, which models count data (such as gene expression) using
    a Zero-Inflated Negative Binomial distribution parameterized by a success
    probability (p), a dispersion parameter (r), and a zero-inflation
    probability (gate). The model uses unconstrained real-valued parameters for
    p, r, and gate, which are transformed to their constrained spaces (0 < p <
    1, r > 0, 0 < gate < 1) via sigmoid (expit) and exponential functions,
    respectively.

    The generative model is:
        - p_unconstrained ~ Normal(loc, scale)
        - r_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene
        - p = sigmoid(p_unconstrained)
        - r = exp(r_unconstrained)
        - gate = sigmoid(gate_unconstrained)
        - For each cell:
            - counts ~ ZeroInflatedNegativeBinomial(r, p, gate)

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(p_unconstrained) = Normal(p_loc, p_scale)
        - q(r_unconstrained) = Normal(r_loc, r_scale) for each gene
        - q(gate_unconstrained) = Normal(gate_loc, gate_scale) for each gene

    The variational parameters (loc and scale for each latent variable) are
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
    # Define guide parameters
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register unconstrained r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc", jnp.full(n_genes, r_guide_params[0])
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full(n_genes, r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale).to_event(1))

    # Register unconstrained gate parameters
    gate_loc = numpyro.param(
        "gate_unconstrained_loc", jnp.full(n_genes, gate_guide_params[0])
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full(n_genes, gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))


# ------------------------------------------------------------------------------
# Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------


def nbvcp_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Negative Binomial model with variable mRNA capture
    probability (NBVCP) using an unconstrained parameterization suitable for
    variational inference in single-cell RNA sequencing data.

    This model assumes that the observed gene expression counts for each cell
    are generated from a Negative Binomial distribution, where the success
    probability (p), dispersion parameter (r), and cell-specific capture
    probability (p_capture) are sampled in unconstrained space using Normal
    distributions and then transformed to their constrained spaces.
    Specifically:

        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter for each gene,
        - p_capture ∈ (0, 1) is the mRNA capture probability for each cell.

    The model introduces a cell-specific mRNA capture probability p_capture,
    which modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    The parameters are sampled in unconstrained space:
        - p_unconstrained ~ Normal(loc, scale)
        - r_unconstrained ~ Normal(loc, scale) for each gene
        - p_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - r = exp(r_unconstrained) > 0
        - p_capture = sigmoid(p_capture_unconstrained) ∈ (0, 1)

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
        ModelConfig object specifying prior parameters for the unconstrained
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
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample global unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params).expand([n_genes]).to_event(1),
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

    # Sample counts
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probabilities
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )

                # Reshape p_capture for broadcasting to genes
                # Shape: (batch_size, 1)
                p_capture_reshaped = p_capture[:, None]

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
                )

                # Define distribution and sample
                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )

                # Reshape p_capture for broadcasting to genes
                # Shape: (batch_size, 1)
                p_capture_reshaped = p_capture[:, None]

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
                )

                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained", dist.Normal(*p_capture_prior_params)
            )
            p_capture = numpyro.deterministic(
                "p_capture", jsp.special.expit(p_capture_unconstrained)
            )

            # Reshape p_capture for broadcasting to genes
            # Shape: (batch_size, 1)
            p_capture_reshaped = p_capture[:, None]

            # Compute effective probability
            p_hat = numpyro.deterministic(
                "p_hat",
                p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
            )

            base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
            numpyro.sample("counts", base_dist)


# ------------------------------------------------------------------------------


def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained Negative Binomial with
    Variable Capture Probability (NBVCP) model.

    This guide defines a mean-field variational approximation for the
    unconstrained NBVCP, which extends the standard Negative Binomial model by
    introducing a cell-specific mRNA capture probability. The model uses
    unconstrained real-valued parameters for p, r, and p_capture, which are
    transformed to their constrained spaces (0 < p < 1, r > 0, 0 < p_capture <
    1) via sigmoid (expit) and exponential functions, respectively.

    The generative model assumes that the observed gene expression counts for
    each cell are generated from a Negative Binomial distribution, where the
    success probability (p), dispersion parameter (r), and cell-specific capture
    probability (p_capture) are linked through unconstrained latent variables.
    The effective success probability for each cell and gene is:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(p_unconstrained) = Normal(p_loc, p_scale)
        - q(r_unconstrained) = Normal(r_loc, r_scale) for each gene
        - q(p_capture_unconstrained) = Normal(p_capture_loc, p_capture_scale)
          for each cell

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable location and scale parameters for each
    unconstrained variable.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        ModelConfig object specifying prior and guide parameters for the
        unconstrained variables.
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
    # Define guide parameters
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register global unconstrained parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    r_loc = numpyro.param(
        "r_unconstrained_loc", jnp.full(n_genes, r_guide_params[0])
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full(n_genes, r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale).to_event(1))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
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
# Zero-Inflated Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------


def zinbvcp_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial model with variable mRNA
    capture probability (ZINBVCP) using an unconstrained parameterization
    suitable for variational inference in single-cell RNA sequencing data.

    This model assumes that the observed gene expression counts for each cell
    are generated from a Zero-Inflated Negative Binomial distribution, where the
    success probability (p), dispersion parameter (r), zero-inflation
    probability (gate), and cell-specific capture probability (p_capture) are
    sampled in unconstrained space using Normal distributions and then
    transformed to their constrained spaces. Specifically:

        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter for each gene,
        - gate ∈ (0, 1) is the zero-inflation probability for each gene,
        - p_capture ∈ (0, 1) is the mRNA capture probability for each cell.

    The model introduces a cell-specific mRNA capture probability p_capture,
    which modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    The parameters are sampled in unconstrained space:
        - p_unconstrained ~ Normal(loc, scale)
        - r_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene
        - p_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - r = exp(r_unconstrained) > 0
        - gate = sigmoid(gate_unconstrained) ∈ (0, 1)
        - p_capture = sigmoid(p_capture_unconstrained) ∈ (0, 1)

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
    n_cells: int
        Number of cells in the dataset.
    n_genes: int
        Number of genes (features) per cell.
    model_config: ModelConfig
        ModelConfig object specifying prior parameters for the unconstrained
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
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample global unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params).expand([n_genes]).to_event(1),
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", dist.Normal(*gate_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Sample counts
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probabilities
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )

                # Reshape p_capture for broadcasting to genes
                # Shape: (batch_size, 1)
                p_capture_reshaped = p_capture[:, None]

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
                )

                # Define distribution and sample
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                numpyro.sample("counts", zinb, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )

                # Reshape p_capture for broadcasting to genes
                # Shape: (batch_size, 1)
                p_capture_reshaped = p_capture[:, None]

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
                )

                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained", dist.Normal(*p_capture_prior_params)
            )
            p_capture = numpyro.deterministic(
                "p_capture", jsp.special.expit(p_capture_unconstrained)
            )

            # Reshape p_capture for broadcasting to genes
            # Shape: (batch_size, 1)
            p_capture_reshaped = p_capture[:, None]

            # Compute effective probability
            p_hat = numpyro.deterministic(
                "p_hat",
                p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
            )

            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            numpyro.sample("counts", zinb)


# ------------------------------------------------------------------------------


def zinbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained Zero-Inflated Negative
    Binomial with Variable Capture Probability (ZINBVCP) model.

    This guide defines a mean-field variational approximation for the
    unconstrained ZINBVCP, which extends the standard Zero-Inflated Negative
    Binomial (ZINB) model by introducing a cell-specific mRNA capture
    probability. The model uses unconstrained real-valued parameters for p, r,
    gate, and p_capture, which are transformed to their constrained spaces (0 <
    p < 1, r > 0, 0 < gate < 1, 0 < p_capture < 1) via sigmoid (expit) and
    exponential functions, respectively.

    The generative model assumes that the observed gene expression counts for
    each cell are generated from a Zero-Inflated Negative Binomial distribution,
    where the success probability (p), dispersion parameter (r), zero-inflation
    probability (gate), and cell-specific capture probability (p_capture) are
    linked through unconstrained latent variables. The effective success
    probability for each cell and gene is:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(p_unconstrained) = Normal(p_loc, p_scale)
        - q(r_unconstrained) = Normal(r_loc, r_scale) for each gene
        - q(gate_unconstrained) = Normal(gate_loc, gate_scale) for each gene
        - q(p_capture_unconstrained) = Normal(p_capture_loc, p_capture_scale)
          for each cell

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable location and scale parameters for each
    unconstrained variable.

    Parameters
    ----------
    n_cells : int
        Number of cells (samples) in the dataset.
    n_genes : int
        Number of genes (features) in the dataset.
    model_config : ModelConfig
        ModelConfig object specifying prior and guide parameters for the
        unconstrained variables.
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
    # Define guide parameters
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register global unconstrained parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    r_loc = numpyro.param(
        "r_unconstrained_loc", jnp.full(n_genes, r_guide_params[0])
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full(n_genes, r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale).to_event(1))

    gate_loc = numpyro.param(
        "gate_unconstrained_loc", jnp.full(n_genes, gate_guide_params[0])
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full(n_genes, gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
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
# Mixture Models
# ------------------------------------------------------------------------------


def nbdm_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Negative Binomial-Dirichlet Multinomial (NBDM) mixture model
    using an unconstrained parameterization suitable for variational inference
    in single-cell RNA sequencing data.

    This model extends the standard unconstrained NBDM model by introducing
    mixture components, where each component has its own set of parameters. The
    model assumes that the observed gene expression counts for each cell are
    generated from a mixture of Negative Binomial distributions, where the
    success probability (p) and dispersion parameter (r) can be either
    component-specific or shared across components, all sampled in unconstrained
    space.

    The model parameters are:
        - mixing_logits_unconstrained: Component mixing logits from Normal prior
        - p_unconstrained: Success probability logits (Normal prior) - can be
          component-specific or shared
        - r_unconstrained: Dispersion parameter logits (Normal prior) for each
          component and gene

    The parameters are sampled in unconstrained space and transformed:
        - mixing_weights = softmax(mixing_logits_unconstrained)
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - r = exp(r_unconstrained) > 0

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ Mixture(mixing_weights, NegativeBinomialLogits(r, p))

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
    n_components = model_config.n_components

    # Define prior parameters - handle None values properly
    if model_config.mixing_logits_unconstrained_prior is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_prior

    if model_config.p_unconstrained_prior is None:
        p_prior_params = (0.0, 1.0)
    else:
        p_prior_params = model_config.p_unconstrained_prior

    if model_config.r_unconstrained_prior is None:
        r_prior_params = (0.0, 1.0)
    else:
        r_prior_params = model_config.r_unconstrained_prior

    # Sample unconstrained mixing logits
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )

    # Define mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        # All components share p, but have their own r
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    # Sample unconstrained r for each component and gene
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params)
        .expand([n_components, n_genes])
        .to_event(2),
    )

    # Deterministic transformations to constrained space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

    # Define component distribution using logits parameterization
    base_dist = dist.NegativeBinomialLogits(
        total_count=r, logits=p_unconstrained
    ).to_event(1)

    # Create mixture distribution
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


def nbdm_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained Negative Binomial
    Dirichlet-Multinomial (NBDM) mixture model.

    This guide defines a mean-field variational approximation for the
    unconstrained NBDM mixture model, which extends the standard unconstrained
    NBDM model by introducing mixture components. Each component can have its
    own set of parameters, allowing for more flexible modeling of heterogeneous
    cell populations, all using unconstrained parameterization.

    The generative model is:
        - mixing_logits_unconstrained ~ Normal(loc, scale) for each component
        - mixing_weights = softmax(mixing_logits_unconstrained)
        - p_unconstrained ~ Normal(loc, scale) - can be component-specific or
          shared
        - r_unconstrained ~ Normal(loc, scale) for each component and gene
        - p = sigmoid(p_unconstrained)
        - r = exp(r_unconstrained)
        - For each cell:
            - counts ~ Mixture(mixing_weights, NegativeBinomialLogits(r, p))

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(mixing_logits_unconstrained) = Normal(mixing_loc, mixing_scale) for
          each component
        - q(p_unconstrained) = Normal(p_loc, p_scale) - can be
          component-specific or shared
        - q(r_unconstrained) = Normal(r_loc, r_scale) for each component and
          gene

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable location and scale parameters for each mixture
    component.

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

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)

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
        # Each component has its own p
        p_loc = numpyro.param(
            "p_unconstrained_loc", jnp.full(n_components, p_guide_params[0])
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))
    else:
        # All components share p, but have their own r
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale).to_event(1))


# ------------------------------------------------------------------------------


def zinb_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial (ZINB) mixture model using an
    unconstrained parameterization suitable for variational inference in
    single-cell RNA sequencing data.

    This model extends the standard unconstrained ZINB model by introducing
    mixture components, where each component has its own set of parameters. The
    model assumes that the observed gene expression counts for each cell are
    generated from a mixture of Zero-Inflated Negative Binomial distributions,
    where the success probability (p), dispersion parameter (r), and
    zero-inflation probability (gate) can be either component-specific or shared
    across components, all sampled in unconstrained space.

    The model parameters are:
        - mixing_logits_unconstrained: Component mixing logits from Normal prior
        - p_unconstrained: Success probability logits (Normal prior) - can be
          component-specific or shared
        - r_unconstrained: Dispersion parameter logits (Normal prior) for each
          component and gene
        - gate_unconstrained: Zero-inflation probability logits (Normal prior)
          for each component and gene

    The parameters are sampled in unconstrained space and transformed:
        - mixing_weights = softmax(mixing_logits_unconstrained)
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - r = exp(r_unconstrained) > 0
        - gate = sigmoid(gate_unconstrained) ∈ (0, 1)

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
    n_components = model_config.n_components

    # Define prior parameters - handle None values properly
    if model_config.mixing_logits_unconstrained_prior is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_prior

    if model_config.p_unconstrained_prior is None:
        p_prior_params = (0.0, 1.0)
    else:
        p_prior_params = model_config.p_unconstrained_prior

    if model_config.r_unconstrained_prior is None:
        r_prior_params = (0.0, 1.0)
    else:
        r_prior_params = model_config.r_unconstrained_prior

    if model_config.gate_unconstrained_prior is None:
        gate_prior_params = (0.0, 1.0)
    else:
        gate_prior_params = model_config.gate_unconstrained_prior

    # Sample unconstrained mixing logits
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        # All components share p, but have their own r
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params)
        .expand([n_components, n_genes])
        .to_event(2),
    )

    gate_unconstrained = numpyro.sample(
        "gate_unconstrained",
        dist.Normal(*gate_prior_params).expand([n_components, n_genes]),
    )

    # Transform to constrained space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Define distributions
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    base_nb_dist = dist.NegativeBinomialLogits(
        total_count=r, logits=p_unconstrained
    )

    zinb_comp_dist = dist.ZeroInflatedDistribution(
        base_nb_dist, gate_logits=gate_unconstrained
    ).to_event(1)

    mixture = dist.MixtureSameFamily(mixing_dist, zinb_comp_dist)

    # Model likelihood
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


def zinb_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained Zero-Inflated Negative
    Binomial (ZINB) mixture model.

    This guide defines a mean-field variational approximation for the
    unconstrained ZINB mixture model, which extends the standard unconstrained
    ZINB model by introducing mixture components. Each component can have its
    own set of parameters, allowing for more flexible modeling of heterogeneous
    cell populations with different zero-inflation patterns, all using
    unconstrained parameterization.

    The generative model is:
        - mixing_logits_unconstrained ~ Normal(loc, scale) for each component
        - mixing_weights = softmax(mixing_logits_unconstrained)
        - p_unconstrained ~ Normal(loc, scale) - can be component-specific or
          shared
        - r_unconstrained ~ Normal(loc, scale) for each component and gene
        - gate_unconstrained ~ Normal(loc, scale) for each component and gene
        - p = sigmoid(p_unconstrained)
        - r = exp(r_unconstrained)
        - gate = sigmoid(gate_unconstrained)
        - For each cell:
            - counts ~ Mixture(mixing_weights, ZeroInflatedNegativeBinomial(r,
              p, gate))

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(mixing_logits_unconstrained) = Normal(mixing_loc, mixing_scale) for
          each component
        - q(p_unconstrained) = Normal(p_loc, p_scale) - can be
          component-specific or shared
        - q(r_unconstrained) = Normal(r_loc, r_scale) for each component and
          gene
        - q(gate_unconstrained) = Normal(gate_loc, gate_scale) for each
          component and gene

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable location and scale parameters for each mixture
    component.

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

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
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

    if model_config.component_specific_params:
        # Each component has its own p
        p_loc = numpyro.param(
            "p_unconstrained_loc", jnp.full(n_components, p_guide_params[0])
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))
    else:
        # All components share p, but have their own r
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale).to_event(1))

    # Register gate parameters
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


# ------------------------------------------------------------------------------


def nbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Negative Binomial with Variable Capture Probability (NBVCP)
    mixture model using an unconstrained parameterization suitable for
    variational inference in single-cell RNA sequencing data.

    This model extends the standard unconstrained NBVCP model by introducing
    mixture components, where each component has its own set of parameters. The
    model assumes that the observed gene expression counts for each cell are
    generated from a mixture of Negative Binomial distributions with variable
    capture probability, where the success probability (p), dispersion parameter
    (r), and cell-specific capture probability (p_capture) can be either
    component-specific or shared across components, all sampled in unconstrained
    space.

    The model parameters are:
        - mixing_logits_unconstrained: Component mixing logits from Normal prior
        - p_unconstrained: Success probability logits (Normal prior) - can be
          component-specific or shared
        - r_unconstrained: Dispersion parameter logits (Normal prior) for each
          component and gene
        - p_capture_unconstrained: Cell-specific capture probability logits
          (Normal prior)

    The model introduces a cell-specific mRNA capture probability p_capture,
    which modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    The parameters are sampled in unconstrained space and transformed:
        - mixing_weights = softmax(mixing_logits_unconstrained)
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - r = exp(r_unconstrained) > 0
        - p_capture = sigmoid(p_capture_unconstrained) ∈ (0, 1)

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

    # Define prior parameters - handle None values properly
    if model_config.mixing_logits_unconstrained_prior is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_prior

    if model_config.p_unconstrained_prior is None:
        p_prior_params = (0.0, 1.0)
    else:
        p_prior_params = model_config.p_unconstrained_prior

    if model_config.r_unconstrained_prior is None:
        r_prior_params = (0.0, 1.0)
    else:
        r_prior_params = model_config.r_unconstrained_prior

    if model_config.p_capture_unconstrained_prior is None:
        p_capture_prior_params = (0.0, 1.0)
    else:
        p_capture_prior_params = model_config.p_capture_unconstrained_prior

    # Sample global unconstrained parameters
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )

    # Define global mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Sample r unconstrained
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params)
        .expand([n_components, n_genes])
        .to_event(2),
    )

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        # All components share p, but have their own r
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

    # Define plate context for sampling
    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )

    with plate_context as idx:
        # Sample unconstrained cell-specific capture probability
        p_capture_unconstrained = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(*p_capture_prior_params),
        )
        # Convert to constrained space
        p_capture = numpyro.deterministic(
            "p_capture", jsp.special.expit(p_capture_unconstrained)
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


def nbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained Negative Binomial with
    Variable Capture Probability (NBVCP) mixture model.

    This guide defines a mean-field variational approximation for the
    unconstrained NBVCP mixture model, which extends the standard unconstrained
    NBVCP model by introducing mixture components. Each component can have its
    own set of parameters, allowing for more flexible modeling of heterogeneous
    cell populations with different capture probability patterns, all using
    unconstrained parameterization.

    The generative model is:
        - mixing_logits_unconstrained ~ Normal(loc, scale) for each component
        - mixing_weights = softmax(mixing_logits_unconstrained)
        - p_unconstrained ~ Normal(loc, scale) - can be component-specific or
          shared
        - r_unconstrained ~ Normal(loc, scale) for each component and gene
        - p_capture_unconstrained ~ Normal(loc, scale) for each cell
        - p = sigmoid(p_unconstrained)
        - r = exp(r_unconstrained)
        - p_capture = sigmoid(p_capture_unconstrained)
        - p_hat = p * p_capture / (1 - p * (1 - p_capture))
        - For each cell:
            - counts ~ Mixture(mixing_weights, NegativeBinomialProbs(r, p_hat))

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(mixing_logits_unconstrained) = Normal(mixing_loc, mixing_scale) for
          each component
        - q(p_unconstrained) = Normal(p_loc, p_scale) - can be
          component-specific or shared
        - q(r_unconstrained) = Normal(r_loc, r_scale) for each component and
          gene
        - q(p_capture_unconstrained) = Normal(p_capture_loc, p_capture_scale)
          for each cell

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable location and scale parameters for each mixture
    component.

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

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide or (
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

    # Register r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale).to_event(1))

    if model_config.component_specific_params:
        # Each component has its own p
        p_loc = numpyro.param(
            "p_unconstrained_loc", jnp.full(n_components, p_guide_params[0])
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))
    else:
        # All components share p, but have their own r
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
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


def zinbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial with Variable Capture
    Probability (ZINBVCP) mixture model using an unconstrained parameterization
    suitable for variational inference in single-cell RNA sequencing data.

    This model extends the standard unconstrained ZINBVCP model by introducing
    mixture components, where each component has its own set of parameters. The
    model assumes that the observed gene expression counts for each cell are
    generated from a mixture of Zero-Inflated Negative Binomial distributions
    with variable capture probability, where the success probability (p),
    dispersion parameter (r), zero-inflation probability (gate), and
    cell-specific capture probability (p_capture) can be either
    component-specific or shared across components, all sampled in unconstrained
    space.

    The model parameters are:
        - mixing_logits_unconstrained: Component mixing logits from Normal prior
        - p_unconstrained: Success probability logits (Normal prior) - can be
          component-specific or shared
        - r_unconstrained: Dispersion parameter logits (Normal prior) for each
          component and gene
        - gate_unconstrained: Zero-inflation probability logits (Normal prior)
          for each component and gene
        - p_capture_unconstrained: Cell-specific capture probability logits
          (Normal prior)

    The model introduces a cell-specific mRNA capture probability p_capture,
    which modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    The parameters are sampled in unconstrained space and transformed:
        - mixing_weights = softmax(mixing_logits_unconstrained)
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - r = exp(r_unconstrained) > 0
        - gate = sigmoid(gate_unconstrained) ∈ (0, 1)
        - p_capture = sigmoid(p_capture_unconstrained) ∈ (0, 1)

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

    # Define prior parameters - handle None values properly
    if model_config.mixing_logits_unconstrained_prior is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_prior

    if model_config.p_unconstrained_prior is None:
        p_prior_params = (0.0, 1.0)
    else:
        p_prior_params = model_config.p_unconstrained_prior

    if model_config.r_unconstrained_prior is None:
        r_prior_params = (0.0, 1.0)
    else:
        r_prior_params = model_config.r_unconstrained_prior

    if model_config.gate_unconstrained_prior is None:
        gate_prior_params = (0.0, 1.0)
    else:
        gate_prior_params = model_config.gate_unconstrained_prior

    if model_config.p_capture_unconstrained_prior is None:
        p_capture_prior_params = (0.0, 1.0)
    else:
        p_capture_prior_params = model_config.p_capture_unconstrained_prior

    # Sample global unconstrained parameters
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )
    # Define global mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Sample r unconstrained
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params)
        .expand([n_components, n_genes])
        .to_event(2),
    )

    # Sample gate unconstrained
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained",
        dist.Normal(*gate_prior_params).expand([n_components, n_genes]),
    )

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        # All components share p, but have their own r
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Define plate context for sampling
    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )

    with plate_context as idx:
        # Sample unconstrained cell-specific capture probability
        p_capture_unconstrained = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(*p_capture_prior_params),
        )
        # Convert to constrained space
        p_capture = numpyro.deterministic(
            "p_capture", jsp.special.expit(p_capture_unconstrained)
        )
        # Reshape p_capture for broadcasting with components
        p_capture_reshaped = p_capture[:, None, None]
        # Compute p_hat using the derived formula
        p_hat = numpyro.deterministic(
            "p_hat", p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
        )
        # Define the base distribution for each component (Negative Binomial)
        base_dist = dist.NegativeBinomialProbs(r, p_hat)

        zinb_base_dist = dist.ZeroInflatedDistribution(
            base_dist,
            gate_logits=gate_unconstrained[None, :, :],
        ).to_event(1)
        # Create the mixture distribution over components
        mixture = dist.MixtureSameFamily(mixing_dist, zinb_base_dist)
        # Define observation context for sampling
        obs = counts[idx] if counts is not None else None
        # Sample the counts from the mixture distribution
        numpyro.sample("counts", mixture, obs=obs)


# ------------------------------------------------------------------------------


def zinbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained Zero-Inflated Negative
    Binomial with Variable Capture Probability (ZINBVCP) mixture model.

    This guide defines a mean-field variational approximation for the
    unconstrained ZINBVCP mixture model, which extends the standard
    unconstrained ZINBVCP model by introducing mixture components. Each
    component can have its own set of parameters, allowing for more flexible
    modeling of heterogeneous cell populations with different zero-inflation and
    capture probability patterns, all using unconstrained parameterization.

    The generative model is:
        - mixing_logits_unconstrained ~ Normal(loc, scale) for each component
        - mixing_weights = softmax(mixing_logits_unconstrained)
        - p_unconstrained ~ Normal(loc, scale) - can be component-specific or
          shared
        - r_unconstrained ~ Normal(loc, scale) for each component and gene
        - gate_unconstrained ~ Normal(loc, scale) for each component and gene
        - p_capture_unconstrained ~ Normal(loc, scale) for each cell
        - p = sigmoid(p_unconstrained)
        - r = exp(r_unconstrained)
        - gate = sigmoid(gate_unconstrained)
        - p_capture = sigmoid(p_capture_unconstrained)
        - p_hat = p * p_capture / (1 - p * (1 - p_capture))
        - For each cell:
            - counts ~ Mixture(mixing_weights, ZeroInflatedNegativeBinomial(r,
              p_hat, gate))

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(mixing_logits_unconstrained) = Normal(mixing_loc, mixing_scale) for
          each component
        - q(p_unconstrained) = Normal(p_loc, p_scale) - can be
          component-specific or shared
        - q(r_unconstrained) = Normal(r_loc, r_scale) for each component and
          gene
        - q(gate_unconstrained) = Normal(gate_loc, gate_scale) for each
          component and gene
        - q(p_capture_unconstrained) = Normal(p_capture_loc, p_capture_scale)
          for each cell

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable location and scale parameters for each mixture
    component.

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

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide or (
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

    # Register r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale).to_event(1))

    # Register gate parameters
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
        # Each component has its own p
        p_loc = numpyro.param(
            "p_unconstrained_loc", jnp.full(n_components, p_guide_params[0])
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))
    else:
        # All components share p, but have their own r
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
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
# Utility function
# ------------------------------------------------------------------------------


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    split: bool = False,
) -> Dict[str, dist.Distribution]:
    """
    Constructs posterior distributions for model parameters from variational
    guide outputs.

    This function is specific to the 'unconstrained' parameterization and builds
    the appropriate `numpyro` distributions based on the guide parameters found
    in the `params` dictionary. It handles both single and mixture models.

    The function constructs posterior distributions for the following
    parameters:
        - p_unconstrained: Success probability logits (Normal distribution)
        - r_unconstrained: Dispersion parameter logits (Normal distribution)
        - gate_unconstrained: Zero-inflation probability logits (Normal
          distribution)
        - p_capture_unconstrained: Cell-specific capture probability logits
          (Normal distribution)
        - mixing_logits_unconstrained: Component mixing logits (Normal
          distribution)

    For vector-valued parameters (e.g., gene- or cell-specific), the function
    can return either a single batched distribution or a list of univariate
    distributions, depending on the `split` parameter.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Dictionary containing estimated variational parameters (location and
        scale for Normal distributions) for each unconstrained latent variable,
        as produced by the guide. Expected keys include:
            - "p_unconstrained_loc", "p_unconstrained_scale"
            - "r_unconstrained_loc", "r_unconstrained_scale"
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
        "r_unconstrained", "gate_unconstrained", etc.) to their corresponding
        posterior distributions. For vector-valued parameters, the value is
        either a batched distribution or a list of univariate distributions,
        depending on `split`.
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

    # r_unconstrained parameter (Normal distribution)
    if "r_unconstrained_loc" in params and "r_unconstrained_scale" in params:
        if split and len(params["r_unconstrained_loc"].shape) == 1:
            # Gene-specific r_unconstrained parameters
            distributions["r_unconstrained"] = [
                dist.Normal(
                    params["r_unconstrained_loc"][c],
                    params["r_unconstrained_scale"][c],
                )
                for c in range(params["r_unconstrained_loc"].shape[0])
            ]
        elif split and len(params["r_unconstrained_loc"].shape) == 2:
            # Component and gene-specific r_unconstrained parameters
            distributions["r_unconstrained"] = [
                [
                    dist.Normal(
                        params["r_unconstrained_loc"][c, g],
                        params["r_unconstrained_scale"][c, g],
                    )
                    for g in range(params["r_unconstrained_loc"].shape[1])
                ]
                for c in range(params["r_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["r_unconstrained"] = dist.Normal(
                params["r_unconstrained_loc"], params["r_unconstrained_scale"]
            )

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
        # mixing_logits_unconstrained is typically not split since it represents
        # a single probability vector
        distributions["mixing_logits_unconstrained"] = mixing_dist

    return distributions
