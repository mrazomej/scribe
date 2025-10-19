"""
Linked unconstrained parameterization models for single-cell RNA sequencing
data.

This parameterization differs from the standard parameterization in that the
mean parameter (mu) is linked to the success probability parameter (p) through
the relationship:
    r = mu * (1 - p) / p
where r is the dispersion parameter.

Parameters are sampled in unconstrained space using Normal distributions.
"""

# Import JAX-related libraries
import jax.numpy as jnp
import jax.scipy as jsp

# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Import typing
from typing import Dict, Optional, Union

# Import model config
from .config import ModelConfig

# Import decorator for model registration
from .model_registry import register

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------


@register(model_type="nbdm", parameterization="linked", unconstrained=True)
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
    are generated from a Negative Binomial distribution, where the mean (mu) and
    success probability (p) are linked through an unconstrained
    parameterization. Specifically, the dispersion parameter r is defined as:

        r = mu * (1 - p) / p

    where:
        - mu > 0 is the mean expression for each gene,
        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter.

    The parameters mu and p are sampled in unconstrained space:
        - p_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - mu = exp(mu_unconstrained) > 0

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
    p_prior_params = model_config.p_unconstrained_prior if model_config.p_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    mu_prior_params = model_config.mu_unconstrained_prior if model_config.mu_unconstrained_prior is not None else (

        0.0,

        1.0,

    )

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    mu_unconstrained = numpyro.sample(
        "mu_unconstrained",
        dist.Normal(*mu_prior_params).expand([n_genes]).to_event(1),
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))

    # Compute r using the linked relationship
    r = numpyro.deterministic("r", mu * (1 - p) / p)

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


@register(model_type="nbdm", parameterization="linked", unconstrained=True)
def nbdm_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained linked Negative Binomial
    Dirichlet-Multinomial Model (NBDM).

    This guide defines a mean-field variational approximation for the
    unconstrained linked NBDM, which models count data (such as gene expression)
    using a Negative Binomial distribution parameterized by a mean (mu) and a
    success probability (p). The model uses unconstrained real-valued parameters
    for both mu and p, which are transformed to their constrained spaces (mu >
    0, 0 < p < 1) via exponential and sigmoid (expit) functions, respectively.

    The generative model is:
        - p_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale)  (vector of length n_genes)
        - p = sigmoid(p_unconstrained)
        - mu = exp(mu_unconstrained)
        - r = mu * (1 - p) / p
        - For each cell:
            - counts ~ NegativeBinomial(r, p)

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(p_unconstrained) = Normal(p_loc, p_scale)
        - q(mu_unconstrained) = Normal(mu_loc, mu_scale)  (vectorized over
          genes)

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
    p_guide_params = model_config.p_unconstrained_guide if model_config.p_unconstrained_guide is not None else (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide if model_config.mu_unconstrained_guide is not None else (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register unconstrained mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc", jnp.full(n_genes, mu_guide_params[0])
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full(n_genes, mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mu_unconstrained", dist.Normal(mu_loc, mu_scale).to_event(1)
    )


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Model
# ------------------------------------------------------------------------------


@register(model_type="zinb", parameterization="linked", unconstrained=True)
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
    are generated from a Zero-Inflated Negative Binomial distribution, where
    the mean (mu) and success probability (p) are linked through an
    unconstrained parameterization. Specifically, the dispersion parameter r is
    defined as:

        r = mu * (1 - p) / p

    where:
        - mu > 0 is the mean expression for each gene,
        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter.

    The model also includes a zero-inflation parameter (gate) for each gene,
    which controls the probability of excess zeros:

        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    The parameters mu, p, and gate are sampled in unconstrained space:
        - p_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - mu = exp(mu_unconstrained) > 0
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
    p_prior_params = model_config.p_unconstrained_prior if model_config.p_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    mu_prior_params = model_config.mu_unconstrained_prior if model_config.mu_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    gate_prior_params = model_config.gate_unconstrained_prior if model_config.gate_unconstrained_prior is not None else (

        0.0,

        1.0,

    )

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    mu_unconstrained = numpyro.sample(
        "mu_unconstrained",
        dist.Normal(*mu_prior_params).expand([n_genes]).to_event(1),
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", dist.Normal(*gate_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Compute r using the linked relationship
    r = numpyro.deterministic("r", mu * (1 - p) / p)

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


@register(model_type="zinb", parameterization="linked", unconstrained=True)
def zinb_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained linked Zero-Inflated
    Negative Binomial (ZINB) model.

    This guide defines a variational family for the ZINB model using an
    unconstrained parameterization suitable for variational inference in
    single-cell RNA sequencing data.

    The ZINB model assumes that the observed gene expression counts for each
    cell are generated from a Zero-Inflated Negative Binomial distribution,
    where the mean (mu), success probability (p), and zero-inflation gate
    parameter (gate) are linked through unconstrained latent variables.
    Specifically:

        - The dispersion parameter r is defined as:
            r = mu * (1 - p) / p

        - mu > 0 is the mean expression for each gene,
        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter,
        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    The variational guide samples the following unconstrained latent variables:
        - p_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - mu = exp(mu_unconstrained) > 0
        - gate = sigmoid(gate_unconstrained) ∈ (0, 1)

    The variational parameters (loc and scale for each latent variable) are
    registered as learnable parameters in the guide.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes (features) per cell.
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
        This function defines the variational guide for use with NumPyro.
    """
    # Define guide parameters
    p_guide_params = model_config.p_unconstrained_guide if model_config.p_unconstrained_guide is not None else (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide if model_config.mu_unconstrained_guide is not None else (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide if model_config.gate_unconstrained_guide is not None else (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register unconstrained mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc", jnp.full(n_genes, mu_guide_params[0])
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full(n_genes, mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mu_unconstrained", dist.Normal(mu_loc, mu_scale).to_event(1)
    )

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


@register(model_type="nbvcp", parameterization="linked", unconstrained=True)
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
    are generated from a Negative Binomial distribution, where the mean (mu) and
    success probability (p) are linked through an unconstrained
    parameterization, and each cell has its own mRNA capture probability
    (p_capture). The dispersion parameter r is defined as:

        r = mu * (1 - p) / p

    where:
        - mu > 0 is the mean expression for each gene,
        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter.

    The model introduces a cell-specific mRNA capture probability p_capture ∈
    (0, 1), which modifies the effective success probability for each cell and
    gene. The effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    The parameters p, mu, and p_capture are sampled in unconstrained space:
        - p_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - p_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - mu = exp(mu_unconstrained) > 0
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
    p_prior_params = model_config.p_unconstrained_prior if model_config.p_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    mu_prior_params = model_config.mu_unconstrained_prior if model_config.mu_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    p_capture_prior_params = model_config.p_capture_unconstrained_prior if model_config.p_capture_unconstrained_prior is not None else (

        0.0,

        1.0,

    )

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    mu_unconstrained = numpyro.sample(
        "mu_unconstrained",
        dist.Normal(*mu_prior_params).expand([n_genes]).to_event(1),
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))

    # Compute r using the linked relationship
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    # If observed counts are provided, use them as observations
    if counts is not None:
        if batch_size is None:
            # No batching: sample p_capture for all cells, then sample counts
            # for all cells
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probability from unconstrained prior
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                # Transform to constrained space
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
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
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                # Transform to constrained space
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
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
            # Sample cell-specific capture probability from unconstrained prior
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained", dist.Normal(*p_capture_prior_params)
            )
            # Transform to constrained space
            p_capture = numpyro.deterministic(
                "p_capture", jsp.special.expit(p_capture_unconstrained)
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


@register(model_type="nbvcp", parameterization="linked", unconstrained=True)
def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained linked Negative Binomial
    with Variable Capture Probability (NBVCP) model.

    This guide defines a mean-field variational approximation for the NBVCP
    model, which extends the standard Negative Binomial model by introducing a
    cell-specific mRNA capture probability. The model is parameterized in an
    unconstrained space to facilitate variational inference.

    The generative model assumes that the observed gene expression counts for
    each cell are generated from a Negative Binomial distribution, where the
    mean (mu) and success probability (p) are linked through an unconstrained
    parameterization. The dispersion parameter r is defined as:

        r = mu * (1 - p) / p

    where:
        - mu > 0 is the mean expression for each gene,
        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter.

    In addition, each cell has a capture probability p_capture ∈ (0, 1), which
    is also parameterized in unconstrained space. The effective success
    probability for each cell and gene is:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))

    The variational guide defines distributions for the unconstrained
    parameters:
        - p_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - p_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - mu = exp(mu_unconstrained) > 0
        - p_capture = sigmoid(p_capture_unconstrained) ∈ (0, 1)

    The variational distributions are mean-field (fully factorized) and
    parameterized by learnable location and scale parameters for each
    unconstrained variable.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes (features) per cell.
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
    p_guide_params = model_config.p_unconstrained_guide if model_config.p_unconstrained_guide is not None else (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide if model_config.mu_unconstrained_guide is not None else (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide if model_config.p_capture_unconstrained_guide is not None else (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register unconstrained mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc", jnp.full(n_genes, mu_guide_params[0])
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full(n_genes, mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mu_unconstrained", dist.Normal(mu_loc, mu_scale).to_event(1)
    )

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


@register(model_type="zinbvcp", parameterization="linked", unconstrained=True)
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
    mean (mu) and success probability (p) are linked through an unconstrained
    parameterization, and each cell has its own mRNA capture probability
    (p_capture). The dispersion parameter r is defined as:

        r = mu * (1 - p) / p

    where:
        - mu > 0 is the mean expression for each gene,
        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter.

    The model also includes a zero-inflation parameter (gate) for each gene,
    which controls the probability of excess zeros:

        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    The model introduces a cell-specific mRNA capture probability p_capture ∈
    (0, 1), which modifies the effective success probability for each cell and
    gene. The effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    The parameters p, mu, gate, and p_capture are sampled in unconstrained
    space:
        - p_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene
        - p_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - mu = exp(mu_unconstrained) > 0
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
    # Get prior parameters from model_config, or use defaults if not provided
    p_prior_params = model_config.p_unconstrained_prior if model_config.p_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    mu_prior_params = model_config.mu_unconstrained_prior if model_config.mu_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    gate_prior_params = model_config.gate_unconstrained_prior if model_config.gate_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    p_capture_prior_params = model_config.p_capture_unconstrained_prior if model_config.p_capture_unconstrained_prior is not None else (

        0.0,

        1.0,

    )

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    mu_unconstrained = numpyro.sample(
        "mu_unconstrained",
        dist.Normal(*mu_prior_params).expand([n_genes]).to_event(1),
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", dist.Normal(*gate_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Compute r using the linked relationship
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    if counts is not None:
        # If observed counts are provided
        if batch_size is None:
            # No batching: sample for all cells
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probability from unconstrained
                # prior
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                # Transform to constrained space
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
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
                # Sample cell-specific capture probability from unconstrained
                # prior
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                # Transform to constrained space
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
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
            # Sample cell-specific capture probability from unconstrained prior
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained", dist.Normal(*p_capture_prior_params)
            )
            # Transform to constrained space
            p_capture = numpyro.deterministic(
                "p_capture", jsp.special.expit(p_capture_unconstrained)
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


@register(model_type="zinbvcp", parameterization="linked", unconstrained=True)
def zinbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained linked Zero-Inflated
    Negative Binomial with Variable Capture Probability (ZINBVCP) model.

    This guide defines a mean-field variational approximation for the ZINBVCP
    model, which extends the standard Zero-Inflated Negative Binomial (ZINB)
    model by introducing a cell-specific mRNA capture probability. The model is
    parameterized in an unconstrained space to facilitate variational inference.

    The generative model assumes that the observed gene expression counts for
    each cell are generated from a Zero-Inflated Negative Binomial distribution,
    where the mean (mu), success probability (p), zero-inflation probability
    (gate), and cell-specific capture probability (p_capture) are linked through
    unconstrained parameters. The dispersion parameter r is defined as:

        r = mu * (1 - p) / p

    where:
        - mu > 0 is the mean expression for each gene,
        - p ∈ (0, 1) is the success probability for the Negative Binomial,
        - r > 0 is the dispersion parameter.

    The model also includes a zero-inflation parameter (gate) for each gene,
    which controls the probability of excess zeros:

        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    In addition, each cell has a capture probability p_capture ∈ (0, 1), which
    modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = p * p_capture[i] / (1 - p * (1 - p_capture[i]))

    The variational guide defines distributions for the unconstrained
    parameters:
        - p_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene
        - p_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - p = sigmoid(p_unconstrained) ∈ (0, 1)
        - mu = exp(mu_unconstrained) > 0
        - gate = sigmoid(gate_unconstrained) ∈ (0, 1)
        - p_capture = sigmoid(p_capture_unconstrained) ∈ (0, 1)

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
    p_guide_params = model_config.p_unconstrained_guide if model_config.p_unconstrained_guide is not None else (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide if model_config.mu_unconstrained_guide is not None else (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide if model_config.gate_unconstrained_guide is not None else (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide if model_config.p_capture_unconstrained_guide is not None else (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register unconstrained mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc", jnp.full(n_genes, mu_guide_params[0])
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full(n_genes, mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mu_unconstrained", dist.Normal(mu_loc, mu_scale).to_event(1)
    )

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


@register(model_type="nbdm_mix", parameterization="linked", unconstrained=True)
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

    This model extends the linked unconstrained NBDM model by introducing
    mixture components, where each component has its own set of parameters. The
    model assumes that the observed gene expression counts for each cell are
    generated from a mixture of Negative Binomial distributions.

    Parameters are sampled in unconstrained space and transformed:
        - mixing_logits_unconstrained ~ Normal
        - p_unconstrained ~ Normal
        - mu_unconstrained ~ Normal
        - p = sigmoid(p_unconstrained)
        - mu = exp(mu_unconstrained)
        - r = mu * (1 - p) / p

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ Mixture(mixing_weights, NegativeBinomialLogits(r, p))

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
        Observed count data.
    batch_size: Optional[int], default=None
        If specified, enables subsampling of cells for stochastic variational
        inference.

    Returns
    -------
    None
        This function defines the probabilistic model for use with NumPyro.
    """
    n_components = model_config.n_components

    # Define prior parameters
    mixing_prior_params = model_config.mixing_logits_unconstrained_prior if model_config.mixing_logits_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    p_prior_params = model_config.p_unconstrained_prior if model_config.p_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    mu_prior_params = model_config.mu_unconstrained_prior if model_config.mu_unconstrained_prior is not None else (

        0.0,

        1.0,

    )

    # Sample unconstrained mixing logits
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )

    # Compute mixing weights from logits
    mixing_weights = numpyro.deterministic(
        "mixing_weights", jnp.exp(mixing_logits_unconstrained) / jnp.sum(jnp.exp(mixing_logits_unconstrained))
    )

    # Define mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Sample component-specific or shared parameters
    if model_config.component_specific_params:
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    mu_unconstrained = numpyro.sample(
        "mu_unconstrained",
        dist.Normal(*mu_prior_params)
        .expand([n_components, n_genes])
        .to_event(1),
    )

    # Deterministic transformations
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    # Define component distribution
    base_dist = dist.NegativeBinomialLogits(
        total_count=r, logits=p_unconstrained
    ).to_event(1)

    # Create mixture distribution
    mixture = dist.MixtureSameFamily(mixing_dist, base_dist)

    # Sample observed or latent counts
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


@register(model_type="nbdm_mix", parameterization="linked", unconstrained=True)
def nbdm_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the linked NBDM mixture model.

    This guide defines a mean-field variational approximation for the
    unconstrained NBDM mixture model.

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(mixing_logits_unconstrained)
        - q(p_unconstrained)
        - q(mu_unconstrained)

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    model_config : ModelConfig
        Configuration object.
    counts : Optional[jnp.ndarray], default=None
        Observed count data.
    batch_size : Optional[int], default=None
        Batch size for subsampling.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide if model_config.mixing_logits_unconstrained_guide is not None else (0.0, 1.0)
    p_guide_params = model_config.p_unconstrained_guide if model_config.p_unconstrained_guide is not None else (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide if model_config.mu_unconstrained_guide is not None else (0.0, 1.0)

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
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc",
        jnp.full((n_components, n_genes), mu_guide_params[0]),
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full((n_components, n_genes), mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mu_unconstrained", dist.Normal(mu_loc, mu_scale).to_event(1)
    )


# ------------------------------------------------------------------------------


@register(model_type="zinb_mix", parameterization="linked", unconstrained=True)
def zinb_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial (ZINB) mixture model
    using a linked unconstrained parameterization.

    This model extends the linked ZINB model by introducing mixture components.

    Parameters
    ----------
    n_cells: int
        Number of cells.
    n_genes: int
        Number of genes.
    model_config: ModelConfig
        Model configuration.
    counts: Optional[jnp.ndarray], default=None
        Observed count data.
    batch_size: Optional[int], default=None
        Batch size for subsampling.
    """
    n_components = model_config.n_components

    # Define prior parameters
    mixing_prior_params = model_config.mixing_logits_unconstrained_prior if model_config.mixing_logits_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    p_prior_params = model_config.p_unconstrained_prior if model_config.p_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    mu_prior_params = model_config.mu_unconstrained_prior if model_config.mu_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    gate_prior_params = model_config.gate_unconstrained_prior if model_config.gate_unconstrained_prior is not None else (

        0.0,

        1.0,

    )

    # Sample unconstrained mixing logits
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )

    # Sample component-specific or shared parameters
    if model_config.component_specific_params:
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    mu_unconstrained = numpyro.sample(
        "mu_unconstrained",
        dist.Normal(*mu_prior_params)
        .expand([n_components, n_genes])
        .to_event(1),
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained",
        dist.Normal(*gate_prior_params).expand([n_components, n_genes]),
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))
    r = numpyro.deterministic("r", mu * (1 - p) / p)
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Define distributions
    # Compute mixing weights from logits using softmax
    mixing_weights = numpyro.deterministic(
        "mixing_weights",
        jnp.exp(mixing_logits_unconstrained - jnp.max(mixing_logits_unconstrained))
        / jnp.sum(jnp.exp(mixing_logits_unconstrained - jnp.max(mixing_logits_unconstrained))),
    )

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


@register(model_type="zinb_mix", parameterization="linked", unconstrained=True)
def zinb_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the linked ZINB mixture model.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    model_config : ModelConfig
        Configuration object.
    counts : Optional[jnp.ndarray], default=None
        Observed count data.
    batch_size : Optional[int], default=None
        Batch size for subsampling.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide if model_config.mixing_logits_unconstrained_guide is not None else (0.0, 1.0)
    p_guide_params = model_config.p_unconstrained_guide if model_config.p_unconstrained_guide is not None else (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide if model_config.mu_unconstrained_guide is not None else (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide if model_config.gate_unconstrained_guide is not None else (0.0, 1.0)

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
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc",
        jnp.full((n_components, n_genes), mu_guide_params[0]),
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full((n_components, n_genes), mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mu_unconstrained", dist.Normal(mu_loc, mu_scale).to_event(1)
    )

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


@register(model_type="nbvcp_mix", parameterization="linked", unconstrained=True)
def nbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Negative Binomial with Variable Capture Probability (NBVCP)
    mixture model using a linked unconstrained parameterization.

    Parameters
    ----------
    n_cells: int
        Number of cells.
    n_genes: int
        Number of genes.
    model_config: ModelConfig
        Model configuration.
    counts: Optional[jnp.ndarray], default=None
        Observed count data.
    batch_size: Optional[int], default=None
        Batch size for subsampling.
    """
    n_components = model_config.n_components

    # Define prior parameters
    mixing_prior_params = model_config.mixing_logits_unconstrained_prior if model_config.mixing_logits_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    p_prior_params = model_config.p_unconstrained_prior if model_config.p_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    mu_prior_params = model_config.mu_unconstrained_prior if model_config.mu_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    p_capture_prior_params = model_config.p_capture_unconstrained_prior if model_config.p_capture_unconstrained_prior is not None else (

        0.0,

        1.0,

    )

    # Sample global unconstrained parameters
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )
    # Compute mixing weights from logits using softmax
    mixing_weights = numpyro.deterministic(
        "mixing_weights",
        jnp.exp(mixing_logits_unconstrained - jnp.max(mixing_logits_unconstrained))
        / jnp.sum(jnp.exp(mixing_logits_unconstrained - jnp.max(mixing_logits_unconstrained))),
    )

    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    if model_config.component_specific_params:
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    mu_unconstrained = numpyro.sample(
        "mu_unconstrained",
        dist.Normal(*mu_prior_params)
        .expand([n_components, n_genes])
        .to_event(1),
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )

    with plate_context as idx:
        p_capture_unconstrained = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(*p_capture_prior_params),
        )
        p_capture = numpyro.deterministic(
            "p_capture", jsp.special.expit(p_capture_unconstrained)
        )
        p_capture_reshaped = p_capture[:, None, None]
        p_hat = numpyro.deterministic(
            "p_hat", p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
        )
        base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
        mixture = dist.MixtureSameFamily(mixing_dist, base_dist)
        obs = counts[idx] if counts is not None and idx is not None else None
        numpyro.sample("counts", mixture, obs=obs)


# ------------------------------------------------------------------------------


@register(model_type="nbvcp_mix", parameterization="linked", unconstrained=True)
def nbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the linked NBVCP mixture model.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    model_config : ModelConfig
        Configuration object.
    counts : Optional[jnp.ndarray], default=None
        Observed count data.
    batch_size : Optional[int], default=None
        Batch size for subsampling.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide if model_config.mixing_logits_unconstrained_guide is not None else (0.0, 1.0)
    p_guide_params = model_config.p_unconstrained_guide if model_config.p_unconstrained_guide is not None else (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide if model_config.mu_unconstrained_guide is not None else (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide if model_config.p_capture_unconstrained_guide is not None else (0.0, 1.0)

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
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    mu_loc = numpyro.param(
        "mu_unconstrained_loc",
        jnp.full((n_components, n_genes), mu_guide_params[0]),
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full((n_components, n_genes), mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mu_unconstrained", dist.Normal(mu_loc, mu_scale).to_event(1)
    )

    # Cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

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


@register(model_type="zinbvcp_mix", parameterization="linked", unconstrained=True)
def zinbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Implements the Zero-Inflated Negative Binomial with Variable Capture
    Probability (ZINBVCP) mixture model using a linked unconstrained
    parameterization.

    Parameters
    ----------
    n_cells: int
        Number of cells.
    n_genes: int
        Number of genes.
    model_config: ModelConfig
        Model configuration.
    counts: Optional[jnp.ndarray], default=None
        Observed count data.
    batch_size: Optional[int], default=None
        Batch size for subsampling.
    """
    n_components = model_config.n_components

    # Define prior parameters
    mixing_prior_params = model_config.mixing_logits_unconstrained_prior if model_config.mixing_logits_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    p_prior_params = model_config.p_unconstrained_prior if model_config.p_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    mu_prior_params = model_config.mu_unconstrained_prior if model_config.mu_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    gate_prior_params = model_config.gate_unconstrained_prior if model_config.gate_unconstrained_prior is not None else (

        0.0,

        1.0,

    )
    p_capture_prior_params = model_config.p_capture_unconstrained_prior if model_config.p_capture_unconstrained_prior is not None else (

        0.0,

        1.0,

    )

    # Sample global unconstrained parameters
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )
    # Compute mixing weights from logits using softmax
    mixing_weights = numpyro.deterministic(
        "mixing_weights",
        jnp.exp(mixing_logits_unconstrained - jnp.max(mixing_logits_unconstrained))
        / jnp.sum(jnp.exp(mixing_logits_unconstrained - jnp.max(mixing_logits_unconstrained))),
    )

    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    gate_unconstrained = numpyro.sample(
        "gate_unconstrained",
        dist.Normal(*gate_prior_params).expand([n_components, n_genes]),
    )

    if model_config.component_specific_params:
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    mu_unconstrained = numpyro.sample(
        "mu_unconstrained",
        dist.Normal(*mu_prior_params)
        .expand([n_components, n_genes])
        .to_event(1),
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))
    r = numpyro.deterministic("r", mu * (1 - p) / p)
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )

    with plate_context as idx:
        p_capture_unconstrained = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(*p_capture_prior_params),
        )
        p_capture = numpyro.deterministic(
            "p_capture", jsp.special.expit(p_capture_unconstrained)
        )
        p_capture_reshaped = p_capture[:, None, None]
        p_hat = numpyro.deterministic(
            "p_hat", p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
        )
        base_dist = dist.NegativeBinomialProbs(r, p_hat)
        zinb_base_dist = dist.ZeroInflatedDistribution(
            base_dist,
            gate_logits=gate_unconstrained[None, :, :],
        ).to_event(1)
        mixture = dist.MixtureSameFamily(mixing_dist, zinb_base_dist)
        obs = counts[idx] if counts is not None and idx is not None else None
        numpyro.sample("counts", mixture, obs=obs)


# ------------------------------------------------------------------------------


@register(model_type="zinbvcp_mix", parameterization="linked", unconstrained=True)
def zinbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the linked ZINBVCP mixture model.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    model_config : ModelConfig
        Configuration object.
    counts : Optional[jnp.ndarray], default=None
        Observed count data.
    batch_size : Optional[int], default=None
        Batch size for subsampling.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide if model_config.mixing_logits_unconstrained_guide is not None else (0.0, 1.0)
    p_guide_params = model_config.p_unconstrained_guide if model_config.p_unconstrained_guide is not None else (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide if model_config.mu_unconstrained_guide is not None else (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide if model_config.gate_unconstrained_guide is not None else (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide if model_config.p_capture_unconstrained_guide is not None else (0.0, 1.0)

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
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    mu_loc = numpyro.param(
        "mu_unconstrained_loc",
        jnp.full((n_components, n_genes), mu_guide_params[0]),
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full((n_components, n_genes), mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mu_unconstrained", dist.Normal(mu_loc, mu_scale).to_event(1)
    )

    # Cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

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
) -> Dict[str, Union[dist.Distribution]]:
    """
    Construct posterior distributions for model parameters from variational
    guide outputs.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Dictionary containing estimated variational parameters (means and
        standard deviations) for each unconstrained latent variable, as produced
        by the guide. Expected keys include:
            - "p_unconstrained_loc", "p_unconstrained_scale"
            - "mu_unconstrained_loc", "mu_unconstrained_scale"
            - "gate_unconstrained_loc", "gate_unconstrained_scale"
            - "p_capture_unconstrained_loc", "p_capture_unconstrained_scale"
        Each value is a JAX array of appropriate shape (scalar or vector).
    model_config : ModelConfig
        Model configuration object (not used in this function, but included for
        interface compatibility).
    split : bool, optional (default=False)
        If True, for vector-valued parameters (e.g., gene- or cell-specific),
        return a list of univariate distributions (one per element). If False,
        return a single batched distribution.

    Returns
    -------
    Dict[str, Union[dist.Distribution]]:
        Dictionary mapping parameter names (e.g., "p_unconstrained",
        "mu_unconstrained", etc.) to their corresponding posterior
        distributions. For vector-valued parameters, the value is either a
        batched distribution or a list of univariate distributions, depending on
        `split`.
    """
    distributions = {}

    # p_unconstrained parameter (Normal distribution)
    if "p_unconstrained_loc" in params and "p_unconstrained_scale" in params:
        if split and model_config.component_specific_params:
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

    # mu_unconstrained parameter (Normal distribution)
    if "mu_unconstrained_loc" in params and "mu_unconstrained_scale" in params:
        if split and len(params["mu_unconstrained_loc"].shape) == 1:
            # Gene-specific mu parameters
            distributions["mu_unconstrained"] = [
                dist.Normal(
                    params["mu_unconstrained_loc"][c],
                    params["mu_unconstrained_scale"][c],
                )
                for c in range(params["mu_unconstrained_loc"].shape[0])
            ]
        elif split and len(params["mu_unconstrained_loc"].shape) == 2:
            # Component and gene-specific mu parameters
            distributions["mu_unconstrained"] = [
                [
                    dist.Normal(
                        params["mu_unconstrained_loc"][c, g],
                        params["mu_unconstrained_scale"][c, g],
                    )
                    for g in range(params["mu_unconstrained_loc"].shape[1])
                ]
                for c in range(params["mu_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["mu_unconstrained"] = dist.Normal(
                params["mu_unconstrained_loc"], params["mu_unconstrained_scale"]
            )

    # gate_unconstrained parameter (Normal distribution)
    if (
        "gate_unconstrained_loc" in params
        and "gate_unconstrained_scale" in params
    ):
        if split and len(params["gate_unconstrained_loc"].shape) == 1:
            # Gene-specific gate parameters
            distributions["gate_unconstrained"] = [
                dist.Normal(
                    params["gate_unconstrained_loc"][c],
                    params["gate_unconstrained_scale"][c],
                )
                for c in range(params["gate_unconstrained_loc"].shape[0])
            ]
        elif split and len(params["gate_unconstrained_loc"].shape) == 2:
            # Component and gene-specific gate parameters
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
            # Cell-specific p_capture parameters
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
        distributions["mixing_logits_unconstrained"] = dist.Normal(
            params["mixing_logits_unconstrained_loc"],
            params["mixing_logits_unconstrained_scale"],
        )

    return distributions
