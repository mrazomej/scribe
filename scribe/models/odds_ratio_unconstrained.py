"""
Odds Ratio unconstrained parameterization models for single-cell RNA sequencing
data.

This parameterization differs from the standard parameterization in that the
mean parameter (mu) is linked to the odds ratio parameter (phi) through
the relationship:
    r = mu * phi
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
from typing import Dict, Optional, Union, List

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
    are generated from a Negative Binomial distribution, where the mean (mu) and
    odds ratio parameter (phi) are linked through an unconstrained
    parameterization. Specifically, the dispersion parameter r is defined as:

        r = mu * phi

    where:
        - mu > 0 is the mean expression for each gene,
        - phi > 0 is the odds ratio parameter that controls dispersion,
        - r > 0 is the dispersion parameter.

    The parameters mu and phi are sampled in unconstrained space:
        - phi_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene

    These are transformed to the constrained space as follows:
        - phi = exp(phi_unconstrained) > 0
        - mu = exp(mu_unconstrained) > 0

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ NegativeBinomialLogits(r, -log(phi))

    where NegativeBinomialLogits denotes the Negative Binomial distribution
    parameterized by the number of failures (r) and the logits parameter
    (-log(phi)), and the distribution is applied independently to each gene
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
    phi_prior_params = model_config.phi_unconstrained_prior or (0.0, 1.0)
    mu_prior_params = model_config.mu_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained parameters
    phi_unconstrained = numpyro.sample(
        "phi_unconstrained", dist.Normal(*phi_prior_params)
    )
    mu_unconstrained = numpyro.sample(
        "mu_unconstrained", dist.Normal(*mu_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    phi = numpyro.deterministic("phi", jnp.exp(phi_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))

    # Compute r using the odds ratio relationship
    r = numpyro.deterministic("r", mu * phi)

    # Define base distribution
    base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi)).to_event(1)

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
    Mean-field variational guide for the unconstrained odds ratio Negative
    Binomial Dirichlet-Multinomial Model (NBDM).

    This guide defines a mean-field variational approximation for the
    unconstrained odds ratio NBDM, which models count data (such as gene
    expression) using a Negative Binomial distribution parameterized by a mean
    (mu) and an odds ratio parameter (phi). The model uses unconstrained
    real-valued parameters for both mu and phi, which are transformed to their
    constrained spaces (mu > 0, phi > 0) via exponential functions.

    The generative model is:
        - phi_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale)  (vector of length n_genes)
        - phi = exp(phi_unconstrained)
        - mu = exp(mu_unconstrained)
        - r = mu * phi
        - For each cell:
            - counts ~ NegativeBinomialLogits(r, -log(phi))

    The guide defines variational distributions for the unconstrained
    parameters:
        - q(phi_unconstrained) = Normal(phi_loc, phi_scale)
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
    phi_guide_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide or (0.0, 1.0)

    # Register unconstrained phi parameters
    phi_loc = numpyro.param("phi_unconstrained_loc", phi_guide_params[0])
    phi_scale = numpyro.param(
        "phi_unconstrained_scale",
        phi_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Register unconstrained mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc", jnp.full(n_genes, mu_guide_params[0])
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full(n_genes, mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("mu_unconstrained", dist.Normal(mu_loc, mu_scale))


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
    mean (mu) and odds ratio parameter (phi) are linked through an unconstrained
    parameterization. Specifically, the dispersion parameter r is defined as:

        r = mu * phi

    where:
        - mu > 0 is the mean expression for each gene,
        - phi > 0 is the odds ratio parameter that controls dispersion,
        - r > 0 is the dispersion parameter.

    The model also includes a zero-inflation parameter (gate) for each gene,
    which controls the probability of excess zeros:

        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    The parameters mu, phi, and gate are sampled in unconstrained space:
        - phi_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene

    These are transformed to the constrained space as follows:
        - phi = exp(phi_unconstrained) > 0
        - mu = exp(mu_unconstrained) > 0
        - gate = sigmoid(gate_unconstrained) ∈ (0, 1)

    For each cell, the observed counts vector (of length n_genes) is modeled as:

        counts[cell, :] ~ ZeroInflatedNegativeBinomial(r, -log(phi), gate)

    where ZeroInflatedNegativeBinomial denotes the zero-inflated Negative
    Binomial distribution parameterized by the number of failures (r), the
    logits parameter (-log(phi)), and the zero-inflation probability (gate), and
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
    phi_prior_params = model_config.phi_unconstrained_prior or (0.0, 1.0)
    mu_prior_params = model_config.mu_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained parameters
    phi_unconstrained = numpyro.sample(
        "phi_unconstrained", dist.Normal(*phi_prior_params)
    )
    mu_unconstrained = numpyro.sample(
        "mu_unconstrained", dist.Normal(*mu_prior_params).expand([n_genes])
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", dist.Normal(*gate_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    phi = numpyro.deterministic("phi", jnp.exp(phi_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Compute r using the odds ratio relationship
    r = numpyro.deterministic("r", mu * phi)

    # Define distributions
    base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi))
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
    Mean-field variational guide for the unconstrained odds ratio Zero-Inflated
    Negative Binomial (ZINB) model.

    This guide defines a variational family for the ZINB model using an
    unconstrained parameterization suitable for variational inference in
    single-cell RNA sequencing data.

    The ZINB model assumes that the observed gene expression counts for each
    cell are generated from a Zero-Inflated Negative Binomial distribution,
    where the mean (mu), odds ratio parameter (phi), and zero-inflation gate
    parameter (gate) are linked through unconstrained latent variables.
    Specifically:

        - The dispersion parameter r is defined as:
            r = mu * phi

        - mu > 0 is the mean expression for each gene,
        - phi > 0 is the odds ratio parameter that controls dispersion,
        - r > 0 is the dispersion parameter,
        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    The variational guide samples the following unconstrained latent variables:
        - phi_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene

    These are transformed to the constrained space as follows:
        - phi = exp(phi_unconstrained) > 0
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
    phi_guide_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)

    # Register unconstrained phi parameters
    phi_loc = numpyro.param("phi_unconstrained_loc", phi_guide_params[0])
    phi_scale = numpyro.param(
        "phi_unconstrained_scale",
        phi_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Register unconstrained mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc", jnp.full(n_genes, mu_guide_params[0])
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full(n_genes, mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("mu_unconstrained", dist.Normal(mu_loc, mu_scale))

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
    are generated from a Negative Binomial distribution, where the mean (mu) and
    odds ratio parameter (phi) are linked through an unconstrained
    parameterization, and each cell has its own mRNA capture probability
    (phi_capture). The dispersion parameter r is defined as:

        r = mu * phi

    where:
        - mu > 0 is the mean expression for each gene,
        - phi > 0 is the odds ratio parameter that controls dispersion,
        - r > 0 is the dispersion parameter.

    The model introduces a cell-specific mRNA capture probability phi_capture >
    0, which modifies the effective success probability for each cell and gene.
    The effective success probability for cell i and gene j is:

        p_hat[i, j] = 1 / (1 + phi + phi * phi_capture[i])

    The parameters phi, mu, and phi_capture are sampled in unconstrained space:
        - phi_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - phi_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - phi = exp(phi_unconstrained) > 0
        - mu = exp(mu_unconstrained) > 0
        - phi_capture = exp(phi_capture_unconstrained) > 0

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
    phi_prior_params = model_config.phi_unconstrained_prior or (0.0, 1.0)
    mu_prior_params = model_config.mu_unconstrained_prior or (0.0, 1.0)
    phi_capture_prior_params = model_config.phi_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample unconstrained parameters
    phi_unconstrained = numpyro.sample(
        "phi_unconstrained", dist.Normal(*phi_prior_params)
    )
    mu_unconstrained = numpyro.sample(
        "mu_unconstrained", dist.Normal(*mu_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    phi = numpyro.deterministic("phi", jnp.exp(phi_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))

    # Compute r using the odds ratio relationship
    r = numpyro.deterministic("r", mu * phi)

    # If observed counts are provided, use them as observations
    if counts is not None:
        if batch_size is None:
            # No batching: sample phi_capture for all cells, then sample counts
            # for all cells
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probability from unconstrained prior
                phi_capture_unconstrained = numpyro.sample(
                    "phi_capture_unconstrained",
                    dist.Normal(*phi_capture_prior_params),
                )
                # Transform to constrained space
                phi_capture = numpyro.deterministic(
                    "phi_capture", jnp.exp(phi_capture_unconstrained)
                )
                # Reshape phi_capture for broadcasting to (n_cells, n_genes)
                phi_capture_reshaped = phi_capture[:, None]
                # Compute p_hat using the derived formula
                p_hat = numpyro.deterministic(
                    "p_hat", 1.0 / (1 + phi + phi * phi_capture_reshaped)
                )
                # Sample observed counts from Negative Binomial with p_hat
                numpyro.sample(
                    "counts",
                    dist.NegativeBinomialProbs(r, p_hat).to_event(1),
                    obs=counts,
                )
        else:
            # With batching: sample phi_capture and counts for a batch of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample cell-specific capture probability for the batch
                phi_capture_unconstrained = numpyro.sample(
                    "phi_capture_unconstrained",
                    dist.Normal(*phi_capture_prior_params),
                )
                # Transform to constrained space
                phi_capture = numpyro.deterministic(
                    "phi_capture", jnp.exp(phi_capture_unconstrained)
                )
                # Reshape phi_capture for broadcasting to (batch_size, n_genes)
                phi_capture_reshaped = phi_capture[:, None]
                # Compute p_hat using the derived formula
                p_hat = numpyro.deterministic(
                    "p_hat", 1.0 / (1 + phi + phi * phi_capture_reshaped)
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
            phi_capture_unconstrained = numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(*phi_capture_prior_params),
            )
            # Transform to constrained space
            phi_capture = numpyro.deterministic(
                "phi_capture", jnp.exp(phi_capture_unconstrained)
            )
            # Reshape phi_capture for broadcasting to (n_cells, n_genes)
            phi_capture_reshaped = phi_capture[:, None]
            # Compute effective success probability p_hat for each cell/gene
            p_hat = numpyro.deterministic(
                "p_hat", 1.0 / (1 + phi + phi * phi_capture_reshaped)
            )
            # Sample latent counts from Negative Binomial with p_hat
            numpyro.sample(
                "counts",
                dist.NegativeBinomialProbs(r, p_hat).to_event(1),
            )


# ------------------------------------------------------------------------------


def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained odds ratio Negative
    Binomial with Variable Capture Probability (NBVCP) model.

    This guide defines a mean-field variational approximation for the NBVCP
    model, which extends the standard Negative Binomial model by introducing a
    cell-specific mRNA capture probability. The model is parameterized in an
    unconstrained space to facilitate variational inference.

    The generative model assumes that the observed gene expression counts for
    each cell are generated from a Negative Binomial distribution, where the
    mean (mu) and odds ratio parameter (phi) are linked through an unconstrained
    parameterization. The dispersion parameter r is defined as:

        r = mu * phi

    where:
        - mu > 0 is the mean expression for each gene,
        - phi > 0 is the odds ratio parameter that controls dispersion,
        - r > 0 is the dispersion parameter.

    In addition, each cell has a capture probability phi_capture > 0, which is
    also parameterized in unconstrained space. The effective success probability
    for each cell and gene is:

        p_hat = 1 / (1 + phi + phi * phi_capture)

    The variational guide defines distributions for the unconstrained
    parameters:
        - phi_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - phi_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - phi = exp(phi_unconstrained) > 0
        - mu = exp(mu_unconstrained) > 0
        - phi_capture = exp(phi_capture_unconstrained) > 0

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
    phi_guide_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide or (0.0, 1.0)
    phi_capture_guide_params = model_config.phi_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register unconstrained phi parameters
    phi_loc = numpyro.param("phi_unconstrained_loc", phi_guide_params[0])
    phi_scale = numpyro.param(
        "phi_unconstrained_scale",
        phi_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Register unconstrained mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc", jnp.full(n_genes, mu_guide_params[0])
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full(n_genes, mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("mu_unconstrained", dist.Normal(mu_loc, mu_scale))

    # Set up cell-specific capture probability parameters
    phi_capture_loc = numpyro.param(
        "phi_capture_unconstrained_loc",
        jnp.full(n_cells, phi_capture_guide_params[0]),
    )
    phi_capture_scale = numpyro.param(
        "phi_capture_unconstrained_scale",
        jnp.full(n_cells, phi_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample phi_capture depending on batch size
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
    mean (mu) and odds ratio parameter (phi) are linked through an unconstrained
    parameterization, and each cell has its own mRNA capture probability
    (phi_capture). The dispersion parameter r is defined as:

        r = mu * phi

    where:
        - mu > 0 is the mean expression for each gene,
        - phi > 0 is the odds ratio parameter that controls dispersion,
        - r > 0 is the dispersion parameter.

    The model also includes a zero-inflation parameter (gate) for each gene,
    which controls the probability of excess zeros:

        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    The model introduces a cell-specific mRNA capture probability phi_capture >
    0, which modifies the effective success probability for each cell and gene.
    The effective success probability for cell i and gene j is:

        p_hat[i, j] = 1 / (1 + phi + phi * phi_capture[i])

    The parameters phi, mu, gate, and phi_capture are sampled in unconstrained
    space:
        - phi_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene
        - phi_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - phi = exp(phi_unconstrained) > 0
        - mu = exp(mu_unconstrained) > 0
        - gate = sigmoid(gate_unconstrained) ∈ (0, 1)
        - phi_capture = exp(phi_capture_unconstrained) > 0

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
    phi_prior_params = model_config.phi_unconstrained_prior or (0.0, 1.0)
    mu_prior_params = model_config.mu_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)
    phi_capture_prior_params = model_config.phi_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample unconstrained parameters
    phi_unconstrained = numpyro.sample(
        "phi_unconstrained", dist.Normal(*phi_prior_params)
    )
    mu_unconstrained = numpyro.sample(
        "mu_unconstrained", dist.Normal(*mu_prior_params).expand([n_genes])
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", dist.Normal(*gate_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    phi = numpyro.deterministic("phi", jnp.exp(phi_unconstrained))
    mu = numpyro.deterministic("mu", jnp.exp(mu_unconstrained))
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Compute r using the odds ratio relationship
    r = numpyro.deterministic("r", mu * phi)

    if counts is not None:
        # If observed counts are provided
        if batch_size is None:
            # No batching: sample for all cells
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probability from unconstrained
                # prior
                phi_capture_unconstrained = numpyro.sample(
                    "phi_capture_unconstrained",
                    dist.Normal(*phi_capture_prior_params),
                )
                # Transform to constrained space
                phi_capture = numpyro.deterministic(
                    "phi_capture", jnp.exp(phi_capture_unconstrained)
                )
                # Reshape phi_capture for broadcasting to genes
                phi_capture_reshaped = phi_capture[:, None]
                # Compute p_hat using the derived formula
                p_hat = numpyro.deterministic(
                    "p_hat", 1.0 / (1 + phi + phi * phi_capture_reshaped)
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
                phi_capture_unconstrained = numpyro.sample(
                    "phi_capture_unconstrained",
                    dist.Normal(*phi_capture_prior_params),
                )
                # Transform to constrained space
                phi_capture = numpyro.deterministic(
                    "phi_capture", jnp.exp(phi_capture_unconstrained)
                )
                # Reshape phi_capture for broadcasting to genes
                phi_capture_reshaped = phi_capture[:, None]
                # Compute effective success probability for each cell/gene
                p_hat = numpyro.deterministic(
                    "p_hat", 1.0 / (1 + phi + phi * phi_capture_reshaped)
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
            phi_capture_unconstrained = numpyro.sample(
                "phi_capture_unconstrained",
                dist.Normal(*phi_capture_prior_params),
            )
            # Transform to constrained space
            phi_capture = numpyro.deterministic(
                "phi_capture", jnp.exp(phi_capture_unconstrained)
            )
            # Reshape phi_capture for broadcasting to genes
            phi_capture_reshaped = phi_capture[:, None]
            # Compute effective success probability for each cell/gene
            p_hat = numpyro.deterministic(
                "p_hat", 1.0 / (1 + phi + phi * phi_capture_reshaped)
            )
            # Define base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Define zero-inflated NB distribution
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Sample counts (not observed)
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
    Mean-field variational guide for the unconstrained odds ratio Zero-Inflated
    Negative Binomial with Variable Capture Probability (ZINBVCP) model.

    This guide defines a mean-field variational approximation for the ZINBVCP
    model, which extends the standard Zero-Inflated Negative Binomial (ZINB)
    model by introducing a cell-specific mRNA capture probability. The model is
    parameterized in an unconstrained space to facilitate variational inference.

    The generative model assumes that the observed gene expression counts for
    each cell are generated from a Zero-Inflated Negative Binomial distribution,
    where the mean (mu), odds ratio parameter (phi), zero-inflation probability
    (gate), and cell-specific capture probability (phi_capture) are linked
    through unconstrained parameters. The dispersion parameter r is defined as:

        r = mu * phi

    where:
        - mu > 0 is the mean expression for each gene,
        - phi > 0 is the odds ratio parameter that controls dispersion,
        - r > 0 is the dispersion parameter.

    The model also includes a zero-inflation parameter (gate) for each gene,
    which controls the probability of excess zeros:

        - gate ∈ (0, 1) is the zero-inflation probability for each gene.

    In addition, each cell has a capture probability phi_capture > 0, which
    modifies the effective success probability for each cell and gene. The
    effective success probability for cell i and gene j is:

        p_hat[i, j] = 1 / (1 + phi + phi * phi_capture[i])

    The variational guide defines distributions for the unconstrained
    parameters:
        - phi_unconstrained ~ Normal(loc, scale)
        - mu_unconstrained ~ Normal(loc, scale) for each gene
        - gate_unconstrained ~ Normal(loc, scale) for each gene
        - phi_capture_unconstrained ~ Normal(loc, scale) for each cell

    These are transformed to the constrained space as follows:
        - phi = exp(phi_unconstrained) > 0
        - mu = exp(mu_unconstrained) > 0
        - gate = sigmoid(gate_unconstrained) ∈ (0, 1)
        - phi_capture = exp(phi_capture_unconstrained) > 0

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
    phi_guide_params = model_config.phi_unconstrained_guide or (0.0, 1.0)
    mu_guide_params = model_config.mu_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)
    phi_capture_guide_params = model_config.phi_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register unconstrained phi parameters
    phi_loc = numpyro.param("phi_unconstrained_loc", phi_guide_params[0])
    phi_scale = numpyro.param(
        "phi_unconstrained_scale",
        phi_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("phi_unconstrained", dist.Normal(phi_loc, phi_scale))

    # Register unconstrained mu parameters
    mu_loc = numpyro.param(
        "mu_unconstrained_loc", jnp.full(n_genes, mu_guide_params[0])
    )
    mu_scale = numpyro.param(
        "mu_unconstrained_scale",
        jnp.full(n_genes, mu_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("mu_unconstrained", dist.Normal(mu_loc, mu_scale))

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
    phi_capture_loc = numpyro.param(
        "phi_capture_unconstrained_loc",
        jnp.full(n_cells, phi_capture_guide_params[0]),
    )
    phi_capture_scale = numpyro.param(
        "phi_capture_unconstrained_scale",
        jnp.full(n_cells, phi_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample phi_capture depending on batch size
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
    Construct posterior distributions for model parameters from variational
    guide outputs.

    This function is specific to the 'odds_ratio_unconstrained' parameterization
    and builds the appropriate `numpyro` distributions based on the guide
    parameters found in the `params` dictionary. It handles both single and
    mixture models.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Dictionary containing estimated variational parameters (means and
        standard deviations) for each unconstrained latent variable, as produced
        by the guide. Expected keys include:
            - "phi_unconstrained_loc", "phi_unconstrained_scale"
            - "mu_unconstrained_loc", "mu_unconstrained_scale"
            - "gate_unconstrained_loc", "gate_unconstrained_scale"
            - "phi_capture_unconstrained_loc", "phi_capture_unconstrained_scale"
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
        distributions["phi_unconstrained"] = dist.Normal(
            params["phi_unconstrained_loc"], params["phi_unconstrained_scale"]
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
            # Cell-specific phi_capture parameters
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

    return distributions
