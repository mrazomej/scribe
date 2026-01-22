"""Guide builder for composing NumPyro variational guides.

This module provides a builder pattern for constructing NumPyro guide functions
from parameter specifications. It uses multiple dispatch to route guide setup
to the appropriate implementation based on (SpecType, GuideFamily) pairs.

Classes
-------
GuideBuilder
    Builder for constructing NumPyro guide functions.

Functions
---------
setup_guide
    Setup guide for a parameter (dispatched on spec type and guide family).
setup_cell_specific_guide
    Setup guide for cell-specific parameters with batch indexing support.

Examples
--------
>>> from scribe.models.builders import GuideBuilder, BetaSpec, LogNormalSpec
>>> from scribe.models.components import MeanFieldGuide, LowRankGuide
>>>
>>> specs = [
...     BetaSpec("p", (), (1.0, 1.0), guide_family=MeanFieldGuide()),
...     LogNormalSpec("r", ("n_genes",), (0.0, 1.0), guide_family=LowRankGuide(rank=10)),
... ]
>>> guide = GuideBuilder().from_specs(specs).build()

See Also
--------
scribe.models.builders.parameter_specs : Parameter specification classes.
scribe.models.components.guide_families : Guide family markers.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch
from numpyro.distributions import constraints

from .parameter_specs import (
    BetaPrimeSpec,
    BetaSpec,
    DirichletSpec,
    ExpNormalSpec,
    LogNormalSpec,
    NormalWithTransformSpec,
    ParamSpec,
    SigmoidNormalSpec,
    resolve_shape,
)
from scribe.stats.distributions import BetaPrime
from ..components.guide_families import (
    AmortizedGuide,
    GuideFamily,
    LowRankGuide,
    MeanFieldGuide,
)

if TYPE_CHECKING:
    from ..components.amortizers import Amortizer
    from ..config import ModelConfig


# ==============================================================================
# Multiple Dispatch: (SpecType, GuideFamily) -> Implementation
# ==============================================================================

# ------------------------------------------------------------------------------
# Beta Distribution MeanField Guide
# ------------------------------------------------------------------------------


@dispatch(BetaSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: BetaSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for Beta parameter.

    Creates learnable variational parameters (alpha, beta) for a Beta
    distribution approximation.

    Parameters
    ----------
    spec : BetaSpec
        Parameter specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from variational distribution.

    Notes
    -----
    Beta distribution parameters (alpha, beta) must be positive.
    The sampled value will satisfy spec.constraint (unit_interval).
    """
    params = spec.guide if spec.guide is not None else spec.default_params
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    # Variational params for Beta must be positive
    alpha = numpyro.param(
        f"{spec.name}_alpha",
        jnp.full(shape, params[0]) if shape else jnp.array(params[0]),
        constraint=constraints.positive,
    )
    beta = numpyro.param(
        f"{spec.name}_beta",
        jnp.full(shape, params[1]) if shape else jnp.array(params[1]),
        constraint=constraints.positive,
    )

    if shape == ():
        return numpyro.sample(spec.name, dist.Beta(alpha, beta))
    return numpyro.sample(
        spec.name, dist.Beta(alpha, beta).to_event(len(shape))
    )


# ------------------------------------------------------------------------------
# BetaPrime Distribution MeanField Guide
# ------------------------------------------------------------------------------


@dispatch(BetaPrimeSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: BetaPrimeSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for BetaPrime parameter.

    Creates learnable variational parameters (alpha, beta) for a BetaPrime
    distribution approximation.

    Parameters
    ----------
    spec : BetaPrimeSpec
        Parameter specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from variational distribution.
    """
    params = spec.guide if spec.guide is not None else spec.default_params
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    # Variational params for BetaPrime must be positive
    alpha = numpyro.param(
        f"{spec.name}_alpha",
        jnp.full(shape, params[0]) if shape else jnp.array(params[0]),
        constraint=constraints.positive,
    )
    beta = numpyro.param(
        f"{spec.name}_beta",
        jnp.full(shape, params[1]) if shape else jnp.array(params[1]),
        constraint=constraints.positive,
    )

    # Sample from BetaPrime distribution for scalar parameters
    if shape == ():
        return numpyro.sample(spec.name, BetaPrime(alpha, beta))
    # Sample from BetaPrime distribution for non-scalar parameters
    return numpyro.sample(
        spec.name, BetaPrime(alpha, beta).to_event(len(shape))
    )


# ------------------------------------------------------------------------------
# LogNormal Distribution MeanField Guide
# ------------------------------------------------------------------------------


@dispatch(LogNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: LogNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for LogNormal parameter.

    Creates learnable variational parameters (loc, scale) for a LogNormal
    distribution approximation.

    Parameters
    ----------
    spec : LogNormalSpec
        Parameter specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from variational distribution.

    Notes
    -----
    scale must be positive. The sampled value will satisfy
    spec.constraint (positive).
    """
    params = spec.guide if spec.guide is not None else spec.default_params
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    loc = numpyro.param(f"{spec.name}_loc", jnp.full(shape, params[0]))
    scale = numpyro.param(
        f"{spec.name}_scale",
        jnp.full(shape, params[1]),
        constraint=constraints.positive,
    )

    return numpyro.sample(
        spec.name, dist.LogNormal(loc, scale).to_event(len(shape))
    )


# ------------------------------------------------------------------------------
# Normal with Transform Distribution MeanField Guide
# ------------------------------------------------------------------------------


@dispatch(NormalWithTransformSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: NormalWithTransformSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for transformed Normal parameters.

    Uses TransformedDistribution(Normal, spec.transform) for:
    - Automatic Jacobian handling in ELBO
    - Samples directly in constrained space
    - Constraint derived from spec.transform.codomain

    Works for SigmoidNormalSpec, ExpNormalSpec, SoftplusNormalSpec, etc.

    Parameters
    ----------
    spec : NormalWithTransformSpec
        Parameter specification (or subclass).
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space.
    """
    params = spec.guide if spec.guide is not None else spec.default_params
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    # Variational parameters for the base Normal
    loc = numpyro.param(
        f"{spec.name}_loc",
        jnp.full(shape, params[0]) if shape else jnp.array(params[0]),
    )
    scale = numpyro.param(
        f"{spec.name}_scale",
        jnp.full(shape, params[1]) if shape else jnp.array(params[1]),
        constraint=constraints.positive,
    )

    # Create base Normal and wrap with transform
    if shape == ():
        base_dist = dist.Normal(loc, scale)
    else:
        base_dist = dist.Normal(loc, scale).to_event(len(shape))

    transformed_dist = dist.TransformedDistribution(base_dist, spec.transform)

    # Sample in constrained space (Jacobian handled automatically)
    return numpyro.sample(spec.constrained_name, transformed_dist)


# ------------------------------------------------------------------------------
# LogNormal Distribution LowRank Guide
# ------------------------------------------------------------------------------


@dispatch(LogNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: LogNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for LogNormal - captures gene correlations.

    Uses a low-rank multivariate normal in log-space with covariance:
        Σ = W @ W.T + diag(D)

    This captures correlations between genes with O(n × rank) memory
    instead of O(n²) for full covariance.

    Parameters
    ----------
    spec : LogNormalSpec
        Parameter specification.
    guide : LowRankGuide
        Low-rank guide marker with rank attribute.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from low-rank variational distribution.
    """
    resolved_shape = resolve_shape(
        spec.shape_dims, dims, is_mixture=spec.is_mixture
    )
    k = guide.rank

    # For mixture models, resolved_shape is (n_components, n_genes)
    # For non-mixture, resolved_shape is (n_genes,)
    if spec.is_mixture:
        # Mixture model: shape is (n_components, n_genes)
        n_components = resolved_shape[0]
        n_genes = resolved_shape[1] if len(resolved_shape) > 1 else 1

        # Variational parameters with shape (n_components, n_genes)
        loc = numpyro.param(
            f"log_{spec.name}_loc", jnp.zeros((n_components, n_genes))
        )
        W = numpyro.param(
            f"log_{spec.name}_W", 0.01 * jnp.ones((n_components, n_genes, k))
        )
        raw_diag = numpyro.param(
            f"log_{spec.name}_raw_diag",
            -3.0 * jnp.ones((n_components, n_genes)),
        )

        # Ensure diagonal is positive
        D = jax.nn.softplus(raw_diag) + 1e-4

        # Create low-rank MVN:
        # batch_shape=(n_components,), event_shape=(n_genes,)
        base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
        # Use to_event(1) to convert batch dimension to event, resulting in
        # event_shape=(n_components, n_genes) to match model's event_dims=2
        transformed_dist = dist.TransformedDistribution(
            base, dist.transforms.ExpTransform()
        ).to_event(1)
    else:
        # Non-mixture: shape is (n_genes,)
        G = resolved_shape[0] if resolved_shape else 1

        # Low-rank MVN parameters in log-space
        loc = numpyro.param(f"log_{spec.name}_loc", jnp.zeros(G))
        W = numpyro.param(f"log_{spec.name}_W", 0.01 * jnp.ones((G, k)))
        raw_diag = numpyro.param(
            f"log_{spec.name}_raw_diag", -3.0 * jnp.ones(G)
        )

        # Ensure diagonal is positive
        D = jax.nn.softplus(raw_diag) + 1e-4

        # Create low-rank MVN and transform to positive via exp
        base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
        transformed_dist = dist.TransformedDistribution(
            base, dist.transforms.ExpTransform()
        )

    return numpyro.sample(spec.name, transformed_dist)


# ------------------------------------------------------------------------------
# Normal with Transform Distribution LowRank Guide (Unconstrained Models)
# ------------------------------------------------------------------------------


@dispatch(NormalWithTransformSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: NormalWithTransformSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for unconstrained parameters (Normal + Transform).

    Uses a low-rank multivariate normal wrapped with TransformedDistribution to
    handle the transform to constrained space. The covariance structure is:
        Σ = W @ W.T + diag(D)

    This matches the pattern used for mean-field guides, ensuring consistent
    Jacobian handling via TransformedDistribution.

    Parameters
    ----------
    spec : NormalWithTransformSpec
        Parameter specification (e.g., ExpNormalSpec for positive parameters).
    guide : LowRankGuide
        Low-rank guide marker with rank attribute.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space (transform applied via
        TransformedDistribution).

    Notes
    -----
    This handler enables low-rank guides for unconstrained models (e.g., models
    using ExpNormalSpec, SigmoidNormalSpec). The low-rank MVN is defined in
    unconstrained space, then wrapped with TransformedDistribution to apply the
    transform (exp, sigmoid, etc.) and handle Jacobian automatically.

    Examples
    --------
    Works with ExpNormalSpec for positive parameters:
        >>> spec = ExpNormalSpec("r", ("n_genes",), (0.0, 1.0))
        >>> guide = LowRankGuide(rank=10)
        >>> setup_guide(spec, guide, {"n_genes": 100}, model_config)
        # Samples "r" from TransformedDistribution(LowRankMultivariateNormal, ExpTransform)
    """
    resolved_shape = resolve_shape(
        spec.shape_dims, dims, is_mixture=spec.is_mixture
    )
    k = guide.rank

    # For mixture models, resolved_shape is (n_components, n_genes)
    # For non-mixture, resolved_shape is (n_genes,)
    if spec.is_mixture:
        # Mixture model: shape is (n_components, n_genes)
        n_components = resolved_shape[0]
        n_genes = resolved_shape[1] if len(resolved_shape) > 1 else 1

        # Variational parameters with shape (n_components, n_genes)
        # Use base parameter name for variational parameters (matching
        # mean-field pattern)
        loc = numpyro.param(
            f"{spec.name}_loc", jnp.zeros((n_components, n_genes))
        )
        W = numpyro.param(
            f"{spec.name}_W", 0.01 * jnp.ones((n_components, n_genes, k))
        )
        raw_diag = numpyro.param(
            f"{spec.name}_raw_diag", -3.0 * jnp.ones((n_components, n_genes))
        )

        # Ensure diagonal is positive
        D = jax.nn.softplus(raw_diag) + 1e-4

        # Create low-rank MVN in unconstrained space
        # batch_shape=(n_components,), event_shape=(n_genes,)
        base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

        # Wrap with transform - handles Jacobian automatically
        # Use to_event(1) to convert batch dimension to event, resulting in
        # event_shape=(n_components, n_genes) to match model's event_dims=2
        transformed_dist = dist.TransformedDistribution(
            base, spec.transform
        ).to_event(1)
    else:
        # Non-mixture: shape is (n_genes,)
        G = resolved_shape[0] if resolved_shape else 1

        # Low-rank MVN parameters in unconstrained space
        # Use base parameter name for variational parameters (matching
        # mean-field pattern)
        loc = numpyro.param(f"{spec.name}_loc", jnp.zeros(G))
        W = numpyro.param(f"{spec.name}_W", 0.01 * jnp.ones((G, k)))
        raw_diag = numpyro.param(f"{spec.name}_raw_diag", -3.0 * jnp.ones(G))

        # Ensure diagonal is positive
        D = jax.nn.softplus(raw_diag) + 1e-4

        # Create low-rank MVN in unconstrained space
        base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

        # Wrap with transform - handles Jacobian automatically
        transformed_dist = dist.TransformedDistribution(base, spec.transform)

    # Sample in constrained space (transform applied internally)
    return numpyro.sample(spec.constrained_name, transformed_dist)


# ==============================================================================
# Cell-Specific Parameter Guides (with batch indexing support)
# ==============================================================================


@dispatch(BetaSpec, MeanFieldGuide, dict, object)
def setup_cell_specific_guide(
    spec: BetaSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for cell-specific Beta parameter (e.g., p_capture).

    When batch_idx is provided, we index into the full parameter arrays
    to get only the parameters for the current mini-batch.

    Parameters
    ----------
    spec : BetaSpec
        Parameter specification (must have is_cell_specific=True).
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimensions including n_cells.
    model_config : ModelConfig
        Model configuration.
    batch_idx : Optional[jnp.ndarray]
        Indices of cells in current mini-batch. None for full sampling.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value for the current batch.
    """
    n_cells = dims["n_cells"]
    params = spec.guide if spec.guide is not None else spec.default_params

    # Variational parameters for ALL cells (allocated once, indexed into)
    alpha = numpyro.param(
        f"{spec.name}_alpha",
        jnp.full(n_cells, params[0]),
        constraint=constraints.positive,
    )
    beta = numpyro.param(
        f"{spec.name}_beta",
        jnp.full(n_cells, params[1]),
        constraint=constraints.positive,
    )

    if batch_idx is None:
        # Full sampling: use all parameters
        return numpyro.sample(spec.name, dist.Beta(alpha, beta))
    else:
        # Batch sampling: index into parameters for this mini-batch
        return numpyro.sample(
            spec.name, dist.Beta(alpha[batch_idx], beta[batch_idx])
        )


# ------------------------------------------------------------------------------
# BetaPrime Distribution MeanField Guide (Cell-Specific)
# ------------------------------------------------------------------------------


@dispatch(BetaPrimeSpec, MeanFieldGuide, dict, object)
def setup_cell_specific_guide(
    spec: BetaPrimeSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """
    MeanField guide for cell-specific BetaPrime parameter (e.g., phi_capture).

    When batch_idx is provided, we index into the full parameter arrays to get
    only the parameters for the current mini-batch.

    Parameters
    ----------
    spec : BetaPrimeSpec
        Parameter specification (must have is_cell_specific=True).
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimensions including n_cells.
    model_config : ModelConfig
        Model configuration.
    batch_idx : Optional[jnp.ndarray]
        Indices of cells in current mini-batch. None for full sampling.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value for the current batch.
    """
    n_cells = dims["n_cells"]
    params = spec.guide if spec.guide is not None else spec.default_params

    # Variational parameters for ALL cells (allocated once, indexed into)
    alpha = numpyro.param(
        f"{spec.name}_alpha",
        jnp.full(n_cells, params[0]),
        constraint=constraints.positive,
    )
    beta = numpyro.param(
        f"{spec.name}_beta",
        jnp.full(n_cells, params[1]),
        constraint=constraints.positive,
    )

    if batch_idx is None:
        # Full sampling: use all parameters
        return numpyro.sample(spec.name, BetaPrime(alpha, beta))
    else:
        # Batch sampling: index into parameters for this mini-batch
        return numpyro.sample(
            spec.name, BetaPrime(alpha[batch_idx], beta[batch_idx])
        )


# ------------------------------------------------------------------------------
# Beta Distribution Amortized Guide
# ------------------------------------------------------------------------------


@dispatch(BetaSpec, AmortizedGuide, dict, object)
def setup_cell_specific_guide(
    spec: BetaSpec,
    guide: AmortizedGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    counts: Optional[jnp.ndarray] = None,
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """Amortized guide for cell-specific Beta parameter (e.g., p_capture).

    Uses an amortizer network to predict variational parameters from
    sufficient statistics of the data.

    Parameters
    ----------
    spec : BetaSpec
        Parameter specification.
    guide : AmortizedGuide
        Amortized guide with attached amortizer network.
    dims : Dict[str, int]
        Dimensions.
    model_config : ModelConfig
        Model configuration.
    counts : jnp.ndarray
        Count data (used to compute sufficient statistics).
    batch_idx : Optional[jnp.ndarray]
        Indices for mini-batch. Amortizer processes batched data directly.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value.

    Raises
    ------
    ValueError
        If counts is None (required for amortization).
    """
    if counts is None:
        raise ValueError("Amortized guide requires counts data")

    # Get data for current batch (amortizer handles per-cell computation)
    data = counts if batch_idx is None else counts[batch_idx]

    # Amortizer predicts variational params from sufficient statistics
    var_params = guide.amortizer(data)
    alpha = jnp.exp(var_params["log_alpha"])
    beta = jnp.exp(var_params["log_beta"])

    return numpyro.sample(spec.name, dist.Beta(alpha, beta))


# ------------------------------------------------------------------------------
# BetaPrime Distribution Amortized Guide
# ------------------------------------------------------------------------------


@dispatch(BetaPrimeSpec, AmortizedGuide, dict, object)
def setup_cell_specific_guide(
    spec: BetaPrimeSpec,
    guide: AmortizedGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    counts: Optional[jnp.ndarray] = None,
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """Amortized guide for cell-specific BetaPrime parameter (e.g., phi_capture).

    Uses an amortizer network to predict variational parameters from
    sufficient statistics of the data.

    Parameters
    ----------
    spec : BetaPrimeSpec
        Parameter specification.
    guide : AmortizedGuide
        Amortized guide with attached amortizer network.
    dims : Dict[str, int]
        Dimensions.
    model_config : ModelConfig
        Model configuration.
    counts : jnp.ndarray
        Count data (used to compute sufficient statistics).
    batch_idx : Optional[jnp.ndarray]
        Indices for mini-batch. Amortizer processes batched data directly.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value.

    Raises
    ------
    ValueError
        If counts is None (required for amortization).
    """
    if counts is None:
        raise ValueError("Amortized guide requires counts data")

    # Get data for current batch (amortizer handles per-cell computation)
    data = counts if batch_idx is None else counts[batch_idx]

    # Amortizer predicts variational params from sufficient statistics
    var_params = guide.amortizer(data)
    alpha = jnp.exp(var_params["log_alpha"])
    beta = jnp.exp(var_params["log_beta"])

    return numpyro.sample(spec.name, BetaPrime(alpha, beta))


# ------------------------------------------------------------------------------
# SigmoidNormal Distribution Amortized Guide
# ------------------------------------------------------------------------------


@dispatch(SigmoidNormalSpec, AmortizedGuide, dict, object)
def setup_cell_specific_guide(
    spec: SigmoidNormalSpec,
    guide: AmortizedGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    counts: Optional[jnp.ndarray] = None,
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """Amortized guide for cell-specific SigmoidNormal parameter (e.g., p_capture).

    Uses an amortizer network to predict variational parameters from
    sufficient statistics of the data. Samples from Normal, then applies
    sigmoid transform to constrain to [0, 1].

    Parameters
    ----------
    spec : SigmoidNormalSpec
        Parameter specification.
    guide : AmortizedGuide
        Amortized guide with attached amortizer network.
    dims : Dict[str, int]
        Dimensions.
    model_config : ModelConfig
        Model configuration.
    counts : jnp.ndarray
        Count data (used to compute sufficient statistics).
    batch_idx : Optional[jnp.ndarray]
        Indices for mini-batch. Amortizer processes batched data directly.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space [0, 1].

    Raises
    ------
    ValueError
        If counts is None (required for amortization).
    """
    if counts is None:
        raise ValueError("Amortized guide requires counts data")

    # Get data for current batch (amortizer handles per-cell computation)
    data = counts if batch_idx is None else counts[batch_idx]

    # Amortizer predicts variational params from sufficient statistics
    var_params = guide.amortizer(data)
    loc = var_params["loc"]
    scale = jnp.exp(var_params["log_scale"])

    # Create base Normal distribution and apply sigmoid transform
    base_dist = dist.Normal(loc, scale)
    transformed_dist = dist.TransformedDistribution(base_dist, spec.transform)
    return numpyro.sample(spec.constrained_name, transformed_dist)


# ------------------------------------------------------------------------------
# ExpNormal Distribution Amortized Guide
# ------------------------------------------------------------------------------


@dispatch(ExpNormalSpec, AmortizedGuide, dict, object)
def setup_cell_specific_guide(
    spec: ExpNormalSpec,
    guide: AmortizedGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    counts: Optional[jnp.ndarray] = None,
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """Amortized guide for cell-specific ExpNormal parameter (e.g., phi_capture).

    Uses an amortizer network to predict variational parameters from
    sufficient statistics of the data. Samples from Normal, then applies
    exp transform to constrain to [0, +∞).

    Parameters
    ----------
    spec : ExpNormalSpec
        Parameter specification.
    guide : AmortizedGuide
        Amortized guide with attached amortizer network.
    dims : Dict[str, int]
        Dimensions.
    model_config : ModelConfig
        Model configuration.
    counts : jnp.ndarray
        Count data (used to compute sufficient statistics).
    batch_idx : Optional[jnp.ndarray]
        Indices for mini-batch. Amortizer processes batched data directly.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space [0, +∞).

    Raises
    ------
    ValueError
        If counts is None (required for amortization).
    """
    if counts is None:
        raise ValueError("Amortized guide requires counts data")

    # Get data for current batch (amortizer handles per-cell computation)
    data = counts if batch_idx is None else counts[batch_idx]

    # Amortizer predicts variational params from sufficient statistics
    var_params = guide.amortizer(data)
    loc = var_params["loc"]
    scale = jnp.exp(var_params["log_scale"])

    # Create base Normal distribution and apply exp transform
    base_dist = dist.Normal(loc, scale)
    transformed_dist = dist.TransformedDistribution(base_dist, spec.transform)
    return numpyro.sample(spec.constrained_name, transformed_dist)


# ------------------------------------------------------------------------------
# Normal with Transform Distribution MeanField Guide (Cell-Specific)
# ------------------------------------------------------------------------------


@dispatch(NormalWithTransformSpec, MeanFieldGuide, dict, object)
def setup_cell_specific_guide(
    spec: NormalWithTransformSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for cell-specific unconstrained parameters.

    Works for SigmoidNormalSpec (Beta -> [0,1]), ExpNormalSpec (BetaPrime -> [0,+inf)),
    and other NormalWithTransformSpec subclasses.

    Parameters
    ----------
    spec : NormalWithTransformSpec
        Parameter specification (SigmoidNormalSpec, ExpNormalSpec, etc.).
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimensions including n_cells.
    model_config : ModelConfig
        Model configuration.
    batch_idx : Optional[jnp.ndarray]
        Indices for mini-batch. None for full sampling.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space.
    """
    n_cells = dims["n_cells"]
    params = spec.guide if spec.guide is not None else spec.default_params

    loc = numpyro.param(f"{spec.name}_loc", jnp.full(n_cells, params[0]))
    scale = numpyro.param(
        f"{spec.name}_scale",
        jnp.full(n_cells, params[1]),
        constraint=constraints.positive,
    )

    if batch_idx is None:
        base_dist = dist.Normal(loc, scale)
    else:
        base_dist = dist.Normal(loc[batch_idx], scale[batch_idx])

    transformed_dist = dist.TransformedDistribution(base_dist, spec.transform)
    return numpyro.sample(spec.constrained_name, transformed_dist)


# ------------------------------------------------------------------------------
# Dirichlet Distribution MeanField Guide
# ------------------------------------------------------------------------------


@dispatch(DirichletSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: DirichletSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for Dirichlet parameter.

    Creates learnable variational parameters (concentrations) for a Dirichlet
    distribution approximation. Used for mixture weights and compositional
    parameters.

    Parameters
    ----------
    spec : DirichletSpec
        Parameter specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from Dirichlet distribution (on simplex).

    Notes
    -----
    - Concentration parameters must all be positive.
    - The sampled value will be on the simplex (sums to 1).
    - For mixture weights, the concentration vector length equals n_components.
    """
    # Get guide params or use default
    params = spec.guide if spec.guide is not None else spec.default_params

    # Convert tuple to array and ensure all values are positive
    concentrations = jnp.array(params)

    # Determine parameter name: use "mixing_concentrations" for mixing_weights
    # (matching legacy code), otherwise use "{name}_concentrations"
    if spec.name == "mixing_weights":
        param_name = "mixing_concentrations"
    else:
        param_name = f"{spec.name}_concentrations"

    # Create variational parameter with positive constraint
    concentrations_param = numpyro.param(
        param_name,
        concentrations,
        constraint=constraints.positive,
    )

    # Sample from Dirichlet distribution
    return numpyro.sample(spec.name, dist.Dirichlet(concentrations_param))


# ==============================================================================
# Guide Builder
# ==============================================================================


class GuideBuilder:
    """Builder for constructing NumPyro guide functions from parameter specs.

    The GuideBuilder takes parameter specifications and constructs a guide
    function that samples from the variational posterior. Each parameter can
    have its own guide family (MeanField, LowRank, Amortized).

    Attributes
    ----------
    param_specs : List[ParamSpec]
        Parameter specifications for the guide.

    Examples
    --------
    >>> from scribe.models.builders import GuideBuilder, BetaSpec, LogNormalSpec
    >>> from scribe.models.components import MeanFieldGuide, LowRankGuide
    >>>
    >>> specs = [
    ...     BetaSpec("p", (), (1.0, 1.0), guide_family=MeanFieldGuide()),
    ...     LogNormalSpec("r", ("n_genes",), (0.0, 1.0), guide_family=LowRankGuide(rank=10)),
    ... ]
    >>> guide = GuideBuilder().from_specs(specs).build()
    """

    def __init__(self):
        """Initialize an empty GuideBuilder."""
        self.param_specs: List[ParamSpec] = []

    # --------------------------------------------------------------------------

    @property
    def is_mixture(self) -> bool:
        """Check if any parameter is mixture-specific.

        Returns
        -------
        bool
            True if any parameter has is_mixture=True.
        """
        return any(spec.is_mixture for spec in self.param_specs)

    # --------------------------------------------------------------------------

    def from_specs(self, specs: List[ParamSpec]) -> "GuideBuilder":
        """Set parameter specifications from a list.

        Parameters
        ----------
        specs : List[ParamSpec]
            List of parameter specifications.

        Returns
        -------
        GuideBuilder
            Self, for method chaining.
        """
        self.param_specs = specs
        return self

    # --------------------------------------------------------------------------

    def add_param(self, spec: ParamSpec) -> "GuideBuilder":
        """Add a single parameter specification.

        Parameters
        ----------
        spec : ParamSpec
            Parameter specification to add.

        Returns
        -------
        GuideBuilder
            Self, for method chaining.
        """
        self.param_specs.append(spec)
        return self

    # --------------------------------------------------------------------------

    def build(self) -> Callable:
        """Build and return the NumPyro guide function.

        Returns
        -------
        Callable
            A NumPyro guide function with signature:
            guide(n_cells, n_genes, model_config, counts=None, batch_size=None)
        """
        specs = self.param_specs

        def guide(
            n_cells: int,
            n_genes: int,
            model_config: "ModelConfig",
            counts: Optional[jnp.ndarray] = None,
            batch_size: Optional[int] = None,
        ):
            """NumPyro guide function.

            Parameters
            ----------
            n_cells : int
                Total number of cells in the dataset.
            n_genes : int
                Number of genes.
            model_config : ModelConfig
                Configuration containing guide hyperparameters.
            counts : Optional[jnp.ndarray], shape (n_cells, n_genes)
                Observed count matrix (needed for amortized guides).
            batch_size : Optional[int]
                Mini-batch size for stochastic VI.
            """
            # ================================================================
            # Setup dimensions dict
            # ================================================================
            dims = {"n_cells": n_cells, "n_genes": n_genes}
            if (
                hasattr(model_config, "n_components")
                and model_config.n_components
            ):
                dims["n_components"] = model_config.n_components

            # ================================================================
            # 0. Setup guide for MIXING WEIGHTS if this is a mixture model
            # ================================================================
            is_mixture = any(s.is_mixture for s in specs)
            if is_mixture:
                if "n_components" not in dims:
                    raise ValueError(
                        "n_components must be set in model_config when "
                        "using mixture parameters"
                    )
                # Check if mixing_weights spec exists
                mixing_spec = next(
                    (s for s in specs if s.name == "mixing_weights"), None
                )
                if mixing_spec is None:
                    # Create default Dirichlet spec for mixing weights
                    n_components = dims["n_components"]
                    # Get mixing prior from param_specs if available
                    mixing_prior_params = None
                    if model_config.param_specs:
                        for spec in model_config.param_specs:
                            if spec.name == "mixing" and spec.prior is not None:
                                mixing_prior_params = spec.prior
                                break
                    if mixing_prior_params is None:
                        mixing_prior_params = tuple([1.0] * n_components)
                    from .parameter_specs import DirichletSpec

                    mixing_spec = DirichletSpec(
                        name="mixing_weights",
                        shape_dims=(),
                        default_params=mixing_prior_params,
                        prior=mixing_prior_params,
                        is_mixture=False,  # Mixing weights are not mixture-specific
                    )
                guide_family = mixing_spec.guide_family or MeanFieldGuide()
                setup_guide(mixing_spec, guide_family, dims, model_config)

            # ================================================================
            # 1. Setup guides for GLOBAL parameters
            # ================================================================
            global_specs = [
                s
                for s in specs
                if not s.is_gene_specific
                and not s.is_cell_specific
                and s.name != "mixing_weights"  # Already handled above
            ]
            for spec in global_specs:
                guide_family = spec.guide_family or MeanFieldGuide()
                setup_guide(spec, guide_family, dims, model_config)

            # ================================================================
            # 2. Setup guides for GENE-SPECIFIC parameters
            # ================================================================
            gene_specs = [s for s in specs if s.is_gene_specific]
            for spec in gene_specs:
                guide_family = spec.guide_family or MeanFieldGuide()
                setup_guide(spec, guide_family, dims, model_config)

            # ================================================================
            # 3. Setup guides for CELL-SPECIFIC parameters (inside cell plate)
            #    Handle batch indexing for non-amortized guides
            # ================================================================
            cell_specs = [s for s in specs if s.is_cell_specific]
            if cell_specs:
                if batch_size is None:
                    # Full sampling
                    with numpyro.plate("cells", n_cells):
                        for spec in cell_specs:
                            guide_family = spec.guide_family or MeanFieldGuide()
                            setup_cell_specific_guide(
                                spec,
                                guide_family,
                                dims,
                                model_config,
                                counts=counts,
                                batch_idx=None,
                            )
                else:
                    # Batch sampling
                    with numpyro.plate(
                        "cells", n_cells, subsample_size=batch_size
                    ) as idx:
                        for spec in cell_specs:
                            guide_family = spec.guide_family or MeanFieldGuide()
                            setup_cell_specific_guide(
                                spec,
                                guide_family,
                                dims,
                                model_config,
                                counts=counts,
                                batch_idx=idx,
                            )

        return guide
