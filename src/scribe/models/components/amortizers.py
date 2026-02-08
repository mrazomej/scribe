"""
General amortization framework for variational inference.

This module provides infrastructure for amortized inference, where neural
networks predict variational parameters from data features (sufficient
statistics) instead of learning separate parameters for each data point.

The key components are:

1. **SufficientStatistic**: Defines how to compute summary statistics from data
2. **Amortizer**: Neural network that maps statistics to variational parameters

This framework is designed to be general and extensible for:
- Cell-specific parameters like p_capture (using total UMI count)
- Future: Component-specific parameters in mixture models
- Future: VAE-style joint amortization

Classes
-------
AmortizedOutput
    Contract for amortizer return value (keys and value space semantics).
SufficientStatistic
    Defines computation of sufficient statistics from data.
Amortizer
    Neural network for predicting variational parameters.

Constants
---------
TOTAL_COUNT
    Built-in sufficient statistic: log(1 + sum(counts)).

Examples
--------
>>> from scribe.models.components import Amortizer, TOTAL_COUNT
>>> # Create an amortizer for p_capture using total UMI count
>>> amortizer = Amortizer(
...     sufficient_statistic=TOTAL_COUNT,
...     hidden_dims=[64, 32],
...     output_params=["alpha", "beta"],
... )
>>> # Use with AmortizedGuide
>>> guide_family = AmortizedGuide(amortizer=amortizer)

See Also
--------
scribe.models.components.guide_families.AmortizedGuide : Guide using amortizers.

Notes
-----
The theoretical justification for using total UMI count as a sufficient
statistic for p_capture is given in the paper's Dirichlet-Multinomial
derivation. Briefly, in the NB-DM model, the marginal distribution of
total counts depends only on p_capture, making it a sufficient statistic.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import linen as nn

# ------------------------------------------------------------------------------
# Sufficient Statistic Base Class
# ------------------------------------------------------------------------------


@dataclass
class SufficientStatistic:
    """
    Defines how to compute a sufficient statistic from data.

    A sufficient statistic is a function of the data that captures all
    information relevant for estimating a particular parameter. Using sufficient
    statistics for amortization is both theoretically motivated and
    computationally efficient.

    Parameters
    ----------
    name : str
        Human-readable name for the statistic.
    compute : Callable[[jnp.ndarray], jnp.ndarray]
        Function that computes the statistic from count data.
        Input: counts array of shape (..., n_genes)
        Output: statistic array of shape (..., statistic_dim)

    Examples
    --------
    >>> # Total UMI count per cell (log-transformed for stability)
    >>> TOTAL_COUNT = SufficientStatistic(
    ...     name="total_count",
    ...     compute=lambda counts: jnp.log1p(counts.sum(axis=-1, keepdims=True))
    ... )
    >>> # Mean expression per cell
    >>> MEAN_EXPRESSION = SufficientStatistic(
    ...     name="mean_expression",
    ...     compute=lambda counts: counts.mean(axis=-1, keepdims=True)
    ... )
    """

    name: str
    compute: Callable[[jnp.ndarray], jnp.ndarray]


# ==============================================================================
# Built-in Sufficient Statistics
# ==============================================================================


def _compute_total_count(counts: jnp.ndarray) -> jnp.ndarray:
    """Compute log1p of total UMI count per cell.

    This is a regular function (not a lambda) for better JIT compilation
    performance and tracing compatibility.

    Parameters
    ----------
    counts : jnp.ndarray
        Count data of shape (..., n_genes).

    Returns
    -------
    jnp.ndarray
        Log-transformed total counts of shape (..., 1).
    """
    return jnp.log1p(counts.sum(axis=-1, keepdims=True))


# ------------------------------------------------------------------------------


def _compute_total_count_log(counts: jnp.ndarray) -> jnp.ndarray:
    """
    Compute log of total UMI count per cell (with small epsilon for stability).

    Module-level function (not lambda) for picklability and JIT compatibility.
    """
    return jnp.log(counts.sum(axis=-1, keepdims=True) + 1e-8)


# ------------------------------------------------------------------------------


def _compute_total_count_sqrt(counts: jnp.ndarray) -> jnp.ndarray:
    """Compute sqrt of total UMI count per cell.

    Module-level function (not lambda) for picklability and JIT compatibility.
    """
    return jnp.sqrt(counts.sum(axis=-1, keepdims=True))


# ------------------------------------------------------------------------------


def _compute_total_count_identity(counts: jnp.ndarray) -> jnp.ndarray:
    """Compute total UMI count per cell (no transform).

    Module-level function (not lambda) for picklability and JIT compatibility.
    """
    return counts.sum(axis=-1, keepdims=True)


# ------------------------------------------------------------------------------

TOTAL_COUNT = SufficientStatistic(
    name="total_count",
    compute=_compute_total_count,
)
"""Total UMI count per cell (log-transformed).

This is a natural sufficient statistic for the capture probability p_capture
in the NB-DM model. The total count T = Σᵢ xᵢ follows a Negative Binomial
distribution that depends on p_capture, making T sufficient for p_capture.

The log1p transform stabilizes the values for neural network input.

Examples
--------
>>> counts = jnp.array([[10, 20, 30], [5, 5, 5]])
>>> TOTAL_COUNT.compute(counts)
Array([[4.094345], [2.772589]], dtype=float32)
"""


# ------------------------------------------------------------------------------
# Identity Function for Transforms
# ------------------------------------------------------------------------------


def _identity_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Identity function for output transforms when no transform is specified.

    This is a regular function (not a lambda) for better JIT compilation
    performance and tracing compatibility.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        Same array (identity transform).
    """
    return x


# ------------------------------------------------------------------------------
# Activation Function Mapping
# ------------------------------------------------------------------------------


def _get_activation_fn(activation: str) -> Callable:
    """Map activation function name to JAX activation function.

    Parameters
    ----------
    activation : str
        Name of activation function (e.g., "relu", "gelu", "leaky_relu").

    Returns
    -------
    Callable
        JAX activation function.

    Raises
    ------
    ValueError
        If activation name is not supported.
    """
    ACTIVATION_MAP = {
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "swish": jax.nn.silu,  # swish is an alias for silu
        "tanh": jnp.tanh,
        "sigmoid": jax.nn.sigmoid,
        "elu": jax.nn.elu,
        "leaky_relu": jax.nn.leaky_relu,
        "softplus": jax.nn.softplus,
        "celu": jax.nn.celu,
        "selu": jax.nn.selu,
    }

    activation_lower = activation.lower()
    if activation_lower not in ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation '{activation}'. "
            f"Valid options: {list(ACTIVATION_MAP.keys())}"
        )

    return ACTIVATION_MAP[activation_lower]


# ------------------------------------------------------------------------------
# Amortizer Output Contract
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class AmortizedOutput:
    """Contract for amortizer return value: semantics of keys and value space.

    This is the single source of truth for what the amortizer outputs mean.
    Consumers (e.g. guide_builder) must not re-apply transforms that the
    amortizer already applied; see parameterization below.

    **Constrained parameterization** (keys ``alpha``, ``beta``):
        Values are already in **constrained (positive) space**. The amortizer
        applies output_transforms (e.g. softplus+offset+clamp or exp)
        internally. Consumers must use the values as-is (e.g. BetaPrime(alpha,
        beta)) and must **not** apply any positivity transform.

    **Unconstrained parameterization** (keys ``loc``, ``log_scale``):
        ``loc`` is in **unconstrained space** (real line).
        ``log_scale`` is in **log-space** (real line).
        Consumers must apply ``scale = exp(log_scale)`` before using as a
        scale parameter, then use Normal(loc, scale) and the spec's transform.
        The amortizer does not apply exp to log_scale.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Map from parameter name to array. Keys and semantics depend on
        parameterization (see above).
    parameterization : Literal["constrained", "unconstrained"]
        How to interpret params. Inferred from output_params when created
        by Amortizer.__call__.
    """

    params: Dict[str, jnp.ndarray]
    parameterization: Literal["constrained", "unconstrained"] = "constrained"


# ------------------------------------------------------------------------------


def _amortized_output_flatten(out: AmortizedOutput):
    """Flatten for JAX pytree registration."""
    return (out.params,), (out.parameterization,)


# ------------------------------------------------------------------------------


def _amortized_output_unflatten(aux, children):
    """Unflatten for JAX pytree registration."""
    (params,) = children
    (parameterization,) = aux
    return AmortizedOutput(params=params, parameterization=parameterization)


# ------------------------------------------------------------------------------

jtu.register_pytree_node(
    AmortizedOutput,
    _amortized_output_flatten,
    _amortized_output_unflatten,
)


# ------------------------------------------------------------------------------
# Amortizer Base Class
# ------------------------------------------------------------------------------


class Amortizer(nn.Module):
    """General amortization network for variational inference.

    Maps sufficient statistics to variational parameters using a multi-layer
    perceptron (MLP). The network has separate output heads for each variational
    parameter, allowing different output transformations.

    Architecture
    ------------
    The network has the following structure:

        sufficient_statistic → [Linear → activation] × n_layers → output_heads

    Each output head is a separate linear layer producing one variational
    parameter per data point.

    Parameters
    ----------
    sufficient_statistic : SufficientStatistic
        Defines how to compute input features from count data.
    hidden_dims : List[int]
        Dimensions of hidden layers. E.g., [64, 32] creates a network
        with two hidden layers of size 64 and 32.
    output_params : List[str]
        Names of output variational parameters. Each gets its own
        output head. E.g., ["alpha", "beta"] for Beta parameters.
    output_transforms : Dict[str, Callable], optional
        Optional transforms to apply to each output. Keys should match
        output_params. E.g., {"alpha": jnp.exp} to ensure positivity.
    input_dim : int, optional
        Dimension of the sufficient statistic. Default is 1 (scalar).
    activation : str, optional
        Activation function for hidden layers. Default is "relu".
        Supported: "relu", "gelu", "silu", "swish", "tanh", "sigmoid",
        "elu", "leaky_relu", "softplus", "celu", "selu".

    Attributes
    ----------
    sufficient_statistic : SufficientStatistic
        The statistic computation function.
    hidden_dims : List[int]
        Hidden layer dimensions (stored for serialization).
    output_params : List[str]
        Names of output parameters.
    output_transforms : Dict[str, Callable]
        Transforms for each output.
    input_dim : int
        Input dimension (stored for serialization).
    activation : str
        Activation function name (stored for serialization).

    Examples
    --------
    >>> # Amortize p_capture using total UMI count
    >>> amortizer = Amortizer(
    ...     sufficient_statistic=TOTAL_COUNT,
    ...     hidden_dims=[64, 32],
    ...     output_params=["alpha", "beta"],
    ... )
    >>> # Forward pass (requires initialization via flax_module in NumPyro)
    >>> # In NumPyro: net = flax_module("amortizer", amortizer, input_shape=(1,))
    >>> # Then: out = net(counts); var_params = out.params

    Notes
    -----
    This is a Flax Linen module. The network weights are registered as Flax
    parameters and will be included in the optimization when training with
    NumPyro/SVI via `flax_module`.

    See Also
    --------
    SufficientStatistic : Input feature computation.
    scribe.models.components.guide_families.AmortizedGuide : Uses amortizers.
    numpyro.contrib.module.flax_module : Register Linen modules with NumPyro.
    """

    # Linen modules use class attributes for configuration
    sufficient_statistic: SufficientStatistic
    hidden_dims: List[int]
    output_params: List[str]
    output_transforms: Optional[Dict[str, Callable]] = None
    input_dim: int = 1
    activation: str = "relu"

    def setup(self):
        """Setup method to pre-compute transforms.

        This is called once during module initialization to set up
        non-parameter attributes that are used in the forward pass.
        """
        # Pre-compute transform functions (identity if no transform specified)
        # Use regular function instead of lambda for JIT compatibility
        transforms = self.output_transforms or {}
        self._output_transforms_list = [
            transforms.get(name, _identity_transform)
            for name in self.output_params
        ]

    @nn.compact
    def __call__(self, data: jnp.ndarray) -> AmortizedOutput:
        """Forward pass: compute variational parameters from data.

        Return value satisfies the **Amortizer output contract** (see
        :class:`AmortizedOutput`): constrained (alpha, beta) vs unconstrained
        (loc, log_scale) semantics. Use ``.params`` to get the dict of arrays.

        Parameters
        ----------
        data : jnp.ndarray
            Count data of shape (..., n_genes).

        Returns
        -------
        AmortizedOutput
            Contract object; ``.params`` maps parameter names to predicted
            values. Each value has shape (...) matching the batch dimensions
            of data. Semantics depend on ``.parameterization`` (see
            AmortizedOutput docstring).

        Examples
        --------
        >>> counts = jnp.ones((100, 2000))
        >>> out = amortizer(counts)
        >>> list(out.params.keys())
        ['alpha', 'beta']
        >>> out.params["alpha"].shape
        (100,)
        """
        # ====================================================================
        # Compute sufficient statistic from data
        # Shape: (..., n_genes) → (..., statistic_dim)
        # ====================================================================
        h = self.sufficient_statistic.compute(data)

        # ====================================================================
        # Forward through hidden layers with activation function
        # Linen pattern: define layers inline within @nn.compact
        # ====================================================================
        activation_fn = _get_activation_fn(self.activation)
        in_dim = self.input_dim
        for i, h_dim in enumerate(self.hidden_dims):
            h = nn.Dense(features=h_dim, name=f"hidden_{i}")(h)
            h = activation_fn(h)
            in_dim = h_dim

        # ====================================================================
        # Compute outputs from each head and apply optional transforms
        # Squeeze the last dimension since output heads produce (..., 1)
        # NOTE: Use pre-computed transform list to eliminate dict lookups
        # for maximum JIT compilation performance. Iteration order is fixed
        # based on self.output_params for consistent control flow.
        # ====================================================================
        outputs = {}
        for i, name in enumerate(self.output_params):
            # Create output head with unique name for each parameter.
            # Zero-initialize kernel so initial output is 0 regardless of input,
            # giving stable starting values (e.g., exp(0)=1 for BetaPrime params).
            out = nn.Dense(
                features=1,
                name=f"{name}_head",
                kernel_init=nn.initializers.zeros,
            )(h)
            out = out.squeeze(-1)
            # Apply transform (pre-computed in setup)
            transform_fn = self._output_transforms_list[i]
            outputs[name] = transform_fn(out)

        # Infer parameterization for contract: alpha/beta => constrained
        parameterization: Literal["constrained", "unconstrained"] = (
            "constrained" if "alpha" in self.output_params else "unconstrained"
        )
        return AmortizedOutput(
            params=outputs, parameterization=parameterization
        )


# ==============================================================================
# Future: Component-specific amortization for mixture models
# ==============================================================================

# This is a placeholder for future extensions where we might want to
# amortize component-specific parameters in mixture models.
#
# Example use case: In a mixture model with K cell types, we might want
# to learn cell-type-specific deviations from a base distribution using
# an amortizer that takes cell embeddings as input.
#
# component_amortizer = Amortizer(
#     sufficient_statistic=SufficientStatistic("embedding", lambda x: x),
#     hidden_dims=[128, 64],
#     output_params=["delta_loc", "delta_scale"],  # Additive adjustments
# )
