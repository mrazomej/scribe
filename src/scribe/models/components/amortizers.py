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
...     output_params=["log_alpha", "log_beta"],
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
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
from flax import nnx

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
# Amortizer Base Class
# ------------------------------------------------------------------------------


class Amortizer(nnx.Module):
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
        output head. E.g., ["log_alpha", "log_beta"] for Beta parameters.
    output_transforms : Dict[str, Callable], optional
        Optional transforms to apply to each output. Keys should match
        output_params. E.g., {"alpha": jnp.exp} to ensure positivity.
    input_dim : int, optional
        Dimension of the sufficient statistic. Default is 1 (scalar).
    activation : str, optional
        Activation function for hidden layers. Default is "relu".
        Supported: "relu", "gelu", "silu", "swish", "tanh", "sigmoid",
        "elu", "leaky_relu", "softplus", "celu", "selu".
    rngs : nnx.Rngs, optional
        Random number generators for initialization. If None, creates
        a default RNG.

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
    activation_fn : Callable
        The actual activation function to use in forward pass.
    layers : List[nnx.Linear]
        Hidden layers of the MLP.
    output_heads : Dict[str, nnx.Linear]
        Output heads for each variational parameter.

    Examples
    --------
    >>> # Amortize p_capture using total UMI count
    >>> amortizer = Amortizer(
    ...     sufficient_statistic=TOTAL_COUNT,
    ...     hidden_dims=[64, 32],
    ...     output_params=["log_alpha", "log_beta"],
    ... )
    >>> # Forward pass
    >>> counts = jnp.ones((100, 2000))  # 100 cells, 2000 genes
    >>> var_params = amortizer(counts)
    >>> var_params["log_alpha"].shape
    (100,)

    Notes
    -----
    The network weights are registered as Flax NNX parameters and will be
    included in the optimization when training with NumPyro/SVI.

    See Also
    --------
    SufficientStatistic : Input feature computation.
    scribe.models.components.guide_families.AmortizedGuide : Uses amortizers.
    """

    def __init__(
        self,
        sufficient_statistic: SufficientStatistic,
        hidden_dims: List[int],
        output_params: List[str],
        output_transforms: Optional[Dict[str, Callable]] = None,
        input_dim: int = 1,
        activation: str = "relu",
        rngs: Optional[nnx.Rngs] = None,
    ):
        """Initialize the amortizer network.

        Parameters
        ----------
        sufficient_statistic : SufficientStatistic
            Defines input feature computation.
        hidden_dims : List[int]
            Hidden layer dimensions.
        output_params : List[str]
            Names of output variational parameters.
        output_transforms : Dict[str, Callable], optional
            Output transforms per parameter.
        input_dim : int, optional
            Input dimension (default 1 for scalar statistics).
        activation : str, optional
            Activation function for hidden layers (default "relu").
            Supported: "relu", "gelu", "silu", "swish", "tanh", "sigmoid",
            "elu", "leaky_relu", "softplus", "celu", "selu".
        rngs : nnx.Rngs, optional
            Random number generators.
        """
        # Use default RNG if none provided
        if rngs is None:
            rngs = nnx.Rngs(0)

        # ====================================================================
        # Store configuration for serialization/reconstruction
        # ====================================================================
        self.sufficient_statistic = sufficient_statistic
        self.hidden_dims = hidden_dims
        self.output_params = output_params
        self.output_transforms = output_transforms or {}
        self.input_dim = input_dim
        self.activation = activation

        # Get the actual activation function
        self.activation_fn = _get_activation_fn(activation)

        # ====================================================================
        # Build MLP hidden layers
        # Architecture: input → [Linear → activation] × n → output_heads
        # ====================================================================
        layers_list = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers_list.append(nnx.Linear(in_dim, h_dim, rngs=rngs))
            in_dim = h_dim
        self.layers = nnx.List(layers_list)

        # ====================================================================
        # Build output heads (one per variational parameter)
        # Each head produces a scalar per data point
        # ====================================================================
        output_heads_dict = {
            name: nnx.Linear(in_dim, 1, rngs=rngs) for name in output_params
        }
        self.output_heads = nnx.Dict(output_heads_dict)

        # ====================================================================
        # Pre-compute output processing pipeline for JIT optimization
        # Store heads and transforms in parallel lists to eliminate dict
        # lookups and conditionals in the forward pass
        # ====================================================================
        self._output_heads_list = nnx.List([
            self.output_heads[name] for name in output_params
        ])
        # Pre-compute transform functions (identity if no transform specified)
        # Use regular function instead of lambda for JIT compatibility
        self._output_transforms_list = [
            self.output_transforms.get(name, _identity_transform)  # Identity if no transform
            for name in output_params
        ]

    # --------------------------------------------------------------------------

    def __call__(self, data: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Forward pass: compute variational parameters from data.

        Parameters
        ----------
        data : jnp.ndarray
            Count data of shape (..., n_genes).

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their predicted values.
            Each value has shape (...) matching the batch dimensions of data.

        Examples
        --------
        >>> counts = jnp.ones((100, 2000))
        >>> var_params = amortizer(counts)
        >>> var_params.keys()
        dict_keys(['log_alpha', 'log_beta'])
        >>> var_params["log_alpha"].shape
        (100,)
        """
        # ====================================================================
        # Compute sufficient statistic from data
        # Shape: (..., n_genes) → (..., statistic_dim)
        # ====================================================================
        h = self.sufficient_statistic.compute(data)

        # ====================================================================
        # Forward through hidden layers with activation function
        # ====================================================================
        for layer in self.layers:
            h = self.activation_fn(layer(h))

        # ====================================================================
        # Compute outputs from each head and apply optional transforms
        # Squeeze the last dimension since output heads produce (..., 1)
        # NOTE: Use pre-computed lists to eliminate dict lookups and conditionals
        # for maximum JIT compilation performance. Iteration order is fixed
        # based on self.output_params for consistent control flow.
        # ====================================================================
        outputs = {}
        for i, name in enumerate(self.output_params):
            head = self._output_heads_list[i]
            transform_fn = self._output_transforms_list[i]
            out = transform_fn(head(h).squeeze(-1))
            outputs[name] = out

        return outputs


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
