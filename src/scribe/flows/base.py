"""
Base classes for normalizing flows.

This module defines the abstract Flow interface and the FlowChain
compositor that stacks multiple flows sequentially.

Convention
----------
All flows use the following direction convention:

- ``forward(x)``: Transforms from data space to latent space.
  Returns ``(z, log_det)`` where ``log_det = log|det(dz/dx)|``.
- ``inverse(z)``: Transforms from latent space to data space.
  Returns ``(x, log_det)`` where ``log_det = log|det(dx/dz)| = -log|det(dz/dx)|``.

For use as a learned prior (FlowDistribution):

- ``sample``: Draw z ~ base, apply ``inverse`` to get x in data space.
- ``log_prob(x)``: Apply ``forward`` to get z, compute
  ``log p_base(z) + log_det_forward``.

Classes
-------
FlowChain
    Sequential composition of multiple flow layers.
"""

# Type hints for layer config and return types.
from typing import Callable, Dict, List, Optional, Tuple

# JAX arrays and Flax Linen for module definition.
import jax.numpy as jnp
from flax import linen as nn

from scribe.models.components.covariate_embedding import (
    CovariateEmbedding,
    CovariateSpec,
)

# ---------------------------------------------------------------------------
# Flow layer registry: flow_type -> factory(chain, layer_idx) -> nn.Module
# ---------------------------------------------------------------------------


def _context_dim_from(chain) -> int:
    """
    Compute total context dimensionality from covariate specs and explicit
    context_dim.
    """
    dim = 0
    if chain.covariate_specs:
        dim += sum(s.embedding_dim for s in chain.covariate_specs)
    if chain.context_dim:
        dim += chain.context_dim
    return dim


def _make_affine_coupling(chain, layer_idx: int) -> nn.Module:
    from .coupling import AffineCoupling

    return AffineCoupling(
        features=chain.features,
        hidden_dims=list(chain.hidden_dims),
        mask_parity=layer_idx % 2,
        activation=chain.activation,
        context_dim=_context_dim_from(chain),
        name=f"layer_{layer_idx}",
    )


def _make_spline_coupling(chain, layer_idx: int) -> nn.Module:
    from .coupling import SplineCoupling

    return SplineCoupling(
        features=chain.features,
        hidden_dims=list(chain.hidden_dims),
        mask_parity=layer_idx % 2,
        activation=chain.activation,
        context_dim=_context_dim_from(chain),
        n_bins=chain.n_bins,
        name=f"layer_{layer_idx}",
    )


def _make_maf(chain, layer_idx: int) -> nn.Module:
    from .autoregressive import MAF

    return MAF(
        features=chain.features,
        hidden_dims=list(chain.hidden_dims),
        activation=chain.activation,
        context_dim=_context_dim_from(chain),
        name=f"layer_{layer_idx}",
    )


def _make_iaf(chain, layer_idx: int) -> nn.Module:
    from .autoregressive import IAF

    return IAF(
        features=chain.features,
        hidden_dims=list(chain.hidden_dims),
        activation=chain.activation,
        context_dim=_context_dim_from(chain),
        name=f"layer_{layer_idx}",
    )


FlowLayerFactory = Callable[[nn.Module, int], nn.Module]

FLOW_REGISTRY: dict[str, FlowLayerFactory] = {
    "affine_coupling": _make_affine_coupling,
    "spline_coupling": _make_spline_coupling,
    "maf": _make_maf,
    "iaf": _make_iaf,
}


# ---------------------------------------------------------------------------
# FlowChain
# ---------------------------------------------------------------------------


class FlowChain(nn.Module):
    """Sequential composition of normalizing flow layers.

    Applies a stack of flow layers in sequence (forward) or reverse
    (inverse), accumulating log-determinant Jacobians across layers.
    Alternating mask parities are applied automatically for coupling flows.

    Parameters
    ----------
    features : int
        Dimensionality of the input/output space.
    num_layers : int
        Number of flow layers to stack.
    flow_type : str
        Type of flow layer. One of ``"affine_coupling"``,
        ``"spline_coupling"``, ``"maf"``, ``"iaf"``.
    hidden_dims : List[int]
        Hidden dimensions for the conditioner networks in each layer.
    activation : str
        Activation function for conditioner networks.
    n_bins : int
        Number of bins for spline flows (ignored for affine flows).
    covariate_specs : list of CovariateSpec, optional
        Categorical covariates to embed and pass as context to each layer.

    Examples
    --------
    >>> chain = FlowChain(features=10, num_layers=4, flow_type="affine_coupling",
    ...                   hidden_dims=[64, 64], activation="relu")
    >>> params = chain.init(jax.random.PRNGKey(0), jnp.zeros(10))
    >>> z, log_det = chain.apply(params, x)
    """

    # Input/output dimensionality (same for data and latent).
    features: int
    # Number of flow layers to stack in sequence.
    num_layers: int
    # Layer type: affine_coupling, spline_coupling, maf, or iaf.
    flow_type: str = "affine_coupling"
    # Hidden layer sizes for the conditioner networks.
    hidden_dims: List[int] = (64, 64)
    # Activation used in conditioner nets (e.g. relu, gelu).
    activation: str = "relu"
    # Number of bins for spline coupling (only used when flow_type is
    # spline_coupling).
    n_bins: int = 8
    # Optional covariate specs for conditioning; when provided, a shared
    # CovariateEmbedding is created and its output is passed as context
    # to each flow layer.
    covariate_specs: Optional[List[CovariateSpec]] = None
    # Optional dimensionality for continuous context vectors passed
    # directly to the flow layers (e.g., for conditioning on previously
    # sampled parameters in a joint guide).  Unlike ``covariate_specs``
    # (categorical → embedding), this expects a pre-formed float vector.
    context_dim: int = 0

    def setup(self):
        """Create the stack of flow layers with alternating masks."""
        factory = FLOW_REGISTRY.get(self.flow_type)
        if factory is None:
            raise ValueError(
                f"Unknown flow_type '{self.flow_type}'. "
                f"Choose from: {list(FLOW_REGISTRY)}"
            )
        self.layers = [factory(self, i) for i in range(self.num_layers)]
        if self.covariate_specs:
            self.cov_embed = CovariateEmbedding(
                covariate_specs=self.covariate_specs,
            )

    # --------------------------------------------------------------------------

    def __call__(
        self,
        x: jnp.ndarray,
        reverse: bool = False,
        covariates: Optional[Dict[str, jnp.ndarray]] = None,
        context: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the flow chain.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape ``(..., features)``.
        reverse : bool
            If True, apply layers in reverse order with ``reverse=True``.
        covariates : dict, optional
            Maps covariate name → integer ID array, shape ``(...)``.
            Only used when ``covariate_specs`` was provided.
        context : jnp.ndarray, optional
            Pre-formed continuous context vector, shape ``(..., context_dim)``.
            Passed directly to each flow layer without embedding.  Used for
            conditioning on previously sampled parameters in joint guides.
            If both ``covariates`` and ``context`` are provided, the embedded
            covariates and the continuous context are concatenated.

        Returns
        -------
        y : jnp.ndarray
            Transformed tensor, same shape as input.
        log_det : jnp.ndarray
            Total log-determinant Jacobian, shape ``(...)``.
        """
        # Build the combined context vector for all layers.
        # Categorical covariates are embedded; continuous context is used as-is.
        parts = []
        if covariates is not None and self.covariate_specs:
            parts.append(self.cov_embed(covariates))
        if context is not None:
            parts.append(context)
        combined_context = jnp.concatenate(parts, axis=-1) if parts else None

        # Start with zero log-det; same batch/event shape as input (no feature
        # dim).
        total_log_det = jnp.zeros(x.shape[:-1])
        # Inverse direction: apply layers in reverse order for correct
        # composition.
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers:
            # Each layer returns updated x and its log|det(J)|; chain rule sums
            # log-dets.
            x, log_det = layer(x, reverse=reverse, context=combined_context)
            total_log_det = total_log_det + log_det
        # Final transformed tensor and total log-determinant (for density
        # computation).
        return x, total_log_det
