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
# LOFT (Log Soft Extension) — Andrade 2024, arXiv:2402.16408
# ---------------------------------------------------------------------------


def loft_forward(
    z: jnp.ndarray,
    tau: float = 100.0,
    log_det_dtype=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """LOFT forward pass: compress extreme values logarithmically.

    For ``|z| <= tau`` the transform is identity.  Beyond ``tau`` it
    grows as ``log(|z| - tau + 1) + tau``, bounding the output while
    remaining bijective and differentiable everywhere.

    Parameters
    ----------
    z : jnp.ndarray
        Input tensor, shape ``(..., D)``.
    tau : float
        Threshold below which the transform is identity.
    log_det_dtype : dtype, optional
        If provided, upcast the per-dimension log-det values to this
        dtype before summing.  Use ``jnp.float64`` with high-dimensional
        flows to reduce precision loss in the reduction.

    Returns
    -------
    y : jnp.ndarray
        Compressed output, same shape as ``z``.
    log_det : jnp.ndarray
        Sum of per-dimension log-determinant Jacobian, shape ``(...)``.

    References
    ----------
    Andrade, "Stable Training of Normalizing Flows for High-dimensional
    Variational Inference", 2024. arXiv:2402.16408, Eq. 6-8.
    """
    abs_z = jnp.abs(z)
    excess = jnp.maximum(abs_z - tau, 0.0)
    y = jnp.sign(z) * (jnp.log(excess + 1.0) + jnp.minimum(abs_z, tau))
    log_det_per_dim = -jnp.log(excess + 1.0)
    if log_det_dtype is not None:
        log_det_per_dim = log_det_per_dim.astype(log_det_dtype)
    return y, jnp.sum(log_det_per_dim, axis=-1)


def loft_inverse(
    y: jnp.ndarray,
    tau: float = 100.0,
    log_det_dtype=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """LOFT inverse pass: expand compressed values back.

    Parameters
    ----------
    y : jnp.ndarray
        Compressed tensor from :func:`loft_forward`.
    tau : float
        Same threshold used in the forward pass.
    log_det_dtype : dtype, optional
        If provided, upcast the per-dimension log-det values to this
        dtype before summing (see :func:`loft_forward`).

    Returns
    -------
    z : jnp.ndarray
        Reconstructed tensor, same shape as ``y``.
    log_det : jnp.ndarray
        Sum of per-dimension log-determinant of the *inverse* Jacobian
        (positive sign: ``log|dz/dy|``), shape ``(...)``.
    """
    abs_y = jnp.abs(y)
    excess = jnp.maximum(abs_y - tau, 0.0)
    z = jnp.sign(y) * (jnp.exp(excess) - 1.0 + jnp.minimum(abs_y, tau))
    # log|dz/dy| per dimension = max(|y| - tau, 0)
    log_det_per_dim = excess
    if log_det_dtype is not None:
        log_det_per_dim = log_det_per_dim.astype(log_det_dtype)
    return z, jnp.sum(log_det_per_dim, axis=-1)


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
        zero_init_output=chain.zero_init_output,
        use_layer_norm=chain.use_layer_norm,
        use_residual=chain.use_residual,
        soft_clamp=chain.soft_clamp,
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
        zero_init_output=chain.zero_init_output,
        use_layer_norm=chain.use_layer_norm,
        use_residual=chain.use_residual,
        name=f"layer_{layer_idx}",
    )


def _make_maf(chain, layer_idx: int) -> nn.Module:
    from .autoregressive import MAF

    return MAF(
        features=chain.features,
        hidden_dims=list(chain.hidden_dims),
        activation=chain.activation,
        context_dim=_context_dim_from(chain),
        zero_init_output=chain.zero_init_output,
        use_layer_norm=chain.use_layer_norm,
        use_residual=chain.use_residual,
        name=f"layer_{layer_idx}",
    )


def _make_iaf(chain, layer_idx: int) -> nn.Module:
    from .autoregressive import IAF

    return IAF(
        features=chain.features,
        hidden_dims=list(chain.hidden_dims),
        activation=chain.activation,
        context_dim=_context_dim_from(chain),
        zero_init_output=chain.zero_init_output,
        use_layer_norm=chain.use_layer_norm,
        use_residual=chain.use_residual,
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
    zero_init_output : bool
        Zero-init conditioner output layers so the flow starts as identity.
        Default True.
    use_layer_norm : bool
        Apply ``nn.LayerNorm`` after each hidden Dense in the conditioner.
        Default True.
    use_residual : bool
        Add skip connections between consecutive hidden layers of the
        same width. Default True.
    soft_clamp : bool
        Use smooth asymmetric arctan clamp on affine coupling log-scale
        (Andrade 2024). Only affects ``"affine_coupling"`` flow type.
        Default True.
    use_loft : bool
        Apply LOFT (Log Soft Extension) compression and a trainable
        final affine layer after all coupling layers.  Bounds sample
        magnitudes while preserving identity near zero.  Default True.
    loft_tau : float
        LOFT threshold: values with ``|z| < tau`` pass through
        unchanged; beyond ``tau``, growth is logarithmic. Default 100.
    log_det_f64 : bool
        Accumulate the log-determinant Jacobian in float64 to reduce
        precision loss when summing many small per-layer contributions
        in high-dimensional flows.  Requires ``jax_enable_x64=True``
        to be effective (otherwise JAX silently downcasts to float32).
        Off by default because most consumer GPUs throttle float64;
        recommended for datacenter GPUs (A100, H100, MI250X).
        Default False.

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
    # Zero-init the conditioner output layers so the flow starts as identity.
    zero_init_output: bool = True
    # Add LayerNorm after each hidden Dense in the conditioner MLP.
    use_layer_norm: bool = True
    # Add skip connections between same-width hidden layers.
    use_residual: bool = True
    # Use smooth asymmetric arctan clamp on affine log-scale (Andrade 2024).
    soft_clamp: bool = True
    # Apply LOFT compression + trainable final affine after coupling layers.
    use_loft: bool = True
    # LOFT threshold: identity for |z| < tau, logarithmic beyond.
    loft_tau: float = 100.0
    # Accumulate log-det Jacobian in float64 for high-dimensional precision.
    # Requires jax_enable_x64=True; off by default (consumer GPU friendly).
    log_det_f64: bool = False

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
        # Trainable element-wise affine applied after LOFT compression.
        # Initialized to identity (mu=0, sigma=1) so the full chain still
        # starts as identity when combined with zero-init coupling layers.
        if self.use_loft:
            self.final_mu = self.param(
                "final_mu", nn.initializers.zeros, (self.features,)
            )
            self.final_log_sigma = self.param(
                "final_log_sigma", nn.initializers.zeros, (self.features,)
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
        # dim).  When log_det_f64 is enabled, accumulate in float64 to reduce
        # rounding error in high-dimensional flows (requires jax_enable_x64).
        _ld_dtype = jnp.float64 if self.log_det_f64 else x.dtype
        total_log_det = jnp.zeros(x.shape[:-1], dtype=_ld_dtype)

        # Dtype for LOFT per-dim reduction (None = keep native float32)
        _loft_ld_dtype = jnp.float64 if self.log_det_f64 else None

        if reverse:
            # Inverse: final_affine_inv → LOFT_inv → coupling_layers (reversed)
            if self.use_loft:
                sigma = jnp.exp(self.final_log_sigma)
                x = (x - self.final_mu) / sigma
                _affine_ld = jnp.sum(self.final_log_sigma)
                if self.log_det_f64:
                    _affine_ld = _affine_ld.astype(jnp.float64)
                total_log_det = total_log_det - _affine_ld

                x, ld = loft_inverse(
                    x, tau=self.loft_tau, log_det_dtype=_loft_ld_dtype,
                )
                total_log_det = total_log_det + ld

            for layer in reversed(self.layers):
                x, log_det = layer(x, reverse=True, context=combined_context)
                total_log_det = total_log_det + log_det
        else:
            # Forward: coupling_layers → LOFT → final_affine
            for layer in self.layers:
                x, log_det = layer(x, reverse=False, context=combined_context)
                total_log_det = total_log_det + log_det

            if self.use_loft:
                x, ld = loft_forward(
                    x, tau=self.loft_tau, log_det_dtype=_loft_ld_dtype,
                )
                total_log_det = total_log_det + ld

                sigma = jnp.exp(self.final_log_sigma)
                x = sigma * x + self.final_mu
                _affine_ld = jnp.sum(self.final_log_sigma)
                if self.log_det_f64:
                    _affine_ld = _affine_ld.astype(jnp.float64)
                total_log_det = total_log_det + _affine_ld

        return x, total_log_det
