"""
Normalizing flows for SCRIBE.

A Flax Linen-based normalizing flow library providing multiple
flow architectures for use as learned priors and posterior flows
in variational autoencoders.

Flow Types
----------
AffineCoupling
    Real NVP affine coupling (fast, simple baseline).
SplineCoupling
    Neural spline flow with rational-quadratic splines (more expressive).
MAF
    Masked Autoregressive Flow (fast density, slow sampling — good for priors).
IAF
    Inverse Autoregressive Flow (fast sampling, slow density — good for posteriors).
FlowChain
    Sequential composition of flow layers.
FlowDistribution
    NumPyro distribution wrapper for using flows in probabilistic models.

Examples
--------
>>> from scribe.flows import FlowChain, FlowDistribution
>>> import jax, jax.numpy as jnp
>>> import numpyro.distributions as dist
>>>
>>> # Create a 4-layer spline coupling flow
>>> chain = FlowChain(features=10, num_layers=4,
...                   flow_type="spline_coupling",
...                   hidden_dims=[64, 64])
>>> params = chain.init(jax.random.PRNGKey(0), jnp.zeros(10))
>>> z, log_det = chain.apply(params, jnp.ones(10))
"""

# Composite flow (stack of layers).
from .base import FlowChain

# Coupling layers: affine (Real NVP) and spline (NSF).
from .coupling import AffineCoupling, SplineCoupling

# Autoregressive layers and MADE conditioner.
from .autoregressive import MAF, IAF, MADE

# NumPyro distribution wrappers for use in probabilistic models.
from .distributions import (
    ComponentFlowDistribution,
    FlowDistribution,
    SlicedTransform,
)

# RQS spline primitives (used by SplineCoupling).
from .transforms import rqs_forward, rqs_inverse, unconstrained_to_rqs_params

# Public API; used by "from scribe.flows import ...".
__all__ = [
    "FlowChain",
    # Coupling flows
    "AffineCoupling",
    "SplineCoupling",
    # Autoregressive flows
    "MAF",
    "IAF",
    "MADE",
    # Distribution wrappers
    "FlowDistribution",
    "ComponentFlowDistribution",
    "SlicedTransform",
    # Spline primitives
    "rqs_forward",
    "rqs_inverse",
    "unconstrained_to_rqs_params",
]


# Register the SlicedTransform handler for jacobian_corrected_map.
# SlicedTransform is defined here in scribe.flows.distributions, so it
# cannot be imported by scribe.stats.jacobian_map without creating a
# circular dependency (scribe.flows -> scribe.models -> scribe.flows).
# Instead, scribe.stats.jacobian_map exposes
# _register_sliced_transform_handler, which we call here once SlicedTransform
# is in scope. The call is idempotent and only runs when scribe.flows is
# actually imported (so scribe.stats remains usable without scribe.flows
# loaded, e.g., in environments that skip the model builders).
from scribe.stats.jacobian_map import _register_sliced_transform_handler

_register_sliced_transform_handler()
