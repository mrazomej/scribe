"""
NumPyro distribution wrappers for normalizing flows.

Wraps any flow (or chain of flows) as a proper
``numpyro.distributions.Distribution`` so it can be used directly in
``numpyro.sample`` statements. This is the key integration point between the
flows module and NumPyro's probabilistic programming framework.

Classes
-------
FlowDistribution
    Wraps a flow callable as a NumPyro distribution.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.util import validate_sample


class FlowDistribution(dist.Distribution):
    """NumPyro distribution defined by a normalizing flow.

    Applies a learned bijective transformation to a simple base
    distribution (typically standard normal) to produce a more
    flexible distribution. Supports both sampling and density
    evaluation via the change-of-variables formula.

    The flow callable should accept ``(x, reverse=bool)`` and return
    ``(y, log_det)``. In practice, this is the output of
    ``numpyro.contrib.module.flax_module`` applied to a ``FlowChain``.

    Parameters
    ----------
    flow : callable
        A callable ``(x, reverse=bool) -> (y, log_det)``.
        Typically produced by ``numpyro.contrib.module.flax_module``.
    base_distribution : numpyro.distributions.Distribution
        The base distribution (e.g., ``Normal(0, 1)``).
    validate_args : bool, optional
        Whether to validate arguments.

    Examples
    --------
    In a NumPyro model::

        flow_fn = flax_module("prior_flow", FlowChain(...),
                              input_shape=(latent_dim,))
        base = dist.Normal(jnp.zeros(latent_dim), 1.0).to_event(1)
        prior = FlowDistribution(flow_fn, base)
        z = numpyro.sample("z", prior)
    """

    def __init__(
        self,
        flow,
        base_distribution: dist.Distribution,
        validate_args=None,
    ):
        self.flow = flow
        self.base_distribution = base_distribution

        batch_shape = base_distribution.batch_shape
        event_shape = base_distribution.event_shape

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    # --------------------------------------------------------------------------

    @property
    def support(self):
        return self.base_distribution.support

    # --------------------------------------------------------------------------

    @validate_sample
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Compute log-probability using the change-of-variables formula.

        ``log p(x) = log p_base(z) + log|det(dz/dx)|``

        where ``z = flow.forward(x)`` and ``log|det(dz/dx)|`` is the
        forward log-determinant.

        Parameters
        ----------
        value : jnp.ndarray
            Point at which to evaluate the density.

        Returns
        -------
        jnp.ndarray
            Log-probability.
        """
        z, log_det = self.flow(value, reverse=False)
        return self.base_distribution.log_prob(z) + log_det

    # --------------------------------------------------------------------------

    def sample(self, key, sample_shape=()):
        """Sample from the flow distribution.

        Draws from the base distribution and applies the inverse
        (generative) direction of the flow.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        sample_shape : tuple of int
            Shape prefix for the samples.

        Returns
        -------
        jnp.ndarray
            Samples from the flow distribution.
        """
        z = self.base_distribution.sample(key, sample_shape)
        x, _ = self.flow(z, reverse=True)
        return x

    # --------------------------------------------------------------------------

    def sample_with_intermediates(self, key, sample_shape=()):
        """Sample and return intermediate values for debugging.

        Returns
        -------
        x : jnp.ndarray
            Samples from the flow distribution.
        intermediates : dict
            Dictionary with ``z_base``, ``x``, ``log_det``.
        """
        z = self.base_distribution.sample(key, sample_shape)
        x, log_det = self.flow(z, reverse=True)
        return x, {"z_base": z, "x": x, "log_det": log_det}
