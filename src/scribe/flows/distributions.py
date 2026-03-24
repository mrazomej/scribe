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
ComponentFlowDistribution
    Stacks K independent ``FlowDistribution`` instances along a leading
    batch axis (e.g. mixture components or datasets).
"""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
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

    Convenience methods for point estimation::

        mean = prior.estimate_mean(rng_key, n_samples=1000)
        mode = prior.find_mode(rng_key, n_steps=300)
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

    # --------------------------------------------------------------------------

    def estimate_mean(
        self,
        key: jax.Array,
        n_samples: int = 1000,
    ) -> jnp.ndarray:
        """Monte Carlo estimate of the distribution mean.

        Draws ``n_samples`` independent samples and averages them.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        n_samples : int, default=1000
            Number of samples for the Monte Carlo estimate.

        Returns
        -------
        jnp.ndarray
            Estimated mean with the same shape as ``event_shape``.
        """
        keys = random.split(key, n_samples)
        samples = jax.vmap(lambda k: self.sample(k))(keys)
        return jnp.mean(samples, axis=0)

    # --------------------------------------------------------------------------

    def find_mode(
        self,
        key: jax.Array,
        n_init_samples: int = 100,
        n_steps: int = 300,
        lr: float = 1e-3,
    ) -> jnp.ndarray:
        """Approximate mode via gradient ascent on ``log_prob``.

        Initializes from the best of ``n_init_samples`` candidates,
        then runs Adam for ``n_steps`` gradient-ascent steps on the
        flow density.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        n_init_samples : int, default=100
            Number of initial candidates to seed the optimization.
        n_steps : int, default=300
            Number of Adam optimization steps.
        lr : float, default=1e-3
            Learning rate for Adam.

        Returns
        -------
        jnp.ndarray
            Approximate mode with the same shape as ``event_shape``.
        """
        try:
            import optax
        except ImportError:
            # Graceful fallback: return Monte Carlo mean instead
            return self.estimate_mean(key, n_samples=n_init_samples)

        # Draw initial candidates and pick the one with highest density
        keys = random.split(key, n_init_samples + 1)
        init_key, opt_key = keys[0], keys[1:]
        candidates = jax.vmap(lambda k: self.sample(k))(opt_key)
        log_probs = jax.vmap(self.log_prob)(candidates)
        best_idx = jnp.argmax(log_probs)
        x = candidates[best_idx]

        # Gradient ascent on log_prob
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(x)

        def step_fn(carry, _):
            x_cur, state = carry
            grad = jax.grad(self.log_prob)(x_cur)
            # Ascent: negate grad for optax (which minimizes)
            updates, new_state = optimizer.update(-grad, state, x_cur)
            # apply_updates adds updates, so with negated grad this
            # moves in the ascent direction
            x_new = optax.apply_updates(x_cur, updates)
            return (x_new, new_state), None

        (x_opt, _), _ = jax.lax.scan(step_fn, (x, opt_state), None, n_steps)
        return x_opt


# ==============================================================================


class ComponentFlowDistribution(dist.Distribution):
    """K independent flow distributions stacked along a leading batch axis.

    Wraps *K* ``FlowDistribution`` (or nested ``ComponentFlowDistribution``)
    instances into a single NumPyro distribution whose ``event_shape`` gains
    an extra leading *K* dimension.  This is the core primitive for
    mixture-aware and dataset-aware normalizing-flow guides: each mixture
    component (or dataset) gets its own flow transformation while the
    combined object remains a single ``numpyro.sample`` site.

    For the **shared** strategy each wrapped distribution is a closure that
    already binds the correct one-hot covariate, so
    ``get_component(k)`` always returns a distribution ready to use.

    Nesting is supported: for a parameter with shape
    ``(n_components, n_datasets, n_genes)`` the outer
    ``ComponentFlowDistribution`` (axis ``"component"``) wraps K inner
    ``ComponentFlowDistribution`` objects (axis ``"dataset"``), each of
    which wraps D ``FlowDistribution`` instances.

    Parameters
    ----------
    component_dists : list of Distribution
        One distribution per index along this axis.  Each should share the
        same ``batch_shape`` and ``event_shape``.
    axis_name : str, default ``"component"``
        Descriptive label (``"component"`` or ``"dataset"``).
    validate_args : bool, optional
        Whether to validate distribution arguments.

    Examples
    --------
    Build three independent flows for a 3-component mixture::

        dists = [FlowDistribution(fn_k, base) for fn_k in per_comp_fns]
        comp = ComponentFlowDistribution(dists, axis_name="component")
        sample = comp.sample(key)          # shape (3, G)
        lp     = comp.log_prob(sample)     # scalar
        d1     = comp.get_component(1)     # FlowDistribution for comp 1

    Nested (components × datasets)::

        inner = [ComponentFlowDistribution(ds, "dataset") for ds in per_comp]
        outer = ComponentFlowDistribution(inner, "component")
        outer.get_component(0).get_component(2)  # comp 0, dataset 2
    """

    def __init__(
        self,
        component_dists: List[dist.Distribution],
        axis_name: str = "component",
        validate_args=None,
    ):
        if not component_dists:
            raise ValueError("component_dists must be non-empty")

        self.component_dists = list(component_dists)
        self.axis_name = axis_name

        inner_event = component_dists[0].event_shape
        K = len(component_dists)
        event_shape = (K, *inner_event)
        batch_shape = component_dists[0].batch_shape

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of indices along this axis."""
        return len(self.component_dists)

    @property
    def support(self):
        return self.component_dists[0].support

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def get_component(self, index: int) -> dist.Distribution:
        """Return the distribution for a specific index.

        For the *shared* strategy the returned distribution already has
        its one-hot covariate bound, so callers need not manage context.

        Parameters
        ----------
        index : int
            Position along this axis (0-based).

        Returns
        -------
        Distribution
            ``FlowDistribution`` (leaf) or nested
            ``ComponentFlowDistribution``.
        """
        return self.component_dists[index]

    # ------------------------------------------------------------------
    # Core distribution interface
    # ------------------------------------------------------------------

    def sample(self, key, sample_shape=()):
        """
        Draw a sample of shape ``(*sample_shape, *batch_shape, K, *inner_event)``.

        Each component is sampled independently with its own PRNG split.
        """
        keys = random.split(key, self.n_components)
        inner_event_ndims = len(self.component_dists[0].event_shape)
        # Stack along the axis just before inner_event dims
        samples = [
            d.sample(k, sample_shape)
            for k, d in zip(keys, self.component_dists)
        ]
        return jnp.stack(samples, axis=-(inner_event_ndims + 1))

    @validate_sample
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Sum of per-component log-probabilities.

        Parameters
        ----------
        value : jnp.ndarray
            Tensor whose shape ends with ``(K, *inner_event)``.
        """
        inner_ndims = len(self.component_dists[0].event_shape)
        # Axis position of the K dimension in *value*
        k_axis = -(inner_ndims + 1)
        lps = [
            d.log_prob(jnp.take(value, k, axis=k_axis))
            for k, d in enumerate(self.component_dists)
        ]
        return sum(lps)

    def sample_with_intermediates(self, key, sample_shape=()):
        """Sample and collect per-component intermediates."""
        keys = random.split(key, self.n_components)
        inner_event_ndims = len(self.component_dists[0].event_shape)
        samples, all_inter = [], []
        for k, d in zip(keys, self.component_dists):
            if hasattr(d, "sample_with_intermediates"):
                s, inter = d.sample_with_intermediates(k, sample_shape)
            else:
                s = d.sample(k, sample_shape)
                inter = {}
            samples.append(s)
            all_inter.append(inter)
        stacked = jnp.stack(samples, axis=-(inner_event_ndims + 1))
        return stacked, {"per_component": all_inter}

    # ------------------------------------------------------------------
    # Point-estimation helpers
    # ------------------------------------------------------------------

    def estimate_mean(
        self,
        key: jax.Array,
        n_samples: int = 1000,
    ) -> jnp.ndarray:
        """Monte Carlo mean — delegates to each component independently.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        n_samples : int
            Samples per component.

        Returns
        -------
        jnp.ndarray
            Shape ``(K, *inner_event)``.
        """
        keys = random.split(key, self.n_components)
        inner_event_ndims = len(self.component_dists[0].event_shape)
        means = [
            (
                d.estimate_mean(k, n_samples)
                if hasattr(d, "estimate_mean")
                else _mc_mean(d, k, n_samples)
            )
            for k, d in zip(keys, self.component_dists)
        ]
        return jnp.stack(means, axis=-(inner_event_ndims + 1))

    def find_mode(
        self,
        key: jax.Array,
        n_init_samples: int = 100,
        n_steps: int = 300,
        lr: float = 1e-3,
    ) -> jnp.ndarray:
        """Per-component mode finding via gradient ascent.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        n_init_samples : int
            Initial candidates per component.
        n_steps : int
            Adam steps per component.
        lr : float
            Learning rate.

        Returns
        -------
        jnp.ndarray
            Shape ``(K, *inner_event)``.
        """
        keys = random.split(key, self.n_components)
        inner_event_ndims = len(self.component_dists[0].event_shape)
        modes = [
            (
                d.find_mode(k, n_init_samples, n_steps, lr)
                if hasattr(d, "find_mode")
                else _mc_mean(d, k, n_init_samples)
            )
            for k, d in zip(keys, self.component_dists)
        ]
        return jnp.stack(modes, axis=-(inner_event_ndims + 1))


# ==============================================================================
# Helpers
# ==============================================================================


def _mc_mean(d: dist.Distribution, key: jax.Array, n: int) -> jnp.ndarray:
    """Fallback Monte Carlo mean for distributions without ``estimate_mean``."""
    keys = random.split(key, n)
    samples = jax.vmap(lambda k: d.sample(k))(keys)
    return jnp.mean(samples, axis=0)
