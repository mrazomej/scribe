"""KL annealing utilities for variational inference with VAE-based models.

The standard ELBO weights the data log-likelihood and the KL between the
guide and the prior equally:

    ELBO = E_q[log p(x, u)] - E_q[log q(x)]
         = (reconstruction term)  -  (KL term)

For VAE-style models with high-dimensional latents and capacity-bound
encoders, optimizing this combined objective from step 0 has two known
failure modes:

1. **Posterior collapse**: the encoder collapses ``q(z|u) -> N(0, I)``
   regardless of input, because the reconstruction term has not yet
   developed an informative gradient and the KL term aggressively
   pushes ``q`` toward the prior.

2. **Aggregate-posterior drift**: the converse — the encoder learns
   to fit reconstruction well but the *aggregate* ``q(z) = mean_c
   q(z|u_c)`` drifts from the prior ``N(0, I)``. Jensen's inequality
   on convex decoder paths (``exp(W·z)`` in PLN) amplifies the drift
   into prediction bias at sampling time.

KL annealing addresses both by introducing a step-dependent weight
``beta(step)`` on the KL term:

    L(theta, phi; step) = E_q[log p(x, u)] - beta(step) * E_q[log q(x) - log p(x)]

When ``beta = 0`` the loss is pure reconstruction (encoder free to fit
data without prior pressure); when ``beta = 1`` it is the standard
ELBO. Linearly ramping ``beta: 0 -> 1`` over the first ``warmup``
steps lets the encoder fit reconstruction first, then gradually pulls
the aggregate posterior toward the prior — the standard β-VAE
warmup schedule (Bowman et al. 2016, Sønderby et al. 2016).

Implementation strategy
-----------------------
Rather than rewriting the SVI loop to multiply the KL term post-hoc,
we subclass NumPyro's :class:`TraceMeanField_ELBO` and intercept the
per-site ELBO contributions before they are summed. Observed sites
keep their full weight; latent and auxiliary sites are scaled by
``beta``. This preserves the analytic-KL fast path when available and
falls back to the Monte-Carlo KL otherwise — both are scaled
identically.

The ``beta`` value is passed as a JAX scalar via ``**kwargs`` to
``loss_with_mutable_state`` so it traces cleanly through JIT without
recompilation per step. The model and guide functions never see
``beta`` (we pop it off ``kwargs`` before forwarding to them) so that
existing model and guide signatures are not affected.

Public API
----------
- :class:`AnnealedTraceMeanField_ELBO` — drop-in replacement for
  :class:`numpyro.infer.TraceMeanField_ELBO` that respects a
  per-step ``beta`` kwarg.
- :func:`linear_beta_schedule` — JIT-traceable linear ramp from 0 to 1.
- :func:`make_beta_schedule` — dispatcher that returns a callable
  ``step -> beta`` for the configured schedule kind.

Examples
--------
>>> from scribe.svi.kl_annealing import (
...     AnnealedTraceMeanField_ELBO, make_beta_schedule)
>>> elbo = AnnealedTraceMeanField_ELBO()
>>> schedule = make_beta_schedule("linear", warmup=2000)
>>> # Inside the SVI loop:
>>> beta = schedule(step)
>>> loss = svi.update(svi_state, beta=beta, **model_args)

References
----------
- Bowman et al. (2016). Generating Sentences from a Continuous Space.
  CoNLL.
- Sønderby et al. (2016). Ladder Variational Autoencoders. NeurIPS.
- Higgins et al. (2017). β-VAE: Learning basic visual concepts with a
  constrained variational framework. ICLR.
"""

from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp
import logging
from numpyro.distributions.kl import kl_divergence
from numpyro.distributions.util import scale_and_mask
from numpyro.handlers import replay, seed, substitute, trace
from numpyro.infer import TraceMeanField_ELBO

# These three helpers live in numpyro.infer.elbo as private functions
# (no public re-export). We import them directly so our subclass can
# replicate the upstream loss-computation body verbatim. The dependency
# is intentionally tight: if numpyro changes the helper layout, this
# subclass must follow.
from numpyro.infer.elbo import _get_log_prob_sum, _check_mean_field_requirement
from numpyro.util import _validate_model, check_model_guide_match
from jax import random

_log = logging.getLogger(__name__)


# =====================================================================
# Schedule helpers
# =====================================================================


def linear_beta_schedule(
    step: int | jnp.ndarray,
    warmup: int,
    *,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
) -> jnp.ndarray:
    """Compute a linear ``beta`` schedule for KL annealing.

    Parameters
    ----------
    step : int or jnp.ndarray
        Current SVI step (0-indexed). May be a Python ``int`` (host-side)
        or a JAX scalar (traced); the function is JIT-traceable in both
        cases.
    warmup : int
        Number of steps over which to ramp ``beta`` linearly from
        ``beta_min`` (at ``step=0``) to ``beta_max`` (at ``step=warmup``).
        After ``warmup`` steps the value is clamped at ``beta_max``.
        When ``warmup <= 0``, this function returns ``beta_max`` immediately
        for any ``step`` (annealing effectively disabled).
    beta_min : float, default=0.0
        Starting value of the schedule (inclusive). ``0.0`` means the
        first step is pure reconstruction; ``0.1`` keeps a faint KL
        signal alive throughout warmup.
    beta_max : float, default=1.0
        Ending value of the schedule (inclusive). ``1.0`` recovers the
        standard ELBO after warmup. Lower values (e.g. ``0.5``)
        implement a permanent β-VAE-style down-weighting.

    Returns
    -------
    jnp.ndarray
        ``float32`` scalar in ``[beta_min, beta_max]``.

    Examples
    --------
    >>> linear_beta_schedule(0, 100)                # 0.0
    >>> linear_beta_schedule(50, 100)               # 0.5
    >>> linear_beta_schedule(100, 100)              # 1.0
    >>> linear_beta_schedule(200, 100)              # 1.0 (clamped)
    >>> linear_beta_schedule(50, 0)                 # 1.0 (annealing off)
    >>> linear_beta_schedule(50, 100, beta_min=0.2)  # 0.6
    """
    # When warmup <= 0 we want beta_max for every step. Use the safe
    # divisor ``max(warmup, 1)`` so the math is JIT-traceable; we then
    # use ``where`` to mask out the result.
    safe_warmup = jnp.maximum(jnp.asarray(warmup, dtype=jnp.float32), 1.0)
    raw = jnp.asarray(step, dtype=jnp.float32) / safe_warmup
    # Clamp to [0, 1], then re-map to [beta_min, beta_max].
    fraction = jnp.clip(raw, 0.0, 1.0)
    annealed = beta_min + fraction * (beta_max - beta_min)
    # When warmup <= 0, short-circuit to beta_max.
    return jnp.where(
        jnp.asarray(warmup, dtype=jnp.int32) <= 0,
        jnp.asarray(beta_max, dtype=jnp.float32),
        annealed.astype(jnp.float32),
    )


def make_beta_schedule(
    kind: Literal["linear"] = "linear",
    *,
    warmup: int,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
) -> Callable[[int | jnp.ndarray], jnp.ndarray]:
    """Build a ``step -> beta`` callable for the configured schedule.

    Returns a closure capturing the schedule parameters so the
    inference loop can call ``schedule(step)`` once per iteration
    without re-passing the warmup configuration.

    Parameters
    ----------
    kind : {"linear"}, default="linear"
        Shape of the schedule. Only ``"linear"`` is implemented in v1
        — the dispatcher exists so adding ``"cosine"``, ``"cyclic"``,
        or other schedules in the future is a one-line change.
    warmup : int
        Number of steps over which to ramp from ``beta_min`` to
        ``beta_max``. Forwarded to the underlying schedule function.
    beta_min, beta_max : float
        Schedule endpoints. See :func:`linear_beta_schedule`.

    Returns
    -------
    Callable[[int | jnp.ndarray], jnp.ndarray]
        A function ``step -> beta_at_step``. JIT-traceable.

    Raises
    ------
    ValueError
        If ``kind`` is not a known schedule name.
    """
    if kind == "linear":
        return lambda step: linear_beta_schedule(
            step, warmup, beta_min=beta_min, beta_max=beta_max
        )
    raise ValueError(
        f"Unknown KL annealing schedule {kind!r}. "
        "Supported schedules: 'linear'."
    )


# =====================================================================
# Annealed ELBO
# =====================================================================


class AnnealedTraceMeanField_ELBO(TraceMeanField_ELBO):
    """Mean-field ELBO with a per-step ``beta`` weight on the KL term.

    A drop-in replacement for :class:`numpyro.infer.TraceMeanField_ELBO`
    that intercepts the per-site contributions before they are summed
    and scales the KL term (latent sample sites + auxiliary sites) by
    a caller-supplied ``beta`` weight. Observed sites — which carry
    the data log-likelihood, the "reconstruction" half of the ELBO —
    are kept at full weight regardless of ``beta``.

    The expected loss form (per particle) is

    .. math::
        L(\\theta, \\phi; \\beta) = -\\sum_{\\text{obs}} \\log p(u | x)
            + \\beta \\sum_{\\text{latent}} \\bigl[ \\log q(x) - \\log p(x) \\bigr]
            + \\beta \\sum_{\\text{aux}} \\log q(\\text{aux})

    so that ``beta = 1`` exactly reproduces the standard ELBO and
    ``beta = 0`` yields a pure-reconstruction loss with no gradient
    on guide-only parameters (e.g. encoder weights).

    Parameters
    ----------
    num_particles : int, default=1
        Number of particles used for the ELBO estimator. Mirrors the
        upstream ``TraceMeanField_ELBO`` semantics.
    vectorize_particles : bool, default=True
        Whether to use ``jax.vmap`` to compute the per-particle ELBOs
        in parallel.

    Notes
    -----
    The ``beta`` value is consumed via ``**kwargs`` in
    :meth:`loss_with_mutable_state`. It is *not* forwarded to the
    model or guide; if those functions accepted a ``beta`` kwarg
    upstream it would silently shadow ours. We therefore pop ``beta``
    out of ``kwargs`` before invoking ``model(*args, **kwargs)`` and
    ``guide(*args, **kwargs)``.

    Mean-field-only restriction: this subclass does not support
    ``sum_sites=False`` (the per-site dict mode), because the scaling
    we apply collapses the per-site contributions into a single scalar.
    Users who need per-site diagnostics should use the upstream
    ``TraceMeanField_ELBO`` directly.

    Examples
    --------
    >>> elbo = AnnealedTraceMeanField_ELBO()
    >>> # In the SVI loop:
    >>> beta_at_step = jnp.asarray(0.3, dtype=jnp.float32)
    >>> svi_state, loss = svi.update(svi_state, beta=beta_at_step, **args)
    """

    def __init__(
        self,
        num_particles: int = 1,
        vectorize_particles: bool = True,
    ) -> None:
        # ``sum_sites=True`` is the only mode this subclass supports —
        # see the class docstring for why. Hard-code it here so users
        # cannot accidentally request the per-site dict mode.
        super().__init__(
            num_particles=num_particles,
            vectorize_particles=vectorize_particles,
            sum_sites=True,
        )

    def loss_with_mutable_state(
        self,
        rng_key,
        param_map,
        model,
        guide,
        *args,
        **kwargs,
    ):
        """Compute the annealed mean-field ELBO with a per-step ``beta`` weight.

        Mirrors the upstream
        :meth:`TraceMeanField_ELBO.loss_with_mutable_state` body but
        scales every non-observed contribution by the value of
        ``kwargs.pop("beta")``. ``beta`` defaults to ``1.0`` (no
        annealing) when the kwarg is not supplied, so this method is
        safe to call from code paths that have not been updated to
        pass it.

        Parameters
        ----------
        rng_key : jax.Array
            PRNG key for the sampling-based ELBO estimator.
        param_map : dict
            Variational parameter values from the SVI optimiser state.
        model, guide : callables
            The model and guide functions.
        *args, **kwargs
            Forwarded to ``model`` and ``guide``. The ``beta`` kwarg
            (if present) is intercepted here and not forwarded.

        Returns
        -------
        dict
            ``{"loss": <scalar>, "mutable_state": <dict-or-None>}`` —
            the same return shape as upstream.
        """
        # Pop the annealing weight off kwargs so neither model nor guide
        # ever sees it. Default to 1.0 = no annealing, so existing
        # callers that don't know about beta get the standard ELBO.
        beta = jnp.asarray(kwargs.pop("beta", 1.0), dtype=jnp.float32)

        def single_particle_elbo(rng_key):
            # The body below is a near-verbatim copy of NumPyro's
            # ``TraceMeanField_ELBO.single_particle_elbo`` (see
            # numpyro/infer/elbo.py:395-457). Differences vs upstream:
            #   1. ``beta`` multiplies every non-observed contribution.
            #   2. ``sum_sites`` is hard-coded to True.
            # All other lines are byte-identical to upstream so the
            # validation and trace-gathering machinery is preserved.
            params = param_map.copy()
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            subs_guide = substitute(seeded_guide, data=param_map)
            guide_trace = trace(subs_guide).get_trace(*args, **kwargs)
            # Mutable state from guide trace (e.g. running mean of
            # batchnorm layers). Carried over to the model trace so
            # downstream sites can read it.
            mutable_params = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "mutable"
            }
            params.update(mutable_params)
            subs_model = substitute(
                replay(seeded_model, guide_trace), data=params
            )
            model_trace = trace(subs_model).get_trace(*args, **kwargs)
            mutable_params.update(
                {
                    name: site["value"]
                    for name, site in model_trace.items()
                    if site["type"] == "mutable"
                }
            )
            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")
            _check_mean_field_requirement(model_trace, guide_trace)

            _elbo_particle = {}
            for name, model_site in model_trace.items():
                if model_site["type"] == "sample":
                    if model_site["is_observed"]:
                        # Observed (data) site: full weight regardless of
                        # beta. This is the reconstruction half of the
                        # ELBO — annealing only touches the KL half.
                        _elbo_particle[name] = _get_log_prob_sum(model_site)
                    else:
                        # Latent site: closed-form analytic KL when
                        # available, otherwise Monte-Carlo. Both are
                        # contributions to the KL term, both scaled by
                        # beta.
                        guide_site = guide_trace[name]
                        try:
                            kl_qp = kl_divergence(
                                guide_site["fn"], model_site["fn"]
                            )
                            kl_qp = scale_and_mask(
                                kl_qp, scale=guide_site["scale"]
                            )
                            # ``-jnp.sum(kl_qp)`` is the analytic KL
                            # contribution to the ELBO; multiply by
                            # ``beta`` to anneal.
                            _elbo_particle[name] = -beta * jnp.sum(kl_qp)
                        except NotImplementedError:
                            # MC-KL fallback. Both the model and guide
                            # log-prob terms together represent
                            # ``-KL_estimator``; we scale the *whole*
                            # estimator by ``beta``.
                            _elbo_particle[name] = beta * (
                                _get_log_prob_sum(model_site)
                                - _get_log_prob_sum(guide_site)
                            )

            # Auxiliary guide-only sites (e.g. ``numpyro.sample(...,
            # infer={"is_auxiliary": True})``) contribute ``-log
            # q(aux)`` to the ELBO; treat them as part of the KL term
            # for annealing purposes.
            for name, site in guide_trace.items():
                if site["type"] == "sample" and name not in model_trace:
                    assert (
                        site["infer"].get("is_auxiliary")
                        or site["is_observed"]
                    )
                    _elbo_particle[name] = -beta * _get_log_prob_sum(site)

            elbo_particle = sum(
                _elbo_particle.values(), start=jnp.array(0.0)
            )

            if mutable_params:
                if self.num_particles == 1:
                    return elbo_particle, mutable_params
                _log.warning(
                    "mutable state is currently ignored when "
                    "num_particles > 1."
                )
            return elbo_particle, None

        if self.num_particles == 1:
            elbo, mutable_state = single_particle_elbo(rng_key)
            return {
                "loss": jax.tree.map(jnp.negative, elbo),
                "mutable_state": mutable_state,
            }
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            elbos, mutable_state = self.vectorize_particles_fn(
                single_particle_elbo, rng_keys
            )
            return {
                "loss": jax.tree.map(lambda x: -jnp.mean(x), elbos),
                "mutable_state": mutable_state,
            }


__all__ = [
    "AnnealedTraceMeanField_ELBO",
    "linear_beta_schedule",
    "make_beta_schedule",
]
