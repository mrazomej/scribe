"""Regression tests for NumPyro>=0.20 mixture compatibility.

These tests validate that SCRIBE likelihood builders construct mixture
distributions without triggering the ``MixtureSameFamily`` support
validation introduced in NumPyro 0.20.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from scribe.models.components.likelihoods.negative_binomial import (
    NegativeBinomialLikelihood,
)
from scribe.models.components.likelihoods.zero_inflated import (
    ZeroInflatedNBLikelihood,
)


def test_negative_binomial_mixture_builds_with_numpyro_0201_rules() -> None:
    """NB mixture path should build a valid distribution under NumPyro 0.20+.

    Returns
    -------
    None
        Asserts the constructed distribution is sampleable and finite.
    """
    likelihood = NegativeBinomialLikelihood()
    mixture_dist = likelihood._build_dist(
        {
            "mixing_weights": jnp.array([0.6, 0.4]),
            "r": jnp.full((2, 3), 5.0),
            "p": jnp.full((2, 3), 0.35),
        }
    )

    # NumPyro 0.20 compatibility relies on MixtureGeneral construction.
    assert isinstance(mixture_dist, dist.MixtureGeneral)
    sample = mixture_dist.sample(jax.random.PRNGKey(0))
    assert sample.shape == (3,)
    assert jnp.all(jnp.isfinite(sample))


def test_zero_inflated_mixture_builds_with_numpyro_0201_rules() -> None:
    """ZINB mixture path should build a valid distribution under NumPyro 0.20+.

    Returns
    -------
    None
        Asserts the constructed distribution is sampleable and finite.
    """
    likelihood = ZeroInflatedNBLikelihood()
    mixture_dist = likelihood._build_dist(
        {
            "mixing_weights": jnp.array([0.7, 0.3]),
            "r": jnp.full((2, 4), 6.0),
            "p": jnp.full((2, 4), 0.4),
            "gate": jnp.full((2, 4), 0.1),
        }
    )

    # NumPyro 0.20 compatibility relies on MixtureGeneral construction.
    assert isinstance(mixture_dist, dist.MixtureGeneral)
    sample = mixture_dist.sample(jax.random.PRNGKey(1))
    assert sample.shape == (4,)
    assert jnp.all(jnp.isfinite(sample))
