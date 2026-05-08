"""Tests for LNM composition sampling helpers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from scribe.de._empirical import _batched_lnm_sample


def test_batched_lnm_sample_valid_simplex():
    """Samples lie in the interior of the simplex (strictly positive, sum to one)."""
    g1, k = 9, 2
    mu = jnp.zeros(g1)
    W = random.normal(random.PRNGKey(0), (g1, k)) * 0.1
    out = _batched_lnm_sample(
        mu=mu,
        W=W,
        d=None,
        n_samples=32,
        key=random.PRNGKey(1),
    )
    assert jnp.all(out > 0)
    assert jnp.allclose(out.sum(axis=-1), 1.0)


def test_batched_lnm_sample_shape():
    """Output batch dimension matches ``n_samples`` and trailing axis is ``G``."""
    g1, k = 7, 2
    g = g1 + 1
    mu = jnp.linspace(-0.2, 0.2, g1)
    W = jnp.ones((g1, k)) * 0.05
    n_samples = 24
    out = _batched_lnm_sample(
        mu=mu,
        W=W,
        d=jnp.full((g1,), 0.02),
        n_samples=n_samples,
        key=random.PRNGKey(3),
    )
    assert out.shape == (n_samples, g)


def test_batched_lnm_sample_with_d_none():
    """Low-rank mode (``d is None``) uses a diagonal floor and still produces simplices."""
    g1, k = 4, 2
    mu = jnp.zeros(g1)
    W = random.normal(random.PRNGKey(10), (g1, k)) * 0.2
    out = _batched_lnm_sample(
        mu=mu,
        W=W,
        d=None,
        n_samples=16,
        key=random.PRNGKey(11),
        floor_var=1e-10,
    )
    assert out.shape[0] == 16
    assert jnp.all(out > 0)
    assert jnp.allclose(out.sum(axis=-1), 1.0)


def test_batched_lnm_sample_with_d():
    """Learned diagonal path passes explicit per-coordinate ALR variance."""
    g1, k = 5, 3
    mu = random.normal(random.PRNGKey(20), (g1,))
    W = random.normal(random.PRNGKey(21), (g1, k)) * 0.08
    d = jnp.full((g1,), 0.05)
    out = _batched_lnm_sample(
        mu=mu,
        W=W,
        d=d,
        n_samples=20,
        key=random.PRNGKey(22),
    )
    assert out.shape == (20, g1 + 1)
    assert jnp.all(out > 0)
    assert jnp.allclose(out.sum(axis=-1), 1.0)
