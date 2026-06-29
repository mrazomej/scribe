"""Tests for LNM diagnostic utilities."""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from scribe.de._lnm_diagnostics import effective_per_gene_nb
from scribe.stats.distributions import LowRankLogisticNormal


def test_effective_per_gene_nb_output_keys():
    """Moment-matching helper returns mean, variance, NB parameters, and mask."""
    g1, k = 9, 2
    mu = jnp.zeros(g1)
    W = jnp.ones((g1, k)) * 0.05
    out = effective_per_gene_nb(
        mu=mu,
        W=W,
        d=None,
        r_T=50.0,
        p=0.3,
        n_mc_samples=256,
        key=random.PRNGKey(0),
    )
    assert set(out.keys()) == {
        "mean",
        "var",
        "r_eff",
        "p_eff",
        "well_defined",
    }


def test_effective_per_gene_nb_shapes():
    """Gene-level summaries are length ``G`` (simplex dimension)."""
    g = 6
    g1 = g - 1
    mu = jnp.linspace(-0.1, 0.1, g1)
    W = jnp.ones((g1, 2)) * 0.02
    d = jnp.full((g1,), 0.01)
    out = effective_per_gene_nb(
        mu=mu,
        W=W,
        d=d,
        r_T=40.0,
        p=0.25,
        n_mc_samples=128,
        key=random.PRNGKey(2),
    )
    for name, arr in out.items():
        assert arr.shape == (g,), name


def test_effective_per_gene_nb_zero_W():
    """With ``W = 0``, ALR uncertainty is tiny so counts marginally behave like fixed ``rho`` times ``u_T``."""
    g = 5
    g1 = g - 1
    mu = jnp.array([0.1, -0.05, 0.0, 0.02])
    W = jnp.zeros((g1, 2))
    d = jnp.full((g1,), 1e-12)
    key = random.PRNGKey(5)

    cov_diag = jnp.maximum(d, 1e-12)
    dist_ln = LowRankLogisticNormal(loc=mu, cov_factor=W, cov_diag=cov_diag)
    rho = dist_ln.sample(key, (4096,))
    v_rho = jnp.var(rho, axis=0, ddof=1)
    assert jnp.all(v_rho < 1e-6)

    out = effective_per_gene_nb(
        mu=mu,
        W=W,
        d=d,
        r_T=30.0,
        p=0.4,
        n_mc_samples=2048,
        key=key,
    )
    assert jnp.all(out["mean"] > 0)


def test_effective_per_gene_nb_positive_mean():
    """Posterior mean count per gene is strictly positive under LNM–NB coupling."""
    g1, k = 7, 2
    mu = random.normal(random.PRNGKey(99), (g1,))
    W = random.normal(random.PRNGKey(98), (g1, k)) * 0.03
    out = effective_per_gene_nb(
        mu=mu,
        W=W,
        d=None,
        r_T=25.0,
        p=0.35,
        n_mc_samples=512,
        key=random.PRNGKey(7),
    )
    assert jnp.all(out["mean"] > 0)
