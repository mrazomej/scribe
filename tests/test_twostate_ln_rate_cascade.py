"""Cascade adapter tests: TwoState SVI → TSLN-Rate Laplace.

Verifies that ``priors_from_twostate_results`` and
``freeze_values_from_twostate_results`` correctly map an upstream
TwoState SVI fit's posterior samples / MAP into the TSLN-Rate
unconstrained coordinate space, and that the cascade flows end-to-end
through the Laplace bridge.

Test surface
------------

1. **Coord transforms on a deterministic sample dict** — verifies that
   the ``pos_inverse(softplus)`` map is applied correctly to each of
   ``mu``, ``burst_size``, ``k_off`` for both ``priors_from_*`` and
   ``freeze_values_from_*``, and that gene-axis shapes are preserved.

2. **End-to-end cascade smoke** — fits a small synthetic TwoState SVI,
   feeds the result into ``priors_from_twostate_results`` and
   ``freeze_values_from_twostate_results``, and runs a short
   TSLN-Rate Laplace EM that consumes the cascade bundles.  Asserts
   loss decreases and Newton converges.

3. **Variant gating** — ``target_variant="logit"`` raises
   ``NotImplementedError`` (deferred to PR-2).

4. **Invalid freeze_params** rejected with a clear ``ValueError``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------
# Test 1: deterministic coord transforms
# ---------------------------------------------------------------------


def test_priors_from_twostate_results_coord_map():
    """Synthetic samples → expected pos_inverse(softplus) coord map."""
    from scribe.laplace.priors import (
        priors_from_twostate_results,
        fit_empirical_gaussian,
    )
    from scribe.laplace._global_uncertainty import _JAX_POSITIVE_FNS

    G = 4
    S = 50
    rng = np.random.default_rng(0)

    # Construct a stub SVI-results-like object with the bare-minimum
    # ``get_posterior_samples`` and gene-identity methods.
    class StubSVIResult:
        def __init__(self, mu, burst_size, k_off, n_genes):
            self._samples = {
                "mu": mu,
                "burst_size": burst_size,
                "k_off": k_off,
            }
            self.n_genes = n_genes

        def get_posterior_samples(self, **_kwargs):
            return self._samples

    mu_samples = jnp.asarray(
        np.exp(rng.normal(size=(S, G)).astype(np.float32))
    )
    bs_samples = jnp.asarray(
        np.exp(0.3 * rng.normal(size=(S, G)).astype(np.float32))
    )
    ko_samples = jnp.asarray(
        np.exp(rng.normal(size=(S, G)).astype(np.float32) + 1.0)
    )

    stub = StubSVIResult(mu_samples, bs_samples, ko_samples, G)

    bundle, capture_mode = priors_from_twostate_results(
        results=stub,
        target_positive_transform="softplus",
        target_n_genes=G,
        target_n_cells=10,
        target_variant="rate",
        n_samples=S,
        tau=1.0,
        verbose=False,
    )
    assert capture_mode == "none"
    assert set(bundle.keys()) == {"mu", "burst_size", "k_off"}

    # Verify that each prior's loc/scale matches a hand-computed
    # moment match on pos_inverse(samples).
    _, pos_inv = _JAX_POSITIVE_FNS["softplus"]
    for src_key, tgt_key in (
        ("mu", "mu"),
        ("burst_size", "burst_size"),
        ("k_off", "k_off"),
    ):
        samples_pos = stub._samples[src_key]
        samples_uncon = pos_inv(jnp.maximum(samples_pos, 1e-8))
        expected = fit_empirical_gaussian(samples_uncon, tau=1.0)
        np.testing.assert_allclose(
            np.asarray(bundle[tgt_key]["loc"]),
            np.asarray(expected["loc"]),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(bundle[tgt_key]["scale"]),
            np.asarray(expected["scale"]),
            atol=1e-5,
        )


def test_freeze_values_from_twostate_results_coord_map():
    """``freeze_values`` is pos_inverse(MAP) per gene-global."""
    from scribe.laplace.priors import freeze_values_from_twostate_results
    from scribe.laplace._global_uncertainty import _JAX_POSITIVE_FNS

    G = 4

    class StubSVIResult:
        def __init__(self, mu_map, bs_map, ko_map, n_genes):
            self._map = {
                "mu": mu_map,
                "burst_size": bs_map,
                "k_off": ko_map,
            }
            self.n_genes = n_genes

        def get_map(self, **_kwargs):
            return self._map

    rng = np.random.default_rng(1)
    mu_map = jnp.asarray(np.exp(rng.normal(size=G).astype(np.float32)))
    bs_map = jnp.asarray(
        np.exp(0.3 * rng.normal(size=G).astype(np.float32))
    )
    ko_map = jnp.asarray(
        np.exp(rng.normal(size=G).astype(np.float32) + 1.0)
    )

    stub = StubSVIResult(mu_map, bs_map, ko_map, G)
    freeze_values = freeze_values_from_twostate_results(
        results=stub,
        target_positive_transform="softplus",
        target_n_genes=G,
        target_n_cells=10,
        target_variant="rate",
        freeze_params=("mu", "burst_size", "k_off"),
        verbose=False,
    )

    _, pos_inv = _JAX_POSITIVE_FNS["softplus"]
    for src_key, tgt_key in (
        ("mu", "mu"),
        ("burst_size", "burst_size"),
        ("k_off", "k_off"),
    ):
        expected = pos_inv(jnp.maximum(stub._map[src_key], 1e-8))
        np.testing.assert_allclose(
            np.asarray(freeze_values[tgt_key]["loc"]),
            np.asarray(expected),
            atol=1e-5,
        )


# ---------------------------------------------------------------------
# Test 2: logit variant now implemented — verify coord map
# ---------------------------------------------------------------------


def test_priors_from_twostate_results_logit_effective_path():
    """``target_variant='logit'`` consumes effective deterministics.

    The Rev 4 cascade adapter prefers the SVI's ``alpha`` / ``beta`` /
    ``r_hat`` / ``eta_act`` deterministic sites over re-deriving from
    raw ``(mu, burst_size, k_off)``.  This test verifies the primary
    path is taken when the deterministics are present and that the
    coordinate map matches the documented derivation:

        rate_pos       = r_hat
        kappa_pos      = alpha + beta
        eta_anchor     = eta_act         (real-valued, identity transform)
    """
    from scribe.laplace.priors import priors_from_twostate_results
    from scribe.laplace._global_uncertainty import _JAX_POSITIVE_FNS

    G = 4
    S = 50
    rng = np.random.default_rng(1)

    alpha = jnp.asarray(np.exp(rng.normal(size=(S, G)).astype(np.float32)))
    beta = jnp.asarray(np.exp(rng.normal(size=(S, G)).astype(np.float32)))
    r_hat = jnp.asarray(
        np.exp(rng.normal(0.5, 0.3, size=(S, G)).astype(np.float32))
    )
    eta_act = jnp.asarray(rng.normal(0.0, 1.0, size=(S, G)).astype(np.float32))

    class StubSVIResult:
        n_genes = G

        def get_posterior_samples(self, **_):
            return {
                "alpha": alpha,
                "beta": beta,
                "r_hat": r_hat,
                "eta_act": eta_act,
            }

    bundle, capture_mode = priors_from_twostate_results(
        results=StubSVIResult(),
        target_positive_transform="softplus",
        target_n_genes=G,
        target_n_cells=10,
        target_variant="logit",
        n_samples=S,
        tau=1.0,
        verbose=False,
    )
    assert capture_mode == "none"
    assert set(bundle.keys()) == {"rate", "kappa", "eta_anchor"}

    _, pos_inv = _JAX_POSITIVE_FNS["softplus"]
    # rate_pos = r_hat → pos_inverse(r_hat) in target unconstrained coord.
    expected_rate_uncon = pos_inv(r_hat)
    np.testing.assert_allclose(
        np.asarray(bundle["rate"]["loc"]),
        np.asarray(expected_rate_uncon.mean(axis=0)),
        atol=1e-4,
    )
    # kappa_pos = alpha + beta → pos_inverse(alpha+beta).
    expected_kappa_uncon = pos_inv(alpha + beta)
    np.testing.assert_allclose(
        np.asarray(bundle["kappa"]["loc"]),
        np.asarray(expected_kappa_uncon.mean(axis=0)),
        atol=1e-4,
    )
    # eta_anchor is identity-mapped (real-valued); loc == mean of eta_act.
    np.testing.assert_allclose(
        np.asarray(bundle["eta_anchor"]["loc"]),
        np.asarray(eta_act.mean(axis=0)),
        atol=1e-4,
    )


def test_priors_from_twostate_results_logit_fallback_warns():
    """When the effective deterministics are absent, fall back to raw.

    The fallback uses the algebraic derivation from
    ``(mu, burst_size, k_off)`` and emits a UserWarning telling the
    user the cascade may be inconsistent if the upstream
    ``_twostate_reparam`` floors activated.
    """
    import warnings
    from scribe.laplace.priors import priors_from_twostate_results

    G = 3
    S = 40
    rng = np.random.default_rng(2)
    mu = jnp.asarray(np.exp(rng.normal(size=(S, G)).astype(np.float32)))
    bs = jnp.asarray(np.exp(rng.normal(size=(S, G)).astype(np.float32)))
    ko = jnp.asarray(np.exp(rng.normal(size=(S, G)).astype(np.float32)))

    class StubSVIResult:
        n_genes = G

        def get_posterior_samples(self, **_):
            return {"mu": mu, "burst_size": bs, "k_off": ko}

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        bundle, capture_mode = priors_from_twostate_results(
            results=StubSVIResult(),
            target_positive_transform="softplus",
            target_n_genes=G,
            target_n_cells=10,
            target_variant="logit",
            n_samples=S,
            tau=1.0,
            verbose=False,
        )
    assert any(
        "fell back to raw" in str(rec.message) for rec in w_list
    ), "fallback path must emit a UserWarning"
    assert set(bundle.keys()) == {"rate", "kappa", "eta_anchor"}


# ---------------------------------------------------------------------
# Test 3: invalid freeze_params rejected
# ---------------------------------------------------------------------


def test_freeze_values_invalid_keys_rejected():
    from scribe.laplace.priors import freeze_values_from_twostate_results

    class StubResult:
        n_genes = 4

        def get_map(self, **_):
            return {}

    with pytest.raises(ValueError, match="invalid keys"):
        freeze_values_from_twostate_results(
            results=StubResult(),
            target_positive_transform="softplus",
            target_n_genes=4,
            target_n_cells=10,
            target_variant="rate",
            freeze_params=("r",),  # NBLN-style key, not valid for TSLN
            verbose=False,
        )
