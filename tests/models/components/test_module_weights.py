"""
Tests for the additive per-leaf module-weight SVI primitives
(:mod:`scribe.models.components.module_weights`).

These cover the *shared* Rung-1.5 hierarchical-correlation primitive used by
both the SVI/VAE sanity-check (Phase A) and the Laplace production path
(Phase B):

- ``effective_loadings``: the ``W_eff,d = W . diag(s_d)`` collapse that keeps
  ``Sigma_d`` low-rank-plus-diagonal (the algebraic identity Phase B relies on).
- ``sample_module_weights_hierarchical``: the emitted site names + shapes for a
  single-factor spec AND a crossed 2-factor + interaction spec; the *global
  leaf-anchor* gauge (``sum_leaf log s == 0`` per module) on
  ``module_weight_log``; strict positivity of ``module_weight``; and the
  degenerate single-leaf (``n_leaves == 1``) case where the gauge forces
  ``s == 1``.
- ``guide_module_weights_hierarchical``: the mean-field guide registers exactly
  the latent sites the model sampler emits (so the mean-field ELBO pairs).

The `GroupingSpec` construction helpers mirror
``tests/models/components/test_module_weight_factors.py``.

Note on what is intentionally NOT asserted: absolute uniqueness of ``(W, s)``.
The leaf anchor removes only the *column scale* gauge; column sign/permutation
always remain and the rotation gauge is only generically broken. We therefore
test scale-gauge removal (``sum_leaf log s == 0``), not uniqueness.
"""

import pytest
import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp
import numpyro

from scribe.models.config.grouping import (
    Factor,
    GroupingSpec,
    PriorFamilySpec,
)
from scribe.models.components.module_weights import (
    sample_module_weights_hierarchical,
    guide_module_weights_hierarchical,
    effective_loadings,
)


# ---------------------------------------------------------------------------
# GroupingSpec construction helpers (mirror test_module_weight_factors.py)
# ---------------------------------------------------------------------------


def _factor(name, kind, levels, leaf_to_level, *, family="gaussian",
            effect_type="random", fixed_scale=None):
    priors = (
        {}
        if family is None
        else {"module_weight": PriorFamilySpec(type=family)}
    )
    return Factor(
        name=name,
        kind=kind,
        nested_in=None,
        effect_type=effect_type,
        fixed_scale=fixed_scale,
        levels=tuple(levels),
        leaf_to_level=tuple(leaf_to_level),
        priors=priors,
    )


def _single_factor_spec(n_leaves):
    """One random base factor with identity gather (the flat fast path)."""
    donors = [f"D{i}" for i in range(n_leaves)]
    samp = _factor("donor", "base", donors, list(range(n_leaves)))
    return GroupingSpec(
        factors=(samp,), leaf_labels=tuple(donors), n_leaves=n_leaves
    )


def _crossed_2x7_spec():
    """A complete 2 (perturbation) x 7 (sample) crossed + interaction spec."""
    donors = [f"D{i}" for i in range(7)]
    pert = _factor(
        "perturbation", "base", ["ctrl", "pano"], [0] * 7 + [1] * 7,
        effect_type="fixed",
    )
    samp = _factor("sample", "base", donors, list(range(7)) * 2)
    inter = _factor(
        "perturbation:sample", "interaction",
        [f"c{i}" for i in range(14)], list(range(14)),
    )
    return GroupingSpec(
        factors=(pert, samp, inter),
        leaf_labels=tuple(f"L{i}" for i in range(14)),
        n_leaves=14,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def dims():
    # (G genes, K modules, D leaves) -- small but non-degenerate.
    return {"G": 12, "K": 4, "D": 5}


def _trace(fn, key, *args, **kwargs):
    """Seed + trace a NumPyro-effectful callable, return its site trace."""
    seeded = numpyro.handlers.seed(fn, key)
    return numpyro.handlers.trace(seeded).get_trace(*args, **kwargs)


# ---------------------------------------------------------------------------
# effective_loadings: the W_eff collapse
# ---------------------------------------------------------------------------


def test_effective_loadings_single_donor_collapse(rng, dims):
    """1-D ``s`` yields (G, K) loadings with W_eff W_eff^T == W diag(s^2) W^T."""
    G, K = dims["G"], dims["K"]
    kW, ks = jax.random.split(rng)
    W = jax.random.normal(kW, (G, K))
    s = jnp.exp(0.3 * jax.random.normal(ks, (K,)))  # positive scales

    W_eff = effective_loadings(W, s)
    assert W_eff.shape == (G, K)

    # The defining identity: the low-rank part of Sigma is unchanged when the
    # column scaling is folded into the loadings.
    lhs = W_eff @ W_eff.T
    rhs = W @ jnp.diag(s**2) @ W.T
    npt.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-5, atol=1e-5)


def test_effective_loadings_multi_donor_collapse(rng, dims):
    """2-D ``s`` yields (D, G, K) loadings; each slice reproduces Sigma_d."""
    G, K, D = dims["G"], dims["K"], dims["D"]
    kW, ks = jax.random.split(rng)
    W = jax.random.normal(kW, (G, K))
    s = jnp.exp(0.3 * jax.random.normal(ks, (D, K)))

    W_eff = effective_loadings(W, s)
    assert W_eff.shape == (D, G, K)

    # Every donor slice must reconstruct that donor's low-rank covariance.
    for d in range(D):
        lhs = W_eff[d] @ W_eff[d].T
        rhs = W @ jnp.diag(s[d] ** 2) @ W.T
        npt.assert_allclose(
            np.asarray(lhs), np.asarray(rhs), rtol=1e-5, atol=1e-5
        )


def test_effective_loadings_scale_gauge_invariance(rng, dims):
    """Sigma_d is invariant under W[:,k]->a_k W[:,k], s->s/a_k (the scale gauge)."""
    G, K, D = dims["G"], dims["K"], dims["D"]
    kW, ks, ka = jax.random.split(rng, 3)
    W = jax.random.normal(kW, (G, K))
    s = jnp.exp(0.3 * jax.random.normal(ks, (D, K)))
    a = jnp.exp(0.5 * jax.random.normal(ka, (K,)))  # arbitrary positive per-column

    W_g = W * a[None, :]  # rescale columns
    s_g = s / a[None, :]  # compensate in the scales

    for d in range(D):
        base = effective_loadings(W, s)[d]
        gauged = effective_loadings(W_g, s_g)[d]
        npt.assert_allclose(
            np.asarray(base @ base.T),
            np.asarray(gauged @ gauged.T),
            rtol=1e-5,
            atol=1e-5,
        )


def test_effective_loadings_validates_shapes(dims):
    """Mismatched K and bad ndim raise informative ValueErrors."""
    G, K = dims["G"], dims["K"]
    W = jnp.ones((G, K))
    with pytest.raises(ValueError, match="W has K"):
        effective_loadings(W, jnp.ones((K + 1,)))
    with pytest.raises(ValueError, match="W has K"):
        effective_loadings(W, jnp.ones((3, K + 1)))
    with pytest.raises(ValueError, match="1-D .* or 2-D"):
        effective_loadings(W, jnp.ones((2, 3, K)))


# ---------------------------------------------------------------------------
# sample_module_weights_hierarchical: site names + shapes
# ---------------------------------------------------------------------------


def test_single_factor_site_names_and_shapes(rng, dims):
    """A single random base factor emits the flat-path sites with right shapes."""
    K, D = dims["K"], dims["D"]
    spec = _single_factor_spec(D)
    tr = _trace(sample_module_weights_hierarchical, rng, spec, K)

    # Per-factor latents (site == factor name with ':' -> '__'; here "donor").
    assert tr["module_weight_raw__donor"]["value"].shape == (D, K)
    assert tr["module_weight_tau_raw__donor"]["value"].shape == ()  # random
    # Per-factor deterministic effect.
    assert tr["module_weight_effect__donor"]["value"].shape == (D, K)
    # Global deterministics.
    assert tr["module_weight_log"]["value"].shape == (D, K)
    assert tr["module_weight"]["value"].shape == (D, K)


def test_crossed_site_names_and_shapes(rng):
    """The 2x7 crossed + interaction spec emits one block per factor.

    ``perturbation`` is a FIXED factor (2 levels) -> no ``tau_raw`` site;
    ``sample`` (7 levels) and the interaction (14 levels) are RANDOM.
    """
    K = 3
    spec = _crossed_2x7_spec()
    tr = _trace(sample_module_weights_hierarchical, rng, spec, K)

    # perturbation: 2 levels, FIXED (no tau_raw).
    assert tr["module_weight_raw__perturbation"]["value"].shape == (2, K)
    assert "module_weight_tau_raw__perturbation" not in tr
    assert tr["module_weight_effect__perturbation"]["value"].shape == (2, K)

    # sample: 7 levels, RANDOM (has tau_raw).
    assert tr["module_weight_raw__sample"]["value"].shape == (7, K)
    assert tr["module_weight_tau_raw__sample"]["value"].shape == ()
    assert tr["module_weight_effect__sample"]["value"].shape == (7, K)

    # interaction: 14 levels, RANDOM. Site uses '__' for the ':' separator.
    assert tr["module_weight_raw__perturbation__sample"]["value"].shape == (14, K)
    assert tr["module_weight_tau_raw__perturbation__sample"]["value"].shape == ()
    assert (
        tr["module_weight_effect__perturbation__sample"]["value"].shape
        == (14, K)
    )

    # Global deterministics on the leaf axis (n_leaves == 14).
    assert tr["module_weight_log"]["value"].shape == (14, K)
    assert tr["module_weight"]["value"].shape == (14, K)


# ---------------------------------------------------------------------------
# The global leaf-anchor gauge + positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_fn, n_leaves, K",
    [
        (lambda: _single_factor_spec(5), 5, 4),
        (_crossed_2x7_spec, 14, 3),
    ],
    ids=["single_factor", "crossed_2x7"],
)
def test_global_leaf_anchor_gauge(rng, spec_fn, n_leaves, K):
    """The scale gauge is fixed: sum over leaves of log s is ~0 per module."""
    spec = spec_fn()
    tr = _trace(sample_module_weights_hierarchical, rng, spec, K)
    log_s = np.asarray(tr["module_weight_log"]["value"])
    assert log_s.shape == (n_leaves, K)
    # Sum (equivalently mean) over the leaf axis must vanish for every module.
    npt.assert_allclose(log_s.sum(axis=0), np.zeros(K), atol=1e-5)


@pytest.mark.parametrize(
    "spec_fn, K",
    [
        (lambda: _single_factor_spec(5), 4),
        (_crossed_2x7_spec, 3),
    ],
    ids=["single_factor", "crossed_2x7"],
)
def test_module_weight_positive(rng, spec_fn, K):
    """Constrained module weights are strictly positive."""
    spec = spec_fn()
    tr = _trace(sample_module_weights_hierarchical, rng, spec, K)
    s = np.asarray(tr["module_weight"]["value"])
    assert np.all(s > 0.0)
    # ``module_weight == exp(module_weight_log)``.
    npt.assert_allclose(
        s, np.exp(np.asarray(tr["module_weight_log"]["value"])),
        rtol=1e-5, atol=1e-6,
    )


def test_degenerate_single_leaf(rng):
    """With n_leaves == 1 the centering forces s == 1 (no hierarchy)."""
    K = 6
    spec = _single_factor_spec(1)
    tr = _trace(sample_module_weights_hierarchical, rng, spec, K)
    s = np.asarray(tr["module_weight"]["value"])
    assert s.shape == (1, K)
    npt.assert_allclose(s, np.ones((1, K)), atol=1e-6)


def test_custom_site_prefix(rng, dims):
    """A custom site_prefix renames every site consistently."""
    K, D = dims["K"], dims["D"]
    spec = _single_factor_spec(D)
    tr = _trace(
        sample_module_weights_hierarchical, rng, spec, K, site_prefix="act"
    )
    assert "act" in tr
    assert "act_log" in tr
    assert "act_raw__donor" in tr
    assert "act_tau_raw__donor" in tr
    assert "act_effect__donor" in tr
    # No leakage of the default prefix.
    assert "module_weight" not in tr
    assert "module_weight_raw__donor" not in tr


# ---------------------------------------------------------------------------
# guide_module_weights_hierarchical: latent-site pairing with the model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_fn, K, expected_latents",
    [
        (
            lambda: _single_factor_spec(5),
            4,
            {"module_weight_raw__donor", "module_weight_tau_raw__donor"},
        ),
        (
            _crossed_2x7_spec,
            3,
            {
                # perturbation is FIXED -> raw only, no tau_raw.
                "module_weight_raw__perturbation",
                "module_weight_raw__sample",
                "module_weight_tau_raw__sample",
                "module_weight_raw__perturbation__sample",
                "module_weight_tau_raw__perturbation__sample",
            },
        ),
    ],
    ids=["single_factor", "crossed_2x7"],
)
def test_guide_pairs_model_latent_sites(rng, spec_fn, K, expected_latents):
    """The guide registers exactly the latent sites the model emits.

    The mean-field ELBO requires every *latent* (non-deterministic,
    non-observed) model site to have a matching guide site. The deterministic
    sites (``module_weight``, ``module_weight_log``, and every
    ``module_weight_effect__*``) are functions of the latents and must NOT
    appear in the guide.
    """
    spec = spec_fn()

    model_tr = _trace(sample_module_weights_hierarchical, rng, spec, K)
    model_latents = {
        name
        for name, site in model_tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }

    guide_tr = _trace(guide_module_weights_hierarchical, rng, spec, K)
    guide_latents = {
        name
        for name, site in guide_tr.items()
        if site["type"] == "sample"
    }

    assert model_latents == guide_latents == expected_latents
    # Guide latent shapes must match the model's.
    for name in expected_latents:
        assert (
            guide_tr[name]["value"].shape == model_tr[name]["value"].shape
        ), name
