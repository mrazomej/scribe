"""Unit tests for the SVI-to-empirical-Gaussian-prior adapter.

Covers:

1. ``fit_empirical_gaussian`` round-trip on synthetic LogNormal samples.
2. ``priors_from_results`` coordinate dispatch (``r`` uses positive
   inverse; ``mu`` uses ``jnp.log``, never ``inv_softplus``).
3. Capture-mode detection trichotomy (``eta`` / ``phi_only`` / ``none``).
4. Gene-identity safeguards: var-names mismatch, mask mismatch,
   count-only fallback with warning.
5. Amortized-capture defensive fallback hierarchy
   (``_original_counts`` > strict var-name + source_counts > refusal).
6. Per-method dispatcher: SVI / MCMC handlers do not see
   ``informative_priors`` (Round-5 Finding 1).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import pytest

from scribe.laplace.priors import (
    fit_empirical_gaussian,
    priors_from_results,
)


# =====================================================================
# Fake SVI-results stub
# =====================================================================


class _FakeSVIResults:
    """Stub ScribeSVIResults exposing the surface ``priors_from_results`` uses.

    Tests can configure ``var_names``, ``_gene_coverage_mask``, ``n_genes``,
    the samples returned by ``get_posterior_samples``, and whether
    amortization is reported.
    """

    def __init__(
        self,
        n_genes: int,
        n_cells: int,
        samples: Dict[str, jnp.ndarray],
        var_names: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        amortized: bool = False,
        original_counts: Optional[jnp.ndarray] = None,
        requires_counts: bool = False,
    ):
        self.n_genes = n_genes
        self.n_cells = n_cells
        self._samples = samples
        if var_names is not None:
            self.var_names = var_names
        if mask is not None:
            self._gene_coverage_mask = mask
        self._amortized = amortized
        if original_counts is not None:
            self._original_counts = original_counts
        self._requires_counts = requires_counts

    def _uses_amortized_capture(self) -> bool:
        return self._amortized

    def get_posterior_samples(
        self,
        rng_key: Any = None,
        n_samples: int = 100,
        counts: Optional[jnp.ndarray] = None,
        store_samples: bool = False,
        **_kwargs: Any,
    ) -> Dict[str, jnp.ndarray]:
        if self._requires_counts and counts is None:
            raise RuntimeError(
                "Encoder requires counts but none were passed — this "
                "indicates a bug in the adapter's amortization branch."
            )
        # Trim leading axis to n_samples on each sample array.
        return {k: v[:n_samples] for k, v in self._samples.items()}


def _lognormal_samples(
    n_samples: int, shape: tuple, loc: float = 0.0, scale: float = 0.5
) -> jnp.ndarray:
    """Draw LogNormal samples for a fake posterior."""
    rng = np.random.default_rng(0)
    z = rng.normal(loc=loc, scale=scale, size=(n_samples,) + shape)
    return jnp.asarray(np.exp(z), dtype=jnp.float32)


# =====================================================================
# Layer 1 — fit_empirical_gaussian
# =====================================================================


def test_fit_empirical_gaussian_recovers_log_moments():
    """log of LogNormal samples → Gaussian (loc, scale) within MC error."""
    rng = np.random.default_rng(42)
    true_loc = jnp.asarray([0.5, -0.2, 1.0])
    true_scale = jnp.asarray([0.3, 0.6, 0.1])
    n = 5000
    samples = jnp.asarray(
        rng.normal(loc=np.asarray(true_loc), scale=np.asarray(true_scale),
                   size=(n, 3))
    )
    result = fit_empirical_gaussian(samples, tau=1.0)
    np.testing.assert_allclose(result["loc"], true_loc, atol=0.02)
    np.testing.assert_allclose(result["scale"], true_scale, atol=0.02)


def test_fit_empirical_gaussian_tau_scales_only_scale():
    samples = jnp.asarray(np.random.default_rng(0).normal(0, 1, (1000, 5)))
    a = fit_empirical_gaussian(samples, tau=1.0)
    b = fit_empirical_gaussian(samples, tau=2.5)
    np.testing.assert_allclose(a["loc"], b["loc"])
    np.testing.assert_allclose(b["scale"], 2.5 * a["scale"], atol=1e-6)


def test_fit_empirical_gaussian_eps_floor():
    """Zero-variance coordinates are floored."""
    samples = jnp.asarray(np.ones((10, 3), dtype=np.float32))
    result = fit_empirical_gaussian(samples, tau=1.0, eps_scale=1e-3)
    np.testing.assert_array_less(0.0, np.asarray(result["scale"]))
    assert float(jnp.min(result["scale"])) >= 1e-3 - 1e-9


def test_fit_empirical_gaussian_rejects_singleton():
    with pytest.raises(ValueError, match="at least 2 samples"):
        fit_empirical_gaussian(jnp.ones((1, 5)))


# =====================================================================
# Layer 2 — coordinate dispatch
# =====================================================================


def _make_basic_results(
    G: int = 10,
    N: int = 8,
    n_samples: int = 500,
    *,
    with_eta: bool = True,
    with_phi: bool = False,
    var_names: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    amortized: bool = False,
    requires_counts: bool = False,
):
    """Build a ``_FakeSVIResults`` with standard NBVCP keys."""
    samples: Dict[str, jnp.ndarray] = {
        "r": _lognormal_samples(n_samples, (G,), loc=0.0, scale=0.4),
        "mu": _lognormal_samples(n_samples, (G,), loc=2.0, scale=0.3),
    }
    if with_eta:
        # Truncated-normal at zero — use positive samples.
        rng = np.random.default_rng(1)
        samples["eta_capture"] = jnp.asarray(
            np.abs(rng.normal(0.5, 0.2, (n_samples, N))).astype(np.float32)
        )
    if with_phi:
        samples["phi_capture"] = _lognormal_samples(n_samples, (N,), loc=0.0)

    return _FakeSVIResults(
        n_genes=G,
        n_cells=N,
        samples=samples,
        var_names=var_names,
        mask=mask,
        amortized=amortized,
        requires_counts=requires_counts,
    )


def test_priors_from_results_mu_uses_log_not_pos_inverse():
    """mu coord must use jnp.log regardless of positive_transform.

    This is the Round-1 Finding 1 fix: NBLN's ``params["mu"]`` is the
    prior mean of a real-valued latent log-rate, not a positive
    parameter.  log and inv_softplus disagree at small mu, so use mu
    samples in the range where they diverge.
    """
    G = 5
    rng = np.random.default_rng(2)
    # Choose mu values near 1 where log(mu) != inv_softplus(mu).
    raw_mu = np.abs(rng.normal(0.5, 0.1, (500, G))).astype(np.float32)
    samples = {
        "r": _lognormal_samples(500, (G,)),
        "mu": jnp.asarray(raw_mu),
    }
    var_names = np.array([f"g{i}" for i in range(G)])
    results = _FakeSVIResults(
        n_genes=G, n_cells=4, samples=samples, var_names=var_names
    )

    for pt in ("exp", "softplus"):
        bundle, _ = priors_from_results(
            results,
            target_positive_transform=pt,
            target_n_genes=G,
            target_n_cells=4,
            target_gene_names=var_names,
            n_samples=500,
        )
        # mu prior loc should match mean of log(mu_samples) for BOTH
        # transforms — proving the log path is taken.
        expected_loc = jnp.mean(jnp.log(jnp.maximum(raw_mu, 1e-8)), axis=0)
        np.testing.assert_allclose(
            bundle["mu"]["loc"], expected_loc, atol=1e-5
        )


def test_priors_from_results_r_uses_target_positive_inverse():
    """r coord must use the inverse of target positive_transform."""
    G = 5
    rng = np.random.default_rng(3)
    r_pos = np.abs(rng.normal(2.0, 0.5, (1000, G))).astype(np.float32)
    samples = {
        "r": jnp.asarray(r_pos),
        "mu": _lognormal_samples(1000, (G,)),
    }
    var_names = np.array([f"g{i}" for i in range(G)])
    results = _FakeSVIResults(
        n_genes=G, n_cells=4, samples=samples, var_names=var_names
    )

    # Under exp transform, inverse is log.
    bundle_exp, _ = priors_from_results(
        results, target_positive_transform="exp",
        target_n_genes=G, target_n_cells=4,
        target_gene_names=var_names, n_samples=1000,
    )
    expected_loc_exp = jnp.mean(jnp.log(jnp.maximum(r_pos, 1e-8)), axis=0)
    np.testing.assert_allclose(
        bundle_exp["r"]["loc"], expected_loc_exp, atol=1e-5
    )

    # Under softplus transform, inverse is log(expm1).
    bundle_sp, _ = priors_from_results(
        results, target_positive_transform="softplus",
        target_n_genes=G, target_n_cells=4,
        target_gene_names=var_names, n_samples=1000,
    )
    # They should differ — proving the transform is consulted.
    assert not np.allclose(
        np.asarray(bundle_exp["r"]["loc"]),
        np.asarray(bundle_sp["r"]["loc"]),
        atol=1e-4,
    )


# =====================================================================
# Capture-mode detection
# =====================================================================


def test_capture_mode_eta():
    G, N = 5, 4
    var_names = np.array([f"g{i}" for i in range(G)])
    results = _make_basic_results(
        G=G, N=N, with_eta=True, with_phi=False, var_names=var_names
    )
    bundle, mode = priors_from_results(
        results, target_positive_transform="exp",
        target_n_genes=G, target_n_cells=N,
        target_gene_names=var_names, n_samples=200,
    )
    assert mode == "eta"
    assert "eta" in bundle
    assert bundle["eta"]["loc"].shape == (N,)
    assert bundle["eta"]["scale"].shape == (N,)


def test_capture_mode_phi_only_warns_and_omits_eta():
    G, N = 5, 4
    var_names = np.array([f"g{i}" for i in range(G)])
    results = _make_basic_results(
        G=G, N=N, with_eta=False, with_phi=True, var_names=var_names
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle, mode = priors_from_results(
            results, target_positive_transform="exp",
            target_n_genes=G, target_n_cells=N,
            target_gene_names=var_names, n_samples=200,
        )
    assert mode == "phi_only"
    assert "eta" not in bundle
    assert "r" in bundle and "mu" in bundle
    assert any("phi_capture" in str(w.message) for w in caught)


def test_capture_mode_none_warns_and_omits_eta():
    G, N = 5, 4
    var_names = np.array([f"g{i}" for i in range(G)])
    results = _make_basic_results(
        G=G, N=N, with_eta=False, with_phi=False, var_names=var_names
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle, mode = priors_from_results(
            results, target_positive_transform="exp",
            target_n_genes=G, target_n_cells=N,
            target_gene_names=var_names, n_samples=200,
        )
    assert mode == "none"
    assert "eta" not in bundle
    assert "r" in bundle and "mu" in bundle
    assert any("no capture" in str(w.message) for w in caught)


# =====================================================================
# Gene-identity safeguards
# =====================================================================


def test_gene_identity_var_names_mismatch_raises():
    G, N = 5, 4
    source_names = np.array([f"g{i}" for i in range(G)])
    target_names = np.array([f"x{i}" for i in range(G)])  # different names!
    results = _make_basic_results(G=G, N=N, var_names=source_names)
    with pytest.raises(ValueError, match="var_names"):
        priors_from_results(
            results, target_positive_transform="exp",
            target_n_genes=G, target_n_cells=N,
            target_gene_names=target_names,
        )


def test_gene_identity_count_mismatch_raises():
    G, N = 5, 4
    var_names = np.array([f"g{i}" for i in range(G)])
    results = _make_basic_results(G=G, N=N, var_names=var_names)
    with pytest.raises(ValueError, match="disagree on genes"):
        priors_from_results(
            results, target_positive_transform="exp",
            target_n_genes=G + 1, target_n_cells=N,
        )


def test_gene_identity_count_only_warns():
    """When neither var_names nor mask are available, warn and proceed."""
    G, N = 5, 4
    results = _make_basic_results(G=G, N=N, var_names=None, mask=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        priors_from_results(
            results, target_positive_transform="exp",
            target_n_genes=G, target_n_cells=N,
        )
    assert any("count" in str(w.message).lower() for w in caught)


# =====================================================================
# Amortized-capture defensive fallback
# =====================================================================


def test_amortized_with_original_counts_proceeds():
    """``_original_counts`` on results → encoder fed those."""
    G, N = 4, 3
    var_names = np.array([f"g{i}" for i in range(G)])
    samples = _make_basic_results(G=G, N=N, var_names=var_names)._samples
    original_counts = jnp.asarray(
        np.random.default_rng(0).integers(0, 10, (N, G)), dtype=jnp.float32
    )
    results = _FakeSVIResults(
        n_genes=G, n_cells=N, samples=samples,
        var_names=var_names, amortized=True,
        original_counts=original_counts,
        requires_counts=True,
    )
    # No source_counts passed — the _original_counts path should kick in.
    bundle, mode = priors_from_results(
        results, target_positive_transform="exp",
        target_n_genes=G, target_n_cells=N,
        target_gene_names=var_names, n_samples=100,
    )
    assert "r" in bundle and "mu" in bundle


def test_amortized_no_original_counts_with_source_counts_and_strict_id():
    """Var-name verified + source_counts → encoder fed source_counts."""
    G, N = 4, 3
    var_names = np.array([f"g{i}" for i in range(G)])
    samples = _make_basic_results(G=G, N=N, var_names=var_names)._samples
    source_counts = jnp.asarray(
        np.random.default_rng(0).integers(0, 10, (N, G)), dtype=jnp.float32
    )
    results = _FakeSVIResults(
        n_genes=G, n_cells=N, samples=samples,
        var_names=var_names, amortized=True,
        requires_counts=True,
    )
    bundle, _ = priors_from_results(
        results, target_positive_transform="exp",
        target_n_genes=G, target_n_cells=N,
        target_gene_names=var_names,
        source_counts=source_counts,
        n_samples=100,
    )
    assert "r" in bundle


def test_amortized_no_original_counts_without_strict_id_refuses():
    """Mask-only identity verification is NOT enough for amortized sources."""
    G, N = 4, 3
    mask = np.ones(G, dtype=bool)
    samples = _make_basic_results(
        G=G, N=N, var_names=None, mask=mask
    )._samples
    source_counts = jnp.asarray(
        np.random.default_rng(0).integers(0, 10, (N, G)), dtype=jnp.float32
    )
    # No var_names exposed on results — only mask.  Amortized + no
    # _original_counts + no strict var-name identity → ValueError.
    results = _FakeSVIResults(
        n_genes=G, n_cells=N, samples=samples,
        var_names=None, mask=mask, amortized=True,
        requires_counts=True,
    )
    with pytest.raises(ValueError, match="amortized capture"):
        priors_from_results(
            results, target_positive_transform="exp",
            target_n_genes=G, target_n_cells=N,
            target_gene_mask=mask,  # mask only — no var-names
            source_counts=source_counts,
            n_samples=100,
        )


# =====================================================================
# Per-method dispatcher: ensure handlers' signatures stay clean.
# =====================================================================


def test_svi_handler_does_not_accept_informative_priors():
    """SVI handler must NOT accept ``informative_priors`` (Round-5 Finding 1)."""
    import inspect
    from scribe.inference.dispatcher import _svi_handler, _mcmc_handler

    for handler in (_svi_handler, _mcmc_handler):
        sig = inspect.signature(handler)
        assert "informative_priors" not in sig.parameters, (
            f"{handler.__name__} unexpectedly accepts informative_priors"
        )
        assert "capture_mode_override" not in sig.parameters, (
            f"{handler.__name__} unexpectedly accepts capture_mode_override"
        )


def test_laplace_handler_does_accept_informative_priors():
    """Laplace handler must accept both kwargs."""
    import inspect
    from scribe.inference.dispatcher import _laplace_handler

    sig = inspect.signature(_laplace_handler)
    assert "informative_priors" in sig.parameters
    assert "capture_mode_override" in sig.parameters


# =====================================================================
# Progress messages (verbose flag)
# =====================================================================


def test_verbose_default_prints_progress(capsys):
    """Default ``verbose=True`` emits user-facing progress lines."""
    G, N = 5, 4
    var_names = np.array([f"g{i}" for i in range(G)])
    results = _make_basic_results(G=G, N=N, var_names=var_names)
    priors_from_results(
        results, target_positive_transform="exp",
        target_n_genes=G, target_n_cells=N,
        target_gene_names=var_names, n_samples=200,
    )
    captured = capsys.readouterr().out
    assert "Building informative priors" in captured
    assert "Sampling SVI posterior" in captured
    assert "Fitting empirical Gaussian priors" in captured
    assert "Built informative prior bundle" in captured


def test_verbose_false_silences_progress(capsys):
    """``verbose=False`` suppresses all stdout from priors_from_results."""
    G, N = 5, 4
    var_names = np.array([f"g{i}" for i in range(G)])
    results = _make_basic_results(G=G, N=N, var_names=var_names)
    priors_from_results(
        results, target_positive_transform="exp",
        target_n_genes=G, target_n_cells=N,
        target_gene_names=var_names, n_samples=200,
        verbose=False,
    )
    captured = capsys.readouterr().out
    assert "[scribe.laplace.priors]" not in captured
