"""Regression tests for the TwoState regime-coordinate prior override.

The two-state ``moment_delta`` parameterization carries a per-gene regime
coordinate ``inv_concentration`` (= delta = 1 / (kappa + 1) in (0, 1)).  Its
logit-Normal prior controls how strongly the fit is pulled toward the NB limit
(small delta) versus the bursty regime (delta near 1).

Two bugs were fixed (see ``build_inv_concentration_spec`` in
``models/presets/registry.py`` and ``create_model`` Step 6 in
``models/presets/factory.py``):

* The user override stored on ``model_config.priors`` (from
  ``scribe.fit(priors={"inv_concentration": (loc, scale)})``) never reached the
  ``inv_concentration`` spec, because that spec is built *inside* the factory
  and the Step-6 merge only pulled from the (empty-at-that-point)
  ``model_config.param_specs`` plus the ``priors`` *argument*.  The override is
  now folded in for currently-valid spec names, and the mean-field guide init
  is centered on it (the guide initializes from ``default_params`` / ``guide``,
  not ``prior``, so without this the override would not move the optimum).

* The built-in default ``prior_loc`` was softened from ``-4.0`` to ``-2.0``.

These tests assert (a) the override reaches the spec / model trace / guide
init, and (b) a short fit with two different ``inv_concentration`` priors
yields materially different ``k_off``.
"""

import io
import contextlib
import logging

import numpy as np
import numpyro
import pytest
from numpyro.handlers import seed, trace

import scribe
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models.model_registry import get_model_and_guide


# ==============================================================================
# (a) Override reaches the spec / model trace / guide init
# ==============================================================================


def _build(priors):
    """Build the (model, guide, resolved config) via the real fit path."""
    cfg = build_config_from_preset(
        model="twostatevcp",
        parameterization="moment_delta",
        inference_method="svi",
        unconstrained=True,
        positive_transform="exp",
        priors=priors,
    )
    model, guide, cfg_res = get_model_and_guide(cfg, n_genes=4)
    return model, guide, cfg_res


def _inv_conc_spec(cfg_res):
    for s in cfg_res.param_specs:
        if s.name == "inv_concentration":
            return s
    raise AssertionError("inv_concentration spec not found")


def _normal_base(dist_fn):
    base = dist_fn
    while hasattr(base, "base_dist"):
        base = base.base_dist
    return base


class TestOverrideReachesSpec:
    """The ``priors={"inv_concentration": ...}`` override must flow through."""

    def test_default_prior_loc_is_softened(self):
        # Bug B: the built-in default is now Normal(-2, 2), not Normal(-4, 2).
        _, _, cfg_res = _build(None)
        spec = _inv_conc_spec(cfg_res)
        assert spec.prior is None
        assert spec.default_params == (-2.0, 2.0)

    def test_override_reaches_spec_prior(self):
        _, _, cfg_res = _build({"inv_concentration": (0.0, 2.0)})
        spec = _inv_conc_spec(cfg_res)
        # The value the model samples from: .prior wins over default_params.
        assert spec.prior == (0.0, 2.0)
        # Sanity: not the softened default and not the old aggressive tilt.
        assert spec.default_params != (0.0, 2.0)
        assert spec.prior != (-2.0, 2.0)

    def test_override_reaches_model_trace(self):
        n_cells, n_genes = 8, 4
        counts = np.zeros((n_cells, n_genes), dtype="int64")
        model, _, cfg_res = _build({"inv_concentration": (0.0, 2.0)})
        with seed(rng_seed=0):
            tr = trace(model).get_trace(
                n_cells=n_cells,
                n_genes=n_genes,
                model_config=cfg_res,
                counts=counts,
            )
        site = next(
            s
            for name, s in tr.items()
            if name == "inv_concentration" and s["type"] == "sample"
        )
        base = _normal_base(site["fn"])
        assert float(np.asarray(base.loc).flatten()[0]) == pytest.approx(0.0)

    def test_override_moves_guide_init(self):
        # The mean-field guide init must follow the override so SVI starts in
        # the right place; otherwise the override would not control the fit.
        n_cells, n_genes = 8, 4
        counts = np.zeros((n_cells, n_genes), dtype="int64")
        _, guide, cfg_res = _build({"inv_concentration": (0.0, 2.0)})
        with seed(rng_seed=0):
            gtr = trace(guide).get_trace(
                n_cells=n_cells,
                n_genes=n_genes,
                model_config=cfg_res,
                counts=counts,
            )
        loc = next(
            s["value"]
            for name, s in gtr.items()
            if name.endswith("inv_concentration_loc") and s["type"] == "param"
        )
        assert float(np.asarray(loc).flatten()[0]) == pytest.approx(0.0)


# ==============================================================================
# (b) Two different priors yield materially different k_off
# ==============================================================================


@pytest.fixture(scope="module")
def bursty_counts():
    """Synthetic counts from a genuinely bursty regime (k_off ~ k_on)."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 400, 3
    k_on = np.full(n_genes, 0.3)
    k_off = np.full(n_genes, 0.3)
    rhat = np.full(n_genes, 80.0)
    nu = rng.beta(40, 8, n_cells)
    p = rng.beta(k_on[None], k_off[None], size=(n_cells, n_genes))
    return rng.poisson(rhat[None] * p * nu[:, None]).astype("int64")


def _fit_median_k_off(counts, priors):
    logging.disable(logging.INFO)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="moment_delta",
            inference_method="svi",
            unconstrained=True,
            positive_transform="exp",
            n_steps=3000,
            batch_size=256,
            seed=42,
            optimizer_config={
                "name": "clipped_adam",
                "step_size": 1e-3,
                "grad_clip_norm": 10.0,
            },
            priors=priors,
        )
        ps = res.get_posterior_samples(
            n_samples=200, store_samples=False, convert_to_numpy=True
        )
    return np.median(ps["k_off"], axis=0)


@pytest.mark.slow
def test_regime_prior_controls_fit(bursty_counts):
    """A neutral regime prior recovers burstiness; an NB-tilt crushes it.

    Truth is ``k_off ~ 0.3`` (bursty).  The aggressive NB-tilt
    (``loc=-4``) inflates fitted ``k_off`` far above truth, while the
    neutral prior (``loc=0``) stays close to it.  The two fits must differ
    materially (the whole point of the override fix).
    """
    nb_tilt = _fit_median_k_off(bursty_counts, {"inv_concentration": (-4.0, 2.0)})
    neutral = _fit_median_k_off(bursty_counts, {"inv_concentration": (0.0, 2.0)})

    # They must not be (bit-)identical: the override genuinely changes the fit.
    assert not np.allclose(nb_tilt, neutral), (
        f"override had no effect: nb_tilt={nb_tilt}, neutral={neutral}"
    )

    # The NB-tilt should sit well above the neutral fit on every gene.
    assert np.all(nb_tilt > neutral)

    # Neutral must be much closer to the bursty truth (k_off ~ 0.3) than the
    # NB-tilt.  Use the median across genes for a robust scalar comparison.
    truth = 0.3
    assert np.median(np.abs(neutral - truth)) < np.median(
        np.abs(nb_tilt - truth)
    )
    # Neutral should land in the bursty ballpark, not the NB limit.
    assert np.median(neutral) < 2.0
    # NB-tilt should be inflated several-fold above neutral.
    assert np.median(nb_tilt) > 2.0 * max(np.median(neutral), 1e-3)
