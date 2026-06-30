"""End-to-end tests for context-aware posterior sampling on a real fit.

Fits a tiny NBVCP model (variable capture probability => has per-cell capture
sites) and checks that:

- a ``purpose="de_paired"`` draw drops the capture sites and flags the cache
  narrowed (``_posterior_is_full is False``);
- full-generative consumers refuse a narrowed cache
  (``get_predictive_samples`` raises) while ``get_ppc_samples`` re-draws full;
- an explicit ``return_sites`` list filters the returned dict;
- a default (``purpose=None``) draw is unchanged (full key-set, capture kept).
"""

import os

import jax
import pytest

from scribe import fit
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models.config import InferenceConfig, SVIConfig

# Keep this off the GPU and deterministic.
os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture(scope="module")
def nbvcp_fit(small_dataset):
    counts, _ = small_dataset
    model_config = build_config_from_preset(
        model="nbvcp",
        parameterization="standard",
        inference_method="svi",
        unconstrained=False,
        priors={"r": (2, 0.1), "p": (1, 1), "p_capture": (1, 1)},
    )
    inference_config = InferenceConfig.from_svi(
        SVIConfig(n_steps=3, batch_size=5)
    )
    return fit(
        counts=counts,
        model_config=model_config,
        inference_config=inference_config,
        seed=0,
    )


def _has_capture(samples):
    return any("capture" in k for k in samples)


def test_full_draw_keeps_capture_and_flags_full(nbvcp_fit):
    samples = nbvcp_fit.get_posterior_samples(
        rng_key=jax.random.PRNGKey(1), n_samples=8, store_samples=True
    )
    assert "r" in samples
    assert _has_capture(samples)  # NBVCP => capture sites present in a full draw
    assert nbvcp_fit._posterior_is_full is True
    assert nbvcp_fit._posterior_sites is None


def test_de_paired_drops_capture_and_flags_narrowed(nbvcp_fit):
    samples = nbvcp_fit.get_posterior_samples(
        rng_key=jax.random.PRNGKey(2),
        n_samples=8,
        store_samples=True,
        purpose="de_paired",
    )
    assert "r" in samples  # DE's required input is kept
    assert not _has_capture(samples)  # capture noise dropped
    assert all(k in {"r", "p", "mu", "phi"} for k in samples), sorted(samples)
    assert nbvcp_fit._posterior_is_full is False
    assert nbvcp_fit._posterior_sites is not None
    assert "r" in nbvcp_fit._posterior_sites


def test_predictive_raises_on_narrowed_cache(nbvcp_fit):
    # Narrow the cache, then a full-generative consumer must refuse it.
    nbvcp_fit.get_posterior_samples(
        rng_key=jax.random.PRNGKey(3),
        n_samples=8,
        store_samples=True,
        purpose="de_paired",
    )
    assert nbvcp_fit._posterior_is_full is False
    with pytest.raises(RuntimeError, match="narrowed for differential expression"):
        nbvcp_fit.get_predictive_samples(rng_key=jax.random.PRNGKey(4))


def test_ppc_redraws_full_after_narrowing(nbvcp_fit):
    nbvcp_fit.get_posterior_samples(
        rng_key=jax.random.PRNGKey(5),
        n_samples=8,
        store_samples=True,
        purpose="de_paired",
    )
    assert nbvcp_fit._posterior_is_full is False
    # get_ppc_samples should transparently re-draw a FULL posterior and succeed.
    out = nbvcp_fit.get_ppc_samples(rng_key=jax.random.PRNGKey(6), n_samples=8)
    assert "predictive_samples" in out
    assert nbvcp_fit._posterior_is_full is True
    assert _has_capture(nbvcp_fit.posterior_samples)


def test_explicit_return_sites_filters(nbvcp_fit):
    samples = nbvcp_fit.get_posterior_samples(
        rng_key=jax.random.PRNGKey(7),
        n_samples=8,
        store_samples=False,  # don't clobber the shared fixture cache
        return_sites=["r"],
    )
    assert set(samples) == {"r"}


def test_ppc_plot_path_after_de_narrowing_does_not_falsely_raise(
    nbvcp_fit, small_dataset
):
    """Regression for the viz/dispatch stale-flags bug.

    After a DE call narrows the cache, the PPC-plot dispatch draws a fresh FULL
    posterior and stashes it onto the result; the full-cache guard must NOT trip
    (it would if the stale narrowed flag were not reset in lock-step).
    """
    from scribe.viz.dispatch import _get_predictive_samples_for_plot

    counts, _ = small_dataset
    # Narrow the cache exactly as compare_groups would.
    nbvcp_fit.get_posterior_samples(
        rng_key=jax.random.PRNGKey(11),
        n_samples=8,
        store_samples=True,
        purpose="de_paired",
    )
    assert nbvcp_fit._posterior_is_full is False  # cache is narrowed

    # The PPC-plot path draws a full posterior + generates predictive counts.
    # Before the fix this raised RuntimeError from the full-cache guard.
    out = _get_predictive_samples_for_plot(
        nbvcp_fit,
        rng_key=jax.random.PRNGKey(12),
        n_samples=8,
        counts=counts,
        store_samples=False,
    )
    assert out is not None and out.shape[0] == 8
    # store_samples=False restored the previous (narrowed) cache AND its flags
    # in lock-step.
    assert nbvcp_fit._posterior_is_full is False
