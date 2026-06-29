"""Tests for conditional vs generative PPC modes on LNM-family fits.

When ``counts`` is passed to ``get_predictive_samples`` for an LNM
or LNMVCP fit, scribe should switch to *conditional* PPC: the
per-cell ``u_T`` total is fixed at ``sum(counts, axis=-1)`` via
``numpyro.handlers.condition``, and only the multinomial draw over
genes is sampled fresh. When ``counts`` is ``None``, the original
generative PPC behaviour is preserved (every latent re-sampled
from its posterior draw, including ``u_T``).

These tests use small synthetic LNM data to keep run-time low and
verify:

* The per-cell totals of conditional-PPC samples exactly match the
  observed totals (so ``u_T`` was indeed fixed).
* The cross-sample variance of per-cell totals is zero in
  conditional mode and **non**-zero in generative mode (so the two
  modes are observably different).
* For non-LNM models the ``counts`` argument is silently ignored
  (no error, no behavioural change).
"""

from __future__ import annotations

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _make_synthetic_lnm_adata(n_cells=80, n_genes=8, seed=0):
    """Generate a small synthetic LNM dataset for fast PPC tests.

    Returns an AnnData with biologically-plausible counts: each cell
    draws a total from a NB-ish distribution, then composition from
    a softmax of a Gaussian latent. We don't try to be faithful to
    the model — just non-degenerate counts where every gene has
    positive variance and total counts vary cell-to-cell.
    """
    rng = np.random.default_rng(seed)
    # Latent factor scores per cell.
    z = rng.normal(size=(n_cells, 2)).astype(np.float32)
    # Decoder loadings.
    W = rng.normal(size=(n_genes, 2)).astype(np.float32) * 0.5
    mu = rng.normal(size=n_genes).astype(np.float32) * 0.3
    logits = mu + z @ W.T
    probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    # Per-cell totals: vary from ~50 to ~500.
    totals = rng.integers(50, 500, size=n_cells)
    counts = np.zeros((n_cells, n_genes), dtype=np.int64)
    for c in range(n_cells):
        counts[c] = rng.multinomial(int(totals[c]), np.asarray(probs[c]))
    return ad.AnnData(counts.astype(np.float32))


@pytest.fixture(scope="module")
def lnm_fit():
    """Cheap LNM fit on synthetic data — kept tiny for fast iteration."""
    import scribe

    adata = _make_synthetic_lnm_adata(n_cells=60, n_genes=6, seed=1)
    result = scribe.fit(
        adata,
        model="lnm",
        inference_method="vae",
        vae_latent_dim=2,
        vae_encoder_hidden_dims=[16, 8],
        n_steps=200,
        batch_size=None,
        seed=1,
    )
    return result, adata


# =====================================================================
# Conditional PPC — per-cell totals match observed
# =====================================================================


class TestConditionalPPC:
    """When ``counts`` is provided, the predictive's per-cell totals
    must match the observed library sizes exactly.
    """

    def test_per_cell_totals_match_observed_when_counts_provided(
        self, lnm_fit
    ):
        result, adata = lnm_fit
        counts = np.asarray(adata.X)
        # Posterior samples first (encoder uses counts).
        result.get_posterior_samples(
            counts=jnp.asarray(counts),
            n_samples=8,
            rng_key=jax.random.PRNGKey(0),
            store_samples=True,
        )
        # Conditional PPC: pass counts → u_T fixed at observed totals.
        pred = result.get_predictive_samples(
            rng_key=jax.random.PRNGKey(0),
            counts=jnp.asarray(counts),
            store_samples=False,
        )
        pred_arr = np.asarray(pred)
        # Sum across genes for each (sample, cell). The result must
        # equal the observed per-cell totals broadcast across the
        # n_samples axis exactly — no NB resampling happened.
        observed_totals = counts.sum(axis=-1).astype(np.int64)
        per_cell_totals = pred_arr.sum(axis=-1)  # (n_samples, n_cells)
        for s in range(per_cell_totals.shape[0]):
            np.testing.assert_array_equal(
                per_cell_totals[s].astype(np.int64),
                observed_totals,
                err_msg=(
                    f"sample {s}: predicted per-cell totals do not "
                    f"match observed library sizes — conditional "
                    f"PPC failed to fix u_T."
                ),
            )

    def test_per_cell_total_variance_is_zero_conditional(self, lnm_fit):
        """Across PPC samples for the same cell, conditional mode
        should produce zero variance in per-cell totals (because
        u_T is fixed). Generative mode produces non-zero variance.
        """
        result, adata = lnm_fit
        counts = np.asarray(adata.X)
        result.get_posterior_samples(
            counts=jnp.asarray(counts),
            n_samples=12,
            rng_key=jax.random.PRNGKey(2),
            store_samples=True,
        )
        # Conditional: across-sample std of per-cell totals = 0.
        pred_cond = result.get_predictive_samples(
            rng_key=jax.random.PRNGKey(3),
            counts=jnp.asarray(counts),
            store_samples=False,
        )
        totals_cond = np.asarray(pred_cond).sum(axis=-1)  # (S, C)
        std_per_cell_cond = totals_cond.std(axis=0)  # across samples
        assert float(std_per_cell_cond.max()) == 0.0


# =====================================================================
# Generative PPC — totals re-rolled
# =====================================================================


class TestGenerativePPC:
    """When ``counts`` is None, the predictive resamples u_T from the
    NB total-count distribution. Per-cell totals across PPC samples
    should vary.
    """

    def test_per_cell_totals_vary_when_counts_none(self, lnm_fit):
        result, adata = lnm_fit
        counts = np.asarray(adata.X)
        result.get_posterior_samples(
            counts=jnp.asarray(counts),
            n_samples=12,
            rng_key=jax.random.PRNGKey(4),
            store_samples=True,
        )
        pred_gen = result.get_predictive_samples(
            rng_key=jax.random.PRNGKey(5),
            counts=None,  # generative mode
            store_samples=False,
        )
        totals_gen = np.asarray(pred_gen).sum(axis=-1)  # (S, C)
        std_per_cell_gen = totals_gen.std(axis=0)
        # At least some cells have non-zero across-sample variance —
        # if every cell has zero variance, NB resampling didn't happen.
        assert float(std_per_cell_gen.max()) > 0.0


# =====================================================================
# Non-LNM models — counts kwarg ignored
# =====================================================================


class TestNonLNMIgnoresCounts:
    """For non-LNM models, the ``counts`` argument to
    ``get_predictive_samples`` is silently ignored (no
    ``NotImplementedError``, no behaviour change).

    We verify this on a tiny NBDM fit: passing counts must not
    raise, and the resulting predictive has the same shape as
    when counts is None. The actual values may differ run-to-run
    so we don't assert array equality.
    """

    def test_nbdm_ignores_counts_kwarg(self):
        import scribe

        adata = _make_synthetic_lnm_adata(n_cells=40, n_genes=5, seed=6)
        result = scribe.fit(
            adata,
            model="nbdm",
            inference_method="svi",
            n_steps=50,
            batch_size=None,
            seed=6,
        )
        result.get_posterior_samples(
            n_samples=4,
            rng_key=jax.random.PRNGKey(7),
            store_samples=True,
        )
        # Should not raise even though counts is provided.
        pred = result.get_predictive_samples(
            rng_key=jax.random.PRNGKey(8),
            counts=jnp.asarray(adata.X),
            store_samples=False,
        )
        # Shape sanity: (n_samples, n_cells, n_genes).
        assert pred.shape == (4, 40, 5)
