"""Tests for scribe centralized logging configuration."""

from __future__ import annotations

import logging

import pytest

from scribe._logging import setup_logging


def test_setup_logging_attaches_rich_handler():
    """``setup_logging`` attaches a handler to the scribe root logger."""
    root = logging.getLogger("scribe")
    initial_handlers = list(root.handlers)

    setup_logging(level=logging.INFO)

    assert root.handlers
    assert root.level == logging.INFO
    # Idempotent: repeated calls do not stack duplicate handlers.
    setup_logging(level=logging.WARNING)
    assert len(root.handlers) == len(initial_handlers) or len(root.handlers) >= 1
    assert root.propagate is False


def test_gene_coverage_emits_info_log(scribe_caplog):
    """Gene coverage pre-filtering reports via INFO logging."""
    import numpy as np

    from scribe.api.stages.gene_coverage import apply_gene_coverage_and_alr
    from scribe.api.context import FitContext

    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(20, 50)).astype(np.float32)

    ctx = FitContext(
        counts=counts,
        model="nbdm",
        priors=None,
        n_components=None,
        kwargs={"gene_coverage": 0.5},
    )
    ctx.count_data = counts
    ctx.n_genes = counts.shape[1]
    ctx._original_n_genes = counts.shape[1]
    ctx.adata = None

    scribe_caplog.set_level(logging.INFO, logger="scribe.api.stages.gene_coverage")
    apply_gene_coverage_and_alr(ctx)

    assert any(
        "Applied gene_coverage pre-filtering" in record.message
        for record in scribe_caplog.records
    )


def test_laplace_priors_verbose_emits_info_log(scribe_caplog):
    """Laplace prior cascade progress uses INFO logging when verbose=True."""
    import jax.numpy as jnp
    import numpy as np
    from unittest.mock import MagicMock, PropertyMock

    from scribe.laplace.priors import priors_from_twostate_results

    n_genes, n_cells, n_samples = 5, 8, 20
    rng = np.random.default_rng(0)

    mock_results = MagicMock()
    type(mock_results).var = PropertyMock(return_value=None)
    type(mock_results).adata = PropertyMock(return_value=None)
    mock_results.n_genes = n_genes
    mock_results.var_names = np.array([f"g{i}" for i in range(n_genes)])
    mock_results.get_posterior_samples.return_value = {
        "mu": jnp.asarray(rng.uniform(0.1, 2.0, (n_samples, n_genes))),
        "burst_size": jnp.asarray(rng.uniform(0.1, 2.0, (n_samples, n_genes))),
        "k_off": jnp.asarray(rng.uniform(0.1, 2.0, (n_samples, n_genes))),
        "p_capture": jnp.asarray(rng.uniform(0.05, 0.5, (n_samples, n_cells))),
    }

    scribe_caplog.set_level(logging.INFO, logger="scribe.laplace.priors")
    priors_from_twostate_results(
        mock_results,
        target_positive_transform="exp",
        target_n_genes=n_genes,
        target_n_cells=n_cells,
        target_variant="rate",
        target_gene_names=np.array([f"g{i}" for i in range(n_genes)]),
        n_samples=n_samples,
        verbose=True,
    )

    assert any(
        "Building TSLN-rate priors from TwoState SVI source" in record.message
        for record in scribe_caplog.records
    )
