"""
Tests for explicit gene selection in PPC plotting.

Covers ``scribe.viz.gene_selection._resolve_explicit_genes``, the helper
backing ``plot_ppc(..., genes=[...])``.  These tests do not require a
fitted model — they exercise name/index resolution, order preservation,
and the error paths against a minimal results-like stand-in.
"""

import numpy as np
import pytest

from scribe.viz.gene_selection import _resolve_explicit_genes


class _FakeVar:
    """Minimal stand-in for ``results.var`` with a pandas-like index."""

    def __init__(self, names):
        self.index = np.array(names)


class _FakeResults:
    """Minimal results-like object exposing gene names and ``n_genes``."""

    def __init__(self, n_genes, gene_names=None):
        self.n_genes = n_genes
        self.var = _FakeVar(gene_names) if gene_names is not None else None


@pytest.fixture
def fake_results():
    names = [f"Gene_{i}" for i in range(10)]
    return _FakeResults(n_genes=10, gene_names=names)


def test_resolve_names_preserves_order(fake_results):
    # Deliberately out-of-order names -> resolved indices follow the
    # caller order, not the gene-axis order.
    out = _resolve_explicit_genes(["Gene_5", "Gene_2", "Gene_9"], fake_results)
    assert out.tolist() == [5, 2, 9]
    assert out.dtype == np.dtype(int)


def test_resolve_integer_indices(fake_results):
    out = _resolve_explicit_genes([3, 0, 7], fake_results)
    assert out.tolist() == [3, 0, 7]


def test_resolve_mixed_names_and_indices(fake_results):
    out = _resolve_explicit_genes(["Gene_4", 1, "Gene_8"], fake_results)
    assert out.tolist() == [4, 1, 8]


def test_unknown_name_raises(fake_results):
    with pytest.raises(ValueError, match="not found"):
        _resolve_explicit_genes(["Gene_4", "NoSuchGene"], fake_results)


def test_out_of_range_index_raises(fake_results):
    with pytest.raises(ValueError, match="out of range"):
        _resolve_explicit_genes([0, 99], fake_results)


def test_name_without_gene_names_raises():
    results = _FakeResults(n_genes=10, gene_names=None)
    with pytest.raises(ValueError, match="no gene names"):
        _resolve_explicit_genes(["Gene_3"], results)


def test_integer_index_without_gene_names_ok():
    # Integer indices do not need names to resolve.
    results = _FakeResults(n_genes=10, gene_names=None)
    out = _resolve_explicit_genes([2, 5], results, counts=np.zeros((4, 10)))
    assert out.tolist() == [2, 5]


def test_empty_selection_raises(fake_results):
    with pytest.raises(ValueError, match="at least one gene"):
        _resolve_explicit_genes([], fake_results)


def test_bool_is_not_treated_as_index(fake_results):
    # A bare bool must not silently resolve as int(0)/int(1); it is coerced
    # to a name lookup ("True"/"False"), which is absent -> ValueError.
    with pytest.raises(ValueError, match="not found"):
        _resolve_explicit_genes([True], fake_results)
