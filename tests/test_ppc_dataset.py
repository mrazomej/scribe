"""Tests for ``plot_ppc(dataset=...)`` leaf resolution + cell restriction.

The per-leaf PPC keeps the **full** multi-dataset model and restricts the cells
to one leaf (``_dataset_cell_view``), rather than collapsing to a single-dataset
view via ``get_dataset`` — the latter does not reconstruct the leaf mean for an
additive multifactor fit when the PPC re-samples the guide. The integer-leaf and
error paths are unit-tested here against a fake results object; the
``{factor: level}`` dict-resolution path needs a real ``grouping_spec`` and is
exercised end-to-end on a real fit in the tutorial / manual verification.
"""

import copy

import numpy as np
import pytest

from scribe.viz.ppc import _dataset_cell_view, _subset_ppc_to_dataset


class _FakeResults:
    """Minimal stand-in for the bits the per-leaf PPC helpers touch."""

    def __init__(self, dataset_indices):
        self._dataset_indices = (
            None if dataset_indices is None else np.asarray(dataset_indices)
        )
        self.params = {"phi_capture_loc": np.zeros(0)}
        self.posterior_samples = "stored"
        self.predictive_samples = "stored"
        self.n_cells = (
            None if dataset_indices is None else len(dataset_indices)
        )

    def _subset_cell_specific_params(self, params, dataset_indices, leaf):
        # The real method slices per-cell params; passthrough is enough here.
        return params


def test_cell_view_keeps_full_model_and_restricts_cells():
    res = _FakeResults([0, 0, 1, 1, 2])
    view, mask = _dataset_cell_view(res, 1)
    assert view is not res  # a copy, not a mutation of the original
    assert view.n_cells == 2
    assert np.all(np.asarray(view._dataset_indices) == 1)  # leaf cells only
    assert view.posterior_samples is None  # forces a fresh draw from full guide
    np.testing.assert_array_equal(mask, [False, False, True, True, False])


def test_int_leaf_subsets_view_and_counts():
    res = _FakeResults([0, 0, 1, 1, 2])
    counts = np.arange(5 * 3).reshape(5, 3).astype(float)
    view, sub = _subset_ppc_to_dataset(res, counts, 1)
    assert view.n_cells == 2
    np.testing.assert_array_equal(sub, counts[[2, 3]])  # cells with index 1


def test_no_dataset_indices_raises():
    res = _FakeResults(None)
    with pytest.raises(ValueError, match="multi-dataset fit"):
        _subset_ppc_to_dataset(res, np.zeros((5, 3)), 0)


def test_wrong_length_counts_raises():
    res = _FakeResults([0, 0, 1, 1, 2])
    with pytest.raises(ValueError, match="cells but the fit has"):
        _subset_ppc_to_dataset(res, np.zeros((3, 3)), 0)


def test_empty_leaf_raises():
    res = _FakeResults([0, 0, 1, 1, 2])
    with pytest.raises(ValueError, match="no cells found"):
        _subset_ppc_to_dataset(res, np.zeros((5, 3)), 9)
