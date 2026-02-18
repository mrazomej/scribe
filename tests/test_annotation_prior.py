# tests/test_annotation_prior.py
"""
Tests for the annotation prior functionality.

Covers:
- Unit tests for ``build_annotation_prior_logits``
- Unit tests for ``validate_annotation_prior_logits``
- Unit tests for ``compute_cell_specific_mixing``
- Model trace tests (NumPyro handlers)
- SVI smoke tests
- Behavioral tests (annotation influence on posterior)
"""

import numpy as np
import pandas as pd
import pytest
import jax
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.handlers as handlers

from scribe.core.annotation_prior import (
    COMPOSITE_LABEL_SEP,
    build_annotation_prior_logits,
    validate_annotation_prior_logits,
    _resolve_composite_annotations,
)
from scribe.models.components.likelihoods.base import (
    compute_cell_specific_mixing,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def simple_adata():
    """Create a simple AnnData with cell type annotations."""
    import anndata

    n_cells, n_genes = 20, 5
    rng = np.random.default_rng(42)
    X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
    labels = ["A"] * 8 + ["B"] * 7 + [np.nan] * 5
    obs = pd.DataFrame({"cell_type": labels})
    return anndata.AnnData(X=X, obs=obs)


@pytest.fixture
def adata_three_types():
    """AnnData with three cell types and no missing labels."""
    import anndata

    n_cells, n_genes = 30, 5
    rng = np.random.default_rng(123)
    X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
    labels = ["T_cell"] * 10 + ["B_cell"] * 10 + ["Mono"] * 10
    obs = pd.DataFrame({"cell_type": labels})
    return anndata.AnnData(X=X, obs=obs)


@pytest.fixture
def adata_multi_column():
    """AnnData with two annotation columns (cell_type x treatment)."""
    import anndata

    n_genes = 5
    rng = np.random.default_rng(77)
    # 8 cells: 2 cell types x 2 treatments = 4 composite labels, + 2 with NaN
    cell_types = ["T", "T", "B", "B", "T", "B", np.nan, "T"]
    treatments = [
        "ctrl",
        "stim",
        "ctrl",
        "stim",
        "ctrl",
        "stim",
        "ctrl",
        np.nan,
    ]
    X = rng.poisson(5, (len(cell_types), n_genes)).astype(np.float32)
    obs = pd.DataFrame({"cell_type": cell_types, "treatment": treatments})
    return anndata.AnnData(X=X, obs=obs)


@pytest.fixture
def adata_three_columns():
    """AnnData with three annotation columns."""
    import anndata

    n_genes = 3
    rng = np.random.default_rng(99)
    ct = ["T", "T", "B", "B"]
    tx = ["ctrl", "stim", "ctrl", "stim"]
    batch = ["1", "1", "2", "2"]
    X = rng.poisson(5, (4, n_genes)).astype(np.float32)
    obs = pd.DataFrame({"cell_type": ct, "treatment": tx, "batch": batch})
    return anndata.AnnData(X=X, obs=obs)


# ==============================================================================
# Tests for build_annotation_prior_logits
# ==============================================================================


class TestBuildAnnotationPriorLogits:
    """Unit tests for build_annotation_prior_logits."""

    def test_basic_shape(self, simple_adata):
        """Output has shape (n_cells, n_components)."""
        logits, label_map = build_annotation_prior_logits(
            simple_adata, "cell_type", n_components=3, confidence=3.0
        )
        assert logits.shape == (20, 3)

    def test_label_map_alphabetical(self, simple_adata):
        """Labels are mapped alphabetically by default."""
        _, label_map = build_annotation_prior_logits(
            simple_adata, "cell_type", n_components=3, confidence=3.0
        )
        assert label_map == {"A": 0, "B": 1}

    def test_unlabeled_cells_zero_logits(self, simple_adata):
        """Cells with NaN annotation get all-zero logit rows."""
        logits, _ = build_annotation_prior_logits(
            simple_adata, "cell_type", n_components=3, confidence=3.0
        )
        # Last 5 cells have NaN labels
        unlabeled_logits = logits[15:, :]
        assert jnp.allclose(unlabeled_logits, 0.0)

    def test_labeled_cells_have_confidence(self, simple_adata):
        """Labeled cells have exactly confidence in the annotated component."""
        confidence = 5.0
        logits, label_map = build_annotation_prior_logits(
            simple_adata, "cell_type", n_components=3, confidence=confidence
        )
        # First 8 cells are "A" -> component 0
        for i in range(8):
            assert float(logits[i, label_map["A"]]) == pytest.approx(confidence)
            # Other components should be zero
            for k in range(3):
                if k != label_map["A"]:
                    assert float(logits[i, k]) == pytest.approx(0.0)

    def test_component_order(self, simple_adata):
        """component_order controls label-to-component mapping."""
        logits, label_map = build_annotation_prior_logits(
            simple_adata,
            "cell_type",
            n_components=3,
            confidence=3.0,
            component_order=["B", "A", "C"],
        )
        assert label_map == {"B": 0, "A": 1, "C": 2}
        # First cell is "A" -> should be in component 1
        assert float(logits[0, 1]) == pytest.approx(3.0)
        assert float(logits[0, 0]) == pytest.approx(0.0)

    def test_confidence_zero(self, simple_adata):
        """confidence=0 produces all-zero logits."""
        logits, _ = build_annotation_prior_logits(
            simple_adata, "cell_type", n_components=3, confidence=0.0
        )
        assert jnp.allclose(logits, 0.0)

    def test_three_types_all_labeled(self, adata_three_types):
        """All cells labeled with 3 types, n_components=3."""
        logits, label_map = build_annotation_prior_logits(
            adata_three_types, "cell_type", n_components=3, confidence=2.0
        )
        assert logits.shape == (30, 3)
        assert len(label_map) == 3
        # Each row should have exactly one entry equal to 2.0
        row_sums = jnp.sum(logits, axis=1)
        assert jnp.allclose(row_sums, 2.0)

    # --- Validation error tests ---

    def test_missing_obs_key(self, simple_adata):
        """Raises ValueError if obs_key is not in adata.obs."""
        with pytest.raises(ValueError, match="not found in adata.obs"):
            build_annotation_prior_logits(
                simple_adata, "nonexistent", n_components=3
            )

    def test_negative_confidence(self, simple_adata):
        """Raises ValueError if confidence < 0."""
        with pytest.raises(ValueError, match="confidence must be >= 0"):
            build_annotation_prior_logits(
                simple_adata, "cell_type", n_components=3, confidence=-1.0
            )

    def test_too_many_labels(self, adata_three_types):
        """Raises ValueError if unique labels > n_components."""
        with pytest.raises(ValueError, match="exceeds n_components"):
            build_annotation_prior_logits(
                adata_three_types, "cell_type", n_components=2, confidence=3.0
            )

    def test_component_order_missing_label(self, simple_adata):
        """Raises ValueError if component_order doesn't cover all labels."""
        with pytest.raises(ValueError, match="not in component_order"):
            build_annotation_prior_logits(
                simple_adata,
                "cell_type",
                n_components=3,
                confidence=3.0,
                component_order=["A", "C", "D"],  # Missing "B"
            )

    # --- min_cells filtering tests ---

    def test_min_cells_filters_rare_labels(self):
        """Labels with fewer than min_cells are treated as unlabeled."""
        import anndata

        n_genes = 5
        rng = np.random.default_rng(42)
        # 10 "A" cells, 3 "B" cells, 2 NaN
        labels = ["A"] * 10 + ["B"] * 3 + [np.nan, np.nan]
        X = rng.poisson(5, (len(labels), n_genes)).astype(np.float32)
        adata = anndata.AnnData(X=X, obs=pd.DataFrame({"ct": labels}))

        logits, label_map = build_annotation_prior_logits(
            adata, "ct", n_components=2, confidence=3.0, min_cells=5
        )

        # "B" has only 3 cells (< 5), should be excluded
        assert "B" not in label_map
        assert "A" in label_map
        assert len(label_map) == 1

        # "B" cells (indices 10-12) should have zero logits
        for i in range(10, 13):
            assert jnp.allclose(logits[i], 0.0)

        # "A" cells should still have confidence
        for i in range(10):
            assert float(logits[i, label_map["A"]]) == pytest.approx(3.0)

    def test_min_cells_zero_no_filtering(self, simple_adata):
        """min_cells=0 should not filter any labels (default behavior)."""
        logits_default, map_default = build_annotation_prior_logits(
            simple_adata, "cell_type", n_components=3, confidence=3.0
        )
        logits_zero, map_zero = build_annotation_prior_logits(
            simple_adata,
            "cell_type",
            n_components=3,
            confidence=3.0,
            min_cells=0,
        )
        assert map_default == map_zero
        assert jnp.allclose(logits_default, logits_zero)

    def test_min_cells_all_labels_pass(self, adata_three_types):
        """When all labels meet the threshold, nothing is filtered."""
        # Each type has 10 cells; min_cells=5 should keep all
        logits, label_map = build_annotation_prior_logits(
            adata_three_types,
            "cell_type",
            n_components=3,
            confidence=2.0,
            min_cells=5,
        )
        assert len(label_map) == 3

    def test_min_cells_all_labels_filtered(self):
        """When all labels are below threshold, all cells are unlabeled."""
        import anndata

        n_genes = 3
        rng = np.random.default_rng(42)
        labels = ["A"] * 2 + ["B"] * 2
        X = rng.poisson(5, (4, n_genes)).astype(np.float32)
        adata = anndata.AnnData(X=X, obs=pd.DataFrame({"ct": labels}))

        logits, label_map = build_annotation_prior_logits(
            adata, "ct", n_components=2, confidence=3.0, min_cells=5
        )
        assert len(label_map) == 0
        assert jnp.allclose(logits, 0.0)

    def test_min_cells_with_composite_labels(self):
        """min_cells filtering works with multi-column composite labels."""
        import anndata

        n_genes = 3
        rng = np.random.default_rng(42)
        # 6 cells: T__ctrl (3), T__stim (1), B__ctrl (2)
        ct = ["T", "T", "T", "T", "B", "B"]
        tx = ["ctrl", "ctrl", "ctrl", "stim", "ctrl", "ctrl"]
        X = rng.poisson(5, (6, n_genes)).astype(np.float32)
        adata = anndata.AnnData(
            X=X, obs=pd.DataFrame({"ct": ct, "tx": tx})
        )

        logits, label_map = build_annotation_prior_logits(
            adata,
            ["ct", "tx"],
            n_components=3,
            confidence=3.0,
            min_cells=2,
        )

        # T__stim has only 1 cell (< 2), should be excluded
        assert "T__stim" not in label_map
        assert "T__ctrl" in label_map
        assert "B__ctrl" in label_map
        assert len(label_map) == 2

        # Cell 3 (T__stim) should have zero logits
        assert jnp.allclose(logits[3], 0.0)

    def test_min_cells_logs_warning(self, caplog):
        """A warning is logged listing the filtered labels."""
        import anndata
        import logging

        n_genes = 3
        rng = np.random.default_rng(42)
        labels = ["A"] * 10 + ["B"] * 2
        X = rng.poisson(5, (12, n_genes)).astype(np.float32)
        adata = anndata.AnnData(X=X, obs=pd.DataFrame({"ct": labels}))

        with caplog.at_level(logging.WARNING, logger="scribe.core.annotation_prior"):
            build_annotation_prior_logits(
                adata, "ct", n_components=2, confidence=3.0, min_cells=5
            )

        assert "fewer than 5 cells" in caplog.text
        assert "'B'" in caplog.text or "B" in caplog.text


# ==============================================================================
# Tests for multi-column (composite) annotation priors
# ==============================================================================


class TestMultiColumnAnnotationPrior:
    """Unit tests for multi-column annotation keys."""

    def test_two_columns_composite_labels(self, adata_multi_column):
        """Two columns produce composite labels joined with '__'."""
        logits, label_map = build_annotation_prior_logits(
            adata_multi_column,
            ["cell_type", "treatment"],
            n_components=6,
            confidence=3.0,
        )
        # 4 unique composite labels: B__ctrl, B__stim, T__ctrl, T__stim
        assert len(label_map) == 4
        assert "T__ctrl" in label_map
        assert "T__stim" in label_map
        assert "B__ctrl" in label_map
        assert "B__stim" in label_map
        assert logits.shape == (8, 6)

    def test_two_columns_correct_shape(self, adata_multi_column):
        """Output shape is (n_cells, n_components)."""
        logits, _ = build_annotation_prior_logits(
            adata_multi_column,
            ["cell_type", "treatment"],
            n_components=4,
            confidence=2.0,
        )
        assert logits.shape == (8, 4)

    def test_two_columns_nan_propagation(self, adata_multi_column):
        """Cells with NaN in any column get zero logits."""
        logits, _ = build_annotation_prior_logits(
            adata_multi_column,
            ["cell_type", "treatment"],
            n_components=4,
            confidence=3.0,
        )
        # Cell 6: cell_type=NaN -> unlabeled
        assert jnp.allclose(logits[6], 0.0)
        # Cell 7: treatment=NaN -> unlabeled
        assert jnp.allclose(logits[7], 0.0)

    def test_two_columns_labeled_cells_have_confidence(
        self, adata_multi_column
    ):
        """Labeled cells get exactly confidence in the correct component."""
        confidence = 5.0
        logits, label_map = build_annotation_prior_logits(
            adata_multi_column,
            ["cell_type", "treatment"],
            n_components=4,
            confidence=confidence,
        )
        # Cell 0: cell_type="T", treatment="ctrl" -> "T__ctrl"
        comp_idx = label_map["T__ctrl"]
        assert float(logits[0, comp_idx]) == pytest.approx(confidence)
        # Non-annotated components should be zero
        for k in range(4):
            if k != comp_idx:
                assert float(logits[0, k]) == pytest.approx(0.0)

    def test_two_columns_component_order(self, adata_multi_column):
        """component_order controls mapping for composite labels."""
        order = ["T__stim", "T__ctrl", "B__stim", "B__ctrl"]
        logits, label_map = build_annotation_prior_logits(
            adata_multi_column,
            ["cell_type", "treatment"],
            n_components=4,
            confidence=3.0,
            component_order=order,
        )
        assert label_map == {
            "T__stim": 0,
            "T__ctrl": 1,
            "B__stim": 2,
            "B__ctrl": 3,
        }
        # Cell 0 is T, ctrl -> index 1
        assert float(logits[0, 1]) == pytest.approx(3.0)

    def test_three_columns(self, adata_three_columns):
        """Three columns produce three-part composite labels."""
        logits, label_map = build_annotation_prior_logits(
            adata_three_columns,
            ["cell_type", "treatment", "batch"],
            n_components=4,
            confidence=2.0,
        )
        assert logits.shape == (4, 4)
        assert len(label_map) == 4
        # Cell 0: T, ctrl, 1 -> "T__ctrl__1"
        assert "T__ctrl__1" in label_map
        assert "B__stim__2" in label_map

    def test_single_key_in_list(self, simple_adata):
        """Passing a single key as a list behaves like a plain string."""
        logits_str, map_str = build_annotation_prior_logits(
            simple_adata, "cell_type", n_components=3, confidence=3.0
        )
        logits_list, map_list = build_annotation_prior_logits(
            simple_adata, ["cell_type"], n_components=3, confidence=3.0
        )
        assert map_str == map_list
        assert jnp.allclose(logits_str, logits_list)

    def test_missing_column_in_multi_key(self, adata_multi_column):
        """Raises ValueError if any column in the list is missing."""
        with pytest.raises(ValueError, match="not found in adata.obs"):
            build_annotation_prior_logits(
                adata_multi_column,
                ["cell_type", "nonexistent"],
                n_components=4,
            )

    def test_empty_key_list(self, simple_adata):
        """Raises ValueError for empty list."""
        with pytest.raises(ValueError, match="non-empty"):
            build_annotation_prior_logits(simple_adata, [], n_components=3)

    def test_composite_too_many_labels(self, adata_multi_column):
        """Raises ValueError if composite labels exceed n_components."""
        with pytest.raises(ValueError, match="exceeds n_components"):
            build_annotation_prior_logits(
                adata_multi_column,
                ["cell_type", "treatment"],
                n_components=3,  # 4 unique composites > 3
                confidence=3.0,
            )

    def test_resolve_composite_annotations_directly(self, adata_multi_column):
        """The internal helper builds correct composite Series."""
        composite = _resolve_composite_annotations(
            adata_multi_column, ["cell_type", "treatment"]
        )
        assert composite.iloc[0] == "T__ctrl"
        assert composite.iloc[1] == "T__stim"
        assert composite.iloc[2] == "B__ctrl"
        assert composite.iloc[3] == "B__stim"
        # NaN cells
        assert pd.isna(composite.iloc[6])
        assert pd.isna(composite.iloc[7])

    def test_fit_api_with_multi_key(self):
        """Run scribe.fit() with annotation_key as a list of columns."""
        import anndata
        import scribe

        n_cells, n_genes = 8, 5
        rng = np.random.default_rng(42)
        X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
        # 2 cell types x 2 batches = 4 composites, but we allow extra
        # components by setting n_components >= n_unique_labels.
        # n_components=4 with default (1.0, 1.0) Dirichlet prior works.
        obs = pd.DataFrame(
            {
                "cell_type": ["A", "A", "B", "B"] * 2,
                "batch": ["1", "1", "1", "1"] * 2,
            }
        )
        adata = anndata.AnnData(X=X, obs=obs)

        # 2 unique composites: A__1, B__1 -> n_components=2 is sufficient
        result = scribe.fit(
            adata,
            model="nbdm",
            n_components=2,
            n_steps=3,
            batch_size=4,
            annotation_key=["cell_type", "batch"],
            annotation_confidence=3.0,
            seed=42,
        )

        assert result.n_cells == n_cells
        assert result.n_components == 2


# ==============================================================================
# Tests for validate_annotation_prior_logits
# ==============================================================================


class TestValidateAnnotationPriorLogits:
    """Unit tests for validate_annotation_prior_logits."""

    def test_valid_logits(self):
        """No error for correctly shaped, finite logits."""
        logits = jnp.zeros((10, 3))
        validate_annotation_prior_logits(logits, 10, 3)  # Should not raise

    def test_wrong_shape(self):
        """Raises ValueError for wrong shape."""
        logits = jnp.zeros((10, 3))
        with pytest.raises(ValueError, match="expected"):
            validate_annotation_prior_logits(logits, 10, 4)

    def test_non_finite(self):
        """Raises ValueError for non-finite values."""
        logits = jnp.array([[0.0, float("inf")], [0.0, 0.0]])
        with pytest.raises(ValueError, match="non-finite"):
            validate_annotation_prior_logits(logits, 2, 2)


# ==============================================================================
# Tests for compute_cell_specific_mixing
# ==============================================================================


class TestComputeCellSpecificMixing:
    """Unit tests for compute_cell_specific_mixing."""

    def test_zero_logits_recover_global(self):
        """Zero annotation logits should recover the global mixing weights."""
        mixing = jnp.array([0.3, 0.5, 0.2])
        logits = jnp.zeros((4, 3))
        result = compute_cell_specific_mixing(mixing, logits)

        # Each row should approximate the original mixing weights
        for i in range(4):
            assert jnp.allclose(result[i], mixing, atol=1e-5)

    def test_output_sums_to_one(self):
        """Each row of output should sum to 1."""
        mixing = jnp.array([0.25, 0.25, 0.25, 0.25])
        logits = jnp.array([[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]])
        result = compute_cell_specific_mixing(mixing, logits)

        row_sums = jnp.sum(result, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_large_logit_concentrates_mass(self):
        """Large positive logit should put most mass on that component."""
        mixing = jnp.array([0.5, 0.5])
        logits = jnp.array([[100.0, 0.0]])
        result = compute_cell_specific_mixing(mixing, logits)

        assert float(result[0, 0]) > 0.99
        assert float(result[0, 1]) < 0.01

    def test_shape_preservation(self):
        """Output shape matches (batch, K)."""
        mixing = jnp.array([0.4, 0.6])
        logits = jnp.zeros((7, 2))
        result = compute_cell_specific_mixing(mixing, logits)
        assert result.shape == (7, 2)


# ==============================================================================
# Model trace tests (using numpyro handlers)
# ==============================================================================


class TestModelTrace:
    """Test that the model function works correctly with annotation priors."""

    def _get_mixture_model_and_guide(self):
        """Build a simple 2-component NBDM mixture model and guide."""
        from scribe.inference.preset_builder import build_config_from_preset
        from scribe.models.model_registry import get_model_and_guide

        model_config = build_config_from_preset(
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            n_components=2,
            priors={"p": (1, 1), "r": (2, 0.1), "mixing": (1.0, 1.0)},
        )
        model, guide, mc = get_model_and_guide(model_config)
        return model, guide, model_config

    def test_model_trace_without_annotation(self):
        """Model produces valid trace without annotation_prior_logits."""
        model, _, model_config = self._get_mixture_model_and_guide()
        n_cells, n_genes = 10, 5

        with handlers.seed(rng_seed=0):
            tr = handlers.trace(model).get_trace(
                n_cells=n_cells,
                n_genes=n_genes,
                model_config=model_config,
                counts=None,
                batch_size=None,
                annotation_prior_logits=None,
            )

        assert "mixing_weights" in tr
        assert "counts" in tr
        assert tr["counts"]["value"].shape == (n_cells, n_genes)

    def test_model_trace_with_annotation(self):
        """Model produces valid trace with annotation_prior_logits."""
        model, _, model_config = self._get_mixture_model_and_guide()
        n_cells, n_genes = 10, 5

        # Build annotation logits: first 5 cells -> component 0, rest -> 1
        ann_logits = jnp.zeros((n_cells, 2))
        ann_logits = ann_logits.at[:5, 0].set(3.0)
        ann_logits = ann_logits.at[5:, 1].set(3.0)

        with handlers.seed(rng_seed=0):
            tr = handlers.trace(model).get_trace(
                n_cells=n_cells,
                n_genes=n_genes,
                model_config=model_config,
                counts=None,
                batch_size=None,
                annotation_prior_logits=ann_logits,
            )

        assert "mixing_weights" in tr
        assert "counts" in tr
        counts = tr["counts"]["value"]
        assert counts.shape == (n_cells, n_genes)
        assert jnp.all(jnp.isfinite(counts))

    def test_model_trace_with_observed_data_and_annotation(self):
        """Model runs with observed counts and annotation priors."""
        model, _, model_config = self._get_mixture_model_and_guide()
        n_cells, n_genes = 10, 5
        rng = np.random.default_rng(42)
        counts = jnp.array(
            rng.poisson(5, (n_cells, n_genes)), dtype=jnp.float32
        )

        ann_logits = jnp.zeros((n_cells, 2))
        ann_logits = ann_logits.at[:5, 0].set(3.0)

        with handlers.seed(rng_seed=0):
            tr = handlers.trace(model).get_trace(
                n_cells=n_cells,
                n_genes=n_genes,
                model_config=model_config,
                counts=counts,
                batch_size=None,
                annotation_prior_logits=ann_logits,
            )

        assert "counts" in tr
        # The log probability should be finite
        assert jnp.all(jnp.isfinite(tr["counts"]["fn"].log_prob(counts)))


# ==============================================================================
# SVI smoke tests
# ==============================================================================


class TestSVISmoke:
    """Smoke tests: SVI runs without errors when annotation priors are used."""

    def test_svi_with_annotation_logits(self):
        """Run 3 steps of SVI on NBDM mixture with annotation_prior_logits."""
        from scribe.inference import run_scribe
        from scribe.inference.preset_builder import build_config_from_preset
        from scribe.models.config import InferenceConfig, SVIConfig

        n_cells, n_genes = 10, 5
        rng = np.random.default_rng(42)
        counts = jnp.array(
            rng.poisson(5, (n_cells, n_genes)), dtype=jnp.float32
        )

        model_config = build_config_from_preset(
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            n_components=2,
            priors={"p": (1, 1), "r": (2, 0.1), "mixing": (1.0, 1.0)},
        )
        svi_config = SVIConfig(n_steps=3, batch_size=5)
        inference_config = InferenceConfig.from_svi(svi_config)

        # Build annotation logits manually
        ann_logits = jnp.zeros((n_cells, 2))
        ann_logits = ann_logits.at[:5, 0].set(3.0)
        ann_logits = ann_logits.at[5:, 1].set(3.0)

        result = run_scribe(
            counts=counts,
            model_config=model_config,
            inference_config=inference_config,
            seed=42,
            annotation_prior_logits=ann_logits,
        )

        assert result.n_cells == n_cells
        assert result.n_genes == n_genes
        assert result.n_components == 2
        assert hasattr(result, "loss_history")
        assert len(result.loss_history) > 0

    def test_fit_api_with_annotation_key(self):
        """Run scribe.fit() with annotation_key on AnnData."""
        import anndata
        import scribe

        n_cells, n_genes = 10, 5
        rng = np.random.default_rng(42)
        X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
        labels = ["A"] * 5 + ["B"] * 5
        adata = anndata.AnnData(X=X, obs=pd.DataFrame({"cell_type": labels}))

        result = scribe.fit(
            adata,
            model="nbdm",
            n_components=2,
            n_steps=3,
            batch_size=5,
            annotation_key="cell_type",
            annotation_confidence=3.0,
            seed=42,
        )

        assert result.n_cells == n_cells
        assert result.n_genes == n_genes
        assert result.n_components == 2

    def test_fit_api_coerces_oversized_batch_size_to_full_batch(self):
        """
        Oversized batch_size is coerced to full-batch mode.

        The fit API should preserve bounded mini-batching when possible, but
        if ``batch_size`` exceeds ``n_cells`` it should transparently pass
        ``None`` to the SVI configuration and emit a user-facing warning.
        """
        import anndata
        import scribe

        n_cells, n_genes = 10, 5
        rng = np.random.default_rng(42)
        X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
        labels = ["A"] * 5 + ["B"] * 5
        adata = anndata.AnnData(X=X, obs=pd.DataFrame({"cell_type": labels}))

        with pytest.warns(
            UserWarning,
            match=r"batch_size=20 exceeds n_cells=10; using full-batch mode",
        ):
            result = scribe.fit(
                adata,
                model="nbdm",
                n_components=2,
                n_steps=3,
                batch_size=20,
                annotation_key="cell_type",
                annotation_confidence=3.0,
                seed=42,
            )

        assert result.n_cells == n_cells
        assert result.n_genes == n_genes
        assert result.n_components == 2

    def test_fit_api_infers_n_components_single_key(self):
        """n_components is auto-inferred from annotation labels when omitted."""
        import anndata
        import scribe

        n_cells, n_genes = 12, 5
        rng = np.random.default_rng(42)
        X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
        labels = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
        adata = anndata.AnnData(X=X, obs=pd.DataFrame({"ct": labels}))

        # Do NOT pass n_components — should be inferred as 3
        result = scribe.fit(
            adata,
            model="nbdm",
            n_steps=3,
            batch_size=6,
            annotation_key="ct",
            annotation_confidence=3.0,
            seed=42,
        )

        assert result.n_components == 3

    def test_fit_api_infers_n_components_multi_key(self):
        """n_components is auto-inferred from composite annotation labels."""
        import anndata
        import scribe

        n_cells, n_genes = 8, 5
        rng = np.random.default_rng(42)
        X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
        obs = pd.DataFrame(
            {
                "cell_type": ["A", "A", "B", "B"] * 2,
                "batch": ["x", "y", "x", "y"] * 2,
            }
        )
        adata = anndata.AnnData(X=X, obs=obs)

        # 4 unique composites: A__x, A__y, B__x, B__y -> n_components=4
        result = scribe.fit(
            adata,
            model="nbdm",
            n_steps=3,
            batch_size=4,
            annotation_key=["cell_type", "batch"],
            annotation_confidence=3.0,
            seed=42,
        )

        assert result.n_components == 4

    def test_fit_api_infers_n_components_with_nan(self):
        """n_components ignores NaN labels when auto-inferring."""
        import anndata
        import scribe

        n_cells, n_genes = 10, 5
        rng = np.random.default_rng(42)
        X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
        labels = ["A"] * 4 + ["B"] * 4 + [np.nan, np.nan]
        adata = anndata.AnnData(X=X, obs=pd.DataFrame({"ct": labels}))

        # 2 unique non-null labels → n_components=2
        result = scribe.fit(
            adata,
            model="nbdm",
            n_steps=3,
            batch_size=5,
            annotation_key="ct",
            annotation_confidence=3.0,
            seed=42,
        )

        assert result.n_components == 2

    def test_fit_api_annotation_min_cells(self):
        """annotation_min_cells filters rare labels and adjusts n_components."""
        import anndata
        import scribe

        n_genes = 5
        rng = np.random.default_rng(42)
        # 10 "A", 10 "B", 2 "C" (rare)
        labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 2
        X = rng.poisson(5, (22, n_genes)).astype(np.float32)
        adata = anndata.AnnData(X=X, obs=pd.DataFrame({"ct": labels}))

        # Without min_cells: 3 labels → n_components=3
        result_all = scribe.fit(
            adata,
            model="nbdm",
            n_steps=3,
            batch_size=11,
            annotation_key="ct",
            annotation_confidence=3.0,
            seed=42,
        )
        assert result_all.n_components == 3

        # With min_cells=5: "C" filtered → n_components=2
        result_filtered = scribe.fit(
            adata,
            model="nbdm",
            n_steps=3,
            batch_size=11,
            annotation_key="ct",
            annotation_confidence=3.0,
            annotation_min_cells=5,
            seed=42,
        )
        assert result_filtered.n_components == 2

    def test_svi_zinb_mixture_with_annotation(self):
        """Run SVI on ZINB mixture with annotation priors."""
        from scribe.inference import run_scribe
        from scribe.inference.preset_builder import build_config_from_preset
        from scribe.models.config import InferenceConfig, SVIConfig

        n_cells, n_genes = 10, 5
        rng = np.random.default_rng(42)
        counts = jnp.array(
            rng.poisson(5, (n_cells, n_genes)), dtype=jnp.float32
        )

        model_config = build_config_from_preset(
            model="zinb",
            parameterization="canonical",
            inference_method="svi",
            n_components=2,
        )
        svi_config = SVIConfig(n_steps=3, batch_size=5)
        inference_config = InferenceConfig.from_svi(svi_config)

        ann_logits = jnp.zeros((n_cells, 2))
        ann_logits = ann_logits.at[:5, 0].set(3.0)
        ann_logits = ann_logits.at[5:, 1].set(3.0)

        result = run_scribe(
            counts=counts,
            model_config=model_config,
            inference_config=inference_config,
            seed=42,
            annotation_prior_logits=ann_logits,
        )

        assert result.n_cells == n_cells
        assert result.n_components == 2


# ==============================================================================
# Behavioral tests
# ==============================================================================


class TestAnnotationBehavior:
    """
    Behavioral tests verifying that annotation priors actually influence
    the model.
    """

    def test_strong_annotation_influences_log_prob(self):
        """
        With strong annotation priors, the log probability of the observed
        data should differ from the case without annotation priors.
        """
        from scribe.inference.preset_builder import build_config_from_preset
        from scribe.models.model_registry import get_model_and_guide

        model_config = build_config_from_preset(
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            n_components=2,
            priors={"p": (1, 1), "r": (2, 0.1), "mixing": (1.0, 1.0)},
        )
        model, _, mc = get_model_and_guide(model_config)

        n_cells, n_genes = 10, 5
        rng = np.random.default_rng(42)
        counts = jnp.array(
            rng.poisson(5, (n_cells, n_genes)), dtype=jnp.float32
        )

        # Build strong annotation: all cells assigned to component 0
        strong_logits = jnp.zeros((n_cells, 2))
        strong_logits = strong_logits.at[:, 0].set(10.0)

        # Trace without annotation
        with handlers.seed(rng_seed=0):
            tr_no_ann = handlers.trace(model).get_trace(
                n_cells=n_cells,
                n_genes=n_genes,
                model_config=mc,
                counts=counts,
                batch_size=None,
                annotation_prior_logits=None,
            )

        # Trace with strong annotation
        with handlers.seed(rng_seed=0):
            tr_ann = handlers.trace(model).get_trace(
                n_cells=n_cells,
                n_genes=n_genes,
                model_config=mc,
                counts=counts,
                batch_size=None,
                annotation_prior_logits=strong_logits,
            )

        # The log probabilities of the counts site should differ
        log_prob_no_ann = tr_no_ann["counts"]["fn"].log_prob(counts)
        log_prob_ann = tr_ann["counts"]["fn"].log_prob(counts)

        # They should not be identical (annotation changes the mixing)
        assert not jnp.allclose(log_prob_no_ann, log_prob_ann, atol=1e-6)

    def test_annotation_vs_no_annotation_mixing_differ(self):
        """
        With annotation priors, the effective per-cell mixing should differ
        from the global mixing weights.
        """
        mixing_weights = jnp.array([0.5, 0.5])

        # No annotation: cell mixing == global mixing for all cells
        no_ann_logits = jnp.zeros((4, 2))
        result_no_ann = compute_cell_specific_mixing(
            mixing_weights, no_ann_logits
        )
        for i in range(4):
            assert jnp.allclose(result_no_ann[i], mixing_weights, atol=1e-5)

        # With annotation: cell 0 biased toward component 0
        ann_logits = jnp.zeros((4, 2))
        ann_logits = ann_logits.at[0, 0].set(5.0)
        result_ann = compute_cell_specific_mixing(mixing_weights, ann_logits)

        # Cell 0 should now have higher weight on component 0
        assert float(result_ann[0, 0]) > 0.9
        # Cell 1 should still be close to uniform
        assert jnp.allclose(result_ann[1], mixing_weights, atol=1e-5)
