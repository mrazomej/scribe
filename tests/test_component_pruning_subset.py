"""Regression tests for mixture-component posterior subsetting.

These tests protect the component-pruning workflow used by
``scribe.mc.compare_models(..., component_threshold=...)``.  In some
parameterizations, canonical posterior tensors (for example ``p`` and ``r``)
are present in ``posterior_samples`` but absent from ``model_config.param_specs``.

If those tensors are not subset along the selected component axis, the model
ends up with inconsistent component counts across parameters (for example
``mixing_weights`` pruned to K=3 but ``p``/``r`` still at K=4), which later
causes broadcasting failures in mixture log-likelihood evaluation.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from scribe.models.log_likelihood import nbvcp_mixture_log_likelihood
from scribe.svi._component import ComponentMixin


class _DummyComponentResult(ComponentMixin):
    """Small harness exposing ComponentMixin subsetting methods.

    Parameters
    ----------
    n_components : int, default=4
        Number of mixture components for synthetic posterior samples.
    """

    def __init__(
        self, n_components: int = 4, include_reference_specs: bool = True
    ):
        self.n_components = n_components
        # Canonical keys such as `p`/`r` may be absent from param_specs.
        # Include non-canonical reference specs (`phi`, `mu`) so fallback logic
        # can infer whether canonical tensors are truly mixture-specific.
        _param_specs = [SimpleNamespace(name="mixing_weights", is_mixture=True)]
        if include_reference_specs:
            _param_specs.extend(
                [
                    SimpleNamespace(name="phi", is_mixture=True),
                    SimpleNamespace(name="mu", is_mixture=True),
                ]
            )
        self.model_config = SimpleNamespace(param_specs=_param_specs)


def test_subset_posterior_samples_by_components_handles_canonical_keys():
    """Subset canonical mixture tensors across multiple selected components.

    Notes
    -----
    The `p`/`r` entries intentionally do not exist in param_specs.  The mixin
    must still detect their component axis and subset them consistently with
    `mixing_weights`.
    """

    _dummy = _DummyComponentResult(n_components=4)
    _samples = {
        # (S, K)
        "mixing_weights": jnp.array([[0.10, 0.20, 0.30, 0.40]]),
        # Non-canonical reference keys explicitly marked as mixture.
        "phi": jnp.full((1, 4), 0.5, dtype=jnp.float32),
        "mu": jnp.full((1, 4, 5), 2.0, dtype=jnp.float32),
        # (S, K, G) canonical mixture tensors
        "p": jnp.arange(1 * 4 * 5, dtype=jnp.float32).reshape(1, 4, 5),
        "r": (100.0 + jnp.arange(1 * 4 * 5, dtype=jnp.float32)).reshape(1, 4, 5),
    }
    _selected = jnp.array([0, 2, 3])

    _subset = _dummy._subset_posterior_samples_by_components(
        _samples, _selected, renormalize=False
    )

    assert _subset["mixing_weights"].shape == (1, 3)
    assert _subset["p"].shape == (1, 3, 5)
    assert _subset["r"].shape == (1, 3, 5)
    assert jnp.allclose(_subset["p"], _samples["p"][:, _selected, :])
    assert jnp.allclose(_subset["r"], _samples["r"][:, _selected, :])


def test_subset_posterior_samples_by_component_handles_canonical_keys():
    """Subset canonical mixture tensors when selecting a single component."""

    _dummy = _DummyComponentResult(n_components=4)
    _samples = {
        # (S, K)
        "mixing_weights": jnp.array([[0.10, 0.20, 0.30, 0.40]]),
        # Non-canonical reference keys explicitly marked as mixture.
        "phi": jnp.full((1, 4), 0.5, dtype=jnp.float32),
        "mu": jnp.full((1, 4, 5), 2.0, dtype=jnp.float32),
        # (S, K, G)
        "p": jnp.arange(1 * 4 * 5, dtype=jnp.float32).reshape(1, 4, 5),
        "r": (100.0 + jnp.arange(1 * 4 * 5, dtype=jnp.float32)).reshape(1, 4, 5),
    }
    _component_idx = 2

    _subset = _dummy._subset_posterior_samples_by_component(
        _samples, _component_idx
    )

    assert _subset["mixing_weights"].shape == (1,)
    assert _subset["p"].shape == (1, 5)
    assert _subset["r"].shape == (1, 5)
    assert jnp.allclose(_subset["p"], _samples["p"][:, _component_idx, :])
    assert jnp.allclose(_subset["r"], _samples["r"][:, _component_idx, :])


def test_nbvcp_mixture_raises_clear_error_on_component_mismatch():
    """Raise a clear error when component tensors are shape-inconsistent.

    Notes
    -----
    This test mimics a stale-canonical-tensor scenario after pruning:
    ``mixing_weights`` has ``K=3`` active components, while ``p`` still carries
    a leading component-like size of 4.  The guard should fail fast with a
    model-specific message instead of bubbling up a low-level JAX
    broadcasting traceback.
    """

    _counts = jnp.ones((8, 5), dtype=jnp.float32)
    _params = {
        "mixing_weights": jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32),
        # Intentionally incompatible with K=3.
        "p": jnp.full((4, 5), 0.2, dtype=jnp.float32),
        "r": jnp.full((3, 5), 1.0, dtype=jnp.float32),
        "p_capture": jnp.full((8,), 0.9, dtype=jnp.float32),
    }

    with pytest.raises(ValueError, match="nbvcp_mixture_log_likelihood"):
        nbvcp_mixture_log_likelihood(_counts, _params, batch_size=4)


def test_fallback_does_not_subset_shared_gene_specific_p_without_evidence():
    """Do not subset canonical `p` when no mixture evidence exists.

    Notes
    -----
    This guards against a subtle false-positive case where a shared gene axis
    accidentally equals `n_components`.  Name-only fallback logic would wrongly
    slice `p`; evidence-aware fallback must leave it untouched.
    """

    _dummy = _DummyComponentResult(
        n_components=4, include_reference_specs=False
    )
    _samples = {
        # Explicit mixture key (only one declared in this harness).
        "mixing_weights": jnp.array([[0.10, 0.20, 0.30, 0.40]]),
        # Gene-specific shared p: shape (S, G) with G == n_components by chance.
        "p": jnp.arange(1 * 4, dtype=jnp.float32).reshape(1, 4),
    }
    _selected = jnp.array([0, 2, 3])

    _subset = _dummy._subset_posterior_samples_by_components(
        _samples, _selected, renormalize=False
    )

    # mixing_weights is subset as expected.
    assert _subset["mixing_weights"].shape == (1, 3)
    # p must be preserved because there is no explicit mixture evidence for p/phi.
    assert _subset["p"].shape == (1, 4)
    assert jnp.allclose(_subset["p"], _samples["p"])
