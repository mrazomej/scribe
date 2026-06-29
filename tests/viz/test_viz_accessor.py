"""Tests for the object-oriented ``results.viz`` accessor.

These exercise :mod:`scribe.viz.registry` end-to-end: automatic registration of
results-aware plots (via both the ``@plot_function`` decorator and the explicit
``register_plot`` helper), type-aware routing, argument forwarding, and the
equivalence between the functional and object-oriented entry points.

The tests stay lightweight and deterministic: results instances are built with
``object.__new__`` (no heavy fit/init), and the bound plotting callable is
monkeypatched so no real matplotlib rendering runs.
"""

from __future__ import annotations

import inspect

import pytest

import scribe
from scribe.viz.registry import (
    VIZ_REGISTRY,
    VizAccessor,
    VizEntry,
    _resolve,
    register_plot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Concrete results classes that belong to the "inference" supports family.
INFERENCE_CLASSES = [
    scribe.ScribeSVIResults,
    scribe.ScribeMCMCResults,
    scribe.ScribeLaplaceResults,
]


def _blank(cls):
    """Build a results instance bypassing its (heavy) dataclass ``__init__``.

    The accessor only inspects ``type(results)`` and never touches fitted
    fields, so an uninitialized instance is sufficient and fast.
    """
    return object.__new__(cls)


def _ensure_imported(name: str):
    """Force ``scribe.viz``'s lazy import of ``name`` (runs registration)."""
    return getattr(scribe.viz, name)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_decorated_plot_auto_registers():
    """A ``@plot_function`` plot registers itself on first lazy import."""
    _ensure_imported("plot_ppc")
    entry = VIZ_REGISTRY.get("plot_ppc")
    assert entry is not None
    assert entry.results_param == "results"
    # Default supports => the inference family (no DE).
    assert entry.supports is None


def test_de_plots_register_with_de_support():
    """DE plots register with ``supports="de"`` and a ``de_results`` param."""
    for name in (
        "plot_de_volcano",
        "plot_de_ma",
        "plot_de_evidence",
        "plot_de_mean_expression",
        "plot_de_mask_threshold",
    ):
        _ensure_imported(name)
        entry = VIZ_REGISTRY.get(name)
        assert entry is not None, name
        assert entry.supports == "de", name
        assert entry.results_param == "de_results", name


def test_undecorated_mixture_wrappers_register():
    """Hand-written mixture plotters register via ``register_plot``."""
    _ensure_imported("plot_mixture_ppc_overview")
    for name in (
        "plot_mixture_ppc",
        "plot_mixture_ppc_overview",
        "plot_mixture_ppc_components",
        "plot_mixture_ppc_comparison",
    ):
        assert name in VIZ_REGISTRY, name


def test_non_results_plot_excluded():
    """``plot_ecdf`` (counts-first) opts out via ``accessor=False``."""
    _ensure_imported("plot_ecdf")
    assert "plot_ecdf" not in VIZ_REGISTRY


def test_register_plot_helper_bare_and_parameterized():
    """``register_plot`` works both bare and parameterized; returns func."""

    @register_plot
    def _demo_plot(results, value=1):
        return value

    @register_plot(supports="de")
    def _demo_de_plot(de_results, value=2):
        return value

    try:
        assert VIZ_REGISTRY["_demo_plot"].supports is None
        assert VIZ_REGISTRY["_demo_plot"].results_param == "results"
        assert VIZ_REGISTRY["_demo_de_plot"].supports == "de"
        assert VIZ_REGISTRY["_demo_de_plot"].results_param == "de_results"
        # Decorator is transparent: the function is returned unchanged.
        assert _demo_plot(None, value=7) == 7
    finally:
        VIZ_REGISTRY.pop("_demo_plot", None)
        VIZ_REGISTRY.pop("_demo_de_plot", None)


# ---------------------------------------------------------------------------
# Accessor behavior
# ---------------------------------------------------------------------------


def test_viz_property_returns_accessor():
    """``results.viz`` is a fresh :class:`VizAccessor` bound to the object."""
    svi = _blank(scribe.ScribeSVIResults)
    acc = svi.viz
    assert isinstance(acc, VizAccessor)
    assert acc._results is svi


def test_accessor_forwards_results_as_first_arg(monkeypatch):
    """``viz.plot_ppc(counts, **kw)`` -> ``func(results, counts, **kw)``."""
    _ensure_imported("plot_ppc")
    svi = _blank(scribe.ScribeSVIResults)
    captured = {}

    def _spy(results, counts, **kwargs):
        captured.update(results=results, counts=counts, kwargs=kwargs)
        return "RESULT"

    monkeypatch.setattr(VIZ_REGISTRY["plot_ppc"], "func", _spy)

    out = svi.viz.plot_ppc("COUNTS", n_genes=5)
    assert out == "RESULT"
    assert captured["results"] is svi
    assert captured["counts"] == "COUNTS"
    assert captured["kwargs"] == {"n_genes": 5}


@pytest.mark.parametrize("cls", INFERENCE_CLASSES)
def test_inference_results_expose_inference_plots(cls):
    """SVI/MCMC/Laplace expose default ("inference") plots but not DE plots."""
    obj = _blank(cls)
    names = set(dir(obj.viz))
    assert "plot_ppc" in names
    assert "plot_de_volcano" not in names
    assert "plot_ecdf" not in names


def test_de_results_expose_only_de_plots():
    """DE results expose DE plots and reject inference-only plots."""
    de = _blank(scribe.de.ScribeParametricDEResults)
    names = set(dir(de.viz))
    assert "plot_de_volcano" in names
    assert "plot_ppc" not in names


def test_unsupported_plot_raises_informative_error():
    """An out-of-family plot raises ``AttributeError`` naming the type."""
    svi = _blank(scribe.ScribeSVIResults)
    with pytest.raises(AttributeError) as exc:
        svi.viz.plot_de_volcano
    msg = str(exc.value)
    assert "plot_de_volcano" in msg and "ScribeSVIResults" in msg


def test_non_results_bound_plot_raises():
    """A real ``scribe.viz`` function that isn't results-bound is rejected."""
    svi = _blank(scribe.ScribeSVIResults)
    with pytest.raises(AttributeError) as exc:
        svi.viz.plot_ecdf
    assert "results-bound" in str(exc.value)


def test_unknown_plot_raises():
    """An unknown attribute raises ``AttributeError`` (no recursion)."""
    svi = _blank(scribe.ScribeSVIResults)
    with pytest.raises(AttributeError):
        svi.viz.definitely_not_a_plot


def test_repr_lists_available_plots():
    """``repr(results.viz)`` names the results type and its plots."""
    svi = _blank(scribe.ScribeSVIResults)
    text = repr(svi.viz)
    assert "ScribeSVIResults" in text
    assert "plot_ppc" in text


# ---------------------------------------------------------------------------
# Equivalence with the functional API
# ---------------------------------------------------------------------------


def test_functional_signature_unchanged():
    """Functional ``scribe.viz.plot_ppc`` still leads with results/counts."""
    params = list(inspect.signature(scribe.viz.plot_ppc).parameters)
    assert params[0] == "results"
    assert params[1] == "counts"


def test_bound_signature_strips_results():
    """The bound accessor method hides the results parameter from help()."""
    svi = _blank(scribe.ScribeSVIResults)
    params = list(inspect.signature(svi.viz.plot_ppc).parameters)
    assert params[0] == "counts"
    assert "results" not in params


# ---------------------------------------------------------------------------
# Category resolution
# ---------------------------------------------------------------------------


def test_resolve_categories():
    """``_resolve`` maps category keys (and None) to results classes."""
    inference = _resolve(None)
    assert scribe.ScribeMCMCResults in inference
    assert scribe.ScribeLaplaceResults in inference
    # The variational ABC covers both SVI and VAE concrete classes.
    assert issubclass(scribe.ScribeSVIResults, inference)
    de = _resolve("de")
    assert scribe.ScribeDEResults in de
    assert scribe.ScribeDEResults not in inference
