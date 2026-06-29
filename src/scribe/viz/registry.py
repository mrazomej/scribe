"""Registry powering the object-oriented ``.viz`` accessor on results objects.

SCRIBE plotting has two equivalent entry points:

* Functional: ``scribe.viz.plot_ppc(results, counts, ...)``.
* Object-oriented: ``results.viz.plot_ppc(counts, ...)`` (this module).

The object-oriented path is registry-driven. Every results-aware plotting
function registers itself here -- automatically through the
:func:`scribe.viz._interactive.plot_function` decorator, or explicitly via
:func:`register_plot` for the few hand-written public plotters that do not use
that decorator. The :class:`VizAccessor` returned by ``results.viz`` then
exposes exactly the registered plots whose ``supports`` category includes the
results object's type.

This module is intentionally dependency-light (standard library only) so that
results classes can import :class:`VizAccessorMixin` at definition time without
pulling in matplotlib or creating a viz<->results import cycle. The results
classes referenced by the ``supports`` categories are imported lazily, only
when a type check actually runs.
"""

from __future__ import annotations

import functools
import importlib
import inspect
from typing import Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# Supported-results categories
# ---------------------------------------------------------------------------

# Category key -> dotted paths of the results classes the category covers.
# Resolved lazily (and cached) so this module never imports the results
# classes at import time.
_CATEGORIES = {
    "inference": (
        "scribe.svi.variational_results_base.ScribeVariationalResults",
        "scribe.mcmc.results.ScribeMCMCResults",
        "scribe.laplace.results.ScribeLaplaceResults",
    ),
    "de": ("scribe.de.results.ScribeDEResults",),
}

# Cache of resolved category key -> tuple[type, ...].
_RESOLVED: dict = {}


def _resolve(category: Optional[str]) -> Tuple[type, ...]:
    """Resolve a ``supports`` category to a tuple of results classes.

    Parameters
    ----------
    category : str or None
        Category key. ``None`` is treated as the default ``"inference"``
        family.

    Returns
    -------
    tuple of type
        The results classes the category covers. Classes that cannot be
        imported (e.g. an optional submodule) are skipped.
    """
    key = category or "inference"
    cached = _RESOLVED.get(key)
    if cached is not None:
        return cached
    classes = []
    for dotted in _CATEGORIES.get(key, ()):
        module_name, _, attr = dotted.rpartition(".")
        try:
            module = importlib.import_module(module_name)
            classes.append(getattr(module, attr))
        except (ImportError, AttributeError):
            continue
    resolved = tuple(classes)
    _RESOLVED[key] = resolved
    return resolved


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class VizEntry:
    """Metadata for a single results-aware plotting function.

    Attributes
    ----------
    name : str
        Public function name; matches the ``scribe.viz`` export.
    func : callable
        The public plotting callable whose first positional parameter is the
        results object.
    supports : str or None
        Category key (see ``_CATEGORIES``) restricting which results types
        expose this plot. ``None`` means the default ``"inference"`` family.
    results_param : str
        Name of the first positional parameter (``"results"`` or
        ``"de_results"``). Informational only; binding is positional.
    """

    __slots__ = ("name", "func", "supports", "results_param")

    def __init__(
        self,
        name: str,
        func: Callable,
        supports: Optional[str] = None,
        results_param: str = "results",
    ) -> None:
        self.name = name
        self.func = func
        self.supports = supports
        self.results_param = results_param

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"VizEntry(name={self.name!r}, supports={self.supports!r}, "
            f"results_param={self.results_param!r})"
        )


# name -> VizEntry, populated as a side effect of importing each viz submodule.
VIZ_REGISTRY: dict = {}


def register_viz(entry: VizEntry) -> None:
    """Add or replace a registry entry, keyed by ``entry.name``."""
    VIZ_REGISTRY[entry.name] = entry


def _first_positional_name(func: Callable) -> str:
    """Return the name of ``func``'s first positional parameter.

    Falls back to ``"results"`` for ``*args``-style wrappers whose first
    parameter is variadic.
    """
    for p in inspect.signature(func).parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            return p.name
    return "results"


def register_plot(
    func: Optional[Callable] = None,
    *,
    supports: Optional[str] = None,
    accessor: bool = True,
) -> Callable:
    """Register a results-aware plotting function for the ``.viz`` accessor.

    Use this for hand-written public plotters that are *not* wrapped by
    :func:`scribe.viz._interactive.plot_function` (decorated functions register
    themselves). Usable bare or parameterized::

        @register_plot
        def plot_thing(results, ...): ...

        @register_plot(supports="de")
        def plot_de_thing(de_results, ...): ...

    Returns ``func`` unchanged so it is transparent as a decorator.

    Parameters
    ----------
    func : callable, optional
        The plotting function, supplied when used as a bare decorator.
    supports : str or None
        Category key restricting applicable results types (``None`` ->
        ``"inference"``).
    accessor : bool
        When ``False`` the function is not registered; returns ``func``
        untouched (parity with the ``plot_function`` flag).
    """

    def _apply(fn: Callable) -> Callable:
        if accessor:
            register_viz(
                VizEntry(
                    fn.__name__,
                    fn,
                    supports,
                    _first_positional_name(fn),
                )
            )
        return fn

    return _apply(func) if func is not None else _apply


# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------


def _supports(entry: VizEntry, results) -> bool:
    """Return ``True`` if ``entry`` applies to ``results``'s type."""
    classes = _resolve(entry.supports)
    return bool(classes) and isinstance(results, classes)


def _bound_signature(func: Callable):
    """Return ``func``'s signature with its leading results param removed.

    Used so ``help(results.viz.plot_ppc)`` shows ``(counts, ...)`` rather than
    ``(results, counts, ...)``. Returns ``None`` if introspection fails or the
    first parameter is variadic (``*args`` wrappers).
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return None
    params = list(sig.parameters.values())
    if params and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return sig.replace(parameters=params[1:])
    return sig


class VizAccessor:
    """Bound plotting namespace returned by ``results.viz``.

    Each attribute is the matching ``scribe.viz`` plotting function with the
    results object pre-bound as the first positional argument, so
    ``results.viz.plot_ppc(counts)`` is equivalent to
    ``scribe.viz.plot_ppc(results, counts)``. Only plots whose ``supports``
    category includes the results type are exposed; the rest raise
    ``AttributeError``.
    """

    __slots__ = ("_results",)

    def __init__(self, results) -> None:
        self._results = results

    def __getattr__(self, name: str):
        # Guard private/dunder lookups first. This both avoids exposing
        # non-plot names and prevents infinite recursion: the error paths
        # below read ``self._results``, whose own (unset) slot lookup would
        # otherwise re-enter __getattr__.
        if name.startswith("_"):
            raise AttributeError(name)

        # Accessing ``name`` on scribe.viz triggers its lazy import, which
        # runs the plotting module and (for results-aware plots) populates
        # VIZ_REGISTRY as a side effect.
        import scribe.viz as _viz

        if not hasattr(_viz, name):
            raise AttributeError(
                f"{type(self._results).__name__}.viz has no plot {name!r}."
            )
        entry = VIZ_REGISTRY.get(name)
        if entry is None:
            raise AttributeError(
                f"{name!r} is a scribe.viz function but is not a "
                f"results-bound plot; call scribe.viz.{name}(...) directly."
            )
        if not _supports(entry, self._results):
            family = entry.supports or "inference"
            raise AttributeError(
                f"{name!r} is not available for "
                f"{type(self._results).__name__} "
                f"(supported results family: {family!r})."
            )

        bound = functools.partial(entry.func, self._results)
        functools.update_wrapper(bound, entry.func)
        sig = _bound_signature(entry.func)
        if sig is not None:
            bound.__signature__ = sig
        return bound

    def __dir__(self):
        # Force the lazy import of every export so the registry is fully
        # populated, then list only the plots that apply to this type.
        import scribe.viz as _viz

        for export in _viz._LAZY_EXPORTS:
            try:
                getattr(_viz, export)
            except AttributeError:
                continue
        return sorted(
            name
            for name, entry in VIZ_REGISTRY.items()
            if _supports(entry, self._results)
        )

    def __repr__(self) -> str:
        plots = ", ".join(self.__dir__()) or "<none>"
        return f"<VizAccessor for {type(self._results).__name__}: {plots}>"


class VizAccessorMixin:
    """Mixin adding a lazily-constructed ``.viz`` accessor to results classes.

    ``viz`` is defined as a ``property`` (not a dataclass field), so it does
    not affect dataclass field ordering on the concrete results classes.
    """

    @property
    def viz(self) -> "VizAccessor":
        """Object-oriented plotting namespace for this results object."""
        return VizAccessor(self)
