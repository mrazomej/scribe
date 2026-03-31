"""Interactive plotting helpers for dual-mode viz APIs.

These utilities centralize the save/show/close policy used by high-level
visualization entry points so CLI and notebook usage stay consistent.
"""

from __future__ import annotations

import functools
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from ._common import console


class PlotResult:
    """Single-display wrapper for visualization output.

    Notebooks (marimo, Jupyter, etc.) call ``_repr_html_`` or
    ``_repr_png_`` and render **exactly one image** of the full figure.
    Attribute access to ``.fig`` and ``.axes`` is still available for
    programmatic customization after the plot is returned.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure backing this result.
    axes : tuple of matplotlib.axes.Axes
        Axes used by the plot, stored as a flat tuple.
    n_panels : int or None
        Number of logical panels in the plot.
    output_path : str or None
        Filesystem path where the figure was saved, if applicable.
    """

    def __init__(self, fig, axes, n_panels=None, output_path=None):
        self.fig = fig
        self.axes = axes
        self.n_panels = n_panels
        self.output_path = output_path

    # ------------------------------------------------------------------
    # Rich display protocol (Jupyter / marimo / IPython)
    # ------------------------------------------------------------------

    def _repr_png_(self):
        """Render the figure as PNG bytes for notebook frontends."""
        from io import BytesIO

        buf = BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        return buf.getvalue()

    def _repr_html_(self):
        """Render the figure as an inline HTML ``<img>`` tag."""
        import base64

        png = self._repr_png_()
        b64 = base64.b64encode(png).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}"/>'

    def __repr__(self):
        n = self.n_panels or 0
        saved = f", saved='{self.output_path}'" if self.output_path else ""
        return f"PlotResult(n_panels={n}{saved})"


class PlotResultCollection:
    """Container for multiple ``PlotResult`` objects from one call.

    Multi-figure plot functions (e.g. ``plot_mixture_ppc``,
    ``plot_correlation_heatmap``) produce several independent figures.
    Wrapping them in a collection allows notebook rich-display to
    render *all* figures inline, while pipeline code that only inspects
    ``.fig`` or ``.output_path`` sees the first result transparently.

    Parameters
    ----------
    results : list of PlotResult
        Individual plot results comprising the collection.
    """

    def __init__(self, results):
        if not results:
            raise ValueError(
                "PlotResultCollection requires at least one PlotResult."
            )
        self._results = list(results)

    # ---- Sequence protocol ----

    def __len__(self):
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def __getitem__(self, idx):
        return self._results[idx]

    # ---- Forward first-result attributes for backward compat ----

    @property
    def fig(self):
        """Figure from the first result (convenience accessor)."""
        return self._results[0].fig

    @property
    def axes(self):
        """Axes from the first result (convenience accessor)."""
        return self._results[0].axes

    @property
    def n_panels(self):
        """Panel count from the first result."""
        return self._results[0].n_panels

    @property
    def output_path(self):
        """Output path from the first result."""
        return self._results[0].output_path

    @property
    def output_paths(self):
        """List of output paths from all results."""
        return [r.output_path for r in self._results]

    # ---- Rich display ----

    def _repr_html_(self):
        """Render all figures inline as sequential ``<img>`` tags."""
        import base64

        parts = []
        for r in self._results:
            png = r._repr_png_()
            b64 = base64.b64encode(png).decode("ascii")
            parts.append(
                f'<img src="data:image/png;base64,{b64}" '
                f'style="display:block;margin:8px 0"/>'
            )
        return "\n".join(parts)

    def _repr_png_(self):
        """PNG bytes of the first figure (fallback for plain PNG renderers)."""
        return self._results[0]._repr_png_()

    def __repr__(self):
        n = len(self._results)
        paths = [r.output_path for r in self._results if r.output_path]
        suffix = f", saved={paths}" if paths else ""
        return f"PlotResultCollection(n_figures={n}{suffix})"


def _resolve_bool_default(value, default):
    """Resolve optional boolean arguments with explicit defaults.

    Parameters
    ----------
    value : bool or None
        User-provided value.
    default : bool
        Default fallback when ``value`` is ``None``.

    Returns
    -------
    bool
        Resolved boolean.
    """
    return bool(default) if value is None else bool(value)


def _is_interactive_session():
    """Return ``True`` when running under an interactive frontend.

    Detects both IPython-based notebooks (Jupyter, Colab, etc.) and
    marimo reactive notebooks.
    """
    # Marimo detection: marimo exposes ``running_as_script`` which is
    # ``False`` inside a reactive notebook and ``True`` in script mode.
    try:
        import marimo as mo

        if mo.running_as_script() is False:
            return True
    except Exception:
        pass

    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def _resolve_render_flags(figs_dir, save, show, close):
    """Compute save/show/close policy for dual-mode plotting.

    Display in notebooks is handled by the ``PlotResult`` rich-repr
    protocol, so ``show`` defaults to ``False`` (``plt.show()`` is only
    invoked when the caller explicitly passes ``show=True``).

    Parameters
    ----------
    figs_dir : str or None
        Target directory for file output.
    save : bool or None
        Whether to save a file. ``None`` → ``True`` when *figs_dir* is
        provided.
    show : bool or None
        Whether to call ``plt.show()``. Defaults to ``False``.
    close : bool or None
        Whether to close the figure after rendering.  Defaults to match
        the resolved *save* flag.

    Returns
    -------
    tuple of bool
        Tuple ``(save_final, show_final, close_final)``.
    """
    save_final = _resolve_bool_default(save, figs_dir is not None)
    show_final = _resolve_bool_default(show, False)
    close_final = _resolve_bool_default(close, save_final)
    return save_final, show_final, close_final


@dataclass
class PlotContext:
    """Centralizes save/show/close policy and filename construction.

    Created via the ``from_kwargs`` classmethod, which resolves render
    flags and ownership tracking so individual plot functions need only
    two lines of boilerplate::

        ctx = PlotContext.from_kwargs(
            figs_dir=figs_dir, cfg=cfg, viz_cfg=viz_cfg,
            fig=fig, ax=ax, axes=axes,
            save=save, show=show, close=close,
        )
        # ... draw on fig / axes ...
        return ctx.finalize(fig, axes_flat, n_panels,
                            filename=ctx.build_filename("loss", results=results))

    Parameters
    ----------
    figs_dir : str or None
        Target directory for file output.
    cfg : Any
        Run configuration (OmegaConf or dict) used for filenames.
    viz_cfg : Any
        Visualization configuration carrying format / option keys.
    save : bool
        Whether saving to disk is enabled.
    show : bool
        Whether ``plt.show()`` should be called.
    close : bool
        Whether the figure is explicitly closed after rendering.
    output_format : str
        File extension (``"png"``, ``"pdf"``, ...).
    fig_owned : bool
        ``True`` when the figure was created internally (not injected
        by the caller).
    """

    figs_dir: str | None
    cfg: Any
    viz_cfg: Any
    save: bool
    show: bool
    close: bool
    output_format: str = field(default="png")
    fig_owned: bool = field(default=True)

    @classmethod
    def from_kwargs(
        cls,
        *,
        figs_dir=None,
        cfg=None,
        viz_cfg=None,
        fig=None,
        ax=None,
        axes=None,
        save=None,
        show=None,
        close=None,
    ):
        """Build a context from the standard plot-function keyword set.

        Parameters
        ----------
        figs_dir : str or None
            Output directory.
        cfg, viz_cfg : object or None
            Configuration objects forwarded through for filename
            construction.
        fig, ax, axes : matplotlib objects or None
            Caller-injected figure / axes.  Used only to determine
            ``fig_owned``.
        save, show, close : bool or None
            Explicit user overrides resolved by
            ``_resolve_render_flags``.

        Returns
        -------
        PlotContext
        """
        save_f, show_f, close_f = _resolve_render_flags(
            figs_dir,
            save,
            show,
            close,
        )
        fig_owned = fig is None and ax is None and axes is None
        fmt = "png"
        if save_f and viz_cfg is not None:
            fmt = (
                viz_cfg.get("format", "png")
                if hasattr(viz_cfg, "get")
                else getattr(viz_cfg, "format", "png")
            )
        return cls(
            figs_dir=figs_dir,
            cfg=cfg,
            viz_cfg=viz_cfg,
            save=save_f,
            show=show_f,
            close=close_f,
            output_format=fmt,
            fig_owned=fig_owned,
        )

    def build_filename(self, suffix, *, results=None):
        """Build the standardized output filename.

        Parameters
        ----------
        suffix : str
            Plot-type suffix appended to the base name
            (e.g. ``"loss"``, ``"ppc"``).
        results : object, optional
            Fitted results object forwarded to ``_get_config_values``
            for ``n_components`` resolution.

        Returns
        -------
        str or None
            Full filename when saving is enabled, ``None`` otherwise.
        """
        if not self.save:
            return None
        from .config import _get_config_values

        config_vals = _get_config_values(self.cfg, results=results)
        base = (
            f"{config_vals['method']}_"
            f"{config_vals['parameterization'].replace('-', '_')}_"
            f"{config_vals['model_type'].replace('_', '-')}_"
            f"{config_vals['n_components']:02d}components_"
            f"{config_vals['run_size_token']}"
        )
        return f"{base}_{suffix}.{self.output_format}"

    def finalize(
        self,
        fig,
        axes,
        n_panels,
        *,
        filename=None,
        save_kwargs=None,
        save_label=None,
    ):
        """Delegate to ``_finalize_figure`` using the stored policy.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to finalize.
        axes : list of matplotlib.axes.Axes
            Axes used by the plot.
        n_panels : int
            Number of logical panels.
        filename : str, optional
            Output filename (typically from ``build_filename``).
        save_kwargs : dict, optional
            Extra keyword arguments for ``fig.savefig``.
        save_label : str, optional
            Human-readable label for console output.

        Returns
        -------
        PlotResult
        """
        return _finalize_figure(
            fig=fig,
            axes=axes,
            n_panels=n_panels,
            save=self.save,
            show=self.show,
            close=self.close,
            figs_dir=self.figs_dir,
            filename=filename,
            save_kwargs=save_kwargs,
            save_label=save_label,
            _fig_owned=self.fig_owned,
        )


def _flatten_axes(axes):
    """Return a flat list of axes from scalar/list/ndarray axis inputs."""
    if axes is None:
        return []
    if isinstance(axes, np.ndarray):
        return list(axes.ravel())
    if isinstance(axes, Iterable) and not hasattr(axes, "plot"):
        return [ax for ax in axes]
    return [axes]


def _create_or_validate_single_axis(
    *, fig=None, ax=None, axes=None, figsize=None
):
    """Create or validate axis inputs for single-panel plots.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or None
        Optional figure to host a newly created axis.
    ax : matplotlib.axes.Axes or None
        Optional pre-existing axis.
    axes : object, optional
        Optional alias for one-axis layouts.
    figsize : tuple, optional
        Figure size used when creating a new figure.

    Returns
    -------
    tuple
        Tuple ``(fig, ax)`` ready for plotting.
    """
    if ax is not None and axes is not None:
        raise ValueError(
            "Provide only one of `ax` or `axes` for single-panel plots."
        )
    if ax is not None and fig is not None and ax.figure is not fig:
        raise ValueError("Provided `ax` does not belong to provided `fig`.")

    resolved_ax = ax
    if resolved_ax is None and axes is not None:
        flat_axes = _flatten_axes(axes)
        if len(flat_axes) != 1:
            raise ValueError(
                f"Single-panel plot requires exactly 1 axis, but received {len(flat_axes)}."
            )
        resolved_ax = flat_axes[0]

    if resolved_ax is not None:
        return resolved_ax.figure if fig is None else fig, resolved_ax

    if fig is not None:
        return fig, fig.add_subplot(1, 1, 1)

    new_fig, new_ax = plt.subplots(1, 1, figsize=figsize)
    return new_fig, new_ax


def _create_or_validate_grid_axes(
    *,
    n_rows,
    n_cols,
    fig=None,
    axes=None,
    figsize=None,
    fig_kw=None,
):
    """Create or validate axis inputs for multi-panel grids.

    Parameters
    ----------
    n_rows : int
        Number of subplot rows.
    n_cols : int
        Number of subplot columns.
    fig : matplotlib.figure.Figure or None
        Optional figure used to create/host the grid.
    axes : object, optional
        Optional user-provided axes container.
    figsize : tuple, optional
        Figure size for new figure creation.
    fig_kw : dict or None
        Additional figure keyword arguments passed to subplot creation.

    Returns
    -------
    tuple
        Tuple ``(fig, axes_array_2d, axes_flat)``.
    """
    expected_n = int(n_rows) * int(n_cols)
    fig_kw = {} if fig_kw is None else dict(fig_kw)

    if axes is not None:
        axes_flat = _flatten_axes(axes)
        if len(axes_flat) != expected_n:
            raise ValueError(
                "Invalid `axes` shape for this plot: "
                f"expected {expected_n} axes arranged as ({n_rows}, {n_cols}), "
                f"received {len(axes_flat)} axes."
            )
        if fig is None:
            fig = axes_flat[0].figure
        axes_array = np.asarray(axes_flat, dtype=object).reshape(n_rows, n_cols)
        return fig, axes_array, list(axes_flat)

    if fig is not None:
        axes_array = np.asarray(
            fig.subplots(n_rows, n_cols, squeeze=False, **fig_kw), dtype=object
        )
        return fig, axes_array, list(axes_array.ravel())

    fig, axes_array = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        **fig_kw,
    )
    axes_array = np.asarray(axes_array, dtype=object)
    return fig, axes_array, list(axes_array.ravel())


def _finalize_figure(
    *,
    fig,
    axes,
    n_panels,
    save,
    show,
    close,
    figs_dir=None,
    filename=None,
    save_kwargs=None,
    save_label=None,
    _fig_owned=True,
):
    """Save / show / detach a figure and wrap it in a ``PlotResult``.

    When the figure was created internally (``_fig_owned=True``) and
    ``show`` is ``False``, the figure is detached from pyplot's figure
    manager via ``plt.close(fig)``.  This prevents notebook backends
    from auto-displaying it a second time.  The ``Figure`` object
    itself remains fully usable (``fig.savefig``, axis access, etc.).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalize.
    axes : list of matplotlib.axes.Axes
        Axes used by the plot (will be stored as a tuple).
    n_panels : int
        Number of logical panels.
    save : bool
        Whether to save to disk.
    show : bool
        Whether to call ``plt.show()``.
    close : bool
        Whether to explicitly close the figure via ``plt.close(fig)``.
    figs_dir : str, optional
        Output directory for save mode.
    filename : str, optional
        Output file name for save mode.
    save_kwargs : dict, optional
        Extra keyword arguments forwarded to ``fig.savefig``.
    save_label : str, optional
        Human-readable save label for console output.
    _fig_owned : bool
        ``True`` when the figure was created internally (not injected by
        the caller).  Internally-created figures are detached from
        pyplot state to prevent duplicate display.

    Returns
    -------
    PlotResult
        Wrapped result containing figure, axes, and metadata.
    """
    output_path = None
    if save:
        if figs_dir is None or filename is None:
            raise ValueError(
                "Saving is enabled but `figs_dir`/`filename` were not provided."
            )
        os.makedirs(figs_dir, exist_ok=True)
        output_path = os.path.join(figs_dir, filename)
        save_kwargs = {} if save_kwargs is None else dict(save_kwargs)
        fig.savefig(output_path, **save_kwargs)
        label = "plot" if save_label is None else str(save_label)
        console.print(
            f"[green]✓[/green] [dim]Saved {label} to[/dim] [cyan]{output_path}[/cyan]"
        )

    if show:
        plt.show()

    # Detach internally-created figures from pyplot's figure manager so
    # that notebook backends (Jupyter post_execute hook, marimo capture)
    # do not auto-render a duplicate.  The Figure object stays valid for
    # PlotResult._repr_png_() and direct attribute access.
    if _fig_owned and not show:
        plt.close(fig)

    # Honour explicit close request (e.g. CLI batch mode).
    if close:
        plt.close(fig)

    return PlotResult(
        fig=fig,
        axes=tuple(axes),
        n_panels=n_panels,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# PPC grid resolution
# ---------------------------------------------------------------------------

# Built-in defaults matching conf/viz/default.yaml so that
# viz_cfg=None works out of the box in interactive sessions.
_PPC_DEFAULTS = {
    "n_rows": 5,
    "n_cols": 5,
    "n_samples": 512,
}


def _resolve_ppc_grid(
    *,
    n_rows=None,
    n_cols=None,
    n_genes=None,
    n_samples=None,
    viz_cfg=None,
    opts_key="ppc_opts",
    default_rows=None,
    default_cols=None,
    default_samples=None,
):
    """Resolve PPC grid dimensions from explicit kwargs > viz_cfg > defaults.

    Priority order for each value:

    1. Explicit keyword argument (``n_rows=4``).
    2. ``viz_cfg.<opts_key>.<key>`` when ``viz_cfg`` is provided.
    3. Built-in default.

    When ``n_genes`` is given **without** ``n_cols``, the column count is
    derived as ``ceil(n_genes / n_rows)`` so that the grid has at least
    ``n_genes`` panels.

    Parameters
    ----------
    n_rows, n_cols, n_genes, n_samples : int or None
        Explicit overrides.
    viz_cfg : OmegaConf or mapping-like or None
        Visualization configuration.
    opts_key : str
        Key under ``viz_cfg`` holding the options dict
        (``"ppc_opts"``, ``"mixture_ppc_opts"``, …).
    default_rows, default_cols, default_samples : int or None
        Per-call overrides for the built-in defaults.  ``None`` falls
        through to ``_PPC_DEFAULTS``.

    Returns
    -------
    dict
        ``{"n_rows": int, "n_cols": int, "n_samples": int}``.

    Raises
    ------
    ValueError
        If both ``n_genes`` and ``n_cols`` are given and are inconsistent.
    """
    d_rows = (
        default_rows if default_rows is not None else _PPC_DEFAULTS["n_rows"]
    )
    d_cols = (
        default_cols if default_cols is not None else _PPC_DEFAULTS["n_cols"]
    )
    d_samples = (
        default_samples
        if default_samples is not None
        else _PPC_DEFAULTS["n_samples"]
    )

    # Read from viz_cfg when available
    def _cfg_get(key, default):
        if viz_cfg is None:
            return default
        opts = viz_cfg.get(opts_key, {}) if hasattr(viz_cfg, "get") else {}
        return opts.get(key, default) if hasattr(opts, "get") else default

    rows = n_rows if n_rows is not None else int(_cfg_get("n_rows", d_rows))
    cols = n_cols if n_cols is not None else None
    samples = (
        n_samples
        if n_samples is not None
        else int(_cfg_get("n_samples", d_samples))
    )

    import math

    if n_genes is not None:
        derived_cols = math.ceil(int(n_genes) / rows)
        if cols is not None and cols != derived_cols:
            raise ValueError(
                f"n_genes={n_genes} with n_rows={rows} requires "
                f"n_cols={derived_cols}, but n_cols={cols} was also given."
            )
        cols = derived_cols

    if cols is None:
        cols = int(_cfg_get("n_cols", d_cols))

    return {"n_rows": int(rows), "n_cols": int(cols), "n_samples": int(samples)}


# ---------------------------------------------------------------------------
# @plot_function decorator
# ---------------------------------------------------------------------------

# Kwargs consumed by the decorator (not forwarded to the inner function).
_RENDER_KWARGS = frozenset({"figs_dir", "cfg", "save", "show", "close"})


def _build_public_params(inner_params):
    """Construct a public-API parameter list from the inner function's params.

    The transformation:

    1. Positional params from the inner function are preserved.
    2. ``figs_dir=None`` and ``cfg=None`` are inserted as positional-or-
       keyword after the domain positional params.
    3. ``viz_cfg`` (if present as keyword-only in the inner function) is
       promoted to positional-or-keyword and placed right after ``cfg``.
    4. ``ctx`` is removed (injected by the wrapper at call time).
    5. ``save=None``, ``show=None``, ``close=None`` are appended as
       keyword-only.
    """
    pos_params = []
    kw_params = []

    for p in inner_params:
        if p.name == "ctx":
            continue
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            pos_params.append(p)
        elif p.kind == p.KEYWORD_ONLY:
            kw_params.append(p)

    # Promote viz_cfg from keyword-only → positional-or-keyword so that
    # callers can keep passing it positionally alongside figs_dir / cfg.
    viz_cfg_param = None
    kw_filtered = []
    for p in kw_params:
        if p.name == "viz_cfg":
            viz_cfg_param = p.replace(
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        else:
            kw_filtered.append(p)

    injected_pos = [
        inspect.Parameter(
            "figs_dir",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
        ),
        inspect.Parameter(
            "cfg",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
        ),
    ]
    if viz_cfg_param is not None:
        injected_pos.append(viz_cfg_param)

    all_pos = pos_params + injected_pos

    render_kw = [
        inspect.Parameter("save", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("show", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter(
            "close", inspect.Parameter.KEYWORD_ONLY, default=None
        ),
    ]

    return all_pos + kw_filtered + render_kw


def plot_function(*, suffix=None, save_label=None, save_kwargs=None):
    """Decorator that automates ``PlotContext`` boilerplate.

    The decorated function's *public* signature preserves the standard
    ``(…, figs_dir=, cfg=, viz_cfg=, *, …, save=, show=, close=)``
    layout so that CLI pipelines and notebooks keep working unchanged.
    Internally the function receives a ``ctx`` keyword argument
    (``PlotContext``) instead and returns one of:

    * ``None`` — propagated as-is (early-exit).
    * ``PlotResult`` or ``PlotResultCollection`` — returned unchanged.
    * ``(fig, axes_list, n_panels)`` — decorator calls
      ``ctx.finalize(…)``.
    * ``(fig, axes_list, n_panels, extra)`` — *extra* ``dict`` is
      merged into the ``ctx.finalize(…)`` call.  Recognised keys:
      ``suffix``, ``save_label``, ``save_kwargs``, ``filename``.

    Parameters
    ----------
    suffix : str, optional
        Default filename suffix passed to ``ctx.build_filename``.
    save_label : str, optional
        Default human-readable label for console save messages.
    save_kwargs : dict, optional
        Default keyword arguments forwarded to ``fig.savefig``.

    Examples
    --------
    >>> @plot_function(suffix="ecdf", save_label="ECDF plot",
    ...               save_kwargs={"bbox_inches": "tight"})
    ... def plot_ecdf(counts, *, ctx, viz_cfg=None,
    ...               fig=None, ax=None, axes=None):
    ...     fig, ax = _create_or_validate_single_axis(
    ...         fig=fig, ax=ax, axes=axes, figsize=(3.5, 3.0),
    ...     )
    ...     # … drawing …
    ...     return fig, [ax], 1
    """
    _default_save_kwargs = dict(save_kwargs) if save_kwargs else {}

    def decorator(fn):
        inner_sig = inspect.signature(fn)
        public_params = _build_public_params(
            list(inner_sig.parameters.values()),
        )
        public_sig = inspect.Signature(
            public_params,
            return_annotation=inner_sig.return_annotation,
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Bind against the public signature so that positional calls
            # from the CLI pipeline are resolved correctly.
            bound = public_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            all_args = dict(bound.arguments)

            # Extract render kwargs consumed by the decorator.
            figs_dir = all_args.pop("figs_dir", None)
            cfg = all_args.pop("cfg", None)
            save_val = all_args.pop("save", None)
            show_val = all_args.pop("show", None)
            close_val = all_args.pop("close", None)

            ctx = PlotContext.from_kwargs(
                figs_dir=figs_dir,
                cfg=cfg,
                viz_cfg=all_args.get("viz_cfg"),
                fig=all_args.get("fig"),
                ax=all_args.get("ax"),
                axes=all_args.get("axes"),
                save=save_val,
                show=show_val,
                close=close_val,
            )
            all_args["ctx"] = ctx

            result = fn(**all_args)

            # --- pass-through cases ---
            if result is None:
                return None
            if isinstance(result, (PlotResult, PlotResultCollection)):
                return result

            # --- auto-finalize from tuple return ---
            if not isinstance(result, tuple) or len(result) < 3:
                raise TypeError(
                    f"@plot_function-decorated '{fn.__name__}' must return "
                    f"None, PlotResult, PlotResultCollection, or "
                    f"(fig, axes_list, n_panels[, extra_dict])."
                )

            fig_out, axes_out, n_panels = result[0], result[1], result[2]
            extra = dict(result[3]) if len(result) > 3 else {}

            # Resolve filename: extra["filename"] > extra["suffix"] > default suffix
            fname = extra.pop("filename", None)
            if fname is None:
                resolved_suffix = extra.pop("suffix", suffix)
                if resolved_suffix:
                    results_obj = all_args.get("results")
                    fname = ctx.build_filename(
                        resolved_suffix,
                        results=results_obj,
                    )

            merged_kw = dict(_default_save_kwargs)
            merged_kw.update(extra.pop("save_kwargs", {}))

            return ctx.finalize(
                fig_out,
                axes_out,
                n_panels,
                filename=fname,
                save_kwargs=merged_kw or None,
                save_label=extra.pop("save_label", save_label),
            )

        wrapper.__signature__ = public_sig
        return wrapper

    return decorator
