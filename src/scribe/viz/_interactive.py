"""Interactive plotting helpers for dual-mode viz APIs.

These utilities centralize the save/show/close policy used by high-level
visualization entry points so CLI and notebook usage stay consistent.
"""

from __future__ import annotations

import os
from typing import Iterable

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
