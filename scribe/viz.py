"""
Plotting functions
"""

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Import seaborn for plotting
import seaborn as sns

# Import numpy for array manipulation
import numpy as np
import scipy.stats as stats

# Import typing
from typing import Union

# ------------------------------------------------------------------------------
# General plotting functions
# ------------------------------------------------------------------------------


def matplotlib_style():
    """
    Sets plotting defaults to personal style for matplotlib.
    """
    # Check if Roboto is available
    import matplotlib.font_manager as fm

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_family = (
        ["Roboto", "sans-serif"]
        if "Roboto" in available_fonts
        else ["sans-serif"]
    )

    # Define the matplotlib styles.
    rc = {
        # Axes formatting
        "axes.facecolor": "#E6E6EF",
        "axes.edgecolor": "none",  # Remove spines
        "axes.labelcolor": "#000000",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        # Font sizes
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        # Grid formatting
        "grid.linestyle": "-",
        "grid.linewidth": 1.25,
        "grid.color": "#FFFFFF",
        # Lines formatting
        "lines.linewidth": 2.0,
        # Legend formatting
        "legend.fontsize": 15,
        "legend.title_fontsize": 15,
        "legend.frameon": True,
        "legend.facecolor": "#E6E6EF",
        # Tick formatting
        "xtick.bottom": False,
        "ytick.left": False,
        # Font styling - with fallbacks
        "font.family": font_family,
        "font.style": "normal",
        "axes.titleweight": "bold",
        # Higher-order things
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.default": "regular",
    }

    # Use seaborn's colorblind palette
    sns.set_style(rc)
    sns.set_palette("colorblind")


# ------------------------------------------------------------------------------


def colors():
    """
    Returns dictionary with personal color palette.
    """
    col = {
        "dark_black": "#000000",
        "black": "#000000",
        "light_black": "#05080F",
        "pale_black": "#1F1F1F",
        "dark_blue": "#2957A8",
        "blue": "#3876C0",
        "light_blue": "#81A9DA",
        "pale_blue": "#C0D4ED",
        "dark_green": "#2E5C0A",
        "green": "#468C12",
        "light_green": "#6EBC24",
        "pale_green": "#A9EB70",
        "dark_red": "#912E27",
        "red": "#CB4338",
        "light_red": "#D57A72",
        "pale_red": "#E8B5B0",
        "dark_gold": "#B68816",
        "gold": "#EBC21F",
        "light_gold": "#F2D769",
        "pale_gold": "#F7E6A1",
        "dark_purple": "#5E315E",
        "purple": "#934D93",
        "light_purple": "#BC7FBC",
        "pale_purple": "#D5AFD5",
    }

    # Convert hex strings to RGB tuples
    colors = {}
    for key, hex_color in col.items():
        # Convert hex to RGB (0-1 scale)
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
        colors[key] = rgb

    return colors


# ------------------------------------------------------------------------------
# Posterior diagnostic plots
# ------------------------------------------------------------------------------


def plot_posterior(
    ax,
    distribution,
    ground_truth=None,
    plot_quantiles=(0.001, 0.999),
    n_points=200,
    color="blue",
    ground_truth_color="red",
    ground_truth_linestyle="--",
    ground_truth_linewidth=1.5,
    ground_truth_label=None,
    alpha=0.6,
    label=None,
    fill_between=True,
    fill_color="blue",
    fill_alpha=0.2,
) -> None:
    """
    Plot a posterior distribution on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    distribution : scipy.stats.rv_continuous
        The distribution to plot
    ground_truth : float, optional
        Ground truth value to plot as vertical line
    plot_quantiles : tuple of float, optional
        Lower and upper quantiles for plot range (default: (0.001, 0.999))
    n_points : int, optional
        Number of points to use for plotting (default: 200)
    color : str, optional
        Color for the distribution plot (default: 'blue')
    ground_truth_color : str, optional
        Color for the ground truth line (default: 'red')
    ground_truth_linestyle : str, optional
        Line style for ground truth (default: '--')
    ground_truth_linewidth : float, optional
        Line width for ground truth (default: 1.5)
    ground_truth_label : str, optional
        Label for the ground truth in the legend
    alpha : float, optional
        Transparency for the distribution plot (default: 0.6)
    label : str, optional
        Label for the distribution in the legend
    fill_between : bool, optional
        Whether to fill the distribution plot (default: True)
    fill_color : str, optional
        Color for the fill_between plot (default: 'blue')
    fill_alpha : float, optional
        Transparency for the fill_between plot (default: 0.2)

    Returns
    -------
    None
        The plot is created on the provided axis
    """
    # For scipy.stats distributions
    q_lower = float(distribution.ppf(plot_quantiles[0]))
    q_upper = float(distribution.ppf(plot_quantiles[1]))

    # Create x values for plotting
    x = np.linspace(q_lower, q_upper, n_points)

    # Get PDF values
    pdf = distribution.pdf(x)

    # Create the plot
    if fill_between:
        # Fill between the lower and upper quantiles
        ax.fill_between(x, pdf, alpha=fill_alpha, color=fill_color, label=label)
    # Plot the PDF
    ax.plot(x, pdf, color=color, alpha=alpha)

    # Add ground truth if provided
    if ground_truth is not None:
        ax.axvline(
            ground_truth,
            color=ground_truth_color,
            linestyle=ground_truth_linestyle,
            linewidth=ground_truth_linewidth,
            label=(
                ground_truth_label if ground_truth_label is not None else None
            ),
        )

    # Add legend if label was provided
    if label is not None:
        ax.legend()

    # Set axes labels
    ax.set_ylabel("Density")


# ------------------------------------------------------------------------------


def plot_histogram_credible_regions(
    ax,
    hist_results,
    colors=None,
    cmap=None,
    alpha=0.2,
    plot_median=True,
    median_color="black",
    median_alpha=0.2,
    median_linewidth=1.5,
    label_prefix="",
    max_bin=None,
):
    """
    Plot credible regions as fill_between on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    hist_results : dict
        Results dictionary from compute_histogram_credible_regions
    colors : list, optional
        List of colors for each credible region. Must match length of
        credible regions. If None and cmap is None, defaults to grays.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to generate colors from. Ignored if colors is provided.
    alpha : float or list, optional
        Transparency for fill_between plots. If float, same alpha used for all
        regions. If list, must match length of credible regions.
    plot_median : bool, optional
        Whether to plot the median line (default: True)
    median_color : str, optional
        Color for median line (default: 'black')
    median_alpha : float, optional
        Transparency for median line (default: 0.8)
    median_linewidth : float, optional
        Line width for median line (default: 1.5)
    label_prefix : str, optional
        Prefix for legend labels (default: '')
    max_bin : int, optional
        Maximum bin index to plot. If provided, only bins up to this index will
        be shown. Default is None (show all bins).

    Returns
    -------
    matplotlib.axes.Axes
        The axis with plots added
    """
    bin_edges = hist_results["bin_edges"]
    x = bin_edges[:-1]  # Use left edges for plotting

    # Apply max_bin limit if specified
    if max_bin is not None:
        x = x[:max_bin]
        bin_edges = bin_edges[
            : max_bin + 1
        ]  # Include one more edge for proper plotting

    # Sort credible regions from largest to smallest for proper layering
    cr_values = sorted(hist_results["regions"].keys(), reverse=True)
    n_regions = len(cr_values)

    # Handle colors
    if colors is None:
        if cmap is None:
            # Default to grays if no colors specified
            colors = [f"gray" for _ in range(n_regions)][::-1]
        else:
            # Generate colors from colormap
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            colors = [cmap(i / (n_regions - 1)) for i in range(n_regions)][::-1]

    # Handle alpha
    if isinstance(alpha, (int, float)):
        alphas = [alpha] * n_regions
    else:
        alphas = alpha

    # Plot credible regions
    for cr, color, alpha in zip(cr_values, colors, alphas):
        region = hist_results["regions"][cr]
        ax.fill_between(
            x,
            (
                region["lower"][:max_bin]
                if max_bin is not None
                else region["lower"]
            ),
            (
                region["upper"][:max_bin]
                if max_bin is not None
                else region["upper"]
            ),
            color=color,
            alpha=alpha,
            label=f"{label_prefix}{cr}% CR",
        )

    # Plot median
    if plot_median:
        # Use the median from any region (they're all the same)
        median = hist_results["regions"][cr_values[0]]["median"]
        ax.plot(
            x,
            median[:max_bin] if max_bin is not None else median,
            color=median_color,
            alpha=median_alpha,
            linewidth=median_linewidth,
            label=f"{label_prefix}median",
        )

    return ax


# ------------------------------------------------------------------------------


def plot_histogram_credible_regions_stairs(
    ax,
    hist_results,
    colors=None,
    cmap=None,
    alpha=0.2,
    plot_median=True,
    median_color="black",
    median_alpha=0.2,
    median_linewidth=1.5,
    label_prefix="",
    max_bin=None,
):
    """
    Plot credible regions as fill_between on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    hist_results : dict
        Results dictionary from compute_histogram_credible_regions
    colors : list, optional
        List of colors for each credible region. Must match length of
        credible regions. If None and cmap is None, defaults to grays.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to generate colors from. Ignored if colors is provided.
    alpha : float or list, optional
        Transparency for fill_between plots. If float, same alpha used for all
        regions. If list, must match length of credible regions.
    plot_median : bool, optional
        Whether to plot the median line (default: True)
    median_color : str, optional
        Color for median line (default: 'black')
    median_alpha : float, optional
        Transparency for median line (default: 0.8)
    median_linewidth : float, optional
        Line width for median line (default: 1.5)
    label_prefix : str, optional
        Prefix for legend labels (default: '')
    max_bin : int, optional
        Maximum bin index to plot. If provided, only bins up to this index will
        be shown. Default is None (show all bins).

    Returns
    -------
    matplotlib.axes.Axes
        The axis with plots added
    """
    bin_edges = hist_results["bin_edges"]

    # Apply max_bin limit if specified
    if max_bin is not None:
        bin_edges = bin_edges[: max_bin + 1]

    # Sort credible regions from largest to smallest for proper layering
    cr_values = sorted(hist_results["regions"].keys(), reverse=True)
    n_regions = len(cr_values)

    # Handle colors
    if colors is None:
        if cmap is None:
            # Default to grays if no colors specified
            colors = [f"gray" for _ in range(n_regions)][::-1]
        else:
            # Generate colors from colormap
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            colors = [cmap(i / (n_regions - 1)) for i in range(n_regions)][::-1]

    # Handle alpha
    if isinstance(alpha, (int, float)):
        alphas = [alpha] * n_regions
    else:
        alphas = alpha

    # Plot credible regions
    for cr, color, alpha in zip(cr_values, colors, alphas):
        # Get region data
        region = hist_results["regions"][cr]
        # Apply max_bin limit to the data if specified
        upper = (
            region["upper"][:max_bin]
            if max_bin is not None
            else region["upper"]
        )
        lower = (
            region["lower"][:max_bin]
            if max_bin is not None
            else region["lower"]
        )

        # Plot credible region
        ax.stairs(
            upper,
            bin_edges,
            fill=True,
            baseline=lower,
            color=color,
            alpha=alpha,
            label=f"{label_prefix}{cr}% CR",
        )

    # Plot median
    if plot_median:
        median = hist_results["regions"][cr_values[0]]["median"]
        # Apply max_bin limit to median if specified
        if max_bin is not None:
            median = median[:max_bin]

        # Plot median
        ax.stairs(
            median,
            bin_edges,
            color=median_color,
            alpha=median_alpha,
            linewidth=median_linewidth,
            label=f"{label_prefix}median",
        )

    return ax


# ------------------------------------------------------------------------------


def plot_ecdf_credible_regions_stairs(
    ax,
    ecdf_results,
    colors=None,
    cmap=None,
    alpha=0.2,
    plot_median=True,
    median_color="black",
    median_alpha=0.2,
    median_linewidth=1.5,
    label_prefix="",
    max_bin=None,
):
    """
    Plot ECDF credible regions as stairs on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    ecdf_results : dict
        Results dictionary from compute_ecdf_credible_regions
    colors : list, optional
        List of colors for each credible region. Must match length of credible
        regions. If None and cmap is None, defaults to grays.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to generate colors from. Ignored if colors is provided.
    alpha : float or list, optional
        Transparency for fill_between plots. If float, same alpha used for all
        regions. If list, must match length of credible regions.
    plot_median : bool, optional
        Whether to plot the median line (default: True)
    median_color : str, optional
        Color for median line (default: 'black')
    median_alpha : float, optional
        Transparency for median line (default: 0.2)
    median_linewidth : float, optional
        Line width for median line (default: 1.5)
    label_prefix : str, optional
        Prefix for legend labels (default: '')
    max_bin : int, optional
        Maximum x-value index to plot. If provided, only points up to this index
        will be shown. Default is None (show all points).

    Returns
    -------
    matplotlib.axes.Axes
        The axis with plots added
    """
    bin_edges = ecdf_results["bin_edges"]

    # Apply max_bin limit if specified
    if max_bin is not None:
        bin_edges = bin_edges[: max_bin + 1]

    # Sort credible regions from largest to smallest for proper layering
    cr_values = sorted(ecdf_results["regions"].keys(), reverse=True)
    n_regions = len(cr_values)

    # Handle colors
    if colors is None:
        if cmap is None:
            # Default to grays if no colors specified
            colors = [f"gray" for _ in range(n_regions)][::-1]
        else:
            # Generate colors from colormap
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            colors = [cmap(i / (n_regions - 1)) for i in range(n_regions)][::-1]

    # Handle alpha
    if isinstance(alpha, (int, float)):
        alphas = [alpha] * n_regions
    else:
        alphas = alpha

    # Plot credible regions
    for cr, color, alpha in zip(cr_values, colors, alphas):
        # Get region data
        region = ecdf_results["regions"][cr]

        # Apply max_bin limit to the data if specified
        upper = (
            region["upper"][:max_bin]
            if max_bin is not None
            else region["upper"]
        )
        lower = (
            region["lower"][:max_bin]
            if max_bin is not None
            else region["lower"]
        )

        # Plot credible region as stairs
        ax.stairs(
            upper,
            bin_edges,
            fill=True,
            baseline=lower,
            color=color,
            alpha=alpha,
            label=f"{label_prefix}{cr}% CR",
        )

    # Plot median
    if plot_median:
        # Use the median from any region (they're all the same)
        median = ecdf_results["regions"][cr_values[0]]["median"]

        # Apply max_bin limit to median if specified
        if max_bin is not None:
            median = median[:max_bin]

        ax.stairs(
            median,
            bin_edges,
            color=median_color,
            alpha=median_alpha,
            linewidth=median_linewidth,
            label=f"{label_prefix}median",
        )

    return ax
