import matplotlib
import seaborn as sns
import numpy as np


def get_colors(all_palettes=False):
    """
    Generates a dictionary of standard colors and returns a sequential color
    palette.

    Parameters
    ----------
    all_palettes : bool
        If True, lists of `dark`, `primary`, and `light` palettes will be returned. If
        False, only the `primary` palette will be returned.
    """
    # Define the colors
    colors = {
        'dark_black': '#2b2b2a',
        'black': '#3d3d3d',
        'primary_black': '#4c4b4c',
        'light_black': '#8c8c8c',
        'pale_black': '#afafaf',
        'dark_blue': '#154577',
        'blue': '#005da2',
        'primary_blue': '#3373ba',
        'light_blue': '#5fa6db',
        'pale_blue': '#8ec1e8',
        'dark_green': '#356835',
        'green': '#488d48',
        'primary_green': '#5cb75b',
        'light_green': '#99d097',
        'pale_green': '#b8ddb6',
        'dark_red': '#79302e',
        'red': '#a3433f',
        'primary_red': '#d8534f',
        'light_red': '#e89290',
        'pale_red': '#eeb3b0',
        'dark_gold': '#84622c',
        'gold': '#b1843e',
        'primary_gold': '#f0ad4d',
        'light_gold': '#f7cd8e',
        'pale_gold': '#f8dab0',
        'dark_purple': '#43355d',
        'purple': '#5d4a7e',
        'primary_purple': '#8066ad',
        'light_purple': '#a897c5',
        'pale_purple': '#c2b6d6'
    }

    # Generate the sequential color palettes.
    keys = ['black', 'blue', 'green', 'red', 'purple', 'gold']
    dark_palette = [colors[f'dark_{k}'] for k in keys]
    primary_palette = [colors[f'primary_{k}'] for k in keys]
    light_palette = [colors[f'light_{k}'] for k in keys]

    # Determine what to return.
    if all_palettes:
        palette = [dark_palette, primary_palette, light_palette]
    else:
        palette = primary_palette

    return [colors, palette]


def matplotlib_style():
    """
    Sets plotting defaults to personal style for matplotlib.
    """
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
        "grid.linestyle": '-',
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

        # Font styling
        "font.family": "Roboto",
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
