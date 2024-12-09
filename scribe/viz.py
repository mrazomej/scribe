import matplotlib
import seaborn as sns
import numpy as np

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

# ------------------------------------------------------------------------------

def plot_parameter_posteriors(
    svi_result,
    p_true=None,
    r_true=None,
    n_r_examples=5,
    n_rows=2,
    n_cols=3,
    n_points=200,
    figsize=None
):
    """
    Plot posterior distributions versus ground truth values for model
    parameters.

    This function creates a grid of plots showing: 1. The Beta posterior
    distribution for the p parameter (success probability) 2. Multiple Gamma
    posterior distributions for randomly selected r parameters
       (overdispersion parameters)

    For each parameter, the posterior distribution is plotted along with a
    vertical line indicating the true value (if provided).

    Parameters
    ----------
    svi_result : numpyro.infer.SVI
        Results from stochastic variational inference containing the optimized
        variational parameters (alpha and beta) for both p and r distributions.
    p_true : float, optional
        True value of the p parameter (success probability). If provided, will
        be shown as a vertical line on the plot.
    r_true : array-like, optional
        Array of true values for the r parameters (overdispersion). If provided,
        will be shown as vertical lines on the respective plots.
    n_r_examples : int, default=5
        Number of r parameters to randomly sample and plot. Should be less than
        the total number of subplots available (n_rows * n_cols - 1).
    n_rows : int, default=2
        Number of rows in the subplot grid.
    n_cols : int, default=3
        Number of columns in the subplot grid.
    n_points : int, default=200
        Number of points to use when plotting the posterior distributions.
        Higher values give smoother curves but increase computation time.
    figsize : tuple of float, optional
        Figure dimensions (width, height) in inches. If None, defaults to
        (4*n_cols, 4*n_rows).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the grid of posterior plots.

    Notes
    -----
    - The first subplot always shows the p parameter posterior
    - Subsequent subplots show randomly selected r parameter posteriors
    - Unused subplots (if any) are removed from the figure
    - The posterior distributions are colored blue
    - Ground truth values (if provided) are shown as red dashed lines
    """
    # Set figure size if not provided
    if figsize is None:
        figsize = (4*n_cols, 4*n_rows)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot p posterior

    # Extract alpha and beta parameters from svi_result
    alpha_p = svi_result.params['alpha_p']
    beta_p = svi_result.params['beta_p']
    # Define x values for plotting
    x_p = np.linspace(0, 1, n_points)
    # Define posterior p
    posterior_p = stats.beta.pdf(x_p, alpha_p, beta_p)
    
    # Plot posterior p
    axes[0].plot(x_p, posterior_p, 'b-', label='posterior')
    if p_true is not None:  
        # Plot true p
        axes[0].axvline(p_true, color='r', linestyle='--', label='ground truth')
    # Set title
    axes[0].set_title('p parameter')
    # Set x label
    axes[0].set_xlabel('value')
    # Set y label
    axes[0].set_ylabel('density')
    # Add legend
    axes[0].legend()
    
    # Extract alpha and beta parameters from svi_result
    alpha_r = svi_result.params['alpha_r']
    beta_r = svi_result.params['beta_r']
    
    # Randomly select r parameters to plot
    r_indices = np.random.choice(len(r_true), size=n_r_examples, replace=False)
    
    # Loop through r parameters
    for i, idx in enumerate(r_indices, 1):
        if i >= len(axes):  # Skip if we run out of subplot space
            break

        # Define x range based on the posterior mean if r_true not provided
        r_mean = alpha_r[idx] / beta_r[idx]
        x_max = r_true[idx]*2 if r_true is not None else r_mean*2
        x_r = np.linspace(0, x_max, n_points)

        # Define posterior r
        posterior_r = stats.gamma.pdf(x_r, alpha_r[idx], scale=1/beta_r[idx])
        
        # Plot posterior r
        axes[i].plot(x_r, posterior_r, 'b-', label='posterior')
        if r_true is not None:
            # Plot true r
            axes[i].axvline(
                r_true[idx], color='r', linestyle='--', label='ground truth'
            )
        # Set title
        axes[i].set_title(f'r parameter {idx}')
        # Set x label
        axes[i].set_xlabel('parameter value')
        # Set y label
        axes[i].set_ylabel('density')
        # Add legend
        axes[i].legend()
    
    # Remove any unused subplots
    for i in range(n_r_examples + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    return fig