import matplotlib
import seaborn as sns
import numpy as np
import mpmath
import scipy
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp


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


def matplotlib_style(return_colors=True, return_palette=True, **kwargs):
    """
    Assigns the plotting style for matplotlib generated figures.

    Parameters
    ----------
    return_colors : bool
        If True, a dictionary of the colors is returned. Default is True.
    return_palette: bool
        If True, a sequential color palette is returned. Default is True.
    """
    # Define the matplotlib styles.
    rc = {
        # Axes formatting
        "axes.facecolor": "#E6E6EF",
        "axes.edgecolor": "#ffffff",  # 5b5b5b" ,
        "axes.labelcolor": "#000000",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
        "axes.linewidth": 0.15,
        "axes.grid": True,

        # Formatting of lines and points.
        "lines.linewidth": 0.5,
        "lines.dash_capstyle": "butt",
        "patch.linewidth": 0.25,
        "lines.markeredgecolor": '#E6E6EF',
        "lines.markeredgewidth": 0.5,

        # Grid formatting
        "grid.linestyle": '-',
        "grid.linewidth": 0.5,
        "grid.color": "#FFFFFF",

        # Title formatting
        "axes.titlesize": 8,
        # "axes.titleweight": 700,
        "axes.titlepad": 3,
        "axes.titlelocation": "center",

        # Axes label formatting.
        "axes.labelpad": 0,
        # "axes.labelweight": 700,
        "xaxis.labellocation": "center",
        "yaxis.labellocation": "center",
        "axes.labelsize": 8,
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.03,

        # Legend formatting
        "legend.fontsize": 6,
        "legend.labelspacing": 0.25,
        "legend.title_fontsize": 6,
        "legend.frameon": True,
        "legend.edgecolor": "#5b5b5b",

        # Tick formatting
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.width": 0.25,
        "ytick.major.width": 0.25,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,

        # General Font styling
        "font.family": "Roboto",
        # "font.weight": 400,  # Weight of all fonts unless overriden.
        "font.style": "normal",

        # Higher-order things
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "errorbar.capsize": 1,
        "savefig.bbox": "tight",
        "mathtext.default": "regular",
    }
    matplotlib.style.use(rc)

    # Load the colors and palettes.
    colors, palette = get_colors(**kwargs)
    sns.set_palette(palette)

    # Determine what, if anything should be returned
    out = []
    if return_colors == True:
        out.append(colors)
    if return_palette == True:
        out.append(palette)

    if len(out) == 1:
        return out[0]
    else:
        return out

# %% ===========================================================================


# define a np.frompyfunc that allows us to evaluate the sympy.mp.math.hyp1f1
np_log_hyp = np.frompyfunc(
    lambda x, y, z: mpmath.ln(mpmath.hyp1f1(x, y, z, zeroprec=1024)), 3, 1
)

# ------------------------------------------------------------------------------

g_k_dict = {
    "g_0": 1,
    "g_1": 1/12,
    "g_2": 1/288,
    "g_3": - 139/51840,
    "g_4": - 571/2488320,
    "g_5": 163879/209018880,
    "g_6": 5246819/75246796800,
    "g_7": - 534703531/902961561600,
    "g_8": - 4483131259/86684309913600,
    "g_9": 432261921612371/514904800886784000,
    "g_10": 6232523202521089/86504006548979712000,
    "g_11": - 25834629665134204969/13494625021640835072000,
    "g_12": - 1579029138854919086429/9716130015581401251840000,
    "g_13": 746590869962651602203151/116593560186976815022080000,
    "g_14": 1511513601028097903631961/2798245444487443560529920000,
    "g_15": - 8849272268392873147705987190261/299692087104605205332754432000000,
    "g_16": - 142801712490607530608130701097701/57540880724084199423888850944000000,
    "g_17": 2355444393109967510921431436000087153/13119320805091197468646658015232000000,
    "g_18": 2346608607351903737647919577082115121863/155857531164483425927522297220956160000000,
    "g_19": - 2603072187220373277150999431416562396331667/1870290373973801111130267566651473920000000,
    "g_20": - 73239727426811935976967471475430268695630993/628417565655197173339769902394895237120000000
}

# ------------------------------------------------------------------------------


def log_scaled_gamma_function(z, n_terms, coefficients):
    """
    Computes the logarithm of the scaled gamma function log(Γ*(z)) up to a
    specified number of terms.

    The scaled gamma function is given by: 
    Γ*(z) ∼ e^(-z) * z^z * (2π/z)^(1/2) * Σ (g_k / z^k), for k=0 to ∞

    Parameters
    ----------
    - z (float or complex): The argument of the scaled gamma function.
    - n_terms (int): The number of terms to include in the series approximation.
    - coefficients (dict): A dictionary containing the coefficients g_k.

    Returns
    -------
    - float or complex: The computed value of the logarithm of the scaled gamma
      function log(Γ*(z)).
    """
    # Compute the logarithm of the prefactor log(e^(-z) * z^z * (2π/z)^(1/2))
    log_prefactor = -z + z * np.log(z) + 0.5 * np.log(2 * np.pi / z)

    # Compute the series sum Σ (g_k / z^k) for k=0 to n_terms-1
    series_sum = sum(coefficients[f"g_{k}"] / (z ** k) for k in range(n_terms))

    # Compute the logarithm of the series sum
    log_series_sum = np.log(series_sum)

    # Compute the logarithm of the scaled gamma function log(Γ*(z))
    log_gamma_star = log_prefactor + log_series_sum

    return log_gamma_star

# ------------------------------------------------------------------------------


def log_large_a_b_kummer(a, b, z, n_terms=20, coefficients=g_k_dict):
    """
    Computes the logarithm of the Kummer function M(a, b, z) for large values of
    the parameters a and b using the function M(a, b, z) as

    M(a, b, z) = e^(νz) * (Γ*(b) / Γ*(a)) * 
                 (1 + ((1-ν) * (1 + 6ν^2z^2)) / (12a) + O(1/min(a^2, b^2)))

    Parameters
    ----------
    - a (float): The parameter 'a' in the function.
    - b (float): The parameter 'b' in the function.
    - z (float or complex): The argument 'z' in the function.
    - n_terms (int): The number of terms to include in the series approximation
      for the scaled gamma functions.
    - coefficients (dict): A dictionary containing the coefficients g_k for the
      scaled gamma functions.

    Returns
    -------
    - float or complex: The computed value of the logarithm of the function M(a,
      b, z).
    """
    # Define the parameter ν = a / b
    nu = a / b

    # Compute the logarithm of the exponential term log(e^(νz)) = νz
    log_exp_term = nu * z

    # Compute the logarithms of the scaled gamma functions log(Γ*(b)) and
    # log(Γ*(a))
    log_gamma_star_b = log_scaled_gamma_function(b, n_terms, coefficients)
    log_gamma_star_a = log_scaled_gamma_function(a, n_terms, coefficients)

    # Compute the logarithm of the ratio Γ*(b) / Γ*(a)
    log_gamma_ratio = log_gamma_star_b - log_gamma_star_a

    # Compute the correction term (1 + ((1-ν) * (1 + 6ν^2z^2)) / (12a))
    correction_term = 1 + ((1 - nu) * (1 + 6 * nu**2 * z**2)) / (12 * a)

    # Compute the logarithm of the correction term
    log_correction_term = np.log(correction_term)

    # Compute the logarithm of the function M(a, b, z)
    log_M_value = log_exp_term + log_gamma_ratio + log_correction_term

    return log_M_value

# ------------------------------------------------------------------------------


def log_kummer_integral(a, b, z):
    """
    Computes the logarithm of the Kummer function M(a, b, z) using an integral
    representation.

    This function calculates log(M(a, b, z)) where M is the Kummer function
    (confluent hypergeometric function of the first kind) using the integral
    representation:

    M(a, b, z) = [Γ(b) / (Γ(a) * Γ(b-a))] * 
                 ∫[0 to 1] e^(zt) * t^(b-a-1) * (1-t)^(a-1) dt

    The function computes this integral in log space to avoid numerical overflow
    for large parameter values.

    Parameters:
    -----------
    a : float
        The first parameter of the Kummer function.
    b : float
        The second parameter of the Kummer function. Must be greater than a.
    z : float or complex
        The argument of the Kummer function.

    Returns:
    --------
    float or complex
        The logarithm of the Kummer function M(a, b, z).

    Notes:
    ------
    - This method can be numerically stable for a wider range of parameters
      compared to direct computation.
    - The function uses scipy's quad for numerical integration and special
      functions for log-gamma calculations.
    - Care should be taken when b - a is not a positive integer, as the integral
      might not converge.
    """
    # Define the log of the integrand for the integral
    def log_integrand(t):
        log_term1 = z * t  # log(e^(zt)) = zt
        log_term2 = (a - 1) * np.log(t)
        log_term3 = (b - a - 1) * np.log(1 - t)
        return log_term1 + log_term2 + log_term3

    # Perform the integration over the interval [0, 1]
    integral, error = quad(lambda t: np.exp(log_integrand(t)), 0, 1)
    log_integral = np.log(integral)

    # Calculate the log of the prefactor: log(Gamma(b) / (Gamma(a) * Gamma(b-a)))
    log_prefactor = gammaln(b) - (gammaln(a) + gammaln(b - a))

    # Return the log of the result
    return log_prefactor + log_integral

# ------------------------------------------------------------------------------


def log_p_m(mRNA, kp_on, kp_off, rm, gm=1, log_M_func=np_log_hyp):
    '''
    Computes the log probability lnP(m) for a two-state promoter model, i.e. the
    probability of having m mRNA.

    Parameters
    ----------
    mRNA : float.
        mRNA copy number at which evaluate the probability.
    kp_on : float.
        rate of activation of the promoter in the chemical master equation
    kp_off : float.
        rate of deactivation of the promoter in the chemical master equation
    rm : float.
        production rate of the mRNA
    gm : float.
        1 / half-life time for the mRNA.
    log_M_func : function.
        Function to evaluate the log Kummer function.

    Returns
    -------
    log probability lnP(m)
    '''
    # Convert the mRNA copy number to a  numpy array
    mRNA = np.array(mRNA)

    # Compute the probability
    lnp = scipy.special.gammaln(kp_on / gm + mRNA) \
        - scipy.special.gammaln(mRNA + 1) \
        - scipy.special.gammaln((kp_off + kp_on) / gm + mRNA) \
        + scipy.special.gammaln((kp_off + kp_on) / gm) \
        - scipy.special.gammaln(kp_on / gm) \
        + mRNA * np.log(rm / gm) \
        + log_M_func(
            kp_on / gm + mRNA,
            (kp_off + kp_on) / gm + mRNA,
            -rm / gm
    )

    return lnp

# ------------------------------------------------------------------------------


def two_state_log_probability(
    mRNA_values,
    kp_on,
    kp_off,
    rm,
    log_M_func=np_log_hyp,
    log_M_approx=log_kummer_integral,
    gm=1,
):
    """
    Evaluates the log probability for a range of mRNA values.

    Parameters
    ----------
    - mRNA_values (array-like): The range of mRNA values to evaluate.
    - kp_on (float): Rate of activation of the promoter.
    - kp_off (float): Rate of deactivation of the promoter.
    - rm (float): Production rate of the mRNA.
    - gm (float): 1 / half-life time for the mRNA.
    - n_terms (int): The number of terms to include in the series approximation
      for the scaled gamma functions.
    - coefficients (dict): A dictionary containing the coefficients g_k for the
      scaled gamma functions.
    - M_func (function): The default function to compute M(a, b, z).
    - log_large_a_b_kummer (function): The fallback function to compute the log
      of M(a, b, z).

    Returns
    -------
    - list: The computed log probabilities for the range of mRNA values.
    """
    log_probs = []
    use_large_a_b = False

    for mRNA in mRNA_values:
        if use_large_a_b:
            # Use log_large_a_b_kummer for the rest of the evaluations
            log_prob = log_p_m(
                mRNA, kp_on, kp_off, rm, gm, log_M_approx
            )
        else:
            try:
                # Compute the log probability using the default M_func
                log_prob = float(
                    log_p_m(mRNA, kp_on, kp_off, rm, gm, log_M_func)
                )

                # Check if the result is infinity or NaN
                if np.isinf(log_prob) or np.isnan(log_prob) or log_prob > 0:
                    raise ValueError("Result is infinity or NaN")

            except:
                # Switch to using log_large_a_b_kummer if M_func fails
                log_prob = log_p_m(
                    mRNA, kp_on, kp_off, rm, gm, log_M_approx
                )
                use_large_a_b = True

        log_probs.append(log_prob)

    return log_probs

# ------------------------------------------------------------------------------


def two_state_neg_binom_log_probability(
    mRNA_values,
    kp_on,
    kp_off,
    rm,
    gm=1,
):
    """
    Compute the log probability of the negative binomial approximation of the
    two-state promoter model.

    Parameters
    ----------
    mRNA_values : array-like
        The range of mRNA values to evaluate.
    kp_on : float
        Rate of activation of the promoter.
    kp_off : float
        Rate of deactivation of the promoter.
    rm : float
        Production rate of the mRNA.
    gm : float, optional
        1 / half-life time for the mRNA. Default is 1.

    Returns
    -------
    numpy.ndarray
        The computed log probabilities for the range of mRNA values.

    Notes
    -----
    This function uses the negative binomial approximation of the two-state
    promoter model, which is valid when kp_off >> 1.
    """
    # Define the parameters
    n = mRNA_values
    k = kp_on / gm
    p = 1 / (1 + rm / kp_off)
    # Compute the log probability using scipy.stats.nbinom
    log_prob = scipy.stats.nbinom.logpmf(n, k, p)
    return log_prob
