"""Per-gene Gaussianity diagnostics for ALR/CLR samples.

The differential expression analysis in this package assumes that the
marginal ALR (and therefore CLR) distribution of each gene is
well-approximated by a Gaussian.  When this assumption fails --- e.g.
because of skewness, heavy tails, or multimodality --- the
``norm.cdf``-based lfsr/PEFP calculations lose calibration.

This module provides a single, fully vectorized function that computes
per-feature diagnostic statistics in one pass over an (N, D) sample
matrix.  All operations use pure JAX and run entirely on GPU.

Diagnostics computed
--------------------
- **Skewness** ``S_g``: Third standardised moment.  Gaussian: 0.
- **Excess kurtosis** ``K_g``: Fourth standardised moment minus 3.
  Gaussian: 0.  Positive values indicate heavy tails.
- **Jarque-Bera statistic** ``JB_g``:
  ``JB = (N / 6) * (S^2 + K^2 / 4)``.  Under the null (Gaussianity),
  ``JB ~ chi2(2)`` asymptotically.
- **Jarque-Bera p-value**: ``1 - chi2.cdf(JB, df=2)``.  Small values
  indicate departure from Gaussianity.

Suggested descriptive thresholds
---------------------------------
- |skewness| > 0.5  : moderate skewness
- |excess kurtosis| > 1.0  : moderate tail weight
- JB statistic : continuous score for ranking genes by non-Gaussianity

Note: The JB p-value is retained for convenience but should **not** be
thresholded with frequentist multiple-testing corrections (BH, Bonferroni,
etc.) in a Bayesian workflow.  Use skewness/kurtosis directly as
descriptive flags.

References
----------
Jarque, C.M. & Bera, A.K. (1987). A test for normality of observations
and regression residuals. *International Statistical Review*, 55(2),
163--172.
"""

from typing import Dict

import jax.numpy as jnp
from jax.scipy.stats import chi2

# ------------------------------------------------------------------------------
# Per-feature Gaussianity diagnostics
# ------------------------------------------------------------------------------


def gaussianity_diagnostics(
    samples: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Per-feature Gaussianity diagnostics on an (N, D) sample matrix.

    Computes skewness, excess kurtosis, and the Jarque-Bera test
    statistic for each column (feature / gene) of the input matrix.
    All operations are fully vectorized JAX and run on GPU.

    Parameters
    ----------
    samples : jnp.ndarray, shape (N, D)
        Sample matrix to test.  Rows are independent draws, columns are
        features (typically ALR-transformed gene expression).  Must have
        ``N >= 2``.

    Returns
    -------
    dict
        Dictionary with the following keys, each of shape ``(D,)``:

        - **skewness** : Sample skewness (third standardised moment).
          A Gaussian has skewness 0.
        - **kurtosis** : Sample excess kurtosis (fourth standardised
          moment minus 3).  A Gaussian has excess kurtosis 0.  Positive
          values indicate heavier tails than the Gaussian.
        - **jarque_bera** : Jarque-Bera test statistic,
          ``JB = (N / 6) * (S^2 + K^2 / 4)``.
        - **jb_pvalue** : Asymptotic p-value from comparing JB to a
          chi-squared distribution with 2 degrees of freedom.  Small
          values (e.g. < 0.05) suggest non-Gaussianity.

    Notes
    -----
    The computation uses a single pass over the data:

    1. Compute sample mean and standard deviation per feature.
    2. Standardise: ``z = (x - mu) / sigma``.
    3. Compute ``E[z^3]`` (skewness) and ``E[z^4] - 3`` (excess
       kurtosis) as vectorized reductions along the sample axis.
    4. Combine into the Jarque-Bera statistic and its p-value.

    Total cost is O(N * D) with constant memory overhead beyond the
    input, making it negligible relative to the SVD fitting step.

    A stability guard of ``1e-30`` on the standard deviation prevents
    division-by-zero for constant features (which produce skewness = 0,
    kurtosis = 0, JB = 0, p-value = 1).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>> from scribe.de import gaussianity_diagnostics
    >>> key = random.PRNGKey(0)
    >>> samples = random.normal(key, shape=(5000, 100))
    >>> diag = gaussianity_diagnostics(samples)
    >>> diag["skewness"].shape
    (100,)
    >>> # Most p-values should be > 0.05 for truly Gaussian data
    >>> float(jnp.mean(diag["jb_pvalue"] > 0.05))  # doctest: +SKIP
    0.95
    """
    # Number of samples along the first axis
    N = samples.shape[0]

    # ---- Step 1: Per-feature mean and standard deviation ----
    mu = jnp.mean(samples, axis=0)  # (D,)
    # Bessel-corrected std for unbiased variance estimate
    sigma = jnp.std(samples, axis=0, ddof=1)  # (D,)

    # ---- Step 2: Standardise (guard against zero variance) ----
    z = (samples - mu) / jnp.maximum(sigma, 1e-30)  # (N, D)

    # ---- Step 3: Skewness and excess kurtosis (vectorized) ----
    # Third and fourth standardised moments as reductions along axis 0
    skewness = jnp.mean(z**3, axis=0)  # (D,)
    kurtosis = jnp.mean(z**4, axis=0) - 3.0  # (D,)  excess

    # ---- Step 4: Jarque-Bera statistic and p-value ----
    jarque_bera = (N / 6.0) * (skewness**2 + kurtosis**2 / 4.0)  # (D,)
    # Asymptotic null: JB ~ chi2(2)
    jb_pvalue = 1.0 - chi2.cdf(jarque_bera, df=2)  # (D,)

    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "jarque_bera": jarque_bera,
        "jb_pvalue": jb_pvalue,
    }
