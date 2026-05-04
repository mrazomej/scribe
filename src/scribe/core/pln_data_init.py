"""Data-derived initializers for the Poisson-LogNormal (PLN) model.

This module bundles the data-driven computations that the public API
performs once, up front, when the user requests a PLN
(``model="pln"``) fit. The goal is to warm-start the linear-decoder VAE
with sensible initial conditions derived from the count matrix.

Why these quantities?
---------------------
- **Empirical log-mean bias** (``empirical_log_mean_from_counts``):
  anchors the decoder bias to the per-gene log-mean expression. Without
  this, the Poisson rates at step 0 are random-scale and the optimizer
  must discover the gene-level baselines from scratch.
- **PCA-based loadings init** (``pca_loadings_init``): initializes the
  decoder weight matrix W with the principal components of the
  log-count matrix. Without this, the optimizer must rediscover the
  leading PCs from random initialization.
- **Encoder standardization stats** (reused from
  :mod:`scribe.core.lnm_data_init`): z-standardizes the encoder input
  in the post-transform space for gradient conditioning.

Pseudocount choice
------------------
Both ``empirical_log_mean_from_counts`` and ``pca_loadings_init`` add a
pseudocount ``c`` before taking the logarithm:
``log(count + c)``.  The default ``c = 1`` (``log1p``) is standard and
makes ``log(u + 1) ≈ log(u)`` for ``u >> 1``, which covers most genes
retained after quality control. For very sparse datasets where many genes
have mean count < 1, alternative pseudocounts (``c = 0.5`` or data-derived
values) may be more appropriate. The choice is configurable via the
``pseudocount`` argument.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from .lnm_data_init import compute_encoder_standardization

# Re-export for a single PLN-init import path.
__all__ = [
    "empirical_log_mean_from_counts",
    "pca_loadings_init",
    "compute_encoder_standardization",
    "inject_pln_vae_data_init",
]


def empirical_log_mean_from_counts(
    counts,
    pseudocount: float = 1.0,
) -> jnp.ndarray:
    """Compute per-gene log-mean expression for decoder bias initialization.

    The decoder bias encodes the population mean log-rate ``mu_g``. We
    initialize it as ``mean(log(u_g + c))``, the per-gene mean of the
    log-counts (with a pseudocount ``c`` that stabilizes ``log`` for
    genes with zero counts in some cells).

    Note on the formula
    -------------------
    Earlier revisions of this helper computed ``log(mean(u_g) + c)``
    instead of ``mean(log(u_g + c))``. By Jensen's inequality the two
    are not equal — the former overestimates the latent log-rate mean
    on right-skewed gene-count distributions, by a factor that depends
    on ``Var[log u_g]``. Crucially, this helper is paired with
    :func:`pca_loadings_init`, which centers the *log*-counts by their
    *mean of logs* and then SVDs the centered matrix. Using
    ``mean(log(u_g + c))`` here aligns the bias initialization with the
    centering reference of the PCA loadings, so the decoder's initial
    output ``mu + W @ z = mu + 0`` lands on the same per-gene anchor
    the loadings were derived around. Without this alignment the bias
    and kernel inits would live in two slightly different log-spaces
    and the warm-up benefit would be partially negated.

    Parameters
    ----------
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix.
    pseudocount : float, default=1.0
        Additive pseudocount before log. Default ``1.0`` matches the
        standard ``log1p`` transform applied by
        :func:`pca_loadings_init` so that the two initializations live
        in the same log-space.

    Returns
    -------
    jnp.ndarray, shape (n_genes,)
        Per-gene mean of log-counts ``mean_c log(u_{c,g} + c)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> counts = jnp.array([[10, 5, 0], [12, 6, 1]])
    >>> bias = empirical_log_mean_from_counts(counts)
    >>> bias.shape
    (3,)
    """
    counts_np = np.asarray(counts, dtype=np.float32)
    if counts_np.ndim != 2:
        raise ValueError(
            "counts must be a 2-D matrix (n_cells, n_genes), "
            f"got rank {counts_np.ndim} array with shape {counts_np.shape}."
        )
    # ``mean(log(u + c))`` (not ``log(mean(u) + c)``) so this matches
    # the centering reference of ``pca_loadings_init`` and the two
    # initializations live in the same log-space.
    log_means = np.log(counts_np + pseudocount).mean(axis=0)
    return jnp.asarray(log_means, dtype=jnp.float32)


# Maximum number of cells used for PCA initialization.  Beyond this
# threshold, cells are randomly subsampled before computing the SVD.
# PCA loadings capture population-level covariance structure, so 50k
# cells is more than enough for a stable estimate while capping peak
# memory at ~50k × G × 4 bytes (float32).
_PCA_MAX_CELLS: int = 50_000


def pca_loadings_init(
    counts,
    latent_dim: int,
    pseudocount: float = 1.0,
    max_cells: int = _PCA_MAX_CELLS,
    random_state: int = 0,
) -> jnp.ndarray:
    """PCA-based initialization for the decoder weight matrix W.

    Computes a randomized truncated SVD of the centered log-count matrix
    and returns the top-k right singular vectors scaled by their singular
    values, suitable for initializing the linear decoder's kernel.

    This places the initial decoder loadings in the subspace of the k
    principal components of the log-count matrix, giving the optimizer a
    warm start on the covariance structure.

    Scalability
    -----------
    The implementation is designed for large scRNA-seq datasets:

    * **Randomized SVD** (Halko et al. 2011) is used instead of ARPACK.
      It is single-pass, O(n_cells × n_genes × k), numerically stable,
      and avoids ARPACK convergence failures on matrices with small
      spectral gaps.
    * **Cell subsampling**: When ``n_cells > max_cells``, a random subset
      of ``max_cells`` cells is used.  PCA loadings capture
      population-level covariance, so subsampling introduces negligible
      error while capping peak memory.
    * **float32**: All intermediate arrays are float32, halving memory
      relative to float64 (sufficient precision for an initialization).

    Parameters
    ----------
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix.
    latent_dim : int
        Number of latent dimensions (k).  Must be <= min(n_cells, n_genes).
    pseudocount : float, default=1.0
        Additive pseudocount before log (same as ``empirical_log_mean``).
    max_cells : int, default=50000
        Maximum number of cells to use.  If the dataset is larger, cells
        are randomly subsampled (without replacement) before computing
        the SVD.  Set to ``None`` or ``0`` to disable subsampling.
    random_state : int, default=0
        Seed for both the cell subsampling RNG and the randomized SVD
        internal projections, ensuring reproducibility.

    Returns
    -------
    jnp.ndarray, shape (n_genes, latent_dim)
        PCA-based initialization for the decoder kernel W.
        Columns are the leading right singular vectors scaled by
        singular_values / sqrt(n_cells_used).

    Notes
    -----
    The initialization follows ``W = V_k @ diag(S_k) / sqrt(N)`` where
    ``U_k S_k V_k^T`` is the rank-k truncated SVD of the centered
    log-count matrix.  This scaling ensures that ``W W^T`` approximates
    the sample covariance of the log-counts.
    """
    from sklearn.utils.extmath import randomized_svd

    counts_np = np.asarray(counts, dtype=np.float32)
    if counts_np.ndim != 2:
        raise ValueError(
            "counts must be a 2-D matrix (n_cells, n_genes), "
            f"got rank {counts_np.ndim} array with shape {counts_np.shape}."
        )
    n_cells, n_genes = counts_np.shape

    # Subsample cells if the dataset exceeds the memory threshold.
    if max_cells and n_cells > max_cells:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_cells, size=max_cells, replace=False)
        counts_np = counts_np[idx]
        n_cells = max_cells

    max_k = min(n_cells, n_genes) - 1
    if latent_dim > max_k:
        raise ValueError(
            f"latent_dim={latent_dim} exceeds max rank "
            f"min(n_cells, n_genes) - 1 = {max_k}."
        )

    # Compute log(counts + c) and center per gene (in-place to save memory).
    log_counts = np.log(counts_np + pseudocount)
    del counts_np
    gene_means = log_counts.mean(axis=0)
    log_counts -= gene_means[None, :]

    # Randomized SVD (Halko et al. 2011): single-pass, O(N*G*k),
    # numerically stable, and no ARPACK convergence issues.
    # Returns singular values in descending order (no reversal needed).
    _, s, vt = randomized_svd(
        log_counts, n_components=latent_dim, random_state=random_state
    )
    del log_counts

    # W = V_k @ diag(S_k) / sqrt(N) so that W W^T ≈ sample covariance.
    w_init = (vt.T * s[None, :]) / np.sqrt(n_cells)
    return jnp.asarray(w_init, dtype=jnp.float32)


def inject_pln_vae_data_init(
    model_config,
    counts,
    latent_dim: int,
    pseudocount: float = 1.0,
):
    """Return a copy of ``model_config`` with PLN data-init fields populated.

    This helper centralizes the transformation that ``scribe.api.fit``
    performs on a freshly-built ``ModelConfig`` when the user requests a
    PLN model. It computes the empirical log-mean bias, PCA-based
    loadings initialization, and per-feature encoder standardization
    stats from the count matrix and stashes them on ``model_config.vae``.

    Parameters
    ----------
    model_config : ModelConfig
        Freshly-built configuration.
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix.
    latent_dim : int
        Number of latent dimensions for PCA init.
    pseudocount : float, default=1.0
        Pseudocount for log transforms.

    Returns
    -------
    ModelConfig
        New ``ModelConfig`` with data-init fields filled in.  The
        original is not mutated.
    """
    vae = model_config.vae

    # Decoder bias from empirical log-mean expression.
    log_mean_bias = empirical_log_mean_from_counts(
        counts, pseudocount=pseudocount
    )

    # PCA-based loadings for W initialization.
    w_init = pca_loadings_init(
        counts, latent_dim=latent_dim, pseudocount=pseudocount
    )

    # Encoder standardization (reused from LNM infrastructure).
    stand_mean, stand_std = compute_encoder_standardization(
        counts, input_transform=vae.input_transform
    )

    new_vae = vae.model_copy(
        update={
            "empirical_log_mean_bias_init": log_mean_bias,
            "pca_loadings_init": w_init,
            "standardize_mean": stand_mean,
            "standardize_std": stand_std,
        }
    )
    return model_config.model_copy(update={"vae": new_vae})
