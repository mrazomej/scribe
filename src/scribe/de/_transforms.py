"""Coordinate transformations for compositional data analysis.

This module provides transformations between different coordinate systems for
compositional data:

- **ALR** (Additive Log-Ratio): asymmetric, (D-1)-dimensional
- **CLR** (Centered Log-Ratio): symmetric, D-dimensional (constrained:
  entries sum to zero)
- **ILR** (Isometric Log-Ratio): orthonormal, (D-1)-dimensional

All transformations are implemented efficiently without materializing large
transformation matrices, keeping memory and compute at O(D) or O(kD) for
low-rank structures.
"""

from typing import Tuple

import jax.numpy as jnp


# --------------------------------------------------------------------------
# ALR (Additive Log-Ratio) → CLR (Centered Log-Ratio) Transform
# --------------------------------------------------------------------------


def alr_to_clr(z_alr: jnp.ndarray) -> jnp.ndarray:
    """Map ALR coordinates to CLR (centered log-ratio) coordinates.

    The CLR transformation centers the log-ratios::

        CLR_i = log(rho_i) - (1/D) * sum_j log(rho_j)

    This is computed from ALR by embedding ``[z_alr, 0]`` and then centering.

    Parameters
    ----------
    z_alr : jnp.ndarray
        ALR coordinates, shape ``(..., D-1)``.

    Returns
    -------
    clr : jnp.ndarray
        CLR coordinates, shape ``(..., D)``, sum to 0 along last axis.

    Examples
    --------
    >>> z_alr = jnp.array([0.5, -0.3])
    >>> clr = alr_to_clr(z_alr)
    >>> assert jnp.allclose(clr.sum(), 0.0)
    """
    # Embed: [z_alr, 0] to get D-dimensional log-ratios
    z_full = jnp.concatenate(
        [z_alr, jnp.zeros_like(z_alr[..., :1])], axis=-1
    )
    # Center: subtract mean to get CLR
    return z_full - z_full.mean(axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# Exact diagonal after ALR → CLR (no big matrices)
# --------------------------------------------------------------------------


def _exact_diag_after_centering(d_alr: jnp.ndarray) -> jnp.ndarray:
    """Compute exact diagonal of centered covariance.

    For ``d_full = [d_alr, 0]`` and centering matrix
    ``C = I - (1/D) 1 1^T``, the diagonal of ``C diag(d_full) C^T`` is::

        (C diag(d) C^T)_{ii} = d_full_i (1 - 2/D) + (1/D^2) sum_j d_full_j

    This formula gives exact marginal variances after centering without
    materializing the ``D x D`` centering matrix.

    Parameters
    ----------
    d_alr : jnp.ndarray
        Diagonal entries of covariance in ALR space, shape ``(D-1,)``.

    Returns
    -------
    d_clr : jnp.ndarray
        Exact diagonal in CLR space, shape ``(D,)``.

    Notes
    -----
    The reference component (last) has zero variance in ALR, which
    contributes to all components after centering.
    """
    # Embed: d_full = [d_alr, 0] (reference has zero variance)
    d_full = jnp.concatenate([d_alr, jnp.array([0.0])], axis=0)
    D = d_full.shape[0]

    # Exact formula for diagonal after centering
    mean_sum_over_D2 = jnp.sum(d_full) / (D * D)
    d_clr = d_full * (1.0 - 2.0 / D) + mean_sum_over_D2

    return d_clr


# --------------------------------------------------------------------------
# Transform low-rank Gaussian from ALR to CLR space (exact)
# --------------------------------------------------------------------------


def transform_gaussian_alr_to_clr(
    mu_alr: jnp.ndarray, W_alr: jnp.ndarray, d_alr: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Transform low-rank Gaussian from ALR to CLR space (exact).

    For ``z_alr ~ N(mu, Sigma)`` with ``Sigma = W W^T + diag(d)``, compute
    exact parameters of ``z_clr = H z_alr`` where ``H = C E``.

    Here:

    - **E** embeds (D-1)-dimensional ALR to D dimensions by appending 0.
    - **C** = ``I - (1/D) 1 1^T`` is the centering matrix.

    The transformation preserves the low-rank structure and gives exact
    marginal variances (crucial for gene-level inference).

    Parameters
    ----------
    mu_alr : jnp.ndarray
        Mean in ALR space, shape ``(D-1,)``.
    W_alr : jnp.ndarray
        Low-rank factor in ALR space, shape ``(D-1, k)``.
    d_alr : jnp.ndarray
        Diagonal in ALR space, shape ``(D-1,)``.

    Returns
    -------
    mu_clr : jnp.ndarray
        Mean in CLR space, shape ``(D,)``.
    W_clr : jnp.ndarray
        Low-rank factor in CLR space, shape ``(D, k)``.
    d_clr : jnp.ndarray
        Exact diagonal in CLR space, shape ``(D,)``.

    Examples
    --------
    >>> mu_alr = jnp.array([1.0, -0.5])
    >>> W_alr = jnp.ones((2, 3)) * 0.1
    >>> d_alr = jnp.array([0.1, 0.2])
    >>> mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(
    ...     mu_alr, W_alr, d_alr
    ... )
    >>> assert mu_clr.shape == (3,)
    >>> assert jnp.allclose(mu_clr.sum(), 0.0)
    """
    # Number of components in CLR space
    k = W_alr.shape[-1]

    # 1. Transform mean: embed [mu_alr, 0], then center
    mu_full = jnp.concatenate([mu_alr, jnp.array([0.0])], axis=-1)
    mu_clr = mu_full - mu_full.mean()

    # 2. Transform W: embed [W_alr; 0_row], then apply centering to each col
    W_full = jnp.concatenate([W_alr, jnp.zeros((1, k))], axis=0)  # (D, k)
    # Centering matrix C = I - (1/D) 1 1^T applied to W gives W_clr = C W_full
    W_clr = W_full - W_full.mean(axis=0, keepdims=True)

    # 3. Transform diagonal: exact formula for marginal variances after
    #    centering
    d_clr = _exact_diag_after_centering(d_alr)

    return mu_clr, W_clr, d_clr


# --------------------------------------------------------------------------
# Build ILR basis (orthonormal basis of CLR subspace)
# --------------------------------------------------------------------------


def build_ilr_basis(D: int) -> jnp.ndarray:
    """Build orthonormal ILR basis using Gram-Schmidt on CLR subspace.

    The ILR basis ``V`` is a ``(D-1) x D`` matrix with orthonormal rows that
    are orthogonal to the CLR constraint (``sum = 0``).

    This implementation uses the Helmert contrast (sequential binary
    partition), where the i-th basis vector separates the first ``i+1``
    components from the rest.

    Parameters
    ----------
    D : int
        Number of components (genes).

    Returns
    -------
    V : jnp.ndarray
        ILR basis matrix, shape ``(D-1, D)``, with orthonormal rows.

    Notes
    -----
    The rows of ``V`` are orthonormal: ``V V^T = I_{D-1}``.
    The rows sum to zero: ``V 1 = 0``.

    Examples
    --------
    >>> V = build_ilr_basis(4)
    >>> assert V.shape == (3, 4)
    >>> assert jnp.allclose(V @ V.T, jnp.eye(3))
    >>> assert jnp.allclose(V.sum(axis=1), 0.0)
    """
    # Use sequential binary partition (Helmert contrast)
    V = jnp.zeros((D - 1, D))
    for i in range(D - 1):
        # i-th basis vector separates first i+1 components from rest
        # Normalization ensures unit length
        norm_factor = jnp.sqrt((i + 1) * (i + 2))
        V = V.at[i, : i + 1].set(1.0 / norm_factor)
        V = V.at[i, i + 1].set(-(i + 1) / norm_factor)
    return V


# --------------------------------------------------------------------------
# ILR balance vector for a two-group partition (pathway vs complement)
# --------------------------------------------------------------------------


def build_ilr_balance(
    gene_set_indices: jnp.ndarray, D: int
) -> jnp.ndarray:
    """Build the ILR-normalized balance vector for a pathway-vs-complement partition.

    Given a gene set (pathway) and the total number of genes, constructs a
    unit-norm, sum-to-zero vector that represents the compositional balance
    between the geometric means of the two groups.

    The balance vector entries are::

        v_g = +sqrt(n_- / (n_+ * (n_+ + n_-)))   if g in pathway
        v_g = -sqrt(n_+ / (n_- * (n_+ + n_-)))   if g not in pathway

    where ``n_+`` = pathway size, ``n_-`` = complement size, ``n_+ + n_- = D``.

    Parameters
    ----------
    gene_set_indices : jnp.ndarray
        Integer indices of genes in the pathway, shape ``(n_+,)``.
        Must be non-empty and a proper subset of ``{0, ..., D-1}``.
    D : int
        Total number of genes (components).

    Returns
    -------
    v : jnp.ndarray
        ILR balance vector, shape ``(D,)``, with unit norm and zero sum.

    Raises
    ------
    ValueError
        If the gene set is empty or contains all genes.

    Notes
    -----
    The balance vector is a positive scalar multiple of the unnormalized CLR
    contrast used in ``build_balance_contrast``. The scaling factor is
    ``alpha = sqrt(n_+ * n_- / (n_+ + n_-))``. Because the scaling is
    positive, the empirical lfsr is identical under either normalization.

    Examples
    --------
    >>> v = build_ilr_balance(jnp.array([0, 2]), D=5)
    >>> assert v.shape == (5,)
    >>> assert jnp.allclose(jnp.sum(v**2), 1.0)
    >>> assert jnp.allclose(jnp.sum(v), 0.0)
    """
    gene_set_indices = jnp.asarray(gene_set_indices, dtype=jnp.int32)
    n_plus = gene_set_indices.shape[0]
    n_minus = D - n_plus
    if n_plus == 0 or n_minus == 0:
        raise ValueError(
            f"Gene set must be a proper subset: got n_+={n_plus}, "
            f"n_-={n_minus}, D={D}."
        )

    # Positive coefficient for pathway genes, negative for complement
    pos_val = jnp.sqrt(n_minus / (n_plus * (n_plus + n_minus)))
    neg_val = -jnp.sqrt(n_plus / (n_minus * (n_plus + n_minus)))

    # Start with complement value everywhere, then overwrite pathway positions
    v = jnp.full(D, neg_val)
    v = v.at[gene_set_indices].set(pos_val)
    return v


# --------------------------------------------------------------------------
# Pathway-aware sequential binary partition (SBP) basis
# --------------------------------------------------------------------------


def build_pathway_sbp_basis(
    gene_set_indices: jnp.ndarray, D: int
) -> jnp.ndarray:
    """Build a pathway-aware ILR basis via sequential binary partition.

    Constructs a ``(D-1) x D`` orthonormal matrix organized into three blocks:

    - **Row 0**: the pathway-vs-complement balance (from ``build_ilr_balance``).
    - **Rows 1 to n_+-1**: Helmert basis within the pathway genes
      (``n_+ - 1`` rows with non-zero entries only at pathway positions).
    - **Rows n_+ to D-2**: Helmert basis within the complement genes
      (``n_- - 1`` rows with non-zero entries only at complement positions).

    The three blocks are mutually orthogonal (disjoint support) and internally
    orthonormal (Helmert construction), giving an overall orthonormal basis.

    Parameters
    ----------
    gene_set_indices : jnp.ndarray
        Integer indices of genes in the pathway, shape ``(n_+,)``.
        Must be non-empty, a proper subset of ``{0, ..., D-1}``, and
        the pathway must contain at least 2 genes (otherwise the
        within-pathway subspace is empty).
    D : int
        Total number of genes (components).

    Returns
    -------
    V_sbp : jnp.ndarray
        Pathway-aware SBP basis, shape ``(D-1, D)``, with orthonormal rows.

    Raises
    ------
    ValueError
        If the gene set is empty, contains all genes, or has fewer than
        2 genes (making the within-pathway subspace trivial).

    Notes
    -----
    The within-pathway Helmert rows have zero entries for complement genes
    and vice versa. Row 0 coincides with ``build_ilr_balance``.

    Examples
    --------
    >>> V = build_pathway_sbp_basis(jnp.array([0, 2, 4]), D=6)
    >>> assert V.shape == (5, 6)
    >>> assert jnp.allclose(V @ V.T, jnp.eye(5), atol=1e-6)
    >>> assert jnp.allclose(V.sum(axis=1), 0.0, atol=1e-6)
    """
    gene_set_indices = jnp.asarray(gene_set_indices, dtype=jnp.int32)
    n_plus = gene_set_indices.shape[0]
    n_minus = D - n_plus
    if n_plus == 0 or n_minus == 0:
        raise ValueError(
            f"Gene set must be a proper subset: got n_+={n_plus}, "
            f"n_-={n_minus}, D={D}."
        )
    if n_plus < 2:
        raise ValueError(
            f"Pathway must contain at least 2 genes for a non-trivial "
            f"within-pathway subspace, got n_+={n_plus}."
        )

    V_sbp = jnp.zeros((D - 1, D))

    # Row 0: pathway-vs-complement balance
    V_sbp = V_sbp.at[0].set(build_ilr_balance(gene_set_indices, D))

    # Complement indices (sorted for deterministic ordering)
    all_indices = jnp.arange(D)
    mask = jnp.ones(D, dtype=bool)
    mask = mask.at[gene_set_indices].set(False)
    complement_indices = all_indices[mask]

    # Within-pathway Helmert basis (n_+ - 1 rows over n_+ positions)
    H_pathway = build_ilr_basis(n_plus)  # (n_+-1, n_+)
    for i in range(n_plus - 1):
        row_idx = 1 + i
        for j in range(n_plus):
            V_sbp = V_sbp.at[row_idx, gene_set_indices[j]].set(H_pathway[i, j])

    # Within-complement Helmert basis (n_- - 1 rows over n_- positions)
    if n_minus >= 2:
        H_complement = build_ilr_basis(n_minus)  # (n_--1, n_-)
        for i in range(n_minus - 1):
            row_idx = n_plus + i
            for j in range(n_minus):
                V_sbp = V_sbp.at[row_idx, complement_indices[j]].set(
                    H_complement[i, j]
                )

    return V_sbp


# --------------------------------------------------------------------------
# Project CLR → ILR (orthonormal basis)
# --------------------------------------------------------------------------


def clr_to_ilr(clr: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Project CLR coordinates to ILR coordinates.

    ILR coordinates are orthonormal projections of CLR onto a
    (D-1)-dimensional basis.  This transformation is an isometry
    (preserves distances).

    Parameters
    ----------
    clr : jnp.ndarray
        CLR coordinates, shape ``(..., D)``.
    V : jnp.ndarray
        ILR basis matrix, shape ``(D-1, D)``.

    Returns
    -------
    ilr : jnp.ndarray
        ILR coordinates, shape ``(..., D-1)``.

    Examples
    --------
    >>> clr = jnp.array([0.5, -0.2, -0.3])
    >>> V = build_ilr_basis(3)
    >>> ilr = clr_to_ilr(clr, V)
    >>> assert ilr.shape == (2,)
    """
    # Matrix multiply: (..., D) @ (D, D-1) -> (..., D-1)
    return clr @ V.T


# --------------------------------------------------------------------------
# Project ILR → CLR (orthonormal basis)
# --------------------------------------------------------------------------


def ilr_to_clr(ilr: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Project ILR coordinates back to CLR coordinates.

    This is the inverse of ``clr_to_ilr``, recovering the CLR representation
    from orthonormal ILR coordinates.

    Parameters
    ----------
    ilr : jnp.ndarray
        ILR coordinates, shape ``(..., D-1)``.
    V : jnp.ndarray
        ILR basis matrix, shape ``(D-1, D)``.

    Returns
    -------
    clr : jnp.ndarray
        CLR coordinates, shape ``(..., D)``, sum to 0 along last axis.

    Examples
    --------
    >>> V = build_ilr_basis(3)
    >>> ilr = jnp.array([0.3, 0.1])
    >>> clr = ilr_to_clr(ilr, V)
    >>> assert clr.shape == (3,)
    >>> assert jnp.allclose(clr.sum(), 0.0)
    """
    # Matrix multiply: (..., D-1) @ (D-1, D) -> (..., D)
    return ilr @ V
