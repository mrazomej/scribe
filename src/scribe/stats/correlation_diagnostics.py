"""Diagnostics for the latent correlation structure of low-rank-Gaussian fits.

The PLN and LNM models share a low-rank-plus-diagonal covariance
parameterisation of their per-cell latent prior:

Σ = 𝑊 𝑊ᵗ + diag(𝑑),

with 𝑊 ∈ ℝ^{Gₑₑ𝚏𝚏}ˣᵏ (rows = genes, cols = latent factors) and
𝑑 ∈ ℝ^{Gₑₑ𝚏𝚏}}_{>0}. The gene-gene correlation matrix derived from this Σ is the
analytic, model-implied object that the heatmap visualisation renders.

This module provides three pure-function diagnostics that depend only on
(𝑊, 𝑑):

* `library_size_direction` — the unit vector 𝒆 ∈ ℝᵏ whose image 𝑊𝒆 is closest
  to 𝟏_{Gₑₑ𝚏𝚏}. Identifies the "all-genes-shift-together" axis hidden in 𝑊.
* `correlation_residual` — the gene-gene correlation matrix after projecting out
  one or more latent directions (the library-size axis or the top principal
  components of 𝑊ᵗ𝑊).
* `summarize_correlation_structure` — a one-call diagnostic that computes
  alignment / concentration / share quantities, the latent eigenspectrum, and
  off-diagonal correlation quantiles before vs. after projection. Returns a
  dict; optionally prints a rich-formatted report.

These are model-agnostic over the *structure* of 𝑊: they work identically for
PLN (where rows index G log-rate genes), LNM (where rows index G−1 ALR
composition logits), LNMVCP (same as LNM at this layer), and across both Laplace
and VAE inference paths. The caller is responsible for passing 𝑊 and 𝑑
extracted from whichever results object it has.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np


def library_size_direction(W: jnp.ndarray) -> jnp.ndarray:
    """Latent-space unit vector whose image is closest to 𝟏₍𝗚₍ₑₑ𝚏𝚏₎₎.

    Returns the least-squares solution to
    𝑾𝒆 ≈ 𝟏₍𝗚₍ₑₑ𝚏𝚏₎₎, normalized to unit length. Geometrically, this is the
    direction in the column space of 𝑾 that best represents the
    "all-genes-shift-together" pattern characteristic of per-cell library-size
    variation.

    Parameters
    ----------
    W : jnp.ndarray, shape (Gₑₑ𝚏𝚏, k)

    Returns
    -------
    jnp.ndarray, shape (k,)
        Unit-norm direction in latent space.
    """
    ones_G = jnp.ones(W.shape[0], dtype=W.dtype)
    WtW = W.T @ W
    Wt1 = W.T @ ones_G
    e_raw = jnp.linalg.solve(WtW, Wt1)
    return e_raw / jnp.maximum(jnp.linalg.norm(e_raw), 1e-30)


def correlation_residual(
    W: jnp.ndarray,
    d: Optional[jnp.ndarray] = None,
    *,
    method: str = "library_size",
    n_components: int = 1,
    include_diagonal_d: bool = False,
) -> jnp.ndarray:
    """Correlation matrix after projecting out latent direction(s).

    Two strategies, both designed to surface the *secondary* block
    structure of Σ = W Wᵗ + diag(d) when one
    or more latent directions are dominated by nuisance signal
    (typically library-size in scRNA-seq):

    * ``method="library_size"`` (default): project out the unit direction
      returned by :func:`library_size_direction` — the column-space direction
      whose image is closest to 𝟏₍G₍ₑₑ𝚏𝚏₎₎. Always uses one component.
    * ``method="pc"``: project out the top ``n_components`` principal directions
      of Wᵗ W in latent space.


    Parameters
    ----------
    W : jnp.ndarray, shape ``(G_eff, k)``
    d : jnp.ndarray, shape ``(G_eff,)``, optional
        Diagonal residual. Required when ``include_diagonal_d=True``.
    method : {"library_size", "pc"}, default "library_size"
    n_components : int, default 1
        Used only for ``method="pc"``.
    include_diagonal_d : bool, default False
        Whether to add ``diag(d)`` to the residual covariance before
        normalising. Excluding gives a "factor-only" correlation that
        surfaces blocks more sharply.

    Returns
    -------
    jnp.ndarray, shape ``(G_eff, G_eff)``
    """
    if method == "library_size":
        U = library_size_direction(W)[:, None]  # (k, 1)
    elif method == "pc":
        n = max(1, int(n_components))
        WtW = W.T @ W
        eigvals, eigvecs = jnp.linalg.eigh(WtW)
        U = eigvecs[:, -n:]  # (k, n)
    else:
        raise ValueError(
            f"method must be 'library_size' or 'pc'; got {method!r}"
        )

    # P_perp = I_k - U U^T; sigma_perp = W P_perp P_perp^T W^T.
    # Use the identity sigma_perp = W W^T - (W U)(W U)^T to avoid
    # forming the full G x G projector.
    WU = W @ U  # (G_eff, n)
    sigma_perp = W @ W.T - WU @ WU.T

    if include_diagonal_d:
        if d is None:
            raise ValueError("include_diagonal_d=True requires the d array.")
        sigma_perp = sigma_perp + jnp.diag(d)

    std = jnp.sqrt(jnp.maximum(jnp.diag(sigma_perp), 1e-30))
    return sigma_perp / (std[:, None] * std[None, :])


def summarize_correlation_structure(
    W: jnp.ndarray,
    d: Optional[jnp.ndarray] = None,
    *,
    space_label: str = "log-rate space",
    model_label: str = "PLN",
    n_top_eig: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Print and return a diagnostic summary of the latent correlation structure.

    Computes the four quantities that together characterise whether the
    correlation matrix is dominated by a library-size axis, by a
    single non-library latent direction, or by genuinely diffuse multi-
    block structure — plus an automatic suggestion of which projection
    mode (if any) is likely to clean up the heatmap.

    Parameters
    ----------
    W : jnp.ndarray, shape ``(G_eff, k)``
    d : jnp.ndarray, shape ``(G_eff,)``, optional
        Diagonal residual variance. Used only for the
        ``include_diagonal_d=True`` mode of
        :func:`correlation_residual`; the default summary path
        does not include it.
    space_label : str, default "log-rate space"
        Space the correlation matrix lives in (e.g. ``"ALR space"``
        for LNM). Surfaces in the printed header.
    model_label : str, default "PLN"
        Model identifier (e.g. ``"PLN VAE"``, ``"LNM Laplace"``).
        Surfaces in the printed header.
    n_top_eig : int, default 10
        Number of top eigenvalues of ``W^T W`` to report.
    verbose : bool, default True
        Print a rich-formatted summary to the console.

    Returns
    -------
    dict
        Diagnostics dict (see source for keys).
    """
    W_np = np.asarray(W)
    G_eff, k = W_np.shape
    ones_G = np.ones(G_eff, dtype=W_np.dtype)

    # --- Library-size axis diagnostics --------------------------------
    e = np.asarray(library_size_direction(jnp.asarray(W_np)))
    We = W_np @ e

    We_norm = float(np.linalg.norm(We))
    cos_We_1G = float(We @ ones_G / max(We_norm * np.sqrt(G_eff), 1e-30))
    We_mean = float(We.mean())
    We_std = float(We.std())
    We_concentration = abs(We_mean) / We_std if We_std > 1e-30 else float("inf")
    We_rms = We_norm / np.sqrt(G_eff)

    # --- Latent eigenspectrum -----------------------------------------
    WtW = W_np.T @ W_np
    eigvals = np.linalg.eigvalsh(WtW)[::-1]
    eig_total = float(eigvals.sum()) or 1.0
    eig_fractions = eigvals / eig_total

    sigma_factor_norm = float(np.linalg.norm(W_np @ W_np.T))
    library_axis_share = (We_norm**2) / max(sigma_factor_norm, 1e-30)

    op_norm = float(np.sqrt(eigvals[0])) if eigvals.size else 1.0
    fro_norm = float(np.sqrt(eigvals.sum()))
    effective_rank = (fro_norm / op_norm) ** 2 if op_norm > 0 else float("nan")

    # --- Off-diagonal correlation quantiles ---------------------------
    def _offdiag_quantiles(C: np.ndarray) -> Dict[str, float]:
        mask = ~np.eye(C.shape[0], dtype=bool)
        vals = C[mask]
        return {
            "min": float(vals.min()),
            "p25": float(np.quantile(vals, 0.25)),
            "p50": float(np.quantile(vals, 0.5)),
            "p75": float(np.quantile(vals, 0.75)),
            "max": float(vals.max()),
        }

    # Build full / library / pc1 correlation matrices.
    if d is None:
        d_jnp = jnp.zeros(G_eff, dtype=W_np.dtype)
    else:
        d_jnp = jnp.asarray(d)
    sigma_full = jnp.asarray(W_np) @ jnp.asarray(W_np).T + jnp.diag(d_jnp)
    std_full = jnp.sqrt(jnp.maximum(jnp.diag(sigma_full), 1e-30))
    C_full = np.asarray(sigma_full / (std_full[:, None] * std_full[None, :]))

    C_lib = np.asarray(
        correlation_residual(
            jnp.asarray(W_np),
            d_jnp,
            method="library_size",
            include_diagonal_d=False,
        )
    )
    C_pc1 = np.asarray(
        correlation_residual(
            jnp.asarray(W_np),
            d_jnp,
            method="pc",
            n_components=1,
            include_diagonal_d=False,
        )
    )
    q_full = _offdiag_quantiles(C_full)
    q_lib = _offdiag_quantiles(C_lib)
    q_pc1 = _offdiag_quantiles(C_pc1)

    n_top = min(int(n_top_eig), int(eigvals.size))
    result: Dict[str, Any] = {
        "n_genes_effective": int(G_eff),
        "n_latent_factors": int(k),
        "cos_We_1G": cos_We_1G,
        "We_concentration": We_concentration,
        "We_rms": We_rms,
        "library_axis_share": float(library_axis_share),
        "eigenvalues": eigvals.tolist(),
        "eigenvalue_fractions": eig_fractions.tolist(),
        "effective_rank": float(effective_rank),
        "offdiag_quantiles_full": q_full,
        "offdiag_quantiles_after_library": q_lib,
        "offdiag_quantiles_after_pc1": q_pc1,
    }

    if not verbose:
        return result

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Header.
    header_lines = [
        f"Model    : {model_label}",
        f"Space    : {space_label}",
        f"G_eff    : {G_eff}",
        f"k        : {k} latent factors",
    ]
    console.print(
        Panel(
            "\n".join(header_lines),
            title="Correlation structure summary",
            expand=False,
        )
    )

    # Library-size axis table.
    lib_table = Table(
        title="Library-size axis  (W column-space projection of 1_G)",
        show_header=True,
        header_style="bold",
    )
    lib_table.add_column("Quantity")
    lib_table.add_column("Value", justify="right")
    lib_table.add_column("Interpretation")
    lib_table.add_row(
        "cos(We, 1_G)",
        f"{cos_We_1G:.3f}",
        "1.0 = perfectly all-genes-shift",
    )
    lib_table.add_row(
        "|mean(We)| / std(We)",
        f"{We_concentration:.2f}" if np.isfinite(We_concentration) else "inf",
        "high → uniform loadings across genes",
    )
    lib_table.add_row(
        "||We|| / sqrt(G)",
        f"{We_rms:.3f}",
        "RMS gene loading on the axis",
    )
    lib_table.add_row(
        "share of ||W W^T||_F",
        f"{library_axis_share * 100:.1f}%",
        "rank-1 fraction of model covariance",
    )
    console.print(lib_table)

    # Eigenspectrum table.
    eig_table = Table(
        title=f"Latent eigenspectrum  (top {n_top} of W^T W)",
        show_header=True,
        header_style="bold",
    )
    eig_table.add_column("PC", justify="right")
    eig_table.add_column("Eigenvalue", justify="right")
    eig_table.add_column("Fraction", justify="right")
    eig_table.add_column("Cumulative", justify="right")
    cum = 0.0
    for i in range(n_top):
        cum += float(eig_fractions[i])
        eig_table.add_row(
            f"{i+1}",
            f"{float(eigvals[i]):.3g}",
            f"{float(eig_fractions[i]) * 100:.1f}%",
            f"{cum * 100:.1f}%",
        )
    console.print(eig_table)
    console.print(
        f"Effective rank (Frobenius/spectral): "
        f"[bold]{effective_rank:.2f}[/bold]"
    )

    # Off-diagonal correlation table.
    corr_table = Table(
        title="Off-diagonal correlation distribution",
        show_header=True,
        header_style="bold",
    )
    corr_table.add_column("Variant")
    corr_table.add_column("min", justify="right")
    corr_table.add_column("p25", justify="right")
    corr_table.add_column("p50", justify="right")
    corr_table.add_column("p75", justify="right")
    corr_table.add_column("max", justify="right")
    for label, q in (
        ("full", q_full),
        ("after library_size", q_lib),
        ("after PC1", q_pc1),
    ):
        corr_table.add_row(
            label,
            f"{q['min']:+.3f}",
            f"{q['p25']:+.3f}",
            f"{q['p50']:+.3f}",
            f"{q['p75']:+.3f}",
            f"{q['max']:+.3f}",
        )
    console.print(corr_table)

    # Suggestions block.
    suggestions = []
    if cos_We_1G > 0.9 and library_axis_share > 0.10:
        suggestions.append(
            "[yellow]Library-size axis is strongly aligned and "
            "carries >10% of the model covariance — "
            "[bold]subtract_direction='library_size'[/bold] is "
            "recommended.[/yellow]"
        )
    elif cos_We_1G > 0.9 and library_axis_share <= 0.10:
        suggestions.append(
            "[dim]Library-size axis is well-aligned but contributes "
            "<10% of the covariance — projection-out will subtly "
            "improve the heatmap. Try it as a sanity check.[/dim]"
        )
    elif cos_We_1G < 0.5:
        suggestions.append(
            "[dim]Library-size axis is poorly aligned with W's "
            "column space — most likely the capture term η has "
            "absorbed the library-size signal. No projection "
            "needed.[/dim]"
        )
    if eig_fractions.size > 0 and eig_fractions[0] > 0.30:
        suggestions.append(
            f"[yellow]PC1 of W^T W carries "
            f"{eig_fractions[0]*100:.0f}% of the latent variance — "
            "one direction dominates. "
            "[bold]subtract_direction='pc', n_pcs_to_remove=1[/bold] "
            "is likely to surface secondary structure.[/yellow]"
        )
    elif n_top >= 3 and eig_fractions[:3].sum() > 0.70:
        suggestions.append(
            f"[dim]Top 3 PCs explain "
            f"{eig_fractions[:3].sum()*100:.0f}% of the latent "
            "variance — try n_pcs_to_remove=2 or 3 for stronger "
            "denoising.[/dim]"
        )
    if suggestions:
        console.print()
        for s in suggestions:
            console.print(textwrap.fill(s, width=78))

    return result


__all__ = [
    "library_size_direction",
    "correlation_residual",
    "summarize_correlation_structure",
]
