"""UMAP projection plotting."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from jax import random
import scribe

from ._common import console
from .cache import _build_umap_cache_path
from .dispatch import _get_predictive_samples_for_plot
from ._interactive import (
    _create_or_validate_single_axis,
    plot_function,
)
from .gene_selection import _coerce_counts


def _scanpy_umap_pipeline_available(sc_module):
    """Return True if Scanpy exposes scale, PCA, neighbors, UMAP, and ingest.

    In minimal test environments, ``scanpy`` may be partially mocked; this
    gates the full graph-based UMAP path so a sklearn + umap-learn fallback
    remains available.

    Parameters
    ----------
    sc_module
        The imported ``scanpy`` module, or ``None``.

    Returns
    -------
    bool
        Whether the full pipeline API is present.
    """
    if sc_module is None:
        return False
    try:
        return (
            hasattr(sc_module.pp, "scale")
            and hasattr(sc_module.pp, "neighbors")
            and hasattr(sc_module.pp, "pca")
            and hasattr(sc_module.tl, "umap")
            and hasattr(sc_module.tl, "ingest")
        )
    except AttributeError:
        return False


def _apply_scale_to_synth(synth_norm, scale_meta):
    """Apply the same scaling used on experimental cells to synthetic counts.

    Parameters
    ----------
    synth_norm : ndarray
        Log-normalized synthetic matrix ``(n_cells, n_genes)``.
    scale_meta : dict
        Output from the experimental scaling step (``kind`` and parameters).

    Returns
    -------
    ndarray
        Scaled matrix ready for PCA or UMAP projection.
    """
    kind = scale_meta.get("kind", "none")
    if kind == "none":
        return synth_norm
    if kind == "scanpy":
        mean = scale_meta["mean"]
        std = scale_meta["std"]
        max_value = scale_meta.get("max_value")
        out = (synth_norm - mean) / std
        if max_value is not None:
            out = np.clip(out, -max_value, max_value)
        return out
    if kind == "sklearn":
        scaler = scale_meta["scaler"]
        max_value = scale_meta.get("max_value")
        out = scaler.transform(synth_norm)
        if max_value is not None:
            out = np.clip(out, -max_value, max_value)
        return out
    return synth_norm


@plot_function(
    suffix="umap",
    save_label="UMAP plot",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_umap(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    force_refit=False,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Plot UMAP projection of experimental and synthetic data.

    The embedding follows a Scanpy-style preprocessing workflow before fitting
    UMAP: log-normalization, optional highly-variable gene selection,
    :func:`scanpy.pp.scale` (z-score per gene with optional clipping),
    :func:`scanpy.pp.pca`, :func:`scanpy.pp.neighbors`, and
    :func:`scanpy.tl.umap` on the neighborhood graph. Synthetic counts use the
    same scaling statistics and are placed in the embedding with
    :func:`scanpy.tl.ingest`. When Scanpy is unavailable or partially mocked,
    the code falls back to scikit-learn scaling/PCA and ``umap-learn`` on the
    PC matrix (previous behavior, plus scaling).

    Parameters
    ----------
    results
        Fitted SCRIBE results object (posterior predictive sampling uses a
        gene subset consistent with HVG selection).
    counts
        Observed count matrix aligned with ``results``.
    ctx
        Plot context (configuration paths, output dirs).
    viz_cfg : dict-like, optional
        May include ``umap_opts`` with keys such as ``n_neighbors``,
        ``min_dist``, ``spread``, ``use_scale``, ``scale_max_value``,
        ``use_pca``, ``pca_n_comps``, ``hvg_n_top_genes``, ``cache_umap``,
        ``n_ppc_samples``, and color overrides.
    force_refit : bool, optional
        If True, ignore cache and recompute the embedding.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.
    fig, axes, ax : optional
        Matplotlib objects for embedding the plot (see ``plot_function``).

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axes, and metadata.
    """
    console.print("[dim]Plotting UMAP projection...[/dim]")

    umap_opts = viz_cfg.get("umap_opts", {}) if viz_cfg is not None else {}

    try:
        import umap
    except ImportError:
        console.print(
            "[bold red]❌ ERROR:[/bold red] [red]umap-learn is not installed.[/red]"
            " [yellow]Install it with: pip install umap-learn[/yellow]"
        )
        return None

    n_neighbors = umap_opts.get("n_neighbors", 15)
    min_dist = umap_opts.get("min_dist", 0.1)
    spread = umap_opts.get("spread", 1.0)
    n_components = umap_opts.get("n_components", 2)
    random_state = umap_opts.get("random_state", 42)
    data_color = umap_opts.get("data_color", "dark_blue")
    synthetic_color = umap_opts.get("synthetic_color", "dark_red")
    cache_umap = umap_opts.get("cache_umap", True)
    target_sum = umap_opts.get("target_sum", 1e4)
    gene_filter_min_cells = int(umap_opts.get("gene_filter_min_cells", 3))
    use_hvg = bool(umap_opts.get("use_hvg", True))
    hvg_n_top_genes = int(umap_opts.get("hvg_n_top_genes", 4000))
    hvg_flavor = umap_opts.get("hvg_flavor", "seurat")
    use_pca = bool(umap_opts.get("use_pca", True))
    pca_n_comps = int(umap_opts.get("pca_n_comps", 50))
    n_ppc_samples = int(umap_opts.get("n_ppc_samples", 1))
    use_scale = bool(umap_opts.get("use_scale", True))
    scale_max_value = umap_opts.get("scale_max_value", 10.0)

    if force_refit:
        console.print(
            "[yellow]⚠️[/yellow] [yellow]Overwrite requested: forcing UMAP refit and cache overwrite[/yellow]"
        )

    try:
        import scanpy as sc
        import anndata as ad

        has_scanpy = True
    except ImportError:
        has_scanpy = False
        sc = None
        ad = None
        if use_hvg:
            console.print(
                "[yellow]⚠️[/yellow] [yellow]scanpy/anndata not available; skipping HVG selection and using all retained genes[/yellow]"
            )

    scanpy_full = _scanpy_umap_pipeline_available(sc) if has_scanpy else False

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        has_sklearn = True
    except ImportError:
        has_sklearn = False
        PCA = None
        StandardScaler = None
        if use_pca:
            console.print(
                "[yellow]⚠️[/yellow] [yellow]scikit-learn not available; skipping PCA and fitting UMAP directly on gene space[/yellow]"
            )

    def _normalize_log1p(matrix):
        matrix = np.array(matrix, dtype=np.float64)
        lib_sizes = matrix.sum(axis=1, keepdims=True)
        lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
        return np.log1p(matrix / lib_sizes * target_sum)

    try:
        colors = scribe.viz.colors
        if hasattr(colors, data_color):
            data_color = getattr(colors, data_color)
        if hasattr(colors, synthetic_color):
            synthetic_color = getattr(colors, synthetic_color)
    except (AttributeError, ImportError):
        pass

    cache_path = _build_umap_cache_path(ctx.cfg, cache_umap)
    data_cfg = ctx.cfg.data if hasattr(ctx.cfg, "data") else None
    subset_column = data_cfg.get("subset_column", None) if data_cfg else None
    subset_value = data_cfg.get("subset_value", None) if data_cfg else None

    current_params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "spread": spread,
        "n_components": n_components,
        "random_state": random_state,
        "target_sum": target_sum,
        "gene_filter_min_cells": gene_filter_min_cells,
        "use_hvg": use_hvg,
        "hvg_n_top_genes": hvg_n_top_genes,
        "hvg_flavor": hvg_flavor,
        "use_pca": use_pca,
        "pca_n_comps": pca_n_comps,
        "use_scale": use_scale,
        "scale_max_value": scale_max_value,
        "n_cells": counts.shape[0],
        "n_genes": counts.shape[1],
        "subset_column": subset_column,
        "subset_value": subset_value,
    }

    umap_reducer = None
    umap_embedding = None
    gene_mask = None
    hvg_mask = None
    pca_model = None
    adata_ref = None
    scale_meta = {"kind": "none"}
    pipeline_kind = "fallback"

    if cache_path and os.path.exists(cache_path) and not force_refit:
        console.print(
            f"[dim]Found cached UMAP at:[/dim] [cyan]{cache_path}[/cyan]"
        )
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            cached_params = cached.get("params", {})
            params_match = all(
                cached_params.get(k) == v for k, v in current_params.items()
            )
            has_preprocessing = "preprocessing" in cached

            if params_match and has_preprocessing:
                cached_pre = cached["preprocessing"]
                gene_mask = cached_pre.get("gene_mask")
                hvg_mask = cached_pre.get("hvg_mask")
                pca_model = cached_pre.get("pca_model")
                scale_meta = cached_pre.get("scale_meta", {"kind": "none"})
                adata_ref = cached_pre.get("adata_ref")
                pipeline_kind = cached_pre.get("pipeline_kind", "fallback")

                valid_masks = (
                    gene_mask is not None
                    and hvg_mask is not None
                    and len(gene_mask) == counts.shape[1]
                    and int(np.sum(gene_mask)) == len(hvg_mask)
                )

                if not valid_masks:
                    console.print(
                        "[yellow]⚠️[/yellow] [yellow]Cached preprocessing metadata invalid - will re-fit UMAP[/yellow]"
                    )
                    adata_ref = None
                elif pipeline_kind == "scanpy" and adata_ref is None:
                    console.print(
                        "[yellow]⚠️[/yellow] [yellow]Cached Scanpy pipeline missing reference AnnData - will re-fit UMAP[/yellow]"
                    )
                elif (
                    pipeline_kind == "fallback"
                    and cached.get("reducer") is None
                ):
                    console.print(
                        "[yellow]⚠️[/yellow] [yellow]Cached fallback pipeline missing reducer - will re-fit UMAP[/yellow]"
                    )
                else:
                    console.print(
                        "[green]✅[/green] [dim]Cache valid - loading UMAP from cache...[/dim]"
                    )
                    umap_reducer = cached.get("reducer")
                    umap_embedding = cached["embedding"]
            elif params_match and not has_preprocessing:
                console.print(
                    "[yellow]⚠️[/yellow] [yellow]Cache is from an older format (missing preprocessing metadata) - will re-fit UMAP[/yellow]"
                )
            else:
                console.print(
                    "[yellow]⚠️[/yellow] [yellow]Cache parameters mismatch - will re-fit UMAP[/yellow]"
                )
                console.print(f"[dim]   Cached:[/dim] {cached_params}")
                console.print(f"[dim]   Current:[/dim] {current_params}")
        except Exception as e:
            console.print(
                f"[yellow]⚠️[/yellow] [yellow]Failed to load cache: {e} - will re-fit UMAP[/yellow]"
            )

    # Graph-based Scanpy UMAP + ingest requires a real Scanpy install (not a stub).
    if (
        umap_embedding is not None
        and pipeline_kind == "scanpy"
        and not scanpy_full
    ):
        console.print(
            "[yellow]⚠️[/yellow] [yellow]Cached graph UMAP requires full Scanpy; "
            "will re-fit UMAP[/yellow]"
        )
        umap_embedding = None
        umap_reducer = None

    if umap_embedding is None:
        console.print(
            "[dim]Preparing experimental data with Scanpy-style preprocessing...[/dim]"
        )
        counts_np = np.asarray(_coerce_counts(counts), dtype=np.float64)

        detected_cells_per_gene = np.sum(counts_np > 0, axis=0)
        gene_mask = detected_cells_per_gene >= gene_filter_min_cells
        n_genes_retained = int(np.sum(gene_mask))
        if n_genes_retained == 0:
            console.print(
                "[yellow]⚠️[/yellow] [yellow]Gene filtering removed all genes; reverting to all genes[/yellow]"
            )
            gene_mask = np.ones(counts_np.shape[1], dtype=bool)
            n_genes_retained = counts_np.shape[1]

        console.print(
            f"[dim]Gene filter: kept {n_genes_retained}/{counts_np.shape[1]} genes "
            f"(min_cells={gene_filter_min_cells})[/dim]"
        )

        counts_gene_filtered = counts_np[:, gene_mask]
        counts_norm = _normalize_log1p(counts_gene_filtered)

        hvg_mask = np.ones(counts_norm.shape[1], dtype=bool)
        if use_hvg and counts_norm.shape[1] > 1 and has_scanpy:
            n_top_genes = min(hvg_n_top_genes, counts_norm.shape[1])
            if n_top_genes >= 1:
                console.print(
                    f"[dim]Selecting HVGs with scanpy.pp.highly_variable_genes "
                    f"(flavor={hvg_flavor}, n_top_genes={n_top_genes})...[/dim]"
                )
                adata_hvg = ad.AnnData(X=counts_norm.copy())
                sc.pp.highly_variable_genes(
                    adata_hvg,
                    n_top_genes=n_top_genes,
                    flavor=hvg_flavor,
                    inplace=True,
                )
                hvg_mask = np.asarray(
                    adata_hvg.var["highly_variable"], dtype=bool
                )
                if np.sum(hvg_mask) == 0:
                    console.print(
                        "[yellow]⚠️[/yellow] [yellow]HVG selection returned no genes; using all retained genes[/yellow]"
                    )
                    hvg_mask = np.ones(counts_norm.shape[1], dtype=bool)
        elif use_hvg and counts_norm.shape[1] <= 1:
            console.print(
                "[yellow]⚠️[/yellow] [yellow]Too few genes for HVG selection; using retained genes directly[/yellow]"
            )

        counts_embed = counts_norm[:, hvg_mask]
        console.print(
            f"[dim]Embedding feature space: {counts_embed.shape[1]} genes[/dim]"
        )

        # --- Per-gene scaling (Scanpy pp.scale) before PCA / neighbors ---
        scale_meta = {"kind": "none"}
        counts_scaled = counts_embed
        if use_scale and has_scanpy and scanpy_full:
            console.print(
                f"[dim]Scaling with scanpy.pp.scale "
                f"(max_value={scale_max_value})...[/dim]"
            )
            adata_scale = ad.AnnData(X=counts_embed.copy())
            adata_scale.var_names = [
                f"gene_{i}" for i in range(adata_scale.n_vars)
            ]
            sc.pp.scale(
                adata_scale,
                max_value=scale_max_value,
                zero_center=True,
            )
            counts_scaled = np.asarray(adata_scale.X, dtype=np.float64)
            scale_meta = {
                "kind": "scanpy",
                "mean": adata_scale.var["mean"].to_numpy(dtype=np.float64),
                "std": adata_scale.var["std"].to_numpy(dtype=np.float64),
                "max_value": scale_max_value,
            }
        elif use_scale and has_sklearn and StandardScaler is not None:
            console.print(
                f"[dim]Scaling with sklearn StandardScaler "
                f"(clip ±{scale_max_value})...[/dim]"
            )
            scaler = StandardScaler(with_mean=True, with_std=True)
            counts_scaled = scaler.fit_transform(counts_embed)
            if scale_max_value is not None:
                counts_scaled = np.clip(
                    counts_scaled, -scale_max_value, scale_max_value
                )
            scale_meta = {
                "kind": "sklearn",
                "scaler": scaler,
                "max_value": scale_max_value,
            }
        elif use_scale and not has_sklearn:
            console.print(
                "[yellow]⚠️[/yellow] [yellow]Cannot scale without scanpy or sklearn; using log-normalized matrix[/yellow]"
            )

        # --- Scanpy: PCA -> neighbors -> graph UMAP; else umap-learn on PCs ---
        did_scanpy_graph = False
        if scanpy_full and use_pca and has_sklearn:
            max_pcs = min(
                pca_n_comps,
                counts_scaled.shape[1],
                max(1, counts_scaled.shape[0] - 1),
            )
            if max_pcs >= 2:
                console.print(
                    "[dim]Fitting Scanpy graph UMAP (scale → PCA → neighbors → "
                    f"tl.umap; n_neighbors={n_neighbors}, min_dist={min_dist}, "
                    f"spread={spread})...[/dim]"
                )
                adata_ref = ad.AnnData(X=counts_scaled.copy())
                adata_ref.var_names = [
                    f"gene_{i}" for i in range(adata_ref.n_vars)
                ]
                sc.pp.pca(
                    adata_ref,
                    n_comps=max_pcs,
                    random_state=random_state,
                )
                sc.pp.neighbors(
                    adata_ref,
                    n_neighbors=n_neighbors,
                    n_pcs=max_pcs,
                    random_state=random_state,
                )
                sc.tl.umap(
                    adata_ref,
                    min_dist=min_dist,
                    spread=spread,
                    random_state=random_state,
                )
                umap_embedding = np.asarray(adata_ref.obsm["X_umap"])
                pipeline_kind = "scanpy"
                pca_model = None
                did_scanpy_graph = True
            else:
                console.print(
                    "[yellow]⚠️[/yellow] [yellow]Too few components/cells for PCA; "
                    "using fallback UMAP on scaled gene space[/yellow]"
                )

        if not did_scanpy_graph:
            embedding_input = counts_scaled
            pca_model = None
            if use_pca and has_sklearn and counts_scaled.shape[1] > 1:
                max_pcs = min(
                    pca_n_comps,
                    counts_scaled.shape[1],
                    max(1, counts_scaled.shape[0] - 1),
                )
                if max_pcs >= 2:
                    console.print(
                        f"[dim]Running PCA before UMAP (n_components={max_pcs})...[/dim]"
                    )
                    pca_model = PCA(
                        n_components=max_pcs, random_state=random_state
                    )
                    embedding_input = pca_model.fit_transform(counts_scaled)
                else:
                    console.print(
                        "[yellow]⚠️[/yellow] [yellow]Too few components/cells for PCA; fitting UMAP directly on scaled gene space[/yellow]"
                    )
            elif not use_pca:
                console.print(
                    "[dim]Skipping PCA; fitting UMAP on scaled log-expression[/dim]"
                )

            console.print(
                f"[dim]Fitting UMAP on experimental data "
                f"(n_neighbors={n_neighbors}, min_dist={min_dist})...[/dim]"
            )

            umap_reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                random_state=random_state,
            )
            umap_embedding = umap_reducer.fit_transform(embedding_input)
            pipeline_kind = "fallback"
            adata_ref = None

        if cache_path:
            console.print(
                f"[dim]💾 Saving UMAP cache to:[/dim] [cyan]{cache_path}[/cyan]"
            )
            try:
                cache_data = {
                    "reducer": umap_reducer,
                    "embedding": umap_embedding,
                    "params": current_params,
                    "preprocessing": {
                        "gene_mask": gene_mask,
                        "hvg_mask": hvg_mask,
                        "pca_model": pca_model,
                        "scale_meta": scale_meta,
                        "adata_ref": adata_ref,
                        "pipeline_kind": pipeline_kind,
                    },
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                console.print(
                    "[green]✅[/green] [dim]UMAP cache saved successfully[/dim]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]⚠️[/yellow] [yellow]Failed to save cache: {e}[/yellow]"
                )

    console.print(
        f"[dim]Generating {n_ppc_samples} posterior predictive sample(s) for "
        f"synthetic dataset...[/dim]"
    )

    # Use the exact same feature space for synthetic PPC generation as for
    # UMAP fitting: detected genes followed by optional HVG selection.
    # This avoids sampling thousands of genes that are later discarded before
    # projection and keeps runtime aligned with `hvg_n_top_genes`.
    detected_gene_indices = np.where(gene_mask)[0]
    umap_gene_indices = detected_gene_indices[np.asarray(hvg_mask, dtype=bool)]
    results_sub = results[umap_gene_indices]

    console.print(
        f"[dim]Gene-subsetted results to {len(umap_gene_indices)} UMAP genes "
        f"(detected={int(gene_mask.sum())}, total={len(gene_mask)})[/dim]"
    )

    all_umap_synthetic = []
    for i in range(n_ppc_samples):
        sample_arr = _get_predictive_samples_for_plot(
            results_sub,
            rng_key=random.PRNGKey(42 + i),
            n_samples=1,
            store_samples=False,
            counts=counts,
        )

        synth_i = np.array(sample_arr[0, :, :], dtype=np.float64)
        synth_norm = _normalize_log1p(synth_i)
        # Results were already subset to UMAP feature genes above.
        synth_scaled = _apply_scale_to_synth(synth_norm, scale_meta)

        if pipeline_kind == "scanpy" and adata_ref is not None and scanpy_full:
            adata_synth = ad.AnnData(X=synth_scaled.copy())
            adata_synth.var_names = adata_ref.var_names.copy()
            sc.tl.ingest(adata_synth, adata_ref, embedding_method="umap")
            proj = np.asarray(adata_synth.obsm["X_umap"])
        else:
            synth_embed = synth_scaled
            if pca_model is not None:
                synth_embed = pca_model.transform(synth_embed)
            proj = umap_reducer.transform(synth_embed)

        all_umap_synthetic.append(proj)

        if (i + 1) % max(1, n_ppc_samples // 5) == 0 or i == n_ppc_samples - 1:
            console.print(
                f"[dim]  PPC sample {i + 1}/{n_ppc_samples} projected[/dim]"
            )

    umap_synthetic = np.concatenate(all_umap_synthetic, axis=0)

    console.print("[dim]Creating overlay plot...[/dim]")

    synth_alpha = 0.6 if n_ppc_samples == 1 else max(0.05, 0.6 / n_ppc_samples)
    synth_size = (
        1 if n_ppc_samples == 1 else max(0.2, 1.0 / np.sqrt(n_ppc_samples))
    )

    fig, ax = _create_or_validate_single_axis(
        fig=fig,
        ax=ax,
        axes=axes,
        figsize=figsize or (6.0, 6.0),
    )

    # Use color= not c= so a single RGB/RGBA tuple is not mistaken for per-point
    # values (see Matplotlib scatter UserWarning for c= with length-3/4
    # sequences).
    ax.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        color=data_color,
        alpha=0.6,
        s=1,
        label="experimental data",
    )

    ax.scatter(
        umap_synthetic[:, 0],
        umap_synthetic[:, 1],
        color=synthetic_color,
        alpha=synth_alpha,
        s=synth_size,
        label=f"synthetic data ({n_ppc_samples} sample{'s' if n_ppc_samples > 1 else ''})",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Projection: Experimental vs Synthetic Data")
    ax.legend(loc="best")

    fig.tight_layout()

    return fig, [ax], 1
