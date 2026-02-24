"""UMAP projection plotting."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from jax import random
import scribe

from ._common import console
from .cache import _build_umap_cache_path
from .config import _get_config_values
from .dispatch import _get_map_like_predictive_samples_for_plot


def plot_umap(results, counts, figs_dir, cfg, viz_cfg, force_refit=False):
    """
    Plot UMAP projection of experimental and synthetic data.

    The embedding follows a Scanpy-style preprocessing workflow before fitting
    UMAP. The same learned preprocessing is applied to synthetic counts before
    projection.
    """
    console.print("[dim]Plotting UMAP projection...[/dim]")

    umap_opts = viz_cfg.get("umap_opts", {})

    try:
        import umap
    except ImportError:
        console.print(
            "[bold red]❌ ERROR:[/bold red] [red]umap-learn is not installed.[/red]"
            " [yellow]Install it with: pip install umap-learn[/yellow]"
        )
        return

    n_neighbors = umap_opts.get("n_neighbors", 15)
    min_dist = umap_opts.get("min_dist", 0.1)
    n_components = umap_opts.get("n_components", 2)
    random_state = umap_opts.get("random_state", 42)
    data_color = umap_opts.get("data_color", "dark_blue")
    synthetic_color = umap_opts.get("synthetic_color", "dark_red")
    cache_umap = umap_opts.get("cache_umap", True)
    target_sum = umap_opts.get("target_sum", 1e4)
    gene_filter_min_cells = int(umap_opts.get("gene_filter_min_cells", 3))
    use_hvg = bool(umap_opts.get("use_hvg", True))
    hvg_n_top_genes = int(umap_opts.get("hvg_n_top_genes", 2000))
    hvg_flavor = umap_opts.get("hvg_flavor", "seurat")
    use_pca = bool(umap_opts.get("use_pca", True))
    pca_n_comps = int(umap_opts.get("pca_n_comps", 50))
    n_ppc_samples = int(umap_opts.get("n_ppc_samples", 50))

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

    try:
        from sklearn.decomposition import PCA
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        PCA = None
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

    cache_path = _build_umap_cache_path(cfg, cache_umap)
    data_cfg = cfg.data if hasattr(cfg, "data") else None
    subset_column = data_cfg.get("subset_column", None) if data_cfg else None
    subset_value = data_cfg.get("subset_value", None) if data_cfg else None

    current_params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "n_components": n_components,
        "random_state": random_state,
        "target_sum": target_sum,
        "gene_filter_min_cells": gene_filter_min_cells,
        "use_hvg": use_hvg,
        "hvg_n_top_genes": hvg_n_top_genes,
        "hvg_flavor": hvg_flavor,
        "use_pca": use_pca,
        "pca_n_comps": pca_n_comps,
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
                else:
                    console.print(
                        "[green]✅[/green] [dim]Cache valid - loading UMAP from cache...[/dim]"
                    )
                    umap_reducer = cached["reducer"]
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

    if umap_reducer is None:
        console.print(
            "[dim]Preparing experimental data with Scanpy-style preprocessing...[/dim]"
        )
        counts_np = np.array(counts, dtype=np.float64)

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

        embedding_input = counts_embed
        pca_model = None
        if use_pca and has_sklearn and counts_embed.shape[1] > 1:
            max_pcs = min(
                pca_n_comps,
                counts_embed.shape[1],
                max(1, counts_embed.shape[0] - 1),
            )
            if max_pcs >= 2:
                console.print(
                    f"[dim]Running PCA before UMAP (n_components={max_pcs})...[/dim]"
                )
                pca_model = PCA(
                    n_components=max_pcs, random_state=random_state
                )
                embedding_input = pca_model.fit_transform(counts_embed)
            else:
                console.print(
                    "[yellow]⚠️[/yellow] [yellow]Too few components/cells for PCA; fitting UMAP directly on gene space[/yellow]"
                )

        console.print(
            f"[dim]Fitting UMAP on experimental data "
            f"(n_neighbors={n_neighbors}, min_dist={min_dist})...[/dim]"
        )

        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        umap_embedding = umap_reducer.fit_transform(embedding_input)

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
        f"[dim]Generating {n_ppc_samples} predictive sample(s) for "
        f"synthetic dataset...[/dim]"
    )

    batch_size = umap_opts.get("batch_size", None)
    gene_mask_indices = np.where(gene_mask)[0]
    results_sub = results[gene_mask_indices]

    console.print(
        f"[dim]Gene-subsetted results to {int(gene_mask.sum())} detected "
        f"genes (out of {len(gene_mask)})[/dim]"
    )

    all_umap_synthetic = []
    for i in range(n_ppc_samples):
        sample_arr = _get_map_like_predictive_samples_for_plot(
            results_sub,
            rng_key=random.PRNGKey(42 + i),
            n_samples=1,
            cell_batch_size=batch_size or 1000,
            use_mean=True,
            store_samples=False,
            verbose=(i == 0),
            counts=counts,
        )

        synth_i = np.array(sample_arr[0, :, :], dtype=np.float64)
        synth_norm = _normalize_log1p(synth_i)
        synth_embed = synth_norm[:, hvg_mask]
        if pca_model is not None:
            synth_embed = pca_model.transform(synth_embed)

        all_umap_synthetic.append(umap_reducer.transform(synth_embed))

        if (i + 1) % max(1, n_ppc_samples // 5) == 0 or i == n_ppc_samples - 1:
            console.print(
                f"[dim]  PPC sample {i + 1}/{n_ppc_samples} projected[/dim]"
            )

    umap_synthetic = np.concatenate(all_umap_synthetic, axis=0)

    console.print("[dim]Creating overlay plot...[/dim]")

    synth_alpha = 0.6 if n_ppc_samples == 1 else max(0.05, 0.6 / n_ppc_samples)
    synth_size = 1 if n_ppc_samples == 1 else max(0.2, 1.0 / np.sqrt(n_ppc_samples))

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        c=data_color,
        alpha=0.6,
        s=1,
        label="experimental data",
    )

    ax.scatter(
        umap_synthetic[:, 0],
        umap_synthetic[:, 1],
        c=synthetic_color,
        alpha=synth_alpha,
        s=synth_size,
        label=f"synthetic data ({n_ppc_samples} sample{'s' if n_ppc_samples > 1 else ''})",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Projection: Experimental vs Synthetic Data")
    ax.legend(loc="best")

    plt.tight_layout()

    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}_umap.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved UMAP plot to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)
