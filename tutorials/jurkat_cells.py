import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modeling Assumptions for Single-Cell RNA-seq with `scribe`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this tutorial we walk through a series of modeling and inference choices that `scribe`'s Bayesian engine makes available for single-cell RNA-seq data. We use a public dataset from 10x Genomics ([Jurkat Cells](https://www.10xgenomics.com/datasets/jurkat-cells-1-standard-1-1-0)), described as

    - ~3,200 cells detected
    - Sequenced on Illumina HiSeq 2500 Rapid Run V2 with ~33,900 reads per cell

    Because this is a monoculture (one cell type), it lets us focus on the statistical modeling rather than biological heterogeneity. We will progressively build up from the simplest count model to structured variational families, checking at each step whether the added complexity improves the fit.

    Let us begin by importing the necessary packages.
    """)
    return


@app.cell
def _():
    # Import basic packages
    from pathlib import Path
    import pickle

    # Import functionality to clear GPU memory
    import gc
    from jax import clear_caches

    # Import our main package
    import scribe

    # Import useful tools
    import numpy as np

    # Import plotting packages
    import matplotlib.pyplot as plt
    import seaborn as sns
    import corner

    # Set our plotting style (totally optional)
    scribe.viz.matplotlib_style()
    return Path, clear_caches, corner, gc, np, pickle, plt, scribe, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Having loaded the packages, let us proceed to load the data.

    > **Note:** `scribe` borrows `scanpy`'s loading data functionality for `h5ad` and `mtx` files.
    """)
    return


@app.cell
def _(Path, scribe):
    # Load the local tutorial path configuration.
    import json
    with Path(__file__).with_name("tutorial_paths.local.json").open(
        "r", encoding="utf-8"
    ) as _f:
        _data_root = json.load(_f)["SCRIBE_TUTORIAL_DATA_ROOT"]

    # Build the dataset-specific path relative to the configured root.
    data_dir = Path(_data_root).expanduser() / "jurkat_cells"

    # Load the data 
    adata = scribe.data_loader.load_and_preprocess_anndata(
        data_dir, return_jax=False
    )

    adata
    return adata, data_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before fitting any model, let us look at one basic summary of the data: the total UMI count per cell (library size). Most analysis pipelines treat this quantity as a nuisance and "normalize" it away by scaling every cell to an arbitrary target (often 10,000 UMIs). `scribe` takes a different approach: it explicitly models the per-cell capture probability that generates the library-size variation in the first place. So the empirical cumulative distribution function (ECDF) below is not just a quality-control plot—it previews the kind of cell-to-cell variability the model will need to account for.
    """)
    return


@app.cell
def _(adata, np, plt, sns):
    # Total UMI per cell; .A1 flattens a sparse-matrix row sum to a 1-D array
    _library_sizes = np.asarray(adata.X.sum(axis=1)).ravel()

    _fig, _ax = plt.subplots(figsize=(5, 4))
    sns.ecdfplot(_library_sizes, ax=_ax)
    _ax.set_xlabel('total UMI count per cell')
    _ax.set_ylabel('ECDF')
    _ax.set_ylim(-0.01, 1.01)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ECDF shows roughly a tenfold range in library size—about one order of magnitude of spread. In a Jurkat monoculture, that variation is much more plausibly technical (sequencing and capture depth) than biological. The rest of this notebook walks through how `scribe` treats that kind of depth variation inside the generative model: we start with the simplest count model, then progressively layer in the pieces that make depth explicit rather than washing it out with ad hoc scaling in preprocessing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting the simplest `scribe` model (no explicit capture)

    We begin with the smallest model `scribe` implements. The motivation is not a vague “overdispersion ⇒ pick an NB” heuristic. In the standard two-state (bursty) promoter model, the steady-state mRNA count for a gene follows a negative binomial distribution under well-characterized biophysical assumptions (see, e.g., Raj *et al.*, 2006). That is the starting point: a concrete biophysical model whose algebra lands on a per-gene negative binomial for the counts we observe.

    Concretely, let $m_g$ be the latent mRNA count for gene $g$ in a cell. The generative story is:

    $$
    m_g \sim \operatorname{NegBinom}(r_g, p),
    $$

    where $r_g$ is gene-specific (controlling how bursty or variable a gene is) and $p$ is shared across genes (related to burst size). This is not the only parameterization `scribe` offers—you will see alternatives later—but it is the simplest place to begin.

    `scribe`’s most basic inference strategy uses a mean-field variational approximation. In practical terms, this means the approximate posterior treats each parameter as independently varying: it does not represent gene–gene correlations. The results are best interpreted as marginals (per-gene summaries, per-cell quantities, shared parameters) rather than as a full joint picture of co-expression. We will relax this independence assumption later in the tutorial.

    For this first fit we set `variable_capture=False`, meaning we are not yet adding a per-cell capture/sequencing-depth layer. In effect, we treat the observed UMI counts $u_g$ as if they were on the same scale as the latent transcript counts $m_g$—no thinning or rescaling step. This is obviously the wrong assumption, but it serves a pedagogical purpose: it lets us isolate what the NB piece is doing before we introduce the technical sampling story.

    If you are used to Scanpy or Seurat:

    - Most workflows fix depth first (normalize, scale, log-transform) and then cluster or run differential expression on the transformed matrix.
    - Here we take the opposite order: start with the simplest count model, then add depth explicitly as a model ingredient via `variable_capture=True`.

    In short, `variable_capture=False` is a stepping stone, not a biological claim. It lets us see the NB layer in isolation before mixing in the capture model.

    We also pass `early_stopping={...}` so training does not run indefinitely. The `enabled` flag turns the mechanism on or off; `patience` controls how many steps without meaningful improvement the optimizer tolerates before stopping. Under the hood, `scribe` tracks a moving average of the loss so a single noisy step does not trigger a premature stop. Additional knobs (`min_delta`, `min_delta_pct`, `check_every`, `warmup`, `smoothing_window`, …) are available for finer control.
    """)
    return


@app.cell
def _(adata, data_dir, pickle, scribe):
    # Define output directory
    out_dir = data_dir / "scribe_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define parameterization
    _parameterization = "canonical"

    # Define output file path
    _out_path = out_dir / f"scribe_results_nbdm_{_parameterization}.pkl"

    if _out_path.exists():
        # Load model from pkl file
        with open(_out_path, "rb") as _f:
            results_nbdm = pickle.load(_f)
    else:
        # Fit basic model to data with variable capture probability fit per cell
        results_nbdm = scribe.fit(
            adata,
            variable_capture=False,
            parameterization=_parameterization,
            early_stopping={"enabled": True, "patience": 1000}
        )
        # Save the fitted model
        with open(_out_path, "wb") as _f:
            pickle.dump(results_nbdm, _f)

    results_nbdm
    return out_dir, results_nbdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To check whether training converged, we examine the [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) (evidence lower bound) loss curve. The ELBO is the quantity variational inference optimizes: roughly, the negative log-evidence plus a gap measuring how well the approximate posterior matches the true one. For a sanity check we do not need to read it quantitatively: a clean run produces a curve that drops quickly at first and then flattens out. Spikes, slow drifts upward, or oscillations that never settle are warnings worth investigating.
    """)
    return


@app.cell
def _(results_nbdm, scribe):
    # Plot ELBO loss
    scribe.viz.plot_loss(results_nbdm, figsize=(6, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The training loss has converged, but that alone does not prove the model describes the *counts* well. A posterior predictive check (PPC) asks a more direct question: if we simulate new data from the fitted model, do those simulations look like the real measurements? Concretely, we draw synthetic count matrices from the posterior predictive distribution and compare their marginal distributions (gene by gene) to the observed histograms. When the model is adequate, the synthetic and real histograms overlap; when it is not, you will see systematic mismatches—shifted means, wrong dispersion, missing zeros, or heavy tails—even if the optimizer stopped happily.

    In `scribe`, `scribe.viz.plot_ppc` automates that comparison for a grid of genes. It stratifies genes by per-gene median expression using log-spaced bins, then fills a panel layout (`n_genes`, `n_rows`, and derived columns) so each row samples a different part of the dynamic range. For each gene it overlays the observed UMI histogram against the posterior predictive distribution using `n_samples` Monte Carlo draws, showing the model’s implied variability rather than a single predicted curve. The point is a quick visual sanity check that the generative model matches what you actually measured.
    """)
    return


@app.cell
def _(adata, results_nbdm, scribe):
    scribe.viz.plot_ppc(
        results_nbdm,
        adata,
        n_genes=16,
        n_rows=4,
        figsize=(8, 8),
        n_samples=512,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looking at the PPC panels, three patterns stand out:

    - *Systematic mismatch* — for several genes the observed UMI histogram does not sit where the model’s predictive band expects it.
    - *Over-wide predictive bands* — even when the match is roughly acceptable, the shaded posterior predictive region looks unnecessarily diffuse, as if the model is hedging.
    - *High-abundance housekeeping genes look odd* — very highly expressed genes like FAU, RPS15, and RPS2 do not even show a PPC. Did the fit fail entirely for these genes?

    None of this contradicts a “nice-looking” training loss. PPCs ask a different question: *does the fitted generative model actually reproduce the count-level patterns you measured?*

    In practice, the issues above stem from two interacting limitations of this first fit:

    1. **Library size varies substantially across cells, even in a monoculture.**  Our ECDF showed close to an order-of-magnitude spread in total UMIs. In a Jurkat-only dataset, that variation is most plausibly sequencing and capture depth, not a biological axis. When the model pretends every cell is on the same effective depth scale (`variable_capture=False`), it must explain depth-driven shifts through the gene-level count model instead—much like running differential expression without size factors. You can still get a fit, but the marginal count distributions will look strained.

        The fix is analogous to per-cell normalization, but implemented as an explicit model ingredient: give each cell its own capture parameter (`variable_capture=True`) so depth is not silently absorbed into gene parameters.

    2. **Negative Binomial parameters can be “slippery” in canonical $(r,p)$ coordinates.**  Many different $(r,p)$ pairs produce nearly the same mean–variance trade-off, creating a curved ridge of near-equivalent solutions. Because our guide is mean-field (treating parameters as independent), the approximation smears posterior mass across that ridge rather than concentrating it along it. A common symptom is predictive bands that are wider than the data warrant.

    We will address depth and guide flexibility as we go: first add variable capture, then explore richer variational families that can represent parameter dependence.

    To make the $(r,p)$ coupling more concrete before we change the model, we pick a few genes from the PPC grid, draw posterior samples with `get_posterior_matrix`, and look at a corner plot of $p$ versus the gene-specific $r_g$ values.
    """)
    return


@app.cell
def _(adata, corner, results_nbdm):
    # Build integer indices for a handful of example genes
    gene_list = ["MKKS", "EIF5", "RPS29", "RPS2"]
    _name_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    gene_index = [_name_to_idx[g] for g in gene_list if g in _name_to_idx]

    _results_subset = results_nbdm[gene_index]

    # Export directly as a corner-ready matrix plus labels/metadata.
    _samples_2d, _labels, _ = _results_subset.get_posterior_matrix(
        n_samples=5_000,
        include=["p", "r"],
        exclude_deterministic=True,
        store_samples=False,
        convert_to_numpy=True,
    )

    corner.corner(_samples_2d, labels=_labels)
    return (gene_index,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Reading the corner plot

    A corner plot (sometimes called a pair plot) is a compact way to look at uncertainty across several parameters at once.

    - Along the diagonal you see one-dimensional histograms: the marginal posterior distribution of each parameter (here, the shared $p$ plus gene-specific $r_g$ for a handful of genes).
    - In the lower triangle you see two-dimensional scatter/density panels for every pair of parameters, showing how their posterior samples relate to each other.

    Think of it as a multi-way version of “plot gene A vs. gene B and look for correlation,” except the axes are model parameters and the cloud of points represents posterior uncertainty, not cells.

    ### What to look for in this particular figure

    We are focusing on the negative binomial’s canonical parameters: a shared success probability $p$ and gene-specific dispersion parameters $r_g$. In principle, many $(r_g,p)$ combinations can predict similar counts (a curved ridge in the full posterior). When a mean-field variational approximation encounters that ridge, it cannot represent the coupling and instead inflates each marginal independently. The result is that the 2D panels between $p$ and each $r_g$ look like broad blobs rather than tight, tilted ellipses—extra marginal uncertainty that arises from the approximation, not from genuine parameter ambiguity.

    The practical punchline: PPC weirdness is often less about “the optimizer failed” and more about whether the model plus its approximation represent depth and NB geometry in a way that matches how counts were actually generated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting a model with variable capture

    Our first fit essentially pretended that the UMI matrix is already on the “right” scale for a single negative binomial story. But the ECDF of library size told us otherwise: in a Jurkat monoculture, an order-of-magnitude spread in total counts is far easier to attribute to technical depth/capture variability than to a hidden biological axis.

    In a standard pipeline, one forces cells onto a common scale with normalization (size factors, scaling to 10K UMIs, etc.). In `scribe`, you can instead treat depth as a first-class parameter: each cell gets its own capture probability $\nu_c \in (0,1)$—a concrete way to say “this cell’s mRNA molecules were converted into UMIs with this efficiency.”

    The algebra (worked out in the paper’s Dirichlet–multinomial derivation) is pleasantly tidy. The story for one gene in one cell is:

    1. **Latent biology:** a transcript count $m_g \sim \mathrm{NB}(r_g, p)$ (shared $p$ across genes in this formulation).
    2. **Observation:** UMIs are a thinning of transcripts, $u_g \mid m_g, \nu_c \sim \mathrm{Binomial}(m_g, \nu_c)$.

    If you marginalize out the unobserved $m_g$, the observed UMIs are still negative binomial—same $r_g$, but with an effective success probability that combines biology ($p$) and technology ($\nu_c$):

    $$
    \hat{p}_c = \frac{p}{\nu_c + p(1-\nu_c)}.
    $$

    In words: $p$ and $r_g$ control the shape of expression at the “true transcript” level, while $\nu_c$ asks “how much of that signal survived the experiment for this cell?” When $\nu_c$ is low (a shallowly sequenced cell), the cell effectively observes a more thinned draw; the formula packages that thinning into a single cell-specific $\hat{p}_c$ so the likelihood stays a clean $\mathrm{NB}(r_g,\hat{p}_c)$ for UMIs.

    Setting `variable_capture=True` is therefore not a magic extra knob—it is the principled analogue of accounting for sequencing depth, except the depth correction lives inside the generative model (with its own posterior uncertainty) instead of being a one-off preprocessing division.

    In the next cell we refit with this extension turned on (still in canonical $(r,p)$ coordinates for the moment), and we will revisit PPCs to see whether the count-level story looks more coherent once cells are allowed to differ in $\nu_c$.

    Before each heavier refit, we call JAX’s `clear_caches()` plus `gc.collect()` so GPU memory stays manageable—we repeat this housekeeping pattern whenever we swap models in this tutorial to avoid out-of-memory issues.
    """)
    return


@app.cell
def _(adata, clear_caches, gc, out_dir, pickle, scribe):
    # Clear GPU memory from previous fits
    clear_caches()
    gc.collect()

    # Define parameterization
    _parameterization = "canonical"

    # Define output file path
    _out_path = out_dir / f"scribe_results_nbvcp_{_parameterization}.pkl"

    if _out_path.exists():
        # Load model from pkl file
        with open(_out_path, "rb") as _f:
            results_nbvcp = pickle.load(_f)
    else:
        # Fit basic model to data with variable capture probability fit per cell
        results_nbvcp = scribe.fit(
            adata,
            variable_capture=True,
            parameterization=_parameterization,
            early_stopping={"enabled": True, "patience": 1000}
        )
        # Save the fitted model
        with open(_out_path, "wb") as _f:
            pickle.dump(results_nbvcp, _f)

    results_nbvcp
    return (results_nbvcp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Right after a variable-capture fit, it is worth checking that the model is using per-cell capture sensibly. `scribe.viz.plot_p_capture_scaling` scatters each cell’s estimated capture probability (MAP of `p_capture`) against its library size $L_c$ (total UMIs), so you can see whether shallow and deep cells receive sensible relative adjustments.
    """)
    return


@app.cell
def _(adata, results_nbvcp, scribe):
    scribe.viz.plot_p_capture_scaling(
        results_nbvcp, counts=adata, figsize=(4,4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The capture-scaling plot is a diagnostic for “did variable capture do anything?” Each point is a cell, with library size $L_c$ on the $x$-axis and the fitted per-cell capture probability $\nu_c$ on the $y$-axis.

    The pattern—capture probability rising with depth at modest library sizes, then flattening near 1 for the largest libraries—shows how the fitted model distributes the same depth variation we saw in the ECDF: shallow cells are interpreted as having lower effective capture, while the deepest cells are pushed toward near-complete capture.

    It is tempting to read the plateau as “above some UMI threshold, everything is biology.” Be careful: from a generative-model perspective, absolute capture is not something counts pin down by themselves. Observed library size mainly constrains a *product* involving $\nu_c$ and the cell’s total mRNA budget; separating “how much RNA was in the cell” from “how efficiently it was converted into UMIs” requires extra information. If you had an independent total-mRNA quantification (or another trustworthy anchor for total RNA content), you could inject that as stronger prior knowledge and make the capture curve more interpretable in absolute terms. In routine scRNA-seq we usually lack that direct measurement, so this plot is best read as a sanity check that depth is being routed through a technical channel in a coherent way, not as a calibrated readout of physical capture efficiency.

    The reassuring structural point—for the kinds of gene-by-gene comparisons we often care about—is that when capture acts like random thinning of transcripts into UMIs, relative expression (gene A vs. gene B within a cell) can be much more stable than the absolute “how many molecules” story. In the paper we show that misspecifying $\nu_c$ forces the model to compensate elsewhere, but many between-gene comparisons based on relative abundance remain unchanged. By contrast, summaries that speak literally about total RNA or other absolute biological scales can absorb capture error more directly and deserve extra caution.

    One more guardrail: a smooth capture-vs.-depth curve can still be misleading if the model is not actually predicting counts well. Do not treat this figure as authoritative when PPCs look systematically off. Trust the capture story only insofar as the generative model passes count-level sanity checks—which is why we revisit PPCs next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us revisit posterior predictive checks now that variable capture is turned on. The hope is straightforward: much of what looked wrong in the first PPC grid was probably depth/sampling variation masquerading as gene-level count behavior. By giving each cell its own capture parameter $\nu_c$, the model can absorb that cell-to-cell library-size spread as technical rather than forcing the negative binomial gene parameters to explain everything.

    When you scan the new panels, do not look for perfection—look for systematic repair: predicted count distributions should track the observed histograms more closely, especially for genes where depth drives most of the marginal weirdness, and the predictive bands should not be artificially wide from confusing sampling depth with biological dispersion.

    Variable capture is not a universal solvent, however. If you still see strain in the PPCs, that is a hint that other parts of the modeling/inference story matter—most notably the $(r,p)$ geometry under a canonical parameterization and what a mean-field posterior can (and cannot) represent. We will chase those refinements next; for now, treat this PPC pass as the “did depth help?” checkpoint.
    """)
    return


@app.cell
def _(adata, results_nbvcp, scribe):
    scribe.viz.plot_ppc(
        results_nbvcp,
        adata,
        n_genes=16,
        n_rows=4,
        figsize=(8, 8),
        n_samples=512,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compared to the first fit, many genes stop looking systematically shifted relative to the model’s predictive band, because depth is no longer being smuggled into every gene-level parameter.

    But it is also common—exactly what we see here—that the predictive shaded region still looks “too fat.” When that happens, it is a hint that capture was only part of the story. The other part is geometry: in canonical coordinates we explicitly sample $r_g$ and $p$, and those coordinates carry a strong, curved dependence (the same “banana” posterior shape people discuss for negative binomials). A mean-field variational posterior tries to approximate that curved object with something rectangular, which often manifests as extra uncertainty: the model “knows” it cannot be arbitrarily confident, so predictive simulations become unnecessarily diffuse even when the mean trend improves.

    Our next move is therefore not “change the biology”—it is to change the coordinates we use for inference while keeping the same generative story for UMIs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same generative model, new coordinates: the mean–odds parameterization

    In `scribe`, the `mean_odds` parameterization swaps the primary free parameters to:

    - $\mu_g$: a gene-specific mean parameter for the negative binomial (at the biological level, i.e., the mean mRNA count $\langle m \rangle$), and
    - $\phi$: a positive odds-style parameter that sets the NB success probability through

    $$
    p = \frac{1}{1+\phi},
    $$

    with the dispersion reconstructed deterministically as

    $$
    r_g = \mu_g\,\phi.
    $$

    Two points make this a big deal in practice:

    1. **A friendlier landscape for optimization.** A probability like $p$ lives in a bounded interval and can spend time pressed against boundaries in tricky regimes. Working with $\phi \in (0,\infty)$ avoids much of the numerical awkwardness that comes from navigating a near-hard wall at $p \approx 1$.

    2. **Correlation becomes partly built-in rather than fought.** In canonical form, many $(r_g, p)$ pairs reproduce similar means, creating the ridge that mean-field approximations struggle with. In mean–odds coordinates, $\mu_g$ is the natural knob for “how highly expressed is this gene,” and $\phi$ absorbs complementary mean–variance / odds degrees of freedom, with $r_g$ no longer an independent floating parameter—it is computed from $(\mu_g, \phi)$. This does not magically eliminate all posterior dependence (the data can still couple genes), but it is the same trick used throughout statistics: reparameterize so the parameters you fit are closer to orthogonal in practice, even when the underlying model is unchanged.

    The right mental model is cartographic: you are not changing the terrain (the NB sampling model, capture thinning, etc.); you are changing the map projection the optimizer and variational approximation use to walk over it. Same model, different geometry—and often a noticeably easier path to tighter, more honest uncertainty in PPCs. The change is as simple as setting `parameterization="mean_odds"` in `scribe.fit`.
    """)
    return


@app.cell
def _(adata, clear_caches, gc, out_dir, pickle, scribe):
    # Clear GPU memory from previous fits
    clear_caches()
    gc.collect()

    # Define parameterization
    _parameterization = "mean_odds"

    # Define output file path
    _out_path = out_dir / f"scribe_results_nbvcp_{_parameterization}.pkl"

    if _out_path.exists():
        # Load model from pkl file
        with open(_out_path, "rb") as _f:
            results_odds = pickle.load(_f)
    else:
        # Fit basic model to data with variable capture probability fit per cell
        results_odds = scribe.fit(
            adata,
            variable_capture=True,
            parameterization=_parameterization,
            early_stopping={"enabled": True, "patience": 1000}
        )
        # Save the fitted model
        with open(_out_path, "wb") as _f:
            pickle.dump(results_odds, _f)

    results_odds
    return (results_odds,)


@app.cell
def _(results_odds, scribe):
    # Plot ELBO loss for mean_odds fit
    scribe.viz.plot_loss(results_odds, figsize=(6, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us look again at the $p_\text{capture}$ scaling plot.
    """)
    return


@app.cell
def _(adata, results_odds, scribe):
    scribe.viz.plot_p_capture_scaling(
        results_odds, counts=adata, figsize=(4,4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this fit, the model places the full-capture saturation point around 30K UMI reads. As discussed earlier, the absolute scale of $\nu_c$ is not identifiable from counts alone—but for downstream analysis the degeneracy does not affect our conclusions.

    Let us look at the resulting PPCs for the same set of genes.
    """)
    return


@app.cell
def _(adata, results_odds, scribe):
    scribe.viz.plot_ppc(
        results_odds,
        adata,
        n_genes=16,
        n_rows=4,
        figsize=(8, 8),
        n_samples=512,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a substantial step forward: after switching to `mean_odds`, the PPC panels track the observed UMI histograms much more closely, and the predictive bands no longer look artificially inflated in the way we saw when the variational approximation was fighting canonical $(r,p)$ geometry.

    PPCs are useful for a handful of genes, but it is natural to wonder whether the improvement holds genome-wide. `scribe.viz.plot_mean_calibration` provides a compact global check: one point per gene comparing the empirical average UMI count to the model’s predicted average from the MAP parameters. For variable-capture fits, the prediction converts the latent NB “biological” mean into an observed-scale mean using the average capture $\bar{\nu}$ across cells, so the comparison is like-for-like against what you actually measured. On a log–log plot, points hugging the identity line mean the model’s best single-parameter explanation of overall expression level matches the data across the dynamic range; systematic curvature or a drifting cloud flags global miscalibration that might be easy to miss in individual PPC panels.
    """)
    return


@app.cell
def _(adata, results_odds, scribe):
    scribe.viz.plot_mean_calibration(
        results_odds, counts=adata, figsize=(4, 4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With good PPCs in hand, we can look at the same interpretable NB coordinates biologists think in—even though the fit was done in $(\mu_g, \phi)$ space. In `scribe`, $p$ and $r_g$ are deterministic functions of the sampled parameters (they inherit uncertainty from $(\mu, \phi)$, but they are not separate free knobs in this parameterization). The next corner plot is not contradicting the reparameterization: we are simply exporting the implied $(p, r_g)$ samples to compare apples-to-apples with the earlier canonical-only exercise.

    If things went well, the 2D panels involving shared $p$ and gene-specific $r_g$ should look more structured and less like an arbitrary fat blob than under the canonical mean-field regime. Good PPCs mean the model has found a self-consistent explanation of the counts; when you map that explanation back to $(r,p)$ coordinates, you often see correlations closer to the mean–dispersion constraints implied by the NB, rather than the approximation smearing mass in a way that made predictions overly wide.

    Below we reuse the same illustrative genes and call `get_posterior_matrix` with `include=["p","r"]` and `exclude_deterministic=False` so those derived canonical parameters are actually materialized for plotting.
    """)
    return


@app.cell
def _(corner, gene_index, results_odds):
    _results_subset = results_odds[gene_index]

    # Export directly as a corner-ready matrix plus labels/metadata.
    _samples_2d, _labels, _ = _results_subset.get_posterior_matrix(
        n_samples=5_000,
        include=["p", "r"],
        exclude_deterministic=False,
        store_samples=False,
        convert_to_numpy=True,
    )

    corner.corner(_samples_2d, labels=_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see the expected coupling between $r_g$ and the shared $p$: in a negative binomial, many $(r_g, p)$ pairs reproduce a similar mean along a curved ridge, so a strong $r_g$–$p$ association is exactly what the sampling model (not necessarily a surprise from the data) tells you to expect.

    A corner plot can over-interpret that as “everything is correlated with everything” for a few mechanical reasons. First, $p$ is the same across genes in this fit, so the $p$ axis is literally one shared draw replayed in every $r_g$–$p$ panel—of course the clouds line up. Second, you are exporting coordinates $(p, r_g)$ that are not meant to be orthogonal in the original parameterization, while the fit itself was done with a mean-field posterior that pretends each dimension has an independent margin. The combination—shared parameter + ridge geometry + mean-field—can make the off-diagonals look like extreme global correlation even when a large portion of the pattern is (i) structural sharing of $p$ and (ii) a variational/parameterization artifact rather than extra biological coupling.

    When we fit in mean–odds $(\mu_g, \phi)$ coordinates, $p$ and $r_g$ become deterministic functions of $(\mu_g, \phi)$: $p$ depends on $\phi$ alone, and $r_g = \mu_g \phi$. That does not erase dependence in the true posterior, but it changes what the guide is allowed to smear. To make the contrast obvious, the next panel plots $(\mu, \phi)$—where the fit actually lives—where we expect the spurious “global correlation” to fall away and the posterior samples to look like relatively uncoupled 2D marginals.
    """)
    return


@app.cell
def _(corner, gene_index, results_odds):
    _results_subset = results_odds[gene_index]

    # Export directly as a corner-ready matrix plus labels/metadata.
    _samples_2d, _labels, _ = _results_subset.get_posterior_matrix(
        n_samples=5_000,
        include=["phi", "mu"],
        exclude_deterministic=True,
        store_samples=False,
        convert_to_numpy=True,
    )

    corner.corner(_samples_2d, labels=_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A joint, low-rank variational guide (still the same generative model)

    So far, variable capture helped the model treat depth more honestly, but the posterior approximation itself can still be a limiting factor. The default mean-field story is computationally convenient, but it has a structural blind spot: it pretends different parameter blocks vary independently even when the data naturally tie them together.

    Two kinds of coupling matter for negative binomial counts:

    - **Within a gene:** the NB still has a mean–variance / mean–dispersion structure. In canonical $(r_g, p)$ space that shows up as the famous curved ridge. Mean–odds $(\mu_g, \phi)$ already made mean-field life easier because $\mu_g$ is the natural “level” knob and $\phi$ carries complementary degrees of freedom, with $(p, r_g)$ computed deterministically.
    - **Across genes:** even with a shared success probability, every gene’s data tug on the same shared piece of the model, inducing indirect posterior coupling between genes. A factorized guide can smear that coupling into extra marginal uncertainty, which often looks like over-wide PPC bands.

    The paper’s joint low-rank guide is a pragmatic upgrade that is not tied to one parameterization: `scribe` can pair low-rank + joint structure with `canonical`, `mean_prob`, or `mean_odds`. Mathematically, the idea is to replace an overly factorized family with one multivariate Normal (in an unconstrained/reparameterized space) whose covariance is low rank:

    $$
    \underline{\underline{\Sigma}} \;\approx\; \underline{\underline{W}}\,\underline{\underline{W}}^\top + \underline{\underline{D}},
    $$

    with a small number of columns in $\underline{\underline{W}}$ (here `guide_rank=128`)—analogous to keeping only the first few principal components to summarize coordinated variation. The guide can move parameters along a low-dimensional set of shared joint modes instead of forcing strict independence, while still avoiding the $O(G^2)$ cost of a dense covariance across all genes.

    We run it under `mean_odds` because that parameterization was already empirically well-behaved at the mean-field stage. In code, three knobs implement the idea:

    - `guide_rank=128`: how many joint directions of coordinated posterior variability the guide can use.
    - `joint_params="biological"`: shorthand for the two sampled core parameters in the active parameterization. For `mean_odds`, that means $\underline{\mu}$ and $\phi$ are guided jointly, so the approximation can represent correlation between mean intensity and odds/dispersion-like degrees of freedom.

    We also increase `n_steps=500_000` because richer guides often need more optimization time, and we turn off `early_stopping` to let the optimizer use all available steps. The loss curve may look flat for long stretches, but small incremental improvements still tighten the fit.
    """)
    return


@app.cell
def _(adata, clear_caches, gc, out_dir, pickle, scribe):
    # Clear GPU memory from previous fits
    clear_caches()
    gc.collect()

    # Define parameterization
    _parameterization = "mean_odds"

    # Define output file path
    _out_path = out_dir / f"scribe_results_nbvcp-low-rank_{_parameterization}.pkl"

    if _out_path.exists():
        # Load model from pkl file
        with open(_out_path, "rb") as _f:
            results_lowrank = pickle.load(_f)
    else:
        # Fit basic model to data with variable capture probability fit per cell
        results_lowrank = scribe.fit(
            adata,
            variable_capture=True,
            parameterization=_parameterization,
            guide_rank=128,
            joint_params="biological",
            n_steps=500_000,
            early_stopping={"enabled": False}
        )
        # Save the fitted model
        with open(_out_path, "wb") as _f:
            pickle.dump(results_lowrank, _f)

    results_lowrank
    return (results_lowrank,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us check the loss to make sure training converged.

    > **Note:** we are not showing the loss curve for every model to save space, but this should always be the first plot you make.
    """)
    return


@app.cell
def _(results_lowrank, scribe):
    # Plot ELBO loss
    scribe.viz.plot_loss(results_lowrank, figsize=(6, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let us look at the PPC plots under the low-rank posterior.
    """)
    return


@app.cell
def _(adata, results_lowrank, scribe):
    scribe.viz.plot_ppc(
        results_lowrank,
        adata,
        n_genes=16,
        n_rows=4,
        figsize=(8, 8),
        n_samples=512,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After the joint low-rank fit, the PPCs for these genes look essentially the same as the mean-field `mean_odds` run—the marginal count predictions were already well calibrated. Let us check the global mean calibration as well.
    """)
    return


@app.cell
def _(adata, results_lowrank, scribe):
    scribe.viz.plot_mean_calibration(
        results_lowrank, counts=adata, figsize=(4, 4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So far, mean-field variational inference only gave us independent approximate marginals—there was no notion of “this gene’s parameter co-moves with that gene’s” inside the fit. A joint guide (here, low-rank structure across genes and parameters) is exactly where that becomes possible: the approximation can place non-negligible posterior correlation between gene-level parameters when the data support it.

    `scribe.viz.plot_correlation_heatmap` visualizes a correlation matrix of posterior samples (clustered, focusing on a subset of the most variable genes for readability). We might think of this as a partical first pass at spotting putative co-regulation. However, the correlations shown here are correlated parameters on a model that assumes independent genes. It is not an explicit parameterization of the gene-gene correlations that the data might have. For that, `scribe` has a whole family of models that explicitly account for gene-gene correlation matrices.
    """)
    return


@app.cell
def _(adata, results_lowrank, scribe):
    scribe.viz.plot_correlation_heatmap(
        results_lowrank, 
        counts=adata, 
        n_genes=100,
        n_samples=1_000,
        figsize=(7,7)
    )
    return


@app.cell
def _(corner, gene_index, results_lowrank):
    _results_subset = results_lowrank[gene_index]

    # Export directly as a corner-ready matrix plus labels/metadata.
    _samples_2d, _labels, _ = _results_subset.get_posterior_matrix(
        n_samples=5_000,
        include=["p", "r"],
        exclude_deterministic=False,
        store_samples=False,
        convert_to_numpy=True,
    )

    corner.corner(_samples_2d, labels=_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why is the $p$ range so much narrower here?

    In the mean-field corner plot above, $p$ wandered freely from roughly 0.24 to 0.48—and yet the PPCs looked excellent. Here, the joint low-rank guide pins $p$ down to a sliver around 0.314. At first glance that feels wrong: if so many $p$ values gave good fits before, should a richer posterior not keep that flexibility?

    The short answer is: the mean-field’s wide $p$ was the artifact, not this narrow one.

    A mean-field guide treats every parameter as independent. It *cannot* represent the fact that when $p$ goes up, each $r_g$ should go down in a coordinated way to keep the gene means stable. Faced with that limitation, the optimizer makes $q(\phi)$ wide and each $q(\mu_g)$ wide independently, so that at least some random draws land near the good-fit region by chance. The wide $p$ range you saw is not the posterior saying “$p$ is genuinely uncertain”—it is the guide saying “I cannot model the coupling, so I will be vague about everything and hope for the best.” The dramatic banana shapes in the off-diagonal panels literally show the ridge structure the guide is failing to capture.

    Think of it like giving directions without a map. You might say “the restaurant is somewhere between 1st and 10th Street”—not because the address is uncertain, but because you cannot explain how to get there. Hand someone a map (model the coupling) and the directions collapse to “it’s at 5th and Main.”

    The low-rank guide *can* model the coupling: each gene’s $\mu_g$ shifts in response to $\phi$ through a learned per-gene regression coefficient $\alpha_g$. Once that coupling is explicit, the marginal width of $\phi$ reflects the actual posterior uncertainty perpendicular to the ridge—how much $\phi$ can vary when all the $\mu_g$ values adjust optimally alongside it. With $\sim\!30{,}000$ genes and $\sim\!3{,}000$ cells all constraining a single shared scalar, that residual uncertainty is genuinely tiny. The narrow $p$ is not a limitation of the guide; it is the guide being precise enough to show how tightly the data actually constrain this parameter.

    A sanity check for the intuition: the $r_g$–$r_g$ panels also tightened dramatically. Under the mean-field, the wide off-diagonals between $r_g$ values came almost entirely from replaying the same noisy $\phi$ draw across genes—shared-parameter bookkeeping, not genuine gene–gene biology. The joint guide absorbs that mechanical coupling into $\alpha_g$ and leaves only the real residual structure.

    In summary: the mean-field inflates marginal uncertainty to compensate for missing coupling; the joint guide models the coupling directly and reveals that the true uncertainty is small. Both are internally consistent given their assumptions, but the joint guide’s assumptions are closer to reality.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relaxing the shared-$p$ assumption: gene-specific $p_g$

    So far, every model we have fit assumes a single success probability $p$ shared across all genes. That assumption is what makes the Dirichlet–multinomial factorization work cleanly: when every gene uses the same $p$, the success probability cancels out of the compositional distribution and differential expression can operate on $\underline{r}$ alone.

    But what if the data do not support that assumption? Perhaps the negative binomial’s mean–variance relationship is not uniform across the transcriptome. `scribe` provides a principled way to check: replace the scalar $p$ with a hierarchical gene-specific $p_g$, where each gene draws its own success probability from a learned population distribution:

    $$
    \text{logit}(p_g) \sim \mathcal{N}(\mu_p, \sigma_p), \quad g = 1, \ldots, G.
    $$

    The hyperparameters $\mu_p$ and $\sigma_p$ are inferred from the data. If the data are consistent with shared $p$, the posterior will shrink $\sigma_p$ toward zero and all $p_g$ will collapse to the same value—recovering the standard model automatically. If gene-to-gene variation is real, the model can accommodate it without forcing it.

    Crucially, `scribe` knows how to turn the inferred parameters into compositions—the fraction of the total transcriptome that each gene occupies. Under the standard shared-$p$ model, compositions come from a Dirichlet distribution that depends only on $\underline{r}$. Under the gene-specific $p_g$ model, the mapping is slightly more involved (each gene’s contribution is scaled by its own $p_g$), but the math is exact and reduces to the standard Dirichlet when all $p_g$ happen to be equal. This means differential expression analysis remains on solid mathematical footing even with the extra flexibility.

    In code, three new knobs appear compared to the previous low-rank model:

    - `parameterization="canonical"`: we switch back to canonical $(r_g, p_g)$ coordinates because $p_g$ is now gene-specific and no longer a derived quantity—it is a first-class sampled parameter alongside $r_g$.
    - `dense_params="mean"`: tells the structured joint guide which parameters receive the full low-rank covariance treatment. Here, `"mean"` refers to $r_g$: it gets the rank-128 $\underline{\underline{W}}\underline{\underline{W}}^\top + \underline{\underline{D}}$ structure for gene–gene correlations. The $p_g$ parameters receive a simpler diagonal posterior with a linear regression on $r_g$—enough to capture per-gene $r_g$–$p_g$ coupling without consuming rank-$k$ capacity on $p_g$–$p_g$ correlations across genes.
    - `prob_prior="gaussian"`: activates the hierarchical gene-specific $p_g$ model, replacing the default flat or fixed prior on a shared $p$ with the $\mathcal{N}(\mu_p, \sigma_p)$ hierarchy.

    The result is a model that is more flexible in its generative assumptions (each gene has its own $p_g$) and still compositionally valid (the Gamma-based normalization is exact). The tradeoff—which we will examine shortly—is that the extra degrees of freedom may widen posterior uncertainty on quantities that depend on $p_g$, including compositions. Whether that extra uncertainty is a feature (honest reflection of what the data can tell us) or a cost (estimating parameters the data did not need) is exactly the kind of question Bayesian model comparison can answer formally.
    """)
    return


@app.cell
def _(adata, clear_caches, gc, out_dir, pickle, scribe):
    # Clear GPU memory from previous fits
    clear_caches()
    gc.collect()

    # Define parameterization
    _parameterization = "canonical"

    # Define output file path
    _out_path = out_dir / f"scribe_results_nbvcp-low-rank-prob_{_parameterization}.pkl"

    if _out_path.exists():
        # Load model from pkl file
        with open(_out_path, "rb") as _f:
            results_hierprob = pickle.load(_f)
    else:
        # Fit basic model to data with variable capture probability fit per cell
        results_hierprob = scribe.fit(
            adata,
            variable_capture=True,
            parameterization=_parameterization,
            guide_rank=128,
            joint_params="biological",
            dense_params="mean",
            prob_prior="gaussian",
            n_steps=100_000,
            early_stopping={"enabled": True, "patience": 1000}
        )
        # Save the fitted model
        with open(_out_path, "wb") as _f:
            pickle.dump(results_hierprob, _f)

    results_hierprob
    return (results_hierprob,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us look again at the capture-scaling plot.
    """)
    return


@app.cell
def _(adata, results_hierprob, scribe):
    scribe.viz.plot_p_capture_scaling(
        results_hierprob, counts=adata, figsize=(4,4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This time, the model places the fully captured saturation point around 50K reads.

    Now let us look at the PPCs.
    """)
    return


@app.cell
def _(adata, results_hierprob, scribe):
    scribe.viz.plot_ppc(
        results_hierprob,
        adata,
        n_genes=16,
        n_rows=4,
        figsize=(8, 8),
        n_samples=512,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The PPCs for this run look excellent—and that is worth a pause, because we are no longer in the mean–odds $(\mu_g, \phi)$ reparameterization that we leaned on earlier. What changed is not “forgetting the geometry problem” but addressing the same object in a different way: the $(r_g, p_g)$ ridge is still the hard part of a negative binomial, but here each gene’s $(r_g, p_g)$ pair is allowed to move together. The hierarchical $\text{logit}(p_g)$ prior plus a guide that explicitly models intra-pair dependence supplies enough coupling that the variational family is no longer forced to smear mass along a fake axis in $(r, p)$. In other words, we traded the $\mu / \phi$ reparameterization trick for gene-specific $p_g$ and structured per-gene $(r_g, p_g)$ dependence, and the PPCs suggest that is enough to recover the good predictive behavior.

    Let us check the global calibration overview.
    """)
    return


@app.cell
def _(adata, results_hierprob, scribe):
    scribe.viz.plot_mean_calibration(
        results_hierprob, counts=adata, figsize=(4, 4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With gene-specific $p_g$, a corner plot in `include=["p","r"]` becomes a bigger grid: you are no longer overlaying many genes on top of a single shared $p$, but laying out each gene’s own $(r_g, p_g)$ pair. For a given gene you should see strong intra-gene dependence between its $r_g$ and its $p_g$ (the same ridge geometry as before, but now per gene), while off-diagonal blocks mixing $p_g$ from one gene with $r_{g’}$ from another should look much more like independent blobs.

    The lack of obvious correlation between unmatched $(p_g, r_{g’})$ pairs in this figure is not a claim that the model has forgotten gene–gene structure in general (the low-rank joint structure you used still lives in the fit). It is largely a statement about which genes we put in the panel for visibility: the particular set of names in `gene_index` can easily be weakly coupled to each other under the posterior, so the cross-gene corners look flat even when other gene sets—or a heatmap across many loci—would show shared axes of variation. In other words, this corner is a pairwise microscope on local $(r_g \leftrightarrow p_g)$ coupling, not a full summary of all cross-gene co-movement.
    """)
    return


@app.cell
def _(corner, gene_index, results_hierprob):
    _results_subset = results_hierprob[gene_index]

    # Export directly as a corner-ready matrix plus labels/metadata.
    _samples_2d, _labels, _ = _results_subset.get_posterior_matrix(
        n_samples=5_000,
        include=["p", "r"],
        exclude_deterministic=True,
        store_samples=False,
        convert_to_numpy=True,
    )

    corner.corner(_samples_2d, labels=_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do compositions actually change between models?

    Both models—shared $p$ with `mean_odds` and gene-specific $p_g$ with `canonical`—gave excellent calibration and PPCs. But the question we ultimately care about for downstream analysis is: **do the inferred compositions differ?**

    Compositions tell us what fraction of the transcriptome each gene occupies. Under the shared-$p$ model, each gene's contribution is determined solely by its dispersion $r_g$: draw $\gamma_g \sim \text{Gamma}(r_g, 1)$, normalize, and the shared $p$ drops out entirely. Under the gene-specific $p_g$ model, each gene's contribution is additionally scaled by $(1 - p_g)/p_g$ before normalizing—so genes with lower $p_g$ get a proportionally larger share. If the $p_g$ values are all nearly identical (as the hierarchical prior encourages), the two models should agree.

    To test this, we draw 1,000 compositional samples from each fitted model and compare the median composition of every gene between the two. The error bars show the 95% credible interval from each model's posterior, giving us a sense of how uncertain each model is about each gene's share of the transcriptome.
    """)
    return


@app.cell
def _(clear_caches, gc, np, out_dir, pickle, plt):
    # Clear GPU memory from previous fits
    clear_caches()
    gc.collect()

    # Define output file path
    _out_path1 = out_dir / f"scribe_results_nbvcp-low-rank_mean_odds.pkl"
    _out_path2 = out_dir / f"scribe_results_nbvcp-low-rank-prob_canonical.pkl"

    if _out_path1.exists():
        with open(_out_path1, "rb") as _f:
            _results_lowrank = pickle.load(_f)

    if _out_path2.exists():
        with open(_out_path2, "rb") as _f:
            _results_hierprob = pickle.load(_f)

    _N_SAMPLES = 1_000

    _simplex_lowrank = _results_lowrank.get_compositional_samples(
        n_samples=_N_SAMPLES, store_samples=False
    )
    _simplex_hierprob = _results_hierprob.get_compositional_samples(
        n_samples=_N_SAMPLES, store_samples=False
    )

    _lower_quantile = 2.5
    _upper_quantile = 97.5

    _median_x = np.median(_simplex_lowrank, axis=0)
    _median_y = np.median(_simplex_hierprob, axis=0)

    _low_x = np.percentile(_simplex_lowrank, _lower_quantile, axis=0)
    _high_x = np.percentile(_simplex_lowrank, _upper_quantile, axis=0)
    _low_y = np.percentile(_simplex_hierprob, _lower_quantile, axis=0)
    _high_y = np.percentile(_simplex_hierprob, _upper_quantile, axis=0)

    _error_x = [
        _median_x - _low_x,
        _high_x - _median_x,
    ]
    _error_y = [
        _median_y - _low_y,
        _high_y - _median_y,
    ]

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4))

    _min_val = min(_median_x.min(), _median_y.min())
    _max_val = max(_median_x.max(), _median_y.max())
    _ax.plot(
        [_min_val, _max_val],
        [_min_val, _max_val],
        color="black",
        linestyle="--",
    )

    _ax.errorbar(
        _median_x,
        _median_y,
        xerr=_error_x,
        yerr=_error_y,
        fmt="o",
        ms=3,
        elinewidth=1,
        capsize=0,
        alpha=0.7,
        color="C0",
        ecolor="gray",
    )

    _ax.set_xlabel("composition low-rank model\n(shared $p$)")
    _ax.set_ylabel("composition low-rank\n(gene-specific $p_g$)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The medians fall almost exactly on the identity line: the two models agree on what fraction each gene occupies. This is reassuring—the modeling decision (shared vs. gene-specific $p$) did not materially change the compositional point estimates that would feed into differential expression.

    The error bars tell a more interesting story. The gene-specific $p_g$ model (vertical bars) has noticeably wider uncertainty than the shared-$p$ model (horizontal bars), especially for higher-expression genes. This makes sense:

    - **Shared $p$:** compositions depend only on $\underline{r}$, because the common $p$ cancels in the normalization. Each $r_g$ is well-constrained by thousands of cells, so compositional uncertainty is tight.
    - **Gene-specific $p_g$:** compositions now depend on both $r_g$ and $p_g$. The $r_g$ are still well-constrained, but each $p_g$ is informed by data from just one gene (plus the hierarchical prior pulling it toward the population mean). That per-gene $p_g$ uncertainty propagates directly into the compositional fractions.

    Is the wider uncertainty a problem? Not necessarily—it may be the model being more honest about what the data can actually tell us. But for a monoculture like Jurkat cells, where there is no strong biological reason to expect gene-to-gene variation in $p$, the extra uncertainty is likely a statistical cost of estimating parameters the data did not need. Bayesian model comparison (e.g., via marginal likelihood or leave-one-out cross-validation) can formalize this intuition—a topic we will revisit in a later tutorial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Beyond Gaussians: normalizing flow guides

    Every posterior approximation we have used so far—mean-field, low-rank, structured joint—works by optimizing a location-scale family in some (possibly transformed) space. In the unconstrained setting these are Gaussians passed through fixed bijections like `exp` or `sigmoid`; in the constrained setting they are Beta or LogNormal distributions. Either way, the shape of each marginal is predetermined: the optimizer can shift and stretch it, but it cannot bend it into something fundamentally different. If the true posterior has curved ridges, skewness, multimodality, or other geometry that these fixed families cannot represent, the approximation will always leave something on the table.

    Normalizing flows take a different approach entirely. A normalizing flow starts from a simple base distribution (a standard Normal) and pushes it through a chain of learned invertible transformations—small neural networks whose parameters are optimized alongside the ELBO. Each layer warps the density, and by stacking enough layers the flow can mold an initially spherical Gaussian into essentially any smooth distribution. If the true posterior has banana-shaped ridges, heavy tails, or asymmetric modes, the flow can learn to reproduce that geometry rather than forcing an elliptical approximation.

    `scribe` provides normalizing flow guides that work at the full gene-dimensional scale (tens of thousands of dimensions) through carefully stabilized coupling architectures. When combined with `joint_params`, the flow operates on the stacked parameter vector—here, the scalar $p$ concatenated with all $G$ gene-specific $r_g$ values—so a single flow chain can learn nonlinear cross-parameter and cross-gene correlations simultaneously.

    The flexibility comes with its own tradeoffs. Each flow layer contains a conditioner neural network with trainable weights, so the parameter count is much larger than a low-rank guide. Training can be slower and more sensitive to hyperparameters (learning rate, number of layers, activation function). And because flows are universal approximators in principle, they can also be universal overfitters in practice—absorbing noise as fake structure if not regularized carefully.

    In code, the new arguments are:

    - `guide_flow="affine_coupling"`: selects the flow architecture. Affine coupling layers split the input dimensions into two halves; one half stays fixed while a neural network predicts a shift and scale for the other half, then the roles swap. This is fast (both forward and inverse passes are parallel) and a good default.
    - `guide_flow_num_layers=8`: how many coupling layers to stack. More layers mean more expressive warping but also more parameters and longer training. Eight is a reasonable starting point.
    - `guide_flow_activation="gelu"`: the activation function inside the conditioner networks. GELU (Gaussian Error Linear Unit) is a smooth alternative to ReLU that often trains more stably in flow architectures.
    - `joint_params="biological"`: as before, this tells `scribe` to model $p$ and $\underline{r}$ jointly. With a flow guide, both are concatenated into a single vector and processed by the same flow chain, so the flow can learn arbitrary (including nonlinear) coupling.

    We return to the canonical parameterization with a scalar $p$ (no `prob_prior="gaussian"`). The flow is expressive enough in principle to represent whatever coupling exists between $p$ and $\underline{r}$ without needing the mean–odds reparameterization or gene-specific $p_g$ as structural crutches. Whether it actually converges to a better solution than the low-rank Gaussian is an empirical question.
    """)
    return


@app.cell
def _(adata, clear_caches, gc, out_dir, pickle, scribe):
    # Clear GPU memory from previous fits
    clear_caches()
    gc.collect()

    # Define parameterization
    _parameterization = "canonical"

    # Define output file path
    _out_path = out_dir / f"scribe_results_nbvcp-flow_{_parameterization}.pkl"

    if _out_path.exists():
        # Load model from pkl file
        with open(_out_path, "rb") as _f:
            results_flow = pickle.load(_f)
    else:
        # Fit basic model to data with variable capture probability fit per cell
        results_flow = scribe.fit(
            adata,
            variable_capture=True,
            parameterization=_parameterization,
            joint_params="biological",
            guide_flow="affine_coupling",
            guide_flow_activation="gelu",
            guide_flow_num_layers=8,
            n_steps=200_000,
            early_stopping={"enabled": True, "patience": 1000}
        )
        # Save the fitted model
        with open(_out_path, "wb") as _f:
            pickle.dump(results_flow, _f)

    results_flow
    return (results_flow,)


@app.cell
def _(results_flow, scribe):
    # Plot ELBO loss for normalizing flow fit
    scribe.viz.plot_loss(results_flow, figsize=(6, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us examine the corner plot for the same illustrative genes. Because the flow can learn nonlinear warping of the posterior, we might see shapes in the 2D panels that the low-rank Gaussian guide could not represent: asymmetric tails, curved ridges tracked more faithfully, or tighter concentration along the NB mean-dispersion constraint.
    """)
    return


@app.cell
def _(corner, gene_index, results_flow):
    _results_subset = results_flow[gene_index]

    # Export directly as a corner-ready matrix plus labels/metadata.
    _samples_2d, _labels, _ = _results_subset.get_posterior_matrix(
        n_samples=5_000,
        include=["p", "r"],
        exclude_deterministic=True,
        store_samples=False,
        convert_to_numpy=True,
    )

    corner.corner(_samples_2d, labels=_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A striking feature of the flow's corner plot is how much narrower the posteriors are compared to the low-rank Gaussian guide. The shared $p$ is pinned to a sliver roughly a thousand times tighter than the low-rank result, and the $r_g$ marginals have shrunk accordingly. Is the flow simply more precise, or is this a symptom of mode collapse?

    Both explanations are plausible, and good PPCs alone cannot distinguish them. The ELBO objective, $\text{KL}(q \| p)$, is mode-seeking: it penalizes the approximation for placing mass where the true posterior is low, but it does not penalize failing to cover regions where the true posterior is high. A flow with many trainable parameters can therefore "snap" to one narrow region of parameter space that scores well without being penalized for ignoring nearby mass. Because PPCs test the predictive distribution—which is dominated by the posterior mean, not the tails—overconfident posteriors can produce perfectly adequate count-level predictions.

    Two diagnostics can help resolve the ambiguity. First, compare the final ELBO values: if the flow achieves a materially better (less negative) ELBO than the low-rank guide, the tighter posterior is more likely a faithful approximation; if the ELBOs are comparable, the flow may have traded coverage for sharpness without gaining evidence. Second, run MCMC (which `scribe` supports) on a subset of genes and compare the marginal widths—that is the gold-standard check. Notice also that $p$ settled at a completely different value here than in the low-rank `mean_odds` fit; both give good PPCs because the capture probability can compensate. Such flat directions in the likelihood are exactly where mode-seeking variational families are most prone to overconfidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the PPCs for the normalizing flow fit. Because the flow operates in canonical $(r, p)$ coordinates with joint structure, it should be able to capture the ridge geometry that gave mean-field canonical fits trouble earlier.
    """)
    return


@app.cell
def _(adata, results_flow, scribe):
    scribe.viz.plot_ppc(
        results_flow,
        adata,
        n_genes=16,
        n_rows=4,
        figsize=(8, 8),
        n_samples=512,
    )
    return


@app.cell
def _(adata, results_flow, scribe):
    scribe.viz.plot_mean_calibration(
        results_flow, counts=adata, figsize=(4, 4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This tutorial walked through a progression of modeling and inference choices that `scribe` makes available for single-cell RNA-seq data, all applied to the same Jurkat monoculture dataset.

    ### What we explored

    1. **The negative binomial as a biophysically motivated count model.** Rather than adopting the NB for vague "overdispersion" reasons, `scribe` starts from the two-state promoter model where the NB arises as a steady-state distribution. We fitted the simplest version of this model (shared $p$, no capture adjustment) and used posterior predictive checks to see where it fell short.

    2. **Variable capture probability.** By giving each cell its own capture parameter $\nu_c$, we moved depth correction from an ad hoc preprocessing step into the generative model itself. PPCs improved substantially, and the capture-scaling plot provided a sanity check on how depth is being allocated across cells.

    3. **Reparameterization matters.** Switching from canonical $(r, p)$ to mean-odds $(\mu, \phi)$ coordinates did not change the generative model, but it changed the geometry the optimizer and variational guide navigate. The result was tighter, more honest posterior uncertainty and better PPCs—a reminder that inference is not just about the model but also about how you parameterize it.

    4. **Joint low-rank guides.** By upgrading the variational family from mean-field to a low-rank covariance structure, we let the approximation represent cross-gene and cross-parameter correlations. The marginal PPCs stayed good, but the corner plots revealed that the mean-field had been inflating marginal uncertainty to compensate for missing coupling. The low-rank guide recovered the true (narrow) uncertainty on shared parameters like $p$ and opened the door to gene-gene correlation heatmaps.

    5. **Hierarchical gene-specific $p_g$.** Relaxing the shared-$p$ assumption gave each gene its own success probability, regularized by a population-level prior. PPCs remained excellent, and the compositional point estimates barely changed—but the compositional uncertainty widened, reflecting the extra degrees of freedom. Whether that extra flexibility is worth the statistical cost is a question Bayesian model comparison can answer.

    6. **Normalizing flow guides.** As the most flexible variational family in this tutorial, flows can learn nonlinear posterior geometry that Gaussian families cannot represent. We demonstrated that `scribe` can train flow-based guides at genome scale in canonical coordinates, providing another tool for cases where simpler approximations leave structure on the table.

    ### What we did not explore

    Several important capabilities of `scribe` were deliberately left for other tutorials:

    - **Differential expression analysis.** `scribe`'s Bayesian DE framework—including centered and isometric log-ratio transformations, the local false sign rate (lfsr), and the posterior expected false discovery proportion (PEFP)—is the subject of a dedicated tutorial using a mixed-population dataset.
    - **Unsupervised cell-type discovery.** Mixture models and their interaction with the compositional framework are covered separately.
    - **Biology-informed capture priors.** When external information about total mRNA content is available, `scribe` can use it to anchor the capture probability more tightly—resolving the identifiability issue we noted in the capture-scaling plots.
    - **MCMC sampling.** All fits in this tutorial used stochastic variational inference (SVI). `scribe` also supports Hamiltonian Monte Carlo for gold-standard posterior exploration, at higher computational cost.
    - **Formal Bayesian model comparison.** We compared models qualitatively (PPCs, calibration, compositional agreement); quantitative comparison via marginal likelihoods or predictive metrics is a natural follow-up.

    ### The broader message

    The progression in this tutorial—from a bare count model to structured variational families with tens of thousands of parameters—illustrates a core design principle of `scribe`: modeling assumptions should be *explicit, composable, and testable*. Each ingredient (capture, parameterization, guide structure, hierarchical priors) can be turned on or off independently, and posterior predictive checks provide a consistent diagnostic language for asking whether each addition actually helps. The goal is not to find the one "right" model, but to let the analyst build up complexity incrementally, checking at each step whether the data justify the added flexibility.
    """)
    return


if __name__ == "__main__":
    app.run()
