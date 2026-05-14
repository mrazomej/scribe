import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mixture model + differential expression with `scribe`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this tutorial we will use `scribe` to do two things that are routine in a single-cell analysis pipeline:

    1. **Unsupervised cell-type discovery** — fit a two-component mixture model to an unlabeled dataset and let it assign each cell to a cluster.
    2. **Differential expression between the discovered cell types**, following the compositional framework described in the `scribe` paper.

    If you are coming from a standard Scanpy/Seurat workflow, the mental model to carry in is:

    - Instead of `scanpy.pp.normalize_total` → `scanpy.pp.log1p` → `scanpy.tl.leiden` → `scanpy.tl.rank_genes_groups`, we fit **one generative model** of the raw UMI counts that simultaneously handles normalization, clustering, and uncertainty quantification.
    - Instead of a single point estimate per gene per cluster, we get a **full posterior distribution** — so every downstream quantity (mean expression, fold change, differential-expression call) comes with a natural measure of uncertainty.
    - Instead of p-values, we use **local false sign rates (lfsr)** — a Bayesian quantity with a direct probabilistic interpretation: "the posterior probability that I have the direction of the effect wrong."

    For our running example we use a public 10x Genomics dataset ([link](https://www.10xgenomics.com/datasets/50-percent-50-percent-jurkat-293-t-cell-mixture-1-standard-1-1-0)) consisting of a **50/50 mixture of Jurkat and HEK 293T cells**. The true mixture proportions are known, which makes it an ideal pedagogical dataset: we can check whether the model recovers the expected 50/50 split and whether the differentially expressed genes look like the textbook markers of each cell type.

    Key facts about the dataset:

    - ~3,400 cells
    - Sequenced on Illumina HiSeq 2500 Rapid Run V2, ~33,000 reads per cell
    - 98 bp read 1 (transcript), 8 bp I5 sample barcode, 14 bp I7 GemCode barcode, 10 bp read 2 (UMI)

    Let's begin by importing the necessary packages for our analysis.
    """)
    return


@app.cell
def _():
    # Import basic packages
    from pathlib import Path
    import pickle

    # Import our main package
    import scribe

    # Import package to release GPU memory
    from jax import clear_caches
    import gc

    # Import useful tools
    import numpy as np
    import scanpy as sc
    import jax.numpy as jnp

    # Import plotting packages
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set our plotting style (totally optional)
    scribe.viz.matplotlib_style()
    return Path, clear_caches, gc, jnp, np, pickle, plt, scribe, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, let's load the data into memory and perform a simple exploratory analysis.
    """)
    return


@app.cell
def _(Path, scribe):
    # Define data directory
    data_dir = Path(
        "/path/to/5050_jurkat-293t/"
    )

    # Load the data
    adata = scribe.data_loader.load_and_preprocess_anndata(
        data_dir, return_jax=False
    )

    adata
    return adata, data_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first plot we always look at is the distribution of total UMI counts per cell. Beyond the usual quality-control role, this plot is also a modelling decision: if the library size varies a lot from cell to cell, then every cell is effectively sequenced at its own depth and we need the model to know that explicitly. In `scribe` that translates to enabling a **per-cell capture probability** (the `variable_capture=True` flag below), which is the principled analogue of "dividing by total counts" in traditional pipelines — only here it is a parameter of the generative model with its own posterior, not a preprocessing step.
    """)
    return


@app.cell
def _(adata, plt, sns):
    # Initialize a figure
    _fig, _ax = plt.subplots(figsize=(5, 4))
    # Plot the histogram of total UMI counts
    sns.ecdfplot(adata.X.sum(axis=1), ax=_ax)
    # Label the axes
    _ax.set_xlabel("total UMI count per cell")
    _ax.set_ylabel("ECDF")
    # Set y-axis limits
    _ax.set_ylim(-0.01, 1.01)
    # Turn off legend
    _ax.legend_.remove()
    # Show the plot
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With nearly an order of magnitude difference between the smallest and largest library sizes in this dataset, there is no doubt that we will need to model variable capture probabilities on a per-cell basis.

    Given that we have no annotations for which cell is which, this is a perfect opportunity to show `scribe`'s cell annotation capabilities via mixture models. However, we must emphasize that prior knowledge from biologists is incredibly valuable. In our experience, `scribe` is able to resolve cell types at a very coarse-grained level. Finer details require domain expertise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting a mixture model with variable capture probability
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now fit our first `scribe` model. Before looking at the code, let's unpack what each argument does.

    **What model are we fitting?** At its core, `scribe` models UMI counts with a **Negative Binomial distribution** — the same distribution that motivates DESeq2 and edgeR, and one that can be derived from first principles from a two-state promoter model of transcription. Each gene has a parameter $r_g$ that captures its **transcriptional burstiness** (genes that fire in big bursts have larger $r_g$), and each cell has a parameter $\hat{p}_c$ that absorbs both intrinsic transcription and technical capture efficiency.

    **Why `variable_capture=True`?** Because the cells differ substantially in library size, we let every cell have its own capture probability $\nu_c \in (0, 1)$. Internally this produces a cell-specific effective success probability
    $$\hat{p}_c = \frac{p}{\nu_c + p(1 - \nu_c)},$$
    which is `scribe`'s principled replacement for ad-hoc size-factor normalization.

    **Why `parameterization="mean_odds"`?** Instead of fitting the raw Negative Binomial parameters $(r_g, p)$, we fit the **gene mean expression** $\mu_g$ together with an **odds ratio** $\phi = p / (1 - p)$. The two views are algebraically equivalent — $r_g = \mu_g \phi$ and $p = 1/(1+\phi)$ — but they have very different inference behaviour. In raw $(r_g, p)$ coordinates, many pairs produce the same mean, which creates a long banana-shaped ridge in the posterior that is painful to sample. The mean–odds view breaks that degeneracy: $\mu_g$ is tightly pinned by the data (it *is* the average), and the odds ratio is left to pick up the residual variance.

    **Why `n_components=2`?** This turns the model into a **mixture model with two clusters**. We do *not* tell the model which cell is Jurkat and which is 293T — instead, each component learns its own gene-expression profile, and every cell receives a posterior probability of belonging to each cluster. This is a "soft" clustering: a cell can be 95%/5% or 60%/40% between the two components, and that uncertainty is preserved in every downstream computation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `early_stopping` argument does exactly what it says on the tin: it watches the training objective and stops once the loss has plateaued for `patience` steps (`1000` here). The `checkpoint_dir` path also tells `scribe` where to drop periodic checkpoints so that long runs can be resumed if they get interrupted. You can leave early stopping off for short experiments; it is most useful when you do not want to hand-tune the number of optimization steps.
    """)
    return


@app.cell
def _(adata, data_dir, pickle, scribe):
    # Define output directory
    out_dir = data_dir / "scribe_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define parameterization
    _parameterization = "mean_odds"

    # Define output file path
    _out_path = out_dir / f"scribe_results_nbvcp_{_parameterization}.pkl"

    if _out_path.exists():
        # Load model from pkl file
        with open(_out_path, "rb") as f:
            results_nbvcp = pickle.load(f)
    else:
        # Fit basic model to data with variable capture probability fit per cell
        results_nbvcp = scribe.fit(
            adata,
            variable_capture=True,
            parameterization=_parameterization,
            n_components=2,
            early_stopping={
                "enabled": True,
                "patience": 1000,
                "checkpoint_dir": str(_out_path.with_suffix("")),
            },
        )
        # Save the fitted model
        with open(_out_path, "wb") as f:
            pickle.dump(results_nbvcp, f)

    results_nbvcp
    return (results_nbvcp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To check whether training successfully converged, we examine the **[ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) loss** curve. The ELBO is the quantity that variational inference is optimizing under the hood — roughly, the negative log-evidence plus a gap that measures how well the approximate posterior matches the true one. For a sanity check we do not need to read it quantitatively: a clean run produces a curve that **drops quickly at first and then flattens out**, and that is exactly what we want to see. Spikes, slow drifts upward, or oscillations that never settle are warnings worth investigating.
    """)
    return


@app.cell
def _(results_nbvcp, scribe):
    # Plot ELBO loss
    scribe.viz.plot_loss(results_nbvcp, figsize=(7, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The training loss has converged, but that alone does not prove the model describes the *counts* well. **Posterior predictive checks (PPCs)** address that directly: we simulate new UMI data from the fitted model and compare its distribution to the histogram of what was actually observed. If the model’s assumptions and fit are reasonable, synthetic replicates should track the real data; systematic shifts or heavy tails in the observations that the model never reproduces flag misspecification—regardless of how tidy downstream plots look.

    For mixture models, `scribe.viz.plot_mixture_ppc_overview` focuses on genes that are both **expressed across a wide dynamic range** and **strongly separated between components**. Genes are grouped into bins by overall median expression (so you see low-, mid-, and high-expression loci), and within each bin the **largest component-to-component log fold-changes** (from the MAP estimates) are prioritized. Those genes are exactly where differential expression and mixing interact: they stress-test whether the mixture’s generative story matches the marginal count distributions you actually measured.
    """)
    return


@app.cell
def _(adata, results_nbvcp, scribe):
    scribe.viz.plot_mixture_ppc_overview(
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
    Nice — the overview PPC already gave us a sanity check on the **whole** library. Next, **`scribe.viz.plot_mixture_ppc_comparison`** is an easy way to sanity-check the **mixture itself**: are the two cell types behaving the way we think?

    You get the **same handful of interesting genes** as before (picked to span expression levels and to differ a lot between components). On each small panel, the **shaded regions** are synthetic count distributions from **component 1** and **component 2**—stacked on top of each other so you can compare them at a glance.

    The model also makes a simple call on every cell: “you’re more likely **this** type than **that** type.” Those are **MAP assignments** (each cell goes to its best-matching component). The **colored lines** are the **real UMI histograms** for just the cells assigned to each component, using colors that line up with the two predictive bands.

    So in plain language: *if our Jurkat vs. 293T story is decent, each colored data trace should hug “its” shaded region more than the other one—and the two shaded regions should sit where the two peaks actually live.* This view is only set up for a **two-component** fit; with more components you’d lean on the per-component plots instead.
    """)
    return


@app.cell
def _(adata, clear_caches, gc, results_nbvcp, scribe):
    # Clear gpu memory
    clear_caches()
    gc.collect()

    scribe.viz.plot_mixture_ppc_comparison(
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
    This plot clearly shows that the model is able to separate the two different cell types and predict their independent distributions.

    A second quick sanity check is available from the model itself. Because this dataset is advertised as a **50/50** mixture, the model's estimate of the **mixing weights** (the prior probability that a randomly chosen cell belongs to each component) should come out roughly balanced. Those weights live in the MAP dictionary under `mixing_weights`.
    """)
    return


@app.cell
def _(np, results_nbvcp):
    # Pull MAP mixing weights from the fitted mixture model.  For a balanced
    # 50/50 mixture we expect both entries to be close to 0.5 - anything very
    # lopsided would hint at a convergence problem or a genuinely unbalanced
    # dataset.
    _mixing_weights = np.asarray(
        results_nbvcp.get_map(descriptive_names=True)["mixing_weights"]
    )
    print(
        "MAP mixing weights (component 0, component 1):",
        np.round(_mixing_weights, 3),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That is remarkably close! We have one cell type with ≈ 53% and the other one with 47%.

    For a **bigger-picture** view, we can also look at a **UMAP** scatter. That sounds like “the” global picture of the library—but it is worth being a little cautious.

    **UMAP is a visualization, not a ground-truth summary of biology.** It compresses thousands of genes into two dimensions with nonlinear geometry, neighborhood choices, and other knobs, so distances and cluster shapes are easy to over-interpret. It also depends on a standard single-cell recipe (filtering, normalization, HVGs, sometimes PCA) that is **not** the same objective the generative model was trained on.

    That said, for a **quick global sanity check**, it can still be useful. In `scribe`, the UMAP is fit on the **experimental** counts after a Scanpy-style pipeline; **synthetic** counts are then simulated from the fitted model, passed through the **same** preprocessing and PCA, and **projected** with the **same** UMAP transform—so you are asking a simple question: *do model-generated cells land in roughly the same regions as the real ones?* Treat overlaps as encouraging and gross mismatches as a prompt to dig deeper (starting with count-level checks like the PPCs above), not as proof by themselves.
    """)
    return


@app.cell
def _(adata, clear_caches, gc, results_nbvcp, scribe):
    # Clear GPU memory
    clear_caches()
    gc.collect()

    scribe.viz.plot_umap(results_nbvcp, adata, figsize=(4, 4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When we overlay synthetic draws from the fitted model on the same UMAP used for the observed cells, the two clouds line up surprisingly well. That agreement is worth pausing on, because the variational approximation we used is *mean-field*: each gene’s approximate posterior is updated as if the others were summarized by their averages, so the approximation does not explicitly model residual **gene–gene dependence** (beyond what is mediated by shared latent structure such as cell type or batch). In other words, we are prioritizing accurate **marginal** distributions per gene rather than a fully flexible **joint** distribution over all genes at once.

    For many downstream tasks—especially **gene-level differential expression**, where the goal is to rank genes by shifts between conditions—those marginals are often exactly what we need. A mean-field fit can therefore be a practical sweet spot: enough flexibility to capture strong biological signal in per-gene expression while keeping inference and sampling tractable on high-dimensional count data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's start comparing the two components quantitatively. Because we chose the mean–odds parameterization, the most interpretable quantity per gene is $\mu_g$: the **gene mean expression** in each component. Recall that `scribe` does not return a single number per parameter — it returns an **approximate posterior distribution**. That posterior is the thing that carries uncertainty.

    A convenient single-number summary is the **[Maximum a posteriori (MAP)](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) estimate**, the mode of the posterior. Think of it as the Bayesian analogue of a maximum-likelihood point estimate — useful for scatter plots and quick comparisons, as long as we remember that the full posterior has more to say.

    We pull MAP estimates with `results_nbvcp.get_map(descriptive_names=True)`. The `descriptive_names=True` flag renames the internal parameters (`mu`, `phi`, ...) to human-readable keys (`mean_expression`, `odds`, ...).
    """)
    return


@app.cell
def _(results_nbvcp):
    #  Extract map from results
    results_map = results_nbvcp.get_map(descriptive_names=True)

    results_map.keys()
    return (results_map,)


@app.cell
def _(mo):
    mo.md(r"""
    We can the extract the parameters we care about. Let's look at the shape of the resulting array.
    """)
    return


@app.cell
def _(results_map):
    results_mean = results_map["mean_expression"]

    results_mean.shape
    return (results_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This array has dimensions (`n_components`, `n_genes`). Thus, we can easily index each component and plot the comparison of these mean expression values.
    """)
    return


@app.cell
def _(plt, results_mean):
    # Initialize figure
    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4))

    # Plot scatter of mean expression MAP
    _ax.scatter(
        results_mean[0, :],
        results_mean[1, :],
        s=3,
    )

    # Add identity line as black dashed line from the min to the max of the array
    _min_val = min(results_mean[0, :].min(), results_mean[1, :].min())
    _max_val = max(results_mean[0, :].max(), results_mean[1, :].max())
    _ax.plot(
        [_min_val, _max_val],
        [_min_val, _max_val],
        color="black",
        linestyle="--",
    )

    # Label axis
    _ax.set_xlabel(r"component A mean expression ($\mu_{\text{MAP}}^{(A)}$)")
    _ax.set_ylabel(r"component B mean expression ($\mu_{\text{MAP}}^{(B)}$)")

    # Set title
    _ax.set_title("Mean expression comparison")

    # Set to log-log scale
    _ax.set_xscale("log")
    _ax.set_yscale("log")

    # Show plot
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As expected, the two components’ MAP means track each other closely on a log–log plot, with a clear tail of genes that shift between components. Those MAP values are only **summaries** of the posterior, however—each point is a single best guess for $\mu$ per gene and component. To show **uncertainty**, we will redraw the same comparison with vertical and horizontal “error bars” derived from the full posterior over $\mu$.

    In `scribe`, approximate posterior draws are available from the fitted results object via **`get_posterior_samples`**. Here we request many draws (for example `n_samples=512`) and pull the **`"mu"`** tensor so we obtain an array of shape `(n_samples, n_components, n_genes)` matching the mean-expression parameters. We then summarize each gene’s marginal posterior along the sample axis—for instance with the 2.5th and 97.5th **percentiles**—to obtain a **95% credible interval** per gene and component. A **credible interval** answers a *Bayesian* question: “Given the model and data, with 95% posterior probability, $\mu$ lies between these two values.” That is different from a **confidence interval** in *frequentist* inference, which is tied to a hypothetical repetition of the experiment: “If we repeated sampling many times and built intervals the same way, about 95% of them would cover the true fixed parameter.” For communication in this notebook, “uncertainty bars from posterior quantiles” is the right mental model.

    We attach those intervals to the same MAP scatter as asymmetric error bars (distance from MAP down to the lower quantile and up to the upper quantile on each axis).

    Let's first obtain the samples. We use the `store-samples=False`argument to avoid storing large tensors directly on the `results_nbvcp` object to save GPU memory.
    """)
    return


@app.cell
def _(np, results_nbvcp):
    # Generate posterior samples
    mean_samples = np.array(
        results_nbvcp.get_posterior_samples(n_samples=512, store_samples=False)[
            "mu"
        ]
    )
    return (mean_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we are ready to generate the plot with the corresponding errorbars.
    """)
    return


@app.cell
def _(mean_samples, np, plt, results_mean):
    # Define quantiles for error bars (easy to edit)
    _lower_quantile = 2.5
    _upper_quantile = 97.5

    # Compute quantiles along the samples axis for both components
    _mean_samples_lower = np.percentile(mean_samples, _lower_quantile, axis=0)
    _mean_samples_upper = np.percentile(mean_samples, _upper_quantile, axis=0)

    # The mean_samples array has shape (n_samples, n_components, n_genes)
    # Compute error bar lengths for each component and gene
    _error_a = [
        results_mean[0, :] - _mean_samples_lower[0, :],  # lower errors
        _mean_samples_upper[0, :] - results_mean[0, :],  # upper errors
    ]
    _error_b = [
        results_mean[1, :] - _mean_samples_lower[1, :],  # lower errors
        _mean_samples_upper[1, :] - results_mean[1, :],  # upper errors
    ]

    # Initialize figure
    _fig2, _ax2 = plt.subplots(1, 1, figsize=(4, 4))

    # Plot with errorbars for each gene
    _ax2.errorbar(
        results_mean[0, :],
        results_mean[1, :],
        xerr=_error_a,
        yerr=_error_b,
        fmt="o",
        ms=3,
        elinewidth=1,
        capsize=0,
        alpha=0.7,
        color="C0",
        ecolor="gray",
    )

    # Add identity line as black dashed line from the min to the max of the array
    _min_val2 = min(results_mean[0, :].min(), results_mean[1, :].min())
    _max_val2 = max(results_mean[0, :].max(), results_mean[1, :].max())
    _ax2.plot(
        [_min_val2, _max_val2],
        [_min_val2, _max_val2],
        color="black",
        linestyle="--",
    )

    # Label axis
    _ax2.set_xlabel(r"component A mean expression ($\mu_{\text{MAP}}^{(A)}$)")
    _ax2.set_ylabel(r"component B mean expression ($\mu_{\text{MAP}}^{(B)}$)")

    # Set title
    _ax2.set_title("Mean expression comparison (with error bars)")

    # Set to log-log scale
    _ax2.set_xscale("log")
    _ax2.set_yscale("log")

    # Show plot
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Differential Expression Analysis

    Most single-cell DE workflows follow a familiar recipe: take raw UMI counts, divide by the total per cell, optionally log-transform, then run gene-by-gene tests. Normalization is treated as a practical preprocessing step — a way to make cells comparable — **before** the real statistics begin.

    `scribe` takes a different view. Rather than normalizing first and asking questions later, the model's generative structure *itself* tells us what we should be comparing.

    ### The argument in one paragraph

    Under the Negative Binomial model of transcript counts, the full profile of a cell factors exactly into two pieces: **(i)** the **total UMI count** (how deeply the cell was sequenced) and **(ii)** a **composition on the gene simplex** — the fractions $\underline{\rho} = (\rho_1, \ldots, \rho_G)$ that say how the cell's transcription is allocated across genes. The depth piece is a nuisance; the composition piece is the biology. Crucially, these compositions are **latent parameters of the model**, not a preprocessing artefact, and `scribe` gives us a **posterior distribution** over them.

    So the differential-expression question becomes: *how do the two components' posterior compositions $\underline{\rho}^{(A)}$ and $\underline{\rho}^{(B)}$ differ, gene by gene, while properly respecting compositional geometry?*

    ### Why CLR coordinates?

    Compositions live on the simplex, not on the real line, so we cannot just subtract them. The standard fix is the **centered log-ratio (CLR)** transform:
    $$
    z_{\mathrm{CLR},g} \;=\; \log \rho_g \;-\; \frac{1}{G}\sum_{j=1}^{G}\log \rho_j.
    $$
    In words, each gene's log-fraction is compared to the **geometric mean across all genes**. Two features make this the natural choice here: it is **reference-free** (no gene is privileged as the denominator), and it turns the multiplicative geometry of the simplex into **additive** contrasts — so differences between conditions behave like ordinary numbers.

    Because `scribe` carries a full posterior over $\underline{\rho}$, every posterior draw can be transformed into CLR coordinates, giving us a posterior over **CLR contrasts** $\Delta_g = z_{\mathrm{CLR},g}^{(A)} - z_{\mathrm{CLR},g}^{(B)}$. All differential-expression statistics below come from direct counting over these draws — no Gaussian assumptions, no p-values, just Monte Carlo summaries of a posterior we already have.

    > For the full derivation see the supplementary material of the paper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The empirical comparison in `scribe` works by turning each posterior draw of the expression parameters into **relative abundances**—normalized compositions on the gene simplex—and then contrasting those compositions between conditions in **CLR** space. To do that reliably we first need a **large** set of posterior samples.

    We generate those draws with **`get_posterior_samples`** on the fitted results object and **keep them on the results** so the DE routines can read them directly. We set **`convert_to_numpy=True`** so the stored tensors are converted to plain **NumPy arrays** after sampling, which frees **GPU memory** immediately for later work. All downstream DE functions use array-backend dispatch and transparently run on the NumPy/SciPy stack when given NumPy inputs. Finally, because each mixture **component** is a separate condition in this comparison, we will extract component-specific views—either with **`get_component(k)`** or the shorthand **`results[:, k]`** (all genes, component `k`)—so the compositional contrast is computed between the right cell types.
    """)
    return


@app.cell
def _(results_nbvcp):
    # Define number of posterior samples
    n_samples = 10_000

    # Sample posterior and store in results object
    results_nbvcp.get_posterior_samples(
        n_samples=n_samples, convert_to_numpy=True
    )

    # Extract components
    component0 = results_nbvcp.get_component(0)
    component1 = results_nbvcp[:, 1]
    return component0, component1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's call `scribe.de.compare` with the default arguments. The ones that matter most for this notebook are:

    - **`method="empirical"`**: Use the Monte Carlo DE path: build CLR contrasts from posterior samples and summarize with counting-based posterior quantities (not the closed-form Gaussian **`"parametric"`** path, nor **`"shrinkage"`**).

    - **`n_samples_dirichlet=1`**: Take **one** Dirichlet–multinomial draw on the gene simplex **per** posterior sample of the concentration parameters. Raising this fans out more simplex draws per posterior draw (more averaging of Monte Carlo noise, higher cost).

    - **`batch_size=2048`**: Process compositional sampling / CLR differencing in **chunks** of this many rows to cap memory use on large gene sets and many posterior samples.
    """)
    return


@app.cell
def _(component0, component1, scribe):
    # Run differential expression analysis
    results_de = scribe.de.compare(model_A=component0, model_B=component1)

    results_de
    return (results_de,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before summarizing the differential expression results, we set three parameters that control how we threshold and report the analysis. They map directly onto choices a biologist already makes with tools like DESeq2 or edgeR (expression floor, fold-change cutoff, FDR target), with small but important Bayesian twists.

    - **`MIN_MEAN_EXPRESSION = 5.0`** — the minimum mean UMI count a gene needs to be analyzed individually. Genes that fall below this floor are **not dropped**: they are pooled into a single "other" pseudo-gene so the simplex still sums to one. Why bother? Because the CLR transform divides by the geometric mean of log-fractions across *all* genes, and the thousands of nearly-unexpressed genes in a typical dataset have very negative, very noisy log-fractions that make that geometric mean jitter from one posterior sample to the next. That jitter propagates into the CLR contrast and inflates uncertainty on the genes we actually care about. Aggregating low-expression genes into a single pooled component stabilizes the reference, and thanks to the **Dirichlet closure property** (summing a subset of Dirichlet components gives another Dirichlet) the aggregation is **exact** — no information loss. The full argument and formula are in the supplementary material.

    - **`TAU = log(2)`** — the minimum CLR shift we treat as biologically meaningful. In CLR space, $\log 2$ is roughly a doubling of a gene's fraction relative to the geometric mean of the expressed transcriptome: the Bayesian-compositional analogue of the $\log_2$ fold-change cutoff familiar from volcano plots.

    - **`TARGET_PEFP = 0.05`** — the desired **Posterior Expected False discovery Proportion**, the Bayesian analogue of the 5% FDR that biologists routinely control for. Instead of ranking genes by p-values and applying Benjamini–Hochberg, `scribe` ranks them by the **local false sign rate (lfsr)** — the posterior probability that we have the *direction* of the effect wrong. An lfsr of 0.02 means "given the data, there is a 2% chance the sign is wrong." The PEFP is the average lfsr across the declared DE set, and `target_pefp=0.05` is the threshold that keeps that average below 5%.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `scribe` DE interface is built around two chained steps. We first call `results_de.set_expression_threshold(MIN_MEAN_EXPRESSION)`, which applies the expression mask in-place on the results object—pooling filtered genes into the "other" pseudo-gene before any CLR coordinates are formed. We then call `.to_dataframe(tau=TAU, target_pefp=TARGET_PEFP, metrics="clr")` to convert the masked results into a tidy pandas DataFrame. The `tau` and `target_pefp` arguments are passed directly here rather than set globally, making it easy to explore different effect-size cutoffs or FDR levels without re-running the sampling. Setting `metrics="clr"` tells the method to report gene-level statistics in CLR space—posterior mean shift, posterior standard deviation, lfsr, and the PEFP-controlled call—as opposed to other coordinate systems available in the package.
    """)
    return


@app.cell
def _(jnp, results_de):
    # Defie minimum mean expression
    MIN_MEAN_EXPRESSION = 5.0
    # Define effective CLR change (TAU)
    TAU = jnp.log(2.0)
    # Define target PEFP
    TARGET_PEFP = 0.05

    # Filter results based on expression
    results_de_filtered = results_de.set_expression_threshold(
        MIN_MEAN_EXPRESSION
    )

    # CLR output stays on the built-in path.
    df_clr = results_de_filtered.to_dataframe(
        tau=TAU,
        target_pefp=TARGET_PEFP,
        metrics="clr",
    )

    df_clr.head()
    return TARGET_PEFP, TAU, df_clr, results_de_filtered


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now visualize how each gene’s **average compositional expression** differs between the two mixture components using `scribe.viz.plot_de_mean_expression`. With the default `mode="clr"`, every point is one retained gene (after the expression mask and `"other"` pooling). The horizontal axis is the **posterior summary of mean CLR expression in component A**; the vertical axis is the same for **component B**. Those are the gene-level **CLR means** exported with the DE table—not raw NB $\mu$ on the natural scale—so the plot lives in the same **compositional** coordinate system used for the differential contrasts.

    Interpreting the geometry is straightforward. Genes whose average abundance is **similar** in the two cell types lie **near the dashed identity line** $y = x$ on the **log–log** axes. Genes that fall **away from that line** are driven off-diagonal because one cell type allocates a systematically larger or smaller **fraction** of its transcriptome to that gene (relative to the CLR reference), which is exactly the kind of pattern differential expression is meant to surface.

    The call passes **`tau=TAU`** and **`target_pefp=TARGET_PEFP`** through to the same dataframe export logic as `to_dataframe`: `scribe` combines your **minimum meaningful CLR shift** ($\tau$) with a **posterior expected false discovery proportion (PEFP)** target to decide which genes are **called** differentially expressed. On the figure, genes that **fail** that joint call are drawn in **light gray**; genes that **pass** are highlighted in **dark red**, so the plot doubles as a quick **DE hit list** aligned with the thresholds you set above—without replacing the full table of posterior means, uncertainties, and local false sign rates stored in `df_clr`.
    """)
    return


@app.cell
def _(TARGET_PEFP, TAU, results_de_filtered, scribe):
    scribe.viz.plot_de_mean_expression(
        results_de_filtered, tau=TAU, target_pefp=TARGET_PEFP, figsize=(4, 4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The scatter of mean expression in CLR space is a useful first look, but each point is still a **summary** of a richer object: for every gene, `scribe` carries a **full posterior distribution** of CLR coordinates (and hence of the contrast between cell types) over posterior samples. The figures therefore emphasize interpretable point summaries—typically posterior means—while the underlying uncertainty lives in those draws.

    A familiar way to read the same two ingredients—**how large is the shift?** and **how confident are we?**—is a **volcano plot**. In classical **frequentist** differential expression, one common layout puts an **effect size** on the horizontal axis—often a log fold change between conditions, e.g. $\log_2$ fold change—and a **significance score** on the vertical axis, almost always
    $$
    y = -\log_{10} p,
    $$
    or the same transformation applied to an **FDR-adjusted** $p$-value (e.g. Benjamini–Hochberg $q$). Small $p$-values become large $y$, so genes that are both **strong** (far from zero on $x$) and **statistically extreme** under the chosen test **float upward** on the plot, producing the characteristic “erupting” shape.

    That frequentist picture answers a sampling-based question: *under a null hypothesis and a chosen error model, how surprising is this fold change?* The $y$-axis is tied to a **$p$-value**: the probability, under the null, of seeing a test statistic at least as extreme as the one observed.

    `scribe`’s **Bayesian** volcano keeps the **same visual grammar**—effect on $x$, evidence on $y$—but swaps the ingredients for quantities defined **with respect to the posterior**:

    - **Horizontal axis ($x$).** We plot the **posterior mean CLR contrast** between the two conditions (components), i.e. the average shift in CLR coordinates for that gene. In compositional terms this is the natural analogue of a **fold-change–style contrast**, but expressed in **CLR geometry** on the simplex rather than as a raw ratio of normalized counts.

    - **Vertical axis ($y$).** Instead of $-\log_{10} p$, we use
      $$
      y = -\log_{10}(\mathrm{lfsr}),
      $$
      where **lfsr** is the **local false sign rate**: the posterior probability that the **sign** of the differential effect is wrong if we call that gene differentially expressed. Large $y$ therefore means **small lfsr**, i.e. high posterior confidence in the **direction** of the shift—not a frequentist tail probability under a null.

    So the two volcano styles are **parallel in layout** but **different in meaning**: the traditional plot ranks genes by compatibility with a **null sampling model** via $p$-values (optionally FDR-adjusted); `scribe` ranks genes by **posterior uncertainty about the direction of change** via the lfsr, with the CLR contrast as the effect-size coordinate. In both cases, points in the **upper corners** are the usual suspects—large shifts with strong statistical support—but the support is **frequentist significance** in one case and **Bayesian sign certainty** in the other.
    """)
    return


@app.cell
def _(TARGET_PEFP, TAU, results_de_filtered, scribe):
    scribe.viz.plot_de_volcano(
        results_de_filtered, tau=TAU, target_pefp=TARGET_PEFP, figsize=(4, 4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The volcano plot is great for getting a sense of the global structure, but at some point we want to look at **actual gene names**. Let's sort the DE table by **local false sign rate** (lowest first — these are the genes for which the direction of the effect is most confident), and inspect the top hits in each direction.

    In the DataFrame, `clr_delta_mean` is the posterior mean CLR contrast (positive → higher in component A; negative → higher in component B), `clr_lfsr` is the lfsr, and `clr_is_de` is the boolean DE call that combines `tau` and `target_pefp`. The mean expression in each component is available in `clr_mean_expression_A` / `clr_mean_expression_B` for context.
    """)
    return


@app.cell
def _(TARGET_PEFP, TAU, df_clr):
    # Sort the DE table by lfsr ascending so the most confident calls appear
    # first.  We keep only DE-called genes (clr_is_de=True) because those are
    # the ones that pass both the effect-size (tau) and error-control
    # (target_pefp) thresholds simultaneously.
    _de_hits = df_clr.loc[df_clr["clr_is_de"]].copy()

    # Split by direction of the effect so we can display "up in A" and
    # "up in B" top hits separately - this is usually what a biologist
    # asks for when scanning for marker genes.
    _top_up_in_A = (
        _de_hits[_de_hits["clr_delta_mean"] > 0]
        .sort_values("clr_lfsr")
        .head(15)
    )
    _top_up_in_B = (
        _de_hits[_de_hits["clr_delta_mean"] < 0]
        .sort_values("clr_lfsr")
        .head(15)
    )

    print(
        f"Total DE-called genes at PEFP={float(TARGET_PEFP):.2f}, "
        f"tau={float(TAU):.3f}: {len(_de_hits)}"
    )
    print("\nTop 15 genes up in component A (check Jurkat or 293T markers):")
    print(
        _top_up_in_A[["gene", "clr_delta_mean", "clr_lfsr"]].to_string(
            index=False
        )
    )
    print("\nTop 15 genes up in component B:")
    print(
        _top_up_in_B[["gene", "clr_delta_mean", "clr_lfsr"]].to_string(
            index=False
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the output: which component is which?

    Remember that we never told `scribe` which cells were Jurkat and which were 293T — the labels "component A" and "component B" are assigned by the model in arbitrary order. Now that we have a ranked gene list we can actually **annotate** each component biologically.

    **Component A looks like Jurkat (T-cell line).** The up-regulated hits are dominated by canonical T-lymphocyte markers:

    - **LCK** — lymphocyte-specific protein tyrosine kinase, the classical Jurkat marker.
    - **LEF1** — a lymphoid enhancer-binding factor essential for T-cell development.
    - **CD1E**, **FYB**, **ITM2A** — all T-cell-associated surface/signaling proteins.
    - **PSMB8**, **ISG15** — interferon-inducible / immunoproteasome genes typical of immune cells.

    This is exactly the kind of signature you would hope to see for a T-ALL-derived line.

    **Component B looks like HEK 293T (embryonic kidney / neural-crest-like).** The hits here are a mix of kidney-epithelial, translation, and — interestingly — neuronal markers:

    - **CA2** (carbonic anhydrase II), **CST3** (cystatin C), **MDK** (midkine) — classical kidney / embryonic genes.
    - **UCHL1**, **GAL**, **PCSK1N** — neuroendocrine / neural markers. This is a real biological curiosity: despite being called "embryonic kidney," HEK 293 cells express a surprisingly neural gene program, and some analyses argue they are closer in lineage to a neural-crest–derived cell than to kidney epithelium. The model picks this up directly from the counts.

    **The cleanest sanity check of all: XIST.** XIST is the long non-coding RNA that coats the inactive X chromosome in female cells. It appears strongly up in component B, and is essentially absent in component A. Jurkat is a **male** (XY) cell line; HEK 293T is **female** (XX). So the model has independently recovered a sex-chromosome-based cell-line difference that no single-cell clustering tool could "cheat" on — it comes straight from the biology.

    ### What this tells us about the model

    Without a single supervised label, `scribe` has:

    1. Recovered the known **50/50** mixing proportion (check the MAP mixing weights above).
    2. Cleanly separated two cell lines in posterior space, confirmed by count-level PPCs.
    3. Returned a differential-expression table whose top hits are textbook markers, with **lfsr = 0.0** on every single one — i.e., the posterior has essentially no ambiguity about the direction of the effect.

    ### Where to go next

    - **Rerun with more components** (`n_components=3, 4, ...`) if the PPCs or UMAP suggest substructure within a single cluster (e.g., cell-cycle phases within Jurkat).
    - **Use `scribe.de.compare(..., metrics="all")`** to pull not only CLR-space contrasts but also biological log-fold-change (`bio_lfc`), log-variance ratio (`bio_lvr`), and Jeffreys divergence (`bio_kl`) summaries. The CLR view answers "how do compositions differ?"; the `bio_*` views answer "how do the underlying NB distributions differ?" — often complementary.
    - **Carry the full posterior downstream.** The main thing a generative Bayesian model buys you over a traditional pipeline is not just a point estimate but calibrated **uncertainty** on every gene, every contrast, every pathway enrichment. That becomes invaluable the moment you move from toy datasets like this one to noisier real-world atlases.
    """)
    return


if __name__ == "__main__":
    app.run()
