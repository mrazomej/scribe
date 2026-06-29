import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Which model should we trust? *Bayesian model comparison* for a crossed hierarchical fit

    In the [companion differential-expression tutorial](https://mrazomej.github.io/scribe/tutorials/zhao_2021_hierarchical/) we fit **one** crossed *donor × condition* model to the Zhao et al. (2021) panobinostat data and read the treatment effect off its posterior. But that tutorial quietly made a **modeling choice**: it put the additive hierarchy only on the gene **mean** $\mu_g$, leaving the gene **dispersion** $r_g$ as a single per-gene value shared across all leaves.

    That is not the only option. Because `scribe`'s `mean_disp` parameterization samples $(\mu_g, r_g)$ as *independent* coordinates, we can just as well give **both** of them the donor × condition hierarchy — letting each gene's *over-dispersion* shift with treatment and vary across donors, not just its mean. So we now have **two competing models** of the same data:

    - **Model A — `mu_only`**: hierarchy on $\mu$; one shared $r_g$ per gene.
    - **Model B — `mu_plus_r`**: hierarchy on **both** $\mu$ and $r$.

    Model B is strictly more flexible. The question this tutorial answers is the one you should *always* ask before trusting a more complex model: **does that extra flexibility actually buy better predictions, or is it just fitting noise?** We answer it with `scribe.mc` — the Bayesian model-comparison toolkit (WAIC, PSIS-LOO, gene-level comparison) — and in doing so we learn *where* the two models genuinely differ, which turns out to connect directly back to the differential-expression result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The two models, written out

    Both models share the same Negative-Binomial-with-variable-capture likelihood and the same additive *crossed* hierarchy on the log-mean (population baseline + fixed treatment effect + random donor effect; see the DE tutorial for the full derivation):

    $$
    \log \mu_g^{(\ell)} = \log \mu_g^{\mathrm{pop}} + \alpha_g^{\mu}\!\big[\mathrm{cond}(\ell)\big] + \beta_g^{\mu}\!\big[\mathrm{donor}(\ell)\big].
    $$

    They differ only in how the **dispersion** $r_g$ is treated:

    - **Model A (`mu_only`).** A single $r_g$ per gene, shared by every leaf. The NB variance $\sigma^2 = \mu + \mu^2/r$ can only change between conditions through $\mu$ — the *shape* of the count distribution is locked.

    - **Model B (`mu_plus_r`).** The *same* additive crossed hierarchy, now also on $\log r_g$:
      $$
      \log r_g^{(\ell)} = \log r_g^{\mathrm{pop}} + \alpha_g^{r}\!\big[\mathrm{cond}(\ell)\big] + \beta_g^{r}\!\big[\mathrm{donor}(\ell)\big].
      $$
      Now a gene's over-dispersion is free to shift with treatment ($\alpha_g^{r}$) and vary across donors ($\beta_g^{r}$).

    Why is Model B even *possible*? Because under `parameterization="mean_disp"` the pair $(\mu, r)$ is the **Fisher-orthogonal** coordinate system of the Negative Binomial — $r$ is a directly-sampled parameter, so it can carry its own hierarchy. (Under `mean_odds`, $r$ is a *derived* quantity and a hierarchy on it is not expressible. This is the practical pay-off of the orthogonal parameterization, beyond the DE-time stability discussed in the other tutorial.)

    Model B nests Model A (set $\alpha^r = \beta^r = 0$), so it can never fit the *training* data worse. The whole point of model comparison is to penalize that extra freedom by its effect on **out-of-sample** predictive accuracy.
    """)
    return


@app.cell
def _():
    # Import basic packages
    from pathlib import Path
    import json
    import pickle

    # Import our main package
    import scribe

    # Import data tooling
    import pertpy
    import scanpy as sc

    # Numerics and plotting
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set our plotting style (totally optional)
    scribe.viz.matplotlib_style()
    return Path, json, pd, pertpy, pickle, plt, sc, scribe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the data

    Exactly the same loading and QC as the DE tutorial: the seven panobinostat donors, their treated and matched-control cells, low-UMI cells dropped. This is the crossed $7 \times 2 = 14$-leaf table both models are fit to.
    """)
    return


@app.cell
def _(pertpy, sc):
    adata = pertpy.data.zhao_2021()
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1_000)
    print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes after QC")
    return (adata,)


@app.cell
def _(adata, pd):
    # Control arm as dose 0 (keep the categorical dtype happy).
    adata.obs["dose_value"] = adata.obs["dose_value"].cat.add_categories(["0"])
    adata.obs.loc[adata.obs["perturbation"] == "control", "dose_value"] = "0"

    _perturbation = "panobinostat"
    _samples = adata.obs.loc[
        adata.obs["perturbation"] == _perturbation, "sample"
    ].unique()
    _mask = adata.obs["sample"].isin(_samples) & adata.obs["perturbation"].isin(
        ["control", _perturbation]
    )
    adata_joint = adata[_mask].copy()
    perturbation = _perturbation

    crosstab = pd.crosstab(
        adata_joint.obs["sample"], adata_joint.obs["perturbation"]
    )
    crosstab
    return adata_joint, perturbation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting the two models

    The two `scribe.fit` calls are **identical except for the `priors` dict**.  Model A declares a hierarchy only on `mean_expression`; Model B adds the same structure on `dispersion`. Everything else — the likelihood, the crossed `hierarchy`, the variable capture, the gene-coverage closure — is held fixed, so any difference in predictive accuracy is attributable to the dispersion hierarchy alone.

    Fitting both at this scale takes a while, so we cache each fit and only run the optimizer when a cached result is absent.
    """)
    return


@app.cell
def _(Path, adata_joint, json, perturbation, pickle, scribe):
    # Resolve the dataset directory from the local tutorial path config.
    with (
        Path(__file__)
        .with_name("tutorial_paths.local.json")
        .open("r", encoding="utf-8") as _f
    ):
        _data_root = json.load(_f)["SCRIBE_TUTORIAL_DATA_ROOT"]
    data_dir = Path(_data_root).expanduser() / "zhao_2021"
    (data_dir / "scribe_results").mkdir(parents=True, exist_ok=True)

    # --- Model A: hierarchy on mu only (the model fit in the DE tutorial) ---
    # Same cached fit the DE tutorial produced; load it if present.
    _path_a = (
        data_dir
        / "scribe_results"
        / f"scribe_hierarchical_mean_disp_{perturbation}_joint.pkl"
    )
    if _path_a.exists():
        with open(_path_a, "rb") as _f:
            results_mu_only = pickle.load(_f)
    else:
        results_mu_only = scribe.fit(
            adata_joint,
            parameterization="mean_disp",
            variable_capture=True,
            unconstrained=True,
            positive_transform="exp",
            hierarchy=[
                scribe.GroupLevel("perturbation", effect_type="fixed"),
                scribe.GroupLevel("sample"),
            ],
            # Hierarchy on the EXPRESSION target only -> single shared r_g.
            # (Model B adds a matching "dispersion" entry; that is the only
            #  difference between the two priors dicts.)
            priors={
                "mean_expression": {
                    "perturbation": "gaussian",  # fixed-scale contrast
                    "sample": "horseshoe",  # shrink across donors
                },
            },
            early_stopping={
                "enabled": True,
                "patience": 1000,
                "restore_best": True,
            },
            n_steps=100_000,
            batch_size=4096,
            gene_coverage=0.99,
        )
        with open(_path_a, "wb") as _f:
            pickle.dump(results_mu_only, _f)

    # --- Model B: the SAME crossed hierarchy on BOTH mu and r ---
    _path_b = (
        data_dir
        / "scribe_results"
        / f"scribe_eda_{perturbation}_hierarchical_joint.pkl"
    )
    if _path_b.exists():
        with open(_path_b, "rb") as _f:
            results_mu_plus_r = pickle.load(_f)
    else:
        results_mu_plus_r = scribe.fit(
            adata_joint,
            parameterization="mean_disp",
            variable_capture=True,
            unconstrained=True,
            # exp link on mu; the matching exp link on r is auto-forced
            # whenever a dispersion hierarchy is active, so the per-factor
            # effects are Delta-log-r, comparable to mu's log-fold-changes.
            positive_transform={"mean_expression": "exp"},
            hierarchy=[
                scribe.GroupLevel("perturbation", effect_type="fixed"),
                scribe.GroupLevel("sample"),
            ],
            # The SAME two-level (perturbation x sample) hierarchy on BOTH
            # mu and r -- the only difference from Model A.
            priors={
                "mean_expression": {
                    "perturbation": "gaussian",
                    "sample": "horseshoe",
                },
                "dispersion": {
                    "perturbation": "gaussian",
                    "sample": "horseshoe",
                },
            },
            early_stopping={
                "enabled": True,
                "patience": 1000,
                "restore_best": True,
            },
            n_steps=100_000,
            batch_size=4096,
            gene_coverage=0.99,
        )
        with open(_path_b, "wb") as _f:
            pickle.dump(results_mu_plus_r, _f)

    print("Model A (mu_only):  ", repr(results_mu_only))
    print("Model B (mu_plus_r):", repr(results_mu_plus_r))
    return results_mu_only, results_mu_plus_r


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The comparison framework, briefly

    We compare models by **expected log predictive density (elpd)** — how well a fitted model predicts *new* data drawn from the same process. Since we cannot sample new data, `scribe.mc` estimates the leave-one-out elpd two ways (full derivations in the supplementary material):

    - **WAIC** — an analytical approximation: $\widehat{\mathrm{elpd}} = \mathrm{lppd} - p_{\mathrm{WAIC}}$, where the in-sample log pointwise predictive density `lppd` is penalized by $p_{\mathrm{WAIC}}$, the posterior **variance** of the per-observation log-likelihood (the "effective number of parameters"). Fully vectorized, computed from the posterior draws we already have.
    - **PSIS-LOO** — Pareto-smoothed importance-sampling LOO: more reliable, and it carries a **per-observation diagnostic $\hat{k}$** (the fitted Pareto-tail shape) that tells us *when the estimate itself is trustworthy* — $\hat{k} < 0.7$ good, $\hat{k} \ge 0.7$ unreliable.

    The subtle question — *at what unit do we score predictions?* — is the subject of the next section. For these variable-capture models the answer is the **gene**, and `compare_models` detects that automatically: one call computes the per-gene log-likelihoods for both models and reports cell-level leave-one-out as unsupported. (The likelihood of a crossed-hierarchical fit gathers each cell's leaf parameters internally and `compare_models` evaluates it as a single batched pass, so this is fast even at 75k cells × 17.5k genes.)
    """)
    return


@app.cell
def _(adata_joint, results_mu_only, results_mu_plus_r, scribe):
    import jax

    # Build the model's gene axis (kept genes + pooled `_other`) from the stored
    # coverage mask, then compare. For variable-capture models compare_models
    # detects the cell-specific `p_capture` latent and computes only the
    # gene-level log-likelihoods (cell-level leave-one-out is ill-posed; see the
    # next cell). `compute_gene_liks=True` is passed explicitly for clarity but
    # is auto-enabled in that case.
    from scribe.core.gene_coverage import build_filtered_gene_names
    import numpy as _np
    import scipy.sparse as _sp

    # Reconstruct the matrix the model was fit on: the genes kept by the
    # gene_coverage=0.99 filter, plus the pooled `_other` column that closes
    # the composition. The stored mask records which genes were kept; we align
    # cells to the fit-time order (results.obs) and pool on CPU (a column sum,
    # cheap) so this prep never competes with the GPU for memory.
    _mask = _np.asarray(results_mu_only.gene_coverage_mask)
    _X = adata_joint[results_mu_only.obs.index].X
    _X = _X.toarray() if _sp.issparse(_X) else _np.asarray(_X)
    _X = _np.asarray(_X, dtype=_np.float32)
    counts_model = _np.concatenate(
        [_X[:, _mask], _X[:, ~_mask].sum(axis=1, keepdims=True)], axis=1
    )
    gene_names = build_filtered_gene_names(list(adata_joint.var_names), _mask)[0]

    mc = scribe.mc.compare_models(
        [results_mu_only, results_mu_plus_r],
        counts=counts_model,
        model_names=["mu_only", "mu_plus_r"],
        gene_names=gene_names,
        n_samples=500,
        posterior_sample_chunk_size=2,
        compute_gene_liks=True,
        ignore_nans=True,
        rng_key=jax.random.PRNGKey(0),
    )
    mc
    # Export `counts_model`/`gene_names` too: a later cell reuses the exact
    # fit-time count matrix (already aligned to the model's gene axis) to
    # compute per-gene mean UMI and over-dispersion without rebuilding it.
    return counts_model, gene_names, mc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why the comparison is at the *gene* level, not the cell

    Notice what `compare_models` printed: it computed **gene-level** log-likelihoods and declared cell-level leave-one-out *unsupported*. That is deliberate, and it is the subtle part of comparing single-cell models — worth understanding before reading the result.

    The instinctive thing is to score each model by leave-one-**cell**-out (cell-level WAIC / PSIS-LOO). But this model gives every cell its **own** capture probability $\nu_c$ (`p_capture`) — a latent informed by *that cell alone*. An honest held-out-cell prediction must integrate $\nu_c$ over its **prior**, because the other cells say nothing about it:

    $$
    p(\tilde u_c \mid U_{-c}) =
    \iint p(\tilde u_c \mid \theta, \nu)\, p(\nu)\,
    p(\theta \mid U_{-c})\, d\nu\, d\theta .
    $$

    The pointwise log-likelihood, however, conditions on the $\nu_c$ that was fit *to $u_c$ itself*. To turn that into a leave-one-out estimate, PSIS-LOO would have to reweight a **posterior-to-prior collapse** of $\nu_c$ — which it cannot — so the Pareto diagnostic gives $\hat{k} \ge 0.7$ for *essentially every cell*, and the WAIC variance penalty explodes ($p_{\mathrm{eff}}$ in the tens of millions). This is the textbook failure of leave-one-out under **per-observation latent variables**; it is a property of the design, not a defect of either model. (Full derivation in supplementary materials *Leave-one-cell-out with cell-specific latents* section.)

    The resolution is to change the **unit of observation** to the gene.  Dropping one gene barely moves any parameter — the gene-level $\mu_g, r_g$ are pinned by ~75k cells, and every $\nu_c$ is still informed by the thousands of *retained* genes in its cell — so leave-one-gene-out is well-posed and stable. Because **every realistic single-cell model needs variable capture**, the gene is the right unit for `scribe` model comparison, and `compare_models` switches to it automatically whenever a cell-specific latent is present (hence the message above, and why `mc.rank()` / `mc.psis_loo()` deliberately refuse here and point you to the gene-level comparison).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The right lens: gene-level comparison

    `mc.gene_level_comparison(A, B)` works with the **per-gene** log-likelihoods (each gene summed over all cells). For every gene it reports the elpd difference between the two models, its standard error, and a z-score — so we can see exactly **which genes** the dispersion hierarchy helps, and by how much. A positive `elpd_diff` favours `mu_plus_r`.
    """)
    return


@app.cell
def _(mc):
    gene_cmp = mc.gene_level_comparison(
        "mu_plus_r", "mu_only", criterion="waic_2"
    )
    # Most improved genes under the dispersion hierarchy.
    top_genes = gene_cmp.sort_values("elpd_diff", ascending=False).head(25)
    top_genes.round(1)
    return (gene_cmp,)


@app.cell
def _(gene_cmp, plt):
    # Per-gene elpd difference vs significance. Positive => favours $\mu + r$.
    _d = gene_cmp["elpd_diff"].to_numpy()
    _z = gene_cmp["z_score"].to_numpy()
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.scatter(_d, _z, s=5, alpha=0.3)
    _ax.axhline(0, c="k", lw=0.5)
    _ax.axvline(0, c="k", lw=0.5)
    _ax.set_xscale("symlog")
    _ax.set_xlabel(r"per-gene elpd difference  ($\mu + r$ − $\mu$ only)")
    _ax.set_ylabel(r"z-score of the difference")
    _ax.set_title(
        r"$%s$ genes favour $\mu{+}r$ at $z > 2$"
        "  vs  "
        r"$%s$ favouring $\mu$ only" % (
            int((_z > 2).sum()),
            int((_z < -2).sum()))
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does the dispersion hierarchy actually buy?

    The verdict is **concentrated, not uniform** — and at the level of raw gene counts it is lopsided the *opposite* way from what "more flexible model" might suggest. The large majority of genes (look at the scatter title: ~15k of ~17.5k) significantly favour the **simpler** `mu_only` at $z < -2$, while only a few hundred favour `mu_plus_r`. This is WAIC doing its job: for a gene whose over-dispersion does *not* vary across leaves, the dispersion hierarchy only adds posterior **variance** (a larger $p_{\mathrm{WAIC}}$ penalty) without improving the fit, so leave-one-out predictive accuracy goes *down*. Model B nests Model A, but the nesting is paid for everywhere and recouped only where it is used.

    What makes `mu_plus_r` worth having, then, is not breadth but the **identity and magnitude** of its wins. The top of the elpd-gain list is dominated by:

    - **Immunoglobulin genes** — `IGHM`, `IGHA2`, `IGHG1`–`IGHG4`, `IGKV*`, `JCHAIN`. These are the textbook *over-dispersed* transcripts of single-cell data: a few plasma/B cells express them at enormous, wildly variable levels while most cells sit near zero. A single shared $r_g$ cannot describe both regimes; a per-leaf dispersion can.
    - **Mitochondrially-encoded transcripts** — `MT-ND1`, `MT-ND5`, `MT-CO1`, `MT-CYB`, `MT-ATP6`, …, whose cell-to-cell variability differs sharply between the control and panobinostat arms.

    In other words, **the extra flexibility is spent exactly where over-dispersion genuinely varies across conditions and donors** — not smeared across the transcriptome. That is the signature of a model improvement that is *real* rather than overfit: it concentrates on a biologically coherent, mechanistically expected set of genes, and the simpler model is preferred everywhere else.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where in expression space does the gain live — and should we trust it?

    A concentrated, biologically-coherent win is reassuring, but it raises a sharper question we should always ask of a flexibility-driven improvement: **at what expression level do the gains sit?** If the dispersion hierarchy mostly "helps" genes that are barely sampled — a handful of nonzero counts spread across 75k cells — then the apparent improvement may just be the flexible model drawing a slightly better curve through points that are too noisy to pin down *any* shape. We would not want to trust a per-leaf over-dispersion estimate for a gene that fires in five cells.

    So we plot the per-gene elpd difference against the gene's **mean UMI per cell** (its raw sampling depth), and colour each point by its **variance-to-mean ratio** — the empirical over-dispersion index ($\mathrm{VMR}=1$ is Poisson; $\mathrm{VMR}\gg 1$ is strongly over-dispersed). The trend need not be linear; we just want to *see* it. The colour is the crucial second axis: it separates two very different reasons a low-UMI gene might appear to favour `mu_plus_r` —

    - **low mean UMI *with* high VMR** → a genuinely *bimodal* gene (zero in almost every cell, enormous in a few): the over-dispersion is real and a single $r_g$ truly cannot fit it, so the win is trustworthy even though the mean is tiny;
    - **low mean UMI *with* VMR near 1** → a gene that is simply under-sampled and near-Poisson: any "improvement" here rides on a few counts and should be treated with suspicion.
    """)
    return


@app.cell
def _(counts_model, gene_cmp, gene_names, pd, plt):
    from matplotlib.colors import LogNorm
    import numpy as _np

    # Per-gene empirical statistics, computed once from the exact fit-time count
    # matrix so they line up with the model's gene axis (kept genes + `_other`).
    #   mean_umi : mean counts per cell  -> sampling depth / sparsity
    #   vmr      : variance-to-mean ratio -> empirical over-dispersion index
    #              (1 = Poisson; >> 1 = strongly over-dispersed / bimodal)
    _mean_umi = counts_model.mean(axis=0)
    _var_umi = counts_model.var(axis=0)
    _vmr = _np.where(
        _mean_umi > 0, _var_umi / _np.maximum(_mean_umi, 1e-12), 1.0
    )
    gene_stats = pd.DataFrame(
        {"gene": gene_names, "mean_umi": _mean_umi, "vmr": _vmr}
    )

    # Join the empirical stats onto the per-gene comparison and drop the pooled
    # `_other` column (an aggregate, not a real gene).
    _df = gene_cmp.merge(gene_stats, on="gene", how="left")
    _df = _df[_df["gene"] != "_other"].copy()

    # Scatter: x = mean UMI (log), y = elpd difference (symlog, since it spans
    # +1.5e4 down to ~-6e5), colour = over-dispersion (log). Draw the most
    # over-dispersed genes last so the trustworthy winners sit on top.
    _order = _df["vmr"].to_numpy().argsort()
    _d = _df.iloc[_order]
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _scat = _ax.scatter(
        _d["mean_umi"].clip(lower=1e-3),
        _d["elpd_diff"],
        c=_d["vmr"].clip(lower=1.0),
        s=7,
        alpha=0.6,
        norm=LogNorm(vmin=1.0, vmax=1e3),
        cmap="viridis",
    )
    _ax.set_xscale("log")
    _ax.set_yscale("symlog", linthresh=10)
    _ax.axhline(0, c="k", lw=0.6)
    _ax.set_xlabel(r"mean UMI per gene (counts / cell)")
    _ax.set_ylabel(r"per-gene elpd difference  ($\mu + r$ − $\mu$ only)")
    _cb = _fig.colorbar(_scat)
    _cb.set_label("variance-to-mean ratio (over-dispersion)")
    # Label a few archetypes: ultra-sparse over-dispersed IGs vs moderate-UMI MT.
    for _g in ["IGHM", "IGHG2", "MT-ND1", "MT-CO1", "VGF"]:
        _r = _df[_df["gene"] == _g]
        if len(_r):
            _ax.annotate(
                _g,
                (_r["mean_umi"].clip(lower=1e-3).iloc[0], _r["elpd_diff"].iloc[0]),
                fontsize=7,
                ha="left",
            )
    _ax.set_title("elpd gain concentrates at low mean UMI — but tracks over-dispersion")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the trend

    Two things jump out, and together they answer the question.

    **The bulk of the transcriptome sits *below* zero, and sinks as sampling depth grows.** Every well-expressed gene reliably favours `mu_only`, and the deficit *grows* with mean UMI — the most-sampled genes (top decile) carry by far the largest negative elpd differences. That is partly mechanical (a gene with more counts contributes more total log-likelihood, so any per-cell penalty is amplified), but the *sign* is the message: where a gene is sampled deeply enough to estimate its dispersion precisely, the data say a single $r_g$ already suffices, and the hierarchy's extra variance is pure cost.

    **The positive cloud lives at low-to-moderate mean UMI — and it is bright.** Exactly as you might worry, the genes the dispersion hierarchy *helps* are the sparsely-sampled ones (the immunoglobulins sit near $10^{-2}$ UMI/cell). But the colour rescues the interpretation: those low-UMI winners are the **brightest points on the plot** (VMR in the hundreds to >1000). They are not noise — they are *bimodal*: silent in ~99.5% of cells and explosive in a few plasma/B cells. No single $r_g$ can be both, so the per-leaf dispersion earns its keep despite the tiny mean. The median winner has $\mathrm{VMR}\approx 4$ against $\approx 1.3$ for the transcriptome as a whole — the gains are concentrated on genuine over-dispersion, not on sampling depth.

    **The caveat you should still carry.** A minority of the low-UMI "winners" are *dark* — low mean UMI **and** VMR near 1. For those genes the favourable elpd rests on a few scattered counts and should be treated as suggestive at best, not as evidence the dispersion hierarchy is needed. The honest reading of the whole figure is therefore: **trust the win where it is backed by over-dispersion you can see (bright points), discount it where it is not (dark points at the far-left edge).** This is also why a sane default is to keep the simpler `mu_only` for routine differential-expression and reach for `mu_plus_r` deliberately — precisely on the over-dispersed gene families (immunoglobulins, mitochondrial transcripts) where the comparison says the flexibility is real.
    """)
    return


if __name__ == "__main__":
    app.run()
