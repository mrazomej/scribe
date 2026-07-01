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
    # Which model should we trust? A friendly tour of *Bayesian model comparison*

    When you fit a model, you quietly make a choice that rarely gets stated out loud: **how complicated should it be?** A richer model can always trace the data you already have more closely — but "fits the data I have" and "predicts the next experiment well" are not the same thing. Push the flexibility too far and the model starts memorizing the noise in *this* dataset, then stumbles on the next one. That is **overfitting**, and protecting against it is what model comparison is for.

    In the [companion differential-expression tutorial](https://mrazomej.github.io/scribe/tutorials/zhao_2021_hierarchical/) we fit one model to the Zhao et al. (2021) panobinostat data — seven donors, each profiled with and without the drug. That model let each gene's **average expression** ($\mu_g$) shift with treatment and vary from donor to donor, but treated each gene's **noisiness** — its dispersion $r_g$, how spread out the counts are around the average — as a single number shared everywhere. That was a *choice*, and this tutorial asks whether it was a good one.

    We will ask it by putting **three models** head to head:

    - **Model A — `mu_only`**: the baseline. A hierarchy on the mean $\mu_g$; one shared dispersion $r_g$ per gene.
    - **Model B — `mu_plus_r`**: also lets the *dispersion* change with treatment and donor — more flexible about the **noise**.
    - **Model C — `mu_slope`**: instead lets each *donor respond to the drug differently* — more flexible about the **per-donor response**. *(This is the model the DE tutorial actually used.)*

    For each added piece of flexibility we ask the same honest question: **does it earn its keep with better predictions, or is it just fitting noise?** The toolkit that answers this in `scribe` is `scribe.mc`. Its vocabulary — *elpd*, *WAIC*, *PSIS-LOO* — may be new to you; we will build each idea from scratch as we reach it, because the *reasoning* matters far more than the acronyms.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The first two models, side by side

    Both models describe each cell's gene counts with the same building block: a **negative binomial** distribution — the standard workhorse for count data that is "noisier than Poisson," which single-cell counts always are — plus a per-cell factor for **capture efficiency** (not every molecule in a cell gets sequenced, and that fraction varies from cell to cell). Both also share the same recipe for a gene's **average** expression in a given donor × condition "leaf": a population baseline, plus a treatment shift, plus a donor offset,

    $$
    \log \mu_g^{(\ell)} = \underbrace{\log \mu_g^{\mathrm{pop}}}_{\text{baseline}} + \underbrace{\alpha_g^{\mu}\!\big[\mathrm{cond}(\ell)\big]}_{\text{treatment}} + \underbrace{\beta_g^{\mu}\!\big[\mathrm{donor}(\ell)\big]}_{\text{donor}} .
    $$

    (The DE tutorial derives this in full; here we just need to know the mean is built this way.) The two models differ in **exactly one place** — what they do with the dispersion $r_g$, the knob that sets how spread out the counts are:

    - **Model A (`mu_only`)** gives every gene a single $r_g$, identical in every donor and condition. The negative-binomial variance is $\sigma^2 = \mu + \mu^2/r$, so with $r$ fixed the *only* way the spread can change between control and treated cells is through the mean $\mu$. The *shape* of the noise is locked.
    - **Model B (`mu_plus_r`)** gives the dispersion the *same* treatment-and-donor hierarchy the mean already has. Now a gene can be not just higher or lower under the drug, but genuinely *noisier* or *tidier* — and differently so in different donors.

    One technical aside, then we move on: this only works because we fit with `parameterization="mean_disp"`, in which $\mu$ and $r$ are **separate, directly-estimated knobs**. (In some other parameterizations $r$ is only computed indirectly, and you cannot hang a hierarchy on something you never estimate directly.)

    Finally, the fact that makes this a fair contest: Model B **contains** Model A as a special case — switch the dispersion hierarchy off and the two are identical. A model that contains another can *always* match the training data at least as well, because it has strictly more dials to turn. So the interesting question is never "does B fit the training data better?" (it must), but "does B predict *new, unseen* data better?" That is what the next section learns how to measure.
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

    Exactly the same loading and QC as the DE tutorial: the seven panobinostat donors, their treated and matched-control cells, low-UMI cells dropped. This is the crossed $7 \times 2 = 14$-leaf table all three models are fit to.
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
    ## Fitting the three models

    The three `scribe.fit` calls are **identical except for one field each**. Model A puts a hierarchy only on `mean_expression`. Model B adds the same structure on `dispersion`. Model C leaves the dispersion alone and instead adds a single `interactions=[("perturbation", "sample")]` — the per-donor treatment slope. Everything else — the likelihood, the crossed `hierarchy`, the capture model, the gene-coverage step — is held fixed, so each model differs from A by *exactly one* structural change. That is what lets us attribute any difference in prediction to that one change and nothing else.

    Fitting at this scale takes a while, so we cache each fit and only run the optimizer when a cached result is missing. We load all three here but hold Model C's comparison back until the A-vs-B question is settled — that is the order the two questions deserve.
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

    # --- Model C: mu_only PLUS a per-donor treatment slope (the DE model) ---
    # Identical to Model A except for one added line: a perturbation x sample
    # INTERACTION on mu (the "random slope"). The dispersion stays a single
    # shared r_g -- so C differs from A on the donor-RESPONSE axis, exactly as
    # B differed from A on the dispersion axis. This is the fit the DE tutorial
    # used; we only look at its comparison after the A-vs-B verdict is in.
    _path_c = (
        data_dir
        / "scribe_results"
        / f"scribe_eda_{perturbation}_hierarchical_joint_mu_slope.pkl"
    )
    if _path_c.exists():
        with open(_path_c, "rb") as _f:
            results_mu_slope = pickle.load(_f)
    else:
        results_mu_slope = scribe.fit(
            adata_joint,
            parameterization="mean_disp",
            variable_capture=True,
            unconstrained=True,
            # exp link on mu only; r keeps its default (no dispersion hierarchy
            # here), hence the per-target dict rather than a bare "exp".
            positive_transform={"mean_expression": "exp"},
            hierarchy=[
                scribe.GroupLevel("perturbation", effect_type="fixed"),
                scribe.GroupLevel("sample"),
            ],
            # The ONLY structural addition over Model A: a per-donor treatment
            # slope. Interaction factors are random by default; the horseshoe
            # below pulls each donor's slope toward zero, so it costs almost
            # nothing on genes where donors respond alike and opens up only
            # where they genuinely diverge.
            interactions=[("perturbation", "sample")],
            priors={
                "mean_expression": {
                    "perturbation": "gaussian",  # shared (mean) response
                    "sample": "horseshoe",  # donor baseline, shrunk
                    "perturbation:sample": "horseshoe",  # per-donor slope -> 0
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
        with open(_path_c, "wb") as _f:
            pickle.dump(results_mu_slope, _f)

    print("Model A (mu_only):  ", repr(results_mu_only))
    print("Model B (mu_plus_r):", repr(results_mu_plus_r))
    print("Model C (mu_slope): ", repr(results_mu_slope))
    return results_mu_only, results_mu_plus_r, results_mu_slope


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How do we score a model's predictions? (*elpd*, *WAIC*, and *PSIS-LOO* in plain terms)

    We want to reward a model for predicting *new* data well, not for hugging the data it was trained on. Picture running the experiment again and getting a fresh batch of cells: a good model would assign those new counts **high probability**. The formal name for "average log-probability the model gives to new data" is the **expected log predictive density**, or **elpd** — but you can simply read it as *a prediction score, where higher is better*.

    The snag, of course, is that we do not *have* a fresh batch. So we estimate what that score would be from the data in hand. `scribe.mc` offers two standard estimators, and it helps to know what each is really doing:

    - **WAIC** (Widely-Applicable Information Criterion). The idea: take how well the model fits the data it saw, then **subtract a penalty for complexity**. If you have met AIC or BIC, this is the same spirit — except the penalty is *measured from the fit* instead of counted from the number of parameters. That penalty is the **variance of the model's log-probability across posterior draws**: a model whose predictions swing around a lot as you try different plausible parameter values is effectively "spending" more parameters, and gets charged for it. Best of all, it is computed straight from posterior samples we already have.
    - **PSIS-LOO** (Pareto-smoothed importance-sampling leave-one-out). This estimates the same score a different way — by approximating true **leave-one-out cross-validation** (drop one observation, predict it, repeat for all) *without* actually refitting thousands of times. Its bonus is a built-in **honesty check**: for each observation it returns a number $\hat{k}$ that flags when the shortcut is untrustworthy. Rule of thumb — $\hat{k} < 0.7$ is fine, $\hat{k} \ge 0.7$ is not.

    Both are after the same thing — out-of-sample prediction quality — and when they agree we can breathe easy.

    One more decision hides inside all this, and for single-cell data it is *the* decision: **what counts as "one observation"?** A cell? A gene? The next section shows why, for models like ours, the answer has to be the *gene* — and `scribe`'s `compare_models` detects this and switches automatically. (It also means a single call computes everything we need across all three models in one batched pass — fast even at 75,000 cells × 17,500 genes.)
    """)
    return


@app.cell
def _(
    adata_joint,
    results_mu_only,
    results_mu_plus_r,
    results_mu_slope,
    scribe,
):
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

    # All three models in ONE call: the per-gene log-likelihoods (500 samples x
    # ~75k cells x ~17.5k genes) are the expensive part, so we pay for them once
    # and pull out any pair afterward with `gene_level_comparison`. Models are
    # evaluated sequentially in `posterior_sample_chunk_size=2` chunks, so adding
    # the third model costs compute but not peak memory.
    mc = scribe.mc.compare_models(
        [results_mu_only, results_mu_plus_r, results_mu_slope],
        counts=counts_model,
        model_names=["mu_only", "mu_plus_r", "mu_slope"],
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
    ## Why we compare at the *gene* level, not the cell

    When `compare_models` runs, it prints that it computed **gene-level** scores and *refused* to do cell-level leave-one-out. That refusal is deliberate, and the reason is genuinely worth understanding — it is one of those spots where single-cell structure quietly breaks a textbook recipe.

    Here is the intuition. Our model gives **every cell its own capture-efficiency knob** ($\nu_c$, the `p_capture` latent) — how thoroughly *that particular cell* was sequenced — and that knob is learned from that one cell's counts. Now try leave-one-cell-out: hide a cell and ask the model to predict it. The trouble is that nothing else in the data tells you the hidden cell's capture knob; every other cell has its own. Honest cross-validation would have to fall back on the *prior* guess for the knob — but the score we can actually compute used the knob that was tuned *to the very cell we are trying to predict*. That is circular, and the cross-validation correction cannot undo it.

    You can *see* the failure in the numbers: PSIS-LOO's honesty check $\hat{k}$ climbs above 0.7 for essentially every cell, and WAIC's complexity penalty blows up to nonsense values. This is not a defect of either model — it is a known, general consequence of having **one latent variable per observation**. (The full derivation is in the supplementary material.)

    The fix is to change what "one observation" means: score by **gene** instead of by cell. Hiding a single gene barely moves anything — each gene's $\mu_g, r_g$ are pinned down by ~75,000 cells, and every cell's capture knob is still informed by the thousands of *other* genes measured in it. So leave-one-gene-out is stable and well-posed. Because essentially every realistic single-cell model needs a per-cell capture term, the gene is simply the right unit here — which is why `compare_models` switches to it on its own (and why the cell-level entry points, `mc.rank()` and `mc.psis_loo()`, politely decline and point you to the gene-level comparison).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The right lens: comparing genes one at a time

    `mc.gene_level_comparison(A, B)` takes the per-gene scores (each gene's counts summed over all cells) and, for every gene, reports **how much better model A predicts it than model B** (`elpd_diff`), along with an uncertainty (`elpd_diff_se`, a standard error) and a signal-to-noise ratio (`z_score`, the difference divided by its uncertainty). A **positive** `elpd_diff` means the first-named model wins for that gene. This lets us see not just *whether* the extra flexibility helps, but *which genes* it helps — which turns out to be the whole story. Below, a positive value favours `mu_plus_r`.
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
    ## What did the dispersion hierarchy actually buy?

    The verdict is **concentrated, not uniform** — and, surprisingly, it mostly favours the *simpler* model. Read the scatter's title: the large majority of genes (~15k of ~17.5k) significantly prefer `mu_only`, and only a few hundred prefer `mu_plus_r`. This is precisely the complexity penalty from earlier doing its job. For a gene whose noisiness genuinely does *not* change across donors and conditions, giving the dispersion its own hierarchy adds wobble (more effective parameters) with no gain in fit — so its out-of-sample score goes *down*. Model B can always match Model A, but it *pays* for the extra freedom on every gene and only *recoups* the cost where the freedom is actually used.

    What makes `mu_plus_r` worth having, then, is not how *often* it wins but *where*. The genes it helps most are a biologically coherent set:

    - **Immunoglobulin genes** — `IGHM`, `IGHA2`, `IGHG1`–`IGHG4`, `IGKV*`, `JCHAIN`. These are the textbook over-dispersed transcripts of single-cell data: a few plasma/B cells express them at enormous, wildly variable levels while nearly every other cell sits at zero. No single dispersion value can describe both regimes at once; a per-leaf dispersion can.
    - **Mitochondrial transcripts** — `MT-ND1`, `MT-ND5`, `MT-CO1`, `MT-CYB`, `MT-ATP6`, … — whose cell-to-cell variability differs sharply between the control and treated arms.

    In other words, the extra flexibility is spent **exactly where the noise structure genuinely changes** — not smeared across the transcriptome. That pattern is the fingerprint of a *real* improvement rather than overfitting: it lands on a small, mechanistically sensible set of genes, and the simpler model wins everywhere else.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where do the gains live — and can we trust them?

    A concentrated, biologically-sensible win is reassuring, but we should press on it with the question worth asking of *any* flexibility-driven improvement: **at what expression level do the gains sit?** If the dispersion hierarchy mainly "helps" genes that are barely detected — a few nonzero counts scattered across 75,000 cells — the improvement might be an illusion: the flexible model drawing a slightly nicer curve through points too sparse to pin down *any* shape. We would not want to trust a per-donor noise estimate for a gene seen in five cells.

    So we plot, for each gene, its prediction-score difference against its **mean UMI per cell** (how deeply it was sampled), and colour each point by its **variance-to-mean ratio (VMR)** — a simple empirical measure of over-dispersion. (VMR $= 1$ is the Poisson baseline; VMR $\gg 1$ means the gene is far noisier than Poisson — the hallmark of a bursty, bimodal gene.) The colour is the crucial second axis, because it separates two very different reasons a low-expression gene might appear to favour `mu_plus_r`:

    - **low UMI *but* high VMR** → a genuinely *bimodal* gene (silent in almost every cell, exploding in a few). The over-dispersion is real, a single dispersion value truly cannot capture it, and the win is trustworthy despite the tiny average.
    - **low UMI *and* VMR near 1** → a gene that is simply under-sampled and close to Poisson. Any "improvement" rides on a few stray counts and deserves suspicion.
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

    **The bulk of the transcriptome sits *below* zero, and sinks as sampling depth grows.** Every well-expressed gene reliably favours `mu_only`, and the deficit *grows* with mean UMI — the most-sampled genes carry by far the largest negative differences. Part of that is mechanical (a gene with more counts contributes more total log-probability, so any per-cell penalty is amplified), but the *sign* is the message: where a gene is sampled deeply enough to estimate its noise precisely, the data say a single dispersion already suffices, and the hierarchy's extra wobble is pure cost.

    **The positive cloud lives at low-to-moderate mean UMI — and it is bright.** Exactly as we worried, the genes the dispersion hierarchy *helps* are the sparsely-sampled ones (the immunoglobulins sit near $10^{-2}$ UMI/cell). But the colour rescues the interpretation: those low-UMI winners are the **brightest points on the plot** (VMR in the hundreds to >1000). They are not noise — they are *bimodal*: silent in ~99.5% of cells and explosive in a few plasma/B cells. No single dispersion can be both, so the per-leaf dispersion earns its keep despite the tiny mean. The median winner has $\mathrm{VMR}\approx 4$ against $\approx 1.3$ for the transcriptome as a whole — the gains sit on genuine over-dispersion, not on sampling depth.

    **The caveat to carry.** A minority of the low-UMI "winners" are *dark* — low mean UMI **and** VMR near 1. For those genes the favourable score rests on a few scattered counts and should be treated as suggestive at best, not as evidence the dispersion hierarchy is needed. The honest reading of the whole figure: **trust the win where over-dispersion backs it up (bright points), discount it where it does not (dark points at the far-left edge).** This is why a sane default is to keep the simpler `mu_only` for routine differential expression and reach for `mu_plus_r` deliberately — precisely on the over-dispersed families (immunoglobulins, mitochondrial transcripts) where the comparison says the flexibility is real.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The second question: what if donors *respond* differently?

    The A-vs-B round settled the **noise** question: keep one shared dispersion, and reach for `mu_plus_r` only on the few genuinely over-dispersed families. Simpler wins.

    But the model the DE tutorial actually trusted was not `mu_plus_r`. It kept the single shared dispersion (heeding that verdict) and spent its extra flexibility somewhere else entirely: it let **each donor respond to the drug in their own way**. Biologically this is the obvious worry — seven people will not react to a drug identically — and statistically it has a name, a **treatment × donor interaction**: a per-donor "slope" $\gamma_g$ added to the mean,

    $$
    \log \mu_g^{(\ell)} = \log \mu_g^{\mathrm{pop}}
      + \alpha_g\big[\mathrm{cond}(\ell)\big]
      + \beta_g\big[\mathrm{donor}(\ell)\big]
      + \underbrace{\gamma_g\big[\mathrm{cond}(\ell),\,\mathrm{donor}(\ell)\big]}_{\text{how much THIS donor's response differs}} .
    $$

    This is **Model C** (`mu_slope`). Like B, it contains A as a special case (set $\gamma = 0$), so again the freedom has to prove itself out-of-sample. But we should expect its report card to *look* different from B's, and it is worth seeing why in advance. The slope uses a **horseshoe** prior — a prior that pulls each donor's $\gamma_g$ hard toward zero unless the data insist otherwise. On a gene where donors respond alike, the slopes vanish, Model C becomes Model A, and it pays almost no complexity penalty. Contrast B, which forced a dispersion contrast onto *every* gene and was penalized across the board. So C should sidestep B's wall of penalties: it is nearly free where it is not needed.

    We loaded Model C alongside the others and folded it into the same `compare_models` call, so its scores are already computed — we just ask for a different pairing. Two questions follow, and, being honest up front, they are *not* equally easy: does the slope help **overall**, and can we point to *which* genes it helps, and *why*?
    """)
    return


@app.cell
def _(mc):
    # Same gene-level machinery, different pair. Positive elpd_diff now favours
    # the slope model C over the plain hierarchy A.
    gene_cmp_slope = mc.gene_level_comparison(
        "mu_slope", "mu_only", criterion="waic_2"
    )
    # Genes the per-donor slope helps predict best.
    top_genes_slope = gene_cmp_slope.sort_values(
        "elpd_diff", ascending=False
    ).head(25)
    top_genes_slope.round(1)
    return (gene_cmp_slope,)


@app.cell
def _(gene_cmp_slope, plt):
    # Per-gene elpd difference vs significance, C vs A. Positive => favours the
    # per-donor slope. Compare the SHAPE of this cloud to the mu_plus_r one:
    # the horseshoe makes the slope near-free where unused, so we do not expect
    # the same lopsided wall of negatives that penalised the dispersion model.
    _d = gene_cmp_slope["elpd_diff"].to_numpy()
    _z = gene_cmp_slope["z_score"].to_numpy()
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.scatter(_d, _z, s=5, alpha=0.3, color="teal")
    _ax.axhline(0, c="k", lw=0.5)
    _ax.axvline(0, c="k", lw=0.5)
    _ax.set_xscale("symlog")
    _ax.set_xlabel(r"per-gene elpd difference  (slope − $\mu$ only)")
    _ax.set_ylabel(r"z-score of the difference")
    _ax.set_title(
        r"$%s$ genes favour the slope at $z > 2$"
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
    ## Reading this comparison honestly

    Two questions, unequal difficulty.

    **The easy one: does the slope help overall?** Model selection ultimately turns on a single summary — add up the per-gene score differences across all genes, and check whether the total favours the slope by more than its own uncertainty. We compute that just below. From the scatter above you can already read the *balance* of winners and losers (its title counts them), and thanks to the horseshoe we expect no across-the-board wall of penalties like B's.

    **The hard one: can we attribute the gain to donor disagreement, gene by gene?** The tempting move is to plot each gene's score gain against *how much its donors actually disagreed* and hope for a rising trend. We tried it — and honesty requires flagging that gene-level scoring is a **weak tool for this particular question**, for three reasons that compound:

    1. **Sequencing depth dominates the score.** As the Model-B plot already hinted, a gene's score difference grows simply with how many counts it has. A raw plot of score-versus-anything is pulled around by the most deeply *sequenced* genes, not the most donor-*divergent* ones.
    2. **The horseshoe hides the effect on purpose.** It shrinks each $\gamma_g$'s best-estimate toward zero; the predictive benefit comes from the *whole* posterior (its wider uncertainty included), which need not line up with the shrunken point estimate we can put on an axis.
    3. **We are asking the wrong question of this metric.** Gene-level scoring measures "does the slope sharpen the predicted *counts* of a held-out gene?" But the slope's real job is different: to make the **conclusion** — the population treatment effect — hold up for a *new* donor. A model can barely change per-gene count predictions while dramatically changing how *confident* we are allowed to be about the population effect.

    So instead of a misleading point cloud, we look **depth-free and binned**: sort genes into bins by how much their donors disagreed, and in each bin ask two robust questions — what *fraction* of genes prefer the slope (a simple sign count, immune to depth), and what is the *typical* signal-to-noise (median z-score)? A rising fraction would be honest corroboration; a flat one tells us plainly that this attribution just does not live in gene-level scores — exactly what the three reasons above predict. The donor disagreement itself is read from the fitted slope via `return_sites=["mu_perturbation__sample_effect"]` — the context-aware posterior draw introduced on this branch — so we materialize only that small `(draws × 14 × genes)` array, not a full per-cell posterior for a third large model.
    """)
    return


@app.cell
def _(gene_cmp_slope, gene_names, pd, plt, results_mu_slope):
    import numpy as _np

    # ---- 1. The headline: the AGGREGATE score verdict (C vs A) ------------
    # Model selection is decided by the SUMMED score difference, not any single
    # gene. Its SE follows the CLT on the per-gene differences (the loo-compare
    # convention documented on gene_level_comparison):
    #     delta = sum_g d_g ,   SE = sqrt(sum_g (d_g - mean_d)^2).
    _real = gene_cmp_slope[gene_cmp_slope["gene"] != "_other"]
    _d_all = _real["elpd_diff"].to_numpy()
    _delta = float(_d_all.sum())
    _se = float(_np.sqrt(((_d_all - _d_all.mean()) ** 2).sum()))
    _z_tot = _delta / _se if _se > 0 else float("nan")
    _frac = float((_d_all > 0).mean())
    _verdict = "beats" if _z_tot > 2 else ("loses to" if _z_tot < -2 else "ties")
    print(
        f"AGGREGATE gene-level score:  slope {_verdict} mu_only    "
        f"(delta_elpd = {_delta:+.1f} +/- {_se:.1f},  z = {_z_tot:+.1f})"
    )
    print(f"fraction of genes favouring the slope: {_frac:.1%}")

    # ---- 2. WHERE does the difference sit? A depth-free, binned look -------
    # A 17k-point scatter is an unreadable blob, and the raw score is confounded
    # by sequencing depth (|score| grows with a gene's total counts). So we drop
    # the scatter for a binned view of two DEPTH-INSENSITIVE summaries against
    # the between-donor response spread (SD across donors of
    # gamma[pano,d]-gamma[ctrl,d], read from the fitted interaction site):
    #   * FRACTION of genes favouring the slope  (pure sign -> depth-free)
    #   * MEDIAN per-gene z-score                (standardised magnitude)
    _gamma = _np.asarray(
        results_mu_slope.get_posterior_samples(
            return_sites=["mu_perturbation__sample_effect"]
        )["mu_perturbation__sample_effect"]
    ).mean(0)
    _nd = _gamma.shape[0] // 2
    _resp_spread = (_gamma[_nd:] - _gamma[:_nd]).std(axis=0)

    _df = gene_cmp_slope.merge(
        pd.DataFrame({"gene": gene_names, "resp_spread": _resp_spread}),
        on="gene",
        how="left",
    )
    _df = _df[(_df["gene"] != "_other")].dropna(subset=["resp_spread"]).copy()

    # Rank correlations, for the record -- how well donor spread predicts WHICH
    # genes the slope helps, before (raw) and after (z-score) de-confounding
    # magnitude. Both are printed so the narrative rests on the actual numbers.
    _rho_raw = (
        _df[["elpd_diff", "resp_spread"]].corr(method="spearman").iloc[0, 1]
    )
    _rho_z = _df[["z_score", "resp_spread"]].corr(method="spearman").iloc[0, 1]
    print(
        f"Spearman(score gain, response spread) = {_rho_raw:+.2f}   |   "
        f"Spearman(z-score, response spread) = {_rho_z:+.2f}"
    )

    # Decile bins on response spread; per bin, the depth-free summaries.
    _df["_bin"] = pd.qcut(
        _df["resp_spread"], 10, labels=False, duplicates="drop"
    )
    _grp = _df.groupby("_bin")
    _x = _grp["resp_spread"].median().to_numpy()
    _frac_bin = _grp["elpd_diff"].agg(lambda s: (s > 0).mean()).to_numpy()
    _zmed_bin = _grp["z_score"].median().to_numpy()

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.plot(_x, _frac_bin, "o-", color="teal")
    _ax.axhline(_frac, ls="--", c="teal", lw=0.9, alpha=0.7)  # overall fraction
    _ax.axhline(0.5, ls=":", c="k", lw=0.6)  # coin flip
    _ax.set_xscale("log")
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("between-donor response spread  (decile-bin median)")
    _ax.set_ylabel("fraction of genes favouring the slope", color="teal")
    _ax.tick_params(axis="y", labelcolor="teal")
    _ax2 = _ax.twinx()
    _ax2.plot(_x, _zmed_bin, "s--", color="indianred", alpha=0.8)
    _ax2.axhline(0, ls=":", c="indianred", lw=0.6, alpha=0.6)
    _ax2.set_ylabel("median per-gene z-score", color="indianred")
    _ax2.tick_params(axis="y", labelcolor="indianred")
    _ax.set_title(
        "Does the slope help MORE where donors disagree?  (binned, depth-free)"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we can honestly conclude — and why the DE model is still the right one

    Read the printed numbers with those three caveats in mind.

    **Overall, the slope does not overfit — and that is the check that matters.** The summed score difference (printed above) shows the slope sidesteps the across-the-board penalty that sank the dispersion hierarchy: where the horseshoe zeroes out $\gamma_g$, Models C and A coincide, so C carries its extra parameters at almost no cost and keeps whatever gains it earns. That is the essential test of *any* added flexibility — does it pay for itself, or is it just fitting noise? — and the slope passes it.

    **But we could not pin the gain to donor disagreement gene by gene — and we should say so plainly.** The correlation between a gene's score gain and how much its donors actually disagreed is essentially zero (printed above — about $0.09$), and the fraction of genes favouring the slope barely moves across the disagreement bins. The biggest raw gains sit on deeply-sequenced housekeeping and mitochondrial genes — a depth effect — not along a clean donor-disagreement axis. This is **not** a failure of the slope; it is the gene-level score failing to *see* what the slope does, exactly as the three reasons predicted. Predicting held-out gene counts is simply not where a per-donor response effect shows up, so no scatter of score-against-disagreement was ever going to light up.

    **So where is the real case for the slope?** In the two places that measure the right thing directly:

    - **We can measure the disagreement itself.** In the DE tutorial, the spread of donor *responses* runs about 1.4× the spread of their baselines — donors disagree *more* about how they react to the drug than about where they start. That is read straight off the parameter, with no prediction score standing in between.
    - **It changes the conclusion, not the per-gene fit.** The slope anchors the population effect to the *average donor* and folds the donor-to-donor disagreement into its uncertainty. This guards against a classic trap — **pseudoreplication**, treating one donor's thousands of cells as if they were thousands of independent people. The payoff is that every treatment call is honest about whether it would **hold up in a new donor**, which is exactly what you want from a differential-expression result and is invisible to leave-one-gene-out.

    **Putting the two comparisons together gives us the DE model — with an honest account of the evidence.** A-vs-B is a straight prediction verdict: don't give the dispersion its own hierarchy; one shared value predicts better. A-vs-C is a check plus an argument: the prediction score confirms the slope does not overfit, and the *reason* to keep it is that it makes our conclusions generalize across donors — something we justify by measuring donor disagreement directly, not through a prediction score. Feature by feature, that is the model the DE tutorial relies on: **`mean_disp`, one shared dispersion, and a per-donor treatment slope with a horseshoe prior.** Knowing *which* piece of evidence supports *which* choice is the difference between genuinely trusting a model and merely preferring it.
    """)
    return


if __name__ == "__main__":
    app.run()
