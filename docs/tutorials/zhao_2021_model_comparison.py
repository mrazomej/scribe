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

    In the companion differential-expression tutorial we fit **one** crossed *donor × condition* model to the Zhao et al. (2021) panobinostat data and read the treatment effect off its posterior. But that tutorial quietly made a **modeling choice**: it put the additive hierarchy only on the gene **mean** $\mu_g$, leaving the gene **dispersion** $r_g$ as a single per-gene value shared across all leaves.

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
    return Path, json, np, pd, pertpy, pickle, plt, sc, scribe


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
            expression_dataset_prior={
                "perturbation": "gaussian",  # fixed-scale contrast
                "sample": "horseshoe",  # shrink across donors
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

    The subtle question — *at what unit do we score predictions?* — is the
    subject of the next section. For these variable-capture models the answer is
    the **gene**, and `compare_models` detects that automatically: one call
    computes the per-gene log-likelihoods for both models and reports cell-level
    leave-one-out as unsupported. (The likelihood of a crossed-hierarchical fit
    gathers each cell's leaf parameters internally and `compare_models` evaluates
    it as a single batched pass, so this is fast even at 75k cells × 17.5k genes.)
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
        n_samples=200,
        posterior_sample_chunk_size=2,
        compute_gene_liks=True,
        ignore_nans=True,
        rng_key=jax.random.PRNGKey(0),
    )
    mc
    return (mc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why the comparison is at the *gene* level, not the cell

    Notice what `compare_models` printed: it computed **gene-level**
    log-likelihoods and declared cell-level leave-one-out *unsupported*. That is
    deliberate, and it is the subtle part of comparing single-cell models —
    worth understanding before reading the result.

    The instinctive thing is to score each model by leave-one-**cell**-out
    (cell-level WAIC / PSIS-LOO). But this model gives every cell its **own**
    capture probability $\nu_c$ (`p_capture`) — a latent informed by *that cell
    alone*. An honest held-out-cell prediction must integrate $\nu_c$ over its
    **prior**, because the other cells say nothing about it:

    $$
    p(\tilde u_c \mid U_{-c}) =
    \iint p(\tilde u_c \mid \theta, \nu)\, p(\nu)\,
    p(\theta \mid U_{-c})\, d\nu\, d\theta .
    $$

    The pointwise log-likelihood, however, conditions on the $\nu_c$ that was
    fit *to $u_c$ itself*. To turn that into a leave-one-out estimate, PSIS-LOO
    would have to reweight a **posterior-to-prior collapse** of $\nu_c$ — which
    it cannot — so the Pareto diagnostic gives $\hat{k} \ge 0.7$ for
    *essentially every cell*, and the WAIC variance penalty explodes
    ($p_{\mathrm{eff}}$ in the tens of millions). This is the textbook failure
    of leave-one-out under **per-observation latent variables**; it is a
    property of the design, not a defect of either model. (Full derivation:
    `paper/_model_comparison.qmd`, *Leave-one-cell-out with cell-specific
    latents*.)

    The resolution is to change the **unit of observation** to the gene.
    Dropping one gene barely moves any parameter — the gene-level $\mu_g, r_g$
    are pinned by ~75k cells, and every $\nu_c$ is still informed by the
    thousands of *retained* genes in its cell — so leave-one-gene-out is
    well-posed and stable. Because **every realistic single-cell model needs
    variable capture**, the gene is the right unit for `scribe` model
    comparison, and `compare_models` switches to it automatically whenever a
    cell-specific latent is present (hence the message above, and why
    `mc.rank()` / `mc.psis_loo()` deliberately refuse here and point you to the
    gene-level comparison).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The right lens: gene-level comparison

    `mc.gene_level_comparison(A, B)` works with the **per-gene** log-likelihoods
    (each gene summed over all cells). For every gene it reports the elpd
    difference between the two models, its standard error, and a z-score — so we
    can see exactly **which genes** the dispersion hierarchy helps, and by how
    much. A positive `elpd_diff` favours `mu_plus_r`.
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
    # Per-gene elpd difference vs significance. Positive => favours mu_plus_r.
    _d = gene_cmp["elpd_diff"].to_numpy()
    _z = gene_cmp["z_score"].to_numpy()
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.scatter(_d, _z, s=5, alpha=0.3)
    _ax.axhline(0, c="k", lw=0.5)
    _ax.axvline(0, c="k", lw=0.5)
    _ax.set_xscale("symlog")
    _ax.set_xlabel("per-gene elpd difference  (mu_plus_r − mu_only)")
    _ax.set_ylabel("z-score of the difference")
    _ax.set_title(
        f"{int((_z > 2).sum()):,} genes favour mu+r at z>2  vs  "
        f"{int((_z < -2).sum()):,} favouring mu-only"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does the dispersion hierarchy actually buy?

    The verdict is **decisive and one-sided**: a large block of genes is fit much
    better by `mu_plus_r` (high positive elpd differences, z-scores in the tens),
    and almost nothing prefers `mu_only`. And the **identity** of the winning genes
    is the real result. The top of the list is dominated by:

    - **Immunoglobulin genes** — `IGHM`, `IGHA2`, `IGHG1`–`IGHG4`, `IGKV*`, `JCHAIN`. These are the textbook *over-dispersed* transcripts of single-cell data: a few plasma/B cells express them at enormous, wildly variable levels while most cells sit near zero. A single shared $r_g$ cannot describe both regimes; a per-leaf dispersion can.
    - **Mitochondrially-encoded transcripts** — `MT-ND1`, `MT-ND5`, `MT-CO1`, `MT-CYB`, `MT-ATP6`, …, whose cell-to-cell variability differs sharply between the control and panobinostat arms.

    In other words, **the extra flexibility is spent exactly where over-dispersion genuinely varies across conditions and donors** — not smeared across the transcriptome. That is the signature of a model improvement that is *real* rather than overfit: it concentrates on a biologically coherent, mechanistically expected set of genes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The connection back to differential expression

    This is where model comparison and differential expression turn out to be two
    views of **one** phenomenon. In the DE tutorial, fitting the dispersion
    hierarchy unlocked a second axis of differential expression — the **log
    variance ratio** (`bio_lvr`), "did this gene's *variability* change with
    treatment?" — distinct from the usual log-fold-change in the mean. The genes
    that lit up there were the mitochondrial transcripts and the
    induced/suppressed programs whose *shape*, not just *level*, shifted.

    Those are the **same genes** that model comparison just identified as the ones
    `mu_plus_r` fits better. The logic closes neatly:

    - **Model comparison** asks *"is the dispersion hierarchy worth it?"* and answers *"yes — on the over-dispersed, differentially-variable genes."*
    - **Differential expression** asks *"what did the dispersion hierarchy reveal?"* and answers *"a differential-variance signal on those very genes."*

    A gene only earns a large `bio_lvr` if its dispersion really moved between
    arms; and a gene only earns a large gene-level `elpd_diff` if modeling that
    movement improved prediction. They are the same condition seen from two sides.
    The model comparison is the *license* to read the `bio_lvr` axis at all — it is
    the out-of-sample evidence that the dispersion structure is signal, not noise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    We took the modeling choice that the DE tutorial made implicitly — hierarchy on
    the mean only — and tested it head-to-head against the richer model that also
    puts the hierarchy on the dispersion, using `scribe.mc`.

    Three things to carry forward:

    1. **The machinery works on genuinely complex fits.** `compare_models` runs
       WAIC, PSIS-LOO, and gene-level comparison directly on crossed-hierarchical,
       variable-capture `mean_disp` models — the per-cell likelihood gathers each
       cell's leaf parameters internally, and PSIS-LOO is GPU-vectorized, so the
       whole comparison is fast at single-cell scale.

    2. **Choose the unit of observation deliberately.** Cell-level WAIC/PSIS-LOO is
       degenerate for high-dimensional count data — $p_{\mathrm{eff}}$ in the
       millions and $\hat{k} \ge 0.7$ everywhere. The $\hat{k}$ diagnostic is what
       *tells you* to stop trusting it and move to the gene level, where the
       comparison is stable and interpretable.

    3. **A real improvement is concentrated and explicable.** The dispersion
       hierarchy earns its keep on the immunoglobulin and mitochondrial genes — the
       famously over-dispersed transcripts whose variability shifts across
       conditions and donors — which are exactly the genes the differential-variance
       (`bio_lvr`) axis flagged in the DE analysis. Model selection and differential
       expression converge on the same biology.
    """)
    return


if __name__ == "__main__":
    app.run()
