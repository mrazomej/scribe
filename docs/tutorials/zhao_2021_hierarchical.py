import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Multi-donor differential expression with a *crossed hierarchical* `scribe` model

    Most differential-expression tutorials compare two conditions in a single sample. Real perturbation experiments are rarely that tidy: the same drug is applied to **several donors**, each with its own baseline, and we want the **population-level treatment effect** — not an artifact of which donor happened to dominate one arm.

    In this tutorial we fit **one joint model** to a crossed *donor × condition* design and read off the treatment effect while explicitly accounting for donor-to-donor heterogeneity. The running example is the **Zhao et al. (2021)** dataset, where peripheral cells from **seven donors** were profiled at baseline (`control`) and after treatment with **panobinostat**, a pan-**HDAC inhibitor**. Because HDAC inhibitors have a well-characterized transcriptional fingerprint (histone genes up, heat-shock and p53-stress programs up, MYC and the cell cycle down), we can check the recovered signature against textbook biology.

    We will not be shy about the math: the whole point of a generative model is that *every* quantity — the treatment effect, the donor deviations, the differential-expression call — is a parameter with a posterior, so knowing what is being computed is what lets you trust (or question) the answer.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The design, and why a *joint* model

    Seven donors, two conditions each, gives a $7 \times 2$ grid of **groups**
    (we call each present combination a *leaf*). Three ways to analyze it:

    1. **Pool everything**, ignore the donor labels, compare control vs.  panobinostat. Fast, but a single donor with many cells or an extreme baseline can drive the result, and the donor structure is thrown away.
    2. **Fit each donor separately**, do seven pairwise comparisons, average the answers. Honest about pairing, but every fit sees only its own cells — no sharing of statistical strength, and no coherent population estimand.
    3. **One joint model** in which a cell's mean expression is its donor's baseline *plus* a shared treatment effect *plus* that donor's own deviation. This is what we do here. It keeps the pairing (control and treated cells from the same donor are linked through that donor's parameters), shares strength across donors, and — crucially — lets us *separate* the treatment effect we care about from the donor heterogeneity we want to average over.

    The rest of the notebook builds option 3 and shows how to interrogate it.
    """
    )
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
    return Path, json, np, pd, pertpy, pickle, plt, sc, scribe, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Loading and filtering the data

    We load the dataset through `pertpy`, drop genes that are never observed, and remove low-quality cells with fewer than 1,000 UMIs. These are ordinary quality-control steps; nothing about normalization happens here, because in `scribe` normalization is part of the *model* (more on that below).
    """
    )
    return


@app.cell
def _(pertpy, sc):
    # Load the Zhao 2021 perturbation dataset.
    adata = pertpy.data.zhao_2021()

    # Drop genes with zero total counts (does not change per-cell UMI totals).
    sc.pp.filter_genes(adata, min_counts=1)

    # Remove low-UMI cells.
    _umi_thresh = 1_000
    sc.pp.filter_cells(adata, min_counts=_umi_thresh)

    print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes after QC")
    adata
    return (adata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We focus on the donors that received **panobinostat** and pull both their treated cells and their matched `control` cells into a single object, `adata_joint`. This is the crossed *donor × condition* table the model will see. The two grouping columns are:

    - `perturbation` — the **contrast of interest** (`control` vs `panobinostat`),
    - `sample` — the **donor**, which we want to account for but not test.
    """
    )
    return


@app.cell
def _(adata, pd):
    # Treat the control arm as dose 0 (keep the column's categorical dtype happy).
    adata.obs["dose_value"] = adata.obs["dose_value"].cat.add_categories(["0"])
    adata.obs.loc[adata.obs["perturbation"] == "control", "dose_value"] = "0"

    # Donors that received panobinostat...
    _perturbation = "panobinostat"
    _samples = adata.obs.loc[
        adata.obs["perturbation"] == _perturbation, "sample"
    ].unique()

    # ...and the matched control + treated cells from exactly those donors.
    _mask = adata.obs["sample"].isin(_samples) & adata.obs["perturbation"].isin(
        ["control", _perturbation]
    )
    adata_joint = adata[_mask].copy()
    perturbation = _perturbation

    # The crossed donor x condition table: this is our 7 x 2 = 14 leaves.
    crosstab = pd.crosstab(
        adata_joint.obs["sample"], adata_joint.obs["perturbation"]
    )
    crosstab
    return adata_joint, perturbation


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Every donor appears in **both** columns — a fully crossed, balanced-ish design. Those 14 non-empty cells are the model's *leaves*. Reading **down a column** later will compare a donor against itself across the two conditions (the paired contrast); reading **across a row** compares donors within a condition (the heterogeneity we average over).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Before modeling, the one plot we always look at: the distribution of total UMIs per cell. If library size varies by close to an order of magnitude, each cell is effectively sequenced at its own depth, and the model must be told so explicitly. In `scribe` that is the **per-cell capture probability** (`variable_capture=True`), the principled replacement for "divide by total counts".
    """
    )
    return


@app.cell
def _(adata_joint, np, plt, sns):
    # Plot total UMI count distribution
    _fig, _ax = plt.subplots(figsize=(5, 4))
    sns.ecdfplot(np.asarray(adata_joint.X.sum(axis=1)).ravel(), ax=_ax)
    _ax.set_xscale("log")
    _ax.set_xlabel("total UMI count per cell")
    _ax.set_ylabel("ECDF")
    _ax.set_ylim(-0.01, 1.01)
    if _ax.legend_ is not None:
        _ax.legend_.remove()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The model, written out

    ### Counts: a Negative Binomial with per-cell capture

    At its core `scribe` models each UMI count $u_{cg}$ (cell $c$, gene $g$) with a **Negative Binomial**, the distribution that already underlies DESeq2/edgeR and that drops out of a two-state promoter model of transcription. With `parameterization="mean_odds"` we work in terms of a gene **mean expression** $\mu_g$ and a shared **odds** $\phi = p/(1-p)$, related to the raw NB parameters by $r_g = \mu_g\,\phi$ and $p = 1/(1+\phi)$. The mean–odds view breaks the banana-shaped $(r,p)$ degeneracy and makes $\mu_g$ — the thing the data pins down directly — the primary object.

    Because library sizes vary, each cell gets its own **capture probability** $\nu_c \in (0,1)$, producing a cell-specific effective success probability

    $$
    \hat{p}_c \;=\; \frac{p}{\nu_c + p\,(1-\nu_c)} .
    $$

    So far this is the standard `scribe` NB-with-variable-capture model. The new part is what sits **on top of $\mu_g$**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Mean expression: an additive *crossed* hierarchy

    Each leaf $\ell$ is a (condition, donor) pair. Instead of giving every leaf a free mean, we **decompose** the log-mean additively:

    $$
    \log \mu_g^{(\ell)}
    \;=\;
    \underbrace{\log \mu_g^{\mathrm{pop}}}_{\text{population baseline}}
    \;+\;
    \underbrace{\alpha_g\big[\mathrm{cond}(\ell)\big]}_{\text{treatment effect (fixed)}}
    \;+\;
    \underbrace{\beta_g\big[\mathrm{donor}(\ell)\big]}_{\text{donor effect (random)}} .
    $$

    Two factors, two very different roles:

    - **$\alpha_g$ — the treatment effect — is a *fixed* effect.** It is the contrast we are actually asking about, so we give it a **weakly-informative Gaussian** prior with a *fixed* scale and **no learned shrinkage**: $\alpha_{g,c} = \sigma_\alpha\, z_{g,c}^{\alpha}$, $z\sim\mathcal N(0,1)$.  Why no adaptive shrinkage? A two-level factor has almost no information to estimate its own variance, and a learned scale would happily pull the contrast toward zero — exactly the effect we are trying to measure. The identified quantity is the **difference** $\alpha_g[\text{panobinostat}] - \alpha_g[\text{control}]$.

    - **$\beta_g$ — the donor effect — is a *random* effect** with a **regularized horseshoe** prior. With seven donors there *is* information to learn how much donors vary, and the horseshoe adaptively shrinks: donors that look alike on a gene are pulled together, while a genuinely deviant donor is left alone. Each $\beta_g[d]$ is **zero-mean**, so it captures deviations *from* the population baseline rather than competing with it.

    The population baseline $\log\mu_g^{\mathrm{pop}}$ is the only free intercept, which is what makes the decomposition identifiable.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Why this is the right shape for the question

    Splitting $\log\mu_g^{(\ell)}$ into a fixed treatment term and a random donor term is the formal statement of "estimate the average drug effect while controlling for the donor." The treatment effect is **shared** across donors (so all 14 leaves inform it), the donor deviations soak up the baseline-and-response variability we do not want to mistake for a drug effect, and because control and treated cells of one donor pass through the *same* $\beta_g[d]$, the design stays **paired**.

    Only the **expression** target ($\mu$) gets this additive decomposition; the technical odds parameter is left free per leaf. That keeps the per-cell likelihood — the computational hot path — completely unchanged: the leaves are still an ordinary "dataset" axis, with structure layered *above* them.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fitting the model

    The call below is the whole model. `hierarchy=[...]` declares the two grouping factors and their effect types; `expression_dataset_prior` picks the prior family per factor (a fixed-scale Gaussian for the treatment contrast, a horseshoe for the donors). Crossing is implicit — listing two factors with no nesting means "donor crossed with condition."

    Fitting at this scale takes a while, so we **load a cached fit** when one is present and only run the optimizer otherwise.
    """
    )
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

    # Define output path
    _out_path = (
        data_dir
        / "scribe_results"
        / f"scribe_hierarchical_{perturbation}_joint.pkl"
    )

    # Fit/load pre-fit model
    if _out_path.exists():
        with open(_out_path, "rb") as _f:
            results_joint = pickle.load(_f)
    else:
        results_joint = scribe.fit(
            adata_joint,
            parameterization="mean_odds",
            variable_capture=True,
            unconstrained=True,
            positive_transform="exp",
            # Donor (sample) CROSSED with condition (perturbation):
            hierarchy=[
                # 2-level contrast of interest -> fixed, weakly-informative.
                scribe.GroupLevel("perturbation", effect_type="fixed"),
                # 7 donors -> random effect with adaptive shrinkage.
                scribe.GroupLevel("sample"),
            ],
            expression_dataset_prior={
                "perturbation": "gaussian",  # fixed-scale contrast
                "sample": "horseshoe",  # shrink across donors
            },
            prob_dataset_prior="gaussian",  # technical odds: leaf-exchangeable
            early_stopping={
                "enabled": True,
                "patience": 1000,
                "restore_best": True,
            },
            n_steps=100_000,
            batch_size=4096,
            gene_coverage=0.99,
        )
        with open(_out_path, "wb") as _f:
            pickle.dump(results_joint, _f)

    results_joint
    return (results_joint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Did the fit work? Three diagnostics

    ### 1. The ELBO loss

    Variational inference maximizes the **evidence lower bound (ELBO)**; the loss we plot is its negative. We do not read it quantitatively — we just want the textbook shape: a fast initial drop that flattens into a plateau. Spikes or upward drift would be warnings.
    """
    )
    return


@app.cell
def _(results_joint, scribe):
    _fig = scribe.viz.plot_loss(results_joint, figsize=(7, 3))
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2. Mean calibration — *per leaf*

    A converged loss does not prove the model describes the **counts**. The mean-calibration plot compares, for each gene, the **observed** mean count to the model's **predicted** mean; points on the diagonal mean the fit reproduces the first moment of the data.

    Because we have a crossed hierarchy, `scribe` lays the panels out on a **condition × donor grid**: rows are the two conditions, columns are the seven donors. This is the multi-factor layout doing its job — reading **down a column** shows control vs panobinostat *for one donor*, exactly the paired comparison the model encodes. A leaf that fell off the diagonal would point to a donor or condition the additive structure is failing to capture.
    """
    )
    return


@app.cell
def _(adata_joint, results_joint, scribe):
    # Per-leaf observed-vs-predicted means. The panels auto-arrange as a
    # condition (rows) x donor (cols) grid from the model's grouping spec.
    _fig = scribe.viz.plot_mean_calibration(results_joint, counts=adata_joint)
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For the most part, we have a very good fit as most points are close to the diagonal. The exceptions are samples `PW051` and `PW053`. However, because of this diagnostic, if wanted, we could look deeper into what could have caused this mismatch.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 3. Posterior predictive check

    Finally, a **posterior predictive check (PPC)**: simulate fresh UMI counts from the fitted model and overlay their distribution on the real histograms, for a spread of genes across the expression range. If the generative story is adequate, the simulated bands should track the observed counts.

    Because this is a multi-dataset fit, each simulated cell is drawn from **its own leaf's** parameters, so the plot below is an honest aggregate over all fourteen donor × condition leaves. That makes it a good global check — but it cannot tell us whether any *individual* fit is good. We start global, then zoom in.
    """
    )
    return


@app.cell
def _(adata_joint, results_joint, scribe):
    _fig = scribe.viz.plot_ppc(
        results_joint,
        adata_joint,
        n_genes=16,
        n_rows=4,
        n_samples=256,
        figsize=(7, 7),
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Per-dataset PPC: the quality of an *individual* fit

    The mean-calibration grid flagged `PW051` and `PW053`. The PPC can zoom into exactly those leaves: `plot_ppc(..., dataset=...)` swaps in a single leaf's parameter view (`results.get_dataset(leaf)` under the hood) and the **observed cells for that leaf only**, so we judge one donor × condition fit at a time. Address a leaf either by integer index or, more readably, by a `{factor: level}` dict.

    The two donors below sit at opposite ends of the data-richness scale, which is the structural reason their fits differ: **`PW030`** contributes ≈19,000 control cells, while **`PW051`** — one of the donors the calibration plot flagged — contributes only ≈1,200. A data-rich leaf's predictive bands should hug the observed histograms; a sparse leaf's bands are wider and the fit is more stretched, which is what the calibration plot was picking up.
    """
    )
    return


@app.cell
def _(adata_joint, results_joint, scribe):
    # A data-rich donor: its individual fit should track the data tightly.
    _fig = scribe.viz.plot_ppc(
        results_joint,
        adata_joint,
        dataset={"sample": "PW030", "perturbation": "control"},
        n_genes=16,
        n_rows=4,
        n_samples=256,
        figsize=(7, 7),
    )
    _fig.fig
    return


@app.cell
def _(adata_joint, results_joint, scribe):
    # A sparse donor the mean-calibration flagged: the same diagnostic, where
    # the individual fit is most stretched (≈1,200 cells vs ≈19,000 above).
    _fig = scribe.viz.plot_ppc(
        results_joint,
        adata_joint,
        dataset={"sample": "PW051", "perturbation": "control"},
        n_genes=16,
        n_rows=4,
        n_samples=256,
        figsize=(7, 7),
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Reading the model's structure

    Before any differential-expression machinery, the fitted hierarchy is already interpretable. The additive effects are stored as parameters with posteriors, and `scribe` exposes them directly.

    ### The treatment effect

    `results.get_factor_effect("perturbation")` returns the fitted $\alpha_g$ effect. Since `perturbation` is the fixed contrast factor, the identified quantity is the **contrast** — the per-gene log-mean shift from control to panobinostat — which we summarize by its posterior mean.
    """
    )
    return


@app.cell
def _(np, plt, results_joint):
    # The fixed treatment effect: alpha[panobinostat] - alpha[control], in log-mu.
    # (Gene *names* live in the DE table below; here we work with the array.)
    _tx = results_joint.get_factor_effect("perturbation")
    treatment_effect = np.asarray(_tx.map_contrast("panobinostat", "control"))
    print(f"treatment-effect factor: {_tx!r}")
    print(
        f"{int((np.abs(treatment_effect) > np.log(2)).sum()):,} genes shift by "
        f"more than 2-fold (|log-mu effect| > log 2)"
    )

    # Plot distribution of treatment effects
    _fig, _ax = plt.subplots(figsize=(5, 3.2))
    _ax.hist(treatment_effect, bins=80, color="steelblue")
    _ax.axvline(0, ls="--", c="k", lw=1)
    _ax.set_xlabel(r"per-gene treatment effect $\bar\alpha_g$ (log-mean)")
    _ax.set_ylabel("genes")
    _fig
    return (treatment_effect,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Donor heterogeneity — the reason we went joint

    `results.get_factor_effect("sample")` returns the **random** donor effects $\beta_g[d]$: a $7 \times G$ matrix of zero-mean, log-mean deviations. Their spread across donors is the heterogeneity the model absorbed so that it did *not* leak into the treatment effect. If this spread were negligible, simple pooling would have been fine; if it is large, the joint model earned its keep.

    Below we compare, gene by gene, the **magnitude of the treatment effect** against the **donor-to-donor spread**. Genes where the donor spread rivals or exceeds the treatment effect are exactly the ones a naive pooled analysis would get wrong.
    """
    )
    return


@app.cell
def _(np, plt, results_joint, treatment_effect):
    # Extract random donor effect
    _donor = results_joint.get_factor_effect("sample")
    _eff = np.asarray(
        _donor.effects()
    )  # (n_donors, G) posterior-mean log-mu devs
    _donor_spread = _eff.std(axis=0)  # per-gene SD across donors
    _tx_abs = np.abs(treatment_effect)

    # Plot donor-to-donor spread vs treatment effect
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(_donor_spread, _tx_abs, s=4, alpha=0.3)
    _lim = float(np.nanmax([_donor_spread.max(), _tx_abs.max()])) * 1.05
    _ax.plot([0, _lim], [0, _lim], ls="--", c="k", lw=1)
    _ax.set_xlabel("donor-to-donor spread  (SD of $\\beta_g[d]$)")
    _ax.set_ylabel("|treatment effect|  $|\\bar\\alpha_g|$")
    _ax.set_title(f"{_donor.n_levels} donors: heterogeneity vs treatment")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Recovering the paired groups

    `scribe` also lets us pull back the leaf structure for any downstream slicing.  `results.get_group(sample="...")` returns the leaves for one donor, indexed by condition; `iter_groups("sample")` walks every donor. Each value is an ordinary single-dataset results view, so you can drop it into any per-leaf analysis.
    """
    )
    return


@app.cell
def _(results_joint):
    # One donor's paired (control, panobinostat) views, keyed by condition.
    _g = results_joint.get_group(sample=results_joint.group_levels("sample")[0])
    print("group:", repr(_g))
    print("leaf labels:", _g.labels)
    # Enumerate all seven paired donor groups.
    print("\nall donor groups:")
    for _level, _grp in results_joint.iter_groups("sample"):
        print(f"  {_level}: conditions = {_grp.keys()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Differential expression: the donor-averaged treatment effect

    ### The estimand

    We want the population treatment effect, *paired within donor*. `scribe`'s compositional DE works in **centered-log-ratio (CLR)** space, where each gene is compared to the geometric mean across genes — a **reference-free** coordinate in which differences are additive. For each donor $d$ present in both arms and each posterior draw $s$, we form the within-donor CLR difference

    $$
    \Delta_g^{(d,s)}
    = \mathrm{clr}\!\big(\rho_g^{(\text{control},\,d,\,s)}\big)
    - \mathrm{clr}\!\big(\rho_g^{(\text{panobinostat},\,d,\,s)}\big),
    $$

    and then **average over donors** to get the paired main effect

    $$
    \bar\Delta_g^{(s)} = \sum_d w_d\, \Delta_g^{(d,s)} .
    $$

    The donor effect $\beta_g[d]$ cancels inside each within-donor difference, so $\bar\Delta_g$ is a clean draw from the posterior of the *average treatment effect*. The whole population contrast is one call:
    """
    )
    return


@app.cell
def _(results_joint, scribe):
    # Paired main effect across donors. delta = CLR(control) - CLR(panobinostat),
    # so a gene UP in panobinostat has a NEGATIVE delta.
    # n_samples sets N in delta_samples (more draws -> smoother lfsr/PEFP).
    # batch_size chunks the draw and offloads it to host RAM (automatic for
    # n_samples > 500), keeping GPU memory free for the composition sampling.
    results_de_raw = scribe.compare_groups(
        results_joint,
        "perturbation",
        "control",
        "panobinostat",
        n_samples=5_000,
        batch_size=500,
    )
    results_de_raw
    return (results_de_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### A cautionary detour: CLR needs a stable reference

    CLR is principled, but it has a sharp edge. The reference it divides by is the **geometric mean of the log-fractions across *all* genes** — and a typical single-cell matrix has thousands of genes sitting at essentially zero, whose log-fractions are hugely negative and jitter from draw to draw. That jitter contaminates the reference and blows up the CLR contrast of the genes we actually care about.

    Look at the top "hits" from the unfiltered run: CLR shifts of $\pm 40$ (an $e^{40}$-fold change is not biology) on near-zero pseudogenes, mitochondrial tRNAs, and stray neuronal markers. This is the reference exploding, not a drug effect.
    """
    )
    return


@app.cell
def _(np, results_de_raw):
    _df_raw = results_de_raw.to_dataframe(
        tau=np.log(1.1),
        target_pefp=0.05,
        metrics="clr",
        column_naming="prefixed",
    )
    # Most extreme |CLR shift| -- the unstable, near-zero genes.
    _df_raw.reindex(
        _df_raw["clr_delta_mean"].abs().sort_values(ascending=False).index
    )[["gene", "clr_delta_mean", "clr_lfsr"]].head(12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The principled fix: keep the genes that carry the mass, pool the rest into "other"

    The cure is *not* to delete the quiet genes — that would break the simplex. Instead we **pool** the long tail of near-silent genes into a single "other" pseudo-gene before forming CLR coordinates. By the **Dirichlet closure property** (summing a subset of Dirichlet components gives another Dirichlet) this aggregation is **exact** — no information is lost — and it gives the geometric-mean reference something stable to stand on. CLR stays exactly as principled; we have only stopped asking it to resolve genes that carry no signal.

    How aggressively should we pool? Rather than an arbitrary UMI floor, we use a **compositional-coverage** rule — the natural choice *inside* a compositional framework: keep the smallest set of genes whose MAP composition reaches a target fraction of the total expressed mass (set below), in *either* arm, and pool the rest. It is **scale-free** (invariant to sequencing depth) and keeps exactly the genes that actually make up the transcriptome, rather than a hand-picked count cutoff.

    This nests cleanly with a filter we already applied at *fit* time: `gene_coverage=0.99` pooled the un-modeled tail into the model's own `_other` pseudo-gene, so the fitted composition already sums to one over the full transcriptome. The DE composition samples that whole simplex — `_other` included — and our coverage mask simply folds `_other` back into the DE "other" tail (it is the compositional anchor, never a reportable gene). So the two filtering stages compose: closure is preserved end to end, and the CLR reference is always the full transcriptome.

    One API note: `compare_groups` does not retain the per-pair simplex (averaging simplices would not equal the paired CLR average), so the in-place `set_composition_coverage` does not apply to it. Instead we **build** the mask from the model's mean expression — `results_de_raw.composition_coverage_mask(...)`, which reads only `mu_map` — and pass it up front as `gene_mask=`, so the pooling happens **inside each donor pair, before CLR**.
    """
    )
    return


@app.cell
def _(results_de_raw, results_joint, scribe):
    COMPOSITION_COVERAGE = 0.95

    # Keep the genes making up this fraction of the expressed mass in either
    # arm; pool the rest into "other". The builder reads only the model's mean
    # expression, so it works even though compare_groups stores no simplex.
    gene_mask = results_de_raw.composition_coverage_mask(COMPOSITION_COVERAGE)
    print(
        f"keeping {int(gene_mask.sum()):,} / {gene_mask.size:,} genes "
        f"({gene_mask.mean():.0%}); the rest pool into 'other'"
    )

    # Re-run the paired contrast with the mask applied within each pair. The same
    # n_samples sets the posterior draw count behind the lfsr/PEFP estimates;
    # batch_size keeps the large draw within memory.
    results_de = scribe.compare_groups(
        results_joint,
        "perturbation",
        "control",
        "panobinostat",
        gene_mask=gene_mask,
        n_samples=5_000,
        batch_size=500,
    )
    results_de
    return gene_mask, results_de


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The compositional view

    With a stable reference, the standard `scribe` DE plots are well-defined. The **volcano** puts the CLR contrast on the $x$-axis and $-\log_{10}(\text{lfsr})$ on the $y$-axis, where **lfsr** (local false sign rate) is the posterior probability that we have the *direction* of the effect wrong — the Bayesian replacement for a $p$-value. Genes called differentially expressed (combining an effect-size threshold $\tau$ and a target false-sign rate) are highlighted; the mean-expression panel shows each gene's CLR mean in the two arms.

    A note worth internalizing — and a real result on this dataset. Under a **broad** perturbation like an HDAC inhibitor *many* genes move, so the compositional reference itself drifts, and CLR's *confident* calls become the genes whose **proportion** swings hardest rather than the coherent expression program. Here those turn out to be **mitochondrial genes and MT-pseudogenes** (`MT-ND6`, `MT-TD`, `MTRNR2L1`, …) — mitochondrial fraction is a classic single-cell confounder — while the canonical HDAC markers carry **high CLR lfsr**: once the whole composition is shifting, the sign of any single gene's *relative* change is genuinely uncertain. So CLR is behaving exactly as designed — guarding against being fooled by composition — but for this question it is the **more conservative** lens: at a strict error threshold its only confident calls are these compositional outliers. That is **not** the same as CLR contradicting the biology. As we show at the end of this section, CLR and the biological view agree in *direction* — CLR simply holds the coherent program at **lower confidence**: it is **under-confident here, not incoherent**. (Routinely dropping mitochondrial genes at QC would clean it up; we keep them in the headline run to make the point — and then drop them, by *name*, to see what's left.)
    """
    )
    return


@app.cell
def _(np, results_de, scribe):
    TAU = np.log(1.1)
    TARGET_PEFP = 0.05
    _fig = scribe.viz.plot_de_volcano(
        results_de, mode="clr", tau=TAU, target_pefp=TARGET_PEFP, figsize=(5, 4)
    )
    _fig.fig
    return TARGET_PEFP, TAU


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Cleaning up by identity — and what it reveals

    The mitochondrial dominance is a *known technical confounder*, so the principled fix is to drop those genes by **name** — mitochondrial, ribosomal, and hemoglobin families — a decision made from gene *identity*, a priori and **independent of the contrast**. (The tempting alternative — thresholding on the large CLR values themselves and re-running — is *circular*: it selects genes by the very effect we are testing, defining away real signal along with artifacts. Filtering on identity, or on expression/precision, avoids that.) `exclude_gene_name_mask` builds a keep-mask from name prefixes and regexes; we `&` it with the composition-coverage mask and re-run the same CLR contrast.
    """
    )
    return


@app.cell
def _(gene_mask, results_de_raw, results_joint, scribe):
    # Drop mitochondrial (incl. MT-pseudogenes), ribosomal, and hemoglobin
    # genes by name — known confounders, chosen by identity, not by effect size.
    _nuisance_keep = results_de_raw.exclude_gene_name_mask(
        prefixes=("MT-", "RPL", "RPS", "MRPL", "MRPS"),
        patterns=(r"^MT(RNR|CO\d|ND\d|ATP\d|CYB)", r"^HB[ABDEGMQZ]\d?$"),
    )
    _gene_mask_clean = gene_mask & _nuisance_keep
    print(
        f"dropping {int((gene_mask & ~_nuisance_keep).sum())} mito/ribo/HB genes; "
        f"keeping {int(_gene_mask_clean.sum())} of {gene_mask.size:,}"
    )
    results_de_clean = scribe.compare_groups(
        results_joint,
        "perturbation",
        "control",
        "panobinostat",
        gene_mask=_gene_mask_clean,
        n_samples=5_000,
        batch_size=500,
    )
    return (results_de_clean,)


@app.cell
def _(TARGET_PEFP, TAU, results_de_clean, scribe):
    _fig = scribe.viz.plot_de_volcano(
        results_de_clean,
        mode="clr",
        tau=TAU,
        target_pefp=TARGET_PEFP,
        figsize=(5, 4),
    )
    _fig.fig
    return


@app.cell
def _(np, results_de, results_de_clean):
    # With the confounders gone, what confident CLR calls remain — and do the
    # canonical markers recover? Compare the cleaned CLR lfsr to the biological
    # lfsr for the same markers, and the CLR<->bio correlation.
    _clean = results_de_clean.to_dataframe(
        tau=np.log(1.1),
        target_pefp=0.05,
        metrics="clr",
        column_naming="prefixed",
    )
    _n_de = int(_clean["clr_is_de"].sum())
    _top = (
        _clean[_clean["clr_lfsr"] <= 0.05]
        .reindex(
            _clean[_clean["clr_lfsr"] <= 0.05]["clr_delta_mean"]
            .abs()
            .sort_values(ascending=False)
            .index
        )[["gene", "clr_delta_mean", "clr_lfsr"]]
        .head(10)
    )
    print(f"confident CLR calls after cleaning (lfsr <= 0.05): {_n_de}")
    _bio = results_de.to_dataframe(
        tau=np.log(1.1), metrics="all", column_naming="prefixed"
    )[["gene", "bio_lfc_mean"]]
    _merged = _clean[["gene", "clr_delta_mean"]].merge(_bio, on="gene")
    print(
        "cleaned-CLR vs biological log-fold-change: Pearson "
        f"{_merged['clr_delta_mean'].corr(_merged['bio_lfc_mean']):.3f}, "
        f"Spearman {_merged['clr_delta_mean'].corr(_merged['bio_lfc_mean'], method='spearman'):.3f}"
    )
    _top
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The result is the cleanest statement of the whole section. Dropping the confounders does **not** surface the coherent program — it *empties the CLR hit list almost entirely*. The mitochondrial genes are gone, and what remains are at most a handful of confident calls; the canonical HDAC markers **keep their high CLR lfsr** (essentially unchanged), and the correlation with the biological view barely moves. In other words, CLR's confidence here was concentrated *in the confounders themselves* — strip them and there is little confident *compositional* signal left, because the markers' limitation was never the reference or the confounders but their genuine per-draw uncertainty under a shifting composition.

    So name-based filtering is the right, non-circular hygiene — and it *confirms*, rather than fixes, the conclusion: for a broad perturbation, the coherent program is not hiding in the composition behind a few loud genes. It lives in the **biological** view, which we read next.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The biological view, and the panobinostat signature

    The complementary lens is the **biological log-fold-change** (`mode="bio"`): the change in each gene's underlying NB **mean expression**, with its own lfsr. Where CLR asks "how did the composition shift," the biological view asks "how did this gene's expression change" — the directly interpretable question for a global drug response. It is unaffected by the reference drift that made CLR uncertain, so the canonical markers that carried high *CLR* lfsr come back here with **biological lfsr ≈ 0** (the posterior is certain of their direction). For a broad perturbation like this, the biological view is the **primary** readout and CLR is the compositional guardrail; the textbook signature lives here.
    """
    )
    return


@app.cell
def _(TARGET_PEFP, TAU, results_de, scribe):
    _fig = scribe.viz.plot_de_volcano(
        results_de, mode="bio", tau=TAU, target_pefp=TARGET_PEFP, figsize=(5, 4)
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's read the actual hit list. We export the full table and orient the biological log-fold-change as **panobinostat vs control** (positive = up after treatment). One honest caveat: a fold-change ranking rewards the largest *relative* swings, and for the least-expressed retained genes those are the noisiest. So we read the table with **mean expression in view** and focus the headline on **well-expressed** hits (mean $\geq 5$ UMIs in an arm), then validate against known markers below.
    """
    )
    return


@app.cell
def _(np, results_de):
    _df = results_de.to_dataframe(
        tau=np.log(2.0),
        target_pefp=0.05,
        metrics="all",
        column_naming="prefixed",
    )
    # bio_lfc_mean is log(mu_control / mu_panobinostat); flip + rescale to log2(pano/ctrl).
    de_table = _df.assign(
        log2fc_pano=-_df["bio_lfc_mean"] / np.log(2.0),
        lfsr=_df["bio_lfc_lfsr"],
        mu_control=_df["bio_mu_A_mean"],
        mu_pano=_df["bio_mu_B_mean"],
    )[["gene", "log2fc_pano", "lfsr", "mu_control", "mu_pano"]]

    # DE call: confident direction (lfsr <= 0.05) and at least a 2-fold change.
    _sig = de_table[
        (de_table["lfsr"] <= 0.05) & (de_table["log2fc_pano"].abs() >= 1.0)
    ]
    print(
        f"{len(_sig)} genes called DE (|log2FC| >= 1, lfsr <= 0.05): "
        f"{int((_sig['log2fc_pano'] > 0).sum())} up, "
        f"{int((_sig['log2fc_pano'] < 0).sum())} down in panobinostat"
    )
    # Headline: the well-expressed strong hits, split by direction so the up and
    # down tables don't overlap (positive log2fc = up in panobinostat).
    _well = _sig[(_sig["mu_control"] >= 5) | (_sig["mu_pano"] >= 5)]
    up_in_pano = (
        _well[_well["log2fc_pano"] > 0]
        .sort_values("log2fc_pano", ascending=False)
        .head(20)
        .round(2)
    )
    down_in_pano = (
        _well[_well["log2fc_pano"] < 0]
        .sort_values("log2fc_pano")
        .head(20)
        .round(2)
    )
    up_in_pano
    return de_table, down_in_pano


@app.cell
def _(down_in_pano):
    down_in_pano
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Validating against known markers

    The cleanest sanity check is a targeted one: do the **canonical HDAC-inhibitor / panobinostat markers** come back with the sign biology predicts, and with the posterior confident about it? We pull a curated panel straight from the DE table.
    """
    )
    return


@app.cell
def _(de_table):
    _markers = [
        # expected UP after panobinostat
        "MT2A",
        "MT1X",
        "H1F0",
        "HIST1H1C",
        "HSPA1A",
        "HSPB1",
        "GADD45A",
        "DDIT3",
        "CDKN1A",
        "NEAT1",
        # expected DOWN after panobinostat
        "MYC",
        "CCND1",
        "MDM2",
        "CD163",
        "CD14",
        "MSR1",
        "FCGR3A",
    ]
    marker_panel = (
        de_table[de_table["gene"].isin(_markers)]
        .set_index("gene")
        .reindex(_markers)
        .round(3)
    )
    marker_panel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Does it make biological sense?

    Yes — emphatically. Every canonical marker returns with the expected direction and **lfsr $\approx 0$** (the posterior is essentially certain of the sign), and the well-expressed hit list is a coherent program:

    **Up after treatment.**

    - **Metallothioneins** (`MT2A`: mean 26 → 108 UMIs; `MT1X`, `MT1G`, `MT1H`, `MT1E`) — a hallmark metal/oxidative-stress response, a dominant high-confidence block.
    - **Linker histones** (`H1F0`, `HIST1H1C`) — the chromatin-remodeling response that is the defining footprint of HDAC inhibition.
    - **Heat-shock and the integrated stress / p53 program** (`HSPA1A`, `HSPB1`, `GADD45A`, `DDIT3`, `NEAT1`, and the cell-cycle brake **`CDKN1A`**/p21), plus acute-phase genes (`SAA1`, `PI3`).

    **Down after treatment.**

    - **`MYC`** and cyclins (`CCND1`, `CCNE2`) — proliferation shutting down, the expected cytostatic effect.
    - **`MDM2`** — the p53 E3 ligase falling, consistent with **p53 activation**.
    - A coherent **myeloid / macrophage program** (`CD163`, `CD14`, `MSR1`, `FCGR3A`, `FCER1G`, `SRGN`) — the drug suppressing/depleting the monocyte–macrophage compartment.

    Crucially, this is the effect **averaged over seven donors with their heterogeneity modeled away** — not an artifact of one dominant sample. The hierarchy estimated the contrast we asked about while absorbing the donor variation we did not.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## A shortcut: differential expression straight from the fixed effect

    Everything above used `estimand="paired_main_effect"` — it samples per-donor compositions and contrasts them in CLR space. But recall what the model encodes: the treatment is a **fixed, shared effect** $\alpha$, and within any donor the baseline $\beta$ cancels, so the biological log-fold-change *is* $\alpha[\text{panobinostat}] - \alpha[\text{control}]$ — the same for every donor. We can therefore read the treatment DEG **straight from the fitted $\alpha$**, with no composition sampling at all.

    `scribe.compare_groups(..., estimand="effect")` does exactly that: it pulls the $\alpha$ contrast from the hierarchy (`get_factor_effect` under the hood), so it is essentially **instant and memory-light** — the practical choice when the full paired-CLR pass is too heavy, and the *coherent* view for a broad perturbation like this. The returned `delta` is now a **log-mean** effect (not a CLR contrast), carrying its own lfsr; the `_other` pseudo-gene is dropped automatically. Because it is the same quantity as the biological log-fold-change computed the long way, the two agree to numerical precision (we check the correlation below).
    """
    )
    return


@app.cell
def _(gene_mask, np, results_de, results_joint, scribe):
    # The treatment effect, read directly from the fitted alpha — no composition
    # sampling. Reuses the already-drawn posterior, so this is near-instant.
    results_effect = scribe.compare_groups(
        results_joint,
        "perturbation",
        "control",
        "panobinostat",
        estimand="effect",
        gene_mask=gene_mask,
        n_samples=5_000,
    )

    # It is the same quantity as the biological log-fold-change (clr_* columns
    # here hold the log-mean effect, not a CLR contrast), so they agree:
    _eff = results_effect.to_dataframe(
        tau=np.log(2.0),
        target_pefp=0.05,
        metrics="clr",
        column_naming="prefixed",
    )
    _bio = results_de.to_dataframe(
        tau=np.log(2.0), metrics="all", column_naming="prefixed"
    )[["gene", "bio_lfc_mean"]]
    _merged = _eff.merge(_bio, on="gene")
    print(
        "effect-estimand delta vs biological log-fold-change: corr = "
        f"{_merged['clr_delta_mean'].corr(_merged['bio_lfc_mean']):.4f}  "
        f"({int(_eff['clr_is_de'].sum())} genes called DE)"
    )

    # The canonical markers, recovered by the fast alpha path (positive
    # log2FC = up in panobinostat) — the same signature as the bio view above.
    effect_de = _eff.assign(log2fc_pano=-_eff["clr_delta_mean"] / np.log(2.0))[
        ["gene", "log2fc_pano", "clr_lfsr"]
    ]
    _markers = [
        "MT2A",
        "MT1X",
        "H1F0",
        "HIST1H1C",
        "CDKN1A",
        "MYC",
        "MDM2",
        "CD163",
        "CD14",
        "MSR1",
        "FCGR3A",
    ]
    effect_de[effect_de["gene"].isin(_markers)].set_index("gene").reindex(
        [m for m in _markers if m in set(effect_de["gene"])]
    ).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### CLR and the biological view agree — conservative, not incoherent

    We have treated the compositional (CLR) and biological views as answering different questions — and they do — but it is worth asking directly: do they *disagree*? They do not. Plotting every gene's CLR contrast against its biological log-fold-change (left) shows a clear positive correlation: the canonical markers fall in the expected quadrants (metallothioneins / histones / p53-stress up, the myeloid program down), and the two lenses rank the response the same way. The scatter also makes **compositional closure** visible — the cloud of genes that rise in absolute expression yet *fall* in proportion, squeezed as the high-mass drivers claim more of the simplex. That is real compositional structure, not error.
    """
    )
    return


@app.cell
def _(np, pd, plt, results_de):
    _df = results_de.to_dataframe(
        tau=np.log(2.0), metrics="all", column_naming="prefixed"
    )
    # Orient both contrasts as panobinostat vs control (positive = up in pano).
    _bio = -_df["bio_lfc_mean"].to_numpy() / np.log(2.0)
    _clr = -_df["clr_delta_mean"].to_numpy() / np.log(2.0)
    _lfsr = _df["clr_lfsr"].to_numpy()
    _rho = pd.Series(_clr).corr(pd.Series(_bio), method="spearman")
    _g2i = {g: i for i, g in enumerate(_df["gene"].astype(str))}

    # As we relax the tolerated false proportion (PEFP), how much of the
    # canonical signature does CLR recover, and do the calls keep the right sign?
    _markers = [
        "MT2A",
        "MT1X",
        "H1F0",
        "HIST1H1C",
        "CDKN1A",
        "NEAT1",
        "MYC",
        "MDM2",
        "CD163",
        "CD14",
        "FCGR3A",
        "MSR1",
    ]
    _mset = set(_markers)
    _pefps = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    _nmark, _sign = [], []
    for _pe in _pefps:
        _d = results_de.to_dataframe(
            tau=np.log(2.0),
            target_pefp=_pe,
            metrics="all",
            column_naming="prefixed",
        )
        _c = _d[_d["clr_is_de"]]
        _nmark.append(len(set(_c["gene"].astype(str)) & _mset))
        _s = np.sign(_c["clr_delta_mean"]) == np.sign(_c["bio_lfc_mean"])
        _sign.append(float(_s.mean()) if len(_c) else np.nan)

    _fig, (_axA, _axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    _sc = _axA.scatter(_bio, _clr, c=_lfsr, s=6, cmap="viridis", alpha=0.5)
    _fig.colorbar(_sc, ax=_axA, label="CLR lfsr (sign uncertainty)")
    for _g in _markers:
        if _g in _g2i:
            _i = _g2i[_g]
            _axA.scatter(
                _bio[_i],
                _clr[_i],
                c="crimson",
                s=22,
                edgecolor="k",
                lw=0.4,
                zorder=5,
            )
            _axA.annotate(
                _g,
                (_bio[_i], _clr[_i]),
                fontsize=6.5,
                color="crimson",
                xytext=(2, 2),
                textcoords="offset points",
            )
    _axA.axhline(0, c="k", lw=0.5)
    _axA.axvline(0, c="k", lw=0.5)
    _axA.set_xlabel("biological log2 fold-change (pano / control)")
    _axA.set_ylabel("CLR contrast (log2 units, pano / control)")
    _axA.set_title(f"CLR vs biological DE — Spearman {_rho:.2f}")

    _axB.plot(_pefps, _nmark, "o-", color="crimson")
    _axB.set_xlabel("target PEFP (tolerated false proportion)")
    _axB.set_ylabel("# canonical markers called", color="crimson")
    _axB.tick_params(axis="y", labelcolor="crimson")
    _axB2 = _axB.twinx()
    _axB2.plot(_pefps, _sign, "s--", color="steelblue")
    _axB2.set_ylabel(
        "fraction of CLR calls agreeing with bio sign", color="steelblue"
    )
    _axB2.set_ylim(0.7, 1.005)
    _axB.set_title(
        "Conservative, not incoherent:\nthe signature emerges as PEFP relaxes"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    What separates the two views is **confidence, not direction.** CLR assigns the markers higher lfsr, so at a strict PEFP only the few ultra-confident driver genes clear the bar. But that is a property of the *threshold*, not the signal: relax the tolerated false proportion (right panel) and the full canonical signature emerges from CLR, the calls still agreeing with biology in sign.

    And CLR is the more conservative lens for a principled reason — it **assumes less and accounts for more.** It is provably invariant to capture-probability misspecification (the biological mean is *not* — it must trust the capture / library model); it references each gene against the whole composition; and under the gene-specific $p_g$ model it carries *shape* / overdispersion uncertainty into its lfsr, a term that simply cancels in the biological log-fold-change. Its wider intervals are **earned honesty** about what compositional data alone can claim — not noise.

    The scientific reading, then, is not "believe whichever lens clears a $p$-value-like cutoff." It is that a **robust, assumption-light estimator (CLR) and a confident, model-dependent one (the biological mean) converge** on the same panobinostat signature. That convergence — read with knowledge of the marker biology — is what earns confidence in the result, far more than any single PEFP threshold could. The generative model hands us *both* lenses from one posterior; the scientific judgment is ours to bring.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Recap

    We built **one** joint generative model of a crossed *donor × condition* experiment and, from a single posterior, read off:

    1. a **fit** that calibrates per leaf and passes posterior predictive checks;
    2. the **fitted structure** — a fixed treatment effect and the random donor deviations it was protected from, both as inspectable posteriors;
    3. a **donor-averaged, paired differential-expression** result whose **biological** log-fold-changes reproduce the canonical panobinostat / HDAC-inhibitor signature with high sign-confidence (lfsr ≈ 0 on every marker) — and the same calls, essentially for free, by reading the treatment effect $\alpha$ directly (`estimand="effect"`).

    Three lessons worth carrying forward. First, **CLR is principled but needs a stable reference**: pooling near-silent genes into "other" (an exact Dirichlet operation) is what lets the compositional contrast see signal instead of noise. Second, **CLR and the biological view answer different questions, and which one is the *confident* headline depends on the perturbation.** CLR asks "did the *relative* composition shift?" — invaluable as a guard against compositional artifacts, but under a broad, genome-wide response (like this one) the only calls it is *confident* about are the genes whose proportion swings hardest (here, mitochondrial genes). The biological log-fold-change asks "did the *mean expression* change?" — unaffected by the reference drift, and where the textbook signature emerged with high confidence. Third — and this is the subtle one — **conservative is not the same as wrong.** CLR is *under-confident here, not incoherent*: it agrees with the biological view in direction, recovers the full signature as the error tolerance relaxes, and is conservative precisely because it assumes less (capture-invariant) and accounts for more (global structure, gene-specific shape). The trustworthy conclusion comes from the **convergence** of the robust and the confident lens — read with scientific judgment — not from whichever one clears a threshold. The hierarchy ties it together: it estimates the effect we care about while explicitly accounting for the donor variation we do not.
    """
    )
    return


if __name__ == "__main__":
    app.run()
