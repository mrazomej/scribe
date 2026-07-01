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
    # Multi-donor differential expression with a *crossed hierarchical* `scribe` model

    Most differential-expression tutorials compare two conditions in a single sample. Real perturbation experiments are rarely that tidy: the same drug is applied to **several donors**, each with its own baseline **and its own response**, and we want the **population-level treatment effect** — the change we would expect in a *new* donor, not an artifact of which donor happened to contribute the most cells to one arm.

    In this tutorial we fit **one joint model** to a crossed *donor × condition* design and read off that population treatment effect while explicitly accounting for **two kinds of donor heterogeneity**: donors differ in their baseline expression, and they differ in *how strongly they respond* to the drug. The running example is the **Zhao et al. (2021)** dataset, where peripheral cells from **seven donors** were profiled at baseline (`control`) and after treatment with **panobinostat**, a pan-**HDAC inhibitor**. Because HDAC inhibitors have a well-characterized transcriptional fingerprint (histone genes up, heat-shock and p53-stress programs up, MYC and the cell cycle down), we can check the recovered signature against textbook biology.

    This tutorial does not hide the math — but it does not assume you arrived with it, either. Ideas that may be new if you come from a standard single-cell workflow — the *hierarchy* that separates the drug effect from each donor's quirks, the *compositional* (CLR) contrast, and the two different ways to read a differential-expression result — are each built up from the ground as we reach them. The reason it is worth the effort: in a generative model *every* quantity (the treatment effect, the donor deviations, the DE call itself) is a parameter with a full posterior, so understanding what is being computed is exactly what lets you trust — or question — the answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Compute note.** The differential-expression cells draw thousands of
    > posterior samples over ~17k genes. To keep GPU memory bounded they pass
    > `convert_to_numpy=True` — offloading each finished result to host RAM so it
    > does not stay resident on the device and accumulate across cells — together
    > with `batch_size=...` to chunk the draw. If you still hit a GPU
    > `RESOURCE_EXHAUSTED` out-of-memory error on a shared or smaller device,
    > give XLA more headroom *before launching* the notebook, e.g.
    > `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` or
    > `TF_GPU_ALLOCATOR=cuda_malloc_async`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The design, and why a *joint* model

    Seven donors, two conditions each, gives a $7 \times 2$ grid of **groups**
    (we call each present combination a *leaf*). Three ways to analyze it:

    1. **Pool everything**, ignore the donor labels, compare control vs.  panobinostat. Fast, but a single donor with many cells or an extreme baseline can drive the result, and the donor structure is thrown away.
    2. **Fit each donor separately**, do seven pairwise comparisons, average the answers. Honest about pairing, but every fit sees only its own cells — no sharing of statistical strength, and no coherent population estimand.
    3. **One joint model** in which a cell's mean expression is its donor's baseline, *plus* a treatment effect, *plus* that donor's own deviations from both — its baseline shift **and** how strongly the drug acts on it. This is what we do here. It keeps the pairing (control and treated cells from the same donor are linked through that donor's parameters), shares strength across donors, and — crucially — lets us *separate* the treatment effect we care about from the two kinds of donor heterogeneity (different baselines, different responses) we want to average over. Folding the response heterogeneity into the effect's uncertainty is what lets the estimate **generalize to a donor we have not yet seen**.

    The rest of the notebook builds option 3 and shows how to interrogate it.
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
    return Path, json, np, pd, pertpy, pickle, plt, sc, scribe, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading and filtering the data

    We load the dataset through `pertpy`, drop genes that are never observed, and remove low-quality cells with fewer than 1,000 UMIs. These are ordinary quality-control steps; nothing about normalization happens here, because in `scribe` normalization is part of the *model* (more on that below).
    """)
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
    mo.md(r"""
    We focus on the donors that received **panobinostat** and pull both their treated cells and their matched `control` cells into a single object, `adata_joint`. This is the crossed *donor × condition* table the model will see. The two grouping columns are:

    - `perturbation` — the **contrast of interest** (`control` vs `panobinostat`),
    - `sample` — the **donor**, which we want to account for but not test.
    """)
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
    mo.md(r"""
    Every donor appears in **both** columns — a fully crossed, balanced-ish design. Those 14 non-empty cells are the model's *leaves*. Reading **down a column** later will compare a donor against itself across the two conditions (the paired contrast); reading **across a row** compares donors within a condition (the heterogeneity we average over).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before modeling, the one plot we always look at: the distribution of total UMIs per cell. If library size varies by close to an order of magnitude, each cell is effectively sequenced at its own depth, and the model must be told so explicitly. In `scribe` that is the **per-cell capture probability** (`variable_capture=True`), the principled replacement for "divide by total counts".
    """)
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
    mo.md(r"""
    ## The model, written out

    ### Counts: a Negative Binomial in its *orthogonal* coordinates

    At its core `scribe` models each UMI count $u_{cg}$ (cell $c$, gene $g$) with a **Negative Binomial**, the distribution that already underlies DESeq2/edgeR and that drops out of a two-state promoter model of transcription. With `parameterization="mean_disp"` we sample the two coordinates the data actually speaks about directly: a gene **mean expression** $\mu_g$ and a gene **dispersion** (the NB *size*) $r_g$, with variance

    $$
    \sigma_g^2 \;=\; \mu_g \;+\; \frac{\mu_g^2}{r_g} .
    $$

    Read $\mu_g$ as *how much* a gene is expressed on average, and $r_g$ as *how bursty* that expression is — a small $r_g$ means a handful of cells carry huge counts while most sit near zero, a large $r_g$ means the counts cluster tightly around the mean. Why sample these two, rather than the negative binomial's raw "success probability" $p$? Because $(\mu, r)$ behave as **independent knobs**: learning one tells you nothing about the other. (Formally, their Fisher information is uncoupled, $\mathcal{I}_{\mu r}=0$; the technical name is *Fisher-orthogonal*.) In plain terms, the two dials do not fight each other, so the posterior has none of the curved, tangled uncertainty — the "banana" — that ties the raw $(r, p)$ pair together and makes it hard to fit. (The companion `_guide_reparam` note derives this orthogonality in full.) This choice is not cosmetic: as the differential-expression section will show, sampling $(\mu_g, r_g)$ directly is what lets the compositional machinery read each gene's compositional weight $\mu_g / r_g$ **straight off the sampled parameters** — and that faithful weight is exactly what the CLR contrast needs to stay honest.

    Because library sizes vary, each cell gets its own **capture probability** $\nu_c \in (0,1)$, and the count is drawn from a Negative Binomial whose **mean is thinned to $\nu_c\,\mu_g$** with the dispersion $r_g$ left unchanged. `scribe` evaluates this thinned likelihood **natively in the $(\mu, r)$ coordinates** — it never materializes $p$ — which also keeps the per-cell likelihood (the computational hot path) fast.

    So far this is the standard `scribe` NB-with-variable-capture model. The new part is what sits **on top of $\mu_g$**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mean expression: a *crossed* hierarchy with a per-donor slope

    Each leaf $\ell$ is a (condition, donor) pair. Instead of giving every leaf a free mean, we **decompose** the log-mean into a baseline plus three structured terms:

    $$
    \log \mu_g^{(\ell)}
    \;=\;
    \underbrace{\log \mu_g^{\mathrm{pop}}}_{\text{population baseline}}
    \;+\;
    \underbrace{\alpha_g\big[\mathrm{cond}(\ell)\big]}_{\text{treatment effect (fixed)}}
    \;+\;
    \underbrace{\beta_g\big[\mathrm{donor}(\ell)\big]}_{\text{donor baseline (random)}}
    \;+\;
    \underbrace{\gamma_g\big[\mathrm{cond}(\ell),\,\mathrm{donor}(\ell)\big]}_{\text{per-donor response (random slope)}} .
    $$

    Three terms on top of the baseline, three very different roles:

    - **$\alpha_g$ — the treatment effect — is a *fixed* effect**, with a **weakly-informative Gaussian** prior of *fixed* scale and **no learned shrinkage**: $\alpha_{g,c} = \sigma_\alpha\, z_{g,c}^{\alpha}$, $z\sim\mathcal N(0,1)$. (Why no adaptive shrinkage? A two-level factor has almost no information to estimate its own variance, and a learned scale would happily pull the contrast toward zero — exactly the effect we are trying to measure.) With the slope $\gamma_g$ in the model, $\alpha_g$ is no longer an effect every donor is *forced* to share: it is the **mean** of the per-donor responses — the shift we would expect in a typical donor — and the identified quantity is the contrast $\alpha_g[\text{panobinostat}] - \alpha_g[\text{control}]$.

    - **$\beta_g$ — the donor baseline — is a *random* effect** with a **regularized horseshoe** prior. With seven donors there *is* information to learn how much baselines vary, and the horseshoe adaptively shrinks: donors that look alike on a gene are pulled together, while a genuinely deviant donor is left alone. Each $\beta_g[d]$ is **zero-mean**, capturing deviations *from* the population baseline rather than competing with it.

    - **$\gamma_g$ — the per-donor response — is the *random slope*** (the treatment $\times$ donor interaction), also horseshoe-shrunk. It lets each donor deviate from the shared response $\alpha_g$: a donor that reacts more strongly, more weakly, or even in the opposite direction on gene $g$ is *represented* rather than averaged away. Because the horseshoe pulls these slopes toward zero, they cost nothing on the genes where donors respond alike, and switch on only where the data demand it.

    The population baseline $\log\mu_g^{\mathrm{pop}}$ is the only free intercept; the three structured terms are deviations around it, which is what makes the decomposition identifiable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why this is the right shape for the question

    Splitting $\log\mu_g^{(\ell)}$ into a fixed mean response, a random donor baseline, and a random per-donor slope is the formal statement of "estimate the average drug effect *that generalizes across donors*, while controlling for the donor." Two things make the slope $\gamma_g$ earn its keep:

    - **It guards against pseudoreplication.** Cells from one donor are not independent replicates of the *population* — they share that donor's idiosyncrasies. Without a slope, a donor that contributes many cells can pin the shared $\alpha_g$ to *its own* response, and the reported uncertainty (which counts cells) comes out far too tight. With the slope, the population effect is anchored to the **donor mean**, and its uncertainty reflects the **between-donor spread** $\tau_g^2$ — the variation that actually limits what we can say about a new donor.

    - **It is adaptive.** The horseshoe learns $\tau_g^2$ per gene from the data. Where donors respond alike, the slopes shrink to zero and the model behaves like a single shared effect; where donors genuinely diverge, the slopes open up and the population effect widens to admit it. We never have to decide up front how much donors agree — the model interpolates between "all donors identical" and "every donor its own," gene by gene.

    Only the **expression** target ($\mu$) carries this structure; the gene **dispersion** $r_g$ is a single per-gene value, shared across all leaves. That keeps the per-cell likelihood — the computational hot path — completely unchanged: the leaves are still an ordinary "dataset" axis, with the crossed structure (now including the slope) layered *above* them on $\mu$ alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting the model

    The call below is the whole model. `hierarchy=[...]` declares the two grouping factors and their effect types (a *fixed* contrast for the condition, a *random* effect for the donor); `interactions=[("perturbation", "sample")]` adds the per-donor treatment slope; and the unified `priors` dict picks the prior family per factor — a fixed-scale Gaussian for the treatment contrast, a horseshoe for the donor baseline, and a horseshoe for the per-donor slope. Crossing is implicit: listing two factors with no nesting means "donor crossed with condition."

    Fitting at this scale takes a while, so we **load a cached fit** when one is present and only run the optimizer otherwise.
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

    # Define parameterization
    _parameterization = "mean_disp"

    # Define output path. The "_mu_slope" tag marks the per-donor treatment
    # slope on the gene mean -- the perturbation x sample interaction added
    # below.
    _out_path = (
        data_dir
        / "scribe_results"
        / f"scribe_eda_{perturbation}_hierarchical_joint_mu_slope.pkl"
    )

    # Fit/load pre-fit model
    if _out_path.exists():
        with open(_out_path, "rb") as _f:
            results_joint = pickle.load(_f)
    else:
        results_joint = scribe.fit(
            adata_joint,
            parameterization=_parameterization,
            variable_capture=True,
            unconstrained=True,
            # exp link on the gene mean: SVI moves multiplicatively across the
            # decades of expression. (r carries no hierarchy, so it keeps the
            # softplus default -- hence the per-target dict, not a bare "exp".)
            positive_transform={"mean_expression": "exp"},
            # Donor (sample) CROSSED with condition (perturbation):
            hierarchy=[
                # 2-level contrast of interest -> fixed, weakly-informative.
                scribe.GroupLevel("perturbation", effect_type="fixed"),
                # 7 donors -> random effect with adaptive shrinkage.
                scribe.GroupLevel("sample"),
            ],
            # The per-donor treatment slope: a perturbation x sample interaction
            # on the gene mean. Interaction factors are random by default; the
            # horseshoe family below shrinks each donor's slope toward zero.
            interactions=[("perturbation", "sample")],
            # Per-factor prior families are declared through the unified
            # ``priors`` dict, keyed by the canonical target name. A
            # ``{factor: family}`` value attaches a per-factor hierarchy on that
            # target -- here the gene mean, ``mean_expression``. The effect
            # *type* (fixed vs random) comes from the ``GroupLevel``/interaction
            # above; the *family* comes from here. The interaction is keyed by
            # its ``":"``-joined operands.
            priors={
                "mean_expression": {
                    "perturbation": "gaussian",  # fixed-scale mean response
                    "sample": "horseshoe",  # donor baseline, shrunk across donors
                    "perturbation:sample": "horseshoe",  # per-donor slope -> 0
                },
            },
            # (mean_disp samples a single per-gene dispersion r_g, shared across
            #  all leaves; only the mean carries the crossed structure and the
            #  slope. To give the dispersion its own crossed structure, add a
            #  ``"dispersion": {"perturbation": ..., "sample": ...}`` entry.)
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
    mo.md(r"""
    ## Did the fit work? Three diagnostics

    ### 1. The ELBO loss

    Variational inference maximizes the **evidence lower bound (ELBO)**; the loss we plot is its negative. We do not read it quantitatively — we just want the textbook shape: a fast initial drop that flattens into a plateau. Spikes or upward drift would be warnings.
    """)
    return


@app.cell
def _(results_joint):
    _fig = results_joint.viz.plot_loss(figsize=(7, 3))
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Mean calibration — *per leaf*

    A converged loss does not prove the model describes the **counts**. The mean-calibration plot compares, for each gene, the **observed** mean count to the model's **predicted** mean; points on the diagonal mean the fit reproduces the first moment of the data.

    Because we have a crossed hierarchy, `scribe` lays the panels out on a **condition × donor grid**: rows are the two conditions, columns are the seven donors. This is the multi-factor layout doing its job — reading **down a column** shows control vs panobinostat *for one donor*, exactly the paired comparison the model encodes. A leaf that fell off the diagonal would point to a donor or condition the crossed structure is failing to capture.
    """)
    return


@app.cell
def _(adata_joint, results_joint):
    # Per-leaf observed-vs-predicted means. The panels auto-arrange as a
    # condition (rows) x donor (cols) grid from the model's grouping spec.
    _fig = results_joint.viz.plot_mean_calibration(counts=adata_joint)
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The fit calibrates tightly across the whole grid — nearly every leaf sits on the diagonal, in both conditions and for all seven donors. Reading **down a column** (one donor, control vs panobinostat), the model reproduces that donor's mean expression in *each* arm, which is exactly what the per-donor structure is built to do. Any residual stretch is confined to the **sparsest** leaves, where there are simply fewer cells to pin the mean — which the per-dataset check below makes concrete.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Posterior predictive check

    Finally, a **posterior predictive check (PPC)**: simulate fresh UMI counts from the fitted model and overlay their distribution on the real histograms, for a spread of genes across the expression range. If the generative story is adequate, the simulated bands should track the observed counts.

    Because this is a multi-dataset fit, each simulated cell is drawn from **its own leaf's** parameters, so the plot below is an honest aggregate over all fourteen donor × condition leaves. That makes it a good global check — but it cannot tell us whether any *individual* fit is good. We start global, then zoom in.
    """)
    return


@app.cell
def _(adata_joint, results_joint):
    _fig = results_joint.viz.plot_ppc(
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
    mo.md(r"""
    #### Per-dataset PPC: the quality of an *individual* fit

    The aggregate PPC hides per-leaf quality, so we zoom in. `plot_ppc(..., dataset=...)` swaps in a single leaf's parameter view (`results.get_dataset(leaf)` under the hood) and the **observed cells for that leaf only**, so we judge one donor × condition fit at a time. Address a leaf either by integer index or, more readably, by a `{factor: level}` dict. The axis worth probing is **data richness** — the structural reason fits differ from leaf to leaf.

    The two donors below sit at opposite ends of that scale: **`PW030`** contributes ≈19,000 control cells, while **`PW051`** contributes only ≈1,200. A data-rich leaf's predictive bands should hug the observed histograms; a sparse leaf's bands are wider and the fit is more stretched — the expected signature of thin data, not of a misspecified model.
    """)
    return


@app.cell
def _(adata_joint, results_joint):
    # A data-rich donor: its individual fit should track the data tightly.
    _fig = results_joint.viz.plot_ppc(
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
def _(adata_joint, results_joint):
    # A sparse donor the mean-calibration flagged: the same diagnostic, where
    # the individual fit is most stretched (≈1,200 cells vs ≈19,000 above).
    _fig = results_joint.viz.plot_ppc(
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
    mo.md(r"""
    ## Reading the model's structure

    Before any differential-expression machinery, the fitted hierarchy is already interpretable. The structured effects — the treatment contrast, the donor baseline, and the per-donor slope — are stored as parameters with posteriors, and `scribe` exposes them directly.

    ### The treatment effect

    `results.get_factor_effect("perturbation")` returns the fitted $\alpha_g$ effect. Since `perturbation` is the fixed contrast factor, the identified quantity is the **contrast** — the per-gene log-mean shift from control to panobinostat — which we summarize by its posterior mean.
    """)
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
    mo.md(r"""
    ### Donor heterogeneity — the two kinds, side by side

    The model separated **two** donor effects, and both are inspectable posteriors. Start with the **baseline**: `results.get_factor_effect("sample")` returns the random donor effects $\beta_g[d]$, a $7 \times G$ matrix of zero-mean log-mean deviations. Their spread across donors is the baseline heterogeneity the model absorbed so that it did *not* leak into the treatment effect.

    Below we compare, gene by gene, the **magnitude of the treatment effect** against this **baseline donor spread**. Genes where the baseline spread rivals or exceeds the treatment effect are ones that a pooled analysis — ignoring the donor label entirely — would get wrong.
    """)
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
    _ax.set_xlabel("baseline donor spread  (SD of $\\beta_g[d]$)")
    _ax.set_ylabel("|treatment effect|  $|\\bar\\alpha_g|$")
    _ax.set_title(f"{_donor.n_levels} donors: baseline spread vs treatment")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Baseline differences, though, are only half the story — and the simpler half. The reason the model carries a **slope** is the second kind of heterogeneity: donors that *respond* differently to the drug. The interaction $\gamma_g$ stores exactly that. A donor's response deviation is $\gamma_g[\text{panobinostat},d] - \gamma_g[\text{control},d]$ — how far donor $d$'s treatment effect departs from the shared $\alpha_g$ — and its between-donor spread (below) is the slope's report card. Where the cloud lifts off the floor, donors genuinely disagree about the drug's effect on that gene, and the population estimate must carry that disagreement in its uncertainty.

    The interaction is not a base factor, so it is read straight from the posterior site `mu_perturbation__sample_effect` (a `draws × 14 leaves × genes` array; the 14 leaves are `control × {7 donors}` then `panobinostat × {7 donors}`, in the same donor order as `get_factor_effect("sample")`).
    """)
    return


@app.cell
def _(np, plt, results_joint, treatment_effect):
    # The slope's payoff: how much donors differ in their RESPONSE, not just
    # their baseline. gamma is the interaction effect, a (draws, 14, G) site;
    # leaves 0..6 are control x donor, 7..13 are panobinostat x donor (same
    # donor order as get_factor_effect("sample")). A donor's response deviation
    # is gamma[pano, d] - gamma[control, d].
    _gamma = np.asarray(
        results_joint.get_posterior_samples()["mu_perturbation__sample_effect"]
    ).mean(0)  # posterior-mean, (14, G)
    _nd = _gamma.shape[0] // 2
    _resp_dev = _gamma[_nd:] - _gamma[:_nd]  # (n_donors, G) per-donor slope
    _resp_spread = _resp_dev.std(axis=0)  # per-gene between-donor response SD
    _tx_abs = np.abs(treatment_effect)

    # ...and the baseline spread again, to state the comparison numerically.
    _beta = np.asarray(results_joint.get_factor_effect("sample").effects())
    _base_spread = _beta.std(axis=0)
    print(
        "median response spread / baseline spread = "
        f"{np.median(_resp_spread) / np.median(_base_spread):.2f}"
    )
    print(
        "genes where response spread exceeds half the mean effect: "
        f"{np.mean(_resp_spread > 0.5 * _tx_abs):.0%}"
    )

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(_resp_spread, _tx_abs, s=4, alpha=0.3, color="indianred")
    _lim = float(np.nanmax([_resp_spread.max(), _tx_abs.max()])) * 1.05
    _ax.plot([0, _lim], [0, _lim], ls="--", c="k", lw=1)
    _ax.set_xlabel(
        "between-donor RESPONSE spread  "
        "(SD of $\\gamma_g[\\mathrm{pano},d]-\\gamma_g[\\mathrm{ctrl},d]$)"
    )
    _ax.set_ylabel("|treatment effect|  $|\\bar\\alpha_g|$")
    _ax.set_title("Per-donor response heterogeneity vs the mean effect")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The verdict is decisive — and it is the empirical case for the slope. Across genes the between-donor **response** spread runs about **1.4× the baseline spread**: donors disagree *more* about how they respond to panobinostat than about where they start. And on the large majority of genes that response spread is an appreciable fraction of the mean effect itself. This is exactly the variation a single shared treatment effect would have swept under the rug — quietly crediting one or two donors' strong response to the whole population, with a falsely tight interval. The slope keeps it visible and, as the differential-expression section will show, folds it into the uncertainty of every call.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Recovering the paired groups

    `scribe` also lets us pull back the leaf structure for any downstream slicing.  `results.get_group(sample="...")` returns the leaves for one donor, indexed by condition; `iter_groups("sample")` walks every donor. Each value is an ordinary single-dataset results view, so you can drop it into any per-leaf analysis.
    """)
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
    mo.md(r"""
    ## Differential expression: the donor-averaged treatment effect

    ### First: sequencing measures *proportions*, not amounts

    Before any contrast, one fact about the data shapes everything that follows. A sequencer does not report how many molecules a cell contained — it reports a fixed-size *sample* of them. Double a cell's true RNA content and, after library prep and sequencing to a fixed depth, you get back roughly the *same* total counts, just redistributed. So what single-cell counts actually carry is **composition**: the *fractions* of the library each gene claims, which are forced to sum to one.

    That constraint has a sharp consequence for differential expression. If one gene genuinely shoots up, it claims a larger slice of a fixed pie, so *every other gene's fraction drops* — even genes whose absolute expression never budged. Comparing raw fractions therefore manufactures fake "down" calls. This is the well-known **compositional bias** of sequencing data, and it is why you cannot read a treatment effect straight off proportions.

    The standard fix from compositional data analysis is the **centered-log-ratio (CLR)** transform. Instead of a gene's fraction, it uses the *log-ratio of that fraction to the geometric mean of all genes' fractions* — each gene measured against the "typical" gene rather than against any hand-picked reference. Dividing by that whole-composition anchor is what makes CLR **reference-free** (no assumed-unchanged housekeeping set), and it turns the multiplicative, sum-to-one distortions above into ordinary *additive* shifts you can add, subtract, and average. The catch is that CLR speaks in **relative** terms: a positive CLR change means "this gene's share of the pie grew *relative to the typical gene*," not "this gene's absolute expression grew." Keeping both readings — relative (CLR) and absolute (biological) — straight is exactly what the two views later in this section are for.

    ### The estimand

    We want the population treatment effect, *paired within donor*, and — following the section above — we measure it in **CLR space**. For each donor $d$ present in both arms and each posterior draw $s$, we form the within-donor CLR difference

    $$
    \Delta_g^{(d,s)}
    = \mathrm{clr}\!\big(\rho_g^{(\text{control},\,d,\,s)}\big)
    - \mathrm{clr}\!\big(\rho_g^{(\text{panobinostat},\,d,\,s)}\big),
    $$

    and then **average over donors** to get the paired main effect

    $$
    \bar\Delta_g^{(s)} = \sum_d w_d\, \Delta_g^{(d,s)},
    \qquad w_d = \tfrac{1}{D},
    $$

    with **uniform** donor weights — each donor counts once, not once per cell, so the average is a *donor*-level mean rather than a cell-weighted pool. Inside each within-donor difference the donor **baseline** $\beta_g[d]$ cancels (control and treated cells of donor $d$ share it). What does *not* cancel is the donor's **own response**: the slope $\gamma_g[\cdot,d]$ is condition-specific, so each $\Delta_g^{(d,s)}$ carries donor $d$'s *actual* treatment response — the shared mean *plus* its personal deviation. Averaging over donors then returns the population effect, and because every draw $s$ resamples the per-donor slopes, the spread of $\bar\Delta_g^{(s)}$ across draws now folds in the **between-donor variability**. That is the slope paying off at the estimand: the reported uncertainty is uncertainty about a *new* donor, not the artificially tight interval one gets by treating all cells as exchangeable. The whole population contrast is one call:
    """)
    return


@app.cell
def _(results_joint, scribe):
    # Paired main effect across donors. delta = CLR(control) - CLR(panobinostat),
    # so a gene UP in panobinostat has a NEGATIVE delta.
    # n_samples sets N in delta_samples (more draws -> smoother lfsr/PEFP).
    # batch_size chunks the draw and offloads it to host RAM (automatic for
    # n_samples > 500), keeping GPU memory free for the composition sampling.
    # No gene_mask is needed: the fit's own gene_coverage=0.99 already pooled the
    # un-modeled tail into the `_other` pseudo-gene, so the composition closes
    # over the whole transcriptome and the CLR reference is stable as-is.
    results_de = scribe.compare_groups(
        results_joint,
        "perturbation",
        "control",
        "panobinostat",
        n_samples=3_000,
        batch_size=500,
        convert_to_numpy=True,
    )
    results_de
    return (results_de,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A cautionary detour: CLR needs a stable reference

    CLR is principled, but it has a sharp edge. The reference it divides by is the **geometric mean of the log-fractions across *all* genes** — and a typical single-cell matrix has thousands of genes sitting at essentially zero, whose log-fractions are hugely negative and can jitter from draw to draw. If that jitter contaminated the reference, it would inflate the CLR contrast of the genes we actually care about.

    So look at the top "hits" from the unfiltered run (table below). They are the usual low-mass suspects — pseudogenes and tissue-restricted markers expressed in almost no cells here — exactly the genes that carry no real signal. And notice **two** things about their numbers. The shifts are *modest* (the largest is only ≈5 log2 units), and every one carries a **high lfsr** (the *local false sign rate* — the posterior probability that we have the effect's direction wrong, defined in full at the volcano below; here roughly 0.15–0.5): the model is openly unsure of their sign. This is the orthogonal $(\mu, r)$ parameterization paying off upstream — because each gene's compositional weight $\mu_g/r_g$ is read faithfully from the sampled parameters, the geometric-mean reference stays stable even with the low-mass tail in the mix, and these genes *wobble* rather than *explode*.

    And the one pooling step that genuinely matters already happened at **fit** time. There is a convenient property of compositions here — the **Dirichlet closure property**: lumping several genes into one category and keeping their *combined* fraction loses no information about the composition (the math works out exactly). So when we passed `gene_coverage=0.99` to the fit, `scribe` folded the rarely-seen tail of genes into a single `_other` pseudo-gene, and the fitted composition already closes over the whole transcriptome. The DE samples that same simplex, `_other` included, as the stable compositional anchor. There is therefore **no need for a second, DE-time masking pass**: we read this contrast directly.
    """)
    return


@app.cell
def _(np, results_de):
    _df_raw = results_de.to_dataframe(
        tau=np.log(1.1),
        target_pefp=0.05,
        metrics="clr",
        column_naming="prefixed",
    )
    # Most extreme |CLR shift| -- the low-mass, sign-uncertain genes.
    _df_raw.reindex(
        _df_raw["clr_delta_mean"].abs().sort_values(ascending=False).index
    )[["gene", "clr_delta_mean", "clr_lfsr"]].head(12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The compositional view

    With a stable reference, the standard `scribe` DE plots are well-defined — but first, the two numbers every call rests on, because they replace the frequentist vocabulary you may be used to.

    **`lfsr` — the local false sign rate.** A $p$-value asks "how surprising would this data be *if the gene didn't change at all*?" A generative model can ask something more directly useful: *given everything we've learned, how sure are we which way the gene moved?* The lfsr is the posterior probability that we have the **sign wrong**. An lfsr near $0$ means "almost certainly up" (or almost certainly down); an lfsr near $0.5$ means "a coin flip — the model can't tell you the direction." It needs no null hypothesis, and it fails gracefully: a gene the model is unsure about simply earns a high lfsr, instead of a misleadingly tiny $p$-value.

    **`tau` and the target false-sign rate.** A gene is *called* differentially expressed only when it clears two bars at once — an effect-size floor $\tau$ (so we ignore vanishingly small shifts) and a confidence bar: we admit genes, most-confident first, until the *average* lfsr of the called set reaches a tolerance we set in advance (the "proportion of expected false signs," PEFP). It is the Bayesian analogue of an FDR cutoff, stated in terms of *direction* rather than existence.

    The **volcano** plot then places the CLR contrast on the $x$-axis and confidence ($-\log_{10}\text{lfsr}$) on the $y$-axis, highlighting the genes that clear both bars; the companion panel shows each gene's CLR mean in the two arms.

    A note worth internalizing — and the real result on this dataset. Under a **broad** perturbation like an HDAC inhibitor *many* genes move at once, so the whole composition shifts together — and when everything moves, the *relative* change of any single gene is genuinely hard to sign. With the per-donor slope in the model, that difficulty is compounded by honesty: every CLR interval is widened to reflect how differently donors respond, so a gene must clear a **donor-generalizable** bar, not a within-this-sample one. The consequence is striking — at a strict error threshold the CLR volcano highlights **not one gene**. Nothing clears the bar: not the canonical program, and not even the compositional outliers (the mitochondrial transcripts and pseudogenes whose *proportion* swings hardest). Asked to sign a *relative* change that generalizes across donors under a genome-wide shift, the composition alone declines to commit to any single gene.

    This is CLR behaving exactly as designed, taken to its logical end: it is the **conservative** lens, and once the bar is donor-generalizable there is no *relative* change it will commit to. But — and this is the crucial point, which the scatter at the end of this section makes quantitative — that conservatism is **only about confidence, not direction**. Gene by gene, the CLR contrast still tracks the biological fold-change closely (Spearman around 0.9): the same sign throughout, and a similar magnitude. CLR is **under-confident here, not incoherent.** And since it makes no confident calls of its own, the natural robustness check is to drop the known technical confounders — mitochondrial, ribosomal, and hemoglobin genes — by *name*, and confirm nothing of substance was hiding in the low-mass tail.
    """)
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
    mo.md(r"""
    ### Cleaning up by identity — a robustness check

    Mitochondrial, ribosomal, and hemoglobin genes are *known technical confounders* in single-cell data, so a natural robustness check is to drop them by **name** — a decision made from gene *identity*, a priori and **independent of the contrast**. (The tempting alternative — thresholding on the large CLR values themselves and re-running — is *circular*: it selects genes by the very effect we are testing, defining away real signal along with artifacts. Filtering on identity, or on expression/precision, avoids that.) `exclude_gene_name_mask` builds a keep-mask from name prefixes and regexes; we pass it as the `gene_mask` and re-run the same CLR contrast — checking whether any confident compositional call was leaning on a confounder.
    """)
    return


@app.cell
def _(results_de, results_joint, scribe):
    # Drop mitochondrial (incl. MT-pseudogenes), ribosomal, and hemoglobin
    # genes by name — known confounders, chosen by identity, not by effect size.
    _nuisance_keep = results_de.exclude_gene_name_mask(
        prefixes=("MT-", "RPL", "RPS", "MRPL", "MRPS"),
        patterns=(r"^MT(RNR|CO\d|ND\d|ATP\d|CYB)", r"^HB[ABDEGMQZ]\d?$"),
    )
    print(
        f"dropping {int((~_nuisance_keep).sum())} mito/ribo/HB genes; "
        f"keeping {int(_nuisance_keep.sum())} of {_nuisance_keep.size:,}"
    )
    results_de_clean = scribe.compare_groups(
        results_joint,
        "perturbation",
        "control",
        "panobinostat",
        gene_mask=_nuisance_keep,
        n_samples=3_000,
        batch_size=500,
        # Offload the result to host RAM as it finishes so its sample arrays do
        # not stay resident on the GPU and accumulate across cells.
        convert_to_numpy=True,
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
    mo.md(r"""
    The result confirms the reading above. There was no confident CLR hit list to begin with — and after dropping the mitochondrial, ribosomal, and hemoglobin genes there still is none: the count stays at **zero**. The low-mass genes that topped the *unfiltered* shift ranking were never confident, and removing the classic confounders changes nothing — no real signal was hiding among them. The number that matters holds: across the ~17,000 surviving genes the cleaned CLR contrast still correlates with the biological log-fold-change at **Spearman around 0.9**. So name-based filtering does its job as non-circular hygiene, and makes the real point unmistakable — under this broad perturbation, asked to generalize across donors, there is no coherent program hiding *in the composition* behind a few loud genes. CLR and the biological mean are measuring the **same** response; they differ only in how confidently each is willing to commit to it. We read the biological view — the confident one — next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The biological view, and the panobinostat signature

    We now have one lens; here is the other, and the contrast between the two is the conceptual heart of this section.

    **Two questions, two views.** Recall that CLR speaks in *relative* terms — shares of a fixed pie. The **biological log-fold-change** (`mode="bio"`) speaks in *absolute* terms: the change in each gene's underlying mean expression $\mu_g$, with its own lfsr. The pie analogy makes the difference concrete. Suppose the drug switches a few genes on hard; they claim more of the pie, so *every other gene's slice shrinks* even if its absolute expression never moved. CLR will read those bystander genes as mildly "down" (their *share* really did fall), while the biological view reads them as flat (their *amount* did not change). Neither lens is wrong — they answer different questions. CLR asks *"did this gene's share of the composition shift?"* — the right guard against compositional artifacts. The biological view asks *"did this gene's expression actually change?"* — the directly interpretable question for a drug response.

    Crucially, the biological view never has to sign a *relative* change against a moving composition, so the canonical markers that carried a high *CLR* lfsr come back here with **biological lfsr ≈ 0** — the posterior is certain of their direction. For a broad perturbation like this one, the biological view is the **primary** readout and CLR is the compositional guardrail; the textbook signature lives here.
    """)
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
    mo.md(r"""
    Let's read the actual hit list. We export the full table and orient the biological log-fold-change as **panobinostat vs control** (positive = up after treatment). One honest caveat: a fold-change ranking rewards the largest *relative* swings, and for the least-expressed retained genes those are the noisiest. So we read the table with **mean expression in view** and focus the headline on **well-expressed** hits (mean $\geq 5$ UMIs in an arm), then validate against known markers below.
    """)
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
    mo.md(r"""
    ### Validating against known markers

    The cleanest sanity check is a targeted one: do the **canonical HDAC-inhibitor / panobinostat markers** come back with the sign biology predicts, and with the posterior confident about it? We pull a curated panel straight from the DE table.
    """)
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
    mo.md(r"""
    ### Does it make biological sense?

    The dominant program returns exactly as textbook panobinostat biology predicts — and, just as important, the slope makes the model **honest about the parts that do not generalize**. Read the well-expressed hit list and the marker panel together.

    **Up after treatment — the donor-robust core.**

    - **Metallothioneins** (`MT1G`, `MT1H`, `MT1X`, `MT1E`, and `MT2A`: mean ≈53 → 113 UMIs) — a hallmark metal/oxidative-stress response, the dominant high-confidence block, every member at lfsr ≈ 0.
    - **Linker histones** (`H1F0`, `HIST1H1C`) — the chromatin-remodeling footprint of HDAC inhibition, confidently up.
    - **Acute-phase and integrated-stress genes** (`KNG1`, `PI3`, `IER3`, the stress lncRNA `NEAT1`, and the modest but confident `GADD45A`/`DDIT3`).

    **Down after treatment — also donor-robust.**

    - **`MYC`** and the cyclin **`CCND1`** — proliferation shutting down, the expected cytostatic effect.
    - **`MDM2`** — the p53 E3 ligase falling, consistent with **p53 activation**.
    - A coherent **myeloid / macrophage program** (`SPP1`, `APOC1`, `CD163`, `CD14`, `MSR1`, `FCGR3A`, `FCER1G`) — the drug suppressing the monocyte–macrophage compartment.
    - A block of **mitochondrially-encoded transcripts** (`MT-RNR2`, `MT-CO1`, `MT-ND4`, `MT-ND5`) — the mitochondrial fraction dropping as that compartment is suppressed. Read cleanly as **biology** here, with confident sign — not the compositional confounder they would look like if we trusted proportions alone.

    **What the donor-generalizable view will *not* commit to.** The acute heat-shock chaperones (`HSPA1A`, `HSPB1`) and the cell-cycle brake `CDKN1A`/p21 — markers a single-donor or pooled analysis often reports up — come back here **flat or sign-uncertain** (`CDKN1A`: lfsr ≈ 0.2; `HSPA1A`, if anything, modestly *down*). This is not a failure; it is the random slope doing its job. These are exactly the genes where donors disagree most about the response, so once the population effect is anchored to the donor *mean* with the between-donor spread folded into its uncertainty, their shared signal no longer clears the bar. A model without the slope would have reported them as confident hits on the strength of one or two strong responders.

    That is the payoff of the whole construction: the effect is **the one we would expect in a new donor** — the donor-robust metallothionein/chromatin-up, myeloid/p53-down program stated with confidence, and the donor-variable stress markers held honestly at arm's length.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A shortcut: the population main effect, straight from $\alpha$

    Everything above used `estimand="paired_main_effect"` — it samples each donor's composition and contrasts them in CLR space, so it carries every donor's *own* response (the slope $\gamma$) and averages them. But the hierarchy also exposes the shared term $\alpha$ directly, and $\alpha[\text{panobinostat}] - \alpha[\text{control}]$ *is* the population main effect — the donor **mean** response — readable with no composition sampling at all.

    `scribe.compare_groups(..., estimand="effect")` reads exactly that: it pulls the $\alpha$ contrast from the hierarchy (`get_factor_effect` under the hood), so it is essentially **instant and memory-light** — the practical fast path to the population effect. The returned `delta` is a **log-mean** effect (not a CLR contrast) with its own lfsr; the `_other` pseudo-gene is dropped automatically. With the random slope in the model this is *close to but no longer identical to* the paired biological log-fold-change: the paired pass additionally folds in each donor's deviation $\gamma$, so the two agree **strongly** (correlation around 0.93 below) rather than exactly. The gap between them is precisely the per-donor response heterogeneity — the same quantity the slope plot made visible earlier.
    """)
    return


@app.cell
def _(np, results_de, results_joint, scribe):
    # The treatment effect, read directly from the fitted alpha — no composition
    # sampling. Reuses the already-drawn posterior, so this is near-instant.
    results_effect = scribe.compare_groups(
        results_joint,
        "perturbation",
        "control",
        "panobinostat",
        estimand="effect",
        n_samples=3_000,
        # Offload to host RAM (keep the GPU clear for later cells).
        convert_to_numpy=True,
    )

    # The donor-mean main effect alpha (clr_* columns here hold the log-mean
    # effect, not a CLR contrast) vs the paired biological log-fold-change. With
    # the slope they are close but not identical -- the gap is the heterogeneity:
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
    # log2FC = up in panobinostat) — the donor-mean main effect for each.
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
    mo.md(r"""
    ### CLR and the biological view agree — conservative, not incoherent

    We have treated the compositional (CLR) and biological views as answering different questions — and they do — but it is worth asking directly: do they *disagree*? They do not. Plotting every gene's CLR contrast against its biological log-fold-change (left) traces a clear diagonal — **Spearman around 0.9** — with the canonical markers in the expected quadrants (metallothioneins / histones up, the myeloid, p53, and mitochondrial programs down). The two lenses agree on **sign** throughout and broadly on **magnitude**. The visible residual is a *compression* at the down end: the CLR contrast reaches only about $-3$ log2 where the biological fold-change reaches $-6$, because closure pulls the largest losers toward the reference as the high-mass winners claim more of the simplex — and because the donor-generalizable CLR intervals shrink each gene's signed estimate toward zero. The diagonal is looser than a homogeneous model's would be, and that extra scatter is not noise: it is the between-donor response uncertainty, now honestly present in every CLR estimate.
    """)
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
    mo.md(r"""
    What separates the two views is **confidence, not direction.** CLR assigns every gene a higher lfsr — under a genome-wide shift, made donor-generalizable by the slope, the *relative* change is intrinsically hard to sign — so at a strict PEFP **nothing** clears the bar. But that is a property of the *threshold*, not the signal: relax the tolerated false proportion (right panel) and the canonical signature emerges from CLR a few genes at a time — only past a PEFP of about **0.4** here, but emerge it does — and **every single CLR call agrees with biology in sign** (the dashed line sits flat at 1.0 across the whole sweep). CLR is slow to *commit*, never *wrong* about direction.

    And CLR is the more conservative lens for a principled reason — it **assumes less and accounts for more.** It is provably invariant to capture-probability misspecification (the biological mean is *not* — it must trust the capture / library model); it references each gene against the whole composition; and because we sample a gene-specific dispersion $r_g$, it carries *shape* / over-dispersion uncertainty into its lfsr — a term that simply cancels in the biological log-fold-change. Its wider intervals are **earned honesty** about what compositional data alone can claim — not noise.

    The scientific reading, then, is not "believe whichever lens clears a $p$-value-like cutoff." It is that a **robust, assumption-light estimator (CLR) and a confident, model-dependent one (the biological mean) converge** on the same panobinostat signature — here, strongly (Spearman around 0.9), in sign everywhere. That convergence — read with knowledge of the marker biology — is what earns confidence in the result, far more than any single PEFP threshold could. And the residual gap between the lenses is not noise: it is the honest width the per-donor slope adds, so that what survives is the response we would expect in a *new* donor. The generative model hands us *both* lenses from one posterior; the scientific judgment is ours to bring.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    We built **one** joint generative model of a crossed *donor × condition* experiment and, from a single posterior, read off:

    1. a **fit** that calibrates per leaf and passes posterior predictive checks;
    2. the **fitted structure** — a fixed treatment effect and the random donor deviations it was protected from, both as inspectable posteriors;
    3. a **donor-generalizable, paired differential-expression** result whose **biological** log-fold-changes reproduce the donor-robust panobinostat / HDAC-inhibitor signature (metallothioneins and histones up, the myeloid and p53 programs down) with high sign-confidence — while honestly declining to commit to the donor-variable stress markers — and the population main effect $\alpha$ available essentially for free (`estimand="effect"`).

    Three lessons worth carrying forward. First, **the parameterization is what makes the composition trustworthy.** Sampling the orthogonal $(\mu, r)$ pair directly — rather than a derived success probability — gives each gene's compositional weight $\mu_g/r_g$ faithfully, so the CLR reference is stable on the raw transcriptome and the contrast is well-behaved with no post-hoc surgery (the fit's `gene_coverage=0.99` `_other` pool is the only pooling needed). Second, **CLR and the biological view answer different questions, and which one is the *confident* headline depends on the perturbation.** CLR asks "did the *relative* composition shift?" — invaluable as a guard against compositional artifacts, but under a broad, genome-wide response made *donor-generalizable* it confidently signs **nothing**: the relative change of any single gene, required to hold across donors, is simply not identifiable here. The biological log-fold-change asks "did the *mean expression* change?" — and that is where the donor-robust signature emerges with high confidence. Third — and this is the subtle one — **conservative is not the same as wrong.** CLR is *under-confident here, not incoherent*: across the transcriptome it agrees with the biological view in **direction** everywhere and broadly in **magnitude** (Spearman around 0.9), and it recovers the signature as the error tolerance relaxes. It is conservative precisely because it assumes less (capture-invariant) and accounts for more (global compositional structure, the gene-specific dispersion $r_g$, and now the between-donor response spread). The thread through all three is the **slope**: by anchoring the population effect to the donor *mean* and folding the between-donor variability into every interval, it makes the whole analysis report the effect we would expect in a **new** donor — the textbook program where donors agree, and honest uncertainty where they do not.
    """)
    return


if __name__ == "__main__":
    app.run()
