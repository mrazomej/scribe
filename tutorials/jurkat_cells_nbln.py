import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Principled gene–gene correlations with NBLN + SVI cascade + sparse loadings prior

    *This tutorial is a sequel to* [Modeling Assumptions for Single-Cell RNA-seq with `scribe`](jurkat_cells.md). *It assumes you have already worked through that one (or are comfortable with negative binomial counts, variable capture, parameterizations, and low-rank variational guides) and want to ask a sharper question: when the model says two genes covary, what exactly is the model saying — and how do we keep it honest?*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The question we want to answer

    In the previous tutorial we ended at a low-rank-Gaussian variational guide over $(\underline{\mu}, \phi)$ and looked at the resulting correlation heatmap. That heatmap is *useful* — it shows which gene-level parameters are coupled in the variational posterior — but it is **not** a correlation between gene expressions. It is a correlation between *parameters of a model whose generative story emits independent counts gene by gene* given the per-cell capture and the shared $\phi$. Two genes look correlated there because their NB parameters are pulled by the same data, not because the model has any built-in mechanism for one gene's expression to track another.

    For the question *"which genes covary biologically?"* we need a generative model that lets gene expression vary jointly across cells. That is the move we make in this tutorial: from NB-with-shared-$p$ models (NBDM / NBVCP) to the **Negative-Binomial LogNormal** (NBLN), where the per-cell rates live in a correlated log-normal space:

    $$
    \underline{x}_c \;\sim\; \mathcal{N}\!\left(\underline{\mu},\; \underline{\underline{W}}\,\underline{\underline{W}}^{\!\top} + \mathrm{diag}(\underline{d})\right),
    \qquad
    u_{c,g} \mid \underline{x}_c \;\sim\; \mathrm{NB}\!\left(r_g,\; \mathrm{logit}^{-1}\!(x_{c,g} - \eta_c)\right).
    $$

    Now gene-gene correlation is a **property of the model itself** — encoded in the loadings matrix $\underline{\underline{W}} \in \mathbb{R}^{G\times k}$ — not a byproduct of a richer variational guide. Asking "do these two genes covary?" becomes asking "what does $\underline{\underline{W}}\underline{\underline{W}}^{\!\top}$ say?", which has a clean answer.

    But NBLN is not a free lunch. It introduces a per-cell *rigid-translation gauge* that creates a numerical and conceptual hazard, and the loadings matrix $\underline{\underline{W}}$ has many more columns than the data can support — so without care, the "principled" correlations we extract will be polluted by gauge slop and noise. The rest of this tutorial walks through how `scribe` deals with both problems using three composable pieces:

    1. **SVI cascade** — anchor the NBLN Laplace fit on a converged NBVCP-SVI fit.
    2. **Cascade freeze** — pin the gauge-vulnerable parameters at their SVI values.
    3. **Loadings shrinkage** — let a sparse prior on $\underline{\underline{W}}$ pick the effective rank from data.

    The payoff is a correlation structure on genes that you can defend as biologically meaningful rather than as an artifact of how the optimizer landed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why NBLN needs a per-cell latent layer (and NBDM doesn't)

    A practical question often comes up at this point: *the previous tutorial worked entirely with NBDM / NBVCP models, where each cell's biology was just a vector of NB draws — no extra "latents per cell" to fit. Why does NBLN suddenly need one?* The answer is the most important conceptual difference between the two model families, and it explains a lot of NBLN's complexity. Let us take it slowly.

    ### NBDM: the per-cell composition integrates out over the simplex in closed form

    In NBDM / NBVCP, the per-cell generative process is grounded in the Poisson-Gamma-Dirichlet hierarchy. As shown in the paper, when all genes share the same success probability $p$, the joint distribution of independent negative binomial UMI counts admits a two-level decomposition:

    $$
    \pi(\underline{u}_c \mid \underline{r}, p) =
    \int_0^{\infty} d\Lambda \,
    \overbrace{
    \pi(u_{T,c} \mid \Lambda)
    }^{\text{Poisson}}
    \times
    \underbrace{
    \pi(\Lambda \mid r_T, \theta)
    }_{\text{Gamma}}
    \int d^{G-1} \underline{\rho} \,
    \overbrace{
    \pi(\underline{\rho} \mid \underline{r})
    }^{\text{Dirichlet}}
    \times
    \underbrace{
    \pi(\underline{u}_c \mid u_{T,c}, \underline{\rho})
    }_{\text{Multinomial}},
    $$

    where $\underline{\rho} \in \Delta^{G-1}$ is the per-cell gene-expression composition on the $(G-1)$-simplex (the space of "fractions of the transcriptome occupied by each gene"), $u_{T,c} = \sum_g u_{c,g}$ is the total UMI count, $r_T = \sum_g r_g$, and $\theta = p/(1-p)$.

    In principle, each cell "samples" a value for the two continuous latent quantities: a total rate $\Lambda_c$ and a composition $\underline{\rho}_c$. However, part of the beauty of the NBDM problem is that **both integrals collapse to closed forms**, freeing us from having to determine on a per-cell basis the value of either of these variables. The inner simplex integral (Dirichlet $\times$ Multinomial) evaluates to the Dirichlet-Multinomial ($\mathrm{DM}$) distribution, and the outer total integral (Gamma $\times$ Poisson) evaluates to the Negative Binomial ($\mathrm{NB}$). The result is

    $$
    \pi(\underline{u}_c \mid \underline{r}, p) =
    \mathrm{NB}(u_{T,c} \mid r_T, p) \times \mathrm{DM}(\underline{u}_c \mid u_{T,c}, \underline{r}).
    $$

    Neither $\Lambda_c$ nor $\underline{\rho}_c$ survives as a leftover latent to be fitted or approximated per cell — the model is fully tractable, with no per-cell numerical work required.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### NBLN: the same integral has no closed form

    NBLN replaces "independent NB per gene with a shared $p$" with a *correlated* log-rate vector per cell:

    $$
    \underline{x}_c \;\sim\; \mathcal{N}(\underline{\mu},\; \underline{\underline{W}}\underline{\underline{W}}^{\!\top} + \mathrm{diag}(\underline{d})),
    \qquad
    u_{c,g} \mid \underline{x}_c \;\sim\; \mathrm{NB}\!\left(r_g,\; \mathrm{logit}^{-1}\!(x_{c,g} - \eta_c)\right).
    $$

    In other words, instead of having a globally-shared $p$ parameter, the success probability becomes gene- *and* cell-specific:

    $$
    p_{c,g} = \mathrm{logit}^{-1}(x_{c,g} - \eta_c) = \frac{1}{1 + e^{-(x_{c,g} - \eta_c)}},
    $$

    where $\eta_c$ is a per-cell offset that absorbs total expression (another reparameterization of the UMI capture probability that models how efficiently each cell was sequenced), and $x_{c,g}$ is the coordinate of the shared Gaussian latent for gene $g$ in cell $c$.  Every gene in every cell lands at a different point in $[0,1]$, driven by where $\underline{x}_c$ happens to fall.

    This single change destroys the algebraic glue that made NBDM tractable. The closed-form Dirichlet-Multinomial structure we described above relied on *all genes sharing the same $p$* — that is precisely the condition under which the composition $\underline{\rho}$ is Dirichlet and the simplex integral closes. Once $p$ becomes gene-specific through $x_{c,g}$, the genes no longer share a common "currency", the composition is no longer Dirichlet, and the per-cell marginal requires the $G$-dimensional integral that follows.

    To get the cell-level marginal we would need

    $$
    p(\underline{u}_c \mid \underline{\mu}, \underline{\underline{W}}, \underline{d}, \underline{r}, \eta_c)
    \;=\;
    \int_{\mathbb{R}^G}
    \mathrm{d}\underline{x}_c\;
    \underbrace{\mathcal{N}\!\left(\underline{x}_c\mid \underline{\mu},\, \underline{\underline{W}}\underline{\underline{W}}^{\!\top} + \mathrm{diag}(\underline{d})\right)}_{\text{Gaussian prior on log-rates}}\;
    \underbrace{\prod_{g=1}^G \mathrm{NB}\!\left(u_{c,g} \,;\, r_g,\, \mathrm{logit}^{-1}(x_{c,g} - \eta_c)\right)}_{\text{NB likelihood per gene}}.
    $$

    This integral has **no closed form**. The Gaussian wants to be conjugate to a Gaussian likelihood; the NB likelihood, parameterized through a logit transform of the latent, is not Gaussian. There is no magic cancellation that removes $\underline{x}_c$. We are stuck with a $G$-dimensional integral that has to be approximated ***for every cell***.

    ### What "Laplace inference" actually does about it

    `scribe`'s NBLN-Laplace path handles each cell's integral by approximating the integrand around its mode:

    1. Find the [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) $\underline{x}_c^\ast$ — the value of $\underline{x}_c$ that maximizes the integrand via Newton iteration per cell.
    2. Compute the Hessian of the negative log-integrand at that mode (curvature of the local fit).
    3. Approximate the integrand as a Gaussian centered at $\underline{x}_c^\ast$ with covariance equal to the inverse Hessian.

    Under that local-Gaussian approximation the $G$-dimensional integral collapses into a `log-det` term: a tractable scalar that the optimizer can differentiate.

    ### The outer loop: it is an EM algorithm

    The per-cell Laplace step above is only the inner half of the picture. NBLN-Laplace alternates between **two** optimization problems, in the spirit of the classical [Expectation-Maximization (EM)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) algorithm — fix one block of unknowns, solve the other, swap, repeat:

    - **E-step ("expectation"):** with the global parameters $\theta = (\underline{\mu}, \underline{\underline{W}}, \underline{d}, \ldots)$ held fixed, find each cell's per-cell latent MAP $\underline{x}_c^\ast$ (and the Hessian needed for the `log-det` term) via the Newton iteration described above. `laplace_config` (`n_newton_steps`, `damping`, `newton_max_step`, …) controls this inner step.
    - **M-step ("maximization"):** with the per-cell MAPs held fixed, take optimizer steps on the global parameters $\theta$ using the Laplace-approximated ELBO as the objective — the same kind of stochastic-gradient loop as a regular SVI fit. `n_steps`, `optimizer_config`, and `early_stopping` control this outer step.

    The two halves take turns: refresh the per-cell MAPs, run a chunk of M-step gradient updates, refresh the MAPs again, run more M-step updates, and so on until the loss envelope settles.

    The reason this split is necessary goes back to the identifiability story above. The per-cell latents $\underline{x}_c$ are "annoying" — there are $C \times G$ of them, they have no closed-form posterior, and they tangle non-linearly with the global parameters through the NB-logit. Holding them fixed at their current MAPs lets the M-step look like a familiar SVI/Adam loop on a few global parameters. Holding the global parameters fixed lets each cell's E-step look like an isolated, well-conditioned Newton problem. Doing both at once is intractable; alternating between them is what makes NBLN-Laplace tick.

    This is why NBLN-Laplace explicitly **owns a per-cell $\underline{x}_c$ MAP** as a stored field on the result (`x_loc`) and why the Newton iterations show up in the loss accounting. It is not a quirk of the implementation — it is the inevitable cost of moving from a model whose per-cell integral is closed-form (NBDM) to one whose per-cell integral is not (NBLN). The reward is the joint correlation structure encoded in $\underline{\underline{W}}\underline{\underline{W}}^{\!\top}$, which is the only thing NBDM cannot give us no matter how clever the variational guide is.

    Keep this picture in mind throughout the rest of the tutorial: every time you see a "per-cell Newton" or a "latent $\underline{x}_c$", that is the price of replacing "independent NB per gene" with "correlated log-rates across genes".
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

    # Set our plotting style (totally optional)
    scribe.viz.matplotlib_style()
    return Path, clear_caches, gc, np, pickle, plt, scribe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use the same [Jurkat 10x dataset](https://www.10xgenomics.com/datasets/jurkat-cells-1-standard-1-1-0) as the first tutorial — a monoculture of ~3,200 cells. Because there is only one cell type, any "correlation structure" the NBLN model reports has to come from something other than population substructure: it must be either real co-regulation within Jurkat biology, residual technical/library-size effects the model failed to isolate, or noise. Discriminating among those three is exactly what the diagnostics in this notebook are for.
    """)
    return


@app.cell
def _(Path, scribe):
    # Define data directory
    data_dir = Path(
        "/path/to/jurkat_cells/"
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
    ## Stage 1 — the SVI cascade source

    The first ingredient is a converged NBVCP-SVI fit on the same data. Why bother fitting an "old" model first — and what makes its parameters trustworthy as a seed for NBLN?

    The real reason is **structural identifiability**. NBLN couples three quantities — the per-gene dispersion $r_g$, the per-cell capture $\eta_c$, and the per-cell latent log-rate $\underline{x}_c$ — through a likelihood that cannot pin them all down at once. The three knobs trade off against each other: shift $\eta_c$ and you can compensate with $\underline{x}_c$; rescale $r_g$ and you change how the NB curvature reads the latent; and the gene-gene covariance $\underline{\underline{\Sigma}} = \underline{\underline{W}}\,\underline{\underline{W}}^{\!\top} + \mathrm{diag}(\underline{d})$ leaks into all of it through the per-cell Gaussian prior on $\underline{x}_c$. Cold-starting NBLN-Laplace on the full parameter set is asking the optimizer to disentangle several near-equivalent answers from one likelihood surface; what survives is gauge slop rather than biology.

    The cascade trick exploits a **structural fact about the model family**: NBVCP is exactly the limit of NBLN as $\underline{\underline{\Sigma}} \to 0$. When the gene-gene covariance vanishes, the per-cell log-rate prior collapses to a point, the latent $\underline{x}_c$ drops out as a free parameter, and the NBLN likelihood reduces gene-by-gene to the variable-capture NB you fit in the previous tutorial.

    The crucial observation is that **$r_g$ plays the same role in both models** and does not depend on $\underline{\underline{\Sigma}}$. It is the per-gene dispersion of the NB observation channel — a property of how counts are generated *given* a log-rate — not of how log-rates covary across genes. So the value of $r_g$ that best explains a gene's marginal count distribution under NBVCP is also the value that best explains it under NBLN. In NBVCP the identifiability landscape is dramatically friendlier (no $\underline{\underline{\Sigma}}$-mediated coupling, no per-cell latent layer, the Dirichlet-multinomial story works out cleanly), so the data pins $r_g$ down sharply. We fit there, and carry $r_g$ forward.

    Per-cell capture $\eta_c$ is the harder case. Even in NBVCP it is only partly identified by the data — we need a biology-informed prior (`priors={"capture_efficiency": ...}`) to break the residual ambiguity. And in NBLN there is an additional, exact *rigid-translation gauge*: shift every $x_{c,g}$ up by some per-cell constant $\Delta_c$ and shift $\eta_c$ down by the same $\Delta_c$, and the observed counts are unchanged because the likelihood only sees $x_{c,g} - \eta_c$. So the $\eta_c$ we extract from NBVCP is not the "true" $\eta_c$ in any absolute sense, and we should not pretend it is.

    Here is the payoff that makes the cascade defensible: **downstream compositional inference is robust to misspecification of $\eta_c$**. The compositional-robustness theorem (worked out in the paper) shows that the per-cell composition $\underline{\rho}_c = \mathrm{softmax}(\underline{x}_c)$ is invariant under the rigid translation. Any biologically interesting question that can be phrased compositionally — differential expression in CLR/ILR coordinates, gene-gene correlations from $\underline{\underline{W}}_\perp \underline{\underline{W}}_\perp^{\!\top}$, pathway enrichment on ALR balances — is invariant to whatever per-cell constant the gauge picked up. So when we hand $\eta_c$ to NBLN as a fixed quantity, we are fixing it in a coordinate system where the questions we care about don't see the error.

    The cascade is therefore not "use a richer model's posterior to seed NBLN". It is sharper than that:

    > *Fit the two parameters NBLN cannot pin down cleanly — $r_g$ and $\eta_c$ — in the limit where the model is well-identified ($\underline{\underline{\Sigma}} \to 0$, which is NBVCP), and hand them to NBLN as fixed quantities. The $r_g$ values are exact in the limit (the same parameter, the same role, just easier to fit). The $\eta_c$ values are only fixed up to a gauge that compositional questions don't see anyway.*

    That is the cascade. The next two pieces — the freeze and the loadings shrinkage prior — build on this foundation by pinning the cascade values structurally (Piece 2) and by regularizing the remaining free parameter, $\underline{\underline{W}}$, so that what it picks up is data-supported and not noise (Piece 3).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use:

    - `model="nbvcp"` with `parameterization="mean_odds"` — the same combination that gave clean PPCs in the first tutorial.
    - `priors={"capture_efficiency": (np.log(100_000), 0.1)}` — the **biology-informed capture-efficiency prior**. The two numbers are the mean and standard deviation of $\log(M_c\,\nu_c) = x_{c,g} - \eta_c$ on a reasonable mRNA scale: `log(100,000)` says "this cell sample's true mRNA count is around $10^5$ molecules" (a typical mammalian-cell ballpark) and `0.1` says "we are quite confident in that scale". The number is not a precise measurement — it just has to break the gauge degeneracy.
    - `gene_coverage=0.99` — pool low-coverage genes into one "other" pseudo-gene. Reduces dimensionality without throwing data away.
    - `early_stopping={...checkpoint_dir...}` — Orbax checkpoints let us interrupt and resume the (long) SVI fit without losing progress.

    The block below either loads a previously-fit cached pickle or runs SVI from scratch. On first run, expect this stage to take some time.
    """)
    return


@app.cell
def _(adata, data_dir, np, pickle, scribe):
    # Define output directory
    out_dir = data_dir / "scribe_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cascade-stage knobs (named exactly as in the companion EDA script).
    _gene_coverage = 0.99
    _model_type_svi = "nbvcp"
    _parameterization_svi = "mean_odds"
    _capture_efficiency_prior = (np.log(100_000), 0.1)

    _out_path_svi = out_dir / (
        f"scribe_results_{_model_type_svi}-"
        f"{_parameterization_svi}-vcp-"
        f"{_gene_coverage}gene_coverage-"
        f"anchor_svi.pkl"
    )
    _svi_checkpoint_dir = out_dir / (
        f"checkpoints_{_model_type_svi}-"
        f"{_parameterization_svi}-vcp-"
        f"{_gene_coverage}gene_coverage-"
        f"anchor_svi"
    )

    if _out_path_svi.exists():
        with open(_out_path_svi, "rb") as _f:
            svi_results = pickle.load(_f)
    else:
        _svi_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        svi_results = scribe.fit(
            adata,
            model=_model_type_svi,
            parameterization=_parameterization_svi,
            inference_method="svi",
            gene_coverage=_gene_coverage,
            n_steps=500_000,
            unconstrained=True,
            priors={"capture_efficiency": _capture_efficiency_prior},
            early_stopping={
                "enabled": True,
                "checkpoint_dir": str(_svi_checkpoint_dir),
            },
        )
        with open(_out_path_svi, "wb") as _f:
            pickle.dump(svi_results, _f)

    svi_results
    return out_dir, svi_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sanity-checking the cascade source

    Before we use this fit as an anchor for NBLN, it must clear three sanity checks:

    1. **Training converged.** The ELBO curve should drop fast and then flatten. If the SVI source has not converged, neither will the cascade.
    2. **Marginal PPCs look right.** Per-gene UMI histograms should overlap with the model's predictive band across the abundance range.
    3. **Per-cell capture behaves sensibly.** With a tight prior on $\log(M_c\nu_c)$, the inferred per-cell $\nu_c$ should be a monotone function of library size and saturate near 1 only for the deepest cells.

    Skipping any of these is a recipe for cascade contamination — garbage in, garbage out.
    """)
    return


@app.cell
def _(scribe, svi_results):
    scribe.viz.plot_loss(svi_results, figsize=(6, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The training loss is clean — fast initial drop and a flat tail. Next, marginal PPCs.
    """)
    return


@app.cell
def _(adata, scribe, svi_results):
    scribe.viz.plot_ppc(
        svi_results,
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
    The PPC panels track the observed histograms well across abundance — the cascade source has a coherent marginal story.

    Now we look at the per-cell capture diagnostic. With the biology-informed prior, $\nu_c$ should rise gently with library size and asymptote near 1 only for cells with very deep coverage.
    """)
    return


@app.cell
def _(adata, scribe, svi_results):
    scribe.viz.plot_p_capture_scaling(svi_results, counts=adata, figsize=(4, 4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Capture saturates near $\sim$ 30K UMIs — the deepest Jurkat cells in this dataset are interpreted as essentially fully-captured, while the shallowest cells are routed through a lower effective $\nu_c$. This is the model's account of "library size as a technical channel" rather than "library size as biology". This per-cell $\eta_c$ posterior is what we will hand to NBLN-Laplace as a frozen anchor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A first look at compositional PPCs

    Marginal PPCs ask "does the model reproduce the histogram of UMI counts gene by gene?". That is necessary but not sufficient. The other thing we care about is *whether the joint distribution of expression across genes within a cell is right* — what fraction of a cell's transcriptome is gene A versus gene B versus gene C? This is the **compositional** question, and it is the natural object for downstream differential expression.

    `scribe.viz.plot_compositional_ppc` overlays the empirical distribution of per-cell *fractions* $u_{c,g} / \sum_{g'} u_{c,g'}$ on the model's compositional predictive distribution, for an automatically chosen set of well-expressed genes. If the model has the marginal counts right but the joint structure wrong, this is where you would see it.
    """)
    return


@app.cell
def _(adata, scribe, svi_results):
    scribe.viz.plot_compositional_ppc(
        svi_results,
        adata,
        n_genes=16,
        n_rows=4,
        n_samples=2048,
        min_mean_umi=5.0,
        figsize=(10, 10),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The marginal compositional fractions look fine — but there is a striking visual feature in these panels that is worth pausing on: **the empirical (data) distribution is much wider than the model's predictive distribution.** It looks like the model is "underconfident" about how much spread there should be in each gene's fraction across cells. Is the model wrong?

    No — the *empirical fraction itself* is a noisy estimate, and that is exactly the noise the model is trying to look through.

    Consider what goes into the empirical fraction $\hat{\rho}_{c,g} = u_{c,g} / \sum_{g'} u_{c,g'}$ that we plot for the data. To get one of those numbers we:

    1. Started with a true biological mRNA composition $\underline{\rho}_c$ for cell $c$.
    2. Captured a fraction $\nu_c$ of those molecules (the *technical capture noise*).
    3. Sequenced and counted the survivors — a finite-sample multinomial draw from a finite library (the *multinomial sampling noise*).
    4. Divided counts by total counts, which propagates both noise sources into the resulting fraction.

    The empirical fraction $\hat{\rho}_{c,g}$ is therefore $\rho_{c,g}$ plus *all the noise that happened between cell and FASTQ*. For a gene at fraction $10^{-3}$ in a cell that ended up with 10,000 UMIs, the empirical fraction has roughly $\sqrt{10/10{,}000^2} \approx 3 \times 10^{-4}$ of sampling noise on top of whatever the true biological fraction is — a 30% relative noise floor that has nothing to do with biology.

    What the **model** plots as its predictive distribution, on the other hand, is the *posterior over the true biological $\rho_{c,g}$* — the thing the data is a noisy view of. Of course it is narrower. That gap between empirical and predicted is not the model being overconfident; it is the model correctly distinguishing biological signal from observation noise, and reporting the signal alone. **That distillation is exactly the value-add of having a probabilistic model in the first place.**

    Two consequences worth flagging for downstream analysis:

    - **Naive composition estimators** ($\hat{\rho}_{c,g} = u_{c,g} / \text{total}_c$) are not "the truth" — they are a particular estimator with known noise properties, mostly dominated by sampling at low UMI counts. Differential-expression methods that use $\hat{\rho}$ directly inherit that noise floor.
    - **Size-factor normalization** (divide by a per-cell scalar) does not fix this either. It treats the total UMI count as a single fixed number per cell, when in fact that number is itself a draw from a distribution; and once divided, the per-cell variance of the result is still dominated by multinomial sampling at low counts. A *model* that owns both the biology and the technical noise gets to integrate over both — and that is what produces the narrow predictive bands here.

    Now, compositional **marginals** can still hide a multitude of sins even when the noise gap looks reasonable. What if gene A and gene B individually look right, but when you look at the *joint* distribution of $(\rho_A, \rho_B)$ across cells, the model produces a round blob while the data clearly shows a tilted ellipse? That is a gene-gene correlation the model is missing, and a marginal check will never reveal it.

    The right tool for this question is a compositional **corner plot** — a grid of pairwise 2D projections plus marginals — comparing data and predictive samples panel by panel.
    """)
    return


@app.cell
def _(adata, scribe, svi_results):
    scribe.viz.plot_compositional_corner_ppc(
        svi_results,
        adata,
        n_genes=6,
        n_samples=2048,
        min_mean_umi=5.0,
        density_method="kde",
        figsize=(12, 12),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Diagonals (1D marginals) match. Off-diagonals (2D joint compositions) do not. The data shows tilted ridges in several panels — pairs of genes whose compositional fractions covary across cells — that the SVI model's predictive contours render as round blobs. **The model's generative story does not encode that joint structure.** It cannot: NBVCP emits counts independently across genes given the per-cell capture, so the only cross-gene "correlation" available is what flows through the shared $\phi$ (or the shared posterior on cell-level quantities). That is fundamentally a *marginal* coupling, not a joint one.

    This is the gap NBLN closes. Its log-rate prior $\mathcal{N}(\underline{\mu}, \underline{\underline{W}}\underline{\underline{W}}^{\!\top} + \mathrm{diag}(\underline{d}))$ is a **joint** model for cell-level expression, so the same picture above is the criterion we will use to judge whether NBLN succeeds: do the compositional corner panels move from round blobs to tilted ellipses that match the data?

    Before fitting NBLN, one more useful preview from the SVI cascade source.
    """)
    return


@app.cell
def _(adata, scribe, svi_results):
    scribe.viz.plot_corner_ppc(
        svi_results,
        adata,
        n_genes=5,
        n_samples=512,
        figsize=(8, 8),
        n_contour_levels=3,
        density_method="kde",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_corner_ppc` is the *count-space* analogue of the compositional corner: instead of comparing fractions, it compares raw UMI joints. You see the same pattern there — marginals OK, off-diagonals systematically rounder than the data. NBVCP has no mechanism to fix this. NBLN does.

    There is one important nuance to call out before we move on. The data clouds in the count-space corner show strong-looking diagonal alignments: as gene A's UMI count goes up across cells, so does gene B's. *Most of that diagonal alignment is not gene-gene biology — it is library-size.* Deeply-sequenced cells score higher on essentially every gene, and shallowly-sequenced cells score lower on essentially every gene, simply because they have more (or fewer) total reads. So a count-space pair plot pools "two genes covary because the cell happens to be deep" with "two genes covary because they belong to the same regulatory module", and the former dominates whenever library size varies (which is always, in real data).

    This is exactly why **compositional** corner plots are the better diagnostic for the joint-structure question. By dividing through by the total per cell, the composition removes the library-size axis and leaves whatever residual coupling actually represents co-regulation. The model is being asked to reproduce that residual, not the library-size echo. Keep this in mind when you scan the count-space corner panels: the empirical clouds will look more correlated than the compositional clouds for purely mechanical reasons, and the *gap* between them is roughly the size of the library-size effect we are about to subtract away.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 2 — NBLN-Laplace with the full cascade

    Now we run the NBLN-Laplace fit that consumes the SVI source. There are three new things going on under the hood; here is what each is doing and *why*.

    ### Piece 1: `informative_priors_from=svi_results`

    This tells `scribe.fit` to derive empirical Gaussian priors on the NBLN parameters from the SVI source's posterior samples. Concretely, posterior draws from the SVI fit are moment-matched in NBLN's target coordinate system to produce per-gene priors on the dispersion $r_g$, on the per-gene log-rate offset $\mu_g$, and per-cell priors on $\eta_c$. The NBLN-Laplace loss then includes these as proper Gaussian log-prob terms (in the unconstrained coordinate). The effect: optimization starts in a basin that the SVI source already established as "consistent with the data", and the new NBLN-specific parameters ($\underline{\underline{W}}$, $\underline{d}$, and the latent $\underline{x}_c$ per cell) have somewhere sensible to anchor to.

    Without this anchor, NBLN-Laplace cold-starts on a non-convex landscape with a strong gauge degeneracy. The cascade replaces "where do we start?" with "we start from a fit that is already calibrated".

    #### A Bayesian footnote: this is empirical Bayes, on purpose

    A statistically-careful reader will notice we are doing something unusual: we are *learning prior hyperparameters from the same data we are about to use to fit the posterior*. That is the textbook definition of **empirical Bayes**, and there is a fair amount of (warranted) caution in the Bayesian literature about it — done carelessly, it can underestimate posterior uncertainty.

    Why are we comfortable doing it here?

    1. **The parameters we are turning into priors are the ones the correlated model has the most trouble pinning down on its own.** Per-cell capture $\eta_c$ is gauge-degenerate in NBLN; per-gene dispersion $r_g$ couples nonlinearly to the latent log-rate through the NB. Estimating these in the *uncorrelated* model (NBVCP), where the per-cell log-rates collapse into a tidy Dirichlet-multinomial story with no extra latent layer, is dramatically easier and more stable. We are using the easy fit to inform the hard fit, not running two copies of the same fit on the same data.

    2. **In the large-data limit, empirical-Bayes double-dipping has a vanishing cost.** With ~3,200 cells and tens of thousands of UMIs each, the prior hyperparameters we learn are pinned down to a fraction of their plausible range — far tighter than the residual uncertainty in the NBLN posterior. So when NBLN-Laplace re-uses these priors, the "data was already used to inform the prior" effect is statistically dominated by the much larger amount of data being explained by the new model parameters. This is a property of large-N empirical Bayes that is worked out in detail in the Bayesian statistics literature; a useful intuition is that the priors are informative on a subset of nuisance parameters whose posterior the new model's data would barely sharpen anyway.

    3. **The alternative — uninformative priors on $r_g$ and $\eta_c$ — is genuinely worse.** Without the cascade, NBLN-Laplace has to estimate the gauge-vulnerable parameters and the new correlation parameters at the same time, and the optimizer wanders along the gauge axis because the likelihood is flat there. The empirical-Bayes step gives the optimizer a non-flat objective to follow. The (small) statistical cost of double-dipping is paid for many times over by the (large) numerical-stability gain.

    The right mental model is "we are doing maximum-likelihood estimation of certain nuisance parameters under a simpler, well-identified model, and then conditioning on those estimates as we move to a richer, less-identified model." This is a well-trodden Bayesian workflow — much like using a converged SVI fit to initialize MCMC for fields where MCMC alone would not mix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Piece 2: `informative_priors_freeze=("dispersion", "capture_efficiency")`

    This is where the gauge gets killed. The default freeze tuple — equivalent to using the parameter names `("r", "eta")` — tells NBLN to **not refine** $r_g$ and $\eta_c$ further. They are pinned at the SVI cascade source's MAP values and excluded from the optimizer's parameter dict entirely. Only $\underline{\mu}$, $\underline{\underline{W}}$, $\underline{d}$, and the per-cell latents update during NBLN-Laplace.

    Pinning $\eta_c$ structurally fixes the rigid-translation gauge: now there is no $\Delta_c$ direction along which the loss is flat, because every $\eta_c$ is a constant. Pinning $r_g$ at SVI's values is largely a numerical-stability choice — NBLN cannot meaningfully improve on the SVI source's dispersion posterior here, and refining $r_g$ alongside $\underline{\underline{W}}$ adds noise to the M-step without buying anything.

    Two important details:

    - The descriptive aliases `"dispersion"` and `"capture_efficiency"` resolve to the canonical names `"r"` and `"eta"` automatically (see `FREEZE_KEY_ALIASES`). Either form works.
    - Frozen parameters are still **honest probability distributions** when you ask the result for them — they route through the embedded SVI cascade source, so `result.get_distributions()` returns moment-matched Normals reflecting the SVI posterior, not point masses. The point masses are only used during the optimization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Piece 3: `priors={"loadings": {"type": "horseshoe_columnwise", "tau_scale": 1.0}}`

    The loadings matrix $\underline{\underline{W}}$ used to build the gene-gene low-rank covariance matrix has shape $(G, k)$ with $k = 32$. We do not know in advance how many of those 32 factors the data actually supports. If we let the optimizer do whatever it wants, we will get a fit that uses all 32 — but most of them will be fitting noise. Worse, in a low-N regime the resulting "correlations" will be dominated by overfitting structure.

    A column-wise horseshoe prior says: *each column of $\underline{\underline{W}}$ has its own scale $\lambda_k$, and that scale is drawn from a hierarchical prior that strongly prefers small values but has heavy tails for the few columns that genuinely need to be large.* In effect, the prior performs **adaptive rank selection** during fitting — columns that the data does not support shrink toward zero; columns it does support remain unshrunk. Mathematically this is a sparsity-inducing prior in the *column* dimension, analogous to (but more flexible than) hard-truncating $k$ to a smaller number.

    The benefit over manually picking $k$: you can set $k$ generously (here, 32), let the prior figure out the effective rank, and inspect the spectrum after the fact to see what the data actually supported. We will do exactly that with the `plot_w_shrinkage_spectrum` diagnostic.

    ### Other knobs

    - `parameterization="canonical"` — NBLN's only supported parameterization.
    - `vae_latent_dim=32` — the $k$ in the low-rank covariance. Generous; the shrinkage prior decides the effective value.
    - `d_mode="learned"` — let the per-gene residual variance $d_g$ be optimized rather than fixed.
    - `optimizer_config={...clipped_adam...}` and `laplace_config={...newton_max_step...}` — numerical-stability knobs for the inner Newton solve and the outer M-step in the EM optimization. NBLN is more demanding than NBVCP in this respect.
    - `n_steps=20_000` is much smaller than the SVI source's `500_000` because the M-step here is doing a lighter job: the heavy lifting is in the SVI cascade source, and the freeze removes two-thirds of the parameter blocks from active optimization.
    """)
    return


@app.cell
def _(adata, clear_caches, gc, np, out_dir, pickle, scribe, svi_results):
    clear_caches()
    gc.collect()

    # Match the EDA file's naming convention so the cached pickle from
    # the exploration is picked up automatically when present.
    _gene_coverage_laplace = 0.99
    _latent_dim = 32
    _capture_efficiency_prior = (np.log(100_000), 0.1)

    _out_path_laplace = out_dir / (
        f"scribe_results_nbln-"
        f"canonical-vcp-"
        f"{_latent_dim}latent-"
        f"{_gene_coverage_laplace}gene_coverage-"
        f"anchor-learneddmode_"
        f"horseshoe_columnwisewprior_laplace.pkl"
    )
    _laplace_checkpoint_dir = out_dir / (
        f"checkpoints_nbln-"
        f"canonical-vcp-"
        f"{_latent_dim}latent-"
        f"{_gene_coverage_laplace}gene_coverage-"
        f"horseshoe_columnwisewprior_"
        f"anchor-learneddmode_laplace"
    )

    if _out_path_laplace.exists():
        with open(_out_path_laplace, "rb") as _f:
            laplace_results = pickle.load(_f)
    else:
        _laplace_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        laplace_results = scribe.fit(
            adata,
            model="nbln",
            parameterization="canonical",
            inference_method="laplace",
            informative_priors_from=svi_results,
            informative_priors_freeze=("dispersion", "capture_efficiency"),
            informative_priors_tau=1.0,
            informative_priors_n_samples=1000,
            gene_coverage=_gene_coverage_laplace,
            d_mode="learned",
            latent_dim=_latent_dim,
            n_steps=20_000,
            batch_size=None,
            unconstrained=True,
            priors={
                "capture_efficiency": _capture_efficiency_prior,
                "loadings": {
                    "type": "horseshoe_columnwise",
                    "tau_scale": 1.0,
                },
            },
            early_stopping={
                "enabled": False,
                "checkpoint_dir": str(_laplace_checkpoint_dir),
            },
            optimizer_config={
                "name": "clipped_adam",
                "step_size": 1e-3,
                "grad_clip_norm": 10.0,
            },
            laplace_config={
                "n_newton_steps": 5,
                "damping": 1e-6,
                "newton_tolerance": 1e-2,
                "newton_max_step": 10.0,
            },
        )
        with open(_out_path_laplace, "wb") as _f:
            pickle.dump(laplace_results, _f)

    laplace_results
    return (laplace_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Did training converge? (or: why NBLN-Laplace loss curves look weird)

    NBLN-Laplace's loss is a Laplace approximation to the ELBO: it is the negative log-likelihood of the data evaluated at each cell's MAP latent, plus a half-log-det term that accounts for posterior curvature, plus the global prior log-probs (including the cascade priors and the loadings shrinkage prior). For models like SVI or the Gaussian-conjugate paths we have seen before, the convergence story is simple: the loss drops fast and then flattens.

    For NBLN-Laplace it is **not** that simple, and you should not panic when you see what comes next.

    The loss curve typically does drop fast — it usually settles into a low-loss region within the first ~10K outer-loop iterations — but past that point it is common to see the curve **oscillate wildly**, occasionally spiking back up by orders of magnitude before recovering. This is not a bug in the model or in the fit; it is a structural property of the optimization problem we are solving.

    Why does it happen? Two things compose into instability:

    1. **The inner Newton solve is approximate.** Each outer-loop step needs the per-cell $\underline{x}_c^\ast$ MAP that satisfies $\nabla \log \pi(\underline{u}_c \mid \underline{x}_c, \theta) = 0$. We get there by a small fixed number of Newton iterations per cell (`n_newton_steps=5`), with a step-size cap (`newton_max_step`) and damping. Near a well-shaped basin those few iterations land essentially on the mode. Far from the basin — or when the outer-loop $\theta$ updates push the basin sharply — the inner Newton can take a step that lands somewhere with much higher loss. That higher loss feeds back into the outer step, the outer step over-corrects, and the cycle repeats.
    2. **The Hessian's `log-det` term is non-monotone.** The Laplace correction includes $\tfrac{1}{2} \log \det(-H)$, which is unbounded above and can change quickly as the parameters move around the curvature landscape. Tiny outer-loop steps can change `log-det` by a large amount even when the data fit is essentially unchanged.

    `scribe`'s implementation has two insurance policies for this. The first is **gradient clipping and damping** on the outer Adam step (`grad_clip_norm=10.0`, `damping=1e-6`) — these prevent the worst spikes from happening at all by capping each outer move. The second, and the one that actually matters for your final result, is **best-loss snapshotting**: every outer-loop iteration, the driver keeps a copy of the parameters whenever a new global minimum of the loss is achieved, and at the end of training it returns *those* parameters, not the last-iteration parameters. So even if the curve looks frantic in the second half of training, the returned `laplace_results` object reflects the best-loss snapshot, not whatever volatile state the optimizer ended on. The (well-tested) machinery for this lives in the Laplace-EM driver and is on by default; you do not need to configure it.

    Practically, when you look at the loss curve below, focus on whether the *envelope* of the curve settles low and stays low. Late-stage oscillations on top of a low envelope are fine — `scribe` will pick out the best moment. A loss curve that drifts steadily upward and never settles is the actually-bad pattern that does deserve worry, and in our cascade-frozen regime it almost never happens because the cascade priors anchor the optimizer to a good basin.
    """)
    return


@app.cell
def _(laplace_results, scribe):
    scribe.viz.plot_loss(laplace_results, figsize=(6, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The curve above shows exactly the picture we described: a clean initial drop into a low-loss region, then oscillations on top of that envelope. Because `scribe` tracks the running best-loss snapshot, the parameters in `laplace_results` correspond to the *bottom* of that envelope — not whatever spike the curve happened to end on.

    ### Sanity check the cascade-frozen quantities

    The freeze should mean that $r_g$ and $\eta_c$ from the NBLN result are *the same* as the SVI source's values (up to the moment-matching transform). The `frozen_params` field on the result tells you which parameters were frozen, and `cascade_source` embeds the SVI results object by reference so you can recover full-fidelity posterior samples for any frozen quantity.
    """)
    return


@app.cell
def _(laplace_results):
    print("frozen_params:        ", laplace_results.frozen_params)
    print("cascade_source:       ", type(laplace_results.cascade_source).__name__)
    print("w_prior_diagnostics keys:", list(
        laplace_results.w_prior_diagnostics.keys()
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `frozen_params={'r', 'eta'}` confirms the freeze contract; the `cascade_source` is the embedded SVI result; and `w_prior_diagnostics` is a fresh dict with the loadings-shrinkage summary we will inspect next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 3 — diagnose the loadings spectrum

    With the cascade and the freeze handling capture/dispersion, the only "discovery" the NBLN-Laplace fit is doing is on $\underline{\underline{W}}$ and $\underline{d}$. The horseshoe prior is supposed to have killed the noise columns of $\underline{\underline{W}}$ — but we should *check*, because if we set $k=32$ and only $\sim 5$ columns survive, knowing that explicitly changes how we interpret the correlation structure.

    `scribe.viz.plot_w_shrinkage_spectrum` plots two things on the same axis:

    1. **Solid line:** the per-column compositional norm $\|\underline{\underline{W}}_{\perp,k}\|$ — the gauge-invariant "size" of each factor. This is the headline diagnostic: factors with large compositional norm carry biological signal; factors near zero are noise the prior shrunk away.
    2. **Dashed line:** the per-column aux scale $\sigma_k$ — the variational parameter that controls how tightly column $k$ is shrunk. Heavy-tailed horseshoe scales are slightly less identifiable than the column norms, so the column-norm spectrum is the headline. The aux scales should track the column norms qualitatively.

    The horizontal dashed line marks the 5%-of-maximum threshold that `column_norm_effective_rank` (a number on the result) uses to count "active" factors. **Look for an elbow:** if the spectrum drops sharply at some $k_\text{eff}$ and then flattens, the data supports $k_\text{eff}$ factors. If the spectrum is flat all the way out to $k=32$, the prior was not strong enough and you need to tighten `tau_scale`.
    """)
    return


@app.cell
def _(laplace_results, scribe):
    scribe.viz.plot_w_shrinkage_spectrum(laplace_results, figsize=(5, 3.5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The spectrum shows a clear elbow. A handful of leading factors carry essentially all the compositional signal; the long tail beyond the elbow is at or below the 5%-of-max threshold. The `column_norm_effective_rank` field summarizes this as a single number.
    """)
    return


@app.cell
def _(laplace_results, np):
    _diag = laplace_results.w_prior_diagnostics
    _sigma_k = np.asarray(_diag["sigma_k"])
    print("strategy_type:                  ", _diag["strategy_type"])
    print("column_norm_effective_rank:     ", _diag["column_norm_effective_rank"])
    print("effective_rank (alias):         ", _diag["effective_rank"])
    print(
        "tau (global horseshoe scale):    "
        f"{float(_diag['tau']):.4f}"
    )
    print(
        "sigma_k min/median/max:          "
        f"{float(_sigma_k.min()):.4f} / "
        f"{float(np.median(_sigma_k)):.4f} / "
        f"{float(_sigma_k.max()):.4f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The effective rank is the headline number to report alongside the model. It tells you: "out of 32 latent factors I gave the model, the data supports roughly $k_\text{eff}$ of them after shrinkage." For Jurkat — a monoculture with limited biological heterogeneity — a small $k_\text{eff}$ is sensible.

    Without the loadings prior, this number would have been 32 (by construction) and we would not be able to distinguish "data-supported factor" from "noise direction the model fit anyway".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross-check: the singular spectrum of $\underline{\underline{W}}_\perp$

    The shrinkage-spectrum plot tells you the *per-column* compositional norm — but column norms are not the only thing that matters. Two columns with the same norm can encode very different amounts of information depending on how they line up with each other and with the data's actual covariance structure.

    The right object to look at for *how much variance the model assigns to each underlying mode* is the **singular value spectrum of $\underline{\underline{W}}_\perp$** — the gauge-invariant compositional loadings. The squared singular values $\sigma_i^2$ are the variance contributions of mode $i$ to the implied compositional covariance. If the loadings shrinkage was doing what we hoped, we expect to see the same elbow shape: a few large singular values, then a fast decay.
    """)
    return


@app.cell
def _(laplace_results, np, plt):
    # W_perp = W - mean(W, axis=0) is the gauge-invariant compositional loadings.
    _W_perp = np.asarray(laplace_results.get_W_compositional())
    _singular_values = np.linalg.svd(_W_perp, compute_uv=False)

    _fig, _ax = plt.subplots(figsize=(5, 3.5))
    _k_idx = np.arange(1, _singular_values.shape[0] + 1)
    _ax.plot(_k_idx, _singular_values, marker="o", ms=4, lw=1.0)
    _ax.set_yscale("log")
    _ax.set_xlabel("rank index $i$")
    _ax.set_ylabel(r"$\sigma_i(W_\perp)$  (singular value)")
    _ax.set_title("Singular spectrum of compositional loadings")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same elbow appears in the singular spectrum: a few dominant singular values capture most of the variance, then a long tail at or near the shrinkage floor. This is structurally what the horseshoe prior was designed to produce, and it is what we want for downstream interpretation: the dominant left-singular vectors of $\underline{\underline{W}}_\perp$ are the **gene programs** the data supports — each one a linear combination of genes that co-vary across cells.

    A useful aside on what the gauge projection is doing. The raw $\underline{\underline{W}}$ has a rank-1 component aligned with the all-ones direction in gene space, which corresponds to "uniform up/down across all genes" — i.e., a pure library-size or capture mode. Because the cascade freeze has already pinned $\eta_c$, that component cannot be moving in response to library size during the M-step. But $\underline{\underline{W}}$ can still accidentally pick up a small piece of it. Projecting $\underline{\underline{W}}$ onto the subspace orthogonal to the all-ones direction (which is what `get_W_compositional()` returns) removes that piece by construction. The `get_gauge_diagnostics()` call summarizes how much of $\underline{\underline{W}}$'s mass lives in the gauge versus orthogonal to it.
    """)
    return


@app.cell
def _(laplace_results):
    _gauge_diag = laplace_results.get_gauge_diagnostics()
    print("W_compositional_norm (W_perp):       ", _gauge_diag["W_compositional_norm"])
    print("W_all_ones_component_norm (W_para):  ", _gauge_diag["W_all_ones_component_norm"])
    print("gauge_contamination_ratio:           ", _gauge_diag["gauge_contamination_ratio"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting the gauge-contamination ratio

    The ratio $\|\underline{\underline{W}}_\|\| \,/\, \|\underline{\underline{W}}_\perp\|$ has two very different "healthy" regimes:

    - **Without loadings shrinkage** (or with a very weak prior): a ratio above $\sim 0.2$ is a red flag — it means a sizable fraction of $\underline{\underline{W}}$ is being absorbed by the all-ones gauge mode instead of representing real cross-gene correlation. A ratio below $\sim 0.05$ is clean.
    - **With column-wise horseshoe shrinkage** (the regime we are in): ratios in the $0.5$–$0.8$ range are *routine and benign*. Why? Because the shrinkage prior actively pulls $\underline{\underline{W}}_\perp$ toward zero on noise columns, deflating the denominator. The numerator ($\underline{\underline{W}}_\|$) is small in absolute terms but no longer small *relative* to the deflated denominator. The right question to ask is whether the singular spectrum has a clean elbow (yes — we just saw it) and whether the cascade freeze is doing its job (yes — `frozen_params` confirmed it). The ratio is a secondary diagnostic.

    The full theoretical story for these regimes is in `docs/theory/loadings-shrinkage.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 4 — does the joint structure actually match the data?

    We have a fit that converged, a sparse shrinkage spectrum, and a clean gauge. But that is all *internal* diagnostics. The real test is whether the model's joint compositional predictive distribution matches the empirical joints — the same diagnostic we used on the SVI source, but now with NBLN's covariance structure in play.

    Marginals first.
    """)
    return


@app.cell
def _(adata, laplace_results, scribe):
    scribe.viz.plot_ppc(
        laplace_results,
        adata,
        n_genes=16,
        n_rows=4,
        figsize=(8, 8),
        n_samples=512,
        ppc_level="marginal",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Marginal PPCs look essentially as clean as the SVI source — the cascade succeeded in preserving the marginal fit while moving to a model with real joint structure.

    `plot_mean_calibration` is the global single-number-per-gene version: predicted mean UMIs vs. empirical mean UMIs, on log–log axes. Points hugging the identity line means the model's MAP reproduces gene-level expression levels across the dynamic range.
    """)
    return


@app.cell
def _(adata, laplace_results, scribe):
    scribe.viz.plot_mean_calibration(laplace_results, counts=adata, figsize=(4, 4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the *joint* test — the one NBVCP failed earlier. We render the same compositional corner grid for the NBLN-Laplace fit and look for those tilted ellipses that the SVI source could not reproduce.
    """)
    return


@app.cell
def _(adata, laplace_results, scribe):
    scribe.viz.plot_compositional_corner_ppc(
        laplace_results,
        adata,
        n_genes=6,
        n_samples=2048,
        min_mean_umi=5.0,
        density_method="kde",
        figsize=(12, 12),
        gene_selection="correlation_diverse",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare this against the NBVCP-SVI compositional corner from earlier. The off-diagonal panels are no longer round blobs — the model's predictive contours now follow the tilts and ridges in the empirical joint. **This is the principled gene–gene correlation story:** the contours are coming from NBLN's $\underline{\underline{W}}\underline{\underline{W}}^{\!\top}$ structure, not from a richer variational guide on a model that emits independent counts.

    The argument `gene_selection="correlation_diverse"` asks the plot to pick a set of genes whose pairwise compositional correlations span a range of magnitudes (not all clustered around the same value). That makes the panel grid more informative — you see panels where the model says "tight ridge" alongside panels where it says "essentially independent", and you can check both against the data.

    A pair-level cross-check with the count-space corner:
    """)
    return


@app.cell
def _(adata, laplace_results, scribe):
    scribe.viz.plot_corner_ppc(
        laplace_results,
        adata,
        n_genes=5,
        n_samples=512,
        figsize=(8, 8),
        n_contour_levels=3,
        density_method="kde",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Count-space joint structure also matches. The model has internalized the dependence rather than smearing it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 5 — reading the cross-gene correlation structure

    With the joints validated, we can interpret the correlation matrix that NBLN's covariance implies. `scribe.viz.plot_correlation_heatmap` constructs the implied compositional correlation across genes from $\underline{\underline{W}}_\perp\underline{\underline{W}}_\perp^{\!\top} + \mathrm{diag}(\underline{d})$, picks the most variable subset for readability, and hierarchically clusters them.
    """)
    return


@app.cell
def _(adata, laplace_results, scribe):
    scribe.viz.plot_correlation_heatmap(
        laplace_results,
        counts=adata,
        n_genes=1000,
        figsize=(7, 7),
        n_clusters=4,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The clustered heatmap reveals block structure: groups of co-expressed genes. Because we routed everything through the gauge-killed, sparsity-regularized NBLN model, these blocks correspond to **gene programs the data actually supports** rather than to library-size or noise modes.

    ### One more refinement: subtract the library-size direction explicitly

    Even with the cascade freeze, there is one more interpretive layer worth knowing about. The compositional projection ($\underline{\underline{W}}_\perp = (I - \frac{1}{G}\mathbf{1}\mathbf{1}^\top)\underline{\underline{W}}$) removes the *uniform* all-ones component — "every gene goes up together". But cells in real data do not vary *uniformly* in library size; they vary along a direction in gene space that is correlated with library size in a *non-uniform* way (highly-expressed genes contribute more to library size than rare ones). That cell-to-cell library-size axis is its own one-dimensional component of variation that you may want to subtract before interpreting the rest.

    The `subtract_direction="library_size"` argument fits a library-size axis from the data and projects $\underline{\underline{W}}\underline{\underline{W}}^{\!\top}$ onto its orthogonal complement before computing correlations. What remains is correlation structure perpendicular to library size — what is usually meant by "real co-regulation".
    """)
    return


@app.cell
def _(adata, laplace_results, scribe):
    scribe.viz.plot_correlation_heatmap(
        laplace_results,
        counts=adata,
        n_genes=1000,
        n_clusters=4,
        figsize=(7, 7),
        subtract_direction="library_size",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The library-size-projected heatmap typically shows **fewer, sharper blocks** than the raw one — the dominant block of "all the housekeeping / highly-expressed genes co-vary together" is partly a library-size echo, and once you take it out the remaining structure is closer to what most analysts would call cell-state or gene-program correlation.

    Whether you should look at the raw heatmap or the library-size-projected one depends on the question. For the question "which genes look like they belong to the same regulatory module?" the projected version is usually the right object. For the question "which genes are highly expressed alongside which other highly-expressed genes?" the raw one is fine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why this is the principled answer

    It helps to compare what we just did to the obvious alternatives.

    ### Alternative 1: empirical correlation of normalized counts

    The Scanpy/Seurat path is: normalize to a common library size (or log-transform with $\log(1+x)$), then compute pairwise Pearson correlation across cells. This is fast and easy. But:

    - The normalization is **not a model**; it does not propagate uncertainty. If two genes have correlated *counts* across cells because both have high library-size dependence and you happened to normalize imperfectly, you cannot tell that apart from a real biological covariance.
    - Low-coverage cells contribute as much weight as high-coverage cells, even though they carry much less information.
    - There is no built-in mechanism for separating "two genes co-vary" from "two genes happen to both be high".

    ### Alternative 2: low-rank variational guide on NBVCP (the previous tutorial)

    With `guide_rank=128` and `joint_params="biological"`, the variational guide carries gene-gene posterior correlations. But:

    - The **generative** model still emits independent gene counts. Any joint structure is a property of the posterior over parameters, not of the model itself. You can read the heatmap, but the model does not actually believe in cross-gene covariance — the heatmap is summarizing how the data couples parameter estimates.
    - This is not nothing — when many cells observe gene A together with gene B, posterior over parameters does pick that up. But it confounds "co-expression" with "co-information" in a way that is hard to disentangle.

    ### What NBLN-Laplace + cascade + shrinkage gives you

    - The generative model **has** gene-gene covariance in $\underline{\underline{W}}\underline{\underline{W}}^{\!\top}$. The heatmap is computed from that object directly. Sampling from the predictive distribution produces cells with the right joint structure (we verified this with compositional corner PPCs).
    - The cascade gives us a calibrated starting point and pins capture, so the answer is not contaminated by gauge degeneracies.
    - The shrinkage prior performs adaptive rank selection, so the heatmap is built from data-supported factors only, not from noise the model fit because we let it.
    - Every step is checkable. We can compare PPCs at the marginal level (`plot_ppc`), at the compositional 1D level (`plot_compositional_ppc`), at the compositional 2D level (`plot_compositional_corner_ppc`), and at the gene-pair count level (`plot_corner_ppc`). Each is a different lens on "does this model match the data?", and we passed all of them.

    None of these other workflows let you ask the question this carefully. The cost is a more involved pipeline; the benefit is that the resulting gene-gene correlation matrix is genuinely a property of a calibrated probabilistic model, not a heuristic on transformed counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    The three composable pieces of the cascade workflow:

    1. **SVI cascade source** (`informative_priors_from=svi_results`) — fit NBVCP under SVI with a tight biology-informed capture-efficiency prior, then carry that posterior forward as empirical Gaussian priors on NBLN's dispersion, mean log-rate, and per-cell capture.
    2. **Cascade freeze** (`informative_priors_freeze=("dispersion", "capture_efficiency")`) — pin the cascade-derived $r$ and $\eta$ as constants during the NBLN M-step. This structurally kills the per-cell rigid-translation gauge so that $\underline{\underline{W}}$ can encode cross-gene correlations without being polluted by library-size slop.
    3. **Loadings shrinkage** (`priors={"loadings": {"type": "horseshoe_columnwise", ...}}`) — a column-wise horseshoe prior on the loadings matrix performs adaptive rank selection. With `latent_dim=32` and the prior active, only data-supported factors survive; the rest are shrunk away.

    The diagnostics that confirm the workflow is doing what we want:

    - `plot_loss` — training converges cleanly.
    - `plot_ppc` — marginal count distributions match across the abundance range.
    - `plot_compositional_ppc` — marginal compositional fractions match.
    - `plot_compositional_corner_ppc` — *joint* compositional structure matches; this is what NBVCP could not do.
    - `plot_w_shrinkage_spectrum` — the spectrum of compositional column norms has a clean elbow at $k_\text{eff} \ll 32$.
    - `singular spectrum of W_perp` — independent verification that variance is concentrated in a few modes.
    - `get_gauge_diagnostics()` — the gauge-contamination ratio is benign for shrinkage-active fits (a ratio in $[0.5, 0.8]$ is fine here).
    - `plot_correlation_heatmap` — the headline gene-gene correlation structure, optionally with `subtract_direction="library_size"` for additional interpretability.

    The result is a correlation matrix on genes you can defend as a property of a calibrated joint generative model, with a record of every assumption that went into it and a posterior predictive that demonstrably matches the data's joint structure.

    ### Where to go from here

    - **Differential expression.** With a calibrated joint model in hand, posterior contrasts between gene pairs become honest probabilistic statements. `scribe`'s DE framework (CLR/ILR, lfsr, PEFP) is the next stop.
    - **Multiple cell types.** Jurkat is a monoculture. Heterogeneous tissues require either mixture extensions or per-cluster fits; the cascade machinery generalizes naturally.
    - **Larger latent dim.** For deeply heterogeneous data, push `latent_dim` higher and let the shrinkage prior pick a larger $k_\text{eff}$.
    """)
    return


if __name__ == "__main__":
    app.run()
