import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bursty genes and the two-state promoter model in `scribe`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *This tutorial is a follow-up to* [Modeling Assumptions for Single-Cell RNA-seq with `scribe`](jurkat_cells.md). *It assumes you are comfortable with the ideas built up there — negative binomial counts, variable capture, parameterizations, hierarchical gene-specific $p_g$ — and want to ask a question those models structurally cannot answer.*

    Every model in the previous tutorial was, at heart, a **negative binomial**. We made it more flexible in many ways: per-cell capture $\nu_c$, friendlier coordinates $(\mu_g, \phi)$, joint low-rank guides, even a gene-specific success probability $p_g$. But all of those moves change the *parameters* of a negative binomial — none of them changes its *shape*. And a negative binomial, for any choice of $(r, p)$, is always either monotonically decreasing from zero or unimodal away from the boundary. **It can never be genuinely bimodal.**

    Some mammalian genes, however, have promoters that switch slowly between an OFF and an ON state. When that switching is slow relative to mRNA degradation, a population of cells splits into two groups — cells caught in a long OFF stretch (counts near zero) and cells caught in a long ON stretch (a second mode at moderate counts) — producing a two-mode count distribution. No negative binomial can fit that.

    This tutorial has **two phases**:

    1. **Validate on synthetic data.** We generate counts from a *known* two-state process — some genes deeply bursty, some negative-binomial — fit the model, and check two things: (a) that it recovers the bimodality a negative binomial structurally cannot, and (b) *which* biophysical parameters are actually identifiable from snapshot counts and which are not. This is where the model's power is unambiguous, because we control the ground truth.
    2. **Apply to a real monoculture.** We then fit a [deeply-sequenced **K562** dataset](https://www.10xgenomics.com/datasets/10k-human-k562-r-cells-singleplex-sample-1-standard). The honest spoiler: genuine promoter-bursting bimodality is *hard* to see in droplet total counts — for reasons we will make precise — so most genes sit in the negative-binomial limit, which the two-state model correctly reports. We use the fit to find the genes that deviate and ask whether they make biological sense.

    > **A note on dataset choice.** It is tempting to reach for a heterogeneous tissue (we tried [PBMC](https://www.10xgenomics.com/datasets/10-k-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-single-indexed-3-1-standard-4-0-0)), where marker genes look strikingly bimodal. But that conflates two very different mechanisms: *promoter bursting within one cell type* versus *a gene being on in one cell type and off in another*. The two-state model is built for the first; the second is a job for a mixture model. To study bursting cleanly we need a **monoculture** — Jurkat from the previous tutorial was one, but too shallow; K562 is a deeper one.
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

    # Set our plotting style (totally optional)
    scribe.viz.matplotlib_style()
    return Path, clear_caches, gc, np, pickle, plt, scribe, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One backbone, different priors on the latent rate

    It is worth pausing to see how cleanly the three models in the paper relate. Every count model `scribe` implements shares the same backbone: an observed UMI count is a **Poisson draw from a latent per-gene rate**, and the per-cell capture efficiency $\nu_c$ thins that rate. What distinguishes the models is *the prior placed on the latent rate* — and that single choice controls two things at once: the **shape** the count distribution can take, and whether the clean Dirichlet–Multinomial composition (the backbone of `scribe`'s differential-expression machinery) survives.

    **The negative binomial is the Poisson–Gamma compound.** This is the model from the previous tutorial:

    $$
    \lambda_g \sim \mathrm{Gamma}(r_g, \theta),
    \qquad
    m_g \mid \lambda_g \sim \mathrm{Poisson}(\lambda_g)
    \;\;\Longrightarrow\;\;
    m_g \sim \mathrm{NB}(r_g, p).
    $$

    Biophysically this is the steady state of a **one-state, constitutively bursty promoter**: the gene is always able to fire, transcription arrives in geometric bursts, and the *unbounded* support of the Gamma reflects the fact that bursts can, in principle, pile up arbitrarily high.

    **The two-state promoter model replaces the Gamma with a Beta.** A promoter that switches $\mathrm{OFF}\xrightarrow{k^+}\mathrm{ON}$ and $\mathrm{ON}\xrightarrow{k^-}\mathrm{OFF}$, producing mRNA at rate $r$ only while ON and degrading it at rate $\gamma$ (we non-dimensionalize all rates by $\gamma$, labeling them as $\hat x$), has a steady-state count distribution that is *exactly* a **Poisson–Beta** compound:

    $$
    p \sim \mathrm{Beta}(\hat{k}^+_g, \hat{k}^-_g),
    \qquad
    m_g \mid p \sim \mathrm{Poisson}(\hat r_g\, p).
    $$

    Here $p \in [0, 1]$ is the **memory-weighted fraction of recent history the promoter spent ON** (weighted by the mRNA-lifetime kernel), and $\hat k^+, \hat k^-$ are the switching rates in units of the mRNA lifetime. The structural parallel to the negative binomial is exact — *same Poisson, different mixing distribution for the rate* — but the consequence is dramatic:

    - The Gamma has **unbounded support** $[0, \infty)$: it can be peaked but never has a second mode at the top.
    - The Beta has **bounded support** $[0, 1]$, and when both shape parameters are below 1 it is **U-shaped** — mass piled at *both* ends. Pushed through the Poisson, that produces a genuinely **bimodal** count distribution: a spike at zero (cells caught OFF) and a second mode near $\mathrm{Poisson}(\hat r)$ (cells caught ON). The bounded support is the biophysical signature: even a permanently-ON promoter caps out at $\mathrm{Poisson}(\hat r)$, because production and degradation are finite.

    The next cell makes this concrete with synthetic draws — same Poisson, the only difference is Gamma vs. Beta for the rate.
    """)
    return


@app.cell
def _(np, plt):
    # Same Poisson backbone, two different priors on the latent rate.
    _rng = np.random.default_rng(0)
    _n = 50_000
    _target_mean = 20.0

    # --- Poisson-Gamma (negative binomial): unbounded Gamma rate ---------
    _r = 2.0
    _lam = _rng.gamma(shape=_r, scale=_target_mean / _r, size=_n)
    _counts_nb = _rng.poisson(_lam)

    # --- Poisson-Beta (two-state): bounded, U-shaped Beta rate -----------
    # k+ = k- = 0.3 < 1 makes the Beta U-shaped (promoter mostly fully ON
    # or fully OFF), and r_hat sets where the "ON" mode lands.
    _k_on, _k_off = 0.3, 0.3
    _r_hat = _target_mean / (_k_on / (_k_on + _k_off))  # so E[count] = target
    _p = _rng.beta(_k_on, _k_off, size=_n)
    _counts_ts = _rng.poisson(_r_hat * _p)

    # Plot distributions
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    _bins = np.arange(0, 90)

    _axes[0].hist(_counts_nb, bins=_bins, color="C0", density=True)
    _axes[0].set_title(r"Poisson–Gamma (negative binomial)")
    _axes[0].set_xlabel("count $m$")
    _axes[0].set_ylabel("density")

    _axes[1].hist(_counts_ts, bins=_bins, color="C1", density=True)
    _axes[1].set_title(r"Poisson–Beta (two-state promoter)")
    _axes[1].set_xlabel("count $m$")

    _fig.suptitle(
        f"Same mean ($\\approx{_target_mean:.0f}$), same Poisson — "
        f"only the rate prior differs"
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both panels have the *same mean* and the *same Poisson observation step*. The negative binomial (left) is unimodal — a peak near zero with a tail. The Poisson–Beta (right) is unmistakably **bimodal**: a population of cells with the promoter caught OFF (counts near zero) and a separate population caught ON (a second mode near $\hat r$). This is the qualitative behaviour the negative binomial structurally cannot produce, and it is the entire reason the two-state model exists.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### When does the two-state model collapse back to the negative binomial?

    Crucially, the two-state model **contains** the negative binomial as a special case. As the OFF rate grows, $\hat k^-_g \gg 1$ (the promoter spends only brief, frequent ON visits, and the long mRNA lifetime averages over many of them), the Poisson–Beta limits *exactly* to a negative binomial:

    $$
    \pi_{\text{steady-state}}(m) \;\xrightarrow{\; \hat k^-_g \gg 1\;}\;
    \mathrm{NegBinom}\!\left(m \,\middle|\, \hat k^+_g,\; \frac{1}{1 + \hat r_g / \hat k^-_g}\right).
    $$

    This gives a precise biophysical dictionary for the negative binomial parameters we have been fitting all along:

    | Negative binomial parameter | Two-state biophysical meaning |
    |---|---|
    | shape $r_{\mathrm{NB}} = \hat k^+_g$ | dimensionless ON rate $=$ **burst frequency** |
    | burst size $b_g = \hat r_g / \hat k^-_g$ | mean mRNA yield per ON visit $=$ **burst size** |

    The burst size has a clean derivation: each ON visit lasts, on average, $1/\hat k^-_g$ mRNA lifetimes and produces mRNA at rate $\hat r_g$, so the expected yield per visit is $\hat r_g / \hat k^-_g$.

    The **contrapositive is the whole point of the model**: the genes for which the negative binomial fits *poorly* are exactly the genes whose data *reject* the fast-switching limit. A gene with $\hat k^-_g \lesssim 1$ has ON/OFF dwell times comparable to the mRNA lifetime; the lifetime cannot smooth over the switching, and the steady-state distribution is genuinely bimodal. **That is what "bursty" means here, and $\hat k^-_g$ is the knob that measures it.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The `moment_delta` parameterization (and the fit configuration)

    There is a practical wrinkle. The natural parameters $(\hat r_g, \hat k^+_g, \hat k^-_g)$ are **not jointly identifiable**: you can scale $\hat r_g \to \lambda \hat r_g$ and $\hat k^+_g \to \lambda \hat k^+_g$ and leave both the mean and (to leading order) the Fano factor unchanged — a flat, banana-shaped ridge in parameter space that mean-field variational inference handles badly, exactly like the canonical $(r, p)$ ridge in the first tutorial.

    `scribe` fixes this the same way it did before: reparameterize into data-identifiable combinations. The two-state model offers four mean-preserving parameterizations; we use **`moment_delta`**, whose sampled coordinates are

    - $\mu_g$ — the gene mean expression (a level knob, exactly the quantity most researchers think about);
    - $F_g = \mathrm{Var}/\mu_g - 1$ — the **excess Fano factor**, the overdispersion-above-Poisson knob;
    - $\delta_g = \dfrac{1}{\kappa_g + 1} \in (0, 1)$, where $\kappa_g = \hat k^+_g + \hat k^-_g$ — the **regime knob**.

    The first two moments are preserved *by construction*, so $\mu_g$ and $F_g$ are pinned down directly by the data. The regime coordinate $\delta_g$ carries the part the data constrain most weakly, and it has a clean interpretation:

    - $\delta_g \to 0$ ($\kappa_g \to \infty$, fast switching) is the **negative binomial limit**;
    - $\delta_g \to 1$ is the **extreme bursty** regime (the Beta degenerates to mass at $\{0, 1\}$ — the promoter is fully ON or fully OFF).

    Mapping it to a bounded interval $(0, 1)$ keeps the mean-field guide from wasting probability mass over an arbitrarily long ridge. From the fitted model we can read off any of the biophysical quantities — $\hat k^+_g$, $\hat k^-_g$, $\hat r_g$, the burst size — as deterministic functions of $(\mu_g, F_g, \delta_g)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The exact `scribe.fit` configuration we use for the two-state fit is annotated below:

    - `model="twostatevcp"` — the two-state promoter likelihood with per-cell **v**ariable **c**apture **p**robability. Because the Poisson–Beta is *closed under binomial thinning* (a thinned Poisson is just a Poisson with a scaled rate), capture enters as $\hat r_g \mapsto \nu_c\,\hat r_g$, exactly the clean story we had for the negative binomial. (In **Phase 1** the synthetic data are generated with no capture, so we fit the plain `model="twostate"` — model and data-generating process match exactly; in **Phase 2** the real K562 data have capture, so we use `twostatevcp`.)
    - `parameterization="moment_delta"` — the $(\mu_g, F_g, \delta_g)$ coordinates above.
    - `unconstrained=True` — fit in an unconstrained real space (positive params via a transform, the bounded $\delta_g$ via a sigmoid), which is numerically better behaved.
    - `priors={"inv_concentration": (0.0, 2.0)}` — a **neutral prior on the regime coordinate** $\delta$. `scribe`'s built-in default is deliberately NB-leaning (`logit(δ) ~ Normal(-2, 2)`): it will not declare a gene bursty without strong evidence, which is the right conservative default for routine fits. But for a study that is *about* burstiness, that prior actively fights the signal — so we relax it to a symmetric `Normal(0, 2)` and let the data, not the prior, decide the regime. (We will see in Phase 1 exactly how much this prior matters.)
    - `n_quad_nodes=256` — the number of Gauss–Legendre nodes used to integrate the Poisson–Beta likelihood over the latent $p$. This is **the `scribe` default**, and it matters specifically in the bursty regime: a U-shaped Beta ($\hat k^+_g, \hat k^-_g < 1$) has near-singular mass at the endpoints $p \to 0, 1$, and a coarse grid under-resolves it — systematically biasing the inferred regime coordinate toward the negative-binomial limit, which flattens the predicted ON mode.
    - `positive_transform="exp"` and `optimizer_config={... step_size: 1e-3 ...}` — put every positive parameter on a log scale and use a large enough step. The regime coordinate converges *slowly*; an `exp` transform and a healthy step size are what let it actually reach the bursty solution rather than stalling NB-ward (see the negative binomial foil's "one transform for everything" note below).
    - `n_steps`, `batch_size`, `early_stopping` — standard SVI controls with `clipped_adam` (gradient clipping keeps the bursty-regime gradients well behaved). We fit on the full gene set (no `gene_coverage` pre-filter), so the two-state and negative binomial panels share exactly the same genes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase 1 — Validate on synthetic data

    Before trusting the model on real data, we ask the most basic question: *if the data really were generated by a two-state promoter, would the fit recover it?* We can answer that exactly, because we generate the data ourselves.

    We simulate $G = 20$ genes in $C = 5{,}000$ cells. Half of the genes are **deeply bursty** (U-shaped Beta, $\hat k^+_g, \hat k^-_g < 1$ — slow switching, genuinely bimodal); half are **negative-binomial-like** controls ($\hat k^-_g \gg 1$ — fast switching).
    """)
    return


@app.cell
def _(np):
    # Ground-truth two-state parameters: 10 bursty (U-shaped) + 10 NB-like.
    _rng = np.random.default_rng(0)
    n_cells_sim = 5_000
    n_bursty = 10
    n_nb = 10
    _n_genes = n_bursty + n_nb

    k_on_true = np.concatenate([
        _rng.uniform(0.25, 0.45, n_bursty),   # bursty: U-shaped
        _rng.uniform(0.50, 2.00, n_nb),       # NB-like
    ])
    k_off_true = np.concatenate([
        _rng.uniform(0.10, 0.25, n_bursty),   # bursty: U-shaped (deep valley)
        _rng.uniform(15.0, 50.0, n_nb),       # NB limit (fast OFF)
    ])
    r_hat_true = _rng.uniform(10.0, 120.0, _n_genes)   # ON-mode location

    # p_gc ~ Beta(k_on, k_off);  u_gc ~ Poisson(r_hat * p)
    _p = _rng.beta(k_on_true[None, :], k_off_true[None, :], size=(n_cells_sim, _n_genes))
    counts_sim = _rng.poisson(r_hat_true[None, :] * _p).astype(np.int64)

    counts_sim.shape
    return counts_sim, k_off_true, k_on_true, n_bursty, n_nb, r_hat_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, confirm the synthetic data actually look the way we intend: the bursty genes (top row) should show a deep-valley, two-mode shape; the NB-like controls (bottom row) should be unimodal.
    """)
    return


@app.cell
def _(counts_sim, k_off_true, k_on_true, n_bursty, np, plt):
    _idx = [0, 1, 2, 3, 10, 11, 12, 13]   # 4 bursty, 4 NB-like
    _fig, _axes = plt.subplots(2, 4, figsize=(11, 5))
    for _ax, _i in zip(_axes.ravel(), _idx):
        _ax.hist(
            counts_sim[:, _i],
            bins=np.arange(0, counts_sim[:, _i].max() + 2),
            density=True, histtype="step", color="black",
        )
        _grp = "bursty" if _i < n_bursty else "NB-like"
        _ax.set_title(
            f"gene {_i} ({_grp})\n$k^+$={k_on_true[_i]:.2f}, $k^-$={k_off_true[_i]:.2f}",
            fontsize=8,
        )
        _ax.set_xlabel("counts")
    _fig.suptitle("Synthetic data — bursty genes (top) are bimodal; NB-like (bottom) are not")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now fit both models to the synthetic counts. Because the synthetic data carry **no capture**, we fit the plain `model="twostate"` — the fitted model is then *exactly* the data-generating process, which is what makes this a clean recovery test. We use the neutral regime prior, `positive_transform="exp"`, and `n_quad_nodes=256` (the settings discussed above; the node count is what stops the U-shaped Beta from being mis-integrated and pulled NB-ward). The **negative binomial foil** is fit for the head-to-head. Both are cached to disk so the GPU fit runs once.
    """)
    return


@app.cell
def _(Path, clear_caches, counts_sim, gc, pickle, scribe):
    clear_caches()
    gc.collect()

    # Cache synthetic fits under the configured data root (out of the repo).
    import json as _json
    with Path(__file__).with_name("tutorial_paths.local.json").open(
        "r", encoding="utf-8"
    ) as _f:
        _root = _json.load(_f)["SCRIBE_TUTORIAL_DATA_ROOT"]
    synth_dir = Path(_root).expanduser() / "twostate_synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)

    _out_path = synth_dir / "synthetic_twostatevcp_moment_delta.pkl"
    if _out_path.exists():
        with open(_out_path, "rb") as _f:
            results_ts_sim = pickle.load(_f)
    else:
        results_ts_sim = scribe.fit(
            counts_sim,
            model="twostate",
            parameterization="moment_delta",
            inference_method="svi",
            unconstrained=True,
            n_steps=200_000,
            positive_transform="exp",
            priors={"inv_concentration": (0.0, 2.0)},   # neutral regime prior
            optimizer_config={
                "name": "clipped_adam", "step_size": 1e-3, "grad_clip_norm": 10.0,
            },
            n_quad_nodes=256,
        )
        with open(_out_path, "wb") as _f:
            pickle.dump(results_ts_sim, _f)

    results_ts_sim
    return results_ts_sim, synth_dir


@app.cell
def _(clear_caches, counts_sim, gc, pickle, scribe, synth_dir):
    clear_caches()
    gc.collect()

    _out_path = synth_dir / "synthetic_nbvcp_mean_odds.pkl"
    if _out_path.exists():
        with open(_out_path, "rb") as _f:
            results_nb_sim = pickle.load(_f)
    else:
        results_nb_sim = scribe.fit(
            counts_sim,
            variable_capture=True,
            parameterization="mean_odds",
            prob_prior="gaussian",
            unconstrained=True,
            n_steps=100_000,
        )
        with open(_out_path, "wb") as _f:
            pickle.dump(results_nb_sim, _f)

    results_nb_sim
    return (results_nb_sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The showcase: two-state recovers what the negative binomial cannot

    Posterior predictive checks for the bursty synthetic genes (integer indices `0–7`), same genes in both panels. The negative binomial foil first — watch its band straddle the empty middle of the bimodal data.
    """)
    return


@app.cell
def _(counts_sim, results_nb_sim, scribe):
    _pr = scribe.viz.plot_ppc(
        results_nb_sim, counts_sim, genes=list(range(8)),
        n_rows=2, figsize=(10, 5), n_samples=512,
    )
    _pr.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the two-state model on the same genes. Because its latent rate is Beta-distributed, it can place mass at zero *and* at a second mode at once — so its predictive band should follow the deep-valley shape the negative binomial could only average over.
    """)
    return


@app.cell
def _(counts_sim, results_ts_sim, scribe):
    _pr = scribe.viz.plot_ppc(
        results_ts_sim, counts_sim, genes=list(range(8)),
        n_rows=2, figsize=(10, 5), n_samples=512,
    )
    _pr.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That is the whole point of the model, demonstrated on data where we *know* the answer: the negative binomial cannot represent the bimodality at any setting of its parameters; the two-state model recovers it. Now we ask the harder, more honest question — given a good PPC, *which biophysical parameters did we actually learn?*
    """)
    return


@app.cell
def _(k_off_true, k_on_true, np, r_hat_true, results_ts_sim):
    # Posterior medians for the fit, in BOTH coordinate systems. moment_delta
    # *samples* (mu, excess_fano, delta) and *derives* (k_on, k_off, r_hat) as
    # deterministic functions of them — so the same posterior can look pinned in
    # one chart and degenerate in the other. We extract both to show the contrast.
    _post = results_ts_sim.get_posterior_samples(
        n_samples=400, store_samples=False, convert_to_numpy=True
    )
    # Derived biophysical rates (top row of the recovery plot).
    kon_fit = np.median(_post["k_on"], axis=0)
    koff_fit = np.median(_post["k_off"], axis=0)
    rhat_fit = np.median(_post["r_hat"], axis=0)
    # Sampled moment_delta coordinates (bottom row) — what the data constrains:
    # mu = E[count], excess_fano = Var/Mean - 1, delta = 1/(kappa+1) in (0, 1).
    mu_fit = np.median(_post["mu"], axis=0)
    fano_fit = np.median(_post["excess_fano"], axis=0)
    delta_fit = np.median(_post["inv_concentration"], axis=0)
    # log10 inter-quantile width of the k_off posterior (>1 spans >10x = non-identified)
    _lo, _hi = np.percentile(_post["k_off"], [16, 84], axis=0)
    koff_logwidth = np.log10(_hi) - np.log10(_lo)

    # Ground-truth moment coordinates, mapped from the generative
    # (k_on, k_off, r_hat) with the SAME definitions scribe samples in (see
    # TwoStateMomentDeltaParameterization). For p ~ Beta(k_on, k_off) and
    # count ~ Poisson(r_hat * p), with kappa = k_on + k_off:
    #   mu          = r_hat * k_on / kappa                 (= E[count])
    #   excess_fano = r_hat * k_off / (kappa*(kappa+1))    (= Var/Mean - 1)
    #   delta       = 1 / (kappa + 1)
    _kappa = k_on_true + k_off_true
    mu_true = r_hat_true * k_on_true / _kappa
    fano_true = r_hat_true * k_off_true / (_kappa * (_kappa + 1.0))
    delta_true = 1.0 / (_kappa + 1.0)
    return (
        delta_fit,
        delta_true,
        fano_fit,
        fano_true,
        koff_fit,
        koff_logwidth,
        kon_fit,
        mu_fit,
        mu_true,
        rhat_fit,
    )


@app.cell
def _(
    delta_fit,
    delta_true,
    fano_fit,
    fano_true,
    k_off_true,
    k_on_true,
    koff_fit,
    kon_fit,
    mu_fit,
    mu_true,
    n_bursty,
    n_nb,
    plt,
    r_hat_true,
    rhat_fit,
):
    _colors = ["C1"] * n_bursty + ["C0"] * n_nb
    _fig, _axes = plt.subplots(2, 3, figsize=(12, 8))
    _panels = [
        # Top row — DERIVED biophysical rates. Read these for honesty: k^- and
        # r_hat are singular in the NB limit and will scatter for NB-like genes.
        (k_on_true, kon_fit, "$k^+$ (burst frequency)"),
        (k_off_true, koff_fit, "$k^-$ (OFF rate)"),
        (r_hat_true, rhat_fit, r"$\hat r$ (ON-mode rate)"),
        # Bottom row — SAMPLED moment coordinates. These are what the data
        # directly constrains; all three land on the diagonal for every gene.
        (mu_true, mu_fit, r"$\mu$ (mean expression)"),
        (fano_true, fano_fit, "excess Fano $F-1$"),
        (delta_true, delta_fit, r"$\delta$ (regime)"),
    ]
    for _ax, (_t, _f, _name) in zip(_axes.ravel(), _panels):
        _ax.scatter(_t, _f, c=_colors, s=25)
        _lo = min(_t.min(), _f.min())
        _hi = max(_t.max(), _f.max())
        _ax.plot([_lo, _hi], [_lo, _hi], "k--", lw=1)
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel(f"true {_name}")
        _ax.set_ylabel(f"fitted {_name}")
    _fig.suptitle(
        "Parameter recovery (orange = bursty, blue = NB-like)\n"
        "top: derived biophysical rates    bottom: sampled moment coordinates"
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(delta_fit, k_off_true, koff_fit, koff_logwidth, n_bursty, np):
    # Quantify identifiability per group: how well is k_off recovered, and
    # how wide is its posterior?
    for _lbl, _sl in [
        ("BURSTY  (truth k_off < 1)", slice(0, n_bursty)),
        ("NB-LIKE (truth k_off >> 1)", slice(n_bursty, None)),
    ]:
        print(f"--- {_lbl} ---")
        print(f"  k_off recovery   median(fit/true) = "
              f"{np.median(koff_fit[_sl] / k_off_true[_sl]):.2f}")
        print(f"  k_off posterior  median log10-IQR = "
              f"{np.median(koff_logwidth[_sl]):.2f}   (>1 spans >10x => non-identified)")
        print(f"  delta (sampled regime)  median = {np.median(delta_fit[_sl]):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What is identifiable, and what is not

    The recovery plot has **two rows**, and reading them together is the most important methodological lesson of this model. The **top row** shows the *derived* biophysical rates $(k^+, k^-, \hat r)$; the **bottom row** shows the coordinates the model actually *samples* $(\mu,\, F\!-\!1,\, \delta)$. They are smooth functions of one another — the *same* posterior — yet the top row can look badly degenerate while the bottom row is far better behaved: two of its three panels are pinned for every gene, and the third fails *gracefully* (a bounded bias) rather than blowing up. That gap is the whole point — identifiability is a property of the *coordinate* you read, not of the fit.

    **Pinned for every gene — the observable moments $\mu$ and $F-1$ (and the burst frequency $k^+$).** The mean $\mu$ (bottom-left) and the excess Fano $F-1$ (bottom-middle) sit on the diagonal across the *entire* range — bursty and NB-like genes alike — because they are pure functions of the observed count distribution, so the data nail them directly. The burst frequency $k^+$ (top-left) is recovered too. These are the directions you can trust per gene, and they are exactly the scale-free, observable-moment directions.

    **Non-identifiable in the NB limit — the OFF rate $k^-$ and the exact regime $\delta$.** For the bursty genes both are pinned: $k^-$ lands on the diagonal (top-middle) and so does $\delta$ (bottom-right). For the NB-like genes *neither* does — and that is *correct*, not a fit failure. Once switching is fast the counts are negative-binomial, with only two degrees of freedom (burst frequency and burst size), so the data carry no information about how *deep* into the NB limit a gene sits. The two coordinates express that same blind spot differently. The *unbounded* $k^-$ shows it as a **blow-up**: it scatters over a decade toward $\infty$ (the `log10-IQR` printed above is large for the NB-like group, tiny for the bursty group). The *bounded* $\delta = 1/(\kappa+1)\in(0,1)$ cannot run to $\infty$, so the NB-like genes instead pile up just *above* the diagonal at small $\delta$ — biased toward "not-quite-as-deep-in-the-NB-limit-as-truth," held off the boundary by the prior. The payoff for practice is decisive: even un-pinned, $\delta$ keeps the two regimes **cleanly separated** (here every NB-like gene has $\delta<0.15$, every bursty gene $\delta>0.4$), so the bursty-vs-NB *classification* is robust even where the precise rate is not.

    **Built from the identified moments, but singular in the NB limit — the absolute rate $\hat r$ (and burst size).** Compare the top-right ($\hat r$) and bottom-left ($\mu$) panels for the *same* genes: $\mu$ sits on the diagonal everywhere, yet $\hat r$ scatters for the NB-like genes. There is no contradiction — $\hat r = \mu + (F-1)/\delta$, so as $\delta\to0$ (the NB limit) $\hat r\propto 1/\delta$ diverges. The data pin $\mu$, $F-1$, and the regime, but not the *exact* tiny $\delta$, and $\hat r$ amplifies precisely that weakly-constrained direction (the same ridge as $k^-$). Only the finite combination — the NB shape $r_{\mathrm{NB}} = \mu/(F-1)$ — is identified for those genes. This is the bottom-right bias seen from the other side: the NB-like genes whose $\delta$ sits *above* its diagonal have $\hat r$ *below* its diagonal, because $\hat r$ falls as $\delta$ rises — one fact, two coordinates. A *second*, independent scale ambiguity enters on **real** data: with variable capture (Phase 2) counts constrain only the *product* $\hat r_g\,\nu_c$ — rescale every gene's $\hat r$ and inversely rescale every cell's $\nu_c$ and the likelihood is unchanged — so the absolute transcript rate rides on the capture *prior*, not the data. If you need it absolutely, anchor capture with an informative prior (the biology-informed capture anchor from the previous tutorial).

    **The one tension no parameterization escapes — valley depth vs. ON-population breadth.** A single Beta couples *how deep the valley is* and *how broad the ON population is*. A deep valley needs an extreme U-shaped Beta, which forces the ON cells to a narrow Poisson; a broad ON needs a milder Beta, which fills the valley. A real **marker gene** — bimodal because of discrete cell types, with an *over-dispersed* expressing population — needs both at once, and no single Beta delivers. That is a job for a **mixture model**, and it is exactly why we insist on a monoculture: to study bursting we must remove the cell-type axis the two-state model was never built to represent.

    With that understanding in hand — the observed moments $\mu$ and $F-1$, the burst frequency $k^+$, and the bursty-vs-NB *regime* are what we trust per gene; the exact switching rates ($k^-$, and $\delta$ deep in the NB limit) and the absolute rate $\hat r$ are *not* pinned by total-count data; and cell-type bimodality is out of scope — we turn to real data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase 2 — A real deeply-sequenced monoculture (K562)

    K562 is a human chronic-myelogenous-leukemia line — a **monoculture**, so any bimodality we find is *within* one cell type, not a mix of types. We use a deeply-sequenced 10x dataset ([10k K562-r, singleplex](https://www.10xgenomics.com/datasets/10k-human-k562-r-cells-singleplex-sample-1-standard)).

    Set expectations honestly before we look. Genuine telegraph-bursting bimodality is **hard to see in droplet total counts**, for a fundamental reason: each diploid cell's count is the *sum of two independently-bursting alleles*, and the sum of two ON/OFF processes is far less bimodal than either allele (the mixed ON/OFF cells fill the valley). Capture noise compounds it by smearing the OFF state into the dropout floor. This is exactly why the bursting-inference literature works at the **allele level** — total counts are a blunt observable for switching. So we *expect* most genes here to sit in the negative-binomial limit, which the two-state model should report as such. The interesting question is whether the handful of genes it flags as deviating make biological sense.

    We load the data through `scribe.data_loader`, as before.
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
    data_dir = Path(_data_root).expanduser() / "10k_k562r"

    # Load the data
    adata_total = scribe.data_loader.load_and_preprocess_anndata(
        data_dir, return_jax=False
    )

    adata_total
    return adata_total, data_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before fitting anything, let us look at the per-cell library size (total UMI count). This is the depth axis from the discussion above: it tells us how much of each cell's transcriptome actually made it into the data, and therefore how much room we have to resolve a true OFF state from a technical zero.
    """)
    return


@app.cell
def _(adata_total, np, plt, sns):
    # Total UMI per cell; .A1 flattens a sparse-matrix row sum to a 1-D array
    _library_sizes = np.asarray(adata_total.X.sum(axis=1)).ravel()

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
    The library-size distribution spans a wide range, and many barcodes are shallow. Since depth is precisely what lets bimodality survive into the counts, we keep only the **deeply-sequenced cells** — those with at least `umi_thresh` total UMIs. This is not the usual quality-control reflex of discarding "bad" cells; it is a deliberate enrichment for the regime where the two-state signal is detectable. The trade-off is fewer cells, but each carries more information about the *shape* of every gene's distribution — which is exactly what we are trying to read. The ECDF below shows the library-size distribution after the cut.
    """)
    return


@app.cell
def _(adata_total, np, plt, sns):
    # Define UMI count threshold
    umi_thresh = 10_000

    # Filter data
    adata = adata_total[adata_total.X.sum(axis=1) >= umi_thresh]

    # Total UMI per cell; .A1 flattens a sparse-matrix row sum to a 1-D array
    _library_sizes = np.asarray(adata.X.sum(axis=1)).ravel()

    _fig, _ax = plt.subplots(figsize=(5, 4))
    sns.ecdfplot(_library_sizes, ax=_ax)
    _ax.set_xlabel('total UMI count per cell')
    _ax.set_ylabel('ECDF')
    _ax.set_ylim(-0.01, 1.01)
    plt.show()
    return adata, umi_thresh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting the negative binomial foil

    Our point of comparison is a **gene-specific $p_g$** negative binomial — the most flexible negative binomial from the previous tutorial, with each gene drawing its own success probability (`prob_prior="gaussian"`, i.e. a learned $\mathrm{logit}(p_g)\sim\mathcal{N}(\mu_p,\sigma_p)$ hierarchy) on top of per-cell variable capture, fit in the `mean_odds` parameterization. Because we are on a new dataset, we refit it from scratch rather than reusing the previous tutorial's pickle.

    Why this particular foil? Because it makes the punchline sharp. Per-gene $p_g$ is the most a negative binomial can flex within its own family, and yet — being a negative binomial — it is *still structurally unimodal*. If the two-state model captures bimodality that this model misses, the gap cannot be blamed on "the negative binomial just needed more parameters." It is a difference in *shape*, not in *flexibility*.

    ### One transform for everything: `positive_transform="exp"`

    There is one change from the previous tutorial worth dwelling on, because it bit **both** models here. `scribe` fits in an unconstrained space and maps back to the positive parameters through a transform; the default is `softplus`. On the shallower Jurkat data, `softplus` was fine for everything except the mean (where we used `exp`). On a deeply-sequenced dataset like K562, *every* positive parameter needed `exp` — hence `positive_transform="exp"` applied globally to both the negative binomial and the two-state fit (and to the synthetic fits in Phase 1).

    The reason is the depth itself. A deeply-sequenced library stretches the true dynamic range of the parameters across many orders of magnitude: gene means, dispersions, burst sizes and switching rates now genuinely span from $\sim\!10^{-2}$ to $\sim\!10^{3}$. `softplus` is nearly linear for large arguments, so its Jacobian *saturates* — to move a parameter from $1$ to $1000$ the optimizer must take an enormous unconstrained step, and gradient descent crawls. `exp` makes the step *multiplicative*: a fixed step in unconstrained space is a fixed *ratio* in parameter space, so the optimizer traverses several decades as easily as one. When the data occupy only a narrow range (a shallow library), `softplus` is perfectly adequate and you never notice; when the data reveal the full log-scale spread, `exp` becomes necessary.

    This is worth stating plainly: **there is no single, monolithic configuration of these models that fits every dataset.** The right parameterization, transform, optimizer, and depth filter all depend on what the data actually contain. That is not a defect of the approach — it is the reality of probabilistic modeling and optimization at this scale. The workflow is iterative by design: fit, read the diagnostics, adjust the piece they flag, refit. The diagnostics we run next are exactly how you catch a saturating transform or an under-converged fit *before* trusting any downstream biology.
    """)
    return


@app.cell
def _(adata, clear_caches, data_dir, gc, pickle, scribe, umi_thresh):
    # Output directory shared with the previous tutorial's fits.
    out_dir = data_dir / "scribe_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear GPU memory from previous fits
    clear_caches()
    gc.collect()

    # Define parameterization
    _parameterization = "mean_odds"

    # Define output file path
    _out_path = out_dir / f"scribe_results_nbvcp-prob_{_parameterization}_{umi_thresh}umi.pkl"

    if _out_path.exists():
        # Load model from pkl file
        with open(_out_path, "rb") as _f:
            results_nb = pickle.load(_f)
    else:
        # Fit basic model to data with variable capture probability fit per cell
        results_nb = scribe.fit(
            adata,
            variable_capture=True,
            parameterization=_parameterization,
            prob_prior="gaussian",
            unconstrained=True,
            positive_transform="exp",
            n_steps=100_000,
            batch_size=2048,
            n_quad_nodes=256,
        )
        # Save the fitted model
        with open(_out_path, "wb") as _f:
            pickle.dump(results_nb, _f)

    results_nb
    return out_dir, results_nb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first diagnostic, as always, is the ELBO loss curve: a healthy run drops fast and then flattens. If the curve were still sliding downward at the end — a classic symptom of a saturating transform starving the optimizer — that would be the cue to revisit the configuration above before reading anything into the fit.
    """)
    return


@app.cell
def _(results_nb, scribe):
    # Plot ELBO loss
    _fig = scribe.viz.plot_loss(results_nb, figsize=(6, 3))
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, posterior predictive checks across a grid of genes spanning the dynamic range. For the great majority of genes the negative binomial is an excellent model — it should track the observed histograms closely here. That is the point of the foil: it is *not* a strawman. Keep this in mind for later, because the genes where it fails are a specific, mechanistically meaningful minority, not a sign of general inadequacy.
    """)
    return


@app.cell
def _(adata, results_nb, scribe):
    _fig = scribe.viz.plot_ppc(
        results_nb,
        adata,
        n_genes=16,
        n_rows=4,
        figsize=(8, 8),
        n_samples=512,
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A global check complements the per-gene PPCs: the mean-calibration plot compares each gene's observed mean UMI count to the model's predicted mean from the MAP parameters. Points hugging the identity line mean the model reproduces overall expression levels across the whole transcriptome, not just the handful of genes that happen to land in the PPC grid.
    """)
    return


@app.cell
def _(adata, results_nb, scribe):
    _fig = scribe.viz.plot_mean_calibration(
        results_nb, counts=adata, figsize=(4, 4)
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, the capture-scaling diagnostic: each cell's fitted capture probability $\nu_c$ against its library size. Because we deliberately kept only deep cells, this also confirms that the depth we selected for is being absorbed by the technical capture channel rather than leaking into the gene-level parameters. With the negative binomial foil validated, we can fit the two-state model and compare.
    """)
    return


@app.cell
def _(adata, results_nb, scribe):
    _fig = scribe.viz.plot_p_capture_scaling(
        results_nb, counts=adata, figsize=(4, 4)
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting the two-state model

    Now we fit the two-state promoter model with the `moment_delta` configuration described earlier — and, for the same reason we just discussed, with `positive_transform="exp"` on every positive parameter. We clear GPU memory before the fit and cache the result to a pickle so the (GPU-only) fit runs once. The two-state likelihood is heavier than the negative binomial (it integrates over the Beta latent at every evaluation), so this is the most expensive fit in the notebook.
    """)
    return


@app.cell
def _(adata, clear_caches, gc, out_dir, pickle, scribe, umi_thresh):
    # Clear GPU memory from previous fits
    clear_caches()
    gc.collect()

    # Define the fit identity (used for the cache filename)
    _model_type = "twostatevcp"
    _parameterization = "moment_delta"
    _inference_method = "svi"

    _out_path = out_dir / (
        f"scribe_results_{_model_type}-"
        f"{_parameterization}-"
        f"{umi_thresh}umi.pkl"
    )

    if _out_path.exists():
        # Load model from pkl file
        with open(_out_path, "rb") as _f:
            results_twostate = pickle.load(_f)
    else:
        # Fit the two-state promoter model with per-cell variable capture.
        results_twostate = scribe.fit(
            adata,
            model=_model_type,
            parameterization=_parameterization,
            inference_method=_inference_method,
            unconstrained=True,
            positive_transform="exp",
            n_steps=100_000,
            batch_size=1024,
            priors={"inv_concentration": (0.0, 2.0)},
            optimizer_config={
                "name": "clipped_adam",
                "step_size": 1e-3,
                "grad_clip_norm": 10.0,
            },
            early_stopping={"enabled": True, "patience": 10_000},
            n_quad_nodes=256,
        )
        # Save the fitted model
        with open(_out_path, "wb") as _f:
            pickle.dump(results_twostate, _f)

    results_twostate
    return (results_twostate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As always, the first plot to make is the ELBO loss curve: a clean run drops quickly and then flattens.
    """)
    return


@app.cell
def _(results_twostate, scribe):
    # Plot ELBO loss
    _fig = scribe.viz.plot_loss(results_twostate, figsize=(6, 3))
    _fig.fig
    return


@app.cell
def _(adata, results_twostate, scribe):
    _fig = scribe.viz.plot_ppc(

        results_twostate,
        adata,
        n_genes=16,
        n_rows=4,
        figsize=(8, 8),
        n_samples=512,
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The fit to these example genes is great. That is a good sanity check for the two-state promoter model. Let's now look at the global diagnostic for the predicted and observed mean expression across all genes.
    """)
    return


@app.cell
def _(adata, results_twostate, scribe):
    _fig = scribe.viz.plot_mean_calibration(
        results_twostate, counts=adata, figsize=(4, 4)
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The mean calibration looks great. That means that that the combination of the two-state and the variable capture probability is able to reproduce the observed UMI counts.

    Because `twostatevcp` carries a per-cell capture parameter $\nu_c$, the capture-scaling diagnostic from the previous tutorial applies unchanged — it confirms depth is being routed through the technical channel sensibly.
    """)
    return


@app.cell
def _(adata, results_twostate, scribe):
    _fig = scribe.viz.plot_p_capture_scaling(
        results_twostate, counts=adata, figsize=(4, 4)
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the burstiness off the fit

    The two-state fit gives us, for every gene, a posterior over its biophysical parameters. The quantity that most directly measures "how far from a negative binomial" a gene is sits in the **OFF rate $\hat k^-_g$**:

    - $\hat k^-_g \gg 1$ — fast switching, the negative binomial limit. The gene is effectively constitutively bursty; its counts are unimodal.
    - $\hat k^-_g \lesssim 1$ — slow switching. The OFF/ON dwell times are comparable to the mRNA lifetime, the lifetime cannot average over the switching, and the counts are genuinely **bimodal**.

    Even though we sampled in `moment_delta` coordinates $(\mu_g, F_g, \delta_g)$, `scribe` materializes the derived biophysical parameters (`k_off`, `k_on`, `burst_size`, `r_hat`, …) as deterministic sites in the posterior, so we can pull them directly with `get_posterior_samples` and take per-gene posterior medians.

    Here we fit both models on the full gene set (no `gene_coverage` pre-filter), so the two-state and negative binomial panels share the same genes and line up directly — there is no pooled `_other` column to account for.
    """)
    return


@app.cell
def _(adata, np, results_twostate):
    # Draw posterior samples of the (derived) biophysical parameters and
    # summarize each gene by its posterior median.
    _post = results_twostate.get_posterior_samples(
        n_samples=1_000, store_samples=False, convert_to_numpy=True
    )

    k_off = np.median(_post["k_off"], axis=0)
    excess_fano = np.median(_post["excess_fano"], axis=0)
    mu_ts = np.median(_post["mu"], axis=0)
    burst_size = np.median(_post["burst_size"], axis=0)

    gene_names_retained = np.asarray(adata.var_names)

    print(
        f"{(k_off < 1.0).sum()} have posterior-median k_off < 1 "
        f"(slow-switching / bursty)."
    )
    return excess_fano, gene_names_retained, k_off, mu_ts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plotting the OFF rate $k^-_g$ against the gene mean $\mu_g$, coloured by the excess Fano factor, gives a map of the transcriptome's burstiness. Genes below the dashed line ($k^-_g < 1$) are the slow-switching, bimodal-capable genes — and they should also carry the largest excess overdispersion, since the two-state model attributes overdispersion to promoter switching (the excess variance is maximized at symmetric switching $k^+_g = k^-_g$ and vanishes when either rate dominates).
    """)
    return


@app.cell
def _(excess_fano, k_off, mu_ts, np, plt):
    _fig, _ax = plt.subplots(figsize=(5.5, 4.5))
    _sc = _ax.scatter(
        mu_ts,
        k_off,
        c=np.log10(excess_fano + 1e-3),
        cmap="viridis",
        s=8,
        alpha=0.6,
        edgecolor="none",
    )
    _ax.axhline(1.0, color="black", linestyle="--", lw=1)
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_xlabel(r"gene mean expression $\mu_g$")
    _ax.set_ylabel(r"OFF rate $k^-_g$ (mRNA-lifetime units)")
    _ax.text(
        _ax.get_xlim()[0] * 1.5,
        0.6,
        "bursty / bimodal  ($k^-_g < 1$)",
        fontsize=9,
        color="black",
    )
    _cb = plt.colorbar(_sc, ax=_ax)
    _cb.set_label(r"$\log_{10}$ excess Fano factor $F_g$")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which genes leave the NB limit — and are they cell-cycle genes?

    On a monoculture we do not expect a forest of bimodal genes (the allele-averaging argument above), so the question is sharper: of the genes the two-state model flags as *most* over-dispersed and slowest-switching, do any make biological sense? In a proliferating line like K562 the most reliable source of genuine within-population spread is the **cell cycle** — histones, mitotic cyclins and kinetochore genes (`HIST1H4C`, `CCNB1`, `CCNB2`, `MKI67`, `TOP2A`, `UBE2C`, `CENPF`, …) swing with cycle phase. If the model's "bursty" flags pick those up, that is a reassuring sanity check — with the honest caveat that the mechanism is cycle phase, not promoter telegraph.

    We pull the most-bursty genes (smallest $\hat k^-_g$), note how many are cell-cycle markers, then run the head-to-head PPC on them: the gene-specific $p_g$ negative binomial vs. the two-state model, the same genes by name in both panels (via `plot_ppc`'s `genes=` argument).
    """)
    return


@app.cell
def _(gene_names_retained, k_off, mu_ts, np):
    # Most bursty genes (smallest k_off) among adequately expressed genes.
    _expressed = (mu_ts >= 1.0) & (mu_ts <= 500)
    _cand = np.where(_expressed)[0]
    _top = _cand[np.argsort(k_off[_cand])][:8]

    bursty_names = list(gene_names_retained[_top])

    # Cell-cycle markers we'd expect to be the most over-dispersed in a
    # proliferating monoculture (intersected with genes actually present).
    _cc = ["HIST1H4C", "HIST1H1C", "HIST1H2AC", "CCNB1", "CCNB2", "MKI67",
           "TOP2A", "UBE2C", "CENPF", "PCNA", "TYMS", "CKS1B", "CKS2"]
    _name_to_idx = {g: i for i, g in enumerate(gene_names_retained)}
    _cc_here = [g for g in _cc if g in _name_to_idx]
    _cc_set = set(_cc_here)

    print("Most bursty genes (smallest k_off):")
    for _name, _ko in zip(bursty_names, k_off[_top]):
        _flag = "   <- cell-cycle" if _name in _cc_set else ""
        print(f"  {_name:>12s}   k_off = {_ko:.3f}{_flag}")

    if _cc_here:
        print("\nCell-cycle markers present, ranked by k_off (smaller = burstier):")
        for _g in sorted(_cc_here, key=lambda g: k_off[_name_to_idx[g]]):
            print(f"  {_g:>12s}   k_off = {k_off[_name_to_idx[_g]]:.3f}")
    return (bursty_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Negative binomial (gene-specific $p_g$).** Posterior predictive checks for the eight most bursty genes. Watch for the model's predictive band sitting *between* the data's two modes — a single hump trying to straddle a bimodal histogram.
    """)
    return


@app.cell
def _(adata, bursty_names, results_nb, scribe):
    _fig = scribe.viz.plot_ppc(
        results_nb,
        adata,
        genes=bursty_names,
        n_rows=2,
        figsize=(9, 4),
        n_samples=512,
    )
    _fig.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Two-state promoter model.** The same eight genes, in the same order. The Poisson–Beta likelihood can place mass at zero *and* at a second mode simultaneously, so its predictive band should track the bimodal shape the negative binomial could only average over.
    """)
    return


@app.cell
def _(adata, bursty_names, results_twostate, scribe):
    scribe.viz.plot_ppc(
        results_twostate,
        adata,
        genes=bursty_names,
        n_rows=2,
        figsize=(9, 4),
        n_samples=512,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On K562 the contrast is **muted compared with the synthetic showcase**, and that is the honest, expected result. Most genes sit so close to the negative-binomial limit that the two predictive bands are nearly indistinguishable — the two-state model is *correctly* reporting that bursting is not needed. The genes it does flag as most over-dispersed tend to be the cell-cycle markers above; for those the two-state band is modestly heavier-tailed and better-centered than the negative binomial, but the deep, clean valley we manufactured in Phase 1 is absent. Allele averaging and capture noise have done exactly what we warned they would — blunted the telegraph signal in total counts.

    The takeaway is not "the model failed." It is "the data do not contain strong telegraph bimodality, and the model says so." That is the most useful thing a generative model can do: fit the bimodal genes when they exist (Phase 1), reduce to the negative binomial when they do not (most of K562), and hand you a per-gene burst-frequency / regime readout either way. Reading absolute switching rates off a monoculture's total counts is asking more than droplet data can give — that is a job for allele-resolved or imaging assays, not a different `scribe` knob.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A closing thread: the same algebraic fault line, twice

    It is worth connecting this back to the previous tutorial's *other* extension — the hierarchical gene-specific $p_g$ model — because the two are deeper cousins than they look.

    Recall what made the negative binomial Dirichlet–Multinomial (NBDM) factorization so clean: when every gene shares the same success probability $p$, the latent rates are Gammas with a *common* rate parameter, and independent Gammas with a shared rate normalize to a **Dirichlet** on the simplex. That single algebraic fact — and *only* that fact — is what lets `scribe`'s differential-expression machinery operate on the dispersion vector $\underline{r}$ alone, with $p$ canceling out of the composition.

    Both extensions break that fact, for the same underlying reason:

    - **Hierarchical $p_g$** ([`_hierarchical_p.qmd`](https://github.com)): once each gene has its own $p_g$, the Gamma rates $\theta_g = p_g/(1-p_g)$ are no longer shared, the total and the composition couple, and the closed-form Dirichlet is gone. `scribe` replaces it with a Gamma-based sampler that reduces *exactly* to the Dirichlet when all $p_g$ coincide.
    - **Two-state** ([`_two_state_promoter.qmd`](https://github.com)): the latent rate is now a **Beta**, and the Beta simply does not have the Gamma's normalize-to-a-Dirichlet property. The composition $\rho_g = \hat r_g p_g / \sum_j \hat r_j p_j$ is still a perfectly valid simplex vector, and a sample-based CLR differential-expression pipeline still works, but the closed-form Dirichlet step disappears.

    So "gene-specific $p_g$" and "bursty promoters" are two faces of the same move: *leaving the special Gamma-with-shared-rate geometry that makes NBDM analytically clean.* The hierarchical model leaves it to let the mean–variance relationship vary across genes; the two-state model leaves it to let the count distribution change *shape*. Both pay the same price (no closed-form Dirichlet) and both recover NBDM in a limit ($\sigma_p \to 0$ for the hierarchy, $k^-_g \to \infty$ for the two-state model).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This tutorial extended the negative binomial story of the previous one in the single direction the negative binomial cannot go on its own — **distribution shape** — and then tested that extension honestly against reality.

    ### What we did

    1. **Identified the pivot.** Every `scribe` count model is `Poisson(latent rate)`; the prior on that rate (Gamma vs. Beta) sets both what shapes the counts can take and whether the Dirichlet composition survives.
    2. **Validated on synthetic data (Phase 1).** On data generated from a known two-state process, the negative binomial could not fit the bursty genes and the two-state model recovered them — proving the capability where we control the truth. We then mapped out *identifiability*: burst frequency $k^+$ and the regime $\delta$ are recovered; the OFF rate $k^-$ is non-identifiable in the NB limit (and the fit says so, with a wide posterior); the absolute rate $\hat r$ is pinned only up to the capture scale.
    3. **Applied to a real monoculture (Phase 2).** On deeply-sequenced K562 the two-state model reduced to the negative binomial for the vast majority of genes — the correct, honest verdict, since allele averaging and capture noise suppress telegraph bimodality in total counts. The genes it flagged as most over-dispersed lined up with the cell cycle.

    ### When to reach for it

    The two-state model earns its keep when you suspect genuine **bimodal expression from slow promoter switching** *within a homogeneous population*. It is not free: the likelihood requires a one-dimensional quadrature over the Beta latent (Gauss–Legendre, in `scribe`, with a configurable `n_quad_nodes`), and you lose the closed-form Dirichlet composition. For genes in the fast-switching limit it simply reproduces the negative binomial — so let posterior predictive checks, not the loss curve, decide whether the extra structure paid off.

    ### The broader message

    Three threads run through this tutorial. First, **validate on data you control before trusting real data** — Phase 1 is what lets us read Phase 2 honestly instead of hopefully. Second, **a good generative model is as valuable when it says "no" as when it says "yes"**: reducing to the negative binomial across most of K562 is information, not failure. Third, **the limits are often in the data, not the model** — total counts from a diploid monoculture are a blunt instrument for telegraph kinetics (the field uses allele-resolved or imaging assays for that), and priors on weakly-identified coordinates have to be chosen deliberately, as our neutral regime prior was.

    ### What we did not cover

    - **Cell-type bimodality** — when the two modes are *different cell types* (as in PBMC), the right tool is a **mixture model**, not a single two-state promoter; a single Beta cannot match a deep valley *and* a broad, over-dispersed ON population at once.
    - **Differential expression with two-state compositions** — the sample-based CLR pipeline that replaces the closed-form Dirichlet.
    - **Correlated two-state models (TSLN)** — a per-cell latent layer so bursty genes can covary (and the ON population can be over-dispersed), fit via an SVI→Laplace cascade anchored on the fit we built here.
    """)
    return


if __name__ == "__main__":
    app.run()
