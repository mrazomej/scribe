# Bayesian model comparison: should the dispersion be hierarchical too?

This short tutorial is a direct sequel to the [**Crossed hierarchies &
multi-donor differential expression**](zhao_2021_hierarchical.md) tutorial. There
we fit a single crossed *donor × condition* model to the [Zhao et al.
(2021)](https://doi.org/10.1038/s41587-021-01066-4) panobinostat data and read
off the treatment effect — but we made a quiet modeling choice: the additive
hierarchy was placed only on the gene **mean** $\mu_g$, while the gene
**dispersion** $r_g$ stayed a single value shared across all leaves.

Here we ask whether that choice was the right one. Because `scribe`'s
`mean_disp` parameterization samples $(\mu_g, r_g)$ as *independent* coordinates,
we can give **both** of them the donor × condition hierarchy and let each gene's
over-dispersion shift with treatment. That yields two competing models —

- **Model A — `mu_only`**: hierarchy on $\mu$; one shared $r_g$ per gene (the DE-tutorial model).
- **Model B — `mu_plus_r`**: the *same* additive crossed hierarchy on **both** $\mu$ and $r$.

Model B is strictly more flexible. The tutorial uses `scribe.mc` — the Bayesian
model-comparison toolkit — to answer the question you should always ask before
trusting a more complex model: **does the extra flexibility actually buy better
out-of-sample predictions, or is it just fitting noise?** It covers:

1. **The two models, written out** — why the `mean_disp` (Fisher-orthogonal)
   parameterization is what makes a hierarchy on $r$ *expressible* at all, and the
   sense in which Model B nests Model A.
2. **The comparison framework** — expected log predictive density (elpd), the
   `WAIC` analytical approximation and its complexity penalty, and `PSIS-LOO`
   with its per-observation reliability diagnostic $\hat{k}$.
3. **Why the comparison is at the *gene* level, not the cell** — leave-one-cell-out
   is structurally **ill-posed** for any variable-capture model: the per-cell
   capture latent $\nu_c$ (`p_capture`) reverts to its prior for a held-out cell,
   so importance sampling cannot bridge the posterior→prior collapse and
   $\hat{k}\to 1$ for essentially every cell. `compare_models` detects the
   cell-specific latent and switches to the gene as the unit automatically.
4. **The gene-level verdict** — per-gene elpd differences, standard errors, and
   z-scores. The result is *concentrated, not uniform*: the large majority of
   genes favour the simpler `mu_only` (WAIC correctly penalizes unused
   flexibility), and `mu_plus_r` wins decisively only on a biologically coherent,
   genuinely over-dispersed set — the **immunoglobulin** and **mitochondrial**
   gene families.
5. **Where in expression space the gain lives — and whether to trust it** — the
   per-gene elpd gain plotted against mean UMI and coloured by over-dispersion,
   separating *sparse-but-genuinely-bimodal* wins (trustworthy) from
   *sparse-and-near-Poisson* ones (too noisy to trust).

!!! note "Pre-computed outputs"
    This notebook requires a GPU to run. All outputs shown below were
    pre-computed and exported to static HTML. To re-run it yourself, clone
    the repository and execute the notebook with `marimo edit docs/tutorials/zhao_2021_model_comparison.py`
    on a GPU-enabled machine.

[:material-open-in-new: Open notebook in full page](zhao_2021_model_comparison.html){:target="_blank"}

<iframe
  src="../zhao_2021_model_comparison.html"
  width="100%"
  height="900px"
  style="border: none; display: block;"
></iframe>
