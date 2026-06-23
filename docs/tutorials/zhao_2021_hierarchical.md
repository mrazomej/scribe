# Crossed hierarchical models & multi-donor differential expression

This tutorial fits a single **crossed hierarchical** `scribe` model to a
*donor × condition* perturbation experiment and reads off the population-level
treatment effect while explicitly accounting for donor-to-donor heterogeneity.
The data are from [Zhao et al.
(2021)](https://doi.org/10.1038/s41587-021-01066-4): peripheral cells from
**seven donors**, profiled at baseline (`control`) and after treatment with
**panobinostat**, a pan-HDAC inhibitor.

The notebook is built around the multi-factor hierarchy. Mean expression is
decomposed additively as

$$
\log \mu_g^{(\ell)} = \log \mu_g^{\mathrm{pop}} + \alpha_g[\mathrm{cond}(\ell)] + \beta_g[\mathrm{donor}(\ell)],
$$

with a **fixed**, weakly-informative treatment effect $\alpha_g$ (the contrast of
interest) and a **random**, horseshoe-shrunk donor effect $\beta_g$ (the
heterogeneity we average over). It covers:

1. **The model, written out** — negative binomial with per-cell capture, plus
   the crossed additive hierarchy, and *why* the treatment factor is fixed while
   the donor factor is random.
2. **Fit diagnostics** — ELBO loss, a per-leaf mean-calibration grid laid out as
   condition × donor, and a posterior predictive check.
3. **Reading the fitted structure** — exposing the treatment effect and the donor
   deviations as posteriors (`get_factor_effect`), and recovering the paired
   donor groups (`get_group` / `iter_groups`).
4. **Differential expression** — the donor-averaged, paired CLR estimand
   (`compare_groups`), why CLR needs a stable reference (pooling near-silent
   genes into "other"), and the complementary compositional and biological
   views. The recovered hits reproduce the canonical panobinostat / HDAC-inhibitor
   signature (histones, metallothioneins, heat-shock and p53-stress up; MYC,
   cyclins, the myeloid program and MDM2 down) at high sign-confidence.

!!! note "Pre-computed outputs"
    This notebook requires a GPU to run. All outputs shown below were
    pre-computed and exported to static HTML. To re-run it yourself, clone
    the repository and execute the notebook with `marimo edit docs/tutorials/zhao_2021_hierarchical.py`
    on a GPU-enabled machine.

[:material-open-in-new: Open notebook in full page](zhao_2021_hierarchical.html){:target="_blank"}

<iframe
  src="../zhao_2021_hierarchical.html"
  width="100%"
  height="900px"
  style="border: none; display: block;"
></iframe>
