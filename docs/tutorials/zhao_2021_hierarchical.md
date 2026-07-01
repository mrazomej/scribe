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
\log \mu_g^{(\ell)} = \log \mu_g^{\mathrm{pop}} + \alpha_g[\mathrm{cond}(\ell)] + \beta_g[\mathrm{donor}(\ell)] + \gamma_g[\mathrm{cond}(\ell), \mathrm{donor}(\ell)],
$$

with a **fixed**, weakly-informative treatment effect $\alpha_g$ (the contrast of
interest), a **random**, horseshoe-shrunk donor baseline $\beta_g$, and a
**random**, horseshoe-shrunk per-donor treatment slope $\gamma_g$ — the treatment
× donor interaction that lets each donor *respond* to the drug differently (the
heterogeneity we must average over honestly). It covers:

1. **The model, written out** — negative binomial with per-cell capture, plus
   the crossed additive hierarchy, and *why* the treatment factor is fixed, the
   donor factor is random, and the per-donor treatment slope is a
   horseshoe-shrunk interaction that costs nothing where donors agree.
2. **Fit diagnostics** — ELBO loss, a per-leaf mean-calibration grid laid out as
   condition × donor, and a posterior predictive check.
3. **Reading the fitted structure** — exposing the treatment effect, the donor
   baseline deviations, and the per-donor response slope as posteriors
   (`get_factor_effect` and the interaction site), whose between-donor spread is
   the empirical case for carrying it, and recovering the paired donor groups
   (`get_group` / `iter_groups`).
4. **Differential expression** — the donor-averaged, paired CLR estimand
   (`compare_groups`), why CLR needs a stable reference (pooling near-silent
   genes into "other"), and the complementary compositional and biological
   views. The recovered hits reproduce the canonical panobinostat / HDAC-inhibitor
   signature (histones, metallothioneins, heat-shock and p53-stress up; MYC,
   cyclins, the myeloid program and MDM2 down) at high sign-confidence — while the
   slope keeps the calls **donor-generalizable**, holding back genes on which
   donors disagree (e.g. acute heat-shock, p21) rather than crediting one or two
   strong responders to the whole population.

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
