# Bursty genes and the two-state promoter model

This tutorial is a follow-up to [Modeling Assumptions for Single-Cell RNA-seq
with `scribe`](jurkat_cells.md). Every model in that tutorial was, at heart, a
**negative binomial** — and a negative binomial can never be genuinely
*bimodal*. This one asks the question those models structurally cannot answer:
which genes are genuinely **bursty** (a two-mode count distribution from a
promoter that switches slowly between OFF and ON states), and what does the
**two-state (telegraph) promoter** model capture that the negative binomial —
even with a gene-specific $p_g$ — cannot?

The notebook has **two phases**:

1. **Validate on synthetic data**, where the ground truth is known. We generate
   counts from a known two-state process (some genes deeply bursty, some
   negative-binomial), recover the bimodality a negative binomial cannot, and
   work carefully through *which* biophysical parameters are identifiable from
   snapshot counts and which are not (the mean, excess Fano, burst frequency,
   and bursting regime are; the absolute switching rates are not).
2. **Apply to a real monoculture** — a [deeply-sequenced **K562**
   dataset](https://www.10xgenomics.com/datasets/10k-human-k562-r-cells-singleplex-sample-1-standard)
   from 10x Genomics. A monoculture is essential: it isolates promoter bursting
   *within one cell type* from the very different signal of a gene being on in
   one cell type and off in another (a job for a mixture model). The honest
   result is that most genes sit in the negative-binomial limit, which the
   two-state model correctly reports — and we examine the handful it flags as
   deviating.

!!! note "Pre-computed outputs"
    This notebook requires a GPU to run. All outputs shown below were
    pre-computed and exported to static HTML. To re-run it yourself, clone
    the repository and execute the notebook with `marimo edit docs/tutorials/k562_cells_twostate.py`
    on a GPU-enabled machine.

[:material-open-in-new: Open notebook in full page](k562_cells_twostate.html){:target="_blank"}

<iframe
  src="../k562_cells_twostate.html"
  width="100%"
  height="900px"
  style="border: none; display: block;"
></iframe>
