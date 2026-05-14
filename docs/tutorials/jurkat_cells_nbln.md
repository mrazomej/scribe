# Principled gene–gene correlations with NBLN + SVI cascade + sparse loadings prior

This tutorial is a sequel to [Modeling Assumptions for Single-Cell RNA-seq with
`scribe`](jurkat_cells.md). It uses the same [Jurkat Cells
dataset](https://www.10xgenomics.com/datasets/jurkat-cells-1-standard-1-1-0)
(~3,200 cells) and focuses on the Negative-Binomial LogNormal (NBLN) model:
SVI cascade anchoring, gauge handling, and sparse loadings priors so that
reported gene–gene correlations are easier to interpret and defend.

!!! note "Pre-computed outputs"
    This notebook requires a GPU to run. All outputs shown below were
    pre-computed and exported to static HTML. To re-run it yourself, clone
    the repository and execute the notebook with `marimo edit docs/tutorials/jurkat_cells_nbln.py`
    on a GPU-enabled machine.

[:material-open-in-new: Open notebook in full page](jurkat_cells_nbln.html){:target="_blank"}

<iframe
  src="jurkat_cells_nbln.html"
  width="100%"
  height="900px"
  style="border: none; display: block;"
></iframe>
