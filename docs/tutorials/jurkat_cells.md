# Modeling Assumptions for Single-Cell RNA-seq with `scribe`

This tutorial walks through a series of modeling and inference choices that
`scribe`'s Bayesian engine makes available for single-cell RNA-seq data, using
the [Jurkat Cells dataset](https://www.10xgenomics.com/datasets/jurkat-cells-1-standard-1-1-0)
from 10x Genomics (~3,200 cells, monoculture). Because there is only one cell
type, the focus is entirely on the statistical models rather than biological
heterogeneity.

!!! note "Pre-computed outputs"
    This notebook requires a GPU to run. All outputs shown below were
    pre-computed and exported to static HTML. To re-run it yourself, clone
    the repository and execute the notebook with `marimo edit docs/tutorials/jurkat_cells.py`
    on a GPU-enabled machine.

[:material-open-in-new: Open notebook in full page](jurkat_cells.html){:target="_blank"}

<iframe
  src="../jurkat_cells.html"
  width="100%"
  height="900px"
  style="border: none; display: block;"
></iframe>
