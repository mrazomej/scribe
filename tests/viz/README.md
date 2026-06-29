# tests/viz

**Purpose.** Visualization — the plotting layer, including posterior-predictive
(PPC) plot rendering, corner/compositional plots, and plot-config gating.

**Source under test.** `src/scribe/viz` (`_interactive`, `ppc`, `ppc_rendering`,
`corner_ppc`, `compositional_ppc`, `compositional_corner_ppc`, `capture_anchor`,
`mean_calibration`, `mixture_ppc`, `gene_selection`, `pipeline`, ...).

**What lives here.**
- `test_viz_utils_module_refactor` — the consolidated coverage for the viz utilities (largest file; centralizes what used to be several modules).
- `test_visualize_capture_anchor`, `test_visualize_compositional_plots` — `viz.pipeline` config detection / branch gating.
- `test_corner_ppc`, `test_compositional_ppc` — corner-plot and compositional PPC rendering.
- `test_ppc_dataset`, `test_ppc_explicit_genes` — per-dataset PPC subsetting and explicit gene selection in plotting.

**What does NOT live here.**
- The PPC *sampling* that produces the draws being plotted → `tests/sampling/`.
- The `scribe visualize` **CLI** entry point → `tests/cli/` (`test_scribe_visualize_cli`).

**Key fixtures.** Root `tests/conftest.py`. `test_compositional_ppc` uses the
shared `_nbln_result` factory from `tests/_synthetic_results.py`. No folder-local conftest.
