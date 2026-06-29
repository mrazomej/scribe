# tests/sampling

**Purpose.** Posterior-predictive and denoising *sampling* (the draw-generating
layer, not the plotting layer).

**Source under test.** `src/scribe/sampling` (`sample_posterior_ppc`,
`sample_biological_nb`, `denoise_counts`, `_denoising`, `_denoising_twostate`).

**What lives here.**
- `test_biological_ppc` — biological (denoised) NB posterior-predictive sampling.
- `test_denoising` — Bayesian denoising of observed profiles (standard path).
- `test_denoising_twostate` — denoising under the two-state (Poisson–Beta) model.
- `test_lnm_conditional_ppc` — conditional predictive sampling for the LNM model.

**What does NOT live here.**
- PPC *plotting* (corner, compositional, per-dataset PPC panels) → `tests/viz/`.
- Gene/likelihood definitions the samplers draw from → `tests/models/`.

**Key fixtures.** Root `tests/conftest.py`. No folder-local conftest.
