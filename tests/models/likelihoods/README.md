# tests/models/likelihoods

**Purpose.** Likelihood functions and their numerical correctness — the
`log_prob` kernels and the guards/floors around them.

**Source under test.** `src/scribe/models/components/likelihoods` (`_log_prob`,
`negative_binomial`, `zero_inflated`, `two_state`, `lnm`, `pln`, `vcp`,
`beta_negative_binomial`, `base`).

**What lives here.**
- `test_twostate_likelihood`, `test_lnm_likelihood`, `test_pln_likelihood` — per-family `log_prob` correctness.
- `test_log_likelihood_parity` — pins numerical output against a checked-in golden `.npz` (loaded via the `data_dir` fixture).
- `test_floor` — numerical-floor protection in the NB / VCP likelihoods.
- `test_numpyro_0201_mixture_compat` — mixture-likelihood construction compatibility.
- `test_alr_reference_selection` — ALR-reference gene selection in the LNM likelihood.

**What does NOT live here.**
- End-to-end fits that *recover* these models → `../families/`.
- Parameterization classes → `../parameterizations/`.
- LNM/PLN/NBLN **Laplace** observation models → `tests/laplace/`.

**Key fixtures.** Root `tests/conftest.py`; `test_log_likelihood_parity` uses the session-scoped `data_dir` fixture for the golden file. No folder-local conftest.
