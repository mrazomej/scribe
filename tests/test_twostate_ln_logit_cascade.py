"""Cascade adapter end-to-end tests: TwoState SVI → TSLN-Logit Laplace.

Complements ``test_twostate_ln_rate_cascade.py`` which already covers
the coord-map and primary/fallback path unit tests for both rate AND
logit variants.  This file focuses on the **end-to-end** SVI → Laplace
flow for the logit variant — fitting a small synthetic TwoState SVI,
feeding the result through ``_run_laplace_inference`` with
``informative_priors_from``, and checking that the result is
well-formed.

Tests in scope
--------------

1. End-to-end cascade smoke: TwoState SVI fit → TSLN-Logit Laplace fit
   via ``_run_laplace_inference`` with the cascade source.

2. End-to-end via the high-level ``scribe.fit`` API — verifies the
   API plumbing (``model_flags`` validation, freeze defaults,
   cascade dispatch through ``api/stages/run_inference``).

3. Variant rejection: cascade adapter rejects unknown
   ``target_variant``.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _make_synthetic_counts(C=40, G=12, seed=0):
    rng = np.random.default_rng(seed)
    # Modest dispersion + capture variation so the SVI has something
    # to fit.  Lambda per gene ~ Gamma(1, 1) * 3.
    gene_mean = rng.gamma(2.0, 1.0, size=G).astype(np.float32) * 3.0
    counts = np.column_stack(
        [rng.poisson(m, size=C) for m in gene_mean]
    ).astype(np.float32)
    return jnp.asarray(counts), gene_mean


# ---------------------------------------------------------------------
# Test 1: end-to-end cascade via _run_laplace_inference
# ---------------------------------------------------------------------


def test_cascade_end_to_end_via_bridge():
    """Fit a small TwoState SVI, then cascade into TSLN-Logit Laplace.

    Verifies:
      - The cascade-priors adapter consumes the SVI's effective
        deterministics (``alpha``, ``beta``, ``r_hat``, ``eta_act``).
      - The Laplace fit completes with finite loss and finite final
        gradient norms.
      - The resulting fit has the expected freeze flags
        (L4 default for TSLN-Logit: ``rate``, ``kappa``, ``eta_anchor``
        all frozen).
      - The frozen values surface on ``ScribeLaplaceResults`` as the
        constrained MAPs.
    """
    import scribe

    counts, _ = _make_synthetic_counts(C=40, G=12, seed=0)

    # --- Step 1: fit a TwoState SVI source --------------------------
    svi_result = scribe.fit(
        counts,
        model="twostate",
        inference_method="svi",
        parameterization="moment_delta",
        n_steps=120,
        unconstrained=True,
        positive_transform="softplus",
    )

    # --- Step 2: cascade into a TSLN-Logit Laplace fit --------------
    # ``informative_priors_verbose=False`` to keep the test output
    # quiet; the cascade adapter still asserts that ``alpha`` /
    # ``beta`` / ``r_hat`` / ``eta_act`` are present internally (it
    # raises ``ValueError`` if any are missing).
    laplace_result = scribe.fit(
        counts,
        model="twostate_ln_logit",
        inference_method="laplace",
        positive_transform="softplus",
        informative_priors_from=svi_result,
        informative_priors_n_samples=50,
        informative_priors_verbose=False,
        n_steps=8,
        vae_latent_dim=3,
        laplace_config={"n_newton_steps": 3, "batch_size": 10},
    )

    # Result is well-formed.
    assert laplace_result.model_config.base_model == "twostate_ln_logit"
    assert laplace_result.rate.shape == (12,)
    assert laplace_result.kappa.shape == (12,)
    assert laplace_result.eta_anchor.shape == (12,)

    # Default L4 freeze: rate, kappa, eta_anchor all hard-frozen.
    for k in ("rate", "kappa", "eta_anchor"):
        assert k in laplace_result.frozen_params, (
            f"Default L4 cascade must freeze {k!r}; "
            f"got frozen_params={laplace_result.frozen_params}"
        )

    # Frozen values lifted into the constrained MAP.
    assert jnp.all(jnp.isfinite(laplace_result.rate))
    assert jnp.all(laplace_result.rate > 0)
    assert jnp.all(jnp.isfinite(laplace_result.kappa))
    assert jnp.all(laplace_result.kappa > 0)
    assert jnp.all(jnp.isfinite(laplace_result.eta_anchor))

    # Loss bounded; Newton converged.
    assert jnp.all(jnp.isfinite(laplace_result.losses))
    assert jnp.all(jnp.isfinite(laplace_result.final_grad_norms))

    # Cascade source plumbed through.
    assert laplace_result.cascade_source is svi_result


# ---------------------------------------------------------------------
# Test 2: cascade rejects unknown variants
# ---------------------------------------------------------------------


def test_priors_from_twostate_results_unknown_variant():
    """``target_variant`` outside ``{"rate", "logit"}`` raises ValueError."""
    from scribe.laplace.priors import (
        priors_from_twostate_results,
        freeze_values_from_twostate_results,
    )

    class StubResult:
        n_genes = 3

        def get_posterior_samples(self, **_):
            return {}

        def get_map(self, **_):
            return {}

    with pytest.raises(ValueError, match="Unknown target_variant"):
        priors_from_twostate_results(
            results=StubResult(),
            target_positive_transform="softplus",
            target_n_genes=3,
            target_n_cells=5,
            target_variant="not_a_variant",
            verbose=False,
        )
    with pytest.raises(ValueError, match="Unknown target_variant"):
        freeze_values_from_twostate_results(
            results=StubResult(),
            target_positive_transform="softplus",
            target_n_genes=3,
            target_n_cells=5,
            target_variant="not_a_variant",
            verbose=False,
        )


# ---------------------------------------------------------------------
# Test 3: capture auto-routing on cascade with VCP source
# ---------------------------------------------------------------------


def test_cascade_with_capture_auto_freezes_eta():
    """When the SVI source has per-cell capture (``twostatevcp``),
    the run-inference stage auto-adds ``"eta"`` to ``freeze_params``
    so the cascade routes through ``x_only_offset`` (Rev 4 invariant).
    """
    import scribe

    counts, _ = _make_synthetic_counts(C=30, G=12, seed=1)

    # Fit a TwoStateVCP SVI source — exposes p_capture.
    svi_result = scribe.fit(
        counts,
        model="twostatevcp",
        inference_method="svi",
        parameterization="moment_delta",
        n_steps=120,
        unconstrained=True,
        positive_transform="softplus",
        priors={
            "capture_efficiency": (float(np.log(15_000.0)), 0.1),
        },
    )

    # Cascade into TSLN-Logit.  User passes the default freeze (no
    # explicit "eta" in informative_priors_freeze); the api stage
    # MUST auto-add "eta" because the cascade has capture.
    laplace_result = scribe.fit(
        counts,
        model="twostate_ln_logit",
        inference_method="laplace",
        positive_transform="softplus",
        informative_priors_from=svi_result,
        informative_priors_n_samples=30,
        informative_priors_verbose=False,
        n_steps=5,
        vae_latent_dim=3,
        laplace_config={"n_newton_steps": 3, "batch_size": 10},
    )
    # The Rev 4 safeguard must have promoted ``"eta"`` into
    # freeze_params, so the result reflects fixed-offset capture.
    assert "eta" in laplace_result.frozen_params, (
        "TSLN-Logit cascade with VCP source must auto-freeze 'eta' "
        "to route through x_only_offset (Rev 4)."
    )
    assert laplace_result.eta_loc is not None
    assert jnp.all(jnp.isfinite(laplace_result.eta_loc))
