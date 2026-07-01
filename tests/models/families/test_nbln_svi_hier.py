"""Hierarchical gene-gene correlation (Rung 1.5) on the NB-LogNormal SVI/VAE path.

These tests exercise the *first-class* SVI implementation of the per-leaf
module-weight hierarchy (declared via ``priors={"module_weight": {...}}`` on a
grouped fit):

- the model samples the global per-leaf module weights ``s`` and makes the
  K-dim VAE latent prior leaf-specific (``z_c ~ Normal(0, diag(s_{sigma(c)}^2))``);
- the guide registers the matching mean-field block for ``s``;
- model and guide latent sites pair, so the production ``TraceMeanField_ELBO``
  runs end-to-end without shape or pairing errors;
- the hierarchy is correctly *inert* for single-leaf or non-grouped fits.

The shared additive module-weight primitive itself is unit-tested separately in
``tests/models/components/test_module_weights.py`` (and its operator builder in
``tests/models/components/test_module_weight_factors.py``); here we test the
*factory wiring* into the real model/guide builders.

A single ``donor`` base factor (identity leaf gather, gaussian ``module_weight``
family, random effect) reproduces the old flat program-scale behaviour and
takes the fast path -- so the per-factor site names carry the ``__donor``
suffix (``module_weight_raw__donor`` etc.).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import numpyro
import pytest
from jax import random
from numpyro.infer import SVI, TraceMeanField_ELBO

from scribe.models.config import ModelConfigBuilder
from scribe.models.config.grouping import (
    Factor,
    GroupingSpec,
    PriorFamilySpec,
)
from scribe.models.presets.factory import create_model
from scribe.models.components.guide_families import VAELatentGuide


def _vae_guide(param_specs):
    """Return the VAELatentGuide guide-family carried by the param specs."""
    for s in param_specs:
        gf = getattr(s, "guide_family", None)
        if isinstance(gf, VAELatentGuide):
            return gf
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _donor_grouping_spec(n_datasets: int) -> GroupingSpec:
    """A single ``donor`` base factor carrying a gaussian ``module_weight`` prior.

    Identity ``leaf_to_level`` (one level per leaf) + a random gaussian
    ``module_weight`` family reproduces the flat per-leaf module-weight
    hierarchy (the single-factor fast path). The factor name ``"donor"`` becomes
    the per-factor site suffix (``module_weight_raw__donor`` etc.).
    """
    donors = [f"D{i}" for i in range(n_datasets)]
    donor = Factor(
        name="donor",
        kind="base",
        nested_in=None,
        effect_type="random",
        fixed_scale=None,
        levels=tuple(donors),
        leaf_to_level=tuple(range(n_datasets)),
        priors={"module_weight": PriorFamilySpec(type="gaussian")},
    )
    return GroupingSpec(
        factors=(donor,), leaf_labels=tuple(donors), n_leaves=n_datasets
    )


def _nbln_hier_config(
    *,
    n_datasets: int,
    latent_dim: int = 3,
    module_weight_hierarchy: bool = True,
    d_mode: str = "low_rank",
):
    """Build a tiny grouped ``nbln`` VAE config for hierarchy smoke tests.

    Parameters
    ----------
    n_datasets : int
        Number of donors/leaves to register on the config.
    latent_dim : int, optional
        VAE latent dimensionality ``K`` (number of regulatory modules).
    module_weight_hierarchy : bool, optional
        When ``True`` (default) attach a ``donor`` grouping factor whose
        ``module_weight`` prior family is gaussian, enabling the per-leaf
        module-weight hierarchy. When ``False`` no grouping is attached, so the
        hierarchy is disabled (the migrated replacement for the old
        flat-hierarchy-off case).
    d_mode : str, optional
        Diagonal mode (``"low_rank"`` or ``"learned"``).
    """
    built = (
        ModelConfigBuilder()
        .for_model("nbln")
        .with_parameterization("count_lognormal")
        .with_inference("vae")
        .with_vae(
            latent_dim=latent_dim,
            encoder_hidden_dims=[16],
            decoder_hidden_dims=[16],
        )
        .build()
    )
    update = {
        "d_mode": d_mode,
        "n_datasets": n_datasets,
    }
    if module_weight_hierarchy:
        # The module-weight hierarchy is declared through the grouping: a
        # ``donor`` factor carrying a gaussian ``module_weight`` prior family.
        update["grouping_spec"] = _donor_grouping_spec(n_datasets)
    # ``model_copy`` bypasses field validators (so ``n_datasets < 2`` and the
    # leaf-count check are not re-run), matching the old test's use of it.
    return built.model_copy(update=update)


def _dataset_indices(n_cells: int, n_datasets: int) -> jnp.ndarray:
    """Round-robin per-cell donor index hitting every donor at least once."""
    return jnp.arange(n_cells, dtype=jnp.int32) % n_datasets


# ---------------------------------------------------------------------------
# Model-side wiring
# ---------------------------------------------------------------------------


def test_model_samples_module_weight_sites():
    """The model emits the per-leaf module-weight block with documented shapes."""
    n_cells, n_genes, D, K = 12, 8, 3, 3
    config = _nbln_hier_config(n_datasets=D, latent_dim=K)
    model, _guide, _specs = create_model(config, n_genes=n_genes, validate=False)

    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    tr = numpyro.handlers.trace(
        numpyro.handlers.seed(model, random.PRNGKey(1))
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
        dataset_indices=_dataset_indices(n_cells, D),
    )

    assert tr["module_weight_tau_raw__donor"]["value"].shape == ()
    assert tr["module_weight_raw__donor"]["value"].shape == (D, K)
    assert tr["module_weight"]["value"].shape == (D, K)
    # Scale gauge: sum over leaves of log s is ~0 per module.
    log_s = np.asarray(tr["module_weight_log"]["value"])
    npt.assert_allclose(log_s.sum(axis=0), np.zeros(K), atol=1e-5)
    assert "z" in tr and "counts" in tr


def test_latent_prior_is_leaf_specific():
    """The z prior scale equals s gathered by each cell's donor index."""
    n_cells, n_genes, D, K = 12, 8, 4, 3
    config = _nbln_hier_config(n_datasets=D, latent_dim=K)
    model, _guide, _specs = create_model(config, n_genes=n_genes, validate=False)

    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    ds = _dataset_indices(n_cells, D)
    tr = numpyro.handlers.trace(
        numpyro.handlers.seed(model, random.PRNGKey(2))
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
        dataset_indices=ds,
    )

    # The z prior is Normal(0, s_cell).to_event(1) -> an Independent whose
    # base Normal scale must equal the per-cell gathered module weights.
    z_fn = tr["z"]["fn"]
    base = getattr(z_fn, "base_dist", z_fn)
    prior_scale = np.asarray(base.scale)  # (n_cells, K)
    expected = np.asarray(tr["module_weight"]["value"])[np.asarray(ds)]
    npt.assert_allclose(prior_scale, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Inert when not applicable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_datasets=1, module_weight_hierarchy=True),  # D < 2
        dict(n_datasets=3, module_weight_hierarchy=False),  # disabled
    ],
    ids=["single_dataset", "disabled"],
)
def test_hierarchy_inert_when_not_applicable(kwargs):
    """No module-weight sites when D < 2 or the hierarchy is disabled."""
    n_cells, n_genes, K = 10, 8, 3
    config = _nbln_hier_config(latent_dim=K, **kwargs)
    model, _guide, _specs = create_model(config, n_genes=n_genes, validate=False)

    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    ds = _dataset_indices(n_cells, max(kwargs["n_datasets"], 1))
    tr = numpyro.handlers.trace(
        numpyro.handlers.seed(model, random.PRNGKey(3))
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
        dataset_indices=ds,
    )
    assert "module_weight" not in tr
    assert "module_weight_raw__donor" not in tr
    # The base model still works (z + counts present).
    assert "z" in tr and "counts" in tr


def test_inert_when_dataset_indices_missing():
    """No module-weight sites when dataset_indices is None even if requested."""
    n_cells, n_genes, K = 10, 8, 3
    config = _nbln_hier_config(n_datasets=3, latent_dim=K)
    model, _guide, _specs = create_model(config, n_genes=n_genes, validate=False)
    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    tr = numpyro.handlers.trace(
        numpyro.handlers.seed(model, random.PRNGKey(4))
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
        dataset_indices=None,
    )
    assert "module_weight" not in tr


# ---------------------------------------------------------------------------
# Model/guide pairing + end-to-end TraceMeanField_ELBO
# ---------------------------------------------------------------------------


def test_model_guide_latent_sites_pair():
    """Every model latent has a matching guide site (incl. module weights)."""
    n_cells, n_genes, D, K = 12, 8, 3, 3
    config = _nbln_hier_config(n_datasets=D, latent_dim=K)
    model, guide, _specs = create_model(config, n_genes=n_genes, validate=False)
    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    ds = _dataset_indices(n_cells, D)

    common = dict(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
        dataset_indices=ds,
    )
    model_tr = numpyro.handlers.trace(
        numpyro.handlers.seed(model, random.PRNGKey(5))
    ).get_trace(**common)
    guide_tr = numpyro.handlers.trace(
        numpyro.handlers.seed(guide, random.PRNGKey(5))
    ).get_trace(**common)

    model_latents = {
        n
        for n, s in model_tr.items()
        if s["type"] == "sample" and not s.get("is_observed", False)
    }
    guide_latents = {
        n for n, s in guide_tr.items() if s["type"] == "sample"
    }
    mw_latents = {"module_weight_tau_raw__donor", "module_weight_raw__donor"}
    # Both module-weight latent sites must be present in each trace and pair.
    assert mw_latents <= model_latents
    assert mw_latents <= guide_latents
    # Every model latent must be covered by the guide (mean-field requirement).
    assert model_latents <= guide_latents, (
        f"model latents not covered by guide: {model_latents - guide_latents}"
    )


def test_trace_mean_field_elbo_runs_under_jit():
    """A few production-ELBO SVI steps complete with finite loss, *under jit*.

    This is the load-bearing integration check. Two things must hold and both
    only fail on the real engine path:

    1. The leaf-specific latent prior makes ``z``'s prior depend on the global
       module-weight latent, and the production loss is ``TraceMeanField_ELBO``
       -- a wrong dependency structure / shape would raise.
    2. The production engine wraps the update in ``jax.jit`` (``body_fn``), so
       any ``numpyro.param`` init that calls ``float(jnp...)`` would raise a
       ``ConcretizationTypeError`` *only* under jit. We therefore jit the
       update here (a bare ``svi.update`` loop would miss that class of bug).
    """
    import jax

    n_cells, n_genes, D, K = 24, 10, 3, 3
    config = _nbln_hier_config(n_datasets=D, latent_dim=K)
    model, guide, _specs = create_model(config, n_genes=n_genes, validate=False)
    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    ds = _dataset_indices(n_cells, D)

    svi = SVI(
        model,
        guide,
        numpyro.optim.Adam(1e-3),
        loss=TraceMeanField_ELBO(),
    )
    init_args = dict(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
        dataset_indices=ds,
    )
    state = svi.init(random.PRNGKey(6), **init_args)

    # Mirror the engine's ``jit(body_fn)``: jit the update over the state, with
    # the (static) model args captured in the closure.
    @jax.jit
    def step(svi_state):
        return svi.update(svi_state, **init_args)

    losses = []
    for _ in range(5):
        state, loss = step(state)
        losses.append(float(loss))

    assert all(np.isfinite(losses)), f"non-finite ELBO: {losses}"


# ---------------------------------------------------------------------------
# Leaf-covariate encoder (encoder conditions on leaf; decoder does NOT)
# ---------------------------------------------------------------------------


def test_encoder_is_leaf_covariate_aware_decoder_is_not():
    """With the hierarchy on, the ENCODER gets a leaf covariate; decoder stays
    leaf-free (the Rung-1 identifiability requirement)."""
    D, K = 4, 3
    config = _nbln_hier_config(n_datasets=D, latent_dim=K)
    _model, _guide, specs = create_model(config, n_genes=8, validate=False)
    vg = _vae_guide(specs)
    assert vg is not None, "no VAELatentGuide found in param specs"

    enc_specs = vg.encoder.covariate_specs
    assert enc_specs and len(enc_specs) == 1, "encoder should have one covariate"
    assert enc_specs[0].name == "leaf"
    assert enc_specs[0].num_categories == D
    # The decoder MUST remain leaf-free so s is the sole carrier of
    # between-leaf module activity.
    assert getattr(vg.decoder, "covariate_specs", None) in (None, [], ())


def test_encoder_has_no_covariate_when_hierarchy_off():
    """No covariate on the encoder when the hierarchy is disabled."""
    config = _nbln_hier_config(
        n_datasets=4, latent_dim=3, module_weight_hierarchy=False
    )
    _model, _guide, specs = create_model(config, n_genes=8, validate=False)
    vg = _vae_guide(specs)
    assert getattr(vg.encoder, "covariate_specs", None) in (None, [], ())


def test_leaf_covariate_embedding_params_materialize():
    """The encoder's leaf-embedding table is actually created during init.

    Guards against the covariate being silently skipped (e.g. if init never
    saw the covariates): we assert a ``cov_embed`` param appears somewhere
    inside the fitted encoder params.
    """
    n_cells, n_genes, D, K = 24, 10, 3, 3
    config = _nbln_hier_config(n_datasets=D, latent_dim=K)
    model, guide, _specs = create_model(config, n_genes=n_genes, validate=False)
    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    ds = _dataset_indices(n_cells, D)

    svi = SVI(model, guide, numpyro.optim.Adam(1e-3), loss=TraceMeanField_ELBO())
    state = svi.init(
        random.PRNGKey(7),
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
        dataset_indices=ds,
    )
    params = svi.get_params(state)

    # Flatten nested param pytrees to leaf-path strings and look for cov_embed.
    import jax

    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    keys = [jax.tree_util.keystr(p) for p, _ in flat]
    assert any("cov_embed" in k for k in keys), (
        "encoder leaf-covariate embedding params not found; the covariate "
        f"was not initialized. param leaves: {keys}"
    )
