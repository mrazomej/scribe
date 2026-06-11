"""Tests for the configurable ``n_quad_nodes`` on the two-state likelihood.

``n_quad_nodes`` controls the number of Gauss-Legendre quadrature nodes
used by the two-state Poisson-Beta likelihood (``PoissonBetaCompound``).
It is exposed through ``scribe.fit(...)`` and threaded down to every
``PoissonBetaCompound`` construction in the SVI ``twostate`` /
``twostatevcp`` likelihoods.

Validates:

- ``scribe.fit(..., n_quad_nodes=120)`` runs end-to-end on a tiny
  ``twostatevcp`` model and returns a result.
- The configured value actually propagates to the constructed
  ``PoissonBetaCompound`` distribution (via a numpyro trace).
- The default (``n_quad_nodes`` unset / ``None``) keeps the historical
  60-node behavior, so existing fits stay bit-identical.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace

from scribe.models import ModelConfigBuilder, get_model_and_guide


N_CELLS = 8
N_GENES = 4


def _make_counts(seed_val=0, n_cells=N_CELLS, n_genes=N_GENES):
    rng = np.random.default_rng(seed_val)
    per_gene = np.linspace(2.0, 50.0, n_genes)
    counts = np.stack(
        [rng.poisson(m, n_cells) for m in per_gene], axis=1
    )
    return jnp.asarray(counts, dtype=jnp.int32)


def _build_cfg(model="twostate", n_quad_nodes=None):
    """Build a moment_delta two-state config with an optional node count."""
    builder = (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization("two_state_moment_delta")
        .with_inference("svi")
        .unconstrained()
    )
    if model == "twostatevcp":
        builder = builder.with_priors(p_capture=(1.0, 1.0))
    builder._n_quad_nodes = n_quad_nodes
    return builder.build()


def _trace_counts_dist(cfg, counts):
    """Run a single seeded trace and return the ``counts`` site fn."""
    model, _, cfg_full = get_model_and_guide(cfg)
    tr = trace(seed(model, jax.random.PRNGKey(0))).get_trace(
        n_cells=counts.shape[0],
        n_genes=counts.shape[1],
        model_config=cfg_full,
        counts=counts,
    )
    return tr["counts"]["fn"]


def _unwrap_poisson_beta(fn):
    """Peel ``Independent`` (``.to_event(1)``) to the PoissonBetaCompound."""
    base = getattr(fn, "base_dist", fn)
    # Defensive: handle double-wrapping if present.
    base = getattr(base, "base_dist", base)
    return base


# ==============================================================================
# Propagation: configured value reaches the PoissonBetaCompound
# ==============================================================================


class TestNQuadNodesPropagation:
    def test_config_field_default_none(self):
        cfg = _build_cfg("twostate", n_quad_nodes=None)
        assert cfg.n_quad_nodes is None

    def test_config_field_set(self):
        cfg = _build_cfg("twostate", n_quad_nodes=120)
        assert cfg.n_quad_nodes == 120

    def test_propagates_to_distribution(self):
        """A non-default value reaches the constructed distribution."""
        counts = _make_counts()
        cfg = _build_cfg("twostate", n_quad_nodes=120)
        base = _unwrap_poisson_beta(_trace_counts_dist(cfg, counts))
        assert hasattr(base, "n_quad_nodes")
        assert base.n_quad_nodes == 120

    def test_default_is_sixty(self):
        """Unset n_quad_nodes keeps the historical 60-node default."""
        counts = _make_counts()
        cfg = _build_cfg("twostate", n_quad_nodes=None)
        base = _unwrap_poisson_beta(_trace_counts_dist(cfg, counts))
        assert base.n_quad_nodes == 60

    def test_vcp_propagates_to_distribution(self):
        """The VCP build path also threads the configured node count."""
        counts = _make_counts()
        cfg = _build_cfg("twostatevcp", n_quad_nodes=120)
        base = _unwrap_poisson_beta(_trace_counts_dist(cfg, counts))
        assert base.n_quad_nodes == 120


# ==============================================================================
# End-to-end fit through the public API
# ==============================================================================


class TestNQuadNodesFit:
    def test_fit_runs_with_custom_n_quad_nodes(self):
        import scribe

        counts = _make_counts(n_cells=40)
        res = scribe.fit(
            counts,
            model="twostatevcp",
            parameterization="moment_delta",
            inference_method="svi",
            n_steps=5,
            unconstrained=True,
            n_quad_nodes=120,
        )
        assert res is not None
        assert jnp.isfinite(res.loss_history[-1])
        # The configured node count is persisted on the result's config.
        assert res.model_config.n_quad_nodes == 120

    def test_fit_signature_exposes_n_quad_nodes(self):
        import inspect

        import scribe

        sig = inspect.signature(scribe.fit)
        assert "n_quad_nodes" in sig.parameters
