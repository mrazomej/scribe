"""Parity tests for the constructor-based :meth:`Likelihood.log_prob`.

These tests pin the numerical output of the new
``Likelihood.log_prob`` methods (added during the constructor-migration of
the log-likelihood evaluation code) against a golden ``.npz`` captured
from the legacy free-function implementation that previously lived at
``scribe.models.log_likelihood``.

The golden file lives at ``tests/data/log_likelihood_golden.npz``.  It is
a pinned, checked-in numerical regression: the legacy module has been
deleted (final step of the migration) and the ``.npz`` is now the
authoritative reference for the historical legacy behaviour.

Coverage
--------
For every one of the eight NB-family model variants (non-mixture and
mixture, NB and ZINB, VCP and non-VCP) we check:

* ``return_by`` in ``{"cell", "gene"}``.
* Mixture variants additionally iterate ``split_components`` in
  ``{False, True}`` and weighting configurations (no weights, and
  additive gene weights on the cell-branch only - the legacy code never
  supported any other weighted configuration for this layout).

Rationale for a single consolidated file rather than one-per-model: the
parity logic is identical across all eight model classes; spreading the
assertions eight ways would duplicate >90% of the code.  The per-model
``tests/test_<model>.py`` files already cover functional behaviour such
as shape checks, SVI/MCMC integration, and API ergonomics.
"""

# Standard library
from pathlib import Path

# Scientific stack
import jax.numpy as jnp
import numpy as np
import pytest

# Likelihood subclasses that expose the new ``.log_prob`` method.
from scribe.models.components.likelihoods import (
    NegativeBinomialLikelihood,
    ZeroInflatedNBLikelihood,
    NBWithVCPLikelihood,
    ZINBWithVCPLikelihood,
    BetaNegativeBinomialLikelihood,
    ZeroInflatedBNBLikelihood,
    BNBWithVCPLikelihood,
    ZIBNBWithVCPLikelihood,
)


# =========================================================================
# Shared fixture: load the golden .npz once per test session.
# =========================================================================

# Path to the pinned golden file.  The ``.npz`` packs both the synthetic
# inputs (counts, parameter tensors, weighting arrays) and the legacy
# outputs against which we assert parity.
_GOLDEN_PATH = Path(__file__).parent / "data" / "log_likelihood_golden.npz"


@pytest.fixture(scope="session")
def golden():
    """Load the log-likelihood golden ``.npz`` once for the session.

    Returns
    -------
    dict
        ``{name: np.ndarray}`` mapping of every array bundled inside
        ``log_likelihood_golden.npz`` (inputs + legacy outputs).
    """
    # ``np.load`` with ``allow_pickle=False`` keeps the file structure
    # strictly numerical - a security-conscious default for checked-in
    # regression artefacts.
    data = np.load(_GOLDEN_PATH, allow_pickle=False)
    return {key: data[key] for key in data.files}


# =========================================================================
# Helpers that reconstruct ``params`` dicts from the flattened .npz keys.
# =========================================================================

# Parameters each model variant consumes.  Keys mirror the flattening
# convention used when serialising to ``.npz`` (``"params_<model>__<name>"``).
_PARAM_KEYS = {
    "nbdm": ("p", "r"),
    "zinb": ("p", "r", "gate"),
    "nbvcp": ("p", "r", "p_capture"),
    "zinbvcp": ("p", "r", "gate", "p_capture"),
    "nbdm_mix": ("p", "r", "mixing_weights"),
    "zinb_mix": ("p", "r", "gate", "mixing_weights"),
    "nbvcp_mix": ("p", "r", "p_capture", "mixing_weights"),
    "zinbvcp_mix": ("p", "r", "gate", "p_capture", "mixing_weights"),
}

# Concrete ``Likelihood`` subclass for each model variant.  The mixture
# variants reuse the same classes because the ``.log_prob`` method branches
# on the presence of ``"mixing_weights"`` in ``params``.
_LIKELIHOOD_CLASS = {
    "nbdm": NegativeBinomialLikelihood,
    "zinb": ZeroInflatedNBLikelihood,
    "nbvcp": NBWithVCPLikelihood,
    "zinbvcp": ZINBWithVCPLikelihood,
    "nbdm_mix": NegativeBinomialLikelihood,
    "zinb_mix": ZeroInflatedNBLikelihood,
    "nbvcp_mix": NBWithVCPLikelihood,
    "zinbvcp_mix": ZINBWithVCPLikelihood,
}


def _build_params(golden, model: str):
    """Reconstruct the posterior-parameter dict for one model variant.

    Parameters
    ----------
    golden : dict
        The loaded golden payload.
    model : str
        Model-variant key (e.g., ``"nbdm"``, ``"zinb_mix"``).

    Returns
    -------
    dict
        Mapping of parameter name to ``jnp.ndarray``.
    """
    return {
        pname: jnp.asarray(golden[f"params_{model}__{pname}"])
        for pname in _PARAM_KEYS[model]
    }


# =========================================================================
# Non-mixture parity tests (8 cases = 4 models x 2 return_by axes).
# =========================================================================


@pytest.mark.parametrize(
    "model",
    ["nbdm", "zinb", "nbvcp", "zinbvcp"],
)
@pytest.mark.parametrize("return_by", ["cell", "gene"])
def test_non_mixture_log_prob_parity(golden, model, return_by):
    """``Likelihood.log_prob`` matches the legacy free function exactly.

    Parameters
    ----------
    golden : dict
        Session-scoped fixture delivering the pinned numerical reference.
    model : str
        One of ``{"nbdm", "zinb", "nbvcp", "zinbvcp"}``.
    return_by : str
        Reduction axis the legacy function was invoked with.
    """
    # Instantiate the likelihood and rebuild the parameter dict.
    likelihood = _LIKELIHOOD_CLASS[model]()
    counts = jnp.asarray(golden["counts"])
    params = _build_params(golden, model)

    # New implementation under test.
    new_output = np.asarray(
        likelihood.log_prob(counts, params, return_by=return_by)
    )

    # Legacy reference pinned at generation time.
    golden_key = f"{model}__rb-{return_by}"
    golden_output = golden[golden_key]

    # Exact parity is achievable because the two implementations execute
    # identical algebra (no loop-induced reassociation of floating-point
    # sums); we still use atol=1e-5 as a defensive tolerance.
    assert np.allclose(new_output, golden_output, atol=1e-5), (
        f"Parity failure for {model} rb={return_by}: "
        f"max|diff|={np.max(np.abs(new_output - golden_output)):.6e}"
    )


# =========================================================================
# Mixture parity tests - full grid of (return_by, split_components,
# weight config).  The ``split_components`` axis exercises both the
# marginalised (logsumexp) and per-component return paths.
# =========================================================================


@pytest.mark.parametrize(
    "model",
    ["nbdm_mix", "zinb_mix", "nbvcp_mix", "zinbvcp_mix"],
)
@pytest.mark.parametrize("return_by", ["cell", "gene"])
@pytest.mark.parametrize("split_components", [False, True])
def test_mixture_log_prob_parity_no_weights(
    golden, model, return_by, split_components
):
    """Unweighted mixture parity across ``return_by`` and ``split_components``.

    Covers the most common usage of the mixture log likelihood: no
    per-gene / per-cell weighting, both reduction axes, and both the
    marginalised and per-component outputs.
    """
    likelihood = _LIKELIHOOD_CLASS[model]()
    counts = jnp.asarray(golden["counts"])
    params = _build_params(golden, model)

    new_output = np.asarray(
        likelihood.log_prob(
            counts,
            params,
            return_by=return_by,
            split_components=split_components,
        )
    )

    golden_key = (
        f"{model}__rb-{return_by}__sc-{int(split_components)}__w-none"
    )
    golden_output = golden[golden_key]

    assert np.allclose(new_output, golden_output, atol=1e-5), (
        f"Parity failure for {model} rb={return_by} "
        f"sc={split_components} (no weights): "
        f"max|diff|={np.max(np.abs(new_output - golden_output)):.6e}"
    )


@pytest.mark.parametrize(
    "model",
    ["nbdm_mix", "zinb_mix", "nbvcp_mix", "zinbvcp_mix"],
)
@pytest.mark.parametrize("split_components", [False, True])
def test_mixture_log_prob_parity_additive_gene_weights(
    golden, model, split_components
):
    """Additive per-gene weights parity on the cell-reduction branch.

    The legacy implementation only supports additive weights on the
    ``return_by="cell"`` branch - the gene-branch path relies on a
    broadcast that fails for the canonical
    ``(n_cells, n_genes, n_components)`` layout and was never exercised.
    See ``_generate_log_likelihood_golden.py`` for the full rationale.
    """
    likelihood = _LIKELIHOOD_CLASS[model]()
    counts = jnp.asarray(golden["counts"])
    params = _build_params(golden, model)
    weights = jnp.asarray(golden["weights_gene"])

    new_output = np.asarray(
        likelihood.log_prob(
            counts,
            params,
            return_by="cell",
            split_components=split_components,
            weights=weights,
            weight_type="additive",
        )
    )

    golden_key = (
        f"{model}__rb-cell__sc-{int(split_components)}__w-add"
    )
    golden_output = golden[golden_key]

    assert np.allclose(new_output, golden_output, atol=1e-5), (
        f"Parity failure for {model} rb=cell sc={split_components} "
        f"(additive weights): "
        f"max|diff|={np.max(np.abs(new_output - golden_output)):.6e}"
    )


# =========================================================================
# Shape / dtype sanity checks (not parity, but catch trivial regressions).
# =========================================================================


@pytest.mark.parametrize(
    "model,expected_cell_shape",
    [
        ("nbdm", (50,)),
        ("zinb", (50,)),
        ("nbvcp", (50,)),
        ("zinbvcp", (50,)),
    ],
)
def test_non_mixture_output_shape(golden, model, expected_cell_shape):
    """Non-mixture ``return_by="cell"`` output has shape ``(n_cells,)``."""
    likelihood = _LIKELIHOOD_CLASS[model]()
    counts = jnp.asarray(golden["counts"])
    params = _build_params(golden, model)
    out = likelihood.log_prob(counts, params, return_by="cell")
    assert out.shape == expected_cell_shape


@pytest.mark.parametrize(
    "model",
    ["nbdm_mix", "zinb_mix", "nbvcp_mix", "zinbvcp_mix"],
)
def test_mixture_split_components_shape(golden, model):
    """``split_components=True`` on cell-branch yields (n_cells, n_components)."""
    likelihood = _LIKELIHOOD_CLASS[model]()
    counts = jnp.asarray(golden["counts"])
    params = _build_params(golden, model)
    out = likelihood.log_prob(
        counts, params, return_by="cell", split_components=True
    )
    # N_CELLS=50, N_COMPONENTS=3 - see _generate_log_likelihood_golden.py.
    assert out.shape == (50, 3)


# =========================================================================
# BNB family smoke tests
# =========================================================================

# No legacy free functions exist for the BNB family, so parity against the
# checked-in ``.npz`` is not possible.  Instead we exercise the BNB
# ``.log_prob`` path on the same synthetic inputs as the NB family,
# augmented with a ``bnb_concentration`` tensor, and verify three
# properties the method must satisfy:
#
# 1. Shape contract matches the NB counterpart exactly.
# 2. Output is finite (no NaN / +/-inf leakage from the BNB log-prob).
# 3. BNB differs from NB (the extra concentration parameter must have an
#    observable effect - otherwise the dispatch in ``_build_ll_count_dist``
#    would be silently broken).


# Pairing of NB base class -> BNB subclass for each of the four variants.
# The BNB classes inherit ``.log_prob`` directly from their NB parent;
# these tests verify that inheritance plus the params-dict dispatch inside
# :func:`_build_ll_count_dist` is wired up correctly.
_NB_BNB_PAIRS = [
    ("nbdm", NegativeBinomialLikelihood, BetaNegativeBinomialLikelihood),
    ("zinb", ZeroInflatedNBLikelihood, ZeroInflatedBNBLikelihood),
    ("nbvcp", NBWithVCPLikelihood, BNBWithVCPLikelihood),
    ("zinbvcp", ZINBWithVCPLikelihood, ZIBNBWithVCPLikelihood),
]


def _attach_bnb_concentration(
    params: dict, n_genes: int, value: float = 0.3
) -> dict:
    """Return a copy of ``params`` augmented with ``bnb_concentration``.

    A small per-gene ``omega`` shifts the count distribution away from NB
    enough to make the BNB vs NB comparison numerically distinguishable,
    while keeping the BNB ``kappa`` comfortably above the numerical
    stability floor used inside ``build_bnb_dist``.
    """
    out = dict(params)
    out["bnb_concentration"] = jnp.full((n_genes,), value, dtype=jnp.float32)
    return out


@pytest.mark.parametrize("model,nb_cls,bnb_cls", _NB_BNB_PAIRS)
@pytest.mark.parametrize("return_by", ["cell", "gene"])
def test_bnb_non_mixture_smoke(golden, model, nb_cls, bnb_cls, return_by):
    """BNB log_prob matches NB's shape, is finite, and differs numerically."""
    counts = jnp.asarray(golden["counts"])
    n_genes = counts.shape[1]

    nb_params = _build_params(golden, model)
    bnb_params = _attach_bnb_concentration(nb_params, n_genes)

    nb_out = np.asarray(
        nb_cls().log_prob(counts, nb_params, return_by=return_by)
    )
    bnb_out = np.asarray(
        bnb_cls().log_prob(counts, bnb_params, return_by=return_by)
    )

    assert nb_out.shape == bnb_out.shape
    assert np.all(np.isfinite(bnb_out)), (
        f"{bnb_cls.__name__} produced non-finite log-probs"
    )
    assert not np.allclose(nb_out, bnb_out), (
        f"{bnb_cls.__name__} output indistinguishable from "
        f"{nb_cls.__name__} - bnb_concentration may not be wired up."
    )


@pytest.mark.parametrize(
    "model,nb_cls,bnb_cls",
    [
        ("nbdm_mix", NegativeBinomialLikelihood, BetaNegativeBinomialLikelihood),
        ("zinb_mix", ZeroInflatedNBLikelihood, ZeroInflatedBNBLikelihood),
        ("nbvcp_mix", NBWithVCPLikelihood, BNBWithVCPLikelihood),
        (
            "zinbvcp_mix",
            ZINBWithVCPLikelihood,
            ZIBNBWithVCPLikelihood,
        ),
    ],
)
@pytest.mark.parametrize("split_components", [False, True])
def test_bnb_mixture_smoke(
    golden, model, nb_cls, bnb_cls, split_components
):
    """Mixture BNB log_prob produces finite, shape-matching, NB-distinct output.

    Additionally exercises :func:`_build_ll_count_dist`'s BNB
    concentration-axis re-alignment for the mixture broadcast layout
    ``(1, n_genes, n_components)`` - a per-gene concentration vector must
    be reshaped to ``(1, n_genes, 1)`` so it broadcasts against the
    component axis.
    """
    counts = jnp.asarray(golden["counts"])
    n_genes = counts.shape[1]

    nb_params = _build_params(golden, model)
    bnb_params = _attach_bnb_concentration(nb_params, n_genes)

    nb_out = np.asarray(
        nb_cls().log_prob(
            counts,
            nb_params,
            return_by="cell",
            split_components=split_components,
        )
    )
    bnb_out = np.asarray(
        bnb_cls().log_prob(
            counts,
            bnb_params,
            return_by="cell",
            split_components=split_components,
        )
    )

    assert nb_out.shape == bnb_out.shape
    assert np.all(np.isfinite(bnb_out)), (
        f"{bnb_cls.__name__} mixture produced non-finite log-probs"
    )
    assert not np.allclose(nb_out, bnb_out), (
        f"{bnb_cls.__name__} mixture output indistinguishable from "
        f"{nb_cls.__name__} - bnb_concentration may not be wired up "
        "for mixtures."
    )
