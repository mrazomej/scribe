"""Tests for the log-ratio reference frame (CLR / IQLR / explicit set).

Covers the core transform generalisation (``_clr_transform`` with a reference
mask, ``_iqlr_reference_mask``, ``compute_delta_from_simplex(reference=...)``),
the reference-spec resolver and its validation, propagation of ``reference``
through every empirical route (standard / marginal / mixture / shrinkage), the
recompute discipline on results objects (``set_reference``, mask preservation,
pathway rejection), and an end-to-end recover-the-truth check that IQLR
de-contaminates the reference where full CLR cannot.

See ``sec-diffexp-iqlr`` in the paper for the math.
"""

import warnings

import jax
import numpy as np
import pytest

from scribe.de import compare
from scribe.de._empirical import (
    _clr_transform,
    _iqlr_reference_mask,
    _resolve_reference,
    compute_delta_from_simplex,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dirichlet(rng, alpha, n):
    return rng.dirichlet(alpha, size=n)


def _softmax_rows(eta):
    eta = eta - eta.max(axis=-1, keepdims=True)
    e = np.exp(eta)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# _clr_transform with a reference mask
# ---------------------------------------------------------------------------


def test_clr_transform_default_is_legacy_clr():
    rng = np.random.default_rng(0)
    s = _dirichlet(rng, np.ones(6) * 4, 50)
    out = _clr_transform(s)
    log = np.log(np.maximum(s, 1e-30))
    expected = log - log.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, expected)
    # CLR columns sum to zero.
    assert np.abs(out.sum(axis=1)).max() < 1e-10


def test_clr_transform_subset_reference_only_reference_sums_to_zero():
    rng = np.random.default_rng(1)
    s = _dirichlet(rng, np.ones(6) * 4, 50)
    mask = np.array([True, True, True, False, False, False])
    out = _clr_transform(s, mask)
    # Reference columns are centred; the full vector is not (in general).
    assert np.abs(out[:, mask].sum(axis=1)).max() < 1e-9
    assert np.abs(out.sum(axis=1)).mean() > 1e-3


def test_clr_transform_bad_reference_mask():
    s = np.full((4, 5), 0.2)
    with pytest.raises(ValueError, match="length"):
        _clr_transform(s, np.array([True, False, True]))
    with pytest.raises(ValueError, match="at least one"):
        _clr_transform(s, np.zeros(5, dtype=bool))


# ---------------------------------------------------------------------------
# _iqlr_reference_mask
# ---------------------------------------------------------------------------


def test_iqlr_reference_mask_selects_iqr_band():
    rng = np.random.default_rng(2)
    # Per-column variance increasing with index; the IQR band drops the
    # lowest and highest quartiles.
    cols = [rng.normal(0, scale, size=400) for scale in np.linspace(0.1, 4.0, 8)]
    clr_pooled = np.stack(cols, axis=1)
    mask = _iqlr_reference_mask(clr_pooled, exclude_last=False)
    # Highest-variance (last index) and lowest-variance (first) excluded.
    assert not mask[-1]
    assert not mask[0]
    assert mask.sum() >= 1


def test_iqlr_reference_mask_excludes_other_column():
    rng = np.random.default_rng(3)
    clr_pooled = rng.normal(size=(300, 5))
    mask = _iqlr_reference_mask(clr_pooled, exclude_last=True)
    assert not mask[-1]  # trailing "other" never a reference


def test_iqlr_reference_mask_degenerate_falls_back():
    clr_pooled = np.zeros((10, 4))  # all variances equal -> empty band guard
    mask = _iqlr_reference_mask(clr_pooled, exclude_last=False)
    assert mask.all()


# ---------------------------------------------------------------------------
# compute_delta_from_simplex(reference=...)
# ---------------------------------------------------------------------------


def test_compute_delta_clr_is_bit_identical_to_legacy():
    rng = np.random.default_rng(4)
    A = _dirichlet(rng, np.ones(8) * 5, 60)
    B = _dirichlet(rng, np.ones(8) * 5, 60)
    mask = np.array([True] * 6 + [False] * 2)
    legacy = compute_delta_from_simplex(A, B, gene_mask=mask)
    explicit_clr = compute_delta_from_simplex(A, B, gene_mask=mask, reference="clr")
    np.testing.assert_array_equal(legacy, explicit_clr)


def test_compute_delta_iqlr_differs_and_shape():
    rng = np.random.default_rng(5)
    A = _dirichlet(rng, np.ones(8) * 5, 60)
    B = _dirichlet(rng, np.ones(8) * 5, 60)
    mask = np.array([True] * 6 + [False] * 2)
    d_clr = compute_delta_from_simplex(A, B, gene_mask=mask, reference="clr")
    d_iqlr = compute_delta_from_simplex(A, B, gene_mask=mask, reference="iqlr")
    assert d_clr.shape == d_iqlr.shape == (60, 6)
    assert not np.allclose(d_clr, d_iqlr)


def test_compute_delta_bad_reference_string():
    A = np.full((4, 4), 0.25)
    B = np.full((4, 4), 0.25)
    with pytest.raises(ValueError, match="clr.*iqlr|iqlr"):
        compute_delta_from_simplex(A, B, reference="bogus")


# ---------------------------------------------------------------------------
# _resolve_reference (resolution + validation)
# ---------------------------------------------------------------------------


NAMES = ["a", "b", "c", "d", "e"]
GM = np.array([True, True, True, False, False])  # d,e -> "other"; agg cols = 4


def test_resolve_reference_passthrough_strings():
    assert _resolve_reference("clr", NAMES, GM) == "clr"
    assert _resolve_reference("iqlr", NAMES, GM) == "iqlr"


def test_resolve_reference_bad_string():
    with pytest.raises(ValueError, match="clr.*iqlr|iqlr"):
        _resolve_reference("housekeeping", NAMES, GM)


def test_resolve_reference_gene_names():
    out = _resolve_reference(["a", "c"], NAMES, GM)
    # aggregated columns: kept [a,b,c] + other -> [T,F,T,F]
    np.testing.assert_array_equal(out, np.array([True, False, True, False]))


def test_resolve_reference_unknown_name_raises():
    with pytest.raises(ValueError, match="not among the comparison genes"):
        _resolve_reference(["zzz"], NAMES, GM)


def test_resolve_reference_filtered_name_raises():
    with pytest.raises(ValueError, match="filtered into the 'other' pool"):
        _resolve_reference(["d"], NAMES, GM)


def test_resolve_reference_full_d_boolean():
    out = _resolve_reference(np.array([True, False, True, False, False]), NAMES, GM)
    np.testing.assert_array_equal(out, np.array([True, False, True, False]))


def test_resolve_reference_full_d_selecting_filtered_raises():
    with pytest.raises(ValueError, match="filtered into the 'other' pool"):
        _resolve_reference(np.array([True, False, False, True, False]), NAMES, GM)


def test_resolve_reference_aggregated_boolean():
    out = _resolve_reference(np.array([True, False, True, False]), NAMES, GM)
    np.testing.assert_array_equal(out, np.array([True, False, True, False]))


def test_resolve_reference_wrong_length_raises():
    with pytest.raises(ValueError, match="full gene length|aggregated length"):
        _resolve_reference(np.array([True, False]), NAMES, GM)


def test_resolve_reference_no_mask_full_length():
    out = _resolve_reference(["a", "e"], NAMES, None)
    np.testing.assert_array_equal(
        out, np.array([True, False, False, False, True])
    )


def test_resolve_reference_equals_clr_when_all_genes_no_other():
    # reference = all genes and NO gene_mask -> equals plain CLR.
    rng = np.random.default_rng(6)
    A = _dirichlet(rng, np.ones(5) * 5, 40)
    B = _dirichlet(rng, np.ones(5) * 5, 40)
    all_ref = _resolve_reference(NAMES, NAMES, None)  # every gene
    d_explicit = compute_delta_from_simplex(A, B, reference=all_ref)
    d_clr = compute_delta_from_simplex(A, B, reference="clr")
    np.testing.assert_allclose(d_explicit, d_clr)


def test_explicit_all_kept_differs_from_masked_clr():
    # With a gene_mask, "clr" keeps "other" in the reference, while an
    # explicit reference over all kept genes excludes "other" -> they differ
    # by design.
    rng = np.random.default_rng(7)
    A = _dirichlet(rng, np.ones(5) * 5, 40)
    B = _dirichlet(rng, np.ones(5) * 5, 40)
    explicit_kept = _resolve_reference(["a", "b", "c"], NAMES, GM)  # all kept
    d_explicit = compute_delta_from_simplex(A, B, gene_mask=GM, reference=explicit_kept)
    d_clr = compute_delta_from_simplex(A, B, gene_mask=GM, reference="clr")
    assert not np.allclose(d_explicit, d_clr)


# ---------------------------------------------------------------------------
# recover-the-truth (C1): IQLR recovers the relative-to-stable signal,
# full CLR is biased by the driver via the unweighted reference.
# ---------------------------------------------------------------------------


def test_iqlr_recovers_relative_truth_clr_is_biased():
    rng = np.random.default_rng(11)
    N = 6000
    n_stable = 12
    D = n_stable + 2  # + driver + signal
    sig = D - 1  # signal index (low variance)
    dvr = D - 2  # driver index (high variance + large shift)
    true_signal = 1.0
    shift_big = 6.0

    # Latent log-abundances. Stable genes: mid variance, zero arm shift.
    eta_A = np.zeros((N, D))
    eta_B = np.zeros((N, D))
    eta_A[:, :n_stable] = rng.normal(0.0, 0.3, size=(N, n_stable))
    eta_B[:, :n_stable] = rng.normal(0.0, 0.3, size=(N, n_stable))
    # Driver: large variance and large arm shift -> top variance quartile.
    eta_A[:, dvr] = rng.normal(0.0, 2.0, size=N)
    eta_B[:, dvr] = rng.normal(shift_big, 2.0, size=N)
    # Signal: small variance, known relative shift -> bottom variance quartile.
    eta_A[:, sig] = rng.normal(0.0, 0.05, size=N)
    eta_B[:, sig] = rng.normal(true_signal, 0.05, size=N)

    A = _softmax_rows(eta_A)
    B = _softmax_rows(eta_B)

    d_clr = compute_delta_from_simplex(A, B, reference="clr")
    d_iqlr = compute_delta_from_simplex(A, B, reference="iqlr")

    # delta = CLR(A) - CLR(B); signal is up in B so the truth is -true_signal.
    iqlr_sig = d_iqlr[:, sig].mean()
    clr_sig = d_clr[:, sig].mean()

    # IQLR recovers the relative-to-stable truth.
    assert abs(iqlr_sig - (-true_signal)) < 0.2, iqlr_sig
    # Full CLR is biased by ~ +(shift_big + true_signal)/D away from truth.
    assert abs(clr_sig - (-true_signal)) > 0.3, clr_sig
    assert abs(iqlr_sig - (-true_signal)) < abs(clr_sig - (-true_signal))

    # The IQLR reference set drops the driver and the signal, keeps stable.
    clr_pooled = np.concatenate([_clr_transform(A), _clr_transform(B)], axis=0)
    ref_mask = _iqlr_reference_mask(clr_pooled, exclude_last=False)
    assert not ref_mask[dvr]
    assert not ref_mask[sig]
    assert ref_mask[:n_stable].sum() >= n_stable // 2


# ---------------------------------------------------------------------------
# Per-route propagation: reference must reach every empirical builder.
# ---------------------------------------------------------------------------


def _raw_arms(seed=20, N=80, D=8):
    rng = np.random.default_rng(seed)
    rA = np.abs(rng.normal(5, 1, size=(N, D)))
    rB = np.abs(rng.normal(5, 1, size=(N, D)))
    return rA, rB


def test_reference_propagates_standard_empirical():
    rA, rB = _raw_arms()
    key = jax.random.PRNGKey(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clr = compare(rA, rB, method="empirical", paired=True, rng_key=key)
        iqlr = compare(
            rA, rB, method="empirical", paired=True, rng_key=key, reference="iqlr"
        )
    assert clr._reference == "clr" and iqlr._reference == "iqlr"
    assert not np.allclose(clr.delta_samples, iqlr.delta_samples)


def test_reference_propagates_shrinkage():
    rA, rB = _raw_arms()
    key = jax.random.PRNGKey(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clr = compare(rA, rB, method="shrinkage", paired=True, rng_key=key)
        iqlr = compare(
            rA, rB, method="shrinkage", paired=True, rng_key=key, reference="iqlr"
        )
    # Metadata copied onto the shrinkage result.
    assert clr._reference == "clr" and iqlr._reference == "iqlr"
    assert not np.allclose(clr.delta_samples, iqlr.delta_samples)


def test_reference_propagates_mixture():
    from scribe.de._results_factory import _compare_empirical_mixture

    rng = np.random.default_rng(21)
    N, K, D = 60, 2, 7
    rA = np.abs(rng.normal(5, 1, size=(N, K, D)))
    rB = np.abs(rng.normal(5, 1, size=(N, K, D)))
    wA = rng.dirichlet(np.ones(K), size=N)
    wB = rng.dirichlet(np.ones(K), size=N)
    key = jax.random.PRNGKey(1)
    common = dict(
        r_samples_A=rA,
        r_samples_B=rB,
        weights_A=wA,
        weights_B=wB,
        gene_names=[f"g{i}" for i in range(D)],
        label_A="A",
        label_B="B",
        paired=True,
        n_samples_dirichlet=1,
        rng_key=key,
        batch_size=2048,
    )
    clr = _compare_empirical_mixture(**common, reference="clr")
    iqlr = _compare_empirical_mixture(**common, reference="iqlr")
    assert clr._reference == "clr" and iqlr._reference == "iqlr"
    assert not np.allclose(clr.delta_samples, iqlr.delta_samples)


def test_reference_propagates_marginal():
    from scribe.de._results_factory import _compare_empirical_from_marginal

    rng = np.random.default_rng(22)
    D = 8
    sA = _dirichlet(rng, np.ones(D) * 5, 200)
    sB = _dirichlet(rng, np.ones(D) * 5, 200)

    class _FakeMarginal:
        def __init__(self, simplex):
            self._simplex = simplex

        def get_compositional_samples(self, n_samples, rng_key, store_samples):
            return self._simplex

    fake_A = _FakeMarginal(sA)
    fake_B = _FakeMarginal(sB)
    common = dict(
        gene_names=[f"g{i}" for i in range(D)],
        label_A="A",
        label_B="B",
        n_samples_marginal=200,
        rng_key=jax.random.PRNGKey(2),
    )
    clr = _compare_empirical_from_marginal(fake_A, fake_B, **common, reference="clr")
    iqlr = _compare_empirical_from_marginal(
        fake_A, fake_B, **common, reference="iqlr"
    )
    assert clr._reference == "clr" and iqlr._reference == "iqlr"
    assert not np.allclose(clr.delta_samples, iqlr.delta_samples)


def test_parametric_rejects_non_clr_reference():
    # A non-CLR reference is not supported by the analytic parametric path.
    rng = np.random.default_rng(23)
    D = 5
    model = {
        "loc": rng.normal(size=D - 1),
        "cov_factor": rng.normal(size=(D - 1, 2)),
        "cov_diag": np.abs(rng.normal(size=D - 1)) + 0.1,
    }
    with pytest.raises(NotImplementedError, match="reference"):
        compare(model, model, method="parametric", reference="iqlr")


# ---------------------------------------------------------------------------
# Recompute discipline on results objects
# ---------------------------------------------------------------------------


def test_set_reference_recompute_and_introspection():
    rA, rB = _raw_arms(seed=24)
    key = jax.random.PRNGKey(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = compare(rA, rB, method="empirical", paired=True, rng_key=key)
    before = np.array(res.delta_samples)
    res.set_reference("iqlr")
    assert res._reference == "iqlr"
    assert not np.allclose(before, res.delta_samples)
    mask = res.iqlr_reference_mask()
    assert mask.dtype == bool and mask.shape[0] == before.shape[1]


def test_set_gene_mask_preserves_reference():
    rA, rB = _raw_arms(seed=25, D=8)
    key = jax.random.PRNGKey(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = compare(rA, rB, method="empirical", paired=True, rng_key=key)
    res.set_reference("iqlr")
    res.set_gene_mask(np.array([True] * 6 + [False] * 2))
    assert res._reference == "iqlr"
    assert res.delta_samples.shape[1] == 6


def test_pathway_tests_reject_non_clr_reference():
    rA, rB = _raw_arms(seed=26)
    key = jax.random.PRNGKey(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = compare(rA, rB, method="empirical", paired=True, rng_key=key)
    res.set_reference("iqlr")
    with pytest.raises(ValueError, match="require the CLR reference"):
        res.test_gene_set(np.array([0, 1, 2]))
    # Back to CLR -> pathway allowed.
    res.set_reference("clr")
    res.test_gene_set(np.array([0, 1, 2]))


def test_set_reference_requires_simplex():
    rA, rB = _raw_arms(seed=27)
    key = jax.random.PRNGKey(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = compare(rA, rB, method="empirical", paired=True, rng_key=key)
    res.simplex_A = None
    res.simplex_B = None
    with pytest.raises(ValueError, match="simplex samples were not stored"):
        res.set_reference("iqlr")
