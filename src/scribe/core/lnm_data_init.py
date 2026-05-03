"""Data-derived initializers for the Logistic-Normal Multinomial (LNM) model.

This module bundles the three small data-driven computations that the
public API performs once, up front, when the user requests an LNM
(``model in {"lnm", "lnmvcp"}``) fit. The aim is to take the noisiest,
most ill-conditioned phase of training — the first few thousand SVI
steps where the linear-decoder VAE is still discovering the dataset's
marginal composition and library-size distribution — and replace it
with a closed-form initialization derived from the count matrix.

Each function is small, side-effect-free, and tested independently. They
are intentionally placed in :mod:`scribe.core` rather than in the LNM
likelihood module so that they can be re-used by tests, notebooks, and
ablation experiments without pulling in the full model-building stack.

Why these three quantities?
---------------------------
- **Empirical ALR bias** (``empirical_alr_mean_from_counts``): anchors
  the linear decoder's bias to the dataset's marginal composition.
  Without this, ``softmax(y_alr) ≈ 1/G`` at step 0 and the optimizer
  must fight a 20 000-dimensional multinomial gradient just to recover
  the obvious. Already provided in
  :mod:`scribe.core.normalization_logistic`; re-exported here for a
  single import surface.
- **Encoder standardization stats** (``compute_encoder_standardization``):
  ``log1p_prop`` of a sparse count matrix is mostly tiny non-negative
  values, which leaves the encoder's first Dense layer
  near-rank-deficient at init. Z-standardizing in the transformed space
  preconditions the optimization and is approximately free.
- **Total-count NB prior** (``moments_to_lognormal_r_T``): the default
  ``LogNormal(0, 1)`` prior on ``r_T`` has prior mean ``≈1.65`` whereas
  10x library sizes routinely require ``r_T`` in the tens to hundreds.
  A method-of-moments NB inversion places the prior at the data's
  centre of mass so the KL is not fighting the data from step 0.

All three operate on raw counts (``n_cells × n_genes``); the API plumbs
them into ``model_config.vae`` (bias / standardize stats) and
``priors`` (``r_T``) before model construction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .normalization_logistic import empirical_alr_mean_from_counts
from ..models.components.vae_components import _get_input_transform


# Re-export to provide a single LNM-init import path. ``api.py`` and
# tests can simply ``from scribe.core.lnm_data_init import …``.
__all__ = [
    "empirical_alr_mean_from_counts",
    "compute_encoder_standardization",
    "moments_to_lognormal_r_T",
    "resolve_r_T_prior",
    "resolve_lnm_priors",
    "inject_lnm_vae_data_init",
    "CAPTURE_ANCHOR_KEYS",
    "BIOLOGY_DEFAULT_R_T_MEDIAN",
    "BIOLOGY_DEFAULT_R_T_SIGMA_LOG",
]


# Single source of truth for the priors-dict keys that signal the user has
# activated the biology-informed capture-probability prior. Kept here (not
# in ``api.py``) so the main API layer and the LNM helpers agree on what
# "the user opted into the capture anchor" means.
CAPTURE_ANCHOR_KEYS: Tuple[str, ...] = ("eta_capture", "mu_eta", "organism")


# Biology-informed default for the LogNormal prior on the total-count NB
# dispersion ``r_T`` when the capture anchor is active. See
# ``resolve_r_T_prior`` for the full justification. Median 50 is a typical
# 10x-Chromium-scale value; sigma_log = 1.5 gives a 95 % prior interval of
# roughly [2.5, 1000], wide enough not to fight the posterior.
BIOLOGY_DEFAULT_R_T_MEDIAN: float = 50.0
BIOLOGY_DEFAULT_R_T_SIGMA_LOG: float = 1.5


# Floor applied to the per-feature standard deviation before the encoder
# divides by it. The same floor is applied inside the encoder
# (``_preprocess`` adds ``1e-8``); we use a slightly larger floor here so
# constant-column features become exactly zero after standardization
# rather than being dominated by the floor's reciprocal.
_STD_FLOOR: float = 1e-3


def compute_encoder_standardization(
    counts,
    input_transform: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute per-feature mean/std for the VAE encoder's input.

    The encoder applies a configured input transformation (``log1p``,
    ``log1p_prop``, ``clr`` …) before feeding data through its MLP. When
    ``standardize=True`` the encoder additionally subtracts a per-feature
    mean and divides by a per-feature standard deviation. **Both stats
    must therefore be computed in the post-transform space**, not in raw
    count space — otherwise standardization would shift the
    transformation's domain and silently change the network's input
    distribution.

    This helper applies the named input transform to the count matrix
    (using the same ``INPUT_TRANSFORMS`` registry the encoder uses), then
    returns the per-gene mean and standard deviation across cells.

    Parameters
    ----------
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix. Accepts numpy or JAX arrays. Float / int
        entries are tolerated.
    input_transform : str
        Name of the transform from ``INPUT_TRANSFORMS`` (e.g.
        ``"log1p"``, ``"log1p_prop"``, ``"clr"``, ``"log1p_norm"``).
        Must match the value the encoder will use at training time.

    Returns
    -------
    mean : jnp.ndarray, shape (n_genes,)
        Per-gene mean of the transformed inputs.
    std : jnp.ndarray, shape (n_genes,)
        Per-gene standard deviation of the transformed inputs, floored
        at ``_STD_FLOOR`` to prevent division by zero on
        constant-column features.

    Raises
    ------
    ValueError
        If ``input_transform`` is not a recognized transform name. The
        error includes the list of valid names so the caller does not
        have to chase the registry.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> counts = jnp.array([[10, 5, 0], [12, 6, 1], [11, 4, 0]])
    >>> mean, std = compute_encoder_standardization(counts, "log1p_prop")
    >>> mean.shape, std.shape
    ((3,), (3,))

    See Also
    --------
    scribe.models.components.vae_components.INPUT_TRANSFORMS :
        Source-of-truth registry of input transforms.
    """
    # Bring counts into JAX. We pin float32 because the encoder's Dense
    # layers are float32 and any mixed-precision boundary inside
    # ``_preprocess`` would force per-step casting.
    counts_j = jnp.asarray(counts, dtype=jnp.float32)

    if counts_j.ndim != 2:
        raise ValueError(
            "counts must be a 2-D matrix (n_cells, n_genes), "
            f"got rank {counts_j.ndim} array with shape {counts_j.shape}."
        )

    # Apply the same input transform the encoder will use. Re-using
    # ``_get_input_transform`` keeps this function in lock-step with the
    # encoder's preprocessing — if a new transform is added, both the
    # encoder and this helper pick it up automatically.
    transform_fn = _get_input_transform(input_transform)
    transformed = transform_fn(counts_j)

    # Per-feature (gene) statistics across the cell axis. ``ddof=0`` —
    # the population standard deviation — is appropriate here because we
    # are computing a centering scale, not estimating a parameter.
    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0)

    # Floor at ``_STD_FLOOR`` to avoid numerical issues on
    # constant-column features (e.g. genes that are zero across every
    # cell after a hard-zero filter). The corresponding standardized
    # value will then be ``(x - mean) / std`` ≈ 0, i.e. the constant
    # feature contributes no signal — which is exactly what we want.
    std = jnp.maximum(std, _STD_FLOOR)

    return mean, std


def moments_to_lognormal_r_T(
    counts,
    sigma_log: float = 1.0,
    min_r_T: float = 1.0,
) -> Tuple[float, float]:
    """Method-of-moments NB inversion to set the LogNormal ``r_T`` prior.

    The total-count submodel of the LNM is
    ``u_T^{(c)} ~ NegBin(r_T, p)``. The default LogNormal(0, 1) prior on
    ``r_T`` has median ``1.0`` and 95 % CI ``[0.14, 7.4]`` — which is
    multiple orders of magnitude below the values appropriate for typical
    droplet-scRNA-seq library sizes. The KL term then fights the
    likelihood throughout early training, contributing to the spiky loss
    profile users observe with the unmodified prior.

    This helper inverts the negative-binomial moment relations to obtain
    a data-driven point estimate of ``r_T``, then reports it as the
    median (and a user-chosen log-spread) of a LogNormal prior. The user
    can still override by passing ``priors={"r_T": ...}`` to
    ``scribe.fit``; this function is only used to produce a sensible
    automatic default when the user does not.

    Method-of-moments inversion
    ---------------------------
    For “X ~ NB(r, p)” parameterized by success probability “p”,

        ⟨X⟩ = r·p/(1−p)
        Var[X] = r·p/(1−p)²

    Letting m = mean(u_T) and v = var(u_T):

        1 − p = m/v
        r̂_T = m²/(v − m)

    The estimator requires over-dispersion (v > m); for the rare case where the
    empirical v ≤ m, we floor at min_r_T to keep the prior in a defensible
    region.

    Parameters
    ----------
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix. Total counts are computed as ``counts.sum(-1)``.
    sigma_log : float, default=1.0
        Standard deviation (in log space) of the resulting LogNormal
        prior. The default is moderately informative: ``±1`` log-unit
        is roughly a factor of e (≈ 2.7) on each side of the median,
        wide enough to permit dataset-to-dataset variation while still
        anchoring the prior near the data.
    min_r_T : float, default=1.0
        Floor on the point estimate to handle the under-dispersed corner
        case ``var(u_T) <= mean(u_T)``. ``r_T = 1`` corresponds to a
        geometric distribution on totals — a soft choice that always
        admits any realistic library-size distribution.

    Returns
    -------
    mu_log : float
        Location parameter of the LogNormal prior (the log-median).
    sigma_log : float
        Scale parameter of the LogNormal prior (echoed for symmetry
        with the input; useful for the caller's own bookkeeping).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> counts = rng.negative_binomial(n=50, p=0.005, size=(200, 5))
    >>> mu, sigma = moments_to_lognormal_r_T(counts)
    >>> 0.0 < np.exp(mu) < 1e4  # plausible r_T range
    True

    See Also
    --------
    scribe.models.parameterizations.LogisticNormalParameterization :
        Where the ``r_T`` LogNormalSpec is registered.
    """
    # Use numpy here (not JAX) because we pass the result directly to
    # the prior-spec system as Python floats. Computing the moments
    # under JAX and immediately calling ``float(...)`` would round-trip
    # to host memory anyway, defeating any tracing benefit.
    counts_np = np.asarray(counts)

    if counts_np.ndim != 2:
        raise ValueError(
            "counts must be a 2-D matrix (n_cells, n_genes), "
            f"got rank {counts_np.ndim} array with shape {counts_np.shape}."
        )

    # Per-cell library size and its first two moments across cells.
    u_T = counts_np.sum(axis=-1).astype(np.float64)
    m = float(u_T.mean())
    v = float(u_T.var(ddof=0))

    # Method-of-moments inversion. We guard against the under-dispersed
    # corner case by falling back to ``min_r_T``: the LNM total-count
    # submodel still trains in that regime, but the data-driven estimate
    # would be negative or infinite and is therefore not usable.
    if v > m and (v - m) > 0:
        r_T_estimate = (m * m) / (v - m)
    else:
        r_T_estimate = min_r_T

    # The LogNormal(mu_log, sigma_log) distribution has median
    # ``exp(mu_log)``; setting ``mu_log = log(r_T_estimate)`` therefore
    # places the prior median exactly at the moment-of-moments estimate,
    # which is the most defensible single-point summary we have.
    r_T_estimate = max(float(r_T_estimate), float(min_r_T))
    mu_log = float(np.log(r_T_estimate))
    return mu_log, float(sigma_log)


def inject_lnm_vae_data_init(
    model_config,
    counts,
    alr_reference_idx: int = -1,
):
    """Return a copy of ``model_config`` with LNM data-init fields populated.

    This helper centralizes the transformation that
    :func:`scribe.api.fit` performs on a freshly-built ``ModelConfig``
    when the user requests an LNM (or LNMVCP) model. It computes the
    empirical ALR bias and per-feature encoder standardization stats
    from the count matrix and stashes them on ``model_config.vae`` via
    the standard Pydantic ``model_copy`` pattern.

    Pulling the logic out of ``api.py`` keeps the public API
    side-effect-free (``api.py`` is already long enough) and lets the
    test suite verify the wiring without spinning up an SVI run.

    Parameters
    ----------
    model_config : ModelConfig
        Freshly-built configuration. The function reads
        ``model_config.vae.input_transform`` to compute standardization
        stats in the correct space, then returns a new ``ModelConfig``
        with the data-init fields filled in. Must be a VAE-inference
        config built for the ``logistic_normal`` parameterization;
        callers are expected to gate the call on this themselves.
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix used to derive the ALR mean and the
        per-feature encoder standardization stats.
    alr_reference_idx : int, default=-1
        Reference index for the ALR transform. ``-1`` selects the last
        gene; in production the API resolves this earlier from a
        geometric-mean argmax via ``select_alr_reference``.

    Returns
    -------
    ModelConfig
        New ``ModelConfig`` whose ``.vae`` carries non-``None``
        ``empirical_alr_bias_init``, ``standardize_mean`` and
        ``standardize_std`` arrays. The original ``model_config`` is
        not mutated.

    Notes
    -----
    Standardization stats are computed unconditionally — even when
    ``model_config.vae.standardize`` is ``False`` — so that the values
    are present on the config for downstream introspection and so that
    flipping the ``standardize`` flag does not require re-deriving
    stats from data.
    """
    vae = model_config.vae

    alr_bias = empirical_alr_mean_from_counts(
        counts, reference_idx=alr_reference_idx
    )
    stand_mean, stand_std = compute_encoder_standardization(
        counts, input_transform=vae.input_transform
    )

    new_vae = vae.model_copy(
        update={
            "empirical_alr_bias_init": alr_bias,
            "standardize_mean": stand_mean,
            "standardize_std": stand_std,
        }
    )
    return model_config.model_copy(update={"vae": new_vae})


def resolve_r_T_prior(
    model: str,
    counts,
    priors,
) -> "Optional[Tuple[float, float]]":
    """Resolve the LNM ``r_T`` LogNormal prior under both anchor regimes.

    This is the single decision-making helper invoked by ``scribe.api.fit``
    for LNM-family models when populating an automatic prior on the
    total-count NB dispersion ``r_T``. The function returns either a
    ``(mu_log, sigma_log)`` tuple to be inserted into the user's priors
    dict, or ``None`` when no automatic prior should be set (i.e., the
    user already supplied one explicitly, or the model is not LNM).

    Two regimes are handled:

    1. **No capture anchor active.** No ``eta_capture`` / ``mu_eta`` /
       ``organism`` key in ``priors``. The cell-to-cell variation in
       ``u_T`` ends up partly absorbed by the totals NB itself, so a
       method-of-moments inversion on the empirical totals gives a
       sensible (slightly biased-low for LNMVCP) ballpark for ``r_T``.
       We use it, with ``sigma_log = 1.5`` for LNMVCP to absorb the
       bias and ``sigma_log = 1.0`` for plain LNM where the inversion
       is exact.

    2. **Capture anchor active.** The user opted into the capture prior
       by passing one of :data:`CAPTURE_ANCHOR_KEYS` in their priors
       dict. With ``p_capture^{(c)}`` pinned to ``L_c / M_0``, the
       cell-to-cell variation in ``u_T`` is consumed by the per-cell
       capture and there is essentially no residual variance from
       which to estimate ``r_T`` via totals moments. We skip the MoM
       and substitute a fixed, biology-informed default
       ``LogNormal(log BIOLOGY_DEFAULT_R_T_MEDIAN,
       BIOLOGY_DEFAULT_R_T_SIGMA_LOG)``.

    In both regimes, an explicit user override via ``priors["r_T"]``
    always wins: this helper short-circuits to ``None`` when ``"r_T"``
    is already present in ``priors``.

    Parameters
    ----------
    model : str
        The model string passed to :func:`scribe.api.fit`. Only ``"lnm"``
        and ``"lnmvcp"`` trigger the helper; every other value returns
        ``None``.
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix used for the method-of-moments branch. Ignored
        when the capture anchor is active.
    priors : dict or None
        The user-supplied priors dict (or ``None``). Read-only here; the
        caller is responsible for cloning and inserting the returned
        tuple into a fresh dict.

    Returns
    -------
    tuple of (mu_log, sigma_log) or None
        The location and scale of a LogNormal prior to assign to
        ``priors["r_T"]``, or ``None`` when no automatic assignment
        should occur.

    Examples
    --------
    No capture anchor → MoM-derived prior:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> counts = rng.negative_binomial(n=50, p=0.005, size=(200, 5))
    >>> mu, sigma = resolve_r_T_prior("lnm", counts, priors=None)
    >>> sigma == 1.0
    True

    Capture anchor active → biology-informed default:

    >>> mu, sigma = resolve_r_T_prior(
    ...     "lnmvcp", counts, priors={"organism": "human"}
    ... )
    >>> bool(np.isclose(np.exp(mu), BIOLOGY_DEFAULT_R_T_MEDIAN))
    True

    User explicit override → no auto-assignment:

    >>> resolve_r_T_prior(
    ...     "lnmvcp", counts, priors={"r_T": (3.0, 0.5)}
    ... ) is None
    True

    Non-LNM model → no auto-assignment:

    >>> resolve_r_T_prior("nbdm", counts, priors=None) is None
    True

    See Also
    --------
    moments_to_lognormal_r_T : The MoM helper used in regime (1).
    """
    # Only act on LNM-family models. Every other model leaves r_T alone.
    if model.lower() not in ("lnm", "lnmvcp"):
        return None

    # User-supplied ``r_T`` always wins. The caller's intent is to fix the
    # prior; we never silently override that choice.
    if isinstance(priors, dict) and "r_T" in priors:
        return None

    # Branch (2): capture anchor active → biology-informed default.
    capture_anchor_active = isinstance(priors, dict) and any(
        k in priors for k in CAPTURE_ANCHOR_KEYS
    )
    if capture_anchor_active:
        mu_log = float(np.log(BIOLOGY_DEFAULT_R_T_MEDIAN))
        return mu_log, float(BIOLOGY_DEFAULT_R_T_SIGMA_LOG)

    # Branch (1): no capture anchor → MoM on totals.
    # LNMVCP gets a wider sigma_log because the empirical variance
    # includes capture variability that biases the MoM estimate downward;
    # a wider prior absorbs that bias. Plain LNM has no such inflation.
    sigma_log = 1.5 if model.lower() == "lnmvcp" else 1.0
    return moments_to_lognormal_r_T(counts, sigma_log=sigma_log)


# ------------------------------------------------------------------------------
# Per-parameterization prior resolver
# ------------------------------------------------------------------------------


def _empirical_mu_T_prior(
    counts, sigma_log: float = 1.0
) -> Tuple[float, float]:
    """Empirical-mean LogNormal prior for the LNM totals mean ``mu_T``.

    The population-level expected library size is one of the most
    precisely identified data quantities — its sampling CV scales as
    ``1/sqrt(C)`` with the cell count. We therefore center a LogNormal
    on the empirical mean ``mean(u_T)`` and use a moderately tight
    log-spread by default. Under the capture anchor in particular,
    ``mu_T`` should land near ``M_0 * mean(p_capture)`` ≈ ``mean(L_c)``,
    which this prior places at the center.

    Parameters
    ----------
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix; the row sums are taken as ``u_T^{(c)}``.
    sigma_log : float, default=1.0
        Log-scale spread of the resulting prior. ``1.0`` is generous
        (factor of e on each side of the median); the empirical mean
        is precise enough that the prior is informative even at this
        width.

    Returns
    -------
    mu_log : float
        ``log(mean(u_T))`` — the LogNormal location.
    sigma_log : float
        Echoed input.
    """
    counts_np = np.asarray(counts)
    if counts_np.ndim != 2:
        raise ValueError(
            "counts must be a 2-D matrix (n_cells, n_genes), "
            f"got rank {counts_np.ndim} array with shape {counts_np.shape}."
        )
    u_T = counts_np.sum(axis=-1).astype(np.float64)
    mu_T_estimate = float(u_T.mean())
    mu_T_estimate = max(mu_T_estimate, 1.0)  # floor for log
    mu_log = float(np.log(mu_T_estimate))
    return mu_log, float(sigma_log)


def _empirical_phi_T_prior(
    counts, sigma_log: float = 1.5, min_phi_T: float = 1e-3
) -> Tuple[float, float]:
    """Empirical odds-ratio LogNormal prior for ``phi_T``.

    The DM relation ``phi = (1 - p) / p`` gives ``phi_T = mean / r_T``
    when expressed in the totals NB. Inverting moments,
    ``phi_T = (v - m) / m^2 * mean = (v - m) / m`` after substitution,
    which equals the variance-to-mean ratio minus 1. We use this as the
    prior median, with a generous log-spread because under the capture
    anchor ``v`` is inflated by capture variability and the empirical
    estimate is biased.

    Parameters
    ----------
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix.
    sigma_log : float, default=1.5
        Log-scale spread; wider than mu_T's because phi_T is less
        precisely identified by the data.
    min_phi_T : float, default=1e-3
        Floor on the point estimate to handle the under-dispersed
        corner case ``var(u_T) <= mean(u_T)``.

    Returns
    -------
    mu_log : float
        ``log(phi_T_estimate)`` — the LogNormal location.
    sigma_log : float
        Echoed input.
    """
    counts_np = np.asarray(counts)
    if counts_np.ndim != 2:
        raise ValueError(
            "counts must be a 2-D matrix (n_cells, n_genes), "
            f"got rank {counts_np.ndim} array with shape {counts_np.shape}."
        )
    u_T = counts_np.sum(axis=-1).astype(np.float64)
    m = float(u_T.mean())
    v = float(u_T.var(ddof=0))
    # ``phi_T = (v - m) / m`` from the NB second-moment inversion.
    if v > m and m > 0:
        phi_T_estimate = (v - m) / m
    else:
        phi_T_estimate = min_phi_T
    phi_T_estimate = max(float(phi_T_estimate), float(min_phi_T))
    mu_log = float(np.log(phi_T_estimate))
    return mu_log, float(sigma_log)


def resolve_lnm_priors(
    model: str,
    parameterization: str,
    counts,
    priors,
) -> Dict[str, Tuple[float, float]]:
    """Resolve auto-default priors for the LNM-family scalars.

    This is the parameterization-aware generalization of
    :func:`resolve_r_T_prior`. It returns a dict keyed by the *sampled*
    scalars of the chosen LNM-family parameterization (``r_T`` and
    ``p`` for canonical, ``mu_T`` and ``p`` for mean_prob, ``mu_T`` and
    ``phi_T`` for mean_odds), populated with sensible auto-defaults
    derived from the empirical totals and the capture-anchor regime.

    User-supplied priors always win: any key the user already set in
    their ``priors`` dict (whether by internal name or descriptive
    alias) is excluded from the returned dict so the caller does not
    accidentally clobber it.

    Parameters
    ----------
    model : str
        Model name; only ``"lnm"`` / ``"lnmvcp"`` produce non-empty
        results. Other models return ``{}``.
    parameterization : str
        Internal LNM-family parameterization key, one of
        ``"logistic_normal_canonical"``, ``"logistic_normal_mean_prob"``,
        ``"logistic_normal_mean_odds"``.
    counts : array_like, shape (n_cells, n_genes)
        Count matrix used for empirical estimates.
    priors : dict or None
        User-supplied priors dict; read-only.

    Returns
    -------
    dict
        Mapping from parameter name (``r_T`` / ``mu_T`` / ``p`` /
        ``phi_T``) to a ``(mu_log, sigma_log)`` LogNormal pair. Empty
        when ``model`` is non-LNM, or when the user already overrode
        every relevant key. The caller is responsible for merging
        these into the user's priors dict via a non-mutating copy.

    Notes
    -----
    The legacy :func:`resolve_r_T_prior` is preserved for
    backwards-compatible call sites that only know about ``r_T``;
    new code should prefer this function because it correctly
    handles ``mean_prob`` and ``mean_odds`` variants.
    """
    if model.lower() not in ("lnm", "lnmvcp"):
        return {}

    # Accept either the internal key (``logistic_normal_canonical``
    # etc.) or the user-facing short form (``canonical``,
    # ``mean_prob``, ``mean_odds``). The function is called from both
    # the public API path and from internal tests; normalizing here
    # keeps both call sites simple.
    param_lower = parameterization.lower()
    if param_lower in ("canonical", "mean_prob", "mean_odds"):
        param_lower = f"logistic_normal_{param_lower}"

    priors_dict: Dict[str, Any] = {} if not isinstance(priors, dict) else priors
    capture_anchor_active = any(k in priors_dict for k in CAPTURE_ANCHOR_KEYS)

    # Resolve each scalar by parameterization, skipping anything the
    # user already set (under either the internal name or its
    # descriptive alias). The descriptive alias check is what lets
    # ``priors={"total_mean": ...}`` short-circuit auto-assignment of
    # ``mu_T``, mirroring the symmetric behavior on the DM family.
    from ..models.config.parameter_mapping import PRIOR_KEY_ALIASES

    def _user_set(internal_name: str) -> bool:
        # Internal name directly set?
        if internal_name in priors_dict:
            return True
        # One of its descriptive aliases set?
        for alias, target in PRIOR_KEY_ALIASES.items():
            if target == internal_name and alias in priors_dict:
                return True
        return False

    out: Dict[str, Tuple[float, float]] = {}

    if param_lower == "logistic_normal_canonical":
        # Same logic as resolve_r_T_prior, but written here for parity.
        if not _user_set("r_T"):
            if capture_anchor_active:
                out["r_T"] = (
                    float(np.log(BIOLOGY_DEFAULT_R_T_MEDIAN)),
                    float(BIOLOGY_DEFAULT_R_T_SIGMA_LOG),
                )
            else:
                sigma_log = (
                    1.5 if model.lower() == "lnmvcp" else 1.0
                )
                out["r_T"] = moments_to_lognormal_r_T(
                    counts, sigma_log=sigma_log
                )
        return out

    if param_lower == "logistic_normal_mean_prob":
        # Sampled scalars: (mu_T, p). r_T is derived. We set mu_T from
        # the empirical mean library size and leave p at its default
        # Beta(1, 1) — under the anchor, p remains aliased with
        # p_capture, which is unavoidable for this variant. (Switch to
        # mean_odds to eliminate the aliasing.)
        if not _user_set("mu_T"):
            out["mu_T"] = _empirical_mu_T_prior(counts, sigma_log=1.0)
        return out

    if param_lower == "logistic_normal_mean_odds":
        # Sampled scalars: (mu_T, phi_T). p is derived as 1/(1+phi_T)
        # and r_T as mu_T*phi_T, so neither has the boundary-collapse
        # problem of the canonical / mean_prob variants. We set both
        # mu_T and phi_T from empirical moments.
        if not _user_set("mu_T"):
            out["mu_T"] = _empirical_mu_T_prior(counts, sigma_log=1.0)
        if not _user_set("phi_T"):
            out["phi_T"] = _empirical_phi_T_prior(counts, sigma_log=1.5)
        return out

    # Fallback: parameterization not recognized — return empty rather
    # than risk a wrong auto-assignment.
    return {}
