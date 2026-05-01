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

from typing import Tuple

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
    "inject_lnm_vae_data_init",
]


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
