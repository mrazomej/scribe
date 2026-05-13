"""
Abstract encoder/decoder hierarchy for VAE-style models.

Provides base classes (``AbstractEncoder``, ``AbstractDecoder``) that
define a common interface — preprocessing, optional covariate
embedding, shared MLP backbone — and delegate the **output-head
logic** to concrete subclasses via a template-method hook.

The first concrete pair is ``GaussianEncoder`` / ``SimpleDecoder``,
which parameterise a diagonal-Gaussian latent posterior and a
continuous reconstruction, respectively. Adding new latent
distributions (LogNormal, von Mises–Fisher, …) or observation models
(NegBin, ZINB, Bernoulli, …) only requires a new subclass that
overrides ``encode_to_params`` or ``decode_to_output``.

Covariate conditioning
----------------------
All encoders, decoders (and, separately, flow conditioners) accept an
optional list of :class:`CovariateSpec` objects. When provided, integer
covariate arrays are mapped through learned ``nn.Embed`` tables and
concatenated to the MLP input, making every module batch/donor/condition
aware without changing the core architecture.

Classes
-------
AbstractEncoder
    Template base class for all encoders.
GaussianEncoder
    Diagonal-Gaussian latent posterior (loc, log_scale).
AbstractDecoder
    Template base class for all decoders.
SimpleDecoder
    Single-vector reconstruction output.

Registries
----------
ENCODER_REGISTRY
    ``str → Type[AbstractEncoder]`` lookup.
DECODER_REGISTRY
    ``str → Type[AbstractDecoder]`` lookup.

See Also
--------
scribe.models.components.covariate_embedding : ``CovariateSpec``, ``CovariateEmbedding``.
scribe.models.components.amortizers.Amortizer : Amortizer MLP pattern.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
from flax import linen as nn

from .covariate_embedding import CovariateEmbedding, CovariateSpec

# ---------------------------------------------------------------------------
# Input transformations (matching legacy VAE)
# ---------------------------------------------------------------------------

# NOTE:
# Use named callables (instead of lambdas) for debuggability and safer
# serialization in toolchains that pickle Python callables.


def _log1p_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Apply element-wise ``log(1 + x)`` transform.

    Parameters
    ----------
    x : jnp.ndarray
        Non-negative count-like input array.

    Returns
    -------
    jnp.ndarray
        Transformed array with the same shape as ``x``.
    """
    return jnp.log1p(x)


def _log_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Apply numerically safe element-wise natural logarithm.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        ``log(x + 1e-8)`` evaluated element-wise.
    """
    return jnp.log(x + 1e-8)


def _sqrt_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Apply element-wise square-root transform.

    Parameters
    ----------
    x : jnp.ndarray
        Non-negative input array.

    Returns
    -------
    jnp.ndarray
        ``sqrt(x)`` evaluated element-wise.
    """
    return jnp.sqrt(x)


def _identity_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Return the input unchanged.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The same input values with no transformation.
    """
    return x


def _log1p_prop_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Apply compositional ``log1p`` transform on row-wise proportions.

    Parameters
    ----------
    x : jnp.ndarray
        Count matrix with genes on the last axis (``..., n_genes``).

    Returns
    -------
    jnp.ndarray
        ``log1p(x / row_total)`` where ``row_total = sum(x, axis=-1)``.
        The denominator is clamped to at least ``1.0`` to avoid division
        by zero for empty rows.
    """
    row_total = jnp.maximum(x.sum(axis=-1, keepdims=True), 1.0)
    return jnp.log1p(x / row_total)


def _clr_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Apply centered log-ratio transform with pseudocount ``0.5``.

    Parameters
    ----------
    x : jnp.ndarray
        Count matrix with genes on the last axis (``..., n_genes``).

    Returns
    -------
    jnp.ndarray
        ``log(x + 0.5) - mean(log(x + 0.5), axis=-1)`` so each row is
        centered in log-ratio coordinates.
    """
    log_x = jnp.log(x + 0.5)
    return log_x - log_x.mean(axis=-1, keepdims=True)


_LOG1P_NORM_TARGET_SIZE = 1e4


def _log1p_norm_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Apply library-size normalization to ``1e4`` followed by ``log1p``.

    Parameters
    ----------
    x : jnp.ndarray
        Count matrix with genes on the last axis (``..., n_genes``).

    Returns
    -------
    jnp.ndarray
        ``log1p(x * 1e4 / row_total)`` with ``row_total`` clamped to at
        least ``1.0`` for numerical safety.
    """
    row_total = jnp.maximum(x.sum(axis=-1, keepdims=True), 1.0)
    return jnp.log1p(x * _LOG1P_NORM_TARGET_SIZE / row_total)


INPUT_TRANSFORMS: Dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "log1p": _log1p_transform,
    "log": _log_transform,
    "sqrt": _sqrt_transform,
    "identity": _identity_transform,
    "log1p_prop": _log1p_prop_transform,
    "clr": _clr_transform,
    "log1p_norm": _log1p_norm_transform,
}


def _get_input_transform(name: str) -> Callable:
    if name not in INPUT_TRANSFORMS:
        raise ValueError(
            f"Unknown input_transformation '{name}'. "
            f"Choose from: {list(INPUT_TRANSFORMS.keys())}"
        )
    return INPUT_TRANSFORMS[name]


# ---------------------------------------------------------------------------
# Output transformations (decoder heads)
# ---------------------------------------------------------------------------


def _identity(x: jnp.ndarray) -> jnp.ndarray:
    return x


def _clamp_exp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(jnp.clip(x, -5.0, 5.0))


OUTPUT_TRANSFORMS: Dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "identity": _identity,
    "exp": jnp.exp,
    "softplus": jax.nn.softplus,
    "sigmoid": jax.nn.sigmoid,
    "clamp_exp": _clamp_exp,
}


def _get_output_transform(name: str) -> Callable:
    if name not in OUTPUT_TRANSFORMS:
        raise ValueError(
            f"Unknown output transform '{name}'. "
            f"Choose from: {list(OUTPUT_TRANSFORMS.keys())}"
        )
    return OUTPUT_TRANSFORMS[name]


# ------------------------------------------------------------------------------
# DecoderOutputHead
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderOutputHead:
    """Configuration for a single decoder output head.

    Each head corresponds to one model parameter produced by the decoder
    (e.g. dispersion ``r``, zero-inflation ``gate``).

    Parameters
    ----------
    param_name : str
        Name of the model parameter (e.g. ``"r"``, ``"gate"``).
    output_dim : int
        Dimensionality of this head's output (e.g. ``n_genes``).
    transform : str
        Key into :data:`OUTPUT_TRANSFORMS` applied to the raw Dense
        output to produce the constrained value.
    bias_init : Optional[Callable]
        Optional flax-style bias initializer for the head's
        :class:`nn.Dense`. The signature is the standard flax initializer
        contract ``(rng_key, shape, dtype) -> jnp.ndarray``. When
        ``None`` (default), the layer uses flax's default zero bias.

        This hook exists primarily for the LNM linear-decoder head
        (``"y_alr"``), where initializing the bias to the empirical ALR
        mean of the count matrix produces a near-marginal-correct output
        at training step 0 — dramatically reducing the gradient pressure
        on the rest of the decoder during the first few thousand
        iterations. See ``empirical_alr_mean_from_counts`` in
        :mod:`scribe.core.normalization_logistic`. The PLN
        ``"y_log_rate"`` head reuses the same hook to anchor its bias
        to the empirical per-gene log-mean expression.
    kernel_init : Optional[Callable]
        Optional flax-style kernel initializer for the head's
        :class:`nn.Dense`. Same ``(rng_key, shape, dtype) -> array``
        contract as ``bias_init``. When ``None`` (default), the layer
        uses flax's standard initializer (Lecun normal).

        This hook exists for the PLN linear-decoder head
        (``"y_log_rate"``), where initializing the kernel to the
        principal components of the centered log-count matrix gives the
        optimizer a warm start on the covariance structure (the kernel
        of a linear decoder *is* the loadings matrix ``W`` in the
        generative model ``Sigma = W W^T + diag(d)``). Only meaningful
        for linear-decoder heads; with a hidden-MLP decoder the kernel
        of the final dense maps a hidden representation to the output,
        not the latent ``z`` itself, and PCA loadings would be the
        wrong shape.
    """

    param_name: str
    output_dim: int
    transform: str = "identity"
    # We use ``Optional[Callable]`` rather than ``Optional[jnp.ndarray]``
    # because flax's ``nn.Dense(bias_init=...)`` API expects an
    # initializer callable, not an array. To pass a fixed array, callers
    # wrap it via ``nn.initializers.constant(arr)``; the factory does
    # exactly this for the LNM ``y_alr`` head.
    bias_init: Optional[Callable] = None
    # Same contract as ``bias_init`` but for the dense kernel — used by
    # the PLN ``y_log_rate`` head to seed ``W`` from PCA loadings of the
    # centered log-count matrix.
    kernel_init: Optional[Callable] = None

    def __post_init__(self):
        if self.transform not in OUTPUT_TRANSFORMS:
            raise ValueError(
                f"Unknown transform '{self.transform}' for head "
                f"'{self.param_name}'. "
                f"Choose from: {list(OUTPUT_TRANSFORMS.keys())}"
            )


# ------------------------------------------------------------------------------
# Activation helper
# ------------------------------------------------------------------------------

_ACTIVATIONS: Dict[str, Callable] = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
    "leaky_relu": jax.nn.leaky_relu,
    "softplus": jax.nn.softplus,
    "selu": jax.nn.selu,
    "celu": jax.nn.celu,
}


def _get_act(name: str) -> Callable:
    name_lower = name.lower()
    if name_lower not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name_lower]


# ===========================================================================
# AbstractEncoder
# ===========================================================================


class AbstractEncoder(nn.Module):
    """Template base class for VAE encoders.

    Orchestrates: input-transform → standardize → [concat covariate
    embeddings] → shared MLP → ``encode_to_params(h)`` (subclass hook).

    Subclasses must override :meth:`encode_to_params` to define the
    output head(s) and their distributional interpretation.

    Attributes
    ----------
    input_dim : int
        Dimensionality of the observed data.
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dims : List[int]
        Hidden-layer sizes for the shared MLP backbone.
    activation : str
        Activation function name.
    input_transformation : str
        Applied to raw input before the MLP.
    standardize_mean, standardize_std : Optional[jnp.ndarray]
        Per-feature statistics for z-standardization.
    covariate_specs : Optional[List[CovariateSpec]]
        Categorical covariates to embed and concatenate.
    """

    input_dim: int
    latent_dim: int
    hidden_dims: List[int]
    activation: str = "relu"
    input_transformation: str = "log1p"
    standardize_mean: Optional[jnp.ndarray] = None
    standardize_std: Optional[jnp.ndarray] = None
    covariate_specs: Optional[List[CovariateSpec]] = None

    # -- internal helpers (called inside @nn.compact __call__) --------------

    def _preprocess(self, x: jnp.ndarray) -> jnp.ndarray:
        transform = _get_input_transform(self.input_transformation)
        h = transform(x)
        if (
            self.standardize_mean is not None
            and self.standardize_std is not None
        ):
            h = (h - self.standardize_mean) / (self.standardize_std + 1e-8)
        return h

    # -- template method ----------------------------------------------------

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        covariates: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """Encode observed data to latent distribution parameters.

        Parameters
        ----------
        x : jnp.ndarray
            Input data, shape ``(..., input_dim)``.
        covariates : dict, optional
            Maps covariate name → integer ID array, shape ``(...)``.

        Returns
        -------
        Subclass-defined tuple of parameter arrays.
        """
        act = _get_act(self.activation)
        h = self._preprocess(x)

        # Optional covariate conditioning
        if covariates is not None and self.covariate_specs:
            emb = CovariateEmbedding(
                covariate_specs=self.covariate_specs,
                name="cov_embed",
            )(covariates)
            h = jnp.concatenate([h, emb], axis=-1)

        # Shared MLP backbone
        for i, dim in enumerate(self.hidden_dims):
            h = nn.Dense(dim, name=f"hidden_{i}")(h)
            h = act(h)

        return self.encode_to_params(h)

    def encode_to_params(self, h: jnp.ndarray):
        """Map the shared hidden representation to distribution parameters.

        Must be overridden by every concrete encoder subclass. Called
        inside ``__call__`` after preprocessing + hidden layers.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement encode_to_params()"
        )


# ===========================================================================
# GaussianEncoder
# ===========================================================================


class GaussianEncoder(AbstractEncoder):
    """Encoder for a diagonal-Gaussian latent posterior.

    Outputs ``(loc, log_scale)`` — mean and log-standard-deviation of
    an independent Normal distribution in latent space.

    Examples
    --------
    >>> encoder = GaussianEncoder(
    ...     input_dim=2000, latent_dim=10,
    ...     hidden_dims=[256, 128],
    ... )
    >>> params = encoder.init(jax.random.PRNGKey(0), jnp.zeros(2000))
    >>> loc, log_scale = encoder.apply(params, counts)
    """

    @nn.compact
    def encode_to_params(
        self, h: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        loc = nn.Dense(self.latent_dim, name="loc_head")(h)
        log_scale = nn.Dense(self.latent_dim, name="log_scale_head")(h)
        return loc, log_scale


# ===========================================================================
# LNMGaussianEncoder
# ===========================================================================


# Default clamp range for the log-scale head of the LNM encoder. These
# bounds map to standard deviations of roughly ``[exp(-7), exp(2)] ≈
# [9.1e-4, 7.4]`` in the latent ``h`` space. The lower bound prevents
# ``σ → 0`` (which would produce a delta-mass posterior whose KL term is
# ``-∞``) and the upper bound prevents ``σ → ∞`` (which would produce
# numerically unstable reparameterised samples). The values are
# deliberately wide enough that they should never bind once the encoder
# has learned, while still short-circuiting the runaway dynamics
# observed at the start of training when the encoder output is noise.
_LNM_LOG_SCALE_MIN: float = -7.0
_LNM_LOG_SCALE_MAX: float = 2.0


class LNMGaussianEncoder(GaussianEncoder):
    """Gaussian encoder with a clamped log-scale head, used by LNM models.

    This is a thin specialization of :class:`GaussianEncoder` that wraps
    the ``log_scale`` output in a fixed clamp ``[log_scale_min,
    log_scale_max]``. The high-dimensional multinomial likelihood used by
    the LNM (and LNMVCP) models is unusually unforgiving in the early
    training regime: an unconstrained log-scale head can drift to
    ``-∞`` (producing a delta-mass posterior on ``h`` whose KL diverges)
    or ``+∞`` (which destabilises reparameterised samples and saturates
    the multinomial). Clamping is the standard fix in scVI/CellBender and
    other count-data VAEs.

    The clamp is applied to the **raw** Dense output (interpreted as a
    log-scale) before any downstream computation. Because the clamp is
    a soft fence that almost never binds in practice, it does not change
    the model's expressive capacity once training has converged.

    Attributes
    ----------
    log_scale_min : float, default=-7.0
        Lower bound for ``log σ``. The default corresponds to
        ``σ ≈ 9.1e-4`` — much smaller than the natural scale of the
        N(0, I) prior on the latent ``h``, but large enough to keep
        the KL contribution finite.
    log_scale_max : float, default=2.0
        Upper bound for ``log σ``. The default corresponds to
        ``σ ≈ 7.4`` — about an order of magnitude larger than the
        prior, so the variational family can still represent broad
        posteriors without saturating the reparameterised sampler.

    Notes
    -----
    The clamp is purely a numerical-stability device. It does **not**
    introduce a non-trivial prior on ``log σ`` (which would distort
    the ELBO); it only prevents the optimiser from wandering into
    regions where gradients are arithmetically meaningless. Outside
    the LNM family, where the encoder output drives a Gaussian
    reconstruction likelihood instead of a multinomial-softmax, the
    plain :class:`GaussianEncoder` should be preferred.

    Examples
    --------
    >>> encoder = LNMGaussianEncoder(
    ...     input_dim=20000, latent_dim=128,
    ...     hidden_dims=[256, 128],
    ...     activation="gelu",
    ...     input_transformation="log1p_prop",
    ... )
    >>> params = encoder.init(jax.random.PRNGKey(0), jnp.zeros(20000))
    >>> loc, log_scale = encoder.apply(params, counts)
    >>> # log_scale is guaranteed in [-7, 2] regardless of training state.

    See Also
    --------
    GaussianEncoder : Unclamped baseline encoder used by other models.
    """

    # Both fields default to the module-level constants but are exposed
    # as module attributes so an ablation experiment or downstream model
    # can tune the clamp without subclassing.
    log_scale_min: float = _LNM_LOG_SCALE_MIN
    log_scale_max: float = _LNM_LOG_SCALE_MAX

    @nn.compact
    def encode_to_params(
        self, h: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # The location head is left untouched: the prior on ``h`` is the
        # standard ``N(0, I)`` for which arbitrarily large means are
        # legitimate (they just incur larger KL).
        loc = nn.Dense(self.latent_dim, name="loc_head")(h)
        # The raw log-scale is clamped element-wise. ``jnp.clip`` is
        # subgradient-safe at the clamp boundaries: gradients flow
        # through the unclamped region and stop at the boundary, which
        # is the desired behaviour for a soft regulariser.
        log_scale_raw = nn.Dense(self.latent_dim, name="log_scale_head")(h)
        log_scale = jnp.clip(
            log_scale_raw, self.log_scale_min, self.log_scale_max
        )
        return loc, log_scale


# ===========================================================================
# AbstractDecoder
# ===========================================================================


class AbstractDecoder(nn.Module):
    """Template base class for VAE decoders.

    Orchestrates: [concat covariate embeddings to z] → reversed MLP →
    ``decode_to_output(h)`` (subclass hook).

    Subclasses must override :meth:`decode_to_output`.

    Attributes
    ----------
    output_dim : int
        Dimensionality of the reconstruction target.
    latent_dim : int
        Dimensionality of the latent input.
    hidden_dims : List[int]
        Hidden-layer sizes in *encoder order* (reversed internally).
    activation : str
        Activation function name.
    standardize_mean, standardize_std : Optional[jnp.ndarray]
        Per-feature statistics for destandardization.
    covariate_specs : Optional[List[CovariateSpec]]
        Categorical covariates to embed and concatenate.
    """

    output_dim: int
    latent_dim: int
    hidden_dims: List[int]
    activation: str = "relu"
    standardize_mean: Optional[jnp.ndarray] = None
    standardize_std: Optional[jnp.ndarray] = None
    covariate_specs: Optional[List[CovariateSpec]] = None

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        covariates: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """Decode latent samples to data-space output.

        Parameters
        ----------
        z : jnp.ndarray
            Latent samples, shape ``(..., latent_dim)``.
        covariates : dict, optional
            Maps covariate name → integer ID array, shape ``(...)``.

        Returns
        -------
        Subclass-defined output array(s).
        """
        act = _get_act(self.activation)
        h = z

        # Optional covariate conditioning
        if covariates is not None and self.covariate_specs:
            emb = CovariateEmbedding(
                covariate_specs=self.covariate_specs,
                name="cov_embed",
            )(covariates)
            h = jnp.concatenate([h, emb], axis=-1)

        # Reversed MLP backbone
        reversed_dims = list(reversed(self.hidden_dims))
        for i, dim in enumerate(reversed_dims):
            h = nn.Dense(dim, name=f"hidden_{i}")(h)
            h = act(h)

        return self.decode_to_output(h)

    def decode_to_output(self, h: jnp.ndarray):
        """Map the hidden representation to the final output.

        Must be overridden by every concrete decoder subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement decode_to_output()"
        )


# ===========================================================================
# MultiHeadDecoder
# ===========================================================================


class MultiHeadDecoder(AbstractDecoder):
    """Decoder with one or more output heads, each producing a named parameter.

    Each head applies its own ``Dense`` projection followed by a transform
    from :data:`OUTPUT_TRANSFORMS`.  The decoder returns a dict mapping
    ``param_name → constrained_array``.

    Attributes
    ----------
    output_heads : Tuple[DecoderOutputHead, ...]
        One entry per model parameter produced by the decoder.

    Examples
    --------
    >>> heads = (
    ...     DecoderOutputHead("r", output_dim=2000, transform="exp"),
    ...     DecoderOutputHead("gate", output_dim=2000, transform="sigmoid"),
    ... )
    >>> decoder = MultiHeadDecoder(
    ...     output_dim=0, latent_dim=10,
    ...     hidden_dims=[128, 64], output_heads=heads,
    ... )
    >>> params = decoder.init(jax.random.PRNGKey(0), jnp.zeros(10))
    >>> out = decoder.apply(params, jnp.zeros(10))
    >>> out["r"].shape, out["gate"].shape
    ((2000,), (2000,))
    """

    output_heads: Tuple[DecoderOutputHead, ...] = ()

    @nn.compact
    def decode_to_output(self, h: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Build one Dense layer per output head and dispatch to the
        # configured output transform. We honour an optional per-head
        # ``bias_init`` callable so that consumers (notably the LNM linear
        # decoder) can anchor a head's bias to a data-derived constant
        # rather than the flax default of zero. Falling back to flax's
        # built-in default keeps existing models bit-identical.
        result: Dict[str, jnp.ndarray] = {}
        for head in self.output_heads:
            # Assemble keyword args lazily so that we only pass
            # ``bias_init`` when a custom initializer was supplied; this
            # leaves the default code path identical to pre-bias-init
            # behavior for every existing decoder head.
            dense_kwargs: Dict[str, Any] = {"name": f"head_{head.param_name}"}
            if head.bias_init is not None:
                dense_kwargs["bias_init"] = head.bias_init
            if head.kernel_init is not None:
                # Same lazy assembly as ``bias_init``: only forward the
                # initializer when the caller supplied one, so heads
                # without a custom kernel init keep flax's default Lecun
                # normal scheme bit-for-bit.
                dense_kwargs["kernel_init"] = head.kernel_init
            raw = nn.Dense(head.output_dim, **dense_kwargs)(h)
            transform_fn = _get_output_transform(head.transform)
            result[head.param_name] = transform_fn(raw)
        return result


# ===========================================================================
# Registries
# ===========================================================================

ENCODER_REGISTRY: Dict[str, Type[AbstractEncoder]] = {
    "gaussian": GaussianEncoder,
    # LNM-specific Gaussian encoder with a clamped log-scale head. The
    # clamp is a numerical-stability fence around the multinomial-softmax
    # likelihood and is registered as a separate encoder type so that
    # non-LNM models continue to use the unclamped baseline.
    "gaussian_lnm": LNMGaussianEncoder,
}

DECODER_REGISTRY: Dict[str, Type[AbstractDecoder]] = {
    "multi_head": MultiHeadDecoder,
}
