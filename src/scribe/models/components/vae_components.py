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
from typing import Callable, Dict, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
from flax import linen as nn

from .covariate_embedding import CovariateEmbedding, CovariateSpec

# ---------------------------------------------------------------------------
# Input transformations (matching legacy VAE)
# ---------------------------------------------------------------------------

INPUT_TRANSFORMS: Dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "log1p": jnp.log1p,
    "log": lambda x: jnp.log(x + 1e-8),
    "sqrt": jnp.sqrt,
    "identity": lambda x: x,
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
    """

    param_name: str
    output_dim: int
    transform: str = "identity"

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

    Parameters
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
# AbstractDecoder
# ===========================================================================


class AbstractDecoder(nn.Module):
    """Template base class for VAE decoders.

    Orchestrates: [concat covariate embeddings to z] → reversed MLP →
    ``decode_to_output(h)`` (subclass hook).

    Subclasses must override :meth:`decode_to_output`.

    Parameters
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

    Parameters
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
        result: Dict[str, jnp.ndarray] = {}
        for head in self.output_heads:
            raw = nn.Dense(head.output_dim, name=f"head_{head.param_name}")(h)
            transform_fn = _get_output_transform(head.transform)
            result[head.param_name] = transform_fn(raw)
        return result


# ===========================================================================
# Registries
# ===========================================================================

ENCODER_REGISTRY: Dict[str, Type[AbstractEncoder]] = {
    "gaussian": GaussianEncoder,
}

DECODER_REGISTRY: Dict[str, Type[AbstractDecoder]] = {
    "multi_head": MultiHeadDecoder,
}
