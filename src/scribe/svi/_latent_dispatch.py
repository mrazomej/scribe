"""
Dispatched free functions for encoder-dependent VAE operations.

This module defines latent-space operations that depend on the encoder type
(e.g. Gaussian, vMF).  Each function dispatches on the ``LatentSpec`` subclass,
so adding a new encoder type only requires new ``@dispatch`` registrations
here — no changes to the ``LatentSpaceMixin`` or ``ScribeVAEResults`` class.

Functions
---------
run_encoder
    Run encoder and return variational parameters as a dict.
sample_latent_posterior
    Sample z from q(z|x) given encoder variational parameters.
get_latent_embedding
    Extract a point embedding from encoder variational parameters.

Notes
-----
This follows the same dispatch pattern used in ``parameter_specs.py``
(``sample_prior``) and ``guide_builder.py`` (``setup_guide``,
``setup_cell_specific_guide``).

See Also
--------
scribe.models.builders.parameter_specs : LatentSpec / GaussianLatentSpec.
scribe.svi._latent_space : LatentSpaceMixin that calls these functions.
"""

from typing import Dict

import jax.numpy as jnp
import numpyro.distributions as dist
from multipledispatch import dispatch

from ..models.builders.parameter_specs import GaussianLatentSpec


# ==============================================================================
# run_encoder — dispatch on LatentSpec subclass
# ==============================================================================


@dispatch(GaussianLatentSpec, object, dict, jnp.ndarray)
def run_encoder(
    spec: GaussianLatentSpec,
    encoder,
    enc_params: dict,
    counts: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Run a Gaussian encoder on counts and return variational parameters.

    Applies the encoder Linen module to the count matrix using trained
    parameters, and packages the output into a standardised dict.

    Parameters
    ----------
    spec : GaussianLatentSpec
        Latent specification (determines output semantics).
    encoder : nn.Module
        The encoder Linen module (un-initialized).
    enc_params : dict
        Trained encoder parameters extracted from NumPyro's param store
        (the ``vae_encoder$params`` subtree).
    counts : jnp.ndarray, shape (n_cells, n_genes) or (batch, n_genes)
        Input count matrix.

    Returns
    -------
    var_params : Dict[str, jnp.ndarray]
        Variational parameters.  For Gaussian:
        ``{"loc": (n_cells, latent_dim), "log_scale": (n_cells, latent_dim)}``.
    """
    # GaussianEncoder.__call__ returns (loc, log_scale)
    loc, log_scale = encoder.apply({"params": enc_params}, counts)
    return {"loc": loc, "log_scale": log_scale}


# ==============================================================================
# sample_latent_posterior — dispatch on LatentSpec subclass
# ==============================================================================


@dispatch(GaussianLatentSpec, dict, object, int)
def sample_latent_posterior(
    spec: GaussianLatentSpec,
    var_params: Dict[str, jnp.ndarray],
    rng_key,
    n_samples: int,
) -> jnp.ndarray:
    """Sample z from q(z|x) using Gaussian reparameterization.

    Draws ``n_samples`` posterior samples for each cell using the
    reparameterization trick:
    ``z = loc + eps * exp(0.5 * log_scale)``  where  ``eps ~ N(0, I)``.

    Parameters
    ----------
    spec : GaussianLatentSpec
        Latent specification.
    var_params : dict
        Output of ``run_encoder``.  Must contain keys ``"loc"`` and
        ``"log_scale"``, each with shape ``(n_cells, latent_dim)``.
    rng_key : PRNGKey
        JAX PRNG key for sampling.
    n_samples : int
        Number of posterior samples to draw per cell.

    Returns
    -------
    z : jnp.ndarray, shape (n_samples, n_cells, latent_dim)
        Reparameterized posterior samples.
    """
    # Extract the mean (loc) and standard deviation (scale) from encoder output
    loc = var_params["loc"]
    scale = jnp.exp(0.5 * var_params["log_scale"])
    # Construct the posterior distribution q(z|x) = N(loc, scale^2)
    posterior = dist.Normal(loc, scale).to_event(1)
    # Sample n_samples draws for each cell from the posterior using the
    # reparameterization trick
    return posterior.sample(rng_key, sample_shape=(n_samples,))


# ==============================================================================
# get_latent_embedding — dispatch on LatentSpec subclass
# ==============================================================================


@dispatch(GaussianLatentSpec, dict)
def get_latent_embedding(
    spec: GaussianLatentSpec,
    var_params: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Extract a point embedding from Gaussian encoder output.

    For Gaussian encoders the natural point embedding is the posterior
    mean (``loc``).

    Parameters
    ----------
    spec : GaussianLatentSpec
        Latent specification.
    var_params : dict
        Output of ``run_encoder``.  Must contain key ``"loc"``.

    Returns
    -------
    embedding : jnp.ndarray, shape (n_cells, latent_dim)
        Point embedding (posterior mean for Gaussian).
    """
    return var_params["loc"]
