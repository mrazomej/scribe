"""Amortized guide helpers and dispatch registration.

This module contains helper utilities for running amortizer networks and
registers the generic amortized guide dispatch.
"""

from typing import TYPE_CHECKING, Dict, Optional

import jax.numpy as jnp
import numpyro
from multipledispatch import dispatch
from numpyro.contrib.module import flax_module

from .parameter_specs import ParamSpec
from ..components.amortizers import AmortizedOutput
from ..components.guide_families import AmortizedGuide, VAELatentGuide

if TYPE_CHECKING:
    from ..config import ModelConfig

def _run_amortizer(
    guide: AmortizedGuide,
    spec: ParamSpec,
    counts: Optional[jnp.ndarray],
    batch_idx: Optional[jnp.ndarray],
) -> AmortizedOutput:
    """
    Run amortizer network and return AmortizedOutput (contract: see
    AmortizedOutput).

    Validates counts, registers flax_module, gets batch data, calls net(data).
    """
    if counts is None:
        raise ValueError("Amortized guide requires counts data")
    module_name = f"{spec.name}_amortizer"
    net = flax_module(
        module_name, guide.amortizer, input_shape=(guide.amortizer.input_dim,)
    )
    data = counts if batch_idx is None else counts[batch_idx]
    return net(data)


# ------------------------------------------------------------------------------


def _setup_grouped_amortized_latent(
    guide: VAELatentGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    counts: Optional[jnp.ndarray],
    batch_idx: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Run VAE encoder, build guide dist from latent_spec, sample z.

    Used when guide has encoder and latent_spec set. Encoder is registered
    as flax_module and run on counts (or batch); latent_spec.make_guide_dist
    turns encoder output into the guide distribution for z.
    """
    if counts is None:
        raise ValueError("VAELatentGuide with encoder requires counts")
    if guide.encoder is None or guide.latent_spec is None:
        raise ValueError(
            "VAELatentGuide VAE path requires encoder and latent_spec"
        )
    n_genes = dims["n_genes"]
    net = flax_module(
        "vae_encoder",
        guide.encoder,
        input_shape=(n_genes,),
    )
    data = counts if batch_idx is None else counts[batch_idx]
    loc, log_scale = net(data)
    var_params = {"loc": loc, "log_scale": log_scale}
    guide_dist = guide.latent_spec.make_guide_dist(var_params)
    return numpyro.sample(guide.latent_spec.sample_site, guide_dist)


# ------------------------------------------------------------------------------


@dispatch(ParamSpec, AmortizedGuide, dict, object)
def setup_cell_specific_guide(
    spec: ParamSpec,
    guide: AmortizedGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    counts: Optional[jnp.ndarray] = None,
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """
    Amortized guide for cell-specific parameters (Beta, BetaPrime,
    SigmoidNormal, ExpNormal).

    Uses a shared helper to run the amortizer, then builds the guide
    distribution via spec.make_amortized_guide_dist(out.params) and samples at
    spec.amortized_guide_sample_site. Transform logic lives in the specs only.

    Supported specs: BetaSpec, BetaPrimeSpec, SigmoidNormalSpec, ExpNormalSpec.
    """
    out = _run_amortizer(guide, spec, counts, batch_idx)
    try:
        guide_dist = spec.make_amortized_guide_dist(out.params)
        site = spec.amortized_guide_sample_site
    except NotImplementedError as e:
        raise ValueError(
            f"AmortizedGuide is not supported for spec type {type(spec).__name__}. "
            "Supported: BetaSpec, BetaPrimeSpec, SigmoidNormalSpec, ExpNormalSpec."
        ) from e
    return numpyro.sample(site, guide_dist)
