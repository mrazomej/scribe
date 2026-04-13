"""Base classes and utilities for likelihood components.

This module provides the abstract base class for all likelihoods and
helper functions for capture parameter sampling and cell-specific mixing.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# Import at module level to avoid runtime import overhead
from scribe.stats.distributions import BetaPrime

if TYPE_CHECKING:
    from ....core.axis_layout import AxisLayout
    from ...builders.parameter_specs import (
        BiologyInformedCaptureSpec,
        ParamSpec,
    )
    from ...config import ModelConfig


# ==============================================================================
# Helper functions for capture parameter sampling
# These are defined at module level to ensure they're available for JIT tracing
# ==============================================================================


def _sample_phi_capture_constrained(
    prior_params: Tuple[float, float],
) -> jnp.ndarray:
    """Sample phi_capture from constrained BetaPrime distribution."""
    return numpyro.sample("phi_capture", BetaPrime(*prior_params))


# ------------------------------------------------------------------------------


def _sample_phi_capture_unconstrained(
    prior_params: Tuple[float, float],
    transform: dist.transforms.Transform,
    constrained_name: str,
) -> jnp.ndarray:
    """Sample phi_capture using TransformedDistribution (unconstrained)."""
    base_dist = dist.Normal(*prior_params)
    transformed_dist = dist.TransformedDistribution(base_dist, transform)
    return numpyro.sample(constrained_name, transformed_dist)


# ------------------------------------------------------------------------------


def _sample_p_capture_constrained(
    prior_params: Tuple[float, float],
) -> jnp.ndarray:
    """Sample p_capture from constrained Beta distribution."""
    return numpyro.sample("p_capture", dist.Beta(*prior_params))


# ------------------------------------------------------------------------------


def _sample_p_capture_unconstrained(
    prior_params: Tuple[float, float],
    transform: dist.transforms.Transform,
    constrained_name: str,
) -> jnp.ndarray:
    """Sample p_capture using TransformedDistribution (unconstrained)."""
    base_dist = dist.Normal(*prior_params)
    transformed_dist = dist.TransformedDistribution(base_dist, transform)
    return numpyro.sample(constrained_name, transformed_dist)


# ------------------------------------------------------------------------------


def _sample_capture_biology_informed(
    log_lib_sizes: jnp.ndarray,
    log_M0: float,
    sigma_M: float,
    use_phi_capture: bool,
) -> jnp.ndarray:
    """Sample capture parameter from biology-informed prior.

    Samples the latent variable eta_c = log(M_c / L_c) from a
    TruncatedNormal prior (low=0) whose mean is anchored to the observed
    library size, then applies the exact transformation to the capture
    parameter.

    The truncation at zero enforces the physical constraint M_c >= L_c
    (a cell cannot emit more molecules than it contains).

    Parameters
    ----------
    log_lib_sizes : jnp.ndarray
        Per-cell log library sizes, shape ``(batch,)``.
    log_M0 : float
        log(M_0) where M_0 is expected total mRNA per cell.
        For data-driven mode, this is the sampled shared parameter.
    sigma_M : float
        Log-scale std-dev of cell-to-cell mRNA variation.
    use_phi_capture : bool
        If True, return phi_capture = exp(eta) - 1.
        If False, return p_capture = exp(-eta).

    Returns
    -------
    jnp.ndarray
        Capture parameter values, shape ``(batch,)``.
    """
    # eta_c ~ TruncatedNormal(log_M0 - log_L_c, sigma_M^2, low=0)
    # Truncation at 0 enforces eta_c >= 0 <=> p_capture <= 1.
    prior_mean = log_M0 - log_lib_sizes
    eta = numpyro.sample(
        "eta_capture",
        dist.TruncatedNormal(prior_mean, sigma_M, low=0.0).to_event(0),
    )

    if use_phi_capture:
        # phi_capture = exp(eta) - 1   (exact, see _capture_prior.qmd)
        capture_value = jnp.exp(eta) - 1.0
        numpyro.deterministic("phi_capture", capture_value)
    else:
        # p_capture = exp(-eta)         (exact, see _capture_prior.qmd)
        capture_value = jnp.exp(-eta)
        numpyro.deterministic("p_capture", capture_value)

    return capture_value


# ==============================================================================
# Hierarchical per-dataset mu_eta sampling helpers
# ==============================================================================
#
# When mu_eta_prior is set, per-dataset mu_eta values are drawn from a
# hierarchical prior centered on a shared population mean.  All variants
# use non-centered parameterization (NCP) for SVI stability:
#
#   mu_eta_pop ~ N(log_M0, sigma_mu)              [population mean]
#   mu_eta_raw ~ N(0, 1), shape (D,)              [per-dataset deviations]
#   mu_eta = mu_eta_pop + effective_scale * mu_eta_raw  [deterministic]
#
# The effective_scale depends on the prior type.


def _sample_hierarchical_mu_eta_gaussian(
    log_M0: float,
    sigma_mu: float,
    n_datasets: int,
) -> jnp.ndarray:
    """Sample per-dataset mu_eta with Gaussian hierarchical shrinkage.

    Parameters
    ----------
    log_M0 : float
        Prior center for the population mean (log total mRNA).
    sigma_mu : float
        Prior std-dev on the population mean.
    n_datasets : int
        Number of datasets.

    Returns
    -------
    jnp.ndarray
        Per-dataset mu_eta values, shape ``(D,)``.
    """
    # Population mean: broad prior allows exploration
    mu_eta_pop = numpyro.sample("mu_eta_pop", dist.Normal(log_M0, sigma_mu))
    # Inter-dataset spread: Softplus-transformed Normal, initialized small
    # so that datasets start nearly identical
    tau_eta = numpyro.sample(
        "tau_eta",
        dist.TransformedDistribution(
            dist.Normal(-2.0, 0.5),
            dist.transforms.SoftplusTransform(),
        ),
    )
    # NCP per-dataset deviations
    mu_eta_raw = numpyro.sample(
        "mu_eta_raw",
        dist.Normal(0.0, 1.0).expand([n_datasets]).to_event(1),
    )
    mu_eta = numpyro.deterministic("mu_eta", mu_eta_pop + tau_eta * mu_eta_raw)
    return mu_eta


def _sample_hierarchical_mu_eta_horseshoe(
    log_M0: float,
    sigma_mu: float,
    n_datasets: int,
    tau0: float = 1.0,
    slab_df: int = 4,
    slab_scale: float = 2.0,
) -> jnp.ndarray:
    """Sample per-dataset mu_eta with regularized horseshoe shrinkage.

    Uses the Finnish (regularized) horseshoe prior with per-dataset
    local shrinkage, encouraging most datasets to share the same
    mu_eta while allowing occasional outliers.

    Parameters
    ----------
    log_M0 : float
        Prior center for the population mean.
    sigma_mu : float
        Prior std-dev on the population mean.
    n_datasets : int
        Number of datasets.
    tau0 : float
        Scale for global shrinkage Half-Cauchy.
    slab_df : int
        Degrees of freedom for the slab Inverse-Gamma.
    slab_scale : float
        Scale for the slab Inverse-Gamma.

    Returns
    -------
    jnp.ndarray
        Per-dataset mu_eta values, shape ``(D,)``.
    """
    mu_eta_pop = numpyro.sample("mu_eta_pop", dist.Normal(log_M0, sigma_mu))
    # Global shrinkage
    tau_mu_eta = numpyro.sample("tau_mu_eta", dist.HalfCauchy(tau0))
    # Per-dataset local shrinkage
    lambda_mu_eta = numpyro.sample(
        "lambda_mu_eta",
        dist.HalfCauchy(1.0).expand([n_datasets]).to_event(1),
    )
    # Slab width (regularization)
    c_sq_mu_eta = numpyro.sample(
        "c_sq_mu_eta",
        dist.InverseGamma(
            0.5 * slab_df,
            0.5 * slab_df * slab_scale**2,
        ),
    )
    # Regularized effective scale
    c = jnp.sqrt(c_sq_mu_eta)
    eff_scale = (
        tau_mu_eta
        * c
        * lambda_mu_eta
        / jnp.sqrt(c_sq_mu_eta + tau_mu_eta**2 * lambda_mu_eta**2)
    )
    # NCP per-dataset deviations
    mu_eta_raw = numpyro.sample(
        "mu_eta_raw",
        dist.Normal(0.0, 1.0).expand([n_datasets]).to_event(1),
    )
    mu_eta = numpyro.deterministic(
        "mu_eta", mu_eta_pop + eff_scale * mu_eta_raw
    )
    return mu_eta


def _sample_hierarchical_mu_eta_neg(
    log_M0: float,
    sigma_mu: float,
    n_datasets: int,
    u: float = 1.0,
    a: float = 1.0,
    tau: float = 1.0,
) -> jnp.ndarray:
    """Sample per-dataset mu_eta with Normal-Exponential-Gamma shrinkage.

    The NEG hierarchy:
        zeta_d ~ Gamma(a, tau)     [per-dataset rate]
        psi_d  ~ Gamma(u, zeta_d)  [per-dataset variance]
        mu_eta_raw_d ~ N(0, 1)
        mu_eta_d = mu_eta_pop + sqrt(psi_d) * mu_eta_raw_d

    Parameters
    ----------
    log_M0 : float
        Prior center for the population mean.
    sigma_mu : float
        Prior std-dev on the population mean.
    n_datasets : int
        Number of datasets.
    u : float
        Shape for the inner Gamma (psi). u=1 gives NEG (Exponential).
    a : float
        Shape for the outer Gamma (zeta).
    tau : float
        Rate for the outer Gamma (global shrinkage).

    Returns
    -------
    jnp.ndarray
        Per-dataset mu_eta values, shape ``(D,)``.
    """
    mu_eta_pop = numpyro.sample("mu_eta_pop", dist.Normal(log_M0, sigma_mu))
    # Outer Gamma: per-dataset rate parameters
    zeta_mu_eta = numpyro.sample(
        "zeta_mu_eta",
        dist.Gamma(a, tau).expand([n_datasets]).to_event(1),
    )
    # Inner Gamma: per-dataset variance (rate = zeta)
    psi_mu_eta = numpyro.sample(
        "psi_mu_eta",
        dist.Gamma(u, zeta_mu_eta).to_event(1),
    )
    # NCP per-dataset deviations
    mu_eta_raw = numpyro.sample(
        "mu_eta_raw",
        dist.Normal(0.0, 1.0).expand([n_datasets]).to_event(1),
    )
    mu_eta = numpyro.deterministic(
        "mu_eta", mu_eta_pop + jnp.sqrt(psi_mu_eta) * mu_eta_raw
    )
    return mu_eta


def _sample_hierarchical_mu_eta(
    spec: "BiologyInformedCaptureSpec",
    n_datasets: int,
) -> jnp.ndarray:
    """Dispatch to the appropriate hierarchical mu_eta sampler.

    Parameters
    ----------
    spec : BiologyInformedCaptureSpec
        Capture spec with ``mu_eta_prior`` set to one of
        ``"gaussian"``, ``"horseshoe"``, ``"neg"``.
    n_datasets : int
        Number of datasets.

    Returns
    -------
    jnp.ndarray
        Per-dataset mu_eta values, shape ``(D,)``.

    Raises
    ------
    ValueError
        If ``spec.mu_eta_prior`` is not recognized.
    """
    prior = spec.mu_eta_prior
    if prior == "gaussian":
        return _sample_hierarchical_mu_eta_gaussian(
            spec.log_M0, spec.sigma_mu, n_datasets
        )
    elif prior == "horseshoe":
        return _sample_hierarchical_mu_eta_horseshoe(
            spec.log_M0, spec.sigma_mu, n_datasets
        )
    elif prior == "neg":
        return _sample_hierarchical_mu_eta_neg(
            spec.log_M0, spec.sigma_mu, n_datasets
        )
    else:
        raise ValueError(
            f"Unknown mu_eta_prior={prior!r}. "
            "Expected 'gaussian', 'horseshoe', or 'neg'."
        )


# ==============================================================================
# Helper for cell-specific mixing weights (annotation priors)
# ==============================================================================


def compute_cell_specific_mixing(
    mixing_weights: jnp.ndarray,
    annotation_logits: jnp.ndarray,
) -> jnp.ndarray:
    """
    Combine mixing weights with per-cell annotation logits.

    Computes cell-specific mixing probabilities by adding annotation logit
    offsets to the log of the global mixing weights and applying softmax.
    This implements the logit-nudging strategy for annotation priors in
    mixture models.

    Parameters
    ----------
    mixing_weights : jnp.ndarray, shape ``(K,)`` or ``(batch, K)``
        Mixing weights before annotation nudging. A global simplex
        ``(K,)`` is broadcast to all cells. A batch-aligned tensor
        ``(batch, K)`` supports dataset-specific mixing after indexing.
    annotation_logits : jnp.ndarray, shape ``(batch, K)``
        Per-cell additive logit offsets.  Zero rows leave the mixing
        weights unchanged; positive entries bias toward the corresponding
        component.

    Returns
    -------
    cell_mixing : jnp.ndarray, shape ``(batch, K)``
        Normalised per-cell mixing probabilities.  Each row sums to 1.

    Notes
    -----
    The computation is:

        πᵢₖ = softmaxₖ ( log πₖ + annotation_logitsᵢₖ )

    A small epsilon (1e-8) is added before taking the log to avoid
    −∞ when a mixing weight is numerically zero.

    This function is the single point where annotation priors interact
    with the mixture distribution.  A future auxiliary-observation
    strategy can replace this function while keeping the rest of the
    likelihood code unchanged.
    """
    log_weights = jnp.log(mixing_weights + 1e-8)
    cell_logits = log_weights + annotation_logits
    return jax.nn.softmax(cell_logits, axis=-1)  # (batch, K)


# ==============================================================================
# Helper for NumPyro>=0.20 mixture compatibility
# ==============================================================================


def build_mixture_general(
    mixing_dist: dist.Categorical,
    component_builder: Callable[[int], dist.Distribution],
) -> dist.MixtureGeneral:
    """
    Build a mixture distribution via ``MixtureGeneral`` from component slices.

    NumPyro 0.20 added stricter validation in ``MixtureSameFamily`` that rejects
    component distributions whose support is wrapped as
    ``IndependentConstraint`` (for example, after ``.to_event(1)``). SCRIBE's
    count likelihoods model genes as event dimensions, so this helper
    constructs an equivalent mixture using ``MixtureGeneral`` by materializing
    per-component distributions.

    Parameters
    ----------
    mixing_dist : dist.Categorical
        Mixture weights distribution with final dimension ``K`` components.
    component_builder : Callable[[int], dist.Distribution]
        Callback returning the component distribution for a component index.

    Returns
    -------
    dist.MixtureGeneral
        Mixture distribution compatible with NumPyro >=0.20 support checks.
    """
    # The number of mixture components is static from the categorical
    # parameter shape and is used to build one distribution per component.
    n_components = int(mixing_dist.probs.shape[-1])
    component_distributions = [
        component_builder(component_idx)
        for component_idx in range(n_components)
    ]
    return dist.MixtureGeneral(mixing_dist, component_distributions)


# ==============================================================================
# Helper: index per-dataset parameters by cell dataset assignment
# ==============================================================================


def index_dataset_params(
    param_values: Dict[str, jnp.ndarray],
    dataset_indices: jnp.ndarray,
    n_datasets: int,
    param_specs: Optional[List] = None,
) -> Dict[str, jnp.ndarray]:
    """Index per-dataset parameters using per-cell dataset assignments.

    For each parameter whose spec has ``is_dataset=True``, slice out the
    dataset axis using ``dataset_indices`` to produce per-cell values.

    When a parameter is **both** mixture and dataset (shape
    ``(K, D, ...)``) the dataset axis is 1.  After indexing, the result
    is transposed to **batch-first** layout ``(batch, K, ...)`` so that
    ``MixtureSameFamily`` sees the component dim as the rightmost batch
    dimension.

    When ``param_specs`` is ``None``, falls back to the legacy heuristic
    (leading dim equals ``n_datasets``).

    Parameters
    ----------
    param_values : Dict[str, jnp.ndarray]
        All sampled parameter values.
    dataset_indices : jnp.ndarray, shape ``(batch,)``
        Integer array mapping each cell in the current batch to a
        dataset index in ``{0, ..., n_datasets - 1}``.
    n_datasets : int
        Number of datasets.
    param_specs : List[ParamSpec], optional
        Parameter specifications.  Used to determine which parameters
        carry a dataset axis and whether they also carry a mixture axis.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Copy with per-dataset arrays replaced by per-cell arrays.
    """
    # Build spec lookup when available
    specs_by_name: Dict[str, object] = {}
    if param_specs is not None:
        for spec in param_specs:
            specs_by_name[spec.name] = spec

    indexed = {}
    for name, val in param_values.items():
        spec = specs_by_name.get(name)
        is_ds = spec is not None and getattr(spec, "is_dataset", False)
        is_mix = spec is not None and getattr(spec, "is_mixture", False)

        if is_ds:
            if is_mix and val.ndim >= 2:
                # Shape (K, D, ...) — dataset axis is 1.
                # Index axis 1 then move component axis after batch so the
                # result is (batch, K, ...) for MixtureSameFamily compat.
                result = jnp.take(
                    val, dataset_indices, axis=1
                )  # (K, batch, ...)
                result = jnp.moveaxis(result, 0, 1)  # (batch, K, ...)
                indexed[name] = result
            else:
                # Shape (D, ...) — dataset axis is 0
                indexed[name] = val[dataset_indices]
        elif (
            spec is None
            and name not in {"mixing_weights", "mixing_concentrations"}
            and val.ndim >= 1
            and val.shape[0] == n_datasets
        ):
            # Legacy fallback for params without specs — dataset axis is 0
            indexed[name] = val[dataset_indices]
        elif (
            spec is None
            and val.ndim >= 3
            and val.shape[1] == n_datasets
            and val.shape[0] != n_datasets
        ):
            # Legacy fallback for derived mixture+dataset params (e.g.
            # r=mu*phi) that inherit (K, D, G) shape from their sampled
            # dependencies but have no ParamSpec.  Dataset axis is 1.
            result = jnp.take(val, dataset_indices, axis=1)  # (K, batch, ...)
            result = jnp.moveaxis(result, 0, 1)  # (batch, K, ...)
            indexed[name] = result
        else:
            indexed[name] = val
    return indexed


# ------------------------------------------------------------------------------
# Likelihood Base Class
# ------------------------------------------------------------------------------


class Likelihood(ABC):
    """
    Abstract base class for likelihood components.

    Subclasses implement the `sample` method which handles:

    1. Cell plate creation (with proper batching mode)
    2. Cell-specific parameter sampling
    3. Observation sampling/conditioning

    All subclasses must handle three plate modes.  ``batch_size`` takes
    priority over ``counts`` when deciding the plate shape so that the
    model's cell plate always matches the guide's plate (which subsamples
    whenever ``batch_size`` is set):

    - **Batch sampling**: batch_size set → subsample cells (counts may
      or may not be provided; obs is ``counts[idx]`` when available)
    - **Prior predictive**: batch_size=None, counts=None → full plate,
      sample counts from prior
    - **Full sampling**: batch_size=None, counts provided → full plate,
      condition on all counts

    Examples
    --------
    >>> class MyLikelihood(Likelihood):
    ...     def sample(self, param_values, cell_specs, counts, dims,
    ...                batch_size, model_config):
    ...         # Implementation
    ...         pass
    """

    @abstractmethod
    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
        vae_cell_fn: Optional[
            Callable[[Optional[jnp.ndarray]], Dict[str, jnp.ndarray]]
        ] = None,
        annotation_prior_logits: Optional[jnp.ndarray] = None,
        dataset_indices: Optional[jnp.ndarray] = None,
        param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
    ) -> None:
        """
        Sample observations given parameters.

        Parameters
        ----------
        param_values : Dict[str, jnp.ndarray]
            Already-sampled parameter values (global and gene-specific).
            Keys are parameter names (e.g., "p", "r", "mu").
        cell_specs : List[ParamSpec]
            Specs for cell-specific parameters to sample inside the cell plate.
            These are sampled within the plate context.
        counts : Optional[jnp.ndarray]
            Observed counts matrix of shape (n_cells, n_genes).
            If None, samples from prior (prior predictive mode).
        dims : Dict[str, int]
            Dimension sizes, e.g., {"n_cells": 10000, "n_genes": 2000}.
        batch_size : Optional[int]
            Mini-batch size for stochastic VI. If None, uses all cells.
        model_config : ModelConfig
            Model configuration with hyperparameters.
        vae_cell_fn : callable, optional
            If provided, called inside the cell plate **before** obs sampling.
            Signature: ``vae_cell_fn(batch_idx) -> Dict[str, jnp.ndarray]``.
            Returns decoder-driven parameter values to merge into
            ``param_values``.
        annotation_prior_logits : jnp.ndarray, optional
            Per-cell logit offsets for mixture component assignment priors,
            shape ``(n_cells, n_components)``.  When provided for a mixture
            model, the global ``mixing_weights`` are combined with these
            logits inside the cell plate to produce cell-specific mixing
            probabilities via :func:`compute_cell_specific_mixing`.
            If ``None``, the global mixing weights are used for all cells
            (standard behaviour).
        param_layouts : dict, optional
            Semantic ``AxisLayout`` for each parameter, built by the model
            builder from ``ParamSpec`` metadata.  Used for layout-aware
            broadcasting in mixture distributions.  When ``None``, the
            likelihood falls back to building layouts from
            ``model_config.param_specs`` (which may itself be empty for
            legacy code paths).

        Notes
        -----
        This method should:
            1. Create the appropriate cell plate (with or without subsampling)
            2. If ``vae_cell_fn`` is provided, call it and update param_values
            3. Sample any cell-specific parameters from cell_specs
            4. Compute the likelihood distribution
            5. Sample or condition on counts
        """
        pass
