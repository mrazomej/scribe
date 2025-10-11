"""Statistics functions and distributions for SCRIBE."""

# Histogram functions
from .histogram import (
    compute_histogram_percentiles,
    compute_histogram_credible_regions,
)

# ECDF functions
from .ecdf import (
    compute_ecdf_percentiles,
    compute_ecdf_credible_regions,
)

# Dirichlet functions
from .dirichlet import (
    sample_dirichlet_from_parameters,
    fit_dirichlet_mle,
    fit_dirichlet_minka,
)

# Custom distributions
from .distributions import (
    BetaPrime,
    LowRankLogisticNormal,
    SoftmaxNormal,
)

# Divergence functions
from .divergences import (
    _kl_betaprime,
    _kl_lognormal,
    jensen_shannon,
    sq_hellinger,
    hellinger,
)

# Mode patches
from .patches import apply_distribution_mode_patches

__all__ = [
    # Histogram
    "compute_histogram_percentiles",
    "compute_histogram_credible_regions",
    # ECDF
    "compute_ecdf_percentiles",
    "compute_ecdf_credible_regions",
    # Dirichlet
    "sample_dirichlet_from_parameters",
    "fit_dirichlet_mle",
    "fit_dirichlet_minka",
    # Distributions
    "BetaPrime",
    "LowRankLogisticNormal",
    "SoftmaxNormal",
    # Divergences (KL registrations)
    "_kl_betaprime",
    "_kl_lognormal",
    # Divergences (user-facing)
    "jensen_shannon",
    "sq_hellinger",
    "hellinger",
    # Patches
    "apply_distribution_mode_patches",
]
