"""
Guide family markers and implementations for variational inference.

This module provides guide family markers (dataclasses) that are used
for multiple dispatch in the guide building process. Each marker represents
a different variational approximation family.

Guide families are **per-parameter** - different parameters in the same
model can use different guide families. For example:

- `r` (dispersion) might use `LowRankGuide` to capture gene correlations
- `p_capture` might use `AmortizedGuide` for efficient inference
- `p` (dropout) might use simple `MeanFieldGuide`

Classes
-------
GuideFamily
    Abstract base class for guide family markers.
MeanFieldGuide
    Marker for mean-field (fully factorized) variational family.
LowRankGuide
    Marker for low-rank MVN covariance structure.
AmortizedGuide
    Marker for amortized inference using neural networks.
VAELatentGuide
    Guide family for VAE latent variable z.

Examples
--------
>>> from scribe.models.components import MeanFieldGuide, LowRankGuide
>>> # Per-parameter guide families
>>> BetaSpec("p", (), (1.0, 1.0), guide_family=MeanFieldGuide())
>>> LogNormalSpec("r", ("n_genes",), (0.0, 1.0), guide_family=LowRankGuide(rank=10))

See Also
--------
scribe.models.builders.guide_builder : Uses guide families to build guides.
scribe.models.components.amortizers : Amortizer implementations.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from .amortizers import Amortizer

# ------------------------------------------------------------------------------
# Guide Family Base Class
# ------------------------------------------------------------------------------


@dataclass
class GuideFamily:
    """
    Abstract base class for guide family markers.

    Guide families are lightweight marker classes that specify which
    variational approximation to use for a parameter. The actual
    implementation is handled by dispatch functions in guide_builder.

    Subclasses should be dataclasses that may contain configuration
    (e.g., rank for LowRankGuide, amortizer for AmortizedGuide).
    """

    pass


# ------------------------------------------------------------------------------
# Mean-Field Guide Family
# ------------------------------------------------------------------------------


@dataclass
class MeanFieldGuide(GuideFamily):
    """
    Marker for mean-field (fully factorized) variational family.

    In a mean-field approximation, the variational distribution factorizes
    completely over parameters:

        q(θ₁, θ₂, ...) = q(θ₁) × q(θ₂) × ...

    Each parameter has its own independent variational distribution with
    learnable parameters.

    Advantages
    ----------
    - Simple and fast
    - Works well for many parameters
    - Low memory footprint

    Disadvantages
    -------------
    - Cannot capture correlations between parameters
    - May underestimate posterior uncertainty

    Examples
    --------
    >>> spec = BetaSpec("p", (), (1.0, 1.0), guide_family=MeanFieldGuide())
    """

    pass


# ------------------------------------------------------------------------------
# Low-Rank Guide Family
# ------------------------------------------------------------------------------


@dataclass
class LowRankGuide(GuideFamily):
    """
    Marker for low-rank MVN covariance structure.

    Uses a low-rank multivariate normal approximation with covariance:

        Σ = W @ W.T + diag(D)

    where W is (n_params, rank) and D is diagonal. This captures
    correlations between parameters with O(n × rank) memory instead
    of O(n²) for full covariance.

    Parameters
    ----------
    rank : int
        Rank of the low-rank factor matrix W.
        Higher rank captures more correlations but uses more memory.
        Typical values: 5-20 for gene-specific parameters.

    Advantages
    ----------
    - Captures parameter correlations
    - Memory-efficient: O(n × rank) vs O(n²)
    - Good for gene-specific parameters where genes may be correlated

    Disadvantages
    -------------
    - More parameters to optimize than mean-field
    - May be slower to converge

    Examples
    --------
    >>> # Capture correlations between gene dispersion parameters
    >>> spec = LogNormalSpec(
    ...     "r", ("n_genes",), (0.0, 1.0),
    ...     is_gene_specific=True,
    ...     guide_family=LowRankGuide(rank=10)
    ... )
    """

    rank: int = 10

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")


# ------------------------------------------------------------------------------
# Amortized Guide Family
# ------------------------------------------------------------------------------


@dataclass
class AmortizedGuide(GuideFamily):
    """
    Marker for amortized inference using neural networks.

    Instead of learning separate variational parameters for each data point,
    amortized inference uses a neural network to predict variational
    parameters from data features (sufficient statistics).

    This is particularly useful for:
        - Cell-specific parameters (e.g., p_capture) with many cells
        - Parameters that depend on observable data features
        - Reducing the number of variational parameters

    Parameters
    ----------
    amortizer : Amortizer, optional
        The amortizer network to use. If None, a default amortizer
        will be created based on the parameter specification.

    Advantages
    ----------
    - Scales to large numbers of data points
    - Shares statistical strength across similar data points
    - Fewer variational parameters (network weights vs per-point params)

    Disadvantages
    -------------
    - Requires choosing network architecture
    - May not be as flexible as per-point parameters
    - Training can be more complex

    Examples
    --------
    >>> from scribe.models.components import Amortizer, TOTAL_COUNT
    >>> amortizer = Amortizer(
    ...     sufficient_statistic=TOTAL_COUNT,
    ...     hidden_dims=[64, 32],
    ...     output_params=["alpha", "beta"],
    ... )
    >>> spec = BetaSpec(
    ...     "p_capture", ("n_cells",), (1.0, 1.0),
    ...     is_cell_specific=True,
    ...     guide_family=AmortizedGuide(amortizer=amortizer)
    ... )

    See Also
    --------
    scribe.models.components.amortizers.Amortizer : Neural network amortizer.
    """

    amortizer: Optional["Amortizer"] = None


# ------------------------------------------------------------------------------
# VAE Latent Guide Family
# ------------------------------------------------------------------------------


@dataclass
class VAELatentGuide(GuideFamily):
    """Guide family for VAE latent variable z.

    In the guide, the encoder maps counts to latent distribution parameters;
    ``latent_spec.make_guide_dist(params)`` builds the guide distribution for
    z.  The decoder is stored here so the **model** builder can access it —
    the guide never runs the decoder.

    ``param_names`` is derived from ``decoder.output_heads`` so there is a
    single source of truth for which model parameters come from the decoder.

    Parameters
    ----------
    encoder : optional
        VAE encoder (e.g. ``GaussianEncoder``).
    decoder : optional
        VAE decoder (e.g. ``MultiHeadDecoder``).
    latent_spec : optional
        Latent specification (e.g. ``GaussianLatentSpec``).
    flow : optional
        Posterior flow on z (future — not implemented yet).

    See Also
    --------
    AmortizedGuide : Single-parameter amortization.
    scribe.models.builders.parameter_specs.GaussianLatentSpec : Latent spec.
    """

    encoder: Optional[Any] = None
    decoder: Optional[Any] = None
    latent_spec: Optional[Any] = None
    flow: Optional[Any] = None

    @property
    def param_names(self) -> List[str]:
        """Parameter names produced by the decoder — single source of truth."""
        if self.decoder is not None and hasattr(self.decoder, "output_heads"):
            return [h.param_name for h in self.decoder.output_heads]
        return []
