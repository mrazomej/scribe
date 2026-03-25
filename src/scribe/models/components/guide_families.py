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
- `mu` + `phi` might share a `JointLowRankGuide` for cross-parameter
  correlations

Classes
-------
GuideFamily
    Abstract base class for guide family markers.
MeanFieldGuide
    Marker for mean-field (fully factorized) variational family.
LowRankGuide
    Marker for low-rank MVN covariance structure.
JointLowRankGuide
    Marker for joint low-rank MVN across multiple parameter groups.
NormalizingFlowGuide
    Marker for normalizing-flow variational family (per-parameter).
JointNormalizingFlowGuide
    Marker for joint normalizing-flow across multiple parameter groups.
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
# Joint Low-Rank Guide Family
# ------------------------------------------------------------------------------


@dataclass
class JointLowRankGuide(GuideFamily):
    """
    Marker for joint low-rank MVN covariance across multiple parameter groups.

    Instead of fitting separate LowRankMVN distributions per parameter, this
    guide defines a single joint LowRankMVN over the stacked unconstrained
    vector of all grouped parameters. The joint covariance captures
    cross-parameter correlations (e.g., between mu_g and phi) that are
    structurally absent from factorized guides.

    **Heterogeneous dimensions**: parameters in a joint group may have
    different trailing dimensions (e.g., scalar phi with G=1 alongside
    gene-specific mu with G=n_genes).  Scalar parameters are internally
    expanded with a trailing dimension of 1 so the Woodbury chain operates
    uniformly, then collapsed back to a scalar ``Normal`` at sampling time
    to match the model's event shape.  Only batch dimensions (all dims
    except the trailing one) must be consistent.

    Implementation uses the chain rule decomposition:

        q(theta_1, theta_2) = q(theta_1) * q(theta_2 | theta_1)

    where both the marginal and the conditional are LowRankMVN of the same
    rank, computed via the Woodbury identity. This ensures exact ELBO
    computation with standard NumPyro sampling sites.

    Parameters
    ----------
    rank : int
        Rank of the joint low-rank factor matrix W.
        At rank k, the guide trades within-block for cross-block expressivity.
        At rank 2k, it strictly generalizes two separate rank-k LowRankGuides.
        Typical values: 10-20.
    group : str
        Identifier linking parameters that share the same joint covariance.
        All ParamSpecs with a JointLowRankGuide having the same ``group``
        are modeled jointly. The sampling order follows the order in which
        specs appear in the parameter list.

    Advantages
    ----------
    - Captures cross-parameter correlations (e.g., scalar phi and gene mu)
    - Each conditional in the chain is itself a LowRankMVN (same rank)
    - Computational overhead is O(G_i * k^2 + k^3) per conditioning step
    - Natural extension to 3+ parameters (e.g., ZINB with gate)
    - Supports heterogeneous dimensions (scalar + gene-specific in one group)

    Disadvantages
    -------------
    - At rank k, within-block expressivity is reduced vs separate rank-k guides
    - More complex optimization landscape

    Examples
    --------
    >>> # Joint guide for gene-specific mu and scalar phi
    >>> joint = JointLowRankGuide(rank=10, group="nb_params")
    >>> PositiveNormalSpec("phi", (), (0.0, 1.0),
    ...     is_gene_specific=False, guide_family=joint)
    >>> PositiveNormalSpec("mu", ("n_genes",), (0.0, 1.0),
    ...     is_gene_specific=True, guide_family=joint)

    See Also
    --------
    LowRankGuide : Per-parameter low-rank guide (no cross-parameter correlations).
    """

    rank: int = 10
    group: str = "default"
    dense_params: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if not self.group:
            raise ValueError("group must be a non-empty string")


# ------------------------------------------------------------------------------
# Normalizing Flow Guide Family
# ------------------------------------------------------------------------------


@dataclass
class NormalizingFlowGuide(GuideFamily):
    """
    Marker for normalizing-flow variational family (per-parameter).

    Replaces the Gaussian base used by ``LowRankGuide`` with a
    flow-transformed distribution, enabling multimodal, skewed, and
    heavy-tailed posterior approximations.

    A ``FlowChain`` is registered as a Flax module via
    ``numpyro.contrib.module.flax_module`` and wrapped in a
    ``FlowDistribution`` that serves as a proper NumPyro distribution
    (with ``log_prob`` and ``sample``).

    Parameters
    ----------
    flow_type : str
        Type of flow layer.  Coupling flows (``"spline_coupling"``,
        ``"affine_coupling"``) are recommended for guides because they
        are O(1) in both the sampling and density-evaluation directions.
        Autoregressive flows (``"maf"``, ``"iaf"``) are also supported.
    num_layers : int
        Number of flow layers to stack.
    hidden_dims : tuple of int
        Hidden-layer sizes for the conditioner networks in each layer.
    activation : str
        Activation function used inside conditioner networks.
    n_bins : int
        Number of rational-quadratic spline bins (only used when
        ``flow_type`` is ``"spline_coupling"``).
    mixture_strategy : str, default ``"independent"``
        How to handle mixture components (and dataset indices) when the
        parameter has ``is_mixture=True`` or ``is_dataset=True``.

        ``"independent"``
            Create a separate ``FlowChain`` for each index along every
            leading batch axis.  Most expressive — each component /
            dataset learns a completely different density.
        ``"shared"``
            Use a single ``FlowChain`` conditioned on a one-hot index
            vector.  Parameter-efficient — components share the
            conditioner backbone and specialise via context.

        When neither ``is_mixture`` nor ``is_dataset`` is set this
        parameter is ignored.
    zero_init_output : bool, default True
        Zero-initialize the conditioner's output Dense layer so the
        flow starts as an identity transform.  Prevents log-determinant
        overflow at init in high dimensions.
    use_layer_norm : bool, default True
        Apply ``nn.LayerNorm`` after each hidden Dense layer in the
        conditioner MLP.  Stabilizes activations when fan-in is large.
    use_residual : bool, default True
        Add skip connections between consecutive hidden layers of the
        same width in the conditioner MLP.  Improves gradient flow.
    soft_clamp : bool, default True
        Use smooth asymmetric arctan clamp on affine coupling log-scale
        (Andrade 2024).  Bounds per-layer expansion to ~10%.
    use_loft : bool, default True
        Apply LOFT compression + trainable final affine after coupling
        layers.  Bounds sample magnitudes at extremes.

    Advantages
    ----------
    - Can represent multimodal, skewed, and heavy-tailed posteriors
    - Strictly more expressive than Gaussian (LowRank / MeanField)
    - Coupling flows are fast in both directions — ideal for SVI

    Disadvantages
    -------------
    - More parameters (flow network weights) than LowRank
    - Harder to interpret the learned covariance structure
    - May require tuning ``num_layers`` / ``hidden_dims``

    Examples
    --------
    >>> spec = LogNormalSpec(
    ...     "r", ("n_genes",), (0.0, 1.0),
    ...     is_gene_specific=True,
    ...     guide_family=NormalizingFlowGuide(
    ...         flow_type="spline_coupling", num_layers=4
    ...     ),
    ... )

    Mixture model with independent per-component flows::

        NormalizingFlowGuide(mixture_strategy="independent")

    Shared flow conditioned on component one-hot::

        NormalizingFlowGuide(mixture_strategy="shared")

    See Also
    --------
    LowRankGuide : Gaussian low-rank alternative.
    JointNormalizingFlowGuide : Joint version with cross-parameter flows.
    """

    flow_type: str = "spline_coupling"
    num_layers: int = 4
    hidden_dims: tuple = (64, 64)
    activation: str = "relu"
    n_bins: int = 8
    mixture_strategy: str = "independent"
    zero_init_output: bool = True
    use_layer_norm: bool = True
    use_residual: bool = True
    soft_clamp: bool = True
    use_loft: bool = True

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        _VALID_MIX = ("independent", "shared")
        if self.mixture_strategy not in _VALID_MIX:
            raise ValueError(
                f"mixture_strategy must be one of {_VALID_MIX}, "
                f"got {self.mixture_strategy!r}"
            )


# ------------------------------------------------------------------------------
# Joint Normalizing Flow Guide Family
# ------------------------------------------------------------------------------


@dataclass
class JointNormalizingFlowGuide(GuideFamily):
    """
    Marker for joint normalizing-flow across multiple parameter groups.

    Analogous to ``JointLowRankGuide`` but uses normalizing flows instead
    of low-rank Gaussians.  Cross-parameter dependencies are captured via
    the chain-rule decomposition:

        q(θ₁, θ₂) = q(θ₁) × q(θ₂ | θ₁)

    where each factor is a full normalizing flow.  The conditional
    ``q(θ₂ | θ₁)`` is implemented by passing the unconstrained sample
    of θ₁ as a continuous *context* vector to the flow for θ₂.

    Parameters
    ----------
    flow_type : str
        Type of flow layer (see ``NormalizingFlowGuide``).
    num_layers : int
        Number of flow layers per parameter block.
    hidden_dims : tuple of int
        Hidden-layer sizes for conditioner networks.
    activation : str
        Activation function for conditioner networks.
    n_bins : int
        Spline bins (for ``"spline_coupling"`` only).
    group : str
        Identifier linking parameters that share the same joint flow.
        All ``ParamSpec`` objects whose guide family is a
        ``JointNormalizingFlowGuide`` with the same ``group`` are
        modeled jointly.
    dense_params : list of str, optional
        Subset of parameter names in the group that go through the
        flow chain.  Non-dense parameters get diagonal Normal
        treatment with learned regression on the dense-flow
        residuals, mirroring ``JointLowRankGuide.dense_params``.
    mixture_strategy : str, default ``"independent"``
        How to handle leading batch axes (mixture components, datasets).
        ``"independent"`` creates per-index flows; ``"shared"`` uses one
        flow conditioned on a one-hot index vector.  Same semantics as
        ``NormalizingFlowGuide.mixture_strategy``.
    zero_init_output : bool, default True
        Zero-initialize the conditioner's output Dense layer so the flow starts
        as an identity transform.
    use_layer_norm : bool, default True
        Apply ``nn.LayerNorm`` after each hidden Dense layer in the conditioner
        MLP.
    use_residual : bool, default True
        Add skip connections between consecutive hidden layers of the same width
        in the conditioner MLP.
    soft_clamp : bool, default True
        Use smooth asymmetric arctan clamp on affine coupling log-scale
        (Andrade 2024).  Bounds per-layer expansion to ~10%.
    use_loft : bool, default True
        Apply LOFT compression + trainable final affine after coupling
        layers.  Bounds sample magnitudes at extremes.

    Advantages
    ----------
    - Captures non-linear cross-parameter dependencies
    - Each conditional is a full normalizing flow — more expressive than
      the Woodbury LowRankMVN conditionals
    - Natural extension to 3+ parameters via cumulative context

    Disadvantages
    -------------
    - More flow parameters than ``JointLowRankGuide``
    - Context-conditioned flows add dimensionality to conditioner nets

    Examples
    --------
    >>> joint = JointNormalizingFlowGuide(
    ...     flow_type="spline_coupling", num_layers=4, group="nb"
    ... )
    >>> PositiveNormalSpec("p", (), (0.0, 1.0), guide_family=joint)
    >>> PositiveNormalSpec("r", ("n_genes",), (0.0, 1.0),
    ...     is_gene_specific=True, guide_family=joint)

    See Also
    --------
    JointLowRankGuide : Gaussian joint alternative using Woodbury identity.
    NormalizingFlowGuide : Per-parameter (non-joint) flow guide.
    """

    flow_type: str = "spline_coupling"
    num_layers: int = 4
    hidden_dims: tuple = (64, 64)
    activation: str = "relu"
    n_bins: int = 8
    group: str = "default"
    dense_params: Optional[List[str]] = None
    mixture_strategy: str = "independent"
    zero_init_output: bool = True
    use_layer_norm: bool = True
    use_residual: bool = True
    soft_clamp: bool = True
    use_loft: bool = True

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        if not self.group:
            raise ValueError("group must be a non-empty string")
        _VALID_MIX = ("independent", "shared")
        if self.mixture_strategy not in _VALID_MIX:
            raise ValueError(
                f"mixture_strategy must be one of {_VALID_MIX}, "
                f"got {self.mixture_strategy!r}"
            )


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
