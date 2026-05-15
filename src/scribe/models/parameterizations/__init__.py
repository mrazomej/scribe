"""
Parameterization strategy classes for model construction.

This module provides a strategy pattern for handling different parameterization
schemes. Each parameterization encapsulates:

- Core parameters it requires (e.g., p+r, p+mu, phi+mu)
- How to build parameter specs (constrained vs unconstrained)
- Derived parameter computations
- Model-specific parameter transformations (e.g., p_capture → phi_capture)

This eliminates nested conditionals in preset factories and makes it easy to
add new parameterizations.

Classes
-------
Parameterization
    Abstract base class for parameterization schemes.
CanonicalParameterization
    Directly samples p (probability) and r (dispersion).
MeanProbParameterization
    Samples p (probability) and mu (mean), derives r.
MeanOddsParameterization
    Samples phi (odds ratio) and mu (mean), derives p and r.
LogisticNormalParameterization
    Population total-count parameters (``r_T``, ``p``) plus VAE-decoded
    ALR coordinates ``y_alr`` for compositional multinomial structure.

Examples
--------
>>> from scribe.models.parameterizations import PARAMETERIZATIONS
>>> param_strategy = PARAMETERIZATIONS["canonical"]
>>> param_specs = param_strategy.build_param_specs(
...     unconstrained=False,
...     guide_families=GuideFamilyConfig(),
... )
"""

from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple

import jax.numpy as jnp

from ..builders.parameter_specs import (
    BetaSpec,
    BetaPrimeSpec,
    DerivedParam,
    PositiveNormalSpec,
    LogNormalSpec,
    ParamSpec,
    SigmoidNormalSpec,
)
from ..config import GuideFamilyConfig

# ==============================================================================
# Abstract Base Class
# ==============================================================================


class Parameterization(ABC):
    """Abstract base class for parameterization schemes.

    Each parameterization defines:
        - What core parameters to sample
        - How to build parameter specs (constrained vs unconstrained)
        - What derived parameters to compute
        - How to transform model-specific parameter names

    Subclasses must implement all abstract methods and properties.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this parameterization."""
        pass

    @property
    @abstractmethod
    def core_parameters(self) -> List[str]:
        """List of core parameter names this parameterization requires."""
        pass

    @property
    @abstractmethod
    def gene_param_name(self) -> str:
        """Name of the gene-specific parameter (e.g., 'r' or 'mu')."""
        pass

    @abstractmethod
    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build parameter specs for this parameterization.

        Parameters
        ----------
        unconstrained : bool
            If True, use Normal+transform specs. Otherwise, use constrained
            distribution specs (Beta, LogNormal, etc.).
        guide_families : GuideFamilyConfig
            Per-parameter guide family configuration.
        n_components : Optional[int], default=None
            Number of mixture components. If provided, parameters in
            mixture_params will be marked as mixture-specific.
        mixture_params : Optional[List[str]], default=None
            List of parameter names to mark as mixture-specific.
            If None and n_components is set, defaults to all sampled core
            parameters for the selected parameterization.

        Returns
        -------
        List[ParamSpec]
            Parameter specifications for core parameters.
        """
        pass

    @abstractmethod
    def build_derived_params(self) -> List[DerivedParam]:
        """Build derived parameter computations.

        Returns
        -------
        List[DerivedParam]
            Derived parameter specifications.
        """
        pass

    def decoder_output_spec(self, base_model: str) -> List[Tuple[str, str]]:
        """Return (param_name, transform) for each decoder output head.

        The decoder produces the gene-specific core parameter for this
        parameterization. Additional heads (e.g., gate) are added based
        on the base_model type.

        Parameters
        ----------
        base_model : str
            Base model type: "nbdm", "zinb", "nbvcp", "zinbvcp".

        Returns
        -------
        List of (param_name, transform) tuples.
        """
        specs = [(self.gene_param_name, "softplus")]
        if base_model in ("zinb", "zinbvcp"):
            specs.append(("gate", "sigmoid"))
        return specs

    def transform_model_param(self, param_name: str) -> str:
        """Transform model-specific parameter names.

        Some parameterizations transform model-specific parameters.
        For example, mean_odds transforms p_capture → phi_capture.

        Parameters
        ----------
        param_name : str
            Original parameter name (e.g., "p_capture").

        Returns
        -------
        str
            Transformed parameter name (e.g., "phi_capture" for mean_odds,
            unchanged for others).
        """
        # Default: no transformation
        return param_name

    @property
    def requires_vae(self) -> bool:
        """Whether this parameterization is only supported with VAE inference.

        Returns
        -------
        bool
            False for standard parameterizations; subclasses may override.
        """
        return False


# ==============================================================================
# Concrete Parameterization Classes
# ==============================================================================

# ------------------------------------------------------------------------------
# Canonical Parameterization
# ------------------------------------------------------------------------------


class CanonicalParameterization(Parameterization):
    """Canonical parameterization: directly samples p and r.

    This is the standard parameterization where both the success probability
    p and dispersion r are sampled directly from their priors.

    Core parameters:
    - p: Success probability (Beta or SigmoidNormal)
    - r: Dispersion parameter (LogNormal or ExpNormal)

    Derived parameters:
    - mu: Mean expression ``mu = r * p / (1 - p)``.  Declared so that
      axis membership (dataset, mixture) propagates correctly from the
      core parameters to ``mu`` during layout inference.
    """

    @property
    def name(self) -> str:
        return "canonical"

    @property
    def core_parameters(self) -> List[str]:
        return ["p", "r"]

    @property
    def gene_param_name(self) -> str:
        return "r"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build parameter specs for canonical parameterization."""
        p_family = guide_families.get("p")
        r_family = guide_families.get("r")

        # Determine which parameters are mixture-specific
        if n_components is not None:
            if mixture_params is None:
                # Default: make all sampled core params mixture-specific
                mixture_params = ["p", "r"]
            is_p_mixture = "p" in mixture_params
            is_r_mixture = "r" in mixture_params
        else:
            is_p_mixture = False
            is_r_mixture = False

        if unconstrained:
            return [
                SigmoidNormalSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
                PositiveNormalSpec(
                    name="r",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=r_family,
                    is_mixture=is_r_mixture,
                ),
            ]
        else:
            return [
                BetaSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(1.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
                LogNormalSpec(
                    name="r",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=r_family,
                    is_mixture=is_r_mixture,
                ),
            ]

    # --------------------------------------------------------------------------

    def build_derived_params(self) -> List[DerivedParam]:
        """Derived parameter: mu = r * p / (1 - p).

        Although ``r`` and ``p`` are the *sampled* parameters in canonical
        mode, ``mu`` is derived from them during MAP / posterior extraction
        (see ``_compute_canonical_parameters``).  Declaring this dependency
        lets ``expand_membership_from_derived`` propagate dataset (and
        mixture) axis membership from ``r`` / ``p`` to ``mu``, ensuring
        that ``get_dataset()`` slices ``mu`` along the correct axis.
        """
        return [DerivedParam("mu", _compute_mu_from_r_p, ["r", "p"])]


# ------------------------------------------------------------------------------
# Mean Probability Parameterization
# ------------------------------------------------------------------------------


class MeanProbParameterization(Parameterization):
    """Mean-probability parameterization: samples p and mu, derives r.

    This parameterization samples the success probability p and mean expression
    mu, then derives the dispersion r as:

        r = mu * (1 - p) / p

    This links the dispersion to the mean, which can help capture correlations
    between p and r in the variational posterior.

    Core parameters:
    - p: Success probability (Beta or SigmoidNormal)
    - mu: Mean expression (LogNormal or ExpNormal)

    Derived parameters:
    - r: Dispersion (computed from p and mu)
    """

    @property
    def name(self) -> str:
        return "mean_prob"

    @property
    def core_parameters(self) -> List[str]:
        return ["p", "mu"]

    @property
    def gene_param_name(self) -> str:
        return "mu"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build parameter specs for mean-probability parameterization."""
        p_family = guide_families.get("p")
        mu_family = guide_families.get("mu")

        # Determine which parameters are mixture-specific
        if n_components is not None:
            if mixture_params is None:
                # Default: make all sampled core params mixture-specific
                mixture_params = ["p", "mu"]
            is_p_mixture = "p" in mixture_params
            is_mu_mixture = "mu" in mixture_params
        else:
            is_p_mixture = False
            is_mu_mixture = False

        if unconstrained:
            return [
                SigmoidNormalSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
                PositiveNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
            ]
        else:
            return [
                BetaSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(1.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
                LogNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
            ]

    # --------------------------------------------------------------------------

    def build_derived_params(self) -> List[DerivedParam]:
        """Build derived parameter: r = mu * (1 - p) / p."""
        return [DerivedParam("r", _compute_r_from_mu_p, ["p", "mu"])]


# ------------------------------------------------------------------------------
# Mean Odds Parameterization
# ------------------------------------------------------------------------------


class MeanOddsParameterization(Parameterization):
    """Mean-odds parameterization: samples phi and mu, derives p and r.

    This parameterization samples the odds ratio phi and mean expression mu,
    then derives both the success probability p and dispersion r:

        p = 1 / (1 + phi)
        r = mu * phi

    This is numerically more stable than mean_prob when p is close to 0 or 1,
    as it avoids division by p in the computation of r.

    Core parameters:
    - phi: Odds ratio (BetaPrime or ExpNormal)
    - mu: Mean expression (LogNormal or ExpNormal)

    Derived parameters:
    - p: Success probability (computed from phi)
    - r: Dispersion (computed from phi and mu)

    Model-specific transformations:
    - p_capture → phi_capture (for VCP models)
    """

    @property
    def name(self) -> str:
        return "mean_odds"

    @property
    def core_parameters(self) -> List[str]:
        return ["phi", "mu"]

    @property
    def gene_param_name(self) -> str:
        return "mu"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build parameter specs for mean-odds parameterization."""
        phi_family = guide_families.get("phi")
        mu_family = guide_families.get("mu")

        # Determine which parameters are mixture-specific
        if n_components is not None:
            if mixture_params is None:
                # Default: make all sampled core params mixture-specific
                mixture_params = ["phi", "mu"]
            is_phi_mixture = "phi" in mixture_params
            is_mu_mixture = "mu" in mixture_params
        else:
            is_phi_mixture = False
            is_mu_mixture = False

        if unconstrained:
            return [
                PositiveNormalSpec(
                    name="phi",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=phi_family,
                    is_mixture=is_phi_mixture,
                ),
                PositiveNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
            ]
        else:
            return [
                BetaPrimeSpec(
                    name="phi",
                    shape_dims=(),
                    default_params=(1.0, 1.0),
                    guide_family=phi_family,
                    is_mixture=is_phi_mixture,
                ),
                LogNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
            ]

    # --------------------------------------------------------------------------

    def build_derived_params(self) -> List[DerivedParam]:
        """Build derived parameters: p = 1/(1+phi), r = mu*phi."""
        return [
            DerivedParam("r", _compute_r_from_mu_phi, ["phi", "mu"]),
            DerivedParam("p", lambda phi: 1.0 / (1.0 + phi), ["phi"]),
        ]

    # --------------------------------------------------------------------------

    def transform_model_param(self, param_name: str) -> str:
        """Transform p_capture to phi_capture for mean_odds parameterization."""
        if param_name == "p_capture":
            return "phi_capture"
        return param_name


# ------------------------------------------------------------------------------
# Logistic-Normal Multinomial Parameterization
# ------------------------------------------------------------------------------


class LogisticNormalParameterization(Parameterization):
    """Logistic-normal multinomial parameterization for VAE-only models.

    The compositional path is identical across all variants: a per-cell
    latent ``z`` produces ALR coordinates ``y_alr`` via a linear decoder,
    then ``softmax(y_alr)`` gives the multinomial probabilities. The
    optional learned diagonal ALR noise (``d_lnm``) is controlled by
    ``ModelConfig.d_mode`` and wired in the VAE factory.

    The **totals NB submodel** ``u_T^{(c)} ~ NB(r_T, p_hat^{(c)})`` is
    the same NB distribution under all three variants below — the
    variants differ only in *which* parameter pair is sampled directly
    versus derived. This mirrors the DM family's
    ``canonical`` / ``mean_prob`` / ``mean_odds`` trio.

    Variants
    --------
    ``"canonical"``: samples ``(r_T, p)`` directly.

        - Faithful to the Poisson-thinning derivation in
          ``_capture_prior.qmd``.
        - Has a known ``(p, p_capture)`` aliasing under the capture
          anchor: with ``p_capture^{(c)}`` pinned by the anchor, the
          global ``p`` has no remaining identifying signal and drifts
          to its boundary.

    ``"mean_prob"``: samples ``(p, mu_T)``; derives ``r_T = mu_T (1-p)/p``.

        - ``mu_T`` is the population-level expected library size,
          directly identified by the empirical mean of ``u_T``.
        - Reduces but does not eliminate the ``p`` aliasing.

    ``"mean_odds"``: samples ``(phi_T, mu_T)``; derives
    ``p = 1/(1+phi_T)`` and ``r_T = mu_T * phi_T``.

        - ``mu_T`` identified by the empirical mean of ``u_T``;
          ``phi_T`` identified by the per-cell variance around the
          predicted mean. Both have independent identifying signal.
        - **Eliminates the ``(p, p_capture)`` aliasing entirely.**
          Recommended whenever the capture anchor is active.

    The compositional path (``y_alr``, multinomial likelihood,
    decoder/encoder architecture) is **identical** across the three
    variants. The variant choice only affects the totals NB submodel.

    Parameters
    ----------
    variant : {"canonical", "mean_prob", "mean_odds"}, default="canonical"
        Selects the totals-NB parameterization. See the per-variant
        notes above. The factory and registry expose three singletons
        under the keys ``"logistic_normal_canonical"``,
        ``"logistic_normal_mean_prob"``, and
        ``"logistic_normal_mean_odds"`` so the dispatch is purely by
        string lookup at config-build time.
    d_mode : {"low_rank", "learned"}, default="low_rank"
        Mirrors ``ModelConfig.d_mode``. Same meaning across variants.
    """

    def __init__(
        self,
        variant: Literal["canonical", "mean_prob", "mean_odds"] = "canonical",
        d_mode: Literal["low_rank", "learned"] = "low_rank",
    ):
        """Store variant + diagonal mode for introspection.

        Parameters
        ----------
        variant : {"canonical", "mean_prob", "mean_odds"}, default="canonical"
            Totals-NB parameterization selection.
        d_mode : {"low_rank", "learned"}, default="low_rank"
            See class docstring.
        """
        if variant not in ("canonical", "mean_prob", "mean_odds"):
            raise ValueError(
                f"variant must be 'canonical', 'mean_prob', or "
                f"'mean_odds'; got {variant!r}."
            )
        self._variant: Literal["canonical", "mean_prob", "mean_odds"] = variant
        self._init_d_mode: Literal["low_rank", "learned"] = d_mode

    # ------------------------------------------------------------------
    # Identity / introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        # The ``name`` property is used as a registry key, so each
        # variant gets its own canonical name. The shared
        # "logistic_normal" prefix is what the factory's family check
        # matches against.
        return f"logistic_normal_{self._variant}"

    @property
    def variant(self) -> str:
        """The totals-NB variant: ``canonical`` / ``mean_prob`` / ``mean_odds``."""
        return self._variant

    @property
    def core_parameters(self) -> List[str]:
        # Each variant samples a different pair of scalars. The ones
        # *not* listed here are derived (computed via
        # ``build_derived_params``) so the model builder still produces
        # ``r_T`` and ``p`` for the likelihood downstream.
        if self._variant == "canonical":
            return ["r_T", "p"]
        if self._variant == "mean_prob":
            return ["mu_T", "p"]
        # mean_odds
        return ["mu_T", "phi_T"]

    @property
    def gene_param_name(self) -> str:
        return "y_alr"

    @property
    def requires_vae(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Param spec construction
    # ------------------------------------------------------------------

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build specs for the variant's *sampled* scalars.

        Returns specs for the two sampled scalars only; the remaining
        NB parameters (``r_T`` / ``p``) are produced by
        :meth:`build_derived_params`.
        """
        # Resolve mixture-aware flags. The variant determines which
        # parameter names appear; ``mixture_params`` defaults to the
        # full sampled-core list so users get sensible mixture behavior
        # without re-specifying.
        if n_components is not None:
            default_mix = self.core_parameters
            if mixture_params is None:
                mixture_params = default_mix
            mixture_set = set(mixture_params)
        else:
            mixture_set = set()

        if self._variant == "canonical":
            return self._build_canonical_specs(
                unconstrained, guide_families, mixture_set
            )
        if self._variant == "mean_prob":
            return self._build_mean_prob_specs(
                unconstrained, guide_families, mixture_set
            )
        return self._build_mean_odds_specs(
            unconstrained, guide_families, mixture_set
        )

    def _build_canonical_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        mixture_set: set,
    ) -> List[ParamSpec]:
        """Specs for ``(r_T, p)`` — the historical LNM canonical path."""
        r_family = guide_families.get("r_T")
        p_family = guide_families.get("p")
        is_r_mixture = "r_T" in mixture_set
        is_p_mixture = "p" in mixture_set

        if unconstrained:
            return [
                PositiveNormalSpec(
                    name="r_T",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=r_family,
                    is_mixture=is_r_mixture,
                ),
                SigmoidNormalSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
            ]
        # Constrained LNM defaults intentionally use natural priors:
        #
        #   * ``p`` lives in (0, 1)         → ``BetaSpec``
        #   * ``r_T`` lives in (0, ∞)       → ``PositiveNormalSpec``
        #
        # ``PositiveNormalSpec`` is a Normal in log-space passed through a
        # configurable positive transform. The factory rewrites the
        # transform from ``model_config.positive_transform`` (default
        # ``"softplus"``) at spec-application time, so the semantics of
        # this branch track the user's chosen transform without hardcoding
        # it here. This avoids the LogNormal mode-vs-median trap where
        # ``LogNormal(μ, σ)`` with large ``σ`` has its mode at
        # ``exp(μ - σ²)`` — far from the median ``exp(μ)`` — and MAP
        # optimization underflows to zero on float32.
        return [
            PositiveNormalSpec(
                name="r_T",
                shape_dims=(),
                default_params=(0.0, 1.0),
                guide_family=r_family,
                is_mixture=is_r_mixture,
            ),
            BetaSpec(
                name="p",
                shape_dims=(),
                default_params=(1.0, 1.0),
                guide_family=p_family,
                is_mixture=is_p_mixture,
            ),
        ]

    def _build_mean_prob_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        mixture_set: set,
    ) -> List[ParamSpec]:
        """Specs for ``(mu_T, p)`` — mean-probability LNM totals.

        ``mu_T`` is a *scalar* (one value per dataset), unlike the DM
        family's per-gene ``mu_g``. This is intentional: the LNM keeps
        all gene-level structure on the compositional path
        (``y_alr``), so the totals submodel needs only the population
        expected library size.
        """
        mu_family = guide_families.get("mu_T")
        p_family = guide_families.get("p")
        is_mu_mixture = "mu_T" in mixture_set
        is_p_mixture = "p" in mixture_set

        if unconstrained:
            return [
                PositiveNormalSpec(
                    name="mu_T",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
                SigmoidNormalSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
            ]
        # Constrained LNM defaults intentionally use natural priors:
        #
        #   * ``p``    lives in (0, 1)      → ``BetaSpec``
        #   * ``mu_T`` lives in (0, ∞)      → ``PositiveNormalSpec``
        #
        # See ``_build_canonical_specs`` for the reasoning behind
        # ``PositiveNormalSpec`` (configurable transform via
        # ``model_config.positive_transform``, avoiding the LogNormal
        # mode-vs-median trap that lets MAP underflow to zero).
        return [
            PositiveNormalSpec(
                name="mu_T",
                shape_dims=(),
                default_params=(0.0, 1.0),
                guide_family=mu_family,
                is_mixture=is_mu_mixture,
            ),
            BetaSpec(
                name="p",
                shape_dims=(),
                default_params=(1.0, 1.0),
                guide_family=p_family,
                is_mixture=is_p_mixture,
            ),
        ]

    def _build_mean_odds_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        mixture_set: set,
    ) -> List[ParamSpec]:
        """Specs for ``(mu_T, phi_T)`` — mean-odds LNM totals.

        Most importantly, ``p`` is *not* sampled here; it is derived
        downstream as ``p = 1/(1+phi_T)``. Under the capture anchor,
        this eliminates the ``(p, p_capture)`` aliasing — both ``mu_T``
        and ``phi_T`` retain independent identifying signal in the
        data (mean and per-cell variance, respectively).
        """
        mu_family = guide_families.get("mu_T")
        phi_family = guide_families.get("phi_T")
        is_mu_mixture = "mu_T" in mixture_set
        is_phi_mixture = "phi_T" in mixture_set

        if unconstrained:
            return [
                PositiveNormalSpec(
                    name="mu_T",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
                PositiveNormalSpec(
                    name="phi_T",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=phi_family,
                    is_mixture=is_phi_mixture,
                ),
            ]
        # Constrained LNM defaults intentionally use natural priors:
        #
        #   * ``phi_T`` lives in (0, ∞)     → ``BetaPrimeSpec`` (mode at
        #     ``(α-1)/(β+1)`` for ``α≥1``, density ``∝ x^{α-1}`` near
        #     zero — actively discourages the boundary that LogNormal
        #     priors fall into when the user picks a wide ``σ``).
        #   * ``mu_T``  lives in (0, ∞)     → ``PositiveNormalSpec``
        #
        # See ``_build_canonical_specs`` for the reasoning behind
        # ``PositiveNormalSpec`` (configurable transform via
        # ``model_config.positive_transform``).
        return [
            PositiveNormalSpec(
                name="mu_T",
                shape_dims=(),
                default_params=(0.0, 1.0),
                guide_family=mu_family,
                is_mixture=is_mu_mixture,
            ),
            BetaPrimeSpec(
                name="phi_T",
                shape_dims=(),
                default_params=(1.0, 1.0),
                guide_family=phi_family,
                is_mixture=is_phi_mixture,
            ),
        ]

    # ------------------------------------------------------------------
    # Derived parameters
    # ------------------------------------------------------------------

    def build_derived_params(self) -> List[DerivedParam]:
        """Derive the NB parameters not directly sampled by this variant.

        The likelihood always reads ``r_T`` and ``p`` from the model's
        parameter dict, so non-canonical variants must declare these as
        derived. The model builder evaluates derived parameters before
        invoking the likelihood, so by the time
        ``LNMWithVCPLikelihood.sample`` reads them, both are present.
        """
        if self._variant == "canonical":
            # Both ``r_T`` and ``p`` are sampled directly; nothing to
            # derive. (We don't declare a "mu_T" derived here because
            # the likelihood doesn't need it.)
            return []
        if self._variant == "mean_prob":
            # r_T = mu_T * (1-p) / p
            return [
                DerivedParam(
                    "r_T", _compute_r_T_from_mu_T_p, ["p", "mu_T"]
                ),
            ]
        # mean_odds: derive p from phi_T, then r_T from mu_T and phi_T.
        return [
            DerivedParam(
                "p", _compute_p_from_phi_T, ["phi_T"]
            ),
            DerivedParam(
                "r_T", _compute_r_T_from_mu_T_phi_T, ["phi_T", "mu_T"]
            ),
        ]

    def decoder_output_spec(self, base_model: str) -> List[Tuple[str, str]]:
        """Return a single identity head for ALR coordinates ``y_alr``.

        The compositional path is unchanged across variants — only the
        totals NB submodel differs.
        """
        del base_model
        return [("y_alr", "identity")]


class PoissonLogNormalParameterization(Parameterization):
    """Poisson-LogNormal parameterization for VAE-only models.

    Each gene's count is Poisson with rate = exp(y_log_rate_g), where
    y_log_rate is the G-dimensional output of a linear-decoder VAE.
    There is no totals NB submodel -- total counts emerge naturally as
    the sum of per-gene Poissons.

    Unlike the LNM, the PLN has a **single natural parameterization**:
    the Gaussian parameters (mu, Sigma) in log-rate space, encoded via
    the linear decoder.  There are no canonical/mean_prob/mean_odds
    variants because there is no NB to reparameterize.

    Attributes
    ----------
    d_mode : {"low_rank", "learned"}, default="low_rank"
        Controls the diagonal noise model.  Same semantics as the LNM.
    """

    def __init__(self) -> None:
        """No-op constructor.

        The PLN parameterization carries no per-instance state. The
        diagonal-noise mode (``low_rank`` vs ``learned``) is read off
        ``model_config.d_mode`` at factory time, not from this class,
        because mode changes must propagate to the spec list and the
        likelihood instance simultaneously -- both of which take
        ``model_config`` directly. Keeping this constructor argument-
        free avoids the foot-gun of a stale per-instance ``d_mode``
        diverging from the active config.
        """
        # No per-instance state. See docstring above for the rationale.
        return

    # ------------------------------------------------------------------
    # Identity / introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "count_lognormal"

    @property
    def core_parameters(self) -> List[str]:
        # PLN has no totals submodel -- no r_T, p, mu_T, phi_T.
        return []

    @property
    def gene_param_name(self) -> str:
        return "y_log_rate"

    @property
    def requires_vae(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Param spec construction
    # ------------------------------------------------------------------

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build specs for PLN parameters.

        PLN has no sampled scalar parameters (no totals submodel).
        The optional diagonal noise ``d_pln`` is wired by the factory
        when ``d_mode="learned"``, analogous to ``d_lnm`` in LNM.

        Raises
        ------
        NotImplementedError
            If ``n_components > 1`` -- mixtures are deferred for PLN v1.
        """
        if n_components is not None and n_components > 1:
            raise NotImplementedError(
                "PLN v1 does not support mixture models "
                f"(n_components={n_components}). Mixtures are "
                "deferred to future work."
            )
        # No sampled core parameters for PLN.
        return []

    def build_derived_params(self) -> List[DerivedParam]:
        """No derived parameters for PLN (no totals submodel)."""
        return []

    def decoder_output_spec(self, base_model: str) -> List[Tuple[str, str]]:
        """Return a single identity head for log-rates ``y_log_rate``.

        The exponentiation to Poisson rates happens inside the
        likelihood, not on the decoder head.

        Parameters
        ----------
        base_model : str
            Base model type (expected to be ``"pln"``).

        Returns
        -------
        list of (str, str)
            ``[("y_log_rate", "identity")]``.
        """
        del base_model
        return [("y_log_rate", "identity")]


# ==============================================================================
# NOTE: Hierarchical parameterization classes have been removed.
# Hierarchical priors on p/phi and gate are now controlled via boolean flags
# (hierarchical_p, hierarchical_gate) in ModelConfig. The factory dynamically
# replaces flat specs with hierarchical triplets when these flags are set.
# ==============================================================================

# ==============================================================================
# Derived Parameter Compute Functions
# ==============================================================================
#
# These functions contain *only* pure math.  Shape alignment across
# different axis layouts (e.g. mu with a dataset axis vs. phi without)
# is handled by the call site in ``ModelBuilder`` using
# ``merge_layouts`` / ``align_to_layout`` before invoking these.


def _compute_r_from_mu_phi(phi: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Compute the dispersion parameter ``r = mu * phi``.

    Inputs are expected to be pre-aligned by the caller (i.e. already
    broadcast-compatible).  The model builder uses ``AxisLayout`` metadata
    to insert singleton dimensions where axes differ before calling this.

    Parameters
    ----------
    phi : jnp.ndarray
        Odds ratio parameter (pre-aligned).
    mu : jnp.ndarray
        Mean parameter (pre-aligned).

    Returns
    -------
    jnp.ndarray
        Dispersion parameter ``r = mu * phi``.
    """
    return mu * phi


# ------------------------------------------------------------------------------


def _compute_mu_from_r_p(r: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Compute the mean parameter ``mu = r * p / (1 - p)``.

    Used by the canonical parameterization's ``DerivedParam`` declaration
    so that ``expand_membership_from_derived`` can propagate axis
    membership from ``r`` and ``p`` to ``mu``.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion parameter (pre-aligned).
    p : jnp.ndarray
        Success probability parameter (pre-aligned).

    Returns
    -------
    jnp.ndarray
        Mean parameter ``mu = r * p / (1 - p)``.
    """
    return r * p / (1 - p)


# ------------------------------------------------------------------------------


def _compute_r_from_mu_p(p: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Compute the dispersion parameter ``r = mu * (1 - p) / p``.

    Inputs are expected to be pre-aligned by the caller (i.e. already
    broadcast-compatible).  The model builder uses ``AxisLayout`` metadata
    to insert singleton dimensions where axes differ before calling this.

    Parameters
    ----------
    p : jnp.ndarray
        Success probability parameter (pre-aligned).
    mu : jnp.ndarray
        Mean parameter (pre-aligned).

    Returns
    -------
    jnp.ndarray
        Dispersion parameter ``r = mu * (1 - p) / p``.
    """
    return mu * (1 - p) / p


# ------------------------------------------------------------------------------
# LNM-totals derived computations.
# ------------------------------------------------------------------------------
#
# These are the LNM-totals analogs of the DM-family ``_compute_r_from_*``
# helpers above. They live in this module so the parameterization classes
# can declare them as ``DerivedParam(name, fn, deps)`` without importing
# anything else; the model builder picks them up through the standard
# derived-param dispatch.


def _compute_r_T_from_mu_T_p(
    p: jnp.ndarray, mu_T: jnp.ndarray
) -> jnp.ndarray:
    """LNM mean-prob: ``r_T = mu_T * (1 - p) / p``.

    Mirrors :func:`_compute_r_from_mu_p` for the LNM totals NB. ``mu_T``
    is a scalar (population-level expected library size); ``p`` is the
    scalar global NB success probability.
    """
    return mu_T * (1 - p) / p


def _compute_r_T_from_mu_T_phi_T(
    phi_T: jnp.ndarray, mu_T: jnp.ndarray
) -> jnp.ndarray:
    """LNM mean-odds: ``r_T = mu_T * phi_T``.

    Same algebraic form as :func:`_compute_r_from_mu_phi`; both
    arguments are scalars in the LNM totals submodel.
    """
    return mu_T * phi_T


def _compute_p_from_phi_T(phi_T: jnp.ndarray) -> jnp.ndarray:
    """LNM mean-odds: ``p = 1 / (1 + phi_T)``.

    Mirrors the DM family's ``p = 1/(1+phi)`` derivation but applies
    to the totals NB success probability rather than the per-gene one.
    Provided as a top-level (rather than ``lambda``) so that
    ``DerivedParam`` instances are picklable.
    """
    return 1.0 / (1.0 + phi_T)


# ==============================================================================
# Two-State Natural Parameterization
# ==============================================================================


class TwoStateParameterization(Parameterization):
    """Two-state natural parameterization for the Poisson-Beta compound.

    Samples ``mu`` (per-gene mean) as the only core parameter. The
    two other per-gene parameters of the two-state model —
    ``burst_size`` and ``k_off`` — are introduced via
    ``MODEL_EXTRA_PARAMS`` rather than as core parameters of this
    class so that the parameterization layer remains a thin
    description of the "mean expression" sampling strategy and the
    model-specific extras describe how that mean is shaped into the
    full distribution. See the paper section
    @sec-twostate-reparam for the math.

    Implementation notes
    --------------------
    - There are no NB-style derived parameters (``r``, ``p``, ``phi``).
      The derived (alpha, beta, rate) quantities are computed inside
      ``TwoStateLikelihood._build_dist`` from (mu, burst_size, k_off)
      and exposed as ``numpyro.deterministic`` sites.
    - The ``unconstrained`` argument to :meth:`build_param_specs` is
      kept for signature compatibility but ignored — the spec class
      chosen reflects the model config's ``positive_transform`` field
      (the factory passes the resolved transform via the constructor).
    """

    @property
    def name(self) -> str:
        return "two_state_natural"

    @property
    def core_parameters(self) -> List[str]:
        return ["mu"]

    @property
    def gene_param_name(self) -> str:
        return "mu"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build the mu spec.

        Uses :class:`PositiveNormalSpec` (Normal + transform). The
        transform itself is set by the factory based on the model
        config's ``positive_transform`` field; this method just
        provides the loc/scale defaults.

        Phase 1 explicitly does not support mixtures, so we ignore
        ``n_components`` and ``mixture_params``.
        """
        del unconstrained, n_components, mixture_params  # phase 1 ignores
        mu_family = guide_families.get("mu")
        return [
            PositiveNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=mu_family,
                is_mixture=False,
            ),
        ]

    def build_derived_params(self) -> List[DerivedParam]:
        """No NB-style derived parameters at the parameterization layer.

        ``alpha``, ``beta``, and ``rate`` are computed inside
        :class:`TwoStateLikelihood._build_dist` from
        (mu, burst_size, k_off) and exposed as
        :func:`numpyro.deterministic` sites at gene rank.
        """
        return []


class TwoStateRatioParameterization(Parameterization):
    """Relative-switching parameterization for the Poisson-Beta compound.

    Samples ``mu`` (per-gene mean) as the only core parameter, exactly
    like :class:`TwoStateParameterization`.  The third per-gene
    parameter — the regime variable — is :math:`s = k_{off} / k_{on}`
    (``switching_ratio``), in place of the absolute ``k_off`` used by
    ``two_state_natural``.  ``burst_size`` is shared between both
    parameterizations.

    Why this exists
    ---------------
    The data identifies the gene's regime — NB-like vs bursty — through
    the ratio :math:`k_{off}/k_{on}`, not through absolute ``k_off``.
    Sampling ``k_off`` directly forces mean-field q to factor across
    coordinates that are *not* approximately independent in the
    posterior: q(k_off) is geometrically coupled to q(mu) and
    q(burst_size).  Sampling the ratio aligns the variational
    factorization with the data's identifiability axes, mirroring how
    NBDM's ``mean_prob`` / ``mean_odds`` aligns with (mean, dispersion)
    rather than (r, p).

    Forward map (computed in the likelihood)::

        k_on  = mu / burst_size
        k_off = switching_ratio * k_on   (= s · k_on)
        alpha = k_on
        beta  = k_off
        r_hat = mu · (1 + switching_ratio)

    Mean preservation::

        E[count] = r_hat · alpha / (alpha + beta)
                 = mu(1+s) · k_on / (k_on + s · k_on)
                 = mu

    NB limit (``switching_ratio → ∞``)::

        beta → ∞ at fixed alpha = mu/burst_size  →  NB(r=mu/burst_size,
        burst=burst_size).

    Implementation notes
    --------------------
    - ``switching_ratio`` is the third per-gene extra (instead of
      ``k_off``) for the ``twostate`` / ``twostatevcp`` models; the
      factory dispatches the extras list at build time based on the
      configured parameterization (see ``presets/factory.py``).
    - ``alpha``, ``beta``, ``r_hat`` are still computed inside
      ``TwoStateLikelihood._build_dist`` and exposed as
      ``numpyro.deterministic`` sites for posterior introspection.
    """

    @property
    def name(self) -> str:
        return "two_state_ratio"

    @property
    def core_parameters(self) -> List[str]:
        return ["mu"]

    @property
    def gene_param_name(self) -> str:
        return "mu"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build the mu spec — identical to the natural parameterization.

        The other two per-gene specs (``burst_size``,
        ``switching_ratio``) come from the model extras dispatch.
        """
        del unconstrained, n_components, mixture_params
        mu_family = guide_families.get("mu")
        return [
            PositiveNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=mu_family,
                is_mixture=False,
            ),
        ]

    def build_derived_params(self) -> List[DerivedParam]:
        return []


class TwoStateMeanFanoParameterization(Parameterization):
    """Mean-Fano parameterization for the Poisson-Beta compound.

    Samples ``(mu, excess_fano, concentration)`` where:

    * ``mu`` is the per-gene mean (same as the other TwoState
      parameterizations).
    * ``excess_fano = Var[X]/E[X] − 1`` is the excess Fano factor.
      Directly bounds the posterior-predictive variance per gene.
    * ``concentration = α + β`` is the Beta concentration (sum of
      ON and OFF rates).  Large ``concentration`` peaks the
      latent ``p`` distribution and yields an NB-like shape;
      small ``concentration`` admits a U-shaped Beta and bursty /
      bimodal counts.

    Forward map (computed in the likelihood)::

        denom = mu + excess_fano · (concentration + 1)
        alpha = concentration · mu / denom        ( = k_on )
        beta  = concentration · excess_fano · (concentration + 1)
              / denom                             ( = k_off )
        r_hat = denom

    Moment guarantees::

        E[count]            = mu                  identically.
        Var[count] / E[count] − 1 = excess_fano   identically.

    NB limit (``concentration → ∞``)::

        alpha → mu / excess_fano   ( = NB shape r_NB )
        beta  → concentration
        r_hat → excess_fano · concentration
        Equivalent NB:  NegBin(r_NB = mu/excess_fano,
                               burst_size = excess_fano).

    Why this parameterization
    -------------------------
    The PPC width — i.e. the posterior-predictive marginal variance
    per gene — is governed by ``Var[count] = mu · (1 + excess_fano)``.
    Mean-field q on the natural parameters ``(b, k_off)`` is forced
    to discover a curved manifold along which the implied Fano stays
    consistent across gene magnitudes; independent q-draws on ``b``
    and ``k_off`` (or on ``b`` and ``switching_ratio``) easily mix
    into too-wide an implied Fano.

    Sampling ``excess_fano`` directly removes that constraint: q(
    excess_fano) is *the* PPC-width control, and q(concentration)
    only describes how peaked the latent Beta is.  This is the
    closest TwoState analog of NBDM's ``mean_odds`` — sample the
    quantities that determine the first two observable moments,
    leave the third for shape.
    """

    @property
    def name(self) -> str:
        return "two_state_mean_fano"

    @property
    def core_parameters(self) -> List[str]:
        return ["mu"]

    @property
    def gene_param_name(self) -> str:
        return "mu"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build the mu spec — identical to the other TwoState parameterizations.

        The other two per-gene specs (``excess_fano``,
        ``concentration``) come from the model extras dispatch.
        """
        del unconstrained, n_components, mixture_params
        mu_family = guide_families.get("mu")
        return [
            PositiveNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=mu_family,
                is_mixture=False,
            ),
        ]

    def build_derived_params(self) -> List[DerivedParam]:
        return []


class TwoStateMomentDeltaParameterization(Parameterization):
    """Moment-delta parameterization for the Poisson-Beta compound.

    Samples ``(mu, excess_fano, inv_concentration)`` where:

    * ``mu`` is the per-gene mean.
    * ``excess_fano`` is ``Var[X]/E[X] − 1`` (identical to its role
      in the mean-Fano parameterization).
    * ``inv_concentration = delta = 1 / (concentration + 1) ∈ (0, 1)``
      is the bounded shape coordinate.

    Same first-two-moment guarantees as the mean-Fano variant
    (``E[count] = mu`` and ``Var/Mean - 1 = excess_fano`` by
    construction), but with the shape axis re-mapped from the
    unbounded ``concentration > 0`` to a bounded ``delta ∈ (0, 1)``.

    Why this exists
    ---------------
    The NB limit in the mean-Fano coordinates is ``concentration →
    ∞`` — an unbounded direction along which the mean-field guide
    can waste posterior mass.  Mapping ``delta = 1/(concentration +
    1)`` compresses that ridge into ``delta → 0``.  Since ``delta``
    is bounded, a logit-normal (sigmoid-Normal) guide is the
    natural choice: ``logit(delta) ~ Normal(loc, scale)`` and
    ``delta = sigmoid(logit(delta))``.  The NB-default prior tilt
    becomes a left-shifted mean on the logit (``loc ≈ -4`` puts
    most mass at small delta, i.e. NB-like concentrations).

    Forward map (computed in the likelihood)::

        denom = mu·delta + excess_fano                 ( = (mu + e/delta) · delta )
        alpha = mu · (1 - delta) / denom               ( = k_on )
        beta  = excess_fano · (1 - delta) / (delta · denom)
                                                       ( = k_off )
        r_hat = (mu·delta + excess_fano) / delta       ( = mu + e/delta )

    Equivalent forms (with ``kappa = (1 - delta)/delta``,
    ``s = excess_fano / (mu·delta)``)::

        alpha = kappa · mu·delta / denom = kappa / (1 + s)
        beta  = kappa · s / (1 + s)
        r_hat = mu · (1 + s)

    Moment guarantees::

        E[count]                   = mu
        Var[count] / E[count] - 1  = excess_fano

    NB limit (``delta → 0``)::

        alpha → mu / excess_fano
        beta  → ∞
        r_hat → excess_fano / delta → ∞

    recovering NB(shape = mu/excess_fano, burst = excess_fano).
    """

    @property
    def name(self) -> str:
        return "two_state_moment_delta"

    @property
    def core_parameters(self) -> List[str]:
        return ["mu"]

    @property
    def gene_param_name(self) -> str:
        return "mu"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build the mu spec; ``excess_fano`` and ``inv_concentration``
        come from the model extras dispatch."""
        del unconstrained, n_components, mixture_params
        mu_family = guide_families.get("mu")
        return [
            PositiveNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=mu_family,
                is_mixture=False,
            ),
        ]

    def build_derived_params(self) -> List[DerivedParam]:
        return []


# ==============================================================================
# Parameterization Registry
# ==============================================================================

# Create singleton instances. The LNM family has three variants
# mirroring the DM trio (canonical / mean_prob / mean_odds) — the
# user-facing ``parameterization=`` string in ``scribe.fit`` is mapped
# to one of these LNM-family keys by the preset builder when the model
# is ``"lnm"`` / ``"lnmvcp"``.
_canonical = CanonicalParameterization()
_mean_prob = MeanProbParameterization()
_mean_odds = MeanOddsParameterization()
_logistic_normal_canonical = LogisticNormalParameterization(variant="canonical")
_logistic_normal_mean_prob = LogisticNormalParameterization(variant="mean_prob")
_logistic_normal_mean_odds = LogisticNormalParameterization(variant="mean_odds")
_poisson_lognormal = PoissonLogNormalParameterization()
_two_state_natural = TwoStateParameterization()
_two_state_ratio = TwoStateRatioParameterization()
_two_state_mean_fano = TwoStateMeanFanoParameterization()
_two_state_moment_delta = TwoStateMomentDeltaParameterization()

# Registry mapping internal parameterization keys to singleton instances.
# The LNM family appears under three keys mirroring the DM-family trio.
# User-facing strings ``"canonical"`` / ``"mean_prob"`` / ``"mean_odds"``
# from ``scribe.fit`` are mapped to the right LNM-family key by the
# preset builder when the model is ``"lnm"`` / ``"lnmvcp"``.
PARAMETERIZATIONS = {
    # DM-family parameterizations
    "canonical": _canonical,
    "mean_prob": _mean_prob,
    "mean_odds": _mean_odds,
    # LNM-family parameterizations (variants of the totals NB)
    "logistic_normal_canonical": _logistic_normal_canonical,
    "logistic_normal_mean_prob": _logistic_normal_mean_prob,
    "logistic_normal_mean_odds": _logistic_normal_mean_odds,
    # PLN parameterization (single variant, no totals submodel)
    "count_lognormal": _poisson_lognormal,
    # Two-state natural parameterization (Poisson-Beta compound; samples
    # mu only — burst_size and k_off come in as MODEL_EXTRA_PARAMS).
    "two_state_natural": _two_state_natural,
    # Two-state relative-switching parameterization: samples (mu,
    # burst_size, switching_ratio) where ratio = k_off/k_on.  See
    # TwoStateRatioParameterization for the math.
    "two_state_ratio": _two_state_ratio,
    # Two-state mean-Fano parameterization: samples (mu, excess_fano,
    # concentration).  Targets PPC width directly via q(excess_fano);
    # concentration carries the NB-vs-bursty shape.  See
    # TwoStateMeanFanoParameterization for the math.
    "two_state_mean_fano": _two_state_mean_fano,
    # Two-state moment-delta parameterization: samples (mu,
    # excess_fano, inv_concentration) with delta = 1/(kappa+1) in
    # (0, 1).  Same moment guarantees as mean_fano; bounded shape
    # coordinate avoids wasted mass over arbitrarily large
    # concentration values.  See TwoStateMomentDeltaParameterization
    # for the math.
    "two_state_moment_delta": _two_state_moment_delta,
    # Short alias for the two-state natural parameterization. Both keys
    # resolve to the same singleton; ``"natural"`` is preferred for
    # interactive use, ``"two_state_natural"`` is the canonical form
    # appearing in serialised configs.
    "natural": _two_state_natural,
    # Short alias for the relative-switching parameterization.
    "ratio": _two_state_ratio,
    # Short aliases for the mean-Fano parameterization.
    "mean_fano": _two_state_mean_fano,
    "fano": _two_state_mean_fano,
    # Short aliases for the moment-delta parameterization.
    "moment_delta": _two_state_moment_delta,
    "delta": _two_state_moment_delta,
    # Backward compatibility for the DM family
    "standard": _canonical,
    "linked": _mean_prob,
    "odds_ratio": _mean_odds,
}


# ------------------------------------------------------------------------------
# LNM-family helpers
# ------------------------------------------------------------------------------


# Set of internal parameterization keys belonging to the LNM family.
# Used by the factory and by ``resolve_lnm_priors`` to detect the LNM
# path without comparing against a hardcoded ``"logistic_normal"`` string
# (which now no longer exists as a single key).
_LNM_FAMILY_KEYS: frozenset = frozenset(
    {
        "logistic_normal_canonical",
        "logistic_normal_mean_prob",
        "logistic_normal_mean_odds",
    }
)


def is_count_lognormal_family(param_key: str) -> bool:
    """Return ``True`` for the count-LogNormal parameterization key.

    Covers both the PLN base model (Poisson observation channel) and
    the NBLN base model (NB observation channel); they share the same
    Gaussian-on-log-rate latent structure.  The legacy name was
    ``is_poisson_lognormal_family`` (kept as a backward-compatible
    alias below).

    Parameters
    ----------
    param_key : str
        Internal parameterization key.

    Returns
    -------
    bool
        ``True`` if ``param_key == "count_lognormal"`` (or its legacy
        alias ``"poisson_lognormal"``).
    """
    return param_key in ("count_lognormal", "poisson_lognormal")


# Backward-compatible alias.  External callers / older imports may
# reference ``is_poisson_lognormal_family``; keep the binding so they
# continue to work without modification.
is_poisson_lognormal_family = is_count_lognormal_family


def is_logistic_normal_family(param_key: str) -> bool:
    """Return ``True`` for any LNM-family parameterization key.

    Used by the unified factory and the prior resolver to dispatch on
    the LNM family without enumerating every variant. Comparing against
    a single ``"logistic_normal"`` string is no longer meaningful since
    we now have three distinct LNM-family keys, so this helper is the
    canonical way to ask "is this an LNM parameterization?" anywhere
    in the codebase.

    Parameters
    ----------
    param_key : str
        Internal parameterization key (lowercased), e.g.
        ``"canonical"``, ``"logistic_normal_mean_odds"``, ``"linked"``.

    Returns
    -------
    bool
        ``True`` if the key is one of ``"logistic_normal_canonical"``,
        ``"logistic_normal_mean_prob"``, or
        ``"logistic_normal_mean_odds"``; ``False`` otherwise.

    Examples
    --------
    >>> is_logistic_normal_family("logistic_normal_canonical")
    True
    >>> is_logistic_normal_family("logistic_normal_mean_odds")
    True
    >>> is_logistic_normal_family("canonical")  # DM-family
    False
    >>> is_logistic_normal_family("nbdm")
    False
    """
    return param_key in _LNM_FAMILY_KEYS


def resolve_user_parameterization_for_model(
    model: str,
    parameterization: str,
) -> str:
    """Map a user-facing ``parameterization=`` string to an internal key.

    The user calls ``scribe.fit(model="lnm", parameterization="mean_odds")``;
    internally we want to dispatch to the
    ``"logistic_normal_mean_odds"`` parameterization. Conversely, for
    DM-family models, ``parameterization="mean_odds"`` should resolve
    to the DM-family key ``"mean_odds"`` unchanged. This helper
    encapsulates that mapping.

    Parameters
    ----------
    model : str
        Model name passed to ``scribe.fit`` (case-insensitive).
    parameterization : str
        User-facing parameterization string, one of
        ``"canonical"`` / ``"mean_prob"`` / ``"mean_odds"`` (or one of
        the legacy aliases ``"standard"`` / ``"linked"`` /
        ``"odds_ratio"``) for both families.

    Returns
    -------
    str
        Internal parameterization key suitable for indexing
        :data:`PARAMETERIZATIONS`.

    Raises
    ------
    ValueError
        If ``parameterization`` is not a recognized DM-family value
        (when the model is non-LNM) or one of the three LNM variants
        (when the model is ``"lnm"`` / ``"lnmvcp"``).
    """
    model_lower = model.lower()

    # PLN and NBLN share a single parameterization (POISSON_LOGNORMAL):
    # both use the VAE log-rate decoder. NBLN adds gene dispersion
    # ``r_g`` via ``MODEL_EXTRA_PARAMS["nbln"] = ["r"]`` rather than
    # through a distinct parameterization. Always resolve to it
    # regardless of the user-supplied parameterization string.
    if model_lower in ("pln", "nbln"):
        return "count_lognormal"

    # TwoState (Poisson-Beta compound) accepts four parameterizations:
    # ``two_state_natural``      — samples (mu, burst_size, k_off)
    # ``two_state_ratio``        — samples (mu, burst_size, switching_ratio)
    # ``two_state_mean_fano``    — samples (mu, excess_fano, concentration)
    # ``two_state_moment_delta`` — samples (mu, excess_fano, inv_concentration)
    # Reject DM-family strings with a directive to the supported
    # values rather than silently re-mapping.
    if model_lower in ("twostate", "twostatevcp"):
        param_lower = parameterization.lower()
        if param_lower in ("two_state_natural", "natural"):
            return "two_state_natural"
        if param_lower in ("two_state_ratio", "ratio"):
            return "two_state_ratio"
        if param_lower in ("two_state_mean_fano", "mean_fano", "fano"):
            return "two_state_mean_fano"
        if param_lower in ("two_state_moment_delta", "moment_delta", "delta"):
            return "two_state_moment_delta"
        raise ValueError(
            f"parameterization={parameterization!r} is not supported for "
            f"model={model!r}. Choose 'two_state_natural' (samples k_off "
            f"directly; aliases: 'natural'), 'two_state_ratio' (samples "
            f"switching_ratio = k_off/k_on; aliases: 'ratio'), "
            f"'two_state_mean_fano' (samples excess_fano + concentration; "
            f"aliases: 'mean_fano', 'fano'), or 'two_state_moment_delta' "
            f"(samples excess_fano + inv_concentration; aliases: "
            f"'moment_delta', 'delta')."
        )

    param_lower = parameterization.lower()

    # Backward-compat: the legacy DM-family aliases collapse to the
    # canonical short names before we branch on the model family.
    legacy_dm_aliases = {
        "standard": "canonical",
        "linked": "mean_prob",
        "odds_ratio": "mean_odds",
    }
    param_lower = legacy_dm_aliases.get(param_lower, param_lower)

    if model_lower in ("lnm", "lnmvcp"):
        # The LNM family accepts only the three variant names. Emit a
        # specific error for the legacy ``"logistic_normal"`` string
        # so users who upgrade get a clear migration message rather
        # than a generic "unknown parameterization" failure.
        if param_lower == "logistic_normal":
            raise ValueError(
                "parameterization='logistic_normal' is no longer "
                "accepted for LNM models. Choose one of "
                "'canonical' (legacy default; faithful to "
                "Poisson-thinning), 'mean_prob', or 'mean_odds' "
                "(recommended under capture anchor)."
            )
        if param_lower not in ("canonical", "mean_prob", "mean_odds"):
            raise ValueError(
                f"parameterization={parameterization!r} not supported "
                f"for LNM models. Choose 'canonical', 'mean_prob', or "
                f"'mean_odds'."
            )
        return f"logistic_normal_{param_lower}"

    # DM-family path. The legacy "logistic_normal" string would never
    # have made sense here, so we don't special-case it.
    if param_lower not in ("canonical", "mean_prob", "mean_odds"):
        raise ValueError(
            f"parameterization={parameterization!r} not recognized."
        )
    return param_lower


__all__ = [
    "Parameterization",
    # Standard
    "CanonicalParameterization",
    "MeanProbParameterization",
    "MeanOddsParameterization",
    "LogisticNormalParameterization",
    "PoissonLogNormalParameterization",
    "TwoStateParameterization",
    "TwoStateRatioParameterization",
    "TwoStateMeanFanoParameterization",
    "TwoStateMomentDeltaParameterization",
    "PARAMETERIZATIONS",
    # Family helpers
    "is_logistic_normal_family",
    "is_count_lognormal_family",
    # Backward-compatible alias.  Prefer ``is_count_lognormal_family``
    # in new code; this name covers both the PLN (Poisson) and NBLN
    # (Negative Binomial) base models which share the same
    # parameterization.
    "is_poisson_lognormal_family",
    "resolve_user_parameterization_for_model",
    # Derived parameter compute functions (pure math, alignment-free)
    "_compute_mu_from_r_p",
    "_compute_r_from_mu_phi",
    "_compute_r_from_mu_p",
    "_compute_r_T_from_mu_T_p",
    "_compute_r_T_from_mu_T_phi_T",
    "_compute_p_from_phi_T",
]
