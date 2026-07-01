"""
Parameter mapping system for SCRIBE model configurations.

This module defines which parameters are active for each parameterization type,
making the system more maintainable and less error-prone than hardcoded if-statements.

It also provides the canonical mappings between internal (short/math) parameter
names and user-friendly descriptive names, used by the ``priors`` dict alias
system and the ``descriptive_names`` option on results objects.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union
from dataclasses import dataclass
from .enums import Parameterization

# ==============================================================================
# Parameter Info Class
# ==============================================================================


class ParameterInfo(NamedTuple):
    """
    Holds metadata about a model parameter in a given context.

    This class encapsulates not only the name and description of the parameter,
    but also flags indicating its necessity with respect to specific model
    parameterizations or model types.

    Attributes
    ----------
    name : str
        The name of the parameter (e.g., "p", "r", "mu", etc.).
    description : str
        A human-readable description of what the parameter represents.
    required_for_parameterization : bool, optional
        Whether this parameter is required for the specific parameterization
        (e.g., "linked", "odds_ratio"). Defaults to False.
    required_for_model_type : bool, optional
        Whether this parameter is required for this kind of model
        (e.g., ZINB, NBDM, models with VCP capture probability, etc.).
        Defaults to False.
    """

    name: str
    description: str
    required_for_parameterization: bool = False
    required_for_model_type: bool = False


# ==============================================================================
# Parameterization Mapping Class
# ==============================================================================


@dataclass(frozen=True)
class ParameterizationMapping:
    """
    Represents the mapping from a model parameterization to its supported
    parameters.

    This class encapsulates the relationship between a parameterization type
    (such as "standard", "linked", or "odds_ratio") and the set of model
    parameters that are valid (core and optional) for that parameterization.

    Attributes
    ----------
    parameterization : Parameterization
        The parameterization type this mapping refers to (e.g.
        Parameterization.STANDARD).
    core_parameters : Set[str]
        The set of parameters required for this parameterization; these must
        always be present for the parameterization to be valid. For example,
        {"p", "r"} for "standard".
    optional_parameters : Set[str]
        The set of parameters that are supported by the parameterization but not
        required; these may be present depending on model extensions or specific
        model types.
    parameter_descriptions : Dict[str, str]
        Human-readable descriptions for each parameter in this mapping, keyed by
        parameter name.

    Methods
    -------
    get_all_parameters() -> Set[str]
        Returns the union of core and optional parameters valid for this
        parameterization.
    is_parameter_supported(param_name: str) -> bool
        Returns True if the given parameter name is either required (core) or
        optional for this parameterization.
    is_parameter_required(param_name: str) -> bool
        Returns True if the given parameter name is required for this
        parameterization.

    Examples
    --------
    >>> mapping = ParameterizationMapping(
    ...     parameterization=Parameterization.STANDARD,
    ...     core_parameters={"p", "r"},
    ...     optional_parameters=set(),
    ...     parameter_descriptions={"p": "Success probability", "r": "Dispersion"}
    ... )
    >>> mapping.is_parameter_supported("p")
    True
    >>> mapping.is_parameter_required("p")
    True
    >>> mapping.is_parameter_supported("gate")
    False
    """

    # The parameterization type for this mapping (e.g., STANDARD, LINKED,
    # ODDS_RATIO).
    parameterization: Parameterization
    # The set of core/required parameters for this parameterization.
    core_parameters: Set[str]
    # The set of optional parameters for this parameterization (may or may not
    # be present).
    optional_parameters: Set[str]
    # Human-readable descriptions for each parameter.
    parameter_descriptions: Dict[str, str]

    # --------------------------------------------------------------------------
    # Methods
    # --------------------------------------------------------------------------

    def get_all_parameters(self) -> Set[str]:
        """Get all parameters (core + optional) for this parameterization."""
        return self.core_parameters | self.optional_parameters

    def is_parameter_supported(self, param_name: str) -> bool:
        """Check if a parameter is supported by this parameterization."""
        return param_name in self.get_all_parameters()

    def is_parameter_required(self, param_name: str) -> bool:
        """Check if a parameter is required (core) for this parameterization."""
        return param_name in self.core_parameters


# ==============================================================================
# Parameter Mappings for Each Parameterization
# ==============================================================================

PARAMETERIZATION_MAPPINGS = {
    # Old names (backward compatibility)
    Parameterization.STANDARD: ParameterizationMapping(
        parameterization=Parameterization.STANDARD,
        core_parameters={"p", "r"},
        optional_parameters=set(),
        parameter_descriptions={
            "p": "Success probability parameter (Beta distribution)",
            "r": "Dispersion parameter (LogNormal distribution)",
        },
    ),
    Parameterization.LINKED: ParameterizationMapping(
        parameterization=Parameterization.LINKED,
        core_parameters={"p", "mu"},
        optional_parameters=set(),
        parameter_descriptions={
            "p": "Success probability parameter (Beta distribution)",
            "mu": "Mean parameter (LogNormal distribution)",
        },
    ),
    Parameterization.ODDS_RATIO: ParameterizationMapping(
        parameterization=Parameterization.ODDS_RATIO,
        core_parameters={"phi", "mu"},
        optional_parameters=set(),
        parameter_descriptions={
            "phi": "Odds ratio parameter (BetaPrime distribution)",
            "mu": "Mean parameter (LogNormal distribution)",
        },
    ),
    # New names (preferred)
    Parameterization.CANONICAL: ParameterizationMapping(
        parameterization=Parameterization.CANONICAL,
        core_parameters={"p", "r"},
        optional_parameters=set(),
        parameter_descriptions={
            "p": "Success probability parameter (Beta distribution)",
            "r": "Dispersion parameter (LogNormal distribution)",
        },
    ),
    Parameterization.MEAN_PROB: ParameterizationMapping(
        parameterization=Parameterization.MEAN_PROB,
        core_parameters={"p", "mu"},
        optional_parameters=set(),
        parameter_descriptions={
            "p": "Success probability parameter (Beta distribution)",
            "mu": "Mean parameter (LogNormal distribution)",
        },
    ),
    Parameterization.MEAN_ODDS: ParameterizationMapping(
        parameterization=Parameterization.MEAN_ODDS,
        core_parameters={"phi", "mu"},
        optional_parameters=set(),
        parameter_descriptions={
            "phi": "Odds ratio parameter (BetaPrime distribution)",
            "mu": "Mean parameter (LogNormal distribution)",
        },
    ),
    Parameterization.MEAN_DISP: ParameterizationMapping(
        parameterization=Parameterization.MEAN_DISP,
        core_parameters={"mu", "r"},
        # p and phi are derived deterministics (NOT sampled), but they are
        # materialised on the posterior so downstream consumers find them;
        # listing them as optional keeps ModelConfig spec validation happy.
        optional_parameters={"p", "phi"},
        parameter_descriptions={
            "mu": "Mean parameter (LogNormal distribution)",
            "r": "Dispersion parameter (LogNormal distribution)",
            "p": "Success probability (derived: p = mu / (mu + r))",
            "phi": "Odds ratio (derived: phi = r / mu)",
        },
    ),
    # ------------------------------------------------------------------
    # LNM family: three parameterizations of the totals NB submodel.
    # Each maps the same compositional path (y_alr, multinomial,
    # decoder/encoder) to a different choice of which scalar globals
    # are sampled directly. Mirrors the DM-family
    # canonical / mean_prob / mean_odds pattern.
    # ------------------------------------------------------------------
    Parameterization.LOGISTIC_NORMAL_CANONICAL: ParameterizationMapping(
        parameterization=Parameterization.LOGISTIC_NORMAL_CANONICAL,
        core_parameters={"r_T", "p"},
        optional_parameters={"d_lnm", "y_alr", "z"},
        parameter_descriptions={
            "r_T": (
                "NB dispersion for total UMI counts per cell "
                "(LogNormal / PositiveNormal). Descriptive alias: "
                "``total_dispersion``. Override the prior via "
                "``priors={'r_T': (mu_log, sigma_log)}`` (or "
                "equivalently ``priors={'total_dispersion': "
                "(mu_log, sigma_log)}``) in ``scribe.fit``; an "
                "explicit value short-circuits the auto-defaults "
                "(MoM inversion when no capture anchor; "
                "biology-informed LogNormal(log 50, 1.5) when the "
                "capture anchor is active). Distinct from the "
                "DM-family gene-level ``r`` parameter (descriptive "
                "alias ``dispersion``), which has shape ``(n_genes,)``."
            ),
            "p": (
                "NB success probability for total counts "
                "(Beta / SigmoidNormal). Note: under the capture "
                "anchor with the canonical LNM parameterization, ``p`` "
                "becomes aliased with ``p_capture`` and may drift to "
                "the upper boundary; switch to ``mean_odds`` to "
                "eliminate the aliasing."
            ),
            "y_alr": (
                "ALR coordinates from VAE decoder (G-1, reference gene "
                "determined by alr_reference_idx)"
            ),
            "d_lnm": (
                "Optional per-coordinate ALR variance scale (learned d_mode)"
            ),
            "z": "VAE latent code",
        },
    ),
    Parameterization.LOGISTIC_NORMAL_MEAN_PROB: ParameterizationMapping(
        parameterization=Parameterization.LOGISTIC_NORMAL_MEAN_PROB,
        core_parameters={"mu_T", "p"},
        optional_parameters={"d_lnm", "y_alr", "z", "r_T"},
        parameter_descriptions={
            "mu_T": (
                "Population-level expected library size (LogNormal / "
                "PositiveNormal). A scalar — one value per dataset — "
                "describing the average ``u_T^(c)`` before per-cell "
                "capture modulation. Descriptive alias: "
                "``total_mean``. ``r_T`` is derived as "
                "``mu_T * (1 - p) / p``."
            ),
            "p": (
                "NB success probability for total counts "
                "(Beta / SigmoidNormal). Same caveat as the canonical "
                "variant: aliased with ``p_capture`` under the "
                "anchor; ``mean_odds`` eliminates the aliasing."
            ),
            "r_T": (
                "DERIVED: dispersion ``r_T = mu_T * (1 - p) / p``."
            ),
            "y_alr": (
                "ALR coordinates from VAE decoder (G-1, reference gene "
                "determined by alr_reference_idx)"
            ),
            "d_lnm": (
                "Optional per-coordinate ALR variance scale (learned d_mode)"
            ),
            "z": "VAE latent code",
        },
    ),
    Parameterization.LOGISTIC_NORMAL_MEAN_ODDS: ParameterizationMapping(
        parameterization=Parameterization.LOGISTIC_NORMAL_MEAN_ODDS,
        core_parameters={"mu_T", "phi_T"},
        optional_parameters={"d_lnm", "y_alr", "z", "r_T", "p"},
        parameter_descriptions={
            "mu_T": (
                "Population-level expected library size "
                "(LogNormal / PositiveNormal). Same as in mean_prob; "
                "directly identified by the empirical mean of ``u_T``. "
                "Descriptive alias: ``total_mean``."
            ),
            "phi_T": (
                "Totals-NB odds ratio ``phi_T = (1 - p) / p`` "
                "(BetaPrime / PositiveNormal). Identified by the "
                "per-cell variance of ``u_T`` around its predicted "
                "mean. Under the capture anchor, both ``mu_T`` and "
                "``phi_T`` retain independent identifying signal in "
                "the data — the ``(p, p_capture)`` aliasing of the "
                "other two variants is gone. Descriptive alias: "
                "``total_odds_ratio``."
            ),
            "r_T": (
                "DERIVED: dispersion ``r_T = mu_T * phi_T``."
            ),
            "p": (
                "DERIVED: success probability ``p = 1 / (1 + phi_T)``. "
                "Because ``p`` is derived rather than sampled, it "
                "cannot drift to the boundary as it does in the "
                "canonical / mean_prob variants under the capture "
                "anchor."
            ),
            "y_alr": (
                "ALR coordinates from VAE decoder (G-1, reference gene "
                "determined by alr_reference_idx)"
            ),
            "d_lnm": (
                "Optional per-coordinate ALR variance scale (learned d_mode)"
            ),
            "z": "VAE latent code",
        },
    ),
    # ------------------------------------------------------------------
    # Two-state promoter (Poisson-Beta compound) — single variant.
    # Samples ``mu`` (per-gene mean) as the only core parameter; the
    # additional per-gene parameters ``burst_size`` and ``k_off`` come
    # in via ``MODEL_EXTRA_PARAMS["twostate"]`` / ``["twostatevcp"]``.
    # The derived (alpha, beta, rate, effective_burst_size) quantities
    # are emitted at gene rank via ``numpyro.deterministic`` inside
    # the likelihood for posterior inspection.
    # ------------------------------------------------------------------
    Parameterization.TWO_STATE_NATURAL: ParameterizationMapping(
        parameterization=Parameterization.TWO_STATE_NATURAL,
        core_parameters={"mu"},
        # ``p_capture`` is deliberately NOT in optional_parameters here;
        # ``get_active_parameters`` adds it only for the ``twostatevcp``
        # variant or when ``uses_variable_capture=True``, mirroring how
        # the standard NB family conditionally adds it.
        optional_parameters={
            # Per-gene extras
            "burst_size",
            "k_off",
            # Derived per-gene quantities (deterministic sites)
            "alpha",
            "beta",
            "k_on",
            "r_hat",
            "effective_burst_size",
            "alpha_floor_active",
            "beta_floor_active",
        },
        parameter_descriptions={
            "mu": (
                "Per-gene mean expression "
                "(LogNormal / PositiveNormal). Descriptive alias: "
                "``mean_expression``."
            ),
            "burst_size": (
                "Per-gene NB-limit mean burst size "
                "(b = r_hat / k_off in the NB limit). "
                "PositiveNormal / SoftplusNormal."
            ),
            "k_off": (
                "Per-gene OFF rate (non-dimensionalised by mRNA decay). "
                "Large values favour the NB regime; small values "
                "produce bursty / bimodal counts. "
                "PositiveNormal / SoftplusNormal."
            ),
            "alpha": "DERIVED: Beta first shape = mu / burst_size.",
            "beta": "DERIVED: Beta second shape = k_off.",
            "k_on": "DERIVED: ON rate = mu / burst_size (= alpha).",
            "r_hat": (
                "DERIVED: Poisson rate scale = mu + burst_size · k_off "
                "(mean-preserving by construction)."
            ),
            "effective_burst_size": (
                "DERIVED: burst size implied by the floored alpha "
                "(= mu / alpha). Equals burst_size when no floor "
                "activates."
            ),
            "alpha_floor_active": (
                "DERIVED: boolean indicator per gene; True when the "
                "alpha = mu / burst_size value hits the numerical "
                "floor in the likelihood. Genes with this flag set "
                "should be interpreted with care."
            ),
            "beta_floor_active": (
                "DERIVED: boolean indicator per gene; True when k_off "
                "hits the numerical floor in the likelihood."
            ),
        },
    ),
    # ------------------------------------------------------------------
    # Two-state promoter — RELATIVE-switching variant.
    # Samples ``mu`` as the only core parameter; the extras are
    # ``burst_size`` and ``switching_ratio = k_off / k_on`` instead of
    # absolute ``k_off``.  Mean-preserving and analogous to NBDM's
    # mean_prob/mean_odds: the regime axis is orthogonal to gene
    # magnitude, which helps mean-field VI.
    # ------------------------------------------------------------------
    Parameterization.TWO_STATE_RATIO: ParameterizationMapping(
        parameterization=Parameterization.TWO_STATE_RATIO,
        core_parameters={"mu"},
        optional_parameters={
            "burst_size",
            "switching_ratio",
            # Derived per-gene quantities (deterministic sites)
            "alpha",
            "beta",
            "k_on",
            "k_off",
            "r_hat",
            "effective_burst_size",
            "alpha_floor_active",
            "beta_floor_active",
        },
        parameter_descriptions={
            "mu": (
                "Per-gene mean expression. "
                "PositiveNormal / SoftplusNormal."
            ),
            "burst_size": (
                "Per-gene NB-limit mean burst size. Same role as in "
                "the natural parameterization."
            ),
            "switching_ratio": (
                "Per-gene regime variable s = k_off / k_on. "
                "Large s → NB regime; small s → bursty / Poisson-like. "
                "PositiveNormal / SoftplusNormal."
            ),
            "alpha": "DERIVED: Beta first shape = mu / burst_size (= k_on).",
            "beta": (
                "DERIVED: Beta second shape = k_off = switching_ratio · k_on."
            ),
            "k_on": "DERIVED: ON rate = mu / burst_size.",
            "k_off": "DERIVED: OFF rate = switching_ratio · k_on.",
            "r_hat": (
                "DERIVED: Poisson rate scale = mu · (1 + switching_ratio) "
                "(mean-preserving)."
            ),
            "effective_burst_size": (
                "DERIVED: burst size implied by the floored alpha."
            ),
            "alpha_floor_active": (
                "DERIVED: boolean indicator; True when alpha hits the "
                "numerical floor in the likelihood."
            ),
            "beta_floor_active": (
                "DERIVED: boolean indicator; True when beta = k_off hits "
                "the numerical floor in the likelihood."
            ),
        },
    ),
    # ------------------------------------------------------------------
    # Two-state promoter — MEAN-FANO variant.
    # Samples ``mu`` as the only core parameter; the extras are
    # ``excess_fano`` (Var/Mean − 1) and ``concentration`` (α + β).
    # Mean- and Fano-preserving by construction.
    # ------------------------------------------------------------------
    Parameterization.TWO_STATE_MEAN_FANO: ParameterizationMapping(
        parameterization=Parameterization.TWO_STATE_MEAN_FANO,
        core_parameters={"mu"},
        optional_parameters={
            "excess_fano",
            "concentration",
            # Derived per-gene quantities (deterministic sites)
            "alpha",
            "beta",
            "k_on",
            "k_off",
            "burst_size",
            "r_hat",
            "effective_burst_size",
            "alpha_floor_active",
            "beta_floor_active",
        },
        parameter_descriptions={
            "mu": (
                "Per-gene mean expression. "
                "PositiveNormal / SoftplusNormal."
            ),
            "excess_fano": (
                "Per-gene excess Fano factor: Var/Mean − 1. "
                "Directly bounds posterior-predictive variance. "
                "PositiveNormal / SoftplusNormal."
            ),
            "concentration": (
                "Per-gene Beta concentration κ = α + β. Large κ "
                "approaches the NB shape (peaked latent p); small κ "
                "admits a U-shaped Beta and bursty / bimodal counts. "
                "PositiveNormal / SoftplusNormal."
            ),
            "alpha": (
                "DERIVED: Beta first shape = κ · μ / "
                "(μ + excess_fano · (κ + 1)). Equals k_on."
            ),
            "beta": (
                "DERIVED: Beta second shape = κ · excess_fano · "
                "(κ + 1) / (μ + excess_fano · (κ + 1)). Equals k_off."
            ),
            "k_on": "DERIVED: ON rate; equals alpha.",
            "k_off": "DERIVED: OFF rate; equals beta.",
            "burst_size": (
                "DERIVED: NB-limit burst size = mu / alpha. Equals "
                "excess_fano in the NB limit (large concentration)."
            ),
            "r_hat": (
                "DERIVED: Poisson rate scale = mu + excess_fano · "
                "(concentration + 1) (mean-preserving)."
            ),
            "effective_burst_size": (
                "DERIVED: burst size implied by the floored alpha."
            ),
            "alpha_floor_active": (
                "DERIVED: boolean indicator; True when alpha hits "
                "the numerical floor in the likelihood."
            ),
            "beta_floor_active": (
                "DERIVED: boolean indicator; True when beta hits "
                "the numerical floor in the likelihood."
            ),
        },
    ),
    # ------------------------------------------------------------------
    # Two-state promoter — MOMENT-DELTA variant.
    # Samples ``mu`` as the only core parameter; the extras are
    # ``excess_fano`` and ``inv_concentration = 1 / (kappa + 1) in
    # (0, 1)``.  Bounded shape coordinate; sigmoid-normal guide.
    # Same moment guarantees as TWO_STATE_MEAN_FANO.
    # ------------------------------------------------------------------
    Parameterization.TWO_STATE_MOMENT_DELTA: ParameterizationMapping(
        parameterization=Parameterization.TWO_STATE_MOMENT_DELTA,
        core_parameters={"mu"},
        optional_parameters={
            "excess_fano",
            "inv_concentration",
            # Derived per-gene quantities (deterministic sites)
            "alpha",
            "beta",
            "k_on",
            "k_off",
            "burst_size",
            "concentration",
            "r_hat",
            "effective_burst_size",
            "alpha_floor_active",
            "beta_floor_active",
        },
        parameter_descriptions={
            "mu": (
                "Per-gene mean expression. "
                "PositiveNormal / SoftplusNormal."
            ),
            "excess_fano": (
                "Per-gene excess Fano factor: Var/Mean - 1. "
                "Directly bounds posterior-predictive variance. "
                "PositiveNormal / SoftplusNormal."
            ),
            "inv_concentration": (
                "Per-gene shape coordinate delta = 1/(kappa + 1) "
                "in (0, 1). delta -> 0 is the NB limit, delta near "
                "1 is the extreme bursty regime. SigmoidNormal."
            ),
            "alpha": (
                "DERIVED: Beta first shape = mu * (1 - delta) / "
                "(mu * delta + excess_fano). Equals k_on."
            ),
            "beta": (
                "DERIVED: Beta second shape = excess_fano * (1 - "
                "delta) / (delta * (mu * delta + excess_fano)). "
                "Equals k_off."
            ),
            "k_on": "DERIVED: ON rate; equals alpha.",
            "k_off": "DERIVED: OFF rate; equals beta.",
            "burst_size": (
                "DERIVED: NB-limit burst size = mu / alpha. Equals "
                "excess_fano in the NB limit (delta -> 0)."
            ),
            "concentration": (
                "DERIVED: Beta concentration kappa = (1 - delta) / "
                "delta."
            ),
            "r_hat": (
                "DERIVED: Poisson rate scale = (mu * delta + "
                "excess_fano) / delta (mean-preserving)."
            ),
            "effective_burst_size": (
                "DERIVED: burst size implied by the floored alpha."
            ),
            "alpha_floor_active": (
                "DERIVED: boolean indicator; True when alpha hits "
                "the numerical floor in the likelihood."
            ),
            "beta_floor_active": (
                "DERIVED: boolean indicator; True when beta hits "
                "the numerical floor in the likelihood."
            ),
        },
    ),
    # ------------------------------------------------------------------
    # PLN (Poisson-LogNormal) — single variant, no totals submodel
    # ------------------------------------------------------------------
    Parameterization.COUNT_LOGNORMAL: ParameterizationMapping(
        parameterization=Parameterization.COUNT_LOGNORMAL,
        core_parameters=set(),
        optional_parameters={"d_pln", "y_log_rate", "z"},
        parameter_descriptions={
            "y_log_rate": (
                "Per-gene log Poisson rates from VAE decoder "
                "(G-dimensional, identity transform). Exponentiated "
                "inside the likelihood to obtain Poisson rates."
            ),
            "d_pln": (
                "Optional per-gene residual variance scale in "
                "log-rate space (learned d_mode, G-dimensional)"
            ),
            "z": "VAE latent code",
        },
    ),
}

# ==============================================================================
# Model-Specific Parameter Mappings
# ==============================================================================

# Parameters that are always optional (not core to any parameterization)
OPTIONAL_MODEL_PARAMETERS = {
    "gate": "Zero-inflation gate parameter (Beta distribution)",
    "p_capture": "Capture probability parameter (Beta distribution)",
    "phi_capture": "Capture phi parameter (BetaPrime distribution)",
    "mixing": "Mixture weights parameter (Dirichlet distribution)",
}

# Parameters that are required for specific model types
MODEL_TYPE_REQUIREMENTS = {
    "zinb": {"gate"},  # Zero-inflated models require gate parameter
    "vcp": {"p_capture"},  # Variable capture models require p_capture
    "vcp_odds_ratio": {
        "phi_capture"
    },  # VCP with odds_ratio requires phi_capture
    "mixture": {"mixing"},  # Mixture models require mixing parameter
}

# ==============================================================================
# Parameter Mapping Utilities
# ==============================================================================


def get_parameterization_mapping(
    parameterization: Parameterization,
) -> ParameterizationMapping:
    """Get the parameter mapping for a specific parameterization."""
    return PARAMETERIZATION_MAPPINGS[parameterization]


# ------------------------------------------------------------------------------


def get_active_parameters(
    parameterization: Parameterization,
    model_type: str,
    is_mixture: bool = False,
    is_zero_inflated: bool = False,
    uses_variable_capture: bool = False,
    uses_biology_informed_capture: bool = False,
    hierarchical_mu: bool = False,
    hierarchical_p: bool = False,
    hierarchical_gate: bool = False,
) -> Set[str]:
    """
    Get the set of active parameters for a given configuration.

    Parameters
    ----------
    parameterization : Parameterization
        The parameterization type
    model_type : str
        The model type (e.g., 'nbdm', 'zinb', 'nbvcp', 'zinbvcp')
    is_mixture : bool
        Whether this is a mixture model
    is_zero_inflated : bool
        Whether this is a zero-inflated model
    uses_variable_capture : bool
        Whether this model uses variable capture (DM-family ``p_capture``
        or ``phi_capture`` site).
    uses_biology_informed_capture : bool
        Whether the biology-informed capture path is active (PLN/NBLN/LNMVCP
        with the ``eta_capture`` truncated-normal log-offset latent).
        When True, ``"eta_capture"`` is added to the active set so
        downstream code that introspects ``model_config.active_parameters``
        sees the capture site.
    hierarchical_mu : bool
        Whether hierarchical mu/r prior across components is enabled
    hierarchical_p : bool
        Whether hierarchical p/phi prior is enabled (True when
        ``p_prior != "none"`` in ModelConfig).
    hierarchical_gate : bool
        Whether hierarchical gate prior is enabled (True when
        ``gate_prior != "none"`` in ModelConfig).

    Returns
    -------
    Set[str]
        Set of active parameter names
    """
    # Start with parameterization-specific parameters
    mapping = get_parameterization_mapping(parameterization)
    active_params = mapping.get_all_parameters().copy()

    # PLN/NBLN share ``Parameterization.COUNT_LOGNORMAL`` (Path B
    # design — see ``Parameterization`` docstring), but the per-gene
    # diagonal residual variance scale uses a *model-specific* site
    # name to avoid collision when both likelihoods coexist:
    # ``d_pln`` (PLN) vs ``d_nbln`` (NBLN).  The likelihood class sets
    # ``d_param_name`` accordingly; mirror that swap here so
    # ``model_config.active_parameters`` matches what the likelihood
    # actually samples.
    if (
        parameterization == Parameterization.COUNT_LOGNORMAL
        and model_type == "nbln"
        and "d_pln" in active_params
    ):
        active_params.discard("d_pln")
        active_params.add("d_nbln")

    # Add hierarchical mu hyperparameters when flag is set
    if hierarchical_mu:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
            Parameterization.MEAN_PROB,
            Parameterization.LINKED,
            # mean_disp also carries the expression hierarchy on mu (its
            # gene mean), so its hyperparameters are log_mu_* like the other
            # mean-parameterized families.
            Parameterization.MEAN_DISP,
        ):
            active_params.update({"log_mu_loc", "log_mu_scale"})
        else:
            active_params.update({"log_r_loc", "log_r_scale"})

    # Add hierarchical hyperparameters when flags are set
    if hierarchical_p:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
            active_params.update({"log_phi_loc", "log_phi_scale"})
        else:
            active_params.update({"logit_p_loc", "logit_p_scale"})

    # Add model-specific parameters
    if is_zero_inflated:
        active_params.add("gate")
        if hierarchical_gate:
            active_params.update({"logit_gate_loc", "logit_gate_scale"})

    if uses_variable_capture:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
            active_params.add("phi_capture")
        else:
            active_params.add("p_capture")

    # Biology-informed capture (PLN/NBLN/LNMVCP eta_c truncated-normal
    # log-offset).  Distinct from ``uses_variable_capture`` (which is
    # the DM-family ``p_capture`` Beta site): biology-informed capture
    # uses a Gaussian-on-log-rate ``eta_capture`` latent.  Both can
    # legitimately coexist conceptually, but in practice
    # ``uses_biology_informed_capture`` is True for PLN/NBLN/LNMVCP
    # with a capture prior and ``uses_variable_capture`` is True for
    # the DM-family ``vcp`` aliases.
    if uses_biology_informed_capture:
        active_params.add("eta_capture")

    if is_mixture:
        active_params.add("mixing")

    # NBLN: gene dispersion ``r_g`` is added on top of the
    # POISSON_LOGNORMAL parameterization. Registered in
    # ``MODEL_EXTRA_PARAMS["nbln"] = ["r"]`` and built by
    # ``build_r_spec`` in the registry.
    if model_type == "nbln":
        active_params.add("r")

    # Two-state promoter: the gene-level extras depend on the
    # parameterization.
    #   - TWO_STATE_NATURAL:      (burst_size, k_off)
    #   - TWO_STATE_RATIO:        (burst_size, switching_ratio)
    #   - TWO_STATE_MEAN_FANO:    (excess_fano, concentration)
    #   - TWO_STATE_MOMENT_DELTA: (excess_fano, inv_concentration)
    # ``p_capture`` is added for the VCP variant.  Derived per-gene
    # deterministics (alpha, beta, r_hat, effective_burst_size, floor
    # indicators) are exposed by the likelihood and stay in the
    # mapping's optional set.
    if model_type in ("twostate", "twostatevcp"):
        if parameterization == Parameterization.TWO_STATE_MOMENT_DELTA:
            active_params.update({"excess_fano", "inv_concentration"})
        elif parameterization == Parameterization.TWO_STATE_MEAN_FANO:
            active_params.update({"excess_fano", "concentration"})
        elif parameterization == Parameterization.TWO_STATE_RATIO:
            active_params.update({"burst_size", "switching_ratio"})
        else:
            active_params.update({"burst_size", "k_off"})
        if model_type == "twostatevcp" or uses_variable_capture:
            active_params.add("p_capture")

    return active_params


# ------------------------------------------------------------------------------


def get_required_parameters(
    parameterization: Parameterization,
    model_type: str,
    is_mixture: bool = False,
    is_zero_inflated: bool = False,
    uses_variable_capture: bool = False,
) -> Set[str]:
    """
    Get the set of required parameters for a given configuration.

    Parameters
    ----------
    parameterization : Parameterization
        The parameterization type
    model_type : str
        The model type (e.g., 'nbdm', 'zinb', 'nbvcp', 'zinbvcp')
    is_mixture : bool
        Whether this is a mixture model
    is_zero_inflated : bool
        Whether this is a zero-inflated model
    uses_variable_capture : bool
        Whether this model uses variable capture

    Returns
    -------
    Set[str]
        Set of required parameter names
    """
    # Start with parameterization-specific required parameters
    mapping = get_parameterization_mapping(parameterization)
    required_params = mapping.core_parameters.copy()

    # Add model-specific required parameters
    if is_zero_inflated:
        required_params.add("gate")

    if uses_variable_capture:
        if parameterization == Parameterization.ODDS_RATIO:
            required_params.add("phi_capture")
        else:
            required_params.add("p_capture")

    if is_mixture:
        required_params.add("mixing")

    return required_params


# ------------------------------------------------------------------------------


def get_parameter_description(
    param_name: str, parameterization: Parameterization
) -> str:
    """
    Get the description for a parameter in a specific parameterization context.

    Parameters
    ----------
    param_name : str
        The parameter name
    parameterization : Parameterization
        The parameterization type

    Returns
    -------
    str
        Description of the parameter
    """
    # Check parameterization-specific descriptions first
    mapping = get_parameterization_mapping(parameterization)
    if param_name in mapping.parameter_descriptions:
        return mapping.parameter_descriptions[param_name]

    # Check optional model parameters
    if param_name in OPTIONAL_MODEL_PARAMETERS:
        return OPTIONAL_MODEL_PARAMETERS[param_name]

    # Default description
    return f"Parameter: {param_name}"


# ------------------------------------------------------------------------------


def validate_parameter_consistency(
    parameterization: Parameterization,
    model_type: str,
    provided_params: Set[str],
    is_mixture: bool = False,
    is_zero_inflated: bool = False,
    uses_variable_capture: bool = False,
) -> List[str]:
    """
    Validate that provided parameters are consistent with the configuration.

    Parameters
    ----------
    parameterization : Parameterization
        The parameterization type
    model_type : str
        The model type
    provided_params : Set[str]
        Set of provided parameter names
    is_mixture : bool
        Whether this is a mixture model
    is_zero_inflated : bool
        Whether this is a zero-inflated model
    uses_variable_capture : bool
        Whether this model uses variable capture

    Returns
    -------
    List[str]
        List of validation error messages (empty if valid)
    """
    errors = []

    # Get expected parameters
    expected_params = get_active_parameters(
        parameterization,
        model_type,
        is_mixture,
        is_zero_inflated,
        uses_variable_capture,
    )
    required_params = get_required_parameters(
        parameterization,
        model_type,
        is_mixture,
        is_zero_inflated,
        uses_variable_capture,
    )

    # Check for missing required parameters
    missing_required = required_params - provided_params
    if missing_required:
        errors.append(
            f"Missing required parameters for {parameterization.value} parameterization: "
            f"{', '.join(sorted(missing_required))}"
        )

    # Check for unsupported parameters
    unsupported = provided_params - expected_params
    if unsupported:
        errors.append(
            f"Unsupported parameters for {parameterization.value} parameterization: "
            f"{', '.join(sorted(unsupported))}"
        )

    return errors


# ==============================================================================
# Parameter Mapping Registry
# ==============================================================================


def get_all_parameterization_mappings() -> (
    Dict[Parameterization, ParameterizationMapping]
):
    """Get all parameterization mappings."""
    return PARAMETERIZATION_MAPPINGS.copy()


def get_parameterization_summary() -> Dict[str, Dict[str, any]]:
    """
    Get a summary of all parameterizations and their supported parameters.

    Returns
    -------
    Dict[str, Dict[str, any]]
        Summary dictionary with parameterization details
    """
    summary = {}

    for param_type, mapping in PARAMETERIZATION_MAPPINGS.items():
        summary[param_type.value] = {
            "core_parameters": sorted(mapping.core_parameters),
            "optional_parameters": sorted(mapping.optional_parameters),
            "all_parameters": sorted(mapping.get_all_parameters()),
            "descriptions": mapping.parameter_descriptions,
        }

    return summary


# ==============================================================================
# Single Source-of-Truth Parameter Name Registry
# ==============================================================================
#
# ONE place mapping each internal sample-site name to its single, canonical,
# parameterization-INDEPENDENT user-facing name. EVERY other name map in this
# module is DERIVED from this registry at import time -- there is no second
# hand-maintained table to drift out of sync.
#
# Rules:
#   * ``site`` <-> ``canonical`` is a strict bijection (enforced by tests).
#   * ``canonical`` is the ONLY user-facing name; there are NO deprecated
#     aliases. (The internal grouping ``TARGET_NAMES`` such as ``"expression"``
#     are a separate, internal routing layer -- not user vocabulary.)
#   * ``hierarchy_target`` records which grouping ``TARGET_NAME`` a site's
#     dataset/factor hierarchy routes to (consumed by the unified ``priors``
#     normalizer); ``None`` for sites that carry no dataset-level hierarchy.


@dataclass(frozen=True)
class ParamName:
    """One entry in the canonical parameter-name registry."""

    site: str
    canonical: str
    hierarchy_target: Optional[str] = None


PARAM_REGISTRY: Tuple[ParamName, ...] = (
    # --- NB-family core parameters ---
    ParamName("mu", "mean_expression", "expression"),
    ParamName("r", "dispersion", "dispersion"),
    ParamName("p", "probability", "prob"),
    ParamName("phi", "odds_ratio", "prob"),
    ParamName("gate", "zero_inflation", "zero_inflation"),
    # --- Totals submodel (LNM family) ---
    ParamName("r_T", "total_dispersion"),
    ParamName("mu_T", "total_mean"),
    ParamName("phi_T", "total_odds_ratio"),
    # --- Capture parameters ---
    ParamName("p_capture", "capture_probability"),
    ParamName("eta_capture", "capture_efficiency"),
    ParamName("phi_capture", "capture_odds_ratio"),
    ParamName("mu_eta", "capture_scaling"),
    # --- BNB overdispersion (concentration kappa); BNB-only, never NB r ---
    ParamName("bnb_concentration", "overdispersion", "overdispersion"),
    # --- Low-rank loadings matrix W (PLN/NBLN/LNM-family) ---
    ParamName("W", "loadings"),
    # --- Per-leaf module weights s (NBLN hierarchical correlation, Rung 1.5) ---
    # The realized ``s`` is a deterministic function of per-factor NCP globals,
    # not a directly-sampled site; this registry entry exists so the unified
    # ``priors`` dict can route ``priors={"module_weight": {level: family}}``
    # into ``Factor.priors["module_weight"]`` via HIERARCHY_TARGET_BY_SITE.
    ParamName("s", "module_weight", "module_weight"),
    # --- LNM compositional parameters ---
    ParamName("y_alr", "alr_coordinates"),
    ParamName("d_lnm", "alr_residual_scale"),
    ParamName("z", "latent_embedding"),
    # --- PLN log-rate parameters ---
    ParamName("y_log_rate", "log_expression_rate"),
    ParamName("d_pln", "expression_residual_scale"),
    # --- Mixture weights ---
    ParamName("mixing_weights", "mixing_weights"),
    # --- Two-state promoter (Poisson-Beta) sampled + derived sites.
    #     Already domain-standard names; canonical == site (identity). ---
    ParamName("burst_size", "burst_size"),
    ParamName("k_off", "k_off"),
    ParamName("k_on", "k_on"),
    ParamName("switching_ratio", "switching_ratio"),
    ParamName("excess_fano", "excess_fano"),
    ParamName("concentration", "concentration"),
    ParamName("inv_concentration", "inv_concentration"),
    ParamName("alpha", "alpha"),
    ParamName("beta", "beta"),
    ParamName("r_hat", "r_hat"),
    ParamName("effective_burst_size", "effective_burst_size"),
    ParamName("alpha_floor_active", "alpha_floor_active"),
    ParamName("beta_floor_active", "beta_floor_active"),
    # --- Two-state regime hierarchy (abstract, hierarchy-only) ---
    #     ``regime`` is NOT a sampled site: it is the parameterization-
    #     independent name for the two-state *bursting regime* hierarchy. The
    #     concrete coordinate it attaches to differs by parameterization
    #     (``k_off`` / ``switching_ratio`` / ``concentration`` /
    #     ``inv_concentration`` -- see ``TWOSTATE_REGIME_COORD``), optionally
    #     pinned via the structural ``regime_dataset_target`` kwarg. Carrying a
    #     single canonical key keeps ``priors={"regime": {...}}`` stable across
    #     all two-state parameterizations. It routes (hierarchy target
    #     ``"regime"``) to the internal ``regime_dataset_prior`` field; it has
    #     no gene-level or base-prior form.
    ParamName("regime", "regime", "regime"),
)

# Forward routing: internal site -> grouping TARGET_NAME (for the unified
# ``priors`` normalizer). Only sites that carry a dataset-level hierarchy.
HIERARCHY_TARGET_BY_SITE: Dict[str, str] = {
    e.site: e.hierarchy_target
    for e in PARAM_REGISTRY
    if e.hierarchy_target is not None
}

# ==============================================================================
# Derived name maps (ALL generated from PARAM_REGISTRY -- do not hand-edit)
# ==============================================================================

# internal site -> canonical user-facing name. Used by
# results.get_map / get_distributions / get_posterior_samples
# (descriptive_names=True). Hierarchical hyperprior keys (logit_p_loc, ...)
# are not listed; they pass through unchanged (the _loc/_scale suffixes are
# correct Normal parameter names in unconstrained space).
DESCRIPTIVE_NAMES: Dict[str, str] = {
    e.site: e.canonical for e in PARAM_REGISTRY
}

# Inverse mapping: canonical descriptive name -> internal site.
_DESCRIPTIVE_TO_INTERNAL: Dict[str, str] = {
    e.canonical: e.site for e in PARAM_REGISTRY
}


# ==============================================================================
# Prior Key Resolution (canonical -> internal)
# ==============================================================================

# Resolution table for the ``priors`` dict. STRICT: only the canonical names
# from PARAM_REGISTRY are accepted as descriptive keys (internal site names
# also pass through unchanged in ``normalize_prior_keys``). There are NO
# deprecated aliases. The ``loadings`` -> ``W`` entry carries a dict-shaped
# strategy spec (not a tuple); the fit() entry point routes it to the W-prior
# plumbing before the priors dict reaches downstream stages. Hierarchical
# hyperprior override keys (logit_p_loc, ...) are left as-is.
PRIOR_KEY_ALIASES: Dict[str, str] = dict(_DESCRIPTIVE_TO_INTERNAL)

# Inverse: internal -> canonical (for documentation / YAML comments).
_INTERNAL_TO_PRIOR_ALIAS: Dict[str, str] = dict(DESCRIPTIVE_NAMES)


# ==============================================================================
# NBLN cascade freeze key aliases
# ==============================================================================

# Maps user-facing descriptive aliases for NBLN cascade-freeze parameter
# names to the internal short names used by the obs model's freeze
# logic.  The freeze internal names are short -- ``"r"``, ``"mu"``,
# ``"eta"`` -- and they intentionally differ from the per-site names
# (``"eta"`` is short for the ``eta_capture`` site; ``"mu"`` is the
# log-rate prior mean, not the ``mu_T`` totals parameter).  This map
# bridges the two naming conventions so that
#
#     informative_priors_freeze=("dispersion", "capture_efficiency")
#
# resolves identically to the historic
#
#     informative_priors_freeze=("r", "eta").
#
# Resolved by ``normalize_freeze_keys`` in the API run-inference stage
# before the tuple is passed to the obs model.
FREEZE_KEY_ALIASES: Dict[str, str] = {
    "dispersion": "r",
    # Both "mean_expression" and "expression" are accepted aliases for the
    # log-rate prior mean ``mu`` -- mirroring the ParamName for ``mu`` used by
    # PRIOR_KEY_ALIASES (``ParamName("mu", "mean_expression", "expression")``).
    "mean_expression": "mu",
    "expression": "mu",
    "capture_efficiency": "eta",
}


def normalize_freeze_keys(freeze):
    """Translate descriptive ``informative_priors_freeze`` aliases.

    Accepts any iterable of strings drawn from
    ``{"r", "mu", "eta"}`` and/or their descriptive aliases
    (``"dispersion"``, ``"expression"`` / ``"mean_expression"``,
    ``"capture_efficiency"``).  Returns a tuple of the internal short
    names.  Empty / ``None`` input passes through unchanged.

    Raises
    ------
    ValueError
        If both an alias and its internal target are present (e.g.
        ``("r", "dispersion")``) — ambiguous.

    Examples
    --------
    >>> normalize_freeze_keys(("r", "eta"))
    ('r', 'eta')
    >>> normalize_freeze_keys(("dispersion", "capture_efficiency"))
    ('r', 'eta')
    >>> normalize_freeze_keys(("r", "dispersion"))
    Traceback (most recent call last):
        ...
    ValueError: Duplicate freeze key: ...
    """
    if not freeze:
        return tuple(freeze) if freeze is not None else freeze
    resolved: List[str] = []
    seen: Set[str] = set()
    for k in freeze:
        internal = FREEZE_KEY_ALIASES.get(k, k)
        if internal in seen:
            raise ValueError(
                f"Duplicate freeze key: '{k}' resolves to internal "
                f"name '{internal}', which is already present in "
                f"informative_priors_freeze. Pass each parameter once, "
                f"using either the internal short name (e.g. 'r') or "
                f"its descriptive alias (e.g. 'dispersion'), not both."
            )
        seen.add(internal)
        resolved.append(internal)
    return tuple(resolved)


# ==============================================================================
# fit() Keyword Argument Rename Mapping
# ==============================================================================

# Maps old (math-notation) fit() kwarg names to new (descriptive) names.
# Used by ModelConfig.__setstate__ for pickle migration and as documentation.
FIT_KWARG_RENAMES: Dict[str, str] = {
    "mu_prior": "expression_prior",
    "mu_dataset_prior": "expression_dataset_prior",
    "mu_eta_prior": "capture_scaling_prior",
    "mu_mean_anchor": "expression_anchor",
    "mu_mean_anchor_sigma": "expression_anchor_sigma",
    "p_prior": "prob_prior",
    "p_dataset_prior": "prob_dataset_prior",
    "p_dataset_mode": "prob_dataset_mode",
    "gate_prior": "zero_inflation_prior",
    "gate_dataset_prior": "zero_inflation_dataset_prior",
}


# ==============================================================================
# Utility Functions
# ==============================================================================


def normalize_prior_keys(priors: Dict[str, Any]) -> Dict[str, Any]:
    """Translate descriptive prior dict keys to their internal equivalents.

    Accepts both internal keys (``"p"``, ``"eta_capture"``) and descriptive
    aliases (``"prob"``, ``"capture_efficiency"``).  Raises ``ValueError`` if
    both an alias and its internal target are present (ambiguous).

    Parameters
    ----------
    priors : dict
        Prior override dictionary as passed by the user.

    Returns
    -------
    dict
        Copy of *priors* with all alias keys replaced by internal names.

    Raises
    ------
    ValueError
        If a descriptive alias and its corresponding internal key are both
        present (e.g. ``{"p": ..., "prob": ...}``).
    """
    if not priors:
        return priors

    result: Dict[str, Any] = {}
    for key, value in priors.items():
        internal_key = PRIOR_KEY_ALIASES.get(key)
        if internal_key is not None:
            # Key is a descriptive alias -- resolve it
            if internal_key in priors:
                raise ValueError(
                    f"Ambiguous priors: both the descriptive alias "
                    f"'{key}' and the internal key '{internal_key}' are "
                    f"present. Use one or the other, not both."
                )
            result[internal_key] = value
        else:
            result[key] = value
    return result


def rename_dict_keys(
    d: Dict[str, Any],
    descriptive: bool = False,
) -> Dict[str, Any]:
    """Optionally rename internal parameter keys to descriptive names.

    Handles suffixed keys such as ``r_0`` (component-indexed) and
    ``r_dataset_0`` (dataset-indexed) by matching the base name before the
    first ``_`` digit boundary.

    Parameters
    ----------
    d : dict
        Dictionary with internal parameter name keys.
    descriptive : bool, default False
        If True, apply the DESCRIPTIVE_NAMES mapping.  If False, return
        *d* unchanged.

    Returns
    -------
    dict
        A new dictionary with renamed keys (if *descriptive* is True),
        or the original dictionary unchanged.
    """
    if not descriptive:
        return d

    renamed: Dict[str, Any] = {}
    for key, value in d.items():
        new_key = DESCRIPTIVE_NAMES.get(key)
        if new_key is not None:
            renamed[new_key] = value
        else:
            # Try to match base_name from suffixed keys like "r_0", "gate_1"
            renamed[_rename_suffixed_key(key)] = value
    return renamed


def _rename_suffixed_key(key: str) -> str:
    """Rename a suffixed key like ``r_0`` -> ``dispersion_0``.

    Tries progressively longer prefixes so that multi-part internal names
    like ``p_capture_0`` are handled correctly (``p_capture`` is in the
    mapping, not ``p``).
    """
    parts = key.split("_")
    # Try longest prefix first (e.g. "p_capture" before "p")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "_".join(parts[:i])
        suffix = "_".join(parts[i:])
        if prefix in DESCRIPTIVE_NAMES:
            return f"{DESCRIPTIVE_NAMES[prefix]}_{suffix}"
    return key


# ==============================================================================
# Parameter Shorthand Resolution
# ==============================================================================

# Recognised string shorthands for mixture_params, joint_params, dense_params.
# These resolve to concrete ``List[str]`` (or ``None``) based on the chosen
# parameterization and model type, so users never need to memorise which
# Greek-letter names belong to which parameterization.
PARAM_SHORTHANDS = frozenset({"all", "biological", "mean", "prob", "gate"})


def resolve_param_shorthand(
    value: "Union[str, List[str], None]",
    param_strategy: "Parameterization",
    base_model: str,
) -> "Optional[List[str]]":
    """Resolve a parameter shorthand to a concrete list of internal names.

    Accepts the semantic shorthands ``"all"``, ``"biological"``, ``"mean"``,
    ``"prob"``, and ``"gate"`` and expands them using the parameterization
    strategy's ``core_parameters`` and ``gene_param_name``.  Explicit
    ``List[str]`` values pass through with canonical descriptive names
    resolved (e.g. ``"mean_expression"`` -> ``"mu"``).

    Parameters
    ----------
    value : str, List[str], or None
        The user-provided value.  ``None`` passes through unchanged.
    param_strategy : Parameterization
        The active parameterization strategy instance (provides
        ``core_parameters`` and ``gene_param_name``).
    base_model : str
        The base model type (``"nbdm"``, ``"zinb"``, ``"nbvcp"``,
        ``"zinbvcp"``).  Needed to determine whether ``"gate"`` is valid.

    Returns
    -------
    Optional[List[str]]
        Resolved list of internal parameter names, or ``None`` if
        *value* was ``None``.

    Raises
    ------
    ValueError
        If a shorthand references an unavailable parameter (e.g.
        ``"gate"`` for a non-ZINB model) or if a list element is not a
        recognised internal or descriptive name.

    Examples
    --------
    >>> from scribe.models.parameterizations import PARAMETERIZATIONS
    >>> strat = PARAMETERIZATIONS["mean_odds"]
    >>> resolve_param_shorthand("biological", strat, "zinb")
    ['phi', 'mu']
    >>> resolve_param_shorthand("mean", strat, "zinb")
    ['mu']
    >>> resolve_param_shorthand("all", strat, "zinb")
    ['phi', 'mu', 'gate']
    >>> resolve_param_shorthand(["mean_expression", "odds_ratio"], strat, "zinb")
    ['mu', 'phi']
    """
    if value is None:
        return None

    # --- String shorthand resolution ---
    if isinstance(value, str):
        shorthand = value.lower()
        core = list(param_strategy.core_parameters)
        gene_param = param_strategy.gene_param_name
        is_zinb = base_model in ("zinb", "zinbvcp")

        # Derive the "prob" parameter: the core param that is NOT the
        # gene-level (mean/dispersion) parameter.
        prob_param = [p for p in core if p != gene_param]

        if shorthand == "all":
            # Build the explicit list: core params + gate for ZINB models.
            # This concrete list works uniformly for mixture_params (where
            # the factory checks membership), joint_params, and dense_params.
            all_params = list(core)
            if is_zinb:
                all_params.append("gate")
            return all_params

        if shorthand == "biological":
            return core

        if shorthand == "mean":
            return [gene_param]

        if shorthand == "prob":
            return prob_param

        if shorthand == "gate":
            if not is_zinb:
                raise ValueError(
                    f"Shorthand 'gate' is only valid for ZINB models "
                    f"(zinb, zinbvcp), but the current model is "
                    f"'{base_model}'."
                )
            return ["gate"]

        raise ValueError(
            f"Unknown parameter shorthand '{value}'. "
            f"Valid shorthands are: {sorted(PARAM_SHORTHANDS)}. "
            f"You can also pass an explicit list of parameter names."
        )

    # --- Explicit list: resolve canonical descriptive names ---
    # Accept the canonical descriptive names (``mean_expression``,
    # ``odds_ratio``, ...) and pass internal site names through unchanged.
    resolved: "List[str]" = []
    for name in value:
        internal = _DESCRIPTIVE_TO_INTERNAL.get(name)
        if internal is None:
            internal = PRIOR_KEY_ALIASES.get(name)
        resolved.append(internal if internal is not None else name)
    return resolved
