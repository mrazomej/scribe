"""Descriptors for the hierarchical-prior builder families.

The factory builds the same hierarchical-prior structure once per parameter
(``mu``/``r``, ``p``/``phi``, ``gate``, and the two-state regime coordinate), at
two levels (gene-level shrinkage across components, and dataset-level shrinkage
across datasets). The variants differ only along predictable axes — transform
family (sigmoid vs positive), site-name scheme, and the matched
Positive/Sigmoid spec-class pair — so a single descriptor can capture every axis
and let one generic core build all of them.

``HierParam`` is that descriptor: a frozen, declarative record of the exact
site/spec names and classes a given hierarchical parameter uses. The generic
NCP cores in :mod:`scribe.models.presets.factory` read these fields instead of
re-deriving them with per-parameter ``if`` ladders.

This module lives in ``builders/`` (the lowest layer) and is **self-contained**:
it imports only the spec classes from its sibling :mod:`parameter_specs` and the
two-state coordinate constants from :mod:`config.enums` (both leaf modules). It
MUST NOT import ``presets.factory`` — that would close a
``builders -> presets -> factory -> builders`` import cycle (the factory already
imports ``..builders``).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Type

from ..config.enums import (
    TWOSTATE_REGIME_COORD,
    TWOSTATE_SIGMOID_REGIME_COORDS,
)
from .parameter_specs import (
    DatasetHierarchicalPositiveNormalSpec,
    DatasetHierarchicalSigmoidNormalSpec,
    HierarchicalPositiveNormalSpec,
    HierarchicalSigmoidNormalSpec,
    HorseshoeDatasetPositiveNormalSpec,
    HorseshoeDatasetSigmoidNormalSpec,
    HorseshoeHierarchicalPositiveNormalSpec,
    HorseshoeHierarchicalSigmoidNormalSpec,
    NEGDatasetPositiveNormalSpec,
    NEGDatasetSigmoidNormalSpec,
    NEGHierarchicalPositiveNormalSpec,
    NEGHierarchicalSigmoidNormalSpec,
    ParamSpec,
)


# ------------------------------------------------------------------------------
# Target resolution (absorbed from factory so this module stays self-contained)
# ------------------------------------------------------------------------------


def expression_target_is_mu(param_key: str) -> bool:
    """Whether the *dataset-level* expression parameter is ``mu`` (not ``r``).

    The dataset-level expression hierarchy operates on whichever parameter
    carries the biological mean: ``mu`` for the NB mean parameterizations
    (``mean_prob`` / ``mean_odds``), for ``mean_disp`` (which samples the gene
    mean ``mu`` directly), and for every two-state parameterization (whose core
    gene parameter is ``mu``); it is ``r`` for the canonical NB parameterization.

    Verbatim copy of ``factory._expression_target_is_mu`` kept here so the
    descriptor module does not import the factory (avoiding an import cycle);
    ``test_hier_descriptors`` asserts the two agree.

    Note the GENE-level expression builders (``_hierarchicalize_mu`` /
    ``_horseshoe_mu`` / ``_neg_mu``) use a *narrower* rule that excludes the
    two-state family — :func:`_gene_expression_target_is_mu`.
    """
    return param_key in (
        "mean_odds",
        "mean_prob",
        "mean_disp",
    ) or param_key.startswith("two_state")


def _gene_expression_target_is_mu(param_key: str) -> bool:
    """Whether the *gene-level* expression parameter is ``mu`` (not ``r``).

    The gene-level builders (``_hierarchicalize_mu`` / ``_horseshoe_mu`` /
    ``_neg_mu``) target ``mu`` only for the explicit NB mean parameterizations;
    unlike the dataset-level rule they do NOT fold in the two-state family.
    Reproduced exactly so the gene-level NCP upgrade stays byte-identical.
    """
    return param_key in ("mean_odds", "mean_prob", "mean_disp")


# ------------------------------------------------------------------------------
# The descriptor
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class HierParam:
    """Declarative description of one hierarchical parameter's site/spec names.

    A descriptor is resolved once from ``(role, param_key)`` at a given level
    (gene or dataset) and then read by the generic factory NCP cores. Every
    field that varies across the per-parameter builder families is an explicit
    slot here — nothing is hidden in an ad-hoc conditional inside a core.

    Attributes
    ----------
    target : str
        Resolved target site name: ``"mu"``/``"r"`` (expression), ``"p"``/
        ``"phi"`` (probability), ``"gate"``, or a two-state regime coordinate.
    is_sigmoid : bool
        Single sigmoid-vs-positive discriminator. ``True`` for the unit-interval
        targets (``p``, ``gate``, ``inv_concentration``); ``False`` for the
        positive targets (``mu``/``r``, ``phi``, other regime coords).
    loc, scale, prefix : str
        Hyperprior site names: the population location site, the hierarchy scale
        site (replaced by the NCP hyper triplet/pair on upgrade), and the naming
        prefix used for that triplet (``tau_{prefix}``, ``lambda_{prefix}``,
        ``c_sq_{prefix}`` for horseshoe; ``zeta_{prefix}``, ``psi_{prefix}`` for
        NEG).
    raw : str
        The non-centered raw-deviation site name. **Stored, never derived** —
        the dataset-level ``p``/``phi``/``gate``/regime use ``{target}_raw_dataset``
        while everything else (all gene-level, and dataset ``mu``/``r``) uses
        ``{target}_raw`` (verified ``factory.py`` gene vs dataset builders).
    hier_cls : Type[ParamSpec]
        The Gaussian hierarchy spec class the NCP upgrade matches against
        (``[Dataset]Hierarchical{Positive,Sigmoid}NormalSpec`` selected by level
        + ``is_sigmoid``).
    hs_cls, neg_cls : Type[ParamSpec]
        The horseshoe / NEG spec classes the upgrade builds
        (``{Horseshoe,NEG}[Dataset]Hierarchical/{...}NormalSpec``).

    Gaussian-construction fields (Core B — building the Gaussian hierarchy
    triplet; ignored by the NCP cores)
    ----------------------------------------------------------------------
    pop_loc_default : tuple
        ``default_params`` of the population-location ``hyper_loc`` spec
        (``(0.0, 1.0)`` for mu/p; ``(-5.0, 0.01)`` for the gate's tight off
        anchor). Ignored when ``inherit_pop_loc_from_flat`` is set.
    pop_scale_default : tuple
        ``default_params`` of the population-scale ``hyper_scale`` spec
        (``(-2.0, 0.5)`` for mu/gate/regime; ``(0.0, 0.5)`` for p).
    pop_loc_gene : bool
        Whether ``hyper_loc`` is per-gene (``shape_dims=("n_genes",)``) vs
        scalar (``()``). For dataset ``p`` this is overridden by the call-time
        ``mode`` argument.
    hier_gene : bool
        Whether the hierarchy spec is per-gene vs scalar. For dataset ``p`` this
        is overridden by the call-time ``mode`` argument.
    pop_loc_inherits_mixture : bool
        Whether ``hyper_loc.is_mixture`` tracks the flat spec's ``is_mixture``
        (dataset mu/p/regime) vs is always ``False`` (gene-level and gate).
    inherit_pop_loc_from_flat : bool
        **Regime only** — ``hyper_loc.default_params`` inherits the flat regime
        spec's prior/default so the per-parameterization "default-to-NB" tilt is
        preserved at the population level.
    is_dataset_level : bool
        Whether the hierarchy spec is marked ``is_dataset=True`` (dataset-level
        builders) — also gates the dataset axis in ``resolve_shape``.
    threads_shared_components : bool
        Whether ``shared_component_indices`` is threaded into the hierarchy spec
        (dataset mu/p/regime, but NOT gate, and not gene-level).
    """

    target: str
    is_sigmoid: bool
    loc: str
    scale: str
    prefix: str
    raw: str
    hier_cls: Type[ParamSpec]
    hs_cls: Type[ParamSpec]
    neg_cls: Type[ParamSpec]
    # Core B (Gaussian construction) — defaults cover the common case; the
    # constructors override the handful of axes that differ per parameter.
    pop_loc_default: Optional[Tuple[float, float]] = (0.0, 1.0)
    pop_scale_default: Tuple[float, float] = (-2.0, 0.5)
    pop_loc_gene: bool = True
    hier_gene: bool = True
    pop_loc_inherits_mixture: bool = False
    inherit_pop_loc_from_flat: bool = False
    is_dataset_level: bool = False
    threads_shared_components: bool = False


# ------------------------------------------------------------------------------
# Matched spec-class triples, selected by level + ``is_sigmoid``
# ------------------------------------------------------------------------------

# Each triple is (gaussian-hierarchy spec [matched], horseshoe spec [built],
# NEG spec [built]).
_GENE_POSITIVE = (
    HierarchicalPositiveNormalSpec,
    HorseshoeHierarchicalPositiveNormalSpec,
    NEGHierarchicalPositiveNormalSpec,
)
_GENE_SIGMOID = (
    HierarchicalSigmoidNormalSpec,
    HorseshoeHierarchicalSigmoidNormalSpec,
    NEGHierarchicalSigmoidNormalSpec,
)
_DS_POSITIVE = (
    DatasetHierarchicalPositiveNormalSpec,
    HorseshoeDatasetPositiveNormalSpec,
    NEGDatasetPositiveNormalSpec,
)
_DS_SIGMOID = (
    DatasetHierarchicalSigmoidNormalSpec,
    HorseshoeDatasetSigmoidNormalSpec,
    NEGDatasetSigmoidNormalSpec,
)


def _classes(level: str, is_sigmoid: bool):
    if level == "gene":
        return _GENE_SIGMOID if is_sigmoid else _GENE_POSITIVE
    return _DS_SIGMOID if is_sigmoid else _DS_POSITIVE


# ------------------------------------------------------------------------------
# Constructors
# ------------------------------------------------------------------------------


def gene_hier_param(role: str, param_key: str) -> HierParam:
    """Resolve the gene-level descriptor for a hierarchy role.

    Parameters
    ----------
    role : str
        One of ``"expression"`` (mu/r), ``"prob"`` (p/phi), or ``"gate"``.
    param_key : str
        Parameterization registry key (e.g. ``"canonical"``, ``"mean_odds"``).

    Returns
    -------
    HierParam
        Descriptor whose fields reproduce the current gene-level builders'
        (``_hierarchicalize_*`` / ``_horseshoe_*`` / ``_neg_*``) names exactly.
    """
    # Gene-level construction is shared/scalar (no dataset axis, no
    # shared_component_indices) — only the p/phi pop-loc geometry and scale
    # default differ from mu/r.
    pop_loc_gene = True
    pop_scale_default = (-2.0, 0.5)
    if role == "expression":
        if _gene_expression_target_is_mu(param_key):
            target, prefix = "mu", "mu"
        else:
            target, prefix = "r", "r"
        loc, scale = f"log_{prefix}_loc", f"log_{prefix}_scale"
        is_sigmoid = False
    elif role == "prob":
        if param_key == "mean_odds":
            target, prefix = "phi", "phi"
            loc, scale = "log_phi_loc", "log_phi_scale"
            is_sigmoid = False
        else:
            target, prefix = "p", "p"
            loc, scale = "logit_p_loc", "logit_p_scale"
            is_sigmoid = True
        # _hierarchicalize_p uses a scalar pop-loc and a tighter scale prior.
        pop_loc_gene = False
        pop_scale_default = (0.0, 0.5)
    elif role == "gate":
        target, prefix = "gate", "gate"
        loc, scale = "logit_gate_loc", "logit_gate_scale"
        is_sigmoid = True
    else:
        raise ValueError(f"unknown hierarchy role: {role!r}")

    hier_cls, hs_cls, neg_cls = _classes("gene", is_sigmoid)
    return HierParam(
        target=target,
        is_sigmoid=is_sigmoid,
        loc=loc,
        scale=scale,
        prefix=prefix,
        raw=f"{target}_raw",  # gene-level raw is uniformly {target}_raw
        hier_cls=hier_cls,
        hs_cls=hs_cls,
        neg_cls=neg_cls,
        pop_loc_default=(0.0, 1.0),
        pop_scale_default=pop_scale_default,
        pop_loc_gene=pop_loc_gene,
        hier_gene=True,
        pop_loc_inherits_mixture=False,
        is_dataset_level=False,
        threads_shared_components=False,
    )


def dataset_hier_param(role: str, param_key: str) -> HierParam:
    """Resolve the dataset-level descriptor for a single-axis hierarchy role.

    Parameters
    ----------
    role : str
        One of ``"expression"`` (mu/r), ``"prob"`` (p/phi), or ``"gate"``.
    param_key : str
        Parameterization registry key (e.g. ``"canonical"``, ``"mean_odds"``).

    Returns
    -------
    HierParam
        Descriptor whose fields reproduce the current dataset-level builders'
        (``_datasetify_*`` / ``_horseshoe_dataset_*`` / ``_neg_dataset_*``) names
        exactly.
    """
    # Dataset-level construction shares: is_dataset=True, pop-loc tracks the
    # flat spec's mixture flag, and shared_component_indices is threaded. The
    # gate is the outlier — a tight scalar off-anchor with NO shrinkage pooling.
    pop_loc_default = (0.0, 1.0)
    pop_scale_default = (-2.0, 0.5)
    pop_loc_gene = True
    pop_loc_inherits_mixture = True
    threads_shared_components = True
    if role == "expression":
        if expression_target_is_mu(param_key):
            target, prefix = "mu", "mu_dataset"
            loc, scale = "log_mu_dataset_loc", "log_mu_dataset_scale"
        else:
            target, prefix = "r", "r_dataset"
            loc, scale = "log_r_dataset_loc", "log_r_dataset_scale"
        is_sigmoid = False
        raw = f"{target}_raw"  # dataset mu/r: {target}_raw (NOT _dataset)
    elif role == "prob":
        if param_key == "mean_odds":
            target, prefix = "phi", "phi_dataset"
            loc, scale = "log_phi_dataset_loc", "log_phi_dataset_scale"
            is_sigmoid = False
        else:
            target, prefix = "p", "p_dataset"
            loc, scale = "logit_p_dataset_loc", "logit_p_dataset_scale"
            is_sigmoid = True
        raw = f"{target}_raw_dataset"
        # _datasetify_p uses a tighter pop-scale; pop_loc_gene follows the
        # call-time ``mode`` (scalar vs gene_specific), so its default is moot.
        pop_scale_default = (0.0, 0.5)
    elif role == "gate":
        target, prefix = "gate", "gate_dataset"
        loc, scale = "logit_gate_dataset_loc", "logit_gate_dataset_scale"
        is_sigmoid = True
        raw = "gate_raw_dataset"
        # _datasetify_gate: scalar tight off-anchor, no cross-dataset pooling.
        pop_loc_default = (-5.0, 0.01)
        pop_loc_gene = False
        pop_loc_inherits_mixture = False
        threads_shared_components = False
    else:
        raise ValueError(f"unknown hierarchy role: {role!r}")

    hier_cls, hs_cls, neg_cls = _classes("dataset", is_sigmoid)
    return HierParam(
        target=target,
        is_sigmoid=is_sigmoid,
        loc=loc,
        scale=scale,
        prefix=prefix,
        raw=raw,
        hier_cls=hier_cls,
        hs_cls=hs_cls,
        neg_cls=neg_cls,
        pop_loc_default=pop_loc_default,
        pop_scale_default=pop_scale_default,
        pop_loc_gene=pop_loc_gene,
        hier_gene=True,
        pop_loc_inherits_mixture=pop_loc_inherits_mixture,
        is_dataset_level=True,
        threads_shared_components=threads_shared_components,
    )


def regime_dataset_hier_param(
    parameterization, target_override: Optional[str] = None
) -> Optional[HierParam]:
    """Resolve the dataset-level descriptor for a two-state regime coordinate.

    The regime coordinate (``k_off`` / ``switching_ratio`` / ``concentration`` /
    ``inv_concentration``) is resolved from ``TWOSTATE_REGIME_COORD`` (or an
    explicit override). Unlike the single-axis roles its site names are *dynamic*
    in the coordinate, and ``is_sigmoid`` is per-coordinate
    (``inv_concentration`` lives on (0, 1); the others on (0, ∞)). The resulting
    descriptor still has the same shape, so it drives the identical NCP cores.

    Parameters
    ----------
    parameterization : ParamEnum
        Active parameterization; selects the regime coordinate.
    target_override : str, optional
        Explicit coordinate name overriding ``TWOSTATE_REGIME_COORD``.

    Returns
    -------
    HierParam or None
        ``None`` when the parameterization has no regime coordinate (the NCP
        upgrade is then a no-op, matching the former builders' guard).
    """
    coord = target_override or TWOSTATE_REGIME_COORD.get(parameterization)
    if coord is None:
        return None

    is_sigmoid = coord in TWOSTATE_SIGMOID_REGIME_COORDS
    link = "logit" if is_sigmoid else "log"
    hier_cls, hs_cls, neg_cls = _classes("dataset", is_sigmoid)
    return HierParam(
        target=coord,
        is_sigmoid=is_sigmoid,
        loc=f"{link}_{coord}_dataset_loc",
        scale=f"{link}_{coord}_dataset_scale",
        prefix=f"{coord}_dataset",
        raw=f"{coord}_raw_dataset",
        hier_cls=hier_cls,
        hs_cls=hs_cls,
        neg_cls=neg_cls,
        # Like dataset expression, but the pop-loc inherits the flat regime
        # spec's prior to preserve the per-parameterization default-to-NB tilt.
        pop_loc_default=None,
        pop_scale_default=(-2.0, 0.5),
        pop_loc_gene=True,
        hier_gene=True,
        pop_loc_inherits_mixture=True,
        inherit_pop_loc_from_flat=True,
        is_dataset_level=True,
        threads_shared_components=True,
    )
