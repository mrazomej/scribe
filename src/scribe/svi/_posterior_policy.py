"""Context-aware posterior-site selection policy for SVI sampling.

When :meth:`get_posterior_samples` draws from the variational posterior it
materialises *every* sample and deterministic site in the model. For a
hierarchical fit with a wide interaction factor (e.g. a treatment x donor
random slope under a horseshoe prior) the model registers large deterministic
tensors that some consumers never read:

- per-factor effect sites ``mu_<factor>_effect`` of shape ``(S, L_f, G)`` — for
  an interaction ``L_f = C * D`` levels, several GB at ``S = 5000``;
- per-cell capture / variable-capture-probability (VCP) sites (``eta_capture``,
  ``p_capture`` / ``phi_capture`` and the ``mu_eta*`` hierarchy) of shape
  ``(S, n_cells)`` — the experimental-noise channel that differential
  expression (DE) explicitly removes.

These bloat the stored ``posterior_samples`` dict that the DE pipeline keeps
alive across every donor-leaf slice, and at large ``S`` push memory past the
host cap. This module is the single source of truth for *which* sites a given
consumer needs, expressed as a ``purpose`` policy that resolves to a keep-set
(or ``None`` = keep everything).

Design rules
------------
- **Opt-in / back-compat.** ``purpose=None`` and ``return_sites=None`` resolve
  to ``None`` (keep everything), so existing callers are unaffected.
- **Defined by consumption.** The DE keep-set is exactly the sites the empirical
  DE pipeline reads (``r`` required; ``p`` / ``mu`` / ``phi`` optional — see
  :func:`scribe.de._results_factory._extract_de_inputs`), plus
  ``mixing_weights`` for mixtures. It is *not* derived from an axis-metadata
  proxy such as "gene-specific", because ``p`` is frequently scalar/shared yet
  still required by DE.
- **Whitelist.** Capture, ``*_effect`` and ``*_raw`` sites fall out of the
  keep-set by *not being listed*, so any unanticipated per-factor / per-cell
  site is dropped by default. Requesting a name a given parameterization never
  emits is harmless — the post-merge filter simply does not find it.
- **NB-family contract.** ``de_paired`` / ``de_effect`` keep-sets assume the
  NB-family ``{r, p, mu, phi}`` / ``mu|r_<factor>_effect`` site convention.
  Callers must gate on model family (see
  :func:`scribe.de._factors._supports_de_narrowing`) and only request these
  purposes for NB-family results; other families bypass narrowing entirely.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set, Union

# ``_as_site_set`` lives in the sampling layer (the post-merge filter's home) and
# is re-used here; importing it keeps the ``svi -> sampling`` dependency
# direction and a single source of truth for the string-vs-iterable rule.
from ..sampling._predictive import _as_site_set  # noqa: F401  (re-exported)

# Canonical DE-consumed concentration/mean sites (see ``_extract_de_inputs``).
# ``r`` is required; ``p`` / ``mu`` / ``phi`` are read opportunistically. The
# set is intentionally a fixed contract rather than a parameterization-derived
# subset: requesting a name the active parameterization does not emit is
# harmless (the post-merge filter no-ops on absent keys), while *omitting* a
# name a consumer needs would be a silent correctness bug.
_DE_CANONICAL_SITES = ("r", "p", "mu", "phi")

# Mixture component-weight site (present only for mixture fits, ``n_components
# >= 2``). Empirical DE reads it for mixture-weighted / per-component contrasts.
_MIXTURE_WEIGHT_SITE = "mixing_weights"

# Recognised ``purpose`` values (besides ``None`` = keep everything).
VALID_PURPOSES = ("de_paired", "de_effect", "ppc", "all")


def _canonical_param_keys(model_config) -> Set[str]:
    """Return the DE-consumed canonical concentration/mean sites.

    The set is the empirical-DE input contract ``{r, p, mu, phi}`` (see
    :func:`scribe.de._results_factory._extract_de_inputs`), independent of
    gene-specificity so that a scalar/shared ``p`` is retained. Names absent
    from the active parameterization are harmless (the post-merge filter no-ops
    on them).

    Parameters
    ----------
    model_config : Any
        Fitted-model configuration. Accepted for interface symmetry and future
        per-parameterization refinement; the contract is currently fixed by what
        DE reads and any requested-but-absent name is discarded downstream.

    Returns
    -------
    set of str
        A fresh copy of ``{"r", "p", "mu", "phi"}``.
    """
    return set(_DE_CANONICAL_SITES)


def _is_mixture(model_config) -> bool:
    """Whether the fit is a finite mixture (``n_components >= 2``).

    Parameters
    ----------
    model_config : Any
        Fitted-model configuration; mixtures are flagged by ``n_components``
        (``None`` or ``< 2`` means non-mixture).

    Returns
    -------
    bool
        ``True`` iff ``model_config.n_components >= 2``.
    """
    n = getattr(model_config, "n_components", None)
    try:
        return n is not None and int(n) >= 2
    except (TypeError, ValueError):
        return False


def _effect_site_candidates(factor_name: str) -> Set[str]:
    """Candidate effect + scale site names for ``estimand="effect"``.

    Mirrors the site naming in :mod:`scribe.core.factor_effect_view`: the
    additive effect is exposed as ``mu_<safe>_effect`` (mean parameterizations)
    or ``r_<safe>_effect`` (canonical/standard), where ``<safe>`` replaces
    ``":"`` with ``"__"`` (so interaction factors map cleanly). A *gaussian*
    random effect additionally carries a ``<target>_<safe>_scale`` site; a
    horseshoe folds the per-gene scale into the effect and has no scale site.
    Both target prefixes and both scale candidates are included; the post-merge
    filter discards whichever do not exist for the fitted model.

    Parameters
    ----------
    factor_name : str
        Base-factor (or interaction) name whose effect is read.

    Returns
    -------
    set of str
        Effect and scale site-name candidates for both ``mu`` and ``r`` targets.
    """
    safe = factor_name.replace(":", "__")
    return {
        f"mu_{safe}_effect",
        f"r_{safe}_effect",
        f"mu_{safe}_scale",
        f"r_{safe}_scale",
    }


def resolve_keep_set(
    model_config,
    *,
    purpose: Optional[str] = None,
    return_sites: Optional[Union[str, Iterable[str]]] = None,
    factor_name: Optional[str] = None,
    mixture_weighted: bool = False,
) -> Optional[Set[str]]:
    """Resolve the posterior-site keep-set for a sampling context.

    The returned set is the *requested* keep-set (intent), used both to filter
    the merged posterior dict (see
    :func:`scribe.sampling._predictive.sample_variational_posterior`) and as the
    cache key for site-aware reuse. ``None`` means "keep every site" (the
    back-compat default and the correct behaviour for posterior-predictive
    consumers, which must replay the full generative model).

    Precedence: an explicit ``return_sites`` list (the low-level escape hatch)
    wins over ``purpose``.

    Parameters
    ----------
    model_config : Any
        Fitted-model configuration; consulted for mixture detection.
    purpose : {"de_paired", "de_effect", "ppc", "all"} or None, optional
        High-level policy. ``None`` (default) keeps everything. ``"ppc"`` /
        ``"all"`` also keep everything (PPC must replay the full model).
        ``"de_paired"`` keeps the empirical-DE concentration/mean contract
        ``{r, p, mu, phi}`` (+ ``mixing_weights`` for mixtures), dropping
        capture, ``*_effect`` and ``*_raw`` sites. ``"de_effect"`` keeps only
        the factor's effect (and scale) sites for ``estimand="effect"``.
    return_sites : str, iterable of str, or None, optional
        Explicit keep-set of *internal* site names (escape hatch). Wins over
        ``purpose`` when not ``None``. A bare ``str`` is one site name.
    factor_name : str, optional
        Required when ``purpose="de_effect"``; names the effect site to keep.
    mixture_weighted : bool, default False
        When ``True`` (or the model is a mixture), include ``mixing_weights`` in
        a ``de_paired`` keep-set so per-component / mixture-weighted DE works.

    Returns
    -------
    set of str or None
        The keep-set of internal site names, or ``None`` to keep everything.

    Raises
    ------
    ValueError
        If ``purpose`` is unrecognised, or ``purpose="de_effect"`` without a
        ``factor_name``.
    """
    # Explicit return_sites is the low-level escape hatch and wins over purpose.
    if return_sites is not None:
        return _as_site_set(return_sites)

    # Back-compat: no policy requested -> keep everything.
    if purpose is None:
        return None

    # PPC / "all" must replay the full generative model (capture included), so
    # they keep every site exactly as today.
    if purpose in ("ppc", "all"):
        return None

    if purpose == "de_paired":
        keep = _canonical_param_keys(model_config)
        if mixture_weighted or _is_mixture(model_config):
            keep.add(_MIXTURE_WEIGHT_SITE)
        # Invariant: the empirical-DE composition path requires ``r``. This is a
        # backstop behind the family gate (see ``_supports_de_narrowing``); a
        # parameterization that somehow lacked ``r`` should fail loudly here
        # rather than deep inside DE with a confusing KeyError.
        if "r" not in keep:
            raise ValueError(
                "de_paired keep-set is missing required site 'r'; this "
                "parameterization is not supported for narrowed DE sampling."
            )
        return keep

    if purpose == "de_effect":
        if factor_name is None:
            raise ValueError(
                "purpose='de_effect' requires factor_name to name the effect "
                "site to keep."
            )
        return _effect_site_candidates(factor_name)

    raise ValueError(
        f"unknown purpose {purpose!r}; valid values are {VALID_PURPOSES} "
        "(or None to keep all sites)."
    )
