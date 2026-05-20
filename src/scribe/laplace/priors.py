"""SVI-results-to-empirical-Gaussian-priors adapter for NBLN Laplace.

This module derives informative Gaussian priors on the NBLN-Laplace
globals (``r``, ``mu``, ``eta_capture``) from a previously-fit SVI
results object on the same data. The priors enter the Laplace loss as
proper log-prob terms so their **uncertainty** (not just their location)
shapes both training dynamics and post-fit global Hessian.

Why this exists
---------------
The Laplace-EM path can diverge on the NBLN per-cell Newton when the
NB curvature ``(u + r) p (1 - p)`` collapses on low-count low-``r``
cells. The root cause is that ``r_loc``, ``mu``, and ``eta_loc`` are on
effectively flat priors in the current loss. An NBVCP-SVI fit on the
same data fits robustly and produces posterior samples that anchor
exactly the parameters NBLN needs to constrain.

Coordinate handling
-------------------
The adapter is responsible for moving SVI posterior samples from their
constrained (positive / [0, 1]) space into the target NBLN-Laplace
**unconstrained coordinate** used by ``params``:

- ``r``    : positive → unconstrained via ``model_config.positive_transform``
  inverse (``log`` for ``exp``, ``inv_softplus`` for ``softplus``).
- ``mu``   : positive NB mean → real-valued log-rate via plain ``jnp.log``,
  **regardless** of ``positive_transform``. The NBLN-Laplace ``params["mu"]``
  is the prior mean of a real-valued latent log-rate, not a positive
  parameter.
- ``eta`` : already in constrained [0, ∞) space — identity transform.

Architecture
------------
Two layers:

* :func:`fit_empirical_gaussian` — pure moment-match per coordinate.
  Knows nothing about scribe; expects samples already in the target
  coordinate. Reusable for any sample source (MCMC chains, hand-crafted
  bundles).

* :func:`priors_from_results` — adapter that knows scribe's results
  conventions and the NBLN coordinate mapping. Handles gene identity,
  capture-mode detection, amortization fallback, and coordinate
  conversion before calling the pure utility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from ._global_uncertainty import _JAX_POSITIVE_FNS

logger = logging.getLogger(__name__)


# =====================================================================
# Layer 1 — pure moment-matcher
# =====================================================================


def fit_empirical_gaussian(
    samples_in_target_coord: jnp.ndarray,
    tau: float = 1.0,
    eps_scale: float = 1e-4,
) -> Dict[str, jnp.ndarray]:
    """Per-coordinate Gaussian moment-match from posterior samples.

    Computes ``loc = mean(samples)`` and ``scale = std(samples) * tau``
    along the leading sample axis, then floors ``scale`` at ``eps_scale``
    to prevent zero-variance priors from degenerate-posterior coordinates.

    This is a pure utility: it expects samples already expressed in the
    target coordinate (e.g., already log-transformed for the
    NBLN-Laplace ``r_loc`` slot). All coordinate-conversion lives in
    :func:`priors_from_results`.

    Parameters
    ----------
    samples_in_target_coord : jnp.ndarray
        Shape ``(S, *D)`` where ``S`` is the number of posterior samples
        and ``*D`` is the parameter's natural shape (e.g., ``(G,)`` for
        per-gene, ``(N,)`` for per-cell, ``()`` for scalar).
    tau : float, default 1.0
        Prior-temperature multiplier on the moment-matched scale. Larger
        values soften the prior so the downstream NBLN data has more
        freedom to override; ``tau=1.0`` trusts the SVI posterior exactly.
    eps_scale : float, default 1e-4
        Absolute floor on the post-``tau`` scale.

    Returns
    -------
    Dict[str, jnp.ndarray]
        ``{"loc": (...,), "scale": (...,)}`` — per-coordinate Normal
        parameters in the target coordinate space.
    """
    if samples_in_target_coord.ndim < 1:
        raise ValueError(
            "samples_in_target_coord must have a leading sample axis; "
            f"got shape {samples_in_target_coord.shape}."
        )
    if samples_in_target_coord.shape[0] < 2:
        raise ValueError(
            "Need at least 2 samples to estimate variance; got "
            f"{samples_in_target_coord.shape[0]}."
        )
    loc = jnp.mean(samples_in_target_coord, axis=0)
    raw_scale = jnp.std(samples_in_target_coord, axis=0, ddof=1)
    scale = jnp.maximum(raw_scale * float(tau), float(eps_scale))
    return {"loc": loc, "scale": scale}


# =====================================================================
# Layer 2 — results-to-priors adapter
# =====================================================================


def _resolve_target_pos_inverse(name: str):
    """Look up the inverse of the target ``positive_transform``."""
    if name not in _JAX_POSITIVE_FNS:
        raise ValueError(
            f"Unknown positive_transform={name!r}; "
            f"expected one of {set(_JAX_POSITIVE_FNS)}."
        )
    _forward, inverse = _JAX_POSITIVE_FNS[name]
    return inverse


def _try_results_gene_names(results: Any) -> Optional[np.ndarray]:
    """Defensive attribute chain for source gene-names.

    Tries (in order) ``results.var.index``, ``results.var_names``,
    ``results.adata.var_names``. Returns ``None`` if none of these
    are populated — caller falls back to mask or count check.
    """
    var = getattr(results, "var", None)
    if var is not None:
        idx = getattr(var, "index", None)
        if idx is not None:
            return np.asarray(idx)
    var_names = getattr(results, "var_names", None)
    if var_names is not None:
        return np.asarray(var_names)
    adata = getattr(results, "adata", None)
    if adata is not None:
        adata_vn = getattr(adata, "var_names", None)
        if adata_vn is not None:
            return np.asarray(adata_vn)
    return None


def _try_results_gene_mask(results: Any) -> Optional[np.ndarray]:
    """Defensive attribute chain for source gene mask / subset index."""
    mask = getattr(results, "_gene_coverage_mask", None)
    if mask is not None:
        return np.asarray(mask)
    idx = getattr(results, "_subset_gene_index", None)
    if idx is not None:
        return np.asarray(idx)
    return None


def _check_gene_identity(
    results: Any,
    target_n_genes: int,
    target_gene_names: Optional[np.ndarray],
    target_gene_mask: Optional[np.ndarray],
) -> Tuple[bool, str]:
    """Verify source and target gene panels match.

    Priority: var-names > mask > count-only-with-warning.

    Returns ``(strict_var_name_verified, identity_method)`` where
    ``identity_method`` is one of ``"var_names"``, ``"mask"``,
    ``"count_only"``. Raises ``ValueError`` on a positive mismatch
    that we *can* verify.
    """
    source_n_genes = getattr(results, "n_genes", None)
    if source_n_genes is None:
        source_n_genes = target_n_genes  # cannot verify count either
    if int(source_n_genes) != int(target_n_genes):
        raise ValueError(
            f"Source SVI fit and Laplace target disagree on genes "
            f"(n_src={int(source_n_genes)}, n_tgt={int(target_n_genes)}). "
            f"Did `gene_coverage` change between fits?"
        )

    source_gene_names = _try_results_gene_names(results)
    if source_gene_names is not None and target_gene_names is not None:
        if not np.array_equal(
            np.asarray(source_gene_names), np.asarray(target_gene_names)
        ):
            raise ValueError(
                "Source SVI fit and Laplace target disagree on gene "
                "var_names (counts match but identities differ). Refit "
                "both models on the same gene panel or pass matching "
                "AnnData objects."
            )
        return True, "var_names"

    source_mask = _try_results_gene_mask(results)
    if source_mask is not None and target_gene_mask is not None:
        if not np.array_equal(
            np.asarray(source_mask), np.asarray(target_gene_mask)
        ):
            raise ValueError(
                "Source SVI fit and Laplace target disagree on the gene "
                "coverage mask (counts match but masks differ)."
            )
        return False, "mask"

    logger.warning(
        "Could not verify gene identity beyond count — gene names and "
        "masks are unavailable on the source SVI results. Proceeding "
        "assuming the same gene panel was used in both fits."
    )
    return False, "count_only"


def _detect_capture_mode(samples: Dict[str, jnp.ndarray]) -> str:
    """Detect the source's capture parameterization from sample keys.

    Returns one of ``"eta"``, ``"phi_only"``, ``"none"``.
    """
    if "eta_capture" in samples:
        return "eta"
    if "phi_capture" in samples or "p_capture" in samples:
        return "phi_only"
    return "none"


def _is_amortized_capture(results: Any) -> bool:
    """Detect amortized-capture SVI sources."""
    if hasattr(results, "_uses_amortized_capture"):
        try:
            return bool(results._uses_amortized_capture())
        except Exception:
            return False
    return False


def _draw_samples(
    results: Any,
    n_samples: int,
    source_counts: Optional[jnp.ndarray],
    strict_var_name_verified: bool,
    rng_seed: int,
) -> Dict[str, jnp.ndarray]:
    """Draw posterior samples from SVI results with amortization safeguards.

    Handles amortized-capture sources with the round-6 audit fallback:
    prefer ``results._original_counts``; else accept ``source_counts``
    only when strict var-name identity has been verified; else refuse
    with an explicit error message listing remediation options.
    """
    from jax import random

    rng_key = random.PRNGKey(int(rng_seed))

    if _is_amortized_capture(results):
        # Defensive fallback hierarchy — see plan Round-6 Finding 1.
        svi_source_counts = getattr(results, "_original_counts", None)
        if svi_source_counts is None:
            svi_source_counts = getattr(results, "counts", None)
        if svi_source_counts is not None:
            counts_for_encoder = svi_source_counts
        elif strict_var_name_verified and source_counts is not None:
            counts_for_encoder = source_counts
        else:
            raise ValueError(
                "Source SVI results use amortized capture, but the "
                "encoder's training counts could not be reconstructed "
                "safely. Either:\n"
                "  (a) re-fit SVI with a non-amortized capture "
                "parameterization, OR\n"
                "  (b) ensure the SVI results object stores its training "
                "counts (e.g., via `_original_counts`), OR\n"
                "  (c) pass `source_counts=` explicitly AND ensure both "
                "fits used identical gene panels (var_names match) — "
                "only then is the amortizer's input shape guaranteed "
                "correct."
            )
        return results.get_posterior_samples(
            rng_key=rng_key,
            n_samples=int(n_samples),
            counts=counts_for_encoder,
            store_samples=False,
        )

    return results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=int(n_samples),
        store_samples=False,
    )


def priors_from_results(
    results: Any,
    target_positive_transform: str,
    target_n_genes: int,
    target_n_cells: int,
    target_gene_names: Optional[np.ndarray] = None,
    target_gene_mask: Optional[np.ndarray] = None,
    source_counts: Optional[jnp.ndarray] = None,
    n_samples: int = 1000,
    tau: float = 1.0,
    rng_seed: int = 0,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, jnp.ndarray]], str]:
    """Adapter: SVI results object → empirical Gaussian prior bundle.

    Builds Gaussian priors on the NBLN-Laplace globals ``r``, ``mu``,
    and ``eta`` (when available) from posterior samples drawn from an
    SVI results object. The priors live in the target NBLN-Laplace's
    unconstrained coordinate space and are intended to enter the loss
    as ``dist.Normal(loc, scale).log_prob(params[...])`` terms.

    See the module docstring for the coordinate-mapping table.

    Parameters
    ----------
    results
        Scribe SVI results object with ``get_posterior_samples()`` and,
        ideally, ``var_names`` / ``_gene_coverage_mask`` / ``n_genes``
        attributes for identity verification.
    target_positive_transform : {"exp", "softplus"}
        Resolved from the target ``model_config.positive_transform``.
        Used **only** for the ``r`` coordinate (positive parameter).
        Not used for ``mu`` (which is unconstrained log-rate) or ``eta``
        (already in the target's constrained space).
    target_n_genes, target_n_cells : int
        Target dataset shape — checked against the source.
    target_gene_names : Optional[np.ndarray]
        Target var-names array, if available. Enables strict gene
        identity verification.
    target_gene_mask : Optional[np.ndarray]
        Target ``_gene_coverage_mask``, used as a fallback identity
        signal when var-names are unavailable on either side.
    source_counts : Optional[jnp.ndarray]
        Target's count matrix, passed opportunistically for amortized-
        capture SVI sources. Honored only when strict var-name identity
        was verified.
    n_samples : int, default 1000
        Number of posterior samples to draw for moment matching.
    tau : float, default 1.0
        Prior-temperature multiplier on every prior scale.
    rng_seed : int, default 0
        JAX PRNG seed for sample-drawing reproducibility.
    verbose : bool, default True
        Print user-facing progress messages to stdout at each stage
        (gene-identity verification, SVI sampling, capture-mode
        detection, per-parameter moment-matching, bundle summary).
        Set ``False`` to silence when wrapping in larger pipelines.

    Returns
    -------
    prior_bundle : Dict[str, Dict[str, jnp.ndarray]]
        Per-parameter ``{"loc": ..., "scale": ...}`` dicts. Keys are a
        subset of ``{"r", "mu", "eta"}`` depending on what the source
        provides.
    capture_mode : str
        One of ``"eta"``, ``"phi_only"``, ``"none"``. Used by the
        upstream caller to decide whether to override the target's
        scalar ``capture_anchor``.

    Raises
    ------
    ValueError
        On gene-identity mismatch, on amortized-capture source without
        reconstructible training counts, or when the source provides
        per-cell ``eta_capture`` of the wrong shape.
    """
    if target_positive_transform not in _JAX_POSITIVE_FNS:
        raise ValueError(
            f"Unknown target_positive_transform={target_positive_transform!r}; "
            f"expected one of {set(_JAX_POSITIVE_FNS)}."
        )

    def _say(msg: str) -> None:
        """Emit a user-facing progress line when ``verbose`` is on."""
        if verbose:
            logger.info(msg)

    _say(
        f"Building informative priors from SVI source "
        f"(target G={int(target_n_genes)}, N={int(target_n_cells)}, "
        f"tau={float(tau):.2f}, n_samples={int(n_samples)})"
    )

    # --- Gene-identity safeguard --------------------------------------
    _say("Verifying gene identity against target...")
    strict_var_name_verified, identity_method = _check_gene_identity(
        results=results,
        target_n_genes=int(target_n_genes),
        target_gene_names=target_gene_names,
        target_gene_mask=target_gene_mask,
    )
    _say(f"  Gene identity verified via {identity_method!r}.")

    # --- Draw samples (with amortized-capture handling) ---------------
    if _is_amortized_capture(results):
        _say(
            "Sampling SVI posterior distribution "
            "(amortized capture detected; consulting encoder)..."
        )
    else:
        _say("Sampling SVI posterior distribution...")
    samples = _draw_samples(
        results=results,
        n_samples=int(n_samples),
        source_counts=source_counts,
        strict_var_name_verified=strict_var_name_verified,
        rng_seed=int(rng_seed),
    )
    _say(
        f"  Drew {int(n_samples)} samples; available keys: "
        f"{sorted(samples.keys())}"
    )

    # --- Capture-mode detection ---------------------------------------
    capture_mode = _detect_capture_mode(samples)
    _say(f"Detected capture mode: {capture_mode!r}")
    if capture_mode == "phi_only":
        logger.warning(
            "SVI source uses odds-ratio capture (phi_capture / p_capture), "
            "which is not directly compatible with NBLN's biology-informed "
            "eta capture. r and mu priors will still be applied; the "
            "target's own capture configuration is left intact."
        )
    elif capture_mode == "none":
        logger.warning(
            "SVI source has no capture latent (no eta_capture, phi_capture, "
            "or p_capture key). r and mu priors will be applied; the "
            "target's capture configuration is left intact."
        )

    pos_inverse = _resolve_target_pos_inverse(target_positive_transform)
    prior_bundle: Dict[str, Dict[str, jnp.ndarray]] = {}

    _say("Fitting empirical Gaussian priors to posterior samples...")

    # --- r prior: positive → target unconstrained ---------------------
    if "r" in samples:
        r_samples = jnp.asarray(samples["r"])
        # SVI may return shape (S, G) or (S, 1) for scalar-r models.
        if r_samples.ndim < 2:
            raise ValueError(
                f"Expected r samples to have shape (S, G); got "
                f"{r_samples.shape}."
            )
        if r_samples.shape[1] != int(target_n_genes):
            raise ValueError(
                f"Source r samples have {r_samples.shape[1]} genes; "
                f"target expects {int(target_n_genes)}."
            )
        # Floor at small positive value to avoid log(0) / inv_softplus(0).
        r_pos = jnp.maximum(r_samples, 1e-8)
        r_unconstrained = pos_inverse(r_pos)
        prior_bundle["r"] = fit_empirical_gaussian(
            r_unconstrained, tau=float(tau)
        )
        _say(
            f"  Fitted r prior (G={int(r_samples.shape[1])}, "
            f"transform={target_positive_transform!r} inverse)."
        )

    # --- mu prior: positive NB mean → log-rate (real-valued) ----------
    if "mu" in samples:
        mu_samples = jnp.asarray(samples["mu"])
        if mu_samples.ndim < 2:
            raise ValueError(
                f"Expected mu samples to have shape (S, G); got "
                f"{mu_samples.shape}."
            )
        if mu_samples.shape[1] != int(target_n_genes):
            raise ValueError(
                f"Source mu samples have {mu_samples.shape[1]} genes; "
                f"target expects {int(target_n_genes)}."
            )
        # IMPORTANT: NBLN-Laplace `params["mu"]` is the prior mean of an
        # unconstrained real-valued log-rate latent — not a positive
        # parameter. So the coordinate conversion is plain log(mu), NOT
        # pos_inverse(mu), regardless of model_config.positive_transform.
        mu_pos = jnp.maximum(mu_samples, 1e-8)
        mu_log = jnp.log(mu_pos)
        prior_bundle["mu"] = fit_empirical_gaussian(mu_log, tau=float(tau))
        _say(
            f"  Fitted mu prior (G={int(mu_samples.shape[1])}, "
            "transform='log' — NBLN mu is real-valued log-rate)."
        )

    # --- eta prior: identity (already in target's [0, ∞) space) -------
    if capture_mode == "eta":
        eta_samples = jnp.asarray(samples["eta_capture"])
        if eta_samples.ndim < 2:
            raise ValueError(
                f"Expected eta_capture samples to have shape (S, N); got "
                f"{eta_samples.shape}."
            )
        if eta_samples.shape[1] != int(target_n_cells):
            raise ValueError(
                f"Source eta_capture samples have {eta_samples.shape[1]} "
                f"cells; target expects {int(target_n_cells)}."
            )
        prior_bundle["eta"] = fit_empirical_gaussian(
            eta_samples, tau=float(tau)
        )
        _say(
            f"  Fitted eta prior (N={int(eta_samples.shape[1])}, "
            "transform='identity' — constrained [0, ∞) matches target)."
        )

    _say(
        f"Built informative prior bundle: keys={sorted(prior_bundle.keys())}, "
        f"capture_mode={capture_mode!r}, identity_method={identity_method!r}, "
        f"n_samples={int(n_samples)}, tau={float(tau):.2f}."
    )
    return prior_bundle, capture_mode


__all__ = [
    "fit_empirical_gaussian",
    "priors_from_results",
    "freeze_values_from_results",
    "priors_from_twostate_results",
    "freeze_values_from_twostate_results",
]


# =====================================================================
# Layer 3 — freeze-value extractor (point estimates, no moment-matching)
# =====================================================================


def freeze_values_from_results(
    results: Any,
    target_positive_transform: str,
    target_n_genes: int,
    target_n_cells: int,
    target_gene_names: Optional[np.ndarray] = None,
    target_gene_mask: Optional[np.ndarray] = None,
    source_counts: Optional[jnp.ndarray] = None,
    freeze_params: Tuple[str, ...] = ("r", "eta"),
    map_method: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Extract point-estimate freeze values from an SVI results object.

    Unlike :func:`priors_from_results` (which moment-matches posterior
    samples into Gaussian priors for the soft-cascade loss term), this
    function extracts a single point per coordinate from the SVI
    variational MAP and converts to the NBLN-Laplace target coordinate.

    The freeze values are the **fixed values** used during NBLN's M-step
    when the corresponding parameter is in ``freeze_params``.  No
    moment-matching, no MC error from sampling — the point estimate is
    directly what SVI converged to.

    For the **reported posterior** on frozen parameters, downstream code
    consults the full SVI guide via the embedded ``cascade_source`` field
    on the Laplace result (see :class:`ScribeLaplaceResults`).  This
    function only extracts the M-step point estimate.

    Parameters
    ----------
    results
        Scribe SVI results object with ``get_map()`` and gene-identity
        metadata.
    target_positive_transform : {"exp", "softplus"}
        Resolved from the target ``model_config.positive_transform``.
        Used only for the ``r`` coordinate.
    target_n_genes, target_n_cells : int
        Target dataset shape — checked against the source.
    target_gene_names : Optional[np.ndarray]
        Target var-names array for strict gene identity verification.
    target_gene_mask : Optional[np.ndarray]
        Target gene coverage mask for fallback identity verification.
    source_counts : Optional[jnp.ndarray]
        Target count matrix, passed for amortized-capture SVI sources.
        Same three-tier defensive hierarchy as
        :func:`priors_from_results`.
    freeze_params : Tuple[str, ...], default ("r", "eta")
        Which parameters to extract freeze values for.  Valid keys are
        a subset of ``{"r", "mu", "eta"}``.
    verbose : bool
        Whether to print user-facing progress messages.

    Returns
    -------
    Dict[str, Dict[str, jnp.ndarray]]
        Per-parameter ``{"loc": ...}`` dicts (no ``scale`` — point
        estimates only).  Keys are the requested subset of
        ``{"r", "mu", "eta"}`` that the SVI source can supply.

    Raises
    ------
    ValueError
        On gene-identity mismatch, on amortized-capture source without
        reconstructible training counts, or when a requested freeze
        parameter is absent from the SVI source.
    """
    if target_positive_transform not in _JAX_POSITIVE_FNS:
        raise ValueError(
            f"Unknown target_positive_transform={target_positive_transform!r}; "
            f"expected one of {set(_JAX_POSITIVE_FNS)}."
        )
    valid = {"r", "mu", "eta"}
    invalid = set(freeze_params) - valid
    if invalid:
        raise ValueError(
            f"freeze_params has invalid keys {invalid}; valid = {valid}."
        )

    def _say(msg: str) -> None:
        """Emit a user-facing progress line when ``verbose`` is on."""
        if verbose:
            logger.info(msg)

    _say(
        f"Extracting freeze values from SVI source "
        f"(freeze_params={list(freeze_params)})"
    )

    # --- Gene-identity safeguard (reuses the priors_from_results helper) ---
    strict_var_name_verified, identity_method = _check_gene_identity(
        results=results,
        target_n_genes=int(target_n_genes),
        target_gene_names=target_gene_names,
        target_gene_mask=target_gene_mask,
    )
    _say(f"  Gene identity verified via {identity_method!r}.")

    # --- Amortized-capture-aware get_map() call ---
    # Same defensive hierarchy as priors_from_results._draw_samples:
    # prefer results._original_counts; else accept source_counts only
    # when strict var-name identity verified; else refuse.
    if _is_amortized_capture(results):
        _say(
            "  SVI source uses amortized capture; resolving counts for "
            "encoder evaluation..."
        )
        svi_source_counts = getattr(results, "_original_counts", None)
        if svi_source_counts is None:
            svi_source_counts = getattr(results, "counts", None)
        if svi_source_counts is not None:
            counts_for_encoder = svi_source_counts
        elif strict_var_name_verified and source_counts is not None:
            counts_for_encoder = source_counts
        else:
            raise ValueError(
                "Source SVI results use amortized capture, but the "
                "encoder's training counts could not be reconstructed "
                "safely for get_map(). Same remediation options as "
                "priors_from_results: refit SVI with non-amortized "
                "capture, store training counts on the SVI result, or "
                "pass source_counts with strict var-name identity."
            )
        # CASCADE PROPAGATION: see analogous block in
        # freeze_values_from_twostate_results for the rationale.
        _map_kwargs = (
            {} if map_method is None else {"map_method": map_method}
        )
        map_dict = results.get_map(
            counts=counts_for_encoder, verbose=False, **_map_kwargs
        )
    else:
        _map_kwargs = (
            {} if map_method is None else {"map_method": map_method}
        )
        map_dict = results.get_map(verbose=False, **_map_kwargs)

    _say(f"  SVI MAP keys: {sorted(map_dict.keys())}")

    pos_inverse = _resolve_target_pos_inverse(target_positive_transform)
    freeze_values: Dict[str, Dict[str, jnp.ndarray]] = {}

    # --- r: positive → NBLN unconstrained (via pos_inverse) ---
    if "r" in freeze_params:
        if "r" not in map_dict:
            raise ValueError(
                "freeze_params requests 'r' but SVI source's get_map() "
                "does not include an 'r' key. "
                f"Available keys: {sorted(map_dict.keys())}."
            )
        r_pos = jnp.asarray(map_dict["r"])
        if r_pos.ndim != 1 or r_pos.shape[0] != int(target_n_genes):
            raise ValueError(
                f"SVI 'r' MAP has shape {r_pos.shape}; expected "
                f"({int(target_n_genes)},)."
            )
        r_uncon = pos_inverse(jnp.maximum(r_pos, 1e-8))
        freeze_values["r"] = {"loc": r_uncon}
        _say(
            f"  Extracted r freeze value (G={target_n_genes}, "
            f"transform={target_positive_transform!r} inverse)."
        )

    # --- mu: positive NB mean → NBLN log-rate (via jnp.log) ---
    if "mu" in freeze_params:
        if "mu" not in map_dict:
            raise ValueError(
                "freeze_params requests 'mu' but SVI source's get_map() "
                "does not include a 'mu' key. "
                f"Available keys: {sorted(map_dict.keys())}."
            )
        mu_pos = jnp.asarray(map_dict["mu"])
        if mu_pos.ndim != 1 or mu_pos.shape[0] != int(target_n_genes):
            raise ValueError(
                f"SVI 'mu' MAP has shape {mu_pos.shape}; expected "
                f"({int(target_n_genes)},)."
            )
        mu_log = jnp.log(jnp.maximum(mu_pos, 1e-8))
        freeze_values["mu"] = {"loc": mu_log}
        _say(
            f"  Extracted mu freeze value (G={target_n_genes}, "
            "transform='log' — NBLN mu is real-valued log-rate)."
        )

    # --- eta: constrained [0, ∞) → identity (NBLN's coord is the same) ---
    if "eta" in freeze_params:
        if "eta_capture" not in map_dict:
            raise ValueError(
                "freeze_params requests 'eta' but SVI source's get_map() "
                "does not include an 'eta_capture' key.  The SVI source "
                "may not be using biology-informed capture (only "
                "odds-ratio capture).  Available keys: "
                f"{sorted(map_dict.keys())}."
            )
        eta = jnp.asarray(map_dict["eta_capture"])
        if eta.ndim != 1 or eta.shape[0] != int(target_n_cells):
            raise ValueError(
                f"SVI 'eta_capture' MAP has shape {eta.shape}; expected "
                f"({int(target_n_cells)},)."
            )
        freeze_values["eta"] = {"loc": eta}
        _say(
            f"  Extracted eta freeze value (N={target_n_cells}, "
            "transform='identity' — constrained [0, ∞) matches target)."
        )

    _say(
        f"Built freeze-values bundle: keys={sorted(freeze_values.keys())}, "
        f"requested={list(freeze_params)!r}, identity_method={identity_method!r}."
    )
    return freeze_values


# =====================================================================
# TwoState-LogNormal cascade (TSLN-Rate / TSLN-Logit) adapter
# =====================================================================
#
# Mirrors ``priors_from_results`` / ``freeze_values_from_results`` for
# the TSLN family, with the TwoState SVI source's parameter set:
#
#   SVI source emits (per gene): mu, burst_size, k_off
#                  + deterministics: alpha (= k_on = mu/burst_size),
#                                    beta (= k_off),
#                                    r_hat (= mu + burst_size * k_off),
#                                    eta_act (= log(alpha/beta)) [PR-2 only]
#
# Coordinate maps:
#
#   TSLN-Rate target (PR-1):
#     mu_loc          : pos_inverse(mu_pos)
#     burst_size_loc  : pos_inverse(burst_size_pos)
#     k_off_loc       : pos_inverse(k_off_pos)
#     eta             : identity (per-cell capture)
#
#   TSLN-Logit target (PR-2, deferred):
#     rate_loc        : pos_inverse(r_hat) [prefer deterministic]
#                       or pos_inverse(mu + burst_size * k_off)
#     kappa_loc       : pos_inverse(alpha + beta)
#                       or pos_inverse(mu/burst_size + k_off)
#     eta_anchor_loc  : eta_act (real-valued)
#                       or log(mu/burst_size) - log(k_off)
#     eta             : identity
#
# Both ``target_variant="rate"`` (PR-1) and ``"logit"`` (PR-2) are
# implemented.  The logit branch prefers the SVI source's effective
# ``alpha``/``beta``/``r_hat``/``eta_act`` deterministics over raw
# (mu, burst_size, k_off) derivation — see the inline documentation
# in the function body for the Rev 4 motivation.


def priors_from_twostate_results(
    results: Any,
    target_positive_transform,
    target_n_genes: int,
    target_n_cells: int,
    target_variant: str = "rate",
    target_gene_names: Optional[np.ndarray] = None,
    target_gene_mask: Optional[np.ndarray] = None,
    source_counts: Optional[jnp.ndarray] = None,
    n_samples: int = 1000,
    tau: float = 1.0,
    rng_seed: int = 0,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, jnp.ndarray]], str]:
    """Adapter: TwoState SVI results → TSLN empirical Gaussian prior bundle.

    Builds Gaussian priors on the TSLN-Laplace globals from posterior
    samples drawn from an upstream TwoState SVI fit. The priors live in
    the TSLN target's unconstrained coordinate space and are intended
    to enter the Laplace loss as ``Normal(loc, scale).log_prob(...)``
    terms.

    Parameters
    ----------
    results
        Scribe SVI results from a ``model="twostate"`` (or
        ``"twostatevcp"``) fit. Must expose ``get_posterior_samples()``
        and ideally var-names / gene-mask metadata.
    target_positive_transform : str or Dict[str, str]
        Either a single transform name (e.g. ``"softplus"`` /
        ``"exp"``) applied uniformly to every positive parameter, OR
        a mapping from internal parameter name to transform name.
        Recognized keys: ``{"mu", "burst_size", "k_off"}`` for the
        rate variant; ``{"rate", "kappa"}`` for the logit variant
        (``eta_anchor`` is real-valued — identity transform).  Missing
        keys fall back to ``"softplus"``.  Pass a dict whenever the
        target's ``model_config.positive_transform`` is itself the
        dict form (e.g.
        ``positive_transform={"rate": "exp", "kappa": "softplus"}``)
        — using a single string in that case would silently apply the
        wrong inverse and corrupt the cascade priors.
    target_n_genes, target_n_cells : int
        Target dataset shape.
    target_variant : {"rate", "logit"}, default ``"rate"``
        Which TSLN target variant to build priors for.  Both are
        implemented; ``"logit"`` (PR-2) prefers the SVI source's
        effective deterministics (``alpha`` / ``beta`` / ``r_hat`` /
        ``eta_act``) over raw ``(mu, burst_size, k_off)`` derivation
        per Rev 4.
    n_samples : int, default 1000
    tau : float, default 1.0
    rng_seed : int, default 0
    verbose : bool, default True

    Returns
    -------
    prior_bundle : Dict[str, Dict[str, jnp.ndarray]]
        Per-parameter ``{"loc", "scale"}`` dicts. Keys depend on
        ``target_variant``:
        - rate: subset of ``{"mu", "burst_size", "k_off", "eta"}``.
        - logit: subset of ``{"rate", "kappa", "eta_anchor", "eta"}``.
    capture_mode : str
        ``"eta"`` (per-cell biology-informed) / ``"phi_only"``
        (odds-ratio capture) / ``"none"``.
    """
    if target_variant not in ("rate", "logit"):
        raise ValueError(
            f"Unknown target_variant={target_variant!r}; expected one of "
            "{'rate', 'logit'}."
        )
    # Validate the transform argument — accept either a string or a
    # dict mapping parameter name → transform name.
    if isinstance(target_positive_transform, str):
        if target_positive_transform not in _JAX_POSITIVE_FNS:
            raise ValueError(
                f"Unknown target_positive_transform="
                f"{target_positive_transform!r}; expected one of "
                f"{set(_JAX_POSITIVE_FNS)}."
            )
    elif isinstance(target_positive_transform, dict):
        for k, v in target_positive_transform.items():
            if v not in _JAX_POSITIVE_FNS:
                raise ValueError(
                    f"target_positive_transform[{k!r}]={v!r} is not a "
                    f"recognised transform; expected one of "
                    f"{set(_JAX_POSITIVE_FNS)}."
                )
    else:
        raise TypeError(
            "target_positive_transform must be a string or a dict "
            f"mapping parameter name → transform name; got "
            f"{type(target_positive_transform).__name__}."
        )

    def _say(msg: str) -> None:
        """Emit a user-facing progress line when ``verbose`` is on."""
        if verbose:
            logger.info(msg)

    _say(
        f"Building TSLN-{target_variant} priors from TwoState SVI source "
        f"(G={int(target_n_genes)}, N={int(target_n_cells)}, "
        f"tau={float(tau):.2f}, n_samples={int(n_samples)})"
    )

    # Gene identity (reuse NBLN-side helper — generic across SVI sources)
    strict_var_name_verified, identity_method = _check_gene_identity(
        results=results,
        target_n_genes=int(target_n_genes),
        target_gene_names=target_gene_names,
        target_gene_mask=target_gene_mask,
    )
    _say(f"  Gene identity verified via {identity_method!r}.")

    # Sample SVI posterior (with amortized-capture safeguards).
    if _is_amortized_capture(results):
        _say(
            "Sampling SVI posterior (amortized capture detected)..."
        )
    else:
        _say("Sampling SVI posterior...")
    samples = _draw_samples(
        results=results,
        n_samples=int(n_samples),
        source_counts=source_counts,
        strict_var_name_verified=strict_var_name_verified,
        rng_seed=int(rng_seed),
    )
    _say(
        f"  Drew {int(n_samples)} samples; available keys: "
        f"{sorted(samples.keys())}"
    )

    capture_mode = _detect_capture_mode(samples)
    _say(f"Detected capture mode: {capture_mode!r}")

    # Per-parameter ``pos_inverse`` resolver — handles both the
    # string-form (``"softplus"``) and the dict-form
    # (``{"rate": "exp", "kappa": "softplus"}``) of the target's
    # ``positive_transform`` (Step 5 auditor fix).  Missing keys in the
    # dict fall back to ``"softplus"``.  Real-valued parameters
    # (``eta_anchor``) bypass this and use identity.
    def _pos_inv_for(param_name: str):
        if isinstance(target_positive_transform, dict):
            xform = target_positive_transform.get(param_name, "softplus")
        else:
            xform = target_positive_transform
        return _resolve_target_pos_inverse(xform)

    prior_bundle: Dict[str, Dict[str, jnp.ndarray]] = {}

    if target_variant == "logit":
        # --- TSLN-Logit coordinate map (plan §4.C.4 Rev 4) -----------
        # PRIMARY PATH: use the effective parameters emitted by the
        # TwoState SVI as ``numpyro.deterministic`` sites
        # (``alpha``, ``beta``, ``r_hat``, ``eta_act``).  These are
        # the POST-FLOOR effective values consistent with the
        # likelihood the upstream fit actually generated — if the
        # SVI's ``_twostate_reparam`` activated mean-preserving
        # floors (``_ALPHA_MIN``, ``_K_OFF_MIN``), the raw
        # ``(mu, burst_size, k_off)`` derivation would produce
        # cascade priors out of phase with the likelihood.
        #
        # FALLBACK PATH: raw (mu, burst_size, k_off) derivation —
        # only consistent when no floors activated upstream.  Emits
        # a UserWarning when the fallback fires.
        has_effective = all(
            k in samples for k in ("alpha", "beta", "r_hat")
        )
        if has_effective:
            alpha_s = jnp.maximum(jnp.asarray(samples["alpha"]), 1e-8)
            beta_s = jnp.maximum(jnp.asarray(samples["beta"]), 1e-8)
            rate_s = jnp.maximum(jnp.asarray(samples["r_hat"]), 1e-8)
            kappa_s = alpha_s + beta_s
            if "eta_act" in samples:
                eta_anchor_s = jnp.asarray(samples["eta_act"])
            else:
                eta_anchor_s = jnp.log(alpha_s) - jnp.log(beta_s)
            source_path = "effective deterministics (alpha, beta, r_hat, eta_act)"
        else:
            # Raw fallback: derive sample-wise from (mu, burst_size, k_off).
            for src in ("mu", "burst_size", "k_off"):
                if src not in samples:
                    raise ValueError(
                        f"SVI source missing required key {src!r} (and "
                        "no effective alpha/beta/r_hat deterministics "
                        f"either). Available keys: {sorted(samples.keys())}."
                    )
            mu_s = jnp.maximum(jnp.asarray(samples["mu"]), 1e-8)
            bs_s = jnp.maximum(jnp.asarray(samples["burst_size"]), 1e-8)
            ko_s = jnp.maximum(jnp.asarray(samples["k_off"]), 1e-8)
            # rate = r_hat = mu + burst_size · k_off  (TwoState reparam).
            rate_s = mu_s + bs_s * ko_s
            # kappa = alpha + beta = mu/burst_size + k_off.
            kappa_s = mu_s / bs_s + ko_s
            # eta_anchor = log(alpha/beta) = log(mu / (burst_size · k_off)).
            eta_anchor_s = jnp.log(mu_s) - jnp.log(bs_s) - jnp.log(ko_s)
            import warnings
            warnings.warn(
                "TSLN-Logit cascade fell back to raw "
                "(mu, burst_size, k_off) derivation because the SVI "
                "source did not expose effective alpha/beta/r_hat "
                "deterministics. If the SVI fit activated "
                "mean-preserving floors in _twostate_reparam, the "
                "cascade priors may be inconsistent with the upstream "
                "likelihood. Re-fit the SVI source with the latest "
                "scribe to get the effective deterministics.",
                UserWarning,
                stacklevel=3,
            )
            source_path = "raw (mu/burst_size/k_off) — fallback"

        # Validate shapes.
        for arr_name, arr in (
            ("rate", rate_s), ("kappa", kappa_s), ("eta_anchor", eta_anchor_s),
        ):
            if arr.ndim < 2 or arr.shape[1] != int(target_n_genes):
                raise ValueError(
                    f"Derived {arr_name!r} samples have shape "
                    f"{arr.shape}; expected (S, {int(target_n_genes)})."
                )

        # Each positive global uses its OWN configured transform so
        # that ``positive_transform={"rate": "exp", "kappa":
        # "softplus"}`` is honored.  ``eta_anchor`` is real-valued
        # so it skips ``pos_inverse`` regardless.
        rate_pos_inv = _pos_inv_for("rate")
        kappa_pos_inv = _pos_inv_for("kappa")
        prior_bundle["rate"] = fit_empirical_gaussian(
            rate_pos_inv(rate_s), tau=float(tau)
        )
        prior_bundle["kappa"] = fit_empirical_gaussian(
            kappa_pos_inv(kappa_s), tau=float(tau)
        )
        prior_bundle["eta_anchor"] = fit_empirical_gaussian(
            eta_anchor_s, tau=float(tau)  # real-valued — identity transform
        )
        _say(
            f"  Built TSLN-Logit priors from {source_path}; "
            f"per-parameter positive_transform={target_positive_transform!r}; "
            "identity for eta_anchor."
        )

    else:
        # --- TSLN-Rate coordinate map --------------------------------
        # All three positive globals pass through their own configured
        # pos_inverse — so ``positive_transform={"mu": "exp",
        # "burst_size": "softplus", ...}`` is honored per-parameter.
        for src_key, tgt_key in (
            ("mu", "mu"),
            ("burst_size", "burst_size"),
            ("k_off", "k_off"),
        ):
            if src_key not in samples:
                raise ValueError(
                    f"SVI source missing required key {src_key!r}. "
                    f"Available keys: {sorted(samples.keys())}. The "
                    "TSLN-Rate cascade requires the natural TwoState "
                    "parameterization (mu, burst_size, k_off) on the "
                    "source."
                )
            s = jnp.asarray(samples[src_key])
            if s.ndim < 2 or s.shape[1] != int(target_n_genes):
                raise ValueError(
                    f"SVI {src_key!r} samples have shape {s.shape}; "
                    f"expected (S, {int(target_n_genes)})."
                )
            s_pos = jnp.maximum(s, 1e-8)
            tgt_pos_inv = _pos_inv_for(tgt_key)
            s_uncon = tgt_pos_inv(s_pos)
            prior_bundle[tgt_key] = fit_empirical_gaussian(
                s_uncon, tau=float(tau)
            )
            _say(
                f"  Fitted {tgt_key!r} prior (G={int(s.shape[1])})."
            )

    # --- eta (per-cell capture) -----------------------------------------
    if capture_mode == "eta":
        eta_samples = jnp.asarray(samples["eta_capture"])
        if eta_samples.ndim < 2 or eta_samples.shape[1] != int(target_n_cells):
            raise ValueError(
                f"SVI eta_capture samples have shape {eta_samples.shape}; "
                f"expected (S, {int(target_n_cells)})."
            )
        prior_bundle["eta"] = fit_empirical_gaussian(
            eta_samples, tau=float(tau)
        )
        _say(
            f"  Fitted eta prior (N={int(eta_samples.shape[1])}, "
            "transform='identity')."
        )
    elif capture_mode == "phi_only":
        # Convert the SVI source's p_capture (or phi_capture) into the
        # TSLN-Rate / NBLN-Laplace ``eta_capture`` coordinate system.
        # The mapping is mathematically clean:
        #
        #   NBVCP / twostatevcp:  log λ_cg = log r_g + log p_capture_c
        #   biology-anchored:     log λ_cg = log r_g − eta_capture_c
        #   ⇒  eta_capture_c = − log p_capture_c.
        #
        # A cell with ``p_capture = 0.1`` maps to ``eta_capture ≈ 2.30``
        # (positive — larger ``eta`` ↔ smaller capture).  Both quantities
        # live in the same per-cell semantic space; only the
        # coordinate / sign differs.
        #
        # ``phi_capture`` (mean-odds form) is first mapped back to
        # ``p_capture`` via ``p = phi / (1 + phi)``, then through the
        # same log transform.
        if "p_capture" in samples:
            p_cap_samples = jnp.maximum(
                jnp.asarray(samples["p_capture"]), 1e-8
            )
        else:
            phi_samples = jnp.maximum(
                jnp.asarray(samples["phi_capture"]), 1e-8
            )
            p_cap_samples = phi_samples / (1.0 + phi_samples)
            p_cap_samples = jnp.maximum(
                jnp.clip(p_cap_samples, 1e-8, 1.0 - 1e-8), 1e-8
            )

        if p_cap_samples.ndim < 2 or p_cap_samples.shape[1] != int(
            target_n_cells
        ):
            raise ValueError(
                f"SVI p_capture / phi_capture samples have shape "
                f"{p_cap_samples.shape}; expected "
                f"(S, {int(target_n_cells)})."
            )
        # Sample-wise log transform.  ``eta`` is real-valued (no
        # positive transform needed); the Laplace target stores it
        # directly in this coordinate.
        eta_samples = -jnp.log(p_cap_samples)
        prior_bundle["eta"] = fit_empirical_gaussian(
            eta_samples, tau=float(tau)
        )
        # Promote the detected mode to "eta" so downstream code
        # treats the cascade as having biology-anchored capture (the
        # bridge then sets up the ``x_only_offset`` Newton path when
        # ``eta`` is hard-frozen, or the joint Newton with the
        # per-cell soft cascade scale otherwise).
        capture_mode = "eta"
        _say(
            f"  Mapped p_capture/phi_capture → eta_capture per cell "
            f"(eta = −log p, N={int(eta_samples.shape[1])}).  "
            "Promoted capture_mode 'phi_only' → 'eta' so the cascade "
            "passes per-cell capture information through to the "
            "Laplace fit."
        )

    _say(
        f"Built TSLN-{target_variant} prior bundle: "
        f"keys={sorted(prior_bundle.keys())}, capture_mode={capture_mode!r}, "
        f"identity_method={identity_method!r}, n_samples={int(n_samples)}, "
        f"tau={float(tau):.2f}."
    )
    return prior_bundle, capture_mode


def freeze_values_from_twostate_results(
    results: Any,
    target_positive_transform,
    target_n_genes: int,
    target_n_cells: int,
    target_variant: str = "rate",
    target_gene_names: Optional[np.ndarray] = None,
    target_gene_mask: Optional[np.ndarray] = None,
    source_counts: Optional[jnp.ndarray] = None,
    freeze_params: Tuple[str, ...] = ("mu", "burst_size", "k_off"),
    map_method: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Extract point-estimate freeze values from a TwoState SVI fit.

    Hard-cascade analog of :func:`priors_from_twostate_results`. For
    each parameter in ``freeze_params``, returns the SVI MAP value
    transformed to the TSLN target's unconstrained coordinate.

    Parameters
    ----------
    results
        TwoState SVI results object with ``get_map()``.
    target_positive_transform : {"exp", "softplus"}
    target_n_genes, target_n_cells : int
    target_variant : {"rate", "logit"}, default ``"rate"``
        Both variants implemented.  For ``"logit"`` the valid keys
        are ``{"rate", "kappa", "eta_anchor", "eta"}``.
    freeze_params : Tuple[str, ...], default ``("mu", "burst_size", "k_off")``
        For TSLN-Rate, valid keys are
        ``{"mu", "burst_size", "k_off", "eta"}``.  For TSLN-Logit,
        valid keys are ``{"rate", "kappa", "eta_anchor", "eta"}``.
    map_method : str, optional
        Controls which ``map_method`` is passed to the SVI source's
        :meth:`~scribe.svi.results.ScribeSVIResults.get_map`. ``None``
        (default) lets the SVI source use its own default (currently
        ``"auto"``, i.e., Jacobian-corrected MAP). Pass ``"transform"``
        to pin the cascade to legacy ``transform(loc)`` semantics —
        useful when reproducing pre-correction cascade fits.

        IMPORTANT: when the SVI default flips to ``"auto"``, this
        cascade silently picks up the corrected (Jacobian-shifted)
        MAP values, which propagate into the downstream Laplace fit.
        For reproducibility of old scripts, pin to ``"transform"``.

    Returns
    -------
    Dict[str, Dict[str, jnp.ndarray]]
        Per-parameter ``{"loc": ...}`` dicts (no ``scale``).
    """
    if target_variant not in ("rate", "logit"):
        raise ValueError(
            f"Unknown target_variant={target_variant!r}; expected one of "
            "{'rate', 'logit'}."
        )
    # Accept str or per-parameter dict, same as priors_from_twostate_results.
    if isinstance(target_positive_transform, str):
        if target_positive_transform not in _JAX_POSITIVE_FNS:
            raise ValueError(
                f"Unknown target_positive_transform="
                f"{target_positive_transform!r}."
            )
    elif isinstance(target_positive_transform, dict):
        for k, v in target_positive_transform.items():
            if v not in _JAX_POSITIVE_FNS:
                raise ValueError(
                    f"target_positive_transform[{k!r}]={v!r} is not a "
                    f"recognised transform; expected one of "
                    f"{set(_JAX_POSITIVE_FNS)}."
                )
    else:
        raise TypeError(
            "target_positive_transform must be a string or a dict; got "
            f"{type(target_positive_transform).__name__}."
        )
    valid_by_variant = {
        "rate": {"mu", "burst_size", "k_off", "eta"},
        "logit": {"rate", "kappa", "eta_anchor", "eta"},
    }
    valid = valid_by_variant[target_variant]
    invalid = set(freeze_params) - valid
    if invalid:
        raise ValueError(
            f"freeze_params has invalid keys {invalid}; valid = {valid} "
            f"for target_variant={target_variant!r}."
        )

    def _say(msg: str) -> None:
        """Emit a user-facing progress line when ``verbose`` is on."""
        if verbose:
            logger.info(msg)

    _say(
        f"Extracting TSLN-{target_variant} freeze values "
        f"(freeze_params={list(freeze_params)})"
    )

    strict_var_name_verified, identity_method = _check_gene_identity(
        results=results,
        target_n_genes=int(target_n_genes),
        target_gene_names=target_gene_names,
        target_gene_mask=target_gene_mask,
    )
    _say(f"  Gene identity verified via {identity_method!r}.")

    # Amortized-capture-aware get_map()
    if _is_amortized_capture(results):
        _say("  Source uses amortized capture; resolving encoder counts...")
        svi_source_counts = getattr(results, "_original_counts", None)
        if svi_source_counts is None:
            svi_source_counts = getattr(results, "counts", None)
        if svi_source_counts is not None:
            counts_for_encoder = svi_source_counts
        elif strict_var_name_verified and source_counts is not None:
            counts_for_encoder = source_counts
        else:
            raise ValueError(
                "Source uses amortized capture but counts can't be "
                "resolved. Same remediation options as the NBLN cascade."
            )
        # CASCADE PROPAGATION: pass map_method through to the SVI
        # source's get_map. ``None`` lets the source use its own
        # default; an explicit value pins the cascade to that method
        # (e.g., ``"transform"`` reproduces legacy uncorrected behavior).
        _map_kwargs = (
            {} if map_method is None else {"map_method": map_method}
        )
        map_dict = results.get_map(
            counts=counts_for_encoder, verbose=False, **_map_kwargs
        )
    else:
        _map_kwargs = (
            {} if map_method is None else {"map_method": map_method}
        )
        map_dict = results.get_map(verbose=False, **_map_kwargs)

    _say(f"  SVI MAP keys: {sorted(map_dict.keys())}")

    # Resolve (burst_size, k_off) MAP from whichever TwoState
    # parameterization the SVI source used.  ``get_map()`` only
    # returns sampled sites (not deterministics), so for non-natural
    # parameterizations we derive (burst_size, k_off) by running the
    # appropriate reparam helper on the MAP of the sampled params.
    #
    # Skip this whole block for TSLN-Logit when the effective
    # deterministics are present — the logit branch below consumes
    # ``(alpha, beta, r_hat, eta_act)`` directly and does NOT need
    # ``(burst_size, k_off)``.
    # ``get_map()`` returns only the *sampled* sites, so deterministics
    # (``alpha`` / ``beta`` / ``r_hat`` / ``eta_act``) are NOT
    # available from the MAP unless we derive them locally.  Run the
    # appropriate reparam helper on the MAP of the sampled params and
    # populate the effective deterministics — this gives the logit
    # variant the same Rev 4 "effective-parameter primary path" the
    # samples-based cascade uses.
    _logit_effective_available = (
        target_variant == "logit"
        and all(k in map_dict for k in ("alpha", "beta", "r_hat"))
    )
    _rate_needs_derivation = (
        target_variant == "rate"
        and ("burst_size" not in map_dict or "k_off" not in map_dict)
    )
    _logit_needs_derivation = (
        target_variant == "logit"
        and not _logit_effective_available
    )
    if _rate_needs_derivation or _logit_needs_derivation:
        from ..models.components.likelihoods.two_state import (
            _twostate_reparam,
            _twostate_ratio_reparam,
            _twostate_moments_reparam,
            _twostate_moment_delta_reparam,
        )

        mu_map = jnp.asarray(map_dict["mu"])
        if "burst_size" in map_dict and "k_off" in map_dict:
            # Natural parameterization — burst_size and k_off already
            # present.  Run _twostate_reparam to get the effective
            # alpha / beta / r_hat (with the floors applied).
            _alpha, _beta, _rate, _eff_b = _twostate_reparam(
                mu_map,
                jnp.asarray(map_dict["burst_size"]),
                jnp.asarray(map_dict["k_off"]),
            )
            bs_derived = jnp.asarray(map_dict["burst_size"])
            ko_derived = _beta
        elif "switching_ratio" in map_dict:
            # ratio parameterization
            _alpha, _beta, _rate, _eff_b = _twostate_ratio_reparam(
                mu_map,
                jnp.asarray(map_dict["burst_size"]),
                jnp.asarray(map_dict["switching_ratio"]),
            )
            bs_derived = jnp.asarray(map_dict["burst_size"])
            ko_derived = _beta
        elif "inv_concentration" in map_dict:
            # moment_delta parameterization
            _alpha, _beta, _rate, _eff_b = _twostate_moment_delta_reparam(
                mu_map,
                jnp.asarray(map_dict["excess_fano"]),
                jnp.asarray(map_dict["inv_concentration"]),
            )
            bs_derived = _eff_b
            ko_derived = _beta
        elif "excess_fano" in map_dict and "concentration" in map_dict:
            # mean_fano parameterization
            _alpha, _beta, _rate, _eff_b = _twostate_moments_reparam(
                mu_map,
                jnp.asarray(map_dict["excess_fano"]),
                jnp.asarray(map_dict["concentration"]),
            )
            bs_derived = _eff_b
            ko_derived = _beta
        else:
            raise ValueError(
                "Cascade source's get_map() does not provide "
                "'burst_size'/'k_off' directly, and the parameterization "
                "is unrecognised (expected one of: natural / ratio / "
                "mean_fano / moment_delta).  Available keys: "
                f"{sorted(map_dict.keys())}."
            )
        # Augment the MAP dict with the derived natural-form values AND
        # the effective deterministics so the per-key loops below find
        # them.  ``eta_act = log(alpha) - log(beta)`` matches the SVI
        # deterministic site emitted by ``_emit_deterministics``.
        map_dict = dict(map_dict)
        map_dict.setdefault("burst_size", bs_derived)
        map_dict.setdefault("k_off", ko_derived)
        map_dict.setdefault("alpha", _alpha)
        map_dict.setdefault("beta", _beta)
        map_dict.setdefault("r_hat", _rate)
        map_dict.setdefault(
            "eta_act",
            jnp.log(jnp.maximum(_alpha, 1e-30))
            - jnp.log(jnp.maximum(_beta, 1e-30)),
        )
        _say(
            "  Derived (burst_size, k_off, alpha, beta, r_hat, eta_act) "
            "from the SVI MAP via _twostate_*_reparam."
        )

    # Per-parameter resolver — same shape as in priors_from_twostate_results.
    def _pos_inv_for(param_name: str):
        if isinstance(target_positive_transform, dict):
            xform = target_positive_transform.get(param_name, "softplus")
        else:
            xform = target_positive_transform
        return _resolve_target_pos_inverse(xform)

    freeze_values: Dict[str, Dict[str, jnp.ndarray]] = {}

    if target_variant == "logit":
        # ---- TSLN-Logit gene-level extraction ------------------------
        # Same primary / fallback split as the priors path: prefer the
        # SVI's effective deterministic MAPs (alpha, beta, r_hat,
        # eta_act) when available; fall back to raw derivation from
        # (mu, burst_size, k_off) with a warning.
        has_effective_map = all(
            k in map_dict for k in ("alpha", "beta", "r_hat")
        )
        if has_effective_map:
            alpha_m = jnp.maximum(jnp.asarray(map_dict["alpha"]), 1e-8)
            beta_m = jnp.maximum(jnp.asarray(map_dict["beta"]), 1e-8)
            rate_m = jnp.maximum(jnp.asarray(map_dict["r_hat"]), 1e-8)
            kappa_m = alpha_m + beta_m
            if "eta_act" in map_dict:
                eta_anchor_m = jnp.asarray(map_dict["eta_act"])
            else:
                eta_anchor_m = jnp.log(alpha_m) - jnp.log(beta_m)
            source_path = "effective deterministics (alpha, beta, r_hat, eta_act)"
        else:
            mu_m = jnp.maximum(jnp.asarray(map_dict["mu"]), 1e-8)
            bs_m = jnp.maximum(jnp.asarray(map_dict["burst_size"]), 1e-8)
            ko_m = jnp.maximum(jnp.asarray(map_dict["k_off"]), 1e-8)
            rate_m = mu_m + bs_m * ko_m
            kappa_m = mu_m / bs_m + ko_m
            eta_anchor_m = jnp.log(mu_m) - jnp.log(bs_m) - jnp.log(ko_m)
            import warnings
            warnings.warn(
                "TSLN-Logit freeze_values fell back to raw "
                "(mu, burst_size, k_off) derivation. See the "
                "priors_from_twostate_results docstring for details.",
                UserWarning,
                stacklevel=3,
            )
            source_path = "raw (mu/burst_size/k_off) — fallback"

        for tgt_key, val, is_positive in (
            ("rate", rate_m, True),
            ("kappa", kappa_m, True),
            ("eta_anchor", eta_anchor_m, False),
        ):
            if tgt_key not in freeze_params:
                continue
            if val.ndim != 1 or val.shape[0] != int(target_n_genes):
                raise ValueError(
                    f"Derived {tgt_key!r} MAP has shape {val.shape}; "
                    f"expected ({int(target_n_genes)},)."
                )
            # Per-parameter positive transform so the dict-form
            # ``positive_transform={"rate": "exp", "kappa": "softplus"}``
            # is honored.
            loc = _pos_inv_for(tgt_key)(val) if is_positive else val
            freeze_values[tgt_key] = {"loc": loc}
            _say(
                f"  Extracted {tgt_key!r} freeze value from "
                f"{source_path} (G={target_n_genes})."
            )
    else:
        # ---- TSLN-Rate gene-level extraction (existing) --------------
        for src_key, tgt_key in (
            ("mu", "mu"),
            ("burst_size", "burst_size"),
            ("k_off", "k_off"),
        ):
            if tgt_key not in freeze_params:
                continue
            if src_key not in map_dict:
                raise ValueError(
                    f"freeze_params requests {tgt_key!r} but SVI MAP "
                    f"has no {src_key!r} key. Available: "
                    f"{sorted(map_dict.keys())}."
                )
            s = jnp.asarray(map_dict[src_key])
            if s.ndim != 1 or s.shape[0] != int(target_n_genes):
                raise ValueError(
                    f"SVI {src_key!r} MAP has shape {s.shape}; expected "
                    f"({int(target_n_genes)},)."
                )
            # Per-parameter positive transform (dict-form support).
            s_uncon = _pos_inv_for(tgt_key)(jnp.maximum(s, 1e-8))
            freeze_values[tgt_key] = {"loc": s_uncon}
            _say(
                f"  Extracted {tgt_key!r} freeze value (G={target_n_genes})."
            )

    if "eta" in freeze_params:
        # Same p_capture / phi_capture → eta mapping as
        # ``priors_from_twostate_results``: ``eta = −log p_capture``.
        # If the SVI source emits ``eta_capture`` directly (biology-
        # anchored), use it unchanged.
        if "eta_capture" in map_dict:
            eta = jnp.asarray(map_dict["eta_capture"])
        elif "p_capture" in map_dict:
            p_cap = jnp.maximum(jnp.asarray(map_dict["p_capture"]), 1e-8)
            eta = -jnp.log(p_cap)
            _say(
                "  Mapped p_capture MAP → eta_capture freeze value "
                "via eta = −log p."
            )
        elif "phi_capture" in map_dict:
            phi = jnp.maximum(jnp.asarray(map_dict["phi_capture"]), 1e-8)
            p_cap = phi / (1.0 + phi)
            eta = -jnp.log(jnp.maximum(p_cap, 1e-8))
            _say(
                "  Mapped phi_capture MAP → eta_capture freeze value "
                "via p = phi/(1+phi), eta = −log p."
            )
        else:
            raise ValueError(
                "freeze_params requests 'eta' but SVI MAP has no "
                "'eta_capture', 'p_capture', or 'phi_capture' key. The "
                "source does not appear to model per-cell capture. "
                f"Available: {sorted(map_dict.keys())}."
            )
        if eta.ndim != 1 or eta.shape[0] != int(target_n_cells):
            raise ValueError(
                f"SVI capture MAP has shape {eta.shape} after mapping; "
                f"expected ({int(target_n_cells)},)."
            )
        freeze_values["eta"] = {"loc": eta}
        _say(
            f"  Extracted eta freeze value (N={target_n_cells})."
        )

    _say(
        f"Built TSLN-{target_variant} freeze-values bundle: "
        f"keys={sorted(freeze_values.keys())}, requested={list(freeze_params)!r}, "
        f"identity_method={identity_method!r}."
    )
    return freeze_values
