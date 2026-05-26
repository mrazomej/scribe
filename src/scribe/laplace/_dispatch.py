"""Model-dispatching accessors for Laplace results.

This module houses public methods whose behavior depends on the fitted base
model. Dispatch is centralized here so callers can interact with a single
``ScribeLaplaceResults`` API while backend details remain model-specific.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np

import warnings

import numpyro.distributions as _dist

from ..stats.distributions import LowRankPoissonLogNormal
from ..stats.jacobian_map import jacobian_corrected_map
from ._derived import (
    lnm_p_from_parents,
    twostate_logit_derived_from_parents,
    twostate_rate_derived_from_parents,
)
from ._global_uncertainty import resolve_numpyro_transform, resolve_positive_fns
from ._results_shared import _base_model


# ==============================================================================
# Helpers for the ``map_method`` plumbing in :meth:`get_map`.
# ==============================================================================
#
# Per the plan (v3, §3.5), positive globals on ScribeLaplaceResults are
# stored as ``pos_forward(_loc)`` — the median of Y in constrained
# space, NOT the mode. To return the constrained-space MAP via
# ``map_method != "transform"``, we recompute from the persisted
# ``(_loc, _scale)`` pairs using :func:`jacobian_corrected_map`.
#
# ``_correct_positive`` handles three cases:
# 1. ``loc is None``: parameter not populated (e.g., a frozen field);
#    return None / pass through stored value.
# 2. ``scale is None`` (or all-NaN): no curvature available, fall back
#    to ``transform(loc)``.
# 3. Partial NaN scale: per-element ``jnp.where`` mask blends the
#    corrected value (finite entries) with ``transform(loc)`` (NaN
#    entries — typically frozen genes in a cascade).


def _correct_positive(
    model_config: Any,
    transform_key: str,
    loc: Optional[jnp.ndarray],
    scale: Optional[jnp.ndarray],
    method: str,
) -> Optional[jnp.ndarray]:
    """Compute the constrained-space MAP for a positive global.

    Parameters
    ----------
    model_config : Any
        Model config used by ``resolve_numpyro_transform`` to look up
        the per-parameter ``Transform``.
    transform_key : str
        Key into ``model_config.positive_transform`` mapping (NOT the
        output dict key — e.g., for TSLN-Rate the ``gene_mean`` output
        is the positive form of the ``"mu"`` latent, so the
        transform_key is ``"mu"``).
    loc, scale : Optional[jnp.ndarray]
        Unconstrained posterior parameterization. ``loc`` is the
        Laplace-fitted MAP in unconstrained space; ``scale`` is the
        post-fit diagonal Hessian standard deviation (may be ``None``
        or contain NaNs for frozen / unprofiled parameters).
    method : str
        Effective ``map_method`` after the dataclass-vs-call default
        resolution. ``"transform"`` always returns ``transform(loc)``;
        other values dispatch through :func:`jacobian_corrected_map`.

    Returns
    -------
    Optional[jnp.ndarray]
        The constrained-space MAP, or ``None`` if ``loc is None``.
    """
    if loc is None:
        return None
    transform = resolve_numpyro_transform(model_config, transform_key)
    if method == "transform" or scale is None:
        return transform(loc)
    # Detect non-finite scales (frozen / unprofiled cascade subsets).
    finite = jnp.isfinite(scale) & (scale > 0)
    # Under method='jacobian' (strict mode), refuse to silently blend
    # corrected entries with transform(loc) fallback. The strict
    # contract is "raise on any unsupported configuration" — a NaN
    # scale is unsupported because there's no curvature information
    # to drive the correction. Promote to NotImplementedError so the
    # caller can decide whether to:
    #   1. Re-run with method='auto' to accept the elementwise blend.
    #   2. Re-run with method='transform' to opt out entirely.
    #   3. Plumb in a real scale (e.g., add curvature for the frozen
    #      subset).
    # The is_concrete guard keeps this jit/vmap-safe: under jit, the
    # finite-mask is a tracer and we can't branch on it; we trust the
    # autograd path to produce the right answer with the masked
    # scale_safe values.
    if method == "jacobian" and _is_concrete_helper(finite):
        if not bool(jnp.all(finite)):
            raise NotImplementedError(
                f"map_method='jacobian' refuses non-finite scales for "
                f"{transform_key!r}; some entries are NaN or non-positive. "
                f"Use map_method='auto' to blend corrected entries with "
                f"transform(loc) fallback for those elements, or "
                f"map_method='transform' to opt out of correction entirely."
            )
    scale_safe = jnp.where(finite, scale, 1.0)
    # Suppress potential NotImplementedError fallback warnings for
    # repeated calls — duplicates are deduped by Python's default
    # warning filter (action="default") via stacklevel=2 inside
    # jacobian_corrected_map.
    corrected = jacobian_corrected_map(
        transform,
        _dist.Normal(loc, scale_safe),
        method=method,
    )
    fallback = transform(loc)
    return jnp.where(finite, corrected, fallback)


def _is_concrete_helper(x) -> bool:
    """Tracer guard for the strict-mode NaN-scale check.

    Wraps the same logic as ``scribe.stats.jacobian_map._is_concrete``
    but lives here to avoid pulling it across the public-API boundary.
    Returns ``True`` for concrete values (Python scalars, eager JAX
    arrays); ``False`` for ``Tracer`` instances (under jit/vmap).
    """
    import jax.core

    return not isinstance(x, jax.core.Tracer)


def _resolve_effective_method(
    stored: Optional[str], call_arg: Optional[str]
) -> str:
    """Resolve the effective ``map_method`` from the dataclass attribute
    and the per-call argument.

    Precedence: explicit ``call_arg`` > stored attribute > ``"auto"``.
    """
    if call_arg is not None:
        return call_arg
    if stored:
        return stored
    return "auto"


# =====================================================================
# Frozen-parameter distribution helpers
# =====================================================================
#
# When a cascade-freeze is active, the NBLN result's r_scale / mu_scale
# fields are NaN sentinels (the post-fit profiled Hessian computation
# was skipped for frozen keys).  Authoritative posterior information
# lives on the embedded `cascade_source` SVI results object.  These
# helpers draw samples from the cascade source and moment-match in
# NBLN's target coordinate to construct distribution objects for
# `get_distributions()`.  PPC sampling uses the raw samples directly
# (full SVI fidelity) — see `_sampling.py`.


def _resolve_cascade_counts(
    cascade,
    cascade_counts: Optional[jnp.ndarray],
) -> Optional[jnp.ndarray]:
    """Resolve the counts to pass to amortized SVI sampling.

    Priority: the dedicated cache field on the Laplace result
    (``cascade_source_counts``), falling back to the SVI source's own
    ``_original_counts`` field.  Returns ``None`` when the SVI source
    is non-amortized and doesn't need counts.
    """
    if cascade_counts is not None:
        return cascade_counts
    return getattr(cascade, "_original_counts", None)


def _nbln_frozen_distributions(
    result,
    frozen: frozenset,
    cascade,
    cascade_counts: Optional[jnp.ndarray],
    n_samples: int = 1000,
) -> Dict[str, Any]:
    """Build moment-matched Distributions for frozen NBLN parameters.

    Routing per parameter:

    - **r**: SVI samples are positive; transform to NBLN unconstrained
      space via ``positive_transform`` inverse, fit per-gene Normal,
      wrap unconstrained as ``Normal.to_event(1)`` and constrained
      as ``TransformedDistribution(Normal, positive_transform)``.
    - **mu**: SVI samples are positive NB means; transform via
      ``jnp.log`` (NBLN ``mu`` is real-valued log-rate, no positive
      transform applies); return ``Normal.to_event(1)``.
    - **eta**: SVI samples are already in ``[0, ∞)`` (NBLN target
      coord); return ``TruncatedNormal(loc, scale, low=0.0)`` to keep
      support consistent with NBLN's existing TruncatedNormal η prior.
    """
    import numpyro.distributions as dist
    from ..models.config import ModelConfig  # noqa: F401  (typing only)

    # Resolve transforms once.
    target_pos_transform = resolve_numpyro_transform(result.model_config)
    pos_fwd, pos_inv = resolve_positive_fns(result.model_config)

    # Draw SVI samples (amortized branch handles counts internally).
    counts = _resolve_cascade_counts(cascade, cascade_counts)
    sample_kwargs = {"n_samples": int(n_samples), "store_samples": False}
    if counts is not None:
        sample_kwargs["counts"] = counts
    svi_samples = cascade.get_posterior_samples(**sample_kwargs)

    out: Dict[str, Any] = {}

    # Subset-aware routing.  When the Laplace target's gene panel is a
    # STRICT subset of the SVI source's panel, the cascade samples live
    # on the source's larger gene axis and the source's ``"_other"``
    # column represents a DIFFERENT aggregate from the Laplace target's
    # ``"_other"``.  Apply the same per-sample NB moment-matching
    # aggregator used by ``priors_from_results`` so the moment-matched
    # ``Normal(loc, scale)`` distribution is in the target gene axis.
    # See ``paper/_nb_lognormal.qmd`` §sec-nbln-cascade-aggregation.
    _cascade_subset_info = getattr(result, "_cascade_subset_info", None)
    _subset_active = (
        _cascade_subset_info is not None
        and _cascade_subset_info.is_subset
        and not _cascade_subset_info.is_equal
    )
    if _subset_active:
        # In subset mode the aggregator couples r and mu, so we require
        # both keys whenever either is frozen.  Silently falling back
        # to source-shaped samples would produce wrong-shape result
        # distributions; raise immediately instead.
        if (("r" in frozen or "mu" in frozen)
                and ("r" not in svi_samples or "mu" not in svi_samples)):
            raise ValueError(
                "Subset-aware frozen-distribution moment-match requires "
                "the SVI source to expose both 'r' and 'mu' per sample; "
                f"got keys {sorted(svi_samples.keys())}."
            )
        from .priors import (
            _aggregate_other_nb,
            _assemble_per_gene_subset_samples,
        )
        _r_src = jnp.asarray(svi_samples["r"])
        _mu_src = jnp.asarray(svi_samples["mu"])
        _r_kept = _assemble_per_gene_subset_samples(
            _r_src, _cascade_subset_info.kept_idx_in_source
        )
        _mu_kept = _assemble_per_gene_subset_samples(
            _mu_src, _cascade_subset_info.kept_idx_in_source
        )
        _r_other_s, _mu_other_s = _aggregate_other_nb(
            _r_src,
            _mu_src,
            _cascade_subset_info.dropped_idx_in_source,
            _cascade_subset_info.source_other_index_in_source,
        )
        # Splice into target-axis arrays the rest of the function consumes.
        svi_samples = dict(svi_samples)
        svi_samples["r"] = jnp.concatenate(
            [_r_kept, _r_other_s[:, None]], axis=1
        )
        svi_samples["mu"] = jnp.concatenate(
            [_mu_kept, _mu_other_s[:, None]], axis=1
        )

    if "r" in frozen and "r" in svi_samples:
        r_pos = jnp.asarray(svi_samples["r"])  # (S, G_target)
        r_uncon = pos_inv(jnp.maximum(r_pos, 1e-8))
        r_loc_mm = jnp.mean(r_uncon, axis=0)
        r_scale_mm = jnp.std(r_uncon, axis=0, ddof=1)
        out["r_unconstrained"] = dist.Normal(
            r_loc_mm, r_scale_mm
        ).to_event(1)
        out["r"] = dist.TransformedDistribution(
            dist.Normal(r_loc_mm, r_scale_mm).to_event(1),
            target_pos_transform,
        )

    if "mu" in frozen and "mu" in svi_samples:
        mu_pos = jnp.asarray(svi_samples["mu"])  # (S, G_target)
        mu_log = jnp.log(jnp.maximum(mu_pos, 1e-8))
        mu_loc_mm = jnp.mean(mu_log, axis=0)
        mu_scale_mm = jnp.std(mu_log, axis=0, ddof=1)
        # NBLN mu is real-valued log-rate; no positive_transform.
        out["mu"] = dist.Normal(mu_loc_mm, mu_scale_mm).to_event(1)

    if "eta" in frozen and "eta_capture" in svi_samples:
        eta_samples = jnp.asarray(svi_samples["eta_capture"])  # (S, N)
        eta_loc_mm = jnp.mean(eta_samples, axis=0)
        eta_scale_mm = jnp.std(eta_samples, axis=0, ddof=1)
        # Support [0, ∞): TruncatedNormal matches NBLN's existing η prior.
        out["eta_capture"] = dist.TruncatedNormal(
            eta_loc_mm, eta_scale_mm, low=0.0
        )
        # p_capture = exp(-eta) is bounded in (0, 1]; expose as a Delta
        # at the MAP for backward compatibility (the cascade's
        # posterior on p_capture is implicit in the eta TruncatedNormal).
        if result.eta_loc is not None:
            out["p_capture"] = dist.Delta(jnp.exp(-result.eta_loc))

    return out


class DispatchResultsMixin:
    """Mixin implementing model-branching public accessors."""

    @property
    def params(self) -> Dict[str, jnp.ndarray]:
        """Fitted-globals + per-cell-MAP dictionary.

        Mirrors :attr:`scribe.svi.results.ScribeSVIResults.params` so
        Laplace and SVI results share a common name for "the dict of
        fitted values keyed by site name".  Internally delegates to
        :meth:`get_map`, so the keys are model-specific (e.g.
        ``{"mu", "W", "d_pln", "y_log_rate"}`` for PLN;
        ``{..., "d_nbln", "r", "eta_capture", "p_capture"}`` for NBLN
        with capture).  See :meth:`get_map` for the full per-model
        key listing.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary of fitted MAP values, keyed by site name.
        """
        return self.get_map()

    def get_latent_embeddings(self) -> jnp.ndarray:
        """Return per-cell latent embeddings for downstream visualization.

        Returns
        -------
        jnp.ndarray
            Per-cell latent array suitable for dimension reduction or
            clustering:

            - PLN: ``x_loc`` with shape ``(n_cells, G)``.
            - LNM/LNMVCP with low-rank latent: ``z_loc`` with shape
              ``(n_cells, k)``.
            - LNM/LNMVCP with learned diagonal latent: ``y_alr_loc`` with
              shape ``(n_cells, G-1)``.

        Raises
        ------
        NotImplementedError
            If the stored ``base_model`` is unknown.
        """
        bm = _base_model(self.model_config)
        if bm in ("pln", "nbln", "twostate_ln_rate", "twostate_ln_logit"):
            # PLN-family base models: ``x_loc`` is the per-cell latent
            # MAP.  Semantics vary by variant — for PLN/NBLN/TSLN-Rate
            # it's the log-rate; for TSLN-Logit it's the additive shift
            # on the activation log-odds ``θ_g + x_g`` — but in every
            # case ``x_loc`` is the right field for "the latent embedding"
            # downstream consumers want.
            return self.x_loc
        if bm in ("lnm", "lnmvcp"):
            if self.z_loc is not None:
                return self.z_loc
            return self.y_alr_loc
        raise NotImplementedError(
            f"get_latent_embeddings not implemented for base_model={bm!r}"
        )

    def get_map(
        self,
        *,
        map_method: Optional[str] = None,
        **_kwargs,
    ) -> Dict[str, jnp.ndarray]:
        """Return point-estimate dictionary for diagnostics and plotting.

        Exposes both unconstrained posterior parameterization
        (``*_loc``, ``*_scale``) and constrained MAPs derived via
        ``model_config.positive_transform``.

        Parameters
        ----------
        map_method : str, optional
            Controls Jacobian correction for the constrained positive
            globals (``r``, ``gene_mean``, ``burst_size``, ``k_off``,
            ``rate``, ``kappa``, ``mu_T``, ``r_T``). Default ``None``
            inherits the result's stored ``self.map_method`` attribute
            (which defaults to ``"auto"``).

            - ``"auto"`` (default): closed-form / grid+Newton /
              autodiff correction, with graceful fallback to
              ``f(loc)`` (with warning) when unsupported.
            - ``"transform"``: legacy uncorrected median ``f(loc)``.
              Use to reproduce pre-correction results byte-for-byte.
            - ``"jacobian"``: like ``"auto"`` but raises on unsupported
              ``(transform, base)`` pairs.

            Note: this kwarg is for the **MAP values returned by this
            call**. To make the correction persist into the stored
            constrained fields (``self.r``, ``self.gene_mean``, etc.),
            use :meth:`with_jacobian_map`. See
            :func:`scribe.stats.jacobian_corrected_map` for the math.
        **_kwargs
            Accepted for interface compatibility with other inference modes.
            Unknown kwargs are ignored.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Model-specific mapping of semantic parameter names to arrays.

            - PLN: ``mu, W, d_pln, y_log_rate`` (+ ``eta_capture,
              p_capture`` when capture anchor on).
            - NBLN: same as PLN plus ``r, r_loc, r_scale`` (+ capture).
            - LNM/LNMVCP: ``mu, W, d_lnm``, composition latent,
              ``mu_T, r_T, p, mu_T_loc, mu_T_scale, r_T_loc, r_T_scale``
              (+ ``p_capture`` for LNMVCP).

        Raises
        ------
        NotImplementedError
            If the stored ``base_model`` is unknown, or if
            ``map_method='jacobian'`` is requested for an unsupported
            ``(transform, base)`` pair.
        """
        effective = _resolve_effective_method(
            getattr(self, "map_method", None), map_method
        )
        bm = _base_model(self.model_config)
        if bm in ("pln", "nbln"):
            d_key = "d_nbln" if bm == "nbln" else "d_pln"
            # Commit 2b: under decoupled NBLN, ``self.x_loc`` is the
            # kept-axis deviation ``x_dev`` (shape ``(N, G_kept)``).
            # ``y_log_rate`` is conceptually the full per-gene log-rate
            # at the MAP, so reconstruct it on G_obs:
            #     y_log_rate[c, kept[k]] = μ[kept[k]] + x_dev[c, k]
            #     y_log_rate[c, other_idx] = μ[other_idx]
            # Under legacy / trivial layout, ``x_loc`` is already the
            # full G_obs log-rate.
            _layout = getattr(self, "axis_layout", None)
            _is_decoupled = _layout is not None and _layout.decoupled
            if _is_decoupled and self.x_loc is not None:
                _kept_idx = jnp.asarray(_layout.kept_idx)
                _n_cells = int(self.x_loc.shape[0])
                _G_obs = int(self.mu.shape[0])
                _y_log_rate_full = jnp.broadcast_to(
                    self.mu[None, :], (_n_cells, _G_obs)
                )
                _y_log_rate_full = _y_log_rate_full.at[:, _kept_idx].add(
                    self.x_loc
                )
            else:
                _y_log_rate_full = self.x_loc
            out = {
                "mu": self.mu,
                "W": self.W,
                d_key: self.d,
                "y_log_rate": _y_log_rate_full,
            }
            if self.eta_loc is not None:
                out["eta_capture"] = self.eta_loc
                # p_capture = exp(-eta_loc). Jacobian correction for
                # this would be exp(-eta_loc - eta_scale**2), but
                # eta_scale is not currently persisted on
                # ScribeLaplaceResults (v1 limitation — tracked under
                # TODO(jacobian-map-eta-scale)). Fall back to the
                # uncorrected exp(-eta_loc) under "auto" with a one-
                # per-call-site warning; raise under "jacobian".
                if effective == "jacobian":
                    raise NotImplementedError(
                        "p_capture Jacobian correction requires "
                        "ScribeLaplaceResults.eta_scale (not persisted "
                        "in v1). Use map_method='auto' to fall back "
                        "with a warning, or map_method='transform' to "
                        "opt out explicitly."
                    )
                elif effective != "transform":
                    warnings.warn(
                        "Jacobian correction for p_capture requires "
                        "ScribeLaplaceResults.eta_scale, which is not "
                        "currently persisted. Returning exp(-eta_loc); "
                        "p_capture will be corrected once eta_scale is "
                        "added (TODO(jacobian-map-eta-scale)).",
                        stacklevel=2,
                    )
                out["p_capture"] = jnp.exp(-self.eta_loc)
            if bm == "nbln":
                # Correct ``r`` from (r_loc, r_scale) if method != "transform".
                # When the result was built with map_method="transform"
                # (or no scale is available), this is byte-equal to the
                # stored self.r.
                if self.r_loc is not None and self.r is not None:
                    out["r"] = _correct_positive(
                        self.model_config,
                        transform_key="r",
                        loc=self.r_loc,
                        scale=getattr(self, "r_scale", None),
                        method=effective,
                    )
                elif self.r is not None:
                    out["r"] = self.r
            # NBLN global posterior parameters in unconstrained space.
            if bm == "nbln" and self.r_loc is not None:
                out["r_loc"] = self.r_loc
            if bm == "nbln" and self.r_scale is not None:
                out["r_scale"] = self.r_scale
            # NBLN ``mu`` posterior (log-rate coordinate; mu IS the loc).
            if bm == "nbln" and self.mu_loc is not None:
                out["mu_loc"] = self.mu_loc
            if bm == "nbln" and self.mu_scale is not None:
                out["mu_scale"] = self.mu_scale
            # Phase-2 cascade-freeze flags so callers can identify
            # cascade-bound parameters.  Always present for NBLN; values
            # come from the frozen_params frozenset on the result.
            if bm == "nbln":
                frozen = getattr(self, "frozen_params", frozenset())
                out["r_frozen"] = "r" in frozen
                out["mu_frozen"] = "mu" in frozen
                out["eta_frozen"] = "eta" in frozen
            # Phase-3: surface W-prior diagnostics under a single key.
            # Callers can read either the field directly
            # (``self.w_prior_diagnostics``) or via ``get_map()``;
            # both routes return the same dict object.
            wpd = getattr(self, "w_prior_diagnostics", None)
            if wpd is not None:
                out["w_prior_diagnostics"] = wpd
            return out

        if bm == "twostate_ln_rate":
            # TSLN-Rate get_map: surface both the latent log-rate prior
            # center (``self.mu`` = log(r_hat); matches NBLN convention)
            # AND the user-facing positive TwoState gene mean
            # (``self.gene_mean``).  The dual exposure prevents
            # downstream confusion: math-side consumers (PPC,
            # distribution accessors) get the latent log-rate; biology-
            # side consumers (cascade-MAP comparison, plotting) get the
            # positive gene mean.
            #
            # Commit 3b: under decoupled TSLN-Rate, ``self.x_loc`` is
            # ``x_dev`` on G_kept.  Reconstruct full G_obs per-cell
            # log-rate as ``μ + x_dev`` (kept) / ``μ`` (other) for
            # the ``y_log_rate`` field — mirrors the NBLN fix in
            # rev-11.
            _layout = getattr(self, "axis_layout", None)
            _is_decoupled_ts = (
                _layout is not None and _layout.decoupled
            )
            if _is_decoupled_ts and self.x_loc is not None:
                _kept_idx = jnp.asarray(_layout.kept_idx)
                _n_cells = int(self.x_loc.shape[0])
                _G_obs = int(self.mu.shape[0])
                _y_log_rate_full = jnp.broadcast_to(
                    self.mu[None, :], (_n_cells, _G_obs)
                )
                _y_log_rate_full = _y_log_rate_full.at[:, _kept_idx].add(
                    self.x_loc
                )
            else:
                _y_log_rate_full = self.x_loc
            out = {
                # ``mu`` here is the LATENT log-rate prior center
                # (= log(r_hat)) to match NBLN/PLN convention.  This
                # is what ``y_log_rate``'s distribution loc is.
                "mu": self.mu,
                "W": self.W,
                "d_tsln": self.d,
                "y_log_rate": _y_log_rate_full,
            }
            # Correct each positive parent from its (_loc, _scale) pair.
            # IMPORTANT: gene_mean is the positive form of the "mu"
            # latent (model_config.positive_transform[mu]), so the
            # transform_key for gene_mean is "mu", NOT "gene_mean".
            # Mixing these up would ignore user-configured per-param
            # transforms like positive_transform={"mu": "exp"}.
            gene_mean_corrected = _correct_positive(
                self.model_config,
                transform_key="mu",
                loc=getattr(self, "gene_mean_loc", None),
                scale=getattr(self, "gene_mean_scale", None),
                method=effective,
            )
            burst_size_corrected = _correct_positive(
                self.model_config,
                transform_key="burst_size",
                loc=getattr(self, "burst_size_loc", None),
                scale=getattr(self, "burst_size_scale", None),
                method=effective,
            )
            k_off_corrected = _correct_positive(
                self.model_config,
                transform_key="k_off",
                loc=getattr(self, "k_off_loc", None),
                scale=getattr(self, "k_off_scale", None),
                method=effective,
            )
            # Fall back to stored fields when no _loc is available
            # (e.g., frozen genes with only `gene_mean` populated).
            out["gene_mean"] = (
                gene_mean_corrected if gene_mean_corrected is not None
                else self.gene_mean
            )
            if burst_size_corrected is not None:
                out["burst_size"] = burst_size_corrected
            elif self.burst_size is not None:
                out["burst_size"] = self.burst_size
            if k_off_corrected is not None:
                out["k_off"] = k_off_corrected
            elif self.k_off is not None:
                out["k_off"] = self.k_off
            if self.eta_loc is not None:
                out["eta_capture"] = self.eta_loc
                # See p_capture block in pln/nbln branch above for the
                # eta_scale-not-persisted limitation.
                if effective == "jacobian":
                    raise NotImplementedError(
                        "p_capture Jacobian correction requires "
                        "ScribeLaplaceResults.eta_scale (not persisted "
                        "in v1). Use map_method='auto' or 'transform'."
                    )
                elif effective != "transform":
                    warnings.warn(
                        "Jacobian correction for p_capture requires "
                        "ScribeLaplaceResults.eta_scale, which is not "
                        "currently persisted (TSLN-Rate). Returning "
                        "exp(-eta_loc).",
                        stacklevel=2,
                    )
                out["p_capture"] = jnp.exp(-self.eta_loc)
            # Derived TSLN quantities: when method != "transform",
            # recompute from corrected parents via _derived.py so the
            # dict is internally consistent
            # (alpha == _twostate_reparam(gene_mean, burst_size, k_off).alpha).
            # When method == "transform", use the stored derived
            # values (byte-equal to pre-change behavior).
            if effective == "transform":
                if self.alpha is not None:
                    out["alpha"] = self.alpha
                if self.beta is not None:
                    out["beta"] = self.beta
                if self.r_hat is not None:
                    out["r_hat"] = self.r_hat
            else:
                # Re-derive from the corrected parents we just computed.
                # Only safe when all three parents are available.
                if (
                    out.get("gene_mean") is not None
                    and out.get("burst_size") is not None
                    and out.get("k_off") is not None
                ):
                    derived = twostate_rate_derived_from_parents(
                        gene_mean=out["gene_mean"],
                        burst_size=out["burst_size"],
                        k_off=out["k_off"],
                    )
                    out["alpha"] = derived["alpha"]
                    out["beta"] = derived["beta"]
                    out["r_hat"] = derived["r_hat"]
                    # CRITICAL: TSLN-Rate convention requires
                    # ``out["mu"] == log(out["r_hat"])`` (mu carries the
                    # latent log-rate prior center). When we correct
                    # r_hat from corrected parents, we MUST also update
                    # mu to maintain the invariant — otherwise downstream
                    # consumers that rely on mu = log(r_hat) (e.g., the
                    # public-API conformance test, or any code that
                    # reconstructs the latent log-rate distribution
                    # from mu) would silently desynchronize.
                    out["mu"] = jnp.log(jnp.maximum(derived["r_hat"], 1e-30))
                else:
                    # Partial population; preserve stored values.
                    if self.alpha is not None:
                        out["alpha"] = self.alpha
                    if self.beta is not None:
                        out["beta"] = self.beta
                    if self.r_hat is not None:
                        out["r_hat"] = self.r_hat
            # Unconstrained loc/scale fields (when populated by
            # compute_global_uncertainty).  NB: for TSLN-Rate, the
            # gene-mean posterior lives on ``gene_mean_loc / _scale``;
            # ``mu_loc / mu_scale`` are NBLN-specific and not populated
            # for TSLN-Rate.
            for key in (
                "gene_mean_loc", "gene_mean_scale",
                "burst_size_loc", "burst_size_scale",
                "k_off_loc", "k_off_scale",
            ):
                val = getattr(self, key, None)
                if val is not None:
                    out[key] = val
            # Clamp diagnostics.
            for key in (
                "a_raw_min", "a_raw_negative_fraction",
                "a_clamp_fraction", "a_clamp_per_gene",
            ):
                val = getattr(self, key, None)
                if val is not None:
                    out[key] = val
            # Cascade freeze flags.
            frozen = getattr(self, "frozen_params", frozenset())
            out["mu_frozen"] = "mu" in frozen
            out["burst_size_frozen"] = "burst_size" in frozen
            out["k_off_frozen"] = "k_off" in frozen
            out["eta_frozen"] = "eta" in frozen
            wpd = getattr(self, "w_prior_diagnostics", None)
            if wpd is not None:
                out["w_prior_diagnostics"] = wpd
            return out

        if bm == "twostate_ln_logit":
            # TSLN-Logit (Variant B) get_map: surface the sampled
            # gene-level globals (rate, kappa, eta_anchor) and the
            # derived reporting quantities (alpha, beta, gene_mean at
            # z=0).  ``self.mu`` is zeros for TSLN-Logit (the latent
            # prior center); ``self.eta_anchor`` is the per-gene
            # activation log-odds θ_g.
            #
            # Commit 4b: under decoupled TSLN-Logit, ``self.x_loc`` is
            # ``z_kept`` on G_kept.  Reconstruct full G_obs per-cell
            # latent as ``z_kept`` scattered at kept positions and
            # ``0`` at ``_other`` — matching the math contract that
            # ``_other``'s activation log-odds has no z modulation
            # (``η_act_other = θ_other``).
            _layout_ts = getattr(self, "axis_layout", None)
            _is_decoupled_ts = (
                _layout_ts is not None and _layout_ts.decoupled
            )
            if _is_decoupled_ts and self.x_loc is not None:
                _kept_idx = jnp.asarray(_layout_ts.kept_idx)
                _n_cells = int(self.x_loc.shape[0])
                _G_obs = int(self.eta_anchor.shape[0])
                _y_latent_full = jnp.zeros(
                    (_n_cells, _G_obs), dtype=self.x_loc.dtype,
                )
                _y_latent_full = _y_latent_full.at[:, _kept_idx].add(
                    self.x_loc
                )
            else:
                _y_latent_full = self.x_loc
            out = {
                # ``mu`` here is the LATENT prior center (zeros for
                # TSLN-Logit — the latent z ~ N(0, Σ) prior).
                "mu": self.mu,
                "W": self.W,
                "d_tsln": self.d,
                "y_latent": _y_latent_full,
            }
            # Correct the positive parents (rate, kappa). eta_anchor is
            # a real-valued parameter (not positive) so no Jacobian
            # correction applies — it's returned as-is (=eta_anchor_loc
            # under the Laplace identity transform).
            rate_corrected = _correct_positive(
                self.model_config,
                transform_key="rate",
                loc=getattr(self, "rate_loc", None),
                scale=getattr(self, "rate_scale", None),
                method=effective,
            )
            kappa_corrected = _correct_positive(
                self.model_config,
                transform_key="kappa",
                loc=getattr(self, "kappa_loc", None),
                scale=getattr(self, "kappa_scale", None),
                method=effective,
            )
            if rate_corrected is not None:
                out["rate"] = rate_corrected
            elif self.rate is not None:
                out["rate"] = self.rate
            if kappa_corrected is not None:
                out["kappa"] = kappa_corrected
            elif self.kappa is not None:
                out["kappa"] = self.kappa
            if self.eta_anchor is not None:
                out["eta_anchor"] = self.eta_anchor
            # Derived (alpha, beta) at z=0 from (kappa, eta_anchor).
            # When method != "transform", recompute from corrected
            # parents; when method == "transform", use stored.
            if effective == "transform":
                if self.gene_mean is not None:
                    out["gene_mean"] = self.gene_mean
                if self.alpha is not None:
                    out["alpha"] = self.alpha
                if self.beta is not None:
                    out["beta"] = self.beta
            else:
                if (
                    out.get("kappa") is not None
                    and self.eta_anchor is not None
                ):
                    derived = twostate_logit_derived_from_parents(
                        kappa=out["kappa"],
                        eta_anchor=self.eta_anchor,
                    )
                    out["alpha"] = derived["alpha"]
                    out["beta"] = derived["beta"]
                # gene_mean is a separate reporting quantity for
                # TSLN-Logit (not derivable from kappa/eta_anchor
                # alone — depends on the sampled rate too); preserve
                # the stored value.
                if self.gene_mean is not None:
                    out["gene_mean"] = self.gene_mean
            if self.eta_loc is not None:
                # Fixed-offset capture path: surface the same
                # eta_capture / p_capture pair as TSLN-Rate / NBLN so
                # the capture-related viz plots transfer.
                out["eta_capture"] = self.eta_loc
                if effective == "jacobian":
                    raise NotImplementedError(
                        "p_capture Jacobian correction requires "
                        "ScribeLaplaceResults.eta_scale (not persisted "
                        "in v1)."
                    )
                elif effective != "transform":
                    warnings.warn(
                        "Jacobian correction for p_capture requires "
                        "eta_scale (TSLN-Logit). Returning exp(-eta_loc).",
                        stacklevel=2,
                    )
                out["p_capture"] = jnp.exp(-self.eta_loc)
            # Unconstrained loc/scale fields (when populated by
            # compute_global_uncertainty).  Default L4 cascade freezes
            # all three gene globals → scales are NaN.
            for key in (
                "rate_loc", "rate_scale",
                "kappa_loc", "kappa_scale",
                "eta_anchor_loc", "eta_anchor_scale",
            ):
                val = getattr(self, key, None)
                if val is not None:
                    out[key] = val
            # Clamp diagnostics.
            for key in (
                "a_raw_min", "a_raw_negative_fraction",
                "a_clamp_fraction", "a_clamp_per_gene",
            ):
                val = getattr(self, key, None)
                if val is not None:
                    out[key] = val
            # Cascade freeze flags — TSLN-Logit's gene-global set.
            frozen = getattr(self, "frozen_params", frozenset())
            out["rate_frozen"] = "rate" in frozen
            out["kappa_frozen"] = "kappa" in frozen
            out["eta_anchor_frozen"] = "eta_anchor" in frozen
            out["eta_frozen"] = "eta" in frozen
            wpd = getattr(self, "w_prior_diagnostics", None)
            if wpd is not None:
                out["w_prior_diagnostics"] = wpd
            return out

        if bm in ("lnm", "lnmvcp"):
            pos_fwd, _ = resolve_positive_fns(self.model_config)
            out = {
                "mu": self.mu,
                "W": self.W,
                "d_lnm": self.d,
            }
            if self.z_loc is not None:
                out["z"] = self.z_loc
            if self.y_alr_loc is not None:
                out["y_alr"] = self.y_alr_loc
            if self.p_capture_loc is not None:
                out["p_capture"] = self.p_capture_loc
            # Correct positive totals (mu_T, r_T).
            mu_T_corrected = _correct_positive(
                self.model_config,
                transform_key="mu_T",
                loc=self.mu_T_loc,
                scale=self.mu_T_scale,
                method=effective,
            )
            r_T_corrected = _correct_positive(
                self.model_config,
                transform_key="r_T",
                loc=self.r_T_loc,
                scale=self.r_T_scale,
                method=effective,
            )
            if mu_T_corrected is not None:
                out["mu_T"] = mu_T_corrected
            elif self.mu_T is not None:
                out["mu_T"] = self.mu_T
            if r_T_corrected is not None:
                out["r_T"] = r_T_corrected
            elif self.r_T is not None:
                out["r_T"] = self.r_T
            # Derived success probability — re-derive from the
            # corrected parents we just placed in ``out``. Uses the NB
            # success-prob convention ``p = r_T / (r_T + mu_T)``, NOT
            # ``mu_T / (mu_T + r_T)``. Verified against
            # tests/test_lnm_laplace.py:417-421.
            if out.get("mu_T") is not None and out.get("r_T") is not None:
                out["p"] = lnm_p_from_parents(
                    mu_T=out["mu_T"], r_T=out["r_T"]
                )
            # Unconstrained posterior parameterization (always exposed
            # as-is for downstream consumers that need the raw _loc /
            # _scale pair).
            if self.mu_T_loc is not None:
                out["mu_T_loc"] = self.mu_T_loc
            if self.mu_T_scale is not None:
                out["mu_T_scale"] = self.mu_T_scale
            if self.r_T_loc is not None:
                out["r_T_loc"] = self.r_T_loc
            if self.r_T_scale is not None:
                out["r_T_scale"] = self.r_T_scale
            return out

        raise NotImplementedError(f"get_map not implemented for base_model={bm!r}")

    def get_distributions(
        self, backend: str = "numpyro", **_kwargs
    ) -> Dict[str, Any]:
        """Return population-level distributions and fitted-global values.

        Parameters
        ----------
        backend : {"numpyro"}, default="numpyro"
            Distribution backend. Laplace currently supports only NumPyro
            distribution objects.
        **_kwargs
            Reserved for compatibility with other result classes.

        Returns
        -------
        Dict[str, Any]
            Per-site dictionary.  Continuous population-level latents are
            returned as proper NumPyro ``Distribution`` objects.

            - PLN: ``y_log_rate``, ``lambda_rate`` (always);
              ``eta_capture``, ``p_capture`` (when capture anchor on).
            - NBLN: ``y_log_rate``, ``r_unconstrained`` (Normal),
              ``r`` (transformed positive distribution) (always);
              ``eta_capture``, ``p_capture`` (when capture anchor on).
            - LNM/LNMVCP: ``y_alr``, ``totals_unconstrained``
              (MultivariateNormal), ``mu_T`` and ``r_T`` (transformed
              marginals); ``p_capture`` for LNMVCP.

            Global parameters that have a Laplace posterior approximation
            are returned as proper Normal (or MultivariateNormal)
            distributions rather than Delta.  Constrained-space
            distributions are TransformedDistribution objects using
            the configured positive transform.

        Raises
        ------
        ValueError
            If ``backend`` is unsupported.
        NotImplementedError
            If the stored ``base_model`` is unknown.
        """
        if backend != "numpyro":
            raise ValueError("Only 'numpyro' backend supported for Laplace results.")
        import numpyro.distributions as dist

        bm = _base_model(self.model_config)
        if bm in ("pln", "nbln"):
            # Commit 2b: under decoupled NBLN, ``W`` / ``d`` live on
            # G_kept while ``mu`` lives on G_obs.  Pad ``W`` with zero
            # rows at ``other_idx`` (no z modulation) and ``d`` with a
            # tiny epsilon at ``other_idx`` (LowRankMultivariateNormal
            # requires positive diagonal entries).  The resulting
            # distribution is effectively singular at ``_other``
            # (variance ≈ 0), matching the math contract: ``_other``'s
            # log-rate is deterministic at ``μ_other``.
            _layout = getattr(self, "axis_layout", None)
            _is_decoupled = _layout is not None and _layout.decoupled
            if _is_decoupled:
                _G_obs = int(self.mu.shape[0])
                _K = int(self.W.shape[1])
                _kept_idx = jnp.asarray(_layout.kept_idx)
                _W_padded = jnp.zeros((_G_obs, _K), dtype=self.W.dtype)
                _W_padded = _W_padded.at[_kept_idx].set(self.W)
                # Tiny positive epsilon so the MVN diagonal is valid;
                # callers should treat ``_other``'s variance as 0.
                _d_padded = jnp.full(
                    (_G_obs,), 1e-12, dtype=self.d.dtype
                )
                _d_padded = _d_padded.at[_kept_idx].set(self.d)
                _W_for_dist = _W_padded
                _d_for_dist = _d_padded
            else:
                _W_for_dist = self.W
                _d_for_dist = self.d
            out: Dict[str, Any] = {
                "y_log_rate": dist.LowRankMultivariateNormal(
                    loc=self.mu,
                    cov_factor=_W_for_dist,
                    cov_diag=_d_for_dist,
                ),
            }
            if bm == "pln":
                out["lambda_rate"] = LowRankPoissonLogNormal(
                    loc=self.mu,
                    cov_factor=_W_for_dist,
                    cov_diag=_d_for_dist,
                )
            if bm == "nbln":
                # Cascade-freeze: when a parameter is frozen, NBLN's
                # profiled-Hessian r_scale/mu_scale is the NaN sentinel
                # (the Hessian computation was skipped).  Route through
                # cascade_source's SVI posterior samples and moment-match
                # in NBLN target coord.
                frozen = getattr(self, "frozen_params", frozenset())
                cascade = getattr(self, "cascade_source", None)
                cascade_counts = getattr(
                    self, "cascade_source_counts", None
                )
                if frozen and cascade is not None:
                    out.update(
                        _nbln_frozen_distributions(
                            self, frozen, cascade, cascade_counts,
                        )
                    )

                # Non-frozen r path: post-fit Laplace Normal.
                if (
                    "r" not in frozen
                    and self.r_loc is not None
                    and self.r_scale is not None
                ):
                    # Unconstrained r posterior: independent Normal
                    # per gene.
                    out["r_unconstrained"] = dist.Normal(
                        self.r_loc, self.r_scale
                    ).to_event(1)
                    # Constrained r via the configured positive transform.
                    out["r"] = dist.TransformedDistribution(
                        dist.Normal(self.r_loc, self.r_scale).to_event(1),
                        resolve_numpyro_transform(self.model_config),
                    )
                elif "r" not in frozen and self.r is not None:
                    out["r"] = dist.Delta(self.r)

                # Non-frozen mu path.
                if (
                    "mu" not in frozen
                    and self.mu_loc is not None
                    and self.mu_scale is not None
                ):
                    # NBLN ``mu`` posterior: Normal in log-rate space.
                    # Unlike ``r``, ``mu`` is already real-valued in
                    # NBLN's coordinate system, so no transform applies.
                    out["mu"] = dist.Normal(
                        self.mu_loc, self.mu_scale
                    ).to_event(1)
                elif "mu" not in frozen and self.mu is not None:
                    out["mu"] = dist.Delta(self.mu)

            # Frozen eta is handled by _nbln_frozen_distributions above
            # (TruncatedNormal summary from cascade samples).  Non-frozen
            # eta retains the legacy Delta accessor.
            if self.eta_loc is not None and "eta" not in (
                getattr(self, "frozen_params", frozenset())
            ):
                out["eta_capture"] = dist.Delta(self.eta_loc)
                out["p_capture"] = dist.Delta(jnp.exp(-self.eta_loc))
            return out

        if bm == "twostate_ln_rate":
            # TSLN-Rate: latent log-rate posterior + Delta on the gene
            # globals. Soft-cascade Normal posteriors when populated by
            # compute_global_uncertainty.
            #
            # CONVENTION (round-5 audit fix): ``self.mu`` is the
            # **latent log-rate prior center** ``log(r_hat)`` for
            # TSLN-Rate (matching NBLN/PLN), set in the bridge via
            # ``common_kwargs["mu"] = jnp.log(r_hat)``.  The TwoState
            # positive gene-mean parameter lives on ``self.gene_mean``.
            # This keeps ``LowRankMultivariateNormal(loc=self.mu, ...)``
            # semantically correct across every Laplace base model.
            # Resolve a NumPyro Transform per positive parameter — the
            # auditor's Step 6 catch: when ``positive_transform`` is a
            # per-parameter dict (e.g. ``{"mu": "exp",
            # "burst_size": "softplus", ...}``), each TransformedDistribution
            # MUST use its own transform.  Sharing a single ``tfm`` here
            # would silently misreport the posterior shape under any
            # config that mixes ``"exp"`` and ``"softplus"`` across
            # positive globals.  ``resolve_numpyro_transform(cfg, name)``
            # honors the per-parameter dict via the model_config's
            # ``resolve_positive_transform`` helper.
            tfm_mu = resolve_numpyro_transform(self.model_config, "mu")
            tfm_bs = resolve_numpyro_transform(
                self.model_config, "burst_size"
            )
            tfm_ko = resolve_numpyro_transform(self.model_config, "k_off")
            # Commit 3b: under decoupled TSLN-Rate, ``W`` / ``d`` live
            # on G_kept while ``self.mu`` lives on G_obs.  Pad ``W``
            # with zero rows at ``other_idx`` and ``d`` with ``1e-12``
            # at ``other_idx`` so the LowRankMultivariateNormal has the
            # correct G_obs event shape and ``_other`` has effectively-
            # zero variance (deterministic at ``μ_other``).  Mirrors
            # the NBLN/PLN pattern in this same dispatch.
            _layout_ts = getattr(self, "axis_layout", None)
            _is_decoupled_ts_dist = (
                _layout_ts is not None and _layout_ts.decoupled
            )
            if _is_decoupled_ts_dist:
                _G_obs_ts = int(self.mu.shape[0])
                _K_ts = int(self.W.shape[1])
                _kept_idx_ts = jnp.asarray(_layout_ts.kept_idx)
                _W_padded_ts = jnp.zeros(
                    (_G_obs_ts, _K_ts), dtype=self.W.dtype
                )
                _W_padded_ts = _W_padded_ts.at[_kept_idx_ts].set(self.W)
                _d_padded_ts = jnp.full(
                    (_G_obs_ts,), 1e-12, dtype=self.d.dtype
                )
                _d_padded_ts = _d_padded_ts.at[_kept_idx_ts].set(self.d)
                _W_for_dist_ts = _W_padded_ts
                _d_for_dist_ts = _d_padded_ts
            else:
                _W_for_dist_ts = self.W
                _d_for_dist_ts = self.d
            out: Dict[str, Any] = {
                "y_log_rate": dist.LowRankMultivariateNormal(
                    loc=self.mu,
                    cov_factor=_W_for_dist_ts,
                    cov_diag=_d_for_dist_ts,
                ),
            }
            frozen = getattr(self, "frozen_params", frozenset())
            # ``gene_mean`` (positive TwoState mu) posterior.  The
            # TwoState positive gene-mean parameter is exposed ONLY
            # under the ``"gene_mean"`` key — NOT aliased to
            # ``"mu"`` — so the ``out["mu"]`` semantics stay consistent
            # with ``get_map()["mu"]`` (which is the latent log-rate
            # prior center ``log(r_hat)``).  Round-6 audit: the
            # previous draft aliased ``out["mu"] = out["gene_mean"]``
            # for back-compat, but TSLN-Rate is brand new and there's
            # no back-compat to maintain; consistency across the two
            # accessors is more important.
            if (
                "mu" not in frozen
                and self.gene_mean_loc is not None
                and self.gene_mean_scale is not None
            ):
                out["gene_mean"] = dist.TransformedDistribution(
                    dist.Normal(
                        self.gene_mean_loc, self.gene_mean_scale
                    ).to_event(1),
                    tfm_mu,
                )
            elif self.gene_mean is not None:
                out["gene_mean"] = dist.Delta(self.gene_mean)
            # ``mu`` is the latent log-rate prior center (= log(r_hat))
            # for TSLN-Rate.  No closed-form posterior is currently
            # populated for the derived ``mu`` (would require the
            # cross-Hessian between gene_mean, burst_size, and k_off;
            # phase-2 work).  Expose a Delta at the MAP — same shape
            # as ``get_map()["mu"]``.
            if self.mu is not None:
                out["mu"] = dist.Delta(self.mu)
            # burst_size posterior
            if (
                "burst_size" not in frozen
                and self.burst_size_loc is not None
                and self.burst_size_scale is not None
            ):
                out["burst_size"] = dist.TransformedDistribution(
                    dist.Normal(
                        self.burst_size_loc, self.burst_size_scale
                    ).to_event(1),
                    tfm_bs,
                )
            elif self.burst_size is not None:
                out["burst_size"] = dist.Delta(self.burst_size)
            # k_off posterior
            if (
                "k_off" not in frozen
                and self.k_off_loc is not None
                and self.k_off_scale is not None
            ):
                out["k_off"] = dist.TransformedDistribution(
                    dist.Normal(
                        self.k_off_loc, self.k_off_scale
                    ).to_event(1),
                    tfm_ko,
                )
            elif self.k_off is not None:
                out["k_off"] = dist.Delta(self.k_off)
            # eta_capture
            if self.eta_loc is not None and "eta" not in frozen:
                out["eta_capture"] = dist.Delta(self.eta_loc)
                out["p_capture"] = dist.Delta(jnp.exp(-self.eta_loc))
            return out

        if bm == "twostate_ln_logit":
            # TSLN-Logit (Variant B): latent ``y_latent`` is a
            # ``LowRankMultivariateNormal(loc=0, cov=WW^T + diag(d))``;
            # the gene baseline lives in ``eta_anchor`` (per-gene
            # activation log-odds θ_g) — distinct from TSLN-Rate where
            # the gene baseline is folded into ``self.mu = log(r_hat)``.
            #
            # ``self.mu`` for TSLN-Logit is zeros (latent prior center).
            #
            # Per-parameter transforms — see the TSLN-Rate dispatch arm
            # comment above for the auditor's Step 6 motivation.  ``rate``
            # and ``kappa`` may use DIFFERENT positive maps when the user
            # configures ``positive_transform={"rate": "exp", "kappa":
            # "softplus"}``; sharing one ``tfm`` would silently misreport
            # one of the two posteriors.  ``eta_anchor`` is real-valued
            # (identity) and does not need a TransformedDistribution.
            tfm_rate = resolve_numpyro_transform(self.model_config, "rate")
            tfm_kappa = resolve_numpyro_transform(self.model_config, "kappa")
            # Commit 4b: under decoupled TSLN-Logit, ``W`` / ``d`` live
            # on G_kept while ``self.mu`` lives on G_obs (zeros).  Pad
            # ``W`` with zero rows at ``other_idx`` and ``d`` with
            # ``1e-12`` at ``other_idx`` so the LowRankMultivariateNormal
            # has the correct G_obs event shape and ``_other`` has
            # effectively-zero variance.  Same pattern as NBLN/PLN/
            # TSLN-Rate get_distributions.
            _layout_ts_dist = getattr(self, "axis_layout", None)
            _is_decoupled_ts_dist = (
                _layout_ts_dist is not None
                and _layout_ts_dist.decoupled
            )
            if _is_decoupled_ts_dist:
                _G_obs_ts = int(self.mu.shape[0])
                _K_ts = int(self.W.shape[1])
                _kept_idx_ts = jnp.asarray(_layout_ts_dist.kept_idx)
                _W_padded_ts = jnp.zeros(
                    (_G_obs_ts, _K_ts), dtype=self.W.dtype
                )
                _W_padded_ts = _W_padded_ts.at[_kept_idx_ts].set(self.W)
                _d_padded_ts = jnp.full(
                    (_G_obs_ts,), 1e-12, dtype=self.d.dtype
                )
                _d_padded_ts = _d_padded_ts.at[_kept_idx_ts].set(self.d)
                _W_for_dist_ts = _W_padded_ts
                _d_for_dist_ts = _d_padded_ts
            else:
                _W_for_dist_ts = self.W
                _d_for_dist_ts = self.d
            out: Dict[str, Any] = {
                "y_latent": dist.LowRankMultivariateNormal(
                    loc=self.mu,
                    cov_factor=_W_for_dist_ts,
                    cov_diag=_d_for_dist_ts,
                ),
            }
            frozen = getattr(self, "frozen_params", frozenset())

            # ``rate`` (positive) posterior.
            if (
                "rate" not in frozen
                and self.rate_loc is not None
                and self.rate_scale is not None
            ):
                out["rate"] = dist.TransformedDistribution(
                    dist.Normal(self.rate_loc, self.rate_scale).to_event(1),
                    tfm_rate,
                )
            elif self.rate is not None:
                out["rate"] = dist.Delta(self.rate)

            # ``kappa`` (positive) posterior.
            if (
                "kappa" not in frozen
                and self.kappa_loc is not None
                and self.kappa_scale is not None
            ):
                out["kappa"] = dist.TransformedDistribution(
                    dist.Normal(self.kappa_loc, self.kappa_scale).to_event(1),
                    tfm_kappa,
                )
            elif self.kappa is not None:
                out["kappa"] = dist.Delta(self.kappa)

            # ``eta_anchor`` (real-valued) posterior — identity transform.
            if (
                "eta_anchor" not in frozen
                and self.eta_anchor_loc is not None
                and self.eta_anchor_scale is not None
            ):
                out["eta_anchor"] = dist.Normal(
                    self.eta_anchor_loc, self.eta_anchor_scale
                ).to_event(1)
            elif self.eta_anchor is not None:
                out["eta_anchor"] = dist.Delta(self.eta_anchor)

            # ``mu`` (latent prior center, zeros for TSLN-Logit) — Delta.
            if self.mu is not None:
                out["mu"] = dist.Delta(self.mu)

            # Derived reporting quantities at ``z = 0`` — Delta marginals.
            if self.gene_mean is not None:
                out["gene_mean"] = dist.Delta(self.gene_mean)
            if self.alpha is not None:
                out["alpha"] = dist.Delta(self.alpha)
            if self.beta is not None:
                out["beta"] = dist.Delta(self.beta)

            # Capture — fixed-offset only in PR-2.  ``eta`` is
            # *always* a frozen offset for TSLN-Logit when capture is
            # active (Rev 4 invariant; soft eta would route to the
            # deferred joint Newton).  Mirror ``get_map`` and expose a
            # Delta at the fixed offset value regardless of whether
            # ``"eta"`` is in ``frozen_params``, so downstream
            # diagnostics see the capture quantity (auditor's Step 3-5
            # fix — frozen ≠ hidden).
            if self.eta_loc is not None:
                out["eta_capture"] = dist.Delta(self.eta_loc)
                out["p_capture"] = dist.Delta(jnp.exp(-self.eta_loc))
            return out

        if bm in ("lnm", "lnmvcp"):
            out = {
                "y_alr": dist.LowRankMultivariateNormal(
                    loc=self.mu, cov_factor=self.W, cov_diag=self.d
                )
            }
            # Totals posterior in unconstrained space.
            if self.totals_cov is not None and self.mu_T_loc is not None:
                totals_loc = jnp.stack(
                    [self.mu_T_loc, self.r_T_loc]
                )
                out["totals_unconstrained"] = dist.MultivariateNormal(
                    loc=totals_loc,
                    covariance_matrix=self.totals_cov,
                )
            # Constrained marginal mu_T and r_T distributions.
            tfm = resolve_numpyro_transform(self.model_config)
            if self.mu_T_loc is not None and self.mu_T_scale is not None:
                out["mu_T"] = dist.TransformedDistribution(
                    dist.Normal(self.mu_T_loc, self.mu_T_scale),
                    tfm,
                )
            if self.r_T_loc is not None and self.r_T_scale is not None:
                out["r_T"] = dist.TransformedDistribution(
                    dist.Normal(self.r_T_loc, self.r_T_scale),
                    tfm,
                )
            if self.p_capture_loc is not None:
                out["p_capture"] = dist.Delta(self.p_capture_loc)
            return out

        raise NotImplementedError(
            f"get_distributions not implemented for base_model={bm!r}"
        )

