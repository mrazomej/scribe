"""Predictive sampling mixin for Laplace results.

This module exposes the public posterior-predictive API used by downstream
analysis and visualization code.  Each public method delegates to
model-specific private helpers so the user-facing signatures remain stable
across PLN and LNM-family Laplace fits.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp

from ._dispatch import _resolve_cascade_counts
from ._global_uncertainty import resolve_positive_fns
from ._results_sampling_helpers import (
    _ppc_lnm_library_anchored,
    _ppc_lnm_marginal,
    _ppc_lnm_per_cell,
    _ppc_lnm_per_cell_laplace,
    _ppc_nbln_marginal,
    _ppc_nbln_per_cell,
    _ppc_nbln_per_cell_laplace,
    _ppc_pln_library_anchored,
    _ppc_pln_marginal,
    _ppc_pln_per_cell,
    _ppc_pln_per_cell_laplace,
    _ppc_twostate_ln_rate_marginal,
    _ppc_twostate_ln_rate_per_cell,
    _ppc_twostate_ln_rate_per_cell_laplace,
    _ppc_twostate_ln_logit_marginal,
    _ppc_twostate_ln_logit_per_cell,
    _ppc_twostate_ln_logit_per_cell_laplace,
)
from ._results_shared import _base_model


# =====================================================================
# Phase-2 R5-2: cascade-aware NBLN PPC array resolver.
# =====================================================================
#
# When a Phase-2 cascade-freeze is active, the post-fit Laplace ``r_scale``
# / ``mu_scale`` fields hold the NaN sentinel from
# ``compute_global_uncertainty``.  Naively sampling
# ``Normal(loc, NaN).sample()`` would produce all-NaN PPC counts.  The
# resolver routes frozen parameters through ``cascade_source``'s SVI
# posterior samples (full guide fidelity preserved) and the non-frozen
# ``mu`` through its Laplace ``Normal(mu_loc, mu_scale)`` posterior — a
# capability the helpers also lacked previously.
# Cascade-pool size cap.  The SVI guide is a fixed posterior approximation;
# drawing 1e6 independent samples from it is no more informative than drawing
# a few thousand.  ``plot_ppc(level="marginal")`` inflates ``n_samples`` to
# ``n_eff * n_cells_obs`` (≈ 1.6M for a typical scRNA-seq dataset), which is
# fine for the cheap legacy ``Normal`` samplers but would force the cascade
# path to materialize an O(GB) SVI sample tensor.  We cap the pool here and
# resample-with-replacement to reach the requested predictive count.
_CASCADE_POOL_MAX = 2048


def _draw_nbln_cascade_samples(
    cascade,
    cascade_counts: Optional[jnp.ndarray],
    n_samples: int,
    rng_key: jax.Array,
    pool_max: int = _CASCADE_POOL_MAX,
) -> Optional[Dict[str, jnp.ndarray]]:
    """Draw a bounded pool of SVI posterior samples from the cascade source.

    Pool size is ``min(n_samples, pool_max)``.  When the caller asks for
    more predictive draws than the pool, expand by resampling with
    replacement at the *parameter-array* level (see
    :func:`_expand_pool_to_n_samples`).  Returns ``None`` when the
    cascade reference is missing.  Forwards ``counts`` (resolved from
    the dedicated cache field or the SVI source's own
    ``_original_counts``) only when the source needs them (amortized
    capture).
    """
    if cascade is None:
        return None
    pool_size = min(int(n_samples), int(pool_max))
    sample_kwargs = {
        "rng_key": rng_key,
        "n_samples": pool_size,
        "store_samples": False,
    }
    counts = _resolve_cascade_counts(cascade, cascade_counts)
    if counts is not None:
        sample_kwargs["counts"] = counts
    return cascade.get_posterior_samples(**sample_kwargs)


def _expand_pool_to_n_samples(
    arr: jnp.ndarray,
    n_samples: int,
    rng_key: jax.Array,
) -> jnp.ndarray:
    """Return ``n_samples`` rows by indexing into ``arr`` with replacement.

    When ``arr.shape[0] >= n_samples``, take the first ``n_samples``
    rows (the SVI sampler already provides independent draws; the
    truncation simply matches the requested count).  Otherwise draw
    ``n_samples`` indices uniformly from ``[0, pool_size)`` — sampling
    with replacement from the SVI empirical posterior.  This keeps the
    cascade memory cost bounded by ``pool_size``, independent of how
    many predictive draws the caller asks for.
    """
    pool = int(arr.shape[0])
    if pool >= n_samples:
        return arr[:n_samples]
    idx = jax.random.randint(rng_key, (n_samples,), 0, pool)
    return arr[idx]


def _resolve_nbln_ppc_arrays(
    result,
    rng_key: jax.Array,
    n_samples: int,
    per_cell: bool,
) -> Dict[str, Optional[jnp.ndarray]]:
    """Build per-draw r/mu/eta sample arrays for NBLN PPC.

    For each parameter the source is picked in this priority order:

    1. **Cascade-frozen + cascade available**: SVI samples from
       ``cascade_source``, coordinate-transformed to NBLN target space.
    2. **Non-frozen + Laplace Normal posterior available**: draw
       ``Normal(loc, scale)`` per draw and (for ``r``) apply
       ``pos_forward``.
    3. **Else**: ``None`` — the helper falls back to the legacy point /
       MAP-shuffle path.

    Parameters
    ----------
    result : ScribeLaplaceResults-like
        Provides ``frozen_params``, ``cascade_source``,
        ``cascade_source_counts``, ``mu_loc``, ``mu_scale``, ``r_loc``,
        ``r_scale``, and ``model_config``.
    rng_key : jax.Array
        PRNG key used both for the cascade SVI draws and for the Laplace
        Normal draws.
    n_samples : int
        Predictive sample count.  Cascade samples are drawn to match.
    per_cell : bool
        If ``True``, ``eta_samples`` has shape ``(S, N)`` (per-cell SVI
        eta posterior).  If ``False``, ``eta_samples`` has shape
        ``(S,)`` (one eta per imaginary cell, chosen by random cell
        index from the per-draw cascade slice — preserves SVI
        uncertainty while remaining a marginal sampler).

    Returns
    -------
    Dict with keys ``"r_samples"``, ``"mu_samples"``, ``"eta_samples"``.
    Any entry may be ``None``.
    """
    frozen = getattr(result, "frozen_params", frozenset()) or frozenset()
    cascade = getattr(result, "cascade_source", None)
    cascade_counts = getattr(result, "cascade_source_counts", None)
    # Gene-subset index lives on results subsetted via
    # ``ScribeLaplaceResults.__getitem__``.  When present, the cascade
    # (which lives in the FULL gene panel — was not gene-subsetted
    # because the amortizer needs full counts) must be sliced to the
    # subset gene axis to match ``self.mu``/``self.W``/``self.d``.
    gene_idx = getattr(result, "_subset_gene_index", None)
    gene_idx_jnp = jnp.asarray(gene_idx) if gene_idx is not None else None

    k_cascade, k_r_lap, k_eta_pick, k_r_exp, k_mu_exp, k_eta_exp = (
        jax.random.split(rng_key, 6)
    )

    cascade_samples = None
    if frozen and cascade is not None:
        cascade_samples = _draw_nbln_cascade_samples(
            cascade,
            cascade_counts,
            n_samples,
            k_cascade,
        )

    # Subset-aware cascade routing.  When the Laplace fit's gene panel
    # is a STRICT subset of the SVI source's, the cascade samples live
    # on the source's larger gene axis and the source's ``"_other"``
    # column means a DIFFERENT aggregate from the Laplace target's
    # ``"_other"``.  Re-aggregate the SVI samples onto the target axis
    # via per-sample NB moment matching before any downstream slicing
    # or PPC sampling sees them.  See ``paper/_nb_lognormal.qmd``
    # §sec-nbln-cascade-aggregation for the math.
    _cascade_subset_info = getattr(result, "_cascade_subset_info", None)
    _subset_active = (
        cascade_samples is not None
        and _cascade_subset_info is not None
        and _cascade_subset_info.is_subset
        and not _cascade_subset_info.is_equal
    )
    if _subset_active and (
        "r" not in cascade_samples or "mu" not in cascade_samples
    ):
        # Subset aggregation couples r and mu; bypassing it would
        # feed source-shape arrays to ``_gene_slice`` and either
        # crash or use the wrong ``_other`` aggregate.  Raise so the
        # bug surfaces clearly at PPC time.
        raise ValueError(
            "Subset-aware cascade PPC requires the SVI source to expose "
            "both 'r' and 'mu' per sample; got keys "
            f"{sorted(cascade_samples.keys())}."
        )
    if _subset_active:
        from .priors import (
            _aggregate_other_nb,
            _assemble_per_gene_subset_samples,
        )

        _r_src = jnp.asarray(cascade_samples["r"])
        _mu_src = jnp.asarray(cascade_samples["mu"])
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
        cascade_samples = dict(cascade_samples)
        cascade_samples["r"] = jnp.concatenate(
            [_r_kept, _r_other_s[:, None]], axis=1
        )
        cascade_samples["mu"] = jnp.concatenate(
            [_mu_kept, _mu_other_s[:, None]], axis=1
        )

    def _gene_slice(arr: jnp.ndarray) -> jnp.ndarray:
        """Slice a (S, G_full) array onto the gene subset, if any."""
        if gene_idx_jnp is None:
            return arr
        return arr[:, gene_idx_jnp]

    r_samples: Optional[jnp.ndarray] = None
    mu_samples: Optional[jnp.ndarray] = None
    eta_samples: Optional[jnp.ndarray] = None

    # ---- r ---------------------------------------------------------
    if "r" in frozen and cascade_samples is not None and "r" in cascade_samples:
        # SVI ``r`` is in positive (constrained) space — feed directly
        # to ``LogMeanNegativeBinomial(concentration=...)``.  Subset to
        # the result's gene axis, then expand pool to n_samples.
        r_pool = _gene_slice(jnp.asarray(cascade_samples["r"]))
        r_samples = _expand_pool_to_n_samples(r_pool, n_samples, k_r_exp)
    elif (
        "r" not in frozen
        and getattr(result, "r_loc", None) is not None
        and getattr(result, "r_scale", None) is not None
    ):
        r_loc = jnp.asarray(result.r_loc)
        r_scale = jnp.asarray(result.r_scale)
        # NaN sentinel guard: only sample if the Laplace path actually
        # produced a finite scale.  If any entry is NaN the resolver
        # bails out so the helper can fall back to the point ``r``.
        if jnp.all(jnp.isfinite(r_scale)):
            pos_fwd, _ = resolve_positive_fns(result.model_config)
            g_genes = int(r_loc.shape[0])
            r_un = r_loc[None, :] + r_scale[None, :] * jax.random.normal(
                k_r_lap, (n_samples, g_genes), dtype=r_loc.dtype
            )
            r_samples = pos_fwd(r_un)

    # ---- mu --------------------------------------------------------
    # Cascade-frozen ``mu`` (Level 4): use SVI samples in NBLN log-rate
    # coord.  Non-frozen ``mu``: fall back to the deterministic point.
    # We do NOT propagate ``Normal(mu_loc, mu_scale)`` for non-frozen
    # mu — the post-fit Laplace ``mu_scale`` carries the rigid-translation
    # gauge contamination flagged in the Phase-2 audits, and pushing it
    # into per-draw predictions blows up compositional structure in
    # library-anchored PPCs (genes randomly re-rank per draw).  Honest
    # ``mu`` uncertainty for cascade fits is available via cascade-freeze
    # at Level 4 (``informative_priors_freeze=("r","mu","eta")``).
    if (
        "mu" in frozen
        and cascade_samples is not None
        and "mu" in cascade_samples
    ):
        # SVI ``mu`` is the NB mean (positive); NBLN's ``mu`` is the
        # log-rate prior mean — apply ``log`` per sample.  Subset to
        # gene axis, then expand pool to n_samples.
        mu_pos = _gene_slice(
            jnp.asarray(cascade_samples["mu"])
        )  # (S, G_subset)
        mu_log = jnp.log(jnp.maximum(mu_pos, 1e-8))
        mu_samples = _expand_pool_to_n_samples(mu_log, n_samples, k_mu_exp)

    # ---- eta -------------------------------------------------------
    # Eta is per-cell, not per-gene; no gene-subset slicing needed.
    # Cell counts in the subsetted result match the cascade source
    # (gene subsetting doesn't touch the cell axis).
    if (
        "eta" in frozen
        and cascade_samples is not None
        and "eta_capture" in cascade_samples
    ):
        eta_full = jnp.asarray(cascade_samples["eta_capture"])  # (pool, N)
        if per_cell:
            # Per-cell PPC: need (n_samples, N).  Pool expansion picks
            # whole-cell rows; pool-mode SVI uncertainty preserved.
            eta_samples = _expand_pool_to_n_samples(
                eta_full, n_samples, k_eta_exp
            )
        else:
            # Marginal PPC: one eta per draw.  Pick a random cell index
            # per draw, then index into the (expanded) pool.  Combines
            # SVI posterior uncertainty with across-cell heterogeneity.
            eta_expanded = _expand_pool_to_n_samples(
                eta_full, n_samples, k_eta_exp
            )  # (n_samples, N)
            n_cells = int(eta_expanded.shape[1])
            cell_idx = jax.random.randint(k_eta_pick, (n_samples,), 0, n_cells)
            eta_samples = eta_expanded[jnp.arange(n_samples), cell_idx]

    return {
        "r_samples": r_samples,
        "mu_samples": mu_samples,
        "eta_samples": eta_samples,
    }


class SamplingResultsMixin:
    """Mixin exposing PPC and predictive-sampling public API."""

    def get_ppc_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        level: str = "library_anchored",
        counts: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Draw posterior predictive samples at a selected conditioning level.

        Parameters
        ----------
        rng_key : jax.Array, optional
            PRNG key used for all stochastic draws. If omitted, a deterministic
            default key is used.
        n_samples : int, default=100
            Number of predictive draws.
        level : {"per_cell", "library_anchored", "marginal"},
        default="library_anchored"
            Predictive-conditioning regime:

            - ``"per_cell"``: conditional on each observed cell.
            - ``"library_anchored"``: fresh latent composition + observed
              per-cell totals.
            - ``"marginal"``: fully marginal population predictive.
        counts : jnp.ndarray, optional
            Observed counts required by ``"library_anchored"`` and some
            per-cell LNM-family paths.
        **kwargs
            Additional backend-specific options (for example ``total_counts`` or
            ``chunk_size``).

        Returns
        -------
        jnp.ndarray or np.ndarray
            Predictive samples. Shape depends on ``level`` and model:

            - ``per_cell`` / ``library_anchored``: ``(n_samples, n_cells, G)``
            - ``marginal``: ``(n_samples, G)``

        Raises
        ------
        ValueError
            If ``level`` is invalid or required ``counts`` are missing.
        NotImplementedError
            If dispatch encounters an unsupported base model.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        if level not in ("per_cell", "library_anchored", "marginal"):
            raise ValueError(
                "level must be 'per_cell', 'library_anchored', or 'marginal'; "
                f"got {level!r}"
            )
        if level == "per_cell":
            return self.get_per_cell_predictive_samples(
                rng_key=rng_key, n_samples=n_samples, counts=counts, **kwargs
            )

        bm = _base_model(self.model_config)
        if level == "library_anchored":
            if counts is None:
                raise ValueError(
                    "level='library_anchored' requires `counts` for observed totals."
                )
            if bm in ("pln", "nbln", "twostate_ln_rate", "twostate_ln_logit"):
                # Library-anchored PPC samples ``softmax(x) ->
                # Multinomial`` against observed library size; the
                # count-noise choice (Poisson / NB / Poisson-Beta
                # compound) does not enter at this level, so NBLN and
                # TSLN-Rate reuse the PLN helper.  For NBLN cascade
                # fits, resolve a per-draw ``mu`` array (frozen-mu →
                # cascade samples; non-frozen-mu with Laplace Normal →
                # Normal(mu_loc, mu_scale)).  PLN and TSLN-Rate never
                # supply ``mu_samples`` here.
                mu_samples = None
                if bm == "nbln":
                    arrays = _resolve_nbln_ppc_arrays(
                        self,
                        rng_key,
                        n_samples,
                        per_cell=False,
                    )
                    mu_samples = arrays["mu_samples"]
                # Commit 2b: decoupled NBLN scatters x_dev (G_kept) into
                # the full G_obs simplex; ``_other`` is deterministic.
                _layout = getattr(self, "axis_layout", None)
                _kept_idx_jx = (
                    jnp.asarray(_layout.kept_idx)
                    if _layout is not None and _layout.decoupled
                    else None
                )
                return _ppc_pln_library_anchored(
                    rng_key,
                    n_samples,
                    self.mu,
                    self.W,
                    self.d,
                    counts=counts,
                    mu_samples=mu_samples,
                    kept_idx=_kept_idx_jx,
                )
            if bm in ("lnm", "lnmvcp"):
                return _ppc_lnm_library_anchored(
                    rng_key,
                    n_samples,
                    self.mu,
                    self.W,
                    self.d,
                    self.alr_reference_idx,
                    counts=counts,
                )
            raise NotImplementedError(
                f"library_anchored PPC not implemented for base_model={bm!r}"
            )

        if bm == "pln":
            return _ppc_pln_marginal(
                rng_key,
                n_samples,
                self.mu,
                self.W,
                self.d,
                eta_loc=self.eta_loc,
            )
        if bm == "nbln":
            if self.r is None:
                raise ValueError(
                    "NBLN PPC requires the gene dispersion 'r' field."
                )
            pos_fwd, _ = resolve_positive_fns(self.model_config)
            # Resolve per-draw r/mu/eta arrays:
            # - cascade-frozen keys → SVI samples (full guide fidelity);
            # - non-frozen with Laplace Normal → Normal(loc, scale);
            # - otherwise → None, helper falls back to legacy logic.
            arrays = _resolve_nbln_ppc_arrays(
                self,
                rng_key,
                n_samples,
                per_cell=False,
            )
            # Commit 2b: thread kept_idx through to the marginal kernel
            # so the latent ``W`` / ``d`` (G_kept) get scattered onto
            # the full G_obs axis under decoupling.
            _layout = getattr(self, "axis_layout", None)
            _kept_idx_jx = (
                jnp.asarray(_layout.kept_idx)
                if _layout is not None and _layout.decoupled
                else None
            )
            return _ppc_nbln_marginal(
                rng_key,
                n_samples,
                self.mu,
                self.W,
                self.d,
                self.r,
                eta_loc=self.eta_loc,
                r_loc=self.r_loc,
                r_scale=self.r_scale,
                pos_forward=pos_fwd,
                r_samples=arrays["r_samples"],
                mu_samples=arrays["mu_samples"],
                eta_samples=arrays["eta_samples"],
                kept_idx=_kept_idx_jx,
            )
        if bm in ("lnm", "lnmvcp"):
            pos_fwd, _ = resolve_positive_fns(self.model_config)
            return _ppc_lnm_marginal(
                rng_key,
                n_samples,
                self.mu,
                self.W,
                self.d,
                self.alr_reference_idx,
                mu_T=self.mu_T,
                r_T=self.r_T,
                p_capture_loc=self.p_capture_loc,
                totals_cov=self.totals_cov,
                mu_T_loc=self.mu_T_loc,
                r_T_loc=self.r_T_loc,
                pos_forward=pos_fwd,
                **kwargs,
            )
        if bm == "twostate_ln_rate":
            if self.alpha is None or self.beta is None:
                raise ValueError(
                    "TSLN-Rate PPC requires the derived 'alpha' / 'beta' "
                    "Beta-shape fields on the result."
                )
            _layout = getattr(self, "axis_layout", None)
            _kept_idx_jx = (
                jnp.asarray(_layout.kept_idx)
                if _layout is not None and _layout.decoupled
                else None
            )
            return _ppc_twostate_ln_rate_marginal(
                rng_key,
                n_samples,
                self.mu,
                self.W,
                self.d,
                self.alpha,
                self.beta,
                eta_loc=self.eta_loc,
                kept_idx=_kept_idx_jx,
            )
        if bm == "twostate_ln_logit":
            if (
                self.rate is None
                or self.kappa is None
                or self.eta_anchor is None
            ):
                raise ValueError(
                    "TSLN-Logit PPC requires 'rate' / 'kappa' / "
                    "'eta_anchor' fields on the result."
                )
            return _ppc_twostate_ln_logit_marginal(
                rng_key,
                n_samples,
                self.mu,
                self.W,
                self.d,
                self.rate,
                self.kappa,
                self.eta_anchor,
                eta_loc=self.eta_loc,
            )
        raise NotImplementedError(
            f"marginal PPC not implemented for base_model={bm!r}"
        )

    def get_predictive_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        **kwargs,
    ) -> jnp.ndarray:
        """Alias for fully marginal population predictive sampling.

        This is kept for API compatibility with other result classes that
        expose ``get_predictive_samples`` as their main PPC entry point.
        """
        return self.get_ppc_samples(
            rng_key=rng_key,
            n_samples=n_samples,
            level="marginal",
            **kwargs,
        )

    def get_per_cell_predictive_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        **kwargs,
    ) -> jnp.ndarray:
        """Draw per-cell posterior predictive samples with Laplace uncertainty.

        This path propagates uncertainty in per-cell latent variables by
        sampling from Laplace posterior approximations around each cell-specific
        MAP state before drawing observation noise.

        Parameters
        ----------
        rng_key : jax.Array, optional
            PRNG key for stochastic sampling.
        n_samples : int, default=100
            Number of predictive draws per cell.
        **kwargs
            Model-specific arguments such as ``counts``/``total_counts`` and
            optional sampling ``chunk_size``.

        Returns
        -------
        jnp.ndarray or np.ndarray
            Predictive counts with shape ``(n_samples, n_cells, G)``.

        Raises
        ------
        NotImplementedError
            If dispatch encounters an unsupported base model.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        bm = _base_model(self.model_config)
        if bm == "pln":
            return _ppc_pln_per_cell_laplace(
                rng_key, n_samples, self.x_loc, self.eta_loc, self.W, self.d
            )
        if bm == "nbln":
            if self.r is None:
                raise ValueError(
                    "NBLN PPC requires the gene dispersion 'r' field."
                )
            pos_fwd, _ = resolve_positive_fns(self.model_config)
            # Per-cell path: eta_samples shape (S, N) when cascade
            # frozen-eta is active; per-draw per-cell r_samples for
            # frozen-r.  mu_samples is unused here (per-cell PPC
            # conditions on the cell-specific MAP ``x_loc`` rather
            # than redrawing ``x`` from its prior).
            arrays = _resolve_nbln_ppc_arrays(
                self,
                rng_key,
                n_samples,
                per_cell=True,
            )
            # Commit 2b: thread mu + kept_idx through for decoupled NBLN
            # so the per-cell kernel reconstructs full G_obs log-rate
            # from kept-axis ``x_dev`` (which is what ``x_loc`` carries
            # under decoupling).
            _layout = getattr(self, "axis_layout", None)
            _kept_idx_jx = (
                jnp.asarray(_layout.kept_idx)
                if _layout is not None and _layout.decoupled
                else None
            )
            return _ppc_nbln_per_cell_laplace(
                rng_key,
                n_samples,
                self.x_loc,
                self.eta_loc,
                self.W,
                self.d,
                self.r,
                r_loc=self.r_loc,
                r_scale=self.r_scale,
                pos_forward=pos_fwd,
                r_samples=arrays["r_samples"],
                eta_samples=arrays["eta_samples"],
                mu=self.mu if _kept_idx_jx is not None else None,
                kept_idx=_kept_idx_jx,
            )
        if bm in ("lnm", "lnmvcp"):
            pos_fwd, _ = resolve_positive_fns(self.model_config)
            return _ppc_lnm_per_cell_laplace(
                rng_key,
                n_samples,
                self.mu,
                self.W,
                self.d,
                self.z_loc,
                self.y_alr_loc,
                self.alr_reference_idx,
                mu_T=self.mu_T,
                r_T=self.r_T,
                p_capture_loc=self.p_capture_loc,
                totals_cov=self.totals_cov,
                mu_T_loc=self.mu_T_loc,
                r_T_loc=self.r_T_loc,
                pos_forward=pos_fwd,
                **kwargs,
            )
        if bm == "twostate_ln_rate":
            if self.alpha is None or self.beta is None:
                raise ValueError(
                    "TSLN-Rate per-cell PPC requires 'alpha' / 'beta' "
                    "Beta-shape fields on the result."
                )
            _layout = getattr(self, "axis_layout", None)
            _kept_idx_jx = (
                jnp.asarray(_layout.kept_idx)
                if _layout is not None and _layout.decoupled
                else None
            )
            return _ppc_twostate_ln_rate_per_cell_laplace(
                rng_key,
                n_samples,
                self.x_loc,
                self.eta_loc,
                self.W,
                self.d,
                self.alpha,
                self.beta,
                mu=self.mu if _kept_idx_jx is not None else None,
                kept_idx=_kept_idx_jx,
            )
        if bm == "twostate_ln_logit":
            if (
                self.rate is None
                or self.kappa is None
                or self.eta_anchor is None
            ):
                raise ValueError(
                    "TSLN-Logit per-cell PPC requires 'rate' / 'kappa' / "
                    "'eta_anchor' fields on the result."
                )
            return _ppc_twostate_ln_logit_per_cell_laplace(
                rng_key,
                n_samples,
                self.x_loc,
                self.eta_loc,
                self.W,
                self.d,
                self.rate,
                self.kappa,
                self.eta_anchor,
            )
        raise NotImplementedError(
            f"get_per_cell_predictive_samples not implemented for base_model={bm!r}"
        )

    def get_map_ppc_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        **kwargs,
    ) -> jnp.ndarray:
        """Draw per-cell MAP-only predictive samples.

        Unlike :meth:`get_per_cell_predictive_samples`, this method keeps
        per-cell latents fixed at their MAP values and samples only observation
        noise. It is useful for quick sanity checks when full Laplace posterior
        sampling is unnecessary.

        Parameters
        ----------
        rng_key : jax.Array, optional
            PRNG key for stochastic sampling.
        n_samples : int, default=100
            Number of predictive draws per cell.
        **kwargs
            Model-specific arguments such as ``counts``/``total_counts`` and
            optional sampling ``chunk_size``.

        Returns
        -------
        jnp.ndarray or np.ndarray
            Predictive counts with shape ``(n_samples, n_cells, G)``.

        Raises
        ------
        NotImplementedError
            If dispatch encounters an unsupported base model.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        bm = _base_model(self.model_config)
        if bm == "pln":
            return _ppc_pln_per_cell(
                rng_key, n_samples, self.x_loc, self.eta_loc
            )
        if bm == "nbln":
            if self.r is None:
                raise ValueError(
                    "NBLN PPC requires the gene dispersion 'r' field."
                )
            # Commit 2b: under decoupling, ``x_loc`` is ``x_dev`` on
            # G_kept; the kernel needs ``mu`` + ``kept_idx`` to
            # reconstruct the full G_obs per-cell log-rate.
            _layout = getattr(self, "axis_layout", None)
            _kept_idx_jx = (
                jnp.asarray(_layout.kept_idx)
                if _layout is not None and _layout.decoupled
                else None
            )
            return _ppc_nbln_per_cell(
                rng_key,
                n_samples,
                self.x_loc,
                self.eta_loc,
                self.r,
                mu=self.mu if _kept_idx_jx is not None else None,
                kept_idx=_kept_idx_jx,
            )
        if bm in ("lnm", "lnmvcp"):
            return _ppc_lnm_per_cell(
                rng_key,
                n_samples,
                self.mu,
                self.W,
                self.d,
                self.z_loc,
                self.y_alr_loc,
                self.alr_reference_idx,
                mu_T=self.mu_T,
                r_T=self.r_T,
                p_capture_loc=self.p_capture_loc,
                **kwargs,
            )
        if bm == "twostate_ln_rate":
            if self.alpha is None or self.beta is None:
                raise ValueError(
                    "TSLN-Rate map PPC requires 'alpha' / 'beta' "
                    "Beta-shape fields on the result."
                )
            _layout = getattr(self, "axis_layout", None)
            _kept_idx_jx = (
                jnp.asarray(_layout.kept_idx)
                if _layout is not None and _layout.decoupled
                else None
            )
            return _ppc_twostate_ln_rate_per_cell(
                rng_key,
                n_samples,
                self.x_loc,
                self.eta_loc,
                self.alpha,
                self.beta,
                mu=self.mu if _kept_idx_jx is not None else None,
                kept_idx=_kept_idx_jx,
            )
        if bm == "twostate_ln_logit":
            if (
                self.rate is None
                or self.kappa is None
                or self.eta_anchor is None
            ):
                raise ValueError(
                    "TSLN-Logit map PPC requires 'rate' / 'kappa' / "
                    "'eta_anchor' fields on the result."
                )
            return _ppc_twostate_ln_logit_per_cell(
                rng_key,
                n_samples,
                self.x_loc,
                self.eta_loc,
                self.rate,
                self.kappa,
                self.eta_anchor,
            )
        raise NotImplementedError(
            f"get_map_ppc_samples not implemented for base_model={bm!r}"
        )

    def get_compositional_samples(
        self,
        n_samples: int = 2048,
        rng_key: Optional[jax.Array] = None,
        chunk_size: int = 256,
        store_samples: bool = True,
    ):
        """Draw simplex compositions from the fitted marginal distribution.

        Generates samples from the model's *generative marginal* — the
        same procedure as the ``level="marginal"`` PPC sampler but
        stops at the simplex step (no observation noise applied):

            z ∼ 𝒩(0, 𝐈ₖ),  ε ∼ 𝒩(0, 𝐈_{Gₑₑ𝚏𝚏})
            latent = μ + 𝑊 z + √d ⊙ ε
            ρ = softmax_full(latent)

        For LNM / LNMVCP, the latent lives in (G−1)-dim ALR space and
        is augmented with a zero at ``alr_reference_idx`` before
        softmax — the standard ALR⁻¹ map. For PLN, the latent is
        G-dim log-rate space and softmax is applied directly; the
        per-cell capture offset η cancels because softmax is
        invariant to a global additive shift (the rigid-translation
        degeneracy in operation).

        Each draw is an *independent imaginary cell* from the
        model's fitted population distribution — no observed-cell
        information enters any of the latents. This is the right
        sample source for population-level downstream analyses
        (empirical DE, gene-set tests, biological summaries on
        compositions) on LNM and PLN fits, mirroring
        ``ScribeMCMCResults.get_compositional_samples`` for the
        DM/NB-family path.

        Parameters
        ----------
        n_samples : int, default 2048
            Total number of fresh simplex draws.
        rng_key : jax.Array, optional
            JAX PRNG key. Defaults to ``jax.random.PRNGKey(0)``.
        chunk_size : int, default 256
            Per-chunk sample count. Each chunk is moved to host
            memory before the next is drawn so the device peak
            stays bounded by ``chunk_size · G · 4 bytes``.
        store_samples : bool, default True
            If True, store the returned array in
            ``self.compositional_samples`` for reuse without
            re-sampling.

        Returns
        -------
        np.ndarray, shape ``(n_samples, G)``
            Full-dimensional simplex samples (rows sum to 1).

        Raises
        ------
        NotImplementedError
            If the base model is not PLN, LNM, or LNMVCP.
        """
        import numpy as _np

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        bm = _base_model(self.model_config)
        if bm in ("lnm", "lnmvcp"):
            mu = jnp.asarray(self.mu)
            W = jnp.asarray(self.W)
            d = self.d
            ref_idx = int(getattr(self.model_config, "alr_reference_idx", -1))
            n_genes_full = int(W.shape[0]) + 1
            if ref_idx < 0:
                ref_idx = n_genes_full + ref_idx
            is_alr = True
        elif bm in ("pln", "nbln", "twostate_ln_rate", "twostate_ln_logit"):
            # PLN-family (PLN / NBLN / TSLN-Rate) all produce log-rates
            # ``y_log_rate = mu + W z``; softmax of those log-rates gives
            # the compositional sample.  The count-noise layer (Poisson /
            # NB / Poisson-Beta compound) never enters here — compositions
            # are pre-observation-noise.  For TSLN-Rate the gene-level
            # Beta concentration is an observation-noise layer too: in
            # expectation ``⟨u_g | log_rate, p⟩ = exp(log_rate) · p`` and
            # under softmax-of-log-rates the gene-level Beta noise
            # averages out, so the compositional view is identical to
            # PLN/NBLN.
            #
            # Use the gauge-invariant projection ``W_perp = W − mean(W, axis=0)``
            # via :meth:`get_W_compositional`.  Math note: ``softmax(mu + W z
            # + √d ε) == softmax(mu + W_perp z + √d ε)`` exactly, because the
            # difference is a per-draw scalar times the all-ones vector and
            # softmax is translation-invariant.  Using W_perp here therefore
            # changes nothing about the output values, but it makes the
            # gauge-invariance (Theorem 1 in `_diffexp_nbln_robustness.qmd`)
            # manifest in the implementation rather than relying on softmax
            # to project the gauge out at evaluation time.  For fits with
            # non-trivial ``gauge_contamination_ratio`` it also keeps the
            # pre-softmax latent magnitudes smaller, which is friendlier to
            # floating-point precision.  For PLN where gauge contamination is
            # typically smaller this is functionally a no-op.
            #
            # Decoupled-layout branch (Commit 2b for NBLN; same pattern
            # applies to PLN/TSLN-* once their math commits land): under
            # ``correlate_other_column=False``, ``W`` and ``d`` live on
            # the kept-gene axis (G_kept) while ``μ`` lives on the full
            # observation axis (G_obs).  We sample ``x_dev`` on G_kept,
            # scatter ``μ_kept + x_dev`` at kept positions, and use
            # ``μ_other`` directly at ``other_idx`` (deterministic, no
            # z modulation — matching the math-contract).
            mu = jnp.asarray(self.mu)
            W = jnp.asarray(self.get_W_compositional())
            d = self.d
            ref_idx = None
            n_genes_full = int(mu.shape[0])
            is_alr = False
            _axis_layout = getattr(self, "axis_layout", None)
            is_decoupled = _axis_layout is not None and getattr(
                _axis_layout, "decoupled", False
            )
            if is_decoupled:
                _kept_idx_dev = jnp.asarray(
                    _axis_layout.kept_idx, dtype=jnp.int32
                )
                _other_idx_dev = int(_axis_layout.other_idx)
            else:
                _kept_idx_dev = None
                _other_idx_dev = None
        else:
            raise NotImplementedError(
                f"get_compositional_samples not implemented for "
                f"base_model={bm!r}"
            )

        if d is None:
            d = jnp.zeros(W.shape[0], dtype=W.dtype)
        else:
            d = jnp.asarray(d)
        sqrt_d = jnp.sqrt(jnp.maximum(d, 0.0))

        # Latent dimension for sampling ``z`` / ``ε``:
        #   • PLN/NBLN legacy: ``G_eff = G_obs`` (mu axis).
        #   • PLN/NBLN decoupled: ``G_eff = G_kept = W.shape[0]`` (since
        #     ``W`` lives on the kept axis under decoupling).
        #   • LNM/LNMVCP: ``G_eff = G_obs − 1`` (ALR latent), already
        #     enforced via the ``n_genes_full = W.shape[0] + 1`` line.
        if (not is_alr) and (
            "_kept_idx_dev" in locals() and _kept_idx_dev is not None
        ):
            G_eff = int(W.shape[0])  # G_kept
        else:
            G_eff = int(W.shape[0])
        k = int(W.shape[1])

        n_total = int(n_samples)
        n_chunks = (n_total + chunk_size - 1) // chunk_size
        chunk_keys = jax.random.split(rng_key, n_chunks)
        pieces = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_total)
            size = end - start
            k_z, k_eps = jax.random.split(chunk_keys[i])
            z = jax.random.normal(k_z, (size, k), dtype=mu.dtype)
            eps = jax.random.normal(k_eps, (size, G_eff), dtype=mu.dtype)
            # Under decoupling, ``z @ W.T + sqrt_d * eps`` lives on
            # G_kept and represents the per-draw ``x_dev``.  We add μ
            # on the G_obs axis by scattering: μ_kept + x_dev at kept
            # positions, μ_other unchanged at ``other_idx``.
            # Under legacy / LNM, ``latent = μ + z @ W.T + sqrt_d * eps``
            # directly on the full axis as before.
            if (not is_alr) and (
                "_kept_idx_dev" in locals() and _kept_idx_dev is not None
            ):
                x_dev = z @ W.T + sqrt_d[None, :] * eps  # (size, G_kept)
                latent = jnp.broadcast_to(mu[None, :], (size, n_genes_full))
                latent = latent.at[:, _kept_idx_dev].add(x_dev)
            else:
                latent = mu[None, :] + z @ W.T + sqrt_d[None, :] * eps

            if is_alr:
                # ALR → simplex: augment with a zero at the reference
                # position, then softmax over G dims.
                full = jnp.zeros((size, n_genes_full), dtype=latent.dtype)
                other = [g for g in range(n_genes_full) if g != ref_idx]
                full = full.at[..., jnp.asarray(other)].set(latent)
                simplex = jax.nn.softmax(full, axis=-1)
            else:
                # PLN: softmax of log-rate; η cancels.  Under decoupled
                # the latent is already on the full G_obs axis (scatter
                # above), so softmax produces a proper simplex of size
                # G_obs with ``_other`` participating deterministically.
                simplex = jax.nn.softmax(latent, axis=-1)

            pieces.append(_np.asarray(simplex))

        out = _np.concatenate(pieces, axis=0)
        if store_samples:
            self.compositional_samples = out
        return out
