"""Core model-agnostic accessors for Laplace results.

This module collects methods that are independent of the selected generative
model branch (PLN, LNM, or LNMVCP).  The methods here expose global decoder
parameters and derived covariance/correlation diagnostics used by both analysis
and visualization layers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp

from ._results_shared import _base_model


class CoreResultsMixin:
    """Mixin providing model-agnostic parameter and correlation accessors.

    Notes
    -----
    The methods in this mixin intentionally avoid assumptions about the
    per-cell latent representation. They consume only shared global fields such
    as ``mu``, ``W``, and ``d``, which are present for all Laplace fits.
    """

    def get_mu(self) -> jnp.ndarray:
        """Return decoder bias ``μ``.

        Returns
        -------
        jnp.ndarray
            Decoder bias in the latent coordinate system used by the fitted
            model:

            - PLN: shape ``(G,)`` in log-rate coordinates.
            - LNM/LNMVCP: shape ``(G-1,)`` in ALR coordinates.
        """
        return self.mu

    def get_W(self) -> jnp.ndarray:
        """Return decoder loading matrix ``W``.

        Returns
        -------
        jnp.ndarray
            Low-rank loading matrix:

            - PLN: shape ``(G, k)``.
            - LNM/LNMVCP: shape ``(G-1, k)``.
        """
        return self.W

    def get_d(self) -> jnp.ndarray:
        """Return diagonal residual variance ``d``.

        Returns
        -------
        jnp.ndarray
            Positive residual diagonal, aligned with ``mu``.
        """
        return self.d

    def get_program_activity(self):
        """Return the per-donor relative program activity ``s`` (Rung 1).

        Populated only for NB-LogNormal fits with
        ``correlation_hierarchy="program_scales"``; ``None`` otherwise.

        Returns
        -------
        jnp.ndarray or None
            Shape ``(n_datasets, K)``. Entry ``s[d, k]`` is donor ``d``'s
            relative activity of regulatory program ``k`` (geometric mean 1
            across donors per program, by the sum-to-zero gauge), so donor
            ``d``'s gene-gene covariance is
            ``Σ_d = W diag(s[d]^2) Wᵀ + diag(d_resid)``. Read alongside
            :meth:`get_W` (the shared programs) and
            ``result.program_scale_tau`` (the between-donor scale ``τ_s``).
        """
        return self.program_activity

    def get_gene_mean_per_dataset(self):
        """Return the per-donor frozen gene means ``mu^(d)`` (step 4b).

        Populated only when the hierarchical-marginal cascade froze a
        per-donor mean (a hierarchical independent-gene SVI source passed
        via ``informative_priors_from=`` with ``"mu"`` in
        ``informative_priors_freeze``); ``None`` otherwise.

        Returns
        -------
        jnp.ndarray or None
            Shape ``(n_datasets, G)`` in the NBLN log-rate coordinate.
            Row ``d`` is dataset ``d``'s per-gene latent prior mean and
            aligns with the target leaf indexing (``dataset_indices``).
            :meth:`get_mu` returns the donor-pooled per-gene mean; this
            method exposes the unpooled per-donor table.
        """
        return self.gene_mean_per_dataset

    def get_sigma(self) -> jnp.ndarray:
        """Return prior covariance ``Σ = W Wᵀ + diag(d)``.

        Returns
        -------
        jnp.ndarray
            Dense covariance matrix in latent observation coordinates:

            - PLN: shape ``(G, G)``.
            - LNM/LNMVCP: shape ``(G-1, G-1)``.
        """
        return self.W @ self.W.T + jnp.diag(self.d)

    def get_correlation(self) -> jnp.ndarray:
        """Return correlation matrix derived from ``Σ``.

        The matrix is computed as ``Corr = Σ / (σ σᵀ)`` where
        ``σ = sqrt(diag(Σ))``. A small floor is used before ``sqrt`` so that
        near-zero diagonal entries do not generate NaNs.

        Returns
        -------
        jnp.ndarray
            Correlation matrix with the same shape as :meth:`get_sigma`.
        """
        sigma = self.get_sigma()
        std = jnp.sqrt(jnp.maximum(jnp.diag(sigma), 1e-30))
        return sigma / (std[:, None] * std[None, :])

    def get_library_size_direction(self) -> jnp.ndarray:
        """Return the latent direction closest to the all-ones gene vector.

        This method is a thin wrapper over
        :func:`scribe.stats.correlation_diagnostics.library_size_direction`.
        It is useful when diagnosing whether dominant correlation structure is
        explained by a global library-size axis.

        Returns
        -------
        jnp.ndarray
            Unit vector in factor space with shape ``(k,)``.
        """
        from ..stats.correlation_diagnostics import library_size_direction

        return library_size_direction(self.W)

    def get_correlation_residual(
        self,
        method: str = "library_size",
        n_components: int = 1,
        include_diagonal_d: bool = False,
    ) -> jnp.ndarray:
        """Return correlation after nuisance-direction projection.

        Parameters
        ----------
        method : {"library_size", "pc"}, default="library_size"
            Projection strategy:

            - ``"library_size"``: remove the library-size direction inferred
              from ``W``.
            - ``"pc"``: remove the top ``n_components`` principal directions.
        n_components : int, default=1
            Number of principal components to remove when ``method="pc"``.
        include_diagonal_d : bool, default=False
            Whether to include ``diag(d)`` before normalization.

        Returns
        -------
        jnp.ndarray
            Residual correlation matrix in the same coordinate system and shape
            as :meth:`get_correlation`.
        """
        from ..stats.correlation_diagnostics import correlation_residual

        return correlation_residual(
            self.W,
            self.d,
            method=method,
            n_components=n_components,
            include_diagonal_d=include_diagonal_d,
        )

    def summarize_correlation_structure(
        self,
        *,
        n_top_eig: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Summarize latent correlation structure with quantitative diagnostics.

        Parameters
        ----------
        n_top_eig : int, default=10
            Number of leading eigenvalues of ``WᵀW`` to include.
        verbose : bool, default=True
            If ``True``, emit a human-readable report; otherwise return only
            the structured dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary of diagnostics including alignment with library-size,
            loading concentration, explained-variance summaries, effective
            rank, and off-diagonal quantiles before/after projection.
        """
        from ..stats.correlation_diagnostics import (
            summarize_correlation_structure as _summarize,
        )

        bm = _base_model(self.model_config)
        space = "ALR space" if bm in ("lnm", "lnmvcp") else "log-rate space"
        return _summarize(
            self.W,
            self.d,
            space_label=space,
            model_label=f"{bm.upper()} Laplace",
            n_top_eig=n_top_eig,
            verbose=verbose,
        )

    def get_p_capture(self) -> Optional[jnp.ndarray]:
        """Return per-cell capture probability when available.

        Two Laplace branches can encode capture effects:

        - LNMVCP stores ``p_capture_loc`` directly.
        - PLN with capture anchor stores ``eta_loc`` and uses
          ``p_capture = exp(-eta_loc)``.

        Returns
        -------
        jnp.ndarray or None
            Per-cell capture probability vector, or ``None`` when the fitted
            model has no capture term.
        """
        if self.p_capture_loc is not None:
            return self.p_capture_loc
        if self.eta_loc is not None:
            return jnp.exp(-self.eta_loc)
        return None

    # --- Phase-2: gauge-invariant W accessors ----------------------------
    #
    # The NBLN (and PLN) model parameterize the cross-gene correlation
    # structure via a low-rank loadings matrix W in absolute log-rate
    # space.  Under the per-cell rigid translation gauge
    # ``(x_c, η_c) → (x_c + Δ_c · 1, η_c + Δ_c)``, W has an unidentified
    # rank-1 component along the all-ones direction that corresponds to
    # cell-scaling slop, not biology.  The biologically meaningful
    # signal is the gene-centered projection
    #
    #     W_perp = (I − 1·1^T / G) W,
    #
    # which is gauge-invariant by construction.  For LNM/LNMVCP, W
    # already lives in ALR compositional coordinates (a gauge-fixed
    # quotient space) and the projection is a no-op.  See
    # ``paper/_diffexp_nbln_robustness.qmd`` for the theorem.

    def get_W_compositional(self) -> jnp.ndarray:
        """Return the gauge-invariant gene-centered loadings matrix.

        For PLN and NBLN, the loadings matrix W lives in absolute
        log-rate space and is subject to a rank-1 gauge along the
        all-ones direction.  The biologically meaningful cross-gene
        correlation structure is captured by

            W_perp = (I - 1 1^T / G) W

        which projects out the all-ones component.  For LNM and LNMVCP,
        W already lives in ALR compositional coordinates and is
        returned unchanged.

        Returns
        -------
        jnp.ndarray
            Shape ``(G, k)`` for PLN/NBLN (gene-centered).  Shape
            ``(G-1, k)`` for LNM-family (unchanged).

        See Also
        --------
        get_gauge_diagnostics : quantify how much gauge-coupled
            structure ``W`` carries (zero for LNM-family by
            construction; small for NBLN with default freeze of
            ``("r", "eta")``).
        """
        bm = _base_model(self.model_config)
        W = self.W
        if bm in ("pln", "nbln", "twostate_ln_rate", "twostate_ln_logit"):
            # PLN-family: W lives in absolute log-rate space.  Project
            # out the all-ones direction so the returned loadings are
            # gauge-invariant.  TSLN-Rate joins PLN/NBLN here: its
            # ``y_log_rate = μ + W z + √d ε`` decoder has the same
            # rigid-translation gauge between ``μ_g`` and a per-cell
            # offset along ``1_G`` (capture / library size).
            return W - W.mean(axis=0, keepdims=True)
        # LNM-family already compositional.
        return W

    def get_gauge_diagnostics(self) -> Dict[str, float]:
        """Quantify the gauge contamination in the fitted W.

        For PLN and NBLN, the loadings matrix has both a biologically
        meaningful gene-centered component (``W_perp``) and a
        gauge-coupled all-ones-direction component (``W_para``).  This
        method returns the Frobenius norms of both plus their ratio.

        **Threshold interpretation depends on whether a Phase-3
        loadings shrinkage prior is active.**

        - **Without loadings shrinkage** (no ``priors={"loadings":
          ...}``).  A clean cascade-frozen fit (default Phase-2 freeze
          of ``r`` and ``eta``) should produce a
          ``gauge_contamination_ratio`` < 0.05; values > 0.2 flag that
          NBVCP's eta estimate is not absorbing all the cell-scaling,
          and the gauge component is taking signal that should live in
          ``W_perp``.

        - **With loadings shrinkage** (e.g.
          ``priors={"loadings": {"type": "horseshoe_columnwise",
          ...}}``).  The shrinkage prior aggressively shrinks
          ``W_perp`` to the data-supported rank while leaving the
          all-ones component ``W_para`` unconstrained — the prior
          targets ``W_perp`` only.  This is by design (the cascade
          freeze on ``eta`` is what pins the gauge structurally, not
          a ridge on the W gauge component).  As a result the *ratio*
          can climb to 0.5–0.8 even on a clean fit, driven by the
          denominator shrinking, not the numerator growing.  Inspect
          the **absolute norms** instead:

            * Healthy: both ``W_compositional_norm`` and
              ``W_all_ones_component_norm`` are modest in absolute
              terms (each a few × the per-gene SD of the data) and
              are similar in scale across different shrinkage prior
              families (e.g. NEG vs horseshoe).  The ratio differing
              across prior families is the *expected* signature of
              an under-determined parameter — the all-ones direction
              is parametrically free, so different priors put it in
              different places.
            * Concerning: ratio ≫ 1 *and* both norms large in
              absolute terms.  This would still flag the original
              failure mode (gauge component taking real signal).

        For LNM/LNMVCP, W is in ALR coordinates by construction and the
        ratio is identically zero (no all-ones contamination possible).

        Returns
        -------
        Dict[str, float]
            ``W_compositional_norm`` — Frobenius norm of ``W_perp``.
            ``W_all_ones_component_norm`` — Frobenius norm of
                ``W − W_perp`` (the all-ones-direction component).
            ``gauge_contamination_ratio`` — ratio of the two; zero
                for LNM-family.
        """
        bm = _base_model(self.model_config)
        W = self.W
        if bm in ("pln", "nbln", "twostate_ln_rate", "twostate_ln_logit"):
            W_perp = W - W.mean(axis=0, keepdims=True)
            W_para = W - W_perp
            perp_norm = float(jnp.linalg.norm(W_perp))
            para_norm = float(jnp.linalg.norm(W_para))
            return {
                "W_compositional_norm": perp_norm,
                "W_all_ones_component_norm": para_norm,
                "gauge_contamination_ratio": (
                    para_norm / max(perp_norm, 1e-12)
                ),
            }
        # LNM-family: W is already in ALR compositional coordinates.
        # No all-ones contamination possible by parameterization.
        return {
            "W_compositional_norm": float(jnp.linalg.norm(W)),
            "W_all_ones_component_norm": 0.0,
            "gauge_contamination_ratio": 0.0,
        }

    def get_correlation_compositional(self) -> jnp.ndarray:
        """Return correlation derived from the gauge-invariant ``W_perp``.

        For PLN and NBLN, this is the correlation matrix of
        ``W_perp W_perp^T + diag(d)`` — the biologically meaningful
        cross-gene correlation structure, after projecting out the
        rigid-translation gauge contamination.  For LNM and LNMVCP, ``W``
        is already in ALR compositional coordinates so this returns the
        same value as :meth:`get_correlation`.

        This is the recommended source for downstream "smart" gene
        selection (most positively correlated, most negatively
        correlated, etc.) since pairwise correlations from raw ``W``
        are biased by the all-ones gauge contamination at non-trivial
        ``gauge_contamination_ratio``.

        Returns
        -------
        jnp.ndarray
            Correlation matrix of shape ``(G, G)`` for PLN/NBLN, or
            ``(G-1, G-1)`` for LNM-family — same shape as
            :meth:`get_correlation`.

        See Also
        --------
        get_W_compositional : the underlying gauge-invariant loadings.
        get_correlation : full ``W``-based correlation, gauge-contaminated
            for PLN/NBLN at non-trivial ``gauge_contamination_ratio``.
        """
        W_perp = self.get_W_compositional()
        sigma_perp = W_perp @ W_perp.T + jnp.diag(self.d)
        std = jnp.sqrt(jnp.maximum(jnp.diag(sigma_perp), 1e-30))
        return sigma_perp / (std[:, None] * std[None, :])

