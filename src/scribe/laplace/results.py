"""Results container for Laplace-mode inference (model-agnostic).

This module defines :class:`ScribeLaplaceResults`, the single results
class returned by every Laplace-mode fit regardless of generative
model. The class follows scribe's "one results class per inference
method" pattern (mirroring :class:`ScribeSVIResults`,
:class:`ScribeVAEResults`, :class:`ScribeMCMCResults`): state is held
in a single dataclass, and methods that depend on the generative
model dispatch internally on ``model_config.base_model``.

Per-cell latent state is stored in ``Optional`` slots that the
engine populates depending on which model ran:

* **PLN**  → ``x_loc`` (per-cell log-rate MAP, shape ``(n_cells, G)``),
  optionally ``eta_loc`` for the capture-anchor variant.
* **LNM, d_mode='low_rank'** → ``z_loc`` (factor-score MAP, shape
  ``(n_cells, k)``).
* **LNM, d_mode='learned'** → ``y_alr_loc`` (ALR-logit MAP, shape
  ``(n_cells, G-1)``).
* **LNMVCP** → as LNM plus ``p_capture_loc`` for the per-cell capture
  probability MAP.

Methods come in two flavours:

* **Model-agnostic accessors** (:meth:`get_mu`, :meth:`get_W`,
  :meth:`get_d`, :meth:`get_sigma`, :meth:`get_correlation`,
  :meth:`__getitem__` for gene subsetting, pickle support) — written
  once, no dispatch.
* **Model-dispatching accessors** (:meth:`get_distributions`,
  :meth:`get_ppc_samples`, :meth:`get_predictive_samples`,
  :meth:`get_per_cell_predictive_samples`, :meth:`get_log_likelihood`,
  :meth:`get_latent_embeddings`, :meth:`get_map`) — switch on
  ``model_config.base_model`` and bottom out in module-private
  helpers (``_ppc_pln``, ``_ppc_lnm``, etc.).

Adding a new model (e.g. LNMVCP) requires:

1. The engine populates the right per-cell state slot(s).
2. Add a branch to each model-dispatching method's ``base_model``
   switch and write the corresponding helper.

No new class, no inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..models.config import ModelConfig
from ..stats.distributions import LowRankPoissonLogNormal


# Floor / ceiling used inside ``exp`` to prevent float32 overflow in
# the rate. Matches the Newton kernel's ``_LOG_RATE_MIN`` /
# ``_LOG_RATE_MAX`` so PPCs and the kernel agree on the safety bound.
_LOG_RATE_MIN = -30.0
_LOG_RATE_MAX = 30.0


# =====================================================================
# Dataclass
# =====================================================================


@dataclass
class ScribeLaplaceResults:
    """Results from any Laplace-mode fit.

    Holds the trained generative globals (``mu``, ``W``, ``d``), the
    per-cell MAP for whichever latent the model uses, plus
    diagnostics (loss curve, final per-cell Newton gradient norms,
    early-stopping bookkeeping). Methods that depend on the
    generative model dispatch on ``model_config.base_model``.

    Parameters
    ----------
    model_config : ModelConfig or None
        Configuration the engine ran with. Carries ``base_model``
        and (for LNM-family models) ``d_mode`` — both are read by
        the dispatch in the model-specific methods. Optional only
        for legacy / programmatic construction; in practice always
        populated by ``scribe.fit``.
    mu : jnp.ndarray
        Decoder bias. Shape ``(G,)`` for PLN; ``(G-1,)`` for LNM
        (ALR coordinates).
    W : jnp.ndarray
        Decoder loadings. Shape ``(G, k)`` or ``(G-1, k)``. Together
        with ``d`` defines the prior covariance
        ``Sigma = W W^T + diag(d)``.
    d : jnp.ndarray
        Diagonal residual variance, constrained ``> 0``. Shape
        matches ``mu``. Always present even when the model was
        configured with ``d_mode='low_rank'`` (in which case the
        engine stores ``d`` as a tiny floor for numerical safety).
    x_loc : jnp.ndarray, optional
        PLN per-cell log-rate MAP, shape ``(n_cells, G)``. Populated
        only by PLN fits.
    eta_loc : jnp.ndarray, optional
        PLN per-cell capture-offset MAP, shape ``(n_cells,)``.
        Populated only when the PLN fit had a capture anchor.
    z_loc : jnp.ndarray, optional
        LNM (``d_mode='low_rank'``) per-cell factor-score MAP,
        shape ``(n_cells, k)``.
    y_alr_loc : jnp.ndarray, optional
        LNM (``d_mode='learned'``) per-cell ALR-logit MAP, shape
        ``(n_cells, G-1)``.
    p_capture_loc : jnp.ndarray, optional
        LNMVCP per-cell capture-probability MAP, shape ``(n_cells,)``.
    alr_reference_idx : int, optional
        Zero-based index of the gene that serves as the ALR
        reference (denominator). Populated for LNM/LNMVCP fits;
        unused for PLN.
    final_grad_norms : jnp.ndarray
        Final per-cell L∞ Newton gradient norm. Shape ``(n_cells,)``.
        Convergence diagnostic — anything above
        ``LaplaceConfig.newton_tolerance`` indicates a cell whose
        MAP did not converge in the allotted Newton iterations.
    losses : jnp.ndarray
        Outer-loop loss history.
    n_genes : int
        Number of genes (after any ``gene_coverage`` filtering, which
        may have introduced a trailing ``_other`` pseudo-gene).
    n_cells : int
        Number of cells.
    var : Any, optional
        AnnData-style ``var`` DataFrame attached at results-building
        time. ``None`` for results created outside ``scribe.fit``.
    obs : Any, optional
        AnnData-style ``obs`` DataFrame.
    n_obs, n_vars : int, optional
        Convenience aliases mirroring ``ScribeVAEResults``.
    early_stopped : bool, default False
        Whether early stopping triggered on the outer loop.
    best_loss : float, default ``inf``
        Best smoothed loss observed during training.
    stopped_at_step : int, default 0
        Number of outer iterations actually executed (may be less
        than ``laplace_config.n_steps`` when early-stopping fired).
    metadata : dict
        Free-form metadata bag for downstream plotting and analysis
        helpers; also backs the ``predictive_samples`` and
        ``posterior_samples`` properties.

    Notes
    -----
    The dataclass is **not** subclassed per model. Each Laplace-
    supported model populates a distinct per-cell-state slot and
    leaves the rest as ``None``. The model-agnostic methods read
    only ``mu``/``W``/``d``; the model-dispatching methods read
    the right slot based on ``model_config.base_model``.

    Examples
    --------
    >>> # PLN fit
    >>> result = scribe.fit(adata, model="pln",
    ...                     inference_method="laplace")
    >>> result.x_loc.shape       # (n_cells, G)
    >>> result.z_loc is None     # True
    >>> result.get_mu().shape    # (G,)

    >>> # LNM fit (low_rank)
    >>> result = scribe.fit(adata, model="lnm", d_mode="low_rank",
    ...                     inference_method="laplace")
    >>> result.x_loc is None     # True
    >>> result.z_loc.shape       # (n_cells, k)
    >>> result.get_mu().shape    # (G-1,)
    """

    # --- Required generative globals ---
    model_config: Optional[ModelConfig]
    mu: jnp.ndarray
    W: jnp.ndarray
    d: jnp.ndarray

    # --- Diagnostics (always present) ---
    final_grad_norms: jnp.ndarray
    losses: jnp.ndarray
    n_genes: int
    n_cells: int

    # --- Per-cell latent state (model-specific; exactly one is
    # populated by the engine, the rest are None) ---
    x_loc: Optional[jnp.ndarray] = None  # PLN: (n_cells, G)
    eta_loc: Optional[jnp.ndarray] = None  # PLN capture anchor: (n_cells,)
    z_loc: Optional[jnp.ndarray] = None  # LNM low_rank: (n_cells, k)
    y_alr_loc: Optional[jnp.ndarray] = None  # LNM learned: (n_cells, G-1)
    p_capture_loc: Optional[jnp.ndarray] = None  # LNMVCP capture: (n_cells,)
    alr_reference_idx: Optional[int] = None  # LNM/LNMVCP only

    # NB-on-totals parameters for the LNM family (mean-NB
    # parameterization: ``E[u_T] = mu_T`` for plain LNM,
    # ``E[u_T | eta_c] = mu_T·exp(-eta_c)`` for LNMVCP, with shape
    # parameter ``r_T``). These reflect the FULL LNM generative
    # model from paper/_logistic_normal_multinomial.qmd; without
    # them, sampling fell back to a placeholder ``total_counts=1000``
    # in PPCs (which silently mismatched the data). Both are scalar
    # JAX arrays.
    mu_T: Optional[jnp.ndarray] = None
    r_T: Optional[jnp.ndarray] = None

    # --- AnnData-style metadata; populated by ``scribe.fit`` post-run
    # alongside the standard VAE/SVI pickles ---
    var: Optional[Any] = None
    obs: Optional[Any] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

    # --- Gene-coverage metadata (parity with ``ScribeVAEResults``) ---
    _gene_coverage: Optional[float] = None
    _gene_coverage_mask: Optional[np.ndarray] = None
    _excluded_gene_names: Optional[List[str]] = None
    _original_n_genes: Optional[int] = None
    _total_count_max: Optional[int] = None

    # --- Early-stopping diagnostics (mirroring ``SVIRunResult``) ---
    early_stopped: bool = False
    best_loss: float = float("inf")
    stopped_at_step: int = 0
    # True when the run aborted early because the divergence
    # detector tripped (the loss climbed sustainedly from its
    # running minimum). The result still carries a usable fit:
    # the engine restores the best snapshot from before the
    # divergence before constructing the result. Programmatic
    # callers can branch on this field to retry with tighter
    # ``LaplaceConfig`` knobs (n_newton_steps, damping) or
    # accept the partial result.
    divergence_aborted: bool = False

    # --- Subset-bookkeeping (populated only by __getitem__) ---
    _subset_gene_index: Optional[np.ndarray] = None

    # --- Free-form metadata for plotting helpers / future use ---
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ==================================================================
    # Model-agnostic accessors
    # ==================================================================

    def get_mu(self) -> jnp.ndarray:
        """Decoder bias ``mu``.

        Returns
        -------
        jnp.ndarray
            Shape ``(G,)`` for PLN; ``(G-1,)`` for LNM (ALR).
        """
        return self.mu

    def get_W(self) -> jnp.ndarray:
        """Decoder loadings ``W``.

        Returns
        -------
        jnp.ndarray
            Shape ``(G, k)`` for PLN; ``(G-1, k)`` for LNM.
        """
        return self.W

    def get_d(self) -> jnp.ndarray:
        """Diagonal residual variance ``d`` (constrained ``> 0``).

        Returns
        -------
        jnp.ndarray
            Same leading-axis shape as ``mu``.
        """
        return self.d

    def get_sigma(self) -> jnp.ndarray:
        """Full prior covariance ``Sigma = W W^T + diag(d)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(G, G)`` for PLN; ``(G-1, G-1)`` for LNM.
        """
        return self.W @ self.W.T + jnp.diag(self.d)

    def get_correlation(self) -> jnp.ndarray:
        """Gene-gene correlation matrix derived from ``Sigma``.

        Returns
        -------
        jnp.ndarray
            Same shape as :meth:`get_sigma`. The 1e-30 floor inside
            ``sqrt`` guards against zero-variance genes producing
            ``nan`` correlations.
        """
        sigma = self.get_sigma()
        std = jnp.sqrt(jnp.maximum(jnp.diag(sigma), 1e-30))
        return sigma / (std[:, None] * std[None, :])

    def get_library_size_direction(self) -> jnp.ndarray:
        """Latent-space unit vector 𝒆 ∈ ℝᵏ whose 𝑊-image is closest to 𝟏_G.

        Thin wrapper over
        :func:`scribe.stats.correlation_diagnostics.library_size_direction`
        — see that helper for the math and structural-redundancy
        rationale.

        Returns
        -------
        jnp.ndarray, shape ``(k,)``
        """
        from ..stats.correlation_diagnostics import library_size_direction
        return library_size_direction(self.W)

    def get_correlation_residual(
        self,
        method: str = "library_size",
        n_components: int = 1,
        include_diagonal_d: bool = False,
    ) -> jnp.ndarray:
        """Correlation matrix after projecting out latent direction(s).

        Thin wrapper over
        :func:`scribe.stats.correlation_diagnostics.correlation_residual`
        — see that helper for the math.

        Parameters
        ----------
        method : {"library_size", "pc"}, default "library_size"
        n_components : int, default 1
            Used only when ``method="pc"``.
        include_diagonal_d : bool, default False
            Whether to add ``diag(d)`` before normalising.

        Returns
        -------
        jnp.ndarray
            Correlation matrix of the same shape as
            :meth:`get_correlation`.
        """
        from ..stats.correlation_diagnostics import correlation_residual
        return correlation_residual(
            self.W, self.d,
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
        """Print and return a diagnostic summary of the latent correlation structure.

        Computes a small set of quantities that together characterise
        whether the model's correlation matrix is dominated by a
        library-size axis (the typical PLN-with-weak-capture pathology),
        by a single non-library latent direction, or by genuinely
        diffuse multi-block structure.

        The four quantities of interest:

        * **Library-size alignment**:
          ``cos(W𝒆, 𝟏_G)`` — how close to 𝟏_G the column-space
          projection of 𝟏_G is. Close to 1 ⇒ the library-size axis
          lives cleanly in the column space of 𝑊. Close to 0 ⇒
          the model has put library-size somewhere outside 𝑊
          (typically in η, when the capture term is doing its job).
        * **Library-size loading concentration**:
          ``|mean(W𝒆)| / std(W𝒆)`` — high ⇒ the gene loadings on
          the library-size axis are uniform across genes (the
          "all-genes-shift" signature). Low ⇒ the projection is
          contaminated by gene-specific variation.
        * **Library-size share of 𝑊**:
          ``‖W𝒆‖² / ‖W Wᵗ‖_F`` — fraction of the model-implied
          covariance Frobenius norm that the library-size rank-1
          contribution accounts for. Small ⇒ projecting it out will
          leave the heatmap mostly unchanged.
        * **Latent eigenspectrum**: top-k eigenvalues of 𝑊ᵗ𝑊 and
          the cumulative variance explained. A heavy-tailed
          spectrum (one or two dominant eigenvalues) suggests
          ``subtract_direction="pc"`` will produce a markedly
          different heatmap.

        Parameters
        ----------
        n_top_eig : int, default 10
            Number of top eigenvalues of ``W^T W`` to report.
        verbose : bool, default True
            Print a rich-formatted summary to the console. Set to
            ``False`` for silent return of the diagnostics dict
            (useful when calling from a notebook cell that already
            has its own display logic).

        Returns
        -------
        dict
            Keys: ``cos_We_1G``, ``We_concentration``,
            ``library_axis_share``, ``We_rms``, ``eigenvalues``,
            ``eigenvalue_fractions``, ``effective_rank``,
            ``offdiag_quantiles_full``,
            ``offdiag_quantiles_after_library``,
            ``offdiag_quantiles_after_pc1``.
        """
        from ..stats.correlation_diagnostics import (
            summarize_correlation_structure as _summarize,
        )
        bm = _base_model(self.model_config)
        space = "ALR space" if bm in ("lnm", "lnmvcp") else "log-rate space"
        return _summarize(
            self.W, self.d,
            space_label=space,
            model_label=f"{bm.upper()} Laplace",
            n_top_eig=n_top_eig,
            verbose=verbose,
        )

    def get_p_capture(self) -> Optional[jnp.ndarray]:
        """Per-cell capture probability ``p_c ∈ (0, 1]``.

        Two paths produce a per-cell capture quantity in scribe:

        * **PLN with capture anchor** stores ``eta_loc`` (the
          additive log offset against the prior anchor); the
          capture probability is ``exp(-eta_loc)``.
        * **LNMVCP** stores ``p_capture_loc`` directly as a per-cell
          MAP in the unit interval.

        Returns
        -------
        jnp.ndarray or None
            Per-cell capture probability, or ``None`` when neither
            slot is populated (e.g. PLN without capture anchor, or
            LNM without VCP).
        """
        if self.p_capture_loc is not None:
            return self.p_capture_loc
        if self.eta_loc is not None:
            return jnp.exp(-self.eta_loc)
        return None

    # ==================================================================
    # Model-dispatching accessors
    # ==================================================================

    def get_latent_embeddings(self) -> jnp.ndarray:
        """Per-cell latent embedding suitable for UMAP / clustering.

        Returns the natural per-cell representation for the fitted
        model:

        * **PLN** → ``x_loc`` (per-cell log-rate, ``(n_cells, G)``).
        * **LNM** with ``d_mode='low_rank'`` → ``z_loc`` (factor
          scores, ``(n_cells, k)``).
        * **LNM** with ``d_mode='learned'`` → ``y_alr_loc``
          (ALR logits, ``(n_cells, G-1)``).

        Mirrors ``ScribeVAEResults.get_latent_embeddings`` so
        downstream UMAP / clustering helpers run on either result
        type. For Laplace this is the per-cell MAP rather than a
        posterior sample, but the shape and semantic role match.

        Returns
        -------
        jnp.ndarray
            Per-cell embeddings; shape depends on the model.

        Raises
        ------
        NotImplementedError
            If ``model_config.base_model`` is unrecognised.
        """
        bm = _base_model(self.model_config)
        if bm == "pln":
            return self.x_loc
        if bm in ("lnm", "lnmvcp"):
            if self.z_loc is not None:
                return self.z_loc
            return self.y_alr_loc
        raise NotImplementedError(
            f"get_latent_embeddings not implemented for base_model={bm!r}"
        )

    def get_map(self, **_kwargs) -> Dict[str, jnp.ndarray]:
        """Return a dict of point estimates suitable for plotting.

        Mirrors ``ScribeVAEResults.get_map`` so calibration /
        diagnostic plotters consume MAP dicts uniformly across
        result types.

        Parameters
        ----------
        **_kwargs
            Accepted for VAE-API compatibility; ignored. Laplace
            results are point estimates by design — no flow or
            canonicalisation knobs apply.

        Returns
        -------
        dict
            Keys depend on the model:

            * **PLN**: ``{"mu", "W", "d_pln", "y_log_rate"}`` plus
              ``{"eta_capture", "p_capture"}`` when the capture
              anchor was active. ``y_log_rate`` is the per-cell
              ``x_loc`` (calibration plotters detect the leading
              axis).
            * **LNM**: ``{"mu", "W", "d_lnm", "z"}`` (low_rank) or
              ``{"mu", "W", "d_lnm", "y_alr"}`` (learned), plus
              ``{"p_capture"}`` for LNMVCP.

        Raises
        ------
        NotImplementedError
            If ``model_config.base_model`` is unrecognised.
        """
        bm = _base_model(self.model_config)
        if bm == "pln":
            out: Dict[str, jnp.ndarray] = {
                "mu": self.mu,
                "W": self.W,
                "d_pln": self.d,
                "y_log_rate": self.x_loc,
            }
            if self.eta_loc is not None:
                out["eta_capture"] = self.eta_loc
                out["p_capture"] = jnp.exp(-self.eta_loc)
            return out
        if bm in ("lnm", "lnmvcp"):
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
            return out
        raise NotImplementedError(
            f"get_map not implemented for base_model={bm!r}"
        )

    def get_distributions(
        self, backend: str = "numpyro", **_kwargs
    ) -> Dict[str, Any]:
        """Return prior distributions for downstream PPC / plotting.

        Always returns the *population* (prior-predictive)
        distribution, not the per-cell posterior — for the latter,
        use :meth:`get_per_cell_predictive_samples`.

        Parameters
        ----------
        backend : {"numpyro"}, default "numpyro"
            Currently only NumPyro is supported.
        **_kwargs
            Accepted for compatibility; ignored.

        Returns
        -------
        dict
            * **PLN**: ``{"y_log_rate": LowRankMultivariateNormal,
              "lambda_rate": LowRankPoissonLogNormal}``.
            * **LNM**: ``{"y_alr": LowRankMultivariateNormal}``
              over the ALR latent. (A registered
              ``LowRankMultinomialLogisticNormal`` distribution
              would slot in cleanly here when implemented.)

        Raises
        ------
        ValueError
            If ``backend`` is not supported.
        NotImplementedError
            If ``model_config.base_model`` is unrecognised.
        """
        if backend != "numpyro":
            raise ValueError(
                "Only 'numpyro' backend supported for Laplace results."
            )
        import numpyro.distributions as dist

        bm = _base_model(self.model_config)
        if bm == "pln":
            return {
                "y_log_rate": dist.LowRankMultivariateNormal(
                    loc=self.mu, cov_factor=self.W, cov_diag=self.d
                ),
                "lambda_rate": LowRankPoissonLogNormal(
                    loc=self.mu, cov_factor=self.W, cov_diag=self.d
                ),
            }
        if bm in ("lnm", "lnmvcp"):
            # Until a registered LowRankMultinomialLogisticNormal
            # exists, surface the Gaussian over y_alr — downstream
            # multinomial sampling can be done by softmax + Multinomial.
            return {
                "y_alr": dist.LowRankMultivariateNormal(
                    loc=self.mu, cov_factor=self.W, cov_diag=self.d
                ),
            }
        raise NotImplementedError(
            f"get_distributions not implemented for base_model={bm!r}"
        )

    def get_ppc_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        per_cell: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """Draw posterior predictive samples.

        Two modes, controlled by ``per_cell``:

        * ``per_cell=False`` (default — population PPC): draw
          ``n_samples`` cells from the *population* (prior-
          predictive) distribution. Captures what the model
          predicts for a *new* cell drawn from the prior — the
          natural diagnostic for "does the population-level model
          match the data?"
        * ``per_cell=True``: per-cell posterior PPC using the
          stored per-cell MAP. For each cell ``c``, draw
          ``n_samples`` count vectors conditioned on its MAP.
          Analogous to NumPyro's ``Predictive`` under guide replay
          for the VAE path.

        Parameters
        ----------
        rng_key : jax.Array, optional
            PRNG key. Defaults to ``random.PRNGKey(0)`` for
            reproducibility.
        n_samples : int, default 100
            Number of samples. Interpretation depends on
            ``per_cell``: total cells in population mode, samples
            per cell in per-cell mode.
        per_cell : bool, default False
            See above.
        **kwargs
            For LNM-family models: ``total_counts`` (per-cell
            scalar or ``(n_samples,)``) is passed through to the
            multinomial-sampling helper; defaults to a per-cell
            sum derived from the data when omitted.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_samples, G)`` in population mode and
            ``(n_samples, n_cells, G)`` in per-cell mode.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        if per_cell:
            return self.get_per_cell_predictive_samples(
                rng_key=rng_key, n_samples=n_samples, **kwargs
            )

        bm = _base_model(self.model_config)
        if bm == "pln":
            return _ppc_pln_population(
                rng_key, n_samples, self.mu, self.W, self.d
            )
        if bm in ("lnm", "lnmvcp"):
            return _ppc_lnm_population(
                rng_key,
                n_samples,
                self.mu,
                self.W,
                self.d,
                self.alr_reference_idx,
                mu_T=self.mu_T,
                r_T=self.r_T,
                **kwargs,
            )
        raise NotImplementedError(
            f"get_ppc_samples not implemented for base_model={bm!r}"
        )

    def get_predictive_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        **kwargs,
    ) -> jnp.ndarray:
        """Alias for :meth:`get_ppc_samples` in population mode.

        Mirrors the SVI/VAE ``get_predictive_samples`` naming.
        """
        return self.get_ppc_samples(
            rng_key=rng_key,
            n_samples=n_samples,
            per_cell=False,
            **kwargs,
        )

    def get_per_cell_predictive_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        **kwargs,
    ) -> jnp.ndarray:
        """Per-cell posterior predictive samples (Laplace-uncertainty aware).

        For each cell, draws ``n_samples`` predictive count vectors
        from the per-cell Laplace posterior on the *composition
        latent* (``x`` for PLN, ``z`` or ``y_alr`` for
        LNM(VCP)) — sampling
        ``N(MAP, (-H_c)^{-1})`` from the Woodbury / Sherman-Morrison
        machinery in :mod:`scribe.laplace._newton_pln` /
        :mod:`scribe.laplace._newton_lnm` — and then drawing the
        count vector conditional on that latent.

        This propagates two sources of stochasticity (per-cell
        composition-latent posterior uncertainty + observation
        noise), unlike :meth:`get_map_ppc_samples` which holds the
        latents at their point estimates. The capture offset
        ``eta_c`` (PLN) is held at its MAP rather than sampled
        jointly with the composition latent — see Notes.

        Parameters
        ----------
        rng_key : jax.Array, optional
        n_samples : int, default 100
        **kwargs
            For LNM-family models: ``total_counts`` and ``counts``
            (when conditioning on observed totals).

        Returns
        -------
        jnp.ndarray, shape (n_samples, n_cells, G)

        See Also
        --------
        :meth:`get_map_ppc_samples`
            Cheaper MAP-only path that does not propagate latent
            posterior uncertainty. Use when you only need a quick
            sanity check or when the per-cell Hessian sampling is
            too expensive for the dataset at hand.

        Notes
        -----
        For PLN with the capture anchor active, the joint
        ``(x, eta)`` posterior has a near-singular direction along
        the rigid-translation null direction
        ``(mu, eta) -> (mu + Delta, eta + Delta)``; sampling
        ``eta`` jointly would amplify variance along that direction
        without injecting biologically meaningful uncertainty.
        Sampling ``x`` conditional on the MAP ``eta`` therefore
        produces a well-conditioned predictive distribution. This
        is the same compositional-robustness logic used in the
        robustness theorem (see ``paper/_diffexp_lnm_pln_robustness.qmd``).
        For LNMVCP fits, the joint Hessian on ``(z, eta)`` is
        block-diagonal and ``eta`` does not appear in the
        composition multinomial, so the same procedure is *exact*
        for the conditional PPC and only the optional unconditional
        path would re-draw ``eta``.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        bm = _base_model(self.model_config)
        if bm == "pln":
            return _ppc_pln_per_cell_laplace(
                rng_key,
                n_samples,
                self.x_loc,
                self.eta_loc,
                self.W,
                self.d,
            )
        if bm in ("lnm", "lnmvcp"):
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
                **kwargs,
            )
        raise NotImplementedError(
            f"get_per_cell_predictive_samples not implemented for "
            f"base_model={bm!r}"
        )

    def get_map_ppc_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        **kwargs,
    ) -> jnp.ndarray:
        """Per-cell MAP-only PPC samples (no Laplace posterior uncertainty).

        For each cell, draws ``n_samples`` predictive count vectors
        with the per-cell latents *held fixed at their MAP*. The
        only stochasticity is the observation-noise step (Poisson
        for PLN, Multinomial-on-totals for LNM(VCP)). This is the
        cheap analogue of the level-2 PPC produced by
        :meth:`get_per_cell_predictive_samples` and is the closest
        Laplace-side equivalent to the MCMC ``get_map_ppc_samples``
        path.

        When to prefer which:

        * **Cheap sanity check** (this method): no Hessian solves,
          one observation-noise draw per (sample, cell). Useful for
          a quick "does the likelihood shape match the data?"
          visual.
        * **Honest posterior predictive**
          (:meth:`get_per_cell_predictive_samples`): samples the
          per-cell latent from its Laplace posterior
          ``N(MAP, (-H_c)^{-1})`` before drawing the observation,
          so it captures both sources of stochasticity. Use this
          when the calibration of the model's *uncertainty* matters
          (e.g., variance-based PPC quantile checks).

        Parameters and return shape match
        :meth:`get_per_cell_predictive_samples`.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        bm = _base_model(self.model_config)
        if bm == "pln":
            return _ppc_pln_per_cell(
                rng_key, n_samples, self.x_loc, self.eta_loc
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
        raise NotImplementedError(
            f"get_map_ppc_samples not implemented for " f"base_model={bm!r}"
        )

    def get_log_likelihood(
        self,
        counts: jnp.ndarray,
        return_by: str = "cell",
    ) -> jnp.ndarray:
        """Per-cell or per-gene log-likelihood at the MAP.

        Evaluates the conditional likelihood of ``counts`` under
        the per-cell MAP. The likelihood family depends on the
        model:

        * **PLN** → Poisson with rate ``exp(x_c - eta_c)``.
        * **LNM** → Multinomial with probabilities
          ``softmax(mu + W z_c)`` (or ``softmax(y_alr_c)`` when
          ``d_mode='learned'``), conditioned on the cell's total
          count.

        Mirrors ``LikelihoodMixin.get_log_likelihood`` from
        ``ScribeVAEResults`` so the same downstream code can call
        this on either result type.

        Parameters
        ----------
        counts : jnp.ndarray, shape (n_cells, G)
            Observed counts (typically the training data).
        return_by : {"cell", "gene"}, default "cell"
            Reduction axis. ``"cell"`` sums over genes →
            ``(n_cells,)``; ``"gene"`` sums over cells → ``(G,)``.

            .. note::
                The ``"gene"`` reduction is well-defined only for
                Poisson (PLN) likelihoods. The multinomial log-pmf
                couples genes through a single ``log_softmax`` term;
                summing over cells is fine but the result is a
                *contribution* to the total log-likelihood, not a
                per-gene Poisson-style score.

        Returns
        -------
        jnp.ndarray
            Per-cell or per-gene log-likelihood.

        Raises
        ------
        ValueError
            If ``return_by`` is not ``"cell"`` or ``"gene"``.
        NotImplementedError
            If ``model_config.base_model`` is unrecognised.
        """
        if return_by not in ("cell", "gene"):
            raise ValueError(
                f"return_by must be 'cell' or 'gene'; got {return_by!r}."
            )
        bm = _base_model(self.model_config)
        if bm == "pln":
            return _ll_pln(counts, self.x_loc, self.eta_loc, return_by)
        if bm in ("lnm", "lnmvcp"):
            return _ll_lnm(
                counts,
                self.mu,
                self.W,
                self.z_loc,
                self.y_alr_loc,
                self.alr_reference_idx,
                return_by,
            )
        raise NotImplementedError(
            f"get_log_likelihood not implemented for base_model={bm!r}"
        )

    # ==================================================================
    # Gene subsetting
    # ==================================================================

    def __getitem__(
        self,
        gene_index: Union[int, slice, np.ndarray, jnp.ndarray, list],
    ) -> "ScribeLaplaceResults":
        """Subset to a gene index (slice / fancy indexing).

        Returns a new ``ScribeLaplaceResults`` with all gene-axis
        arrays sliced. Cell-level fields (``losses``,
        ``final_grad_norms``, ``n_cells``, ``eta_loc``,
        ``p_capture_loc``) are unchanged.

        Subsetting LNM results is more delicate than PLN because
        the natural latent coordinate is ALR (``G-1``), and the
        ALR reference gene must be preserved. For LNM we therefore
        require that the user pass an index that **includes** the
        reference gene, and ``y_alr_loc`` is sliced over the
        ``G-1`` axis (which excludes the reference) using the
        adjusted index.

        Parameters
        ----------
        gene_index : int, slice, ndarray, list
            Anything ``np.asarray(...)`` can interpret as a gene
            index. A bare ``int`` is wrapped in a 1-element array
            so the resulting object still has a gene dimension
            (matches ``ScribeVAEResults`` behaviour).

        Returns
        -------
        ScribeLaplaceResults
            New result with sliced gene-axis arrays.

        Raises
        ------
        ValueError
            For LNM results, if ``gene_index`` does not include the
            ALR reference gene.
        NotImplementedError
            If ``model_config.base_model`` is unrecognised.
        """
        # Normalise the index to a 1-D integer array.
        if isinstance(gene_index, (int, np.integer)):
            idx = np.asarray([int(gene_index)])
        elif isinstance(gene_index, slice):
            idx = np.asarray(range(*gene_index.indices(self.n_genes)))
        else:
            idx = np.asarray(gene_index)
        if idx.dtype == bool:
            idx = np.where(idx)[0]

        bm = _base_model(self.model_config)
        if bm == "pln":
            return self._subset_pln(idx)
        if bm in ("lnm", "lnmvcp"):
            return self._subset_lnm(idx)
        raise NotImplementedError(
            f"__getitem__ not implemented for base_model={bm!r}"
        )

    def _subset_pln(self, idx: np.ndarray) -> "ScribeLaplaceResults":
        """PLN-specific gene subsetting (``mu``, ``W``, ``d``, ``x_loc``)."""
        idx_jnp = jnp.asarray(idx)
        return replace(
            self,
            mu=self.mu[idx_jnp],
            W=self.W[idx_jnp, :],
            d=self.d[idx_jnp],
            x_loc=self.x_loc[:, idx_jnp] if self.x_loc is not None else None,
            n_genes=int(len(idx)),
            n_vars=int(len(idx)) if self.n_vars is not None else None,
            var=_subset_var(self.var, idx),
            _subset_gene_index=idx,
        )

    def _subset_lnm(self, idx: np.ndarray) -> "ScribeLaplaceResults":
        """LNM-specific gene subsetting.

        ALR coordinates leave one gene out (the reference). The
        passed-in ``idx`` is interpreted in the *original* G-gene
        space; we slice the ``(G-1)``-shaped arrays
        (``mu``, ``W``, ``d``, ``y_alr_loc``) using the index
        positions that survive after removing the reference.

        For now we conservatively require that ``idx`` includes the
        reference gene — partial subsets that drop the reference
        would change which gene is the ALR denominator and require
        re-projecting the ALR latent into the new coordinate
        system, which is non-trivial. ``z_loc`` is gene-axis-
        invariant (factor scores) so it carries through unchanged.
        """
        ref = self.alr_reference_idx
        if ref is None:
            raise ValueError(
                "LNM subsetting needs alr_reference_idx; this result "
                "appears to have been constructed without it."
            )
        if ref not in set(idx.tolist()):
            raise ValueError(
                f"Gene subset must include the ALR reference gene "
                f"(index={ref!r}); got idx={idx!r}."
            )
        # Map original-G indices to (G-1)-axis positions.
        # alr_axis_pos[i] = position of original gene i along
        # the (G-1)-axis (None for the reference).
        n_g = self.n_genes
        alr_axis_pos = np.full(n_g, -1, dtype=int)
        alr_axis_pos[:ref] = np.arange(ref)
        alr_axis_pos[ref + 1 :] = np.arange(ref, n_g - 1)
        # Indices into the (G-1) axis, in the user-supplied order,
        # excluding the reference.
        idx_no_ref = idx[idx != ref]
        idx_alr = alr_axis_pos[idx_no_ref]
        idx_alr_jnp = jnp.asarray(idx_alr)
        # The new reference position in the subset is wherever
        # ``ref`` lands in ``idx``.
        new_ref_pos = int(np.where(idx == ref)[0][0])

        return replace(
            self,
            mu=self.mu[idx_alr_jnp],
            W=self.W[idx_alr_jnp, :],
            d=self.d[idx_alr_jnp],
            # z_loc is gene-axis-invariant; copy through.
            z_loc=self.z_loc,
            y_alr_loc=(
                self.y_alr_loc[:, idx_alr_jnp]
                if self.y_alr_loc is not None
                else None
            ),
            alr_reference_idx=new_ref_pos,
            n_genes=int(len(idx)),
            n_vars=int(len(idx)) if self.n_vars is not None else None,
            var=_subset_var(self.var, idx),
            _subset_gene_index=idx,
        )

    # ==================================================================
    # Convenience aliases / properties
    # ==================================================================

    @property
    def predictive_samples(self) -> Optional[jnp.ndarray]:
        """Cached predictive samples (used by plotting helpers).

        ``ScribeVAEResults`` exposes a ``predictive_samples``
        attribute that PPC plotters populate via
        ``_get_predictive_samples_for_plot``. We expose the same
        name here so the dispatch helpers can read/write predictive
        samples on this object too.
        """
        return self.metadata.get("predictive_samples")

    @predictive_samples.setter
    def predictive_samples(self, value: Optional[jnp.ndarray]) -> None:
        if value is None:
            self.metadata.pop("predictive_samples", None)
        else:
            self.metadata["predictive_samples"] = value

    @property
    def posterior_samples(self) -> Optional[Dict[str, jnp.ndarray]]:
        """Cached posterior samples slot (for plotter compatibility)."""
        return self.metadata.get("posterior_samples")

    @posterior_samples.setter
    def posterior_samples(
        self, value: Optional[Dict[str, jnp.ndarray]]
    ) -> None:
        if value is None:
            self.metadata.pop("posterior_samples", None)
        else:
            self.metadata["posterior_samples"] = value

    # ==================================================================
    # Pickle support
    # ==================================================================

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle-safe state.

        ``model_config`` may contain non-picklable Linen modules in
        ``model_config.vae``; we strip those for pickle and let the
        next reload re-build them on demand. Mirrors
        ``ScribeVAEResults.__getstate__`` behaviour.
        """
        from ..svi.vae_results import make_model_config_pickle_safe

        state = dict(self.__dict__)
        state["model_config"] = make_model_config_pickle_safe(
            state.get("model_config")
        )
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)


# =====================================================================
# Module-private helpers (model-specific dispatch backends)
# =====================================================================


def _base_model(model_config: Optional[ModelConfig]) -> str:
    """Extract ``base_model`` from a ``ModelConfig``, defaulting to PLN.

    Falls back to ``"pln"`` when ``model_config`` is ``None`` (e.g.
    legacy / programmatic construction) so the dispatch on a
    minimal ``ScribeLaplaceResults`` instance still works.
    """
    if model_config is None:
        return "pln"
    return getattr(model_config, "base_model", "pln")


def _subset_var(var: Optional[Any], idx: np.ndarray) -> Optional[Any]:
    """Slice an AnnData ``var`` DataFrame; pass through if not a DataFrame."""
    if var is None:
        return None
    try:
        return var.iloc[idx]
    except (TypeError, AttributeError):
        return var


# ------ PLN PPC helpers ------


def _ppc_pln_population(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    """Population-level Poisson-LogNormal PPC.

    Draws ``n_samples`` cells from the prior-predictive
    ``LowRankPoissonLogNormal(mu, W, d)``. Used as the diagnostic
    for "does the population-level fit reproduce the marginal
    count distribution?"
    """
    pln = LowRankPoissonLogNormal(loc=mu, cov_factor=W, cov_diag=d)
    return pln.sample(rng_key, sample_shape=(n_samples,))


def _ppc_pln_per_cell(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Per-cell Poisson PPC at the stored MAP (no posterior uncertainty).

    Effective per-cell log-rate is ``x_c - eta_c`` (or ``x_c``
    when no capture anchor); broadcast to ``(n_samples, n_cells,
    G)`` and Poisson-sample. **Does not propagate per-cell Laplace
    posterior uncertainty** — every PPC sample uses the same
    point-estimate latents, so the only stochasticity is the
    Poisson draw. Use :func:`_ppc_pln_per_cell_laplace` when
    posterior uncertainty matters; this MAP-only path is exposed
    via :meth:`ScribeLaplaceResults.get_map_ppc_samples`.
    """
    if eta_loc is not None:
        eff_log_rate = x_loc - eta_loc[:, None]
    else:
        eff_log_rate = x_loc
    eff_log_rate = jnp.clip(eff_log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
    rate = jnp.exp(eff_log_rate)

    # Batched sampling — same memory consideration as the LNM
    # per-cell path. Poisson sampling has a smaller intermediate
    # than multinomial (no n_max categorical expansion), so this
    # mostly bounds the *output* allocation per chunk.
    def _sample_chunk(chunk_key: jax.Array, size: int) -> jnp.ndarray:
        rate_b = jnp.broadcast_to(rate, (size,) + rate.shape)
        return jax.random.poisson(chunk_key, rate_b)

    return _batched_sample_concat(rng_key, n_samples, _sample_chunk)


def _ppc_pln_per_cell_laplace(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    W: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    """Per-cell PLN PPC propagating composition-latent Laplace uncertainty.

    For each PPC sample ``s`` and each cell ``c``:

    1. Draw ``x_c^(s) ~ N(x_loc[c], (-H_xx_c)^(-1))`` from the
       per-cell Laplace posterior on the composition latent
       ``x``, *conditional on* the MAP capture offset
       ``hat_eta_c``. The covariance is computed in closed form
       from the Woodbury factors at the converged MAP — see
       :func:`scribe.laplace._newton_pln.sample_x_posterior`.
    2. Sample ``u_c^(s) ~ Poisson(exp(x_c^(s) - hat_eta_c))``.

    This is the *conditional* posterior-predictive PPC for PLN
    Laplace fits — it propagates two sources of stochasticity
    (composition-latent posterior uncertainty + Poisson
    likelihood) but holds ``eta_c`` at its MAP rather than
    sampling the full joint ``(x, eta)`` posterior. See the
    ``Notes`` section below for the structural reason. The cost is
    a per-cell, per-sample Woodbury solve; chunked over PPC
    samples to bound device memory.

    Parameters
    ----------
    rng_key : jax.Array
    n_samples : int
        Number of PPC samples per cell.
    x_loc : jnp.ndarray, shape ``(n_cells, G)``
        Per-cell MAP log-rates.
    eta_loc : jnp.ndarray, shape ``(n_cells,)`` or None
        Per-cell capture offsets (held at MAP — see ``Notes``).
    W : jnp.ndarray, shape ``(G, k)``
    d : jnp.ndarray, shape ``(G,)``

    Returns
    -------
    np.ndarray, shape ``(n_samples, n_cells, G)``

    Notes
    -----
    ``eta_c`` is held at its MAP rather than sampled. The
    rigid-translation degeneracy of the PLN means the joint
    ``(x, eta)`` posterior has a near-singular direction along
    ``(mu, eta) -> (mu + Δ, eta + Δ)``; sampling ``eta`` jointly
    would amplify this near-singularity, while sampling ``x``
    conditionally at the MAP ``eta`` produces a well-conditioned
    Gaussian and matches the dominant per-cell uncertainty
    (since ``eta_c`` is one scalar versus ``G`` components in
    ``x_c``).
    """
    from ._newton_pln import sample_x_posterior_batch

    n_cells = int(x_loc.shape[0])

    if eta_loc is None:
        # No capture anchor — sampler still works; pass zeros so
        # the rate is just ``exp(x_c^(s))``.
        eta_arr = jnp.zeros(n_cells, dtype=x_loc.dtype)
    else:
        eta_arr = eta_loc

    chunk_size = _PPC_DEFAULT_SAMPLE_CHUNK
    if chunk_size is None or chunk_size >= n_samples:
        size = int(n_samples)
        k_x, k_p = jax.random.split(rng_key)
        cell_keys = jax.random.split(k_x, n_cells)
        # ``sample_x_posterior_batch`` returns (n_cells, size, G).
        x_samples = sample_x_posterior_batch(
            cell_keys, x_loc, eta_arr, W, d, size, 0.0
        )
        # Transpose to (size, n_cells, G).
        x_samples = jnp.transpose(x_samples, (1, 0, 2))
        log_rate = x_samples - eta_arr[None, :, None]
        log_rate = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
        rate = jnp.exp(log_rate)
        return np.asarray(jax.random.poisson(k_p, rate))

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        k_x, k_p = jax.random.split(chunk_keys[i])
        cell_keys = jax.random.split(k_x, n_cells)
        x_samples = sample_x_posterior_batch(
            cell_keys, x_loc, eta_arr, W, d, size, 0.0
        )
        x_samples = jnp.transpose(x_samples, (1, 0, 2))
        log_rate = x_samples - eta_arr[None, :, None]
        log_rate = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
        rate = jnp.exp(log_rate)
        pieces.append(np.asarray(jax.random.poisson(k_p, rate)))
    return np.concatenate(pieces, axis=0)


# ------ LNM PPC helpers ------


def _alr_to_softmax(
    y_alr: jnp.ndarray, alr_reference_idx: int, n_genes: int
) -> jnp.ndarray:
    """Insert a zero at ``alr_reference_idx`` and softmax to a probability simplex.

    ``y_alr`` has shape ``(..., G-1)``; output has shape ``(..., G)``.
    The reference gene's logit is conventionally fixed at 0; we
    softmax the augmented G-vector to recover probabilities.
    """
    ref = int(alr_reference_idx)
    # Build a (..., G) array with zero in the reference slot.
    leading = y_alr.shape[:-1]
    full_shape = leading + (n_genes,)
    full = jnp.zeros(full_shape, dtype=y_alr.dtype)
    # Indices along the last axis that are NOT the reference.
    other = list(range(n_genes))
    other.remove(ref)
    full = full.at[..., jnp.asarray(other)].set(y_alr)
    return jax.nn.softmax(full, axis=-1)


def _ppc_lnm_population(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alr_reference_idx: Optional[int],
    mu_T: Optional[jnp.ndarray] = None,
    r_T: Optional[jnp.ndarray] = None,
    total_counts: Optional[Union[int, jnp.ndarray]] = None,
    **_kwargs,
) -> jnp.ndarray:
    """Population-level full LNM PPC.

    Generative process per sample:
      1. ``y_alr ~ N(mu, W W^T + diag(d))``  — composition.
      2. ``u_T ~ NB(r_T, mu_T)``               — total mRNA per cell.
      3. ``u ~ Multinomial(u_T, softmax_full(y_alr))``.

    Step 2 uses the fitted ``(mu_T, r_T)`` from the result. When
    those are absent (legacy callers, or models that do not store
    them) the function falls back to ``total_counts`` (default
    1000), preserving backward-compat for shape-only comparisons.

    Parameters
    ----------
    mu_T, r_T : Optional
        Fitted NB-on-totals globals. When both are provided, totals
        are drawn from ``NB(r_T, mu_T)``; otherwise the function
        falls back to the explicit ``total_counts`` argument.
    total_counts : Optional
        Override for the fallback path. Ignored when ``mu_T`` and
        ``r_T`` are present.
    """
    if alr_reference_idx is None:
        raise ValueError(
            "LNM PPC requires alr_reference_idx; this result "
            "appears to have been constructed without it."
        )
    import numpyro.distributions as dist

    g_minus1 = mu.shape[0]
    n_genes = g_minus1 + 1
    mvn = dist.LowRankMultivariateNormal(loc=mu, cov_factor=W, cov_diag=d)
    k1, k2, k3 = jax.random.split(rng_key, 3)
    y_alr = mvn.sample(k1, sample_shape=(n_samples,))  # (n_samples, G-1)
    p = _alr_to_softmax(y_alr, alr_reference_idx, n_genes)  # (n_samples, G)

    # Total counts: prefer the fitted NB; fall back to scalar.
    if mu_T is not None and r_T is not None:
        # NegativeBinomial2(mean=mu_T, concentration=r_T): variance
        # = mu_T + mu_T^2 / r_T. Per-sample independent draws.
        nb = dist.NegativeBinomial2(
            mean=jnp.asarray(mu_T), concentration=jnp.asarray(r_T)
        )
        n_arr = nb.sample(k2, sample_shape=(n_samples,)).astype(jnp.int32)
    else:
        fallback = 1000 if total_counts is None else total_counts
        n_arr = jnp.broadcast_to(
            jnp.asarray(fallback, dtype=jnp.int32), (n_samples,)
        )

    # Chunked multinomial. ``p`` and ``n_arr`` are lightweight
    # (n_samples × G ≈ 14 MB at 512×7K), but NumPyro's multinomial
    # internally expands ``n_max`` categoricals so the
    # ``(n_samples, n_max, G)`` intermediate balloons. Stream each
    # chunk's output to host so device peak stays bounded.
    chunk_size = _kwargs.get("chunk_size", _PPC_DEFAULT_SAMPLE_CHUNK)
    if chunk_size is None or chunk_size >= n_samples:
        return _multinomial_sample(k3, n_arr, p)

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(k3, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        p_chunk = jax.lax.dynamic_slice_in_dim(p, start, end - start, axis=0)
        n_chunk = jax.lax.dynamic_slice_in_dim(
            n_arr, start, end - start, axis=0
        )
        out_chunk = _multinomial_sample(chunk_keys[i], n_chunk, p_chunk)
        pieces.append(np.asarray(out_chunk))
    return np.concatenate(pieces, axis=0)


def _ppc_lnm_per_cell(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    z_loc: Optional[jnp.ndarray],
    y_alr_loc: Optional[jnp.ndarray],
    alr_reference_idx: Optional[int],
    mu_T: Optional[jnp.ndarray] = None,
    r_T: Optional[jnp.ndarray] = None,
    p_capture_loc: Optional[jnp.ndarray] = None,
    counts: Optional[jnp.ndarray] = None,
    total_counts: Optional[jnp.ndarray] = None,
    **_kwargs,
) -> jnp.ndarray:
    """Per-cell PPC for the full LNM(VCP) generative model.

    For each observed cell ``c``:
      1. Composition logits ``y_alr_c = mu + W z_c`` (low_rank) or
         ``y_alr_c`` directly (learned).
      2. Per-cell totals:
           * **counts provided** → ``u_T_c = sum(counts_c)`` exactly
             (per-cell *conditional* PPC; matches the ``counts``-
             aware path used by the encoder VAE side via
             ``viz.dispatch._get_predictive_samples_for_plot``).
           * **counts None and (mu_T, r_T) provided** → draw
             ``u_T_c ~ NB(r_T, mu_T_c)`` where ``mu_T_c =
             mu_T·p_capture_loc[c]`` for LNMVCP and ``mu_T``
             everywhere for plain LNM.
           * **explicit total_counts** → use those values.
           * **otherwise** fall back to ``1000`` (legacy compat).
      3. Sample ``u_c ~ Multinomial(u_T_c, softmax_full(y_alr_c))``
         independently for each PPC sample.
    """
    if alr_reference_idx is None:
        raise ValueError(
            "LNM PPC requires alr_reference_idx; this result "
            "appears to have been constructed without it."
        )
    import numpyro.distributions as dist

    g_minus1 = mu.shape[0]
    n_genes = g_minus1 + 1

    if y_alr_loc is not None:
        y_alr = y_alr_loc  # (n_cells, G-1)
    elif z_loc is not None:
        y_alr = mu[None, :] + z_loc @ W.T  # (n_cells, G-1)
    else:
        raise ValueError("LNM per-cell PPC requires either z_loc or y_alr_loc.")

    p_per_cell = _alr_to_softmax(y_alr, alr_reference_idx, n_genes)
    n_cells = p_per_cell.shape[0]

    # Resolve per-cell totals — see docstring decision table.
    if counts is not None:
        # Conditional PPC: per-cell total fixed at observed.
        observed_totals = jnp.asarray(counts).sum(axis=-1).astype(jnp.int32)
        n_arr_cells = observed_totals
        # Broadcast across PPC samples (same totals every sample).
        k_pred = rng_key
        n_b = jnp.broadcast_to(n_arr_cells, (n_samples,) + n_arr_cells.shape)
    elif total_counts is not None:
        n_arr_cells = jnp.asarray(total_counts, dtype=jnp.int32)
        k_pred = rng_key
        n_b = jnp.broadcast_to(n_arr_cells, (n_samples,) + n_arr_cells.shape)
    elif mu_T is not None and r_T is not None:
        # Fitted-NB generative PPC with per-cell mean
        # mu_T_c = mu_T (plain LNM) or mu_T * p_capture_c (LNMVCP).
        if p_capture_loc is not None:
            mu_T_per_cell = jnp.asarray(mu_T) * jnp.asarray(p_capture_loc)
        else:
            mu_T_per_cell = jnp.broadcast_to(jnp.asarray(mu_T), (n_cells,))
        nb = dist.NegativeBinomial2(
            mean=mu_T_per_cell, concentration=jnp.asarray(r_T)
        )
        k_nb, k_pred = jax.random.split(rng_key)
        # (n_samples, n_cells) draws; each (sample, cell) gets its
        # own NB draw conditional on the per-cell mean.
        n_b = nb.sample(k_nb, sample_shape=(n_samples,)).astype(jnp.int32)
    else:
        # Legacy fallback.
        n_arr_cells = jnp.full((n_cells,), 1000, dtype=jnp.int32)
        k_pred = rng_key
        n_b = jnp.broadcast_to(n_arr_cells, (n_samples,) + n_arr_cells.shape)

    # Batched sampling. NumPyro's multinomial expands n_max
    # categoricals internally, which scales as ``chunk · n_cells ·
    # n_max`` for the intermediate. Without chunking, the full
    # ``(n_samples, n_cells, G)`` output PLUS that intermediate
    # easily exceeds GPU memory on real scRNA-seq dimensions
    # (n_samples=512, n_cells=3K, G=7K ≈ 42 GB output). We sample
    # in chunks of ``_PPC_DEFAULT_SAMPLE_CHUNK`` and accumulate to
    # host memory.
    chunk_size = _kwargs.get("chunk_size", _PPC_DEFAULT_SAMPLE_CHUNK)
    # When the totals come from the fitted NB, ``n_b`` carries
    # *independent* per-(sample, cell) draws of shape
    # ``(n_samples, n_cells)``. The chunked path must slice this
    # array along the leading axis rather than collapse it to a
    # single per-cell vector — collapsing would tie all PPC
    # samples to identical totals, which would underestimate the
    # predictive total-count variance.
    is_per_sample_totals = hasattr(n_b, "shape") and n_b.ndim == 2
    if is_per_sample_totals:
        n_b_static = jnp.asarray(n_b)
    else:
        # Fixed per-cell totals (observed / supplied / fallback) —
        # broadcast across the sample axis at draw time.
        n_b_static = jnp.asarray(n_arr_cells)

    if chunk_size is None or chunk_size >= n_samples:
        if is_per_sample_totals:
            n_b_full = n_b_static
        else:
            n_b_full = jnp.broadcast_to(
                n_b_static, (int(n_samples),) + n_b_static.shape
            )
        p_b = jnp.broadcast_to(p_per_cell, (int(n_samples),) + p_per_cell.shape)
        return np.asarray(_multinomial_sample(k_pred, n_b_full, p_b))

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(k_pred, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        if is_per_sample_totals:
            n_b_chunk = jax.lax.dynamic_slice_in_dim(
                n_b_static, start, size, axis=0
            )
        else:
            n_b_chunk = jnp.broadcast_to(
                n_b_static, (size,) + n_b_static.shape
            )
        p_b_chunk = jnp.broadcast_to(p_per_cell, (size,) + p_per_cell.shape)
        out_chunk = _multinomial_sample(chunk_keys[i], n_b_chunk, p_b_chunk)
        pieces.append(np.asarray(out_chunk))
    return np.concatenate(pieces, axis=0)


def _ppc_lnm_per_cell_laplace(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    z_loc: Optional[jnp.ndarray],
    y_alr_loc: Optional[jnp.ndarray],
    alr_reference_idx: Optional[int],
    mu_T: Optional[jnp.ndarray] = None,
    r_T: Optional[jnp.ndarray] = None,
    p_capture_loc: Optional[jnp.ndarray] = None,
    counts: Optional[jnp.ndarray] = None,
    total_counts: Optional[jnp.ndarray] = None,
    **_kwargs,
) -> jnp.ndarray:
    """Per-cell LNM(VCP) PPC propagating Laplace posterior uncertainty.

    Mirrors :func:`_ppc_lnm_per_cell` but draws the per-cell
    composition latent from its Laplace posterior at the converged
    MAP rather than holding it at the point estimate. Two paths,
    selected by which latent slot is populated:

    * ``z_loc`` (``d_mode='low_rank'``): use
      :func:`scribe.laplace._newton_lnm.sample_z_posterior` to draw
      ``z_c^(s)`` from the ``k``-dim Laplace posterior, then form
      ``y_alr^(s) = mu + W z^(s)``.
    * ``y_alr_loc`` (``d_mode='learned'``): use
      :func:`scribe.laplace._newton_lnm.sample_y_alr_posterior` to
      draw ``y_alr^(s)`` directly from its ``(G-1)``-dim Laplace
      posterior.

    The simplex is then ``softmax_full(y_alr^(s))`` and the count
    sample is ``Multinomial(u_T, simplex)``. Per-cell totals are
    resolved as in :func:`_ppc_lnm_per_cell` (observed > supplied
    > NB-fitted > legacy fallback).

    Returns
    -------
    np.ndarray, shape ``(n_samples, n_cells, G)``
    """
    if alr_reference_idx is None:
        raise ValueError(
            "LNM PPC requires alr_reference_idx; this result "
            "appears to have been constructed without it."
        )
    import numpyro.distributions as dist

    g_minus1 = mu.shape[0]
    n_genes = g_minus1 + 1

    # Determine sampling mode (z vs y_alr) and dimensions.
    if z_loc is not None and y_alr_loc is None:
        mode = "z"
        n_cells = int(z_loc.shape[0])
    elif y_alr_loc is not None:
        mode = "y_alr"
        n_cells = int(y_alr_loc.shape[0])
    else:
        raise ValueError(
            "LNM Laplace per-cell PPC requires either z_loc or y_alr_loc."
        )

    # Resolve per-cell totals (same logic as the MAP path).
    if counts is not None:
        observed_totals = jnp.asarray(counts).sum(axis=-1).astype(jnp.int32)
        n_arr_cells_static = observed_totals
        nb_fitted = False
    elif total_counts is not None:
        n_arr_cells_static = jnp.asarray(total_counts, dtype=jnp.int32)
        nb_fitted = False
    elif mu_T is not None and r_T is not None:
        n_arr_cells_static = None
        nb_fitted = True
        if p_capture_loc is not None:
            mu_T_per_cell = jnp.asarray(mu_T) * jnp.asarray(p_capture_loc)
        else:
            mu_T_per_cell = jnp.broadcast_to(jnp.asarray(mu_T), (n_cells,))
        nb_dist = dist.NegativeBinomial2(
            mean=mu_T_per_cell, concentration=jnp.asarray(r_T)
        )
    else:
        n_arr_cells_static = jnp.full((n_cells,), 1000, dtype=jnp.int32)
        nb_fitted = False

    # Sampler-specific imports — kept inside the function so the
    # MAP-only path doesn't pay for them when this module is imported.
    if mode == "z":
        from ._newton_lnm import sample_z_posterior_batch
    else:
        from ._newton_lnm import sample_y_alr_posterior_batch

    # Static "u_alr" used by the Newton-grad direction inside the
    # samplers — they evaluate at the converged MAP so the gradient
    # term is irrelevant for the *covariance*; passing zeros works
    # because the samplers only consume it in the gradient
    # computation that feeds the multinomial-Fisher matrix at
    # ``p_alr(MAP)``. We pass observed counts when available
    # (mathematically a no-op for sampling but keeps the call sites
    # symmetric); else zeros.
    if counts is not None:
        # Map counts to ALR-aligned u_alr: drop the reference column.
        ref_idx = int(alr_reference_idx)
        all_idx = list(range(n_genes))
        all_idx.remove(ref_idx)
        u_alr_per_cell = jnp.asarray(counts, dtype=jnp.float32)[
            :, jnp.asarray(all_idx)
        ]
    else:
        u_alr_per_cell = jnp.zeros((n_cells, g_minus1), dtype=mu.dtype)

    # Per-cell totals as floats for the sampler signatures.
    if n_arr_cells_static is not None:
        n_total_per_cell = jnp.asarray(n_arr_cells_static, dtype=mu.dtype)
    else:
        # NB-fitted path — we need a representative N for the
        # multinomial-Fisher term used in the per-cell Hessian.
        # Use the per-cell mean of the NB.
        n_total_per_cell = jnp.asarray(mu_T_per_cell, dtype=mu.dtype)

    chunk_size = _kwargs.get("chunk_size", _PPC_DEFAULT_SAMPLE_CHUNK)
    if chunk_size is None or chunk_size >= n_samples:
        n_chunks = 1
        chunk_sizes = [int(n_samples)]
    else:
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        chunk_sizes = [
            min(chunk_size, n_samples - i * chunk_size) for i in range(n_chunks)
        ]

    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []

    for i, size in enumerate(chunk_sizes):
        # Three RNG sub-streams: latent samples, NB totals, multinomial.
        k_lat, k_nb, k_mn = jax.random.split(chunk_keys[i], 3)
        cell_keys = jax.random.split(k_lat, n_cells)

        if mode == "z":
            # (n_cells, size, k) → (size, n_cells, k)
            z_samples = sample_z_posterior_batch(
                cell_keys,
                z_loc,
                u_alr_per_cell,
                n_total_per_cell,
                mu,
                W,
                alr_reference_idx,
                n_genes,
                size,
                0.0,
            )
            z_samples = jnp.transpose(z_samples, (1, 0, 2))
            # y_alr^(s) = mu + W z^(s)
            y_alr_samples = mu[None, None, :] + z_samples @ W.T
        else:
            # (n_cells, size, G-1) → (size, n_cells, G-1)
            y_samples = sample_y_alr_posterior_batch(
                cell_keys,
                y_alr_loc,
                u_alr_per_cell,
                n_total_per_cell,
                mu,
                W,
                d,
                alr_reference_idx,
                n_genes,
                size,
                0.0,
            )
            y_alr_samples = jnp.transpose(y_samples, (1, 0, 2))

        # Convert ALR → simplex per (sample, cell).
        p_per_sample = _alr_to_softmax(
            y_alr_samples, alr_reference_idx, n_genes  # (size, n_cells, G)
        )

        # Per-cell totals: refresh per-chunk for the NB-fitted path.
        if nb_fitted:
            n_b_chunk = nb_dist.sample(k_nb, sample_shape=(size,)).astype(
                jnp.int32
            )  # (size, n_cells)
        else:
            n_b_chunk = jnp.broadcast_to(
                jnp.asarray(n_arr_cells_static, dtype=jnp.int32),
                (size,) + (n_cells,),
            )

        out_chunk = _multinomial_sample(k_mn, n_b_chunk, p_per_sample)
        pieces.append(np.asarray(out_chunk))

    return np.concatenate(pieces, axis=0)


def _multinomial_sample(
    rng_key: jax.Array,
    n: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorised multinomial sampler with broadcast over leading axes.

    ``p`` and ``n`` must broadcast against each other on all but
    the trailing axis of ``p``. Uses NumPyro's ``MultinomialProbs``
    which lifts the underlying JAX RNG and supports vmapped batches.
    """
    import numpyro.distributions as dist

    # MultinomialProbs handles broadcasting; just return a single
    # sample at the broadcast shape.
    return dist.MultinomialProbs(probs=p, total_count=n).sample(rng_key)


# Default per-chunk sample count for the batched samplers below.
# Chosen so the device-side intermediate ``(chunk, n_cells, G)`` for
# typical scRNA-seq dimensions (3K cells × 7K genes ≈ 84 MB per
# sample) stays under ~2 GB. NumPyro's multinomial implementation
# expands ``n_max`` categoricals internally, which is what makes the
# unchunked version OOM on large datasets even on a 80 GB H100. The
# chunk size is conservative; users with more headroom can opt into
# the unchunked path by passing ``chunk_size=None`` to the helpers.
_PPC_DEFAULT_SAMPLE_CHUNK = 16


def _batched_sample_concat(
    rng_key: jax.Array,
    n_samples: int,
    sampler_fn,
    chunk_size: Optional[int] = _PPC_DEFAULT_SAMPLE_CHUNK,
) -> np.ndarray:
    """Run ``sampler_fn`` in n-sample chunks; concatenate on the host.

    The sampler is called as ``sampler_fn(chunk_key, chunk_size_int)``
    and must return a JAX array of shape ``(chunk_size_int, ...)``.
    Each chunk is moved to host (numpy) memory immediately so the
    device-side allocation is bounded by one chunk's intermediate
    arrays, not the full ``(n_samples, ...)`` output. The returned
    numpy array has shape ``(n_samples, ...)``.

    Memory accounting for typical PPC use:

    * **device peak** ~ ``chunk_size · n_cells · G · 4 bytes`` for the
      output, plus NumPyro's categorical-expand intermediate (``n_max
      · chunk_size · n_cells · 4 bytes`` where ``n_max`` is the max
      per-cell total). With ``chunk_size=16``, ``n_cells=3000``,
      ``G=7000``, ``n_max=10000``, the intermediate is ~2 GB and the
      output chunk is ~1.5 GB — comfortably under any modern GPU.
    * **host total** = ``n_samples · n_cells · G · 4 bytes`` for the
      final concatenated array. For ``n_samples=512, n_cells=3000,
      G=7000`` that's ~42 GB; users with less host RAM should reduce
      ``n_samples`` at the call site.

    When ``chunk_size`` is None, the sampler runs in a single call
    (legacy behaviour). Useful for small datasets or when the user
    knows the device has enough memory.
    """
    if chunk_size is None or chunk_size >= n_samples:
        # Single-call path: return a numpy array for parity with the
        # chunked path.
        out = sampler_fn(rng_key, int(n_samples))
        return np.asarray(out)

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        # Sample on device, then move to host immediately so the
        # next chunk's allocation has the device's full memory
        # available again. ``np.asarray`` triggers the
        # device-to-host transfer in JAX 0.4+.
        chunk = sampler_fn(chunk_keys[i], size)
        pieces.append(np.asarray(chunk))
    return np.concatenate(pieces, axis=0)


# ------ Log-likelihood helpers ------


def _ll_pln(
    counts: jnp.ndarray,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    return_by: str,
) -> jnp.ndarray:
    """Per-cell or per-gene Poisson log-likelihood at the PLN MAP."""
    from jax.scipy.special import gammaln

    u = jnp.asarray(counts, dtype=jnp.float32)
    if eta_loc is not None:
        eff = x_loc - eta_loc[:, None]
    else:
        eff = x_loc
    eff = jnp.clip(eff, _LOG_RATE_MIN, _LOG_RATE_MAX)
    rate = jnp.exp(eff)
    log_pmf = u * eff - rate - gammaln(u + 1.0)
    if return_by == "cell":
        return log_pmf.sum(axis=-1)
    return log_pmf.sum(axis=0)


def _ll_lnm(
    counts: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    z_loc: Optional[jnp.ndarray],
    y_alr_loc: Optional[jnp.ndarray],
    alr_reference_idx: Optional[int],
    return_by: str,
) -> jnp.ndarray:
    """Per-cell or per-gene multinomial log-likelihood at the LNM MAP.

    The multinomial log-pmf for a cell with counts ``u`` and total
    count ``N = sum(u)`` and probabilities ``p = softmax(logits)``
    is ``lgamma(N+1) - sum_g lgamma(u_g+1) + sum_g u_g log(p_g)``.
    """
    from jax.scipy.special import gammaln

    u = jnp.asarray(counts, dtype=jnp.float32)
    if alr_reference_idx is None:
        raise ValueError(
            "LNM log-likelihood requires alr_reference_idx; this "
            "result appears to have been constructed without it."
        )
    g_minus1 = mu.shape[0]
    n_genes = g_minus1 + 1

    if y_alr_loc is not None:
        y_alr = y_alr_loc
    elif z_loc is not None:
        y_alr = mu[None, :] + z_loc @ W.T
    else:
        raise ValueError(
            "LNM log-likelihood requires either z_loc or y_alr_loc."
        )
    log_p = jax.nn.log_softmax(
        _augment_with_reference(y_alr, alr_reference_idx, n_genes),
        axis=-1,
    )

    n_per_cell = u.sum(axis=-1)  # (n_cells,)
    log_pmf_per_cell_per_gene = u * log_p  # (n_cells, G)
    # Multinomial normalisation constant per cell:
    norm_per_cell = gammaln(n_per_cell + 1.0) - gammaln(u + 1.0).sum(axis=-1)

    if return_by == "cell":
        return log_pmf_per_cell_per_gene.sum(axis=-1) + norm_per_cell
    # return_by == "gene": per-gene contribution to the
    # multinomial log-likelihood (the data term sum_c u_cg log p_cg
    # only — the normaliser couples genes and cannot be cleanly
    # attributed per-gene).
    return log_pmf_per_cell_per_gene.sum(axis=0)


def _augment_with_reference(
    y_alr: jnp.ndarray, alr_reference_idx: int, n_genes: int
) -> jnp.ndarray:
    """Append a zero logit at ``alr_reference_idx`` to recover full G logits.

    Inverse of "drop the reference gene" used to define ALR
    coordinates. Output shape ``y_alr.shape[:-1] + (n_genes,)``.
    """
    leading = y_alr.shape[:-1]
    full = jnp.zeros(leading + (n_genes,), dtype=y_alr.dtype)
    other = list(range(n_genes))
    other.remove(int(alr_reference_idx))
    full = full.at[..., jnp.asarray(other)].set(y_alr)
    return full


__all__ = ["ScribeLaplaceResults"]
