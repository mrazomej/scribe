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
    x_loc: Optional[jnp.ndarray] = None        # PLN: (n_cells, G)
    eta_loc: Optional[jnp.ndarray] = None      # PLN capture anchor: (n_cells,)
    z_loc: Optional[jnp.ndarray] = None        # LNM low_rank: (n_cells, k)
    y_alr_loc: Optional[jnp.ndarray] = None    # LNM learned: (n_cells, G-1)
    p_capture_loc: Optional[jnp.ndarray] = None  # LNMVCP capture: (n_cells,)
    alr_reference_idx: Optional[int] = None    # LNM/LNMVCP only

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
        """Per-cell posterior predictive samples using the stored MAP.

        For each cell, draws ``n_samples`` predictive count vectors
        conditioned on its MAP. Captures the *conditional* PPC —
        predictions tied to each observed cell — analogous to what
        NumPyro's ``Predictive`` produces for VAE results.

        Parameters
        ----------
        rng_key : jax.Array, optional
        n_samples : int, default 100
        **kwargs
            For LNM-family models: ``total_counts`` per cell. When
            absent, falls back to using the observed total counts
            from the data, if attached via ``metadata['counts']``.

        Returns
        -------
        jnp.ndarray, shape (n_samples, n_cells, G)
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
            f"get_per_cell_predictive_samples not implemented for "
            f"base_model={bm!r}"
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
            x_loc=self.x_loc[:, idx_jnp]
            if self.x_loc is not None
            else None,
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
    """Per-cell Poisson PPC at the stored MAP.

    Effective per-cell log-rate is ``x_c - eta_c`` (or ``x_c``
    when no capture anchor); broadcast to ``(n_samples, n_cells,
    G)`` and Poisson-sample.
    """
    if eta_loc is not None:
        eff_log_rate = x_loc - eta_loc[:, None]
    else:
        eff_log_rate = x_loc
    eff_log_rate = jnp.clip(
        eff_log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX
    )
    rate = jnp.exp(eff_log_rate)
    rate_b = jnp.broadcast_to(rate, (n_samples,) + rate.shape)
    return jax.random.poisson(rng_key, rate_b)


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
    return _multinomial_sample(k3, n_arr, p)


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
        raise ValueError(
            "LNM per-cell PPC requires either z_loc or y_alr_loc."
        )

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
            mu_T_per_cell = jnp.broadcast_to(
                jnp.asarray(mu_T), (n_cells,)
            )
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

    p_b = jnp.broadcast_to(p_per_cell, (n_samples,) + p_per_cell.shape)
    return _multinomial_sample(k_pred, n_b, p_b)


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
    norm_per_cell = (
        gammaln(n_per_cell + 1.0)
        - gammaln(u + 1.0).sum(axis=-1)
    )

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
