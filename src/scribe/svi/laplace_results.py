"""Results container for PLN Laplace-mode inference.

Standalone dataclass — *not* a subclass of ``ScribeVAEResults`` —
because the Laplace engine bypasses the NumPyro SVI pipeline that the
VAE results inherit from. The mixin chain on ``ScribeVAEResults``
assumes a NumPyro params dict with ``vae_encoder$params`` and
``vae_decoder$params`` keys plus an instantiated encoder Linen
module; faking those for Laplace would couple the two paths in ways
that hurt rather than help.

Instead we re-implement the *most-used* downstream surface here in
~250 lines, covering:

* PLN parameter extraction (``get_pln_mu/W/d/sigma/correlation``).
* Laplace-specific extraction (``get_laplace_x_loc/eta_loc/p_capture``).
* MAP-style accessors (``get_map``, ``get_distributions``).
* Population-level posterior predictive sampling
  (``get_ppc_samples``, ``get_predictive_samples``).
* Per-cell posterior predictive sampling that uses the stored MAP
  (``get_per_cell_predictive_samples``).
* Per-gene log-likelihood evaluation (``get_log_likelihood``).
* Gene subsetting via ``__getitem__`` so plot helpers can do
  ``results[selected_idx]`` to focus on a subset.
* Gene-coverage / "_other" pseudo-gene metadata for parity with
  ``ScribeVAEResults``.

Sampling reuses :class:`scribe.stats.distributions.LowRankPoissonLogNormal`
since the *generative model* is identical to the VAE path's PLN —
only the *inference procedure* differs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..models.config import ModelConfig
from ..stats.distributions import LowRankPoissonLogNormal


# Floor used inside ``exp`` to prevent float32 overflow in the rate.
# Matches the kernel's ``_LOG_RATE_MIN/_LOG_RATE_MAX`` so PPCs and
# the Newton kernel agree on the safety bound.
_LOG_RATE_MIN = -30.0
_LOG_RATE_MAX = 30.0


@dataclass
class ScribeLaplaceResults:
    """Results from PLN inference under the Laplace approximation.

    Provides the core downstream surface needed for diagnostics and
    plotting: parameter extraction, posterior predictive sampling
    (population and per-cell), gene subsetting, and log-likelihood
    evaluation. All methods exposed here work without the NumPyro SVI
    machinery — they read directly from the trained globals
    (``mu``, ``W``, ``d``) and per-cell MAP (``x_loc``, ``eta_loc``).

    Attributes
    ----------
    model_config : Optional[ModelConfig]
        Configuration used for training (provenance + extraction).
    mu : jnp.ndarray, shape (G,)
        Decoder bias = mean of the latent log-rate prior.
    W : jnp.ndarray, shape (G, k)
        Decoder loadings = ``Σ = W W^T + diag(d)`` factor.
    d : jnp.ndarray, shape (G,)
        Diagonal residual variance (constrained, > 0).
    x_loc : jnp.ndarray, shape (n_cells, G)
        Per-cell MAP estimate of ``x_c`` from Newton.
    eta_loc : jnp.ndarray or None, shape (n_cells,)
        Per-cell capture offset MAP, or ``None`` when no capture
        anchor was active.
    final_grad_norms : jnp.ndarray, shape (n_cells,)
        Final per-cell L∞ gradient norm — convergence diagnostic.
    losses : jnp.ndarray
        Outer-loop loss history.
    n_genes : int
        Number of genes (after any ``gene_coverage`` filtering, which
        may have introduced a trailing ``_other`` pseudo-gene).
    n_cells : int
        Number of cells.
    var : Optional[Any]
        AnnData-style ``var`` DataFrame (gene metadata) attached at
        results-building time. ``None`` for results created outside
        the ``scribe.fit`` flow.
    obs : Optional[Any]
        AnnData-style ``obs`` DataFrame (cell metadata).
    n_obs : Optional[int]
        Number of observations (cells); convenience alias mirroring
        ``ScribeVAEResults``.
    n_vars : Optional[int]
        Number of variables (genes); convenience alias mirroring
        ``ScribeVAEResults``.
    metadata : dict
        Free-form metadata bag for downstream plotting / analysis
        helpers that don't fit the dataclass shape.
    """

    model_config: Optional[ModelConfig]
    mu: jnp.ndarray
    W: jnp.ndarray
    d: jnp.ndarray
    x_loc: jnp.ndarray
    eta_loc: Optional[jnp.ndarray]
    final_grad_norms: jnp.ndarray
    losses: jnp.ndarray
    n_genes: int
    n_cells: int

    # AnnData-style metadata; populated by ``scribe.fit`` post-run
    # alongside the standard VAE/SVI pickles.
    var: Optional[Any] = None
    obs: Optional[Any] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

    # Gene-coverage metadata (parity with ``ScribeVAEResults``).
    _gene_coverage: Optional[float] = None
    _gene_coverage_mask: Optional[np.ndarray] = None
    _excluded_gene_names: Optional[List[str]] = None
    _original_n_genes: Optional[int] = None
    _total_count_max: Optional[int] = None

    # Free-form metadata for plotting helpers / future use.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # PLN parameter extraction (mirrors ``PLNExtractionMixin`` API).
    # ------------------------------------------------------------------

    def get_pln_mu(self) -> jnp.ndarray:
        """Decoder bias ``μ`` (G,)."""
        return self.mu

    def get_pln_W(self) -> jnp.ndarray:
        """Decoder loadings ``W`` (G, k)."""
        return self.W

    def get_pln_d(self) -> jnp.ndarray:
        """Diagonal residual variance ``d`` (G,), constrained > 0."""
        return self.d

    def get_pln_sigma(self) -> jnp.ndarray:
        """Full covariance ``Σ = W W^T + diag(d)`` (G, G)."""
        return self.W @ self.W.T + jnp.diag(self.d)

    def get_pln_correlation(self) -> jnp.ndarray:
        """Gene-gene correlation matrix derived from ``Σ`` (G, G)."""
        sigma = self.get_pln_sigma()
        std = jnp.sqrt(jnp.maximum(jnp.diag(sigma), 1e-30))
        return sigma / (std[:, None] * std[None, :])

    # ------------------------------------------------------------------
    # Laplace-specific extraction.
    # ------------------------------------------------------------------

    def get_laplace_x_loc(self) -> jnp.ndarray:
        """Per-cell MAP of the latent log-rate ``x_c`` (n_cells, G)."""
        return self.x_loc

    def get_laplace_eta_loc(self) -> Optional[jnp.ndarray]:
        """Per-cell capture offset MAP, or ``None`` if no capture anchor."""
        return self.eta_loc

    def get_laplace_p_capture(self) -> Optional[jnp.ndarray]:
        """Per-cell capture probability ``p_c = exp(-η_c)``, or ``None``."""
        if self.eta_loc is None:
            return None
        return jnp.exp(-self.eta_loc)

    def get_latent_embeddings(self) -> jnp.ndarray:
        """Per-cell latent log-rate embeddings ``x_c`` (n_cells, G).

        Mirrors ``ScribeVAEResults.get_latent_embeddings`` so downstream
        UMAP / clustering helpers can run on either result type. For
        Laplace this is the MAP rather than a posterior sample, but
        the shape and semantic role are the same.
        """
        return self.x_loc

    # ------------------------------------------------------------------
    # MAP / distributions (parity with VAE results).
    # ------------------------------------------------------------------

    def get_map(self, **_kwargs) -> Dict[str, jnp.ndarray]:
        """Return a dict of point estimates.

        Mirrors ``ScribeVAEResults.get_map`` so plotting helpers that
        consume MAP dicts (e.g. ``mean_calibration``, capture-anchor
        plots) work on either result type.

        Parameters
        ----------
        **_kwargs
            Accepted for signature compatibility with the VAE
            ``get_map`` (which has flow / canonicalisation flags).
            All are ignored — Laplace results are already point
            estimates by design.

        Returns
        -------
        dict
            ``{"mu": μ, "W": W, "d_pln": d, "y_log_rate": x_loc,
            "eta_capture": η_loc (if present), "p_capture": ...}``.
            ``y_log_rate`` is per-cell ``(n_cells, G)`` here — the
            calibration plotter handles either shape.
        """
        out: Dict[str, jnp.ndarray] = {
            "mu": self.mu,
            "W": self.W,
            "d_pln": self.d,
            # Per-cell latent log-rate. Calibration plotters detect
            # the leading axis and reduce appropriately.
            "y_log_rate": self.x_loc,
        }
        if self.eta_loc is not None:
            out["eta_capture"] = self.eta_loc
            out["p_capture"] = jnp.exp(-self.eta_loc)
        return out

    def get_distributions(
        self, backend: str = "numpyro", **_kwargs
    ) -> Dict[str, Any]:
        """Return a dict of distributions for downstream PPC / plotting.

        Parameters
        ----------
        backend : {"numpyro", "scipy"}
            Currently only ``"numpyro"`` is supported.
        **_kwargs
            Accepted for compatibility; ignored.

        Returns
        -------
        dict
            ``{"y_log_rate": LowRankMultivariateNormal(μ, W, d),
            "lambda_rate": LowRankPoissonLogNormal(μ, W, d)}`` —
            both *population* distributions (i.e., what the model
            predicts for a new cell drawn from the prior). For per-
            cell posterior predictive samples conditioned on the
            observed data, use :meth:`get_per_cell_predictive_samples`.
        """
        if backend != "numpyro":
            raise ValueError(
                "Only 'numpyro' backend supported for Laplace results."
            )
        import numpyro.distributions as dist

        return {
            "y_log_rate": dist.LowRankMultivariateNormal(
                loc=self.mu, cov_factor=self.W, cov_diag=self.d
            ),
            "lambda_rate": LowRankPoissonLogNormal(
                loc=self.mu, cov_factor=self.W, cov_diag=self.d
            ),
        }

    # ------------------------------------------------------------------
    # Posterior predictive sampling.
    # ------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        per_cell: bool = False,
        **_kwargs,
    ) -> jnp.ndarray:
        """Draw posterior predictive samples.

        Two modes, controlled by ``per_cell``:

        * ``per_cell=False`` (default — population PPC): draw
          ``n_samples`` cells from the *population* distribution
          ``LowRankPoissonLogNormal(μ, W, d)``. Captures what the
          model predicts for a *new* cell drawn from the prior. The
          natural diagnostic for "does the population-level model
          match the data?" — this is what your earlier
          ``mean_calibration`` plot effectively checked.
        * ``per_cell=True`` — per-cell posterior PPC: for each cell
          ``c``, draw ``n_samples`` count vectors from
          ``Poisson(exp(x_c^* − η_c^*))`` using the stored MAP. This
          is the conditional predictive (analogous to what NumPyro's
          ``Predictive`` produces for the VAE path under guide
          replay).

        Parameters
        ----------
        rng_key : jax.Array, optional
            PRNG key. Defaults to ``random.PRNGKey(0)`` for
            reproducibility.
        n_samples : int, default 100
            Number of samples to draw. Interpretation depends on
            ``per_cell``: total cells in population mode, samples per
            cell in per-cell mode.
        per_cell : bool, default False
            See above.
        **_kwargs
            Accepted for compatibility with ``ScribeVAEResults.get_ppc_samples``.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_samples, G)`` for population mode,
            ``(n_samples, n_cells, G)`` for per-cell mode.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        if per_cell:
            return self.get_per_cell_predictive_samples(
                rng_key=rng_key, n_samples=n_samples
            )
        # Population PPC via the LowRankPoissonLogNormal distribution.
        pln = LowRankPoissonLogNormal(
            loc=self.mu, cov_factor=self.W, cov_diag=self.d
        )
        return pln.sample(rng_key, sample_shape=(n_samples,))

    # Alias matching the SVI/VAE result naming.
    def get_predictive_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
        **_kwargs,
    ) -> jnp.ndarray:
        """Alias for ``get_ppc_samples`` with population mode (default)."""
        return self.get_ppc_samples(
            rng_key=rng_key, n_samples=n_samples, per_cell=False
        )

    def get_per_cell_predictive_samples(
        self,
        rng_key: Optional[jax.Array] = None,
        n_samples: int = 100,
    ) -> jnp.ndarray:
        """Per-cell posterior predictive samples using the stored MAP.

        For each cell, draws Poisson samples from
        ``exp(x_c^* − η_c^*)`` (the per-cell effective log-rate at
        the MAP). Captures the *conditional* PPC — predictions tied
        to each observed cell — analogous to what NumPyro's
        ``Predictive`` produces for VAE results.

        Parameters
        ----------
        rng_key : jax.Array, optional
        n_samples : int, default 100

        Returns
        -------
        jnp.ndarray, shape (n_samples, n_cells, G)
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        # Effective per-cell log-rate at the MAP.
        if self.eta_loc is not None:
            eff_log_rate = self.x_loc - self.eta_loc[:, None]
        else:
            eff_log_rate = self.x_loc
        eff_log_rate = jnp.clip(eff_log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
        rate = jnp.exp(eff_log_rate)  # (n_cells, G)
        # Broadcast to (n_samples, n_cells, G) and Poisson-sample.
        rate_b = jnp.broadcast_to(
            rate, (n_samples,) + rate.shape
        )
        return jax.random.poisson(rng_key, rate_b)

    # ------------------------------------------------------------------
    # Log-likelihood (per-gene Poisson at the MAP).
    # ------------------------------------------------------------------

    def get_log_likelihood(
        self,
        counts: jnp.ndarray,
        return_by: str = "cell",
    ) -> jnp.ndarray:
        """Per-cell or per-gene Poisson log-likelihood at the MAP.

        Evaluates ``log p(u_c | x_c^*, η_c^*)`` with the stored MAP.
        Mirrors ``LikelihoodMixin.get_log_likelihood`` from
        ``ScribeVAEResults`` so the same downstream code can call this
        on either result type.

        Parameters
        ----------
        counts : jnp.ndarray, shape (n_cells, G)
            Observed counts (typically the training data).
        return_by : {"cell", "gene"}, default "cell"
            ``"cell"``: sum across genes → shape ``(n_cells,)``.
            ``"gene"``: sum across cells → shape ``(G,)``.

        Returns
        -------
        jnp.ndarray
            Per-cell or per-gene log-likelihood.
        """
        from jax.scipy.special import gammaln

        u = jnp.asarray(counts, dtype=jnp.float32)
        if self.eta_loc is not None:
            eff = self.x_loc - self.eta_loc[:, None]
        else:
            eff = self.x_loc
        eff = jnp.clip(eff, _LOG_RATE_MIN, _LOG_RATE_MAX)
        rate = jnp.exp(eff)  # (n_cells, G)
        # Poisson log-pmf per (cell, gene): u·log(rate) − rate − lgamma(u+1).
        log_pmf = u * eff - rate - gammaln(u + 1.0)
        if return_by == "cell":
            return log_pmf.sum(axis=-1)
        if return_by == "gene":
            return log_pmf.sum(axis=0)
        raise ValueError(
            f"return_by must be 'cell' or 'gene'; got {return_by!r}."
        )

    # ------------------------------------------------------------------
    # Gene subsetting (mirrors ``GeneSubsettingMixin.__getitem__``).
    # ------------------------------------------------------------------

    def __getitem__(
        self, gene_index: Union[int, slice, np.ndarray, jnp.ndarray, list]
    ) -> "ScribeLaplaceResults":
        """Subset to a gene index (slice / fancy indexing).

        Used by viz code that picks N genes from a fit and operates
        on the subset (PPC plot, mean calibration, etc.). Returns a
        new ``ScribeLaplaceResults`` with all gene-axis arrays sliced.

        Parameters
        ----------
        gene_index : int, slice, ndarray, list
            Anything ``np.asarray(...)`` can interpret as a gene index.
            A bare ``int`` is wrapped in a 1-element array so the
            resulting object still has a gene dimension (matches
            ``ScribeVAEResults`` behavior).

        Returns
        -------
        ScribeLaplaceResults
            New result with sliced ``mu``, ``W``, ``d``, ``x_loc``,
            ``var`` (and ``n_genes`` updated). Cell-level fields
            (``eta_loc``, ``losses``, ``final_grad_norms``,
            ``n_cells``) are unchanged.
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
        idx_jnp = jnp.asarray(idx)

        # Slice gene-axis arrays. ``W`` keeps its second axis (rank k).
        new_mu = self.mu[idx_jnp]
        new_W = self.W[idx_jnp, :]
        new_d = self.d[idx_jnp]
        new_x_loc = self.x_loc[:, idx_jnp]

        # AnnData ``var`` dataframe — slice if available, else None.
        new_var = None
        if self.var is not None:
            try:
                new_var = self.var.iloc[idx]
            except (TypeError, AttributeError):
                # Not a pandas DataFrame; leave alone.
                new_var = self.var

        # Gene-coverage metadata is *original*-gene-space and not
        # sliced (mirrors ``GeneSubsettingMixin`` behavior).
        return replace(
            self,
            mu=new_mu,
            W=new_W,
            d=new_d,
            x_loc=new_x_loc,
            n_genes=int(len(idx)),
            n_vars=int(len(idx)) if self.n_vars is not None else None,
            var=new_var,
            # Mark the subset for downstream code that consults this:
            _subset_gene_index=idx,
        )

    # Subset-bookkeeping field — populated only when ``__getitem__``
    # has been called. Public consumers should not set this directly.
    _subset_gene_index: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Convenience aliases / properties.
    # ------------------------------------------------------------------

    @property
    def predictive_samples(self) -> Optional[jnp.ndarray]:
        """Cached predictive samples slot (for plotter compatibility).

        ``ScribeVAEResults`` has a ``predictive_samples`` attribute
        that the PPC plotting helpers populate via
        ``_get_predictive_samples_for_plot``. We expose the same name
        so the dispatch helpers can store and read predictive samples
        on this object too.
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

    # ------------------------------------------------------------------
    # Pickle support.
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle-safe state.

        ``model_config`` may contain non-picklable Linen modules in
        ``model_config.vae``; we strip those for pickle and regenerate
        on load if needed. Mirrors
        ``ScribeVAEResults.__getstate__`` behavior.
        """
        from .vae_results import make_model_config_pickle_safe

        state = dict(self.__dict__)
        state["model_config"] = make_model_config_pickle_safe(
            state.get("model_config")
        )
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)


__all__ = ["ScribeLaplaceResults"]
