"""Model-dispatching accessors for Laplace results.

This module houses public methods whose behavior depends on the fitted base
model. Dispatch is centralized here so callers can interact with a single
``ScribeLaplaceResults`` API while backend details remain model-specific.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from ..stats.distributions import LowRankPoissonLogNormal
from ._global_uncertainty import resolve_numpyro_transform, resolve_positive_fns
from ._results_shared import _base_model


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
        if bm in ("pln", "nbln"):
            # NBLN's per-cell latent has the same shape and semantic
            # role as PLN's: ``x_loc`` is the log-rate MAP at the cell.
            return self.x_loc
        if bm in ("lnm", "lnmvcp"):
            if self.z_loc is not None:
                return self.z_loc
            return self.y_alr_loc
        raise NotImplementedError(
            f"get_latent_embeddings not implemented for base_model={bm!r}"
        )

    def get_map(self, **_kwargs) -> Dict[str, jnp.ndarray]:
        """Return point-estimate dictionary for diagnostics and plotting.

        Exposes both unconstrained posterior parameterisation
        (``*_loc``, ``*_scale``) and constrained MAPs derived via
        ``model_config.positive_transform``.

        Parameters
        ----------
        **_kwargs
            Accepted for interface compatibility with other inference modes.
            Arguments are ignored because Laplace results are already MAP-based.

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
            If the stored ``base_model`` is unknown.
        """
        bm = _base_model(self.model_config)
        if bm in ("pln", "nbln"):
            d_key = "d_nbln" if bm == "nbln" else "d_pln"
            out = {
                "mu": self.mu,
                "W": self.W,
                d_key: self.d,
                "y_log_rate": self.x_loc,
            }
            if self.eta_loc is not None:
                out["eta_capture"] = self.eta_loc
                out["p_capture"] = jnp.exp(-self.eta_loc)
            if bm == "nbln" and self.r is not None:
                out["r"] = self.r
            # NBLN global posterior parameters in unconstrained space.
            if bm == "nbln" and self.r_loc is not None:
                out["r_loc"] = self.r_loc
            if bm == "nbln" and self.r_scale is not None:
                out["r_scale"] = self.r_scale
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
            # Constrained totals MAPs.
            if self.mu_T is not None:
                out["mu_T"] = self.mu_T
            if self.r_T is not None:
                out["r_T"] = self.r_T
            # Derived success probability.
            if self.mu_T is not None and self.r_T is not None:
                out["p"] = self.r_T / (self.r_T + self.mu_T)
            # Unconstrained posterior parameterisation.
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
            out: Dict[str, Any] = {
                "y_log_rate": dist.LowRankMultivariateNormal(
                    loc=self.mu, cov_factor=self.W, cov_diag=self.d
                ),
            }
            if bm == "pln":
                out["lambda_rate"] = LowRankPoissonLogNormal(
                    loc=self.mu, cov_factor=self.W, cov_diag=self.d
                )
            if bm == "nbln":
                if self.r_loc is not None and self.r_scale is not None:
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
                elif self.r is not None:
                    out["r"] = dist.Delta(self.r)
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

