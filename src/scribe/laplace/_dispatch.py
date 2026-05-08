"""Model-dispatching accessors for Laplace results.

This module houses public methods whose behavior depends on the fitted base
model. Dispatch is centralized here so callers can interact with a single
``ScribeLaplaceResults`` API while backend details remain model-specific.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from ..stats.distributions import LowRankPoissonLogNormal
from ._results_shared import _base_model


class DispatchResultsMixin:
    """Mixin implementing model-branching public accessors."""

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

        Parameters
        ----------
        **_kwargs
            Accepted for interface compatibility with other inference modes.
            Arguments are ignored because Laplace results are already MAP-based.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Model-specific mapping of semantic parameter names to arrays.
            The keys match the names expected by plotting and analysis helpers.

        Raises
        ------
        NotImplementedError
            If the stored ``base_model`` is unknown.
        """
        bm = _base_model(self.model_config)
        if bm in ("pln", "nbln"):
            # NBLN map is PLN's map plus the gene dispersion ``r``.
            # The diagonal residual key follows the model-specific
            # site name (``d_pln`` vs ``d_nbln``) so the dictionary
            # is consumable by both PLN- and NBLN-specific
            # downstream code.
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
        raise NotImplementedError(f"get_map not implemented for base_model={bm!r}")

    def get_distributions(
        self, backend: str = "numpyro", **_kwargs
    ) -> Dict[str, Any]:
        """Return population-level distributions implied by fitted globals.

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
            Distribution dictionary:

            - PLN: ``{"y_log_rate", "lambda_rate"}``
            - LNM/LNMVCP: ``{"y_alr"}``

            The returned objects describe the learned population latent
            distribution, not per-cell posterior conditionals.

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
            # The population log-rate distribution is the same for
            # both PLN and NBLN (a low-rank multivariate normal with
            # the same loc/W/d). PLN exposes ``lambda_rate`` as the
            # log-normal-mixed Poisson rate; NBLN's analogous
            # marginal-rate distribution is parameterised by the
            # additional gene dispersion ``r`` and is not yet
            # available as a NumPyro Distribution -- only the
            # log-rate distribution is exposed.
            out = {
                "y_log_rate": dist.LowRankMultivariateNormal(
                    loc=self.mu, cov_factor=self.W, cov_diag=self.d
                ),
            }
            if bm == "pln":
                out["lambda_rate"] = LowRankPoissonLogNormal(
                    loc=self.mu, cov_factor=self.W, cov_diag=self.d
                )
            return out
        if bm in ("lnm", "lnmvcp"):
            return {
                "y_alr": dist.LowRankMultivariateNormal(
                    loc=self.mu, cov_factor=self.W, cov_diag=self.d
                )
            }
        raise NotImplementedError(
            f"get_distributions not implemented for base_model={bm!r}"
        )

