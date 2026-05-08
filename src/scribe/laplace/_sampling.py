"""Predictive sampling mixin for Laplace results.

This module exposes the public posterior-predictive API used by downstream
analysis and visualization code.  Each public method delegates to
model-specific private helpers so the user-facing signatures remain stable
across PLN and LNM-family Laplace fits.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from ._results_sampling_helpers import (
    _ppc_lnm_library_anchored,
    _ppc_lnm_marginal,
    _ppc_lnm_per_cell,
    _ppc_lnm_per_cell_laplace,
    _ppc_pln_library_anchored,
    _ppc_pln_marginal,
    _ppc_pln_per_cell,
    _ppc_pln_per_cell_laplace,
)
from ._results_shared import _base_model


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
            if bm == "pln":
                return _ppc_pln_library_anchored(
                    rng_key, n_samples, self.mu, self.W, self.d, counts=counts
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
        if bm in ("lnm", "lnmvcp"):
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
                **kwargs,
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
            f"get_map_ppc_samples not implemented for base_model={bm!r}"
        )
