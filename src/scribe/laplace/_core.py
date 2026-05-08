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

