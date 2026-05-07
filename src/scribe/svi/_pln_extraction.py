"""
Mixin for extracting PLN (Poisson-LogNormal) population parameters.

Extracts the low-rank log-rate covariance structure (mu, W, d) from the
VAE decoder weights and NumPyro parameter store.  Applicable to ``pln``
base models.

Unlike the LNM mixin which extracts (G-1)-dimensional ALR parameters,
the PLN mixin extracts G-dimensional log-rate parameters directly.

Methods are guarded by :meth:`_require_pln` which raises ``ValueError`` when
the model config is not a PLN variant.

Classes
-------
PLNExtractionMixin
    Mixin providing ``get_pln_mu``, ``get_pln_W``, ``get_pln_d``,
    ``get_pln_sigma``, and ``get_pln_correlation``.

See Also
--------
scribe.svi.vae_results : ScribeVAEResults that composes this mixin.
scribe.svi._lnm_extraction : Analogous mixin for LNM models.
"""

from typing import Any, Optional, Tuple, cast

import jax.numpy as jnp

from ._latent_space import _DECODER_KEY


class PLNExtractionMixin:
    """Mixin providing PLN population parameter extraction from VAE results.

    Requires ``self.params``, ``self.model_config``, and ``self.n_genes`` to
    be present on the host class (satisfied by ``ScribeVAEResults``).

    The PLN decoder emits G-dimensional ``y_log_rate`` (not G-1 ALR
    coordinates), so all extracted parameters are G-dimensional.
    """

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _require_pln(self) -> None:
        """Raise ``ValueError`` if the model is not a PLN variant."""
        bm = getattr(self.model_config, "base_model", None)
        if bm != "pln":
            raise ValueError(
                "PLN extraction methods require model_config.base_model "
                f"== 'pln', got {bm!r}."
            )

    # ------------------------------------------------------------------
    # Decoder parameter helpers
    # ------------------------------------------------------------------

    def _y_log_rate_head_dense_params(
        self,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(kernel, bias)`` for the decoder's ``y_log_rate`` head.

        Navigates the Flax parameter tree to locate the linear dense layer
        that produces log-rate coordinates.

        Returns
        -------
        kernel : jnp.ndarray
            Shape ``(k, G)`` -- Flax Dense kernel (in_features, out_features).
        bias : jnp.ndarray
            Shape ``(G,)`` -- dense bias vector.

        Raises
        ------
        KeyError
            If the decoder parameter subtree is missing or has unexpected
            structure.
        """
        dec_params = self.params.get(_DECODER_KEY)
        if not isinstance(dec_params, dict):
            raise KeyError(
                f"params['{_DECODER_KEY}'] must be a dict, "
                f"got {type(dec_params)}."
            )
        subtree = dec_params.get("head_y_log_rate")
        if subtree is None:
            raise KeyError(
                "Decoder params missing 'head_y_log_rate'. "
                "Expected a MultiHeadDecoder head with "
                "param_name 'y_log_rate'."
            )
        if not isinstance(subtree, dict):
            raise TypeError(
                f"head_y_log_rate params must be a dict, "
                f"got {type(subtree)}."
            )
        # Flax may store params flat or under a Dense_0 sublayer.
        if "kernel" in subtree and "bias" in subtree:
            return cast(jnp.ndarray, subtree["kernel"]), cast(
                jnp.ndarray, subtree["bias"]
            )
        dense0 = subtree.get("Dense_0")
        if (
            isinstance(dense0, dict)
            and "kernel" in dense0
            and "bias" in dense0
        ):
            return cast(jnp.ndarray, dense0["kernel"]), cast(
                jnp.ndarray, dense0["bias"]
            )
        raise KeyError(
            "Could not find Dense kernel/bias under head_y_log_rate "
            "(expected Flax {'kernel','bias'} or nested Dense_0)."
        )

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    def get_pln_mu(self) -> jnp.ndarray:
        """Extract the PLN population mean mu from the decoder bias.

        The ``y_log_rate`` head of the VAE decoder stores the log-rate-space
        mean in the final dense bias vector (identity transform, no ALR).

        Returns
        -------
        jnp.ndarray
            Shape ``(G,)`` -- mean log-rates for each gene.
        """
        self._require_pln()
        _, bias = self._y_log_rate_head_dense_params()
        mu = jnp.asarray(bias)
        g = int(self.n_genes)
        if mu.shape != (g,):
            raise ValueError(
                f"Expected log-rate bias shape (G,) = ({g},), "
                f"got {mu.shape}."
            )
        return mu

    def get_pln_W(self) -> jnp.ndarray:
        """Extract the PLN low-rank factor ``W`` from the decoder kernel.

        Flax :class:`flax.linen.Dense` stores ``kernel`` with shape
        ``(in_features, out_features)`` = ``(k, G)`` for this head.
        The generative log-rate covariance uses ``W`` with shape ``(G, k)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(G, k)`` with ``k`` the latent dimension.
        """
        self._require_pln()
        kernel, _bias = self._y_log_rate_head_dense_params()
        k_mat = jnp.asarray(kernel)
        g = int(self.n_genes)
        if k_mat.shape[-1] != g:
            raise ValueError(
                f"Expected kernel out_features G = {g}, "
                f"got kernel shape {k_mat.shape}."
            )
        # Transpose: (k, G) -> (G, k) to match the generative model
        # convention Sigma = W W^T + diag(d).
        W = k_mat.T
        if W.shape[0] != g:
            raise ValueError(
                f"Internal shape error: W.shape = {W.shape}."
            )
        return W

    def get_pln_d(self) -> Optional[jnp.ndarray]:
        """Extract the learned diagonal ``d`` from NumPyro / SVI params.

        When :attr:`~scribe.models.config.ModelConfig.d_mode` is
        ``'learned'``, the population log-rate covariance includes
        :math:`\\mathrm{diag}(d)` with :math:`d` a learned positive vector of
        length ``G`` (site ``d_pln``).  If ``d_mode='low_rank'``, this
        method returns ``None``.

        Returns
        -------
        jnp.ndarray or None
            Shape ``(G,)`` when ``d`` is learned and found in ``params``;
            ``None`` for low-rank-only log-rate uncertainty.
        """
        self._require_pln()

        def _recursive_find_d_pln(obj: Any) -> Optional[jnp.ndarray]:
            """Walk the param tree looking for d_pln or d_pln_loc.

            ``d_pln`` lives in the constrained ``(0, ∞)`` space of the
            generative model -- it is the diagonal of ``Sigma = W W^T +
            diag(d)``. The mean-field guide stores it via a
            ``LogNormalSpec``, which means the ``..._loc`` numpyro
            parameter is the underlying Normal's location in
            *unconstrained* (log) space; the actual constrained MAP is
            ``exp(loc)``. We apply that transform here so callers get
            the value they expect (positive, on the original scale)
            without having to know the guide internals.
            """
            if isinstance(obj, dict):
                if "d_pln" in obj:
                    raw = obj["d_pln"]
                    # Some guide families store a nested ``{loc, scale}``
                    # dict: the loc is unconstrained, so exponentiate.
                    if isinstance(raw, dict) and "loc" in raw:
                        return jnp.exp(jnp.asarray(raw["loc"]))
                    # Otherwise it's already the constrained value.
                    return jnp.asarray(raw)
                # Mean-field guide stores variational parameters flat as
                # ``{name}_loc`` / ``{name}_scale``. The loc is in
                # unconstrained log-space, so the constrained MAP is
                # ``exp(loc)``.
                if "d_pln_loc" in obj:
                    return jnp.exp(jnp.asarray(obj["d_pln_loc"]))
                for v in obj.values():
                    got = _recursive_find_d_pln(v)
                    if got is not None:
                        return got
            return None

        explicit_mode = getattr(self.model_config, "d_mode", "low_rank")
        d_arr = _recursive_find_d_pln(self.params)

        if explicit_mode == "low_rank":
            return None
        if explicit_mode != "learned":
            raise ValueError(
                f"model_config.d_mode must be 'low_rank' or 'learned', "
                f"got {explicit_mode!r}."
            )
        if d_arr is None:
            raise ValueError(
                "model_config.d_mode is 'learned' but no 'd_pln' "
                "parameter was found in params."
            )
        g = int(self.n_genes)
        if d_arr.shape != (g,):
            raise ValueError(
                f"Expected d_pln shape (G,) = ({g},), "
                f"got {d_arr.shape}."
            )
        return d_arr

    def get_pln_sigma(self) -> jnp.ndarray:
        """Reconstruct full log-rate covariance ``Sigma = W W^T + diag(d)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(G, G)``. Useful only for small ``G``; for large
            ``G`` prefer :meth:`get_pln_W` and :meth:`get_pln_d`.

        Notes
        -----
        In ``low_rank`` mode, ``d`` is treated as zero on the diagonal.
        """
        self._require_pln()
        W = self.get_pln_W()
        d_opt = self.get_pln_d()
        d_vec = (
            jnp.asarray(d_opt)
            if d_opt is not None
            else jnp.zeros(W.shape[0], dtype=W.dtype)
        )
        return W @ W.T + jnp.diag(d_vec)

    def get_pln_correlation(self) -> jnp.ndarray:
        """Gene-gene correlation matrix in log-rate space.

        Computes the full log-rate covariance ``Sigma = W W^T + diag(d)``
        and converts to a correlation matrix.  Unlike LNM, no ALR-to-CLR
        transformation is needed because log-rate space is already
        symmetric across genes.

        Returns
        -------
        jnp.ndarray
            Shape ``(G, G)`` correlation matrix.
        """
        self._require_pln()
        sigma = self.get_pln_sigma()
        std = jnp.sqrt(jnp.maximum(jnp.diag(sigma), 1e-30))
        return sigma / (std[:, None] * std[None, :])

    def get_pln_library_size_direction(self) -> jnp.ndarray:
        """Latent-space unit vector whose ``W``-image is closest to ``1_G``.

        Wrapper over
        :func:`scribe.stats.correlation_diagnostics.library_size_direction`
        for VAE-fit PLN results. See the helper for the structural-
        redundancy rationale.
        """
        self._require_pln()
        from ..stats.correlation_diagnostics import library_size_direction
        return library_size_direction(self.get_pln_W())

    def get_pln_correlation_residual(
        self,
        method: str = "library_size",
        n_components: int = 1,
        include_diagonal_d: bool = False,
    ) -> jnp.ndarray:
        """PLN gene-gene correlation with latent direction(s) projected out.

        See
        :func:`scribe.stats.correlation_diagnostics.correlation_residual`
        for the math. Pulls ``W`` and ``d`` from the VAE decoder /
        learned residual.
        """
        self._require_pln()
        from ..stats.correlation_diagnostics import correlation_residual
        d = self.get_pln_d()
        return correlation_residual(
            self.get_pln_W(), d,
            method=method,
            n_components=n_components,
            include_diagonal_d=include_diagonal_d,
        )

    def summarize_pln_correlation_structure(
        self,
        *,
        n_top_eig: int = 10,
        verbose: bool = True,
    ):
        """Print and return a diagnostic summary of the PLN correlation structure.

        Wrapper over
        :func:`scribe.stats.correlation_diagnostics.summarize_correlation_structure`
        for VAE-fit PLN results — surfaces library-size alignment,
        the latent eigenspectrum, and projection-comparison
        off-diagonal quantiles for picking
        ``subtract_direction`` in the heatmap plot.
        """
        self._require_pln()
        from ..stats.correlation_diagnostics import (
            summarize_correlation_structure,
        )
        return summarize_correlation_structure(
            self.get_pln_W(), self.get_pln_d(),
            space_label="log-rate space",
            model_label="PLN VAE",
            n_top_eig=n_top_eig,
            verbose=verbose,
        )
