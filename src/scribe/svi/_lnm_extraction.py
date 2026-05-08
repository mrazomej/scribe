"""
Mixin for extracting LNM (Logistic-Normal Multinomial) population parameters.

Extracts the low-rank compositional covariance structure (mu, W, d) from the
VAE decoder weights and NumPyro parameter store.  Applicable to both ``lnm``
and ``lnmvcp`` base models.

Methods are guarded by :meth:`_require_lnm` which raises ``ValueError`` when
the model config is not an LNM variant.

Classes
-------
LNMExtractionMixin
    Mixin providing ``get_lnm_mu``, ``get_lnm_W``, ``get_lnm_d``,
    ``get_lnm_sigma``, and ``get_lnm_compositional_correlation``.

See Also
--------
scribe.svi.vae_results : ScribeVAEResults that composes this mixin.
scribe.de._transforms : ALR-to-CLR transformations used for the
    correlation extraction.
"""

from typing import Any, Optional, Tuple, cast

import jax.numpy as jnp

from ._latent_space import _DECODER_KEY


class LNMExtractionMixin:
    """Mixin providing LNM population parameter extraction from VAE results.

    Requires ``self.params``, ``self.model_config``, and ``self.n_genes`` to
    be present on the host class (satisfied by ``ScribeVAEResults``).
    """

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _require_lnm(self) -> None:
        """Raise ``ValueError`` if the model is not an LNM variant."""
        bm = getattr(self.model_config, "base_model", None)
        if bm not in ("lnm", "lnmvcp"):
            raise ValueError(
                "LNM extraction methods require model_config.base_model in "
                "{'lnm', 'lnmvcp'}, "
                f"got {bm!r}."
            )

    # ------------------------------------------------------------------
    # Decoder parameter helpers
    # ------------------------------------------------------------------

    def _y_alr_head_dense_params(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(kernel, bias)`` for the decoder's ``y_alr`` head.

        Navigates the Flax parameter tree to locate the linear dense layer
        that produces ALR coordinates.

        Returns
        -------
        kernel : jnp.ndarray
            Shape ``(k, G-1)`` — Flax Dense kernel (in_features, out_features).
        bias : jnp.ndarray
            Shape ``(G-1,)`` — dense bias vector.

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
        subtree = dec_params.get("head_y_alr")
        if subtree is None:
            raise KeyError(
                "Decoder params missing 'head_y_alr'. "
                "Expected a MultiHeadDecoder head with param_name 'y_alr'."
            )
        if not isinstance(subtree, dict):
            raise TypeError(
                f"head_y_alr params must be a dict, got {type(subtree)}."
            )
        # Flax may store params flat or under a Dense_0 sublayer
        if "kernel" in subtree and "bias" in subtree:
            return cast(jnp.ndarray, subtree["kernel"]), cast(
                jnp.ndarray, subtree["bias"]
            )
        dense0 = subtree.get("Dense_0")
        if isinstance(dense0, dict) and "kernel" in dense0 and "bias" in dense0:
            return cast(jnp.ndarray, dense0["kernel"]), cast(
                jnp.ndarray, dense0["bias"]
            )
        raise KeyError(
            "Could not find Dense kernel/bias under head_y_alr "
            "(expected Flax {'kernel','bias'} or nested Dense_0)."
        )

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    def get_lnm_mu(self) -> jnp.ndarray:
        """Extract the LNM population mean mu from the decoder bias.

        The ``y_alr`` head of the VAE decoder stores the ALR-space mean
        in the final dense bias vector (no output transform, or ``identity``
        transform).

        Returns
        -------
        jnp.ndarray
            Shape ``(G-1,)`` — ALR coordinates; reference gene index is
            ``model_config.alr_reference_idx`` (``-1`` means last gene).
        """
        self._require_lnm()
        _, bias = self._y_alr_head_dense_params()
        mu = jnp.asarray(bias)
        g1 = int(self.n_genes) - 1
        if mu.shape != (g1,):
            raise ValueError(
                f"Expected ALR bias shape (G-1,) = ({g1},), got {mu.shape}."
            )
        return mu

    def get_lnm_W(self) -> jnp.ndarray:
        """Extract the LNM low-rank factor ``W`` from the decoder kernel.

        Flax :class:`flax.linen.Dense` stores ``kernel`` with shape
        ``(in_features, out_features)`` = ``(k, G-1)`` for this head.
        The generative ALR covariance uses ``W`` with shape ``(G-1, k)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(G-1, k)`` with ``k`` the latent dimension.
        """
        self._require_lnm()
        kernel, bias = self._y_alr_head_dense_params()
        k_mat = jnp.asarray(kernel)
        g1 = int(self.n_genes) - 1
        if k_mat.shape[-1] != g1:
            raise ValueError(
                f"Expected kernel out_features G-1 = {g1}, "
                f"got kernel shape {k_mat.shape}."
            )
        W = k_mat.T
        if W.shape[0] != g1:
            raise ValueError(f"Internal shape error: W.shape = {W.shape}.")
        return W

    def get_lnm_d(self) -> Optional[jnp.ndarray]:
        """Extract the learned diagonal ``d`` from NumPyro / SVI params.

        When :attr:`~scribe.models.config.ModelConfig.d_mode` is
        ``'learned'``, the population ALR covariance includes
        :math:`\\mathrm{diag}(d)` with :math:`d` a learned positive vector of
        length ``G-1`` (site ``d_lnm``).  If ``d_mode='low_rank'``, this
        method returns ``None``.

        When ``d_lnm`` is present in :attr:`params` but ``d_mode`` is still the
        default ``low_rank``, the explicit ``d_mode`` flag takes precedence
        (returns ``None`` unless ``learned``).

        Returns
        -------
        jnp.ndarray or None
            Shape ``(G-1,)`` when ``d`` is learned and found in ``params``;
            ``None`` for low-rank-only compositional uncertainty.
        """
        self._require_lnm()

        def _recursive_find_d_lnm(obj: Any) -> Optional[jnp.ndarray]:
            if isinstance(obj, dict):
                if "d_lnm" in obj:
                    raw = obj["d_lnm"]
                    if isinstance(raw, dict) and "loc" in raw:
                        return jnp.asarray(raw["loc"])
                    return jnp.asarray(raw)
                # The mean-field guide builder stores variational parameters
                # as ``{name}_loc`` / ``{name}_scale`` via numpyro.param
                # (see _guide_meanfield_mixin), so the MAP estimate lives
                # at ``d_lnm_loc`` rather than a nested ``d_lnm`` dict.
                if "d_lnm_loc" in obj:
                    return jnp.asarray(obj["d_lnm_loc"])
                for v in obj.values():
                    got = _recursive_find_d_lnm(v)
                    if got is not None:
                        return got
            return None

        explicit_mode = getattr(self.model_config, "d_mode", "low_rank")
        d_arr = _recursive_find_d_lnm(self.params)

        if explicit_mode == "low_rank":
            return None
        if explicit_mode != "learned":
            raise ValueError(
                f"model_config.d_mode must be 'low_rank' or 'learned', "
                f"got {explicit_mode!r}."
            )
        if d_arr is None:
            raise ValueError(
                "model_config.d_mode is 'learned' but no 'd_lnm' "
                "parameter was found in params."
            )
        g1 = int(self.n_genes) - 1
        if d_arr.shape != (g1,):
            raise ValueError(
                f"Expected d_lnm shape (G-1,) = ({g1},), got {d_arr.shape}."
            )
        return d_arr

    def get_lnm_sigma(self) -> jnp.ndarray:
        """Reconstruct full ALR covariance ``Sigma = W W^T + diag(d)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(G-1, G-1)``. Useful only for small ``G``; for large
            ``G`` prefer :meth:`get_lnm_W` and :meth:`get_lnm_d`.

        Notes
        -----
        In ``low_rank`` mode, ``d`` is treated as zero on the diagonal.
        """
        self._require_lnm()
        W = self.get_lnm_W()
        d_opt = self.get_lnm_d()
        d_alr = (
            jnp.asarray(d_opt)
            if d_opt is not None
            else jnp.zeros(W.shape[0], dtype=W.dtype)
        )
        return W @ W.T + jnp.diag(d_alr)

    def _lnm_reference_idx(self) -> int:
        """Return the ALR reference gene index from model config."""
        return getattr(self.model_config, "alr_reference_idx", -1)

    def get_lnm_compositional_correlation(self) -> jnp.ndarray:
        """Compositional correlation matrix in CLR space.

        Transforms the low-rank Gaussian in ALR coordinates to CLR using
        :func:`scribe.de._transforms.transform_gaussian_alr_to_clr`, builds
        the full CLR covariance, then converts to a correlation matrix.

        Returns
        -------
        jnp.ndarray
            Shape ``(G, G)`` correlation matrix in CLR coordinates.
        """
        self._require_lnm()
        from ..de._transforms import transform_gaussian_alr_to_clr

        mu_alr = self.get_lnm_mu()
        W_alr = self.get_lnm_W()
        d_opt = self.get_lnm_d()
        d_alr = (
            jnp.asarray(d_opt)
            if d_opt is not None
            else jnp.zeros(mu_alr.shape[0], dtype=mu_alr.dtype)
        )
        _mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(
            mu_alr, W_alr, d_alr, reference_idx=self._lnm_reference_idx()
        )
        sigma_clr = W_clr @ W_clr.T + jnp.diag(d_clr)
        std = jnp.sqrt(jnp.maximum(jnp.diag(sigma_clr), 1e-30))
        return sigma_clr / (std[:, None] * std[None, :])

    def get_lnm_library_size_direction(self) -> jnp.ndarray:
        """Latent-space unit vector whose ``W``-image is closest to ``1_{G-1}``.

        Wrapper over
        :func:`scribe.stats.correlation_diagnostics.library_size_direction`
        for VAE-fit LNM results. Operates in ALR coordinates (so
        ``1_{G-1}``, not ``1_G``); see the helper for the structural-
        redundancy rationale.

        Note: in ALR space the all-ones direction does not have the
        same biological interpretation as in PLN's log-rate space —
        it corresponds to "all non-reference genes shift in lock-step
        relative to the reference gene". Whether this captures
        library-size leakage depends on which gene was chosen as the
        ALR reference.
        """
        self._require_lnm()
        from ..stats.correlation_diagnostics import library_size_direction
        return library_size_direction(self.get_lnm_W())

    def get_lnm_correlation_residual(
        self,
        method: str = "library_size",
        n_components: int = 1,
        include_diagonal_d: bool = False,
    ) -> jnp.ndarray:
        """LNM gene-gene correlation with latent direction(s) projected out.

        Operates in ALR coordinates. See
        :func:`scribe.stats.correlation_diagnostics.correlation_residual`
        for the math.
        """
        self._require_lnm()
        from ..stats.correlation_diagnostics import correlation_residual
        d = self.get_lnm_d()
        return correlation_residual(
            self.get_lnm_W(), d,
            method=method,
            n_components=n_components,
            include_diagonal_d=include_diagonal_d,
        )

    def summarize_lnm_correlation_structure(
        self,
        *,
        n_top_eig: int = 10,
        verbose: bool = True,
    ):
        """Print and return a diagnostic summary of the LNM correlation structure.

        Wrapper over
        :func:`scribe.stats.correlation_diagnostics.summarize_correlation_structure`
        for VAE-fit LNM results. Operates in ALR coordinates.
        """
        self._require_lnm()
        from ..stats.correlation_diagnostics import (
            summarize_correlation_structure,
        )
        return summarize_correlation_structure(
            self.get_lnm_W(), self.get_lnm_d(),
            space_label="ALR space",
            model_label="LNM VAE",
            n_top_eig=n_top_eig,
            verbose=verbose,
        )

    def get_lnm_compositional_samples(
        self,
        n_samples: int = 2048,
        rng_key=None,
        chunk_size: int = 256,
        store_samples: bool = True,
    ):
        """Draw simplex compositions from the fitted LNM(VCP) marginal.

        Generates ``n_samples`` independent imaginary cells from the
        model's generative marginal:

            z ∼ 𝒩(0, 𝐈ₖ), ε ∼ 𝒩(0, 𝐈_{G−1})
            y_alr = μ + 𝑊 z + √d ⊙ ε
            ρ = softmax_full(augment_with_zero(y_alr, ref_idx))

        Each draw is an independent imaginary cell from the model's
        fitted population distribution.  Mirrors
        :meth:`ScribeLaplaceResults.get_compositional_samples` so the
        DE pipeline can call it polymorphically.

        Parameters
        ----------
        n_samples : int, default 2048
        rng_key : jax.Array, optional
        chunk_size : int, default 256
        store_samples : bool, default True

        Returns
        -------
        np.ndarray, shape ``(n_samples, G)``
        """
        self._require_lnm()
        import jax
        import numpy as _np

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        mu = jnp.asarray(self.get_lnm_mu())
        W = jnp.asarray(self.get_lnm_W())
        d_opt = self.get_lnm_d()
        d = (
            jnp.asarray(d_opt)
            if d_opt is not None
            else jnp.zeros(W.shape[0], dtype=W.dtype)
        )
        sqrt_d = jnp.sqrt(jnp.maximum(d, 0.0))
        G_minus1 = int(mu.shape[0])
        n_genes_full = G_minus1 + 1
        k = int(W.shape[1])

        ref_idx = int(self._lnm_reference_idx())
        if ref_idx < 0:
            ref_idx = n_genes_full + ref_idx

        other = jnp.asarray(
            [g for g in range(n_genes_full) if g != ref_idx]
        )

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
            eps = jax.random.normal(k_eps, (size, G_minus1), dtype=mu.dtype)
            latent = mu[None, :] + z @ W.T + sqrt_d[None, :] * eps
            full = jnp.zeros((size, n_genes_full), dtype=latent.dtype)
            full = full.at[..., other].set(latent)
            simplex = jax.nn.softmax(full, axis=-1)
            pieces.append(_np.asarray(simplex))

        out = _np.concatenate(pieces, axis=0)
        if store_samples:
            self.compositional_samples = out
        return out
