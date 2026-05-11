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

from ._global_uncertainty import resolve_positive_fns
from ._results_sampling_helpers import (
    _ppc_lnm_library_anchored,
    _ppc_lnm_marginal,
    _ppc_lnm_per_cell,
    _ppc_lnm_per_cell_laplace,
    _ppc_nbln_marginal,
    _ppc_nbln_per_cell,
    _ppc_nbln_per_cell_laplace,
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
            if bm in ("pln", "nbln"):
                # Library-anchored PPC samples ``softmax(x) ->
                # Multinomial`` against observed library size; the NB
                # vs Poisson choice does not enter (no count-noise
                # layer at this level), so NBLN reuses the PLN helper.
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
        if bm == "nbln":
            if self.r is None:
                raise ValueError(
                    "NBLN PPC requires the gene dispersion 'r' field."
                )
            pos_fwd, _ = resolve_positive_fns(self.model_config)
            return _ppc_nbln_marginal(
                rng_key,
                n_samples,
                self.mu,
                self.W,
                self.d,
                self.r,
                eta_loc=self.eta_loc,
                r_loc=self.r_loc,
                r_scale=self.r_scale,
                pos_forward=pos_fwd,
            )
        if bm in ("lnm", "lnmvcp"):
            pos_fwd, _ = resolve_positive_fns(self.model_config)
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
                totals_cov=self.totals_cov,
                mu_T_loc=self.mu_T_loc,
                r_T_loc=self.r_T_loc,
                pos_forward=pos_fwd,
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
        if bm == "nbln":
            if self.r is None:
                raise ValueError(
                    "NBLN PPC requires the gene dispersion 'r' field."
                )
            pos_fwd, _ = resolve_positive_fns(self.model_config)
            return _ppc_nbln_per_cell_laplace(
                rng_key,
                n_samples,
                self.x_loc,
                self.eta_loc,
                self.W,
                self.d,
                self.r,
                r_loc=self.r_loc,
                r_scale=self.r_scale,
                pos_forward=pos_fwd,
            )
        if bm in ("lnm", "lnmvcp"):
            pos_fwd, _ = resolve_positive_fns(self.model_config)
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
                totals_cov=self.totals_cov,
                mu_T_loc=self.mu_T_loc,
                r_T_loc=self.r_T_loc,
                pos_forward=pos_fwd,
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
        if bm == "nbln":
            if self.r is None:
                raise ValueError(
                    "NBLN PPC requires the gene dispersion 'r' field."
                )
            return _ppc_nbln_per_cell(
                rng_key,
                n_samples,
                self.x_loc,
                self.eta_loc,
                self.r,
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

    def get_compositional_samples(
        self,
        n_samples: int = 2048,
        rng_key: Optional[jax.Array] = None,
        chunk_size: int = 256,
        store_samples: bool = True,
    ):
        """Draw simplex compositions from the fitted marginal distribution.

        Generates samples from the model's *generative marginal* — the
        same procedure as the ``level="marginal"`` PPC sampler but
        stops at the simplex step (no observation noise applied):

            z ∼ 𝒩(0, 𝐈ₖ),  ε ∼ 𝒩(0, 𝐈_{Gₑₑ𝚏𝚏})
            latent = μ + 𝑊 z + √d ⊙ ε
            ρ = softmax_full(latent)

        For LNM / LNMVCP, the latent lives in (G−1)-dim ALR space and
        is augmented with a zero at ``alr_reference_idx`` before
        softmax — the standard ALR⁻¹ map. For PLN, the latent is
        G-dim log-rate space and softmax is applied directly; the
        per-cell capture offset η cancels because softmax is
        invariant to a global additive shift (the rigid-translation
        degeneracy in operation).

        Each draw is an *independent imaginary cell* from the
        model's fitted population distribution — no observed-cell
        information enters any of the latents. This is the right
        sample source for population-level downstream analyses
        (empirical DE, gene-set tests, biological summaries on
        compositions) on LNM and PLN fits, mirroring
        ``ScribeMCMCResults.get_compositional_samples`` for the
        DM/NB-family path.

        Parameters
        ----------
        n_samples : int, default 2048
            Total number of fresh simplex draws.
        rng_key : jax.Array, optional
            JAX PRNG key. Defaults to ``jax.random.PRNGKey(0)``.
        chunk_size : int, default 256
            Per-chunk sample count. Each chunk is moved to host
            memory before the next is drawn so the device peak
            stays bounded by ``chunk_size · G · 4 bytes``.
        store_samples : bool, default True
            If True, store the returned array in
            ``self.compositional_samples`` for reuse without
            re-sampling.

        Returns
        -------
        np.ndarray, shape ``(n_samples, G)``
            Full-dimensional simplex samples (rows sum to 1).

        Raises
        ------
        NotImplementedError
            If the base model is not PLN, LNM, or LNMVCP.
        """
        import numpy as _np

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        bm = _base_model(self.model_config)
        if bm in ("lnm", "lnmvcp"):
            mu = jnp.asarray(self.mu)
            W = jnp.asarray(self.W)
            d = self.d
            ref_idx = int(getattr(self.model_config, "alr_reference_idx", -1))
            n_genes_full = int(W.shape[0]) + 1
            if ref_idx < 0:
                ref_idx = n_genes_full + ref_idx
            is_alr = True
        elif bm in ("pln", "nbln"):
            # PLN and NBLN both produce log-rates ``y_log_rate = mu + W z``;
            # softmax of those log-rates gives the compositional sample.
            # The NB vs Poisson choice never enters here -- compositions
            # are pre-observation-noise.
            mu = jnp.asarray(self.mu)
            W = jnp.asarray(self.W)
            d = self.d
            ref_idx = None
            n_genes_full = int(W.shape[0])
            is_alr = False
        else:
            raise NotImplementedError(
                f"get_compositional_samples not implemented for "
                f"base_model={bm!r}"
            )

        if d is None:
            d = jnp.zeros(W.shape[0], dtype=W.dtype)
        else:
            d = jnp.asarray(d)
        sqrt_d = jnp.sqrt(jnp.maximum(d, 0.0))

        G_eff = int(mu.shape[0])
        k = int(W.shape[1])

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
            eps = jax.random.normal(k_eps, (size, G_eff), dtype=mu.dtype)
            latent = mu[None, :] + z @ W.T + sqrt_d[None, :] * eps

            if is_alr:
                # ALR → simplex: augment with a zero at the reference
                # position, then softmax over G dims.
                full = jnp.zeros((size, n_genes_full), dtype=latent.dtype)
                other = [g for g in range(n_genes_full) if g != ref_idx]
                full = full.at[..., jnp.asarray(other)].set(latent)
                simplex = jax.nn.softmax(full, axis=-1)
            else:
                # PLN: softmax of log-rate; η cancels.
                simplex = jax.nn.softmax(latent, axis=-1)

            pieces.append(_np.asarray(simplex))

        out = _np.concatenate(pieces, axis=0)
        if store_samples:
            self.compositional_samples = out
        return out
