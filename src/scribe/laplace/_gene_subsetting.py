"""Gene-subsetting mixin for Laplace results.

This module contains gene-axis slicing logic for both PLN and LNM-family
Laplace fits. LNM requires additional ALR-specific handling to preserve the
reference-gene convention when creating subset views.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Union

import jax.numpy as jnp
import numpy as np

from ._results_shared import _base_model, _subset_var


def _subset_w_prior_diagnostics(
    diag: Optional[Dict[str, Any]],
    W_subset: jnp.ndarray,
) -> Optional[Dict[str, Any]]:
    """Recompute gene-dependent W-prior diagnostics on a subsetted W.

    Phase-3 Round-1 fix 5 + Round-3 fix 2: the headline rank diagnostic
    must be computed from the *subset-centered* loadings so it stays
    consistent with the gauge-invariant convention used during fitting
    (the prior targets ``W_⟂``, not raw ``W``).  Factor-level entries
    (``sigma_k``, ``tau``, ``psi_k``, ``gamma``, ``scale_effective_rank``,
    ``strategy_type``) carry over unchanged; gene-dependent entries
    (``column_frobenius_compositional``, ``column_frobenius_raw``,
    ``column_norm_effective_rank``, ``effective_rank``) are recomputed
    from the subsetted W.  A ``"_subset_view": True`` marker is added
    so downstream consumers can distinguish a full-fit diagnostics dict
    from a subset-view one.
    """
    if diag is None:
        return None
    out = dict(diag)
    # Compositional (gauge-cleaned) subset columns drive the headline rank.
    W_subset_perp = W_subset - jnp.mean(W_subset, axis=0, keepdims=True)
    col_norm_comp = jnp.linalg.norm(W_subset_perp, axis=0)
    col_norm_raw = jnp.linalg.norm(W_subset, axis=0)
    out["column_frobenius_compositional"] = col_norm_comp
    out["column_frobenius_raw"] = col_norm_raw
    if col_norm_comp.size > 0:
        thr = 0.05 * float(jnp.max(col_norm_comp))
        rank = int(jnp.sum(col_norm_comp > thr))
        out["column_norm_effective_rank"] = rank
        out["effective_rank"] = rank
    out["_subset_view"] = True
    return out


class GeneSubsettingResultsMixin:
    """Mixin enabling ``results[gene_index]`` style slicing."""

    def __getitem__(
        self,
        gene_index: Union[int, slice, np.ndarray, jnp.ndarray, list],
    ):
        """Return a gene-subset Laplace results view.

        Parameters
        ----------
        gene_index : int, slice, array-like
            Gene indexer interpreted in full-gene coordinates. Boolean masks
            are converted to integer indices.

        Returns
        -------
        ScribeLaplaceResults
            New results object with gene-axis tensors subsetted.

        Raises
        ------
        NotImplementedError
            If the result carries an unsupported base model tag.
        """
        if isinstance(gene_index, (int, np.integer)):
            idx = np.asarray([int(gene_index)])
        elif isinstance(gene_index, slice):
            idx = np.asarray(range(*gene_index.indices(self.n_genes)))
        else:
            idx = np.asarray(gene_index)
        if idx.dtype == bool:
            idx = np.where(idx)[0]

        bm = _base_model(self.model_config)
        if bm in ("pln", "nbln"):
            # PLN and NBLN share the same gene-axis tensor shapes
            # (``mu``, ``W``, ``d`` are length-G; ``x_loc`` is
            # ``(N, G)``). NBLN additionally has a length-G ``r``
            # that needs to be sliced by the same gene index.
            return self._subset_pln(idx)
        if bm in ("lnm", "lnmvcp"):
            return self._subset_lnm(idx)
        raise NotImplementedError(f"__getitem__ not implemented for base_model={bm!r}")

    def _subset_pln(self, idx: np.ndarray):
        """Subset PLN/NBLN gene-axis tensors and metadata.

        Parameters
        ----------
        idx : np.ndarray
            Integer gene indices in original PLN/NBLN gene space.

        Returns
        -------
        ScribeLaplaceResults
            New PLN/NBLN results view with sliced ``mu``, ``W``, ``d``,
            ``x_loc`` (and ``r, r_loc, r_scale`` for NBLN), with aligned
            ``var`` metadata.

        Notes
        -----
        Subsetted uncertainty fields (``r_loc``, ``r_scale``) are
        marginals from the model fitted on the full gene panel, not the
        posterior one would obtain by refitting on only the selected genes.
        """
        idx_jnp = jnp.asarray(idx)
        W_subset = self.W[idx_jnp, :]
        return replace(
            self,
            mu=self.mu[idx_jnp],
            W=W_subset,
            d=self.d[idx_jnp],
            x_loc=self.x_loc[:, idx_jnp] if self.x_loc is not None else None,
            r=self.r[idx_jnp] if self.r is not None else None,
            r_loc=self.r_loc[idx_jnp] if self.r_loc is not None else None,
            r_scale=(
                self.r_scale[idx_jnp] if self.r_scale is not None else None
            ),
            mu_loc=(
                self.mu_loc[idx_jnp] if self.mu_loc is not None else None
            ),
            mu_scale=(
                self.mu_scale[idx_jnp] if self.mu_scale is not None else None
            ),
            # Phase-3: recompute gene-dependent W-prior diagnostics
            # against the subsetted W.  Factor-level entries (sigma_k,
            # tau, etc.) carry over unchanged.  See Round-3 fix 2 +
            # Round-1 fix 5 in the plan.
            w_prior_diagnostics=_subset_w_prior_diagnostics(
                getattr(self, "w_prior_diagnostics", None), W_subset,
            ),
            n_genes=int(len(idx)),
            n_vars=int(len(idx)) if self.n_vars is not None else None,
            var=_subset_var(self.var, idx),
            _subset_gene_index=idx,
        )

    def _subset_lnm(self, idx: np.ndarray):
        """Subset LNM/LNMVCP tensors while preserving ALR reference semantics.

        Parameters
        ----------
        idx : np.ndarray
            Integer gene indices in original full-gene coordinates.

        Returns
        -------
        ScribeLaplaceResults
            New LNM-family results view with ALR-axis tensors correctly mapped
            to the subsetted gene set.

        Raises
        ------
        ValueError
            If ``alr_reference_idx`` is missing or the subset omits the
            reference gene.

        Notes
        -----
        ALR coordinates have length ``G-1`` and exclude the reference gene.
        This method maps user-provided full-gene indices to ALR positions,
        preserving reference-gene validity and updating the new reference
        position in subset coordinates.

        LNM totals uncertainty (``mu_T_loc``, ``mu_T_scale``,
        ``r_T_loc``, ``r_T_scale``, ``totals_cov``) is scalar/global
        and passes through unchanged when the ALR reference gene
        remains in the subset.  These are marginals from the model
        fitted on the full gene panel, not the posterior from a refit
        on the selected gene subset.
        """
        ref = self.alr_reference_idx
        if ref is None:
            raise ValueError(
                "LNM subsetting needs alr_reference_idx; this result appears to "
                "have been constructed without it."
            )
        if ref not in set(idx.tolist()):
            raise ValueError(
                f"Gene subset must include the ALR reference gene "
                f"(index={ref!r}); got idx={idx!r}."
            )

        n_g = self.n_genes
        alr_axis_pos = np.full(n_g, -1, dtype=int)
        alr_axis_pos[:ref] = np.arange(ref)
        alr_axis_pos[ref + 1 :] = np.arange(ref, n_g - 1)
        idx_no_ref = idx[idx != ref]
        idx_alr_jnp = jnp.asarray(alr_axis_pos[idx_no_ref])
        new_ref_pos = int(np.where(idx == ref)[0][0])

        return replace(
            self,
            mu=self.mu[idx_alr_jnp],
            W=self.W[idx_alr_jnp, :],
            d=self.d[idx_alr_jnp],
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

