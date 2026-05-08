"""Gene-subsetting mixin for Laplace results.

This module contains gene-axis slicing logic for both PLN and LNM-family
Laplace fits. LNM requires additional ALR-specific handling to preserve the
reference-gene convention when creating subset views.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Union

import jax.numpy as jnp
import numpy as np

from ._results_shared import _base_model, _subset_var


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
        if bm == "pln":
            return self._subset_pln(idx)
        if bm in ("lnm", "lnmvcp"):
            return self._subset_lnm(idx)
        raise NotImplementedError(f"__getitem__ not implemented for base_model={bm!r}")

    def _subset_pln(self, idx: np.ndarray):
        """Subset PLN gene-axis tensors and metadata.

        Parameters
        ----------
        idx : np.ndarray
            Integer gene indices in original PLN gene space.

        Returns
        -------
        ScribeLaplaceResults
            New PLN results view with sliced ``mu``, ``W``, ``d``, ``x_loc``,
            and aligned ``var`` metadata.
        """
        idx_jnp = jnp.asarray(idx)
        return replace(
            self,
            mu=self.mu[idx_jnp],
            W=self.W[idx_jnp, :],
            d=self.d[idx_jnp],
            x_loc=self.x_loc[:, idx_jnp] if self.x_loc is not None else None,
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

