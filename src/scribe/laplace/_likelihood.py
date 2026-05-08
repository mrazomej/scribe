"""Likelihood mixin for Laplace results.

This module exposes the public MAP log-likelihood entry point and routes to
model-specific helper backends.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._results_likelihood_helpers import _ll_lnm, _ll_nbln, _ll_pln
from ._results_shared import _base_model


class LikelihoodResultsMixin:
    """Mixin exposing MAP log-likelihood helpers for Laplace results."""

    def get_log_likelihood(
        self,
        counts: jnp.ndarray,
        return_by: str = "cell",
    ) -> jnp.ndarray:
        """Evaluate MAP log-likelihood for PLN/LNM-family fits.

        Parameters
        ----------
        counts : jnp.ndarray
            Observed count matrix with shape ``(n_cells, n_genes)``.
        return_by : {"cell", "gene"}, default="cell"
            Aggregation direction for the returned values.

        Returns
        -------
        jnp.ndarray
            Per-cell or per-gene log-likelihood contributions computed under
            the Laplace MAP state.

        Raises
        ------
        ValueError
            If ``return_by`` is not ``"cell"`` or ``"gene"``.
        NotImplementedError
            If dispatch encounters an unsupported base model.
        """
        if return_by not in ("cell", "gene"):
            raise ValueError(
                f"return_by must be 'cell' or 'gene'; got {return_by!r}."
            )
        bm = _base_model(self.model_config)
        if bm == "pln":
            return _ll_pln(counts, self.x_loc, self.eta_loc, return_by)
        if bm == "nbln":
            if self.r is None:
                raise ValueError(
                    "NBLN log-likelihood requires the gene dispersion 'r' "
                    "field on the result. This result was constructed "
                    "without it -- check the inference path."
                )
            return _ll_nbln(
                counts, self.x_loc, self.eta_loc, self.r, return_by
            )
        if bm in ("lnm", "lnmvcp"):
            return _ll_lnm(
                counts,
                self.mu,
                self.W,
                self.z_loc,
                self.y_alr_loc,
                self.alr_reference_idx,
                return_by,
            )
        raise NotImplementedError(
            f"get_log_likelihood not implemented for base_model={bm!r}"
        )

