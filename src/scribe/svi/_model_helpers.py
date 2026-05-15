"""Model and guide helpers shared by SVI-style results classes.

This module centralizes reconstruction of model/guide callables from a stored
``ModelConfig``.  For VAE results, reconstruction must be dimension-consistent
with the decoder parameter subtree to avoid shape errors during dry-run
validation and posterior sampling.
"""

from typing import Callable, Tuple, Optional
import warnings

from ..models.config import InferenceMethod

# ==============================================================================
# Model Helpers Mixin
# ==============================================================================


class ModelHelpersMixin:
    """Mixin providing model and guide access helpers."""

    def _is_vae_result_config(self) -> bool:
        """Return whether this results object uses the composable VAE path.

        Returns
        -------
        bool
            ``True`` when the model config corresponds to VAE inference and
            includes a VAE block; otherwise ``False``.
        """
        model_config = getattr(self, "model_config", None)
        if model_config is None:
            return False
        return (
            getattr(model_config, "inference_method", None)
            == InferenceMethod.VAE
            and getattr(model_config, "vae", None) is not None
        )

    def _infer_decoder_gene_width(self) -> Optional[int]:
        """Infer the factory ``n_genes`` from decoder parameter shapes.

        Some decoder heads produce fewer outputs than ``n_genes`` (e.g. the
        ALR head outputs ``n_genes - 1``).  To correctly recover the factory
        ``n_genes``, this method cross-references each head's *stored*
        ``output_dim`` against the *actual* parameter width from the
        ``vae_decoder$params`` subtree.  The offset ``param_width - output_dim``
        is then subtracted from the param width to recover ``n_genes``.

        Returns
        -------
        Optional[int]
            The inferred factory ``n_genes`` when decoder head params are
            available, otherwise ``None``.
        """
        params = getattr(self, "params", None)
        if not isinstance(params, dict):
            return None
        decoder_params = params.get("vae_decoder$params")
        if not isinstance(decoder_params, dict):
            return None

        # Build a lookup of head param_name → stored output_dim so we can
        # compute the offset each head applies to n_genes.
        decoder = getattr(self, "_decoder", None)
        head_output_dims = {}
        if decoder is not None and hasattr(decoder, "output_heads"):
            for head in decoder.output_heads:
                head_output_dims[head.param_name] = int(head.output_dim)

        inferred_n_genes = []
        for key, value in decoder_params.items():
            if not str(key).startswith("head_") or not isinstance(value, dict):
                continue
            # Extract param width from bias or kernel
            param_width = None
            bias = value.get("bias")
            if hasattr(bias, "shape") and len(bias.shape) == 1:
                param_width = int(bias.shape[0])
            else:
                kernel = value.get("kernel")
                if hasattr(kernel, "shape") and len(kernel.shape) == 2:
                    param_width = int(kernel.shape[1])
            if param_width is None:
                continue

            # Recover factory n_genes using the head's stored output_dim.
            # For a head built with ``output_dim = n_genes + offset``,
            # we know ``param_width == output_dim`` and can solve for n_genes
            # using the stored output_dim's relationship to n_genes.
            head_name = str(key).removeprefix("head_")
            stored_dim = head_output_dims.get(head_name)
            if stored_dim is not None and param_width == stored_dim:
                # Params are consistent with the stored module — no
                # subsetting has corrupted anything.  But output_dim may
                # differ from n_genes by an offset (e.g. ALR = n_genes-1).
                # We don't know the offset directly, so just trust
                # results.n_genes in this consistent case.
                inferred_n_genes.append(None)
            else:
                # Fallback: treat param width as n_genes (original behavior
                # for non-ALR heads or when decoder module is unavailable).
                inferred_n_genes.append(param_width)

        # Filter out None entries (consistent heads where we defer to
        # results.n_genes) and check remaining inferred values.
        concrete = [v for v in inferred_n_genes if v is not None]
        if not concrete:
            # All heads are consistent with the stored module — defer to
            # results.n_genes.
            return None
        width_set = set(concrete)
        if len(width_set) != 1:
            widths_text = ", ".join(str(w) for w in sorted(width_set))
            raise ValueError(
                "Inconsistent VAE decoder head widths found in "
                f"'vae_decoder$params': {{{widths_text}}}."
            )
        return concrete[0]

    def _resolve_vae_factory_n_genes(self, n_genes: int) -> int:
        """Resolve reconstruction width from VAE decoder parameter subtree.

        Parameters
        ----------
        n_genes : int
            Gene width that will be passed to ``get_model_and_guide``.

        Returns
        -------
        int
            Gene width to use for VAE model/guide reconstruction.
        """
        decoder_width = self._infer_decoder_gene_width()
        if decoder_width is None:
            return int(n_genes)
        if int(decoder_width) != int(n_genes):
            warnings.warn(
                "VAE model reconstruction gene-width mismatch: "
                f"results.n_genes={int(n_genes)} but "
                f"'vae_decoder$params' implies {int(decoder_width)} genes. "
                "Using decoder-parameter width for model/guide reconstruction.",
                UserWarning,
                stacklevel=2,
            )
            return int(decoder_width)
        return int(n_genes)

    def _factory_n_genes(self) -> Optional[int]:
        """Resolve ``n_genes`` used to rebuild model and guide callables.

        Returns the fitted-results gene width.  For VAE results the
        width may require an alignment step (``_resolve_vae_factory_n_genes``);
        for non-VAE results we still need to pass ``n_genes`` so the
        factory's dry-run validation works against parameterizations
        whose specs carry per-gene arrays (e.g. TwoState's
        data-driven ``mu_prior_loc``).  Without that the validation
        falls back to ``n_genes=5`` and trips on the array shape.
        """
        n_genes = int(getattr(self, "n_genes"))
        if not self._is_vae_result_config():
            return n_genes
        return self._resolve_vae_factory_n_genes(n_genes)

    # --------------------------------------------------------------------------
    # Get model and guide functions
    # --------------------------------------------------------------------------

    def _model_and_guide(self) -> Tuple[Callable, Optional[Callable]]:
        """Get model and guide functions for the current results object.

        Returns
        -------
        Tuple[Callable, Optional[Callable]]
            Pair ``(model, guide)`` reconstructed from ``model_config``.
        """
        from ..models.model_registry import get_model_and_guide

        model, guide, _ = get_model_and_guide(
            self.model_config,
            n_genes=self._factory_n_genes(),
        )
        return model, guide

    # --------------------------------------------------------------------------
    # Get parameterization
    # --------------------------------------------------------------------------

    def _parameterization(self) -> str:
        """Get the parameterization type."""
        return self.model_config.parameterization or ""

    # --------------------------------------------------------------------------
    # Get if unconstrained
    # --------------------------------------------------------------------------

    def _unconstrained(self) -> bool:
        """Get if the parameterization is unconstrained."""
        return self.model_config.unconstrained

    # --------------------------------------------------------------------------
    # Get log likelihood function
    # --------------------------------------------------------------------------

    def _log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for this model type."""
        from ..models.model_registry import get_log_likelihood_fn

        return get_log_likelihood_fn(self.model_type)
