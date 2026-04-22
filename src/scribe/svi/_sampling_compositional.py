"""
Compositional sampling mixin for SVI results.

Provides ``get_compositional_samples`` — a single-call interface that
auto-generates posterior samples if they do not yet exist, auto-detects
whether gene-specific ``p`` is present (hierarchical-p models), slices
mixture components when requested, and returns a ``(N_total, n_genes)``
simplex array on the host (CPU).
"""

from typing import Optional

import numpy as np
from jax import random


class CompositionalSamplingMixin:
    """Mixin adding :meth:`get_compositional_samples` to SVI results."""

    def get_compositional_samples(
        self,
        n_samples: int = 100,
        n_samples_dirichlet: int = 1,
        component: Optional[int] = None,
        rng_key=None,
        batch_size: int = 2048,
        store_samples: bool = True,
        counts=None,
    ) -> np.ndarray:
        """Draw simplex compositions from the variational posterior.

        Produces a ``(N_total, n_genes)`` array of compositions on the
        simplex, automatically choosing the correct sampling path:

        - **Dirichlet** (default / shared ``p``): draws
          :math:`\\rho^{(s)} \\sim \\mathrm{Dir}(r^{(s)})` per posterior
          sample.  A scalar ``p`` (shared across genes) cancels in the
          normalization and is silently dropped, reducing to this path.
        - **Gamma-normalize** (hierarchical / gene-specific ``p``): draws
          :math:`\\gamma_g \\sim \\Gamma(r_g, 1)`, scales by
          :math:`p_g / (1 - p_g)`, and normalizes.  Automatically
          detected from the ``AxisLayout`` of ``p`` in the posterior
          samples — a gene axis in the layout triggers this path.

        If ``self.posterior_samples`` is ``None`` when this method is
        called, ``get_posterior_samples`` is invoked automatically with
        ``n_samples`` draws.

        Parameters
        ----------
        n_samples : int, default=100
            Number of posterior samples to draw from the variational guide
            when ``self.posterior_samples`` is ``None``.  Ignored when
            posterior samples already exist.
        n_samples_dirichlet : int, default=1
            Number of simplex draws per posterior sample.  The returned
            array has shape ``(N * n_samples_dirichlet, n_genes)`` where
            ``N`` is the number of posterior samples.
        component : int, optional
            Mixture component index to extract from 3-D ``r`` (and ``p``)
            arrays produced by mixture models.  Required when the model has
            a component axis; pass the component index of interest.
        rng_key : jax.random.PRNGKey, optional
            JAX PRNG key used for the Dirichlet / Gamma draws.  Defaults to
            ``jax.random.PRNGKey(0)``.
        batch_size : int, default=2048
            Upper-bound chunk size for GPU-batched composition sampling.
            The adaptive memory layer may use a larger chunk when GPU memory
            allows.
        store_samples : bool, default=True
            If ``True``, stores the returned array in
            ``self.compositional_samples`` so it can be reused without
            re-sampling.
        counts : array-like or None, optional
            Observed count matrix ``(n_cells, n_genes)``.  Required when
            the model uses amortized capture probability so that the guide
            can compute sufficient statistics.  For non-amortized models
            this can be ``None``.

        Returns
        -------
        numpy.ndarray, shape ``(N_total, n_genes)``
            Simplex compositions on the CPU, one row per draw.
            ``N_total = N_posterior_samples * n_samples_dirichlet``.

        Raises
        ------
        ValueError
            If the model has a component axis but ``component`` is ``None``.

        See Also
        --------
        scribe.de.sample_composition : Underlying single-condition sampler.
        sample_compositions : Two-condition version used by the DE pipeline.
        get_posterior_samples : Draws variational posterior parameter samples.

        Examples
        --------
        Standard (Dirichlet) path:

        >>> simplex = results.get_compositional_samples(n_samples=200)
        >>> simplex.shape  # (200, n_genes)

        Hierarchical-p model:

        >>> simplex = results_hier.get_compositional_samples()
        >>> # automatically uses Gamma-normalize path

        Mixture model, component 0:

        >>> simplex = results_mix.get_compositional_samples(component=0)
        """
        # Auto-generate posterior samples if missing
        if self.posterior_samples is None:
            if rng_key is None:
                _post_key = random.PRNGKey(42)
            else:
                _post_key, rng_key = random.split(rng_key)
            self.get_posterior_samples(
                rng_key=_post_key,
                n_samples=n_samples,
                store_samples=True,
                convert_to_numpy=True,
                counts=counts,
            )

        # Extract r and (optionally) p from the cached posterior samples
        r_samples = self.posterior_samples["r"]
        p_samples = self.posterior_samples.get("p")

        # Build semantic axis layouts for the posterior sample tensors.
        # _build_canonical_layouts handles both directly-sampled parameters
        # (from param_specs) and derived keys like "r" and "p" that are
        # registered via numpyro.deterministic (e.g. in non-canonical
        # parameterizations).  has_sample_dim=True because posterior sample
        # tensors carry a leading draw axis.
        from ..sampling import _build_canonical_layouts

        param_layouts = _build_canonical_layouts(
            self.posterior_samples,
            self.model_config,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            n_components=self.n_components,
            has_sample_dim=True,
        )

        from ..de._empirical import sample_composition

        simplex = sample_composition(
            r_samples=r_samples,
            p_samples=p_samples,
            component=component,
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=rng_key,
            batch_size=batch_size,
            param_layouts=param_layouts,
        )

        if store_samples:
            self.compositional_samples = simplex

        return simplex
