"""Empirical DE methods for sample-based results objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Union

import jax.numpy as jnp

from ._results_base_mixin import _format_tau_label, _normalize_tau
from ._set_level import (
    empirical_test_gene_set,
    empirical_test_multiple_gene_sets,
    empirical_test_pathway_perturbation,
)


class EmpiricalResultsMixin:
    """Monte Carlo DE operations for empirical results classes."""

    # Empirical results can export CLR and biological metric families.
    _DATAFRAME_METRIC_ORDER = (
        "clr",
        "bio_lfc",
        "bio_lvr",
        "bio_kl",
        "bio_aux",
    )
    _SUPPORTED_DATAFRAME_METRICS = frozenset(
        {"clr", "bio_lfc", "bio_lvr", "bio_kl", "bio_aux"}
    )

    @property
    def D(self) -> int:
        """Number of genes (CLR dimensionality)."""
        return self.delta_samples.shape[1]

    @property
    def has_biological(self) -> bool:
        """Whether biological-level DE can be computed."""
        return (
            self.r_samples_A is not None
            and self.r_samples_B is not None
            and self.p_samples_A is not None
            and self.p_samples_B is not None
        )

    def biological_level(
        self,
        tau_lfc: float | Sequence[float] = 0.0,
        tau_var: float | Sequence[float] = 0.0,
        tau_kl: float | Sequence[float] = 0.0,
        metric_families: tuple[str, ...] | None = None,
    ) -> dict:
        """Compute biological-level DE statistics from NB parameters.

        Parameters
        ----------
        tau_lfc : float or sequence of float, default=0.0
            Practical threshold(s) for biological log-fold change.
        tau_var : float or sequence of float, default=0.0
            Practical threshold(s) for log-variance ratio.
        tau_kl : float or sequence of float, default=0.0
            Practical threshold(s) for Jeffreys divergence.
        metric_families : tuple of {'bio_lfc', 'bio_lvr', 'bio_kl', 'bio_aux'}, optional
            Biological families to compute. ``None`` computes all families for
            backward compatibility.

        Returns
        -------
        dict
            Biological DE statistics.
        """
        if not self.has_biological:
            raise RuntimeError(
                "Biological-level DE requires stored NB parameter samples "
                "(r_samples_A/B, p_samples_A/B).  Re-run compare() with "
                "compute_biological=True."
            )

        # Include requested families in cache identity to avoid serving stale
        # partial results when callers switch between subsets and full outputs.
        family_key = (
            tuple(sorted(metric_families))
            if metric_families is not None
            else ("bio_lfc", "bio_lvr", "bio_kl", "bio_aux")
        )
        tau_lfc_values = _normalize_tau(tau_lfc)
        tau_var_values = _normalize_tau(tau_var)
        tau_kl_values = _normalize_tau(tau_kl)
        cache_key = (tau_lfc_values, tau_var_values, tau_kl_values, family_key)
        if (
            self._biological_results is None
            or self._cached_bio_taus != cache_key
        ):
            from ._biological import biological_differential_expression

            self._cached_bio_taus = cache_key

            # Pass stored post-sliced layouts so the biological function
            # uses semantic axis info instead of ndim heuristics for
            # p/phi broadcast decisions.
            _p_layout = getattr(self, "p_post_layout", None)
            _phi_layout = getattr(self, "phi_post_layout", None)

            self._biological_results = biological_differential_expression(
                r_samples_A=self.r_samples_A,
                r_samples_B=self.r_samples_B,
                p_samples_A=self.p_samples_A,
                p_samples_B=self.p_samples_B,
                mu_samples_A=self.mu_samples_A,
                mu_samples_B=self.mu_samples_B,
                phi_samples_A=self.phi_samples_A,
                phi_samples_B=self.phi_samples_B,
                tau_lfc=tau_lfc_values,
                tau_var=tau_var_values,
                tau_kl=tau_kl_values,
                gene_names=self.gene_names,
                metric_families=family_key,
                p_layout=_p_layout,
                phi_layout=_phi_layout,
            )
        return self._biological_results

    def gene_level(
        self,
        tau: float | Sequence[float] = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level DE via empirical Monte Carlo counting.

        Parameters
        ----------
        tau : float or sequence of float, default=0.0
            Practical significance threshold(s).
        coordinate : str, default='clr'
            Coordinate system. Only ``'clr'`` is supported.

        Returns
        -------
        dict
            Gene-level DE summary dictionary.
        """
        from ._empirical import empirical_differential_expression

        # Normalize thresholds before caching so equivalent orderings map to
        # the same cache key and deterministic output ordering.
        tau_values = _normalize_tau(tau)
        self._cached_tau = tau_values
        self._gene_results = empirical_differential_expression(
            self.delta_samples,
            tau=tau_values,
            gene_names=self.gene_names,
        )
        return self._gene_results

    def test_contrast(
        self,
        contrast: jnp.ndarray,
        tau: float = 0.0,
    ) -> dict:
        """Test a linear contrast from empirical posterior samples.

        Parameters
        ----------
        contrast : jnp.ndarray
            Contrast vector in CLR space.
        tau : float, default=0.0
            Practical significance threshold.

        Returns
        -------
        dict
            Contrast-level posterior summary statistics.
        """
        contrast_samples = self.delta_samples @ contrast

        contrast_mean = float(jnp.mean(contrast_samples))
        contrast_sd = float(jnp.std(contrast_samples, ddof=1))

        prob_positive = float(jnp.mean(contrast_samples > 0))
        prob_up = float(jnp.mean(contrast_samples > tau))
        prob_down = float(jnp.mean(contrast_samples < -tau))
        prob_effect = prob_up + prob_down

        lfsr = min(prob_positive, 1.0 - prob_positive)
        lfsr_tau = 1.0 - max(prob_up, prob_down)

        return {
            "contrast_mean": contrast_mean,
            "contrast_sd": contrast_sd,
            "prob_positive": prob_positive,
            "prob_effect": prob_effect,
            "lfsr": lfsr,
            "lfsr_tau": lfsr_tau,
        }

    def _resolve_delta_for_gene_set(self, respect_mask: bool) -> jnp.ndarray:
        """Resolve the delta matrix used by empirical pathway tests.

        Parameters
        ----------
        respect_mask : bool
            If ``True``, use the current masked ``delta_samples``.
            If ``False``, recompute unmasked deltas from stored simplex.

        Returns
        -------
        jnp.ndarray
            Matrix of CLR differences for pathway testing.
        """
        # Pathway / balance tooling lives in the CLR sum-to-zero subspace
        # (the full coordinate vector must sum to zero).  A subset reference
        # (IQLR / curated) centers only its reference genes, so its
        # coordinates are not valid ILR balances — reject rather than return
        # silently-wrong enrichment.  See sec-diffexp-iqlr in the paper.
        _ref = getattr(self, "_reference", "clr")
        if not (isinstance(_ref, str) and _ref == "clr"):
            raise ValueError(
                "Pathway / gene-set tests require the CLR reference (the full "
                "coordinate vector must sum to zero), but this result uses "
                f"reference={_ref!r}. Rebuild the comparison with "
                "reference='clr' for pathway / balance analysis."
            )

        if respect_mask or self._gene_mask is None:
            return self.delta_samples

        if not self.has_simplex:
            raise ValueError(
                "respect_mask=False requires stored simplex samples, but "
                "they are not available."
            )

        from ._empirical import compute_delta_from_simplex

        return compute_delta_from_simplex(
            self.simplex_A, self.simplex_B, gene_mask=None
        )

    def test_gene_set(
        self,
        gene_set_indices: jnp.ndarray,
        tau: float = 0.0,
        respect_mask: bool = True,
    ) -> dict:
        """Test pathway enrichment via empirical ILR balances.

        Parameters
        ----------
        gene_set_indices : jnp.ndarray
            Integer gene indices for one pathway.
        tau : float, default=0.0
            Practical significance threshold.
        respect_mask : bool, default=True
            Whether indices refer to the current masked gene space.

        Returns
        -------
        dict
            Pathway balance posterior statistics.
        """
        delta = self._resolve_delta_for_gene_set(respect_mask)
        return empirical_test_gene_set(delta, gene_set_indices, tau=tau)

    def test_pathway_perturbation(
        self,
        gene_set_indices: jnp.ndarray,
        n_permutations: int = 999,
        key=None,
        respect_mask: bool = True,
    ) -> dict:
        """Test within-pathway compositional perturbation.

        Parameters
        ----------
        gene_set_indices : jnp.ndarray
            Integer gene indices for one pathway.
        n_permutations : int, default=999
            Number of permutations for calibration.
        key : jax.random.PRNGKey, optional
            Random key used by the permutation test.
        respect_mask : bool, default=True
            Whether indices refer to masked or full gene space.

        Returns
        -------
        dict
            Perturbation-test statistics and p-value.
        """
        delta = self._resolve_delta_for_gene_set(respect_mask)
        return empirical_test_pathway_perturbation(
            delta,
            gene_set_indices,
            n_permutations=n_permutations,
            key=key,
        )

    def test_multiple_gene_sets(
        self,
        gene_sets,
        tau: float = 0.0,
        target_pefp: float = 0.05,
        respect_mask: bool = True,
    ) -> dict:
        """Run empirical pathway testing with PEFP control.

        Parameters
        ----------
        gene_sets : sequence of jnp.ndarray
            Pathway index arrays.
        tau : float, default=0.0
            Practical significance threshold.
        target_pefp : float, default=0.05
            Target PEFP level.
        respect_mask : bool, default=True
            Whether indices refer to masked or full gene space.

        Returns
        -------
        dict
            Batch pathway results with significance calls.
        """
        delta = self._resolve_delta_for_gene_set(respect_mask)
        return empirical_test_multiple_gene_sets(
            delta, gene_sets, tau=tau, target_pefp=target_pefp
        )

    @property
    def has_simplex(self) -> bool:
        """Whether full simplex samples are available for re-masking."""
        return self.simplex_A is not None and self.simplex_B is not None

    def set_gene_mask(self, mask: jnp.ndarray) -> "ScribeEmpiricalDEResults":
        """Apply a new gene mask and recompute CLR differences.

        Parameters
        ----------
        mask : jnp.ndarray
            Boolean mask over full gene space.

        Returns
        -------
        ScribeEmpiricalDEResults
            Returns ``self`` for method chaining.
        """
        import numpy as np
        from ._empirical import _resolve_reference, compute_delta_from_simplex

        if not self.has_simplex:
            raise ValueError(
                "Cannot change gene mask: simplex samples were not stored. "
                "Re-run compare() — simplex storage is the default."
            )

        mask_arr = np.asarray(mask, dtype=bool).ravel()
        D_full = self.simplex_A.shape[1]

        if mask_arr.shape[0] != D_full:
            raise ValueError(
                f"mask length ({mask_arr.shape[0]}) does not match the full "
                f"gene dimension ({D_full})."
            )

        # Preserve the reference frame across the re-mask by re-resolving the
        # stored spec against the new keep-mask (a gene-name list / "iqlr" /
        # full-gene boolean re-resolves cleanly; an aggregated-length boolean
        # spec raises a clear length error here).
        _resolved_ref = _resolve_reference(
            getattr(self, "_reference", "clr"), self._all_gene_names, mask_arr
        )
        self.delta_samples = compute_delta_from_simplex(
            self.simplex_A,
            self.simplex_B,
            gene_mask=jnp.asarray(mask_arr),
            reference=_resolved_ref,
        )
        self.n_samples = self.delta_samples.shape[0]

        if self._all_gene_names is not None:
            self.gene_names = [
                n for n, m in zip(self._all_gene_names, mask_arr) if m
            ]
        else:
            D_kept = int(mask_arr.sum())
            self.gene_names = [f"gene_{i}" for i in range(D_kept)]

        self._gene_mask = jnp.asarray(mask_arr)

        # Clear every cache derived from the old mask.
        self._gene_results = None
        self._cached_tau = None
        self._biological_results = None
        self._cached_bio_taus = None

        return self

    def _drop_other_from_mask(self, mask):
        """Force the gene_coverage ``"_other"`` pseudo-gene out of a keep-mask.

        ``scribe.fit(..., gene_coverage<1.0)`` pools the un-modelled tail into a
        trailing ``"_other"`` category to preserve compositional closure. It is
        the compositional anchor, never a reportable gene, so it must always
        join the DE "other" pool rather than be kept as an explicit gene —
        otherwise, because its pooled mass is typically large, a coverage /
        expression mask keeps it and it leaks into the reported DE table. This
        zeros the ``"_other"`` position so it is consistently pooled (matching
        the no-mask behaviour, where the trailing category is the reference).
        No-op when there is no ``"_other"`` category; idempotent.
        """
        import numpy as np

        mask = np.asarray(mask, dtype=bool)
        names = self._all_gene_names
        if names is None:
            names = self.gene_names
        if names is None:
            return mask
        names = np.asarray([str(n) for n in names])
        if names.shape[0] != mask.shape[0]:
            return mask
        mask = mask.copy()
        # Matches scribe.core.gene_coverage's other_name default.
        mask[names == "_other"] = False
        return mask

    def expression_mask(self, min_expression: float):
        """Build (but do not apply) a min-expression mask from MAP means.

        A gene is retained if either condition has MAP mean expression at least
        ``min_expression``. Unlike :meth:`set_expression_threshold`, this only
        reads ``mu_map_A`` / ``mu_map_B`` and never touches the simplex samples,
        so it also works on results that do not store the simplex (e.g.
        :func:`scribe.compare_groups`). Pass the returned mask up front via
        ``gene_mask=`` to pool the rest into "other" before CLR.

        Parameters
        ----------
        min_expression : float
            Minimum MAP mean expression (in either condition) to retain a gene.

        Returns
        -------
        numpy.ndarray
            Boolean mask over genes.
        """
        import numpy as np

        if self.mu_map_A is None or self.mu_map_B is None:
            raise ValueError(
                "Cannot build expression mask: MAP mean expression "
                "(mu_map_A / mu_map_B) was not stored in the results object."
            )
        mu_A = np.asarray(self.mu_map_A)
        mu_B = np.asarray(self.mu_map_B)
        return self._drop_other_from_mask(
            (mu_A >= min_expression) | (mu_B >= min_expression)
        )

    def composition_coverage_mask(self, coverage: float = 0.95):
        """Build (but do not apply) a cumulative-composition coverage mask.

        Keeps the smallest set of genes whose MAP composition reaches
        ``coverage`` of the total mass in *either* condition (the union of the
        two per-condition coverage sets). Reads only ``mu_map_A`` /
        ``mu_map_B`` — no simplex required — so it works on
        :func:`scribe.compare_groups` results; pass the mask up front via
        ``gene_mask=``.

        Parameters
        ----------
        coverage : float, default=0.95
            Cumulative compositional coverage target in ``(0, 1]``.

        Returns
        -------
        numpy.ndarray
            Boolean mask over genes.
        """
        import numpy as np
        from ._empirical import _coverage_mask_from_mu

        if self.mu_map_A is None or self.mu_map_B is None:
            raise ValueError(
                "Cannot build coverage mask: MAP mean expression "
                "(mu_map_A / mu_map_B) was not stored in the results object."
            )
        mu_A = np.asarray(self.mu_map_A)
        mu_B = np.asarray(self.mu_map_B)
        mask = _coverage_mask_from_mu(mu_A, coverage) | _coverage_mask_from_mu(
            mu_B, coverage
        )
        return self._drop_other_from_mask(mask)

    def set_expression_threshold(
        self, min_expression: float
    ) -> "ScribeEmpiricalDEResults":
        """Construct and apply a mask from stored MAP mean expression.

        Parameters
        ----------
        min_expression : float
            A gene is retained if either condition has MAP mean expression
            at least this value.

        Returns
        -------
        ScribeEmpiricalDEResults
            Returns ``self`` for method chaining.
        """
        self.set_gene_mask(self.expression_mask(min_expression))
        return self

    def set_composition_coverage(
        self, coverage: float = 0.95
    ) -> "ScribeEmpiricalDEResults":
        """Construct and apply a mask from cumulative MAP composition.

        This method derives composition vectors from the stored MAP mean
        expression (`mu_map_A` and `mu_map_B`) and keeps the smallest set
        of genes that reaches the requested cumulative proportion in each
        condition.  The final mask is the union of both per-condition
        masks, preserving genes that are prominent in either condition.

        Parameters
        ----------
        coverage : float, default=0.95
            Cumulative compositional coverage target in ``(0, 1]``.

        Returns
        -------
        ScribeEmpiricalDEResults
            Returns ``self`` for method chaining.

        Raises
        ------
        ValueError
            If MAP mean expression vectors are unavailable or if coverage
            is outside ``(0, 1]``.
        """
        self.set_gene_mask(self.composition_coverage_mask(coverage))
        return self

    def clear_mask(self) -> "ScribeEmpiricalDEResults":
        """Clear active mask and restore full-gene CLR differences.

        Returns
        -------
        ScribeEmpiricalDEResults
            Returns ``self`` for method chaining.
        """
        from ._empirical import _resolve_reference, compute_delta_from_simplex

        if not self.has_simplex:
            raise ValueError(
                "Cannot clear mask: simplex samples were not stored."
            )

        # Preserve the reference frame over the full (unmasked) gene set.
        _resolved_ref = _resolve_reference(
            getattr(self, "_reference", "clr"), self._all_gene_names, None
        )
        self.delta_samples = compute_delta_from_simplex(
            self.simplex_A,
            self.simplex_B,
            gene_mask=None,
            reference=_resolved_ref,
        )
        self.n_samples = self.delta_samples.shape[0]

        if self._all_gene_names is not None:
            self.gene_names = list(self._all_gene_names)
        else:
            D = self.delta_samples.shape[1]
            self.gene_names = [f"gene_{i}" for i in range(D)]

        self._gene_mask = None

        # Clear every cache derived from the masked view.
        self._gene_results = None
        self._cached_tau = None
        self._biological_results = None
        self._cached_bio_taus = None

        return self

    def set_reference(
        self, reference: Union[str, jnp.ndarray, List[str]]
    ) -> "ScribeEmpiricalDEResults":
        """Recompute the log-ratio differences under a new reference frame.

        Recomputes ``delta_samples`` from the stored simplex samples using
        ``reference`` (``"clr"`` | ``"iqlr"`` | gene-name list | boolean
        mask), preserving the current gene mask.  The reference is also
        stored so subsequent mask changes keep it.

        Requires stored simplex samples, so it works on leaf-vs-leaf
        :func:`compare` results but **not** on grouped
        :func:`compare_groups` results (which average within-pair deltas
        and keep no per-pair simplex).  For those, pass ``reference=`` to
        ``compare_groups`` directly.

        Parameters
        ----------
        reference : {"clr", "iqlr"} | list of str | boolean array
            The new reference frame, resolved against the full gene names
            and the active gene mask.

        Returns
        -------
        ScribeEmpiricalDEResults
            Returns ``self`` for method chaining.
        """
        import numpy as np
        from ._empirical import _resolve_reference, compute_delta_from_simplex

        if not self.has_simplex:
            raise ValueError(
                "Cannot set reference: simplex samples were not stored. "
                "compare_groups() results keep no per-pair simplex — pass "
                "reference= to compare_groups() directly instead."
            )

        gm = None if self._gene_mask is None else np.asarray(
            self._gene_mask, dtype=bool
        ).ravel()
        _resolved_ref = _resolve_reference(
            reference, self._all_gene_names, gm
        )
        self.delta_samples = compute_delta_from_simplex(
            self.simplex_A,
            self.simplex_B,
            gene_mask=(None if gm is None else jnp.asarray(gm)),
            reference=_resolved_ref,
        )
        self.n_samples = self.delta_samples.shape[0]
        self._reference = reference

        # Clear every cache derived from the previous reference.
        self._gene_results = None
        self._cached_tau = None
        self._biological_results = None
        self._cached_bio_taus = None

        return self

    def iqlr_reference_mask(self) -> "np.ndarray":
        """Return the IQLR reference set used for the current differences.

        Recomputes the inter-quartile-variance reference set from the
        stored simplex samples under the active gene mask, for inspection
        (e.g. which genes anchor the IQLR reference).  Returns a boolean
        array over the **aggregated** columns: kept genes in mask order,
        plus a trailing ``False`` for the "other" aggregate when a mask is
        active.  Requires stored simplex samples.
        """
        import numpy as np
        from ._empirical import _aggregate_simplex, _clr_transform, _iqlr_reference_mask

        if not self.has_simplex:
            raise ValueError(
                "Cannot compute the IQLR reference: simplex samples were "
                "not stored."
            )
        s_A, s_B = self.simplex_A, self.simplex_B
        has_other = self._gene_mask is not None
        if has_other:
            gm = np.asarray(self._gene_mask, dtype=bool).ravel()
            s_A = _aggregate_simplex(s_A, gm)
            s_B = _aggregate_simplex(s_B, gm)
        clr_pooled = np.concatenate(
            [_clr_transform(s_A), _clr_transform(s_B)], axis=0
        )
        return _iqlr_reference_mask(clr_pooled, exclude_last=has_other)

    def _resolve_dataframe_tau_grids(
        self,
        tau: float | Sequence[float],
        tau_lfc: float | Sequence[float] | None,
        tau_var: float | Sequence[float] | None,
        tau_kl: float | Sequence[float] | None,
    ) -> dict[str, tuple[float, ...]]:
        """Resolve effective tau grids for CLR and biological exports.

        Parameters
        ----------
        tau : float or sequence of float
            Caller-requested CLR practical threshold(s).
        tau_lfc : float, sequence of float, or None
            Optional explicit threshold(s) for biological LFC.
        tau_var : float, sequence of float, or None
            Optional explicit threshold(s) for biological LVR.
        tau_kl : float, sequence of float, or None
            Optional explicit threshold(s) for biological KL.

        Returns
        -------
        dict[str, tuple[float, ...]]
            Effective sorted tau tuples for ``clr``, ``bio_lfc``,
            ``bio_lvr``, and ``bio_kl``.

        Notes
        -----
        Broadcasting is intentionally conservative for backward compatibility:
        biological families inherit the CLR tau grid only when CLR receives
        multiple tau values and the corresponding metric-specific tau was not
        explicitly provided.
        """

        clr_tau_values = _normalize_tau(tau)

        def _resolve_metric_tau(
            metric_tau: float | Sequence[float] | None,
        ) -> tuple[float, ...]:
            if metric_tau is not None:
                return _normalize_tau(metric_tau)
            if len(clr_tau_values) > 1:
                return clr_tau_values
            return (0.0,)

        return {
            "clr": clr_tau_values,
            "bio_lfc": _resolve_metric_tau(tau_lfc),
            "bio_lvr": _resolve_metric_tau(tau_var),
            "bio_kl": _resolve_metric_tau(tau_kl),
        }

    def to_dataframe(
        self,
        tau: float | Sequence[float] = 0.0,
        target_pefp: float | None = None,
        use_lfsr_tau: bool = True,
        target_pefp_lfc: float | None = None,
        use_lfsr_tau_lfc: bool = True,
        target_pefp_lvr: float | None = None,
        use_lfsr_tau_lvr: bool = True,
        target_pefp_kl: float | None = None,
        metrics: str | list[str] | tuple[str, ...] | None = None,
        tau_lfc: float | Sequence[float] | None = None,
        tau_var: float | Sequence[float] | None = None,
        tau_kl: float | Sequence[float] | None = None,
        column_naming: str = "prefixed",
        tau_format: str = "suffix",
    ):
        """Export selected CLR/biological metric families to a DataFrame.

        Parameters
        ----------
        tau : float or sequence of float, default=0.0
            Practical significance threshold(s) for CLR metrics.
        target_pefp : float, optional
            Optional PEFP target for ``is_de`` column.
        use_lfsr_tau : bool, default=True
            Which lfsr variant to use for PEFP thresholding.
        target_pefp_lfc : float, optional
            PEFP target used to add an LFC call column.
        use_lfsr_tau_lfc : bool, default=True
            Whether LFC calls use ``lfsr_tau`` (if ``True``) or ``lfsr``.
        target_pefp_lvr : float, optional
            PEFP target used to add an LVR call column.
        use_lfsr_tau_lvr : bool, default=True
            Whether LVR calls use ``lfsr_tau`` (if ``True``) or ``lfsr``.
        target_pefp_kl : float, optional
            PEFP target used to add a KL call column. KL calls use
            ``lfer = 1 - prob_effect`` as the error score.
        metrics : {'clr', 'bio_lfc', 'bio_lvr', 'bio_kl', 'bio_aux', 'all'}
            or iterable, optional
            Metric families to include. ``None`` defaults to ``'clr'`` to
            preserve existing behavior.

            - ``'clr'``: compositional CLR gene-level DE summaries
              (``delta_*``, ``lfsr*``, and effect probabilities).
            - ``'bio_lfc'``: biological mean-shift summaries based on
              log-fold-change (``lfc_*`` columns).
            - ``'bio_lvr'``: biological variance-shift summaries based on
              log-variance ratio (``lvr_*`` columns).
            - ``'bio_kl'``: biological distribution-shift summaries from
              Jeffreys divergence (``kl_*`` columns).
            - ``'bio_aux'``: auxiliary biological context columns
              (``mu_*``, ``var_*``, ``max_bio_expr``).
            - ``'all'``: alias that expands to all families supported by
              empirical/shrinkage results.
        tau_lfc : float, sequence of float, or None, default=None
            Practical threshold(s) for biological LFC. If ``None`` and ``tau``
            contains multiple values, the biological LFC block inherits the CLR
            tau grid. Otherwise it defaults to ``0.0``.
        tau_var : float, sequence of float, or None, default=None
            Practical threshold(s) for biological LVR with the same broadcast
            rule as ``tau_lfc``.
        tau_kl : float, sequence of float, or None, default=None
            Practical threshold(s) for biological KL with the same broadcast
            rule as ``tau_lfc``.
        column_naming : {'prefixed', 'legacy'}, default='prefixed'
            Column naming convention. ``'prefixed'`` produces explicit family
            namespaces (for example ``clr_*``, ``bio_lfc_*``). ``'legacy'``
            preserves historical un-prefixed biological names and CLR names.
        tau_format : {'suffix', 'multiindex'}, default='suffix'
            Layout used for multi-tau exports across CLR and biological
            tau-dependent columns.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per gene and selected metric families.
        """
        import numpy as np
        import pandas as pd

        metric_families = self._resolve_dataframe_metrics(metrics)
        include_clr = "clr" in metric_families
        tau_grids = self._resolve_dataframe_tau_grids(
            tau=tau,
            tau_lfc=tau_lfc,
            tau_var=tau_var,
            tau_kl=tau_kl,
        )

        # Keep PEFP semantics explicit: thresholding uses CLR lfsr columns.
        if target_pefp is not None and not include_clr:
            raise ValueError(
                "target_pefp requires metrics to include 'clr' because "
                "'is_de' is defined from CLR lfsr values."
            )
        if target_pefp_lfc is not None and "bio_lfc" not in metric_families:
            raise ValueError(
                "target_pefp_lfc requires metrics to include 'bio_lfc'."
            )
        if target_pefp_lvr is not None and "bio_lvr" not in metric_families:
            raise ValueError(
                "target_pefp_lvr requires metrics to include 'bio_lvr'."
            )
        if target_pefp_kl is not None and "bio_kl" not in metric_families:
            raise ValueError(
                "target_pefp_kl requires metrics to include 'bio_kl'."
            )

        if include_clr:
            df = super().to_dataframe(
                tau=tau_grids["clr"],
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
                metrics="clr",
                column_naming=column_naming,
                tau_format=tau_format,
            )
        else:
            # Build a valid gene index when users request only biological blocks.
            df = pd.DataFrame({"gene": list(self.gene_names)})

        # Align stored expression vectors to the currently active mask.
        if (
            include_clr
            and self.mu_map_A is not None
            and self.mu_map_B is not None
        ):
            mu_A = np.asarray(self.mu_map_A)
            mu_B = np.asarray(self.mu_map_B)
            if self._gene_mask is not None:
                mask = np.asarray(self._gene_mask, dtype=bool)
                mu_A = mu_A[mask]
                mu_B = mu_B[mask]
            if column_naming == "prefixed":
                df["clr_mean_expression_A"] = mu_A
                df["clr_mean_expression_B"] = mu_B
            else:
                df["mean_expression_A"] = mu_A
                df["mean_expression_B"] = mu_B

        bio_families = {
            "bio_lfc",
            "bio_lvr",
            "bio_kl",
            "bio_aux",
        }
        include_any_bio = any(family in metric_families for family in bio_families)
        if include_any_bio:
            requested_bio_families = tuple(
                family
                for family in self._DATAFRAME_METRIC_ORDER
                if family in bio_families and family in metric_families
            )
            bio = self.biological_level(
                tau_lfc=tau_grids["bio_lfc"],
                tau_var=tau_grids["bio_lvr"],
                tau_kl=tau_grids["bio_kl"],
                metric_families=requested_bio_families,
            )
            mask = None
            if self._gene_mask is not None:
                mask = np.asarray(self._gene_mask, dtype=bool)

            def _bio_values(key: str) -> np.ndarray:
                """Return biological vectors aligned to the active gene mask."""
                values = np.asarray(bio[key])
                if mask is not None and values.shape[0] == mask.shape[0]:
                    return values[mask]
                return values

            def _bio_column_name(key: str) -> str:
                """Map internal biological keys to exported column names."""
                if column_naming == "legacy":
                    return key
                if key.startswith("lfc_"):
                    return f"bio_lfc_{key.removeprefix('lfc_')}"
                if key.startswith("lvr_"):
                    return f"bio_lvr_{key.removeprefix('lvr_')}"
                if key.startswith("kl_"):
                    return f"bio_kl_{key.removeprefix('kl_')}"
                return f"bio_{key}"

            def _call_column_name(metric: str) -> str:
                """Name per-metric DE call columns by naming convention."""
                if column_naming == "legacy":
                    legacy_map = {
                        "bio_lfc": "lfc_is_de",
                        "bio_lvr": "lvr_is_de",
                        "bio_kl": "kl_is_de",
                    }
                    return legacy_map[metric]
                return f"{metric}_is_de"

            def _using_multiindex_columns() -> bool:
                return isinstance(df.columns, pd.MultiIndex)

            def _ensure_multiindex_columns() -> None:
                """Convert flat columns to (metric, tau) MultiIndex in-place."""
                nonlocal df
                if _using_multiindex_columns():
                    return
                tuples = [(str(col), "") for col in df.columns]
                df.columns = pd.MultiIndex.from_tuples(
                    tuples, names=("metric", "tau")
                )

            def _assign_scalar_column(name: str, values: np.ndarray) -> None:
                """Assign tau-independent values under current naming mode."""
                if tau_format == "multiindex" and _using_multiindex_columns():
                    df[(name, "")] = np.asarray(values)
                else:
                    df[name] = np.asarray(values)

            def _assign_tau_series(
                base_name: str,
                values: np.ndarray,
                tau_values: tuple[float, ...],
            ) -> None:
                """Assign tau-dependent arrays in suffix or MultiIndex layout."""
                arr = np.asarray(values)
                if arr.ndim == 1:
                    _assign_scalar_column(base_name, arr)
                    return
                if arr.ndim != 2:
                    raise ValueError(
                        f"Expected 1D or 2D tau-dependent values for {base_name}, "
                        f"got shape {arr.shape}."
                    )
                if arr.shape[1] != len(tau_values):
                    raise ValueError(
                        f"Tau-dependent value shape {arr.shape} does not match "
                        f"tau grid length {len(tau_values)} for {base_name}."
                    )
                if len(tau_values) == 1:
                    _assign_scalar_column(base_name, arr[:, 0])
                    return
                if tau_format == "multiindex":
                    _ensure_multiindex_columns()
                    for idx, tau_value in enumerate(tau_values):
                        tau_label = _format_tau_label(tau_value)
                        df[(base_name, tau_label)] = arr[:, idx]
                else:
                    for idx, tau_value in enumerate(tau_values):
                        tau_label = _format_tau_label(tau_value)
                        df[f"{base_name}_tau{tau_label}"] = arr[:, idx]

            def _assign_metric_specific_calls(
                *,
                score_values: np.ndarray,
                tau_values: tuple[float, ...],
                target_pefp_metric: float,
                call_base_name: str,
            ) -> None:
                """Create PEFP-controlled DE calls for scalar or tau-grid scores."""
                score_arr = np.asarray(score_values)
                if score_arr.ndim == 1 or len(tau_values) == 1:
                    score_vec = (
                        score_arr
                        if score_arr.ndim == 1
                        else np.asarray(score_arr)[:, 0]
                    )
                    is_de, _ = self._compute_is_de_mask_from_scores(
                        score_vec,
                        target_pefp=target_pefp_metric,
                    )
                    _assign_scalar_column(call_base_name, np.asarray(is_de, dtype=bool))
                    return
                if score_arr.ndim != 2:
                    raise ValueError(
                        f"Expected 2D score matrix for multi-tau calls, got "
                        f"shape {score_arr.shape}."
                    )
                if score_arr.shape[1] != len(tau_values):
                    raise ValueError(
                        f"Score matrix shape {score_arr.shape} does not match "
                        f"tau grid length {len(tau_values)}."
                    )
                if tau_format == "multiindex":
                    _ensure_multiindex_columns()
                    for idx, tau_value in enumerate(tau_values):
                        tau_label = _format_tau_label(tau_value)
                        is_de, _ = self._compute_is_de_mask_from_scores(
                            score_arr[:, idx],
                            target_pefp=target_pefp_metric,
                        )
                        df[(call_base_name, tau_label)] = np.asarray(
                            is_de, dtype=bool
                        )
                else:
                    for idx, tau_value in enumerate(tau_values):
                        tau_label = _format_tau_label(tau_value)
                        is_de, _ = self._compute_is_de_mask_from_scores(
                            score_arr[:, idx],
                            target_pefp=target_pefp_metric,
                        )
                        df[f"{call_base_name}_tau{tau_label}"] = np.asarray(
                            is_de, dtype=bool
                        )

            # Keep each biological block grouped so callers can request subsets.
            if "bio_lfc" in metric_families:
                lfc_tau_values = tuple(float(v) for v in bio["lfc_tau_values"])
                for key in (
                    "lfc_mean",
                    "lfc_sd",
                    "lfc_prob_positive",
                    "lfc_lfsr",
                ):
                    _assign_scalar_column(_bio_column_name(key), _bio_values(key))
                for key in (
                    "lfc_prob_up",
                    "lfc_prob_down",
                    "lfc_prob_effect",
                    "lfc_lfsr_tau",
                ):
                    _assign_tau_series(
                        _bio_column_name(key),
                        _bio_values(key),
                        lfc_tau_values,
                    )
                if target_pefp_lfc is not None:
                    lfc_score_key = (
                        "lfc_lfsr_tau" if use_lfsr_tau_lfc else "lfc_lfsr"
                    )
                    _assign_metric_specific_calls(
                        score_values=_bio_values(lfc_score_key),
                        tau_values=lfc_tau_values,
                        target_pefp_metric=target_pefp_lfc,
                        call_base_name=_call_column_name("bio_lfc"),
                    )

            if "bio_lvr" in metric_families:
                lvr_tau_values = tuple(float(v) for v in bio["lvr_tau_values"])
                for key in (
                    "lvr_mean",
                    "lvr_sd",
                    "lvr_prob_positive",
                    "lvr_lfsr",
                ):
                    _assign_scalar_column(_bio_column_name(key), _bio_values(key))
                for key in (
                    "lvr_prob_up",
                    "lvr_prob_down",
                    "lvr_prob_effect",
                    "lvr_lfsr_tau",
                ):
                    _assign_tau_series(
                        _bio_column_name(key),
                        _bio_values(key),
                        lvr_tau_values,
                    )
                if target_pefp_lvr is not None:
                    lvr_score_key = (
                        "lvr_lfsr_tau" if use_lfsr_tau_lvr else "lvr_lfsr"
                    )
                    _assign_metric_specific_calls(
                        score_values=_bio_values(lvr_score_key),
                        tau_values=lvr_tau_values,
                        target_pefp_metric=target_pefp_lvr,
                        call_base_name=_call_column_name("bio_lvr"),
                    )

            if "bio_kl" in metric_families:
                kl_tau_values = tuple(float(v) for v in bio["kl_tau_values"])
                for key in ("kl_mean", "kl_sd"):
                    _assign_scalar_column(_bio_column_name(key), _bio_values(key))
                _assign_tau_series(
                    _bio_column_name("kl_prob_effect"),
                    _bio_values("kl_prob_effect"),
                    kl_tau_values,
                )
                if target_pefp_kl is not None:
                    # KL is non-negative and non-directional, so we map
                    # ``prob_effect`` to an error score via lfer.
                    kl_lfer = 1.0 - _bio_values("kl_prob_effect")
                    kl_lfer_name = (
                        "kl_lfer" if column_naming == "legacy" else "bio_kl_lfer"
                    )
                    _assign_tau_series(
                        kl_lfer_name,
                        kl_lfer,
                        kl_tau_values,
                    )
                    _assign_metric_specific_calls(
                        score_values=kl_lfer,
                        tau_values=kl_tau_values,
                        target_pefp_metric=target_pefp_kl,
                        call_base_name=_call_column_name("bio_kl"),
                    )

            if "bio_aux" in metric_families:
                for key in (
                    "mu_A_mean",
                    "mu_B_mean",
                    "var_A_mean",
                    "var_B_mean",
                    "max_bio_expr",
                ):
                    _assign_scalar_column(_bio_column_name(key), _bio_values(key))

        return df

    def shrink(
        self,
        sigma_grid: jnp.ndarray | None = None,
        shrinkage_max_iter: int = 200,
        shrinkage_tol: float = 1e-8,
    ) -> "ScribeShrinkageDEResults":
        """Wrap this empirical result object with shrinkage behavior.

        Parameters
        ----------
        sigma_grid : jnp.ndarray, optional
            Optional prior scale grid for the shrinkage mixture.
        shrinkage_max_iter : int, default=200
            Maximum EM iterations.
        shrinkage_tol : float, default=1e-8
            EM convergence tolerance.

        Returns
        -------
        ScribeShrinkageDEResults
            Shrinkage object sharing underlying sample arrays.
        """
        from .results import ScribeShrinkageDEResults

        obj = ScribeShrinkageDEResults(
            delta_samples=self.delta_samples,
            gene_names=self.gene_names,
            label_A=self.label_A,
            label_B=self.label_B,
            r_samples_A=self.r_samples_A,
            r_samples_B=self.r_samples_B,
            p_samples_A=self.p_samples_A,
            p_samples_B=self.p_samples_B,
            mu_samples_A=self.mu_samples_A,
            mu_samples_B=self.mu_samples_B,
            phi_samples_A=self.phi_samples_A,
            phi_samples_B=self.phi_samples_B,
            simplex_A=self.simplex_A,
            simplex_B=self.simplex_B,
            mu_map_A=self.mu_map_A,
            mu_map_B=self.mu_map_B,
            sigma_grid=sigma_grid,
            shrinkage_max_iter=shrinkage_max_iter,
            shrinkage_tol=shrinkage_tol,
        )

        obj._gene_mask = self._gene_mask
        obj._all_gene_names = self._all_gene_names
        obj._reference = getattr(self, "_reference", "clr")

        return obj

    def __repr__(self) -> str:
        """Return a concise representation of empirical DE comparison."""
        bio_str = ", biological=True" if self.has_biological else ""
        return (
            f"ScribeEmpiricalDEResults("
            f"D={self.D}, "
            f"n_samples={self.n_samples}{bio_str}, "
            f"labels='{self.label_A}' vs '{self.label_B}')"
        )
