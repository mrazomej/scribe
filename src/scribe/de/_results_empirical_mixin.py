"""Empirical DE methods for sample-based results objects."""

from __future__ import annotations

import jax.numpy as jnp

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
        tau_lfc: float = 0.0,
        tau_var: float = 0.0,
        tau_kl: float = 0.0,
        metric_families: tuple[str, ...] | None = None,
    ) -> dict:
        """Compute biological-level DE statistics from NB parameters.

        Parameters
        ----------
        tau_lfc : float, default=0.0
            Practical threshold for biological log-fold change.
        tau_var : float, default=0.0
            Practical threshold for log-variance ratio.
        tau_kl : float, default=0.0
            Practical threshold for Jeffreys divergence.
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
        cache_key = (tau_lfc, tau_var, tau_kl, family_key)
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
                tau_lfc=tau_lfc,
                tau_var=tau_var,
                tau_kl=tau_kl,
                gene_names=self.gene_names,
                metric_families=family_key,
                p_layout=_p_layout,
                phi_layout=_phi_layout,
            )
        return self._biological_results

    def gene_level(
        self,
        tau: float = 0.0,
        coordinate: str = "clr",
    ) -> dict:
        """Compute gene-level DE via empirical Monte Carlo counting.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold.
        coordinate : str, default='clr'
            Coordinate system. Only ``'clr'`` is supported.

        Returns
        -------
        dict
            Gene-level DE summary dictionary.
        """
        from ._empirical import empirical_differential_expression

        self._cached_tau = tau
        self._gene_results = empirical_differential_expression(
            self.delta_samples,
            tau=tau,
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
        from ._empirical import compute_delta_from_simplex

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

        self.delta_samples = compute_delta_from_simplex(
            self.simplex_A, self.simplex_B, gene_mask=jnp.asarray(mask_arr)
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
        import numpy as np

        if self.mu_map_A is None or self.mu_map_B is None:
            raise ValueError(
                "Cannot set expression threshold: MAP mean expression "
                "(mu_map_A / mu_map_B) was not stored in the results object."
            )

        mu_A = np.asarray(self.mu_map_A)
        mu_B = np.asarray(self.mu_map_B)
        mask = (mu_A >= min_expression) | (mu_B >= min_expression)
        self.set_gene_mask(mask)
        return self

    def clear_mask(self) -> "ScribeEmpiricalDEResults":
        """Clear active mask and restore full-gene CLR differences.

        Returns
        -------
        ScribeEmpiricalDEResults
            Returns ``self`` for method chaining.
        """
        from ._empirical import compute_delta_from_simplex

        if not self.has_simplex:
            raise ValueError(
                "Cannot clear mask: simplex samples were not stored."
            )

        self.delta_samples = compute_delta_from_simplex(
            self.simplex_A, self.simplex_B, gene_mask=None
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

    def to_dataframe(
        self,
        tau: float = 0.0,
        target_pefp: float | None = None,
        use_lfsr_tau: bool = True,
        target_pefp_lfc: float | None = None,
        use_lfsr_tau_lfc: bool = True,
        target_pefp_lvr: float | None = None,
        use_lfsr_tau_lvr: bool = True,
        target_pefp_kl: float | None = None,
        metrics: str | list[str] | tuple[str, ...] | None = None,
        tau_lfc: float = 0.0,
        tau_var: float = 0.0,
        tau_kl: float = 0.0,
        column_naming: str = "prefixed",
    ):
        """Export selected CLR/biological metric families to a DataFrame.

        Parameters
        ----------
        tau : float, default=0.0
            Practical significance threshold.
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
        tau_lfc : float, default=0.0
            Practical threshold passed to :meth:`biological_level` for LFC.
        tau_var : float, default=0.0
            Practical threshold passed to :meth:`biological_level` for LVR.
        tau_kl : float, default=0.0
            Practical threshold passed to :meth:`biological_level` for KL.
        column_naming : {'prefixed', 'legacy'}, default='prefixed'
            Column naming convention. ``'prefixed'`` produces explicit family
            namespaces (for example ``clr_*``, ``bio_lfc_*``). ``'legacy'``
            preserves historical un-prefixed biological names and CLR names.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per gene and selected metric families.
        """
        import numpy as np
        import pandas as pd

        metric_families = self._resolve_dataframe_metrics(metrics)
        include_clr = "clr" in metric_families

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
                tau=tau,
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
                metrics="clr",
                column_naming=column_naming,
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
        if any(family in metric_families for family in bio_families):
            requested_bio_families = tuple(
                family
                for family in self._DATAFRAME_METRIC_ORDER
                if family in bio_families and family in metric_families
            )
            bio = self.biological_level(
                tau_lfc=tau_lfc,
                tau_var=tau_var,
                tau_kl=tau_kl,
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

            # Keep each biological block grouped so callers can request subsets.
            if "bio_lfc" in metric_families:
                for key in (
                    "lfc_mean",
                    "lfc_sd",
                    "lfc_prob_positive",
                    "lfc_lfsr",
                    "lfc_prob_up",
                    "lfc_prob_down",
                    "lfc_prob_effect",
                    "lfc_lfsr_tau",
                ):
                    df[_bio_column_name(key)] = _bio_values(key)
                if target_pefp_lfc is not None:
                    lfc_score_key = (
                        "lfc_lfsr_tau" if use_lfsr_tau_lfc else "lfc_lfsr"
                    )
                    lfc_is_de, _ = self._compute_is_de_mask_from_scores(
                        _bio_values(lfc_score_key),
                        target_pefp=target_pefp_lfc,
                    )
                    df[_call_column_name("bio_lfc")] = np.asarray(
                        lfc_is_de, dtype=bool
                    )

            if "bio_lvr" in metric_families:
                for key in (
                    "lvr_mean",
                    "lvr_sd",
                    "lvr_prob_positive",
                    "lvr_lfsr",
                    "lvr_prob_up",
                    "lvr_prob_down",
                    "lvr_prob_effect",
                    "lvr_lfsr_tau",
                ):
                    df[_bio_column_name(key)] = _bio_values(key)
                if target_pefp_lvr is not None:
                    lvr_score_key = (
                        "lvr_lfsr_tau" if use_lfsr_tau_lvr else "lvr_lfsr"
                    )
                    lvr_is_de, _ = self._compute_is_de_mask_from_scores(
                        _bio_values(lvr_score_key),
                        target_pefp=target_pefp_lvr,
                    )
                    df[_call_column_name("bio_lvr")] = np.asarray(
                        lvr_is_de, dtype=bool
                    )

            if "bio_kl" in metric_families:
                for key in ("kl_mean", "kl_sd", "kl_prob_effect"):
                    df[_bio_column_name(key)] = _bio_values(key)
                if target_pefp_kl is not None:
                    # KL is non-negative and non-directional, so we map
                    # ``prob_effect`` to an error score via lfer.
                    kl_lfer = 1.0 - _bio_values("kl_prob_effect")
                    kl_is_de, _ = self._compute_is_de_mask_from_scores(
                        kl_lfer, target_pefp=target_pefp_kl
                    )
                    if column_naming == "legacy":
                        df["kl_lfer"] = kl_lfer
                    else:
                        df["bio_kl_lfer"] = kl_lfer
                    df[_call_column_name("bio_kl")] = np.asarray(
                        kl_is_de, dtype=bool
                    )

            if "bio_aux" in metric_families:
                for key in (
                    "mu_A_mean",
                    "mu_B_mean",
                    "var_A_mean",
                    "var_B_mean",
                    "max_bio_expr",
                ):
                    df[_bio_column_name(key)] = _bio_values(key)

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
