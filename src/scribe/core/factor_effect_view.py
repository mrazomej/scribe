"""Read-out views over the fitted per-factor effects of a multi-factor fit.

The additive multi-factor expression hierarchy decomposes the leaf-level
log-mean as ``log mu^(leaf) = log mu^pop + sum_f effect_f[level_f(leaf)]``,
where each factor's effect is a deterministic site (``mu_<factor>_effect``,
shape ``(N, [K,] L_f, G)``) captured during posterior sampling. ``get_factor_
effect`` exposes those effects for inspection — the treatment contrast, the
per-donor deviations, and the learned shrinkage scale — without re-running any
inference.

Interpretation. The effects live in **log-mean (natural-log) space**:

* For a **random** factor (zero-mean NCP, e.g. donor) each level's effect is a
  meaningful deviation from the population mean, so ``view[level]`` and
  ``view.effects()`` are directly interpretable, and ``view.scale`` is the
  learned heterogeneity magnitude.
* For a **fixed** contrast factor (e.g. condition) individual level effects are
  identified only up to a shift the population intercept absorbs; the
  identified quantity is the **contrast** ``view.contrast(level_A, level_B)``.

These are log-mu effects only — they ignore the ``p``/capture contributions, so
they are a structural read-out, not a compositional DE estimand (use
``compare_groups`` for the latter).
"""

import numpy as np

from .grouping_view import _require_spec


class FactorEffectView:
    """Fitted effect of one grouping factor, indexed by level.

    Attributes
    ----------
    factor_name : str
        The grouping factor.
    levels : list of str
        Factor levels, aligned with the effect site's level axis.
    effect_type : {"random", "fixed"}
        How the effect was parameterised.
    prior : str
        Expression prior family for the factor (e.g. ``"gaussian"``).
    scale : ndarray or float or None
        Learned shrinkage-scale samples ``(N,)`` for a gaussian random effect,
        the fixed constant for a fixed effect, or ``None`` (e.g. horseshoe,
        whose per-gene scale is folded into the effect).
    gene_names : sequence or None
        Gene labels, length ``G`` when available.
    """

    def __init__(
        self,
        factor_name,
        levels,
        effect_samples,
        scale,
        effect_type,
        prior,
        gene_names,
    ):
        self.factor_name = factor_name
        self.levels = list(levels)
        self._effect = np.asarray(effect_samples)
        self.scale = scale
        self.effect_type = effect_type
        self.prior = prior
        self.gene_names = gene_names

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def _level_axis(self) -> int:
        # Effect is (N, [K,] L_f, G); the level axis is second-to-last.
        return self._effect.ndim - 2

    def _level_index(self, level) -> int:
        if level not in self.levels:
            raise KeyError(
                f"unknown level {level!r} for factor {self.factor_name!r}; "
                f"levels: {self.levels}."
            )
        return self.levels.index(level)

    def samples(self, level) -> np.ndarray:
        """Per-draw effect for ``level``, shape ``(N, [K,] G)`` (log-mu space)."""
        return np.take(
            self._effect, self._level_index(level), axis=self._level_axis()
        )

    def __getitem__(self, level) -> np.ndarray:
        return self.samples(level)

    def map_effect(self, level) -> np.ndarray:
        """Posterior-mean effect for ``level``, shape ``([K,] G)``."""
        return self.samples(level).mean(axis=0)

    def contrast(self, level_A, level_B) -> np.ndarray:
        """Per-draw effect contrast ``A - B``, shape ``(N, [K,] G)``.

        The identified quantity for a fixed contrast factor — e.g. the
        treatment log-mean effect ``contrast("drug", "control")``.
        """
        return self.samples(level_A) - self.samples(level_B)

    def map_contrast(self, level_A, level_B) -> np.ndarray:
        """Posterior-mean effect contrast ``A - B``, shape ``([K,] G)``."""
        return self.contrast(level_A, level_B).mean(axis=0)

    def effects(self) -> np.ndarray:
        """Posterior-mean effect per level, shape ``(L_f, [K,] G)``."""
        return np.stack([self.map_effect(lv) for lv in self.levels], axis=0)

    def __len__(self) -> int:
        return self.n_levels

    def __iter__(self):
        return iter(self.levels)

    def __repr__(self):
        scale_kind = (
            "fixed"
            if np.isscalar(self.scale)
            else ("learned" if self.scale is not None else "none")
        )
        return (
            f"FactorEffectView(factor={self.factor_name!r}, "
            f"levels={self.levels}, effect_type={self.effect_type!r}, "
            f"prior={self.prior!r}, scale={scale_kind})"
        )


def get_factor_effect(results, factor_name: str) -> FactorEffectView:
    """Expose the fitted effect of ``factor_name`` for inspection.

    Parameters
    ----------
    results : object
        A multi-factor results object (``model_config.grouping_spec`` set).
    factor_name : str
        A base grouping factor that received an expression effect.

    Returns
    -------
    FactorEffectView
        The per-level effects (log-mu space), the learned/fixed scale, and
        metadata.
    """
    spec = _require_spec(results)
    base = spec.base_factors
    names = [f.name for f in base]
    if factor_name not in names:
        raise ValueError(f"unknown factor {factor_name!r}; base factors: {names}.")
    fac = next(f for f in base if f.name == factor_name)

    samples = getattr(results, "posterior_samples", None)
    if samples is None and hasattr(results, "get_posterior_samples"):
        samples = results.get_posterior_samples()
    if samples is None:
        raise ValueError(
            "get_factor_effect() needs posterior samples; call "
            "results.get_posterior_samples() first."
        )

    # The site prefix mirrors the factory's _multifactor_site_prefix; the
    # expression target is mu (mean parameterizations) or r (canonical/standard).
    safe = factor_name.replace(":", "__")
    candidates = [f"mu_{safe}_effect", f"r_{safe}_effect"]
    effect_key = next((k for k in candidates if k in samples), None)
    if effect_key is None:
        raise ValueError(
            f"no fitted effect for factor {factor_name!r} — its expression "
            f"prior is likely 'none' (looked for {candidates}). Factors with a "
            "non-'none' expression prior carry an additive effect."
        )
    effect = np.asarray(samples[effect_key])
    target = "mu" if effect_key.startswith("mu_") else "r"
    prefix = f"{target}_{safe}"

    family = fac.family("expression")
    if fac.effect_type == "fixed":
        scale = fac.fixed_scale if fac.fixed_scale is not None else 1.0
    elif family == "gaussian" and f"{prefix}_scale" in samples:
        scale = np.asarray(samples[f"{prefix}_scale"])
    else:
        # Horseshoe folds a per-gene scale into the effect; expose effect only.
        scale = None

    gene_names = getattr(results, "gene_names", None)
    return FactorEffectView(
        factor_name=factor_name,
        levels=list(fac.levels),
        effect_samples=effect,
        scale=scale,
        effect_type=fac.effect_type,
        prior=family,
        gene_names=gene_names,
    )
