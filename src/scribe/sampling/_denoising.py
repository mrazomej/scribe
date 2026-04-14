"""Bayesian denoising of observed counts."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

from jax import random
import jax.numpy as jnp
import numpyro.distributions as dist

from scribe.models.components.likelihoods.beta_negative_binomial import (
    build_count_dist,
)

from ._helpers import _has_sample_dim, _slice_posterior_draw
from ._denoising_bnb import (
    _denoise_bnb_quadrature,
    _sample_p_posterior_bnb,
)


# Allowed values for individual method elements
_VALID_DENOISE_METHODS = {"mean", "mode", "sample"}


def _validate_denoise_method(method: Union[str, Tuple[str, str]]) -> None:
    """Validate the ``method`` argument for denoising functions.

    Accepts a single string or a tuple of two strings, each of which
    must be one of ``'mean'``, ``'mode'``, or ``'sample'``.

    Parameters
    ----------
    method : str or tuple of (str, str)
        The method specification to validate.

    Raises
    ------
    ValueError
        If the method is not a valid string or 2-tuple of valid strings.
    """
    if isinstance(method, str):
        if method not in _VALID_DENOISE_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_DENOISE_METHODS} or a "
                f"2-tuple thereof, got '{method}'"
            )
    elif isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError(
                f"method tuple must have exactly 2 elements, "
                f"got {len(method)}"
            )
        for i, m in enumerate(method):
            if m not in _VALID_DENOISE_METHODS:
                raise ValueError(
                    f"method[{i}] must be one of {_VALID_DENOISE_METHODS}, "
                    f"got '{m}'"
                )
    else:
        raise ValueError(
            f"method must be a string or a 2-tuple of strings, "
            f"got {type(method).__name__}"
        )


def _method_needs_rng(method: Union[str, Tuple[str, str]]) -> bool:
    """Return True if any element of ``method`` requires an RNG key."""
    if isinstance(method, str):
        return method == "sample"
    return "sample" in method


def denoise_counts(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray] = None,
    gate: Optional[jnp.ndarray] = None,
    method: Union[str, Tuple[str, str]] = "mean",
    rng_key: Optional[random.PRNGKey] = None,
    return_variance: bool = False,
    mixing_weights: Optional[jnp.ndarray] = None,
    component_assignment: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
    bnb_concentration: Optional[jnp.ndarray] = None,
    param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Denoise observed counts using the Bayesian posterior of true transcripts.

    Given observed UMI counts and posterior parameter estimates, computes
    the posterior distribution of the true (pre-capture, pre-dropout)
    transcript counts for each cell and gene.  The derivation exploits
    Poisson-Gamma conjugacy and the Poisson thinning property; see
    ``paper/_denoising.qmd`` for the full mathematics.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed UMI count matrix of shape ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion (total_count) parameter in canonical form.

        * Standard model, single param set: ``(n_genes,)``.
        * Standard model, multi-sample: ``(n_samples, n_genes)``.
        * Mixture model, single: ``(n_components, n_genes)``.
        * Mixture model, multi-sample: ``(n_samples, n_components, n_genes)``.
    p : jnp.ndarray
        Success probability (numpyro probs convention, *not* the paper's p).

        * Single param set: scalar or ``(n_components,)``.
        * Multi-sample: ``(n_samples,)`` or ``(n_samples, n_components)``.
    p_capture : jnp.ndarray or None, optional
        Per-cell capture probability νc.  Shape ``(n_cells,)``
        for a single param set or ``(n_samples, n_cells)`` for multi-sample.
        ``None`` for models without variable capture probability (nbdm, zinb),
        which is equivalent to νc = 1 (perfect capture).
    gate : jnp.ndarray or None, optional
        Zero-inflation gate probability.  Shape ``(n_genes,)`` or
        ``(n_components, n_genes)`` for a single param set; with a leading
        ``n_samples`` dimension for multi-sample.  ``None`` for models
        without zero-inflation.
    method : str or tuple of (str, str), optional
        Summary statistic to return.  Accepts either a single string
        applied uniformly to all positions, or a tuple
        ``(general_method, zi_zero_method)`` for independent control:

        * ``general_method``: used for non-zero positions and for all
          positions in non-ZINB models (no gate).
        * ``zi_zero_method``: used exclusively for zero positions in
          ZINB models (the gate/NB mixture posterior).

        Valid values for each element:

        * ``'mean'``: closed-form posterior mean (shrinkage estimator).
        * ``'mode'``: posterior mode (MAP denoised count).
        * ``'sample'``: one stochastic draw from the denoised posterior.

        A single string ``s`` is equivalent to ``(s, s)``.
        Default: ``'mean'``.
    rng_key : random.PRNGKey or None, optional
        JAX PRNG key.  Required when any element of ``method`` is
        ``'sample'``.
    return_variance : bool, optional
        If ``True``, return a dictionary with keys ``'denoised_counts'``
        and ``'variance'`` instead of a plain array.  Default: ``False``.
    mixing_weights : jnp.ndarray or None, optional
        Component mixing weights for mixture models.  Shape
        ``(n_components,)`` or ``(n_samples, n_components)``.
    component_assignment : jnp.ndarray or None, optional
        Pre-computed per-cell component assignments of shape
        ``(n_cells,)`` (integer indices).  When provided, each cell uses
        its assigned component's parameters instead of marginalising
        over components.  Ignored for non-mixture models.
    cell_batch_size : int or None, optional
        Process cells in batches of this size to limit peak memory.
        ``None`` processes all cells at once.
    bnb_concentration : jnp.ndarray or None, optional
        BNB concentration parameter.  ``None`` for non-BNB models.
    param_layouts : dict of str to AxisLayout, optional
        Semantic axis layouts keyed by canonical parameter name
        (``"r"``, ``"p"``, ``"gate"``, ``"p_capture"``, …).  These
        are provided automatically by results-object methods.  Passing
        ``None`` triggers a deprecated fallback that infers layouts
        from tensor shapes.

        .. deprecated::
            Omitting *param_layouts* is deprecated and will raise an
            error in a future release.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        If ``return_variance`` is ``False`` (default): denoised count matrix
        with shape ``(n_cells, n_genes)`` (single param set) or
        ``(n_samples, n_cells, n_genes)`` (multi-sample).

        If ``return_variance`` is ``True``: dictionary with keys
        ``'denoised_counts'`` and ``'variance'``, each with the shape above.

    Notes
    -----
    The denoising posterior for uncaptured transcripts is:

        dg | ug  ~  NB(rg + ug,  νc + (1 − νc)(1 − p))

    where ``p`` is the paper's success probability
    (= ``1 - canonical_p``).  The posterior mean simplifies to:

        E[mg | ug] = (ug + rg · p_can · (1 − νc)) / (1 − p_can · (1 − νc))

    For ZINB models, zero observations use a mixture posterior
    weighted by the probability that the zero came from the gate.

    See Also
    --------
    sample_biological_nb : Biological PPC (samples from NB prior, not
        conditioned on observed counts).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from scribe.sampling import denoise_counts
    >>> counts = jnp.array([[5, 0, 3], [0, 2, 0]])
    >>> r = jnp.array([2.0, 1.5, 3.0])
    >>> p = jnp.float32(0.6)
    >>> nu = jnp.array([0.5, 0.7])
    >>> denoised = denoise_counts(counts, r, p, p_capture=nu)
    >>> denoised.shape
    (2, 3)

    Tuple method for independent control of ZINB zeros:

    >>> denoised = denoise_counts(
    ...     counts, r, p, p_capture=nu,
    ...     gate=jnp.array([0.2, 0.3, 0.1]),
    ...     method=("mean", "sample"),
    ... )
    """
    _validate_denoise_method(method)
    if _method_needs_rng(method) and rng_key is None:
        rng_key = random.PRNGKey(42)

    is_mixture = mixing_weights is not None

    # param_layouts should always be provided by results-object callers.
    # The fallback infers layouts from tensor shapes and will be removed
    # in a future release.
    if param_layouts is None:
        warnings.warn(
            "Calling denoise_counts without param_layouts is deprecated. "
            "Pass param_layouts explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        _expected_r_rank = 2 if is_mixture else 1
        _has_sd = r.ndim > _expected_r_rank
        _params: Dict[str, jnp.ndarray] = {"r": r, "p": p}
        if p_capture is not None:
            _params["p_capture"] = p_capture
        if gate is not None:
            _params["gate"] = gate
        if mixing_weights is not None:
            _params["mixing_weights"] = mixing_weights
        if bnb_concentration is not None:
            _params["bnb_concentration"] = bnb_concentration
        from ..core.axis_layout import infer_layout

        param_layouts = {
            k: infer_layout(
                k, v,
                n_genes=counts.shape[-1],
                n_cells=counts.shape[0],
                n_components=(
                    r.shape[-2] if is_mixture and r.ndim >= 2 else None
                ),
                has_sample_dim=_has_sd,
            )
            for k, v in _params.items()
        }

    # Detect whether r carries a leading posterior-sample dimension
    # purely from AxisLayout metadata — no ndim heuristics.
    has_sample_dim = _has_sample_dim(param_layouts)

    # Derive MAP-level layouts (no sample dim) for the denoising functions,
    # since the sample dimension is handled by the outer loop.
    _base_layouts = {
        k: v.without_sample_dim() for k, v in param_layouts.items()
    }

    if not has_sample_dim:
        return _denoise_single(
            counts=counts,
            r=r,
            p=p,
            p_capture=p_capture,
            gate=gate,
            method=method,
            rng_key=rng_key,
            return_variance=return_variance,
            mixing_weights=mixing_weights,
            component_assignment=component_assignment,
            cell_batch_size=cell_batch_size,
            bnb_concentration=bnb_concentration,
            param_layouts=_base_layouts,
        )

    # Multi-sample path: iterate over posterior draws
    n_samples = r.shape[0]
    keys = (
        random.split(rng_key, n_samples)
        if _method_needs_rng(method)
        else [None] * n_samples
    )

    result_list: List[jnp.ndarray] = []
    var_list: List[jnp.ndarray] = []

    for s in range(n_samples):
        # Extract parameters for this single posterior draw, using
        # layout metadata to decide which arrays carry a sample axis.
        draw = _slice_posterior_draw(
            s,
            r=r,
            p=p,
            p_capture=p_capture,
            gate=gate,
            mixing_weights=mixing_weights,
            param_layouts=param_layouts,
            bnb_concentration=bnb_concentration,
        )

        # After slicing, the draw parameters are MAP-level (no sample dim);
        # pass _base_layouts for layout-driven flag computation.
        out = _denoise_single(
            counts=counts,
            r=draw["r"],
            p=draw["p"],
            p_capture=draw["p_capture"],
            gate=draw["gate"],
            method=method,
            rng_key=keys[s],
            return_variance=return_variance,
            mixing_weights=draw["mixing_weights"],
            component_assignment=component_assignment,
            cell_batch_size=cell_batch_size,
            bnb_concentration=draw["bnb_concentration"],
            param_layouts=_base_layouts,
        )

        if return_variance:
            result_list.append(out["denoised_counts"])
            var_list.append(out["variance"])
        else:
            result_list.append(out)

    stacked = jnp.stack(result_list, axis=0)
    if return_variance:
        return {
            "denoised_counts": stacked,
            "variance": jnp.stack(var_list, axis=0),
        }
    return stacked


def _denoise_single(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    return_variance: bool,
    mixing_weights: Optional[jnp.ndarray],
    component_assignment: Optional[jnp.ndarray],
    cell_batch_size: Optional[int],
    bnb_concentration: Optional[jnp.ndarray] = None,
    *,
    param_layouts: Dict[str, "AxisLayout"],
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Dispatch denoising for a single set of parameters.

    Handles both standard and mixture models.  For mixture models with
    ``component_assignment`` provided, gathers per-cell parameters and
    delegates to the standard path.  Otherwise marginalises over components.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts, shape ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion.  ``(n_genes,)`` for standard, ``(n_components, n_genes)``
        for mixture.
    p : jnp.ndarray
        Success probability.  Scalar or ``(n_components,)``.
    p_capture : jnp.ndarray or None
        Capture probability per cell, ``(n_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate probability.  ``(n_genes,)`` or ``(n_components, n_genes)`` or
        ``None``.
    method : str or tuple of (str, str)
        Denoising method.
    rng_key : random.PRNGKey or None
        PRNG key.
    return_variance : bool
        Whether to include variance in the output.
    mixing_weights : jnp.ndarray or None
        Component weights ``(n_components,)`` for mixture models.
    component_assignment : jnp.ndarray or None
        Per-cell component indices ``(n_cells,)``.
    cell_batch_size : int or None
        Batch cells to limit memory.
    bnb_concentration : jnp.ndarray or None
        Optional BNB concentration.
    param_layouts : dict of str to AxisLayout
        MAP-level semantic layouts used to derive boolean flags that
        replace ``ndim``/``shape`` heuristics.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts (and optionally variance).
    """
    is_mixture = mixing_weights is not None

    # Layout-derived flags — no shape heuristics.
    _p_has_comp = (
        param_layouts["p"].component_axis is not None
        if "p" in param_layouts else False
    )
    _p_has_genes = (
        param_layouts["p"].gene_axis is not None
        if "p" in param_layouts else False
    )
    _gate_has_comp = (
        param_layouts["gate"].component_axis is not None
        if "gate" in param_layouts else False
    )

    if is_mixture and component_assignment is not None:
        r_cell = r[component_assignment]

        p_cell = p[component_assignment] if _p_has_comp else p
        # (K,) gathered → (n_cells,); expand to (n_cells, 1) so
        # downstream broadcasts correctly with (n_cells, n_genes).
        if _p_has_comp and not _p_has_genes:
            p_cell = p_cell[:, None]

        g_cell = (
            gate[component_assignment]
            if gate is not None and _gate_has_comp
            else gate
        )

        return _denoise_standard(
            counts,
            r_cell,
            p_cell,
            p_capture,
            g_cell,
            method,
            rng_key,
            return_variance,
            cell_batch_size,
            bnb_concentration=bnb_concentration,
            # r and gate were gathered to per-cell via component_assignment;
            # p was gathered + expanded to (n_cells, 1) when per-component.
            r_is_per_cell=True,
            p_is_per_cell=_p_has_comp,
            gate_is_per_cell=_gate_has_comp,
            bnb_is_per_cell=False,
        )

    if is_mixture and component_assignment is None:
        return _denoise_mixture_marginal(
            counts,
            r,
            p,
            p_capture,
            gate,
            method,
            rng_key,
            return_variance,
            mixing_weights,
            cell_batch_size,
            bnb_concentration=bnb_concentration,
            param_layouts=param_layouts,
        )

    # Standard (non-mixture) model — no per-cell gathering needed.
    return _denoise_standard(
        counts,
        r,
        p,
        p_capture,
        gate,
        method,
        rng_key,
        return_variance,
        cell_batch_size,
        bnb_concentration=bnb_concentration,
    )


def _denoise_standard(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    return_variance: bool,
    cell_batch_size: Optional[int],
    bnb_concentration: Optional[jnp.ndarray] = None,
    *,
    r_is_per_cell: bool = False,
    p_is_per_cell: bool = False,
    gate_is_per_cell: bool = False,
    bnb_is_per_cell: bool = False,
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Denoise counts for a standard (non-mixture) model, single param set.

    Implements the core denoising formulas from ``paper/_denoising.qmd``.

    The key quantity is ``probs_post`` = canonical_p * (1 - nu_c), the
    numpyro-convention success probability of the posterior NB for the
    uncaptured transcripts d_g.  When ``p_capture`` is ``None`` (no VCP),
    ``probs_post = 0`` and denoising reduces to identity (plus any gate
    correction at zeros).

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion ``(n_genes,)`` or ``(n_cells, n_genes)`` when gathered
        from mixture component assignments.
    p : jnp.ndarray
        Success probability (scalar or broadcastable).
    p_capture : jnp.ndarray or None
        Capture probability ``(n_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate probability ``(n_genes,)`` or ``(n_cells, n_genes)`` or
        ``None``.
    method : str or tuple of (str, str)
        Denoising method.  A single string or ``(general, zi_zeros)``
        tuple; see :func:`denoise_counts`.
    rng_key : random.PRNGKey or None
        PRNG key (needed when any element of ``method`` is ``'sample'``).
    return_variance : bool
        Whether to return variance alongside denoised counts.
    cell_batch_size : int or None
        Optional cell batching.
    r_is_per_cell : bool
        ``True`` when ``r`` has been gathered to ``(n_cells, n_genes)``
        via component assignment and must be sliced per batch.
    p_is_per_cell : bool
        ``True`` when ``p`` has been gathered/expanded to
        ``(n_cells, ...)`` and must be sliced per batch.
    gate_is_per_cell : bool
        ``True`` when ``gate`` has been gathered to
        ``(n_cells, n_genes)`` and must be sliced per batch.
    bnb_is_per_cell : bool
        ``True`` when ``bnb_concentration`` has been gathered to
        ``(n_cells, n_genes)`` and must be sliced per batch.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts (and optionally variance).
    """
    n_cells, n_genes = counts.shape

    if cell_batch_size is None:
        cell_batch_size = n_cells

    needs_rng = _method_needs_rng(method)

    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size
    denoised_parts: List[jnp.ndarray] = []
    variance_parts: List[jnp.ndarray] = []

    for b in range(n_batches):
        start = b * cell_batch_size
        end = min(start + cell_batch_size, n_cells)
        counts_b = counts[start:end]

        pc_b = p_capture[start:end] if p_capture is not None else None
        r_b = r[start:end] if r_is_per_cell else r
        gate_b = gate[start:end] if gate is not None and gate_is_per_cell else gate
        bnb_b = (
            bnb_concentration[start:end]
            if bnb_concentration is not None and bnb_is_per_cell
            else bnb_concentration
        )
        p_b = p[start:end] if p_is_per_cell else p

        if needs_rng:
            rng_key, batch_key = random.split(rng_key)
        else:
            batch_key = None

        d, v = _denoise_batch(
            counts_b,
            r_b,
            p_b,
            pc_b,
            gate_b,
            method,
            batch_key,
            bnb_concentration=bnb_b,
        )
        denoised_parts.append(d)
        if return_variance:
            variance_parts.append(v)

    denoised = jnp.concatenate(denoised_parts, axis=0)
    if return_variance:
        variance = jnp.concatenate(variance_parts, axis=0)
        return {"denoised_counts": denoised, "variance": variance}
    return denoised


def _denoise_batch(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    bnb_concentration: Optional[jnp.ndarray] = None,
) -> tuple:
    """Denoise a single batch of cells (no further splitting).

    Returns ``(denoised, variance)`` where ``variance`` is always computed
    (the caller decides whether to keep it).

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts for this batch, ``(batch_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion, ``(n_genes,)`` or ``(batch_cells, n_genes)``.
    p : jnp.ndarray
        Success probability: scalar ``()``, gene-specific ``(n_genes,)``,
        or per-cell ``(batch_cells, 1)`` / ``(batch_cells, n_genes)``.
    p_capture : jnp.ndarray or None
        Capture probability ``(batch_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate probability, ``(n_genes,)`` or ``(batch_cells, n_genes)`` or
        ``None``.
    method : str or tuple of (str, str)
        Denoising method.  A single string applies uniformly; a tuple
        ``(general_method, zi_zero_method)`` allows the ZINB zero
        correction to use a different method from the rest.
    rng_key : random.PRNGKey or None
        PRNG key (needed when any element of ``method`` is ``'sample'``).

    Returns
    -------
    denoised : jnp.ndarray
        Denoised counts ``(batch_cells, n_genes)``.
    variance : jnp.ndarray
        Posterior variance ``(batch_cells, n_genes)``.
    """
    # Normalize method to (general_method, zi_zero_method)
    if isinstance(method, str):
        general_method, zi_zero_method = method, method
    else:
        general_method, zi_zero_method = method

    # Per-cell p arrives as (batch_cells, 1) from the gathering step in
    # _denoise_single / _denoise_mixture_marginal, gene-specific p as
    # (n_genes,), and scalar p as ().  All broadcast correctly with
    # (batch_cells, n_genes) tensors without further reshaping.
    p_eff = p

    # probs_post is the numpyro probs for the posterior NB of uncaptured
    # transcripts d_g.  probs_post = canonical_p * (1 - nu_c).
    # When no VCP (nu_c = 1): probs_post = 0 → d_g = 0 → identity.
    if p_capture is not None:
        nu = p_capture[:, None]  # (batch_cells, 1)
        probs_post = p_eff * (1.0 - nu)
    else:
        probs_post = jnp.zeros(())

    # Complement: 1 - probs_post = p'_paper (the posterior "success" prob
    # in the paper's convention).  Used as denominator in most formulas.
    one_minus_pp = 1.0 - probs_post

    # ------------------------------------------------------------------
    # Denoising: BNB branch (quadrature / augmented sampling) when
    # bnb_concentration is present and VCP is active, NB closed-form
    # otherwise.  When BNB is active, we precompute the quadrature
    # mean/var once so that both the main path and the ZINB zero-
    # correction block can reuse them.
    # ------------------------------------------------------------------
    _use_bnb_denoise = bnb_concentration is not None and p_capture is not None

    # Precompute BNB quadrature results (reused across main + ZINB paths)
    _bnb_mean = _bnb_var = None
    if _use_bnb_denoise:
        _bnb_mean, _bnb_var = _denoise_bnb_quadrature(
            counts, r, p_eff, p_capture, bnb_concentration
        )

    if _use_bnb_denoise and general_method in ("mean", "mode"):
        # BNB quadrature denoising for mean/mode.  The mode is
        # approximated as floor(quadrature_mean) since the BNB
        # denoising posterior has no closed-form mode.
        denoised_nb = _bnb_mean
        var_nb = _bnb_var
        if general_method == "mode":
            denoised_nb = jnp.floor(denoised_nb)

    elif _use_bnb_denoise and general_method == "sample":
        # Augmented BNB sampling: draw p from its 1D posterior,
        # then draw d from the conditional NB denoising posterior.
        key_p, key_nb, rng_key = random.split(rng_key, 3)
        p_sampled = _sample_p_posterior_bnb(
            key_p, counts, r, p_eff, p_capture, bnb_concentration
        )
        # Conditional posterior NB probs: p_sampled * (1 - nu)
        probs_cond = p_sampled * (1.0 - nu)
        alpha_cond = r + counts
        d_sample = dist.NegativeBinomialProbs(
            total_count=alpha_cond, probs=probs_cond
        ).sample(key_nb)
        denoised_nb = counts + d_sample
        # Use quadrature variance as the best estimate
        var_nb = _bnb_var

    else:
        # Standard NB closed-form denoising
        if general_method == "mean":
            denoised_nb = (counts + r * probs_post) / one_minus_pp
        elif general_method == "mode":
            alpha = r + counts
            d_mode = jnp.floor(
                jnp.maximum(alpha - 1.0, 0.0) * probs_post / one_minus_pp
            )
            denoised_nb = counts + d_mode
        else:
            # general_method == "sample"
            alpha = r + counts
            key_nb, rng_key = random.split(rng_key)
            d_sample = build_count_dist(
                alpha, probs_post, bnb_concentration
            ).sample(key_nb)
            denoised_nb = counts + d_sample

        # NB variance: alpha * probs_post / (1-probs_post)^2
        var_nb = (r + counts) * probs_post / one_minus_pp**2

    # ------------------------------------------------------------------
    # ZINB zero correction: when gate is present and u_g = 0, the
    # denoised posterior is a mixture of gate and NB pathways.
    # Uses zi_zero_method for the zero positions.
    # ------------------------------------------------------------------
    if gate is not None:
        is_zero = counts == 0

        # Gate weight w = P(gate fired | u=0)
        w = _compute_gate_weight(gate, r, p_eff, one_minus_pp)

        # Gate pathway: the cell was expressing normally but dropout
        # prevented observation.  Denoised count follows the prior
        # NB(r, p) (or BNB(r, p, omega) when BNB is active).
        if zi_zero_method == "mean":
            gate_val = r * p_eff / (1.0 - p_eff)
        elif zi_zero_method == "mode":
            gate_val = jnp.floor(
                jnp.maximum(r - 1.0, 0.0) * p_eff / (1.0 - p_eff)
            )
        else:
            key_gate, rng_key = random.split(rng_key)
            gate_val = build_count_dist(r, p_eff, bnb_concentration).sample(
                key_gate
            )

        # NB/BNB pathway value at u=0: the posterior for unobserved mRNA
        # given that the NB/BNB component produced the zero.  For VCP
        # models this is positive; without VCP probs_post=0 so it is 0.
        if zi_zero_method == general_method:
            nb_zero_val = denoised_nb
        elif _use_bnb_denoise and zi_zero_method in ("mean", "mode"):
            # Reuse precomputed BNB quadrature mean/mode
            nb_zero_val = _bnb_mean
            if zi_zero_method == "mode":
                nb_zero_val = jnp.floor(nb_zero_val)
        elif _use_bnb_denoise and zi_zero_method == "sample":
            # Augmented BNB sampling for the zero-correction path
            key_p_z, key_nb_z, rng_key = random.split(rng_key, 3)
            p_sampled_z = _sample_p_posterior_bnb(
                key_p_z, counts, r, p_eff, p_capture, bnb_concentration
            )
            probs_cond_z = p_sampled_z * (1.0 - nu)
            alpha_z = r + counts
            d_z = dist.NegativeBinomialProbs(
                total_count=alpha_z, probs=probs_cond_z
            ).sample(key_nb_z)
            nb_zero_val = counts + d_z
        elif zi_zero_method == "mean":
            nb_zero_val = (counts + r * probs_post) / one_minus_pp
        elif zi_zero_method == "mode":
            alpha_z = r + counts
            d_mode_z = jnp.floor(
                jnp.maximum(alpha_z - 1.0, 0.0) * probs_post / one_minus_pp
            )
            nb_zero_val = counts + d_mode_z
        else:
            # Sample from the NB posterior for unobserved transcripts
            # at u=0.  For VCP, d ~ NB(r, probs_post) gives the mRNA
            # lost to capture.  Without VCP probs_post=0 → d=0.
            alpha_z = r + counts
            key_nb_z, rng_key = random.split(rng_key)
            d_sample_z = build_count_dist(
                alpha_z, probs_post, bnb_concentration
            ).sample(key_nb_z)
            nb_zero_val = counts + d_sample_z

        # Combine gate and NB pathways at zero positions
        if zi_zero_method == "mean":
            zinb_zero = w * gate_val + (1.0 - w) * nb_zero_val
        elif zi_zero_method == "mode":
            zinb_zero = jnp.where(w > 0.5, gate_val, nb_zero_val)
        else:
            # Sample: use w to decide whether the zero was from dropout.
            # If gate fired (dropout), sample a replacement from the
            # biological prior NB(r, p).  If genuine NB zero, use the
            # NB posterior (accounts for mRNA lost to capture in VCP;
            # collapses to 0 without VCP since probs_post=0).
            key_bern, rng_key = random.split(rng_key)
            chose_gate = dist.Bernoulli(probs=w).sample(key_bern).astype(bool)
            zinb_zero = jnp.where(chose_gate, gate_val, nb_zero_val)

        denoised = jnp.where(is_zero, zinb_zero, denoised_nb)

        # Variance at zero positions: law of total variance for the mixture
        var_gate = r * p_eff / (1.0 - p_eff) ** 2
        var_nb_zero = var_nb  # already correct at u=0 positions
        mean_gate = r * p_eff / (1.0 - p_eff)
        mean_nb_zero = (r * probs_post) / one_minus_pp
        mixture_var = (
            w * var_gate
            + (1.0 - w) * var_nb_zero
            + w * (1.0 - w) * (mean_gate - mean_nb_zero) ** 2
        )
        variance = jnp.where(is_zero, mixture_var, var_nb)
    else:
        denoised = denoised_nb
        variance = var_nb

    return denoised, variance


def _compute_gate_weight(
    gate: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    one_minus_probs_post: jnp.ndarray,
) -> jnp.ndarray:
    """Posterior probability that a zero observation came from the gate.

    Implements Bayes' rule for the zero-inflation mixture:

        w = g / (g + (1 − g) · p̂_paper^rg)

    where ``p̂_paper = (1 − p_can) / (1 − p_can·(1 − νc))``.

    In the numpyro probs convention, the NB probability of observing zero
    is ``(1 − probs)^r``.  For the *observation* model the relevant probs
    is ``p_hat_numpyro``, and its complement is exactly
    ``(1 - canonical_p) / one_minus_probs_post``.

    Parameters
    ----------
    gate : jnp.ndarray
        Gate probability, ``(n_genes,)`` or ``(batch_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion, ``(n_genes,)`` or ``(batch_cells, n_genes)``.
    p : jnp.ndarray
        Canonical success probability (scalar or broadcastable).
    one_minus_probs_post : jnp.ndarray
        ``1 - probs_post``, the paper's p′.  Shape ``()`` or
        ``(batch_cells, 1)``.

    Returns
    -------
    jnp.ndarray
        Gate weight *w* with the same shape as ``gate`` (broadcast).
    """
    # p_hat_paper = (1 - canonical_p) / one_minus_probs_post
    # P_NB(u=0) = p_hat_paper^r  in the paper convention
    p_hat_paper = (1.0 - p) / one_minus_probs_post
    nb_zero_prob = p_hat_paper**r

    w = gate / (gate + (1.0 - gate) * nb_zero_prob)
    return w


def _denoise_mixture_marginal(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    return_variance: bool,
    mixing_weights: jnp.ndarray,
    cell_batch_size: Optional[int],
    bnb_concentration: Optional[jnp.ndarray] = None,
    *,
    param_layouts: Dict[str, "AxisLayout"],
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Denoise by marginalising over mixture components.

    For ``method='mean'`` (or general_method ``'mean'``):

        E[mg | ug] = Σk wk · E[mg | ug, comp=k]

    For ``method='sample'`` (or general_method ``'sample'``): sample a
    component per cell from ``mixing_weights``, then sample from that
    component's denoised posterior.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion ``(n_components, n_genes)``.
    p : jnp.ndarray
        Success probability, scalar or ``(n_components,)``.
    p_capture : jnp.ndarray or None
        Capture probability ``(n_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate ``(n_genes,)`` or ``(n_components, n_genes)`` or ``None``.
    method : str or tuple of (str, str)
        Denoising method.
    rng_key : random.PRNGKey or None
        PRNG key.
    return_variance : bool
        Whether to return variance.
    mixing_weights : jnp.ndarray
        Component weights ``(n_components,)``.
    cell_batch_size : int or None
        Cell batching.
    bnb_concentration : jnp.ndarray or None
        Optional BNB concentration.
    param_layouts : dict of str to AxisLayout
        MAP-level semantic layouts for layout-driven flag computation.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts (and optionally variance).
    """
    general_method = method[0] if isinstance(method, tuple) else method

    n_components = r.shape[0]

    # Layout-derived flags — no shape heuristics.
    p_is_comp = (
        param_layouts["p"].component_axis is not None
        if "p" in param_layouts else False
    )
    _p_has_genes = (
        param_layouts["p"].gene_axis is not None
        if "p" in param_layouts else False
    )
    _gate_is_comp = (
        param_layouts["gate"].component_axis is not None
        if "gate" in param_layouts else False
    )

    if general_method == "sample":
        key_comp, key_rest = random.split(rng_key)
        comp = dist.Categorical(probs=mixing_weights).sample(
            key_comp, (counts.shape[0],)
        )
        r_cell = r[comp]
        p_cell = p[comp] if p_is_comp else p
        # (K,) gathered → (n_cells,); expand for broadcasting.
        if p_is_comp and not _p_has_genes:
            p_cell = p_cell[:, None]
        g_cell = gate[comp] if gate is not None and _gate_is_comp else gate
        return _denoise_standard(
            counts,
            r_cell,
            p_cell,
            p_capture,
            g_cell,
            method,
            key_rest,
            return_variance,
            cell_batch_size,
            bnb_concentration=bnb_concentration,
            # r and gate gathered per-cell from component assignments.
            r_is_per_cell=True,
            p_is_per_cell=p_is_comp,
            gate_is_per_cell=_gate_is_comp,
            bnb_is_per_cell=False,
        )

    # Marginalise over components (mean or mode for the general path).
    # An rng_key may still be needed if zi_zero_method is "sample".
    needs_rng = _method_needs_rng(method)
    n_cells, n_genes = counts.shape
    denoised_acc = jnp.zeros((n_cells, n_genes))
    variance_acc = jnp.zeros((n_cells, n_genes))

    for k in range(n_components):
        r_k = r[k]
        p_k = p[k] if p_is_comp else p
        g_k = gate[k] if gate is not None and _gate_is_comp else gate

        # Split rng_key per component if the zi_zero path needs sampling
        if needs_rng:
            rng_key, comp_key = random.split(rng_key)
        else:
            comp_key = None

        # Each component slice is gene-level, not per-cell.
        out_k = _denoise_standard(
            counts,
            r_k,
            p_k,
            p_capture,
            g_k,
            method,
            comp_key,
            True,
            cell_batch_size,
            bnb_concentration=bnb_concentration,
        )

        d_k = out_k["denoised_counts"]
        v_k = out_k["variance"]
        w_k = mixing_weights[k]

        denoised_acc = denoised_acc + w_k * d_k
        # Law of total variance: Var = E[Var_k] + Var[E_k]
        variance_acc = variance_acc + w_k * (v_k + d_k**2)

    variance = variance_acc - denoised_acc**2

    if return_variance:
        return {"denoised_counts": denoised_acc, "variance": variance}
    return denoised_acc
