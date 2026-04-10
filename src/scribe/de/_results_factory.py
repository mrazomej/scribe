"""Factory helpers for constructing DE results objects."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as _np
import jax.numpy as jnp

from ._extract import extract_alr_params

if TYPE_CHECKING:
    from .results import (
        ScribeDEResults,
        ScribeEmpiricalDEResults,
        ScribeParametricDEResults,
        ScribeShrinkageDEResults,
    )


def _is_results_object(obj) -> bool:
    """Check whether an object looks like a scribe results object."""
    return (
        hasattr(obj, "posterior_samples")
        and hasattr(obj, "model_config")
        and hasattr(obj, "var")
    )


def _extract_de_inputs(results, component=None):
    """Extract arrays needed by empirical DE from a results object.

    Parameters
    ----------
    results : object
        Fitted results object with posterior samples.
    component : int, optional
        Mixture component index (passed through for downstream slicing).

    Returns
    -------
    tuple
        ``(r_samples, p_samples, mu_samples, phi_samples, gene_names)``.
    """
    if results.posterior_samples is None:
        raise ValueError(
            "Results object has no posterior samples. "
            "Call results.get_posterior_samples() first."
        )

    r_samples = results.posterior_samples["r"]
    p_samples = results.posterior_samples.get("p")
    mu_samples = results.posterior_samples.get("mu")
    phi_samples = results.posterior_samples.get("phi")

    gene_names = None
    if results.var is not None:
        gene_names = results.var.index.tolist()

    return r_samples, p_samples, mu_samples, phi_samples, gene_names


def compare(
    model_A,
    model_B,
    gene_names: Optional[List[str]] = None,
    label_A: str = "A",
    label_B: str = "B",
    method: str = "parametric",
    component_A: Optional[int] = None,
    component_B: Optional[int] = None,
    paired: bool = False,
    n_samples_dirichlet: int = 1,
    rng_key=None,
    batch_size: int = 2048,
    gene_mask: Optional[jnp.ndarray] = None,
    p_samples_A: Optional[jnp.ndarray] = None,
    p_samples_B: Optional[jnp.ndarray] = None,
    sigma_grid: Optional[jnp.ndarray] = None,
    shrinkage_max_iter: int = 200,
    shrinkage_tol: float = 1e-8,
    compute_biological: bool = True,
) -> "ScribeDEResults":
    """Create a DE results object from fitted models or posterior samples.

    Parameters
    ----------
    model_A : object
        Condition-A input (results object, fitted parametric model, or raw
        posterior samples).
    model_B : object
        Condition-B input using the same interface as ``model_A``.
    gene_names : list of str, optional
        Output gene names.
    label_A : str, default='A'
        Display label for condition A.
    label_B : str, default='B'
        Display label for condition B.
    method : {'parametric', 'empirical', 'shrinkage'}, default='parametric'
        Differential-expression strategy.
    component_A : int, optional
        Mixture component index for condition A (empirical/shrinkage).
    component_B : int, optional
        Mixture component index for condition B (empirical/shrinkage).
    paired : bool, default=False
        Preserve posterior pairing for within-model comparisons.
    n_samples_dirichlet : int, default=1
        Number of simplex draws per posterior sample.
    rng_key : jax.random.PRNGKey, optional
        Random key used by composition sampling.
    batch_size : int, default=2048
        Batched sampling chunk size.
    gene_mask : jnp.ndarray, optional
        Optional keep-mask over full genes.
    p_samples_A : jnp.ndarray, optional
        Optional gene-specific success probabilities for condition A.
    p_samples_B : jnp.ndarray, optional
        Optional gene-specific success probabilities for condition B.
    sigma_grid : jnp.ndarray, optional
        Optional shrinkage prior scale grid.
    shrinkage_max_iter : int, default=200
        Maximum EM iterations for shrinkage.
    shrinkage_tol : float, default=1e-8
        EM convergence tolerance for shrinkage.
    compute_biological : bool, default=True
        Whether to retain samples needed for biological-level DE.

    Returns
    -------
    ScribeDEResults
        Concrete DE results object based on ``method``.
    """
    _a_is_results = _is_results_object(model_A)
    _b_is_results = _is_results_object(model_B)

    if _a_is_results or _b_is_results:
        if not (_a_is_results and _b_is_results):
            raise TypeError(
                "Both model_A and model_B must be results objects, "
                "or both must be raw arrays/dicts. "
                f"Got types: {type(model_A).__name__}, "
                f"{type(model_B).__name__}."
            )

        if method == "parametric":
            raise ValueError(
                "method='parametric' requires pre-fitted logistic-normal "
                "models (dicts or distributions), not results objects. "
                "Use method='empirical' or method='shrinkage' with "
                "results objects."
            )

        r_A, p_A, mu_A, phi_A, names_A = _extract_de_inputs(
            model_A, component_A
        )
        r_B, p_B, mu_B, phi_B, names_B = _extract_de_inputs(
            model_B, component_B
        )

        if gene_names is None:
            gene_names = names_A

        if p_samples_A is None:
            p_samples_A = p_A
        if p_samples_B is None:
            p_samples_B = p_B

        _mu_samples_A = mu_A
        _mu_samples_B = mu_B
        _phi_samples_A = phi_A
        _phi_samples_B = phi_B

        model_A = r_A
        model_B = r_B

    if not (_a_is_results and _b_is_results):
        _mu_samples_A = _mu_samples_B = None
        _phi_samples_A = _phi_samples_B = None

    if method == "parametric":
        return _compare_parametric(
            model_A,
            model_B,
            gene_names,
            label_A,
            label_B,
            gene_mask=gene_mask,
        )
    if method == "empirical":
        return _compare_empirical(
            model_A,
            model_B,
            gene_names=gene_names,
            label_A=label_A,
            label_B=label_B,
            component_A=component_A,
            component_B=component_B,
            paired=paired,
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=rng_key,
            batch_size=batch_size,
            gene_mask=gene_mask,
            p_samples_A=p_samples_A,
            p_samples_B=p_samples_B,
            compute_biological=compute_biological,
            mu_samples_A=_mu_samples_A,
            mu_samples_B=_mu_samples_B,
            phi_samples_A=_phi_samples_A,
            phi_samples_B=_phi_samples_B,
        )
    if method == "shrinkage":
        return _compare_shrinkage(
            model_A,
            model_B,
            gene_names=gene_names,
            label_A=label_A,
            label_B=label_B,
            component_A=component_A,
            component_B=component_B,
            paired=paired,
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=rng_key,
            batch_size=batch_size,
            gene_mask=gene_mask,
            p_samples_A=p_samples_A,
            p_samples_B=p_samples_B,
            sigma_grid=sigma_grid,
            shrinkage_max_iter=shrinkage_max_iter,
            shrinkage_tol=shrinkage_tol,
            compute_biological=compute_biological,
            mu_samples_A=_mu_samples_A,
            mu_samples_B=_mu_samples_B,
            phi_samples_A=_phi_samples_A,
            phi_samples_B=_phi_samples_B,
        )

    raise ValueError(
        f"Unknown method '{method}'. "
        f"Use 'parametric', 'empirical', or 'shrinkage'."
    )


def _compare_parametric(
    model_A,
    model_B,
    gene_names: Optional[List[str]],
    label_A: str,
    label_B: str,
    gene_mask: Optional[jnp.ndarray] = None,
) -> "ScribeParametricDEResults":
    """Build a parametric DE comparison from fitted ALR models."""
    from .results import ScribeParametricDEResults

    mu_A, W_A, d_A = extract_alr_params(model_A)
    mu_B, W_B, d_B = extract_alr_params(model_B)

    if mu_A.shape[-1] != mu_B.shape[-1]:
        raise ValueError(
            f"Model dimensions do not match: "
            f"model_A has D-1={mu_A.shape[-1]}, "
            f"model_B has D-1={mu_B.shape[-1]}.  "
            f"Both models must be fitted on the same gene set."
        )

    D_full = mu_A.shape[-1] + 1
    drop_last = gene_mask is not None
    D_user = D_full - 1 if drop_last else D_full

    if gene_mask is not None and gene_names is not None:
        gene_mask_arr = _np.asarray(gene_mask, dtype=bool)
        gene_names = [n for n, m in zip(gene_names, gene_mask_arr) if m]

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(D_user)]
    elif len(gene_names) != D_user:
        raise ValueError(
            f"gene_names has length {len(gene_names)} but models have "
            f"D={D_user} genes (CLR space, excluding 'other')."
        )

    return ScribeParametricDEResults(
        mu_A=mu_A,
        W_A=W_A,
        d_A=d_A,
        mu_B=mu_B,
        W_B=W_B,
        d_B=d_B,
        gene_names=gene_names,
        label_A=label_A,
        label_B=label_B,
        _drop_last_gene=drop_last,
    )


def _compare_empirical(
    r_samples_A: jnp.ndarray,
    r_samples_B: jnp.ndarray,
    gene_names: Optional[List[str]],
    label_A: str,
    label_B: str,
    component_A: Optional[int],
    component_B: Optional[int],
    paired: bool,
    n_samples_dirichlet: int,
    rng_key,
    batch_size: int,
    gene_mask: Optional[jnp.ndarray] = None,
    p_samples_A: Optional[jnp.ndarray] = None,
    p_samples_B: Optional[jnp.ndarray] = None,
    compute_biological: bool = True,
    mu_samples_A: Optional[jnp.ndarray] = None,
    mu_samples_B: Optional[jnp.ndarray] = None,
    phi_samples_A: Optional[jnp.ndarray] = None,
    phi_samples_B: Optional[jnp.ndarray] = None,
) -> "ScribeEmpiricalDEResults":
    """Build an empirical DE comparison from posterior concentration samples."""
    from ._empirical import (
        _slice_component,
        compute_delta_from_simplex,
        sample_compositions,
    )
    from .results import ScribeEmpiricalDEResults

    r_bio_A = _slice_component(r_samples_A, component_A, "A")
    r_bio_B = _slice_component(r_samples_B, component_B, "B")
    p_bio_A = (
        _slice_component(p_samples_A, component_A, "A")
        if p_samples_A is not None
        else None
    )
    p_bio_B = (
        _slice_component(p_samples_B, component_B, "B")
        if p_samples_B is not None
        else None
    )

    mu_bio_A = (
        _slice_component(mu_samples_A, component_A, "A")
        if mu_samples_A is not None
        else None
    )
    mu_bio_B = (
        _slice_component(mu_samples_B, component_B, "B")
        if mu_samples_B is not None
        else None
    )
    phi_bio_A = (
        _slice_component(phi_samples_A, component_A, "A")
        if phi_samples_A is not None
        else None
    )
    phi_bio_B = (
        _slice_component(phi_samples_B, component_B, "B")
        if phi_samples_B is not None
        else None
    )

    N_bio = min(r_bio_A.shape[0], r_bio_B.shape[0])
    r_bio_A, r_bio_B = r_bio_A[:N_bio], r_bio_B[:N_bio]
    if p_bio_A is not None:
        p_bio_A = p_bio_A[:N_bio]
    if p_bio_B is not None:
        p_bio_B = p_bio_B[:N_bio]
    if mu_bio_A is not None:
        mu_bio_A = mu_bio_A[:N_bio]
    if mu_bio_B is not None:
        mu_bio_B = mu_bio_B[:N_bio]
    if phi_bio_A is not None:
        phi_bio_A = phi_bio_A[:N_bio]
    if phi_bio_B is not None:
        phi_bio_B = phi_bio_B[:N_bio]

    # Move bio slices to CPU so the full (N, K, D) device arrays can
    # be freed.  These are only used for small summary statistics below.
    r_bio_A = _np.asarray(r_bio_A)
    r_bio_B = _np.asarray(r_bio_B)
    if p_bio_A is not None:
        p_bio_A = _np.asarray(p_bio_A)
    if p_bio_B is not None:
        p_bio_B = _np.asarray(p_bio_B)
    if mu_bio_A is not None:
        mu_bio_A = _np.asarray(mu_bio_A)
    if mu_bio_B is not None:
        mu_bio_B = _np.asarray(mu_bio_B)
    if phi_bio_A is not None:
        phi_bio_A = _np.asarray(phi_bio_A)
    if phi_bio_B is not None:
        phi_bio_B = _np.asarray(phi_bio_B)

    # Drop references to the full device arrays.  The bio slices above
    # are already on CPU and contain exactly the component-sliced,
    # truncated data that sample_compositions would recompute, so we
    # pass them directly (with component=None since slicing is done).
    # Releasing the locals lets Python's refcount free the GPU buffers
    # as soon as the caller's references also go away.
    del r_samples_A, r_samples_B, p_samples_A, p_samples_B
    del mu_samples_A, mu_samples_B, phi_samples_A, phi_samples_B

    simplex_A, simplex_B = sample_compositions(
        r_samples_A=r_bio_A,
        r_samples_B=r_bio_B,
        component_A=None,
        component_B=None,
        paired=paired,
        n_samples_dirichlet=n_samples_dirichlet,
        rng_key=rng_key,
        batch_size=batch_size,
        p_samples_A=p_bio_A,
        p_samples_B=p_bio_B,
    )

    # simplex_A/B and delta_samples are numpy (CPU) arrays -- the
    # batched sampling functions transfer each chunk to host immediately,
    # so no large GPU allocation ever occurs here.
    delta_samples = compute_delta_from_simplex(
        simplex_A, simplex_B, gene_mask=gene_mask
    )

    # Compute mean biological expression on CPU (bio arrays are already
    # numpy after the transfer above).
    mu_map_A_vec = None
    mu_map_B_vec = None
    if mu_bio_A is not None:
        mu_map_A_vec = mu_bio_A.mean(axis=0)
    elif r_bio_A is not None and p_bio_A is not None:
        _r_mean = r_bio_A.mean(axis=0)
        _p_mean = p_bio_A.mean(axis=0)
        _p_mean = _np.clip(_p_mean, 1e-7, 1.0 - 1e-7)
        mu_map_A_vec = _r_mean * _p_mean / (1.0 - _p_mean)
    if mu_bio_B is not None:
        mu_map_B_vec = mu_bio_B.mean(axis=0)
    elif r_bio_B is not None and p_bio_B is not None:
        _r_mean = r_bio_B.mean(axis=0)
        _p_mean = p_bio_B.mean(axis=0)
        _p_mean = _np.clip(_p_mean, 1e-7, 1.0 - 1e-7)
        mu_map_B_vec = _r_mean * _p_mean / (1.0 - _p_mean)

    all_gene_names = gene_names
    kept_gene_names = gene_names
    if gene_mask is not None and gene_names is not None:
        gene_mask_arr = _np.asarray(gene_mask, dtype=bool)
        kept_gene_names = [n for n, m in zip(gene_names, gene_mask_arr) if m]

    D = delta_samples.shape[1]
    if kept_gene_names is None:
        kept_gene_names = [f"gene_{i}" for i in range(D)]
    elif len(kept_gene_names) != D:
        raise ValueError(
            f"gene_names has length {len(kept_gene_names)} but samples have "
            f"D={D} genes."
        )

    result = ScribeEmpiricalDEResults(
        delta_samples=delta_samples,
        gene_names=kept_gene_names,
        label_A=label_A,
        label_B=label_B,
        r_samples_A=r_bio_A if compute_biological else None,
        r_samples_B=r_bio_B if compute_biological else None,
        p_samples_A=p_bio_A if compute_biological else None,
        p_samples_B=p_bio_B if compute_biological else None,
        mu_samples_A=mu_bio_A if compute_biological else None,
        mu_samples_B=mu_bio_B if compute_biological else None,
        phi_samples_A=phi_bio_A if compute_biological else None,
        phi_samples_B=phi_bio_B if compute_biological else None,
        simplex_A=simplex_A,
        simplex_B=simplex_B,
        mu_map_A=mu_map_A_vec,
        mu_map_B=mu_map_B_vec,
    )

    if gene_mask is not None:
        result._gene_mask = jnp.asarray(_np.asarray(gene_mask, dtype=bool))
    result._all_gene_names = (
        list(all_gene_names) if all_gene_names is not None else None
    )

    return result


def _compare_shrinkage(
    r_samples_A: jnp.ndarray,
    r_samples_B: jnp.ndarray,
    gene_names: Optional[List[str]],
    label_A: str,
    label_B: str,
    component_A: Optional[int],
    component_B: Optional[int],
    paired: bool,
    n_samples_dirichlet: int,
    rng_key,
    batch_size: int,
    gene_mask: Optional[jnp.ndarray] = None,
    p_samples_A: Optional[jnp.ndarray] = None,
    p_samples_B: Optional[jnp.ndarray] = None,
    sigma_grid: Optional[jnp.ndarray] = None,
    shrinkage_max_iter: int = 200,
    shrinkage_tol: float = 1e-8,
    compute_biological: bool = True,
    mu_samples_A: Optional[jnp.ndarray] = None,
    mu_samples_B: Optional[jnp.ndarray] = None,
    phi_samples_A: Optional[jnp.ndarray] = None,
    phi_samples_B: Optional[jnp.ndarray] = None,
) -> "ScribeShrinkageDEResults":
    """Build shrinkage DE by wrapping the empirical result object."""
    empirical = _compare_empirical(
        r_samples_A=r_samples_A,
        r_samples_B=r_samples_B,
        gene_names=gene_names,
        label_A=label_A,
        label_B=label_B,
        component_A=component_A,
        component_B=component_B,
        paired=paired,
        n_samples_dirichlet=n_samples_dirichlet,
        rng_key=rng_key,
        batch_size=batch_size,
        gene_mask=gene_mask,
        p_samples_A=p_samples_A,
        p_samples_B=p_samples_B,
        compute_biological=compute_biological,
        mu_samples_A=mu_samples_A,
        mu_samples_B=mu_samples_B,
        phi_samples_A=phi_samples_A,
        phi_samples_B=phi_samples_B,
    )

    return empirical.shrink(
        sigma_grid=sigma_grid,
        shrinkage_max_iter=shrinkage_max_iter,
        shrinkage_tol=shrinkage_tol,
    )


def compare_datasets(
    results,
    dataset_A: int,
    dataset_B: int,
    label_A: Optional[str] = None,
    label_B: Optional[str] = None,
    method: str = "shrinkage",
    component: Optional[int] = None,
    **kwargs,
) -> "ScribeDEResults":
    """Compare two datasets from a jointly fitted multi-dataset model.

    Parameters
    ----------
    results : object
        Multi-dataset results object supporting ``get_dataset``.
    dataset_A : int
        First dataset index.
    dataset_B : int
        Second dataset index.
    label_A : str, optional
        Label for dataset A.
    label_B : str, optional
        Label for dataset B.
    method : str, default='shrinkage'
        Method forwarded to :func:`compare`.
    component : int, optional
        Optional component selected before dataset slicing.
    **kwargs
        Extra keyword arguments forwarded to :func:`compare`.

    Returns
    -------
    ScribeDEResults
        Differential-expression result object.
    """
    n_datasets = getattr(
        getattr(results, "model_config", None), "n_datasets", None
    )
    if n_datasets is None:
        raise ValueError(
            "compare_datasets() requires a multi-dataset model "
            "(model_config.n_datasets must be set)."
        )

    if label_A is None:
        label_A = f"dataset_{dataset_A}"
    if label_B is None:
        label_B = f"dataset_{dataset_B}"

    working = results
    if component is not None:
        working = working.get_component(component)

    view_A = working.get_dataset(dataset_A)
    view_B = working.get_dataset(dataset_B)

    return compare(
        model_A=view_A,
        model_B=view_B,
        label_A=label_A,
        label_B=label_B,
        method=method,
        paired=True,
        **kwargs,
    )
