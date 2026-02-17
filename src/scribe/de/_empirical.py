"""Non-parametric differential expression from posterior samples.

This module provides the empirical (Monte Carlo) path for Bayesian
differential expression.  Instead of assuming the ALR/CLR marginals are
Gaussian and using ``norm.cdf`` for lfsr, it computes all DE statistics
by **counting** over paired posterior samples — no distributional
assumptions required.

Main functions
--------------
- ``compute_clr_differences`` : Go from Dirichlet concentration samples
  ``r`` to CLR-space differences ``Delta = CLR(rho_A) - CLR(rho_B)``.
  Handles mixture models (component slicing) and both independent and
  within-mixture (paired) comparisons.
- ``empirical_differential_expression`` : Compute gene-level DE
  statistics (lfsr, prob_effect, etc.) from a ``(N, D)`` matrix of CLR
  differences by vectorized counting.

The validity of pairwise differencing rests on posterior independence:

- **Independent models** (``paired=False``): the joint posterior
  factorises as ``pi(rho_A, rho_B | data_A, data_B) = pi(rho_A | data_A)
  * pi(rho_B | data_B)``, so any pairing of samples is valid.  We pair
  by index for convenience.
- **Within-mixture** (``paired=True``): both components come from the
  same posterior draw, so we must preserve the sample-index pairing.
  Dirichlet draws use the same per-sample RNG sub-key for both
  components.

References
----------
- Stephens, M. (2017). "False discovery rates: a new deal."
  *Biostatistics*, 18(2), 275--294.  (lfsr concept)
"""

from typing import Optional, List

import numpy as np
import jax
import jax.numpy as jnp
from jax import random


# --------------------------------------------------------------------------
# Gene aggregation for expression filtering
# --------------------------------------------------------------------------


def _aggregate_genes(
    r_A: jnp.ndarray,
    r_B: jnp.ndarray,
    gene_mask: jnp.ndarray,
) -> tuple:
    """Pool filtered genes into a single "other" pseudo-gene.

    Genes marked ``False`` in ``gene_mask`` are summed into a single
    aggregate concentration that is appended as the last column.  This
    preserves the total Dirichlet concentration exactly so that the
    simplex constraint is maintained downstream.

    Parameters
    ----------
    r_A : jnp.ndarray, shape ``(N, D)``
        Dirichlet concentration samples for condition A.
    r_B : jnp.ndarray, shape ``(N, D)``
        Dirichlet concentration samples for condition B.
    gene_mask : jnp.ndarray, shape ``(D,)``
        Boolean mask.  ``True`` = keep the gene, ``False`` = pool into
        "other".

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        ``(r_A_agg, r_B_agg)`` each of shape ``(N, D_kept + 1)``, where
        ``D_kept = gene_mask.sum()``.  The last column is the summed
        concentration of all filtered genes.

    Raises
    ------
    ValueError
        If ``gene_mask`` has the wrong length or keeps no genes.
    """
    gene_mask = jnp.asarray(gene_mask, dtype=bool)

    if gene_mask.ndim != 1 or gene_mask.shape[0] != r_A.shape[1]:
        raise ValueError(
            f"gene_mask must be a 1-D boolean array of length D={r_A.shape[1]}, "
            f"got shape {gene_mask.shape}."
        )

    D_kept = int(gene_mask.sum())
    if D_kept == 0:
        raise ValueError("gene_mask must keep at least one gene (all False).")

    # Split kept vs. filtered for each condition
    r_A_kept = r_A[:, gene_mask]  # (N, D_kept)
    r_A_other = r_A[:, ~gene_mask].sum(axis=1, keepdims=True)  # (N, 1)
    r_A_agg = jnp.concatenate([r_A_kept, r_A_other], axis=1)

    r_B_kept = r_B[:, gene_mask]
    r_B_other = r_B[:, ~gene_mask].sum(axis=1, keepdims=True)
    r_B_agg = jnp.concatenate([r_B_kept, r_B_other], axis=1)

    return r_A_agg, r_B_agg


# --------------------------------------------------------------------------
# Expression mask helper
# --------------------------------------------------------------------------


def compute_expression_mask(
    results_A,
    results_B,
    component_A: int,
    component_B: int,
    min_mean_expression: float = 1.0,
    counts_A: Optional[jnp.ndarray] = None,
    counts_B: Optional[jnp.ndarray] = None,
) -> np.ndarray:
    """Build a boolean gene mask from MAP mean-expression estimates.

    A gene passes the filter if its MAP mean expression ``mu`` is at
    least ``min_mean_expression`` in **either** condition.  This
    preserves genes that are genuinely condition-specific (highly
    expressed in one condition only).

    The function works with any parameterization: if the MAP estimates
    include ``mu`` directly (``linked`` / ``mean_prob`` / ``mean_odds``
    / ``odds_ratio``), it is used as-is.  For the ``standard``
    parameterization (which provides ``r`` and ``p`` but not ``mu``),
    the mean expression is derived as ``mu = r * p / (1 - p)``.

    Parameters
    ----------
    results_A : ScribeSVIResults
        Fitted model for condition A.
    results_B : ScribeSVIResults
        Fitted model for condition B.
    component_A : int
        Mixture-component index for condition A.
    component_B : int
        Mixture-component index for condition B.
    min_mean_expression : float, default=1.0
        Minimum MAP mean expression (in count space) for a gene to pass
        the filter.  Genes below this threshold in *both* conditions are
        pooled into "other".
    counts_A : jnp.ndarray, optional
        Count matrix for condition A.  Required when the model uses
        amortized capture probability.
    counts_B : jnp.ndarray, optional
        Count matrix for condition B.  Required when the model uses
        amortized capture probability.

    Returns
    -------
    np.ndarray, shape ``(D,)``
        Boolean mask — ``True`` for genes that pass the expression
        filter.
    """
    map_A = results_A.get_component(component_A).get_map(
        use_mean=True, canonical=True, verbose=False, counts=counts_A
    )
    map_B = results_B.get_component(component_B).get_map(
        use_mean=True, canonical=True, verbose=False, counts=counts_B
    )

    mu_A = np.asarray(_extract_mu(map_A))
    mu_B = np.asarray(_extract_mu(map_B))

    return (mu_A >= min_mean_expression) | (mu_B >= min_mean_expression)


# ------------------------------------------------------------------------------


def _extract_mu(map_estimates: dict) -> jnp.ndarray:
    """Extract or derive mean expression ``mu`` from MAP estimates.

    For parameterizations that include ``mu`` directly (``linked``,
    ``mean_prob``, ``mean_odds``, ``odds_ratio``) the value is returned
    as-is.  For the ``standard`` parameterization (``r`` and ``p`` only),
    ``mu`` is computed as ``r * p / (1 - p)`` from the negative-binomial
    mean formula.

    Parameters
    ----------
    map_estimates : dict
        MAP parameter dictionary returned by ``get_map(canonical=True)``.

    Returns
    -------
    jnp.ndarray
        Mean expression vector, shape ``(D,)`` (or ``(K, D)`` for
        mixture models before component slicing).

    Raises
    ------
    ValueError
        If neither ``mu`` nor both ``r`` and ``p`` are present.
    """
    if "mu" in map_estimates:
        return map_estimates["mu"]

    if "r" in map_estimates and "p" in map_estimates:
        r = map_estimates["r"]
        p = map_estimates["p"]
        return r * p / (1.0 - p)

    available = sorted(map_estimates.keys())
    raise ValueError(
        "Cannot determine mean expression: MAP estimates contain neither "
        "'mu' nor both 'r' and 'p'.  "
        f"Available keys: {available}"
    )


# --------------------------------------------------------------------------
# CLR difference computation
# --------------------------------------------------------------------------


def compute_clr_differences(
    r_samples_A: jnp.ndarray,
    r_samples_B: jnp.ndarray,
    component_A: Optional[int] = None,
    component_B: Optional[int] = None,
    paired: bool = False,
    n_samples_dirichlet: int = 1,
    rng_key=None,
    batch_size: int = 2048,
    gene_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute CLR-space posterior differences from Dirichlet concentration samples.

    Takes posterior samples of Dirichlet concentration parameters ``r``
    for two conditions, draws from ``Dirichlet(r)``, transforms to
    centered log-ratio (CLR) coordinates, and returns the paired
    differences.

    Parameters
    ----------
    r_samples_A : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Posterior samples of Dirichlet concentration parameters for
        condition A.  If 3D, ``K`` is the number of mixture components
        and ``component_A`` must be specified.
    r_samples_B : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Posterior samples for condition B.
    component_A : int, optional
        Mixture component index to extract from ``r_samples_A``.
        Required if ``r_samples_A`` is 3D.
    component_B : int, optional
        Mixture component index to extract from ``r_samples_B``.
        Required if ``r_samples_B`` is 3D.
    paired : bool, default=False
        If ``True``, use the **same** RNG sub-key per sample index for
        both conditions (required for within-mixture comparisons).
        If ``False``, use independent RNG keys (valid for independent
        posterior comparisons).
    n_samples_dirichlet : int, default=1
        Number of Dirichlet draws per posterior sample.  When > 1, each
        posterior sample yields multiple simplex draws; the total number
        of CLR difference samples is ``N * n_samples_dirichlet``.
    rng_key : jax.random.PRNGKey, optional
        JAX PRNG key.  If ``None``, uses ``jax.random.PRNGKey(0)``.
    batch_size : int, default=2048
        Number of posterior samples per batched Dirichlet sampling call.
    gene_mask : jnp.ndarray, shape ``(D,)``, optional
        Boolean mask selecting genes to keep.  Genes marked ``False``
        are aggregated into a single "other" pseudo-gene before
        Dirichlet sampling.  The "other" column is dropped from the
        returned differences, so the output has ``D_kept`` columns.
        If ``None`` (default), all genes are kept.

    Returns
    -------
    jnp.ndarray, shape ``(N_total, D)`` or ``(N_total, D_kept)``
        CLR-space differences.  ``N_total = N * n_samples_dirichlet``.
        When ``gene_mask`` is provided, ``D_kept = gene_mask.sum()``.

    Raises
    ------
    ValueError
        If 3D input is given without the corresponding ``component_*``
        argument, or if ``paired=True`` and sample counts differ.
    """
    # Default RNG key
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    # --- Slice mixture components if needed ---
    r_A = _slice_component(r_samples_A, component_A, "A")
    r_B = _slice_component(r_samples_B, component_B, "B")

    # --- Validate sample counts ---
    N_A, D_A = r_A.shape
    N_B, D_B = r_B.shape

    if D_A != D_B:
        raise ValueError(
            f"Gene dimensions do not match: A has D={D_A}, B has D={D_B}."
        )

    if paired and N_A != N_B:
        raise ValueError(
            f"paired=True requires equal sample counts, "
            f"got N_A={N_A}, N_B={N_B}."
        )

    # Truncate to the shorter if independent (pair by index)
    N = min(N_A, N_B)
    r_A = r_A[:N]
    r_B = r_B[:N]

    # --- Gene aggregation (if requested) ---
    if gene_mask is not None:
        r_A, r_B = _aggregate_genes(r_A, r_B, gene_mask)

    # --- Dirichlet sampling ---
    if paired:
        # Same RNG sub-key per sample index preserves joint structure
        simplex_A, simplex_B = _paired_dirichlet_sample(
            r_A, r_B, n_samples_dirichlet, rng_key, batch_size
        )
    else:
        # Independent RNG keys for each condition
        key_A, key_B = random.split(rng_key)
        simplex_A = _batched_dirichlet(
            r_A, n_samples_dirichlet, key_A, batch_size
        )
        simplex_B = _batched_dirichlet(
            r_B, n_samples_dirichlet, key_B, batch_size
        )

    # --- CLR transform ---
    # CLR(rho) = log(rho) - mean(log(rho))
    clr_A = _clr_transform(simplex_A)
    clr_B = _clr_transform(simplex_B)

    # --- Paired differences ---
    delta = clr_A - clr_B

    # --- Drop the "other" pseudo-gene column ---
    if gene_mask is not None:
        delta = delta[:, :-1]

    return delta


# --------------------------------------------------------------------------
# Empirical DE statistics
# --------------------------------------------------------------------------


def empirical_differential_expression(
    delta_samples: jnp.ndarray,
    tau: float = 0.0,
    gene_names: Optional[List[str]] = None,
) -> dict:
    """Compute gene-level DE statistics from CLR difference samples.

    All statistics are computed by vectorized counting over the ``N``
    posterior samples — no distributional assumptions.  The output dict
    has the same keys as the parametric ``differential_expression()``
    so that downstream code (error control, formatting) works
    interchangeably.

    Parameters
    ----------
    delta_samples : jnp.ndarray, shape ``(N, D)``
        Posterior CLR-space differences
        ``Delta_g^(s) = CLR(rho_A)_g^(s) - CLR(rho_B)_g^(s)``.
    tau : float, default=0.0
        Practical significance threshold (log-scale).
    gene_names : list of str, optional
        Gene names.  If ``None``, generic names are generated.

    Returns
    -------
    dict
        Dictionary with the following keys, each of shape ``(D,)``:

        - **delta_mean** : Posterior mean effect per gene.
        - **delta_sd** : Posterior standard deviation per gene.
        - **prob_positive** : ``P(Delta_g > 0 | data)`` estimated as
          the fraction of positive samples.
        - **prob_effect** : ``P(|Delta_g| > tau | data)`` estimated as
          the fraction of samples exceeding the threshold.
        - **lfsr** : Local false sign rate =
          ``min(P(Delta_g > 0), P(Delta_g < 0))``.
        - **lfsr_tau** : Practical-significance lfsr =
          ``1 - max(P(Delta_g > tau), P(Delta_g < -tau))``.
        - **gene_names** : list of str.

    Notes
    -----
    Resolution is limited by the number of samples ``N``.  The smallest
    non-zero lfsr is ``1/N``.  The standard error of the empirical lfsr
    estimate is ``SE = sqrt(lfsr * (1 - lfsr) / N)``, which is ~0.001
    for lfsr=0.01 with N=10,000.

    All operations are fully vectorized JAX and run on GPU.  Cost is
    ``O(N * D)`` with no additional memory beyond the input.
    """
    # Posterior mean and standard deviation per gene
    delta_mean = jnp.mean(delta_samples, axis=0)
    delta_sd = jnp.std(delta_samples, axis=0, ddof=1)

    # Fraction of samples with positive difference
    prob_positive = jnp.mean(delta_samples > 0, axis=0)

    # Local false sign rate: posterior probability of the minority sign
    lfsr = jnp.minimum(prob_positive, 1.0 - prob_positive)

    # Probability of practical effect: fraction of samples where
    # |Delta_g| > tau
    prob_up = jnp.mean(delta_samples > tau, axis=0)
    prob_down = jnp.mean(delta_samples < -tau, axis=0)
    prob_effect = prob_up + prob_down

    # Practical-significance lfsr (paper definition):
    # lfsr_tau = 1 - max(P(Delta > tau), P(Delta < -tau))
    lfsr_tau = 1.0 - jnp.maximum(prob_up, prob_down)

    # Generate gene names if not provided
    if gene_names is None:
        D = delta_samples.shape[1]
        gene_names = [f"gene_{i}" for i in range(D)]

    return {
        "delta_mean": delta_mean,
        "delta_sd": delta_sd,
        "prob_positive": prob_positive,
        "prob_effect": prob_effect,
        "lfsr": lfsr,
        "lfsr_tau": lfsr_tau,
        "gene_names": gene_names,
    }


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _slice_component(
    r_samples: jnp.ndarray,
    component: Optional[int],
    label: str,
) -> jnp.ndarray:
    """Slice a mixture component from 3D r samples, or validate 2D input.

    Parameters
    ----------
    r_samples : jnp.ndarray, shape ``(N, D)`` or ``(N, K, D)``
        Posterior concentration samples.
    component : int or None
        Component index to extract if 3D.
    label : str
        Label for error messages (``"A"`` or ``"B"``).

    Returns
    -------
    jnp.ndarray, shape ``(N, D)``
        2D concentration samples for a single component.
    """
    if r_samples.ndim == 3:
        if component is None:
            raise ValueError(
                f"r_samples_{label} is 3D (mixture model) but "
                f"component_{label} was not specified."
            )
        # Slice the component: (N, K, D) -> (N, D)
        return r_samples[:, component, :]
    elif r_samples.ndim == 2:
        return r_samples
    else:
        raise ValueError(
            f"r_samples_{label} must be 2D (N, D) or 3D (N, K, D), "
            f"got {r_samples.ndim}D."
        )


# ------------------------------------------------------------------------------


def _clr_transform(simplex_samples: jnp.ndarray) -> jnp.ndarray:
    """Centered log-ratio transform on simplex samples.

    Parameters
    ----------
    simplex_samples : jnp.ndarray, shape ``(N, D)``
        Samples on the D-simplex (rows sum to 1).

    Returns
    -------
    jnp.ndarray, shape ``(N, D)``
        CLR-transformed samples.
    """
    # Guard against exact zeros from Dirichlet sampling
    log_samples = jnp.log(jnp.maximum(simplex_samples, 1e-30))
    # CLR: subtract the geometric mean (= arithmetic mean of logs)
    geometric_mean = jnp.mean(log_samples, axis=-1, keepdims=True)
    return log_samples - geometric_mean


# ------------------------------------------------------------------------------


def _batched_dirichlet(
    r_samples: jnp.ndarray,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
) -> jnp.ndarray:
    """Batched Dirichlet sampling from concentration parameters.

    Reuses the batching strategy from
    ``scribe.core.normalization_logistic._batched_dirichlet_sample_raw``
    but implemented locally to avoid circular imports.

    Parameters
    ----------
    r_samples : jnp.ndarray, shape ``(N, D)``
        Dirichlet concentration parameters.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        Number of posterior samples per batch.

    Returns
    -------
    jnp.ndarray, shape ``(N_total, D)``
        Simplex samples.  ``N_total = N * n_samples_dirichlet``.
    """
    N, D = r_samples.shape
    chunks = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        r_batch = r_samples[start:end]  # (B, D)

        # Deterministic sub-key for this batch
        key_batch = random.fold_in(rng_key, start)

        # Sample from Dirichlet for each row in the batch
        if n_samples_dirichlet == 1:
            # Shape: (B, D)
            samples = jax.random.dirichlet(key_batch, r_batch)
        else:
            # Draw multiple Dirichlet samples per posterior sample
            # We vmap over the batch dimension
            keys = random.split(key_batch, end - start)

            def _sample_one(key, alpha):
                return jax.random.dirichlet(
                    key, alpha, shape=(n_samples_dirichlet,)
                )

            # (B, S, D)
            samples = jax.vmap(_sample_one)(keys, r_batch)
            # Flatten to (B * S, D)
            samples = samples.reshape(-1, D)

        chunks.append(samples)

    result = jnp.concatenate(chunks, axis=0)

    # For n_samples_dirichlet == 1, result is (N, D) — correct.
    # For n_samples_dirichlet > 1, result is (N * S, D) — correct.
    return result


# ------------------------------------------------------------------------------


def _paired_dirichlet_sample(
    r_A: jnp.ndarray,
    r_B: jnp.ndarray,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int,
) -> tuple:
    """Paired Dirichlet sampling for within-mixture comparisons.

    Uses the **same** per-sample RNG sub-key for both conditions,
    ensuring that the joint posterior correlation structure between
    components is preserved.

    Parameters
    ----------
    r_A : jnp.ndarray, shape ``(N, D)``
        Concentration parameters for component A.
    r_B : jnp.ndarray, shape ``(N, D)``
        Concentration parameters for component B.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int
        Number of posterior samples per batch.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        ``(simplex_A, simplex_B)`` each of shape ``(N_total, D)``.
    """
    N, D = r_A.shape
    chunks_A = []
    chunks_B = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        r_batch_A = r_A[start:end]
        r_batch_B = r_B[start:end]
        B = end - start

        # Same base key for this batch — both conditions share it
        key_batch = random.fold_in(rng_key, start)

        if n_samples_dirichlet == 1:
            # Split the shared key into per-sample sub-keys
            keys = random.split(key_batch, B)

            # For each sample, split the per-sample key into two
            # sub-keys (one for A, one for B).  The correlation comes
            # from using the same per-sample seed, NOT from sharing
            # the exact key — the Dirichlet draws themselves are
            # independent given the concentrations.
            def _paired_draw(key, alpha_a, alpha_b):
                k_a, k_b = random.split(key)
                return (
                    jax.random.dirichlet(k_a, alpha_a),
                    jax.random.dirichlet(k_b, alpha_b),
                )

            samples_A, samples_B = jax.vmap(_paired_draw)(
                keys, r_batch_A, r_batch_B
            )
        else:
            keys = random.split(key_batch, B)

            def _paired_draw_multi(key, alpha_a, alpha_b):
                k_a, k_b = random.split(key)
                s_a = jax.random.dirichlet(
                    k_a, alpha_a, shape=(n_samples_dirichlet,)
                )
                s_b = jax.random.dirichlet(
                    k_b, alpha_b, shape=(n_samples_dirichlet,)
                )
                return s_a, s_b

            samples_A, samples_B = jax.vmap(_paired_draw_multi)(
                keys, r_batch_A, r_batch_B
            )
            # (B, S, D) -> (B * S, D)
            samples_A = samples_A.reshape(-1, D)
            samples_B = samples_B.reshape(-1, D)

        chunks_A.append(samples_A)
        chunks_B.append(samples_B)

    return jnp.concatenate(chunks_A, axis=0), jnp.concatenate(chunks_B, axis=0)
