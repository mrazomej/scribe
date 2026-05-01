"""Logistic-Normal Multinomial (LNM) likelihood for compositional count data.

This module implements the observation model that couples a Negative Binomial
prior on per-cell total UMI counts with a Multinomial on gene compositions,
where composition probabilities come from an ALR (Additive Log-Ratio) map
applied to latent Gaussian coordinates produced by a linear-decoder VAE.

The low-rank regime (``d_mode="low_rank"``) uses ``y = mu + W z`` with no
per-coordinate residual noise.  The learned-diagonal regime
(``d_mode="learned"``) adds ``sqrt(d) * epsilon`` with
``epsilon ~ Normal(0, I)``, using a global positive vector ``d`` (per
ALR coordinate).
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
)

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import Likelihood
from ._log_prob import _check_return_by, _normalize_counts, _require_layout
from ....core.axis_layout import AxisLayout
from ...builders.parameter_specs import sample_prior

if TYPE_CHECKING:
    from ...builders.parameter_specs import ParamSpec
    from ...config import ModelConfig

# Minimum epsilon for clamping NB ``probs`` away from 0 and 1 so that NB
# log-prob and sampling stay numerically stable (mirrors ``negative_binomial``).
_P_EPS = 1e-6

# Floor for diagonal ``d`` in ``learned`` mode when mapping to ``sqrt(d)``.
_D_EPS = 1e-8


def select_alr_reference(counts) -> int:
    """Select the gene with highest geometric mean expression as ALR reference.

    Parameters
    ----------
    counts : array_like
        Count matrix of shape ``(n_cells, n_genes)``.

    Returns
    -------
    int
        Index of the gene to use as ALR reference (denominator).

    Notes
    -----
    The geometric mean is computed as the arithmetic mean of ``log(count + 1)``
    per gene across cells, then the argmax picks the gene with largest
    typical log-expression.  This is preferred over the raw UMI sum because it
    is less sensitive to outlier cells.  A stably expressed reference gene
    improves numerical conditioning of the ALR coordinates during VAE training.
    """
    import numpy as np

    log_counts = np.log1p(np.asarray(counts))
    geo_mean = log_counts.mean(axis=0)
    return int(np.argmax(geo_mean))


class LogisticNormalMultinomialLikelihood(Likelihood):
    """
    Logistic-Normal Multinomial likelihood (NB total × multinomial composition).

    Total counts follow ``NegativeBinomialProbs(rₜ, p)``. Given each total,
    gene counts are ``Multinomial(total_count = uₜ, probs = ρ)`` where ``ρ =
    softmax(ALR⁻¹(y))`` and ``y`` is decoded inside the cell plate from the VAE
    (linear decoder output ``y_alr``), optionally with isotropic or diagonal
    Gaussian noise in ALR space.

    Parameters
    ----------
    d_mode : {\"low_rank\", \"learned\"}, default=\"low_rank\"
        ``low_rank``: ``y`` equals the decoder output (singular Gaussian on
        ``y`` induced by low-rank ``W`` is acceptable because the ELBO never
        inverts the full ambient covariance).
        ``learned``: adds ``sqrt(d) * epsilon`` with IID standard normal
        ``epsilon``; KL from ``epsilon`` is zero when the guide matches the
        prior.
    d_param_name : str, default=\"d_lnm\"
        Key in ``param_values`` / decoder merge dict for the learned positive
        diagonal ``d`` (length ``n_genes - 1``), used only when
        ``d_mode=\"learned\"``.
    reference_idx : int, default=-1
        Zero-based index of the reference gene (ALR denominator).  ``-1``
        means the last gene, matching historical behavior.  Non-reference
        genes keep the same order as in ``y``; the zero coordinate is inserted
        at ``reference_idx`` before the softmax.
    """

    def __init__(
        self,
        d_mode: Literal["low_rank", "learned"] = "low_rank",
        d_param_name: str = "d_lnm",
        reference_idx: int = -1,
    ) -> None:
        if d_mode not in ("low_rank", "learned"):
            raise ValueError('d_mode must be "low_rank" or "learned"')
        self._d_mode: Literal["low_rank", "learned"] = d_mode
        self._d_param_name = d_param_name
        self._reference_idx = reference_idx

    @property
    def reference_idx(self) -> int:
        """Index of the ALR reference gene (zero-based)."""
        return self._reference_idx

    def _alr_inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        """Map ALR coordinates to simplex probabilities.

        Insert a zero for the reference gene at position ``self._reference_idx``,
        then softmax. When ``reference_idx == -1`` or equals the last position,
        this reduces to appending a zero (original behavior).

        Parameters
        ----------
        y : jnp.ndarray
            ALR coordinates, shape ``(..., G-1)``, in gene order with the
            reference component omitted.

        Returns
        -------
        jnp.ndarray
            Simplex probabilities ``rho``, shape ``(..., G)``.
        """
        n_alr = y.shape[-1]
        g = n_alr + 1
        ref = self._reference_idx if self._reference_idx >= 0 else g - 1

        zero = jnp.zeros_like(y[..., :1])
        if ref == g - 1:
            y_full = jnp.concatenate([y, zero], axis=-1)
        elif ref == 0:
            y_full = jnp.concatenate([zero, y], axis=-1)
        else:
            y_full = jnp.concatenate(
                [y[..., :ref], zero, y[..., ref:]], axis=-1
            )
        return jax.nn.softmax(y_full, axis=-1)

    def _clip_p_hat(self, p_hat: jnp.ndarray) -> jnp.ndarray:
        """Clamp ``p_hat`` to ``(_P_EPS, 1 - _P_EPS)`` for NB stability."""
        return jnp.clip(p_hat, _P_EPS, 1.0 - _P_EPS)

    def _population_totals_dist(self, r_T: jnp.ndarray, p_hat: jnp.ndarray):
        """Build ``NegativeBinomialProbs`` for total UMI counts."""
        r_T = jnp.asarray(r_T)
        p_hat = jnp.asarray(p_hat)
        p_hat = self._clip_p_hat(p_hat)
        return dist.NegativeBinomialProbs(r_T, p_hat)

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
        total_count_max: Optional[int] = None,
        vae_cell_fn: Optional[
            Callable[[Optional[jnp.ndarray]], Dict[str, jnp.ndarray]]
        ] = None,
        annotation_prior_logits: Optional[jnp.ndarray] = None,
        dataset_indices: Optional[jnp.ndarray] = None,
        param_layouts: Optional[Dict[str, AxisLayout]] = None,
    ) -> None:
        """Draw ``u_T`` and gene counts under the LNM or condition on data.

        Requires ``vae_cell_fn`` so that ``z`` is sampled in the guide and the
        decoder output ``y_alr`` is available inside the cell plate.

        Three plate modes match the other likelihoods: mini-batch SVI
        (``batch_size`` set), full training / inference with observed
        counts, and prior predictive (``counts is None``).

        Notes
        -----
        ``annotation_prior_logits`` and ``dataset_indices`` are accepted for
        API compatibility but are not used yet for the LNM path.

        Raises
        ------
        ValueError
            If ``vae_cell_fn`` is ``None`` or ``y_alr`` is missing after the
            decoder merge.
        KeyError
            If ``param_values`` lacks ``r_T`` or ``p``.
        """
        del annotation_prior_logits, dataset_indices, param_layouts

        if vae_cell_fn is None:
            raise ValueError(
                "LogisticNormalMultinomialLikelihood requires vae_cell_fn "
                "to sample the latent code and produce decoder output "
                "'y_alr' inside the cell plate."
            )

        n_cells = dims["n_cells"]
        n_genes = dims["n_genes"]
        g_minus_1 = n_genes - 1

        r_T = param_values["r_T"]
        p_hat = self._clip_p_hat(param_values["p"])
        nb_totals = self._population_totals_dist(r_T, p_hat)

        # Observed total UMIs per cell (None in prior predictive mode).
        if counts is not None:
            u_T_obs = jnp.sum(counts, axis=-1, dtype=counts.dtype)
        else:
            u_T_obs = None

        # ------------------------------------------------------------------
        # Mini-batch SVI: subsampled cells plate + decoder + composition.
        # ------------------------------------------------------------------
        if batch_size is not None:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # (a) Total-count likelihood: NB on library size.
                u_T_obs_batch = u_T_obs[idx] if u_T_obs is not None else None
                u_T_batch = numpyro.sample("u_T", nb_totals, obs=u_T_obs_batch)

                # (b) Decoder inside the plate triggers VAE latent sampling.
                param_values.update(vae_cell_fn(idx))
                if "y_alr" not in param_values:
                    raise ValueError(
                        "vae_cell_fn must return a 'y_alr' tensor (decoder "
                        "ALR head), shape (batch, G-1)."
                    )
                y_decoded = param_values["y_alr"]

                # Any additional cell-plate priors (non-decoder sites).
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)

                # (c)/(d) Optional diagonal Gaussian noise in ALR space.
                if self._d_mode == "learned":
                    d_vec = param_values[self._d_param_name]
                    d_vec = jnp.maximum(jnp.asarray(d_vec), _D_EPS)
                    sigma = jnp.sqrt(d_vec)
                    eps = numpyro.sample(
                        "lnm_eps",
                        dist.Normal(0.0, 1.0).expand([g_minus_1]).to_event(1),
                    )
                    y = y_decoded + sigma * eps
                else:
                    y = y_decoded

                # (e) Map to simplex probabilities.
                rho = self._alr_inverse(y)

                # (f) Multinomial observation model over genes.
                counts_batch = counts[idx] if counts is not None else None
                numpyro.sample(
                    "counts",
                    dist.Multinomial(
                        total_count=u_T_batch,
                        probs=rho,
                        total_count_max=total_count_max,
                    ),
                    obs=counts_batch,
                )
            return

        # ------------------------------------------------------------------
        # Full cell plate (prior predictive or dense observation).
        # ------------------------------------------------------------------
        with numpyro.plate("cells", n_cells):
            u_T_batch = numpyro.sample("u_T", nb_totals, obs=u_T_obs)

            param_values.update(vae_cell_fn(None))
            if "y_alr" not in param_values:
                raise ValueError(
                    "vae_cell_fn must return a 'y_alr' tensor (decoder ALR "
                    "head), shape (n_cells, G-1)."
                )
            y_decoded = param_values["y_alr"]

            for spec in cell_specs:
                sample_prior(spec, dims, model_config)

            if self._d_mode == "learned":
                d_vec = param_values[self._d_param_name]
                d_vec = jnp.maximum(jnp.asarray(d_vec), _D_EPS)
                sigma = jnp.sqrt(d_vec)
                eps = numpyro.sample(
                    "lnm_eps",
                    dist.Normal(0.0, 1.0).expand([g_minus_1]).to_event(1),
                )
                y = y_decoded + sigma * eps
            else:
                y = y_decoded

            rho = self._alr_inverse(y)

            numpyro.sample(
                "counts",
                dist.Multinomial(
                    total_count=u_T_batch,
                    probs=rho,
                    total_count_max=total_count_max,
                ),
                obs=counts,
            )

    def log_prob(
        self,
        counts: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        param_layouts: Mapping[str, AxisLayout],
        *,
        return_by: str = "cell",
        cells_axis: int = 0,
        r_floor: float = 1e-6,
        p_floor: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
    ) -> jnp.ndarray:
        """Evaluate ``log p(counts, u_T | r_T, p, y_alr)`` up to the decoder.

        Decomposes as the sum of:

        - ``log NB(u_T | r_T, p)`` with ``u_T`` the row sum of ``counts``
        - ``log Multinomial(counts | u_T, rho)`` with ``rho`` the softmax ALR
          inverse of ``y_alr``

        The return value does **not** include the density of the latent ``y``
        under the Gaussian prior—only the conditional on a fixed ``y_alr``
        (e.g. a posterior decoder output).

        Parameters
        ----------
        counts : jnp.ndarray
            Count matrix ``(n_cells, n_genes)`` after ``cells_axis``
            normalization.
        params : Dict[str, jnp.ndarray]
            Must contain ``r_T``, ``p``, and ``y_alr`` (decoder mean / MAP
            ALR coordinates), shapes broadcastable to per-cell totals and
            ``(n_cells, G-1)`` respectively.
        param_layouts : Mapping[str, AxisLayout]
            Semantic layouts for ``r_T``, ``p``, and ``y_alr``.
        return_by : {\"cell\", \"gene\"}, default=\"cell\"
            ``cell``: one log-probability per cell (NB + multinomial).
            ``gene``: not supported (multinomial does not factorize per gene).
        cells_axis : int, default=0
            Axis indexing cells in ``counts`` before evaluation.
        r_floor, p_floor : float, default=1e-6
            Numerical clamps on ``r_T`` and ``p``.
        dtype : jnp.dtype, default=jnp.float32
            Working dtype.

        Returns
        -------
        jnp.ndarray
            If ``return_by=\"cell\"``: shape ``(n_cells,)``.

        Raises
        ------
        NotImplementedError
            If ``return_by='gene'``, or mixture / weighting options are
            requested.
        """
        del weight_type
        _check_return_by(return_by)

        if split_components:
            raise NotImplementedError(
                "LogisticNormalMultinomialLikelihood does not implement "
                "split_components=True."
            )
        if weights is not None:
            raise NotImplementedError(
                "LogisticNormalMultinomialLikelihood does not implement "
                "weighted log_prob."
            )
        if return_by == "gene":
            raise NotImplementedError(
                "return_by='gene' is not defined for the LNM multinomial: "
                "per-gene marginal log-probabilities require integration over "
                "the correlated simplex; use return_by='cell'."
            )

        for key in ("r_T", "p", "y_alr"):
            _require_layout(param_layouts, key, context="LNM.log_prob")

        counts_nm = _normalize_counts(counts, cells_axis, dtype)

        r_T = jnp.clip(jnp.asarray(params["r_T"], dtype=dtype), r_floor, None)
        p_hat = jnp.asarray(params["p"], dtype=dtype)
        p_hat = jnp.clip(p_hat, p_floor, 1.0 - p_floor)

        y_alr = jnp.asarray(params["y_alr"], dtype=dtype)
        rho = self._alr_inverse(y_alr)

        u_T = jnp.sum(counts_nm, axis=-1)

        # Total-count factor: ``NB(u_T | r_T, p)`` with evaluation floors.
        log_nb = dist.NegativeBinomialProbs(r_T, p_hat).log_prob(u_T)

        # Single multinomial call over the gene dimension (event shape G).
        log_mn = dist.Multinomial(total_count=u_T, probs=rho).log_prob(
            counts_nm
        )

        return jnp.asarray(log_nb + log_mn, dtype=dtype)


# ==============================================================================
# LNM with Variable Capture Probability (VCP) on totals
# ==============================================================================


class LNMWithVCPLikelihood(LogisticNormalMultinomialLikelihood):
    """Logistic-Normal Multinomial with per-cell Variable Capture Probability.

    Extends :class:`LogisticNormalMultinomialLikelihood` by adding a per-cell
    ``p_capture`` parameter that modulates the effective success probability
    used in the total-count NB submodel.  The compositional path (ALR-decoded
    ``y_alr`` -> softmax -> Multinomial) is **unchanged**.

    The effective per-cell probability is::

        p_hat^(c) = p * p_capture^(c) / (1 - p * (1 - p_capture^(c)))

    where ``p`` is the population-level success probability and
    ``p_capture^(c) ~ Beta(a, b)`` is a cell-specific capture efficiency.

    Parameters
    ----------
    d_mode : {\"low_rank\", \"learned\"}, default=\"low_rank\"
        Same as the base LNM likelihood.
    d_param_name : str, default=\"d_lnm\"
        Same as the base LNM likelihood.
    reference_idx : int, default=-1
        Same as the base LNM likelihood.
    capture_param_name : str or None, default=None
        Name of the capture site (``\"p_capture\"`` or ``\"phi_capture\"``).
        When ``None``, auto-detected from ``cell_specs`` at sample time.
    """

    def __init__(
        self,
        d_mode: Literal["low_rank", "learned"] = "low_rank",
        d_param_name: str = "d_lnm",
        reference_idx: int = -1,
        capture_param_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            d_mode=d_mode,
            d_param_name=d_param_name,
            reference_idx=reference_idx,
        )
        self._capture_param_name = capture_param_name

    # ------------------------------------------------------------------
    # Capture sampling helpers (reuse from base likelihood module)
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_capture(
        use_phi: bool,
        prior_params: tuple,
    ) -> jnp.ndarray:
        """Sample a per-cell capture parameter from its prior.

        Parameters
        ----------
        use_phi : bool
            ``True`` to sample ``phi_capture ~ BetaPrime``; ``False`` to
            sample ``p_capture ~ Beta``.
        prior_params : tuple of float
            ``(alpha, beta)`` for the Beta or BetaPrime prior.

        Returns
        -------
        jnp.ndarray
            Scalar per-cell capture value.
        """
        from .base import (
            _sample_p_capture_constrained,
            _sample_phi_capture_constrained,
        )

        if use_phi:
            return _sample_phi_capture_constrained(prior_params)
        return _sample_p_capture_constrained(prior_params)

    def _compute_cell_p_hat(
        self,
        p: jnp.ndarray,
        capture_value: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply the VCP transform to get per-cell p_hat.

        Parameters
        ----------
        p : jnp.ndarray
            Population success probability (scalar or broadcastable).
        capture_value : jnp.ndarray
            Per-cell capture probability (p_capture convention, in [0, 1]).

        Returns
        -------
        jnp.ndarray
            Per-cell effective ``p_hat``, clipped for NB stability.
        """
        p_hat = p * capture_value / (1.0 - p * (1.0 - capture_value))
        return jnp.clip(p_hat, _P_EPS, 1.0 - _P_EPS)

    # ------------------------------------------------------------------
    # Main sampling method
    # ------------------------------------------------------------------

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
        total_count_max: Optional[int] = None,
        vae_cell_fn: Optional[
            Callable[[Optional[jnp.ndarray]], Dict[str, jnp.ndarray]]
        ] = None,
        annotation_prior_logits: Optional[jnp.ndarray] = None,
        dataset_indices: Optional[jnp.ndarray] = None,
        param_layouts: Optional[Dict[str, AxisLayout]] = None,
    ) -> None:
        """Draw ``u_T`` with per-cell VCP and gene counts under the LNM.

        Same structure as the base LNM ``sample()``, but the total-count NB
        uses a per-cell ``p_hat`` derived from ``p`` and ``p_capture``.

        The capture parameter is sampled inside the cell plate, exactly as
        in the standard VCP likelihoods.
        """
        del annotation_prior_logits, dataset_indices, param_layouts

        if vae_cell_fn is None:
            raise ValueError(
                "LNMWithVCPLikelihood requires vae_cell_fn to sample the "
                "latent code and produce decoder output 'y_alr' inside "
                "the cell plate."
            )

        n_cells = dims["n_cells"]
        n_genes = dims["n_genes"]
        g_minus_1 = n_genes - 1

        # Population-level NB parameters (p is pre-capture).
        r_T = param_values["r_T"]
        p_pop = self._clip_p_hat(param_values["p"])

        # Resolve capture parameterization.
        if self._capture_param_name is not None:
            use_phi = self._capture_param_name == "phi_capture"
        else:
            use_phi = any(spec.name == "phi_capture" for spec in cell_specs)

        # Resolve capture prior (alpha, beta) from model config param specs.
        capture_prior_params = (1.0, 1.0)
        target_name = "phi_capture" if use_phi else "p_capture"
        for pspec in model_config.param_specs:
            if pspec.name == target_name and pspec.prior is not None:
                capture_prior_params = pspec.prior
                break

        # Observed total UMIs per cell (None in prior predictive mode).
        u_T_obs = (
            jnp.sum(counts, axis=-1, dtype=counts.dtype)
            if counts is not None
            else None
        )

        def _cell_body(idx, obs_counts):
            """Shared logic for each cell-plate variant."""
            # (a) Per-cell capture.
            if target_name in param_values:
                capture_value = param_values[target_name]
                if idx is not None:
                    capture_value = capture_value[idx]
            else:
                capture_value = self._sample_capture(
                    use_phi, capture_prior_params
                )

            # Convert phi_capture to p_capture convention for the VCP formula.
            if use_phi:
                p_capture = 1.0 / (1.0 + capture_value)
            else:
                p_capture = capture_value

            # (b) Per-cell effective p_hat and total-count NB.
            p_hat_cell = self._compute_cell_p_hat(p_pop, p_capture)
            nb_cell = dist.NegativeBinomialProbs(r_T, p_hat_cell)
            u_T_obs_local = (
                u_T_obs[idx]
                if (u_T_obs is not None and idx is not None)
                else u_T_obs
            )
            u_T_val = numpyro.sample("u_T", nb_cell, obs=u_T_obs_local)

            # (c) Decoder inside the plate triggers VAE latent sampling.
            param_values.update(vae_cell_fn(idx))
            if "y_alr" not in param_values:
                raise ValueError(
                    "vae_cell_fn must return a 'y_alr' tensor (decoder "
                    "ALR head)."
                )
            y_decoded = param_values["y_alr"]

            # Additional cell-plate prior sites (non-decoder), skipping
            # the capture spec which we already sampled above.
            _capture_names = {"p_capture", "phi_capture"}
            for spec in cell_specs:
                if spec.name in _capture_names:
                    continue
                sample_prior(spec, dims, model_config)

            # (d) Optional diagonal Gaussian noise in ALR space.
            if self._d_mode == "learned":
                d_vec = jnp.maximum(
                    jnp.asarray(param_values[self._d_param_name]), _D_EPS
                )
                eps = numpyro.sample(
                    "lnm_eps",
                    dist.Normal(0.0, 1.0).expand([g_minus_1]).to_event(1),
                )
                y = y_decoded + jnp.sqrt(d_vec) * eps
            else:
                y = y_decoded

            # (e) Simplex probabilities.
            rho = self._alr_inverse(y)

            # (f) Multinomial observation model.
            numpyro.sample(
                "counts",
                dist.Multinomial(
                    total_count=u_T_val,
                    probs=rho,
                    total_count_max=total_count_max,
                ),
                obs=obs_counts,
            )

        # ------------------------------------------------------------------
        # Mini-batch SVI: subsampled cells plate.
        # ------------------------------------------------------------------
        if batch_size is not None:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                obs_batch = counts[idx] if counts is not None else None
                _cell_body(idx, obs_batch)
            return

        # ------------------------------------------------------------------
        # Full cell plate (prior predictive or dense observation).
        # ------------------------------------------------------------------
        with numpyro.plate("cells", n_cells):
            _cell_body(None, counts)
