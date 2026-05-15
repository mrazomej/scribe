"""Custom probability distributions for SCRIBE."""

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax, scipy as jsp
from jax.scipy.special import betaln, gammaln

from numpyro.distributions import (
    Distribution,
    constraints,
    Gamma,
    LowRankMultivariateNormal,
)
from numpyro.distributions.continuous import Beta
from numpyro.distributions.conjugate import NegativeBinomialProbs
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key

# ==============================================================================
# Beta Prime Distribution
# ==============================================================================


class BetaPrime(Distribution):
    """
    Beta Prime distribution (odds-of-Beta convention).

    Convention
    ----------
    If p ~ Beta(α, β) and φ = (1 - p) / p (odds of "success" 1 - p), then
    φ ~ BetaPrime(α, β) in THIS CLASS.

    Implementation detail
    ---------------------
    Mathematically, φ has the *standard* Beta-prime with swapped parameters:
        φ ~ BetaPrime_std(β, α).
    This class accepts (α, β) at the call site and internally uses (β, α),
    so that your models can pass (α, β) unchanged. This keeps the odds
    transform ``φ = (1 - p) / p`` aligned with the model's ``p`` parameter
    usage in SCRIBE.

    Density (with user parameters α, β)
    -----------------------------------
        f(φ; α, β) = φ^(β - 1) * (1 + φ)^(-(α + β)) / B(β, α),    φ > 0

    Note the Beta function arguments B(β, α).

    Parameters
    ----------
    concentration1 : jnp.ndarray
        α (matches the Beta prior's first shape)
    concentration0 : jnp.ndarray
        β (matches the Beta prior's second shape)
    """

    arg_constraints = {
        "concentration1": constraints.positive,  # α
        "concentration0": constraints.positive,  # β
    }
    support = constraints.positive
    has_rsample = False

    def __init__(self, concentration1, concentration0, validate_args=None):
        alpha = jnp.asarray(concentration1, dtype=jnp.float32)  # α
        beta = jnp.asarray(concentration0, dtype=jnp.float32)  # β
        # store user-facing α, β
        self.alpha, self.beta = promote_shapes(alpha, beta)
        # internal standard Beta-prime uses (a_std, b_std) = (β, α)
        self._a_std = self.beta
        self._b_std = self.alpha
        super().__init__(
            batch_shape=self.alpha.shape,
            event_shape=(),
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        key1, key2 = random.split(key)
        # φ = X / Y with X ~ Gamma(β, 1), Y ~ Gamma(α, 1)
        x = Gamma(self._a_std, 1.0).sample(key1, sample_shape)
        y = Gamma(self._b_std, 1.0).sample(key2, sample_shape)
        # Clamp to prevent 0/0 = NaN when shape params are very small
        # (Gamma with tiny shape can underflow to exactly 0.0 in float32)
        x = jnp.clip(x, 1e-30)
        y = jnp.clip(y, 1e-30)
        return x / y

    @validate_sample
    def log_prob(self, value):
        # log f(φ; α, β) = (β-1) log φ - (α+β) log(1+φ) - log B(β, α)
        # Clamp value to prevent log(0) = -inf when samples are near zero
        safe_value = jnp.clip(value, 1e-30)
        log_num = (self.beta - 1) * jnp.log(safe_value) - (
            self.alpha + self.beta
        ) * jnp.log1p(value)
        log_den = (
            jsp.special.gammaln(self._a_std)
            + jsp.special.gammaln(self._b_std)
            - jsp.special.gammaln(self._a_std + self._b_std)
        )
        return log_num - log_den

    @property
    def mean(self):
        # E[φ] = β / (α - 1), defined for α > 1
        return jnp.where(self.alpha > 1, self.beta / (self.alpha - 1), jnp.inf)

    @property
    def variance(self):
        # Var[φ] = β(β + α - 1) / [(α - 1)^2 (α - 2)], defined for α > 2
        return jnp.where(
            self.alpha > 2,
            self.beta
            * (self.beta + self.alpha - 1)
            / ((self.alpha - 1) ** 2 * (self.alpha - 2)),
            jnp.inf,
        )

    @property
    def mode(self):
        # mode = (β - 1) / (α + 1) for β >= 1; else 0
        return jnp.where(
            self.beta >= 1, (self.beta - 1) / (self.alpha + 1), 0.0
        )

    @property
    def concentration1(self):
        """Access to concentration1 parameter (α) for NumPyro compatibility."""
        return self.alpha

    @property
    def concentration0(self):
        """Access to concentration0 parameter (β) for NumPyro compatibility."""
        return self.beta


# ==============================================================================
# Beta Negative Binomial Distribution
# ==============================================================================


class BetaNegativeBinomial(Distribution):
    """Beta Negative Binomial compound distribution.

    Arises by marginalising the success probability of a Negative
    Binomial over a Beta prior:

        p ~ Beta(concentration1, concentration0)
        X | p ~ NegativeBinomial(n, p)

    This is a local fallback for numpyro < 0.20 which added the
    distribution natively.

    Optimisation over the upstream NumPyro version: ``log_prob``
    expands ``betaln`` manually and evaluates ``gammaln`` on the
    original (un-promoted) gene-only arrays ``concentration0`` and
    ``n`` *before* broadcasting.  For typical SCRIBE shapes
    (alpha = C x G, kappa = G, r = G) this avoids ~33 % of
    ``gammaln`` FLOPs.

    Parameters
    ----------
    concentration1 : array_like
        First Beta parameter (alpha).  May be cell x gene shaped.
    concentration0 : array_like
        Second Beta parameter (kappa).  Typically gene-only.
    n : array_like
        Number of successes for the Negative Binomial (r).
        Typically gene-only.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_negative_binomial_distribution
    .. [2] https://num.pyro.ai/en/stable/_modules/numpyro/distributions/conjugate.html#BetaNegativeBinomial
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "n": constraints.positive,
    }
    support = constraints.nonnegative_integer
    pytree_data_fields = (
        "concentration1",
        "concentration0",
        "n",
        "_beta",
        "_c0_raw",
        "_n_raw",
    )

    def __init__(
        self,
        concentration1,
        concentration0,
        n,
        *,
        validate_args=None,
    ):
        # Keep the original (possibly lower-rank) arrays so that
        # log_prob can compute gammaln at the smaller shape.
        self._c0_raw = jnp.asarray(concentration0)
        self._n_raw = jnp.asarray(n)

        self.concentration1, self.concentration0, self.n = promote_shapes(
            concentration1, concentration0, n
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1),
            jnp.shape(concentration0),
            jnp.shape(n),
        )
        concentration1 = jnp.broadcast_to(concentration1, batch_shape)
        concentration0 = jnp.broadcast_to(concentration0, batch_shape)
        self._beta = Beta(concentration1, concentration0)
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """Two-step compound sampling: Beta then NegativeBinomial."""
        assert is_prng_key(key)
        key_beta, key_nb = random.split(key)
        probs = self._beta.sample(key_beta, sample_shape)
        return NegativeBinomialProbs(total_count=self.n, probs=probs).sample(
            key_nb
        )

    @validate_sample
    def log_prob(self, value):
        """Log PMF with optimised gammaln evaluation.

        Expands betaln manually so that the three gene-only terms
        (gammaln(kappa), gammaln(r), gammaln(kappa + r)) are computed
        at their native (G,) shape rather than the broadcast (C, G).
        JAX broadcasting handles the addition with the (C, G) terms.
        """
        c1 = self.concentration1  # alpha — (C, G) or full batch
        c0 = self._c0_raw  # kappa — original shape, e.g. (G,)
        n = self._n_raw  # r     — original shape, e.g. (G,)

        # Gene-only terms: 3 gammaln at the smaller (G,) shape.
        gene_terms = gammaln(c0 + n) - gammaln(n) - gammaln(c0)

        # Cell x gene terms: 6 gammaln at full (C, G) shape.
        # n and c0 broadcast automatically inside the additions.
        cell_terms = (
            gammaln(n + value)
            - gammaln(value + 1)
            + gammaln(c1 + value)
            - gammaln(c1 + value + c0 + n)
            - gammaln(c1)
            + gammaln(c1 + c0)
        )

        return gene_terms + cell_terms

    @property
    def mean(self):
        """E[X] = n * alpha / (beta - 1) when beta > 1."""
        return jnp.where(
            self.concentration0 > 1,
            self.n * self.concentration1 / (self.concentration0 - 1),
            jnp.inf,
        )

    @property
    def variance(self):
        """Var[X] when beta > 2; infinite otherwise."""
        alpha = self.concentration1
        beta = self.concentration0
        n = self.n
        var = (
            n
            * alpha
            * (n + beta - 1)
            * (alpha + beta - 1)
            / (jnp.square(beta - 1) * (beta - 2))
        )
        return jnp.where(beta > 2, var, jnp.inf)


# ==============================================================================
# Log-Mean Negative Binomial
# ==============================================================================


class LogMeanNegativeBinomial(Distribution):
    """Negative-Binomial parameterized by ``(log_mean, concentration)``.

    This parameterization is the natural fit for the NB-LogNormal model,
    where the decoder produces a log-mean ``y_log_rate = log(mu * exp(z))``
    and the gene dispersion ``r`` is a separate global. Compared to
    ``NegativeBinomial2(mean=exp(y), concentration=r)``, this class:

    1. Avoids materializing ``mean = exp(log_mean)`` in the log-prob path.
       Computing log-prob entirely in log-space sidesteps float32 overflow
       that would otherwise force a model-changing clamp on the log-rate.
    2. Matches the failure-logit linear-in-``z`` derivation in
       ``paper/_nb_lognormal.qmd`` directly: the success-logit (NumPyro
       convention) is ``log(r) - log_mean``, so the posterior gradient
       and curvature derivations there carry over to the implementation
       without an extra exp/log round-trip.

    The PMF in the canonical (mean, dispersion) form is

        Var[Y] = ⟨Y⟩ + ⟨Y⟩^2 / r,


    and the mean and variance properties below evaluate ``exp(log_mean)``
    explicitly because they are not on the gradient path.

    Parameters
    ----------
    log_mean : array_like
        Logarithm of the NB mean (broadcasts with ``concentration``).
        Real-valued; no positivity constraint.
    concentration : array_like
        Dispersion parameter ``r > 0`` (NumPyro's "concentration").
        Larger values approach the Poisson limit; smaller values give
        heavier-tailed counts with a stronger zero-spike.

    Notes
    -----
    The log-prob uses the identity

        log NB(k; log_mean, r)
            = lgamma(k+r) - lgamma(k+1) - lgamma(r)
              + k * delta - (k + r) * softplus(delta),

    where ``delta = log_mean - log(r)``.  This formulation never instantiates
    ``mean`` or ``r/(r+mean)`` and is stable for any finite ``log_mean`` and
    positive ``r``.
    """

    arg_constraints = {
        "log_mean": constraints.real,
        "concentration": constraints.positive,
    }
    support = constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, log_mean, concentration, validate_args=None):
        log_mean = jnp.asarray(log_mean, dtype=jnp.float32)
        concentration = jnp.asarray(concentration, dtype=jnp.float32)
        self.log_mean, self.concentration = promote_shapes(
            log_mean, concentration
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(log_mean), jnp.shape(concentration)
        )
        super().__init__(
            batch_shape=batch_shape,
            event_shape=(),
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        """Sample via the Gamma-Poisson decomposition.

        ``Y | lambda ~ Poisson(lambda)`` with ``lambda = exp(log_mean) *
        Gamma(r, 1) / r`` is equivalent to the standard NB sampler. The
        only ``exp`` we evaluate is ``exp(log_mean - log(r))``, which we
        clamp for float32 safety.
        """
        if not is_prng_key(key):
            raise ValueError(f"key must be a JAX PRNGKey, got {type(key)}")
        key_g, key_p = random.split(key)
        # delta = log_mean - log(r) ; mean = r * exp(delta)
        delta = self.log_mean - jnp.log(self.concentration)
        # Gamma(r, 1) sample
        gamma_sample = Gamma(self.concentration, 1.0).sample(
            key_g, sample_shape
        )
        # rate = mean * gamma_sample / r = exp(delta) * gamma_sample
        # Clamp delta to a wide-but-finite range. This is *not* a model
        # modification: it bounds non-finite inputs, never hit in valid
        # inference, while keeping float32 arithmetic well-defined.
        delta_clipped = jnp.clip(delta, -50.0, 50.0)
        rate = jnp.exp(delta_clipped) * gamma_sample
        return jax.random.poisson(key_p, rate)

    @validate_sample
    def log_prob(self, value):
        """Stable log-prob in pure log-space (no ``exp(log_mean)``)."""
        k = jnp.asarray(value, dtype=jnp.float32)
        r = self.concentration
        delta = self.log_mean - jnp.log(r)
        sp = jax.nn.softplus(delta)
        return (
            gammaln(k + r)
            - gammaln(k + 1.0)
            - gammaln(r)
            + k * delta
            - (k + r) * sp
        )

    @property
    def mean(self):
        return jnp.exp(self.log_mean)

    @property
    def variance(self):
        m = jnp.exp(self.log_mean)
        return m + jnp.square(m) / self.concentration

    @property
    def mode(self):
        # Standard NB mode in mean parameterization:
        #   mode = floor((r - 1) * mean / r)  for r >= 1, else 0.
        m = jnp.exp(self.log_mean)
        r = self.concentration
        return jnp.where(
            r >= 1.0, jnp.floor((r - 1.0) * m / r), jnp.zeros_like(m)
        )


# ==============================================================================
# Low-Rank Compositional Distributions
# ==============================================================================


class LowRankLogisticNormal(Distribution):
    """
    Low-rank Logistic-Normal distribution for compositional data.

    This distribution models D-dimensional probability vectors (on the simplex)
    using a (D-1)-dimensional low-rank multivariate normal distribution in
    log-ratio space. It uses the Additive Log-Ratio (ALR) transformation.

    Mathematical Definition
    -----------------------
    Let y ∈ ℝ^(D-1) ~ MVN(μ, Σ) where Σ = WW^T + diag(D) is low-rank.
    The ALR transformation maps y to the simplex Δ^D:

        xᵢ = exp(yᵢ) / (1 + Σⱼ exp(yⱼ))   for i = 1, ..., D-1
        x_D = 1 / (1 + Σⱼ exp(yⱼ))

    The inverse transformation (simplex to log-ratio space) is:

        yᵢ = log(xᵢ / x_D)   for i = 1, ..., D-1

    The Jacobian of the transformation has log-determinant:

        log|det(J)| = -Σᵢ₌₁^D log(xᵢ)

    Low-Rank Covariance Structure
    ------------------------------
    The covariance matrix has the form:

        Σ = WW^T + diag(D)

    where:
    - W is a (D-1) × rank factor matrix
    - D is a (D-1) diagonal vector
    - Memory: O((D-1) × rank) vs O((D-1)²) for full covariance

    This is critical for large D (e.g., 30K+ genes) where storing full
    covariance is prohibitive.

    Asymmetry and Reference Component
    ----------------------------------
    The ALR transformation uses one simplex component as the reference
    (denominator in log-ratios).  By default (``reference_idx=-1``) this is
    the last component.  A non-default index places that component at the
    chosen axis position without reordering latent rows (the Gaussian always
    has dimension ``D-1``).  The distribution is not symmetric under
    arbitrary relabeling unless you use ``SoftmaxNormal`` (which cannot
    compute ``log_prob``).

    Parameters
    ----------
    loc : jnp.ndarray
        Location parameter μ ∈ ℝ^(D-1) (mean in log-ratio space)
    cov_factor : jnp.ndarray
        Low-rank factor matrix W of shape (D-1, rank)
    cov_diag : jnp.ndarray
        Diagonal component D of shape (D-1,)
    reference_idx : int, default=-1
        Zero-based index of the simplex component used as the ALR reference.
        ``-1`` means the last component (legacy default).
    validate_args : bool, optional
        Whether to validate input arguments

    Examples
    --------
    >>> from jax import random
    >>> import jax.numpy as jnp
    >>> # Create a low-rank logistic-normal for 5-dimensional simplex
    >>> D = 5
    >>> rank = 2
    >>> loc = jnp.zeros(D - 1)
    >>> cov_factor = jnp.ones((D - 1, rank)) * 0.1
    >>> cov_diag = jnp.ones(D - 1) * 0.5
    >>> dist = LowRankLogisticNormal(loc, cov_factor, cov_diag)
    >>> # Sample from the distribution (returns D-dimensional simplex points)
    >>> samples = dist.sample(random.PRNGKey(0), (100,))
    >>> samples.shape
    (100, 5)
    >>> # Samples sum to 1
    >>> jnp.allclose(samples.sum(axis=-1), 1.0)
    True
    >>> # Evaluate log probability
    >>> log_p = dist.log_prob(samples[0])

    References
    ----------
    Aitchison, J., & Shen, S. M. (1980). Logistic-normal distributions: Some
    properties and uses. Biometrika, 67(2), 261-272.

    Aitchison, J. (1986). The Statistical Analysis of Compositional Data.
    Chapman & Hall.

    See Also
    --------
    SoftmaxNormal : Symmetric alternative using softmax (no log_prob available)
    """

    # Define constraints
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.independent(constraints.real, 2),
        "cov_diag": constraints.positive,
    }
    support = constraints.simplex
    has_rsample = False

    def __init__(
        self, loc, cov_factor, cov_diag, reference_idx=-1, validate_args=None
    ):
        loc = jnp.asarray(loc)
        cov_factor = jnp.asarray(cov_factor)
        cov_diag = jnp.asarray(cov_diag)

        # Store parameters
        self.loc = loc
        self.cov_factor = cov_factor
        self.cov_diag = cov_diag
        self._reference_idx = int(reference_idx)

        # Infer dimensions
        self.n_dims_unconstrained = loc.shape[-1]  # D - 1
        self.n_dims_simplex = self.n_dims_unconstrained + 1  # D

        # Create internal low-rank multivariate normal
        self.base_dist = LowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
        )

        super().__init__(
            batch_shape=self.base_dist.batch_shape,
            event_shape=(self.n_dims_simplex,),
            validate_args=validate_args,
        )

    @property
    def reference_idx(self) -> int:
        """Index of the ALR reference component on the simplex (zero-based)."""
        return self._reference_idx

    # --------------------------------------------------------------------------

    def sample(self, key, sample_shape=()):
        """
        Sample from the distribution.

        Returns samples on the D-dimensional simplex.

        Parameters
        ----------
        key : random.PRNGKey
            JAX random key
        sample_shape : tuple, optional
            Shape of samples to draw

        Returns
        -------
        jnp.ndarray
            Samples of shape (*sample_shape, *batch_shape, D) on the simplex
        """
        # Sample from base MVN in log-ratio space
        y = self.base_dist.sample(key, sample_shape)  # (..., D-1)

        # Map ALR latent y to simplex probabilities (reference index controls layout)
        exp_y = jnp.exp(y)
        g = self.n_dims_simplex
        ref = self._reference_idx if self._reference_idx >= 0 else g - 1

        sum_exp = 1.0 + jnp.sum(exp_y, axis=-1, keepdims=True)
        x_alr = exp_y / sum_exp
        x_ref = 1.0 / sum_exp

        if ref == g - 1:
            return jnp.concatenate([x_alr, x_ref], axis=-1)
        if ref == 0:
            return jnp.concatenate([x_ref, x_alr], axis=-1)
        return jnp.concatenate(
            [x_alr[..., :ref], x_ref, x_alr[..., ref:]], axis=-1
        )

    # --------------------------------------------------------------------------

    @validate_sample
    def log_prob(self, value):
        """
        Evaluate log probability density.

        Parameters
        ----------
        value : jnp.ndarray
            Points on the simplex of shape (..., D)

        Returns
        -------
        jnp.ndarray
            Log probability density of shape (...)
        """
        # Inverse ALR: extract reference and non-reference simplex components
        g = self.n_dims_simplex
        ref = self._reference_idx if self._reference_idx >= 0 else g - 1

        x_ref = value[..., ref : ref + 1]
        if ref == g - 1:
            x_nonref = value[..., :-1]
        elif ref == 0:
            x_nonref = value[..., 1:]
        else:
            x_nonref = jnp.concatenate(
                [value[..., :ref], value[..., ref + 1 :]], axis=-1
            )

        y = jnp.log(x_nonref + 1e-20) - jnp.log(x_ref + 1e-20)

        # Log probability in unconstrained space
        log_prob_unconstrained = self.base_dist.log_prob(y)  # (...)

        # Jacobian correction: log|det(J)| = -Σᵢ log(xᵢ)
        log_jacob = -jnp.sum(jnp.log(value + 1e-20), axis=-1)  # (...)

        return log_prob_unconstrained + log_jacob

    # --------------------------------------------------------------------------

    @property
    def mean(self):
        """
        Mean of the distribution on the simplex.

        Note: This is an approximation computed via Monte Carlo sampling,
        as the true mean does not have a closed form.

        Returns
        -------
        jnp.ndarray
            Approximate mean of shape (*batch_shape, D)
        """
        # Approximate mean via sampling
        key = random.PRNGKey(0)
        samples = self.sample(key, (10000,))
        return jnp.mean(samples, axis=0)


# ==============================================================================
# Poisson-LogNormal Distribution (PLN)
# ==============================================================================


class LowRankPoissonLogNormal(Distribution):
    """
    Low-rank Poisson-LogNormal distribution for count data.

    This distribution models G-dimensional count vectors where each gene's
    count is drawn from a Poisson whose log-rate follows a multivariate
    normal with low-rank-plus-diagonal covariance structure.

    Mathematical Definition
    -----------------------
    Let x in R^G ~ MVN(mu, Sigma) where Sigma = WW^T + diag(d) is low-rank.
    Each gene's count is an independent Poisson draw given x:

        u_g | x_g  ~  Poisson(exp(x_g)),   g = 1, ..., G

    The marginal over x yields correlated counts despite per-gene
    conditional independence.

    Low-Rank Covariance Structure
    -----------------------------
    The covariance matrix has the form:

        Sigma = W @ W^T + diag(d)

    where:
        W: (G, k) factor loading matrix
        d: (G,) diagonal (positive) vector

    Relationship to LNM
    --------------------
    While LowRankLogisticNormal (LNM) models *compositions* on the simplex
    via ALR transforms, LowRankPoissonLogNormal models *absolute counts*
    directly.  Total counts are **not** a separate parameter -- they emerge
    as the sum of per-gene Poisson draws, naturally coupling composition
    and total through the shared covariance.

    Parameters
    ----------
    loc : jnp.ndarray
        Mean of the latent Gaussian in log-rate space, shape ``(G,)``.
    cov_factor : jnp.ndarray
        Low-rank factor ``W``, shape ``(G, k)``.
    cov_diag : jnp.ndarray
        Diagonal entries ``d``, shape ``(G,)``.  Must be positive.

    See Also
    --------
    LowRankLogisticNormal : Compositional analogue for LNM models.
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.independent(constraints.real, 2),
        "cov_diag": constraints.positive,
    }
    # Count support: non-negative integers for each gene.
    support = constraints.independent(constraints.nonnegative_integer, 1)
    has_rsample = False

    def __init__(
        self,
        loc,
        cov_factor,
        cov_diag,
        validate_args=None,
    ):
        loc = jnp.asarray(loc)
        cov_factor = jnp.asarray(cov_factor)
        cov_diag = jnp.asarray(cov_diag)

        self.loc = loc
        self.cov_factor = cov_factor
        self.cov_diag = cov_diag
        self.n_genes = loc.shape[-1]

        # Internal low-rank MVN for sampling latent log-rates.
        self.base_dist = LowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
        )

        super().__init__(
            batch_shape=self.base_dist.batch_shape,
            event_shape=(self.n_genes,),
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        """Draw count vectors by sampling log-rates then Poisson counts.

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key for sampling.
        sample_shape : tuple
            Shape prefix for batched draws.

        Returns
        -------
        jnp.ndarray
            Integer count array of shape ``(*sample_shape, *batch_shape, G)``.
        """
        k1, k2 = random.split(key)

        # Sample latent log-rates from the low-rank MVN.
        log_rates = self.base_dist.sample(k1, sample_shape)

        # Clamp to prevent float32 overflow in exp (exp(88) ~ 2.4e38).
        log_rates = jnp.clip(log_rates, -30.0, 30.0)
        rates = jnp.exp(log_rates)

        # Draw per-gene Poisson counts.
        return random.poisson(k2, rates)

    def log_prob(self, value):
        """Marginal log-probability of a count vector — not closed form.

        The marginal ``log p(u) = log ∫ p(u|x) p(x) dx`` has no closed
        form for the Poisson-LogNormal: the integrand is
        ``Poisson(exp(x))`` against a multivariate Normal over ``x``,
        and the ``exp(x)`` inside the Poisson rate kills any
        analytical reduction.

        **This is not a barrier to inference.** NumPyro never
        evaluates this method during SVI / MCMC, because PLN is
        specified in the model trace as the *factorization*

            x      ~  MVN(mu, Sigma)             # closed-form log_prob
            u | x  ~  Poisson(exp(x))            # closed-form log_prob

        and the ELBO needs only the *joint* ``log p(x, u)``, not the
        marginal ``log p(u)``. Each factor's ``log_prob`` is closed
        form, so the inference machinery has everything it needs
        without ever touching this method.

        The compound ``LowRankPoissonLogNormal.log_prob`` would only
        be needed for held-out marginal-likelihood evaluation, marginal
        Bayes factors, or PSIS-LOO — all post-fit operations. For
        those use cases see :meth:`importance_log_prob`, which returns
        an importance-sampled (IWAE-style) estimate.

        Raises
        ------
        NotImplementedError
            Always. The marginal has no closed form. Use
            :meth:`importance_log_prob` for an MC estimate or, if a
            looser bound is acceptable, the ELBO computed via
            NumPyro's standard machinery.
        """
        raise NotImplementedError(
            "The marginal log p(u) for Poisson-LogNormal has no "
            "closed form (it requires integrating Poisson(exp(x)) "
            "against a multivariate Normal over x). This does NOT "
            "block inference: NumPyro evaluates the joint p(x, u) "
            "via the model trace, never this method. For post-fit "
            "marginal-likelihood evaluation, use "
            "`importance_log_prob(value, q_dist, n_samples=...)`."
        )

    def importance_log_prob(self, value, q_dist, *, n_samples=64, rng_key=None):
        """Importance-sampled estimate of the marginal ``log p(u)``.

        Returns the IWAE-style estimator

        .. math::
            \\log\\hat p(u) \\;=\\;
              \\operatorname{logsumexp}_{s=1}^{S}\\,
              \\bigl[\\log p(u\\mid x_s) + \\log p(x_s) - \\log q(x_s\\mid u)\\bigr]
              \\;-\\;\\log S

        where ``x_s ~ q(x | u)``.  This is a valid lower bound on
        ``log p(u)``; it is tighter than the ELBO and becomes exact in
        the limit ``q → p(x | u)``.  In practice ``q`` will be the
        amortized posterior produced by the encoder.

        This method is **not** used during training — it is a
        post-fit utility for held-out marginal log-likelihood,
        model comparison (Bayes factors, log-Bayes factors), and
        PSIS-LOO-style cross-validation diagnostics.

        Parameters
        ----------
        value : jnp.ndarray
            Observed counts, shape ``(*batch_shape, G)``.
        q_dist : numpyro.distributions.Distribution
            Variational posterior ``q(x | u)`` whose ``event_shape``
            matches the latent dimension ``G`` and whose
            ``batch_shape`` is broadcast-compatible with ``value``.
            For a SCRIBE PLN fit, this is typically constructed from
            the encoder's amortized parameters at the cell of
            interest.
        n_samples : int, default=64
            Number of importance samples ``S``.  Larger ``S`` gives a
            tighter IWAE bound at linear extra cost.  64 is a sensible
            default for diagnostic use; 1024+ for definitive model
            comparison numbers.
        rng_key : jax.random.PRNGKey
            PRNG key for the importance samples.  Required (no
            implicit default — reproducibility matters for any number
            you publish).

        Returns
        -------
        jnp.ndarray
            Estimated ``log p(u)`` of shape ``(*batch_shape,)``.

        Raises
        ------
        ValueError
            If ``rng_key`` is not supplied.
        """
        if rng_key is None:
            raise ValueError(
                "importance_log_prob requires an explicit rng_key for "
                "reproducible importance sampling. Pass "
                "`rng_key=jax.random.PRNGKey(seed)`."
            )

        # Draw S samples from q in shape (S, *batch_shape, G).
        x_samples = q_dist.sample(rng_key, sample_shape=(n_samples,))

        # Clamp log-rates the same way ``sample`` does so that ``exp``
        # cannot overflow float32 inside the Poisson log-pmf
        # evaluation (exp(88) ~ 2.4e38).
        x_clip = jnp.clip(x_samples, -30.0, 30.0)
        rates = jnp.exp(x_clip)

        # log p(u | x_s) factorizes across genes:
        # Poisson log-pmf  =  u * log(rate) - rate - lgamma(u + 1).
        value_b = jnp.broadcast_to(value, x_samples.shape)
        log_p_u_given_x = jnp.sum(
            value_b * x_clip - rates - gammaln(value_b + 1.0),
            axis=-1,
        )

        # log p(x_s) — closed form via the prior low-rank MVN.
        log_p_x = self.base_dist.log_prob(x_samples)

        # log q(x_s | u) — closed form via the supplied variational
        # posterior. Whatever guide family the caller uses must
        # implement log_prob (every NumPyro Distribution does).
        log_q_x = q_dist.log_prob(x_samples)

        # IWAE aggregation: log mean of importance weights.
        log_w = log_p_u_given_x + log_p_x - log_q_x  # (S, *batch)
        return jsp.special.logsumexp(log_w, axis=0) - jnp.log(n_samples)

    @property
    def mean(self):
        """Marginal mean via log-normal moment formula.

        For a scalar PLN, ⟨u_g⟩ = exp(mu_g + sigma_g^2 / 2).
        Under the low-rank model, sigma_g^2 = (W W^T)_{gg} + d_g.

        Returns
        -------
        jnp.ndarray
            Shape ``(*batch_shape, G)`` expected counts per gene.
        """
        W = self.cov_factor
        sigma_sq = jnp.sum(W**2, axis=-1) + self.cov_diag
        return jnp.exp(self.loc + sigma_sq / 2.0)

    @property
    def variance(self):
        """Marginal variance of count ``u_g`` via the law of total variance.

        For ``u_g | lambda_g ~ Poisson(lambda_g)`` with
        ``lambda_g = exp(x_g)`` and ``x_g ~ Normal(mu_g, sigma_g^2)`` the
        marginal variance decomposes as

        .. math::
            \\mathrm{Var}[u_g] \\;=\\;
              \\mathbb{E}\\bigl[\\mathrm{Var}[u_g \\mid \\lambda_g]\\bigr] +
              \\mathrm{Var}\\bigl[\\mathbb{E}[u_g \\mid \\lambda_g]\\bigr]
            \\;=\\; \\mathbb{E}[\\lambda_g] + \\mathrm{Var}[\\lambda_g],

        i.e. ``E[lambda_g] = exp(mu_g + sigma_g^2 / 2)`` (the Poisson
        conditional-variance contribution) plus ``Var[lambda_g] =
        exp(2 mu_g + sigma_g^2) * (exp(sigma_g^2) - 1)`` (the
        between-cell rate variation).

        Earlier versions of this property dropped the ``E[lambda_g]``
        term and returned only ``Var[lambda_g]``, which under-reports
        the true count variance for high-mean genes by a fixed offset
        equal to the mean. The Monte-Carlo variance of ``sample()``
        agrees with the corrected formula.

        Returns
        -------
        jnp.ndarray
            Shape ``(*batch_shape, G)`` marginal variance per gene.
        """
        W = self.cov_factor
        sigma_sq = jnp.sum(W**2, axis=-1) + self.cov_diag
        mean_lambda = jnp.exp(self.loc + sigma_sq / 2.0)
        var_lambda = jnp.exp(2.0 * self.loc + sigma_sq) * (
            jnp.exp(sigma_sq) - 1.0
        )
        return mean_lambda + var_lambda


# ==============================================================================
# Softmax-Normal Distribution
# ==============================================================================


class SoftmaxNormal(Distribution):
    """
    Softmax-Normal distribution for compositional data (symmetric).

    This distribution models D-dimensional probability vectors (on the simplex)
    using a D-dimensional low-rank multivariate normal distribution with a
    softmax transformation. Unlike LowRankLogisticNormal (which uses ALR),
    this treats all components symmetrically.

    Mathematical Definition
    -----------------------
    Let y ∈ ℝ^D ~ MVN(μ, Σ) where Σ = WW^T + diag(D) is low-rank.
    The softmax transformation maps y to the simplex Δ^D:

        xᵢ = exp(yᵢ) / Σⱼ exp(yⱼ)   for i = 1, ..., D

    Symmetry and Invariance
    -----------------------
    The softmax transformation is:
    - Symmetric: All components treated equally (no reference component)
    - Translation-invariant: softmax(y + c·1) = softmax(y) for any constant c

    This translation invariance means the transformation is SINGULAR - you
    cannot uniquely invert it. Therefore, log_prob() is not available.

    Low-Rank Covariance Structure
    ------------------------------
    The covariance matrix has the form:

        Σ = WW^T + diag(D)

    where:
        - W is a D × rank factor matrix
        - D is a D-dimensional diagonal vector
        - Memory: O(D × rank) vs O(D²) for full covariance

    When to Use
    -----------
    Use SoftmaxNormal when:
        - You want symmetric treatment of all components
        - You only need sampling (not log_prob evaluation)
        - You're summarizing/visualizing posterior distributions

    Use LowRankLogisticNormal when:
        - You need to evaluate log_prob() for observed data
        - You're using the distribution as a likelihood in Bayesian inference
        - Asymmetry (reference component) is acceptable

    Parameters
    ----------
    loc : jnp.ndarray
        Location parameter μ ∈ ℝ^D (mean in log-space)
    cov_factor : jnp.ndarray
        Low-rank factor matrix W of shape (D, rank)
    cov_diag : jnp.ndarray
        Diagonal component D of shape (D,)
    validate_args : bool, optional
        Whether to validate input arguments

    Examples
    --------
    >>> from jax import random
    >>> import jax.numpy as jnp
    >>> # Create a softmax-normal for 5-dimensional simplex
    >>> D = 5
    >>> rank = 2
    >>> loc = jnp.zeros(D)
    >>> cov_factor = jnp.ones((D, rank)) * 0.1
    >>> cov_diag = jnp.ones(D) * 0.5
    >>> dist = SoftmaxNormal(loc, cov_factor, cov_diag)
    >>> # Sample from the distribution
    >>> samples = dist.sample(random.PRNGKey(0), (100,))
    >>> samples.shape
    (100, 5)
    >>> # Samples sum to 1
    >>> jnp.allclose(samples.sum(axis=-1), 1.0)
    True
    >>> # Access underlying log-space distribution
    >>> log_samples = dist.base_dist.sample(random.PRNGKey(1), (100,))
    >>> # Apply softmax manually
    >>> manual_samples = jax.nn.softmax(log_samples, axis=-1)

    See Also
    --------
    LowRankLogisticNormal : ALR-based alternative with log_prob() available
    """

    # Define constraints
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.independent(constraints.real, 2),
        "cov_diag": constraints.positive,
    }
    support = constraints.simplex
    has_rsample = False

    def __init__(self, loc, cov_factor, cov_diag, validate_args=None):
        loc = jnp.asarray(loc)
        cov_factor = jnp.asarray(cov_factor)
        cov_diag = jnp.asarray(cov_diag)

        # Store parameters
        self.loc = loc
        self.cov_factor = cov_factor
        self.cov_diag = cov_diag

        # Infer dimension
        self.n_dims = loc.shape[-1]  # D

        # Create internal low-rank multivariate normal
        self.base_dist = LowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
        )

        super().__init__(
            batch_shape=self.base_dist.batch_shape,
            event_shape=(self.n_dims,),
            validate_args=validate_args,
        )

    # --------------------------------------------------------------------------

    def sample(self, key, sample_shape=()):
        """
        Sample from the distribution.

        Returns samples on the D-dimensional simplex.

        Parameters
        ----------
        key : random.PRNGKey
            JAX random key
        sample_shape : tuple, optional
            Shape of samples to draw

        Returns
        -------
        jnp.ndarray
            Samples of shape (*sample_shape, *batch_shape, D) on the simplex
        """
        # Sample from base MVN in log-space
        y = self.base_dist.sample(key, sample_shape)  # (..., D)

        # Apply softmax transformation
        exp_y = jnp.exp(y)
        x = exp_y / jnp.sum(exp_y, axis=-1, keepdims=True)

        return x

    # --------------------------------------------------------------------------

    def log_prob(self, value):
        """
        Evaluate log probability density.

        NOT IMPLEMENTED: The softmax transformation is singular (adding a
        constant to all log-space coordinates doesn't change the output),
        so the Jacobian determinant is zero and log_prob is undefined.

        Parameters
        ----------
        value : jnp.ndarray
            Points on the simplex

        Raises
        ------
        NotImplementedError
            Always raised. Use LowRankLogisticNormal if you need log_prob(),
            or access base_dist.log_prob() for log-space density.
        """
        raise NotImplementedError(
            "log_prob() is not available for SoftmaxNormal because the softmax "
            "transformation is singular (adding a constant c to all components "
            "of y doesn't change softmax(y)). "
            "\n\nOptions:"
            "\n  1. Use LowRankLogisticNormal if you need log_prob() on the "
            "simplex (note: it uses ALR so the last component is a reference)"
            "\n  2. Use self.base_dist.log_prob() to evaluate density in "
            "log-space (but this doesn't account for the softmax transform)"
        )

    # --------------------------------------------------------------------------

    @property
    def mean(self):
        """
        Mean of the distribution on the simplex.

        Note: This is an approximation computed via Monte Carlo sampling,
        as the true mean does not have a closed form.

        Returns
        -------
        jnp.ndarray
            Approximate mean of shape (*batch_shape, D)
        """
        # Approximate mean via sampling
        key = random.PRNGKey(0)
        samples = self.sample(key, (10000,))
        return jnp.mean(samples, axis=0)


# ==============================================================================
# Poisson-Beta Compound Distribution (two-state promoter)
# ==============================================================================


class PoissonBetaCompound(Distribution):
    """Poisson-Beta compound distribution (non-bursty two-state promoter).

    Marginalises a Beta-distributed latent fraction p out of a Poisson:

        p ~ Beta(α, β)
        X | p ~ Poisson(rate · p)

    The marginal distribution is the steady-state mRNA count distribution
    for the non-bursty two-state promoter model (Peccoud-Ycart 1995). The
    closed-form marginal involves the confluent hypergeometric function
    ₁F₁ and is numerically fragile in the bursty regime; this class
    evaluates log_prob via Gauss-Jacobi quadrature over p, which is
    stable across the full parameter range.

    Shape conventions
    -----------------
    α and β are typically gene-rank (G,). ``rate`` may be gene-rank
    (G,) for the no-capture model or cell-by-gene (C, G) for the
    variable-capture model. The class deliberately does NOT promote
    α/β to the rate's rank: the quadrature nodes depend only on
    (α, β), so computing them at the gene rank avoids a
    per-(cell, gene) eigendecomposition.

    Parameters
    ----------
    alpha : jnp.ndarray
        Beta first shape parameter α (positive). Gene-rank.
    beta : jnp.ndarray
        Beta second shape parameter β (positive). Gene-rank.
    rate : jnp.ndarray, optional
        Poisson rate scale. At least one of ``rate`` or ``log_rate``
        must be provided; the other is derived.
    log_rate : jnp.ndarray, optional
        Logarithm of the Poisson rate scale. Equivalent to ``rate``;
        passing this directly is useful when the caller already has
        the log form and wants to avoid a redundant exp/log round-trip.
    n_quad_nodes : int, default=60
        Number of Gauss-Jacobi nodes used for log_prob. Static under JIT.
    quad_backend : str, default="golub_welsch"
        Quadrature backend; see :mod:`scribe.stats._jacobi_quad`.

    Notes
    -----
    The attributes ``self.alpha`` and ``self.beta`` are stored at
    their original (gene-rank) shape and MUST NOT be reshaped to the
    distribution's broadcast ``batch_shape``. Code that uses them
    for quadrature must respect this contract.
    """

    arg_constraints = {
        "alpha": constraints.positive,
        "beta": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative_integer
    has_rsample = False
    pytree_data_fields = ("alpha", "beta", "rate", "log_rate")
    pytree_aux_fields = ("n_quad_nodes", "quad_backend")

    def __init__(
        self,
        alpha,
        beta,
        rate=None,
        *,
        log_rate=None,
        n_quad_nodes: int = 60,
        quad_backend: str = "golub_welsch",
        validate_args=None,
    ):
        if rate is None and log_rate is None:
            raise ValueError(
                "PoissonBetaCompound requires at least one of "
                "'rate' or 'log_rate'."
            )

        # Gene-rank storage; DO NOT promote to the broadcast shape.
        # The quadrature backend reads self.alpha / self.beta at their
        # native rank, which is the whole point of the design.
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        if rate is not None:
            self.rate = jnp.asarray(rate)
            self.log_rate = (
                jnp.asarray(log_rate)
                if log_rate is not None
                else jnp.log(jnp.clip(self.rate, min=1e-300))
            )
        else:
            self.log_rate = jnp.asarray(log_rate)
            self.rate = jnp.exp(self.log_rate)

        # Static aux fields (Python int/str — not traced).
        self.n_quad_nodes = int(n_quad_nodes)
        self.quad_backend = str(quad_backend)

        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.rate),
            jnp.shape(self.alpha),
            jnp.shape(self.beta),
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    # ------------------------------------------------------------------

    def sample(self, key, sample_shape=()):
        """Ancestral sampling: p ~ Beta(α, β); x ~ Poisson(rate · p).

        Handles ``rate`` of higher rank than (α, β) — e.g. VCP with
        ``rate.shape == (C, G)`` and ``alpha.shape == (G,)`` — by
        inserting axes into the drawn ``p`` so multiplication
        broadcasts correctly.
        """
        assert is_prng_key(key)
        k_beta, k_pois = random.split(key)

        # Draw p at the gene rank, optionally prefixed by sample_shape.
        ab_shape = lax.broadcast_shapes(
            jnp.shape(self.alpha), jnp.shape(self.beta)
        )
        alpha_b = jnp.broadcast_to(self.alpha, ab_shape)
        beta_b = jnp.broadcast_to(self.beta, ab_shape)
        p = Beta(alpha_b, beta_b).sample(k_beta, sample_shape)
        # p now has shape sample_shape + ab_shape.

        # If rate has extra leading dims (e.g. cell axis), insert
        # size-1 axes into p so multiplication broadcasts cell → 1.
        # Both p and self.rate share the trailing gene axis.
        rate_shape = jnp.shape(self.rate)
        extra = len(rate_shape) - len(ab_shape)
        if extra > 0:
            insert_at = tuple(
                range(len(sample_shape), len(sample_shape) + extra)
            )
            p = jnp.expand_dims(p, axis=insert_at)

        # Compute λ; clamp to tiny positive to avoid Poisson(0) edge
        # cases at extreme parameter values.
        lam = self.rate * p
        lam = jnp.clip(lam, min=1e-30)

        return random.poisson(k_pois, lam).astype(jnp.int32)

    # ------------------------------------------------------------------

    def log_prob(self, value):
        """Log-PMF via Gauss-Jacobi quadrature over the latent p.

        Computes

            log L(value | α, β, rate)
              = log ∫₀¹ Poisson(value | rate · p) · Beta(p | α, β) dp

        using ``self.n_quad_nodes`` Gauss-Jacobi nodes. The Poisson
        log-PMF is evaluated in log-rate form,

            value · log λ − exp(log λ) − lgamma(value + 1),

        with log λ = log(rate) + log(p_k). This stays numerically
        stable when ``rate`` is large or when ``p_k`` is near 0 in
        the U-shaped Beta regime.

        Parameters
        ----------
        value : jnp.ndarray
            Non-negative integer counts, broadcastable to
            ``self.batch_shape``.

        Returns
        -------
        jnp.ndarray
            Log-probability values of shape ``self.batch_shape``.
        """
        # Local import to keep stats import-time clean — _jacobi_quad
        # has no scribe-side dependencies.
        from ._jacobi_quad import gauss_jacobi_nodes_weights

        # 1) Nodes and log-weights at gene rank only.
        #    Shapes: nodes_p, log_w == ab_shape + (K,).
        nodes_p, log_w = gauss_jacobi_nodes_weights(
            self.alpha,
            self.beta,
            self.n_quad_nodes,
            backend=self.quad_backend,
        )
        ab_shape = nodes_p.shape[:-1]

        # 2) Broadcast nodes against the (possibly higher-rank) log_rate.
        log_rate_shape = jnp.shape(self.log_rate)
        extra = len(log_rate_shape) - len(ab_shape)
        if extra > 0:
            for _ in range(extra):
                nodes_p = nodes_p[None, ...]
                log_w = log_w[None, ...]

        log_p_nodes = jnp.log(jnp.clip(nodes_p, min=1e-300))
        # log_lambda has shape rate_shape + (K,) after broadcasting.
        log_lambda = self.log_rate[..., None] + log_p_nodes

        # 3) Poisson log-PMF kernel in log-rate form. Insert a trailing
        # K axis on value so it broadcasts against log_lambda.
        value_arr = jnp.asarray(value)
        value_k = value_arr[..., None]
        log_poiss = (
            value_k * log_lambda - jnp.exp(log_lambda) - gammaln(value_k + 1.0)
        )

        # 4) Reduce over the K axis with logsumexp.
        weighted = log_w + log_poiss
        return jsp.special.logsumexp(weighted, axis=-1)

    # ------------------------------------------------------------------
    # Closed-form moments (cheap; no quadrature).
    # ------------------------------------------------------------------

    @property
    def mean(self):
        """⟨X⟩ = rate · α / (α + β)."""

        return self.rate * self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        """Var[X] = ⟨X⟩ + rate² · Var[p].

        Law of total variance:
            Var[X] = ⟨Var[X|p]⟩ + Var(⟨X|p⟩)
                  = ⟨rate · p⟩  + Var(rate · p)
                  = rate · ⟨p⟩  + rate² · Var[p]
        with
            ⟨p⟩   = α / (α + β)
            Var[p] = α·β / ((α + β)² · (α + β + 1))

        """
        ab = self.alpha + self.beta
        var_p = self.alpha * self.beta / (ab**2 * (ab + 1.0))
        return self.mean + self.rate**2 * var_p
