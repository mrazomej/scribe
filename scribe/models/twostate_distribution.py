"""
Beta-Poisson distribution implementation for NumPyro.

This module implements the Beta-Poisson compound distribution, which is
mathematically equivalent to the two-state promoter model.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsps
from jax import random
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.distributions.util import (
    promote_shapes,
    validate_sample,
    is_prng_key,
)

# ------------------------------------------------------------------------------
# Beta-Poisson distribution
# ------------------------------------------------------------------------------


class BetaPoisson(dist.Distribution):
    """
    Compound distribution comprising of a Beta-Poisson pair.

    The rate parameter λ = d*p for the Poisson distribution is constructed by:
        1. Sampling p from Beta(α, β)
        2. Scaling by dose parameter d
        3. Using λ = d*p as the Poisson rate

    This is mathematically equivalent to the steady-state two-state promoter
    model with:
        - α = k_on / g_m
        - β = k_off / g_m
        - d = r_m / g_m

    Parameters
    ----------
    alpha : float or array_like
        Shape parameter α > 0 of the Beta distribution
    beta : float or array_like
        Shape parameter β > 0 of the Beta distribution
    dose : float or array_like
        Dose parameter d > 0 that scales the Beta-distributed rate
    validate_args : bool, optional
        Whether to validate arguments
    """

    # Define constraints for the distribution parameters
    arg_constraints = {
        "alpha": constraints.positive,
        "beta": constraints.positive,
        "dose": constraints.positive,
    }
    # Define the support of the distribution
    support = constraints.nonnegative_integer
    pytree_data_fields = ("alpha", "beta", "dose", "_beta_dist")

    # Initialize the distribution
    def __init__(self, alpha, beta, dose, *, validate_args=None):
        # Store the parameters
        self.alpha, self.beta, self.dose = promote_shapes(alpha, beta, dose)
        # Create a Beta distribution object
        self._beta_dist = dist.Beta(self.alpha, self.beta)
        # Initialize the distribution
        super(BetaPoisson, self).__init__(
            self._beta_dist.batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        """Sample from Beta-Poisson using two-step process."""
        assert is_prng_key(key)
        key_beta, key_poisson = random.split(key)

        # Sample p from Beta(α, β)
        p = self._beta_dist.sample(key_beta, sample_shape)

        # Sample from Poisson(d*p)
        rate = self.dose * p
        return dist.Poisson(rate).sample(key_poisson)

    @validate_sample
    def log_prob(self, value):
        """
        Compute log probability using the exact Beta-Poisson PMF.

        Uses the derived formula:
        P(X=x) = [Γ(α+x)/(x!Γ(α))] × [Γ(α+β)/Γ(α+β+x)] × d^x × ₁F₁(α+x, α+β+x, -d)
        """
        # Convert value to ensure it's an array
        value = jnp.asarray(value)

        # Compute log probability components
        log_prob = (
            # Γ(α + x) / (x! Γ(α))
            jsps.gammaln(self.alpha + value)
            - jsps.gammaln(value + 1)
            - jsps.gammaln(self.alpha)
            # Γ(α + β) / Γ(α + β + x)
            + jsps.gammaln(self.alpha + self.beta)
            - jsps.gammaln(self.alpha + self.beta + value)
            # d^x
            + value * jnp.log(self.dose)
        )

        # Add hypergeometric function term: ₁F₁(α+x, α+β+x, -d)
        hyp_args = (
            self.alpha + value,  # a
            self.alpha + self.beta + value,  # b
            self.dose,  # z
        )

        # Use standard JAX hypergeometric function with current precision
        hyp_value = jsps.hyp1f1(*hyp_args)

        # Handle infinite values using JAX-compatible operations
        log_hyp = jnp.where(
            jnp.isinf(hyp_value),
            -jnp.inf,
            jnp.log(hyp_value) - self.dose
        )

        return log_prob + log_hyp

    @property
    def mean(self):
        """Expected value of Beta-Poisson distribution."""
        # E[X] = E[E[X|p]] = E[d*p] = d*E[p] = d*α/(α+β)
        return self.dose * self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        """Variance of Beta-Poisson distribution."""
        # Var[X] = E[Var[X|p]] + Var[E[X|p]]
        # = E[d*p] + Var[d*p]
        # = d*α/(α+β) + d²*Var[p]
        # = d*α/(α+β) + d²*αβ/[(α+β)²(α+β+1)]
        beta_mean = self.alpha / (self.alpha + self.beta)
        beta_var = (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )
        return self.dose * beta_mean + (self.dose**2) * beta_var


# ------------------------------------------------------------------------------
# Two-State Promoter Steady-State distribution
# ------------------------------------------------------------------------------


class TwoStatePromoter(BetaPoisson):
    """
    Two-State Promoter distribution implemented as a Beta-Poisson.

    This provides the same distribution as the original two-state promoter model
    but with more efficient sampling and a direct connection to the Beta-Poisson
    literature.

    Parameters
    ----------
    k_on : float or array_like
        Rate of promoter switching from OFF to ON state
    k_off : float or array_like
        Rate of promoter switching from ON to OFF state
    r_m : float or array_like
        Production rate of mRNA when promoter is ON
    g_m : float or array_like, optional
        mRNA degradation rate (default=1.0)
    validate_args : bool, optional
        Whether to validate arguments
    """

    arg_constraints = {
        "k_on": constraints.positive,
        "k_off": constraints.positive,
        "r_m": constraints.positive,
        "g_m": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(self, k_on, k_off, r_m, g_m=1.0, *, validate_args=None):
        # Store biological parameters
        self.k_on = k_on
        self.k_off = k_off
        self.r_m = r_m
        self.g_m = g_m

        # Convert to Beta-Poisson parameters
        alpha = k_on / g_m
        beta = k_off / g_m
        dose = r_m / g_m

        super().__init__(alpha, beta, dose, validate_args=validate_args)

    @property
    def mean(self):
        """Expected mRNA count in terms of biological parameters."""
        # E[mRNA] = r_m * k_on / (g_m * (k_on + k_off))
        return self.r_m * self.k_on / (self.g_m * (self.k_on + self.k_off))

    @property
    def variance(self):
        """Variance of mRNA count in terms of biological parameters."""
        mean_val = self.mean
        # Additional variance due to promoter switching
        burst_factor = 1 + self.r_m / (self.k_off + self.g_m)
        return mean_val * burst_factor
