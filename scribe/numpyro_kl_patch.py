"""
Monkey patch to add BetaPrime KL divergence support to Numpyro.

This module extends Numpyro's KL divergence system to support BetaPrime distributions
by adding the necessary dispatch registrations.
"""

from multipledispatch import dispatch
import jax.numpy as jnp
from jax.scipy.special import betaln, digamma, gammaln

from .stats import BetaPrime, kl_betaprime


def register_betaprime_kl_divergence():
    """
    Register KL divergence implementations for BetaPrime distributions with Numpyro.
    
    This function monkey-patches Numpyro's KL divergence system to add support
    for BetaPrime distributions. It should be called once at the start of your
    application.
    """
    
    # Import numpyro KL divergence here to avoid circular imports
    from numpyro.distributions.kl import kl_divergence
    
    @dispatch(BetaPrime, BetaPrime)
    def kl_divergence_betaprime(p, q):
        """
        Compute KL divergence between two BetaPrime distributions.
        
        Parameters
        ----------
        p : BetaPrime
            First BetaPrime distribution
        q : BetaPrime
            Second BetaPrime distribution
            
        Returns
        -------
        jnp.ndarray
            KL divergence KL(p||q)
        """
        # Extract parameters from the distributions
        a, b = p.concentration1, p.concentration0
        alpha, beta = q.concentration1, q.concentration0
        
        # Use the closed-form KL divergence formula
        a_diff = alpha - a
        b_diff = beta - b
        t1 = betaln(alpha, beta) - betaln(a, b)
        t2 = a_diff * digamma(a) + b_diff * digamma(b)
        t3 = (a_diff + b_diff) * digamma(a + b)
        return t1 - t2 + t3
    
    # Register the new dispatch function
    # We need to add it to the existing kl_divergence function
    # This is a bit hacky but works with Numpyro's dispatch system
    kl_divergence.register(BetaPrime, BetaPrime)(kl_divergence_betaprime)
    
    return True


# Auto-register when module is imported
register_betaprime_kl_divergence()
