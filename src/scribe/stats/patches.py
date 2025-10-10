"""Mode property patches for NumPyro distributions."""

import jax.numpy as jnp

# ==============================================================================
# Mode property implementations
# ==============================================================================


def _beta_mode(self):
    """Monkey patch mode property for Beta distribution."""
    a = self.concentration1
    b = self.concentration0
    interior = (a > 1) & (b > 1)
    left_bd = (a <= 1) & (b > 1)  # mode at 0
    right_bd = (a > 1) & (b <= 1)  # mode at 1
    both_bd = (a <= 1) & (b <= 1)  # two boundary modes: 0 and 1

    interior_val = (a - 1) / (a + b - 2)  # safe because interior ⇒ denom > 0

    # Return NaN where the mode is non-unique (both boundaries), else
    # 0/1/interior
    return jnp.where(
        interior,
        interior_val,
        jnp.where(left_bd, 0.0, jnp.where(right_bd, 1.0, jnp.nan)),
    )


def _lognormal_mode(self):
    """Monkey patch mode property for LogNormal distribution."""
    # mode = exp(μ - σ²)
    return jnp.exp(self.loc - self.scale**2)


def _normal_mode(self):
    """Monkey patch mode property for Normal distribution."""
    # mode = μ (mean)
    return self.loc


# ==============================================================================
# Patch application
# ==============================================================================


def apply_distribution_mode_patches():
    """Apply mode property patches to NumPyro distributions."""
    from numpyro.distributions.continuous import Beta, LogNormal, Normal

    # Only add if not already present
    if not hasattr(Beta, "mode"):
        Beta.mode = property(_beta_mode)

    if not hasattr(LogNormal, "mode"):
        LogNormal.mode = property(_lognormal_mode)

    if not hasattr(Normal, "mode"):
        Normal.mode = property(_normal_mode)
