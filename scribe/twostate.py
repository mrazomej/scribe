"""
Probability functions for the two-state promoter model.
"""

# Imports for inference
import jax
import jax.numpy as jnp
import jax.scipy.special as jsps
from jax import random
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints

# ------------------------------------------------------------------------------
# High-precision implementation of the confluent hypergeometric function of the
# first kind
# ------------------------------------------------------------------------------

def hyp1f1_high_precision(a, b, z):
    """
    Compute Kummer's confluent hypergeometric function (1F1) in double
    precision.
    
    This function evaluates the confluent hypergeometric function by first
    casting all inputs to float64 for higher numerical precision, then casting
    the result back to the original input dtype. This helps avoid numerical
    instabilities that can occur when computing 1F1 in lower precision.

    Parameters
    ----------
    a : jax.numpy.ndarray
        First parameter of 1F1 (corresponds to 'a' in M(a,b,z))
    b : jax.numpy.ndarray 
        Second parameter of 1F1 (corresponds to 'b' in M(a,b,z))
    z : jax.numpy.ndarray
        Third parameter of 1F1 (corresponds to 'z' in M(a,b,z))

    Returns
    -------
    jax.numpy.ndarray
        The value of 1F1(a,b,z) computed in float64 precision but cast back to
        the original dtype of the inputs

    Notes
    -----
    This function must be called within a float64-enabled context:
    
    >>> with jax.experimental.enable_x64():
    >>>     result = hyp1f1_high_precision(a, b, z)
    """
    # 1) Cast inputs to float64
    a64 = a.astype(jnp.float64)
    b64 = b.astype(jnp.float64)
    z64 = z.astype(jnp.float64)
    # 2) Evaluate hyp1f1 in float64
    out64 = jsps.hyp1f1(a64, b64, z64)
    # 3) Cast result back to the original dtype (typically float32)
    return out64.astype(a.dtype)

# ------------------------------------------------------------------------------
# Logarithm of Kummer's confluent hypergeometric function
# ------------------------------------------------------------------------------

def log_hyp1f1_jax(
    x, y, z,
    transform=True,
    eps=jnp.finfo(jnp.float32).eps
):
    """
    Compute the logarithm of Kummer's confluent hypergeometric function M(a,b,z)
    using JAX.

    This function computes ln[M(a,b,z)] through the transformation: 
        ln[M(a,b,z)] = z + ln[M(b-a,b,-z)]

    This transformation is useful because it improves numerical stability when
    dealing with large arguments, as the confluent hypergeometric function can
    grow very rapidly.

    Parameters
    ----------
    x : jnp.ndarray
        First parameter of the hypergeometric function (corresponds to 'a')
    y : jnp.ndarray 
        Second parameter of the hypergeometric function (corresponds to 'b')
    z : jnp.ndarray
        Third parameter of the hypergeometric function (corresponds to 'z')
    transform : bool, optional
        Whether to apply the transformation ln[M(a,b,z)] = z + ln[M(b-a,b,-z)]
        (default is True)
    eps : float, optional
        Small value to avoid taking log of zero (default is
        jnp.finfo(jnp.float32).eps)

    Returns
    -------
    jnp.ndarray
        The logarithm of the confluent hypergeometric function evaluated at the
        given parameters: ln[M(x,y,z)]

    Notes
    -----
    - The transformation used here is based on Kummer's transformation: 
        M(a,b,z) = e^z * M(b-a,b,-z)
    - Taking the logarithm of both sides gives: 
        ln[M(a,b,z)] = z + ln[M(b-a,b,-z)]

    - This function must be called within a float64-enabled context:
    
    >>> with jax.experimental.enable_x64():
    >>>     result = log_hyp1f1_jax(a, b, z)
    """
    # Convert all inputs to arrays
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)
    
    # Broadcast to compatible shapes
    broadcast_shape = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
    x = jnp.broadcast_to(x, broadcast_shape)
    y = jnp.broadcast_to(y, broadcast_shape)
    z = jnp.broadcast_to(z, broadcast_shape)
    
    if transform:
        # Compute hyp1f1 value with transformation
        hyp1f1_value = hyp1f1_high_precision(y - x, y, -z)
        # Return the transformed log-hyp1f1 value
        return z + jnp.log(hyp1f1_value + eps)
    else:
        # Compute hyp1f1 value without transformation
        hyp1f1_value = hyp1f1_high_precision(x, y, z)
        # Return the log-hyp1f1 value
        return jnp.log(hyp1f1_value + eps)

# ------------------------------------------------------------------------------
# Logarithm of the probability mass function for the two-state promoter
# steady-state mRNA distribution
# ------------------------------------------------------------------------------

def log_pmf_twostate(
    mRNA,
    k_on,
    k_off,
    r_m,
    g_m=1.0,
    eps=jnp.finfo(jnp.float32).eps
):
    """
    Compute the log probability mass function (PMF) for a two-state promoter
    model.
    
    Parameters
    ----------
    mRNA : float or array-like
        mRNA copy number at which to evaluate the probability
    k_on : float or array-like
        Rate of promoter switching from OFF to ON state (units: 1/time)
    k_off : float or array-like
        Rate of promoter switching from ON to OFF state (units: 1/time) 
    r_m : float or array-like
        Production rate of mRNA when promoter is ON (units: 1/time)
    g_m : float or array-like, optional
        mRNA degradation rate (units: 1/time). Default is 1.0
    eps : float, optional
        Small value added for numerical stability. Default is JAX float32
        epsilon
    
    Returns
    -------
    float or array-like
        The natural logarithm of the probability mass function evaluated at the
        given mRNA copy number(s)
    
    Notes
    -----
    - This function must be called within a float64-enabled context due to the
    use of the confluent hypergeometric function:
    
    >>> with jax.experimental.enable_x64():
    >>>     result = log_pmf_twostate(mRNA, k_on, k_off, r_m, g_m=g_m, eps=eps)
    """
    # Convert inputs to arrays
    mRNA = jnp.asarray(mRNA)
    k_on = jnp.asarray(k_on)
    k_off = jnp.asarray(k_off)
    r_m = jnp.asarray(r_m)
    g_m = jnp.asarray(g_m)

    # Get the broadcast shape for all inputs
    broadcast_shape = jnp.broadcast_shapes(
        mRNA.shape,
        k_on.shape,
        k_off.shape,
        r_m.shape,
        g_m.shape
    )
    
    # Broadcast arrays to the common shape
    mRNA = jnp.broadcast_to(mRNA, broadcast_shape)
    k_on = jnp.broadcast_to(k_on, broadcast_shape)
    k_off = jnp.broadcast_to(k_off, broadcast_shape)
    r_m = jnp.broadcast_to(r_m, broadcast_shape)
    g_m = jnp.broadcast_to(g_m, broadcast_shape)

    # Compute hypergeometric terms
    # Note: these are already broadcast from previous steps
    a = k_on / g_m + mRNA
    b = (k_off + k_on) / g_m + mRNA
    c = -r_m / g_m
    
    # For bad values, try transformed computation
    hyp_term = log_hyp1f1_jax(
        a, b, c, transform=True, eps=eps
    )
    
    # Compute PMF terms
    lnp = (
        jsps.gammaln(k_on / g_m + mRNA + eps)
        - jsps.gammaln(mRNA + 1 + eps)
        - jsps.gammaln((k_off + k_on) / g_m + mRNA + eps)
        + jsps.gammaln((k_off + k_on) / g_m + eps)
        - jsps.gammaln(k_on / g_m + eps)
        + mRNA * jnp.log(r_m / g_m + eps)
        + hyp_term
    )

    return lnp

# ------------------------------------------------------------------------------
# Efficient computation of the PMF for the two-state promoter model
# ------------------------------------------------------------------------------

def log_pmf_twostate_efficient(
    k_on,
    k_off,
    r_m,
    g_m=1.0,
    max_mrna=20_000,
    chunk_size=100,
    prob_threshold=0.99999,
    eps=jnp.finfo(jnp.float32).eps
):
    """
    Efficiently compute the log probability mass function (PMF) for a two-state
    promoter model by processing mRNA values in chunks until a cumulative
    probability threshold is reached.

    This function calculates the PMF in a memory-efficient way by:
        1. Processing mRNA values in smaller chunks rather than all at once
        2. Stopping early once sufficient probability mass is captured
        3. Using numerical stabilization techniques for probability calculations

    Parameters
    ----------
    k_on : float or jax.numpy.ndarray
        Rate of promoter switching from OFF to ON state (units: 1/time)
    k_off : float or jax.numpy.ndarray
        Rate of promoter switching from ON to OFF state (units: 1/time)
    r_m : float or jax.numpy.ndarray
        Production rate of mRNA when promoter is ON (units: 1/time)
    g_m : float or jax.numpy.ndarray, optional
        mRNA degradation rate (units: 1/time). Default is 1.0
    max_mrna : int, optional
        Maximum mRNA value to consider before stopping. Default is 20000
    chunk_size : int, optional
        Number of mRNA values to process in each iteration. Default is 100
    prob_threshold : float, optional
        Stop when cumulative probability exceeds this value. Default is 0.99999
    eps : float, optional
        Small value added for numerical stability. Default is JAX float32
        epsilon

    Returns
    -------
    tuple
        Three-element tuple containing:
            - jax.numpy.ndarray : Array of mRNA values processed
            - jax.numpy.ndarray : Log probabilities for each mRNA value
            - float : Final cumulative probability reached

    Notes
    -----
    - This function must be called within a float64-enabled context due to the
    use of the confluent hypergeometric function:
    
    >>> with jax.experimental.enable_x64():
    >>>     result = log_pmf_twostate_efficient(
    >>>         k_on,
    >>>         k_off,
    >>>         r_m,
    >>>         g_m=g_m,
    >>>         max_mrna=max_mrna,
    >>>         chunk_size=chunk_size,
    >>>         prob_threshold=prob_threshold,
    >>>         eps=eps
    >>>     )
    """
    # Convert inputs to arrays and broadcast them together
    k_on = jnp.asarray(k_on)
    k_off = jnp.asarray(k_off)
    r_m = jnp.asarray(r_m)
    g_m = jnp.asarray(g_m)

    # Broadcast parameters to the same shape
    broadcast_shape = jnp.broadcast_shapes(
        k_on.shape, k_off.shape, r_m.shape, g_m.shape
    )
    k_on = jnp.broadcast_to(k_on, broadcast_shape)
    k_off = jnp.broadcast_to(k_off, broadcast_shape)
    r_m = jnp.broadcast_to(r_m, broadcast_shape)
    g_m = jnp.broadcast_to(g_m, broadcast_shape)


    # Compute theoretical mean for two-state model
    mean_mrna = r_m/g_m * k_on/(k_on + k_off)
    # Compute mean index to build chunks around it
    mean_idx = jnp.int32(mean_mrna)

    # Initialize lists to store results for each chunk
    all_mrna = []
    all_log_probs = []
    # Initialize running cumulative probability
    total_prob = 0.0

    # Start at mean
    center_start = jnp.maximum(0, mean_idx - chunk_size//2)
    center_end = jnp.minimum(max_mrna, mean_idx + chunk_size//2)

    # Compute center chunk mRNA values
    center_mrna = jnp.arange(center_start, center_end)
    # Compute log probabilities for center chunk
    center_log_probs = log_pmf_twostate(
        center_mrna, k_on, k_off, r_m, g_m=g_m, eps=eps
    )

    # Add center probabilities
    probs = jnp.exp(center_log_probs)
    # Handle -inf values
    probs = jnp.where(jnp.isfinite(center_log_probs), probs, 0.0)
    # Add to running cumulative probability
    total_prob += jnp.sum(probs)

    # Check if cumulative probability exceeds threshold
    if total_prob > prob_threshold:
        return center_mrna, center_log_probs, total_prob

    # Store results for center chunk
    all_mrna.append(center_mrna)
    all_log_probs.append(center_log_probs)

    # Initialize left and right chunk indices
    left_end = center_start
    right_start = center_end

    # Iterate over chunks of mRNA values
    while total_prob <= prob_threshold and (left_end > 0 or right_start < max_mrna):
        # Try left chunk if possible
        if left_end > 0:
            # Compute left chunk mRNA values
            left_start = jnp.maximum(0, left_end - chunk_size//2)
            left_mrna = jnp.arange(left_start, left_end)
            # Compute log probabilities for left chunk
            left_log_probs = log_pmf_twostate(
                left_mrna, k_on, k_off, r_m,
                g_m=g_m,
                eps=eps
            )

            # Convert to probabilities
            probs = jnp.exp(left_log_probs)
            # Handle -inf values
            probs = jnp.where(jnp.isfinite(left_log_probs), probs, 0.0)
            # Add to running cumulative probability
            total_prob += jnp.sum(probs)

            # Store results for left chunk
            all_mrna.insert(0, left_mrna)
            all_log_probs.insert(0, left_log_probs)
            # Update left chunk index
            left_end = left_start

            # Check if threshold is reached after left chunk
            if total_prob > prob_threshold:
                break

        # Try right chunk if possible
        if right_start < max_mrna:
            # Compute right chunk mRNA values
            right_end = jnp.minimum(max_mrna, right_start + chunk_size//2)
            right_mrna = jnp.arange(right_start, right_end)
            # Compute log probabilities for right chunk
            right_log_probs = log_pmf_twostate(
                right_mrna, k_on, k_off, r_m,
                g_m=g_m,
                eps=eps
            )

            # Convert to probabilities
            probs = jnp.exp(right_log_probs)
            # Handle -inf values
            probs = jnp.where(jnp.isfinite(right_log_probs), probs, 0.0)
            # Add to running cumulative probability
            total_prob += jnp.sum(probs)

            # Store results for right chunk
            all_mrna.append(right_mrna)
            all_log_probs.append(right_log_probs)
            # Update right chunk index
            right_start = right_end

            # Check if threshold is reached after right chunk
            if total_prob > prob_threshold:
                break

            # Stop if we've processed all chunks
            if left_end == 0 and right_start == max_mrna:
                break

    # Results are already sorted since we maintained order
    mRNA = jnp.concatenate(all_mrna)
    log_probs = jnp.concatenate(all_log_probs)

    return mRNA, log_probs, total_prob


# ------------------------------------------------------------------------------
# Sampling from the two-state promoter model
# ------------------------------------------------------------------------------

def sample_twostate(
    k_on,
    k_off,
    r_m,
    rng_key,
    g_m=1.0,
    max_mrna=20_000,
    n_samples=1000,
):
    """
    Generate samples from the two-state promoter model's steady-state
    distribution.
    
    Parameters
    ----------
    k_on : float or array-like
        Rate of promoter switching from OFF to ON state
    k_off : float or array-like
        Rate of promoter switching from ON to OFF state
    r_m : float or array-like
        Production rate of mRNA when promoter is ON
    rng_key : jax.random.PRNGKey
        Random number generator key
    g_m : float or array-like, optional
        mRNA degradation rate. Default is 1.0
    max_mrna : int, optional
        Maximum mRNA count to consider. Default is 20000
    n_samples : int, optional
        Number of samples to generate. Default is 1000

    Returns
    -------
    jax.numpy.ndarray
        Array of sampled mRNA counts with shape (n_samples,) or (n_samples,
        n_genes)
    
    Notes
    -----
    - This function must be called within a float64-enabled context due to the
      use of the confluent hypergeometric function:
    
    >>> with jax.experimental.enable_x64():
    >>>     result = sample_twostate(
    >>>         k_on,
    >>>         k_off,
    >>>         r_m,
    >>>         rng_key,
    >>>         g_m=g_m,
    >>>         max_mrna=max_mrna,
    >>>         n_samples=n_samples
    >>>     )
    """
    # Convert inputs to arrays
    k_on = jnp.asarray(k_on)
    k_off = jnp.asarray(k_off)
    r_m = jnp.asarray(r_m)
    g_m = jnp.asarray(g_m)
    
    # Get broadcast shape of all parameters
    param_shape = jnp.broadcast_shapes(
        k_on.shape, k_off.shape, r_m.shape, g_m.shape
    )
    
    # Broadcast parameters to common shape
    k_on = jnp.broadcast_to(k_on, param_shape)
    k_off = jnp.broadcast_to(k_off, param_shape)
    r_m = jnp.broadcast_to(r_m, param_shape)
    g_m = jnp.broadcast_to(g_m, param_shape)
    
    # Create mRNA values array
    mrna_values = jnp.arange(max_mrna)
    
    # Add necessary dimensions for broadcasting with parameters
    mrna_expanded = mrna_values.reshape((-1,) + (1,) * len(param_shape))
    
    # Compute log probabilities
    log_probs = log_pmf_twostate(
        mrna_expanded, k_on, k_off, r_m, g_m=g_m
    )
    
    # Replace -inf with very negative number to avoid NaN after exp
    log_probs = jnp.where(
        jnp.isfinite(log_probs),
        log_probs,
        jnp.finfo(log_probs.dtype).min
    )
    
    # Normalize probabilities in log space
    log_norm = jsps.logsumexp(log_probs, axis=0)
    log_pmf = log_probs - log_norm
    
    # Convert to probabilities
    pmf = jnp.exp(log_pmf)
    
    # If parameters are scalar, return samples with shape (n_samples,)
    # Otherwise, return samples with shape (n_samples, n_genes)
    if not param_shape:
        return random.choice(
            rng_key,
            mrna_values,
            shape=(n_samples,),
            p=pmf.squeeze()
        )
    
    # Split key for multiple sampling
    keys = random.split(rng_key, param_shape[0])
    
    # Sample for each set of parameters
    samples = jax.vmap(lambda key, p: random.choice(
        key,
        mrna_values,
        shape=(n_samples,),
        p=p
    ))(keys, pmf.T)
    
    return samples.T

# ------------------------------------------------------------------------------
def sample_twostate_efficient(
    k_on,
    k_off,
    r_m,
    rng_key,
    g_m=1.0,
    max_mrna=20_000,
    chunk_size=100,
    prob_threshold=0.99999,
    n_samples=1000,
    eps=jnp.finfo(jnp.float32).eps
):
    """
    Generate samples from the two-state promoter model's steady-state
    distribution using the efficient PMF computation method.

    Parameters
    ----------
    k_on : float
        Rate of promoter switching from OFF to ON state
    k_off : float
        Rate of promoter switching from ON to OFF state
    r_m : float
        Production rate of mRNA when promoter is ON
    rng_key : jax.random.PRNGKey
        Random number generator key
    n_samples : int, optional
        Number of samples to generate. Default is 1000
    max_mrna : int, optional
        Maximum mRNA count to consider. Default is 20000
    g_m : float, optional
        mRNA degradation rate. Default is 1.0
    eps : float, optional
        Small value added for numerical stability. Default is JAX float32
        epsilon

    Returns
    -------
    jax.numpy.ndarray
        Array of sampled mRNA counts
    
    Notes
    -----
    - This function must be called within a float64-enabled context due to the
    use of the confluent hypergeometric function:
    
    >>> with jax.experimental.enable_x64():
    >>>     result = sample_twostate_efficient(
    >>>         k_on,
    >>>         k_off,
    >>>         r_m,
    >>>         rng_key,
    >>>         g_m=g_m,
    >>>         max_mrna=max_mrna,
    >>>         n_samples=n_samples
    >>>     )
    """
    # Get the PMF using the efficient method
    mrna_range, log_probs, _ = log_pmf_twostate_efficient(
        k_on,
        k_off,
        r_m,
        g_m=g_m,
        max_mrna=max_mrna,
        chunk_size=chunk_size,
        prob_threshold=prob_threshold,
        eps=eps
    )

    # Normalize the PMF in log space
    log_norm = jsps.logsumexp(log_probs)
    log_pmf = log_probs - log_norm

    # Draw samples using JAX random
    samples = random.choice(
        rng_key,
        mrna_range,
        shape=(n_samples,),
        p=jnp.exp(log_pmf)
    )

    return samples


# ------------------------------------------------------------------------------
# NumPyro implementation of the two-state promoter model
# ------------------------------------------------------------------------------

class TwoStatePromoter(dist.Distribution):
    """
    A NumPyro distribution representing the steady-state mRNA distribution of a
    two-state promoter model.

    This class implements the analytical solution for the probability mass
    function (PMF) of mRNA counts produced by a gene switching between active
    and inactive states. The model parameters are:
        - k_on: Rate of switching from inactive to active state
        - k_off: Rate of switching from active to inactive state  
        - r_m: Rate of mRNA production in active state
        - g_m: Rate of mRNA degradation (default=1.0)

    The PMF is computed using Kummer's confluent hypergeometric function and
    involves numerical stabilization techniques to handle large parameter
    values.

    Parameters
    ----------
    k_on : float or jax.numpy.ndarray
        Rate of promoter switching from OFF to ON state
    k_off : float or jax.numpy.ndarray
        Rate of promoter switching from ON to OFF state
    r_m : float or jax.numpy.ndarray
        Production rate of mRNA when promoter is ON
    g_m : float or jax.numpy.ndarray, optional
        mRNA degradation rate (default=1.0)
    validate_args : bool, optional
        Whether to validate distribution parameters (default=None)

    Notes
    -----
    - Any method for this class must be called within a float64-enabled context
      due to the use of the confluent hypergeometric function:
    
    >>> dist = TwoStatePromoter(
    >>>     k_on,
    >>>     k_off,
    >>>     r_m,
    >>>     g_m=g_m,
    >>>     validate_args=validate_args
    >>> )
    >>> with jax.experimental.enable_x64():
    >>>     result = dist.sample(key)
    """
    # Define the valid parameter constraints for the distribution
    arg_constraints = {
        "k_on": constraints.positive,
        "k_off": constraints.positive,
        "r_m": constraints.positive,
        "g_m": constraints.positive,
    }

    def __init__(
            self, k_on, k_off, r_m, g_m=1.0, validate_args=None
        ):
        # Store the model parameters as instance attributes
        self.k_on = k_on
        self.k_off = k_off
        self.r_m = r_m
        self.g_m = g_m
        
        # Determine the batch shape by broadcasting all parameter shapes
        # together
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(k_on),
            jnp.shape(k_off),
            jnp.shape(r_m),
            jnp.shape(g_m)
        )
        
        # Initialize the base Distribution class
        super().__init__(batch_shape=batch_shape, event_shape=(), validate_args=validate_args)

    def log_prob(self, value):
        """
        Compute the log probability mass function for given mRNA counts.

        Parameters
        ----------
        value : int or jax.numpy.ndarray
            The mRNA count(s) to compute probability for

        Returns
        -------
        jax.numpy.ndarray
            Log probability of observing the given mRNA count(s)
        """
        # Call the analytical PMF computation function
        return log_pmf_twostate(
            value,
            self.k_on,
            self.k_off,
            self.r_m,
            g_m=self.g_m,
        )

    @property
    def support(self):
        """
        Define the support of the distribution (non-negative integers).
        
        Note: NumPyro does not have built-in integer constraints, so we use a
        custom one.
        """
        return constraints.nonnegative_integer

    def sample(self, key, sample_shape=()):
        """
        Generate samples from the two-state promoter model using the efficient
        sampling method.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key
        sample_shape : tuple, optional
            Shape of samples to generate (default=())

        Returns
        -------
        jax.numpy.ndarray
            Array of sampled mRNA counts with shape: sample_shape + batch_shape
        """
        # Calculate total number of samples needed
        n_samples = jnp.prod(jnp.array(sample_shape))
        
        # Generate samples using the efficient method
        samples = sample_twostate(
            self.k_on,
            self.k_off,
            self.r_m,
            key,
            g_m=self.g_m,
            n_samples=n_samples
        )
        
        # Reshape samples to include sample_shape and batch_shape
        return samples.reshape(sample_shape + self.batch_shape)

