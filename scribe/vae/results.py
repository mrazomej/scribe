"""
VAE-specific results class for SCRIBE inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass

import jax
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
from flax import nnx
from numpyro.distributions import Beta
try:
    from jax.random import PRNGKey
except ImportError:
    from jax.random import Key as PRNGKey

import numpy as np

# Import base results class
from ..svi.results import ScribeSVIResults
from .architectures import VAE, VAEConfig, dpVAE, EncoderVCP
from ..sampling import sample_variational_posterior, generate_predictive_samples

try:
    from anndata import AnnData
except ImportError:
    AnnData = None


@dataclass
class ScribeVAEResults(ScribeSVIResults):
    """
    VAE-specific results class for SCRIBE inference.

    This class extends ScribeSVIResults to include VAE-specific functionality
    such as latent space analysis, cell clustering, and VAE model access.
    Supports both standard VAE and dpVAE (decoupled prior VAE) models.

    Attributes
    ----------
    vae_model : Union[VAE, dpVAE]
        The trained VAE or dpVAE model used for inference
    latent_samples : Optional[jnp.ndarray]
        Samples from the latent space for analysis
    cell_embeddings : Optional[jnp.ndarray]
        Cell embeddings in latent space
    prior_type : str
        Type of prior used: "standard" for VAE or "decoupled" for dpVAE
    """

    # VAE-specific attributes (using init=False to avoid dataclass field
    # ordering issues)
    _vae_model: Optional[Union[VAE, dpVAE]] = None  # type: ignore
    latent_samples: Optional[jnp.ndarray] = None
    cell_embeddings: Optional[jnp.ndarray] = None
    prior_type: str = "standard"
    # Standardization statistics
    standardize_mean: Optional[jnp.ndarray] = None
    standardize_std: Optional[jnp.ndarray] = None

    def __post_init__(self):
        """Validate VAE-specific configuration."""
        # Call parent validation
        super().__post_init__()

        # Validate that this is a VAE model
        if self.model_config.inference_method != "vae":
            raise ValueError(
                f"Model config inference_method must be 'vae', "
                f"got '{self.model_config.inference_method}'"
            )

        # Determine if this is a dpVAE based on model config
        # Check if prior-specific parameters are present
        if self.model_config.vae_prior_type == "decoupled":
            self.prior_type = "decoupled"

    # --------------------------------------------------------------------------
    # Model and guide access
    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Optional[Callable]]:
        """
        Get the model and guide functions for this VAE model.

        For VAE models, this returns the actual model and guide functions
        (not the factory functions) that can be used for sampling.

        Returns
        -------
        Tuple[Callable, Optional[Callable]]
            A tuple containing (model_function, guide_function)
        """
        # For VAE models, we need to get the actual model and guide functions
        # from the appropriate module, not the factory functions
        if self.model_config.parameterization == "standard":
            if self.prior_type == "standard":
                if self.model_type == "nbdm":
                    from ..models.vae_standard import (
                        nbdm_vae_model,
                        nbdm_vae_guide,
                    )

                    return nbdm_vae_model, nbdm_vae_guide
                elif self.model_type == "zinb":
                    from ..models.vae_standard import (
                        zinb_vae_model,
                        zinb_vae_guide,
                    )

                    return zinb_vae_model, zinb_vae_guide
                elif self.model_type == "nbvcp":
                    from ..models.vae_standard import (
                        nbvcp_vae_model,
                        nbvcp_vae_guide,
                    )

                    return nbvcp_vae_model, nbvcp_vae_guide
            elif self.prior_type == "decoupled":
                if self.model_type == "nbdm":
                    from ..models.vae_standard import (
                        nbdm_dpvae_model,
                        nbdm_vae_guide,
                    )

                    return nbdm_dpvae_model, nbdm_vae_guide
                elif self.model_type == "zinb":
                    from ..models.vae_standard import (
                        zinb_dpvae_model,
                        zinb_vae_guide,
                    )

                    return zinb_dpvae_model, zinb_vae_guide
                elif self.model_type == "nbvcp":
                    from ..models.vae_standard import (
                        nbvcp_dpvae_model,
                        nbvcp_vae_guide,
                    )

                    return nbvcp_dpvae_model, nbvcp_vae_guide
        elif self.model_config.parameterization == "linked":
            if self.prior_type == "standard":
                if self.model_type == "nbdm":
                    from ..models.vae_linked import (
                        nbdm_vae_model,
                        nbdm_vae_guide,
                    )

                    return nbdm_vae_model, nbdm_vae_guide
                elif self.model_type == "zinb":
                    from ..models.vae_linked import (
                        zinb_vae_model,
                        zinb_vae_guide,
                    )

                    return zinb_vae_model, zinb_vae_guide
                elif self.model_type == "nbvcp":
                    from ..models.vae_linked import (
                        nbvcp_vae_model,
                        nbvcp_vae_guide,
                    )

                    return nbvcp_vae_model, nbvcp_vae_guide
            elif self.prior_type == "decoupled":
                if self.model_type == "nbdm":
                    from ..models.vae_linked import (
                        nbdm_dpvae_model,
                        nbdm_vae_guide,
                    )

                    return nbdm_dpvae_model, nbdm_vae_guide
                elif self.model_type == "zinb":
                    from ..models.vae_linked import (
                        zinb_dpvae_model,
                        zinb_vae_guide,
                    )

                    return zinb_dpvae_model, zinb_vae_guide
                elif self.model_type == "nbvcp":
                    from ..models.vae_linked import (
                        nbvcp_dpvae_model,
                        nbvcp_vae_guide,
                    )

                    return nbvcp_dpvae_model, nbvcp_vae_guide
        elif self.model_config.parameterization == "odds_ratio":
            if self.prior_type == "standard":
                if self.model_type == "nbdm":
                    from ..models.vae_odds_ratio import (
                        nbdm_vae_model,
                        nbdm_vae_guide,
                    )

                    return nbdm_vae_model, nbdm_vae_guide
                elif self.model_type == "zinb":
                    from ..models.vae_odds_ratio import (
                        zinb_vae_model,
                        zinb_vae_guide,
                    )

                    return zinb_vae_model, zinb_vae_guide
                elif self.model_type == "nbvcp":
                    from ..models.vae_odds_ratio import (
                        nbvcp_vae_model,
                        nbvcp_vae_guide,
                    )

                    return nbvcp_vae_model, nbvcp_vae_guide
            elif self.prior_type == "decoupled":
                if self.model_type == "nbdm":
                    from ..models.vae_odds_ratio import (
                        nbdm_dpvae_model,
                        nbdm_vae_guide,
                    )

                    return nbdm_dpvae_model, nbdm_vae_guide
                elif self.model_type == "zinb":
                    from ..models.vae_odds_ratio import (
                        zinb_dpvae_model,
                        zinb_vae_guide,
                    )

                    return zinb_dpvae_model, zinb_vae_guide
                elif self.model_type == "nbvcp":
                    from ..models.vae_odds_ratio import (
                        nbvcp_dpvae_model,
                        nbvcp_vae_guide,
                    )

                    return nbvcp_dpvae_model, nbvcp_vae_guide
        elif self.model_config.parameterization == "unconstrained":
            if self.prior_type == "standard":
                if self.model_type == "nbdm":
                    from ..models.vae_unconstrained import (
                        nbdm_vae_model,
                        nbdm_vae_guide,
                    )

                    return nbdm_vae_model, nbdm_vae_guide
                elif self.model_type == "zinb":
                    from ..models.vae_unconstrained import (
                        zinb_vae_model,
                        zinb_vae_guide,
                    )

                    return zinb_vae_model, zinb_vae_guide
                elif self.model_type == "nbvcp":
                    from ..models.vae_unconstrained import (
                        nbvcp_vae_model,
                        nbvcp_vae_guide,
                    )

                    return nbvcp_vae_model, nbvcp_vae_guide
            elif self.prior_type == "decoupled":
                if self.model_type == "nbdm":
                    from ..models.vae_unconstrained import (
                        nbdm_dpvae_model,
                        nbdm_vae_guide,
                    )

                    return nbdm_dpvae_model, nbdm_vae_guide
                elif self.model_type == "zinb":
                    from ..models.vae_unconstrained import (
                        zinb_dpvae_model,
                        zinb_vae_guide,
                    )

                    return zinb_dpvae_model, zinb_vae_guide
                elif self.model_type == "nbvcp":
                    from ..models.vae_unconstrained import (
                        nbvcp_dpvae_model,
                        nbvcp_vae_guide,
                    )

                    return nbvcp_dpvae_model, nbvcp_vae_guide
        else:
            raise NotImplementedError(
                f"get_model_and_guide not implemented for "
                f"'{self.model_config.parameterization}'."
            )

        raise ValueError(f"Unknown model type: {self.model_type}")

    # --------------------------------------------------------------------------
    # Reconstruct VAE model from trained parameters
    # --------------------------------------------------------------------------

    @property
    def vae_model(self) -> Union[VAE, dpVAE]:
        """
        Get the VAE model, reconstructing it if necessary.
        """
        if self._vae_model is None:
            # Import functions
            from .architectures import (
                create_encoder,
                create_decoder,
                DecoupledPrior,
            )

            # Define if VAE has variable capture
            variable_capture = self.model_type == "nbvcp"

            # Create encoder
            encoder = create_encoder(
                input_dim=self.n_genes,
                latent_dim=self.model_config.vae_latent_dim,
                hidden_dims=self.model_config.vae_hidden_dims,
                activation=self.model_config.vae_activation,
                input_transformation=self.model_config.vae_input_transformation,
                standardize_mean=self.standardize_mean,
                standardize_std=self.standardize_std,
                variable_capture=variable_capture,
            )
            # Split encoder
            encoder_graph, encoder_state = nnx.split(encoder)
            # Replace encoder state with trained state
            nnx.replace_by_pure_dict(
                encoder_state, self.params["encoder$params"]
            )
            # Merge encoder
            encoder = nnx.merge(encoder_graph, encoder_state)

            # Create decoder
            decoder = create_decoder(
                input_dim=self.model_config.vae_latent_dim,
                latent_dim=self.n_genes,
                hidden_dims=self.model_config.vae_hidden_dims,
                activation=self.model_config.vae_activation,
                standardize_mean=self.standardize_mean,
                standardize_std=self.standardize_std,
                variable_capture=variable_capture,
            )
            # Split decoder
            decoder_graph, decoder_state = nnx.split(decoder)
            # Replace decoder state with trained state
            nnx.replace_by_pure_dict(
                decoder_state, self.params["decoder$params"]
            )
            # Merge decoder
            decoder = nnx.merge(decoder_graph, decoder_state)

            # Define RNGs
            rngs = nnx.Rngs(params=0)

            # Create VAE config
            vae_config = VAEConfig(
                input_dim=self.n_genes,
                latent_dim=self.model_config.vae_latent_dim,
                hidden_dims=self.model_config.vae_hidden_dims,
                activation=self.model_config.vae_activation,
                standardize_mean=self.standardize_mean,
                standardize_std=self.standardize_std,
                variable_capture=variable_capture,
            )

            if self.prior_type == "decoupled":
                # Create decoupled prior for dpVAE
                decoupled_prior = DecoupledPrior(
                    latent_dim=self.model_config.vae_latent_dim,
                    num_layers=self.model_config.vae_prior_num_layers,
                    hidden_dims=self.model_config.vae_prior_hidden_dims,
                    rngs=rngs,
                    activation=self.model_config.vae_prior_activation,
                    mask_type=self.model_config.vae_prior_mask_type,
                )
                # Split decoupled prior
                prior_graph, prior_state = nnx.split(decoupled_prior)
                # Replace prior state with trained state
                nnx.replace_by_pure_dict(
                    prior_state, self.params["decoupled_prior$params"]
                )
                # Merge decoupled prior
                decoupled_prior = nnx.merge(prior_graph, prior_state)

                # Create dpVAE
                self._vae_model = dpVAE(
                    encoder=encoder,
                    decoder=decoder,
                    config=vae_config,
                    decoupled_prior=decoupled_prior,
                    rngs=rngs,
                )
            else:
                # Create standard VAE
                self._vae_model = VAE(
                    encoder=encoder,
                    decoder=decoder,
                    config=vae_config,
                    rngs=rngs,
                )
        return self._vae_model

    # --------------------------------------------------------------------------

    def get_distributions(self) -> Dict[str, Any]:
        """
        Get the variational distributions for all parameters.

        This method now delegates to the model-specific
        `get_posterior_distributions` function associated with the
        parameterization.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to their distributions.

        Raises
        ------
        ValueError
            If backend is not supported.
        """
        # Dynamically import the correct posterior distribution function
        if self.model_config.parameterization == "standard":
            from ..models.vae_standard import (
                get_posterior_distributions as get_dist_fn,
            )
        elif self.model_config.parameterization == "linked":
            from ..models.vae_linked import (
                get_posterior_distributions as get_dist_fn,
            )
        elif self.model_config.parameterization == "odds_ratio":
            from ..models.vae_odds_ratio import (
                get_posterior_distributions as get_dist_fn,
            )
        elif self.model_config.parameterization == "unconstrained":
            from ..models.vae_unconstrained import (
                get_posterior_distributions as get_dist_fn,
            )
        else:
            raise NotImplementedError(
                f"get_distributions not implemented for "
                f"'{self.model_config.parameterization}'."
            )

        distributions = get_dist_fn(
            self.params, self.model_config, self.vae_model
        )

        return distributions

    # --------------------------------------------------------------------------
    # VAE-specific sampling methods
    # --------------------------------------------------------------------------

    def get_latent_samples(
        self,
        n_samples: int = 100,
        rng_key: Optional[jax.random.PRNGKey] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """
        Get multiple samples from the latent space prior for uncertainty
        quantification.

        This method samples from the prior distribution rather than conditioning
        on data. For standard VAE, it samples from a standard normal prior. For
        dpVAE, it samples from the learned decoupled prior distribution.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate
        rng_key : Optional[jax.random.PRNGKey], default=None
            Random key for sampling
        store_samples : bool, default=True
            Whether to store the generated latent samples in the object. Default
            is True.

        Returns
        -------
        jnp.ndarray
            Latent samples of shape (n_samples, latent_dim)
        """
        if rng_key is None:
            rng_key = jax.random.key(42)

        # Get the prior distribution
        prior_dist = self.vae_model.get_prior_distribution()

        # Sample all samples at once
        samples_array = prior_dist.sample(rng_key, sample_shape=(n_samples,))

        # Store samples if requested
        if store_samples:
            self.latent_samples = samples_array

        return samples_array

    # --------------------------------------------------------------------------

    def get_latent_samples_conditioned_on_data(
        self,
        counts: jnp.ndarray,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        rng_key: Optional[jax.random.PRNGKey] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """
        Generate multiple samples from the latent space posterior conditioned on
        observed data.

        This method encodes the observed count data using the VAE encoder to
        obtain the mean and log-variance of the approximate posterior for each
        cell. It then draws samples from the resulting Gaussian posterior for
        each cell, repeating this process `n_samples` times to quantify
        uncertainty in the latent space.

        If a batch size is provided, the data is processed in batches to reduce
        memory usage for large datasets.

        Parameters
        ----------
        counts : jnp.ndarray
            Array of observed count data with shape (n_cells, n_genes).
        n_samples : int, default=100
            Number of posterior samples to generate.
        batch_size : Optional[int], default=None
            If provided, process the data in batches of this size.
        rng_key : Optional[jax.random.PRNGKey], default=None
            JAX random key for reproducible sampling. If None, a default key is
            used.
        store_samples : bool, default=True
            Whether to store the generated latent samples in the object
            attribute `self.latent_samples`.

        Returns
        -------
        jnp.ndarray
            Array of latent samples with shape (n_samples, n_cells, latent_dim).
        """
        if rng_key is None:
            rng_key = jax.random.key(42)

        # Get VAE encoder
        encoder = self.vae_model.encoder

        # Generate multiple samples
        z_samples = []
        for i in range(n_samples):
            key = jax.random.fold_in(rng_key, i)
            if batch_size is None:
                if isinstance(encoder, EncoderVCP):
                    mean, logvar, _, _ = encoder(counts)
                else:
                    # Process all cells at once
                    mean, logvar = encoder(counts)
                # Sample from latent space
                std = jnp.exp(0.5 * logvar)
                eps = jax.random.normal(key, mean.shape)
                z = mean + eps * std
                z_samples.append(z)
            else:
                # Process in batches
                batch_samples = []
                for j in range(0, counts.shape[0], batch_size):
                    batch = counts[j : j + batch_size]
                    if isinstance(encoder, EncoderVCP):
                        mean, logvar, _, _ = encoder(batch)
                    else:
                        mean, logvar = encoder(batch)
                    # Sample from latent space
                    std = jnp.exp(0.5 * logvar)
                    eps = jax.random.normal(key, mean.shape)
                    z = mean + eps * std
                    batch_samples.append(z)
                z_samples.append(jnp.concatenate(batch_samples, axis=0))

        z_samples_array = jnp.stack(z_samples, axis=0)

        # Store samples if requested
        if store_samples:
            self.latent_samples = z_samples_array

        return z_samples_array

    # --------------------------------------------------------------------------

    def get_p_capture_samples_conditioned_on_data(
        self,
        counts: jnp.ndarray,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        rng_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Generate multiple samples of the p_capture parameter conditioned on
        observed data.

        This method is only applicable for models using the EncoderVCP encoder,
        which parameterizes a Beta distribution for the p_capture variable. For
        each sample, the encoder is used to obtain the log-alpha and log-beta
        parameters for each cell, and a sample is drawn from the corresponding
        Beta distribution. This is repeated `n_samples` times to quantify
        uncertainty in p_capture.

        If a batch size is provided, the data is processed in batches to reduce
        memory usage for large datasets.

        Parameters
        ----------
        counts : jnp.ndarray
            Array of observed count data with shape (n_cells, n_genes).
        n_samples : int, default=100
            Number of posterior samples to generate.
        batch_size : Optional[int], default=None
            If provided, process the data in batches of this size.
        rng_key : Optional[jax.random.PRNGKey], default=None
            JAX random key for reproducible sampling. If None, a default key is
            used.

        Returns
        -------
        jnp.ndarray or None
            Array of p_capture samples with shape (n_samples, n_cells) if
            EncoderVCP is used, otherwise None.
        """
        if rng_key is None:
            rng_key = jax.random.key(42)

        # Get VAE encoder
        encoder = self.vae_model.encoder

        if isinstance(encoder, EncoderVCP):
            from ..stats import BetaPrime
            import jax.scipy as jsp
            import numpyro.distributions as dist

            # Generate multiple samples
            p_capture_samples = []
            for i in range(n_samples):
                key = jax.random.fold_in(rng_key, i)
                if batch_size is None:
                    # Process all cells at once
                    _, _, param1, param2 = encoder(counts)

                    if self.model_config.parameterization in [
                        "standard",
                        "linked",
                    ]:
                        # Extract parameters for Beta distribution
                        alpha = jnp.exp(param1.squeeze(-1))
                        beta = jnp.exp(param2.squeeze(-1))
                        # Sample from Beta distribution
                        p_capture = dist.Beta(alpha, beta).sample(key)
                    elif self.model_config.parameterization == "odds_ratio":
                        # Extract parameters for BetaPrime distribution
                        alpha = jnp.exp(param1.squeeze(-1))
                        beta = jnp.exp(param2.squeeze(-1))
                        # Sample from BetaPrime distribution
                        phi_capture = BetaPrime(alpha, beta).sample(key)
                        # Convert to p_capture
                        p_capture = jsp.special.expit(phi_capture)
                    elif self.model_config.parameterization == "unconstrained":
                        # Extract parameters for Normal distribution
                        loc = param1.squeeze(-1)
                        scale = jnp.exp(param2.squeeze(-1))
                        # Sample from Normal distribution
                        p_capture_unconstrained = dist.Normal(
                            loc, scale
                        ).sample(key)
                        # Convert to p_capture
                        p_capture = jsp.special.expit(p_capture_unconstrained)

                    p_capture_samples.append(p_capture)
                else:
                    # Process in batches
                    batch_samples = []
                    for j in range(0, counts.shape[0], batch_size):
                        batch = counts[j : j + batch_size]
                        _, _, param1, param2 = encoder(batch)

                        if self.model_config.parameterization in [
                            "standard",
                            "linked",
                        ]:
                            # Extract parameters for Beta distribution
                            alpha = jnp.exp(param1.squeeze(-1))
                            beta = jnp.exp(param2.squeeze(-1))
                            # Sample from Beta distribution
                            p_capture = dist.Beta(alpha, beta).sample(key)
                        elif (
                            self.model_config.parameterization == "odds_ratio"
                        ):
                            # Extract parameters for BetaPrime distribution
                            alpha = jnp.exp(param1.squeeze(-1))
                            beta = jnp.exp(param2.squeeze(-1))
                            # Sample from BetaPrime distribution
                            phi_capture = BetaPrime(alpha, beta).sample(key)
                            # Convert to p_capture
                            p_capture = jsp.special.expit(phi_capture)
                        elif (
                            self.model_config.parameterization
                            == "unconstrained"
                        ):
                            # Extract parameters for Normal distribution
                            loc = param1.squeeze(-1)
                            scale = jnp.exp(param2.squeeze(-1))
                            # Sample from Normal distribution
                            p_capture_unconstrained = dist.Normal(
                                loc, scale
                            ).sample(key)
                            # Convert to p_capture
                            p_capture = jsp.special.expit(
                                p_capture_unconstrained
                            )

                        batch_samples.append(p_capture)

                    # Concatenate batch samples
                    p_capture_samples.append(
                        jnp.concatenate(batch_samples, axis=0)
                    )

            # Stack samples
            p_capture_samples_array = jnp.stack(p_capture_samples, axis=0)

            # Return samples
            return p_capture_samples_array
        else:
            return None

    # --------------------------------------------------------------------------

    def get_posterior_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        store_samples: bool = True,
        canonical: bool = False,
    ) -> Dict:
        """
        Sample parameters from the variational posterior distribution for the
        VAE.

        This method samples from the variational posterior by drawing from the
        latent space prior.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data of shape (n_cells, n_genes). This argument is accepted
            for interface compatibility but is not used in this method, as
            sampling is performed from the prior.
        rng_key : jax.random.PRNGKey, optional
            JAX random number generator key. Default is random.PRNGKey(42).
        n_samples : int, optional
            Number of posterior samples to generate. Default is 100.
        store_samples : bool, optional
            Whether to store the generated posterior samples in the object.
            Default is True.
        canonical : bool, optional
            Whether to convert the samples to canonical form. Default is False.

        Returns
        -------
        Dict
            Dictionary containing samples from the variational posterior
            distribution. The structure of the dictionary depends on the model
            and guide used.
        """
        # Get the guide function
        _, guide = self.get_model_and_guide()

        if guide is None:
            raise ValueError(
                f"Could not find a guide for model '{self.model_type}'."
            )

        # Prepare base model arguments
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "model_config": self.model_config,
            "encoder": self.vae_model.encoder,
        }

        # Sample from posterior
        posterior_samples = sample_variational_posterior(
            guide, self.params, model_args, rng_key=rng_key, n_samples=n_samples
        )

        # For dpVAE: sample from the learned decoupled prior
        if self.prior_type == "decoupled":
            # Get the decoupled prior distribution
            decoupled_prior_dist = self.vae_model.get_prior_distribution()

            # Sample all samples at once
            posterior_samples["z"] = decoupled_prior_dist.sample(
                rng_key, sample_shape=(n_samples,)
            )

        # Run z samples through the decoder
        decoded_samples = self.vae_model.decoder(posterior_samples["z"])

        # Store decoded samples with right keys
        if self.model_config.parameterization == "standard":
            posterior_samples["log_r"] = decoded_samples
            posterior_samples["r"] = jnp.exp(decoded_samples)
        elif self.model_config.parameterization == "linked":
            posterior_samples["log_mu"] = decoded_samples
            posterior_samples["mu"] = jnp.exp(decoded_samples)
        elif self.model_config.parameterization == "odds_ratio":
            posterior_samples["log_mu"] = decoded_samples
            posterior_samples["mu"] = jnp.exp(decoded_samples)
        elif self.model_config.parameterization == "unconstrained":
            posterior_samples["r_unconstrained"] = decoded_samples
            posterior_samples["r"] = jnp.exp(decoded_samples)

        # Store samples if requested
        if store_samples:
            self.posterior_samples = posterior_samples

        # Convert to canonical form if requested
        if canonical:
            self._convert_to_canonical()

        return posterior_samples

    # --------------------------------------------------------------------------

    def get_posterior_samples_conditioned_on_data(
        self,
        counts: jnp.ndarray,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        canonical: bool = False,
    ) -> Dict:
        """
        Sample parameters from the variational posterior distribution,
        conditioned on observed data.

        This method generates samples from the variational posterior in a way
        that is conditioned on the provided count data. It first samples global
        and cell-level parameters from the variational posterior, then generates
        latent variable samples (e.g., cell embeddings) by encoding the provided
        data and sampling from the resulting posterior distribution. These
        latent samples are passed through the VAE decoder to obtain
        reconstructed parameters (such as 'r' or 'mu'), which are then
        substituted into the posterior samples dictionary. The latent samples
        themselves are also included in the returned dictionary.

        This approach ensures that the posterior samples reflect the structure
        present in the observed data, rather than being drawn from the latent
        prior. This is particularly important for downstream analyses or
        predictive checks that require data-conditioned posterior samples.

        Parameters
        ----------
        counts : jnp.ndarray
            Observed count data of shape (n_cells, n_genes) used to generate
            data-conditioned latent samples.
        rng_key : jax.random.PRNGKey, optional
            JAX random number generator key. Default is random.PRNGKey(42).
        n_samples : int, optional
            Number of posterior samples to generate. Default is 100.
        batch_size : Optional[int], default=None
            Batch size for processing large datasets
        store_samples : bool, optional
            Whether to store the generated posterior samples in the object.
            Default is True.
        canonical : bool, optional
            Whether to convert the samples to canonical form. Default is False.

        Returns
        -------
        Dict
            Dictionary containing samples from the variational posterior,
            including reconstructed parameters (e.g., 'r' or 'mu') and latent
            variables ('z'), all conditioned on the provided data.

        Notes
        -----
        - This method is preferred over sampling from the latent prior when
          posterior samples should reflect the structure of the observed data.
        - The returned dictionary will contain keys for the reconstructed
          parameters (such as 'r' or 'mu', depending on the model) and for the
          latent variables ('z').
        - If `canonical` is True, the samples will be converted to canonical
          form using the object's internal method.
        """
        # Get the guide function
        _, guide = self.get_model_and_guide()

        if guide is None:
            raise ValueError(
                f"Could not find a guide for model '{self.model_type}'."
            )

        # Prepare base model arguments
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "model_config": self.model_config,
            "encoder": self.vae_model.encoder,
        }

        # Sample from posterior
        posterior_samples = sample_variational_posterior(
            guide, self.params, model_args, rng_key=rng_key, n_samples=n_samples
        )

        # Generate latent samples conditioned on data
        posterior_samples["z"] = self.get_latent_samples_conditioned_on_data(
            counts,
            n_samples=n_samples,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        # Generate p_capture samples conditioned on data
        posterior_samples["p_capture"] = (
            self.get_p_capture_samples_conditioned_on_data(
                counts,
                n_samples=n_samples,
                rng_key=rng_key,
                batch_size=batch_size,
            )
        )

        # Run latent samples through decoder to obtain reconstructed parameters
        decoded_samples = self.vae_model.decoder(posterior_samples["z"])

        # Store decoded samples with right keys
        if self.model_config.parameterization == "standard":
            # Get the log-transformed r values
            posterior_samples["log_r"] = decoded_samples
            # Get the r values
            posterior_samples["r"] = jnp.exp(decoded_samples)
            # Make p samples compatible in shape with r samples
            posterior_samples["p"] = posterior_samples["p"][:, None]
        elif self.model_config.parameterization == "linked":
            # Get the log-transformed mu values
            posterior_samples["log_mu"] = decoded_samples
            # Get the mu values
            posterior_samples["mu"] = jnp.exp(decoded_samples)
            # Make p samples compatible in shape with r samples
            posterior_samples["p"] = posterior_samples["p"][:, None]

        elif self.model_config.parameterization == "odds_ratio":
            # Get the log-transformed mu values
            posterior_samples["log_mu"] = decoded_samples
            # Get the mu values
            posterior_samples["mu"] = jnp.exp(decoded_samples)
            # Make phi samples compatible in shape with r samples
            posterior_samples["phi"] = posterior_samples["phi"][:, None]
        elif self.model_config.parameterization == "unconstrained":
            # Get the log-transformed r values
            posterior_samples["r_unconstrained"] = decoded_samples
            # Get the r values
            posterior_samples["r"] = jnp.exp(decoded_samples)
            # Make p samples compatible in shape with r samples
            posterior_samples["p_unconstrained"] = posterior_samples[
                "p_unconstrained"
            ][:, None]

        # Store samples if requested
        if store_samples:
            self.posterior_samples = posterior_samples

        # Convert to canonical form if requested
        if canonical:
            self._convert_to_canonical()

        return posterior_samples

    # --------------------------------------------------------------------------

    def get_predictive_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples."""
        # Get the model and guide functions
        model, _ = self.get_model_and_guide()

        # Prepare base model arguments
        model_args = {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "model_config": self.model_config,
            "decoder": self.vae_model.decoder,
            "decoupled_prior": self.vae_model.decoupled_prior,
        }

        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Generate predictive samples
        predictive_samples = generate_predictive_samples(
            model,
            self.posterior_samples,
            model_args,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        # Store samples if requested
        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

    # --------------------------------------------------------------------------
    # VAE-specific analysis methods
    # --------------------------------------------------------------------------

    def get_latent_embeddings(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Get latent embeddings for cells using the trained VAE.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data of shape (n_cells, n_genes)
        batch_size : Optional[int], default=None
            Batch size for processing large datasets

        Returns
        -------
        jnp.ndarray
            Latent embeddings of shape (n_cells, latent_dim)
        """
        # Get VAE encoder
        encoder = self.vae_model.encoder

        if batch_size is None:
            # Process all cells at once
            if isinstance(encoder, EncoderVCP):
                mean, _, _, _ = encoder(counts)
            else:
                mean, _ = encoder(counts)
            return mean
        else:
            # Process in batches
            embeddings = []
            for i in range(0, counts.shape[0], batch_size):
                batch = counts[i : i + batch_size]
                if isinstance(encoder, EncoderVCP):
                    mean, _, _, _ = encoder(batch)
                else:
                    mean, _ = encoder(batch)
                embeddings.append(mean)
            return jnp.concatenate(embeddings, axis=0)

    # --------------------------------------------------------------------------
    # Override parent methods for VAE-specific behavior
    # --------------------------------------------------------------------------

    def get_map(
        self,
        use_mean: bool = False,
        canonical: bool = True,
        verbose: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Get MAP estimates including VAE-generated r parameters.

        This method extends the parent method to include r parameters
        generated by the VAE for each cell.
        """
        # Get base MAP estimates
        map_estimates = super().get_map(
            use_mean=use_mean, canonical=canonical, verbose=verbose
        )

        # Add VAE-generated r parameters if we have the original data
        if hasattr(self, "_original_counts"):
            # Generate r parameters using VAE
            r_params = self.vae_model.decoder.decode(self._original_counts)
            map_estimates["r_vae"] = r_params

        return map_estimates

    # --------------------------------------------------------------------------
    # Factory methods
    # --------------------------------------------------------------------------

    @classmethod
    def from_svi_results(
        cls,
        svi_results: ScribeSVIResults,
        vae_model: Union[VAE, dpVAE],
        original_counts: Optional[jnp.ndarray] = None,
        standardize_mean: Optional[jnp.ndarray] = None,
        standardize_std: Optional[jnp.ndarray] = None,
    ) -> "ScribeVAEResults":
        """
        Create VAE results from existing SVI results.

        Parameters
        ----------
        svi_results : ScribeSVIResults
            Base SVI results
        vae_model : Union[VAE, dpVAE]
            Trained VAE or dpVAE model
        original_counts : Optional[jnp.ndarray], default=None
            Original count data used for training

        Returns
        -------
        ScribeVAEResults
            VAE-specific results object
        """
        # Determine the prior type based on model type or config
        prior_type = "standard"
        if hasattr(svi_results.model_config, "vae_prior_hidden_dims"):
            prior_type = "decoupled"
        elif isinstance(vae_model, dpVAE):
            prior_type = "decoupled"

        # Create VAE results with all SVI attributes
        vae_results = cls(
            params=svi_results.params,
            loss_history=svi_results.loss_history,
            n_cells=svi_results.n_cells,
            n_genes=svi_results.n_genes,
            model_type=svi_results.model_type,
            model_config=svi_results.model_config,
            prior_params=svi_results.prior_params,
            obs=svi_results.obs,
            var=svi_results.var,
            uns=svi_results.uns,
            n_obs=svi_results.n_obs,
            n_vars=svi_results.n_vars,
            posterior_samples=svi_results.posterior_samples,
            predictive_samples=svi_results.predictive_samples,
            n_components=svi_results.n_components,
            prior_type=prior_type,
            standardize_mean=standardize_mean,
            standardize_std=standardize_std,
        )

        # Set VAE model after creation (but don't store it directly to avoid pickling issues)
        if vae_model is not None:
            vae_results._vae_model = vae_model

        # Store original counts for later use
        if original_counts is not None:
            vae_results._original_counts = original_counts

        return vae_results

    # --------------------------------------------------------------------------
    # VAE-specific indexing functionality
    # --------------------------------------------------------------------------

    def __getitem__(self, index):
        """
        Enable indexing of ScribeVAEResults object with VAE decoder
        modification.

        This extends the parent indexing to also modify the VAE decoder to
        output only the selected genes, making all VAE calculations much more
        manageable.
        """
        # First, get the subset using parent indexing
        subset_results = super().__getitem__(index)

        # Create a new VAE model with modified decoder
        subset_results._vae_model = self._create_indexed_vae_model(index)

        return subset_results

    def _create_indexed_vae_model(self, index):
        """
        Create a new VAE model with decoder modified to output only indexed
        genes.

        Parameters
        ----------
        index : Union[int, slice, list, np.ndarray, jnp.ndarray]
            Index specifying which genes to keep

        Returns
        -------
        Union[VAE, dpVAE]
            New VAE model with modified decoder
        """
        # Convert index to boolean mask for consistency
        if isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool:
            bool_index = index
        elif isinstance(index, int):
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            bool_index = bool_index.at[index].set(True)
        elif isinstance(index, slice):
            indices = jnp.arange(self.n_genes)[index]
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        elif isinstance(index, (list, np.ndarray, jnp.ndarray)) and not (
            isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool
        ):
            indices = jnp.array(index)
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

        # Get the number of selected genes
        n_selected_genes = int(bool_index.sum())

        # Get the original VAE model
        original_vae = self.vae_model

        # Create modified decoder
        modified_decoder = self._create_indexed_decoder(
            bool_index, n_selected_genes
        )

        # Create new VAE config with modified input_dim
        vae_config = VAEConfig(
            input_dim=n_selected_genes,  # Modified for subset
            latent_dim=self.model_config.vae_latent_dim,
            hidden_dims=self.model_config.vae_hidden_dims,
            activation=self.model_config.vae_activation,
            standardize_mean=(
                self.standardize_mean[bool_index]
                if self.standardize_mean is not None
                else None
            ),
            standardize_std=(
                self.standardize_std[bool_index]
                if self.standardize_std is not None
                else None
            ),
        )

        # Create new VAE with same encoder and prior, but modified decoder
        if self.prior_type == "decoupled":
            # For dpVAE, we need to create a new instance with the modified
            # decoder
            from .architectures import dpVAE

            return dpVAE(
                encoder=original_vae.encoder,
                decoder=modified_decoder,
                config=vae_config,
                decoupled_prior=original_vae.decoupled_prior,
                rngs=original_vae.rngs,
            )
        else:
            # For standard VAE
            from .architectures import VAE

            return VAE(
                encoder=original_vae.encoder,
                decoder=modified_decoder,
                config=vae_config,
                rngs=original_vae.rngs,
            )

    def _create_indexed_decoder(self, bool_index, n_selected_genes):
        """
        Create a new decoder with output layer modified for gene subset.

        Parameters
        ----------
        bool_index : jnp.ndarray
            Boolean mask indicating which genes to keep
        n_selected_genes : int
            Number of selected genes

        Returns
        -------
        Decoder
            New decoder with modified output layer
        """
        from .architectures import Decoder, VAEConfig

        # Create a temporary config for the decoder
        temp_config = VAEConfig(
            input_dim=n_selected_genes,
            latent_dim=self.model_config.vae_latent_dim,
            hidden_dims=self.model_config.vae_hidden_dims,
            activation=self.model_config.vae_activation,
            standardize_mean=(
                self.standardize_mean[bool_index]
                if self.standardize_mean is not None
                else None
            ),
            standardize_std=(
                self.standardize_std[bool_index]
                if self.standardize_std is not None
                else None
            ),
        )

        # Create new decoder with same hidden layers but modified output
        decoder = Decoder(temp_config, rngs=nnx.Rngs(params=0))

        # Split decoder to access its state
        decoder_graph, decoder_state = nnx.split(decoder)

        # Copy hidden layer parameters from original decoder
        original_decoder_state = self.params["decoder$params"]

        # Copy all hidden layer parameters (they remain the same)
        for key in original_decoder_state:
            if key.startswith("decoder_layers"):
                decoder_state[key] = original_decoder_state[key]

        # Modify the output layer parameters
        original_kernel = original_decoder_state["decoder_output"]["kernel"]
        original_bias = original_decoder_state["decoder_output"]["bias"]

        # Extract subset of weights and bias
        subset_kernel = original_kernel[:, bool_index]
        subset_bias = original_bias[bool_index]

        # Set the modified output layer parameters
        decoder_state["decoder_output"]["kernel"] = subset_kernel
        decoder_state["decoder_output"]["bias"] = subset_bias

        # Merge decoder
        decoder = nnx.merge(decoder_graph, decoder_state)

        return decoder

    # --------------------------------------------------------------------------
