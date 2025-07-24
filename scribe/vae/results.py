"""
VAE-specific results class for SCRIBE inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any, List
from dataclasses import dataclass, replace
import warnings

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.nn import sigmoid, softmax
import pandas as pd
import numpyro.distributions as dist
from jax import random, jit, vmap
from flax import nnx

import numpy as np
import scipy.stats as stats

# Import base results class
from ..svi.results import ScribeSVIResults
from ..models.model_config import ModelConfig
from .architectures import VAE, VAEConfig

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

    Attributes
    ----------
    vae_model : VAE
        The trained VAE model used for inference
    latent_samples : Optional[jnp.ndarray]
        Samples from the latent space for analysis
    cell_embeddings : Optional[jnp.ndarray]
        Cell embeddings in latent space
    """

    # VAE-specific attributes (using init=False to avoid dataclass field
    # ordering issues)
    _vae_model: Optional[VAE] = None  # type: ignore
    latent_samples: Optional[jnp.ndarray] = None
    cell_embeddings: Optional[jnp.ndarray] = None

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

    # --------------------------------------------------------------------------
    # Model and guide access
    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Optional[Callable]]:
        """
        Get the model and guide functions for this VAE model.

        Returns
        -------
        Tuple[Callable, Optional[Callable]]
            A tuple containing (model_function, guide_function)
        """
        return self._model_and_guide()

    # --------------------------------------------------------------------------
    # Reconstruct VAE model from trained parameters
    # --------------------------------------------------------------------------

    @property
    def vae_model(self) -> VAE:
        """
        Get the VAE model, reconstructing it if necessary.
        """
        if self._vae_model is None:
            # Import functions
            from .architectures import create_encoder, create_decoder

            # Create encoder
            encoder = create_encoder(
                input_dim=self.n_genes,
                latent_dim=self.model_config.vae_latent_dim,
                hidden_dims=self.model_config.vae_hidden_dims,
                activation=self.model_config.vae_activation,
                input_transformation=self.model_config.vae_input_transformation,
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
                output_activation=self.model_config.vae_output_activation,
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
            # Create VAE
            self._vae_model = VAE(
                encoder=encoder,
                decoder=decoder,
                config=self.model_config,
                rngs=rngs,
            )
        return self._vae_model

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
            mean, _ = encoder(counts)
            return mean
        else:
            # Process in batches
            embeddings = []
            for i in range(0, counts.shape[0], batch_size):
                batch = counts[i : i + batch_size]
                mean, _ = encoder(batch)
                embeddings.append(mean)
            return jnp.concatenate(embeddings, axis=0)

    # --------------------------------------------------------------------------

    def get_latent_samples(
        self,
        counts: jnp.ndarray,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        rng_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Get multiple samples from the latent space for uncertainty
        quantification.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data of shape (n_cells, n_genes)
        n_samples : int, default=100
            Number of samples to generate
        batch_size : Optional[int], default=None
            Batch size for processing large datasets
        rng_key : Optional[jax.random.PRNGKey], default=None
            Random key for sampling

        Returns
        -------
        jnp.ndarray
            Latent samples of shape (n_samples, n_cells, latent_dim)
        """
        if rng_key is None:
            rng_key = jax.random.key(42)

        # Get VAE encoder
        encoder = self.vae_model.encoder

        # Generate multiple samples
        samples = []
        for i in range(n_samples):
            key = jax.random.fold_in(rng_key, i)
            if batch_size is None:
                # Process all cells at once
                mean, logvar = encoder(counts)
                # Sample from latent space
                std = jnp.exp(0.5 * logvar)
                eps = jax.random.normal(key, mean.shape)
                z = mean + eps * std
                samples.append(z)
            else:
                # Process in batches
                batch_samples = []
                for j in range(0, counts.shape[0], batch_size):
                    batch = counts[j : j + batch_size]
                    mean, logvar = encoder(batch)
                    # Sample from latent space
                    std = jnp.exp(0.5 * logvar)
                    eps = jax.random.normal(key, mean.shape)
                    z = mean + eps * std
                    batch_samples.append(z)
                samples.append(jnp.concatenate(batch_samples, axis=0))

        return jnp.stack(samples, axis=0)

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
        latent space prior, rather than conditioning on the observed data. As a
        result, the sampled parameters may not reflect the structure present in
        the actual data, especially if the cell embeddings exhibit meaningful
        structure. For most use cases where posterior samples should reflect the
        observed data, it is recommended to use
        `get_posterior_samples_from_data` instead.

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

        Warnings
        --------
        This method samples from the latent space prior, not conditioned on the
        data. If the cell embeddings have structure, these samples may not be
        representative. Use `get_posterior_samples_from_data` for
        data-conditioned posterior samples.
        """
        print(
            "[WARNING] get_posterior_samples is sampling from the latent space "
            "prior. If there is structure to the embeddings of the cells, this "
            "may not be a good sample. We recommend using "
            "get_posterior_samples_from_data instead."
        )
        # Get posterior samples
        posterior_samples = super().get_posterior_samples(
            rng_key=rng_key, 
            n_samples=n_samples, 
            store_samples=store_samples, 
        )


    # --------------------------------------------------------------------------

    def get_posterior_samples_from_data(
        self,
        counts: jnp.ndarray,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        store_samples: bool = True,
        canonical: bool = False,
    ) -> Dict:
        """
        Sample parameters from the variational posterior distribution,
        conditioned on observed data.

        This method generates samples from the variational posterior in a way
        that is conditioned on the provided count data. It first samples global
        and cell-level parameters from the variational posterior, then generates
        latent variable samples (e.g., cell embeddings) using the provided data.
        These latent samples are passed through the VAE decoder to obtain
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
        # Get posterior samples (global/cell-level parameters)
        posterior_samples = super().get_posterior_samples(
            rng_key=rng_key, 
            n_samples=n_samples, 
            store_samples=store_samples, 
        )

        # Generate latent samples conditioned on data
        latent_samples = self.get_latent_samples(
            counts, n_samples=n_samples, rng_key=rng_key
        )

        # Extract decoder from the VAE model
        decoder = self.vae_model.decoder
        
        # Run latent samples through decoder to obtain reconstructed parameters
        reconstructed = decoder(latent_samples)

        # Substitute reconstructed parameters into posterior samples
        if 'r' in posterior_samples:
            posterior_samples['r'] = reconstructed
        elif 'mu' in posterior_samples:
            posterior_samples['mu'] = reconstructed

        # Add latent samples to posterior samples
        posterior_samples['z'] = latent_samples

        # Convert to canonical form if requested
        if canonical:
            self._convert_to_canonical()

        return posterior_samples

    # --------------------------------------------------------------------------

    def cluster_cells(
        self,
        counts: jnp.ndarray,
        n_clusters: int = 5,
        method: str = "kmeans",
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Cluster cells based on their latent representations.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data of shape (n_cells, n_genes)
        n_clusters : int, default=5
            Number of clusters to create
        method : str, default="kmeans"
            Clustering method. Options: "kmeans", "hierarchical", "dbscan"
        batch_size : Optional[int], default=None
            Batch size for processing large datasets
        **kwargs
            Additional arguments for clustering algorithms

        Returns
        -------
        Dict[str, Any]
            Dictionary containing clustering results:
            - 'labels': Cluster assignments for each cell
            - 'centroids': Cluster centroids in latent space
            - 'embeddings': Cell embeddings in latent space
            - 'method': Clustering method used
        """
        # Get latent embeddings
        embeddings = self.get_latent_embeddings(counts, batch_size=batch_size)

        if method == "kmeans":
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
            labels = kmeans.fit_predict(embeddings)
            centroids = kmeans.cluster_centers_

        elif method == "hierarchical":
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, **kwargs
            )
            labels = clustering.fit_predict(embeddings)
            # Compute centroids manually
            centroids = np.array(
                [
                    embeddings[labels == i].mean(axis=0)
                    for i in range(n_clusters)
                ]
            )

        elif method == "dbscan":
            from sklearn.cluster import DBSCAN

            clustering = DBSCAN(**kwargs)
            labels = clustering.fit_predict(embeddings)
            # Handle noise points (label = -1)
            unique_labels = np.unique(labels[labels != -1])
            centroids = np.array(
                [
                    embeddings[labels == label].mean(axis=0)
                    for label in unique_labels
                ]
            )

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        return {
            "labels": labels,
            "centroids": centroids,
            "embeddings": embeddings,
            "method": method,
        }

    # --------------------------------------------------------------------------

    def compute_latent_statistics(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute statistics of the latent space.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data of shape (n_cells, n_genes)
        batch_size : Optional[int], default=None
            Batch size for processing large datasets

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing latent space statistics:
            - 'mean': Mean of each latent dimension
            - 'std': Standard deviation of each latent dimension
            - 'min': Minimum of each latent dimension
            - 'max': Maximum of each latent dimension
            - 'correlation_matrix': Correlation matrix between latent dimensions
        """
        # Get latent embeddings
        embeddings = self.get_latent_embeddings(counts, batch_size=batch_size)

        # Compute statistics
        mean = jnp.mean(embeddings, axis=0)
        std = jnp.std(embeddings, axis=0)
        min_val = jnp.min(embeddings, axis=0)
        max_val = jnp.max(embeddings, axis=0)

        # Compute correlation matrix
        correlation_matrix = jnp.corrcoef(embeddings.T)

        return {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "correlation_matrix": correlation_matrix,
        }

    # --------------------------------------------------------------------------

    def visualize_latent_space(
        self,
        counts: jnp.ndarray,
        method: str = "tsne",
        n_components: int = 2,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create low-dimensional visualization of the latent space.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data of shape (n_cells, n_genes)
        method : str, default="tsne"
            Dimensionality reduction method. Options: "tsne", "umap", "pca"
        n_components : int, default=2
            Number of components for visualization
        batch_size : Optional[int], default=None
            Batch size for processing large datasets
        **kwargs
            Additional arguments for dimensionality reduction algorithms

        Returns
        -------
        Dict[str, Any]
            Dictionary containing visualization results:
            - 'coordinates': 2D coordinates for plotting
            - 'method': Dimensionality reduction method used
            - 'embeddings': Original latent embeddings
        """
        # Get latent embeddings
        embeddings = self.get_latent_embeddings(counts, batch_size=batch_size)

        if method == "tsne":
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=n_components, random_state=42, **kwargs)
            coordinates = reducer.fit_transform(embeddings)

        elif method == "umap":
            from umap import UMAP

            reducer = UMAP(n_components=n_components, random_state=42, **kwargs)
            coordinates = reducer.fit_transform(embeddings)

        elif method == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=n_components, **kwargs)
            coordinates = reducer.fit_transform(embeddings)

        else:
            raise ValueError(f"Unknown visualization method: {method}")

        return {
            "coordinates": coordinates,
            "method": method,
            "embeddings": embeddings,
        }

    def get_gene_importance(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Compute gene importance based on VAE decoder weights.

        This method analyzes the decoder weights to understand which genes
        are most important for reconstructing the data from the latent space.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data of shape (n_cells, n_genes)
        batch_size : Optional[int], default=None
            Batch size for processing large datasets

        Returns
        -------
        jnp.ndarray
            Gene importance scores of shape (n_genes,)
        """
        # Get the decoder output layer weights
        decoder_weights = self.vae_model.decoder.decoder_output.weight.value

        # Compute importance as the sum of squared weights for each gene
        # This measures how much each gene contributes to the reconstruction
        importance = jnp.sum(decoder_weights**2, axis=0)

        return importance

    # --------------------------------------------------------------------------

    def reconstruct_data(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Reconstruct data using the trained VAE.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data of shape (n_cells, n_genes)
        batch_size : Optional[int], default=None
            Batch size for processing large datasets
        training : bool, default=False
            Whether to use training mode (affects sampling)

        Returns
        -------
        jnp.ndarray
            Reconstructed data of shape (n_cells, n_genes)
        """
        if batch_size is None:
            # Process all cells at once
            reconstructed, _, _ = self.vae_model(counts, training=training)
            return reconstructed
        else:
            # Process in batches
            reconstructions = []
            for i in range(0, counts.shape[0], batch_size):
                batch = counts[i : i + batch_size]
                reconstructed, _, _ = self.vae_model(batch, training=training)
                reconstructions.append(reconstructed)
            return jnp.concatenate(reconstructions, axis=0)

    # --------------------------------------------------------------------------

    def compute_reconstruction_error(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        metric: str = "mse",
    ) -> float:
        """
        Compute reconstruction error between original and reconstructed data.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data of shape (n_cells, n_genes)
        batch_size : Optional[int], default=None
            Batch size for processing large datasets
        metric : str, default="mse"
            Error metric. Options: "mse", "mae", "rmse"

        Returns
        -------
        float
            Reconstruction error
        """
        # Reconstruct data
        reconstructed = self.reconstruct_data(counts, batch_size=batch_size)

        if metric == "mse":
            error = jnp.mean((counts - reconstructed) ** 2)
        elif metric == "mae":
            error = jnp.mean(jnp.abs(counts - reconstructed))
        elif metric == "rmse":
            error = jnp.sqrt(jnp.mean((counts - reconstructed) ** 2))
        else:
            raise ValueError(f"Unknown error metric: {metric}")

        return float(error)

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
        vae_model: VAE,
        original_counts: Optional[jnp.ndarray] = None,
    ) -> "ScribeVAEResults":
        """
        Create VAE results from existing SVI results.

        Parameters
        ----------
        svi_results : ScribeSVIResults
            Base SVI results
        vae_model : VAE
            Trained VAE model
        original_counts : Optional[jnp.ndarray], default=None
            Original count data used for training

        Returns
        -------
        ScribeVAEResults
            VAE-specific results object
        """
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
        )

        # Set VAE model after creation (but don't store it directly to avoid pickling issues)
        if vae_model is not None:
            vae_results._vae_model = vae_model

        # Store original counts for later use
        if original_counts is not None:
            vae_results._original_counts = original_counts

        return vae_results
