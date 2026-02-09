"""
Categorical covariate embedding for conditioning neural network modules.

Provides a reusable Flax Linen module that converts integer-coded categorical
covariates (e.g., batch ID, donor ID, experimental condition) into learned dense
embeddings. The resulting embedding vector can be concatenated to the input of
any encoder, decoder, or flow conditioner to make its output covariate-aware.

Classes
-------
CovariateSpec
    Specification for a single categorical covariate.
CovariateEmbedding
    Flax Linen module that embeds and concatenates multiple covariates.

Examples
--------
>>> specs = [
...     CovariateSpec("batch", num_categories=4, embedding_dim=8),
...     CovariateSpec("donor", num_categories=10, embedding_dim=8),
... ]
>>> embedder = CovariateEmbedding(covariate_specs=specs)
>>> covariates = {"batch": jnp.array([0, 1, 2]), "donor": jnp.array([3, 5, 7])}
>>> params = embedder.init(jax.random.PRNGKey(0), covariates)
>>> emb = embedder.apply(params, covariates)  # shape (3, 16)
"""

from dataclasses import dataclass
from typing import Dict, List

import jax.numpy as jnp
from flax import linen as nn

# ==============================================================================
# Covariate Specification Class
# ==============================================================================


@dataclass(frozen=True)
class CovariateSpec:
    """Specification for a single categorical covariate.

    Parameters
    ----------
    name : str
        Identifier matching the key in the covariates dict
        (e.g., ``"batch"``, ``"donor"``).
    num_categories : int
        Number of unique categories. Integer IDs must be in
        ``[0, num_categories)``.
    embedding_dim : int
        Dimensionality of the learned embedding vector.
    """

    name: str
    num_categories: int
    embedding_dim: int

    def __post_init__(self):
        if self.num_categories < 1:
            raise ValueError(
                f"num_categories must be >= 1, got {self.num_categories} "
                f"for covariate '{self.name}'"
            )
        if self.embedding_dim < 1:
            raise ValueError(
                f"embedding_dim must be >= 1, got {self.embedding_dim} "
                f"for covariate '{self.name}'"
            )


# ==============================================================================
# Covariate Embedding Module
# ==============================================================================


class CovariateEmbedding(nn.Module):
    """Embed multiple categorical covariates and concatenate.

    Each covariate is mapped through its own ``nn.Embed`` lookup table,
    producing a dense vector. All embeddings are concatenated along the
    last axis to form a single conditioning vector.

    Parameters
    ----------
    covariate_specs : List[CovariateSpec]
        One specification per categorical covariate.

    Examples
    --------
    >>> specs = [CovariateSpec("batch", 4, 8)]
    >>> embedder = CovariateEmbedding(covariate_specs=specs)
    >>> ids = {"batch": jnp.array([0, 1, 3])}
    >>> params = embedder.init(jax.random.PRNGKey(0), ids)
    >>> emb = embedder.apply(params, ids)  # (3, 8)
    """

    covariate_specs: List[CovariateSpec]

    @nn.compact
    def __call__(self, covariates: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Embed covariates and concatenate.

        Parameters
        ----------
        covariates : Dict[str, jnp.ndarray]
            Maps covariate name to integer ID array of shape ``(...)``.

        Returns
        -------
        jnp.ndarray
            Concatenated embeddings, shape ``(..., total_embedding_dim)``.
        """
        embeddings = []
        for spec in self.covariate_specs:
            ids = covariates[spec.name]
            emb = nn.Embed(
                num_embeddings=spec.num_categories,
                features=spec.embedding_dim,
                name=f"embed_{spec.name}",
            )(ids)
            embeddings.append(emb)
        return jnp.concatenate(embeddings, axis=-1)

    # --------------------------------------------------------------------------

    @property
    def total_embedding_dim(self) -> int:
        """Total dimensionality of the concatenated embedding vector."""
        return sum(s.embedding_dim for s in self.covariate_specs)
