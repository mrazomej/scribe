"""Serialization and compatibility properties for Laplace results.

The methods in this module ensure Laplace results remain compatible with the
same plotting/cache contracts used by other SCRIBE result classes, while also
providing pickle-safe round-tripping for model configuration objects.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp


class SerializationResultsMixin:
    """Mixin providing pickling hooks and sample-cache compatibility slots."""

    @property
    def predictive_samples(self) -> Optional[jnp.ndarray]:
        """Return cached predictive samples from metadata storage.

        Returns
        -------
        jnp.ndarray or None
            Cached predictive sample tensor when previously populated.
        """
        return self.metadata.get("predictive_samples")

    @predictive_samples.setter
    def predictive_samples(self, value: Optional[jnp.ndarray]) -> None:
        """Store or clear cached predictive samples in metadata.

        Parameters
        ----------
        value : jnp.ndarray or None
            New cached predictive samples. ``None`` removes the key.
        """
        if value is None:
            self.metadata.pop("predictive_samples", None)
        else:
            self.metadata["predictive_samples"] = value

    @property
    def posterior_samples(self) -> Optional[Dict[str, jnp.ndarray]]:
        """Return cached posterior samples from metadata storage.

        Returns
        -------
        Dict[str, jnp.ndarray] or None
            Cached posterior sample dictionary when available.
        """
        return self.metadata.get("posterior_samples")

    @posterior_samples.setter
    def posterior_samples(self, value: Optional[Dict[str, jnp.ndarray]]) -> None:
        """Store or clear cached posterior samples in metadata.

        Parameters
        ----------
        value : Dict[str, jnp.ndarray] or None
            New cached posterior samples. ``None`` removes the key.
        """
        if value is None:
            self.metadata.pop("posterior_samples", None)
        else:
            self.metadata["posterior_samples"] = value

    def __getstate__(self) -> Dict[str, Any]:
        """Build pickle-safe state dictionary for Laplace results.

        Returns
        -------
        Dict[str, Any]
            State payload with a serialization-safe ``model_config``.
        """
        from ..svi.vae_results import make_model_config_pickle_safe

        state = dict(self.__dict__)
        state["model_config"] = make_model_config_pickle_safe(
            state.get("model_config")
        )
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore object state from a pickle payload.

        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary produced by :meth:`__getstate__`.
        """
        self.__dict__.update(state)

