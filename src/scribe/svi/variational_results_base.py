"""
Abstract base class for variational results objects.

This runtime base type is intentionally lightweight: it defines the
viz-facing contract shared by ``ScribeSVIResults`` and ``ScribeVAEResults``
so dispatchers can target one variational type without duplicating
registrations per concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ScribeVariationalResults(ABC):
    """
    Runtime-dispatchable base for variational inference result containers.

    Notes
    -----
    This class avoids declaring dataclass fields so subclass constructor/field
    ordering remains unchanged.  Concrete result classes satisfy this contract
    via existing mixin methods and attributes.
    """

    @abstractmethod
    def get_posterior_samples(self, *args: Any, **kwargs: Any) -> Any:
        """
        Draw posterior parameter samples.

        Returns
        -------
        Any
            Posterior parameter samples in the class-specific format.
        """

    @abstractmethod
    def get_predictive_samples(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generate posterior predictive samples.

        Returns
        -------
        Any
            Predictive samples with a leading sample axis.
        """

    @abstractmethod
    def get_map_ppc_samples(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generate MAP-based predictive samples for plotting workflows.

        Returns
        -------
        Any
            MAP-like predictive samples.
        """

    @abstractmethod
    def get_map(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return MAP estimates (optionally canonicalized) for viz consumers.

        Returns
        -------
        Any
            Mapping or structure containing MAP parameter values.
        """

    @property
    @abstractmethod
    def layouts(self) -> Any:
        """
        Semantic layout metadata for parameter arrays.

        Returns
        -------
        Any
            Mapping of parameter keys to axis-layout metadata.
        """

    @abstractmethod
    def get_ppc_samples_biological(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generate biological-only posterior predictive samples.

        Returns
        -------
        Any
            Biological PPC samples, typically under ``predictive_samples``.
        """

    @abstractmethod
    def denoise_counts_map(self, *args: Any, **kwargs: Any) -> Any:
        """
        Denoise observed counts using MAP estimates.

        Returns
        -------
        Any
            Denoised count matrix.
        """

