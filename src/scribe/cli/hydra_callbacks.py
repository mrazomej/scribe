"""Hydra callbacks used by SCRIBE inference entry points."""

from __future__ import annotations

from omegaconf import DictConfig

from hydra.experimental.callback import Callback

from .output_layout import apply_output_prefix_to_config


class OutputPrefixCallback(Callback):
    """Populate ``data.output_prefix`` before Hydra resolves output directories.

    Hydra 1.3 resolves ``hydra.run.dir`` inside ``run_job`` after
    ``on_run_start`` but before ``on_job_start``. Setting
    ``data.output_prefix`` here ensures single-run configs record the derived
    prefix while multirun jobs still rely on the ``nested_output_prefix``
    resolver at interpolation time.
    """

    def on_run_start(self, config: DictConfig, **kwargs: object) -> None:
        """Apply nested output prefix derivation for single-run launches.

        Parameters
        ----------
        config : DictConfig
            Fully composed Hydra configuration.
        **kwargs : object
            Additional Hydra callback kwargs (unused).
        """
        apply_output_prefix_to_config(config)
