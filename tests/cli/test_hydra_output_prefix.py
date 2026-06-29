"""Tests for Hydra output-prefix callback and resolver.

Run from the scribe-processing uv environment, which installs ``scribe[hydra]``
as an editable dependency of ``../scribe``.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from scribe.cli.hydra_callbacks import OutputPrefixCallback
from scribe.cli.infer_runner import _nested_output_prefix_resolver
from scribe.cli.output_layout import apply_output_prefix_to_config


@pytest.fixture(autouse=True)
def _clear_hydra_singletons():
    """Keep Hydra's process-wide singletons from leaking into other tests.

    ``test_compose_resolves_nested_output_prefix_in_run_dir`` calls
    ``HydraConfig.instance().set_config(...)``.  If left set, ``HydraConfig``
    stays "initialized" for the rest of the process, which flips
    ``hydra.utils.to_absolute_path`` into its ``get_original_cwd()`` branch in
    unrelated downstream tests (notably the infer-runner CLI tests, whose
    mocked ``HydraConfig.get`` has no ``runtime.cwd``).  Reset both singletons
    after every test in this module.
    """
    yield
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.hydra_config import HydraConfig

    GlobalHydra.instance().clear()
    HydraConfig.instance().cfg = None


def _mock_hydra_runtime(data_choice: str, data_cfg: dict | None = None):
    """Build Hydra node + parent config for resolver/callback tests.

    Parameters
    ----------
    data_choice : str
        Selected data config key stored in ``runtime.choices.data``.
    data_cfg : dict or None, optional
        Optional ``data`` node contents for the composed parent config.

    Returns
    -------
    tuple[OmegaConf, OmegaConf]
        ``(hydra_node, parent_cfg)`` pair wired with parent pointers.
    """
    parent_cfg = OmegaConf.create(
        {
            "data": data_cfg or {"name": "foo_dataset"},
            "hydra": {
                "runtime": {
                    "choices": {
                        "data": data_choice,
                    }
                }
            },
        }
    )
    hydra_node = parent_cfg.hydra
    return hydra_node, parent_cfg


def test_nested_output_prefix_resolver_derives_from_runtime_choice():
    """Resolver should derive nested prefix from Hydra runtime data choice."""
    hydra_node, _ = _mock_hydra_runtime("panfibrosis/cell_type_genecorr/CKD/foo")
    with patch(
        "hydra.core.hydra_config.HydraConfig.get",
        return_value=hydra_node,
    ):
        assert (
            _nested_output_prefix_resolver()
            == "panfibrosis/cell_type_genecorr/CKD/"
        )


def test_nested_output_prefix_resolver_prefers_explicit_prefix():
    """Explicit data.output_prefix should override runtime choice derivation."""
    hydra_node, _ = _mock_hydra_runtime(
        "panfibrosis/cell_type_genecorr/CKD/foo",
        data_cfg={"name": "foo_dataset", "output_prefix": "custom/prefix"},
    )
    with patch(
        "hydra.core.hydra_config.HydraConfig.get",
        return_value=hydra_node,
    ):
        assert _nested_output_prefix_resolver() == "custom/prefix/"


def test_nested_output_prefix_resolver_returns_empty_for_flat_choice():
    """Flat data choices should resolve to an empty prefix segment."""
    hydra_node, _ = _mock_hydra_runtime(
        "my_dataset",
        data_cfg={"name": "my_dataset"},
    )
    with patch(
        "hydra.core.hydra_config.HydraConfig.get",
        return_value=hydra_node,
    ):
        assert _nested_output_prefix_resolver() == ""


def test_apply_output_prefix_to_config_sets_derived_prefix():
    """Callback helper should materialize output_prefix on single-run configs."""
    _, parent_cfg = _mock_hydra_runtime("panfibrosis/cell_type_genecorr/CKD/foo")
    with patch(
        "hydra.core.hydra_config.HydraConfig.get",
        return_value=parent_cfg.hydra,
    ):
        apply_output_prefix_to_config(parent_cfg)
    assert parent_cfg.data.output_prefix == "panfibrosis/cell_type_genecorr/CKD"


def test_apply_output_prefix_to_config_does_not_override_explicit_value():
    """Explicit output_prefix values should remain unchanged."""
    cfg = OmegaConf.create(
        {
            "data": {
                "name": "foo_dataset",
                "output_prefix": "custom/prefix",
            }
        }
    )
    apply_output_prefix_to_config(cfg)
    assert cfg.data.output_prefix == "custom/prefix"


def test_output_prefix_callback_delegates_to_apply_helper():
    """Hydra callback should invoke shared output-prefix application logic."""
    cfg = OmegaConf.create({"data": {"name": "foo_dataset"}})
    callback = OutputPrefixCallback()

    with patch(
        "scribe.cli.hydra_callbacks.apply_output_prefix_to_config"
    ) as mock_apply:
        callback.on_run_start(cfg)
        mock_apply.assert_called_once_with(cfg)


def test_compose_resolves_nested_output_prefix_in_run_dir(tmp_path: Path):
    """Composed Hydra config should include nested prefix in run.dir."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    conf_dir = tmp_path / "conf"
    data_dir = conf_dir / "data" / "panfibrosis" / "CKD"
    data_dir.mkdir(parents=True)

    (data_dir / "foo.yaml").write_text(
        "\n".join(
            [
                "# @package data",
                'name: "foo_dataset"',
                'path: "/tmp/mock.h5ad"',
            ]
        )
    )

    config_yaml = "\n".join(
        [
            "defaults:",
            "  - data: panfibrosis/CKD/foo",
            "  - paths: paths",
            "  - inference: svi",
            "  - dirname_aliases: default",
            "  - _self_",
            "",
            "model: nbdm",
            "parameterization: canonical",
            "",
            "hydra:",
            "  callbacks:",
            "    output_prefix:",
            "      _target_: scribe.cli.hydra_callbacks.OutputPrefixCallback",
            "  run:",
            "    dir: ${paths.outputs_dir}/${nested_output_prefix:}${data.name}/${model}/${inference.method}/run",
            "",
        ]
    )
    (conf_dir / "config.yaml").write_text(config_yaml)
    (conf_dir / "paths").mkdir()
    (conf_dir / "paths" / "paths.yaml").write_text("outputs_dir: outputs\n")
    (conf_dir / "inference").mkdir()
    (conf_dir / "inference" / "svi.yaml").write_text("method: svi\n")
    (conf_dir / "dirname_aliases").mkdir()
    (conf_dir / "dirname_aliases" / "default.yaml").write_text(
        "dirname_aliases:\n  aliases: {}\n"
    )

    # Register resolver used by the composed config template.
    import scribe.cli.infer_runner  # noqa: F401

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config", return_hydra_config=True)

    HydraConfig.instance().set_config(cfg)
    assert cfg.hydra.run.dir == "outputs/panfibrosis/CKD/foo_dataset/nbdm/svi/run"
