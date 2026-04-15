"""Split-aware orchestration runner used by ``scribe-infer``.

This module is invoked when at least one selected data config defines
``split_by``. It expands the dataset into split-specific temporary configs,
launches one Hydra multirun, and routes each split job to
``scribe.cli.infer_runner``.

High-level flow
---------------
1. Parse forwarded Hydra overrides and split-specific launcher options.
2. Discover split values from ``adata.obs``.
3. Materialize temporary split configs under ``conf/data/_tmp_split_*``.
4. Run Hydra multirun with joblib or submitit launcher.
5. Clean up temporary config directory.

Primary user entrypoint remains:

    scribe-infer --config-path ./conf data=<dataset_with_split_by>
"""

import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scribe.data_loader import load_and_preprocess_anndata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_DIR = Path("conf")
CONF_DATA_DIR = DEFAULT_CONFIG_DIR / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_data_values(raw_value: str) -> list[str]:
    """Split a ``data=...`` argument into one or more config keys.

    Parameters
    ----------
    raw_value : str
        Raw value from the ``data=...`` argument.

    Returns
    -------
    list[str]
        Data config keys with whitespace trimmed and empty entries removed.
    """
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def _extract_config_options(
    argv: list[str],
) -> tuple[str, str, list[str]]:
    """Extract top-level config options from CLI tokens.

    Parameters
    ----------
    argv : list[str]
        Raw command-line arguments excluding executable/module name.

    Returns
    -------
    tuple[str, str, list[str]]
        A tuple containing:
        - ``config_path``: config root directory (default ``"./conf"``)
        - ``config_name``: top-level Hydra config name (default ``"config"``)
        - ``remaining_args``: all other tokens to parse as overrides
    """
    config_path = str(DEFAULT_CONFIG_DIR)
    config_name = "config"
    remaining_args: list[str] = []

    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--config-path" and idx + 1 < len(argv):
            config_path = argv[idx + 1]
            idx += 2
            continue
        if token == "--config-name" and idx + 1 < len(argv):
            config_name = argv[idx + 1]
            idx += 2
            continue
        remaining_args.append(token)
        idx += 1

    return config_path, config_name, remaining_args


def _detect_gpu_ids() -> list[str]:
    """Detect the physical CUDA GPU IDs visible to the current process.

    If ``CUDA_VISIBLE_DEVICES`` is set in the environment, its value is
    parsed to obtain the physical device IDs.  Otherwise JAX is queried
    for the number of GPUs and IDs ``["0", "1", ...]`` are returned.

    Returns
    -------
    list[str]
        Physical GPU ID strings (e.g. ``["2", "3"]``), or an empty list
        if no GPUs are available.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        # Respect the user's explicit GPU selection
        ids = [x.strip() for x in cvd.split(",") if x.strip()]
        if ids:
            return ids

    # Fall back to JAX device enumeration
    try:
        import jax  # noqa: E402 – import only when needed

        n = len(jax.devices("gpu"))
        return [str(i) for i in range(n)]
    except Exception:
        return []


def _parse_args(argv: list[str]) -> tuple[list[str], dict, dict, list[str]]:
    """Extract ``data=<name>``, ``data.*`` and ``split.*`` overrides.

    Parameters
    ----------
    argv : list[str]
        The raw command-line arguments (excluding the module invocation name).

    Returns
    -------
    data_names : list[str]
        The value(s) of the ``data=...`` argument, split on commas.
    data_overrides : dict
        Any ``data.<key>=<value>`` overrides provided on the command line
        (e.g. ``data.n_jobs=4``).
    split_overrides : dict
        Any ``split.<key>=<value>`` overrides used to control how this
        orchestrator launches child multirun jobs.
    forwarded_args : list[str]
        All remaining arguments that should be forwarded to the direct runner.
    """
    data_names: list[str] | None = None
    data_overrides: dict = {}
    split_overrides: dict = {}
    forwarded_args: list[str] = []

    for arg in argv:
        # Match data=<name> (but not data.<key>=<value>)
        m = re.match(r"^data=([^\s]+)$", arg)
        if m and "." not in arg.split("=", 1)[0]:
            data_names = _split_data_values(m.group(1))
            continue

        # Match data.<key>=<value>
        m_dot = re.match(r"^data\.(\w+)=(.+)$", arg)
        if m_dot:
            data_overrides[m_dot.group(1)] = m_dot.group(2)
            continue

        # Match split.<key>=<value> for orchestrator launch settings
        m_split = re.match(r"^split\.(\w+)=(.+)$", arg)
        if m_split:
            split_overrides[m_split.group(1)] = m_split.group(2)
            continue

        forwarded_args.append(arg)

    if data_names is None or not data_names:
        raise SystemExit(
            "ERROR: A data=<name> argument is required.\n"
            "Usage: scribe-infer --config-path ./conf data=<name> [overrides ...]"
        )

    return data_names, data_overrides, split_overrides, forwarded_args


def _load_data_config(
    data_name: str, conf_data_dir: Path = CONF_DATA_DIR
) -> dict:
    """Load a data YAML config from ``conf/data/<data_name>.yaml``.

    Parameters
    ----------
    data_name : str
        Config name, possibly including subdirectories (e.g.
        ``bleo_splits/bleo_study01``).

    Returns
    -------
    dict
        The parsed YAML contents.
    """
    yaml_path = conf_data_dir / f"{data_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Data config not found: {yaml_path}\n"
            f"Make sure conf/data/{data_name}.yaml exists."
        )
    return OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)


def _discover_covariate_values(
    data_path: str,
    split_by: str | list[str],
    filter_obs: dict[str, list[str]] | None = None,
) -> list[str] | list[tuple[str, ...]]:
    """Load data and return unique value(s) for one or more covariate columns.

    For a single column (``split_by`` is a ``str``), returns a sorted list of
    unique string values found in that column.

    For multiple columns (``split_by`` is a ``list[str]``), uses the DataFrame
    path: ``adata.obs[split_by].drop_duplicates()`` to enumerate all unique
    row combinations, returning a list of tuples sorted lexicographically.

    Parameters
    ----------
    data_path : str
        Path to the input data source. Any format supported by
        ``load_and_preprocess_anndata`` is accepted.
    split_by : str or list[str]
        Column name(s) in ``adata.obs``.
    filter_obs : dict[str, list[str]] or None, optional
        Declarative observation-level pre-filter applied before discovering
        split values.  Keys are column names in ``adata.obs``; values are
        lists of allowed values.  Only observations matching all conditions
        are considered when enumerating unique split_by values.

    Returns
    -------
    list[str] or list[tuple[str, ...]]
        Single-column: sorted unique string values.
        Multi-column: sorted unique tuples of string values, one per column.
    """
    # Resolve relative paths against the current working directory.
    abs_path = (
        data_path if os.path.isabs(data_path) else os.path.abspath(data_path)
    )

    # Reuse the shared data loader so split discovery remains format-aware
    # whenever supported input sources evolve.
    adata = load_and_preprocess_anndata(abs_path, return_jax=False)

    # Apply filter_obs pre-filter so that only relevant rows participate in
    # the unique-value discovery (avoids creating split jobs for combinations
    # that would be empty after filtering).
    if filter_obs:
        n_total = adata.shape[0]
        for col, allowed in filter_obs.items():
            if col not in adata.obs.columns:
                raise ValueError(
                    f"filter_obs column '{col}' not found in adata.obs.\n"
                    f"Available columns: {list(adata.obs.columns)}"
                )
            allowed_str = [str(v) for v in allowed]
            adata = adata[adata.obs[col].astype(str).isin(allowed_str)]
        if adata.shape[0] == 0:
            raise ValueError(
                "No observations remain after applying filter_obs: "
                f"{filter_obs}"
            )
        _ = n_total  # retained for debugging-friendly local context

    if isinstance(split_by, list):
        # Multi-column path: use the DataFrame to get all unique combinations
        missing = [c for c in split_by if c not in adata.obs.columns]
        if missing:
            raise ValueError(
                f"split_by columns {missing} not found in adata.obs.\n"
                f"Available columns: {list(adata.obs.columns)}"
            )
        combos = (
            adata.obs[split_by]
            .astype(str)
            .drop_duplicates()
            .sort_values(split_by)
            .itertuples(index=False, name=None)
        )
        values = list(combos)
        if not values:
            raise ValueError(
                f"No unique combinations found for columns {split_by}."
            )
        return values
    else:
        # Single-column path: existing behaviour
        if split_by not in adata.obs.columns:
            raise ValueError(
                f"split_by column '{split_by}' not found in adata.obs.\n"
                f"Available columns: {list(adata.obs.columns)}"
            )
        values = sorted(adata.obs[split_by].astype(str).unique())
        if len(values) == 0:
            raise ValueError(f"No values found in adata.obs['{split_by}'].")
        return values


def _sanitize_value(value: str) -> str:
    """Make a covariate value safe for use in file/directory names.

    Parameters
    ----------
    value : str
        Raw covariate value.

    Returns
    -------
    str
        Sanitized string with problematic characters replaced.
    """
    return (
        value.replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
    )


def _derive_output_prefix(data_name: str) -> str:
    """Derive a nested output prefix from a data config key.

    Parameters
    ----------
    data_name : str
        Data config key from ``data=...`` (e.g.
        ``panfibrosis/CKD/GSE140023_filter-none_split-disease``).

    Returns
    -------
    str
        Parent path portion of the key (e.g. ``panfibrosis/CKD``), or an empty
        string when no parent directories are present.
    """
    # Build parent path from slash-separated Hydra key, independent of OS
    # separator semantics.
    parent_parts = [part for part in data_name.split("/") if part][:-1]
    return "/".join(parent_parts)


def _make_tmp_split_dir(conf_data_dir: Path = CONF_DATA_DIR) -> Path:
    """Create a unique temporary config directory for this invocation.

    Returns
    -------
    Path
        Newly created directory under ``conf/data``.
    """
    # A unique directory per process avoids collisions when multiple split
    # orchestrators run concurrently.
    suffix = f"{os.getpid()}_{uuid4().hex[:8]}"
    tmp_dir = conf_data_dir / f"_tmp_split_{suffix}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


def _generate_tmp_yamls(
    data_cfg: dict,
    data_name: str,
    split_by: str | list[str],
    covariate_values: list[str] | list[tuple[str, ...]],
    gpu_ids: list[str],
    tmp_dir: Path,
) -> list[str]:
    """Write temporary data YAML configs for each covariate value (or combination).

    For a single ``split_by`` column each config receives scalar
    ``subset_column`` / ``subset_value`` fields.  For multiple columns each
    config receives list-valued ``subset_column`` / ``subset_value`` fields
    (one entry per column), which ``load_and_preprocess_anndata`` handles by
    ANDing per-column equality masks.

    Parameters
    ----------
    data_cfg : dict
        The original data config contents.
    data_name : str
        Source data config key passed via ``data=...``.
    split_by : str or list[str]
        The covariate column name(s).
    covariate_values : list[str] or list[tuple[str, ...]]
        Unique values (single column) or unique row combinations (multi-column)
        as returned by ``_discover_covariate_values``.
    gpu_ids : list[str]
        Physical GPU ID strings (e.g. ``["0", "1"]`` or ``["2", "3"]``).
        Jobs are assigned round-robin.  If empty, all configs get
        ``gpu_id: "0"``.
    tmp_dir : Path
        Per-invocation directory where split YAML files are written.

    Returns
    -------
    list[str]
        File stems of generated configs (without ``.yaml`` suffix).
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = _derive_output_prefix(data_name)
    original_name = data_cfg.get("name", "data")
    original_path = data_cfg.get("path", "")
    source_tag = _sanitize_value(data_name)

    # Resolve to absolute so each Hydra worker finds the file regardless
    # of its working directory.
    abs_path = (
        original_path
        if os.path.isabs(original_path)
        else os.path.abspath(original_path)
    )

    tmp_names: list[str] = []
    effective_gpu_ids = gpu_ids if gpu_ids else ["0"]

    for idx, value in enumerate(covariate_values):
        if isinstance(value, tuple):
            # Multi-column: join sanitized values for the config name
            safe_value = "__".join(_sanitize_value(str(v)) for v in value)
            subset_column = list(split_by)
            subset_value = [str(v) for v in value]
        else:
            # Single-column: existing behaviour
            safe_value = _sanitize_value(value)
            subset_column = split_by
            subset_value = value

        split_leaf_name = f"{original_name}_{safe_value}"
        # Use a filename-safe stem that is unique to the source config key.
        tmp_file_stem = f"{source_tag}__{split_leaf_name}"
        tmp_names.append(tmp_file_stem)

        # Build the temporary config
        tmp_cfg = {
            # Keep data.name slash-safe; nested grouping is injected via
            # hydra.sweep.subdir using data.output_prefix.
            "name": split_leaf_name,
            "output_prefix": output_prefix,
            "path": abs_path,
            "subset_column": subset_column,
            "subset_value": subset_value,
            "gpu_id": effective_gpu_ids[idx % len(effective_gpu_ids)],
        }

        # Carry over preprocessing and any other fields from the original
        # config, except the split-specific keys and the keys we've
        # already set.
        skip_keys = {
            "name",
            "path",
            "split_by",
            "n_jobs",
            "subset_column",
            "subset_value",
            "gpu_id",
        }
        for key, val in data_cfg.items():
            if key not in skip_keys:
                tmp_cfg[key] = val

        # Write with the @package directive so Hydra merges correctly
        yaml_path = tmp_dir / f"{tmp_file_stem}.yaml"
        cfg_obj = OmegaConf.create(tmp_cfg)
        with open(yaml_path, "w") as f:
            f.write("# @package data\n")
            f.write(OmegaConf.to_yaml(cfg_obj))

    return tmp_names


def _generate_passthrough_tmp_yaml(
    data_cfg: dict,
    data_name: str,
    tmp_dir: Path,
    gpu_id: str,
) -> str:
    """Write one temporary YAML for a dataset without split_by.

    Parameters
    ----------
    data_cfg : dict
        Original dataset config.
    data_name : str
        Source config key from ``data=...``.
    tmp_dir : Path
        Invocation-specific temp config directory.
    gpu_id : str
        GPU ID to assign for this passthrough config.

    Returns
    -------
    str
        Generated temp config stem (without ``.yaml``).
    """
    output_prefix = _derive_output_prefix(data_name)
    source_tag = _sanitize_value(data_name)
    original_name = data_cfg.get("name", "data")
    original_path = data_cfg.get("path", "")
    abs_path = (
        original_path
        if os.path.isabs(original_path)
        else os.path.abspath(original_path)
    )

    tmp_file_stem = f"{source_tag}__{_sanitize_value(original_name)}"
    tmp_cfg = {
        "name": original_name,
        "output_prefix": output_prefix,
        "path": abs_path,
        "gpu_id": gpu_id,
    }
    skip_keys = {"name", "path", "split_by", "n_jobs", "gpu_id"}
    for key, val in data_cfg.items():
        if key not in skip_keys:
            tmp_cfg[key] = val

    yaml_path = tmp_dir / f"{tmp_file_stem}.yaml"
    cfg_obj = OmegaConf.create(tmp_cfg)
    with open(yaml_path, "w") as f:
        f.write("# @package data\n")
        f.write(OmegaConf.to_yaml(cfg_obj))

    return tmp_file_stem


def _cleanup_tmp_dir(tmp_dir: Path) -> None:
    """Remove a temporary split config directory if it exists.

    Parameters
    ----------
    tmp_dir : Path
        Temporary directory created for this infer_split invocation.
    """
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def _build_joblib_multirun_command(
    data_list: str,
    n_jobs: int,
    forwarded_args: list[str],
    config_path: str = str(DEFAULT_CONFIG_DIR),
    config_name: str = "config",
) -> list[str]:
    """Build Hydra multirun command using the joblib launcher.

    Parameters
    ----------
    data_list : str
        Comma-separated ``data=...`` values targeting temporary split configs.
    n_jobs : int
        Number of joblib workers.
    forwarded_args : list[str]
        User arguments forwarded unchanged to the direct runner.

    Returns
    -------
    list[str]
        Command list suitable for ``subprocess.run``.
    """
    return [
        sys.executable,
        "-m",
        "scribe.cli.infer_runner",
        "--config-path",
        config_path,
        "--config-name",
        config_name,
        "-m",
        f"data={data_list}",
        "hydra.sweep.subdir='${data.output_prefix}/${data.name}/${model}/${inference.method}/${sanitize_dirname:${hydra:job.override_dirname},${dirname_aliases.aliases}}'",
        "hydra/launcher=joblib",
        f"hydra.launcher.n_jobs={n_jobs}",
        # GPU assignment is handled inside infer_runner via cfg.data.gpu_id
        # (set before JAX initialises CUDA devices).
        *forwarded_args,
    ]


def _build_submitit_multirun_command(
    data_list: str,
    n_jobs: int,
    split_overrides: dict,
    forwarded_args: list[str],
    config_path: str = str(DEFAULT_CONFIG_DIR),
    config_name: str = "config",
) -> list[str]:
    """Build Hydra multirun command using the submitit Slurm launcher.

    Parameters
    ----------
    data_list : str
        Comma-separated ``data=...`` values targeting temporary split configs.
    n_jobs : int
        Number of split jobs to allow in parallel (default fallback).
    split_overrides : dict
        Orchestrator ``split.*`` launch settings parsed from CLI.
    forwarded_args : list[str]
        User arguments forwarded unchanged to the direct runner.

    Returns
    -------
    list[str]
        Command list suitable for ``subprocess.run``.
    """
    array_parallelism = int(split_overrides.get("array_parallelism", n_jobs))
    cpus_per_task = int(split_overrides.get("cpus_per_task", 2))
    mem_gb = int(split_overrides.get("mem_gb", 16))
    timeout_min = int(split_overrides.get("timeout_min", 240))
    partition = split_overrides.get("partition", "base")
    account = split_overrides.get("account", "hybrid-modeling")
    job_name = split_overrides.get("job_name", "scribe_infer_split")
    submitit_folder = split_overrides.get(
        "submitit_folder", "slurm_logs/submitit/%j"
    )

    return [
        sys.executable,
        "-m",
        "scribe.cli.infer_runner",
        "--config-path",
        config_path,
        "--config-name",
        config_name,
        "-m",
        f"data={data_list}",
        "hydra.sweep.subdir='${data.output_prefix}/${data.name}/${model}/${inference.method}/${sanitize_dirname:${hydra:job.override_dirname},${dirname_aliases.aliases}}'",
        "hydra/launcher=submitit_slurm",
        "hydra.launcher.nodes=1",
        "hydra.launcher.tasks_per_node=1",
        "hydra.launcher.gpus_per_node=1",
        f"hydra.launcher.cpus_per_task={cpus_per_task}",
        f"hydra.launcher.mem_gb={mem_gb}",
        f"hydra.launcher.partition={partition}",
        f"hydra.launcher.account={account}",
        f"hydra.launcher.timeout_min={timeout_min}",
        f"hydra.launcher.array_parallelism={array_parallelism}",
        f"hydra.launcher.name={job_name}",
        f"hydra.launcher.submitit_folder={submitit_folder}",
        *forwarded_args,
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the covariate-split inference orchestrator."""
    console = Console()
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_blue]SCRIBE COVARIATE-SPLIT INFERENCE[/bold bright_blue]",
            border_style="bright_blue",
        )
    )

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments
    # ------------------------------------------------------------------
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_path, config_name, parseable_argv = _extract_config_options(raw_argv)
    conf_data_dir = Path(config_path) / "data"
    data_names, data_overrides, split_overrides, forwarded_args = _parse_args(
        parseable_argv
    )
    console.print(
        f"[dim]Data configs:[/dim] [cyan]{', '.join(data_names)}[/cyan]"
    )

    # ------------------------------------------------------------------
    # 2. Load the data config YAML
    # ------------------------------------------------------------------
    loaded_data_cfgs: list[tuple[str, dict]] = []
    for data_name in data_names:
        data_cfg = _load_data_config(data_name, conf_data_dir=conf_data_dir)
        for key, val in data_overrides.items():
            data_cfg[key] = val
        loaded_data_cfgs.append((data_name, data_cfg))

    # ------------------------------------------------------------------
    # 4. Determine parallelism settings
    # ------------------------------------------------------------------
    gpu_ids = _detect_gpu_ids()
    n_gpus = len(gpu_ids)
    console.print(
        f"[dim]GPUs detected:[/dim] [cyan]{n_gpus}[/cyan]"
        + (f" (IDs: {', '.join(gpu_ids)})" if gpu_ids else "")
    )

    # n_jobs: user override > max(data_cfg n_jobs) > number of GPUs > 1
    cfg_n_jobs = [
        cfg.get("n_jobs")
        for _, cfg in loaded_data_cfgs
        if cfg.get("n_jobs") is not None
    ]
    n_jobs_raw = data_overrides.get("n_jobs")
    if n_jobs_raw is None and cfg_n_jobs:
        n_jobs_raw = max(int(v) for v in cfg_n_jobs)
    if n_jobs_raw is not None:
        n_jobs = int(n_jobs_raw)
    else:
        n_jobs = max(n_gpus, 1)
    console.print(f"[dim]Parallel jobs:[/dim] [cyan]{n_jobs}[/cyan]")

    # ------------------------------------------------------------------
    # 5. Generate temporary data YAML configs
    # ------------------------------------------------------------------
    # Determine launcher mode before writing split configs because GPU
    # assignment strategy differs by launcher:
    # - joblib: one process may see many GPUs, so round-robin physical IDs.
    # - submitit_slurm: each task gets exactly one allocated GPU and must use
    #   local device index "0" inside that isolated environment.
    launcher_mode = split_overrides.get("launcher", "joblib")
    assigned_gpu_ids = ["0"] if launcher_mode == "submitit_slurm" else gpu_ids
    if launcher_mode == "submitit_slurm":
        console.print(
            "[dim]Submitit launcher detected; forcing split configs to "
            "use local gpu_id=0 per SLURM task.[/dim]"
        )

    console.print()
    console.print(
        Panel.fit(
            "[bold bright_cyan]GENERATING SPLIT CONFIGS[/bold bright_cyan]",
            border_style="bright_cyan",
        )
    )

    tmp_dir = _make_tmp_split_dir(conf_data_dir=conf_data_dir)
    tmp_names: list[str] = []
    details_table = Table(
        title="Dataset expansion",
        show_header=True,
        header_style="bold magenta",
    )
    details_table.add_column("Dataset", style="cyan")
    details_table.add_column("Mode", style="yellow")
    details_table.add_column("Jobs", justify="right", style="green")

    for data_name, data_cfg in loaded_data_cfgs:
        split_by = data_cfg.get("split_by")
        if split_by is None:
            passthrough_name = _generate_passthrough_tmp_yaml(
                data_cfg=data_cfg,
                data_name=data_name,
                tmp_dir=tmp_dir,
                gpu_id=assigned_gpu_ids[0] if assigned_gpu_ids else "0",
            )
            tmp_names.append(passthrough_name)
            details_table.add_row(data_name, "passthrough", "1")
            continue

        filter_obs = data_cfg.get("filter_obs")
        if filter_obs:
            filter_parts = [
                f"{col} in {vals}" for col, vals in filter_obs.items()
            ]
            console.print(
                f"[bold yellow]Pre-filter ({data_name}):[/bold yellow] "
                f"[cyan]{', '.join(filter_parts)}[/cyan]"
            )

        console.print(
            f"[dim]Loading data for split discovery:[/dim] [cyan]{data_name}[/cyan]"
        )
        covariate_values = _discover_covariate_values(
            data_cfg["path"], split_by, filter_obs=filter_obs
        )
        console.print(
            f"[green]✓[/green] [cyan]{data_name}[/cyan] -> "
            f"[bold]{len(covariate_values)}[/bold] split values"
        )
        created = _generate_tmp_yamls(
            data_cfg=data_cfg,
            data_name=data_name,
            split_by=split_by,
            covariate_values=covariate_values,
            gpu_ids=assigned_gpu_ids,
            tmp_dir=tmp_dir,
        )
        tmp_names.extend(created)
        details_table.add_row(data_name, "split", str(len(created)))

    console.print()
    console.print(details_table)
    tmp_dir_name = tmp_dir.name
    for name in tmp_names:
        console.print(
            f"  [dim]Created:[/dim] conf/data/{tmp_dir_name}/{name}.yaml"
        )

    # ------------------------------------------------------------------
    # 6. Build and launch Hydra multirun command
    # ------------------------------------------------------------------
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_yellow]LAUNCHING HYDRA MULTIRUN[/bold bright_yellow]",
            border_style="bright_yellow",
        )
    )

    data_list = ",".join(f"{tmp_dir_name}/{n}" for n in tmp_names)

    # Default launcher preserves prior behavior for direct infer_split usage.
    if launcher_mode == "submitit_slurm":
        # Fail fast with a clear error when submitit is requested but missing.
        try:
            import hydra_plugins.hydra_submitit_launcher  # noqa: F401
        except ImportError:
            raise SystemExit(
                "ERROR: split.launcher=submitit_slurm requested but Hydra "
                "submitit launcher is not installed.\n"
                "Install it with: uv add --dev hydra-submitit-launcher"
            )
        cmd = _build_submitit_multirun_command(
            data_list=data_list,
            n_jobs=n_jobs,
            split_overrides=split_overrides,
            forwarded_args=forwarded_args,
            config_path=config_path,
            config_name=config_name,
        )
    else:
        cmd = _build_joblib_multirun_command(
            data_list=data_list,
            n_jobs=n_jobs,
            forwarded_args=forwarded_args,
            config_path=config_path,
            config_name=config_name,
        )

    console.print("[dim]Command:[/dim]")
    # Escape brackets for Rich so list values like [mu,phi] aren't
    # swallowed as markup tags.
    cmd_str = " ".join(cmd).replace("[", "\\[")
    console.print(f"  [cyan]{cmd_str}[/cyan]")
    console.print()

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            console.print(
                f"[red]Hydra multirun exited with code "
                f"{result.returncode}[/red]"
            )
            console.print(
                "[yellow]Some split jobs failed.[/yellow] "
                "[dim]Inspect each failed run directory for "
                "`FAILED_MIN_CELLS.json` (when min-cells guard is enabled).[/dim]"
            )
            sys.exit(result.returncode)
    finally:
        # ------------------------------------------------------------------
        # 7. Cleanup
        # ------------------------------------------------------------------
        console.print()
        if launcher_mode == "submitit_slurm":
            # Submitit workers can start after this parent process exits; keep
            # temp split configs available so Hydra can compose data=... keys.
            console.print(
                "[yellow]Skipping temporary config cleanup for submitit mode.[/yellow]"
            )
            console.print(
                "[dim]Temporary configs retained at:[/dim] "
                f"[cyan]{tmp_dir}[/cyan]"
            )
        else:
            console.print("[dim]Cleaning up temporary configs...[/dim]")
            _cleanup_tmp_dir(tmp_dir)
            console.print("[green]✓[/green] Temporary configs removed.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_green]✓ ALL SPLITS COMPLETED SUCCESSFULLY!"
            "[/bold bright_green]",
            border_style="bright_green",
        )
    )
    console.print(
        f"[dim]Executed {len(tmp_names)} total dataset jobs "
        f"(split + passthrough).[/dim]"
    )
    console.print()


if __name__ == "__main__":
    main()
