"""
infer_split.py

Orchestrator script for covariate-split parallel inference within the SCRIBE
framework.  Given a data configuration that includes a ``split_by`` field,
this script:

1. Loads the h5ad file once to discover the unique values of the covariate
   column specified by ``split_by`` in ``adata.obs``.
2. Generates lightweight temporary data YAML configurations---one per unique
   covariate value---in ``conf/data/_tmp_split/``.  Each temporary config
   points to the **same** original h5ad file but adds ``subset_column`` and
   ``subset_value`` fields so that ``load_and_preprocess_anndata`` filters
   the data in memory.
3. Launches a single Hydra **multirun** (``-m``) invocation of ``infer.py``
   with the ``hydra/launcher=joblib`` backend.  This delegates all
   parallelism to Hydra and its joblib launcher, which spawns one worker
   process per covariate value.
4. Assigns GPUs round-robin via the Hydra per-job ``env_set`` mechanism so
   that each worker's ``CUDA_VISIBLE_DEVICES`` is set before JAX
   initialises.
5. Cleans up the temporary YAML files after the multirun completes (or on
   failure).

The interface mirrors ``infer.py``: all Hydra overrides are forwarded
verbatim, so a user can simply swap ``python infer.py`` for
``python infer_split.py`` when their data config contains ``split_by``.


Data YAML requirements
----------------------
The data configuration YAML must include the following additional fields
(beyond the standard ``name`` / ``path`` / ``preprocessing``):

    split_by : str or list[str]
        Column name(s) in ``adata.obs`` whose unique values (or unique
        combinations for multiple columns) define the data subsets.
    n_jobs : int or null, optional
        Number of parallel joblib workers.  Defaults to the number of
        visible CUDA GPUs, falling back to 1 if none are detected.


Typical usage
-------------

    # Split by treatment column, auto-detect GPUs for parallelism
    $ python infer_split.py data=bleo_study01 variable_capture=true \\
          annotation_key=subclass-l1

    # Explicitly set the number of parallel workers
    $ python infer_split.py data=bleo_study01 data.n_jobs=4 \\
          variable_capture=true

This produces output directories like::

    outputs/bleo_study01_bleomycin/nbvcp/svi/...
    outputs/bleo_study01_control/nbvcp/svi/...
"""

import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

import scanpy as sc
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONF_DATA_DIR = Path("conf") / "data"
TMP_SPLIT_DIR = CONF_DATA_DIR / "_tmp_split"
SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _parse_args(argv: list[str]) -> tuple[str, dict, dict, list[str]]:
    """Extract ``data=<name>``, ``data.*`` and ``split.*`` overrides.

    Parameters
    ----------
    argv : list[str]
        The raw command-line arguments (excluding the script name).

    Returns
    -------
    data_name : str
        The value of the ``data=<name>`` argument.
    data_overrides : dict
        Any ``data.<key>=<value>`` overrides provided on the command line
        (e.g. ``data.n_jobs=4``).
    split_overrides : dict
        Any ``split.<key>=<value>`` overrides used to control how this
        orchestrator launches child multirun jobs.
    forwarded_args : list[str]
        All remaining arguments that should be forwarded to ``infer.py``.
    """
    data_name: str | None = None
    data_overrides: dict = {}
    split_overrides: dict = {}
    forwarded_args: list[str] = []

    for arg in argv:
        # Match data=<name> (but not data.<key>=<value>)
        m = re.match(r"^data=([^\s]+)$", arg)
        if m and "." not in arg.split("=", 1)[0]:
            data_name = m.group(1)
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

    if data_name is None:
        raise SystemExit(
            "ERROR: A data=<name> argument is required.\n"
            "Usage: python infer_split.py data=<name> [overrides ...]"
        )

    return data_name, data_overrides, split_overrides, forwarded_args


def _load_data_config(data_name: str) -> dict:
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
    yaml_path = CONF_DATA_DIR / f"{data_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Data config not found: {yaml_path}\n"
            f"Make sure conf/data/{data_name}.yaml exists."
        )
    return OmegaConf.to_container(
        OmegaConf.load(yaml_path), resolve=True
    )


def _discover_covariate_values(
    data_path: str,
    split_by: str | list[str],
    filter_obs: dict[str, list[str]] | None = None,
) -> list[str] | list[tuple[str, ...]]:
    """Load h5ad and return unique value(s) for one or more covariate columns.

    For a single column (``split_by`` is a ``str``), returns a sorted list of
    unique string values found in that column.

    For multiple columns (``split_by`` is a ``list[str]``), uses the DataFrame
    path: ``adata.obs[split_by].drop_duplicates()`` to enumerate all unique
    row combinations, returning a list of tuples sorted lexicographically.

    Parameters
    ----------
    data_path : str
        Path to the h5ad (or CSV) file.
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
    # Resolve relative paths against the project root
    abs_path = (
        data_path
        if os.path.isabs(data_path)
        else str(SCRIPT_DIR / data_path)
    )

    adata = sc.read_h5ad(abs_path)

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
        print(
            f"filter_obs applied: {adata.shape[0]:,}/{n_total:,} cells "
            f"retained after filtering by {filter_obs}"
        )

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
            raise ValueError(
                f"No values found in adata.obs['{split_by}']."
            )
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


def _generate_tmp_yamls(
    data_cfg: dict,
    split_by: str | list[str],
    covariate_values: list[str] | list[tuple[str, ...]],
    gpu_ids: list[str],
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
    split_by : str or list[str]
        The covariate column name(s).
    covariate_values : list[str] or list[tuple[str, ...]]
        Unique values (single column) or unique row combinations (multi-column)
        as returned by ``_discover_covariate_values``.
    gpu_ids : list[str]
        Physical GPU ID strings (e.g. ``["0", "1"]`` or ``["2", "3"]``).
        Jobs are assigned round-robin.  If empty, all configs get
        ``gpu_id: "0"``.

    Returns
    -------
    list[str]
        Names of the generated configs (relative to ``conf/data/``),
        without the ``.yaml`` suffix.  These can be passed to
        ``data=_tmp_split/<name>`` in the Hydra command.
    """
    TMP_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    original_name = data_cfg.get("name", "data")
    original_path = data_cfg.get("path", "")

    # Resolve to absolute so each Hydra worker finds the file regardless
    # of its working directory.
    abs_path = (
        original_path
        if os.path.isabs(original_path)
        else str(SCRIPT_DIR / original_path)
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

        tmp_name = f"{original_name}_{safe_value}"
        tmp_names.append(tmp_name)

        # Build the temporary config
        tmp_cfg = {
            "name": tmp_name,
            "path": abs_path,
            "subset_column": subset_column,
            "subset_value": subset_value,
            "gpu_id": effective_gpu_ids[idx % len(effective_gpu_ids)],
        }

        # Carry over preprocessing and any other fields from the original
        # config, except the split-specific keys and the keys we've
        # already set.
        skip_keys = {
            "name", "path", "split_by", "n_jobs",
            "subset_column", "subset_value", "gpu_id",
        }
        for key, val in data_cfg.items():
            if key not in skip_keys:
                tmp_cfg[key] = val

        # Write with the @package directive so Hydra merges correctly
        yaml_path = TMP_SPLIT_DIR / f"{tmp_name}.yaml"
        cfg_obj = OmegaConf.create(tmp_cfg)
        with open(yaml_path, "w") as f:
            f.write("# @package data\n")
            f.write(OmegaConf.to_yaml(cfg_obj))

    return tmp_names


def _cleanup_tmp_dir() -> None:
    """Remove the temporary split config directory if it exists."""
    if TMP_SPLIT_DIR.exists():
        shutil.rmtree(TMP_SPLIT_DIR)


def _build_joblib_multirun_command(
    data_list: str, n_jobs: int, forwarded_args: list[str]
) -> list[str]:
    """Build Hydra multirun command using the joblib launcher.

    Parameters
    ----------
    data_list : str
        Comma-separated ``data=...`` values targeting temporary split configs.
    n_jobs : int
        Number of joblib workers.
    forwarded_args : list[str]
        User arguments forwarded unchanged to ``infer.py``.

    Returns
    -------
    list[str]
        Command list suitable for ``subprocess.run``.
    """
    return [
        sys.executable,
        str(SCRIPT_DIR / "infer.py"),
        "-m",
        f"data={data_list}",
        "hydra/launcher=joblib",
        f"hydra.launcher.n_jobs={n_jobs}",
        # GPU assignment is handled inside infer.py via cfg.data.gpu_id
        # (set before JAX initialises CUDA devices).
        *forwarded_args,
    ]


def _build_submitit_multirun_command(
    data_list: str,
    n_jobs: int,
    split_overrides: dict,
    forwarded_args: list[str],
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
        User arguments forwarded unchanged to ``infer.py``.

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
        str(SCRIPT_DIR / "infer.py"),
        "-m",
        f"data={data_list}",
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


def main() -> None:
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
    data_name, data_overrides, split_overrides, forwarded_args = _parse_args(
        sys.argv[1:]
    )
    console.print(
        f"[dim]Data config:[/dim] [cyan]{data_name}[/cyan]"
    )

    # ------------------------------------------------------------------
    # 2. Load the data config YAML
    # ------------------------------------------------------------------
    data_cfg = _load_data_config(data_name)

    # Apply any data.* overrides from the CLI
    for key, val in data_overrides.items():
        data_cfg[key] = val

    split_by = data_cfg.get("split_by")
    if split_by is None:
        raise SystemExit(
            "ERROR: The data config must include a 'split_by' field "
            "specifying the .obs column to split on.\n"
            f"Config loaded from: conf/data/{data_name}.yaml"
        )

    console.print(
        f"[dim]Split-by column:[/dim] [cyan]{split_by}[/cyan]"
    )

    # ------------------------------------------------------------------
    # 3. Discover covariate values
    # ------------------------------------------------------------------
    filter_obs = data_cfg.get("filter_obs")
    if filter_obs:
        # Format each filter condition as "column in [values]" for readability
        filter_parts = [
            f"{col} in {vals}" for col, vals in filter_obs.items()
        ]
        console.print(
            f"[bold yellow]Pre-filter (filter_obs):[/bold yellow] "
            f"[cyan]{', '.join(filter_parts)}[/cyan]"
        )

    console.print("[dim]Loading data to discover covariate values...[/dim]")
    data_path = data_cfg["path"]
    covariate_values = _discover_covariate_values(
        data_path, split_by, filter_obs=filter_obs
    )
    console.print(
        f"[green]✓[/green] Found [bold]{len(covariate_values)}[/bold] "
        f"unique values: {covariate_values}"
    )

    # ------------------------------------------------------------------
    # 4. Determine parallelism settings
    # ------------------------------------------------------------------
    gpu_ids = _detect_gpu_ids()
    n_gpus = len(gpu_ids)
    console.print(
        f"[dim]GPUs detected:[/dim] [cyan]{n_gpus}[/cyan]"
        + (f" (IDs: {', '.join(gpu_ids)})" if gpu_ids else "")
    )

    # n_jobs: user override > data config > number of GPUs > 1
    n_jobs_raw = data_overrides.get("n_jobs", data_cfg.get("n_jobs"))
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

    tmp_names = _generate_tmp_yamls(
        data_cfg, split_by, covariate_values, assigned_gpu_ids
    )
    for name in tmp_names:
        console.print(f"  [dim]Created:[/dim] conf/data/_tmp_split/{name}.yaml")

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

    data_list = ",".join(f"_tmp_split/{n}" for n in tmp_names)

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
        )
    else:
        cmd = _build_joblib_multirun_command(
            data_list=data_list,
            n_jobs=n_jobs,
            forwarded_args=forwarded_args,
        )

    console.print(f"[dim]Command:[/dim]")
    console.print(f"  [cyan]{' '.join(cmd)}[/cyan]")
    console.print()

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            console.print(
                f"[red]Hydra multirun exited with code "
                f"{result.returncode}[/red]"
            )
            sys.exit(result.returncode)
    finally:
        # ------------------------------------------------------------------
        # 7. Cleanup
        # ------------------------------------------------------------------
        console.print()
        console.print("[dim]Cleaning up temporary configs...[/dim]")
        _cleanup_tmp_dir()
        console.print(
            "[green]✓[/green] Temporary configs removed."
        )

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
        f"[dim]Fitted {len(covariate_values)} models for "
        f"{split_by} values: {covariate_values}[/dim]"
    )
    console.print()


if __name__ == "__main__":
    main()
