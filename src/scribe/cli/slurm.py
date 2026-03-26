"""SLURM helpers for the unified ``scribe-infer`` CLI.

This module centralizes all submitit/SLURM-specific parsing, profile loading,
interactive prompting, and command construction so that ``infer.py`` stays
focused on top-level dispatch behavior.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_ARRAY_PARALLELISM = 1
_DEFAULT_CPUS_PER_TASK = 4
_DEFAULT_MEM_GB = 64
_DEFAULT_TIMEOUT = "0-04:00"
_DEFAULT_TIMEOUT_MIN = 240
_DEFAULT_JOB_NAME = "scribe_infer"
_DEFAULT_SUBMITIT_FOLDER = "slurm_logs/submitit/%j"

_CORE_SLURM_KEYS = {
    "partition",
    "account",
    "array_parallelism",
    "cpus_per_task",
    "mem_gb",
    "timeout_min",
}
_OPTIONAL_SLURM_KEYS = {
    "qos",
    "constraint",
    "exclude",
    "nodelist",
    "reservation",
    "gres",
    "mail_user",
    "mail_type",
    "job_name",
    "submitit_folder",
}
_PROFILE_ALLOWED_KEYS = (
    _CORE_SLURM_KEYS | _OPTIONAL_SLURM_KEYS | {"timeout", "launcher_overrides"}
)


@dataclass(frozen=True)
class SlurmPromptConfig:
    """Resolved SLURM settings for ``scribe-infer --slurm``.

    Attributes
    ----------
    partition : str
        Required SLURM partition name. This is intentionally required and has
        no default because partition names are cluster-specific.
    account : str | None
        Optional SLURM account to charge. When ``None``, no account override is
        added to launcher settings.
    array_parallelism : int
        Maximum number of concurrent submitit jobs.
    cpus_per_task : int
        Number of CPU cores assigned to each submitted task.
    mem_gb : int
        Memory in GiB assigned to each submitted task.
    timeout_min : int
        Time limit per submitted task in minutes.
    qos : str | None
        Optional SLURM QoS.
    constraint : str | None
        Optional node feature constraint.
    exclude : str | None
        Optional excluded-node list expression.
    nodelist : str | None
        Optional explicit node list expression.
    reservation : str | None
        Optional SLURM reservation name.
    gres : str | None
        Optional generic resources request string.
    mail_user : str | None
        Optional notification email address.
    mail_type : str | None
        Optional comma-separated mail event types.
    job_name : str | None
        Optional submitit job name override.
    submitit_folder : str | None
        Optional submitit log folder override.
    launcher_overrides : dict[str, str] | None
        Additional raw ``hydra.launcher.<key>=<value>`` entries for
        site-specific extensions.
    """

    partition: str
    account: str | None
    array_parallelism: int
    cpus_per_task: int
    mem_gb: int
    timeout_min: int
    qos: str | None = None
    constraint: str | None = None
    exclude: str | None = None
    nodelist: str | None = None
    reservation: str | None = None
    gres: str | None = None
    mail_user: str | None = None
    mail_type: str | None = None
    job_name: str | None = None
    submitit_folder: str | None = None
    launcher_overrides: dict[str, str] | None = None


def _slurm_time_to_minutes(time_limit: str) -> int:
    """Convert a SLURM time string into integer minutes.

    Parameters
    ----------
    time_limit : str
        Time limit in ``D-HH:MM`` or ``HH:MM`` format.

    Returns
    -------
    int
        Total minutes represented by ``time_limit``.

    Raises
    ------
    ValueError
        Raised when ``time_limit`` is empty or does not match a supported
        SLURM format.
    """
    raw = time_limit.strip()
    if not raw:
        raise ValueError("Empty SLURM time string.")

    if "-" in raw:
        day_str, hhmm = raw.split("-", 1)
        days = int(day_str)
    else:
        days = 0
        hhmm = raw

    hhmm_parts = hhmm.split(":")
    if len(hhmm_parts) < 2:
        raise ValueError("Expected D-HH:MM or HH:MM.")

    hours = int(hhmm_parts[0])
    minutes = int(hhmm_parts[1])
    return days * 24 * 60 + hours * 60 + minutes


def _prompt_nonempty(prompt: str) -> str:
    """Prompt repeatedly until a non-empty string is entered."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Value required. Please enter a non-empty value.")


def _prompt_positive_int(prompt: str, default: int) -> int:
    """Prompt for a positive integer with default fallback."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed > 0:
            return parsed
        print("Value must be a positive integer.")


def _prompt_timeout_minutes(default: str = "0-04:00") -> int:
    """Prompt for a SLURM time limit and normalize to integer minutes."""
    while True:
        raw = input(f"Time limit (D-HH:MM or HH:MM) [{default}]: ").strip()
        text = raw or default
        try:
            return _slurm_time_to_minutes(text)
        except ValueError:
            print("Invalid format. Expected D-HH:MM or HH:MM.")


def _coerce_slurm_scalar(key: str, value: Any) -> str | int | None:
    """Coerce one SLURM scalar value into normalized runtime form."""
    if value is None:
        return None

    if key in {"array_parallelism", "cpus_per_task", "mem_gb", "timeout_min"}:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{key} must be a positive integer.")
        return parsed

    text = str(value).strip()
    if text.lower() in {"", "none", "null", "~"}:
        return None
    return text


def _normalize_profile_values(raw_profile: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate SLURM profile values loaded from YAML."""
    unknown_keys = set(raw_profile) - _PROFILE_ALLOWED_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise SystemExit(
            "Unsupported key(s) in SLURM profile: "
            f"{unknown}. Allowed keys: {', '.join(sorted(_PROFILE_ALLOWED_KEYS))}"
        )

    normalized: dict[str, Any] = {}
    for key in _CORE_SLURM_KEYS | _OPTIONAL_SLURM_KEYS:
        if key not in raw_profile:
            continue
        normalized[key] = _coerce_slurm_scalar(key, raw_profile[key])

    if "timeout" in raw_profile and "timeout_min" not in normalized:
        normalized["timeout_min"] = _slurm_time_to_minutes(
            str(raw_profile["timeout"])
        )

    launcher_block = raw_profile.get("launcher_overrides")
    if launcher_block is not None:
        if not isinstance(launcher_block, dict):
            raise SystemExit(
                "Profile field 'launcher_overrides' must be a mapping."
            )
        normalized["launcher_overrides"] = {
            str(key): str(value) for key, value in launcher_block.items()
        }
    return normalized


def load_slurm_profile(
    profile_arg: str | None, *, config_root: Path
) -> tuple[dict[str, Any], Path | None]:
    """Load and normalize a SLURM profile file when requested.

    Parameters
    ----------
    profile_arg : str or None
        Profile name or explicit profile YAML path from CLI.
    config_root : Path
        Resolved Hydra config root (used for named profile lookup).

    Returns
    -------
    tuple[dict[str, Any], Path | None]
        Normalized profile values and loaded profile path (if any).
    """
    if profile_arg is None:
        return {}, None

    requested = profile_arg.strip()
    if requested == "":
        raise SystemExit(
            "`--slurm-profile` requires a non-empty profile name/path."
        )

    explicit_path = Path(requested).expanduser()
    if explicit_path.suffix in {".yaml", ".yml"} or explicit_path.exists():
        profile_path = explicit_path
    else:
        profile_path = config_root / "slurm" / f"{requested}.yaml"

    if not profile_path.exists():
        raise SystemExit(
            "SLURM profile not found: "
            f"{profile_path}. Use a valid path or create {config_root / 'slurm'}."
        )

    try:
        from omegaconf import OmegaConf

        loaded_profile = OmegaConf.to_container(
            OmegaConf.load(profile_path), resolve=True
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to load SLURM profile {profile_path}: {exc}"
        ) from exc

    if not isinstance(loaded_profile, dict):
        raise SystemExit(
            "SLURM profile must be a YAML mapping at the top level."
        )
    return _normalize_profile_values(loaded_profile), profile_path


def parse_slurm_set_entries(
    entries: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse and normalize repeated ``--slurm-set key=value`` entries."""
    values: dict[str, Any] = {}
    launcher_values: dict[str, str] = {}

    for entry in entries:
        if "=" not in entry:
            raise SystemExit(
                f"Invalid --slurm-set '{entry}'. Expected KEY=VALUE syntax."
            )
        raw_key, raw_val = entry.split("=", 1)
        key = raw_key.strip()
        val = raw_val.strip()
        if key == "":
            raise SystemExit(f"Invalid --slurm-set '{entry}'. Empty key.")

        if key == "timeout":
            values["timeout_min"] = _slurm_time_to_minutes(val)
            continue

        if key.startswith("launcher."):
            launcher_key = key[len("launcher.") :]
            if launcher_key == "":
                raise SystemExit(
                    f"Invalid --slurm-set '{entry}'. launcher.<key> is required."
                )
            launcher_values[launcher_key] = val
            continue

        allowed = _CORE_SLURM_KEYS | _OPTIONAL_SLURM_KEYS
        if key not in allowed:
            raise SystemExit(
                "Unsupported --slurm-set key "
                f"'{key}'. Supported keys: {', '.join(sorted(allowed))}, "
                "plus timeout and launcher.<key>."
            )
        values[key] = _coerce_slurm_scalar(key, val)
    return values, launcher_values


def prompt_slurm_config(
    seed_values: dict[str, Any] | None = None,
    *,
    allow_interactive: bool = True,
) -> SlurmPromptConfig:
    """Collect/resolve SLURM settings for submitit launch mode.

    Parameters
    ----------
    seed_values : dict[str, Any] or None, optional
        Pre-resolved values from profile and/or ``--slurm-set``.
    allow_interactive : bool, optional
        Whether this call may prompt for missing values.

    Returns
    -------
    SlurmPromptConfig
        Validated settings required to build submitit overrides.

    Raises
    ------
    SystemExit
        Raised when required values are missing in non-interactive mode.
    """
    merged = dict(seed_values or {})
    launcher_overrides = {
        str(key): str(value)
        for key, value in merged.pop("launcher_overrides", {}).items()
    }

    if allow_interactive and sys.stdin.isatty():
        print("\nSCRIBE SLURM mode")
        print("Provide cluster-specific resources. Partition has no default.\n")

        if merged.get("partition") is None:
            merged["partition"] = _prompt_nonempty("Partition: ")
        if "account" not in merged:
            account_raw = input("Account (optional): ").strip()
            merged["account"] = account_raw or None
        if "array_parallelism" not in merged:
            merged["array_parallelism"] = _prompt_positive_int(
                "Max parallel jobs", default=_DEFAULT_ARRAY_PARALLELISM
            )
        if "cpus_per_task" not in merged:
            merged["cpus_per_task"] = _prompt_positive_int(
                "CPUs per task", default=_DEFAULT_CPUS_PER_TASK
            )
        if "mem_gb" not in merged:
            merged["mem_gb"] = _prompt_positive_int(
                "Memory (GB)", default=_DEFAULT_MEM_GB
            )
        if "timeout_min" not in merged:
            merged["timeout_min"] = _prompt_timeout_minutes(
                default=_DEFAULT_TIMEOUT
            )
    else:
        merged.setdefault("account", None)
        merged.setdefault("array_parallelism", _DEFAULT_ARRAY_PARALLELISM)
        merged.setdefault("cpus_per_task", _DEFAULT_CPUS_PER_TASK)
        merged.setdefault("mem_gb", _DEFAULT_MEM_GB)
        merged.setdefault("timeout_min", _DEFAULT_TIMEOUT_MIN)

    if merged.get("partition") is None:
        if not allow_interactive or not sys.stdin.isatty():
            raise SystemExit(
                "Missing SLURM partition. Provide it via profile, "
                "`--slurm-set partition=<name>`, or interactive terminal."
            )
        merged["partition"] = _prompt_nonempty("Partition: ")

    # Keep stable defaults for optional-but-common submitit metadata.
    merged.setdefault("job_name", _DEFAULT_JOB_NAME)
    merged.setdefault("submitit_folder", _DEFAULT_SUBMITIT_FOLDER)

    return SlurmPromptConfig(
        partition=str(merged["partition"]),
        account=merged.get("account"),
        array_parallelism=int(merged["array_parallelism"]),
        cpus_per_task=int(merged["cpus_per_task"]),
        mem_gb=int(merged["mem_gb"]),
        timeout_min=int(merged["timeout_min"]),
        qos=merged.get("qos"),
        constraint=merged.get("constraint"),
        exclude=merged.get("exclude"),
        nodelist=merged.get("nodelist"),
        reservation=merged.get("reservation"),
        gres=merged.get("gres"),
        mail_user=merged.get("mail_user"),
        mail_type=merged.get("mail_type"),
        job_name=merged.get("job_name"),
        submitit_folder=merged.get("submitit_folder"),
        launcher_overrides=launcher_overrides,
    )


def build_slurm_command(
    *,
    split_mode: bool,
    config_path: str,
    config_name: str,
    forwarded_overrides: list[str],
    slurm_cfg: SlurmPromptConfig,
) -> list[str]:
    """Build submitit-backed command for ``scribe-infer --slurm``.

    Parameters
    ----------
    split_mode : bool
        Whether split orchestration should be used.
    config_path : str
        Hydra config root path.
    config_name : str
        Top-level Hydra config name.
    forwarded_overrides : list[str]
        User-provided Hydra overrides.
    slurm_cfg : SlurmPromptConfig
        Resolved SLURM settings.

    Returns
    -------
    list[str]
        Command vector ready for ``subprocess.call``.
    """
    # Shared optional fields mapped to submitit launcher-style keys.
    shared_optional = {
        "qos": slurm_cfg.qos,
        "constraint": slurm_cfg.constraint,
        "exclude": slurm_cfg.exclude,
        "nodelist": slurm_cfg.nodelist,
        "reservation": slurm_cfg.reservation,
        "gres": slurm_cfg.gres,
        "mail_user": slurm_cfg.mail_user,
        "mail_type": slurm_cfg.mail_type,
    }

    if split_mode:
        split_overrides = [
            "split.launcher=submitit_slurm",
            f"split.partition={slurm_cfg.partition}",
            f"split.array_parallelism={slurm_cfg.array_parallelism}",
            f"split.cpus_per_task={slurm_cfg.cpus_per_task}",
            f"split.mem_gb={slurm_cfg.mem_gb}",
            f"split.timeout_min={slurm_cfg.timeout_min}",
        ]
        if slurm_cfg.account is not None:
            split_overrides.append(f"split.account={slurm_cfg.account}")
        if slurm_cfg.job_name is not None:
            split_overrides.append(f"split.job_name={slurm_cfg.job_name}")
        if slurm_cfg.submitit_folder is not None:
            split_overrides.append(
                f"split.submitit_folder={slurm_cfg.submitit_folder}"
            )
        for key, value in shared_optional.items():
            if value is not None:
                split_overrides.append(f"split.{key}={value}")
        for key, value in (slurm_cfg.launcher_overrides or {}).items():
            split_overrides.append(f"split.{key}={value}")
        return [
            sys.executable,
            "-m",
            "scribe.cli.split_orchestrator",
            "--config-path",
            config_path,
            "--config-name",
            config_name,
            *forwarded_overrides,
            *split_overrides,
        ]

    direct_overrides = [
        "hydra/launcher=submitit_slurm",
        "hydra.launcher.nodes=1",
        "hydra.launcher.tasks_per_node=1",
        "hydra.launcher.gpus_per_node=1",
        f"hydra.launcher.partition={slurm_cfg.partition}",
        f"hydra.launcher.cpus_per_task={slurm_cfg.cpus_per_task}",
        f"hydra.launcher.mem_gb={slurm_cfg.mem_gb}",
        f"hydra.launcher.timeout_min={slurm_cfg.timeout_min}",
        f"hydra.launcher.array_parallelism={slurm_cfg.array_parallelism}",
    ]
    if slurm_cfg.account is not None:
        direct_overrides.append(f"hydra.launcher.account={slurm_cfg.account}")
    if slurm_cfg.job_name is not None:
        direct_overrides.append(f"hydra.launcher.name={slurm_cfg.job_name}")
    if slurm_cfg.submitit_folder is not None:
        direct_overrides.append(
            f"hydra.launcher.submitit_folder={slurm_cfg.submitit_folder}"
        )
    for key, value in shared_optional.items():
        if value is not None:
            direct_overrides.append(f"hydra.launcher.{key}={value}")
    for key, value in (slurm_cfg.launcher_overrides or {}).items():
        direct_overrides.append(f"hydra.launcher.{key}={value}")
    return [
        sys.executable,
        "-m",
        "scribe.cli.infer_runner",
        "--config-path",
        config_path,
        "--config-name",
        config_name,
        "--multirun",
        *forwarded_overrides,
        *direct_overrides,
    ]
