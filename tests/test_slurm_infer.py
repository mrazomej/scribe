"""Tests for SLURM inference launcher helpers."""

from slurm_infer import apply_slurm_default_overrides


def test_apply_slurm_default_overrides_injects_progress_logging_default():
    """Inject default logging override when user does not set the key."""
    overrides = [
        "data=bleo_splits/bleo_study03_bleomycin",
        "model=zinbvcp",
    ]
    effective = apply_slurm_default_overrides(overrides)
    assert "inference.log_progress_lines=true" in effective
    assert effective[-2:] == overrides


def test_apply_slurm_default_overrides_respects_explicit_user_value():
    """Do not inject the default when user sets the key explicitly."""
    overrides = [
        "data=bleo_splits/bleo_study03_bleomycin",
        "inference.log_progress_lines=false",
    ]
    effective = apply_slurm_default_overrides(overrides)
    assert effective == overrides


def test_apply_slurm_default_overrides_normalizes_hydra_prefixes():
    """Treat +/~ Hydra key prefixes as explicit user overrides."""
    overrides = [
        "+inference.log_progress_lines=false",
        "data=bleo_splits/bleo_study06_control",
    ]
    effective = apply_slurm_default_overrides(overrides)
    assert effective == overrides
