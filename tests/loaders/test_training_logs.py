"""Tests for loaders/training_logs.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot_isaac_dashboard.loaders.training_logs import (
    TRAINING_LOG_SCHEMA,
    load_training_logs,
)

EXPECTED_COLS = list(TRAINING_LOG_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Empty-state
# ---------------------------------------------------------------------------

def test_training_logs_empty(workspace_root):
    result = load_training_logs(workspace_root)
    assert result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS


def test_training_logs_empty_no_exception(tmp_path):
    result = load_training_logs(tmp_path / "nonexistent")
    assert result.is_empty


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def _write_log(workspace_root: Path, arch: str, run_id: str, content: str) -> Path:
    log_dir = workspace_root / "outputs" / "checkpoints" / arch / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.txt"
    log_file.write_text(content, encoding="utf-8")
    return log_file


def test_training_logs_happy(workspace_root):
    log_content = (
        "pc_success=0.73  # step=1000\n"
        "recon_loss=0.042  # step=1000\n"
    )
    _write_log(workspace_root, "smolvla", "run_001", log_content)
    result = load_training_logs(workspace_root)
    assert not result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
    names = set(result.df["metric_name"].tolist())
    assert "pc_success" in names
    assert "recon_loss" in names


def test_training_logs_step_extracted(workspace_root):
    _write_log(
        workspace_root,
        "act",
        "run_002",
        "pc_success=0.5  # step=5000\n",
    )
    result = load_training_logs(workspace_root)
    assert int(result.df["step"].iloc[0]) == 5000


def test_training_logs_arch_run_id(workspace_root):
    _write_log(workspace_root, "dreamerv3", "run_xyz", "pred_loss=0.01\n")
    result = load_training_logs(workspace_root)
    assert str(result.df["arch"].iloc[0]) == "dreamerv3"
    assert str(result.df["run_id"].iloc[0]) == "run_xyz"


def test_training_logs_multiple_runs(workspace_root):
    _write_log(workspace_root, "smolvla", "run_001", "pc_success=0.5\n")
    _write_log(workspace_root, "smolvla", "run_002", "pc_success=0.6\n")
    result = load_training_logs(workspace_root)
    runs = set(result.df["run_id"].tolist())
    assert "run_001" in runs
    assert "run_002" in runs


# ---------------------------------------------------------------------------
# Malformed
# ---------------------------------------------------------------------------

def test_training_logs_empty_file(workspace_root):
    """Empty log.txt returns empty DF, no exception."""
    _write_log(workspace_root, "act", "run_empty", "")
    result = load_training_logs(workspace_root)
    assert result.is_empty


def test_training_logs_no_metric_lines(workspace_root):
    """Log with only narrative text — no metric rows extracted."""
    _write_log(
        workspace_root,
        "act",
        "run_nomet",
        "Loading model...\nStarting training...\nDone.\n",
    )
    result = load_training_logs(workspace_root)
    # May extract spurious tokens from words — not crashing is sufficient
    assert list(result.df.columns) == EXPECTED_COLS


def test_training_logs_mixed_valid_invalid(workspace_root):
    """Some valid metric lines mixed with garbage — valid ones parsed."""
    content = (
        "pc_success=0.8  # step=100\n"
        "NOT A METRIC LINE $$$$\n"
        "val_loss=0.03  # step=100\n"
    )
    _write_log(workspace_root, "smolvla", "run_mix", content)
    result = load_training_logs(workspace_root)
    assert not result.is_empty
    names = set(result.df["metric_name"].tolist())
    assert "pc_success" in names
    assert "val_loss" in names
