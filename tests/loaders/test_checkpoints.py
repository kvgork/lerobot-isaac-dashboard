"""Tests for loaders/checkpoints.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot_isaac_dashboard.loaders.checkpoints import CHECKPOINT_SCHEMA, load_checkpoints

EXPECTED_COLS = list(CHECKPOINT_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Empty-state
# ---------------------------------------------------------------------------

def test_checkpoints_empty(workspace_root):
    result = load_checkpoints(workspace_root)
    assert result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
    assert len(result.df) == 0


def test_checkpoints_empty_no_exception(tmp_path):
    result = load_checkpoints(tmp_path / "nonexistent")
    assert result.is_empty


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def _make_checkpoint(workspace_root: Path, arch: str, run_id: str, filename: str) -> Path:
    ckpt_dir = workspace_root / "outputs" / "checkpoints" / arch / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / filename
    ckpt.write_bytes(b"\x00" * 1024)  # 1 KB dummy
    return ckpt


def test_checkpoints_happy(workspace_root):
    _make_checkpoint(workspace_root, "smolvla", "run_001", "step_010000.pt")
    result = load_checkpoints(workspace_root)
    assert not result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
    assert len(result.df) == 1
    row = result.df.iloc[0]
    assert str(row["arch"]) == "smolvla"
    assert str(row["run_id"]) == "run_001"
    assert int(row["step"]) == 10000


def test_checkpoints_step_from_filename_variants(workspace_root):
    """Various step filename patterns are parsed correctly."""
    _make_checkpoint(workspace_root, "act", "run_a", "step_005000.pt")
    _make_checkpoint(workspace_root, "act", "run_a", "checkpoint-020000.safetensors")
    result = load_checkpoints(workspace_root)
    steps = sorted(int(s) for s in result.df["step"].dropna())
    assert 5000 in steps
    assert 20000 in steps


def test_checkpoints_multiple_archs(workspace_root):
    _make_checkpoint(workspace_root, "smolvla", "run_001", "step_001.pt")
    _make_checkpoint(workspace_root, "dreamerv3", "run_002", "step_002.pt")
    result = load_checkpoints(workspace_root)
    archs = set(result.df["arch"].tolist())
    assert "smolvla" in archs
    assert "dreamerv3" in archs


def test_checkpoints_safetensors(workspace_root):
    _make_checkpoint(workspace_root, "act", "run_003", "model_step_050000.safetensors")
    result = load_checkpoints(workspace_root)
    assert not result.is_empty
    assert "safetensors" in str(result.df["path"].iloc[0])


# ---------------------------------------------------------------------------
# Malformed
# ---------------------------------------------------------------------------

def test_checkpoints_no_step_in_name(workspace_root):
    """A checkpoint with no step in name still appears but with step=NA + warning."""
    _make_checkpoint(workspace_root, "act", "run_x", "final_model.pt")
    result = load_checkpoints(workspace_root)
    # step may be extracted from any digit sequence; if no digit the step is NA
    # Either way loader must not crash
    assert list(result.df.columns) == EXPECTED_COLS
