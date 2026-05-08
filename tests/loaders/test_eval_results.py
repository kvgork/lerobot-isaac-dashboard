"""Tests for loaders/eval_results.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_isaac_dashboard.loaders.eval_results import EVAL_SCHEMA, load_eval_results

EXPECTED_COLS = list(EVAL_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Empty-state
# ---------------------------------------------------------------------------

def test_eval_results_empty(workspace_root):
    """Point at non-existent outputs/eval/ — returns empty canonical DF."""
    result = load_eval_results(workspace_root)
    assert result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
    assert len(result.df) == 0


def test_eval_results_empty_no_exception(tmp_path):
    """Completely missing workspace still returns empty result."""
    result = load_eval_results(tmp_path / "does_not_exist")
    assert result.is_empty


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def _write_eval_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_eval_results_happy(workspace_root):
    eval_dir = workspace_root / "outputs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_eval_json(
        eval_dir / "run_001.json",
        {
            "run_id": "run_001",
            "task": "pick_and_place",
            "ts": "2026-05-08T10:00:00Z",
            "pc_success": 0.73,
            "n_episodes": 100,
            "intervention_rate": 0.05,
            "mean_ep_len": 45.2,
        },
    )
    result = load_eval_results(workspace_root)
    assert not result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
    assert len(result.df) == 1
    assert str(result.df["run_id"].iloc[0]) == "run_001"
    assert float(result.df["pc_success"].iloc[0]) == pytest.approx(0.73)


def test_eval_results_multiple_files(workspace_root):
    eval_dir = workspace_root / "outputs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        _write_eval_json(
            eval_dir / f"run_{i:03d}.json",
            {"run_id": f"run_{i:03d}", "pc_success": 0.5 + i * 0.1},
        )
    result = load_eval_results(workspace_root)
    assert not result.is_empty
    assert len(result.df) == 3


# ---------------------------------------------------------------------------
# Malformed
# ---------------------------------------------------------------------------

def test_eval_results_malformed_json(workspace_root):
    """Truncated JSON — loader returns partial results + warning."""
    eval_dir = workspace_root / "outputs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "bad.json").write_text("{broken json", encoding="utf-8")
    result = load_eval_results(workspace_root)
    assert result.is_empty  # no valid files
    assert any("bad.json" in w for w in result.warnings)


def test_eval_results_missing_keys_emit_warnings(workspace_root):
    """JSON with only some keys — missing ones filled with NA + warning."""
    eval_dir = workspace_root / "outputs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_eval_json(eval_dir / "partial.json", {"pc_success": 0.5})
    result = load_eval_results(workspace_root)
    assert not result.is_empty
    # Warnings about missing keys should be present
    assert len(result.warnings) > 0
    # run_id should be derived from filename
    assert "partial" in str(result.df["run_id"].iloc[0])


def test_eval_results_wrong_type(workspace_root):
    """JSON array instead of object — loader skips it with a warning."""
    eval_dir = workspace_root / "outputs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "array.json").write_text("[1, 2, 3]", encoding="utf-8")
    result = load_eval_results(workspace_root)
    assert result.is_empty
    assert any("array.json" in w for w in result.warnings)
