"""Tests for loaders/curriculum.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_isaac_dashboard.loaders.curriculum import (
    CURRICULUM_HISTORY_SCHEMA,
    load_curriculum,
)

EXPECTED_HISTORY_COLS = list(CURRICULUM_HISTORY_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Empty-state
# ---------------------------------------------------------------------------

def test_curriculum_empty(workspace_root):
    result = load_curriculum(workspace_root)
    assert result.is_empty
    assert isinstance(result.df, dict)
    assert result.df["current"] == {}
    assert list(result.df["history"].columns) == EXPECTED_HISTORY_COLS


def test_curriculum_empty_no_exception(tmp_path):
    result = load_curriculum(tmp_path / "nonexistent")
    assert result.is_empty


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def _write_curriculum_files(workspace_root: Path, n_history: int = 3) -> None:
    outputs_dir = workspace_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # curriculum_stage.json
    (outputs_dir / "curriculum_stage.json").write_text(
        json.dumps(
            {
                "stage": "stage_2",
                "task_config": {"task": "pick_and_place", "difficulty": 2},
                "advancement_reason": "success_rate >= 0.8",
                "ts": "2026-05-08T10:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    # curriculum_history.jsonl
    hist_path = outputs_dir / "curriculum_history.jsonl"
    with hist_path.open("w") as fh:
        for i in range(n_history):
            fh.write(
                json.dumps(
                    {
                        "ts": f"2026-05-0{i+1}T10:00:00Z",
                        "stage": f"stage_{i}",
                        "advancement_reason": f"reason_{i}",
                        "task_config_diff": {"difficulty": i},
                    }
                )
                + "\n"
            )


def test_curriculum_happy(workspace_root):
    _write_curriculum_files(workspace_root, n_history=3)
    result = load_curriculum(workspace_root)
    assert not result.is_empty
    assert result.df["current"]["stage"] == "stage_2"
    hist = result.df["history"]
    assert list(hist.columns) == EXPECTED_HISTORY_COLS
    assert len(hist) == 3


def test_curriculum_stage_only(workspace_root):
    """Only curriculum_stage.json present (no history) — not empty."""
    outputs_dir = workspace_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "curriculum_stage.json").write_text(
        json.dumps({"stage": "stage_1", "ts": "2026-05-08T09:00:00Z"}),
        encoding="utf-8",
    )
    result = load_curriculum(workspace_root)
    assert not result.is_empty
    assert result.df["current"]["stage"] == "stage_1"


# ---------------------------------------------------------------------------
# Malformed
# ---------------------------------------------------------------------------

def test_curriculum_malformed_stage_json(workspace_root):
    """Broken curriculum_stage.json — returns empty current dict + warning."""
    outputs_dir = workspace_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "curriculum_stage.json").write_text("{broken", encoding="utf-8")
    result = load_curriculum(workspace_root)
    assert result.df["current"] == {}


def test_curriculum_malformed_history(workspace_root):
    """Broken history JSONL lines — valid ones still loaded."""
    outputs_dir = workspace_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "curriculum_history.jsonl").write_text(
        '{"ts": "2026-05-08T10:00:00Z", "stage": "s1"}\nBAD JSON\n{"ts": "2026-05-08T11:00:00Z", "stage": "s2"}\n',
        encoding="utf-8",
    )
    result = load_curriculum(workspace_root)
    # 2 valid records
    assert len(result.df["history"]) == 2
