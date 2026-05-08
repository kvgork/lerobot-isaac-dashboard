"""Tests for loaders/autoresearch.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_isaac_dashboard.loaders.autoresearch import (
    HISTORY_SCHEMA,
    load_autoresearch,
)

EXPECTED_HISTORY_COLS = list(HISTORY_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Empty-state
# ---------------------------------------------------------------------------

def test_autoresearch_empty(workspace_root):
    result = load_autoresearch(workspace_root)
    assert result.is_empty
    assert isinstance(result.df, dict)
    assert "history" in result.df
    assert list(result.df["history"].columns) == EXPECTED_HISTORY_COLS
    assert result.df["program"] == {}
    assert result.df["best"] == {}
    assert result.df["plateau"] == {}


def test_autoresearch_empty_no_exception(tmp_path):
    result = load_autoresearch(tmp_path / "nonexistent")
    assert result.is_empty


# ---------------------------------------------------------------------------
# Happy path fixture
# ---------------------------------------------------------------------------

def _make_autoresearch_slug(
    workspace_root: Path,
    session_id: str,
    slug: str,
    n_trials: int = 3,
) -> Path:
    slug_dir = workspace_root / ".agent-state" / session_id / "autoresearch" / slug
    slug_dir.mkdir(parents=True, exist_ok=True)

    # history.jsonl
    history_path = slug_dir / "history.jsonl"
    with history_path.open("w") as fh:
        for i in range(n_trials):
            fh.write(
                json.dumps(
                    {
                        "trial": i,
                        "metric_name": "pc_success",
                        "metric_value": 0.5 + i * 0.1,
                        "config": {"lr": 1e-4, "batch_size": 8},
                        "ts": "2026-05-08T10:00:00Z",
                        "status": "complete",
                    }
                )
                + "\n"
            )

    # program.md
    (slug_dir / "program.md").write_text(
        "## Training Script\npath: train.py\n## Metric\nname: pc_success\nregex: pc_success=([0-9.]+)\n",
        encoding="utf-8",
    )

    # best_config.yaml
    (slug_dir / "best_config.yaml").write_text("lr: 0.0001\nbatch_size: 16\n", encoding="utf-8")

    # plateau_tracker.json
    (slug_dir / "plateau_tracker.json").write_text(
        json.dumps({"plateau": False, "best_value": 0.7, "patience_left": 3}),
        encoding="utf-8",
    )

    return slug_dir


def test_autoresearch_happy(workspace_root):
    _make_autoresearch_slug(workspace_root, "sess_001", "smolvla_hp_search", n_trials=3)
    result = load_autoresearch(workspace_root)
    assert not result.is_empty
    hist = result.df["history"]
    assert list(hist.columns) == EXPECTED_HISTORY_COLS
    assert len(hist) == 3
    assert result.df["program"] != {}
    assert result.df["best"] != {}
    assert result.df["plateau"] != {}


def test_autoresearch_session_filter(workspace_root):
    _make_autoresearch_slug(workspace_root, "sess_001", "slug_a", n_trials=2)
    _make_autoresearch_slug(workspace_root, "sess_002", "slug_b", n_trials=5)

    # Only load sess_001
    result = load_autoresearch(workspace_root, session_id="sess_001")
    assert len(result.df["history"]) == 2

    # Only load sess_002
    result2 = load_autoresearch(workspace_root, session_id="sess_002")
    assert len(result2.df["history"]) == 5


def test_autoresearch_all_sessions(workspace_root):
    _make_autoresearch_slug(workspace_root, "sess_001", "slug_a", n_trials=2)
    _make_autoresearch_slug(workspace_root, "sess_002", "slug_b", n_trials=4)
    result = load_autoresearch(workspace_root)
    assert len(result.df["history"]) == 6


# ---------------------------------------------------------------------------
# Malformed
# ---------------------------------------------------------------------------

def test_autoresearch_malformed_history(workspace_root):
    """Broken JSONL lines are skipped, not empty ones are loaded."""
    slug_dir = workspace_root / ".agent-state" / "sess_bad" / "autoresearch" / "slug_x"
    slug_dir.mkdir(parents=True, exist_ok=True)
    (slug_dir / "history.jsonl").write_text(
        '{"trial": 0, "metric_value": 0.5}\nNOT JSON\n{"trial": 2, "metric_value": 0.7}\n',
        encoding="utf-8",
    )
    result = load_autoresearch(workspace_root)
    # 2 valid lines parsed
    assert len(result.df["history"]) == 2


def test_autoresearch_empty_slug_dir(workspace_root):
    """Empty slug dir (no files) — handled gracefully."""
    slug_dir = workspace_root / ".agent-state" / "sess_empty" / "autoresearch" / "empty_slug"
    slug_dir.mkdir(parents=True, exist_ok=True)
    result = load_autoresearch(workspace_root)
    assert result.is_empty
