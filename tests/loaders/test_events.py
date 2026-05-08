"""Tests for loaders/events.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_isaac_dashboard.loaders.events import EVENTS_SCHEMA, load_events

EXPECTED_COLS = list(EVENTS_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Empty-state
# ---------------------------------------------------------------------------

def test_events_empty(workspace_root):
    result = load_events(workspace_root)
    assert result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS


def test_events_empty_no_exception(tmp_path):
    result = load_events(tmp_path / "nonexistent")
    assert result.is_empty


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def _write_events(workspace_root: Path, session_id: str, records: list[dict]) -> Path:
    sess_dir = workspace_root / ".agent-state" / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    events_file = sess_dir / "events.jsonl"
    with events_file.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return events_file


def test_events_happy(workspace_root):
    _write_events(
        workspace_root,
        "sess_001",
        [
            {"ts": "2026-05-08T10:00:00Z", "phase": "training", "event": "start", "data": "{}"},
            {"ts": "2026-05-08T11:00:00Z", "phase": "training", "event": "complete", "data": "{}"},
        ],
    )
    result = load_events(workspace_root)
    assert not result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
    assert len(result.df) == 2
    assert str(result.df["session_id"].iloc[0]) == "sess_001"


def test_events_multiple_sessions(workspace_root):
    _write_events(workspace_root, "sess_001", [{"ts": "2026-05-08T10:00:00Z", "event": "start"}])
    _write_events(workspace_root, "sess_002", [{"ts": "2026-05-08T10:30:00Z", "event": "complete"}])
    result = load_events(workspace_root)
    sessions = set(result.df["session_id"].tolist())
    assert "sess_001" in sessions
    assert "sess_002" in sessions


def test_events_session_filter(workspace_root):
    _write_events(workspace_root, "sess_001", [{"ts": "2026-05-08T10:00:00Z", "event": "a"}] * 3)
    _write_events(workspace_root, "sess_002", [{"ts": "2026-05-08T11:00:00Z", "event": "b"}] * 7)
    result = load_events(workspace_root, session_id="sess_001")
    assert len(result.df) == 3


# ---------------------------------------------------------------------------
# Malformed
# ---------------------------------------------------------------------------

def test_events_malformed_jsonl(workspace_root):
    """Some bad JSONL lines — valid ones still loaded."""
    sess_dir = workspace_root / ".agent-state" / "sess_bad"
    sess_dir.mkdir(parents=True, exist_ok=True)
    (sess_dir / "events.jsonl").write_text(
        '{"ts": "2026-05-08T10:00:00Z", "event": "ok"}\nBAD\n{"ts": "2026-05-08T11:00:00Z", "event": "ok2"}\n',
        encoding="utf-8",
    )
    result = load_events(workspace_root)
    assert not result.is_empty
    assert len(result.df) == 2


def test_events_empty_file(workspace_root):
    """Empty events.jsonl — returns empty DF with warning."""
    sess_dir = workspace_root / ".agent-state" / "sess_empty"
    sess_dir.mkdir(parents=True, exist_ok=True)
    (sess_dir / "events.jsonl").write_text("", encoding="utf-8")
    result = load_events(workspace_root)
    assert result.is_empty
