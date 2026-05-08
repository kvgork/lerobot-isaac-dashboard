"""Tests for loaders/synthetic.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lerobot_isaac_dashboard.loaders.synthetic import SYNTHETIC_SCHEMA, load_synthetic

EXPECTED_COLS = list(SYNTHETIC_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Empty-state
# ---------------------------------------------------------------------------

def test_synthetic_empty(workspace_root):
    result = load_synthetic(workspace_root)
    assert result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS


def test_synthetic_empty_no_exception(tmp_path):
    result = load_synthetic(tmp_path / "nonexistent")
    assert result.is_empty


# ---------------------------------------------------------------------------
# Happy path fixture helpers
# ---------------------------------------------------------------------------

def _write_meta(workspace_root: Path, repo_id: str, episodes: list[dict], tasks: list[dict]) -> Path:
    meta_dir = workspace_root / "datasets" / repo_id / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episodes).to_parquet(meta_dir / "episodes.parquet", index=False)
    pd.DataFrame(tasks).to_parquet(meta_dir / "tasks.parquet", index=False)
    return meta_dir


def test_synthetic_happy(workspace_root):
    _write_meta(
        workspace_root,
        "merged_v1",
        episodes=[
            {"episode_index": 0, "length": 45, "source": "real", "tasks_index": 0},
            {"episode_index": 1, "length": 52, "source": "sim_dr", "tasks_index": 0},
            {"episode_index": 2, "length": 48, "source": "mimicgen", "tasks_index": 0},
        ],
        tasks=[
            {"tasks_index": 0, "task": "pick_and_place"},
        ],
    )
    result = load_synthetic(workspace_root)
    assert not result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
    assert len(result.df) == 3
    sources = set(result.df["source"].tolist())
    assert "real" in sources
    assert "sim_dr" in sources
    assert "mimicgen" in sources


def test_synthetic_task_join(workspace_root):
    _write_meta(
        workspace_root,
        "merged_v2",
        episodes=[{"episode_index": 0, "length": 30, "source": "real", "tasks_index": 0}],
        tasks=[{"tasks_index": 0, "task": "pick_and_place"}],
    )
    result = load_synthetic(workspace_root)
    assert str(result.df["task"].iloc[0]) == "pick_and_place"


def test_synthetic_no_source_column_skipped(workspace_root):
    """Dataset without 'source' column is skipped (not a merged dataset)."""
    meta_dir = workspace_root / "datasets" / "untagged" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"episode_index": 0, "length": 10, "tasks_index": 0}]
    ).to_parquet(meta_dir / "episodes.parquet", index=False)
    result = load_synthetic(workspace_root)
    assert result.is_empty


# ---------------------------------------------------------------------------
# Malformed
# ---------------------------------------------------------------------------

def test_synthetic_malformed_parquet(workspace_root):
    """Corrupt parquet file — loader skips it, returns empty."""
    meta_dir = workspace_root / "datasets" / "bad_ds" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "episodes.parquet").write_bytes(b"not a parquet file")
    result = load_synthetic(workspace_root)
    assert result.is_empty


def test_synthetic_missing_tasks_index(workspace_root):
    """episodes.parquet lacks tasks_index — task falls back to NA with warning."""
    meta_dir = workspace_root / "datasets" / "notask" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"episode_index": 0, "length": 10, "source": "real"}]
    ).to_parquet(meta_dir / "episodes.parquet", index=False)
    pd.DataFrame(
        [{"tasks_index": 0, "task": "pick_and_place"}]
    ).to_parquet(meta_dir / "tasks.parquet", index=False)
    result = load_synthetic(workspace_root)
    assert not result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
