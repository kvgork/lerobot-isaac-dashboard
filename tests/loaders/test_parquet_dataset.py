"""Tests for loaders/parquet_dataset.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lerobot_isaac_dashboard.loaders.parquet_dataset import (
    DATASET_SUMMARY_SCHEMA,
    load_parquet_dataset,
)

EXPECTED_COLS = list(DATASET_SUMMARY_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Empty-state
# ---------------------------------------------------------------------------

def test_parquet_dataset_empty(workspace_root):
    result = load_parquet_dataset(workspace_root)
    assert result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
    assert len(result.df) == 0


def test_parquet_dataset_empty_no_exception(tmp_path):
    result = load_parquet_dataset(tmp_path / "nonexistent")
    assert result.is_empty


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def _write_dataset(workspace_root: Path, repo_id: str, n_episodes: int = 5) -> Path:
    meta_dir = workspace_root / "datasets" / repo_id / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    episodes = [
        {
            "episode_index": i,
            "length": 40 + i,
            "tasks_index": 0,
            "source": "real" if i < 3 else "sim_dr",
        }
        for i in range(n_episodes)
    ]
    pd.DataFrame(episodes).to_parquet(meta_dir / "episodes.parquet", index=False)

    info = {"fps": 30, "total_episodes": n_episodes, "total_frames": sum(40 + i for i in range(n_episodes))}
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    return meta_dir.parent


def test_parquet_dataset_happy(workspace_root):
    _write_dataset(workspace_root, "so101_pick_v1", n_episodes=5)
    result = load_parquet_dataset(workspace_root)
    assert not result.is_empty
    assert list(result.df.columns) == EXPECTED_COLS
    assert len(result.df) == 1
    row = result.df.iloc[0]
    assert str(row["repo_id"]) == "so101_pick_v1"
    assert int(row["n_episodes"]) == 5
    assert int(row["fps"]) == 30
    assert int(row["source_real"]) == 3
    assert int(row["source_sim_dr"]) == 2


def test_parquet_dataset_multiple(workspace_root):
    _write_dataset(workspace_root, "ds_a", n_episodes=3)
    _write_dataset(workspace_root, "ds_b", n_episodes=7)
    result = load_parquet_dataset(workspace_root)
    assert len(result.df) == 2
    repo_ids = set(result.df["repo_id"].tolist())
    assert "ds_a" in repo_ids
    assert "ds_b" in repo_ids


# ---------------------------------------------------------------------------
# Malformed
# ---------------------------------------------------------------------------

def test_parquet_dataset_missing_episodes_parquet(workspace_root):
    """meta/ exists but episodes.parquet missing — warning emitted, row still present."""
    meta_dir = workspace_root / "datasets" / "incomplete" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    # Only info.json, no episodes.parquet
    (meta_dir / "info.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
    result = load_parquet_dataset(workspace_root)
    # Should still find the dataset (meta/ dir exists)
    assert list(result.df.columns) == EXPECTED_COLS
    assert any("incomplete" in w for w in result.warnings)


def test_parquet_dataset_no_source_column(workspace_root):
    """episodes.parquet without source column — source counts zero, warning emitted."""
    meta_dir = workspace_root / "datasets" / "no_src" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"episode_index": 0, "length": 10, "tasks_index": 0}]
    ).to_parquet(meta_dir / "episodes.parquet", index=False)
    result = load_parquet_dataset(workspace_root)
    assert not result.is_empty
    row = result.df.iloc[0]
    assert int(row["source_real"]) == 0
    assert int(row["source_sim_dr"]) == 0
    assert any("source" in w for w in result.warnings)
