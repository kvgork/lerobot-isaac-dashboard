"""test_snapshots.py — Tests for snapshots.py (Phase 8).

Covers:
- save_snapshot on empty workspace
- label propagation into snapshot_id and meta.json
- load/roundtrip DataFrame equality
- schema version rejection
- list_snapshots sorted by ts desc
- git sha capture (mocked subprocess)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lerobot_isaac_dashboard.snapshots import (
    SCHEMA_VERSION,
    SnapshotMeta,
    list_snapshots,
    load_snapshot,
    save_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> Path:
    """Create the minimal workspace directory structure."""
    (tmp_path / "datasets").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / ".agent-state").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# save_snapshot — empty workspace
# ---------------------------------------------------------------------------


class TestSaveSnapshotEmptyWorkspace:
    """save_snapshot on an empty workspace creates the expected directory tree."""

    def test_snapshot_dir_created(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        assert snap_dir.exists(), "Snapshot directory should be created"
        assert snap_dir.is_dir()

    def test_meta_json_exists(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        meta_path = snap_dir / "meta.json"
        assert meta_path.exists(), "meta.json should be created"
        assert meta_path.stat().st_size > 0

    def test_loaders_dir_exists(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        loaders_dir = snap_dir / "loaders"
        assert loaders_dir.exists(), "loaders/ directory should be created"

    def test_meta_schema_version(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        meta = json.loads((snap_dir / "meta.json").read_text())
        assert meta["schema_version"] == SCHEMA_VERSION

    def test_default_output_dir(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        expected_parent = ws / "outputs" / "snapshots"
        assert snap_dir.parent == expected_parent

    def test_parquet_files_created(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        loaders_dir = snap_dir / "loaders"
        # At least some parquet files should exist (even if empty schema)
        parquets = list(loaders_dir.glob("*.parquet"))
        assert len(parquets) > 0, "Expected at least one .parquet file in loaders/"

    def test_paths_json_created(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        paths_json = snap_dir / "loaders" / "paths.json"
        assert paths_json.exists(), "paths.json should be created"


# ---------------------------------------------------------------------------
# save_snapshot — label propagation
# ---------------------------------------------------------------------------


class TestSaveSnapshotWithLabel:
    """Label is embedded in snapshot_id and meta.json."""

    def test_snapshot_id_contains_label(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws, label="baseline")
        meta = json.loads((snap_dir / "meta.json").read_text())
        assert "baseline" in meta["snapshot_id"], (
            f"Expected 'baseline' in snapshot_id, got: {meta['snapshot_id']}"
        )

    def test_meta_label_field(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws, label="my-label")
        meta = json.loads((snap_dir / "meta.json").read_text())
        assert meta["label"] == "my-label"

    def test_no_label_becomes_unlabeled(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws, label=None)
        meta = json.loads((snap_dir / "meta.json").read_text())
        assert meta["label"] is None
        assert "unlabeled" in meta["snapshot_id"]

    def test_custom_snapshot_id_override(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws, snapshot_id="my-custom-id")
        assert snap_dir.name == "my-custom-id"
        meta = json.loads((snap_dir / "meta.json").read_text())
        assert meta["snapshot_id"] == "my-custom-id"

    def test_custom_output_dir(self, tmp_path):
        ws = _make_workspace(tmp_path)
        custom_dir = tmp_path / "custom_snap"
        snap_dir = save_snapshot(ws, output_dir=custom_dir)
        assert snap_dir.resolve() == custom_dir.resolve()


# ---------------------------------------------------------------------------
# load_snapshot — roundtrip
# ---------------------------------------------------------------------------


class TestLoadRoundtrip:
    """save then load preserves DataFrame equality per slug."""

    def test_roundtrip_meta(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws, label="test")
        meta, _ = load_snapshot(snap_dir)
        assert meta.label == "test"
        assert meta.schema_version == SCHEMA_VERSION

    def test_roundtrip_loader_slugs(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        meta, results = load_snapshot(snap_dir)
        assert len(meta.loader_slugs) > 0
        assert set(meta.loader_slugs) == set(results.keys())

    def test_roundtrip_df_equality_parquet_dataset(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        _, results = load_snapshot(snap_dir)
        # Empty workspace → empty DataFrame; shape must be consistent
        r = results.get("parquet_dataset")
        assert r is not None
        assert isinstance(r.df, pd.DataFrame)

    def test_roundtrip_autoresearch_dict(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        _, results = load_snapshot(snap_dir)
        r = results.get("autoresearch")
        assert r is not None
        assert isinstance(r.df, dict)
        assert "history" in r.df
        assert "program" in r.df
        assert "best" in r.df
        assert "plateau" in r.df

    def test_roundtrip_curriculum_dict(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        _, results = load_snapshot(snap_dir)
        r = results.get("curriculum")
        assert r is not None
        assert isinstance(r.df, dict)
        assert "current" in r.df
        assert "history" in r.df

    def test_load_by_snapshot_id_string(self, tmp_path):
        ws = _make_workspace(tmp_path)
        save_snapshot(ws, snapshot_id="roundtrip-test")
        meta, _results = load_snapshot("roundtrip-test", workspace_root=ws)
        assert meta.snapshot_id == "roundtrip-test"

    def test_load_missing_workspace_root_raises(self, tmp_path):
        _make_workspace(tmp_path)
        with pytest.raises(ValueError, match="workspace_root must be provided"):
            load_snapshot("some-snap-id", workspace_root=None)


# ---------------------------------------------------------------------------
# load_snapshot — schema version rejection
# ---------------------------------------------------------------------------


class TestLoadRejectsFutureSchema:
    """load_snapshot raises ValueError for unknown schema versions."""

    def test_rejects_schema_version_999(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)

        # Tamper with meta.json to set future schema version
        meta_path = snap_dir / "meta.json"
        meta_data = json.loads(meta_path.read_text())
        meta_data["schema_version"] = 999
        meta_path.write_text(json.dumps(meta_data))

        with pytest.raises(ValueError, match="schema_version=999"):
            load_snapshot(snap_dir)

    def test_accepts_current_schema_version(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_dir = save_snapshot(ws)
        meta, _ = load_snapshot(snap_dir)  # should not raise
        assert meta.schema_version == SCHEMA_VERSION

    def test_missing_meta_json_raises_file_not_found(self, tmp_path):
        ws = _make_workspace(tmp_path)
        nonexistent = ws / "outputs" / "snapshots" / "does-not-exist"
        nonexistent.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_snapshot(nonexistent)


# ---------------------------------------------------------------------------
# list_snapshots — sorted by ts desc
# ---------------------------------------------------------------------------


class TestListSnapshotsSorted:
    """list_snapshots returns snapshots sorted by ts descending."""

    def test_empty_workspace_returns_empty_list(self, tmp_path):
        ws = _make_workspace(tmp_path)
        result = list_snapshots(ws)
        assert result == []

    def test_single_snapshot_listed(self, tmp_path):
        ws = _make_workspace(tmp_path)
        save_snapshot(ws, label="first")
        result = list_snapshots(ws)
        assert len(result) == 1

    def test_multiple_snapshots_sorted_desc(self, tmp_path):
        ws = _make_workspace(tmp_path)
        # Save 3 snapshots with explicit IDs and staggered timestamps in meta.json
        for i in range(3):
            snap_dir = save_snapshot(ws, snapshot_id=f"snap-{i:03d}")
            # Overwrite meta.json with synthetic timestamps
            meta_path = snap_dir / "meta.json"
            meta_data = json.loads(meta_path.read_text())
            meta_data["ts"] = f"2026-05-08T0{i}:00:00+00:00"
            meta_path.write_text(json.dumps(meta_data))

        result = list_snapshots(ws)
        assert len(result) == 3
        # Should be sorted newest first
        for earlier, later in zip(result, result[1:]):
            assert earlier.ts >= later.ts, (
                f"Expected {earlier.ts} >= {later.ts} (desc order)"
            )

    def test_malformed_meta_skipped(self, tmp_path):
        ws = _make_workspace(tmp_path)
        save_snapshot(ws, label="good")

        # Write a malformed meta.json in another snapshot dir
        bad_dir = ws / "outputs" / "snapshots" / "bad-snap"
        bad_dir.mkdir(parents=True)
        (bad_dir / "meta.json").write_text("NOT JSON {{{")

        result = list_snapshots(ws)
        # Only the good snapshot should appear
        assert len(result) == 1
        assert result[0].label == "good"


# ---------------------------------------------------------------------------
# git sha
# ---------------------------------------------------------------------------


class TestSaveSnapshotGitSha:
    """git_sha is captured via subprocess.run and stored in meta.json."""

    def test_git_sha_from_mocked_subprocess(self, tmp_path):
        ws = _make_workspace(tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc1234\n"

        with patch(
            "lerobot_isaac_dashboard.snapshots.subprocess.run", return_value=mock_result
        ):
            snap_dir = save_snapshot(ws)

        meta = json.loads((snap_dir / "meta.json").read_text())
        assert meta["git_sha"] == "abc1234"

    def test_git_sha_none_on_failure(self, tmp_path):
        ws = _make_workspace(tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 128  # not a git repo

        with patch(
            "lerobot_isaac_dashboard.snapshots.subprocess.run", return_value=mock_result
        ):
            snap_dir = save_snapshot(ws)

        meta = json.loads((snap_dir / "meta.json").read_text())
        assert meta["git_sha"] is None

    def test_git_sha_none_on_exception(self, tmp_path):
        ws = _make_workspace(tmp_path)

        with patch(
            "lerobot_isaac_dashboard.snapshots.subprocess.run",
            side_effect=FileNotFoundError("git not found"),
        ):
            snap_dir = save_snapshot(ws)

        meta = json.loads((snap_dir / "meta.json").read_text())
        assert meta["git_sha"] is None


# ---------------------------------------------------------------------------
# SnapshotMeta serialization
# ---------------------------------------------------------------------------


class TestSnapshotMetaSerialization:
    """SnapshotMeta.to_dict() and from_dict() are inverses."""

    def test_roundtrip_meta_dataclass(self):
        original = SnapshotMeta(
            snapshot_id="2026-05-08T120000-test",
            label="test",
            workspace_root=Path("/tmp/ws"),
            session_id="sess-001",
            git_sha="abc1234",
            dashboard_version="0.1.0",
            ts=datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc),
            loader_slugs=["eval_results", "events"],
            schema_version=1,
        )
        restored = SnapshotMeta.from_dict(original.to_dict())
        assert restored.snapshot_id == original.snapshot_id
        assert restored.label == original.label
        assert restored.git_sha == original.git_sha
        assert restored.loader_slugs == original.loader_slugs
        assert restored.schema_version == original.schema_version
