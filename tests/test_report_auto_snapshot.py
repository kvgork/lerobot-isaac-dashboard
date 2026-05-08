"""test_report_auto_snapshot.py — Tests for auto-snapshot on export_report (Phase 8).

Covers:
- export_report with default settings → manifest.snapshot_path is non-null
- export_report with with_snapshot=False → no snapshot, manifest.snapshot_path null
- CLI --no-snapshot flag disables auto-snapshot
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> Path:
    (tmp_path / "datasets").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / ".agent-state").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Auto-snapshot on export_report
# ---------------------------------------------------------------------------


class TestExportReportWritesSnapshot:
    """export_report with default settings auto-saves a snapshot."""

    def test_manifest_snapshot_path_non_null(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws)
        manifest = json.loads((out.parent / "manifest.json").read_text())

        assert manifest.get("snapshot_path") is not None, (
            "manifest.snapshot_path should be non-null when with_snapshot=True (default)"
        )

    def test_snapshot_dir_exists(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws)
        manifest = json.loads((out.parent / "manifest.json").read_text())
        snap_path = Path(manifest["snapshot_path"])

        assert snap_path.exists(), f"Snapshot directory should exist: {snap_path}"
        assert snap_path.is_dir()

    def test_snapshot_contains_meta_json(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws)
        manifest = json.loads((out.parent / "manifest.json").read_text())
        snap_path = Path(manifest["snapshot_path"])

        assert (snap_path / "meta.json").exists()

    def test_snapshot_label_equals_run_id(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws, run_id="my-test-run")
        manifest = json.loads((out.parent / "manifest.json").read_text())
        snap_path = Path(manifest["snapshot_path"])
        snap_meta = json.loads((snap_path / "meta.json").read_text())

        assert snap_meta["label"] == "my-test-run", (
            f"Snapshot label should match run_id, got: {snap_meta['label']!r}"
        )


class TestExportReportNoSnapshotFlag:
    """export_report with with_snapshot=False skips snapshot creation."""

    def test_manifest_snapshot_path_null(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws, with_snapshot=False)
        manifest = json.loads((out.parent / "manifest.json").read_text())

        assert manifest.get("snapshot_path") is None, (
            "manifest.snapshot_path should be null when with_snapshot=False"
        )

    def test_no_snapshot_dir_created(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        export_report(ws, with_snapshot=False)
        snapshots_dir = ws / "outputs" / "snapshots"

        # Either the directory doesn't exist or is empty
        if snapshots_dir.exists():
            snap_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            assert len(snap_dirs) == 0, (
                f"No snapshot dirs should be created when with_snapshot=False, "
                f"found: {snap_dirs}"
            )

    def test_report_html_still_written(self, tmp_path):
        """Disabling snapshot should not break report generation."""
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws, with_snapshot=False)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Manifest snapshot_path field
# ---------------------------------------------------------------------------


class TestManifestSnapshotPathField:
    """manifest.json has a snapshot_path field in both modes."""

    def test_manifest_has_snapshot_path_key(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws)
        manifest = json.loads((out.parent / "manifest.json").read_text())
        assert "snapshot_path" in manifest, (
            "manifest.json must have 'snapshot_path' key"
        )

    def test_manifest_has_snapshot_path_key_when_disabled(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws, with_snapshot=False)
        manifest = json.loads((out.parent / "manifest.json").read_text())
        assert "snapshot_path" in manifest, (
            "manifest.json must have 'snapshot_path' key even when snapshot is disabled"
        )


# ---------------------------------------------------------------------------
# Regression: existing report tests still pass with snapshot_path field
# ---------------------------------------------------------------------------


class TestExistingReportTestsStillPass:
    """Ensure the P5 export tests are not broken by the new snapshot_path field."""

    def test_manifest_still_has_all_original_fields(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws)
        manifest = json.loads((out.parent / "manifest.json").read_text())

        # All original P5 fields must still be present
        required_fields = {
            "run_id",
            "workspace_root",
            "session_id",
            "generated_at",
            "tabs",
        }
        for field in required_fields:
            assert field in manifest, f"Original manifest field '{field}' is missing"

    def test_manifest_tabs_count_still_8(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(ws)
        manifest = json.loads((out.parent / "manifest.json").read_text())
        assert len(manifest["tabs"]) == 8
