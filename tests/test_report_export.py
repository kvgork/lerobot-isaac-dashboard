"""test_report_export.py — Tests for the static HTML report exporter (Phase 5).

Covers:
- Empty workspace smoke test
- session_id propagation to manifest
- CDN mode vs inline (size check)
- Populated workspace with fixture data
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_eval_fixture(workspace_root: Path, task_name: str = "PickAndPlace") -> None:
    """Write a minimal eval JSON file so eval_results loader returns data."""
    eval_dir = workspace_root / "outputs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_file = eval_dir / "eval_20260508_120000.json"
    eval_file.write_text(
        json.dumps(
            {
                "task": task_name,
                "pc_success": 0.85,
                "mean_ep_len": 42.5,
                "n_episodes": 10,
                "timestamp": "2026-05-08T12:00:00",
            }
        ),
        encoding="utf-8",
    )


def _write_events_fixture(
    workspace_root: Path, session_id: str = "test-session"
) -> None:
    """Write a minimal events.jsonl so events loader returns data."""
    session_dir = workspace_root / ".agent-state" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    events_file = session_dir / "events.jsonl"
    events_file.write_text(
        json.dumps(
            {
                "ts": "2026-05-08T12:00:00",
                "session_id": session_id,
                "phase": "data_collection",
                "event": "start",
                "data": "{}",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _has_cdn_script_tag(html: str) -> bool:
    """Return True if the HTML contains an external CDN plotly <script src=> tag.

    This checks for the specific ``<script src="https://cdn.plot.ly/...`` form,
    NOT just any occurrence of ``cdn.plot.ly`` (which also appears inside the
    minified plotly.min.js JS bundle as a URL reference).
    """
    return bool(re.search(r'<script\s[^>]*src=["\']https://cdn\.plot\.ly', html))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReportExportEmptyWorkspace:
    """Smoke tests against a fresh workspace with no data files."""

    def test_report_html_exists(self, workspace_root):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root)

        assert out.exists(), f"report.html not found at {out}"
        assert out.stat().st_size > 0, "report.html is empty"

    def test_report_html_contains_pipeline_health(self, workspace_root):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root)
        content = out.read_text(encoding="utf-8")

        assert "Pipeline Health" in content, (
            "HTML missing 'Pipeline Health' tab heading"
        )

    def test_report_html_has_eight_sections(self, workspace_root):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root)
        content = out.read_text(encoding="utf-8")

        # Count <section id= tags — one per tab
        sections = re.findall(r"<section\s+id=", content)
        assert len(sections) == 8, (
            f"Expected 8 <section id= tags, found {len(sections)}"
        )

    def test_manifest_exists_with_all_tabs(self, workspace_root):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root)
        manifest_path = out.parent / "manifest.json"

        assert manifest_path.exists(), "manifest.json not found"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "tabs" in manifest
        assert len(manifest["tabs"]) == 8, (
            f"Expected 8 tab entries in manifest, got {len(manifest['tabs'])}"
        )

        # Verify required manifest fields
        assert "run_id" in manifest
        assert "workspace_root" in manifest
        assert "generated_at" in manifest

    def test_manifest_tab_slugs_present(self, workspace_root):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root)
        manifest = json.loads(
            (out.parent / "manifest.json").read_text(encoding="utf-8")
        )

        slugs = {t["slug"] for t in manifest["tabs"]}
        expected_slugs = {
            "data_collection",
            "synthetic",
            "policy_training",
            "world_model",
            "evaluation",
            "autoresearch",
            "curriculum",
            "pipeline_health",
        }
        assert slugs == expected_slugs, f"Manifest tab slugs mismatch: {slugs}"


class TestReportExportSessionId:
    """Session ID is propagated to manifest and run_id."""

    def test_session_id_in_manifest(self, workspace_root):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root, session_id="test-session")
        manifest = json.loads(
            (out.parent / "manifest.json").read_text(encoding="utf-8")
        )

        assert manifest["session_id"] == "test-session"

    def test_session_id_in_run_id(self, workspace_root):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root, session_id="test-session")
        manifest = json.loads(
            (out.parent / "manifest.json").read_text(encoding="utf-8")
        )

        assert "test-session" in manifest["run_id"], (
            f"Expected 'test-session' in run_id, got: {manifest['run_id']}"
        )


class TestReportExportCdnMode:
    """CDN mode produces smaller output and contains the CDN <script> tag."""

    def test_cdn_mode_contains_cdnplotly_script_tag(self, workspace_root):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root, inline_plotlyjs=False)
        content = out.read_text(encoding="utf-8")

        assert _has_cdn_script_tag(content), (
            "CDN mode should contain a <script src='https://cdn.plot.ly/...'> tag"
        )

    def test_cdn_mode_smaller_than_inline(self, workspace_root, tmp_path):
        from lerobot_isaac_dashboard.report import export_report

        out_inline = export_report(
            workspace_root,
            output_dir=tmp_path / "inline",
            inline_plotlyjs=True,
        )
        out_cdn = export_report(
            workspace_root,
            output_dir=tmp_path / "cdn",
            inline_plotlyjs=False,
        )

        size_inline = out_inline.stat().st_size
        size_cdn = out_cdn.stat().st_size

        # Inline includes ~3 MB of plotly.min.js; CDN just has a <script> tag.
        assert size_inline > size_cdn + 100_000, (
            f"Expected inline ({size_inline} B) to be >100 KB larger than CDN ({size_cdn} B)"
        )

    def test_inline_mode_does_not_contain_cdn_script_tag(self, workspace_root):
        """Inline mode must NOT include a <script src='https://cdn.plot.ly...'>  tag.

        Note: the inline plotly.min.js bundle itself contains the string
        'cdn.plot.ly' as a URL literal inside its JS code — that is expected.
        This test checks only for the external <script src=...> CDN tag.
        """
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root, inline_plotlyjs=True)
        content = out.read_text(encoding="utf-8")

        assert not _has_cdn_script_tag(content), (
            "Inline mode must not include a <script src='https://cdn.plot.ly/...'> CDN tag"
        )


class TestReportExportPopulated:
    """With fixture data the exporter succeeds and includes data-derived content."""

    def test_populated_eval_succeeds(self, workspace_root):
        _write_eval_fixture(workspace_root, task_name="PickAndPlace")

        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root)
        assert out.exists()

    def test_populated_eval_data_in_html(self, workspace_root):
        _write_eval_fixture(workspace_root, task_name="PickAndPlace")

        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root)
        content = out.read_text(encoding="utf-8")

        # The eval loader should surface the task name in a Plotly figure or table
        assert "PickAndPlace" in content, (
            "Expected eval task name 'PickAndPlace' to appear in rendered HTML"
        )

    def test_populated_events_succeeds(self, workspace_root):
        _write_events_fixture(workspace_root, session_id="s1")

        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root, session_id="s1")
        assert out.exists()


class TestReportExportCustomOutputDir:
    """output_dir and run_id overrides work correctly."""

    def test_custom_output_dir(self, workspace_root, tmp_path):
        from lerobot_isaac_dashboard.report import export_report

        custom_dir = tmp_path / "my_report"
        out = export_report(workspace_root, output_dir=custom_dir)

        assert out.parent == custom_dir.resolve()
        assert out.name == "report.html"

    def test_custom_run_id(self, workspace_root, tmp_path):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(
            workspace_root,
            output_dir=tmp_path / "r",
            run_id="my-custom-run-id",
        )
        manifest = json.loads(
            (out.parent / "manifest.json").read_text(encoding="utf-8")
        )

        assert manifest["run_id"] == "my-custom-run-id"


class TestReportExportWithCsv:
    """--with-csv writes data/*.csv files."""

    def test_with_csv_creates_data_dir(self, workspace_root, tmp_path):
        from lerobot_isaac_dashboard.report import export_report

        out = export_report(workspace_root, output_dir=tmp_path / "r", with_csv=True)
        data_dir = out.parent / "data"

        # data/ directory created even if all loaders are empty
        assert data_dir.exists(), "data/ directory should be created with --with-csv"


class TestRunLoadersHeadless:
    """run_loaders_headless mirrors LOADERS and never raises."""

    def test_returns_all_slugs(self, workspace_root):
        from lerobot_isaac_dashboard.report import run_loaders_headless

        results = run_loaders_headless(workspace_root)

        expected_slugs = {
            "parquet_dataset",
            "eval_results",
            "checkpoints",
            "training_logs",
            "autoresearch",
            "events",
            "curriculum",
            "synthetic",
        }
        assert set(results.keys()) == expected_slugs

    def test_all_results_are_loader_result(self, workspace_root):
        from lerobot_isaac_dashboard.loaders import LoaderResult
        from lerobot_isaac_dashboard.report import run_loaders_headless

        results = run_loaders_headless(workspace_root)

        for slug, result in results.items():
            assert isinstance(result, LoaderResult), (
                f"Slug {slug!r} returned non-LoaderResult"
            )

    def test_empty_workspace_all_is_empty(self, workspace_root):
        from lerobot_isaac_dashboard.report import run_loaders_headless

        results = run_loaders_headless(workspace_root)

        for slug, result in results.items():
            assert result.is_empty, (
                f"Expected is_empty=True for {slug!r} on empty workspace"
            )
