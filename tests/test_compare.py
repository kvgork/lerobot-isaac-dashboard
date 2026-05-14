"""test_compare.py — Tests for compare.py (Phase 8).

Covers:
- render_compare_2way smoke (two empty snapshots → 8 tab slugs)
- render_compare_nway smoke (three snapshots → 8 tab slugs)
- export_compare_report 2way — writes HTML with both IDs and compare-col divs
- export_compare_report nway — writes HTML with all snapshot labels
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lerobot_isaac_dashboard.snapshots import save_snapshot
from lerobot_isaac_dashboard.tabs import TABS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> Path:
    (tmp_path / "datasets").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / ".agent-state").mkdir()
    return tmp_path


def _make_snap(ws: Path, label: str) -> str:
    """Save a snapshot and return its snapshot_id."""
    snap_dir = save_snapshot(ws, label=label)
    import json

    meta = json.loads((snap_dir / "meta.json").read_text())
    return meta["snapshot_id"]


# ---------------------------------------------------------------------------
# render_compare_2way
# ---------------------------------------------------------------------------


class TestRenderCompare2way:
    """render_compare_2way returns a dict with 8 tab slugs."""

    def test_returns_all_8_tab_slugs(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_a = save_snapshot(ws, label="A")
        snap_b = save_snapshot(ws, label="B")

        from lerobot_isaac_dashboard.compare import render_compare_2way
        from lerobot_isaac_dashboard.snapshots import load_snapshot

        entry_a = load_snapshot(snap_a)
        entry_b = load_snapshot(snap_b)

        result = render_compare_2way(entry_a, entry_b, container=None)

        expected_slugs = {tab_cls.slug for tab_cls in TABS}
        assert set(result.keys()) == expected_slugs, (
            f"Expected tab slugs: {expected_slugs}, got: {set(result.keys())}"
        )

    def test_values_are_lists(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_a = save_snapshot(ws, label="A")
        snap_b = save_snapshot(ws, label="B")

        from lerobot_isaac_dashboard.compare import render_compare_2way
        from lerobot_isaac_dashboard.snapshots import load_snapshot

        entry_a = load_snapshot(snap_a)
        entry_b = load_snapshot(snap_b)

        result = render_compare_2way(entry_a, entry_b, container=None)
        for slug, figs in result.items():
            assert isinstance(figs, list), (
                f"Expected list for slug {slug!r}, got {type(figs)}"
            )


# ---------------------------------------------------------------------------
# render_compare_nway
# ---------------------------------------------------------------------------


class TestRenderCompareNway:
    """render_compare_nway with 3 snapshots returns 8 tab slugs."""

    def test_returns_all_8_tab_slugs(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snaps = [save_snapshot(ws, label=f"snap-{i}") for i in range(3)]

        from lerobot_isaac_dashboard.compare import render_compare_nway
        from lerobot_isaac_dashboard.snapshots import load_snapshot

        loaded = [load_snapshot(s) for s in snaps]
        result = render_compare_nway(loaded, container=None)

        expected_slugs = {tab_cls.slug for tab_cls in TABS}
        assert set(result.keys()) == expected_slugs

    def test_values_are_lists(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snaps = [save_snapshot(ws, label=f"n-{i}") for i in range(3)]

        from lerobot_isaac_dashboard.compare import render_compare_nway
        from lerobot_isaac_dashboard.snapshots import load_snapshot

        loaded = [load_snapshot(s) for s in snaps]
        result = render_compare_nway(loaded, container=None)
        for slug, figs in result.items():
            assert isinstance(figs, list)

    def test_empty_list_returns_empty_dicts(self, tmp_path):
        from lerobot_isaac_dashboard.compare import render_compare_nway

        result = render_compare_nway([], container=None)
        # Should return a dict keyed by tab slugs
        assert isinstance(result, dict)
        for slug, figs in result.items():
            assert figs == []


# ---------------------------------------------------------------------------
# export_compare_report — 2way
# ---------------------------------------------------------------------------


class TestExportCompareReport2way:
    """export_compare_report in 2way mode writes HTML with expected structure."""

    def test_report_html_exists(self, tmp_path):
        ws = _make_workspace(tmp_path)
        id_a = _make_snap(ws, "A")
        id_b = _make_snap(ws, "B")

        from lerobot_isaac_dashboard.compare import export_compare_report

        out = export_compare_report(ws, [id_a, id_b], mode="2way")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_contains_both_snapshot_labels(self, tmp_path):
        ws = _make_workspace(tmp_path)
        id_a = _make_snap(ws, "my-label-A")
        id_b = _make_snap(ws, "my-label-B")

        from lerobot_isaac_dashboard.compare import export_compare_report

        out = export_compare_report(ws, [id_a, id_b], mode="2way")
        content = out.read_text(encoding="utf-8")

        assert "my-label-A" in content, "HTML should contain label A"
        assert "my-label-B" in content, "HTML should contain label B"

    def test_contains_all_sections(self, tmp_path):
        ws = _make_workspace(tmp_path)
        id_a = _make_snap(ws, "A")
        id_b = _make_snap(ws, "B")

        from lerobot_isaac_dashboard.compare import export_compare_report
        from lerobot_isaac_dashboard.tabs import TABS

        out = export_compare_report(ws, [id_a, id_b], mode="2way")
        content = out.read_text(encoding="utf-8")

        sections = re.findall(r"<section\s+id=", content)
        assert len(sections) == len(TABS), (
            f"Expected {len(TABS)} sections, got {len(sections)}"
        )

    def test_contains_compare_col_divs(self, tmp_path):
        ws = _make_workspace(tmp_path)
        id_a = _make_snap(ws, "alpha")
        id_b = _make_snap(ws, "beta")

        from lerobot_isaac_dashboard.compare import export_compare_report
        from lerobot_isaac_dashboard.tabs import TABS

        out = export_compare_report(ws, [id_a, id_b], mode="2way")
        content = out.read_text(encoding="utf-8")

        compare_cols = re.findall(r'class="compare-col"', content)
        # N tabs × 2 columns
        assert len(compare_cols) == 2 * len(TABS), (
            f"Expected {2 * len(TABS)} compare-col divs ({len(TABS)} tabs × 2 cols), "
            f"got {len(compare_cols)}"
        )

    def test_custom_output_dir(self, tmp_path):
        ws = _make_workspace(tmp_path)
        id_a = _make_snap(ws, "A")
        id_b = _make_snap(ws, "B")

        from lerobot_isaac_dashboard.compare import export_compare_report

        custom_out = tmp_path / "my-compare-report"
        out = export_compare_report(
            ws, [id_a, id_b], mode="2way", output_dir=custom_out
        )
        assert out.parent.resolve() == custom_out.resolve()

    def test_raises_with_fewer_than_2_snapshots(self, tmp_path):
        ws = _make_workspace(tmp_path)
        id_a = _make_snap(ws, "only-one")

        from lerobot_isaac_dashboard.compare import export_compare_report

        with pytest.raises(ValueError, match="at least 2"):
            export_compare_report(ws, [id_a], mode="2way")


# ---------------------------------------------------------------------------
# export_compare_report — nway
# ---------------------------------------------------------------------------


class TestExportCompareReportNway:
    """export_compare_report in nway mode writes overlay HTML."""

    def test_report_html_exists(self, tmp_path):
        ws = _make_workspace(tmp_path)
        ids = [_make_snap(ws, f"run-{i}") for i in range(3)]

        from lerobot_isaac_dashboard.compare import export_compare_report

        out = export_compare_report(ws, ids, mode="nway")
        assert out.exists()

    def test_contains_all_snapshot_labels(self, tmp_path):
        ws = _make_workspace(tmp_path)
        labels = ["run-alpha", "run-beta", "run-gamma"]
        ids = [_make_snap(ws, lbl) for lbl in labels]

        from lerobot_isaac_dashboard.compare import export_compare_report

        out = export_compare_report(ws, ids, mode="nway")
        content = out.read_text(encoding="utf-8")

        for lbl in labels:
            assert lbl in content, f"Expected label '{lbl}' in HTML"

    def test_contains_all_sections(self, tmp_path):
        ws = _make_workspace(tmp_path)
        ids = [_make_snap(ws, f"s{i}") for i in range(3)]

        from lerobot_isaac_dashboard.compare import export_compare_report
        from lerobot_isaac_dashboard.tabs import TABS

        out = export_compare_report(ws, ids, mode="nway")
        content = out.read_text(encoding="utf-8")
        sections = re.findall(r"<section\s+id=", content)
        assert len(sections) == len(TABS)

    def test_cdn_mode(self, tmp_path):
        ws = _make_workspace(tmp_path)
        ids = [_make_snap(ws, f"cdn-{i}") for i in range(2)]

        from lerobot_isaac_dashboard.compare import export_compare_report

        out = export_compare_report(ws, ids, mode="nway", inline_plotlyjs=False)
        content = out.read_text(encoding="utf-8")
        # CDN mode → <script src=...cdn.plot.ly...> tag
        assert "cdn.plot.ly" in content


# ---------------------------------------------------------------------------
# build_compare_context
# ---------------------------------------------------------------------------


class TestBuildCompareContext:
    """build_compare_context wraps snapshots into a CompareContext."""

    def test_context_has_correct_snapshots(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snaps = [save_snapshot(ws, label=f"ctx-{i}") for i in range(2)]

        from lerobot_isaac_dashboard.compare import build_compare_context
        from lerobot_isaac_dashboard.snapshots import load_snapshot

        loaded = [load_snapshot(s) for s in snaps]
        ctx = build_compare_context(loaded, mode="2way")

        assert len(ctx.snapshots) == 2
        assert ctx.mode == "2way"

    def test_labels_property(self, tmp_path):
        ws = _make_workspace(tmp_path)
        snap_a = save_snapshot(ws, label="label-alpha")
        snap_b = save_snapshot(ws, label="label-beta")

        from lerobot_isaac_dashboard.compare import build_compare_context
        from lerobot_isaac_dashboard.snapshots import load_snapshot

        loaded = [load_snapshot(snap_a), load_snapshot(snap_b)]
        ctx = build_compare_context(loaded, mode="nway")

        assert "label-alpha" in ctx.labels
        assert "label-beta" in ctx.labels
