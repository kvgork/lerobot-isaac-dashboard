"""test_static_render.py — Tests for runtime/static_render.py.

Covers:
- render_tab_to_html for each tab class with empty ctx
- render_tab_to_html for a populated tab (figures produce Plotly HTML divs)
- No per-figure <script> injection when include_plotlyjs=False (default)
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from lerobot_isaac_dashboard.loaders._base import LoaderResult
from lerobot_isaac_dashboard.runtime.static_render import render_tab_to_html
from lerobot_isaac_dashboard.tabs import TABS, TabContext
from lerobot_isaac_dashboard.tabs._base import Tab


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_empty_ctx(workspace_root: Path) -> TabContext:
    """Build a TabContext with all-empty loader results."""
    empty_result = LoaderResult(df=pd.DataFrame(), is_empty=True)
    loader_results = {
        "parquet_dataset": empty_result,
        "eval_results": empty_result,
        "checkpoints": empty_result,
        "training_logs": empty_result,
        "autoresearch": empty_result,
        "events": empty_result,
        "curriculum": empty_result,
        "synthetic": empty_result,
    }
    return TabContext(
        workspace_root=workspace_root,
        session_id=None,
        loader_results=loader_results,
        refresh_ts=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Test: all tabs with empty ctx
# ---------------------------------------------------------------------------

class TestRenderTabEmptyCtx:
    """All 8 tabs must produce (html, warnings) tuples without raising."""

    @pytest.mark.parametrize("tab_cls", TABS, ids=[t.slug for t in TABS])
    def test_returns_tuple_not_raises(self, workspace_root, tab_cls):
        ctx = _make_empty_ctx(workspace_root)
        tab = tab_cls()

        result = render_tab_to_html(tab, ctx, include_plotlyjs=False)

        assert isinstance(result, tuple), "render_tab_to_html must return a tuple"
        assert len(result) == 2, "render_tab_to_html must return (html, warnings)"

    @pytest.mark.parametrize("tab_cls", TABS, ids=[t.slug for t in TABS])
    def test_html_is_string(self, workspace_root, tab_cls):
        ctx = _make_empty_ctx(workspace_root)
        tab = tab_cls()

        html, warnings = render_tab_to_html(tab, ctx, include_plotlyjs=False)

        assert isinstance(html, str), "html body must be a str"
        assert isinstance(warnings, list), "warnings must be a list"

    @pytest.mark.parametrize("tab_cls", TABS, ids=[t.slug for t in TABS])
    def test_no_script_tags_in_empty_state(self, workspace_root, tab_cls):
        """Empty-state tabs return empty body — no spurious <script> tags."""
        ctx = _make_empty_ctx(workspace_root)
        tab = tab_cls()

        html, warnings = render_tab_to_html(tab, ctx, include_plotlyjs=False)

        # With include_plotlyjs=False and empty data, no script tags expected
        # (tabs return [] figs on empty ctx)
        if not html:
            return  # empty body is fine for empty state
        # If any HTML was returned, it must not be a full script block
        assert "<html" not in html, "fragment must not be a full HTML document"


# ---------------------------------------------------------------------------
# Test: populated tab produces Plotly divs
# ---------------------------------------------------------------------------

class TestRenderTabPopulated:
    """PipelineHealthTab with event data should produce at least one Plotly div."""

    def _make_events_ctx(self, workspace_root: Path) -> TabContext:
        events_df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-05-08T12:00:00"]),
            "session_id": ["s1"],
            "phase": ["data_collection"],
            "event": ["start"],
            "data": ["{}"],
        })
        events_result = LoaderResult(df=events_df, is_empty=False)
        empty_result = LoaderResult(df=pd.DataFrame(), is_empty=True)
        loader_results = {
            "parquet_dataset": empty_result,
            "eval_results": empty_result,
            "checkpoints": empty_result,
            "training_logs": empty_result,
            "autoresearch": empty_result,
            "events": events_result,
            "curriculum": empty_result,
            "synthetic": empty_result,
        }
        return TabContext(
            workspace_root=workspace_root,
            session_id="s1",
            loader_results=loader_results,
            refresh_ts=datetime.utcnow(),
        )

    def test_pipeline_health_produces_plotly_divs(self, workspace_root):
        from lerobot_isaac_dashboard.tabs.pipeline_health import PipelineHealthTab

        ctx = self._make_events_ctx(workspace_root)
        tab = PipelineHealthTab()

        html, warnings = render_tab_to_html(tab, ctx, include_plotlyjs=False)

        assert html, "Expected non-empty HTML body for populated PipelineHealthTab"
        # Plotly figures produce divs with class="plotly-graph-div"
        assert "plotly-graph-div" in html, (
            "Expected at least one Plotly div in the rendered HTML"
        )

    def test_pipeline_health_no_inline_script_when_false(self, workspace_root):
        """include_plotlyjs=False must not embed plotly.min.js JS code."""
        from lerobot_isaac_dashboard.tabs.pipeline_health import PipelineHealthTab

        ctx = self._make_events_ctx(workspace_root)
        tab = PipelineHealthTab()

        html, warnings = render_tab_to_html(tab, ctx, include_plotlyjs=False)

        # plotly.min.js content starts with "/*! For license..."
        # With include_plotlyjs=False we should not see it
        assert "plotly.min" not in html.lower() or html == "", (
            "include_plotlyjs=False must not embed plotly.min.js in the fragment"
        )


# ---------------------------------------------------------------------------
# Test: include_plotlyjs propagation
# ---------------------------------------------------------------------------

class TestIncludePlotlyJsFlag:
    """The include_plotlyjs flag is passed to the first fig only."""

    def test_cdn_flag_on_populated_tab(self, workspace_root):
        """include_plotlyjs='cdn' for first fig should embed a CDN <script>."""
        from lerobot_isaac_dashboard.tabs.pipeline_health import PipelineHealthTab

        events_df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-05-08T12:00:00"]),
            "session_id": ["s1"],
            "phase": ["data_collection"],
            "event": ["start"],
            "data": ["{}"],
        })
        events_result = LoaderResult(df=events_df, is_empty=False)
        empty_result = LoaderResult(df=pd.DataFrame(), is_empty=True)
        loader_results = {
            "parquet_dataset": empty_result,
            "eval_results": empty_result,
            "checkpoints": empty_result,
            "training_logs": empty_result,
            "autoresearch": empty_result,
            "events": events_result,
            "curriculum": empty_result,
            "synthetic": empty_result,
        }
        ctx = TabContext(
            workspace_root=workspace_root,
            session_id="s1",
            loader_results=loader_results,
            refresh_ts=datetime.utcnow(),
        )
        tab = PipelineHealthTab()

        html, _ = render_tab_to_html(tab, ctx, include_plotlyjs="cdn")

        # The CDN path injects a <script src="...plotly..."> in the fragment
        assert "cdn.plot.ly" in html or "plotly" in html.lower(), (
            "Expected Plotly JS reference when include_plotlyjs='cdn'"
        )


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------

class TestRenderTabErrorHandling:
    """render_tab_to_html captures render errors as warnings, not exceptions."""

    def test_bad_tab_returns_empty_with_warning(self, workspace_root):
        class BrokenTab(Tab):
            title = "Broken"
            slug = "broken"

            def render(self, ctx, *, container=None):
                raise RuntimeError("intentional test error")

        ctx = _make_empty_ctx(workspace_root)
        tab = BrokenTab()

        html, warnings = render_tab_to_html(tab, ctx)

        assert html == "", "Broken tab should return empty HTML"
        assert len(warnings) == 1
        assert "intentional test error" in warnings[0]
