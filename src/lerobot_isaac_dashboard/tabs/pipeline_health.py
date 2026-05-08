"""pipeline_health.py — Tab #8: Pipeline Health.

Consumes:
    ``events``         — EVENTS_SCHEMA (ts, session_id, phase, event, data)
    All other loader results (for count KPIs and file-existence checklist)

Figures:
    1. Timeline: events per phase over time
    2. Table: recent failures (event == "error" or contains "fail")
    3. Heatmap: agent invocations by phase × event type
    4. KPI tiles: run status summary
    5. Table: file-existence checklist (datasets/, outputs/, .agent-state/)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    go = None  # type: ignore[assignment]
    _HAS_PLOTLY = False

try:
    import streamlit as st  # noqa: F401

    _HAS_ST = True
except ImportError:
    st = None  # type: ignore[assignment]
    _HAS_ST = False

from lerobot_isaac_dashboard.tabs._base import Tab, TabContext
from lerobot_isaac_dashboard.tabs._kpis import render_kpi_row


class PipelineHealthTab(Tab):
    """Tab #8 — agent event log, error summary, and workspace checklist."""

    title = "Pipeline Health"
    slug = "pipeline_health"
    primary_loader_slug = "events"

    def render(
        self,
        ctx: TabContext,
        *,
        container: Any = None,
    ) -> list[Any]:
        primary = ctx.loader_results.get(self.primary_loader_slug)
        if primary is None or primary.is_empty:
            if container is not None:
                container.info(
                    f"No {self.title} data yet. See docs/runbook/07-dashboard.md."
                )
            return []

        df = primary.df
        figs: list[Any] = []

        # Sort by time
        if "ts" in df.columns:
            df = df.sort_values("ts")

        # ------------------------------------------------------------------ #
        # Figure 1 — timeline: events per phase
        # ------------------------------------------------------------------ #
        timeline_fig = go.Figure()
        phases = df["phase"].dropna().unique() if "phase" in df.columns else []
        for phase in phases:
            phase_df = df[df["phase"] == phase].dropna(subset=["ts"])
            if phase_df.empty:
                continue
            timeline_fig.add_trace(
                go.Scatter(
                    x=list(phase_df["ts"].astype(str)),
                    y=[str(phase)] * len(phase_df),
                    mode="markers",
                    name=str(phase),
                    marker=dict(size=10),
                    text=list(phase_df["event"].astype(str)),
                )
            )

        timeline_fig.update_layout(
            title="Events per Phase",
            xaxis_title="Timestamp",
            yaxis_title="Phase",
            margin=dict(l=120, r=20, t=40, b=80),
        )
        figs.append(timeline_fig)
        if container is not None:
            container.plotly_chart(timeline_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 2 — table: recent failures
        # ------------------------------------------------------------------ #
        if "event" in df.columns:
            failure_mask = df["event"].str.lower().str.contains("error|fail", na=False)
            failures_df = df[failure_mask].tail(20)
        else:
            failures_df = df.head(0)  # empty

        if not failures_df.empty:
            fail_fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=["<b>Timestamp</b>", "<b>Phase</b>", "<b>Event</b>", "<b>Data</b>"],
                            fill_color="salmon",
                            align="left",
                        ),
                        cells=dict(
                            values=[
                                list(failures_df["ts"].astype(str)),
                                list(failures_df["phase"].astype(str)),
                                list(failures_df["event"].astype(str)),
                                list(failures_df["data"].astype(str)),
                            ],
                            fill_color="lavenderblush",
                            align="left",
                        ),
                    )
                ]
            )
            fail_fig.update_layout(
                title="Recent Failures / Errors",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            figs.append(fail_fig)
            if container is not None:
                container.plotly_chart(fail_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 3 — heatmap: agent invocations by phase × event type
        # ------------------------------------------------------------------ #
        if "phase" in df.columns and "event" in df.columns:
            pivot_df = (
                df.groupby(["phase", "event"])
                .size()
                .reset_index(name="count")
            )
            all_phases = sorted(pivot_df["phase"].astype(str).unique())
            all_events = sorted(pivot_df["event"].astype(str).unique())

            z_matrix: list[list[int]] = []
            for phase in all_phases:
                row_vals: list[int] = []
                for event in all_events:
                    mask = (pivot_df["phase"].astype(str) == phase) & (pivot_df["event"].astype(str) == event)
                    count = int(pivot_df[mask]["count"].sum()) if mask.any() else 0
                    row_vals.append(count)
                z_matrix.append(row_vals)

            heatmap_fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=z_matrix,
                        x=all_events,
                        y=all_phases,
                        colorscale="Blues",
                        showscale=True,
                    )
                ]
            )
            heatmap_fig.update_layout(
                title="Agent Invocations: Phase x Event",
                xaxis_title="Event Type",
                yaxis_title="Phase",
                margin=dict(l=120, r=20, t=40, b=100),
            )
            figs.append(heatmap_fig)
            if container is not None:
                container.plotly_chart(heatmap_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # KPI tiles
        # ------------------------------------------------------------------ #
        total_events = len(df)
        n_errors = int(df["event"].str.lower().str.contains("error|fail", na=False).sum()) if "event" in df.columns else 0
        n_sessions = df["session_id"].nunique() if "session_id" in df.columns else 0

        kpi_items: list[tuple[str, Any, str | None]] = [
            ("Total Events", total_events, None),
            ("Errors/Failures", n_errors, None),
            ("Sessions", n_sessions, None),
        ]
        kpi_fig = render_kpi_row(container, kpi_items)
        if kpi_fig is not None:
            figs.append(kpi_fig)

        # ------------------------------------------------------------------ #
        # Figure 5 — file-existence checklist
        # ------------------------------------------------------------------ #
        checklist_dirs = [
            ("datasets/", ctx.workspace_root / "datasets"),
            ("outputs/", ctx.workspace_root / "outputs"),
            (".agent-state/", ctx.workspace_root / ".agent-state"),
            ("outputs/checkpoints/", ctx.workspace_root / "outputs" / "checkpoints"),
            ("outputs/eval/", ctx.workspace_root / "outputs" / "eval"),
        ]
        checklist_names = [name for name, _ in checklist_dirs]
        checklist_status = [
            "EXISTS" if path.exists() else "MISSING"
            for _, path in checklist_dirs
        ]
        checklist_colors = [
            "lightgreen" if s == "EXISTS" else "lightsalmon"
            for s in checklist_status
        ]

        checklist_fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["<b>Path</b>", "<b>Status</b>"],
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=[checklist_names, checklist_status],
                        fill_color=[["lavender"] * len(checklist_names), checklist_colors],
                        align="left",
                    ),
                )
            ]
        )
        checklist_fig.update_layout(
            title="Workspace File-Existence Checklist",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        figs.append(checklist_fig)
        if container is not None:
            container.plotly_chart(checklist_fig, use_container_width=True)

        return figs
