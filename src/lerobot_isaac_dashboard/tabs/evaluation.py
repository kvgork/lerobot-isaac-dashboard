"""evaluation.py — Tab #5: Evaluation / Sim Rollout.

Consumes:
    ``eval_results``  — EVAL_SCHEMA (run_id, task, ts, pc_success, n_episodes,
                                    intervention_rate, mean_ep_len)
    ``curriculum``    — dict with "current" and "history" keys

Figures:
    1. Line chart: pc_success vs timestamp per task
    2. Bar chart: intervention_rate per task (latest run)
    3. Scatter: plateau detection — latest 10 runs vs best pc_success
    4. KPI tiles: overall best pc_success, # tasks, latest run
"""

from __future__ import annotations

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


class EvaluationTab(Tab):
    """Tab #5 — success-rate curves and plateau detection."""

    title = "Evaluation"
    slug = "evaluation"
    primary_loader_slug = "eval_results"

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

        # Sort by timestamp
        if "ts" in df.columns:
            df = df.sort_values("ts")

        # ------------------------------------------------------------------ #
        # Figure 1 — line: pc_success vs timestamp per task
        # ------------------------------------------------------------------ #
        line_fig = go.Figure()
        tasks = df["task"].dropna().unique() if "task" in df.columns else []
        for task in tasks:
            task_df = df[df["task"] == task].dropna(subset=["pc_success"])
            if task_df.empty:
                continue
            line_fig.add_trace(
                go.Scatter(
                    x=list(task_df["ts"].astype(str)),
                    y=list(task_df["pc_success"].astype(float)),
                    mode="lines+markers",
                    name=str(task),
                )
            )

        if not tasks or all(
            df[df["task"] == t].dropna(subset=["pc_success"]).empty for t in tasks
        ):
            # fallback: all data as one trace
            tmp = df.dropna(subset=["pc_success"])
            if not tmp.empty:
                line_fig.add_trace(
                    go.Scatter(
                        x=list(tmp["ts"].astype(str)),
                        y=list(tmp["pc_success"].astype(float)),
                        mode="lines+markers",
                        name="pc_success",
                    )
                )

        line_fig.update_layout(
            title="Success Rate (pc_success) Over Time",
            xaxis_title="Timestamp",
            yaxis_title="pc_success",
            yaxis=dict(range=[0, 1]),
            margin=dict(l=40, r=20, t=40, b=80),
        )
        figs.append(line_fig)
        if container is not None:
            container.plotly_chart(line_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 2 — bar: intervention_rate per task (latest run)
        # ------------------------------------------------------------------ #
        latest_per_task: list[dict] = []
        for task in df["task"].dropna().unique():
            task_rows = df[df["task"] == task].sort_values("ts")
            if task_rows.empty:
                continue
            latest_per_task.append(task_rows.iloc[-1].to_dict())

        if latest_per_task:
            import pandas as pd

            lt_df = pd.DataFrame(latest_per_task)
            bar_fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(lt_df["task"].astype(str)),
                        y=list(lt_df["intervention_rate"].fillna(0).astype(float)),
                        marker_color="coral",
                        name="Intervention Rate",
                    )
                ]
            )
        else:
            bar_fig = go.Figure(data=[go.Bar(x=[], y=[], name="Intervention Rate")])

        bar_fig.update_layout(
            title="Intervention Rate per Task (Latest Run)",
            xaxis_title="Task",
            yaxis_title="Intervention Rate",
            yaxis=dict(range=[0, 1]),
            margin=dict(l=40, r=20, t=40, b=80),
        )
        figs.append(bar_fig)
        if container is not None:
            container.plotly_chart(bar_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 3 — scatter: plateau detection (latest 10 vs best)
        # ------------------------------------------------------------------ #
        valid_pc = df.dropna(subset=["pc_success"]).sort_values("ts")
        last_10 = valid_pc.tail(10)
        best_so_far = valid_pc["pc_success"].cummax()

        plateau_fig = go.Figure()
        if not last_10.empty:
            plateau_fig.add_trace(
                go.Scatter(
                    x=list(range(len(last_10))),
                    y=list(last_10["pc_success"].astype(float)),
                    mode="markers",
                    name="Latest 10 Runs",
                    marker=dict(color="steelblue", size=10),
                )
            )
        if not valid_pc.empty:
            plateau_fig.add_trace(
                go.Scatter(
                    x=list(range(len(valid_pc))),
                    y=list(best_so_far.astype(float)),
                    mode="lines",
                    name="Best So Far",
                    line=dict(color="orange", dash="dash"),
                )
            )

        plateau_fig.update_layout(
            title="Plateau Detection: Latest 10 vs Best",
            xaxis_title="Run Index",
            yaxis_title="pc_success",
            yaxis=dict(range=[0, 1]),
            margin=dict(l=40, r=20, t=40, b=40),
        )
        figs.append(plateau_fig)
        if container is not None:
            container.plotly_chart(plateau_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # KPI tiles
        # ------------------------------------------------------------------ #
        best_pc = valid_pc["pc_success"].max() if not valid_pc.empty else None
        n_tasks = df["task"].nunique()
        latest_run = str(df["run_id"].iloc[-1]) if not df.empty else "—"

        kpi_items: list[tuple[str, Any, str | None]] = [
            ("Best pc_success", f"{best_pc:.3f}" if best_pc is not None else "—", None),
            ("Tasks", n_tasks, None),
            ("Latest Run", latest_run, None),
        ]
        kpi_fig = render_kpi_row(container, kpi_items)
        if kpi_fig is not None:
            figs.append(kpi_fig)

        return figs
