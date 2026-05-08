"""curriculum.py — Tab #7: Curriculum Stages.

Consumes:
    ``curriculum``    — dict with keys:
        "current" : dict (curriculum_stage.json: stage, task_config, advancement_reason, ts)
        "history" : DataFrame (CURRICULUM_HISTORY_SCHEMA: ts, stage, advancement_reason,
                               task_config_diff)
    ``eval_results``  — EVAL_SCHEMA (for current pc_success overlay)

Figures:
    1. Timeline: stage history over time
    2. KPI: current stage name
    3. Table: advancement triggers vs current pc_success
    4. Diff-table: task config across stages (if task_config_diff present)
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


class CurriculumTab(Tab):
    """Tab #7 — curriculum stage timeline and advancement state."""

    title = "Curriculum"
    slug = "curriculum"
    primary_loader_slug = "curriculum"

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

        data = primary.df
        if not isinstance(data, dict):
            return []

        current = data.get("current", {})
        history_df = data.get("history")
        figs: list[Any] = []

        # ------------------------------------------------------------------ #
        # Figure 1 — timeline: stage history
        # ------------------------------------------------------------------ #
        timeline_fig = go.Figure()
        if history_df is not None and not history_df.empty:
            df_sorted = history_df.sort_values("ts")
            stages = df_sorted["stage"].astype(str).tolist()
            times = df_sorted["ts"].astype(str).tolist()
            reasons = df_sorted["advancement_reason"].fillna("").astype(str).tolist()

            # Encode stages as y positions
            unique_stages = list(dict.fromkeys(stages))  # preserve order
            stage_to_y = {s: i for i, s in enumerate(unique_stages)}
            y_vals = [stage_to_y[s] for s in stages]

            timeline_fig.add_trace(
                go.Scatter(
                    x=times,
                    y=y_vals,
                    mode="lines+markers+text",
                    text=stages,
                    textposition="top center",
                    name="Stage",
                    marker=dict(size=12, color="steelblue"),
                    hovertext=reasons,
                )
            )
            timeline_fig.update_layout(
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(unique_stages))),
                    ticktext=unique_stages,
                )
            )

        timeline_fig.update_layout(
            title="Curriculum Stage History",
            xaxis_title="Timestamp",
            yaxis_title="Stage",
            margin=dict(l=80, r=20, t=40, b=80),
        )
        figs.append(timeline_fig)
        if container is not None:
            container.plotly_chart(timeline_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # KPI: current stage
        # ------------------------------------------------------------------ #
        current_stage = current.get("stage", "—")
        advancement_reason = current.get("advancement_reason", "—")

        kpi_items: list[tuple[str, Any, str | None]] = [
            ("Current Stage", current_stage, None),
            ("Advancement Reason", advancement_reason, None),
        ]
        kpi_fig = render_kpi_row(container, kpi_items)
        if kpi_fig is not None:
            figs.append(kpi_fig)

        # ------------------------------------------------------------------ #
        # Figure 3 — table: advancement triggers vs current pc_success
        # ------------------------------------------------------------------ #
        eval_result = ctx.loader_results.get("eval_results")
        current_pc_success = "—"
        if eval_result is not None and not eval_result.is_empty:
            pc_col = eval_result.df["pc_success"].dropna()
            if not pc_col.empty:
                current_pc_success = f"{float(pc_col.iloc[-1]):.3f}"

        task_config = current.get("task_config", {})
        if task_config and isinstance(task_config, dict):
            trigger_keys = list(task_config.keys())
            trigger_vals = [str(task_config[k]) for k in trigger_keys]
            current_pcs = [
                current_pc_success if k == "pc_success_threshold" else "—"
                for k in trigger_keys
            ]

            trigger_fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=[
                                "<b>Trigger Key</b>",
                                "<b>Required</b>",
                                "<b>Current</b>",
                            ],
                            fill_color="paleturquoise",
                            align="left",
                        ),
                        cells=dict(
                            values=[trigger_keys, trigger_vals, current_pcs],
                            fill_color="lavender",
                            align="left",
                        ),
                    )
                ]
            )
            trigger_fig.update_layout(
                title="Advancement Triggers vs Current Performance",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            figs.append(trigger_fig)
            if container is not None:
                container.plotly_chart(trigger_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 4 — diff-table: task config across stages
        # ------------------------------------------------------------------ #
        if (
            history_df is not None
            and not history_df.empty
            and "task_config_diff" in history_df.columns
        ):
            diff_rows = history_df[["stage", "task_config_diff"]].dropna()
            if not diff_rows.empty:
                diff_stages = list(diff_rows["stage"].astype(str))
                diff_configs = [str(d) for d in diff_rows["task_config_diff"]]

                diff_fig = go.Figure(
                    data=[
                        go.Table(
                            header=dict(
                                values=["<b>Stage</b>", "<b>Task Config Diff</b>"],
                                fill_color="paleturquoise",
                                align="left",
                            ),
                            cells=dict(
                                values=[diff_stages, diff_configs],
                                fill_color="lavender",
                                align="left",
                            ),
                        )
                    ]
                )
                diff_fig.update_layout(
                    title="Task Config Changes Across Stages",
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                figs.append(diff_fig)
                if container is not None:
                    container.plotly_chart(diff_fig, use_container_width=True)

        return figs
