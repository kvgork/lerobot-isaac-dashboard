"""synthetic.py — Tab #2: Synthetic Data.

Consumes:
    ``synthetic``        — SYNTHETIC_SCHEMA (episode_index, length, source, task)
    ``parquet_dataset``  — DATASET_SUMMARY_SCHEMA (mtime column for growth line)

Figures:
    1. Stacked bar: episode count by source (real / sim_dr / mimicgen) per task
    2. Line chart: cumulative episode growth over time (via dataset mtime)
    3. Table: dedup summary (source counts, total)
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


class SyntheticTab(Tab):
    """Tab #2 — shows breakdown of real vs synthetic episodes."""

    title = "Synthetic Data"
    slug = "synthetic"
    primary_loader_slug = "synthetic"

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

        # ------------------------------------------------------------------ #
        # Figure 1 — stacked bar: episodes by source per task
        # ------------------------------------------------------------------ #
        sources = ["real", "sim_dr", "mimicgen"]
        colors = {"real": "steelblue", "sim_dr": "coral", "mimicgen": "mediumseagreen"}

        tasks = sorted(df["task"].dropna().unique()) if "task" in df.columns and df["task"].notna().any() else ["(all)"]
        if not tasks:
            tasks = ["(all)"]

        bar_data: list[Any] = []
        for src in sources:
            counts: list[int] = []
            for task in tasks:
                if task == "(all)":
                    mask = df["source"] == src
                else:
                    mask = (df["source"] == src) & (df["task"] == task)
                counts.append(int(mask.sum()))
            bar_data.append(
                go.Bar(
                    name=src,
                    x=list(tasks),
                    y=counts,
                    marker_color=colors.get(src, "grey"),
                )
            )

        stacked_fig = go.Figure(data=bar_data)
        stacked_fig.update_layout(
            barmode="stack",
            title="Episodes by Source per Task",
            xaxis_title="Task",
            yaxis_title="# Episodes",
            margin=dict(l=40, r=20, t=40, b=80),
        )
        figs.append(stacked_fig)
        if container is not None:
            container.plotly_chart(stacked_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 2 — line: dataset growth over time (mtime from parquet_dataset)
        # ------------------------------------------------------------------ #
        ds_result = ctx.loader_results.get("parquet_dataset")
        if ds_result is not None and not ds_result.is_empty:
            ds_df = ds_result.df.dropna(subset=["mtime", "n_episodes"])
            if not ds_df.empty:
                ds_sorted = ds_df.sort_values("mtime")
                growth_fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=list(ds_sorted["mtime"].astype(str)),
                            y=list(ds_sorted["n_episodes"].astype(float).cumsum()),
                            mode="lines+markers",
                            name="Cumulative Episodes",
                            marker_color="purple",
                        )
                    ]
                )
                growth_fig.update_layout(
                    title="Dataset Growth Over Time (by last-modified)",
                    xaxis_title="Last Modified (mtime)",
                    yaxis_title="Cumulative Episodes",
                    margin=dict(l=40, r=20, t=40, b=80),
                )
                figs.append(growth_fig)
                if container is not None:
                    container.plotly_chart(growth_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 3 — table: dedup / source summary
        # ------------------------------------------------------------------ #
        source_counts = df["source"].value_counts().to_dict()
        table_sources = list(source_counts.keys())
        table_counts = [source_counts[s] for s in table_sources]
        table_pct = [
            f"{100 * c / len(df):.1f}%" if len(df) > 0 else "—"
            for c in table_counts
        ]

        dedup_fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["<b>Source</b>", "<b>Episodes</b>", "<b>Fraction</b>"],
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=[table_sources, table_counts, table_pct],
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )
        dedup_fig.update_layout(
            title="Source Breakdown",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        figs.append(dedup_fig)
        if container is not None:
            container.plotly_chart(dedup_fig, use_container_width=True)

        return figs
