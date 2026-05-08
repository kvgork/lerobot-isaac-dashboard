"""data_collection.py — Tab #1: Data Collection.

Consumes:
    ``parquet_dataset``  — DATASET_SUMMARY_SCHEMA

Figures:
    1. Bar chart: n_episodes per repo_id
    2. Histogram: episode length distribution (n_frames / n_episodes proxy)
    3. KPI tiles: total episodes, total frames, intervention rate N/A note
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


class DataCollectionTab(Tab):
    """Tab #1 — summarises collected LeRobot v3 datasets."""

    title = "Data Collection"
    slug = "data_collection"
    primary_loader_slug = "parquet_dataset"

    def render(
        self,
        ctx: TabContext,
        *,
        container: Any = None,
    ) -> list[Any]:
        """Render Data Collection tab figures.

        Returns
        -------
        list[go.Figure]
            [bar_episodes, hist_lengths, kpi_table_or_none_filtered]
        """
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
        # Figure 1 — bar chart: n_episodes per repo_id
        # ------------------------------------------------------------------ #
        bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=list(df["repo_id"].astype(str)),
                    y=list(df["n_episodes"].fillna(0).astype(float)),
                    name="Episodes",
                    marker_color="steelblue",
                )
            ]
        )
        bar_fig.update_layout(
            title="Episodes per Dataset",
            xaxis_title="Dataset (repo_id)",
            yaxis_title="# Episodes",
            margin=dict(l=40, r=20, t=40, b=80),
        )
        figs.append(bar_fig)
        if container is not None:
            container.plotly_chart(bar_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 2 — histogram: mean episode length per dataset
        # ------------------------------------------------------------------ #
        # Use n_frames / n_episodes as episode length proxy (actual per-ep
        # length requires reading individual episode parquets)
        valid = df.dropna(subset=["n_frames", "n_episodes"])
        if not valid.empty:
            mean_len = (
                valid["n_frames"].astype(float) / valid["n_episodes"].astype(float)
            )
            hist_fig = go.Figure(
                data=[
                    go.Histogram(
                        x=list(mean_len),
                        nbinsx=20,
                        name="Mean episode length",
                        marker_color="coral",
                    )
                ]
            )
        else:
            hist_fig = go.Figure(
                data=[go.Histogram(x=[], name="Mean episode length")]
            )
        hist_fig.update_layout(
            title="Distribution of Mean Episode Length (frames)",
            xaxis_title="Frames per Episode (mean)",
            yaxis_title="# Datasets",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        figs.append(hist_fig)
        if container is not None:
            container.plotly_chart(hist_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # KPI tiles
        # ------------------------------------------------------------------ #
        total_episodes = int(df["n_episodes"].fillna(0).sum())
        total_frames = int(df["n_frames"].fillna(0).sum())
        n_datasets = len(df)

        kpi_items: list[tuple[str, Any, str | None]] = [
            ("Total Episodes", total_episodes, None),
            ("Total Frames", total_frames, None),
            ("Datasets", n_datasets, None),
            ("Intervention Rate", "N/A (see runbook)", None),
        ]
        kpi_fig = render_kpi_row(container, kpi_items)
        if kpi_fig is not None:
            figs.append(kpi_fig)

        return figs
