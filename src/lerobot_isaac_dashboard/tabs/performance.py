"""performance.py — Tab: GPU & training throughput.

Consumes:
    ``system_metrics`` — SYSTEM_METRICS_SCHEMA
        (ts, elapsed_s, stage, run_id, gpu_index,
         utilization_pct, memory_used_mb, memory_total_mb, memory_pct,
         temperature_c, power_draw_w)
    ``training_logs``  — to derive ``steps_per_sec`` per (arch, run_id)

Figures (each with a per-stage legend so the user can overlay
LeRobot policy runs against world-model runs):
    1. GPU utilization % over time   (one trace per stage × run)
    2. GPU memory MB over time       (same grouping)
    3. GPU temperature °C over time
    4. GPU power draw W over time
    5. Training throughput steps/sec from training_logs (parsed)
    6. Stage comparison table (avg/max util, mem, temp, power per stage)

KPIs:
    - Latest util %, mem GB, temp °C, power W
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

import pandas as pd

from lerobot_isaac_dashboard.tabs._base import Tab, TabContext
from lerobot_isaac_dashboard.tabs._kpis import render_kpi_row

# Map raw `stage` labels to a friendly "family" so the legend groups all
# LeRobot policy runs vs all world-model runs at a glance.
_POLICY_STAGES = {"policy_train", "policy", "lerobot", "diffusion", "smolvla", "act"}
_WM_STAGES = {"wm_train", "worldmodel", "dreamerv3", "le_world_model", "lewm"}


def _family(stage: str) -> str:
    s = (stage or "").lower()
    if s in _POLICY_STAGES or "policy" in s or "lerobot" in s:
        return "LeRobot policy"
    if s in _WM_STAGES or "world" in s or "dreamer" in s or "lewm" in s:
        return "World model"
    if "eval" in s:
        return "Eval"
    if "synthetic" in s or "dr" in s:
        return "Synthetic"
    return s or "unknown"


def _trace_label(row_stage: str, run_id: str) -> str:
    return f"{_family(row_stage)} · {run_id}"


class PerformanceTab(Tab):
    """GPU + throughput Performance tab."""

    title = "Performance"
    slug = "performance"
    primary_loader_slug = "system_metrics"

    def render(
        self,
        ctx: TabContext,
        *,
        container: Any = None,
    ) -> list[Any]:
        primary = ctx.loader_results.get(self.primary_loader_slug)
        figures: list[Any] = []

        if primary is None or primary.is_empty:
            if container is not None:
                container.info(
                    "No system_metrics yet. The full-pipeline runner spawns "
                    "scripts/_gpu_monitor.py during training stages and writes "
                    "outputs/<run>/system_metrics/gpu_metrics.parquet."
                )
            return figures

        df: pd.DataFrame = primary.df.copy()
        if df.empty:
            if container is not None:
                container.info("system_metrics parquet found but empty.")
            return figures

        df["family"] = df["stage"].fillna("unknown").map(_family)
        df["label"] = [
            _trace_label(s, r) for s, r in zip(df["stage"], df["run_id"])
        ]

        # --- KPI strip: latest sample per metric ---------------------------------
        latest = df.sort_values("ts").groupby("gpu_index").tail(1).iloc[-1]
        kpi_items: list[tuple[str, Any, str | None]] = [
            ("GPU util", f"{float(latest['utilization_pct']):.0f}%", None),
            (
                "GPU mem",
                f"{float(latest['memory_used_mb']) / 1024:.1f} GB",
                f"{float(latest['memory_pct']):.0f}%",
            ),
            ("Temp", f"{float(latest['temperature_c']):.0f} °C", None),
            ("Power", f"{float(latest['power_draw_w']):.0f} W", None),
        ]
        if container is not None and _HAS_ST:
            render_kpi_row(container, kpi_items)

        # --- Optional widgets: stage filter + comparison mode --------------------
        families = sorted(df["family"].unique().tolist())
        selected_families: list[str] = families
        compare_mode = "overlay"
        if container is not None and _HAS_ST:
            with container.expander("Comparison filters", expanded=False):
                selected_families = container.multiselect(
                    "Show stage families",
                    families,
                    default=families,
                    key="perf_family_filter",
                )
                compare_mode = container.radio(
                    "Compare by",
                    ("overlay", "average per family"),
                    index=0,
                    horizontal=True,
                    key="perf_compare_mode",
                )

        plot_df = df[df["family"].isin(selected_families)].copy() if selected_families else df.copy()

        # --- Figure helpers ------------------------------------------------------
        def _line(metric_col: str, title: str, y_unit: str) -> Any | None:
            if not _HAS_PLOTLY or plot_df.empty:
                return None
            fig = go.Figure()
            if compare_mode == "average per family":
                # Resample per family to elapsed_s buckets (5 s).
                pdf = plot_df.copy()
                pdf["bucket"] = (pdf["elapsed_s"] // 5) * 5
                agg = (
                    pdf.groupby(["family", "bucket"])[metric_col]
                    .mean()
                    .reset_index()
                )
                for fam, sub in agg.groupby("family"):
                    fig.add_trace(
                        go.Scatter(
                            x=sub["bucket"],
                            y=sub[metric_col],
                            mode="lines",
                            name=str(fam),
                        )
                    )
                x_title = "elapsed (s)"
            else:
                for label, sub in plot_df.groupby("label"):
                    sub = sub.sort_values("elapsed_s")
                    fig.add_trace(
                        go.Scatter(
                            x=sub["elapsed_s"],
                            y=sub[metric_col],
                            mode="lines",
                            name=str(label),
                        )
                    )
                x_title = "elapsed (s)"
            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title=y_unit,
                hovermode="x unified",
                height=340,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            return fig

        for col, title, unit in (
            ("utilization_pct", "GPU Utilization", "%"),
            ("memory_used_mb", "GPU Memory", "MB"),
            ("temperature_c", "GPU Temperature", "°C"),
            ("power_draw_w", "GPU Power Draw", "W"),
        ):
            fig = _line(col, title, unit)
            if fig is not None:
                figures.append(fig)
                if container is not None and _HAS_ST:
                    container.plotly_chart(fig, use_container_width=True)

        # --- Training throughput (steps/sec) from training_logs ------------------
        tl = ctx.loader_results.get("training_logs")
        if tl is not None and not tl.is_empty and _HAS_PLOTLY:
            tdf = tl.df.copy()
            # Pick the step + per-row mtime; compute steps/sec per (arch, run_id).
            if {"arch", "run_id", "step", "ts"}.issubset(tdf.columns):
                tdf = tdf.dropna(subset=["step", "ts"]).copy()
                if not tdf.empty:
                    tdf["ts"] = pd.to_datetime(tdf["ts"], errors="coerce", utc=True)
                    tdf = tdf.dropna(subset=["ts"])
                    tdf = tdf.sort_values(["arch", "run_id", "step"])
                    tdf["dt"] = tdf.groupby(["arch", "run_id"])["ts"].diff().dt.total_seconds()
                    tdf["dstep"] = tdf.groupby(["arch", "run_id"])["step"].diff()
                    tdf["sps"] = tdf["dstep"] / tdf["dt"]
                    tdf = tdf[(tdf["sps"] > 0) & (tdf["sps"] < 1e4)]
                    if not tdf.empty:
                        fig = go.Figure()
                        for (arch, run_id), sub in tdf.groupby(["arch", "run_id"]):
                            label = _trace_label(arch, run_id)
                            fig.add_trace(
                                go.Scatter(
                                    x=sub["step"],
                                    y=sub["sps"].rolling(window=10, min_periods=1).mean(),
                                    mode="lines",
                                    name=label,
                                )
                            )
                        fig.update_layout(
                            title="Training Throughput (steps/sec, 10-pt rolling)",
                            xaxis_title="step",
                            yaxis_title="steps/sec",
                            hovermode="x unified",
                            height=340,
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        figures.append(fig)
                        if container is not None and _HAS_ST:
                            container.plotly_chart(fig, use_container_width=True)

        # --- Stage comparison summary table -------------------------------------
        if _HAS_PLOTLY:
            agg = (
                plot_df.groupby("family")
                .agg(
                    samples=("elapsed_s", "count"),
                    util_mean=("utilization_pct", "mean"),
                    util_max=("utilization_pct", "max"),
                    mem_mean_mb=("memory_used_mb", "mean"),
                    mem_max_mb=("memory_used_mb", "max"),
                    temp_mean=("temperature_c", "mean"),
                    temp_max=("temperature_c", "max"),
                    power_mean=("power_draw_w", "mean"),
                    power_max=("power_draw_w", "max"),
                )
                .round(1)
                .reset_index()
            )
            if not agg.empty:
                fig = go.Figure(
                    data=[
                        go.Table(
                            header=dict(
                                values=list(agg.columns),
                                fill_color="paleturquoise",
                                align="left",
                            ),
                            cells=dict(
                                values=[agg[c] for c in agg.columns],
                                fill_color="lavender",
                                align="left",
                            ),
                        )
                    ]
                )
                fig.update_layout(
                    title="Stage Comparison — average / peak per family",
                    height=80 + 30 * len(agg),
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                figures.append(fig)
                if container is not None and _HAS_ST:
                    container.plotly_chart(fig, use_container_width=True)

        return figures
