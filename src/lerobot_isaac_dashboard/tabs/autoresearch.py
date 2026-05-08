"""autoresearch.py — Tab #6: Autoresearch Loop.

Consumes:
    ``autoresearch``  — dict with keys:
        "history" : DataFrame (HISTORY_SCHEMA: session_id, slug, trial, metric_name,
                               metric_value, config, ts, status)
        "program" : dict
        "best"    : dict (best_config.yaml)
        "plateau" : dict (plateau_tracker.json)

Figures:
    1. Line: metric_value over trial index with best-so-far envelope
    2. Scatter: one config dimension vs metric (small-multiples where possible)
    3. Bar: operator usage frequency
    4. Table: best_config contents
    5. Gauge: plateau score (0..1 if present)
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


class AutoresearchTab(Tab):
    """Tab #6 — HP search history and plateau state."""

    title = "Autoresearch"
    slug = "autoresearch"
    primary_loader_slug = "autoresearch"

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

        # autoresearch loader returns a dict df
        data = primary.df
        if not isinstance(data, dict):
            return []

        history_df = data.get("history")
        best_config = data.get("best", {})
        plateau_data = data.get("plateau", {})

        figs: list[Any] = []

        # ------------------------------------------------------------------ #
        # Figure 1 — metric_value over trial with best-so-far envelope
        # ------------------------------------------------------------------ #
        line_fig = go.Figure()

        if history_df is not None and not history_df.empty:
            df_sorted = history_df.sort_values("trial")
            metric_vals = df_sorted["metric_value"].astype(float)
            trials = df_sorted["trial"].astype(float)

            line_fig.add_trace(
                go.Scatter(
                    x=list(trials),
                    y=list(metric_vals),
                    mode="lines+markers",
                    name="metric_value",
                    marker_color="steelblue",
                )
            )

            # Best-so-far envelope (cummax, but take the max for maximisation)
            best_so_far = metric_vals.cummax()
            line_fig.add_trace(
                go.Scatter(
                    x=list(trials),
                    y=list(best_so_far),
                    mode="lines",
                    name="Best So Far",
                    line=dict(color="orange", dash="dash"),
                )
            )

        line_fig.update_layout(
            title="HP Search: Metric Value per Trial",
            xaxis_title="Trial",
            yaxis_title="Metric Value",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        figs.append(line_fig)
        if container is not None:
            container.plotly_chart(line_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 2 — scatter: config dimension vs metric
        # ------------------------------------------------------------------ #
        scatter_fig = go.Figure()
        if history_df is not None and not history_df.empty and "config" in history_df.columns:
            # Extract first numeric key from config dicts
            config_col = history_df["config"].dropna()
            if not config_col.empty:
                first_row_config = config_col.iloc[0]
                numeric_key = None
                if isinstance(first_row_config, dict):
                    for k, v in first_row_config.items():
                        if isinstance(v, (int, float)):
                            numeric_key = k
                            break

                if numeric_key is not None:
                    x_vals = [
                        cfg.get(numeric_key) if isinstance(cfg, dict) else None
                        for cfg in history_df["config"]
                    ]
                    y_vals = list(history_df["metric_value"].astype(float))
                    scatter_fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="markers",
                            name=numeric_key,
                            marker=dict(color="purple", size=8),
                        )
                    )
                    scatter_fig.update_layout(
                        title=f"Config ({numeric_key}) vs Metric",
                        xaxis_title=numeric_key,
                        yaxis_title="Metric Value",
                    )

        scatter_fig.update_layout(
            margin=dict(l=40, r=20, t=40, b=40),
        )
        figs.append(scatter_fig)
        if container is not None:
            container.plotly_chart(scatter_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 3 — bar: operator usage frequency
        # ------------------------------------------------------------------ #
        bar_fig = go.Figure()
        if history_df is not None and not history_df.empty and "status" in history_df.columns:
            status_counts = history_df["status"].value_counts()
            bar_fig.add_trace(
                go.Bar(
                    x=list(status_counts.index.astype(str)),
                    y=list(status_counts.values),
                    name="Operator/Status Usage",
                    marker_color="mediumseagreen",
                )
            )

        bar_fig.update_layout(
            title="Operator Usage Frequency",
            xaxis_title="Status / Operator",
            yaxis_title="Count",
            margin=dict(l=40, r=20, t=40, b=80),
        )
        figs.append(bar_fig)
        if container is not None:
            container.plotly_chart(bar_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 4 — table: best_config
        # ------------------------------------------------------------------ #
        if best_config:
            keys = [str(k) for k in best_config.keys()]
            values = [str(v) for v in best_config.values()]
            best_fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=["<b>Parameter</b>", "<b>Value</b>"],
                            fill_color="paleturquoise",
                            align="left",
                        ),
                        cells=dict(
                            values=[keys, values],
                            fill_color="lavender",
                            align="left",
                        ),
                    )
                ]
            )
            best_fig.update_layout(
                title="Best Config",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            figs.append(best_fig)
            if container is not None:
                container.plotly_chart(best_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 5 — plateau gauge
        # ------------------------------------------------------------------ #
        plateau_score = plateau_data.get("plateau_score", plateau_data.get("score"))
        if plateau_score is not None:
            try:
                score_val = float(plateau_score)
                gauge_fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=score_val,
                        title={"text": "Plateau Score"},
                        gauge={
                            "axis": {"range": [0, 1]},
                            "bar": {"color": "orange"},
                            "steps": [
                                {"range": [0, 0.5], "color": "lightgreen"},
                                {"range": [0.5, 0.8], "color": "yellow"},
                                {"range": [0.8, 1.0], "color": "red"},
                            ],
                        },
                    )
                )
                gauge_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
                figs.append(gauge_fig)
                if container is not None:
                    container.plotly_chart(gauge_fig, use_container_width=True)
            except (TypeError, ValueError):
                pass

        return figs
