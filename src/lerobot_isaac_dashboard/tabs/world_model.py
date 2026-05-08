"""world_model.py — Tab #4: World Model Training.

Consumes:
    ``training_logs``  — TRAINING_LOG_SCHEMA filtered to dreamerv3 / le_world_model arches
    ``checkpoints``    — CHECKPOINT_SCHEMA filtered to dreamerv3 / le_world_model

Figures:
    1. Line chart: recon_loss + pred_loss vs step per run_id
    2. Histogram: recon residuals (if available)
    3. Checkpoint inventory table
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

_WM_ARCHES = {"dreamerv3", "le_world_model", "leworldmodel", "dreamer"}
_WM_LOSS_METRICS = {
    "recon_loss",
    "pred_loss",
    "reconstruction_loss",
    "prediction_loss",
    "kl_loss",
}


class WorldModelTab(Tab):
    """Tab #4 — loss curves and checkpoints for world-model training."""

    title = "World Model Training"
    slug = "world_model"
    primary_loader_slug = "training_logs"

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

        logs_df = primary.df
        figs: list[Any] = []

        # Filter to world-model arches
        wm_mask = logs_df["arch"].str.lower().isin(_WM_ARCHES)
        if wm_mask.any():
            logs_df = logs_df[wm_mask]

        # ------------------------------------------------------------------ #
        # Figure 1 — recon_loss + pred_loss vs step
        # ------------------------------------------------------------------ #
        loss_mask = logs_df["metric_name"].str.lower().isin(_WM_LOSS_METRICS)
        loss_df = (
            logs_df[loss_mask]
            if loss_mask.any()
            else logs_df[
                logs_df["metric_name"].str.lower().str.contains("loss", na=False)
            ]
        )

        line_fig = go.Figure()
        if not loss_df.empty:
            for (arch, run_id, metric), grp in loss_df.groupby(
                ["arch", "run_id", "metric_name"]
            ):
                grp_sorted = grp.sort_values("step")
                line_fig.add_trace(
                    go.Scatter(
                        x=list(grp_sorted["step"].astype(float)),
                        y=list(grp_sorted["value"].astype(float)),
                        mode="lines",
                        name=f"{arch}/{run_id}/{metric}",
                    )
                )

        line_fig.update_layout(
            title="World Model Losses vs Step",
            xaxis_title="Step",
            yaxis_title="Loss",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        figs.append(line_fig)
        if container is not None:
            container.plotly_chart(line_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 2 — histogram: recon residuals (proxy: recon_loss values)
        # ------------------------------------------------------------------ #
        recon_mask = loss_df["metric_name"].str.lower().str.contains("recon", na=False)
        recon_vals = loss_df[recon_mask]["value"].dropna()
        if not recon_vals.empty:
            hist_fig = go.Figure(
                data=[
                    go.Histogram(
                        x=list(recon_vals.astype(float)),
                        nbinsx=30,
                        name="Recon Loss",
                        marker_color="coral",
                    )
                ]
            )
            hist_fig.update_layout(
                title="Reconstruction Loss Distribution",
                xaxis_title="recon_loss value",
                yaxis_title="Count",
                margin=dict(l=40, r=20, t=40, b=40),
            )
            figs.append(hist_fig)
            if container is not None:
                container.plotly_chart(hist_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 3 — checkpoint table
        # ------------------------------------------------------------------ #
        ckpt_result = ctx.loader_results.get("checkpoints")
        if ckpt_result is not None and not ckpt_result.is_empty:
            ckpt_df = ckpt_result.df
            wm_ckpt_mask = ckpt_df["arch"].str.lower().isin(_WM_ARCHES)
            if wm_ckpt_mask.any():
                ckpt_df = ckpt_df[wm_ckpt_mask]

            ckpt_fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=[
                                "<b>Arch</b>",
                                "<b>Run ID</b>",
                                "<b>Step</b>",
                                "<b>Size MB</b>",
                                "<b>Val Loss</b>",
                            ],
                            fill_color="paleturquoise",
                            align="left",
                        ),
                        cells=dict(
                            values=[
                                list(ckpt_df["arch"].astype(str)),
                                list(ckpt_df["run_id"].astype(str)),
                                list(ckpt_df["step"].astype(str)),
                                list(
                                    ckpt_df["size_mb"]
                                    .fillna(float("nan"))
                                    .round(2)
                                    .astype(str)
                                ),
                                list(
                                    ckpt_df["val_loss"]
                                    .fillna(float("nan"))
                                    .round(4)
                                    .astype(str)
                                ),
                            ],
                            fill_color="lavender",
                            align="left",
                        ),
                    )
                ]
            )
            ckpt_fig.update_layout(
                title="World Model Checkpoints",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            figs.append(ckpt_fig)
            if container is not None:
                container.plotly_chart(ckpt_fig, use_container_width=True)

        return figs
