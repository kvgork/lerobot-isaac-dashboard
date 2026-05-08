"""policy_training.py — Tab #3: Policy Training.

Consumes:
    ``training_logs``  — TRAINING_LOG_SCHEMA (arch, run_id, step, metric_name, value, ts)
    ``checkpoints``    — CHECKPOINT_SCHEMA (arch, run_id, step, path, size_mb, mtime, val_loss)

Figures:
    1. Line chart: loss vs step, one trace per (arch, run_id)
    2. Checkpoint inventory table
    3. KPI tiles: latest val loss, # runs, GPU note
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

# Arches considered "policy" (not world-model)
_POLICY_ARCHES = {"smolvla", "act", "diffusion", "policy"}


class PolicyTrainingTab(Tab):
    """Tab #3 — loss curves and checkpoint inventory for policy training."""

    title = "Policy Training"
    slug = "policy_training"
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

        # Filter to policy arches if any are labelled
        policy_mask = logs_df["arch"].str.lower().isin(_POLICY_ARCHES)
        if policy_mask.any():
            logs_df = logs_df[policy_mask]

        # ------------------------------------------------------------------ #
        # Figure 1 — loss vs step, one trace per (arch, run_id)
        # ------------------------------------------------------------------ #
        loss_df = logs_df[
            logs_df["metric_name"].str.lower().str.contains("loss", na=False)
        ]
        loss_fig = go.Figure()

        if not loss_df.empty:
            for (arch, run_id), grp in loss_df.groupby(["arch", "run_id"]):
                grp_sorted = grp.sort_values("step")
                loss_fig.add_trace(
                    go.Scatter(
                        x=list(grp_sorted["step"].astype(float)),
                        y=list(grp_sorted["value"].astype(float)),
                        mode="lines",
                        name=f"{arch}/{run_id}",
                    )
                )

        loss_fig.update_layout(
            title="Training Loss vs Step",
            xaxis_title="Step",
            yaxis_title="Loss",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        figs.append(loss_fig)
        if container is not None:
            container.plotly_chart(loss_fig, use_container_width=True)

        # ------------------------------------------------------------------ #
        # Figure 2 — checkpoint table
        # ------------------------------------------------------------------ #
        ckpt_result = ctx.loader_results.get("checkpoints")
        if ckpt_result is not None and not ckpt_result.is_empty:
            ckpt_df = ckpt_result.df
            ckpt_mask = ckpt_df["arch"].str.lower().isin(_POLICY_ARCHES)
            if ckpt_mask.any():
                ckpt_df = ckpt_df[ckpt_mask]

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
                title="Checkpoint Inventory",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            figs.append(ckpt_fig)
            if container is not None:
                container.plotly_chart(ckpt_fig, use_container_width=True)

            # KPI: latest val loss from checkpoints
            latest_val_loss = (
                ckpt_result.df.dropna(subset=["val_loss"])["val_loss"].iloc[-1]
                if ckpt_result.df.dropna(subset=["val_loss"]).shape[0] > 0
                else None
            )
        else:
            latest_val_loss = None

        # ------------------------------------------------------------------ #
        # KPI tiles
        # ------------------------------------------------------------------ #
        n_runs = logs_df["run_id"].nunique() if not logs_df.empty else 0
        val_loss_display = (
            f"{latest_val_loss:.4f}" if latest_val_loss is not None else "—"
        )

        kpi_items: list[tuple[str, Any, str | None]] = [
            ("Runs", n_runs, None),
            ("Latest Val Loss", val_loss_display, None),
            ("GPU Util", "Not tracked locally — see runbook", None),
        ]
        kpi_fig = render_kpi_row(container, kpi_items)
        if kpi_fig is not None:
            figs.append(kpi_fig)

        return figs
