"""compare.py — Multi-run comparison for lerobot-isaac-dashboard.

Provides 2-way side-by-side and N-way overlay comparison of dashboard
snapshots.  Also exports a static HTML compare report.

API::

    from lerobot_isaac_dashboard.compare import (
        build_compare_context,
        render_compare_2way,
        render_compare_nway,
        export_compare_report,
    )

CLI::

    python -m lerobot_isaac_dashboard.compare --workspace=. --snapshots A B [C ...]
    lerobot-isaac-compare --workspace=. --snapshots A B --mode 2way
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from lerobot_isaac_dashboard.loaders._base import LoaderResult
from lerobot_isaac_dashboard.snapshots import SnapshotMeta, load_snapshot

logger = logging.getLogger(__name__)

# Snapshot tuple type alias for clarity
SnapEntry = tuple[SnapshotMeta, dict[str, LoaderResult]]

# ---------------------------------------------------------------------------
# Streamlit availability check
# Streamlit raises RuntimeError when re-initialized in test environments, so
# we catch all exceptions — not just ImportError — when probing availability.
# ---------------------------------------------------------------------------


def _st_available() -> bool:
    """Return True if Streamlit can be imported without errors."""
    try:
        import streamlit  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# CompareContext
# ---------------------------------------------------------------------------


@dataclass
class CompareContext:
    """Wraps multiple snapshots for tab consumption in compare mode.

    Attributes
    ----------
    snapshots:
        List of (meta, loader_results) tuples in display order.
    mode:
        "2way" for side-by-side comparison; "nway" for overlay.
    """

    snapshots: list[SnapEntry]
    mode: Literal["2way", "nway"]

    @property
    def labels(self) -> list[str]:
        """Display labels for each snapshot (label or snapshot_id)."""
        return [m.label or m.snapshot_id for (m, _) in self.snapshots]


def build_compare_context(
    snapshots: list[SnapEntry],
    *,
    mode: Literal["2way", "nway"] = "2way",
) -> CompareContext:
    """Wrap multiple snapshots into a CompareContext that tabs can consume.

    Parameters
    ----------
    snapshots:
        List of (meta, loader_results) tuples.
    mode:
        "2way" (default) or "nway".

    Returns
    -------
    CompareContext
    """
    return CompareContext(snapshots=snapshots, mode=mode)


# ---------------------------------------------------------------------------
# Delta KPI helpers
# ---------------------------------------------------------------------------


def _compute_delta_kpis(
    a_results: dict[str, LoaderResult],
    b_results: dict[str, LoaderResult],
) -> list[dict[str, Any]]:
    """Compute a list of delta KPI dicts for display in the compare strip.

    Each dict has: {name, value_a, value_b, delta, unit}.
    Returns an empty list when data is unavailable.
    """
    kpis: list[dict[str, Any]] = []

    # pc_success from eval_results
    for slug in ("eval_results",):
        ra = a_results.get(slug)
        rb = b_results.get(slug)
        if (
            ra
            and not ra.is_empty
            and isinstance(ra.df, pd.DataFrame)
            and "pc_success" in ra.df.columns
        ):
            va = float(ra.df["pc_success"].dropna().mean()) if not ra.df.empty else None
        else:
            va = None

        if (
            rb
            and not rb.is_empty
            and isinstance(rb.df, pd.DataFrame)
            and "pc_success" in rb.df.columns
        ):
            vb = float(rb.df["pc_success"].dropna().mean()) if not rb.df.empty else None
        else:
            vb = None

        if va is not None or vb is not None:
            delta = (vb - va) if (va is not None and vb is not None) else None
            kpis.append(
                {
                    "name": "pc_success (mean)",
                    "value_a": va,
                    "value_b": vb,
                    "delta": delta,
                    "unit": "%",
                }
            )

    # training loss (latest) from training_logs
    for slug in ("training_logs",):
        ra = a_results.get(slug)
        rb = b_results.get(slug)

        def _last_loss(r: LoaderResult | None) -> float | None:
            if r is None or r.is_empty or not isinstance(r.df, pd.DataFrame):
                return None
            df = r.df
            if df.empty or "metric_name" not in df.columns or "value" not in df.columns:
                return None
            loss = df[df["metric_name"].str.lower().str.contains("loss", na=False)]
            if loss.empty:
                return None
            return float(loss["value"].dropna().iloc[-1])

        va = _last_loss(ra)
        vb = _last_loss(rb)
        if va is not None or vb is not None:
            delta = (vb - va) if (va is not None and vb is not None) else None
            kpis.append(
                {
                    "name": "train_loss (latest)",
                    "value_a": va,
                    "value_b": vb,
                    "delta": delta,
                    "unit": "",
                }
            )

    return kpis


def _delta_kpi_html(kpis: list[dict[str, Any]], label_a: str, label_b: str) -> str:
    """Render delta KPIs as an HTML strip (no Streamlit required)."""
    if not kpis:
        return ""
    parts = [
        '<div class="compare-delta-strip" style="display:flex;gap:16px;padding:8px 0;'
        'border-bottom:1px solid #eee;margin-bottom:12px;">'
    ]
    for kpi in kpis:
        va_str = (
            f"{kpi['value_a']:.4f}{kpi['unit']}"
            if kpi["value_a"] is not None
            else "N/A"
        )
        vb_str = (
            f"{kpi['value_b']:.4f}{kpi['unit']}"
            if kpi["value_b"] is not None
            else "N/A"
        )
        if kpi["delta"] is not None:
            sign = "+" if kpi["delta"] >= 0 else ""
            color = "#2e7d32" if kpi["delta"] >= 0 else "#c62828"
            delta_str = f'<span style="color:{color};font-weight:bold">{sign}{kpi["delta"]:.4f}{kpi["unit"]}</span>'
        else:
            delta_str = "<span>N/A</span>"
        parts.append(
            f'<div style="background:#f5f5f5;padding:8px 12px;border-radius:6px;">'
            f"<strong>{kpi['name']}</strong><br/>"
            f"<small>{label_a}</small>: {va_str}<br/>"
            f"<small>{label_b}</small>: {vb_str}<br/>"
            f"Delta: {delta_str}"
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _make_tab_context(
    meta: SnapshotMeta, loader_results: dict[str, LoaderResult]
) -> Any:
    """Build a TabContext from a snapshot entry."""

    from lerobot_isaac_dashboard.tabs._base import TabContext

    return TabContext(
        workspace_root=meta.workspace_root,
        session_id=meta.session_id,
        loader_results=loader_results,
        refresh_ts=meta.ts.replace(tzinfo=None),
    )


def _render_tab_figs(tab_cls: Any, ctx: Any) -> list[Any]:
    """Render a tab to a list of figures (static path)."""
    try:
        return tab_cls().render(ctx, container=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tab %r render failed: %s", tab_cls.slug, exc)
        return []


# ---------------------------------------------------------------------------
# Public compare functions
# ---------------------------------------------------------------------------


def render_compare_2way(
    a: SnapEntry,
    b: SnapEntry,
    *,
    container: Any = None,
) -> dict[str, list[Any]]:
    """Render a 2-way side-by-side comparison of two snapshots.

    For each tab in TABS, renders the tab in both snapshots in separate columns
    (when container is a Streamlit container) plus a delta KPI strip above.

    Parameters
    ----------
    a:
        First snapshot (meta, loader_results).
    b:
        Second snapshot (meta, loader_results).
    container:
        Streamlit container to render into.  When None, figures are returned
        only (used by the static compare exporter).

    Returns
    -------
    dict mapping tab slug -> list of figures from both sides combined.
    """
    from lerobot_isaac_dashboard.tabs import TABS

    meta_a, results_a = a
    meta_b, results_b = b
    label_a = meta_a.label or meta_a.snapshot_id
    label_b = meta_b.label or meta_b.snapshot_id

    ctx_a = _make_tab_context(meta_a, results_a)
    ctx_b = _make_tab_context(meta_b, results_b)

    all_figs: dict[str, list[Any]] = {}
    delta_kpis = _compute_delta_kpis(results_a, results_b)

    has_st = container is not None and _st_available()

    # Top-level delta KPI strip (live mode only)
    if has_st:
        try:
            import streamlit as st

            delta_kpis_display = delta_kpis
            if delta_kpis_display:
                cols = container.columns(len(delta_kpis_display))
                for col, kpi in zip(cols, delta_kpis_display):
                    va = kpi["value_a"]
                    vb = kpi["value_b"]
                    delta = kpi["delta"]
                    va_str = f"{va:.4f}" if va is not None else "N/A"
                    vb_str = f"{vb:.4f}" if vb is not None else "N/A"
                    delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
                    col.metric(kpi["name"], f"A:{va_str} B:{vb_str}", delta=delta_str)
        except Exception:  # noqa: BLE001
            pass

    tab_titles = [tc.title for tc in TABS]

    if has_st:
        try:
            import streamlit as st

            tab_containers = container.tabs(tab_titles)
            for tab_cls, tab_container in zip(TABS, tab_containers):
                with tab_container:
                    col_a, col_b = st.columns(2)
                    col_a.markdown(f"**{label_a}**")
                    col_b.markdown(f"**{label_b}**")
                    figs_a = _render_tab_figs(tab_cls, ctx_a)
                    figs_b = _render_tab_figs(tab_cls, ctx_b)
                    for fig in figs_a:
                        try:
                            col_a.plotly_chart(fig, use_container_width=True)
                        except Exception:  # noqa: BLE001
                            pass
                    for fig in figs_b:
                        try:
                            col_b.plotly_chart(fig, use_container_width=True)
                        except Exception:  # noqa: BLE001
                            pass
                    all_figs[tab_cls.slug] = figs_a + figs_b
        except Exception:  # noqa: BLE001
            has_st = False  # fall through to headless

    if not has_st:
        for tab_cls in TABS:
            figs_a = _render_tab_figs(tab_cls, ctx_a)
            figs_b = _render_tab_figs(tab_cls, ctx_b)
            all_figs[tab_cls.slug] = figs_a + figs_b

    return all_figs


def render_compare_nway(
    snapshots: list[SnapEntry],
    *,
    container: Any = None,
) -> dict[str, list[Any]]:
    """Render an N-way overlay comparison of multiple snapshots.

    For time-series tabs, overlays traces from each snapshot with the
    snapshot label as the legend entry.  For table-heavy tabs, stacks
    rows with a 'snapshot' column prepended.

    Parameters
    ----------
    snapshots:
        List of (meta, loader_results) tuples (2 or more).
    container:
        Streamlit container.  When None, only figures are returned.

    Returns
    -------
    dict mapping tab slug -> list of overlay figures.
    """
    from lerobot_isaac_dashboard.tabs import TABS

    if not snapshots:
        return {tab_cls.slug: [] for tab_cls in TABS}

    # Build one ctx per snapshot
    contexts = [
        (_make_tab_context(m, r), m.label or m.snapshot_id) for (m, r) in snapshots
    ]

    all_figs: dict[str, list[Any]] = {}
    has_st = container is not None and _st_available()

    tab_titles = [tc.title for tc in TABS]

    def _render_overlay_tab(tab_cls: Any) -> list[Any]:
        """Overlay figures from all snapshots for one tab."""
        try:
            import plotly.graph_objects as go

            _has_plotly = True
        except ImportError:
            _has_plotly = False

        # Collect figures from all snapshots for this tab
        per_snapshot_figs: list[tuple[str, list[Any]]] = []
        for ctx, snap_label in contexts:
            figs = _render_tab_figs(tab_cls, ctx)
            per_snapshot_figs.append((snap_label, figs))

        if not _has_plotly:
            return []

        # Determine max figures from the first non-empty snapshot
        max_figs = max((len(figs) for _, figs in per_snapshot_figs), default=0)
        if max_figs == 0:
            return []

        overlay_figs: list[Any] = []
        for fig_idx in range(max_figs):
            overlay = go.Figure()
            has_any = False
            for snap_label, figs in per_snapshot_figs:
                if fig_idx >= len(figs):
                    continue
                src_fig = figs[fig_idx]
                for trace in src_fig.data:
                    new_trace = trace.__class__(
                        **{
                            k: v
                            for k, v in trace.to_plotly_json().items()
                            if k not in ("type",)
                        },
                        name=f"{snap_label} – {trace.name or ''}".strip(" –"),
                        showlegend=True,
                    )
                    overlay.add_trace(new_trace)
                    has_any = True
            if has_any:
                # Copy layout from first available figure
                for _, figs in per_snapshot_figs:
                    if fig_idx < len(figs):
                        try:
                            overlay.update_layout(figs[fig_idx].layout)
                        except Exception:  # noqa: BLE001
                            pass
                        break
                overlay.update_layout(showlegend=True)
                overlay_figs.append(overlay)

        return overlay_figs

    if has_st:
        try:
            import streamlit as st

            tab_containers = container.tabs(tab_titles)
            for tab_cls, tab_container in zip(TABS, tab_containers):
                figs = _render_overlay_tab(tab_cls)
                all_figs[tab_cls.slug] = figs
                with tab_container:
                    for fig in figs:
                        try:
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:  # noqa: BLE001
                            pass
        except Exception:  # noqa: BLE001
            has_st = False  # fall through to headless

    if not has_st:
        for tab_cls in TABS:
            all_figs[tab_cls.slug] = _render_overlay_tab(tab_cls)

    return all_figs


# ---------------------------------------------------------------------------
# Static HTML compare report exporter
# ---------------------------------------------------------------------------


def export_compare_report(
    workspace_root: Path,
    snapshot_ids: list[str],
    *,
    mode: Literal["2way", "nway"] = "2way",
    output_dir: Path | None = None,
    inline_plotlyjs: bool = True,
) -> Path:
    """Render a static HTML compare report and write it to disk.

    Parameters
    ----------
    workspace_root:
        Workspace root (used to resolve snapshots and the output dir).
    snapshot_ids:
        List of snapshot IDs (or absolute paths) to compare.
        For "2way" mode exactly 2 are expected; for "nway" 2+.
    mode:
        "2way" for side-by-side or "nway" for overlay.
    output_dir:
        Directory to write report.html into.  Defaults to
        ``<workspace_root>/outputs/reports/compare-<joined>/``.
    inline_plotlyjs:
        When True embed plotly.min.js in the HTML (self-contained ~3 MB).
        When False use a CDN script tag.

    Returns
    -------
    Path
        Absolute path to the written ``report.html``.
    """
    from lerobot_isaac_dashboard.report import _build_plotlyjs_tag

    workspace_root = Path(workspace_root).resolve()

    # Resolve snapshots
    loaded: list[SnapEntry] = []
    for sid in snapshot_ids:
        try:
            meta, results = load_snapshot(sid, workspace_root=workspace_root)
            loaded.append((meta, results))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load snapshot %r: %s", sid, exc)

    if len(loaded) < 2:
        raise ValueError(
            f"Need at least 2 valid snapshots for comparison, got {len(loaded)}."
        )

    if mode == "2way" and len(loaded) > 2:
        logger.info("2-way mode: using first two snapshots only")
        loaded = loaded[:2]

    # Determine output dir
    if output_dir is None:
        joined = "-vs-".join((m.label or m.snapshot_id)[:20] for (m, _) in loaded)
        output_dir = workspace_root / "outputs" / "reports" / f"compare-{joined}"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render figures (headless — container=None)
    if mode == "2way":
        all_figs = render_compare_2way(loaded[0], loaded[1], container=None)
    else:
        all_figs = render_compare_nway(loaded, container=None)

    # Build HTML
    from lerobot_isaac_dashboard.tabs import TABS

    html_sections: list[str] = []
    labels = [(m.label or m.snapshot_id) for (m, _) in loaded]

    for tab_cls in TABS:
        slug = tab_cls.slug
        figs = all_figs.get(slug, [])

        if mode == "2way" and len(loaded) == 2:
            # Split figs back into two columns (each side contributed equally)
            mid = len(figs) // 2
            figs_a = figs[:mid]
            figs_b = figs[mid:]
            label_a, label_b = labels[0], labels[1]

            # Delta KPI strip
            delta_kpis = _compute_delta_kpis(loaded[0][1], loaded[1][1])
            delta_html = _delta_kpi_html(delta_kpis, label_a, label_b)

            col_a_html = _figs_to_html(figs_a)
            col_b_html = _figs_to_html(figs_b)
            section_body = (
                f"{delta_html}"
                f'<div style="display:flex;gap:16px;">'
                f'<div class="compare-col" style="flex:1;"><h4>{label_a}</h4>{col_a_html}</div>'
                f'<div class="compare-col" style="flex:1;"><h4>{label_b}</h4>{col_b_html}</div>'
                f"</div>"
            )
        else:
            # N-way: all figs in one column (already overlaid)
            section_body = _figs_to_html(figs)
            label_legend = " vs ".join(labels)
            section_body = f"<p><em>Overlaying: {label_legend}</em></p>{section_body}"

        html_sections.append(
            f'<section id="{slug}"><h2>{tab_cls.title}</h2>{section_body}</section>'
        )

    plotlyjs_tag = _build_plotlyjs_tag(inline_plotlyjs)
    mode_label = (
        "Side-by-Side (2-way)"
        if mode == "2way"
        else f"N-way Overlay ({len(loaded)} runs)"
    )
    title = f"lerobot-isaac Compare Report — {mode_label}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
{plotlyjs_tag}
<style>
body{{font-family:system-ui,sans-serif;margin:0;padding:16px;}}
h1{{font-size:1.5em;border-bottom:2px solid #333;padding-bottom:8px;}}
section{{margin-bottom:32px;}}
h2{{font-size:1.2em;color:#333;}}
.compare-col{{min-width:0;}}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Snapshots: {", ".join(labels)}</p>
{"".join(html_sections)}
</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info("Compare report written: %s", report_path)
    return report_path


def _figs_to_html(figs: list[Any]) -> str:
    """Convert a list of Plotly figures to concatenated HTML fragments."""
    if not figs:
        return "<p><em>No data for this snapshot.</em></p>"
    try:
        import plotly.io as pio
    except ImportError:
        return "<p><em>plotly not installed.</em></p>"

    parts: list[str] = []
    for i, fig in enumerate(figs):
        try:
            part = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True},
            )
            parts.append(part)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fig %d serialization failed: %s", i, exc)
    return "\n".join(parts) if parts else "<p><em>No figures.</em></p>"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def cli_main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m lerobot_isaac_dashboard.compare`` and ``lerobot-isaac-compare``.

    Usage::

        lerobot-isaac-compare --workspace=. --snapshots A B [C ...] [--mode 2way|nway]

    Returns 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        prog="lerobot-isaac-compare",
        description="Generate a static HTML comparison report from lerobot-isaac snapshots.",
    )
    parser.add_argument(
        "--workspace",
        metavar="PATH",
        default=".",
        help="Path to the workspace root (default: current directory).",
    )
    parser.add_argument(
        "--snapshots",
        nargs="+",
        metavar="SNAPSHOT_ID",
        required=True,
        help="Two or more snapshot IDs (or absolute paths) to compare.",
    )
    parser.add_argument(
        "--mode",
        choices=["2way", "nway"],
        default="2way",
        help="Comparison mode: '2way' (default) or 'nway'.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Override the output directory.",
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        default=False,
        help="Use CDN plotly.js instead of inlining it.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    workspace_root = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    try:
        report_path = export_compare_report(
            workspace_root,
            args.snapshots,
            mode=args.mode,
            output_dir=output_dir,
            inline_plotlyjs=not args.cdn,
        )
        print(f"Compare report written: {report_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Compare report failed: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())
