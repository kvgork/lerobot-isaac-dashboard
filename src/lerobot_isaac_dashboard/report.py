"""report.py — Static HTML report exporter for lerobot-isaac-dashboard.

Produces a single self-contained HTML file per run from the same Plotly
figures used by the live Streamlit dashboard.

CLI::

    python -m lerobot_isaac_dashboard.report --workspace=PATH [--session-id=SID] [--cdn] [--with-csv] [--no-snapshot]
    lerobot-isaac-report --workspace=PATH [--session-id=SID] [--cdn] [--with-csv] [--no-snapshot]

API::

    from lerobot_isaac_dashboard.report import export_report
    out = export_report(Path("/path/to/workspace"))
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from lerobot_isaac_dashboard.loaders import (
    LoaderResult,
    load_autoresearch,
    load_checkpoints,
    load_curriculum,
    load_eval_results,
    load_events,
    load_parquet_dataset,
    load_synthetic,
    load_training_logs,
)
from lerobot_isaac_dashboard.tabs import TABS, TabContext
from lerobot_isaac_dashboard.runtime.static_render import render_tab_to_html

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loader registry (mirrors app.py LOADERS — no st.cache_data)
# ---------------------------------------------------------------------------

_LOADERS: dict[str, Any] = {
    "parquet_dataset": load_parquet_dataset,
    "eval_results": load_eval_results,
    "checkpoints": load_checkpoints,
    "training_logs": load_training_logs,
    "autoresearch": load_autoresearch,
    "events": load_events,
    "curriculum": load_curriculum,
    "synthetic": load_synthetic,
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _TabRenderResult:
    """Per-tab rendering artefacts used by the Jinja2 template."""

    slug: str
    title: str
    body: str  # pre-rendered HTML fragment (figures concatenated)
    warnings: list[str]
    n_figures: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_loaders_headless(
    workspace_root: Path,
    session_id: str | None = None,
) -> dict[str, LoaderResult]:
    """Call every loader directly (no Streamlit cache) and return results by slug.

    Never raises — individual loader failures are captured as empty results with
    a warning message, matching the behaviour of the live app.

    Parameters
    ----------
    workspace_root:
        Absolute path to the training workspace root.
    session_id:
        Optional session ID to scope session-aware loaders (events,
        autoresearch).  Pass ``None`` to scan all sessions.

    Returns
    -------
    dict[str, LoaderResult]
        Mapping from loader slug to loader output.
    """
    results: dict[str, LoaderResult] = {}
    for slug, loader_fn in _LOADERS.items():
        try:
            results[slug] = loader_fn(workspace_root, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Loader %r failed during headless run: %s", slug, exc)
            results[slug] = LoaderResult(
                df=pd.DataFrame(),
                is_empty=True,
                warnings=[f"Loader error: {exc}"],
            )
    return results


def export_report(
    workspace_root: Path,
    *,
    session_id: str | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
    inline_plotlyjs: bool = True,
    with_csv: bool = False,
    with_snapshot: bool = True,
) -> Path:
    """Render all 8 tabs to a single self-contained HTML file.

    Parameters
    ----------
    workspace_root:
        Absolute path to the training workspace root.
    session_id:
        Optional session ID string (scopes session-aware loaders).
    output_dir:
        Directory to write ``report.html`` and ``manifest.json`` into.
        Defaults to ``<workspace_root>/outputs/reports/<run_id>/``.
    run_id:
        Identifier string embedded in the manifest and default output path.
        Defaults to ``YYYY-MM-DDTHHMMSS-<session_id or 'no-session'>``.
    inline_plotlyjs:
        When ``True`` (default), embed ``plotly.min.js`` once in the ``<head>``
        of the generated HTML — the file is fully self-contained (~3 MB).
        When ``False``, inject a CDN ``<script>`` tag instead (smaller file,
        requires internet access to view).
    with_csv:
        When ``True``, write ``data/<slug>.csv`` dumps next to the report for
        each loader whose result is non-empty.
    with_snapshot:
        When ``True`` (default), automatically save a snapshot of the loader
        state alongside the report.  Pass ``False`` to skip snapshot creation.

    Returns
    -------
    Path
        Absolute path to the written ``report.html`` file.
    """
    workspace_root = Path(workspace_root).resolve()

    # Derive defaults
    generated_at = datetime.now(tz=timezone.utc)
    if run_id is None:
        ts = generated_at.strftime("%Y-%m-%dT%H%M%S")
        run_id = f"{ts}-{session_id or 'no-session'}"

    if output_dir is None:
        output_dir = workspace_root / "outputs" / "reports" / run_id
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Run all loaders headless (capture state once — shared by HTML and snapshot)
    # ------------------------------------------------------------------
    loader_results = run_loaders_headless(workspace_root, session_id=session_id)

    # ------------------------------------------------------------------
    # 2. Build TabContext
    # ------------------------------------------------------------------
    ctx = TabContext(
        workspace_root=workspace_root,
        session_id=session_id,
        loader_results=loader_results,
        refresh_ts=generated_at.replace(tzinfo=None),
    )

    # ------------------------------------------------------------------
    # 3. Render each tab to an HTML fragment
    # ------------------------------------------------------------------
    tab_results: list[_TabRenderResult] = []
    for tab_cls in TABS:
        tab = tab_cls()
        body, warnings = render_tab_to_html(tab, ctx, include_plotlyjs=False)
        tab_results.append(
            _TabRenderResult(
                slug=tab_cls.slug,
                title=tab_cls.title,
                body=body,
                warnings=warnings,
                n_figures=body.count('class="plotly-graph-div"'),
            )
        )

    # ------------------------------------------------------------------
    # 4. Build Plotly JS tag
    # ------------------------------------------------------------------
    plotlyjs_tag = _build_plotlyjs_tag(inline_plotlyjs)

    # ------------------------------------------------------------------
    # 5. Render the master Jinja2 template
    # ------------------------------------------------------------------
    html_content = _render_html_template(
        run_id=run_id,
        workspace_root=str(workspace_root),
        session_id=session_id,
        generated_at=generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        tabs=tab_results,
        plotlyjs_tag=plotlyjs_tag,
    )

    # ------------------------------------------------------------------
    # 6. Write report.html
    # ------------------------------------------------------------------
    report_path = output_dir / "report.html"
    report_path.write_text(html_content, encoding="utf-8")
    logger.info("Report written: %s", report_path)

    # ------------------------------------------------------------------
    # 7. Auto-save snapshot (default ON; --no-snapshot opts out)
    # ------------------------------------------------------------------
    snapshot_path: str | None = None
    if with_snapshot:
        try:
            from lerobot_isaac_dashboard.snapshots import save_snapshot

            snap_dir = save_snapshot(
                workspace_root,
                session_id=session_id,
                label=run_id,
            )
            snapshot_path = str(snap_dir)
            logger.info("Auto-snapshot saved: %s", snap_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-snapshot failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # 8. Write manifest.json
    # ------------------------------------------------------------------
    manifest = {
        "run_id": run_id,
        "workspace_root": str(workspace_root),
        "session_id": session_id,
        "generated_at": generated_at.isoformat(),
        "snapshot_path": snapshot_path,
        "tabs": [
            {
                "slug": t.slug,
                "title": t.title,
                "n_figures": t.n_figures,
                "n_warnings": len(t.warnings),
            }
            for t in tab_results
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest written: %s", manifest_path)

    # ------------------------------------------------------------------
    # 9. Optional CSV dumps
    # ------------------------------------------------------------------
    if with_csv:
        _write_csv_dumps(output_dir, loader_results)

    return report_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_plotlyjs_tag(inline: bool) -> str:
    """Return a ``<script>`` tag with either inline JS or a CDN URL."""
    if inline:
        try:
            import plotly.offline as po

            js = po.get_plotlyjs()
            return f"<script type='text/javascript'>{js}</script>"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not embed plotly.min.js inline (%s); falling back to CDN", exc
            )
            inline = False  # fall through to CDN

    # CDN path (also fallback if inline failed)
    return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'


def _render_html_template(
    *,
    run_id: str,
    workspace_root: str,
    session_id: str | None,
    generated_at: str,
    tabs: list[_TabRenderResult],
    plotlyjs_tag: str,
) -> str:
    """Render the master Jinja2 template and return the HTML string."""
    from jinja2 import Environment, PackageLoader, select_autoescape

    env = Environment(
        loader=PackageLoader("lerobot_isaac_dashboard", "templates"),
        autoescape=select_autoescape(["html"]),
        keep_trailing_newline=True,
    )

    template = env.get_template("report.html.j2")
    return template.render(
        run_id=run_id,
        workspace_root=workspace_root,
        session_id=session_id,
        generated_at=generated_at,
        tabs=tabs,
        plotlyjs_tag=plotlyjs_tag,
    )


def _write_csv_dumps(output_dir: Path, loader_results: dict[str, LoaderResult]) -> None:
    """Write per-loader CSV files under ``<output_dir>/data/``."""
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for slug, result in loader_results.items():
        df = result.df
        if isinstance(df, dict):
            # Hierarchical loader (e.g. autoresearch, curriculum) — dump each sub-df
            for sub_key, sub_df in df.items():
                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                    csv_path = data_dir / f"{slug}_{sub_key}.csv"
                    try:
                        sub_df.to_csv(csv_path, index=False)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "CSV dump failed for %s.%s: %s", slug, sub_key, exc
                        )
        elif isinstance(df, pd.DataFrame) and not df.empty:
            csv_path = data_dir / f"{slug}.csv"
            try:
                df.to_csv(csv_path, index=False)
            except Exception as exc:  # noqa: BLE001
                logger.debug("CSV dump failed for %s: %s", slug, exc)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def cli_main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m lerobot_isaac_dashboard.report`` and ``lerobot-isaac-report``.

    Usage::

        lerobot-isaac-report --workspace=PATH [--session-id=SID] [--cdn] [--with-csv] [--no-snapshot]
        python -m lerobot_isaac_dashboard.report --workspace=PATH

    Returns 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        prog="lerobot-isaac-report",
        description="Generate a static HTML metrics report from a lerobot-isaac workspace.",
    )
    parser.add_argument(
        "--workspace",
        metavar="PATH",
        default=".",
        help="Path to the workspace root (default: current directory).",
    )
    parser.add_argument(
        "--session-id",
        metavar="SID",
        default=None,
        help="Scope session-aware loaders to this session ID.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Override the output directory (default: <workspace>/outputs/reports/<run_id>/).",
    )
    parser.add_argument(
        "--run-id",
        metavar="ID",
        default=None,
        help="Override the run_id slug embedded in the manifest and output path.",
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        default=False,
        help="Use CDN plotly.js instead of inlining it (smaller file, needs internet).",
    )
    parser.add_argument(
        "--with-csv",
        action="store_true",
        default=False,
        help="Also write per-loader CSV dumps under <output_dir>/data/.",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        default=False,
        help="Disable the automatic snapshot saved alongside the report.",
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
        report_path = export_report(
            workspace_root,
            session_id=args.session_id,
            output_dir=output_dir,
            run_id=args.run_id,
            inline_plotlyjs=not args.cdn,
            with_csv=args.with_csv,
            with_snapshot=not args.no_snapshot,
        )
        print(f"Report written: {report_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Report generation failed: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())
