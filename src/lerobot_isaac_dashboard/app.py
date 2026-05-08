"""app.py — Live Streamlit dashboard entrypoint.

Run via the console script or directly::

    streamlit run src/lerobot_isaac_dashboard/app.py -- --workspace=PATH
    lerobot-isaac-dashboard --workspace=PATH

All sidebar controls (workspace path, session selector, refresh interval,
watchdog toggle, export button) live here.  Tab rendering is delegated to
the TABS registry from :mod:`lerobot_isaac_dashboard.tabs`.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports — all Streamlit calls are inside main() so the module is importable
# in test environments that mock streamlit.
# ---------------------------------------------------------------------------

from lerobot_isaac_dashboard.loaders import (
    LoaderResult,
    load_autoresearch,
    load_checkpoints,
    load_curriculum,
    load_eval_results,
    load_events,
    load_parquet_dataset,
    load_paths,
    load_synthetic,
    load_training_logs,
)
from lerobot_isaac_dashboard.runtime.refresh import register_autorefresh, register_watchdog
from lerobot_isaac_dashboard.runtime.session_state import (
    default_session_id,
    list_session_ids,
    resolve_workspace_root,
)
from lerobot_isaac_dashboard.tabs import (
    TABS,
    AutoresearchTab,
    CurriculumTab,
    DataCollectionTab,
    EvaluationTab,
    PipelineHealthTab,
    PolicyTrainingTab,
    SyntheticTab,
    TabContext,
    WorldModelTab,
)

# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

#: Mapping from loader slug to callable.
#: Each callable has signature (workspace_root, *, session_id=None) -> LoaderResult.
LOADERS: dict[str, Any] = {
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
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_argv_tail() -> dict[str, str | None]:
    """Parse ``--workspace=...`` and ``--session-id=...`` from sys.argv.

    Streamlit passes CLI args after ``--`` directly to the script, so
    ``sys.argv`` contains them when running via ``streamlit run``.

    Returns a dict with keys ``workspace`` and ``session_id`` (both may be None).
    """
    result: dict[str, str | None] = {"workspace": None, "session_id": None}
    for arg in sys.argv[1:]:
        if arg.startswith("--workspace="):
            result["workspace"] = arg.split("=", 1)[1]
        elif arg == "--workspace" and sys.argv.index(arg) + 1 < len(sys.argv):
            idx = sys.argv.index(arg)
            result["workspace"] = sys.argv[idx + 1]
        elif arg.startswith("--session-id="):
            result["session_id"] = arg.split("=", 1)[1]
    return result


def _mtime_tuple(workspace_root: Path, session_id: str | None) -> tuple[float, ...]:
    """Return a tuple of mtimes for key data directories.

    Used as part of the st.cache_data cache key so the cache invalidates
    automatically when source files change.
    """
    dirs = [
        workspace_root / "datasets",
        workspace_root / "outputs",
        workspace_root / ".agent-state",
    ]
    if session_id:
        dirs.append(workspace_root / ".agent-state" / session_id)

    mtimes: list[float] = []
    for d in dirs:
        try:
            mtimes.append(d.stat().st_mtime)
        except OSError:
            mtimes.append(0.0)
    return tuple(mtimes)


def _run_all_loaders(
    workspace_root: Path,
    session_id: str | None,
) -> dict[str, LoaderResult]:
    """Call every loader and collect results keyed by slug.

    This is the un-cached inner function.  The Streamlit-cached wrapper
    ``_cached_loaders`` below wraps this with ``st.cache_data``.
    """
    results: dict[str, LoaderResult] = {}
    for slug, loader_fn in LOADERS.items():
        try:
            # All loaders accept (workspace_root, *, session_id=None)
            results[slug] = loader_fn(workspace_root, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("loader %r failed: %s", slug, exc)
            # Return an empty result so tabs can still render the empty state
            import pandas as pd

            results[slug] = LoaderResult(
                df=pd.DataFrame(),
                is_empty=True,
                warnings=[f"Loader error: {exc}"],
            )
    return results


# ---------------------------------------------------------------------------
# Streamlit-cached loader wrapper
# ---------------------------------------------------------------------------

def _get_cached_loaders(
    workspace_root: Path,
    session_id: str | None,
    ttl: int,
    mtime_tuple: tuple[float, ...],  # noqa: ARG001 — used as cache-key component
) -> dict[str, LoaderResult]:
    """Invoke all loaders with Streamlit cache invalidation based on mtime.

    The ``mtime_tuple`` parameter is intentionally unused inside this function —
    it exists solely so ``st.cache_data`` treats it as part of the cache key,
    causing cache invalidation whenever source files change.

    In non-Streamlit environments (tests) this falls back to calling
    ``_run_all_loaders`` directly.
    """
    try:
        import streamlit as st

        @st.cache_data(ttl=max(ttl, 5), show_spinner=False)
        def _inner(
            _workspace_root: Path,
            _session_id: str | None,
            _ttl: int,
            _mtime_tuple: tuple[float, ...],
        ) -> dict[str, LoaderResult]:
            return _run_all_loaders(_workspace_root, _session_id)

        return _inner(workspace_root, session_id, ttl, mtime_tuple)

    except ImportError:
        return _run_all_loaders(workspace_root, session_id)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Live Streamlit dashboard entrypoint.

    Called by ``streamlit run app.py`` or the ``lerobot-isaac-dashboard``
    console script.  Should not be called directly in tests.
    """
    import streamlit as st

    st.set_page_config(
        page_title="lerobot-isaac dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ------------------------------------------------------------------
    # 1. Parse CLI args
    # ------------------------------------------------------------------
    argv = _parse_argv_tail()
    cli_workspace = argv.get("workspace")
    cli_session_id = argv.get("session_id")

    # Also check st.query_params for browser-URL-based routing
    try:
        qp = st.query_params
        if "workspace" in qp and not cli_workspace:
            cli_workspace = qp["workspace"]
        if "session_id" in qp and not cli_session_id:
            cli_session_id = qp["session_id"]
    except Exception:  # noqa: BLE001
        pass

    # ------------------------------------------------------------------
    # 2. Resolve workspace root
    # ------------------------------------------------------------------
    workspace_root = resolve_workspace_root(cli_workspace)

    # ------------------------------------------------------------------
    # 3. Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.title("lerobot-isaac")
        st.caption("metrics dashboard")
        st.divider()

        # Workspace display (read-only)
        st.write("**Workspace**")
        st.code(str(workspace_root), language=None)

        # Session selector
        session_ids = list_session_ids(workspace_root)
        if session_ids:
            default_idx = 0
            if cli_session_id and cli_session_id in session_ids:
                default_idx = session_ids.index(cli_session_id)
            elif cli_session_id is None:
                default_sid = default_session_id(workspace_root)
                if default_sid in session_ids:
                    default_idx = session_ids.index(default_sid)
            session_id: str | None = st.selectbox(
                "Session",
                options=session_ids,
                index=default_idx,
                key="session_id_selector",
            )
        else:
            st.info("No sessions found in .agent-state/")
            session_id = cli_session_id  # may still be set from CLI

        st.divider()

        # Refresh interval
        refresh_s: int = st.slider(
            "Refresh interval (s)",
            min_value=0,
            max_value=120,
            value=30,
            step=5,
            help="Set to 0 to disable auto-refresh.",
            key="refresh_interval_s",
        )

        # Watchdog toggle
        watchdog_on: bool = st.checkbox(
            "Watch files",
            value=False,
            help=(
                "Enable file-system watcher on outputs/, .agent-state/, datasets/. "
                "Triggers a rerun whenever files change. Reduces idle CPU vs timer."
            ),
            key="watchdog_enabled",
        )

        st.divider()

        # Export button
        export_clicked: bool = st.button(
            "Export static report",
            help="Generate a static HTML report (requires Phase 5 — report.py).",
            key="export_button",
        )

    # ------------------------------------------------------------------
    # 4. Auto-refresh
    # ------------------------------------------------------------------
    if refresh_s > 0:
        register_autorefresh(refresh_s * 1000)

    # ------------------------------------------------------------------
    # 5. Watchdog
    # ------------------------------------------------------------------
    if watchdog_on:
        register_watchdog(workspace_root)

    # ------------------------------------------------------------------
    # 6. Load all data (cached, mtime-keyed)
    # ------------------------------------------------------------------
    mtime = _mtime_tuple(workspace_root, session_id)
    loader_results = _get_cached_loaders(workspace_root, session_id, refresh_s, mtime)

    # ------------------------------------------------------------------
    # 7. Build TabContext
    # ------------------------------------------------------------------
    ctx = TabContext(
        workspace_root=workspace_root,
        session_id=session_id,
        loader_results=loader_results,
        refresh_ts=datetime.utcnow(),
    )

    # ------------------------------------------------------------------
    # 8. KPI banner (renders before tabs, survives tab switching)
    # ------------------------------------------------------------------
    from lerobot_isaac_dashboard.tabs._kpis import render_kpi_row

    render_kpi_row(ctx)

    # ------------------------------------------------------------------
    # 9. Tab routing
    # ------------------------------------------------------------------
    tab_titles = [tab_cls.title for tab_cls in TABS]
    tab_containers = st.tabs(tab_titles)

    for tab_cls, container in zip(TABS, tab_containers):
        with container:
            try:
                tab_cls().render(ctx, container=container)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Tab '{tab_cls.title}' raised: {exc}")
                logger.exception("Tab %r failed", tab_cls.title)

    # ------------------------------------------------------------------
    # 10. Export button handler
    # ------------------------------------------------------------------
    if export_clicked:
        # TODO(P5): wire export_report once report.py lands
        try:
            from lerobot_isaac_dashboard.report import export_report  # type: ignore[import-not-found]

            with st.spinner("Generating static report…"):
                report_path = export_report(ctx)
            st.success(f"Report exported to: {report_path}")
            with open(report_path, "rb") as fh:
                st.download_button(
                    label="Download report",
                    data=fh,
                    file_name="lerobot-isaac-report.html",
                    mime="text/html",
                    key="download_report",
                )
        except ImportError:
            st.info(
                "Static report export is not yet available. "
                "It will land in Phase 5 (report.py). "
                "Re-run the dashboard after P5 ships."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Export failed: {exc}")
            logger.exception("Export failed")

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.divider()
    st.caption(
        f"Last refresh: {ctx.refresh_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"| Workspace: {workspace_root}"
    )


if __name__ == "__main__":
    main()
