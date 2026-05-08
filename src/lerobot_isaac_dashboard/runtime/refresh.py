"""refresh.py — Auto-refresh and watchdog wiring for the live dashboard.

Two mechanisms are supported:

1. **Timer-based** (always available): ``st.autorefresh(interval_ms, key=key)``
   from the ``streamlit-autorefresh`` extra, or a no-op with a logged warning
   when that package is not installed.

2. **File-watcher** (opt-in): a ``watchdog`` observer on workspace directories.
   When files change the observer sets ``st.session_state["dirty"] = True``
   and calls ``st.rerun()`` so the next render picks up fresh data.
   If watchdog is missing or raises, this function degrades gracefully.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timer-based autorefresh
# ---------------------------------------------------------------------------

def register_autorefresh(interval_ms: int, key: str = "refresh") -> None:
    """Register a Streamlit autorefresh timer.

    Uses ``streamlit_autorefresh.st_autorefresh`` when available (the preferred
    Streamlit-native approach).  Falls back to a ``st.empty()`` spinner with a
    Python ``time.sleep`` loop if that package is not installed.

    In non-Streamlit contexts (e.g. pytest) the function silently no-ops so
    that tests can import ``app.py`` without raising.

    Parameters
    ----------
    interval_ms:
        Refresh interval in milliseconds.  Values <= 0 are ignored.
    key:
        Streamlit widget key; prevents duplicate widget errors across reruns.
    """
    if interval_ms <= 0:
        return

    # Try streamlit_autorefresh first (preferred; smaller bundle)
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore[import-not-found]

        st_autorefresh(interval=interval_ms, key=key)
        logger.debug("register_autorefresh: using st_autorefresh (interval=%d ms)", interval_ms)
        return
    except ImportError:
        pass

    # Fall back to plain streamlit experimental_rerun via a session_state counter
    # This works without streamlit-autorefresh but requires the caller to also
    # wire the trigger; we just set a marker in session_state.
    try:
        import streamlit as st  # type: ignore[import-not-found]

        # Use st.empty to schedule a JS-side meta-refresh via html component
        # when experimental_rerun is the only available hook.
        # This is a best-effort fallback: the page auto-refreshes via HTTP
        # meta-refresh, which reloads the entire page (no partial update).
        interval_s = max(1, interval_ms // 1000)
        st.markdown(
            f'<meta http-equiv="refresh" content="{interval_s}">',
            unsafe_allow_html=True,
        )
        logger.debug(
            "register_autorefresh: falling back to meta-refresh (interval=%d s)",
            interval_s,
        )
    except ImportError:
        # Running outside Streamlit (pytest). No-op is correct.
        logger.debug("register_autorefresh: streamlit not available — skipping")
    except Exception as exc:  # noqa: BLE001
        logger.warning("register_autorefresh: unexpected error: %s", exc)


# ---------------------------------------------------------------------------
# Watchdog-based file observer
# ---------------------------------------------------------------------------

class _ChangeHandler:
    """Minimal watchdog event handler that marks session_state dirty."""

    def __init__(self) -> None:
        self._st: Any = None
        self._lock = threading.Lock()

    def _get_st(self) -> Any:
        if self._st is None:
            try:
                import streamlit as st  # type: ignore[import-not-found]
                self._st = st
            except ImportError:
                pass
        return self._st

    def dispatch(self, event: Any) -> None:  # watchdog FileSystemEvent
        if event.is_directory:
            return
        st = self._get_st()
        if st is None:
            return
        try:
            with self._lock:
                st.session_state["dirty"] = True
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            logger.debug("_ChangeHandler.dispatch: %s", exc)


def register_watchdog(
    workspace_root: Path,
    paths: list[str] | None = None,
) -> None:
    """Start a watchdog observer on workspace directories.

    The observer calls ``st.session_state['dirty'] = True`` and
    ``st.rerun()`` whenever any file under the watched paths changes.

    This is **opt-in** (called only when the sidebar "Watch files" checkbox
    is ticked).  The observer is stored in ``st.session_state`` so it
    survives Streamlit reruns and is not re-created on every render.

    Degrades gracefully when:
    - ``watchdog`` is not installed (no-op + warning)
    - The observer fails to start (no-op + warning)
    - Streamlit is not available (no-op)

    Parameters
    ----------
    workspace_root:
        Absolute path to the workspace root.
    paths:
        Relative paths under workspace_root to watch.  Defaults to
        ``["outputs", ".agent-state", "datasets"]``.
    """
    if paths is None:
        paths = ["outputs", ".agent-state", "datasets"]

    # Guard: watchdog available?
    try:
        from watchdog.observers import Observer  # type: ignore[import-not-found]
        from watchdog.events import FileSystemEventHandler  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "register_watchdog: watchdog not installed — file-watch disabled. "
            "Install with: pip install watchdog"
        )
        return

    # Guard: streamlit available?
    try:
        import streamlit as st  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("register_watchdog: streamlit not available — skipping")
        return

    # Avoid creating duplicate observers across reruns
    observer_key = "_watchdog_observer"
    if observer_key in st.session_state and st.session_state[observer_key] is not None:
        existing: Any = st.session_state[observer_key]
        if existing.is_alive():
            logger.debug("register_watchdog: observer already running")
            return

    # Build a watchdog-compatible handler from our minimal _ChangeHandler
    handler = _ChangeHandler()

    class _WatchdogHandler(FileSystemEventHandler):
        def on_any_event(self, event: Any) -> None:
            handler.dispatch(event)

    try:
        observer = Observer()
        for rel_path in paths:
            abs_path = workspace_root / rel_path
            if abs_path.is_dir():
                observer.schedule(_WatchdogHandler(), str(abs_path), recursive=True)
                logger.debug("register_watchdog: watching %s", abs_path)
            else:
                logger.debug("register_watchdog: %s not found — skipping", abs_path)

        observer.start()
        st.session_state[observer_key] = observer
        logger.debug("register_watchdog: observer started")
    except Exception as exc:  # noqa: BLE001
        logger.warning("register_watchdog: failed to start observer: %s", exc)
