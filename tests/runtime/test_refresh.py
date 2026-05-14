"""tests/runtime/test_refresh.py — Unit tests for refresh helpers.

Tests cover:
- register_autorefresh no-ops when streamlit is absent or interval <= 0
- register_watchdog no-ops when watchdog is unavailable
- register_watchdog no-ops when streamlit is unavailable
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _remove_module(name: str) -> None:
    """Remove a module from sys.modules if present."""
    sys.modules.pop(name, None)


def _install_fake_streamlit(*, has_autorefresh: bool = False) -> MagicMock:
    """Install a minimal fake streamlit in sys.modules and return it."""
    fake_st = MagicMock()
    fake_st.session_state = {}
    sys.modules["streamlit"] = fake_st  # type: ignore[assignment]
    if has_autorefresh:
        fake_sar = MagicMock()
        sys.modules["streamlit_autorefresh"] = fake_sar  # type: ignore[assignment]
    return fake_st


# ---------------------------------------------------------------------------
# register_autorefresh
# ---------------------------------------------------------------------------


class TestRegisterAutorefresh:
    def test_noop_when_interval_zero(self):
        """Zero interval should not attempt to call any Streamlit API."""
        # We don't even need a fake st — the function should return early.
        from lerobot_isaac_dashboard.runtime.refresh import register_autorefresh

        register_autorefresh(0)  # should not raise

    def test_noop_when_interval_negative(self):
        from lerobot_isaac_dashboard.runtime.refresh import register_autorefresh

        register_autorefresh(-100)  # should not raise

    def test_noop_when_streamlit_absent(self):
        """Function must not raise when streamlit is not installed."""
        _remove_module("streamlit")
        _remove_module("streamlit_autorefresh")

        with patch.dict(
            sys.modules, {"streamlit": None, "streamlit_autorefresh": None}
        ):
            from lerobot_isaac_dashboard.runtime import refresh as refresh_mod
            import importlib

            importlib.reload(refresh_mod)
            refresh_mod.register_autorefresh(5000)  # should not raise

    def test_uses_st_autorefresh_when_available(self, monkeypatch):
        """When streamlit_autorefresh is present, st_autorefresh should be called."""
        fake_sar = MagicMock()
        monkeypatch.setitem(sys.modules, "streamlit_autorefresh", fake_sar)
        monkeypatch.setitem(sys.modules, "streamlit", MagicMock())

        from lerobot_isaac_dashboard.runtime import refresh as refresh_mod
        import importlib

        importlib.reload(refresh_mod)
        refresh_mod.register_autorefresh(10_000, key="test_key")

        fake_sar.st_autorefresh.assert_called_once_with(interval=10_000, key="test_key")

    def test_warns_when_streamlit_autorefresh_missing(self, monkeypatch):
        """Without streamlit_autorefresh: emit a sidebar warning, do NOT inject
        a meta-refresh tag (full page reloads reset every widget state)."""
        fake_st = MagicMock()
        fake_st.session_state = {}
        # Block any real streamlit_autorefresh import — even if a prior test
        # cached the real module in sys.modules, patch.dict with value=None
        # forces ImportError on `from streamlit_autorefresh import ...`.
        with patch.dict(
            sys.modules,
            {"streamlit": fake_st, "streamlit_autorefresh": None},
        ):
            from lerobot_isaac_dashboard.runtime import refresh as refresh_mod
            import importlib

            importlib.reload(refresh_mod)
            refresh_mod.register_autorefresh(30_000)

        # Sidebar warning shown, NO meta-refresh markdown injected.
        assert fake_st.sidebar.warning.called
        if fake_st.markdown.called:
            for call in fake_st.markdown.call_args_list:
                payload = call[0][0] if call[0] else ""
                assert "http-equiv" not in payload.lower(), (
                    "meta-refresh must NOT be injected — it triggers full page "
                    "reload + widget state reset"
                )


# ---------------------------------------------------------------------------
# register_watchdog
# ---------------------------------------------------------------------------


class TestRegisterWatchdog:
    def test_noop_when_watchdog_absent(self, tmp_path, monkeypatch):
        """No exception raised when watchdog is not installed."""
        monkeypatch.delitem(sys.modules, "watchdog", raising=False)
        monkeypatch.delitem(sys.modules, "watchdog.observers", raising=False)
        monkeypatch.delitem(sys.modules, "watchdog.events", raising=False)

        with patch.dict(
            sys.modules,
            {"watchdog": None, "watchdog.observers": None, "watchdog.events": None},
        ):
            from lerobot_isaac_dashboard.runtime import refresh as refresh_mod
            import importlib

            importlib.reload(refresh_mod)
            refresh_mod.register_watchdog(tmp_path)  # should not raise

    def test_noop_when_streamlit_absent(self, tmp_path, monkeypatch):
        """No exception raised when streamlit is not installed."""
        # Watchdog present, streamlit absent
        fake_watchdog = MagicMock()
        fake_observer_cls = MagicMock()
        fake_watchdog.observers.Observer = fake_observer_cls
        monkeypatch.setitem(sys.modules, "watchdog", fake_watchdog)
        monkeypatch.setitem(sys.modules, "watchdog.observers", fake_watchdog.observers)
        monkeypatch.setitem(sys.modules, "watchdog.events", MagicMock())
        monkeypatch.delitem(sys.modules, "streamlit", raising=False)

        with patch.dict(sys.modules, {"streamlit": None}):
            from lerobot_isaac_dashboard.runtime import refresh as refresh_mod
            import importlib

            importlib.reload(refresh_mod)
            refresh_mod.register_watchdog(tmp_path)  # should not raise

    def test_noop_when_observer_raises(self, tmp_path, monkeypatch):
        """No exception raised when the Observer constructor fails."""
        # Simulate watchdog installed but Observer() raises
        fake_observer_cls = MagicMock(side_effect=RuntimeError("no inotify"))
        fake_observers_mod = MagicMock()
        fake_observers_mod.Observer = fake_observer_cls
        fake_events_mod = MagicMock()

        fake_st = MagicMock()
        fake_st.session_state = {}

        monkeypatch.setitem(sys.modules, "watchdog", MagicMock())
        monkeypatch.setitem(sys.modules, "watchdog.observers", fake_observers_mod)
        monkeypatch.setitem(sys.modules, "watchdog.events", fake_events_mod)
        monkeypatch.setitem(sys.modules, "streamlit", fake_st)

        from lerobot_isaac_dashboard.runtime import refresh as refresh_mod
        import importlib

        importlib.reload(refresh_mod)
        refresh_mod.register_watchdog(tmp_path)  # should not raise

    def test_skips_nonexistent_paths(self, tmp_path, monkeypatch):
        """Paths that do not exist are silently skipped."""
        fake_observer = MagicMock()
        fake_observer.is_alive.return_value = False
        fake_observer_cls = MagicMock(return_value=fake_observer)
        fake_observers_mod = MagicMock()
        fake_observers_mod.Observer = fake_observer_cls
        fake_events_mod = MagicMock()

        fake_st = MagicMock()
        fake_st.session_state = {}

        monkeypatch.setitem(sys.modules, "watchdog", MagicMock())
        monkeypatch.setitem(sys.modules, "watchdog.observers", fake_observers_mod)
        monkeypatch.setitem(sys.modules, "watchdog.events", fake_events_mod)
        monkeypatch.setitem(sys.modules, "streamlit", fake_st)

        from lerobot_isaac_dashboard.runtime import refresh as refresh_mod
        import importlib

        importlib.reload(refresh_mod)

        # Pass a nonexistent path — should not raise
        refresh_mod.register_watchdog(tmp_path, paths=["nonexistent_dir_abc123"])
        # Observer.schedule should NOT have been called (dir doesn't exist)
        fake_observer.schedule.assert_not_called()
