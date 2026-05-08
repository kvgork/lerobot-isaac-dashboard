"""tests/runtime/test_session_state.py — Unit tests for session_state helpers.

Tests cover:
- resolve_workspace_root priority chain (CLI > env > meta > cwd)
- list_session_ids: sorted, excludes hidden entries
- default_session_id: returns most-recent by mtime, None when empty
"""

from __future__ import annotations

import os


from lerobot_isaac_dashboard.runtime.session_state import (
    default_session_id,
    list_session_ids,
    resolve_workspace_root,
)


# ---------------------------------------------------------------------------
# resolve_workspace_root
# ---------------------------------------------------------------------------


class TestResolveWorkspaceRoot:
    def test_cli_arg_takes_priority(self, tmp_path, monkeypatch):
        """CLI arg wins even when env var is set."""
        monkeypatch.setenv("LEROBOT_ISAAC_WORKSPACE", str(tmp_path / "env_path"))
        cli_path = tmp_path / "cli_path"
        cli_path.mkdir()

        result = resolve_workspace_root(cli_arg=cli_path)

        assert result == cli_path.resolve()

    def test_env_var_used_when_no_cli_arg(self, tmp_path, monkeypatch):
        """LEROBOT_ISAAC_WORKSPACE is used when no CLI arg is given."""
        env_path = tmp_path / "from_env"
        env_path.mkdir()
        monkeypatch.setenv("LEROBOT_ISAAC_WORKSPACE", str(env_path))
        # Ensure meta is not importable
        monkeypatch.delitem(
            __import__("sys").modules, "lerobot_isaac_meta", raising=False
        )

        result = resolve_workspace_root(cli_arg=None)

        assert result == env_path.resolve()

    def test_empty_env_var_skipped(self, tmp_path, monkeypatch):
        """Empty LEROBOT_ISAAC_WORKSPACE falls through to next priority."""
        monkeypatch.setenv("LEROBOT_ISAAC_WORKSPACE", "")
        monkeypatch.delitem(
            __import__("sys").modules, "lerobot_isaac_meta", raising=False
        )
        monkeypatch.chdir(tmp_path)

        result = resolve_workspace_root(cli_arg=None)

        # Should fall back to cwd (tmp_path)
        assert result == tmp_path.resolve()

    def test_cwd_fallback(self, tmp_path, monkeypatch):
        """Falls back to cwd when no CLI arg, no env var, no meta package."""
        monkeypatch.delenv("LEROBOT_ISAAC_WORKSPACE", raising=False)
        monkeypatch.delitem(
            __import__("sys").modules, "lerobot_isaac_meta", raising=False
        )
        monkeypatch.chdir(tmp_path)

        result = resolve_workspace_root(cli_arg=None)

        assert result == tmp_path.resolve()

    def test_cli_arg_as_string(self, tmp_path):
        """CLI arg may be a str, not just Path."""
        result = resolve_workspace_root(cli_arg=str(tmp_path))
        assert result == tmp_path.resolve()

    def test_meta_soft_import_used(self, tmp_path, monkeypatch):
        """When meta is importable and returns a path, it is used."""
        meta_path = tmp_path / "from_meta"
        meta_path.mkdir()

        # Inject a fake lerobot_isaac_meta module
        import types
        import sys

        fake_meta = types.ModuleType("lerobot_isaac_meta")
        fake_wp = types.ModuleType("lerobot_isaac_meta.workspace_paths")
        fake_wp.workspace_root = lambda: str(meta_path)  # type: ignore[attr-defined]
        fake_meta.workspace_paths = fake_wp  # type: ignore[attr-defined]
        sys.modules["lerobot_isaac_meta"] = fake_meta
        sys.modules["lerobot_isaac_meta.workspace_paths"] = fake_wp

        monkeypatch.delenv("LEROBOT_ISAAC_WORKSPACE", raising=False)

        try:
            result = resolve_workspace_root(cli_arg=None)
            assert result == meta_path.resolve()
        finally:
            sys.modules.pop("lerobot_isaac_meta", None)
            sys.modules.pop("lerobot_isaac_meta.workspace_paths", None)


# ---------------------------------------------------------------------------
# list_session_ids
# ---------------------------------------------------------------------------


class TestListSessionIds:
    def test_returns_sorted_list(self, tmp_path):
        agent_state = tmp_path / ".agent-state"
        agent_state.mkdir()
        (agent_state / "20260508-120000-beta").mkdir()
        (agent_state / "20260507-080000-alpha").mkdir()
        (agent_state / "20260509-000000-gamma").mkdir()

        result = list_session_ids(tmp_path)

        assert result == [
            "20260507-080000-alpha",
            "20260508-120000-beta",
            "20260509-000000-gamma",
        ]

    def test_excludes_hidden_entries(self, tmp_path):
        agent_state = tmp_path / ".agent-state"
        agent_state.mkdir()
        (agent_state / "real-session").mkdir()
        (agent_state / ".gitkeep").touch()
        (agent_state / ".gitignore").touch()

        result = list_session_ids(tmp_path)

        assert result == ["real-session"]

    def test_empty_when_no_agent_state(self, tmp_path):
        result = list_session_ids(tmp_path)
        assert result == []

    def test_empty_when_agent_state_has_no_dirs(self, tmp_path):
        agent_state = tmp_path / ".agent-state"
        agent_state.mkdir()
        (agent_state / ".gitkeep").touch()

        result = list_session_ids(tmp_path)

        assert result == []

    def test_ignores_files_not_dirs(self, tmp_path):
        agent_state = tmp_path / ".agent-state"
        agent_state.mkdir()
        (agent_state / "session-a").mkdir()
        (agent_state / "events.jsonl").touch()  # file, not a dir

        result = list_session_ids(tmp_path)

        assert result == ["session-a"]


# ---------------------------------------------------------------------------
# default_session_id
# ---------------------------------------------------------------------------


class TestDefaultSessionId:
    def test_returns_most_recent_by_mtime(self, tmp_path):
        agent_state = tmp_path / ".agent-state"
        agent_state.mkdir()

        old = agent_state / "20260507-old"
        old.mkdir()
        # Touch to set a known mtime ordering
        os.utime(old, (1_000_000, 1_000_000))

        new = agent_state / "20260508-new"
        new.mkdir()
        os.utime(new, (2_000_000, 2_000_000))

        result = default_session_id(tmp_path)

        assert result == "20260508-new"

    def test_returns_none_when_no_sessions(self, tmp_path):
        agent_state = tmp_path / ".agent-state"
        agent_state.mkdir()
        (agent_state / ".gitkeep").touch()

        result = default_session_id(tmp_path)

        assert result is None

    def test_returns_none_when_no_agent_state(self, tmp_path):
        result = default_session_id(tmp_path)
        assert result is None

    def test_single_session_returned(self, tmp_path):
        agent_state = tmp_path / ".agent-state"
        agent_state.mkdir()
        (agent_state / "only-session").mkdir()

        result = default_session_id(tmp_path)

        assert result == "only-session"
