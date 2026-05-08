"""session_state.py — Workspace root resolver and session ID helpers.

Priority order for workspace root resolution:
  1. CLI argument (explicit --workspace flag)
  2. LEROBOT_ISAAC_WORKSPACE environment variable
  3. lerobot_isaac_meta.workspace_paths.workspace_root() (soft-import)
  4. Current working directory
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_workspace_root(cli_arg: str | Path | None = None) -> Path:
    """Resolve the workspace root using the priority chain.

    Parameters
    ----------
    cli_arg:
        Explicit path passed via ``--workspace`` CLI flag.  Takes highest
        priority when provided and non-empty.

    Returns
    -------
    Path
        Absolute path to the workspace root (always resolved, always absolute).
    """
    # 1. CLI argument — highest priority
    if cli_arg is not None:
        p = Path(cli_arg).resolve()
        logger.debug("resolve_workspace_root: using CLI arg %s", p)
        return p

    # 2. Environment variable
    env_val = os.environ.get("LEROBOT_ISAAC_WORKSPACE", "").strip()
    if env_val:
        p = Path(env_val).resolve()
        logger.debug("resolve_workspace_root: using env LEROBOT_ISAAC_WORKSPACE %s", p)
        return p

    # 3. Soft-import lerobot_isaac_meta
    try:
        from lerobot_isaac_meta import workspace_paths  # type: ignore[import-not-found]

        p = Path(workspace_paths.workspace_root()).resolve()
        logger.debug("resolve_workspace_root: using lerobot_isaac_meta %s", p)
        return p
    except (ImportError, AttributeError, Exception) as exc:  # noqa: BLE001
        logger.debug(
            "resolve_workspace_root: lerobot_isaac_meta not available: %s", exc
        )

    # 4. Fallback: current working directory
    p = Path.cwd()
    logger.debug("resolve_workspace_root: falling back to cwd %s", p)
    return p


def list_session_ids(workspace_root: Path) -> list[str]:
    """Return a sorted list of session IDs found in ``<workspace_root>/.agent-state/``.

    A session ID is the name of any direct child directory of ``.agent-state/``
    that is not a hidden file (``.gitkeep``, ``.gitignore``, etc.).

    The list is sorted lexicographically (which preserves time-based ID order
    since session IDs are ``YYYYMMDD-HHMMSS-slug`` by convention).

    Parameters
    ----------
    workspace_root:
        Absolute path to the workspace root.

    Returns
    -------
    list[str]
        Sorted list of session ID strings.  Empty list when ``.agent-state/``
        does not exist or contains no valid sessions.
    """
    agent_state_dir = workspace_root / ".agent-state"
    if not agent_state_dir.is_dir():
        return []

    sessions: list[str] = []
    try:
        for entry in agent_state_dir.iterdir():
            # Skip hidden files / placeholders (.gitkeep, .gitignore, etc.)
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                sessions.append(entry.name)
    except OSError as exc:
        logger.warning("list_session_ids: could not read %s: %s", agent_state_dir, exc)
        return []

    return sorted(sessions)


def default_session_id(workspace_root: Path) -> str | None:
    """Return the most recent session ID by modification time, or None.

    Scans the same directories as :func:`list_session_ids` and returns the
    one with the latest ``mtime``.  Returns ``None`` when no sessions exist.

    Parameters
    ----------
    workspace_root:
        Absolute path to the workspace root.

    Returns
    -------
    str | None
        Session ID string or None.
    """
    agent_state_dir = workspace_root / ".agent-state"
    if not agent_state_dir.is_dir():
        return None

    best_name: str | None = None
    best_mtime: float = -1.0

    try:
        for entry in agent_state_dir.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    mtime = 0.0
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_name = entry.name
    except OSError as exc:
        logger.warning(
            "default_session_id: could not read %s: %s", agent_state_dir, exc
        )

    return best_name
