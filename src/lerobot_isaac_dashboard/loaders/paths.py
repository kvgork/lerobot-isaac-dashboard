"""paths.py — WorkspacePaths resolver for the dashboard.

Wraps ``lerobot_isaac_meta.workspace_paths`` when available; falls back to
env-var or explicit workspace_root argument. The dashboard is a read-only
consumer so it never calls ``ensure_dirs()``.

Usage
-----
>>> from lerobot_isaac_dashboard.loaders.paths import load_paths
>>> wp = load_paths(workspace_root=Path("/my/workspace"))
>>> wp.datasets_dir
PosixPath('/my/workspace/datasets')
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    """Canonical path set for a single lerobot-isaac workspace.

    Attributes
    ----------
    workspace_root:
        Absolute path to the top-level workspace directory.
    datasets_dir:
        ``<workspace_root>/datasets``
    outputs_dir:
        ``<workspace_root>/outputs``
    agent_state_dir:
        ``<workspace_root>/.agent-state``
    configs_dir:
        ``<workspace_root>/packages/lerobot-isaac-configs/configs``
    """

    workspace_root: Path
    datasets_dir: Path
    outputs_dir: Path
    agent_state_dir: Path
    configs_dir: Path


def load_paths(workspace_root: Path | None = None) -> WorkspacePaths:
    """Resolve workspace paths.

    Resolution order:
    1. ``workspace_root`` argument (if given).
    2. ``LEROBOT_ISAAC_WORKSPACE`` env var (if set).
    3. ``lerobot_isaac_meta.workspace_paths`` soft-import (if installed).
    4. ``Path.cwd()`` as a last resort.

    Parameters
    ----------
    workspace_root:
        Explicit workspace root override.  Takes highest precedence.

    Returns
    -------
    WorkspacePaths
        Absolute paths to canonical workspace directories.
    """
    if workspace_root is not None:
        root = Path(workspace_root).expanduser().resolve()
    else:
        env_root = os.environ.get("LEROBOT_ISAAC_WORKSPACE")
        if env_root:
            root = Path(env_root).expanduser().resolve()
        else:
            # Try soft-import of lerobot_isaac_meta
            try:
                from lerobot_isaac_meta import workspace_paths as _wp  # type: ignore[import]

                root = _wp.WORKSPACE_ROOT
            except ImportError:
                root = Path.cwd()

    return WorkspacePaths(
        workspace_root=root,
        datasets_dir=root / "datasets",
        outputs_dir=root / "outputs",
        agent_state_dir=root / ".agent-state",
        configs_dir=root / "packages" / "lerobot-isaac-configs" / "configs",
    )
