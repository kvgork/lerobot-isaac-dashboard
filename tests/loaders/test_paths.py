"""Tests for loaders/paths.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lerobot_isaac_dashboard.loaders.paths import WorkspacePaths, load_paths


def test_load_paths_from_explicit_root(tmp_path):
    wp = load_paths(workspace_root=tmp_path)
    assert wp.workspace_root == tmp_path
    assert wp.datasets_dir == tmp_path / "datasets"
    assert wp.outputs_dir == tmp_path / "outputs"
    assert wp.agent_state_dir == tmp_path / ".agent-state"


def test_load_paths_all_absolute(tmp_path):
    wp = load_paths(workspace_root=tmp_path)
    assert wp.workspace_root.is_absolute()
    assert wp.datasets_dir.is_absolute()
    assert wp.outputs_dir.is_absolute()
    assert wp.agent_state_dir.is_absolute()


def test_load_paths_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("LEROBOT_ISAAC_WORKSPACE", str(tmp_path))
    # Remove cached sys.modules if needed
    wp = load_paths()
    assert wp.workspace_root == tmp_path.resolve()


def test_load_paths_explicit_overrides_env(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("LEROBOT_ISAAC_WORKSPACE", str(tmp_path))
    wp = load_paths(workspace_root=other)
    assert wp.workspace_root == other


def test_workspace_paths_is_frozen(tmp_path):
    """WorkspacePaths is a frozen dataclass — attributes cannot be reassigned."""
    wp = load_paths(workspace_root=tmp_path)
    with pytest.raises((AttributeError, TypeError)):
        wp.workspace_root = tmp_path / "other"  # type: ignore[misc]
