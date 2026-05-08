"""conftest.py — pytest fixtures for lerobot-isaac-dashboard tests.

Phase 1: minimal tmp_path-based workspace builder fixture.
Will be expanded in Phase 6 with full fixture factories for each loader.
"""

import pytest


@pytest.fixture
def workspace_root(tmp_path):
    """Return a minimal fake workspace directory tree.

    Phase 6 will populate datasets/, outputs/, .agent-state/ sub-trees
    with canonical fixture files for every loader test.
    """
    # Top-level directories that the dashboard expects to find
    (tmp_path / "datasets").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / ".agent-state").mkdir()
    return tmp_path
