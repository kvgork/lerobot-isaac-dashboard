"""test_imports.py — smoke tests for package importability and soft-import discipline.

Per ADR-0003: lerobot_isaac_dashboard must be importable without any heavy dependencies.
"""

import sys
import importlib
import types


def test_package_importable():
    """Package imports without any optional dependencies installed."""
    import lerobot_isaac_dashboard  # noqa: F401


def test_version_string():
    """__version__ is the expected semver string."""
    import lerobot_isaac_dashboard

    assert lerobot_isaac_dashboard.__version__ == "0.1.0"


def test_version_submodule():
    """version submodule is importable and exposes __version__."""
    from lerobot_isaac_dashboard import version

    assert hasattr(version, "__version__")
    assert version.__version__ == "0.1.0"


def test_soft_import_discipline_lerobot(monkeypatch):
    """Package must not import lerobot at the top level (ADR-0003).

    We inject a sentinel that raises on attribute access if lerobot were
    imported, then force-reload the package. No ImportError should propagate.
    """
    # Inject a broken sentinel for `lerobot` so any top-level import would fail
    sentinel = types.ModuleType("lerobot")
    sentinel.__spec__ = None

    def _bad_getattr(name):
        raise AttributeError(
            f"lerobot soft-import discipline violated: accessed lerobot.{name} at import time"
        )

    sentinel.__getattr__ = _bad_getattr
    monkeypatch.setitem(sys.modules, "lerobot", sentinel)

    # Remove cached copies so the import machinery re-executes module bodies
    for key in list(sys.modules):
        if key.startswith("lerobot_isaac_dashboard"):
            monkeypatch.delitem(sys.modules, key)

    # This must not raise even though sys.modules["lerobot"] is our broken sentinel
    import lerobot_isaac_dashboard  # noqa: F401

    assert lerobot_isaac_dashboard.__version__ == "0.1.0"
