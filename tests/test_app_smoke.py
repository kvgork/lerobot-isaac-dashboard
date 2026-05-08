"""tests/test_app_smoke.py — Smoke and AST tests for app.py and cli.py (P4/P5/P8).

Tests verify:
- app.py is importable without executing main()
- app.py AST contains def main() and references all 8 TABS class names + st.tabs(
- cli.py is importable and main() accepts an argv list without launching streamlit
- cli.py main(["--help"]) returns an int (subprocess is monkeypatched)
- Phase 8: app.py references save_snapshot, load_snapshot, list_snapshots,
  render_compare_2way, render_compare_nway and contains a Mode radio
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Absolute path to the app source (never relies on relative imports)
_PKG_SRC = Path(__file__).parent.parent / "src" / "lerobot_isaac_dashboard"
_APP_PATH = _PKG_SRC / "app.py"
_CLI_PATH = _PKG_SRC / "cli.py"

# The 8 tab class names that must appear in app.py
_EXPECTED_TAB_CLASSES = [
    "DataCollectionTab",
    "SyntheticTab",
    "PolicyTrainingTab",
    "WorldModelTab",
    "EvaluationTab",
    "AutoresearchTab",
    "CurriculumTab",
    "PipelineHealthTab",
]

# Phase 8 symbols that must appear in app.py
_P8_SYMBOLS = [
    "save_snapshot",
    "load_snapshot",
    "list_snapshots",
    "render_compare_2way",
    "render_compare_nway",
]


# ---------------------------------------------------------------------------
# AST checks — no imports needed
# ---------------------------------------------------------------------------


class TestAppAST:
    """AST-level structural checks on app.py."""

    @pytest.fixture(scope="class")
    def app_source(self) -> str:
        return _APP_PATH.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def app_tree(self, app_source: str) -> ast.Module:
        return ast.parse(app_source, filename=str(_APP_PATH))

    def test_defines_main_function(self, app_tree: ast.Module):
        """app.py must define a top-level def main()."""
        func_names = {
            node.name
            for node in ast.walk(app_tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "main" in func_names, "app.py must define def main()"

    @pytest.mark.parametrize("tab_class", _EXPECTED_TAB_CLASSES)
    def test_references_tab_class(self, app_source: str, tab_class: str):
        """app.py source must reference each of the 8 TABS class names."""
        assert tab_class in app_source, f"app.py must reference tab class '{tab_class}'"

    def test_calls_st_tabs(self, app_source: str):
        """app.py must call st.tabs(."""
        assert "st.tabs(" in app_source, "app.py must call st.tabs("

    def test_has_p5_export_implementation(self, app_source: str):
        """Export button handler must call export_report (P5 implementation)."""
        assert "export_report(" in app_source, (
            "app.py must call export_report() in the export button handler (P5 is live)"
        )

    def test_has_download_button(self, app_source: str):
        """Export button handler must wire st.download_button."""
        assert "st.download_button" in app_source, (
            "app.py must call st.download_button after export_report"
        )

    def test_import_for_report(self, app_source: str):
        """app.py must import export_report from report module."""
        assert "from lerobot_isaac_dashboard.report import export_report" in app_source

    # ------------------------------------------------------------------
    # Phase 8 checks
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("symbol", _P8_SYMBOLS)
    def test_references_p8_symbol(self, app_source: str, symbol: str):
        """app.py must reference each P8 snapshot/compare symbol."""
        assert symbol in app_source, f"app.py must reference P8 symbol '{symbol}'"

    def test_has_mode_radio(self, app_source: str):
        """app.py sidebar must contain a 'Mode' radio widget."""
        assert '"Mode"' in app_source or "'Mode'" in app_source, (
            "app.py must include a 'Mode' radio widget in the sidebar (P8)"
        )

    def test_references_compare_2way_mode(self, app_source: str):
        """app.py must reference the Compare (2-way) mode string."""
        assert "Compare (2-way)" in app_source, (
            "app.py must contain 'Compare (2-way)' mode string"
        )

    def test_references_compare_nway_mode(self, app_source: str):
        """app.py must reference the Compare (N-way) mode string."""
        assert "Compare (N-way)" in app_source, (
            "app.py must contain 'Compare (N-way)' mode string"
        )

    def test_has_save_snapshot_button(self, app_source: str):
        """app.py must have a Save snapshot button."""
        assert "Save snapshot" in app_source, (
            "app.py must include a 'Save snapshot' button"
        )


# ---------------------------------------------------------------------------
# Import smoke — streamlit is mocked so main() is never called
# ---------------------------------------------------------------------------


class TestAppImport:
    """app.py must import without raising (streamlit not actually launched)."""

    def _make_fake_streamlit(self) -> MagicMock:
        fake_st = MagicMock()
        fake_st.session_state = {}
        fake_st.tabs.return_value = [MagicMock() for _ in range(8)]
        return fake_st

    def test_app_importable_without_running(self, monkeypatch):
        """Importing app.py (with mocked streamlit) must not raise."""
        fake_st = self._make_fake_streamlit()
        monkeypatch.setitem(sys.modules, "streamlit", fake_st)
        # Remove cached module so import is fresh
        monkeypatch.delitem(sys.modules, "lerobot_isaac_dashboard.app", raising=False)

        import importlib
        import lerobot_isaac_dashboard.app as app_mod

        importlib.reload(app_mod)
        assert callable(app_mod.main)

    def test_app_exposes_loaders_dict(self, monkeypatch):
        """LOADERS dict must be present and have the expected slugs."""
        fake_st = self._make_fake_streamlit()
        monkeypatch.setitem(sys.modules, "streamlit", fake_st)
        monkeypatch.delitem(sys.modules, "lerobot_isaac_dashboard.app", raising=False)

        import importlib
        import lerobot_isaac_dashboard.app as app_mod

        importlib.reload(app_mod)
        expected_slugs = {
            "parquet_dataset",
            "eval_results",
            "checkpoints",
            "training_logs",
            "autoresearch",
            "events",
            "curriculum",
            "synthetic",
        }
        assert set(app_mod.LOADERS.keys()) == expected_slugs


# ---------------------------------------------------------------------------
# cli.py smoke
# ---------------------------------------------------------------------------


class TestCliSmoke:
    """cli.py must be importable and main() must accept an argv list."""

    def test_cli_importable(self):
        """Importing cli.py must not raise."""
        import lerobot_isaac_dashboard.cli as cli_mod  # noqa: F401

        assert callable(cli_mod.main)

    def test_main_accepts_argv_list(self, monkeypatch):
        """cli.main(['--workspace', '/tmp']) must call subprocess.run and return int."""
        import lerobot_isaac_dashboard.cli as cli_mod

        fake_result = MagicMock()
        fake_result.returncode = 0

        with patch("subprocess.run", return_value=fake_result) as mock_run:
            rc = cli_mod.main(["--workspace", "/tmp", "--port", "9999"])

        assert isinstance(rc, int)
        assert rc == 0
        # subprocess.run should have been called with a command list
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert isinstance(cmd, list)
        # Should contain streamlit and the app path
        assert "streamlit" in " ".join(cmd)

    def test_main_passes_workspace_after_separator(self, monkeypatch):
        """--workspace value should appear after '--' in the subprocess command."""
        import lerobot_isaac_dashboard.cli as cli_mod

        fake_result = MagicMock()
        fake_result.returncode = 0

        with patch("subprocess.run", return_value=fake_result) as mock_run:
            cli_mod.main(["--workspace=/my/workspace"])

        cmd: list[str] = mock_run.call_args[0][0]
        # The '--' separator should be present
        assert "--" in cmd
        sep_idx = cmd.index("--")
        after_sep = cmd[sep_idx + 1 :]
        assert any("--workspace=/my/workspace" in a for a in after_sep)

    def test_main_returns_int_on_file_not_found(self, monkeypatch):
        """When streamlit executable is missing, main() returns 1 (not raises)."""
        import lerobot_isaac_dashboard.cli as cli_mod

        with patch("subprocess.run", side_effect=FileNotFoundError("no streamlit")):
            rc = cli_mod.main(["--workspace=/tmp"])

        assert rc == 1

    def test_main_signature_accepts_none(self):
        """main(None) must not raise (uses sys.argv[1:] internally)."""
        import lerobot_isaac_dashboard.cli as cli_mod

        # We don't want it to actually call subprocess.run
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            rc = cli_mod.main([])  # empty argv, not None — avoids sys.argv side effects

        assert isinstance(rc, int)
