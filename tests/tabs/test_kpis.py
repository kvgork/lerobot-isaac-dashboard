"""test_kpis.py — Tests for the KPI tile helper (_kpis.py).

Covers:
    - Static mode (container=None): returns a go.Table figure
    - Live mode (mock container): calls .columns() and per-column .metric()
    - Empty items list: graceful handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import plotly.graph_objects as go
import pytest

from lerobot_isaac_dashboard.tabs._kpis import render_kpi_row


# ---------------------------------------------------------------------------
# Static mode (container=None)
# ---------------------------------------------------------------------------

def test_static_mode_returns_figure():
    """container=None must return a go.Figure."""
    items = [("Accuracy", 0.95, "+0.02"), ("Loss", 0.3, None)]
    result = render_kpi_row(None, items)
    assert isinstance(result, go.Figure)


def test_static_mode_table_has_correct_structure():
    """The returned figure must contain a go.Table with 3 columns."""
    items = [("Metric A", 42, None), ("Metric B", "good", "+1")]
    fig = render_kpi_row(None, items)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    table = fig.data[0]
    assert isinstance(table, go.Table)
    # 3-column header: Metric, Value, Delta
    assert len(table.header.values) == 3


def test_static_mode_values_present():
    """KPI values must appear in the table cells."""
    items = [("Episodes", 1234, None), ("FPS", 30.5, "+2.1")]
    fig = render_kpi_row(None, items)
    table = fig.data[0]
    # Flatten all cell values to strings
    all_cell_values = [str(v) for col in table.cells.values for v in col]
    assert "Episodes" in all_cell_values
    assert "1234" in all_cell_values
    assert "FPS" in all_cell_values
    assert "30.5" in all_cell_values


def test_static_mode_none_delta_shows_empty():
    """None delta must render as empty string (not 'None')."""
    items = [("Steps", 1000, None)]
    fig = render_kpi_row(None, items)
    table = fig.data[0]
    delta_col = table.cells.values[2]  # third column is Delta
    assert delta_col[0] == ""


def test_static_mode_empty_items_returns_figure():
    """Empty items list must still return a go.Figure."""
    result = render_kpi_row(None, [])
    assert isinstance(result, go.Figure)


# ---------------------------------------------------------------------------
# Live mode (mock Streamlit container)
# ---------------------------------------------------------------------------

def test_live_mode_calls_columns():
    """Live mode must call container.columns(n) with the number of items."""
    items = [("A", 1, None), ("B", 2, "+1"), ("C", 3, None)]

    mock_container = MagicMock()
    # columns() returns a list of mock column objects
    mock_cols = [MagicMock(), MagicMock(), MagicMock()]
    mock_container.columns.return_value = mock_cols

    result = render_kpi_row(mock_container, items)

    assert result is None  # live mode returns None
    mock_container.columns.assert_called_once_with(3)


def test_live_mode_calls_metric_on_each_column():
    """Live mode must call .metric(label=, value=, delta=) on each column."""
    items = [("Episodes", 50, None), ("Frames", 5000, "+100")]

    mock_container = MagicMock()
    mock_cols = [MagicMock(), MagicMock()]
    mock_container.columns.return_value = mock_cols

    render_kpi_row(mock_container, items)

    mock_cols[0].metric.assert_called_once_with(label="Episodes", value=50, delta=None)
    mock_cols[1].metric.assert_called_once_with(label="Frames", value=5000, delta="+100")


def test_live_mode_empty_items_no_error():
    """Empty items in live mode must not raise."""
    mock_container = MagicMock()
    result = render_kpi_row(mock_container, [])
    assert result is None


def test_live_mode_returns_none():
    """Live mode always returns None."""
    items = [("X", 1, None)]
    mock_container = MagicMock()
    mock_container.columns.return_value = [MagicMock()]
    result = render_kpi_row(mock_container, items)
    assert result is None
