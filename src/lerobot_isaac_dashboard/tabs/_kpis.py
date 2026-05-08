"""_kpis.py — KPI tile helper for dual-mode rendering.

``render_kpi_row`` emits ``st.metric`` tiles when a Streamlit container is
available, and returns a Plotly table figure when called in static-export mode
(``container=None``).
"""

from __future__ import annotations

from typing import Any

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    go = None  # type: ignore[assignment]
    _HAS_PLOTLY = False

try:
    import streamlit as st

    _HAS_ST = True
except ImportError:
    st = None  # type: ignore[assignment]
    _HAS_ST = False


def render_kpi_row(
    container: Any,
    items: list[tuple[str, Any, str | None]],
) -> Any | None:
    """Render a row of KPI tiles.

    Parameters
    ----------
    container:
        Streamlit container (e.g. a tab returned by ``st.tabs()``).
        Pass ``None`` to use static-export mode.
    items:
        Sequence of ``(label, value, delta)`` tuples.
        ``delta`` may be ``None``.

    Returns
    -------
    go.Figure | None
        In live mode (``container`` is not None): emits ``st.metric`` calls
        on ``container`` and returns ``None``.
        In static mode: builds a ``go.Table`` figure summarising the KPIs and
        returns it so the static exporter can embed it.
    """
    if container is not None and _HAS_ST:
        # Live Streamlit path: render metric tiles
        n = len(items)
        if n == 0:
            return None
        cols = container.columns(n)
        for col, (label, value, delta) in zip(cols, items):
            col.metric(label=label, value=value, delta=delta)
        return None

    # Static / test path: build a Plotly table
    if not _HAS_PLOTLY:
        return None

    labels = [item[0] for item in items]
    values = [str(item[1]) if item[1] is not None else "—" for item in items]
    deltas = [str(item[2]) if item[2] is not None else "" for item in items]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>", "<b>Delta</b>"],
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[labels, values, deltas],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(title_text="KPIs", margin=dict(l=0, r=0, t=30, b=0))
    return fig
