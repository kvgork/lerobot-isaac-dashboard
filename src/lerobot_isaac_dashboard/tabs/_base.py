"""_base.py — Tab Protocol and TabContext dataclass.

Defines the shared contract that every tab module must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore[assignment]

from lerobot_isaac_dashboard.loaders._base import LoaderResult


@dataclass
class TabContext:
    """All data a tab needs to render — passed as a single argument.

    Parameters
    ----------
    workspace_root:
        Absolute path to the workspace root.
    session_id:
        Optional session ID to scope loaders that are session-aware (events,
        autoresearch).  When None the loaders scan all sessions.
    loader_results:
        Mapping from loader slug (e.g. ``"eval_results"``, ``"parquet_dataset"``)
        to the corresponding :class:`LoaderResult` returned by the loader.
        Tabs read this dict — they never call loaders directly.
    refresh_ts:
        Timestamp of the last data refresh (shown in the UI header).
    """

    workspace_root: Path
    session_id: str | None
    loader_results: dict[str, LoaderResult]
    refresh_ts: datetime = field(default_factory=datetime.utcnow)


class Tab:
    """Protocol that every tab class must satisfy.

    This is a concrete base class (not a :class:`typing.Protocol`) so that
    ``isinstance`` checks work and subclasses get sensible defaults.

    Subclasses MUST override:
    - ``title``        — human-readable tab name
    - ``slug``         — filename-safe identifier (used in static export anchors)
    - ``render``       — build and optionally display figures

    Subclasses SHOULD set:
    - ``primary_loader_slug`` — key into ``ctx.loader_results`` for empty-state guard
    """

    title: str = "Unnamed Tab"
    slug: str = "unnamed"
    primary_loader_slug: str = ""

    def render(
        self,
        ctx: TabContext,
        *,
        container: Any = None,
    ) -> list[Any]:
        """Render the tab.

        Parameters
        ----------
        ctx:
            Shared tab context (loaders + workspace metadata).
        container:
            Streamlit container (e.g. ``st.container()``) when running live.
            ``None`` when called from the static exporter.

        Returns
        -------
        list[go.Figure]
            All Plotly figures produced by this render call.  The live path
            also calls ``container.plotly_chart(fig)``; the static path only
            returns the list.  Always a list — may be empty for the empty state.
        """
        raise NotImplementedError
