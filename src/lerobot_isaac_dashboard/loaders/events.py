"""events.py — Agent event log loader.

Reads ``.agent-state/<sessionId>/events.jsonl`` files produced by agent
orchestrators during their lifecycle.

Schema (EVENTS_SCHEMA)
-----------------------
ts        : datetime64[ns, UTC] — event timestamp (ISO-8601)
session_id: string   — session directory name
phase     : string   — pipeline phase or agent name
event     : string   — event type (e.g. "start", "complete", "error")
data      : object   — raw JSON payload (kept as Python object / string)

When session_id is None, scans ALL sessions under ``.agent-state/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lerobot_isaac_dashboard.loaders._base import (
    LoaderResult,
    _align_to_schema,
    empty_df,
    safe_read_jsonl,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

EVENTS_SCHEMA: dict[str, str] = {
    "ts": "datetime64[ns, UTC]",
    "session_id": "string",
    "phase": "string",
    "event": "string",
    "data": "object",
}


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_events(workspace_root: Path, *, session_id: str | None = None) -> LoaderResult:
    """Load agent event logs from ``.agent-state/<sessionId>/events.jsonl``.

    Parameters
    ----------
    workspace_root:
        Workspace root directory.
    session_id:
        If given, restrict to a single session.  If None, scan all sessions.

    Returns
    -------
    LoaderResult
        df has schema EVENTS_SCHEMA.  is_empty=True when no events are found.
    """
    workspace_root = Path(workspace_root)
    agent_state_dir = workspace_root / ".agent-state"
    warnings: list[str] = []
    source_paths: list[Path] = []
    all_frames: list[pd.DataFrame] = []

    if not agent_state_dir.exists():
        return LoaderResult(
            df=empty_df(list(EVENTS_SCHEMA.keys()), EVENTS_SCHEMA),
            is_empty=True,
            source_paths=[],
            warnings=warnings,
        )

    if session_id is not None:
        session_dirs = [agent_state_dir / session_id]
    else:
        session_dirs = [d for d in sorted(agent_state_dir.iterdir()) if d.is_dir()]

    for sess_dir in session_dirs:
        events_file = sess_dir / "events.jsonl"
        df = safe_read_jsonl(events_file)
        if df is None:
            continue
        source_paths.append(events_file)
        if df.empty:
            warnings.append(f"{sess_dir.name}/events.jsonl: empty file")
            continue
        df["session_id"] = sess_dir.name
        all_frames.append(df)

    if not all_frames:
        return LoaderResult(
            df=empty_df(list(EVENTS_SCHEMA.keys()), EVENTS_SCHEMA),
            is_empty=True,
            source_paths=source_paths,
            warnings=warnings,
        )

    combined = pd.concat(all_frames, ignore_index=True)

    # Normalise common field names
    if "timestamp" in combined.columns and "ts" not in combined.columns:
        combined = combined.rename(columns={"timestamp": "ts"})
    if "type" in combined.columns and "event" not in combined.columns:
        combined = combined.rename(columns={"type": "event"})

    # Serialise 'data' column to string if it contains nested objects
    if "data" in combined.columns:
        combined["data"] = combined["data"].apply(
            lambda x: x if isinstance(x, (str, type(None))) else str(x)
        )

    combined, align_warnings = _align_to_schema(combined, EVENTS_SCHEMA)
    warnings.extend(align_warnings)

    return LoaderResult(
        df=combined,
        is_empty=len(combined) == 0,
        source_paths=source_paths,
        warnings=warnings,
    )
