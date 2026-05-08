"""curriculum.py — Curriculum stage loader.

Reads:
- ``outputs/curriculum_stage.json``  — current stage snapshot (contract pending)
- ``outputs/curriculum_history.jsonl`` — history of stage advancements

Contract (pending)
------------------
``lerobot-curriculum-agent`` does not yet write these files on disk.
This loader returns empty dicts/DataFrames when the files are absent.
A coordinated update is required in a follow-up phase when the curriculum
agent finalises its on-disk format.

Assumed curriculum_stage.json shape::

    {
        "stage":              str,
        "task_config":        dict,
        "advancement_reason": str,
        "ts":                 str (ISO-8601)
    }

Assumed curriculum_history.jsonl shape (one record per line)::

    {"ts": str, "stage": str, "advancement_reason": str, "task_config_diff": dict}

Returns dict[str, ...] with keys:
  "current" : dict (parsed curriculum_stage.json, or {})
  "history" : DataFrame (HISTORY_SCHEMA)
"""

from __future__ import annotations

from pathlib import Path


from lerobot_isaac_dashboard.loaders._base import (
    LoaderResult,
    _align_to_schema,
    empty_df,
    safe_read_json,
    safe_read_jsonl,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CURRICULUM_HISTORY_SCHEMA: dict[str, str] = {
    "ts": "datetime64[ns, UTC]",
    "stage": "string",
    "advancement_reason": "string",
    "task_config_diff": "object",
}


def _empty_result(warnings: list[str]) -> LoaderResult:
    return LoaderResult(
        df={
            "current": {},
            "history": empty_df(
                list(CURRICULUM_HISTORY_SCHEMA.keys()), CURRICULUM_HISTORY_SCHEMA
            ),
        },
        is_empty=True,
        source_paths=[],
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_curriculum(
    workspace_root: Path, *, session_id: str | None = None
) -> LoaderResult:
    """Load curriculum stage data from ``outputs/``.

    Parameters
    ----------
    workspace_root:
        Workspace root directory.
    session_id:
        Unused (curriculum data is not session-scoped).

    Returns
    -------
    LoaderResult
        df is a dict with:
          "current" — dict from curriculum_stage.json (or {})
          "history" — DataFrame from curriculum_history.jsonl

        Contract pending — lerobot-curriculum-agent does not yet write these
        files; loader returns is_empty=True with empty structures when absent.
    """
    workspace_root = Path(workspace_root)
    outputs_dir = workspace_root / "outputs"
    warnings: list[str] = []
    source_paths: list[Path] = []

    # --- current stage -------------------------------------------------------
    stage_path = outputs_dir / "curriculum_stage.json"
    current = safe_read_json(stage_path)
    if current is not None:
        source_paths.append(stage_path)
    else:
        current = {}

    # --- history -------------------------------------------------------------
    history_path = outputs_dir / "curriculum_history.jsonl"
    hist_df = safe_read_jsonl(history_path)
    if hist_df is not None and not hist_df.empty:
        source_paths.append(history_path)
        hist_df, align_warnings = _align_to_schema(hist_df, CURRICULUM_HISTORY_SCHEMA)
        warnings.extend(align_warnings)
    else:
        if history_path.exists() and (hist_df is None or hist_df.empty):
            warnings.append(
                "curriculum_history.jsonl exists but is empty or unreadable"
            )
        hist_df = empty_df(
            list(CURRICULUM_HISTORY_SCHEMA.keys()), CURRICULUM_HISTORY_SCHEMA
        )

    is_empty = not current and len(hist_df) == 0

    return LoaderResult(
        df={"current": current, "history": hist_df},
        is_empty=is_empty,
        source_paths=source_paths,
        warnings=warnings,
    )
