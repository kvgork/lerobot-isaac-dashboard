"""eval_results.py — Evaluation result loader.

Reads ``outputs/eval/*.json`` files produced by ``lerobot-evaluation-agent``.

Contract (pending)
------------------
The exact JSON schema written by ``lerobot-evaluation-agent`` is not yet
finalised.  This loader assumes the following top-level keys; missing keys
are filled with NA and a warning is emitted:

    {
        "run_id":            str   — e.g. "smolvla_20260508_001"
        "task":              str   — e.g. "pick_and_place"
        "ts":                str   — ISO-8601 timestamp
        "pc_success":        float — 0.0..1.0
        "n_episodes":        int
        "intervention_rate": float — 0.0..1.0
        "mean_ep_len":       float — mean episode length (steps)
    }

This contract is documented here so that the evaluation agent implementation
can be coordinated in a follow-up phase.

Schema (EVAL_SCHEMA)
--------------------
run_id            : string
task              : string
ts                : datetime64[ns, UTC]
pc_success        : Float64
n_episodes        : Int64
intervention_rate : Float64
mean_ep_len       : Float64
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lerobot_isaac_dashboard.loaders._base import (
    LoaderResult,
    _align_to_schema,
    empty_df,
    glob_runs,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

EVAL_SCHEMA: dict[str, str] = {
    "run_id": "string",
    "task": "string",
    "ts": "datetime64[ns, UTC]",
    "pc_success": "Float64",
    "n_episodes": "Int64",
    "intervention_rate": "Float64",
    "mean_ep_len": "Float64",
}

_EMPTY_DF = empty_df(list(EVAL_SCHEMA.keys()), EVAL_SCHEMA)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_eval_results(
    workspace_root: Path, *, session_id: str | None = None
) -> LoaderResult:
    """Load evaluation results from ``outputs/eval/*.json``.

    Returns
    -------
    LoaderResult
        df has schema EVAL_SCHEMA, one row per eval JSON file.
        is_empty=True when no files are found.

    Notes
    -----
    Contract pending — eval JSON shape assumed from agent interface.
    A coordinated update is required when lerobot-evaluation-agent
    finalises its on-disk format.
    """
    workspace_root = Path(workspace_root)
    eval_dir = workspace_root / "outputs" / "eval"
    warnings: list[str] = []
    source_paths: list[Path] = []

    json_files = glob_runs(eval_dir, "*.json")
    if not json_files:
        return LoaderResult(
            df=_EMPTY_DF.copy(),
            is_empty=True,
            source_paths=[],
            warnings=warnings,
        )

    rows: list[dict] = []
    for jf in json_files:
        try:
            with jf.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                warnings.append(f"{jf.name}: expected JSON object, got {type(data).__name__} — skipped")
                continue
            source_paths.append(jf)
            row = _extract_eval_row(data, jf.name, warnings)
            rows.append(row)
        except json.JSONDecodeError as exc:
            warnings.append(f"{jf.name}: JSON parse error — {exc}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{jf.name}: unexpected error — {exc}")

    if not rows:
        return LoaderResult(
            df=_EMPTY_DF.copy(),
            is_empty=True,
            source_paths=source_paths,
            warnings=warnings,
        )

    df = pd.DataFrame(rows)
    df, align_warnings = _align_to_schema(df, EVAL_SCHEMA)
    warnings.extend(align_warnings)
    return LoaderResult(
        df=df,
        is_empty=len(df) == 0,
        source_paths=source_paths,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_eval_row(data: dict, filename: str, warnings: list[str]) -> dict:
    """Extract a canonical row dict from an eval JSON object."""
    row: dict = {}
    for key in EVAL_SCHEMA:
        if key not in data:
            warnings.append(
                f"{filename}: missing key '{key}' — will be NA"
            )
        row[key] = data.get(key)
    # Derive run_id from filename if absent
    if row.get("run_id") is None:
        row["run_id"] = filename.removesuffix(".json")
    return row
