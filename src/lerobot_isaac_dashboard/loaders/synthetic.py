"""synthetic.py — Synthetic data source breakdown loader.

Reads ``meta/episodes.parquet`` and ``meta/tasks.parquet`` from a merged
dataset directory and produces a per-episode breakdown of real vs sim_dr vs
mimicgen episodes.

The ``source`` column is written by:
- ``lerobot_isaac_synthetic.isaac_dr.parquet_writer._tag_source_column``
  (source="sim_dr")
- ``lerobot_isaac_synthetic.merge_utilities._copy_and_reindex_episodes``
  (propagates source from constituent datasets)
- Real teleop episodes use source="real"

The loader scans ``datasets/`` for any merged dataset (i.e. one that contains
a ``meta/episodes.parquet`` with mixed source values).  If no merged dataset
is found it falls back to loading from any dataset that has a ``source`` column.

Schema (SYNTHETIC_SCHEMA)
--------------------------
episode_index: Int64   — global episode index in the merged dataset
length       : Int64   — episode length (steps)
source       : string  — "real" | "sim_dr" | "mimicgen" | "unknown"
task         : string  — task string from meta/tasks.parquet (joined on tasks_index)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lerobot_isaac_dashboard.loaders._base import (
    LoaderResult,
    _align_to_schema,
    empty_df,
    safe_read_parquet,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SYNTHETIC_SCHEMA: dict[str, str] = {
    "episode_index": "Int64",
    "length": "Int64",
    "source": "string",
    "task": "string",
}


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_synthetic(
    workspace_root: Path, *, session_id: str | None = None
) -> LoaderResult:
    """Load synthetic data source breakdown from dataset meta files.

    Scans ``datasets/`` for directories containing ``meta/episodes.parquet``
    with a ``source`` column.  Returns one row per episode.

    Returns
    -------
    LoaderResult
        df has schema SYNTHETIC_SCHEMA.  is_empty=True when no datasets
        with a source column are found.
    """
    workspace_root = Path(workspace_root)
    datasets_dir = workspace_root / "datasets"
    warnings: list[str] = []
    source_paths: list[Path] = []
    all_frames: list[pd.DataFrame] = []

    if not datasets_dir.exists():
        return LoaderResult(
            df=empty_df(list(SYNTHETIC_SCHEMA.keys()), SYNTHETIC_SCHEMA),
            is_empty=True,
            source_paths=[],
            warnings=warnings,
        )

    # Find all meta/ directories (two levels deep to match datasets/<task>/<repo> or datasets/<repo>)
    meta_dirs: list[Path] = []
    for candidate in datasets_dir.glob("*/meta/"):
        meta_dirs.append(candidate)
    for candidate in datasets_dir.glob("*/*/meta/"):
        meta_dirs.append(candidate)

    seen: set[Path] = set()
    for meta_dir in meta_dirs:
        meta_dir = meta_dir.resolve()
        if meta_dir in seen:
            continue
        seen.add(meta_dir)

        episodes_path = meta_dir / "episodes.parquet"
        tasks_path = meta_dir / "tasks.parquet"

        ep_df = safe_read_parquet(episodes_path)
        if ep_df is None:
            continue
        if "source" not in ep_df.columns:
            # No source column — skip (not a merged/tagged dataset)
            continue

        source_paths.append(episodes_path)

        # Join task string if tasks.parquet exists
        task_series = _load_task_map(tasks_path, warnings)
        ep_df = _build_frame(ep_df, task_series, warnings)
        all_frames.append(ep_df)

    if not all_frames:
        return LoaderResult(
            df=empty_df(list(SYNTHETIC_SCHEMA.keys()), SYNTHETIC_SCHEMA),
            is_empty=True,
            source_paths=source_paths,
            warnings=warnings,
        )

    combined = pd.concat(all_frames, ignore_index=True)
    combined, align_warnings = _align_to_schema(combined, SYNTHETIC_SCHEMA)
    warnings.extend(align_warnings)

    return LoaderResult(
        df=combined,
        is_empty=len(combined) == 0,
        source_paths=source_paths,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_task_map(tasks_path: Path, warnings: list[str]) -> pd.Series | None:
    """Return a tasks_index -> task_name Series, or None if unavailable."""
    tasks_df = safe_read_parquet(tasks_path)
    if tasks_df is None or tasks_df.empty:
        return None
    if "tasks_index" not in tasks_df.columns or "task" not in tasks_df.columns:
        warnings.append(
            f"{tasks_path.parent.name}/tasks.parquet: missing 'tasks_index' or 'task' column"
        )
        return None
    return tasks_df.set_index("tasks_index")["task"]


def _build_frame(
    ep_df: pd.DataFrame,
    task_series: pd.Series | None,
    warnings: list[str],
) -> pd.DataFrame:
    """Build a canonical per-episode DataFrame from episodes.parquet."""
    out: dict = {}

    out["episode_index"] = ep_df.get("episode_index", pd.Series(range(len(ep_df))))
    out["length"] = ep_df.get("length", pd.Series([pd.NA] * len(ep_df)))
    out["source"] = ep_df.get("source", pd.Series(["unknown"] * len(ep_df)))

    if task_series is not None and "tasks_index" in ep_df.columns:
        out["task"] = ep_df["tasks_index"].map(task_series)
    elif "task" in ep_df.columns:
        out["task"] = ep_df["task"]
    else:
        out["task"] = pd.Series([pd.NA] * len(ep_df))
        warnings.append(
            "episodes.parquet: no 'task' or 'tasks_index' column — task set to NA"
        )

    return pd.DataFrame(out)
