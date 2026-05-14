"""parquet_dataset.py — LeRobot v3 dataset summary loader.

Reads ``datasets/<task>/<repo_id>/meta/`` directory tree.  Uses pyarrow
directly; falls back gracefully when lerobot is not installed.

Schema (DATASET_SUMMARY_SCHEMA)
--------------------------------
repo_id        : string   — unique dataset identifier
n_episodes     : Int64    — number of episodes in meta/episodes.parquet
n_frames       : Int64    — sum of episode lengths (rows across all data/)
fps            : Int64    — from meta/info.json
source_real    : Int64    — count of episodes with source=="real"
source_sim_dr  : Int64    — count of episodes with source=="sim_dr"
source_mimicgen: Int64    — count of episodes with source=="mimicgen"
total_size_mb  : Float64  — total disk size of the dataset directory
mtime          : datetime64[ns, UTC] — most-recent file modification time
"""

from __future__ import annotations

import json
import os
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

DATASET_SUMMARY_SCHEMA: dict[str, str] = {
    "repo_id": "string",
    "n_episodes": "Int64",
    "n_frames": "Int64",
    "fps": "Int64",
    "source_real": "Int64",
    "source_sim_dr": "Int64",
    "source_mimicgen": "Int64",
    "total_size_mb": "Float64",
    "mtime": "datetime64[ns, UTC]",
}


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_parquet_dataset(
    workspace_root: Path, *, session_id: str | None = None
) -> LoaderResult:
    """Load LeRobot v3 dataset summaries from ``datasets/``.

    Scans for any directory two levels under ``datasets/`` that contains a
    ``meta/`` sub-directory (i.e. ``datasets/<task>/<repo_id>/meta/``).

    Returns
    -------
    LoaderResult
        df is a summary DataFrame with one row per dataset found.
        is_empty=True when no datasets are present.
    """
    workspace_root = Path(workspace_root)
    datasets_dir = workspace_root / "datasets"
    warnings: list[str] = []
    source_paths: list[Path] = []
    rows: list[dict] = []

    # Find candidate dataset roots: datasets/**/**/meta/
    # Pattern: datasets/<anything>/<repo_id>/meta/episodes.parquet
    candidate_metas = (
        list(datasets_dir.glob("*/*/meta/")) if datasets_dir.exists() else []
    )
    # Also check datasets/<repo_id>/meta/ (single-level)
    candidate_metas += (
        list(datasets_dir.glob("*/meta/")) if datasets_dir.exists() else []
    )

    # Deduplicate
    seen: set[Path] = set()
    unique_metas: list[Path] = []
    for m in candidate_metas:
        m_resolved = m.resolve()
        if m_resolved not in seen:
            seen.add(m_resolved)
            unique_metas.append(m)

    for meta_dir in unique_metas:
        dataset_root = meta_dir.parent
        repo_id = dataset_root.name
        row = _summarise_dataset(dataset_root, repo_id, warnings, source_paths)
        rows.append(row)

    if not rows:
        return LoaderResult(
            df=empty_df(list(DATASET_SUMMARY_SCHEMA.keys()), DATASET_SUMMARY_SCHEMA),
            is_empty=True,
            source_paths=source_paths,
            warnings=warnings,
        )

    df = pd.DataFrame(rows)
    df, align_warnings = _align_to_schema(df, DATASET_SUMMARY_SCHEMA)
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


def _summarise_dataset(
    dataset_root: Path,
    repo_id: str,
    warnings: list[str],
    source_paths: list[Path],
) -> dict:
    """Build one summary row for a single dataset directory."""
    episodes_path = dataset_root / "meta" / "episodes.parquet"
    info_path = dataset_root / "meta" / "info.json"

    n_episodes: int | None = None
    n_frames: int | None = None
    fps: int | None = None
    source_real = 0
    source_sim_dr = 0
    source_mimicgen = 0

    # --- episodes.parquet ---------------------------------------------------
    # Primary layout (LeRobot v2.x): meta/episodes.parquet (single file).
    # Fallback layout (LeRobot v3.0): meta/episodes/chunk-XXX/file-XXX.parquet
    # (sharded). Try the single-file path first; if absent, read all shards.
    ep_df = safe_read_parquet(episodes_path)
    if ep_df is None:
        ep_shards = sorted(
            (dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
        )
        if ep_shards:
            import pandas as _pd

            shard_dfs = [
                d
                for d in (safe_read_parquet(s) for s in ep_shards)
                if d is not None
            ]
            if shard_dfs:
                ep_df = _pd.concat(shard_dfs, ignore_index=True)
                source_paths.extend(ep_shards)

    if ep_df is not None:
        if (dataset_root / "meta" / "episodes.parquet").exists():
            source_paths.append(episodes_path)
        n_episodes = len(ep_df)
        # v3.0 shards expose `length` per-episode; v2.x uses the same column.
        if "length" in ep_df.columns:
            n_frames = int(ep_df["length"].sum())
        if "source" in ep_df.columns:
            vc = ep_df["source"].value_counts()
            source_real = int(vc.get("real", 0))
            source_sim_dr = int(vc.get("sim_dr", 0))
            source_mimicgen = int(vc.get("mimicgen", 0))
        else:
            warnings.append(
                f"{repo_id}: episodes parquet has no 'source' column — "
                "source counts are zero"
            )
    else:
        warnings.append(
            f"{repo_id}: no episodes parquet found "
            f"(checked meta/episodes.parquet + meta/episodes/chunk-*/*.parquet)"
        )

    # --- info.json ----------------------------------------------------------
    # info.json carries the authoritative top-line counts (`total_episodes`,
    # `total_frames`) and `fps`. Always read it — falling back to its counts
    # when the episodes parquet is absent or malformed (v3.0 shard datasets
    # may omit it from the meta dir).
    if info_path.exists():
        try:
            with info_path.open(encoding="utf-8") as fh:
                info = json.load(fh)
            fps = info.get("fps")
            if n_episodes is None and "total_episodes" in info:
                n_episodes = int(info["total_episodes"])
            if n_frames is None and "total_frames" in info:
                n_frames = int(info["total_frames"])
            source_paths.append(info_path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{repo_id}: meta/info.json unreadable: {exc}")

    # --- disk size + mtime --------------------------------------------------
    total_size_mb, mtime = _dir_stats(dataset_root)

    return {
        "repo_id": repo_id,
        "n_episodes": n_episodes,
        "n_frames": n_frames,
        "fps": fps,
        "source_real": source_real,
        "source_sim_dr": source_sim_dr,
        "source_mimicgen": source_mimicgen,
        "total_size_mb": total_size_mb,
        "mtime": mtime,
    }


def _dir_stats(path: Path) -> tuple[float | None, str | None]:
    """Return (total_size_mb, mtime_iso) for a directory tree."""
    try:
        total_bytes = 0
        latest_mtime = 0.0
        for root, _dirs, files in os.walk(path):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    stat = fpath.stat()
                    total_bytes += stat.st_size
                    if stat.st_mtime > latest_mtime:
                        latest_mtime = stat.st_mtime
                except OSError:
                    pass
        import datetime

        mtime_dt = (
            datetime.datetime.fromtimestamp(
                latest_mtime, tz=datetime.timezone.utc
            ).isoformat()
            if latest_mtime > 0
            else None
        )
        return total_bytes / (1024 * 1024), mtime_dt
    except Exception:  # noqa: BLE001
        return None, None
