"""system_metrics — GPU/CPU sample loader for the Performance tab.

Reads ``outputs/system_metrics/<run_id>/gpu_metrics.parquet`` files emitted by
``scripts/_gpu_monitor.py`` while training stages run. Concatenates all
samples across runs into a single DataFrame so the Performance tab can plot
utilization, memory pressure, temperature, and power draw over time.

Also derives ``steps_per_sec`` from the existing training_logs loader output
when present (cross-loader compose — done in the tab, not here).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from ._base import LoaderResult, empty_df, _align_to_schema, safe_read_parquet

SYSTEM_METRICS_SCHEMA: dict[str, str] = {
    "ts": "datetime64[ns, UTC]",
    "elapsed_s": "Float64",
    "stage": "string",
    "run_id": "string",
    "gpu_index": "Int64",
    "utilization_pct": "Float64",
    "memory_used_mb": "Float64",
    "memory_total_mb": "Float64",
    "memory_pct": "Float64",
    "temperature_c": "Float64",
    "power_draw_w": "Float64",
}


def load_system_metrics(
    workspace_root: Path, *, session_id: str | None = None
) -> LoaderResult:
    """Load every ``outputs/system_metrics/*/gpu_metrics.parquet`` shard.

    Also accepts the nested layout
    ``outputs/<run_id>/system_metrics/gpu_metrics.parquet`` (when the
    full-pipeline runner writes alongside its other run artefacts).
    """
    workspace_root = Path(workspace_root)
    warnings: list[str] = []
    source_paths: list[Path] = []
    frames: list[pd.DataFrame] = []

    outputs_root = workspace_root / "outputs"
    if not outputs_root.exists():
        return LoaderResult(
            df=empty_df(list(SYSTEM_METRICS_SCHEMA.keys()), SYSTEM_METRICS_SCHEMA),
            is_empty=True,
            source_paths=[],
            warnings=warnings,
        )

    candidates = []
    candidates.extend(outputs_root.glob("system_metrics/*/gpu_metrics.parquet"))
    candidates.extend(outputs_root.glob("*/system_metrics/gpu_metrics.parquet"))
    candidates.extend(outputs_root.glob("*/system_metrics/*/gpu_metrics.parquet"))

    seen: set[Path] = set()
    for p in sorted(candidates):
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        df = safe_read_parquet(p)
        if df is None:
            warnings.append(f"{p}: unreadable parquet")
            continue
        source_paths.append(p)
        # Backfill run_id from the parent dir if the parquet didn't carry one.
        if "run_id" not in df.columns or df["run_id"].isna().any():
            inferred = p.parent.name
            df["run_id"] = df.get("run_id", inferred).fillna(inferred)
        frames.append(df)

    if not frames:
        return LoaderResult(
            df=empty_df(list(SYSTEM_METRICS_SCHEMA.keys()), SYSTEM_METRICS_SCHEMA),
            is_empty=True,
            source_paths=source_paths,
            warnings=warnings,
        )

    df = pd.concat(frames, ignore_index=True)
    df, align_warnings = _align_to_schema(df, SYSTEM_METRICS_SCHEMA)
    warnings.extend(align_warnings)
    return LoaderResult(
        df=df,
        is_empty=len(df) == 0,
        source_paths=source_paths,
        warnings=warnings,
    )
