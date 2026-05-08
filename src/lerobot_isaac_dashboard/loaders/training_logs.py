"""training_logs.py — Training log parser.

Reads ``outputs/checkpoints/<arch>/<run_id>/log.txt`` (stdout capture) and
parses metric lines using the same regex as metric_extractor.py:

    (\\w+)[=:\\s]+([0-9.eE+-]+)

This mirrors the canonical format emitted by
``lerobot_isaac_adapters.metric_extractor.emit()``.

Schema (TRAINING_LOG_SCHEMA)
-----------------------------
arch        : string   — training architecture directory name
run_id      : string   — run directory name
step        : Int64    — extracted from ``# step=<N>`` comment, else line order
metric_name : string   — matched word before ``=`` / ``:`` / whitespace
value       : Float64  — matched numeric value
ts          : datetime64[ns, UTC] — file mtime (proxy; no per-line timestamps in log)
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import pandas as pd

from lerobot_isaac_dashboard.loaders._base import (
    LoaderResult,
    _align_to_schema,
    empty_df,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TRAINING_LOG_SCHEMA: dict[str, str] = {
    "arch": "string",
    "run_id": "string",
    "step": "Int64",
    "metric_name": "string",
    "value": "Float64",
    "ts": "datetime64[ns, UTC]",
}

# Regex identical to the one documented in metric_extractor.py
_METRIC_RE = re.compile(r"(\w+)[=:\s]+([0-9.eE+\-]+)")
# Optional step comment: ``# step=<N>``
_STEP_COMMENT_RE = re.compile(r"#\s*step=(\d+)")


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_training_logs(
    workspace_root: Path, *, session_id: str | None = None
) -> LoaderResult:
    """Parse training logs from ``outputs/checkpoints/<arch>/<run_id>/log.txt``.

    Returns
    -------
    LoaderResult
        df has schema TRAINING_LOG_SCHEMA, one row per (step, metric_name) pair.
        is_empty=True when no log files are found.
    """
    workspace_root = Path(workspace_root)
    checkpoints_root = workspace_root / "outputs" / "checkpoints"
    warnings: list[str] = []
    source_paths: list[Path] = []
    rows: list[dict] = []

    if not checkpoints_root.exists():
        return LoaderResult(
            df=empty_df(list(TRAINING_LOG_SCHEMA.keys()), TRAINING_LOG_SCHEMA),
            is_empty=True,
            source_paths=[],
            warnings=warnings,
        )

    for arch_dir in sorted(checkpoints_root.iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        for run_dir in sorted(arch_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            log_file = run_dir / "log.txt"
            if not log_file.exists():
                continue
            source_paths.append(log_file)
            try:
                mtime = datetime.datetime.fromtimestamp(
                    log_file.stat().st_mtime, tz=datetime.timezone.utc
                ).isoformat()
            except OSError:
                mtime = None
            _parse_log_file(log_file, arch, run_id, mtime, rows, warnings)

    if not rows:
        return LoaderResult(
            df=empty_df(list(TRAINING_LOG_SCHEMA.keys()), TRAINING_LOG_SCHEMA),
            is_empty=True,
            source_paths=source_paths,
            warnings=warnings,
        )

    df = pd.DataFrame(rows)
    df, align_warnings = _align_to_schema(df, TRAINING_LOG_SCHEMA)
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

# Noise tokens that appear in log lines but are not real metrics
_NOISE_TOKENS = frozenset(
    {
        "step", "epoch", "it", "iter", "batch", "lr", "e", "E",
        "nan", "inf",
    }
)


def _parse_log_file(
    log_file: Path,
    arch: str,
    run_id: str,
    mtime: str | None,
    rows: list[dict],
    warnings: list[str],
) -> None:
    """Parse all metric lines from a single log.txt file."""
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{arch}/{run_id}/log.txt: read error — {exc}")
        return

    line_step = 0  # fallback counter when no step comment found
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Extract step from comment if present
        step_match = _STEP_COMMENT_RE.search(line)
        if step_match:
            line_step = int(step_match.group(1))
        else:
            line_step += 1

        for m in _METRIC_RE.finditer(line):
            name = m.group(1)
            raw_val = m.group(2)
            # Skip noise tokens and pure-integer step markers
            if name in _NOISE_TOKENS:
                continue
            try:
                value = float(raw_val)
            except ValueError:
                continue
            rows.append(
                {
                    "arch": arch,
                    "run_id": run_id,
                    "step": line_step,
                    "metric_name": name,
                    "value": value,
                    "ts": mtime,
                }
            )
