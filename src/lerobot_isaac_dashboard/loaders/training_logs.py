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
    """Parse training logs from training stdout files.

    Primary layout: ``outputs/checkpoints/<arch>/<run_id>/log.txt``.

    Fallback layouts (auto-detected, no config required):
      * ``logs/<session_or_run>/<stage_or_run>-<arch>.log`` — adapter stdout
        captured by the pipeline-validation runner.
      * ``outputs/<run_prefix>/stage-<id>-<arch>/...`` and similar nested run
        directories: any ``*.log`` or ``cli.log`` directly inside the run dir
        is treated as a log.txt-equivalent.

    Returns
    -------
    LoaderResult
        df has schema TRAINING_LOG_SCHEMA, one row per (step, metric_name) pair.
        is_empty=True when no log files are found anywhere.
    """
    workspace_root = Path(workspace_root)
    warnings: list[str] = []
    source_paths: list[Path] = []
    rows: list[dict] = []

    # ----------------------------------------------------------------- primary
    checkpoints_root = workspace_root / "outputs" / "checkpoints"
    if checkpoints_root.exists():
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
                mtime = _file_mtime(log_file)
                _parse_log_file(log_file, arch, run_id, mtime, rows, warnings)

    # --------------------------------------------------------------- fallbacks
    # logs/<dir>/<stem>.log — infer arch + run_id from the filename.
    logs_root = workspace_root / "logs"
    if logs_root.exists():
        for log_file in sorted(logs_root.rglob("*.log")):
            if not log_file.is_file():
                continue
            arch, run_id = _infer_arch_run(log_file)
            source_paths.append(log_file)
            mtime = _file_mtime(log_file)
            _parse_log_file(log_file, arch, run_id, mtime, rows, warnings)

    # outputs/<prefix>/<stage>/cli.log or *.log (sheeprl + adapter side-channels)
    outputs_root = workspace_root / "outputs"
    if outputs_root.exists():
        for log_file in sorted(outputs_root.rglob("cli.log")):
            if checkpoints_root in log_file.parents:
                continue  # already covered above
            arch, run_id = _infer_arch_run(log_file)
            source_paths.append(log_file)
            mtime = _file_mtime(log_file)
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
        "step",
        "epoch",
        "it",
        "iter",
        "batch",
        "lr",
        "e",
        "E",
        "nan",
        "inf",
    }
)


def _file_mtime(p: Path) -> str | None:
    """Return ISO-8601 UTC mtime for ``p``, or None on stat error."""
    try:
        return datetime.datetime.fromtimestamp(
            p.stat().st_mtime, tz=datetime.timezone.utc
        ).isoformat()
    except OSError:
        return None


# Recognise common arch tokens regardless of dir layout.
_ARCH_TOKENS = (
    "smolvla",
    "act",
    "diffusion",
    "dreamerv3",
    "dreamer_v3",
    "le_world_model",
    "leworldmodel",
    "lewm",
)


def _infer_arch_run(p: Path) -> tuple[str, str]:
    """Heuristic: pick arch token from path; run_id from parent dir / stem.

    Used when the log file does NOT live under
    ``outputs/checkpoints/<arch>/<run_id>/log.txt``.
    """
    haystack = "/".join(p.parts).lower()
    arch = "unknown"
    for token in _ARCH_TOKENS:
        if token in haystack:
            # Normalise spelling: prefer canonical ``dreamerv3`` / ``le_world_model``.
            if token == "dreamer_v3":
                arch = "dreamerv3"
            elif token in ("leworldmodel", "lewm"):
                arch = "le_world_model"
            else:
                arch = token
            break
    run_id = p.stem if p.stem != "cli" else p.parent.name
    return arch, run_id


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
