"""checkpoints.py — Training checkpoint loader.

Reads ``outputs/checkpoints/<arch>/<run_id>/`` directories produced by
training adapter targets.

Each run directory may contain:
- ``*.pt`` or ``*.safetensors`` — model weights
- ``*.yaml`` or ``*.yml`` — sibling config snapshot
- ``log.txt`` — stdout capture (parsed by training_logs.py)

Schema (CHECKPOINT_SCHEMA)
--------------------------
arch        : string   — training architecture (smolvla, dreamerv3, ...)
run_id      : string   — run directory name
step        : Int64    — inferred from filename (e.g. step_010000.pt -> 10000)
path        : string   — absolute path to checkpoint file
size_mb     : Float64  — file size in MB
mtime       : datetime64[ns, UTC]
val_loss    : Float64  — optional, read from sibling .yaml if present
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

CHECKPOINT_SCHEMA: dict[str, str] = {
    "arch": "string",
    "run_id": "string",
    "step": "Int64",
    "path": "string",
    "size_mb": "Float64",
    "mtime": "datetime64[ns, UTC]",
    "val_loss": "Float64",
}

_STEP_RE = re.compile(r"step[_\-]?(\d+)", re.IGNORECASE)
_WEIGHT_PATTERNS = ("*.pt", "*.safetensors")

# Same arch tokens as training_logs._ARCH_TOKENS; duplicated to keep loaders
# decoupled (no cross-imports between loader modules).
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


def _infer_arch_run_from_path(p: Path, outputs_root: Path) -> tuple[str, str]:
    """Heuristic arch/run derivation for fallback layouts."""
    haystack = "/".join(p.parts).lower()
    arch = "unknown"
    for token in _ARCH_TOKENS:
        if token in haystack:
            if token == "dreamer_v3":
                arch = "dreamerv3"
            elif token in ("leworldmodel", "lewm"):
                arch = "le_world_model"
            else:
                arch = token
            break
    # Run id = first directory under outputs/<run_prefix>/<stage>/ where the
    # ckpt lives. If the ckpt is at outputs/X/Y/.../file.pt, take Y.
    try:
        rel = p.resolve().relative_to(outputs_root.resolve())
        parts = rel.parts
        if len(parts) >= 2:
            run_id = parts[1] if parts[0] in ("checkpoints", "outputs") else parts[0]
            # For lerobot layout outputs/<prefix>/<stage>/checkpoints/NNN/...
            # prefer the <stage> dir as run_id.
            if len(parts) >= 3 and parts[0] not in ("checkpoints",):
                run_id = parts[1]
        else:
            run_id = parts[0] if parts else "unknown"
    except ValueError:
        run_id = p.parent.name
    return arch, run_id


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_checkpoints(
    workspace_root: Path, *, session_id: str | None = None
) -> LoaderResult:
    """Load checkpoint metadata from training output trees.

    Primary layout: ``outputs/checkpoints/<arch>/<run_id>/*.{pt,safetensors}``.

    Fallback: any ``outputs/<run_prefix>/<stage_dir>/`` tree is scanned
    recursively for ``*.pt`` and ``*.safetensors`` files (this picks up
    lerobot's nested ``checkpoints/NNNNNN/pretrained_model/model.safetensors``
    layout produced by ``lerobot-train`` 0.5+, plus the flat
    ``lewm_minimal_last.pt`` produced by the in-process LeWM trainer).

    Returns
    -------
    LoaderResult
        df has schema CHECKPOINT_SCHEMA, one row per checkpoint file found.
        is_empty=True when no checkpoint files exist anywhere.
    """
    workspace_root = Path(workspace_root)
    outputs_root = workspace_root / "outputs"
    checkpoints_root = outputs_root / "checkpoints"
    warnings: list[str] = []
    source_paths: list[Path] = []
    rows: list[dict] = []

    # Track files we've already accounted for so the fallback doesn't
    # double-count anything under outputs/checkpoints/.
    seen: set[Path] = set()

    # ---- primary layout: outputs/checkpoints/<arch>/<run_id>/*.{pt,safetensors}
    if checkpoints_root.exists():
        for arch_dir in sorted(checkpoints_root.iterdir()):
            if not arch_dir.is_dir():
                continue
            arch = arch_dir.name
            for run_dir in sorted(arch_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name
                val_loss = _read_val_loss(run_dir, warnings)
                # Recurse — lerobot nests the model under
                # `checkpoints/NNNNNN/pretrained_model/model.safetensors`.
                for pattern in _WEIGHT_PATTERNS:
                    for ckpt_file in sorted(run_dir.rglob(pattern)):
                        if ckpt_file.resolve() in seen:
                            continue
                        seen.add(ckpt_file.resolve())
                        row = _checkpoint_row(
                            arch=arch,
                            run_id=run_id,
                            ckpt_file=ckpt_file,
                            val_loss=val_loss,
                            warnings=warnings,
                        )
                        rows.append(row)
                        source_paths.append(ckpt_file)

    # ---- fallback: any other outputs/<...>/ subtree (skip checkpoints_root)
    if outputs_root.exists():
        for pattern in _WEIGHT_PATTERNS:
            for ckpt_file in sorted(outputs_root.rglob(pattern)):
                if not ckpt_file.is_file():
                    continue
                resolved = ckpt_file.resolve()
                if resolved in seen:
                    continue
                if checkpoints_root in ckpt_file.parents:
                    continue
                # Skip auto-generated snapshot copies — those mirror canonical
                # data and would double-count.
                if "snapshots" in ckpt_file.parts:
                    continue
                seen.add(resolved)
                arch, run_id = _infer_arch_run_from_path(ckpt_file, outputs_root)
                row = _checkpoint_row(
                    arch=arch,
                    run_id=run_id,
                    ckpt_file=ckpt_file,
                    val_loss=None,
                    warnings=warnings,
                )
                rows.append(row)
                source_paths.append(ckpt_file)

    if not rows:
        return LoaderResult(
            df=empty_df(list(CHECKPOINT_SCHEMA.keys()), CHECKPOINT_SCHEMA),
            is_empty=True,
            source_paths=source_paths,
            warnings=warnings,
        )

    df = pd.DataFrame(rows)
    df, align_warnings = _align_to_schema(df, CHECKPOINT_SCHEMA)
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


def _checkpoint_row(
    arch: str,
    run_id: str,
    ckpt_file: Path,
    val_loss: float | None,
    warnings: list[str],
) -> dict:
    """Build a single checkpoint row dict."""
    step = _infer_step(ckpt_file.name)
    if step is None:
        warnings.append(
            f"{ckpt_file.name}: could not infer step from filename — step=NA"
        )
    try:
        stat = ckpt_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.datetime.fromtimestamp(
            stat.st_mtime, tz=datetime.timezone.utc
        ).isoformat()
    except OSError:
        size_mb = None
        mtime = None

    return {
        "arch": arch,
        "run_id": run_id,
        "step": step,
        "path": str(ckpt_file),
        "size_mb": size_mb,
        "mtime": mtime,
        "val_loss": val_loss,
    }


def _infer_step(filename: str) -> int | None:
    """Extract step number from a checkpoint filename."""
    m = _STEP_RE.search(filename)
    if m:
        return int(m.group(1))
    # Fallback: any contiguous digit sequence in the stem
    stem = Path(filename).stem
    digits = re.findall(r"\d+", stem)
    if digits:
        return int(digits[-1])
    return None


def _read_val_loss(run_dir: Path, warnings: list[str]) -> float | None:
    """Try to read val_loss from a sibling YAML file in the run directory."""
    try:
        import yaml  # pyyaml is a hard dep

        for yaml_file in sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml")):
            try:
                with yaml_file.open(encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, dict):
                    for key in (
                        "val_loss",
                        "validation_loss",
                        "recon_loss",
                        "pred_loss",
                    ):
                        if key in data:
                            return float(data[key])
            except Exception:
                continue
    except ImportError:
        warnings.append("pyyaml not installed — val_loss not extracted from YAML")
    return None
