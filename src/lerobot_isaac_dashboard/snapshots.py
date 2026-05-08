"""snapshots.py — Snapshot save/load/list for lerobot-isaac-dashboard.

A snapshot captures the full loader state (raw DataFrames + metadata dicts)
for a given workspace at a point in time.  Snapshots can later be reloaded
and compared side-by-side via compare.py.

Snapshot directory layout::

    outputs/snapshots/<snapshot_id>/
    ├── meta.json
    └── loaders/
        ├── paths.json
        ├── parquet_dataset.parquet
        ├── eval_results.parquet
        ├── checkpoints.parquet
        ├── training_logs.parquet
        ├── autoresearch__history.parquet
        ├── autoresearch__program.json
        ├── autoresearch__best.json
        ├── autoresearch__plateau.json
        ├── events.parquet
        ├── curriculum__current.json
        ├── curriculum__history.parquet
        └── synthetic.parquet

CLI::

    python -m lerobot_isaac_dashboard.snapshots --workspace=PATH [--label=LABEL]
    python -m lerobot_isaac_dashboard.snapshots --workspace=PATH list
    lerobot-isaac-snapshot --workspace=PATH [--label=LABEL]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from lerobot_isaac_dashboard.loaders._base import LoaderResult
from lerobot_isaac_dashboard.version import __version__

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version — bump when the snapshot format changes in a breaking way
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Loader specs — declares how each loader's result is stored/restored
#
# Format: slug -> {"type": "df" | "dict", "dict_members": {...}}
#   "df"   : result.df is a plain DataFrame  → saved as <slug>.parquet
#   "dict" : result.df is a dict             → members saved with __ separator
#             dict_members: {key: "df" | "json"}
# ---------------------------------------------------------------------------

_LOADER_SPECS: dict[str, dict[str, Any]] = {
    "parquet_dataset": {"type": "df"},
    "eval_results": {"type": "df"},
    "checkpoints": {"type": "df"},
    "training_logs": {"type": "df"},
    "events": {"type": "df"},
    "synthetic": {"type": "df"},
    "autoresearch": {
        "type": "dict",
        "dict_members": {
            "history": "df",
            "program": "json",
            "best": "json",
            "plateau": "json",
        },
    },
    "curriculum": {
        "type": "dict",
        "dict_members": {
            "current": "json",
            "history": "df",
        },
    },
}

# paths loader returns WorkspacePaths (not a LoaderResult with a df) — handle separately
_PATHS_SLUG = "paths"


# ---------------------------------------------------------------------------
# SnapshotMeta dataclass
# ---------------------------------------------------------------------------


@dataclass
class SnapshotMeta:
    """Metadata header stored in meta.json for each snapshot.

    Attributes
    ----------
    snapshot_id:
        Unique identifier, e.g. ``"2026-05-08T072115-baseline"``.
    label:
        Human-readable label (may be None).
    workspace_root:
        Absolute path to the workspace that was snapshotted.
    session_id:
        Optional session ID that scoped the loaders.
    git_sha:
        Short git commit SHA at snapshot time (None if not in a git repo).
    dashboard_version:
        ``__version__`` of lerobot-isaac-dashboard at snapshot time.
    ts:
        UTC datetime when the snapshot was created.
    loader_slugs:
        List of loader slugs captured (subset of _LOADER_SPECS keys).
    schema_version:
        Integer format version.  ``load_snapshot`` rejects unknown versions.
    """

    snapshot_id: str
    label: str | None
    workspace_root: Path
    session_id: str | None
    git_sha: str | None
    dashboard_version: str
    ts: datetime
    loader_slugs: list[str]
    schema_version: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "snapshot_id": self.snapshot_id,
            "label": self.label,
            "workspace_root": str(self.workspace_root),
            "session_id": self.session_id,
            "git_sha": self.git_sha,
            "dashboard_version": self.dashboard_version,
            "ts": self.ts.isoformat(),
            "loader_slugs": self.loader_slugs,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SnapshotMeta:
        """Deserialize from a dict (e.g. from meta.json)."""
        return cls(
            snapshot_id=d["snapshot_id"],
            label=d.get("label"),
            workspace_root=Path(d["workspace_root"]),
            session_id=d.get("session_id"),
            git_sha=d.get("git_sha"),
            dashboard_version=d.get("dashboard_version", "unknown"),
            ts=datetime.fromisoformat(d["ts"]),
            loader_slugs=d.get("loader_slugs", []),
            schema_version=d.get("schema_version", 1),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_git_sha(workspace_root: Path) -> str | None:
    """Return the short git SHA for workspace_root, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return None


def _make_snapshot_id(label: str | None, ts: datetime) -> str:
    """Build a snapshot ID from timestamp and optional label."""
    ts_str = ts.strftime("%Y-%m-%dT%H%M%S")
    slug = label.strip().replace(" ", "-") if label else "unlabeled"
    return f"{ts_str}-{slug}"


def _write_df(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to a Parquet file via pyarrow (lossless dtypes)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write parquet %s: %s — writing empty file", path, exc)
        # Write an empty schema file so load_snapshot can still reconstruct
        pq.write_table(pa.table({}), path)


def _read_df(path: Path) -> pd.DataFrame:
    """Read a Parquet file; return empty DataFrame on any error."""
    import pyarrow.parquet as pq

    try:
        if not path.exists():
            return pd.DataFrame()
        table = pq.read_table(path)
        return table.to_pandas()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read parquet %s: %s", path, exc)
        return pd.DataFrame()


def _write_json(path: Path, obj: Any) -> None:
    """Write an arbitrary object as JSON (handles non-serialisable types gracefully)."""
    try:
        path.write_text(json.dumps(obj, default=str, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write json %s: %s — writing null", path, exc)
        path.write_text("null", encoding="utf-8")


def _read_json(path: Path) -> Any:
    """Read a JSON file; return None on any error."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read json %s: %s", path, exc)
        return None


def _save_loader_result(loaders_dir: Path, slug: str, result: LoaderResult) -> None:
    """Persist one LoaderResult to the loaders/ directory."""
    spec = _LOADER_SPECS.get(slug)
    if spec is None:
        logger.debug("No spec for slug %r — skipping", slug)
        return

    if spec["type"] == "df":
        df = result.df if isinstance(result.df, pd.DataFrame) else pd.DataFrame()
        _write_df(loaders_dir / f"{slug}.parquet", df)

    elif spec["type"] == "dict":
        df_dict = result.df if isinstance(result.df, dict) else {}
        for member_key, member_type in spec["dict_members"].items():
            member_val = df_dict.get(member_key)
            if member_type == "df":
                df_val = (
                    member_val
                    if isinstance(member_val, pd.DataFrame)
                    else pd.DataFrame()
                )
                _write_df(loaders_dir / f"{slug}__{member_key}.parquet", df_val)
            else:  # json
                _write_json(
                    loaders_dir / f"{slug}__{member_key}.json", member_val or {}
                )


def _load_loader_result(loaders_dir: Path, slug: str) -> LoaderResult:
    """Reconstruct one LoaderResult from the loaders/ directory."""
    spec = _LOADER_SPECS.get(slug)
    if spec is None:
        return LoaderResult(df=pd.DataFrame(), is_empty=True)

    if spec["type"] == "df":
        df = _read_df(loaders_dir / f"{slug}.parquet")
        return LoaderResult(df=df, is_empty=df.empty)

    # dict type
    df_dict: dict[str, Any] = {}
    for member_key, member_type in spec["dict_members"].items():
        if member_type == "df":
            df_dict[member_key] = _read_df(
                loaders_dir / f"{slug}__{member_key}.parquet"
            )
        else:  # json
            df_dict[member_key] = (
                _read_json(loaders_dir / f"{slug}__{member_key}.json") or {}
            )

    is_empty = all(
        (isinstance(v, pd.DataFrame) and v.empty) or (isinstance(v, dict) and not v)
        for v in df_dict.values()
    )
    return LoaderResult(df=df_dict, is_empty=is_empty)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_snapshot(
    workspace_root: Path,
    *,
    session_id: str | None = None,
    label: str | None = None,
    snapshot_id: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run all loaders headless and persist results as a snapshot.

    Parameters
    ----------
    workspace_root:
        Absolute path to the training workspace root.
    session_id:
        Optional session ID to scope session-aware loaders.
    label:
        Human-readable label embedded in the snapshot ID and meta.json.
    snapshot_id:
        Override the auto-generated ID.  Must be filesystem-safe.
    output_dir:
        Directory to write the snapshot into.  Defaults to
        ``<workspace_root>/outputs/snapshots/<snapshot_id>/``.

    Returns
    -------
    Path
        The snapshot directory (contains meta.json + loaders/).
    """
    workspace_root = Path(workspace_root).resolve()
    ts = datetime.now(tz=timezone.utc)

    if snapshot_id is None:
        snapshot_id = _make_snapshot_id(label, ts)

    if output_dir is None:
        output_dir = workspace_root / "outputs" / "snapshots" / snapshot_id
    output_dir = Path(output_dir).resolve()

    loaders_dir = output_dir / "loaders"
    loaders_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Run all loaders headless (reuse report.run_loaders_headless)
    # ------------------------------------------------------------------
    from lerobot_isaac_dashboard.report import run_loaders_headless

    loader_results = run_loaders_headless(workspace_root, session_id=session_id)

    # ------------------------------------------------------------------
    # 2. Persist each loader result
    # ------------------------------------------------------------------
    captured_slugs: list[str] = []
    for slug, result in loader_results.items():
        _save_loader_result(loaders_dir, slug, result)
        captured_slugs.append(slug)

    # paths loader: WorkspacePaths is not a LoaderResult — save minimal JSON
    try:
        from lerobot_isaac_dashboard.loaders.paths import load_paths

        wp = load_paths(workspace_root)
        wp_data = {k: str(v) for k, v in vars(wp).items() if isinstance(v, Path)}
        _write_json(loaders_dir / "paths.json", wp_data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not capture paths loader: %s", exc)
        _write_json(loaders_dir / "paths.json", {})

    # ------------------------------------------------------------------
    # 3. Write meta.json
    # ------------------------------------------------------------------
    meta = SnapshotMeta(
        snapshot_id=snapshot_id,
        label=label,
        workspace_root=workspace_root,
        session_id=session_id,
        git_sha=_get_git_sha(workspace_root),
        dashboard_version=__version__,
        ts=ts,
        loader_slugs=captured_slugs,
        schema_version=SCHEMA_VERSION,
    )
    (output_dir / "meta.json").write_text(
        json.dumps(meta.to_dict(), indent=2), encoding="utf-8"
    )

    logger.info("Snapshot saved: %s", output_dir)
    return output_dir


def load_snapshot(
    path_or_id: Path | str,
    workspace_root: Path | None = None,
) -> tuple[SnapshotMeta, dict[str, LoaderResult]]:
    """Load a snapshot from disk and reconstruct loader results.

    Parameters
    ----------
    path_or_id:
        Either an absolute path to the snapshot directory, or a snapshot_id
        string.  When a string is given, ``workspace_root`` must be provided
        so the function can resolve
        ``<workspace_root>/outputs/snapshots/<snapshot_id>/``.
    workspace_root:
        Required when ``path_or_id`` is a snapshot_id string.

    Returns
    -------
    (meta, loader_results)
        ``meta`` is the deserialized SnapshotMeta.
        ``loader_results`` maps slug -> LoaderResult.

    Raises
    ------
    FileNotFoundError
        When the snapshot directory or meta.json does not exist.
    ValueError
        When schema_version is unknown (future format).
    """
    snap_dir = Path(path_or_id)
    if not snap_dir.is_absolute() or not snap_dir.exists():
        # Treat as snapshot_id
        if workspace_root is None:
            raise ValueError(
                "workspace_root must be provided when path_or_id is a snapshot_id string"
            )
        snap_dir = Path(workspace_root) / "outputs" / "snapshots" / str(path_or_id)

    meta_path = snap_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Snapshot meta.json not found: {meta_path}")

    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    schema_version = meta_data.get("schema_version", 1)
    if schema_version > SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot schema_version={schema_version} is newer than supported "
            f"version={SCHEMA_VERSION}. Please upgrade lerobot-isaac-dashboard."
        )

    meta = SnapshotMeta.from_dict(meta_data)
    loaders_dir = snap_dir / "loaders"

    loader_results: dict[str, LoaderResult] = {}
    for slug in meta.loader_slugs:
        loader_results[slug] = _load_loader_result(loaders_dir, slug)

    return meta, loader_results


def list_snapshots(workspace_root: Path) -> list[SnapshotMeta]:
    """List all snapshots under ``<workspace_root>/outputs/snapshots/``.

    Returns
    -------
    list[SnapshotMeta]
        Sorted by ``ts`` descending (newest first).  Malformed snapshots
        are silently skipped.
    """
    snapshots_dir = Path(workspace_root) / "outputs" / "snapshots"
    if not snapshots_dir.exists():
        return []

    metas: list[SnapshotMeta] = []
    for meta_path in snapshots_dir.glob("*/meta.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            metas.append(SnapshotMeta.from_dict(data))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping malformed snapshot %s: %s", meta_path.parent, exc)

    metas.sort(key=lambda m: m.ts, reverse=True)
    return metas


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def cli_main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m lerobot_isaac_dashboard.snapshots`` and ``lerobot-isaac-snapshot``.

    Usage::

        lerobot-isaac-snapshot --workspace=PATH [--label=LABEL]
        lerobot-isaac-snapshot --workspace=PATH list

    Returns 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        prog="lerobot-isaac-snapshot",
        description="Save or list lerobot-isaac-dashboard workspace snapshots.",
    )
    parser.add_argument(
        "--workspace",
        metavar="PATH",
        default=".",
        help="Path to the workspace root (default: current directory).",
    )
    parser.add_argument(
        "--label",
        metavar="LABEL",
        default=None,
        help="Human-readable label for the snapshot.",
    )
    parser.add_argument(
        "--session-id",
        metavar="SID",
        default=None,
        help="Scope session-aware loaders to this session ID.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="save",
        choices=["save", "list"],
        help="'save' (default) or 'list' existing snapshots.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    workspace_root = Path(args.workspace).resolve()

    if args.command == "list":
        metas = list_snapshots(workspace_root)
        if not metas:
            print("No snapshots found.")
            return 0
        for m in metas:
            label_str = f"[{m.label}]" if m.label else ""
            print(
                f"{m.snapshot_id}  {m.ts.strftime('%Y-%m-%d %H:%M:%S UTC')}  {label_str}"
            )
        return 0

    # save
    try:
        snap_dir = save_snapshot(
            workspace_root,
            session_id=args.session_id,
            label=args.label,
        )
        print(f"Snapshot saved: {snap_dir}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Snapshot failed: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())
