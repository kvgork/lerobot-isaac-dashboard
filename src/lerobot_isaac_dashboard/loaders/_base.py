"""_base.py — Shared base types and helpers for all metric loaders.

LoaderResult is the canonical return type for every loader function.
Helpers provide safe I/O that never raises on missing/malformed files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LoaderResult:
    """Return type for every load_* function.

    Attributes
    ----------
    df:
        The loaded data as a pandas DataFrame, or a dict of DataFrames for
        hierarchical sources (autoresearch, curriculum).
    is_empty:
        True when no source files were found or all files were unreadable.
    source_paths:
        Files that were successfully read (for provenance / staleness display).
    warnings:
        Non-fatal issues encountered during loading (partial parses, missing
        columns, etc.). The UI can display these as banners.
    """

    df: pd.DataFrame | dict[str, pd.DataFrame]
    is_empty: bool
    source_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def empty_df(columns: list[str], dtypes: dict[str, str]) -> pd.DataFrame:
    """Return an empty DataFrame with the given columns and dtypes.

    Parameters
    ----------
    columns:
        Ordered list of column names. All columns in dtypes that are not in
        columns are silently ignored; columns not in dtypes get object dtype.
    dtypes:
        Mapping from column name to pandas-compatible dtype string.
        Supports: "string", "Int64", "Float64", "float64", "int64",
        "bool", "object", and "datetime64[ns, UTC]".
    """
    df = pd.DataFrame(columns=columns)
    for col in columns:
        dtype = dtypes.get(col)
        if dtype is None:
            continue
        try:
            if dtype == "datetime64[ns, UTC]":
                df[col] = pd.Series(dtype="datetime64[ns, utc]")
            else:
                df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            # Leave as object if cast fails (e.g. nullable Int64 on older pandas)
            pass
    return df


def _align_to_schema(
    df: pd.DataFrame, schema: dict[str, str]
) -> tuple[pd.DataFrame, list[str]]:
    """Align df to the declared schema: add missing columns, cast dtypes.

    Parameters
    ----------
    df:
        Input DataFrame (may have wrong or extra columns).
    schema:
        Mapping column -> dtype string (same format as ``empty_df``).

    Returns
    -------
    (aligned_df, warnings)
        aligned_df has all schema columns in order; extra columns are preserved
        at the right. warnings lists any columns that could not be cast.
    """
    warnings: list[str] = []
    for col, dtype in schema.items():
        if col not in df.columns:
            df[col] = pd.NA
            warnings.append(f"Column '{col}' missing — filled with NA")

    # Cast dtypes
    for col, dtype in schema.items():
        try:
            if dtype == "datetime64[ns, UTC]":
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            elif col in df.columns:
                df[col] = df[col].astype(dtype)
        except (TypeError, ValueError, AttributeError) as exc:
            warnings.append(f"Could not cast column '{col}' to {dtype}: {exc}")

    # Reorder: schema columns first, extras after
    schema_cols = list(schema.keys())
    extra_cols = [c for c in df.columns if c not in schema_cols]
    df = df[schema_cols + extra_cols]
    return df, warnings


# ---------------------------------------------------------------------------
# Safe I/O helpers
# ---------------------------------------------------------------------------


def safe_read_parquet(path: Path) -> pd.DataFrame | None:
    """Read a Parquet file; return None on any error.

    Never raises. Returns None if the file is absent, empty, or corrupt.
    """
    try:
        if not path.exists():
            return None
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("safe_read_parquet(%s) failed: %s", path, exc)
        return None


def safe_read_jsonl(path: Path) -> pd.DataFrame | None:
    """Read a newline-delimited JSON file; return None on any error.

    Skips malformed lines, logging them at DEBUG level. Returns None if the
    file is absent or entirely unreadable.
    """
    try:
        if not path.exists():
            return None
        records: list[dict] = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.debug("safe_read_jsonl(%s) line %d: %s", path, lineno, exc)
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
    except Exception as exc:  # noqa: BLE001
        logger.debug("safe_read_jsonl(%s) failed: %s", path, exc)
        return None


def safe_read_json(path: Path) -> dict | None:
    """Read a JSON file; return None on any error."""
    try:
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.debug("safe_read_json(%s) failed: %s", path, exc)
        return None


def glob_runs(root: Path, pattern: str) -> list[Path]:
    """Glob for paths matching pattern under root; return [] if root absent.

    Never raises. Returns an empty list if root does not exist or glob fails.
    """
    try:
        if not root.exists():
            return []
        return sorted(root.glob(pattern))
    except Exception as exc:  # noqa: BLE001
        logger.debug("glob_runs(%s, %r) failed: %s", root, pattern, exc)
        return []
