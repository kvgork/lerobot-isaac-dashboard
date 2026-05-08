"""autoresearch.py — Autoresearch HP search history loader.

Reads ``.agent-state/<session>/autoresearch/<slug>/`` directories.

Each slug directory may contain:
- ``history.jsonl`` — one JSON record per completed trial
- ``program.md``    — raw program spec (returned as dict with key "raw_md")
- ``best_config.yaml`` — best hyperparameter config so far
- ``plateau_tracker.json`` — plateau detector state

Returns dict[str, DataFrame | dict] keyed by:
  "history"  : DataFrame (HISTORY_SCHEMA)
  "program"  : dict (raw + parsed fields)
  "best"     : dict (best_config YAML parsed)
  "plateau"  : dict (plateau_tracker JSON parsed)

When session_id is None, scans ALL sessions under ``.agent-state/``.
When session_id is given, reads only that session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lerobot_isaac_dashboard.loaders._base import (
    LoaderResult,
    _align_to_schema,
    empty_df,
    safe_read_json,
    safe_read_jsonl,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

HISTORY_SCHEMA: dict[str, str] = {
    "session_id": "string",
    "slug": "string",
    "trial": "Int64",
    "metric_name": "string",
    "metric_value": "Float64",
    "config": "object",
    "ts": "datetime64[ns, UTC]",
    "status": "string",
}

_EMPTY_DICT: dict[str, Any] = {
    "history": empty_df(list(HISTORY_SCHEMA.keys()), HISTORY_SCHEMA),
    "program": {},
    "best": {},
    "plateau": {},
}


def _empty_result(warnings: list[str]) -> LoaderResult:
    return LoaderResult(
        df={
            "history": empty_df(list(HISTORY_SCHEMA.keys()), HISTORY_SCHEMA),
            "program": {},
            "best": {},
            "plateau": {},
        },
        is_empty=True,
        source_paths=[],
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_autoresearch(
    workspace_root: Path, *, session_id: str | None = None
) -> LoaderResult:
    """Load autoresearch HP search history.

    Reads ``.agent-state/<session>/autoresearch/<slug>/`` directories.

    Parameters
    ----------
    workspace_root:
        Workspace root directory.
    session_id:
        If given, restrict to a single session.  If None, scan all sessions.

    Returns
    -------
    LoaderResult
        df is a dict with keys:
          "history"  — DataFrame with one row per trial
          "program"  — dict of the last program spec found
          "best"     — dict of the last best_config found
          "plateau"  — dict of the last plateau_tracker found
    """
    workspace_root = Path(workspace_root)
    agent_state_dir = workspace_root / ".agent-state"
    warnings: list[str] = []
    source_paths: list[Path] = []

    if not agent_state_dir.exists():
        return _empty_result(warnings)

    # Collect session directories to scan
    if session_id is not None:
        session_dirs = [agent_state_dir / session_id]
    else:
        session_dirs = [d for d in sorted(agent_state_dir.iterdir()) if d.is_dir()]

    all_history_rows: list[dict] = []
    last_program: dict = {}
    last_best: dict = {}
    last_plateau: dict = {}

    for sess_dir in session_dirs:
        ar_dir = sess_dir / "autoresearch"
        if not ar_dir.exists():
            continue
        for slug_dir in sorted(ar_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            sess_name = sess_dir.name

            # history.jsonl
            history_path = slug_dir / "history.jsonl"
            hist_df = safe_read_jsonl(history_path)
            if hist_df is not None and not hist_df.empty:
                source_paths.append(history_path)
                hist_df["session_id"] = sess_name
                hist_df["slug"] = slug
                # Normalise column names
                if "trial_index" in hist_df.columns and "trial" not in hist_df.columns:
                    hist_df = hist_df.rename(columns={"trial_index": "trial"})
                all_history_rows.append(hist_df)

            # program.md
            prog_path = slug_dir / "program.md"
            if prog_path.exists():
                source_paths.append(prog_path)
                try:
                    last_program = {
                        "raw_md": prog_path.read_text(encoding="utf-8"),
                        "session_id": sess_name,
                        "slug": slug,
                    }
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"{sess_name}/{slug}/program.md: read error — {exc}"
                    )

            # best_config.yaml
            best_path = slug_dir / "best_config.yaml"
            if best_path.exists():
                source_paths.append(best_path)
                best_data = _read_yaml_safe(best_path, warnings)
                if best_data is not None:
                    last_best = {**best_data, "session_id": sess_name, "slug": slug}

            # plateau_tracker.json
            plateau_path = slug_dir / "plateau_tracker.json"
            data = safe_read_json(plateau_path)
            if data is not None:
                source_paths.append(plateau_path)
                last_plateau = {**data, "session_id": sess_name, "slug": slug}

    if not all_history_rows and not last_program and not last_best and not last_plateau:
        return _empty_result(warnings)

    if all_history_rows:
        history_df = pd.concat(all_history_rows, ignore_index=True)
        history_df, align_warnings = _align_to_schema(history_df, HISTORY_SCHEMA)
        warnings.extend(align_warnings)
    else:
        history_df = empty_df(list(HISTORY_SCHEMA.keys()), HISTORY_SCHEMA)

    return LoaderResult(
        df={
            "history": history_df,
            "program": last_program,
            "best": last_best,
            "plateau": last_plateau,
        },
        is_empty=len(history_df) == 0 and not last_program,
        source_paths=source_paths,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_yaml_safe(path: Path, warnings: list[str]) -> dict | None:
    """Read a YAML file; return None and add a warning on failure."""
    try:
        import yaml

        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {"value": data}
    except ImportError:
        warnings.append("pyyaml not installed — best_config.yaml not parsed")
        return None
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{path.name}: YAML parse error — {exc}")
        return None
