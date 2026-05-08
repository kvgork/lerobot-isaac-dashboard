# lerobot-isaac-dashboard — Public API Reference

This document covers the full public API: loaders, tabs, report, snapshots, and compare.
All symbols are importable without any GPU or heavy ML dependencies.

---

## `lerobot_isaac_dashboard.loaders`

### `LoaderResult`

Return type for every `load_*` function.

```python
@dataclass
class LoaderResult:
    df: pd.DataFrame | dict[str, pd.DataFrame]
    is_empty: bool
    source_paths: list[Path]        # files successfully read (provenance)
    warnings: list[str]             # non-fatal issues (missing columns, schema drift)
```

`df` is a plain `DataFrame` for most loaders. For hierarchical loaders (`autoresearch`,
`curriculum`) it is a `dict[str, DataFrame | dict]`.

`is_empty` is `True` when no source files were found or all were unreadable. Tabs must
check this flag and show an empty-state placeholder rather than crashing.

---

### `WorkspacePaths`

Resolved workspace path bundle returned by `load_paths`.

```python
@dataclass
class WorkspacePaths:
    workspace_root: Path
    datasets_dir: Path          # workspace_root / "datasets"
    outputs_dir: Path           # workspace_root / "outputs"
    agent_state_dir: Path       # workspace_root / ".agent-state"
    snapshots_dir: Path         # outputs_dir / "snapshots"
    reports_dir: Path           # outputs_dir / "reports"
```

---

### `load_parquet_dataset`

```python
def load_parquet_dataset(
    workspace_root: Path,
    *,
    session_id: str | None = None,
) -> LoaderResult
```

Scans `datasets/` for LeRobot v3 Parquet layouts. Returns a DataFrame with columns:

| Column | Dtype | Description |
|--------|-------|-------------|
| `repo_id` | string | Dataset identifier |
| `task` | string | Task subdirectory name |
| `n_episodes` | Int64 | Episode count |
| `fps` | Float64 | Recording frame rate |
| `source` | string | `"real"`, `"sim_dr"`, or `"mimicgen"` |
| `total_steps` | Int64 | Total step count across episodes |
| `size_mb` | Float64 | Dataset size on disk |

---

### `load_eval_results`

```python
def load_eval_results(
    workspace_root: Path,
    *,
    session_id: str | None = None,
) -> LoaderResult
```

Reads `outputs/eval/*.json`. Returns a DataFrame with columns:

| Column | Dtype | Description |
|--------|-------|-------------|
| `run_id` | string | Eval run identifier |
| `ts` | datetime64[ns, UTC] | Evaluation timestamp |
| `arch` | string | Policy architecture |
| `pc_success` | Float64 | Pick-and-place success rate [0, 1] |
| `mean_ep_len` | Float64 | Mean episode length (steps) |
| `n_episodes` | Int64 | Number of eval episodes |

---

### `load_checkpoints`

```python
def load_checkpoints(
    workspace_root: Path,
    *,
    session_id: str | None = None,
) -> LoaderResult
```

Inventories `outputs/checkpoints/<arch>/<run_id>/`. Returns a DataFrame with columns:

| Column | Dtype | Description |
|--------|-------|-------------|
| `arch` | string | Target architecture |
| `run_id` | string | Run directory name |
| `checkpoint_step` | Int64 | Training step of checkpoint |
| `ts` | datetime64[ns, UTC] | File mtime |
| `size_mb` | Float64 | Checkpoint directory size |
| `path` | string | Absolute path to checkpoint dir |

---

### `load_training_logs`

```python
def load_training_logs(
    workspace_root: Path,
    *,
    session_id: str | None = None,
) -> LoaderResult
```

Reads `outputs/checkpoints/<arch>/<run_id>/log.txt` (one `metric=value` line per step).
Returns a DataFrame with columns:

| Column | Dtype | Description |
|--------|-------|-------------|
| `run_id` | string | Run identifier |
| `arch` | string | Architecture |
| `step` | Int64 | Training step |
| `metric_name` | string | Metric name (e.g. `"loss"`, `"recon_loss"`) |
| `value` | Float64 | Metric value |
| `ts` | datetime64[ns, UTC] | Log line timestamp (if present) |

---

### `load_autoresearch`

```python
def load_autoresearch(
    workspace_root: Path,
    *,
    session_id: str | None = None,
) -> LoaderResult
```

Reads `.agent-state/<session>/autoresearch/<slug>/history.jsonl`.

Returns `LoaderResult` where `df` is a dict:

| Key | Type | Description |
|-----|------|-------------|
| `"history"` | DataFrame | HP search trial history (step, metric, config, …) |
| `"program"` | dict | Parsed `program.md` metadata |
| `"best"` | dict | Best trial config + metric value |
| `"plateau"` | dict | Plateau detector state (n_no_improve, limit) |

---

### `load_events`

```python
def load_events(
    workspace_root: Path,
    *,
    session_id: str | None = None,
) -> LoaderResult
```

Reads `.agent-state/<session>/events.jsonl`. Returns a DataFrame with columns:

| Column | Dtype | Description |
|--------|-------|-------------|
| `ts` | datetime64[ns, UTC] | Event timestamp |
| `event_type` | string | Agent event type |
| `agent` | string | Agent or worker name |
| `message` | string | Event message or summary |
| `session_id` | string | Session that produced the event |

---

### `load_curriculum`

```python
def load_curriculum(
    workspace_root: Path,
    *,
    session_id: str | None = None,
) -> LoaderResult
```

Reads `outputs/curriculum_stage.json` and `outputs/curriculum_history.jsonl`.

Returns `LoaderResult` where `df` is a dict:

| Key | Type | Description |
|-----|------|-------------|
| `"current"` | dict | Current stage JSON (`{stage, task, advance_threshold, …}`) |
| `"history"` | DataFrame | Advancement event log |

---

### `load_synthetic`

```python
def load_synthetic(
    workspace_root: Path,
    *,
    session_id: str | None = None,
) -> LoaderResult
```

Reads `source` column from merged dataset `meta/episodes.parquet`. Returns a DataFrame with columns:

| Column | Dtype | Description |
|--------|-------|-------------|
| `source` | string | `"real"`, `"sim_dr"`, or `"mimicgen"` |
| `n_episodes` | Int64 | Episode count for this source |
| `dataset` | string | Dataset name |

---

### `load_paths`

```python
def load_paths(workspace_root: Path) -> WorkspacePaths
```

Resolves and returns `WorkspacePaths`. Does not return `LoaderResult`.

---

## `lerobot_isaac_dashboard.tabs`

### `Tab` (base class)

```python
class Tab:
    title: str                      # human-readable tab name (shown in UI)
    slug: str                       # filename-safe identifier (used in HTML anchors)
    primary_loader_slug: str        # key into ctx.loader_results for empty-state guard

    def render(
        self,
        ctx: TabContext,
        *,
        container: Any = None,
    ) -> list[go.Figure]:
        ...
```

`container=None` triggers the static-export path; passing a Streamlit container
triggers the live path which also calls `container.plotly_chart(fig)`.

Always returns a list of figures — may be empty for the empty state.

---

### `TabContext`

```python
@dataclass
class TabContext:
    workspace_root: Path
    session_id: str | None
    loader_results: dict[str, LoaderResult]  # keyed by loader slug
    refresh_ts: datetime                     # last data refresh time
```

---

### `TABS`

```python
TABS: list[type[Tab]]
```

Ordered list of all 8 tab classes in display order:
`[DataCollectionTab, SyntheticTab, PolicyTrainingTab, WorldModelTab,
  EvaluationTab, AutoresearchTab, CurriculumTab, PipelineHealthTab]`

---

### `render_kpi_row`

```python
def render_kpi_row(
    container: Any,
    kpis: list[dict],
    *,
    n_cols: int = 4,
) -> None
```

Renders a row of KPI metric cards in a Streamlit container. Each KPI dict has
`{label, value, delta, help}`. Used by tabs that need a top-level summary strip.

---

## `lerobot_isaac_dashboard.report`

### `export_report`

```python
def export_report(
    workspace_root: Path,
    *,
    session_id: str | None = None,
    output_dir: Path | None = None,
    inline_plotlyjs: bool = True,
    with_csv: bool = False,
    save_snapshot: bool = True,
) -> Path
```

Run all loaders headless, render all tabs to Plotly figures, and write a
self-contained static HTML report.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `workspace_root` | required | Absolute path to workspace |
| `session_id` | None | Scope session-aware loaders to this session |
| `output_dir` | `outputs/reports/<run_id>/` | Override output directory |
| `inline_plotlyjs` | True | Embed plotly.min.js (~3 MB); use False for CDN-dependent ~50 KB report |
| `with_csv` | False | Export each loader DataFrame as a companion CSV |
| `save_snapshot` | True | Auto-save a snapshot alongside the report |

**Returns:** `Path` — absolute path to the written `report.html`.

---

### `run_loaders_headless`

```python
def run_loaders_headless(
    workspace_root: Path,
    *,
    session_id: str | None = None,
) -> dict[str, LoaderResult]
```

Run all registered loaders without Streamlit. Used internally by `export_report`
and `save_snapshot`.

---

## `lerobot_isaac_dashboard.snapshots`

### `SnapshotMeta`

```python
@dataclass
class SnapshotMeta:
    snapshot_id: str            # e.g. "2026-05-08T072115-baseline"
    label: str | None           # human-readable label
    workspace_root: Path        # workspace that was snapshotted
    session_id: str | None      # session ID that scoped the loaders
    git_sha: str | None         # short git SHA at snapshot time
    dashboard_version: str      # __version__ of the dashboard package
    ts: datetime                # UTC creation time
    loader_slugs: list[str]     # loaders captured
    schema_version: int         # format version (current: 1)

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "SnapshotMeta": ...
```

---

### `save_snapshot`

```python
def save_snapshot(
    workspace_root: Path,
    *,
    session_id: str | None = None,
    label: str | None = None,
    snapshot_id: str | None = None,
    output_dir: Path | None = None,
) -> Path
```

Run all loaders headless and persist results as a snapshot directory.

**Returns:** `Path` to the snapshot directory (contains `meta.json` + `loaders/`).

---

### `load_snapshot`

```python
def load_snapshot(
    path_or_id: Path | str,
    workspace_root: Path | None = None,
) -> tuple[SnapshotMeta, dict[str, LoaderResult]]
```

Load a snapshot from disk and reconstruct loader results.

**Raises:**
- `FileNotFoundError` — snapshot directory or `meta.json` missing
- `ValueError` — `schema_version` is newer than the installed dashboard

---

### `list_snapshots`

```python
def list_snapshots(workspace_root: Path) -> list[SnapshotMeta]
```

List all snapshots under `outputs/snapshots/`. Returns list sorted by `ts` descending
(newest first). Malformed snapshots are silently skipped.

---

### `cli_main` (snapshots)

```python
def cli_main(argv: list[str] | None = None) -> int
```

Entrypoint for `lerobot-isaac-snapshot` and `python -m lerobot_isaac_dashboard.snapshots`.

```
lerobot-isaac-snapshot --workspace=PATH [--label=LABEL] [--session-id=SID] [save|list]
```

---

## `lerobot_isaac_dashboard.compare`

### `CompareContext`

```python
@dataclass
class CompareContext:
    snapshots: list[tuple[SnapshotMeta, dict[str, LoaderResult]]]
    mode: Literal["2way", "nway"]

    @property
    def labels(self) -> list[str]: ...    # label or snapshot_id for each entry
```

---

### `build_compare_context`

```python
def build_compare_context(
    snapshots: list[tuple[SnapshotMeta, dict[str, LoaderResult]]],
    *,
    mode: Literal["2way", "nway"] = "2way",
) -> CompareContext
```

Wrap multiple snapshot entries into a `CompareContext`.

---

### `render_compare_2way`

```python
def render_compare_2way(
    a: tuple[SnapshotMeta, dict[str, LoaderResult]],
    b: tuple[SnapshotMeta, dict[str, LoaderResult]],
    *,
    container: Any = None,
) -> dict[str, list[go.Figure]]
```

Render a 2-way side-by-side comparison. For each tab, renders both snapshots with
a delta KPI strip above. Returns `{tab_slug: [figures_a + figures_b]}`.

---

### `render_compare_nway`

```python
def render_compare_nway(
    snapshots: list[tuple[SnapshotMeta, dict[str, LoaderResult]]],
    *,
    container: Any = None,
) -> dict[str, list[go.Figure]]
```

Render an N-way overlay. For each tab, overlays traces from all snapshots with
the snapshot label as the legend. Returns `{tab_slug: [overlay_figures]}`.

---

### `export_compare_report`

```python
def export_compare_report(
    workspace_root: Path,
    snapshot_ids: list[str],
    *,
    mode: Literal["2way", "nway"] = "2way",
    output_dir: Path | None = None,
    inline_plotlyjs: bool = True,
) -> Path
```

Render a static HTML compare report and write it to disk.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `workspace_root` | required | Resolves snapshot paths and default output dir |
| `snapshot_ids` | required | 2+ snapshot IDs or absolute paths |
| `mode` | `"2way"` | Layout mode: `"2way"` or `"nway"` |
| `output_dir` | `outputs/reports/compare-<joined>/` | Override output dir |
| `inline_plotlyjs` | True | Embed plotly.min.js |

**Returns:** `Path` to `report.html`.

**Raises:** `ValueError` when fewer than 2 valid snapshots can be loaded.

---

### `cli_main` (compare)

```python
def cli_main(argv: list[str] | None = None) -> int
```

Entrypoint for `lerobot-isaac-compare` and `python -m lerobot_isaac_dashboard.compare`.

```
lerobot-isaac-compare --workspace=PATH --snapshots A B [C ...] [--mode 2way|nway] [--cdn]
```
