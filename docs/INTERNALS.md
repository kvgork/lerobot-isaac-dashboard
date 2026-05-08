# lerobot-isaac-dashboard — Internals

Internal design decisions, contracts, and implementation details for contributors.

---

## LoaderResult Contract

Every `load_*` function must satisfy these invariants:

1. **Never raises.** Any I/O error, missing file, or schema mismatch is caught internally.
   The function logs at `DEBUG` level and returns an empty result.

2. **Returns `LoaderResult` with canonical columns.** Even when no data is found,
   the returned `df` is an empty DataFrame with all expected columns (correct dtypes).
   Tabs can always do `df["pc_success"]` without key-error guards.

3. **`is_empty` is accurate.** `True` iff no source files were read successfully.
   Tabs use this flag as the single empty-state signal.

4. **`source_paths` is populated.** Lists files that were successfully read.
   The UI displays these for provenance / staleness checks.

5. **`warnings` are non-fatal.** Lists schema mismatches, truncated files, etc.
   The UI may display these as inline banners. A non-empty `warnings` does not
   imply `is_empty`.

### `SCHEMA_COLUMNS` pattern

Each loader declares its schema as a module-level dict:

```python
# loaders/eval_results.py
_SCHEMA: dict[str, str] = {
    "run_id": "string",
    "ts": "datetime64[ns, UTC]",
    "arch": "string",
    "pc_success": "Float64",
    "mean_ep_len": "Float64",
    "n_episodes": "Int64",
}
```

The loader calls `_align_to_schema(df, _SCHEMA)` (from `_base.py`) before returning.
This adds missing columns with `pd.NA`, casts dtypes, and collects alignment warnings.
The empty-state DataFrame is `empty_df(list(_SCHEMA.keys()), _SCHEMA)`.

### Hierarchical loaders

`load_autoresearch` and `load_curriculum` return `df` as a `dict[str, DataFrame | dict]`
because their data spans multiple files with different shapes.

```python
# autoresearch example
result.df == {
    "history": pd.DataFrame(...),   # trials × columns
    "program": {...},               # program.md metadata
    "best": {...},                  # best trial config + metric
    "plateau": {...},               # plateau detector state
}
```

Tabs access these via `ctx.loader_results["autoresearch"].df["history"]`.

---

## Empty-State Discipline

The empty-state pattern is consistent across all tabs:

```python
class EvaluationTab(Tab):
    slug = "evaluation"
    primary_loader_slug = "eval_results"

    def render(self, ctx, *, container=None):
        result = ctx.loader_results.get(self.primary_loader_slug)
        if result is None or result.is_empty:
            fig = _empty_state_fig(self.title)
            if container is not None:
                container.plotly_chart(fig, use_container_width=True)
            return [fig]
        # ... actual render
```

`_empty_state_fig(title)` (from `_kpis.py`) returns a `go.Figure` with an annotation:
`"No data yet — run the training pipeline to populate this tab."` This ensures the
static exporter always gets a figure (no empty sections in the HTML).

---

## Tab Dual-Render Trick

The key architectural decision enabling a single codebase for both live and static
rendering is the `container=None` convention:

```python
def render(self, ctx, *, container=None):
    figs = []
    fig = go.Figure(...)
    figs.append(fig)

    if container is not None:          # live path
        container.plotly_chart(fig, use_container_width=True)

    return figs                        # always returned (static path uses this)
```

The Streamlit app (`app.py`) passes `container=tab_container` (the Streamlit tab widget).
The static exporter (`report.py`) passes `container=None` and uses the returned list.
The compare renderer (`compare.py`) also passes `container=None` for headless rendering,
then optionally calls `container.plotly_chart` itself in the live compare path.

**Rule:** Every tab's `render` must return the same list of figures regardless of the
`container` argument. The list is the canonical output; `container` is a side-effect sink.

---

## Snapshot File Layout

Schema version: `SCHEMA_VERSION = 1` (in `snapshots.py`).
Bump this integer when a breaking change is made to the snapshot format.
`load_snapshot` raises `ValueError` if the on-disk version is higher than the installed one.

```
outputs/snapshots/<snapshot_id>/
├── meta.json
│   {
│     "snapshot_id": "2026-05-08T072115-baseline",
│     "label": "baseline",
│     "workspace_root": "/home/koen/workspaces/lerobot-isaac-training",
│     "session_id": null,
│     "git_sha": "ecd39d1",
│     "dashboard_version": "0.1.0",
│     "ts": "2026-05-08T07:21:15+00:00",
│     "loader_slugs": ["parquet_dataset", "eval_results", ...],
│     "schema_version": 1
│   }
└── loaders/
    ├── parquet_dataset.parquet      — plain DataFrame loader → single .parquet
    ├── eval_results.parquet
    ├── checkpoints.parquet
    ├── training_logs.parquet
    ├── events.parquet
    ├── synthetic.parquet
    ├── autoresearch__history.parquet — dict loader: <slug>__<member_key>.*
    ├── autoresearch__program.json
    ├── autoresearch__best.json
    ├── autoresearch__plateau.json
    ├── curriculum__current.json
    ├── curriculum__history.parquet
    └── paths.json                   — WorkspacePaths (special case, not LoaderResult)
```

### `_LOADER_SPECS` dict

Controls how each loader is persisted and restored:

```python
_LOADER_SPECS = {
    "eval_results": {"type": "df"},          # plain DataFrame -> .parquet
    "autoresearch": {
        "type": "dict",
        "dict_members": {
            "history": "df",                 # DataFrame -> autoresearch__history.parquet
            "program": "json",               # dict    -> autoresearch__program.json
            "best":    "json",
            "plateau": "json",
        },
    },
    ...
}
```

When adding a new loader, add its slug + spec to `_LOADER_SPECS`. If the loader's
`df` shape changes in a breaking way, bump `SCHEMA_VERSION`.

---

## Compare Layout Decisions

### 2-way side-by-side

For the 2-way mode, `render_compare_2way` renders each tab independently for both
snapshots (`container=None`), yielding `figs_a` and `figs_b`. In the live path,
it creates two Streamlit columns and places each figure set in the corresponding
column. In the static path, the HTML exporter uses a flexbox layout:

```html
<div style="display:flex;gap:16px;">
  <div class="compare-col" style="flex:1;"><h4>Snapshot A</h4>...</div>
  <div class="compare-col" style="flex:1;"><h4>Snapshot B</h4>...</div>
</div>
```

Delta KPIs (computed by `_compute_delta_kpis`) are rendered as a strip above
each tab using `st.metric` (live) or an HTML strip (`_delta_kpi_html`, static).
Current delta KPIs: `pc_success (mean)` and `train_loss (latest)`.

### N-way overlay

For N-way, `render_compare_nway` renders each tab for all snapshots independently
(still `container=None`), then merges figures by index: figure[0] from each snapshot
is overlaid into a single `go.Figure` by copying traces and prepending the snapshot
label to the trace name.

```python
new_trace = trace.__class__(
    **{k: v for k, v in trace.to_plotly_json().items() if k != "type"},
    name=f"{snap_label} – {trace.name or ''}".strip(" –"),
    showlegend=True,
)
overlay.add_trace(new_trace)
```

Layout is copied from the first available snapshot's figure. For table-heavy tabs
that return no traces (e.g. Pipeline Health, which renders HTML tables via Streamlit),
the overlay will be empty for those tabs in static mode — this is acceptable because
those tabs do not produce time-series figures suitable for overlay.

---

## Safe I/O Helpers

All file reads go through helpers in `loaders/_base.py`:

| Helper | Returns | On error |
|--------|---------|----------|
| `safe_read_parquet(path)` | `pd.DataFrame | None` | `None` |
| `safe_read_jsonl(path)` | `pd.DataFrame | None` | `None` (malformed lines skipped) |
| `safe_read_json(path)` | `dict | None` | `None` |
| `glob_runs(root, pattern)` | `list[Path]` | `[]` |

All helpers catch broad exceptions and log at `DEBUG`. They must never propagate
exceptions to callers. This keeps the loader contract simple: callers only need
to check for `None` return.

---

## Streamlit Session State Caching

In `app.py`, loaders are wrapped with `@st.cache_data(ttl=...)` keyed on:
- `workspace_root` path
- `session_id`
- File mtimes of the primary source files (computed via `WorkspacePaths`)

Using file mtimes as cache keys means the cached data is invalidated automatically
when source files change, without needing to restart the server. The watchdog integration
(via the Watch files checkbox) additionally triggers a full `st.rerun()` when any file
in `outputs/` or `.agent-state/` is modified.

**Caution:** Streamlit's `st.cache_data` uses object identity for unhashable types.
Always pass `workspace_root` as a `str`, not a `Path`, to avoid cache misses from
`Path` object re-creation between reruns.

---

## Adding a New Loader

Full checklist:

1. Create `loaders/my_source.py` with:
   ```python
   _SCHEMA = {"col1": "string", "col2": "Float64", ...}

   def load_my_source(workspace_root, *, session_id=None):
       root = Path(workspace_root)
       # ... read files via safe_read_* helpers
       if no_data:
           return LoaderResult(df=empty_df(list(_SCHEMA), _SCHEMA), is_empty=True)
       df, warnings = _align_to_schema(raw_df, _SCHEMA)
       return LoaderResult(df=df, is_empty=False, source_paths=[...], warnings=warnings)
   ```

2. Export from `loaders/__init__.py`.

3. Add to `report.py` `_LOADERS` dict.

4. Add to `app.py` loader registry.

5. Add slug + spec to `snapshots._LOADER_SPECS`.

6. Wire loader result into relevant tab(s) via `TabContext.loader_results["my_source"]`.

7. Write `tests/loaders/test_my_source.py` covering: happy path, missing file,
   malformed schema, partial data.
