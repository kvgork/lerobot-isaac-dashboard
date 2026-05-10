# lerobot-isaac-dashboard — Package Orientation

**Role:** Read-only metrics dashboard. Reads local artefacts produced by the
lerobot-isaac-training pipeline and presents them in a Streamlit UI (live) or
static HTML report (offline). Supports snapshot save/load and 2-way / N-way
run comparison. No GPU deps; CPU-only; no writes to training artefacts.
**Phase:** All phases complete (P1 scaffold → P5 live+static → P8 snapshots+compare → P7 docs).
**Status:** 281 tests passing. All 8 tabs implemented. Snapshot schema version 1.

---

## What This Package Does

Aggregates and visualises all local training artefacts from a single workspace:

1. **Data Collection tab** — LeRobot Parquet dataset summaries, episode counts, FPS.
   Source: `datasets/<task>/<repo_id>/`
2. **Synthetic Data tab** — DR replay vs MimicGen episode breakdown.
   Source: merged dataset `meta/episodes.parquet` (source column)
3. **Policy Training tab** — loss curves, checkpoint table, step/wall-clock progress.
   Source: `outputs/checkpoints/<arch>/<run_id>/log.txt`
4. **World Model Training tab** — DreamerV3 / LeWM `recon_loss` / `pred_loss` curves.
   Source: `outputs/checkpoints/<arch>/<run_id>/log.txt`
5. **Evaluation tab** — `pc_success`, `mean_ep_len`, intervention rate over time.
   Source: `outputs/eval/*.json`
6. **Autoresearch tab** — HP search history, plateau detector state, best config.
   Source: `.agent-state/<session>/autoresearch/<slug>/history.jsonl`
7. **Curriculum tab** — current stage, advancement events, task config diffs.
   Source: `outputs/curriculum_stage.json` + `outputs/curriculum_history.jsonl`
8. **Pipeline Health tab** — agent event log, last-run timestamps, error summary.
   Source: `.agent-state/<session>/events.jsonl`

Three run modes:
- `pixi run -e dashboard dashboard` — live Streamlit app with file-system polling
- `pixi run -e dashboard report` — static HTML export to `outputs/reports/<run_id>/`
- `pixi run -e dashboard snapshot` — capture current loader state to `outputs/snapshots/<id>/`

Compare modes (via `pixi run -e dashboard compare` or UI Mode radio):
- **2-way:** each tab split into two columns; delta KPI strip (pc_success, train_loss) above
- **N-way:** traces overlaid on shared axes; snapshot label as legend

---

## Internal Structure

| File/Dir | Role |
|----------|------|
| `src/lerobot_isaac_dashboard/__init__.py` | Public exports + `__version__` |
| `src/lerobot_isaac_dashboard/version.py` | Single-source `__version__` |
| `src/lerobot_isaac_dashboard/loaders/` | 9 loaders: one per data source |
| `src/lerobot_isaac_dashboard/loaders/_base.py` | `LoaderResult`, `empty_df`, safe I/O helpers |
| `src/lerobot_isaac_dashboard/tabs/` | 8 tab modules + `_base.py` + `_kpis.py` + `_compare.py` |
| `src/lerobot_isaac_dashboard/tabs/_base.py` | `Tab` base class, `TabContext` dataclass |
| `src/lerobot_isaac_dashboard/app.py` | Streamlit app shell — assembles sidebar + tabs |
| `src/lerobot_isaac_dashboard/report.py` | Static HTML exporter — Jinja2 template + `export_report` |
| `src/lerobot_isaac_dashboard/snapshots.py` | `save_snapshot`, `load_snapshot`, `list_snapshots`, `SnapshotMeta` |
| `src/lerobot_isaac_dashboard/compare.py` | `render_compare_2way`, `render_compare_nway`, `export_compare_report` |
| `src/lerobot_isaac_dashboard/cli.py` | Unified CLI entrypoint dispatcher |
| `src/lerobot_isaac_dashboard/runtime/` | Static render helpers (`render_tab_to_html`) |
| `src/lerobot_isaac_dashboard/templates/` | Jinja2 HTML templates for report.py |
| `tests/` | 281 tests covering loaders, tabs, report, snapshots, compare |
| `docs/API.md` | Full public API reference |
| `docs/EXAMPLES.md` | 5 worked examples |
| `docs/INTERNALS.md` | Loader contracts, snapshot format, compare bridge details |

---

## Public API

```python
from lerobot_isaac_dashboard.loaders import (
    LoaderResult,
    WorkspacePaths,
    load_parquet_dataset,    # datasets/ → episode summary DF
    load_eval_results,       # outputs/eval/*.json → eval metrics DF
    load_checkpoints,        # outputs/checkpoints/ → checkpoint inventory DF
    load_training_logs,      # log.txt → step/metric DF
    load_autoresearch,       # .agent-state/ → dict with history/program/best/plateau
    load_events,             # .agent-state/ → events DF
    load_curriculum,         # outputs/curriculum_stage.json → dict with current/history
    load_synthetic,          # datasets/ merged meta → source breakdown DF
    load_paths,              # WorkspacePaths resolver
)
from lerobot_isaac_dashboard.tabs import TABS, Tab, TabContext
from lerobot_isaac_dashboard.report import export_report
from lerobot_isaac_dashboard.snapshots import (
    SnapshotMeta, save_snapshot, load_snapshot, list_snapshots,
)
from lerobot_isaac_dashboard.compare import (
    CompareContext, build_compare_context,
    render_compare_2way, render_compare_nway,
    export_compare_report,
)
```

CLI entrypoints (declared in `pyproject.toml`):
- `lerobot-isaac-dashboard` — live Streamlit app
- `lerobot-isaac-report` — static HTML report
- `lerobot-isaac-snapshot` — save or list snapshots
- `lerobot-isaac-compare` — static HTML compare report

---

## Coupling

- **Hard deps:** streamlit, plotly, pandas, pyarrow, jinja2, pyyaml, watchdog
- **Soft deps (ADR-0003):** `lerobot`, `stable-worldmodel`
  — loaders degrade gracefully (return empty DataFrame) when absent
- **Sibling dep:** `lerobot-isaac-meta` (soft) — used for `WorkspacePaths` resolver;
  falls back to env-var-based path discovery if not installed
- **Does NOT depend on:** `lerobot-isaac-env`, `lerobot-isaac-adapters`,
  `lerobot-isaac-synthetic`, `lerobot-isaac-autoresearch`
  (those packages produce artefacts this package reads; no code coupling)
- **No GPU deps.** Runs in the `dashboard` pixi environment.
- **Read-only** with respect to training artefacts. Snapshot and report writes
  go exclusively to `outputs/snapshots/` and `outputs/reports/` (both gitignored).

Dependency graph position: leaf node — reads artefacts, writes only to `outputs/{snapshots,reports}/`.

---

## Soft-Import Pattern

Per ADR-0003, heavy deps are never imported at module top-level:

```python
# loaders/parquet_dataset.py
try:
    import lerobot  # noqa: F401
    _HAS_LEROBOT = True
except ImportError:
    _HAS_LEROBOT = False

def load_parquet_dataset(workspace_root, **kwargs):
    if _HAS_LEROBOT:
        # richer metadata via LeRobotDataset API
        ...
    else:
        # fallback: read parquet directly with pandas
        ...
```

---

## Tab Dual-Render Contract

Every `Tab.render(ctx, *, container=None)` must satisfy:

1. Return `list[go.Figure]` in all cases (live and static paths).
2. When `container` is not None (live): also call `container.plotly_chart(fig)` for each figure.
3. Never raise — catch exceptions internally and return empty list or placeholder figure.
4. Check `ctx.loader_results[primary_loader_slug].is_empty` and return early with an
   empty-state figure when data is absent.

This dual-render design allows the same tab code to serve both the Streamlit UI and the
static HTML exporter without duplication.

---

## Snapshot Format

Schema version: **1** (bumped on breaking format changes).

```
outputs/snapshots/<snapshot_id>/
├── meta.json                    — SnapshotMeta (workspace, git_sha, ts, label, schema_version)
└── loaders/
    ├── parquet_dataset.parquet
    ├── eval_results.parquet
    ├── checkpoints.parquet
    ├── training_logs.parquet
    ├── events.parquet
    ├── synthetic.parquet
    ├── autoresearch__history.parquet
    ├── autoresearch__program.json
    ├── autoresearch__best.json
    ├── autoresearch__plateau.json
    ├── curriculum__current.json
    ├── curriculum__history.parquet
    └── paths.json
```

`load_snapshot` raises `ValueError` when `schema_version > SCHEMA_VERSION`. To reload
a snapshot from a newer dashboard, upgrade the package.

---

## Testing

```bash
pixi run -e dashboard python -m pytest tests/ -q       # 281 tests, no hardware required
pixi run -e dashboard python -m pytest tests/ -v       # verbose
```

No hardware or optional deps needed. All tests use tmp_path fixtures.

---

## How to Extend

### Add a new data source

1. Create `src/lerobot_isaac_dashboard/loaders/my_source.py`.
2. Implement `load_my_source(workspace_root, *, session_id=None) -> LoaderResult`.
3. Return `LoaderResult(df=empty_df(...), is_empty=True)` on any missing/malformed file.
4. Export from `loaders/__init__.py`.
5. Wire into `app.py` loader registry and `report.py` `_LOADERS` dict.
6. Add slug + spec to `snapshots._LOADER_SPECS`.
7. Add tests in `tests/test_loader_my_source.py`.

### Add a new tab

1. Create `src/lerobot_isaac_dashboard/tabs/my_tab.py` with `class MyTab(Tab)`.
2. Set `title`, `slug`, `primary_loader_slug`; implement `render(ctx, *, container=None)`.
3. Append `MyTab` to `TABS` list in `tabs/__init__.py`.
4. Add `MyTab` to `tabs/__init__.py` exports.

### Add a real hardware / integration test

Mark with `@pytest.mark.requires_lerobot` so CI skips it automatically.

---

## Source-of-Truth Pointers

- Build plan: `${CLAUDE_CODE_ROOT}/plans/2026-05-08-lerobot-isaac-dashboard-plan.md`
- ADR-0006 (stack choice): `../../docs/adr/0006-dashboard-stack.md`
- ADR-0003 (soft-import): `../../docs/adr/0003-soft-import-discipline.md`
- Workspace CLAUDE.md: `../../CLAUDE.md`
- Workspace ARCHITECTURE.md: `../../ARCHITECTURE.md`
- Runbook: `../../docs/runbook/07-dashboard.md`
- Package docs: `docs/API.md` | `docs/EXAMPLES.md` | `docs/INTERNALS.md`
