# lerobot-isaac-dashboard — Package Orientation

**Role:** Read-only metrics dashboard. Reads local artefacts produced by the
lerobot-isaac-training pipeline and presents them in a Streamlit UI (live) or
static HTML report (offline). No writes, no GPU deps, CPU-only.
**Phase:** 1 (scaffold) — package structure and pixi env complete; app modules land in P2-P5.
**Status:** scaffold. Import tests passing.

---

## What This Package Does

Aggregates and visualises all local training artefacts from a single workspace:

1. **Data Collection tab** — LeRobot Parquet dataset summaries, episode counts, FPS.
2. **Synthetic Data tab** — DR replay vs MimicGen episode breakdown.
3. **Policy Training tab** — loss curves, checkpoint table, step/wall-clock progress.
4. **World Model Training tab** — DreamerV3 / LeWM training metrics.
5. **Evaluation tab** — `pc_success`, `mean_ep_len`, intervention rate over time.
6. **Autoresearch tab** — HP search history, plateau detector state, best config.
7. **Curriculum tab** — current stage, advancement events, task config diffs.
8. **Pipeline Health tab** — agent event log, last-run timestamps, error summary.

Two run modes:
- `pixi run -e dashboard dashboard` — live Streamlit app with file-system polling.
- `pixi run -e dashboard report` — static HTML export to `outputs/reports/<run_id>/`.

---

## Internal Structure

| File/Dir | Role |
|----------|------|
| `src/lerobot_isaac_dashboard/__init__.py` | Public exports + `__version__` |
| `src/lerobot_isaac_dashboard/version.py` | Single-source-of-truth `__version__` |
| `src/lerobot_isaac_dashboard/loaders/` | One loader per data source (P2) |
| `src/lerobot_isaac_dashboard/tabs/` | One Streamlit tab module per pipeline step (P3) |
| `src/lerobot_isaac_dashboard/app.py` | Streamlit app shell — assembles tabs (P4) |
| `src/lerobot_isaac_dashboard/report.py` | Static HTML exporter — Jinja2 template (P5) |
| `src/lerobot_isaac_dashboard/cli.py` | `lerobot-isaac-dashboard` CLI entrypoint |
| `tests/conftest.py` | tmp_path workspace builder fixture (expanded in P6) |
| `tests/test_imports.py` | Smoke: importable without heavy deps + ADR-0003 checks |

---

## Public API

**TBD — stabilises in Phase 5 after all loaders and tab modules are implemented.**

Anticipated stable exports (Phase 5+):

```python
from lerobot_isaac_dashboard import __version__
from lerobot_isaac_dashboard.loaders.parquet_dataset import load_parquet_dataset
from lerobot_isaac_dashboard.loaders.eval_results import load_eval_results
from lerobot_isaac_dashboard.report import generate_report
```

CLI entrypoints:
- `lerobot-isaac-dashboard` — alias for `streamlit run app.py`
- `lerobot-isaac-report` — static HTML export

---

## Coupling

- **Hard deps:** streamlit, plotly, pandas, pyarrow, jinja2, pyyaml, watchdog
- **Soft deps (ADR-0003):** `lerobot`, `stable-worldmodel`
  — loaders must degrade gracefully (return empty DataFrame) when absent
- **Sibling dep:** `lerobot-isaac-meta` (soft) — used for `WorkspacePaths` resolver;
  falls back to env-var-based path discovery if not installed
- **Does NOT depend on:** `lerobot-isaac-env`, `lerobot-isaac-adapters`,
  `lerobot-isaac-synthetic`, `lerobot-isaac-autoresearch`
  (those packages produce artefacts this package reads, but no code coupling)
- **No GPU deps.** Runs in the `default` pixi env (extended with `dashboard` feature).

Dependency graph position: leaf node — reads artefacts, writes nothing.

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

## Testing

```bash
cd packages/lerobot-isaac-dashboard
python3 -m pytest tests/ -q       # all tests, no hardware or heavy deps required
python3 -m pytest tests/ -v       # verbose
```

Phase 6 will add per-loader tests (happy path, missing file, malformed schema).
All loader tests use tmp_path fixture factories — no real workspace data needed.

---

## How to Extend

### Add a new data source

1. Create `src/lerobot_isaac_dashboard/loaders/my_source.py`.
2. Implement `load_my_source(workspace_root, *, session_id=None) -> pd.DataFrame`.
3. Return an empty DataFrame with declared columns on any missing/malformed file.
4. Wire into `app.py` and the relevant tab module.
5. Add tests in `tests/test_loader_my_source.py` covering happy path + missing file.

### Add a new tab

1. Create `src/lerobot_isaac_dashboard/tabs/my_tab.py` with `render(workspace_root)`.
2. Register in `app.py`'s tab list.
3. Add the tab name to the 8-tab constant in `tabs/__init__.py`.

---

## Source-of-Truth Pointers

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-08-lerobot-isaac-dashboard-plan.md`
- ADR-0003 (soft-import): `../../docs/adr/0003-soft-import-discipline.md`
- Workspace CLAUDE.md: `../../CLAUDE.md`
- Workspace ARCHITECTURE.md: `../../docs/ARCHITECTURE.md`
- Runbook (Phase 7): `../../docs/runbook/07-dashboard.md`
