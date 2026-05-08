# lerobot-isaac-dashboard

Streamlit + Plotly **metrics dashboard** for the `lerobot-isaac-training` pipeline.

Visualises all local training artefacts — datasets, checkpoints, eval results,
world-model runs, autoresearch history, curriculum stage, and pipeline health —
from a single Streamlit app and/or a static HTML report. Supports snapshot
save/load and 2-way or N-way run comparisons without any external services.

---

## Status

**All phases complete (P1–P5 + P8).**

| Component | Status |
|-----------|--------|
| Package scaffold + pixi env | Complete (P1) |
| Metric loaders (9 loaders) | Complete (P2) |
| Tab modules (8 tabs) | Complete (P3) |
| Live Streamlit app (`pixi run -e dashboard dashboard`) | Complete (P4) |
| Static HTML exporter (`pixi run -e dashboard report`) | Complete (P5) |
| Snapshot save/load (`pixi run -e dashboard snapshot`) | Complete (P8) |
| 2-way and N-way compare (`pixi run -e dashboard compare`) | Complete (P8) |
| Docs + runbook 07 | Complete (P7) |

---

## Quick Start

```bash
# Live dashboard (auto-refresh)
pixi run -e dashboard dashboard

# Static HTML report (with auto-snapshot side effect)
pixi run -e dashboard report --workspace=$PWD

# Save a labeled snapshot
pixi run -e dashboard snapshot --workspace=$PWD --label=baseline

# Compare two snapshots side-by-side (2-way)
pixi run -e dashboard compare --workspace=$PWD --snapshots <A> <B>

# Compare 3+ snapshots as overlaid traces (N-way)
pixi run -e dashboard compare --workspace=$PWD --snapshots <A> <B> <C> --mode nway
```

See `docs/runbook/07-dashboard.md` for full usage, tab guide, and troubleshooting.

---

## Installation

### Monorepo mode (pixi)

```bash
cd ~/workspaces/lerobot-isaac-training
pixi install -e dashboard           # installs dashboard feature env
```

### Standalone mode

```bash
cd packages/lerobot-isaac-dashboard
pixi install         # standalone pixi env (dormant in monorepo mode)
```

---

## Dependencies

### Hard (always required)

| Package | Purpose |
|---------|---------|
| `streamlit>=1.32` | Live web UI |
| `plotly>=5.20` | Interactive figures (Streamlit + static HTML) |
| `pandas>=2.1` | Tabular data handling |
| `pyarrow>=15.0` | Parquet dataset reading + snapshot persistence |
| `jinja2>=3.1` | HTML report templating |
| `pyyaml>=6.0` | YAML config/checkpoint metadata reading |
| `watchdog>=3.0` | File-system polling for auto-refresh |

### Soft (optional — loaders degrade gracefully)

| Package | Used for |
|---------|---------|
| `lerobot` | Richer dataset metadata (episode counts, FPS, features) |
| `stable-worldmodel` | World-model HDF5 metadata introspection |

---

## Data Sources (local files only — no external services)

| Source | Path pattern |
|--------|-------------|
| LeRobot datasets | `datasets/<task>/<repo_id>/` |
| Evaluation results | `outputs/eval/*.json` |
| Policy checkpoints | `outputs/checkpoints/<arch>/<run_id>/` |
| Training logs | `outputs/checkpoints/<arch>/<run_id>/log.txt` |
| World-model logs | `outputs/checkpoints/<arch>/<run_id>/log.txt` |
| Autoresearch history | `.agent-state/<session>/autoresearch/<slug>/history.jsonl` |
| Pipeline events | `.agent-state/<session>/events.jsonl` |
| Curriculum stage | `outputs/curriculum_stage.json` |
| Synthetic data meta | `datasets/<merged>/meta/episodes.parquet` |

Snapshots are written to `outputs/snapshots/<snapshot_id>/` (gitignored).
Reports are written to `outputs/reports/<run_id>/` (gitignored).

---

## Tab Guide

| # | Tab | Data source | Key metric |
|---|-----|------------|-----------|
| 1 | Data Collection | `datasets/` Parquet | Episode count, FPS |
| 2 | Synthetic Data | Merged dataset meta | real vs sim_dr vs mimicgen breakdown |
| 3 | Policy Training | `outputs/checkpoints/` log | Loss curves, step progress |
| 4 | World Model | `outputs/checkpoints/` log | recon_loss / pred_loss |
| 5 | Evaluation | `outputs/eval/*.json` | pc_success over time |
| 6 | Autoresearch | `.agent-state/*/autoresearch/` | HP search history, best config |
| 7 | Curriculum | `outputs/curriculum_stage.json` | Current stage, advancement events |
| 8 | Pipeline Health | `.agent-state/*/events.jsonl` | Agent event log, error table |

---

## Snapshot + Compare

### Save a snapshot

```bash
# CLI
lerobot-isaac-snapshot --workspace=$PWD --label=after-epoch-100

# In the UI: sidebar → Save snapshot → enter label → click Save
```

Snapshots are stored in `outputs/snapshots/<timestamp>-<label>/` as:
- `meta.json` — workspace path, git SHA, timestamp, label, schema version
- `loaders/*.parquet` — DataFrames for all data-source loaders
- `loaders/*.json` — dict members for hierarchical loaders (autoresearch, curriculum)

### Compare two snapshots (2-way)

```bash
lerobot-isaac-compare --workspace=$PWD --snapshots before after
```

Output: `outputs/reports/compare-before-vs-after/report.html`
Layout: each tab split into two columns with a delta KPI strip above (pc_success, train_loss).

### Compare 3+ snapshots (N-way overlay)

```bash
lerobot-isaac-compare --workspace=$PWD --snapshots baseline exp1 exp2 --mode nway
```

Output: `outputs/reports/compare-baseline-vs-exp1-vs-exp2/report.html`
Layout: traces from each snapshot overlaid on shared axes with snapshot label as legend.

---

## Public API

```python
# Loaders
from lerobot_isaac_dashboard.loaders import (
    LoaderResult,
    WorkspacePaths,
    load_parquet_dataset,
    load_eval_results,
    load_checkpoints,
    load_training_logs,
    load_autoresearch,
    load_events,
    load_curriculum,
    load_synthetic,
    load_paths,
)

# Tabs
from lerobot_isaac_dashboard.tabs import TABS, Tab, TabContext

# Static report
from lerobot_isaac_dashboard.report import export_report

# Snapshots
from lerobot_isaac_dashboard.snapshots import (
    SnapshotMeta,
    save_snapshot,
    load_snapshot,
    list_snapshots,
)

# Compare
from lerobot_isaac_dashboard.compare import (
    CompareContext,
    build_compare_context,
    render_compare_2way,
    render_compare_nway,
    export_compare_report,
)
```

CLI entrypoints:

| Command | Description |
|---------|-------------|
| `lerobot-isaac-dashboard` | Start live Streamlit app |
| `lerobot-isaac-report` | Export static HTML report |
| `lerobot-isaac-snapshot` | Save or list snapshots |
| `lerobot-isaac-compare` | Export HTML compare report |

---

## Running Tests

```bash
pixi run -e dashboard python -m pytest tests/ -q
# 281 tests, no hardware or heavy deps required
```

---

## Spinout

```bash
git subtree split -P packages/lerobot-isaac-dashboard -b spinout-dashboard
```

See `../../ARCHITECTURE.md §Spinout Mechanics` for the full procedure.

---

## Source-of-Truth Pointers

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-08-lerobot-isaac-dashboard-plan.md`
- ADR-0006 (stack choice): `../../docs/adr/0006-dashboard-stack.md`
- Workspace ARCHITECTURE.md: `../../ARCHITECTURE.md`
- Runbook: `../../docs/runbook/07-dashboard.md`
- Package docs: `docs/API.md` | `docs/EXAMPLES.md` | `docs/INTERNALS.md`
