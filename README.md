# lerobot-isaac-dashboard

Streamlit + Plotly **metrics dashboard** for the `lerobot-isaac-training` pipeline.

Visualises all local training artefacts — datasets, checkpoints, eval results,
world-model runs, autoresearch history, curriculum stage, and pipeline health —
from a single Streamlit app and/or a static HTML report.

---

## Status

**Phase 1 — scaffold complete.**
Package structure, dependencies, and pixi integration are in place.
Loaders (P2), tab modules (P3), the live Streamlit app (P4), and the
static HTML exporter (P5) land in subsequent phases.

| Component | Status |
|-----------|--------|
| Package scaffold + pixi env | Complete (P1) |
| Metric loaders | Phase 2 |
| Tab modules (8 tabs) | Phase 3 |
| Live Streamlit app (`pixi run dashboard`) | Phase 4 |
| Static HTML exporter (`pixi run report`) | Phase 5 |
| Full test suite | Phase 6 |
| Docs + runbook 07 | Phase 7 |

---

## Quick Start (after P4 lands)

```bash
# Live dashboard (auto-refresh)
pixi run -e dashboard dashboard

# Static HTML report
pixi run -e dashboard report
# -> outputs/reports/<run_id>/report.html
```

See `docs/runbook/07-dashboard.md` (created in Phase 7) for full usage.

---

## Installation

### Monorepo mode (pixi)

```bash
cd ~/workspaces/lerobot-isaac-training
pixi install                        # adds dashboard package to default env
pixi install -e dashboard           # installs full dashboard feature env
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
| `pyarrow>=15.0` | Parquet dataset reading |
| `jinja2>=3.1` | HTML report templating |
| `pyyaml>=6.0` | YAML config/checkpoint metadata reading |
| `watchdog>=3.0` | File-system polling for auto-refresh |

### Soft (optional — loaders degrade gracefully)

| Package | Used for |
|---------|---------|
| `lerobot` | Richer dataset metadata (episode counts, FPS, features) |
| `stable-worldmodel` | World-model HDF5 metadata introspection |

---

## Data Sources (local files only)

| Source | Path pattern |
|--------|-------------|
| LeRobot datasets | `datasets/<task>/<repo_id>/` |
| Evaluation results | `outputs/eval/*.json` |
| Policy checkpoints | `outputs/checkpoints/<arch>/<run_id>/` |
| Training logs | `outputs/checkpoints/<arch>/<run_id>/log.txt` |
| Autoresearch history | `.agent-state/<session>/autoresearch/<slug>/history.jsonl` |
| Pipeline events | `.agent-state/<session>/events.jsonl` |
| Curriculum stage | `outputs/curriculum_stage.json` |

---

## Spinout

```bash
git subtree split -P packages/lerobot-isaac-dashboard -b spinout-dashboard
```

See `../../docs/ARCHITECTURE.md` for the full spinout procedure.

---

## Source-of-Truth Pointers

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-08-lerobot-isaac-dashboard-plan.md`
- ADR-0003 (soft-import): `../../docs/adr/0003-soft-import-discipline.md`
- Workspace ARCHITECTURE.md: `../../docs/ARCHITECTURE.md`
- Runbook (Phase 7): `../../docs/runbook/07-dashboard.md`
