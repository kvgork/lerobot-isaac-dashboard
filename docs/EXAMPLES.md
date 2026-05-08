# lerobot-isaac-dashboard — Examples

Five worked examples covering all major usage modes.

---

## Example 1: Start the Live Dashboard with Auto-Refresh

```bash
# Install the dashboard environment
pixi install -e dashboard

# Set workspace path (add to ~/.bashrc for persistence)
export LEROBOT_ISAAC_WORKSPACE=~/workspaces/lerobot-isaac-training

# Launch the live Streamlit app
pixi run -e dashboard dashboard
# Opens http://localhost:8501 in the default browser
```

**What you see:**

The sidebar shows:
- Workspace path (read from `LEROBOT_ISAAC_WORKSPACE` or current directory)
- Session selector (lists subdirectories of `.agent-state/`)
- Refresh interval slider (0–120 s; 0 = manual refresh only)
- Watch files checkbox (enables watchdog auto-reload on file changes)
- Mode radio: **Live** | **Compare (2-way)** | **Compare (N-way)**
- Save snapshot button + label text field
- Export static report button

If `outputs/` is empty, all 8 tabs show a placeholder: "No data yet — run the
training pipeline to populate this tab." This is intentional; tabs never crash
on an empty workspace.

**Python equivalent (headless):**

```python
from pathlib import Path
from lerobot_isaac_dashboard.report import run_loaders_headless

workspace = Path("~/workspaces/lerobot-isaac-training").expanduser()
results = run_loaders_headless(workspace)
for slug, res in results.items():
    print(f"{slug}: is_empty={res.is_empty}")
```

---

## Example 2: Export a Static HTML Report

```bash
# Default: inline plotly.js (~5 MB self-contained), auto-snapshot enabled
pixi run -e dashboard report --workspace=$PWD
# Output: outputs/reports/<run_id>/report.html
# Side effect: outputs/snapshots/<run_id>/ (auto-snapshot, can disable with --no-snapshot)

# CDN mode: ~50 KB HTML but requires internet for offline viewing
pixi run -e dashboard report --workspace=$PWD --cdn

# No auto-snapshot
pixi run -e dashboard report --workspace=$PWD --no-snapshot

# Include raw CSV exports alongside the HTML
pixi run -e dashboard report --workspace=$PWD --with-csv
# Extra files: outputs/reports/<run_id>/{parquet_dataset.csv, eval_results.csv, ...}
```

**Python equivalent:**

```python
from pathlib import Path
from lerobot_isaac_dashboard.report import export_report

workspace = Path("~/workspaces/lerobot-isaac-training").expanduser()
report_path = export_report(
    workspace,
    session_id="20260508-064654-metrics-dashboard",  # optional: scope to one session
    inline_plotlyjs=True,
    with_csv=False,
    save_snapshot=True,
)
print(f"Report written: {report_path}")
```

**Output structure:**

```
outputs/reports/<run_id>/
├── report.html          # single self-contained HTML file
└── (optional) *.csv     # one CSV per loader if --with-csv
```

The HTML file opens in any browser without a server. Plotly figures are fully
interactive (hover, zoom, pan) even in offline mode (inline mode).

---

## Example 3: Save a Labeled Snapshot Manually

```bash
# Save with a human-readable label
pixi run -e dashboard snapshot --workspace=$PWD --label=baseline

# Save for a specific session only
pixi run -e dashboard snapshot --workspace=$PWD --label=epoch100 --session-id=20260508-064654

# List existing snapshots
pixi run -e dashboard snapshot --workspace=$PWD list
# Output (newest first):
#   2026-05-08T072115-baseline  2026-05-08 07:21:15 UTC  [baseline]
#   2026-05-08T064654-unlabeled 2026-05-08 06:46:54 UTC
```

**Python equivalent:**

```python
from pathlib import Path
from lerobot_isaac_dashboard.snapshots import save_snapshot, list_snapshots

workspace = Path("~/workspaces/lerobot-isaac-training").expanduser()

# Save
snap_dir = save_snapshot(workspace, label="baseline")
print(f"Snapshot saved: {snap_dir}")

# List
for meta in list_snapshots(workspace):
    print(f"{meta.snapshot_id}  label={meta.label}  sha={meta.git_sha}")
```

**Snapshot directory layout:**

```
outputs/snapshots/2026-05-08T072115-baseline/
├── meta.json                    — SnapshotMeta (workspace, git_sha, ts, label, schema_version=1)
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

---

## Example 4: Compare Two Snapshots Side-by-Side (2-way)

**CLI:**

```bash
pixi run -e dashboard compare \
  --workspace=$PWD \
  --snapshots 2026-05-08T072115-baseline 2026-05-08T143000-after-dr

# With CDN plotly (smaller file):
pixi run -e dashboard compare \
  --workspace=$PWD \
  --snapshots baseline after-dr \
  --cdn

# Custom output dir:
pixi run -e dashboard compare \
  --workspace=$PWD \
  --snapshots baseline after-dr \
  --output-dir /tmp/compare-run
```

**Output:** `outputs/reports/compare-baseline-vs-after-dr/report.html`

Layout: each of the 8 tabs is split into two columns labelled with the snapshot
label or ID. A delta KPI strip above each tab shows:
- `pc_success (mean)` — A: 0.62, B: 0.81, Delta: +0.19
- `train_loss (latest)` — A: 0.18, B: 0.11, Delta: -0.07

**Via UI:**

1. Sidebar → Mode → **Compare (2-way)**
2. Snapshot A selector → pick `baseline`
3. Snapshot B selector → pick `after-dr`
4. Dashboard renders inline (no HTML file written)
5. Click "Export compare report" to write the HTML

**Python equivalent:**

```python
from pathlib import Path
from lerobot_isaac_dashboard.snapshots import load_snapshot
from lerobot_isaac_dashboard.compare import export_compare_report

workspace = Path("~/workspaces/lerobot-isaac-training").expanduser()

# Export compare report
report_path = export_compare_report(
    workspace,
    snapshot_ids=["2026-05-08T072115-baseline", "2026-05-08T143000-after-dr"],
    mode="2way",
)
print(f"Compare report: {report_path}")

# Or render programmatically (no HTML file)
from lerobot_isaac_dashboard.compare import render_compare_2way
a = load_snapshot("2026-05-08T072115-baseline", workspace_root=workspace)
b = load_snapshot("2026-05-08T143000-after-dr", workspace_root=workspace)
figs_by_tab = render_compare_2way(a, b)
# figs_by_tab["evaluation"] -> [list of go.Figure]
```

---

## Example 5: N-Way Overlay Across 3+ Snapshots

**CLI:**

```bash
pixi run -e dashboard compare \
  --workspace=$PWD \
  --snapshots baseline exp-lr1e3 exp-lr5e4 exp-dr5x \
  --mode nway

# Output: outputs/reports/compare-baseline-vs-exp-lr1e3-vs-exp-lr5e4-.../report.html
```

**Output layout:** Each tab shows overlaid traces from all 4 snapshots on the same
axes. The legend identifies each trace by snapshot label:
- `baseline – loss`
- `exp-lr1e3 – loss`
- `exp-lr5e4 – loss`
- `exp-dr5x – loss`

Time-series tabs (Policy Training, Evaluation) overlay naturally. Table-heavy tabs
(Curriculum, Pipeline Health) stack rows with a `snapshot` column prepended.

**Via UI:**

1. Sidebar → Mode → **Compare (N-way)**
2. Multiselect → pick 3+ snapshot IDs
3. Dashboard renders overlaid figures inline
4. Click "Export compare report" for HTML

**Python equivalent:**

```python
from pathlib import Path
from lerobot_isaac_dashboard.snapshots import load_snapshot
from lerobot_isaac_dashboard.compare import render_compare_nway, export_compare_report

workspace = Path("~/workspaces/lerobot-isaac-training").expanduser()

snap_ids = ["baseline", "exp-lr1e3", "exp-lr5e4", "exp-dr5x"]

# Static HTML report
report_path = export_compare_report(workspace, snap_ids, mode="nway")

# Or render to figures
snapshots = [load_snapshot(sid, workspace_root=workspace) for sid in snap_ids]
figs_by_tab = render_compare_nway(snapshots)
# figs_by_tab["policy_training"] -> [go.Figure with 4 overlaid traces]
```
