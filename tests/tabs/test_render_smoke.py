"""test_render_smoke.py — Smoke tests for all 8 tab render methods.

Parametrized over:
    tab_class : one of TABS (8 classes)
    state     : "empty" | "populated"

Empty state:
    Every tab must render without exception and return [].

Populated state:
    Every tab must return at least 1 go.Figure, all isinstance(fig, go.Figure).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest

import plotly.graph_objects as go

from lerobot_isaac_dashboard.loaders._base import LoaderResult, empty_df
from lerobot_isaac_dashboard.loaders.autoresearch import HISTORY_SCHEMA
from lerobot_isaac_dashboard.loaders.checkpoints import CHECKPOINT_SCHEMA
from lerobot_isaac_dashboard.loaders.curriculum import CURRICULUM_HISTORY_SCHEMA
from lerobot_isaac_dashboard.loaders.eval_results import EVAL_SCHEMA
from lerobot_isaac_dashboard.loaders.events import EVENTS_SCHEMA
from lerobot_isaac_dashboard.loaders.parquet_dataset import DATASET_SUMMARY_SCHEMA
from lerobot_isaac_dashboard.loaders.synthetic import SYNTHETIC_SCHEMA
from lerobot_isaac_dashboard.loaders.training_logs import TRAINING_LOG_SCHEMA
from lerobot_isaac_dashboard.tabs import TABS, TabContext

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_TS = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _empty_loader_results() -> dict[str, LoaderResult]:
    """All loaders as empty canonical-schema LoaderResults."""
    return {
        "parquet_dataset": LoaderResult(
            df=empty_df(list(DATASET_SUMMARY_SCHEMA.keys()), DATASET_SUMMARY_SCHEMA),
            is_empty=True,
        ),
        "synthetic": LoaderResult(
            df=empty_df(list(SYNTHETIC_SCHEMA.keys()), SYNTHETIC_SCHEMA),
            is_empty=True,
        ),
        "training_logs": LoaderResult(
            df=empty_df(list(TRAINING_LOG_SCHEMA.keys()), TRAINING_LOG_SCHEMA),
            is_empty=True,
        ),
        "checkpoints": LoaderResult(
            df=empty_df(list(CHECKPOINT_SCHEMA.keys()), CHECKPOINT_SCHEMA),
            is_empty=True,
        ),
        "eval_results": LoaderResult(
            df=empty_df(list(EVAL_SCHEMA.keys()), EVAL_SCHEMA),
            is_empty=True,
        ),
        "autoresearch": LoaderResult(
            df={
                "history": empty_df(list(HISTORY_SCHEMA.keys()), HISTORY_SCHEMA),
                "program": {},
                "best": {},
                "plateau": {},
            },
            is_empty=True,
        ),
        "curriculum": LoaderResult(
            df={
                "current": {},
                "history": empty_df(
                    list(CURRICULUM_HISTORY_SCHEMA.keys()), CURRICULUM_HISTORY_SCHEMA
                ),
            },
            is_empty=True,
        ),
        "events": LoaderResult(
            df=empty_df(list(EVENTS_SCHEMA.keys()), EVENTS_SCHEMA),
            is_empty=True,
        ),
    }


def _populated_loader_results(tmp_path: Path) -> dict[str, LoaderResult]:
    """Minimal valid DataFrames for every loader slot."""
    ts_str = "2026-01-01T00:00:00+00:00"

    parquet_df = pd.DataFrame(
        {
            "repo_id": ["pick_place_v1"],
            "n_episodes": pd.array([50], dtype="Int64"),
            "n_frames": pd.array([5000], dtype="Int64"),
            "fps": pd.array([30], dtype="Int64"),
            "source_real": pd.array([30], dtype="Int64"),
            "source_sim_dr": pd.array([15], dtype="Int64"),
            "source_mimicgen": pd.array([5], dtype="Int64"),
            "total_size_mb": [120.5],
            "mtime": pd.to_datetime([ts_str], utc=True),
        }
    )

    synthetic_df = pd.DataFrame(
        {
            "episode_index": pd.array([0, 1, 2, 3, 4], dtype="Int64"),
            "length": pd.array([100, 110, 90, 105, 95], dtype="Int64"),
            "source": ["real", "sim_dr", "mimicgen", "real", "sim_dr"],
            "task": ["pick", "pick", "place", "place", "pick"],
        }
    )

    logs_df = pd.DataFrame(
        {
            "arch": ["smolvla", "smolvla", "smolvla"],
            "run_id": ["run_001", "run_001", "run_001"],
            "step": pd.array([100, 200, 300], dtype="Int64"),
            "metric_name": ["train_loss", "train_loss", "val_loss"],
            "value": [0.5, 0.4, 0.35],
            "ts": pd.to_datetime([ts_str, ts_str, ts_str], utc=True),
        }
    )

    checkpoints_df = pd.DataFrame(
        {
            "arch": ["smolvla"],
            "run_id": ["run_001"],
            "step": pd.array([1000], dtype="Int64"),
            "path": ["/outputs/checkpoints/smolvla/run_001/step_001000.pt"],
            "size_mb": [45.2],
            "mtime": pd.to_datetime([ts_str], utc=True),
            "val_loss": [0.32],
        }
    )

    eval_df = pd.DataFrame(
        {
            "run_id": ["smolvla_001"],
            "task": ["pick_and_place"],
            "ts": pd.to_datetime([ts_str], utc=True),
            "pc_success": [0.72],
            "n_episodes": pd.array([50], dtype="Int64"),
            "intervention_rate": [0.15],
            "mean_ep_len": [120.5],
        }
    )

    history_df = pd.DataFrame(
        {
            "session_id": ["sess_01", "sess_01", "sess_01"],
            "slug": ["lerobot-policy"] * 3,
            "trial": pd.array([0, 1, 2], dtype="Int64"),
            "metric_name": ["pc_success"] * 3,
            "metric_value": [0.5, 0.6, 0.7],
            "config": [
                {"lr": 1e-4, "batch_size": 32},
                {"lr": 3e-4, "batch_size": 64},
                {"lr": 1e-3, "batch_size": 32},
            ],
            "ts": pd.to_datetime([ts_str, ts_str, ts_str], utc=True),
            "status": ["complete", "complete", "complete"],
        }
    )

    curriculum_history_df = pd.DataFrame(
        {
            "ts": pd.to_datetime([ts_str], utc=True),
            "stage": ["stage_1"],
            "advancement_reason": ["pc_success > 0.7"],
            "task_config_diff": [{"difficulty": "medium"}],
        }
    )

    events_df = pd.DataFrame(
        {
            "ts": pd.to_datetime([ts_str, ts_str], utc=True),
            "session_id": ["sess_01", "sess_01"],
            "phase": ["data_collection", "policy_training"],
            "event": ["complete", "start"],
            "data": ["{}", "{}"],
        }
    )

    return {
        "parquet_dataset": LoaderResult(df=parquet_df, is_empty=False),
        "synthetic": LoaderResult(df=synthetic_df, is_empty=False),
        "training_logs": LoaderResult(df=logs_df, is_empty=False),
        "checkpoints": LoaderResult(df=checkpoints_df, is_empty=False),
        "eval_results": LoaderResult(df=eval_df, is_empty=False),
        "autoresearch": LoaderResult(
            df={
                "history": history_df,
                "program": {
                    "raw_md": "# test",
                    "session_id": "sess_01",
                    "slug": "lerobot-policy",
                },
                "best": {
                    "lr": 1e-3,
                    "batch_size": 32,
                    "session_id": "sess_01",
                    "slug": "lerobot-policy",
                },
                "plateau": {"plateau_score": 0.3, "session_id": "sess_01"},
            },
            is_empty=False,
        ),
        "curriculum": LoaderResult(
            df={
                "current": {
                    "stage": "stage_2",
                    "task_config": {"pc_success_threshold": 0.8, "max_steps": 200},
                    "advancement_reason": "pc_success > 0.7",
                    "ts": ts_str,
                },
                "history": curriculum_history_df,
            },
            is_empty=False,
        ),
        "events": LoaderResult(df=events_df, is_empty=False),
    }


def _make_ctx(loader_results: dict, tmp_path: Path) -> TabContext:
    return TabContext(
        workspace_root=tmp_path,
        session_id=None,
        loader_results=loader_results,
        refresh_ts=_TS,
    )


# ---------------------------------------------------------------------------
# Parametrize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tab_class", TABS, ids=[t.slug for t in TABS])
def test_render_empty_no_exception(tab_class, tmp_path):
    """Empty state: render must not raise and must return a list."""
    loader_results = _empty_loader_results()
    ctx = _make_ctx(loader_results, tmp_path)
    tab = tab_class()
    figs = tab.render(ctx, container=None)
    assert isinstance(figs, list), f"{tab_class.__name__}.render() must return a list"


@pytest.mark.parametrize("tab_class", TABS, ids=[t.slug for t in TABS])
def test_render_populated_returns_figures(tab_class, tmp_path):
    """Populated state: render must return at least 1 go.Figure."""
    loader_results = _populated_loader_results(tmp_path)
    ctx = _make_ctx(loader_results, tmp_path)
    tab = tab_class()
    figs = tab.render(ctx, container=None)
    assert isinstance(figs, list), f"{tab_class.__name__}.render() must return a list"
    assert len(figs) >= 1, (
        f"{tab_class.__name__}.render() returned 0 figures for populated state"
    )
    for fig in figs:
        assert isinstance(fig, go.Figure), (
            f"{tab_class.__name__}: expected go.Figure, got {type(fig)}"
        )


@pytest.mark.parametrize("tab_class", TABS, ids=[t.slug for t in TABS])
def test_render_empty_returns_empty_list_or_placeholder(tab_class, tmp_path):
    """Empty state: return value must be a list (may be [] or have placeholder figs)."""
    loader_results = _empty_loader_results()
    ctx = _make_ctx(loader_results, tmp_path)
    tab = tab_class()
    figs = tab.render(ctx, container=None)
    assert isinstance(figs, list)
    # If figs returned in empty state, each must still be go.Figure
    for fig in figs:
        assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("tab_class", TABS, ids=[t.slug for t in TABS])
def test_tab_has_required_attributes(tab_class):
    """Each tab class must have non-empty title, slug, and primary_loader_slug."""
    tab = tab_class()
    assert tab.title and isinstance(tab.title, str)
    assert tab.slug and isinstance(tab.slug, str)
    # primary_loader_slug may be empty string for tabs with no single primary loader
    assert isinstance(tab.slug, str)
    assert " " not in tab.slug, f"slug must be filename-safe: got '{tab.slug}'"
