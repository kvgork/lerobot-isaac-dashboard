"""loaders — Metric loader modules for lerobot-isaac-dashboard.

Each loader is a pure function that reads one category of pipeline artefacts
and returns a ``LoaderResult``.  Loaders never raise on missing files; they
return an empty DataFrame with the canonical column schema and is_empty=True.

Public exports
--------------
LoaderResult          — return type for all load_* functions
load_parquet_dataset  — LeRobot v3 dataset summaries
load_eval_results     — outputs/eval/*.json
load_checkpoints      — outputs/checkpoints/<arch>/<run_id>/
load_training_logs    — outputs/checkpoints/<arch>/<run_id>/log.txt
load_autoresearch     — .agent-state/*/autoresearch/<slug>/
load_events           — .agent-state/<sessionId>/events.jsonl
load_curriculum       — outputs/curriculum_stage.json + curriculum_history.jsonl
load_synthetic        — merged dataset meta/episodes.parquet source breakdown
load_paths            — WorkspacePaths resolver (returns WorkspacePaths, not LoaderResult)
"""

from lerobot_isaac_dashboard.loaders._base import LoaderResult
from lerobot_isaac_dashboard.loaders.autoresearch import load_autoresearch
from lerobot_isaac_dashboard.loaders.checkpoints import load_checkpoints
from lerobot_isaac_dashboard.loaders.curriculum import load_curriculum
from lerobot_isaac_dashboard.loaders.eval_results import load_eval_results
from lerobot_isaac_dashboard.loaders.events import load_events
from lerobot_isaac_dashboard.loaders.parquet_dataset import load_parquet_dataset
from lerobot_isaac_dashboard.loaders.paths import WorkspacePaths, load_paths
from lerobot_isaac_dashboard.loaders.synthetic import load_synthetic
from lerobot_isaac_dashboard.loaders.training_logs import load_training_logs

__all__ = [
    "LoaderResult",
    "WorkspacePaths",
    "load_autoresearch",
    "load_checkpoints",
    "load_curriculum",
    "load_eval_results",
    "load_events",
    "load_parquet_dataset",
    "load_paths",
    "load_synthetic",
    "load_training_logs",
]
