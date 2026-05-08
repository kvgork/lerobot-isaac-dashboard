"""tabs — Dashboard tab modules for lerobot-isaac-dashboard.

Each module implements a :class:`Tab` subclass with a ``render(ctx, *, container)``
method that returns a ``list[go.Figure]``.

The :data:`TABS` registry lists all 8 pipeline tabs in display order (1..8).
``container=None`` triggers the static-export path; passing a Streamlit container
triggers the live path that also calls ``container.plotly_chart``.

Tab ordering
------------
1. DataCollectionTab   — LeRobot v3 dataset summaries
2. SyntheticTab        — real vs sim_dr vs mimicgen breakdown
3. PolicyTrainingTab   — loss curves + checkpoint inventory
4. WorldModelTab       — DreamerV3 / LeWM loss curves
5. EvaluationTab       — pc_success curves + plateau detection
6. AutoresearchTab     — HP search history + plateau gauge
7. CurriculumTab       — stage timeline + advancement triggers
8. PipelineHealthTab   — event log + error table + workspace checklist
"""

from lerobot_isaac_dashboard.tabs._base import Tab, TabContext
from lerobot_isaac_dashboard.tabs._kpis import render_kpi_row
from lerobot_isaac_dashboard.tabs.autoresearch import AutoresearchTab
from lerobot_isaac_dashboard.tabs.curriculum import CurriculumTab
from lerobot_isaac_dashboard.tabs.data_collection import DataCollectionTab
from lerobot_isaac_dashboard.tabs.evaluation import EvaluationTab
from lerobot_isaac_dashboard.tabs.pipeline_health import PipelineHealthTab
from lerobot_isaac_dashboard.tabs.policy_training import PolicyTrainingTab
from lerobot_isaac_dashboard.tabs.synthetic import SyntheticTab
from lerobot_isaac_dashboard.tabs.world_model import WorldModelTab

TABS: list[type[Tab]] = [
    DataCollectionTab,
    SyntheticTab,
    PolicyTrainingTab,
    WorldModelTab,
    EvaluationTab,
    AutoresearchTab,
    CurriculumTab,
    PipelineHealthTab,
]

__all__ = [
    "Tab",
    "TabContext",
    "render_kpi_row",
    "TABS",
    "DataCollectionTab",
    "SyntheticTab",
    "PolicyTrainingTab",
    "WorldModelTab",
    "EvaluationTab",
    "AutoresearchTab",
    "CurriculumTab",
    "PipelineHealthTab",
]
