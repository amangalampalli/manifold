"""Trainer composition."""

from __future__ import annotations

from manifold.trainer.core.control import TrainerControlMixin
from manifold.trainer.core.evaluation import TrainerEvaluationMixin
from manifold.trainer.core.lifecycle import TrainerLifecycleMixin


class MinimaxTrainer(
    TrainerLifecycleMixin, TrainerEvaluationMixin, TrainerControlMixin
):
    """Alternating source/controller optimization for synthetic trajectories."""
