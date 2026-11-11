"""FIFO, static-priority, and task/semantic utility-aware policies."""

from __future__ import annotations

import math

from agentcomsim.config import TUSConfig
from agentcomsim.models import EventKind, Modality, SchedulingContext
from agentcomsim.policies.base import SchedulingPolicy

_CLASS_ORDER = {
    EventKind.CONTROL: 0,
    EventKind.PLAN: 1,
    EventKind.TOOL: 2,
    EventKind.SENSE: 3,
}


class FIFOPolicy(SchedulingPolicy):
    name = "fifo"

    def priority(self, context: SchedulingContext) -> float:
        return 0.0


class StaticPriorityPolicy(SchedulingPolicy):
    name = "static"

    def priority(self, context: SchedulingContext) -> float:
        return float(_CLASS_ORDER[context.kind])


class TUSPolicy(SchedulingPolicy):
    """Task criticality and semantic utility-aware scheduling."""

    name = "tus"

    def __init__(self, config: TUSConfig | None = None) -> None:
        self.config = config or TUSConfig()

    def remaining_path_s(self, context: SchedulingContext) -> float:
        """Estimate remaining transmission and computing work from this stage."""

        task = context.task
        bandwidth_bps = context.bandwidth_mbps * 1_000_000
        if context.kind == EventKind.SENSE:
            return task.sense_bytes * 8 / bandwidth_bps + 0.18 + 0.10
        if context.kind == EventKind.TOOL:
            return 0.12 + 0.10
        if context.kind == EventKind.PLAN:
            return 0.14 + 0.01
        return 0.01

    def features(self, context: SchedulingContext) -> dict[str, float]:
        remaining = self.remaining_path_s(context)
        slack = context.task.deadline_s - context.now_s - remaining
        total_window = context.task.relative_deadline_s
        urgency = min(3.0, max(0.0, (total_window - slack) / total_window))
        criticality = min(2.0, remaining / 0.35)
        size_kb = max(context.size_bytes / 1024, 1.0)
        density = min(1.5, math.log1p(1000 * context.task.semantic_quality / size_kb) / 4.0)
        class_bonus = {
            EventKind.CONTROL: self.config.control_bonus,
            EventKind.PLAN: self.config.plan_bonus,
            EventKind.TOOL: self.config.tool_bonus,
            EventKind.SENSE: 0.0,
        }[context.kind]
        return {
            "remaining_path_s": remaining,
            "slack_s": slack,
            "urgency": urgency,
            "criticality": criticality,
            "semantic_density": density,
            "class_bonus": class_bonus,
        }

    def score(self, context: SchedulingContext) -> float:
        f = self.features(context)
        return (
            self.config.critical_weight * f["criticality"]
            + self.config.urgency_weight * f["urgency"]
            + self.config.density_weight * f["semantic_density"]
            + f["class_bonus"]
        )

    def priority(self, context: SchedulingContext) -> float:
        return -self.score(context)

    def should_compress(self, context: SchedulingContext) -> bool:
        if context.kind != EventKind.SENSE or context.task.modality == Modality.TEXT:
            return False
        queue_depth = context.network_queue + context.network_busy
        slack = context.task.deadline_s - context.now_s - self.remaining_path_s(context)
        return queue_depth >= self.config.queue_threshold or slack < self.config.slack_threshold_s

    def compression_parameters(self, modality: Modality) -> tuple[float, float]:
        if modality == Modality.VIDEO:
            return self.config.video_byte_retention, self.config.video_quality_factor
        if modality == Modality.IMAGE:
            return self.config.image_byte_retention, self.config.image_quality_factor
        return 1.0, 1.0
