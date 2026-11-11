"""Domain objects shared by the simulator, policies, and integrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Modality(StrEnum):
    """Payload modalities supported by the message-level workload generator."""

    VIDEO = "video"
    IMAGE = "image"
    TEXT = "text"


class EventKind(StrEnum):
    """Stages in the reference emergency-response task graph."""

    SENSE = "sense"
    TOOL = "tool"
    PLAN = "plan"
    CONTROL = "control"


@dataclass(slots=True)
class AgentProfile:
    """Description of an agent that participates in a task group."""

    agent_id: str
    capabilities: tuple[str, ...]
    modalities: tuple[Modality, ...]
    protocols: tuple[str, ...] = ("A2A", "MCP")
    owner: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskSpec:
    """A fully sampled task used to guarantee common random numbers."""

    task_id: int
    arrival_s: float
    deadline_s: float
    modality: Modality
    sense_bytes: float
    tool_bytes: float
    plan_bytes: float
    control_bytes: float
    tool_work_s: float
    plan_work_s: float
    semantic_quality: float = 1.0
    compressed: bool = False
    finish_s: float | None = None

    @property
    def relative_deadline_s(self) -> float:
        return self.deadline_s - self.arrival_s

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["modality"] = self.modality.value
        return result


@dataclass(slots=True, frozen=True)
class SchedulingContext:
    """Observable state supplied to a scheduling policy."""

    now_s: float
    task: TaskSpec
    kind: EventKind
    size_bytes: float
    network_queue: int
    network_busy: int
    compute_queue: int
    compute_busy: int
    bandwidth_mbps: float


@dataclass(slots=True, frozen=True)
class EventRecord:
    """A causal event emitted by the simulation engine."""

    task_id: int
    cause_id: str
    parent_cause_id: str | None
    kind: str
    resource: str
    enqueue_s: float
    start_s: float
    end_s: float
    size_bytes: float
    priority: float


@dataclass(slots=True)
class RunResult:
    """Metrics and optional event trace produced by one simulation run."""

    policy: str
    arrival_rate: float
    seed: int
    task_success_rate: float
    mean_latency_s: float
    p95_latency_s: float
    mean_semantic_quality: float
    total_bytes: float
    compressed_perception_ratio: float
    arrived_tasks: int
    completed_tasks: int
    events: list[EventRecord] = field(default_factory=list)

    def to_dict(self, include_events: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_events:
            result.pop("events")
        return result
