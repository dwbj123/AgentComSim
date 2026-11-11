from __future__ import annotations

from collections.abc import Callable

import pytest

from agentcomsim.config import ExperimentConfig
from agentcomsim.models import EventKind, Modality, SchedulingContext, TaskSpec


@pytest.fixture
def small_config() -> ExperimentConfig:
    return ExperimentConfig(
        duration_s=16,
        warmup_s=2,
        runs=2,
        base_seed=1234,
        arrival_rates=(1.0,),
        record_events=True,
    )


@pytest.fixture
def task_factory() -> Callable[..., TaskSpec]:
    def make(
        task_id: int = 1,
        arrival_s: float = 3.0,
        modality: Modality = Modality.TEXT,
        sense_bytes: float = 20_000,
        deadline_s: float | None = None,
    ) -> TaskSpec:
        return TaskSpec(
            task_id=task_id,
            arrival_s=arrival_s,
            deadline_s=deadline_s if deadline_s is not None else arrival_s + 5,
            modality=modality,
            sense_bytes=sense_bytes,
            tool_bytes=2_000,
            plan_bytes=1_000,
            control_bytes=500,
            tool_work_s=0.01,
            plan_work_s=0.02,
        )

    return make


@pytest.fixture
def context_factory() -> Callable[..., SchedulingContext]:
    def make(
        task: TaskSpec,
        kind: EventKind = EventKind.SENSE,
        now_s: float = 3.0,
        network_queue: int = 0,
        network_busy: int = 0,
    ) -> SchedulingContext:
        return SchedulingContext(
            now_s=now_s,
            task=task,
            kind=kind,
            size_bytes=task.sense_bytes,
            network_queue=network_queue,
            network_busy=network_busy,
            compute_queue=0,
            compute_busy=0,
            bandwidth_mbps=25,
        )

    return make
