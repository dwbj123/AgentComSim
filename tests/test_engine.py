from __future__ import annotations

import math

from agentcomsim.engine import SimulationEngine, events_as_dicts
from agentcomsim.policies.rules import FIFOPolicy


def test_reference_task_graph_completes(small_config, task_factory):
    task = task_factory()
    original = task.to_dict()
    result = SimulationEngine(small_config, FIFOPolicy()).run([task], arrival_rate=1.0, seed=7)
    assert result.completed_tasks == 1
    assert result.task_success_rate == 1.0
    assert len(result.events) == 7
    assert result.total_bytes == sum(
        [task.sense_bytes, task.tool_bytes, task.plan_bytes, task.control_bytes]
    )
    assert task.to_dict() == original, "The engine must not mutate the shared input trace"
    assert all(event.start_s >= event.enqueue_s for event in result.events)
    assert all(event.end_s >= event.start_s for event in result.events)
    assert len({event.cause_id for event in result.events}) == 7
    assert events_as_dicts(result)[0]["task_id"] == 1


def test_warmup_tasks_are_excluded_from_task_metrics(small_config, task_factory):
    warmup_task = task_factory(task_id=1, arrival_s=0.5)
    measured_task = task_factory(task_id=2, arrival_s=3.0)
    result = SimulationEngine(small_config, FIFOPolicy()).run(
        [warmup_task, measured_task], arrival_rate=1.0, seed=7, record_events=False
    )
    assert result.arrived_tasks == 1
    assert result.completed_tasks == 1
    assert len(result.events) == 0


def test_empty_trace_has_explicit_empty_metrics(small_config):
    result = SimulationEngine(small_config, FIFOPolicy()).run([], arrival_rate=1.0, seed=7)
    assert result.arrived_tasks == 0
    assert result.task_success_rate == 0
    assert math.isnan(result.mean_latency_s)
