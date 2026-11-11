"""SimPy-based message-level co-simulation of task, network, and compute events."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from dataclasses import asdict

import numpy as np
import simpy

from agentcomsim.config import ExperimentConfig
from agentcomsim.models import EventKind, EventRecord, RunResult, SchedulingContext, TaskSpec
from agentcomsim.policies.base import SchedulingPolicy


class SimulationEngine:
    """Run the reference task graph under a pluggable scheduling policy."""

    def __init__(self, config: ExperimentConfig, policy: SchedulingPolicy) -> None:
        self.config = config
        self.policy = policy

    def run(
        self,
        trace: Iterable[TaskSpec],
        *,
        arrival_rate: float,
        seed: int,
        record_events: bool | None = None,
    ) -> RunResult:
        tasks = copy.deepcopy(list(trace))
        tasks.sort(key=lambda task: task.arrival_s)
        record = self.config.record_events if record_events is None else record_events
        events: list[EventRecord] = []
        env = simpy.Environment()
        network = simpy.PriorityResource(env, capacity=1)
        compute = simpy.PriorityResource(env, capacity=self.config.compute.workers)
        counters = {"bytes": 0.0, "compressed": 0, "perception": 0}

        def context(task: TaskSpec, kind: EventKind, size: float) -> SchedulingContext:
            return SchedulingContext(
                now_s=env.now,
                task=task,
                kind=kind,
                size_bytes=size,
                network_queue=len(network.queue),
                network_busy=network.count,
                compute_queue=len(compute.queue),
                compute_busy=compute.count,
                bandwidth_mbps=self.config.network.bandwidth_mbps,
            )

        def transmit(task: TaskSpec, kind: EventKind, size: float, parent: str):
            if kind == EventKind.SENSE:
                counters["perception"] += 1
                if self.policy.should_compress(context(task, kind, size)):
                    retention, quality = self.policy.compression_parameters(task.modality)
                    size *= retention
                    task.semantic_quality *= quality
                    task.compressed = True
                    counters["compressed"] += 1
            priority = float(self.policy.priority(context(task, kind, size)))
            enqueue = env.now
            cause_id = f"task-{task.task_id}-{kind}-network"
            with network.request(priority=priority) as request:
                yield request
                start = env.now
                counters["bytes"] += size
                duration = self.config.network.base_delay_s + size * 8 / (
                    self.config.network.bandwidth_mbps * 1_000_000
                )
                yield env.timeout(duration)
                if record:
                    events.append(
                        EventRecord(
                            task.task_id,
                            cause_id,
                            parent,
                            kind.value,
                            "network",
                            enqueue,
                            start,
                            env.now,
                            size,
                            priority,
                        )
                    )

        def execute(task: TaskSpec, kind: EventKind, work: float, size: float, parent: str):
            priority = float(self.policy.priority(context(task, kind, max(size, 1024))))
            enqueue = env.now
            cause_id = f"task-{task.task_id}-{kind}-compute"
            with compute.request(priority=priority) as request:
                yield request
                start = env.now
                yield env.timeout(work)
                if record:
                    events.append(
                        EventRecord(
                            task.task_id,
                            cause_id,
                            parent,
                            kind.value,
                            "compute",
                            enqueue,
                            start,
                            env.now,
                            size,
                            priority,
                        )
                    )

        def tool_branch(task: TaskSpec):
            parent = f"task-{task.task_id}"
            yield env.process(transmit(task, EventKind.TOOL, task.tool_bytes, parent))
            yield env.process(
                execute(
                    task,
                    EventKind.TOOL,
                    task.tool_work_s,
                    task.tool_bytes,
                    f"task-{task.task_id}-tool-network",
                )
            )

        def task_process(task: TaskSpec):
            root = f"task-{task.task_id}"
            sensing = env.process(transmit(task, EventKind.SENSE, task.sense_bytes, root))
            tooling = env.process(tool_branch(task))
            yield sensing
            work = self.config.compute.sense_base_s + self.config.compute.sense_seconds_per_mb * (
                task.sense_bytes / 1_000_000
            )
            if task.compressed:
                work *= self.config.compute.compressed_work_factor
            if task.modality == "image":
                work *= 0.72
            elif task.modality == "text":
                work *= 0.35
            yield env.process(
                execute(
                    task,
                    EventKind.SENSE,
                    work,
                    task.sense_bytes,
                    f"task-{task.task_id}-sense-network",
                )
            )
            yield tooling
            plan_parent = f"task-{task.task_id}-join"
            yield env.process(transmit(task, EventKind.PLAN, task.plan_bytes, plan_parent))
            yield env.process(
                execute(
                    task,
                    EventKind.PLAN,
                    task.plan_work_s,
                    task.plan_bytes,
                    f"task-{task.task_id}-plan-network",
                )
            )
            yield env.process(
                transmit(
                    task, EventKind.CONTROL, task.control_bytes, f"task-{task.task_id}-plan-compute"
                )
            )
            task.finish_s = env.now

        def arrivals():
            previous = 0.0
            for task in tasks:
                yield env.timeout(max(0.0, task.arrival_s - previous))
                previous = task.arrival_s
                env.process(task_process(task))

        env.process(arrivals())
        env.run()

        measured = [task for task in tasks if task.arrival_s >= self.config.warmup_s]
        completed = [task for task in measured if task.finish_s is not None]
        latencies = [
            task.finish_s - task.arrival_s for task in completed if task.finish_s is not None
        ]
        successful = [
            task
            for task in completed
            if task.finish_s is not None
            and task.finish_s <= task.deadline_s
            and task.semantic_quality >= self.config.workload.minimum_quality
        ]
        return RunResult(
            policy=self.policy.name,
            arrival_rate=arrival_rate,
            seed=seed,
            task_success_rate=len(successful) / max(1, len(measured)),
            mean_latency_s=float(np.mean(latencies)) if latencies else math.nan,
            p95_latency_s=float(np.percentile(latencies, 95)) if latencies else math.nan,
            mean_semantic_quality=float(np.mean([t.semantic_quality for t in completed]))
            if completed
            else math.nan,
            total_bytes=counters["bytes"],
            compressed_perception_ratio=counters["compressed"] / max(1, counters["perception"]),
            arrived_tasks=len(measured),
            completed_tasks=len(completed),
            events=events,
        )


def events_as_dicts(result: RunResult) -> list[dict]:
    """Convert an event trace to JSON-serializable records."""

    return [asdict(event) for event in result.events]
