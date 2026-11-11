"""Common-random-number workload generation."""

from __future__ import annotations

import math
import random

from agentcomsim.config import ExperimentConfig
from agentcomsim.models import Modality, TaskSpec

MB = 1_000_000.0
KB = 1_000.0


def generate_trace(config: ExperimentConfig, arrival_rate: float, seed: int) -> list[TaskSpec]:
    """Pre-generate every random variable so all policies see identical work."""

    rng = random.Random(seed)
    workload = config.workload
    tasks: list[TaskSpec] = []
    arrival = rng.expovariate(arrival_rate)
    task_id = 0
    while arrival < config.duration_s:
        task_id += 1
        draw = rng.random()
        if draw < workload.video_fraction:
            modality = Modality.VIDEO
            size = min(
                max(
                    rng.lognormvariate(
                        math.log(workload.video_median_mb * MB), workload.video_lognormal_sigma
                    ),
                    0.25 * MB,
                ),
                1.8 * MB,
            )
        elif draw < workload.video_fraction + workload.image_fraction:
            modality = Modality.IMAGE
            size = min(
                max(
                    rng.lognormvariate(
                        math.log(workload.image_median_mb * MB), workload.image_lognormal_sigma
                    ),
                    0.08 * MB,
                ),
                0.7 * MB,
            )
        else:
            modality = Modality.TEXT
            size = rng.uniform(8, 30) * 1024
        tasks.append(
            TaskSpec(
                task_id=task_id,
                arrival_s=arrival,
                deadline_s=arrival + rng.uniform(workload.deadline_min_s, workload.deadline_max_s),
                modality=modality,
                sense_bytes=size,
                tool_bytes=rng.uniform(8, 40) * 1024,
                plan_bytes=rng.uniform(6, 14) * 1024,
                control_bytes=rng.uniform(1.5, 3.5) * 1024,
                tool_work_s=rng.uniform(0.025, 0.055),
                plan_work_s=rng.uniform(0.075, 0.14),
            )
        )
        arrival += rng.expovariate(arrival_rate)
    return tasks
