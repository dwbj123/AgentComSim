"""Experiment orchestration and confidence-interval aggregation."""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

from agentcomsim.config import ExperimentConfig
from agentcomsim.engine import SimulationEngine
from agentcomsim.models import RunResult
from agentcomsim.policies.base import SchedulingPolicy
from agentcomsim.policies.rules import FIFOPolicy, StaticPriorityPolicy, TUSPolicy
from agentcomsim.trace import generate_trace

PolicyFactory = Callable[[], SchedulingPolicy]


def default_policies(config: ExperimentConfig) -> dict[str, PolicyFactory]:
    return {
        "fifo": FIFOPolicy,
        "static": StaticPriorityPolicy,
        "tus": lambda: TUSPolicy(config.tus),
    }


def run_experiment(
    config: ExperimentConfig,
    policies: dict[str, PolicyFactory] | None = None,
    *,
    progress: Callable[[int, int, str, float], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[RunResult]]:
    """Run all policy/load/seed combinations using common pre-generated traces."""

    available = default_policies(config) if policies is None else policies
    unknown = set(config.policies) - set(available)
    if unknown:
        raise ValueError(f"Unknown policies: {sorted(unknown)}")
    results: list[RunResult] = []
    total = len(config.arrival_rates) * len(config.policies) * config.runs
    completed = 0
    for arrival_rate in config.arrival_rates:
        for run_index in range(config.runs):
            seed = config.base_seed + run_index
            trace = generate_trace(config, arrival_rate, seed)
            for policy_name in config.policies:
                policy = available[policy_name]()
                result = SimulationEngine(config, policy).run(
                    trace, arrival_rate=arrival_rate, seed=seed
                )
                results.append(result)
                completed += 1
                if progress:
                    progress(completed, total, policy_name, arrival_rate)
    raw = pd.DataFrame([result.to_dict() for result in results])
    return raw, summarize(raw), results


def _mean_ci(series: pd.Series, z: float = 1.96) -> tuple[float, float]:
    values = series.to_numpy(dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    return mean, float(z * values.std(ddof=1) / math.sqrt(len(values)))


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate independent runs; CI columns refer to the mean, not tasks."""

    rows = []
    metrics = [
        "task_success_rate",
        "mean_latency_s",
        "p95_latency_s",
        "mean_semantic_quality",
        "total_bytes",
        "compressed_perception_ratio",
    ]
    for (arrival_rate, policy), group in raw.groupby(["arrival_rate", "policy"], sort=True):
        row: dict[str, float | int | str] = {
            "arrival_rate": float(arrival_rate),
            "policy": str(policy),
            "runs": len(group),
        }
        for metric in metrics:
            mean, ci = _mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows)
