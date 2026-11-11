"""Offline imitation dataset collected from actual simulator queue states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentcomsim.config import ExperimentConfig
from agentcomsim.engine import SimulationEngine
from agentcomsim.models import SchedulingContext
from agentcomsim.neural.features import extract_features
from agentcomsim.policies.rules import TUSPolicy
from agentcomsim.trace import generate_trace


class _CollectorPolicy(TUSPolicy):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config.tus)
        self.feature_rows: list[np.ndarray] = []
        self.targets: list[float] = []

    def priority(self, context: SchedulingContext) -> float:
        self.feature_rows.append(extract_features(context, self))
        self.targets.append(self.score(context) / 4.0)
        return super().priority(context)


@dataclass(slots=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def collect_samples(config: ExperimentConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect one seed-specific sample set; split seeds, never individual rows."""

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for arrival_rate in config.arrival_rates:
        policy = _CollectorPolicy(config)
        trace = generate_trace(config, arrival_rate, seed)
        SimulationEngine(config, policy).run(trace, arrival_rate=arrival_rate, seed=seed)
        x_parts.append(np.stack(policy.feature_rows))
        y_parts.append(np.asarray(policy.targets, dtype=np.float32))
    return np.concatenate(x_parts), np.concatenate(y_parts)


def make_dataset(config: ExperimentConfig, seeds: list[int]) -> DatasetSplit:
    """Create deterministic 60/20/20 splits by independent simulation seed."""

    if len(seeds) < 5:
        raise ValueError("At least five seeds are required for train/validation/test splitting")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Dataset seeds must be unique to prevent train/test trajectory leakage")
    shuffled = list(seeds)
    np.random.default_rng(1729).shuffle(shuffled)
    train_end = max(1, int(len(shuffled) * 0.6))
    validation_end = max(train_end + 1, int(len(shuffled) * 0.8))
    if validation_end >= len(shuffled):
        validation_end = len(shuffled) - 1
    groups = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    collected = {
        name: [collect_samples(config, seed) for seed in group] for name, group in groups.items()
    }

    def join(name: str, index: int) -> np.ndarray:
        return np.concatenate([item[index] for item in collected[name]]).astype(np.float32)

    return DatasetSplit(
        join("train", 0),
        join("train", 1),
        join("validation", 0),
        join("validation", 1),
        join("test", 0),
        join("test", 1),
    )
