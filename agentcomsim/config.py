"""Typed, validated YAML configuration for reproducible experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class NetworkConfig:
    bandwidth_mbps: float = 25.0
    base_delay_s: float = 0.004

    def __post_init__(self) -> None:
        if self.bandwidth_mbps <= 0:
            raise ValueError("network.bandwidth_mbps must be positive")
        if self.base_delay_s < 0:
            raise ValueError("network.base_delay_s cannot be negative")


@dataclass(slots=True)
class ComputeConfig:
    workers: int = 1
    sense_base_s: float = 0.075
    sense_seconds_per_mb: float = 0.055
    compressed_work_factor: float = 0.72

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("compute.workers must be at least one")


@dataclass(slots=True)
class WorkloadConfig:
    video_fraction: float = 0.62
    image_fraction: float = 0.25
    video_median_mb: float = 0.72
    video_lognormal_sigma: float = 0.38
    image_median_mb: float = 0.24
    image_lognormal_sigma: float = 0.35
    deadline_min_s: float = 1.25
    deadline_max_s: float = 1.85
    minimum_quality: float = 0.85

    def __post_init__(self) -> None:
        if not 0 < self.video_fraction + self.image_fraction <= 1:
            raise ValueError("video_fraction + image_fraction must be in (0, 1]")
        if not 0 < self.deadline_min_s < self.deadline_max_s:
            raise ValueError("deadline bounds are invalid")
        if not 0 <= self.minimum_quality <= 1:
            raise ValueError("minimum_quality must be in [0, 1]")


@dataclass(slots=True)
class TUSConfig:
    critical_weight: float = 0.9
    urgency_weight: float = 1.25
    density_weight: float = 0.35
    control_bonus: float = 0.5
    plan_bonus: float = 0.25
    tool_bonus: float = 0.05
    queue_threshold: int = 2
    slack_threshold_s: float = 0.32
    video_byte_retention: float = 0.38
    image_byte_retention: float = 0.55
    video_quality_factor: float = 0.93
    image_quality_factor: float = 0.96


@dataclass(slots=True)
class ExperimentConfig:
    duration_s: float = 240.0
    warmup_s: float = 30.0
    runs: int = 30
    base_seed: int = 202600
    arrival_rates: tuple[float, ...] = (0.8, 1.2, 1.6)
    policies: tuple[str, ...] = ("fifo", "static", "tus")
    record_events: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    tus: TUSConfig = field(default_factory=TUSConfig)

    def __post_init__(self) -> None:
        if not 0 <= self.warmup_s < self.duration_s:
            raise ValueError("warmup_s must be non-negative and smaller than duration_s")
        if self.runs < 1:
            raise ValueError("runs must be at least one")
        if not self.arrival_rates or any(x <= 0 for x in self.arrival_rates):
            raise ValueError("arrival_rates must contain positive values")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _construct(cls: type[Any], values: dict[str, Any] | None) -> Any:
    return cls(**(values or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a configuration without executing arbitrary YAML tags."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    return ExperimentConfig(
        duration_s=raw.get("duration_s", 240.0),
        warmup_s=raw.get("warmup_s", 30.0),
        runs=raw.get("runs", 30),
        base_seed=raw.get("base_seed", 202600),
        arrival_rates=tuple(raw.get("arrival_rates", [0.8, 1.2, 1.6])),
        policies=tuple(raw.get("policies", ["fifo", "static", "tus"])),
        record_events=raw.get("record_events", False),
        network=_construct(NetworkConfig, raw.get("network")),
        compute=_construct(ComputeConfig, raw.get("compute")),
        workload=_construct(WorkloadConfig, raw.get("workload")),
        tus=_construct(TUSConfig, raw.get("tus")),
    )
