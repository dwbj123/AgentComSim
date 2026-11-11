from __future__ import annotations

import pandas as pd
import pytest

from agentcomsim.experiment import run_experiment, summarize


def test_experiment_matrix_and_shared_bytes(small_config):
    small_config.record_events = False
    raw, summary, results = run_experiment(small_config)
    assert len(raw) == 6
    assert len(results) == 6
    assert len(summary) == 3
    for seed, group in raw.groupby("seed"):
        baseline = group[group["policy"] == "fifo"]["total_bytes"].iloc[0]
        static = group[group["policy"] == "static"]["total_bytes"].iloc[0]
        assert baseline == pytest.approx(static), (
            f"Policies received different work for seed {seed}"
        )
    assert raw["task_success_rate"].between(0, 1).all()


def test_unknown_policy_fails_before_simulation(small_config):
    small_config.policies = ("missing",)
    with pytest.raises(ValueError, match="Unknown policies"):
        run_experiment(small_config)


def test_summary_confidence_interval_uses_independent_runs():
    raw = pd.DataFrame(
        {
            "arrival_rate": [1.0, 1.0],
            "policy": ["fifo", "fifo"],
            "task_success_rate": [0.8, 1.0],
            "mean_latency_s": [1.0, 1.0],
            "p95_latency_s": [1.2, 1.2],
            "mean_semantic_quality": [1.0, 1.0],
            "total_bytes": [100.0, 100.0],
            "compressed_perception_ratio": [0.0, 0.0],
        }
    )
    row = summarize(raw).iloc[0]
    assert row["runs"] == 2
    assert row["task_success_rate_mean"] == pytest.approx(0.9)
    assert row["task_success_rate_ci95"] > 0
    assert row["mean_latency_s_ci95"] == 0
