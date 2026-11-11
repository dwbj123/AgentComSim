from __future__ import annotations

import json

from agentcomsim.experiment import run_experiment
from agentcomsim.reporting import save_experiment


def test_save_experiment_writes_reproducibility_bundle(tmp_path, small_config):
    small_config.runs = 1
    small_config.policies = ("fifo",)
    raw, summary, results = run_experiment(small_config)
    save_experiment(tmp_path, small_config, raw, summary, results)
    expected = {
        "config.resolved.yaml",
        "runs.csv",
        "summary.csv",
        "summary.json",
        "performance.png",
        "events.jsonl",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert (tmp_path / "performance.png").stat().st_size > 1000
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert lines
    assert {"task_id", "cause_id", "parent_cause_id"} <= set(json.loads(lines[0]))
