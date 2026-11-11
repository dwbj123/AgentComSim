from __future__ import annotations

from agentcomsim.trace import generate_trace


def test_trace_is_deterministic(small_config):
    first = generate_trace(small_config, 1.2, 99)
    second = generate_trace(small_config, 1.2, 99)
    assert first == second
    assert first
    assert [task.task_id for task in first] == list(range(1, len(first) + 1))
    assert all(task.deadline_s > task.arrival_s for task in first)
    assert all(task.sense_bytes > 0 for task in first)


def test_seed_and_rate_change_trace(small_config):
    baseline = generate_trace(small_config, 1.0, 99)
    other_seed = generate_trace(small_config, 1.0, 100)
    higher_rate = generate_trace(small_config, 2.0, 99)
    assert baseline != other_seed
    assert len(higher_rate) > len(baseline)
