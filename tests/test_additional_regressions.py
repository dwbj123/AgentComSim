from __future__ import annotations

import pytest

from agentcomsim.engine import SimulationEngine
from agentcomsim.neural.dataset import make_dataset
from agentcomsim.policies.rules import FIFOPolicy


def test_engine_accepts_unsorted_tasks(small_config, task_factory):
    later = task_factory(task_id=1, arrival_s=6.0)
    earlier = task_factory(task_id=2, arrival_s=3.0)
    result = SimulationEngine(small_config, FIFOPolicy()).run(
        [later, earlier], arrival_rate=1.0, seed=9
    )
    sense_events = [event for event in result.events if event.kind == "sense"]
    first_network = min(event.enqueue_s for event in sense_events)
    assert first_network == pytest.approx(3.0)
    assert result.completed_tasks == 2


def test_duplicate_neural_seeds_are_rejected(small_config):
    with pytest.raises(ValueError, match="must be unique"):
        make_dataset(small_config, [1, 1, 2, 3, 4])
