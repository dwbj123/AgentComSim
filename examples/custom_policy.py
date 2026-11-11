"""Run the built-in FIFO baseline against earliest-deadline-first scheduling."""

from pathlib import Path

from agentcomsim.config import ExperimentConfig
from agentcomsim.experiment import run_experiment
from agentcomsim.models import SchedulingContext
from agentcomsim.policies.base import SchedulingPolicy
from agentcomsim.policies.rules import FIFOPolicy
from agentcomsim.reporting import save_experiment


class EarliestDeadlineFirst(SchedulingPolicy):
    name = "edf"

    def priority(self, context: SchedulingContext) -> float:
        return context.task.deadline_s


if __name__ == "__main__":
    config = ExperimentConfig(
        duration_s=45,
        warmup_s=5,
        runs=5,
        arrival_rates=(0.8, 1.4),
        policies=("fifo", "edf"),
    )
    raw, summary, results = run_experiment(
        config,
        {"fifo": FIFOPolicy, "edf": EarliestDeadlineFirst},
    )
    output = Path("results/custom-policy")
    save_experiment(output, config, raw, summary, results)
    print(summary.to_string(index=False))
