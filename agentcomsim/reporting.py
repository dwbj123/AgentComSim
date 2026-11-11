"""Persistent experiment artifacts and publication-ready plots."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from agentcomsim.config import ExperimentConfig
from agentcomsim.models import RunResult

POLICY_LABELS = {"fifo": "FIFO", "static": "Static priority", "tus": "TUS", "neural": "Neural"}
POLICY_COLORS = {"fifo": "#74838a", "static": "#a6784f", "tus": "#19777b", "neural": "#715f8d"}


def save_experiment(
    output_dir: str | Path,
    config: ExperimentConfig,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    results: list[RunResult] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config.save_yaml(output / "config.resolved.yaml")
    raw.to_csv(output / "runs.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )
    if results and any(result.events for result in results):
        with (output / "events.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                for event in result.events:
                    handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    plot_summary(summary, output / "performance.png")


def plot_summary(summary: pd.DataFrame, path: str | Path) -> None:
    """Plot task success and tail latency with 95% confidence intervals."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for policy, group in summary.groupby("policy"):
        group = group.sort_values("arrival_rate")
        label = POLICY_LABELS.get(policy, policy)
        color = POLICY_COLORS.get(policy, "#455a64")
        axes[0].errorbar(
            group["arrival_rate"],
            group["task_success_rate_mean"] * 100,
            yerr=group["task_success_rate_ci95"] * 100,
            label=label,
            color=color,
            marker="o",
            capsize=3,
            linewidth=1.8,
        )
        axes[1].errorbar(
            group["arrival_rate"],
            group["p95_latency_s_mean"],
            yerr=group["p95_latency_s_ci95"],
            label=label,
            color=color,
            marker="o",
            capsize=3,
            linewidth=1.8,
        )
    axes[0].set_title("(a) On-time and quality-qualified tasks")
    axes[0].set_ylabel("Success rate (%)")
    axes[1].set_title("(b) 95th-percentile task completion latency")
    axes[1].set_ylabel("Latency (s)")
    for axis in axes:
        axis.set_xlabel("Task arrival rate (tasks/s)")
        axis.grid(alpha=0.22, linestyle="--")
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
