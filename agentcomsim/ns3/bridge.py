"""One-way packet-level ns-3 bridge with validated result parsing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from agentcomsim.models import TaskSpec


class Ns3Error(RuntimeError):
    """Base error for ns-3 integration failures."""


class Ns3NotFoundError(Ns3Error):
    """Raised when the ns-3 launcher or source tree is unavailable."""


class Ns3ExecutionError(Ns3Error):
    """Raised when ns-3 returns a non-zero status or invalid artifacts."""


@dataclass(slots=True, frozen=True)
class PacketMessage:
    message_id: str
    task_id: int
    kind: str
    send_time_s: float
    size_bytes: int


def messages_from_trace(trace: list[TaskSpec]) -> list[PacketMessage]:
    """Flatten task payloads into a deterministic packet-level input trace."""

    messages = []
    for task in trace:
        for kind, attribute in (
            ("sense", "sense_bytes"),
            ("tool", "tool_bytes"),
            ("plan", "plan_bytes"),
            ("control", "control_bytes"),
        ):
            messages.append(
                PacketMessage(
                    f"task-{task.task_id}-{kind}",
                    task.task_id,
                    kind,
                    task.arrival_s,
                    max(1, round(getattr(task, attribute))),
                )
            )
    return sorted(messages, key=lambda message: (message.send_time_s, message.message_id))


def export_scenario(trace: list[TaskSpec], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(message) for message in messages_from_trace(trace)]).to_csv(
        output, index=False
    )
    return output


def parse_results(output_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Validate and load the two artifacts written by the C++ bridge."""

    output = Path(output_dir)
    messages_path = output / "messages.csv"
    summary_path = output / "summary.json"
    if not messages_path.is_file() or not summary_path.is_file():
        raise Ns3ExecutionError("ns-3 output is missing messages.csv or summary.json")
    messages = pd.read_csv(messages_path)
    required = {
        "message_id",
        "task_id",
        "kind",
        "send_time_s",
        "size_bytes",
        "finish_time_s",
        "received_bytes",
    }
    missing = required - set(messages.columns)
    if missing:
        raise Ns3ExecutionError(f"ns-3 messages.csv is missing columns: {sorted(missing)}")
    if (messages["received_bytes"] > messages["size_bytes"]).any():
        raise Ns3ExecutionError("received_bytes cannot exceed size_bytes")
    complete = messages["received_bytes"] == messages["size_bytes"]
    if (messages.loc[complete, "finish_time_s"] < messages.loc[complete, "send_time_s"]).any():
        raise Ns3ExecutionError("finish_time_s cannot precede send_time_s")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("message_count") != len(messages):
        raise Ns3ExecutionError("summary message_count does not match messages.csv")
    return messages, summary


def latency_calibration(messages: pd.DataFrame) -> dict[str, float]:
    """Estimate a message-level base delay from completed ns-3 messages."""

    complete = messages[messages["received_bytes"] == messages["size_bytes"]].copy()
    if complete.empty:
        raise Ns3ExecutionError("No complete ns-3 messages are available for calibration")
    latency = complete["finish_time_s"] - complete["send_time_s"]
    return {
        "samples": float(len(complete)),
        "mean_latency_s": float(latency.mean()),
        "p95_latency_s": float(latency.quantile(0.95)),
        "suggested_base_delay_s": float(latency.min()),
    }


class Ns3Bridge:
    """Install and execute the companion C++ program in an external ns-3 tree."""

    def __init__(self, ns3_dir: str | Path | None = None) -> None:
        self.ns3_dir = Path(ns3_dir or os.environ.get("NS3_HOME", "")).expanduser().resolve()
        self.launcher = self.ns3_dir / "ns3"
        if not self.launcher.is_file():
            raise Ns3NotFoundError(
                f"ns-3 launcher not found at {self.launcher}. Pass --ns3-dir or set NS3_HOME."
            )

    def install(self) -> Path:
        source = Path(__file__).resolve().parent / "agentcomsim_bridge.cc"
        if not source.is_file():
            raise Ns3NotFoundError(f"Companion C++ source not found: {source}")
        destination = self.ns3_dir / "scratch" / "agentcomsim_bridge.cc"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def run(
        self,
        scenario_csv: str | Path,
        output_dir: str | Path,
        *,
        bandwidth_mbps: float = 25.0,
        delay_ms: float = 2.0,
        timeout_s: int = 600,
    ) -> tuple[pd.DataFrame, dict]:
        scenario = Path(scenario_csv).resolve()
        output = Path(output_dir).resolve()
        if not scenario.is_file():
            raise Ns3ExecutionError(f"Scenario does not exist: {scenario}")
        output.mkdir(parents=True, exist_ok=True)
        self.install()
        command = [
            str(self.launcher),
            "run",
            f"agentcomsim_bridge --input={scenario} --output={output} "
            f"--bandwidth={bandwidth_mbps} --delayMs={delay_ms}",
        ]
        completed = subprocess.run(
            command,
            cwd=self.ns3_dir,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        (output / "ns3.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / "ns3.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise Ns3ExecutionError(
                f"ns-3 exited with status {completed.returncode}; see {output / 'ns3.stderr.log'}"
            )
        return parse_results(output)
