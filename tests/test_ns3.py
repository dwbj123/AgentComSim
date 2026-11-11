from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from agentcomsim.ns3.bridge import (
    Ns3Bridge,
    Ns3ExecutionError,
    Ns3NotFoundError,
    export_scenario,
    latency_calibration,
    messages_from_trace,
    parse_results,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ns3"


def copy_fixtures(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("messages.csv", "summary.json"):
        shutil.copy2(FIXTURES / name, destination / name)


def test_export_has_four_messages_per_task(task_factory, tmp_path):
    trace = [task_factory(task_id=2), task_factory(task_id=1, arrival_s=1.0)]
    messages = messages_from_trace(trace)
    assert len(messages) == 8
    assert messages[0].task_id == 1
    output = export_scenario(trace, tmp_path / "scenario.csv")
    frame = pd.read_csv(output)
    assert list(frame.columns) == ["message_id", "task_id", "kind", "send_time_s", "size_bytes"]
    assert frame["message_id"].is_unique


def test_parse_and_calibrate_results():
    messages, summary = parse_results(FIXTURES)
    assert summary["completed_count"] == 2
    assert len(messages) == 3
    calibration = latency_calibration(messages)
    assert calibration["samples"] == 2
    assert calibration["mean_latency_s"] == pytest.approx(0.02)
    assert calibration["suggested_base_delay_s"] == pytest.approx(0.01)


def test_incomplete_message_may_have_negative_finish_time():
    messages, _ = parse_results(FIXTURES)
    incomplete = messages[messages["received_bytes"] < messages["size_bytes"]]
    assert len(incomplete) == 1
    assert incomplete.iloc[0]["finish_time_s"] == -1


@pytest.mark.parametrize("mutation", ["overflow", "time", "count", "column"])
def test_invalid_results_are_rejected(tmp_path, mutation):
    copy_fixtures(tmp_path)
    path = tmp_path / "messages.csv"
    frame = pd.read_csv(path)
    if mutation == "overflow":
        frame.loc[0, "received_bytes"] = frame.loc[0, "size_bytes"] + 1
    elif mutation == "time":
        frame.loc[0, "finish_time_s"] = 0.0
    elif mutation == "column":
        frame = frame.drop(columns=["kind"])
    if mutation != "count":
        frame.to_csv(path, index=False)
    else:
        summary_path = tmp_path / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["message_count"] = 99
        summary_path.write_text(json.dumps(summary))
    with pytest.raises(Ns3ExecutionError):
        parse_results(tmp_path)


def test_missing_ns3_launcher_is_explicit(tmp_path):
    with pytest.raises(Ns3NotFoundError, match="launcher not found"):
        Ns3Bridge(tmp_path)


def test_bridge_invocation_and_logs(tmp_path, monkeypatch):
    ns3_dir = tmp_path / "ns3"
    ns3_dir.mkdir()
    (ns3_dir / "ns3").write_text("fake launcher")
    scenario = tmp_path / "scenario.csv"
    scenario.write_text("message_id,task_id,kind,send_time_s,size_bytes\n")
    output = tmp_path / "output"

    def fake_run(command, **kwargs):
        arguments = shlex.split(command[2])
        destination = Path(next(x.split("=", 1)[1] for x in arguments if x.startswith("--output=")))
        copy_fixtures(destination)
        return subprocess.CompletedProcess(command, 0, "simulated stdout", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    messages, summary = Ns3Bridge(ns3_dir).run(scenario, output)
    assert len(messages) == 3
    assert summary["message_count"] == 3
    assert (ns3_dir / "scratch" / "agentcomsim_bridge.cc").is_file()
    assert (output / "ns3.stdout.log").read_text() == "simulated stdout"


def test_wire_format_regression_guard():
    source = (
        Path(__file__).parents[1] / "agentcomsim" / "ns3" / "agentcomsim_bridge.cc"
    ).read_text()
    assert "packet->CopyData(prefix, 4)" in source
    assert "PeekTag" not in source
