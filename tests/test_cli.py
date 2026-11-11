from __future__ import annotations

from typer.testing import CliRunner

from agentcomsim import __version__
from agentcomsim.cli import app

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_run_command(tmp_path, small_config):
    small_config.runs = 1
    small_config.policies = ("fifo", "tus")
    small_config.record_events = False
    config = tmp_path / "config.yaml"
    small_config.save_yaml(config)
    output = tmp_path / "results"
    result = runner.invoke(app, ["run", "-c", str(config), "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert (output / "runs.csv").is_file()
    assert (output / "summary.json").is_file()


def test_ns3_export_command(tmp_path, small_config):
    config = tmp_path / "config.yaml"
    small_config.save_yaml(config)
    output = tmp_path / "scenario.csv"
    result = runner.invoke(app, ["ns3", "export", "-c", str(config), "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert output.is_file()
