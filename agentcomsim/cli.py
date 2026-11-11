"""Command-line entry points for simulation, training, ns-3, and the local UI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

from agentcomsim import __version__
from agentcomsim.config import ExperimentConfig, load_config
from agentcomsim.experiment import default_policies, run_experiment
from agentcomsim.ns3.bridge import Ns3Bridge, export_scenario, latency_calibration, parse_results
from agentcomsim.reporting import save_experiment
from agentcomsim.trace import generate_trace

app = typer.Typer(
    help="AgentComSim: task-centric agent communication simulation.", no_args_is_help=True
)
neural_app = typer.Typer(
    help="Train and evaluate PyTorch scheduling policies.", no_args_is_help=True
)
ns3_app = typer.Typer(
    help="Export, execute, and parse packet-level ns-3 experiments.", no_args_is_help=True
)
app.add_typer(neural_app, name="neural")
app.add_typer(ns3_app, name="ns3")


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    neural_checkpoint: Path | None = typer.Option(None, exists=True, readable=True),
) -> None:
    """Run a configured experiment matrix and save CSV, JSON, and plots."""

    cfg = load_config(config)
    policies = default_policies(cfg)
    if "neural" in cfg.policies:
        if neural_checkpoint is None:
            raise typer.BadParameter(
                "--neural-checkpoint is required when the neural policy is enabled"
            )
        from agentcomsim.policies.neural import NeuralPolicy

        policies["neural"] = lambda: NeuralPolicy(str(neural_checkpoint))

    def progress(done: int, total: int, policy: str, rate: float) -> None:
        interval = max(1, total // 20)
        if done % interval == 0 or done == total:
            typer.echo(f"Completed {done}/{total} runs ({policy}, arrival_rate={rate})")

    raw, summary, results = run_experiment(cfg, policies, progress=progress)
    save_experiment(output, cfg, raw, summary, results)
    typer.echo(summary.to_string(index=False))
    typer.echo(f"Artifacts saved to {output.resolve()}")


@app.command()
def quickstart(output: Path = typer.Option("results/quickstart", "--output", "-o")) -> None:
    """Run a short, CPU-only demonstration without a configuration file."""

    cfg = ExperimentConfig(duration_s=35, warmup_s=5, runs=3, arrival_rates=(0.8, 1.4))
    raw, summary, results = run_experiment(cfg)
    save_experiment(output, cfg, raw, summary, results)
    typer.echo(summary.to_string(index=False))
    typer.echo(f"Quickstart artifacts saved to {output.resolve()}")


@neural_app.command("train")
def neural_train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    seeds: int = typer.Option(10, min=5),
    epochs: int = typer.Option(40, min=1),
    device: str = "cpu",
) -> None:
    """Collect TUS demonstrations and train a seed-separated MLP scheduler."""

    from agentcomsim.neural.train import train_neural_scheduler

    cfg = load_config(config)
    metrics = train_neural_scheduler(
        cfg,
        output,
        seeds=[cfg.base_seed + 1000 + i for i in range(seeds)],
        epochs=epochs,
        device=device,
    )
    typer.echo(json.dumps(metrics, indent=2))


@neural_app.command("evaluate")
def neural_evaluate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    checkpoint: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Compare neural and TUS policies on the configured held-out traces."""

    from agentcomsim.policies.neural import NeuralPolicy

    cfg = load_config(config)
    cfg.policies = ("tus", "neural")
    policies = default_policies(cfg)
    policies["neural"] = lambda: NeuralPolicy(str(checkpoint))
    raw, summary, results = run_experiment(cfg, policies)
    save_experiment(output, cfg, raw, summary, results)
    typer.echo(summary.to_string(index=False))


@ns3_app.command("export")
def ns3_export(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    arrival_rate: float | None = typer.Option(None, min=0.01),
    seed: int | None = None,
) -> None:
    """Export one pre-generated task trace to the ns-3 CSV schema."""

    cfg = load_config(config)
    rate = arrival_rate or cfg.arrival_rates[0]
    actual_seed = cfg.base_seed if seed is None else seed
    trace = generate_trace(cfg, rate, actual_seed)
    path = export_scenario(trace, output)
    typer.echo(f"Exported {len(trace) * 4} messages to {path}")


@ns3_app.command("run")
def ns3_run(
    scenario: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    ns3_dir: Path | None = typer.Option(None, "--ns3-dir"),
    bandwidth: float = typer.Option(25.0, min=0.01),
    delay_ms: float = typer.Option(2.0, min=0.0),
) -> None:
    """Install the companion C++ source and execute an external ns-3 tree."""

    messages, summary = Ns3Bridge(ns3_dir).run(
        scenario, output, bandwidth_mbps=bandwidth, delay_ms=delay_ms
    )
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(json.dumps(latency_calibration(messages), indent=2))


@ns3_app.command("calibrate")
def ns3_calibrate(results: Path = typer.Option(..., exists=True)) -> None:
    """Parse an existing ns-3 result directory and estimate delay statistics."""

    messages, summary = parse_results(results)
    typer.echo(
        json.dumps({"summary": summary, "calibration": latency_calibration(messages)}, indent=2)
    )


@app.command()
def web() -> None:
    """Launch the local Streamlit interface."""

    app_path = Path(__file__).resolve().parent / "web" / "app.py"
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)]))


if __name__ == "__main__":
    app()
