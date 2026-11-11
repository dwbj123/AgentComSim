"""Deterministic offline training with validation-based early stopping."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agentcomsim.config import ExperimentConfig
from agentcomsim.neural.dataset import make_dataset
from agentcomsim.neural.model import NeuralSchedulerModel, checkpoint_metadata


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _evaluate(model: NeuralSchedulerModel, loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device))
            losses.append(nn.functional.mse_loss(prediction, y.to(device)).item())
    return float(np.mean(losses))


def train_neural_scheduler(
    config: ExperimentConfig,
    output_path: str | Path,
    *,
    seeds: list[int] | None = None,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    patience: int = 6,
    device: str = "cpu",
) -> dict[str, float | int]:
    """Train an MLP to imitate TUS priorities from observed simulator states."""

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(1729)
    np.random.seed(1729)
    torch.manual_seed(1729)
    seeds = seeds or [config.base_seed + 1000 + index for index in range(10)]
    dataset = make_dataset(config, seeds)
    train_loader = _loader(dataset.x_train, dataset.y_train, batch_size, True, 1729)
    validation_loader = _loader(dataset.x_validation, dataset.y_validation, batch_size, False, 1729)
    test_loader = _loader(dataset.x_test, dataset.y_test, batch_size, False, 1729)
    model = NeuralSchedulerModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())
        validation_loss = _evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse": float(np.mean(train_losses)),
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    test_loss = _evaluate(model, test_loader, device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_metadata(model) | {
        "model_state": best_state,
        "tus_config": asdict(config.tus),
        "training": {
            "seed": 1729,
            "seeds": seeds,
            "epochs_completed": len(history),
            "train_samples": len(dataset.x_train),
            "validation_samples": len(dataset.x_validation),
            "test_samples": len(dataset.x_test),
            "history": history,
        },
    }
    torch.save(checkpoint, output)
    metrics = {
        "best_validation_mse": best_validation,
        "test_mse": test_loss,
        "epochs_completed": len(history),
        "train_samples": len(dataset.x_train),
        "test_samples": len(dataset.x_test),
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
