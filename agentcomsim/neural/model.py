"""A compact multilayer perceptron for event-priority regression."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from agentcomsim.neural.features import FEATURE_DIM, FEATURE_NAMES


class NeuralSchedulerModel(nn.Module):
    """Predict a TUS-compatible score from cross-layer scheduling features."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def checkpoint_metadata(model: NeuralSchedulerModel) -> dict[str, Any]:
    return {
        "format_version": 1,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "feature_names": list(FEATURE_NAMES),
        "target": "tus_score_divided_by_4",
    }


def load_model(path: str, device: str = "cpu") -> tuple[NeuralSchedulerModel, dict[str, Any]]:
    """Load a restricted checkpoint without executing pickled Python objects."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported or missing checkpoint format_version")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Checkpoint feature schema does not match this AgentComSim version")
    model = NeuralSchedulerModel(checkpoint["input_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
