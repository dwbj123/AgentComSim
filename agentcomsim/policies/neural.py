"""Inference adapter for a trained PyTorch scheduler."""

from __future__ import annotations

import numpy as np

from agentcomsim.config import TUSConfig
from agentcomsim.models import SchedulingContext
from agentcomsim.neural.features import extract_features
from agentcomsim.neural.model import load_model
from agentcomsim.policies.base import SchedulingPolicy
from agentcomsim.policies.rules import TUSPolicy


class NeuralPolicy(SchedulingPolicy):
    """Schedule with a local checkpoint; no lossy compression by default."""

    name = "neural"

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.model, self.checkpoint = load_model(checkpoint_path, device)
        self.device = device
        self.tus = TUSPolicy(TUSConfig(**self.checkpoint.get("tus_config", {})))

    def priority(self, context: SchedulingContext) -> float:
        features = extract_features(context, self.tus)
        tensor = np.expand_dims(features, axis=0)
        import torch

        with torch.no_grad():
            score = self.model(torch.from_numpy(tensor).to(self.device)).item()
        return -float(score * 4.0)
