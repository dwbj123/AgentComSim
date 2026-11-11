from __future__ import annotations

import numpy as np

from agentcomsim.neural.features import FEATURE_DIM, extract_features
from agentcomsim.neural.model import load_model
from agentcomsim.neural.train import train_neural_scheduler
from agentcomsim.policies.neural import NeuralPolicy


def test_feature_vector_is_finite(task_factory, context_factory):
    features = extract_features(context_factory(task_factory()))
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_training_checkpoint_and_inference_round_trip(tmp_path, task_factory, context_factory):
    from agentcomsim.config import ExperimentConfig

    config = ExperimentConfig(
        duration_s=8,
        warmup_s=1,
        runs=1,
        arrival_rates=(1.2,),
        policies=("tus",),
    )
    config.tus.control_bonus = 0.7
    checkpoint = tmp_path / "scheduler.pt"
    metrics = train_neural_scheduler(
        config,
        checkpoint,
        seeds=[100, 101, 102, 103, 104],
        epochs=2,
        batch_size=64,
        patience=2,
    )
    assert checkpoint.is_file()
    assert metrics["train_samples"] > 0
    assert metrics["test_samples"] > 0
    assert np.isfinite(metrics["test_mse"])
    model, metadata = load_model(str(checkpoint))
    assert not model.training
    assert metadata["tus_config"]["control_bonus"] == 0.7
    policy = NeuralPolicy(str(checkpoint))
    assert policy.tus.config.control_bonus == 0.7
    score = policy.priority(context_factory(task_factory()))
    assert np.isfinite(score)
