"""Shared feature extraction for training and inference."""

from __future__ import annotations

import math

import numpy as np

from agentcomsim.models import EventKind, Modality, SchedulingContext
from agentcomsim.policies.rules import TUSPolicy

FEATURE_NAMES = (
    "remaining_path_s",
    "slack_s",
    "urgency",
    "criticality",
    "semantic_density",
    "class_bonus",
    "log_size_kb",
    "network_queue",
    "compute_queue",
    "modality_video",
    "modality_image",
    "modality_text",
    "kind_sense",
    "kind_tool",
    "kind_plan",
    "kind_control",
)
FEATURE_DIM = len(FEATURE_NAMES)


def extract_features(context: SchedulingContext, tus: TUSPolicy | None = None) -> np.ndarray:
    """Return a bounded, model-independent numerical representation."""

    f = (tus or TUSPolicy()).features(context)
    values = [
        np.clip(f["remaining_path_s"], 0, 3) / 3,
        np.clip(f["slack_s"], -2, 2) / 2,
        f["urgency"] / 3,
        f["criticality"] / 2,
        f["semantic_density"] / 1.5,
        f["class_bonus"] / 0.5,
        min(math.log1p(max(context.size_bytes / 1024, 1)) / 10, 1),
        min(context.network_queue / 10, 1),
        min(context.compute_queue / 10, 1),
        float(context.task.modality == Modality.VIDEO),
        float(context.task.modality == Modality.IMAGE),
        float(context.task.modality == Modality.TEXT),
        float(context.kind == EventKind.SENSE),
        float(context.kind == EventKind.TOOL),
        float(context.kind == EventKind.PLAN),
        float(context.kind == EventKind.CONTROL),
    ]
    return np.asarray(values, dtype=np.float32)
