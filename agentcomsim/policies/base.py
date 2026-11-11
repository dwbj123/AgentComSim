"""Scheduling policy API. Lower numerical priority values are served first."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentcomsim.models import Modality, SchedulingContext


class SchedulingPolicy(ABC):
    """Interface implemented by rule-based and learned schedulers."""

    name = "abstract"

    @abstractmethod
    def priority(self, context: SchedulingContext) -> float:
        """Return a queue priority; smaller values have higher precedence."""

    def should_compress(self, context: SchedulingContext) -> bool:
        return False

    def compression_parameters(self, modality: Modality) -> tuple[float, float]:
        """Return byte-retention and semantic-quality factors."""

        return 1.0, 1.0
