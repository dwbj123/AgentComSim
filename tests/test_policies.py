from __future__ import annotations

from agentcomsim.models import EventKind, Modality
from agentcomsim.policies.rules import FIFOPolicy, StaticPriorityPolicy, TUSPolicy


def test_static_priority_order(context_factory, task_factory):
    task = task_factory()
    policy = StaticPriorityPolicy()
    scores = [policy.priority(context_factory(task, kind)) for kind in EventKind]
    assert scores == [3.0, 2.0, 1.0, 0.0]
    assert FIFOPolicy().priority(context_factory(task)) == 0


def test_tus_urgency_increases_priority(context_factory, task_factory):
    task = task_factory()
    policy = TUSPolicy()
    early = policy.priority(context_factory(task, now_s=3.0))
    late = policy.priority(context_factory(task, now_s=7.7))
    assert late < early


def test_tus_compression_is_limited_to_visual_sensing(context_factory, task_factory):
    policy = TUSPolicy()
    video = task_factory(modality=Modality.VIDEO)
    text = task_factory(modality=Modality.TEXT)
    assert policy.should_compress(context_factory(video, network_queue=2))
    assert not policy.should_compress(context_factory(text, network_queue=10))
    assert not policy.should_compress(
        context_factory(video, kind=EventKind.CONTROL, network_queue=10)
    )
    assert policy.compression_parameters(Modality.VIDEO) == (0.38, 0.93)
    assert policy.compression_parameters(Modality.TEXT) == (1.0, 1.0)
