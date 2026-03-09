"""BrainBackend Protocol — the interface all brain implementations must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agents.actions import AntAction
from agents.sensory import SensoryInput


@runtime_checkable
class BrainBackend(Protocol):
    """Decision engine for an ant agent.

    All brain backends (rule-based, NN, transformer) implement this protocol.
    """

    def decide(self, sensory: SensoryInput) -> AntAction:
        """Given the ant's current sensory input, return an action."""
        ...

    def learn(self, reward: float) -> None:
        """Receive a reward signal. No-op for non-learning brains."""
        ...
