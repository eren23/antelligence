"""PPO data structures — rollout buffer and step dataclass.

The actual PPO trainer lives in brains/torch_ppo.py (GPU-batched).
This module provides only the shared data structures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.actions import AntAction


@dataclass
class PPOStep:
    """Single step of a PPO rollout."""
    state: np.ndarray         # sensory vector (39,)
    action: AntAction         # action taken
    log_prob: float           # log π_old(a|s)
    value: float              # V(s) from old policy
    reward: float             # r_t
    done: bool                # terminal flag


class PPORolloutBuffer:
    """Collects rollout data and computes GAE advantages."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._steps: list[PPOStep] = []

    def add(self, step: PPOStep) -> None:
        self._steps.append(step)

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def ready(self) -> bool:
        return len(self._steps) >= self.capacity

    def compute_gae(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[AntAction], np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Returns:
            states, advantages, returns, actions, old_log_probs, old_values
        """
        n = len(self._steps)
        advantages = np.zeros(n, dtype=np.float64)
        rewards = np.array([s.reward for s in self._steps])
        values = np.array([s.value for s in self._steps])
        dones = np.array([s.done for s in self._steps])

        gae = 0.0
        for t in range(n - 1, -1, -1):
            if t == n - 1:
                next_val = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        states = np.array([s.state for s in self._steps])
        actions = [s.action for s in self._steps]
        old_log_probs = np.array([s.log_prob for s in self._steps])

        return states, advantages, returns, actions, old_log_probs, values

    def clear(self) -> None:
        self._steps.clear()
