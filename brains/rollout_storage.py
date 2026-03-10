"""Pre-allocated GPU tensor buffer for PPO rollout collection.

Stores trajectory data as tensors (not Python objects) for GPU-native
GAE computation and batched PPO updates.
"""

from __future__ import annotations

import torch


class RolloutStorage:
    """Pre-allocated GPU tensor buffer for rollout collection.

    All tensors are pre-allocated at init time. The `insert()` method
    fills them one step at a time. When full, `compute_gae()` runs
    GAE entirely on GPU.
    """

    def __init__(self, capacity: int, input_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self._ptr = 0

        self.states = torch.zeros(capacity, input_dim, device=device)
        self.actions = torch.zeros(capacity, 7, device=device)
        self.log_probs = torch.zeros(capacity, device=device)
        self.values = torch.zeros(capacity, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)

    @property
    def ready(self) -> bool:
        return self._ptr >= self.capacity

    def __len__(self) -> int:
        return self._ptr

    def insert(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        """Insert a single step into the buffer."""
        if self._ptr >= self.capacity:
            return
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self._ptr += 1

    def insert_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Insert a batch of steps (e.g. from all ants in one tick)."""
        n = states.shape[0]
        space = self.capacity - self._ptr
        n = min(n, space)
        if n <= 0:
            return
        end = self._ptr + n
        self.states[self._ptr:end] = states[:n]
        self.actions[self._ptr:end] = actions[:n]
        self.log_probs[self._ptr:end] = log_probs[:n]
        self.values[self._ptr:end] = values[:n]
        self.rewards[self._ptr:end] = rewards[:n]
        self.dones[self._ptr:end] = dones[:n]
        self._ptr = end

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns on GPU.

        Args:
            last_value: V(s_{T+1}) bootstrap value
            gamma: discount factor
            lam: GAE lambda

        Returns:
            advantages: (N,)
            returns: (N,)
        """
        n = self._ptr
        advantages = torch.zeros(n, device=self.device)
        gae = 0.0

        for t in range(n - 1, -1, -1):
            if t == n - 1:
                next_val = last_value
            else:
                next_val = self.values[t + 1].item()
            non_terminal = 1.0 - self.dones[t].item()
            delta = self.rewards[t].item() + gamma * next_val * non_terminal - self.values[t].item()
            gae = delta + gamma * lam * non_terminal * gae
            advantages[t] = gae

        returns = advantages + self.values[:n]
        return advantages, returns

    def clear(self) -> None:
        self._ptr = 0
