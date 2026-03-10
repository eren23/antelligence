"""PyTorch PPO trainer — works with both TorchMLPModel and TorchTransformerModel.

Reuses PPORolloutBuffer and PPOStep from brains/ppo.py for trajectory storage.
Converts numpy arrays to tensors at training time.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from agents.actions import AntAction
from brains.ppo import PPORolloutBuffer, PPOStep
from brains.torch_utils import get_device, torch_entropy, torch_log_prob_of_action
from config import PPOConfig


class TorchPPOTrainer:
    """PPO with autograd. Works with any nn.Module that returns (logits, value)."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.02,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        rollout_length: int = 1024,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length
        self.buffer = PPORolloutBuffer(rollout_length)
        self.global_step: int = 0
        self.baseline: float = 0.0  # compatibility with trainer state save/load
        self._device = next(model.parameters()).device

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        cfg: PPOConfig,
        lr: float | None = None,
        rollout_length: int | None = None,
    ) -> TorchPPOTrainer:
        return cls(
            model=model,
            lr=lr or cfg.learning_rate,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_eps=cfg.clip_epsilon,
            value_loss_coef=cfg.value_loss_coef,
            entropy_coef=cfg.entropy_coef,
            epochs_per_update=cfg.epochs_per_update,
            batch_size=cfg.batch_size,
            max_grad_norm=cfg.max_grad_norm,
            rollout_length=rollout_length or cfg.rollout_length,
        )

    def update(self) -> dict[str, float]:
        """Run PPO update if buffer is ready. Returns training metrics."""
        if not self.buffer.ready:
            return {}

        self.global_step += 1

        # Bootstrap value from last state
        last_step = self.buffer._steps[-1]
        with torch.no_grad():
            last_state_t = torch.tensor(
                last_step.state, dtype=torch.float32, device=self._device,
            ).unsqueeze(0)
            _, last_val = self.model(last_state_t)
            last_value = 0.0 if last_step.done else float(last_val.item())

        # Compute GAE (numpy)
        states, advantages, returns, actions, old_log_probs, old_values = \
            self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self._device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self._device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self._device)
        old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self._device)

        # Normalize advantages
        adv_std = advantages_t.std()
        if adv_std > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / adv_std

        n = len(states)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.epochs_per_update):
            perm = torch.randperm(n, device=self._device)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]

                batch_states = states_t[idx]
                batch_adv = advantages_t[idx]
                batch_returns = returns_t[idx]
                batch_old_lp = old_lp_t[idx]
                batch_actions = [actions[i] for i in idx.cpu().tolist()]

                # Forward
                logits, values = self.model(batch_states)

                # Compute new log probs
                new_log_probs = torch.stack([
                    torch_log_prob_of_action(logits[i], batch_actions[i])
                    for i in range(len(batch_actions))
                ])

                # Ratio
                ratio = (new_log_probs - batch_old_lp).exp()

                # Clipped surrogate objective
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()

                # Entropy bonus
                entropy = torch_entropy(logits).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                num_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "num_updates": num_updates,
        }
