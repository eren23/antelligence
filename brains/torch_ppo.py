"""PyTorch PPO trainer — GPU-batched via ActionDistribution.

Works with any nn.Module that returns (logits, value).
Uses RolloutStorage for pre-allocated GPU tensor buffers and
ActionDistribution for fully batched log_prob/entropy computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from brains.action_dist import ActionDistribution
from brains.rollout_storage import RolloutStorage
from config import PPOConfig


class TorchPPOTrainer:
    """PPO with autograd and batched GPU action distributions."""

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
        input_dim: int = 39,
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
        self._device = next(model.parameters()).device
        self.buffer = RolloutStorage(rollout_length, input_dim, self._device)
        self.global_step: int = 0
        self.baseline: float = 0.0  # compatibility with trainer state save/load

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        cfg: PPOConfig,
        lr: float | None = None,
        rollout_length: int | None = None,
        input_dim: int = 39,
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
            input_dim=input_dim,
        )

    def update(self) -> dict[str, float]:
        """Run PPO update if buffer is ready. Returns training metrics."""
        if not self.buffer.ready:
            return {}

        self.global_step += 1
        n = len(self.buffer)

        # Bootstrap value from last state
        with torch.no_grad():
            last_state = self.buffer.states[n - 1].unsqueeze(0)
            _, last_val = self.model(last_state)
            last_done = self.buffer.dones[n - 1].item()
            last_value = 0.0 if last_done > 0.5 else float(last_val.item())

        # Compute GAE
        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda,
        )

        # Slice active buffer data
        states_t = self.buffer.states[:n]
        actions_t = self.buffer.actions[:n]
        old_lp_t = self.buffer.log_probs[:n]

        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

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
                batch_adv = advantages[idx]
                batch_returns = returns[idx]
                batch_old_lp = old_lp_t[idx]
                batch_actions = actions_t[idx]

                # Forward
                logits, values = self.model(batch_states)

                # Batched log_prob + entropy via ActionDistribution
                dist = ActionDistribution(logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Ratio
                ratio = (new_log_probs - batch_old_lp).exp()

                # Clipped surrogate objective
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()

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
