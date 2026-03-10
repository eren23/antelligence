"""PyTorch Elastic Weight Consolidation using autograd for Fisher computation.

Replaces manual Fisher diagonal estimation in ewc.py with autograd.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from brains.torch_utils import torch_log_prob_of_action


class TorchEWC:
    """Elastic Weight Consolidation with autograd Fisher diagonal.

    After a task is learned, call consolidate() to snapshot weights and
    compute Fisher information via autograd. During subsequent training,
    call penalty_loss() to get the EWC regularization loss.
    """

    def __init__(self, lambda_ewc: float = 100.0) -> None:
        self._lambda = lambda_ewc
        self._star_params: dict[str, torch.Tensor] = {}
        self._fisher_diag: dict[str, torch.Tensor] = {}

    @property
    def has_consolidated(self) -> bool:
        return len(self._star_params) > 0

    def compute_fisher(
        self,
        model: nn.Module,
        experiences: list[Any],
    ) -> dict[str, torch.Tensor]:
        """Compute diagonal Fisher information using autograd.

        Args:
            model: the nn.Module (must return logits as first output).
            experiences: list of objects with .state (numpy) and .action (AntAction).

        Returns:
            Dict mapping param name -> Fisher diagonal tensor.
        """
        fisher: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param)

        if not experiences:
            return fisher

        device = next(model.parameters()).device
        # nn.Module.eval — sets evaluation mode (not Python eval)
        model.train(False)  # noqa: S307 — nn.Module.train(False), not builtins.eval

        for exp in experiences:
            model.zero_grad()

            state_t = torch.tensor(
                exp.state, dtype=torch.float32, device=device,
            ).unsqueeze(0)
            logits, _ = model(state_t)
            log_prob = torch_log_prob_of_action(logits.squeeze(0), exp.action)
            log_prob.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

        # Average
        n = len(experiences)
        for name in fisher:
            fisher[name] /= n

        model.train(True)
        return fisher

    def consolidate(
        self,
        model: nn.Module,
        experiences: list[Any],
    ) -> None:
        """Snapshot weights and compute Fisher information.

        Call after training on a task to protect learned knowledge.
        """
        self._star_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        self._fisher_diag = self.compute_fisher(model, experiences)

    def penalty_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty: 0.5 * lambda * sum F_i * (theta_i - theta*_i)^2.

        Add this to the training loss to regularize toward previous solution.
        """
        if not self.has_consolidated:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self._star_params:
                diff = param - self._star_params[name]
                loss = loss + (self._fisher_diag[name] * diff.pow(2)).sum()

        return 0.5 * self._lambda * loss
