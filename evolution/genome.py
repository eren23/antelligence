"""Architecture gene encoding for evolved brains.

An ArchitectureGenome describes the hyperparameters and architecture of a
brain, including hidden layer sizes, activation functions, learning rates,
and transformer-specific parameters. Genomes can be mutated and crossed over
to explore the architecture search space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArchitectureGenome:
    """Encodes brain architecture and training hyperparameters."""

    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"          # "relu", "tanh", "gelu"
    learning_rate: float = 1e-4
    entropy_coef: float = 0.01
    intrinsic_coef: float = 0.5
    brain_type: str = "nn"            # "nn" or "transformer"

    # Transformer-specific
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    context_length: int = 16

    def mutate(
        self,
        rng: np.random.Generator,
        rate: float = 0.1,
    ) -> ArchitectureGenome:
        """Return a mutated copy. Each field has `rate` probability of mutation."""
        g = ArchitectureGenome(
            hidden_sizes=list(self.hidden_sizes),
            activation=self.activation,
            learning_rate=self.learning_rate,
            entropy_coef=self.entropy_coef,
            intrinsic_coef=self.intrinsic_coef,
            brain_type=self.brain_type,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            context_length=self.context_length,
        )

        if rng.random() < rate:
            # Mutate hidden sizes: scale each layer independently
            g.hidden_sizes = [
                max(8, int(h * rng.lognormal(0, 0.3)))
                for h in g.hidden_sizes
            ]

        if rng.random() < rate:
            g.activation = rng.choice(["relu", "tanh", "gelu"])

        if rng.random() < rate:
            g.learning_rate = float(g.learning_rate * rng.lognormal(0, 0.5))
            g.learning_rate = max(1e-6, min(1e-2, g.learning_rate))

        if rng.random() < rate:
            g.entropy_coef = float(g.entropy_coef * rng.lognormal(0, 0.3))
            g.entropy_coef = max(1e-4, min(0.5, g.entropy_coef))

        if rng.random() < rate:
            g.intrinsic_coef = float(g.intrinsic_coef * rng.lognormal(0, 0.3))
            g.intrinsic_coef = max(0.0, min(2.0, g.intrinsic_coef))

        # Transformer mutations
        if g.brain_type == "transformer":
            if rng.random() < rate:
                options = [16, 32, 64]
                g.d_model = int(rng.choice(options))
                # Ensure n_heads divides d_model
                valid_heads = [h for h in [1, 2, 4, 8] if g.d_model % h == 0]
                g.n_heads = int(rng.choice(valid_heads))

            if rng.random() < rate:
                g.n_layers = max(1, min(6, g.n_layers + int(rng.choice([-1, 0, 1]))))

            if rng.random() < rate:
                g.context_length = int(rng.choice([8, 16, 32]))

        return g

    @staticmethod
    def crossover(
        parent_a: ArchitectureGenome,
        parent_b: ArchitectureGenome,
        rng: np.random.Generator,
    ) -> ArchitectureGenome:
        """Uniform crossover between two genomes."""
        def pick(a, b):
            return a if rng.random() < 0.5 else b

        return ArchitectureGenome(
            hidden_sizes=pick(parent_a.hidden_sizes, parent_b.hidden_sizes),
            activation=pick(parent_a.activation, parent_b.activation),
            learning_rate=pick(parent_a.learning_rate, parent_b.learning_rate),
            entropy_coef=pick(parent_a.entropy_coef, parent_b.entropy_coef),
            intrinsic_coef=pick(parent_a.intrinsic_coef, parent_b.intrinsic_coef),
            brain_type=pick(parent_a.brain_type, parent_b.brain_type),
            d_model=pick(parent_a.d_model, parent_b.d_model),
            n_heads=pick(parent_a.n_heads, parent_b.n_heads),
            n_layers=pick(parent_a.n_layers, parent_b.n_layers),
            context_length=pick(parent_a.context_length, parent_b.context_length),
        )
