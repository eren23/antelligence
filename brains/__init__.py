"""Brain backends — decision engines for ant agents."""

from brains.interface import BrainBackend
from brains.transformer_brain import TransformerBrain

__all__ = ["BrainBackend", "TransformerBrain"]
