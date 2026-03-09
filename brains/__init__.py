"""Brain backends — decision engines for ant agents."""

from brains.interface import BrainBackend
from brains.transformer_brain import TransformerBrain

# MLX backends (optional — require mlx package)
try:
    from brains.mlx_nn_brain import MLXNNBrain
    from brains.mlx_transformer_brain import MLXTransformerBrain
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

__all__ = ["BrainBackend", "TransformerBrain"]
if MLX_AVAILABLE:
    __all__ += ["MLXNNBrain", "MLXTransformerBrain"]
