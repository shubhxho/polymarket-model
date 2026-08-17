"""Cross-market fusion: C++ microstructure + MLX dual-stream model."""

from cmf.config import TrainConfig
from cmf.policy import decide_from_logit, decide_from_prob

__all__ = ["TrainConfig", "decide_from_logit", "decide_from_prob", "__version__"]
__version__ = "0.1.0"
