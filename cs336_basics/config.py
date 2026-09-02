from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class DataConfig:
    """Data configuration."""
    path: str = "data/tinystories"
    tokenizer_path: str = "hf_tokenizer/tinystories/tokenizer.json"

@dataclass
class ModelConfig:
    """Transformer Language Model configuration."""
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344  # From model/default.yaml
    rope_theta: float = 10000.0
    # Ablation study parameters
    # ffn_type: str = 'swiglu' # 'swiglu' or 'silu'

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    max_lr: float = 3e-4
    min_lr: Optional[float] = None # 3e-5
    warmup_iters: Optional[int] = None # 500
    max_l2_norm: float = 1.0 # For gradient clipping
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

@dataclass
class TrainingConfig:
    """Training loop configuration."""
    seed: int = 1337
    is_compile: bool = False # torch.compile the model or not
    batch_size: int = 256
    max_iters: Optional[int] = None # 5000
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 200
    resume_from: Optional[str] = None
    out_dir: str = "outputs" # From training/default.yaml
    save_ckpt: bool = False

@dataclass
class TrainConfig:
    """
    The main configuration object, composed of all sub-configs.
    The `defaults` list is used by Hydra to compose the final config.
    """
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"data": "default"},
            {"model": "default"},
            {"optimizer": "default"},
            {"training": "default"},
        ]
    )

    # Sub-configs are defined with default_factory to be instantiated correctly
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)