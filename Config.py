from dataclasses import dataclass
from typing import Optional
@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_local_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 15000

    # Qwen3-like parameters
    sliding_window: int = 4096  # Set a large default, effectively disabling it unless specified
    attention_bias: bool = False  # Qwen3 often sets this to False
    rms_norm_eps: float = 1e-6  # Epsilon for RMSNorm

    # Training parameters
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-3
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    muon_lr: float = 0.07

    q_lora_rank: int = 0
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 32
    qk_head_dim: int = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim: int = 128
    rope_factor: float = 40.0
    mscale: float = 1.0
    original_seq_len: int = 4096

    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"

    beta_fast: int = 32
    beta_slow: int = 1
    rope_theta: float = 10000.0
    rope_factor: float = 40

    # Data parameters
    max_seq_len: int = 1024
    num_documents: int = 300000
    max_tokens: int = 900000000

    eval_interval: int = 100
    eval_batches: int = 50

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100
    log_interval: int = 50

    grad_clip: float = 1.0

    # Regularization
    weight_decay: float = 0.1
    warmup_steps: int = 100
    dropout: float = 0.1
    gradient_clip: float = 1.0

    # MoE specific parameters
    num_experts: int = 8
    top_k: int = 2
    load_balancing_weight: float = 0.01

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
