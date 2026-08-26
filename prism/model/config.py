"""Prism AI Model Configuration.

Default values correspond to the 10B parameter configuration:
- 46 transformer layers
- 4096 hidden dimension
- 32 query heads, 8 KV heads (GQA ratio 4:1)
- SwiGLU FFN with 14336 intermediate dimension
- 32000 BPE vocabulary
- ~10.16B total parameters
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
import copy


@dataclass
class PrismConfig:
    """Configuration for the Prism AI transformer model.

    All model hyperparameters are defined here. Changing these values
    is the ONLY thing needed to scale the model up or down — the
    architecture code remains identical.
    """

    # ── Model Architecture ──────────────────────────────────────────
    hidden_size: int = 4096
    num_layers: int = 46
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 14336
    vocab_size: int = 32000

    # ── Sequence Length ──────────────────────────────────────────────
    max_seq_len: int = 4096

    # ── Rotary Position Embeddings ──────────────────────────────────
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None  # For YaRN/NTK scaling to 64k

    # ── Normalization ───────────────────────────────────────────────
    rms_norm_eps: float = 1e-6

    # ── Regularization ──────────────────────────────────────────────
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # ── Initialization ──────────────────────────────────────────────
    initializer_range: float = 0.02

    # ── Training Options ────────────────────────────────────────────
    use_cache: bool = True
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False

    # ── Special Tokens ──────────────────────────────────────────────
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        """Total dimension of key/value projections."""
        return self.num_kv_heads * self.head_dim

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head (GQA group size)."""
        return self.num_attention_heads // self.num_kv_heads

    @property
    def num_params(self) -> int:
        """Estimate total parameter count."""
        # Token embeddings
        embed_params = self.vocab_size * self.hidden_size

        # Per transformer layer
        kv_dim = self.num_kv_heads * self.head_dim
        attn_params = (
            self.hidden_size * self.hidden_size      # Q projection
            + self.hidden_size * kv_dim              # K projection
            + self.hidden_size * kv_dim              # V projection
            + self.hidden_size * self.hidden_size    # O projection
        )
        ffn_params = (
            self.hidden_size * self.intermediate_size  # gate_proj
            + self.hidden_size * self.intermediate_size  # up_proj
            + self.intermediate_size * self.hidden_size  # down_proj
        )
        norm_params = self.hidden_size * 2  # 2 RMSNorms per layer

        per_layer = attn_params + ffn_params + norm_params

        # Final normalization
        final_norm = self.hidden_size

        # LM head (tied with embeddings = 0 additional params)
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size

        return embed_params + self.num_layers * per_layer + final_norm + lm_head

    @property
    def num_params_billions(self) -> float:
        """Total parameters in billions."""
        return self.num_params / 1e9

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises:
            AssertionError: If any constraint is violated.
        """
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.intermediate_size > 0, "intermediate_size must be positive"

        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        assert self.num_attention_heads % self.num_kv_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )

        assert 0.0 <= self.attention_dropout <= 1.0, "attention_dropout must be in [0, 1]"
        assert 0.0 <= self.hidden_dropout <= 1.0, "hidden_dropout must be in [0, 1]"
        assert self.rms_norm_eps > 0, "rms_norm_eps must be positive"
        assert self.initializer_range > 0, "initializer_range must be positive"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrismConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "PrismConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str) -> "PrismConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def copy(self) -> "PrismConfig":
        """Return a deep copy of this config."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        params_b = self.num_params_billions
        return (
            f"PrismConfig(\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  num_kv_heads={self.num_kv_heads},\n"
            f"  intermediate_size={self.intermediate_size},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  total_params={params_b:.2f}B\n"
            f")"
        )
