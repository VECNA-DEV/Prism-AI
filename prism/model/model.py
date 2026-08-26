"""Prism AI — Full Causal Language Model.

This is the top-level model class that assembles all components:
  - Token embeddings (shared with LM head via weight tying)
  - Stack of N TransformerBlocks (46 for the 10B config)
  - Final RMSNorm
  - Language model head (logits projection)

Supports:
  - Gradient checkpointing (for memory-efficient training)
  - KV caching (for efficient autoregressive generation)
  - Standard cross-entropy loss computation
  - Weight initialization following the LLaMA convention
"""

from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from prism.model.config import PrismConfig
from prism.model.normalization import RMSNorm
from prism.model.transformer import TransformerBlock


class PrismForCausalLM(nn.Module):
    """Prism AI decoder-only transformer for causal language modeling.

    Args:
        config: PrismConfig with all model hyperparameters.
    """

    def __init__(self, config: PrismConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # ── Token Embeddings ────────────────────────────────────────
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # ── Transformer Layers ──────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                attention_dropout=config.attention_dropout,
                hidden_dropout=config.hidden_dropout,
                layer_idx=i,
            )
            for i in range(config.num_layers)
        ])

        # ── Final Normalization ─────────────────────────────────────
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ── Language Model Head ─────────────────────────────────────
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ── Weight Tying ────────────────────────────────────────────
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # ── Initialize Weights ──────────────────────────────────────
        self.apply(self._init_weights)

        # Apply special scaled initialization to output projections
        # (residual projections are scaled by 1/sqrt(2*num_layers))
        self._init_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with normal distribution.

        Following the LLaMA/GPT convention:
          - Linear layers: N(0, initializer_range)
          - Embeddings: N(0, initializer_range)
          - RMSNorm: ones (already set in RMSNorm.__init__)
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def _init_residual_projections(self) -> None:
        """Apply scaled initialization to residual stream projections.

        The output projection of attention (o_proj) and down projection
        of FFN (down_proj) feed into the residual stream. Scaling their
        initialization by 1/sqrt(2*num_layers) prevents the residual
        stream variance from growing with depth.

        This follows GPT-2 / Megatron-LM initialization conventions.
        """
        factor = 1.0 / (2.0 * self.config.num_layers) ** 0.5
        for layer in self.layers:
            torch.nn.init.normal_(
                layer.attention.o_proj.weight,
                mean=0.0,
                std=self.config.initializer_range * factor,
            )
            torch.nn.init.normal_(
                layer.feed_forward.down_proj.weight,
                mean=0.0,
                std=self.config.initializer_range * factor,
            )

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to reduce memory usage.

        Trades ~30% more compute for ~60% less activation memory.
        Essential for training the 10B model on limited GPU memory.
        """
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through the full model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Optional mask of shape (batch, 1, seq_len, kv_seq_len).
            position_ids: Optional position indices of shape (batch, seq_len).
            past_key_values: Optional list of (K, V) caches, one per layer.
            labels: Optional target token IDs for loss computation.
                   Shape (batch, seq_len). Use -100 for positions to ignore.
            use_cache: Whether to return KV caches for generation.

        Returns:
            Dictionary with:
              - "loss": Cross-entropy loss (only if labels provided)
              - "logits": Output logits of shape (batch, seq_len, vocab_size)
              - "past_key_values": Updated KV caches (only if use_cache=True)
        """
        # ── Embed tokens ────────────────────────────────────────────
        hidden_states = self.tok_embeddings(input_ids)

        # ── Pass through transformer layers ─────────────────────────
        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing: recompute activations during backward
                # This saves ~60% activation memory at the cost of ~30% more compute
                hidden_states, kv_cache = gradient_checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_kv,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, kv_cache = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

            if use_cache:
                new_caches.append(kv_cache)

        # ── Final normalization ─────────────────────────────────────
        hidden_states = self.norm(hidden_states)

        # ── Language model head (project to vocabulary) ─────────────
        logits = self.lm_head(hidden_states)

        # ── Compute loss if labels provided ─────────────────────────
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction:
            # logits[t] should predict labels[t+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss (ignores positions where label = -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_caches if use_cache else None,
        }

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Count total trainable parameters.

        Args:
            non_embedding: If True, exclude embedding parameters.

        Returns:
            Total number of parameters.
        """
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.tok_embeddings.weight.numel()
        return num_params

    @torch.no_grad()
    def estimate_flops_per_token(self) -> int:
        """Estimate FLOPs per token for a forward pass.

        Uses the approximation: FLOPs ≈ 2 * num_params (for matmuls).
        Backward pass is ~2× forward, so total training FLOPs per token ≈ 6 * num_params.

        Returns:
            Estimated FLOPs per token (forward pass only).
        """
        return 2 * self.get_num_params(non_embedding=True)

    def __repr__(self) -> str:
        num_params = self.get_num_params()
        return (
            f"PrismForCausalLM(\n"
            f"  params={num_params / 1e9:.2f}B,\n"
            f"  layers={self.config.num_layers},\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  heads={self.config.num_attention_heads}q/{self.config.num_kv_heads}kv,\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  max_seq_len={self.config.max_seq_len},\n"
            f"  gradient_checkpointing={self.gradient_checkpointing}\n"
            f")"
        )
