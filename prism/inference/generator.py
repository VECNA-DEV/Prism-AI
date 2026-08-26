"""Text generation for Prism AI.

Implements autoregressive text generation with multiple sampling
strategies:
  - Greedy decoding (argmax)
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Temperature scaling
  - Repetition penalty

Uses KV caching for efficient token-by-token generation.
"""

from typing import Optional, List

import torch
import torch.nn.functional as F

from prism.model.config import PrismConfig
from prism.model.model import PrismForCausalLM
from prism.data.tokenizer import PrismTokenizer


class PrismGenerator:
    """Autoregressive text generator for Prism AI.

    Wraps the model with sampling strategies and handles the
    generation loop with KV caching.

    Args:
        model: Trained PrismForCausalLM instance.
        tokenizer: PrismTokenizer instance.
        device: Device for generation.
    """

    def __init__(
        self,
        model: PrismForCausalLM,
        tokenizer: PrismTokenizer,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_k: Keep only top-k logits for sampling (0 = disabled).
            top_p: Keep tokens with cumulative probability <= top_p (nucleus sampling).
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).
            stop_tokens: List of token IDs that stop generation.

        Returns:
            Generated text (prompt + completion).
        """
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_id]

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Track generated token IDs for repetition penalty
        generated_ids = list(input_ids)

        # Initialize KV cache via the first forward pass
        past_key_values = None

        # Prefill: process the entire prompt at once
        outputs = self.model(
            input_ids=input_tensor,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = outputs["past_key_values"]
        next_logits = outputs["logits"][:, -1, :]  # Logits for the next token

        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated_ids, repetition_penalty
                )

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                next_logits = self._top_k_filter(next_logits, top_k)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                next_logits = self._top_p_filter(next_logits, top_p)

            # Sample from the filtered distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            next_token_id = next_token.item()

            # Check stop condition
            if next_token_id in stop_tokens:
                break

            generated_ids.append(next_token_id)

            # Forward pass with just the new token (using KV cache)
            outputs = self.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs["past_key_values"]
            next_logits = outputs["logits"][:, -1, :]

        # Decode all generated tokens
        return self.tokenizer.decode(generated_ids)

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate text using greedy decoding (argmax at each step).

        Deterministic — always produces the same output for the same prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_ids = list(input_ids)
        past_key_values = None

        # Prefill
        outputs = self.model(
            input_ids=input_tensor,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = outputs["past_key_values"]
        next_logits = outputs["logits"][:, -1, :]

        for _ in range(max_new_tokens):
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            next_token_id = next_token.item()

            if next_token_id == self.tokenizer.eos_id:
                break

            generated_ids.append(next_token_id)

            outputs = self.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs["past_key_values"]
            next_logits = outputs["logits"][:, -1, :]

        return self.tokenizer.decode(generated_ids)

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: List[int],
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repeated tokens.

        Tokens that have already been generated get their logits
        divided (if positive) or multiplied (if negative) by the
        penalty factor.

        Args:
            logits: Raw logits of shape (1, vocab_size).
            generated_ids: List of already-generated token IDs.
            penalty: Penalty factor (> 1.0 to penalize repetition).

        Returns:
            Modified logits.
        """
        if not generated_ids:
            return logits

        # Get unique generated tokens
        unique_ids = list(set(generated_ids))
        penalty_tensor = torch.ones_like(logits)
        penalty_tensor[0, unique_ids] = penalty

        # Penalize: divide positive logits, multiply negative logits
        logits = torch.where(
            logits > 0,
            logits / penalty_tensor,
            logits * penalty_tensor,
        )

        return logits

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only the top-k highest logits, set others to -inf.

        Args:
            logits: Raw logits of shape (batch, vocab_size).
            k: Number of top logits to keep.

        Returns:
            Filtered logits.
        """
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus sampling: keep smallest set of tokens with cumulative prob >= p.

        Args:
            logits: Raw logits of shape (batch, vocab_size).
            p: Cumulative probability threshold.

        Returns:
            Filtered logits.
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # Shift right to keep at least one token
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back to original ordering
        logits = logits.scatter(1, sorted_indices, sorted_logits)
        return logits
