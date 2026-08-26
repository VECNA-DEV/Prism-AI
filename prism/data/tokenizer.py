"""SentencePiece BPE Tokenizer for Prism AI.

Trains and wraps a SentencePiece BPE tokenizer optimized for source code.
Key design decisions:
  - BPE (not Unigram): Better for code where token boundaries matter
  - 32,000 vocabulary: Standard size, good compression ratio on code
  - byte_fallback=True: Handles ANY byte sequence (no <unk> during inference)
  - Trained on our actual code corpus for optimal compression

The tokenizer training script samples from The Stack v2 via HuggingFace
streaming to build the vocabulary. The resulting model is a .model file
that can be loaded for both training and inference.
"""

import os
from typing import List, Optional

import sentencepiece as spm


# ── Special Token IDs ────────────────────────────────────────────────
# SentencePiece reserves these by default
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class PrismTokenizer:
    """Wrapper around SentencePiece for Prism AI tokenization.

    Provides a clean interface for encoding/decoding text with
    support for special tokens, batch processing, and the specific
    conventions needed for causal language modeling.

    Args:
        model_path: Path to a trained .model file.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found at {model_path}. "
                f"Train one first with scripts/train_tokenizer.py"
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path

    # ── Core API ────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.
            add_bos: Prepend beginning-of-sequence token.
            add_eos: Append end-of-sequence token.

        Returns:
            List of integer token IDs.
        """
        tokens = self.sp.Encode(text)

        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        # Filter out special tokens for clean output
        filtered = [t for t in token_ids if t not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.Decode(filtered)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """Encode a batch of texts.

        Args:
            texts: List of input text strings.
            add_bos: Prepend BOS to each sequence.
            add_eos: Append EOS to each sequence.

        Returns:
            List of token ID lists.
        """
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode_batch(self, token_id_lists: List[List[int]]) -> List[str]:
        """Decode a batch of token ID sequences.

        Args:
            token_id_lists: List of token ID lists.

        Returns:
            List of decoded strings.
        """
        return [self.decode(ids) for ids in token_id_lists]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into string tokens (for inspection).

        Args:
            text: Input text.

        Returns:
            List of token strings.
        """
        return self.sp.EncodeAsPieces(text)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return self.sp.GetPieceSize()

    @property
    def bos_id(self) -> int:
        """Beginning-of-sequence token ID."""
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self.sp.eos_id()

    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return self.sp.unk_id()

    # ── Utility ─────────────────────────────────────────────────────

    def id_to_piece(self, token_id: int) -> str:
        """Convert a token ID to its string representation."""
        return self.sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """Convert a token string to its ID."""
        return self.sp.PieceToId(piece)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"PrismTokenizer(vocab_size={self.vocab_size}, model='{self.model_path}')"


# ── Tokenizer Training ──────────────────────────────────────────────


def train_tokenizer(
    input_file: str,
    model_prefix: str = "prism_tokenizer",
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9999,
    num_threads: int = 16,
    max_sentence_length: int = 16384,
    shuffle_input_sentence: bool = True,
) -> str:
    """Train a SentencePiece BPE tokenizer from a text file.

    The input file should contain one document per line (or raw text).
    For code, we typically concatenate source files separated by newlines.

    Args:
        input_file: Path to training text file.
        model_prefix: Output prefix (produces {prefix}.model and {prefix}.vocab).
        vocab_size: Target vocabulary size.
        model_type: "bpe" or "unigram".
        character_coverage: Fraction of characters to cover (0.9999 for code).
        num_threads: Number of training threads.
        max_sentence_length: Maximum input sentence length in bytes.
        shuffle_input_sentence: Shuffle input lines before training.

    Returns:
        Path to the trained .model file.
    """
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        max_sentence_length=max_sentence_length,
        shuffle_input_sentence=shuffle_input_sentence,
        # Special tokens (SentencePiece adds these by default at IDs 0-3)
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        # Byte fallback: encode unknown bytes as <0xXX> tokens
        # This ensures the tokenizer can handle ANY input without <unk>
        byte_fallback=True,
        # Code-specific settings
        split_digits=True,           # Separate individual digits
        split_by_whitespace=True,    # Respect whitespace boundaries
        split_by_unicode_script=True,
        # Normalization: minimal for code (preserve whitespace, case)
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,      # Don't add leading space
        # Training control
        train_extremely_large_corpus=False,
        hard_vocab_limit=False,
        input_sentence_size=5000000,  # Sample 5M sentences for training
    )

    model_path = f"{model_prefix}.model"
    print(f"Tokenizer trained successfully: {model_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model type: {model_type}")

    return model_path
