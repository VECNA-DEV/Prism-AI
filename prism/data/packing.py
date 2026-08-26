"""Sequence Packing for efficient training.

In standard training, documents shorter than max_seq_len are padded,
wasting compute on padding tokens. Sequence packing solves this by
concatenating multiple documents into a single max_seq_len sequence
with EOS separators, achieving ~100% token utilization.

Example:
    Document A (500 tokens) + EOS + Document B (1200 tokens) + EOS +
    Document C (2394 tokens) = 4096 tokens (one training sequence)

The packing is done during preprocessing — the training loop simply
reads contiguous chunks of tokens, maximizing throughput.
"""

import json
import os
from typing import List, Optional, Tuple

import numpy as np

from prism.data.tokenizer import PrismTokenizer


class SequencePacker:
    """Packs tokenized documents into fixed-length sequences.

    Documents are concatenated with EOS tokens between them, then
    chunked into sequences of exactly `max_seq_len` tokens. Any
    leftover tokens at the end are discarded (negligible data loss).

    Args:
        tokenizer: PrismTokenizer instance.
        max_seq_len: Target sequence length (e.g., 4096).
    """

    def __init__(self, tokenizer: PrismTokenizer, max_seq_len: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_id = tokenizer.eos_id

        # Buffer accumulates tokens across documents
        self._buffer: List[int] = []
        self._packed_sequences: List[np.ndarray] = []

    def add_document(self, text: str) -> None:
        """Add a document to the packing buffer.

        The document is tokenized and appended to the internal buffer
        with an EOS separator. When the buffer reaches max_seq_len,
        a packed sequence is extracted.

        Args:
            text: Raw document text.
        """
        # Tokenize without BOS (we're packing, not individual sequences)
        tokens = self.tokenizer.encode(text, add_bos=False, add_eos=True)
        self._buffer.extend(tokens)

        # Extract full sequences from buffer
        while len(self._buffer) >= self.max_seq_len:
            sequence = self._buffer[:self.max_seq_len]
            self._buffer = self._buffer[self.max_seq_len:]
            self._packed_sequences.append(
                np.array(sequence, dtype=np.uint16)
            )

    def add_tokens(self, token_ids: List[int]) -> None:
        """Add pre-tokenized tokens to the buffer.

        Args:
            token_ids: List of token IDs (should end with EOS).
        """
        self._buffer.extend(token_ids)

        while len(self._buffer) >= self.max_seq_len:
            sequence = self._buffer[:self.max_seq_len]
            self._buffer = self._buffer[self.max_seq_len:]
            self._packed_sequences.append(
                np.array(sequence, dtype=np.uint16)
            )

    def flush(self) -> List[np.ndarray]:
        """Return all packed sequences and clear the internal state.

        Any remaining tokens in the buffer that don't fill a complete
        sequence are discarded.

        Returns:
            List of numpy arrays, each of shape (max_seq_len,) with dtype uint16.
        """
        sequences = self._packed_sequences
        self._packed_sequences = []
        self._buffer = []
        return sequences

    @property
    def num_ready(self) -> int:
        """Number of packed sequences ready to be flushed."""
        return len(self._packed_sequences)

    @property
    def buffer_size(self) -> int:
        """Current number of tokens in the buffer."""
        return len(self._buffer)


def save_packed_shards(
    sequences: List[np.ndarray],
    output_dir: str,
    shard_size: int = 100_000,
    prefix: str = "train",
    start_shard_idx: int = 0,
) -> Tuple[List[str], int]:
    """Save packed sequences as memory-mapped binary shards.

    Each shard is a flat numpy array of uint16 token IDs that can
    be memory-mapped during training for zero-copy data loading.

    Args:
        sequences: List of packed numpy arrays.
        output_dir: Directory to save shards.
        shard_size: Number of sequences per shard.
        prefix: Filename prefix for shards.
        start_shard_idx: Starting index for naming shard files.

    Returns:
        Tuple of (list of shard file paths, next starting shard index).
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_paths = []
    current_shard_idx = start_shard_idx

    for idx in range(0, len(sequences), shard_size):
        shard_data = sequences[idx:idx + shard_size]
        shard_array = np.concatenate(shard_data)

        shard_filename = f"{prefix}_{current_shard_idx:05d}.bin"
        shard_path = os.path.join(output_dir, shard_filename)

        # Save as raw binary
        shard_array.tofile(shard_path)
        shard_paths.append(shard_path)
        current_shard_idx += 1

        print(
            f"  Saved shard {shard_filename}: "
            f"{len(shard_data)} sequences, "
            f"{shard_array.nbytes / (1024**2):.1f} MB"
        )

    # Update / merge metadata
    meta_path = os.path.join(output_dir, f"{prefix}_metadata.json")
    existing_meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                existing_meta = json.load(f)
        except Exception:
            existing_meta = {}

    total_seqs = existing_meta.get("total_sequences", 0) + len(sequences)
    total_shards = existing_meta.get("num_shards", 0) + len(shard_paths)
    seq_len = len(sequences[0]) if sequences else existing_meta.get("seq_length", 4096)

    metadata = {
        "num_shards": total_shards,
        "total_sequences": total_seqs,
        "seq_length": seq_len,
        "dtype": "uint16",
        "shard_size": shard_size,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return shard_paths, current_shard_idx

