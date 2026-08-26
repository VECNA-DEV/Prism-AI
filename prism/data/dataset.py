"""Pre-tokenized Dataset for Prism AI training.

Loads packed, tokenized binary shards via memory-mapping for
zero-copy, high-throughput data loading during training.

Data format (produced by prism/data/packing.py):
  - Each shard: flat uint16 array of concatenated sequences
  - Each sequence: (max_seq_len + 1) tokens (+1 for the label shift)
  - Sequences are packed (multiple documents per sequence)

During training, input_ids = tokens[:-1] and labels = tokens[1:],
implementing the standard next-token prediction objective.
"""

import os
import json
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PreTokenizedDataset(Dataset):
    """Memory-mapped dataset of pre-tokenized sequences.

    Uses numpy memory-mapping to load data on-demand without reading
    the entire dataset into RAM. This is critical for large-scale
    training where the tokenized dataset can be hundreds of GB.

    Args:
        data_dir: Directory containing .bin shards and metadata JSON.
        max_seq_len: Sequence length (must match what was used during packing).
        split: Data split prefix (e.g., "train", "val").
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 4096,
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.split = split

        # Each sequence has max_seq_len tokens
        self.seq_stride = max_seq_len

        # Load metadata
        meta_path = os.path.join(data_dir, f"{split}_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Discover and memory-map all shards
        self._shards: List[np.ndarray] = []
        self._shard_sizes: List[int] = []  # Number of sequences per shard
        self._cumulative_sizes: List[int] = []

        self._load_shards()

    def _load_shards(self) -> None:
        """Discover and memory-map all binary shards."""
        shard_files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.startswith(self.split) and f.endswith(".bin")
        ])

        if not shard_files:
            raise FileNotFoundError(
                f"No {self.split}_*.bin shards found in {self.data_dir}. "
                f"Run scripts/preprocess_data.py first."
            )

        cumulative = 0
        for shard_file in shard_files:
            shard_path = os.path.join(self.data_dir, shard_file)

            # Memory-map the shard (read-only, copy-on-write)
            mmap = np.memmap(shard_path, dtype=np.uint16, mode="r")

            # Calculate number of complete sequences in this shard
            num_sequences = len(mmap) // self.seq_stride

            if num_sequences == 0:
                continue

            # Trim to exact multiple of seq_stride
            mmap = mmap[:num_sequences * self.seq_stride]

            self._shards.append(mmap)
            self._shard_sizes.append(num_sequences)
            cumulative += num_sequences
            self._cumulative_sizes.append(cumulative)

        print(
            f"Loaded {len(self._shards)} shards with "
            f"{self._cumulative_sizes[-1] if self._cumulative_sizes else 0} "
            f"total sequences (seq_len={self.max_seq_len})"
        )

    def _locate_sequence(self, idx: int):
        """Find which shard and offset a global index maps to.

        Args:
            idx: Global sequence index.

        Returns:
            Tuple of (shard_index, local_index_within_shard).
        """
        # Binary search for the shard containing this index
        for shard_idx, cum_size in enumerate(self._cumulative_sizes):
            if idx < cum_size:
                # Local index within this shard
                prev_cum = self._cumulative_sizes[shard_idx - 1] if shard_idx > 0 else 0
                local_idx = idx - prev_cum
                return shard_idx, local_idx

        raise IndexError(f"Index {idx} out of range (dataset has {len(self)} sequences)")

    def __len__(self) -> int:
        """Total number of sequences across all shards."""
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example.

        Returns input_ids and labels with the standard next-token
        prediction shift: input_ids = tokens[:-1], labels = tokens[1:].

        Args:
            idx: Sequence index.

        Returns:
            Dict with 'input_ids' and 'labels' tensors of shape (max_seq_len,).
        """
        shard_idx, local_idx = self._locate_sequence(idx)

        # Read sequence from memory-mapped shard
        start = local_idx * self.seq_stride
        end = start + self.seq_stride
        tokens = self._shards[shard_idx][start:end].astype(np.int64)

        # Return full sequence as input_ids and matching labels
        # The model's forward pass performs the next-token shift:
        # logits[:-1] predicts labels[1:]
        input_ids = torch.from_numpy(tokens.copy())
        labels = torch.from_numpy(tokens.copy())

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def get_total_tokens(self) -> int:
        """Total number of tokens in the dataset."""
        return len(self) * self.max_seq_len

    def close(self) -> None:
        """Close all memory-mapped shards to release OS file handles."""
        for shard in self._shards:
            try:
                if hasattr(shard, "_mmap") and shard._mmap is not None:
                    shard._mmap.close()
            except Exception:
                pass
        self._shards.clear()


class StreamingCodeDataset:
    """Streaming dataset that processes data from HuggingFace on-the-fly.

    Used during tokenizer training and data preprocessing to avoid
    downloading the entire dataset upfront. Streams from The Stack v2
    with configurable language filtering and quality controls.

    This is NOT used during model training (we use PreTokenizedDataset
    for that). This is for the preprocessing pipeline only.

    Args:
        dataset_name: HuggingFace dataset identifier.
        languages: List of programming languages to include.
        streaming: Whether to use streaming mode.
    """

    def __init__(
        self,
        dataset_name: str = "bigcode/the-stack-v2-dedup",
        languages: Optional[List[str]] = None,
        streaming: bool = True,
    ):
        from datasets import load_dataset

        self.dataset_name = dataset_name
        self.languages = languages

        # Load with streaming to avoid downloading the full dataset
        self.dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=streaming,
            trust_remote_code=True,
        )

    def iter_samples(self, max_samples: Optional[int] = None):
        """Iterate over code samples with optional limit.

        Yields:
            Dict with 'content' (code text) and 'lang' (language).
        """
        count = 0
        for sample in self.dataset:
            # The Stack v2 uses 'content' for code and 'lang' for language
            content = sample.get("content", "")
            lang = sample.get("lang", sample.get("language", "unknown"))

            # Language filter
            if self.languages and lang.lower() not in {l.lower() for l in self.languages}:
                continue

            yield {"content": content, "lang": lang}

            count += 1
            if max_samples is not None and count >= max_samples:
                break
