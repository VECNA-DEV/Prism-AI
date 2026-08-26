# Prism AI — Distributed Training Guide (DeepSpeed ZeRO-3)

## Overview

Training a 10.16B parameter model requires substantial memory and compute. Prism AI uses **DeepSpeed ZeRO-3** to partition model states across multi-GPU nodes.

---

## 1. Hardware Requirements

| Setup | Recommended Hardware | Effective Batch Size |
|---|---|---|
| **Single-Node Multi-GPU** | 8× NVIDIA A100-80GB SXM4 or H100-80GB | ~2M tokens / step |
| **Multi-Node Cluster** | 4 nodes × 8× A100-80GB (32 GPUs, InfiniBand HDR) | ~8M tokens / step |

---

## 2. Launching Training

### Single Node (8 GPUs)
```bash
deepspeed --num_gpus=8 scripts/train.py \
    --model_config configs/model_10b.yaml \
    --train_config configs/training_10b.yaml \
    --deepspeed configs/deepspeed/ds_zero3.json
```

### Multi-Node Cluster (Torchrun / Slurm)
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    scripts/train.py \
    --model_config configs/model_10b.yaml \
    --train_config configs/training_10b.yaml \
    --deepspeed configs/deepspeed/ds_zero3.json
```

---

## 3. ZeRO-3 Memory Breakdown

For a 10.16B model in mixed precision:
- **Optimizer State Partitioning**: 16-byte states partitioned across $N$ GPUs.
- **Gradient Partitioning**: 2-byte gradients partitioned across $N$ GPUs.
- **Parameter Partitioning**: 2-byte weights partitioned across $N$ GPUs.
- **Activation Checkpointing**: Activations are discarded during forward pass and recomputed during backward pass, saving over 60% dynamic memory.
