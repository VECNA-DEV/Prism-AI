# Prism AI (10B) Architecture Deep Dive

## Overview

Prism-10B is a decoder-only transformer engineered specifically for code generation and programming language reasoning. It incorporates modern architectural best practices from LLaMA-2, Mistral, and DeepSeek.

---

## 1. Grouped-Query Attention (GQA)

Multi-Head Attention (MHA) creates independent Key ($K$) and Value ($V$) heads for every Query ($Q$) head. At scale, the memory required to cache $K$ and $V$ tensors during autoregressive generation becomes a major bottleneck.

Prism-10B adopts a **4:1 Grouped-Query Attention** design:
- **Query Heads ($N_q$)**: 32
- **Key/Value Heads ($N_{kv}$)**: 8
- **Head Dimension ($d_{\text{head}}$)**: 128
- **Group Ratio ($G$)**: 4 ($32 / 8 = 4$)

### Memory Comparison (Batch Size = 1, Sequence Length = 4096, 46 Layers, FP16):
$$\text{KV Memory} = 2 \times N_{\text{layers}} \times S \times N_{kv} \times d_{\text{head}} \times 2 \text{ bytes}$$

- **Standard MHA (32 KV Heads)**: $2 \times 46 \times 4096 \times 32 \times 128 \times 2 \approx \mathbf{30.8\text{ GB}}$
- **Prism GQA (8 KV Heads)**: $2 \times 46 \times 4096 \times 8 \times 128 \times 2 \approx \mathbf{7.7\text{ GB}}$
- **Reduction**: **75% reduction in KV-cache footprint**, allowing 4× larger batch sizes on the same hardware.

---

## 2. Rotary Position Embeddings (RoPE)

Unlike absolute positional embeddings that are learned as lookup tables or sinusoidal additions, RoPE applies a multiplicative complex rotation to Query and Key representations at position $m$:

$$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, \dots, R_{\theta_{d/2}, m}\right)$$

where $\theta_i = 10000^{-2(i-1)/d}$.

This guarantees that the inner product between query $q_m$ and key $k_n$ is a direct function of their relative distance $m - n$:

$$\langle R_{\Theta, m} q_m, R_{\Theta, n} k_n \rangle = g(q_m, k_n, m - n)$$

---

## 3. SwiGLU Gated Feed-Forward Network

Standard FFNs apply $\text{FFN}(x) = \text{GELU}(x W_1) W_2$. Prism-10B employs the SwiGLU variant:

$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$

- Hidden Dimension ($d_{\text{model}}$): 4096
- Intermediate Dimension ($d_{\text{mlp}}$): 14,336 ($\approx \frac{8}{3} d_{\text{model}}$, aligned to multiples of 256 for tensor core tiling efficiency).

---

## 4. Parameter Breakdown

| Component | Dimensions | Parameter Calculation | Total Count |
|---|---|---|---|
| **Token Embeddings** | $32000 \times 4096$ | $32000 \times 4096$ | $131,072,000$ |
| **Attention $W_q$** | $4096 \times (32 \times 128)$ | $4096 \times 4096 \times 46$ | $771,751,936$ |
| **Attention $W_k, W_v$** | $4096 \times (8 \times 128)$ | $2 \times (4096 \times 1024) \times 46$ | $385,875,968$ |
| **Attention $W_o$** | $4096 \times 4096$ | $4096 \times 4096 \times 46$ | $771,751,936$ |
| **SwiGLU $W_{\text{gate}}, W_{\text{up}}$** | $4096 \times 14336$ | $2 \times (4096 \times 14336) \times 46$ | $5,401,264,128$ |
| **SwiGLU $W_{\text{down}}$** | $14336 \times 4096$ | $(14336 \times 4096) \times 46$ | $2,700,632,064$ |
| **RMSNorm Weights** | $4096$ | $(2 \times 4096 \times 46) + 4096$ | $380,928$ |
| **Tied LM Head** | $4096 \times 32000$ | (Shared with Token Embeddings) | $0$ |
| **Grand Total** | — | — | **$10,163,433,472$ ($\approx 10.16\text{B}$)** |
