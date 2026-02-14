---
layout: post
title: Transformer Block FLOPs & Parameters Calculations
date: 2026-02-13 10:31:00-0400
featured: false
description: Resource accounting for Transformer block
tags: 
categories: LLMR-NYU
giscus_comments: true
related_posts: false
---


<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/tf-block.png" title="tf-block" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

#### **Notation**

- $$B$$: Batch size
- $$S$$: Sequence length  
- $$D$$: Hidden dimension / Model dimension
- $$N$$: Number of attention heads
- $$H$$: Head dimension $$(H = D/N)$$
- $$F$$: Feed-forward hidden dimension (typically $$4D$$)
- $$S_1$$/$$S_2$$: Sequence length in attention context, $$S_1$$=of query & $$S_2$$=of key (often $$S$$)
- $$\text{SiLU}$$: Sigmoid Linear Unit activation (also known as Swish)
- $$\text{RoPE}$$: Rotary Position Embeddings
- $$[X \times Y]$$: Matrix/tensor dimensions
- $$@$$: Matrix multiplication operator
- $$[X]\cdot[Y]$$: Elementwise multiplication/dot-product
- $$L$$: Number of layers

#### **Transformer Block Operations**

##### **Input**

$$X = [B \times S \times D] \quad \text{(Input tensor)}$$

##### **RMS Normalization 1**

$$\text{Rmsnorm}(x) = \text{normed\_x} * \text{gains}, \quad [B \times S \times D] \cdot [D] \rightarrow \quad [B \times S \times D]$$

##### **Multi-Head Attention (MHA)**

Attention applied on Rmsnorm output:

- **(a)** $$Q = W_q @ x$$, $$\quad [D \times D] @ [B \times S \times D] \rightarrow [B \times S \times D]$$

- **(b)** $$K = W_k @ x$$, $$\quad [D \times D] @ [B \times S \times D] \rightarrow [B \times S \times D]$$

- **(c)** $$V = W_v @ x$$, $$\quad [D \times D] @ [B \times S \times D] \rightarrow [B \times S \times D]$$

- **(d)** Rearrange for multi-head attention($$Q, K, V$$):
  $$[B \times S \times D] \rightarrow [B \times S \times (N \cdot H)] \rightarrow [B \times N \times S \times H]$$

- **(e)** Apply RoPE to $$Q$$ and $$K$$ (element-wise) $$\rightarrow [B \times N \times S \times H]$$

- **(f)** Scaled Dot-Product Attention with Causal Mask:
  $$\begin{align}
  QK^T &= [B \times N \times S_1 \times H] @ [B \times N \times S_2 \times H]^T \rightarrow [B \times N \times S_1 \times S_2] \\
  \text{output} &= \text{attn-weights} @ V = [B \times N \times S_1 \times S_2] @ [B \times N \times S_2 \times H] \rightarrow [B \times N \times S_1 \times H]
  \end{align}$$

- **(g)** Rearrange back:
  $$[B \times N \times S \times H] \rightarrow [B \times S \times (N \cdot H)] \rightarrow [B \times S \times D]$$

- **(h)** Output Projection:
  $$X_{\text{out}} = W_o @ X_{\text{attn}}, \quad [D \times D] @ [B \times S \times D] \rightarrow [B \times S \times D]$$

##### **RMS Normalization 2**

$$\text{Rmsnorm}_2 = [B \times S \times D] \cdot [D] \rightarrow [B \times S \times D]$$

##### **Feed-Forward Network (FFN)**

1. **Gate projection:**
   $$\text{Gate} = W_1 @ X, \quad [D \times F] @ [B \times S \times D] \rightarrow [B \times S \times F]$$

2. **Activation:**
   $$\text{Gate} = \text{SiLU}(\text{Gate}) \quad \text{(element-wise activation)}$$

3. **Linear projection:**
   $$\text{Linear} = W_3 @ X, \quad [D \times F] @ [B \times S \times D] \rightarrow [B \times S \times F]$$

4. **Gated multiplication:**
   $$\text{Gated} = \text{Gate} \otimes \text{linear} \quad \text{(element-wise multiplication)}$$

5. **Output projection:**
   $$\text{Net} = W_2 @ \text{Gated}, \quad [F \times D] @ [B \times S \times F] \rightarrow [B \times S \times D]$$

#### **FLOPs and Parameter Count**

##### **Forward Pass FLOPs**

$$\begin{align}
\text{Q, K, V Projections:} &\quad 3 \times [2 \cdot D \cdot (B \cdot S \cdot D)] = 6 \cdot B \cdot S \cdot D^2 \\
\text{Attention QK}^T\text{:} &\quad 2 \cdot B \cdot N \cdot S^2 \cdot H = 2 \cdot B \cdot S^2 \cdot D \\
\text{Attention weights @ V:} &\quad [2 \cdot S \cdot (B \cdot N \cdot S \cdot H)] = 2 \cdot B \cdot S^2 \cdot D \\
\text{O Projections:} &\quad 2 \cdot D \cdot (B \cdot S \cdot D) = 2 \cdot B \cdot S \cdot D^2 \\
\text{FFN Linear 1 (Gate + Linear):} &\quad 2 \times [2 \cdot D \cdot (B \cdot S \cdot F)] = 4 \cdot B \cdot S \cdot D \cdot F \\
\text{FFN Linear 2 (Net):} &\quad 2 \cdot F \cdot (B \cdot S \cdot D) = 2 \cdot B \cdot S \cdot D \cdot F \\
\textbf{Total FLOPs:} &\quad \boxed{8 \cdot B \cdot S \cdot D^2 + 4 \cdot B \cdot S^2 \cdot D + 6 \cdot B \cdot S \cdot D \cdot F}
\end{align}$$

##### **Parameter Count**

$$\begin{align}
\text{Q Projection:} &\quad W_q: [D \times D] = D^2 \\
\text{K Projection:} &\quad W_k: [D \times D] = D^2 \\
\text{V Projection:} &\quad W_v: [D \times D] = D^2 \\
\text{O Projection:} &\quad W_o: [D \times D] = D^2 \\
\text{FFN W}_1\text{:} &\quad [D \times F] = D \cdot F \\
\text{FFN W}_3\text{:} &\quad [D \times F] = D \cdot F \\
\text{FFN W}_2\text{:} &\quad [F \times D] = F \cdot D \\
\text{Total Parameters:} &\quad 4D^2 + 3DF
\end{align}$$

#### **Others/outside Transformer Block: Unembed Projection (LM Output Head)**

$$\begin{align}
\text{Unembed:} &\quad W_{\text{out}} @ X, \quad [D \times V] @ [B \times S \times D] \rightarrow [B \times S \times V] \\
\text{FLOPs:} &\quad 2 \cdot D \cdot (B \cdot S \cdot V) = 2 \cdot B \cdot S \cdot D \cdot V \\
\text{Parameters:} &\quad D \cdot V
\end{align}$$


#### **AdamW Memory Accounting**

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/adamw.png" title="adamw" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

##### Problem Setup
Calculate peak memory required for training a Transformer LM with AdamW optimizer.

**Assumptions:**
- All tensors in **float32** (4 bytes per element)
- Batch size: `B`
- Context length: `T`
- Vocabulary size: `V`
- Number of layers: `L`
- Model dimension: `d`
- Number of heads: `h`
- Feed-forward dimension: `d_ff = 4d`


##### **1. Parameters Memory**

**Embedding Layers:**
- Token embeddings: `V × d`

**Per Transformer Block:**
- RMSNorm (pre-attention): `d`
- Attention QKV projection: `3d²`
- Attention output projection: `d²`
- RMSNorm (pre-FFN): `d`
- FFN W1: `4d²`
- FFN W2: `4d²`

**Per block:** `2d + 12d²`

**Final Layers:**
- Final RMSNorm: `d`
- Output embedding (unembedding): `d × V`

**Total Parameters:**
```
P = Vd + L(2d + 12d²) + d + dV
P = 2Vd + L(2d + 12d²) + d
P = d(2V + 2L + 1) + 12Ld²
```

**Parameters Memory:** `4P` bytes

##### **2. Activations Memory**

Tracking intermediate activations needed for backward pass.

**Per Transformer Block:**

| Component | Shape | Elements |
|-----------|-------|----------|
| Input to block | `(B, T, d)` | `BTd` |
| RMSNorm output (pre-attn) | `(B, T, d)` | `BTd` |
| RMSNorm statistics | `(B, T)` | `BT` |
| Q, K, V projections | `3 × (B, T, d)` | `3BTd` |
| Attention scores (Q^T K) | `(B, h, T, T)` | `BhT²` |
| Softmax statistics | `(B, h, T)` | `BhT` |
| Attention weights (post-softmax) | `(B, h, T, T)` | `BhT²` |
| Weighted sum output | `(B, T, d)` | `BTd` |
| Attention output projection | `(B, T, d)` | `BTd` |
| RMSNorm output (pre-FFN) | `(B, T, d)` | `BTd` |
| RMSNorm statistics | `(B, T)` | `BT` |
| FFN W1 output | `(B, T, 4d)` | `4BTd` |
| SiLU output | `(B, T, 4d)` | `4BTd` |
| FFN W2 output | `(B, T, d)` | `BTd` |

**Per block total:**
```
15BTd + 2BhT² + BhT + 2BT
```

**All L blocks:**
```
L(15BTd + 2BhT² + BhT + 2BT)
```

**Final Layers:**

| Component | Shape | Elements |
|-----------|-------|----------|
| Final RMSNorm output | `(B, T, d)` | `BTd` |
| Final RMSNorm statistics | `(B, T)` | `BT` |
| Logits (output embedding) | `(B, T, V)` | `BTV` |
| Softmax probabilities | `(B, T, V)` | `BTV` |

**Total Activations:**
```
A = L(15BTd + 2BhT² + BhT + 2BT) + BTd + BT + 2BTV
A = BTd(15L + 1) + 2BhT²L + BhTL + 2BTL + BT + 2BTV
```

**Activations Memory:** `4A` bytes

##### **3. Gradients Memory**

Gradients have the same shape as parameters.

**Gradients Memory:** `4P` bytes

##### **4. Optimizer State (AdamW)**

AdamW maintains two states per parameter:
- **First moment (momentum):** `m_t` (same shape as params)
- **Second moment (variance):** `v_t` (same shape as params)

**Optimizer State Memory:** `2 × 4P = 8P` bytes

##### **5. Total Peak Memory**

```
Total = Parameters + Activations + Gradients + Optimizer State
Total = 4P + 4A + 4P + 8P
Total = 16P + 4A
```

**Substituting P and A:**

```
Total = 16[d(2V + 2L + 1) + 12Ld²] + 4[BTd(15L + 1) + 2BhT²L + BhTL + 2BTL + BT + 2BTV]
```

**Expanded:**

```
Total = 32Vd + 32Ld + 16d + 192Ld² + 60BTdL + 4BTd + 8BhT²L + 4BhTL + 8BTL + 4BT + 8BTV
```

| Component | Memory (bytes) | Multiplier of P |
|-----------|----------------|-----------------|
| **Parameters** | `4P` | `1×` |
| **Gradients** | `4P` | `1×` |
| **Optimizer State** | `8P` | `2×` |
| **Activations** | `4A` | - |
| **Total** | `16P + 4A` | - |

**Key Observations:**
- Optimizer state dominates static memory: `8P` (2× params)
- Total static memory (params + grads + optimizer): `16P`
- Activations scale with batch size and sequence length: `O(BT)`
- Static memory scales with model size: `O(Ld²)`

**Where:**
- `P = d(2V + 2L + 1) + 12Ld²`
- `A = BTd(15L + 1) + 2BhT²L + BhTL + 2BTL + BT + 2BTV`