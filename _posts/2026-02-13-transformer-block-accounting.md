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