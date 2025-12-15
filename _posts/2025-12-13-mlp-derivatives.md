---
layout: post
title: MLP Standard Derivatives Derivation
date: 2025-12-13 09:14:00-0400
featured: False
description: MLP Standard Derivatives Derivation
tags: ML
categories: 
giscus_comments: true
related_posts: false

---

**Softmax Definition**

$$
y_i = \mathrm{softmax}(a)_i = \frac{e^{a_i}}{\sum_k e^{a_k}}
\qquad\text{let } S = \sum_k e^{a_k}.
$$


**Softmax Jacobian:** $$ \frac{\partial y_i}{\partial a_j} $$

Start from:

$$
y_i = \frac{e^{a_i}}{S}.
$$

Differentiate w.r.t. $$a_j$$ using the quotient rule:

$$
\frac{\partial y_i}{\partial a_j}
= \frac{S\cdot\frac{\partial}{\partial a_j}e^{a_i}
      \;-\;
      e^{a_i}\cdot\frac{\partial S}{\partial a_j}}{S^2}.
$$

Compute derivatives:

- $$ \frac{\partial}{\partial a_j} e^{a_i} = e^{a_i}\delta_{ij} $$
- $$ \frac{\partial S}{\partial a_j} = e^{a_j} $$

Substitute:

$$
\frac{\partial y_i}{\partial a_j}
= \frac{S(e^{a_i}\delta_{ij}) - e^{a_i}e^{a_j}}{S^2}
= \frac{e^{a_i}}{S^2}(S\delta_{ij} - e^{a_j}).
$$

Use softmax definitions:

$$
y_i = \frac{e^{a_i}}{S}, \qquad
y_j = \frac{e^{a_j}}{S}.
$$

Final form:

$$
\frac{\partial y_i}{\partial a_j}
= y_i\delta_{ij} - y_i y_j
= y_i (\delta_{ij} - y_j).
$$


**Cross-Entropy Loss**

For one-hot target $$t$$:

$$
L = -\sum_i t_i \log y_i.
$$

Differentiate w.r.t. $$a_j$$:

$$
\frac{\partial L}{\partial a_j}
= \sum_i \frac{\partial L}{\partial y_i}
       \frac{\partial y_i}{\partial a_j}
= \sum_i \left( -\frac{t_i}{y_i} \right)
           y_i(\delta_{ij} - y_j).
$$

Simplify:

$$
= \sum_i -t_i(\delta_{ij} - y_j)
= -\sum_i t_i\delta_{ij}
  + \sum_i t_i y_j.
$$

Use:
- $$ \sum_i t_i\delta_{ij} = t_j $$
- $$ \sum_i t_i = 1 $$ for one-hot $$ t $$

Thus:

$$
\frac{\partial L}{\partial a_j}
= -t_j + y_j
= y_j - t_j.
$$



$$
\boxed{
\frac{\partial L}{\partial a_j} = y_j - t_j
}
$$

**Note:**  
$$\delta_{ij}$$ is the **Kronecker delta**, defined as

$$
\delta_{ij} =
\begin{cases}
1, & i = j \\
0, & i \ne j
\end{cases}
$$

It acts as an index selector in summations.  
For example:

$$
\sum_i a_i \delta_{ij} = a_j.
$$

In the softmax Jacobian derivation, $$\delta_{ij}$$ appears because

$$
\frac{\partial e^{a_i}}{\partial a_j} = e^{a_i} \delta_{ij},
$$

meaning only the term with matching indices contributes to the derivative.

**Sigmoid Derivative (Element-wise)**

The sigmoid function is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}.
$$

Let the activation be applied element-wise:
$$
y_i = \sigma(a_i).
$$


**Derivative of Sigmoid**

Differentiate w.r.t. $$a_i$$:

$$
\frac{d\sigma(x)}{dx}
= \frac{d}{dx}\left( \frac{1}{1 + e^{-x}} \right)
= \frac{e^{-x}}{(1 + e^{-x})^2}.
$$

Rewrite in terms of $$\sigma(x)$$:

$$
\sigma(x) = \frac{1}{1 + e^{-x}},
\qquad
1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}.
$$

Thus:

$$
\frac{d\sigma(x)}{dx}
= \sigma(x)\bigl(1 - \sigma(x)\bigr).
$$


**Jacobian Form**

Since sigmoid acts independently on each component,

$$
\frac{\partial y_i}{\partial a_j}
= \sigma(a_i)\bigl(1 - \sigma(a_i)\bigr)\delta_{ij}.
$$

The Jacobian is diagonal.

**Derivative of Matrix Multiplication**

Let:

$$
Y = A X
$$

where:
- $$A \in \mathbb{R}^{m \times n}$$
- $$X \in \mathbb{R}^{n \times p}$$
- $$Y \in \mathbb{R}^{m \times p}$$

**Gradient with Respect to $$A$$**

$$
\frac{\partial L}{\partial A}
=
\frac{\partial L}{\partial Y}
\, X^\top
$$

**Gradient with Respect to $$X$$**

$$
\frac{\partial L}{\partial X}
=
A^\top
\frac{\partial L}{\partial Y}
$$

**Matrix Multiplication Gradients in `einsum` Form**

Let:

$$
Y = A X
$$

with components:

$$
Y_{ij} = \sum_k A_{ik} X_{kj}.
$$

Let the upstream gradient be:

$$
G_{ij} = \frac{\partial L}{\partial Y_{ij}}.
$$

**Gradient with Respect to \(A\)**

Using the chain rule:

$$
\frac{\partial L}{\partial A_{ik}}
= \sum_j \frac{\partial L}{\partial Y_{ij}}
         \frac{\partial Y_{ij}}{\partial A_{ik}}
= \sum_j G_{ij} X_{kj}.
$$

In Einstein summation notation:

$$
\frac{\partial L}{\partial A_{ik}} = G_{ij} X_{kj}.
$$

**`einsum` implementation:**
```python
Y = np.einsum("ik,kj->ij", A, X)


dLdX = np.einsum("ik,ij->kj", A, G)
dLdA = np.einsum("ij,kj->ik", G, X)
```