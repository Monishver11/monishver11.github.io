---
layout: post
title: SVM Solution in the Span of the Data
date: 2025-01-16 14:13:00-0400
featured: false
description: This blog explores how the span property simplifies optimization in SVM and ridge regression, introduces the Representer Theorem, and highlights the computational benefits of kernelization.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

Previously, we explored the **kernel trick**, a powerful concept that allows Support Vector Machines (SVMs) to operate efficiently in high-dimensional feature spaces without explicitly computing the coordinates. Building on that foundation, we now turn our attention to an intriguing property of SVM solutions: they lie in the **span of the data**. This observation not only deepens our understanding of the connection between the dual and primal formulations of SVM but also provides a unifying perspective on how solutions in machine learning are inherently tied to the training data.

---

#### **SVM Dual Problem: A Quick Recap**

To understand this property, let’s first revisit the SVM dual problem. It is formulated as:

$$
\sup_{\alpha \in \mathbb{R}^n} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_j^T x_i
$$

subject to the constraints:

1. $$\sum_{i=1}^n \alpha_i y_i = 0$$
2. $$\alpha_i \in [0, \frac{c}{n}], \quad i = 1, \dots, n$$

Here, $$\alpha_i$$ are the dual variables that correspond to the Lagrange multipliers, and $$c$$ is the regularization parameter that controls the margin.

The dual problem focuses on maximizing this quadratic function, which involves pairwise interactions between training samples. Once the optimal dual solution $$\alpha^*$$ is obtained, it can be used to compute the primal solution as:

$$
w^* = \sum_{i=1}^n \alpha^*_i y_i x_i
$$

This equation reveals a critical insight: the primal solution $$w^*$$ is expressed as a **linear combination of the training inputs** $$x_1, x_2, \dots, x_n$$. This means that $$w^*$$ is confined to the span of these inputs, or mathematically:

$$
w^* \in \text{span}(x_1, \dots, x_n)
$$

We refer to this phenomenon as "the SVM solution lies in the span of the data." It underscores the dependency of $$w^*$$ on the training data, aligning it with the geometric intuition of SVMs: the decision boundary is shaped by a subset of data points (the support vectors).

---

#### **Ridge Regression: Another Perspective on Span of the Data**

Interestingly, this concept is not unique to SVMs. A similar property emerges in **ridge regression**, a linear regression method that incorporates $$\ell_2$$ regularization to prevent overfitting. Let’s delve into this and see how the ridge regression solution also resides in the span of the data.

##### **Ridge Regression Objective**

The objective function for ridge regression, with a regularization parameter $$\lambda > 0$$, is given by:

$$
w^* = \arg\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \{ w^T x_i - y_i \}^2 + \lambda \|w\|_2^2
$$

Here, $$w$$ is the weight vector, $$x_i$$ are the input data points, and $$y_i$$ are the corresponding target values. The regularization term $$\lambda \|w\|_2^2$$ penalizes large weights to improve generalization.

###### **The Closed-Form Solutio**n

The ridge regression problem has a closed-form solution:

$$
w^* = \left( X^T X + \lambda I \right)^{-1} X^T y
$$

where $$X$$ is the design matrix (rows are $$x_1, \dots, x_n$$), and $$y$$ is the vector of target values.

At first glance, this expression might seem abstract. However, by rearranging it, we can show that the solution also lies in the span of the training data.


##### **Showing the Span Property for Ridge Regression**

To reveal the span property, let’s rewrite the ridge regression solution. Using matrix algebra:

$$
w^* = \left( X^T X + \lambda I \right)^{-1} X^T y
$$

We can express $$w^*$$ as:

$$
w^* = X^T \left[ \frac{1}{\lambda} y - \frac{1}{\lambda} X w^* \right]  \tag{1}
$$

Now, define:

$$
\alpha^* = \frac{1}{\lambda} y - \frac{1}{\lambda} X w^*
$$

Substituting this back, we get:

$$
w^* = X^T \alpha^*
$$

Expanding further, it becomes:

$$
w^* = \sum_{i=1}^n \alpha^*_i x_i
$$

This clearly shows that the ridge regression solution $$w^*$$ is also a linear combination of the training inputs. Thus, like SVMs, ridge regression solutions lie in the span of $$x_1, x_2, \dots, x_n$$:

$$
w^* \in \text{span}(x_1, \dots, x_n)
$$

You may wonder how we arrived at this specific form for $$w^*$$ at the start $$(1)$$. To understand this, we utilized the following lemma and a series of transformations to reframe it accordingly.


###### **The Lemma: Matrix Inverse Decomposition**

We use the following lemma:

**If $$A$$ and $$A + B$$ are non-singular, then:**

$$
(A + B)^{-1} = A^{-1} - A^{-1} B (A + B)^{-1}
$$

This allows us to break down the inverse of a sum of matrices into manageable parts. Let’s apply it to our problem.


###### **Applying the Lemma to Ridge Regression**

Let:
- $$A = \lambda I$$ (scaled identity matrix),
- $$B = X^T X$$ (Gram matrix).

Substituting into the ridge regression solution:

$$
w^* = (X^T X + \lambda I)^{-1} X^T y
$$

Using the lemma, we expand the inverse:

$$
w^* = \left( \lambda^{-1} - \lambda^{-1} X^T X (X^T X + \lambda I)^{-1} \right) X^T y
$$


We simplify the terms step by step:

1. **Expand the first term:**
   
   $$ 
   w^* = X^T \lambda^{-1} y - \lambda^{-1} X^T X (X^T X + \lambda I)^{-1} X^T y 
   $$

2. **Notice the recursive structure:**
   
   $$ 
   w^* = X^T \lambda^{-1} y - \lambda^{-1} X^T X w^* 
   $$

3. **Rearrange to highlight the span of data:**
   
   $$ 
   w^* = X^T \left( \frac{1}{\lambda} y - \frac{1}{\lambda} X w^* \right)
   $$


###### **Supporting Details**

To solidify our understanding, here’s how the **Matrix Sum Inverse Lemma** is derived using the Woodbury identity:

**The Woodbury Identity:**

$$
(A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}
$$

Substituting:
- $$C = I$$, $$V = I$$, and $$U = B$$, we get:

$$
(A + B)^{-1} = A^{-1} - A^{-1} B (I + A^{-1} B)^{-1} A^{-1}
$$

Simplify:

$$
(A + B)^{-1} = A^{-1} - A^{-1} B (A (I + A^{-1} B))^{-1}
$$

$$
(A + B)^{-1} = A^{-1} - A^{-1} B (A + B)^{-1}
$$

This completes the proof of the lemma and justifies its use in our derivation.


#### **Key Takeaway:**

Both SVMs and ridge regression share the property that their solutions lie in the span of the training data. For SVMs, this emerges naturally from the dual-primal connection, highlighting how support vectors define the decision boundary. In ridge regression, the span property arises through matrix algebra and the closed-form solution.

This unifying view provides a deeper understanding of how machine learning models leverage training data to construct solutions. Next, we’ll explore **how this property influences kernelized methods** and its implications for scalability and interpretability in machine learning.

---


#### **Reparameterizing Optimization Problems: Building on the Span Property**


In the previous section, we established that both SVM and ridge regression solutions lie in the **span of the training data**. This insight opens up a new avenue: we can **reparameterize the optimization problem** by restricting our search space to this span. Let’s explore how this simplifies the optimization process and why it’s particularly useful in high-dimensional settings.



##### **Reparameterization of Ridge Regression**

To recap, the ridge regression problem for regularization parameter $$\lambda > 0$$ is:

$$
w^* = \arg\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \{ w^T x_i - y_i \}^2 + \lambda \|w\|_2^2
$$

We know that $$w^* \in \text{span}(x_1, \dots, x_n) \subset \mathbb{R}^d$$. Therefore, instead of minimizing over all of $$\mathbb{R}^d$$, we can restrict our optimization to the span of the training data:

$$
w^* = \arg\min_{w \in \text{span}(x_1, \dots, x_n)} \frac{1}{n} \sum_{i=1}^n \{ w^T x_i - y_i \}^2 + \lambda \|w\|_2^2
$$

Now, let’s **reparameterize** the objective function. Since $$w \in \text{span}(x_1, \dots, x_n)$$, we can express $$w$$ as a linear combination of the inputs:

$$
w = X^T \alpha, \quad \alpha \in \mathbb{R}^n
$$

Substituting this into the optimization problem gives:

##### **Reparameterized Objective**

The original formulation becomes:

$$
\alpha^* = \arg\min_{\alpha \in \mathbb{R}^n} \frac{1}{n} \sum_{i=1}^n \{ (X^T \alpha)^T x_i - y_i \}^2 + \lambda \|X^T \alpha\|_2^2
$$

Once $$\alpha^*$$ is obtained, the optimal weight vector $$w^*$$ can be recovered as:

$$
w^* = X^T \alpha^*
$$


##### **Why Does This Matter?**


By reparameterizing, we’ve effectively reduced the dimension of the optimization problem:

- **Original Problem**: Optimize over $$\mathbb{R}^d$$ (where $$d$$ is the feature space dimension).
- **Reparameterized Problem**: Optimize over $$\mathbb{R}^n$$ (where $$n$$ is the number of training examples).

This reduction is significant in scenarios where $$d \gg n$$. For instance:

- **Very Large Feature Space**: Suppose $$d = 300 \, \text{million}$$ (e.g., using high-order polynomial interactions).
- **Moderate Training Set Size**: Suppose $$n = 300,000$$ examples.

In the original formulation, we solve a $$300 \, \text{million}$$-dimensional optimization problem. After reparameterization, we solve a much smaller $$300,000$$-dimensional problem. This simplification highlights why the span property is crucial, particularly when the number of features vastly exceeds the number of training examples.

---

#### **Generalization: The Representer Theorem**

The span property is not unique to SVM and ridge regression. A powerful result known as the **Representer Theorem** shows that this property applies broadly to all norm-regularized linear models.Here's how it works:

##### **Generalized Objective**
We start with a generalized objective for a norm-regularized model:

$$
w^* = \arg \min_{w \in \mathcal{H}} R(\|w\|) + L\big((\langle w, x_1 \rangle), \dots, (\langle w, x_n \rangle)\big).
$$

Here:
- $$ R(\|w\|) $$: Regularization term to control model complexity.
- $$ L $$: Loss function that measures the fit of the model to the data.
- $$ \mathcal{H} $$: Hypothesis space where $$ w $$ resides.

##### **Key Insight from the Representer Theorem**
The Representer Theorem tells us that instead of searching for $$ w^* $$ in the entire hypothesis space $$ \mathcal{H} $$, we can restrict our search to the span of the training data. Mathematically:

$$
w^* = \arg \min_{w \in \text{span}(x_1, \dots, x_n)} R(\|w\|) + L\big((\langle w, x_1 \rangle), \dots, (\langle w, x_n \rangle)\big).
$$

This dramatically reduces the complexity of the optimization problem.

##### **Reparameterization**
Using this insight, we can reparameterize the optimization problem as before. Let $$ w = \sum_{i=1}^n \alpha_i x_i $$, where $$ \alpha = (\alpha_1, \dots, \alpha_n) \in \mathbb{R}^n $$. Substituting this into the objective:

$$
\alpha^* = \arg \min_{\alpha \in \mathbb{R}^n} R\left(\left\| \sum_{i=1}^n \alpha_i x_i \right\|\right) + L\Big(\big\langle \sum_{i=1}^n \alpha_i x_i, x_1 \big\rangle, \dots, \big\langle \sum_{i=1}^n \alpha_i x_i, x_n \big\rangle\Big).
$$

##### **Why This Matters**
By reparameterizing the problem, we transform the optimization from a potentially infinite-dimensional space $$ \mathcal{H} $$ to a finite-dimensional space (spanned by the data points). This makes the problem computationally feasible and reveals why the solution lies in the span of the data.


#### **Implications: Kernelization and the Kernel Trick**

The Representer Theorem plays a pivotal role in enabling **kernelization**. Here's how it connects:

Using the Representer Theorem, we know that the solution $$ w^* $$ resides in the span of the data. This insight allows us to replace the feature space $$ \phi(x) $$ with a kernel function $$ K(x, x') $$, where:

$$
K(x, x') = \langle \phi(x), \phi(x') \rangle.
$$

The kernel function computes the inner product in the transformed feature space without explicitly constructing $$ \phi(x) $$. This process is called **kernelization**.

##### **Kernelized Representer Theorem**
The Representer Theorem in the context of kernels can be expressed as:

$$
w^* = \sum_{i=1}^n \alpha_i \phi(x_i),
$$

where the coefficients $$ \alpha $$ are obtained by solving an optimization problem that depends only on the kernel $$ K(x_i, x_j) $$.


The Representer Theorem provides a unifying framework for kernelization. By recognizing that solutions lie in the span of the data, we can seamlessly replace explicit feature mappings with kernel functions. This powerful insight underpins many modern machine learning techniques, making high-dimensional learning tasks computationally feasible.


---

##### **Summary**

1. **Reparameterization**: If a solution lies in the span of the training data, we can reparameterize the optimization problem to reduce its dimensionality, simplifying the computation.
2. **High-Dimensional Settings**: This approach is especially useful when $$d \gg n$$, where the feature space dimension far exceeds the number of training examples.
3. **Representer Theorem**: The span property generalizes to all norm-regularized linear models, forming the theoretical foundation for kernelization and advocates that linear models can be kernelized.
4. **Kernel Trick**: By kernelizing models, we can solve complex problems in high-dimensional spaces efficiently and without the need to represent $$\phi(x)$$ explicitly.

In the next post, we will explore specific applications of kernelized algorithms and how they can handle non-linear decision boundaries with remarkable ease.


Understanding the span property and its implications is not just a mathematical curiosity—it’s a foundational principle that unifies many machine learning models and opens up practical avenues for efficient computation in challenging scenarios.

Next, we’ll dive deeper into specific topics related to SVM, focusing on building intuitive understanding for each concept. Stay tuned!

##### **References**
- Representer Theorem
- Visualization elements