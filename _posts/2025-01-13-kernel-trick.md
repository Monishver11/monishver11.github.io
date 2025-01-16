---
layout: post
title: Understanding the Kernel Trick
date: 2025-01-13 19:03:00-0400
featured: false
description: A step-by-step exploration of kernel methods, unraveling their role in enabling powerful nonlinear modeling through the elegance of the kernel trick.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---


When working with machine learning models, especially Support Vector Machines (SVMs), the idea of mapping data into a higher-dimensional space often comes into play. This mapping helps transform non-linearly separable data into a space where linear decision boundaries can be applied. But what happens when the dimensionality of the feature space becomes overwhelmingly large? This is where the **kernel trick** saves the day. In this post, we will explore the kernel trick, starting with SVMs, their reliance on feature mappings, and how inner products in feature space can be computed without ever explicitly constructing that space.

---

#### **SVMs with Explicit Feature Maps**

To understand the kernel trick, let’s begin with SVMs. In the simplest case, an SVM aims to find a hyperplane that separates data into classes with the largest possible margin. To handle more complex data, we map the input data $$ \mathbf{x} $$ into a higher-dimensional feature space using a feature map $$ \psi: X \to \mathbb{R}^d $$. In this space, the SVM optimization problem can be written as:

$$
\min_{\mathbf{w} \in \mathbb{R}^d} \frac{1}{2} \|\mathbf{w}\|^2 + \frac{c}{n} \sum_{i=1}^n \max(0, 1 - y_i \mathbf{w}^T \psi(\mathbf{x}_i)).
$$

Here, $$ \mathbf{w} $$ is the weight vector, $$ c $$ is a regularization parameter, and $$ y_i $$ are the labels of the data points. While this approach works well for small $$ d $$, it becomes computationally expensive as $$ d $$ increases, especially when using high-degree polynomial mappings. 

To address this issue, we turn to a reformulation of the SVM problem, derived from **Lagrangian duality**.


#### **The SVM Dual Problem**

Through Lagrangian duality, the SVM optimization problem can be re-expressed as a dual problem:

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \psi(\mathbf{x}_j)^T \psi(\mathbf{x}_i),
$$

subject to:

$$
\sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \in \left[ 0, \frac{c}{n} \right] \quad \forall i.
$$

Here, $$ \alpha_i $$ are the dual variables (Lagrange multipliers). Once the optimal $$ \boldsymbol{\alpha}^* $$ is obtained, the weight vector in the feature space can be reconstructed as:

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i^* y_i \psi(\mathbf{x}_i).
$$

The decision function for a new input $$ \mathbf{x} $$ is given by:

$$
\hat{f}(\mathbf{x}) = \sum_{i=1}^n \alpha_i^* y_i \psi(\mathbf{x}_i)^T \psi(\mathbf{x}).
$$

##### **Observing the Role of Inner Products**

An important observation here is that the feature map $$ \psi(\mathbf{x}) $$ appears only through inner products of the form $$ \psi(\mathbf{x}_j)^T \psi(\mathbf{x}_i) $$. This means we don’t actually need the explicit feature representation $$ \psi(\mathbf{x}) $$; instead, we just need the ability to compute these inner products efficiently.


#### **Computing Inner Products in Practice**

Let’s explore the kernel trick with an example.

##### **Example: Degree-2 Monomials**

Suppose we are working with 2D data points $$ \mathbf{x} = (x_1, x_2) $$. If we map the data into a space of degree-2 monomials, the feature map becomes:

$$
\psi: \mathbb{R}^2 \to \mathbb{R}^6, \quad (x_1, x_2) \mapsto (1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}x_1x_2, x_2^2).
$$

The inner product in the feature space is:

$$
\psi(\mathbf{x})^T \psi(\mathbf{x}') = 1 + 2x_1x_1' + 2x_2x_2' + (x_1x_1')^2 + 2x_1x_2x_1'x_2' + (x_2x_2')^2.
$$

Simplifying, we observe:

$$
\psi(\mathbf{x})^T \psi(\mathbf{x}') = (1 + x_1x_1' + x_2x_2')^2 = (1 + \mathbf{x}^T \mathbf{x}')^2.
$$

This shows that we can compute $$ \psi(\mathbf{x})^T \psi(\mathbf{x}') $$ directly from the original input space without explicitly constructing $$ \psi(\mathbf{x}) $$—a key insight behind the kernel trick.

##### **General Case: Monomials Up to Degree $$p$$**

For feature maps that produce monomials up to degree $$p$$, the inner product generalizes as:

$$
\psi(x)^T \psi(x') = (1 + x^T x')^p.
$$

It is worth noting that the coefficients of the monomials in $$\psi(x)$$ may vary depending on the specific feature map.


#### **Efficiency of the Kernel Trick: From Exponential to Linear Complexity**

One of the key advantages of the kernel trick is its ability to reduce the computational complexity of working with high-dimensional feature spaces. Let's break this down:

##### **Explicit Computation Complexity**

When we map an input vector $$\mathbf{x} \in \mathbb{R}^d$$ to a feature space with monomials up to degree $$p$$, the dimensionality of the feature space increases significantly. Specifically:

- **Feature Space Dimension**: The number of features in the expansion is:

  $$
  \binom{d + p}{p} = \frac{(d + p)!}{d! \, p!}.
  $$

  For large $$p$$ or $$d$$, this grows rapidly and can quickly become computationally prohibitive.

- **Explicit Inner Product**: Computing the inner product directly in this expanded space has a complexity of:

  $$
  O\left(\binom{d + p}{p}\right),
  $$

  which is exponential in $$p$$ for fixed $$d$$.

##### **Implicit Computation Complexity**

Using the kernel trick, we avoid explicitly constructing the feature space. For a kernel function like:

$$
k(\mathbf{x}, \mathbf{x}') = (1 + \mathbf{x}^T \mathbf{x}')^p,
$$

the computation operates directly in the input space. 

- **Input Space Computation**: Computing the kernel function involves:

  1. **Dot Product**: $$\mathbf{x}^T \mathbf{x}'$$ is computed in $$O(d)$$.
  2. **Polynomial Evaluation**: Raising this result to power $$p$$ is done in constant time, independent of $$d$$.

Thus, the complexity is reduced to:

$$
O(d),
$$

which is **linear** in the input dimensionality $$d$$, regardless of $$p$$.


##### **Why This Matters**

- **Explicit Features**: For high $$p$$, the feature space grows exponentially, leading to a **curse of dimensionality** if explicit computation is used.
- **Implicit Kernel Computation**: The kernel trick sidesteps the explicit feature space, allowing efficient computation even when the feature space is high-dimensional or infinite (e.g., with RBF kernels).

This transformation from **exponential** to **linear complexity** is one of the core reasons kernel methods are powerful tools in machine learning.


**Key Takeaway** : The kernel trick enables efficient computation in high-dimensional feature spaces by directly working in the input space. This reduces the complexity from $$O\left(\binom{d + p}{p}\right)$$ to $$O(d)$$, making it feasible to apply machine learning methods to problems with high-degree polynomial or infinite-dimensional feature spaces.


---

#### **Exploring the Kernel Function**

To fully appreciate the kernel trick, we need to formalize the concept of the **kernel function**. In our earlier discussion, we introduced the idea of a feature map $$ \psi: X \to \mathcal{H} $$, which maps input data from the original space $$ X $$ to a higher-dimensional feature space $$ \mathcal{H} $$. The kernel function $$ k $$ corresponding to this feature map is defined as:

$$
k(\mathbf{x}, \mathbf{x}') = \langle \psi(\mathbf{x}), \psi(\mathbf{x}') \rangle,
$$

where $$ \langle \cdot, \cdot \rangle $$ represents the inner product in $$ \mathcal{H} $$. 


##### **Why Use Kernel Functions?**

At first glance, this notation might seem like a trivial restatement of the inner product, but it’s far more powerful. The key insight is that we can often evaluate $$ k(\mathbf{x}, \mathbf{x}') $$ directly, without explicitly computing $$ \psi(\mathbf{x}) $$ and $$ \psi(\mathbf{x}') $$. This is crucial for efficiently working with high-dimensional or infinite-dimensional feature spaces. But this efficiency only applies to certain methods — those that can be **kernelized**.


#### **Kernelized Methods**

A method is said to be **kernelized** if it uses the feature vectors $$ \psi(\mathbf{x}) $$ only inside inner products of the form $$ \langle \psi(\mathbf{x}), \psi(\mathbf{x}') \rangle $$. For such methods, we can replace these inner products with a kernel function $$ k(\mathbf{x}, \mathbf{x}') $$, avoiding explicit feature computation.  This applies to both the optimization problem and the prediction function. Let’s revisit the SVM example to see kernelization in action.

##### **Kernelized SVM Dual Formulation**

Recall the dual problem for SVMs:

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \langle \psi(\mathbf{x}_i), \psi(\mathbf{x}_j) \rangle,
$$

subject to:

$$
\sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \in \left[ 0, \frac{c}{n} \right] \quad \forall i.
$$

**Here’s the key**: because every occurrence of $$ \psi(\mathbf{x}) $$ is inside an inner product, we can replace $$ \langle \psi(\mathbf{x}_i), \psi(\mathbf{x}_j) \rangle $$ with $$ k(\mathbf{x}_i, \mathbf{x}_j) $$, the kernel function. The resulting dual optimization problem becomes:

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j k(\mathbf{x}_i, \mathbf{x}_j),
$$

subject to the same constraints.

For predictions, the decision function can also be written in terms of the kernel:

$$
\hat{f}(\mathbf{x}) = \sum_{i=1}^n \alpha_i^* y_i k(\mathbf{x}_i, \mathbf{x})
= \sum_{i=1}^n \alpha_i^* y_i \psi(\mathbf{x}_i)^T \psi(\mathbf{x}) 
$$

This reformulation is what allows SVMs to operate efficiently in high-dimensional spaces.


##### **The Kernel Matrix**

A key component in kernelized methods is the **kernel matrix**, which encapsulates the pairwise kernel values for all data points. For a dataset $$ \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\} $$, the kernel matrix $$ \mathbf{K} $$ is defined as:

$$
\mathbf{K} = \begin{bmatrix}
k(\mathbf{x}_1, \mathbf{x}_1) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\
\vdots & \ddots & \vdots \\
k(\mathbf{x}_n, \mathbf{x}_1) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n)
\end{bmatrix}.
$$

This $$ n \times n $$ matrix, also known as the **Gram matrix** in machine learning, summarizes all the information about the training data necessary for solving the kernelized optimization problem.

For the kernelized SVM, we can replace $$ \langle \psi(\mathbf{x}_i), \psi(\mathbf{x}_j) \rangle $$ with $$ K_{ij} $$, reducing the dual problem to:

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K_{ij},
$$

subject to the same constraints.



**So, Given a kernelized ML algorithm** (i.e., all $$ \psi(x) $$'s show up as $$ \langle \psi(x), \psi(x') \rangle $$) :

1. **Flexibility**: By substituting the kernel function, we can implicitly use very high-dimensional or even infinite-dimensional feature spaces.
2. **Scalability**: Once the kernel matrix is computed, the computational cost depends on the number of data points $$ n $$, rather than the dimension of the feature space $$ d $$.
3. **Efficiency**: For many kernels, $$ k(\mathbf{x}, \mathbf{x}') $$ can be computed without directly accessing the high-dimensional feature representation $$ \psi(\mathbf{x}) $$, avoiding the $$ O(d) $$ dependence.

These properties make kernelized methods invaluable when $$ d \gg n $$, a common scenario in machine learning tasks.

The kernel trick revolutionizes how we think about high-dimensional data. Next, we will delve into popular kernel functions, their interpretations, and how to choose the right one for your problem.

---

#### **Example Kernels**

In many cases, it's useful to think of the kernel function $$ k(x, x') $$ as a **similarity score** between the data points $$ x $$ and $$ x' $$. This perspective allows us to design similarity functions without explicitly considering the feature map.

For example, we can create **string kernels** or **graph kernels**—functions that define similarity based on the structure of strings or graphs, respectively. The key question, however, is: **How do we know that our kernel functions truly correspond to inner products in some feature space?**

This is an essential consideration, as it ensures that the kernel method preserves the properties necessary for various machine learning algorithms to work effectively. Let’s break this down.


##### **How to Obtain Kernels?**

There are two primary ways to define kernels:

1. **Explicit Construction**: Define the feature map $$ \psi(\mathbf{x}) $$ and use it to compute the kernel:
   $$
   k(\mathbf{x}, \mathbf{x}') = \langle \psi(\mathbf{x}), \psi(\mathbf{x}') \rangle.
   $$ (e.g. monomials)

2. **Direct Definition**: Directly define the kernel $$ k(\mathbf{x}, \mathbf{x}') $$ as a similarity score and verify that it corresponds to an inner product for some $$ \psi $$. This verification is often guided by mathematical theorems. 
   
To understand this better, let's first equip ourselves with some essential linear algebra concepts.


##### **Positive Semidefinite Matrices and Kernels**

To verify if a kernel corresponds to a valid inner product, we rely on the concept of **positive semidefinite (PSD) matrices**. Here’s a quick refresher:

- A matrix $$ \mathbf{M} \in \mathbb{R}^{n \times n} $$ is positive semidefinite if:
  $$
  \mathbf{x}^\top \mathbf{M} \mathbf{x} \geq 0, \quad \forall \mathbf{x} \in \mathbb{R}^n.
  $$

- Equivalent conditions, each necessary and sufficient for a symmetric matrixfor $$ \mathbf{M} $$ being **PSD**:
  - $$ \mathbf{M} = \mathbf{R}^\top \mathbf{R} $$, for some matrix $$ \mathbf{R} $$.
  - All eigenvalues of $$ \mathbf{M} $$ are non-negative or $$\geq 0$$.

Next, we define a **positive definite (PD) kernel**:


##### **Positive Definite Kernel**

**Definition:**

A symmetric function $$ k: X \times X \to \mathbb{R} $$ is a **PD** kernel if, for any finite set $$ \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\} \subset X $$, the kernel matrix:

$$
\mathbf{K} = \begin{bmatrix}
k(\mathbf{x}_1, \mathbf{x}_1) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\
\vdots & \ddots & \vdots \\
k(\mathbf{x}_n, \mathbf{x}_1) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n)
\end{bmatrix}
$$

is positive semidefinite. 

1. Symmetry: $$ k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}', \mathbf{x}) $$.
2. The kernel matrix needs to be positive semidefinite for any finite set of points. 
3. Equivalently: $$ \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j k(\mathbf{x}_i, \mathbf{x}_j) \geq 0 $$, for all $$ \alpha_i \in \mathbb{R}$$ $$  \forall i$$.

[How, better way of stating it!]

##### **Mercer’s Theorem**

Mercer’s Theorem provides a foundational result for kernels. It states:

- A symmetric function $$ k(\mathbf{x}, \mathbf{x}') $$ can be expressed as an inner product 
  
  $$
   k(\mathbf{x}, \mathbf{x}') = \langle \psi(\mathbf{x}), \psi(\mathbf{x}') \rangle 
  $$
  
   if and only if $$ k(\mathbf{x}, \mathbf{x}')$$ is positive definite.

While proving that a kernel is **PD** can be challenging, we can use known kernels to construct new ones.


##### **Constructing New Kernels from Existing Ones**

Given valid PD kernels $$ k_1 $$ and $$ k_2 $$, we can create new kernels using the following operations:

1. **Non-Negative Scaling**: $$ k_{\text{new}}(\mathbf{x}, \mathbf{x}') = \alpha k(\mathbf{x}, \mathbf{x}') $$, where $$ \alpha \geq 0 $$.
2. **Addition**: $$ k_{\text{new}}(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') + k_2(\mathbf{x}, \mathbf{x}') $$.
3. **Multiplication**: $$ k_{\text{new}}(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') k_2(\mathbf{x}, \mathbf{x}') $$.
4. **Recursion**: $$ k_{\text{new}}(\mathbf{x}, \mathbf{x}') = k(\psi(\mathbf{x}), \psi(\mathbf{x}')) $$, for any function $$ \psi(\cdot) $$.
5. **Feature Mapping**: $$ k_{\text{new}}(\mathbf{x}, \mathbf{x}') = f(\mathbf{x}) f(\mathbf{x}') $$, for any function $$ f(\cdot) $$.

And, Lots more theorems to help you construct new kernels from old.

[Add reference to mercer theorem]

---

##### **Popular Kernel Functions**

###### **Linear Kernel**

The simplest kernel, corresponding to the standard dot product:

- **Input Space**: $$ X = \mathbb{R}^d $$
- **Kernel Function**: $$ k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}' $$


###### **Polynomial Kernel**

Generalizes the linear kernel by including higher-degree interactions:

- **Kernel Function**: $$ k(\mathbf{x}, \mathbf{x}') = (1 + \mathbf{x}^\top \mathbf{x}')^M $$, where $$ M $$ is the degree.

This kernel maps data to a feature space containing monomials up to degree $$ M $$, but the computational cost of explicit computation grows with $$ M $$.


###### **Quadratic Kernel**

A specific case of the polynomial kernel with $$ M = 2 $$:

- **Feature Map**: Includes all individual terms and pairwise products: 
  $$
  \psi(\mathbf{x}) = (x_1, \dots, x_d, x_1^2, \dots, x_d^2, \sqrt{2}x_1x_2, \dots, \sqrt{2}x_{d-1}x_d)^\top.
  $$

- **Kernel Function**: $$ k(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^\top \mathbf{x}') + (\mathbf{x}^\top \mathbf{x}')^2 $$.


###### **Radial Basis Function (RBF) / Gaussian Kernel**

Perhaps the most commonly used nonlinear kernel:

- **Kernel Function**:
  $$
  k(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\sigma^2}\right),
  $$
  where $$ \sigma^2 $$ is the bandwidth parameter.

The RBF kernel corresponds to an infinite-dimensional feature space and acts as a sophisticated similarity measure.

---

[1]

#### **Popular Kernel Functions**


##### **The Linear Kernel**

The linear kernel is the simplest and most intuitive kernel function. Imagine working with data in an input space represented as $$X = \mathbb{R}^d$$. Here, the feature space, denoted as $$\mathcal{H}$$, is the same as the input space $$\mathbb{R}^d$$. The feature map for this kernel is straightforward: $$\psi(x) = x$$. 

The kernel function itself is defined as:

$$
k(x, x') = \langle x, x' \rangle = x^\top x',
$$

where $$\langle x, x' \rangle$$ represents the standard inner product. This simplicity makes the linear kernel computationally efficient and ideal for linear models.


##### **The Quadratic Kernel**

The quadratic kernel takes us a step further by mapping the input space $$X = \mathbb{R}^d$$ into a higher-dimensional feature space $$\mathcal{H} = \mathbb{R}^D$$, where $$D$$ is approximately $$d + \binom d2 \approx \frac{d^2}{2}$$. This expanded feature space enables the kernel to capture quadratic relationships in the data.

The feature map for the quadratic kernel is given by:

$$
\psi(x) = \left(x_1, \dots, x_d, x_1^2, \dots, x_d^2, \sqrt{2}x_1x_2, \dots, \sqrt{2}x_ix_j, \dots, \sqrt{2}x_{d-1}x_d\right)^\top.
$$

To compute the kernel function, we use the inner product of the feature maps:

$$
k(x, x') = \langle \psi(x), \psi(x') \rangle.
$$

Expanding this yields:

$$
k(x, x') = \langle x, x' \rangle + \langle x, x' \rangle^2.
$$


**Derivation of the Quadratic Kernel:**

The quadratic kernel is defined as the inner product in a higher-dimensional feature space. The feature map $$ \psi(x) $$ includes:

1. Original features: $$ x_1, x_2, \dots, x_d $$
2. Squared features: $$ x_1^2, x_2^2, \dots, x_d^2 $$
3. Cross-product terms: $$ \sqrt{2}x_i x_j $$ for $$ i \neq j $$

Thus:

$$
\psi(x) = \left(x_1, x_2, \dots, x_d, x_1^2, x_2^2, \dots, x_d^2, \sqrt{2}x_1x_2, \sqrt{2}x_1x_3, \dots, \sqrt{2}x_{d-1}x_d\right)^\top
$$

The kernel is computed as:

$$
k(x, x') = \langle \psi(x), \psi(x') \rangle
$$

Expanding this, we have:

1. **Linear terms**:  
   
   $$
   \langle x, x' \rangle = \sum_{i} x_i x_i'
   $$

2. **Squared terms**:  
   
   $$
   \sum_{i} x_i^2 x_i'^2
   $$

3. **Cross-product terms**:
     
   $$
   2 \sum_{i \neq j} x_i x_j x_i' x_j'
   $$

Combining these, the kernel becomes:

$$
k(x, x') = \langle x, x' \rangle + \sum_{i} x_i^2 x_i'^2 + 2 \sum_{i \neq j} x_i x_j x_i' x_j'
$$

Recognizing that:

$$
\langle x, x' \rangle^2 = \left( \sum_{i} x_i x_i' \right)^2 = \sum_{i} x_i^2 x_i'^2 + 2 \sum_{i \neq j} x_i x_j x_i' x_j'
$$

The kernel simplifies to:

$$
k(x, x') = \langle x, x' \rangle + \langle x, x' \rangle^2
$$


One of the key advantages of kernel methods is computational efficiency. While the explicit computation of the inner product in the feature space requires $$O(d^2)$$ operations, the implicit kernel calculation only requires $$O(d)$$ operations.

A good example will make it much clearer.

Let $$ x = [1, 2] $$ and $$ x' = [3, 4] $$.

The quadratic kernel is defined as:

$$
k(x, x') = \langle x, x' \rangle + \langle x, x' \rangle^2
$$

**Step 1**: Compute $$ \langle x, x' \rangle $$
$$
\langle x, x' \rangle = (1)(3) + (2)(4) = 3 + 8 = 11
$$

**Step 2**: Compute $$ \langle x, x' \rangle^2 $$
$$
\langle x, x' \rangle^2 = 11^2 = 121
$$

**Step 3**: Compute $$ k(x, x') $$
$$
k(x, x') = \langle x, x' \rangle + \langle x, x' \rangle^2 = 11 + 121 = 132
$$

**Step 4**: Verify with the Feature Map

The feature map for the quadratic kernel is:

$$
\psi(x) = [x_1, x_2, x_1^2, x_2^2, \sqrt{2}x_1x_2]
$$

For $$ x = [1, 2] $$:
$$
\psi(x) = [1, 2, 1^2, 2^2, \sqrt{2}(1)(2)] = [1, 2, 1, 4, 2\sqrt{2}]
$$

For $$ x' = [3, 4] $$:
$$
\psi(x') = [3, 4, 3^2, 4^2, \sqrt{2}(3)(4)] = [3, 4, 9, 16, 12\sqrt{2}]
$$

Compute the inner product:

$$
\langle \psi(x), \psi(x') \rangle = (1)(3) + (2)(4) + (1)(9) + (4)(16) + (2\sqrt{2})(12\sqrt{2})
$$
$$
= 3 + 8 + 9 + 64 + 48 = 132
$$

Thus, the quadratic kernel gives:
$$
k(x, x') = 132
$$


##### **The Polynomial Kernel**

Building on the quadratic kernel, the polynomial kernel generalizes the concept by introducing a degree parameter $$M$$. The kernel function is defined as:

$$
k(x, x') = (1 + \langle x, x' \rangle)^M.
$$

This kernel corresponds to a feature space that includes all monomials of the input features up to degree $$M$$. Notably, the computational cost of evaluating the kernel function remains constant, regardless of $$M$$. However, explicitly computing the inner product in the feature space grows rapidly as $$M$$ increases.


##### **The Radial Basis Function (RBF) or Gaussian Kernel**

The RBF kernel, also known as the Gaussian kernel, is one of the most widely used kernels for nonlinear problems. The input space remains $$X = \mathbb{R}^d$$, but the feature space is infinite-dimensional, making it capable of capturing complex relationships in the data. 

The kernel function is expressed as:

$$
k(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right),
$$

where $$\sigma^2$$ is a parameter known as the bandwidth, controlling the smoothness of the kernel.

One might wonder if this kernel still adheres to the principle of inner products in a feature space. The answer is both yes and no. While it acts like a similarity score, it corresponds to the inner product of feature vectors in an infinite-dimensional space.

[Explain it intuitively, and how to make sense of it]

[some visualization needed]

---

#### **Kernelization: The Recipe for Success**

To effectively leverage kernel methods, follow this general recipe:

1. Recognize problems that can benefit from kernelization. These are cases where the feature map $$\psi(x)$$ only appears in inner products $$\langle \psi(x), \psi(x') \rangle$$.
2. Select an appropriate kernel function('similarity score') that suits the data and the task at hand.
3. Compute the kernel matrix, a symmetric matrix of size $$n \times n$$ for a dataset with $$n$$ data points.
4. Use the kernel matrix to optimize the model and make predictions.

This approach allows us to solve problems in high-dimensional feature spaces without the computational burden of explicit mappings.

---

##### **What’s Next?**

 We explored the theoretical foundations of kernel functions, how to construct valid kernels, and the properties of popular kernels. But, a key question remains: under what conditions can we apply kernelization effectively? Understanding this involves diving deeper into the properties of kernel functions and their applicability to different problem domains. In the next post, we will explore these conditions in detail and discuss their implications for solving SVM problems.

Stay tuned!

##### **References**
- Add some visualization for kernels intuition
- 
