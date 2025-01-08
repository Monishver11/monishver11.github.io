---
layout: post
title: Subgradient and Subgradient Descent
date: 2025-01-07 20:00:00-0400
featured: false
description: An deep dive into subgradients, subgradient descent, and their application in optimizing non-differentiable functions like SVMs.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---


Optimization is a cornerstone of machine learning, as it allows us to fine-tune models and minimize error. For smooth, differentiable functions, gradient descent is the go-to method. However, in the real world, many functions are not differentiable. This is where **subgradients** come into play. In this post, we’ll explore subgradients, understand their properties, and see how they are used in subgradient descent. Finally, we’ll dive into their application in support vector machines (SVMs).

---

#### **First-Order Condition for Convex, Differentiable Functions**

Let’s start with the basics of convex functions. For a function $$ f : \mathbb{R}^d \to \mathbb{R} $$ that is convex and differentiable, the **first-order condition** states that:

$$
f(y) \geq f(x) + \nabla f(x)^\top (y - x), \quad \forall x, y \in \mathbb{R}^d
$$

This equation tells us something profound: the linear approximation to $$ f $$ at $$ x $$ is a global underestimator of the function. In other words, the gradient provides a plane below the function, ensuring the convexity property holds. A direct implication is that if $$ \nabla f(x) = 0 $$, then $$ x $$ is a global minimizer of $$ f $$. This serves as the foundation for gradient-based optimization.

---

#### **Subgradients: A Generalization of Gradients**

While gradients work well for differentiable functions, what happens when a function has kinks or sharp corners? This is where **subgradients** step in. A subgradient is a generalization of the gradient, defined as follows:

A vector $$ g \in \mathbb{R}^d $$ is a subgradient of a convex function $$ f : \mathbb{R}^d \to \mathbb{R} $$ at $$ x $$ if:

$$
f(z) \geq f(x) + g^\top (z - x), \quad \forall z \in \mathbb{R}^d
$$

Subgradients essentially maintain the same idea as gradients: they provide a linear function that underestimates $$ f $$. However, while the gradient is unique, a subgradient can belong to a set of possible vectors called the **subdifferential**, denoted $$ \partial f(x) $$.

For convex functions, the subdifferential is always non-empty. If $$ f $$ is differentiable at $$ x $$, the subdifferential collapses to a single point: $$ \{ \nabla f(x) \} $$. Importantly, for a convex function, a point $$ x $$ is a global minimizer if and only if $$ 0 \in \partial f(x) $$. This property allows us to apply subgradients even when gradients don’t exist.


##### **Subgradient Example: Absolute Value Function**

Let’s consider a classic example: $$f(x) = \lvert x \rvert $$. This function is non-differentiable at $$x = 0$$, making it an ideal candidate to illustrate subgradients. The subdifferential of $$f(x)$$ is:

$$
\partial f(x) =
\begin{cases} 
\{-1\} & \text{if } x < 0, \\
\{1\} & \text{if } x > 0, \\
[-1, 1] & \text{if } x = 0.
\end{cases}
$$

At $$ x = 0 $$, the subgradient set contains all values in the interval $$[-1, 1]$$, which corresponds to all possible slopes of lines that underapproximate $$ f(x) $$ at that point. This example highlights the flexibility of subgradients in handling non-differentiable points.


#### **Subgradient Descent: The Optimization Method**

Subgradient descent extends gradient descent to non-differentiable convex functions. The update rule is simple:

$$
x_{t+1} = x_t - \eta_t g,
$$

where $$ g \in \partial f(x_t) $$ is a subgradient, and $$ \eta_t > 0 $$ is the step size. Unlike gradients, subgradients do not always converge to zero as the algorithm progresses. This means we must carefully choose the step size $$ \eta_t $$ to ensure convergence.

For convex functions, subgradient descent ensures that the iterates move closer to the minimizer:

$$
\|x_{t+1} - x^\ast\| < \|x_t - x^\ast\|,
$$

provided that the step size is small enough. However, subgradient methods tend to be slower than gradient descent because they rely on weaker information about the function’s structure.

---

#### **Subgradient Descent for SVMs: The Pegasos Algorithm**

Subgradients are particularly useful in optimizing the objective function of support vector machines (SVMs). The SVM objective is:

$$
J(w) = \frac{1}{n} \sum_{i=1}^n \max(0, 1 - y_i w^\top x_i) + \frac{\lambda}{2} \|w\|^2,
$$

where the first term represents the hinge loss, and the second term penalizes large weights to prevent overfitting. Optimizing this objective using gradient-based methods is tricky due to the non-differentiability of the hinge loss. Enter **Pegasos**, a stochastic subgradient descent algorithm.

##### **The Pegasos Algorithm**

The Pegasos algorithm follows these steps:

1. Initialize $$ w_1 = 0 $$ and $$ t = 0 $$.
2. Randomly permute the data at the beginning of each epoch.
3. For each data point $$ (x_j, y_j) $$, update the parameters:
   - Increment $$ t $$: $$ t = t + 1 $$.
   - Compute the step size: $$ \eta_t = \frac{1}{t \lambda} $$.
   - If $$ y_j w_t^\top x_j < 1 $$, update:
     $$
     w_{t+1} = (1 - \eta_t \lambda) w_t + \eta_t y_j x_j.
     $$
   - Otherwise, update:
     $$
     w_{t+1} = (1 - \eta_t \lambda) w_t.
     $$

This algorithm cleverly combines subgradient updates with a decreasing step size to ensure convergence. The hinge loss ensures that the model maintains a margin of separation, while the regularization term prevents overfitting. By processing one data point at a time, Pegasos achieves efficiency and scalability.

---

##### **Wrapping Up**

Subgradients are a powerful tool for dealing with non-differentiable convex functions, and subgradient descent provides a straightforward yet effective way to optimize such functions. While slower than gradient descent, subgradient descent shines in scenarios where gradients are unavailable. The Pegasos algorithm demonstrates how subgradient descent can be applied to large-scale SVM problems, balancing efficiency and performance.

In the next part of this series, we’ll delve into the **dual problem** and uncover its connection to the primal SVM formulation. Stay tuned for more insights into the fascinating world of optimization!
