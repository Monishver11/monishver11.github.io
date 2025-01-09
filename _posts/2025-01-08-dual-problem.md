---
layout: post
title: The Dual Problem of SVM
date: 2025-01-07 20:00:00-0400
featured: false
description: To Add
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

In machine learning, optimization problems often present themselves as key challenges. For example, when training a **Support Vector Machine (SVM)**, we are tasked with finding the optimal hyperplane that separates two classes in a high-dimensional feature space. While we can solve this directly using methods like subgradient descent, we can also leverage a more analytical approach through **Quadratic Programming (QP)** solvers.

Moreover, for convex optimization problems, there is a powerful technique known as solving the **dual problem**. Understanding this duality is not only essential for theory, but it also offers computational advantages. In this blog, we’ll dive into the dual formulation of SVM and its implications.

---

#### **SVM as a Quadratic Program**

To understand the dual problem of SVM, let’s first revisit the primal optimization problem for SVMs. The goal of an SVM is to find the hyperplane that maximizes the margin between two classes. The optimization problem can be written as:

$$
\begin{aligned}
\min_{w, b, \xi} \quad & \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \\
\text{subject to} \quad & -\xi_i \leq 0 \quad \text{for } i = 1, \dots, n, \\
& 1 - y_i (w^T x_i + b) - \xi_i \leq 0 \quad \text{for } i = 1, \dots, n,
\end{aligned}
$$


**Primal**—sounds technical, right? Here’s what it means: The **primal problem** refers to the original formulation of the optimization problem in terms of the decision variables (in this case, $$w$$, $$b$$, and $$\xi$$) and the objective function. It is called "primal" because it directly represents the problem without transformation. The primal form of SVM is concerned with finding the optimal hyperplane parameters that minimize the classification error while balancing the margin size. 


##### **Breakdown of the Problem**

- **Objective Function**: The term $$\frac{1}{2} \|w\|^2$$ is a regularization term that aims to minimize the complexity of the model, ensuring that the hyperplane is as "wide" as possible. The second term, $$C \sum_{i=1}^n \xi_i$$, penalizes the violations of the margin (through slack variables $$\xi_i$$) and controls the trade-off between margin size and misclassification.

- **Constraints**: The constraints consist of two parts:
  - The first part ensures that slack variables are non-negative: $$-\xi_i \leq 0$$, meaning that each slack variable must be at least zero. **Why?** The slack variables represent the margin violations, and they must be non-negative because they quantify how much a data point violates the margin, which cannot be negative.
  - The second part enforces that the data points are correctly classified or their margin violations are captured by $$\xi_i$$. For each data point $$i$$, the condition $$1 - y_i (w^T x_i + b) - \xi_i \leq 0$$ must hold. Here, $$y_i$$ is the true label of the data point, and $$w^T x_i + b$$ represents the signed distance of the point from the hyperplane. If the point is correctly classified and lies outside the margin (i.e., its signed distance from the hyperplane is greater than 1), the constraint holds true. If the point is misclassified or falls inside the margin, the slack variable $$\xi_i$$ will account for this violation.


This problem has a **differentiable objective function**, **affine constraints**, and includes a number of **unknowns** that can be solved using Quadratic Programming (QP) solvers.

So, **Quadratic Programming (QP)** is an optimization problem where the objective function is quadratic (i.e., includes squared terms like $$\|w\|^2$$), and the constraints are linear. In the context of SVM, QP is utilized because the objective function involves the squared norm of $$w$$ (which is quadratic), and the constraints are linear inequalities.

The QP formulation for SVM involves minimizing a quadratic objective function (with respect to $$w$$, $$b$$, and $$\xi$$) subject to linear constraints. Now, while QP solvers provide an efficient way to tackle this problem, let’s explore the **dual problem(next)** to gain further insights.

But, why bother with the **dual problem**? Here’s why it’s worth the dive:

1. **Simpler Computation with Large Datasets**: The dual formulation focuses on Lagrange multipliers instead of the decision variables $$w$$ and $$b$$. This approach reduces the number of variables related to the feature space's dimensionality and instead scales with the number of data points. For large datasets with many points but relatively low-dimensional features, this leads to simpler and more efficient computation.
   
2. **Kernel Trick**: The dual problem is well-suited for the **kernel trick**, allowing us to compute inner products between data points in higher-dimensional spaces without explicitly mapping the data. This is particularly useful for non-linearly separable data, making kernel methods computationally feasible.

3. **Geometrical Insights**: The dual formulation emphasizes the relationship between support vectors and the margin, offering a clearer geometrical interpretation. It shows that only the support vectors (the points closest to the decision boundary) determine the optimal hyperplane.

4. **Convexity and Global Optimality**: The dual problem is convex, ensuring that solving it leads to the global optimal solution. This is particularly beneficial when the primal problem has a large number of variables and constraints.

In short, while QP solvers can efficiently solve the primal problem, the dual problem formulation offers computational benefits, the potential for kernel methods, and a clearer understanding of the SVM model’s properties. This makes the dual approach a powerful tool in SVM optimization.


No need to worry about the details above—we’ll cover them step by step. For now, keep reading!

---

#### **The Lagrangian**

To begin understanding the dual problem, we need to define the **Lagrangian** of the optimization problem. For general inequality-constrained optimization problems, the goal is:

$$
\begin{aligned}
\min_x \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \leq 0, \quad i = 1, \dots, m.
\end{aligned}
$$

The corresponding **Lagrangian** is defined as:

$$
L(x, \lambda) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x),
$$

where:

- $$\lambda_i$$ are the **Lagrange multipliers** (also known as **dual variables**).
- The Lagrangian function combines the objective function $$f_0(x)$$ with the constraints $$f_i(x)$$, weighted by the Lagrange multipliers $$\lambda_i$$. These multipliers represent how much the objective function will change if we relax or tighten the corresponding constraint.

The concept behind the Lagrangian is that it transforms **hard constraints** into **soft penalties** within the objective function. The Lagrange multipliers give us the flexibility to penalize the violation of constraints in a controlled manner.

[Need more idea and intuition on the above points. And why do we need to have it this way?]

#### **Lagrange Dual Function**

Next, we define the **Lagrange dual function**, which plays a crucial role in deriving the dual problem. The dual function is obtained by minimizing the Lagrangian with respect to the primal variables (denoted as $$x$$):

$$
g(\lambda) = \inf_x L(x, \lambda) = \inf_x \left[ f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) \right].
$$

[What do we mean by this?]

[The reason we're doing it is because, we need to induce the below properties somehow in our problem to help us solve the optimization we've at hand. For now, just trust this way and follow. It'll make sense. Understand, as in any language, we'll play with words to get to the results we want, in math we do the same thing, haha!]

##### **Key Properties of the Lagrange Dual Function:**

1. **Concavity**: The dual function $$g(\lambda)$$ is always **concave**. This is a significant property because concave functions are easier to maximize.
   
2. **Lower Bound**: The dual function provides a **lower bound** on the primal objective value. Specifically, if $$\lambda \geq 0$$, then:

   $$
   g(\lambda) \leq p^*,
   $$

   where $$p^*$$ is the optimal value of the primal problem. This property ensures that solving the dual problem will never overestimate the primal optimal value.

However, $$g(\lambda)$$ can be **uninformative** (e.g., $$-\infty$$) depending on the feasibility of the primal problem and the values of $$\lambda$$.

[How does this work? Why does it work?]
[With this in place, next we'll explain how are we fitting this into the picture?]

#### **The Primal and the Dual**

For any general primal optimization problem:

$$
\begin{aligned}
\min_x \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \leq 0, \quad i = 1, \dots, m,
\end{aligned}
$$

we can formulate the corresponding **dual problem** as:

$$
\begin{aligned}
\max_\lambda \quad & g(\lambda) \\
\text{subject to} \quad & \lambda_i \geq 0, \quad i = 1, \dots, m.
\end{aligned}
$$

The dual problem has some remarkable properties:

- **Convexity**: The dual problem is always a **convex optimization problem**, even if the primal problem is not convex.
- **Simplification**: In some cases, solving the dual problem is easier than solving the primal problem directly. This is particularly true when the primal problem is difficult to solve or the number of constraints is large.

[Give an analogy here, full story kinda thing from lagrangian, dual properties and the primal & dual formulation]

[Below, are the next two important properties derived from the above formulation]

#### **Weak and Strong Duality**

Let’s now look at two important duality concepts: **weak duality** and **strong duality**.

1. **Weak Duality**: This property tells us that the optimal value of the primal problem is always greater than or equal to the optimal value of the dual problem:

   $$
   p^* \geq d^*,
   $$

   where $$p^*$$ and $$d^*$$ are the optimal values of the primal and dual problems, respectively. This is a fundamental result in optimization theory. [Why?]

2. **Strong Duality**: In some special cases (such as when the problem satisfies **Slater’s condition**), **strong duality** holds, meaning the optimal values of the primal and dual problems are equal:

   $$
   p^* = d^*.
   $$

   Strong duality is particularly useful because it allows us to solve the dual problem instead of the primal one, often simplifying the problem or reducing computational complexity.

[Slater's conditions ?]

[To put simply, what can we say about this?]

#### **Complementary Slackness**

When **strong duality** holds, we can derive the **complementary slackness** condition. This condition provides deeper insight into the relationship between the primal and dual solutions. Specifically, if $$x^*$$ is the optimal primal solution and $$\lambda^*$$ is the optimal dual solution, we have:

$$
f_0(x^*) = g(\lambda^*) = \inf_x L(x, \lambda^*) \leq L(x^*, \lambda^*) = f_0(x^*) + \sum_{i=1}^m \lambda_i^* f_i(x^*).
$$

For this equality to hold, the term $$\sum_{i=1}^m \lambda_i^* f_i(x^*)$$ must be zero for each constraint. This leads to the following **complementary slackness condition**:

1. If $$\lambda_i^* > 0$$, then $$f_i(x^*) = 0$$, meaning that the corresponding constraint is **active** at the optimal point.
2. If $$f_i(x^*) < 0$$, then $$\lambda_i^* = 0$$, meaning that the corresponding constraint is **inactive**.

This condition tells us which constraints are binding (active) at the optimal solution and which are not, providing critical information about the structure of the optimal solution.

[More details, not very easy to make sense with the given points]

[Intuition] 

[To put simply, what can we say about this?]

---

##### **Conclusion**

By exploring the **dual problem** of SVM, we gain both theoretical insights and practical benefits. The dual formulation provides a new perspective on the original optimization problem, and solving it can sometimes be more efficient or insightful. The duality between the primal and dual problems underpins many of the optimization techniques used in machine learning, particularly in the context of support vector machines.

Understanding duality, the Lagrangian, weak and strong duality, and complementary slackness is crucial for anyone working with SVMs and optimization problems in general. Next, ..

##### **References**
- 