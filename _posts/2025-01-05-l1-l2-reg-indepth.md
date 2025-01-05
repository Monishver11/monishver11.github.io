---
layout: post
title: L1 and L2 Regularization - Nuanced Details
date: 2025-01-05 13:50:00-0400
featured: false
description: A detailed explanation of L1 and L2 regularization, focusing on their theoretical insights, geometric interpretations, and practical implications for machine learning models.
tags: ML
categories: sample-posts
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

Regularization is a cornerstone in machine learning, providing a mechanism to prevent overfitting while controlling model complexity. Among the most popular techniques are **L1** and **L2 regularization**, which serve different purposes but share a common goal of improving model generalization. In this post, we will delve deep into the theory, mathematics, and practical implications of these regularization methods.

Let’s set the stage with linear regression. For a dataset 

$$D_n = \{(x_1, y_1), \dots, (x_n, y_n)\},$$ 

the objective in ordinary least squares is to minimize the mean squared error:

$$ 
\hat{w} = \arg\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \left( w^\top x_i - y_i \right)^2. 
$$

While effective, this approach can overfit when the number of features $$d$$ is large compared to the number of samples $$n$$. For example, in natural language processing, it is common to have millions of features but only thousands of documents.

##### **Addressing Overfitting with Regularization**

To mitigate overfitting, **$$L_2$$ regularization** (also known as **ridge regression**) adds a penalty term proportional to the $$L_2$$ norm of the weights:

$$ 
\hat{w} = \arg\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \left( w^\top x_i - y_i \right)^2 + \lambda \|w\|_2^2, 
$$

where:

$$ 
\|w\|_2^2 = w_1^2 + w_2^2 + \dots + w_d^2.
$$

This penalty term discourages large weight values, effectively shrinking them toward zero. When $$\lambda = 0$$, the solution reduces to ordinary least squares. As $$\lambda$$ increases, the penalty grows, favoring simpler models with smaller weights.

##### **Understanding $$L_2$$ Regularization**

L2 regularization is particularly effective at reducing sensitivity to fluctuations in the input data. To understand this, consider a simple linear function:

$$
\hat{f}(x) = \hat{w}^\top x.
$$

The function $$\hat{f}(x)$$ is said to be **Lipschitz continuous**, with a Lipschitz constant defined as:

$$
L = \|\hat{w}\|_2.
$$

This implies that when the input changes from $$x$$ to $$x + h$$, the function's output change is bounded by $$L\|h\|_2$$. In simpler terms, $$L_2$$ regularization controls the rate of change of $$\hat{f}(x)$$, making the model less sensitive to variations in the input data.

##### **Mathematical Proof of Lipschitz Continuity**

To formalize this property, let’s derive the Lipschitz bound:

$$
|\hat{f}(x + h) - \hat{f}(x)| = |\hat{w}^\top (x + h) - \hat{w}^\top x| = |\hat{w}^\top h|.
$$

Using the **Cauchy-Schwarz inequality**, this can be bounded as:

$$
|\hat{w}^\top h| \leq \|\hat{w}\|_2 \|h\|_2.
$$

Thus, the Lipschitz constant $$L = \|\hat{w}\|_2$$ quantifies the maximum rate of change for the function $$\hat{f}(x)$$.

##### **Generalization to Other Norms**

The generalization to other norms comes from the equivalence of norms in finite-dimensional vector spaces. Here's the reasoning:

**Norm Equivalence:**

In finite-dimensional spaces (e.g., $$ \mathbb{R}^d $$), all norms are equivalent. This means there exist constants $$ C_1, C_2 > 0 $$ such that for any vector $$ \mathbf{w} \in \mathbb{R}^d $$:

$$
C_1 \| \mathbf{w} \|_p \leq \| \mathbf{w} \|_q \leq C_2 \| \mathbf{w} \|_p
$$

For example, the $$ L_1 $$, $$ L_2 $$, and $$ L_\infty $$ norms can all bound one another with appropriate scaling constants.

**Lipschitz Continuity:**

The Lipschitz constant for $$ \hat{f}(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} $$ depends on the norm of $$ \mathbf{w} $$ because the bound for the rate of change involves the norm of $$ \mathbf{w} $$. When using a different norm $$ \| \cdot \|_p $$ to regularize, the Lipschitz constant adapts to that norm.

Specifically, for the $$ L_p $$ norm:

$$
| \hat{f}(\mathbf{x} + \mathbf{h}) - \hat{f}(\mathbf{x}) | \leq \| \mathbf{w} \|_p \| \mathbf{h} \|_q
$$

where $$ p $$ and $$ q $$ are Hölder conjugates, satisfying:

$$
\frac{1}{p} + \frac{1}{q} = 1
$$

**Key Insight:**

This shows that the idea of controlling the sensitivity of the model (through the Lipschitz constant) extends naturally to any norm. The choice of norm alters how the regularization penalizes weights but retains the fundamental property of bounding the function's rate of change.

###### **Analogy for Ending**

Think of $$ L_2 $$ regularization as a bungee cord attached to a daring rock climber. The climber represents the model trying to navigate a complex landscape (data). Without the cord (regularization), they might venture too far and fall into overfitting. The cord adds just enough tension (penalty) to keep the climber balanced and safe, ensuring they explore the terrain without taking reckless leaps. Similarly, regularization helps the model stay grounded, generalizing well without succumbing to overfitting.

Now, imagine different types of bungee cords for different norms. The $$ L_2 $$ regularization bungee cord is like a standard elastic cord, providing a smooth and consistent tension, ensuring the climber doesn't over-extend but can still make significant progress.

For $$ L_1 $$ regularization, the bungee cord is more rigid and less forgiving, preventing large movements in any direction. It forces the climber to stick to fewer, more significant paths, like sparsity in feature selection — only the most important features remain.

In the case of $$ L_\infty $$ regularization, the bungee cord has a fixed maximum stretch. No matter how hard the climber tries to move, they cannot go beyond a certain point, ensuring the model remains under tight control, limiting the complexity of each individual parameter.

In each case, the regularization (the cord) helps the climber (the model) stay within safe bounds, preventing them from falling into overfitting while ensuring they can still navigate the data effectively.

--- 

#### **Linear Regression vs. Ridge Regression**

The inclusion of L2 regularization modifies the optimization objective, as illustrated by the difference between **linear regression** and **ridge regression**.

In **linear regression**, the goal is to minimize the sum of squared residuals, expressed as:

$$
L(w) = \frac{1}{2} \|Xw - y\|_2^2
$$

In contrast, **ridge regression** introduces an additional penalty term proportional to the L2 norm of the weights:

$$
L(w) = \frac{1}{2} \|Xw - y\|_2^2 + \frac{\lambda}{2} \|w\|_2^2
$$

This additional term penalizes large weights, helping to control model complexity and reduce overfitting.

##### **Gradients of the Objective**

The inclusion of the regularization term affects the gradient of the loss function. For linear regression, the gradient is:

$$
\nabla L(w) = X^T (Xw - y)
$$

For ridge regression, the gradient becomes:

$$
\nabla L(w) = X^T (Xw - y) + \lambda w
$$

The regularization term $$\lambda w$$ biases the solution toward smaller weights, thereby stabilizing the optimization.

##### **Closed-form Solutions**

Both linear regression and ridge regression admit closed-form solutions. For linear regression, the weights are given by:

$$
w = (X^T X)^{-1} X^T y
$$

For ridge regression, the solution is slightly modified:

$$
w = (X^T X + \lambda I)^{-1} X^T y
$$

The addition of $$\lambda I$$ ensures that $$X^T X + \lambda I$$ is always invertible, addressing potential issues of singularity in the design matrix.

#### **A Constrained Optimization Perspective**

The L2 regularization term can also be interpreted through the lens of constrained optimization. In this view, the ridge regression objective is expressed in its **Tikhonov form** as:

$$
w^* = \arg\min_w \frac{1}{2} \|Xw - y\|_2^2 + \frac{\lambda}{2} \|w\|_2^2
$$

Alternatively, using **Lagrangian theory**, we can reformulate this as a constraint on the norm of the weights:

$$
w^* = \arg\min_{w : \|w\|_2^2 \leq r} \frac{1}{2} \|Xw - y\|_2^2
$$

At the optimum, the gradients of the main objective and the constraint balance each other, providing a geometric interpretation of regularization.

[Express both Ivanov and Tikhonov form more clearly]

[Add a point stating that the Lagrangian Theory will be explain later while we discuss about SVM]

#### **Lasso Regression and $$L_1$$ Regularization**

While L2 regularization minimizes the sum of squared weights, **L1 regularization** (used in Lasso regression) minimizes the sum of absolute weights. This is expressed as:

$$
w^* = \arg\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n (\hat{w}^T x_i - y_i)^2 + \lambda \|w\|_1
$$

Here, the L1 norm 

$$
\|w\|_1 = |w_1| + |w_2| + \dots + |w_d|
$$

 encourages sparsity in the weight vector, setting some coefficients exactly to zero.

#### **Ridge vs. Lasso: A Comparative Analysis**

The key difference between ridge and lasso regression lies in their impact on the weights. Ridge regression tends to shrink all coefficients toward zero but does not eliminate any of them. In contrast, lasso regression produces sparse solutions, where some coefficients are exactly zero. **But why?** Keep reading!

This sparsity has significant practical advantages. By zeroing out irrelevant features, lasso regression simplifies the model, making it:

- **Faster** to compute, as fewer features need to be processed.
- **Cheaper** to store and deploy, especially on resource-constrained devices.
- **More interpretable**, as it highlights the most important features.
- **Less prone to overfitting**, since the reduced complexity often leads to better generalization.

##### **Quick Recap:**

Both L1 and L2 regularization are powerful techniques for controlling overfitting, improving interpretability, and ensuring numerical stability. L2 regularization excels in stabilizing solutions and handling multicollinearity, while L1 regularization shines when feature selection or sparsity is desired.

[Do I need this part?]

---

#### **Why Does $$L_1$$ Regularization Lead to Sparsity?**

A distinctive property of **L1 regularization** is its ability to produce sparse solutions, where some weights are exactly zero. This characteristic makes L1 regularization particularly useful for feature selection, as it effectively identifies the most important features by eliminating irrelevant ones. To understand this better, let’s explore the theoretical underpinnings and geometric intuition behind this phenomenon.

##### **Revisiting Lasso Regression**

Lasso regression penalizes the **L1 norm** of the weights. The objective function, also known as the **Tikhonov form**, is given by:

$$
\hat{w} = \arg\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \big(w^T x_i - y_i\big)^2 + \lambda \|w\|_1
$$

Here, the L1 norm is defined as:

$$
\|w\|_1 = |w_1| + |w_2| + \dots + |w_d|
$$

This formulation encourages sparsity by applying a uniform penalty across all weights, effectively "pushing" some weights to zero when they contribute minimally to the prediction.

##### **Regularization as Constrained Empirical Risk Minimization (ERM)**

Regularization can also be viewed through the lens of **constrained ERM**. For a given complexity measure $$\Omega$$ and a fixed threshold $$r \geq 0$$, the optimization problem is expressed as:

$$
\min_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \ell(f(x_i), y_i) \quad \text{s.t.} \quad \Omega(f) \leq r
$$

In the case of Lasso regression, this is equivalent to the **Ivanov form**:

$$
\hat{w} = \arg\min_{\|w\|_1 \leq r} \frac{1}{n} \sum_{i=1}^n \big(w^T x_i - y_i\big)^2
$$

Here, $$r$$ plays the same role as the regularization parameter $$\lambda$$ in the penalized ERM (Tikhonov) form. The choice between these forms depends on whether the complexity is penalized directly or constrained explicitly.

##### **The ℓ1 and ℓ2 Norm Constraints**

To understand why L1 regularization promotes sparsity, consider a simple hypothesis space $$\mathcal{F} = \{f(x) = w_1x_1 + w_2x_2\}$$. Each function can be represented as a point $$(w_1, w_2)$$ in $$\mathbb{R}^2$$. The regularization constraints can be visualized as follows:

- **L2 norm constraint:** 
$$w_1^2 + w_2^2 \leq r$$ (a circle in $$\mathbb{R}^2$$).
- **L1 norm constraint:** 
$$|w_1| + |w_2| \leq r$$ (a diamond in $$\mathbb{R}^2$$).

The sparse solutions correspond to the vertices of the diamond, where at least one weight is zero.

[Need more details and intuitons!]

##### **Visualizing Regularization**

To build intuition, let’s analyze the geometry of the optimization:

1. The **blue region** represents the feasible space defined by the regularization constraint 
(e.g., $$w_1^2 + w_2^2 \leq r$$ for L2, or $$|w_1| + |w_2| \leq r$$ for L1).
2. The **red contours** represent the level sets of the empirical risk 
$$
\hat{R}_n(w) = \frac{1}{n} \sum_{i=1}^n \big(w^T x_i - y_i\big)^2
$$.

The optimal solution is found where the smallest contour intersects the feasible region. For L1 regularization, this intersection tends to occur at the corners of the diamond, where one or more weights are exactly zero.

[Add the visualization!]

##### **Why Does ℓ1 Regularization Encourage Sparse Solutions?**

The sparsity induced by L1 regularization can be understood geometrically. Suppose the loss contours grow as perfect circles (or spheres in higher dimensions). When these contours intersect the diamond-shaped feasible region of L1 regularization, the corners of the diamond are more likely to be touched. These corners correspond to solutions where at least one weight is zero.

In contrast, for L2 regularization, the feasible region is a circle (or sphere), and the intersection is equally likely to occur in any direction. This results in small, but non-zero, weights across all features.

[Need things to Visualize!]

##### **Optimization Perspective**

From an optimization viewpoint, the difference between L1 and L2 regularization lies in how the penalty affects the gradient:

- For **L2 regularization**, as a weight $$w_i$$ becomes smaller, the penalty $$\lambda w_i^2$$ decreases more rapidly. However, the gradient of the penalty also diminishes, providing less incentive to shrink the weight to exactly zero.
- For **L1 regularization**, the penalty 
  $$\lambda |w_i|$$ decreases linearly, and its gradient remains constant regardless of the weight's size. This consistent gradient drives small weights to zero, promoting sparsity.

##### **Generalizing to ℓq Regularization**

L1 and L2 regularization are specific cases of the more general $$\ell_q$$ regularization, defined as:

$$
\|w\|_q^q = |w_1|^q + |w_2|^q + \dots + |w_d|^q
$$

Here are some notable cases:

- For $$q \geq 1$$, $$\|w\|_q$$ is a valid norm.
- For $$0 < q < 1$$, the constraint becomes non-convex, making optimization challenging. While $$\ell_q$$ regularization with $$q < 1$$ can induce even sparser solutions than L1, it is often impractical in real-world scenarios.
- The $$\ell_0$$ norm, defined as the number of non-zero weights, corresponds to **subset selection** but is computationally infeasible due to its combinatorial nature.

---

##### **Conclusion**

L1 regularization’s sparsity-inducing property makes it an indispensable tool in feature selection and high-dimensional problems. Its geometric intuition, optimization characteristics, and ability to simplify models while retaining interpretability set it apart from L2 regularization. By understanding the nuances of L1, L2, and generalized $$\ell_q$$ regularization, practitioners can leverage these techniques effectively to address diverse challenges in machine learning.
