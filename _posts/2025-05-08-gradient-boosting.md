---
layout: post
title: Gradient Boosting / "Anyboost"
date: 2025-05-08 12:56:00-0400
featured: false
description: From FSAM to Gradient Boosting - Extending Beyond Exponential Loss
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

In our previous post, we explored how **Forward Stagewise Additive Modeling (FSAM)** with exponential loss recovers the AdaBoost algorithm. But FSAM is not limited to exponential loss — it can be extended to **any differentiable loss**, leading to the powerful and flexible framework of **Gradient Boosting Machines (GBMs)**.

This post walks through the derivation of gradient boosting, starting with the squared loss, and builds toward a general functional gradient descent interpretation of boosting.

---

##### **FSAM with Squared Loss**

Let’s begin with FSAM using squared loss. At the $$m$$-th boosting round, we wish to find a new function $$v h(x)$$ to add to the model $$f_{m-1}(x)$$. The objective becomes:

$$
J(v, h) = \frac{1}{n} \sum_{i=1}^n \left( y_i - \underbrace{[f_{m-1}(x_i) + v h(x_i)]}_{\text{new model}} \right)^2
$$

If the hypothesis space $$\mathcal{H}$$ is closed under rescaling (i.e., if $$h \in \mathcal{H}$$ implies $$v h \in \mathcal{H}$$ for all $$v \in \mathbb{R}$$), we can drop $$v$$ and simplify the objective:

$$
J(h) = \frac{1}{n} \sum_{i=1}^n \left( \underbrace{[y_i - f_{m-1}(x_i)]}_{\text{residual}} - h(x_i) \right)^2
$$

This is just **least squares regression on the residuals** — fit $$h(x)$$ to approximate the residuals $$[y_i - f_{m-1}(x_i)]$$.

##### **Interpreting the Residuals**

Let’s take a closer look at how **residuals relate to gradients** in the context of boosting with squared loss.

The objective for squared loss is:

$$
J(f) = \frac{1}{n} \sum_{i=1}^n (y_i - f(x_i))^2
$$

This measures how far the model predictions $$f(x_i)$$ are from the true labels $$y_i$$ across the dataset.

To minimize this objective, we can perform gradient descent. So we ask: **what is the gradient of $$J(f)$$ with respect to $$f(x_i)$$?**

Let’s compute it:

$$
\frac{\partial}{\partial f(x_i)} J(f) = \frac{\partial}{\partial f(x_i)} \left[ \frac{1}{n} \sum_{j=1}^n (y_j - f(x_j))^2 \right]
$$

Because only the $$i$$-th term depends on $$f(x_i)$$, this simplifies to:

$$
\frac{\partial}{\partial f(x_i)} J(f) = \frac{1}{n} \cdot (-2)(y_i - f(x_i)) = -\frac{2}{n}(y_i - f(x_i)) \tag{20}
$$

The term $$y_i - f(x_i)$$ is the **residual** at point $$x_i$$. So we see:

> The residual is proportional to the **negative gradient** of the squared loss.

**Why This Matters for Boosting**

Gradient descent updates a parameter in the opposite direction of the gradient. Similarly, in **gradient boosting**, we update our model $$f$$ in the direction of a function that tries to approximate the **negative gradient** at every data point.

In the case of squared loss, this just means:

- Compute the residuals $$r_i = y_i - f(x_i)$$.
- Fit a new base learner $$h_m$$ to those residuals.
- Add $$h_m$$ to the model: $$f \leftarrow f + v h_m$$.

This process mimics the behavior of **gradient descent in function space**.

We can now draw an analogy:

- **Boosting update:**  $$f \leftarrow f + v h$$
- **Gradient descent:** $$f \leftarrow f - \alpha \nabla_f J(f)$$

Where:
- $$h$$ approximates the direction of steepest descent (the gradient),
- $$v$$ is the step size (akin to learning rate $$\alpha$$),
- and the update improves the model's predictions to reduce the loss.

This perspective generalizes easily to other loss functions, which we explore in the next sections on **functional gradient descent**.

---

##### **Functional Gradient Descent: Intuition and Setup**

To generalize FSAM to arbitrary (differentiable) loss functions, we adopt the **functional gradient descent** perspective.

Suppose we have a loss function that depends on predictions $$f(x_i)$$ at $$n$$ training examples:

$$
J(f) = \sum_{i=1}^n \ell(y_i, f(x_i))
$$

Note that $$f$$ is a function, but this loss only depends on $$f$$ through its values on the training points. So, we can treat:

$$
f = (f(x_1), f(x_2), \dots, f(x_n))^\top
$$

as a vector in $$\mathbb{R}^n$$, and write:

$$
J(f) = \sum_{i=1}^n \ell(y_i, f_i)
$$

where $$f_i := f(x_i)$$. We want to minimize $$J(f)$$ by updating our predictions $$f_i$$ in the **steepest descent direction**.


##### **Unconstrained Step Direction (Pseudo-Residuals)**

We compute the gradient of the loss with respect to each prediction:

$$
g = \nabla_f J(f) = \left( \frac{\partial \ell(y_1, f_1)}{\partial f_1}, \dots, \frac{\partial \ell(y_n, f_n)}{\partial f_n} \right)
$$

The **negative gradient direction** is:

$$
-g = -\nabla_f J(f) = \left( -\frac{\partial \ell(y_1, f_1)}{\partial f_1}, \dots, -\frac{\partial \ell(y_n, f_n)}{\partial f_n} \right)
$$

This tells us how to change each $$f(x_i)$$ to decrease the loss — it's the direction of steepest descent in $$\mathbb{R}^n$$.

We call this vector the **pseudo-residuals**. In the case of squared loss:

$$
\ell(y_i, f_i) = (y_i - f_i)^2 \quad \Rightarrow \quad -g_i = y_i - f_i
$$

So, pseudo-residuals coincide with actual residuals for squared loss.


##### **Projection Step: Fitting the Pseudo-Residuals**

We now want to update the function $$f$$ by stepping in direction $$-g$$. However, we can’t directly take a step in $$\mathbb{R}^n$$ — we must stay within our base hypothesis space $$\mathcal{H}$$.

So we find the **function $$h \in \mathcal{H}$$** that best fits the negative gradient at the training points. This is a projection of $$-g$$ onto the function class:

$$
h = \arg\min_{h \in \mathcal{H}} \sum_{i=1}^n \left( -g_i - h(x_i) \right)^2
\tag{26}
$$

This is just **least squares regression** of pseudo-residuals.

Once we have $$h$$, we take a step:

$$
f \leftarrow f + v h
\tag{27}
$$

where $$v$$ is a step size, which can either be fixed (e.g., shrinkage factor $$\lambda$$) or found via line search.


**All Together;**

- **Objective**:
  $$
  J(f) = \sum_{i=1}^n \ell(y_i, f(x_i))
  \tag{24}
  $$

- **Unconstrained gradient**:
  $$
  g = \nabla_f J(f) = \left( \frac{\partial \ell(y_1, f_1)}{\partial f_1}, \dots, \frac{\partial \ell(y_n, f_n)}{\partial f_n} \right)
  \tag{25}
  $$

- **Projection**:
  $$
  h = \arg\min_{h \in \mathcal{H}} \sum_{i=1}^n (-g_i - h(x_i))^2
  \tag{26}
  $$

- **Boosting update**:
  $$
  f \leftarrow f + v h
  \tag{27}
  $$

This gives a general recipe for boosting: **approximate the negative gradient with a weak learner and take a small step in that direction** — hence the name **gradient boosting**.


##### **Functional Gradient Descent: Hyperparameters and Regularization**

Once we have our base learner $$h_m$$ for iteration $$m$$, we need to decide how far to move in that direction.

We can either:

- **Choose a step size $$v_m$$ by line search**:
  $$
  v_m = \arg\min_v \sum_{i=1}^n \ell\left(y_i, f_{m-1}(x_i) + v h_m(x_i)\right)
  $$

- **Or** use a fixed step size $$v$$ as a hyperparameter. Line search is not strictly necessary but can improve performance.

To **regularize** and control overfitting, we scale the step size with a **shrinkage factor** $$\lambda \in (0, 1]$$:

$$
f_m(x) = f_{m-1}(x) + \lambda v_m h_m(x)
\tag{28}
$$

This shrinkage slows down learning and typically improves generalization. A common choice is $$\lambda = 0.1$$.

We also need to decide:

- The **number of steps $$M$$** (i.e., number of boosting rounds).
  - This is typically chosen via early stopping or tuning on a validation set.

---

##### **Gradient Boosting Algorithm**

Putting everything together, the gradient boosting algorithm proceeds as follows:

1. **Initialize the model** with a constant function:
   $$
   f_0(x) = \arg\min_\gamma \sum_{i=1}^n \ell(y_i, \gamma)
   $$

2. **For** $$m = 1$$ to $$M$$:

   1. **Compute the pseudo-residuals** (i.e., the negative gradients):

      $$
      r_{im} = -\left[ \frac{\partial}{\partial f(x_i)} \ell(y_i, f(x_i)) \right]_{f(x_i) = f_{m-1}(x_i)}
      \tag{29}
      $$

   2. **Fit a base learner** $$h_m$$ (e.g., regression tree) to the dataset $$\{(x_i, r_{im})\}_{i=1}^n$$ using squared error.

   3. *(Optional)* **Line search** to find best step size:
      $$
      v_m = \arg\min_v \sum_{i=1}^n \ell(y_i, f_{m-1}(x_i) + v h_m(x_i))
      $$

   4. **Update the model**:
      $$
      f_m(x) = f_{m-1}(x) + \lambda v_m h_m(x)
      $$

3. **Return** the final model: $$f_M(x)$$

---

##### **Conclusion: The Gradient Boosting Machine Ingredients**

To implement gradient boosting, you need:

- A loss function $$\ell(y, f(x))$$ that is differentiable w.r.t. $$f(x)$$
- A base hypothesis space $$\mathcal{H}$$ (e.g., decision trees) for regression
- A method to choose step size: fixed, or via line search
- A stopping criterion: number of iterations $$M$$, or early stopping
- Regularization through **shrinkage** ($$\lambda$$)

Once these ingredients are in place, you're ready to build powerful models with **Gradient Boosting Machines (GBMs)**!

In the next post, we’ll explore specific loss functions like **logistic loss** for classification and how gradient boosting compares to other ensemble methods.
