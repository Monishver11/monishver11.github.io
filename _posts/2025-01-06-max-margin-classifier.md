---
layout: post
title: Understanding the Maximum Margin Classifier
date: 2025-01-06 01:57:00-0400
featured: false
description: An engaging walkthrough of maximum margin classifiers, exploring their foundations, geometric insights, and the transition to support vector machines.
tags: ML
categories: sample-posts
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

#### **Linearly Separable Data**

Let’s start with the simplest case: linearly separable data. Imagine a dataset where we can draw a straight line (or more generally, a hyperplane in higher dimensions) to perfectly separate two classes of points. Formally, for a dataset $$ D $$ with points $$ (x_i, y_i) $$, we seek a hyperplane that satisfies the following conditions:

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Max_Margin_Classifier_1.png" title="Max_Margin_Classifier_1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- $$w^T x_i > 0$$ for all $$x_i$$ where $$y_i = +1$$,
- $$w^T x_i < 0$$ for all $$x_i$$ where $$y_i = -1$$.

This hyperplane is defined by a weight vector $$w$$ and a bias $$b$$, and our goal is to find $$w$$ and $$b$$ such that all points are correctly classified.

But how do we design a learning algorithm to find such a hyperplane? This brings us to the **Perceptron Algorithm**.

#### **The Perceptron Algorithm**

The perceptron is one of the earliest learning algorithms developed to find a separating hyperplane. Here’s how it works: we start with an initial guess for $$w$$ (usually a zero vector) and iteratively adjust it based on misclassified examples.

Each time we encounter a point $$ (x_i, y_i) $$ that is misclassified (i.e., $$y_i w^T x_i < 0$$), we update the weight vector as follows:

$$
w \gets w + y_i x_i.
$$

This update rule ensures that the algorithm moves the hyperplane towards misclassified positive examples and away from misclassified negative examples.

The perceptron algorithm has a remarkable property: if the data is linearly separable, it will converge to a solution with zero classification error in a finite number of steps.

In terms of loss functions, the perceptron can be viewed as minimizing the **hinge loss**:

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Max_Margin_Classifier_2.png" title="Max_Margin_Classifier_2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

$$
\ell(x, y, w) = \max(0, -y w^T x).
$$

However, while the perceptron guarantees a solution, it doesn’t always find the best one. This brings us to the concept of **maximum-margin classifiers**.

[How this rule update works, a better thought;]

---

#### **Maximum-Margin Separating Hyperplane**

When the data is linearly separable, there are infinitely many hyperplanes that can separate the classes. The perceptron algorithm, for instance, might return any one of these. But not all hyperplanes are equally desirable.

We prefer a hyperplane that is farthest from both classes of points. This idea leads to the concept of the **maximum-margin classifier**, which finds the hyperplane that maximizes the smallest distance between the hyperplane and the data points.

##### **Geometric Margin**

The **geometric margin** of a hyperplane is defined as the smallest distance between the hyperplane and any data point. For a hyperplane defined by $$w$$ and $$b$$, this margin can be expressed as:

$$
\gamma = \min_i \frac{y_i (w^T x_i + b)}{\|w\|_2}.
$$

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Max_Margin_Classifier_3.png" title="Max_Margin_Classifier_3" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Maximizing this geometric margin provides a hyperplane that is robust to small perturbations in the data, making it a desirable choice.

##### **Distance Between a Point and a Hyperplane**

To understand the geometric margin more concretely, let’s calculate the distance from a point $$x'$$ to a hyperplane $$H: w^T v + b = 0$$. The signed distance is given by:

$$
d(x', H) = \frac{w^T x' + b}{\|w\|_2}.
$$

Taking into account the label $$y$$, the distance becomes:

$$
d(x', H) = \frac{y (w^T x' + b)}{\|w\|_2}.
$$

This distance is the foundation for defining and maximizing the geometric margin.

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Max_Margin_Classifier_4.png" title="Max_Margin_Classifier_4" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


##### **Maximizing the Margin**

To maximize the margin, we solve the following optimization problem:

$$
\max \min_i \frac{y_i (w^T x_i + b)}{\|w\|_2}.
$$

To simplify, let $$M = \min_i \frac{y_i (w^T x_i + b)}{\|w\|_2}$$. The problem becomes:

$$
\max M, \quad \text{subject to } \frac{y_i (w^T x_i + b)}{\|w\|_2} \geq M, \; \forall i.
$$

[Explain this more clearly, how this is written;]

By fixing $$\|w\|_2 = \frac{1}{M}$$, we reformulate it as:

$$
\min \frac{1}{2} \|w\|_2^2, \quad \text{subject to } y_i (w^T x_i + b) \geq 1, \; \forall i.
$$

This is the optimization problem solved by a **hard margin support vector machine (SVM)**.

##### **What If the Data Is Not Linearly Separable?**

In real-world scenarios, data is often not perfectly linearly separable. For any $$w$$, there might be points with negative margins. To handle such cases, we introduce **slack variables** $$\xi_i$$, which allow some margin violations.

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Max_Margin_Classifier_5.png" title="Max_Margin_Classifier_5" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

#### **Soft Margin SVM**

The optimization problem for a soft margin SVM is:

$$
\min \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \xi_i,
$$

subject to:

$$
y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \; \forall i.
$$

[More explanation on the equation;]

Here, $$C$$ is a parameter that controls the trade-off between maximizing the margin and penalizing violations. The slack variable $$\xi_i$$ measures how far the point $$x_i$$ violates the margin:
- $$\xi_i = 0$$: $$x_i$$ satisfies the margin condition.
- $$\xi_i > 0$$: $$x_i$$ violates the margin by a factor proportional to $$\xi_i$$.

[Slack variables example, missed it from slide;]

---

##### **Final Thoughts**

The maximum-margin classifier forms the foundation of modern support vector machines. By focusing on the hyperplane with the largest margin, it ensures robustness and generalizability. For non-linearly separable data, the introduction of slack variables allows SVMs to adapt while maintaining their core principle of maximizing the margin.

In the next post, we’ll explore how the world of SVMs and how it works under the hood. Stay tuned!
