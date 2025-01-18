---
layout: post
title: Unveiling Probabilistic Modeling
date: 2025-01-17 00:51:00-0400
featured: false
description: Explore the fundamentals of probabilistic modeling and how it enhances our understanding of linear regression, from parameter estimation to error distribution.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

#### **Why Probabilistic Modeling?**

Probabilistic modeling offers a unified framework that underpins many machine learning methods, from linear regression to logistic regression and beyond. At its core, probabilistic modeling allows us to handle uncertainty and make informed decisions based on observed data. It provides a principled way to update our beliefs about the data-generating process as new information becomes available.

In machine learning, we often think of learning as statistical inference, where the goal is to use data to draw conclusions about the underlying distribution or process that generated it, rather than simply fitting a model to the observed data. In this view, the goal is not just to fit a model to data, but to estimate the underlying distribution that best explains the observed data. Probabilistic methods give us a powerful tool to incorporate our beliefs about the world—often referred to as inductive biases—into the learning process. This allows us to make more informed predictions and gain deeper insights into the data.

For example, in Bayesian inference, prior beliefs are combined with evidence from the data to update our understanding of the underlying process. This principled approach enables us to not only predict outcomes but also quantify our confidence in those predictions, making probabilistic modeling a powerful tool for developing robust, interpretable, and informed machine learning systems.


##### **Two Ways of Generating Data**

When we think about how data is generated, there are two main perspectives to consider. The first is through **conditional models**, where we model the likelihood of the output $$y$$ given the input $$x$$. This is often denoted as $$p(y \vert x)$$.

The second perspective is through **generative models**, where we model the joint distribution of both the input $$x$$ and the output $$y$$, denoted as $$p(x, y)$$. 


To understand the distinction between conditional models and generative models, let’s use the analogy of handwriting recognition:

- **Conditional Models**: Imagine you are given a handwritten letter and asked to identify the corresponding alphabet. Here, you focus only on the relationship between the handwriting ($$x$$) and the letter it represents ($$y$$). This corresponds to modeling $$p(y \mid x)$$, where you predict the output $$y$$ (the letter) conditioned on the input $$x$$ (the handwriting). Essentially, you're answering the question, *"What is the most likely letter given this handwriting?"*

- **Generative Models**: Now, imagine that instead of just recognizing handwriting, you also aim to generate realistic handwriting for any letter. To do this, you need to understand how the letters ($$y$$) and handwriting styles ($$x$$) are generated together. This involves modeling the joint distribution $$p(x, y)$$, where you learn how inputs and outputs are related as part of a larger generative process. The question here becomes, *"How are handwriting ($$x$$) and letters ($$y$$) jointly produced?"*


Each approach offers different advantages depending on the context. However, both share the common goal of estimating the parameters of the model, often using a technique called **Maximum Likelihood Estimation (MLE)**.

---

#### **Conditional Models**

Conditional models focus on predicting the output given the input. One of the most well-known and widely used conditional models is **linear regression**. Let's take a closer look at linear regression and how it fits within this probabilistic framework.

#### **Linear Regression**

Linear regression is a fundamental technique in both machine learning and statistics. Its primary goal is to predict a real-valued target $$ y $$ (also called the response variable) from a vector of features $$ x $$ (also known as covariates). Linear regression is often used in situations where we want to predict a continuous value, such as:

- Predicting house prices based on factors like location, condition, and age of the house.
- Estimating medical costs of a person based on their age, sex, region, and BMI.
- Predicting someone's age from their photograph.

##### **The Problem Setup**

In linear regression, we are given a set of training examples, $$ D = \{(x^{(n)}, y^{(n)})\}_{n=1}^N $$, where $$ x^{(n)} \in \mathbb{R}^d $$ represents the features and $$ y^{(n)} \in \mathbb{R} $$ represents the target. The task is to model the relationship between the features $$ x $$ and the target $$ y $$.

To do this, we assume that there is a linear relationship between $$ x $$ and $$ y $$, which can be expressed as:

$$
h(x) = \theta^T x = \sum_{i=0}^{d} \theta_i x_i
$$

Here, $$ \theta \in \mathbb{R}^d $$ represents the parameters (also known as the weights) of the model, and $$ x_0 = 1 $$ is the bias term. The goal is to find the values of $$ \theta $$ that best explain the observed data.

> *"We use superscript to denote the example id and subscript to denote the dimension id"*

##### **Unveiling Probabilistic Modeling**

To estimate the parameters $$ \theta $$, we use the **least squares method**, which involves minimizing the squared loss between the predicted and observed values. The loss function is defined as:

$$
J(\theta) = \frac{1}{N} \sum_{n=1}^{N} \left( y^{(n)} - \theta^T x^{(n)} \right)^2
$$

This function represents the **empirical risk**, which quantifies the difference between the predicted and actual values across all the training examples.

##### **Matrix Formulation**

We can also express this problem in matrix form for efficiency. Let $$ X \in \mathbb{R}^{N \times d} $$ be the design matrix, whose rows represent the input features for each training example. Let $$ y \in \mathbb{R}^N $$ be the vector of all target values. The objective is to solve for the parameter vector $$ \hat{\theta} $$ that minimizes the loss:

$$
\hat{\theta} = \arg\min_\theta \left( (X\theta - y)^T (X\theta - y) \right)
$$

##### **Closed-Form Solution**

The closed-form solution to this optimization problem is:

$$
\hat{\theta} = (X^T X)^{-1} X^T y
$$

This gives us the values for $$ \theta $$ that minimize the squared loss, and hence, provide the best linear model for the data.


---

Before proceeding further, here are a few review questions. Ask yourself these and check the answers.

##### **How do we derive the solution for linear regression?**  

The squared loss function in matrix form is:  

$$ 
J(\theta) = \frac{1}{N} (X\theta - y)^T (X\theta - y) 
$$  

To minimize $$ J(\theta) $$, we compute the gradient with respect to $$ \theta $$:  

- Expand the quadratic term:  
   
$$ J(\theta) = \frac{1}{N} \left[ \theta^T X^T X \theta - 2y^T X \theta + y^T y \right] $$  

- Take the derivative with respect to $$ \theta $$:  
    - Recall that for any vector $$ a $$, $$ b $$, and matrix $$ A $$, the following derivatives are useful:  
        - $$ \frac{\partial (a^T b)}{\partial a} = b $$  
        - $$
        \frac{\partial (a^T A a)}{\partial a} = 2A a 
        $$ 
        
        \(when $$ A $$ is symmetric\).  

Applying these rules:  

$$ \nabla_\theta J(\theta) = \frac{1}{N} \left[ 2X^T X \theta - 2X^T y \right]. $$  

- Set the gradient to zero to find the minimizer:  
   
$$ X^T X \theta = X^T y $$  

- Solve for $$ \theta $$:  
   
$$ \theta = (X^T X)^{-1} X^T y, $$

provided $$ X^T X $$ is invertible.

**Note:** The $$ \frac{1}{N} $$ normalization factor is constant and cancels out when setting the gradient to zero, so it does not affect the solution for $$ \theta $$.


##### **Why Do Transposes Appear or Disappear?**

1. **Symmetry of Quadratic Terms**:  
In the term $$ \theta^T X^T X \theta $$, note that $$ X^T X $$ is a symmetric matrix (because $$ X^T X = (X^T X)^T $$). This symmetry ensures that when taking the derivative, we don’t need to explicitly add or remove transposes; they naturally align.

1. **Consistency of Vector-Matrix Multiplication**:  
When differentiating terms like $$ y^T X \theta $$, we use the rule $$ \frac{\partial (a^T b)}{\partial a} = b $$, ensuring dimensions match. This often introduces or removes a transpose based on the structure of the derivative. For example:  
- $$ \nabla_\theta (-2y^T X \theta) = -2X^T y $$, where $$ X^T $$ arises naturally to align dimensions.

1. **Gradient Conventions**:  
The transpose changes are necessary to ensure the resulting gradient is a column vector (matching $$ \theta $$’s shape), as gradients are typically represented in the same dimensionality as the parameter being differentiated.


##### **What happens if $$ X^T X $$ is not invertible?**  
  If $$X^T X$$ is not invertible (also called singular or degenerate), the normal equations do not have a unique solution. This happens in cases such as:  
  - **Linearly dependent features**: Some columns of $$X$$ are linear combinations of others.  
  - **Too few data points**: If $$N < d$$ (more features than samples), $$X^T X$$ will not be full rank.  

  To address this issue, we can:  
  1. **Add regularization**: Use techniques like Ridge Regression, which modifies the normal equation to include a penalty term:  
     $$ \theta = (X^T X + \lambda I)^{-1} X^T y, $$  
     where $$\lambda > 0$$ is the regularization parameter.  
  2. **Remove redundant features**: Perform feature selection or dimensionality reduction (e.g., PCA) to eliminate linear dependencies.  
  3. **Use pseudo-inverse**: Compute the Moore-Penrose pseudo-inverse of $$X^T X$$ to find a solution.  

---

#### **Understanding Linear Regression Through a Probabilistic Lens**

So far, we've discussed how linear regression can be understood as minimizing the squared loss. But why is the squared loss a reasonable choice for regression problems? To answer this, we need to think about the assumptions we are making on the data.

Let’s approach linear regression from a **probabilistic modeling perspective**.

##### **Assumptions in Linear Regression**

In this framework, we assume that the target $$ y $$ and the features $$ x $$ are related through a linear function, with an added error term $$ \epsilon $$:

$$
y = \theta^T x + \epsilon
$$

Here, $$ \epsilon $$ represents the residual error that accounts for all unmodeled effects, such as noise or other sources of variation in the data. We assume that these errors $$ \epsilon $$ are independent and identically distributed (iid) and follow a normal distribution:

$$
\epsilon \sim \mathcal{N}(0, \sigma^2)
$$

Given this assumption, the conditional distribution of $$ y $$ given $$ x $$ is a normal distribution with mean $$ \theta^T x $$ and variance $$ \sigma^2 $$:

$$
p(y | x; \theta) = \mathcal{N}(\theta^T x, \sigma^2)
$$

##### **Intuition Behind the Gaussian Distribution**

This distribution suggests that, for each value of $$ x $$, the output $$ y $$ is normally distributed around the value predicted by the linear model $$ \theta^T x $$, with a fixed variance $$ \sigma^2 $$ that captures the uncertainty or noise in the data. In other words, we place a Gaussian "bump" around the output of the linear predictor, reflecting the uncertainty in our prediction.

With this, we've laid the groundwork for our discussion on Maximum Likelihood Estimation.

---

##### **Conclusion**

In this post, we introduced how probabilistic modeling can be used for understanding and estimating machine learning models, such as linear regression. By thinking of learning as statistical inference, we can incorporate our prior beliefs about the data-generating process and make more informed predictions.

Next, we'll dive into **Maximum Likelihood Estimation (MLE)** and examine how it can be applied to solve probabilistic linear regression and other machine learning algorithms. We'll also explore how to formalize this understanding—stay tuned!

