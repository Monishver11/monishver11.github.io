---
layout: post
title: Introduction to Ensemble Methods
date: 2025-04-24 20:24:00-0400
featured: false
description: A beginner's guide to ensemble methods in machine learning, explaining how averaging and bootstrapping reduce variance and improve model performance.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

Ensemble methods are a powerful set of techniques in machine learning that aim to improve prediction performance by combining the outputs of multiple models. Before diving into ensemble strategies, let’s revisit some foundational concepts that lead us to the rationale behind ensemble methods.

---

##### **Review: Decision Trees**

- **Non-linear**, **non-metric**, and **non-parametric** models
- Capable of **regression** or **classification**
- **Interpretable**, especially when shallow
- Constructed using a **greedy algorithm** that seeks to maximize the **purity of nodes**
- Prone to **overfitting**, unless properly regularized

These models serve as the building blocks for many ensemble techniques like Random Forests.

##### **Recap: Statistics and Point Estimators**

We begin with data: 

$$
D = (x_1, x_2, \dots, x_n)
$$  

sampled i.i.d. from a parametric distribution  

$$
p(\cdot \mid \theta)
$$

A **statistic** is any function of the data, e.g.,

- Sample mean
- Sample variance
- Histogram
- Empirical distribution

A **point estimator** is a statistic used to estimate a parameter:

$$
\hat{\theta} = \hat{\theta}(D) \approx \theta
$$

**Example:**  Suppose we're estimating the average height $$\theta$$ of a population. We collect a sample of $$n$$ people and compute the sample mean:

$$
\hat{\theta}(D) = \frac{1}{n} \sum_{i=1}^n x_i
$$

This sample mean is a **point estimator** for the true average height $$\theta$$ of the entire population.

##### **Recap: Bias and Variance of an Estimator**

Since statistics are derived from random samples, they themselves are **random variables**. The distribution of a statistic across different random samples is called the **sampling distribution**.

Understanding the **bias** and **variance** of an estimator helps us evaluate how good the estimator is.

- **Bias** measures the systematic error — how far, on average, the estimator is from the true parameter:
  
  $$
  \text{Bias}(\hat{\theta}) \overset{\text{def}}{=} \mathbb{E}[\hat{\theta}] - \theta
  $$

- **Variance** measures the variability of the estimator due to sampling randomness:
  
  $$
  \text{Var}(\hat{\theta}) \overset{\text{def}}{=} \mathbb{E}[\hat{\theta}^2] - \left(\mathbb{E}[\hat{\theta}]\right)^2
  $$

Intuitively:

- **Low bias** means the estimator is *accurate* on average.
- **Low variance** means the estimator is *stable* across different samples.

Even an **unbiased estimator** can be **unreliable** if its variance is high. That is, it may give wildly different results on different samples, even if the average over many samples is correct.

**Example:**  Suppose we are trying to estimate the **true mean** $$\theta$$ of a population — for example, the average height of all adults in a city. We collect a sample of size $$n$$:

$$
D = (x_1, x_2, \dots, x_n)
$$

where each $$x_i$$ is drawn i.i.d. from a distribution with **mean** $$\theta$$ and some unknown variance $$\sigma^2$$.

Consider two different estimators of the population mean:

1. $$\hat{\theta}_1(D) = x_1$$ — just the first point in the sample  
2. $$\hat{\theta}_2(D) = \frac{1}{n} \sum_{i=1}^n x_i$$ — the sample mean

We say an estimator $$\hat{\theta}$$ is **unbiased** if, on average, it correctly estimates the true value of the parameter $$\theta$$:

$$
\mathbb{E}[\hat{\theta}] = \theta
$$

In this case:

- For $$\hat{\theta}_1$$, since $$x_1$$ is sampled from the distribution with mean $$\theta$$, we have:

  $$
  \mathbb{E}[x_1] = \theta \Rightarrow \mathbb{E}[\hat{\theta}_1] = \theta
  $$

- For $$\hat{\theta}_2$$, because of the linearity of expectation:

  $$
  \mathbb{E}[\hat{\theta}_2] = \mathbb{E}\left[\frac{1}{n} \sum_{i=1}^n x_i\right] = \frac{1}{n} \sum_{i=1}^n \mathbb{E}[x_i] = \frac{1}{n} \cdot n \cdot \theta = \theta
  $$

Thus, **both estimators are unbiased** — their expected value equals the true mean $$\theta$$.

However, they differ in **variance**:

- $$\hat{\theta}_1$$ uses only one data point, so its value can fluctuate greatly between different samples — it has **high variance**.
- $$\hat{\theta}_2$$ averages over all $$n$$ data points, which helps cancel out individual fluctuations — it has **lower variance**.

**Key takeaway:**  Although both estimators are unbiased, the sample mean $$\hat{\theta}_2$$ is **more reliable** due to its lower variance. This highlights the importance of considering both **bias** and **variance** when evaluating an estimator — a foundational idea for ensemble methods.

##### **Variance of a Mean**

Suppose we have an unbiased estimator $$\hat{\theta}$$ with variance $$\sigma^2$$:

$$
\mathbb{E}[\hat{\theta}] = \theta, \quad \text{Var}(\hat{\theta}) = \sigma^2
$$

Now imagine we have $$n$$ independent copies of this estimator — say, from different data samples or different random seeds — denoted by:

$$
\hat{\theta}_1, \hat{\theta}_2, \dots, \hat{\theta}_n
$$

We form a new estimator by averaging them:

$$
\hat{\theta}_{\text{avg}} = \frac{1}{n} \sum_{i=1}^n \hat{\theta}_i
$$

This average is still an unbiased estimator of $$\theta$$, because the expectation of a sum is the sum of expectations:

$$
\mathbb{E}[\hat{\theta}_{\text{avg}}] = \frac{1}{n} \sum_{i=1}^n \mathbb{E}[\hat{\theta}_i] = \frac{1}{n} \cdot n \cdot \theta = \theta
$$

But here's the key insight: its **variance is smaller**. Since the $$\hat{\theta}_i$$ are independent and each has variance $$\sigma^2$$, we get:

$$
\text{Var}(\hat{\theta}_{\text{avg}}) = \text{Var} \left( \frac{1}{n} \sum_{i=1}^n \hat{\theta}_i \right) = \frac{1}{n^2} \sum_{i=1}^n \text{Var}(\hat{\theta}_i) = \frac{n \cdot \sigma^2}{n^2} = \frac{\sigma^2}{n}
$$

So, by averaging multiple estimators, we **preserve unbiasedness** while **reducing variance**.  

This simple statistical property is the backbone of many ensemble methods in machine learning — especially those like bagging and random forests, where we average multiple models to get a more stable and reliable prediction.

---

##### **Averaging Independent Prediction Functions**

Let’s now connect the earlier statistical insight to machine learning.

Suppose we train $$B$$ models **independently** on $$B$$ different training sets, each drawn from the same underlying data distribution. This gives us a set of prediction functions:

$$
\hat{f}_1(x),\ \hat{f}_2(x),\ \dots,\ \hat{f}_B(x)
$$

We define their **average prediction function** as:

$$
\hat{f}_{\text{avg}}(x) \overset{\text{def}}{=} \frac{1}{B} \sum_{b=1}^B \hat{f}_b(x)
$$

At any specific input point $$x_0$$:

- Each model's prediction $$\hat{f}_b(x_0)$$ is a random variable (since it depends on the randomly drawn training set), but they all share the same **expected prediction**.
- The average prediction:

  $$
  \hat{f}_{\text{avg}}(x_0) = \frac{1}{B} \sum_{b=1}^B \hat{f}_b(x_0)
  $$

  has the **same expected value** as each individual $$\hat{f}_b(x_0)$$, but with a **reduced variance**:

  $$
  \text{Var}(\hat{f}_{\text{avg}}(x_0)) = \frac{1}{B} \cdot \text{Var}(\hat{f}_1(x_0))
  $$

This means that by averaging multiple independent models, we can achieve a **more stable and less noisy prediction**, without increasing bias.

However, here’s the challenge:  
> In practice, we don’t have access to $$B$$ truly independent training sets — we usually only have **one dataset**.

##### **The Bootstrap Sample**

So, how do we simulate multiple datasets when we only have **one**?

The answer is: use **bootstrap sampling** — a clever statistical trick to mimic sampling variability.

A **bootstrap sample** is formed by sampling **with replacement** from the original dataset:

$$
D_n = (x_1, x_2, \dots, x_n)
$$

We draw $$n$$ points *with replacement* from $$D_n$$, forming a new dataset (also of size $$n$$). Because sampling is done with replacement, some data points will appear multiple times, while others might not appear at all.

What's the chance a particular data point $$x_i$$ is **not** selected in one draw?  
That’s $$1 - \frac{1}{n}$$.  

The chance it's not selected **in any of the $$n$$ draws** is:

$$
\left(1 - \frac{1}{n} \right)^n \approx \frac{1}{e} \approx 0.368
$$

So, on average, **about 36.8%** of the data points are **not** included in a given bootstrap sample. This also means:

> Around **63.2%** of the original data points are expected to appear **at least once** in each bootstrap sample.

##### **The Bootstrap Method**

Bootstrap gives us a way to **simulate variability** and generate multiple pseudo-datasets — without requiring any new data.

Here's how it works:

1. From the original dataset $$D_n$$, generate $$B$$ bootstrap samples:

   $$
   D_n^1, D_n^2, \dots, D_n^B
   $$

2. Compute some function of interest — such as a statistic or a trained model — on each bootstrap sample:

   $$
   \phi(D_n^1), \phi(D_n^2), \dots, \phi(D_n^B)
   $$

These values behave almost like they were computed from **$$B$$ independent samples** drawn from the original population distribution.

**Why?** Because each bootstrap sample is a randomized resampling of the original dataset — introducing enough variability to approximate the natural randomness we'd expect from drawing entirely new datasets from the true distribution.

Although bootstrap samples are not truly independent, the statistical properties of estimators (like variance and confidence intervals) computed using bootstrapping tend to closely mirror those from actual independent samples.

> This makes bootstrap an incredibly useful tool for simulating sampling distributions, especially when acquiring more data is costly or impossible.


##### **Independent Samples vs. Bootstrap Samples**

Let’s say we want to estimate a parameter $$\alpha$$ using a point estimator:

$$
\hat{\alpha} = \hat{\alpha}(D_{100})
$$

Now, consider two scenarios:

- **Case 1:** You collect **1000 independent samples**, each of size 100  (**Left**)
- **Case 2:** You generate **1000 bootstrap samples** from a **single dataset** of size 100 (**Right**)

If you compute $$\hat{\alpha}$$ for each sample and plot the resulting histograms, you’ll notice something powerful:

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/ensemble-1.png" title="ensemble-1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

> The distribution of estimates from bootstrap samples closely resembles that from truly independent samples.

While not exact, the **bootstrap approximation** to the sampling distribution is often **good enough** for practical applications — especially when collecting new data is expensive or infeasible.

---

##### **Ensemble Methods**

This naturally leads us to the concept of **ensemble learning** — a powerful technique in modern machine learning.

**Core idea:** Combine multiple **weak models** into a **strong, robust predictor**.

Why ensemble methods work:

- Averaging predictions from i.i.d. models **reduces variance** without increasing bias
- Bootstrap lets us **simulate multiple training sets**, even from just one dataset

There are two primary flavors of ensemble methods:

- **Parallel Ensembles** (e.g., **Bagging**):  
  Models are trained **independently** on different subsets of data

- **Sequential Ensembles** (e.g., **Boosting**):  
  Models are trained **sequentially**, with each new model **focusing on the errors** made by previous ones

By leveraging these strategies, ensemble methods often achieve **greater accuracy**, **stability**, and **generalization** than any single model could alone.

---

Stay tuned as we dive deeper into specific ensemble techniques — starting with **Bagging** and the incredibly popular **Random Forests** — and see how these ideas come to life in practice!
