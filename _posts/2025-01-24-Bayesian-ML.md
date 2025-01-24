---
layout: post
title: Bayesian Machine Learning - Mathematical Foundations
date: 2025-01-24 10:56:00-0400
featured: false
description: A beginner-friendly guide to Bayesian statistics, explaining priors, likelihoods, posteriors, and real-world examples like coin-flipping to build a clear and intuitive understanding.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---


When working with machine learning models, it's crucial to understand the underlying statistical principles that drive our methods. Whether you're a frequentist or a Bayesian, the starting point often involves a **parametric family of densities**. This concept forms the foundation for inference and is used to model the data we observe.

#### **Parametric Family of Densities**

A **parametric family of densities** is defined as a set 

$$
\{p(y \mid \theta) : \theta \in \Theta\},
$$ 

where $$p(y \mid \theta)$$ is a density function over some sample space $$Y$$, and $$\theta$$ represents a parameter in a finite-dimensional parameter space $$\Theta$$. 

In simpler terms, this is a collection of probability distributions, each associated with a specific value of the parameter $$\theta$$. When we refer to "density," it’s worth noting that this can be replaced with "mass function" if we’re dealing with discrete random variables. Similarly, integrals can be replaced with summations in such cases.

This framework is the common starting point for both **classical statistics** and **Bayesian statistics**, as it provides a structured way to think about modeling the data.


##### **Frequentist or “Classical” Statistics**

In frequentist statistics, we also work with the parametric family of densities $$\{p(y \mid \theta) : \theta \in \Theta\}$$, assuming that the true distribution $$p(y \mid \theta)$$ governs the world we observe. This means there exists some unknown parameter $$\theta \in \Theta$$ that determines the true nature of the data.

If we had direct access to this true parameter $$\theta$$, we wouldn’t need statistics at all! However, in practice, we only have a dataset, denoted as 

$$
D = \{y_1, y_2, \dots, y_n\},
$$ 

where each $$y_i$$ is sampled independently from the true distribution $$p(y \mid \theta)$$.

This brings us to the heart of statistics: **how do we make inferences about the unknown parameter $$\theta$$ using only the observed data $$D$$?**


##### **Point Estimation**

One fundamental problem in statistics is **point estimation**, where the goal is to estimate the true value of the parameter $$\theta$$ as accurately as possible. 

To do this, we use a **statistic**, denoted as $$s = s(D)$$, which is simply a function of the observed data. When this statistic is designed to estimate $$\theta$$, we call it a **point estimator**, represented as $$\hat{\theta} = \hat{\theta}(D)$$. 

A **good point estimator** is one that is both:

- **Consistent**: As the sample size $$n$$ grows larger, the estimator $$\hat{\theta}_n$$ converges to the true parameter $$\theta$$.
- **Efficient**: The estimator $$\hat{\theta}_n$$ extracts the maximum amount of information about $$\theta$$ from the data, achieving the best possible accuracy for a given sample size.

One of the most popular methods for point estimation is the **maximum likelihood estimator (MLE)**, which we’ll now explore through a concrete example.


##### **Example: Coin Flipping and Maximum Likelihood Estimation**

Let’s consider the simple yet illustrative problem of estimating the probability of a coin landing on heads. 

**Parametric Family**

Here, the parametric family of mass functions is given by:

$$
p(\text{Heads} \mid \theta) = \theta, \quad \text{where } \theta \in \Theta = (0, 1).
$$

The parameter $$\theta$$ represents the probability of the coin landing on heads. Our goal is to estimate this parameter based on observed data.

**Data and Likelihood Function**

Suppose we observe the outcomes of $$n$$ independent coin flips, represented as:

$$
D = (\text{H, H, T, T, T, T, T, H, \dots, T}),
$$

where $$n_h$$ is the number of heads, and $$n_t$$ is the number of tails. Since each flip is independent, the likelihood function for the observed data is:

$$
L_D(\theta) = p(D \mid \theta) = \theta^{n_h} (1 - \theta)^{n_t}.
$$

**Log-Likelihood and Optimization**

Rather than working directly with the likelihood function, which involves products and can become cumbersome, we typically maximize the **log-likelihood function** for computational simplicity. The log-likelihood is:

$$
\log L_D(\theta) = n_h \log \theta + n_t \log (1 - \theta).
$$

The **maximum likelihood estimate (MLE)** of $$\theta$$ is the value that maximizes this log-likelihood:

$$
\hat{\theta}_{\text{MLE}} = \underset{\theta \in \Theta}{\text{argmax}} \, \log L_D(\theta).
$$

**Derivation of the MLE**

To find the MLE, we compute the derivative of the log-likelihood with respect to $$\theta$$, set it to zero, and solve for $$\theta$$:

$$
\frac{\partial}{\partial \theta} \big[ n_h \log \theta + n_t \log (1 - \theta) \big] = \frac{n_h}{\theta} - \frac{n_t}{1 - \theta}.
$$

Setting this derivative to zero:

$$
\frac{n_h}{\theta} = \frac{n_t}{1 - \theta}.
$$

Simplifying this equation gives:

$$
\theta = \frac{n_h}{n_h + n_t}.
$$

Thus, the MLE for $$\theta$$ is:

$$
\hat{\theta}_{\text{MLE}} = \frac{n_h}{n_h + n_t}.
$$

**Intuition Behind the MLE**

The result makes intuitive sense: the MLE simply calculates the proportion of heads observed in the data. It uses the empirical frequency as the best estimate of the true probability of heads, given the observed outcomes.

---

#### **What About Bayesian Methods?**

While frequentist approaches like MLE provide a single "best" estimate for $$\theta$$, Bayesian methods take a different perspective. Instead of finding a point estimate, Bayesian inference quantifies uncertainty about $$\theta$$ using probability distributions. This leads to the concepts of **prior distributions** and **posterior inference**, which we will explore in the next part of this series.

Stay tuned as we dive into the Bayesian paradigm and uncover how it complements and contrasts with frequentist methods!

---

#### **Bayesian Statistics: An Introduction**

In the frequentist framework, the goal is to estimate the true parameter $$\theta$$ using the observed data. However, **Bayesian statistics** takes a fundamentally different approach by introducing an important concept: the **prior distribution**. This addition allows us to explicitly incorporate prior beliefs about the parameter into our analysis and update them rationally as we observe new data.


---

#### **The Prior Distribution: Reflecting Prior Beliefs**

A **prior distribution**, denoted as $$p(\theta)$$, is a probability distribution over the parameter space $$\Theta$$. It represents our belief about the value of $$\theta$$ **before** observing any data. For instance, if we believe that $$\theta$$ is more likely to lie in a specific range, we can encode this belief directly into the prior.


##### **A Bayesian Model: Combining Prior and Data**

A **[parametric] Bayesian model** is constructed from two key components:

1. A **parametric family of densities** $$\{p(D \mid \theta) : \theta \in \Theta\}$$ that models the likelihood of the observed data $$D$$ given $$\theta$$.
2. A **prior distribution** $$p(\theta)$$ on the parameter space $$\Theta$$.

These two components combine to form a **joint density** over $$\theta$$ and $$D$$:

$$
p(D, \theta) = p(D \mid \theta) p(\theta).
$$

This joint density encapsulates both the likelihood of the data and our prior beliefs about the parameter.


##### **Posterior Distribution: Updating Beliefs**

The real power of Bayesian statistics lies in the ability to **update prior beliefs** after observing data. This is achieved through the **posterior distribution**, denoted as $$p(\theta \mid D)$$. 

- The **prior distribution** $$p(\theta)$$ captures our initial beliefs about $$\theta$$.
- The **posterior distribution** $$p(\theta \mid D)$$ reflects our updated beliefs after observing the data $$D$$.

By applying **Bayes’ rule**, we can express the posterior distribution as:

$$
p(\theta \mid D) = \frac{p(D \mid \theta) p(\theta)}{p(D)},
$$

where:

- $$p(D \mid \theta)$$ is the **likelihood**, capturing how well $$\theta$$ explains the observed data.
- $$p(\theta)$$ is the **prior**, encoding our initial beliefs about $$\theta$$.
- $$p(D)$$ is a normalizing constant, ensuring the posterior integrates to 1.


##### **Simplifying the Posterior**

When analyzing the posterior distribution, we often focus on terms that depend on $$\theta$$. Dropping constant factors that are independent of $$\theta$$, we write:

$$
p(\theta \mid D) \propto p(D \mid \theta) \cdot p(\theta),
$$

where $$\propto$$ denotes proportionality.

In practice, this allows us to analyze and work with the posterior distribution more efficiently. For instance, the **maximum a posteriori (MAP) estimate** of $$\theta$$ is given by:

$$
\hat{\theta}_{\text{MAP}} = \underset{\theta \in \Theta}{\text{argmax}} \, p(\theta \mid D).
$$

---

#### **Example: Bayesian Coin Flipping**

Let’s revisit the coin-flipping example, but this time from a Bayesian perspective. We start with the parametric family of mass functions:

$$
p(\text{Heads} \mid \theta) = \theta, \quad \text{where } \theta \in \Theta = (0, 1).
$$

To complete our Bayesian model, we also need to specify a **prior distribution** over $$\theta$$. One common choice is the **Beta distribution**, which is particularly convenient for this problem.

##### **Beta Prior Distribution**

The Beta distribution, denoted as $$\text{Beta}(\alpha, \beta)$$, is a flexible family of distributions defined on the interval $$(0, 1)$$. Its density function is:

$$
p(\theta) \propto \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}.
$$

For our coin-flipping example, we can use:

$$
p(\theta) \propto \theta^{h - 1} (1 - \theta)^{t - 1},
$$

where $$h$$ and $$t$$ represent our prior "counts" of heads and tails, respectively.

The **mean** of the Beta distribution is:

$$
\mathbb{E}[\theta] = \frac{h}{h + t},
$$

and its **mode** (for $$h, t > 1$$) is:

$$
\text{Mode} = \frac{h - 1}{h + t - 2}.
$$

---

#### **Posterior Distribution: Updating with Data**

After observing data $$D = (\text{H, H, T, T, T, H, ...})$$, where $$n_h$$ is the number of heads and $$n_t$$ is the number of tails, we combine the **prior** and **likelihood** to obtain the **posterior distribution**. 

##### **Likelihood Function**

The likelihood function, based on the observed data, is:

$$
L(\theta) = p(D \mid \theta) = \theta^{n_h} (1 - \theta)^{n_t}.
$$

##### **Posterior Density**

Combining the prior and likelihood, the posterior density is:

$$
p(\theta \mid D) \propto p(\theta) \cdot L(\theta),
$$

which simplifies to:

$$
p(\theta \mid D) \propto \theta^{h - 1} (1 - \theta)^{t - 1} \cdot \theta^{n_h} (1 - \theta)^{n_t}.
$$

Simplifying further, we get:

$$
p(\theta \mid D) \propto \theta^{h - 1 + n_h} (1 - \theta)^{t - 1 + n_t}.
$$

This posterior distribution is also a Beta distribution:

$$
\theta \mid D \sim \text{Beta}(h + n_h, t + n_t).
$$

---

#### **Interpreting the Posterior**

The posterior distribution shows how our prior beliefs are updated by the observed data:

- The prior $$\text{Beta}(h, t)$$ initializes our counts with $$h$$ heads and $$t$$ tails.
- The posterior $$\text{Beta}(h + n_h, t + n_t)$$ updates these counts by adding the observed $$n_h$$ heads and $$n_t$$ tails.

For example, if our prior belief was $$\text{Beta}(2, 2)$$ (a uniform prior), and we observed $$n_h = 3$$ heads and $$n_t = 1$$ tails, the posterior would be:

$$
\text{Beta}(2 + 3, 2 + 1) = \text{Beta}(5, 3).
$$

This reflects our updated belief about the probability of heads after observing the data.

---

In the next part of this series, we’ll explore more advanced Bayesian concepts and their applications in machine learning. Stay tuned as we continue to build intuition and delve deeper into Bayesian inference!


---

#### **Sidebar: Conjugate Priors**

In Bayesian analysis, **conjugate priors** simplify computations and allow for elegant posterior updates. A prior distribution is said to be **conjugate** to a likelihood model when the posterior distribution belongs to the same family as the prior. 

### Definition: Conjugate Priors
Let $$\pi$$ represent a family of prior distributions on the parameter space $$\Theta$$, and let $$P$$ be a parametric family of likelihoods with the same parameter space $$\Theta$$. The family of priors $$\pi$$ is **conjugate** to the parametric model $$P$$ if, for any prior in $$\pi$$, the posterior distribution is also in $$\pi$$.  

For example, the **Beta distribution** is conjugate to the **Bernoulli model** (coin-flipping), which makes it especially useful for Bayesian inference in such cases.

---

#### **Concrete Example: Bayesian Coin Flipping Revisited** 

Let’s work through a specific example to illustrate these ideas.  

##### **Problem Setup**  
We’re flipping a potentially biased coin, where the probability of getting heads is $$\theta$$. The setup includes:  

- **Likelihood Model**: $$p(\text{Heads} \mid \theta) = \theta, \quad \text{with } \theta \in [0, 1].$$  
- **Prior Distribution**: $$\theta \sim \text{Beta}(2, 2).$$  

This prior reflects an initial belief that the coin is approximately fair (a uniform prior over $$(0,1)$$).  

##### **Observing the Data**  
We collect data from a series of flips:  
$$D = \{\text{H, H, T, T, T, T, T, H, ...}\}.$$  
From this dataset, we observe:  
- **Number of Heads**: $$n_h = 75$$  
- **Number of Tails**: $$n_t = 60.$$  

Using the **MLE**, the estimate for $$\theta$$ is:  
$$
\hat{\theta}_{\text{MLE}} = \frac{n_h}{n_h + n_t} = \frac{75}{75 + 60} \approx 0.556.
$$  

##### **Posterior Distribution**  
The posterior distribution for $$\theta$$ is:  
$$
\theta \mid D \sim \text{Beta}(h + n_h, t + n_t) = \text{Beta}(2 + 75, 2 + 60) = \text{Beta}(77, 62).
$$  

This posterior reflects our updated belief about $$\theta$$ after observing the data.  

---

#### **Bayesian Point Estimates**  

Once we have the posterior distribution, we might want a single **point estimate** for $$\theta$$. Bayesian statistics offers several options for this:  

1. **Posterior Mean**:  
   $$\hat{\theta} = \mathbb{E}[\theta \mid D],$$  
   which is the mean of the posterior distribution.  

2. **Maximum a Posteriori (MAP) Estimate**:  
   $$\hat{\theta} = \underset{\theta}{\text{argmax}} \, p(\theta \mid D),$$  
   which corresponds to the **mode** of the posterior distribution.  

For a **Beta posterior** $$\text{Beta}(\alpha, \beta)$$:  
- The **mean** is:  
  $$\mathbb{E}[\theta] = \frac{\alpha}{\alpha + \beta}.$$  
- The **mode** (for $$\alpha, \beta > 1$$) is:  
  $$\text{Mode} = \frac{\alpha - 1}{\alpha + \beta - 2}.$$  

Using the posterior $$\text{Beta}(77, 62)$$ from our example:  
- **Posterior Mean**:  
  $$\mathbb{E}[\theta] = \frac{77}{77 + 62} \approx 0.554.$$  
- **Posterior Mode (MAP)**:  
  $$\text{Mode} = \frac{77 - 1}{77 + 62 - 2} \approx 0.553.$$  

---

#### **Beyond Point Estimates: What Else Can We Do with the Posterior?** 

Bayesian analysis offers more than just point estimates. Here are a few additional insights we can extract:  

1. **Visualizing the Posterior**:  
   Plot the posterior distribution $$p(\theta \mid D)$$ to show uncertainty around $$\theta$$. This can help communicate results to stakeholders.  

2. **Credible Intervals**:  
   A **credible interval** is the Bayesian analog of a confidence interval. For example, a 95% credible interval $$[a, b]$$ satisfies:  
   $$P(\theta \in [a, b] \mid D) \geq 0.95.$$  

3. **Bayesian Decision Theory**:  
   If making a decision based on $$\theta$$, we can:  
   - Define a **loss function** to quantify the cost of different actions.  
   - Choose the action that minimizes the **expected risk** with respect to the posterior.  

### Example: 95% Credible Interval  
For our posterior $$\text{Beta}(77, 62)$$, we can calculate a 95% credible interval numerically. This interval reflects the range where we believe $$\theta$$ is most likely to lie, given the observed data.

---

##### **Summary**  

In this series, we explored the essence of Bayesian statistics, focusing on how priors, likelihoods, and posteriors interact to update our beliefs. Using the coin-flipping example, we demonstrated key Bayesian tools like the Beta distribution, MAP estimation, and credible intervals. Bayesian methods not only provide point estimates but also offer a rich framework for quantifying and visualizing uncertainty.  

In the next post, we’ll dive deeper into how Bayesian principles apply to machine learning, exploring models like Bayesian linear regression and Gaussian processes. Stay tuned!

---

Add references that explains the Bayesian Intuition well.