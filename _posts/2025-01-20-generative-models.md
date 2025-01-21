---
layout: post
title: An Introduction to Generative Models - Naive Bayes for Binary Features
date: 2025-01-20 16:29:00-0400
featured: false
description: Learn the fundamentals of Naive Bayes, from its conditional independence assumption to the maximum likelihood estimation (MLE) of parameters, using a binary feature example.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

Generative models represent a powerful class of machine learning techniques. Unlike methods that directly map inputs $$ x $$ to outputs $$ y $$, such as generalized linear models or perceptrons, generative models take a broader approach. They aim to model the **joint distribution** $$ p(x, y; \theta) $$, which allows us to capture the underlying relationships between inputs and outputs in a holistic manner.

#### **Generalized Linear Models vs. Generative Models**

To recall, generalized linear models focus on the conditional distribution $$ p(y \mid x; \theta) $$. By contrast, generative models model $$ p(x, y; \theta) $$. Once we have the joint distribution, we can predict labels for new data by leveraging the following rule:

$$
\hat{y} = \arg\max_{y \in \mathcal{Y}} p(x, y; \theta)
$$

This prediction process connects naturally to conditional distributions. Using **Bayes' Rule**, we can rewrite $$ p(y \mid x) $$ as:

$$
p(y \mid x) = \frac{p(x \mid y) p(y)}{p(x)}
$$

In practice, we often bypass computing $$ p(x) $$, as it is independent of $$ y $$. Instead, predictions simplify to:

$$
\hat{y} = \arg\max_{y} p(y \mid x) = \arg\max_{y} p(x \mid y) p(y)
$$


With this foundation, let us explore one of the most straightforward and widely used generative models: **Naive Bayes (NB)**.

If you're unable to follow the above formulation, here's a quick refresher on Bayes' Rule to help you out.

Bayes' Rule relates conditional probabilities to joint and marginal probabilities. It can be expressed as:

$$
p(y \mid x) = \frac{p(x, y)}{p(x)} = \frac{p(x \mid y) p(y)}{p(x)},
$$

where:

- $$ p(y \mid x) $$: Posterior probability of $$ y $$ given $$ x $$,
- $$ p(x, y) $$: Joint probability of $$ x $$ and $$ y $$,
- $$ p(x \mid y) $$: Likelihood of $$ x $$ given $$ y $$,
- $$ p(y) $$: Prior probability of $$ y $$,
- $$ p(x) $$: Marginal probability of $$ x $$, which ensures proper normalization.


---

#### **Naive Bayes: A Simple and Effective Generative Model**

To understand Naive Bayes, consider a simple yet practical problem: binary text classification. Imagine we want to classify a document as either a **fake review** or a **genuine review**. This setup offers a clear context to explore the mechanics of generative modeling.

##### **Representing Documents as Features**

To make this task computationally feasible, we use a **bag-of-words representation**. A document is expressed as a binary vector $$ x $$, where:

$$
x = [x_1, x_2, \dots, x_d].
$$

Here, $$ d $$ represents the vocabulary size, and each $$ x_i $$ indicates whether the $$ i $$-th word in the vocabulary exists in the document ($$ x_i = 1 $$) or not ($$ x_i = 0 $$).

##### **Modeling the Joint Probability of Documents and Labels**

For a document $$ x $$ with label $$ y $$, the joint probability $$ p(x, y) $$ can be expressed using the **chain rule of probability**:

$$
p(x \mid y) = p(x_1, x_2, \dots, x_d \mid y) = p(x_1 \mid y) p(x_2 \mid y, x_1) \cdots p(x_d \mid y, x_{d-1}, \dots, x_1).
$$

$$
p(x \mid y) = \prod_{i=1}^d p(x_i \mid y, x_{<i}),
$$

However, modeling the dependencies between features ($$ x_1, x_2, \dots, x_d $$) becomes intractable as the number of features grows and hard to estimate. This is where Naive Bayes introduces its defining assumption.

##### **The Naive Bayes Assumption**

Naive Bayes simplifies the problem by assuming that **features are conditionally independent given the label $$ y $$**. Mathematically, this means:

$$
p(x \mid y) = \prod_{i=1}^d p(x_i \mid y)
$$

This assumption significantly reduces computational complexity while often delivering excellent results in practice. While the assumption of conditional independence may not hold in all cases, it is surprisingly effective in many real-world applications.

---

#### **Parameterizing the Naive Bayes Model**

To make predictions, we need to parameterize the probabilities $$ p(x_i \mid y) $$ and $$ p(y) $$.

***Why?*** Parameterizing these distributions allows us to learn the necessary values (e.g., $$ \theta $$) from data in a structured way.


##### **Binary Features**

For simplicity, let us assume the features $$ x_i $$ are binary ($$ x_i \in \{0, 1\} $$). We model $$ p(x_i \mid y) $$ as Bernoulli distributions:

$$
p(x_i = 1 \mid y = 1) = \theta_{i,1}, \quad p(x_i = 1 \mid y = 0) = \theta_{i,0}
$$

Similarly, the label distribution is modeled as:

$$
p(y = 1) = \theta_0
$$


***How do we arrive at these definitions?***
These definitions arise from the following assumptions and modeling principles:

1. **Binary Nature of Features**: Since the features $$ x_i $$ are binary ($$ x_i \in \{0, 1\} $$), we need a probability distribution that models the likelihood of binary outcomes. The Bernoulli distribution is a natural choice for this.
2. **Parameterization with Bernoulli Distributions**:
   - For $$ p(x_i \mid y) $$, the Bernoulli distribution models the probability that $$ x_i = 1 $$ for each possible value of $$ y $$.  
   - We introduce parameters $$ \theta_{i,1} $$ and $$ \theta_{i,0} $$, which represent the probability of $$ x_i = 1 $$ given $$ y = 1 $$ and $$ y = 0 $$, respectively.
3. **Label Distribution $$ p(y) $$**:  
   - The label $$ y $$ is also binary ($$ y \in \{0, 1\} $$), so we model $$ p(y) $$ using a Bernoulli distribution with a single parameter $$ \theta_0 $$, where $$ \theta_0 = p(y = 1) $$.  
   - This parameter reflects the prior probability of the positive class.
4. **Learning from Data**: These parameters ($$ \theta_{i,1}, \theta_{i,0}, \theta_0 $$) are learned from data using methods like Maximum Likelihood Estimation (MLE), ensuring that the model reflects the observed distribution of features and labels in the dataset.

Thus, the definitions provide a straightforward and interpretable way to model binary features and labels within the Naive Bayes framework.



With these definitions, the joint probability $$ p(x, y) $$ can be written as (**with NB assumption**):

$$
p(x, y) = p(y) \prod_{i=1}^d p(x_i \mid y)
$$

Substituting the probabilities for binary features:

$$
p(x, y) = p(y) \prod_{i=1}^d \theta_{i,y}{\mathbb{I}\{x_i = 1\}} + (1 - \theta_{i,y}){\mathbb{I}\{x_i = 0\}}
$$

Here, $$ \mathbb{I}\{\text{condition}\} $$ is an indicator function that evaluates to 1 if the condition is true and 0 otherwise.

***How to intuitively understand this equation?***
This equation represents the joint probability $$ p(x, y) $$ by combining the prior probability $$ p(y) $$ with the product of the individual probabilities $$ p(x_i \mid y) $$ for each feature $$ x_i $$. Here:

1. For each feature $$ x_i $$, the term $$ \mathbb{I}\{x_i = 1\} $$ ensures that the corresponding parameter $$ \theta_{i,y} $$ is used if $$ x_i = 1 $$, while $$ \mathbb{I}\{x_i = 0\} $$ ensures that $$ (1 - \theta_{i,y}) $$ is used if $$ x_i = 0 $$.
2. The product $$ \prod_{i=1}^d $$ combines the contributions of all features under the Naive Bayes assumption of conditional independence.
3. Finally, multiplying by $$ p(y) $$ incorporates the prior belief about the label $$ y $$, providing the full joint distribution $$ p(x, y) $$.

By this decomposition, we can efficiently compute $$ p(x, y) $$ for classification tasks.

---


#### **Learning Parameters with Maximum Likelihood Estimation (MLE)**

The parameters $$ \theta $$ of the Naive Bayes model are learned by maximizing the likelihood of the observed data. Given a dataset of $$ N $$ labeled examples $$ \{(x^{(n)}, y^{(n)})\}_{n=1}^N $$, the likelihood  of the data is:

$$
\prod_{n=1}^N p_\theta(x^{(n)}, y^{(n)})
$$

Taking the logarithm of the likelihood to simplify optimization, we obtain the log-likelihood:

$$
\ell(\theta) = \sum_{n=1}^N \log p_\theta(x^{(n)}, y^{(n)})
$$

For binary features, substituting the joint probability $$ p_\theta(x, y) $$ (as defined earlier) gives:

$$
\ell(\theta) = \sum_{n=1}^N \left[ \sum_{i=1}^d \log ( \mathbb{I}\{x_i^{(n)} = 1\} \theta_{i,y^{(n)}} + \mathbb{I}\{x_i^{(n)} = 0\} (1 - \theta_{i,y^{(n)}}) ) \right] + \log p_\theta(y^{(n)})
$$

Focusing on a specific feature $$ x_j $$ and label $$ y = 1 $$, the relevant portion of the log-likelihood is:

$$
\ell(\theta) = \sum_{n=1}^N \log \left[ \mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 1\} \theta_{j,1} + \mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 0\} (1 - \theta_{j,1}) \right] \tag{1}
$$

**Step 1: Derivative of the Log-Likelihood**

Taking the derivative of the log-likelihood with respect to $$ \theta_{j,1} $$:

$$
\frac{\partial \ell}{\partial \theta_{j,1}} = \sum_{n=1}^N \left[ \frac{\mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 1\}}{\theta_{j,1}} - \frac{\mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 0\}}{1 - \theta_{j,1}} \right] \tag{2}
$$

**Did you follow the derivative?** You might be wondering how derivative $$ \log(a + b) $$ can be written as $$ \frac{1}{a} + \frac{1}{b} $$, right? If so, that’s a completely valid question — I had the same thought myself. Here’s the explanation.


The transition from equation (1) to equation (2) involves taking the derivative of the log-likelihood with respect to $$ \theta_{j,1} $$. Let’s break it down:

$$
\frac{\partial}{\partial \theta_{j,1}} \ell = \frac{\partial}{\partial \theta_{j,1}} \sum_{n=1}^{N} \left[ \log \left( \theta_{j,1} I\{ x_j(n) = 1 \} + (1 - \theta_{j,1}) I\{ x_j(n) = 0 \} \right) \right]
$$

Here, the derivative is applied to the logarithm term. Using the chain rule, we first compute the derivative of the logarithm, which is:

$$
\frac{\partial}{\partial \theta_{j,1}} \log(f(\theta_{j,1})) = \frac{1}{f(\theta_{j,1})} \cdot \frac{\partial f(\theta_{j,1})}{\partial \theta_{j,1}},
$$

where 

$$
f(\theta_{j,1}) = \theta_{j,1} I\{ x_j(n) = 1 \} + (1 - \theta_{j,1}) I\{ x_j(n) = 0 \}.
$$


For a single $$ n $$, the term inside the logarithm is:

$$
f(\theta_{j,1}) =
\begin{cases}
\theta_{j,1}, & \text{if } x_j(n) = 1, \\
1 - \theta_{j,1}, & \text{if } x_j(n) = 0.
\end{cases}
$$


The derivative of $$ f(\theta_{j,1}) $$ with respect to $$ \theta_{j,1} $$ is:

$$
\frac{\partial f(\theta_{j,1})}{\partial \theta_{j,1}} =
\begin{cases}
1, & \text{if } x_j(n) = 1, \\
-1, & \text{if } x_j(n) = 0.
\end{cases}
$$


Using the chain rule:

$$
\frac{\partial}{\partial \theta_{j,1}} \log(f(\theta_{j,1})) =
\begin{cases}
\frac{1}{\theta_{j,1}}, & \text{if } x_j(n) = 1, \\
\frac{-1}{1 - \theta_{j,1}}, & \text{if } x_j(n) = 0.
\end{cases}
$$


Applying this to the summation over $$ N $$:

$$
\frac{\partial}{\partial \theta_{j,1}} \ell = \sum_{n=1}^{N} \left[ I\{ y(n) = 1 \land x_j(n) = 1 \} \frac{1}{\theta_{j,1}} - I\{ y(n) = 1 \land x_j(n) = 0 \} \frac{1}{1 - \theta_{j,1}} \right].
$$

This is exactly what equation (48) represents, showing the decomposition of the derivative into two terms for $$ x_j(n) = 1 $$ and $$ x_j(n) = 0 $$.

The simplification uses the indicator functions $$ I $$ to select the appropriate cases, where:

$$
I\{ x_j(n) = 1 \} \quad \text{contributes} \quad \frac{1}{\theta_{j,1}},
$$

and 

$$
I\{ x_j(n) = 0 \} \quad \text{contributes} \quad \frac{1}{1 - \theta_{j,1}}.
$$

**Key Insight:**

At each step of the derivation, we are dealing with a **single term** inside the logarithm. As a result, when we take the derivative of the logarithm, the result is simply $$ \frac{1}{\text{term}} $$, where the term is either $$ \theta_{j,1} $$ or $$ 1 - \theta_{j,1} $$, depending on the value of $$ x_j(n) $$.

- If $$ x_j(n) = 1 $$, the term inside the log is $$ \theta_{j,1} $$, and its derivative is $$ \frac{1}{\theta_{j,1}} $$.
- If $$ x_j(n) = 0 $$, the term inside the log is $$ 1 - \theta_{j,1} $$, and its derivative is $$ \frac{-1}{1 - \theta_{j,1}} $$.

Thus, at each $$ n $$, we compute the derivative as $$ \frac{1}{\text{term}} $$, with the specific term depending on the value of $$ x_j(n) $$. This makes the process more straightforward as we apply it term by term across all $$ N $$ data points.

I hope that makes sense now. Let's continue.

**Step 2: Setting the Derivative to Zero**

To find the maximum likelihood estimate, we set the derivative to zero:

$$
\sum_{n=1}^N \left[ \frac{\mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 1\}}{\theta_{j,1}} \right] = \sum_{n=1}^N \left[ \frac{\mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 0\}}{1 - \theta_{j,1}} \right]
$$

Simplifying:

$$
\theta_{j,1} \sum_{n=1}^N \mathbb{I}\{y^{(n)} = 1 \} = \sum_{n=1}^N \mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 1\}
$$

The above simplification is quite straightforward. I encourage you to write it out for yourself and work through the steps. Simply **multiply both sides** by $$ \theta_{j,1}(1 - \theta_{j,1}) $$ to eliminate the denominators, then expand both sides and **isolate** $$ \theta_{j,1} $$. 

**Note**:

$$
\sum_{n=1}^N \mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 1\} + \sum_{n=1}^N \mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 0\} = \sum_{n=1}^N \mathbb{I}\{y^{(n)} = 1 \}
$$


**Step 3: Solving for $$ \theta_{j,1} $$**

Rearranging to isolate $$ \theta_{j,1} $$:

$$
\theta_{j,1} = \frac{\sum_{n=1}^N \mathbb{I}\{y^{(n)} = 1 \wedge x_j^{(n)} = 1\}}{\sum_{n=1}^N \mathbb{I}\{y^{(n)} = 1\}}
$$

**Interpretation:**
This estimate corresponds to the fraction of examples with $$ y = 1 $$ in which the $$ j $$-th feature $$ x_j $$ is active (i.e., $$ x_j = 1 $$). Intuitively, it represents the conditional probability $$ p(x_j = 1 \mid y = 1) $$ under the Naive Bayes assumption.

---

##### **Next Steps:**

1. **Compute the other $$ \theta_{i,y} $$ values**: You should calculate the parameters for all other features in the model (for example, $$ \theta_{i,0} $$ and $$ \theta_{i,1} $$ for binary features). These values represent the probability of a feature given a class, so you'll continue by maximizing the likelihood for each $$ i $$ and class $$ y $$ to estimate these parameters.
   
2. **Estimate $$ p(y) $$**: You'll also need to compute the class prior probability $$ p(y) $$, which is simply the proportion of each class in the training data. This can be done by counting how many times each class label appears and normalizing by the total number of examples.

$$
\theta_0 = \frac{\sum_{n=1}^N \mathbb{I}\{y^{(n)} = 1\}}{N}
$$

   - $$ \theta_0 $$ is the proportion of samples in the dataset that belong to the class $$ y = 1 $$.
   - It serves as the prior probability of $$ y = 1 $$.


**Substituting the Probabilities for Binary Features:**

The likelihood of the joint probability $$ p(x, y) $$ can be expressed as:

$$
p(x, y) = p(y) \prod_{i=1}^d \theta_{i,y} {\mathbb{I}\{x_i = 1\}} + (1 - \theta_{i,y}) {\mathbb{I}\{x_i = 0\}}
$$

Where $$ \mathbb{I}\{x_i = 1\} $$ is an indicator function that equals 1 when $$ x_i = 1 $$, and $$ \mathbb{I}\{x_i = 0\} $$ equals 1 when $$ x_i = 0 $$.

**Remember this equation; it's the one we started with.**

Once all parameters are estimated, you will have a fully parameterized Naive Bayes model. The model can then be used for prediction by computing the posterior probabilities for each class $$ y $$ given an input $$ x $$. For prediction, you would use the formula:

$$
\hat{y} = \arg\max_{y \in Y} p(y) \prod_{i=1}^d p(x_i \mid y)
$$

Where $$ p(x_i \mid y) $$ are the feature likelihoods, and $$ p(y) $$ is the class prior. The class with the highest posterior probability is chosen as the predicted label. This approach allows Naive Bayes to make efficient, probabilistic predictions based on the learned parameters.

**So, the fundamental idea is:**

You are estimating the parameters $$ \theta $$ for all possible features and classes. Once the parameters are learned, you apply Bayes' rule to compute the posterior probability for each class $$ y $$. Finally, you take the class that maximizes the posterior probability using:

$$
\hat{y} = \arg\max_y p(y \mid x)
$$

This gives you the predicted class $$ \hat{y} $$, based on the learned parameters from the training data.

---

##### **Recipe for Learning a Naive Bayes Model:**

1. **Choose $$ p(x_i \mid y) $$**: Select an appropriate distribution for the features, e.g., Bernoulli distribution for binary features $$ x_i $$.
2. **Choose $$ p(y) $$**: Typically, use a categorical distribution for the class labels.
3. **Estimate Parameters by MLE**: Use Maximum Likelihood Estimation (MLE) to estimate the parameters, following the same strategy used in conditional models.

##### **Where Do We Go From Here?**

So far, we have focused on modeling binary features. However, many real-world datasets involve continuous features. How can Naive Bayes be extended to handle such cases? In the next blog, we’ll explore Naive Bayes for continuous features and see how this simple model adapts to more complex data types. See you there!

<!-- Hopefully, by now, you can appreciate how efficient this process is. All of this computation can now be wrapped up in just a few lines of code, and it does all the heavy lifting for you:

```python
gnb = GaussianNB()
# Train the model
gnb.fit(X_train, y_train)
# Predict the labels for the test set
y_pred = gnb.predict(X_test)
```

This makes it clear how much work is being done under the hood with minimal code. -->