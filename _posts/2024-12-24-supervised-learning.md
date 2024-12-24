---
layout: post
title: Understanding the Supervised Learning Setup
date: 2024-12-24 11:35:00-0400
featured: false
description: An in-depth exploration of the supervised learning setup, covering key concepts like prediction functions, loss functions, risk evaluation, and the Bayes optimal predictor.
tags: ML
categories: sample-posts
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

Supervised learning is a cornerstone of machine learning, enabling systems to learn from labeled data to make predictions or decisions. In this post, we will explore the various components and formalizations of supervised learning to build a solid foundation.

#### **Goals in Supervised Learning**

In supervised learning problems, we typically aim to:

- **Make a decision:** For instance, determining whether to move an email to a spam folder.
- **Take an action:** As in self-driving cars, deciding when to make a right turn.
- **Reject a hypothesis:** Such as testing the hypothesis that $$\theta = 0$$ in classical statistics.
- **Produce some output:** Examples include identifying whose face is in an image, translating a Japanese sentence into Hindi, or predicting the location of a storm an hour into the future.

Each of these goals involves predicting or generating some form of output based on given inputs.

##### **Labels: The Key to Supervised Learning**

Supervised learning involves pairing inputs with **labels**, which serve as the ground truth. Examples of labels include:

- Whether or not a picture contains an animal.
- The storm's location one hour after a query.
- Which, if any, of the suggested URLs were selected.

These labels allow us to evaluate the performance of our predictions systematically.

---

#### **Evaluation Criterion**

The next step in supervised learning is finding **optimal outputs** under various definitions of optimality. Some examples of evaluation criteria include:

- **Classification Accuracy:** Is the predicted class correct?
- **Exact Match:** Does the transcription exactly match the spoken words?
- **Partial Credit:** How do we account for partially correct answers (e.g., getting some words right)?
- **Prediction Distance:** How far is the storm's actual location from the predicted one?
- **Density Prediction:** How likely is the storm's actual location under the predicted distribution?

These criteria ensure that we can quantitatively measure the performance of our models.

---

#### **Typical Sequence of Events**

Supervised learning problems can often be formalized through the following sequence:

1. **Observe Input ($$x$$):** Receive an input data point.
2. **Predict Output ($$\hat{y}$$):** Use a prediction function to generate an output.
3. **Observe Label ($$y$$):** Compare the predicted output with the true label.
4. **Evaluate Output:** Assess the prediction's quality based on the label.

This sequence is at the heart of most supervised learning frameworks.

---


#### **Formalizing Supervised Learning**

###### **Prediction Function**

A **prediction function** is a mathematical function $$f: X \to Y$$ that takes an input $$x \in X$$ and produces an output $$\hat{y} \in Y$$.

###### **Loss Function**

A **loss function** evaluates the discrepancy between the predicted output $$\hat{y}$$ and the true outcome $$y$$. It quantifies the "cost" of making incorrect predictions.

##### **The Goal: Optimal Prediction**

The primary goal is to find the **optimal prediction function**. The intuition is simple: If we can evaluate how good a prediction function is, we can turn this into an optimization problem.

- The loss function $$\ell$$ evaluates a single output.
- To evaluate the prediction function as a whole, we need to formalize the concept of "average performance."

###### **Data Generating Distribution**

Assume there exists a data-generating distribution $$P_{X \times Y}$$. All input-output pairs $$(x, y)$$ are generated independently and identically distributed (i.i.d.) from this distribution.

A common objective is to have a prediction function $$f(x)$$ that performs well **on average**:

$$
\ell(f(x), y)
$$ 

is small, in some sense.

###### **Risk Definition**

The **risk** of a prediction function $$f: X \to Y$$ is defined as:

$$
R(f) = \mathbb{E}_{(x, y) \sim P_{X \times Y}} [\ell(f(x), y)].
$$

In words, this is the expected loss of $$f$$ over the data-generating distribution $$P_{X \times Y}$$. However, since we do not know $$P_{X \times Y}$$, we cannot compute this expectation directly. Instead, we estimate it.

---

#### **The Bayes Prediction Function**

**Definition**

The **Bayes prediction function** $$f^*: X \to Y$$ achieves the minimal risk among all possible functions:

$$
f^* \in \underset{f}{\text{argmin}} \ R(f),
$$

where the minimum is taken over all functions from $$X$$ to $$Y$$.

**Bayes Risk**

The risk associated with the Bayes prediction function is called the **Bayes risk**. This function is often referred to as the "target function" because it represents the best possible predictor.

---


#### **Example: Multiclass Classification**

In multiclass classification, the output space is:

$$
Y = \{1, 2, \dots, k\}.
$$

The **0-1 loss** function is defined as:

$$ \ell(\hat{y}, y) = \mathbb{1}[\hat{y} \neq y] := \begin{cases} 1 & \text{if } \hat{y} \neq y, \\ 0 & \text{otherwise.} \end{cases} $$

- Here, $$\mathbb{1}[\hat{y} \neq y]$$ is an **indicator function**. It returns a value of 1 when the condition $$\hat{y} \neq y$$ is true (i.e., the prediction is incorrect) and 0 otherwise. This signifies whether the prediction is correct or incorrect and is commonly used to measure classification errors.

The risk $$R(f)$$ under the 0-1 loss can be expanded as follows:

$$
R(f) = \mathbb{E}[\mathbb{1}[f(x) \neq y]] = \mathbb{P}(f(x) \neq y),
$$

where:

- $$\mathbb{E}[\mathbb{1}[f(x) \neq y]]$$ represents the expected value of the indicator function, which counts the proportion of incorrect predictions.
- $$\mathbb{P}(f(x) \neq y)$$ is the probability of the prediction $$f(x)$$ being different from the true label $$y$$.

Further, this can be rewritten using the decomposition of probabilities:

$$
R(f) = 0 \cdot \mathbb{P}(f(x) = y) + 1 \cdot \mathbb{P}(f(x) \neq y),
$$

which simplifies back to:

$$
R(f) = \mathbb{P}(f(x) \neq y).
$$

**Explanation**

- The term $$0 \cdot \mathbb{P}(f(x) = y)$$ accounts for the cases where the prediction is correct (loss is 0).
- The term $$1 \cdot \mathbb{P}(f(x) \neq y)$$ accounts for the cases where the prediction is incorrect (loss is 1).

Thus, $$R(f)$$ directly measures the **misclassification error rate**, which is the probability of the model making an incorrect prediction.

**Bayes Prediction Function**

The Bayes prediction function returns the most likely class:

$$
f^*(x) \in \underset{1 \leq c \leq k}{\text{argmax}} \ P(y = c \mid x).
$$

---


#### **Estimating Risk**

We cannot compute the true risk $$R(f) = \mathbb{E}[\ell(f(x), y)]$$ because the true distribution $$P_{X \times Y}$$ is unknown. However, we can estimate it.

Assume we have sample data:

$$
D_n = \{(x_1, y_1), \dots, (x_n, y_n)\},
$$

where the samples are i.i.d. from $$P_{X \times Y}$$. By the strong law of large numbers, the empirical average of losses converges to the expected value. If $$z_1, . . . , z_n$$ are i.i.d. with expected value $$\mathbb{E}[z]$$, then

$$
\lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^n z_i = \mathbb{E}[z],
$$

with probability 1.

This leads us to the concept of **empirical risk** and its minimization, which we will explore in the next post.

---

In the next blog, we will dive into **empirical risk minimization** and how it helps solve supervised learning problems effectively. 

**Stay tuned!**
