---
layout: post
title: Structured Perceptron & Structured SVM
date: 2025-04-16 15:07:00-0400
featured: false
description: Understanding how Structured Perceptron and Structured SVM learn to predict structured outputs with interdependent components.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

In the [previous post](http://localhost:8080/blog/2025/structured-prediction/), we explored how structured prediction works under the hood—we defined a compatibility function $$h(x, y) = \langle w, \Psi(x, y) \rangle$$, designed rich feature representations using Unary and Markov features, and saw how the overall score of an output sequence decomposes into local parts.

We also discussed how this score guides prediction by choosing the most compatible output structure:

$$
f(x) = \arg\max_{y \in Y(x)} h(x, y)
$$

But how do we actually **learn** the weight vector $$w$$ that makes good predictions?

In this post, we’ll walk through two popular algorithms for learning structured models:

- The **Structured Perceptron**, which extends the classic perceptron to structured outputs
- The **Structured SVM**, which builds in **margins** and **regularization** for better generalization

Let’s dive in!

##### **Learning with Structured Perceptron**

To learn the weight vector $$w$$ that scores correct outputs higher than incorrect ones, we can use the **structured perceptron algorithm**.

It works just like the multiclass perceptron, except the prediction is over a **structured output space**:

---

**Structured Perceptron Algorithm**

1. **Initialize** the weights:
   $$
   w \leftarrow 0
   $$

2. **For each training example** $$(x, y)$$:
   - Predict the best structure under the current model:
     $$
     \hat{y} = \arg\max_{y' \in Y(x)} \langle w, \Psi(x, y') \rangle
     $$

   - If the prediction is incorrect $$(\hat{y} \ne y)$$:
     - **Update** the weight vector:
       $$
       w \leftarrow w + \Psi(x, y) - \Psi(x, \hat{y})
       $$


This update **encourages the correct structure** by increasing its score and **penalizes the incorrect one** by decreasing its score. This is identical to multiclass perceptron, except that the prediction $$\hat{y}$$ comes from a structured space.


---

Up to this point, we've seen how to score structured outputs and how to train with the structured perceptron. But the perceptron only updates on mistakes and doesn’t consider ***"how wrong"*** a prediction is.

So, what if we want a **more principled way to penalize incorrect outputs** based on how different they are from the correct one?

This brings us to **structured hinge loss**.


##### **Generalized Hinge Loss**

In structured prediction, we want the correct output to **not only score highest**, but to **beat all incorrect outputs by a margin**.

This leads to the generalized hinge loss:

$$
\ell_{\text{hinge}}(x, y) = \max_{y' \in Y(x)} \left[ \Delta(y, y') + \langle w, \Psi(x, y') - \Psi(x, y) \rangle \right]
$$

Let’s break it down:

- $$\Psi(x, y)$$ is the feature vector for the true output.
- $$\Psi(x, y')$$ is the feature vector for a wrong prediction.
- $$\Delta(y, y')$$ is the **loss function** that tells us *how bad* the prediction $$y'$$ is compared to the ground truth $$y$$.

A common choice for $$\Delta$$ is the **Hamming loss** (i.e., how many labels are incorrect):

$$
\Delta(y, y') = \frac{1}{L} \sum_{i=1}^L 1[y_i \ne y'_i]
$$

This loss forces the model to **separate the true output from the rest**, with a margin proportional to how different they are.

**Picture it this way:**

Imagine you're a teacher grading structured answers—say, full sentences submitted by students.

You don’t just care whether a sentence is right or wrong—you also care **how wrong** a student’s answer is. If a student writes something that’s close to the correct answer, you might give partial credit. But if their answer is completely off, you'd deduct more points.

This is exactly what structured hinge loss does.

- It ensures that the correct output **not only wins**, but wins **by enough**—with a *margin* that reflects how different the incorrect output is.
- If an incorrect output $$y'$$ is very different from the ground truth $$y$$ (i.e., high $$\Delta(y, y')$$), then the model is penalized more if it scores $$y'$$ too closely to $$y$$.

**Example: POS Tagging with Margin-Based Loss**

Suppose you have the input sentence:  **"He runs fast"**

The correct POS tags are:  

$$
y = [\text{Pronoun}, \text{Verb}, \text{Adverb}]
$$

Now, imagine two possible incorrect predictions:

- $$y'_1 = [\text{Pronoun}, \text{Noun}, \text{Adverb}]$$ — only **one** mistake
- $$y'_2 = [\text{Noun}, \text{Noun}, \text{Noun}]$$ — **three** mistakes

The Hamming losses are:

- $$\Delta(y, y'_1) = \frac{1}{3}$$
- $$\Delta(y, y'_2) = 1$$

According to the hinge loss, $$y'_2$$ should be separated by a **larger margin** from the correct output than $$y'_1$$. That is, the model must not only prefer the true output, but **strongly penalize** very wrong outputs.


##### **Structured SVM Objective**

We now define a learning objective that uses this hinge loss, along with regularization:

$$
\min_{w} \frac{1}{2} \|w\|^2 + C \sum_{(x, y) \in D} \ell_{\text{hinge}}(x, y)
$$

Where:

- The first term $$\frac{1}{2} \|w\|^2$$ controls model complexity (regularization).
- The second term penalizes incorrect predictions.
- $$C$$ is a hyperparameter that trades off margin size vs training error.

##### **How Do We Optimize This?**

Optimizing the structured SVM objective might seem tricky because the hinge loss involves a **max over all possible outputs**:

$$
\ell_{\text{hinge}}(x, y) = \max_{y' \in Y(x)} \left[ \Delta(y, y') + \langle w, \Psi(x, y') - \Psi(x, y) \rangle \right]
$$

But here's the clever trick:

We don’t need to check **all** possible $$y'$$—we only need to find the **worst violator**; The structure $$y'$$ that scores too close to or even higher than the true output $$y$$—**after accounting for how wrong it is**.

This process is called **loss-augmented inference**.

##### **Understanding Loss-Augmented Inference**

To optimize the structured SVM objective, we need to minimize the hinge loss:

$$
\ell_{\text{hinge}}(x, y) = \max_{y' \in Y(x)} \left[ \Delta(y, y') + \langle w, \Psi(x, y') - \Psi(x, y) \rangle \right]
$$

At a high level, this loss tries to ensure:

> "The score of the **true output** $$y$$ is higher than that of **every incorrect output** $$y'$$, by at least how wrong $$y'$$ is."

This encourages **margin-based separation** between $$y$$ and each $$y'$$.

But how do we compute or optimize this in practice?

**Intuition Behind Loss-Augmented Inference**

Think of the model as making predictions based on scores:

- $$\langle w, \Psi(x, y) \rangle$$ — score for the **true output**
- $$\langle w, \Psi(x, y') \rangle$$ — score for a **candidate output**

The structured SVM doesn’t just care about which $$y'$$ scores highest.  

It asks: *"Which wrong output $$y'$$ is both very wrong **and** scores too well?"*

That’s where **loss-augmented inference** comes in:

$$
y' = \arg\max_{y' \in Y(x)} \left[ \Delta(y, y') + \langle w, \Psi(x, y') \rangle \right]
$$

- The model score $$\langle w, \Psi(x, y') \rangle$$ captures how likely the model thinks $$y'$$ is.
- The task loss $$\Delta(y, y')$$ captures how bad that $$y'$$ is in the real world.

By combining them, we search for the **most offending structure**—the one that violates the margin the most.


##### **The Update Step**

Once we find this $$y'$$, we update the weights:

$$
w \leftarrow w + \eta \cdot \left( \Psi(x, y) - \Psi(x, y') \right)
$$

- This increases the score for the correct output $$y$$
- And decreases the score for the worst violator $$y'$$

It’s almost like the perceptron, **but more aware of the "danger level" of the mistake**.

**Analogy: Hiring Candidates**

Imagine you're hiring someone (the structured model) to assign tags to words in a sentence. You ask them to tag a sentence and also explain **why** they think the tags are right (that’s their score).

Now, suppose they make a mistake. You don’t just say “wrong!”—you also ask:

> “How bad is this mistake? Did you call a verb a noun? Or a verb an adjective?”

If the mistake is **very wrong** and the candidate seems **very confident** (high score), that’s a **serious violation**.

So you correct them in a way that says:

> "This bad answer was **too confident**. Next time, lower your score for this kind of mistake."

That’s what loss-augmented inference does: it focuses on **mistakes the model is confident about but shouldn’t be**.

To close things out, here’s a clear comparison between Structured Perceptron and Structured SVM to highlight their key differences.

##### **Structured Perceptron vs Structured SVM**

---

| **Aspect**              | **Structured Perceptron**           | **Structured SVM**                          |
|-------------------------|--------------------------------------|---------------------------------------------|
| **Loss Function**       | Zero-one (mistake-driven)            | Hinge loss with structured margin            |
| **Optimization**        | Perceptron-style updates             | Convex optimization                          |
| **Margin**              | No explicit margin                   | Enforces margin via structured hinge loss    |
| **Regularization**      | None                                 | $$\ell_2$$ regularization                    |
| **Stability**           | Less stable                          | More stable and generalizes better           |

---

##### **Summary**

- Structured prediction is essential when outputs are **interdependent**—such as sequences or trees.
- We define a **compatibility score** $$h(x, y)$$ to evaluate how well an output $$y$$ matches input $$x$$.
- The score decomposes using **local features** like unary and Markov features, enabling efficient learning.
- **Structured Perceptron** uses mistake-driven updates, like its multiclass counterpart.
- **Structured SVM** introduces **margins and hinge loss**, providing a more robust and generalizable model.

---

In upcoming posts, we’ll dive into **Conditional Random Fields (CRFs)** and other practical models for sequence labeling and NLP tasks.

Stay tuned!



