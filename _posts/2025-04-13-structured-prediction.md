---
layout: post
title: Structured Prediction and Multiclass SVM - A Continuation
date: 2025-04-13 14:05:00-0400
featured: false
description: An in-depth yet intuitive walkthrough of structured prediction and structured SVMs, covering sequence labeling, feature engineering, and margin-based learning for complex outputs.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

Structured prediction is a powerful framework used when our output space is complex and structured — such as sequences, trees, or graphs — rather than simple class labels. This post continues from multiclass SVMs to delve deeper into structured prediction, how we define and learn over complex output spaces, and how the structured hinge loss extends ideas from standard classification.

---

##### **What is Structured Prediction?**

In standard classification, we predict a single label for each input—like identifying whether an image contains a cat or a dog.

But what if our outputs aren’t that simple?  
What if the prediction itself has structure?

That’s where **structured prediction** comes in. It refers to machine learning tasks where the output is not a single label but a **structured object**—like a sequence, a tree, or even a segmentation map. These outputs have dependencies and internal organization that we want to model directly.

##### **Example 1: Part-of-Speech (POS) Tagging**

In POS tagging, we’re given a sentence and need to assign a grammatical label to each word—like "noun", "verb", or "pronoun".

Here’s an example:

$$
\begin{aligned}
x &: [\text{START}],\ \text{He},\ \text{eats},\ \text{apples} \\
y &: [\text{START}],\ \text{Pronoun},\ \text{Verb},\ \text{Noun}
\end{aligned}
$$

To formalize this:

- **Vocabulary**  
  Words we might encounter, including a special `[START]` symbol and punctuation:

  $$
  V = \text{All English words} \cup \{ \text{[START]}, \text{.} \}
  $$

- **Input space**  
  A sequence of words of any length:

  $$
  X = V^n, \quad n = 1, 2, 3, \dots
  $$

- **Label set**  
  The set of possible POS tags:

  $$
  P = \{ \text{START, Pronoun, Verb, Noun, Adjective} \}
  $$

- **Output space**  
  A sequence of POS tags of the same length as the input:

  $$
  Y = P^n, \quad n = 1, 2, 3, \dots
  $$

This is a classic case of sequence labeling, where each position in the input has a corresponding label in the output.

##### **Example 2: Action Grounding in Long-Form Videos**

Structured prediction also shines in vision tasks like **action grounding**. Here, we’re given a long video and need to segment it and assign actions like “chopping” or “frying” to different time spans.

- **Input**  
  A video frame is represented as a feature vector:

  $$
  V = \mathbb{R}^D
  $$

- **Input sequence**  
  A video is a sequence of these frame-level features:

  $$
  X = V^n
  $$

- **Label set**  
  The set of possible actions:

  $$
  P = \{ \text{Slicing, Chopping, Frying, Washing, ...} \}
  $$

- **Output sequence**  
  A sequence of actions corresponding to segments or frames:

  $$
  Y = P^n
  $$

This setup allows us to model real-world tasks where outputs have temporal structure—actions occur over time and are dependent on previous context. Structured prediction opens the door to powerful models that understand more than just isolated labels—they reason over entire sequences and structures.

>**But wait—doesn't the model just predict POS tags for the given input? Where does context come in?**

Great question! It might seem like we're simply classifying each word. But in **structured prediction**, we **don't** predict each tag independently. Instead, we predict the **entire sequence jointly**—which means the model **does consider context** while assigning tags.

**How?**

Structured prediction models use features that depend on both the current and **previous tags** (Markov dependencies). For example:

- If the previous tag is `Pronoun`, it's likely the current tag is `Verb`.
- If the previous word is `He` and the current word is `runs`, the current tag is likely `Verb`.

These dependencies are built into the model using **joint feature vectors** and **structured scoring**. Instead of a single-label classifier, we score the entire output sequence and pick the best-scoring one:

$$
\hat{y} = \arg\max_{y \in Y(x)} h(x, y)
$$

Now that we understand how structured models use context, let's explore the hypothesis space that makes this possible.

---

##### **Hypothesis Space for Structured Outputs**

In structured prediction, the output space $$Y(x)$$ is **large and structured**—its size depends on the input $$x$$.

We define:

- **Base hypothesis space**:
  
  $$
  H = \{ h : X \times Y \to \mathbb{R} \}
  $$

- **Compatibility score**:  
  
  $$
  h(x, y)
  $$

  gives a real-valued score that measures how compatible an input $$x$$ is with a candidate output $$y$$.

- **Final prediction function**:
  
  $$
  f(x) = \arg\max_{y \in Y} h(x, y), \quad f \in F
  $$

So, our model chooses the **most compatible output structure** based on the scoring function.

##### **Designing the Compatibility Score**

We use a **linear model** to define the compatibility score:

$$
h(x, y) = \langle w, \Psi(x, y) \rangle
$$

Where:

- $$w$$ is a parameter vector to be learned.
- $$\Psi(x, y)$$ is a **joint feature representation** of the input-output pair.

Let’s break down how to construct this feature vector.

Structured prediction leverages **decomposable features** that split complex structures into simpler parts.


**Unary Features**

Unary features depend on the label at a single position $$i$$:

- Example features:
  
  $$
  \phi_1(x, y_i) = 1[x_i = \text{runs}] \cdot 1[y_i = \text{Verb}]
  $$

  $$
  \phi_2(x, y_i) = 1[x_i = \text{runs}] \cdot 1[y_i = \text{Noun}]
  $$

  $$
  \phi_3(x, y_i) = 1[x_{i-1} = \text{He}] \cdot 1[x_i = \text{runs}] \cdot 1[y_i = \text{Verb}]
  $$


**Markov Features**

Markov features capture dependencies between **adjacent labels** (like in HMMs):

- Example features:
  
  $$
  \theta_1(x, y_{i-1}, y_i) = 1[y_{i-1} = \text{Pronoun}] \cdot 1[y_i = \text{Verb}]
  $$

  $$
  \theta_2(x, y_{i-1}, y_i) = 1[y_{i-1} = \text{Pronoun}] \cdot 1[y_i = \text{Noun}]
  $$

These features are key to modeling the **structure** in structured prediction tasks. By combining them across all positions in a sequence, we construct the full joint feature vector $$\Psi(x, y)$$.

---

##### **Local Compatibility Score**

At each position $$i$$ in a sequence:

- Define the local feature vector:
  $$
  \Psi_i(x, y_{i-1}, y_i) = (\phi_1(x, y_i), \phi_2(x, y_i), \dots, \theta_1(x, y_{i-1}, y_i), \theta_2(x, y_{i-1}, y_i), \dots)
  $$

- Local compatibility score:
  $$
  \langle w, \Psi_i(x, y_{i-1}, y_i) \rangle
  $$

The total compatibility score is the sum over the sequence:

$$
h(x, y) = \sum_i \langle w, \Psi_i(x, y_{i-1}, y_i) \rangle = \langle w, \Psi(x, y) \rangle
$$

where

$$
\Psi(x, y) = \sum_i \Psi_i(x, y_{i-1}, y_i)
$$


##### **Learning with Structured Perceptron**

The structured perceptron learns by updating the weight vector when the predicted structure doesn't match the true output:

**Algorithm**

1. Initialize:  
   $$ w \leftarrow 0 $$
2. For multiple passes over data:
   - For each training example $$(x, y)$$:
     - Predict:
       $$ \hat{y} = \arg\max_{y' \in Y(x)} \langle w, \Psi(x, y') \rangle $$
     - If $$ \hat{y} \ne y $$:
       $$
       w \leftarrow w + \Psi(x, y) - \Psi(x, \hat{y})
       $$

This is **identical to multiclass perceptron**, except that the prediction $$\hat{y}$$ comes from a structured space.


##### **Structured Hinge Loss and Structured SVM**

**Generalized Hinge Loss**

We want to ensure the correct output scores higher than incorrect outputs **by a margin** proportional to their difference:

$$
\ell_{\text{hinge}}(x, y) = \max_{y' \in Y(x)} \left[ \Delta(y, y') + \langle w, \Psi(x, y') - \Psi(x, y) \rangle \right]
$$

- $$\Delta(y, y')$$ is the **loss between sequences**, commonly the Hamming loss:

  $$
  \Delta(y, y') = \frac{1}{L} \sum_{i=1}^L 1[y_i \ne y'_i]
  $$

This loss encourages the model to prefer the correct output by a margin of at least $$\Delta(y, y')$$.

##### **Structured SVM Objective**

We minimize a regularized empirical risk based on the structured hinge loss:

$$
\min_{w} \frac{1}{2} \|w\|^2 + C \sum_{(x, y) \in D} \ell_{\text{hinge}}(x, y)
$$

This is a convex optimization problem, typically solved using **stochastic sub-gradient descent**.

##### **Structured Perceptron vs Structured SVM**

| Aspect                  | Structured Perceptron             | Structured SVM                          |
|-------------------------|------------------------------------|------------------------------------------|
| Loss function           | Zero-one (mistake-driven)          | Hinge loss with margin                   |
| Optimization            | Perceptron-style updates           | Convex optimization                      |
| Margin                  | No explicit margin                 | Enforces margin via hinge loss           |
| Regularization          | None                               | $$\ell_2$$ regularization                 |
| Stability               | Less stable                        | More stable and generalizes better       |

##### **Summary**

- Structured prediction is useful for tasks with interdependent outputs.
- The hypothesis space is built using compatibility functions between inputs and structured outputs.
- Features can be unary or Markov, and the overall score decomposes over the sequence.
- Structured perceptron is a natural extension of multiclass perceptron.
- Structured SVM introduces margins and hinge loss for better generalization.

---

In upcoming posts, we'll look into efficient inference algorithms (like Viterbi) for structured prediction and practical applications such as CRFs and sequence tagging in NLP.

