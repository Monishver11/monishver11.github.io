---
layout: post
title: Structured Prediction and Multiclass SVM
date: 2025-04-13 14:05:00-0400
featured: false
description: An in-depth yet intuitive walkthrough of structured prediction, covering sequence labeling, feature engineering, and scoring methods for complex outputs.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

Structured prediction is a powerful framework used when our output space is complex and structured — such as sequences, trees, or graphs — rather than simple class labels. This post builds on multiclass SVMs to delve deeper into structured prediction, exploring how we define and learn over complex output spaces, as well as the notions of joint feature representations and local compatibility scores.

---

##### **What is Structured Prediction?**

In standard classification, we predict a single label for each input—like identifying whether an image contains a cat or a dog.

But what if our outputs aren’t that simple? What if the prediction itself has structure?

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

Now that we've seen how structured prediction breaks down sequences into parts using Unary and Markov features, the next question is:

**How do we combine these local components to score an entire sequence?**

This leads us to the idea of **local compatibility scores**.

##### **Local Compatibility Score**

At each position $$i$$ in the sequence, we compute a **local feature vector** that captures both the current label and the transition from the previous label.

- Local feature vector:
  $$
  \Psi_i(x, y_{i-1}, y_i) = \big( \phi_1(x, y_i), \phi_2(x, y_i), \dots, \theta_1(x, y_{i-1}, y_i), \theta_2(x, y_{i-1}, y_i), \dots \big)
  $$

- Local compatibility score:
  $$
  \langle w, \Psi_i(x, y_{i-1}, y_i) \rangle
  $$

To get the **total compatibility score** for the input-output pair $$(x, y)$$, we **sum these local scores** over the sequence:

$$
h(x, y) = \sum_i \langle w, \Psi_i(x, y_{i-1}, y_i) \rangle
$$

This is equivalent to:

$$
h(x, y) = \langle w, \Psi(x, y) \rangle
$$

Where the **global feature vector** is the sum of all local feature vectors:

$$
\Psi(x, y) = \sum_i \Psi_i(x, y_{i-1}, y_i)
$$

This decomposition is what makes learning and inference tractable in structured models like CRFs, structured perceptrons, and structured SVMs.

---

##### **Let’s walk through the logic with an example: Part-of-Speech (POS) Tagging for the sentence**

Input (x): [START] He runs fast

Goal: Predict the most likely sequence of POS tags:

Output (y): [START] Pronoun Verb Adverb

**Step 1: What are we learning?**

We want to **learn a scoring function**:

$$
h(x, y) = \langle w, \Psi(x, y) \rangle
$$

This function gives a **score** to a candidate output sequence $$y$$ for a given input $$x$$. The higher the score, the more compatible we believe $$x$$ and $$y$$ are.

**Step 2: Why structured outputs are different**

In structured prediction, the output $$y$$ isn't just a single label—it’s a whole **sequence** (or tree, or grid, etc.).

For our sentence, that means predicting:

[Pronoun, Verb, Adverb]

instead of a single class like just “Verb”.

**Step 3: Representing compatibility with features**

We use **feature functions** to capture useful information from $$(x, y)$$:

- **Unary features** look at the input and the label at a single position (e.g., “He” → “Pronoun”)
- **Markov features** look at **transitions between labels** (e.g., “Pronoun” → “Verb”)

These become the building blocks of our model.

**Step 4: Breaking down the full sequence**

For a sequence of length 3 (ignoring [START] token), we define **local features** at each position $$i$$:

- At $$i = 1$$: "He" tagged as Pronoun  
- At $$i = 2$$: "runs" tagged as Verb  
- At $$i = 3$$: "fast" tagged as Adverb  

At each step, we build a **local feature vector**:

$$
\Psi_i(x, y_{i-1}, y_i)
$$

This vector includes both:
- Unary features for $$x_i$$ and $$y_i$$
- Markov features for $$y_{i-1}$$ and $$y_i$$

**Step 5: Computing local scores**

We compute a **local score** at each position:

$$
\langle w, \Psi_i(x, y_{i-1}, y_i) \rangle
$$

This tells us how well the current word and label (and label transition) fit the model.

Do this for all positions $$i$$ in the sequence.

**Let’s walk through this sequence step-by-step.**

At $$i = 1$$ (Word: *He*, Tag: *Pronoun*)

Since this is the first word, we assume the previous tag is `START`:

$$
y_0 = \text{START}
$$

We define:

$$
\Psi_1(x, y_0, y_1) =
\begin{cases}
\phi_1(x_1 = \text{He}, y_1 = \text{Pronoun}) = 1 \\
\theta_1(y_0 = \text{START}, y_1 = \text{Pronoun}) = 1
\end{cases}
$$

All other components of $$\Psi_1$$ are zero.

At $$i = 2$$ (Word: *runs*, Tag: *Verb*)

$$
y_1 = \text{Pronoun}, \quad y_2 = \text{Verb}
$$

We define:

$$
\Psi_2(x, y_1, y_2) =
\begin{cases}
\phi_2(x_2 = \text{runs}, y_2 = \text{Verb}) = 1 \\
\theta_2(y_1 = \text{Pronoun}, y_2 = \text{Verb}) = 1
\end{cases}
$$

Other entries in $$\Psi_2$$ are zero.

At $$i = 3$$ (Word: *fast*, Tag: *Adverb*)

$$
y_2 = \text{Verb}, \quad y_3 = \text{Adverb}
$$

We define:

$$
\Psi_3(x, y_2, y_3) =
\begin{cases}
\phi_3(x_3 = \text{fast}, y_3 = \text{Adverb}) = 1 \\
\theta_3(y_2 = \text{Verb}, y_3 = \text{Adverb}) = 1
\end{cases}
$$


**Step 6: Summing up the local scores**

To score the full sequence $$(x, y)$$, we **sum all local scores**:

$$
h(x, y) = \sum_i \langle w, \Psi_i(x, y_{i-1}, y_i) \rangle
$$

This total score tells us how compatible this **entire sequence of labels** is with the input.

We also define the **global feature vector** as:

$$
\Psi(x, y) = \sum_i \Psi_i(x, y_{i-1}, y_i)
$$

So that the score becomes:

$$
h(x, y) = \langle w, \Psi(x, y) \rangle
$$

So, in our example, this will be:

$$
\Psi(x, y) = \Psi_1(x, y_0, y_1) + \Psi_2(x, y_1, y_2) + \Psi_3(x, y_2, y_3)
$$

Then, the **total compatibility score** is:

$$
h(x, y) = \langle w, \Psi(x, y) \rangle = \sum_{i=1}^3 \langle w, \Psi_i(x, y_{i-1}, y_i) \rangle
$$


**Step 7: Prediction**

Finally, to predict the best output sequence for a new input $$x$$, we find:

$$
f(x) = \arg\max_{y \in Y} \langle w, \Psi(x, y) \rangle
$$

This means: "Find the label sequence $$y$$ that gives the highest compatibility score with $$x$$."

---

##### **Summary of the Complete Flow**

1. **Input**: Sentence $$x =$$ [START] He runs fast
2. **Output space**: All possible tag sequences of same length
3. **For each sequence $$y$$**:
   - Break it into local pairs: $$(y_{i-1}, y_i)$$
   - Construct local features $$\Psi_i(x, y_{i-1}, y_i)$$
   - Compute local scores and sum them
4. **Choose** the sequence $$y$$ with highest score $$h(x, y)$$

We’ve now built a clear understanding of how compatibility scores work in structured prediction—by combining **decomposable local features** across a sequence. This formulation helps capture both local label associations and dependencies between adjacent labels.

In the **next section**, we’ll dive into **Structured Perceptron** and **Structured SVMs**, where we learn how to train these models using mistake-driven updates and margin-based losses.

Take care!

