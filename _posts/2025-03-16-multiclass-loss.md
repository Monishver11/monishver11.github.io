---
layout: post
title: Multiclass Logistic Regression & Multiclass Perceptron Algorithm
date: 2025-04-11 20:32:00-0400
featured: false
description: Learn the essentials of multiclass classification, focusing on logistic regression, perceptron algorithms, and efficient model building techniques.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

In real-world machine learning problems, we often need to classify data into multiple categories, not just two. While binary classification is a fundamental building block, it's crucial to understand how we can extend these ideas to handle multiple classes. This transition from binary to multiclass classification is what we’ll explore in this blog. We’ll start by revisiting **binary logistic regression**, then step into **multiclass logistic regression**, and finally discuss how we can generalize algorithms like the perceptron for multiclass classification.

---

##### **Binary Logistic Regression Recap**

Let's begin with the most basic form of classification: binary logistic regression. 

Given an input $$x$$, our goal is to predict whether it belongs to class 1 or class 0. The function we use for binary classification is called the **sigmoid function**, which outputs a probability between 0 and 1:

$$
f(x) = \sigma(z) = \frac{1}{1 + \exp(-z)} = \frac{1}{1 + \exp(-w^\top x - b)} \quad (1)
$$

The output $$f(x)$$ represents the probability of class 1. The probability of the other class (class 0) is simply:

$$
1 - f(x) = \frac{\exp(-w^\top x - b)}{1 + \exp(-w^\top x - b)} = \frac{1}{1 + \exp(w^\top x + b)} = \sigma(-z) \quad (2)
$$

Another way to think about this is that one class corresponds to the parameters $$w$$ and $$b$$, while the other class corresponds to the parameters $$-w$$ and $$-b$$. This helps set the foundation for extending this concept to multiple classes.

---

##### **Extending to Multiclass Logistic Regression**

Now that we have a solid understanding of binary logistic regression, let’s consider the case where we have more than two classes. This is where **multiclass logistic regression** comes in. For each class $$c$$, we assign a weight vector $$w_c$$ and a bias $$b_c$$. The probability of belonging to class $$c$$ given an input $$x$$ is computed using the **softmax function**:

$$
f_c(x) = \frac{\exp(w_c^\top x + b_c)}{\sum_{c'} \exp(w_{c'}^\top x + b_{c'})} \quad (3)
$$

This formulation, known as **softmax regression**, allows us to calculate the probability for each class and select the class with the highest probability.

##### **The Loss Function**

To train the model, we use a **cross-entropy loss** function, which measures how well the model's predicted probabilities match the true labels. Given a dataset $$\{(x^{(i)}, y^{(i)})\}$$, the loss is defined as:

$$
L = \sum_i -\log f_{y^{(i)}}(x^{(i)})
$$

This loss function encourages the model to assign higher probabilities to the correct class. The gradient of the loss with respect to the pre-activation (logits) is:

$$
\frac{\partial L}{\partial z} = f - y
$$

**Derivation:**

Assume the true class is $$k = y^{(i)}$$. The loss for this example is:

$$
\ell^{(i)} = -\log f_k^{(i)}
$$

Substituting in the softmax definition:

$$
\ell^{(i)} = -\log\left( \frac{\exp(z_k^{(i)})}{\sum_{j=1}^C \exp(z_j^{(i)})} \right) = -z_k^{(i)} + \log \left( \sum_{j=1}^C \exp(z_j^{(i)}) \right)
$$

We differentiate the loss $$\ell^{(i)}$$ with respect to each logit $$z_c^{(i)}$$. There are two cases:

- **Case 1: $$c = k$$ (the correct class)**

$$
\frac{\partial \ell^{(i)}}{\partial z_k^{(i)}} = -1 + \frac{\exp(z_k^{(i)})}{\sum_{j=1}^C \exp(z_j^{(i)})} = f_k^{(i)} - 1
$$

- **Case 2: $$c \ne k$$**

$$
\frac{\partial \ell^{(i)}}{\partial z_c^{(i)}} = \frac{\exp(z_c^{(i)})}{\sum_{j=1}^C \exp(z_j^{(i)})} = f_c^{(i)}
$$

We can express both cases together using the one-hot encoded label vector $$y^{(i)}$$:

$$
\frac{\partial \ell^{(i)}}{\partial z^{(i)}} = f^{(i)} - y^{(i)}
$$

Now, let $$f$$ and $$y$$ now represent the matrices of predicted probabilities and one-hot labels over the entire dataset. Then the total loss is:

$$
L = \sum_{i=1}^N \ell^{(i)} = -\sum_{i=1}^N \log f_{y^{(i)}}^{(i)}
$$

By stacking all gradients, the overall gradient of the loss with respect to the logits becomes:

$$
\frac{\partial L}{\partial z} = f - y
$$

This fully vectorized form allows efficient implementation and is similar to the gradient descent update used in binary logistic regression but generalized to multiple classes.

---

##### **Quick Comparison to One-vs-All (OvA) Approach**

In many multiclass problems, instead of learning a separate model for each class, we can use the **One-vs-All (OvA)** strategy. In OvA, we train a binary classifier for each class, where the classifier tries to distinguish one class from all others. The base hypothesis space in this case is:

$$
\mathcal{H} = \{ h: \mathcal{X} \to \mathbb{R} \} \quad \text{(score functions)}
$$

For $$k$$ classes, the **multiclass hypothesis space** is:

$$
\mathcal{F} = \left\{ x \mapsto \arg\max_i h_i(x) \ \big| \ h_1, \ldots, h_k \in \mathcal{H} \right\}
$$

Intuitively, each function $$h_i(x)$$ scores how likely $$x$$ belongs to class $$i$$. During training, we want each classifier to output positive values for examples from its own class and negative values for examples from all other classes. At test time, the classifier that outputs the highest score determines the predicted class.

---

##### **Multiclass Perceptron: Generalizing the Perceptron Algorithm**

The classic Perceptron algorithm is designed for binary classification, but it can be naturally extended to multiclass problems. In the multiclass setting, instead of a single weight vector, we maintain **one weight vector per class**.

For each class $$i$$, we define a **linear scoring function**:

$$
h_i(x) = w_i^\top x, \quad w_i \in \mathbb{R}^d
$$

Given an input $$x$$, the model predicts the class with the highest score:

$$
\hat{y} = \arg\max_{i} w_i^\top x
$$

The algorithm proceeds iteratively, updating the weights when it makes a mistake:

1. **Initialize**: Set all weight vectors to zero, $$w_i = 0$$ for all classes $i$.
2. For $$T$$ iterations over the training set:
   - For each training example $$(x, y)$$:
     - Predict the label:
       $$
       \hat{y} = \arg\max_{i} w_i^\top x
       $$
     - If $$\hat{y} \neq y$$ (i.e., the prediction is incorrect):
       - **Promote** the correct class:  
         $$
         w_y \leftarrow w_y + x
         $$
       - **Demote** the incorrect prediction:
         $$
         w_{\hat{y}} \leftarrow w_{\hat{y}} - x
         $$

This update increases the score for the true class and decreases the score for the incorrect one, helping the model learn to separate them better in future iterations.

##### **Rewrite the scoring function**

When the number of classes $$k$$ is large, storing and updating $$k$$ separate weight vectors can become computationally expensive. To address this, we can rewrite the scoring function in a more compact form using a **shared weight vector**.

We define a **joint feature map** $$\psi(x, i)$$ that combines both the input $$x$$ and a class label $$i$$. Then, the score for class $$i$$ can be written as:

$$
h_i(x) = w_i^\top x = w^\top \psi(x, i) \tag{4}
$$

Now, instead of maintaining a separate $$w_i$$ for each class, we use **a single global weight vector** $$w$$ that interacts with $$\psi(x, i)$$ to compute scores for all classes:

$$
h(x, i) = w^\top \psi(x, i) \tag{5}
$$


This transformation allows us to use a single weight vector for all classes, which significantly reduces memory usage and computational complexity.

**Concrete Example**


Let:

- Input vector $$x \in \mathbb{R}^2$$, e.g.,

$$
x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

- Number of classes $$k = 3$$

We define $$\psi(x, i)$$ as a vector in $$\mathbb{R}^{2k}$$ (i.e., 6 dimensions). It places $$x$$ into the block corresponding to class $$i$$ and zeros elsewhere.

For example, for class $$i = 2$$:

$$
\psi(x, 2) =
\begin{bmatrix}
0 \\
0 \\
1 \\
2 \\
0 \\
0 \\
\end{bmatrix}
$$


Let $$w \in \mathbb{R}^6$$ (since $$x \in \mathbb{R}^2$$ and $$k = 3$$):

$$
w =
\begin{bmatrix}
0.5 \\
-1.0 \\
0.2 \\
0.3 \\
-0.4 \\
1.0 \\
\end{bmatrix}
$$

To compute the score for class 2:

$$
h(x, 2) = w^\top \psi(x, 2)
$$

Only the block for class 2 is active:

$$
h(x, 2) = [0.2, 0.3]^\top \cdot [1, 2] = 0.2 \cdot 1 + 0.3 \cdot 2 = 0.8
$$


We can now compute scores for all classes:

- Class 1 uses block: $$[0.5, -1.0]$$

  $$
  h(x, 1) = 0.5 \cdot 1 + (-1.0) \cdot 2 = -1.5
  $$

- Class 2: (already computed) $$0.8$$

- Class 3 uses block: $$[-0.4, 1.0]$$

  $$
  h(x, 3) = -0.4 \cdot 1 + 1.0 \cdot 2 = 1.6
  $$

For final prediction, we select the class with the highest score:

$$
\hat{y} = \arg\max_i h(x, i) = 3
$$

So, for input $$x = [1, 2]$$, the predicted class is **3**.

And suppose the true label is:

$$
y = 2
$$

Since $$\hat{y} \ne y$$, the prediction is incorrect.

The classic multiclass Perceptron updates:

- **Promote** the correct class (add input to the correct class block)
- **Demote** the predicted class (subtract input from the predicted class block)

Using the joint feature map:

$$
w \leftarrow w + \psi(x, y) - \psi(x, \hat{y})
$$

In our case:

- $$\psi(x, y) = \psi(x, 2) = [0, 0, 1, 2, 0, 0]^\top$$  
- $$\psi(x, \hat{y}) = \psi(x, 3) = [0, 0, 0, 0, 1, 2]^\top$$

Then:

$$
w_{\text{new}} = w + \psi(x, 2) - \psi(x, 3)
$$

Apply this update:

$$
w =
\begin{bmatrix}
0.5 \\
-1.0 \\
0.2 \\
0.3 \\
-0.4 \\
1.0 \\
\end{bmatrix}
+
\begin{bmatrix}
0 \\
0 \\
1 \\
2 \\
0 \\
0 \\
\end{bmatrix}
-
\begin{bmatrix}
0 \\
0 \\
0 \\
0 \\
1 \\
2 \\
\end{bmatrix}
=
\begin{bmatrix}
0.5 \\
-1.0 \\
1.2 \\
2.3 \\
-1.4 \\
-1.0 \\
\end{bmatrix}
$$

This update increases the score for the correct class (2) and decreases the score for the incorrect prediction (3), just like the original multiclass Perceptron but using a single shared weight vector and structured feature representation.

---

##### **Formalising via Multivector Construction**

Consider a simple example where $$x \in \mathbb{R}^2$$ and we have 3 classes $$Y = \{1, 2, 3\}$$. Suppose we stack the weight vectors for each class together in the following way:

$$
w = \begin{pmatrix}
-\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}, 0, 1, \frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}
\end{pmatrix}^\top
$$

Now, define the feature map $$\Psi(x, y)$$ as follows:

- $$\Psi(x, 1) = (x_1, x_2, 0, 0, 0, 0)$$
- $$\Psi(x, 2) = (0, 0, x_1, x_2, 0, 0)$$
- $$\Psi(x, 3) = (0, 0, 0, 0, x_1, x_2)$$

The dot product between the weight vector $$w$$ and the feature map $$\Psi(x, y)$$ is then:

$$
\langle w, \Psi(x, y) \rangle = \langle w_y, x \rangle
$$

This approach allows us to represent all classes using a single weight vector, which is more efficient and scalable.

With the multivector construction in place, the multiclass perceptron algorithm can be rewritten as follows:

1. **Initialize** the weight vector $$w = 0$$.
2. For $$T$$ iterations, repeat the following for each training example $$(x, y)$$:
   - Predict $$\hat{y} = \arg\max_{y'} w^\top \psi(x, y')$$ (choose the class with the highest score).
   - If $$\hat{y} \neq y$$:
     - Update the weight vector: $$w \leftarrow w + \psi(x, y)$$.
     - Update the weight vector: $$w \leftarrow w - \psi(x, \hat{y})$$.

This version of the algorithm is computationally efficient and scales well to large datasets.

**Question**: What is the **base binary classification problem** in multiclass perceptron?

**Answer**: At each update step, the multiclass Perceptron reduces to a binary classification problem between the correct class $$y$$ and the predicted class $$\hat{y}$$. The model must adjust the weights so that $$y$$ scores higher than $$\hat{y}$$ — just like in a binary classification setting where one class must be separated from another.

---

##### **Feature Engineering for Multiclass Tasks**

To apply the multivector construction in practice, we need to define meaningful and informative features that capture the relationship between the input and each possible class. This is especially important in structured prediction tasks like **part-of-speech (POS) tagging**.

Suppose our input space $$X$$ consists of all possible words, and our label space $$Y$$ contains the categories {NOUN, VERB, ADJECTIVE, ADVERB, etc.}. Each input word needs to be classified into one of these grammatical categories.

We can define features that depend on both the input word and the target label — a natural fit for the joint feature map $$\Psi(x, y)$$ introduced earlier. For example, some useful features might include:

- Whether the word is exactly a specific token (e.g., "run", "apple")
- Whether the word ends in certain suffixes (e.g., “ly” for adverbs)
- Capitalization or presence of digits (in named entity recognition)

Here are a few sample features written in the multivector style:

- $$\psi_1(x, y) = 1[x = \text{apple} \land y = \text{NOUN}]$$  
- $$\psi_2(x, y) = 1[x = \text{run} \land y = \text{NOUN}]$$  
- $$\psi_3(x, y) = 1[x = \text{run} \land y = \text{VERB}]$$  
- $$\psi_4(x, y) = 1[x \text{ ends in } \text{ly} \land y = \text{ADVERB}]$$

Each of these features "activates" only when both the input word and the predicted class match a certain pattern. This is perfectly aligned with the multivector framework, where the model learns weights for specific combinations of features and labels.


##### **Feature Templates**

In real-world applications, especially in natural language processing (NLP), we rarely hand-code features for every word. Instead, we use **feature templates** that automatically generate features from observed patterns.

**What is a Feature Template?**


A feature template is a **rule or function** that, given an input and a label, produces one or more binary features of the form $$\psi(x, y)$$.

Templates help create thousands or even millions of features in a structured and consistent way.

Let’s say we want to predict the POS tag for the word **"running"** in the sentence:

> I am **running** late.

We might use the following templates:

---

| Template Description                          | Template Rule                                      | Example Feature |
|-----------------------------------------------|----------------------------------------------------|-----------------|
| Current word                                   | $$\psi(x, y) = 1[x = w \land y = y']$$             | $$x = \text{"running"}, y = \text{VERB}$$ |
| Word suffix (3 chars)                          | $$\psi(x, y) = 1[x[-3:] = s \land y = y']$$        | $$x[-3:] = \text{"ing"}, y = \text{VERB}$$ |
| Previous word is "am"                          | $$\psi(x, y) = 1[\text{prev}(x) = \text{"am"} \land y = y']$$ | $$y = \text{VERB}$$ |
| Is capitalized                                 | $$\psi(x, y) = 1[x[0].\text{isupper()} \land y = y']$$ | — |
| Prefix (first 2 letters)                       | $$\psi(x, y) = 1[x[:2] = p \land y = y']$$         | $$x[:2] = \text{"ru"}, y = \text{VERB}$$ |

---

Each of these templates would produce many feature instances across a dataset — and each instance activates only when the corresponding condition holds.


**Integration with the Model**

In the multivector model, we don't store a giant feature matrix explicitly. Instead, we treat each **feature-label pair** $$\psi(x, y)$$ as a key that can be mapped to an **index** in a long feature vector. This is done using either:

- A **dictionary lookup**, if we predefine all feature-label pairs, or  
- A **hash function**, if we want to compute the index on the fly (common in online or large-scale settings)

**Why is this needed?**

When $$\psi(x, y)$$ is represented as a very large sparse vector (e.g. size 100,000+), we don't want to store all zeros. So instead, we store only the **nonzero features** — each one identified by its **feature name and associated label**.

Say we define a feature template:

- "Does the word end with 'ing'?"

Then for the input word **"running"**, and possible labels:

- $$\psi(x = \text{running}, y = \text{VERB}) = 1[\text{suffix} = ing \land y = \text{VERB}]$$  
- $$\psi(x = \text{running}, y = \text{NOUN}) = 1[\text{suffix} = ing \land y = \text{NOUN}]$$

These are **two different features**, because they are tied to different labels.

We can assign an index to each:

| Feature Name                                   | Label  | Combined Key                     | Index |
|------------------------------------------------|--------|----------------------------------|--------|
| `suffix=ing`                                   | VERB   | `suffix=ing_VERB`               | 1921   |
| `suffix=ing`                                   | NOUN   | `suffix=ing_NOUN`               | 2390   |

So $$\psi(x, y)$$ is implemented as:

- A vector of size (say) 50,000,
- With a single non-zero at position 1921 or 2390, depending on the label,
- And the model's weight vector $$w$$ has learned weights at those positions.
- During prediction, we compute:

  $$
  \hat{y} = \arg\max_{y'} w^\top \psi(x, y')
  $$


This is how the model can **distinguish between "ing" being a verb signal vs a noun signal**, just by associating label-specific versions of the feature. And this feature-to-index mapping is what makes it possible to use linear classifiers with sparse high-dimensional features efficiently.


**So, Why Feature Templates Matter?**

- They **automate** feature construction and ensure consistency across training and test data.
- They **generalize well** — e.g., instead of memorizing that "running" is a verb, a suffix-based feature can generalize that any word ending in "ing" is likely a verb.
- They are **language-agnostic** to some extent — and can be extended to other structured tasks like NER, chunking, or even machine translation.


This feature-based view, combined with the multivector construction, gives us a powerful and scalable way to build multiclass classifiers, especially in domains like NLP where feature engineering plays a key role.

---

##### **Conclusion**

We covered how multiclass classification can be tackled using multiclass loss and perceptron algorithms. We highlighted the importance of feature engineering, specifically through feature templates, which help automatically create relevant features for each class. This approach enables efficient, scalable models, especially in tasks like POS tagging. By mapping feature-label pairs to indices, we can handle large datasets without excessive memory usage.

Having seen how to generalize the perceptron algorithm, we’ll now move on to explore how **Support Vector Machines (SVMs)** can be extended to handle multiclass classification. Stay tuned and Take care!

