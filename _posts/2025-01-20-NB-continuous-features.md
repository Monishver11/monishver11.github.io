---
layout: post
title: Gaussian Naive Bayes - A Natural Extension
date: 2025-01-20 16:39:00-0400
featured: false
description: Explore how Gaussian Naive Bayes adapts to continuous inputs, including parameter estimation, decision boundaries, and its relation to logistic regression.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

In the previous blog, we explored the Naive Bayes (NB) model for binary features and how it works under the assumption of conditional independence. However, real-world datasets often include continuous features. How can we extend the NB framework to handle such cases? Let’s dive into Gaussian Naive Bayes (GNB), a variant of NB that uses Gaussian distributions to model continuous inputs.


Consider a multiclass classification problem where each input feature $$ x_i $$ is continuous. To model $$ p(x_i \mid y) $$, we assume that the feature values follow a Gaussian (normal) distribution:

$$
p(x_i \mid y = k) \sim \mathcal{N}(\mu_{i,k}, \sigma^2_{i,k}),
$$

where $$ \mu_{i,k} $$ and $$ \sigma^2_{i,k} $$ are the mean and variance of $$ x_i $$ for class $$ y = k $$, respectively. Additionally, we model the class prior probabilities as:

$$
p(y = k) = \theta_k.
$$

With these assumptions, the likelihood of the dataset becomes:

$$
p(D) = \prod_{n=1}^N p_\theta(x^{(n)}, y^{(n)})
$$


$$
p(D) = \prod_{n=1}^N p(y^{(n)}) \prod_{i=1}^d p(x_i^{(n)} \mid y^{(n)}).
$$

Substituting the Gaussian distribution for $$ p(x_i \mid y) $$, we get:

$$
p(D) = \prod_{n=1}^N \theta_{y^{(n)}} \prod_{i=1}^d \frac{1}{\sqrt{2\pi\sigma_{i,y^{(n)}}^2}} \exp\left(-\frac{\left(x_i^{(n)} - \mu_{i,y^{(n)}}\right)^2}{2\sigma_{i,y^{(n)}}^2}\right).
$$

It may seem complex at first, but if you look closely, you'll see that we're applying the same principle. The only difference is in the distribution. To visualize this, we've essentially applied the distribution to a familiar form $$(1)$$ once again to obtain the result. Take a moment to reflect on this.

$$
\hat{y} = \arg\max_{y \in \mathcal{Y}} p(x, y; \theta) = \arg\max_{y} p(y \mid x; \theta) = \arg\max_{y} p(x \mid y; \theta) p(y; \theta) \tag{1}
$$

---

#### **Learning Parameters with Maximum Likelihood Estimation (MLE)**

To train the Gaussian Naive Bayes model, we estimate the parameters $$ \mu_{i,k} $$, $$ \sigma^2_{i,k} $$, and $$ \theta_k $$ using MLE.

##### **Mean ($$ \mu_{i,k} $$):**

The log-likelihood of the data is:

$$
\ell = \sum_{n=1}^N \log \theta_{y^{(n)}} + \sum_{n=1}^N \sum_{i=1}^d \left[-\frac{1}{2} \log (2\pi \sigma_{i,y^{(n)}}^2) - \frac{\left(x_i^{(n)} - \mu_{i,y^{(n)}}\right)^2}{2\sigma_{i,y^{(n)}}^2}\right]
$$

Taking the derivative with respect to $$ \mu_{j,k} $$ and setting it to zero gives:

$$
\mu_{j,k} = \frac{\sum_{n:y^{(n)}=k} x_j^{(n)}}{\sum_{n:y^{(n)}=k} 1}
$$

This is simply the sample mean of $$ x_j $$ for class $$ k $$.

##### **Derivation of $$ \mu_{j,k} $$ for Gaussian Naive Bayes**

To estimate the parameter $$ \mu_{j,k} $$, the mean of feature $$ x_j $$ for class $$ k $$, we maximize the log-likelihood with respect to $$ \mu_{j,k} $$. 


**Step 1: Compute the Derivative of the Log-Likelihood**

The log-likelihood is differentiated with respect to $$ \mu_{j,k} $$:

$$
\frac{\partial}{\partial \mu_{j,k}} \ell = \frac{\partial}{\partial \mu_{j,k}} \sum_{n: y^{(n)} = k} \left( -\frac{1}{2 \sigma_{j,k}^2} \left( x_j^{(n)} - \mu_{j,k} \right)^2 \right)
$$

Ignoring irrelevant terms (constants that do not depend on $$ \mu_{j,k} $$), this simplifies to:

$$
\frac{\partial}{\partial \mu_{j,k}} \ell = \sum_{n: y^{(n)} = k} \frac{1}{\sigma_{j,k}^2} \left( x_j^{(n)} - \mu_{j,k} \right)
$$


**Step 2: Set the Derivative to Zero**

To find the maximum likelihood estimate, set the derivative to zero:

$$
\sum_{n: y^{(n)} = k} \frac{1}{\sigma_{j,k}^2} \left( x_j^{(n)} - \mu_{j,k} \right) = 0
$$


**Step 3: Solve for $$ \mu_{j,k} $$**

Rearranging terms:

$$
\sum_{n: y^{(n)} = k} x_j^{(n)} = \mu_{j,k} \sum_{n: y^{(n)} = k} 1
$$

Divide both sides by $$ \sum_{n: y^{(n)} = k} 1 $$:

$$
\mu_{j,k} = \frac{\sum_{n: y^{(n)} = k} x_j^{(n)}}{\sum_{n: y^{(n)} = k} 1}
$$

**Final Expression**

The maximum likelihood estimate of $$ \mu_{j,k} $$ is:

$$
\mu_{j,k} = \frac{\sum_{n: y^{(n)} = k} x_j^{(n)}}{\sum_{n: y^{(n)} = k} 1}
$$

**Interpretation:**
- $$ \mu_{j,k} $$ is the sample mean of $$ x_j $$ for all data points in class $$ k $$.
- This parameter is essential for defining the Gaussian distribution for feature $$ x_j $$ given class $$ k $$ in Gaussian Naive Bayes.



##### **Variance ($$ \sigma^2_{i,k} $$):**

Similarly, the variance for feature $$ x_j $$ in class $$ k $$ is:

$$
\sigma^2_{j,k} = \frac{\sum_{n:y^{(n)}=k} \left(x_j^{(n)} - \mu_{j,k}\right)^2}{\sum_{n:y^{(n)}=k} 1}
$$

##### **Class Prior ($$ \theta_k $$):**

The class prior $$ \theta_k $$ is estimated as the proportion of data points belonging to class $$ k $$:

$$
\theta_k = \frac{\sum_{n:y^{(n)}=k} 1}{N}
$$


##### **Derivation of $$ \sigma_{j,k}^2 $$ (Sample Variance) and $$ \theta_k $$ (Class Prior)**

**1. Derivation of $$ \sigma_{j,k}^2 $$ (Sample Variance)**

To derive the sample variance $$ \sigma_{j,k}^2 $$, we start from the log-likelihood of the Gaussian distribution for feature $$ x_j $$ within class $$ k $$:

$$
\ell = \sum_{n: y^{(n)} = k} \left[ -\frac{1}{2} \log(2\pi \sigma_{j,k}^2) - \frac{\left( x_j^{(n)} - \mu_{j,k} \right)^2}{2\sigma_{j,k}^2} \right]
$$

We take the derivative of $$ \ell $$ with respect to $$ \sigma_{j,k}^2 $$ and set it to zero:

$$
\frac{\partial \ell}{\partial \sigma_{j,k}^2} = \sum_{n: y^{(n)} = k} \left[ -\frac{1}{2\sigma_{j,k}^2} + \frac{\left( x_j^{(n)} - \mu_{j,k} \right)^2}{2\sigma_{j,k}^4} \right] = 0
$$

Simplify the equation:

$$
\sum_{n: y^{(n)} = k} \left[ -\sigma_{j,k}^2 + \left( x_j^{(n)} - \mu_{j,k} \right)^2 \right] = 0
$$

Divide by $$ \sigma_{j,k}^2 $$ and rearrange:

$$
\sigma_{j,k}^2 = \frac{\sum_{n: y^{(n)} = k} \left( x_j^{(n)} - \mu_{j,k} \right)^2}{\sum_{n: y^{(n)} = k} 1}
$$

Thus, the MLE for $$ \sigma_{j,k}^2 $$ is:

$$
\sigma_{j,k}^2 = \frac{\sum_{n: y^{(n)} = k} \left( x_j^{(n)} - \mu_{j,k} \right)^2}{\sum_{n: y^{(n)} = k} 1}
$$


**2. Derivation of $$ \theta_k $$ (Class Prior)**

The class prior $$ \theta_k $$ represents the proportion of data points belonging to class $$ k $$ in the dataset. It is given by:

$$
\theta_k = \frac{\sum_{n: y^{(n)} = k} 1}{N}
$$

**Steps:**
1. **Numerator**: $$ \sum_{n: y^{(n)} = k} 1 $$ counts the total number of data points that belong to class $$ k $$.
2. **Denominator**: $$ N $$ is the total number of data points in the entire dataset.


**Finally,**

1. **Sample Variance**:
   
   $$
   \sigma_{j,k}^2 = \frac{\sum_{n: y^{(n)} = k} \left( x_j^{(n)} - \mu_{j,k} \right)^2}{\sum_{n: y^{(n)} = k} 1}
   $$

2. **Class Prior**:
   
   $$
   \theta_k = \frac{\sum_{n: y^{(n)} = k} 1}{N}
   $$


- The sample variance $$ \sigma_{j,k}^2 $$ measures the spread of feature $$ x_j $$ for class $$ k $$, derived using MLE.
- The class prior $$ \theta_k $$ represents the proportion of data points in class $$ k $$, computed directly from the dataset.

---

#### **Decision Boundary of Gaussian Naive Bayes**

Now that we have the model parameters, let’s examine the decision boundary of GNB. For binary classification ($$ y \in \{0, 1\} $$), the log odds ratio is given by:

$$
\log \frac{p(y=1 \mid x)}{p(y=0 \mid x)} = \log \frac{p(x \mid y=1)p(y=1)}{p(x \mid y=0)p(y=0)}.
$$

Substituting the Gaussian distributions for $$ p(x \mid y) $$, this simplifies to:

$$
\log \frac{p(y=1 \mid x)}{p(y=0 \mid x)} = \log \frac{\theta_1}{\theta_0} + \sum_{i=1}^d \left[\log \frac{\sigma_{i,0}^2}{\sigma_{i,1}^2} + \frac{\left(x_i - \mu_{i,0}\right)^2}{2\sigma_{i,0}^2} - \frac{\left(x_i - \mu_{i,1}\right)^2}{2\sigma_{i,1}^2}\right].
$$

##### **Linear vs. Quadratic Decision Boundary**

1. **General Case**: If the variances ($$ \sigma_{i,0}^2 $$ and $$ \sigma_{i,1}^2 $$) differ between classes, the decision boundary is **quadratic**.
2. **Shared Variance Assumption**: If we assume that variances are equal for all classes ($$ \sigma_{i,0}^2 = \sigma_{i,1}^2 = \sigma_i^2 $$), the decision boundary becomes **linear**:
   
   $$
   \log \frac{p(y=1 \mid x)}{p(y=0 \mid x)} = \sum_{i=1}^d \frac{\mu_{i,1} - \mu_{i,0}}{\sigma_i^2} x_i + \sum_{i=1}^d \frac{\mu_{i,0}^2 - \mu_{i,1}^2}{2\sigma_i^2}.
   $$

   This form resembles the decision boundary of logistic regression, emphasizing the close connection between the two models.

---

#### **Naive Bayes vs. Logistic Regression**

Both Naive Bayes and logistic regression are popular classifiers, but they differ fundamentally in their approach:

| Feature                 | Naive Bayes            | Logistic Regression      |
|-------------------------|------------------------|--------------------------|
| **Model Type**          | Generative             | Discriminative           |
| **Parametrization**     | $$ p(x \mid y), p(y) $$| $$ p(y \mid x) $$        |
| **Assumptions on $$ y $$** | Bernoulli              | Bernoulli                |
| **Assumptions on $$ x $$** | Gaussian (in GNB)      | None                     |
| **Decision Boundary**   | Linear or Quadratic    | Linear                   |

Interestingly, when GNB's assumptions hold (e.g., Gaussian features, independent dimensions, shared variance), it converges to the same decision boundary as logistic regression asymptotically. However, GNB often converges faster on smaller datasets, while logistic regression achieves lower asymptotic error on larger datasets.

---

#### **Generative vs. Discriminative Models: A Broader Perspective**

The contrast between Naive Bayes and logistic regression highlights the differences between **generative** and **discriminative** models. Generative models like Naive Bayes model the joint distribution $$ p(x, y) $$, allowing them to generate data as well as make predictions. In contrast, discriminative models like logistic regression focus directly on $$ p(y \mid x) $$, optimizing for classification accuracy.

This tradeoff is explored in the classic paper by Ng and Jordan (2002), which shows that generative models converge faster but may have higher asymptotic error compared to their discriminative counterparts.

---

In the next section, we’ll dive deeper into practical implementations of generative models and explore their applications across various domains. Stay tuned!
