---
layout: post
title: Multivariate Gaussian Distribution and Naive Bayes
date: 2025-01-22 23:01:00-0400
featured: false
description: Dive into the multivariate Gaussian distribution, its role in probabilistic modeling, and how it powers Naive Bayes classifiers with practical insights and mathematical intuition.
tags: ML Math
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

When analyzing data in higher dimensions, we often encounter scenarios where input features are not independent. In such cases, the **Multivariate Gaussian Distribution** provides a robust probabilistic framework to model these relationships. It extends the familiar univariate Gaussian distribution to multiple dimensions, enabling us to capture dependencies and correlations between variables effectively.


#### **Understanding the Multivariate Gaussian Distribution**

A multivariate Gaussian distribution is defined as:

$$
x \sim \mathcal{N}(\mu, \Sigma),
$$

where $$\mu$$ is the mean vector, and $$\Sigma$$ is the covariance matrix. Its probability density function is given by:

$$
p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right),
$$

Here, $$d$$ represents the dimensionality of the input $$x$$, $$\vert \Sigma \vert$$ denotes the determinant of the covariance matrix, and $$\Sigma^{-1}$$ is its inverse. 

The term $$(x - \mu)^\top \Sigma^{-1} (x - \mu)$$ is referred to as the **Mahalanobis distance**, which measures the distance of a point $$x$$ from the mean $$\mu$$. Unlike the Euclidean distance, the Mahalanobis distance normalizes for differences in variances and accounts for correlations between the dimensions. This normalization makes it particularly useful in multivariate data analysis.

[Add a better way to understand this + an analogy/intuition to think of this multivariate gaussian]


##### **Intuition and Analogy for Multivariate Gaussian**

Think of the multivariate Gaussian distribution as a **3D bell-shaped curve** (or higher-dimensional equivalent) where:

- The **peak** of the bell is at $$\mu$$, the mean vector.
- The **spread** of the bell in different directions is determined by $$\Sigma$$, the covariance matrix. It stretches or compresses the curve along certain axes depending on the variances and correlations.

###### **Analogy: A Weighted Balloon**
Imagine a balloon filled with air. If the balloon is perfectly spherical, it represents a distribution where all dimensions are independent and have the same variance (this corresponds to $$\Sigma$$ being a diagonal matrix with equal entries). 

Now, if you squeeze the balloon in one direction:
- It elongates in one direction and compresses in another. This reflects **correlations** between dimensions in the data, encoded by the off-diagonal elements of $$\Sigma$$.
- The shape of the balloon changes, and distances (like Mahalanobis distance) now account for these correlations, unlike Euclidean distance.

##### **How to Think About Mahalanobis Distance**
The Mahalanobis distance:

$$
d_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}
$$

can be understood as the distance from a point $$x$$ to the center $$\mu$$, scaled by the shape and orientation of the distribution:

1. **Scaling by Variance**: In directions where the variance is large (the distribution is "spread out"), the Mahalanobis distance will consider points farther from the mean as less unusual. Conversely, in directions where variance is small, even small deviations from the mean are considered significant.
   
2. **Accounting for Correlations**: 
   If two dimensions are correlated, the Mahalanobis distance adjusts for this by using the covariance matrix $$\Sigma$$. The covariance matrix captures both the variances of individual dimensions and the relationships (correlations) between them. 

**Role of the Inverse Covariance Matrix:**

The term $$\Sigma^{-1}$$ (the inverse of the covariance matrix) in the Mahalanobis distance ensures that the contribution of each dimension is scaled appropriately. For example:

- If two dimensions are strongly correlated, deviations along one dimension are partially "explained" by deviations along the other. The Mahalanobis distance reduces the weight of such deviations, treating them as less unusual.
- Conversely, if two dimensions are uncorrelated, the deviations are treated independently.

**Example:**

In a dataset of height and weight, a taller-than-average person is likely to weigh more than average. The covariance matrix captures this relationship, and $$\Sigma^{-1}$$ adjusts the distance calculation to reflect that such deviations are expected. Without this adjustment (as in Euclidean distance), the relationship would be ignored, leading to an overestimation of the "unusualness" of the point.


**Returning to the Balloon Analogy,**

The Mahalanobis distance incorporates the "shape" of the balloon (determined by $$\Sigma$$) to measure distances:

**Shape and Scaling:**

- A spherical balloon corresponds to a covariance matrix where all dimensions are independent and have equal variance. In this case, the Mahalanobis distance reduces to the Euclidean distance.
- A stretched or compressed balloon reflects correlations or differences in variance. The Mahalanobis distance scales the contribution of each dimension based on the covariance structure, ensuring that distances are measured relative to the shape of the distribution.

**How It Works:**

- Points on the surface of the balloon correspond to a Mahalanobis distance of 1, regardless of the balloon's shape. This is because the Mahalanobis distance normalizes for the stretching or compressing of the balloon in different directions.
- Mathematically, this is achieved by transforming the space using $$\Sigma^{-1}$$, effectively "flattening" the correlations and variances. In this transformed space, the balloon becomes a perfect sphere, and distances are measured uniformly.

These adjustments make the Mahalanobis distance a powerful metric for detecting outliers and understanding the distribution of data in a multivariate context.

---

If you're still unsure about the concept, let's walk through an example and explore it together.

###### **1. Covariance Matrix**:

For a dataset with two variables, say **height** ($$x_1$$) and **weight** ($$x_2$$), the covariance matrix $$\Sigma$$ looks like this:

$$
\Sigma = \begin{pmatrix}
\sigma_{11} & \sigma_{12} \\
\sigma_{21} & \sigma_{22}
\end{pmatrix}
$$

Where:
- $$\sigma_{11}$$ is the variance of height ($$x_1$$).
- $$\sigma_{22}$$ is the variance of weight ($$x_2$$).
- $$\sigma_{12} = \sigma_{21}$$ is the covariance between height and weight.

###### **2. Inverse Covariance Matrix**:

The inverse of the covariance matrix $$\Sigma^{-1}$$ is used to "normalize" the data and account for correlations. The inverse of a 2x2 matrix is given by:

$$
\Sigma^{-1} = \frac{1}{\text{det}(\Sigma)} \begin{pmatrix}
\sigma_{22} & -\sigma_{12} \\
-\sigma_{21} & \sigma_{11}
\end{pmatrix}
$$

Where the determinant of the covariance matrix is:

$$
\text{det}(\Sigma) = \sigma_{11} \sigma_{22} - \sigma_{12}^2
$$

###### **3. Example: Correlated Data (Height and Weight)**

Suppose we have a dataset of heights and weights, and the covariance matrix looks like this:

$$
\Sigma = \begin{pmatrix}
100 & 80 \\
80 & 200
\end{pmatrix}
$$

This means:
- The variance of height ($$\sigma_{11}$$) is 100.
- The variance of weight ($$\sigma_{22}$$) is 200.
- The covariance between height and weight ($$\sigma_{12} = \sigma_{21}$$) is 80, indicating a strong positive correlation between height and weight.

Now, let's say we have a data point:

$$
x = \begin{pmatrix} 180 \\ 75 \end{pmatrix}
$$

This means the person is 180 cm tall and weighs 75 kg. The mean of the dataset is:

$$
\mu = \begin{pmatrix} 170 \\ 70 \end{pmatrix}
$$

###### **3.1. Euclidean Distance** (Without Accounting for Correlation)

The Euclidean distance between the data point $$x$$ and the mean $$\mu$$ is:

$$
D_E(x) = \sqrt{(x_1 - \mu_1)^2 + (x_2 - \mu_2)^2}
$$

Substituting the values:

$$
D_E(x) = \sqrt{(180 - 170)^2 + (75 - 70)^2} = \sqrt{10^2 + 5^2} = \sqrt{100 + 25} = \sqrt{125} \approx 11.18
$$

This distance doesn't account for the correlation between height and weight. It treats the two dimensions as if they are independent, and gives a straightforward measure of how far the point is from the mean in Euclidean space.

###### **3.2. Mahalanobis Distance** (With Covariance Adjustment)

Now, let’s compute the Mahalanobis distance. First, we need to compute the inverse of the covariance matrix $$\Sigma^{-1}$$.

The determinant of $$\Sigma$$ is:

$$
\text{det}(\Sigma) = 100 \times 200 - 80^2 = 20000 - 6400 = 13600
$$

So, the inverse covariance matrix is:

$$
\Sigma^{-1} = \frac{1}{13600} \begin{pmatrix} 200 & -80 \\ -80 & 100 \end{pmatrix} = \begin{pmatrix} 0.0147 & -0.0059 \\ -0.0059 & 0.0074 \end{pmatrix}
$$

Now, we compute the Mahalanobis distance:

$$
D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$

Substituting the values:

$$
x - \mu = \begin{pmatrix} 180 - 170 \\ 75 - 70 \end{pmatrix} = \begin{pmatrix} 10 \\ 5 \end{pmatrix}
$$

Now, calculate the Mahalanobis distance:

$$
D_M(x) = \sqrt{\begin{pmatrix} 10 & 5 \end{pmatrix} \begin{pmatrix} 0.0147 & -0.0059 \\ -0.0059 & 0.0074 \end{pmatrix} \begin{pmatrix} 10 \\ 5 \end{pmatrix}}
$$

First, multiply the vectors:

$$
\begin{pmatrix} 10 & 5 \end{pmatrix} \begin{pmatrix} 0.0147 & -0.0059 \\ -0.0059 & 0.0074 \end{pmatrix} = \begin{pmatrix} 10 \times 0.0147 + 5 \times (-0.0059) \\ 10 \times (-0.0059) + 5 \times 0.0074 \end{pmatrix} = \begin{pmatrix} 0.147 - 0.0295 \\ -0.059 + 0.037 \end{pmatrix} = \begin{pmatrix} 0.1175 \\ -0.022 \end{pmatrix}
$$

Now, multiply this result by the vector $$\begin{pmatrix} 10 \\ 5 \end{pmatrix}$$:

$$
\begin{pmatrix} 0.1175 & -0.022 \end{pmatrix} \begin{pmatrix} 10 \\ 5 \end{pmatrix} = 0.1175 \times 10 + (-0.022) \times 5 = 1.175 - 0.11 = 1.065
$$

Thus, the Mahalanobis distance is:

$$
D_M(x) = \sqrt{1.065} \approx 1.03
$$

###### **4. Interpretation of the Results**:

- The **Euclidean distance** between the point and the mean was approximately **11.18**. This suggests that the point is far from the mean, without considering the correlation between height and weight.
- The **Mahalanobis distance** is **1.03**, which is much smaller. This is because the Mahalanobis distance accounts for the fact that height and weight are correlated. The deviation in weight is expected given the deviation in height, so the Mahalanobis distance treats this as less "unusual."

###### **Takeaways:**

- **Euclidean distance** treats each dimension as independent, ignoring correlations, which can lead to an overestimation of how unusual a point is.
- **Mahalanobis distance**, by using the inverse covariance matrix $$\Sigma^{-1}$$, adjusts for correlations and scales the deviations accordingly. This results in a more accurate measure of how far a point is from the mean, considering the underlying structure of the data (e.g., the correlation between height and weight in this example).





---

##### **Grasping Better with Bivariate Normal Distributions**

To build a deeper understanding, let’s focus on a specific case: the two-dimensional Gaussian, commonly referred to as the **bivariate normal distribution**.

###### **Case 1: Identity Covariance Matrix**

Suppose the covariance matrix is given as:

$$
\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}.
$$

In this scenario, the contours of the distribution are circular. This indicates that there is no correlation between the two variables, and both have equal variances. The shape of the contours reflects the isotropic nature of the distribution.

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Multivariate-GNB-1.png" title="Multivariate-GNB-1" class="img-fluid rounded z-depth-1" %}
   </div>
</div>


###### **Case 2: Scaled Identity Covariance**

If we scale the covariance matrix, say:

$$
\Sigma = 0.5 \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},
$$

the variances of both variables decrease, resulting in smaller circular contours. Conversely, if we scale it up:

$$
\Sigma = 2 \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},
$$

the variances increase, leading to larger circular contours. This demonstrates how scaling the covariance matrix affects the spread of the distribution.



###### **Case 3: Anisotropic Variance**

When the variances of the variables are different, such as when $$\text{var}(x_1) \neq \text{var}(x_2)$$, the contours take on an elliptical shape. The orientation and eccentricity of the ellipse are determined by the relative variances along each axis.


<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Multivariate-GNB-2.png" title="Multivariate-GNB-2" class="img-fluid rounded z-depth-1" %}
   </div>
</div>


###### **Case 4: Correlated Variables**

Correlation between variables introduces an additional layer of complexity. 

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Multivariate-GNB-3.png" title="Multivariate-GNB-3" class="img-fluid rounded z-depth-1" %}
   </div>
</div>


For instance, if:

$$
\Sigma = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix},
$$

where $$\rho$$ is the correlation coefficient:
- When $$\rho > 0$$, the variables are positively correlated, and the ellipse tilts along the diagonal.
- When $$\rho < 0$$, the variables are negatively correlated, and the ellipse tilts in the opposite direction.
- When $$\rho = 0$$, the variables remain uncorrelated, resulting in circular or axis-aligned ellipses.



<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Multivariate-GNB-4.png" title="Multivariate-GNB-4" class="img-fluid rounded z-depth-1" %}
   </div>
</div>


---

#### **Gaussian Bayes Classifier**

The **Gaussian Bayes Classifier (GBC)** extends the Gaussian framework to classification tasks. It assumes that the conditional distribution $$p(x \vert y)$$ follows a multivariate Gaussian distribution. Mathematically, for a class $$k$$:

$$
p(x|t = k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k)\right),
$$

where each class $$k$$ has its own mean vector $$\mu_k$$ and covariance matrix $$\Sigma_k$$. The determinant $$\vert \Sigma_k \vert$$ and the inverse $$\Sigma_k^{-1}$$ are crucial components for computing probabilities.

Estimating the parameters for each class becomes computationally challenging in high dimensions, as the covariance matrix has $$O(d^2)$$ parameters. This complexity often necessitates simplifying assumptions to make the model tractable.

[How $$O(d^2)$$ parameters, and more explanation]

##### **Special Cases of Gaussian Bayes Classifier**

To address the computational challenges, we consider the following special cases of the Gaussian Bayes Classifier:

1. **Full Covariance Matrix**  
   Each class has its own covariance matrix. While this allows for flexible modeling, the decision boundary is quadratic, and the computational cost can be prohibitive.

2. **Shared Covariance Matrix**  
   If all classes share a common covariance matrix, the decision boundary becomes linear. This simplification reduces computational complexity and is effective when data approximately follows a Gaussian distribution.

3. **Naive Bayes Assumption**  
   The Naive Bayes classifier assumes a diagonal covariance matrix, meaning the features are conditionally independent given the class. Despite this strong assumption, the decision boundary remains quadratic due to feature-wise variances.

[Add how we got this decision boundaries on each case, little derivation or proof and intuition]

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Multivariate-GNB-5.png" title="Multivariate-GNB-5" class="img-fluid rounded z-depth-1" %}
   </div>
</div>


---

##### **Gaussian Bayes Classifier vs. Logistic Regression**

One interesting connection between GBC and logistic regression arises when the data is truly Gaussian. If we assume shared covariance matrices, the decision boundaries produced by GBC become identical to those of logistic regression. However, logistic regression is more versatile since it does not rely on Gaussian assumptions and can learn other types of decision boundaries.


[How?]


##### **Final Thoughts**

The multivariate Gaussian distribution provides a probabilistic framework for understanding data with correlated features. By extending this to classification tasks, the Gaussian Bayes Classifier offers an elegant and interpretable approach to modeling. However, its reliance on assumptions like Gaussianity and the complexity of covariance estimation in high dimensions present practical challenges.

Generative models, like GBC, aim to model the joint distribution $$p(x, y)$$, which contrasts with discriminative models, such as logistic regression, that focus directly on $$p(y \vert x)$$. While generative models offer a principled way to derive loss functions via maximum likelihood, they can struggle with small datasets, where estimating the joint distribution becomes difficult.


[Need a practical example to think of it. like how its used in case of non-independent features and how we tackle it and then because of using it, whats the computational complexity?]

As you delve deeper into probabilistic frameworks, a question worth pondering is: Do generative models have an equivalent form of regularization to mitigate overfitting? This opens up avenues for exploring how these models can be made more robust in practice.
