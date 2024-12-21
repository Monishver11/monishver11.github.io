---
layout: post
title: Linear Algebra - Prerequisites for Machine Learning
date: 2024-12-19 20:10:00-0400
featured: false
description: This blog post covers the key linear algebra concepts and their applications in machine learning.
tags: ML
categories: sample-posts
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

Linear algebra forms the backbone of modern machine learning. As a branch of mathematics, it deals with vector spaces and the linear transformations between them. This area of study allows for the manipulation and efficient computation of datasets, making it fundamental to various machine learning algorithms. Whether you're working with deep learning, regression models, or optimization techniques, a solid understanding of linear algebra will be crucial to mastering machine learning.

---

## Core Components of Linear Algebra

### Vectors and Matrices

#### 1. Vectors

A **vector** is a fundamental concept in linear algebra and is essentially a one-dimensional array of numbers. In machine learning, vectors can represent different elements, including features, weights, or data points.

- **Definition:** A vector is a set of numbers arranged in a specific order, and it can be represented either as a **row vector** or a **column vector**.
  - Row vector: $$\mathbf{v} = [v_1, v_2, \dots, v_n]$$
  - Column vector:$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n\end{bmatrix}$$
- **Properties:**
  - **Magnitude (Norm):** The magnitude of a vector, often referred to as its norm, measures the vector's length. The most common norms used are the L2 norm (Euclidean norm) and L1 norm. 
  
  $$ 
    \|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} \ \ ; \quad \|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i| 
  $$

  - **Dot Product:** The dot product of two vectors measures their similarity. The dot product between two vectors $$\mathbf{v}_1​$$ and $$\mathbf{v}_2$$​ is computed as: 
  
  $$ 
    \mathbf{v}_1 \cdot \mathbf{v}_2 = \sum_{i=1}^{n} v_{1i} v_{2i} 
  $$

  - **Distance:** The Euclidean distance is a common way to measure the difference between two vectors: 
  
  $$ 
  d(\mathbf{v}_1, \mathbf{v}_2) = \sqrt{\sum_{i=1}^{n} (v_{1i} - v_{2i})^2} 
  $$

- **Operations on Vectors:**
  - **Addition:** Vectors of the same size can be added element-wise.
  - **Scalar Multiplication:** Multiplying each element of a vector by a scalar.
  - **Dot Product:** A fundamental operation for determining the similarity between two vectors.

#### 2. Matrices

A **matrix** is a two-dimensional array of numbers, and it is widely used in machine learning for data storage, transformations, and solving systems of equations.

- **Definition:** A matrix consists of rows and columns and is denoted as $$A$$, where $$A_{ij}$$​ represents the element in the i-th row and j-th column.
  
  $$
  A = \begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\ a_{21} & a_{22} & \dots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \dots & a_{mn} \end{bmatrix}
  $$

- **Properties of Matrices:**
  - **Rank:** The rank of a matrix is the maximum number of linearly independent rows or columns, indicating the number of independent dimensions in the matrix.
  - **Trace:** The trace is the sum of the diagonal elements of a square matrix. It is often involved in optimization problems.
  - **Determinant:** The determinant helps in determining whether a matrix is invertible. A non-zero determinant implies that the matrix is invertible.
  - **Invertibility:** A matrix $$A$$ is invertible if it has full rank and a non-zero determinant. The inverse of a matrix $$A$$ is denoted by $$A^{-1}$$, and it satisfies the equation: $$A A^{-1} = I$$ where $$I$$ is the identity matrix.
- **Operations on Matrices:**
  - **Matrix Addition/Subtraction:** Matrices of the same dimension can be added or subtracted element-wise.
  - **Matrix Multiplication:** Matrix multiplication is the dot product of rows and columns between two matrices. This operation is central to machine learning algorithms.
  - **Transpose:** The transpose of a matrix $$A$$ is denoted as $$A^T$$ and involves flipping its rows and columns.
  - **Inverse:** If a matrix is invertible, its inverse can be used to solve systems of linear equations.

### Vectors and Matrices in ML

Vectors and matrices play a pivotal role in representing both data and models in machine learning.

#### 1. Data Representation

In supervised learning, each data point is typically represented as a feature vector. For example, if a dataset has mmm samples and n features, it can be represented as an $$m \times n$$ matrix, where each row corresponds to a feature vector for a data point. The corresponding labels or target values are often stored in a vector.

#### 2. Model Representation

- **Weight Vectors and Matrices:** In models like linear regression and neural networks, the weights that transform input data are stored in vectors or matrices. For example, in linear regression, the model is defined as: $$\hat{y} = Xw + b$$ where $$X$$ is the data matrix, $$w$$ is the weight vector, and $$b$$ is the bias term.

#### 3. Operations in Machine Learning Algorithms

- **Linear Regression:** In linear regression, matrix operations are used to solve for the optimal weights. The normal equation for linear regression is:

$$ 
w=(X^TX)^{−1}X^Ty 
$$

  where $$X$$ is the matrix of input features and $$y$$ is the vector of target values.
- **Neural Networks:** Each layer of a neural network applies a linear transformation to its input, which is represented by matrix multiplication: 

$$
y=XW+b
$$

where $$X$$ is the input matrix, $$W$$ is the weight matrix, and $$b$$ is the bias vector.
- **Gradient Descent:** The gradient descent optimization algorithm frequently uses vector and matrix operations to update model parameters iteratively. In deep learning, the gradient of the loss function with respect to the weights and biases is calculated using matrix operations during back-propagation.

#### 4. Dimensionality Reduction

- **Principal Component Analysis (PCA):** PCA is a popular technique for dimensionality reduction. It involves finding the eigenvectors and eigenvalues of the covariance matrix of the data. The eigenvectors represent the directions of maximum variance, and the eigenvalues indicate the magnitude of the variance along those directions.

#### 5. Optimization

- **Loss Functions:** Machine learning models are optimized by minimizing a loss function, which often involves vectors and matrices. For instance, in deep learning, back-propagation uses gradients to update the weights by calculating the derivative of the loss function with respect to each weight matrix.

---

### Eigenvalues and Eigenvectors

#### **1. What Are Eigenvalues and Eigenvectors?**

- **Eigenvector:**  
   An **eigenvector** of a square matrix $$\mathbf{A}$$ is a non-zero vector $$\mathbf{v}$$ that, when the matrix $$\mathbf{A}$$ is applied to it, only scales the vector without changing its direction:
   
   $$
   \mathbf{A} \mathbf{v} = \lambda \mathbf{v}
   $$

  where:
  - $$\mathbf{v}$$ is the **eigenvector**,
  - $$\lambda$$ is the **eigenvalue**, the scalar that represents how much the eigenvector is scaled by the transformation.
- **Eigenvalue:**  
   The **eigenvalue** $$\lambda$$ is the factor by which the eigenvector is scaled when the matrix $$\mathbf{A}$$ acts on it.

**To build intuition, consider this analogy:**

Imagine a **squishy sheet** of rubber (the matrix) and a **point** in space (the vector). If you apply a transformation (like stretching, rotating, or shearing) to the point using the rubber sheet, most points move to new locations. However, some special points, called **eigenvectors**, only get **stretched** or **compressed** but **stay in the same direction**. The amount of stretching or compression is determined by the **eigenvalue**.

#### **2. Mathematical Properties of Eigenvalues and Eigenvectors**

- **Eigenvalue Equation:**  
  For a matrix $$\mathbf{A}$$ and a vector $$\mathbf{v}$$:

  $$
  \mathbf{A} \mathbf{v} = \lambda \mathbf{v}
  $$

  where $$\mathbf{A}$$ is an $$n \times n$$ matrix, $$\mathbf{v}$$ is the eigenvector, and $$\lambda$$ is the eigenvalue.
- Eigenvectors must be **non-zero vectors**.
- Eigenvalues can be **real** or **complex** (but are often real in machine learning applications).
- A matrix can have multiple eigenvectors corresponding to the **same eigenvalue** (if it is **degenerate**) or distinct eigenvalues corresponding to distinct eigenvectors.

#### **3. Why Are Eigenvalues and Eigenvectors Important in Machine Learning?**

**Principal Component Analysis (PCA)** is a widely used technique for **dimensionality reduction** in machine learning. It reduces the number of features while retaining the most important information in the dataset.

- **Covariance Matrix:** PCA begins by computing the **covariance matrix** to capture relationships between features.
- **Eigenvectors of Covariance Matrix:** The **eigenvectors** represent the directions of maximum variance in the data—these are the **principal components**.
- **Eigenvalues:** The corresponding **eigenvalues** indicate the magnitude of variance in each direction.

Key steps in PCA:

- Sort eigenvectors in decreasing order of their eigenvalues.
- Select the top **k** eigenvectors to reduce dimensionality while preserving most of the variance.

#### **4. Key Properties of Eigenvalues and Eigenvectors in Machine Learning**

- **Diagonalizability:**  
   A matrix is **diagonalizable** if it has enough eigenvectors to form a full basis. This property is essential in PCA and **Singular Value Decomposition (SVD)**, enabling efficient computation and interpretation.
- **Magnitude of Eigenvalues:**  
   The magnitude of eigenvalues corresponds to the **variance captured** by the associated eigenvectors (principal components). Larger eigenvalues imply more variance explained.
- **Orthogonality of Eigenvectors (Symmetric Matrices):**  
   For **symmetric matrices**, eigenvectors are **orthogonal**. This is critical in PCA, where the principal components are orthogonal, ensuring that reduced dimensions remain **uncorrelated**.

---

### A Few More Key Matrices Relevant to ML

#### 1. **Symmetric Matrix:**

- **Definition:** A matrix $$A$$ is **symmetric** if $$A = A^T$$, meaning it is equal to its transpose.
- **Properties:**
  - A symmetric matrix always has real eigenvalues and orthogonal eigenvectors.
  - If $$A$$ is symmetric, it is always diagonalizable.
- **Relevance in Machine Learning:**
  - **Covariance Matrices:** Covariance matrices are always symmetric because the covariance between two features is the same regardless of the order.
  - **Optimization Problems:** Many optimization problems in machine learning involve symmetric matrices (e.g., in second-order optimization methods like Newton's method or in regularization).

#### 2. **Orthogonal Matrix:**

- **Definition:** A matrix $$A$$ is **orthogonal** if $$A^T A = I$$, where $$I$$ is the identity matrix.
- **Properties:**
  - The rows and columns of an orthogonal matrix are orthonormal (i.e., they are both orthogonal and of unit length).
  - The inverse of an orthogonal matrix is equal to its transpose $$(A^{-1} = A^T)$$.
  - The determinant of an orthogonal matrix is either +1 or -1.
- **Relevance in Machine Learning:**
  - **Rotation and Transformation:** Orthogonal matrices are used in certain machine learning algorithms for transformations that preserve distances and angles. For example, in PCA, orthogonal transformation is used to create new orthogonal basis vectors.

#### 3. **Positive Definite Matrix (PD):**

- **Definition:** A square matrix $$A$$ is **positive definite** if for any non-zero vector $$\mathbf{v}$$, $$\mathbf{v}^T A \mathbf{v} > 0$$. In simpler terms, it means that the matrix has strictly positive eigenvalues.
- **Properties:**
  - All eigenvalues are positive.
  - The matrix is invertible (non-singular).
  - It implies that the quadratic form is always positive.
- **Relevance in Machine Learning:**
  - **Optimization Problems:** In convex optimization, the Hessian matrix of a convex function is often positive definite. This ensures that a function has a unique local minimum, making optimization well-posed.
  - **Covariance Matrices:** The covariance matrix of any dataset with multiple features is positive semi-definite. In special cases (e.g., full rank), it can be positive definite.

#### 4. **Positive Semi-Definite Matrix (PSD):**

- **Definition:** A matrix $$A$$ is **positive semi-definite** if for any vector $$\mathbf{v}$$, $$\mathbf{v}^T A \mathbf{v} \geq 0$$. In other words, all eigenvalues are non-negative (i.e., zero or positive).
- **Properties:**
  - Eigenvalues are non-negative $$(\lambda_i \geq 0)$$.
  - The matrix may not be invertible if it has zero eigenvalues.
- **Relevance in Machine Learning:**
  - **Covariance Matrices:** As mentioned, the covariance matrix of a dataset is positive semi-definite. This is essential because covariance cannot be negative and the matrix represents the relationship between features.
  - **Kernel Matrices (in SVMs, Gaussian Processes, etc.):** The kernel matrix in algorithms like SVM and kernel PCA is always positive semi-definite. It measures similarity between data points in a transformed feature space.

#### **5. Covariance Matrix:**

- **Definition:** A **covariance matrix** is a square matrix that contains the covariances between pairs of features in a dataset. If a dataset has $$n$$ features, the covariance matrix will be an $$n \times n$$ matrix, where each entry represents the covariance between two features.

- **Covariance of two variables XXX and YYY:**

  $$
  \text{Cov}(X, Y) = \frac{1}{m} \sum_{i=1}^{m} (x_i - \bar{x})(y_i - \bar{y})
  $$

  where $$m$$ is the number of data points, and $$\bar{x}$$ and $$\bar{y}$$​ are the means of the features $$X$$ and $$Y$$, respectively.
- **Covariance Matrix Definition:** For a dataset with nnn features, the covariance matrix $$\Sigma$$ is an $$n \times n$$ matrix where each entry is:

  $$
  \Sigma_{ij} = \text{Cov}(X_i, X_j)
  $$

  where $$X_i$$​ and $$X_j$$​ are the $$i$$-th and $$j$$-th features, respectively.
- **Properties:**
  - **Symmetry:** The covariance matrix is always symmetric, i.e., $$\Sigma_{ij} = \Sigma_{ji}$$
  - **Positive Semi-Definiteness (PSD):** The covariance matrix is always positive semi-definite, meaning for any vector $$\mathbf{v}$$, $$\mathbf{v}^T \Sigma \mathbf{v} \geq 0$$.
  - **Diagonal Entries (Variance):** The diagonal entries represent the variance of individual features.
  - **Off-Diagonal Entries (Covariance):** The off-diagonal entries represent the covariance between different features. Positive covariance indicates that the features increase or decrease together, while negative covariance suggests they move inversely.
  - **Eigenvalues and Eigenvectors:** The eigenvectors of the covariance matrix represent the directions of maximum variance, while the eigenvalues represent the magnitude of variance along these directions.
  - **Rank:** The rank of the covariance matrix corresponds to the number of **independent** features. If the matrix is rank-deficient, it indicates linearly dependent features.
- **Relevance in Machine Learning:**
  - **Principal Component Analysis (PCA):** In PCA, the covariance matrix is used to identify the directions (eigenvectors) of maximum variance in the dataset. Eigenvalues indicate how much variance is explained by each principal component. This helps in dimensionality reduction by selecting the most important components.
  - **Multivariate Gaussian Distribution:** In probabilistic models like **Gaussian Mixture Models (GMM)**, the covariance matrix defines the shape of the data distribution. It is used to model the distribution of features in a multi-dimensional space.
  - **Feature Selection:** Covariance matrices help identify correlated features. Features that show high covariance (i.e., strong correlation) can be dropped or combined to improve model performance and reduce dimensionality.

#### 6. **Full Rank Matrix:**

- **Definition:** A matrix is **full rank** if its rank is equal to the smallest of its number of rows or columns. In other words, all rows (or columns) are linearly independent.
- **Properties:**
  - A matrix $$A$$ with full rank has no redundant or dependent rows or columns.
  - If $$A$$ is an $$m \times n$$ matrix, and $$\text{rank}(A) = \min(m, n)$$, the matrix is full rank.
  - A full rank matrix is **invertible** if it is square (i.e., if $$m = n$$).
- **Relevance in Machine Learning:**
  - **Linear Regression:** In linear regression, the design matrix $$X$$ must be full rank to ensure a unique solution. If $$X$$ is not full rank, the matrix $$X^T X$$ is singular and cannot be inverted.
  - **Solving Systems of Equations:** Full rank matrices guarantee that a system of linear equations has a unique solution. This is important for models that involve solving for weights (like in linear regression or neural networks).

#### 7. **Singular Matrix:**

- **Definition:** A matrix is **singular** if it is not invertible, meaning its determinant is zero. A singular matrix has linearly dependent rows or columns.
- **Properties:**
  - The determinant of a singular matrix is 0.
  - The matrix has at least one eigenvalue equal to 0.
- **Relevance in Machine Learning:**
  - **Linear Dependence:** If the feature matrix $$X$$ in a linear model is singular, some features are perfectly correlated, and this leads to instability in training and difficulties in solving for the model parameters.

---

That wraps up the key linear algebra concepts for machine learning. This post is designed as a quick reference rather than an exhaustive guide. Don't stress about memorizing everything—focus instead on understanding the concepts and knowing when to revisit them if needed.

Math is a language, and like any language, it's more about learning to use it than memorizing rules. Treat this as a foundation to build on, and come back to refresh your knowledge whenever necessary.

Up next, we'll explore the prerequisites of **probability theory** for machine learning. Since probability can often feel trickier, we'll focus more on "what," "why," and "how" questions to make the concepts intuitive and approachable.

See you in the next one!

### **References**:
