---
layout: post
title: Online Learning in ML - A Beginner’s Guide to Adaptive Learning
date: 2025-01-23 10:45:00-0400
featured: false
description: Learn how online learning transforms machine learning by handling dynamic, real-time data and adversarial scenarios. Explore its advantages, real-world applications, and key concepts like regret minimization and the Halving Algorithm in this beginner-friendly guide to adaptive AI.
tags: ML Math
categories: ADV-ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

In this post, we’ll dive into one of the foundational topics that was discussed in the first class of my advanced machine learning course: online learning. This course is centered on theoretical insights and encourages students to think critically, experiment fearlessly, and embrace confusion as a stepping stone toward innovation. Let’s explore how online learning fits into the broader landscape of machine learning and why it’s such a powerful concept.

##### **The Advanced ML Course: A Brief Overview**

The course takes a deeply theoretical approach, focusing on critical analysis and research to build innovative machine learning algorithms. It’s not just about solving problems but about challenging established ideas, learning from mistakes, and having the courage to be wrong. Confusion, in this context, is not a roadblock—it’s a catalyst for deeper thinking.

Two primary domains form the core of present ML/AI space:
1. **Neural Networks**: The cornerstone of modern machine learning.
2. **Online Learning**: A versatile and influential approach with deep connections to game theory and optimization.

#### **What is Online Learning?**

Online learning stands out as an area of machine learning with rich literature and numerous practical applications. It bridges the gap between supervised learning and game-theoretic optimization while offering efficient solutions for large-scale problems.

Unlike traditional batch learning, where algorithms process the entire dataset at once, online learning operates iteratively, processing one sample at a time. This makes it computationally efficient and ideal for large datasets. Moreover, online learning does not rely on the common assumption that data points are independent and identically distributed (i.i.d.). Instead, it is designed to handle adversarial scenarios, making it incredibly flexible and applicable to situations where data distributions are unknown or variable.

Even though these algorithms are inherently designed for adversarial settings, they can, under specific conditions, yield accurate predictions in scenarios where data does follow a distribution.


**What do we mean by adversarial in the context of online learning?**

In online learning, "adversarial" refers to settings where data is not assumed to follow a fixed probabilistic distribution. Instead, the data sequence might be unpredictable, dependent, or deliberately chosen to challenge the algorithm. This flexibility makes online learning particularly robust in real-world applications.

- **Real-world Examples**:
   - **Financial Models**: Adapting to volatile or externally influenced stock price movements.
   - **Recommendation Systems**: Managing biased or strategically influenced user feedback.
   - **Security Systems**: Responding to data manipulations or malicious attacks.

By focusing on resilience and adaptability, online learning algorithms excel in handling evolving data and challenging environments, making them indispensable for a wide range of applications.

##### **Why Online Learning?**

Traditional machine learning approaches often rely on the PAC (Probably Approximately Correct) framework, where:
- The data distribution remains fixed over time.
- Both training and testing data are assumed to follow the same i.i.d. distribution.


**What is the PAC Learning Framework?**

The PAC learning framework provides a theoretical foundation for understanding the feasibility of learning in a probabilistic setting. Under this framework:
- The algorithm's goal is to find a hypothesis that is *probably approximately correct*, meaning it performs well on the training data and generalizes to unseen data with high probability.
- It assumes that data points are drawn independently and identically distributed (i.i.d.) from a fixed, unknown distribution.
- Key metrics include the error rate of the hypothesis on future samples and its convergence to the true distribution as more data is provided.

While this framework is powerful for traditional batch learning, it relies on strong assumptions about the stability and predictability of the data distribution, making it less suitable for dynamic or adversarial scenarios.


In contrast, online learning assumes no such distributional stability. It operates under the following key principles:

- **No Assumptions on Data Distribution**: The data can follow any sequence, including adversarially generated ones. This flexibility allows online learning to adapt to real-world scenarios where data patterns may shift unpredictably.  
- **Mixed Training and Testing**: Training and testing are not separate phases but occur simultaneously, enabling the algorithm to continuously learn and improve from new data.  
- **Worst-Case Analysis**: Algorithms are designed to perform well even under the most challenging conditions, ensuring robustness in unpredictable environments.  
- **Performance Metrics**: Instead of accuracy or loss functions commonly used in batch learning, online learning evaluates performance using measures like:  
  - ***Mistake Model***: The total number of incorrect predictions made during the learning process.  
  - ***Regret***: The difference between the cumulative loss of the algorithm and the loss of the best possible strategy in hindsight.

**What are some practical applications of online learning?**

Online learning has proven invaluable in a variety of real-world domains where data is dynamic, unpredictable, or arrives sequentially:

1. **Stock Market Predictions**: Continuously adapting to ever-changing financial data, helping traders and financial systems make real-time decisions.  
2. **Online Advertising**: Personalizing ads based on user behavior that evolves with every click or interaction.  
3. **Recommendation Systems**: Adapting suggestions in real time as users interact with platforms like Netflix, Amazon, or YouTube.  
4. **Autonomous Systems**: Enabling self-driving cars or robots to learn and adapt to new scenarios as they encounter them.  
5. **Spam Filtering**: Continuously updating filters to catch new spam types as they emerge.  
6. **Security Systems**: Responding to cyberattacks or new threats by learning and adapting on the fly.  


This shift in perspective allows online learning to address a broader range of real-world problems. For now, if all of this feels a bit abstract, don’t worry—hang tight! We’ll dive deeper and make sure to explore it thoroughly, leaving no stone unturned.


#### **The General Online Learning Framework**

The online learning process follows a simple yet powerful framework. At each step:
1. The algorithm receives an instance, denoted as $$x_t$$.
2. It makes a prediction, $$\hat{y}_t$$.
3. The true label, $$y_t$$, is revealed.
4. A loss is incurred, calculated as $$L(\hat{y}_t, y_t)$$, which quantifies the prediction error.

The overarching goal of online learning is to minimize the total loss over a sequence of predictions:
$$
\sum_{t=1}^T L(\hat{y}_t, y_t)
$$

For classification tasks, a common choice of loss is the 0-1 loss:
$$
L(\hat{y}_t, y_t) = \mathbb{1}(\hat{y}_t \neq y_t) \; or \; \vert \hat{y}_t - y_t \vert
$$
For regression tasks, the squared loss is often used:
$$
L(\hat{y}_t, y_t) = (\hat{y}_t - y_t)^2
$$


---


#### **Prediction with Expert Advice**

One particularly compelling framework in online learning is **Prediction with Expert Advice**. Imagine you have multiple "experts," each providing advice on how to predict the label for a given instance. The challenge lies in aggregating their advice to make accurate predictions while minimizing the regret associated with poor decisions.

The process unfolds as follows:
1. At each time step, the algorithm receives an instance, $$x_t$$, and predictions from $$N$$ experts, $$\{y_{t,1}, y_{t,2}, \dots, y_{t,N}\}$$.
2. Based on this advice, the algorithm predicts $$\hat{y}_t$$.
3. The true label, $$y_t$$, is revealed, and the loss, $$L(\hat{y}_t, y_t)$$, is incurred.

The performance of the algorithm is measured by its ***regret***, which is the difference between the total loss incurred by the algorithm and the total loss of the best-performing expert:

$$
\text{Regret}(T) = \sum_{t=1}^T L(\hat{y}_t, y_t) - \min_{i=1, \dots, N} \sum_{t=1}^T L(\hat{y}_{t,i}, y_t)
$$

Minimizing regret ensures that the algorithm's predictions improve over time and closely approximate the performance of the best expert.


**What does the regret equation convey and how do we interpret it?**

The regret equation provides a way to evaluate the algorithm's performance in hindsight by comparing it to the best expert. Here’s what each term means:

1. **Algorithm's Loss** ($$\sum_{t=1}^T L(\hat{y}_t, y_t)$$):  
   This is the cumulative loss incurred by the algorithm over $$T$$ time steps. It reflects how well the algorithm performs when making predictions based on the aggregated advice of all experts.

2. **Best Expert's Loss** ($$\min_{i=1, \dots, N} \sum_{t=1}^T L(\hat{y}_{t,i}, y_t)$$):  
   This represents the cumulative loss of the single best-performing expert in hindsight. Note that the best expert is identified after observing all $$T$$ instances, which gives it an advantage over the algorithm that has to predict in real time.

3. **Regret**:  
   The difference between these two terms quantifies how much worse the algorithm performs compared to the best expert.  
   - **Low regret** indicates that the algorithm's predictions are close to those of the best expert, demonstrating effective learning.  
   - **High regret** suggests that the algorithm is failing to learn effectively from the experts' advice.


**Why is regret important?**

Regret is a crucial metric in online learning because:
- It provides a measure of how well the algorithm adapts to the expert advice over time.
- It ensures that, as the number of time steps $$T$$ increases, the algorithm's performance converges to that of the best expert (ideally achieving sublinear regret, such as $$O(\sqrt{T})$$ or better).
- It accounts for the dynamic nature of predictions, focusing on learning improvement rather than static accuracy.

A few more questions to make our understadning better.


**How to Calculate the Best Expert's Loss?**

The **Best Expert’s Loss** is the cumulative loss of the single expert that performs best over the entire sequence of predictions, $$T$$. Here's how to calculate it:

1. **Track each expert's cumulative loss**:  
   For each expert $$i$$, maintain a running sum of their losses over the rounds:

   $$
   L_{\text{expert } i} = \sum_{t=1}^T L(\hat{y}_{t,i}, y_t)
   $$

   Here, $$\hat{y}_{t,i}$$ is the prediction made by expert $$i$$ at time $$t$$, and $$y_t$$ is the true label. $$L$$ could represent any loss function, such as zero-one loss or squared loss.

2. **Find the expert with the minimum cumulative loss**:  
   After summing the losses for all $$N$$ experts over $$T$$ rounds, identify the expert whose cumulative loss is the smallest:

   $$
   \min_{i=1, \dots, N} \sum_{t=1}^T L(\hat{y}_{t,i}, y_t)
   $$

   This value represents the **Best Expert’s Loss**, which serves as the benchmark for evaluating the algorithm's regret.


**Do We Pick the Best Expert After Each Round?**

No, the **Best Expert’s Loss** is determined in hindsight, ***after*** observing the entire sequence of $$T$$ rounds. The algorithm does not know in advance which expert is the best. Instead, it aggregates predictions from all experts during the process (e.g., using techniques like weighted averaging).

- The **best expert** is identified retrospectively after all rounds.
- The cumulative loss of this best expert is used to compute regret.


**Example:**

Suppose we have 3 experts, and their losses over 5 rounds are:

| Round (t) | Expert 1 Loss | Expert 2 Loss | Expert 3 Loss |
|-----------|---------------|---------------|---------------|
| 1         | 0.2           | 0.3           | 0.1           |
| 2         | 0.1           | 0.4           | 0.2           |
| 3         | 0.3           | 0.2           | 0.3           |
| 4         | 0.4           | 0.1           | 0.3           |
| 5         | 0.2           | 0.5           | 0.1           |

1. **Calculate the cumulative loss for each expert**:
   - **Expert 1**: $$0.2 + 0.1 + 0.3 + 0.4 + 0.2 = 1.2$$
   - **Expert 2**: $$0.3 + 0.4 + 0.2 + 0.1 + 0.5 = 1.5$$
   - **Expert 3**: $$0.1 + 0.2 + 0.3 + 0.3 + 0.1 = 1.0$$

2. **Find the minimum cumulative loss**:
   $$
   \min(1.2, 1.5, 1.0) = 1.0
   $$

   Hence, the **Best Expert’s Loss** is **1.0**, achieved by Expert 3.


Prediction with Expert Advice is a powerful framework for dynamic environments where multiple sources of information or strategies need to be combined effectively. It ensures robustness and adaptability by iteratively improving predictions while minimizing regret.

---

#### **The Halving Algorithm: Simple and Powerful**

The **Halving Algorithm** is a simple yet effective online learning algorithm designed to minimize mistakes. It works by maintaining a set of hypotheses (or experts) and systematically eliminating those that make incorrect predictions.

Here’s how it works:
1. **Initialization**: Start with a set of hypotheses, $$H_1 = H$$.
2. **Iteration**: At each time step, $$t$$:
   - Receive an instance, $$x_t$$.
   - Predict the label, $$\hat{y}_t$$, using majority voting among the hypotheses in $$H_t$$.
   - Receive the true label, $$y_t$$.
   - If $$\hat{y}_t \neq y_t$$, update the hypothesis set:
     $$
     H_{t+1} = \{h \in H_t : h(x_t) = y_t\}
     $$
3. **Termination**: After all iterations, return the final hypothesis set, $$H_{T+1}$$.

##### **Mistake Bound for the Halving Algorithm**

**Theorem**: If the initial hypothesis set $$H$$ is finite, the number of mistakes made by the Halving Algorithm is bounded by:

$$
M_{Halving(H)} \leq \log_2 |H|
$$

**Proof Outline**:
- Each mistake reduces the size of the hypothesis set by at least half:
  $$
  |H_{t+1}| \leq \frac{|H_t|}{2}
  $$
- Initially, $$|H_1| = |H|$$. After $$M$$ mistakes:
  $$
  |H_{M+1}| \leq \frac{|H|}{2^M}
  $$
- To ensure $$|H_{M+1}| \geq 1$$ (at least one hypothesis remains), we require:
  $$
  M \leq \log_2 |H|
  $$

This logarithmic bound demonstrates the efficiency of the Halving Algorithm, even in adversarial settings.


---

##### **Conclusion**

Online learning offers a powerful framework for making predictions in dynamic and adversarial environments. Its ability to adapt, operate under minimal assumptions, and deliver robust performance makes it a cornerstone of modern machine learning research. The Halving Algorithm provides a concrete example of how online learning methods can be both intuitive and theoretically grounded.

In upcoming posts, we’ll delve deeper into other online learning algorithms and explore their theoretical guarantees, practical applications, and connections to broader machine learning principles. Stay tuned!


##### **References**
- PAC
- Online Learning Intro



