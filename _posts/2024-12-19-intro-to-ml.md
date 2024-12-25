---
layout: post
title: Introduction to Machine Learning(ML)
date: 2024-12-19 15:44:00-0400
featured: true
description: An easy guide to machine learning, its applications, and how it connects to AI and human learning.
tags: ML
categories: sample-posts
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

What does it mean to learn? The Merriam-Webster dictionary defines learning as “the activity or process of gaining knowledge or skill by studying, practicing, being taught, or experiencing something.” Tom Mitchell, a pioneer in machine learning, extends this concept to machines:

> “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.”

Machine learning (ML) is about teaching machines to learn from data and solve problems. Let’s dive into what this means, how it works, and why it’s transforming the way we use technology.

#### **What is Machine Learning?**

At its core, machine learning is like **meta-programming**—programming a system to program itself. For tasks like recognizing faces or understanding speech, coding specific rules is impractical. Instead, ML lets machines learn directly from data to solve problems such as:

- Predicting whether an email is spam.
- Diagnosing diseases from symptoms.
- Forecasting stock prices.

The goal? Take an input x and predict an output y—it can be a label, a number, or a decision.

---

#### **Why Use Machine Learning?**

Machine learning is particularly valuable when:

- **Rules are too complex to define manually**: Recognizing a face or interpreting speech involves subtleties that are hard to encode in rules.
- **The system needs to adapt**: For instance, spam filters must evolve as spammers devise new tricks.
- **It outperforms human-built solutions**: Algorithms can detect patterns or nuances humans might miss.
- **Fairness and privacy are critical**: For example, ranking search results or filtering harmful content.

### **Canonical Examples of Machine Learning**

1. **Spam Detection**
   - **Input**: Incoming email
   - **Output**: “SPAM” or “NOT SPAM”
   - Problem Type: Binary Classification
2. **Medical Diagnosis**
   - **Input**: Patient symptoms (e.g., fever, cough)
   - **Output**: Diagnosis (e.g., flu, pneumonia)
   - Problem Type: Multiclass Classification
   - Probabilistic Classification: Uncertainty is expressed as probabilities, e.g., P(pneumonia)=0.7
3. **Stock Price Prediction**
   - **Input**: Historical stock prices
   - **Output**: Price at the close of the next day
   - Problem Type: Regression (continuous outputs)

#### **ML vs. Rule-Based Systems**

Before ML, many problems were solved with **rule-based systems** (or expert systems). For instance, medical diagnosis might involve encoding expert knowledge into rules that map symptoms to diseases.

**Strengths of Rule-Based Systems**

- Leverage domain expertise.
- Interpretable and explainable.
- Reliable for known scenarios.

**Weaknesses of Rule-Based Systems**

- Labor-intensive to build and maintain.
- Poor generalization to new or unseen scenarios.
- Struggle with uncertainty and probabilistic reasoning.

#### **How ML Overcomes These Weaknesses**

Instead of encoding rules, ML systems learn directly from **training data**—examples of input-output pairs. For instance:

- Input: Emails
- Output: SPAM or NOT SPAM  
   This approach, called **supervised learning**, involves learning from labeled examples of input-output pairs.

#### **Key Concepts in ML**

- **Common Problem Types**:
  - Classification (binary, multi-class)
  - Regression (continuous prediction)
- **Core Elements**:
  - Prediction function: Maps x (input) to y (output).
  - Training data: A collection of input-output pairs for the model to learn from.
  - Algorithms: Methods to produce the best prediction function from data.
- **Beyond Supervised Learning**:
  - **Unsupervised Learning**: Discovering patterns, like clustering similar users.
  - **Reinforcement Learning**: Optimizing long-term objectives or learning through rewards, e.g., chess and other gameplays.
  - **Representation Learning**: Automatically discovering useful features, e.g., learning word embeddings.

#### **Core Questions in Machine Learning**

1. **Modeling**: What kinds of prediction functions should we consider?
2. **Learning**: How do we find the best prediction function from training data?
3. **Inference**: How do we compute predictions for new inputs?

We'll tackle each of these questions as we move forward, so stick around!

#### **ML vs. Statistics: Key Differences**

While both fields use mathematical tools like calculus, probability, and linear algebra, they differ in focus:

- **Statistics**: Emphasizes interpretability and aiding human decision-making.
- **ML**: Prioritizes scalability, automation, and predictive performance.

#### **ML in AI and Human Learning**

**Relation to AI:**
Machine learning is a crucial subset of artificial intelligence (AI). While AI encompasses a broad range of approaches to simulate human-like intelligence, machine learning focuses specifically on learning patterns from data to make predictions or decisions.

**Relation to Human Learning:**
Though inspired by human cognition, ML differs significantly:

- We humans are highly efficient with limited data.
- Machines require large datasets but excel at specific tasks.

ML systems, like neural networks, borrow ideas from biology but don’t aim to replicate human learning entirely. Instead, their focus remains on solving specialized problems effectively.

#### **Conclusion**

Machine learning bridges the gap between raw data and intelligent decision-making. By enabling systems to learn and adapt, ML transforms how we approach problems in healthcare, finance, and beyond. With its foundations rooted in mathematics and its applications spanning diverse domains, ML continues to redefine the boundaries of what technology can achieve.

That's it for this, see you in the next post! 👋
