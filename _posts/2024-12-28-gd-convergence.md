---
layout: post
title: Gradient Descent Convergence - Prerequisites and Detailed Derivation
date: 2024-12-28 21:44:00-0400
featured: false
description: Understanding the convergence of gradient descent with a fixed step size and proving its rate of convergence for convex, differentiable functions.
tags: ML
categories: sample-posts
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

To understand the **Convergence Theorem for Fixed Step Size**, it is essential to grasp a few foundational concepts like **Lipschitz continuity** and **convexity**. This section introduces these concepts and establishes the necessary prerequisites.

#### **Lipschitz Continuity?**

At its core, Lipschitz continuity imposes a **limit on how fast a function can change**. Mathematically, a function $$ g : \mathbb{R}^d \to \mathbb{R} $$ is said to be **Lipschitz continuous** if there exists a constant $$ L > 0 $$ such that:  

$$
\|g(x) - g(x')\| \leq L \|x - x'\|, \quad \forall x, x' \in \mathbb{R}^d.
$$

This means the function’s rate of change is bounded by $$ L $$. For differentiable functions, Lipschitz continuity is often applied to the gradient. If $$ \nabla f(x) $$ is Lipschitz continuous with constant $$ L > 0 $$, then:

$$
\|\nabla f(x) - \nabla f(x')\| \leq L \|x - x'\|, \quad \forall x, x' \in \mathbb{R}^d.
$$

This ensures the gradient does not change too rapidly, which is crucial for the convergence of optimization algorithms like gradient descent.

###### **Intuition Behind Lipschitz Continuity**

1. **Bounding the Slope**: Lipschitz continuity ensures that the slope of the function (or the steepness of the graph) is bounded by $$ L $$. You can think of it as saying, "No part of the function can change too steeply."
2. **Gradient Smoothness**: For $$ \nabla f(x) $$, Lipschitz continuity means the gradient varies smoothly between nearby points. This avoids abrupt jumps or erratic behavior in the optimization landscape.

###### **Visual Way to Think About It**

Imagine walking along a path represented by the graph of $$ f(x) $$. Lipschitz continuity guarantees:
- No sudden steep hills or cliffs.
- A smooth path where the steepness (gradient) is capped.

Alternatively, picture a **rubber band stretched smoothly over some pegs**. The tension in the rubber band ensures there are no sharp kinks, making the graph smooth and predictable.

###### **Examples of Lipschitz Continuous Functions**
1. **Linear Function**: $$ f(x) = mx + b $$ is Lipschitz continuous because the slope $$ m $$ is constant, and 
$$ 
|f'(x)| = |m| 
$$ 
is bounded.
1. **Quadratic Function**: $$ f(x) = x^2 $$ is $$ L $$-smooth with $$ L = 2 $$. Its gradient $$ f'(x) = 2x $$ satisfies:
   
$$
|f'(x) - f'(x')| = |2x - 2x'| = 2|x - x'|.
$$

1. **Non-Lipschitz Example**: $$ f(x) = \sqrt{x} $$ (for $$ x > 0 $$) is **not Lipschitz continuous** at $$ x = 0 $$ because the slope becomes infinitely steep as $$ x \to 0 $$. (If you're not getting this, just plot $$\sqrt{x}$$ function in [Desmos](https://www.desmos.com/) and you'll get it.)

###### **Why Does Lipschitz Continuity Matter?**

1. **Predictability**: Lipschitz continuity ensures that a function behaves predictably, without sudden spikes or erratic changes.
2. **Gradient Descent**: If $$ \nabla f(x) $$ is Lipschitz continuous, we can choose a step size $$ \eta \leq \frac{1}{L} $$ to ensure gradient descent converges smoothly without overshooting the minimum.

But Why? We'll see that in the Convergence Theorem down below. For now, lets equip ourselves with the next important concept needed.

---

#### **2. Convex Functions and Convexity Condition**

A function $$ f : \mathbb{R}^d \to \mathbb{R} $$ is **convex** if for any $$ x, x' \in \mathbb{R}^d $$ and $$ \alpha \in [0, 1] $$: 

$$
f(\alpha x + (1 - \alpha)x') \leq \alpha f(x) + (1 - \alpha)f(x').
$$

Intuitively, the line segment between any two points on the graph of $$ f $$ lies above the graph itself.

###### **Convexity Condition Using Gradients**

If $$ f $$ is differentiable, convexity is equivalent to the following condition:

$$
f(x') \geq f(x) + \langle \nabla f(x), x' - x \rangle, \quad \forall x, x' \in \mathbb{R}^d.
$$

This means that the function lies above its tangent plane at any point.

---

#### **3. $$ L $$-Smoothness**

A function $$ f $$ is said to be $$ L $$-smooth if its gradient is Lipschitz continuous. This implies the following inequality:

$$
f(x') \leq f(x) + \langle \nabla f(x), x' - x \rangle + \frac{L}{2} \|x' - x\|^2.
$$

This property bounds the change in the function value using the gradient and the distance between $$ x $$ and $$ x' $$.

---

#### **4. Optimality Conditions for Convex Functions**

For convex functions, the following is true:
- If $$ x^* $$ is a minimizer of $$ f $$, then:

$$
\nabla f(x^*) = 0.
$$

- For any $$ x $$, the difference between $$ f(x) $$ and $$ f(x^*) $$ can be bounded using the gradient:

$$
f(x) - f(x^*) \leq \langle \nabla f(x), x - x^* \rangle.
$$

These conditions help in deriving the convergence results for gradient descent.

---

**To quickly summarize, before we proceed further:**

1. **Lipschitz continuity** ensures the gradient does not change too rapidly.
2. **Convexity** guarantees that the function behaves well, with no local minima other than the global minimum.
3. **$$ L $$-smoothness** combines convexity and Lipschitz continuity to bound the function's behavior using gradients.

---

With these concepts in place, we can now proceed to derive the **Convergence Theorem for Fixed Step Size**.

#### **Convergence of Gradient Descent with Fixed Step Size**


Suppose the function $$ f : \mathbb{R}^n \to \mathbb{R} $$ is convex and differentiable, and that its gradient is Lipschitz continuous with constant $$ L > 0 $$, i.e.,

$$
\|\nabla f(x) - \nabla f(y)\|_2 \leq L \|x - y\|_2 \quad \text{for any} \quad x, y.
$$

Then, if we run gradient descent for $$ k $$ iterations with a fixed step size $$ t \leq \frac{1}{L} $$, it will yield a solution $$ x^{(k)} $$ that satisfies:

$$
f(x^{(k)}) - f(x^*) \leq \frac{\|x^{(0)} - x^*\|^2}{2 t k}.
$$

###### **Proof:**

**Step 1: Using the Lipschitz Continuity of the Gradient**
 
Since $$ \nabla f $$ is Lipschitz continuous with constant $$ L $$, we can expand $$ f $$ around a point $$ x $$ using a second-order Taylor expansion. This gives us the following inequality:

$$
f(y) \leq f(x) + \nabla f(x)^T (y - x) + \frac{1}{2} \nabla^2 f(x) \|y - x\|_2^2 \leq f(x) + \nabla f(x)^T (y - x) + \frac{L}{2} \|y - x\|_2^2.
$$

**Step 2: Applying the Gradient Descent Update**

Let $$ y = x^{+} = x - t \nabla f(x) $$, where $$ t $$ is the step size. Substituting this into the inequality above:

$$
f(x^{+}) \leq f(x) + \nabla f(x)^T (x - t \nabla f(x) - x) + \frac{L}{2} \|x - t \nabla f(x) - x\|_2^2.
$$


Simplifying the terms:

$$
f(x^{+}) \leq f(x) - t \|\nabla f(x)\|_2^2 + \frac{L t^2}{2} \|\nabla f(x)\|_2^2.
$$


Thus, we obtain the following inequality:

$$
f(x^{+}) \leq f(x) - \left( 1 - \frac{L t}{2} \right) t \|\nabla f(x)\|_2^2.
$$

**Step 3: Bounding the Objective Value**

Since $$ f(x) $$ is convex, we can use the first-order condition for convexity:

$$
f(x^*) \geq f(x) + \nabla f(x)^T (x^* - x),
$$

which can be rearranged to:

$$
f(x) \leq f(x^*) + \nabla f(x)^T (x - x^*).
$$

Substituting this into the inequality for $$ f(x^{+}) $$, we get:

$$
f(x^{+}) \leq f(x^*) + \nabla f(x)^T (x - x^*) - \frac{t}{2} \|\nabla f(x)\|_2^2.
$$

Rearranging terms:

$$
f(x^{+}) - f(x^*) \leq \frac{1}{2t} \left( \|x - x^*\|_2^2 - \|x^{+} - x^*\|_2^2 \right).
$$

**Step 4: Summing Over All Iterations**

Summing over $$ k $$ iterations, we obtain:

$$
\sum_{i=1}^{k} \left( f(x^{(i)}) - f(x^*) \right) \leq \sum_{i=1}^{k} \frac{1}{2t} \left( \|x^{(i-1)} - x^*\|_2^2 - \|x^{(i)} - x^*\|_2^2 \right).
$$

This sum is telescoping, and thus simplifies to:

$$
\sum_{i=1}^{k} \left( f(x^{(i)}) - f(x^*) \right) \leq \frac{1}{2t} \left( \|x^{(0)} - x^*\|_2^2 - \|x^{(k)} - x^*\|_2^2 \right).
$$

Since $$ f(x^{(i)}) $$ is decreasing with each iteration, we have:

$$
f(x^{(k)}) - f(x^*) \leq \frac{1}{k} \sum_{i=1}^{k} \left( f(x^{(i)}) - f(x^*) \right).
$$

Using the above inequality:

$$
f(x^{(k)}) - f(x^*) \leq \frac{\|x^{(0)} - x^*\|_2^2}{2 t k}.
$$

##### Conclusion

We have derived the convergence bound for gradient descent with a fixed step size $$ t \leq \frac{1}{L} $$. The objective function value decreases at a rate of $$ O(1/k) $$, and the final value $$ f(x^{(k)}) $$ is guaranteed to be close to the optimal value $$ f(x^*) $$, with the rate of convergence depending on the step size $$ t $$ and the initial distance from the optimal solution.

Thus, we have shown that:

$$
f(x^{(k)}) - f(x^*) \leq \frac{\|x^{(0)} - x^*\|_2^2}{2 t k}.
$$

This proves the **Convergence Theorem for Fixed Step Size** in gradient descent.