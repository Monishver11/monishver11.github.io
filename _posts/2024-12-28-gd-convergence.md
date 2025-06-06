---
layout: post
title: Gradient Descent Convergence - Prerequisites and Detailed Derivation
date: 2024-12-28 21:44:00-0400
featured: false
description: Understanding the convergence of gradient descent with a fixed step size and proving its rate of convergence for convex, differentiable functions.
tags: ML
categories: ML-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

To understand the **Convergence Theorem for Fixed Step Size**, it is essential to grasp a few foundational concepts like **Lipschitz continuity** and **convexity**. This section introduces these concepts and establishes the necessary prerequisites.

**Quick note:** If you find yourself struggling with any part or step, don't worry—just copy and paste it into ChatGPT or Perplexity for an explanation. In most cases, you’ll be able to grasp the concept and move forward. If you're still stuck, feel free to ask for help. The key is not to let small obstacles slow you down—keep going and seek assistance when needed!

---

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


##### **Theorem:**

Suppose the function $$ f : \mathbb{R}^n \to \mathbb{R} $$ is convex and differentiable, and its gradient is Lipschitz continuous with constant $$ L > 0 $$, i.e.,

$$
\|\nabla f(x) - \nabla f(y)\|_2 \leq L \|x - y\|_2 \quad \text{for any} \quad x, y.
$$

Then, if we run gradient descent for $$ k $$ iterations with a fixed step size $$ t \leq \frac{1}{L} $$, the solution $$ x^{(k)} $$ satisfies:

$$
f(x^{(k)}) - f(x^*) \leq \frac{\|x^{(0)} - x^*\|_2^2}{2 t k},
$$

where $$ f(x^*) $$ is the optimal value.

##### **Proof:**

###### **Step 1: Lipschitz Continuity and Smoothness**

From the Lipschitz continuity of $$ \nabla f $$, the function $$ f $$ satisfies the following inequality for any $$ x, y \in \mathbb{R}^n $$:

$$
f(y) \leq f(x) + \nabla f(x)^T (y - x) + \frac{L}{2} \|y - x\|_2^2.
$$

This inequality allows us to bound how the function $$ f $$ changes as we move from $$ x $$ to $$ y $$, given the Lipschitz constant $$ L $$.

###### **Step 2: Gradient Descent Update**

The gradient descent update step is defined as:

$$
x^{+} = x - t \nabla f(x),
$$

where $$ t $$ is the step size. Letting $$ y = x^+ $$ in the smoothness inequality gives:

$$
f(x^+) \leq f(x) + \nabla f(x)^T (x^+ - x) + \frac{L}{2} \|x^+ - x\|_2^2.
$$

###### **Step 3: Substituting the Update Rule**

Substituting $$ x^+ - x = -t \nabla f(x) $$, we get:

$$
f(x^+) \leq f(x) + \nabla f(x)^T (-t \nabla f(x)) + \frac{L}{2} \| -t \nabla f(x)\|_2^2.
$$

Simplifying each term:

- The second term simplifies to:

$$
\nabla f(x)^T (-t \nabla f(x)) = -t \|\nabla f(x)\|_2^2.
$$

- The third term simplifies to:

$$
\frac{L}{2} \| -t \nabla f(x)\|_2^2 = \frac{L t^2}{2} \|\nabla f(x)\|_2^2.
$$

Combining these, we have:

$$
f(x^+) \leq f(x) - t \|\nabla f(x)\|_2^2 + \frac{L t^2}{2} \|\nabla f(x)\|_2^2.
$$

Factoring out $$ \|\nabla f(x)\|_2^2 $$:

$$
f(x^+) \leq f(x) - \left( t - \frac{L t^2}{2} \right) \|\nabla f(x)\|_2^2.
$$

###### **Step 4: Ensuring Decrease in $$ f(x) $$**

To ensure that the function value decreases at each iteration, the coefficient $$ t - \frac{L t^2}{2} $$ must be non-negative. This holds when $$ t \leq \frac{1}{L} $$. Substituting $$ t = \frac{1}{L} $$, we verify:

$$
t - \frac{L t^2}{2} = \frac{1}{L} - \frac{L}{2} \cdot \frac{1}{L^2} = \frac{1}{L} - \frac{1}{2L} = \frac{1}{2L}.
$$

Thus, with $$ t \leq \frac{1}{L} $$, the function value strictly decreases:

$$
f(x^+) \leq f(x) - \frac{t}{2} \|\nabla f(x)\|_2^2.
$$

###### **Step 5: Bounding $$ f(x^+) - f(x^*) $$**
From the convexity of $$ f $$, we know:

$$
f(x^*) \geq f(x) + \nabla f(x)^T (x^* - x).
$$

Rearranging:
$$
f(x) \leq f(x^*) + \nabla f(x)^T (x - x^*).
$$

Substituting this into the inequality for $$ f(x^+) $$:

$$
f(x^+) \leq f(x^*) + \nabla f(x)^T (x - x^*) - \frac{t}{2} \|\nabla f(x)\|_2^2.
$$

Rearranging terms:

$$
f(x^+) - f(x^*) \leq \frac{1}{2t} \left( \|x - x^*\|_2^2 - \|x^+ - x^*\|_2^2 \right).
$$

This shows how the objective value at $$ x^+ $$ is related to the distance between $$ x $$ and the optimal solution $$ x^* $$.

###### **Step 6: Summing Over $$ k $$ Iterations**

Let $$ x^{(i)} $$ denote the iterate after $$ i $$ steps. Applying the inequality iteratively, we have:

$$

f(x^{(i)}) - f(x^*) \leq \frac{1}{2t} \left( \|x^{(i-1)} - x^*\|_2^2 - \|x^{(i)} - x^*\|_2^2 \right).

$$

Summing over $$ i = 1, 2, \dots, k $$:

$$

\sum_{i=1}^k \left( f(x^{(i)}) - f(x^*) \right) \leq \frac{1}{2t} \sum_{i=1}^k \left( \|x^{(i-1)} - x^*\|_2^2 - \|x^{(i)} - x^*\|_2^2 \right).

$$

###### **Step 7: Telescoping Sum**

The terms on the right-hand side form a telescoping sum:

$$
\sum_{i=1}^k \left( \|x^{(i-1)} - x^*\|_2^2 - \|x^{(i)} - x^*\|_2^2 \right) = \|x^{(0)} - x^*\|_2^2 - \|x^{(k)} - x^*\|_2^2.
$$

Thus, we have:

$$
\sum_{i=1}^k \left( f(x^{(i)}) - f(x^*) \right) \leq \frac{1}{2t} \left( \|x^{(0)} - x^*\|_2^2 - \|x^{(k)} - x^*\|_2^2 \right).
$$

Since $$ f(x^{(i)}) $$ is decreasing with each iteration, the largest term dominates the average:

$$
f(x^{(k)}) - f(x^*) \leq \frac{1}{k} \sum_{i=1}^k \left( f(x^{(i)}) - f(x^*) \right).
$$

But, why is the above inequality right? Let's find out:

The inequality

$$
f(x^{(k)}) - f(x^*) \leq \frac{1}{k} \sum_{i=1}^k \left( f(x^{(i)}) - f(x^*) \right)
$$

is derived based on the property that $$ f(x^{(i)}) $$ is **monotonically decreasing** during gradient descent. Let’s break it down step by step.

- **Key Property: Monotonic Decrease**: In gradient descent, the function value decreases with each iteration due to the fixed step size $$ t \leq \frac{1}{L} $$. This means:

$$
f(x^{(1)}) \geq f(x^{(2)}) \geq \cdots \geq f(x^{(k)}).
$$

Thus, the latest value $$ f(x^{(k)}) $$ is the smallest among all iterations.

- **Averaging the Function Values**: The sum of the differences $$ f(x^{(i)}) - f(x^*) $$ over all $$ k $$ iterations can be written as:

$$
\frac{1}{k} \sum_{i=1}^k \left( f(x^{(i)}) - f(x^*) \right),
$$

which represents the average difference between the function values at each iteration and the optimal value $$ f(x^*) $$.

- **Bounding the Smallest Term by the Average**: Since $$ f(x^{(k)}) $$ is the smallest value (due to monotonic decrease), it cannot exceed the average value. In mathematical terms:

$$
f(x^{(k)}) - f(x^*) \leq \frac{1}{k} \sum_{i=1}^k \left( f(x^{(i)}) - f(x^*) \right).
$$

- **Intuition Behind the Inequality**: This inequality reflects a simple fact: the smallest value in a decreasing sequence of numbers is less than or equal to their average. For example, if we have values $$ 10, 8, 7, 6 $$, the smallest value (6) will always be less than or equal to the average of these values.

- **Significance in Gradient Descent**: This inequality is important because it allows us to bound the final iterate $$ f(x^{(k)}) $$ using the sum of all previous iterations. 

###### **Step 8: Final Substitution to Derive the Convergence Result**

From the telescoping sum, we have:

$$
\sum_{i=1}^k \left( f(x^{(i)}) - f(x^*) \right) \leq \frac{1}{2t} \left( \|x^{(0)} - x^*\|_2^2 - \|x^{(k)} - x^*\|_2^2 \right).
$$

Using the inequality:

$$
f(x^{(k)}) - f(x^*) \leq \frac{1}{k} \sum_{i=1}^k \left( f(x^{(i)}) - f(x^*) \right),
$$

we substitute the bound on the sum into this expression:

$$
f(x^{(k)}) - f(x^*) \leq \frac{1}{k} \cdot \frac{1}{2t} \left( \|x^{(0)} - x^*\|_2^2 - \|x^{(k)} - x^*\|_2^2 \right).
$$

Since $$ \|x^{(k)} - x^*\|_2^2 \geq 0 $$, we drop this term to get the worst-case bound:

$$
f(x^{(k)}) - f(x^*) \leq \frac{\|x^{(0)} - x^*\|_2^2}{2tk}.
$$


##### **Conclusion**

We have derived the convergence guarantee for gradient descent with a fixed step size $$ t \leq \frac{1}{L} $$. The final result:

$$
f(x^{(k)}) - f(x^*) \leq \frac{\|x^{(0)} - x^*\|_2^2}{2 t k},
$$

shows that the function value $$ f(x^{(k)}) $$ decreases towards the optimal value $$ f(x^*) $$ at a rate proportional to $$ O(1/k) $$. This rate depends on the step size $$ t $$ and the initial distance $$ \|x^{(0)} - x^*\|_2^2 $$.

The result highlights that gradient descent converges reliably under the conditions of convexity, differentiability, and Lipschitz continuity of the gradient. As $$ k \to \infty $$, the function value approaches the optimal value, demonstrating the effectiveness of gradient descent for optimization problems with these properties.


---
Next, 
- Convergence of gradient descent with adaptive step size
- Strongly convex - "linear convergence" rate

#### **Convergence of gradient descent with adaptive step size**

In the above section, we derived the convergence rate for gradient descent with a **fixed step size**. In this part, we extend this analysis to the case where the step size is chosen adaptively using a **backtracking line search**. This method ensures that the step size decreases as necessary to guarantee sufficient decrease in the objective function at each iteration.

###### **Step 1: Setup and Assumptions**

Consider a differentiable convex function $$ f: \mathbb{R}^n \to \mathbb{R} $$ with a **Lipschitz continuous gradient**. That is, for any two points $$ x, y \in \mathbb{R}^n $$,

$$
\|\nabla f(x) - \nabla f(y)\|_2 \leq L \|x - y\|_2,
$$

where $$ L $$ is the **Lipschitz constant** of the gradient.

Let $$ x^* $$ be the minimizer of $$ f $$, and let $$ x^{(i)} $$ represent the iterates of gradient descent. The update rule for gradient descent with backtracking line search is:

$$
x^{(i+1)} = x^{(i)} - t_i \nabla f(x^{(i)}),
$$

where $$ t_i $$ is the step size at iteration $$ i $$, chosen adaptively using the backtracking procedure.

###### **Step 2: Descent Lemma**

In the case of gradient descent with a **fixed step size** $$ t $$, we know from the **descent lemma** (for smooth convex functions) that:

$$
f(x^{(i+1)}) \leq f(x^{(i)}) - t \|\nabla f(x^{(i)})\|_2^2 + \frac{L}{2} t^2 \|\nabla f(x^{(i)})\|_2^2.
$$

This inequality states that at each iteration, the function value decreases by a term proportional to the gradient's squared norm, and this decrease depends on the step size $$ t $$.

###### **Step 3: Backtracking Line Search**

With **backtracking line search**, the step size $$ t_i $$ is chosen at each iteration to ensure sufficient decrease in the function value. Specifically, the step size is selected such that:

$$
f(x^{(i+1)}) \leq f(x^{(i)}) + \alpha t_i \nabla f(x^{(i)})^T \nabla f(x^{(i)}),
$$

where $$ 0 < \alpha < 1 $$ is a constant. The backtracking line search ensures that $$ t_i $$ satisfies the condition:

$$
t_i \leq \frac{1}{L}.
$$

Thus, the step size at each iteration is bounded by $$ \frac{1}{L} $$, which prevents the gradient from changing too rapidly and ensures that the update does not overshoot the optimal point.


**Why "Adaptive"?**

The step size is called **adaptive** because it changes at each iteration depending on the function’s behavior. If the function is steep or the gradient is large, the backtracking line search may choose a smaller step size to avoid overshooting. If the function is shallow or the gradient is small, it might allow a larger step size. This adaptive process uses a parameter $$ \beta $$ to control how the step size is reduced when the decrease condition is not met.

  

###### **Step 4: Backtracking Process and $$ \beta $$**

The process of backtracking works as follows:

- **Initial Step Size**: Start with an initial guess for the step size, typically $$ t_0 = 1 $$.

- **Condition Check**: Check whether the condition

$$
f(x^{(i+1)}) \leq f(x^{(i)}) + \alpha t_i \nabla f(x^{(i)})^T \nabla f(x^{(i)})
$$

holds. If it does, accept $$ t_i $$; if not, reduce the step size.

- **Reduce Step Size**: If the condition is not satisfied, reduce the step size $$ t_i $$ by a factor $$ \beta $$:

$$
t_{i+1} = \beta t_i,
$$

where $$ \beta $$ is a constant between 0 and 1 (usually around 0.5 or 0.8). This step size reduction continues until the condition is met.

- **Accept the Step Size**: Once the condition is satisfied, the current $$ t_i $$ is accepted for the update.

The use of $$ \beta $$ helps to ensure that the step size does not become too large, allowing the algorithm to converge smoothly without overshooting.


###### **Step 5: Bounding the Convergence**

Now, let's derive the convergence bound for gradient descent with backtracking line search. From the descent lemma, the change in the function value at each iteration can be bounded as:

$$
f(x^{(i+1)}) - f(x^{(i)}) \leq - t_i \|\nabla f(x^{(i)})\|_2^2 \left( 1 - \frac{L}{2} t_i \right).
$$

Because the backtracking line search ensures that $$ t_i \leq t_{\text{min}} = \min\left( 1, \frac{\beta}{L} \right) $$, we can bound the function value decrease as:

$$
f(x^{(i+1)}) - f(x^{(i)}) \leq - t_{\text{min}} \|\nabla f(x^{(i)})\|_2^2 \left( 1 - \frac{L}{2} t_{\text{min}} \right).
$$

This shows that the function value decreases at each iteration, with the step size $$ t_{\text{min}} $$ controlling the rate of decrease.

Now, if you observe carefully, the equation above closely resembles the one we encountered in the fixed step size proof. The only minor difference is that $$ t $$ has been replaced with $$ t_{\text{min}} $$. Therefore, we can follow the same steps as in the fixed step size case and eventually arrive at the following result:

$$
f(x^{(k)}) - f(x^*) \leq \frac{\|x^{(0)} - x^*\|_2^2}{2 t_{\text{min}} k}.
$$

This shows that by adaptively choosing the step size, we can achieve a convergence rate similar to that of the fixed step size approach, but without needing to manually set a fixed value for \( t \).

**Quick Note:** I'm still not completely satisfied with the proof for Adaptive Step Size. I'll be working on refining the explanation further and will update you with any improvements.


##### **And finally...**

We've reached the end of this blog post! A huge kudos to you for making it all the way through and sticking with me. The reason we went through all of this is that understanding such proofs will lay the foundation for exploring the intricate details that drive machine learning and produce its remarkable results. To truly dive into ML research, we need to immerse ourselves in these depths and make it happen. 

So, take a well-deserved break, and in the next post, we’ll delve into the tips and tricks of SGD that are widely practiced in the industry. Until then, take care and see you soon!


##### **References:**
- [ Gradient Descent: Convergence Analysis - Ryan Tibshirani](https://nyu-cs2565.github.io/mlcourse-public/2024-fall/lectures/lec02/gradient_descent_converge.pdf)