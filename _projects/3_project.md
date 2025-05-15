---
layout: page
title: Understanding Swap Regret 2.0
description: This blog post unpacks the "Swap Regret 2.0" paper through slide-by-slide insights, showing how the TreeSwap algorithm closes the gap between external and swap regret, advancing equilibrium computation in game theory and reinforcement learning.
img: assets/img/project_3/p3-1.jpeg
importance: 2
category: Work
giscus_comments: true
toc:
  sidebar: left
---

In this blog post, I'll walk through my presentation on the groundbreaking paper ["From External to Swap Regret 2.0: An Efficient Reduction for Large Action Spaces"](https://arxiv.org/pdf/2310.19786) by Chen et al., which introduces significant advances in online learning algorithms and game theory.

This paper tackles a long-standing problem in the field: the exponential gap between external regret and swap regret in online learning settings. The authors present TreeSwap, a novel algorithm that bridges this gap and enables efficient swap regret minimization even with large or infinite action spaces.

##### **How to use this guide:**

1. **First steps**: I recommend reading pages 1-20 of the [original paper](https://arxiv.org/pdf/2310.19786) to get familiar with the basic concepts and main results.

2. **Follow along**: Open my [presentation slides](https://drive.google.com/file/d/1Iv2VlHRLW3pbN4DWd4CFByxSo1p8kFGP/view?usp=sharing) side-by-side with this blog post. Each section below corresponds to specific slides and elaborates on the key points.

3. **Slide-by-slide explanations**: I've expanded each slide's bullet points into coherent explanations that should make the technical concepts more accessible.

Let's begin by understanding the fundamentals of regret minimization and why this paper represents such an important advancement in the field.

##### **Slide 1: Online Learning Introduction**

No-regret learning is a fundamental paradigm in machine learning where an agent repeatedly makes decisions and aims to minimize its regret over time. What is regret exactly? It measures how much worse the agent performs compared to the best fixed strategy in hindsight. The core idea is that the agent's cumulative regret should grow sublinearly with the number of rounds, meaning that on average, the per-round regret approaches zero as time progresses.

This concept has profound implications for game theory. When players in a game employ no-regret learning algorithms, something remarkable happens: the empirical distribution of their strategies converges to an equilibrium. This is a powerful result that connects online learning to game-theoretic solution concepts.

The type of equilibrium we get depends critically on the specific notion of regret that players minimize. This creates a fascinating relationship between learning dynamics and equilibrium concepts that has been central to algorithmic game theory for decades.

##### **Slide 2: Continuing on No-Regret Learning**

Let's explore the connection between regret types and equilibrium concepts more concretely. When players minimize external regret (comparing their performance to the best single fixed action), their joint play converges to what's called a Coarse Correlated Equilibrium (CCE).

In a CCE, no player can improve their utility by unilaterally switching to a fixed action in hindsight. Essentially, this means that if a mediator recommends actions according to the equilibrium distribution, no player has an incentive to ignore this recommendation and play a different fixed action instead.

CCE is a relaxation of Nash equilibrium. While Nash requires independent actions and mutual best responses, CCE only ensures that no player gains by switching to a different action. The benefit? CCE is computationally much easier to achieve than Nash, especially in large games where computing Nash equilibria becomes intractable.

However, CCE has a significant drawback: the recommended action might reveal information about other players' strategies, which could potentially be exploited. This is why we sometimes prefer a stronger equilibrium concept called Correlated Equilibrium (CE), which is achieved through swap regret minimization.

##### **Slide 3: Swap Regret Challenges with Large Action Spaces**

Now, let's delve into swap regret. Unlike external regret (which compares against the best single action), swap regret measures the extra utility a player could have gained by replacing each action they played with another action according to a fixed transformation rule. It's a more demanding criterion that leads to the stronger Correlated Equilibrium concept.

In a CE, each player maximizes their expected utility against others' action distributions while conditioning on their own sampled action. They must answer: "Given my recommended action, would switching to any other action increase my expected utility?" If the answer is no for all players, we have a Correlated Equilibrium.

The challenge? Swap regret minimization is significantly more computationally demanding than external regret minimization, especially when dealing with large or infinite action spaces. This computational gap has been a major obstacle in applying CE to complex real-world problems.

##### **Slide 4: Expert Settings and Regret Bounds**

Let's examine the computational gap more precisely. In what's called the "Expert Setting," where a learner selects from $$N$$ possible actions each round:

For external regret, the average regret can be bounded by $$\varepsilon$$ after approximately $$T \gtrsim \frac{\log N}{\varepsilon^2}$$ rounds. This is a very efficient scaling – just logarithmic in the number of actions!

In contrast, for swap regret, the best-known bounds required approximately $$T \gtrsim \frac{N \log N}{\varepsilon^2}$$ rounds. Notice that crucial factor of $$N$$ – it means the number of rounds needed scales linearly with the number of actions, creating an exponential gap compared to external regret.

This gap becomes particularly problematic in applications like Poker, Diplomacy, or multi-agent reinforcement learning, which often involve moderate to large action spaces. The computational burden of swap regret minimization has made it less practical for these settings, despite its theoretical advantages.

##### **Slide 5: Bandit Settings and Challenges**

The situation becomes even more challenging in bandit settings, where the learner only observes the reward of the chosen action rather than rewards for all possible actions. In this partial-information scenario, regret minimization generally requires more exploration.

In bandit settings, the best-known algorithms for swap regret require approximately $$T \gtrsim \frac{N^2 \log N}{\varepsilon^2}$$ rounds, while the previous best lower bound suggested that at least $$\frac{N \log N}{\varepsilon^2}$$ rounds are necessary.

This gap has significant implications for reinforcement learning applications, which often operate in bandit-like settings where agents receive feedback only for the actions they take. The quadratic dependence on $$N$$ makes swap regret minimization particularly impractical for large-scale RL problems.

The paper we're discussing aims to close this gap, making swap regret minimization practical even for these challenging settings with large action spaces. The TreeSwap algorithm they introduce represents a major step forward in addressing these long-standing challenges.


##### **Slide 6: Gaps in Equilibrium Computation**

The gaps between swap and external regret aren't just theoretical curiosities—they manifest in significant discrepancies in equilibrium computation across various computational models. In normal-form games with $$N$$ actions per player, these discrepancies become striking when comparing the complexity of computing $$\epsilon$$-approximate Correlated Equilibrium (CE) versus Coarse Correlated Equilibrium (CCE).

For communication complexity, computing $$\epsilon$$-CCE requires $$O(\log^2 N)$$ bits, while $$\epsilon$$-CE needs $$O(N \log^2 N)$$ bits—an exponential increase! Similarly, query complexity jumps from $$O(N \log N)$$ for $$\epsilon$$-CCE to $$O(N^2 \log N)$$ for $$\epsilon$$-CE, making it quadratically worse. Even the sparsity characteristics differ dramatically—a CCE can be represented compactly with $$\text{polylog}(N)$$ sparsity, while the sparsity of CE remains an open question.

These gaps extend beyond normal-form games. In games with Littlestone dimension $$L$$, we can compute $$\epsilon$$-CCE efficiently in $$O(L)$$ rounds, but prior to this paper, it wasn't even clear whether $$\epsilon$$-CE existed in these settings! Similarly, for extensive-form games with tree size description length $$n$$, computing $$\epsilon$$-CCE takes polynomial time, while $$\epsilon$$-CE previously required exponential time.

##### **Slide 7: Littlestone Dimension**

The Littlestone dimension is a critical complexity measure in online learning theory. It generalizes the more familiar VC dimension to sequential prediction settings, helping us characterize when no-regret learning is possible.

In online learning, where examples arrive sequentially rather than all at once, we need a measure of how long an adversary can force a learner to make mistakes before running out of options. The Littlestone dimension precisely measures the length of the longest such sequence.

For a function class $$H$$ (which here represents action spaces), if $$\text{LDim}(H)$$ is finite, then there exists a no-regret online learning algorithm for that class. Conversely, if $$\text{LDim}(H)$$ is infinite, no online algorithm can achieve sublinear regret—meaning effective learning is impossible.

This dimension plays a fundamental role in understanding the difference between batch learning (characterized by VC-dimension) and online learning (characterized by Littlestone dimension). For action spaces with finite Littlestone dimension $$L$$, external regret can be bounded after $$T \geq L/\epsilon^2$$ rounds, but prior to this work, it was unknown whether swap regret could be effectively minimized in such spaces.

##### **Slide 8: Main Results - Near-Optimal Upper and Lower Bounds for Swap Regret**

The paper introduces a groundbreaking reduction that converts any external-regret learning algorithm into a no-distributional swap-regret learner. This is the core innovation presented as "TreeSwap" (Algorithm 1).

Theorem 1.1 states that if there's a learner for some function class $$F$$ that achieves external regret of $$\epsilon$$ after $$M$$ iterations, then TreeSwap achieves swap regret of at most $$\epsilon + 1/d$$ after $$T = M^d$$ iterations. Perhaps most impressively, if the per-iteration runtime complexity of the external-regret learner is $$C$$, then TreeSwap maintains the same $$O(C)$$ per-iteration amortized runtime complexity!

This result is transformative because it means that any no-external-regret algorithm can be efficiently transformed into a no-swap-regret learner, regardless of action space size—even for infinite action spaces. Previous reductions required finite or bounded action spaces, severely limiting their applicability.

The impact is substantial: if a game allows no-external-regret learning, it also allows no-swap-regret learning, leading to significantly better equilibrium computation results and removing the exponential dependence on action space size that previously hampered swap-regret algorithms.

##### **Slide 9: Tree Swap Algorithm**

The TreeSwap algorithm organizes multiple instances of a no-external-regret algorithm in a hierarchical structure using a depth-$$d$$, $$M$$-ary tree. Each node in the tree corresponds to an instance of the external-regret algorithm $$\text{Alg}$$.

For each time step $$t$$, the algorithm selects a mixture of the distributions produced by different instances of $$\text{Alg}$$. The key innovation lies in how these instances are updated: each instance follows a "lazy update" scheme, with instances at level $$h$$ getting updated only every $$M^{d-h}$$ rounds.

This design ensures that each instance of $$\text{Alg}$$ is updated a total of $$M$$ times over the course of $$T = M^d$$ rounds, allowing TreeSwap to leverage the external regret guarantees of the base algorithm while achieving bounded swap regret.

The per-iteration amortized runtime complexity of TreeSwap is $$O(C)$$, where $$C$$ is the per-iteration runtime complexity of the underlying external-regret learner. This efficiency is crucial for practical applications in large action spaces.

##### **Slide 10: Understanding the Setup**

To understand TreeSwap's structure, consider a concrete example: a binary tree ($$M = 2$$) of depth $$d = 3$$. This tree has $$T = 2^3 = 8$$ leaves, each corresponding to a time step $$t$$. Each node in the tree corresponds to a subproblem handled by an instance of the external-regret algorithm $$\text{Alg}$$.

This hierarchical organization helps in learning by averaging utility functions over different levels of the tree. The depth parameter $$d$$ controls the trade-off between the number of rounds $$T$$ and the additional term $$1/d$$ in the swap regret bound.

The arrangement of instances in this tree-like structure is central to TreeSwap's ability to convert external regret guarantees into swap regret guarantees without incurring exponential computational costs in terms of the action space size.

##### **Slide 11: Key Variables**

Understanding TreeSwap requires familiarity with several key variables:

1. $$\sigma_{1:d}$$ represents the base-$$M$$ representation of the time step $$t-1$$. This representation determines the path in the tree for the current round.

2. $$\sigma_{1:h-1}$$ is the prefix of $$\sigma_{1:d}$$ up to depth $$h-1$$. It determines which instance of $$\text{Alg}$$ is being used at a given level of the tree.

3. Algorithm Instances ($$\text{Alg}_{\sigma_{1:h-1}}$$): Each node in the tree corresponds to an instance of the external-regret algorithm. The instance at level $$h$$ gets updated every $$M^{d-h}$$ rounds, creating a "lazy" update schedule.

This variable structure enables TreeSwap to distribute the learning task across multiple instances of the external-regret algorithm, with each instance focusing on a different aspect of the overall problem.

##### **Slide 12: Workings of the Algorithm**

For each time step $$t$$, the distribution $$x^{(t)}$$ played by TreeSwap is the uniform average over the distributions played by the instances of $$\text{Alg}$$ at each node along the root-to-leaf path at that step.

The instances $$\text{Alg}_{\sigma_{1:h-1}}$$ are updated in a lazy fashion: every $$M^{d-h}$$ rounds when $$\sigma_{1:h-1}$$ lies on the current root-to-leaf path, the utility functions $$f^{(t)}$$ are averaged and fed to the update procedure of the corresponding instance.

As a result, each instance $$\text{Alg}_{\sigma_{1:h-1}}$$ is updated a total of $$M$$ times throughout the entire run of TreeSwap. This lazy update scheme is crucial for the algorithm's efficiency, allowing it to maintain the same per-iteration runtime complexity as the base external-regret algorithm despite providing stronger swap regret guarantees.

##### **Slide 13: Walk-through of Full TreeSwap**

The TreeSwap algorithm operates as follows:

1. Initialization: For each sequence $$\sigma$$ in the tree, initialize an instance of the external-regret algorithm $$\text{Alg}$$ with time horizon $$M$$.

2. For each round $$t$$:
   - Compute $$\sigma$$, the base-$$M$$ representation of $$t-1$$
   - Update the appropriate instances of $$\text{Alg}$$ based on their position in the tree
   - Output the uniform mixture of actions from the instances along the path from root to leaf

The lazy update mechanism is particularly important: instances closer to the root are updated less frequently (focusing on longer-term trends), while instances closer to the leaves update more often (adapting to shorter-term fluctuations).

This multi-resolution approach helps the algorithm react to changes in the adversary's strategy at various time scales, creating a robust learning process that effectively minimizes swap regret while maintaining computational efficiency.

##### **Slide 14: Main Theorem on Upper Bound**

The central result of the paper is formalized in a theorem that precisely quantifies the swap regret guarantees of TreeSwap:

**Theorem**:
Suppose that an action set $$\mathcal{X}$$ and a utility function class $$\mathcal{F}$$ are given, together with an algorithm Alg satisfying the conditions of Assumption 1. Suppose that $$T, M, d \in \mathbb{N}$$ are given for which $$M \geq 2$$ and $$M^{d-1} \leq T \leq M^d$$. Then given an adversarial sequence $$f^{(1)}, \ldots, f^{(T)} \in \mathcal{F}$$, TreeSwap$$(F, X, \text{Alg}, T)$$ produces a sequence of iterates $$x^{(1)}, \ldots, x^{(T)}$$ satisfying the following swap regret bound:

$$\text{SwapRegret}(x^{(1:T)}, f^{(1:T)}) \leq R_{\text{Alg}}(M) + \frac{3}{d}.$$

This theorem elegantly shows that the swap regret of TreeSwap can be bounded in terms of the external regret of the base algorithm ($$R_{\text{Alg}}(M)$$) plus an additional term ($$\frac{3}{d}$$) that decreases as the depth of the tree increases.

##### **Slide 15: Points to Note**

When applying the theorem in practice, several important points should be considered:

- For simplicity, the theorem assumes $$T = M^d$$, but the result generalizes to cases where $$T$$ isn't exactly $$M^d$$.

- The theorem is typically applied by choosing $$M$$ as a function of $$T$$, then selecting $$d$$ to satisfy $$M^{d-1} \leq T \leq M^d$$.

- As long as $$T$$ is sufficiently large, the additional term $$\frac{3}{d}$$ can be made arbitrarily small, ensuring that SwapRegret($$T$$) approaches the external regret bound.

- The depth parameter $$d$$ controls the trade-off between the number of rounds $$T$$ and the additional term in the swap regret bound, allowing flexibility in practical applications.

##### **Slide 16: Applications: Concrete Swap Regret Bounds**

The TreeSwap framework resolves multiple long-standing gaps in regret minimization and equilibrium computation. For constant $$\epsilon$$, the authors address all the previously discussed gaps.

In the expert setting with $$N$$ actions, applying Theorem 1.1 with action set $$X = [N]$$ and reward class defined by all [0,1]-bounded functions, i.e., $$F = [0,1]^{[N]}$$, produces powerful concrete bounds.

This application leads to significant improvements over prior work, particularly in terms of the dependence on the number of actions $$N$$. Where previous approaches required a linear dependence on $$N$$, TreeSwap achieves a remarkable logarithmic dependence.

##### **Slide 17: Corollary 1.2**

**Corollary 1.2 (Upper bound for finite action swap regret; informal version)**:

Fix $$N \in \mathbb{N}$$ and $$\epsilon \in (0,1)$$, and consider the setting of online learning with $$N$$ actions. Then for any $$T$$ satisfying $$T \geq (\log(N)/\epsilon^2)^{\Omega(1/\epsilon)}$$, there is an algorithm that, when faced with any adaptive adversary, has swap regret bounded above by $$\epsilon$$. Further, the amortized per-iteration runtime of the algorithm is $$O(N)$$, its worst-iteration runtime is $$O(N/\epsilon)$$ and its space complexity is $$O(N/\epsilon)$$.

This bound is exponentially better in $$N$$ compared to prior work, which required $$T \geq \tilde{\Omega}(N/\epsilon^2)$$ rounds. TreeSwap achieves an improved total runtime of $$\tilde{O}(N)$$, compared to the previous $$\Omega(N^3)$$ runtime of [BM07].

The term $$O(1/\epsilon)$$ in the exponent shows that as we demand lower regret ($$\epsilon$$ smaller), the number of rounds increases exponentially—implying that very small $$\epsilon$$ remains challenging to achieve efficiently.

##### **Slide 18: Corollary 1.3**

The authors further apply Theorem 1.1 to function classes with finite Littlestone dimension, yielding another important result:

**Corollary 1.3 (Swap regret for Littlestone classes; informal version)**:

If the class $$\mathcal{X}$$ has Littlestone dimension at most $$L$$, then for any $$T \geq (L/\epsilon^2)^{\Omega(1/\epsilon)}$$, there is a learner whose swap regret is at most $$\epsilon$$. In particular, games with finite Littlestone dimension admit no-swap-regret learners and thus have $$\epsilon$$-approximate CE for all $$\epsilon > 0$$.

This is a significant advancement because even the existence of approximate Correlated Equilibria in games of finite Littlestone dimension was previously unknown. The result extends swap regret minimization to continuous or infinite action spaces with structured constraints, broadening the applicability of CE to more complex decision-making domains.

##### **Slide 19: Theorem 1.4 (Bandit swap regret setting)**

The final major result addresses the challenging bandit setting, where the learner only observes the reward of the chosen action rather than the full reward vector:

**Theorem 1.4 (Bandit swap regret; informal version)**:

Let $$N \in \mathbb{N}, \epsilon \in (0,1)$$ be given, and consider any $$T \geq N \cdot (\log(N)/\epsilon)^{O(1/\epsilon)}$$. Then there is an algorithm in the adversarial bandit setting with $$N$$ actions (BanditTreeSwap; Algorithm 4) which achieves swap regret bounded above by $$\epsilon$$ after $$T$$ iterations.

This result closes a key gap in bandit learning. For $$\epsilon = O(1)$$, it guarantees that $$\tilde{O}(N)$$ rounds suffice to achieve swap regret of at most $$\epsilon$$, creating only a polylogarithmic gap between the adversarial bandit setting and the full-information non-distributional setting.

This is particularly noteworthy because for external regret, there's an exponential gap between these settings: $$O(\log N)$$ rounds suffice in the full-information case, while $$\Omega(N)$$ rounds are needed in the bandit setting.

##### **Slide 20: Applications: Equilibrium Computation**

The TreeSwap results lead to transformative improvements in equilibrium computation. The authors establish:

**Corollary 1.5 (Query and communication complexity upper bound; informal version)**: 

In normal-form games with a constant number of players and $$N$$ actions per player, the communication complexity of computing an $$\epsilon$$-approximate CE is $$\log(N)^{\tilde{O}(1/\epsilon)}$$ and the query complexity of computing an $$\epsilon$$-approximate CE is $$N \cdot \log(N)^{\tilde{O}(1/\epsilon)}$$.

These bounds represent exponential improvements over previous best-known results for correlated equilibrium computation. Prior methods required quadratically more queries and exponentially more communication than what TreeSwap achieves.

The main reduction can be used to obtain efficient algorithms for computing $$\epsilon$$-CE even when $$N$$ is exponentially large, provided there are efficient external regret algorithms available. This dramatically expands the range of games for which we can practically compute correlated equilibria.

##### **Slide 21: Extensive Form Games**

The TreeSwap approach yields particularly impactful results for extensive-form games, which model sequential decision-making:

**Corollary 1.6 (Extensive form games; informal version)**: 

For any constant $$\epsilon$$, there is an algorithm which computes an $$\epsilon$$-approximate CE of any given extensive form game, with a runtime polynomial in the representation of the game (i.e., polynomial in the number of nodes in the game tree and in the number of outgoing edges per node).

This is a remarkable achievement because extensive-form games typically have action spaces that scale exponentially with the description length of the game. Previous best-known methods for $$\epsilon$$-CE in extensive-form games required exponential time in the game tree size.

The polynomial-time algorithm for computing approximate CE in extensive-form games makes correlated equilibrium computation feasible even for large decision trees with complex information structures. This opens up new possibilities for modeling and solving realistic multi-agent interactions across domains like economics, autonomous systems, and security.

##### **Slide 22: Near-Matching Lower Bounds**

While TreeSwap provides powerful upper bounds, the authors also establish important lower bounds that show these results are nearly optimal:

Theorem 1.1 and Corollary 1.2 require the number of rounds $$T$$ to be exponential in $$1/\epsilon$$, where $$\epsilon$$ denotes the desired swap regret. The following lower bound shows this dependence is necessary, even facing an oblivious adversary that is constrained to choose reward vectors with constant $$\ell_1$$ norm:

**Theorem 1.7 (Lower bound for swap regret with oblivious adversary)**: 

Fix $$N \in \mathbb{N}, \epsilon \in (0,1)$$, and let $$T$$ be any number of rounds satisfying:

$$T \leq O(1) \cdot \min \left\{\exp(O(\epsilon^{-1/6})), \frac{N}{\log^{12}(N) \cdot \epsilon^2} \right\}.$$

Then, there exists an oblivious adversary on the function class $$\mathcal{F} = \{f \in [0,1]^N \mid \|f\|_1 \leq 1\}$$ such that any learning algorithm run over $$T$$ steps will incur swap regret at least $$\epsilon$$.

This establishes the first $$\tilde{\Omega}\left(\min(1, \sqrt{N/T})\right)$$ swap regret lower bound for distributional swap regret, achieved by an oblivious adversary. The lower bound means the exponential dependence on $$1/\epsilon$$ in the upper bounds is unavoidable, confirming the near-optimality of TreeSwap.

##### **Slide 23: Theorem 1.7**

This lower bound theorem establishes fundamental limits on how efficiently swap regret can be minimized. The theorem demonstrates that any algorithm attempting to minimize swap regret must take at least a certain number of rounds ($$T$$) to achieve a desired regret bound ($$\epsilon$$).

A key aspect of this result is that it holds even with an oblivious adversary—one that chooses the sequence of reward functions before the learning process starts and does not adapt to the learner's actions. This makes the lower bound particularly strong, as oblivious adversaries are generally easier to learn against than adaptive ones.

The constraint that the adversary is restricted to choosing reward functions with bounded $$\ell_1$$-norm (meaning the total sum of rewards across all actions is limited to some fixed value) is also significant. Even with this limitation, which prevents the adversary from assigning arbitrarily large rewards, minimizing swap regret still requires a large number of rounds.

This suggests that the hardness of swap regret minimization is not due to extreme reward values but rather from the inherent complexity of tracking all possible action swaps. Bounded rewards only affect the magnitude of regret but do not reduce the structural complexity of evaluating swap regret.

##### **Slide 24: Further Implications of Lower Bounds**

The distributional swap regret lower bound established in Theorem 1.7 holds even when the learner is allowed to randomize over actions rather than picking a single action. This demonstrates that the difficulty is inherent to swap regret minimization itself, not just a restriction of the learning model.

Remarkably, this lower bound holds even for a simple adversary and for function classes with constant Littlestone dimension. This indicates that swap regret minimization is fundamentally hard, even in the simplest possible settings.

For external regret, upper and lower bounds are usually much closer, leading to tight characterizations of optimal algorithms. In contrast, for swap regret, the gap between the upper bound ($$\exp(\epsilon^{-1})$$ from Corollary 1.2) and the lower bound ($$\exp(\epsilon^{-1/6})$$ from Theorem 1.7) suggests we might be overestimating the difficulty of swap regret minimization, leaving open the possibility of improved algorithms.

The authors also establish a stronger lower bound by allowing for an adaptive adversary—one that adjusts its rewards based on the learner's actions. This construction shows that $$T \geq \exp(\Omega(\epsilon^{-3}))$$ rounds are necessary, bringing the lower bound closer to the upper bound of $$\exp(\Omega(\epsilon^{-1}))$$.

##### **Slide 25: Conclusion**

The TreeSwap algorithm represents a significant breakthrough in overcoming the computational challenges of swap regret minimization. Its success lies in several key innovations:

1. **Hierarchical Structure**: Instead of directly minimizing swap regret (which is computationally expensive), TreeSwap breaks down the problem by structuring it hierarchically using a tree.

2. **Learning at Different Time Scales**: The lazy update scheme ensures that learners at different levels of the tree operate at different time scales. Learners closer to the root are updated less frequently, focusing on longer-term trends, while learners closer to the leaves are updated more often, adapting to shorter-term fluctuations.

3. **Leveraging External Regret Guarantees**: Each level of the tree runs an external regret minimization algorithm, which is already well understood and efficient.

4. **Combining Expert Advice**: The algorithm takes a uniform mixture of the distributions suggested by the learners along the path from root to leaf, effectively combining advice from multiple "experts" operating at different time scales.

By combining these elements, TreeSwap achieves swap regret minimization as efficiently as external regret minimization in many cases. The "2.0" in the paper's title signifies the substantial improvement over prior approaches, bringing swap regret minimization into the realm of practical applicability for large-scale problems.

##### **Slide 26: Discussion & Questions**

The near-optimality of the bounds established in this paper represents a major advancement in our understanding of swap regret. While there remains a gap between the upper and lower bounds (roughly $$\exp(\epsilon^{-1})$$ versus $$\exp(\epsilon^{-1/6})$$ for oblivious adversaries), these results are already close enough to be called near-optimal.

This suggests that TreeSwap is among the most efficient known algorithms for swap regret minimization, though there may still be room for theoretical improvements. The gap indicates that we might not have fully characterized the true complexity of swap regret minimization yet.

The paper leaves several interesting questions for future work:

1. Can the gap between upper and lower bounds be further reduced?
2. Are there specific classes of games or learning problems where the bounds can be tightened?
3. How do these theoretical guarantees translate to practical performance in real-world applications like multi-agent reinforcement learning?
4. Can the TreeSwap approach be extended to other regret notions beyond swap regret?

These questions represent promising directions for continued research in this area, building on the foundation established by the TreeSwap algorithm.

##### **Conclusion**

This presentation has walked through the groundbreaking "Swap Regret 2.0" paper, which introduces TreeSwap—an algorithm that fundamentally changes our ability to minimize swap regret in large and infinite action spaces. By leveraging a hierarchical tree structure of external regret minimizers and a clever lazy update scheme, TreeSwap achieves what was previously thought impractical: efficient swap regret minimization with only logarithmic dependence on the number of actions.

The implications of this work extend far beyond theoretical interest. By making correlated equilibrium computation feasible in extensive-form games and spaces with large action sets, this research opens new possibilities for applications in multi-agent reinforcement learning, economic modeling, and game-theoretic analysis of complex strategic interactions.

While some gaps remain between the upper and lower bounds, particularly in the dependence on the approximation parameter ε, the near-optimality of these results represents a significant milestone in online learning theory. Future work might further tighten these bounds or explore specialized versions of TreeSwap for specific application domains.

As machine learning continues to tackle increasingly complex multi-agent problems, algorithms like TreeSwap that efficiently bridge the gap between external and swap regret will become essential tools in our theoretical and practical toolkit.

##### **Acknowledgments**

Special thanks to the Advanced Machine Learning class at NYU's Courant Institute for the opportunity to present this work, and to the authors of the original paper for their significant contribution to the field of online learning and game theory.

