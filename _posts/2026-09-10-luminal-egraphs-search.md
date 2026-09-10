---
layout: post
title: How Luminal Works - E-Graphs, Equality Saturation, and Search
date: 2026-09-10 09:00:00-0400
featured: false
description: A first-principles walk through Luminal, the Rust inference compiler that replaces destructive rewrite passes with equality saturation, and picks its final program by measuring candidates on real hardware
tags: GPU ML
categories:
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Hello all. This post is about [Luminal](https://github.com/luminal-ai/luminal), an inference compiler written in Rust, and about the single idea the whole thing is built on.

Nearly every compiler you have used works by rewriting. It finds a pattern, replaces it with something better, and moves on. Luminal refuses to do that. When it discovers that two programs are equivalent, it does not pick one; it records the equality and keeps both. Do that for every rule it knows, over and over until nothing new can be learned, and you end up with a data structure that holds an enormous set of equivalent programs at once. Only then does the compiler choose, and it chooses by compiling candidates and timing them on the actual device.

That is the thesis. The rest of the design follows from it, and a lot of things that look strange at first, the tiny operator set, the symbolic shapes, the fact that a matmul is not an operation, all stop looking strange once you see what the search needs from them.

The plan:

- The one idea: why rewriting destroys information, and what equality saturation does instead
- The map: the pipeline end to end, and what you actually write
- E-graphs from first principles: e-nodes, e-classes, congruence, and saturation
- egglog: rules instead of passes, read on a real Luminal rule
- Extraction and search: why choosing is the hard part, and why Luminal measures
- When things compile: ahead-of-time, just-in-time, and what "shape-specific kernels compiled at runtime" means
- The core and runtime boundary, and where a backend plugs in
- Where this sits next to torch.compile, XLA, TVM, and tinygrad
- Running it on a Mac
- Takeaways, and a question bank over everything above

I'm assuming you know roughly what a computation graph is and have seen a GPU kernel before, at the level of my [GPU notes](/blog/2025/gpu-notes/). No Rust knowledge is needed; the few snippets are short and I'll explain them. Everything quoted from the codebase was checked against commit `d18376d1` as of 2026-09-10, and Luminal moves fast, so treat file paths as pointers rather than promises.

Let's get started.

---

#### **The One Idea**

##### **Rewriting throws information away**

Take the expression $$(a \times 2) / 2$$ and give a compiler two perfectly good rules:

1. Multiplying by a power of two is a left shift: $$x \times 2 \rightarrow x \ll 1$$.
2. Multiplying and then dividing by the same value cancels: $$(x \times y) / y \rightarrow x$$.

Rule 1 is a real optimization; shifts are cheaper than multiplies on most hardware. Apply it and you get $$(a \ll 1) / 2$$. You have made the program faster and you have also made it impossible to reach the correct answer, which is just $$a$$. Rule 2 can no longer fire, because the multiply it was looking for is gone.

Apply rule 2 first and you get $$a$$, which is better than anything rule 1 could produce. So the rules are order-dependent, and the right order depends on the expression. This is the **phase ordering problem**, and it is not a corner case. It is the central difficulty in optimizing compilers, and it gets worse as you add rules, because every new rule interacts with all the existing ones.

The usual answer is to make rules conservative. Only fire when you are sure it helps. But "sure it helps" is a judgment that depends on everything downstream, so the guards get complicated, and special-cased, and numerous. Luminal's README makes exactly this complaint about the traditional stacks: destructive rules must only fire when a benefit is certain, and that requirement is what makes them balloon.

If you want to see what that balloon looks like in production, I wrote up one instance of it in detail: [vLLM's pattern matching pipeline](/blog/2026/silu-mul-fp8-block-quant-compile-vLLM/) registers sixteen structurally distinct FX graph patterns to cover a single fusion, because the pattern matcher matches exactly and each combination of flags produces a different graph. Every one of those sixteen is correct. The count is not a bug; it is the cost of exact, destructive, one-directional matching.

##### **Equality saturation does not choose**

[Tate, Stepp, Tatlock, and Lerner (POPL 2009)](https://cseweb.ucsd.edu/~lerner/papers/popl09.html) proposed the alternative. Instead of an optimization being "replace this with that", an optimization becomes "this equals that". The compiler applies its rules repeatedly, and each application adds equality information to the intermediate representation rather than mutating it. Run until no rule can add anything new, and the IR is **saturated**: it now encodes many optimized versions of the input program simultaneously. Only after that does a separate step pick one.

On our example, saturation records that $$(a \times 2)/2$$, $$(a \ll 1)/2$$, and $$a$$ are all the same thing. Nothing was lost, no order was committed to, and the choice of which one to emit is made at the end with full knowledge of all three.

The data structure that makes this affordable is the **e-graph**, and we'll build one by hand shortly. The short version is that it compresses an exponentially large set of equivalent programs into something roughly the size of the original, by sharing.

##### **Why an ML compiler wants this**

Two reasons, and they compound.

The first is that ML graph optimization is where phase ordering bites hardest. [Yang et al.](https://arxiv.org/abs/2101.01332) made the case directly: production frameworks apply graph substitutions sequentially and are therefore sensitive to order, and they explore only a small fragment of the exponential space of equivalent graphs. Applying all substitutions at once via equality saturation, they found graphs up to 16% faster while spending on average 48 times less time optimizing.

The second is that on modern accelerators nobody actually knows which version is faster. Whether to fuse two kernels, whether to call a vendor library or emit your own, which tiling to use, all of it depends on cache behavior, occupancy, and launch overhead in ways that defeat static cost models. If your compiler has already committed to a rewrite, that question never gets asked. If it has kept every alternative alive, you can just run them and find out.

Luminal takes the second option to its logical end. Its README puts the position bluntly: "The best heuristic is no heuristic." It searches instead.

**References**
- [Equality Saturation: a New Approach to Optimization, Tate et al., POPL 2009](https://cseweb.ucsd.edu/~lerner/papers/popl09.html) ([PDF](https://homes.cs.washington.edu/~ztatlock/pubs/eqsat-tate-popl09.pdf))
- [Equality Saturation for Tensor Graph Superoptimization, Yang et al.](https://arxiv.org/abs/2101.01332)
- [Luminal README](https://github.com/luminal-ai/luminal)
- [SiLU+Mul+FP8 Block Quant Pattern Matching Pipeline](/blog/2026/silu-mul-fp8-block-quant-compile-vLLM/) (the destructive-matching cost, worked out)

---

#### **The Map**

Before any details, the shape of the whole thing.

<div class="row justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/luminal-egraphs-search/luminal-pipeline.svg" title="The Luminal compilation pipeline" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The path from a line of Rust to a running kernel. The core crate stops at the saturated e-graph; everything right of the dashed line belongs to the runtime. The red loop is the part that makes Luminal unusual: candidates are compiled and timed rather than scored by a model. The purple box is what keeps that cost one-time. Editable source: <a href="/assets/img/luminal-egraphs-search/luminal-pipeline.excalidraw">luminal-pipeline.excalidraw</a>.
</div>

Five stages, and it is worth naming them now because the vocabulary recurs:

| Stage | What it produces | Where it lives |
|---|---|---|
| Front end | an HLIR graph of primitive ops | `src/frontend/` |
| Lowering to egglog | an egglog program | `src/egglog_utils/` |
| Saturation | one e-graph per dynamic-dimension bucket | `Graph::build_search_space` in `src/graph.rs` |
| Extraction and search | one selected LLIR program per bucket | `src/search/`, driven by the runtime |
| Load and execute | device kernels, compiled and launched | `crates/luminal_metal/`, `crates/luminal_cuda_lite/` |

HLIR is the high-level IR, the graph of primitives you built. LLIR is the low-level IR, a graph of backend ops that map onto real kernels. Saturation is the step that turns one into a space containing many of the other.

##### **What you actually write**

Here is a complete Luminal program, from the repository's `examples/simple`:

```rust
use luminal::prelude::*;

let mut cx = Graph::new();
let a = cx.tensor((3, 1));
let b = cx.tensor((1, 4));

let c = a.matmul(b).output();

let mut rt = cx.compile(ReferenceRuntime::default(), CompileOptions::default());

rt.set_data(a, vec![1.0, 2.0, 3.0]);
rt.set_data(b, vec![1.0, 2.0, 3.0, 3.0]);
rt.execute(&cx.dyn_map);

println!("Result: {:?}", rt.get_f32(c));
```

Three things about this are worth slowing down on.

**Nothing computes until `execute`.** `cx.tensor((3, 1))` does not allocate a 3-by-1 array. It adds a node to a graph and hands you back a `GraphTensor`, which as of this commit is a small `Copy` struct holding a node index, a pointer to the graph, a `ShapeTracker`, and a dtype ([`src/frontend/tensor.rs`](https://github.com/luminal-ai/luminal/blob/main/src/frontend/tensor.rs)). It holds no data. `a.matmul(b)` does no arithmetic; it appends nodes. This is lazy execution, but unlike libraries that bolt laziness onto an eager core, here *everything* is built this way, so the compiler always sees the entire network.

**Data comes after the computation.** You define the graph, compile it, and only then say what the inputs are. That inversion is deliberate: the shape of the computation is a compile-time fact, the contents of the tensors are a runtime one, and separating them lets you re-run with new data without rebuilding anything.

**`compile` returns the runtime.** It is not a mutation of the graph. `Graph::compile` takes a runtime and options and gives you back a prepared runtime holding a selected program. The graph itself stays as the specification.

##### **Fifteen operations**

Everything reduces to a small primitive set. The math primitives, as the README groups them:

| Group | Ops |
|---|---|
| Unary | `Log2`, `Exp2`, `Sin`, `Sqrt`, `Recip` |
| Binary | `Add`, `Mul`, `Mod`, `LessThan` |
| Other | `SumReduce`, `MaxReduce`, `Iota`, `Gather`, `Scatter`, `Cast` |

Fifteen. You will also find a handful of structural entries alongside them in the `HLIROps` tuple in [`src/hlir.rs`](https://github.com/luminal-ai/luminal/blob/main/src/hlir.rs): `Input`, `Output`, `Constant`, `CustomOpKind`, and the loop markers `LoopStart`, `LoopEnd`, `LoopInput`, and friends. Those are not arithmetic; they are how the graph marks its boundaries and its repeated structure. The arithmetic really is fifteen ops.

Everything else is derived in the front end before the compiler ever sees it, and the derivations are exactly what you would write on paper:

| You write | The graph gets |
|---|---|
| `a - b` | `a + (-b)` |
| `a / b` | `a * recip(b)` |
| `x.exp()` | `exp2(x * (1 / ln 2))` |
| `x.log()` | `log2(x) * ln 2` |
| `x.softmax(d)` | subtract the max, `exp`, sum, reciprocal, multiply |

That last row is worth noticing. Softmax is not an operation in Luminal. It is a shape of subgraph. So is layer norm, so is attention, so is a matmul.

##### **A matmul is not an operation**

This is the part that surprises people, so let's do it concretely. `matmul` for the 2D case, from [`src/frontend/matmul.rs`](https://github.com/luminal-ai/luminal/blob/main/src/frontend/matmul.rs), is essentially three lines:

```rust
let mul = self.expand_dim(1, n) * rhs.permute((1, 0)).expand_dim(0, m);
let ret = mul.sum(2);
```

Broadcast both operands to a common $$(m, n, k)$$ shape, multiply elementwise, sum over the last axis. That is the textbook definition of matrix multiplication, written out, and it is exactly what lands in the graph.

For `(3, 1)` times `(1, 4)`, the resulting HLIR graph is **five nodes**: two `Input`s, one `Mul`, one `SumReduce`, one `Output`. The `expand_dim` and the `permute` contribute no nodes at all.

That last fact deserves its own beat. **Movement operations are free.** A permute, a broadcast, a reshape, a slice: none of them add work, because none of them move data. They change the `ShapeTracker` attached to the `GraphTensor`, which records dimensions and strides. Reading a transposed tensor is reading the same buffer with the indices swapped. The compiler sees strides, not copies, and a later kernel simply indexes accordingly.

So the graph the optimizer receives is maximally decomposed. There is no `MatMul` node to protect, no fused softmax to preserve. Every structure that a normal framework would hand-write as a kernel is, here, just a subgraph that the rewrite rules are free to recognize, or not, in whatever grouping turns out to be fastest. Flash Attention is not a special case in Luminal; it is a fusion the search is allowed to find.

**References**
- [Luminal source](https://github.com/luminal-ai/luminal), files as cited, as of commit `d18376d1`, 2026-09-10
- [Luminal documentation](https://docs.luminalai.com/docs/introduction)

---

#### **E-Graphs From First Principles**

This is the section that makes the rest of the post legible, so we'll build the structure from nothing.

##### **Terms, and the problem with sets of terms**

Start with an expression as a tree. $$(a \times 2) / 2$$ is a `Div` node whose children are a `Mul` node and the constant 2; the `Mul` node's children are $$a$$ and 2.

Now suppose we want to keep every equivalent form of that expression. We know three: the original, the shifted version $$(a \ll 1)/2$$, and $$a$$. Storing them as three separate trees works, but it does not scale. If an expression has ten independent subexpressions and each has three equivalent forms, there are $$3^{10}$$, about 59,000, whole trees to store, and almost all of them share almost all of their structure. Store them separately and you drown.

The e-graph is the fix, and it is fundamentally a compression scheme.

##### **E-nodes and e-classes**

Two definitions, and everything follows.

An **e-class** is a set of things that are known to be equal to each other. It is an equivalence class.

An **e-node** is an operator together with a list of children, where each child is an **e-class**, not another e-node. This is the whole trick. In an ordinary tree, `Mul`'s children are specific expressions. In an e-graph, `Mul`'s children are *sets* of expressions, and the e-node stands for the multiply of any member of the first set by any member of the second.

An **e-graph** is a collection of e-classes, each holding one or more e-nodes, plus a designated root class.

To read a program out of an e-graph, start at the root class, pick any one e-node from it, then recursively pick one e-node from each of that e-node's child classes. Every set of choices yields a valid term, and every term the e-graph represents arises from some set of choices. That is what "it holds many programs at once" means, concretely: it is a choice per class, and the terms are the combinations.

The counting follows immediately. If the root class has two e-nodes, and one of its children's classes has three, and those choices are independent, you have six terms stored as five e-nodes. Add one e-node to one class and you can double the number of programs represented while growing the structure by one. That multiplicative behavior is why the compression works, and it is also, as we'll see, why choosing well is hard.

##### **Congruence, the property that makes it an e-graph**

If an e-graph were only "sets of equal things", a union-find would do. The extra property, and the reason the [egg paper](https://arxiv.org/abs/2004.03082) describes an e-graph as efficiently representing a **congruence relation**, is this:

$$
a \equiv b \quad \Longrightarrow \quad f(a) \equiv f(b)
$$

Equality has to propagate upward through operators. If you learn that $$x \times 2$$ equals $$x \ll 1$$, then anything built on top of $$x \times 2$$ is equal to the same thing built on top of $$x \ll 1$$, automatically, without any rule firing. Maintaining that closure as merges happen is the hard engineering, and it is exactly what the egg paper's **rebuilding** technique makes fast: rather than restoring the invariant after every single merge, defer and restore it in batches, which the paper reports gives asymptotic speedups in practice.

##### **Doing it by hand**

Let's saturate our example. Rules: $$x \times 2 \rightarrow x \ll 1$$ and $$(x \times y)/y \rightarrow x$$.

<div class="row justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/luminal-egraphs-search/egraph-rewrite.svg" title="An e-graph before and after two rewrites" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Dashed boxes are e-classes, solid boxes inside them are e-nodes, and arrows point from an e-node to the e-class of each child. Panel 1: the initial e-graph, one e-node per class, representing exactly one program. Panel 2: the shift rule fires and its result joins the class it was proved equal to, so that class now holds two e-nodes, and the multiply is still there. Panel 3: the cancellation rule fires on the root, merging the root class with the class holding a. Three programs, none of them lost. Note the red edges in panel 3: merging two classes can create a cycle, which is why extraction has to check for one. Editable source: <a href="/assets/img/luminal-egraphs-search/egraph-rewrite.excalidraw">egraph-rewrite.excalidraw</a>.
</div>

Walking it in words:

**Initial.** Build the tree, one e-node per e-class. Class $$c_a$$ holds the leaf $$a$$. Class $$c_2$$ holds the constant 2. Class $$c_m$$ holds `Mul(`$$c_a$$`, `$$c_2$$`)`. The root class $$c_r$$ holds `Div(`$$c_m$$`, `$$c_2$$`)`. Four classes, four e-nodes, one program represented.

**Rule 1 fires.** The left side $$x \times 2$$ matches the e-node in $$c_m$$, binding $$x$$ to $$c_a$$. The right side is $$x \ll 1$$, so we add an e-node `Shl(`$$c_a$$`, `$$c_1$$`)` and, crucially, we **union** it with the class we matched, $$c_m$$, rather than replacing anything. Now $$c_m$$ holds two e-nodes. The multiply is still there. Two programs represented.

**Rule 2 fires.** The left side $$(x \times y)/y$$ matches the root: the `Div`'s first child class $$c_m$$ contains a `Mul` whose second child class is $$c_2$$, and the `Div`'s second child class is also $$c_2$$. Binding $$x$$ to $$c_a$$, the right side is just $$x$$, so we union the root class $$c_r$$ with $$c_a$$. Those two classes become one.

**Saturation.** Try every rule again. Nothing new is learned, so we are at a fixed point. Done.

The final e-graph represents three programs: $$(a \times 2)/2$$, $$(a \ll 1)/2$$, and $$a$$. No order was chosen. No information was destroyed. Both rules fired and neither blocked the other.

That last sentence is the whole reason this machinery exists. Compare it to the destructive version, where firing rule 1 first permanently prevented rule 2.

##### **Saturation is a fixed point, sometimes with a budget**

"Saturate" means run rules until no rule can add anything new. Sometimes that terminates quickly. Sometimes the rule set is such that it does not terminate at all; commutativity plus associativity can generate structure forever. Real systems therefore cap it, by iteration count, node count, or time, and accept a partially saturated e-graph. That is fine: a partial saturation still represents more programs than a destructive pipeline ever could, and the search downstream simply has fewer candidates.

Luminal expresses its budget as an explicit **schedule**, which we'll look at next.

**References**
- [egg: Fast and Extensible Equality Saturation, Willsey et al., POPL 2021](https://arxiv.org/abs/2004.03082)
- [Equality Saturation: a New Approach to Optimization, Tate et al., POPL 2009](https://cseweb.ucsd.edu/~lerner/papers/popl09.html)

---

#### **egglog: Rules Instead of Passes**

Luminal does not implement e-graphs itself. It depends on [egglog](https://github.com/egraphs-good/egglog), pinned to a specific git revision in its `Cargo.toml`.

##### **Why a language and not a library**

The egg library lets you write rewrite rules in Rust. egglog goes further: it is a language, and its contribution, per [Zhang et al. (PLDI 2023)](https://arxiv.org/abs/2304.04332), is unifying Datalog with equality saturation. From Datalog it takes efficient incremental execution, cooperating analyses, and lattice-based reasoning; from equality saturation it takes term rewriting, congruence closure, and extraction.

The Datalog half matters more than it sounds. A pure rewrite system can only say "this term equals that term". A compiler needs to say things like "the dtype of this node is f32", "these list elements are all identical", "this subgraph was already recognized as a rotary embedding". Those are **relations**, and being able to derive them incrementally, have rules depend on them, and have them cooperate with the equality reasoning is what lets a real backend be expressed as rules at all.

##### **A real rule, read closely**

Here is an actual Luminal rule, from `src/egglog_utils/matmul_flattening/squeeze.egg`, trimmed of comments. Its job is to strip a leading dimension of size 1 off a multiply-then-sum pair, so that downstream two-dimensional matmul rules, the ones that can hand the work to cuBLAS, become applicable.

```lisp
(rule
    (
        (= ?mul (Op (Mul ?mul_shape ?a_stride ?b_stride ?mul_out_stride)
                    (ICons ?a (ICons ?b (INil)))))
        (= ?sum (Op (Sum ?sum_shape ?k ?sum_in_stride ?k_stride ?sum_out_stride)
                    (ICons ?mul (INil))))

        (= ?sum_shape (ECons (MNum 1) ?rest_sum_shape))
        (!= ?rest_sum_shape (ENil))
        (= ?rest_sum_len (len ?rest_sum_shape))
        (>= ?rest_sum_len 2)

        (= ?mul_shape (ECons (MNum 1) ?rest_mul_shape))
        (= ?a_stride (ECons ?as0 ?rest_as))
        ...
        (= ?dt (dtype ?a))
    )
    (
        (let ?new_mul (Op (Mul ?rest_mul_shape ?rest_as ?rest_bs ?rest_mos)
                          (ICons ?a (ICons ?b (INil)))))
        (let ?new_sum (Op (Sum ?rest_sum_shape ?k ?rest_sis ?k_stride ?rest_sos)
                          (ICons ?new_mul (INil))))
        (union ?sum ?new_sum)
        (set (dtype ?new_mul) ?dt)
        (set (dtype ?new_sum) ?dt)
    )
    :ruleset matmul_flatten
    :name "batch-collapse squeeze dim=1"
)
```

A rule has two halves. The first parenthesized block is the **query**: facts that must hold for the rule to fire. The second is the **action**: what to assert when it does.

Reading the query top to bottom. Find a `Mul` and a `Sum` where the `Sum` consumes the `Mul`; `ICons` and `INil` are cons-list constructors for the input list, so `(ICons ?mul (INil))` means "a one-element input list containing the multiply". Require that the `Sum`'s output shape begins with a literal 1, and that what remains is at least two dimensions, so we do not collapse below 2D. Require the same leading 1 on the `Mul`. Destructure each stride list to peel off its first element. Look up the dtype of `?a` from the `dtype` relation, which other rules populate.

Then the action. Build a new `Mul` and a new `Sum` over the shortened shapes and strides. And then the line that is the entire point of this post:

```lisp
(union ?sum ?new_sum)
```

Not "replace `?sum`". Not "delete the old one". **Union.** It asserts that the old summation and the new one are the same value, and drops both into one e-class. The four-dimensional spelling and the three-dimensional spelling now coexist, and downstream rules, including the ones that only match 2D shapes, can work from whichever they need. The comment in the file notes the rule fires recursively, 4D to 3D to 2D, and every intermediate spelling stays available.

The last two lines propagate the dtype relation onto the new terms, so the analysis stays consistent as the e-graph grows. That is the Datalog half doing its job.

##### **The schedule**

Rules are grouped into named **rulesets**, and Luminal specifies exactly how they run. The main schedule, from `src/egglog_utils/mod.rs`, is roughly:

```lisp
(saturate (seq
    (saturate (seq
        (saturate expr)
        (saturate dtype_prop)
        (run matmul_flatten)
        (run kernel_lower)
        (run direct_kernel)
        (run kernel_specialize)
        (run buffer_reuse)
        (run matmul_backend)
        (run glumoe)
        (run fusion_pair)
    ))
    (saturate (seq
        (saturate expr)
        (saturate dtype_prop)
        (run fusion_grow)
        (run fusion_merge)
    ))
))
```

`(run r)` applies ruleset `r` once. `(saturate r)` applies it until it stops learning. `(seq ...)` runs its arguments in order. So the reading is: bring the algebraic rules and the dtype analysis to a fixed point, then apply the lowering and kernel-discovery rules once each, and repeat that whole block until it stabilizes; then do the same for the fusion growth rules; then repeat the pair.

The ordering here is a performance decision, not a correctness one. The comment in the source explains it: producer rules create the raw alternatives that fusion later consumes, so saturating discovery before growing fusions avoids re-running expensive pair-discovery scans on every iteration. Because the rules are non-destructive, a different schedule would find a different amount, but it could not find something *wrong*.

After the main cycles come cleanup phases. This is where the `Runtime::CLEANUP_HLIR` flag from `src/op.rs` comes in: for a real backend, the raw HLIR spellings are deleted from the e-graph after saturation, so only lowered, executable LLIR remains extractable. The reference runtime, whose LLIR *is* HLIR, sets the flag to false and keeps them. Backend lowering rules can also mark an HLIR term as **subsumed**, which tells the extractor to stop offering it as a choice. That is how a backend says "there is no such thing as an unlowered multiply in my output".

##### **The contributor rule that falls out of all this**

The repository's `AGENTS.md` states a boundary that only makes sense once you have read the above: all graph pattern matching and op selection must be expressed in egglog rewrites, and no Rust-side passes may search for op patterns, fuse kernels, or rewrite extracted graphs after egglog has run. If a backend wants a fused op, it adds the match as a rule and lets extraction produce it.

That is not stylistic. A Rust post-pass would be destructive by construction, and it would sit outside the search, so its decisions could never be measured against the alternatives. One such pass and the whole property is gone.

**References**
- [Better Together: Unifying Datalog and Equality Saturation, Zhang et al., PLDI 2023](https://arxiv.org/abs/2304.04332)
- [egglog on GitHub](https://github.com/egraphs-good/egglog)
- Luminal `src/egglog_utils/`, `src/op.rs`, `AGENTS.md`, as of commit `d18376d1`, 2026-09-10

---

#### **Extraction and Search**

Saturation gives you a space. It does not give you a program. Getting one out is called **extraction**, and it is where the difficulty moved to.

##### **A genome is one choice per e-class**

Recall how you read a term out of an e-graph: pick one e-node from the root class, then one from each child class, recursively. So a complete selection is a function from e-classes to e-nodes, and Luminal represents exactly that. The type is `IndexedChoiceSet` in `src/egglog_utils/mod.rs`, and its documentation calls it a **genome**, which tells you where this is going.

The size of the space follows directly. If $$C$$ is the set of e-classes the extractor is allowed to choose in, the number of distinct selections is

$$
\prod_{c \in C} |c|
$$

where $$|c|$$ is the number of e-nodes in class $$c$$. Luminal computes precisely this, capped so it cannot overflow, in `count_choice_sets_up_to`. A product over classes is a multiplicative space: it does not grow with the size of your model, it grows with the number of places where the compiler found more than one way to do something. Add one rule that offers an alternative in fifty places and you have multiplied the space by $$2^{50}$$.

Not every combination is legal, though, and the reason is visible in panel 3 of the figure above. Merging two e-classes can produce a cycle, where a class contains an e-node that eventually depends on that same class. A term is finite, so a selection that follows a cycle does not name a program at all. Luminal's approach, per the comment in `src/egglog_utils/mod.rs`, is to draw a complete random genome and then repair only the cycles in the reachable part of it, leaving every legal choice outside those cycles untouched so that specialization stays a measured decision rather than a repaired one.

##### **Choosing optimally is NP-hard**

You might hope to just assign a cost to each e-node and pick the cheapest tree bottom-up. That works if the cost function is additive and you ignore sharing. Real cost functions are not, because two branches of the program may share a subexpression and you should pay for it once, and that turns extraction into a global optimization problem.

The complexity is settled. [Fast and Optimal Extraction for Sparse Equality Graphs (OOPSLA 2024)](https://dl.acm.org/doi/10.1145/3689801) states that e-graph extraction is NP-hard in general and, further, hard to approximate within any constant ratio; existing tools therefore rely either on integer linear programming, which is optimal but slow, or on heuristics, which are fast but not optimal. The paper's own contribution is a parameterized algorithm exploiting low treewidth. Tate et al. had already formulated extraction as an ILP back in 2009.

So there is no clean way out. And Luminal's situation is worse than the general one, because it does not even have a trustworthy cost function.

##### **Which is why Luminal measures**

On a GPU, the cost of a candidate depends on occupancy, register pressure, cache behavior, memory coalescing, launch overhead, and how the vendor's library happens to be tuned for that exact shape on that exact architecture. Any static number you assign is a guess. Luminal's answer is to stop guessing: **compile the candidate and time it.**

The mechanism is a genetic search, and `src/search/genetic.rs` implements it as a pull-style state machine. The runtime asks for a candidate, evaluates it however it likes, and reports back one of three outcomes: `Measured` with a metric, `Rejected` if it could not even be compiled or exceeded a resource cap, or `Invalid` if evaluation started but produced nothing usable. The state machine owns everything else: generations, mutation, deduplication of genomes and of programs, budgets, timeouts, and progress reporting.

The loop, in outline. Draw random genomes to seed a population. Have the runtime measure each. Keep the best. Produce the next generation by mutating the survivors, where a mutation is simply changing which e-node is selected in some class. Measure, keep, repeat, until the budget runs out. The defaults live in `CompileOptions` in `src/graph.rs`:

| Option | Default | What it controls |
|---|---|---|
| `limit` | 100 | total candidate programs evaluated |
| `generation_size` | 10 | offspring per generation |
| `mutations` | 10 | e-class choices flipped per offspring |
| `trials` | 3 | timing runs per candidate |
| `keep_best` | 1 | survivors kept as parents |
| `candidate_timeout` | 60 s | budget covering compile plus run, per candidate |
| `execution_timeout` | 1 s | cap on a single timing trial |

A hundred candidates is not a large number against a space that can be astronomically large, and that is the honest tradeoff. Genetic search does not find the optimum; it finds something good, quickly, using real measurements rather than a model that might be wrong in an unknown direction. If you want more, `limit` is a knob, and `restart_stagnation` escalates mutation counts and resamples fresh genomes when a run stops improving.

Two details worth knowing. Candidates are deduplicated both by genome and by the resulting program, because different choices often extract to the same graph. And `early_stop_factor`, off by default, lets a runtime abandon timing a candidate once its running mean has already lost to the best by a given margin, which stops the search from spending its trial budget confirming that something is slow.

##### **Dimension buckets: searching more than once**

Real inference has dynamic shapes. Sequence length changes every request; batch size changes with load. Luminal models these as **symbolic dimensions**, so a shape can be `(s, 4096)` or `(b, h, w + 3)`, with the symbols carried through the graph rather than erased.

Symbolic is good for correctness and bad for specialization: a kernel that must work for any $$s$$ cannot bake in loop bounds. So Luminal lets you declare **buckets** over a dynamic dimension, and then does the whole pipeline once per bucket combination. `DimBucket` in `src/graph.rs` is an inclusive range with an optional representative value used during profiling, and `build_search_space` produces one saturated e-graph per combination of buckets. The search then selects one program per bucket, and at execution time the runtime dispatches on the actual value.

Concretely, if you bucket sequence length into short, medium, and long, you get three e-graphs, three searches, and three specialized programs. Each was measured at a representative length inside its own range, so each is specialized to shapes it will actually see. Buckets must not overlap, and the code asserts it.

**References**
- [Fast and Optimal Extraction for Sparse Equality Graphs, OOPSLA 2024](https://dl.acm.org/doi/10.1145/3689801) ([open version](https://cse.hkust.edu.hk/~parreaux/publication/oopsla24a/))
- Luminal `src/search/`, `src/graph.rs`, `src/egglog_utils/mod.rs`, as of commit `d18376d1`, 2026-09-10

---

#### **When Things Compile**

The word "compile" does four different jobs in a Luminal discussion, and untangling them answers both the ahead-of-time versus just-in-time question and the README line about shape-specific kernels.

##### **Four things called compilation**

| Sense | What happens | When |
|---|---|---|
| Building the crate | `cargo` turns Rust into a binary | before you ship |
| Building the graph | your `matmul` calls append HLIR nodes | when your program runs, before `compile` |
| `Graph::compile` | saturate, search, select, load | once, at process startup |
| Device compilation | generated kernel source becomes machine code | inside `Graph::compile` |

The last row is the one people mean by JIT, and it is real. The Metal backend hands shader source to `new_library_with_source` at runtime (`crates/luminal_metal/src/kernel/ops.rs`). The CUDA backend hands C source to NVRTC, with architecture flags chosen from the detected compute capability (`crates/luminal_cuda_lite/src/lib.rs`). Neither kernel exists as text until the compiler decides what it wants.

##### **So which is it?**

Luminal is **ahead-of-time in philosophy and just-in-time in mechanism**, and the two are not in tension because they refer to different clocks.

Ahead-of-time here means: relative to your model's execution, everything is decided in advance. The full network is a static graph, so devices, dtypes, memory allocation, fusion, and kernel selection are all compile-time facts. There is no interpreter in the loop, no per-op dispatch, no Python at inference. The README states the policy directly: "push everything to compile time and leave nothing to run time."

Just-in-time here means: relative to the operating system process, the machine code is produced while the process is running. It has to be, because the compiler cannot know until search time which kernels it wants, and it cannot know until it sees your dimension buckets what shapes to specialize them for.

Contrast that with `torch.compile`, which is just-in-time in a stronger sense: [TorchDynamo](https://dl.acm.org/doi/10.1145/3620665.3640366) hooks Python bytecode as it executes, captures what it can into an FX graph, falls back to the interpreter when it cannot, and recompiles when guards fail. Luminal has no interpreter to fall back to and no guards to fail. The graph is complete or it does not exist.

##### **"Shape-specific kernels compiled at runtime", decoded**

Now that line reads cleanly. A dynamic dimension is symbolic in the graph. You declare buckets over it. Within a bucket the dimension is pinned to a representative, so the emitted kernel can hardcode loop trip counts, tile sizes, and unroll factors for that shape rather than reading them from a parameter and branching. That specialization is only possible once the bucket values are known, which is at process runtime, and the resulting source is then compiled by NVRTC or the Metal compiler on the spot.

The payoff is that a generically written network gets hyper-specific machine code, and you did not write the specialization.

##### **Artifacts: paying for the search once**

Searching a hundred candidates, each of which must be compiled and timed, is not free. If you had to do it on every process start, the AOT story would be a lie.

You do not. `src/graph/artifact.rs` serializes the selected schedule: which program was chosen for which bucket, fingerprinted so it can be validated against the e-graph it came from. The CUDA backend goes further in `crates/luminal_cuda_lite/src/artifact.rs`, capturing the compiled module images themselves, keyed by a signature covering the target architecture, the NVRTC options, and the NVRTC version, so a mismatched toolchain cannot silently load stale binaries.

The intended shape of a deployment, then: search once on a machine like the target, save the artifact, ship it, and load it. Startup becomes a deserialize and a module load rather than a search.

**References**
- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation, ASPLOS 2024](https://dl.acm.org/doi/10.1145/3620665.3640366) ([PDF](https://docs.pytorch.org/assets/pytorch2-2.pdf))
- Luminal `src/graph/artifact.rs`, `crates/luminal_cuda_lite/`, `crates/luminal_metal/`, as of commit `d18376d1`, 2026-09-10

---

#### **The Core and Runtime Boundary**

One structural fact explains how the codebase is laid out, and it is stated plainly in `AGENTS.md`: **the core's compile pipeline ends at egglog saturation.**

`Graph::build_search_space` produces a `SearchSpace`, which is one saturated e-graph per bucket combination plus the registered ops. `Runtime::compile` owns everything after that: search strategy, profiling, and loading. The core never runs a search itself. The genetic machinery in `luminal::search` is offered as utilities that a runtime may use, compose, or ignore.

The three runtimes in the tree take three different options, which is a good demonstration that the boundary is real:

| Runtime | Strategy | Why |
|---|---|---|
| `ReferenceRuntime` | `extract_one_selected`, no search | It is the semantic reference. Its LLIR is HLIR, executed directly, so there is nothing to rank. Used to validate correctness. |
| `MetalRuntime` | the stock `genetic_search` utility | Nothing special to do between steps, so it takes the batteries-included path. |
| `CudaRuntime` | drives the state machine explicitly | It has its own concerns, graph capture, resource caps, module caching, so it steps the search by hand in `crates/luminal_cuda_lite/src/search.rs`. |

A new backend therefore needs to supply four things: op definitions for the ops it can execute, egglog rewrite rules that lower HLIR into those ops, an implementation of `Runtime::compile` that picks a program from the space, and code that loads and executes the selected LLIR. It does not need to reimplement the graph, the shape tracker, the primitives, or the search.

That last point is the real argument for the tiny operator set. Fifteen primitives is not minimalism for its own sake; it is roughly the number of things a new backend has to be able to express before it works at all. The [Luminal docs](https://docs.luminalai.com/docs/why) make the same case for dtypes, devices, and even autograd: keep the core small and add capability back through rules and compilers, in composable pieces.

**References**
- Luminal `AGENTS.md`, `src/op.rs`, `src/search/mod.rs`, `crates/luminal_cuda_lite/src/search.rs`, as of commit `d18376d1`, 2026-09-10
- [Why Luminal](https://docs.luminalai.com/docs/why)

---

#### **Where This Sits**

Every stack below faces the same problem, which is that a high-level graph has to become machine code and there are many ways to do it. They differ on how the rewrites are written, whether writing one destroys the alternative, who picks the final program, and when the kernel text becomes machine code.

| | Rewrites expressed as | Destructive? | Who picks the final program | Kernel machine code produced |
|---|---|---|---|---|
| torch.compile / Inductor | Python passes over an FX graph, plus a registered pattern matcher | yes | the pass order, fixed by the developer | at runtime; Inductor emits Triton for GPU and C++ for CPU |
| XLA | HLO passes, target-independent then target-dependent | yes | the pass pipeline, plus library pattern matching in the backend | at compile time via LLVM, from emitted LLVM IR |
| TVM with Ansor | graph-level passes, plus schedules sampled from a hierarchical space | yes at graph level; the search is over schedules, not graphs | evolutionary search guided by a learned cost model | at tuning time, per operator |
| tinygrad | kernel optimizations applied to a lowered AST | yes, per candidate | BEAM search over kernel implementations, timed on device, cached | at runtime |
| Luminal | egglog rules that union rather than replace | **no** | genetic search over e-graph extractions, timed on device | at runtime via NVRTC or the Metal compiler |

Three observations.

**Luminal is the only row whose rewrites are non-destructive.** That is the actual novelty. Everything else in the table has to be careful about rule ordering because a rewrite forecloses alternatives; Luminal does not, which is why its rule authors can be aggressive and why `AGENTS.md` can forbid Rust post-passes outright.

**Search is not novel; searching the graph is.** TVM's [Ansor](https://arxiv.org/abs/2006.06762) uses evolutionary search with a learned cost model, and tinygrad's [BEAM search](https://docs.tinygrad.org/mnist/) tries many implementations and keeps what is fastest on your hardware, caching the result. Both search **within** an operator, over schedules or kernel parameters, after the graph structure has already been fixed by destructive passes. Luminal's search space is the graph structure itself, including which operators exist at all. That is why fusions like Flash Attention are reachable rather than hand-written.

**Cost model versus stopwatch.** Ansor learns a cost model because tuning every candidate on hardware is expensive; the model is an approximation that buys throughput. tinygrad and Luminal both measure directly and cache or serialize the answer. Luminal's `trials` default of 3 and its per-candidate timeout are the visible price of that choice.

**References**
- [PyTorch 2, ASPLOS 2024](https://dl.acm.org/doi/10.1145/3620665.3640366)
- [XLA architecture](https://openxla.org/xla/architecture)
- [Ansor: Generating High-Performance Tensor Programs for Deep Learning, OSDI 2020](https://arxiv.org/abs/2006.06762)
- [tinygrad documentation](https://docs.tinygrad.org/mnist/)

---

#### **Running It on a Mac**

Everything below was run on an Apple silicon Mac on 2026-09-10.

Install Rust if you have not:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then open a new shell, or source the environment file in your current one:

```bash
. "$HOME/.cargo/env"
```

Clone and run the hello world, which uses `ReferenceRuntime` and is portable CPU code with no GPU involved:

```bash
git clone https://github.com/luminal-ai/luminal && cd luminal && cargo run --release -p simple
```

Two things to expect. The first build is slow, because egglog is pulled from a pinned git revision and compiled from source. And `examples/simple/src/main.rs` calls `display_graph`, which encodes the graph into a URL and opens a browser tab at the Luminal visualizer; delete that line if you would rather it did not.

For an actual model on Apple silicon, Metal is wired up for the Qwen example:

```bash
cargo run --release -p qwen --features metal
```

The first run downloads Qwen3-4B, converts the safetensors weights, compiles the Metal graph, and generates text. Every other example in the tree, llama, gemma, paged_llama, whisper, yolo, and flux among them, is CUDA-only as of this commit.

If you want to watch the compiler work, the environment flags are documented in the README. `SEARCH_LOG=1` prints per-bucket progress and best-so-far metrics, `EGGLOG_LOG=1` prints e-graph build and schedule diagnostics, and `LUMINAL_LOG=1` turns on everything.

---

#### **Takeaways**

- **A destructive rewrite forecloses alternatives.** Applying $$x \times 2 \rightarrow x \ll 1$$ makes $$(x \times y)/y \rightarrow x$$ unable to fire. That is the phase ordering problem, and it is why production rule sets grow guards until they are unmaintainable.
- **Equality saturation records equalities instead of applying rewrites.** Nothing is deleted, so no order is committed to, and the saturated IR encodes many optimized versions at once.
- **An e-graph is the data structure that makes this affordable.** E-nodes have e-classes as children, so one structure represents a multiplicative number of programs, and congruence propagates equality upward for free.
- **Luminal writes every rewrite as an egglog rule ending in `union`**, and forbids Rust-side graph passes, because such a pass would be destructive and would sit outside the search.
- **Choosing a program out of the e-graph is the hard part.** A selection is one e-node per e-class, the space is $$\prod_{c} |c|$$, and optimal extraction is NP-hard and hard to approximate within any constant.
- **Luminal picks by measuring, not modelling.** A genetic search hands candidates to the runtime, which compiles and times them on the real device. Defaults are 100 candidates, generations of 10, 3 trials each.
- **The primitive set is small because backends have to implement it.** Fifteen math ops, with subtraction, division, exp, log, softmax, and matmul all derived in the front end. Movement ops are free because they live in the shape tracker.
- **AOT and JIT are both true, on different clocks.** Everything about the model is decided before it runs; the kernel machine code is produced while the process runs, by NVRTC or the Metal compiler, specialized to a dimension bucket.
- **Artifacts make the search a one-time cost.** Serialize the selected schedule and the compiled modules, ship them, load them.

---

#### **Test Yourself**

Try each from memory before reading the answer.

**1. Give a concrete pair of rewrite rules that interfere, and say what goes wrong.**
$$x \times 2 \rightarrow x \ll 1$$ and $$(x \times y)/y \rightarrow x$$, on $$(a \times 2)/2$$. Fire the first and the multiply is gone, so the second cannot match and you are stuck with $$(a \ll 1)/2$$ instead of $$a$$. The correct order depends on the expression, which is the phase ordering problem.

**2. What is an e-node, precisely, and how does it differ from a tree node?**
An operator together with a list of children, where each child is an e-class rather than a specific term. A tree node points at one subexpression; an e-node points at a set of equivalent subexpressions and stands for the operation applied to any member.

**3. How do you read a single program out of an e-graph?**
Pick one e-node from the root e-class, then recursively pick one e-node from each of its children's e-classes. Every complete set of choices yields a valid term.

**4. What does congruence mean here, and why does a plain union-find not suffice?**
Congruence is that $$a \equiv b$$ implies $$f(a) \equiv f(b)$$, so equality propagates upward through operators. A union-find tracks equality of leaves but does not close it under application; maintaining that closure as merges happen is the extra work, and egg's rebuilding technique is what makes it fast.

**5. In the squeeze rule, what does `(union ?sum ?new_sum)` do, and what would a normal compiler have written there instead?**
It asserts the two summations are equal and merges their e-classes, keeping both spellings available. A normal compiler would replace the old node with the new one, deleting the 4D form permanently.

**6. What is the difference between `(run r)` and `(saturate r)` in a schedule, and why does Luminal use both?**
`(run r)` applies a ruleset once; `(saturate r)` applies it to a fixed point. Algebraic rules and the dtype analysis are saturated because they are cheap and need to be complete; the expensive lowering and fusion-discovery rules are run once per outer iteration to avoid re-running large joins.

**7. What is a genome in Luminal's search, and how large is the space it lives in?**
A choice of one e-node per searchable e-class, stored as `IndexedChoiceSet`. The space has $$\prod_{c} |c|$$ members, the product over classes of the number of e-nodes in each.

**8. Why does Luminal not just assign costs to e-nodes and extract the cheapest tree?**
Because a good cost function has to account for shared subexpressions, which makes extraction a global optimization problem that is NP-hard and hard to approximate within any constant; and because on a GPU no static cost estimate is trustworthy anyway. So it compiles candidates and times them.

**9. What are the three outcomes a runtime can report for a candidate, and how do they differ?**
`Measured` with a metric, which is ranked and counts against the budget. `Rejected`, meaning it was not viable before evaluation, for example it failed to compile or blew a resource cap, and does not count against the budget. `Invalid`, meaning evaluation started but produced no usable metric, which counts against the budget but is never ranked.

**10. What is a dimension bucket, and what does declaring one actually cause to happen?**
An inclusive range over a symbolic dimension with a representative value. Declaring buckets causes `build_search_space` to produce one saturated e-graph per bucket combination, the search to select one program per bucket, and the runtime to dispatch on the real value at execution time.

**11. Explain "shape-specific kernels compiled at runtime" in one sentence each for "shape-specific" and "at runtime".**
Shape-specific: within a bucket the dynamic dimension is pinned to a representative, so trip counts, tiles, and unroll factors can be baked into the emitted source rather than branched on. At runtime: that source only exists once the compiler has searched and the bucket values are known, so NVRTC or the Metal compiler turns it into machine code inside the running process.

**12. Is Luminal AOT or JIT? Defend the answer.**
Both, on different clocks. AOT relative to the model: the entire network is a static graph, so devices, dtypes, allocation, fusion, and kernel choice are settled before a single number moves, and there is no interpreter at inference. JIT relative to the process: kernel machine code is generated while the process runs. Unlike `torch.compile`, there is no eager fallback and no guard-triggered recompile.

**13. Why does a matmul produce no `MatMul` node, and why is that good for the search?**
The front end expands it into a broadcast multiply plus a sum reduce, with the broadcast and permute absorbed into the shape tracker. Because there is no matmul node to protect, the rewrite rules are free to regroup that arithmetic into whatever kernel is fastest, including fusing it with its neighbours, rather than being forced to preserve a hand-drawn boundary.

**14. Why does `AGENTS.md` forbid Rust passes that rewrite the extracted graph?**
Such a pass would necessarily be destructive, and it would run after the search, so its decisions could never be compared against the alternatives it destroyed. One of them and the non-destructive property of the whole pipeline is gone.

**15. What does `ReferenceRuntime` search, and why?**
Nothing. It extracts one program and runs it. Its LLIR is HLIR executed directly, so it is a semantic reference for validating that everything the search selects computes the same answer, and there is nothing to rank.

---

#### **Wrapping up**

If one thing survives from this post, make it the shape of the trade. A conventional compiler spends its intelligence deciding *whether* to apply a rewrite, and it has to be right, because applying one destroys the alternative. Luminal spends nothing there. It applies every rewrite it knows, keeps all of them, and moves the entire difficulty to the end, where the question is no longer "will this help?" but "which of these is fastest?", and that question can be answered with a stopwatch instead of a heuristic.

Everything else is downstream. The operator set is small because backends must implement it and rules must match against it. Movement ops are free because a rewrite should be able to regroup arithmetic without paying for copies. Shapes are symbolic so the compiler keeps visibility, and bucketed so it can still specialize. The core stops at saturation because choosing is the backend's business. Artifacts exist because measuring a hundred candidates is only acceptable if you do it once.

Whether it wins is a separate question from whether it is well-designed, and it is a question about numbers I have not measured. What I can say is that the design is unusually coherent: pull on any thread and you end up back at the same sentence, which is that a rewrite should add information rather than remove it.

If you find a mistake anywhere in here, please let me know and I'll fix it.
