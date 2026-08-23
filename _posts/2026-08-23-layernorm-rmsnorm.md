---
layout: post
title: LayerNorm and RMSNorm - Forward Pass, Backward Pass, and Every Derivative
date: 2026-08-23 14:00:00-0400
featured: false
description: Deriving the normalization backward pass from first principles, why it has exactly three terms, and why RMSNorm has two
tags: ML
categories:
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Hello all. This continues from the [MLP forward and backward pass post](/blog/2026/mlp-forward-backward/), and uses the `einsum` notation from the [einsum post](/blog/2026/einsum/). The conventions from the MLP post carry over unchanged, so I won't restate them: a subscripted symbol is a scalar, a bare symbol is the whole array, vectors are columns, $$\odot$$ is elementwise, and $$\partial L/\partial u$$ always has the shape of $$u$$. We derive in index form first and read off the packed form last.

This post does LayerNorm and RMSNorm end to end. It's a good next step after the MLP for one specific reason: normalization is the first operation where an output depends on *every* input through a shared statistic. The MLP had one of these, softmax, and it was the only place where the chain rule's sum didn't collapse. In a norm, the mean and the variance depend on all $$D$$ components, so nothing collapses anywhere, and the full machinery gets a workout.

The result we're heading for is that LayerNorm's input gradient has exactly three terms:

$$
\frac{\partial L}{\partial x} = \frac{1}{s}\Big[\hat g \;-\; \underbrace{c_1}_{\text{shift}} \;-\; \underbrace{c_2\,\hat x}_{\text{scale}}\Big]
$$

and that the two subtracted terms are there because LayerNorm is invariant to shifting and scaling its input. RMSNorm drops the mean subtraction, which costs it the shift invariance, and its backward pass is the same expression with that one term removed. One structural fact explains the cost difference in both directions, and it hands you two free correctness checks.

New indices for this post: $$d$$ and $$e$$, both running $$1 \dots D$$ over the feature dimension. $$d$$ is the index we're differentiating with respect to, $$e$$ is the dummy being summed. Code is PyTorch.

Let's get started.

---

#### **What normalization does**

Take a single token's feature vector $$x \in \mathbb{R}^{D}$$. Normalizing it means: subtract its mean, divide by its standard deviation, then apply a learned scale and shift.

$$
x \;\longrightarrow\; \hat x = \frac{x - \mu}{s} \;\longrightarrow\; y = \gamma \odot \hat x + \beta
$$

The middle step forces the vector to have mean 0 and variance 1. That's the stabilizing part: whatever the previous layer did, the next layer sees something with known first and second moments, so activations can't drift or blow up as they pass through a deep stack.

The third step looks like it undoes the second, and in a sense it does. If the network wants a mean of 3 and a standard deviation of 7, it can set $$\beta_d = 3$$ and $$\gamma_d = 7$$. The difference is that those are now *learned parameters* rather than whatever happened to come out of the layer below. You remove two degrees of freedom from the data and hand them back as weights. $$\gamma$$ initializes to 1 and $$\beta$$ to 0, so the layer starts as pure normalization and learns to deviate if that helps.

The important structural detail is **which axis gets reduced**. The mean and variance here are over $$D$$, the feature dimension, computed independently for each token. Nothing is shared across tokens or across the batch. That's what makes it usable in a transformer: training and inference behave identically, a batch of 1 works exactly like a batch of 512, and sequence length is irrelevant.

---

#### **LayerNorm forward**

Index form first, then packed, as usual.

$$
\mu = \frac{1}{D}\sum_e x_e \tag{1}
$$

$$
v = \frac{1}{D}\sum_e (x_e - \mu)^2 \tag{2}
$$

$$
s = \sqrt{v + \varepsilon}, \qquad \hat x_d = \frac{x_d - \mu}{s} \tag{3}
$$

$$
y_d = \gamma_d \hat x_d + \beta_d \tag{4}
$$

$$
y = \gamma \odot \hat x + \beta \qquad (D \times 1)
$$

$$\mu$$, $$v$$ and $$s$$ are **scalars**, one per token, not vectors. $$x$$, $$\hat x$$, $$\gamma$$, $$\beta$$, $$y$$ are all $$D \times 1$$. The $$\varepsilon$$ is there so that $$s$$ can't be zero when a token's features are all identical; typical values are $$10^{-5}$$ or $$10^{-6}$$. It's inside the square root, not added after, and that placement matters for the derivation below.

Note that $$v$$ uses the biased variance, dividing by $$D$$ rather than $$D-1$$. Every framework does this for normalization layers, and it's what makes $$\sum_d \hat x_d^2 = D$$ exactly when $$\varepsilon = 0$$, an identity we'll use.

---

#### **The derivative we need**

Every gradient below flows through $$\hat x$$, so the object we need is $$\partial \hat x_d / \partial x_e$$: a full $$D \times D$$ Jacobian, because $$\mu$$ and $$v$$ depend on all of $$x$$ and therefore every output component depends on every input component. Same situation as the softmax Jacobian in the MLP post.

Build it from the pieces. From (1), each $$x_e$$ contributes $$1/D$$ to the mean:

$$
\frac{\partial \mu}{\partial x_e} = \frac{1}{D} \tag{5}
$$

For the variance, differentiate (2) remembering that $$\mu$$ itself depends on $$x_e$$:

$$
\frac{\partial v}{\partial x_e}
= \frac{2}{D}\sum_f (x_f - \mu)\left(\delta_{fe} - \frac{1}{D}\right)
= \frac{2}{D}\left[(x_e - \mu) - \frac{1}{D}\sum_f (x_f - \mu)\right]
$$

The second sum is zero, since deviations from the mean always sum to zero, which is worth noticing because it's the reason the variance derivative comes out as clean as it does:

$$
\frac{\partial v}{\partial x_e} = \frac{2}{D}(x_e - \mu) \tag{6}
$$

Then $$s = \sqrt{v + \varepsilon}$$, so $$\partial s / \partial v = 1/(2s)$$, and:

$$
\frac{\partial s}{\partial x_e} = \frac{1}{2s}\cdot\frac{2}{D}(x_e - \mu) = \frac{x_e - \mu}{D\,s} = \frac{\hat x_e}{D} \tag{7}
$$

That is a genuinely nice result: the derivative of the standard deviation with respect to an input is just that input's own normalized value, over $$D$$.

Now assemble, using the quotient rule on $$\hat x_d = (x_d - \mu)/s$$ and remembering both $$\mu$$ and $$s$$ depend on $$x_e$$:

$$
\frac{\partial \hat x_d}{\partial x_e}
= \frac{\left(\delta_{de} - \frac{1}{D}\right)s - (x_d - \mu)\frac{\hat x_e}{D}}{s^2}
= \frac{1}{s}\left[\delta_{de} - \frac{1}{D} - \frac{\hat x_d \hat x_e}{D}\right] \tag{8}
$$

Three terms, and they'll stay three terms all the way to the end. The $$\delta_{de}$$ is the direct path (changing $$x_d$$ changes $$\hat x_d$$), the $$1/D$$ is the path through the mean, and the $$\hat x_d \hat x_e / D$$ is the path through the standard deviation.

---

#### **LayerNorm backward**

Let $$g_d = \partial L / \partial y_d$$ be the gradient arriving from downstream.

##### **The parameters**

From (4), $$y_d$$ depends on $$\gamma_d$$ and $$\beta_d$$ only, so no sum survives:

$$
\frac{\partial L}{\partial \gamma_d} = g_d \hat x_d,
\qquad
\frac{\partial L}{\partial \beta_d} = g_d
\tag{9}
$$

$$
\frac{\partial L}{\partial \gamma} = g \odot \hat x, \qquad
\frac{\partial L}{\partial \beta} = g
\qquad (D \times 1) \;\checkmark
$$

##### **Through the affine, into the normalized vector**

$$
\hat g_d := \frac{\partial L}{\partial \hat x_d} = g_d \gamma_d
\qquad\Longrightarrow\qquad
\hat g = g \odot \gamma
\tag{10}
$$

This is the gradient we actually propagate. Everything below is in terms of $$\hat g$$, not $$g$$.

##### **The input**

Contract $$\hat g$$ with the Jacobian (8), summing over $$d$$:

$$
\frac{\partial L}{\partial x_e}
= \sum_d \hat g_d \frac{\partial \hat x_d}{\partial x_e}
= \frac{1}{s}\sum_d \hat g_d \left[\delta_{de} - \frac{1}{D} - \frac{\hat x_d \hat x_e}{D}\right]
$$

Distribute the sum over the three terms. The delta picks out $$\hat g_e$$; the second term gives the mean of $$\hat g$$; the third gives $$\hat x_e$$ times the mean of $$\hat g \odot \hat x$$:

$$
\frac{\partial L}{\partial x_e}
= \frac{1}{s}\left[\hat g_e - \frac{1}{D}\sum_d \hat g_d - \hat x_e \cdot \frac{1}{D}\sum_d \hat g_d \hat x_d\right]
$$

Name the two scalars, since they're the only things that need reducing:

$$
c_1 = \frac{1}{D}\sum_d \hat g_d,
\qquad
c_2 = \frac{1}{D}\sum_d \hat g_d \hat x_d
\tag{11}
$$

$$
\boxed{\;\frac{\partial L}{\partial x} = \frac{1}{s}\Big[\hat g - c_1 - c_2\,\hat x\Big]\;}
\qquad (D \times 1) \;\checkmark
\tag{12}
$$

($$c_1$$ is a scalar being subtracted from every component, so read it as $$c_1 \mathbf{1}$$ if you prefer to keep the shapes explicit.)

##### **What the three terms actually are**

This is worth spelling out, because it turns a formula you'd have to memorize into one you can reconstruct.

Look at $$c_1$$ and $$c_2$$ as projections. Let $$\mathbf{1}$$ be the all-ones vector. Then $$\mathbf{1}\cdot\mathbf{1} = D$$, and (at $$\varepsilon = 0$$) $$\hat x \cdot \hat x = D$$ as well. So:

$$
c_1 \mathbf{1} = \frac{\hat g \cdot \mathbf{1}}{\mathbf{1}\cdot\mathbf{1}}\,\mathbf{1}
= \text{proj}_{\mathbf{1}}(\hat g),
\qquad
c_2 \hat x = \frac{\hat g \cdot \hat x}{\hat x \cdot \hat x}\,\hat x
= \text{proj}_{\hat x}(\hat g)
$$

So equation (12) says: **take the incoming gradient, remove its component along $$\mathbf{1}$$, remove its component along $$\hat x$$, and divide by $$s$$.** It's one step of Gram-Schmidt against two specific directions.

And because $$\sum_d \hat x_d = 0$$, those two directions are orthogonal to each other, so the projections are independent and can be removed in either order without correction terms. (I checked this numerically: $$\mathbf{1}\cdot\hat x$$ comes out at $$10^{-16}$$, and (12) agrees with explicit Gram-Schmidt to $$2 \times 10^{-16}$$.)

Which raises the obvious question: why *those* two directions?

---

#### **Two free correctness checks**

Because those are exactly the directions LayerNorm ignores.

**Shift.** Adding a constant to every component of $$x$$ changes nothing downstream. $$\mu$$ shifts by the same constant, $$x - \mu$$ is unchanged, $$v$$ is unchanged, so $$y$$ is unchanged:

$$
\mathrm{LN}(x + a\mathbf{1}) = \mathrm{LN}(x) \quad \text{for any } a
$$

If moving along $$\mathbf{1}$$ can't change the output, it can't change the loss, so the directional derivative along $$\mathbf{1}$$ must be zero:

$$
\sum_d \frac{\partial L}{\partial x_d} = 0 \tag{13}
$$

**Scale.** Multiplying $$x$$ by a positive constant also changes nothing, *provided* $$\varepsilon = 0$$: $$\mu$$ scales by $$c$$, $$s$$ scales by $$c$$, and the ratio is unchanged. So the directional derivative along $$\hat x$$ is zero too:

$$
\sum_d \hat x_d \frac{\partial L}{\partial x_d} = 0 \tag{14}
$$

Both fall straight out of (12): substituting gives $$\sum_d \hat g_d - Dc_1 - c_2\sum_d \hat x_d$$, and $$\sum_d \hat g_d = Dc_1$$ and $$\sum_d \hat x_d = 0$$ kill it. Similarly for (14).

**Two invariances, two subtracted terms.** That's the whole structure of the backward pass. A normalization layer throws away information, and the gradient must be blind in exactly the directions the forward pass was blind in.

##### **The $$\varepsilon$$ caveat, stated precisely**

These two identities are not equally exact, and it's worth being honest about which is which.

**(13) is exact for any $$\varepsilon$$.** Shift invariance doesn't involve $$s$$ at all, so $$\varepsilon$$ never enters. Numerically, $$\sum_d \partial L/\partial x_d$$ comes out as exactly $$0$$ in float64.

**(14) is exact only at $$\varepsilon = 0$$.** With $$\varepsilon > 0$$, LayerNorm isn't quite scale invariant, because $$s = \sqrt{c^2 v + \varepsilon}$$ isn't $$c\sqrt{v}$$. The identity picks up a residual, and you can write it down in closed form. Since $$\sum_d \hat x_d^2 = D\,v/(v+\varepsilon)$$ rather than $$D$$:

$$
\sum_d \hat x_d \frac{\partial L}{\partial x_d} = \frac{c_2 D}{s}\cdot\frac{\varepsilon}{v + \varepsilon}
$$

I checked this against the actual gradients over 150 random trials at $$\varepsilon = 10^{-5}$$, $$10^{-3}$$ and $$10^{-1}$$, and the formula holds to better than $$10^{-15}$$ in the worst case. In practice $$\varepsilon \ll v$$, so the residual lands around $$10^{-7}$$ and (14) is a perfectly good check with a loose tolerance. Just don't assert it to machine precision and then wonder why it fails.

Used as tests, these are excellent: they need no reference implementation, no finite differences, and no autograd. If a hand-written norm backward is wrong, (13) almost always catches it immediately.

---

#### **RMSNorm**

RMSNorm removes the mean subtraction. That's the entire change.

$$
r = \sqrt{\frac{1}{D}\sum_e x_e^2 + \varepsilon},
\qquad
\hat x_d = \frac{x_d}{r},
\qquad
y_d = \gamma_d \hat x_d
\tag{15}
$$

No $$\mu$$, and conventionally no $$\beta$$ either. It divides by the root mean square instead of the standard deviation, which are the same thing only when the mean is zero.

The derivation runs identically. With $$m = \frac{1}{D}\sum_e x_e^2$$ we get $$\partial m/\partial x_e = 2x_e/D$$, so:

$$
\frac{\partial r}{\partial x_e} = \frac{1}{2r}\cdot\frac{2x_e}{D} = \frac{x_e}{D\,r} = \frac{\hat x_e}{D}
$$

the same clean form as (7). And the Jacobian:

$$
\frac{\partial \hat x_d}{\partial x_e}
= \frac{\delta_{de}}{r} - \frac{x_d}{r^2}\cdot\frac{\hat x_e}{D}
= \frac{1}{r}\left[\delta_{de} - \frac{\hat x_d \hat x_e}{D}\right]
\tag{16}
$$

Compare with (8). The $$1/D$$ term is gone, because there's no mean for $$x_e$$ to flow through. Contracting with $$\hat g$$:

$$
\boxed{\;\frac{\partial L}{\partial x} = \frac{1}{r}\Big[\hat g - c_2\,\hat x\Big]\;}
\qquad
\frac{\partial L}{\partial \gamma} = g \odot \hat x
\tag{17}
$$

**LayerNorm's result with the $$c_1$$ term deleted.** Nothing else changes, and $$c_2$$ is defined exactly as before.

The invariance story predicts this. RMSNorm is still scale invariant, so it still has the $$\hat x$$ projection. It is *not* shift invariant, since adding a constant changes $$\sum x_e^2$$ and there's no mean to absorb it. So it loses the $$\mathbf{1}$$ projection, and with it $$c_1$$. Numerically: $$\mathrm{RMS}(x + 5\mathbf{1})$$ differs from $$\mathrm{RMS}(x)$$ by $$2.4$$, and correspondingly $$\sum_d \partial L/\partial x_d = -0.123$$, nowhere near zero, while $$\sum_d \hat x_d\,\partial L/\partial x_d$$ is zero to machine precision at $$\varepsilon = 0$$.

---

#### **Side by side**

| | LayerNorm | RMSNorm |
|---|---|---|
| Statistic | mean and variance | mean square |
| Parameters | $$\gamma$$, $$\beta$$ | $$\gamma$$ |
| Shift invariant | yes, exactly | **no** |
| Scale invariant | yes (at $$\varepsilon = 0$$) | yes (at $$\varepsilon = 0$$) |
| Forward reductions | 2 ($$\sum x$$, $$\sum x^2$$) | 1 ($$\sum x^2$$) |
| Backward reductions | 2 ($$c_1$$, $$c_2$$) | 1 ($$c_2$$) |
| Backward terms | 3 | 2 |

Every row of the bottom half follows from the top half. That's the argument for RMSNorm in a sentence: dropping the mean buys you one fewer reduction in each direction, one fewer parameter tensor, and one fewer term in the backward pass, and the empirical finding is that the shift invariance wasn't doing much work anyway.

---

#### **Residual connections, and the fork rule**

Norms in a transformer never appear alone; they come attached to a residual. And residuals introduce the one core backprop rule the MLP post never needed.

In the MLP, every intermediate value was consumed exactly once. In a residual block, $$x$$ is used **twice**: once by the identity path and once by the sublayer.

$$
y = x + F(x) \tag{18}
$$

The rule for that is not new machinery; it's the same many-routes chain rule from the MLP post, with the routes now being different *consumers* rather than different components. If $$u$$ feeds into several places, perturbing $$u$$ perturbs all of them, and the effects add:

$$
\boxed{\;\frac{\partial L}{\partial u} = \sum_{\text{uses of } u} \frac{\partial L}{\partial u}\bigg|_{\text{that use}}\;}
$$

So for (18):

$$
\frac{\partial L}{\partial x} = \underbrace{\frac{\partial L}{\partial y}}_{\text{identity path}} + \underbrace{\left(\frac{\partial F}{\partial x}\right)^{\!\top}\frac{\partial L}{\partial y}}_{\text{sublayer path}}
$$

In code this is `+=` rather than `=`, and forgetting it is a silent, plausible-looking bug: you get a gradient, it has the right shape, it's just wrong. (I checked: for a random $$y = x + \tanh(Ax)$$, taking only the identity path is off by $$1.4$$ in max absolute error, while the sum matches autograd to $$2 \times 10^{-16}$$.)

##### **Pre-norm and post-norm**

Where the norm sits relative to the fork is the whole pre-norm versus post-norm question.

$$
\text{post-norm:} \quad y = \mathrm{LN}\big(x + F(x)\big)
\qquad\qquad
\text{pre-norm:} \quad y = x + F\big(\mathrm{LN}(x)\big)
$$

In pre-norm, the identity path is a bare $$+$$. The gradient reaching $$x$$ contains $$\partial L/\partial y$$ *unmodified*, plus whatever comes through the sublayer. Stack a hundred blocks and there's still a clean path from the loss to every layer, with nothing attenuating it. That's the "gradient highway".

In post-norm, the norm sits *on* the residual path, so every gradient must pass through the norm's Jacobian once per layer. And we know exactly what that Jacobian does: by (12), it strips out the components along $$\mathbf{1}$$ and $$\hat x$$ and rescales by $$1/s$$. Doing that a hundred times in a row is why deep post-norm transformers need learning-rate warmup to train stably, and why essentially everything modern is pre-norm.

---

#### **The code**

Both norms, forward and backward by hand, checked against autograd. The two reductions are written with `einsum` because they're genuine contractions and the strings map straight onto (11); the elementwise work stays as ordinary operators.

```python
import torch

torch.set_default_dtype(torch.float64)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu   = x.mean()                                  # eq (1), scalar
    v    = ((x - mu) ** 2).mean()                    # eq (2), scalar
    s    = torch.sqrt(v + eps)                       # eq (3)
    xhat = (x - mu) / s
    return gamma * xhat + beta, xhat, s              # eq (4)


def layernorm_backward(g, gamma, xhat, s):
    D    = xhat.shape[0]
    ghat = g * gamma                                 # eq (10)
    c1   = torch.einsum("d->", ghat) / D             # eq (11)
    c2   = torch.einsum("d,d->", ghat, xhat) / D     # eq (11)
    dx   = (ghat - c1 - c2 * xhat) / s               # eq (12)
    return dx, g * xhat, g.clone()                   # dx, dgamma, dbeta  eq (9)


def rmsnorm_forward(x, gamma, eps=1e-5):
    r    = torch.sqrt((x ** 2).mean() + eps)         # eq (15)
    xhat = x / r
    return gamma * xhat, xhat, r


def rmsnorm_backward(g, gamma, xhat, r):
    D    = xhat.shape[0]
    ghat = g * gamma
    c2   = torch.einsum("d,d->", ghat, xhat) / D     # the only reduction
    dx   = (ghat - c2 * xhat) / r                    # eq (17), no c1
    return dx, g * xhat                              # dx, dgamma
```

Checking the forward against the framework and the backward against autograd:

```python
torch.manual_seed(0)

D, eps = 9, 1e-5
x     = torch.randn(D) * 3 + 2                       # deliberately not zero-mean
gamma = torch.randn(D)
beta  = torch.randn(D)

y, _, _ = layernorm_forward(x, gamma, beta, eps)
ref     = torch.nn.functional.layer_norm(x, (D,), gamma, beta, eps)
print(f"forward vs F.layer_norm: {(y - ref).abs().max():.2e}")

xg = x.clone().requires_grad_()
gg = gamma.clone().requires_grad_()
bg = beta.clone().requires_grad_()
w  = torch.randn(D)                                  # arbitrary downstream
yg, xhat, s = layernorm_forward(xg, gg, bg, eps)
(w * yg).sum().backward()                            # so g = w

with torch.no_grad():
    dx, dgamma, dbeta = layernorm_backward(w, gamma, xhat, s)

print(f"dx     {(dx     - xg.grad).abs().max():.2e}")
print(f"dgamma {(dgamma - gg.grad).abs().max():.2e}")
print(f"dbeta  {(dbeta  - bg.grad).abs().max():.2e}")
print(f"check (13) sum(dx)        = {dx.sum():+.2e}")
print(f"check (14) sum(xhat * dx) = {(xhat * dx).sum():+.2e}")
```

```
forward vs F.layer_norm: 2.22e-16
dx     5.55e-17
dgamma 0.00e+00
dbeta  0.00e+00
check (13) sum(dx)        = +0.00e+00
check (14) sum(xhat * dx) = +7.47e-07
```

Exactly as predicted: (13) is identically zero, and (14) sits at $$7 \times 10^{-7}$$, which is the $$\varepsilon$$ residual and not an error. RMSNorm checks out the same way against `F.rms_norm`.

---

#### **The token dimension, in one index**

Real input is $$N$$ tokens, $$X$$ of shape $$(N, D)$$. Every statistic is per token, so all the reductions gain an $$n$$ that is **kept**, while the parameter gradients gain an $$n$$ that is **summed away**, because $$\gamma$$ and $$\beta$$ are shared across every token. This is the fork rule again: one parameter used $$N$$ times receives $$N$$ contributions, and they add.

`einsum` makes the distinction visible, since it's entirely a question of which letters survive the arrow:

```python
Ghat = G * gamma                                     # (N, D)

c1 = torch.einsum("nd->n",    Ghat)       / D        # (N,)  n kept, d summed
c2 = torch.einsum("nd,nd->n", Ghat, Xhat) / D        # (N,)  n kept, d summed

dX = (Ghat - c1[:, None] - c2[:, None] * Xhat) / s[:, None]   # (N, D)

dgamma = torch.einsum("nd,nd->d", G, Xhat)           # (D,)  n summed, d kept
dbeta  = torch.einsum("nd->d",    G)                 # (D,)  n summed, d kept
```

Read the four strings and the entire batching story is right there. `->n` for per-token statistics, `->d` for shared parameters, and the letter that disappears is the one being summed over. Verified against autograd at $$N = 7$$, $$D = 11$$: `dX` matches to $$1.1\times10^{-16}$$, `dgamma` to $$2.2\times10^{-16}$$, and `dbeta` exactly.

---

#### **A note on kernels**

Worth a paragraph, since this is where norms differ most from the MLP.

A norm does $$O(D)$$ arithmetic and moves $$O(D)$$ bytes, so its arithmetic intensity is a small constant, independent of $$D$$. A matmul's grows with the tile size. Norms are therefore **memory bound**, always, and the only thing that matters for making them fast is how many times you touch memory. Written naively as separate framework ops, a LayerNorm forward is around five kernel launches, each streaming the whole tensor in and out; fused into one kernel it's a single read and a single write. That's the whole game, and it's why every serious implementation ships a fused norm.

Two consequences fall out of the derivation:

**One pass suffices in each direction.** Forward needs $$\sum_d x_d$$ and $$\sum_d x_d^2$$, and both can be accumulated in the same sweep, recovering the variance as $$\mathbb{E}[x^2] - \mathbb{E}[x]^2$$ afterwards (or with Welford's algorithm if you're worried about cancellation). Backward needs $$c_1$$ and $$c_2$$ from (11), which are two reductions over the *same* two vectors, so they also fuse into one sweep. Then a second pass applies (12) elementwise.

**What to save from the forward pass is a real trade-off.** The backward needs $$\hat x$$, $$s$$ and $$\gamma$$. You can store $$\hat x$$, which costs $$D$$ values per token, or store just $$\mu$$ and the reciprocal $$1/s$$ (two scalars per token) and recompute $$\hat x = (x - \mu)/s$$ from the input you already have to read. Kernels overwhelmingly do the latter: it turns $$D$$ values of activation memory per token into 2, at the cost of arithmetic that was free anyway because the kernel is memory bound. Storing the reciprocal rather than $$s$$ also replaces a per-element divide with a multiply.

RMSNorm's advantage in this framing is concrete: one accumulator instead of two in the forward, one instead of two in the backward, one scalar of saved state per token instead of two, and no $$\beta$$ gradient to reduce.

---

#### **Wrapping up**

The mechanical part is the same as always. Write the Jacobian of the normalization in index form, contract the incoming gradient against it, and the terms sort themselves out. Equation (8) is the only real work in the post; everything after it is bookkeeping.

The part worth remembering is the structure. A normalization layer is invariant to certain changes in its input, and its backward pass removes exactly those directions from the gradient. LayerNorm ignores shifts and scales, so it subtracts two projections. RMSNorm ignores only scales, so it subtracts one. If you remember that, you can reconstruct both formulas without looking them up, and you get two assertions you can drop into a test that need no reference implementation at all.

And the fork rule, which we'll lean on constantly from here: a value used more than once accumulates the gradients from each use. Attention is next, where $$x$$ feeds Q, K and V simultaneously and that rule stops being a footnote.

If you find a mistake anywhere in here, please let me know and I'll fix it.
