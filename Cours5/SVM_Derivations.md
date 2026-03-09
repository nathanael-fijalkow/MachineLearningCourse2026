# Support Vector Machines (SVM)

## Notation

| Symbol | Meaning |
|--------|---------|
| $m$ | Number of training examples |
| $n$ | Number of features |
| $x^{(i)} \in \mathbb{R}^n$ | The $i$-th training example |
| $y^{(i)} \in \{-1, +1\}$ | The label of the $i$-th example |
| $w \in \mathbb{R}^n$ | Weight vector (normal to the decision boundary) |
| $b \in \mathbb{R}$ | Bias (offset) term |

---

## Part 1: Maximising the Margin

### 1.1 The Decision Boundary

A linear classifier separates two classes using a **hyperplane**:

$$w^T x + b = 0$$

For a new point $x$, we predict:
- $y = +1$ if $w^T x + b > 0$
- $y = -1$ if $w^T x + b < 0$

Many hyperplanes can separate the same data. Which one should we choose?

### 1.2 What is the Margin?

The **margin** is the distance between the decision boundary and the closest data points from either class. These closest points are called **support vectors**.

The margin from a point $x^{(i)}$ to the hyperplane $w^T x + b = 0$ is:

$$\gamma^{(i)} = \frac{w^T x^{(i)} + b}{\|w\|}$$

The overall margin of the classifier is the smallest margin across all training examples:

$$\gamma = \min_{i=1,\ldots,m} \gamma^{(i)}$$

### 1.3 The SVM Optimisation Problem

The SVM idea: **find the hyperplane that maximises the margin**.

$$\max_{w, b} \quad \gamma = \max_{w, b} \quad \frac{1}{\|w\|} \min_{i} \; y^{(i)}(w^T x^{(i)} + b)$$

Since $w$ and $b$ can be rescaled freely (multiplying both by a constant does not change the hyperplane), we fix the **functional margin** (that is, the margin computed without dividing by $\|w\|$) of the closest point to be exactly 1:

$$y^{(i)}(w^T x^{(i)} + b) \geq 1 \qquad \text{for all } i$$

Under this convention, the margin is $\frac{1}{\|w\|}$. So maximising the margin becomes:

$$\max_{w, b} \frac{1}{\|w\|} \quad \Longleftrightarrow \quad \min_{w, b} \frac{1}{2}\|w\|^2$$

The **primal SVM optimisation problem** (hard-margin) is:

$$\boxed{\min_{w, b} \frac{1}{2}\|w\|^2 \qquad \text{subject to} \quad y^{(i)}(w^T x^{(i)} + b) \geq 1, \quad i = 1, \ldots, m}$$

---

## Part 2: Lagrange Multipliers

### 2.1 Constrained Optimisation in General

Suppose we want to solve:

$$\min_x f(x) \qquad \text{subject to} \quad g(x) \leq 0$$

The idea of **Lagrange multipliers** is to fold the constraint into the objective by introducing a new variable $\alpha \geq 0$ (the multiplier):

$$\mathcal{L}(x, \alpha) = f(x) + \alpha \, g(x)$$

This function $\mathcal{L}$ is called the **Lagrangian**.

### 2.2 Intuition

Think of $\alpha$ as the "price" of violating the constraint:

- If the constraint $g(x) \leq 0$ is satisfied, the term $\alpha \, g(x) \leq 0$ gives a bonus (reducing the objective), so the multiplier does not hurt.
- If the constraint is violated ($g(x) > 0$), the term $\alpha \, g(x) > 0$ acts as a penalty that increases the objective.
- Making $\alpha$ as large as possible maximises the penalty for constraint violation.

This gives us the **minimax** formulation:

$$\min_x \max_{\alpha \geq 0} \mathcal{L}(x, \alpha) = \min_x f(x) \quad \text{(subject to } g(x) \leq 0\text{)}$$

**Why?** For a fixed $x$:
- If $g(x) \leq 0$: $\max_{\alpha \geq 0} \alpha \, g(x) = 0$ (choose $\alpha = 0$), so $\max_\alpha \mathcal{L} = f(x)$.
- If $g(x) > 0$: $\max_{\alpha \geq 0} \alpha \, g(x) = +\infty$ (choose $\alpha \to \infty$), so $\max_\alpha \mathcal{L} = +\infty$.

So the outer $\min_x$ naturally avoids points where $g(x) > 0$, effectively enforcing the constraint.

### 2.3 Multiple Constraints

For the general problem:

$$\min_x f(x) \qquad \text{subject to} \quad g_i(x) \leq 0, \quad i = 1, \ldots, m$$

The Lagrangian is:

$$\mathcal{L}(x, \alpha) = f(x) + \sum_{i=1}^{m} \alpha_i \, g_i(x), \qquad \alpha_i \geq 0$$

### 2.4 The Dual Problem

The **primal problem** is: $\min_x \max_{\alpha \geq 0} \mathcal{L}(x, \alpha)$.

The **dual problem** is obtained by swapping $\min$ and $\max$:

$$\max_{\alpha \geq 0} \min_x \mathcal{L}(x, \alpha)$$

**Weak duality** states: dual optimum $\leq$ primal optimum (always true).

**Strong duality** states: dual optimum $=$ primal optimum. This holds when $f$ is convex and the constraints $g_i$ are convex (and a mild regularity condition called Slater's condition is satisfied). For SVM, strong duality holds.

### 2.5 The KKT Conditions

At the optimal solution, the **Karush-Kuhn-Tucker (KKT) conditions** must hold:

1. **Stationarity:** $\nabla_x \mathcal{L} = 0$
2. **Primal feasibility:** $g_i(x) \leq 0$ for all $i$
3. **Dual feasibility:** $\alpha_i \geq 0$ for all $i$
4. **Complementary slackness:** $\alpha_i \, g_i(x) = 0$ for all $i$

Condition 4 is the most informative: it says that for each constraint, **either** the constraint is active ($g_i(x) = 0$) **or** the multiplier is zero ($\alpha_i = 0$). In the SVM context, this means:

- $\alpha_i > 0$ only if $y^{(i)}(w^T x^{(i)} + b) = 1$ — the point is on the margin (it is a **support vector**)
- $\alpha_i = 0$ for points strictly inside the margin — they do not influence the solution

---

## Part 3: Application to SVM — The Dual Problem

### 3.1 The SVM Lagrangian

We rewrite each constraint $y^{(i)}(w^T x^{(i)} + b) \geq 1$ as $1 - y^{(i)}(w^T x^{(i)} + b) \leq 0$.

The Lagrangian is:

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 + \sum_{i=1}^{m} \alpha_i \left[ 1 - y^{(i)}(w^T x^{(i)} + b) \right], \qquad \alpha_i \geq 0$$

Expanding:

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^{m} \alpha_i \, y^{(i)}(w^T x^{(i)} + b) + \sum_{i=1}^{m} \alpha_i$$

### 3.2 Deriving the Dual: Minimise over $w$ and $b$

**Step 1: Differentiate w.r.t. $w$ and set to zero (stationarity).**

$$\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)} = 0$$

$$\boxed{w = \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)}}$$

This is a key result: the optimal $w$ is a **linear combination of the training examples**, weighted by $\alpha_i y^{(i)}$. Only support vectors (where $\alpha_i > 0$) contribute.

**Step 2: Differentiate w.r.t. $b$ and set to zero.**

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^{m} \alpha_i y^{(i)} = 0$$

$$\boxed{\sum_{i=1}^{m} \alpha_i y^{(i)} = 0}$$

**Step 3: Substitute back into the Lagrangian.**

First, compute $\|w\|^2$:

$$\|w\|^2 = w^T w = \left(\sum_{i} \alpha_i y^{(i)} x^{(i)}\right)^T \left(\sum_{j} \alpha_j y^{(j)} x^{(j)}\right) = \sum_{i}\sum_{j} \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)}$$

Next, compute $\sum_i \alpha_i y^{(i)} w^T x^{(i)}$:

$$\sum_{i} \alpha_i y^{(i)} w^T x^{(i)} = \sum_{i}\sum_{j} \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(j)})^T x^{(i)} = \sum_{i}\sum_{j} \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)}$$

And the term $\sum_i \alpha_i y^{(i)} b = b \sum_i \alpha_i y^{(i)} = 0$ (from Step 2).

Substituting:

$$\mathcal{L} = \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} - \sum_{i,j} \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} + \sum_i \alpha_i$$

$$= \sum_{i=1}^{m} \alpha_i - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)}$$

### 3.3 The Dual SVM Problem

$$\boxed{\max_{\alpha} \quad \sum_{i=1}^{m} \alpha_i - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)}}$$

$$\text{subject to} \quad \alpha_i \geq 0, \quad i = 1, \ldots, m$$

$$\sum_{i=1}^{m} \alpha_i y^{(i)} = 0$$

This is a **quadratic programming** (QP) problem in the $\alpha_i$ variables.

### 3.4 KKT Conditions for SVM

At the optimal solution:

1. $w = \sum_i \alpha_i y^{(i)} x^{(i)}$ — (stationarity w.r.t. $w$)
2. $\sum_i \alpha_i y^{(i)} = 0$ — (stationarity w.r.t. $b$)
3. $y^{(i)}(w^T x^{(i)} + b) \geq 1$ — (primal feasibility)
4. $\alpha_i \geq 0$ — (dual feasibility)
5. $\alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1] = 0$ — (complementary slackness)

Condition 5 tells us:
- If $\alpha_i > 0$: the point is **on the margin** ($y^{(i)}(w^T x^{(i)} + b) = 1$). It is a support vector.
- If $\alpha_i = 0$: the point is **beyond the margin** and does not influence the solution.

### 3.5 Making Predictions with the Dual

To classify a new point $x$:

$$h(x) = \text{sign}(w^T x + b) = \text{sign}\left(\sum_{i=1}^{m} \alpha_i y^{(i)} (x^{(i)})^T x + b\right)$$

Note that the prediction depends only on **dot products** between training examples and the new point. This observation is the foundation of the kernel trick (Part 5).

### 3.6 Soft-Margin SVM

Real data is rarely perfectly separable. The **soft-margin** SVM introduces **slack variables** $\xi_i \geq 0$ that allow some points to violate the margin:

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{m} \xi_i$$

$$\text{subject to} \quad y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

- $C > 0$ is a hyperparameter that controls the trade-off between maximising the margin and minimising violations.
- Large $C$: few violations allowed (closer to hard margin, risk of overfitting).
- Small $C$: more violations tolerated (wider margin, risk of underfitting).

The dual of the soft-margin SVM is the same as before, except $\alpha_i$ is bounded:

$$0 \leq \alpha_i \leq C$$

---

## Part 4: The Hinge Loss Formulation

### 4.1 An Equivalent View

The soft-margin SVM can be reformulated as an **unconstrained** optimisation problem using the **hinge loss**.

Recall that $\xi_i$ measures how much the $i$-th point violates the margin. We can write:

$$\xi_i = \max(0, \; 1 - y^{(i)}(w^T x^{(i)} + b))$$

This is 0 when the point is correctly classified with margin $\geq 1$, and positive otherwise.

Substituting into the soft-margin objective:

$$\boxed{J(w, b) = \frac{\lambda}{2}\|w\|^2 + \frac{1}{m}\sum_{i=1}^{m} \max\left(0, \; 1 - y^{(i)}(w^T x^{(i)} + b)\right)}$$

where $\lambda = \frac{1}{mC}$ (reparametrisation). The first term is **regularization** and the second is the **hinge loss**.

### 4.2 The Hinge Loss Function

The **hinge loss** for a single example is:

$$\ell_{\text{hinge}}(z) = \max(0, 1 - z) \qquad \text{where } z = y^{(i)}(w^T x^{(i)} + b)$$

- If $z \geq 1$ (correct classification with margin $\geq 1$): loss = 0.
- If $0 < z < 1$ (correct classification but within the margin): loss = $1 - z > 0$.
- If $z \leq 0$ (misclassification): loss = $1 - z \geq 1$.

### 4.3 Gradient of the Hinge Loss

The hinge loss is piecewise linear, so its gradient is:

$$\frac{\partial}{\partial w_j} \max(0, 1 - y^{(i)}(w^T x^{(i)} + b)) = \begin{cases} 0 & \text{if } y^{(i)}(w^T x^{(i)} + b) \geq 1 \\ -y^{(i)} x_j^{(i)} & \text{if } y^{(i)}(w^T x^{(i)} + b) < 1\end{cases}$$

### 4.4 Gradient Descent for SVM

The full gradient of the objective $J(w, b)$ with respect to $w_j$ is:

$$\frac{\partial J}{\partial w_j} = \lambda \, w_j + \frac{1}{m}\sum_{i=1}^{m} \begin{cases} 0 & \text{if } y^{(i)}(w^T x^{(i)} + b) \geq 1 \\ -y^{(i)} x_j^{(i)} & \text{otherwise}\end{cases}$$

The **gradient descent update** is:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

Explicitly, for each example $i$:

- **If the point is correctly classified with sufficient margin** ($y^{(i)}(w^T x^{(i)} + b) \geq 1$):

$$w := w - \alpha \lambda w$$

Only the regularization term contributes (shrink the weights).

- **If the point is on the wrong side or within the margin** ($y^{(i)}(w^T x^{(i)} + b) < 1$):

$$w := w - \alpha (\lambda w - y^{(i)} x^{(i)})$$

Both the regularization and the hinge loss contribute (shrink the weights AND push the boundary towards the correct side).

### 4.5 Sub-gradient Descent

Technically, the hinge loss is not differentiable at $z = 1$ (it has a "kink"). The gradient computed above is actually a **sub-gradient**. Sub-gradient descent works similarly to standard gradient descent but converges more slowly ($O(1/\sqrt{t})$ vs. $O(1/t)$). In practice, this is rarely a problem.

---

## Part 5: Kernels

### 5.1 The Key Observation

Look at the dual SVM problem again:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y^{(i)} y^{(j)} \underbrace{(x^{(i)})^T x^{(j)}}_{\text{dot product}}$$

And the prediction rule:

$$h(x) = \text{sign}\left(\sum_i \alpha_i y^{(i)} \underbrace{(x^{(i)})^T x}_{\text{dot product}} + b\right)$$

Both depend on the data **only through dot products** $\langle x^{(i)}, x^{(j)} \rangle$.

### 5.2 The Kernel Trick

**Idea:** Replace the dot product with a more general function $K(x^{(i)}, x^{(j)})$ called a **kernel**:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle$$

where $\phi: \mathbb{R}^n \to \mathbb{R}^N$ is a mapping into a (possibly much) higher-dimensional space.

The kernel trick lets us compute dot products in the high-dimensional space **without ever computing $\phi(x)$ explicitly**.

### 5.3 Why is this Useful?

Data that is not linearly separable in the original space may become separable in a higher-dimensional feature space:

The SVM in the feature space finds a linear boundary in $\mathbb{R}^N$, which corresponds to a **non-linear** boundary in the original $\mathbb{R}^n$.

### 5.4 Example: Polynomial Kernel

Consider data in $\mathbb{R}^2$ with features $x = (x_1, x_2)$.

**The mapping $\phi$ for degree 2:**

$$\phi(x) = (x_1^2, \; x_2^2, \; \sqrt{2}\, x_1 x_2, \; \sqrt{2}\, x_1, \; \sqrt{2}\, x_2, \; 1)$$

This maps from $\mathbb{R}^2$ to $\mathbb{R}^6$.

**Direct computation of the dot product in feature space:**

$$\langle \phi(x), \phi(x') \rangle = x_1^2 {x_1'}^2 + x_2^2 {x_2'}^2 + 2x_1 x_2 x_1' x_2' + 2x_1 x_1' + 2x_2 x_2' + 1$$

**Using the kernel shortcut:**

$$(x^T x' + 1)^2 = (x_1 x_1' + x_2 x_2' + 1)^2$$

Expanding:

$$= x_1^2 {x_1'}^2 + x_2^2 {x_2'}^2 + 1 + 2x_1 x_1' x_2 x_2' + 2x_1 x_1' + 2x_2 x_2'$$

**These are identical!** So:

$$K(x, x') = (x^T x' + 1)^2 = \langle \phi(x), \phi(x') \rangle$$

We can compute the 6-dimensional dot product using a **single scalar operation** in the original 2D space. No need to explicitly construct $\phi(x)$.

**The general polynomial kernel of degree $d$:**

$$\boxed{K(x, x') = (x^T x' + c)^d}$$

where $c \geq 0$ is a constant. This implicitly works in a feature space of dimension $\binom{n+d}{d}$, which grows combinatorially — but the kernel computation remains $O(n)$.

### 5.5 Common Kernels

| Kernel | Formula | Feature space |
|--------|---------|---------------|
| **Linear** | $K(x,x') = x^T x'$ | Original space (standard SVM) |
| **Polynomial** | $K(x,x') = (x^T x' + c)^d$ | All monomials up to degree $d$ |
| **RBF (Gaussian)** | $K(x,x') = \exp\left(-\gamma \|x-x'\|^2\right)$ | Infinite-dimensional! |
| **Sigmoid** | $K(x,x') = \tanh(\kappa \, x^T x' + c)$ | Related to neural networks |

### 5.6 The RBF Kernel

The **Radial Basis Function** (Gaussian) kernel is the most popular:

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right), \qquad \gamma > 0$$

$\gamma$ controls how "local" the kernel is:
  - **Large $\gamma$:** Only very close points have high similarity. The decision boundary can be very complex (risk of overfitting).
  - **Small $\gamma$:** Points far apart still have non-negligible similarity. The decision boundary is smoother (risk of underfitting).

The RBF kernel corresponds to an **infinite-dimensional** feature space. Yet we never need to compute $\phi(x)$ — we only need the kernel value, which is a simple exponential.

### 5.7 The Kernelised SVM

The dual problem with a kernel becomes:

$$\max_\alpha \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)}, x^{(j)})$$

And prediction:

$$h(x) = \text{sign}\left(\sum_{i=1}^m \alpha_i y^{(i)} K(x^{(i)}, x) + b\right)$$

No modification to the algorithm is needed — just replace every dot product with a kernel evaluation.