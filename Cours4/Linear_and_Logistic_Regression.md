# Gradient Derivations for Linear and Logistic Regression

## Notation

| Symbol | Meaning |
|--------|---------|
| $m$ | Number of training examples |
| $n$ | Number of features |
| $X \in \mathbb{R}^{m \times (n+1)}$ | Feature matrix (with a column of ones for the bias) |
| $x^{(i)} \in \mathbb{R}^{n+1}$ | The $i$-th training example (row of $X$) |
| $x_j^{(i)}$ | The $j$-th feature of the $i$-th example |
| $\theta \in \mathbb{R}^{n+1}$ | Parameter vector (weights + bias) |
| $y^{(i)}$ | The target value for the $i$-th example |
| $h_\theta(x)$ | The hypothesis (prediction) function |

---

## Part 1: Linear Regression

### 1.1 The Model

The hypothesis function for linear regression is:

$$h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

### 1.2 The Cost Function

We use the **Mean Squared Error** (MSE) cost function:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

The factor $\frac{1}{2}$ is a convenience that simplifies the derivative (the 2 from the power rule cancels with it).

### 1.3 Computing the Gradient

We want to compute $\frac{\partial J}{\partial \theta_j}$ for each parameter $\theta_j$.

**Step 1: Write out the cost function explicitly.**

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

**Step 2: Apply the chain rule.**

> **The Chain Rule.** If $f(g(x))$ is a composition of functions, then:
> $$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$
> In words: *the derivative of the outer function evaluated at the inner function, times the derivative of the inner function.*

Let us define the **error** for example $i$:

$$e^{(i)} = h_\theta(x^{(i)}) - y^{(i)}$$

Then $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(e^{(i)}\right)^2$.

We identify the composition:
- **Outer function:** $f(u) = \frac{1}{2m} u^2$, so $f'(u) = \frac{1}{m} u$
- **Inner function:** $u = e^{(i)} = h_\theta(x^{(i)}) - y^{(i)} = \theta^T x^{(i)} - y^{(i)}$

**Step 3: Differentiate with respect to $\theta_j$.**

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{2m} \sum_{i=1}^{m} \frac{\partial}{\partial \theta_j} \left( e^{(i)} \right)^2$$

Applying the chain rule to each term:

$$\frac{\partial}{\partial \theta_j} \left( e^{(i)} \right)^2 = 2 \, e^{(i)} \cdot \frac{\partial \, e^{(i)}}{\partial \theta_j}$$

**Step 4: Compute the derivative of the inner function.**

$$\frac{\partial \, e^{(i)}}{\partial \theta_j} = \frac{\partial}{\partial \theta_j} \left( \theta^T x^{(i)} - y^{(i)} \right) = \frac{\partial}{\partial \theta_j} \left( \theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \cdots + \theta_j x_j^{(i)} + \cdots + \theta_n x_n^{(i)} - y^{(i)} \right)$$

Since only the term $\theta_j x_j^{(i)}$ depends on $\theta_j$:

$$\frac{\partial \, e^{(i)}}{\partial \theta_j} = x_j^{(i)}$$

**Step 5: Combine everything.**

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{2m} \sum_{i=1}^{m} 2 \, e^{(i)} \cdot x_j^{(i)} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

### 1.4 Result: Gradient for Linear Regression

$$\boxed{\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}}$$

In **vectorized form**, the full gradient vector is:

$$\nabla_\theta J = \frac{1}{m} X^T (X\theta - y)$$

---

## Part 2: Logistic Regression

### 2.1 The Model

For binary classification ($y \in \{0, 1\}$), we use the **sigmoid** (logistic) function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The hypothesis function is:

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

This outputs a value in $(0, 1)$, interpreted as the probability $P(y = 1 \mid x; \theta)$.

### 2.2 Key Property of the Sigmoid

The sigmoid has a very convenient derivative. Let us derive it.

$$\sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$$

Using the chain rule:

$$\sigma'(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

We can rewrite this by noticing that:

$$\frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = \sigma(z) \cdot (1 - \sigma(z))$$

Therefore:

$$\boxed{\sigma'(z) = \sigma(z)(1 - \sigma(z))}$$

### 2.3 The Cost Function

We cannot use MSE for logistic regression because the resulting cost function would be **non-convex** (many local minima). Instead, we use the **binary cross-entropy** (log loss):

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

**Intuition:** 
- When $y^{(i)} = 1$: the cost is $-\log(h_\theta(x^{(i)}))$. If the model predicts close to 1, $\log(1) = 0$ (no penalty). If it predicts close to 0, $-\log(0) \to +\infty$ (huge penalty).
- When $y^{(i)} = 0$: the cost is $-\log(1 - h_\theta(x^{(i)}))$. Symmetric reasoning.

### 2.4 Computing the Gradient

We want $\frac{\partial J}{\partial \theta_j}$. For clarity, we compute the derivative for a single example $i$ and then sum.

Define:

$$\ell^{(i)}(\theta) = y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))$$

so that $J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \ell^{(i)}(\theta)$.

**Step 1: Identify the chain of compositions.**

We have three nested functions:
1. The **outermost** functions: $\log(\cdot)$ applied to $h$ and $1 - h$
2. The **middle** function: $h = \sigma(z)$ (the sigmoid)
3. The **innermost** function: $z = \theta^T x^{(i)}$ (the linear combination)

So by the chain rule:

$$\frac{\partial \ell^{(i)}}{\partial \theta_j} = \frac{\partial \ell^{(i)}}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial \theta_j}$$

where $h = h_\theta(x^{(i)})$ and $z = \theta^T x^{(i)}$.

**Step 2: Compute $\frac{\partial \ell^{(i)}}{\partial h}$ — derivative of the loss w.r.t. the prediction.**

$$\ell^{(i)} = y^{(i)} \log(h) + (1 - y^{(i)}) \log(1 - h)$$

$$\frac{\partial \ell^{(i)}}{\partial h} = \frac{y^{(i)}}{h} + (1 - y^{(i)}) \cdot \frac{-1}{1 - h} = \frac{y^{(i)}}{h} - \frac{1 - y^{(i)}}{1 - h}$$

**Step 3: Compute $\frac{\partial h}{\partial z}$ — derivative of the sigmoid.**

From Section 2.2:

$$\frac{\partial h}{\partial z} = \sigma'(z) = h(1 - h)$$

where we used $h = \sigma(z)$.

**Step 4: Compute $\frac{\partial z}{\partial \theta_j}$ — derivative of the linear combination.**

$$z = \theta^T x^{(i)} = \theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \cdots + \theta_n x_n^{(i)}$$

$$\frac{\partial z}{\partial \theta_j} = x_j^{(i)}$$

**Step 5: Multiply the three terms together (chain rule).**

$$\frac{\partial \ell^{(i)}}{\partial \theta_j} = \left( \frac{y^{(i)}}{h} - \frac{1 - y^{(i)}}{1 - h} \right) \cdot h(1-h) \cdot x_j^{(i)}$$

**Step 6: Simplify.**

Distribute $h(1-h)$:

$$\frac{\partial \ell^{(i)}}{\partial \theta_j} = \left( \frac{y^{(i)}}{h} \cdot h(1-h) - \frac{1 - y^{(i)}}{1-h} \cdot h(1-h) \right) \cdot x_j^{(i)}$$

$$= \left( y^{(i)}(1-h) - (1 - y^{(i)})h \right) \cdot x_j^{(i)}$$

$$= \left( y^{(i)} - y^{(i)}h - h + y^{(i)}h \right) \cdot x_j^{(i)}$$

$$= \left( y^{(i)} - h \right) \cdot x_j^{(i)}$$

Therefore:

$$\frac{\partial \ell^{(i)}}{\partial \theta_j} = \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

**Step 7: Compute the full gradient.**

Since $J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \ell^{(i)}(\theta)$:

$$\frac{\partial J}{\partial \theta_j} = -\frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

### 2.5 Result: Gradient for Logistic Regression

$$\boxed{\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}}$$

In **vectorized form**:

$$\nabla_\theta J = \frac{1}{m} X^T (\sigma(X\theta) - y)$$

### 2.6 A Remarkable Coincidence

The gradient formula for logistic regression has **exactly the same form** as for linear regression! The only difference is what $h_\theta(x)$ means:

| | Linear Regression | Logistic Regression |
|---|---|---|
| $h_\theta(x)$ | $\theta^T x$ | $\sigma(\theta^T x)$ |
| Gradient formula | $\frac{1}{m} \sum (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$ | $\frac{1}{m} \sum (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$ |

This is not a coincidence — it is a consequence of choosing the cross-entropy loss, which is the natural loss for the sigmoid function. Together they form what is called an **exponential family** pairing.

---

## Part 3: Gradient Descent

### 3.1 The Algorithm

Gradient descent is an iterative optimization algorithm. The idea is simple: move the parameters in the direction opposite to the gradient (the direction of steepest descent).

**Repeat until convergence:**

$$\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j} \qquad \text{for all } j = 0, 1, \ldots, n$$

where $\alpha > 0$ is the **learning rate** — a hyperparameter that controls the step size.

In vectorized form:

$$\theta := \theta - \alpha \, \nabla_\theta J$$

> **Important:** All parameters $\theta_0, \theta_1, \ldots, \theta_n$ must be updated **simultaneously**. That is, compute all the partial derivatives using the current values of $\theta$, and only then update all parameters at once.

### 3.2 Gradient Descent for Linear Regression

Plugging in the gradient from Section 1.4:

$$\theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

where $h_\theta(x^{(i)}) = \theta^T x^{(i)}$.

In vectorized form:

$$\theta := \theta - \frac{\alpha}{m} X^T(X\theta - y)$$

**Properties:**
- The MSE cost function for linear regression is **convex** (bowl-shaped), so gradient descent is guaranteed to converge to the global minimum (given a small enough learning rate).
- There also exists a closed-form solution: $\theta = (X^T X)^{-1} X^T y$ (the **normal equation**), but gradient descent scales better to large datasets.

### 3.3 Gradient Descent for Logistic Regression

Plugging in the gradient from Section 2.5:

$$\theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

where $h_\theta(x^{(i)}) = \sigma(\theta^T x^{(i)})$.

In vectorized form:

$$\theta := \theta - \frac{\alpha}{m} X^T(\sigma(X\theta) - y)$$

**Properties:**
- The cross-entropy cost function for logistic regression is also **convex**, so gradient descent converges to the global minimum.
- There is **no closed-form solution** for logistic regression, so iterative methods like gradient descent are necessary.

### 3.4 Choosing the Learning Rate $\alpha$

- **Too large:** The algorithm may overshoot the minimum and diverge (the cost increases).
- **Too small:** The algorithm converges very slowly (many iterations needed).
- **In practice:** Start with a value like $\alpha = 0.01$ and adjust. Plot $J(\theta)$ vs. iteration number — it should decrease steadily.

### 3.5 Summary

| | Linear Regression | Logistic Regression |
|---|---|---|
| **Hypothesis** | $h_\theta(x) = \theta^T x$ | $h_\theta(x) = \sigma(\theta^T x)$ |
| **Cost function** | $\frac{1}{2m} \sum (h_\theta(x^{(i)}) - y^{(i)})^2$ | $-\frac{1}{m} \sum [y^{(i)} \log h + (1-y^{(i)}) \log(1-h)]$ |
| **Gradient** | $\frac{1}{m} \sum (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$ | $\frac{1}{m} \sum (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$ |
| **Update rule** | $\theta := \theta - \alpha \nabla_\theta J$ | $\theta := \theta - \alpha \nabla_\theta J$ |
| **Convex?** | Yes | Yes |
| **Closed-form?** | Yes (normal equation) | No |
