# Neural Networks

## Notation

| Symbol | Meaning |
|--------|---------|
| $m$ | Number of training examples |
| $n$ | Number of input features |
| $L$ | Number of layers (including input and output) |
| $n^{[l]}$ | Number of units in layer $l$ |
| $W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$ | Weight matrix of layer $l$ |
| $b^{[l]} \in \mathbb{R}^{n^{[l]}}$ | Bias vector of layer $l$ |
| $z^{[l]} \in \mathbb{R}^{n^{[l]}}$ | Pre-activation vector of layer $l$ |
| $a^{[l]} \in \mathbb{R}^{n^{[l]}}$ | Activation (output) vector of layer $l$ |
| $\sigma$ | An activation function |
| $\hat{y}$ | The model's prediction |
| $J(\theta)$ | Loss function |
| $\alpha$ | Learning rate |

We use **superscripts in square brackets** $[\cdot]$ for layer indices and **superscripts in round brackets** $(\cdot)$ for training example indices, e.g. $a^{[l](i)}$ is the activation at layer $l$ for example $i$.

---

## Part 1: From Linear Models to Neural Networks

### 1.1 The Limitation of Linear Models

A logistic regression model computes:

$$\hat{y} = \sigma(W x + b)$$

where $\sigma$ is the sigmoid function and $W x + b$ is a linear combination of the input features. The **decision boundary** in input space is always a hyperplane. Linear models cannot learn non-linear patterns, such as:

- XOR: the output is 1 iff exactly one input is 1.
- Concentric classes (e.g., inner circle vs. outer ring).
- Complex image and text structure.

### 1.2 Adding Depth and Width

A **neural network** overcomes this by composing multiple linear transformations interleaved with non-linear **activation functions**:

$$a^{[1]} = \sigma(W^{[1]} x + b^{[1]})$$
$$a^{[2]} = \sigma(W^{[2]} a^{[1]} + b^{[2]})$$
$$\vdots$$
$$\hat{y} = f(W^{[L]} a^{[L-1]} + b^{[L]})$$

where $f$ is the output activation (e.g., softmax for classification, identity for regression).

> **Universal Approximation Theorem (informal).** A feedforward neural network with at least one hidden layer and a non-linear activation function can approximate any continuous function on a compact domain to arbitrary precision, given enough hidden units.

This theorem guarantees expressiveness, but says nothing about how to *find* the right weights efficiently — that is the learning problem.

---

## Part 2: The Building Blocks

### 2.1 The Artificial Neuron

A single neuron computes:

$$z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = w^\top x + b$$
$$a = \sigma(z)$$

where:
- $w \in \mathbb{R}^n$ are the **weights** (learned parameters),
- $b \in \mathbb{R}$ is the **bias** (learned parameter),
- $\sigma$ is the **activation function**.

The output $a$ is then passed as an input to neurons in the next layer.

### 2.2 Activation Functions

| Name | Formula | Range | Notes |
|------|---------|-------|-------|
| **Sigmoid** | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | $(0, 1)$ | Saturates; vanishing gradient at extremes |
| **Tanh** | $\tanh(z) = \dfrac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Zero-centred; still saturates |
| **ReLU** | $\max(0, z)$ | $[0, +\infty)$ | Default choice; fast; dead neurons |
| **Leaky ReLU** | $\max(\alpha z, z)$, $\alpha \approx 0.01$ | $\mathbb{R}$ | Fixes dead neuron issue |
| **GELU** | $z \cdot \Phi(z)$ | $\mathbb{R}$ | Smooth; used in transformers. $\Phi(z)$ is the standard normal CDF |
| **Softmax** | $\text{softmax}(z)_k = \dfrac{e^{z_k}}{\sum_j e^{z_j}}$ | $(0,1)^K$, sums to 1 | Output layer of multi-class classifier |

**Why ReLU became the default:**
- No saturation for positive values → no vanishing gradient there.
- Sparse activations → computationally efficient.
- Simple gradient: $\sigma'(z) = \mathbf{1}[z > 0]$.

**Dead neuron problem.** If $z < 0$ for all training examples, the ReLU gradient is 0 and the neuron never updates. Leaky ReLU avoids this.

### 2.3 Feedforward Pass (Forward Propagation)

For a network with $L$ layers and a single training example $x$:

$$a^{[0]} = x \quad \text{(input layer)}$$

For $l = 1, \ldots, L$:
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = \sigma^{[l]}(z^{[l]})$$

The output $\hat{y} = a^{[L]}$ is used to compute the loss.

In **matrix form** over all $m$ training examples simultaneously:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = \sigma^{[l]}(Z^{[l]})$$

where $A^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$ stacks all $m$ activation vectors as columns.

---

## Part 3: Loss Functions

### 3.1 Regression — Mean Squared Error

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

The output activation is the **identity** ($\hat{y} = z^{[L]}$).

### 3.2 Binary Classification — Binary Cross-Entropy

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

The output activation is **sigmoid** so that $\hat{y}^{(i)} \in (0, 1)$ represents the predicted probability of class 1.

### 3.3 Multi-Class Classification — Cross-Entropy

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log \hat{y}_k^{(i)}$$

where $y^{(i)}$ is a **one-hot vector** and $\hat{y}^{(i)} = \text{softmax}(z^{[L](i)})$.

> **Note on PyTorch.** `nn.CrossEntropyLoss` combines `nn.LogSoftmax` and `nn.NLLLoss`. You should **not** apply softmax yourself before passing logits to this loss.

---

## Part 4: Backpropagation

### 4.1 The Core Idea

Training minimises $J(\theta)$ by gradient descent. This requires the gradient $\frac{\partial J}{\partial W^{[l]}}$ and $\frac{\partial J}{\partial b^{[l]}}$ for every layer $l$. **Backpropagation** computes these gradients efficiently using the chain rule, propagating errors from the output layer back to the input.

### 4.2 The Chain Rule

For a composition $J = f(g(h(x)))$:

$$\frac{dJ}{dx} = \frac{dJ}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

Backpropagation is simply the organised, layer-by-layer application of this rule.

### 4.3 The Backpropagation Equations

Define the **error signal** at layer $l$:

$$\delta^{[l]} = \frac{\partial J}{\partial z^{[l]}}$$

**Output layer** ($l = L$):

$$\delta^{[L]} = \frac{\partial J}{\partial a^{[L]}} \odot \sigma'^{[L]}(z^{[L]})$$

where $\odot$ denotes element-wise multiplication. For MSE with identity output activation, $\delta^{[L]} = \frac{2}{m}(a^{[L]} - y)$.

**Hidden layers** (propagate backwards for $l = L-1, \ldots, 1$):

$$\delta^{[l]} = \left(W^{[l+1]\top} \delta^{[l+1]}\right) \odot \sigma'^{[l]}(z^{[l]})$$

**Gradients**:

$$\frac{\partial J}{\partial W^{[l]}} = \delta^{[l]} {a^{[l-1]}}^\top, \qquad \frac{\partial J}{\partial b^{[l]}} = \delta^{[l]}$$

### 4.4 The Vanishing Gradient Problem

When the network is deep and activation functions like sigmoid or tanh are used, the gradient signal can **shrink exponentially** as it propagates backwards. Concretely:

- Sigmoid derivative: $\sigma'(z) \leq 0.25$, with maximum only near $z=0$.
- After $L$ layers the gradient is scaled by factors like $\prod_{l} \sigma'^{[l]}$, which can become negligibly small.

**Remedies:**
- Use **ReLU** (gradient is exactly 1 for $z > 0$).
- Use **batch normalisation** (see Part 6).
- Use **residual connections** (skip connections, as in ResNets).

### 4.5 Automatic Differentiation

In modern frameworks (PyTorch, JAX), backpropagation is implemented automatically via **computational graphs**:

1. The forward pass records operations in a graph.
2. Calling `.backward()` traverses the graph in reverse and accumulates gradients.

You never implement backpropagation manually in PyTorch.

---

## Part 5: Optimisation

### 5.1 Gradient Descent

Starting from random weights, update all parameters simultaneously:

$$\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)$$

Computing the full gradient over all $m$ examples is called **batch gradient descent**. It is exact but expensive for large datasets.

### 5.2 Stochastic Gradient Descent (SGD)

Update using the gradient from a single randomly selected example:

$$\theta \leftarrow \theta - \alpha \nabla_\theta J_i(\theta)$$

**Noisy** but very fast per update; can escape shallow local minima.

### 5.3 Mini-Batch Gradient Descent

The standard in practice: update using a **mini-batch** of $B$ examples ($B \in [32, 512]$):

$$\theta \leftarrow \theta - \alpha \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\theta J_i(\theta)$$

One pass over all $m$ training examples is called an **epoch**.

### 5.4 Momentum

Standard SGD can oscillate or be slow in directions with small but consistent gradients. **Momentum** maintains a running average of past gradients (the "velocity" $v$):

$$v \leftarrow \beta v - \alpha \nabla_\theta J(\theta), \qquad \theta \leftarrow \theta + v$$

Typical value: $\beta = 0.9$. Momentum accelerates learning in consistent directions and dampens oscillations.

### 5.5 Adam

**Adam** (Adaptive Moment Estimation) is the most popular optimizer. It keeps per-parameter running estimates of:
- the **first moment** $m_t$ (mean of gradients, like momentum),
- the **second moment** $v_t$ (uncentred variance of gradients, like RMSProp).

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias-corrected estimates: $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$.

Update rule:

$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Typical hyperparameters: $\alpha = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

| Optimizer | Adaptive LR | Momentum | Notes |
|-----------|------------|---------|-------|
| SGD | No | No | Simple; needs careful tuning |
| SGD + Momentum | No | Yes | Faster convergence |
| RMSProp | Yes | No | Good for RNNs |
| **Adam** | **Yes** | **Yes** | **Default choice for most tasks** |

---

## Part 6: Regularisation

Overfitting in neural networks is addressed by several techniques.

### 6.1 $L_2$ Regularisation (Weight Decay)

Add a penalty term to the loss:

$$J_{\text{reg}}(\theta) = J(\theta) + \frac{\lambda}{2m} \|\theta\|_2^2$$

This penalises large weights and pushes them towards zero. Equivalent to **weight decay** in the gradient update:

$$W \leftarrow W - \alpha \left( \frac{\partial J}{\partial W} + \frac{\lambda}{m} W \right) = (1 - \alpha \lambda / m) W - \alpha \frac{\partial J}{\partial W}$$

In PyTorch: pass `weight_decay=λ` to the optimizer.

### 6.2 Dropout

During training, each neuron is independently **zeroed out** with probability $p$ (typically $p \in [0.1, 0.5]$). To preserve expected activation magnitudes, the surviving activations are scaled by $1/(1-p)$.

- At **test time**, dropout is disabled and all neurons are used.
- Forces the network to learn **redundant representations**.
- Acts like training an implicit ensemble of $2^n$ sub-networks.

```python
nn.Dropout(p=0.5)   # place after the activation, before the next layer
```

### 6.3 Batch Normalisation

Applied to the pre-activation $z^{[l]}$ in each mini-batch of size $B$:

$$\hat{z}_i = \frac{z_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \qquad \tilde{z}_i = \gamma \hat{z}_i + \beta$$

where $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ are the batch mean and variance, and $\gamma$, $\beta$ are learned affine parameters.

**Benefits:**
- Reduces internal covariate shift, enabling higher learning rates.
- Acts as a mild regulariser.
- Mitigates the vanishing gradient problem.

At test time, the running statistics accumulated during training are used.

---

## Part 7: Practical PyTorch

### 7.1 Core Abstractions

| Concept | PyTorch class / function | Role |
|---------|--------------------------|------|
| Tensor | `torch.Tensor` | N-dimensional array with gradient support |
| Model | `nn.Module` | Container for layers and parameters |
| Sequential | `nn.Sequential` | Ordered stack of layers |
| Loss | `nn.MSELoss`, `nn.CrossEntropyLoss`, … | Scalar objective to minimise |
| Optimizer | `torch.optim.SGD`, `.Adam`, … | Updates parameters given gradients |

### 7.2 Defining a Model

```python
import torch.nn as nn

# Option 1 – Sequential (compact, layers applied in order)
model = nn.Sequential(
    nn.Linear(n_features, 128), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),         nn.ReLU(),
    nn.Linear(64, n_classes)
)

# Option 2 – nn.Module subclass (more flexible)
class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### 7.3 The Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()                       # enable dropout / batch norm
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()           # clear stale gradients
        logits = model(X_batch)         # forward pass
        loss = loss_fn(logits, y_batch) # compute loss
        loss.backward()                 # backpropagation
        optimizer.step()                # update weights
```

### 7.4 Evaluation

```python
model.eval()                            # disable dropout / batch norm
with torch.no_grad():                   # do not build computation graph
    logits = model(X_test)
    preds  = logits.argmax(dim=1)
    acc    = (preds == y_test).float().mean()
```

> **`model.train()` vs `model.eval()`** controls behaviour of layers like Dropout and BatchNorm. Always switch to `eval()` mode when evaluating or making predictions, and back to `train()` when resuming training.
