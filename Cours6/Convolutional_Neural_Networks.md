# Convolutional Neural Networks

## Notation

| Symbol | Meaning |
|--------|---------|
| $H, W$ | Height and width of an image (or feature map) |
| $C$ | Number of channels (depth) |
| $K$ | Spatial size of a convolutional filter (assumed square: $K \times K$) |
| $F$ | Number of filters in a convolutional layer |
| $s$ | Stride |
| $p$ | Padding (number of zeros added on each side) |
| $H', W'$ | Height and width of the output feature map |
| $x \in \mathbb{R}^{H \times W \times C}$ | Input volume |
| $w^{(f)} \in \mathbb{R}^{K \times K \times C}$ | Weights of filter $f$ |
| $b^{(f)} \in \mathbb{R}$ | Bias of filter $f$ |
| $z^{(f)} \in \mathbb{R}^{H' \times W'}$ | Pre-activation feature map for filter $f$ |
| $a^{(f)} \in \mathbb{R}^{H' \times W'}$ | Activation feature map for filter $f$ |

---

## Part 1: Motivation — Why Not Just Use an MLP?

### 1.1 The Problem with Fully Connected Networks on Images

A modest image of size $224 \times 224 \times 3$ has $224 \times 224 \times 3 = 150{,}528$ input values. If the first hidden layer has 1,000 neurons, the weight matrix alone has $150{,}528 \times 1000 \approx 150$ million parameters. This leads to:

- **Excessive memory and compute cost.**
- **Severe overfitting**, because the model has far more parameters than training examples.
- **No use of spatial structure**: nearby pixels are related, but a fully connected layer treats all inputs as independent.

### 1.2 The Key Inductive Biases of CNNs

Convolutional Neural Networks (CNNs) avoid these issues by exploiting two structural properties of images:

| Inductive bias | Meaning | Mechanism |
|----------------|---------|-----------|
| **Translation equivariance** | A pattern looks the same wherever it appears in the image | Shared weights (the same filter is applied everywhere) |
| **Locality** | Relevant patterns are spatially local | Small filter size $K \ll H, W$ |

These two biases lead to **parameter sharing** and **sparse connectivity**, drastically reducing the number of parameters.

---

## Part 2: The Convolutional Layer

### 2.1 The 2D Convolution Operation

A filter $w^{(f)} \in \mathbb{R}^{K \times K \times C}$ is slid across the input volume $x$. At each spatial position $(i, j)$, it computes a dot product:

$$z^{(f)}_{i,j} = \sum_{k_1=0}^{K-1} \sum_{k_2=0}^{K-1} \sum_{c=1}^{C} w^{(f)}_{k_1, k_2, c} \cdot x_{i \cdot s + k_1,\; j \cdot s + k_2,\; c} + b^{(f)}$$

where $s$ is the stride (step size between positions) and $p$ is the number of zero-padding rows/columns added around the input.

The output feature map for filter $f$ is $a^{(f)} = \text{ReLU}(z^{(f)})$.

Stacking all $F$ filters: the output volume is $\mathbb{R}^{H' \times W' \times F}$.

### 2.2 Output Size Formula

$$H' = \left\lfloor \frac{H - K + 2p}{s} \right\rfloor + 1, \qquad W' = \left\lfloor \frac{W - K + 2p}{s} \right\rfloor + 1$$

**Common configurations:**

| Name | Padding $p$ | Stride $s$ | Effect on spatial size |
|------|------------|------------|------------------------|
| Valid convolution | $0$ | $1$ | Shrinks: $H' = H - K + 1$ |
| Same convolution | $\lfloor K/2 \rfloor$ | $1$ | Preserves: $H' = H$ |
| Strided convolution | any | $s > 1$ | Downsamples |

### 2.3 Parameter Count

Each filter has $K \times K \times C$ weights plus $1$ bias. With $F$ filters:

$$\text{parameters} = F \times (K^2 C + 1)$$

This is **independent of the spatial dimensions** $H$ and $W$ — a huge saving over fully connected layers.

### 2.4 What Does a Filter Learn?

Each filter learns to detect a specific local pattern:
- **Early layers:** edges, corners, colour gradients.
- **Middle layers:** textures, simple shapes.
- **Deep layers:** object parts, semantic features.

This hierarchical feature learning is what makes CNNs so effective for vision.

---

## Part 3: Pooling Layers

Pooling reduces the spatial dimensions of feature maps, providing:
- **Computational efficiency** (smaller tensors for subsequent layers).
- **Translation invariance** (small shifts in the input have no effect on the output).

### 3.1 Max Pooling

A $K \times K$ window is slid over each feature map with stride $s$. At each position, the **maximum** value is kept:

$$a^{\text{pool}}_{i,j} = \max_{0 \leq k_1, k_2 < K} a_{i \cdot s + k_1,\; j \cdot s + k_2}$$

Most common: $K=2$, $s=2$, which halves both spatial dimensions.

### 3.2 Average Pooling

Takes the **mean** instead of the maximum. Used less often for intermediate layers, but common as a **global average pooling** (GAP) layer at the end of a network:

$$a^{\text{GAP}}_f = \frac{1}{H' W'} \sum_{i,j} a^{(f)}_{i,j}$$

GAP collapses each feature map to a single value, replacing the flatten + large fully connected layer combination.

### 3.3 Pooling Has No Learnable Parameters

Unlike convolutional layers, pooling layers have no weights — they are fixed operations.

---

## Part 4: The Full CNN Architecture

A typical image classification CNN alternates convolutional blocks with pooling, then ends with fully connected layers:

```
Input (H × W × C)
    ↓
[Conv → BN → ReLU] × n₁   ← feature extraction block 1
    ↓ MaxPool
[Conv → BN → ReLU] × n₂   ← feature extraction block 2
    ↓ MaxPool
...
    ↓ Global Average Pooling (or Flatten)
[Linear → ReLU → Dropout] × k   ← classifier head
    ↓
Linear → Softmax            ← output
```

### 4.1 Typical Design Choices

| Aspect | Common practice |
|--------|----------------|
| Filter size | $3 \times 3$ (small filters stacked are cheaper and more expressive than large ones) |
| Activation | ReLU (sometimes GELU in modern architectures) |
| Batch normalisation | After each convolution, before activation |
| Pooling | Max pool $2 \times 2$, stride 2 after each block |
| Depth vs width | Increase the number of filters as spatial size decreases |
| Regularisation | Dropout before the final linear layer; weight decay |

### 4.2 Spatial Resolution vs Channel Depth Trade-off

As we go deeper:
- **Spatial dimensions** decrease (pooling, strided convolutions).
- **Number of channels** increases (more abstract, richer features).

A common pattern: $32 \to 64 \to 128 \to 256$ filters, while $H, W$ are halved at each block.

---

## Part 5: Batch Normalisation

### 5.1 The Problem

During training, the distribution of the inputs to each layer changes as the parameters of the previous layers change. This is called **internal covariate shift** and forces us to use small learning rates and careful weight initialisation.

### 5.2 The Solution

**Batch Normalisation (BN)** standardises the activations within each mini-batch. For a feature map value $z$ at a given spatial position and channel, over a mini-batch $\mathcal{B}$ of size $B$:

$$\hat{z} = \frac{z - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \qquad \tilde{z} = \gamma \hat{z} + \beta$$

where $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ are the batch mean and variance, and $\gamma$, $\beta$ are **learned per-channel scale and shift parameters**.

At test time, running statistics (accumulated during training) replace the batch statistics.

### 5.3 Benefits

- Allows higher learning rates.
- Reduces sensitivity to weight initialisation.
- Acts as a mild regulariser (no dropout needed in many architectures).
- Stabilises training and speeds up convergence.

In PyTorch: `nn.BatchNorm2d(num_features)` where `num_features` is the number of channels.

---

## Part 6: CNNs in PyTorch

### 6.1 `nn.Conv2d` Parameters

`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)`

| Parameter | Type | Meaning |
|-----------|------|---------|
| `in_channels` | `int` | Number of channels $C$ in the input tensor, e.g. 3 for RGB, 1 for grayscale |
| `out_channels` | `int` | Number of filters $F$; equals the number of channels in the output feature map |
| `kernel_size` | `int` or `(int, int)` | Spatial size $K$ of each filter. `3` means $3 \times 3$; `(3, 5)` means $3 \times 5$ |
| `stride` | `int` or `(int, int)` | Step size $s$ when sliding the filter. Default `1` (no skipping). `2` halves the spatial size |
| `padding` | `int` or `(int, int)` | Zeros added around the border $p$. Default `0`. Use `padding=kernel_size//2` to preserve spatial size with stride 1 |
| `bias` | `bool` | Whether to add a learnable bias term $b^{(f)}$ per filter. Default `True`. Often set to `False` when followed by `BatchNorm2d` (which has its own shift parameter $\beta$) |

**Output spatial size:**

$$H' = \left\lfloor \frac{H - K + 2p}{s} \right\rfloor + 1$$

(same formula for $W'$)

**Number of learnable parameters:**

$$F \times (K^2 \times C_{\text{in}} + \mathbf{1}_{\text{bias}})$$

**Example:**

```python
nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
# in_channels=3   → expects RGB input
# out_channels=64 → produces 64 feature maps
# kernel_size=3   → 3×3 filters
# padding=1       → same convolution: output H×W equals input H×W
# Parameters: 64 × (9×3 + 1) = 1,792
```

### 6.2 Other Key Layers

```python
import torch.nn as nn

# Batch normalisation (after conv, before activation)
nn.BatchNorm2d(num_features)   # num_features = out_channels of the preceding conv

# Pooling
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d((1, 1))   # global average pooling → output is (batch, C, 1, 1)

# Reshape between conv and linear blocks
nn.Flatten()
```

### 6.3 Input Tensor Convention

PyTorch uses the `(N, C, H, W)` convention:
- $N$: batch size
- $C$: number of channels
- $H$, $W$: spatial dimensions

When loading images or converting NumPy arrays, make sure the shape is `(N, C, H, W)`.

### 6.4 A Small CNN Block

```python
class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)
```

### 6.5 A Simple CNN Classifier

```python
class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),          # (N,  1, H,  W) → (N, 32, H,  W)
            nn.MaxPool2d(2),           # → (N, 32, H/2, W/2)
            ConvBlock(32, 64),         # → (N, 64, H/2, W/2)
            nn.MaxPool2d(2),           # → (N, 64, H/4, W/4)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),        # → (N, 64, 4, 4)
            nn.Flatten(),                        # → (N, 64*4*4)
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

---

## Part 7: Transfer Learning

Training a CNN from scratch requires large datasets and significant compute. **Transfer learning** reuses a network pre-trained on a large dataset (typically ImageNet) for a new task.

Two Strategies

**Feature extraction.** Freeze all convolutional layers; replace and train only the final classification head.

```python
import torchvision.models as models

backbone = models.resnet18(weights="IMAGENET1K_V1")
for param in backbone.parameters():
    param.requires_grad = False            # freeze all weights

# Replace the classifier head
backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
# Only backbone.fc parameters will be updated
optimizer = torch.optim.Adam(backbone.fc.parameters(), lr=1e-3)
```

**Fine-tuning.** Unfreeze some or all layers and train the whole network with a small learning rate.

```python
for param in backbone.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)
```
