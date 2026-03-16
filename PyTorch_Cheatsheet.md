# PyTorch Cheatsheet

---

## 1. Tensors — Creation

A `torch.Tensor` is an N-dimensional array like a NumPy array, but with GPU support and automatic differentiation.

```python
import torch

# From scratch
x = torch.empty(3, 4)          # uninitialized (whatever is in memory)
x = torch.zeros(3, 4)          # filled with zeros
x = torch.ones(3, 4)           # filled with ones
x = torch.full((3, 4), 7.0)   # filled with 7.0
x = torch.randn(3, 4)          # random normal N(0,1)
x = torch.arange(12)           # [0, 1, 2, ..., 11]

# From Python data
x = torch.tensor([1.0, 2.0, 3.0])
m = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# From NumPy (zero-copy — they share memory)
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
x = torch.from_numpy(arr)
```

| Constructor | Description |
|-------------|-------------|
| `torch.empty(shape)` | Uninitialized memory |
| `torch.zeros(shape)` | All zeros |
| `torch.ones(shape)` | All ones |
| `torch.full(shape, val)` | Constant value |
| `torch.randn(shape)` | Normal N(0,1) |
| `torch.arange(n)` | Integer range 0..n-1 |
| `torch.tensor(data)` | From Python list or NumPy array |
| `torch.from_numpy(arr)` | Zero-copy from NumPy |

### Key tensor attributes
```python
x.shape      # equivalent to x.size()
x.dtype      # e.g. torch.float32, torch.int64
x.device     # 'cpu' or 'cuda:0'
x.ndim       # number of dimensions
```

---

## 2. Tensors — Indexing, Slicing, Aggregation

```python
x = torch.randn(6, 5)

# Indexing (same syntax as NumPy)
x[0, 2]           # element at row 0, col 2
x[:, :2]          # all rows, first 2 columns
x[1::2, :]        # every other row starting at row 1

# Extract a Python scalar from a 0-d (scalar) tensor
val = x[0, 0].item()

# Aggregation — over whole tensor
x.sum()
x.mean()
x.std()
x.min()
x.max()

# Aggregation — along a dimension
x.mean(dim=0)    # shape (5,)  — mean of each column
x.mean(dim=1)    # shape (6,)  — mean of each row
x.sum(dim=-1)    # same as dim=1 for 2D tensors
```

**Dimension convention:**
- `dim=0` → collapse rows → result shape = `(cols,)`
- `dim=1` → collapse columns → result shape = `(rows,)`

---

## 3. Tensors — Operations

### Element-wise arithmetic
```python
x + y       # addition
x - y       # subtraction
x * y       # element-wise multiplication (NOT matrix multiply)
x / y       # division
x ** 2      # power
```

### Matrix operations
```python
A @ B           # matrix multiplication (works for 2-D and batched)
torch.mm(A, B)  # matrix × matrix (2-D only)
A.mv(v)         # matrix × vector

# Least-squares solution to Ax = b
sol = torch.linalg.lstsq(A, b).solution
```

### In-place operations
Suffixed with `_` — modify the tensor **in place** instead of returning a new one:
```python
x.fill_(0.0)    # set all elements to 0
x.add_(y)       # x += y
x.mul_(2.0)     # x *= 2
x -= x.mean(0)  # broadcasting in-place subtraction
```

> **Important**: In-place operations on tensors that require gradients can cause autograd errors. Avoid them inside the computational graph.

### Broadcasting
Operations on tensors with compatible shapes expand automatically (NumPy rules):
```python
x = torch.randn(100, 4)
x -= x.mean(dim=0)   # (100,4) - (4,) → each row centred
```

---

## 4. Tensors — Reshaping

```python
a = torch.arange(24)

a.view(4, 6)         # reshape (requires contiguous memory)
a.reshape(4, 6)      # reshape (always works, may copy)
a.view(-1, 6)        # infer first dim automatically → (4, 6)
a.view(2, 3, 4)      # 3-D reshape

a.unsqueeze(0)       # insert dim of size 1 at position 0 → (1, 24)
a.unsqueeze(1)       # → (24, 1)
a.squeeze()          # remove all dims of size 1
a.t()                # transpose 2-D tensor
a.permute(2, 0, 1)   # reorder dims arbitrarily (e.g. for images)
```

| Method | Description |
|--------|-------------|
| `.view(shape)` | Reshape without copying (contiguous required) |
| `.reshape(shape)` | Reshape, copies if needed |
| `.unsqueeze(dim)` | Add dimension of size 1 |
| `.squeeze()` | Remove all size-1 dimensions |
| `.t()` | Transpose |
| `.permute(dims)` | Arbitrary dimension reorder |

Use `-1` to let PyTorch infer one dimension:
```python
x.view(x.shape[0], -1)   # flatten all dims except batch
```

---

## 5. Autograd — Automatic Differentiation

PyTorch builds a **computational graph** during the forward pass and uses it to compute gradients via backpropagation.

```python
# Scalar example: f(x) = x² + 3x, f'(x) = 2x + 3
x = torch.tensor(2.0, requires_grad=True)

f = x**2 + 3*x     # forward pass — builds graph
f.backward()        # backward pass — computes df/dx

print(x.grad)       # tensor(7.)  ← 2×2 + 3 = 7
```

```python
# Vector example: gradient of MSE loss w.r.t. weights
X = torch.randn(50, 3)
y = X @ torch.tensor([1.0, -2.0, 0.5]) + 0.1 * torch.randn(50)

w = torch.randn(3, requires_grad=True)
loss = ((X @ w - y) ** 2).mean()
loss.backward()

print(w.grad)       # gradient ∂loss/∂w
```

### Key rules
- Set `requires_grad=True` on parameters you want to differentiate.
- Call `.backward()` **once** on a scalar loss.
- After calling `.backward()`, gradients accumulate in `.grad` — **zero them before the next step** with `optimizer.zero_grad()`.
- Use `torch.no_grad()` to disable graph construction (evaluation, data preprocessing).

```python
with torch.no_grad():
    predictions = model(X_test)   # no gradient tracking
```

---

## 6. Defining Models

### nn.Sequential — quick linear stacks
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(11, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 2)               # output logits
)
```

### nn.Module — full control
Subclass `nn.Module` when you need skip connections, multiple inputs/outputs, or custom logic:
```python
class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, dropout=0.3):
        super().__init__()
        self.fc1     = nn.Linear(n_in, n_hidden)
        self.fc2     = nn.Linear(n_hidden, n_hidden // 2)
        self.fc3     = nn.Linear(n_hidden // 2, n_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)           # raw logits

model = MLP(n_in=11, n_hidden=64, n_out=2)
```

### Common layers

| Layer | Description |
|-------|-------------|
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.ReLU()` | ReLU activation $\max(0, x)$ |
| `nn.Sigmoid()` | Sigmoid activation |
| `nn.Tanh()` | Tanh activation |
| `nn.Dropout(p)` | Randomly zero out neurons (training only) |
| `nn.BatchNorm1d(n)` | Batch normalization for 1-D features |
| `nn.Flatten()` | Flatten all dims except batch |

### Inspecting a model
```python
print(model)    # layer summary

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## 7. Loss Functions

```python
loss_fn = nn.CrossEntropyLoss()          # multi-class classification
loss_fn = nn.NLLLoss()                   # multi-class, use with LogSoftmax output
loss_fn = nn.BCEWithLogitsLoss()         # binary classification (logits)
loss_fn = nn.MSELoss()                   # regression (mean squared error)
loss_fn = nn.L1Loss()                    # regression (mean absolute error)
```

| Loss | Use case | Expected model output |
|------|-----------|-----------------------|
| `CrossEntropyLoss` | Multi-class classification | Raw logits, shape `(N, C)` |
| `NLLLoss` | Multi-class classification | Log-probabilities (after `LogSoftmax`) |
| `BCEWithLogitsLoss` | Binary classification | Single logit, shape `(N,)` or `(N,1)` |
| `MSELoss` | Regression | Predicted values |
| `L1Loss` | Robust regression | Predicted values |

> **Tip:** Prefer `CrossEntropyLoss` over `NLLLoss + LogSoftmax` — it is numerically more stable.

```python
# Usage: loss(output, target)
logits = model(X_batch)               # shape (N, n_classes)
loss = loss_fn(logits, y_batch)       # y_batch: integers in [0, n_classes-1]
```

---

## 8. Optimizers

```python
import torch.optim as optim

# Adam — adaptive learning rates, good default choice
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# RMSProp
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
```

| Optimizer | When to use |
|-----------|-------------|
| `Adam` | General purpose, good default |
| `SGD` | When you want full control; often better with tuning |
| `RMSprop` | Recurrent networks |

### Learning rate schedulers
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# or
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Call after each epoch:
scheduler.step()
```

---

## 9. The Training Loop

### Full-batch training
```python
loss_fn   = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()                        # enable training mode (activates Dropout etc.)

    optimizer.zero_grad()                # 1. clear accumulated gradients
    logits = model(X_train)             # 2. forward pass
    loss   = loss_fn(logits, y_train)   # 3. compute loss
    loss.backward()                      # 4. backpropagation
    optimizer.step()                     # 5. gradient descent step

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | loss: {loss.item():.4f}")
```

### Mini-batch training with DataLoader
```python
from torch.utils.data import TensorDataset, DataLoader

dataset    = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
```

> **Important:** Always call `optimizer.zero_grad()` **before** `loss.backward()`. Gradients accumulate by default in PyTorch.

---

## 10. Evaluation

```python
model.eval()                          # disable Dropout, BatchNorm uses running stats
with torch.no_grad():                 # disable gradient computation (saves memory)
    logits = model(X_test)            # forward pass only
    preds  = logits.argmax(dim=1)     # predicted class indices

accuracy = (preds == y_test).float().mean().item()
print(f"Accuracy: {accuracy:.3f}")
```

| Mode | Effect |
|------|--------|
| `model.train()` | Enables Dropout, BatchNorm uses batch stats |
| `model.eval()` | Disables Dropout, BatchNorm uses running stats |
| `torch.no_grad()` | Disables autograd (faster, uses less memory) |

---

## 11. Saving and Loading Models

```python
# Save weights only (recommended)
torch.save(model.state_dict(), 'model.pth')

# Reload
model = MyModel()                             # recreate architecture first
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save entire model (less portable)
torch.save(model, 'model_full.pth')
model = torch.load('model_full.pth')
```

---

## 12. Convolutional Neural Networks (CNN)

### Image data format: channels-first
PyTorch's `Conv2d` expects tensors in **(N, C, H, W)** format:
- `N` = batch size
- `C` = number of channels (3 for RGB)
- `H`, `W` = height, width

```python
# Convert from (N, H, W, C) → (N, C, H, W) and normalise
X = torch.tensor(images_nhwc).permute(0, 3, 1, 2)
X = X.to(torch.float32) / 255.0 - 0.5    # normalise to [-0.5, 0.5]
```

### Key CNN layers

```python
nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
# Output spatial size: (H - kernel_size + 1) with default padding=0
# Use padding=kernel_size//2 to keep the same spatial size ("same" padding)

nn.MaxPool2d(kernel_size=2, stride=2)   # halves H and W

nn.Flatten()                            # (N, C, H, W) → (N, C*H*W)
```

### Classic LeNet-style CNN for CIFAR-10
```python
model = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5),     # (N,  3, 32, 32) → (N,  6, 28, 28)
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),           # → (N,  6, 14, 14)
    nn.Conv2d(6, 16, kernel_size=5),     # → (N, 16, 10, 10)
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),           # → (N, 16,  5,  5)
    nn.Flatten(),                        # → (N, 400)
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84),         nn.ReLU(),
    nn.Linear(84, 10),
    nn.LogSoftmax(dim=1)
)
loss_fn = nn.NLLLoss()
```

### Spatial size formula
$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

where $k$ = kernel size, $p$ = padding, $s$ = stride (default $s=1$, $p=0$).

### Counting CNN parameters
- **Conv2d(C_in, C_out, k):** $C_{out} \times (k^2 \times C_{in} + 1)$
  *(weights + one bias per output filter)*
- **Linear(in, out):** $in \times out + out$

---

## 13. Data Augmentation (torchvision)

Random transforms applied **on-the-fly during training** to improve generalisation:

```python
import torchvision
import torchvision.transforms.v2 as transforms

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),         # flip left-right with 50% probability
    transforms.RandomRotation(10),             # rotate ±10°
    transforms.RandomCrop(32, padding=4),      # random 32×32 crop after 4-pixel padding
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise each channel
])

# Apply only normalisation at test time (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

Load a torchvision dataset with transforms:
```python
train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transform_train, download=True
)
test_set  = torchvision.datasets.CIFAR10(
    root='./data', train=False, transform=transform_test
)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_set,  batch_size=64, shuffle=False)
```

| Transform | Effect |
|-----------|--------|
| `RandomHorizontalFlip()` | Mirror image left-right (50% chance) |
| `RandomRotation(deg)` | Random rotation by ±deg degrees |
| `RandomCrop(size, padding)` | Pad then crop randomly |
| `ColorJitter(...)` | Random brightness/contrast/saturation |
| `Normalize(mean, std)` | Standardise per channel |
| `ToTensor()` | Convert PIL image to tensor, scale to [0,1] |

---

## 14. GPU Acceleration

Move tensors and models to GPU when available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

# In the training loop — move each batch to the device
for X_batch, y_batch in dataloader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    ...
```

---

## 15. Quick Reference: Full Training Workflow

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Prepare data ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test  = scaler.transform(X_test).astype("float32")

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)          # dtype: torch.int64 for classification
X_test_t  = torch.from_numpy(X_test)
y_test_t  = torch.from_numpy(y_test)

# ── 2. DataLoader ─────────────────────────────────────────────────────────────
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# ── 3. Model ──────────────────────────────────────────────────────────────────
n_features = X_train_t.shape[1]
model = nn.Sequential(
    nn.Linear(n_features, 64), nn.ReLU(),
    nn.Linear(64, 32),         nn.ReLU(),
    nn.Linear(32, n_classes)
)

# ── 4. Loss and optimizer ─────────────────────────────────────────────────────
loss_fn   = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── 5. Training loop ──────────────────────────────────────────────────────────
for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1)

accuracy = (preds == y_test_t).float().mean().item()
print(f"Test accuracy: {accuracy:.3f}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), 'model.pth')
```
