# Decision Trees and Ensemble Methods

## Notation

| Symbol | Meaning |
|--------|---------|
| $m$ | Number of training examples |
| $n$ | Number of features |
| $x^{(i)} \in \mathbb{R}^n$ | The $i$-th training example |
| $y^{(i)}$ | The label of the $i$-th example |
| $S$ | A set of training examples |
| $K$ | Number of classes |
| $p_k$ | Proportion of class $k$ in a set |
| $T$ | Number of trees (ensemble methods) |

---

## Part 1: Decision Trees

### 1.1 The Idea

A decision tree is a model that makes predictions by following a sequence of **if-then-else** rules, organised in a tree structure:

- Each **internal node** tests a condition on a feature (e.g., $x_3 \leq 5.2$).
- Each **branch** corresponds to an outcome of the test.
- Each **leaf** assigns a prediction (a class label for classification, a value for regression).

To predict for a new point $x$: start at the root and follow the branches according to the feature values of $x$ until you reach a leaf.

### 1.2 How to Build a Tree: Recursive Splitting

The tree is built **top-down** by recursively choosing the best feature and threshold to split the data. The algorithm is sometimes called **greedy recursive binary splitting**:

> **Algorithm: Build Decision Tree**
> 
> **Input:** A set of examples $S$
> 
> 1. If a **stopping criterion** is met (e.g., all examples have the same label, maximum depth reached, or too few examples), create a **leaf** node with the majority class (or mean value for regression).
> 2. Otherwise:
>    - For each feature $j$ and each threshold $t$:
>      - Split $S$ into $S_{\text{left}} = \{(x, y) \in S : x_j \leq t\}$ and $S_{\text{right}} = \{(x, y) \in S : x_j > t\}$.
>      - Compute the **impurity reduction** (information gain).
>    - Choose the split $(j^*, t^*)$ that **maximises** the impurity reduction.
>    - Recursively build the left subtree on $S_{\text{left}}$ and the right subtree on $S_{\text{right}}$.

### 1.3 Impurity Measures (Classification)

An impurity measure quantifies how "mixed" the classes are in a set. A pure node contains only one class.

Let $p_k$ denote the proportion of class $k$ in a node, so $\sum_{k=1}^{K} p_k = 1$.

#### Entropy (used by ID3, C4.5)

$$H(S) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

with the convention $0 \log_2 0 = 0$.

- $H = 0$ when all examples belong to one class (pure node).
- $H$ is maximised when all classes are equally represented ($p_k = \frac{1}{K}$ for all $k$).
- For binary classification: $H = -p \log_2 p - (1-p) \log_2 (1-p)$, with maximum at $p = 0.5$ where $H = 1$.

#### Gini Impurity (used by CART)

$$G(S) = 1 - \sum_{k=1}^{K} p_k^2 = \sum_{k=1}^{K} p_k (1 - p_k)$$

**Interpretation:** $G(S)$ is the probability that a randomly chosen example would be misclassified if it were labelled randomly according to the class distribution in $S$.

- $G = 0$ when all examples belong to one class.
- For binary classification: $G = 2p(1-p)$, with maximum at $p = 0.5$ where $G = 0.5$.

#### Comparison of Entropy and Gini

For binary classification with positive class proportion $p$:

| $p$ | Entropy $H$ | Gini $G$ |
|-----|---------|------|
| 0.0 | 0.000 | 0.000 |
| 0.1 | 0.469 | 0.180 |
| 0.2 | 0.722 | 0.320 |
| 0.3 | 0.881 | 0.420 |
| 0.5 | 1.000 | 0.500 |

Both are concave functions that peak at $p = 0.5$. In practice, they yield very similar trees. Gini is slightly faster to compute (no logarithm).

### 1.4 Information Gain

The **information gain** of a split measures the reduction in impurity. For a node with set $S$ split into $S_{\text{left}}$ and $S_{\text{right}}$:

$$\text{Gain}(S, j, t) = I(S) - \frac{|S_{\text{left}}|}{|S|} \, I(S_{\text{left}}) - \frac{|S_{\text{right}}|}{|S|} \, I(S_{\text{right}})$$

where $I$ is the impurity measure (entropy or Gini). The split that **maximises** the gain is chosen.

### 1.5 Splitting for Regression Trees

For regression, we use the **variance** (or equivalently the MSE) as the impurity measure:

$$I(S) = \frac{1}{|S|} \sum_{(x,y) \in S} (y - \bar{y}_S)^2$$

where $\bar{y}_S = \frac{1}{|S|} \sum_{(x,y) \in S} y$ is the mean of the target values in $S$.

The prediction at a leaf is the **mean** of the target values of the training examples that fall in that leaf.

### 1.6 Classical Algorithms

| Algorithm | Year | Impurity | Split type | Features |
|-----------|------|----------|-----------|----------|
| **ID3** | 1986 | Entropy | Multi-way (one branch per value) | Categorical only |
| **C4.5** | 1993 | Gain ratio | Multi-way | Categorical + numerical |
| **CART** | 1984 | Gini (classification) / MSE (regression) | Binary | Categorical + numerical |

> **scikit-learn** implements CART (binary splits, Gini or entropy criterion).

### 1.7 Controlling Complexity: Pruning and Hyperparameters

Decision trees are **high-variance** models: they tend to overfit by growing deep and memorising the training data.

**Pre-pruning** (stopping criteria):
- `max_depth`: Maximum depth of the tree.
- `min_samples_split`: Minimum number of examples to allow a split.
- `min_samples_leaf`: Minimum number of examples in a leaf.
- `max_features`: Maximum number of features to consider at each split.

**Post-pruning** (grow full tree, then remove branches):
- **Cost-complexity pruning** (CART): Penalise the tree complexity. For a tree $T$ with $|T|$ leaves:

$$R_\alpha(T) = \sum_{\ell \in \text{leaves}(T)} \frac{|S_\ell|}{|S|} \, I(S_\ell) + \alpha \, |T|$$

where $\alpha \geq 0$ is the complexity parameter. Larger $\alpha$ favours simpler trees.

In scikit-learn, this is controlled by `ccp_alpha`.

### 1.8 Advantages and Limitations

| Advantages | Limitations |
|-----------|------------|
| Interpretable (white-box model) | High variance (tends to overfit) |
| No feature scaling needed | Unstable (small data changes → different tree) |
| Handles both numerical and categorical features | Greedy algorithm (no guarantee of global optimum) |
| Fast training and prediction | Decision boundaries are axis-aligned |
| Naturally handles multi-class problems | Poor performance compared to ensembles |

---

## Part 2: Bagging (Bootstrap Aggregating)

### 2.1 The Key Insight: Variance Reduction

If we have $T$ independent random variables $Z_1, \ldots, Z_T$, each with variance $\sigma^2$, then the variance of their mean is:

$$\text{Var}\left(\frac{1}{T} \sum_{t=1}^{T} Z_t\right) = \frac{\sigma^2}{T}$$

**Idea:** If a single decision tree has high variance, can we average multiple trees to reduce it?

**Problem:** We only have one training set, so the trees would be identical.

**Solution:** Create artificial "new" training sets using the **bootstrap**.

### 2.2 The Bootstrap

A **bootstrap sample** is obtained by sampling $m$ examples **with replacement** from the original training set of $m$ examples.

Key properties:
- Each bootstrap sample has the same size $m$ as the original dataset.
- Some examples appear multiple times, others not at all.
- On average, each bootstrap sample contains about $1 - (1 - \frac{1}{m})^m \approx 1 - \frac{1}{e} \approx 63.2\%$ of the unique original examples.
- The remaining $\approx 36.8\%$ are called the **out-of-bag (OOB)** examples for that sample.

### 2.3 The Bagging Algorithm

> **Algorithm: Bagging**
> 
> **Input:** Training set $S$, number of trees $T$
> 
> **Training:**
> 1. For $t = 1, \ldots, T$:
>    - Draw a bootstrap sample $S_t$ from $S$.
>    - Train a decision tree $h_t$ on $S_t$ (typically grown deep, without pruning).
> 
> **Prediction:**
> - **Classification:** $\hat{y} = \text{majority vote}\{h_1(x), h_2(x), \ldots, h_T(x)\}$
> - **Regression:** $\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(x)$

---

## Part 3: Random Forests

### 3.1 The Idea: Decorrelate the Trees

Random Forests improve upon bagging by **reducing the correlation** $\rho$ between trees. The key modification: at each split, only a **random subset of features** is considered.

### 3.2 The Algorithm

> **Algorithm: Random Forest**
> 
> **Input:** Training set $S$, number of trees $T$, number of features per split $q$
> 
> **Training:**
> 1. For $t = 1, \ldots, T$:
>    - Draw a bootstrap sample $S_t$ from $S$.
>    - Train a decision tree $h_t$ on $S_t$ with the following modification:
>      - At each node, select $q$ features uniformly at random from all $n$ features.
>      - Find the best split using **only** these $q$ features.
>    - Grow the tree fully (no pruning).
> 
> **Prediction:** Same as bagging (majority vote or average).

### 3.3 Choosing $q$ (the `max_features` parameter)

| Task | Recommended $q$ |
|------|-----------------|
| Classification | $q = \lfloor\sqrt{n}\rfloor$ |
| Regression | $q = \lfloor n/3 \rfloor$ |

- $q = n$: identical to bagging (no feature randomness).
- $q = 1$: maximum randomness (each split uses a single random feature).
- Smaller $q$ → lower correlation $\rho$ but higher bias per tree. The sweet spot is typically $q = \sqrt{n}$.

### 3.5 Summary: Bagging vs. Random Forest

| | Bagging | Random Forest |
|---|---------|---------------|
| Bootstrap samples | Yes | Yes |
| Features at each split | All $n$ | Random subset of $q$ |
| Tree correlation $\rho$ | Higher | Lower |
| Variance reduction | Good | Better |
| Typically better performance | — | Yes |

---

## Part 4: Boosting (AdaBoost)

### 4.1 A Different Philosophy

Bagging and Random Forests build **strong learners in parallel** and average them.

Boosting builds **weak learners sequentially**, each one focusing on the mistakes of the previous ones. The final model is a weighted combination of all weak learners.

A **weak learner** is a model only slightly better than random guessing.

### 4.2 AdaBoost (Adaptive Boosting)

> **Algorithm: AdaBoost (binary classification, $y \in \{-1, +1\}$)**
> 
> **Input:** Training set $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$, number of rounds $T$
> 
> **Initialisation:** Set equal weights $w_1^{(i)} = \frac{1}{m}$ for all $i$.
> 
> **For** $t = 1, \ldots, T$:
> 
> 1. **Train** a weak learner $h_t$ on the **weighted** training set (weights $w_t^{(i)}$).
> 
> 2. **Compute the weighted error:**
> $$\epsilon_t = \sum_{i=1}^{m} w_t^{(i)} \, \mathbb{1}[h_t(x^{(i)}) \neq y^{(i)}]$$
> 
> 3. **Compute the learner weight:**
> $$\alpha_t = \frac{1}{2} \ln\!\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
> 
> 4. **Update the example weights:**
> $$w_{t+1}^{(i)} = w_t^{(i)} \cdot \exp\!\left(-\alpha_t \, y^{(i)} \, h_t(x^{(i)})\right)$$
> 
> 5. **Normalise** the weights so that $\sum_i w_{t+1}^{(i)} = 1$.
> 
> **Final prediction:**
> $$H(x) = \text{sign}\!\left(\sum_{t=1}^{T} \alpha_t \, h_t(x)\right)$$

### 4.3 Understanding the Weight Updates

The update rule applies differently depending on whether example $i$ is correctly or incorrectly classified:

- **Correctly classified** ($y^{(i)} h_t(x^{(i)}) = +1$): weight is **multiplied** by $e^{-\alpha_t} < 1$ → weight **decreases**.
- **Incorrectly classified** ($y^{(i)} h_t(x^{(i)}) = -1$): weight is **multiplied** by $e^{+\alpha_t} > 1$ → weight **increases**.

This forces the next weak learner to focus on the previously misclassified examples.

### 4.4 Understanding $\alpha_t$

The formula $\alpha_t = \frac{1}{2} \ln\!\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$ assigns larger weights to more accurate learners:

| $\epsilon_t$ | $\alpha_t$ | Interpretation |
|---|---|---|
| 0 | $+\infty$ | Perfect classifier: infinite weight |
| 0.1 | 1.10 | Good classifier: high weight |
| 0.3 | 0.42 | Decent classifier: moderate weight |
| 0.5 | 0 | Random guessing: zero weight |
| $> 0.5$ | $< 0$ | Worse than random: negative weight (flips predictions) |

---

## Part 5: Gradient Boosting

### 5.1 The General Framework

Gradient boosting generalises the boosting idea to **arbitrary differentiable loss functions**. Instead of reweighting examples (like AdaBoost), it fits each new weak learner to the **negative gradient** (pseudo-residuals) of the loss function.

The model is an **additive expansion**:

$$F(x) = \sum_{t=1}^{T} \eta \, h_t(x)$$

where $\eta$ is the **learning rate** (shrinkage parameter) and $h_t$ is a regression tree.

### 5.2 The Key Idea: Gradient Descent in Function Space

We want to minimise:

$$J(F) = \sum_{i=1}^{m} L(y^{(i)}, F(x^{(i)}))$$

In ordinary gradient descent (parameter space), we update parameters by moving in the direction of the negative gradient.

In **functional gradient descent**, we update the *function* $F$ by adding a new component $h_t$ that approximates the negative gradient of the loss:

$$F_t(x) = F_{t-1}(x) + \eta \, h_t(x)$$

where $h_t$ is trained to predict the **pseudo-residuals**:

$$r_t^{(i)} = -\frac{\partial L(y^{(i)}, F(x^{(i)}))}{\partial F(x^{(i)})} \Bigg|_{F = F_{t-1}}$$

### 5.3 The Algorithm

> **Algorithm: Gradient Boosting**
> 
> **Input:** Training set $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$, loss function $L$, number of rounds $T$, learning rate $\eta$
> 
> **Initialisation:** $F_0(x) = \arg\min_c \sum_{i=1}^{m} L(y^{(i)}, c)$ (a constant prediction, e.g., the mean for regression)
> 
> **For** $t = 1, \ldots, T$:
> 
> 1. **Compute pseudo-residuals:** For each $i = 1, \ldots, m$:
> $$r_t^{(i)} = -\frac{\partial L(y^{(i)}, F(x^{(i)}))}{\partial F(x^{(i)})} \Bigg|_{F = F_{t-1}}$$
> 
> 2. **Fit a regression tree** $h_t$ to the pseudo-residuals $\{(x^{(i)}, r_t^{(i)})\}_{i=1}^m$.
> 
> 3. **Update the model:**
> $$F_t(x) = F_{t-1}(x) + \eta \, h_t(x)$$
> 
> **Final prediction:** $F_T(x)$

### 5.4 Pseudo-Residuals for Common Loss Functions

| Loss | $L(y, F)$ | Pseudo-residual $r = -\frac{\partial L}{\partial F}$ | Task |
|------|-----------|------------------------------------------------------|------|
| **Squared error** | $\frac{1}{2}(y - F)^2$ | $y - F$ (the ordinary residual) | Regression |
| **Absolute error** | $\|y - F\|$ | $\text{sign}(y - F)$ | Regression |
| **Log loss** (binary) | $\log(1 + e^{-yF})$ | $\frac{y}{1 + e^{yF}}$ | Classification |
| **Exponential** | $e^{-yF}$ | $y \, e^{-yF}$ | AdaBoost equiv. |

For **squared error**: the pseudo-residuals are simply $r_t^{(i)} = y^{(i)} - F_{t-1}(x^{(i)})$, so each new tree is fit to the **residual errors** of the current ensemble. This is the most intuitive case.

### 5.5 The Role of the Learning Rate (Shrinkage)

The learning rate $\eta \in (0, 1]$ controls how much each tree contributes:

$$F_t(x) = F_{t-1}(x) + \eta \, h_t(x)$$

- **Small $\eta$** (e.g., 0.01–0.1): requires more trees $T$, but generalises better (strong regularisation effect).
- **Large $\eta$** (e.g., 1.0): fewer trees needed, but higher risk of overfitting.

> **Rule of thumb:** Use a small learning rate (0.01–0.1) with a large number of trees, and select $T$ by early stopping on a validation set.

### 5.6 Regularisation Techniques

Gradient boosting is prone to overfitting. Several techniques help:

| Technique | Description | Parameter |
|-----------|-------------|-----------|
| **Learning rate (shrinkage)** | Scale each tree's contribution | $\eta$ (`learning_rate`) |
| **Number of trees** | Stop before overfitting | $T$ (`n_estimators`) |
| **Tree depth** | Use shallow trees (typically 3–8) | `max_depth` |
| **Subsampling (stochastic GB)** | Train each tree on a random fraction of the data | `subsample` |
| **Column subsampling** | Use a random subset of features per tree | `max_features` |
| **Min samples per leaf** | Minimum examples in each leaf | `min_samples_leaf` |
| **L2 regularisation** | Penalise leaf values | `reg_lambda` |

### 5.7 XGBoost, LightGBM, and CatBoost

These are optimised implementations of gradient boosting that add engineering improvements:

| Library | Key innovations |
|---------|----------------|
| **XGBoost** | Second-order approximation (Newton step), regularised objective, column subsampling, sparsity-aware splits, parallel tree construction |
| **LightGBM** | Gradient-based one-side sampling (GOSS), exclusive feature bundling (EFB), histogram-based splitting, leaf-wise tree growth |
| **CatBoost** | Ordered boosting (reduces prediction shift), native categorical feature handling, symmetric trees |

---

## Part 6: Comparison and Summary

| Method | Strategy | Bias | Variance | Key mechanism |
|--------|----------|------|----------|---------------|
| **Single tree** | — | Low | High | — |
| **Bagging** | Parallel averaging | Low | Reduced | Bootstrap + averaging |
| **Random Forest** | Parallel averaging | Low (slightly higher) | Further reduced | Bootstrap + feature randomness |
| **AdaBoost** | Sequential focusing | Reduced | Can increase | Reweighting examples |
| **Gradient Boosting** | Sequential correcting | Reduced | Can increase | Fitting residuals |

- **Bagging / Random Forest:** Reduce **variance** by averaging many high-variance models.
- **Boosting:** Reduce **bias** by iteratively correcting errors (each tree fixes mistakes of the ensemble so far).
