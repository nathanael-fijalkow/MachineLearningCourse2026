# Scikit-Learn Methods Cheatsheet

---

## 1. Core API Methods

### fit()
```python
model.fit(X_train, y_train)
scaler.fit(X_train)
imputer.fit(X_train)
```
- **Purpose**: Learn parameters from training data
- **For models**: Learns weights/coefficients from features and target
- **For transformers**: Calculates statistics (mean, std, min, max, etc.) from data
- **Important**: Only call on training data, never on test data

### transform()
```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- **Purpose**: Apply the learned transformation to data
- **Uses**: Parameters learned during `fit()`
- **Important**: Use same transformer for train and test data

### fit_transform()
```python
X_train_scaled = scaler.fit_transform(X_train)
```
- **Purpose**: Combines `fit()` and `transform()` in one step
- **Equivalent to**: `scaler.fit(X_train); scaler.transform(X_train)`
- **Use only on**: Training data (never on test data)

### predict()
```python
y_pred = model.predict(X_test)
```
- **Purpose**: Generate predictions for new data
- **Returns**: Predicted class labels (classification) or values (regression)

### score()
```python
accuracy = model.score(X_test, y_test)
```
- **Purpose**: Evaluate model on given data
- **Returns**:
  - Classification: Accuracy by default
  - Regression: R-squared (R^2) by default

---

## 2. Data Splitting

### train_test_split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 30% for testing
    random_state=42,    # Reproducibility
    stratify=y,         # Maintain class proportions
    shuffle=True        # Default: shuffle before split
)
```

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `test_size` | Proportion for test set | 0.25 |
| `random_state` | Seed for reproducibility | None |
| `stratify` | Variable to stratify by | None |
| `shuffle` | Shuffle before split | True |

**When to use `stratify=y`**: Imbalanced classification datasets

---

## 3. Scalers

### StandardScaler (Z-score normalization)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- **Formula**: $z = \frac{x - \mu}{\sigma}$
- **Result**: Mean = 0, Standard deviation = 1
- **Use when**: Features have different scales, for algorithms sensitive to feature magnitude (SVM, logistic regression, neural networks)

### MinMaxScaler
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```
- **Formula**: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$
- **Result**: Values in range [0, 1]
- **Use when**: You need bounded values, algorithms that expect input in specific range

### RobustScaler
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```
- **Formula**: Uses median and IQR instead of mean and std
- **Result**: More robust to outliers
- **Use when**: Data has many outliers

### MaxAbsScaler
```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_train)
```
- **Formula**: $x_{scaled} = \frac{x}{|x_{max}|}$
- **Result**: Values in range [-1, 1]
- **Use when**: Data is already centered at zero, sparse data

### Scaler Comparison Table

| Scaler | Formula | Result Range | Best For |
|--------|---------|--------------|----------|
| StandardScaler | (x - mean) / std | Unbounded | General use, normally distributed data |
| MinMaxScaler | (x - min) / (max - min) | [0, 1] | Neural networks, bounded data |
| RobustScaler | (x - median) / IQR | Unbounded | Data with outliers |
| MaxAbsScaler | x / abs(max) | [-1, 1] | Sparse data |

---

## 4. Handling Missing Data

### SimpleImputer
```python
from sklearn.impute import SimpleImputer

# Mean imputation (for numerical)
imputer = SimpleImputer(strategy='mean')

# Median imputation (for numerical with outliers)
imputer = SimpleImputer(strategy='median')

# Most frequent (for categorical)
imputer = SimpleImputer(strategy='most_frequent')

# Constant value
imputer = SimpleImputer(strategy='constant', fill_value=0)

# In all cases:
X_imputed = imputer.fit_transform(X_train)
```

| Strategy | Description | Use When |
|----------|-------------|----------|
| `'mean'` | Replaces with column mean | Numerical, no outliers |
| `'median'` | Replaces with column median | Numerical with outliers |
| `'most_frequent'` | Replaces with mode | Categorical data |
| `'constant'` | Replaces with specified value | When you have domain knowledge |

**Important**: fit on training data, transform both train and test
- `fit()` calculates the replacement value from training data
- `transform()` applies that value to fill missing data

```python
imputer.fit(X_train)
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)  # Uses mean from X_train!
```

---

## 5. Categorical Encoding (Embeddings)

### OneHotEncoder
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_categorical)
```
- **Result**: Creates binary columns for each category
- **Example**: Color [Red, Blue, Green] becomes 3 binary columns
- **Parameters**:
  - `sparse_output=False`: Returns dense array instead of sparse matrix
  - `handle_unknown='ignore'`: Ignores unknown categories in transform

### OrdinalEncoder
```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_encoded = encoder.fit_transform(X_categorical)
```
- **Result**: Maps categories to integers (0, 1, 2, ...)
- **Use when**: Categories have natural ordering

### LabelEncoder (for target variable only)
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```
- **Purpose**: Encode target labels with values 0 to n_classes-1
- **Note**: Use only for target variable, not features

### Comparison Table

| Encoder | Output | Use Case |
|---------|--------|----------|
| OneHotEncoder | Binary columns | Nominal categories (no order) |
| OrdinalEncoder | Integer values | Ordinal categories (with order) |
| LabelEncoder | Integer values | Target variable encoding |

---

## 6. Cross-Validation

### cross_val_score
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
```
- **Purpose**: Evaluate model using K-Fold cross-validation in one line
- **Returns**: Array of scores for each fold
- **Parameters**:
  - `cv`: Number of folds (default=5) or cross-validation splitter
  - `scoring`: Metric to use ('accuracy', 'f1', 'neg_mean_squared_error', etc.)

### cross_validate
```python
from sklearn.model_selection import cross_validate

results = cross_validate(model, X, y, cv=5, 
                         scoring=['accuracy', 'f1'],
                         return_train_score=True)
print(results['test_accuracy'])
print(results['train_accuracy'])
```
- **Purpose**: More detailed cross-validation results
- **Returns**: Dictionary with train/test scores, fit times

### KFold
```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

### StratifiedKFold
```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skfold.split(X, y):
    # Maintains class proportions in each fold
    pass
```
- **Difference from KFold**: Preserves class distribution in each fold
- **Use for**: Classification with imbalanced classes

### ShuffleSplit
```python
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```
- **Purpose**: Random permutation cross-validation
- **Advantage**: More control over train/test size and number of iterations

---

## 7. Validation and Learning Curves

### validation_curve
```python
from sklearn.model_selection import validation_curve
import numpy as np

param_range = np.logspace(-3, 2, num=10)
train_scores, val_scores = validation_curve(
    model, X, y,
    param_name="svc__gamma",  # Pipeline parameter notation
    param_range=param_range,
    cv=5,
    scoring='accuracy'
)
```
- **Purpose**: Evaluate how a single hyperparameter affects training/validation scores
- **X-axis**: Hyperparameter values
- **Returns**: Training and validation scores for each parameter value

### ValidationCurveDisplay
```python
from sklearn.model_selection import ValidationCurveDisplay

ValidationCurveDisplay.from_estimator(
    model, X, y,
    param_name="svc__gamma",
    param_range=np.logspace(-3, 2, num=30),
    cv=5
)
```
- **Purpose**: Plot validation curve directly

### learning_curve
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5
)
```
- **Purpose**: Evaluate model performance for different training set sizes
- **X-axis**: Number of training samples
- **Returns**: Training sizes and corresponding train/validation scores

### LearningCurveDisplay
```python
from sklearn.model_selection import LearningCurveDisplay

LearningCurveDisplay.from_estimator(model, X, y, cv=5)
```
- **Purpose**: Plot learning curve directly

---

## 8. Evaluation Metrics and Scores

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Individual metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Comprehensive report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
```

### Regression Metrics
```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

### Scoring Parameters for Cross-Validation

| Classification | Regression |
|---------------|------------|
| 'accuracy' | 'r2' |
| 'balanced_accuracy' | 'neg_mean_squared_error' |
| 'f1' | 'neg_root_mean_squared_error' |
| 'precision' | 'neg_mean_absolute_error' |
| 'recall' | |
| 'roc_auc' | |

**Note**: Regression scores are negated because sklearn maximizes scores

---

## 9. Pipelines

### Pipeline
```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)
```
- **Purpose**: Chain multiple transformers and a final estimator
- **Advantage**: Prevents data leakage, cleaner code

### make_pipeline
```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf')
)
```
- **Purpose**: Simplified pipeline creation with automatic naming
- **Names**: Generated automatically (e.g., 'standardscaler', 'svc')

### Accessing Pipeline Parameters
```python
# Get parameter names
print(pipe.get_params().keys())

# For nested parameters use __
# Example: svc__gamma for gamma parameter of SVC in pipeline
```

---

## 10. ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'  # or 'drop'
)

X_transformed = preprocessor.fit_transform(X_train)
```
- **Purpose**: Apply different transformations to different columns
- **Use when**: Dataset has mixed numerical and categorical features

### Complete Example
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Define column types
numerical_cols = ['Age', 'Fare']
categorical_cols = ['Sex', 'Embarked']

# Define transformers
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine in ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Create full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

model.fit(X_train, y_train)
```

---

## 11. Quick Reference: Common Workflow

```python
# 1. Imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. Load and split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocess (fit on train only!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

# 6. Cross-validation for more robust estimate
scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Score: {scores.mean():.3f} +/- {scores.std():.3f}")
```
