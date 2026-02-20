# Machine Learning Concepts Cheatsheet

---

## 1. Problem Types

### Classification
- **Definition**: Predicting a discrete categorical outcome (class label)
- **Examples**: 
  - Binary classification: email spam detection (spam/not spam), customer churn (Yes/No)
  - Multi-class classification: image recognition (dog/cat/bird), species prediction
- **Output**: Class labels (discrete categories)
- **Typical metrics**: Accuracy, Precision, Recall, F1-Score

### Regression
- **Definition**: Predicting a continuous numerical value
- **Examples**: 
  - Predicting house prices
  - Estimating temperature
  - Forecasting stock prices
- **Output**: Continuous values (real numbers)
- **Typical metrics**: Mean Squared Error (MSE), R-squared (R^2), Mean Absolute Error (MAE)

---

## 2. Data Types

### Features (Input Variables)
- Also called: predictors, independent variables, attributes
- The input data used to make predictions

### Target (Output Variable)
- Also called: label, dependent variable, response
- The value we want to predict

### Categorical Features
- **Definition**: Features that take discrete values from a finite set
- **Examples**: 
  - Sex (Male/Female)
  - Ticket class (1st/2nd/3rd)
  - Embarked port (C/Q/S)
- **Encoding required**: Must be converted to numerical format for most ML algorithms

### Numerical Features
- **Definition**: Features that take continuous or discrete numerical values
- **Subtypes**:
  - **Continuous**: Age, Fare, Temperature (can take any value in a range)
  - **Discrete**: Count of siblings, number of rooms (whole numbers)
- **Scaling often required**: Algorithms like SVM, logistic regression perform better with scaled features

### Ordinal Features
- **Definition**: Categorical features with a natural order
- **Examples**: Education level (High School < Bachelor < Master < PhD)
- **Note**: Order matters, but distances between values may not be equal

---

## 3. Data Splitting

### Train/Test Split
- **Purpose**: Evaluate model on data it hasn't seen during training
- **Typical ratio**: 70-80% training, 20-30% testing
- **Key parameter**: `random_state` ensures reproducibility

### Why Split Data?
- Prevents overfitting to the entire dataset
- Provides an unbiased estimate of model performance
- Simulates real-world deployment scenario

### Important Considerations
- **Stratification**: Maintains class proportions in splits (critical for imbalanced datasets)
- **random_state**: Set for reproducible results

---

## 4. Overfitting vs Underfitting

### Overfitting (High Variance)
- **Definition**: Model is too complex and captures noise instead of the underlying pattern
- **Symptoms**:
  - High training accuracy, low test/validation accuracy
  - Large gap between training and validation curves
- **Causes**: 
  - Too many features
  - Too complex model (e.g., high polynomial degree)
  - Too little training data
- **Solutions**:
  - Add regularization
  - Reduce model complexity
  - Get more training data
  - Remove features (feature selection)

### Underfitting (High Bias)
- **Definition**: Model is too simple to capture the underlying structure of the data
- **Symptoms**:
  - Low training accuracy AND low test accuracy
  - Both curves converge at a high error value
- **Causes**:
  - Model too simple
  - Not enough features
  - Too much regularization
- **Solutions**:
  - Use a more complex model
  - Add more features
  - Reduce regularization
  - Engineer new features

### Bias-Variance Tradeoff
- **Bias**: Error from overly simplistic assumptions (leads to underfitting)
- **Variance**: Error from sensitivity to small fluctuations in training data (leads to overfitting)
- **Goal**: Find the sweet spot that minimizes both

---

## 5. Cross-Validation

### Purpose
- More robust estimate of model performance than single train/test split
- Uses all data for both training and validation
- Reduces variance in performance estimates

### K-Fold Cross-Validation
- **Process**:
  1. Split data into K equal folds
  2. For each fold: train on K-1 folds, validate on 1 fold
  3. Average the K validation scores
- **Example**: K=5 means model is trained 5 times, each time using 80% for training, 20% for validation
- **Note**: Each sample appears in validation set exactly once

### Stratified K-Fold
- Maintains class proportions in each fold
- Essential for imbalanced classification problems

### High Variance in Cross-Validation Scores
- If scores vary widely across folds (e.g., 0.45 to 0.92)
- Suggests dataset is too small or has inconsistent distributions

---

## 6. Learning Curves

### Definition
- Plot of model performance vs training set size
- Shows training score and validation score as more data is added

### X-axis
- Number of training samples (training set size)

### How to Interpret
| Scenario | Training Score | Validation Score | Diagnosis |
|----------|---------------|------------------|-----------|
| Converge at high value | High error | High error | Underfitting (High Bias) |
| Large gap | Very high | Much lower | Overfitting (High Variance) |
| Both high, converging | Good | Good | Good fit |
| Validation still increasing | - | Still improving | More data would help |

### Key Insights
- If validation score is still increasing: model would benefit from more training data
- If both scores converge at poor value: adding more data won't help (need more complex model)

---

## 7. Validation Curves

### Definition
- Plot of model performance vs hyperparameter value
- Shows training score and validation score as hyperparameter changes

### X-axis
- Value of a single hyperparameter (e.g., polynomial degree, regularization strength, gamma)

### How to Interpret
- **Sweet spot**: Where validation score is maximized
- **Left of sweet spot**: Underfitting (model too simple)
- **Right of sweet spot**: Overfitting (model too complex)

### Example
- For Decision Tree `max_depth`:
  - Low depth: underfitting
  - Optimal depth: best generalization
  - High depth: overfitting (training = 1.0, validation drops)

---

## 8. Data Leakage

### Definition
- Information from outside the training dataset is used in model training
- Results in overly optimistic performance estimates

### Common Causes
- Scaling entire dataset before splitting (most common mistake)
- Including features derived from target variable
- Not respecting temporal order in time series

### Prevention
- Always split data FIRST, then preprocess
- Fit transformers (scalers, imputers) only on training data
- Use pipelines to encapsulate preprocessing and model

### Example of Data Leakage
```python
# WRONG - causes data leakage
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# CORRECT - no data leakage
X_train, X_test = train_test_split(X)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 9. Model Evaluation Metrics

### Classification Metrics

| Metric | Definition | Use When |
|--------|------------|----------|
| **Accuracy** | (TP + TN) / Total | Balanced classes |
| **Precision** | TP / (TP + FP) | Cost of false positives is high |
| **Recall** | TP / (TP + FN) | Cost of false negatives is high |
| **F1-Score** | Harmonic mean of precision and recall | Imbalanced classes |
| **Balanced Accuracy** | Average of recall for each class | Imbalanced classes |

### Regression Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **MSE** | Mean of squared errors | Lower is better, penalizes large errors |
| **RMSE** | Square root of MSE | Same units as target |
| **MAE** | Mean of absolute errors | More robust to outliers |
| **R-squared** | Proportion of variance explained | 1.0 is perfect, 0 means no better than mean |

---

## 10. Handling Imbalanced Data

### Problem
- When one class is much more frequent than others
- Standard accuracy can be misleading

### Solutions
- Use `stratify=y` in train_test_split
- Use balanced metrics (F1, balanced accuracy)
- Resampling techniques (oversampling minority, undersampling majority)
- Class weights in algorithms