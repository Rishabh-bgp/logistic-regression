# Iris Multi-Class Classification with Logistic Regression

**Graduate capstone** — a complete, reproducible pipeline for multi-class logistic regression on the classic Iris dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikitlearn&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.67%25-success)]()

---

## Overview

This project implements a rigorous baseline for **three-class species classification** (setosa, versicolor, virginica) using all four morphometric features:

- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)

The workflow is designed as a graduate-level capstone: exploratory analysis → stratified splitting → training → evaluation → cross-validation → hyperparameter search → interpretability → decision-boundary visualization → honest discussion of limits.

**One-line description (≤350 characters)**

This graduate capstone applies multi-class logistic regression to the Iris dataset. It includes EDA, stratified 80/20 split, liblinear OvR training, 96.67% accuracy with confusion matrix and report, 5-fold stratified CV, GridSearchCV, coefficient analysis, decision-boundary plots, plus limitations and future work.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Hold-out test accuracy | **96.67%** (29 / 30 correct) |
| Split | 120 train / 30 test, **stratified**, `random_state=42` |
| Solver / strategy | `liblinear` + One-vs-Rest |
| Confusion (test) | setosa 10/10 · versicolor 9/10 · virginica 10/10 |
| Weakest class | versicolor (1 sample predicted as virginica) |

Classification report (test):

```
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10
    accuracy                           0.97        30
```

Petal length and petal width dominate the learned coefficients — consistent with the well-known linear separability of setosa and the partial overlap of versicolor / virginica.

---

## Pipeline

1. **Load & inspect** — `sklearn.datasets.load_iris()` (150 × 4, balanced 50/50/50).
2. **Prepare** — `X = iris.data`, `y = iris.target`.
3. **Split** — `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
4. **Train** — `LogisticRegression(solver="liblinear", multi_class="ovr", random_state=42)`.
5. **Evaluate** — accuracy, confusion matrix, classification report, seaborn heatmap.
6. **Validate** — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` + `cross_val_score`.
7. **Tune** — `GridSearchCV` over regularization / solver settings.
8. **Interpret** — class-wise coefficients; 2-D decision boundaries on petal features.
9. **Reflect** — linearity assumption, feature dependence, small-n limits, next models.

---

## Quick Start

```bash
# clone / open the notebook environment
pip install scikit-learn matplotlib seaborn pandas numpy

jupyter notebook Logistic_Regression.ipynb
# or
jupyter lab Logistic_Regression.ipynb
```

Core training block:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(random_state=42, solver="liblinear", multi_class="ovr")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## Project Layout

```
.
├── Logistic_Regression.ipynb   # full narrative + executed cells
└── README.md                   # this file
```

All figures (confusion heatmap, decision boundaries) are generated inside the notebook.

---

## Limitations (called out on purpose)

- Logistic regression assumes a **linear** log-odds relationship.
- Features are treated as independent; sepal/petal measurements are correlated.
- n = 150 is a teaching set, not a stress test.
- OvR + `liblinear` is a solid baseline, not the accuracy ceiling.

## Future Work

- Polynomial / interaction features
- Feature scaling ablation
- SVM, trees, random forests, small MLP
- OvR ROC / AUC per class
- Error analysis on the single versicolor miss
- Ensembles (AdaBoost / gradient boosting)

---

## Why this project exists

Iris is solved. That is the point.

A capstone is not “can I beat 96%.” It is “can I run a *defensible* classification study end-to-end — split correctly, validate honestly, interpret coefficients, draw the boundary, and say what the model cannot do.”

This notebook does that.

---

## License

MIT. Use it, fork it, cite it, break it.
