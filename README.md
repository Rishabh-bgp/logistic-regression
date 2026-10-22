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

## 🔗 Other repositories by Rishabh Aryan

| Repository | Description | Live |
|---|---|---|
| [100-days-of-python](https://github.com/Rishabh-bgp/100-days-of-python) | My original, offline, no-API-key solutions to all 100 projects of the 100 Days of Code Python bootcamp curriculum… | — |
| [Agentic-Engineering-Ebook](https://github.com/Rishabh-bgp/Agentic-Engineering-Ebook) | Agentic Engineering is a practical notebook-book on specifying, generating, verifying, and refining multi-tenant… | — |
| [beed-ml-study](https://github.com/Rishabh-bgp/beed-ml-study) | Provenance audit and leakage-aware benchmark of seizure-type classifiers on the Bangalore EEG Epilepsy Dataset — IIT… | — |
| [Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction](https://github.com/Rishabh-bgp/Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction) | Brain–computer interfaces (BCIs) link neural activity to external devices, translating brain signals via algorithms… | — |
| [building-a-virtual-cell-in-python](https://github.com/Rishabh-bgp/building-a-virtual-cell-in-python) | A cell has a normal gene-activity pattern. Knock out one gene and that pattern changes. This project trains a small… | — |
| [Chronic-Kidney-Disease-Stage-Identification-in-HIV-Infected-Patients-Using-Supervised-ML](https://github.com/Rishabh-bgp/Chronic-Kidney-Disease-Stage-Identification-in-HIV-Infected-Patients-Using-Supervised-ML) | ML is transforming Chronic Kidney Disease (CKD) detection in HIV patients, catching early asymptomatic stages to… | — |
| [cobratate](https://github.com/Rishabh-bgp/cobratate) | Cobratate, an esoteric imperative programming language. Tate-flavored syntax, ordinary semantics, and a Python… | [open](https://rishabh-bgp.github.io/cobratate/) |
| [coda-synth](https://github.com/Rishabh-bgp/coda-synth) | CoDA-Synth: stylometric detection of LLM-rewritten dark-web documents with sparse TF-IDF + LinearSVC. Research lab,… | [open](https://shale-ember-delta-kite.grok.me/) |
| [IIT-ROPAR-GENAI-FOUNDATIONS](https://github.com/Rishabh-bgp/IIT-ROPAR-GENAI-FOUNDATIONS) | Free open-source Generative AI foundations ebooks (2026) covering Claude, Grok, Gemini, Cursor, OpenAI and more —… | — |
| [iris-classification](https://github.com/Rishabh-bgp/iris-classification) | Iris flower classification in Python: scikit-learn model comparison, tuning and evaluation, FastAPI /predict API,… | [open](https://rishabh-bgp.github.io/iris-classification/) |
| [kbc-game](https://github.com/Rishabh-bgp/kbc-game) | Kaun Banega Crorepati (KBC) quiz game for the terminal in Python: 16-question prize ladder, safe levels and 4… | — |
| [knn-classifier](https://github.com/Rishabh-bgp/knn-classifier) | Simple K-Nearest Neighbors classifier on the classic Iris dataset using scikit-learn. Loads data, trains the model,… | — |
| [linear-regression-model-](https://github.com/Rishabh-bgp/linear-regression-model-) | Supervised linear regression on the scikit-learn diabetes dataset. A single feature (BMI) is used to predict… | — |
| [logistics-website-capstone](https://github.com/Rishabh-bgp/logistics-website-capstone) | Logistics website capstone: Kargonaut Logistics, a premium freight site mixing claymorphism, glassmorphism and… | [open](https://rishabh-bgp.github.io/logistics-website-capstone/) |
| [lovelight-leap](https://github.com/Rishabh-bgp/lovelight-leap) | 💖 Lovelight Leap: a love-themed 2D platformer in vanilla JS canvas. 4 worlds, a boss you cheer up, original… | [open](https://rishabh-bgp.github.io/lovelight-leap/) |
| [machine-learning-part-1](https://github.com/Rishabh-bgp/machine-learning-part-1) | Machine learning in Python with scikit-learn: heart disease classification and medical insurance cost regression… | — |
| [Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail](https://github.com/Rishabh-bgp/Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail) | Agentic commerce lets shopping and seller agents place orders without showing operational truth. This IS case argues… | — |
| [morocco-city-aware-load-forecasting](https://github.com/Rishabh-bgp/morocco-city-aware-load-forecasting) | City-aware short-term load forecasting and peak detection on high-resolution Moroccan smart-meter data (5 zones).… | — |
| [NIELIT-DATA-SCIENCE](https://github.com/Rishabh-bgp/NIELIT-DATA-SCIENCE) | Hands-on NumPy, Pandas, and Matplotlib Jupyter notebooks for the NIELIT Data Science curriculum — practical,… | — |
| [NIELIT-MACHINE-LEARNING-PROJECTS](https://github.com/Rishabh-bgp/NIELIT-MACHINE-LEARNING-PROJECTS) | Nineteen end-to-end machine learning projects: classification, regression, clustering, and recommendation, with… | — |
| [NIELIT-PYTHON-](https://github.com/Rishabh-bgp/NIELIT-PYTHON-) | Open-source Python teaching notebooks: a language reference and five capstone programs, with detailed documentation.… | — |
| [pan-cancer-classification](https://github.com/Rishabh-bgp/pan-cancer-classification) | Leakage, equivalence, conformal reliability and data efficiency on the UCI TCGA Pan-Cancer RNA-Seq dataset — IIT… | — |
| [Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks](https://github.com/Rishabh-bgp/Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks) | This notebook details the implementation and evaluation of a Bayesian perspective Early stage Event Prediction… | — |
| [prediction-as-power](https://github.com/Rishabh-bgp/prediction-as-power) | Laboratory and working paper on prediction as power in European electricity markets. A DE-like hourly simulator… | — |
| [real-estate-valuation-taiwan](https://github.com/Rishabh-bgp/real-estate-valuation-taiwan) | Taiwan real estate valuation (Xindian, New Taipei): house price per ping prediction with scikit-learn gradient… | [open](https://rishabh-bgp.github.io/real-estate-valuation-taiwan/) |
| [snakes-and-ladders](https://github.com/Rishabh-bgp/snakes-and-ladders) | 🐍🪜 Original Snakes & Ladders browser game: hand-drawn SVG snakes and ladders, 3D dice, 2-4 humans or bots, sound,… | [open](https://rishabh-bgp.github.io/snakes-and-ladders/) |
| [spam-scan](https://github.com/Rishabh-bgp/spam-scan) | Explainable spam, phishing & Indian-scam detector for SMS/email (English, Hindi, Hinglish): TF-IDF + Logistic… | [open](https://spam-scan.onrender.com) |
| [vedic-math-hybrid-kt](https://github.com/Rishabh-bgp/vedic-math-hybrid-kt) | Hybrid Vedic Math AI: rule-based sutras (Urdhva-Tiryagbhyam, Nikhilam, Yavadunam) plus RF classifier that cleans… | — |
| [wind-turbine-power-model](https://github.com/Rishabh-bgp/wind-turbine-power-model) | Single-turbine power curves + wind-farm simulation. Weibull wind stats, hourly NASA-style speed data, and… | — |
| [zoo-website-capstone](https://github.com/Rishabh-bgp/zoo-website-capstone) | Capstone project: a fictional zoo website built twice — a beginner-style simple version and a cinematic advanced… | [open](https://rishabh-bgp.github.io/zoo-website-capstone/) |
