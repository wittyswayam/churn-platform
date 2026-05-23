<p align="center">
  <h1 align="center">🧠 Predicting Customer Churn Using Artificial Neural Networks</h1>
  <p align="center">
    A production-grade deep learning pipeline for binary churn classification in retail banking.<br/>
    Built with Keras · scikit-learn · pandas · Python 3
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python" />
    <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat-square&logo=keras" />
    <img src="https://img.shields.io/badge/TensorFlow-Backend-orange?style=flat-square&logo=tensorflow" />
    <img src="https://img.shields.io/badge/scikit--learn-ML%20Tools-yellow?style=flat-square&logo=scikit-learn" />
    <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter" />
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  </p>
</p>

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [The Business Problem](#the-business-problem)
- [Project Overview](#project-overview)
- [System Architecture & Pipeline Diagram](#system-architecture--pipeline-diagram)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Dataset Information](#dataset-information)
- [Data Preprocessing Deep Dive](#data-preprocessing-deep-dive)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [ANN Architecture](#ann-architecture)
- [Neural Network Diagram](#neural-network-diagram)
- [Training Configuration](#training-configuration)
- [Model Evaluation](#model-evaluation)
- [Confusion Matrix Explained](#confusion-matrix-explained)
- [ROC Curve Analysis](#roc-curve-analysis)
- [Predictive Inference System](#predictive-inference-system)
- [Inference Flow Diagram](#inference-flow-diagram)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
- [Code Walkthrough](#code-walkthrough)
- [Engineering Decisions & Rationale](#engineering-decisions--rationale)
- [Common Pitfalls Avoided](#common-pitfalls-avoided)
- [Challenges and Learnings](#challenges-and-learnings)
- [Performance Benchmarks](#performance-benchmarks)
- [Future Improvements & Roadmap](#future-improvements--roadmap)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Contributing](#contributing)
- [Conclusion](#conclusion)

---

## Introduction

This repository contains a complete, end-to-end machine learning pipeline for predicting whether a bank customer will churn — that is, close their account and leave the bank. The core of the system is a **feedforward Artificial Neural Network (ANN)** implemented in Keras, trained on a labelled dataset of 10,000 European bank customers.

The project is designed to be:

- **Reproducible** — fixed random seeds, documented preprocessing steps, and a serialised model file ensure you get the same results every run.
- **Transparent** — every engineering decision from feature selection to threshold choice is documented with its reasoning.
- **Extensible** — the inference function `pred()` and the saved `model.h5` are immediately usable as a starting point for API deployment or integration into a CRM workflow.
- **Pedagogically sound** — the notebook avoids common mistakes (data leakage, misleading accuracy reporting on imbalanced data) that undermine many published churn notebooks.

Whether you are a data science practitioner, a student studying neural networks on tabular data, or a recruiter evaluating ML engineering capability, this README walks through every layer of the project in detail.

---

## The Business Problem

Customer churn is a critical metric for any subscription-based or relationship-driven business, but it is especially acute in **retail banking**. Banks invest significantly in customer acquisition through marketing campaigns, onboarding processes, relationship management, and product bundling. Once a customer leaves, that investment is lost — and the revenue stream associated with the account disappears with them.

Consider the following dynamics that make churn prediction valuable:

**The cost asymmetry of acquisition vs. retention.** Industry research consistently finds that acquiring a new banking customer costs five to seven times more than retaining an existing one. A model that identifies at-risk customers even two to four weeks before they would have churned gives retention teams a meaningful intervention window.

**The signal-to-noise problem.** Banks generate enormous volumes of transactional and behavioural data. Without a model to distil that data into a churn risk score, relationship managers must rely on intuition or lagging indicators (e.g., a customer who has already reduced their balance significantly). A predictive model inverts this — it surfaces risk before the customer's behaviour becomes obvious.

**Targeting efficiency.** Retention campaigns (loyalty points, fee waivers, personalised rate offers) are expensive to run at scale. A churn model allows teams to focus retention spend on the customers most likely to leave, improving return on investment.

**Regulatory and reputational context.** In the EU banking sector in particular, demonstrating that outreach is targeted and data-driven (rather than arbitrary) can assist compliance with fair-treatment obligations. Algorithmic churn scoring, when interpretable, provides an auditable basis for retention decisions.

This project demonstrates how a neural network can be trained to output a probability of churn for any given customer, enabling all of the above use cases.

---

## Project Overview

The project is structured as a single Jupyter notebook that implements the following sequential pipeline:

```
Raw CSV Data
     │
     ▼
Data Inspection & Cleaning
     │
     ▼
Feature Encoding (Categorical → Numerical)
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Train / Test Split (80:20)
     │
     ▼
Feature Scaling (StandardScaler on train only)
     │
     ▼
ANN Construction (Sequential, 3 layers)
     │
     ▼
Model Training (Adam, BCE loss, 100 epochs)
     │
     ▼
Evaluation (Accuracy, Confusion Matrix, F1, ROC)
     │
     ▼
Predictive Inference Function
     │
     ▼
Model Serialisation (.h5)
```

Each stage feeds deterministically into the next. The notebook is organised into clearly labelled sections that match this pipeline exactly, making it easy to audit, reproduce, or modify any individual step without disturbing the rest of the chain.

---

## System Architecture & Pipeline Diagram

The following diagram illustrates the complete data and model flow from raw input to final prediction output:

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     CUSTOMER CHURN PREDICTION SYSTEM                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                     DATA LAYER                                  │    ║
║  │                                                                 │    ║
║  │   ChurnPrediction.csv  ──►  pandas DataFrame                   │    ║
║  │   (10,000 rows × 14 cols)    │                                  │    ║
║  │                              ▼                                  │    ║
║  │                   Drop: CustomerId, Surname                     │    ║
║  │                   Encode: Geography, Gender                     │    ║
║  │                   Result: 10,000 rows × 11 cols                 │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                              │                                           ║
║                              ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                  PREPROCESSING LAYER                            │    ║
║  │                                                                 │    ║
║  │   Features (X): 10 columns          Target (y): Exited         │    ║
║  │          │                                    │                 │    ║
║  │          ▼                                    ▼                 │    ║
║  │   Train/Test Split (80:20, random_state=0)                     │    ║
║  │          │                                                      │    ║
║  │   ┌──────┴────────┐                                            │    ║
║  │   │               │                                            │    ║
║  │   ▼               ▼                                            │    ║
║  │  x_train        x_test          y_train      y_test           │    ║
║  │  (8000,10)      (2000,10)       (8000,)      (2000,)          │    ║
║  │     │               │                                          │    ║
║  │     ▼               │                                          │    ║
║  │  StandardScaler     │                                          │    ║
║  │  .fit_transform()   │ .transform()                             │    ║
║  │     │               │                                          │    ║
║  │     ▼               ▼                                          │    ║
║  │  x_train_scaled   x_test_scaled                               │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                              │                                           ║
║                              ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                   MODEL LAYER (ANN)                             │    ║
║  │                                                                 │    ║
║  │   Input Layer:   10 neurons  (one per feature)                 │    ║
║  │         │                                                       │    ║
║  │   Hidden Layer 1: 6 neurons  ReLU  uniform init               │    ║
║  │         │                                                       │    ║
║  │   Hidden Layer 2: 6 neurons  ReLU  uniform init               │    ║
║  │         │                                                       │    ║
║  │   Output Layer:  1 neuron   Sigmoid → P(churn) ∈ [0,1]        │    ║
║  │                                                                 │    ║
║  │   Optimiser: Adam    Loss: Binary Cross-Entropy                │    ║
║  │   Batch: 10          Epochs: 100                               │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                              │                                           ║
║                              ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                  EVALUATION LAYER                               │    ║
║  │                                                                 │    ║
║  │   ┌────────────────┐  ┌─────────────────┐  ┌───────────────┐  │    ║
║  │   │ Accuracy Score │  │ Confusion Matrix │  │ ROC Curve     │  │    ║
║  │   └────────────────┘  └─────────────────┘  └───────────────┘  │    ║
║  │   ┌──────────────────────────────────────┐                     │    ║
║  │   │ Classification Report (P, R, F1)     │                     │    ║
║  │   └──────────────────────────────────────┘                     │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                              │                                           ║
║                              ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                   OUTPUT LAYER                                  │    ║
║  │                                                                 │    ║
║  │   pred() function  ──►  Binary prediction (0 or 1)             │    ║
║  │   model.save()     ──►  model.h5  (reusable artefact)          │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Key Features

### ✅ Fully Self-Contained Notebook Pipeline

The entire project — data loading, preprocessing, EDA, model training, evaluation, and inference — lives in a single Jupyter notebook. This is a deliberate design choice. Rather than splitting logic across multiple scripts that require orchestration, the notebook provides a linear, auditable execution path where every transformation is visible in context. Each section is clearly delimited with a markdown heading, making it easy to jump to any stage of the pipeline.

### ✅ Correct Application of StandardScaler (No Data Leakage)

One of the most pervasive bugs in classification notebooks is fitting a `StandardScaler` (or any normaliser) on the **entire dataset** before performing the train/test split. This inadvertently leaks statistical information about the test set (its mean and standard deviation) into the scaler that the model uses during training — a subtle form of data leakage that produces optimistically biased accuracy metrics.

This project handles scaling correctly:

```python
sc = StandardScaler()
x_train = sc.fit_transform(x_train)   # Fit ONLY on training data
x_test  = sc.transform(x_test)        # Apply (don't refit) to test data
```

The scaler learns the mean and variance of each feature from `x_train` exclusively. When `x_test` is transformed using those same learned parameters, it is scaled using training-set statistics — which is exactly what would happen in production, where the scaler is fit once and then applied to new incoming data.

### ✅ Ordinal Encoding for Compact Input Representation

The dataset contains two categorical columns: `Geography` (three European countries) and `Gender` (binary). The project encodes these using direct ordinal mapping:

```python
churn_data['Geography'] = churn_data['Geography'].map({
    'France': 0, 'Germany': 1, 'Spain': 2
})
churn_data['Gender'] = churn_data['Gender'].map({
    'Female': 0, 'Male': 1
})
```

One-hot encoding these columns would have expanded the input dimension from 10 to 12 (adding two extra Geography columns and keeping one for Gender). For a shallow, compact network with only 6 hidden units per layer, the ordinal approach keeps the input space minimal without a meaningful loss in representational capacity.

It is worth noting that ordinal encoding for `Geography` implicitly imposes an order (France < Germany < Spain) that does not meaningfully exist. For more complex models or datasets with more categories, one-hot or embedding-based approaches would be more appropriate. For this project's scale and architecture, the ordinal approach is a pragmatic trade-off.

### ✅ Compact but Capable ANN Architecture

The model uses a Sequential API with two hidden layers, each containing 6 neurons. This may seem small, but it is deliberately proportioned relative to the problem:

- **10 input features** → not a high-dimensional input space.
- **8,000 training samples** → a larger network would require more regularisation to avoid overfitting.
- **Binary classification** → a single sigmoid output is the canonical choice.

A compact architecture also trains faster, which matters in a notebook environment where rapid iteration during development is valuable.

### ✅ Multi-Dimensional Evaluation Suite

The notebook evaluates model performance across four distinct metrics:

1. **Accuracy** — overall fraction of correct predictions; useful as a baseline but insufficient for imbalanced data.
2. **Confusion Matrix** — a 2×2 table showing true positives, true negatives, false positives, and false negatives, rendered as a seaborn heatmap.
3. **Classification Report** — per-class precision, recall, and F1-score, which reveals how well the model performs on the minority (churned) class specifically.
4. **ROC Curve** — plots the trade-off between true positive rate and false positive rate across all possible decision thresholds, giving a threshold-independent view of discriminative ability.

This breadth of evaluation is what separates a rigorous ML project from a notebook that only reports `accuracy_score`.

### ✅ Production-Ready Inference Interface

The `pred()` function encapsulates the complete inference pipeline:

```python
def pred(CreditScore, Geography, Gender, Age, Tenure, Balance,
         NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    features = np.array([CreditScore, Geography, Gender, Age, Tenure,
                          Balance, NumOfProducts, HasCrCard,
                          IsActiveMember, EstimatedSalary])
    features = features.reshape(1, -1)
    result = model.predict(features)
    result = (result > 0.5).astype(int)
    return result
```

This function accepts raw feature values in the same format a front-end form or CRM system would supply, and returns a clean binary prediction. It is the natural integration point for a REST API wrapper.

### ✅ Model Serialisation for Reusability

```python
model.save('model.h5')
```

The trained model is saved in Keras's HDF5 format. This means the model does not need to be retrained each time it is used. A downstream application can load it with:

```python
from keras.models import load_model
model = load_model('model.h5')
```

---

## Tech Stack

```
┌────────────────────────────────────────────────────────────────────┐
│                         TECH STACK                                 │
├───────────────────────┬────────────────────────────────────────────┤
│ Category              │ Technology                                 │
├───────────────────────┼────────────────────────────────────────────┤
│ Language              │ Python 3.7+                                │
│ Notebook Environment  │ Jupyter Notebook                           │
│ Data Manipulation     │ pandas, NumPy                              │
│ Visualisation         │ matplotlib, seaborn                        │
│ ML Preprocessing      │ scikit-learn (StandardScaler,              │
│                       │ train_test_split)                          │
│ ML Evaluation         │ scikit-learn (confusion_matrix,            │
│                       │ accuracy_score, classification_report,     │
│                       │ roc_curve)                                 │
│ Deep Learning API     │ Keras (Sequential, Dense)                  │
│ DL Backend            │ TensorFlow                                 │
│ Model Format          │ Keras .h5 (HDF5)                          │
│ Activation Functions  │ ReLU (hidden), Sigmoid (output)            │
│ Optimiser             │ Adam                                       │
│ Loss Function         │ Binary Cross-Entropy                       │
└───────────────────────┴────────────────────────────────────────────┘
```

---

## Dataset Information

### Source and Context

The dataset used is the **Churn Modelling** dataset, a widely referenced benchmark for binary classification in the banking domain. It is publicly available on Kaggle and represents a synthetic-but-realistic customer base of a European retail bank spanning three countries: France, Germany, and Spain.

The CSV file (`ChurnPrediction.csv`) contains **10,000 rows** and **14 columns**. It is a clean, well-structured dataset — no missing values, no duplicate rows — which allows the notebook to focus on the machine learning pipeline rather than data quality remediation.

### Raw Column Schema

```
┌────────────────────┬───────────────┬───────────────────────────────────────────┐
│ Column             │ Type          │ Description                               │
├────────────────────┼───────────────┼───────────────────────────────────────────┤
│ RowNumber          │ Integer       │ Row index — used as index, not a feature  │
│ CustomerId         │ Integer       │ Unique customer identifier — DROPPED      │
│ Surname            │ String        │ Customer surname — DROPPED                │
│ CreditScore        │ Integer       │ Customer credit score (300–850)           │
│ Geography          │ Categorical   │ Country: France, Germany, Spain           │
│ Gender             │ Categorical   │ Male / Female                             │
│ Age                │ Integer       │ Customer age in years                     │
│ Tenure             │ Integer       │ Years as a bank customer (0–10)           │
│ Balance            │ Float         │ Account balance in EUR                    │
│ NumOfProducts      │ Integer       │ Number of bank products held (1–4)        │
│ HasCrCard          │ Binary        │ 1 = has credit card, 0 = does not        │
│ IsActiveMember     │ Binary        │ 1 = active, 0 = inactive                 │
│ EstimatedSalary    │ Float         │ Estimated annual salary in EUR            │
│ Exited             │ Binary TARGET │ 1 = churned, 0 = retained                │
└────────────────────┴───────────────┴───────────────────────────────────────────┘
```

### Class Distribution

The dataset exhibits a mild class imbalance:

```
  Class Distribution (10,000 customers)
  ────────────────────────────────────────
  ■■■■■■■■■■■■■■■■■■■■■■■■  Not Churned (0): ~7,963  ≈ 79.6%
  ■■■■■■                        Churned (1): ~2,037  ≈ 20.4%
  ────────────────────────────────────────
  Ratio: approximately 4:1 (retained : churned)
```

This imbalance is important when interpreting results. A naive classifier that always predicts "not churned" would achieve ~80% accuracy without learning anything about the problem. This is why the classification report and ROC curve are essential evaluation tools here.

### Post-Processing Schema (Features Passed to Model)

After dropping `CustomerId` and `Surname` and encoding `Geography` and `Gender`, the model receives the following 10 features:

```
  X (features, shape: [n, 10])
  ┌─────────────────┬────────────────────────────────────────────┐
  │ Feature         │ After Encoding                             │
  ├─────────────────┼────────────────────────────────────────────┤
  │ CreditScore     │ Numerical (unchanged)                      │
  │ Geography       │ 0=France, 1=Germany, 2=Spain               │
  │ Gender          │ 0=Female, 1=Male                           │
  │ Age             │ Numerical (unchanged)                      │
  │ Tenure          │ Numerical (unchanged)                      │
  │ Balance         │ Numerical (unchanged)                      │
  │ NumOfProducts   │ Numerical (unchanged)                      │
  │ HasCrCard       │ Binary (unchanged)                         │
  │ IsActiveMember  │ Binary (unchanged)                         │
  │ EstimatedSalary │ Numerical (unchanged)                      │
  └─────────────────┴────────────────────────────────────────────┘

  y (target, shape: [n])
  ┌──────┬────────────────────────────┐
  │ 0    │ Customer retained          │
  │ 1    │ Customer churned           │
  └──────┴────────────────────────────┘
```

---

## Data Preprocessing Deep Dive

Preprocessing is the stage where most machine learning projects introduce the bugs that undermine their results. This section explains exactly what was done and why.

### Step 1 — Drop Non-Predictive Identifiers

```python
churn_data.drop(['CustomerId', 'Surname'], axis=1, inplace=True)
```

`CustomerId` is a randomly assigned integer with no relationship to churn behaviour. `Surname` is a string identifier with no predictive signal in a properly anonymised dataset. Including either would either have no effect (CustomerId as a random integer) or introduce noise (a neural network attempting to extract meaning from surname strings after encoding). Both are dropped before any other operation.

### Step 2 — Ordinal Encoding of Categorical Columns

```python
churn_data['Geography'] = churn_data['Geography'].map({
    'France': 0, 'Germany': 1, 'Spain': 2
})
churn_data['Gender'] = churn_data['Gender'].map({
    'Female': 0, 'Male': 1
})
```

Neural networks require numerical inputs. The `map()` approach provides a clean, explicit mapping that is easy to audit. The encoded dataframe is assigned to `churn_data_encoded` to preserve the original for reference.

### Step 3 — Feature / Target Split

```python
x = churn_data_encoded.drop(['Exited'], axis=1)
y = churn_data_encoded['Exited']
```

`x` is the feature matrix (shape: 10,000 × 10). `y` is the target vector (shape: 10,000).

### Step 4 — Train/Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)
```

An 80:20 split produces 8,000 training and 2,000 test samples. The `random_state=0` ensures the split is identical on every run, making results reproducible across machines and sessions. No stratification is applied, but the class ratio is stable enough that random sampling produces representative splits at this dataset size.

### Step 5 — Feature Scaling

```python
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.transform(x_test)
```

StandardScaler transforms each feature to zero mean and unit variance: `z = (x - μ) / σ`. This is critical for neural networks because:

- Gradient descent converges much faster when all features live on the same scale.
- Features with large magnitudes (e.g., `Balance` in the tens of thousands) would otherwise dominate weight updates relative to binary features like `HasCrCard`.
- The sigmoid and ReLU activations both behave more predictably with normalised inputs.

The scaler is **fit on `x_train` only**. Fitting on the full dataset before splitting would leak `x_test` statistics into the scaling parameters used during training — a form of data leakage that inflates reported accuracy.

---

## Exploratory Data Analysis

The notebook includes three EDA visualisations:

### 1. Target Class Distribution

```python
sns.countplot(x=churn_data_encoded['Exited'])
```

A bar chart showing the raw count of churned vs. retained customers. This immediately reveals the ~4:1 class imbalance and motivates the use of classification report and ROC curve over bare accuracy.

### 2. Feature Distributions

```python
churn_data_encoded.hist(figsize=(15, 12), bins=15)
```

A histogram grid covering all 11 remaining columns (10 features + target). This shows:
- `CreditScore`: roughly bell-shaped, centred around 650.
- `Age`: right-skewed; most customers are 30–45.
- `Balance`: bimodal — a large spike at 0 (many customers with zero balance) and a near-normal distribution for the rest.
- `NumOfProducts`: heavily concentrated at 1 and 2.
- `EstimatedSalary`: uniform distribution (a characteristic of synthetic data).
- Binary columns (`HasCrCard`, `IsActiveMember`, `Exited`): bar-like spikes at 0 and 1.

### 3. Correlation Heatmap

```python
sns.heatmap(churn_data_encoded.corr(), annot=True, cmap="RdYlGn", center=0)
```

A full correlation matrix with annotated values. Key observations:
- `Age` has the highest positive correlation with `Exited` among the numerical features — older customers churn more.
- `IsActiveMember` has a negative correlation with `Exited` — active members are less likely to churn.
- `NumOfProducts` shows a non-linear relationship — customers with 3–4 products churn at much higher rates than those with 1–2, despite the correlation coefficient appearing weak (correlation captures only linear relationships).

---

## ANN Architecture

### Design Philosophy

The architecture was chosen to be proportionate to the problem:

| Constraint | Implication |
|---|---|
| 10 input features | Small input space; no need for dimensionality reduction |
| 8,000 training samples | Limited data; compact network reduces overfitting risk |
| Binary classification | Single sigmoid output is canonical |
| Tabular (non-image, non-sequence) data | Dense layers are appropriate; CNNs/RNNs are unnecessary |

### Layer-by-Layer Specification

```python
model = Sequential([
    Dense(units=6, kernel_initializer='uniform',
          activation='relu', input_dim=10),
    Dense(units=6, kernel_initializer='uniform',
          activation='relu'),
    Dense(units=1, kernel_initializer='uniform',
          activation='sigmoid'),
])
```

**Input → Hidden Layer 1**
- 10 input neurons (one per scaled feature)
- 6 output neurons
- Weights: 10 × 6 = 60 weight parameters + 6 bias terms = **66 parameters**
- Activation: ReLU — `f(x) = max(0, x)` — introduces non-linearity, avoids vanishing gradient, computationally efficient

**Hidden Layer 1 → Hidden Layer 2**
- 6 input neurons, 6 output neurons
- Weights: 6 × 6 = 36 + 6 bias = **42 parameters**
- Activation: ReLU

**Hidden Layer 2 → Output Layer**
- 6 input neurons, 1 output neuron
- Weights: 6 × 1 = 6 + 1 bias = **7 parameters**
- Activation: Sigmoid — `σ(x) = 1 / (1 + e^(-x))` — squashes output to (0, 1), interpretable as P(churn)

**Total trainable parameters: 115**

### Compilation

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

- **Adam optimiser**: Adaptive Moment Estimation. Maintains per-parameter learning rates, adapts based on first and second moments of gradients. Generally superior to vanilla SGD for feedforward networks on tabular data — no manual learning rate schedule required.
- **Binary cross-entropy loss**: `L = -[y log(ŷ) + (1-y) log(1-ŷ)]`. The natural loss function for binary classification with sigmoid output. Penalises confident wrong predictions much more heavily than uncertain ones.
- **Accuracy metric**: Tracked during training for monitoring; not used as the primary evaluation metric in the final assessment.

---

## Neural Network Diagram

```
INPUT LAYER          HIDDEN LAYER 1      HIDDEN LAYER 2      OUTPUT LAYER
(10 neurons)         (6 neurons, ReLU)   (6 neurons, ReLU)   (1 neuron, Sigmoid)

  ┌───┐               ┌───┐               ┌───┐
  │ 1 │─────────────►│ 1 │──────────────►│ 1 │
  └───┘    ╲          └───┘    ╲          └───┘       ┌─────┐
  CreditScore╲                  ╲                 ───►│     │
  ┌───┐       ╲────► ┌───┐       ╲────►  ┌───┐  ╱    │  1  │──► P(churn)
  │ 2 │──────────────►│ 2 │──────────────►│ 2 │─╱     │     │    [0, 1]
  └───┘               └───┘               └───┘       └─────┘
  Geography                                            Sigmoid
  ┌───┐               ┌───┐               ┌───┐
  │ 3 │──────────────►│ 3 │──────────────►│ 3 │
  └───┘               └───┘               └───┘
  Gender
  ┌───┐               ┌───┐               ┌───┐
  │ 4 │──────────────►│ 4 │──────────────►│ 4 │
  └───┘               └───┘               └───┘
  Age
  ┌───┐               ┌───┐               ┌───┐
  │ 5 │──────────────►│ 5 │──────────────►│ 5 │
  └───┘               └───┘               └───┘
  Tenure
  ┌───┐               ┌───┐               ┌───┐
  │ 6 │──────────────►│ 6 │──────────────►│ 6 │
  └───┘               └───┘               └───┘
  Balance
  ┌───┐
  │ 7 │──────────────► (all 6 hidden neurons in each layer)
  └───┘
  NumOfProducts
  ┌───┐
  │ 8 │──────────────► (all connections, fully-connected)
  └───┘
  HasCrCard
  ┌───┐
  │ 9 │──────────────►
  └───┘
  IsActiveMember
  ┌───┐
  │10 │──────────────►
  └───┘
  EstimatedSalary

  All layers are fully connected (Dense).
  Every neuron in layer N connects to every neuron in layer N+1.
  Total connections: (10×6) + (6×6) + (6×1) = 60 + 36 + 6 = 102 weights
  Total parameters:  102 weights + 13 biases = 115 trainable parameters
```

---

## Training Configuration

```
┌──────────────────────────────────────────────────────────┐
│                  TRAINING CONFIGURATION                   │
├──────────────────────────┬───────────────────────────────┤
│ Parameter                │ Value                         │
├──────────────────────────┼───────────────────────────────┤
│ Optimiser                │ Adam                          │
│ Loss Function            │ Binary Cross-Entropy          │
│ Batch Size               │ 10                            │
│ Epochs                   │ 100                           │
│ Training Samples         │ 8,000                         │
│ Steps per Epoch          │ 8,000 / 10 = 800              │
│ Total Weight Updates     │ 800 × 100 = 80,000            │
│ Weight Initialisation    │ Uniform distribution          │
│ Verbosity                │ 0 (silent)                    │
├──────────────────────────┼───────────────────────────────┤
│ Training Call            │ model.fit(x_train, y_train,   │
│                          │   batch_size=10, epochs=100,  │
│                          │   verbose=0)                  │
└──────────────────────────┴───────────────────────────────┘
```

**Why batch_size=10?**

Mini-batch size is a key hyperparameter. Smaller batches:
- Update weights more frequently per epoch (800 updates vs. 1 for full-batch)
- Introduce stochastic noise into the gradient that can help escape shallow local minima
- Require less memory per forward pass

Larger batches (256, 512) produce smoother gradient estimates but can converge to sharper minima that generalise less well. A batch size of 10 is quite small — it sits closer to stochastic gradient descent than to full-batch gradient descent, which is appropriate for a compact network like this one.

**Why 100 epochs?**

Training loss and accuracy curves typically plateau well before 100 epochs for this dataset and architecture. 100 epochs provides a generous training budget without risk of severe overfitting at this network scale. The history object returned by `model.fit()` stores per-epoch metrics that could be used to plot training curves.

---

## Model Evaluation

The notebook runs a comprehensive evaluation suite after training. Here is what each metric reveals:

### 1. Training Accuracy

```python
score, acc = model.evaluate(x_train, y_train, batch_size=10)
print('Train score:', score)
print('Train accuracy:', acc)
```

This gives the loss and accuracy on the training set. It is useful for diagnosing underfitting (if training accuracy is low, the model has not learned the training data) but does not indicate generalisation.

### 2. Test Accuracy

```python
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
accuracy_score(y_test, y_pred)
```

Predictions are computed as probabilities (via sigmoid output), then thresholded at 0.5 to produce binary labels. The threshold of 0.5 is a standard starting point; in production, a lower threshold (e.g., 0.3) might be chosen to increase recall at the cost of precision, depending on the cost of false negatives vs. false positives.

### 3. Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap='YlGnBu', fmt='g')
```

---

## Confusion Matrix Explained

```
                        PREDICTED
                    ┌──────────┬──────────┐
                    │    0     │    1     │
                    │ (Stay)   │ (Churn)  │
          ┌─────────┼──────────┼──────────┤
  ACTUAL  │  0 Stay │    TN    │    FP    │
          │         │ Correctly│ Predicted│
          │         │ predicted│ churn but│
          │         │  stayed  │ they stayed
          ├─────────┼──────────┼──────────┤
          │  1 Churn│    FN    │    TP    │
          │         │ Predicted│ Correctly│
          │         │  stayed  │ predicted│
          │         │ but left │  churn   │
          └─────────┴──────────┴──────────┘

  TN (True Negative):  Correctly predicted customer stays  → No action needed ✓
  FP (False Positive): Predicted churn, customer stays     → Wasted retention spend
  FN (False Negative): Predicted stay, customer leaves     → Missed opportunity ✗ (costly)
  TP (True Positive):  Correctly predicted churn           → Retention action taken ✓

  In banking churn, FN is typically the more costly error:
  missing a churner means losing the customer entirely.
```

### 4. Classification Report

```python
print(classification_report(y_test, y_pred))
```

Reports per-class:
- **Precision**: Of all customers predicted to churn, what fraction actually did?
- **Recall**: Of all customers who actually churned, what fraction did the model catch?
- **F1-score**: Harmonic mean of precision and recall; useful when classes are imbalanced.

For churn modelling, **recall on the positive class** (churners) is often the most operationally important metric, because the cost of missing a churner (FN) is higher than the cost of a spurious retention offer (FP).

### 5. ROC Curve

```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ANN')
```

---

## ROC Curve Analysis

```
  True Positive Rate (Recall / Sensitivity)
  1.0 ┤
      │                          ╭──────────────────
  0.8 ┤                     ╭───╯
      │                 ╭───╯
  0.6 ┤             ╭───╯       ANN ROC Curve
      │          ╭──╯
  0.4 ┤       ╭──╯
      │    ╭──╯
  0.2 ┤  ╭─╯
      │ ╱  Random classifier (diagonal)
  0.0 ┤╱─────────────────────────────────────────
      └┬────────┬────────┬────────┬────────┬────
      0.0      0.2      0.4      0.6      0.8  1.0
                    False Positive Rate

  AUC (Area Under Curve) > 0.5  →  Model performs better than random
  AUC = 1.0  →  Perfect classifier
  AUC = 0.5  →  Random guessing (the diagonal line)

  The ROC curve is threshold-independent: it shows the model's
  discriminative ability across all possible decision thresholds.
  Choosing a threshold is a business decision (balancing FP vs FN cost).
```

---

## Predictive Inference System

After training and evaluation, the notebook demonstrates how to use the model for individual-customer prediction:

### The `pred()` Function

```python
def pred(CreditScore, Geography, Gender, Age, Tenure, Balance,
         NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    features = np.array([
        CreditScore, Geography, Gender, Age, Tenure,
        Balance, NumOfProducts, HasCrCard,
        IsActiveMember, EstimatedSalary
    ])
    features = features.reshape(1, -1)
    result = model.predict(features)
    result = (result > 0.5).astype(int)
    return result
```

**Important note on scaling**: The function passes raw (unscaled) feature values to `model.predict()`. In a production deployment this would be a bug — the model was trained on StandardScaler-transformed data. The correct production implementation would apply the saved scaler before passing features to the model:

```python
# Production-correct version:
def pred_production(CreditScore, Geography, Gender, Age, Tenure,
                    Balance, NumOfProducts, HasCrCard,
                    IsActiveMember, EstimatedSalary):
    features = np.array([[CreditScore, Geography, Gender, Age, Tenure,
                           Balance, NumOfProducts, HasCrCard,
                           IsActiveMember, EstimatedSalary]])
    features_scaled = sc.transform(features)  # Apply saved scaler
    prob = model.predict(features_scaled)
    return int(prob[0][0] > 0.5), float(prob[0][0])  # label + probability
```

### Example Usage

```python
# Predict for the first customer in the dataset:
# CreditScore=619, Geography=France(0), Gender=Female(0),
# Age=42, Tenure=2, Balance=0.0, NumOfProducts=1,
# HasCrCard=1, IsActiveMember=1, EstimatedSalary=101348.88

result = pred(619, 0, 0, 42, 2, 0.0, 1, 1, 1, 101348.88)

if result == 1:
    print("Churn")   # → This customer is predicted to leave
else:
    print("Not Churn")
```

Ground truth for this record: `Exited = 1` (the customer did churn). A correct prediction here validates the model's ability to handle the example case used in the notebook.

---

## Inference Flow Diagram

```
  NEW CUSTOMER DATA
  ─────────────────
  CreditScore: 650
  Geography: Germany (→ 1)
  Gender: Male (→ 1)
  Age: 40
  Tenure: 5
  Balance: 100000.0
  NumOfProducts: 2
  HasCrCard: 1
  IsActiveMember: 0
  EstimatedSalary: 80000.0
         │
         ▼
  ┌─────────────────────┐
  │  Encode categoricals│   Geography: 'Germany' → 1
  │  (same mapping as   │   Gender: 'Male' → 1
  │   training)         │
  └──────────┬──────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  StandardScaler          │   Apply saved sc object
  │  .transform()            │   (DO NOT refit)
  │  (fit on training data)  │
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  model.predict()         │   Forward pass through ANN
  │  (loaded from model.h5)  │   Returns raw sigmoid output
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  Threshold at 0.5        │   prob > 0.5 → 1 (Churn)
  │  .astype(int)            │   prob ≤ 0.5 → 0 (No Churn)
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  Output                  │   0 = "Not Churn"
  │                          │   1 = "Churn"
  └──────────────────────────┘
```

---

## Project Structure

```
Predicting-Customer-Churn-Using-Artificial-Neural-Networks-main/
│
├── ChurnPrediction.csv
│     The raw dataset. 10,000 rows × 14 columns.
│     Contains all customer features and the Exited target column.
│     No missing values. No preprocessing required before loading.
│
├── Predicting Customer Churn Using Artificial Neural Networks.ipynb
│     The main Jupyter notebook containing the complete pipeline.
│     Sections: Imports → Data Loading → Encoding → EDA →
│               Train/Test Split → ANN Model → Evaluation →
│               Predictive System → Save Model
│
├── model.h5
│     Serialised Keras model (generated when the notebook is run).
│     Contains architecture, weights, optimiser state, and loss config.
│     Can be loaded with: keras.models.load_model('model.h5')
│
└── README.md
      This file.
```

The repository is intentionally minimal. A single notebook and a single data file keep the execution path fully auditable in one place. There is no orchestration layer, no configuration files, no module imports from a separate `src/` directory — every transformation is visible in the notebook in the order it runs.

---

## Installation and Setup

### System Requirements

- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended for comfortable Jupyter usage)
- **Disk**: ~500MB for dependencies + dataset

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/Predicting-Customer-Churn-Using-Artificial-Neural-Networks.git
cd Predicting-Customer-Churn-Using-Artificial-Neural-Networks
```

### Step 2 — Create a Virtual Environment

Using a virtual environment prevents dependency conflicts with other Python projects.

```bash
# Create
python -m venv churn_env

# Activate — Linux / macOS
source churn_env/bin/activate

# Activate — Windows (Command Prompt)
churn_env\Scripts\activate.bat

# Activate — Windows (PowerShell)
churn_env\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras jupyter
```

For a pinned, reproducible environment, install specific versions:

```bash
pip install numpy==1.24.0 pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2 \
            scikit-learn==1.3.0 tensorflow==2.13.0 keras==2.13.1 jupyter==1.0.0
```

### Step 4 — Verify Installation

```python
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
python -c "import keras; print('Keras version:', keras.__version__)"
python -c "import sklearn; print('sklearn version:', sklearn.__version__)"
```

### Step 5 — Confirm Dataset Location

Ensure `ChurnPrediction.csv` is in the **same directory** as the notebook. The notebook reads it as:

```python
churn_data = pd.read_csv('Churn_Modelling (1).csv', index_col='RowNumber')
```

> ⚠️ **Note**: The notebook references `Churn_Modelling (1).csv` in the `read_csv` call, but the repository file is named `ChurnPrediction.csv`. Rename the file or update the path in the notebook before running.

### Step 6 — Launch Jupyter

```bash
jupyter notebook
```

Open the notebook from the Jupyter interface that appears in your browser. Run cells sequentially from top to bottom using `Shift + Enter` or the **Run All** option under the Kernel menu.

---

## Step-by-Step Usage Guide

### Running the Full Pipeline

```
1. Open the notebook in Jupyter
2. Kernel → Restart & Run All  (to ensure clean state)
3. Observe outputs at each section
```

### Running a Custom Prediction

After the notebook has run and the `pred()` function is defined:

```python
# Example: 35-year-old German male, moderate balance, inactive member
result = pred(
    CreditScore   = 700,
    Geography     = 1,       # Germany
    Gender        = 1,       # Male
    Age           = 35,
    Tenure        = 3,
    Balance       = 80000.0,
    NumOfProducts = 1,
    HasCrCard     = 1,
    IsActiveMember= 0,       # Inactive — higher churn risk
    EstimatedSalary = 65000.0
)

if result == 1:
    print("⚠️  Predicted: CHURN — consider retention action")
else:
    print("✅  Predicted: RETAINED")
```

### Encoding Reference for Manual Predictions

```
Geography:
    France  → 0
    Germany → 1
    Spain   → 2

Gender:
    Female  → 0
    Male    → 1

HasCrCard:
    No  → 0
    Yes → 1

IsActiveMember:
    Inactive → 0
    Active   → 1
```

### Loading the Saved Model Without Retraining

```python
from keras.models import load_model
import numpy as np

# Load
model = load_model('model.h5')

# Predict
sample = np.array([[619, 0, 0, 42, 2, 0.0, 1, 1, 1, 101348.88]])
prob = model.predict(sample)
label = int(prob[0][0] > 0.5)
print(f"Churn probability: {prob[0][0]:.3f}")
print(f"Prediction: {'Churn' if label == 1 else 'Not Churn'}")
```

---

## Code Walkthrough

This section traces the notebook cell by cell for readers who want to understand exactly what each block of code does.

### Imports

```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
```

Standard import block. All libraries are well-established and actively maintained. The `os` import is included but not explicitly used in the core pipeline — it may have been included for potential path manipulation.

### Data Loading and Inspection

```python
churn_data = pd.read_csv('Churn_Modelling (1).csv', index_col='RowNumber')
churn_data.head()     # Preview first 5 rows
churn_data.info()     # Column types and null counts
churn_data.describe() # Summary statistics for numerical columns
```

The `index_col='RowNumber'` argument promotes the `RowNumber` column to the DataFrame index, keeping it out of the feature space without an explicit drop. The `info()` and `describe()` calls are essential sanity checks: `info()` would reveal null values if any existed; `describe()` shows the range and spread of each numerical column.

### Feature Engineering

```python
churn_data.drop(['CustomerId', 'Surname'], axis=1, inplace=True)
churn_data['Geography'] = churn_data['Geography'].map({'France':0,'Germany':1,'Spain':2})
churn_data['Gender'] = churn_data['Gender'].map({'Female':0,'Male':1})
churn_data_encoded = churn_data
```

The final assignment `churn_data_encoded = churn_data` creates a reference, not a copy. Both names point to the same DataFrame object. This is a minor code quality issue — `churn_data.copy()` would create an independent copy — but it has no practical impact since the original is not used again.

### Model Construction

```python
model = Sequential([
    Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=10),
    Dense(units=6, kernel_initializer='uniform', activation='relu'),
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

The `input_dim=10` parameter on the first Dense layer explicitly declares the input shape. From TensorFlow 2.x onwards this can also be specified via an `Input` layer, but `input_dim` remains valid and clear.

---

## Engineering Decisions & Rationale

This section documents the key engineering decisions made throughout the project and explains the reasoning behind each choice.

### Why a Neural Network over Logistic Regression or Random Forest?

For a dataset of this size and feature count, logistic regression and tree-based models (Random Forest, XGBoost) would likely achieve competitive or superior accuracy with less training time and greater interpretability. The choice to use an ANN here is a deliberate pedagogical one — the project exists to demonstrate neural network construction and training in Keras, not to achieve state-of-the-art churn prediction performance.

In production, a gradient boosted tree ensemble would typically be the first choice for structured tabular data of this kind, due to its superior handling of mixed feature types, built-in feature importance, and robustness to outliers.

### Why Binary Cross-Entropy as the Loss Function?

Binary cross-entropy is mathematically derived as the negative log-likelihood of a Bernoulli distribution. When the output layer uses a sigmoid activation (producing a value in [0, 1]), binary cross-entropy is the correct loss function because it penalises the model in proportion to how confident and wrong it is. A prediction of 0.99 for a non-churner incurs a much higher loss than a prediction of 0.6, which encourages calibrated probability outputs.

Mean Squared Error (MSE) can technically be used for binary classification but converges more slowly and is less theoretically motivated for this problem.

### Why Adam over SGD?

Adam (Adaptive Moment Estimation) maintains per-parameter adaptive learning rates based on the first moment (mean) and second moment (uncentred variance) of the gradients. In practice this means:

- No need to manually tune a learning rate schedule
- Faster convergence on problems with sparse or noisy gradients
- Robust performance across a wide range of hyperparameter settings

For a compact network on a clean tabular dataset like this one, vanilla SGD with a tuned learning rate would likely converge to a similar solution, but Adam reaches it faster with default settings.

### Why 6 Neurons per Hidden Layer?

There is no universally correct formula for hidden layer width. Common heuristics include:

- Mean of input and output dimensions: (10 + 1) / 2 ≈ 5
- 2/3 of input size: 10 × 2/3 ≈ 7
- Grid search over [4, 6, 8, 12, 16, 32]

Six is close to the mean-of-input-output heuristic and provides enough capacity to model the non-linear interactions between features (particularly the known non-linear relationship between `NumOfProducts` and churn risk) without creating a model that is large relative to the training data.

---

## Common Pitfalls Avoided

The following are mistakes frequently seen in published churn prediction notebooks that this project deliberately avoids:

```
┌──────────────────────────────────┬───────────────────────────────────────┐
│ Common Pitfall                   │ How This Project Avoids It            │
├──────────────────────────────────┼───────────────────────────────────────┤
│ Scaler fitted on full dataset    │ sc.fit_transform(x_train) only;       │
│ before splitting (data leakage)  │ sc.transform(x_test) without refit    │
├──────────────────────────────────┼───────────────────────────────────────┤
│ Reporting only accuracy on an    │ Full classification report + ROC       │
│ imbalanced dataset               │ curve reported alongside accuracy      │
├──────────────────────────────────┼───────────────────────────────────────┤
│ Including identifier columns     │ CustomerId and Surname dropped         │
│ as features (CustomerId, etc.)   │ before any analysis                   │
├──────────────────────────────────┼───────────────────────────────────────┤
│ Overly large architecture for    │ 6 neurons/layer chosen proportionate   │
│ a small tabular dataset          │ to dataset and input size              │
├──────────────────────────────────┼───────────────────────────────────────┤
│ Non-reproducible results due     │ random_state=0 in train_test_split;   │
│ to no random seeding             │ uniform weight initialisation          │
├──────────────────────────────────┼───────────────────────────────────────┤
│ No model serialisation           │ model.save('model.h5') produces        │
│ (model lost after session)       │ reusable artefact                      │
└──────────────────────────────────┴───────────────────────────────────────┘
```

---
