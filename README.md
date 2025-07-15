
# Credit Card Fraud Detection using CNN

![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-TensorFlow%20%7C%20Scikit--Learn%20%7C%20Pandas-orange.svg)

A highly optimized implementation of a **Convolutional Neural Network (CNN)** for detecting fraudulent credit card transactions 🧠💳. This project leverages advanced data preprocessing and robust model architecture to achieve near state-of-the-art results ✅.

---

## 📚 Table of Contents

- [📌 Project Overview](#project-overview)
- [📂 Dataset Information](#dataset-information)
- [⚙️ Methodology & Implementation](#methodology--implementation)
  - [🔧 Data Preparation](#data-preparation)
  - [🏗️ Model Architecture (CNN)](#model-architecture-cnn-implementation)
  - [🎯 Training Strategy](#training-strategy)
- [📈 Results & Performance](#results--performance)
- [🚀 Getting Started (Setup & Installation)](#getting-started-setup--installation)
  - [📋 Prerequisites](#prerequisites)
  - [🔗 Installation](#installation)
  - [📓 Running the Notebook](#running-the-notebook)
- [🛠️ Usage](#usage)
- [🤝 Contributing](#contributing)

---

## 📌 Project Overview

Credit card fraud detection is a critical challenge due to **extreme class imbalance**. This project builds a **1D CNN-based deep learning model** capable of learning hidden patterns in transaction data to detect fraudulent activity with **exceptionally high precision and recall**.

✅ Uses strategic data balancing  
✅ Builds an optimized CNN pipeline  
✅ Delivers real-world production-level accuracy

---

## 📂 Dataset Information

- **📦 Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **🧮 Features**:
  - `V1-V28`: PCA-anonymized features
  - `Time`: Seconds since first transaction
  - `Amount`: Transaction amount
- **🎯 Target**: `Class` (0 = Legitimate, 1 = Fraudulent)
- **⚠️ Class Imbalance**: Only 0.172% fraudulent samples (492 out of 284,807)

---

## ⚙️ Methodology & Implementation

A structured approach from raw data to model evaluation 📊.

### 🔧 Data Preparation

- 🧱 **Library Setup & Reproducibility**: TensorFlow, Sklearn, Pandas, Random Seeds
- ⚖️ **Balanced Sampling**: 10:1 ratio of non-fraud to fraud to retain contextual information
- ✂️ **Train-Test Split**: 80/20 with stratification to preserve class ratio
- 🧼 **Standardization**: `StandardScaler` applied only to training set
- 📐 **Reshaping**: Converted to `(samples, timesteps, features)` for CNN

### 🏗️ Model Architecture (CNN Implementation)

A deep and regularized 1D CNN model:

- 🧠 `Conv1D` Layers: 64 and 128 filters to learn complex features
- 🔄 `BatchNormalization`: Faster & more stable training
- 🚫 `Dropout`: Prevents overfitting
- 🔁 `Flatten` + `Dense`: Final prediction via sigmoid activation

> **Optimizer**: Adam  
> **Loss Function**: Binary Crossentropy

### 🎯 Training Strategy

- 📉 `ReduceLROnPlateau`: Dynamic learning rate tuning
- 🛑 `EarlyStopping`: Stops training at best validation performance
- 🗓️ Trained for up to **20 epochs** with **batch size = 256**

---

## 📈 Results & Performance

Model was evaluated on a held-out test set for real-world effectiveness 🔍

### 📊 Final Evaluation Metrics

```text
Final Model Accuracy: 0.9981

Classification Report:
              precision    recall  f1-score   support
    Non-Fraud (0)   1.00      1.00      1.00       984
        Fraud (1)   0.99      0.90      0.94        99

    Accuracy                           1.00      1083
    Macro avg       0.99      0.95      0.97      1083
 Weighted avg       1.00      1.00      1.00      1083
```

### 📌 Machine-Readable Metrics

```python
precision: [0.9989, 0.9888]
recall:    [0.9989, 0.8990]
fscore:    [0.9989, 0.9418]
support:   [984, 99]
```

---

## 🚀 Getting Started (Setup & Installation)

Get the project up and running locally ⚙️

### 📋 Prerequisites

- Python 3.9+
- pip (package installer)

### 🔗 Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

# 2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt
# OR manually
pip install tensorflow pandas scikit-learn seaborn matplotlib jupyterlab
```

### 📓 Running the Notebook

```bash
jupyter lab
```

Then open `Credit_Card_Fraud_Detection_CNN.ipynb` and run all cells sequentially 🔁

---

## 🛠️ Usage

This model can serve as a **production-ready baseline** for fraud detection in banking/finance. You can:

- Export & deploy the model (`.h5` or `.pb`)
- Integrate into APIs to score real-time transactions
- Adapt for any binary classification with imbalanced data 💡

---

## 🤝 Contributing

🙌 Contributions are welcome!

1. Fork the project  
2. Create your feature branch: `git checkout -b feature/AmazingFeature`  
3. Commit changes: `git commit -m 'Add some AmazingFeature'`  
4. Push: `git push origin feature/AmazingFeature`  
5. Open a Pull Request 🛠️

---

> *Built with ❤️ for deep learning and making systems safer.*
