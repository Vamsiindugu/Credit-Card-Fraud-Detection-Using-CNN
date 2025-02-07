
# Credit Card Fraud Detection Using CNN

## Description

This project demonstrates the application of a Convolutional Neural Network (CNN) to detect fraudulent credit card transactions. Using a highly imbalanced dataset sourced from Kaggle, the model employs advanced data preprocessing and oversampling techniques to achieve high accuracy in classifying transactions as legitimate or fraudulent. This project showcases the potential of deep learning in addressing financial fraud.

---

## Overview

- **Objective**: Detect fraudulent credit card transactions using CNN.
- **Dataset**: Kaggle credit card fraud dataset with 284,807 transactions, where 492 are labeled as fraudulent.
- **Key Challenges**: 
  - Handling highly imbalanced data.
  - Achieving high precision and recall to reduce false positives and negatives.
- **Outcome**: CNN outperforms traditional classifiers, demonstrating robustness in fraud detection.

---

## Implementation

### Interesting Techniques
1. **Synthetic Minority Oversampling Technique (SMOTE)**:
   - Balances the dataset by generating synthetic examples for the minority class.
   - Learn more on [SMOTE](https://imbalanced-learn.org/stable/over_sampling.html).

2. **Convolutional Layers**:
   - Extract meaningful features from the data.
   - See [MDN's guide on CNNs](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview).

3. **Max Pooling**:
   - Reduces the spatial dimensions of feature maps.
   - Check [MDN Max Pooling](https://developer.mozilla.org/).

4. **Dropout Layers**:
   - Prevents overfitting by randomly dropping neurons during training.

### Technologies and Libraries
- **TensorFlow/Keras**: Frameworks for building and training the CNN.
- **SMOTE (imbalanced-learn)**: For dataset balancing.
- **NumPy, Pandas**: For data manipulation and preprocessing.
- **Matplotlib/Seaborn**: For visualization.

---

## Project Structure

```plaintext
CreditCardFraudDetection/
├── data/               # Contains raw and processed datasets
├── models/             # Saved trained CNN models
├── notebooks/          # Jupyter notebooks for experiments and visualizations
├── src/                # Source code for model and preprocessing
└── results/            # Evaluation metrics and result visualizations
```

- **data/**: Includes the Kaggle dataset and preprocessed subsets.
- **notebooks/**: Contains detailed exploratory data analysis (EDA).
- **src/**: Houses scripts for preprocessing, model building, and training.

---

## Data Exploration

The dataset contains 31 features:
- **Time**: Transaction timestamp.
- **Amount**: Transaction value.
- **Class**: 1 for fraud, 0 for non-fraud.
- Features V1-V28 are PCA-transformed for privacy reasons.

Key insights:
- Fraudulent transactions constitute only 0.17% of the dataset.
- Significant preprocessing is required to address data imbalance.

---

## Model Evaluation and Prediction

- **Metrics Used**: Precision, Recall, F1-Score, and Accuracy.
- **Performance**:
  - Precision: 98.7%
  - Recall: 97.5%
  - F1-Score: 98.05%
- The model achieves a significant reduction in false positives and negatives, ensuring reliability.

---

## Multi-Layer Perceptron

The CNN architecture comprises:
1. Two convolutional layers with ReLU activation.
2. Max pooling for dimensionality reduction.
3. Dropout layers to prevent overfitting.
4. A dense layer with a sigmoid activation for binary classification.

---

## Lessons Learned

- **Data Imbalance** : Handling imbalanced datasets with SMOTE is critical in fraud detection.
- **Model Generalization** : Dropout layers significantly improve model robustness.
- **CNN Superiority** : Convolutional layers effectively capture patterns in skewed datasets.
