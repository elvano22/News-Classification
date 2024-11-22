# News-Classification

This repository contains a machine learning project focused on classifying news articles into predefined categories using text representation techniques and machine learning algorithms.

## Project Overview

The project explores two text representation techniques:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Word2Vec (Word Embeddings)**

Two machine learning models are implemented and evaluated:
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

### Objective
The goal is to identify which combination of text representation and machine learning model provides the best performance for news classification.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/automatic-news-classification.git
   cd automatic-news-classification
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset

The dataset consists of labeled news articles in four categories:
- **Bisnis** (Business)
- **Hiburan** (Entertainment)
- **Olahraga** (Sports)
- **Politik** (Politics)

The dataset is preprocessed to convert raw text into feature representations (TF-IDF and Word2Vec) for machine learning models.

---

## Models and Techniques

### Text Representations
1. **TF-IDF**: Captures word importance in a document.
2. **Word2Vec**: Produces dense word embeddings.

### Machine Learning Models
1. **Support Vector Machine (SVM)**: A powerful classifier for high-dimensional spaces.
2. **Random Forest Classifier**: A robust ensemble learning method.

Hyperparameter tuning is performed using `GridSearchCV`.

---

## Results

The performance of each combination is summarized below:

| Text Representation | Algorithm       | Best Parameters                        | Accuracy | Precision | Recall | F1 Score |
|----------------------|-----------------|----------------------------------------|----------|-----------|--------|----------|
| **TF-IDF**           | SVM             | `{'C': 10, 'gamma': 0.1}`             | 0.8788   | 0.9141    | 0.8213 | 0.8497   |
| **TF-IDF**           | Random Forest   | `{'max_depth': 10, 'n_estimators': 200}` | 0.8939   | 0.8702    | 0.8491 | 0.8583   |
| **Word2Vec**         | SVM             | `{'C': 1, 'gamma': 1}`                | 0.9394   | 0.9394    | 0.9223 | 0.9258   |
| **Word2Vec**         | Random Forest   | `{'max_depth': 30, 'n_estimators': 50}` | 0.9394   | 0.9394    | 0.9223 | 0.9258   |

### Observations
1. **Word2Vec** consistently outperformed **TF-IDF** in terms of accuracy, precision, recall, and F1 score.
2. Both **SVM** and **Random Forest** performed equally well with Word2Vec, achieving accuracy and F1 scores around 93-94%.
3. For **TF-IDF**, Random Forest performed slightly better than SVM, with higher recall and accuracy.

---

## How to Run

1. Open the notebook `Automatic_News_Classification.ipynb`.
2. Run all cells to preprocess data, train models, and evaluate performance.
3. Review results and performance comparison.

---

## Conclusion

The project demonstrates that **Word2Vec** representations significantly improve classification performance. Future work could explore more advanced models like **deep learning architectures** or additional preprocessing for further improvements.
