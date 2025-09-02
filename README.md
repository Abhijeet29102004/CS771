# CS771 Mini Project - Binary Classification Ensemble

## Instructor
*Prof. Piyush Rai*

## Project Overview
This project focuses on developing and evaluating binary classification models for three distinct datasets derived from an original raw dataset. Each dataset represents the same underlying data but with different feature representations. The goal is to identify the best-performing model based on validation accuracy and demonstrate the benefits of combining complementary information from different feature types.

## Dataset

You can download the datasets and some helper code from the following URL:

```
https://tinyurl.com/cs771-autumn24-mp-1-data
```


## Project Objectives
1. **Task 1:** Develop individual binary classification models for each dataset, exploring different model architectures and training strategies.
2. **Task 2:** Create a unified ensemble model by combining the datasets to leverage complementary information from different feature representations.

## Datasets

The project uses three different datasets, each with unique characteristics:

### 1. Emoticon Features
- **Description:** 13 categorical features, each represented by an emoji.
- **Preprocessing:** 
  - Adds spaces between emojis for tokenization.
  - Tokenization using Keras/TensorFlow tokenizer.
  - One-hot encoding and padding to length 13.

### 2. Deep Features
- **Description:** Each input is a 13x786 matrix of embeddings.
- **Preprocessing:** 
  - Flattened to a single vector for compatibility with ML models.

### 3. Text Sequence Features
- **Description:** Each input is a string of 50 digits.
- **Preprocessing:** 
  - Converts strings to integer sequences.
  - Pads sequences to max length (50).

## Model Development & Methodology

### Emoticon Model (`1.1.py`)
- **Data Processing Strategies:**
  1. One-hot encoding of emoticon sequences.
  2. Tokenization with unique numerical values.
- **Models Trained:**
  - Custom Neural Network (Embedding + Dense layers) with early stopping and hyperparameter tuning.
  - Traditional Models: Logistic Regression, SVM, Random Forest, XGBoost, KNN.
- **Feature Extraction:** Penultimate layer used for downstream tasks.
- **Output:** Predictions saved to `pred_emoticon.txt`.

### Deep Feature Model (`1.2.py`)
- **Models Used:**
  - XGBoost (with extensive hyperparameter tuning).
  - Random Forest (200 estimators).
  - Logistic Regression (baseline).
  - SVM (RBF kernel for high-dimensional data).
- **Output:** Predictions saved to `pred_deepfeat.txt`.

### Text Sequence Model (`1.3.py`)
- **Model Architecture:**
  - Hybrid model with CNN feature extractor (Embedding, Conv1D, MaxPooling, Dense).
  - Traditional classifiers (Logistic Regression, SVM, XGBoost, LightGBM) evaluated on extracted features.
  - Focus on parameter efficiency (total trainable parameters < 10,000).
- **Output:** Predictions saved to `pred_textseq.txt`.

### Combined Ensemble Model (`2.py`)
- **Methodology:**
  - Extracts features from each individual model.
  - Applies PCA and standard scaling for dimensionality reduction and normalization.
  - Concatenates features from all three sources.
  - Trains an SVM classifier on the combined feature set.
- **Output:** Predictions saved to `pred_combined.txt`.

### Automation (`32.py`)
- Runs all scripts in sequence and prints colored status messages for success/failure.

## Results and Evaluation

- Each model's performance was assessed based on validation accuracy across varying training sizes (20%, 40%, 60%, 80%, 100%).
- **Key Findings:**
  - Neural networks outperformed traditional models in the Emoticons Dataset.
  - XGBoost demonstrated superior performance on the Deep Features Dataset.
  - Logistic Regression and XGBoost performed well on the Text Sequence Dataset.
  - The ensemble model leverages feature diversity for improved classification.

### Performance Summary

| Model                     | 20%   | 40%   | 60%   | 80%   | 100%  |
|---------------------------|-------|-------|-------|-------|-------|
| Logistic Regression       | 0.7198 | 0.8200 | 0.8650 | 0.9162 | 0.9243 |
| Random Forest             | 0.6033 | 0.7301 | 0.7464 | 0.8384 | 0.8589 |
| Custom Neural Network     | 0.9141 | 0.9284 | 0.9427 | 0.9714 | 0.9734 |
| XGBoost                   | 0.9489 | 0.9734 | 0.9816 | 0.9836 | 0.9877 |

## Key Learnings

- Combining diverse feature representations can improve classification performance.
- Deep learning and classical ML models can be integrated via feature extraction.
- Automated pipelines streamline experimentation and evaluation.
- Model architecture and preprocessing strategies are crucial for high validation accuracy.

## Tech Stack

- **Python**: Main programming language
- **TensorFlow/Keras**: Deep learning models (Embedding, CNN, Dense)
- **scikit-learn**: Classical ML models, preprocessing, metrics, PCA
- **XGBoost, LightGBM**: Gradient boosting classifiers
- **Pandas, NumPy**: Data manipulation and numerical operations
- **Matplotlib**: Visualization
- **IPython.display**: Tabular result display

## Installation and Usage

To replicate the results or explore further:
1. Ensure you have the required libraries installed (e.g., TensorFlow, Scikit-learn, XGBoost, LightGBM, etc.).
2. Place datasets in the `datasets/train`, `datasets/valid`, and `datasets/test` folders.
3. Run the main automation script:
   ```bash
   python 32.py
   ```
4. Individual scripts can be run for specific models:
   - `python 1.1.py` (Emoticon)
   - `python 1.2.py` (Deep Features)
   - `python 1.3.py` (Text Sequence)
   - `python 2.py` (Combined Model)

## Output Files

- `pred_emoticon.txt`: Predictions from the emoticon model
- `pred_deepfeat.txt`: Predictions from the deep feature model
- `pred_textseq.txt`: Predictions from the text sequence model
- `pred_combined.txt`: Predictions from the ensemble model

## Conclusion

This project successfully implemented and evaluated multiple binary classification models across various datasets. The findings highlight the importance of model architecture, feature engineering, and ensemble strategies in achieving high validation