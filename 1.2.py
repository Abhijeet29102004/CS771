import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from IPython.display import display

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# metrics and selection
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score ,  roc_curve, roc_auc_score



# Load the data
train_data2 = np.load(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\train\train_feature.npz")
valid_data2 = np.load(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\valid\valid_feature.npz")

# Split into train and test sets (assuming 'valid_data' is your test data)
x_train2 = train_data2['features']
y_train2 = train_data2['label']  # Assuming 'target_variable' is your target column name
x_valid2 = valid_data2['features']
y_valid2 = valid_data2['label']

# Reshape x_train and x_valid to 2 dimensions
x_train2 = x_train2.reshape(x_train2.shape[0], -1)
x_valid2 = x_valid2.reshape(x_valid2.shape[0], -1)

x_train2.shape, y_train2.shape, x_valid2.shape, y_valid2.shape


# prompt: train a xgbclassifier model

# Initialize the XGBoost classifier
xgb_model = XGBClassifier()

# Train the model
xgb_model.fit(x_train2, y_train2)

# Make predictions on the validation set
y_pred2 = xgb_model.predict(x_valid2)

# Evaluate the model (you can use various metrics like accuracy, precision, recall, F1-score)
from sklearn.metrics import accuracy_score

accuracy2 = accuracy_score(y_valid2, y_pred2)
print("Accuracy:", accuracy2)



# Load the test data
test_data2 = np.load(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\test\test_feature.npz")

# Extract features from the test data
x_test2 = test_data2['features']

# Reshape x_test to match the shape expected by the model
x_test2 = x_test2.reshape(x_test2.shape[0], -1)

# Make predictions on the test set using the trained XGBoost model
y_test_pred2 = xgb_model.predict(x_test2)

# Save the predictions to pred_deepfeat.txt
np.savetxt("pred_deepfeat.txt", y_test_pred2, fmt='%d')

print("Predictions saved to pred_deepfeat.txt")
