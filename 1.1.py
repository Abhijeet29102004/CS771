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
from tensorflow.keras.preprocessing.sequence import pad_sequences

# metrics and selection
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score ,  roc_curve, roc_auc_score


# Load the data
train_data1 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\train\train_emoticon.csv")
valid_data1 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\valid\valid_emoticon.csv")

x_train1 = train_data1['input_emoticon']
y_train1 = train_data1['label']

x_valid1 = valid_data1['input_emoticon']
y_valid1 = valid_data1['label']

train_data1.head()


# Add spaces to emojis
def add_spaces_to_emojis(text):
    # Split the text into parts
  
    new_text = text[0]
    for i in range(12):
        new_text = new_text + " "
        new_text = new_text + text[i + 1]

    return new_text

x_train1 = [add_spaces_to_emojis(line) for line in x_train1]
x_valid1 = [add_spaces_to_emojis(line) for line in x_valid1]

x_train1[0:5]


# Preprocess the input data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train1)
word_index = tokenizer.word_index
vocabulary_size = len(word_index) + 1  # +1 for padding token

x_train_token = tokenizer.texts_to_sequences(x_train1)
x_valid_token = tokenizer.texts_to_sequences(x_valid1)

vocabulary_size


# Convert to numpy arrays
x_train1 = np.array(x_train_token)
x_valid1 = np.array(x_valid_token)
y_train1 = np.array(y_train1)
y_valid1 = np.array(y_valid1)


# Define the model
def create_model1(vocabulary_size, embedding_dim, input_length):
    inputs = Input(shape=(input_length,))
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=input_length)(inputs)
    # conv1 = Conv1D(32, kernel_size=3, activation='relu')(embedding)
    # maxpool1 = MaxPooling1D(pool_size=2)(conv1)
    # conv2 = Conv1D(32, kernel_size=3, activation='relu')(maxpool1)
    # maxpool2 = MaxPooling1D(pool_size=2)(conv2)
    flatten = Flatten()(embedding)
    hidden1 = Dense(32, activation='relu')(flatten)
    hidden2 = Dense(32, activation='relu')(hidden1)
    outputs = Dense(1, activation='sigmoid')(hidden2)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define model parameters
embedding_dim = 5
input_length = 13

# Create the model
model = create_model1(vocabulary_size, embedding_dim, input_length)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train1, y_train1, epochs=5, batch_size=32, validation_data=(x_valid1, y_valid1))

model.summary()

# Create a feature extractor model
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract features
x_train1_features = feature_extractor.predict(x_train1)
x_valid1_features = feature_extractor.predict(x_valid1)

print(x_train1_features.shape, x_valid1_features.shape)


# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(x_valid1, y_valid1, verbose=0)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")



# Make predictions


# Load the test data
test_data1 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\test\test_emoticon.csv")
x_test1 = test_data1['input_emoticon']

# Add spaces to emojis for the test data
x_test1 = [add_spaces_to_emojis(line) for line in x_test1]

# Tokenize the test data using the same tokenizer
x_test_token = tokenizer.texts_to_sequences(x_test1)

# Pad the test data to ensure it has the same length as the training data
x_test1 = pad_sequences(x_test_token, maxlen=input_length, padding='post')

# Predict using the trained model
y_test_pred = model.predict(x_test1)

# Convert predictions to binary (0 or 1) using a threshold of 0.5
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

# Save the predictions to pred_emoticon.txt
np.savetxt("pred_emoticon.txt", y_test_pred_binary, fmt='%d')

print("Predictions saved to pred_emoticon.txt")
