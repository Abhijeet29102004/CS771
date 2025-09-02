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
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score ,  roc_curve, roc_auc_score

# prompt: mount google drive

# Load the data
train_data1 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\train\train_emoticon.csv")
valid_data1 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\valid\valid_emoticon.csv")

train_data2 = np.load(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\train\train_feature.npz")
valid_data2 = np.load(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\valid\valid_feature.npz")

train_data3 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\train\train_text_seq.csv")
valid_data3 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\valid\valid_text_seq.csv")

x_train1 = train_data1['input_emoticon']
y_train1 = train_data1['label']

x_valid1 = valid_data1['input_emoticon']
y_valid1 = valid_data1['label']
print(f"x_train1 : {x_train1[0]}")
print(f"x_valid1 : {x_valid1[0]}")
# Split into train and test sets (assuming 'valid_data' is your test data)
x_train2 = train_data2['features']
y_train2 = train_data2['label']  # Assuming 'target_variable' is your target column name
x_valid2 = valid_data2['features']
y_valid2 = valid_data2['label']

# # Reshape x_train and x_valid to 2 dimensions
x_train2 = x_train2.reshape(x_train2.shape[0], -1)
x_valid2 = x_valid2.reshape(x_valid2.shape[0], -1)

x_train3 = train_data3['input_str']
y_train3 = train_data3['label']

x_valid3 = valid_data3['input_str']
y_valid3 = valid_data3['label']


# Split the data or use all data if percentage is 1.0
# if percentage == 1.0:
# x_train1_cur = x_train1
# y_train1_cur = y_train1

# x_train2_cur = x_train2
# y_train2_cur = y_train2

# x_train3_cur = x_train3
# y_train3_cur = y_train3
# else:
# x_train1_cur = x_train1[:num_samples]
# y_train1_cur = y_train1[:num_samples]

# x_train2_cur = x_train2[:num_samples]
# y_train2_cur = y_train2[:num_samples]

# x_train3_cur = x_train3[:num_samples]
# y_train3_cur = y_train3[:num_samples]

# x_valid1_cur = x_valid1
# y_valid1_cur = y_valid1

# x_valid2_cur = x_valid2
# y_valid2_cur = y_valid2

# x_valid3_cur = x_valid3
# y_valid3_cur = y_valid3

########################### part 1 #####

# adding blanckspace between emojis
def add_spaces_to_emojis(text):
# Split the text into parts

    new_text = text[0]
    for i in range(12):
        new_text = new_text + " "
        new_text = new_text + text[i + 1]

    return new_text

x_train1 = [add_spaces_to_emojis(line) for line in x_train1]
x_valid1 = [add_spaces_to_emojis(line) for line in x_valid1]
print(f"x_train1 2 : {x_train1[0]}")
print(f"x_valid1 2: {x_valid1[0]}")
# Preprocess the input data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train1)
word_index = tokenizer.word_index
vocabulary_size = len(word_index) + 1  # +1 for padding token

x_train1 = tokenizer.texts_to_sequences(x_train1)
x_valid1 = tokenizer.texts_to_sequences(x_valid1)
print(f"x_train1 : {x_train1[0]}")
print(f"x_valid1 : {x_valid1[0]}")
# Convert to numpy arrays
x_train1 = np.array(x_train1)
x_valid1 = np.array(x_valid1)
y_train1 = np.array(y_train1)
y_valid1 = np.array(y_valid1)

def create_model1(vocabulary_size, embedding_dim, input_length):
    inputs = Input(shape=(input_length,))
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=input_length)(inputs)
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

print(f"x_train1 : {x_train1[0]}")
print(f"x_valid1 : {x_valid1[0]}")

# Train the model
model.fit(x_train1, y_train1, epochs=5, batch_size=32, validation_data=(x_valid1, y_valid1))

# print accuracy 
loss, accuracy = model.evaluate(x_valid1, y_valid1)
print(f"Accuracy: {accuracy}")

# Create a feature extractor model
feature_extractor1 = Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract features
x_train1_features = feature_extractor1.predict(x_train1)
x_valid1_features = feature_extractor1.predict(x_valid1)

# ################ part 2##

x_train2_features = np.array(x_train2)
x_valid2_features = np.array(x_valid2)

############### part 3##

# Function to preprocess data from string to int
def string_to_int(data):
    text_data = data.tolist()

    # Convert strings of numbers to numerical sequences
    numerical_sequences = [[int(digit) for digit in text] for text in text_data]

    # Pad sequences to the same length for input into CNNs
    max_length = max(len(seq) for seq in numerical_sequences)
    padded_sequences = pad_sequences(numerical_sequences, maxlen=max_length, padding='post')

    return np.array(padded_sequences)

x_train3 = string_to_int(x_train3)
x_valid3 = string_to_int(x_valid3)

y_train3 = np.array(y_train3)
y_valid3 = np.array(y_valid3)

def create_cnn_model(vocab_size, embedding_dim, max_length):
    inputs = Input(shape=(max_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, name='embedding_layer')(inputs)
    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(16, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define model parameters
max_length = 50  # Length of the input sequences (adjust as per your data)
vocab_size = 10  # Number of unique tokens in your dataset (adjust as per your data)
embedding_dim = 20

# Create the model
model = create_cnn_model(vocab_size, embedding_dim, max_length)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train the model (Ensure that x_train3, y_train3, x_valid3, y_valid3 are defined)
history = model.fit(
    x_train3, y_train3,
    epochs=20,  # Increased epochs, but we'll use early stopping
    batch_size=32,  # Increased batch size
    validation_data=(x_valid3, y_valid3),
    callbacks=[early_stopping, reduce_lr]
)

# Print model summary
model.summary()

# Create a feature extractor model
feature_extractor3 = Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract embeddings
x_train3_features = feature_extractor3.predict(x_train3)
x_valid3_features = feature_extractor3.predict(x_valid3)

print("Embedding shapes:")
print(f"Training embeddings shape: {x_train3_features.shape}")
print(f"Validation embeddings shape: {x_valid3_features.shape}")


# Function to apply PCA and return both the transformed data and the fitted PCA object
def apply_pca(X, n_components=0.95):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

# Function to preprocess data
def dim_reduction(x1, x2, x3, pca1=None, pca2=None, pca3=None):
    scaler = StandardScaler()
    
    x1_scaled = scaler.fit_transform(x1)
    x2_scaled = scaler.fit_transform(x2)
    x3_scaled = scaler.fit_transform(x3)
    
    if pca1 is None:
        x1_pca, pca1 = apply_pca(x1_scaled)
        x2_pca, pca2 = apply_pca(x2_scaled)
        x3_pca, pca3 = apply_pca(x3_scaled)
    else:
        x1_pca = pca1.transform(x1_scaled)
        x2_pca = pca2.transform(x2_scaled)
        x3_pca = pca3.transform(x3_scaled)
    
    X_combined = np.hstack((x1_pca, x2_pca, x3_pca))
    return X_combined, pca1, pca2, pca3

# Preprocess training data
X_train_combined, pca1, pca2, pca3 = dim_reduction(x_train1_features, x_train2_features, x_train3_features)

# Preprocess validation data using the same PCA objects
X_valid_combined, _, _, _ = dim_reduction(x_valid1_features, x_valid2_features, x_valid3_features, pca1, pca2, pca3)

print("Combined features shapes:")
print(f"Training combined features shape: {X_train_combined.shape}")
print(f"Validation combined features shape: {X_valid_combined.shape}")



# Define the model
model = SVC(kernel = 'rbf', random_state=42)

# Fit the model
model.fit(X_train_combined, y_train1)

# Make predictions
y_pred = model.predict(X_valid_combined)

test_data1 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\test\test_emoticon.csv")
test_data2 = np.load(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\test\test_feature.npz")
test_data3 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\test\test_text_seq.csv")

x_test1 = test_data1['input_emoticon']
print(f"x_test1 1: {len(x_test1[0])}")
x_test2 = test_data2['features']
x_test3 = test_data3['input_str']

# Preprocess the test data
x_test1 = [add_spaces_to_emojis(line) for line in x_test1]
print(f"x_test1 2: {len(x_test1[0])}")
x_test1 = tokenizer.texts_to_sequences(x_test1)
x_test1 = pad_sequences(x_test1, maxlen=13, padding='post')

print(f"x_test1 3: {len(x_test1[0])}")

x_test2 = x_test2.reshape(x_test2.shape[0], -1)
print(f"x_test2 4: {x_test2[0]}")

x_test3 = string_to_int(x_test3)
print(f"x_test3 : {x_test3[0]}")

# Convert to numpy arrays
x_test1 = np.array(x_test1)
x_test2 = np.array(x_test2)

# Extract features
x_test1_features = feature_extractor1.predict(x_test1)
x_test2_features = np.array(x_test2)
x_test3_features = feature_extractor3.predict(x_test3)

# Preprocess test data
X_test_combined, _, _, _ = dim_reduction(x_test1_features, x_test2_features, x_test3_features, pca1, pca2, pca3)

# Make predictions
y_pred_test = model.predict(X_test_combined)
print(f"y_pred_test : {y_pred_test.shape}")
print(f"x_test1 : {x_test1.shape}")


# Save predictions to prer_combined.txt
np.savetxt("pred_combined.txt", y_pred_test, fmt='%d')

print("Predictions saved to pred_combined.txt")



    

    
