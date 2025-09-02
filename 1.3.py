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


print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("CUDA Available:", tf.test.is_built_with_cuda())



print("Is GPU Available: ", tf.test.is_gpu_available())


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  print("Name:", gpu.name, "  Type:", gpu.device_type)

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)


# Load the data
train_data3 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\train\train_text_seq.csv")
valid_data3 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\valid\valid_text_seq.csv")

x_train3 = train_data3['input_str']
y_train3 = train_data3['label']

x_valid3 = valid_data3['input_str']
y_valid3 = valid_data3['label']

x_train3[0]

# string to int
# Function to preprocess data
def preprocess_data(data):
    text_data = data['input_str'].tolist()
    # labels = data['label'].tolist()
    if 'label' in data.columns:
        labels = data['label'].tolist()
    else:
        labels = None  # or handle as needed


    # Convert strings of numbers to numerical sequences
    numerical_sequences = [[int(digit) for digit in text] for text in text_data]

    # Pad sequences to the same length for input into CNNs
    max_length = max(len(seq) for seq in numerical_sequences)
    padded_sequences = pad_sequences(numerical_sequences, maxlen=max_length, padding='post')

    return np.array(padded_sequences), np.array(labels)

# Preprocess the train and valid datasets
x_train3, y_train3 = preprocess_data(train_data3)
x_valid3, y_valid3 = preprocess_data(valid_data3)

(x_train3[0])

# Model training
def create_model(vocab_size, embedding_dim, max_length):
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
model = create_model(vocab_size, embedding_dim, max_length)

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

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_valid3, y_valid3)
print(f"Test accuracy: {test_accuracy:.4f}")

# Create a feature extractor model
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract embeddings
x_train3_embeddings = feature_extractor.predict(x_train3)
x_valid3_embeddings = feature_extractor.predict(x_valid3)

print("Embedding shapes:")
print(f"Training embeddings shape: {x_train3_embeddings.shape}")
print(f"Validation embeddings shape: {x_valid3_embeddings.shape}")

# # If you want to get the embedding matrix itself
# embedding_matrix = embedding_layer.get_weights()[0]
# print(f"Embedding matrix shape: {embedding_matrix.shape}")


models = {
    'Logistic Regression': LogisticRegression(max_iter = 1000),
    # 'Decision Tree': DecisionTreeClassifier(),
    # 'Random Forest': RandomForestClassifier(),
    # 'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    # 'Neural Network': MLPClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(verbosity=-1)
}


# Function to train and evaluate models
def train_and_evaluate_models(models, x_train, y_train, x_valid, y_valid):
    # Dataframe to store results
    results = pd.DataFrame(columns=['Accuracy', 'F1 Score', 'ROC AUC'], index=models.keys())

    # Loop through models
    for name, model in models.items():
        # Train the model
        model.fit(x_train, y_train)

        # Make predictions
        y_pred = model.predict(x_valid)

        # Calculate metrics
        accuracy = accuracy_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred)
        roc_auc = roc_auc_score(y_valid, y_pred)

        # Store results
        results.loc[name] = [accuracy, f1, roc_auc]

    return results


final_results = []

# Evaluate each model using different training sizes
for train_size in [1.0]:  # 20%, 40%, 60%, 80% training sizes
    # Determine the number of training samples to use
    num_train_samples = int(train_size * len(x_train3_embeddings))

    # Train and evaluate each model
    result = train_and_evaluate_models(models, x_train3_embeddings[:num_train_samples], y_train3[:num_train_samples], x_valid3_embeddings, y_valid3)
    final_results.append(result)


# display the results using display from IPython.display
for i, result in enumerate(final_results):
    display(result.style.set_caption(f"Training size: {int([1.0][i] * 100)}%"))


# Prediction
# Load the test data
test_data3 = pd.read_csv(r"C:\Users\saini\Downloads\CS771\mini-project-1\mini-project-1\datasets\test\test_text_seq.csv")
x_test3 = test_data3['input_str']

# Preprocess the test data using the same function
x_test3, _ = preprocess_data(test_data3)  # No labels, so ignore the second return value

# Extract embeddings for the test data
x_test3_embeddings = feature_extractor.predict(x_test3)

# Function to save predictions to a file
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as file:
        for pred in predictions:
            file.write(f"{pred}\n")

# Find the best model based on a specific metric (e.g., accuracy)
def get_best_model(results):
    best_model_name = results['Accuracy'].idxmax()  # You can also choose 'F1 Score' or 'ROC AUC'
    return best_model_name

# Train and evaluate models
final_results = []
for train_size in [1.0]:  # Adjust the training size if needed
    num_train_samples = int(train_size * len(x_train3_embeddings))
    result = train_and_evaluate_models(models, x_train3_embeddings[:num_train_samples], y_train3[:num_train_samples], x_valid3_embeddings, y_valid3)
    final_results.append(result)

# Get the results for the full training size
best_model_name = get_best_model(final_results[0])

# Print the name of the best model
print(f"Best model: {best_model_name}")

# Use the best model to predict on the test embeddings
best_model = models[best_model_name]
y_test_pred = best_model.predict(x_test3_embeddings)

# Save the predictions to pred_emoticon.txt
save_predictions_to_file(y_test_pred, "pred_textseq.txt")
print("Predictions saved to pred_textseq.txt")



