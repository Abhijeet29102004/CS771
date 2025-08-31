import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Embedding, Dropout, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



class COMBINED_MODEL:
    def __init__(self):
        self.lstm_model = None
        self.logistic_model = None
        self.cnn_lstm_model = None
        self.weights = {}

    def train(self):
        # Load training datasets
        emoji_dataset_train = pd.read_csv('datasets/train/train_emoticon.csv')
        train_data = np.load('datasets/train/train_feature.npz')
        X_train_deep = train_data['features']
        y_train_deep = train_data['label']

        train_data_path = 'datasets/train/train_text_seq.csv'
        train_dataset = pd.read_csv(train_data_path)
        X_train_text = train_dataset['input_str'].apply(lambda x: [int(digit) for digit in x]).tolist()
        X_train_text = pad_sequences(X_train_text, maxlen=50, padding='post')
        y_train_text = train_dataset['label'].values

        emoji_to_int = {emoji: idx + 1 for idx, emoji in enumerate(set(''.join(emoji_dataset_train['input_emoticon'])))}
        emoji_dataset_train['encoded_emoticons'] = emoji_dataset_train['input_emoticon'].apply(
            lambda emojis: [emoji_to_int[emoji] for emoji in emojis if emoji in emoji_to_int]
        )
        X_train_emoji = pad_sequences(emoji_dataset_train['encoded_emoticons'], maxlen=10, padding='post')

        self.lstm_model = load_model('lstm_model_emoticon.h5')
        self.lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.lstm_model.fit(X_train_emoji, emoji_dataset_train['label'].values, epochs=10, batch_size=32, validation_split=0.2)

        X_train_deep_flattened = X_train_deep.reshape(X_train_deep.shape[0], -1)
        self.logistic_model = joblib.load('model_deep_feature.joblib')
        self.logistic_model.fit(X_train_deep_flattened, y_train_deep)

        self.cnn_lstm_model = load_model('cnn_lstm_model_text_seq.h5')
        self.cnn_lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.cnn_lstm_model.fit(X_train_text, y_train_text, epochs=10, batch_size=32, validation_split=0.2)

        # Validation datasets (FIXED PATHS)
        emoji_dataset_valid = pd.read_csv('datasets/valid/valid_emoticon.csv')
        validation_data = np.load('datasets/valid/valid_feature.npz')
        X_val_deep = validation_data['features']
        y_val_deep = validation_data['label']

        val_data_path = 'datasets/valid/valid_text_seq.csv'
        val_dataset = pd.read_csv(val_data_path)
        X_val_text = val_dataset['input_str'].apply(lambda x: [int(digit) for digit in x]).tolist()
        X_val_text = pad_sequences(X_val_text, maxlen=50, padding='post')
        y_val_text = val_dataset['label'].values

        emoji_dataset_valid['encoded_emoticons'] = emoji_dataset_valid['input_emoticon'].apply(
            lambda emojis: [emoji_to_int[emoji] for emoji in emojis if emoji in emoji_to_int]
        )
        X_val_emoji = pad_sequences(emoji_dataset_valid['encoded_emoticons'], maxlen=10, padding='post')

        y_pred_lstm = np.argmax(self.lstm_model.predict(X_val_emoji), axis=-1)
        accuracy_lstm = accuracy_score(emoji_dataset_valid['label'].values, y_pred_lstm)

        X_val_deep_flattened = X_val_deep.reshape(X_val_deep.shape[0], -1)
        y_pred_logistic = self.logistic_model.predict(X_val_deep_flattened)
        accuracy_logistic = accuracy_score(y_val_deep, y_pred_logistic)

        y_pred_cnn_lstm = (self.cnn_lstm_model.predict(X_val_text) > 0.5).astype(int).flatten()
        accuracy_cnn_lstm = accuracy_score(y_val_text, y_pred_cnn_lstm)

        print(f"Logistic Model Accuracy: {accuracy_logistic * 100:.2f}%")
        print(f"LSTM Model Accuracy: {accuracy_lstm * 100:.2f}%")
        print(f"CNN-LSTM Model Accuracy: {accuracy_cnn_lstm * 100:.2f}%")

        total_accuracy = accuracy_logistic + accuracy_lstm + accuracy_cnn_lstm
        weights = {
            'logistic': accuracy_logistic / total_accuracy,
            'lstm': accuracy_lstm / total_accuracy,
            'cnn_lstm': accuracy_cnn_lstm / total_accuracy
        }
        
        combined_predictions = np.array([y_pred_logistic, y_pred_lstm, y_pred_cnn_lstm]).T
        weighted_vote_predictions = []
        for votes in combined_predictions:
            weighted_sum = (weights['logistic'] * votes[0] +
                            weights['lstm'] * votes[1] +
                            weights['cnn_lstm'] * votes[2])
            final_prediction = 1 if weighted_sum > 0.5 else 0
            weighted_vote_predictions.append(final_prediction)

        weighted_vote_predictions = np.array(weighted_vote_predictions)
        true_labels = emoji_dataset_valid['label'].values
        accuracy = accuracy_score(true_labels, weighted_vote_predictions)
        print(f"Weighted Combined Model Accuracy: {accuracy * 100:.2f}%")

        # Load test datasets (FIXED PATHS)
        emoji_dataset_test = pd.read_csv('datasets/test/test_emoticon.csv')
        test_data = np.load('datasets/test/test_feature.npz')
        X_test_deep = test_data['features']

        test_data_path = 'datasets/test/test_text_seq.csv'
        test_dataset = pd.read_csv(test_data_path)
        X_test_text = test_dataset['input_str'].apply(lambda x: [int(digit) for digit in x]).tolist()
        X_test_text = pad_sequences(X_test_text, maxlen=50, padding='post')

        emoji_to_int = {emoji: idx + 1 for idx, emoji in enumerate(set(''.join(emoji_dataset_test['input_emoticon'])))}
        emoji_dataset_test['encoded_emoticons'] = emoji_dataset_test['input_emoticon'].apply(
            lambda emojis: [emoji_to_int[emoji] for emoji in emojis if emoji in emoji_to_int]
        )

        MAX_EMOJI_INDEX = 214

        def preprocess_emojis(emoji_data, max_index=MAX_EMOJI_INDEX):
            return [[min(emoji, max_index) for emoji in seq] for seq in emoji_data]

        X_test_emoji = preprocess_emojis(emoji_dataset_test['encoded_emoticons'])
        X_test_emoji = pad_sequences(X_test_emoji, maxlen=10, padding='post')

        y_pred_lstm = np.argmax(self.lstm_model.predict(X_test_emoji), axis=-1)
        X_test_deep_flattened = X_test_deep.reshape(X_test_deep.shape[0], -1)
        y_pred_logistic = self.logistic_model.predict(X_test_deep_flattened)
        y_pred_cnn_lstm = (self.cnn_lstm_model.predict(X_test_text) > 0.5).astype(int).flatten()
        
        combined_predictions_test = np.array([y_pred_logistic, y_pred_lstm, y_pred_cnn_lstm]).T
        weighted_vote_predictions_test = []
        for votes in combined_predictions_test:
            weighted_sum = (weights['logistic'] * votes[0] +
                            weights['lstm'] * votes[1] +
                            weights['cnn_lstm'] * votes[2])
            final_prediction = 1 if weighted_sum > 0.5 else 0
            weighted_vote_predictions_test.append(final_prediction)

        weighted_vote_predictions_test = np.array(weighted_vote_predictions_test)
        np.savetxt('weighted_vote_predictions_test.txt', weighted_vote_predictions_test, fmt='%d', header='Weighted Vote Test Predictions')
        print("Prediction saved in weighted_vote_predictions_test.txt")





class CNNLSTMModel:
    def __init__(self, max_sequence_length=50, num_unique_digits=10, embedding_output_dim=32, filters_1=32, filters_2=8, dense_units=64):
        self.max_sequence_length = max_sequence_length
        self.num_unique_digits = num_unique_digits
        self.embedding_output_dim = embedding_output_dim
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.dense_units = dense_units
        self.model = None

    def load_data(self, train_data_path, val_data_path):
        # Load the datasets
        train_dataset = pd.read_csv(train_data_path)
        val_dataset = pd.read_csv(val_data_path)

        # Convert digit sequences to lists of integers
        train_dataset['encoded_sequences'] = train_dataset['input_str'].apply(lambda x: [int(digit) for digit in x])
        val_dataset['encoded_sequences'] = val_dataset['input_str'].apply(lambda x: [int(digit) for digit in x])

        # Pad the sequences
        X_train = pad_sequences(train_dataset['encoded_sequences'], maxlen=self.max_sequence_length, padding='post')
        X_val = pad_sequences(val_dataset['encoded_sequences'], maxlen=self.max_sequence_length, padding='post')

        # Define labels
        y_train = train_dataset['label'].values
        y_val = val_dataset['label'].values

        # Encode the labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)

        return X_train, X_val, y_train_encoded, y_val_encoded

    def build_model(self):
        # Build the CNN-LSTM model
        model = Sequential()
        model.add(Embedding(input_dim=self.num_unique_digits, output_dim=self.embedding_output_dim, input_length=self.max_sequence_length))
        model.add(Conv1D(filters=self.filters_1, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))
        model.add(Conv1D(filters=self.filters_2, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # For binary classification

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return self.model

    def train(self, X_train, y_train_encoded, X_val, y_val_encoded, epochs=10, batch_size=16):
        if self.model is None:
            raise ValueError("Model has not been built. Call 'build_model()' first.")
        
        # Train the CNN-LSTM model
        history = self.model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val_encoded))
        return history

    def evaluate(self, X_val, y_val_encoded):
        if self.model is None:
            raise ValueError("Model has not been built. Call 'build_model()' first.")
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_val, y_val_encoded)
        print(f"CNN-LSTM Model Validation Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("Model has not been built. Call 'build_model()' first.")
        
        # Save the model to a file
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    @staticmethod
    def load_saved_model(model_path):
        # Load a saved model
        return load_model(model_path)

    def predict(self, model, test_data_path, output_file_path):
        # Load test data
        test_data = pd.read_csv(test_data_path)
        test_data['encoded_sequences'] = test_data['input_str'].apply(lambda x: [int(digit) for digit in x])

        # Pad the test sequences
        X_test_padded = pad_sequences(test_data['encoded_sequences'], maxlen=self.max_sequence_length, padding='post')

        # Predict the labels for the test dataset
        y_test_pred = model.predict(X_test_padded)
        y_test_pred_labels = (y_test_pred > 0.5).astype(int).flatten()

        # Save the predictions
        with open(output_file_path, 'w') as f:
            f.write("Input String\tPredicted Label\n")
            for inp_str, pred_label in zip(test_data['input_str'], y_test_pred_labels):
                f.write(f"{inp_str}\t{pred_label}\n")
        
        print(f"Predictions saved to {output_file_path}")


class LogisticRegressionModel:
    def __init__(self, train_file_path, validation_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.validation_file_path = validation_file_path
        self.test_file_path = test_file_path
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()

    def load_data(self, file_path, features_key='features', label_key='label'):
        """Loads data from an npz file."""
        if not os.path.isfile(file_path) or not file_path.endswith(".npz"):
            raise FileNotFoundError(f"{file_path} does not exist or is not an npz file!")
        try:
            data = np.load(file_path)
            features = data[features_key]
            labels = data.get(label_key, None)  # Labels may not be available (e.g., in the test set)
            return features, labels
        except Exception as e:
            raise ValueError(f"Error loading the file: {e}")

    def preprocess_data(self, X):
        """Flattens and normalizes the feature matrix."""
        X_flattened = X.reshape(X.shape[0], -1)
        X_normalized = self.scaler.transform(X_flattened)
        return X_normalized

    def train_model(self, train_percentages=[0.2, 0.4, 0.6, 0.8, 1.0]):
        """Train the Logistic Regression model with different percentages of the training data."""
        # Load training and validation data
        X_train, y_train = self.load_data(self.train_file_path)
        X_val, y_val = self.load_data(self.validation_file_path)

        # Flatten and normalize the data
        X_train_flattened = X_train.reshape(X_train.shape[0], -1)
        X_val_flattened = X_val.reshape(X_val.shape[0], -1)
        self.scaler.fit(X_train_flattened)  # Fit scaler on training data
        X_train_normalized = self.scaler.transform(X_train_flattened)
        X_val_normalized = self.scaler.transform(X_val_flattened)

        # Store results
        results = []

        for pct in train_percentages:
            num_samples = int(pct * len(X_train_normalized))
            X_train_subset = X_train_normalized[:num_samples]
            y_train_subset = y_train[:num_samples]

            # Train the model
            self.model.fit(X_train_subset, y_train_subset)

            # Evaluate on the validation set
            y_val_pred = self.model.predict(X_val_normalized)
            accuracy = accuracy_score(y_val, y_val_pred)
            results.append(accuracy)
            print(f"Logistic Regression with {int(pct * 100)}% data: Accuracy = {accuracy:.4f}")

        # Plot the results
        self.plot_results(train_percentages, results)

        # Save the trained model
        joblib.dump(self.model, 'model_deep_feature.joblib')

    def plot_results(self, train_percentages, results):
        """Plot the accuracy results."""
        plt.figure(figsize=(12, 8))
        plt.plot([int(p * 100) for p in train_percentages], results, label='Logistic Regression', marker='o')
        plt.xlabel('Percentage of Training Data')
        plt.ylabel('Validation Accuracy')
        plt.title('Logistic Regression Performance with Varying Training Data Sizes')
        plt.legend()
        plt.grid()
        plt.savefig('logistic_regression_plot.png')  # Save instead of show
        plt.close()  # Close the plot so script continues

    def test_model(self, output_file_path='pred_deepfeat_class.txt'):
        """Test the saved model on the test dataset."""
        # Load the test data
        X_test, _ = self.load_data(self.test_file_path)

        # Flatten and normalize the test data
        X_test_flattened = X_test.reshape(X_test.shape[0], -1)
        X_test_normalized = self.scaler.transform(X_test_flattened)

        # Load the saved model
        self.model = joblib.load('model_deep_feature.joblib')

        # Predict the labels for the test dataset
        y_test_pred = self.model.predict(X_test_normalized)

        # Save predictions to a file
        with open(output_file_path, 'w') as f:
            f.write("Predicted Label\n")  # Write header
            for pred in y_test_pred:
                f.write(f"{pred}\n")  # Write each predicted label

        print(f"Predictions saved to {output_file_path}")

    def get_num_parameters(self):
        """Get the number of trainable parameters (weights + bias)."""
        num_features = self.model.coef_.shape[1]
        num_parameters = num_features + 1  # Weights + bias for Logistic Regression
        return num_parameters


class EmojiLSTMModel:
    def __init__(self, max_sequence_length=10, embedding_dim=20, lstm_units=16, dense_units=8):
        # Hyperparameters
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.model = None
        self.label_encoder = LabelEncoder()

    def train(self, train_path, valid_path, epochs=10, batch_size=32):
        # Load the training and validation datasets
        emoji_dataset_train = pd.read_csv(train_path)
        emoji_dataset_valid = pd.read_csv(valid_path)

        # Create a mapping for emojis to integers based on the training dataset
        emoji_to_int = {emoji: idx + 1 for idx, emoji in enumerate(set(''.join(emoji_dataset_train['input_emoticon'])))}

        # Function to convert emoji sequences to integer sequences
        def convert_to_integer_sequence(emojis):
            return [emoji_to_int[emoji] for emoji in emojis if emoji in emoji_to_int]

        # Transform the training dataset
        emoji_dataset_train['encoded_emoticons'] = emoji_dataset_train['input_emoticon'].apply(convert_to_integer_sequence)
        X_train = pad_sequences(emoji_dataset_train['encoded_emoticons'], maxlen=self.max_sequence_length, padding='post')
        y_train = emoji_dataset_train['label'].values

        # Transform the validation dataset
        emoji_dataset_valid['encoded_emoticons'] = emoji_dataset_valid['input_emoticon'].apply(convert_to_integer_sequence)
        X_valid = pad_sequences(emoji_dataset_valid['encoded_emoticons'], maxlen=self.max_sequence_length, padding='post')
        y_valid = emoji_dataset_valid['label'].values

        # Encode the labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_valid_encoded = self.label_encoder.transform(y_valid)

        # Build the LSTM model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(emoji_to_int) + 1, output_dim=self.embedding_dim, input_length=self.max_sequence_length))
        self.model.add(LSTM(self.lstm_units, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(self.dense_units, activation='relu'))
        self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))

        # Compile the LSTM model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the LSTM model
        self.model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid_encoded))

        # Evaluate the LSTM model
        lstm_loss, lstm_accuracy = self.model.evaluate(X_valid, y_valid_encoded)
        print(f"LSTM Model Validation Accuracy: {lstm_accuracy * 100:.2f}%")

        # Optionally: Print classification report for the LSTM model on validation data
        y_pred = np.argmax(self.model.predict(X_valid), axis=-1)
        print(classification_report(y_valid_encoded, y_pred, target_names=self.label_encoder.classes_.astype(str)))

        # Save the trained model
        self.model.save('lstm_model_emoticon.h5')
        print("Model saved as 'lstm_model_emoticon.h5'.")

    def predict(self, test_path, model_path='lstm_model_emoticon.h5'):
        # Load the saved model
        self.model = load_model(model_path)

        # Load the test dataset
        test_dataset = pd.read_csv(test_path)

        # Create a mapping for emojis to integers based on the training dataset
        emoji_to_int = {emoji: idx + 1 for idx, emoji in enumerate(set(''.join(pd.read_csv('datasets/train/train_emoticon.csv')['input_emoticon'])))}

        # Function to convert emoji sequences to integer sequences
        def convert_to_integer_sequence(emojis):
            return [emoji_to_int[emoji] for emoji in emojis if emoji in emoji_to_int]

        # Transform the test dataset
        test_dataset['encoded_emoticons'] = test_dataset['input_emoticon'].apply(convert_to_integer_sequence)
        X_test = pad_sequences(test_dataset['encoded_emoticons'], maxlen=self.max_sequence_length, padding='post')

        # Make predictions using the trained LSTM model
        predictions = np.argmax(self.model.predict(X_test), axis=-1)

        # Convert predicted class labels back to original string representations
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        # Create a DataFrame for the output
        output_df = pd.DataFrame({
            'input_emoticon': test_dataset['input_emoticon'],
            'predicted_label': predicted_labels
        })

        # Save the predictions to a text file
        output_df.to_csv('pred_emoticon_class.txt', index=False, header=True, sep='\t')  # Using tab as the separator
        print("Predictions saved to 'pred_emoticon_class.txt'.")


# Example Usage:
if __name__=='__main__':

  # Initialize the class                                          
  emoji_model = EmojiLSTMModel()

  # Call the train function
  emoji_model.train('datasets/train/train_emoticon.csv', 'datasets/valid/valid_emoticon.csv')

  # Call the predict function
  emoji_model.predict('datasets/test/test_emoticon.csv')
  

  train_file = 'datasets/train/train_feature.npz'
  val_file = 'datasets/valid/valid_feature.npz'
  test_file = 'datasets/test/test_feature.npz'
    
    # Instantiate the model class
  logistic_model = LogisticRegressionModel(train_file, val_file, test_file)
    
    # Train the model
  logistic_model.train_model()

    # Test the model
  logistic_model.test_model()
  print(f"trainable parameters:{logistic_model.get_num_parameters()}")
    
    

  

# Initialize the CNN-LSTM model
cnn_lstm = CNNLSTMModel(
    max_sequence_length=50,  # Adjust if needed
    num_unique_digits=10,    # Number of unique digits (0-9)
    embedding_output_dim=32,  # Embedding dimension
    filters_1=32,            # Number of filters in first Conv1D layer
    filters_2=8,             # Number of filters in second Conv1D layer
    dense_units=64           # Number of units in Dense layer
)

# Paths to training and validation datasets
train_data_path = 'datasets/train/train_text_seq.csv'
val_data_path = 'datasets/valid/valid_text_seq.csv'

# Load and preprocess the data
X_train, X_val, y_train_encoded, y_val_encoded = cnn_lstm.load_data(train_data_path, val_data_path)

# Build the CNN-LSTM model
cnn_lstm.build_model()

# Train the model (keeping epoch=50 and batch_size=16 as specified)
cnn_lstm.train(X_train, y_train_encoded, X_val, y_val_encoded, epochs=50, batch_size=16)

# Evaluate the model
cnn_lstm.evaluate(X_val, y_val_encoded)

# Save the trained model
cnn_lstm.save_model('cnn_lstm_model_text_seq.h5')

# Load the saved model for predictions
saved_model = CNNLSTMModel.load_saved_model('cnn_lstm_model_text_seq.h5')

# Path to the test dataset and output file for predictions
test_data_path = 'datasets/test/test_text_seq.csv'
output_file_path = 'pred_textseq_class.txt'

# Make predictions and save them to the output file
cnn_lstm.predict(saved_model, test_data_path, output_file_path)



# for combined model

# Instantiate the class
combined_model = COMBINED_MODEL()
combined_model.train()



