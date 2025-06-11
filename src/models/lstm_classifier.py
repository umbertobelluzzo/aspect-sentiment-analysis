import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class WordEmbeddingLSTMClassifier:
    def __init__(self,
                 embedding_dim=100,  # Reduced embedding dimension
                 max_sequence_length=200,  # Maximum review length
                 max_vocab_size=20000):  # Maximum vocabulary size
        """
        Initialize Word Embedding LSTM Classifier

        Parameters:
        -----------
        embedding_dim : int, optional (default=100)
            Dimension of word embeddings
        max_sequence_length : int, optional (default=200)
            Maximum number of words in a review
        max_vocab_size : int, optional (default=20000)
            Maximum number of unique words to use
        """
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.max_vocab_size = max_vocab_size

        # Placeholders for key components
        self.tokenizer = None
        self.embedding_matrix = None
        self.model = None

    def create_embedding_matrix(self, word_index):
        """
        Create embedding matrix using simple initialization strategy

        Parameters:
        -----------
        word_index : dict
            Mapping of words to their integer indices

        Returns:
        --------
        np.ndarray
            Embedding matrix for model initialization
        """
        # Create embedding matrix
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))

        # Initialize with random values
        np.random.seed(42)
        for word, i in word_index.items():
            if i >= self.max_vocab_size:
                continue
            # Randomly initialize embedding vector
            embedding_matrix[i] = np.random.normal(
                size=(self.embedding_dim,)
            )

        return embedding_matrix

    def prepare_text_data(self, reviews):
        """
        Tokenize and prepare text data for LSTM

        Parameters:
        -----------
        reviews : list of str
            Raw review texts

        Returns:
        --------
        tuple
            (padded sequences, tokenizer, word_index)
        """
        # Preprocess reviews to handle potential None or NaN values
        reviews = [str(review).lower() for review in reviews]

        # Initialize tokenizer
        self.tokenizer = Tokenizer(
            num_words=self.max_vocab_size,
            oov_token='<OOV>'
        )

        # Fit tokenizer on reviews
        self.tokenizer.fit_on_texts(reviews)

        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(reviews)

        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        # Get word index
        word_index = self.tokenizer.word_index

        return padded_sequences, word_index

    def build_lstm_model(self, input_length, embedding_matrix):
        """
        Construct LSTM model with embeddings

        Parameters:
        -----------
        input_length : int
            Maximum sequence length
        embedding_matrix : np.ndarray
            Word embedding matrix

        Returns:
        --------
        tf.keras.Model
            Compiled LSTM model
        """
        model = Sequential([
            # Embedding layer with initialized weights
            Embedding(
                input_dim=embedding_matrix.shape[0],
                output_dim=self.embedding_dim,
                weights=[embedding_matrix],
                input_length=input_length,
                trainable=True  # Allow fine-tuning
            ),
            # LSTM layer with dropout for regularization
            LSTM(128, return_sequences=True),
            Dropout(0.5),
            LSTM(64),
            Dropout(0.5),
            # Dense layers for classification
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        # Compile model with balanced learning rate
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, X, y):
        """
        Train LSTM model with embeddings

        Parameters:
        -----------
        X : list of str
            Review texts
        y : np.ndarray
            Sentiment labels

        Returns:
        --------
        dict
            Training results and model performance
        """
        # Prepare text data
        X_seq, word_index = self.prepare_text_data(X)

        # Create embedding matrix
        self.embedding_matrix = self.create_embedding_matrix(word_index)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y, test_size=0.2, random_state=42, stratify=y
        )

        # Build LSTM model
        self.model = self.build_lstm_model(
            input_length=self.max_sequence_length,
            embedding_matrix=self.embedding_matrix
        )

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Calculate class weights
        class_weights = {
            0: len(y[y == 1]) / len(y[y == 0]),  # Weight for negative class
            1: 1.0  # Weight for positive class
        }

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=64,
            callbacks=[early_stopping],
            class_weight=class_weights
        )

        # Evaluate model
        y_pred_proba = self.model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion Matrix Visualization
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('LSTM Model Confusion Matrix\n(Embedding Features)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('lstm_confusion_matrix.png')
        plt.close()

        # Probability Distribution Visualization
        plt.figure(figsize=(8, 6))
        # Flatten the arrays to ensure 1D
        negative_probs = y_pred_proba[y_test == 0]
        positive_probs = y_pred_proba[y_test == 1]

        plt.hist([negative_probs, positive_probs],
                 label=['Negative', 'Positive'],
                 bins=30,
                 alpha=0.5)
        plt.title('Probability Distribution of Predictions\n(Embedding LSTM)')
        plt.xlabel('Predicted Probability of Positive Class')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig('lstm_probability_distribution.png')
        plt.close()

        return {
            'model': self.model,
            'report': report,
            'history': history.history,
            'class_weights': class_weights
        }

    def analyze_embedding_importance(self, X):
        """
        Analyze importance of word embeddings

        Parameters:
        -----------
        X : list of str
            Review texts

        Returns:
        --------
        dict
            Insights about word embedding importance
        """
        # Prepare text data
        X_seq, word_index = self.prepare_text_data(X)

        # Get top important words based on embedding characteristics
        inv_word_index = {v: k for k, v in word_index.items()}

        # Extract first layer weights (embedding layer)
        if self.model is None:
            raise ValueError("Model must be trained first")

        # Visualize top words by their embedding magnitude
        embedding_layer = self.model.layers[0]
        embeddings = embedding_layer.get_weights()[0]

        # Compute word vector norms (magnitude)
        word_norms = np.linalg.norm(embeddings, axis=1)

        # Sort words by their embedding magnitude
        top_word_indices = np.argsort(word_norms)[::-1][:50]

        top_words = [inv_word_index.get(idx, f'Unknown_{idx}') for idx in top_word_indices]
        top_norms = word_norms[top_word_indices]

        # Visualization of top word embedding magnitudes
        plt.figure(figsize=(12, 6))
        plt.bar(top_words, top_norms)
        plt.title('Top 50 Words by Embedding Magnitude')
        plt.xlabel('Words')
        plt.ylabel('Embedding Vector Magnitude')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('top_words_embedding_magnitude.png')
        plt.close()

        return {
            'top_words': dict(zip(top_words, top_norms.tolist())),
            'embedding_dimension': self.embedding_dim
        }