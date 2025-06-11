import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


def train_naive_bayes(X, y):
    """
    Train Naive Bayes model with sample weighting

    Parameters:
    -----------
    X : scipy.sparse matrix
        Feature matrix
    y : np.ndarray
        Sentiment labels

    Returns:
    --------
    dict
        Model performance metrics and trained model
    """
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate sample weights
    class_counts = np.bincount(y_train)
    sample_weights = np.zeros_like(y_train, dtype=float)
    sample_weights[y_train == 0] = class_counts[1] / class_counts[0]
    sample_weights[y_train == 1] = 1.0

    # Train Naive Bayes with sample weights
    nb_model = MultinomialNB(alpha=0.1)  # Add Laplace smoothing
    nb_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Predictions
    y_pred = nb_model.predict(X_test)

    # Performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Naive Bayes Confusion Matrix\n(Hybrid Features)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('naive_bayes_confusion_matrix_hybrid.png')
    plt.close()

    # Probability Distribution Visualization
    plt.figure(figsize=(8, 6))
    y_pred_proba = nb_model.predict_proba(X_test)[:, 1]
    plt.hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]],
             label=['Negative', 'Positive'],
             bins=30,
             alpha=0.5)
    plt.title('Probability Distribution of Predictions\n(Hybrid Features)')
    plt.xlabel('Predicted Probability of Positive Class')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('naive_bayes_probability_distribution_hybrid.png')
    plt.close()

    return {
        'model': nb_model,
        'report': report,
        'sample_weights': {
            'negative_class_weight': class_counts[1] / class_counts[0],
            'positive_class_weight': 1.0
        }
    }