import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils import resample


def cv_metrics(model_cls, X, y, **model_kwargs):
    """
    Cross-validation metrics helper function
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, fs = [], []
    for train_i, test_i in skf.split(X, y):
        Xtr, Xt = X[train_i], X[test_i]
        ytr, yt = y[train_i], y[test_i]
        m = model_cls(**model_kwargs).fit(Xtr, ytr)
        pred = m.predict(Xt)
        accs.append(m.score(Xt, yt))
        _, _, f1, _ = precision_recall_fscore_support(yt, pred, average='binary', pos_label=0)
        fs.append(f1)
    return np.mean(accs), np.std(accs), np.mean(fs), np.std(fs)


def train_logistic_regression(X, y):
    """
    Train Logistic Regression model with class balancing

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

    # Calculate class weights
    class_weights = {
        0: len(y[y == 1]) / len(y[y == 0]),  # Weight for negative class
        1: 1.0  # Weight for positive class
    }

    # Train Logistic Regression with balanced class weights
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight=class_weights,
        C=0.1  # Regularization strength
    )
    lr_model.fit(X_train, y_train)

    # Predictions
    y_pred = lr_model.predict(X_test)

    # Performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Logistic Regression Confusion Matrix\n(Hybrid Features)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('logistic_regression_confusion_matrix_hybrid.png')
    plt.close()

    # Probability Distribution Visualization
    plt.figure(figsize=(8, 6))
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    plt.hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]],
             label=['Negative', 'Positive'],
             bins=30,
             alpha=0.5)
    plt.title('Probability Distribution of Predictions\n(Hybrid Features)')
    plt.xlabel('Predicted Probability of Positive Class')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('logistic_regression_probability_distribution_hybrid.png')
    plt.close()

    return {
        'model': lr_model,
        'report': report,
        'class_weights': class_weights,
        'X_test': X_test
    }


def negativity_bias_ci(Xtest, proba_pos, aspect_names,
                       neg_thresh=-0.3, pos_thresh=0.2,
                       n_boot=1000, random_state=42):
    """
    Calculate negativity bias confidence interval
    """
    n_aspects   = len(aspect_names)
    aspect_cols = np.arange(Xtest.shape[1] - n_aspects, Xtest.shape[1])
    aspect_raw  = (Xtest[:, aspect_cols].toarray() * 2) - 1   # back to [-1,1]

    mask_neg = (aspect_raw <= neg_thresh).any(axis=1)
    mask_pos = (aspect_raw >=  pos_thresh).sum(axis=1) >= 2
    idx      = mask_neg & mask_pos

    base = proba_pos.mean()
    drops = proba_pos[idx] - base

    rng = np.random.RandomState(random_state)
    boots = [rng.choice(drops, size=drops.size, replace=True).mean()
             for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return drops.mean(), ci_lo, ci_hi, idx.sum()