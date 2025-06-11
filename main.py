#!/usr/bin/env python3
"""
Main script to run the aspect-based sentiment analysis pipeline
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Import custom modules
from src.data_preprocessing import process_amazon_reviews, setup_data_directories
from src.aspect_extractor import AspectSentimentAnalyzer, download_opinion_lexicon, test_opinion_lexicon
from src.feature_engineering import FeatureEngineer
from src.models.logistic_regression import train_logistic_regression, cv_metrics, negativity_bias_ci
from src.models.naive_bayes import train_naive_bayes
from src.models.lstm_classifier import WordEmbeddingLSTMClassifier
from src.evaluation.evaluation import (
    evaluate_aspect_extractor, 
    analyze_aspect_importance, 
    run_cross_validation,
    print_classification_results,
    compare_model_performance
)


def run_traditional_ml_pipeline(file_path):
    """
    Run the traditional ML pipeline (Logistic Regression + Naive Bayes)
    """
    print("=" * 60)
    print("TRADITIONAL ML PIPELINE")
    print("=" * 60)
    
    # Load processed reviews
    df = pd.read_csv(file_path)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Prepare features
    X = feature_engineer.prepare_features(df['review_text'])
    y = df['sentiment'].map({'positive': 1, 'negative': 0}).values
    
    # Compute class weights
    pos_cnt = (y == 1).sum()
    neg_cnt = (y == 0).sum()
    class_weights = {0: pos_cnt / neg_cnt, 1: 1.0}
    
    print(f"Total number of features: {X.shape[1]}")
    print(f"TF-IDF + Aspect features shape: {X.shape}")
    
    # === LOGISTIC REGRESSION ===
    print("\n" + "-" * 40)
    print("LOGISTIC REGRESSION")
    print("-" * 40)
    
    # Train Logistic Regression
    lr_results = train_logistic_regression(X, y)
    
    # Cross-validation for Logistic Regression
    lr_acc_mean, lr_acc_sd, lr_f1_mean, lr_f1_sd = cv_metrics(
        LogisticRegression,
        X, y,
        class_weight=class_weights,
        max_iter=1000,
        C=0.1,
        random_state=42
    )
    
    print(f"5‑fold CV Logistic: Acc = {lr_acc_mean:.3f} ± {lr_acc_sd:.3f}, "
          f"F1_neg = {lr_f1_mean:.3f} ± {lr_f1_sd:.3f}")
    
    # Negativity bias analysis
    X_test_lr = lr_results['X_test']
    proba_test = lr_results['model'].predict_proba(X_test_lr)[:,1]
    mean_drop, ci_lo, ci_hi, n_cases = negativity_bias_ci(
        X_test_lr, proba_test,
        feature_engineer.aspect_analyzer.get_aspect_names())
    
    print(f"\nNegativity-bias drop = {mean_drop*100:.1f} pp "
          f"(95 % CI {ci_lo*100:.1f}, {ci_hi*100:.1f}; n={n_cases})")
    
    print_classification_results(lr_results, "Logistic Regression (Hybrid Features)")
    
    # === NAIVE BAYES ===
    print("\n" + "-" * 40)
    print("NAIVE BAYES")
    print("-" * 40)
    
    # Train Naive Bayes
    nb_results = train_naive_bayes(X, y)
    
    # Cross-validation for Naive Bayes
    nb_acc_mean, nb_acc_sd, nb_f1_mean, nb_f1_sd = cv_metrics(
        MultinomialNB,
        X, y,
        alpha=0.1
    )
    
    print(f"5‑fold CV Naive Bayes: Acc = {nb_acc_mean:.3f} ± {nb_acc_sd:.3f}, "
          f"F1_neg = {nb_f1_mean:.3f} ± {nb_f1_sd:.3f}")
    
    print_classification_results(nb_results, "Naive Bayes (Hybrid Features)")
    
    # === ASPECT IMPORTANCE ANALYSIS ===
    print("\n" + "-" * 40)
    print("ASPECT IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    importance = analyze_aspect_importance(feature_engineer, lr_results, X)
    
    return {
        'feature_engineer': feature_engineer,
        'lr_results': lr_results,
        'nb_results': nb_results,
        'X': X, 'y': y
    }


def run_deep_learning_pipeline(file_path):
    """
    Run the deep learning pipeline (LSTM)
    """
    print("\n" + "=" * 60)
    print("DEEP LEARNING PIPELINE")
    print("=" * 60)
    
    # Load processed reviews
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: CSV file not found. Please check the file path.")
        return None
    
    # Prepare data
    X = df['review_text']
    y = (df['sentiment'] == 'positive').astype(int)
    
    # Initialize and train LSTM model
    lstm_classifier = WordEmbeddingLSTMClassifier()
    lstm_results = lstm_classifier.train_model(X, y)
    
    print_classification_results(lstm_results, "LSTM Model (Embedding Features)")
    
    # Analyze embedding importance
    embedding_insights = lstm_classifier.analyze_embedding_importance(X)
    
    # Print top words by embedding magnitude
    print("\nTop 10 Words by Embedding Magnitude:")
    for word, magnitude in list(embedding_insights['top_words'].items())[:10]:
        print(f"{word}: {magnitude:.4f}")
    
    # Optional: Save model for future use
    lstm_classifier.model.save('word_embedding_lstm_model.h5')
    
    return {
        'lstm_classifier': lstm_classifier,
        'lstm_results': lstm_results,
        'embedding_insights': embedding_insights
    }


def main():
    """
    Main function to run the complete pipeline
    """
    print("ASPECT-BASED SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Download required NLTK data
    print("Downloading required NLTK data...")
    download_opinion_lexicon()
    test_opinion_lexicon()
    
    # Set up file paths
    file_path = '/content/drive/MyDrive/nlp_summative/amazon_data/processed_electronics_reviews.csv'
    gold_path = '/content/drive/MyDrive/nlp_summative/aspect_gold100.csv'
    
    # Evaluate aspect extractor if gold standard is available
    print("\n" + "-" * 40)
    print("ASPECT EXTRACTOR EVALUATION")
    print("-" * 40)
    
    evaluator = AspectSentimentAnalyzer()
    aspect_eval_results = evaluate_aspect_extractor(gold_path, evaluator)
    
    # Run traditional ML pipeline
    traditional_results = run_traditional_ml_pipeline(file_path)
    
    # Run deep learning pipeline
    deep_learning_results = run_deep_learning_pipeline(file_path)
    
    # Compare all models
    if traditional_results and deep_learning_results:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        compare_model_performance(
            ("Logistic Regression", traditional_results['lr_results']),
            ("Naive Bayes", traditional_results['nb_results']),
            ("LSTM", deep_learning_results['lstm_results'])
        )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()