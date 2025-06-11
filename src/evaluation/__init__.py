"""
Evaluation utilities for sentiment analysis models
"""

from .evaluation import (
    evaluate_aspect_extractor,
    analyze_aspect_importance,
    run_cross_validation,
    print_classification_results,
    compare_model_performance
)

__all__ = [
    'evaluate_aspect_extractor',
    'analyze_aspect_importance', 
    'run_cross_validation',
    'print_classification_results',
    'compare_model_performance'
]