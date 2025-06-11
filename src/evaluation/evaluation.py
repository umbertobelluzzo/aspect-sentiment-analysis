import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support


def evaluate_aspect_extractor(gold_path, evaluator):
    """
    Evaluate aspect extractor on gold standard data
    
    Parameters:
    -----------
    gold_path : str
        Path to gold standard CSV file
    evaluator : AspectSentimentAnalyzer
        Aspect extractor to evaluate
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    import os
    
    if os.path.exists(gold_path):
        gold_df = pd.read_csv(gold_path).dropna(subset=['sentence', 'aspects_pipe'])
        
        y_true, y_pred = [], []
        
        for _, row in gold_df.iterrows():
            sent = str(row['sentence'])
            true_set = set(str(row['aspects_pipe']).lower().split('|'))          # gold labels
            pred_set = set(evaluator.extract_aspects(sent).keys())              # system output
            
            # convert multilabel → binary-per-aspect
            for asp in evaluator.product_aspects:
                y_true.append(1 if asp in true_set else 0)
                y_pred.append(1 if asp in pred_set else 0)
        
        P, R, F1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        print(f"✅ Aspect extractor on gold-100 P = {P:.2f} R = {R:.2f} F1 = {F1:.2f}")
        
        return {'precision': P, 'recall': R, 'f1': F1}
    else:
        print("⚠️  Gold-set CSV not found – skipping intrinsic aspect evaluation")
        return None


def analyze_aspect_importance(fe, lr_results, X):
    """
    Analyze and visualize aspect importance from Logistic Regression coefficients

    Parameters:
    -----------
    fe : FeatureEngineer
        Feature engineering object
    lr_results : dict
        Results from logistic regression training
    X : scipy.sparse matrix
        Feature matrix
    """
    # Get aspect names
    aspect_names = fe.aspect_analyzer.get_aspect_names()

    # Extract model
    lr_model = lr_results['model']

    # Identify aspect feature indices
    total_feats = X.shape[1]
    N = len(aspect_names)
    aspect_idxs = list(range(total_feats - N, total_feats))

    # Pull out the logistic regression coefficients for aspect columns
    coefs = lr_model.coef_[0][aspect_idxs]

    # Create importance list
    importance = list(zip(aspect_names, coefs, np.abs(coefs)))
    importance.sort(key=lambda x: x[2], reverse=True)

    # Print ranked list
    print("Aspect importance (coef, abs_coef):\n")
    for name, coef, ab in importance:
        print(f"  {name:15s}  {coef:7.4f}  {ab:7.4f}")

    print(f"\n➡️ Most important: {importance[0][0]}")
    print(f"➡️ Least important: {importance[-1][0]}")

    # Bar plot of raw coefficients
    plt.figure(figsize=(8,4))
    names, rawcs, _ = zip(*importance)
    sns.barplot(x=list(names), y=list(rawcs))
    plt.title("LogReg Aspect Coefficients")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Heatmap of absolute importances
    abs_vals = np.array([ab for _,_,ab in importance]).reshape(1, -1)
    plt.figure(figsize=(8,2))
    sns.heatmap(abs_vals, annot=True, fmt=".3f",
                xticklabels=names, yticklabels=["abs_imp"])
    plt.title("Absolute Aspect Importance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return importance


def run_cross_validation(model_cls, X, y, model_kwargs):
    """
    Run 5-fold cross validation for a model
    
    Parameters:
    -----------
    model_cls : class
        Model class to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    model_kwargs : dict
        Model parameters
        
    Returns:
    --------
    tuple
        (accuracy_mean, accuracy_std, f1_mean, f1_std)
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import precision_recall_fscore_support
    
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


def print_classification_results(results, model_name):
    """
    Print classification results in a formatted way
    
    Parameters:
    -----------
    results : dict
        Model results dictionary
    model_name : str
        Name of the model
    """
    print(f"{model_name} Results:")
    print(pd.DataFrame(results['report']).transpose())
    
    if 'class_weights' in results:
        print(f"\nClass Weights:")
        print(results['class_weights'])
    elif 'sample_weights' in results:
        print(f"\nSample Weights:")
        print(results['sample_weights'])


def compare_model_performance(*model_results):
    """
    Compare performance across multiple models
    
    Parameters:
    -----------
    model_results : tuple of (model_name, results_dict)
        Multiple model results to compare
    """
    comparison_data = []
    
    for model_name, results in model_results:
        report = results['report']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': report['accuracy'],
            'Precision (Neg)': report['0']['precision'],
            'Recall (Neg)': report['0']['recall'],
            'F1 (Neg)': report['0']['f1-score'],
            'Precision (Pos)': report['1']['precision'],
            'Recall (Pos)': report['1']['recall'],
            'F1 (Pos)': report['1']['f1-score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("Model Performance Comparison:")
    print(comparison_df.round(4))