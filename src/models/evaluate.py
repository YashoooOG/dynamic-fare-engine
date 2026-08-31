"""
Evaluation metrics and benchmarking utilities for Regression and Classification models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_regression(y_true, y_pred, model_name: str = "Regression Model") -> dict:
    """Compute regression performance metrics: MAE, RMSE, R2 score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "model": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }
    return metrics


def evaluate_classification(y_true, y_pred_proba, threshold: float = 0.5, model_name: str = "Classifier") -> dict:
    """Compute binary classification metrics: ROC-AUC, Log Loss, Accuracy, F1."""
    y_pred_binary = (np.array(y_pred_proba) >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
    loss = log_loss(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    metrics = {
        "model": model_name,
        "roc_auc": float(auc),
        "log_loss": float(loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }
    return metrics
