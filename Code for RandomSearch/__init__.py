"""
src package

This package contains modules for data processing, model training, evaluation, and utility functions.
"""

from .data_processing import load_data, preprocess_data
from .model_training import train_without_tuning, train_with_tuning, save_models
from .evaluation import evaluate, save_metrics, plot_confusion_matrix, plot_roc_curve
from .utils import ensure_dir

__all__ = [
    "load_data",
    "preprocess_data",
    "train_without_tuning",
    "train_with_tuning",
    "save_models",
    "evaluate_model",
    "evaluate_saved_models",
    "save_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "ensure_dir",
]
