"""This is a util module mainly for the 'BaseClassifier' class and some other utility

This module provides the 'BaseClassifier' class along with some utility for internal 
use in the cqu module
"""

import os

PLOT_FOLDER_NAME = "cqu_plots"
os.makedirs(PLOT_FOLDER_NAME, exist_ok=True)

from .classifier import BaseClassifier
from .metrics import ClassifierMetrics, get_metrics
from .plotting import (
    plot_all_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_report,
    plot_roc_auc,
)
