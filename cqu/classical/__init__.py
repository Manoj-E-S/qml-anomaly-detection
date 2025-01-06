"""The classical_models module with tools to test your dataset on various types of classical ML models for CQU.

This module provides the following functions:
a. To train and analyze various models:
    1. logistic_regression_with_analysis, 
    2. random_forest_with_analysis, 
    3. gradient_boosting_with_analysis, 
    4. knn_model_with_analysis, 
    5. naive_bayes_model_with_analysis
b. To be able to select important features
    1. get_feature_importance
"""

from cqu.typing import ClassicalModels

from .important_features import (
    get_feature_importances,
    gradient_boosting_importance,
    knn_importance,
    logistic_regression_importance,
    naive_bayes_importance,
    neural_network_importance,
    random_forest_importance,
)
from .models import (
    ensemble_model_with_analysis,
    get_the_best_classical_model,
    gradient_boosting_with_analysis,
    knn_model_with_analysis,
    logistic_regression_with_analysis,
    naive_bayes_model_with_analysis,
    neural_network_with_analysis,
    random_forest_with_analysis,
)
