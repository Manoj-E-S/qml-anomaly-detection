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

Typical usage example:

>>> import cqu.preprocessing as cqupp
>>> pp = cqupp.Preprocessor("path/to/dataset")
>>> pp.get_missing_summary()
{ 'v1': 0, 'v2': 0, 'v3': 4, 'time': 6, 'class': 2 }

>>> strategies = { 
        'v3': cqupp.MissingValueStrategies.FILL_MEDIAN, 
        'time': cqupp.MissingValueStrategies.FILL_NOCB, 
        'class': cqupp.MissingValueStrategies.DROP_ROWS 
    }
>>> pp.clean_missing(strategies)
>>> pp.get_missing_summary()
{ 'v1': 0, 'v2': 0, 'v3': 0, 'time': 0, 'class': 0 }

"""

from enum import Enum


class ClassicalModels(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
