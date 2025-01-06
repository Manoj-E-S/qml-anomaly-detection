"""This is a typing module providing all type aliases and type hints for the cqu module
"""

from enum import Enum
from typing import TypeAlias

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from torch import nn


class ClassicalModels(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    ENSEMBLE = "ensemble"


class QuantumModels(Enum):
    QSVM = "qsvm"


ClassicalModelTypes: TypeAlias = (
    BaseEstimator | nn.Module
)  # SKLearn Models | PyTorch neural networks
ModelType: TypeAlias = ClassicalModels | QuantumModels | str
Dataset: TypeAlias = ArrayLike | pd.DataFrame
