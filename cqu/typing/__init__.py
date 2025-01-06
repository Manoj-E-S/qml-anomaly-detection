"""This is a typing module providing all type aliases and type hints for the cqu module
"""

from enum import Enum
from typing import TypeAlias

import pandas as pd
from numpy.typing import ArrayLike


class ClassicalModels(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"


ModelType: TypeAlias = ClassicalModels | str
Dataset: TypeAlias = ArrayLike | pd.DataFrame
