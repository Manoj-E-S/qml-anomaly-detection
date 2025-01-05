"""This is a typing module providing all type aliases and type hints for the cqu module
"""

from enum import Enum
from typing import TypeAlias

import pandas as pd
from numpy.typing import ArrayLike

from cqu.classical import ClassicalModels

ModelType: TypeAlias = ClassicalModels | str
Dataset: TypeAlias = ArrayLike | pd.DataFrame
