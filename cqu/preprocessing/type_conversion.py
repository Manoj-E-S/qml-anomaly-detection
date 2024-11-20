from typing import Any, Dict

import numpy as np
import pandas as pd


def convert_types(
    dataframe: pd.DataFrame, column_types: Dict[str, Any]
) -> pd.DataFrame:
    for column, dtype in column_types.items():
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame!")
        try:
            dataframe[column] = dataframe[column].astype(dtype)
        except ValueError as e:
            raise ValueError(
                f"Failed to convert column '{column}' to {dtype.__name__}: {e}"
            )

    return dataframe
