from typing import List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def standardize_numeric(
    dataframe: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    if columns is None:
        numeric_columns = dataframe.select_dtypes(include=["number"]).columns
        dataframe[numeric_columns] = scaler.fit_transform(dataframe[numeric_columns])
    else:
        for column in columns:
            if column not in dataframe.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(dataframe[column]):
                raise ValueError(
                    f"Column '{column}' is not numeric and cannot be standardized numerically."
                )

            dataframe[column] = scaler.fit_transform(dataframe[column])

    return dataframe
