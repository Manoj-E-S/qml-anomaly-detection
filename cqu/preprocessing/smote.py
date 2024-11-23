from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE


def smote_on_column(
    dataframe: pd.DataFrame, target_column: str, random_state: int = None
) -> pd.DataFrame:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    non_numeric_columns = X.select_dtypes(exclude=["number"]).columns
    if not non_numeric_columns.empty:
        raise ValueError(
            f"The following columns are non-numeric: {list(non_numeric_columns)}. "
            "SMOTE requires all feature columns to be numeric."
        )

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[target_column] = y_resampled

    return resampled_df
