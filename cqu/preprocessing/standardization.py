import re
from enum import Enum
from typing import Dict, List, Optional, overload

import pandas as pd
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

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


class StringStandardizers(Enum):
    CLEAN = "clean"
    CLEAN_KEEP_SPECIAL_CHARS = "cksp"
    LABEL_ENCODING = "le"
    LABEL_BINARIZER = "lb"
    ONE_HOT_ENCODING = "ohe"


def clean_string(s: str, remove_special_chars: bool = True) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip().lower()
    if remove_special_chars:
        s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
    return s


def apply_standardizer(
    series: pd.Series, standardizer: StringStandardizers
) -> pd.Series | pd.DataFrame:
    match standardizer:
        case StringStandardizers.CLEAN:
            return series.apply(lambda x: clean_string(x, remove_special_chars=True))
        case StringStandardizers.CLEAN_KEEP_SPECIAL_CHARS:
            return series.apply(lambda x: clean_string(x, remove_special_chars=False))
        case StringStandardizers.LABEL_ENCODING:
            le = LabelEncoder()
            return pd.Series(le.fit_transform(series.astype(str)), index=series.index)
        case StringStandardizers.LABEL_BINARIZER:
            lb = LabelBinarizer()
            encoded = lb.fit_transform(series.astype(str))
            if encoded.shape[1] == 1:
                return pd.Series(encoded.flatten(), index=series.index)
            else:
                encoded_df = pd.DataFrame(
                    encoded, index=series.index, columns=lb.classes_
                )
                return encoded_df
        case StringStandardizers.ONE_HOT_ENCODING:
            ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
            ohe_df = pd.DataFrame(
                ohe.fit_transform(series.astype(str.to_frame())),
                index=series.index,
                columns=ohe.categories_[0],
            )
            return ohe_df
        case _:
            raise ValueError(
                "Invalid standardizer. Please provide a valid standardizer."
            )


@overload
def standardize_strings(
    dataframe: pd.DataFrame, standardizer: StringStandardizers
) -> pd.DataFrame: ...


@overload
def standardize_strings(
    dataframe: pd.DataFrame, standardizers: Dict[str, StringStandardizers]
) -> pd.DataFrame: ...


def standardize_strings(
    dataframe: pd.DataFrame,
    standardizer: StringStandardizers | Dict[str, StringStandardizers],
) -> pd.DataFrame:
    if isinstance(standardizer, StringStandardizers):
        string_columns = dataframe.select_dtypes(include=["object"]).columns
        for col in string_columns:
            dataframe[col] = apply_standardizer(dataframe[col], standardizer)
    elif isinstance(standardizer, dict):
        for col, std in standardizer.items():
            if col not in dataframe.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            if not pd.api.types.is_string_dtype(dataframe[col]):
                raise ValueError(
                    f"Column '{col}' is not string and cannot be standardized."
                )
            dataframe[col] = apply_standardizer(dataframe[col], std)
    else:
        raise ValueError(
            "Invalid input: provide a single StringStandardizer or a dictionary of column-specific standardizers."
        )

    return dataframe
