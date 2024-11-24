import inspect
import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, overload

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

            dataframe[column] = scaler.fit_transform(dataframe[column].to_frame())

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
    if remove_special_chars:
        s = re.sub(r"[^a-zA-Z0-9_\s]", "", s)
    s = s.strip().lower().replace(" ", "_")
    return s


def apply_standardizer(
    series: pd.Series, standardizer: StringStandardizers
) -> pd.Series | pd.DataFrame | Tuple[pd.Series, Dict[str, int]]:
    match standardizer:
        case StringStandardizers.CLEAN:
            return series.apply(lambda x: clean_string(x, remove_special_chars=True))
        case StringStandardizers.CLEAN_KEEP_SPECIAL_CHARS:
            return series.apply(lambda x: clean_string(x, remove_special_chars=False))
        case StringStandardizers.LABEL_ENCODING:
            le = LabelEncoder()
            encoded_series = pd.Series(
                le.fit_transform(series.astype(str)), index=series.index
            )
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            return encoded_series, mapping
        case StringStandardizers.LABEL_BINARIZER:
            lb = LabelBinarizer()
            encoded = lb.fit_transform(series.astype(str))
            if encoded.shape[1] == 1:
                mapping = {cls: i for i, cls in enumerate(lb.classes_)}
                return pd.Series(encoded.flatten(), index=series.index), mapping
            else:
                encoded_df = pd.DataFrame(
                    encoded, index=series.index, columns=lb.classes_
                )
                return encoded_df
        case StringStandardizers.ONE_HOT_ENCODING:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            ohe_df = pd.DataFrame(
                ohe.fit_transform(series.astype(str).to_frame()),
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
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    mappings = {}

    def standardizer_side_effects(
        result: pd.Series | pd.DataFrame | Tuple[pd.Series, Dict[str, int]],
        colname: str,
        stdizer: StringStandardizers,
    ) -> None:
        nonlocal dataframe, mappings

        match stdizer:
            case StringStandardizers.LABEL_ENCODING:
                dataframe[colname] = result[0]
                mappings[colname] = result[1]
            case StringStandardizers.LABEL_BINARIZER:
                if isinstance(result, tuple):
                    dataframe[colname] = result[0]
                    mappings[colname] = result[1]
                else:
                    dataframe = pd.concat([dataframe, result], axis=1)
                    dataframe = dataframe.drop(colname, axis=1)
            case StringStandardizers.ONE_HOT_ENCODING:
                dataframe = pd.concat([dataframe, result], axis=1)
                dataframe = dataframe.drop(colname, axis=1)
            case _:
                dataframe[colname] = result
        pass

    if isinstance(standardizer, StringStandardizers):
        string_columns = dataframe.select_dtypes(include=["object"]).columns
        for col in string_columns:
            result = apply_standardizer(dataframe[col], standardizer)
            standardizer_side_effects(result, col, standardizer)
    elif isinstance(standardizer, dict):
        for col, std in standardizer.items():
            if col not in dataframe.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            if not pd.api.types.is_string_dtype(dataframe[col]):
                raise ValueError(
                    f"Column '{col}' is not string and cannot be standardized."
                )
            result = apply_standardizer(dataframe[col], std)
            standardizer_side_effects(result, col, std)
    else:
        raise ValueError(
            "Invalid input: provide a single StringStandardizer or a dictionary of column-specific standardizers."
        )

    return dataframe, mappings


def filter_columns(
    dataframe: pd.DataFrame, columns: Dict[str, Callable[[Any], bool]]
) -> pd.DataFrame:
    if not isinstance(columns, dict):
        raise ValueError(
            "Invalid input. Please provide a dictionary of column names and filter functions."
        )

    for column, condition in columns.items():
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        if not callable(condition):
            raise ValueError(
                f"Condition for column '{column}' must be a callable function or lambda."
            )

        sig = inspect.signature(condition)
        if len(sig.parameters) != 1:
            raise ValueError(
                f"The condition for column '{column}' must take exactly one parameter."
            )

        dataframe = dataframe[dataframe[column].apply(condition)]

    return dataframe
