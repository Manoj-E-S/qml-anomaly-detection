from enum import Enum

import pandas as pd


class MissingValueStrategies(Enum):
    DROP_ROWS = "drop_rows"
    DROP_COLUMNS = "drop_columns"
    FILL_ZERO = "fill_zero"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_LERP = "fill_lerp"
    FILL_LOCF = "fill_locf"
    FILL_NOCB = "fill_nocb"


def handle_missing_values(
    dataframe: pd.DataFrame, strategy: MissingValueStrategies
) -> pd.DataFrame:
    match strategy:
        case MissingValueStrategies.DROP_ROWS:
            return dataframe.dropna(axis=0)
        case MissingValueStrategies.DROP_COLUMNS:
            return dataframe.dropna(axis=1)
        case MissingValueStrategies.FILL_ZERO:
            return dataframe.fillna(0)
        case MissingValueStrategies.FILL_MEAN:
            return dataframe.fillna(dataframe.mean(numeric_only=True))
        case MissingValueStrategies.FILL_MEDIAN:
            return dataframe.fillna(dataframe.median(numeric_only=True))
        case MissingValueStrategies.FILL_MODE:
            return dataframe.fillna(dataframe.mode().iloc[0])
        case MissingValueStrategies.FILL_LERP:
            return dataframe.interpolate(method="linear")
        case MissingValueStrategies.FILL_LOCF:
            return dataframe.fillna(method="ffill")
        case MissingValueStrategies.FILL_NOCB:
            return dataframe.fillna(method="bfill")
        case _:
            raise ValueError("Invalid strategy. Please provide a valid strategy.")
