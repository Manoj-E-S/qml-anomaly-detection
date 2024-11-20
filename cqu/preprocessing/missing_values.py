from enum import Enum
from typing import Dict, overload

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


def is_strategy_applicable(column: str, strategy: MissingValueStrategies) -> bool:
    is_numeric = pd.api.types.is_numeric_dtype(column)

    if strategy in {
        MissingValueStrategies.FILL_ZERO,
        MissingValueStrategies.FILL_MEAN,
        MissingValueStrategies.FILL_MEDIAN,
        MissingValueStrategies.FILL_MODE,
        MissingValueStrategies.FILL_LERP,
    }:
        return is_numeric

    return True


def handle_single_strategy(
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
            return dataframe.ffill()
        case MissingValueStrategies.FILL_NOCB:
            return dataframe.bfill()
        case _:
            raise ValueError("Invalid strategy. Please provide a valid strategy.")


@overload
def handle_missing_values(
    dataframe: pd.DataFrame, strategy: MissingValueStrategies
) -> pd.DataFrame: ...


@overload
def handle_missing_values(
    dataframe: pd.DataFrame, strategy: Dict[str, MissingValueStrategies]
) -> pd.DataFrame: ...


def handle_missing_values(
    dataframe: pd.DataFrame,
    strategy_or_strategies: MissingValueStrategies | Dict[str, MissingValueStrategies],
) -> pd.DataFrame:
    if isinstance(strategy_or_strategies, MissingValueStrategies):
        dataframe = handle_single_strategy(dataframe, strategy_or_strategies)
    elif isinstance(strategy_or_strategies, dict):
        for column, strategy in strategy_or_strategies.items():
            if column not in dataframe.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame!")

            if strategy == MissingValueStrategies.DROP_ROWS:
                dataframe = dataframe.dropna(subset=[column])
            elif strategy == MissingValueStrategies.DROP_COLUMNS:
                dataframe = dataframe.drop(columns=[column])
            else:
                if not is_strategy_applicable(dataframe[column], strategy):
                    raise ValueError(
                        f"Strategy '{strategy}' is not applicable on non-numeric column '{column}'"
                    )

                dataframe[column] = handle_single_strategy(dataframe[column], strategy)
    else:
        raise ValueError(
            "Invalid input: Provide a single strategy or a dictionary of column-specific strategies."
        )

    return dataframe
