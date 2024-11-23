import pytest

import cqu.preprocessing as cqupp

from . import (
    test_data_DCOLS,
    test_data_DROWS,
    test_data_LERP,
    test_data_LOCF,
    test_data_MEAN,
    test_data_MEDIAN,
    test_data_MODE,
    test_data_NOCB,
    test_data_SAL_LERP_TIME_DROP,
    test_data_SAL_LERP_TIME_DROP_ROWS,
    test_data_SAL_LERP_TIME_NOCB,
    test_data_ZERO,
    test_dataframe,
)


def test_invalid_strategy():
    pp = cqupp.Preprocessor(test_dataframe)

    with pytest.raises(
        ValueError,
        match="Invalid input: Provide a single strategy or a dictionary of column-specific strategies.",
    ):
        pp.clean_missing("invalid_strategy")


def test_drop_rows():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_ROWS)

    assert pp.dataframe.to_dict(orient="list") == test_data_DROWS


def test_drop_columns():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_COLUMNS)

    assert pp.dataframe.to_dict(orient="list") == test_data_DCOLS


def test_fill_zero():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.FILL_ZERO)

    assert pp.dataframe.to_dict(orient="list") == test_data_ZERO


def test_fill_mean():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.FILL_MEAN)

    assert pp.dataframe.to_dict(orient="list") == test_data_MEAN


def test_fill_median():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.FILL_MEDIAN)

    assert pp.dataframe.to_dict(orient="list") == test_data_MEDIAN


def test_fill_mode():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.FILL_MODE)

    assert pp.dataframe.to_dict(orient="list") == test_data_MODE


def test_fill_lerp():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.FILL_LERP)

    assert pp.dataframe.to_dict(orient="list") == test_data_LERP


def test_fill_locf():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.FILL_LOCF)

    assert pp.dataframe.to_dict(orient="list") == test_data_LOCF


def test_fill_nocb():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.FILL_NOCB)

    assert pp.dataframe.to_dict(orient="list") == test_data_NOCB


def test_get_missing_summary():
    pp = cqupp.Preprocessor(test_dataframe)

    assert pp.get_missing_summary() == {
        "name": 0,
        "age": 0,
        "origin_country": 0,
        "salary": 1,
        "company": 0,
        "time": 2,
    }


def test_invalid_column():
    pp = cqupp.Preprocessor(test_dataframe)

    strategies = {
        "invalid_column": cqupp.MissingValueStrategies.FILL_ZERO,
    }

    with pytest.raises(
        ValueError, match="Column 'invalid_column' not found in DataFrame!"
    ):
        pp.clean_missing(strategies)


def test_nonnumeric_col_numeric_strategy():
    pp = cqupp.Preprocessor(test_dataframe)

    strategies = {
        "name": cqupp.MissingValueStrategies.FILL_ZERO,
    }

    with pytest.raises(
        ValueError,
        match="Strategy 'MissingValueStrategies.FILL_ZERO' is not applicable on non-numeric column 'name'",
    ):
        pp.clean_missing(strategies)


def test_fill_lerp_nocb():
    pp = cqupp.Preprocessor(test_dataframe)

    strategies = {
        "salary": cqupp.MissingValueStrategies.FILL_LERP,
        "time": cqupp.MissingValueStrategies.FILL_NOCB,
    }

    pp.clean_missing(strategies)

    assert pp.dataframe.to_dict(orient="list") == test_data_SAL_LERP_TIME_NOCB


def test_fill_lerp_dropcol():
    pp = cqupp.Preprocessor(test_dataframe)

    strategies = {
        "salary": cqupp.MissingValueStrategies.FILL_LERP,
        "time": cqupp.MissingValueStrategies.DROP_COLUMNS,
    }

    pp.clean_missing(strategies)

    assert pp.dataframe.to_dict(orient="list") == test_data_SAL_LERP_TIME_DROP


def test_fill_lerp_droprows():
    pp = cqupp.Preprocessor(test_dataframe)

    strategies = {
        "salary": cqupp.MissingValueStrategies.FILL_LERP,
        "time": cqupp.MissingValueStrategies.DROP_ROWS,
    }

    pp.clean_missing(strategies)

    assert pp.dataframe.to_dict(orient="list") == test_data_SAL_LERP_TIME_DROP_ROWS
