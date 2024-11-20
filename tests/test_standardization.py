import pytest

import cqu.preprocessing as cqupp

from . import test_data_numeric_std, test_dataframe


def test_standardization():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_ROWS)
    pp.standardize_numeric_data()

    assert pp.dataframe.to_dict(orient="list") == test_data_numeric_std


def test_invalid_col_standardization():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_ROWS)

    with pytest.raises(
        ValueError,
        match="Column 'name' is not numeric and cannot be standardized numerically.",
    ):
        pp.standardize_numeric_data(["name"])
