import pytest

import cqu.preprocessing as cqupp

from . import test_data_MEAN_INT, test_dataframe


def test_int_conversion():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.FILL_MEAN)
    pp.convert_dtypes({"time": int, "salary": int})

    assert pp.dataframe.to_dict(orient="list") == test_data_MEAN_INT


def test_convert_invalid_column():
    pp = cqupp.Preprocessor(test_dataframe)

    with pytest.raises(
        ValueError, match="Column 'invalid_column' not found in DataFrame!"
    ):
        pp.convert_dtypes({"invalid_column": int})


def test_invalid_dtype_conversion():
    pp = cqupp.Preprocessor(test_dataframe)

    with pytest.raises(
        ValueError, match="Failed to convert column 'name' from object to int!"
    ):
        pp.convert_dtypes({"name": int})
