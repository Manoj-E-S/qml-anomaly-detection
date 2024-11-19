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
    test_data_ZERO,
    test_dataframe,
)


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
