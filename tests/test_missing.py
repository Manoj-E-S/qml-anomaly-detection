import cqu.preprocessing as cqupp

from . import test_data_DCOLS, test_data_DROWS, test_dataframe


def test_drop_rows():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_ROWS)

    assert pp.dataframe.to_dict(orient="list") == test_data_DROWS


def test_drop_columns():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_COLUMNS)

    assert pp.dataframe.to_dict(orient="list") == test_data_DCOLS
