import cqu.preprocessing as cqupp

from . import test_data_MEAN_INT, test_dataframe


def test_int_conversion():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.FILL_MEAN)
    pp.convert_datatypes({"time": int, "salary": int})

    assert pp.dataframe.to_dict(orient="list") == test_data_MEAN_INT
