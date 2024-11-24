import pytest

import cqu.preprocessing as cqupp

from . import test_data_numeric_std, test_dataframe, test_string_data_df


def test_standardization():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_ROWS)
    pp.standardize_numeric_data()

    assert pp.dataframe.to_dict(orient="list") == test_data_numeric_std


def test_standardization_columns():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_ROWS)
    pp.standardize_numeric_data(["age", "salary", "time"])

    assert pp.dataframe.to_dict(orient="list") == test_data_numeric_std


def test_invalid_col_standardization():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_ROWS)

    with pytest.raises(
        ValueError,
        match="Column 'name' is not numeric and cannot be standardized numerically.",
    ):
        pp.standardize_numeric_data(["name"])


def test_string_std_clean():
    pp = cqupp.Preprocessor(test_string_data_df)
    pp.standardize_string_data()

    assert pp.dataframe.to_dict(orient="list") == {
        "name": ["john_doe34", "jane_doe", "4john_smith", "john_doe34"],
        "age": [25, 30, 35, 25],
        "city": ["new_york", "los_angeles", "chicago", "new_york"],
        "working": ["no", "no", "yes", "no"],
    }


def test_string_std_clean_keep_spc():
    pp = cqupp.Preprocessor(test_string_data_df)
    pp.standardize_string_data(cqupp.StringStandardizers.CLEAN_KEEP_SPECIAL_CHARS)

    assert pp.dataframe.to_dict(orient="list") == {
        "name": ["john_doe34", "jane_doe___!", "4john_smith", "john_doe34"],
        "age": [25, 30, 35, 25],
        "city": ["new_york", "los_angeles~", "chicago!", "new_york"],
        "working": ["no", "no", "yes", "no"],
    }


def test_string_std_label_encode():
    pp = cqupp.Preprocessor(test_string_data_df)
    pp.standardize_string_data()
    pp.standardize_string_data(cqupp.StringStandardizers.LABEL_ENCODING)

    assert pp.dataframe.to_dict(orient="list") == {
        "name": [2, 1, 0, 2],
        "age": [25, 30, 35, 25],
        "city": [2, 1, 0, 2],
        "working": [0, 0, 1, 0],
    }

    assert pp.label_mappings == {
        "name": {"4john_smith": 0, "jane_doe": 1, "john_doe34": 2},
        "city": {"chicago": 0, "los_angeles": 1, "new_york": 2},
        "working": {"no": 0, "yes": 1},
    }


def test_string_std_label_binarizer():
    pp = cqupp.Preprocessor(test_string_data_df)
    pp.standardize_string_data()
    pp.standardize_string_data({"working": cqupp.StringStandardizers.LABEL_BINARIZER})

    assert pp.dataframe.to_dict(orient="list") == {
        "name": ["john_doe34", "jane_doe", "4john_smith", "john_doe34"],
        "age": [25, 30, 35, 25],
        "city": ["new_york", "los_angeles", "chicago", "new_york"],
        "working": [0, 0, 1, 0],
    }

    assert pp.label_mappings == {
        "working": {"no": 0, "yes": 1},
    }


def test_string_std_label_binarizer_multiclass():
    pp = cqupp.Preprocessor(test_string_data_df)
    pp.standardize_string_data()
    pp.standardize_string_data({"name": cqupp.StringStandardizers.LABEL_BINARIZER})

    assert pp.dataframe.to_dict(orient="list") == {
        "age": [25, 30, 35, 25],
        "city": ["new_york", "los_angeles", "chicago", "new_york"],
        "working": ["no", "no", "yes", "no"],
        "4john_smith": [0.0, 0.0, 1.0, 0.0],
        "jane_doe": [0.0, 1.0, 0.0, 0.0],
        "john_doe34": [1.0, 0.0, 0.0, 1.0],
    }


def test_string_std_onehotencoding():
    pp = cqupp.Preprocessor(test_string_data_df)
    pp.standardize_string_data()
    pp.standardize_string_data({"name": cqupp.StringStandardizers.ONE_HOT_ENCODING})

    assert pp.dataframe.to_dict(orient="list") == {
        "age": [25, 30, 35, 25],
        "city": ["new_york", "los_angeles", "chicago", "new_york"],
        "working": ["no", "no", "yes", "no"],
        "4john_smith": [0.0, 0.0, 1.0, 0.0],
        "jane_doe": [0.0, 1.0, 0.0, 0.0],
        "john_doe34": [1.0, 0.0, 0.0, 1.0],
    }


def test_filter_func():
    pp = cqupp.Preprocessor(test_string_data_df)
    pp.standardize_string_data()
    pp.filter_columns({"age": lambda x: x > 25})

    assert pp.dataframe.to_dict(orient="list") == {
        "name": ["jane_doe", "4john_smith"],
        "age": [30, 35],
        "city": ["los_angeles", "chicago"],
        "working": ["no", "yes"],
    }


def test_filter_exceptions():
    pp = cqupp.Preprocessor(test_string_data_df)
    pp.standardize_string_data()

    with pytest.raises(
        ValueError,
        match="Invalid input. Please provide a dictionary of column names and filter functions.",
    ):
        pp.filter_columns(lambda x: x > 25)

    with pytest.raises(ValueError, match="Column 'age1' not found in DataFrame."):
        pp.filter_columns({"age1": lambda x: x > 25})

    with pytest.raises(
        ValueError,
        match="Condition for column 'age' must be a callable function or lambda.",
    ):
        pp.filter_columns({"age": 23})

    with pytest.raises(
        ValueError,
        match="The condition for column 'age' must take exactly one parameter.",
    ):
        pp.filter_columns({"age": lambda x, y: x > 25})
