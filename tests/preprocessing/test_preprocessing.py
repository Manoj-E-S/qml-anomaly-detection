import os

import pandas as pd
import pytest

import cqu.preprocessing as cqupp

from . import test_dataframe


def test_valid_path():
    pp = cqupp.Preprocessor("datasets/ccfraud/creditcard.csv")

    assert pp.dataframe is not None
    assert not pp.dataframe.empty


def test_invalid_path():
    invalid_path = "datasets/ccfraud/creditcard.txt"

    with pytest.raises(FileNotFoundError, match=f"File not found: {invalid_path}"):
        pp = cqupp.Preprocessor(invalid_path)


def test_invalid_keep_value():
    invalid_keep_value = "middle"

    with pytest.raises(
        ValueError,
        match="Invalid value for keep_duplicates. Please provide 'first', 'last', or False.",
    ):
        pp = cqupp.Preprocessor("datasets/ccfraud/creditcard.csv", invalid_keep_value)


def test_unsupported_filetype():
    invalid_filetype = "datasets/eda.ipynb"

    with pytest.raises(
        ValueError, match=cqupp.unsupported_message.format(file_extension=".ipynb")
    ):
        pp = cqupp.Preprocessor(invalid_filetype)


def test_dataframe_input():
    pp = cqupp.Preprocessor(test_dataframe)

    assert pp.dataframe is not None
    assert not pp.dataframe.empty


def test_invalid_input_type():
    invalid_input = 123

    with pytest.raises(
        ValueError,
        match="Invalid input type. Please provide a file path or a DataFrame.",
    ):
        pp = cqupp.Preprocessor(invalid_input)


def test_invalid_write_to():
    pp = cqupp.Preprocessor("datasets/ccfraud/creditcard.csv")
    output_file = "datasets/ccfraud/creditcard.docx"

    with pytest.raises(
        ValueError, match=cqupp.unsupported_message.format(file_extension=".docx")
    ):
        pp.write_to(output_file)


def test_valid_write_to():
    pp = cqupp.Preprocessor("datasets/ccfraud/creditcard.csv")
    output_file = "datasets/ccfraud/creditcard.json"

    pp.write_to(output_file)

    assert os.path.exists(output_file)

    os.remove(output_file)


def test_handle_columns():
    pp = cqupp.Preprocessor(test_dataframe)

    assert pp.dataframe.columns.tolist() == [
        "name",
        "age",
        "origin_country",
        "salary",
        "company",
        "time",
    ]


# MANUAL TEST, CHECK THE cqu_logs FOLDER TO CHECK THE GENERATED LOG FILE AND CHECK FOR ANY ERRORS
def test_log_generation():
    pp = cqupp.Preprocessor(test_dataframe)
    pp.clean_missing(cqupp.MissingValueStrategies.DROP_ROWS)
    pp.standardize_numeric_data(["salary"])
    pp.standardize_string_data()
    pp.standardize_string_data({"company": cqupp.StringStandardizers.ONE_HOT_ENCODING})
    pp.filter_columns({"age": lambda x: x > 18})

    pp.write_to("cqu_logs/test_log_dataset.csv")
    pp.generate_logfile()
