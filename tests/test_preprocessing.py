import pytest

import cqu.preprocessing as cqupp


def test_valid_path():
    pp = cqupp.Preprocessor("datasets/ccfraud/creditcard.csv")

    assert pp.dataframe is not None
    assert not pp.dataframe.empty


def test_invalid_path():
    invalid_path = "datasets/ccfraud/creditcard.txt"

    with pytest.raises(FileNotFoundError, match=f"File not found: {invalid_path}"):
        pp = cqupp.Preprocessor(invalid_path)


def test_unsupported_filetype():
    invalid_filetype = "datasets/eda.ipynb"

    with pytest.raises(
        ValueError, match=cqupp.unsupported_message.format(file_extension=".ipynb")
    ):
        pp = cqupp.Preprocessor(invalid_filetype)
