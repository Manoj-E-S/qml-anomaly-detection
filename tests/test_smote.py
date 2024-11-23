import pytest

import cqu.preprocessing as cqupp

from . import smote_test_data_df, smote_test_data_result


def test_smote_invalid_col():
    pp = cqupp.Preprocessor(smote_test_data_df)
    pp.standardize_numeric_data()
    pp.standardize_string_data({"status": cqupp.StringStandardizers.LABEL_BINARIZER})
    pp.smote_on_column(target_column="status", random_state=127, k_neighbors=4)

    assert pp.dataframe.to_dict(orient="list") == smote_test_data_result
