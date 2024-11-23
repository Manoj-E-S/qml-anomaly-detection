import pytest

import cqu.preprocessing as cqupp

from . import smote_test_data_df, smote_test_data_result


def test_smote_valid_col():
    pp = cqupp.Preprocessor(smote_test_data_df)
    pp.standardize_numeric_data()
    pp.standardize_string_data({"status": cqupp.StringStandardizers.LABEL_BINARIZER})
    pp.smote_on_column(target_column="status", random_state=127, k_neighbors=4)

    assert pp.dataframe.to_dict(orient="list") == smote_test_data_result


def test_smote_exceptions():
    pp = cqupp.Preprocessor(smote_test_data_df)

    with pytest.raises(
        ValueError, match=r"Target column 'invalid_column' not found in DataFrame\."
    ):
        pp.smote_on_column(
            target_column="invalid_column", random_state=127, k_neighbors=4
        )

    with pytest.raises(
        ValueError,
        match=r"The following columns are non-numeric: \['status'\]\. SMOTE requires all feature columns to be numeric.",
    ):
        pp.smote_on_column(target_column="age", random_state=127, k_neighbors=4)

    with pytest.raises(
        ValueError,
        match="Expected n_neighbors <= n_samples_fit, but n_neighbors = 6, n_samples_fit = 5, n_samples = 5",
    ):
        pp.smote_on_column(target_column="status", random_state=127)
