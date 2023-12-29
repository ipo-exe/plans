import os
import pandas as pd


def assert_columns_in_dataframe(parser_data, expected_columns):
    # Check if the 'data' attribute is a DataFrame
    assert isinstance(parser_data, pd.DataFrame)

    # Check if the 'data' DataFrame has exactly the expected columns
    actual_columns = list(parser_data.columns)
    msg = f"The 'data' DataFrame should have columns {expected_columns} and no others."
    assert actual_columns == expected_columns, msg


def assert_dtypes_in_dataframe(parser_data, expected_dtypes):
    # Check the dtypes for each column in the 'data' DataFrame
    for col, valid_types in expected_dtypes.items():
        actual_dtype = parser_data[col].dtype
        # Check if the actual dtype is a subclass of any valid type
        valid_dtype = any(actual_dtype == t for t in valid_types)
        msg = f"The '{col}' column in the 'data' DataFrame should have dtype {valid_types}."
        assert valid_dtype, msg


def assert_file_exists(file_path):
    """Assert that the given file path points to an existing file."""
    msg = f"The file path '{file_path}' does not point to an existing file."
    assert os.path.isfile(file_path), msg
