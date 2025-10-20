import pandas as pd

def convert_str_to_numeric(column):
    """
    Convert a pandas Series of string values to numeric dtype.

    - Strips whitespace from string values.
    - Converts values to numeric using pandas.to_numeric.
    - Non-convertible values are set to NaN (errors="coerce").

    Parameters
    ----------
    column : pandas.Series
        The input column containing string values.

    Returns
    -------
    pandas.Series
        Series with numeric dtype, where invalid parsing will produce NaN.
    """
    column.str.strip()
    new_column = column = pd.to_numeric(column, errors="coerce")
    return new_column


def fill_na_values_with_mean(column):
    """
    Fill missing values in a pandas Series with the column mean.

    Parameters
    ----------
    column : pandas.Series
        The input column, which may contain NaN values.

    Returns
    -------
    pandas.Series
        A new Series where all NaN values are replaced with the mean of the column.
    """
    new_column = column.fillna(value=column.mean())
    return new_column

def create_churn_flag(column):
    """
    Map a Series of 'Yes'/'No' values to a binary churn flag.

    This function converts categorical churn indicators to numeric labels,
    mapping 'Yes' -> 1 and 'No' -> 0. Values other than the exact strings
    'Yes' or 'No' are mapped to NaN.

    Parameters
    ----------
    column : pandas.Series
        Input Series containing churn indicators as strings ('Yes'/'No').

    Returns
    -------
    pandas.Series
        New Series of integers/floats where 'Yes' is 1, 'No' is 0, and any
        other values are NaN. The input is not modified.
    """
    column = column.map({'Yes': 1, 'No': 0})
    return column