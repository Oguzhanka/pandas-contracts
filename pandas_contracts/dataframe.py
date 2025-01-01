import typing

import pandas as pd

from pandas_contracts.core import DataFrameContract

"""
Module: pandas_contracts.extensions

This module provides extensions to the DataFrameContract class for validating
specific requirements such as column existence, data types, missing values, and
unique indices in pandas DataFrames.

Key Components:
- RequiresColumns: Ensures specific columns are present in a DataFrame.
- RequiresDtypes: Validates the data types of specified columns.
- RequiresNotNaN: Validates that certain columns have no missing values.
- RequiresUniqueIndex: Ensures that the index of a DataFrame contains only unique values.

Usage:
1. Extend `DataFrameContract` to define new validation requirements.
2. Use these validation classes with pandas DataFrames to enforce constraints.

Example:
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    validator = RequiresColumns(columns=['a', 'b'], coerce=True)
    check_pass, validated_df = validator(df)
"""


class RequiresColumns(DataFrameContract):
    def __init__(self, columns: list, coerce: bool):
        """
        Initialize the RequiresColumns validator.

        Args:
            columns (list): List of required column names.
            coerce (bool): Whether to add missing columns with null values.
        """
        self.required_columns = columns
        super().__init__(coerce, f"Columns {columns} are violated.")

    def forward(self, df: pd.DataFrame) -> bool:
        """
        Check if the required columns are present in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if all required columns are present, False otherwise.
        """
        return set(self.required_columns).issubset(set(df.columns))

    def _coerce_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add missing columns with null values to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to coerce.

        Returns:
            pd.DataFrame: The coerced DataFrame with missing columns added.
        """
        missing_columns = list(set(self.required_columns) - set(df.columns))
        return df.assign(**{col_name: None for col_name in missing_columns})


class RequiresDtypes(DataFrameContract):
    """
    Validates the data types of specified columns in the DataFrame.

    Optionally, columns with incorrect data types can be coerced to the required types.
    """

    def __init__(self, dtypes: dict[str, type], coerce: bool):
        """
        Initialize the RequiresDtypes validator.

        Args:
            dtypes (dict[str, type]): A mapping of column names to their required data types.
            coerce (bool): Whether to coerce columns to the required data types.
        """
        self.dtypes = dtypes
        super().__init__(coerce, f"Dtypes {dtypes} are violated.")

    def forward(self, df: pd.DataFrame) -> bool:
        """
        Check if the specified columns have the required data types.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if all specified columns have the required data types, False otherwise.
        """
        if set(self.dtypes.keys()).issubset(set(df.columns)):
            return all(df.dtypes[key] == self.dtypes[key] for key in self.dtypes.keys())
        return False

    def _coerce_input(self, df: pd.DataFrame) -> typing.Union[pd.DataFrame, bool]:
        """
        Coerce columns to the required data types.

        Args:
            df (pd.DataFrame): The DataFrame to coerce.

        Returns:
            pd.DataFrame or bool: The coerced DataFrame if successful, False otherwise.
        """
        try:
            return df.astype(self.dtypes)
        except Exception:
            return False


class RequiresNotNaN(DataFrameContract):
    """
    Validates that specific columns have no missing values.
    """

    def __init__(self, columns: list):
        """
        Initialize the RequiresNotNaN validator.

        Args:
            columns (list): List of columns to validate for missing values.
        """
        self.columns = columns
        super().__init__(coerce=False, warning_message="Columns contain NaN values.")

    def forward(self, df: pd.DataFrame) -> bool:
        """
        Check if the specified columns have no missing values.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if no missing values are found, False otherwise.
        """
        return all((~df[col].isna()).all() for col in self.columns)


class RequiresUniqueIndex(DataFrameContract):
    """
    Validates that the DataFrame index contains only unique values.
    """

    def __init__(self):
        """
        Initialize the RequiresUniqueIndex validator.
        """
        super().__init__(coerce=False, warning_message="Index contains duplicates.")

    def forward(self, df: pd.DataFrame) -> bool:
        """
        Check if the DataFrame index is unique.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if the index is unique, False otherwise.
        """
        return not df.index.duplicated().any()
