import pandas as pd
from pandas_contracts.dataframe import (
    RequiresColumns,
    RequiresDtypes,
    RequiresNotNaN,
    RequiresUniqueIndex,
)


class TestRequiresColumns:
    """
    Test cases for the RequiresColumns validator.

    Ensures that specified columns are present in the DataFrame, with optional coercion.
    """

    def test_valid_columns(self):
        """
        Test that the validation passes when all required columns are present.
        """
        validator = RequiresColumns(columns=["a", "b"], coerce=False)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert validator.forward(df) is True

    def test_missing_columns_no_coerce(self):
        """
        Test that validation fails when required columns are missing and coercion is disabled.
        """
        validator = RequiresColumns(columns=["a", "b", "c"], coerce=False)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert validator.forward(df) is False

    def test_missing_columns_with_coerce(self):
        """
        Test that missing columns are added with null values when coercion is enabled.
        """
        validator = RequiresColumns(columns=["a", "b", "c"], coerce=True)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        coerced_df = validator._coerce_input(df)
        assert "c" in coerced_df.columns
        assert coerced_df["c"].isnull().all()


class TestRequiresDtypes:
    """
    Test cases for the RequiresDtypes validator.

    Ensures that specified columns have the required data types, with optional coercion.
    """

    def test_valid_dtypes(self):
        """
        Test that validation passes when all columns have the correct data types.
        """
        validator = RequiresDtypes(dtypes={"a": int, "b": float}, coerce=False)
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert validator.forward(df) is True

    def test_invalid_dtypes_no_coerce(self):
        """
        Test that validation fails when columns have incorrect data types and coercion is disabled.
        """
        validator = RequiresDtypes(dtypes={"a": int, "b": float}, coerce=False)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})  # 'b' is int, not float
        assert validator.forward(df) is False

    def test_invalid_dtypes_with_coerce(self):
        """
        Test that columns are coerced to the correct data types when coercion is enabled.
        """
        validator = RequiresDtypes(dtypes={"a": int, "b": float}, coerce=True)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        coerced_df = validator._coerce_input(df)
        assert coerced_df["b"].dtype == float


class TestRequiresNotNaN:
    """
    Test cases for the RequiresNotNaN validator.

    Ensures that specified columns contain no missing values.
    """

    def test_no_missing_values(self):
        """
        Test that validation passes when all specified columns have no missing values.
        """
        validator = RequiresNotNaN(columns=["a", "b"])
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert validator.forward(df) is True

    def test_with_missing_values(self):
        """
        Test that validation fails when specified columns contain missing values.
        """
        validator = RequiresNotNaN(columns=["a", "b"])
        df = pd.DataFrame({"a": [1, 2], "b": [3, None]})
        assert validator.forward(df) is False


class TestRequiresUniqueIndex:
    """
    Test cases for the RequiresUniqueIndex validator.

    Ensures that the DataFrame index contains only unique values.
    """

    def test_unique_index(self):
        """
        Test that validation passes when the DataFrame index is unique.
        """
        validator = RequiresUniqueIndex()
        df = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        assert validator.forward(df) is True

    def test_duplicate_index(self):
        """
        Test that validation fails when the DataFrame index contains duplicates.
        """
        validator = RequiresUniqueIndex()
        df = pd.DataFrame({"a": [1, 2]}, index=[0, 0])
        assert validator.forward(df) is False
