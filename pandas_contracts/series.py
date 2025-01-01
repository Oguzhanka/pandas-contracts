"""
Module: series

This module provides extensions to the `SeriesContract` class for validating and coercing pandas
Series objects.

Key Components:
- RequiresNonNegative: Ensures all values in a Series are non-negative, with optional coercion to
set negative values to zero.
- RequiresNotNaN: Validates that a Series contains no missing (NaN) values, with optional coercion
to fill missing
values.
- RequiresUniqueValues: Ensures that all values in a Series are unique, with optional coercion to
drop duplicates.
- RequiresPositive: Validates that all values in a Series are strictly positive, with optional
coercion to replace non-positive values with a minimum positive value.

Usage:
    These extensions are designed to validate and optionally coerce pandas Series objects
    to meet specific conditions, making them suitable for preprocessing and data integrity checks.

Example:
    ```python
    series = pd.Series([-1, 0, 1])
    validator = RequiresNonNegative(coerce=True)
    valid_series = validator(series)
    print(valid_series)
    ```
"""

from pandas_contracts.core import SeriesContract
import pandas as pd


class RequiresNonNegative(SeriesContract):
    """
    Validates that all values in a Series are non-negative.
    """

    def forward(self, series: pd.Series) -> bool:
        """
        Check if all values in the Series are non-negative.

        Args:
            series (pd.Series): The Series to validate.

        Returns:
            bool: True if all values are non-negative, False otherwise.
        """
        return (series >= 0).all()

    def _coerce_input(self, series: pd.Series) -> pd.Series:
        """
        Coerce negative values to zero in the Series.

        Args:
            series (pd.Series): The Series to coerce.

        Returns:
            pd.Series: The coerced Series with negative values set to zero.
        """
        return series.clip(lower=0)


class RequiresNotNaN(SeriesContract):
    """
    Validates that a Series contains no missing (NaN) values.
    """

    def forward(self, series: pd.Series) -> bool:
        """
        Check if the Series contains no missing values.

        Args:
            series (pd.Series): The Series to validate.

        Returns:
            bool: True if no missing values are present, False otherwise.
        """
        return series.notna().all()

    def _coerce_input(self, series: pd.Series) -> pd.Series:
        """
        Fill missing values in the Series with a default value (0).

        Args:
            series (pd.Series): The Series to coerce.

        Returns:
            pd.Series: The coerced Series with missing values filled.
        """
        return series.fillna(0)


class RequiresUniqueValues(SeriesContract):
    """
    Validates that all values in a Series are unique.
    """

    def forward(self, series: pd.Series) -> bool:
        """
        Check if all values in the Series are unique.

        Args:
            series (pd.Series): The Series to validate.

        Returns:
            bool: True if all values are unique, False otherwise.
        """
        return series.is_unique

    def _coerce_input(self, series: pd.Series) -> pd.Series:
        """
        Drop duplicate values from the Series.

        Args:
            series (pd.Series): The Series to coerce.

        Returns:
            pd.Series: The coerced Series with duplicates removed.
        """
        return series.drop_duplicates()


class RequiresPositive(SeriesContract):
    """
    Validates that all values in a Series are strictly positive.
    """

    def forward(self, series: pd.Series) -> bool:
        """
        Check if all values in the Series are strictly positive.

        Args:
            series (pd.Series): The Series to validate.

        Returns:
            bool: True if all values are positive, False otherwise.
        """
        return (series > 0).all()

    def _coerce_input(self, series: pd.Series) -> pd.Series:
        """
        Replace non-positive values with the smallest positive value in the Series.
        If the Series is empty or has no positive values, fill with 1.

        Args:
            series (pd.Series): The Series to coerce.

        Returns:
            pd.Series: The coerced Series with non-positive values replaced.
        """
        min_positive = series[series > 0].min() if (series > 0).any() else 1
        return series.apply(lambda x: x if x > 0 else min_positive)
