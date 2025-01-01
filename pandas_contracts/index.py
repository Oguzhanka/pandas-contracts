"""
Module: index

This module provides extensions to the `IndexContract` class for validating and coercing pandas
Index objects.

Key Components:
- RequiresUniqueIndex: Ensures that all values in the Index are unique, with optional coercion to
drop duplicates.
- RequiresMonotonicIndex: Validates that the Index is monotonic (increasing or decreasing), with
optional coercion to sort the Index.
- RequiresNonNegativeIndex: Ensures that all values in the Index are non-negative, with optional
coercion to replace negative values with zero.
- RequiresPositiveIndex: Validates that all values in the Index are strictly positive, with optional
 coercion to replace non-positive values with the smallest positive value.
- RequiresIndexNames: Validates that the Index has specific names, with optional coercion to set the
 required names.

Usage:
    These extensions are designed to validate and optionally coerce pandas Index objects
    to meet specific conditions, making them suitable for preprocessing and data integrity checks.

Example:
    ```python
    index = pd.Index([-1, 0, 1], name='example')
    validator = RequiresNonNegativeIndex()
    is_valid = validator.forward(index)
    print(is_valid)  # Output: False

    coerced_index = validator._coerce_input(index)
    print(coerced_index)  # Output: Int64Index([0, 0, 1], dtype='int64', name='example')
    ```
"""

from pandas_contracts.core import IndexContract
import pandas as pd


class RequiresUniqueIndex(IndexContract):
    """
    Validates that the pandas Index contains only unique values.
    """

    def forward(self, index: pd.Index) -> bool:
        """
        Check if the Index has unique values.

        Args:
            index (pd.Index): The Index to validate.

        Returns:
            bool: True if all values in the Index are unique, False otherwise.
        """
        return index.is_unique

    def _coerce_input(self, index: pd.Index) -> pd.Index:
        """
        Coerce the Index to unique values by dropping duplicates.

        Args:
            index (pd.Index): The Index to coerce.

        Returns:
            pd.Index: The coerced Index with duplicates removed.
        """
        return pd.Index(index.drop_duplicates())


class RequiresMonotonicIndex(IndexContract):
    """
    Validates that the pandas Index is monotonic (increasing or decreasing).
    """

    def forward(self, index: pd.Index) -> bool:
        """
        Check if the Index is monotonic (either increasing or decreasing).

        Args:
            index (pd.Index): The Index to validate.

        Returns:
            bool: True if the Index is monotonic, False otherwise.
        """
        return index.is_monotonic_increasing or index.is_monotonic_decreasing

    def _coerce_input(self, index: pd.Index) -> pd.Index:
        """
        Coerce the Index to be monotonic by sorting it.

        Args:
            index (pd.Index): The Index to coerce.

        Returns:
            pd.Index: The coerced, sorted Index.
        """
        return pd.Index(index.sort_values())


class RequiresNonNegativeIndex(IndexContract):
    """
    Validates that the pandas Index contains only non-negative values.
    """

    def forward(self, index: pd.Index) -> bool:
        """
        Check if the Index contains only non-negative values.

        Args:
            index (pd.Index): The Index to validate.

        Returns:
            bool: True if all values in the Index are non-negative, False otherwise.
        """
        return (index >= 0).all()

    def _coerce_input(self, index: pd.Index) -> pd.Index:
        """
        Coerce the Index by setting negative values to zero.

        Args:
            index (pd.Index): The Index to coerce.

        Returns:
            pd.Index: The coerced Index with negative values set to zero.
        """
        return pd.Index(index.map(lambda x: max(x, 0)))


class RequiresPositiveIndex(IndexContract):
    """
    Validates that the pandas Index contains only strictly positive values.
    """

    def forward(self, index: pd.Index) -> bool:
        """
        Check if the Index contains only strictly positive values.

        Args:
            index (pd.Index): The Index to validate.

        Returns:
            bool: True if all values in the Index are strictly positive, False otherwise.
        """
        return (index > 0).all()

    def _coerce_input(self, index: pd.Index) -> pd.Index:
        """
        Coerce the Index by setting non-positive values to the smallest positive value.

        Args:
            index (pd.Index): The Index to coerce.

        Returns:
            pd.Index: The coerced Index with non-positive values replaced.
        """
        min_positive = index[index > 0].min() if (index > 0).any() else 1
        return pd.Index(index.map(lambda x: x if x > 0 else min_positive))


class RequiresIndexNames(IndexContract):
    """
    Validates that the pandas Index has specific names.
    """

    def __init__(self, required_names: list):
        """
        Initialize the validator with required index names.

        Args:
            required_names (list): A list of required index names.
        """
        self.required_names = required_names
        super().__init__()

    def forward(self, index: pd.Index) -> bool:
        """
        Check if the Index contains the required names.

        Args:
            index (pd.Index): The Index to validate.

        Returns:
            bool: True if the Index names match the required names, False otherwise.
        """
        return index.names == self.required_names

    def _coerce_input(self, index: pd.Index) -> pd.Index:
        """
        Coerce the Index by setting the required names.

        Args:
            index (pd.Index): The Index to coerce.

        Returns:
            pd.Index: The coerced Index with the required names.
        """
        index.names = self.required_names
        return index
