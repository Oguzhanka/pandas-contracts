import pandas as pd
from pandas_contracts.series import (
    RequiresNonNegative,
    RequiresNotNaN,
    RequiresUniqueValues,
    RequiresPositive,
)


class TestRequiresNonNegative:
    """
    Test cases for the RequiresNonNegative validator.

    Ensures that all values in a Series are non-negative, with optional coercion.
    """

    def test_valid_non_negative(self):
        """
        Test that validation passes when all values are non-negative.
        """
        validator = RequiresNonNegative(coerce=False)
        series = pd.Series([0, 1, 2, 3])
        assert validator.forward(series)

    def test_negative_values_no_coerce(self):
        """
        Test that validation fails when negative values are present and coercion is disabled.
        """
        validator = RequiresNonNegative(coerce=False)
        series = pd.Series([-1, 0, 1])
        assert not validator.forward(series)

    def test_negative_values_with_coerce(self):
        """
        Test that negative values are coerced to zero when coercion is enabled.
        """
        validator = RequiresNonNegative(coerce=True)
        series = pd.Series([-1, 0, 1])
        coerced_series = validator._coerce_input(series)
        assert (coerced_series >= 0).all()


class TestRequiresNotNaN:
    """
    Test cases for the RequiresNotNaN validator.

    Ensures that no values in a Series are missing (NaN), with optional coercion.
    """

    def test_no_missing_values(self):
        """
        Test that validation passes when there are no missing values.
        """
        validator = RequiresNotNaN(coerce=False)
        series = pd.Series([1, 2, 3])
        assert validator.forward(series)

    def test_missing_values_no_coerce(self):
        """
        Test that validation fails when missing values are present and coercion is disabled.
        """
        validator = RequiresNotNaN(coerce=False)
        series = pd.Series([1, None, 3])
        assert not validator.forward(series)

    def test_missing_values_with_coerce(self):
        """
        Test that missing values are filled with zero when coercion is enabled.
        """
        validator = RequiresNotNaN(coerce=True)
        series = pd.Series([1, None, 3])
        coerced_series = validator._coerce_input(series)
        assert coerced_series.notna().all()
        assert coerced_series[1] == 0


class TestRequiresUniqueValues:
    """
    Test cases for the RequiresUniqueValues validator.

    Ensures that all values in a Series are unique, with optional coercion.
    """

    def test_unique_values(self):
        """
        Test that validation passes when all values are unique.
        """
        validator = RequiresUniqueValues(coerce=False)
        series = pd.Series([1, 2, 3])
        assert validator.forward(series)

    def test_duplicate_values_no_coerce(self):
        """
        Test that validation fails when duplicate values are present and coercion is disabled.
        """
        validator = RequiresUniqueValues(coerce=False)
        series = pd.Series([1, 1, 2])
        assert not validator.forward(series)

    def test_duplicate_values_with_coerce(self):
        """
        Test that duplicate values are removed when coercion is enabled.
        """
        validator = RequiresUniqueValues(coerce=True)
        series = pd.Series([1, 1, 2])
        coerced_series = validator._coerce_input(series)
        assert coerced_series.is_unique


class TestRequiresPositive:
    """
    Test cases for the RequiresPositive validator.

    Ensures that all values in a Series are strictly positive, with optional coercion.
    """

    def test_all_positive(self):
        """
        Test that validation passes when all values are strictly positive.
        """
        validator = RequiresPositive(coerce=False)
        series = pd.Series([1, 2, 3])
        assert validator.forward(series)

    def test_non_positive_values_no_coerce(self):
        """
        Test that validation fails when non-positive values are present and coercion is disabled.
        """
        validator = RequiresPositive(coerce=False)
        series = pd.Series([0, 1, -1])
        assert not validator.forward(series)

    def test_non_positive_values_with_coerce(self):
        """
        Test that non-positive values are replaced with the smallest positive value when coercion is
        enabled.
        """
        validator = RequiresPositive(coerce=True)
        series = pd.Series([0, -1, 2])
        coerced_series = validator._coerce_input(series)
        assert (coerced_series > 0).all()
        assert coerced_series[0] > 0  # Coerced value
