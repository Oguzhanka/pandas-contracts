import pandas as pd
from pandas_contracts.index import (
    RequiresUniqueIndex,
    RequiresMonotonicIndex,
    RequiresNonNegativeIndex,
    RequiresPositiveIndex,
    RequiresIndexNames,
)


class TestRequiresUniqueIndex:
    """
    Test cases for the RequiresUniqueIndex validator.

    Ensures that the pandas Index contains only unique values, with optional coercion.
    """

    def test_unique_index(self):
        """
        Test that validation passes when the Index has unique values.
        """
        validator = RequiresUniqueIndex()
        index = pd.Index([1, 2, 3])
        assert validator.forward(index) is True

    def test_duplicate_index_no_coerce(self):
        """
        Test that validation fails when duplicates are present and coercion is disabled.
        """
        validator = RequiresUniqueIndex()
        index = pd.Index([1, 1, 2])
        assert validator.forward(index) is False

    def test_duplicate_index_with_coerce(self):
        """
        Test that duplicates are removed when coercion is enabled.
        """
        validator = RequiresUniqueIndex()
        index = pd.Index([1, 1, 2])
        coerced_index = validator._coerce_input(index)
        assert coerced_index.is_unique


class TestRequiresMonotonicIndex:
    """
    Test cases for the RequiresMonotonicIndex validator.

    Ensures that the pandas Index is monotonic, with optional coercion.
    """

    def test_monotonic_increasing_index(self):
        """
        Test that validation passes for a monotonic increasing Index.
        """
        validator = RequiresMonotonicIndex()
        index = pd.Index([1, 2, 3])
        assert validator.forward(index) is True

    def test_non_monotonic_index_no_coerce(self):
        """
        Test that validation fails for a non-monotonic Index and coercion is disabled.
        """
        validator = RequiresMonotonicIndex()
        index = pd.Index([3, 1, 2])
        assert validator.forward(index) is False

    def test_non_monotonic_index_with_coerce(self):
        """
        Test that the Index is sorted to be monotonic when coercion is enabled.
        """
        validator = RequiresMonotonicIndex()
        index = pd.Index([3, 1, 2])
        coerced_index = validator._coerce_input(index)
        assert coerced_index.is_monotonic_increasing


class TestRequiresNonNegativeIndex:
    """
    Test cases for the RequiresNonNegativeIndex validator.

    Ensures that the pandas Index contains only non-negative values, with optional coercion.
    """

    def test_non_negative_index(self):
        """
        Test that validation passes when all values are non-negative.
        """
        validator = RequiresNonNegativeIndex()
        index = pd.Index([0, 1, 2])
        assert validator.forward(index)

    def test_negative_index_no_coerce(self):
        """
        Test that validation fails when negative values are present and coercion is disabled.
        """
        validator = RequiresNonNegativeIndex()
        index = pd.Index([-1, 0, 1])
        assert not validator.forward(index)

    def test_negative_index_with_coerce(self):
        """
        Test that negative values are set to zero when coercion is enabled.
        """
        validator = RequiresNonNegativeIndex()
        index = pd.Index([-1, 0, 1])
        coerced_index = validator._coerce_input(index)
        assert (coerced_index >= 0).all()


class TestRequiresPositiveIndex:
    """
    Test cases for the RequiresPositiveIndex validator.

    Ensures that the pandas Index contains only strictly positive values, with optional coercion.
    """

    def test_positive_index(self):
        """
        Test that validation passes when all values are strictly positive.
        """
        validator = RequiresPositiveIndex()
        index = pd.Index([1, 2, 3])
        assert validator.forward(index)

    def test_non_positive_index_no_coerce(self):
        """
        Test that validation fails when non-positive values are present and coercion is disabled.
        """
        validator = RequiresPositiveIndex()
        index = pd.Index([0, -1, 2])
        assert not validator.forward(index)

    def test_non_positive_index_with_coerce(self):
        """
        Test that non-positive values are replaced with the smallest positive value when coercion is
        enabled.
        """
        validator = RequiresPositiveIndex()
        index = pd.Index([0, -1, 2])
        coerced_index = validator._coerce_input(index)
        assert (coerced_index > 0).all()


class TestRequiresIndexNames:
    """
    Test cases for the RequiresIndexNames validator.

    Ensures that the pandas Index has specific names, with optional coercion.
    """

    def test_valid_index_names(self):
        """
        Test that validation passes when the Index names match the required names.
        """
        validator = RequiresIndexNames(required_names=["index_name"])
        index = pd.Index([1, 2, 3], name="index_name")
        assert validator.forward(index) is True

    def test_invalid_index_names_no_coerce(self):
        """
        Test that validation fails when the Index names do not match the required names and coercion
        is disabled.
        """
        validator = RequiresIndexNames(required_names=["index_name"])
        index = pd.Index([1, 2, 3], name="wrong_name")
        assert validator.forward(index) is False

    def test_invalid_index_names_with_coerce(self):
        """
        Test that the Index names are set to the required names when coercion is enabled.
        """
        validator = RequiresIndexNames(required_names=["index_name"])
        index = pd.Index([1, 2, 3], name="wrong_name")
        coerced_index = validator._coerce_input(index)
        assert coerced_index.names == ["index_name"]
