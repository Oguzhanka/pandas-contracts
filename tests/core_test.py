import pytest
import pandas as pd
from abc import ABC

from pandas_contracts.options import COERCION_FAILURE_MSG
from pandas_contracts.core import (
    _PandasDataContract,
    IndexContract,
    SeriesContract,
    DataFrameContract,
    requires,
)


# -- Base Class Contract Testing --
# Mock implementation for testing purposes
class PandasDataContract(_PandasDataContract, ABC):
    def forward(self, data):
        # For test purposes, consider data valid if it's a DataFrame
        return isinstance(data, pd.DataFrame)


@pytest.fixture
def pandas_data_contract():
    return PandasDataContract(coerce=True, warning_message="Test warning message")


class Test_PandasDataContract:
    def test_forward_method(self, pandas_data_contract):
        """
        Test the forward method of PandasDataContract.

        - Ensures that valid data (DataFrame) passes the forward check.
        - Ensures that invalid data (non-DataFrame) fails the forward check.
        """
        valid_data = pd.DataFrame({"a": [1, 2, 3]})
        invalid_data = pd.Series([1, 2, 3])

        # Should return True for valid DataFrame
        assert pandas_data_contract.forward(valid_data) is True

        # Should return False for invalid input
        assert pandas_data_contract.forward(invalid_data) is False

    def test_call_with_valid_data(self, pandas_data_contract):
        """
        Test the __call__ method with valid data.

        - Ensures that valid data passes the check.
        - Ensures the data remains unchanged after processing.
        """
        valid_data = pd.DataFrame({"a": [1, 2, 3]})

        check_pass, processed_data = pandas_data_contract(valid_data)

        # Check that the valid data passes and remains unchanged
        assert check_pass is True
        pd.testing.assert_frame_equal(processed_data, valid_data)

    def test_call_with_invalid_data_coerce(self, pandas_data_contract):
        """
        Test the __call__ method with invalid data when coercion is enabled.

        - Ensures that a warning is issued.
        - Ensures that coercion fails as per the default implementation.
        """
        invalid_data = pd.Series([1, 2, 3])

        with pytest.warns(UserWarning, match="Test warning message"):
            check_pass, processed_data = pandas_data_contract(invalid_data)

        # Check that coercion fails (default implementation)
        assert check_pass is False
        assert processed_data is False

    def test_call_with_invalid_data_no_coerce(self):
        """
        Test the __call__ method with invalid data when coercion is disabled.

        - Ensures that invalid data fails directly without coercion.
        - Ensures that the data remains unchanged.
        """
        contract = PandasDataContract(
            coerce=False, warning_message="Test warning message"
        )
        invalid_data = pd.Series([1, 2, 3])

        check_pass, processed_data = contract(invalid_data)

        # Without coercion, invalid data should fail directly
        assert check_pass is False
        pd.testing.assert_series_equal(processed_data, invalid_data)

    def test_coerce_input(self, pandas_data_contract):
        """
        Test the _coerce_input method.

        - Ensures that the default implementation returns False for invalid data.
        """
        invalid_data = pd.Series([1, 2, 3])

        # The default implementation of _coerce_input should return False
        assert pandas_data_contract._coerce_input(invalid_data) is False

    def test_warning_on_coercion_failure(self, pandas_data_contract):
        """
        Test warnings issued during coercion failure.

        - Ensures that appropriate warnings are triggered for invalid data.
        - Ensures that coercion fails as expected.
        """
        invalid_data = pd.Series([1, 2, 3])

        with pytest.warns(Warning, match="Test warning message"):
            with pytest.warns(Warning, match=COERCION_FAILURE_MSG):
                check_pass, processed_data = pandas_data_contract(invalid_data)

        # Coercion should fail, and warnings should be triggered
        assert check_pass is False
        assert processed_data is False


# -- Index Contract Testing --
# Mock implementation of IndexContract for testing purposes
class MockIndexContract(IndexContract):
    def forward(self, index: pd.Index) -> bool:
        # For testing, consider the Index valid if it is not empty
        return not index.empty


@pytest.fixture(scope="function")
def mock_index_contract():
    return MockIndexContract(coerce=True, warning_message="Index validation failed.")


class Test_IndexContract:
    def test_index_contract_valid_index(self, mock_index_contract):
        """
        Test the __call__ method with a valid Index.

        - Ensures the Index passes validation.
        - Ensures the Index remains unchanged.
        """
        valid_index = pd.Index(["a", "b", "c"])

        check_pass, processed_index = mock_index_contract(valid_index)

        assert check_pass is True
        pd.testing.assert_index_equal(processed_index, valid_index)

    def test_index_contract_invalid_index_with_coerce(self, mock_index_contract):
        """
        Test the __call__ method with an invalid Index when coercion is enabled.

        - Ensures a warning is issued.
        - Ensures coercion fails as per the default implementation.
        """
        invalid_index = pd.Index([])  # Empty Index

        with pytest.warns(UserWarning, match="Index validation failed."):
            check_pass, processed_index = mock_index_contract(invalid_index)

        assert check_pass is False
        assert processed_index is False

    def test_index_contract_invalid_index_without_coerce(self):
        """
        Test the __call__ method with an invalid Index when coercion is disabled.

        - Ensures the Index fails validation directly.
        - Ensures the Index remains unchanged.
        """
        contract = MockIndexContract(
            coerce=False, warning_message="Index validation failed."
        )
        invalid_index = pd.Index([])  # Empty Index

        check_pass, processed_index = contract(invalid_index)

        assert check_pass is False
        pd.testing.assert_index_equal(processed_index, invalid_index)

    def test_index_contract_coerce_input(self, mock_index_contract):
        """
        Test the _coerce_input method.

        - Ensures the default implementation returns False for invalid data.
        """
        invalid_index = pd.Index([])  # Empty Index

        assert mock_index_contract._coerce_input(invalid_index) is False

    def test_index_contract_warning_on_coercion_failure(self, mock_index_contract):
        """
        Test warnings issued during coercion failure.

        - Ensures appropriate warnings are triggered for invalid Index.
        - Ensures coercion fails as expected.
        """
        invalid_index = pd.Index([])  # Empty Index

        with pytest.warns(UserWarning, match="Index validation failed."):
            with pytest.warns(Warning, match=COERCION_FAILURE_MSG):
                check_pass, processed_index = mock_index_contract(invalid_index)

        assert check_pass is False
        assert processed_index is False


# -- Series Contract Testing --
# Mock implementation of SeriesContract for testing purposes
class MockSeriesContract(SeriesContract):
    def forward(self, series: pd.Series) -> bool:
        # For testing, consider the Series valid if it contains no null values
        return series.notnull().all()


@pytest.fixture
def mock_series_contract():
    return MockSeriesContract(coerce=True, warning_message="Series validation failed.")


class Test_SeriesContract:
    def test_series_contract_valid_series(self, mock_series_contract):
        """
        Test the __call__ method with a valid Series.

        - Ensures the Series passes validation.
        - Ensures the Series remains unchanged.
        """
        valid_series = pd.Series([1, 2, 3])

        check_pass, processed_series = mock_series_contract(valid_series)

        assert check_pass
        pd.testing.assert_series_equal(processed_series, valid_series)

    def test_series_contract_invalid_series_with_coerce(self, mock_series_contract):
        """
        Test the __call__ method with an invalid Series when coercion is enabled.

        - Ensures a warning is issued.
        - Ensures coercion fails as per the default implementation.
        """
        invalid_series = pd.Series([1, None, 3])  # Series with a null value

        with pytest.warns(UserWarning, match="Series validation failed."):
            check_pass, processed_series = mock_series_contract(invalid_series)

        assert check_pass is False
        assert processed_series is False

    def test_series_contract_invalid_series_without_coerce(self):
        """
        Test the __call__ method with an invalid Series when coercion is disabled.

        - Ensures the Series fails validation directly.
        - Ensures the Series remains unchanged.
        """
        contract = MockSeriesContract(
            coerce=False, warning_message="Series validation failed."
        )
        invalid_series = pd.Series([1, None, 3])  # Series with a null value

        check_pass, processed_series = contract(invalid_series)

        assert not check_pass
        pd.testing.assert_series_equal(processed_series, invalid_series)

    def test_series_contract_coerce_input(self, mock_series_contract):
        """
        Test the _coerce_input method.

        - Ensures the default implementation returns False for invalid data.
        """
        invalid_series = pd.Series([1, None, 3])  # Series with a null value

        assert mock_series_contract._coerce_input(invalid_series) is False

    def test_series_contract_warning_on_coercion_failure(self, mock_series_contract):
        """
        Test warnings issued during coercion failure.

        - Ensures appropriate warnings are triggered for invalid Series.
        - Ensures coercion fails as expected.
        """
        invalid_series = pd.Series([1, None, 3])  # Series with a null value

        with pytest.warns(UserWarning, match="Series validation failed."):
            with pytest.warns(Warning, match=COERCION_FAILURE_MSG):
                check_pass, processed_series = mock_series_contract(invalid_series)

        assert not check_pass
        assert not processed_series


# -- DataFrame Contract Testing --
# Mock implementation of DataFrameContract for testing purposes
class MockDataFrameContract(DataFrameContract):
    def forward(self, dataframe: pd.DataFrame) -> bool:
        # For testing, consider the DataFrame valid if it has more than 0 rows
        return not dataframe.empty


@pytest.fixture
def mock_dataframe_contract():
    return MockDataFrameContract(
        coerce=True, warning_message="DataFrame validation failed."
    )


class Test_DataFrameContract:
    def test_dataframe_contract_valid_dataframe(self, mock_dataframe_contract):
        """
        Test the __call__ method with a valid DataFrame.

        - Ensures the DataFrame passes validation.
        - Ensures the DataFrame remains unchanged.
        """
        valid_dataframe = pd.DataFrame({"a": [1, 2, 3]})

        check_pass, processed_dataframe = mock_dataframe_contract(valid_dataframe)

        assert check_pass is True
        pd.testing.assert_frame_equal(processed_dataframe, valid_dataframe)

    def test_dataframe_contract_invalid_dataframe_with_coerce(
        self, mock_dataframe_contract
    ):
        """
        Test the __call__ method with an invalid DataFrame when coercion is enabled.

        - Ensures a warning is issued.
        - Ensures coercion fails as per the default implementation.
        """
        invalid_dataframe = pd.DataFrame()  # Empty DataFrame

        with pytest.warns(UserWarning, match="DataFrame validation failed."):
            check_pass, processed_dataframe = mock_dataframe_contract(invalid_dataframe)

        assert check_pass is False
        assert processed_dataframe is False

    def test_dataframe_contract_invalid_dataframe_without_coerce(self):
        """
        Test the __call__ method with an invalid DataFrame when coercion is disabled.

        - Ensures the DataFrame fails validation directly.
        - Ensures the DataFrame remains unchanged.
        """
        contract = MockDataFrameContract(
            coerce=False, warning_message="DataFrame validation failed."
        )
        invalid_dataframe = pd.DataFrame()  # Empty DataFrame

        check_pass, processed_dataframe = contract(invalid_dataframe)

        assert check_pass is False
        pd.testing.assert_frame_equal(processed_dataframe, invalid_dataframe)

    def test_dataframe_contract_coerce_input(self, mock_dataframe_contract):
        """
        Test the _coerce_input method.

        - Ensures the default implementation returns False for invalid data.
        """
        invalid_dataframe = pd.DataFrame()  # Empty DataFrame

        assert mock_dataframe_contract._coerce_input(invalid_dataframe) is False

    def test_dataframe_contract_warning_on_coercion_failure(
        self, mock_dataframe_contract
    ):
        """
        Test warnings issued during coercion failure.

        - Ensures appropriate warnings are triggered for invalid DataFrame.
        - Ensures coercion fails as expected.
        """
        invalid_dataframe = pd.DataFrame()  # Empty DataFrame

        with pytest.warns(UserWarning, match="DataFrame validation failed."):
            with pytest.warns(Warning, match=COERCION_FAILURE_MSG):
                check_pass, processed_dataframe = mock_dataframe_contract(
                    invalid_dataframe
                )

        assert check_pass is False
        assert processed_dataframe is False


# -- Wrapper Testing --


class TestRequiresDecorator:
    """
    Unit tests for the `requires` decorator.

    The `requires` decorator enforces validation and optional coercion of function arguments
    using a specified contract. These tests ensure that the decorator behaves correctly
    for valid data, invalid data with coercion, and invalid data without coercion.
    """

    class MockDataContract(_PandasDataContract):
        """
        Mock implementation of _PandasDataContract for testing purposes.
        """

        def forward(self, data: pd.DataFrame) -> bool:
            """
            Validate that the DataFrame has a column named 'valid'.

            Args:
                data (pd.DataFrame): The DataFrame to validate.

            Returns:
                bool: True if the 'valid' column exists, False otherwise.
            """
            return "valid" in data.columns

        def _coerce_input(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Coerce the DataFrame to include a 'valid' column with default values if missing.

            Args:
                data (pd.DataFrame): The DataFrame to coerce.

            Returns:
                pd.DataFrame: The coerced DataFrame with a 'valid' column.
            """
            if "valid" not in data.columns:
                data["valid"] = True
            return data

    @requires("data", MockDataContract(coerce=True))
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sample function to process data after validation and coercion.

        Args:
            data (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        data["processed"] = True
        return data

    def test_requires_valid_data(self):
        """
        Test that the decorator allows execution when the data is valid.
        """
        valid_data = pd.DataFrame({"valid": [True, True]})
        result = self.process_data(data=valid_data)
        assert "processed" in result.columns
        assert result["processed"].all()

    def test_requires_invalid_data_with_coerce(self):
        """
        Test that the decorator coerces the data when it is invalid.
        """
        invalid_data = pd.DataFrame({"other": [1, 2]})
        result = self.process_data(data=invalid_data)
        assert "valid" in result.columns
        assert result["valid"].all()
        assert "processed" in result.columns
        assert result["processed"].all()

    def test_requires_invalid_data_no_coerce(self):
        """
        Test that the decorator raises an exception when validation fails and coercion is disabled.
        """

        class NonCoercingMockDataContract(self.MockDataContract):
            def __init__(self):
                super().__init__(coerce=False)

        @requires("data", NonCoercingMockDataContract())
        def process_non_coerce_data(data: pd.DataFrame):
            return data

        invalid_data = pd.DataFrame({"other": [1, 2]})
        with pytest.raises(Exception, match="Validation failed for argument: data"):
            process_non_coerce_data(data=invalid_data)
