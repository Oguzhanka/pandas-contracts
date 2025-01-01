"""
Module: pandas_data_contracts.core

This module provides a framework for validating and coercing pandas data structures
using a contract-based approach. It includes classes for handling DataFrame, Series,
and Index objects, as well as a decorator to enforce validation requirements for
function arguments.

Key Components:
- PandasDataContract: An abstract base class for creating validation contracts.
- DataFrameContract: A base class for creating validation contracts for pandas DataFrame.
- SeriesContract: A base class for creating validation contracts for pandas Series.
- IndexContract: A concrete implementation for validating and coercing pandas Index objects.
- requires: A decorator for enforcing validation requirements on function arguments.

Usage:
1. Extend `PandasDataContract` to create custom validation contracts.
2. Use `IndexContract` as an example for creating validation rules for specific pandas objects.
3. Apply the `requires` decorator to functions, specifying the argument name and the contract to
enforce validation.

Example:
    class CustomContract(PandasDataContract):
        def forward(self, data):
            return isinstance(data, pd.DataFrame) and not data.empty

    @requires("data", CustomContract())
    def process_data(data):
        print("Data is valid")
"""

import abc
import typing
import warnings

import pandas as pd

from pandas_contracts import options


__all__ = [
    "requires",
    "IndexContract",
    "SeriesContract",
    "DataFrameContract",
]


PandasData = typing.TypeVar("PandasData", pd.DataFrame, pd.Series, pd.Index)


class _PandasDataContract(abc.ABC):
    """
    Abstract base class for validating and optionally coercing pandas data objects.

    This class provides a contract to check whether input data adheres to specific requirements.
    If the validation fails, the data can be optionally coerced based on the configuration.
    """

    def __init__(self, coerce: bool = True, warning_message: str = ""):
        """
        Initialize the _PandasDataContract with optional coercion settings.

        Args:
            coerce (bool): Whether to attempt coercion when the data validation fails.
            warning_message (str): A custom warning message to display during coercion.
        """
        self._coerce: typing.Final[bool] = coerce
        self._warning_message: typing.Final[str] = warning_message

    def __call__(self, data: PandasData) -> tuple[bool, PandasData]:
        """
        Validate the data and optionally coerce it if validation fails.

        Args:
            data (PandasData): The input data to validate, which can be a DataFrame, Series, or
            Index.

        Returns:
            tuple[bool, PandasData]: A tuple where the first element is a boolean indicating
                                     whether the validation passed, and the second element
                                     is the original or coerced data.
        """
        # Perform the forward validation
        check_pass = self.forward(data)

        if not check_pass:
            if self._coerce:
                # Issue a warning and attempt to coerce the data
                warnings.warn(self._warning_message)
                data = self._coerce_input(data)

                if isinstance(data, bool) and not data:
                    # Coercion failed; raise a failure warning
                    check_pass = False
                    warnings.warn(Warning(options.COERCION_FAILURE_MSG))
                else:
                    # Coercion succeeded
                    check_pass = True

        return check_pass, data

    @abc.abstractmethod
    def forward(self, data: PandasData) -> bool:
        """
        Abstract method to perform validation on the data.

        Args:
            data (PandasData): The input data to validate.

        Returns:
            bool: True if the data meets the validation criteria, False otherwise.
        """
        pass

    def _coerce_input(self, data: PandasData) -> typing.Union[bool, PandasData]:
        """
        Attempt to coerce the input data to a valid format.

        Args:
            data (PandasData): The input data to coerce.

        Returns:
            Union[bool, PandasData]: The coerced data if successful, or False if coercion fails.
        """
        # Default implementation does not support coercion
        return False


class IndexContract(_PandasDataContract):
    """
    A contract for validating and optionally coercing pandas Index objects.

    This class extends the _PandasDataContract to specifically handle validation
    and coercion logic for pandas Index objects.
    """

    def __call__(self, index: pd.Index) -> tuple[bool, pd.Index]:
        """
        Validate the pandas Index and optionally coerce it if validation fails.

        Args:
            index (pd.Index): The pandas Index to validate.

        Returns:
            tuple[bool, pd.Index]: A tuple where the first element is a boolean indicating
                                   whether the validation passed, and the second element
                                   is the original or coerced Index.
        """
        # Delegate the validation and optional coercion logic to the parent class
        return super().__call__(index)

    def _coerce_input(self, index: pd.Index) -> typing.Union[pd.Index, bool]:
        """
        Attempt to coerce the pandas Index to a valid format.

        Args:
            index (pd.Index): The pandas Index to coerce.

        Returns:
            Union[pd.Index, bool]: The coerced Index if successful, or False if coercion fails.
        """
        # Use the default coercion logic from the parent class
        return super()._coerce_input(index)

    @abc.abstractmethod
    def forward(self, index: pd.Index) -> bool:
        """
        Abstract method to perform validation on the pandas Index.

        Args:
            index (pd.Index): The pandas Index to validate.

        Returns:
            bool: True if the Index meets the validation criteria, False otherwise.

        This method must be implemented by any subclass of IndexContract.
        """
        pass


class SeriesContract(_PandasDataContract):
    """
    A contract for validating and optionally coercing pandas Series objects.

    This class extends the _PandasDataContract to specifically handle validation
    and coercion logic for pandas Series objects.
    """

    def __call__(self, series: pd.Series) -> tuple[bool, pd.Series]:
        """
        Validate the pandas Series and optionally coerce it if validation fails.

        Args:
            series (pd.Series): The pandas Series to validate.

        Returns:
            tuple[bool, pd.Series]: A tuple where the first element is a boolean indicating
                                    whether the validation passed, and the second element
                                    is the original or coerced Series.
        """
        # Delegate the validation and optional coercion logic to the parent class
        return super().__call__(series)

    def _coerce_input(self, series: pd.Series) -> typing.Union[pd.Series, bool]:
        """
        Attempt to coerce the pandas Series to a valid format.

        Args:
            series (pd.Series): The pandas Series to coerce.

        Returns:
            Union[pd.Series, bool]: The coerced Series if successful, or False if coercion fails.
        """
        # Use the default coercion logic from the parent class
        return super()._coerce_input(series)

    @abc.abstractmethod
    def forward(self, series: pd.Series) -> bool:
        """
        Abstract method to perform validation on the pandas Series.

        Args:
            series (pd.Series): The pandas Series to validate.

        Returns:
            bool: True if the Series meets the validation criteria, False otherwise.

        This method must be implemented by any subclass of SeriesContract.
        """
        pass


class DataFrameContract(_PandasDataContract):
    """
    A contract for validating and optionally coercing pandas DataFrame objects.

    This class extends the _PandasDataContract to specifically handle validation
    and coercion logic for pandas DataFrame objects.
    """

    def __call__(self, dataframe: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
        """
        Validate the pandas DataFrame and optionally coerce it if validation fails.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to validate.

        Returns:
            tuple[bool, pd.DataFrame]: A tuple where the first element is a boolean indicating
                                       whether the validation passed, and the second element
                                       is the original or coerced DataFrame.
        """
        # Delegate the validation and optional coercion logic to the parent class
        return super().__call__(dataframe)

    def _coerce_input(
        self, dataframe: pd.DataFrame
    ) -> typing.Union[pd.DataFrame, bool]:
        """
        Attempt to coerce the pandas DataFrame to a valid format.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to coerce.

        Returns:
            Union[pd.DataFrame, bool]: The coerced DataFrame if successful, or False if coercion
            fails.
        """
        # Use the default coercion logic from the parent class
        return super()._coerce_input(dataframe)

    @abc.abstractmethod
    def forward(self, dataframe: pd.DataFrame) -> bool:
        """
        Abstract method to perform validation on the pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to validate.

        Returns:
            bool: True if the DataFrame meets the validation criteria, False otherwise.

        This method must be implemented by any subclass of DataFrameContract.
        """
        pass


def requires(arg_name: str, contract: _PandasDataContract) -> typing.Callable:
    """
    Decorator to enforce that a function argument satisfies a specified contract.

    This decorator validates an argument passed to a function using a provided _PandasDataContract.
    If validation fails, an exception is raised. If validation succeeds, the argument can optionally
    be coerced into a valid form and passed to the function.

    Args:
        arg_name (str): The name of the argument to validate.
        contract (_PandasDataContract): The contract used to validate the argument.

    Returns:
        Callable: A decorator that validates the argument before the function is executed.

    Example:
        @requires("data", SomeDataContract())
        def my_function(data):
            pass
    """

    def requires_wrapper(func: typing.Callable) -> typing.Callable:
        """
        Wrap the original function to include validation of the specified argument.

        Args:
            func (Callable): The original function to wrap.

        Returns:
            Callable: The wrapped function with argument validation.
        """

        def wrapped_func(*args, **kwargs):
            """
            Validate the specified argument before executing the function.

            Args:
                *args: Positional arguments for the function.
                **kwargs: Keyword arguments for the function.

            Raises:
                Exception: If the argument fails validation.

            Returns:
                Any: The return value of the original function.
            """
            # Validate the specified argument using the provided contract
            check_pass, coerced_data = contract(kwargs[arg_name])
            if check_pass:
                # Update the argument with the coerced value if validation passes
                kwargs[arg_name] = coerced_data
            else:
                # Raise an exception if validation fails
                raise Exception("Validation failed for argument: " + arg_name)
            # Execute the original function
            return func(*args, **kwargs)

        return wrapped_func

    return requires_wrapper
