# pandas-contracts

`pandas-contracts` is a Python library that provides a contract-based approach to validate and optionally coerce pandas DataFrame, Series, and Index objects. It simplifies enforcing data integrity and preprocessing for pandas-based workflows.

---

## Features

- **Modular Contracts**: Validate and coerce DataFrame, Series, and Index objects.
- **Custom Validators**: Easily extend the base contracts for specific use cases.
- **Decorator Support**: Enforce validation on function arguments with a simple decorator.
- **Prebuilt Extensions**: Ready-to-use contracts for common requirements like uniqueness, positivity, monotonicity, and more.

---

## Installation

```bash
pip install pandas-contracts
```

---

## Module Structure

The library is organized into the following modules:

- **`pandas_contracts/__init__.py`**: Initialization file for the package.
- **`pandas_contracts/core.py`**: Contains the abstract base class `_PandasDataContract` for creating contracts.
- **`pandas_contracts/index.py`**: Extensions for validating and coercing pandas Index objects.
- **`pandas_contracts/series.py`**: Extensions for validating and coercing pandas Series objects.
- **`pandas_contracts/dataframe.py`**: Extensions for validating and coercing pandas DataFrame objects.
- **`pandas_contracts/options.py`**: Configuration options for the library.

---

## Usage

### 1. Validating and Coercing Index

```python
import pandas as pd
from pandas_contracts.index import RequiresUniqueIndex

# Example Index
index = pd.Index([1, 1, 2, 3])

# Validator
validator = RequiresUniqueIndex()

# Validation
is_valid = validator.forward(index)
print(is_valid)  # Output: False

# Coercion
coerced_index = validator._coerce_input(index)
print(coerced_index)  # Output: Int64Index([1, 2, 3], dtype='int64')
```

### 2. Validating and Coercing Series

```python
import pandas as pd
from pandas_contracts.series import RequiresNonNegative

# Example Series
series = pd.Series([-1, 0, 1, 2])

# Validator
validator = RequiresNonNegative(coerce=True)

# Validation
is_valid = validator.forward(series)
print(is_valid)  # Output: False

# Coercion
coerced_series = validator._coerce_input(series)
print(coerced_series)  # Output: 0    0
                      #         1    0
                      #         2    1
                      #         3    2
```

### 3. Validating and Coercing DataFrame

```python
import pandas as pd
from pandas_contracts.dataframe import RequiresColumns

# Example DataFrame
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

# Validator
validator = RequiresColumns(columns=['a', 'b', 'c'], coerce=True)

# Validation
is_valid = validator.forward(df)
print(is_valid)  # Output: False

# Coercion
coerced_df = validator._coerce_input(df)
print(coerced_df)
# Output:
#    a  b     c
# 0  1  3  None
# 1  2  4  None
```

### 4. Using the `requires` Decorator

```python
from pandas_contracts.decorators import requires
from pandas_contracts.series import RequiresNotNaN

@requires("data", RequiresNotNaN(coerce=True))
def process_data(data):
    data["processed"] = True
    return data

# Example Data
data = pd.Series([1, None, 3])

# Process
result = process_data(data=data)
print(result)
# Output:
# 0    1.0
# 1    0.0
# 2    3.0
# dtype: float64
```

---

## Prebuilt Contracts

### Index Contracts

| Contract               | Description                                          |
|------------------------|------------------------------------------------------|
| `RequiresUniqueIndex`  | Validates that the Index contains unique values.     |
| `RequiresMonotonicIndex` | Validates that the Index is monotonic.             |
| `RequiresNonNegativeIndex` | Validates that the Index contains non-negative values. |
| `RequiresPositiveIndex` | Validates that the Index contains strictly positive values. |
| `RequiresIndexNames`    | Validates that the Index has specific names.        |

### Series Contracts

| Contract               | Description                                          |
|------------------------|------------------------------------------------------|
| `RequiresNonNegative`  | Validates that the Series contains non-negative values. |
| `RequiresNotNaN`       | Validates that the Series has no missing values.     |
| `RequiresUniqueValues` | Validates that the Series contains unique values.    |
| `RequiresPositive`     | Validates that the Series contains strictly positive values. |

### DataFrame Contracts

| Contract               | Description                                          |
|------------------------|------------------------------------------------------|
| `RequiresColumns`      | Validates that the DataFrame contains specific columns. |
| `RequiresDtypes`       | Validates the data types of specific columns.        |
| `RequiresNotNaN`       | Validates that specific columns have no missing values. |
| `RequiresUniqueIndex`  | Ensures the DataFrame index has unique values.       |

---

## Configuration

Modify the behavior of the library using `pandas_contracts.options`. Example:

```python
from pandas_contracts import options

options.COERCE_DATA = True  # Enable coercion globally
options.COERCION_FAILURE_MSG = "Coercion failed!"  # Custom failure message
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
