import pandas as pd
import numpy as np

from typing import List

def _check_columns(obj):
    try:
        # Attempt to convert the object to a NumPy array
        arr = np.asarray(obj)
    except Exception as e:
        # Raise TypeError if the object isn't array-like
        raise TypeError("The given object is not array-like.") from e

    # Check if the array is one-dimensional
    if arr.ndim != 1:
        raise ValueError("The given object is not one-dimensional.")
    
        

def _create_intervals(point_row: List[float]) -> List[str]:
    """
    Creates intervals from a list of points.

    Parameters
    ----------
    point_row : list of float
        A list of points to create intervals from.

    Returns
    -------
    intervals : list of str
        A list of interval strings in the format '(a, b]'.

    Examples
    --------
    >>> create_intervals([1.0, 2.0, 3.0])
    ['(-inf, 1.000]', '(1.000, 2.000]', '(2.000, inf)']
    """

    intervals = []
    point_row = np.array(point_row)
    intervals.append(f"(-inf, {point_row[0]:.3f}]")

    for i in range(1, len(point_row) - 1):
        intervals.append(f"({point_row[i-1]:.3f}, {point_row[i]:.3f}]")

    intervals.append(f"({point_row[-2]:.3f}, inf)")
    return intervals

def _replace_infinity(
    data: pd.DataFrame, column_name: str, infinity_type: str = "positive"
):
    """
    Replaces positive or negative infinity values in a DataFrame column with a calculated value
    based on the finite values in the column.

    Parameters:
    - data: pd.DataFrame
        The DataFrame containing the data.
    - column_name: str
        The name of the column to process.
    - infinity_type: str, optional (default='positive')
        Specify 'positive' to replace positive infinity values,
        'negative' to replace negative infinity values.

    Returns:
    - pd.DataFrame
        A copy of the DataFrame with infinity values replaced.
    """
    data = data.copy()
    # Extract unique finite values
    unique_values = np.asarray(data[column_name].unique(), dtype=np.float64)
    finite_values = unique_values[~np.isinf(unique_values)]
    finite_values_sorted = np.sort(finite_values)

    if infinity_type == "positive":
        # Handle positive infinity
        max_value = finite_values_sorted.max()
        if finite_values_sorted.size > 1:
            second_max = finite_values_sorted[-2]
            difference = max_value - second_max
        else:
            difference = (
                max_value * 0.1
            )  # Use 10% of max_value if only one unique point
        # Replace positive infinity values
        data.loc[np.isposinf(data[column_name]), column_name] = max_value + difference

    elif infinity_type == "negative":
        # Handle negative infinity
        min_value = finite_values_sorted.min()
        if finite_values_sorted.size > 1:
            second_min = finite_values_sorted[1]
            difference = second_min - min_value
        else:
            difference = (
                abs(min_value) * 0.1
            )  # Use 10% of min_value if only one unique point
        # Replace negative infinity values
        data.loc[np.isneginf(data[column_name]), column_name] = min_value - difference

    else:
        raise ValueError("infinity_type must be 'positive' or 'negative'")

    return data