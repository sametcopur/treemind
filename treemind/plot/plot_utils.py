import pandas as pd
import numpy as np


def _find_tick_decimal(ticks: np.ndarray, n_ticks: int) -> int:
    tick_min, tick_max = np.min(ticks), np.max(ticks)
    tick_range = tick_max - tick_min

    if tick_range < n_ticks:
        for decimals_count in range(1, 5):
            rounded_ticks = np.round(ticks, decimals_count)
            if np.allclose(ticks, rounded_ticks, atol=tick_range / (n_ticks * 10)):
                return decimals_count
        return 4
    else:
        return 0


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
