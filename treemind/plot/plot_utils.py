from typing import Tuple, Optional, Union

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


def _check_columns(columns):
    try:
        # Attempt to convert columns to a NumPy array
        arr = np.asarray(columns)
    except Exception as e:
        # Raise TypeError if columns isn't array-like
        raise TypeError("`columns` must be an array-like object.") from e

    # Check if the array is one-dimensional
    if arr.ndim != 1:
        raise ValueError("`columns` must be a one-dimensional array-like object.")

    # Check if all elements in columns are strings
    if not all(isinstance(col, str) for col in arr):
        raise ValueError("All elements in `columns` must be strings.")


def _validate_interaction_plot_parameters(
    df: pd.DataFrame,
    figsize: Tuple[float, float],
    axis_ticks_n: int,
    ticks_fontsize: Union[int, float],
    title_fontsize: Union[int, float],
    label_fontsizes: Union[int, float],
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
    color_bar_label: Optional[str],
) -> None:
    """
    Validates inputs for `interaction_plot` to ensure proper types, dimensions, and values.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame expected to contain interaction data with columns ending in `_lb`, `_ub`,
        `_lb`, `_ub`, and `value` as the last column.
    figsize : Tuple[int, int]
        Size of the figure, should be a tuple of two positive integers.
    xticks_n : int
        Number of ticks on x-axis, must be a positive integer.
    yticks_n : int
        Number of ticks on y-axis, must be a positive integer.
    ticks_decimal : int
        Number of decimal places for tick labels, must be a non-negative integer.
    ticks_fontsize : int or float
        Font size for axis tick labels, must be a positive number.
    title_fontsize : int or float
        Font size for plot title, must be a positive number.
    label_fontsizes : int or float
        Font size for axis labels, must be a positive number.
    title : str, optional
        Plot title, if provided must be a string.
    xlabel : str, optional
        X-axis label, if provided must be a string.
    ylabel : str, optional
        Y-axis label, if provided must be a string.
    color_bar_label : str, optional
        Colorbar label, if provided must be a string.

    Raises
    ------
    TypeError, ValueError
        If any input has an invalid type, dimension, or value.
    """
    # Validate `df` structure and column names
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    if df.shape[0] <= 2:
        raise ValueError("There is no interaction between features to plot.")

    # Ensure column names end as required
    expected_endings = ["_lb", "_ub", "_lb", "_ub"]
    if (
        len(df.columns) < 5
        or not all(
            col.endswith(end) for col, end in zip(df.columns[:4], expected_endings)
        )
        or df.columns[-3] != "value"
        or df.columns[-2] != "std"
        or df.columns[-1] != "count"
    ):
        raise ValueError(
            "The first four columns of `df` must end with '_lb', '_ub', '_lb', '_ub' respectively, "
            "and the last columns must be 'value', 'count'."
        )

    # Check figsize
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError(
            "The 'figsize' parameter must be a tuple of two numeric values."
        )
    if not all(isinstance(dim, (int, float)) for dim in figsize):
        raise ValueError("Both dimensions in 'figsize' must be numeric.")

    # Check `xticks_n`, `yticks_n`, `cbar_ticks_n` are positive integers
    for param, name in [(axis_ticks_n, "axis_ticks_n")]:
        if not isinstance(param, int) or param <= 0:
            raise ValueError(f"`{name}` must be a positive integer.")

    # Validate font sizes are positive numbers
    for font_size, name in [
        (ticks_fontsize, "ticks_fontsize"),
        (title_fontsize, "title_fontsize"),
        (label_fontsizes, "label_fontsizes"),
    ]:
        if not isinstance(font_size, (int, float)) or font_size <= 0:
            raise ValueError(f"`{name}` must be a positive number.")

    # Check `title`, `xlabel`, `ylabel`, `color_bar_label` if provided, are strings
    for label, name in [
        (title, "title"),
        (xlabel, "xlabel"),
        (ylabel, "ylabel"),
        (color_bar_label, "color_bar_label"),
    ]:
        if label is not None and not isinstance(label, str):
            raise TypeError(f"`{name}` must be a string if provided.")


def _validate_feature_plot_parameters(
    df: pd.DataFrame,
    figsize: Tuple[float, float],
    show_std: bool,
    show_range: bool,
    xticks_n: int,
    yticks_n: int,
    ticks_fontsize: Union[int, float],
    title_fontsize: Union[int, float],
    label_fontsizes: Union[int, float],
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
) -> None:
    """
    Validates the input parameters for the feature_plot function.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to validate.
    figsize : tuple of int
        The figure size as a tuple of two positive integers.
    show_std : bool
        Whether to show the std range.
    xticks_n : int
        Number of ticks on the x-axis.
    yticks_n : int
        Number of ticks on the y-axis.
    ticks_fontsize : int
        Font size for the tick labels.
    title_fontsize : int
        Font size for the tick labels.
    ticks_decimal : int
        Number of decimal places for the tick labels.
    title : str or None
        Title of the plot, or None for automatic title.
    xlabel : str or None
        Label for the x-axis, or None for default.
    ylabel : str or None
        Label for the y-axis, or None for default.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    # Validate DataFrame columns
    required_columns = {"value", "std", "count"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"The DataFrame must contain the following columns: {required_columns}"
        )

    # Check for columns ending with '_lb' and '_ub'
    lb_columns = [col for col in df.columns if col.endswith("_lb")]
    ub_columns = [col for col in df.columns if col.endswith("_ub")]

    if len(lb_columns) != 1 or len(ub_columns) != 1:
        raise ValueError(
            "The DataFrame must contain exactly one '_lb' column and one '_ub' column."
        )

    # Check figsize
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError(
            "The 'figsize' parameter must be a tuple of two numeric values."
        )
    if not all(isinstance(dim, (int, float)) for dim in figsize):
        raise ValueError("Both dimensions in 'figsize' must be numeric.")

    # Validate show_min_max
    if not isinstance(show_std, bool):
        raise ValueError("show_std must be a boolean value.")

    if not isinstance(show_range, bool):
        raise ValueError("show_range must be a boolean value.")

    # Validate xticks_n and yticks_n
    if not (isinstance(xticks_n, int) and xticks_n > 0):
        raise ValueError("xticks_n must be a positive integer.")
    if not (isinstance(yticks_n, int) and yticks_n > 0):
        raise ValueError("yticks_n must be a positive integer.")

    # Validate ticks_fontsize
    if not (isinstance(ticks_fontsize, (float, int)) and ticks_fontsize > 0):
        raise ValueError("ticks_fontsize must be a positive float or integer.")

    if not (isinstance(title_fontsize, (float, int)) and title_fontsize > 0):
        raise ValueError("title_fontsize must be a positive float or integer.")

    if not (isinstance(label_fontsizes, (float, int)) and label_fontsizes > 0):
        raise ValueError("label_fontsizes must be a positive float or integer.")

    # Validate title
    if title is not None and not isinstance(title, str):
        raise ValueError("title must be a string or None.")

    # Validate xlabel and ylabel
    if xlabel is not None and not isinstance(xlabel, str):
        raise ValueError("xlabel must be a string or None.")
    if ylabel is not None and not isinstance(ylabel, str):
        raise ValueError("ylabel must be a string or None.")


def _validate_interaction_scatter_plot_parameters(
    X: np.ndarray,
    df: pd.DataFrame,
    col_1: int,
    col_2: int,
    figsize: Tuple[float, float],
    ticks_fontsize: Tuple[int, float],
    title_fontsize: Tuple[int, float],
    label_fontsizes: Tuple[int, float],
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
    color_bar_label: Optional[str],
) -> None:
    """
    Validates inputs for `interaction_plot` to ensure proper types, dimensions, and values.

    Parameters
    ----------
    X : np.ndarray
        A two-dimensional numeric array containing the feature data.
    df : pd.DataFrame
        A DataFrame expected to contain interaction data with columns ending in `_lb`, `_ub`,
        `_lb`, `_ub`, and `value` as the last column.
    col1 : int
        Index of the first feature to use in the interaction plot.
    col2 : int
        Index of the second feature to use in the interaction plot.
    figsize : Tuple[float, float]
        Size of the figure, should be a tuple of two positive numbers.
    axis_ticks_n : int
        Number of ticks on both axes, must be a positive integer.
    ticks_fontsize : int or float
        Font size for axis tick labels, must be a positive number.
    title_fontsize : int or float
        Font size for plot title, must be a positive number.
    label_fontsizes : int or float
        Font size for axis labels, must be a positive number.
    title : str, optional
        Plot title, if provided must be a string.
    xlabel : str, optional
        X-axis label, if provided must be a string.
    ylabel : str, optional
        Y-axis label, if provided must be a string.
    color_bar_label : str, optional
        Colorbar label, if provided must be a string.

    Raises
    ------
    TypeError, ValueError
        If any input has an invalid type, dimension, or value.
    """
    # Validate `X` is a two-dimensional numeric array
    if X.ndim != 2:
        raise TypeError("`X` must be a two-dimensional NumPy array.")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("`X` must contain only numeric values.")

    # Validate `col1` and `col2` are valid column indices
    if not isinstance(col_1, int) or not isinstance(col_2, int):
        raise TypeError("`col1` and `col2` must be integers.")
    if col_1 < 0 or col_2 < 0 or col_1 >= X.shape[1] or col_2 >= X.shape[1]:
        raise ValueError(
            "`col1` and `col2` must be valid column indices within the range of `X`."
        )

    # Validate `df` structure and column names
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    if df.shape[0] <= 2:
        raise ValueError("There is no interaction between features to plot.")

    # Ensure column names end as required
    expected_endings = ["_lb", "_ub", "_lb", "_ub"]
    if (
        len(df.columns) < 5
        or not all(
            col.endswith(end) for col, end in zip(df.columns[:4], expected_endings)
        )
        or df.columns[-3] != "value"
        or df.columns[-2] != "std"
        or df.columns[-1] != "count"
    ):
        raise ValueError(
            "The first four columns of `df` must end with '_lb', '_ub', '_lb', '_ub' respectively, "
            "and the last columns must be 'value', 'std', 'count'."
        )

    # Check figsize
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError(
            "The 'figsize' parameter must be a tuple of two numeric values."
        )
    if not all(isinstance(dim, (int, float)) for dim in figsize):
        raise ValueError("Both dimensions in 'figsize' must be numeric.")

    # Validate font sizes are positive numbers
    for font_size, name in [
        (ticks_fontsize, "ticks_fontsize"),
        (title_fontsize, "title_fontsize"),
        (label_fontsizes, "label_fontsizes"),
    ]:
        if not isinstance(font_size, (int, float)) or font_size <= 0:
            raise ValueError(f"`{name}` must be a positive number.")

    # Check `title`, `xlabel`, `ylabel`, `color_bar_label` if provided, are strings
    for label, name in [
        (title, "title"),
        (xlabel, "xlabel"),
        (ylabel, "ylabel"),
        (color_bar_label, "color_bar_label"),
    ]:
        if label is not None and not isinstance(label, str):
            raise TypeError(f"`{name}` must be a string if provided.")


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
