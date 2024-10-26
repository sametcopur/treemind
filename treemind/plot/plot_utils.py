import pandas as pd
import numpy as np

from typing import List, Tuple, Optional, Union
from numpy.typing import ArrayLike


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

def _validate_bar_plot_parameters(
    values: np.ndarray,
    raw_score: float,
    columns: ArrayLike,
    max_col: Union[int, None],
    figsize: Tuple[float, float],
    title: Union[str, None],
    title_fontsize: float,
    label_fontsize: float,
    show_raw_score: bool,
) -> None:
    """
    Validates parameters for the bar_plot function.

    Parameters
    ----------
    values : np.ndarray
        An array of contribution values for each feature.
    raw_score : float
        The raw score associated with the contributions.
    columns : list, optional
        A list of column names for labeling. If None, column indices will be used.
    max_col : int, optional
        The maximum number of features to display in the plot.
    figsize : Tuple[int, int]
        The figure size for the plot.
    title : str, optional
        Custom title for the plot. If None, a default title will be used.
    title_fontsize : float
        Font size for the title.
    label_fontsize : float
        Font size for the y-axis labels.
    show_raw_score : bool, optional
        Whether to show the raw score in the plot title.

    Returns
    -------
    None
        Raises an error if any validation check fails.
    """

    # Check values
    if not isinstance(values, np.ndarray):
        raise TypeError("The 'values' parameter must be a numpy.ndarray.")
    if values.ndim != 1:
        raise ValueError("The 'values' array must be one-dimensional.")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("All elements in 'values' must be numeric.")

    # Check raw_score
    if not isinstance(raw_score, (int, float)):
        raise TypeError(
            "The 'raw_score' parameter must be a numeric type (int or float)."
        )

    # Check columns
    if columns is not None:
        # Assuming _check_columns is a function defined elsewhere
        _check_columns(columns)

        if len(columns) != len(values):
            raise ValueError(
                "The length of 'columns' must match the length of 'values'."
            )

    # Check max_col
    if max_col is not None:
        if not isinstance(max_col, int) or max_col <= 0:
            raise ValueError(
                "The 'max_col' parameter must be a positive integer or None."
            )

    # Check figsize
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError("The 'figsize' parameter must be a tuple of two numeric values.")
    if not all(isinstance(dim, (int, float)) for dim in figsize):
        raise ValueError("Both dimensions in 'figsize' must be numeric.")

    # Check title
    if title is not None and not isinstance(title, str):
        raise TypeError("The 'title' parameter must be a string or None.")

    # Check show_raw_score
    if not isinstance(show_raw_score, bool):
        raise TypeError("The 'show_raw_score' parameter must be a boolean.")

    # Check title_fontsize and label_fontsize
    if not isinstance(title_fontsize, (int, float)) or title_fontsize <= 0:
        raise ValueError("The 'title_fontsize' must be a positive number.")
    if not isinstance(label_fontsize, (int, float)) or label_fontsize <= 0:
        raise ValueError("The 'label_fontsize' must be a positive number.")


def _validate_interaction_plot_parameters(
    df: pd.DataFrame,
    figsize: Tuple[float, float],
    axis_ticks_n: int,
    cbar_ticks_n: int,
    ticks_decimal: int,
    ticks_fontsize: int | float,
    title_fontsize: int | float,
    label_fontsizes: int | float,
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
        A DataFrame expected to contain interaction data with columns ending in `_lb`, `_ub`, `_lb`, `_ub`,
        and `value` as the last column.
    figsize : Tuple[int, int]
        Size of the figure, should be a tuple of two positive integers.
    xticks_n : int
        Number of ticks on x-axis, must be a positive integer.
    yticks_n : int
        Number of ticks on y-axis, must be a positive integer.
    cbar_ticks_n : int
        Number of ticks on colorbar, must be a positive integer.
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
        or df.columns[-1] != "value"
    ):
        raise ValueError(
            "The first four columns of `df` must end with '_lb', '_ub', '_lb', '_ub' respectively, "
            "and the last column must be 'value'."
        )


    # Check figsize
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError("The 'figsize' parameter must be a tuple of two numeric values.")
    if not all(isinstance(dim, (int, float)) for dim in figsize):
        raise ValueError("Both dimensions in 'figsize' must be numeric.")

    # Check `xticks_n`, `yticks_n`, `cbar_ticks_n` are positive integers
    for param, name in [(axis_ticks_n, "axis_ticks_n"), (cbar_ticks_n, "cbar_ticks_n")]:
        if not isinstance(param, int) or param <= 0:
            raise ValueError(f"`{name}` must be a positive integer.")

    # Check `ticks_decimal` is a non-negative integer
    if not isinstance(ticks_decimal, int) or ticks_decimal < 0:
        raise ValueError("`ticks_decimal` must be a non-negative integer.")

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


def _validate_range_plot_parameters(
    values: np.ndarray,
    raw_score: float,
    split_points: List[List[float]],
    scale: float = 1,
    columns: Optional[List[str]] = None,
    max_col: Optional[int] = None,
    title: Optional[str] = None,
    label_fontsize: float = 9,
    title_fontsize: float = 12,
    interval_fontsize: float = 4.5,
    value_fontsize: float = 5.5,
    show_raw_score: bool = True,
) -> None:
    """
    Validates inputs for `range_plot` to ensure proper types, dimensions, and values.

    Parameters
    ----------
    values : np.ndarray
        2D numeric array where each row holds values to be plotted.
    raw_score : float
        A numeric score for display, typically representing an overall metric.
    points : List[List[float]]
        List of lists where each inner list has interval points matching each row in `values`.
    scale : float
        Scaling factor for the plot, must be a positive float.
    columns : List[str], optional
        Labels for each row, should match the number of rows in `values` if provided.
    max_col : int, optional
        Maximum rows to display, must be a positive integer if specified.
    title : str, optional
        Title text for the plot.
    label_fontsize : float
        Font size for y-axis labels, must be a positive float.
    title_fontsize : float
        Font size for the title, must be a positive float.
    interval_fontsize : float
        Font size for interval labels on bars, must be a positive float.
    value_fontsize : float
        Font size for value labels on bars, must be a positive float.
    show_raw_score : bool
        Flag to display raw_score, must be a boolean.

    Raises
    ------
    TypeError, ValueError
        If any input has an invalid type, dimension, or value.
    """

    # Check `values` is a 2D numpy array
    if not isinstance(values, np.ndarray):
        raise TypeError("`values` must be a numpy ndarray.")
    if values.ndim != 2:
        raise ValueError("`values` must be a 2D array.")

    # Check `raw_score` is a float
    if not isinstance(raw_score, (int, float)):
        raise TypeError("`raw_score` must be a numeric (int or float).")

    # Check `points` is a list of lists and matches the row count in `values`
    if not isinstance(split_points, list) or not all(
        isinstance(p, np.ndarray) for p in split_points
    ):
        raise TypeError("`points` must be a list of numpy arrays.")
    if len(split_points) != values.shape[0]:
        raise ValueError("`points` must have the same number of rows as `values`.")

    # Check `scale` is a positive float
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("`scale` must be a positive number.")

    # Check `columns` if provided, is a list of strings and matches the row count in `values`
    if columns is not None:
        _check_columns(columns)

        if len(columns) != values.shape[0]:
            raise ValueError(
                "`columns` length must match the number of rows in `values`."
            )

    # Check `max_col` if provided, is a positive integer
    if max_col is not None:
        if not isinstance(max_col, int) or max_col <= 0:
            raise ValueError("`max_col` must be a positive integer if specified.")

    # Check font sizes are positive floats
    for font_size, name in [
        (label_fontsize, "label_fontsize"),
        (title_fontsize, "title_fontsize"),
        (interval_fontsize, "interval_fontsize"),
        (value_fontsize, "value_fontsize"),
    ]:
        if not isinstance(font_size, (int, float)) or font_size <= 0:
            raise ValueError(f"`{name}` must be a positive number.")

    # Check `title` is either None or a string
    if title is not None and not isinstance(title, str):
        raise TypeError("`title` must be a string if provided.")

    # Check `show_raw_score` is a boolean
    if not isinstance(show_raw_score, bool):
        raise TypeError("`show_raw_score` must be a boolean.")


def _validate_feature_plot_parameters(
    df: pd.DataFrame,
    figsize: Tuple[float, float],
    show_min_max: bool,
    xticks_n: int,
    yticks_n: int,
    ticks_fontsize: int | float,
    title_fontsize: int | float,
    label_fontsizes: int | float,
    ticks_decimal: int,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    """
    Validates the input parameters for the feature_plot function.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to validate.
    figsize : tuple of int
        The figure size as a tuple of two positive integers.
    show_min_max : bool
        Whether to show the min-max range.
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
    required_columns = {"mean", "min", "max"}
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
        raise TypeError("The 'figsize' parameter must be a tuple of two numeric values.")
    if not all(isinstance(dim, (int, float)) for dim in figsize):
        raise ValueError("Both dimensions in 'figsize' must be numeric.")

    # Validate show_min_max
    if not isinstance(show_min_max, bool):
        raise ValueError("show_min_max must be a boolean value.")

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

    # Validate ticks_decimal
    if not (isinstance(ticks_decimal, int) and ticks_decimal >= 0):
        raise ValueError("ticks_decimal must be a non-negative integer.")

    # Validate title
    if title is not None and not isinstance(title, str):
        raise ValueError("title must be a string or None.")

    # Validate xlabel and ylabel
    if xlabel is not None and not isinstance(xlabel, str):
        raise ValueError("xlabel must be a string or None.")
    if ylabel is not None and not isinstance(ylabel, str):
        raise ValueError("ylabel must be a string or None.")


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
