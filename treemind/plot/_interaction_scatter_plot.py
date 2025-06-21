from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


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


def _build_lookup(df: pd.DataFrame):
    """
    Same purpose as before, but tolerates extra columns.
    Looks for the first four columns as bounds + a column named 'value'.
    """
    # ---- 1) Kolonları belirle ------------------------------------------
    bound_cols = df.columns[:4]  # ilk 4 → sınırlar
    value_col = "value" if "value" in df.columns else df.columns[4]

    # Sadece gereken beş kolonu çek
    sub = df[list(bound_cols) + [value_col]]

    # ---- 2) Kenar dizileri ---------------------------------------------
    x_edges = np.unique(np.r_[sub.iloc[:, 0].values, sub.iloc[:, 1].values])
    y_edges = np.unique(np.r_[sub.iloc[:, 2].values, sub.iloc[:, 3].values])

    grid = np.full(
        (x_edges.size - 1, y_edges.size - 1),
        np.nan,
        dtype=sub[value_col].dtype,
    )

    # ---- 3) Izgarayı doldur --------------------------------------------
    for lb_x, ub_x, lb_y, ub_y, v in sub.itertuples(index=False, name=None):
        ix = np.searchsorted(x_edges, lb_x, side="left")
        iy = np.searchsorted(y_edges, lb_y, side="left")
        grid[ix, iy] = v

    return x_edges, y_edges, grid


def _lookup_values(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Vectorised lookup of interaction values for each point.

    Points falling outside **any** rectangle get value 0.
    """
    # Which cell?  (right-side search, then step back 1)
    ix = np.searchsorted(x_edges, x_vals, side="right") - 1
    iy = np.searchsorted(y_edges, y_vals, side="right") - 1

    # Valid indices lie within the grid
    valid = (ix >= 0) & (ix < grid.shape[0]) & (iy >= 0) & (iy < grid.shape[1])

    values = np.zeros_like(x_vals, dtype=grid.dtype)
    values[valid] = grid[ix[valid], iy[valid]]

    return values


def interaction_scatter_plot(
    X: pd.DataFrame | np.ndarray,
    df: pd.DataFrame,
    col_1: int,
    col_2: int,
    figsize: Tuple[float, float] = (10.0, 8.0),
    ticks_fontsize: float = 10.0,
    title_fontsize: float = 16.0,
    label_fontsizes: float = 14.0,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color_bar_label: Optional[str] = None,
) -> None:
    """
    Creates a scatter plot of feature values with colors representing interaction values.

    Parameters
    ----------
    X : pd.DataFrame
        Input data containing feature values
    df : pd.DataFrame
        A DataFrame containing interaction data with columns `_lb`, `_ub`, `_lb`, `_ub`, and `value`.
        The first four columns represent intervals for two features, where each pair (_lb, _ub) defines
        the bounds of one feature. The last column, `value`, contains the interaction values for each pair.
    col_1 : int
        Index of first feature in X
    col_2 : int
        Index of second feature in X
    figsize : tuple of float, default (10.0, 6.0)
        Width and height of the plot in inches.
    ticks_fontsize : float, default 10.0
        Font size for axis tick labels,
    title_fontsize : float, default 16.
        Font size for plot title, by
    title : str, optional, default None
        The title displayed at the top of the plot. If `None`, no title is shown.
    xlabel : str, optional, default None
        Label for the x-axis. If None, it will default to the feature name.
    xlabel : str, optional, default None
        Label for the y-axis. If None, it will default to the feature name.
    color_bar_label : str, optional, default None
        Colorbar label, If None, it will default to "Impact".
    """
    # ---------------------------------------------------------------------
    # 3-a  Data wrangling
    # ---------------------------------------------------------------------
    try:
        X = np.asarray(X)
    except Exception as exc:
        raise ValueError(
            "Cannot convert X to a NumPy array – check its type/shape."
        ) from exc

    _validate_interaction_scatter_plot_parameters(
        X=X,
        df=df,
        col_1=col_1,
        col_2=col_2,
        figsize=figsize,
        ticks_fontsize=ticks_fontsize,
        title_fontsize=title_fontsize,
        label_fontsizes=label_fontsizes,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color_bar_label=color_bar_label,
    )

    x_vals = X[:, col_1].astype(float)
    y_vals = X[:, col_2].astype(float)

    # ---------------------------------------------------------------------
    # 3-b  Build or reuse the lookup grid
    # ---------------------------------------------------------------------
    x_edges, y_edges, grid = _build_lookup(df)

    values = _lookup_values(x_vals, y_vals, x_edges, y_edges, grid)

    # ---------------------------------------------------------------------
    # 3-c  Plot
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Symmetric normalisation around 0
    max_abs = float(np.nanmax(np.abs(values))) or 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    scatter = ax.scatter(
        x_vals,
        y_vals,
        c=values,
        cmap="coolwarm",
        norm=norm,
        edgecolors="black",
    )

    # Axis labels
    ax.set_xlabel(
        xlabel if xlabel is not None else df.columns[0][:-3],
        fontsize=label_fontsizes,
    )
    ax.set_ylabel(
        ylabel if ylabel is not None else df.columns[2][:-3],
        fontsize=label_fontsizes,
    )

    # Colour-bar
    cbar = plt.colorbar(scatter)
    cbar.ax.tick_params(labelsize=ticks_fontsize)
    cbar.set_label(
        color_bar_label if color_bar_label is not None else "Impact",
        fontsize=label_fontsizes,
    )

    # Title
    ax.set_title(
        title if title is not None else "Interaction Scatter Plot",
        fontsize=title_fontsize,
    )

    plt.tight_layout()
    plt.show()
