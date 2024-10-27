import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from .plot_utils import (
    _create_intervals,
    _replace_infinity,
    _validate_feature_plot_parameters,
    _validate_range_plot_parameters,
    _validate_interaction_plot_parameters,
    _validate_bar_plot_parameters,
)

from typing import List, Tuple
from numpy.typing import ArrayLike


def bar_plot(
    values: np.ndarray,
    raw_score: float,
    figsize: Tuple[int, int] = (8, 6),
    columns: ArrayLike = None,
    max_col: int | None = 20,
    title: str | None = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 12.0,
    show_raw_score: bool = True,
) -> None:
    """
    Creates a horizontal bar plot illustrating the contribution of each feature 
    in a dataset. This plot highlights the positive and negative contributions 
    distinctly with color-coded bars, providing a clear visual representation 
    of each feature’s impact on a model’s output or a decision-making process.
    
    Parameters
    ----------
    values : np.ndarray
        An array containing the contribution values of each feature. Each value
        represents the magnitude and direction (positive or negative) of the
        feature's contribution to the overall outcome.
    raw_score : float
        The expected value of the model given the provided dataset.
    figsize : tuple of float, optional, default=(8.0, 6.0)
        Width and height of the plot in inches.
    columns : list or ArrayLike, optional
        A list of names for the features, used as labels on the y-axis. If `None`,
        feature indices are labeled as "Column X" for each feature.
    max_col : int or None, optional, default=20
        The maximum number of features to display in the plot, chosen based on 
        their absolute contribution values. If `None`, all features will be shown.
    title : str or None, optional
        The title displayed at the top of the plot. If `None`, no title is shown.
    title_fontsize : float, optional, default=12.0
        Font size for the plot title.
    label_fontsize : float, optional, default=12.0
        Font size for the y-axis labels (feature names).
    show_raw_score : bool, optional, default=True
        Whether to display the `raw_score` value on the plot. If `True`, the raw 
        score is displayed at the top-right corner of the plot area.
    
    Returns
    -------
    None
        Displays the plot.
    
    Notes
    -----
    - Rows with only zero values are automatically excluded.
    """

    _validate_bar_plot_parameters(
        values=values,
        raw_score=raw_score,
        columns=columns,
        title=title,
        max_col=max_col,
        figsize=figsize,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        show_raw_score=show_raw_score,
    )

    # Identify non-zero contributions
    used_cols = np.where(values != 0)[0]
    if len(used_cols) == 0:
        raise ValueError("All contribution values are zero. There is nothing to plot.")
    values = values[used_cols]

    # Sort contributions by absolute value
    sorted_indices = np.argsort(np.abs(values))
    adjusted_values = values[sorted_indices]
    used_cols = used_cols[sorted_indices]

    # Limit the number of features displayed
    if max_col is not None:
        adjusted_values = adjusted_values[-max_col:]
        used_cols = used_cols[-max_col:]

    # Assign colors based on positive or negative contributions
    colors_list = ["green" if val > 0 else "red" for val in adjusted_values]

    fig, ax = plt.subplots(figsize=figsize)

    # Use provided columns or default to "Column X" labels
    if columns is None:
        ylabels = [f"Column {i}" for i in used_cols]
    else:
        ylabels = [columns[i] for i in used_cols]

    y_positions = np.arange(len(used_cols))

    # Create horizontal bars
    bars = ax.barh(y_positions, adjusted_values, color=colors_list)

    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, fontsize=label_fontsize)
    ax.set_xlabel("Contribution")
    ax.invert_yaxis()  # Highest contributions at the top

    # Calculate data range
    max_val = max(adjusted_values)
    min_val = min(adjusted_values)
    data_range = max_val - min_val if max_val != min_val else max_val

    # Define padding as a fraction of the data range
    padding_fraction = 0.05  # 5% of the data range
    left_padding = data_range * padding_fraction
    right_padding = data_range * padding_fraction

    # Adjust x-limits with scaled padding
    ax.set_xlim([min_val - left_padding, max_val + right_padding])

    # Add text labels with proportional offset
    texts = []
    for index, bar in enumerate(bars):
        bar_value = adjusted_values[index]
        sign = "+" if bar_value > 0 else "-"
        text_color = "green" if bar_value > 0 else "red"
        ha = "left" if bar_value > 0 else "right"
        # Use the exact y-position of the bar
        y_pos = y_positions[index]

        # Add text labels
        text = ax.text(
            bar.get_width(),
            y_pos,
            f"{sign}{abs(bar_value):.2f}",
            ha=ha,
            va="center",
            color=text_color,
        )
        texts.append(text)

    # Draw the figure to get accurate text positions
    fig.canvas.draw()

    # Get the renderer
    renderer = fig.canvas.get_renderer()

    # Measure text extents and adjust xlim if necessary
    max_right = ax.get_xlim()[1]
    max_left = ax.get_xlim()[0]

    for text in texts:
        bbox = text.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())

        if text.get_ha() == "left":
            if bbox_data.x1 > max_right:
                # Add extra space on the right
                max_right = bbox_data.x1 + data_range * 0.05  # Add extra padding
        else:  # 'right' alignment
            if bbox_data.x0 < max_left:
                # Add extra space on the left
                max_left = bbox_data.x0 - data_range * 0.05  # Add extra padding

    # Apply the new xlim
    ax.set_xlim([max_left, max_right])

    # Position the raw score text inside the plot area
    if show_raw_score:
        ax.text(
            1.00,
            1.00,
            f"Raw Score: {raw_score:.3f}",
            fontsize=10,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

    plt.tight_layout()

    if title is not None:
        plt.title(title, fontsize=title_fontsize)

    plt.show()


def range_plot(
    values: np.ndarray,
    raw_score: float,
    split_points: List[List[float]],
    scale: float = 2.0,
    columns: List[str] = None,
    max_col: int = 20,
    title: str = None,
    label_fontsize: float = 9.0,
    title_fontsize: float = 12.0,
    interval_fontsize: float = 4.5,
    value_fontsize: float = 5.5,
    show_raw_score: bool = True,
) -> None:
    """
    Plots a combined grid of values and intervals with color intensity representing the magnitude of values.
    Rows are sorted by a custom range calculation to emphasize the most variable rows:

    This method requires the `detailed` parameter to be `True` in the output from
    the `analyze_data` method of the `treemind.Explainer` class (`analyze_data(self, x: ArrayLike, detailed: bool = True) 
    -> Tuple[np.ndarray, List[np.ndarray], float`).

    Parameters
    ----------
    values : np.ndarray
        A 2D array where each row contains values to plot in the grid.
    raw_score : float
        The raw score associated with the values, displayed in the plot's upper right.
    split_points : List[np.ndarray[float]]
        A list of point intervals corresponding to the values in each row.
    scale : float, optional, default 2.0
        Scaling factor for figure size
    columns : list or ArrayLike, optional
        A list of names for the features, used as labels on the y-axis. If `None`,
        feature indices are labeled as "Column X" for each feature.
    max_col : int or None, optional, default=20
        The maximum number of features to display in the plot, chosen based on 
        their absolute contribution values. If `None`, all features will be shown.
    title : str or None, optional
        The title displayed at the top of the plot. If `None`, no title is shown.
    label_fontsize : float, optional, default is 9.0
        Font size for the y-axis labels
    title_fontsize : float, optional, default 12.0
        Font size for the plot title
    interval_fontsize : float, optional, default 4.5.
        Font size for interval labels displayed on each bar,
    value_fontsize : float, optional, default 5.5.
        Font size for value labels displayed below each bar
    show_raw_score : bool, optional, default True
        If True, displays the raw score in the plot

    Returns
    -------
    None
        Displays the plot.

    Notes
    -----
    - Rows with only zero values are automatically excluded.
    """

    _validate_range_plot_parameters(
        values=values,
        raw_score=raw_score,
        split_points=split_points,
        scale=scale,
        columns=columns,
        max_col=max_col,
        title=title,
        label_fontsize=label_fontsize,
        title_fontsize=title_fontsize,
        interval_fontsize=interval_fontsize,
        value_fontsize=value_fontsize,
        show_raw_score=show_raw_score,
    )

    # Filter out rows where all values are zero
    used_cols = np.where(np.logical_not(np.all(values == 0, axis=1)))[0]
    values = values[used_cols, :]
    split_points = [x for i, x in enumerate(split_points) if i in used_cols]

    if columns is not None:
        columns = [columns[i] for i in used_cols]

    # Calculate custom ranges for each row based on actual value lengths
    value_ranges = []
    for row, row_points in zip(values, split_points):
        # Use only the length of values for this row
        row_values = row[: len(row_points)]
        max_val = np.max(row_values)
        min_val = np.min(row_values)

        if max_val > 0 and min_val < 0:
            # If we have both positive and negative values, use algebraic sum
            range_val = abs(max_val + min_val)
        else:
            # Otherwise use absolute difference
            range_val = abs(max_val - min_val)

        value_ranges.append(range_val)

    # Convert to numpy array for sorting
    value_ranges = np.array(value_ranges)

    # Get sorting indices (ascending order - smaller differences first)
    sort_indices = np.argsort(value_ranges)[::-1]

    # Sort the values, points, and columns
    values = values[sort_indices]
    split_points = [split_points[i] for i in sort_indices]
    if columns is not None:
        columns = [columns[i] for i in sort_indices]

    # Apply max_col filter from bottom up
    if max_col is not None:
        values = values[:max_col, :]
        split_points = split_points[:max_col]
        if columns is not None:
            columns = columns[:max_col]

    n_rows, n_cols = values.shape

    fig, ax = plt.subplots(figsize=(n_cols * 0.7 * scale, n_rows * 0.6 * scale))

    # Color maps for positive and negative values
    cmap_pos = sns.light_palette("green", as_cmap=True)
    cmap_neg = sns.light_palette("red", as_cmap=True)

    # Calculate max and min values using only valid lengths per row
    max_values = []
    min_values = []
    for row, row_points in zip(values, split_points):
        row_values = row[: len(row_points)]
        max_values.append(np.max(row_values))
        min_values.append(np.min(row_values))

    max_value = max(max_values) if max_values and max(max_values) > 0 else 1
    min_value = min(min_values) if min_values and min(min_values) < 0 else -1

    bar_width = 0.7  # Width of each bar
    bar_height = 0.6  # Height of each bar

    # Calculate base font sizes
    base_interval_font_size = interval_fontsize * scale  # Reduced size
    base_value_font_size = value_fontsize * scale  # Reduced size

    # Plot each value and interval
    for i, (row, row_points) in enumerate(zip(values, split_points)):
        intervals = _create_intervals(row_points)
        # Only plot up to the length of points for this row
        for j, (val, interval) in enumerate(zip(row[: len(row_points)], intervals)):
            if val != 0:
                # Select color based on value
                if val > 0:
                    color = cmap_pos(val / max_value)
                else:
                    color = cmap_neg(abs(val) / abs(min_value))

                # Draw the bar
                ax.barh(
                    i,
                    bar_width,
                    left=j * bar_width,
                    color=color,
                    height=bar_height,
                    alpha=0.9,
                )

                # Compute brightness for text color
                color_hsv = rgb_to_hsv(color[:3])
                brightness = color_hsv[2]
                text_color = "white" if brightness < 0.5 else "black"

                # Place the range text at the top of the bar
                ax.text(
                    j * bar_width + bar_width / 2,
                    i + 0.15,
                    interval,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=base_interval_font_size,
                )

                # Place the value text at the bottom of the bar
                ax.text(
                    j * bar_width + bar_width / 2,
                    i - 0.15,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=base_value_font_size,
                )

    # Set y-axis labels
    ax.set_yticks(np.arange(n_rows))

    ax.set_xticks([])
    ax.set_xticklabels([])

    if columns is not None:
        ax.set_yticklabels(columns, fontsize=label_fontsize * scale)
    else:
        ax.set_yticklabels(
            [f"Column {i}" for i in range(n_rows)], fontsize=label_fontsize * scale
        )

    # Set x-axis limits
    ax.set_xlim(0, n_cols * bar_width)

    # Remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add raw score text
    if show_raw_score:
        ax.text(
            1,
            1,
            f"Raw Score: {raw_score:.3f}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=7 * scale,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    if title is not None:
        plt.title(title, fontsize=title_fontsize * scale)

    plt.tight_layout()
    plt.show()


def feature_plot(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    show_min_max: bool = False,
    xticks_n: int = 10,
    yticks_n: int = 10,
    ticks_decimal: int = 3,
    ticks_fontsize: float = 10.0,
    title_fontsize: float = 16.0,
    label_fontsizes: float = 14.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """
    Plots the mean, min, and max values of a feature based on tree split points.
    
    This method takes as input the output DataFrame from the `analyze_feature` 
    method of the `treemind.Explainer` class (`analyze_feature(self, col: int) -> pd.DataFrame`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the feature data with the following columns:
        - 'feature_lb': Lower bound of the feature range (tree split point).
        - 'feature_ub': Upper bound of the feature range (tree split point).
        - 'mean': Mean value of the feature within this range.
        - 'min': Minimum value of the feature within this range.
        - 'max': Maximum value of the feature within this range.
    figsize : tuple of int, optional, default (10.0, 6.0)
        Width and height of the plot in inches.
    show_min_max : bool, optional, default False
        If True, shaded areas representing the min and max values will be displayed.
    xticks_n : int, optional, default 10
        Number of tick marks to display on the x-axis.
    yticks_n : int, optional, default 10
        Number of tick marks to display on the y-axis.
    ticks_decimal : int, optional, default 3
        Number of decimal places for tick labels
    ticks_fontsize : float, optional, default 10.0
        Font size for axis tick labels,
    title_fontsize : float, optional, default 16.0
        Font size for the plot title.
    title : str, optional, default None
        The title displayed at the top of the plot. If `None`, no title is shown.
    xlabel : str, optional, default None
        Label for the x-axis. If None, it will default to the feature name.
    ylabel : str, optional, default None
        Label for the y-axis. Defaults to "Value" if not specified.

    Returns
    -------
    None
        Displays the plot.
    """

    # Validate parameters
    _validate_feature_plot_parameters(
        df=df,
        figsize=figsize,
        show_min_max=show_min_max,
        xticks_n=xticks_n,
        yticks_n=yticks_n,
        ticks_fontsize=ticks_fontsize,
        ticks_decimal=ticks_decimal,
        title=title,
        title_fontsize=title_fontsize,
        label_fontsizes=label_fontsizes,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    column_name = df.columns[1]

    # Set default labels if None
    xlabel = xlabel if xlabel is not None else column_name[:-3]
    ylabel = ylabel if ylabel is not None else "Value"

    df = _replace_infinity(df, column_name=column_name)

    extend_ratio = 0.05 * (df[column_name].max() - df[column_name].min())

    min_row = df.iloc[df[column_name].argmin()].copy()
    min_row[column_name] -= extend_ratio

    max_row = df.iloc[df[column_name].argmax()].copy()
    max_row[column_name] += extend_ratio

    df = pd.concat([min_row.to_frame().T, df, max_row.to_frame().T], ignore_index=True)

    plt.figure(figsize=figsize)

    sns.lineplot(
        data=df,
        x=column_name,
        y="mean",
        color="blue",
        linewidth=2,
        drawstyle="steps-pre",
    )

    if show_min_max:
        plt.fill_between(
            df[column_name],
            df["min"],
            df["max"],
            color="gray",
            alpha=0.3,
            label="Min-Max Range",
            step="post",
        )

    # Set the plot title
    if title is None:
        plt.title(
            f"Contribution of {xlabel}", fontsize=title_fontsize, fontweight="bold"
        )
    else:
        plt.title(title, fontsize=title_fontsize, fontweight="bold")

    plt.gca().set_facecolor("whitesmoke")

    # Set x-ticks
    x_ticks = np.linspace(df[column_name].min(), df[column_name].max(), num=xticks_n)
    plt.xticks(
        x_ticks,
        ["-∞"] + [f"{tick:.{ticks_decimal}f}" for tick in x_ticks[1:-1]] + ["+∞"],
        fontsize=ticks_fontsize,
    )

    # Determine y-ticks range
    if show_min_max:
        y_min = df["min"].min()
        y_max = df["max"].max()
    else:
        y_min = df["mean"].min()
        y_max = df["mean"].max()

    # Set y-ticks
    y_ticks = np.linspace(y_min, y_max, num=yticks_n)
    plt.yticks(
        y_ticks,
        [f"{tick:.{ticks_decimal}f}" for tick in y_ticks],
        fontsize=ticks_fontsize,
    )

    plt.xlabel(xlabel, fontsize=label_fontsizes)
    plt.ylabel(ylabel, fontsize=label_fontsizes)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if show_min_max:
        plt.legend()

    plt.tight_layout()
    plt.show()


def interaction_plot(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (10.0, 8.0),
    axis_ticks_n: int = 10,
    cbar_ticks_n: int = 10,
    ticks_decimal: int = 3,
    ticks_fontsize: float = 10.0,
    title_fontsize: float = 16.0,
    label_fontsizes: float = 14.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color_bar_label: str | None = None,
) -> None:
    """
    Plots to visualize interactions between two features using model split points.
    
    This method takes as input the output DataFrame from the `analyze_interaction` 
    method of the `treemind.Explainer` class (`analyze_interaction(self, main_col: int, sub_col: int) -> pd.DataFrame`).

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing interaction data with columns `_lb`, `_ub`, `_lb`, `_ub`, and `value`.
        The first four columns represent intervals for two features, where each pair (_lb, _ub) defines
        the bounds of one feature. The last column, `value`, contains the interaction values for each pair.
    figsize : tuple of float, optional, default (10.0, 6.0)
        Width and height of the plot in inches.
    axis_ticks_n : int, optional, default 10
        Number of ticks on both axis
    cbar_ticks_n : int, optional, default 10
        Number of ticks on the colorbar
    ticks_decimal : int, optional, default 3
        Number of decimal places for tick labels
    ticks_fontsize : float, optional, default 10.0
        Font size for axis tick labels,
    title_fontsize : int or float, optional, default 16.
        Font size for plot title, by
    title : str or None, optional
        The title displayed at the top of the plot. If `None`, no title is shown.
    xlabel : str, optional, default None
        Label for the x-axis. If None, it will default to the feature name.
    xlabel : str, optional, default None
        Label for the y-axis. If None, it will default to the feature name.
    color_bar_label : str, optional, default None
        Colorbar label, If None, it will default to "Impact".
        
    Returns
    -------
    None
        Displays the plot.
    """
    _validate_interaction_plot_parameters(
        df=df,
        figsize=figsize,
        axis_ticks_n=axis_ticks_n,
        cbar_ticks_n=cbar_ticks_n,
        ticks_decimal=ticks_decimal,
        ticks_fontsize=ticks_fontsize,
        title_fontsize=title_fontsize,
        label_fontsizes=label_fontsizes,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color_bar_label=color_bar_label,
    )

    df = (
        pd.concat(
            [
                df,
                df[np.isneginf(df.iloc[:, 2])]
                .copy()
                .assign(**{df.columns[3]: -np.inf}),
                df[np.isneginf(df.iloc[:, 0])]
                .copy()
                .assign(**{df.columns[1]: -np.inf}),
            ],
            ignore_index=True,
        )
        .iloc[:, [1, 3, 4]]
        .pipe(
            lambda df: df.rename(columns={col: col.rstrip("_ub") for col in df.columns})
        )
    )

    column1 = xlabel if xlabel is not None else df.columns[0]
    column2 = ylabel if ylabel is not None else df.columns[1]

    df = _replace_infinity(df, column1, infinity_type="positive")
    df = _replace_infinity(df, column1, infinity_type="negative")
    df = _replace_infinity(df, column2, infinity_type="positive")
    df = _replace_infinity(df, column2, infinity_type="negative")

    values = df["value"]

    # Sort the dataframe to ensure correct rectangle placement
    df_sorted = df.sort_values([column1, column2])

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique sorted values for the two columns
    unique_x = np.sort(df[column1].unique())
    unique_y = np.sort(df[column2].unique())

    # Create mappings from value to index for quick lookup
    x_to_index = {x: idx for idx, x in enumerate(unique_x)}
    y_to_index = {y: idx for idx, y in enumerate(unique_y)}

    # Convert data columns to NumPy arrays for vectorized operations
    x = df_sorted[column1].values
    y = df_sorted[column2].values
    values = df_sorted["value"].values

    # Map x and y values to their indices
    x_idx = np.array([x_to_index[val] for val in x])
    y_idx = np.array([y_to_index[val] for val in y])

    # Compute the left and bottom edges of rectangles
    left = np.where(x_idx > 0, unique_x[x_idx - 1], df[column1].min())
    bottom = np.where(y_idx > 0, unique_y[y_idx - 1], df[column2].min())

    # Compute width and height of rectangles
    width = x - left
    height = y - bottom

    if values.max() < 0:  # All values are negative
        colormap = plt.get_cmap("Blues")
        norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    elif values.min() > 0:  # All values are positive
        colormap = plt.get_cmap("Reds")
        norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    else:  # Both negative and positive values
        colormap = plt.get_cmap("coolwarm")
        norm = TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())

    colors = colormap(norm(values))

    # Create rectangles using list comprehension
    rectangles = [
        Rectangle((l, b), w, h) for l, b, w, h in zip(left, bottom, width, height)
    ]

    # Use PatchCollection to add all rectangles at once
    pc = PatchCollection(rectangles, facecolor=colors, edgecolor="none")
    ax.add_collection(pc)

    # Set axis limits
    x_min, x_max = df[column1].min(), df[column1].max()
    y_min, y_max = df[column2].min(), df[column2].max()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Set x-axis ticks
    x_ticks = np.linspace(x_min, x_max, axis_ticks_n)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        ["-∞"] + [f"{tick:.{ticks_decimal}f}" for tick in x_ticks[1:-1]] + ["+∞"],
        fontsize=ticks_fontsize,
    )

    # Set y-axis ticks
    y_ticks = np.linspace(y_min, y_max, axis_ticks_n)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        ["-∞"] + [f"{tick:.{ticks_decimal}f}" for tick in y_ticks[1:-1]] + ["+∞"],
        fontsize=ticks_fontsize,
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)

    # Get real minimum and maximum values
    real_min = values.min()
    real_max = values.max()

    if real_min < 0 and real_max > 0:  # Both positive and negative values
        pos_ticks = np.linspace(0, real_max, cbar_ticks_n // 2 + 1)
        neg_ticks = np.linspace(real_min, 0, cbar_ticks_n // 2 + 1)[:-1]
        ticks = np.concatenate([neg_ticks, pos_ticks])
    else:  # Only negative or only positive values
        ticks = np.linspace(real_min, real_max, cbar_ticks_n)

    # Update colorbar
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{x:.{ticks_decimal}f}" for x in ticks])
    cbar.ax.tick_params(labelsize=ticks_fontsize)
    cbar.set_label(
        "Impact" if color_bar_label is None else color_bar_label,
        fontsize=label_fontsizes,
    )

    ax.set_xlabel(column1, fontsize=label_fontsizes)
    ax.set_ylabel(column2, fontsize=label_fontsizes)
    ax.set_title(
        "Interaction Plot" if title is None else title, fontsize=title_fontsize
    )
    plt.tight_layout()
    plt.show()


def interaction_scatter_plot(
    X: pd.DataFrame,
    df: pd.DataFrame,
    col_1_index: int,
    col_2_index: int,
    figsize: Tuple[float, float] = (10.0, 8.0),
    ticks_fontsize: float = 10.0,
    title_fontsize: float = 16.0,
    label_fontsizes: float = 14.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color_bar_label: str | None = None,
) -> None:
    """
    Creates a scatter plot based on interaction data and feature values.

    Parameters
    ----------
    X : pd.DataFrame
        Input data containing feature values
    df : pd.DataFrame
        Interaction data with columns _lb, _ub for both features and value column
    col_1_index : int
        Index of first feature in X
    col_2_index : int
        Index of second feature in X
    """

    # Extract values from X
    if type(X) == pd.DataFrame:
        x_values = X.iloc[:, col_1_index].values
        y_values = X.iloc[:, col_2_index].values
        
    # Extract values from X
    else:
        x_values = X[:, col_1_index]
        y_values = X[:, col_2_index]

    cols = df.columns

    # Convert bounds to arrays for vectorized comparison
    x_lb_vals = df[cols[0]].values[:, np.newaxis]
    x_ub_vals = df[cols[1]].values[:, np.newaxis]
    y_lb_vals = df[cols[2]].values[:, np.newaxis]
    y_ub_vals = df[cols[3]].values[:, np.newaxis]
    df_values = df["value"].values

    # Check if x and y values fall within any of the intervals
    x_in_bounds = (x_values >= x_lb_vals) & (x_values <= x_ub_vals)
    y_in_bounds = (y_values >= y_lb_vals) & (y_values <= y_ub_vals)

    # Initialize array for values
    values = np.zeros(len(X))

    # Combine x and y conditions and assign values
    mask = x_in_bounds & y_in_bounds
    matching_indices = mask.any(axis=0)
    values[matching_indices] = df_values[mask.argmax(axis=0)[matching_indices]]

    fig, ax = plt.subplots(figsize=figsize)

    # Determine colormap based on value range
    if values.max() < 0:  # All values are negative
        colormap = plt.get_cmap("Blues")
        norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    elif values.min() > 0:  # All values are positive
        colormap = plt.get_cmap("Reds")
        norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    else:  # Both negative and positive values
        colormap = plt.get_cmap("coolwarm")
        norm = TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())

    # Create scatter plot
    scatter = ax.scatter(
        x_values, y_values, c=values, cmap=colormap, norm=norm, edgecolors="black"
    )


    # Set axis labels
    ax.set_xlabel(
        xlabel if xlabel is not None else f"Feature {col_1_index}",
        fontsize=label_fontsizes,
    )
    ax.set_ylabel(
        ylabel if ylabel is not None else f"Feature {col_2_index}",
        fontsize=label_fontsizes,
    )

    # Add colorbar with specified number of ticks
    cbar = plt.colorbar(scatter)

    cbar.ax.tick_params(labelsize=ticks_fontsize)

    cbar.set_label(
        "Impact" if color_bar_label is None else color_bar_label,
        fontsize=label_fontsizes,
    )

    # Set title
    ax.set_title(
        "Interaction Scatter Plot" if title is None else title, fontsize=title_fontsize
    )

    plt.tight_layout()
    plt.show()

    print()