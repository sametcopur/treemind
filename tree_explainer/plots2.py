import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import seaborn as sns
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import List, Tuple

# sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


def plot_bar(values: np.ndarray, raw_score: float, columns: list = None) -> None:
    """
    Creates a horizontal bar plot showing the contribution of each feature.

    Parameters
    ----------
    values : np.ndarray
        An array of contribution values for each feature.
    raw_score : float
        The raw score associated with the contributions.
    columns : list, optional
        A list of column names for labeling. If None, column indices will be used.

    Returns
    -------
    None
        Displays the bar plot.

    Notes
    -----
    Positive contributions are shown in green, negative in red.
    The function only considers features with non-zero contributions.
    """
    # Identify non-zero contributions
    used_cols = np.where(values != 0)[0]
    values = values[used_cols]

    # Sort contributions by absolute value
    sorted_indices = np.argsort(np.abs(values))
    adjusted_values = values[sorted_indices]
    used_cols = used_cols[sorted_indices]

    # Assign colors based on positive or negative contributions
    colors_list = ["green" if val > 0 else "red" for val in adjusted_values]

    fig, ax = plt.subplots()
    bars = ax.barh(np.arange(len(used_cols)), adjusted_values, color=colors_list)

    ax.axvline(0, color="black", linewidth=0.8)

    # Use provided columns or default to "Column X" labels
    if columns is None:
        y_labels = [f"Column {i}" for i in used_cols]
    else:
        y_labels = [columns[i] for i in used_cols]

    ax.set_yticks(np.arange(len(used_cols)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Contribution")
    ax.invert_yaxis()  # Highest contributions at the top

    max_val = max(adjusted_values)
    min_val = min(adjusted_values)

    # Set consistent padding for both small and large bars
    # Define a reasonable minimum and maximum padding
    min_padding = 0.0 if min_val > 0 else 0.4  # Minimum padding for very small bars
    max_padding = 0.0 if max_val < 0 else 1.15  # Maximum padding for very large bars

    # Calculate the padding relative to the magnitude of the bars but within the defined limits
    left_padding = max(min_padding, min(abs(min_val) * 0.2, max_padding))
    right_padding = max(min_padding, min(abs(max_val) * 0.2, max_padding))

    # Set the x-limits with consistent padding
    ax.set_xlim([min_val - left_padding, max_val + right_padding])

    for index, bar in enumerate(bars):
        bar_value = adjusted_values[index]
        sign = "+" if bar_value > 0 else "-"  # Adding the sign before the value

        # Set text color to match the bar color (green or red)
        text_color = "green" if bar_value > 0 else "red"

        # Adjust the text position to be closer to the bar and in line with its edge
        offset = 0.04 if bar_value > 0 else -0.04
        ax.text(
            bar.get_width() + offset,  # Closer offset for outside text
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{abs(bar_value):.2f}",  # Formatting value with sign
            ha="left" if bar_value > 0 else "right",  # Align text based on the sign
            va="center",
            color=text_color,  # Text color matches the bar
        )

    # Position the raw score text inside the plot area
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
    plt.show()


def plot_values_points(
    values: np.ndarray, raw_score: float, points: List[List[float]], scale: float = 1
) -> None:
    """
    Plots a grid of values with color intensity representing the magnitude.

    Parameters
    ----------
    values : np.ndarray
        A 2D array of values to be plotted.
    raw_score : float
        The raw score associated with the values.
    points : list of list of float
        A list of points associated with the values.
    scale : float, optional
        Scaling factor for the figure size, by default 1.

    Returns
    -------
    None
        Displays the grid plot.

    Notes
    -----
    Positive values are shown in shades of green, negative in shades of red.
    The brightness of the color corresponds to the magnitude of the value.
    Only non-zero values are plotted.
    """

    # Filter out rows where all values are zero
    used_cols = np.where(np.logical_not(np.all(values == 0, axis=1)))[0]
    values = values[used_cols, :]
    points = [x for i, x in enumerate(points) if i in used_cols]

    n_rows, n_cols = values.shape

    fig, ax = plt.subplots(figsize=(n_cols * 2 * scale, n_rows * scale))

    # Color maps for positive and negative values
    cmap_pos = sns.light_palette("green", as_cmap=True)
    cmap_neg = sns.light_palette("red", as_cmap=True)

    # Find the maximum and minimum values for normalization
    max_value = np.max(values) if np.max(values) > 0 else 1
    min_value = np.min(values) if np.min(values) < 0 else -1

    bar_width = 1  # Width of each bar
    bar_height = 0.8  # Height of each bar
    spacing = 0  # No spacing between bars

    # Plot each value
    for i, row in enumerate(values):
        for j, val in enumerate(row):
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
                    left=j * (bar_width + spacing),
                    color=color,
                    height=bar_height,
                    alpha=0.9,
                )

                # Compute brightness for text color
                color_hsv = rgb_to_hsv(color[:3])
                brightness = color_hsv[2]
                text_color = "white" if brightness < 0.5 else "black"

                # Place the value text
                ax.text(
                    j * (bar_width + spacing) + bar_width / 2,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                )

    # Set y-axis labels
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f"Column {i}" for i in used_cols])

    # Set x-axis limits
    ax.set_xlim(0, n_cols)

    # Remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add raw score text
    ax.text(
        0.98,
        0.98,
        f"Raw Score: {raw_score:.3f}",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    plt.title("Grid Plot of Values")
    plt.tight_layout()
    plt.show()


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


def plot_points(
    values: np.ndarray, points: List[List[float]], scale: float = 1
) -> None:
    """
    Plots a grid of intervals with color intensity representing the magnitude.

    Parameters
    ----------
    values : np.ndarray
        A 2D array of values corresponding to intervals.
    points : list of list of float
        A list of points for each row to create intervals.
    scale : float, optional
        Scaling factor for the figure size, by default 1.

    Returns
    -------
    None
        Displays the interval grid plot.

    Notes
    -----
    Positive values are shown in shades of green, negative in shades of red.
    The intervals are displayed within the bars.
    """

    # Filter out rows where all values are zero
    used_cols = np.where(np.logical_not(np.all(values == 0, axis=1)))[0]
    values = values[used_cols, :]
    points = [points[i] for i in used_cols]

    # Convert points to intervals
    intervals = [_create_intervals(row) for row in points]

    n_rows = len(points)
    max_cols = max(len(row_points) for row_points in points)

    fig, ax = plt.subplots(figsize=(max_cols * 2 * scale, n_rows * scale))

    # Color maps for positive and negative values
    cmap_pos = sns.light_palette("green", as_cmap=True)
    cmap_neg = sns.light_palette("red", as_cmap=True)

    # Find the maximum and minimum values for normalization
    max_value = np.max(values) if np.max(values) > 0 else 1
    min_value = np.min(values) if np.min(values) < 0 else -1

    bar_width = 1  # Width of each bar
    bar_height = 0.8  # Height of each bar
    spacing = 0  # No spacing between bars

    # Plot each interval
    for i, (row_values, row_intervals) in enumerate(zip(values, intervals)):
        for j, (val, interval) in enumerate(zip(row_values, row_intervals)):
            # Select color based on value
            if val > 0:
                color = cmap_pos(val / max_value)
            else:
                color = cmap_neg(abs(val) / abs(min_value))

            # Draw the bar
            ax.barh(
                i,
                bar_width,
                left=j * (bar_width + spacing),
                color=color,
                height=bar_height,
                alpha=0.9,
            )

            # Compute brightness for text color
            color_hsv = rgb_to_hsv(color[:3])
            brightness = color_hsv[2]
            text_color = "white" if brightness < 0.5 else "black"

            # Place the interval text
            ax.text(
                j * (bar_width + spacing) + bar_width / 2,
                i,
                interval,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    # Set y-axis labels
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f"Column {i}" for i in used_cols])

    # Set x-axis limits
    ax.set_xlim(0, max_cols)

    # Remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.title("Grid Plot of Intervals")
    plt.tight_layout()
    plt.show()


def plot_feature(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plots the mean, min, and max values of a feature.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the feature data with columns 'mean', 'min', 'max'.
    figsize : tuple of int, optional
        Figure size, by default (10, 6).

    Returns
    -------
    None
        Displays the feature plot.

    Notes
    -----
    The function plots the mean line and fills between min and max values.
    """

    column_name = df.columns[0]

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
        drawstyle="steps-post",
    )

    plt.fill_between(
        df[column_name],
        df["min"],
        df["max"],
        color="gray",
        alpha=0.3,
        label="Min-Max Range",
        step="post",
    )

    plt.title(f"Contribution of {column_name}", fontsize=16, fontweight="bold")

    plt.gca().set_facecolor("whitesmoke")

    x_ticks = np.linspace(df[column_name].min(), df[column_name].max(), num=10)
    plt.xticks(x_ticks, [f"{tick:.2f}" for tick in x_ticks], fontsize=10)

    plt.xlabel(column_name, fontsize=14)
    plt.ylabel("Values", fontsize=14)

    plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.7)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_interaction(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 6),
    sigma: int = 3,
    grid_size: int = 100,
) -> None:
    """
    Plots a smooth filled contour plot of interactions between two features.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing columns for two features and their interaction values.
    figsize : tuple of int, optional
        Figure size, by default (8, 6).
    sigma : int, optional
        Standard deviation for Gaussian filter, by default 3.
    grid_size : int, optional
        Number of grid points along each axis, by default 100.

    Returns
    -------
    None
        Displays the contour plot.

    Notes
    -----
    The function creates a smooth contour plot using Gaussian filtering.
    """

    column1 = df.columns[1]
    column2 = df.columns[0]

    xi = np.linspace(df[column1].min(), df[column1].max(), grid_size)
    yi = np.linspace(df[column2].min(), df[column2].max(), grid_size)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((df[column1], df[column2]), df["values"], (xi, yi), method="linear")

    # Handle NaN values that may result from griddata interpolation
    zi = np.nan_to_num(zi, nan=0.0)

    zi_smooth = gaussian_filter(zi, sigma=sigma)

    plt.figure(figsize=figsize)
    contour = plt.contourf(xi, yi, zi_smooth, levels=50, cmap="coolwarm")

    cb = plt.colorbar(contour)
    cb.set_label("Impact")
    # plt.title("Smooth Interaction Plot with Gaussian Filter")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.tight_layout()
    plt.show()
