import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import seaborn as sns
import pandas as pd
from typing import List, Tuple
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from .plot_utils import _create_intervals, _replace_infinity, _check_columns
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def bar_plot(
    values: np.ndarray,
    raw_score: float,
    columns: List[str] = None,
    max_col: int = 20,
) -> None:
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
    max_col : int, optional
        The maximum number of features to display in the plot. If None, all features with
        non-zero contributions will be displayed.

    Returns
    -------
    None
        Displays the bar plot.

    Notes
    -----
    Positive contributions are shown in green, negative in red.
    The function only considers features with non-zero contributions.
    If `max_col` is specified, only the features with the largest absolute contributions
    will be displayed.
    """
    # Input validation
    if not isinstance(values, np.ndarray):
        raise TypeError("The 'values' parameter must be a numpy.ndarray.")
    if not isinstance(raw_score, (int, float)):
        raise TypeError(
            "The 'raw_score' parameter must be a numeric type (int or float)."
        )
    if columns is not None:
        # Assuming _check_columns is a function defined elsewhere
        _check_columns(columns)
        if len(columns) != len(values):
            raise ValueError(
                "The length of 'columns' must match the length of 'values'."
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

    fig, ax = plt.subplots()

    # Use provided columns or default to "Column X" labels
    if columns is None:
        y_labels = [f"Column {i}" for i in used_cols]
    else:
        y_labels = [columns[i] for i in used_cols]

    y_positions = np.arange(len(used_cols))

    # Create horizontal bars
    bars = ax.barh(y_positions, adjusted_values, color=colors_list)

    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
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

def range_plot(
    values: np.ndarray,
    raw_score: float,
    points: List[List[float]],
    scale: float = 1,
    columns: List[str] = None,
) -> None:
    """
    Plots a combined grid of values and intervals with color intensity representing the magnitude.
    Both the range and value are displayed within a single bar. Bars have width 0.7 and height 0.6 with no spacing.

    Parameters
    ----------
    values : np.ndarray
        A 2D array of values to be plotted.
    raw_score : float
        The raw score associated with the values.
    points : list of list of float
        A list of points associated with the values.
    scale : float, optional
        Scaling factor for the figure size and text sizes, by default 1.
    columns : list, optional
        A list of column names for labeling. If None, column indices will be used.

    Returns
    -------
    None
        Displays the combined grid plot.
    """

    # Filter out rows where all values are zero
    used_cols = np.where(np.logical_not(np.all(values == 0, axis=1)))[0]
    values = values[used_cols, :]
    points = [x for i, x in enumerate(points) if i in used_cols]

    n_rows, n_cols = values.shape

    fig, ax = plt.subplots(figsize=(n_cols * 0.7 * scale, n_rows * 0.6 * scale))

    # Color maps for positive and negative values
    cmap_pos = sns.light_palette("green", as_cmap=True)
    cmap_neg = sns.light_palette("red", as_cmap=True)

    # Find the maximum and minimum values for normalization
    max_value = np.max(values) if np.max(values) > 0 else 1
    min_value = np.min(values) if np.min(values) < 0 else -1

    bar_width = 0.7  # Width of each bar
    bar_height = 0.6  # Height of each bar

    # Calculate base font sizes
    base_interval_font_size = 4.5 * scale  # Reduced size
    base_value_font_size = 5.5 * scale     # Reduced size

    # Plot each value and interval
    for i, (row, row_points) in enumerate(zip(values, points)):
        intervals = _create_intervals(row_points)
        for j, (val, interval) in enumerate(zip(row, intervals)):
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

    if columns is not None:
        ax.set_yticklabels([columns[i] for i in used_cols], fontsize=9 * scale)
    else:
        ax.set_yticklabels([f"Column {i}" for i in used_cols], fontsize=9 * scale)

    # Set x-axis limits
    ax.set_xlim(0, n_cols * bar_width)

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
        fontsize=12 * scale,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    plt.title("Combined Grid Plot of Values and Intervals", fontsize=12 * scale)
    plt.tight_layout()
    plt.show()

def feature_plot(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plots the mean, min, and max values of a feature based on tree split points.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the feature data with the following columns:
        - 'feature_lb': The lower bound of the feature range (tree split point).
        - 'feature_ub': The upper bound of the feature range (tree split point).
        - 'mean': The mean value of the feature within this range.
        - 'min': The minimum value of the feature within this range.
        - 'max': The maximum value of the feature within this range.

    figsize : tuple of int, optional
        Size of the figure to be created, by default (10, 6). This parameter controls the
        width and height of the plot.

    Returns
    -------
    None
        This function does not return any values. Instead, it displays the feature plot.

    Notes
    -----
    This function visualizes how a particular feature behaves over different ranges of
    values, as defined by the split points of a decision tree-based model. The tree split
    points represent thresholds used by the model to partition the feature space into
    segments. The plot shows the mean value of the feature for each segment, while shading
    between the minimum and maximum values to illustrate the variability within that range.

    The x-axis represents the feature values, divided into intervals determined by the
    tree's split points. The y-axis shows the corresponding mean, minimum, and maximum
    values for each interval. The plot provides insight into how the feature's value
    distribution varies across the different split-defined segments. It can be helpful for
    understanding the relationship between the feature and the model's predictions.
    """

    column_name = df.columns[1]
    plot_column_name = column_name[:-3]

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

    plt.title(f"Contribution of {plot_column_name}", fontsize=16, fontweight="bold")

    plt.gca().set_facecolor("whitesmoke")

    x_ticks = np.linspace(df[column_name].min(), df[column_name].max(), num=10)
    plt.xticks(x_ticks, [f"{tick:.2f}" for tick in x_ticks], fontsize=10)

    plt.xlabel(plot_column_name, fontsize=14)
    plt.ylabel("Value", fontsize=14)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def interaction_plot(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "coolwarm",
    column_names: List[str] | None = None,
) -> None:
    """
    Plots a filled rectangle plot to visualize interactions between two features,
    using model split points and filling gaps to the left and bottom.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing columns for two features and their interaction values.
        The DataFrame should include at least three columns:

        - The first column represents the primary feature (main_col).
        - The second column represents the secondary feature (sub_col).
        - The third column contains the interaction values (typically the impact on
          the model's prediction scores).

    figsize : tuple of int, optional
        Size of the figure to be created, by default (10, 8).
        This parameter controls the width and height of the plot.

    cmap : str, optional
        Colormap to use for filling the rectangles, by default "coolwarm".
        The colormap represents the intensity of the interaction values.

    column_names : list of str, optional
        Names of the columns to be used for plotting. Should be a list of exactly
        two column names, corresponding to the features whose interactions are being plotted.
        If None, the function will use the first two columns of the DataFrame. If provided,
        it must be a list with two elements.

    Returns
    -------
    None
        This function does not return any values. Instead, it displays the interaction plot.

    Notes
    -----
    This function visualizes the interaction between two features based on model split points
    rather than using raw data points. The model split points refer to the thresholds or
    decision boundaries used by a decision tree-based model (such as gradient boosting or
    random forest) to partition the feature space. Each rectangle on the plot represents
    a region defined by these split points.

    The filled rectangles extend to the left and bottom to cover the gaps between the split
    points. The color of each rectangle corresponds to the interaction value, which indicates
    how much the combination of the two features influences the model's prediction. The
    color intensity is determined by the specified colormap (`cmap`), with a legend displayed
    to the right showing the range of interaction values.
    """

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

    if column_names is not None:
        if not isinstance(column_names, list):
            raise TypeError(
                "`column_names` should be a list of two strings representing column names."
            )
        if len(column_names) != 2:
            raise ValueError("`column_names` must contain exactly two elements.")

        column1 = column_names[0]
        column2 = column_names[1]

    else:
        column1 = df.columns[0]
        column2 = df.columns[1]

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

    # Normalize values for color mapping
    norm = plt.Normalize(values.min(), values.max())
    colormap = plt.get_cmap(cmap)
    colors = colormap(norm(values))

    # Create rectangles using list comprehension
    rectangles = [
        Rectangle((l, b), w, h) for l, b, w, h in zip(left, bottom, width, height)
    ]

    # Use PatchCollection to add all rectangles at once
    pc = PatchCollection(rectangles, facecolor=colors, edgecolor="none")
    ax.add_collection(pc)

    ax.set_xlim(df[column1].min(), df[column1].max())
    ax.set_ylim(df[column2].min(), df[column2].max())

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Impact")

    ax.set_xlabel(column1)
    ax.set_ylabel(column2)
    ax.set_title("Interaction Plot")
    plt.tight_layout()
    plt.show()
