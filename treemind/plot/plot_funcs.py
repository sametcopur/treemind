import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from scipy.stats import gaussian_kde

from .plot_utils import (
    _replace_infinity,
    _validate_feature_plot_parameters,
    _validate_interaction_plot_parameters,
    _validate_bar_plot_parameters,
    _find_tick_decimal,
)

from typing import List, Tuple
from numpy.typing import ArrayLike

import plotly.graph_objects as go
from ipywidgets import FloatSlider, VBox, HBox, Dropdown, Layout
from IPython.display import display


def bar_plot(
    values: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    columns: ArrayLike = None,
    max_col: int | None = 20,
    title: str | None = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 12.0,
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
        columns=columns,
        title=title,
        max_col=max_col,
        figsize=figsize,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    if values.shape[0] > 1:
        values = np.abs(values).mean(axis=0)

    else:
        values = values.ravel()

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

    plt.tight_layout()

    if title is not None:
        plt.title(title, fontsize=title_fontsize)

    plt.show()


def feature_plot(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    show_std: bool = False,
    show_range: bool = True,
    xticks_n: int = 10,
    yticks_n: int = 10,
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
        - 'count' : Average leaf_count within this range.
    figsize : tuple of int, optional, default (10.0, 6.0)
        Width and height of the plot in inches.
    show_std : bool, optional, default False
        If True, shaded areas representing the standart deviation values will be displayed.
    show_range : bool, default True
        If True, show leaf distribution within range.
    xticks_n : int, optional, default 10
        Number of tick marks to display on the x-axis.
    yticks_n : int, optional, default 10
        Number of tick marks to display on the y-axis.
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
    # Parameter validation
    _validate_feature_plot_parameters(
        df=df,
        figsize=figsize,
        show_std=show_std,
        show_range=show_range,
        xticks_n=xticks_n,
        yticks_n=yticks_n,
        ticks_fontsize=ticks_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        label_fontsizes=label_fontsizes,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    column_name = df.columns[1]
    lb_column_name = df.columns[0]

    # Set default labels if None
    xlabel = xlabel if xlabel is not None else column_name[:-3]
    ylabel = ylabel if ylabel is not None else "Value"

    df = _replace_infinity(df, column_name=column_name)
    df = _replace_infinity(df, column_name=lb_column_name, infinity_type="negative")

    extend_ratio = 0.05 * (df[column_name].max() - df[column_name].min())

    min_row = df.iloc[df[column_name].argmin()].copy()
    min_row[column_name] -= extend_ratio

    max_row = df.iloc[df[column_name].argmax()].copy()
    max_row[column_name] += extend_ratio

    df = pd.concat([min_row.to_frame().T, df, max_row.to_frame().T], ignore_index=True)

    if show_range:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 4]}, sharex=True, figsize=figsize
        )
        main_ax = ax2
    else:
        fig, main_ax = plt.subplots(figsize=figsize)

    # HISTOGRAM - only if show_range is True
    if show_range:
        mid_points = (df[lb_column_name] + df[column_name]) / 2
        weights = df["count"]
        min_value = df[column_name].min()
        max_value = df[column_name].max()

        kde = gaussian_kde(
            np.repeat(mid_points, weights.astype(int)), bw_method="silverman"
        )

        x_range = np.linspace(min_value, max_value, 200)
        kde_values = kde(x_range)

        ax1.plot(x_range, kde_values, color="#4C72B0", lw=2, alpha=0.7)
        ax1.fill_between(x_range, kde_values, color="#4C72B0", alpha=0.3)

        ax1.set_ylabel("")
        ax1.set_yticklabels([])
        ax1.tick_params(left=False, labelleft=False, bottom=False)

        for spine in ax1.spines.values():
            spine.set_visible(False)

        ax1.spines["bottom"].set_visible(True)
        ax1.spines["bottom"].set_color("black")
        ax1.spines["bottom"].set_linewidth(2)

        # Set the plot title on ax1 if show_range is True
        if title is None:
            ax1.set_title(
                f"Contribution of {xlabel}", fontsize=title_fontsize, fontweight="bold"
            )
        else:
            ax1.set_title(title, fontsize=title_fontsize, fontweight="bold")

    # Plot the mean line with steps-pre drawstyle
    sns.lineplot(
        data=df,
        x=column_name,
        y="mean",
        color="#3A5894",
        linewidth=2,
        drawstyle="steps-pre",
        ax=main_ax,
    )

    # Fill between the min and max if specified
    if show_std:
        main_ax.fill_between(
            df[column_name],
            df["mean"] - np.abs(df["std"]),
            df["mean"] + np.abs(df["std"]),
            color="gray",
            alpha=0.3,
            label="Standart Deviation",
            step="pre",
        )

    # Set the plot title on main_ax if show_range is False
    if not show_range:
        if title is None:
            main_ax.set_title(
                f"Contribution of {xlabel}", fontsize=title_fontsize, fontweight="bold"
            )
        else:
            main_ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    # Set background color
    main_ax.set_facecolor("whitesmoke")

    # Set x-ticks
    x_ticks = np.linspace(df[column_name].min(), df[column_name].max(), num=xticks_n)
    x_decimal = _find_tick_decimal(x_ticks, xticks_n)
    main_ax.set_xticks(x_ticks)
    main_ax.set_xticklabels(
        ["-∞"] + [f"{tick:.{x_decimal}f}" for tick in x_ticks[1:-1]] + ["+∞"],
        fontsize=ticks_fontsize,
    )

    # Determine y-ticks range
    if show_std:
        y_min = (df["mean"] - np.abs(df["std"])).min()
        y_max = (df["mean"] + np.abs(df["std"])).max()
    else:
        y_min = df["mean"].min()
        y_max = df["mean"].max()

    # Set y-ticks
    y_ticks = np.linspace(y_min, y_max, num=yticks_n)
    y_decimal = _find_tick_decimal(y_ticks, yticks_n)
    main_ax.set_yticks(y_ticks)
    main_ax.set_yticklabels(
        [f"{tick:.{y_decimal}f}" for tick in y_ticks],
        fontsize=ticks_fontsize,
    )

    # Set labels
    main_ax.set_xlabel(xlabel, fontsize=label_fontsizes)
    main_ax.set_ylabel(ylabel, fontsize=label_fontsizes)

    # Grid configuration
    main_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Add legend if showing min and max
    if show_std:
        main_ax.legend()

    main_ax.spines["right"].set_visible(False)
    main_ax.spines["top"].set_visible(False)

    # Tight layout
    if show_range:
        plt.subplots_adjust(hspace=-0.1)

    plt.tight_layout()
    plt.show()


def interaction_plot(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (10.0, 8.0),
    axis_ticks_n: int = 10,
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

    max_val = values.max()
    min_val = values.min()

    if max_val < 0:  # All values are negative
        colormap = plt.get_cmap("Blues")
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    elif min_val > 0:  # All values are positive
        colormap = plt.get_cmap("Reds")
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    else:  # Both negative and positive values
        colormap = plt.get_cmap("coolwarm")
        abs_max = max(abs(min_val), max_val)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

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
    x_decimal = _find_tick_decimal(x_ticks, axis_ticks_n)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        ["-∞"] + [f"{tick:.{x_decimal}f}" for tick in x_ticks[1:-1]] + ["+∞"],
        fontsize=ticks_fontsize,
    )

    # Set y-axis ticks
    y_ticks = np.linspace(y_min, y_max, axis_ticks_n)
    y_decimal = _find_tick_decimal(y_ticks, axis_ticks_n)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        ["-∞"] + [f"{tick:.{y_decimal}f}" for tick in y_ticks[1:-1]] + ["+∞"],
        fontsize=ticks_fontsize,
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, spacing="proportional")

    cbar.ax.set_yscale("linear")
    cbar.ax.set_ylim(min_val, max_val)
    cbar.ax.tick_params(labelsize=ticks_fontsize)
    cbar.set_label(
        "Impact" if color_bar_label is None else color_bar_label,
        fontsize=label_fontsizes,
    )

    ax.set_xlabel(xlabel if xlabel is not None else column1, fontsize=label_fontsizes)
    ax.set_ylabel(ylabel if ylabel is not None else column2, fontsize=label_fontsizes)
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
    if isinstance(X, pd.DataFrame):
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

    max_val = values.max()
    min_val = values.min()

    if max_val < 0:  # All values are negative
        colormap = plt.get_cmap("Blues")
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    elif min_val > 0:  # All values are positive
        colormap = plt.get_cmap("Reds")
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    else:  # Both negative and positive values
        colormap = plt.get_cmap("coolwarm")
        abs_max = max(abs(min_val), max_val)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # Create scatter plot
    scatter = ax.scatter(
        x_values, y_values, c=values, cmap=colormap, norm=norm, edgecolors="black"
    )

    # Set axis labels
    ax.set_xlabel(
        xlabel if xlabel is not None else df.columns[0][:-3],
        fontsize=label_fontsizes,
    )
    ax.set_ylabel(
        ylabel if ylabel is not None else df.columns[2][:-3],
        fontsize=label_fontsizes,
    )

    # Add colorbar with specified number of ticks
    cbar = plt.colorbar(scatter)
    cbar.ax.set_yscale("linear")
    cbar.ax.set_ylim(min_val, max_val)
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


def scatter_3d_plot(
    df,
    width=1000,
    height=800,
    opacity=0.8,
    point_size=5
):
    """
    Enhanced 3D scatter plot with smooth color transitions.
    Red indicates positive values, blue indicates negative values, yellow for zero.

    Parameters:
    - df (pd.DataFrame): Input dataframe with '_lb' and '_ub' columns for three variables and a 'value' column
    - width (int): Width of the plot. Default is 1000
    - height (int): Height of the plot. Default is 800
    - opacity (float): Opacity of the scatter points. Default is 0.8
    - point_size (int): Size of scatter points. Default is 5
    """
    # Column detection and infinity handling
    lb_columns = [col for col in df.columns if "_lb" in col]
    ub_columns = [col for col in df.columns if "_ub" in col]
    value_column = "value" if "value" in df.columns else df.columns[-1]

    # Handle infinity values
    for col in ub_columns:
        max_val = df[col][df[col] != float("inf")].max()
        df[col] = df[col].replace([float("inf")], max_val * 1.1)
    
    for col in lb_columns:
        min_val = df[col][df[col] != -float("inf")].min()
        df[col] = df[col].replace([-float("inf")], min_val * 1.1)

    # Calculate midpoints
    mid_columns = []
    ranges = {}
    for lb, ub in zip(lb_columns, ub_columns):
        mid_col = lb.replace("_lb", "_mid")
        df[mid_col] = (df[lb] + df[ub]) / 2
        mid_columns.append(mid_col)
        ranges[mid_col] = {
            'min': df[mid_col].min(),
            'max': df[mid_col].max()
        }

    # Updated color scale with blue-yellow-red transition
    colorscale = [
        [0.0, 'rgb(0,0,255)'],
        [0.5, 'rgb(255,255,0)'],
        [1.0, 'rgb(255,0,0)']
    ]
    
    # Get value range for consistent color scaling
    value_min = df[value_column].min()
    value_max = df[value_column].max()
    abs_max = max(abs(value_min), abs(value_max))
    cmin, cmax = -abs_max, abs_max

    # Create figure
    fig = go.FigureWidget()

    # Add scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=df[mid_columns[0]],
            y=df[mid_columns[1]],
            z=df[mid_columns[2]],
            mode="markers",
            marker=dict(
                size=point_size,
                color=df[value_column],
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title="Intensity",
                    tickformat=".2f",
                    len=0.8,
                ),
                opacity=opacity,
            ),
            hovertemplate=(
                f"<b>{mid_columns[0]}</b>: %{{x:.2f}}<br>"
                f"<b>{mid_columns[1]}</b>: %{{y:.2f}}<br>"
                f"<b>{mid_columns[2]}</b>: %{{z:.2f}}<br>"
                f"<b>Value</b>: %{{marker.color:.2f}}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=mid_columns[0],
                titlefont=dict(size=15),
                range=[ranges[mid_columns[0]]['min'], ranges[mid_columns[0]]['max']],
            ),
            yaxis=dict(
                title=mid_columns[1],
                titlefont=dict(size=15),
                range=[ranges[mid_columns[1]]['min'], ranges[mid_columns[1]]['max']],
            ),
            zaxis=dict(
                title=mid_columns[2],
                titlefont=dict(size=15),
                range=[ranges[mid_columns[2]]['min'], ranges[mid_columns[2]]['max']],
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5),
            ),
        ),
        width=width,
        height=height,
        title=dict(
            text="Interactive 3D Scatter Plot",
            font=dict(size=20),
            x=0.5,
            y=0.95,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Interactive controls
    min_slider = FloatSlider(
        value=df[value_column].min(),
        min=df[value_column].min(),
        max=df[value_column].max(),
        step=(df[value_column].max() - df[value_column].min()) / 100,
        description="Min Intensity",
        style={'description_width': '100px'},
        layout=Layout(width='400px'),
    )
    
    max_slider = FloatSlider(
        value=df[value_column].max(),
        min=df[value_column].min(),
        max=df[value_column].max(),
        step=(df[value_column].max() - df[value_column].min()) / 100,
        description="Max Intensity",
        style={'description_width': '100px'},
        layout=Layout(width='400px'),
    )

    size_slider = FloatSlider(
        value=point_size,
        min=1,
        max=20,
        step=1,
        description="Point Size",
        style={'description_width': '100px'},
        layout=Layout(width='400px'),
    )

    view_presets = {
        'Default': dict(x=1.5, y=1.5, z=1.5),
        'Top': dict(x=0, y=0, z=2.5),
        'Front': dict(x=0, y=2.5, z=0),
        'Side': dict(x=2.5, y=0, z=0),
    }
    
    view_dropdown = Dropdown(
        options=list(view_presets.keys()),
        value='Default',
        description='View Preset:',
        style={'description_width': '100px'},
        layout=Layout(width='400px'),
    )

    def update_plot(*args):
        min_intensity = min_slider.value
        max_intensity = max_slider.value
        point_size = size_slider.value
        view_preset = view_dropdown.value

        mask = (df[value_column] >= min_intensity) & (df[value_column] <= max_intensity)
        filtered_df = df[mask]

        with fig.batch_update():
            fig.data[0].x = filtered_df[mid_columns[0]]
            fig.data[0].y = filtered_df[mid_columns[1]]
            fig.data[0].z = filtered_df[mid_columns[2]]
            fig.data[0].marker.color = filtered_df[value_column]
            fig.data[0].marker.size = point_size
            fig.layout.scene.camera.eye = view_presets[view_preset]

    # Set up observers
    min_slider.observe(update_plot, names='value')
    max_slider.observe(update_plot, names='value')
    size_slider.observe(update_plot, names='value')
    view_dropdown.observe(update_plot, names='value')

    # Create and display interactive controls
    controls = VBox([
        HBox([min_slider, max_slider]),
        HBox([size_slider, view_dropdown])
    ])
    
    display(controls)
    display(fig)
