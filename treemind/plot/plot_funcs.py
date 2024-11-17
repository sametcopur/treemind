import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from scipy.stats import gaussian_kde

from .plot_utils import (
    _replace_infinity,
    _validate_feature_plot_parameters,
    _validate_interaction_plot_parameters,
    _validate_interaction_scatter_plot_parameters,
    _find_tick_decimal,
)

from typing import Tuple, Optional

def feature_plot(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (12.0, 8.0),
    show_std: bool = False,
    show_range: bool = True,
    xticks_n: int = 10,
    yticks_n: int = 10,
    ticks_fontsize: float = 10.0,
    title_fontsize: float = 16.0,
    label_fontsizes: float = 14.0,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Visualizes feature statistics across ranges defined by tree split points. The plot includes
    the difference from the mean, optional standard deviation, and the distribution of feature
    counts for each tree split point range.
    
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
    figsize : tuple of float, default (12.0, 8.0)
        Width and height of the plot in inches.
    show_std : bool, default False
        If True, shaded areas representing the standard deviation values will be displayed.
    show_range : bool, default True
        If True, show leaf distribution within range.
    xticks_n : int, default 10
        Number of tick marks to display on the x-axis.
    yticks_n : int, default 10
        Number of tick marks to display on the y-axis.
    ticks_fontsize : float, default 10.0
        Font size for axis tick labels,
    title_fontsize : float, default 16.0
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
        y="value",
        color="#3A5894",
        linewidth=2,
        drawstyle="steps-pre",
        ax=main_ax,
    )

    # Fill between the min and max if specified
    if show_std:
        main_ax.fill_between(
            df[column_name],
            df["value"] - np.abs(df["std"]),
            df["value"] + np.abs(df["std"]),
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
        y_min = (df["value"] - np.abs(df["std"])).min()
        y_max = (df["value"] + np.abs(df["std"])).max()
    else:
        y_min = df["value"].min()
        y_max = df["value"].max()

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
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color_bar_label: Optional[str] = None,
) -> None:
    """
    Creates a heatmap-style plot to visualize feature interactions. Each rectangle represents 
    an interaction region, with its color indicating the interaction value.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing interaction data with columns `_lb`, `_ub`, `_lb`, `_ub`, and `value`.
        The first four columns represent intervals for two features, where each pair (_lb, _ub) defines
        the bounds of one feature. The last column, `value`, contains the interaction values for each pair.
    figsize : tuple of float, default (10.0, 6.0)
        Width and height of the plot in inches.
    axis_ticks_n : int, default 10
        Number of ticks on both axis
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
    try:
        X = np.asarray(X)
    except:
        raise ValueError(
            f"Failed to convert input data to a NumPy array. Ensure X is a compatible format."
        )

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

    # Extract values from X
    x_values = X[:, col_1]
    y_values = X[:, col_2]

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
