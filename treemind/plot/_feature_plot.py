import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from typing import Tuple, Union, Optional
from .plot_utils import (
    _replace_infinity,
    _find_tick_decimal,
)
from ..algorithm import Result


def _validate_feature_plot_parameters(
    result: Result,
    index: int,
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
    Now supports both continuous (range-based) and categorical data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to validate.
    figsize : tuple of int
        The figure size as a tuple of two positive integers.
    show_std : bool
        Whether to show the std range.
    show_range : bool
        Whether to show the range distribution.
    xticks_n : int
        Number of ticks on the x-axis.
    yticks_n : int
        Number of ticks on the y-axis.
    ticks_fontsize : int
        Font size for the tick labels.
    title_fontsize : int
        Font size for the title.
    label_fontsizes : int
        Font size for the axis labels.
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
    if not isinstance(result, Result):
        raise TypeError("result must be an instance of Result.")
    
    if not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer.")

    df = result[index].copy()

    # Validate basic required columns
    required_columns = {"value", "std", "count"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"The DataFrame must contain the following columns: {required_columns}"
        )

    # Determine data type: continuous (range-based) or categorical
    lb_columns = [col for col in df.columns if col.endswith("_lb")]
    ub_columns = [col for col in df.columns if col.endswith("_ub")]

    # Check if it's continuous data (has _lb and _ub columns)
    is_continuous = len(lb_columns) > 0 and len(ub_columns) > 0

    if is_continuous:
        # Validate continuous data structure
        if len(lb_columns) != 1 or len(ub_columns) != 1:
            raise ValueError(
                "For continuous data, the DataFrame must contain exactly one '_lb' column and one '_ub' column."
            )
    else:
        # For categorical data, we expect the first column to be the category identifier
        # (but excluding 'class' column if it exists)
        non_class_cols = [col for col in df.columns if col != "class"]
        if len(non_class_cols) < 4:  # category, value, std, count (minimum)
            raise ValueError(
                "For categorical data, the DataFrame must have at least 4 columns: category identifier, value, std, count."
            )

    # Check figsize
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError(
            "The 'figsize' parameter must be a tuple of two numeric values."
        )
    if not all(isinstance(dim, (int, float)) for dim in figsize):
        raise ValueError("Both dimensions in 'figsize' must be numeric.")

    # Validate boolean parameters
    if not isinstance(show_std, bool):
        raise ValueError("show_std must be a boolean value.")

    if not isinstance(show_range, bool):
        raise ValueError("show_range must be a boolean value.")

    # Validate xticks_n and yticks_n
    if not (isinstance(xticks_n, int) and xticks_n > 0):
        raise ValueError("xticks_n must be a positive integer.")
    if not (isinstance(yticks_n, int) and yticks_n > 0):
        raise ValueError("yticks_n must be a positive integer.")

    # Validate font sizes
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


def _is_categorical_data(df: pd.DataFrame) -> bool:
    """
    Determines if the DataFrame contains categorical or continuous data.

    Returns True if categorical, False if continuous (range-based).
    """
    lb_columns = [col for col in df.columns if col.endswith("_lb")]
    ub_columns = [col for col in df.columns if col.endswith("_ub")]

    return len(lb_columns) == 0 or len(ub_columns) == 0


def _has_multiclass_data(df: pd.DataFrame) -> bool:
    """
    Determines if the DataFrame contains multiclass data (has 'class' column).

    Returns True if multiclass, False otherwise.
    """
    return "class" in df.columns


def feature_plot(
    result: Result,
    index: int,
    figsize: Tuple[float, float] = (12.0, 8.0),
    show_std: bool = False,
    show_range: bool = False,
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
    Plot a single feature's contribution profile, handling **either**

    * a **continuous** feature stored as split-point intervals
      (``<feature>_lb`` / ``<feature>_ub``), **or**
    * a **categorical** feature stored as one identifier column.

    The plot shows the mean *impact* (``value``) and can optionally display
    one or both of

    * ± ``std`` shaded bands, and
    * a count / frequency distribution (*kde* for continuous, line plot for
      categorical).

    **New**: Supports multiclass problems. If a 'class' column is present,
    separate plots will be generated for each class.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary statistics for a *single* feature.  Two layout variants
        are supported.

        **Continuous layout**

        ========  ========================================================
        Column    Meaning
        --------  --------------------------------------------------------
        `<name>_lb`  Lower bound of the interval
        `<name>_ub`  Upper bound of the interval
        `value`      Mean impact inside the interval
        `std`        Standard deviation of the impact
        `count`      Row count in the interval
        `class`      (Optional) Class identifier for multiclass problems
        ========  ========================================================

        **Categorical layout**

        ========  ========================================================
        Column    Meaning
        --------  --------------------------------------------------------
        `<name>`    Category identifier (first non-class column in *df*)
        `value`     Mean impact for the category
        `std`       Standard deviation of the impact
        `count`     Row count for the category
        `class`     (Optional) Class identifier for multiclass problems
        ========  ========================================================

    figsize : (float, float), default ``(12, 8)``
        Width × height of the figure (in inches).

    show_std : bool, default ``False``
        If *True*, plot ± ``std`` shaded regions.

    show_range : bool, default ``True``
        If *True*, draw the count / frequency distribution:
        a KDE curve for continuous data or a line-and-fill plot for
        categorical data.

    xticks_n, yticks_n : int, default ``10``
        Desired number of tick marks on the *x*- and *y*-axes respectively.

    ticks_fontsize : float, default ``10.0``
        Font size for tick labels.

    title_fontsize : float, default ``16.0``
        Font size for the plot title (if drawn).

    label_fontsizes : float, default ``14.0``
        Font size for *x*- and *y*-axis labels and legend text.

    title : str or None, default ``None``
        Custom figure title.  If *None*, a title of the form
        ``"Contribution of <feature> - Class X"`` is generated automatically.

    xlabel, ylabel : str or None, default ``None``
        Custom axis labels.  If *None*, they default to the feature name
        (for *x*) and ``"Value"`` (for *y*).

    Returns
    -------
    None
        The function draws the figure(s) and shows them; nothing is returned.

    Notes
    -----
    * **Continuous data** is rendered with a *step* line (tree-split style).
      Interval endpoints of ± ∞ are compressed to the outermost finite split
      so the geometry remains well defined.
    * **Categorical data** is rendered with vertical bars originating from
      zero; categories appear in the order they occur in *df*.
    * **Multiclass data** generates separate plots for each class, with class
      information appended to the plot title.
    """

    _validate_feature_plot_parameters(
        result=result,
        index=index,
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
    
    df = result[index].copy()

    # Check if this is multiclass data
    is_multiclass = _has_multiclass_data(df)

    if is_multiclass:
        # Get unique classes and plot for each
        unique_classes = sorted(df["class"].unique())

        for class_idx in unique_classes:
            # Filter data for current class
            class_df = df[df["class"] == class_idx].copy()
            # Remove the class column for plotting
            class_df = class_df.drop("class", axis=1)

            # Create class-specific title
            if title is None:
                # Determine feature name for auto title
                is_categorical = _is_categorical_data(class_df)
                if is_categorical:
                    feature_name = class_df.columns[0].replace("_", " ").title()
                else:
                    ub_columns = [
                        col for col in class_df.columns if col.endswith("_ub")
                    ]
                    feature_name = ub_columns[0][:-3] if ub_columns else "Feature"

                class_title = f"Contribution of {feature_name} - Class {class_idx}"
            else:
                class_title = f"{title} - Class {class_idx}"

            # Plot for this class
            _plot_single_class(
                class_df,
                figsize,
                show_std,
                show_range,
                xticks_n,
                yticks_n,
                ticks_fontsize,
                title_fontsize,
                label_fontsizes,
                class_title,
                xlabel,
                ylabel,
            )
    else:
        # Single class data - use existing logic
        _plot_single_class(
            df,
            figsize,
            show_std,
            show_range,
            xticks_n,
            yticks_n,
            ticks_fontsize,
            title_fontsize,
            label_fontsizes,
            title,
            xlabel,
            ylabel,
        )


def _plot_single_class(
    df: pd.DataFrame,
    figsize: Tuple[float, float],
    show_std: bool,
    show_range: bool,
    xticks_n: int,
    yticks_n: int,
    ticks_fontsize: float,
    title_fontsize: float,
    label_fontsizes: float,
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
) -> None:
    """
    Plot a single class's feature data.
    """
    # Determine if data is categorical or continuous
    is_categorical = _is_categorical_data(df)

    if is_categorical:
        _plot_categorical_feature(
            df,
            figsize,
            show_std,
            show_range,
            yticks_n,
            ticks_fontsize,
            title_fontsize,
            label_fontsizes,
            title,
            xlabel,
            ylabel,
        )
    else:
        _plot_continuous_feature(
            df,
            figsize,
            show_std,
            show_range,
            xticks_n,
            yticks_n,
            ticks_fontsize,
            title_fontsize,
            label_fontsizes,
            title,
            xlabel,
            ylabel,
        )


def _plot_categorical_feature(
    df: pd.DataFrame,
    figsize: Tuple[float, float],
    show_std: bool,
    show_range: bool,
    yticks_n: int,
    ticks_fontsize: float,
    title_fontsize: float,
    label_fontsizes: float,
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
) -> None:
    """
    Plot categorical feature as vertical bars from y=0, with optional std and count distribution.
    """
    category_col = df.columns[0]
    xlabel = xlabel or category_col.replace("_", " ").title()
    ylabel = ylabel or "Value"

    categories = df[category_col].astype(str).values
    values = df["value"].values
    stds = df["std"].values
    counts = df["count"].values
    x = np.arange(len(categories))

    # Setup figure
    if show_range:
        fig, (ax_dist, ax_main) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 4]}, sharex=True
        )
        plt.subplots_adjust(hspace=-0.1)
    else:
        fig, ax_main = plt.subplots(figsize=figsize)

    # Colors - aynı continuous ile
    bar_color = "#3A5894"
    std_color = "gray"
    dist_color = "#4C72B0"

    # === 1. Üstteki Distribution - Line Plot ===
    if show_range:
        # Simple line plot with markers
        ax_dist.plot(
            x,
            counts,
            color=dist_color,
            linewidth=2,
            marker="o",
            markersize=6,
            alpha=0.8,
            zorder=3,
        )

        # Fill area under the line
        ax_dist.fill_between(x, counts, color=dist_color, alpha=0.3, zorder=2)

        # Stil - continuous ile aynı
        ax_dist.set_ylabel("")
        ax_dist.set_yticklabels([])
        ax_dist.tick_params(left=False, labelleft=False, bottom=False)

        for spine in ax_dist.spines.values():
            spine.set_visible(False)
        ax_dist.spines["bottom"].set_visible(True)
        ax_dist.spines["bottom"].set_color("black")
        ax_dist.spines["bottom"].set_linewidth(2)

        # Başlık - continuous ile aynı format
        if title is None:
            ax_dist.set_title(
                f"Contribution of {xlabel}",
                fontsize=title_fontsize,
                fontweight="bold",
            )
        else:
            ax_dist.set_title(title, fontsize=title_fontsize, fontweight="bold")

    # === 2. Ana Değer Grafiği ===
    ax_main.bar(x, values, color=bar_color, edgecolor="white", width=0.7, zorder=3)

    # Error bars
    if show_std:
        ax_main.fill_between(
            x,
            values - stds,
            values + stds,
            color=std_color,
            alpha=0.3,
            label="Standard Deviation",
            zorder=2,
        )

    # 0 referans çizgisi
    ax_main.axhline(y=0, color="black", linestyle="--", linewidth=1, zorder=1)

    # Başlık - continuous ile aynı format (sadece show_range False ise)
    if not show_range:
        if title is None:
            ax_main.set_title(
                f"Contribution of {xlabel}", fontsize=title_fontsize, fontweight="bold"
            )
        else:
            ax_main.set_title(title, fontsize=title_fontsize, fontweight="bold")

    # Stil - continuous ile aynı
    ax_main.set_facecolor("whitesmoke")
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(categories, fontsize=ticks_fontsize)
    ax_main.set_xlabel(xlabel, fontsize=label_fontsizes)
    ax_main.set_ylabel(ylabel, fontsize=label_fontsizes)

    # Y-ticks
    if show_std:
        y_min = min(values - stds)
        y_max = max(values + stds)
    else:
        y_min = min(values)
        y_max = max(values)

    y_ticks = np.linspace(y_min, y_max, num=yticks_n)
    y_decimal = _find_tick_decimal(y_ticks, yticks_n)

    ax_main.set_yticks(y_ticks)
    ax_main.set_yticklabels(
        [f"{y:.{y_decimal}f}" for y in y_ticks], fontsize=ticks_fontsize
    )

    # Grid & spines - continuous ile aynı
    ax_main.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["top"].set_visible(False)

    # Legend
    if show_std:
        ax_main.legend()

    plt.tight_layout()
    plt.show()


def _plot_continuous_feature(
    df: pd.DataFrame,
    figsize: Tuple[float, float],
    show_std: bool,
    show_range: bool,
    xticks_n: int,
    yticks_n: int,
    ticks_fontsize: float,
    title_fontsize: float,
    label_fontsizes: float,
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
) -> None:
    """Plot continuous feature data (original implementation)."""

    # Find the _ub column (assuming it's the main feature column)
    ub_columns = [col for col in df.columns if col.endswith("_ub")]
    lb_columns = [col for col in df.columns if col.endswith("_lb")]

    column_name = ub_columns[0]
    lb_column_name = lb_columns[0]

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
            label="Standard Deviation",
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

    # Add legend if showing std
    if show_std:
        main_ax.legend()

    main_ax.spines["right"].set_visible(False)
    main_ax.spines["top"].set_visible(False)

    # Tight layout
    if show_range:
        plt.subplots_adjust(hspace=-0.1)

    plt.tight_layout()
    plt.show()
