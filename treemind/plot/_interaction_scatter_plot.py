from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .. import Result


def _validate_interaction_scatter_plot_parameters(
    X: np.ndarray,
    result: Result,
    cols: Union[Tuple[int, int], List[int]],
    figsize: Tuple[float, float],
    ticks_fontsize: Tuple[int, float],
    title_fontsize: Tuple[int, float],
    label_fontsizes: Tuple[int, float],
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
    color_bar_label: Optional[str],
) -> None:
    # X validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("`X` must be a two-dimensional NumPy array.")

    # Feature indices
    if not all(isinstance(c, int) for c in cols) or len(cols) != 2:
        raise TypeError("`cols` must be a tuple or list of exactly two integers.")
    if any(c < 0 or c >= X.shape[1] for c in cols):
        raise ValueError("`cols` must refer to valid column indices in `X`.")

    # result type check
    if not isinstance(result, Result):
        raise TypeError("`result` must be a Result object.")

    df = result[cols].copy()

    # Detect class column (if exists) and drop it temporarily for validation
    df_check = df.drop(columns=["class"]) if "class" in df.columns else df.copy()

    # Identify logical features: continuous features must come in (_lb, _ub) pairs
    bound_pairs = []
    columns = df_check.columns

    for i in range(len(columns) - 1):
        col1, col2 = columns[i], columns[i + 1]
        if col1.endswith("_lb") and col2 == col1.replace("_lb", "_ub"):
            bound_pairs.append((col1, col2))

    if len(bound_pairs) != 2:
        raise ValueError(
            "Exactly two logical features are required as (_lb, _ub) column pairs."
        )

    # Check for fully infinite bounds in any feature
    for lb_col, ub_col in bound_pairs:
        if np.isinf(df_check[lb_col]).all() or np.isinf(df_check[ub_col]).all():
            raise ValueError(
                "No interaction found between the two features - one of them contains only infinite values."
            )

    # figsize
    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise TypeError("`figsize` must be a tuple of two numeric values.")
    if not all(isinstance(f, (int, float)) for f in figsize):
        raise ValueError("`figsize` values must be numeric.")

    # font sizes
    for val, name in [
        (ticks_fontsize, "ticks_fontsize"),
        (title_fontsize, "title_fontsize"),
        (label_fontsizes, "label_fontsizes"),
    ]:
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(f"`{name}` must be a positive number.")

    # Optional labels
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
    result: Result,
    cols: Union[Tuple[int, int], List[int]],
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
    If 'class' column exists in df, creates separate plots for each class.

    Parameters
    ----------
    X : pd.DataFrame
        Input data containing feature values
    result : Result
        Result object containing interaction statistics.
    cols : tuple of int or list of int
        Indices of the two features to plot.
    figsize : tuple of float, default (10.0, 8.0)
        Width and height of the plot in inches.
    ticks_fontsize : float, default 10.0
        Font size for axis tick labels,
    title_fontsize : float, default 16.
        Font size for plot title, by
    title : str, optional, default None
        The title displayed at the top of the plot. If `None`, no title is shown.
    xlabel : str, optional, default None
        Label for the x-axis. If None, it will default to the feature name.
    ylabel : str, optional, default None
        Label for the y-axis. If None, it will default to the feature name.
    color_bar_label : str, optional, default None
        Colorbar label, If None, it will default to "Impact".
    """
    # ---------------------------------------------------------------------
    # Data wrangling
    # ---------------------------------------------------------------------
    try:
        X = np.asarray(X)
    except Exception as exc:
        raise ValueError(
            "Cannot convert X to a NumPy array – check its type/shape."
        ) from exc

    _validate_interaction_scatter_plot_parameters(
        X=X,
        result=result,
        cols=cols,
        figsize=figsize,
        ticks_fontsize=ticks_fontsize,
        title_fontsize=title_fontsize,
        label_fontsizes=label_fontsizes,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color_bar_label=color_bar_label,
    )

    x_vals = X[:, cols[0]].astype(float)
    y_vals = X[:, cols[1]].astype(float)

    df = result[cols].copy()

    # Check if 'class' column exists
    has_class_column = "class" in df.columns

    if has_class_column:
        # Get unique classes
        unique_classes = df["class"].unique()

        # Create separate plot for each class
        for class_value in unique_classes:
            # Filter dataframe for current class
            df_class = df[df["class"] == class_value].copy()

            # Skip if no data for this class
            if df_class.empty:
                continue

            # Build lookup grid for this class
            x_edges, y_edges, grid = _build_lookup(df_class)
            values = _lookup_values(x_vals, y_vals, x_edges, y_edges, grid)

            # Create plot for this class
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
            if title:
                plot_title = f"{title} - Class {class_value}"
            else:
                plot_title = f"Contribution of {ax.get_xlabel()} and {ax.get_ylabel()} - Class {class_value}"

            ax.set_title(plot_title, fontsize=title_fontsize)

            plt.tight_layout()
            plt.show()

    else:
        # Original behavior when no class column exists
        x_edges, y_edges, grid = _build_lookup(df)
        values = _lookup_values(x_vals, y_vals, x_edges, y_edges, grid)

        # Create single plot
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
            title
            if title is not None
            else f"Contribution of {ax.get_xlabel()} and {ax.get_ylabel()}",
            fontsize=title_fontsize,
        )

        plt.tight_layout()
        plt.show()
