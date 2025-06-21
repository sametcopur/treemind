import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
from typing import Tuple, Union, Optional

from treemind.plot.plot_utils import _replace_infinity, _find_tick_decimal


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
    Validate inputs for :func:`interaction_plot`.

    A *logical feature* is either

    * a **continuous** pair of columns ``<feature>_lb`` & ``<feature>_ub``, or
    * a single **categorical** column ``<feature>``.

    Exactly two logical features plus the numeric columns ``value``, ``std`` and
    ``count`` are required.

    Parameters
    ----------
    df : pandas.DataFrame
        Interaction summary table.
    figsize : (float, float)
        Figure size in inches.
    axis_ticks_n : int
        Tick count for each continuous axis.
    ticks_fontsize, title_fontsize, label_fontsizes : int or float
        Font sizes for ticks, title and labels respectively.
    title, xlabel, ylabel, color_bar_label : str or None
        Optional labelling strings.

    Raises
    ------
    TypeError
        For wrongly-typed arguments.
    ValueError
        If *df* is missing required columns or does not contain exactly two
        logical features, or has fewer than three rows.
    """
    # basic DataFrame sanity checks ------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if df.shape[0] <= 2:
        raise ValueError("There is no interaction between features to plot.")

    required_meta = {"value", "std", "count"}
    if not required_meta.issubset(df.columns):
        missing = required_meta - set(df.columns)
        raise ValueError(f"`df` is missing required columns: {missing}")

    feature_cols = [c for c in df.columns if c not in required_meta]

    # discover logical features ----------------------------------------------------
    cont_pairs, cat_cols, visited = {}, [], set()
    for col in feature_cols:
        if col in visited:
            continue
        if col.endswith("_lb") and f"{col[:-3]}_ub" in df.columns:
            cont_pairs[col[:-3]] = (col, f"{col[:-3]}_ub")
            visited.update(cont_pairs[col[:-3]])
        elif col.endswith("_ub"):
            continue  # handled by its _lb partner
        else:
            cat_cols.append(col)
            visited.add(col)

    if len(cont_pairs) + len(cat_cols) != 2:
        raise ValueError(
            "Exactly two logical features are required. "
            f"Found {len(cont_pairs)} continuous + {len(cat_cols)} categorical."
        )

    # figure & typography parameters ----------------------------------------------
    if not (
        isinstance(figsize, tuple)
        and len(figsize) == 2
        and all(isinstance(dim, (int, float)) for dim in figsize)
    ):
        raise TypeError("`figsize` must be a tuple of two numeric values.")

    if not isinstance(axis_ticks_n, int) or axis_ticks_n <= 0:
        raise ValueError("`axis_ticks_n` must be a positive integer.")

    for val, name in [
        (ticks_fontsize, "ticks_fontsize"),
        (title_fontsize, "title_fontsize"),
        (label_fontsizes, "label_fontsizes"),
    ]:
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(f"`{name}` must be a positive number.")

    for txt, name in [
        (title, "title"),
        (xlabel, "xlabel"),
        (ylabel, "ylabel"),
        (color_bar_label, "color_bar_label"),
    ]:
        if txt is not None and not isinstance(txt, str):
            raise TypeError(f"`{name}` must be a string if provided.")


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
    Plot a heat-map of interaction strengths for **two** features that may be
    continuous, categorical, or one of each.

    A continuous feature must appear as column pair ``<name>_lb`` +
    ``<name>_ub`` (interval bounds); a categorical feature appears as a single
    column ``<name>``.  The DataFrame must also include ``value`` (mean impact),
    ``std`` (standard deviation) and ``count`` (bin size).

    Parameters
    ----------
    df : pandas.DataFrame
        Interaction summary table (see above).
    figsize : (float, float), default (10, 8)
        Width × height of the figure.
    axis_ticks_n : int, default 10
        Number of ticks to show on a continuous axis.
    ticks_fontsize : float, default 10
        Font size for tick labels.
    title_fontsize : float, default 16
        Font size for the title.
    label_fontsizes : float, default 14
        Font size for axis- and colour-bar labels.
    title : str or None, default None
        Optional figure title.
    xlabel, ylabel : str or None, default None
        Optional axis labels (fall back to feature names).
    color_bar_label : str or None, default None
        Label for the colour-bar (“Impact” if *None*).

    Returns
    -------
    None
        Displays the figure.
    """
    # validate all inputs ----------------------------------------------------------
    _validate_interaction_plot_parameters(
        df,
        figsize,
        axis_ticks_n,
        ticks_fontsize,
        title_fontsize,
        label_fontsizes,
        title,
        xlabel,
        ylabel,
        color_bar_label,
    )

    # ---------------- 1. identify features ---------------------------------------
    meta_cols = {"value", "std", "count"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    cont_pairs, cat_cols, used = {}, [], set()
    for col in feature_cols:
        if col in used:
            continue
        if col.endswith("_lb") and (ub := f"{col[:-3]}_ub") in df.columns:
            cont_pairs[col[:-3]] = (col, ub)
            used.update({col, ub})
        elif col.endswith("_ub"):
            continue
        else:
            cat_cols.append(col)
            used.add(col)

    ordered: list[tuple[str, str]] = []
    for col in feature_cols:
        if col in used:
            if col.endswith("_lb"):
                name = col[:-3]
                if name not in [f[0] for f in ordered]:
                    ordered.append((name, "continuous"))
            elif not col.endswith("_ub"):
                ordered.append((col, "categorical"))

    (feat1, type1), (feat2, type2) = ordered  # guaranteed by validator

    # ---------------- 2. rectangle geometry --------------------------------------
    if type1 == "continuous":
        lb1, ub1 = cont_pairs[feat1]
        df = _replace_infinity(df, lb1, "negative")
        df = _replace_infinity(df, ub1, "positive")
        x_left, x_right = df[lb1].values, df[ub1].values
    else:
        cats1 = list(dict.fromkeys(df[feat1].astype(str)))
        cat_to_x = {c: i for i, c in enumerate(cats1)}
        x_left = np.array([cat_to_x[v] for v in df[feat1].astype(str)])
        x_right = x_left + 1

    if type2 == "continuous":
        lb2, ub2 = cont_pairs[feat2]
        df = _replace_infinity(df, lb2, "negative")
        df = _replace_infinity(df, ub2, "positive")
        y_bottom, y_top = df[lb2].values, df[ub2].values
    else:
        cats2 = list(dict.fromkeys(df[feat2].astype(str)))
        cat_to_y = {c: i for i, c in enumerate(cats2)}
        y_bottom = np.array([cat_to_y[v] for v in df[feat2].astype(str)])
        y_top = y_bottom + 1

    # ---------------- 3. rectangles & colours ------------------------------------
    rects = [
        Rectangle((lx, ly), w, h)
        for lx, ly, w, h in zip(x_left, y_bottom, x_right - x_left, y_top - y_bottom)
    ]

    values = df["value"].values
    vmax, vmin = values.max(), values.min()
    abs_max = max(abs(vmax), abs(vmin))
    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    colors = cmap(norm(values))

    # ---------------- 4. plotting -------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(PatchCollection(rects, facecolor=colors, edgecolor="none"))

    ax.set_xlim(x_left.min(), x_right.max())
    ax.set_ylim(y_bottom.min(), y_top.max())

    # x-axis ticks ---------------------------------------------------------------
    if type1 == "continuous":
        ticks = np.linspace(x_left.min(), x_right.max(), axis_ticks_n)
        dec = _find_tick_decimal(ticks, axis_ticks_n)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            ["-∞"] + [f"{t:.{dec}f}" for t in ticks[1:-1]] + ["+∞"],
            fontsize=ticks_fontsize,
        )
    else:
        ax.set_xticks(np.arange(len(cats1)) + 0.5)
        ax.set_xticklabels(cats1, rotation=45, ha="right", fontsize=ticks_fontsize)

    # y-axis ticks ---------------------------------------------------------------
    if type2 == "continuous":
        ticks = np.linspace(y_bottom.min(), y_top.max(), axis_ticks_n)
        dec = _find_tick_decimal(ticks, axis_ticks_n)
        ax.set_yticks(ticks)
        ax.set_yticklabels(
            ["-∞"] + [f"{t:.{dec}f}" for t in ticks[1:-1]] + ["+∞"],
            fontsize=ticks_fontsize,
        )
    else:
        ax.set_yticks(np.arange(len(cats2)) + 0.5)
        ax.set_yticklabels(cats2, fontsize=ticks_fontsize)

    # colour-bar ------------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, spacing="proportional")
    cbar.ax.set_ylim(vmin, vmax)
    cbar.ax.tick_params(labelsize=ticks_fontsize)
    cbar.set_label(color_bar_label or "Impact", fontsize=label_fontsizes)

    # labels & title --------------------------------------------------------------
    ax.set_xlabel(xlabel or feat1, fontsize=label_fontsizes)
    ax.set_ylabel(ylabel or feat2, fontsize=label_fontsizes)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()
