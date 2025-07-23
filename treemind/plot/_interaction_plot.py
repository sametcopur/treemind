import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
from typing import Tuple, Union, Optional, List

from .plot_utils import _replace_infinity, _find_tick_decimal
from .. import Result

def _validate_interaction_plot_parameters(
    result: Result,
    cols: Union[Tuple[int, int], List[int]],
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
    ``count`` are required. If a ``class`` column is present, separate plots
    will be created for each class.

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
    # result type check
    if not isinstance(result, Result):
        raise TypeError("`result` must be a Result object.")

    df = result[cols].copy()
    
    if df.shape[0] <= 2:
        raise ValueError("There is no interaction between features to plot.")

    required_meta = {"value", "std", "count"}
    if not required_meta.issubset(df.columns):
        missing = required_meta - set(df.columns)
        raise ValueError(f"`df` is missing required columns: {missing}")

    # Check if class column exists (optional)
    has_class_column = "class" in df.columns
    if has_class_column:
        required_meta.add("class")

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
        
    for feature, (lb_col, ub_col) in cont_pairs.items():
        if np.isinf(df[lb_col]).all() or np.isinf(df[ub_col]).all():
            raise ValueError(
                "No interaction found between the two features, one of them contains only infinite values."
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


def _plot_single_class(
    df_class: pd.DataFrame,
    class_name: str,
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
    Plot interaction heat-map for a single class.

    Parameters
    ----------
    df_class : pandas.DataFrame
        Interaction data for a single class.
    class_name : str
        Name of the class for title.
    Other parameters : same as interaction_plot
    """

    # ---------------- 1. identify features ---------------------------------------
    meta_cols = {"value", "std", "count"}
    feature_cols = [c for c in df_class.columns if c not in meta_cols]

    df_class = df_class.copy()

    cont_pairs, used = {}, set()
    cat_cols: List[str] = []
    for col in feature_cols:
        if col in used:
            continue
        if col.endswith("_lb") and (ub := f"{col[:-3]}_ub") in df_class.columns:
            cont_pairs[col[:-3]] = (col, ub)
            used.update({col, ub})
        elif col.endswith("_ub"):
            continue
        else:
            cat_cols.append(col)
            used.add(col)

    ordered: List[Tuple[str, str]] = []
    for col in feature_cols:
        if col not in used:
            continue
        if col.endswith("_lb") and (name := col[:-3]) not in [f[0] for f in ordered]:
            ordered.append((name, "continuous"))
        elif not col.endswith("_ub"):
            ordered.append((col, "categorical"))

    if len(ordered) != 2:
        raise ValueError("Exactly two logical features are required.")

    (feat1, type1), (feat2, type2) = ordered

    # --------------------------- 2. Kenar Dizileri/Kodlama
    if type1 == "continuous":
        lb1, ub1 = cont_pairs[feat1]

        # Infinity değerlerini değiştir
        df_class = _replace_infinity(df_class, lb1, "negative")
        df_class = _replace_infinity(df_class, ub1, "positive")

        x_left = df_class[lb1].to_numpy()
        x_right = df_class[ub1].to_numpy()

    else:
        cats1 = df_class[feat1].astype("category")
        codes1 = cats1.cat.codes.to_numpy()
        x_left, x_right = codes1, codes1 + 1
        cats1_labels = cats1.cat.categories.tolist()

    if type2 == "continuous":
        lb2, ub2 = cont_pairs[feat2]

        # Infinity değerlerini değiştir
        df_class = _replace_infinity(df_class, lb2, "negative")
        df_class = _replace_infinity(df_class, ub2, "positive")

        y_bottom = df_class[lb2].to_numpy()
        y_top = df_class[ub2].to_numpy()

    else:
        cats2 = df_class[feat2].astype("category")
        codes2 = cats2.cat.codes.to_numpy()
        y_bottom, y_top = codes2, codes2 + 1
        cats2_labels = cats2.cat.categories.tolist()

    # --------------------------- 3. Renkler
    values = df_class["value"].to_numpy()
    vmax, vmin = values.max(), values.min()
    abs_max = max(abs(vmax), abs(vmin))
    if abs_max == 0:
        abs_max = 1e-8

    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    # --------------------------- 4. Çizim
    fig, ax = plt.subplots(figsize=figsize)

    if type1 == type2 == "continuous":
        # Continuous-continuous case için pcolormesh kullan
        x_edges = np.sort(np.unique(np.concatenate([x_left, x_right])))
        y_edges = np.sort(np.unique(np.concatenate([y_bottom, y_top])))

        mat = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)
        ix = np.searchsorted(x_edges, x_left, side="left")
        iy = np.searchsorted(y_edges, y_bottom, side="left")

        # Index bounds kontrolü
        ix = np.clip(ix, 0, len(x_edges) - 2)
        iy = np.clip(iy, 0, len(y_edges) - 2)

        for i, j, v in zip(iy, ix, values):
            if 0 <= i < mat.shape[0] and 0 <= j < mat.shape[1]:
                mat[i, j] = v if np.isnan(mat[i, j]) else (mat[i, j] + v) / 2.0

        mesh = ax.pcolormesh(
            x_edges, y_edges, mat, cmap=cmap, norm=norm, shading="auto"
        )
        cbar = fig.colorbar(mesh, ax=ax, spacing="proportional")
    else:
        # Mixed case için rectangles kullan
        rects = [
            Rectangle((lx, ly), w, h)
            for lx, ly, w, h in zip(
                x_left, y_bottom, x_right - x_left, y_top - y_bottom
            )
        ]
        pc = PatchCollection(rects, cmap=cmap, norm=norm)
        pc.set_array(values)
        ax.add_collection(pc)
        cbar = fig.colorbar(pc, ax=ax, spacing="proportional")
        ax.set_xlim(x_left.min(), x_right.max())
        ax.set_ylim(y_bottom.min(), y_top.max())

    # --------------------------- 5. Eksen Tick'leri (Orijinal kodla uyumlu)
    if type1 == "continuous":
        ticks = np.linspace(x_left.min(), x_right.max(), axis_ticks_n)
        dec = _find_tick_decimal(ticks, axis_ticks_n)
        ax.set_xticks(ticks)
        # Orijinal kodla aynı tick labeling mantığı
        ax.set_xticklabels(
            ["-∞"] + [f"{t:.{dec}f}" for t in ticks[1:-1]] + ["+∞"],
            fontsize=ticks_fontsize,
        )
    else:
        ax.set_xticks(np.arange(len(cats1_labels)) + 0.5)
        ax.set_xticklabels(
            cats1_labels, rotation=45, ha="right", fontsize=ticks_fontsize
        )

    if type2 == "continuous":
        ticks = np.linspace(y_bottom.min(), y_top.max(), axis_ticks_n)
        dec = _find_tick_decimal(ticks, axis_ticks_n)
        ax.set_yticks(ticks)
        # Orijinal kodla aynı tick labeling mantığı
        ax.set_yticklabels(
            ["-∞"] + [f"{t:.{dec}f}" for t in ticks[1:-1]] + ["+∞"],
            fontsize=ticks_fontsize,
        )
    else:
        ax.set_yticks(np.arange(len(cats2_labels)) + 0.5)
        ax.set_yticklabels(cats2_labels, fontsize=ticks_fontsize)

    # --------------------------- 6. Başlık & Etiketler
    ax.set_xlabel(xlabel or feat1, fontsize=label_fontsizes)
    ax.set_ylabel(ylabel or feat2, fontsize=label_fontsizes)

    # Title'a class bilgisini ekle
    if title:
        full_title = f"{title} - {'Class ' if class_name else ''}{class_name}"
    else:
        full_title = f"Contribution of {ax.get_xlabel()} and {ax.get_ylabel()}{' - Class ' if class_name else ''}{class_name}"

    ax.set_title(full_title, fontsize=title_fontsize)

    # Color bar ayarları (orijinal kodla uyumlu)
    cbar.ax.set_ylim(vmin, vmax)
    cbar.ax.tick_params(labelsize=ticks_fontsize)
    cbar.set_label(color_bar_label or "Impact", fontsize=label_fontsizes)

    plt.tight_layout()
    plt.show()


def interaction_plot(
    result: Result,
    cols: Union[Tuple[int, int], List[int]],
    figsize: Tuple[float, float] = (10.0, 8.0),
    axis_ticks_n: int = 10,
    ticks_fontsize: Union[int, float] = 10,
    title_fontsize: Union[int, float] = 16,
    label_fontsizes: Union[int, float] = 14,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color_bar_label: Optional[str] = None,
) -> None:
    """
    Plot a heat-map of interaction strengths for **two** features that may be
    continuous, categorical, or one of each. If a class column is provided,
    creates separate plots for each class.

    A continuous feature must appear as column pair ``<name>_lb`` +
    ``<name>_ub`` (interval bounds); a categorical feature appears as a single
    column ``<name>``.  The DataFrame must also include ``value`` (mean impact),
    ``std`` (standard deviation) and ``count`` (bin size).

    Parameters
    ----------
    result : Result
        Result object containing interaction statistics.
    cols : tuple of int or list of int
        Indices of the two features to plot.
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
        Label for the colour-bar ("Impact" if *None*).
    class_column : str or None, default None
        Name of the class column. If provided, creates separate plots for each class.

    Returns
    -------
    None
        Displays the figure(s).
    """

    # validate all inputs ----------------------------------------------------------
    _validate_interaction_plot_parameters(
        result,
        cols,
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
    
    df = result[cols]

    # Check if class column exists for multi-class plotting
    if "class" in df.columns:
        # Get unique classes
        unique_classes = df["class"].unique()

        # Create a plot for each class
        for class_name in unique_classes:
            df_class = df[df["class"] == class_name].copy()

            # Remove class column from the class-specific dataframe
            df_class = df_class.drop(columns=["class"])

            # Plot for this class
            _plot_single_class(
                df_class=df_class,
                class_name=str(class_name),
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
    else:
        # Original single plot behavior
        _plot_single_class(
            df_class=df.copy(),
            class_name="",
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
