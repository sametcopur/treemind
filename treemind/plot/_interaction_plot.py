import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
from typing import Tuple, Union, Optional, List

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

    df = df.copy()

    cont_pairs, used = {}, set()
    cat_cols: List[str] = []
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
        df = _replace_infinity(df, lb1, "negative")
        df = _replace_infinity(df, ub1, "positive")

        x_left = df[lb1].to_numpy()
        x_right = df[ub1].to_numpy()

    else:
        cats1 = df[feat1].astype("category")
        codes1 = cats1.cat.codes.to_numpy()
        x_left, x_right = codes1, codes1 + 1
        cats1_labels = cats1.cat.categories.tolist()

    if type2 == "continuous":
        lb2, ub2 = cont_pairs[feat2]

        # Infinity değerlerini değiştir
        df = _replace_infinity(df, lb2, "negative")
        df = _replace_infinity(df, ub2, "positive")

        y_bottom = df[lb2].to_numpy()
        y_top = df[ub2].to_numpy()

    else:
        cats2 = df[feat2].astype("category")
        codes2 = cats2.cat.codes.to_numpy()
        y_bottom, y_top = codes2, codes2 + 1
        cats2_labels = cats2.cat.categories.tolist()

    # --------------------------- 3. Renkler
    values = df["value"].to_numpy()
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
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    # Color bar ayarları (orijinal kodla uyumlu)
    cbar.ax.set_ylim(vmin, vmax)
    cbar.ax.tick_params(labelsize=ticks_fontsize)
    cbar.set_label(color_bar_label or "Impact", fontsize=label_fontsizes)

    plt.tight_layout()
    plt.show()
